"""MIST realtime protocol 2.0.0: the binary frame codec and the control message schemas.

This module is the backend half of the contract with the mist-frontend client. The contract is
recorded in `tests/fixtures/realtime_protocol_v2/vectors.json`: `test_protocol.py` compares
encoded binary frames with its hex byte for byte, and control messages with its JSON as parsed
objects (key order and number formatting are not compared). The frontend is planned to copy the
file and assert the same parity, so a change here must be mirrored in the vectors file.

Binary frames carry audio. Each is a 16-byte little-endian header (`<IHHII`: magic, msg_type,
turn_tag, seq, sample_rate) followed by the payload; the payload length is the WebSocket message
length minus 16. Uplink frames (`mic_audio`, and `ref_audio`, the PC's own output that a desktop
sends as a reference) are PCM16 mono at 16 kHz with `turn_tag` 0. Downlink frames are PCM16 mono at 24 kHz, tagged with the turn's audio tag (1..65535, wrapping
past 0) so the client can flush every frame of a stopped turn by tag.

Control messages are JSON text frames discriminated by `type`. `parse_client_message` and
`parse_server_message` validate in strict mode, so a client sending `"active": "yes"` gets a
`ProtocolError` instead of a silent coercion; a JavaScript client must round integer fields
before sending. Fields a newer client adds are ignored rather than rejected, which keeps a minor
protocol bump compatible with this server; `alarm_report` is the one exception, and refuses an
unknown key.

Protocol 2.0.0 is main's 1.x control messages plus protocol 2's own. Every client message
main's `/ws` handles is modelled, with main's fields, except `audio` and `reset_vad`, which
protocol 2 removed: `text`, `interrupt`, `log_config` and `subscribe_logs` from protocol 1, and
capabilities' `action_decision`, `alarm_action` and `alarm_report` from 1.4.0. Main's server
messages are not modelled: the server messages here are the ones protocol 2 adds or changes, so
a server event protocol 2 leaves unchanged, such as `stream_token` or a capability event, is an
unknown type to `parse_server_message`.
"""

import json
import struct
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    SerializerFunctionWrapHandler,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic_core import PydanticCustomError

from .errors import ProtocolError

PROTOCOL_VERSION = "2.0.0"
CONTRACT = f"MIST realtime protocol {PROTOCOL_VERSION}"

# -- Binary frames -------------------------------------------------------------

MAGIC = 0x4D495354  # "MIST"
HEADER_FORMAT = "<IHHII"
_HEADER = struct.Struct(HEADER_FORMAT)
HEADER_SIZE = _HEADER.size

UPLINK_SAMPLE_RATE = 16000
DOWNLINK_SAMPLE_RATE = 24000
PCM16_SAMPLE_BYTES = 2
MAX_TURN_TAG = 0xFFFF
MAX_SEQ = 0xFFFFFFFF
MAX_REQ = 0xFFFFFFFF
# The wire bound on `wake.backlog_ms`. The addressee gate's minimum is a server setting, not a
# wire rule, so a smaller backlog parses and is then rejected `no_backlog`.
MAX_WAKE_BACKLOG_MS = 10000


class MsgType(IntEnum):
    """Binary frame message types.

    `REF_AUDIO` (0x11) is the PC's system output, sent by a desktop whose `voice_start` had
    `reference` true. Each frame covers the same 20 ms as the `MIC_AUDIO` frame with the same
    `seq`, so the server can tell whether a name it heard came from the PC's own playback.
    """

    AUDIO_CHUNK = 0x01
    AUDIO_COMPLETE = 0x02
    INTERRUPT_FADE = 0x03
    MIC_AUDIO = 0x10
    REF_AUDIO = 0x11


class Direction(StrEnum):
    """Which way a frame travels: `up` is client to server, `down` server to client."""

    UP = "up"
    DOWN = "down"


MSG_DIRECTION: dict[MsgType, Direction] = {
    MsgType.AUDIO_CHUNK: Direction.DOWN,
    MsgType.AUDIO_COMPLETE: Direction.DOWN,
    MsgType.INTERRUPT_FADE: Direction.DOWN,
    MsgType.MIC_AUDIO: Direction.UP,
    MsgType.REF_AUDIO: Direction.UP,
}

_SAMPLE_RATE: dict[Direction, int] = {
    Direction.UP: UPLINK_SAMPLE_RATE,
    Direction.DOWN: DOWNLINK_SAMPLE_RATE,
}

# Types that carry PCM16; the others are markers whose payload must be empty.
_PCM_TYPES = frozenset({MsgType.AUDIO_CHUNK, MsgType.MIC_AUDIO, MsgType.REF_AUDIO})


@dataclass(frozen=True, slots=True)
class Frame:
    """One decoded binary frame.

    Attributes:
        msg_type: The frame's message type.
        turn_tag: The turn's audio tag on downlink frames; 0 on uplink frames.
        seq: Per-stream sequence number. The receiver only counts gaps with it: it never
            reorders or cuts audio by `seq`. On the uplink the contract has the server rebase
            the count whenever `seq` does not rise: the first frame, or a `seq` not above the
            last one, sets a new base with no gap, and only a forward jump counts the missing
            frames. So a client may restart `seq` at 0 when it applies the answer to its newest
            `voice_start`, and the restart counts no gap.
        sample_rate: 16000 uplink, 24000 downlink.
        payload: PCM16 mono little-endian samples, or empty for the marker types.
    """

    msg_type: MsgType
    turn_tag: int
    seq: int
    sample_rate: int
    payload: bytes = b""

    @property
    def direction(self) -> Direction:
        """Which way this frame travels, derived from its message type."""
        return MSG_DIRECTION[_check_msg_type(self.msg_type)]


def _check_msg_type(value: int) -> MsgType:
    try:
        return MsgType(value)
    except ValueError:
        raise ProtocolError("msg_type", f"unknown message type 0x{value:02X}") from None


def _check_frame(msg_type: MsgType, turn_tag: int, sample_rate: int, payload: bytes) -> None:
    direction = MSG_DIRECTION[msg_type]
    expected_rate = _SAMPLE_RATE[direction]
    if sample_rate != expected_rate:
        raise ProtocolError(
            "sample_rate",
            f"{msg_type.name.lower()} frames are {expected_rate} Hz, got {sample_rate}",
        )
    if direction is Direction.UP and turn_tag != 0:
        raise ProtocolError("turn_tag", f"uplink frames carry turn_tag 0, got {turn_tag}")
    if direction is Direction.DOWN and not 1 <= turn_tag <= MAX_TURN_TAG:
        raise ProtocolError("turn_tag", f"downlink turn_tag must be 1..65535, got {turn_tag}")
    if msg_type not in _PCM_TYPES:
        if payload:
            raise ProtocolError(
                "payload", f"{msg_type.name.lower()} frames are empty, got {len(payload)} bytes"
            )
        return
    if len(payload) % PCM16_SAMPLE_BYTES:
        raise ProtocolError("payload", f"odd PCM16 byte count {len(payload)}")
    one_second = expected_rate * PCM16_SAMPLE_BYTES
    if len(payload) > one_second:
        raise ProtocolError(
            "payload", f"{len(payload)} bytes is over one second ({one_second} bytes)"
        )


def encode_frame(frame: Frame) -> bytes:
    """Pack a frame into its wire bytes.

    The same header rules as `decode_frame` apply, so the server cannot emit a frame the client
    would reject.

    Raises:
        ProtocolError: The frame breaks a header or payload rule; `field` names which.
    """
    msg_type = _check_msg_type(frame.msg_type)
    _check_frame(msg_type, frame.turn_tag, frame.sample_rate, frame.payload)
    if not 0 <= frame.seq <= MAX_SEQ:
        raise ProtocolError("seq", f"seq must fit in 32 bits, got {frame.seq}")
    header = _HEADER.pack(MAGIC, msg_type, frame.turn_tag, frame.seq, frame.sample_rate)
    return header + frame.payload


def decode_frame(data: bytes, *, direction: Direction | None = None) -> Frame:
    """Unpack and validate one binary WebSocket message.

    Args:
        data: The whole WebSocket message.
        direction: When given, a frame travelling the other way is rejected; the server passes
            `Direction.UP` so a client cannot inject downlink frames.

    Raises:
        ProtocolError: A bad magic, unknown type, wrong direction or sample rate, bad turn tag,
            odd PCM byte count or a payload over one second; `field` names which.
    """
    if len(data) < HEADER_SIZE:
        raise ProtocolError("header", f"frame is {len(data)} bytes, the header needs 16")
    magic, raw_type, turn_tag, seq, sample_rate = _HEADER.unpack_from(data)
    if magic != MAGIC:
        raise ProtocolError("magic", f"expected 0x{MAGIC:08X}, got 0x{magic:08X}")
    msg_type = _check_msg_type(raw_type)
    if direction is not None and MSG_DIRECTION[msg_type] is not direction:
        raise ProtocolError(
            "msg_type",
            f"{msg_type.name.lower()} travels {MSG_DIRECTION[msg_type]}, expected {direction}",
        )
    payload = bytes(data[HEADER_SIZE:])
    _check_frame(msg_type, turn_tag, sample_rate, payload)
    return Frame(
        msg_type=msg_type, turn_tag=turn_tag, seq=seq, sample_rate=sample_rate, payload=payload
    )


def next_turn_tag(tag: int) -> int:
    """Return the audio tag after `tag`: 1..65535, wrapping past 0 because 0 means untagged."""
    if not 1 <= tag <= MAX_TURN_TAG:
        raise ProtocolError("turn_tag", f"turn_tag must be 1..65535, got {tag}")
    return tag % MAX_TURN_TAG + 1


# -- Control messages: shared types --------------------------------------------

# MIST wakes by name on every endpoint, so `wake` is the one listening mode. It stays a `Literal`
# so `voice_start.mode` and `capabilities.voice.modes` keep a place for a later mode.
VoiceMode = Literal["wake"]
# Where MIST's names are spotted: `server` on the uplink the client streams, `endpoint` on the
# device, which then sends `wake`.
Spotter = Literal["server", "endpoint"]
# The same values as capabilities' power level. Only gaming mode sets it; nothing in the voice
# path detects games, full-screen windows or VRAM pressure.
PowerLevel = Literal["full", "low", "off"]
# `attend` is the provisional state sent at a name hit, before the addressee verdict.
BackendMistState = Literal["idle", "attend", "listen", "think", "speak"]
StreamChannel = Literal["tts", "text"]
TurnOrigin = Literal["voice", "text", "system"]
ErrorKind = Literal[
    "model",
    "tool",
    "retrieval",
    "timeout",
    "offline",
    "network",
    "server",
    "validation",
    "stt",
    "protocol",
]


class _WireModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", protected_namespaces=())


def _drop_none(data: dict[str, object], *names: str) -> dict[str, object]:
    for name in names:
        if data.get(name) is None:
            data.pop(name, None)
    return data


# -- Control messages: client to server ----------------------------------------


class VoiceStartMessage(_WireModel):
    """Open the session's voice path with the client's capture and playback parameters.

    While voice is active it reconfigures the running voice session instead. Each
    `voice_start` gets exactly one answer, a `voice_state` with `cause` `answer`.

    Attributes:
        req: Assigned by the client from 1 (its first `voice_start` on the connection),
            strictly increasing per connection, at most 4294967295. The answer carries it in
            `voice_state.req`. The client finds its answer by `cause` together with `req`, not
            by `req` alone: a broadcast and the answer to a `voice_start` whose `req` could not
            be read both carry `req` null, and only `cause` tells them apart. The schema checks
            only the range; the strictly-increasing check needs the connection's
            history, so it belongs to the session.
        endpoint: The client's view of its endpoint, which is advisory: the server binds each
            connection's endpoint when it accepts it, and logs and ignores a value that
            differs from the bound one.
        spotter: Who spots MIST's names and the stop keywords. With `server` the client
            streams while voice is on and unmuted, and the server spots both on the uplink; the
            desktop sends this. With `endpoint` the device spots both and sends `wake` and
            `barge_in`; the room unit sends this.
        reference: Whether the client sends `0x11` reference frames of the PC's output. Only
            `spotter` `server` may set it, because the server scores the reference at its own
            name hit; with `spotter` `endpoint` it fails validation naming `reference`, and the
            session answers `refused: "invalid"`. A missing `reference` reads as false.
        muted: The client's mute state, which the session applies before it accepts any uplink
            frame of this voice session, so a reconnect cannot unmute the client. A client that
            offers mute must send it. A missing `muted` reads as false, so frontend v1's C1,
            which has no mute, works unchanged.
    """

    type: Literal["voice_start"] = "voice_start"
    req: int = Field(ge=1, le=MAX_REQ)
    mode: VoiceMode
    sample_rate: Literal[16000]
    frame_ms: Literal[20]
    aec: Literal["browser", "native", "device", "headset", "none"]
    playback: bool
    endpoint: Literal["desktop", "room"]
    spotter: Spotter
    reference: bool = False
    muted: bool = False

    @field_validator("reference")
    @classmethod
    def _reference_needs_the_server_spotter(cls, reference: bool, info: ValidationInfo) -> bool:
        # `spotter` is declared before `reference`, so it is in `info.data` once it validated.
        if reference and info.data.get("spotter") == "endpoint":
            raise ValueError("reference frames are scored only with spotter server")
        return reference


class VoiceStopMessage(_WireModel):
    """Close the session's voice path."""

    type: Literal["voice_stop"] = "voice_stop"


class PlaybackProgressMessage(_WireModel):
    """How much of a turn's audio the client has played, sent while it plays."""

    type: Literal["playback_progress"] = "playback_progress"
    turn_id: str
    played_ms: int = Field(ge=0)


class PlaybackDoneMessage(_WireModel):
    """The client finished playing a turn's audio; completes a `tts`-channel turn."""

    type: Literal["playback_done"] = "playback_done"
    turn_id: str
    played_ms: int = Field(ge=0)


class BargeInMessage(_WireModel):
    """The endpoint's keyword spotter heard a barge-in keyword during a turn."""

    type: Literal["barge_in"] = "barge_in"
    turn_id: str
    keyword: str


class WakeMessage(_WireModel):
    """An endpoint heard one of MIST's names; `score` arbitrates against the other endpoints.

    `backlog_ms` is required, because every sender knows how much audio it buffered: the first
    `backlog_ms` of audio the endpoint sends after this message was captured before the wake, so
    the words before the name are kept. Below the server's `endpoint_min_backlog_ms` the wake is
    rejected `no_backlog`, so 0 parses but never opens a turn.
    """

    type: Literal["wake"] = "wake"
    keyword: str
    score: float
    backlog_ms: int = Field(ge=0, le=MAX_WAKE_BACKLOG_MS)


class GpuYieldMessage(_WireModel):
    """Frontend v1's request to yield the GPU or take it back, which the session ignores.

    It still parses because frontend v1's C1 sends it, and an unknown `type` would raise. The
    session logs it once per session at INFO and changes nothing, because only gaming mode sets
    the power level. A later major removes it once no client sends it.
    """

    type: Literal["gpu_yield"] = "gpu_yield"
    active: bool
    source: Literal["manual", "fullscreen"]


class MuteMessage(_WireModel):
    """Mute or unmute this session's microphone.

    Muting closes the session's open utterance with a `dropped` decision of reason `muted`;
    ingress then discards every uplink frame until a `mute` with `active` false. It carries no
    `req`: the session follows each one with a `voice_state` of cause `broadcast` and `req` null,
    sent to the muting session only, because `muted` is that session's own state. So the
    `answer` cause stays tied to `voice_start`.
    """

    type: Literal["mute"] = "mute"
    active: bool


class TextMessage(_WireModel):
    """Typed user input (unchanged from protocol 1)."""

    type: Literal["text"] = "text"
    text: str


class InterruptMessage(_WireModel):
    """Stop the current turn (unchanged from protocol 1)."""

    type: Literal["interrupt"] = "interrupt"


class LogConfigMessage(_WireModel):
    """Set a backend logger's level at runtime (unchanged from protocol 1)."""

    type: Literal["log_config"] = "log_config"
    action: Literal["set_level"]
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
    logger: str


class SubscribeLogsMessage(_WireModel):
    """Turn this connection's log stream on or off (unchanged from protocol 1)."""

    type: Literal["subscribe_logs"] = "subscribe_logs"
    enabled: bool
    levels: list[str] | None


# The three capability messages restate the fields of capabilities' inbound models on main
# (`backend/capabilities/events.py`, protocol 1.4.0) rather than import them, so this module still
# imports no backend package but its own. `test_protocol.py` compares each field's type,
# constraints and default with capabilities' model, so the two cannot drift apart.


class ActionDecisionMessage(_WireModel):
    """Approve or reject a capability's proposal (unchanged from protocol 1.4.0)."""

    type: Literal["action_decision"] = "action_decision"
    proposal_id: str
    decision: Literal["approve", "reject"]


class AlarmActionMessage(_WireModel):
    """Stop, snooze or dismiss an alarm (unchanged from protocol 1.4.0).

    `minutes` applies to a snooze only. `request_id` is an opaque client string the server's
    answer echoes; it is left out of the serialized message when none was sent, as capabilities'
    `to_payload` leaves it out, while a missing `minutes` serializes as null.
    """

    type: Literal["alarm_action"] = "alarm_action"
    alarm_id: str
    action: Literal["stop", "snooze", "dismiss"]
    minutes: int | None = Field(default=None, ge=1, le=60)
    request_id: str | None = Field(default=None, min_length=1, max_length=64)

    @model_serializer(mode="wrap")
    def _omit_absent(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        return _drop_none(handler(self), "request_id")


class AlarmReportMessage(_WireModel):
    """An endpoint's report of a ring it started playing, or of its local ring's outcome.

    Unchanged from protocol 1.4.0. It has no `endpoint` field, and unlike every other client
    message an unknown key is refused rather than ignored: the server attaches the endpoint of
    the connection the report arrived on, so a report that names its own endpoint fails
    validation naming `endpoint`. `snoozed_until` comes with outcome `snoozed` and with nothing
    else.

    Capabilities' model also refuses `at` or `rang_at` outside a window around the owner's
    clock. That check reads the clock, so it is not a wire rule and is not repeated here: on
    main the server validates the report with capabilities' model, and the capability host
    checks the window again against its own clock (`CapabilityHost.alarm_report`).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    type: Literal["alarm_report"] = "alarm_report"
    alarm_id: str = Field(min_length=1)
    outcome: Literal["rang", "stopped", "snoozed"]
    at: AwareDatetime
    rang_at: AwareDatetime
    snoozed_until: AwareDatetime | None = None
    request_id: str | None = Field(default=None, min_length=1, max_length=64)

    @model_validator(mode="after")
    def _snoozed_until_comes_with_snoozed(self) -> "AlarmReportMessage":
        if (self.snoozed_until is None) == (self.outcome == "snoozed"):
            raise PydanticCustomError(
                "snoozed_until_pairing",
                "{wire_field} is required with snoozed and refused otherwise",
                {"wire_field": "snoozed_until"},
            )
        return self

    @model_serializer(mode="wrap")
    def _omit_absent(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        return _drop_none(handler(self), "snoozed_until", "request_id")


ClientMessage = (
    VoiceStartMessage
    | VoiceStopMessage
    | PlaybackProgressMessage
    | PlaybackDoneMessage
    | BargeInMessage
    | WakeMessage
    | GpuYieldMessage
    | MuteMessage
    | TextMessage
    | InterruptMessage
    | LogConfigMessage
    | SubscribeLogsMessage
    | ActionDecisionMessage
    | AlarmActionMessage
    | AlarmReportMessage
)


# -- Control messages: server to client, nested parts --------------------------


class StreamFormat(_WireModel):
    """Sample rate and frame length of one audio direction."""

    sample_rate: int
    frame_ms: int


class PlaybackSettings(_WireModel):
    """Client playback parameters the server chose for the session."""

    client_prebuffer_ms: int
    fade_ms: int
    progress_interval_ms: int


class BargeInSettings(_WireModel):
    """The barge-in policy in force and the keywords that trigger it."""

    policy: Literal["keyword"]
    keywords: list[str]
    arm_delay_ms: int


class LatencyBudget(_WireModel):
    """A latency budget preset and its per-stage targets in milliseconds.

    The stage names are the Latency budget keys, which grow as stages are measured, so they are
    extra fields typed as integers rather than a fixed list.
    """

    model_config = ConfigDict(frozen=True, extra="allow", protected_namespaces=())
    __pydantic_extra__: dict[str, int] = Field(init=False)

    preset: str


class EngineStatus(_WireModel):
    """Where one speech engine runs and what it holds."""

    loaded: bool
    location: str = Field(pattern=r"^(cpu|cuda|cuda:\d+|off)$")
    model_id: str | None
    vram_mib: int | None


class EngineStatuses(_WireModel):
    """Status of the five speech engines; `kws` is the server's name spotter, on `cpu` or `off`."""

    vad: EngineStatus
    eot: EngineStatus
    stt: EngineStatus
    tts: EngineStatus
    kws: EngineStatus


class TtsVoice(_WireModel):
    """The voice the synthesizer speaks with: MIST's voice identity, voiced by one engine.

    The identity's clips and transcripts are private, so the message names the identity by its
    id and carries no reference audio, path or hash.

    Attributes:
        engine: The TTS engine id, for example `pocket-clone`.
        voice: The voice identity id (`VoiceIdentity.voice_id`, `PlacementConfig.tts_voice`),
            for example `mist-v1`.
        profile: The active voice profile, `mist` by default.
    """

    engine: str
    voice: str
    profile: str


class EnginePlacement(_WireModel):
    """Where the STT and TTS engines run for a turn or after a placement change."""

    stt: str
    tts: str


class TurnPromptStats(_WireModel):
    """Prompt size estimates and the cache reuse the LLM backend reported for a turn."""

    prefix_tokens_est: int
    history_tokens_est: int
    delta_tokens_est: int
    prompt_tokens: int | None
    cached_tokens: int | None
    prefix_changed: bool
    history_compacted: bool


class TurnLaneStats(_WireModel):
    """Time a turn spent waiting on the shared inference lanes."""

    stt_wait_ms: int
    tts_wait_ms: int
    tts_busy_after_cancel_ms: int


class VoiceCapabilities(_WireModel):
    """What the server's voice path supports; its presence marks a protocol 2 server.

    Attributes:
        modes: `wake` while at least one spotter is offered, else empty.
        spotters: `endpoint` once VAD, STT and the wake verifier are loaded, since the
            verifier accepts every wake, and `server` as well once `kws` is.
        reference: Whether the server scores `0x11` frames, which needs `kws` loaded.
    """

    protocol: Literal[2]
    modes: list[VoiceMode]
    spotters: list[Spotter]
    reference: bool
    uplink_sample_rate: Literal[16000]
    downlink_sample_rate: Literal[24000]
    barge_in: Literal["keyword"]
    tts: bool
    stt: bool
    eot: bool


class SessionCapabilities(_WireModel):
    """Capabilities announced in `session_started`."""

    tts_enabled: bool
    vad_enabled: bool
    voice: VoiceCapabilities


# -- Control messages: server to client ----------------------------------------


class VoiceStateMessage(_WireModel):
    """The session's voice configuration, engines, power level and mute.

    `cause` says why it was sent, and has two values:

    - `answer`: the reply to one `voice_start`, sent only to the session that sent it and
      exactly once per `voice_start`, a refused one and one that failed validation included.
      Answers go out in the order the `voice_start`s were received; an answer ready early waits
      for every earlier one.
    - `broadcast`: every other `voice_state`. Every broadcast is composed per session, with that
      session's own `muted`: one session gets one after each `mute` it sends, sent to it alone,
      and each session gets its own whenever the power level or an engine's location changes,
      whether or not it has started voice.

    A client takes the answer to its `voice_start` from `cause` and `req`, never from `active`
    or `mode`: a broadcast can carry the same mode, and matching by arrival order alone would
    shift every later answer if one went missing.

    Attributes:
        cause: `answer` or `broadcast`, as above.
        req: The answered `voice_start`'s `req`. Null for a broadcast, and in an answer only when
            that `voice_start`'s `req` could not be read. A broadcast with `req` set fails
            validation, which `parse_server_message` raises as `ProtocolError`.
        active: Whether voice is active in the session when the message is sent, in an answer
            and a broadcast alike. True with `power_level` `off` means voice takes capability
            commands only.
        mode: The session's active voice mode, null while voice is off; so a broadcast to a
            session that has not started voice carries null, and a broadcast sent while a
            reconfigure is pending carries the current mode, not the requested one.
        power_level: The process's power level when the message is sent.
        muted: This session's mute as the server applies it, so the client's indicator shows
            what the server does.
        refused: In an answer, null when the request was applied, and otherwise why it was
            not: a reason the requested mode is unavailable (`vad unavailable`;
            `kws unavailable` for the `server` spotter; `verify unavailable` for the
            `endpoint` spotter with no wake verifier loaded), `power_off` (the level is off and
            voice cannot take capability commands, so `active` is false), `superseded` (a newer
            valid `voice_start` arrived first), `voice_stop` (a `voice_stop` arrived first) or
            `invalid` (the `voice_start` failed validation). In a broadcast it is `power_off`,
            with `active` false, when the level went off and voice cannot take capability
            commands. Otherwise it names a mode the session lost to an engine change, for
            example when a `reload_failed` placement leaves an engine `off`; `active` stays true
            then, because a lost mode does not turn voice off.
    """

    type: Literal["voice_state"] = "voice_state"
    cause: Literal["answer", "broadcast"]
    req: int | None = Field(ge=1, le=MAX_REQ)
    active: bool
    mode: VoiceMode | None
    uplink: StreamFormat
    downlink: StreamFormat
    playback: PlaybackSettings
    barge_in: BargeInSettings
    budget: LatencyBudget
    engines: EngineStatuses
    tts_voice: TtsVoice
    vram_budget_mib: int
    power_level: PowerLevel
    muted: bool
    refused: str | None

    @field_validator("req")
    @classmethod
    def _broadcast_answers_no_request(cls, req: int | None, info: ValidationInfo) -> int | None:
        # `cause` is declared before `req`, so it is in `info.data` once it validated.
        if req is not None and info.data.get("cause") == "broadcast":
            raise ValueError("a broadcast answers no voice_start, so req must be null")
        return req


class TranscriptPartialMessage(_WireModel):
    """An interim transcript; the first `stable_chars` characters will not change."""

    type: Literal["transcript_partial"] = "transcript_partial"
    utterance_id: str
    text: str
    stable_chars: int = Field(ge=0)


class TranscriptionMessage(_WireModel):
    """The final transcript of an utterance; it closes the utterance it names.

    A closing message is the last message that names its utterance, and the client finalizes
    the utterance's partial on it. `text` is what the turn takes from the utterance: the whole
    text, or the text after its keyword.
    """

    type: Literal["transcription"] = "transcription"
    text: str
    utterance_id: str


class VadStatusMessage(_WireModel):
    """Speech start and end as the session's ingress detected them."""

    type: Literal["vad_status"] = "vad_status"
    status: Literal["speech_started", "speech_ended"]


class UtteranceDetail(_WireModel):
    """Detail of a `hold` or `release` decision."""

    utterance_id: str


class EndpointDetail(_WireModel):
    """Detail of an `endpoint` decision."""

    utterance_id: str
    eot_probability: float | None
    silence_ms: int


class SupersedeDetail(_WireModel):
    """Detail of a `supersede` decision.

    `merged` is the negation of `tool_dispatched`. With no tool dispatched, the superseded turn
    is not recorded on its own when the utterance that superseded it starts a turn: its request
    text leads that turn's. If that utterance starts none, the superseded turn is recorded alone.
    The merged text is not on the wire: it joins, with one space, the `transcription`s of the
    utterances whose words it carries.
    """

    merged: bool
    tool_dispatched: bool


class BargeInDetail(_WireModel):
    """Detail of a `barge_in` decision.

    `utterance_id` is set or null by what produced the verdict, not by its reason: it names the
    utterance when a transcript (partial or final) or a wake produced it, and is null
    when an endpoint `barge_in` report, the UI stop or `voice_stop` did.
    """

    keyword: str | None
    phase: str
    utterance_id: str | None


class BargeInIgnoredDetail(_WireModel):
    """Detail of a `barge_in_ignored` decision.

    `utterance_id` follows the same rule as in `BargeInDetail`: for example `not_armed` for an
    endpoint report that arrived before the arm delay carries null.
    """

    keyword: str | None
    text: str
    utterance_id: str | None


class DroppedDetail(_WireModel):
    """Detail of a `dropped` decision, with the STT scores that justified it."""

    utterance_id: str
    duration_ms: int
    text: str | None
    no_speech_prob: float | None
    avg_logprob: float | None


class DegradedDetail(_WireModel):
    """Detail of a `degraded` decision.

    `voiced_text` and `text_from_token` are both set for reason `tts_failed`, where the turn
    switches from speech to text mid-turn, and both null for every other reason. The detail
    cannot see the reason, so `VoiceDecisionDegraded` rejects a decision that breaks the rule,
    naming the offending key; `text_from_token` is also never negative.

    - `voiced_text` is the text of the audio synthesized before the failure and still sent
      after it, the most the client will play: "" for a text reply. It is an exact prefix of the
      concatenation of the turn's `stream_token` texts, so the client compares strings and does
      no offset arithmetic. A later `stream_cancelled` replaces it with its `spoken_text`.
    - `text_from_token` is the 0-based index, among the turn's `stream_token` messages, of the
      first one delivered as text after the switch. It equals the number of `stream_token`
      messages sent before the switch, so when none follows it is one past the last.

    The client renders the turn's text after `voiced_text` as unspoken: the part of the tokens
    sent before the switch that `voiced_text` does not cover, and every token from
    `text_from_token` on.
    """

    engine: str | None
    error: str | None
    voiced_text: str | None
    text_from_token: int | None = Field(ge=0)


class OverloadDetail(_WireModel):
    """Detail of an `overload` decision."""

    dropped_ms: int
    lane: str


class WakeLostDetail(_WireModel):
    """Detail of a `wake_lost` decision.

    `winner_score` is the accepted winner's raw spotter score and `score` the loser's; they may
    come from different spotter models, so they are reported, not compared. `utterance_id`
    names the utterance the lost wake opened, or a fresh id when an endpoint wake's backlog had
    not arrived to open one.
    """

    winner_score: float
    score: float
    utterance_id: str


class WakeRejectedDetail(_WireModel):
    """Detail of a `wake_rejected` decision: no utterance id and no text.

    The candidate was never published, so nothing about its speech may travel: only which
    spotter heard the name, its score, and the reference score, null when no `0x11` reference
    was scored.
    """

    source: Spotter
    score: float
    reference_score: float | None


class PlacementDetail(_WireModel):
    """Detail of a `placement` decision.

    `power_level` is the level the placement follows, `engines` where STT and TTS settled, and
    `released_mib` the VRAM a release freed, which a `power_off` placement carries.
    """

    power_level: PowerLevel
    engines: EnginePlacement
    released_mib: int | None


class VoiceDecisionEndpoint(_WireModel):
    """The utterance ended and a turn was opened from it."""

    type: Literal["voice_decision"] = "voice_decision"
    kind: Literal["endpoint"] = "endpoint"
    turn_id: str | None
    reason: Literal["semantic", "silence_cap", "silence_cap_no_model"]
    detail: EndpointDetail


class VoiceDecisionHold(_WireModel):
    """A turn's output is held while the user may still be speaking."""

    type: Literal["voice_decision"] = "voice_decision"
    kind: Literal["hold"] = "hold"
    turn_id: str | None
    reason: str
    detail: UtteranceDetail


class VoiceDecisionRelease(_WireModel):
    """A held turn's output is released to the client."""

    type: Literal["voice_decision"] = "voice_decision"
    kind: Literal["release"] = "release"
    turn_id: str | None
    reason: str
    detail: UtteranceDetail


class VoiceDecisionSupersede(_WireModel):
    """A turn was replaced by a newer utterance."""

    type: Literal["voice_decision"] = "voice_decision"
    kind: Literal["supersede"] = "supersede"
    turn_id: str | None
    reason: str
    detail: SupersedeDetail


class VoiceDecisionBargeIn(_WireModel):
    """The user interrupted a turn."""

    type: Literal["voice_decision"] = "voice_decision"
    kind: Literal["barge_in"] = "barge_in"
    turn_id: str | None
    reason: Literal["keyword", "endpoint_kws", "ui", "wake", "voice_stop"]
    detail: BargeInDetail


class VoiceDecisionBargeInIgnored(_WireModel):
    """Speech during a turn that did not interrupt it."""

    type: Literal["voice_decision"] = "voice_decision"
    kind: Literal["barge_in_ignored"] = "barge_in_ignored"
    turn_id: str | None
    reason: Literal["not_armed", "echo_guard", "no_keyword", "stale_turn", "idle"]
    detail: BargeInIgnoredDetail


class VoiceDecisionDropped(_WireModel):
    """An utterance was discarded without opening a turn; it closes the utterance it names.

    `noise`, `no_speech` and `stt_error` are ingress drops. `text_input`: a `text` message
    closed it. `keyword`: a keyword interrupt consumed it with nothing after the keyword.
    `ignored`: it ended while MIST spoke without interrupting. `voice_stop`: `voice_stop`
    closed it. `muted`: a `mute` closed it. `power_off`: the power level went off and voice
    cannot take capability commands. `detail.no_speech_prob` and `detail.avg_logprob` are null
    when no final transcript exists.
    """

    type: Literal["voice_decision"] = "voice_decision"
    kind: Literal["dropped"] = "dropped"
    turn_id: str | None
    reason: Literal[
        "noise",
        "no_speech",
        "stt_error",
        "text_input",
        "keyword",
        "voice_stop",
        "ignored",
        "muted",
        "power_off",
    ]
    detail: DroppedDetail


class VoiceDecisionDegraded(_WireModel):
    """Part of the voice path failed and the turn continued without it.

    For reason `tts_failed` both `detail.voiced_text` and `detail.text_from_token` must be set,
    and for every other reason both must be null; a decision that breaks this fails validation
    naming the first offending key, which `parse_server_message` raises as `ProtocolError`.
    """

    type: Literal["voice_decision"] = "voice_decision"
    kind: Literal["degraded"] = "degraded"
    turn_id: str | None
    reason: Literal[
        "tts_unavailable",
        "tts_engine_unsupported",
        "tts_failed",
        "turn_error",
        "record_failed",
        "vram_over_budget",
    ]
    detail: DegradedDetail

    @model_validator(mode="after")
    def _switch_keys_follow_the_reason(self) -> "VoiceDecisionDegraded":
        switched = self.reason == "tts_failed"
        for key in ("voiced_text", "text_from_token"):
            if (getattr(self.detail, key) is None) == switched:
                state = "set" if switched else "null"
                raise PydanticCustomError(
                    "degraded_switch_key",
                    "{wire_field} must be {state} for reason {reason}",
                    {"wire_field": f"detail.{key}", "state": state, "reason": self.reason},
                )
        return self


class VoiceDecisionOverload(_WireModel):
    """Ingress fell behind and dropped audio."""

    type: Literal["voice_decision"] = "voice_decision"
    kind: Literal["overload"] = "overload"
    turn_id: str | None
    reason: str
    detail: OverloadDetail


class VoiceDecisionWakeLost(_WireModel):
    """Another room unit won wake arbitration; it closes the utterance it names.

    Sent once to each losing session, with reason `arbitration`, when the winner's name is
    accepted. It names the utterance the lost wake opened, which closes with it. An endpoint
    wake whose backlog had not arrived opened none, so it names a fresh id the client never saw
    and can ignore. `turn_id` is the turn in flight when the name was heard, else null.
    """

    type: Literal["voice_decision"] = "voice_decision"
    kind: Literal["wake_lost"] = "wake_lost"
    turn_id: str | None
    reason: str
    detail: WakeLostDetail


class VoiceDecisionWakeRejected(_WireModel):
    """A name was heard but the addressee gate rejected it; the one trace such speech leaves.

    `turn_id` is the turn in flight when the name was heard, else null. `self_speech`: MIST said
    the name herself. `self_media`: the PC's own playback said it. `not_addressed`: the
    transcript does not address her. `no_backlog`: an endpoint `wake` carried less pre-wake
    audio than the server's minimum. A name that loses cross-session arbitration is not
    rejected: it gets `wake_lost`.
    """

    type: Literal["voice_decision"] = "voice_decision"
    kind: Literal["wake_rejected"] = "wake_rejected"
    turn_id: str | None
    reason: Literal["self_speech", "self_media", "not_addressed", "no_backlog"]
    detail: WakeRejectedDetail


class VoiceDecisionPlacement(_WireModel):
    """The speech engines moved; process-wide, so it belongs to no turn.

    `power_off`: a release finished, and
    `detail.released_mib` says what it freed. `release_timeout`: a job outlived the release
    timeout, so every slot is `off` and that job keeps its memory until it ends, when a
    `power_off` follows. `power_restored`: engines loaded again after the level rose, or a retry
    loaded a slot that had settled below its device before the release. `reload_failed`: a slot
    failed to load on the reload and fell down its device list; `detail.engines` says where each
    slot settled.
    """

    type: Literal["voice_decision"] = "voice_decision"
    kind: Literal["placement"] = "placement"
    turn_id: None = None
    reason: Literal["power_off", "power_restored", "reload_failed", "release_timeout"]
    detail: PlacementDetail


VoiceDecisionMessage = (
    VoiceDecisionEndpoint
    | VoiceDecisionHold
    | VoiceDecisionRelease
    | VoiceDecisionSupersede
    | VoiceDecisionBargeIn
    | VoiceDecisionBargeInIgnored
    | VoiceDecisionDropped
    | VoiceDecisionDegraded
    | VoiceDecisionOverload
    | VoiceDecisionWakeLost
    | VoiceDecisionWakeRejected
    | VoiceDecisionPlacement
)


class PlaybackStopMessage(_WireModel):
    """Fade out over `fade_ms` and flush every frame tagged `audio_tag`."""

    type: Literal["playback_stop"] = "playback_stop"
    turn_id: str
    audio_tag: int = Field(ge=1, le=MAX_TURN_TAG)
    fade_ms: int = Field(ge=0)
    reason: str


class PlaybackDuckMessage(_WireModel):
    """Lower playback by `db` decibels while the user may be speaking; 0 restores."""

    type: Literal["playback_duck"] = "playback_duck"
    turn_id: str
    db: float


class TurnMetricsMessage(_WireModel):
    """Timings, budget and outcome of one turn.

    `power_level` is the level at the turn's open. `timings` is keyed by stage timestamp, plus
    `caption_lag_ms` (the turn's P90 over its utterance's partials), and two keys set when a
    name or an endpoint wake opened the utterance:

    - `wake_accept_ms`: the hit frame's arrival to the addressee verdict.
    - `wake_screen_ms`: the name's end to the first `attend` or `listen` sent. The name's end is
      estimated as the hit frame's arrival less the manifest's `hit_lag_ms`, or for an endpoint
      wake as the `wake` message's arrival. It is the server side only: it uses a fixed median
      lag and leaves out capture to server and server to pixel.

    New keys keep the schema, which is any string to an int.
    """

    type: Literal["turn_metrics"] = "turn_metrics"
    turn_id: str
    origin: TurnOrigin
    channel: StreamChannel
    kind: Literal["plain", "tool"]
    outcome: Literal["completed", "interrupted", "superseded", "error"]
    recorded: bool
    power_level: PowerLevel
    engines: EnginePlacement
    timings: dict[str, int]
    budget: LatencyBudget
    over_budget: list[str]
    ledger_source: Literal["client", "estimate"]
    prompt: TurnPromptStats
    lanes: TurnLaneStats
    stale_events: int
    bulk_dropped: int


class StreamStartMessage(_WireModel):
    """A turn opened; `audio_tag` is present only when the turn will send audio."""

    type: Literal["stream_start"] = "stream_start"
    turn_id: str
    channel: StreamChannel
    origin: TurnOrigin
    audio_tag: int | None = Field(default=None, ge=1, le=MAX_TURN_TAG)

    @model_serializer(mode="wrap")
    def _omit_absent(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        return _drop_none(handler(self), "audio_tag")


class StreamCancelledMessage(_WireModel):
    """A turn stopped before completing; `spoken_text` is what the user heard of it.

    `spoken_text` is null exactly when the turn's `stream_start` had channel `text`. A
    `tts`-channel turn carries its spoken prefix, or "" when nothing played, a turn that
    `tts_failed` moved to text mid-turn included, since that turn keeps its `tts` channel.
    """

    type: Literal["stream_cancelled"] = "stream_cancelled"
    turn_id: str
    partial_text: str
    reason: Literal[
        "user_interrupt", "barge_in", "superseded", "new_input", "disconnect", "voice_stop"
    ]
    spoken_text: str | None


class StateCycleMessage(_WireModel):
    """MIST's visible state; `turn_id` is absent for `listen` when no turn is speaking.

    `attend` is provisional: with the `attend_cue` setting on, it is sent at a name hit before
    the addressee verdict, and is followed by `listen` on acceptance or by the state of the
    controller's current phase on rejection.
    """

    type: Literal["state_cycle"] = "state_cycle"
    state: BackendMistState
    turn_id: str | None = None

    @model_serializer(mode="wrap")
    def _omit_absent(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        return _drop_none(handler(self), "turn_id")


class ErrorMessage(_WireModel):
    """A discriminated error event; protocol 2 adds the `stt` and `protocol` kinds."""

    type: Literal["error"] = "error"
    kind: ErrorKind
    message: str
    retriable: bool
    context: dict[str, object] | None


class SessionStartedMessage(_WireModel):
    """The connection handshake; `capabilities.voice` tells the client it may speak protocol 2."""

    type: Literal["session_started"] = "session_started"
    session_id: str
    protocol_version: str
    mist_state: BackendMistState
    capabilities: SessionCapabilities


_ServerMessageExceptDecision = (
    VoiceStateMessage
    | TranscriptPartialMessage
    | TranscriptionMessage
    | VadStatusMessage
    | PlaybackStopMessage
    | PlaybackDuckMessage
    | TurnMetricsMessage
    | StreamStartMessage
    | StreamCancelledMessage
    | StateCycleMessage
    | ErrorMessage
    | SessionStartedMessage
)

ServerMessage = _ServerMessageExceptDecision | VoiceDecisionMessage


# -- Parsing and serialization -------------------------------------------------

# `voice_decision` is discriminated twice, by `type` and then by `kind`. Pydantic can nest the two
# only through `typing.Annotated`, which this repository does not import, so each level is its own
# envelope and `parse_server_message` routes between them.


class _ClientEnvelope(RootModel[ClientMessage]):
    root: ClientMessage = Field(discriminator="type")


class _ServerEnvelope(RootModel[_ServerMessageExceptDecision]):
    root: _ServerMessageExceptDecision = Field(discriminator="type")


class _VoiceDecisionEnvelope(RootModel[VoiceDecisionMessage]):
    root: VoiceDecisionMessage = Field(discriminator="kind")


def _type_tags(union: object, field: str) -> frozenset[str]:
    return frozenset(member.model_fields[field].default for member in union.__args__)


CLIENT_MESSAGE_TYPES = _type_tags(ClientMessage, "type")
# What a `room` session accepts: the text types the room device sends, and no other. A device
# token may wake MIST, speak a turn and answer an alarm, but not type a turn, read or gate the
# log stream, or decide a proposal. `tests/unit/firmware/test_core.py` ties the set to the
# types the firmware's contract driver sends, so a new device message fails there until it is
# allowed here.
ROOM_CLIENT_MESSAGE_TYPES = frozenset(
    {
        "voice_start",
        "wake",
        "barge_in",
        "mute",
        "interrupt",
        "playback_progress",
        "playback_done",
        "alarm_action",
        "alarm_report",
    }
)
SERVER_MESSAGE_TYPES = _type_tags(ServerMessage, "type")
# What a `room` session is sent: the text types the room device's codec decodes, and no other.
# Each is a protocol message or a capability event (`backend/capabilities/events.py`). The
# proposal a room's own turn makes, every `action_resolved`, the capabilities and system status
# and the heartbeat are withheld, as the device would ignore them.
# `tests/unit/firmware/test_core.py` ties the set to `read_inbound`'s source, so a type the
# firmware starts to decode fails there until it is allowed here.
ROOM_SERVER_MESSAGE_TYPES = frozenset(
    {
        "session_started",
        "voice_state",
        "state_cycle",
        "voice_decision",
        "stream_start",
        "playback_stop",
        "playback_duck",
        "error",
        "alarm_state",
        "alarm_fired",
        "alarm_action_result",
    }
)
VOICE_DECISION_KINDS = _type_tags(VoiceDecisionMessage, "kind")


def _protocol_error(exc: ValidationError) -> ProtocolError:
    """Name the first failing field, dropping the union tags pydantic puts in the location.

    A model validator that checks fields against each other fails at the model's own location,
    so it names the field in its error's `wire_field` context, relative to that location.
    """
    error = exc.errors(include_url=False)[0]
    ctx = error.get("ctx", {})
    if error["type"] in ("union_tag_invalid", "union_tag_not_found"):
        discriminator = str(ctx.get("discriminator", "type")).strip("'")
        return ProtocolError(discriminator, error["msg"])
    parts = [str(part) for part in error["loc"][1:]]
    if "wire_field" in ctx:
        parts.append(str(ctx["wire_field"]))
    field = ".".join(parts) if parts else "message"
    return ProtocolError(field, error["msg"])


def _validate(
    envelope: type[_ClientEnvelope] | type[_ServerEnvelope] | type[_VoiceDecisionEnvelope],
    raw: str | bytes,
) -> ClientMessage | ServerMessage:
    try:
        return envelope.model_validate_json(raw, strict=True).root
    except ValidationError as exc:
        raise _protocol_error(exc) from exc


def parse_client_message(raw: str | bytes) -> ClientMessage:
    """Validate one client-to-server JSON text frame.

    The removed protocol 1 messages `audio` and `reset_vad` are unknown types here.

    Raises:
        ProtocolError: Invalid JSON, an unknown or missing `type`, a missing required field, a
            field of the wrong type or value, or an unknown key on `alarm_report`; `field`
            names it (dotted for nested fields).
    """
    return _validate(_ClientEnvelope, raw)


@dataclass(frozen=True, slots=True)
class VoiceStartPeek:
    """The one field read from text that may be a `voice_start`, valid or not.

    Attributes:
        req: The message's `req` when it is a JSON integer from 1 to 4294967295, else None.
    """

    req: int | None


def peek_voice_start(raw: str | bytes) -> VoiceStartPeek | None:
    """Read the `req` of a `voice_start` without validating the rest of it.

    The session answers every `voice_start`, one that fails validation included, with a
    `voice_state` carrying its `req`. `parse_client_message` raises on such a message without
    returning any field, so the session reads the one field the answer needs here. The codec
    keeps no state: the strictly-increasing check on `req` is the session's.

    Returns:
        None unless `raw` is a JSON object whose `type` is `voice_start`. Otherwise a peek whose
        `req` is the message's `req` when that is a JSON integer from 1 to 4294967295, and None
        when it is missing, out of range or of another JSON type (`true` included). Text nested
        past the JSON decoder's recursion limit counts as unreadable and returns None, even when
        its `type` is `voice_start`.
    """
    try:
        data = json.loads(raw)
    except (ValueError, RecursionError):
        # RecursionError: the stdlib decoder recurses once per nesting level, so deeply nested
        # client text would otherwise escape as an exception the caller does not expect.
        return None
    if not isinstance(data, dict) or data.get("type") != "voice_start":
        return None
    req = data.get("req")
    # JSON `true` decodes to a bool, which is an int subclass, so it is excluded by name.
    if isinstance(req, int) and not isinstance(req, bool) and 1 <= req <= MAX_REQ:
        return VoiceStartPeek(req=req)
    return VoiceStartPeek(req=None)


def parse_server_message(raw: str | bytes) -> ServerMessage:
    """Validate one server-to-client JSON text frame defined by this contract.

    Covers the messages protocol 2 adds or changes. Server events protocol 2 leaves unchanged,
    such as `stream_token`, are not modelled here and raise as unknown types.

    Raises:
        ProtocolError: As for `parse_client_message`; an unknown `voice_decision` kind names
            `kind`.
    """
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise ProtocolError("message", f"not valid JSON: {exc}") from exc
    is_decision = isinstance(data, dict) and data.get("type") == "voice_decision"
    return _validate(_VoiceDecisionEnvelope if is_decision else _ServerEnvelope, raw)


def encode_message(message: ClientMessage | ServerMessage) -> str:
    """Serialize a control message to the JSON text frame the vectors file records."""
    return message.model_dump_json()
