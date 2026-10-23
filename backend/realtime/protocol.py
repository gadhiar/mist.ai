"""MIST realtime protocol 2.0.0: the binary frame codec and the control message schemas.

This module is the backend half of the contract with the mist-frontend client. The contract is
recorded in `tests/fixtures/realtime_protocol_v2/vectors.json`: `test_protocol.py` compares
encoded binary frames with its hex byte for byte, and control messages with its JSON as parsed
objects (key order and number formatting are not compared). The frontend is planned to copy the
file and assert the same parity, so a change here must be mirrored in the vectors file.

Binary frames carry audio. Each is a 16-byte little-endian header (`<IHHII`: magic, msg_type,
turn_tag, seq, sample_rate) followed by the payload; the payload length is the WebSocket message
length minus 16. Uplink frames (`mic_audio`) are PCM16 mono at 16 kHz with `turn_tag` 0.
Downlink frames are PCM16 mono at 24 kHz, tagged with the turn's audio tag (1..65535, wrapping
past 0) so the client can flush every frame of a stopped turn by tag.

Control messages are JSON text frames discriminated by `type`. `parse_client_message` and
`parse_server_message` validate in strict mode, so a client sending `"pressed": "yes"` gets a
`ProtocolError` instead of a silent coercion; a JavaScript client must round integer fields
before sending. Fields a newer client adds are ignored rather than rejected, which keeps a minor
protocol bump compatible with this server.
"""

import json
import struct
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    SerializerFunctionWrapHandler,
    ValidationError,
    model_serializer,
)

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


class MsgType(IntEnum):
    """Binary frame message types."""

    AUDIO_CHUNK = 0x01
    AUDIO_COMPLETE = 0x02
    INTERRUPT_FADE = 0x03
    MIC_AUDIO = 0x10


class Direction(StrEnum):
    """Which way a frame travels: `up` is client to server, `down` server to client."""

    UP = "up"
    DOWN = "down"


MSG_DIRECTION: dict[MsgType, Direction] = {
    MsgType.AUDIO_CHUNK: Direction.DOWN,
    MsgType.AUDIO_COMPLETE: Direction.DOWN,
    MsgType.INTERRUPT_FADE: Direction.DOWN,
    MsgType.MIC_AUDIO: Direction.UP,
}

_SAMPLE_RATE: dict[Direction, int] = {
    Direction.UP: UPLINK_SAMPLE_RATE,
    Direction.DOWN: DOWNLINK_SAMPLE_RATE,
}

# Types that carry PCM16; the others are markers whose payload must be empty.
_PCM_TYPES = frozenset({MsgType.AUDIO_CHUNK, MsgType.MIC_AUDIO})


@dataclass(frozen=True, slots=True)
class Frame:
    """One decoded binary frame.

    Attributes:
        msg_type: The frame's message type.
        turn_tag: The turn's audio tag on downlink frames; 0 on uplink frames.
        seq: Per-stream sequence number; the receiver counts gaps rather than reordering.
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

VoiceMode = Literal["open", "ptt", "wake"]
GpuMode = Literal["primary", "yield"]
BackendMistState = Literal["idle", "listen", "think", "speak"]
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
    """Open the session's voice path with the client's capture and playback parameters."""

    type: Literal["voice_start"] = "voice_start"
    mode: VoiceMode
    sample_rate: Literal[16000]
    frame_ms: Literal[20]
    aec: Literal["browser", "native", "device", "headset", "none"]
    playback: bool
    endpoint: Literal["desktop", "room"]
    local_kws: bool


class VoiceStopMessage(_WireModel):
    """Close the session's voice path."""

    type: Literal["voice_stop"] = "voice_stop"


class PttMessage(_WireModel):
    """Push-to-talk key state; a release ends the utterance."""

    type: Literal["ptt"] = "ptt"
    pressed: bool


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
    """A room unit heard its wake word; `score` arbitrates between units that heard it."""

    type: Literal["wake"] = "wake"
    keyword: str
    score: float


class GpuYieldMessage(_WireModel):
    """Ask the process to yield the GPU, or to take it back; process-wide."""

    type: Literal["gpu_yield"] = "gpu_yield"
    active: bool
    source: Literal["manual", "fullscreen"]


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


ClientMessage = (
    VoiceStartMessage
    | VoiceStopMessage
    | PttMessage
    | PlaybackProgressMessage
    | PlaybackDoneMessage
    | BargeInMessage
    | WakeMessage
    | GpuYieldMessage
    | TextMessage
    | InterruptMessage
    | LogConfigMessage
    | SubscribeLogsMessage
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
    location: str = Field(pattern=r"^(cpu|cuda|cuda:\d+|remote|off)$")
    model_id: str | None
    vram_mib: int | None


class EngineStatuses(_WireModel):
    """Status of the four speech engines."""

    vad: EngineStatus
    eot: EngineStatus
    stt: EngineStatus
    tts: EngineStatus


class TtsVoice(_WireModel):
    """The synthesizer variant and the reference voice it speaks with."""

    variant: Literal["turbo", "original"]
    profile: str | None
    source: Literal["profile", "builtin"]
    seconds: float | None
    sha256_prefix: str | None


class GpuState(_WireModel):
    """The process GPU mode and the triggers holding it in yield."""

    mode: GpuMode
    triggers: list[str]


class SpeechServiceState(_WireModel):
    """The remote speech service connection, when one is configured."""

    configured: bool
    connected: bool
    rtt_ms: int | None
    location_label: str | None
    model_mismatch: bool


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
    remote_rtt_ms: int | None


class VoiceCapabilities(_WireModel):
    """What the server's voice path supports; its presence marks a protocol 2 server."""

    protocol: Literal[2]
    modes: list[VoiceMode]
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
    """The session's voice configuration, engines and GPU placement.

    Sent on `voice_start` and to every session when the GPU mode changes, so `mode` is null for
    a session that has not started voice.
    """

    type: Literal["voice_state"] = "voice_state"
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
    gpu: GpuState
    speech_service: SpeechServiceState
    refused: str | None


class TranscriptPartialMessage(_WireModel):
    """An interim transcript; the first `stable_chars` characters will not change."""

    type: Literal["transcript_partial"] = "transcript_partial"
    utterance_id: str
    text: str
    stable_chars: int = Field(ge=0)


class TranscriptionMessage(_WireModel):
    """The final transcript of an utterance."""

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
    """Detail of a `supersede` decision."""

    merged: bool
    tool_dispatched: bool


class BargeInDetail(_WireModel):
    """Detail of a `barge_in` decision."""

    keyword: str | None
    phase: str


class BargeInIgnoredDetail(_WireModel):
    """Detail of a `barge_in_ignored` decision."""

    keyword: str | None
    text: str


class DroppedDetail(_WireModel):
    """Detail of a `dropped` decision, with the STT scores that justified it."""

    utterance_id: str
    duration_ms: int
    text: str | None
    no_speech_prob: float | None
    avg_logprob: float | None


class DegradedDetail(_WireModel):
    """Detail of a `degraded` decision."""

    engine: str | None
    error: str | None


class OverloadDetail(_WireModel):
    """Detail of an `overload` decision."""

    dropped_ms: int
    lane: str


class WakeLostDetail(_WireModel):
    """Detail of a `wake_lost` decision."""

    winner_score: float
    score: float


class PlacementDetail(_WireModel):
    """Detail of a `placement` decision."""

    mode: GpuMode
    engines: EnginePlacement
    released_mib: int | None


class VoiceDecisionEndpoint(_WireModel):
    """The utterance ended and a turn was opened from it."""

    type: Literal["voice_decision"] = "voice_decision"
    kind: Literal["endpoint"] = "endpoint"
    turn_id: str | None
    reason: Literal["semantic", "silence_cap", "silence_cap_no_model", "ptt_release"]
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
    reason: Literal["keyword", "endpoint_kws", "ptt", "ui", "wake", "voice_stop"]
    detail: BargeInDetail


class VoiceDecisionBargeInIgnored(_WireModel):
    """Speech during a turn that did not interrupt it."""

    type: Literal["voice_decision"] = "voice_decision"
    kind: Literal["barge_in_ignored"] = "barge_in_ignored"
    turn_id: str | None
    reason: Literal["not_armed", "echo_guard", "no_keyword", "stale_turn", "idle"]
    detail: BargeInIgnoredDetail


class VoiceDecisionDropped(_WireModel):
    """An utterance was discarded without opening a turn."""

    type: Literal["voice_decision"] = "voice_decision"
    kind: Literal["dropped"] = "dropped"
    turn_id: str | None
    reason: Literal["noise", "no_speech", "stt_error", "text_input"]
    detail: DroppedDetail


class VoiceDecisionDegraded(_WireModel):
    """Part of the voice path failed and the turn continued without it."""

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


class VoiceDecisionOverload(_WireModel):
    """Ingress fell behind and dropped audio."""

    type: Literal["voice_decision"] = "voice_decision"
    kind: Literal["overload"] = "overload"
    turn_id: str | None
    reason: str
    detail: OverloadDetail


class VoiceDecisionWakeLost(_WireModel):
    """Another room unit won wake arbitration."""

    type: Literal["voice_decision"] = "voice_decision"
    kind: Literal["wake_lost"] = "wake_lost"
    turn_id: str | None
    reason: str
    detail: WakeLostDetail


class VoiceDecisionPlacement(_WireModel):
    """The speech engines moved; process-wide, so it belongs to no turn."""

    type: Literal["voice_decision"] = "voice_decision"
    kind: Literal["placement"] = "placement"
    turn_id: None = None
    reason: Literal[
        "manual",
        "fullscreen",
        "pressure",
        "resume",
        "switch_failed",
        "remote_lost",
        "remote_restored",
    ]
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
    """Timings, budget and outcome of one turn, judged against its placement mode's budget."""

    type: Literal["turn_metrics"] = "turn_metrics"
    turn_id: str
    origin: TurnOrigin
    channel: StreamChannel
    kind: Literal["plain", "tool"]
    outcome: Literal["completed", "interrupted", "superseded", "error"]
    recorded: bool
    placement_mode: GpuMode
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
    """A turn stopped before completing; `spoken_text` is what the user heard of it."""

    type: Literal["stream_cancelled"] = "stream_cancelled"
    turn_id: str
    partial_text: str
    reason: Literal[
        "user_interrupt", "barge_in", "superseded", "new_input", "disconnect", "voice_stop"
    ]
    spoken_text: str | None


class StateCycleMessage(_WireModel):
    """MIST's visible state; `turn_id` is absent for `listen` when no turn is speaking."""

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
SERVER_MESSAGE_TYPES = _type_tags(ServerMessage, "type")
VOICE_DECISION_KINDS = _type_tags(VoiceDecisionMessage, "kind")


def _protocol_error(exc: ValidationError) -> ProtocolError:
    """Name the first failing field, dropping the union tags pydantic puts in the location."""
    error = exc.errors(include_url=False)[0]
    if error["type"] in ("union_tag_invalid", "union_tag_not_found"):
        discriminator = str(error.get("ctx", {}).get("discriminator", "type")).strip("'")
        return ProtocolError(discriminator, error["msg"])
    path = error["loc"][1:]
    field = ".".join(str(part) for part in path) if path else "message"
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
        ProtocolError: Invalid JSON, an unknown or missing `type`, a missing required field or a
            field of the wrong type or value; `field` names it (dotted for nested fields).
    """
    return _validate(_ClientEnvelope, raw)


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
