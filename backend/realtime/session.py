"""One realtime session per WebSocket: uplink routing, voice settings, and the outbound lanes.

The session owns the socket and the voice configuration; the `TurnController` owns turns and is
the only caller of ingress levers. The session dispatches each client message to the mailbox,
routes uplink frames to the current `AudioIngress`, answers every `voice_start` in the order
received, and drains three outbound lanes through one sender:

- `control`: unbounded; voice decisions, `voice_state`, errors. Sent first, except an item
  fenced behind the `turn` lane (below).
- `turn`: a `TurnLane` bounded at 2048 items; a turn's stream and frames, purged by turn id. A
  full lane means the client stopped reading, so the session closes with 1013.
- `bulk`: logs and broadcasts, bounded at 512 items, dropping the oldest and counting the drops.

The sender takes one item at a time, so a purge between two takes removes everything the client
has not been sent yet.

A session is bound to an endpoint of a kind: `desktop`, the loopback `/ws`, or `room`, a device
the listener authenticated. A kind not given is `desktop` for the `desktop` endpoint and `room`
for any other, so it fails closed. A `room` session accepts only the text types the device sends
(`ROOM_CLIENT_MESSAGE_TYPES`) and `0x10` frames: any other type is refused after parsing and
before dispatch, with a throttled protocol error naming `type`, a `room_message_refused` WARNING
once per type, and a count in `refused_messages`; it reaches no turn, no log subscription, no log
gate and no proposal. Every `0x11` frame is dropped and counted in `dropped_references`.

Going out, a `room` session's socket carries binary frames and only the text types its device
acts on (`ROOM_SERVER_MESSAGE_TYPES`). The sender checks each text item's `type` as it writes
it, after the lanes, so broadcasts, targeted sends and the session's own items are all held to
the set, while lane order, bounds and fences stay as they are: a withheld `turn` item still moves
its lane on. Every other item, one with no readable `type` included, is withheld, counted in
`withheld`, and logged `room_message_withheld` at DEBUG once per type. Another session's
proposals, the capabilities and system status, the heartbeat, and a turn's tokens, transcripts
and metrics never reach the device.

A `control` item may carry a fence, a `TurnLane` mark, and waits until the lane has passed it.
The newest waiting fence is the largest fence on `control` the lane has not passed yet. Three
rules set a fence (A53):

1. A `voice_state` answer that `apply` readies is fenced at the larger of the lane's mark and
   the newest waiting fence. An answer applied at a turn's end therefore follows that turn's
   `state_cycle idle` and `turn_metrics`, which the controller queued on the lane before its
   mailbox step reached the reconfigure. Every other answer is sent at a message's receipt,
   never at a turn's end, and sets no fence of its own.
2. A `voice_state`, answer or broadcast, queued while a fenced item waits takes the newest
   waiting fence, so answers keep `req` order and no broadcast overtakes an older answer.
3. The `voice_stop` window: when a `voice_stop` is received, or the power suspends voice, while
   a fenced item waits, every `control` item queued until the newest waiting fence passes
   takes that fence. The stop's refusals, `barge_in`, `state_cycle idle` and `dropped` then
   follow the waiting answer, in M63's order.

The sender sends the first `control` item whose fence is None or passed, else the `turn` lane's
next item, else `bulk`'s. Outside the `voice_stop` window no item but a `voice_state` waits
for an answer.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
import uuid
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel

from .addressee import AddresseeGate
from .bargein import KeywordBargeIn
from .budget import BUDGET_FIELDS, VoiceSettings
from .errors import ProtocolError
from .ingress import (
    AudioIngress,
    EndpointCandidate,
    IngressEngines,
    IngressLanes,
    IngressSeed,
    UtteranceDropped,
    UtteranceEnded,
    ring_retention_ms,
)
from .ingress import Lane as InferenceLane
from .playback import LaneFull, LanePurge, MonotonicPlaybackClock, PlaybackClock, TurnLane
from .power import PowerGate, PowerGatedResponder, SynthesizerSlot, TranscriberSlot
from .protocol import (
    ROOM_CLIENT_MESSAGE_TYPES,
    ROOM_SERVER_MESSAGE_TYPES,
    ActionDecisionMessage,
    AlarmActionMessage,
    AlarmReportMessage,
    BargeInMessage,
    BargeInSettings,
    DegradedDetail,
    Direction,
    EngineStatus,
    EngineStatuses,
    ErrorMessage,
    Frame,
    GpuYieldMessage,
    InterruptMessage,
    LatencyBudget,
    LogConfigMessage,
    MsgType,
    MuteMessage,
    PlaybackDoneMessage,
    PlaybackProgressMessage,
    PlaybackSettings,
    PowerLevel,
    StreamFormat,
    StreamStartMessage,
    SubscribeLogsMessage,
    TextMessage,
    TtsVoice,
    VoiceDecisionDegraded,
    VoiceStartMessage,
    VoiceStateMessage,
    VoiceStopMessage,
    WakeMessage,
    decode_frame,
    encode_frame,
    parse_client_message,
    peek_voice_start,
)
from .speech import (
    NameSpotter,
    SpeechSynthesizer,
    StreamingTranscriber,
    TurnEndDetector,
    VoiceActivityDetector,
    WakeVerifier,
)
from .turn import (
    BargeInReport,
    Interrupt,
    Mute,
    PlaybackDone,
    PlaybackProgress,
    Reconfigure,
    Text,
    TurnController,
    TurnPlacement,
    TurnResponder,
    VoiceStart,
    VoiceStop,
    Wake,
    WakeArbitration,
)

logger = logging.getLogger(__name__)

# A session's endpoint: `desktop`, or a room device's registered name (`backend/devices/`), so
# a ring or a broadcast can name one device. What kind of endpoint it is lives in `EndpointKind`.
Endpoint = str
EndpointKind = Literal["desktop", "room"]
CapabilityMessage = ActionDecisionMessage | AlarmActionMessage | AlarmReportMessage

TURN_LANE_MAX_ITEMS = 2048
BULK_LANE_MAX_ITEMS = 512
PROTOCOL_ERROR_INTERVAL_MS = 1000
UPLINK_FORMAT = StreamFormat(sample_rate=16000, frame_ms=20)
DOWNLINK_SAMPLE_RATE = 24000
CLOSE_TRY_AGAIN_LATER = 1013
CLOSE_INTERNAL_ERROR = 1011

OFF = EngineStatus(loaded=False, location="off", model_id=None, vram_mib=None)
ALL_OFF = EngineStatuses(vad=OFF, eot=OFF, stt=OFF, tts=OFF, kws=OFF)


@dataclass(slots=True)
class SpeechEngines:
    """The loaded speech engines, shared by every session; a slot is None when it is off.

    Mutable on purpose: T22's GPU release and reload swap engines and statuses in place, and
    every session reads them at the next `voice_start` and `voice_state`. `stt_slot` and
    `tts_slot` are the slots the runtime swaps; a session opens its streams and synthesizes on
    them when they are set, so work started before a swap keeps its engine, and `transcriber`
    and `synthesizer` name the engine loaded now.

    `verifier` is the acoustic wake verifier. With none, no wake can be accepted, so the runtime
    loads no `spotter` and an endpoint `voice_start` is refused; it has no status of its own in
    `statuses`.
    """

    vad: VoiceActivityDetector | None = None
    turn_end: TurnEndDetector | None = None
    transcriber: StreamingTranscriber | None = None
    synthesizer: SpeechSynthesizer | None = None
    spotter: NameSpotter | None = None
    statuses: EngineStatuses = ALL_OFF
    vram_over_budget: bool = False
    stt_slot: TranscriberSlot | None = None
    tts_slot: SynthesizerSlot | None = None
    verifier: WakeVerifier | None = None


@dataclass(frozen=True, slots=True)
class InferenceLanes:
    """The process-wide lanes each engine's calls run on, one per engine kind.

    `verify` is the wake verifier's own lane, so a verify job never queues behind the spotter's
    hops or an end-of-turn check.
    """

    vad: InferenceLane
    eot: InferenceLane
    stt: InferenceLane
    tts: InferenceLane
    kws: InferenceLane
    verify: InferenceLane


def _full() -> PowerLevel:
    return "full"


def _utterance_ids() -> Callable[[], str]:
    """One id source per session, shared by its ingresses and its controller (v4)."""
    return lambda: f"utt-{uuid.uuid4().hex[:16]}"


def _ignore_capability(session_id: str, message: CapabilityMessage) -> None:
    logger.debug("session %s: capability message %s has no handler", session_id, message.type)


def _no_alarm() -> bool:
    return False


def _on_cpu(status: EngineStatus) -> bool:
    return status.loaded and status.location == "cpu"


@dataclass(frozen=True, slots=True)
class SessionServices:
    """What every session shares, built once by `MistRuntime`; tests pass fakes.

    `ingress_factory` and `controller_factory` default to the real classes; a test passes its
    own to record what the session builds. `on_capability_message` receives `action_decision`,
    `alarm_action` and `alarm_report`, which belong to the capability host (T13).

    `power_gate` gates turns on the model tier and voice on spoken commands at off (T22); with
    None every turn reaches `responder` and the level is `power_level()`, which is otherwise
    unread. `alarm_ringing` is passed to each controller, and T13 sets it.
    """

    responder: TurnResponder
    engines: SpeechEngines
    lanes: InferenceLanes
    settings: VoiceSettings
    placement: Callable[[], TurnPlacement]
    power_level: Callable[[], PowerLevel] = _full
    tts_enabled: bool = True
    clock_factory: Callable[[], PlaybackClock] = MonotonicPlaybackClock
    utterance_ids: Callable[[], Callable[[], str]] = _utterance_ids
    ingress_factory: Callable[..., AudioIngress] = AudioIngress
    controller_factory: Callable[..., TurnController] = TurnController
    on_capability_message: Callable[[str, CapabilityMessage], None] = _ignore_capability
    power_gate: PowerGate | None = None
    alarm_ringing: Callable[[], bool] = _no_alarm


class SessionSocket(Protocol):
    """The Starlette `WebSocket` surface the session uses."""

    async def receive(self) -> Mapping[str, object]:
        """The next ASGI message: `websocket.receive` with `text` or `bytes`, or a disconnect."""
        ...

    async def send_text(self, data: str) -> None:
        """Send one text frame."""
        ...

    async def send_bytes(self, data: bytes) -> None:
        """Send one binary frame."""
        ...

    async def close(self, code: int = 1000) -> None:
        """Close the connection with `code`."""
        ...


class SessionHost(Protocol):
    """The registry's side of a session."""

    arbiter: WakeArbitration

    def recent_tts_text(self, window_ms: int) -> str:
        """Every session's text played within `window_ms`, for the addressee gate's echo guard."""
        ...

    def session_closed(self, session: RealtimeSession) -> None:
        """The session has committed its turn and stopped; forget it."""
        ...

    def voice_activity_changed(self, session: RealtimeSession) -> None:
        """The session's `voice_activity` may have changed."""
        ...

    def set_log_subscription(
        self, session: RealtimeSession, levels: list[str] | None, enabled: bool
    ) -> None:
        """The session's `subscribe_logs`: stream log lines at `levels` (None: all) or stop."""
        ...

    def set_logger_level(self, logger_name: str, level: str) -> None:
        """The session's `log_config`: gate one logger at `level` for every subscriber."""
        ...


@dataclass(slots=True)
class _Answer:
    """One `voice_start`'s answer slot, sent once every earlier slot has been sent."""

    req: int | None
    request: Reconfigure | None = None
    ready: bool = False
    refused: str | None = None


@dataclass(slots=True)
class _Voice:
    """The voice settings and ingress the session routes uplink to."""

    settings: VoiceStartMessage
    ingress: AudioIngress
    epoch: int


def wire_budget(settings: VoiceSettings) -> LatencyBudget:
    budget = settings.budget
    return LatencyBudget(preset=budget.preset, **{n: getattr(budget, n) for n in BUDGET_FIELDS})


def tts_voice_of(engines: SpeechEngines, settings: VoiceSettings) -> TtsVoice:
    """The synthesizer's voice, or with TTS off the configured identity on engine `off`."""
    synthesizer = engines.synthesizer
    if synthesizer is not None:
        voice = synthesizer.voice
        return TtsVoice(engine=voice.engine, voice=voice.voice, profile=voice.profile)
    return TtsVoice(
        engine="off", voice=settings.placement.tts_voice or "none", profile=settings.profile
    )


class RealtimeSession:
    """One WebSocket's session: the controller's `Outbound` and `ReconfigureHost`.

    Args:
        session_id: The session's id, as `session_started` announced it.
        websocket: The accepted socket.
        endpoint: The endpoint the connection is bound to: `desktop`, or a room device's name.
        endpoint_kind: What kind of endpoint it is, which picks the spotter calibration and is
            what a `voice_start`'s advisory `endpoint` is compared with: one naming another kind
            is logged and the bound one kept. A `room` session also accepts only the device's
            message types (`ROOM_CLIENT_MESSAGE_TYPES`) and no `0x11` frame, and is sent only
            the types it acts on (`ROOM_SERVER_MESSAGE_TYPES`) and binary frames. None: `desktop`
            for the `desktop` endpoint and `room` for any other, so a session bound to a device
            is never given the desktop's types by omission.
        services: What every session shares.
        host: The registry.
        max_turn_items: The `turn` lane's bound.
        max_bulk_items: The `bulk` lane's bound.
    """

    def __init__(
        self,
        *,
        session_id: str,
        websocket: SessionSocket,
        endpoint: Endpoint,
        services: SessionServices,
        host: SessionHost,
        endpoint_kind: EndpointKind | None = None,
        max_turn_items: int = TURN_LANE_MAX_ITEMS,
        max_bulk_items: int = BULK_LANE_MAX_ITEMS,
    ) -> None:
        self.session_id = session_id
        self.endpoint: Endpoint = endpoint
        if endpoint_kind is None:
            endpoint_kind = "desktop" if endpoint == "desktop" else "room"
        self.endpoint_kind: EndpointKind = endpoint_kind
        self._socket = websocket
        self._services = services
        self._host = host
        self._settings = services.settings
        self._clock = services.clock_factory()
        self._new_utterance_id = services.utterance_ids()

        self._control: deque[tuple[object, int | None]] = deque()  # (item, fence)
        self._stop_window = False
        self._turn = TurnLane(max_turn_items)
        self._bulk: deque[object] = deque()
        self._max_bulk = max_bulk_items
        self._ready = asyncio.Event()
        self.bulk_dropped = 0

        self._voice: _Voice | None = None
        self._kept: VoiceStartMessage | None = None
        self._power_check: asyncio.Handle | None = None
        self._spoken_turn: str | None = None
        self._unsubscribe_tts: Callable[[], None] | None = None
        self._epoch = 0
        self._muted = False
        self._pos = 0
        self._max_req = 0
        self._answers: deque[_Answer] = deque()
        self.ignored_frames = 0
        self.dropped_references = 0
        self._last_protocol_error: int | None = None
        self._gpu_yield_logged = False
        self.refused_messages = 0
        self._refused_types: set[str] = set()
        self.withheld = 0
        self._withheld_types: set[str | None] = set()

        self._activity = False
        self._activity_check: asyncio.Handle | None = None
        self._close_code: int | None = None
        self._closing = asyncio.Event()
        self._sender: asyncio.Task[None] | None = None
        self._controller = self._build_controller()

    # -- Wiring ------------------------------------------------------------------------------

    def _build_controller(self) -> TurnController:
        services = self._services
        settings = self._settings
        engines = services.engines
        spotter = engines.spotter
        own = "server" if self.endpoint_kind == "desktop" else "endpoint"
        synthesizer = engines.tts_slot if engines.tts_slot is not None else engines.synthesizer
        responder = services.responder
        if services.power_gate is not None:
            responder = PowerGatedResponder(responder, services.power_gate)
        return services.controller_factory(
            session_id=self.session_id,
            responder=responder,
            outbound=self,
            host=self,
            placement=services.placement,
            arbiter=self._host.arbiter,
            gate=AddresseeGate(settings.wake, echo_guard_ms=settings.barge_in.echo_guard_ms),
            wake=settings.wake,
            calibration=None if spotter is None else spotter.calibration.get(own),
            hit_lag_ms=0 if spotter is None else spotter.hit_lag_ms,
            recent_tts_text=self._host.recent_tts_text,
            new_utterance_id=self._new_utterance_id,
            clock=self._clock,
            synthesizer=synthesizer,
            tts_lane=None if synthesizer is None else services.lanes.tts,
            tts_enabled=services.tts_enabled,
            segmenter_config=settings.segmenter,
            playback_config=settings.playback,
            turn_config=settings.turn,
            barge_in_config=settings.barge_in,
            alarm_ringing=services.alarm_ringing,
            power_level=self.power_level,
        )

    def power_level(self) -> PowerLevel:
        """The level shown in `voice_state` and `turn_metrics`: the gate's, else the services'."""
        gate = self._services.power_gate
        return gate.level() if gate is not None else self._services.power_level()

    def _voice_blocked(self) -> bool:
        """The tier has no provider and voice may not take spoken commands (D2 item 4)."""
        gate = self._services.power_gate
        if gate is None or gate.model_available():
            return False
        statuses = self._services.engines.statuses
        return not gate.spoken_commands_allowed(_on_cpu(statuses.stt), _on_cpu(statuses.tts))

    @property
    def controller(self) -> TurnController:
        """The session's turn controller."""
        return self._controller

    @property
    def voice_on(self) -> bool:
        """Whether voice is active: a `voice_start` was accepted and no `voice_stop` followed."""
        return self._voice is not None

    @property
    def voice_activity(self) -> bool:
        """An utterance is open, or a turn is held, thinking or speaking."""
        voice = self._voice
        if voice is not None and voice.ingress.idle_since() is None:
            return True
        return self._controller.phase in ("held", "thinking", "speaking")

    def recent_tts_text(self, window_ms: int) -> str:
        """The text this session played within `window_ms`, from its latest spoken turn.

        The controller keeps that turn's ledger after the turn ends, so the text a turn ended on
        stays readable for the echo guard.
        """
        return self._controller.recent_tts_text(window_ms)

    # -- Lifecycle ---------------------------------------------------------------------------

    async def run(self) -> None:
        """Serve the connection until it closes, then commit the turn and unregister."""
        loop = asyncio.get_running_loop()
        tts_slot = self._services.engines.tts_slot
        if tts_slot is not None:
            self._unsubscribe_tts = tts_slot.subscribe(self._on_tts_release)
        self._controller.start()
        self._sender = loop.create_task(self._send_loop(), name=f"session-send-{self.session_id}")
        self._sender.add_done_callback(self._sender_done)
        receiver = loop.create_task(self._receive_loop(), name=f"session-receive-{self.session_id}")
        receiver.add_done_callback(lambda _task: self._closing.set())
        try:
            await self._closing.wait()
        finally:
            receiver.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await receiver
            await self._teardown()

    def close(self, code: int) -> None:
        """Close the connection with `code`; `run` tears the session down."""
        if self._close_code is None and not self._closing.is_set():
            self._close_code = code
        self._closing.set()

    def _on_tts_release(self) -> None:
        """A GPU release cancelled TTS: cut a spoken reply whole, as the UI's stop does.

        The cancel already ended the synthesis; the interrupt makes the controller end the turn,
        so it is recorded as interrupted. A typed turn streams no audio and goes on.
        """
        if self._spoken_turn is not None and self._controller.phase in ("thinking", "speaking"):
            self._controller.post(Interrupt())

    async def _teardown(self) -> None:
        if self._unsubscribe_tts is not None:
            self._unsubscribe_tts()
            self._unsubscribe_tts = None
        await self._controller.close()
        voice, self._voice = self._voice, None
        if voice is not None:
            voice.ingress.close()
        sender = self._sender
        if sender is not None and not sender.done():
            sender.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sender
        if self._activity_check is not None:
            self._activity_check.cancel()
        if self._power_check is not None:
            self._power_check.cancel()
        self._host.session_closed(self)
        if self._close_code is not None:
            await self._socket.close(self._close_code)

    def _sender_done(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error("session %s: sender failed: %r; closing", self.session_id, error)
            self.close(CLOSE_INTERNAL_ERROR)

    # -- Outbound lanes ------------------------------------------------------------------------

    def send_control(self, item: object) -> None:
        """Queue an item on the `control` lane, which the sender drains first unless it is fenced.

        Only the module's rules 2 and 3 can fence an item queued here: a `voice_state` queued
        while a fenced item waits, and any item queued in the `voice_stop` window, takes the
        newest waiting fence. Rule 1, a new fence at the `turn` lane's mark, is `apply`'s alone.
        """
        self._queue_control(item, None)

    def _queue_control(self, item: object, fence: int | None) -> None:
        """Queue `item` with its own fence (rule 1) or None, then apply rules 2 and 3."""
        waiting = self._waiting_fence()
        if waiting is None:
            self._stop_window = False  # the window closes once its fence has passed
        elif fence is not None:
            fence = max(fence, waiting)
        elif self._stop_window or isinstance(item, VoiceStateMessage):
            fence = waiting
        self._control.append((item, fence))
        self._ready.set()
        self._schedule_activity_check()

    def _waiting_fence(self) -> int | None:
        """The newest fence on `control` the `turn` lane has not passed, None when none waits.

        Fences never decrease along `control`: a rule 1 fence is at least the lane's mark at
        its queueing, which no earlier fence exceeds, and rules 2 and 3 copy the newest waiting
        fence. So the last fenced item has the largest, and once the lane has passed it, it has
        passed every other.
        """
        for _item, fence in reversed(self._control):
            if fence is not None:
                return None if self._turn.passed(fence) else fence
        return None

    def _open_stop_window(self) -> None:
        """Rule 3: hold every later `control` item behind a waiting answer (M63)."""
        if self._waiting_fence() is not None:
            self._stop_window = True

    def send_turn(
        self,
        turn_id: str,
        item: object,
        terminal: bool = False,
        kind: Literal["stream_start", "token"] | None = None,
        token_index: int | None = None,
    ) -> bool:
        """Queue a turn's item; a full lane closes the session with 1013 and returns False."""
        try:
            self._turn.put(turn_id, item, terminal=terminal, kind=kind, token_index=token_index)
            if isinstance(item, StreamStartMessage):
                self._on_stream_start(item)
        except LaneFull:
            if not self._closing.is_set():
                logger.error("session %s: turn lane full; closing with 1013", self.session_id)
            self.close(CLOSE_TRY_AGAIN_LATER)
            return False
        self._ready.set()
        self._schedule_activity_check()
        return True

    def purge_turn(self, turn_id: str) -> LanePurge:
        """Remove the turn's items the sender has not taken, and report what went."""
        return self._turn.purge(turn_id)

    def send_bulk(self, item: object) -> None:
        """Queue a log line or broadcast; past the bound the oldest is dropped and counted."""
        self._bulk.append(item)
        while len(self._bulk) > self._max_bulk:
            self._bulk.popleft()
            self.bulk_dropped += 1
        self._ready.set()

    def _on_stream_start(self, start: StreamStartMessage) -> None:
        """A voice turn on the text channel because no synthesizer loaded is `tts_unavailable`.

        The controller picks the text channel without saying why; the session knows the slot
        is off, so it queues the decision right after the turn's `stream_start`. A `tts` turn
        is noted, so a GPU release knows a spoken reply is the one to cut.
        """
        self._spoken_turn = start.turn_id if start.channel == "tts" else None
        voice = self._voice
        if (
            start.origin == "voice"
            and start.channel == "text"
            and self._services.tts_enabled
            and self._services.engines.synthesizer is None
            and voice is not None
            and voice.settings.playback
        ):
            self._turn.put(
                start.turn_id,
                VoiceDecisionDegraded(
                    turn_id=start.turn_id,
                    reason="tts_unavailable",
                    detail=DegradedDetail(
                        engine=None, error=None, voiced_text=None, text_from_token=None
                    ),
                ),
            )

    def _next_item(self) -> object | None:
        for index, (control, fence) in enumerate(self._control):
            if fence is None or self._turn.passed(fence):
                del self._control[index]
                return control
        item = self._turn.take()
        if item is not None:
            return item
        if self._bulk:
            return self._bulk.popleft()
        return None

    async def _send_loop(self) -> None:
        while True:
            item = self._next_item()
            if item is None:
                self._ready.clear()
                await self._ready.wait()
                continue
            await self._write(item)

    async def _write(self, item: object) -> None:
        """Send one item the lanes gave up; a `room` session is sent only what its device reads.

        The check runs here, after the lanes, so a withheld item has already moved its lane on
        and a fence behind it passes as if it had been sent.
        """
        if isinstance(item, Frame):
            await self._socket.send_bytes(encode_frame(item))
            return
        if self.endpoint_kind != "desktop":
            message_type = self._type_of(item)
            if message_type not in ROOM_SERVER_MESSAGE_TYPES:
                self._withhold(message_type)
                return
        if isinstance(item, BaseModel):
            await self._socket.send_text(item.model_dump_json())
        elif isinstance(item, str):
            await self._socket.send_text(item)
        else:
            await self._socket.send_text(json.dumps(item))

    @staticmethod
    def _type_of(item: object) -> str | None:
        """A text item's `type`: a model's field, a mapping's key, or a JSON string's key.

        None when it has none that is a string, an unparsable string included, so the caller
        withholds it: an item whose type cannot be read is not one the device reads.
        """
        if isinstance(item, BaseModel):
            message_type = getattr(item, "type", None)
        elif isinstance(item, Mapping):
            message_type = item.get("type")
        elif isinstance(item, str):
            try:
                data = json.loads(item)
            except (ValueError, RecursionError):
                return None
            message_type = data.get("type") if isinstance(data, dict) else None
        else:
            return None
        return message_type if isinstance(message_type, str) else None

    def _withhold(self, message_type: str | None) -> None:
        """Count an item a `room` session is not sent, and log its type once per session."""
        self.withheld += 1
        if message_type not in self._withheld_types:
            self._withheld_types.add(message_type)
            logger.debug("room_message_withheld session=%s type=%s", self.session_id, message_type)

    # -- Voice activity ------------------------------------------------------------------------

    def _schedule_activity_check(self) -> None:
        if self._activity_check is None:
            loop = asyncio.get_running_loop()
            self._activity_check = loop.call_soon(self._check_activity)

    def _check_activity(self) -> None:
        self._activity_check = None
        active = self.voice_activity
        if active != self._activity:
            self._activity = active
            self._host.voice_activity_changed(self)

    def _emit(self, event: object) -> None:
        if not self._power_admits(event):
            return
        self._controller.post(event)  # type: ignore[arg-type]
        self._schedule_activity_check()

    def _power_admits(self, event: object) -> bool:
        """Whether an ingress event may reach the controller at the current power.

        Without a model no turn starts speculatively, so an `EndpointCandidate` is dropped: a
        held turn would run the capability path, which acts on the world, before its utterance
        is known to have ended. While voice is blocked an ended utterance closes with `dropped
        power_off` instead of opening a turn, and the session stops voice on the next step,
        outside the ingress callback that emitted the event.
        """
        gate = self._services.power_gate
        if gate is None or gate.model_available():
            return True
        if isinstance(event, EndpointCandidate):
            return False
        if not self._voice_blocked():
            return True
        self._schedule_power_check()
        if isinstance(event, UtteranceEnded):
            self._controller.post(
                UtteranceDropped(
                    epoch=event.epoch,
                    event_seq=event.event_seq,
                    utterance_id=event.utterance_id,
                    first_pos=event.first_pos,
                    reason="power_off",  # type: ignore[arg-type]
                    text=event.text,
                    duration_ms=event.duration_ms,
                    no_speech_prob=event.no_speech_prob,
                    avg_logprob=event.avg_logprob,
                )
            )
            return False
        return True

    # -- Power ---------------------------------------------------------------------------------

    def _schedule_power_check(self) -> None:
        if self._power_check is None:
            loop = asyncio.get_running_loop()
            self._power_check = loop.call_soon(self._apply_power)

    def _apply_power(self) -> None:
        """Stop voice the power blocks, keeping its `voice_start`; restart it once it does not."""
        self._power_check = None
        if self._services.power_gate is None or self._closing.is_set():
            return
        blocked = self._voice_blocked()
        if blocked and self._voice is not None:
            self._suspend_voice()
        elif (
            not blocked
            and self._voice is None
            and self._kept is not None
            and self._refusal(self._kept) is None
        ):
            settings, self._kept = self._kept, None
            self._controller.post(self._start_voice(settings, seed=None))

    def _suspend_voice(self) -> None:
        """Close ingress, its open utterance with `dropped power_off`, then stop the controller.

        The drop is posted before `VoiceStop`, so the controller closes the utterance with the
        power's reason, and the stop then finds nothing open. The settings are kept, so the
        rise restarts voice with no new `voice_start` from the client. An applied answer still
        waiting opens the `voice_stop` window, as a `voice_stop` does.
        """
        voice, self._voice = self._voice, None
        assert voice is not None  # noqa: S101 - the caller checked it
        self._open_stop_window()
        self._kept = voice.settings
        drop = voice.ingress.close()
        if drop is not None:
            # `power_off` is a wire reason that ingress's `DropReason` does not list yet.
            power_drop = dataclasses.replace(drop, reason="power_off")  # type: ignore[arg-type]
            self._controller.post(power_drop)
        self._controller.post(VoiceStop())
        self._schedule_activity_check()

    # -- Receive -------------------------------------------------------------------------------

    async def _receive_loop(self) -> None:
        while True:
            message = await self._socket.receive()
            if message.get("type") == "websocket.disconnect":
                return
            data = message.get("bytes")
            if isinstance(data, bytes):
                self._on_binary(data)
                continue
            text = message.get("text")
            if isinstance(text, str):
                self._on_text(text)

    def _protocol_error(self, error: ProtocolError, *, message: str | None = None) -> None:
        """Answer a protocol violation, at most once a second; `message` replaces the text."""
        now = self._clock.now_ms()
        last = self._last_protocol_error
        if last is not None and now - last < PROTOCOL_ERROR_INTERVAL_MS:
            return
        self._last_protocol_error = now
        self.send_control(
            ErrorMessage(
                kind="protocol",
                message=str(error) if message is None else message,
                retriable=False,
                context={"field": error.field},
            )
        )

    def _refuse(self, message_type: str) -> None:
        """Refuse a type a `room` session does not accept: answered, counted, never dispatched.

        The session stays open, so a firmware build that sends a new type is told so and keeps
        its turns, rather than reconnecting in a loop.
        """
        self.refused_messages += 1
        if message_type not in self._refused_types:
            self._refused_types.add(message_type)
            logger.warning(
                "room_message_refused session=%s endpoint=%s type=%s",
                self.session_id,
                self.endpoint,
                message_type,
            )
        reason = f"{message_type} is not accepted from a room endpoint"
        self._protocol_error(ProtocolError("type", reason), message=reason)

    def _on_binary(self, data: bytes) -> None:
        try:
            frame = decode_frame(data, direction=Direction.UP)
        except ProtocolError as error:
            self._protocol_error(error)
            return
        voice = self._voice
        if frame.msg_type == MsgType.MIC_AUDIO:
            pos = self._pos
            self._pos += 1
            if voice is None:
                self.ignored_frames += 1
                return
            voice.ingress.push_frame(frame, pos)
        elif frame.msg_type == MsgType.REF_AUDIO:
            # A room device sends no reference frames, so a `room` session takes none.
            if self.endpoint_kind != "desktop" or voice is None or not voice.settings.reference:
                self.dropped_references += 1
                return
            voice.ingress.push_reference(frame.seq, frame.payload)

    def _on_text(self, raw: str) -> None:
        peek = peek_voice_start(raw)
        try:
            message = parse_client_message(raw)
        except ProtocolError as error:
            if peek is not None:
                self._answer_now(_Answer(req=peek.req), "invalid")
            else:
                self._protocol_error(error)
            return
        # Checked on the parsed type, the one `_dispatch` matches on; anything but the desktop
        # is held to the device's types.
        if self.endpoint_kind != "desktop" and message.type not in ROOM_CLIENT_MESSAGE_TYPES:
            self._refuse(message.type)
            return
        self._dispatch(message)

    def _dispatch(self, message: BaseModel) -> None:
        post = self._controller.post
        match message:
            case VoiceStartMessage():
                self._on_voice_start(message)
            case VoiceStopMessage():
                self._on_voice_stop()
            case MuteMessage():
                self._on_mute(message.active)
            case TextMessage():
                post(Text(message.text))
            case InterruptMessage():
                post(Interrupt())
            case WakeMessage():
                post(
                    Wake(
                        keyword=message.keyword,
                        score=message.score,
                        backlog_ms=message.backlog_ms,
                        arrival=self._clock.now_ms(),
                    )
                )
            case BargeInMessage():
                post(BargeInReport(turn_id=message.turn_id, keyword=message.keyword))
            case PlaybackProgressMessage():
                post(PlaybackProgress(turn_id=message.turn_id, played_ms=message.played_ms))
            case PlaybackDoneMessage():
                post(PlaybackDone(turn_id=message.turn_id, played_ms=message.played_ms))
            case GpuYieldMessage():
                if not self._gpu_yield_logged:
                    self._gpu_yield_logged = True
                    logger.info(
                        "session %s: gpu_yield is not used in protocol 2; ignored",
                        self.session_id,
                    )
            case SubscribeLogsMessage():
                self._host.set_log_subscription(self, message.levels, message.enabled)
                state = "enabled" if message.enabled else "disabled"
                self.send_control({"type": "status", "message": f"Log streaming: {state}"})
            case LogConfigMessage():
                self._host.set_logger_level(message.logger, message.level)
                self.send_control(
                    {"type": "log_config_ack", "logger": message.logger, "level": message.level}
                )
            case ActionDecisionMessage() | AlarmActionMessage() | AlarmReportMessage():
                self._services.on_capability_message(self.session_id, message)
        self._schedule_activity_check()

    # -- Voice start, stop and mute ----------------------------------------------------------

    def _refusal(self, settings: VoiceStartMessage) -> str | None:
        if self._voice_blocked():
            return "power_off"
        engines = self._services.engines
        if engines.vad is None or engines.transcriber is None:
            return "vad unavailable"
        if settings.spotter == "server" and engines.spotter is None:
            return "kws unavailable"
        # No verifier, no wake accepted: refused before the device opens its uplink for one.
        if settings.spotter == "endpoint" and engines.verifier is None:
            return "verify unavailable"
        return None

    def _on_voice_start(self, settings: VoiceStartMessage) -> None:
        if settings.req <= self._max_req:
            self._answer_now(_Answer(req=settings.req), "invalid")
            return
        self._max_req = settings.req
        if settings.endpoint != self.endpoint_kind:
            logger.warning(
                "session %s: voice_start names endpoint %s; the connection is bound to %s (%s)",
                self.session_id,
                settings.endpoint,
                self.endpoint,
                self.endpoint_kind,
            )
        refusal = self._refusal(settings)
        voice = self._voice
        if voice is None:
            if refusal is not None:
                self._answer_now(_Answer(req=settings.req), refusal)
                return
            changed = settings.muted != self._muted
            self._muted = settings.muted
            self._kept = None
            start = self._start_voice(settings, seed=None)
            self._answer_now(_Answer(req=settings.req), None)
            self._controller.post(start)
            if changed:
                self._controller.post(Mute(self._muted))
            return
        if settings.muted != self._muted:
            self._set_muted(settings.muted)
        request = Reconfigure(settings=settings, refusal=refusal)
        self._answers.append(_Answer(req=settings.req, request=request))
        self._controller.post(request)

    def _start_voice(self, settings: VoiceStartMessage, seed: IngressSeed | None) -> VoiceStart:
        """Build an ingress and a policy for `settings` and route uplink to it."""
        services = self._services
        engines = services.engines
        lanes = services.lanes
        config = self._settings
        self._epoch += 1
        ingress = services.ingress_factory(
            engines=IngressEngines(
                vad=engines.vad,
                transcriber=(
                    engines.stt_slot if engines.stt_slot is not None else engines.transcriber
                ),
                turn_end=engines.turn_end,
                spotter=engines.spotter,
                verifier=engines.verifier,
            ),
            lanes=IngressLanes(
                vad=lanes.vad, stt=lanes.stt, eot=lanes.eot, kws=lanes.kws, verify=lanes.verify
            ),
            endpointing=config.endpointing,
            wake=config.wake,
            emit=self._emit,
            retention_ms=ring_retention_ms(config.turn, config.wake),
            epoch=self._epoch,
            new_utterance_id=self._new_utterance_id,
            clock=self._clock.now_ms,
            seed=seed,
        )
        if self._muted:
            ingress.set_muted(True)  # before any frame reaches it
        policy = KeywordBargeIn(
            config.barge_in,
            aec=settings.aec,
            names=config.wake.names,
            recent_tts_text=self.recent_tts_text,
            clock=self._clock.now_ms,
        )
        self._voice = _Voice(settings=settings, ingress=ingress, epoch=self._epoch)
        return VoiceStart(ingress=ingress, policy=policy, settings=settings, epoch=self._epoch)

    def _on_voice_stop(self) -> None:
        """Voice is off from receipt (M63): pending answers are refused, then `VoiceStop` posted.

        An applied answer still waiting opens the `voice_stop` window first, so the refusals
        and the controller's stop messages follow it.
        """
        self._open_stop_window()
        self._kept = None
        if self._voice is None:
            return
        self._voice = None
        for slot in self._answers:
            if not slot.ready:
                slot.ready, slot.refused = True, "voice_stop"
        self._flush_answers()
        self._controller.post(VoiceStop())

    def _set_muted(self, muted: bool) -> None:
        """Mute ingress at receipt, so no later frame reaches the VAD, then tell the mailbox."""
        self._muted = muted
        voice = self._voice
        if voice is not None:
            voice.ingress.set_muted(muted)
        self._controller.post(Mute(muted))

    def _on_mute(self, active: bool) -> None:
        self._set_muted(active)
        self.send_voice_state()

    # -- ReconfigureHost -----------------------------------------------------------------------

    def _slot_of(self, request: Reconfigure) -> _Answer | None:
        return next((slot for slot in self._answers if slot.request is request), None)

    def apply(self, request: Reconfigure, seed: IngressSeed) -> VoiceStart | None:
        """Build the seeded ingress, switch routing to it, and answer the request accepted.

        The answer is fenced at the `turn` lane's mark (rule 1): the controller applies at a
        quiescent point, after the ended turn queued its `state_cycle idle` and `turn_metrics`
        on the lane, and the answer must not overtake them.
        """
        slot = self._slot_of(request)
        if slot is None or slot.ready or self._voice is None:
            return None
        start = self._start_voice(request.settings, seed=seed)
        slot.ready = True
        self._flush_answers(applied=slot)
        return start

    def refuse(self, request: Reconfigure, reason: str) -> None:
        """Answer the request refused, unless the session already answered it."""
        slot = self._slot_of(request)
        if slot is None or slot.ready:
            return
        slot.ready, slot.refused = True, reason
        self._flush_answers()

    # -- voice_state ---------------------------------------------------------------------------

    def _answer_now(self, slot: _Answer, refused: str | None) -> None:
        slot.ready, slot.refused = True, refused
        self._answers.append(slot)
        self._flush_answers()

    def _flush_answers(self, applied: _Answer | None = None) -> None:
        """Send every ready answer at the head, in `req` order; only `applied` sets a fence."""
        while self._answers and self._answers[0].ready:
            slot = self._answers.popleft()
            fence = self._turn.mark() if slot is applied else None
            answer = self.voice_state("answer", req=slot.req, refused=slot.refused)
            self._queue_control(answer, fence)

    def voice_state(
        self,
        cause: Literal["answer", "broadcast"],
        *,
        req: int | None = None,
        refused: str | None = None,
    ) -> VoiceStateMessage:
        """This session's `voice_state`, composed now.

        A broadcast to a session whose voice the power stopped says `refused: power_off`.
        """
        settings = self._settings
        engines = self._services.engines
        voice = self._voice
        playback = settings.playback
        if (
            cause == "broadcast"
            and voice is None
            and self._kept is not None
            and self._voice_blocked()
        ):
            refused = "power_off"
        return VoiceStateMessage(
            cause=cause,
            req=req,
            active=voice is not None,
            mode=None if voice is None else voice.settings.mode,
            uplink=UPLINK_FORMAT,
            downlink=StreamFormat(sample_rate=DOWNLINK_SAMPLE_RATE, frame_ms=playback.frame_ms),
            playback=PlaybackSettings(
                client_prebuffer_ms=playback.client_prebuffer_ms,
                fade_ms=playback.fade_ms,
                progress_interval_ms=playback.progress_interval_ms,
            ),
            barge_in=BargeInSettings(
                policy="keyword",
                keywords=list(settings.barge_in.keywords),
                arm_delay_ms=settings.barge_in.arm_delay_ms,
            ),
            budget=wire_budget(settings),
            engines=engines.statuses,
            tts_voice=tts_voice_of(engines, settings),
            vram_budget_mib=settings.placement.vram_budget_mib,
            power_level=self.power_level(),
            muted=self._muted,
            refused=refused,
        )

    def send_voice_state(self) -> None:
        """Apply the power to voice, then send this session a broadcast `voice_state`.

        The registry sends one after every power level or engine change, so this is where a
        session stops voice the power blocks and restarts it from its kept `voice_start`.
        """
        self._apply_power()
        self.send_control(self.voice_state("broadcast"))
