"""One realtime session per WebSocket: uplink routing, voice settings, and the outbound lanes.

The session owns the socket and the voice configuration; the `TurnController` owns turns and is
the only caller of ingress levers. The session dispatches each client message to the mailbox,
routes uplink frames to the current `AudioIngress`, answers every `voice_start` in the order
received, and drains three outbound lanes through one sender:

- `control`: unbounded; voice decisions, `voice_state`, errors. Always sent first.
- `turn`: a `TurnLane` bounded at 2048 items; a turn's stream and frames, purged by turn id. A
  full lane means the client stopped reading, so the session closes with 1013.
- `bulk`: logs and broadcasts, bounded at 512 items, dropping the oldest and counting the drops.

The sender takes one item at a time, so a purge between two takes removes everything the client
has not been sent yet.
"""

from __future__ import annotations

import asyncio
import contextlib
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
from .ingress import AudioIngress, IngressEngines, IngressLanes, IngressSeed, ring_retention_ms
from .ingress import Lane as InferenceLane
from .playback import LaneFull, LanePurge, MonotonicPlaybackClock, PlaybackClock, TurnLane
from .protocol import (
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

Endpoint = Literal["desktop", "room"]
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
    every session reads them at the next `voice_start` and `voice_state`.
    """

    vad: VoiceActivityDetector | None = None
    turn_end: TurnEndDetector | None = None
    transcriber: StreamingTranscriber | None = None
    synthesizer: SpeechSynthesizer | None = None
    spotter: NameSpotter | None = None
    statuses: EngineStatuses = ALL_OFF
    vram_over_budget: bool = False


@dataclass(frozen=True, slots=True)
class InferenceLanes:
    """The process-wide lanes each engine's calls run on, one per engine kind."""

    vad: InferenceLane
    eot: InferenceLane
    stt: InferenceLane
    tts: InferenceLane
    kws: InferenceLane


def _full() -> PowerLevel:
    return "full"


def _utterance_ids() -> Callable[[], str]:
    """One id source per session, shared by its ingresses and its controller (v4)."""
    return lambda: f"utt-{uuid.uuid4().hex[:16]}"


def _ignore_capability(session_id: str, message: CapabilityMessage) -> None:
    logger.debug("session %s: capability message %s has no handler", session_id, message.type)


@dataclass(frozen=True, slots=True)
class SessionServices:
    """What every session shares, built once by `MistRuntime`; tests pass fakes.

    `ingress_factory` and `controller_factory` default to the real classes; a test passes its
    own to record what the session builds. `on_capability_message` receives `action_decision`,
    `alarm_action` and `alarm_report`, which belong to the capability host (T22).
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
        endpoint: The endpoint the connection is bound to; a `voice_start` naming another one
            is logged and the bound one kept.
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
        max_turn_items: int = TURN_LANE_MAX_ITEMS,
        max_bulk_items: int = BULK_LANE_MAX_ITEMS,
    ) -> None:
        self.session_id = session_id
        self.endpoint: Endpoint = endpoint
        self._socket = websocket
        self._services = services
        self._host = host
        self._settings = services.settings
        self._clock = services.clock_factory()
        self._new_utterance_id = services.utterance_ids()

        self._control: deque[object] = deque()
        self._turn = TurnLane(max_turn_items)
        self._bulk: deque[object] = deque()
        self._max_bulk = max_bulk_items
        self._ready = asyncio.Event()
        self.bulk_dropped = 0

        self._voice: _Voice | None = None
        self._epoch = 0
        self._muted = False
        self._pos = 0
        self._max_req = 0
        self._answers: deque[_Answer] = deque()
        self.ignored_frames = 0
        self.dropped_references = 0
        self._last_protocol_error: int | None = None
        self._gpu_yield_logged = False
        self._ledger: object | None = None

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
        own = "server" if self.endpoint == "desktop" else "endpoint"
        synthesizer = engines.synthesizer
        return services.controller_factory(
            session_id=self.session_id,
            responder=services.responder,
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
            power_level=services.power_level,
        )

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

        The ledger is the controller's, read through its private turn record: the controller
        offers no public view of it, and the ledger outlives the turn, so the text a turn ended
        on stays readable for the echo guard after it ends.
        """
        ledger = self._ledger
        if ledger is None:
            return ""
        return ledger.recent_tts_text(window_ms)  # type: ignore[attr-defined]

    # -- Lifecycle ---------------------------------------------------------------------------

    async def run(self) -> None:
        """Serve the connection until it closes, then commit the turn and unregister."""
        loop = asyncio.get_running_loop()
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

    async def _teardown(self) -> None:
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
        """Queue an item on the `control` lane, which the sender always drains first."""
        self._control.append(item)
        self._ready.set()
        self._schedule_activity_check()

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
        self._note_ledger()
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
        is off, so it queues the decision right after the turn's `stream_start`.
        """
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

    def _note_ledger(self) -> None:
        turn = getattr(self._controller, "_turn", None)
        ledger = getattr(turn, "ledger", None)
        if ledger is not None:
            self._ledger = ledger

    def _next_item(self) -> object | None:
        if self._control:
            return self._control.popleft()
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
        if isinstance(item, Frame):
            await self._socket.send_bytes(encode_frame(item))
        elif isinstance(item, BaseModel):
            await self._socket.send_text(item.model_dump_json())
        elif isinstance(item, str):
            await self._socket.send_text(item)
        else:
            await self._socket.send_text(json.dumps(item))

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
        self._controller.post(event)  # type: ignore[arg-type]
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

    def _protocol_error(self, error: ProtocolError) -> None:
        now = self._clock.now_ms()
        last = self._last_protocol_error
        if last is not None and now - last < PROTOCOL_ERROR_INTERVAL_MS:
            return
        self._last_protocol_error = now
        self.send_control(
            ErrorMessage(
                kind="protocol",
                message=str(error),
                retriable=False,
                context={"field": error.field},
            )
        )

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
            if voice is None or not voice.settings.reference:
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
        engines = self._services.engines
        if engines.vad is None or engines.transcriber is None:
            return "vad unavailable"
        if settings.spotter == "server" and engines.spotter is None:
            return "kws unavailable"
        return None

    def _on_voice_start(self, settings: VoiceStartMessage) -> None:
        if settings.req <= self._max_req:
            self._answer_now(_Answer(req=settings.req), "invalid")
            return
        self._max_req = settings.req
        if settings.endpoint != self.endpoint:
            logger.warning(
                "session %s: voice_start names endpoint %s; the connection is bound to %s",
                self.session_id,
                settings.endpoint,
                self.endpoint,
            )
        refusal = self._refusal(settings)
        voice = self._voice
        if voice is None:
            if refusal is not None:
                self._answer_now(_Answer(req=settings.req), refusal)
                return
            changed = settings.muted != self._muted
            self._muted = settings.muted
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
                transcriber=engines.transcriber,
                turn_end=engines.turn_end,
                spotter=engines.spotter,
            ),
            lanes=IngressLanes(vad=lanes.vad, stt=lanes.stt, eot=lanes.eot, kws=lanes.kws),
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
            recent_tts_text=self.recent_tts_text,
            clock=self._clock.now_ms,
        )
        self._voice = _Voice(settings=settings, ingress=ingress, epoch=self._epoch)
        return VoiceStart(ingress=ingress, policy=policy, settings=settings, epoch=self._epoch)

    def _on_voice_stop(self) -> None:
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
        """Build the seeded ingress, switch routing to it, and answer the request accepted."""
        slot = self._slot_of(request)
        if slot is None or slot.ready or self._voice is None:
            return None
        start = self._start_voice(request.settings, seed=seed)
        slot.ready = True
        self._flush_answers()
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

    def _flush_answers(self) -> None:
        while self._answers and self._answers[0].ready:
            slot = self._answers.popleft()
            self.send_control(self.voice_state("answer", req=slot.req, refused=slot.refused))

    def voice_state(
        self,
        cause: Literal["answer", "broadcast"],
        *,
        req: int | None = None,
        refused: str | None = None,
    ) -> VoiceStateMessage:
        """This session's `voice_state`, composed now."""
        settings = self._settings
        engines = self._services.engines
        voice = self._voice
        playback = settings.playback
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
            power_level=self._services.power_level(),
            muted=self._muted,
            refused=refused,
        )

    def send_voice_state(self) -> None:
        """Send this session a broadcast `voice_state`, on the `control` lane."""
        self.send_control(self.voice_state("broadcast"))
