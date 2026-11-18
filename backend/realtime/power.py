"""The power level on the realtime path: the gate that turns and voice read.

Turns gate on the model tier, never on the level (D2): capabilities' `PowerState.level()` is
shown and logged only, and a turn that needs the model runs only while the `interaction` tier
has a provider. While it has none, a turn goes to the capability-only path, and a message that
path does not handle gets the daily build's refusal envelope. Voice takes spoken commands then
only when `spoken_commands_at_off` is on and STT and TTS are both loaded on the CPU.

`PowerGatedResponder` is how the session applies the gate without the controller knowing: it
wraps the turn responder and decides at each `open_turn`, so a turn without a model streams the
capability-only result through the controller as any other turn streams.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import TYPE_CHECKING

import numpy as np

from backend.capabilities.power import PowerLevel, PowerState, TierProviders
from backend.chat.stream_events import Complete, StreamEvent, Token, WSEvent
from backend.chat.turn_scope import CommitResult, TurnDelivery, TurnRequest, TurnScope
from backend.gpu_yield import YIELDED_MESSAGE, turn_refusal

from .errors import SpeechEngineError
from .lanes import LaneJobCancelled
from .speech import (
    NEUTRAL_STATE,
    CancelToken,
    SpeechSynthesizer,
    StreamingTranscriber,
    SynthChunk,
    TextSegment,
    TranscriptStream,
    TranscriptUpdate,
    VoiceInfo,
)

if TYPE_CHECKING:
    from backend.chat.conversation_handler import CapabilityOnlyTurn

    from .turn import TurnResponder

logger = logging.getLogger(__name__)

OFF_SAMPLE_RATE = 24000
OFF_VOICE = VoiceInfo(engine="off", voice="", profile="")
_IDLE_POLL_S = 0.05
LEVEL_RANK: dict[str, int] = {"off": 0, "low": 1, "full": 2}
"""The levels in order, so a change reads as a drop or a rise."""

NO_MODEL_MESSAGE = (
    "MIST's language model is not available right now, so she can only take commands."
)


def gpu_yield_refusal(gpu_state: Callable[[], str]) -> Callable[[], str]:
    """The refusal text by gaming mode's state: the yield's message while it is `yielded`.

    Otherwise the tier has no provider while the GPU is not yielded, so the text says the model
    is missing. `RESUME_FAILED_MESSAGE` is never chosen, because the realtime reload never lets
    `GpuYield` enter `resume_failed` (D2).
    """

    def refusal() -> str:
        return YIELDED_MESSAGE if gpu_state() == "yielded" else NO_MODEL_MESSAGE

    return refusal


class PowerGate:
    """What the session asks of the power level and the model tiers. Safe from any thread.

    Args:
        power: The level, shown in `voice_state` and `turn_metrics` and naming `power_off`.
        tiers: The provider of each model tier; a turn needs the `interaction` one.
        refusal: The text an unhandled message is refused with, read at each refusal.
        spoken_commands_at_off: `MIST_VOICE_SPOKEN_COMMANDS_AT_OFF`, on by Raj's ruling of
            7 Nov (A23).
    """

    def __init__(
        self,
        power: PowerState,
        tiers: TierProviders,
        refusal: Callable[[], str],
        *,
        spoken_commands_at_off: bool = True,
    ) -> None:
        self._power = power
        self._tiers = tiers
        self._refusal = refusal
        self._spoken_commands_at_off = spoken_commands_at_off

    @property
    def spoken_commands_at_off(self) -> bool:
        """Whether voice may take capability commands while the tier has no provider."""
        return self._spoken_commands_at_off

    def level(self) -> PowerLevel:
        """The level now; shown and logged, never decided on."""
        return self._power.level()

    def model_available(self) -> bool:
        """Whether a turn may reach the model: the `interaction` tier has a provider now."""
        return self._tiers.provider("interaction") is not None

    def spoken_commands_allowed(self, stt_on_cpu: bool, tts_on_cpu: bool) -> bool:
        """The setting, and STT and TTS both loaded on the CPU; the tier is not consulted.

        The setting has no effect when either engine needs the GPU, because a release never
        moves a CUDA engine to the CPU on its own: only the chain's next option does.
        """
        return self._spoken_commands_at_off and stt_on_cpu and tts_on_cpu

    def refusal_message(self) -> str:
        """The injected refusal's text: `YIELDED_MESSAGE` or `NO_MODEL_MESSAGE` in production."""
        return self._refusal()


class _CapabilityScope(TurnScope):
    """The scope of a turn opened while the tier had no provider; it never reaches the model."""

    __slots__ = ()


class PowerGatedResponder:
    """A `TurnResponder` that sends a turn to the model only while the tier has a provider.

    Without one, the turn's stream is the capability-only result: a handled message streams the
    events the handler produced and already committed; an unhandled one streams the payloads
    the pre-LLM step drained, then the refusal envelope, and in a `tts` turn the refusal's text
    as a token so it is spoken. A held turn waits for its release before the capability path
    runs, because that path acts on the world. Such a turn records nothing at commit: the
    handler committed a handled one, and an unhandled one is not recorded (its contract).

    The capability-only path is the inner responder's port (T52): it never reaches the model,
    so it is passed through at every level, and a stop word while an alarm rings reaches it
    with or without a provider.

    Args:
        inner: The model path, and the capability-only path behind its port.
        gate: The power gate.
    """

    def __init__(self, inner: TurnResponder, gate: PowerGate) -> None:
        self._inner = inner
        self._gate = gate

    def open_turn(self, request: TurnRequest) -> TurnScope:
        """Open on the model path while the tier has a provider, else on the capability path."""
        if self._gate.model_available():
            return self._inner.open_turn(request)
        scope = _CapabilityScope(session_id=request.session_id, turn_index=-1, request=request)
        if not request.held:
            scope.dispatch_gate.set()
        return scope

    def stream_turn(self, scope: TurnScope) -> AsyncIterator[StreamEvent]:
        """The model's stream, or the capability-only one for a scope opened without a model."""
        if isinstance(scope, _CapabilityScope):
            return self._capability_stream(scope)
        return self._inner.stream_turn(scope)

    async def _capability_stream(self, scope: _CapabilityScope) -> AsyncIterator[StreamEvent]:
        request = scope.request
        assert request is not None  # noqa: S101 - `open_turn` sets it
        await scope.dispatch_gate.wait()
        result = await self._inner.handle_capability_only(request.user_text, request.session_id)
        if result is not None and result.handled:
            for event in result.events:
                if isinstance(event, Token):
                    scope.stream.emitted += event.text
                yield event
            return
        if result is not None:
            for event in result.events:
                yield event
        message = self._gate.refusal_message()
        yield WSEvent(payload=turn_refusal(message))
        spoken = ""
        if request.channel == "tts":
            spoken = message
            scope.stream.emitted += message
            yield Token(message)
        yield Complete(final_response=spoken)

    def commit_turn(self, scope: TurnScope, delivery: TurnDelivery) -> CommitResult:
        """The model's commit; a capability-path turn records nothing here."""
        if isinstance(scope, _CapabilityScope):
            return CommitResult(recorded=False, event_id=None, error=None)
        return self._inner.commit_turn(scope, delivery)

    def prefetch(self, session_id: str, partial_text: str | None) -> None:
        """Forward the hint only while a model can use it."""
        if self._gate.model_available():
            self._inner.prefetch(session_id, partial_text)

    async def handle_capability_only(self, text: str, session_id: str) -> CapabilityOnlyTurn | None:
        """The inner responder's capability-only path, at any level; it never reaches the model."""
        return await self._inner.handle_capability_only(text, session_id)


# -- Engine slots --------------------------------------------------------------------------------


class _EngineSlot:
    """One speech slot's current engine, swapped by the GPU release and reload.

    Work holds the engine it started on, so a swap never pulls an engine from under a job:
    `busy(engine)` is true while any hold on it remains, and the release waits on `wait_idle`
    before it drops the engine. `option` is the chain index of the engine loaded now (None while
    off) and `home` the index a reload brings the slot back to (None when no option can load);
    the runtime owns both.
    """

    def __init__(self, engine: object | None, *, option: int | None, home: int | None) -> None:
        self._engine = engine
        self._cond = threading.Condition()
        self._holds: dict[int, int] = {}
        self.option = option
        self.home = home

    @property
    def engine(self) -> object | None:
        """The engine new work starts on; None while the slot is off."""
        return self._engine

    def swap(self, engine: object | None) -> object | None:
        """Make `engine` the one new work starts on and return the previous one."""
        with self._cond:
            old, self._engine = self._engine, engine
            return old

    def busy(self, engine: object) -> bool:
        """Whether a job or an open stream still holds `engine`."""
        with self._cond:
            return self._holds.get(id(engine), 0) > 0

    def wait_idle(self, engine: object, timeout: float) -> bool:
        """Wait at most `timeout` seconds for `engine`'s holds to end; True once they have."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._holds.get(id(engine), 0) > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(min(remaining, _IDLE_POLL_S))
            return True

    def _hold(self, engine: object) -> None:
        with self._cond:
            self._holds[id(engine)] = self._holds.get(id(engine), 0) + 1

    def _drop(self, engine: object) -> None:
        with self._cond:
            count = self._holds.get(id(engine), 0) - 1
            if count > 0:
                self._holds[id(engine)] = count
            else:
                self._holds.pop(id(engine), None)
                self._cond.notify_all()


class _OffStream:
    """The stream of an off STT slot: it takes audio and decodes nothing."""

    def push(self, pcm: np.ndarray) -> None:
        """Discard the audio; nothing transcribes it."""

    def decode_partial(self, token: CancelToken) -> TranscriptUpdate:
        """Refuse: the slot has no engine."""
        raise SpeechEngineError("the STT slot is off")

    def decode_final(self, token: CancelToken) -> TranscriptUpdate:
        """Refuse: the slot has no engine."""
        raise SpeechEngineError("the STT slot is off")

    def close(self) -> None:
        """Nothing to close."""


class _SlotStream:
    """A transcript stream that holds its engine while it is open and while a decode runs.

    A release closes it from the loop; a later push is ignored, and a decode, or one that was
    running and is cancelled, raises `SpeechEngineError`, which the ingress treats as an engine
    failure for the utterance.
    """

    def __init__(self, slot: TranscriberSlot, engine: object, inner: TranscriptStream) -> None:
        self._slot = slot
        self._engine = engine
        self._inner = inner
        self._closed = False
        self._released = False
        self._tokens: list[CancelToken] = []

    @property
    def engine(self) -> object:
        """The engine the stream was opened on."""
        return self._engine

    def push(self, pcm: np.ndarray) -> None:
        """Feed the engine's stream; after a close the audio is ignored."""
        if self._closed:
            return
        self._inner.push(pcm)

    def decode_partial(self, token: CancelToken) -> TranscriptUpdate:
        """A partial decode on the stream's engine."""
        return self._decode(self._inner.decode_partial, token)

    def decode_final(self, token: CancelToken) -> TranscriptUpdate:
        """The final decode on the stream's engine."""
        return self._decode(self._inner.decode_final, token)

    def _decode(
        self, call: Callable[[CancelToken], TranscriptUpdate], token: CancelToken
    ) -> TranscriptUpdate:
        slot = self._slot
        with slot._cond:
            if self._released:
                raise SpeechEngineError("the STT engine was released")
            slot._hold(self._engine)
            self._tokens.append(token)
        try:
            update = call(token)
        except LaneJobCancelled:
            if self._released:
                raise SpeechEngineError("the STT engine was released") from None
            raise
        finally:
            with slot._cond:
                self._tokens.remove(token)
                slot._drop(self._engine)
        if self._released:
            raise SpeechEngineError("the STT engine was released")
        return update

    def close(self) -> None:
        """Close the engine's stream and end the hold it kept."""
        slot = self._slot
        with slot._cond:
            if self._closed:
                return
            self._closed = True
            slot._streams.discard(self)
        try:
            self._inner.close()
        finally:
            slot._drop(self._engine)

    def release_close(self) -> None:
        """The release's close: cancel the decodes running on it, then close."""
        with self._slot._cond:
            self._released = True
            tokens = list(self._tokens)
        for token in tokens:
            token.set()
        self.close()


class TranscriberSlot(_EngineSlot):
    """The STT slot the ingress opens its streams on.

    A stream opened before a swap keeps the engine it opened on until it closes; a stream
    opened while the slot is off decodes nothing and raises `SpeechEngineError` on a decode.
    """

    def __init__(
        self,
        engine: StreamingTranscriber | None,
        *,
        option: int | None = 0,
        home: int | None = 0,
    ) -> None:
        super().__init__(engine, option=option, home=home)
        self._streams: set[_SlotStream] = set()

    @property
    def model_id(self) -> str:
        """The current engine's model id; empty while the slot is off."""
        engine = self._engine
        return "" if engine is None else engine.model_id  # type: ignore[attr-defined]

    def open_stream(self) -> TranscriptStream:
        """A stream on the current engine, holding it until the stream closes."""
        with self._cond:
            engine = self._engine
            if engine is None:
                return _OffStream()
            self._hold(engine)
        opened = False
        try:
            inner = engine.open_stream()  # type: ignore[attr-defined]
            opened = True
        finally:
            if not opened:
                self._drop(engine)
        stream = _SlotStream(self, engine, inner)
        with self._cond:
            self._streams.add(stream)
        return stream

    def release(self) -> None:
        """Nothing: the runtime releases the engines it drops, never a session."""

    def cancel_jobs(self, engine: object) -> None:
        """Close every stream open on `engine`, cancelling its running decodes. Loop thread."""
        with self._cond:
            streams = [stream for stream in self._streams if stream.engine is engine]
        for stream in streams:
            stream.release_close()


class _SlotChunks:
    """One segment's chunks; they hold the engine until exhausted, closed or cut."""

    def __init__(
        self,
        slot: SynthesizerSlot,
        engine: object,
        inner: Iterator[SynthChunk],
        token: CancelToken,
    ) -> None:
        self._slot = slot
        self._engine = engine
        self._inner = inner
        self._token = token
        self._live = True

    @property
    def engine(self) -> object:
        """The engine the segment is synthesized on."""
        return self._engine

    def __iter__(self) -> Iterator[SynthChunk]:
        return self

    def __next__(self) -> SynthChunk:
        slot = self._slot
        with slot._cond:
            if not self._live or self._token.is_set():
                self._finish()
                raise StopIteration
            slot._hold(self._engine)
        try:
            return next(self._inner)
        except StopIteration:
            self._finish()
            raise
        finally:
            slot._drop(self._engine)

    def close(self) -> None:
        """End the segment early, as a generator's `close` does."""
        self._finish()
        close = getattr(self._inner, "close", None)
        if close is not None:
            close()

    def cut(self) -> None:
        """The release's cancel: set the token and end the hold the segment kept."""
        self._token.set()
        self._finish()

    def _finish(self) -> None:
        slot = self._slot
        with slot._cond:
            if not self._live:
                return
            self._live = False
            slot._live.discard(self)
            slot._drop(self._engine)


class SynthesizerSlot(_EngineSlot):
    """The TTS slot the controller synthesizes on.

    A segment in synthesis at a swap finishes on the engine it started on. While the slot is
    off, `synthesize` raises `SpeechEngineError` and the descriptive fields read as engine `off`.
    `subscribe` registers who hears a release's cancel: the session cuts its spoken reply.
    """

    def __init__(
        self,
        engine: SpeechSynthesizer | None,
        *,
        option: int | None = 0,
        home: int | None = 0,
    ) -> None:
        super().__init__(engine, option=option, home=home)
        self._live: set[_SlotChunks] = set()
        self._subscribers: list[Callable[[], None]] = []

    @property
    def model_id(self) -> str:
        """The current engine's model id; empty while the slot is off."""
        engine = self._engine
        return "" if engine is None else engine.model_id  # type: ignore[attr-defined]

    @property
    def engine_id(self) -> str:
        """The current engine's id; `off` while the slot is off."""
        engine = self._engine
        return "off" if engine is None else engine.engine_id  # type: ignore[attr-defined]

    @property
    def sample_rate(self) -> int:
        """The current engine's output rate; the downlink's while the slot is off."""
        engine = self._engine
        if engine is None:
            return OFF_SAMPLE_RATE
        return engine.sample_rate  # type: ignore[attr-defined]

    @property
    def voice(self) -> VoiceInfo:
        """The current engine's voice; engine `off` while the slot is off."""
        engine = self._engine
        return OFF_VOICE if engine is None else engine.voice  # type: ignore[attr-defined]

    @property
    def states(self) -> frozenset[str]:
        """The current engine's speaking states; neutral only while the slot is off."""
        engine = self._engine
        if engine is None:
            return frozenset({NEUTRAL_STATE})
        return engine.states  # type: ignore[attr-defined]

    def synthesize(self, segment: TextSegment, token: CancelToken) -> Iterator[SynthChunk]:
        """The segment's chunks from the current engine, which they hold until they end."""
        with self._cond:
            engine = self._engine
            if engine is None:
                raise SpeechEngineError("the TTS slot is off")
            self._hold(engine)
        started = False
        try:
            inner = engine.synthesize(segment, token)  # type: ignore[attr-defined]
            started = True
        finally:
            if not started:
                self._drop(engine)
        chunks = _SlotChunks(self, engine, iter(inner), token)
        with self._cond:
            self._live.add(chunks)
        return chunks

    def release(self) -> None:
        """Nothing: the runtime releases the engines it drops, never a session."""

    def subscribe(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Call `callback` on the loop at each release's cancel; returns the unsubscribe."""
        self._subscribers.append(callback)

        def unsubscribe() -> None:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

        return unsubscribe

    def cancel_jobs(self, engine: object) -> None:
        """Cut every segment in synthesis on `engine`, then tell the subscribers. Loop thread."""
        with self._cond:
            live = [chunks for chunks in self._live if chunks.engine is engine]
        for chunks in live:
            chunks.cut()
        for callback in list(self._subscribers):
            callback()


# -- The watcher ---------------------------------------------------------------------------------


class PowerWatcher:
    """Reads the level every `interval_s` and reports each change once.

    A drop is reported on the read that sees it. A rise waits until `gpu_state()` is `normal`,
    so the rise follows the reload's return and its `power_restored` never comes from inside
    the reload (D2).

    Args:
        gate: The power gate whose level is read.
        gpu_state: `GpuYield.state`.
        on_change: Called with the new level on the loop; T13 wires `GpuEngines.level_changed`.
        interval_s: Seconds between reads.
        sleep: Awaitable sleep between reads.
    """

    def __init__(
        self,
        gate: PowerGate,
        gpu_state: Callable[[], str],
        on_change: Callable[[PowerLevel], None],
        *,
        interval_s: float = 0.25,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._gate = gate
        self._gpu_state = gpu_state
        self._on_change = on_change
        self._interval_s = interval_s
        self._sleep = sleep
        self._last: PowerLevel | None = None

    def check(self) -> None:
        """Read the level once; the first read sets the baseline."""
        level = self._gate.level()
        last = self._last
        if last is None:
            self._last = level
            return
        if level == last:
            return
        if LEVEL_RANK[level] > LEVEL_RANK[last] and self._gpu_state() != "normal":
            return
        self._last = level
        self._on_change(level)

    async def run(self) -> None:
        """Read now, then every `interval_s`, until cancelled."""
        self.check()
        while True:
            await self._sleep(self._interval_s)
            self.check()
