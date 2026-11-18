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

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING

from backend.capabilities.power import PowerLevel, PowerState, TierProviders
from backend.chat.stream_events import Complete, StreamEvent, Token, WSEvent
from backend.chat.turn_scope import CommitResult, TurnDelivery, TurnRequest, TurnScope
from backend.gpu_yield import YIELDED_MESSAGE, turn_refusal

if TYPE_CHECKING:
    from backend.chat.conversation_handler import CapabilityOnlyTurn

    from .turn import TurnResponder

logger = logging.getLogger(__name__)

NO_MODEL_MESSAGE = (
    "MIST's language model is not available right now, so she can only take commands."
)

CapabilityOnly = Callable[[str, str], Awaitable["CapabilityOnlyTurn | None"]]
"""`ConversationHandler.handle_capability_only(text, session_id)`, injected until T52 adds it to
the responder."""


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

    Args:
        inner: The model path.
        gate: The power gate.
        capability_only: The capability-only path; None refuses every message without a model.
    """

    def __init__(
        self,
        inner: TurnResponder,
        gate: PowerGate,
        capability_only: CapabilityOnly | None = None,
    ) -> None:
        self._inner = inner
        self._gate = gate
        self._capability_only = capability_only

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
        result = None
        if self._capability_only is not None:
            result = await self._capability_only(request.user_text, request.session_id)
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
