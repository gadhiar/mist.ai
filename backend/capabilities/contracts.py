"""The types shared by capabilities, the capability host and the turn handler.

A capability is code MIST owns, even when its provider is remote (Home Assistant over MCP, the
Spotify Web API). One contract for all of them is what allows one policy, one validation step
and one action log to apply to every call, and a turn handler to depend on a narrow port
instead of on the host itself.
"""

from __future__ import annotations

import abc
import json
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import BaseModel

from .power import PowerLevel

if TYPE_CHECKING:
    # `RouteDecision` belongs to the router module, which builds on these contracts. A
    # type-only import keeps this module free of any runtime dependency on the router.
    from .router import RouteDecision

RiskClass = Literal["read", "reversible", "physical"]
ConfirmPolicy = Literal["allow", "confirm_inferred", "confirm", "deny"]
CallSource = Literal["model", "fastpath", "proposal"]
OutcomeStatus = Literal[
    "ok", "invalid_args", "pending_confirmation", "unavailable", "error", "denied"
]
CapabilityState = Literal["available", "degraded", "unavailable", "disabled"]

# The model reads outcomes in the actions block; a pending call must never read as done.
_PENDING_NOTE = "not run: waiting for the user's yes or no"


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """One tool a capability offers.

    Attributes:
        name: `"<capability>.<verb>"`, for example `"lights.power"`.
        description: The catalog text the decode prompt shows the model.
        args_model: Validates arguments; also the source of the grammar schema.
        risk: What a wrong call costs, shown with proposals and tool-call events.
        default_policy: The confirmation mode unless the policy file overrides it.
        summarize: Renders validated arguments as a short phrase for proposals and
            `args_summary` ("turn off the bedroom lights").
    """

    name: str
    description: str
    args_model: type[BaseModel]
    risk: RiskClass
    default_policy: ConfirmPolicy
    summarize: Callable[[BaseModel], str]


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    """The result of one tool call, whichever path ran it.

    Attributes:
        status: The closed outcome vocabulary the replies and events key on.
        summary: A speakable sentence written by the capability ("Bedroom lights off.").
        data: Optional structured detail for the model.
        proposal_id: Set when the call waits for confirmation.
        reason: A short cause for a non-ok status ("unreachable", "premium_required").
    """

    status: OutcomeStatus
    summary: str
    data: Mapping[str, object] | None = None
    proposal_id: str | None = None
    reason: str | None = None

    def to_llm_json(self) -> str:
        """Render the outcome as compact JSON for the main pass's actions block.

        Unset fields are left out to keep the block short, and a pending outcome carries a note
        that it has not run, so the model never describes it as done.
        """
        payload: dict[str, object] = {"status": self.status, "summary": self.summary}
        if self.status == "pending_confirmation":
            payload["note"] = _PENDING_NOTE
        if self.reason is not None:
            payload["reason"] = self.reason
        if self.proposal_id is not None:
            payload["proposal_id"] = self.proposal_id
        if self.data is not None:
            payload["data"] = dict(self.data)
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False, default=str)


@dataclass(frozen=True, slots=True)
class CapabilityStatus:
    """A capability's health, computed from in-memory state with no network call.

    Attributes:
        name: The capability name.
        state: The closed state vocabulary of `capabilities_status`.
        reason: Why the state is not `available`.
        checked_at: Aware UTC time of the check.
        task_dead: A supervised task this capability needs has died. The health probe reads
            it; no WS payload carries it.
    """

    name: str
    state: CapabilityState
    reason: str | None
    checked_at: datetime
    task_dead: bool = False


@dataclass(frozen=True, slots=True)
class RoutingHints:
    """What the router matches an utterance against for this capability."""

    keywords: tuple[str, ...]
    exemplars: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class InvocationContext:
    """Who asked for a call and how, passed to `Capability.invoke` and the policy.

    Attributes:
        session_id: The WebSocket session.
        turn_id: The turn that produced the call.
        utterance: The user's text for that turn.
        tool_call_id: Correlates the call's `tool_call_started` and `tool_call_completed`.
        source: Policy applies by source; an approved proposal skips it.
        now: Aware UTC time of the call.
    """

    session_id: str
    turn_id: str
    utterance: str
    tool_call_id: str
    source: CallSource
    now: datetime


@dataclass(frozen=True, slots=True)
class FastIntent:
    """A call the fast-path grammar matched, before validation.

    `args` are unvalidated: they go through the tool's `args_model` like a model call's, so a
    grammar cannot bypass the argument rules.
    """

    tool: str
    args: Mapping[str, object]


class Capability(abc.ABC):
    """A group of at most four tools behind one provider.

    The optional hooks below the abstract members have inert defaults, so a native capability
    implements only what it uses.
    """

    name: str
    description: str
    refine_allowance_tokens: int = 0
    """The most tokens `refine_schema` may add to this capability's catalog lines."""

    @abc.abstractmethod
    def tools(self) -> Sequence[ToolSpec]:
        """Return the capability's tools, at most four."""

    @abc.abstractmethod
    def routing_hints(self) -> RoutingHints:
        """Return the static keywords and exemplars the router matches."""

    def dynamic_keywords(self) -> tuple[str, ...]:
        """Return keywords learned at run time, such as light target names."""
        return ()

    @abc.abstractmethod
    async def start(self) -> None:
        """Start the capability; on failure raise `CapabilityError`, reported as `unavailable`."""

    @abc.abstractmethod
    async def stop(self) -> None:
        """Release the capability's resources."""

    @abc.abstractmethod
    def status(self) -> CapabilityStatus:
        """Return the current status from in-memory state, with no network call."""

    @abc.abstractmethod
    async def invoke(self, tool: str, args: BaseModel, ctx: InvocationContext) -> ToolOutcome:
        """Run a validated call that policy has allowed."""

    def refine_schema(self, tool: str, schema: dict[str, object]) -> dict[str, object]:
        """Return `schema` with live detail added, such as an enum of known light targets."""
        return schema

    def match_intent(self, text: str) -> FastIntent | None:
        """Match a whole `normalize_utterance` output against the fast-path grammar."""
        return None

    def escalate(self, tool: str, args: BaseModel) -> bool:
        """Return True when these arguments need confirmation whatever the policy.

        It can only raise a `model` or `fastpath` call to confirm, never lower a decision, and
        never applies to a call an approved proposal runs.
        """
        return False

    def active(self) -> bool:
        """Return True while an alarm or timer of this capability is ringing."""
        return False

    async def on_active_utterance(self, utterance: str, session_id: str) -> ToolOutcome | None:
        """Handle a stop or snooze phrase while `active()`; None when it is not one."""
        return None

    def provider_tools(self) -> tuple[str, ...]:
        """Return the provider's own tool names, such as Home Assistant's MCP tools."""
        return ()


@dataclass(frozen=True, slots=True)
class ActionDecode:
    """One constrained LLM request, built by the host and run by the handler.

    Attributes:
        messages: The compact decode prompt; shares nothing with the main chat prompt.
        response_schema: Sent as llama-server's JSON-schema grammar.
        max_tokens: 256 for a turn decode.
        temperature: 0.0, so the same prompt decodes the same way.
    """

    messages: tuple[dict[str, object], ...]
    response_schema: Mapping[str, object]
    max_tokens: int
    temperature: float


@dataclass(frozen=True, slots=True)
class DecodeReply:
    """The joined content of a decode and why it stopped (`"length"` means truncated)."""

    content: str
    finish_reason: str | None


DecodeFn = Callable[[ActionDecode], Awaitable[DecodeReply]]


@dataclass(frozen=True, slots=True)
class TurnPlan:
    """What the host routed for a turn.

    Attributes:
        route: The router's decision and its reasons.
        offered: Tool names the decode may choose.
        decode: The decode request; None when nothing is routed.
    """

    route: RouteDecision
    offered: frozenset[str]
    decode: ActionDecode | None


@dataclass(frozen=True, slots=True)
class ActionRecord:
    """One executed or proposed call, for the handler's tool-call events.

    `power_level` is the level the host read at the call. It defaults to None so a record built
    before the power port existed stays valid.
    """

    tool_call_id: str
    tool: str
    capability: str
    risk: RiskClass
    args_summary: str
    outcome: ToolOutcome
    duration_ms: int
    power_level: PowerLevel | None = None


@dataclass(frozen=True, slots=True)
class ActionResult:
    """What `run_actions` hands back to the handler.

    Attributes:
        records: Empty when the decode chose no call or failed.
        also_reply: Whether the main passes still run.
        reply: The templated reply, used when `also_reply` is false.
        block: The actions block for the main pass; empty when there is no call.
        notices: Appended after main-pass text.
        events: WS payloads for this turn.
    """

    records: tuple[ActionRecord, ...]
    also_reply: bool
    reply: str | None
    block: str
    notices: tuple[str, ...]
    events: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class PreTurnResolution:
    """A turn answered before routing: a ringing item, a proposal decision or the fast path.

    Attributes:
        handled: True when the turn is answered without the LLM.
        reply: The templated reply.
        remainder: Text after a leading "yes"; the handler runs it through `pre_llm` once more.
        records: The calls this resolution ran.
        events: WS payloads for this turn.
    """

    handled: bool
    reply: str | None
    remainder: str | None
    records: tuple[ActionRecord, ...]
    events: tuple[dict[str, object], ...]


class CapabilityTurnPort(Protocol):
    """The port a turn handler depends on, so it needs no import of the host implementation.

    `pre_llm` and `plan_turn` take `received_at`, the `time.monotonic()` at which the handler
    received the user message, so each call's `turn_ms` in the action log runs from there. None,
    the default, means the time of the call, so a handler that does not pass it stays valid.
    """

    async def pre_llm(
        self, session_id: str, turn_id: str, utterance: str, *, received_at: float | None = None
    ) -> PreTurnResolution | None:
        """Resolve the turn deterministically before routing, or return None."""
        ...

    def plan_turn(
        self,
        session_id: str,
        utterance: str,
        recent: Sequence[tuple[str, str]],
        *,
        received_at: float | None = None,
    ) -> TurnPlan:
        """Route the utterance and build the decode request."""
        ...

    async def run_actions(
        self, session_id: str, turn_id: str, utterance: str, plan: TurnPlan, decode: DecodeFn
    ) -> ActionResult:
        """Run the decode, then each chosen call through the invoke pipeline."""
        ...

    async def decide(
        self, session_id: str, proposal_id: str, decision: Literal["approve", "reject"]
    ) -> PreTurnResolution:
        """Resolve a live proposal from the client's `action_decision`."""
        ...

    def static_sentence(self) -> str:
        """Return the action-honesty sentence the static prompt gains."""
        ...

    def redact(self, text: str) -> str:
        """Replace every registered secret in `text`."""
        ...

    def record_turn(self, session_id: str, used: Iterable[str]) -> None:
        """Record the capabilities a turn used, for router stickiness."""
        ...

    def drain_events(self, session_id: str) -> list[dict[str, object]]:
        """Return and clear this session's in-turn events."""
        ...
