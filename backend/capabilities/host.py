"""The capability host: the one place a capability call is admitted, decided and run.

A turn handler depends on `CapabilityTurnPort` alone, and this class implements it. Every call,
whether the fast-path grammar, the action decode or an approved proposal chose it, goes through
the same invoke pipeline, so one validation step, one policy, one redaction pass and one action
log apply to all of them:

1. lookup of the tool;
2. the offered-this-turn check (a backstop; the decode grammar already limits names);
3. Pydantic validation, with one constrained repair decode on the model path;
4. the status gate;
5. policy by source: deny, run, or hold for the turn's one bundled proposal;
6. `capability.invoke` under the per-call timeout;
7. `CapabilityError` subclasses normalised to `error`, `unavailable` or `invalid_args`;
8. `redact()` over the summary and reason;
9. an action-log row, a stats update and an `ActionRecord`.

Confirmation is cross-turn. A turn holds `generation_lock` until it ends, so a "yes" can never
reach a turn that waits for it; a call that needs confirmation is stored as a proposal instead,
and a later utterance (`pre_llm`) or the client's `action_decision` (`decide`) resolves it. Both
can be in flight on the main loop at once. Each takes the proposal with `ProposalStore.pop`,
which is synchronous, so only the first to call it gets the proposal and resolves it.

The host never catches a bare `Exception`. The pipeline contains a `CapabilityError` from
`invoke` and the per-call timeout, and `on_active_utterance` is contained the same way: its
`CapabilityError` is normalised by step 7 and its overrun gives the timeout outcome, each
redacted and recorded. A decode contains `ActionDecodeError` and its timeout. Any other
exception from `invoke` or `on_active_utterance`, and any exception at all from `match_intent`
(a synchronous grammar over in-memory data, so it stays uncontained), reaches the host's
caller: inside a turn, the turn's own error backstop. A capability's `start` runs in a task of
its own, and a failure of any kind there makes that capability `unavailable`.

Clients learn of status changes from `capabilities_status`, which `status_changed` emits only
when the status set, the power level or `decode_available()` differs from the one last
broadcast. The host's own transition sites (a finished start, `stop`) and the MCP connection's
state listener call it at once; a watcher task polls it every second for changes with no
transition site, such as the alarm scheduler's health or the power level. The watcher contains
`MistError` per tick and backs off after a failing tick; any other end it was not asked for is
logged at ERROR and costs only the poll.

Power (`power.py`): the host reads the level into every record, log row and broadcast, and
decides nothing on it. Only the constrained decode needs a model, so `plan_turn` builds none
while `decode_available()` is False; every `pre_llm` step and every client action runs at
every level, off included.

Voice: each record carries its spec's `ack` and `view` and the `source` that ran it, which
`tool_events.py` turns into tool-event fields. The `ReplyRegister`'s address form ends the first
templated reply each session hears: a handled `pre_llm` or `decide` reply, or a `run_actions`
reply with `also_reply` false. The records keep the capability's own summaries.
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from backend.errors import (
    ActionDecodeError,
    CapabilityError,
    CapabilityInvocationError,
    CapabilityRegistrationError,
    CapabilityUnavailableError,
    McpTransportError,
    MistError,
    TimeParseError,
)
from backend.llm.models import ToolCall

from .action_log import ActionLog, ActionLogEntry
from .alarms.records import ReportOutcome, ReportResult
from .clock import Clock
from .config import CapabilitiesConfig
from .contracts import (
    ActionRecord,
    ActionResult,
    CallSource,
    Capability,
    CapabilityState,
    CapabilityStatus,
    CapabilityTurnPort,
    DecodeFn,
    InvocationContext,
    OutcomeStatus,
    PreTurnResolution,
    RiskClass,
    ToolOutcome,
    ToolSpec,
    TurnPlan,
)
from .decoder import ActionDecoder
from .events import (
    ActionProposedEvent,
    ActionResolvedEvent,
    AlarmReportMessage,
    AlarmStateEvent,
    CapabilitiesStatusEvent,
    CapabilityStatusEntry,
    EventSink,
    OutcomeEntry,
    ProposedCallEntry,
    ResolutionDecision,
)
from .intents import normalize_utterance
from .policy import PolicyTable
from .power import FixedPower, PowerLevel, PowerState, TierProviders
from .proposals import DecisionMatcher, Proposal, ProposalStore, ProposedCall
from .replies import (
    NoAddress,
    ReplyRegister,
    actions_block,
    notices_for,
    reply_for,
    with_address,
)
from .router import ToolRouter, count_catalog_tokens
from .schema import to_args_schema, to_catalog_line
from .secrets import SecretRegistry, redact

if TYPE_CHECKING:
    from backend.chat.context_budget import TokenCounter

logger = logging.getLogger(__name__)

T = TypeVar("T")

STATIC_SENTENCE = (
    "MIST changes devices, music or alarms only through its action system, and describes an"
    " action as done only when the actions block says so."
)
"""The action-honesty sentence the static prompt gains while a host is attached.

It is the same on every turn, so it stays in the cached prompt prefix. It covers a router miss,
where no decode ran but the model might still claim an action.
"""

DECODE_FAILURE_NOTICE = "(I could not work out which action you meant, so I did not run it.)"
"""The notice appended after the main-pass text when the action decode fails or times out.

It says only that the meant action did not run, not that nothing ran: a "yes, and ..." turn
may already have run the calls it approved.
"""

NO_LIVE_PROPOSAL_REPLY = "There is no live proposal to decide."
"""What `decide` answers when the session has no live proposal with that id."""

REJECTED_REPLY = "Okay, I won't do that."
"""The reply to a rejected proposal; none of its calls ran."""

_STATUSES: tuple[OutcomeStatus, ...] = (
    "ok",
    "invalid_args",
    "pending_confirmation",
    "unavailable",
    "error",
    "denied",
)
_SOURCES: tuple[CallSource, ...] = ("fastpath", "model", "proposal")
_RESOLUTIONS: tuple[ResolutionDecision, ...] = ("approved", "rejected", "expired", "superseded")
# A ringing item's stop or snooze is not a tool call, but it acts on the world, so it gets an
# `ActionRecord` and a log row under a name that says which hook ran.
_ACTIVE_HOOK_SUFFIX = ".on_active_utterance"
_RAW_ARGS_CHARS = 120

AlarmAction = Literal["stop", "snooze", "dismiss"]

ALARMS_NOT_BUILT_REASON = "alarms and timers are not in MIST_CAPABILITIES"
"""The `alarm_action` reason when the config builds neither alarms nor timers."""

ALARM_CONTROL_NOT_WIRED_REASON = "alarm control is not wired"
"""The `alarm_action` reason when alarms or timers is built but no `AlarmControl` was passed."""

# The capabilities whose status gates `alarm_action`, in order: timers stand in for alarms
# only when alarms is not built.
_ALARM_GATES = ("alarms", "timers")

# The watcher's poll interval, and the first and largest backoff after a failing tick.
_WATCH_INTERVAL_S = 1.0
_WATCH_BACKOFF_CAP_S = 60.0
# Bounds the doubling exponent so a long run of failures cannot overflow the float.
_WATCH_MAX_EXPONENT = 30
_WATCHER_NAME = "capability-status-watcher"

# Failures one watcher tick contains rather than dying on (Supervision). Anything else is a
# bug and ends the watcher, which `_on_watcher_done` records. No broader catch is added.
_WATCHER_CONTAINED_ERRORS: tuple[type[BaseException], ...] = (MistError,)

# Per capability, in name order, its name, state, reason and tool names.
_StatusSet = tuple[tuple[str, CapabilityState, str | None, tuple[str, ...]], ...]
# The compared part of a broadcast: the power level, whether a decode can run, and the status
# set. `at` and `checked_at` are left out, so a poll that finds only a newer check time emits
# nothing.
_Broadcast = tuple[PowerLevel, bool, _StatusSet]


class AlarmControl(Protocol):
    """The host's route to alarm state, for the connect snapshot and the client's alarm actions.

    The `Capability` contract carries no alarm state, so the factory passes one of these to
    the host for alarms and timers together, because `alarm_state` carries both kinds.
    """

    async def snapshot(self) -> dict[str, object]:
        """Return the `alarm_state` payload, its `rev` and alarms from one store read.

        Raises:
            CapabilityUnavailableError: The scheduler's health is not `running`.
            CapabilityStoreError: The alarm store cannot be read.
        """
        ...

    async def act(
        self, alarm_id: str, action: AlarmAction, minutes: int | None, *, endpoint: str
    ) -> ToolOutcome:
        """Stop or snooze a ringing alarm, or dismiss a missed one.

        `endpoint` is the endpoint of the connection the action arrived on, recorded as the
        actor; a stop or snooze from any endpoint ends or snoozes the alarm everywhere.

        Raises:
            CapabilityStoreError: The alarm store cannot be written.
            CapabilityUnavailableError: The scheduler is not running; the message is its
                health reason.
        """
        ...

    async def report(
        self,
        alarm_id: str,
        *,
        endpoint: str,
        outcome: ReportOutcome,
        at: datetime,
        rang_at: datetime,
        snoozed_until: datetime | None,
    ) -> ReportResult:
        """Apply an endpoint's `alarm_report` (`AlarmScheduler.report`).

        Raises:
            CapabilityStoreError: The alarm store cannot be read or written.
            CapabilityUnavailableError: The scheduler is not running; the message is its
                health reason.
        """
        ...


class TimeoutRunner(Protocol):
    """Runs an awaitable under a timeout, raising `TimeoutError` when it passes.

    `asyncio.wait_for` in production; tests inject a runner to expire a call deterministically.
    """

    def __call__(self, awaitable: Awaitable[T], timeout: float) -> Awaitable[T]:
        """Return an awaitable of `awaitable`'s result, bounded by `timeout` seconds."""
        ...


@dataclass(frozen=True, slots=True)
class HostSettings:
    """The host's per-call and per-turn settings, with the environment's defaults.

    Attributes:
        tool_timeout_s: Bound on one `capability.invoke` (`MIST_CAPABILITY_TOOL_TIMEOUT_S`).
        proposal_ttl_s: How long a proposal stays live (`MIST_CAPABILITY_PROPOSAL_TTL_S`).
        decode_timeout_s: Bound on each decode, the turn decode and each repair decode alike
            (`MIST_CAPABILITY_DECODE_TIMEOUT_S`).
        fast_path: Whether `pre_llm` asks capabilities' grammars (`MIST_CAPABILITY_FAST_PATH`).
        wake_words: Leading words normalisation strips (`MIST_CAPABILITY_WAKE_WORDS`).
    """

    tool_timeout_s: float = 8.0
    proposal_ttl_s: float = 60.0
    decode_timeout_s: float = 10.0
    fast_path: bool = True
    wake_words: tuple[str, ...] = ("mist", "hey mist", "ok mist", "okay mist")

    @classmethod
    def from_config(cls, config: CapabilitiesConfig) -> HostSettings:
        """Take the host's settings from the parsed environment."""
        return cls(
            tool_timeout_s=config.tool_timeout_s,
            proposal_ttl_s=config.proposal_ttl_s,
            decode_timeout_s=config.decode_timeout_s,
            fast_path=config.fast_path,
            wake_words=config.wake_words,
        )


@dataclass(frozen=True, slots=True)
class _Ready:
    """A call past lookup and validation: what the status gate and policy decide on."""

    capability: Capability
    spec: ToolSpec
    args: BaseModel
    summary: str


_Repair = Callable[
    [Capability, ToolSpec, Mapping[str, object], str], Awaitable[dict[str, object] | None]
]


class _Stats:
    """The in-memory counters `CapabilityHost.stats` returns, kept since the host was built."""

    def __init__(self, since: datetime) -> None:
        self.since = since
        self.invocations: dict[str, int] = dict.fromkeys(_STATUSES, 0)
        self.by_source: dict[str, int] = dict.fromkeys(_SOURCES, 0)
        self.decodes = 0
        self.decode_failures = 0
        self.repairs = 0
        self.repairs_ok = 0
        self.routed_no_call = 0
        self.proposals: dict[str, int] = dict.fromkeys(_RESOLUTIONS, 0)

    def snapshot(self) -> dict[str, object]:
        return {
            "since": self.since.isoformat(),
            "invocations": dict(self.invocations),
            "by_source": dict(self.by_source),
            "decodes": self.decodes,
            "decode_failures": self.decode_failures,
            "repairs": self.repairs,
            "repairs_ok": self.repairs_ok,
            "routed_no_call": self.routed_no_call,
            "proposals": dict(self.proposals),
        }


class CapabilityHost(CapabilityTurnPort):
    """Owns the capabilities and runs every call through the invoke pipeline (module docstring).

    All methods run on the event loop's thread. In-turn WS payloads (`action_proposed` and
    `action_resolved` produced by `pre_llm`, `run_actions` and `invoke`) are buffered per
    session and delivered only through `drain_events`, so the `events` of the resolutions and
    results those methods return are empty. `decide` runs outside any turn: it returns its
    payload in `PreTurnResolution.events` and never touches a session's buffer. Expiry goes to
    the event sink. The store reports it through `on_expire`, which is given no turn: it runs
    from the store's timer task, outside any turn, or synchronously inside a `pop` made by a
    turn or by `decide`. `capabilities_status` goes to the event sink too (module docstring).
    """

    def __init__(
        self,
        capabilities: Sequence[Capability],
        *,
        router: ToolRouter,
        counter: TokenCounter,
        policy: PolicyTable,
        decoder: ActionDecoder,
        secrets: SecretRegistry,
        action_log: ActionLog,
        event_sink: EventSink,
        clock: Clock,
        settings: HostSettings,
        timeout_runner: TimeoutRunner = asyncio.wait_for,
        alarm_control: AlarmControl | None = None,
        power: PowerState = FixedPower(),
        tiers: TierProviders | None = None,
        register: ReplyRegister = NoAddress(),
        turn_endpoint: str = "desktop",
    ) -> None:
        """Register the capabilities; nothing is started and no file is touched.

        Args:
            capabilities: In registration order, which `pre_llm` and `statuses` follow.
            router: Built over the same capabilities; it selects a turn's capabilities.
            counter: The router's token counter, for the refinement allowance check.
            policy: The effective confirmation mode of each tool.
            decoder: Builds and parses the action decodes.
            secrets: Every secret value, for `redact`.
            action_log: Where each invocation is recorded.
            event_sink: Receives each proposal expiry, which the store reports from its timer
                task or from inside `pop`, and each `capabilities_status` broadcast.
            clock: Supplies call times, proposal deadlines, the expiry timer and the status
                watcher's sleeps.
            settings: Timeouts, the proposal lifetime, the fast-path switch and wake words.
            timeout_runner: Bounds each invoke and decode; `asyncio.wait_for` by default.
            alarm_control: The alarm state and actions behind `snapshot_events` and
                `alarm_action`; None when no alarm scheduler is built.
            power: The level each record, log row and status broadcast carries; the host
                makes no decision on it. `FixedPower()` (full) by default.
            tiers: Whose `interaction` provider gates the decode (`decode_available`). None
                means the decode is not gated by a tier. Both defaults serve callers built
                before the power port; the factory always passes both.
            register: Supplies the address form that ends each session's first templated
                reply (`with_address`). `NoAddress()` by default, which leaves every reply
                as `replies.py` renders it.
            turn_endpoint: The endpoint every call's `InvocationContext.endpoint` names when
                its context names none: `desktop`, the endpoint of typed turns, until turns
                are routed per connection.

        Raises:
            CapabilityRegistrationError: Two capabilities share a name, or two offer the same
                tool name.
        """
        self._capabilities = tuple(capabilities)
        self._by_name: dict[str, Capability] = {}
        self._tools: dict[str, tuple[Capability, ToolSpec]] = {}
        for capability in self._capabilities:
            if capability.name in self._by_name:
                raise CapabilityRegistrationError(
                    f"capability '{capability.name}' is registered twice"
                )
            self._by_name[capability.name] = capability
            for spec in capability.tools():
                owner = self._tools.get(spec.name)
                if owner is not None:
                    raise CapabilityRegistrationError(
                        f"tool '{spec.name}' is offered by both '{owner[0].name}' and "
                        f"'{capability.name}'"
                    )
                self._tools[spec.name] = (capability, spec)
        self._router = router
        self._counter = counter
        self._policy = policy
        self._decoder = decoder
        self._secrets = secrets
        self._log = action_log
        self._sink = event_sink
        self._clock = clock
        self._settings = settings
        self._timeout = timeout_runner
        self._matcher = DecisionMatcher()
        self._proposals = ProposalStore(clock, settings.proposal_ttl_s, self._on_expire)
        self._locks: dict[str, asyncio.Lock] = {}
        self._events: dict[str, list[dict[str, object]]] = {}
        self._start_tasks: dict[str, asyncio.Task[None]] = {}
        # Per capability, the refined catalog lines last warned about, so an oversized catalog
        # warns once per change rather than on every turn.
        self._oversized: dict[str, tuple[str, ...]] = {}
        self._stats = _Stats(clock.now())
        self._alarm_control = alarm_control
        self._power = power
        self._tiers = tiers
        self._register = register
        self._turn_endpoint = turn_endpoint
        # Sessions that have had a templated reply, so only each one's first is addressed.
        self._addressed: set[str] = set()
        # Per session, the `received_at` of the turn `plan_turn` built a decode for, which its
        # `run_actions` takes, because `TurnPlan` has no field for it. A turn holds the voice
        # processor's `generation_lock` until it ends (module docstring), so no other turn's
        # `plan_turn` falls between a turn's two calls and one entry per session is enough.
        self._received_at: dict[str, float] = {}
        self._status_rev = 0
        # None until the first broadcast, so the first `status_changed` always emits.
        self._broadcast: _Broadcast | None = None
        self._watcher: asyncio.Task[None] | None = None
        self._watcher_stopping = False
        self._watcher_dead = False

    # -- Lifecycle -------------------------------------------------------------------------

    async def start(self) -> None:
        """Start every capability concurrently, and return without waiting for any.

        A start that raises `CapabilityError` makes that capability `unavailable` with the
        error's message. Any other exception is logged at ERROR with its traceback, and the
        capability is `unavailable` with the exception's class name. Each start that finishes
        is broadcast. The status watcher starts too; its first poll is a second later. Call it
        once.
        """
        loop = asyncio.get_running_loop()
        for capability in self._capabilities:
            task = loop.create_task(
                self._start_one(capability), name=f"capability-start-{capability.name}"
            )
            task.add_done_callback(functools.partial(self._on_start_done, capability.name))
            self._start_tasks[capability.name] = task
        if self._watcher is None:
            self._watcher = loop.create_task(self._watch_statuses(), name=_WATCHER_NAME)
            self._watcher.add_done_callback(self._on_watcher_done)

    async def wait_started(self) -> None:
        """Return once every start has finished, whether it succeeded or raised.

        It returns at once when `start` has not run or there are no capabilities. For tests:
        nothing in production awaits it.
        """
        if self._start_tasks:
            await asyncio.wait(list(self._start_tasks.values()))

    async def stop(self) -> None:
        """Stop the watcher, cancel pending starts, stop each capability, then close the store.

        A `CapabilityError` from one capability's `stop` is logged at WARNING and the others
        still stop. Once every capability has stopped, the stopped states are broadcast once;
        a start this cancelled is covered by that broadcast. The store closes in a `finally`,
        so it closes after the capabilities stop even when a `stop` raises outside
        `CapabilityError`, which still propagates. Closing drops the live proposals without
        announcing an expiry and waits for every expiry timer to end, so none outlives the
        host; a later `decide` finds no live proposal, and a call that policy holds is
        recorded as an error.
        """
        try:
            await self._stop_watcher()
            pending = [task for task in self._start_tasks.values() if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.wait(pending)
            for capability in self._capabilities:
                try:
                    await capability.stop()
                except CapabilityError as exc:
                    logger.warning("Capability %s failed to stop: %s", capability.name, exc)
            self.status_changed()
        finally:
            await self._proposals.close()

    def statuses(self) -> list[CapabilityStatus]:
        """Return each capability's status, in registration order.

        Each is the capability's own `status()`, with its reason redacted and `task_dead` passed
        through unchanged, except that a capability whose start failed reads `unavailable` with
        the start's reason.
        """
        return [self._status_of(capability) for capability in self._capabilities]

    def stats(self) -> dict[str, object]:
        """Return the counters since the host was built, in the `GET /capabilities` shape.

        `invocations` counts every pipeline record by outcome status and `by_source` by call
        source; a ringing item's stop or snooze counts under `fastpath`, the deterministic
        source. `decodes` counts turn decodes, not repairs; `decode_failures` those that failed
        or timed out; `repairs` each repair decode and `repairs_ok` those whose arguments then
        validated. `routed_no_call` counts decodes that chose no call. `proposals` counts each
        proposal's resolution. `status_watcher` is `running`, `stopped` (before `start` and
        after `stop`) or `dead` (it ended without `stop`, so only transition sites broadcast).
        """
        stats = self._stats.snapshot()
        stats["status_watcher"] = self._watcher_state()
        return stats

    # -- Status broadcast, connect snapshot and alarm actions ------------------------------

    def decode_available(self) -> bool:
        """Return whether a constrained decode can run now.

        It is `tiers.provider("interaction") is not None`, read at each call; with no tiers it
        is True. Which tiers a level allows is the router's policy, so the level itself is not
        consulted here.
        """
        return self._tiers is None or self._tiers.provider("interaction") is not None

    def status_changed(self) -> None:
        """Broadcast `capabilities_status` if the compared state differs from the last broadcast.

        The compared state is the power level, `decode_available()` and the status set, so a
        level change that only the watcher can see is broadcast at its next tick. This is the
        one entry point for every broadcast: the host's transition sites, the MCP connection's
        state listener and the watcher all call it, so one change is emitted once. It is
        synchronous and runs on the loop. On a difference it bumps `rev` and emits one payload
        through the event sink; otherwise it emits nothing.
        """
        self._broadcast_if_changed()

    def _broadcast_if_changed(self) -> _Broadcast:
        """Run `status_changed`'s comparison and broadcast, and return the compared state."""
        current = (self._power.level(), self.decode_available(), self._status_set())
        if current == self._broadcast:
            return current
        payload = _status_payload(self._status_rev + 1, current, self._clock.now())
        self._status_rev += 1
        self._broadcast = current
        self._sink.emit(payload)
        return current

    async def snapshot_events(self) -> list[dict[str, object]]:
        """Return the payloads a connecting client gets: status first, then alarm state.

        `status_changed` runs first, so the `capabilities_status` returned carries the current
        `rev` and no broadcast of a change can be newer than it. `alarm_state` follows when
        there is an alarm control. A `CapabilityError` from it is logged at WARNING and
        `alarm_state` left out, because `capabilities_status` already shows alarms and timers
        `unavailable` with the reason; any other exception propagates to the caller.
        """
        broadcast = self._broadcast_if_changed()
        events = [_status_payload(self._status_rev, broadcast, self._clock.now())]
        if self._alarm_control is not None:
            try:
                state = await self._alarm_control.snapshot()
            except CapabilityError as exc:
                logger.warning(
                    "Alarm state left out of the connect snapshot: %s",
                    self.redact(_describe(exc)),
                )
            else:
                events.append(AlarmStateEvent.model_validate(state).to_payload())
        return events

    async def alarm_action(
        self, alarm_id: str, action: AlarmAction, minutes: int | None, *, endpoint: str
    ) -> ToolOutcome:
        """Run a client's `stop`, `snooze` or `dismiss` on an alarm through the alarm control.

        `endpoint` is the endpoint of the connection the action arrived on, which `act`
        records as the actor. The status of alarms, or of timers when alarms is not built,
        gates it as step 4 of the pipeline gates a tool call: while it is `unavailable` or
        `disabled`, the answer is `unavailable` with the reason the status surface shows, and
        `act` is not called. A `CapabilityError` from `act` becomes the outcome step 7 gives
        it. The outcome is redacted. Any other exception from `act` propagates to the caller.

        No action-log row is written: the log's sources cover turn and proposal calls, and
        the alarm store records the transition itself.
        """
        logger.info("Alarm action %s for alarm %s from %s", action, alarm_id, endpoint)
        control = self._alarm_gate(f"Alarm action {action}", alarm_id)
        if isinstance(control, ToolOutcome):
            return control
        try:
            outcome = await control.act(alarm_id, action, minutes, endpoint=endpoint)
        except CapabilityError as exc:
            logger.warning(
                "Alarm action %s for alarm %s failed: %s",
                action,
                alarm_id,
                self.redact(_describe(exc)),
            )
            outcome = _outcome_for_error(f"alarm {action}", exc)
        return self._redacted(outcome)

    async def alarm_report(self, message: AlarmReportMessage, *, endpoint: str) -> ToolOutcome:
        """Apply an endpoint's `alarm_report` through the alarm control's `report`.

        `endpoint` is the endpoint of the connection the report arrived on; the message
        carries none. It is gated as `alarm_action` is. The report's times are checked again
        against the host's clock, since the message was validated on arrival: one outside the
        window answers `invalid_args` and `report` is not called. The `ReportResult` is the
        outcome's `data` `{"result": ...}`: `ok` for `applied`, `late`, `already_ended` and
        `duplicate`, and `invalid_args` for `not_due` and `unknown`. A `CapabilityError` from
        `report` becomes the outcome step 7 gives it; the outcome is redacted, and any other
        exception propagates. As for `alarm_action`, no action-log row is written.
        """
        alarm_id = message.alarm_id
        logger.info("Alarm report %s for alarm %s from %s", message.outcome, alarm_id, endpoint)
        control = self._alarm_gate("Alarm report", alarm_id)
        if isinstance(control, ToolOutcome):
            return control
        window_error = message.window_error(self._clock.now())
        if window_error is not None:
            return ToolOutcome(
                status="invalid_args",
                summary=f"Alarm report refused: {window_error}.",
                reason=window_error,
            )
        try:
            result = await control.report(
                alarm_id,
                endpoint=endpoint,
                outcome=message.outcome,
                at=message.at,
                rang_at=message.rang_at,
                snoozed_until=message.snoozed_until,
            )
        except CapabilityError as exc:
            logger.warning(
                "Alarm report for alarm %s failed: %s", alarm_id, self.redact(_describe(exc))
            )
            return self._redacted(_outcome_for_error("alarm report", exc))
        return self._redacted(_report_outcome(result))

    def _alarm_gate(self, what: str, alarm_id: str) -> AlarmControl | ToolOutcome:
        """Return the alarm control when the alarm status lets a client's request through.

        Otherwise return the `unavailable` outcome that answers it: alarms and timers not
        built, their gating status `unavailable` or `disabled`, or no alarm control passed.
        """
        statuses = {
            capability.name: status
            for capability, status in zip(self._capabilities, self.statuses())
        }
        gate = next((statuses[name] for name in _ALARM_GATES if name in statuses), None)
        if gate is None:
            return ToolOutcome(
                status="unavailable",
                summary="Alarms and timers are not in MIST_CAPABILITIES.",
                reason=ALARMS_NOT_BUILT_REASON,
            )
        if gate.state in ("unavailable", "disabled"):
            reason = gate.reason or gate.state
            return ToolOutcome(
                status="unavailable",
                summary=f"Alarms and timers are {gate.state}: {reason}.",
                reason=reason,
            )
        if self._alarm_control is None:
            logger.error(
                "%s for alarm %s cannot run: alarms or timers is built but no alarm control "
                "was passed to the host",
                what,
                alarm_id,
            )
            return ToolOutcome(
                status="unavailable",
                summary=f"Alarms and timers are unavailable: {ALARM_CONTROL_NOT_WIRED_REASON}.",
                reason=ALARM_CONTROL_NOT_WIRED_REASON,
            )
        return self._alarm_control

    # -- CapabilityTurnPort ----------------------------------------------------------------

    async def pre_llm(
        self, session_id: str, turn_id: str, utterance: str, *, received_at: float | None = None
    ) -> PreTurnResolution | None:
        """Resolve the turn without the model, in the Design doc's pre-LLM order.

        No step needs a model or reads the tiers, so all of them run at every power level,
        off included. `received_at` (`CapabilityTurnPort`) starts each record's `turn_ms`.

        1. While a capability is `active()`, it gets the normalised utterance through
           `on_active_utterance`, under the per-call timeout; an outcome answers the turn, and
           a live proposal is left as it is. A `CapabilityError` from the hook, or its overrun,
           answers with the outcome the pipeline gives it (module docstring).
        2. Otherwise a yes or a no to the session's live proposal pops and decides it. A
           leading yes followed by more ("yes, and dim them") returns that text as `remainder`.
        3. Otherwise a live proposal is superseded by the utterance, and the turn continues.
        4. Then the fast path: exactly one capability whose `match_intent` matches runs its call
           with source `fastpath`. None, or two or more, return None.

        A yes or no with no live proposal goes on to the fast path. A remainder's second pass
        starts again at step 1; the first pass popped the proposal, so steps 2 and 3 find
        nothing.
        """
        received = _received(received_at)
        text = normalize_utterance(utterance, self._settings.wake_words)
        ringing = await self._handle_active(session_id, turn_id, utterance, text, received)
        if ringing is not None:
            return self._addressed_resolution(session_id, ringing)

        match = self._matcher.match(text)
        if match is None:
            self._supersede_live(session_id)
        else:
            # `pop` is synchronous, so this turn and a concurrent `decide` cannot both get it.
            proposal = self._proposals.pop(session_id, None)
            if proposal is not None:
                decision, remainder = match
                async with self._lock(session_id):
                    records, payload, reply = await self._resolve(
                        proposal, decision, turn_id, utterance, received
                    )
                self._buffer(session_id, payload)
                return self._addressed_resolution(
                    session_id,
                    PreTurnResolution(
                        handled=True, reply=reply, remainder=remainder, records=records, events=()
                    ),
                )
        return self._addressed_resolution(
            session_id, await self._fast_path(session_id, turn_id, utterance, text, received)
        )

    def plan_turn(
        self,
        session_id: str,
        utterance: str,
        recent: Sequence[tuple[str, str]],
        *,
        received_at: float | None = None,
    ) -> TurnPlan:
        """Route the utterance and build the decode over the routed capabilities' tools.

        Each routed capability's schemas are refined through `refine_schema`, unless its
        refined catalog lines cost more tokens than its static lines plus its
        `refine_allowance_tokens`. That capability's unrefined schemas are then used for the
        turn, and one warning is logged per change of its refined catalog.

        Args:
            session_id: Whose stickiness the router applies.
            utterance: The text routing and the decode read.
            recent: Passed to `ActionDecoder.build` as given.
            received_at: Starts `turn_ms` for the records of this plan's `run_actions`
                (`CapabilityTurnPort`).

        Returns:
            A plan with no decode and nothing offered when nothing is routed, or when
            `decode_available()` is False; the route is still the router's decision.
        """
        received = _received(received_at)
        self._received_at.pop(session_id, None)
        route = self._router.select(utterance, session_id)
        if not route.capabilities or not self.decode_available():
            return TurnPlan(route=route, offered=frozenset(), decode=None)
        self._received_at[session_id] = received
        offered: list[tuple[ToolSpec, Mapping[str, object]]] = []
        for name in route.capabilities:
            capability = self._by_name[name]
            schemas = self._turn_schemas(capability)
            offered.extend((spec, schemas[spec.name]) for spec in capability.tools())
        return TurnPlan(
            route=route,
            offered=frozenset(spec.name for spec, _ in offered),
            decode=self._decoder.build(offered, recent, utterance),
        )

    async def run_actions(
        self, session_id: str, turn_id: str, utterance: str, plan: TurnPlan, decode: DecodeFn
    ) -> ActionResult:
        """Run the plan's decode, then each chosen call through the pipeline with source `model`.

        The decode is bounded by `decode_timeout_s`. A call whose arguments fail validation
        gets one repair decode, under the same bound. Calls that policy sends to confirm are
        bundled into one proposal, while the turn's allowed calls run, in the model's order.
        An `ActionDecodeError` or a decode timeout gives no records, `also_reply` true and
        `DECODE_FAILURE_NOTICE`, so the turn continues as it would without capabilities.

        Returns:
            The records, the decode's `also_reply`, and `reply`, `block` and `notices` from
            `replies.py`; with no records, `reply` is None and `block` empty. `events` is empty:
            in-turn payloads go to `drain_events`.
        """
        if plan.decode is None:
            return ActionResult((), True, None, "", (), ())
        # The `received_at` this session's last `plan_turn` stored; with none (the plan came
        # from elsewhere), `turn_ms` counts from here.
        received = self._received_at.pop(session_id, None)
        if received is None:
            received = time.monotonic()
        self._stats.decodes += 1
        try:
            reply = await self._timeout(decode(plan.decode), self._settings.decode_timeout_s)
            calls, also_reply = self._decoder.parse(reply, plan.offered)
        except (ActionDecodeError, TimeoutError) as exc:
            self._stats.decode_failures += 1
            logger.warning(
                "Capability action decode failed; no action runs this turn: %s",
                _describe(exc),
            )
            return ActionResult((), True, None, "", (DECODE_FAILURE_NOTICE,), ())
        if not calls:
            self._stats.routed_no_call += 1
            return ActionResult((), also_reply, None, "", (), ())

        async def repair(
            capability: Capability, spec: ToolSpec, arguments: Mapping[str, object], error: str
        ) -> dict[str, object] | None:
            return await self._repair(decode, capability, spec, arguments, error, utterance)

        now = self._clock.now()
        items = [
            (
                call,
                InvocationContext(
                    session_id, turn_id, utterance, call.id, "model", now, self._turn_endpoint
                ),
            )
            for call in calls
        ]
        async with self._lock(session_id):
            records = await self._run_calls(session_id, items, plan.offered, repair, received)
        reply = reply_for(records)
        if not also_reply:
            # Only a reply that is spoken in place of the main passes counts as the first.
            reply = self._addressed_reply(session_id, reply)
        return ActionResult(
            records=records,
            also_reply=also_reply,
            reply=reply,
            block=actions_block(records),
            notices=notices_for(records),
            events=(),
        )

    async def decide(
        self, session_id: str, proposal_id: str, decision: Literal["approve", "reject"]
    ) -> PreTurnResolution:
        """Resolve a live proposal from the client's `action_decision`.

        The proposal is popped before any await, so a racing spoken decision runs it at most
        once. A proposal of another session is never resolved.

        Returns:
            On a resolved proposal, `handled` true, the templated reply, the records of an
            approval and its `action_resolved` payload in `events`. Otherwise `handled` false
            and a reply saying why: `NO_LIVE_PROPOSAL_REPLY` when nothing matching is live (the
            id is another session's, already resolved or superseded), or the store's reason
            when the id has expired or is unknown.
        """
        try:
            proposal = self._proposals.pop(session_id, proposal_id)
        except CapabilityInvocationError as exc:
            return PreTurnResolution(False, self.redact(str(exc)), None, (), ())
        if proposal is None:
            return PreTurnResolution(False, NO_LIVE_PROPOSAL_REPLY, None, (), ())
        received = time.monotonic()
        async with self._lock(session_id):
            records, payload, reply = await self._resolve(
                proposal, decision, proposal.proposal_id, "", received
            )
        reply = self._addressed_reply(session_id, reply)
        return PreTurnResolution(True, reply, None, records, (payload,))

    def static_sentence(self) -> str:
        """Return `STATIC_SENTENCE`."""
        return STATIC_SENTENCE

    def redact(self, text: str) -> str:
        """Replace every registered secret in `text` with `***`."""
        return redact(text, self._secrets)

    def record_turn(self, session_id: str, used: Iterable[str]) -> None:
        """Record the capabilities a turn used, for the router's stickiness."""
        self._router.record_turn(session_id, used)

    def drain_events(self, session_id: str) -> list[dict[str, object]]:
        """Return and clear this session's buffered in-turn payloads, oldest first."""
        return self._events.pop(session_id, [])

    # -- The invoke pipeline ---------------------------------------------------------------

    async def invoke(
        self, tool_call: ToolCall, ctx: InvocationContext, offered: frozenset[str] | None
    ) -> ActionRecord:
        """Run one call through the whole pipeline, under the session's lock.

        Args:
            tool_call: The tool name, raw arguments and the id its record carries.
            ctx: The call's context; `ctx.source` selects the policy. Validation failures get
                no repair here: only `run_actions` holds a decode to repair with.
            offered: The tool names this turn offered, or None to skip the check.

        Returns:
            The record. A call that needs confirmation becomes a proposal of its own,
            superseding the session's live one, and its outcome is `pending_confirmation`.
            Its `turn_ms` counts from this call. A context that names no endpoint is given the
            host's turn endpoint.
        """
        received = time.monotonic()
        if ctx.endpoint is None:
            ctx = dataclasses.replace(ctx, endpoint=self._turn_endpoint)
        async with self._lock(ctx.session_id):
            (record,) = await self._run_calls(
                ctx.session_id, [(tool_call, ctx)], offered, None, received
            )
        return record

    async def _run_calls(
        self,
        session_id: str,
        items: Sequence[tuple[ToolCall, InvocationContext]],
        offered: frozenset[str] | None,
        repair: _Repair | None,
        received: float,
    ) -> tuple[ActionRecord, ...]:
        """Run stages 1-9 for each call in order; the caller holds the session's lock.

        `received` is the monotonic time each record's `turn_ms` counts from.
        """
        slots: list[ActionRecord | None] = []
        waiting: list[tuple[int, _Ready, InvocationContext]] = []
        for call, ctx in items:
            admitted = await self._admit(call, ctx, offered, repair, received)
            if isinstance(admitted, ActionRecord):
                slots.append(admitted)
                continue
            processed = await self._decide_and_run(admitted, ctx, received)
            if isinstance(processed, _Ready):
                waiting.append((len(slots), processed, ctx))
                slots.append(None)
            else:
                slots.append(processed)
        if waiting:
            await self._propose(session_id, waiting, slots, received)
        return tuple(record for record in slots if record is not None)

    async def _admit(
        self,
        call: ToolCall,
        ctx: InvocationContext,
        offered: frozenset[str] | None,
        repair: _Repair | None,
        received: float,
    ) -> _Ready | ActionRecord:
        """Stages 1-3: lookup, the offered check and validation, with one repair if given."""
        arguments: dict[str, object] = dict(call.arguments)
        found = self._tools.get(call.name)
        if found is None:
            outcome = ToolOutcome(
                status="invalid_args",
                summary=f"There is no action called {call.name}.",
                reason="unknown tool",
            )
            # Nothing ran and no capability owns the name, so the record names the prefix.
            return await self._record(
                ctx,
                call.id,
                call.name.partition(".")[0],
                "read",
                call.name,
                arguments,
                outcome,
                received=received,
            )
        capability, spec = found
        if offered is not None and call.name not in offered:
            outcome = ToolOutcome(
                status="invalid_args",
                summary=f"The action {call.name} was not offered this turn.",
                reason="not offered",
            )
            return await self._record(
                ctx,
                call.id,
                capability.name,
                spec.risk,
                call.name,
                arguments,
                outcome,
                received=received,
                spec=spec,
            )

        args, error = _validate(spec, arguments)
        if args is None and repair is not None:
            repaired = await repair(capability, spec, arguments, error)
            if repaired is not None:
                arguments = repaired
                args, error = _validate(spec, arguments)
                if args is not None:
                    self._stats.repairs_ok += 1
        if args is None:
            outcome = ToolOutcome(
                status="invalid_args",
                summary=f"Invalid details for {spec.name}: {error}.",
                reason=error,
            )
            return await self._record(
                ctx,
                call.id,
                capability.name,
                spec.risk,
                call.name,
                arguments,
                outcome,
                received=received,
                spec=spec,
            )
        return _Ready(capability, spec, args, self.redact(spec.summarize(args)))

    async def _decide_and_run(
        self, ready: _Ready, ctx: InvocationContext, received: float
    ) -> ActionRecord | _Ready:
        """Stages 4-5, then 6-9: the status gate, policy, and the call itself.

        Returns the `_Ready` call unchanged when policy holds it for confirmation.
        """
        status = self._status_of(ready.capability)
        if status.state in ("unavailable", "disabled"):
            outcome = ToolOutcome(
                status="unavailable",
                summary=f"{ready.capability.name} is {status.state}.",
                reason=status.reason or status.state,
            )
            return await self._finish(ready, ctx, outcome, 0, received)
        decision = self._policy.decide(ready.spec, ready.args, ctx, ready.capability)
        if decision == "deny":
            outcome = ToolOutcome(
                status="denied", summary="That action is turned off.", reason="denied by policy"
            )
            return await self._finish(ready, ctx, outcome, 0, received)
        if decision == "confirm":
            return ready
        return await self._execute(ready, ctx, received)

    async def _execute(
        self, ready: _Ready, ctx: InvocationContext, received: float
    ) -> ActionRecord:
        """Stages 6-9: invoke under the timeout and normalise a `CapabilityError`."""
        tool = ready.spec.name
        timeout = self._settings.tool_timeout_s
        started = time.monotonic()
        try:
            outcome = await self._timeout(ready.capability.invoke(tool, ready.args, ctx), timeout)
        except TimeoutError:
            logger.warning("Capability call %s timed out after %g s", tool, timeout)
            outcome = _timeout_outcome(tool, timeout)
        except CapabilityError as exc:
            logger.warning("Capability call %s failed: %s", tool, _describe(exc))
            outcome = _outcome_for_error(tool, exc)
        duration_ms = int((time.monotonic() - started) * 1000)
        return await self._finish(ready, ctx, outcome, duration_ms, received)

    async def _finish(
        self,
        ready: _Ready,
        ctx: InvocationContext,
        outcome: ToolOutcome,
        duration_ms: int,
        received: float,
    ) -> ActionRecord:
        return await self._record(
            ctx,
            ctx.tool_call_id,
            ready.capability.name,
            ready.spec.risk,
            ready.spec.name,
            ready.summary,
            outcome,
            duration_ms,
            received=received,
            spec=ready.spec,
        )

    async def _record(
        self,
        ctx: InvocationContext,
        tool_call_id: str,
        capability: str,
        risk: RiskClass,
        tool: str,
        args: str | Mapping[str, object],
        outcome: ToolOutcome,
        duration_ms: int = 0,
        *,
        received: float,
        spec: ToolSpec | None = None,
    ) -> ActionRecord:
        """Stages 8-9: redact, count, log, and build the record.

        `args` is the call's summary, or its raw arguments when they never validated, rendered
        as compact JSON and capped. The power level is read here, at the outcome, and
        `turn_ms` runs from `received` (a `time.monotonic()` value) to now. The record's `ack`
        and `view` come from `spec`, None when no spec ran the call, and its `source` from
        `ctx`, the path that ran it.
        """
        args_summary = self.redact(args if isinstance(args, str) else _raw_args(args))
        outcome = self._redacted(outcome)
        power_level = self._power.level()
        turn_ms = max(0, int((time.monotonic() - received) * 1000))
        record = ActionRecord(
            tool_call_id=tool_call_id,
            tool=tool,
            capability=capability,
            risk=risk,
            args_summary=args_summary,
            outcome=outcome,
            duration_ms=duration_ms,
            power_level=power_level,
            ack=None if spec is None else spec.ack,
            view=None if spec is None else spec.view,
            source=ctx.source,
        )
        self._stats.invocations[outcome.status] += 1
        self._stats.by_source[ctx.source] += 1
        await self._log.record(
            ActionLogEntry(
                at=ctx.now,
                session_id=ctx.session_id,
                turn_id=ctx.turn_id,
                tool_call_id=tool_call_id,
                tool=tool,
                capability=capability,
                source=ctx.source,
                status=outcome.status,
                args_summary=args_summary,
                summary=outcome.summary,
                reason=outcome.reason,
                proposal_id=outcome.proposal_id,
                duration_ms=duration_ms,
                power_level=power_level,
                turn_ms=turn_ms,
            )
        )
        return record

    async def _repair(
        self,
        decode: DecodeFn,
        capability: Capability,
        spec: ToolSpec,
        arguments: Mapping[str, object],
        error: str,
        utterance: str,
    ) -> dict[str, object] | None:
        """Run the one repair decode for a call; None when it fails or times out."""
        self._stats.repairs += 1
        schema = self._turn_schemas(capability)[spec.name]
        request = self._decoder.build_repair(spec, schema, arguments, error, utterance)
        try:
            reply = await self._timeout(decode(request), self._settings.decode_timeout_s)
            return self._decoder.parse_repair(reply)
        except (ActionDecodeError, TimeoutError) as exc:
            logger.warning("Repair decode for %s failed: %s", spec.name, _describe(exc))
            return None

    # -- Proposals -------------------------------------------------------------------------

    async def _propose(
        self,
        session_id: str,
        waiting: Sequence[tuple[int, _Ready, InvocationContext]],
        slots: list[ActionRecord | None],
        received: float,
    ) -> None:
        """Bundle a turn's held calls into one proposal and fill their pending records.

        Once `stop` has closed the store, `put` refuses: each held call is then recorded with
        the outcome `_outcome_for_error` gives that refusal, and no `action_proposed` is
        buffered.
        """
        self._supersede_live(session_id)
        try:
            proposal = self._proposals.put(
                session_id,
                [
                    ProposedCall(
                        capability=ready.capability.name,
                        tool=ready.spec.name,
                        args=ready.args,
                        summary=ready.summary,
                        risk=ready.spec.risk,
                    )
                    for _, ready, _ in waiting
                ],
            )
        except CapabilityInvocationError as exc:
            logger.warning("Capability calls could not be proposed: %s", _describe(exc))
            for index, ready, ctx in waiting:
                outcome = _outcome_for_error(ready.spec.name, exc)
                slots[index] = await self._finish(ready, ctx, outcome, 0, received)
            return
        self._buffer(session_id, _proposed_payload(proposal))
        for index, ready, ctx in waiting:
            outcome = ToolOutcome(
                status="pending_confirmation",
                summary=ready.summary,
                proposal_id=proposal.proposal_id,
            )
            slots[index] = await self._finish(ready, ctx, outcome, 0, received)

    async def _resolve(
        self,
        proposal: Proposal,
        decision: Literal["approve", "reject"],
        turn_id: str,
        utterance: str,
        received: float,
    ) -> tuple[tuple[ActionRecord, ...], dict[str, object], str]:
        """Reject a popped proposal, or run each of its calls in order with source `proposal`.

        Returns:
            The records (none for a rejection), the `action_resolved` payload and the reply.
        """
        if decision == "reject":
            self._stats.proposals["rejected"] += 1
            return (), _resolved_payload(proposal, "rejected", ()), REJECTED_REPLY
        records: list[ActionRecord] = []
        for call in proposal.calls:
            capability, spec = self._tools[call.tool]
            ctx = InvocationContext(
                session_id=proposal.session_id,
                turn_id=turn_id,
                utterance=utterance,
                tool_call_id=str(uuid.uuid4()),
                source="proposal",
                now=self._clock.now(),
                endpoint=self._turn_endpoint,
            )
            processed = await self._decide_and_run(
                _Ready(capability, spec, call.args, call.summary), ctx, received
            )
            if isinstance(processed, _Ready):
                # `PolicyTable.decide` never confirms source `proposal`; re-proposing an
                # approved call would stop it from ever running.
                raise RuntimeError(f"policy held approved call {call.tool} for confirmation")
            records.append(processed)
        self._stats.proposals["approved"] += 1
        result = tuple(records)
        return result, _resolved_payload(proposal, "approved", result), reply_for(result)

    def _supersede_live(self, session_id: str) -> None:
        proposal = self._proposals.pop(session_id, None)
        if proposal is None:
            return
        self._stats.proposals["superseded"] += 1
        self._buffer(session_id, _resolved_payload(proposal, "superseded", ()))

    def _on_expire(self, proposal: Proposal) -> None:
        # Called by the store from its timer task or inside `pop`; it must not await.
        self._stats.proposals["expired"] += 1
        self._sink.emit(_resolved_payload(proposal, "expired", ()))

    # -- Pre-LLM steps ---------------------------------------------------------------------

    async def _handle_active(
        self, session_id: str, turn_id: str, utterance: str, text: str, received: float
    ) -> PreTurnResolution | None:
        timeout = self._settings.tool_timeout_s
        for capability in self._capabilities:
            if not capability.active():
                continue
            tool = f"{capability.name}{_ACTIVE_HOOK_SUFFIX}"
            started = time.monotonic()
            # Contained as `_execute` contains `invoke`: a stop that fails or hangs still acts
            # on the world as far as the user can tell, so it gets an outcome and a record.
            try:
                outcome = await self._timeout(
                    capability.on_active_utterance(text, session_id), timeout
                )
            except TimeoutError:
                logger.warning("Capability hook %s timed out after %g s", tool, timeout)
                outcome = _timeout_outcome(tool, timeout)
            except CapabilityError as exc:
                logger.warning("Capability hook %s failed: %s", tool, _describe(exc))
                outcome = _outcome_for_error(tool, exc)
            if outcome is None:
                continue
            duration_ms = int((time.monotonic() - started) * 1000)
            ctx = InvocationContext(
                session_id=session_id,
                turn_id=turn_id,
                utterance=utterance,
                tool_call_id=str(uuid.uuid4()),
                source="fastpath",
                now=self._clock.now(),
                endpoint=self._turn_endpoint,
            )
            record = await self._record(
                ctx,
                ctx.tool_call_id,
                capability.name,
                "reversible",
                tool,
                text,
                outcome,
                duration_ms,
                received=received,
            )
            return PreTurnResolution(True, reply_for([record]), None, (record,), ())
        return None

    async def _fast_path(
        self, session_id: str, turn_id: str, utterance: str, text: str, received: float
    ) -> PreTurnResolution | None:
        if not self._settings.fast_path:
            return None
        intents = [
            intent
            for capability in self._capabilities
            if (intent := capability.match_intent(text)) is not None
        ]
        if len(intents) != 1:
            return None
        (intent,) = intents
        call = ToolCall(id=str(uuid.uuid4()), name=intent.tool, arguments=dict(intent.args))
        ctx = InvocationContext(
            session_id=session_id,
            turn_id=turn_id,
            utterance=utterance,
            tool_call_id=call.id,
            source="fastpath",
            now=self._clock.now(),
            endpoint=self._turn_endpoint,
        )
        async with self._lock(session_id):
            records = await self._run_calls(session_id, [(call, ctx)], None, None, received)
        return PreTurnResolution(True, reply_for(records), None, records, ())

    # -- Helpers ---------------------------------------------------------------------------

    def _addressed_reply(self, session_id: str, reply: str) -> str:
        """Give a templated reply the address form when it is the session's first.

        A blank reply is not a reply, so it neither gains the form nor uses up the first.
        """
        if not reply.strip():
            return reply
        first = session_id not in self._addressed
        self._addressed.add(session_id)
        return with_address(reply, self._register, first_in_session=first)

    def _addressed_resolution(
        self, session_id: str, resolution: PreTurnResolution | None
    ) -> PreTurnResolution | None:
        """`_addressed_reply` over a handled resolution's reply; anything else unchanged."""
        if resolution is None or not resolution.handled or resolution.reply is None:
            return resolution
        return dataclasses.replace(
            resolution, reply=self._addressed_reply(session_id, resolution.reply)
        )

    def _status_of(self, capability: Capability) -> CapabilityStatus:
        status = capability.status()
        failure = self._start_failure(capability.name)
        if failure is not None:
            return dataclasses.replace(status, state="unavailable", reason=failure)
        if status.reason is None:
            return status
        return dataclasses.replace(status, reason=self.redact(status.reason))

    def _start_failure(self, name: str) -> str | None:
        """Read a finished start's failure from its task, so no callback ordering matters."""
        task = self._start_tasks.get(name)
        if task is None or not task.done():
            return None
        if task.cancelled():
            return "start was cancelled"
        error = task.exception()
        if error is None:
            return None
        if isinstance(error, CapabilityError):
            return self.redact(str(error) or type(error).__name__)
        return type(error).__name__

    async def _start_one(self, capability: Capability) -> None:
        await capability.start()

    def _on_start_done(self, name: str, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            # Only `stop()` cancels a start, and it broadcasts once every capability stops.
            return
        error = task.exception()
        if isinstance(error, CapabilityError):
            logger.warning("Capability %s failed to start: %s", name, error)
        elif error is not None:
            logger.error(
                "Capability %s start raised an unexpected %s",
                name,
                type(error).__name__,
                exc_info=(type(error), error, error.__traceback__),
            )
        self.status_changed()

    def _redacted(self, outcome: ToolOutcome) -> ToolOutcome:
        """Stage 8: the outcome with its summary and reason through `redact()`."""
        return dataclasses.replace(
            outcome,
            summary=self.redact(outcome.summary),
            reason=None if outcome.reason is None else self.redact(outcome.reason),
        )

    def _status_set(self) -> _StatusSet:
        """Read `statuses()` into the compared form, keyed by capability name."""
        rows = [
            (
                capability.name,
                status.state,
                status.reason,
                tuple(spec.name for spec in capability.tools()),
            )
            for capability, status in zip(self._capabilities, self.statuses())
        ]
        return tuple(sorted(rows, key=lambda row: row[0]))

    # -- The status watcher ----------------------------------------------------------------

    async def _watch_statuses(self) -> None:
        """Poll `status_changed` every second, backing off after a failing tick."""
        failures = 0
        while not self._watcher_stopping:
            if failures:
                exponent = min(failures - 1, _WATCH_MAX_EXPONENT)
                delay = min(_WATCH_BACKOFF_CAP_S, _WATCH_INTERVAL_S * 2.0**exponent)
            else:
                delay = _WATCH_INTERVAL_S
            # Outside the contained `try`: a sleep's own error is not a failed tick.
            await self._clock.sleep(delay)
            try:
                self.status_changed()
            except _WATCHER_CONTAINED_ERRORS as exc:
                failures += 1
                logger.warning(
                    "Capability status poll failed (%d in a row): %s",
                    failures,
                    self.redact(_describe(exc)),
                )
            else:
                failures = 0

    async def _stop_watcher(self) -> None:
        # The flag is set before the cancel, so the done-callback reads the end as orderly.
        self._watcher_stopping = True
        watcher = self._watcher
        if watcher is not None and not watcher.done():
            watcher.cancel()
            await asyncio.wait({watcher})

    def _on_watcher_done(self, task: asyncio.Task[None]) -> None:
        """Record how the watcher ended, keyed on the stopping flag rather than cancellation.

        A cancellation `stop()` did not ask for is a death like an exception or a return, so
        a watcher that stops polling never does so silently.
        """
        if self._watcher_stopping:
            if not task.cancelled():
                exc = task.exception()
                if exc is not None:
                    logger.warning(
                        "Capability status watcher raised while the host stopped: %s",
                        self.redact(_describe(exc)),
                        exc_info=exc,
                    )
            return
        self._watcher_dead = True
        consequence = "status changes now reach clients only from transition sites"
        if task.cancelled():
            logger.error("Capability status watcher was cancelled without stop(); %s", consequence)
            return
        exc = task.exception()
        if exc is None:
            logger.error(
                "Capability status watcher returned, which it never does by design; %s",
                consequence,
            )
            return
        logger.error(
            "Capability status watcher died: %s: %s; %s",
            type(exc).__name__,
            self.redact(str(exc)),
            consequence,
            exc_info=exc,
        )

    def _watcher_state(self) -> str:
        if self._watcher_dead:
            return "dead"
        if self._watcher is not None and not self._watcher.done():
            return "running"
        return "stopped"

    def _turn_schemas(self, capability: Capability) -> dict[str, dict[str, object]]:
        """Return the capability's arguments schemas for this turn, refined when they fit."""
        specs = list(capability.tools())
        static = {spec.name: to_args_schema(spec) for spec in specs}
        try:
            refined = {spec.name: to_args_schema(spec, capability.refine_schema) for spec in specs}
        except CapabilityRegistrationError as exc:
            self._warn_once(
                capability.name,
                ("refine error", str(exc)),
                "Capability %s refinement is invalid (%s); using its unrefined schemas",
                capability.name,
                exc,
            )
            return static
        static_lines = [to_catalog_line(spec, static[spec.name]) for spec in specs]
        refined_lines = tuple(to_catalog_line(spec, refined[spec.name]) for spec in specs)
        limit = (
            count_catalog_tokens(static_lines, self._counter) + capability.refine_allowance_tokens
        )
        measured = count_catalog_tokens(refined_lines, self._counter)
        if measured <= limit:
            self._oversized.pop(capability.name, None)
            return refined
        self._warn_once(
            capability.name,
            refined_lines,
            "Capability %s refined catalog needs %d tokens, over its %d (static lines plus "
            "refine allowance); using its unrefined schemas",
            capability.name,
            measured,
            limit,
        )
        return static

    def _warn_once(self, name: str, key: tuple[str, ...], message: str, *args: object) -> None:
        if self._oversized.get(name) == key:
            return
        self._oversized[name] = key
        logger.warning(message, *args)

    def _lock(self, session_id: str) -> asyncio.Lock:
        return self._locks.setdefault(session_id, asyncio.Lock())

    def _buffer(self, session_id: str, payload: dict[str, object]) -> None:
        self._events.setdefault(session_id, []).append(payload)


def _validate(spec: ToolSpec, arguments: Mapping[str, object]) -> tuple[BaseModel | None, str]:
    try:
        return spec.args_model.model_validate(dict(arguments)), ""
    except ValidationError as exc:
        return None, _compact_error(exc)


def _compact_error(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "args"
        parts.append(f"{location}: {error['msg']}")
    return "; ".join(parts)


def _outcome_for_error(tool: str, exc: CapabilityError) -> ToolOutcome:
    """Stage 7: map a capability's failure to the outcome vocabulary.

    The provider cannot be reached (`CapabilityUnavailableError`, `McpTransportError`):
    `unavailable`. A spoken time that does not parse (`TimeParseError`): `invalid_args`. Any
    other `CapabilityError`: `error`.
    """
    reason = str(exc) or type(exc).__name__
    if isinstance(exc, CapabilityUnavailableError | McpTransportError):
        return ToolOutcome(status="unavailable", summary=f"{tool} is unavailable.", reason=reason)
    if isinstance(exc, TimeParseError):
        return ToolOutcome(
            status="invalid_args", summary=f"Invalid details for {tool}: {reason}.", reason=reason
        )
    return ToolOutcome(status="error", summary=f"{tool} failed.", reason=reason)


_REPORT_SUMMARIES: dict[ReportResult, tuple[OutcomeStatus, str, str | None]] = {
    "applied": ("ok", "Alarm report applied.", None),
    "late": ("ok", "Alarm report recorded: the ring started after the grace window.", None),
    "already_ended": ("ok", "Alarm report recorded: the alarm had already ended.", None),
    "duplicate": ("ok", "Alarm report already received.", None),
    "not_due": ("invalid_args", "Alarm report refused: the alarm is not due yet.", "not due"),
    "unknown": ("invalid_args", "Alarm report refused: no such alarm.", "unknown alarm"),
}


def _report_outcome(result: ReportResult) -> ToolOutcome:
    """The outcome of an `alarm_report` the records answered, with the result in its data."""
    status, summary, reason = _REPORT_SUMMARIES[result]
    return ToolOutcome(status=status, summary=summary, data={"result": result}, reason=reason)


def _timeout_outcome(tool: str, timeout: float) -> ToolOutcome:
    """Stage 6's outcome for a call or an active-item hook that overran the per-call timeout."""
    return ToolOutcome(
        status="error", summary=f"{tool} took too long.", reason=f"timed out after {timeout:g} s"
    )


def _raw_args(arguments: Mapping[str, object]) -> str:
    text = json.dumps(
        dict(arguments), sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )
    return text if len(text) <= _RAW_ARGS_CHARS else f"{text[: _RAW_ARGS_CHARS - 3]}..."


def _received(received_at: float | None) -> float:
    """Return the handler's `received_at`, or the time of the call when it passed none."""
    return time.monotonic() if received_at is None else received_at


def _describe(exc: BaseException) -> str:
    return str(exc) or type(exc).__name__


def _status_payload(rev: int, broadcast: _Broadcast, at: datetime) -> dict[str, object]:
    power_level, decode_available, statuses = broadcast
    return CapabilitiesStatusEvent(
        type="capabilities_status",
        rev=rev,
        capabilities=[
            CapabilityStatusEntry(name=name, state=state, reason=reason, tools=list(tools))
            for name, state, reason, tools in statuses
        ],
        power_level=power_level,
        decode_available=decode_available,
        at=at,
    ).to_payload()


def _proposed_payload(proposal: Proposal) -> dict[str, object]:
    return ActionProposedEvent(
        type="action_proposed",
        proposal_id=proposal.proposal_id,
        session_id=proposal.session_id,
        summary=proposal.summary,
        calls=[
            ProposedCallEntry(
                capability=call.capability, tool=call.tool, summary=call.summary, risk=call.risk
            )
            for call in proposal.calls
        ],
        expires_at=proposal.expires_at,
    ).to_payload()


def _resolved_payload(
    proposal: Proposal, decision: ResolutionDecision, records: Sequence[ActionRecord]
) -> dict[str, object]:
    return ActionResolvedEvent(
        type="action_resolved",
        proposal_id=proposal.proposal_id,
        session_id=proposal.session_id,
        decision=decision,
        outcomes=[
            OutcomeEntry(tool=r.tool, status=r.outcome.status, summary=r.outcome.summary)
            for r in records
        ],
    ).to_payload()
