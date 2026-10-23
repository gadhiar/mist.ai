"""Shared fakes for the capability-layer unit tests.

Every later capability task builds its tests on these, so each fake exposes its
scripting surface as plain public attributes that a test sets directly, and
records what it received so a test can assert on forwarding.
"""

from __future__ import annotations

import asyncio
import contextlib
import heapq
import itertools
import threading
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel, ConfigDict

from backend.capabilities.contracts import (
    ActionDecode,
    Capability,
    CapabilityState,
    CapabilityStatus,
    ConfirmPolicy,
    DecodeReply,
    FastIntent,
    InvocationContext,
    RiskClass,
    RoutingHints,
    ToolOutcome,
    ToolSpec,
)

DEFAULT_START = datetime(2026, 1, 5, 7, 0, tzinfo=UTC)
"""A fixed Monday morning, so tests never depend on the real date."""

# Loop iterations `FakeClock.advance` runs after moving time to its target, so tasks
# that were not sleeping on the clock see the final time before the call returns.
_SETTLE_YIELDS = 10


class EmptyArgs(BaseModel):
    """Arguments model for a tool that takes none."""

    model_config = ConfigDict(extra="forbid")


def make_tool_spec(
    name: str,
    args_model: type[BaseModel] = EmptyArgs,
    *,
    description: str = "a fake tool",
    risk: RiskClass = "reversible",
    default_policy: ConfirmPolicy = "allow",
    summarize: Callable[[BaseModel], str] | None = None,
) -> ToolSpec:
    """Build a `ToolSpec` with test defaults, so a test names only what it varies."""
    return ToolSpec(
        name=name,
        description=description,
        args_model=args_model,
        risk=risk,
        default_policy=default_policy,
        summarize=summarize or (lambda args: name),
    )


class FakeClock:
    """A `Clock` whose time moves only when a test calls `advance`.

    Sleepers wake in deadline order (ties in the order they slept), and each one
    sees `now()` equal to its own deadline, so a scheduler under test observes the
    same sequence of instants it would on a real clock.

    Contract of `advance(seconds)`:

    - It wakes due sleepers one at a time, earliest deadline first, setting `now()`
      to each deadline before waking it.
    - After each wake it waits until the woken task has either called `sleep` again
      with a positive duration or finished. Awaits in between, including worker
      threads such as `CapabilitiesDb.run`, therefore complete before time moves on,
      and a new deadline that falls inside the same advance is woken by that call.
    - The wait is bounded by `settle_timeout_s` of real time. At the bound it stops
      waiting, appends the deadline to `stalled` and carries on; a test that expects
      every woken task to settle asserts `stalled == []`.
    - It then sets `now()` to the target and runs a few loop iterations.

    The task that matters is the one that calls `sleep`. Code that sleeps in a task
    of its own (for example `asyncio.wait` of a sleep task against a wake event) has
    that inner task finish on wake, so its parent is not waited for; such a test
    calls `wait_for_sleepers` after `advance` to wait for the parent's next sleep.
    """

    def __init__(self, start: datetime = DEFAULT_START, *, settle_timeout_s: float = 2.0) -> None:
        if start.tzinfo is None:
            raise ValueError("FakeClock needs an aware start time")
        self._now = start.astimezone(UTC)
        self._sequence = itertools.count()
        self._sleepers: list[
            tuple[datetime, int, asyncio.Future[None], asyncio.Task[object] | None]
        ] = []
        self._watchers: dict[asyncio.Task[object], asyncio.Future[None]] = {}
        self._slept = asyncio.Event()
        self.settle_timeout_s = settle_timeout_s
        self.sleeps: list[float] = []
        """Every duration passed to `sleep`, in call order."""
        self.stalled: list[datetime] = []
        """Deadlines whose woken task neither slept again nor finished within the bound."""

    def now(self) -> datetime:
        return self._now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        if seconds <= 0:
            await asyncio.sleep(0)
            return
        task = asyncio.current_task()
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        deadline = self._now + timedelta(seconds=seconds)
        heapq.heappush(self._sleepers, (deadline, next(self._sequence), future, task))
        if task is not None:
            watcher = self._watchers.pop(task, None)
            if watcher is not None and not watcher.done():
                watcher.set_result(None)
        self._slept.set()
        await future

    @property
    def pending(self) -> int:
        """The number of sleepers still waiting (cancelled ones excluded)."""
        return sum(1 for _, _, future, _ in self._sleepers if not future.done())

    async def advance(self, seconds: float) -> None:
        """Move time forward by `seconds`, waking each due sleeper in order (class docstring)."""
        target = self._now + timedelta(seconds=seconds)
        while self._sleepers and self._sleepers[0][0] <= target:
            deadline, _, future, task = heapq.heappop(self._sleepers)
            if future.done():
                continue
            self._now = deadline
            if task is None:
                future.set_result(None)
                await self._settle()
                continue
            await self._wake_and_wait(deadline, future, task)
        self._now = target
        await self._settle()

    async def wait_for_sleepers(self, count: int) -> None:
        """Wait until at least `count` sleepers are pending, within `settle_timeout_s`.

        Raises:
            AssertionError: The bound passed first; the message gives both counts.
        """
        loop = asyncio.get_running_loop()
        give_up_at = loop.time() + self.settle_timeout_s
        while self.pending < count:
            remaining = give_up_at - loop.time()
            if remaining <= 0:
                raise AssertionError(
                    f"waited for {count} sleeper(s) for {self.settle_timeout_s} s; "
                    f"{self.pending} pending"
                )
            self._slept.clear()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._slept.wait(), remaining)

    async def _wake_and_wait(
        self, deadline: datetime, future: asyncio.Future[None], task: asyncio.Task[object]
    ) -> None:
        watcher: asyncio.Future[None] = asyncio.get_running_loop().create_future()

        def on_done(_: object) -> None:
            if not watcher.done():
                watcher.set_result(None)

        self._watchers[task] = watcher
        task.add_done_callback(on_done)
        future.set_result(None)
        try:
            await asyncio.wait_for(watcher, self.settle_timeout_s)
        except TimeoutError:
            self.stalled.append(deadline)
        finally:
            task.remove_done_callback(on_done)
            self._watchers.pop(task, None)

    async def _settle(self) -> None:
        for _ in range(_SETTLE_YIELDS):
            await asyncio.sleep(0)


class RecordingEventSink:
    """An `EventSink` that keeps every payload, safe to call from any thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.payloads: list[dict[str, object]] = []

    def emit(self, payload: dict[str, object]) -> None:
        with self._lock:
            self.payloads.append(payload)

    def of_type(self, event_type: str) -> list[dict[str, object]]:
        """Return the payloads whose `type` is `event_type`, in emit order."""
        with self._lock:
            return [p for p in self.payloads if p.get("type") == event_type]


class ScriptedDecode:
    """A `DecodeFn` that answers from a queue and records every request.

    A queued exception is raised instead of returned. When `gate` is set, each call
    waits on it first, which lets a test hold a decode open to exercise a timeout.
    """

    def __init__(
        self, *replies: DecodeReply | BaseException, gate: asyncio.Event | None = None
    ) -> None:
        self._replies: deque[DecodeReply | BaseException] = deque(replies)
        self.gate = gate
        self.requests: list[ActionDecode] = []

    def push(self, reply: DecodeReply | BaseException) -> None:
        self._replies.append(reply)

    async def __call__(self, decode: ActionDecode) -> DecodeReply:
        self.requests.append(decode)
        if self.gate is not None:
            await self.gate.wait()
        if not self._replies:
            raise AssertionError(
                f"ScriptedDecode has no reply left for request {len(self.requests)}"
            )
        reply = self._replies.popleft()
        if isinstance(reply, BaseException):
            raise reply
        return reply


class FakeCapability(Capability):
    """A minimal `Capability` whose every hook is scripted through attributes.

    `invoke` answers from `outcomes` in order, falling back to an ok outcome, and
    records each call in `invocations`. `match_intent` looks the normalised text up
    in `intents`. `start` raises `start_error` when set, after waiting on
    `start_gate` when that is set.
    """

    def __init__(
        self,
        name: str = "fake",
        *,
        tools: Sequence[ToolSpec] | None = None,
        keywords: Sequence[str] = (),
        intents: Mapping[str, FastIntent] | None = None,
        description: str = "a fake capability",
    ) -> None:
        self.name = name
        self.description = description
        self.tool_specs: tuple[ToolSpec, ...] = (
            tuple(tools) if tools is not None else (make_tool_spec(f"{name}.do"),)
        )
        self.hints = RoutingHints(keywords=tuple(keywords))
        self.extra_keywords: tuple[str, ...] = ()
        self.intents: dict[str, FastIntent] = dict(intents or {})
        self.intent_queries: list[str] = []
        self.outcomes: deque[ToolOutcome | BaseException] = deque()
        self.invocations: list[tuple[str, BaseModel, InvocationContext]] = []
        self.invoke_gate: asyncio.Event | None = None
        self.start_error: BaseException | None = None
        self.start_gate: asyncio.Event | None = None
        self.started = False
        self.stopped = False
        self.current_status = CapabilityStatus(
            name=name, state="available", reason=None, checked_at=DEFAULT_START
        )
        self.escalate_fn: Callable[[str, BaseModel], bool] | None = None
        self.refine_fn: Callable[[str, dict[str, object]], dict[str, object]] | None = None
        self.ringing = False
        self.active_replies: dict[str, ToolOutcome] = {}
        self.active_utterances: list[tuple[str, str]] = []
        self.provider_tool_names: tuple[str, ...] = ()

    def set_status(
        self, state: CapabilityState, reason: str | None = None, *, task_dead: bool = False
    ) -> None:
        self.current_status = CapabilityStatus(
            name=self.name,
            state=state,
            reason=reason,
            checked_at=DEFAULT_START,
            task_dead=task_dead,
        )

    def tools(self) -> Sequence[ToolSpec]:
        return self.tool_specs

    def routing_hints(self) -> RoutingHints:
        return self.hints

    def dynamic_keywords(self) -> tuple[str, ...]:
        return self.extra_keywords

    async def start(self) -> None:
        if self.start_gate is not None:
            await self.start_gate.wait()
        if self.start_error is not None:
            raise self.start_error
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def status(self) -> CapabilityStatus:
        return self.current_status

    async def invoke(self, tool: str, args: BaseModel, ctx: InvocationContext) -> ToolOutcome:
        self.invocations.append((tool, args, ctx))
        if self.invoke_gate is not None:
            await self.invoke_gate.wait()
        if self.outcomes:
            outcome = self.outcomes.popleft()
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        return ToolOutcome(status="ok", summary=f"{tool} done.")

    def refine_schema(self, tool: str, schema: dict[str, object]) -> dict[str, object]:
        if self.refine_fn is None:
            return schema
        return self.refine_fn(tool, schema)

    def match_intent(self, text: str) -> FastIntent | None:
        self.intent_queries.append(text)
        return self.intents.get(text)

    def escalate(self, tool: str, args: BaseModel) -> bool:
        if self.escalate_fn is None:
            return False
        return self.escalate_fn(tool, args)

    def active(self) -> bool:
        return self.ringing

    async def on_active_utterance(self, utterance: str, session_id: str) -> ToolOutcome | None:
        self.active_utterances.append((utterance, session_id))
        return self.active_replies.get(utterance)

    def provider_tools(self) -> tuple[str, ...]:
        return self.provider_tool_names
