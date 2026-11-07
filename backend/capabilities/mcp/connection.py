"""`McpConnection`: one MCP transport, kept connected by a supervised task on the main loop.

The supervisor task owns the transport's lifetime. It alone calls `open` and `close`, because an
SDK-backed transport holds anyio cancel scopes, which must be entered and exited in the same
task; other tasks only call tools on the open session.

A failed connect attempt is an expected event, not a fault: Home Assistant may be down or
restarting, and MIST must start and run without it (ADR-025). Such a failure is contained, the
connection reports itself `unavailable` with the reason, and the supervisor backs off and tries
again. An exception outside the contained set is a bug; it ends the supervisor task, and the
done-callback records the death in `state()` and `task_dead()`, so that whatever reads them
reports it instead of waiting forever for a reconnect that will not come.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from backend.errors import CapabilityUnavailableError, McpTransportError, MistError

from ..clock import Clock
from ..contracts import CapabilityStatus
from .transport import McpToolResult, McpToolSpec, McpTransport

logger = logging.getLogger(__name__)

# Failures one connect attempt contains rather than dying on: the MIST error tree (a transport
# reports its failures as `McpTransportError`) and OS-level I/O. Anything else is a bug and ends the
# supervisor (see `_on_supervisor_done`). No broader catch is added.
_CONTAINED_ERRORS = (MistError, OSError)

_STATUS_NAME = "mcp"

# Bounds the doubling exponent so a long outage cannot overflow the float; 2**30 s is far
# past any cap.
_MAX_EXPONENT = 30


@dataclass(frozen=True, slots=True)
class BackoffPolicy:
    """How long the supervisor waits after consecutive failed connect attempts.

    Attributes:
        jitter: Maps the nominal delay to the delay actually slept. Injected so tests run
            deterministic delays and production can spread reconnects from several clients.
        initial_s: The delay after the first failure; it doubles per consecutive failure.
        cap_s: The most the doubling reaches.
        auth_delay_s: The delay after an auth rejection. Retrying a refused credential soon
            cannot succeed, so this ignores the doubling.
    """

    jitter: Callable[[float], float]
    initial_s: float = 1.0
    cap_s: float = 60.0
    auth_delay_s: float = 300.0

    def delay(self, failures: int, *, auth_rejected: bool) -> float:
        """Return the delay after `failures` consecutive failures, with jitter applied."""
        if auth_rejected:
            nominal = self.auth_delay_s
        else:
            exponent = min(max(failures - 1, 0), _MAX_EXPONENT)
            nominal = min(self.cap_s, self.initial_s * 2.0**exponent)
        return self.jitter(nominal)


def _describe(exc: BaseException) -> str:
    return str(exc) or type(exc).__name__


class McpConnection:
    """Keeps one MCP transport connected and lists its tools.

    A connection is started once and stopped once. Each connect attempt builds a new transport
    from `transport_factory`, opens it and lists its tools. On success the state is `available`
    and calls go through. When a call hits a transport error the session is dropped, and the
    supervisor closes it and reconnects at once, listing the tools again. When a call's
    credentials are rejected the session is dropped the same way, but the supervisor waits the
    auth delay before it reconnects, as after a rejected connect. Consecutive failures back off
    by `backoff`. A call is never retried here: the caller decides.

    A state listener, when set, hears of every change of the (state, reason) pair `state()`
    reports, so the host can broadcast it without waiting for its poll.
    """

    def __init__(
        self,
        transport_factory: Callable[[], McpTransport],
        clock: Clock,
        backoff: BackoffPolicy,
    ) -> None:
        self._transport_factory = transport_factory
        self._clock = clock
        self._backoff = backoff
        self._task: asyncio.Task[None] | None = None
        self._stopping = False
        self._dead = False
        # `_open` is the transport the supervisor has opened and not yet closed; only the
        # supervisor reads or writes it. `_session` is the same transport while calls may use
        # it, cleared as soon as a call reports it broken.
        self._open: McpTransport | None = None
        self._session: McpTransport | None = None
        self._tools: list[McpToolSpec] = []
        self._reason = "not started"
        self._reconnect = asyncio.Event()
        # Set with `_reconnect` when a call's credentials were rejected, so the supervisor
        # backs off before reconnecting instead of retrying a refused credential at once.
        self._dropped_for_auth: CapabilityUnavailableError | None = None
        self._listener: Callable[[], None] | None = None
        # The (state, reason) pair last notified, so only a real change schedules the listener.
        self._notified: tuple[str, str | None] = ("unavailable", self._reason)

    def set_state_listener(self, listener: Callable[[], None] | None) -> None:
        """Register the one callable told of each state change, or clear it with None.

        It is scheduled with `loop.call_soon`, never called in place, so a listener that raises
        reaches the loop's exception handler and cannot end the supervisor.
        """
        self._listener = listener

    async def start(self) -> None:
        """Schedule the supervisor task and return without waiting for the first connect."""
        if self._task is not None:
            return
        self._reason = "connecting"
        self._notify_if_changed()
        self._task = asyncio.create_task(self._supervise(), name="mcp-connection")
        self._task.add_done_callback(self._on_supervisor_done)

    async def stop(self) -> None:
        """Stop the supervisor, which closes an open transport in its own task first.

        The stopping flag is set before the cancel, so the done-callback reads the end as
        orderly. A dead supervisor stays recorded as dead. The reason is set to "stopped"
        before the cancel, so a stop notifies the listener once: here when no session is open,
        or from the supervisor's `finally` when it clears the open one.
        """
        self._stopping = True
        if not self._dead:
            self._reason = "stopped"
        self._notify_if_changed()
        task = self._task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.wait({task})
        self._session = None
        self._tools = []
        if not self._dead:
            self._reason = "stopped"
        self._notify_if_changed()

    def state(self) -> CapabilityStatus:
        """Return the connection's status; `task_dead` is set once the supervisor has died."""
        if self._session is not None:
            return CapabilityStatus(
                name=_STATUS_NAME, state="available", reason=None, checked_at=self._clock.now()
            )
        return CapabilityStatus(
            name=_STATUS_NAME,
            state="unavailable",
            reason=self._reason,
            checked_at=self._clock.now(),
            task_dead=self._dead,
        )

    def task_dead(self) -> bool:
        """Return True once the supervisor task has ended other than by `stop()`."""
        return self._dead

    def tools(self) -> list[McpToolSpec]:
        """Return the tools listed on the current session, or an empty list while down."""
        return list(self._tools)

    async def call_tool(
        self, name: str, arguments: Mapping[str, object], timeout_s: float
    ) -> McpToolResult:
        """Call one tool on the open session.

        Raises:
            CapabilityUnavailableError: No session is open, and the message is the last reason;
                or the transport reports the call's credentials rejected ("auth rejected"),
                and the session is dropped for the supervisor to reconnect after the auth delay.
            McpTransportError: The call failed in the transport. The session is dropped for
                the supervisor to reconnect at once.

        A failed call is never retried.
        """
        session = self._session
        if session is None:
            raise CapabilityUnavailableError(self._reason)
        try:
            return await session.call_tool(name, arguments, timeout_s)
        except (CapabilityUnavailableError, McpTransportError) as exc:
            self._drop(session, exc)
            raise

    def _drop(
        self, session: McpTransport, failure: CapabilityUnavailableError | McpTransportError
    ) -> None:
        # A call that started on an earlier session must not drop the current one.
        if self._session is not session:
            return
        reason = _describe(failure)
        self._session = None
        self._tools = []
        self._reason = reason
        if isinstance(failure, CapabilityUnavailableError):
            self._dropped_for_auth = failure
        self._reconnect.set()
        logger.warning("MCP session dropped after a failed call: %s; reconnecting", reason)
        self._notify_if_changed()

    async def _supervise(self) -> None:
        failures = 0
        try:
            while True:
                failure = await self._connect()
                if failure is None:
                    failures = 0
                    failure = await self._serve()
                    if failure is None:
                        continue
                failures += 1
                delay = self._backoff.delay(
                    failures, auth_rejected=isinstance(failure, CapabilityUnavailableError)
                )
                logger.warning(
                    "MCP connection failure %d in a row: %s; retrying in %.1f s",
                    failures,
                    self._reason,
                    delay,
                )
                # Outside the contained `try` in `_connect`: `TimeoutError` is an `OSError`, so
                # a timed sleep's own error must never read as a failed attempt.
                await self._clock.sleep(delay)
        finally:
            # Runs in this task on every end, a stop's cancel included, so `close` keeps to
            # the task that called `open`.
            self._session = None
            self._tools = []
            self._notify_if_changed()
            await self._close_open()

    async def _serve(self) -> CapabilityUnavailableError | None:
        """Wait until a call drops the session, then close it.

        Returns the auth rejection that dropped it, which the caller backs off from like a
        rejected connect, or None to reconnect at once.
        """
        await self._reconnect.wait()
        await self._close_open()
        return self._dropped_for_auth

    async def _connect(self) -> BaseException | None:
        """Make one connect attempt; return the contained failure, or None once connected."""
        self._reconnect.clear()
        self._dropped_for_auth = None
        try:
            transport = self._transport_factory()
            await transport.open()
            self._open = transport
            tools = await transport.list_tools()
        except _CONTAINED_ERRORS as exc:
            failure = exc
        else:
            self._tools = list(tools)
            self._session = transport
            logger.info("MCP session open with %d tools", len(tools))
            self._notify_if_changed()
            return None
        self._reason = _describe(failure)
        self._notify_if_changed()
        await self._close_open()
        return failure

    async def _close_open(self) -> None:
        transport = self._open
        if transport is None:
            return
        self._open = None
        try:
            await transport.close()
        except _CONTAINED_ERRORS as exc:
            logger.warning("MCP transport close failed: %s", _describe(exc))

    def _on_supervisor_done(self, task: asyncio.Task[None]) -> None:
        """Record how the supervisor ended; it never closes the transport.

        `close` must run in the supervisor task, and a done-callback runs outside any task, so
        the supervisor's own `finally` closes instead. The flag `stop()` sets, not the
        cancellation, decides what an end means: an anyio cancel scope inside the SDK session
        can cancel this task without `stop()`, and reading that as orderly would let the
        connection die silently.
        """
        if self._stopping:
            if not task.cancelled():
                exc = task.exception()
                if exc is not None:
                    logger.warning(
                        "MCP transport raised while the connection stopped: %s",
                        _describe(exc),
                        exc_info=exc,
                    )
            return
        if task.cancelled():
            reason = "MCP supervisor task was cancelled without stop()"
            logger.error("%s; the connection stays down until the backend restarts", reason)
        else:
            exc = task.exception()
            if exc is not None:
                reason = f"MCP supervisor task died: {type(exc).__name__}: {exc}"
                logger.error("%s", reason, exc_info=exc)
            else:
                reason = "MCP supervisor task returned, which it never does by design"
                logger.error("%s", reason)
        self._dead = True
        self._session = None
        self._tools = []
        self._reason = reason
        self._notify_if_changed()

    def _notify_if_changed(self) -> None:
        """Schedule the listener when the (state, reason) pair differs from the last notified.

        Every write to the session, the reason or the dead flag is followed by this call. The
        pair is recorded even with no listener set, so a listener set later hears only of
        changes made after it.
        """
        status = self.state()
        pair = (status.state, status.reason)
        if pair == self._notified:
            return
        self._notified = pair
        if self._listener is not None:
            asyncio.get_running_loop().call_soon(self._listener)
