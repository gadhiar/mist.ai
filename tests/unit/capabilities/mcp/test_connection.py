"""Tests for `McpConnection`, its supervisor task, and the fake Home Assistant it runs against."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
from collections.abc import AsyncIterator, Callable

import pytest

from backend.capabilities.mcp.connection import (
    _CONTAINED_ERRORS,
    BackoffPolicy,
    McpConnection,
)
from backend.capabilities.mcp.transport import McpToolResult, McpToolSpec
from backend.errors import (
    CapabilityUnavailableError,
    McpProtocolError,
    McpTransportError,
    MistError,
)
from tests.unit.capabilities.fakes import FakeClock
from tests.unit.capabilities.mcp.fake_home_assistant import (
    GET_STATE,
    LIGHT_SET,
    LIVE_CONTEXT,
    TURN_OFF,
    TURN_ON,
    FakeHomeAssistant,
    auth_rejected,
    connect_refused,
    tool_error,
    unexpected_error,
)

LOGGER = "backend.capabilities.mcp.connection"
CORE_TOOLS = [TURN_ON, TURN_OFF, LIGHT_SET, LIVE_CONTEXT]


def _identity(delay: float) -> float:
    return delay


def _connection(
    server: FakeHomeAssistant,
    clock: FakeClock,
    jitter: Callable[[float], float] = _identity,
) -> McpConnection:
    return McpConnection(server.transport, clock, BackoffPolicy(jitter=jitter))


async def _until(predicate: Callable[[], bool], timeout_s: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    give_up_at = loop.time() + timeout_s
    while not predicate():
        if loop.time() > give_up_at:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.001)


@contextlib.asynccontextmanager
async def _running(connection: McpConnection) -> AsyncIterator[McpConnection]:
    await connection.start()
    try:
        yield connection
    finally:
        await connection.stop()


async def _connected(connection: McpConnection) -> None:
    await _until(lambda: connection.state().state == "available")


def _traceback_text(record: logging.LogRecord) -> str:
    # Once a secret is registered, log redaction moves the traceback into `exc_text`.
    if record.exc_text:
        return record.exc_text
    if record.exc_info:
        return logging.Formatter().formatException(record.exc_info)
    return ""


def _records(caplog: pytest.LogCaptureFixture, level: int) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == LOGGER and r.levelno == level]


class TestTypes:
    def test_transport_types_are_frozen_slotted_dataclasses(self):
        spec = McpToolSpec(name="HassTurnOn", description="d", input_schema={})
        result = McpToolResult(is_error=False, text="ok", structured=None)
        assert not hasattr(spec, "__dict__")
        assert not hasattr(result, "__dict__")
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.name = "HassTurnOff"  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.text = "changed"  # type: ignore[misc]

    def test_contained_set_is_mist_error_and_os_error(self):
        assert (MistError, OSError) == _CONTAINED_ERRORS

    def test_backoff_policy_delays(self):
        policy = BackoffPolicy(jitter=_identity)
        assert [policy.delay(n, auth_rejected=False) for n in range(1, 10)] == [
            1.0,
            2.0,
            4.0,
            8.0,
            16.0,
            32.0,
            60.0,
            60.0,
            60.0,
        ]
        assert policy.delay(1, auth_rejected=True) == 300.0
        halved = BackoffPolicy(jitter=lambda d: d / 2)
        assert halved.delay(3, auth_rejected=False) == 2.0
        assert halved.delay(1, auth_rejected=True) == 150.0


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_start_returns_before_first_connect(self):
        server = FakeHomeAssistant()
        server.open_gate = asyncio.Event()
        clock = FakeClock()

        async def scenario() -> None:
            async with _running(_connection(server, clock)) as conn:
                status = conn.state()
                assert status.state == "unavailable"
                assert status.reason == "connecting"
                assert conn.tools() == []
                await _until(lambda: len(server.open_tasks) == 1)
                assert conn.state().state == "unavailable"
                server.open_gate.set()
                await _connected(conn)
                assert conn.state().reason is None
                assert [t.name for t in conn.tools()] == CORE_TOOLS

        # Bounded so a `start()` that waits for the gated connect fails in seconds, not hangs.
        await asyncio.wait_for(scenario(), 5.0)

    @pytest.mark.asyncio
    async def test_open_close_on_supervisor_and_call_from_another_task(self):
        server = FakeHomeAssistant()
        conn = _connection(server, FakeClock())
        async with _running(conn):
            await _connected(conn)
            caller = asyncio.create_task(conn.call_tool(TURN_OFF, {"area": "Bedroom"}, 5.0))
            result = await caller
            assert result.is_error is False
        supervisor = server.open_tasks[0]
        assert supervisor is not None
        assert server.close_tasks == [supervisor]
        assert server.list_tools_tasks == [supervisor]
        assert server.calls[0].task is caller
        assert server.calls[0].arguments == {"area": "Bedroom"}
        assert server.calls[0].timeout_s == 5.0

    @pytest.mark.asyncio
    async def test_second_start_does_not_start_another_supervisor(self):
        server = FakeHomeAssistant()
        conn = _connection(server, FakeClock())
        async with _running(conn):
            await conn.start()
            await _connected(conn)
            assert len(server.transports) == 1

    @pytest.mark.asyncio
    async def test_stop_before_start_returns(self):
        conn = _connection(FakeHomeAssistant(), FakeClock())
        await conn.stop()
        assert conn.task_dead() is False


class TestBackoff:
    @pytest.mark.asyncio
    async def test_backoff_doubles_to_cap_with_injected_jitter(self):
        server = FakeHomeAssistant()
        server.open_failures.extend(connect_refused() for _ in range(8))
        seen: list[float] = []

        def jitter(delay: float) -> float:
            seen.append(delay)
            return delay + 0.25

        clock = FakeClock(settle_timeout_s=0.3)
        async with _running(_connection(server, clock, jitter)) as conn:
            await clock.wait_for_sleepers(1)
            for _ in range(7):
                assert conn.state().reason == "connection refused"
                await clock.advance(clock.sleeps[-1])
            assert seen == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0]
            assert clock.sleeps == [d + 0.25 for d in seen]
            assert clock.stalled == []
            await clock.advance(clock.sleeps[-1])
            await _connected(conn)
            assert len(server.open_tasks) == 9

    @pytest.mark.asyncio
    async def test_auth_rejection_backs_off_300_seconds(self):
        server = FakeHomeAssistant()
        server.open_failures.extend([auth_rejected(), connect_refused()])
        clock = FakeClock(settle_timeout_s=0.3)
        async with _running(_connection(server, clock)) as conn:
            await clock.wait_for_sleepers(1)
            assert clock.sleeps == [300.0]
            assert conn.state().reason == "auth rejected"
            await clock.advance(299.0)
            assert len(server.open_tasks) == 1
            await clock.advance(1.0)
            assert clock.sleeps == [300.0, 2.0]
            assert conn.state().reason == "connection refused"
            await clock.advance(2.0)
            await _connected(conn)

    @pytest.mark.asyncio
    async def test_os_error_and_list_tools_failure_are_contained(self, caplog):
        server = FakeHomeAssistant()
        server.open_failures.append(ConnectionRefusedError(111, "Connection refused"))
        server.list_tools_failures.append(McpProtocolError("tools/list answer has no tools"))
        clock = FakeClock(settle_timeout_s=0.3)
        caplog.set_level(logging.WARNING, logger=LOGGER)
        async with _running(_connection(server, clock)) as conn:
            await clock.wait_for_sleepers(1)
            assert "Connection refused" in (conn.state().reason or "")
            await clock.advance(1.0)
            assert conn.state().reason == "tools/list answer has no tools"
            assert clock.sleeps == [1.0, 2.0]
            # The second transport opened before its listing failed, so it was closed.
            assert server.close_tasks == [server.open_tasks[1]]
            await clock.advance(2.0)
            await _connected(conn)
            assert conn.task_dead() is False
        assert _records(caplog, logging.ERROR) == []

    @pytest.mark.asyncio
    async def test_backoff_resets_after_a_successful_connect(self):
        server = FakeHomeAssistant()
        server.open_failures.extend([connect_refused(), connect_refused()])
        clock = FakeClock(settle_timeout_s=0.3)
        async with _running(_connection(server, clock)) as conn:
            await clock.wait_for_sleepers(1)
            await clock.advance(1.0)
            await clock.advance(2.0)
            await _connected(conn)
            server.call_script.append(McpTransportError("connection reset"))
            server.open_failures.append(connect_refused())
            with pytest.raises(McpTransportError):
                await conn.call_tool(TURN_ON, {}, 5.0)
            await clock.wait_for_sleepers(1)
            assert clock.sleeps == [1.0, 2.0, 1.0]

    @pytest.mark.asyncio
    async def test_tools_refresh_on_reconnect(self):
        server = FakeHomeAssistant(include_get_state=True)
        conn = _connection(server, FakeClock())
        async with _running(conn):
            await _connected(conn)
            assert [t.name for t in conn.tools()] == [*CORE_TOOLS, GET_STATE]
            server.omit = {LIGHT_SET}
            server.call_script.append(McpTransportError("connection reset"))
            with pytest.raises(McpTransportError):
                await conn.call_tool(TURN_ON, {}, 5.0)
            await _until(lambda: len(server.list_tools_tasks) == 2)
            await _connected(conn)
            assert [t.name for t in conn.tools()] == [TURN_ON, TURN_OFF, LIVE_CONTEXT, GET_STATE]


class TestCallTool:
    @pytest.mark.asyncio
    async def test_call_while_disconnected_raises_with_last_reason(self):
        server = FakeHomeAssistant()
        server.open_failures.append(connect_refused())
        clock = FakeClock()
        conn = _connection(server, clock)
        with pytest.raises(CapabilityUnavailableError, match="not started"):
            await conn.call_tool(TURN_ON, {}, 5.0)
        async with _running(conn):
            await clock.wait_for_sleepers(1)
            with pytest.raises(CapabilityUnavailableError) as raised:
                await conn.call_tool(TURN_ON, {}, 5.0)
            assert str(raised.value) == "connection refused"
            assert conn.tools() == []
        assert server.calls == []

    @pytest.mark.asyncio
    async def test_transport_error_marks_reconnect_and_is_not_retried(self):
        server = FakeHomeAssistant()
        server.open_gate = asyncio.Event()
        server.open_gate.set()
        conn = _connection(server, FakeClock())
        async with _running(conn):
            await _connected(conn)
            server.open_gate.clear()
            server.call_script.append(McpTransportError("connection reset"))
            with pytest.raises(McpTransportError, match="connection reset"):
                await conn.call_tool(TURN_ON, {"name": "Bedroom Lamp"}, 5.0)
            assert len(server.calls) == 1
            status = conn.state()
            assert status.state == "unavailable"
            assert status.reason == "connection reset"
            assert conn.tools() == []
            with pytest.raises(CapabilityUnavailableError, match="connection reset"):
                await conn.call_tool(TURN_ON, {}, 5.0)
            await _until(lambda: len(server.open_tasks) == 2)
            assert server.close_tasks == [server.open_tasks[0]]
            server.open_gate.set()
            await _connected(conn)
            assert len(server.calls) == 1

    @pytest.mark.asyncio
    async def test_slow_call_times_out_as_transport_error(self):
        server = FakeHomeAssistant()
        server.call_gate = asyncio.Event()
        conn = _connection(server, FakeClock())
        async with _running(conn):
            await _connected(conn)
            with pytest.raises(McpTransportError, match="timed out"):
                await conn.call_tool(LIVE_CONTEXT, {}, 0.01)
            assert conn.state().state == "unavailable"

    @pytest.mark.asyncio
    async def test_late_failure_from_an_earlier_session_keeps_the_current_one(self):
        server = FakeHomeAssistant()
        server.call_gate = asyncio.Event()
        conn = _connection(server, FakeClock())
        async with _running(conn):
            await _connected(conn)
            # Held on the first session until the gate opens.
            late = asyncio.create_task(conn.call_tool(TURN_ON, {}, 30.0))
            await _until(lambda: len(server.calls) == 1)
            with pytest.raises(McpTransportError, match="timed out"):
                await conn.call_tool(TURN_OFF, {}, 0.01)
            await _until(lambda: len(server.list_tools_tasks) == 2)
            await _connected(conn)
            server.call_script.append(McpTransportError("late reset on the first session"))
            server.call_gate.set()
            with pytest.raises(McpTransportError, match="late reset"):
                await late
            assert conn.state().state == "available"
            assert [t.name for t in conn.tools()] == CORE_TOOLS
            assert len(server.open_tasks) == 2
            result = await conn.call_tool(TURN_ON, {}, 5.0)
            assert result.is_error is False

    @pytest.mark.asyncio
    async def test_auth_rejected_call_drops_the_session_and_waits_the_auth_delay(self):
        server = FakeHomeAssistant()
        clock = FakeClock(settle_timeout_s=0.3)
        conn = _connection(server, clock, jitter=lambda d: d + 0.5)
        async with _running(conn):
            await _connected(conn)
            server.call_script.append(auth_rejected())
            with pytest.raises(CapabilityUnavailableError, match="auth rejected"):
                await conn.call_tool(TURN_ON, {"name": "Bedroom Lamp"}, 5.0)
            assert len(server.calls) == 1
            status = conn.state()
            assert status.state == "unavailable"
            assert status.reason == "auth rejected"
            assert status.task_dead is False
            assert conn.tools() == []
            await clock.wait_for_sleepers(1)
            assert clock.sleeps == [300.5]
            assert server.close_tasks == [server.open_tasks[0]]
            await clock.advance(300.0)
            assert len(server.open_tasks) == 1
            await clock.advance(0.5)
            await _connected(conn)
            assert len(server.open_tasks) == 2
            assert len(server.calls) == 1

    @pytest.mark.asyncio
    async def test_tool_error_result_is_returned_and_keeps_the_session(self):
        server = FakeHomeAssistant()
        server.call_script.append(tool_error("Unable to find a device named Attic Lamp"))
        conn = _connection(server, FakeClock())
        async with _running(conn):
            await _connected(conn)
            result = await conn.call_tool(TURN_ON, {"name": "Attic Lamp"}, 5.0)
            assert result.is_error is True
            assert "Attic Lamp" in result.text
            assert conn.state().state == "available"
            assert len(server.open_tasks) == 1


class TestSupervision:
    @pytest.mark.asyncio
    async def test_stop_closes_from_supervisor_and_logs_no_error(self, caplog):
        server = FakeHomeAssistant()
        conn = _connection(server, FakeClock())
        caplog.set_level(logging.DEBUG, logger=LOGGER)
        async with _running(conn):
            await _connected(conn)
        assert server.close_tasks == [server.open_tasks[0]]
        assert server.transports[0].is_open is False
        assert conn.task_dead() is False
        status = conn.state()
        assert status.state == "unavailable"
        assert status.task_dead is False
        assert _records(caplog, logging.ERROR) == []
        assert _records(caplog, logging.WARNING) == []
        with pytest.raises(CapabilityUnavailableError):
            await conn.call_tool(TURN_ON, {}, 5.0)

    @pytest.mark.asyncio
    async def test_stop_during_backoff_logs_no_error(self, caplog):
        server = FakeHomeAssistant()
        server.open_failures.append(connect_refused())
        clock = FakeClock()
        conn = _connection(server, clock)
        caplog.set_level(logging.WARNING, logger=LOGGER)
        async with _running(conn):
            await clock.wait_for_sleepers(1)
        assert conn.task_dead() is False
        assert server.close_tasks == []
        assert _records(caplog, logging.ERROR) == []

    @pytest.mark.asyncio
    async def test_uncontained_error_ends_the_supervisor_visibly(self, caplog):
        server = FakeHomeAssistant()
        server.open_failures.append(unexpected_error())
        conn = _connection(server, FakeClock())
        caplog.set_level(logging.WARNING, logger=LOGGER)
        await conn.start()
        await _until(conn.task_dead)
        status = conn.state()
        assert status.state == "unavailable"
        assert status.task_dead is True
        assert "RuntimeError" in (status.reason or "")
        errors = _records(caplog, logging.ERROR)
        assert len(errors) == 1
        assert "RuntimeError: unexpected fault in the transport" in _traceback_text(errors[0])
        with pytest.raises(CapabilityUnavailableError) as raised:
            await conn.call_tool(TURN_ON, {}, 5.0)
        assert str(raised.value) == status.reason
        # The done-callback closed nothing: `close` may only run in the supervisor task.
        assert server.close_tasks == []
        await conn.stop()
        assert conn.task_dead() is True
        assert conn.state() == status
        assert server.close_tasks == []
        assert len(_records(caplog, logging.ERROR)) == 1

    @pytest.mark.asyncio
    async def test_death_after_open_closes_in_the_supervisor_not_the_callback(self, caplog):
        server = FakeHomeAssistant()
        server.list_tools_failures.append(unexpected_error())
        conn = _connection(server, FakeClock())
        caplog.set_level(logging.WARNING, logger=LOGGER)
        await conn.start()
        await _until(conn.task_dead)
        assert server.close_tasks == [server.open_tasks[0]]
        assert server.close_tasks[0] is not None
        await conn.stop()
        assert len(_records(caplog, logging.ERROR)) == 1

    @pytest.mark.asyncio
    async def test_cancel_without_stop_is_a_death(self, caplog):
        server = FakeHomeAssistant()
        conn = _connection(server, FakeClock())
        caplog.set_level(logging.WARNING, logger=LOGGER)
        await conn.start()
        await _connected(conn)
        supervisor = server.open_tasks[0]
        assert supervisor is not None
        supervisor.cancel()
        await _until(conn.task_dead)
        status = conn.state()
        assert status.state == "unavailable"
        assert status.task_dead is True
        assert "cancel" in (status.reason or "")
        assert len(_records(caplog, logging.ERROR)) == 1
        with pytest.raises(CapabilityUnavailableError):
            await conn.call_tool(TURN_ON, {}, 5.0)
        await conn.stop()
        assert conn.task_dead() is True

    @pytest.mark.parametrize(
        "error",
        [McpTransportError("close failed"), RuntimeError("close blew up")],
        ids=["contained", "uncontained"],
    )
    @pytest.mark.asyncio
    async def test_close_raising_during_stop_warns_once(self, caplog, error):
        server = FakeHomeAssistant()
        server.close_error = error
        conn = _connection(server, FakeClock())
        caplog.set_level(logging.DEBUG, logger=LOGGER)
        async with _running(conn):
            await _connected(conn)
        warnings = _records(caplog, logging.WARNING)
        assert len(warnings) == 1
        assert str(error) in warnings[0].getMessage() + _traceback_text(warnings[0])
        assert _records(caplog, logging.ERROR) == []
        assert conn.task_dead() is False
        assert conn.state().task_dead is False


class _Listener:
    """Counts its calls; `error`, when set, is raised from every call after counting it."""

    def __init__(self, error: BaseException | None = None) -> None:
        self.calls = 0
        self.error = error

    def __call__(self) -> None:
        self.calls += 1
        if self.error is not None:
            raise self.error


async def _settle() -> None:
    # Lets every `call_soon` callback scheduled so far run.
    for _ in range(5):
        await asyncio.sleep(0)


class TestStateListener:
    @pytest.mark.asyncio
    async def test_listener_is_scheduled_not_called(self):
        server = FakeHomeAssistant()
        server.open_gate = asyncio.Event()
        conn = _connection(server, FakeClock())
        listener = _Listener()
        conn.set_state_listener(listener)
        async with _running(conn):
            # `start` wrote "connecting"; the listener runs only once the loop turns.
            assert listener.calls == 0
            await _settle()
            assert listener.calls == 1

    @pytest.mark.asyncio
    async def test_each_transition_schedules_the_listener_once(self):
        server = FakeHomeAssistant()
        server.open_gate = asyncio.Event()
        conn = _connection(server, FakeClock())
        listener = _Listener()
        conn.set_state_listener(listener)
        await conn.start()
        await _until(lambda: len(server.open_tasks) == 1)
        await _settle()
        assert listener.calls == 1  # connecting

        server.open_gate.set()
        await _connected(conn)
        await _settle()
        assert listener.calls == 2  # connected

        server.open_gate.clear()
        server.call_script.append(McpTransportError("connection reset"))
        with pytest.raises(McpTransportError):
            await conn.call_tool(TURN_ON, {}, 5.0)
        await _until(lambda: len(server.open_tasks) == 2)
        await _settle()
        assert listener.calls == 3  # unavailable, with the reconnect held at `open`

        await conn.stop()
        await _settle()
        assert listener.calls == 4  # stopped
        assert (conn.state().state, conn.state().reason) == ("unavailable", "stopped")

    @pytest.mark.asyncio
    async def test_stop_while_connected_schedules_the_listener_once(self):
        server = FakeHomeAssistant()
        conn = _connection(server, FakeClock())
        listener = _Listener()
        conn.set_state_listener(listener)
        await conn.start()
        await _connected(conn)
        await _settle()
        before = listener.calls

        await conn.stop()
        await _settle()

        assert listener.calls == before + 1

    @pytest.mark.asyncio
    async def test_death_from_the_done_callback_schedules_the_listener_once(self):
        server = FakeHomeAssistant()
        server.open_gate = asyncio.Event()
        server.open_failures.append(unexpected_error())
        conn = _connection(server, FakeClock())
        listener = _Listener()
        conn.set_state_listener(listener)
        await conn.start()
        await _until(lambda: len(server.open_tasks) == 1)
        await _settle()
        assert listener.calls == 1

        server.open_gate.set()
        await _until(conn.task_dead)
        await _settle()

        assert listener.calls == 2
        await conn.stop()
        await _settle()
        # A dead connection keeps its death reason, so stopping it changes nothing.
        assert listener.calls == 2

    @pytest.mark.asyncio
    async def test_failed_connect_connect_and_drop_schedule_three_times(self):
        server = FakeHomeAssistant()
        server.open_gate = asyncio.Event()
        server.open_failures.append(connect_refused())
        clock = FakeClock(settle_timeout_s=0.3)
        conn = _connection(server, clock)
        listener = _Listener()
        async with _running(conn):
            conn.set_state_listener(listener)
            await _settle()
            server.open_gate.set()
            await clock.wait_for_sleepers(1)
            await _settle()
            assert listener.calls == 1  # the failed connect

            await clock.advance(1.0)
            await _connected(conn)
            await _settle()
            assert listener.calls == 2  # connected

            server.open_gate.clear()
            server.call_script.append(McpTransportError("connection refused"))
            with pytest.raises(McpTransportError):
                await conn.call_tool(TURN_ON, {}, 5.0)
            await _until(lambda: len(server.open_tasks) == 3)
            await _settle()
            assert listener.calls == 3  # the drop, though its reason matches the failed connect

    @pytest.mark.asyncio
    async def test_repeated_failed_connect_with_the_same_reason_schedules_nothing_new(self):
        server = FakeHomeAssistant()
        server.open_failures.extend([connect_refused(), connect_refused(), connect_refused()])
        clock = FakeClock(settle_timeout_s=0.3)
        conn = _connection(server, clock)
        listener = _Listener()
        conn.set_state_listener(listener)
        async with _running(conn):
            await clock.wait_for_sleepers(1)
            await _settle()
            assert listener.calls == 2  # connecting, then the first failure

            await clock.advance(1.0)
            await clock.advance(2.0)
            await _settle()

            assert len(server.open_tasks) == 3
            assert listener.calls == 2

    @pytest.mark.asyncio
    async def test_raising_listener_leaves_the_supervisor_running(self):
        loop = asyncio.get_running_loop()
        handled: list[dict[str, object]] = []
        previous = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: handled.append(context))
        try:
            server = FakeHomeAssistant()
            conn = _connection(server, FakeClock())
            conn.set_state_listener(_Listener(RuntimeError("listener bug")))
            async with _running(conn):
                await _connected(conn)
                await _settle()
                assert conn.task_dead() is False
                assert conn.state().state == "available"
                assert [t.name for t in conn.tools()] == CORE_TOOLS
                result = await conn.call_tool(TURN_ON, {}, 5.0)
                assert result.is_error is False
        finally:
            loop.set_exception_handler(previous)
        assert handled
        assert all(isinstance(c.get("exception"), RuntimeError) for c in handled)
        assert conn.task_dead() is False

    @pytest.mark.asyncio
    async def test_cleared_listener_is_not_scheduled(self):
        server = FakeHomeAssistant()
        conn = _connection(server, FakeClock())
        listener = _Listener()
        conn.set_state_listener(listener)
        conn.set_state_listener(None)
        async with _running(conn):
            await _connected(conn)
            await _settle()
        await _settle()
        assert listener.calls == 0

    @pytest.mark.asyncio
    async def test_reconnect_after_an_auth_drop_resets_the_auth_delay(self):
        server = FakeHomeAssistant()
        clock = FakeClock(settle_timeout_s=0.3)
        conn = _connection(server, clock)
        async with _running(conn):
            await _connected(conn)
            server.call_script.append(auth_rejected())
            with pytest.raises(CapabilityUnavailableError):
                await conn.call_tool(TURN_ON, {}, 5.0)
            await clock.wait_for_sleepers(1)
            await clock.advance(300.0)
            await _connected(conn)
            assert len(server.open_tasks) == 2

            server.call_script.append(McpTransportError("connection reset"))
            with pytest.raises(McpTransportError):
                await conn.call_tool(TURN_ON, {}, 5.0)
            await _until(lambda: len(server.open_tasks) == 3)
            await _connected(conn)

            assert clock.sleeps == [300.0]


class TestFakeHomeAssistant:
    def test_lists_the_documented_intents_with_their_slots(self):
        tools = {spec.name: spec for spec in FakeHomeAssistant().tool_specs()}
        assert list(tools) == CORE_TOOLS
        for name in (TURN_ON, TURN_OFF):
            props = tools[name].input_schema["properties"]
            assert set(props) == {"name", "area", "floor", "domain", "device_class"}
        light_set = tools[LIGHT_SET].input_schema["properties"]
        assert set(light_set) == {
            "name",
            "area",
            "floor",
            "domain",
            "brightness",
            "color",
            "temperature",
        }
        assert light_set["brightness"] == {"type": "integer", "minimum": 0, "maximum": 100}
        assert light_set["temperature"] == {"type": "integer", "minimum": 1000, "maximum": 10000}
        assert light_set["color"] == {"type": "string"}
        assert tools[LIVE_CONTEXT].input_schema["properties"] == {}

    def test_optional_get_state_and_omitted_tools(self):
        with_state = [s.name for s in FakeHomeAssistant(include_get_state=True).tool_specs()]
        assert with_state == [*CORE_TOOLS, GET_STATE]
        omitted = [s.name for s in FakeHomeAssistant(omit=[TURN_OFF, LIVE_CONTEXT]).tool_specs()]
        assert omitted == [TURN_ON, LIGHT_SET]

    @pytest.mark.asyncio
    async def test_records_calls_and_answers_from_script_then_defaults(self):
        server = FakeHomeAssistant(omit=[LIGHT_SET])
        transport = server.transport()
        await transport.open()
        server.call_script.append(tool_error("nope"))
        first = await transport.call_tool(TURN_ON, {"area": "Kitchen"}, 3.0)
        second = await transport.call_tool(TURN_ON, {"area": "Kitchen"}, 3.0)
        context = await transport.call_tool(LIVE_CONTEXT, {}, 3.0)
        omitted = await transport.call_tool(LIGHT_SET, {"brightness": 30}, 3.0)
        assert (first.is_error, first.text) == (True, "nope")
        assert second.is_error is False
        assert context.text == server.live_context
        assert omitted.is_error is True
        assert [(c.name, c.arguments) for c in server.calls] == [
            (TURN_ON, {"area": "Kitchen"}),
            (TURN_ON, {"area": "Kitchen"}),
            (LIVE_CONTEXT, {}),
            (LIGHT_SET, {"brightness": 30}),
        ]
        await transport.close()
        with pytest.raises(McpTransportError, match="not open"):
            await transport.call_tool(TURN_ON, {}, 3.0)

    @pytest.mark.asyncio
    async def test_scripts_connect_and_close_failures(self):
        server = FakeHomeAssistant()
        server.open_failures.extend([connect_refused(), auth_rejected(), unexpected_error()])
        with pytest.raises(McpTransportError, match="connection refused"):
            await server.transport().open()
        with pytest.raises(CapabilityUnavailableError, match="auth rejected"):
            await server.transport().open()
        with pytest.raises(RuntimeError):
            await server.transport().open()
        transport = server.transport()
        await transport.open()
        server.close_error = McpTransportError("close failed")
        with pytest.raises(McpTransportError, match="close failed"):
            await transport.close()
        assert len(server.transports) == 4
