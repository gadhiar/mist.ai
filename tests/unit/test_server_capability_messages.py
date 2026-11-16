"""WebSocket protocol 1.4.0 capability messages on the real `websocket_endpoint` (build T14).

Every test drives `server.websocket_endpoint` over the scripted fake websocket of
`tests/unit/test_session_id_propagation.py`. The capability host is a real `CapabilityHost`
over fake capabilities and a scripted alarm control, reached through `_capability_build` as
`lifespan` leaves it, so the server's own wiring is what runs.

The containment rule these pin: the receive loop's catch-all `except Exception` ends the
connection, so no capability exception may reach it. A failing capability call is answered on
its socket and the loop goes on to the next message; a `reset_vad` later in the same script is
the witness that the loop survived.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Literal

import pytest
import pytest_asyncio
from fastapi import WebSocketDisconnect

from backend.capabilities.action_log import ActionLog
from backend.capabilities.alarms.records import ReportOutcome, ReportResult
from backend.capabilities.contracts import ToolOutcome
from backend.capabilities.db import CapabilitiesDb
from backend.capabilities.decoder import ActionDecoder
from backend.capabilities.host import CapabilityHost, HostSettings
from backend.capabilities.policy import PolicyTable
from backend.capabilities.router import RouterConfig, ToolRouter
from backend.capabilities.secrets import Secret, SecretRegistry
from backend.chat.context_budget import ApproximateTokenCounter
from backend.errors import CapabilityStoreError, CapabilityUnavailableError
from backend.factories import CapabilityBuild
from tests.unit.capabilities.fakes import FakeCapability, FakeClock, RecordingEventSink
from tests.unit.test_session_id_propagation import (
    FakeWebSocket,
    RecordingConversationHandler,
    build_voice_processor,
    reset_session_context,  # noqa: F401 -- autouse fixture, applied by import
    run_endpoint,
)

TOKEN = "tok-3c7d9e1f"
SERVER_LOGGER = "backend.server"
RESET_VAD = {"type": "reset_vad"}
VAD_RESET_ANSWER = {"type": "status", "message": "VAD reset"}

AlarmAction = Literal["stop", "snooze", "dismiss"]


# -- Fakes -----------------------------------------------------------------------------------


def _alarm_state() -> dict[str, object]:
    """A valid `alarm_state` payload, as the alarm scheduler's control builds it."""
    return {
        "type": "alarm_state",
        "rev": 7,
        "alarms": [
            {
                "alarm_id": "a-ringing",
                "record_id": "r1",
                "kind": "alarm",
                "category": "tagged",
                "label": "work",
                "state": "ringing",
                "next_fire_at": None,
                "wall_time": "07:00",
                "repeat": "none",
                "days": None,
                "duration_s": None,
                "ring_on": "all",
            }
        ],
        "ringing": [{"alarm_id": "a-ringing", "late_by_s": 0.0}],
        "missed": [],
        "fallback_after_s": 30.0,
        "missed_grace_s": 900.0,
        "ring_timeout_s": 600.0,
        "snooze_minutes": 9,
        "tz": "UTC",
        "at": "2026-01-05T07:00:00Z",
    }


class ScriptedAlarmControl:
    """An `AlarmControl` holding one ringing and one missed alarm, recording every call.

    `act` answers as the scheduler's control does: `stop` and `snooze` apply to a ringing alarm
    and `dismiss` to a missed one, and anything else is `invalid_args` naming the alarm. A
    report answers `applied` for an alarm it knows and `unknown` otherwise.
    """

    def __init__(self) -> None:
        self.ringing = {"a-ringing"}
        self.missed = {"a-missed"}
        self.snapshot_error: BaseException | None = None
        self.act_error: BaseException | None = None
        self.acts: list[tuple[str, str, int | None, str]] = []
        self.reports: list[dict[str, object]] = []

    async def snapshot(self) -> dict[str, object]:
        if self.snapshot_error is not None:
            raise self.snapshot_error
        return _alarm_state()

    async def act(
        self, alarm_id: str, action: AlarmAction, minutes: int | None, *, endpoint: str
    ) -> ToolOutcome:
        self.acts.append((alarm_id, action, minutes, endpoint))
        if self.act_error is not None:
            raise self.act_error
        if action == "dismiss":
            if alarm_id in self.missed:
                self.missed.discard(alarm_id)
                return ToolOutcome(status="ok", summary="Missed alarm dismissed.")
            return ToolOutcome(
                status="invalid_args",
                summary=f"Alarm {alarm_id} is not missed.",
                reason=f"alarm {alarm_id} is not missed",
            )
        if alarm_id in self.ringing:
            self.ringing.discard(alarm_id)
            return ToolOutcome(status="ok", summary=f"Alarm {action} applied.")
        return ToolOutcome(
            status="invalid_args",
            summary=f"Alarm {alarm_id} is not ringing.",
            reason=f"alarm {alarm_id} is not ringing",
        )

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
        self.reports.append({"alarm_id": alarm_id, "endpoint": endpoint, "outcome": outcome})
        if alarm_id in self.ringing or alarm_id in self.missed:
            return "applied"
        return "unknown"


class DecidingHandler(RecordingConversationHandler):
    """A conversation handler whose `decide_proposal` answers as scripted and records its calls.

    `decide_error` is raised instead of answering when set. When `decide_gate` is set the call
    waits on it first, and `decide_cancelled` records that the wait was cancelled.
    """

    def __init__(self) -> None:
        super().__init__()
        self.decisions: list[tuple[str, str, str]] = []
        self.decide_payloads: list[dict[str, object]] = []
        self.decide_error: BaseException | None = None
        self.decide_gate: asyncio.Event | None = None
        self.decide_started = asyncio.Event()
        self.decide_cancelled = False

    async def decide_proposal(
        self, session_id: str, proposal_id: str, decision: Literal["approve", "reject"]
    ) -> list[dict[str, object]]:
        self.decisions.append((session_id, proposal_id, decision))
        self.decide_started.set()
        if self.decide_gate is not None:
            try:
                await self.decide_gate.wait()
            except asyncio.CancelledError:
                self.decide_cancelled = True
                raise
        if self.decide_error is not None:
            raise self.decide_error
        return list(self.decide_payloads)


class Rig:
    """A real host over an `alarms` fake capability and a scripted alarm control."""

    def __init__(self) -> None:
        self.alarms = FakeCapability("alarms")
        self.control = ScriptedAlarmControl()
        self.secrets = SecretRegistry()
        self.secrets.register(Secret(TOKEN))
        counter = ApproximateTokenCounter()
        capabilities = [self.alarms]
        self.host = CapabilityHost(
            capabilities,
            router=ToolRouter(
                capabilities, RouterConfig(token_budget=1200, reserved_tokens=100), counter, None
            ),
            counter=counter,
            policy=PolicyTable.from_specs(list(self.alarms.tools()), None),
            decoder=ActionDecoder(),
            secrets=self.secrets,
            action_log=ActionLog(CapabilitiesDb(":memory:")),
            event_sink=RecordingEventSink(),
            # The real clock's time, so a report valid on arrival is valid for the host too.
            clock=FakeClock(datetime.now(UTC), settle_timeout_s=0.3),
            settings=HostSettings(),
            alarm_control=self.control,
        )


@pytest_asyncio.fixture
async def rig(monkeypatch):
    """Install a host as `lifespan` would, and a fresh broadcast queue; stop the host after."""
    from backend import server

    built = Rig()
    monkeypatch.setattr(server, "_capability_build", CapabilityBuild(built.host, None))
    monkeypatch.setattr(server, "message_queue", asyncio.Queue())
    yield built
    await built.host.stop()


@pytest.fixture
def no_host(monkeypatch):
    """The disabled build `lifespan` leaves when capabilities are off: no host."""
    from backend import server

    monkeypatch.setattr(server, "_capability_build", CapabilityBuild(None, "capabilities disabled"))
    monkeypatch.setattr(server, "message_queue", asyncio.Queue())


# -- Helpers ---------------------------------------------------------------------------------


def _after_handshake(ws: FakeWebSocket) -> list[dict]:
    """The frames sent after `session_started`, which must be the first frame."""
    assert ws.sent[0]["type"] == "session_started", ws.sent
    return ws.sent[1:]


def _of_type(frames: list[dict], event_type: str) -> list[dict]:
    return [frame for frame in frames if frame.get("type") == event_type]


def _without_snapshot(frames: list[dict]) -> list[dict]:
    return [f for f in frames if f.get("type") not in ("capabilities_status", "alarm_state")]


def _broadcast() -> list[dict]:
    """Every payload queued for broadcast, decoded."""
    from backend import server

    queued = []
    while not server.message_queue.empty():
        queued.append(json.loads(server.message_queue.get_nowait()))
    return queued


def _alarm_action(**fields: object) -> dict[str, object]:
    message: dict[str, object] = {"type": "alarm_action", "alarm_id": "a-ringing", "action": "stop"}
    message.update(fields)
    return message


def _alarm_report(**fields: object) -> dict[str, object]:
    at = (datetime.now(UTC) - timedelta(seconds=5)).isoformat()
    message: dict[str, object] = {
        "type": "alarm_report",
        "alarm_id": "a-ringing",
        "outcome": "rang",
        "at": at,
        "rang_at": at,
    }
    message.update(fields)
    return message


async def _run(
    monkeypatch, inbound: list[dict], handler: RecordingConversationHandler | None = None
):
    handler = handler or DecidingHandler()
    vp = build_voice_processor(handler, asyncio.get_running_loop())
    return await run_endpoint(monkeypatch, vp, inbound)


# -- Connect snapshot ------------------------------------------------------------------------


class TestConnectSnapshot:
    @pytest.mark.asyncio
    async def test_status_then_alarm_state_follow_session_started_on_the_socket(
        self, monkeypatch, rig
    ):
        ws = await _run(monkeypatch, [])

        frames = _after_handshake(ws)
        assert [f["type"] for f in frames] == ["capabilities_status", "alarm_state"]
        status, alarm_state = frames
        assert isinstance(status["rev"], int)
        assert alarm_state["rev"] == 7
        assert _broadcast() == [], "the snapshot goes to the connecting socket only"

    @pytest.mark.asyncio
    async def test_status_carries_the_power_level_and_decode_availability(self, monkeypatch, rig):
        ws = await _run(monkeypatch, [])

        (status,) = _of_type(ws.sent, "capabilities_status")
        assert status["power_level"] == "full"
        assert status["decode_available"] is True
        assert [c["name"] for c in status["capabilities"]] == ["alarms"]

    @pytest.mark.asyncio
    async def test_a_snapshot_without_alarm_state_sends_only_the_status(self, monkeypatch, rig):
        rig.control.snapshot_error = CapabilityUnavailableError("scheduler is not running")

        ws = await _run(monkeypatch, [])

        assert [f["type"] for f in _after_handshake(ws)] == ["capabilities_status"]

    @pytest.mark.asyncio
    async def test_a_raising_snapshot_sends_a_redacted_server_error_and_the_loop_goes_on(
        self, monkeypatch, rig
    ):
        rig.control.snapshot_error = RuntimeError(f"store exploded {TOKEN}")

        ws = await _run(monkeypatch, [RESET_VAD])

        frames = _after_handshake(ws)
        assert frames[0] == {
            "type": "error",
            "kind": "server",
            "message": "RuntimeError: store exploded ***",
            "retriable": False,
            "context": None,
        }
        assert TOKEN not in json.dumps(ws.sent)
        assert frames[1:] == [VAD_RESET_ANSWER]

    @pytest.mark.asyncio
    async def test_a_send_to_a_gone_client_takes_the_disconnect_path(
        self, monkeypatch, rig, caplog
    ):
        from backend import server

        class GoneOnSnapshot(FakeWebSocket):
            async def send_json(self, payload: dict) -> None:
                if payload.get("type") == "capabilities_status":
                    raise WebSocketDisconnect(code=1001)
                await super().send_json(payload)

        handler = DecidingHandler()
        vp = build_voice_processor(handler, asyncio.get_running_loop())
        monkeypatch.setattr(server, "voice_processor", vp)
        monkeypatch.setattr(server, "log_handler", None)
        ws = GoneOnSnapshot([RESET_VAD])

        with caplog.at_level(logging.INFO, logger=SERVER_LOGGER):
            await server.websocket_endpoint(ws)

        assert handler.ended_sessions == [ws.handshake_session_id]
        assert ws not in server.active_connections
        assert [f["type"] for f in ws.sent] == ["session_started"]
        assert not [r for r in caplog.records if r.getMessage().startswith("WebSocket error")]


# -- Containment -----------------------------------------------------------------------------


class TestContainment:
    @pytest.mark.asyncio
    async def test_a_raising_decision_is_answered_and_the_loop_goes_on(self, monkeypatch, rig):
        handler = DecidingHandler()
        handler.decide_error = RuntimeError(f"decide blew up {TOKEN}")
        decision = {"type": "action_decision", "proposal_id": "p1", "decision": "approve"}

        ws = await _run(monkeypatch, [decision, RESET_VAD], handler)

        frames = _without_snapshot(_after_handshake(ws))
        assert frames == [
            {
                "type": "error",
                "kind": "server",
                "message": "RuntimeError: decide blew up ***",
                "retriable": False,
                "context": None,
            },
            VAD_RESET_ANSWER,
        ]
        assert handler.ended_sessions == [ws.handshake_session_id]

    @pytest.mark.asyncio
    async def test_the_failure_is_logged_at_error_with_its_traceback(
        self, monkeypatch, rig, caplog
    ):
        handler = DecidingHandler()
        handler.decide_error = RuntimeError("decide blew up")
        decision = {"type": "action_decision", "proposal_id": "p1", "decision": "approve"}

        with caplog.at_level(logging.ERROR, logger=SERVER_LOGGER):
            await _run(monkeypatch, [decision], handler)

        failures = [r for r in caplog.records if r.name == SERVER_LOGGER and r.exc_info]
        assert len(failures) == 1
        assert failures[0].levelno == logging.ERROR
        assert failures[0].exc_info[1] is handler.decide_error
        assert not [r for r in caplog.records if r.getMessage().startswith("WebSocket error")]

    @pytest.mark.asyncio
    async def test_a_cancelled_call_sends_nothing_and_the_loop_goes_on(self, monkeypatch, rig):
        handler = DecidingHandler()
        handler.decide_error = asyncio.CancelledError()
        decision = {"type": "action_decision", "proposal_id": "p1", "decision": "approve"}

        ws = await _run(monkeypatch, [decision, RESET_VAD], handler)

        assert _without_snapshot(_after_handshake(ws)) == [VAD_RESET_ANSWER]
        assert _broadcast() == []

    @pytest.mark.asyncio
    async def test_cancelling_the_endpoint_cancels_the_capability_call(self, monkeypatch, rig):
        from backend import server

        handler = DecidingHandler()
        handler.decide_gate = asyncio.Event()
        vp = build_voice_processor(handler, asyncio.get_running_loop())
        monkeypatch.setattr(server, "voice_processor", vp)
        monkeypatch.setattr(server, "log_handler", None)
        ws = FakeWebSocket([{"type": "action_decision", "proposal_id": "p1", "decision": "reject"}])

        endpoint = asyncio.create_task(server.websocket_endpoint(ws))
        await asyncio.wait_for(handler.decide_started.wait(), timeout=5.0)
        endpoint.cancel()
        with pytest.raises(asyncio.CancelledError):
            await endpoint
        for _ in range(10):
            await asyncio.sleep(0)

        assert handler.decide_cancelled, "the capability call outlived its connection"
        assert _without_snapshot(_after_handshake(ws)) == []

    @pytest.mark.asyncio
    async def test_the_connection_stays_registered_until_the_script_ends(self, monkeypatch, rig):
        from backend import server

        class Tracking(FakeWebSocket):
            def __init__(self, inbound: list[dict]) -> None:
                super().__init__(inbound)
                self.registered: list[bool] = []

            async def receive_json(self) -> dict:
                self.registered.append(self in server.active_connections)
                return await super().receive_json()

        rig.control.act_error = RuntimeError("act blew up")
        handler = DecidingHandler()
        handler.decide_error = RuntimeError("decide blew up")
        vp = build_voice_processor(handler, asyncio.get_running_loop())
        monkeypatch.setattr(server, "voice_processor", vp)
        monkeypatch.setattr(server, "log_handler", None)
        ws = Tracking(
            [
                {"type": "action_decision", "proposal_id": "p1", "decision": "approve"},
                _alarm_action(request_id="r1"),
                RESET_VAD,
            ]
        )

        await server.websocket_endpoint(ws)

        assert ws.registered == [True, True, True, True]
        assert ws not in server.active_connections
        assert handler.ended_sessions == [ws.handshake_session_id]


# -- action_decision -------------------------------------------------------------------------


class TestActionDecision:
    @pytest.mark.asyncio
    async def test_it_decides_for_the_connections_own_session_and_broadcasts(
        self, monkeypatch, rig
    ):
        handler = DecidingHandler()
        resolved = {
            "type": "action_resolved",
            "proposal_id": "p1",
            "session_id": "s",
            "decision": "approved",
            "outcomes": [],
        }
        handler.decide_payloads = [{"type": "tool_call_started", "tool": "x"}, resolved]
        decision = {"type": "action_decision", "proposal_id": "p1", "decision": "approve"}

        ws = await _run(monkeypatch, [decision], handler)

        assert handler.decisions == [(ws.handshake_session_id, "p1", "approve")]
        assert _broadcast() == handler.decide_payloads
        assert _without_snapshot(_after_handshake(ws)) == []

    @pytest.mark.asyncio
    async def test_an_invalid_decision_is_a_validation_error(self, monkeypatch, rig):
        handler = DecidingHandler()
        decision = {"type": "action_decision", "proposal_id": "p1", "decision": "maybe"}

        ws = await _run(monkeypatch, [decision], handler)

        (error,) = _without_snapshot(_after_handshake(ws))
        assert error["type"] == "error"
        assert error["kind"] == "validation"
        assert error["retriable"] is False
        assert "decision" in error["message"]
        assert handler.decisions == []


# -- alarm_action ----------------------------------------------------------------------------


class TestAlarmAction:
    async def _result(self, monkeypatch, message: dict[str, object]) -> dict:
        ws = await _run(monkeypatch, [message, RESET_VAD])
        frames = _without_snapshot(_after_handshake(ws))
        assert frames[-1] == VAD_RESET_ANSWER
        (result,) = frames[:-1]
        assert result["type"] == "alarm_action_result"
        assert _broadcast() == [], "the result goes to the requesting socket only"
        return result

    @pytest.mark.asyncio
    async def test_stop_of_a_ringing_alarm_is_ok_and_names_the_desktop(self, monkeypatch, rig):
        result = await self._result(monkeypatch, _alarm_action(request_id="req-1"))

        assert result["alarm_id"] == "a-ringing"
        assert result["action"] == "stop"
        assert result["request_id"] == "req-1"
        assert result["outcome"] == "ok"
        assert result["reason"] is None
        assert rig.control.acts == [("a-ringing", "stop", None, "desktop")]

    @pytest.mark.asyncio
    async def test_dismiss_of_a_missed_alarm_is_ok(self, monkeypatch, rig):
        result = await self._result(
            monkeypatch, _alarm_action(alarm_id="a-missed", action="dismiss", request_id="r")
        )

        assert (result["action"], result["outcome"], result["reason"]) == ("dismiss", "ok", None)

    @pytest.mark.asyncio
    async def test_dismiss_of_an_alarm_that_is_not_missed_is_invalid(self, monkeypatch, rig):
        result = await self._result(monkeypatch, _alarm_action(action="dismiss", request_id="r"))

        assert result["outcome"] == "invalid_args"
        assert result["reason"] == "alarm a-ringing is not missed"

    @pytest.mark.asyncio
    async def test_an_alarm_that_is_not_ringing_is_invalid(self, monkeypatch, rig):
        result = await self._result(
            monkeypatch, _alarm_action(alarm_id="a-quiet", action="snooze", minutes=5)
        )

        assert result["outcome"] == "invalid_args"
        assert result["reason"] == "alarm a-quiet is not ringing"
        assert rig.control.acts == [("a-quiet", "snooze", 5, "desktop")]

    @pytest.mark.asyncio
    async def test_unavailable_alarms_answer_unavailable_with_their_reason(self, monkeypatch, rig):
        rig.alarms.set_status("unavailable", "alarm scheduler stopped")

        result = await self._result(monkeypatch, _alarm_action(request_id="r"))

        assert result["outcome"] == "unavailable"
        assert result["reason"] == "alarm scheduler stopped"
        assert rig.control.acts == []

    @pytest.mark.asyncio
    async def test_a_store_error_is_failed(self, monkeypatch, rig):
        rig.control.act_error = CapabilityStoreError(f"alarm store locked {TOKEN}")

        result = await self._result(monkeypatch, _alarm_action(request_id="r"))

        assert result["outcome"] == "failed"
        assert result["reason"] == "alarm store locked ***"

    @pytest.mark.asyncio
    async def test_an_unexpected_exception_is_one_failed_result_and_no_error(
        self, monkeypatch, rig
    ):
        rig.control.act_error = RuntimeError(f"act blew up {TOKEN}")

        ws = await _run(monkeypatch, [_alarm_action(request_id="req-9"), RESET_VAD])

        frames = _without_snapshot(_after_handshake(ws))
        assert _of_type(frames, "error") == []
        (result,) = _of_type(frames, "alarm_action_result")
        assert result["request_id"] == "req-9"
        assert result["outcome"] == "failed"
        assert result["reason"] == "RuntimeError: act blew up ***"
        assert TOKEN not in json.dumps(ws.sent)
        assert frames[-1] == VAD_RESET_ANSWER

    @pytest.mark.asyncio
    async def test_no_request_id_key_when_the_client_sent_none(self, monkeypatch, rig):
        result = await self._result(monkeypatch, _alarm_action())

        assert "request_id" not in result
        assert result["outcome"] == "ok"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "change",
        [
            {"request_id": ""},
            {"request_id": "x" * 65},
            {"action": "explode"},
            {"minutes": 0},
            {"alarm_id": None},
        ],
    )
    async def test_an_invalid_action_is_a_validation_error_and_no_result(
        self, monkeypatch, rig, change
    ):
        ws = await _run(monkeypatch, [_alarm_action(**change)])

        (error,) = _without_snapshot(_after_handshake(ws))
        assert error["type"] == "error"
        assert error["kind"] == "validation"
        assert rig.control.acts == []


# -- alarm_report ----------------------------------------------------------------------------


class TestAlarmReport:
    async def _result(self, monkeypatch, message: dict[str, object]) -> dict:
        ws = await _run(monkeypatch, [message, RESET_VAD])
        frames = _without_snapshot(_after_handshake(ws))
        assert frames[-1] == VAD_RESET_ANSWER
        (result,) = frames[:-1]
        assert result["type"] == "alarm_action_result"
        return result

    @pytest.mark.asyncio
    async def test_rang_is_a_ring_acknowledgement_from_the_desktop(self, monkeypatch, rig):
        result = await self._result(monkeypatch, _alarm_report(request_id="rep-1"))

        assert result["action"] == "report"
        assert result["outcome"] == "ok"
        assert result["detail"] == "applied"
        assert result["request_id"] == "rep-1"
        assert rig.control.reports == [
            {"alarm_id": "a-ringing", "endpoint": "desktop", "outcome": "rang"}
        ]

    @pytest.mark.asyncio
    async def test_stopped_on_a_missed_occurrence_is_applied(self, monkeypatch, rig):
        at = (datetime.now(UTC) - timedelta(seconds=5)).isoformat()
        rang_at = (datetime.now(UTC) - timedelta(seconds=60)).isoformat()

        result = await self._result(
            monkeypatch,
            _alarm_report(alarm_id="a-missed", outcome="stopped", at=at, rang_at=rang_at),
        )

        assert (result["outcome"], result["detail"]) == ("ok", "applied")
        assert "request_id" not in result
        assert rig.control.reports[0]["outcome"] == "stopped"

    @pytest.mark.asyncio
    async def test_an_unknown_id_is_invalid(self, monkeypatch, rig):
        result = await self._result(monkeypatch, _alarm_report(alarm_id="a-nowhere"))

        assert result["outcome"] == "invalid_args"
        assert result["detail"] == "unknown"
        assert result["reason"] == "unknown alarm"

    @pytest.mark.asyncio
    async def test_a_payload_endpoint_is_refused_as_an_unknown_field(self, monkeypatch, rig):
        ws = await _run(monkeypatch, [_alarm_report(endpoint="room")])

        (error,) = _without_snapshot(_after_handshake(ws))
        assert (error["type"], error["kind"]) == ("error", "validation")
        assert "endpoint" in error["message"]
        assert rig.control.reports == []

    @pytest.mark.asyncio
    async def test_an_unexpected_exception_is_one_failed_result(self, monkeypatch, rig):
        async def explode(*args: object, **kwargs: object) -> ReportResult:
            raise RuntimeError(f"report blew up {TOKEN}")

        monkeypatch.setattr(rig.control, "report", explode)

        result = await self._result(monkeypatch, _alarm_report(request_id="rep-2"))

        assert result["outcome"] == "failed"
        assert result["request_id"] == "rep-2"
        assert result["reason"] == "RuntimeError: report blew up ***"
        assert "detail" not in result


# -- No host ---------------------------------------------------------------------------------


class TestNoHost:
    @pytest.mark.asyncio
    async def test_nothing_extra_is_sent_on_connect(self, monkeypatch, no_host):
        ws = await _run(monkeypatch, [RESET_VAD])

        assert _after_handshake(ws) == [VAD_RESET_ANSWER]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "message",
        [
            {"type": "action_decision", "proposal_id": "p1", "decision": "approve"},
            {"type": "alarm_action", "alarm_id": "a1", "action": "stop"},
            {"type": "alarm_report", "alarm_id": "a1", "outcome": "rang"},
        ],
    )
    async def test_capability_messages_answer_capabilities_disabled(
        self, monkeypatch, no_host, message
    ):
        handler = DecidingHandler()

        ws = await _run(monkeypatch, [message, RESET_VAD], handler)

        assert _after_handshake(ws) == [
            {
                "type": "error",
                "kind": "validation",
                "message": "capabilities disabled",
                "retriable": False,
                "context": None,
            },
            VAD_RESET_ANSWER,
        ]
        assert handler.decisions == []


# -- The endpoint's docstring ----------------------------------------------------------------


def test_the_endpoint_docstring_lists_the_capability_messages():
    from backend import server

    doc = server.websocket_endpoint.__doc__ or ""
    for name in (
        "action_decision",
        "alarm_action",
        "alarm_report",
        "capabilities_status",
        "alarm_state",
        "alarm_action_result",
        "action_proposed",
        "action_resolved",
        "alarm_fired",
    ):
        assert name in doc, f"{name} is missing from the websocket_endpoint docstring"
