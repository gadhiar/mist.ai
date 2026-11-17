"""Tests for the host's alarm port widened by endpoints: `alarm_report`, the endpoint `act` is
given, and the turn endpoint every `InvocationContext` carries (T23).
"""

from __future__ import annotations

import dataclasses
import logging
from datetime import datetime, timedelta
from typing import Literal

import pytest
import pytest_asyncio

from backend.capabilities.action_log import ActionLog
from backend.capabilities.alarms.records import ReportOutcome, ReportResult
from backend.capabilities.contracts import FastIntent, InvocationContext, ToolOutcome
from backend.capabilities.db import CapabilitiesDb
from backend.capabilities.decoder import ActionDecoder
from backend.capabilities.events import (
    AlarmActionResultEvent,
    AlarmReportMessage,
    alarm_action_result_payload,
)
from backend.capabilities.host import AlarmControl, CapabilityHost, HostSettings
from backend.capabilities.policy import PolicyTable
from backend.capabilities.router import RouterConfig, ToolRouter
from backend.capabilities.secrets import Secret, SecretRegistry
from backend.chat.context_budget import ApproximateTokenCounter
from backend.errors import CapabilityStoreError, CapabilityUnavailableError
from backend.llm.models import ToolCall
from tests.unit.capabilities.fakes import (
    DEFAULT_START,
    FakeCapability,
    FakeClock,
    RecordingEventSink,
    make_tool_spec,
)

LOGGER = "backend.capabilities.host"
TOKEN = "tok-9a8b7c6d"
T0 = DEFAULT_START

AlarmAction = Literal["stop", "snooze", "dismiss"]


class FakeAlarmControl:
    """An `AlarmControl` that records every `act` and `report` and answers as scripted."""

    def __init__(self) -> None:
        self.act_outcome = ToolOutcome(status="ok", summary="Alarm stopped.")
        self.result: ReportResult = "applied"
        self.report_error: BaseException | None = None
        self.acts: list[tuple[str, str, int | None, str]] = []
        self.reports: list[dict[str, object]] = []

    async def snapshot(self) -> dict[str, object]:
        raise CapabilityUnavailableError("no alarm state in these tests")

    async def act(
        self, alarm_id: str, action: AlarmAction, minutes: int | None, *, endpoint: str
    ) -> ToolOutcome:
        self.acts.append((alarm_id, action, minutes, endpoint))
        return self.act_outcome

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
        self.reports.append(
            {
                "alarm_id": alarm_id,
                "endpoint": endpoint,
                "outcome": outcome,
                "at": at,
                "rang_at": rang_at,
                "snoozed_until": snoozed_until,
            }
        )
        if self.report_error is not None:
            raise self.report_error
        return self.result


@dataclasses.dataclass
class Rig:
    host: CapabilityHost
    clock: FakeClock
    secrets: SecretRegistry


_HOSTS: list[CapabilityHost] = []


@pytest_asyncio.fixture(autouse=True)
async def _stop_hosts():
    """Stop every host a test built, so no proposal timer outlives the test."""
    yield
    while _HOSTS:
        await _HOSTS.pop().stop()


def _rig(
    *capabilities: FakeCapability,
    alarm_control: AlarmControl | None = None,
    **host_options: str,
) -> Rig:
    counter = ApproximateTokenCounter()
    router = ToolRouter(
        list(capabilities), RouterConfig(token_budget=1200, reserved_tokens=100), counter, None
    )
    specs = [spec for capability in capabilities for spec in capability.tools()]
    clock = FakeClock(settle_timeout_s=0.3)
    secrets = SecretRegistry()
    host = CapabilityHost(
        list(capabilities),
        router=router,
        counter=counter,
        policy=PolicyTable.from_specs(specs, None),
        decoder=ActionDecoder(),
        secrets=secrets,
        action_log=ActionLog(CapabilitiesDb(":memory:")),
        event_sink=RecordingEventSink(),
        clock=clock,
        settings=HostSettings(),
        alarm_control=alarm_control,
        **host_options,  # type: ignore[arg-type]
    )
    _HOSTS.append(host)
    return Rig(host, clock, secrets)


def _report(**changes: object) -> AlarmReportMessage:
    message: dict[str, object] = {
        "type": "alarm_report",
        "alarm_id": "a1",
        "outcome": "rang",
        "at": "2026-01-05T06:59:30Z",
        "rang_at": "2026-01-05T06:59:30Z",
    }
    message.update(changes)
    return AlarmReportMessage.model_validate(message, context={"now": T0})


class TestAct:
    @pytest.mark.parametrize("endpoint", ["desktop", "room"])
    @pytest.mark.asyncio
    async def test_alarm_action_passes_the_endpoint_to_act(self, endpoint):
        control = FakeAlarmControl()
        rig = _rig(FakeCapability("alarms"), alarm_control=control)

        outcome = await rig.host.alarm_action("a1", "snooze", 5, endpoint=endpoint)

        assert outcome.status == "ok"
        assert control.acts == [("a1", "snooze", 5, endpoint)]


class TestAlarmReportGate:
    @pytest.mark.parametrize(
        ("state", "reason", "expected"),
        [
            ("unavailable", "scheduler failing", "scheduler failing"),
            ("disabled", None, "disabled"),
        ],
    )
    @pytest.mark.asyncio
    async def test_an_alarms_status_that_refuses_answers_without_calling_report(
        self, state, reason, expected
    ):
        alarms = FakeCapability("alarms")
        alarms.set_status(state, reason)
        control = FakeAlarmControl()
        rig = _rig(alarms, alarm_control=control)

        outcome = await rig.host.alarm_report(_report(), endpoint="desktop")

        assert (outcome.status, outcome.reason) == ("unavailable", expected)
        assert control.reports == []

    @pytest.mark.asyncio
    async def test_timers_gate_when_alarms_is_not_built(self):
        timers = FakeCapability("timers")
        timers.set_status("unavailable", "MIST_TIMEZONE unset")
        control = FakeAlarmControl()
        rig = _rig(timers, alarm_control=control)

        outcome = await rig.host.alarm_report(_report(), endpoint="desktop")

        assert (outcome.status, outcome.reason) == ("unavailable", "MIST_TIMEZONE unset")
        assert control.reports == []

    @pytest.mark.asyncio
    async def test_neither_alarms_nor_timers_built(self):
        rig = _rig(FakeCapability("clock"))

        outcome = await rig.host.alarm_report(_report(), endpoint="desktop")

        assert (outcome.status, outcome.reason) == (
            "unavailable",
            "alarms and timers are not in MIST_CAPABILITIES",
        )

    @pytest.mark.asyncio
    async def test_available_alarms_with_no_control_is_not_wired(self):
        rig = _rig(FakeCapability("alarms"))

        outcome = await rig.host.alarm_report(_report(), endpoint="desktop")

        assert (outcome.status, outcome.reason) == ("unavailable", "alarm control is not wired")


class TestAlarmReport:
    @pytest.mark.asyncio
    async def test_passes_the_report_and_the_connections_endpoint_to_the_control(self):
        control = FakeAlarmControl()
        rig = _rig(FakeCapability("alarms"), alarm_control=control)
        message = _report(
            outcome="snoozed",
            rang_at="2026-01-05T06:50:00Z",
            snoozed_until="2026-01-05T07:09:00Z",
        )

        await rig.host.alarm_report(message, endpoint="room")

        assert control.reports == [
            {
                "alarm_id": "a1",
                "endpoint": "room",
                "outcome": "snoozed",
                "at": message.at,
                "rang_at": message.rang_at,
                "snoozed_until": message.snoozed_until,
            }
        ]

    @pytest.mark.parametrize(
        ("result", "status"),
        [
            ("applied", "ok"),
            ("late", "ok"),
            ("already_ended", "ok"),
            ("duplicate", "ok"),
            ("not_due", "invalid_args"),
            ("unknown", "invalid_args"),
        ],
    )
    @pytest.mark.asyncio
    async def test_the_report_result_is_in_the_outcomes_data(self, result, status):
        control = FakeAlarmControl()
        control.result = result
        rig = _rig(FakeCapability("alarms"), alarm_control=control)

        outcome = await rig.host.alarm_report(_report(), endpoint="desktop")

        assert outcome.status == status
        assert outcome.data == {"result": result}
        assert (outcome.reason is None) is (status == "ok")

    @pytest.mark.parametrize(
        ("result", "wire"), [("applied", "ok"), ("late", "ok"), ("not_due", "invalid_args")]
    )
    @pytest.mark.asyncio
    async def test_the_outcome_builds_an_alarm_action_result_with_detail(self, result, wire):
        control = FakeAlarmControl()
        control.result = result
        rig = _rig(FakeCapability("alarms"), alarm_control=control)
        message = _report(request_id="q7")

        outcome = await rig.host.alarm_report(message, endpoint="desktop")
        payload = alarm_action_result_payload(
            message.alarm_id, "report", message.request_id, outcome, T0
        )

        assert (payload["action"], payload["outcome"], payload["detail"]) == (
            "report",
            wire,
            result,
        )
        assert payload["request_id"] == "q7"
        assert AlarmActionResultEvent.model_validate(payload).to_payload() == payload

    @pytest.mark.asyncio
    async def test_a_report_outside_the_window_by_the_hosts_clock_is_refused(self):
        control = FakeAlarmControl()
        rig = _rig(FakeCapability("alarms"), alarm_control=control)
        message = _report()
        await rig.clock.advance(timedelta(days=2).total_seconds())

        outcome = await rig.host.alarm_report(message, endpoint="desktop")

        assert outcome.status == "invalid_args"
        assert outcome.reason is not None and "24 h" in outcome.reason
        assert outcome.data is None
        assert control.reports == []

    @pytest.mark.asyncio
    async def test_a_store_error_is_a_redacted_error_outcome(self):
        control = FakeAlarmControl()
        control.report_error = CapabilityStoreError(f"alarm store locked {TOKEN}")
        rig = _rig(FakeCapability("alarms"), alarm_control=control)
        rig.secrets.register(Secret(TOKEN))

        outcome = await rig.host.alarm_report(_report(), endpoint="desktop")

        assert outcome.status == "error"
        assert outcome.reason == "alarm store locked ***"
        assert TOKEN not in outcome.summary

    @pytest.mark.asyncio
    async def test_an_unavailable_error_is_an_unavailable_outcome(self):
        control = FakeAlarmControl()
        control.report_error = CapabilityUnavailableError("scheduler is dead")
        rig = _rig(FakeCapability("alarms"), alarm_control=control)

        outcome = await rig.host.alarm_report(_report(), endpoint="desktop")

        assert (outcome.status, outcome.reason) == ("unavailable", "scheduler is dead")

    @pytest.mark.asyncio
    async def test_another_exception_propagates(self):
        control = FakeAlarmControl()
        control.report_error = RuntimeError("bug in report")
        rig = _rig(FakeCapability("alarms"), alarm_control=control)

        with pytest.raises(RuntimeError, match="bug in report"):
            await rig.host.alarm_report(_report(), endpoint="desktop")

    @pytest.mark.asyncio
    async def test_logs_one_info_line_naming_the_alarm_endpoint_and_outcome(self, caplog):
        rig = _rig(FakeCapability("alarms"), alarm_control=FakeAlarmControl())
        caplog.set_level(logging.INFO, logger=LOGGER)

        await rig.host.alarm_report(_report(alarm_id="alarm-42"), endpoint="desktop")

        infos = [r for r in caplog.records if r.name == LOGGER and r.levelno == logging.INFO]
        assert len(infos) == 1
        message = infos[0].getMessage()
        assert "alarm-42" in message and "desktop" in message and "rang" in message


class TestTurnEndpoint:
    @staticmethod
    def _lights() -> FakeCapability:
        return FakeCapability(
            "lights", intents={"lights on": FastIntent(tool="lights.do", args={})}
        )

    @staticmethod
    def _ctx(endpoint: str | None = None) -> InvocationContext:
        return InvocationContext("s1", "t1", "lights on", "c1", "model", T0, endpoint)

    @pytest.mark.asyncio
    async def test_the_default_turn_endpoint_is_desktop_on_a_fast_path_call(self):
        lights = self._lights()
        rig = _rig(lights)

        resolution = await rig.host.pre_llm("s1", "t1", "lights on")

        assert resolution is not None and resolution.handled
        [(_, _, ctx)] = lights.invocations
        assert (ctx.source, ctx.endpoint) == ("fastpath", "desktop")

    @pytest.mark.asyncio
    async def test_a_configured_turn_endpoint_fills_every_call(self):
        lights = FakeCapability(
            "lights",
            tools=[
                make_tool_spec("lights.do"),
                make_tool_spec("lights.all_off", default_policy="confirm"),
            ],
            intents={"lights on": FastIntent(tool="lights.do", args={})},
        )
        rig = _rig(lights, turn_endpoint="room")

        await rig.host.pre_llm("s1", "t1", "lights on")
        await rig.host.invoke(ToolCall(id="c1", name="lights.do", arguments={}), self._ctx(), None)
        pending = await rig.host.invoke(
            ToolCall(id="c2", name="lights.all_off", arguments={}), self._ctx(), None
        )
        assert pending.outcome.proposal_id is not None
        await rig.host.decide("s1", pending.outcome.proposal_id, "approve")

        assert [(ctx.source, ctx.endpoint) for _, _, ctx in lights.invocations] == [
            ("fastpath", "room"),
            ("model", "room"),
            ("proposal", "room"),
        ]

    @pytest.mark.asyncio
    async def test_a_call_whose_context_names_an_endpoint_keeps_it(self):
        lights = self._lights()
        rig = _rig(lights)

        await rig.host.invoke(
            ToolCall(id="c1", name="lights.do", arguments={}), self._ctx("room"), None
        )

        [(_, _, ctx)] = lights.invocations
        assert ctx.endpoint == "room"

    def test_the_context_endpoint_defaults_to_none(self):
        ctx = InvocationContext("s1", "t1", "x", "c1", "model", T0)

        assert ctx.endpoint is None
