"""Tests for the host's `capabilities_status` broadcast, its watcher, the connect snapshot and
alarm actions (build T19, amendment A2).
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections import deque
from collections.abc import Sequence
from datetime import timedelta
from typing import Literal

import pytest
import pytest_asyncio

from backend.capabilities import host as host_module
from backend.capabilities.action_log import ActionLog
from backend.capabilities.contracts import ToolOutcome, ToolSpec
from backend.capabilities.db import CapabilitiesDb
from backend.capabilities.decoder import ActionDecoder
from backend.capabilities.events import AlarmStateEvent, CapabilitiesStatusEvent
from backend.capabilities.host import AlarmControl, CapabilityHost, HostSettings
from backend.capabilities.policy import PolicyTable
from backend.capabilities.router import RouterConfig, ToolRouter
from backend.capabilities.secrets import Secret, SecretRegistry
from backend.chat.context_budget import ApproximateTokenCounter
from backend.errors import (
    CapabilityInvocationError,
    CapabilityStoreError,
    CapabilityUnavailableError,
    MistError,
)
from tests.unit.capabilities.fakes import (
    DEFAULT_START,
    FakeCapability,
    FakeClock,
    RecordingEventSink,
    make_tool_spec,
)

LOGGER = "backend.capabilities.host"
TOKEN = "tok-5f1e2d9c"
STATUS = "capabilities_status"

AlarmAction = Literal["stop", "snooze", "dismiss"]


# -- Fakes -----------------------------------------------------------------------------------


class Tracked(FakeCapability):
    """A fake whose status follows its lifecycle, and whose `status` can be made to raise.

    It reads `unavailable` "starting" until `start` finishes and `unavailable` "stopped" once
    `stop` runs. Each `status()` pops one error from `status_errors` and raises it while any
    remain.
    """

    def __init__(self, name: str, tools: Sequence[ToolSpec] | None = None) -> None:
        super().__init__(name, tools=tools)
        self.set_status("unavailable", "starting")
        self.status_errors: deque[BaseException] = deque()

    async def start(self) -> None:
        await super().start()
        self.set_status("available")

    async def stop(self) -> None:
        await super().stop()
        self.set_status("unavailable", "stopped")

    def status(self):
        if self.status_errors:
            raise self.status_errors.popleft()
        return super().status()


def _alarm_state(rev: int = 4) -> dict[str, object]:
    return {
        "type": "alarm_state",
        "rev": rev,
        "alarms": [
            {
                "alarm_id": "a1",
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
        "ringing": [{"alarm_id": "a1", "late_by_s": 0.0}],
        "missed": [],
        "fallback_after_s": 30.0,
        "missed_grace_s": 900.0,
        "ring_timeout_s": 600.0,
        "snooze_minutes": 9,
        "tz": "UTC",
        "at": "2026-01-05T07:00:00Z",
    }


class FakeAlarmControl:
    """An `AlarmControl` scripted through attributes, recording each `act`."""

    def __init__(self) -> None:
        self.state = _alarm_state()
        self.snapshot_error: BaseException | None = None
        self.outcome = ToolOutcome(status="ok", summary="Alarm stopped.")
        self.act_error: BaseException | None = None
        self.acts: list[tuple[str, str, int | None]] = []

    async def snapshot(self) -> dict[str, object]:
        if self.snapshot_error is not None:
            raise self.snapshot_error
        return dict(self.state)

    async def act(self, alarm_id: str, action: AlarmAction, minutes: int | None) -> ToolOutcome:
        self.acts.append((alarm_id, action, minutes))
        if self.act_error is not None:
            raise self.act_error
        return self.outcome


class FailingSleepClock(FakeClock):
    """A `FakeClock` whose next `sleep` raises `sleep_error` once, when it is set.

    One-shot, so a watcher that wrongly contained the error would sleep again and keep
    running, failing the test rather than spinning.
    """

    def __init__(self) -> None:
        super().__init__()
        self.sleep_error: BaseException | None = None

    async def sleep(self, seconds: float) -> None:
        error, self.sleep_error = self.sleep_error, None
        if error is not None:
            raise error
        await super().sleep(seconds)


# -- The rig ---------------------------------------------------------------------------------


@dataclasses.dataclass
class Rig:
    host: CapabilityHost
    sink: RecordingEventSink
    clock: FakeClock
    secrets: SecretRegistry
    db: CapabilitiesDb


_HOSTS: list[CapabilityHost] = []


@pytest_asyncio.fixture(autouse=True)
async def _stop_hosts():
    """Stop every host a test built, so neither the watcher nor a timer outlives the test."""
    yield
    while _HOSTS:
        await _HOSTS.pop().stop()


def _rig(
    *capabilities: FakeCapability,
    alarm_control: AlarmControl | None = None,
    clock: FakeClock | None = None,
) -> Rig:
    counter = ApproximateTokenCounter()
    router = ToolRouter(
        list(capabilities), RouterConfig(token_budget=1200, reserved_tokens=100), counter, None
    )
    specs = [spec for capability in capabilities for spec in capability.tools()]
    sink = RecordingEventSink()
    clock = clock or FakeClock(settle_timeout_s=0.3)
    secrets = SecretRegistry()
    db = CapabilitiesDb(":memory:")
    host = CapabilityHost(
        list(capabilities),
        router=router,
        counter=counter,
        policy=PolicyTable.from_specs(specs, None),
        decoder=ActionDecoder(),
        secrets=secrets,
        action_log=ActionLog(db),
        event_sink=sink,
        clock=clock,
        settings=HostSettings(),
        alarm_control=alarm_control,
    )
    _HOSTS.append(host)
    return Rig(host, sink, clock, secrets, db)


async def _settle() -> None:
    for _ in range(10):
        await asyncio.sleep(0)


async def _started(rig: Rig) -> None:
    await rig.host.start()
    await rig.host.wait_started()
    await _settle()


def _entries(payload: dict[str, object]) -> list[tuple[object, ...]]:
    capabilities = payload["capabilities"]
    assert isinstance(capabilities, list)
    return [(c["name"], c["state"], c["reason"], tuple(c["tools"])) for c in capabilities]


def _records(caplog: pytest.LogCaptureFixture, level: int) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == LOGGER and r.levelno == level]


def _traceback_text(record: logging.LogRecord) -> str:
    if record.exc_text:
        return record.exc_text
    if record.exc_info:
        return logging.Formatter().formatException(record.exc_info)
    return ""


async def _log_rows(db: CapabilitiesDb) -> list[tuple[object, ...]]:
    def read(conn) -> list[tuple[object, ...]]:
        table = "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'action_log'"
        if conn.execute(table).fetchone() is None:
            return []
        return conn.execute("SELECT tool FROM action_log").fetchall()

    return await db.run(read)


# -- status_changed and the transition sites -------------------------------------------------


class TestStatusBroadcast:
    @pytest.mark.asyncio
    async def test_finished_start_is_broadcast_without_advancing_the_clock(self):
        lights = Tracked("lights")
        lights.start_gate = asyncio.Event()
        rig = _rig(lights)

        await rig.host.start()
        await _settle()
        assert rig.sink.of_type(STATUS) == []

        lights.start_gate.set()
        await rig.host.wait_started()
        await _settle()

        (payload,) = rig.sink.of_type(STATUS)
        assert CapabilitiesStatusEvent.model_validate(payload).to_payload() == payload
        assert payload["rev"] == 1
        assert _entries(payload) == [("lights", "available", None, ("lights.do",))]
        assert rig.clock.now() == DEFAULT_START

    @pytest.mark.asyncio
    async def test_start_raising_unavailable_is_broadcast_with_its_reason(self):
        broken = Tracked("broken")
        broken.start_error = CapabilityUnavailableError("no token")
        rig = _rig(broken)

        await _started(rig)

        (payload,) = rig.sink.of_type(STATUS)
        assert _entries(payload) == [("broken", "unavailable", "no token", ("broken.do",))]
        assert rig.clock.now() == DEFAULT_START

    @pytest.mark.asyncio
    async def test_stop_emits_one_status_with_the_stopped_states(self):
        rig = _rig(Tracked("lights"), Tracked("music"))
        await _started(rig)
        before = rig.sink.of_type(STATUS)

        await rig.host.stop()

        after = rig.sink.of_type(STATUS)[len(before) :]
        (payload,) = after
        assert payload["rev"] == before[-1]["rev"] + 1
        assert _entries(payload) == [
            ("lights", "unavailable", "stopped", ("lights.do",)),
            ("music", "unavailable", "stopped", ("music.do",)),
        ]

    @pytest.mark.asyncio
    async def test_stop_broadcasts_a_start_it_cancelled(self):
        slow = Tracked("slow")
        slow.start_gate = asyncio.Event()
        rig = _rig(slow)
        await rig.host.start()
        await _settle()

        await rig.host.stop()

        (payload,) = rig.sink.of_type(STATUS)
        assert _entries(payload) == [("slow", "unavailable", "start was cancelled", ("slow.do",))]

    def test_second_call_with_nothing_changed_emits_nothing(self):
        lights = FakeCapability("lights")
        rig = _rig(lights)

        rig.host.status_changed()
        rig.host.status_changed()

        assert len(rig.sink.of_type(STATUS)) == 1
        lights.set_status("degraded", "slow bridge")
        rig.host.status_changed()
        rig.host.status_changed()
        payloads = rig.sink.of_type(STATUS)
        assert [p["rev"] for p in payloads] == [1, 2]
        assert _entries(payloads[1]) == [("lights", "degraded", "slow bridge", ("lights.do",))]

    @pytest.mark.asyncio
    async def test_at_and_checked_at_are_not_compared(self):
        lights = FakeCapability("lights")
        rig = _rig(lights)
        rig.host.status_changed()

        lights.current_status = dataclasses.replace(
            lights.current_status, checked_at=DEFAULT_START + timedelta(minutes=5)
        )
        await rig.clock.advance(300)
        rig.host.status_changed()

        assert len(rig.sink.of_type(STATUS)) == 1

    def test_reason_and_tools_changes_are_broadcast(self):
        lights = FakeCapability("lights")
        rig = _rig(lights)
        rig.host.status_changed()

        lights.set_status("available", "note")
        rig.host.status_changed()
        lights.tool_specs = (make_tool_spec("lights.do"), make_tool_spec("lights.dim"))
        rig.host.status_changed()

        payloads = rig.sink.of_type(STATUS)
        assert [p["rev"] for p in payloads] == [1, 2, 3]
        assert _entries(payloads[2]) == [
            ("lights", "available", "note", ("lights.do", "lights.dim"))
        ]

    def test_capabilities_are_listed_in_name_order(self):
        rig = _rig(FakeCapability("music"), FakeCapability("clock"), FakeCapability("lights"))

        rig.host.status_changed()

        (payload,) = rig.sink.of_type(STATUS)
        assert [entry[0] for entry in _entries(payload)] == ["clock", "lights", "music"]

    def test_reason_is_redacted(self):
        lights = FakeCapability("lights")
        rig = _rig(lights)
        rig.secrets.register(Secret(TOKEN))
        lights.set_status("unavailable", f"401 for {TOKEN}")

        rig.host.status_changed()

        (payload,) = rig.sink.of_type(STATUS)
        assert _entries(payload) == [("lights", "unavailable", "401 for ***", ("lights.do",))]


# -- The reconciling watcher -----------------------------------------------------------------


class TestWatcher:
    @pytest.mark.asyncio
    async def test_change_only_the_poll_can_see_is_broadcast_at_the_next_tick(self):
        lights = Tracked("lights")
        rig = _rig(lights)
        await _started(rig)
        count = len(rig.sink.of_type(STATUS))

        lights.set_status("unavailable", "scheduler failing")
        await _settle()
        assert len(rig.sink.of_type(STATUS)) == count

        await rig.clock.advance(1.0)

        payloads = rig.sink.of_type(STATUS)
        assert len(payloads) == count + 1
        assert _entries(payloads[-1]) == [
            ("lights", "unavailable", "scheduler failing", ("lights.do",))
        ]
        assert rig.clock.stalled == []

    @pytest.mark.asyncio
    async def test_unchanged_tick_emits_nothing(self):
        rig = _rig(Tracked("lights"))
        await _started(rig)
        count = len(rig.sink.of_type(STATUS))

        for _ in range(3):
            await rig.clock.advance(1.0)

        assert len(rig.sink.of_type(STATUS)) == count
        assert rig.clock.sleeps == [1.0, 1.0, 1.0, 1.0]

    @pytest.mark.asyncio
    async def test_transition_site_change_then_tick_emits_once(self):
        lights = Tracked("lights")
        rig = _rig(lights)
        await _started(rig)
        count = len(rig.sink.of_type(STATUS))

        lights.set_status("degraded", "slow")
        rig.host.status_changed()
        await rig.clock.advance(1.0)

        assert len(rig.sink.of_type(STATUS)) == count + 1

    def test_contained_set_is_mist_error(self):
        assert (MistError,) == host_module._WATCHER_CONTAINED_ERRORS

    @pytest.mark.asyncio
    async def test_failing_ticks_back_off_doubling_to_60_seconds(self, caplog):
        lights = Tracked("lights")
        rig = _rig(lights)
        await _started(rig)
        caplog.set_level(logging.WARNING, logger=LOGGER)
        lights.status_errors.extend(CapabilityInvocationError("status broke") for _ in range(8))

        for _ in range(8):
            await rig.clock.advance(rig.clock.sleeps[-1])

        assert rig.clock.sleeps == [1.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0]
        assert len(_records(caplog, logging.WARNING)) == 8
        assert rig.host.stats()["status_watcher"] == "running"

        lights.set_status("degraded", "recovered")
        await rig.clock.advance(60.0)

        assert rig.clock.sleeps[-1] == 1.0
        assert _entries(rig.sink.of_type(STATUS)[-1])[0][1:3] == ("degraded", "recovered")
        assert _records(caplog, logging.ERROR) == []

    @pytest.mark.asyncio
    async def test_sleep_runs_outside_the_contained_try(self, caplog):
        clock = FailingSleepClock()
        rig = _rig(Tracked("lights"), clock=clock)
        caplog.set_level(logging.WARNING, logger=LOGGER)
        await _started(rig)

        # A contained error from the sleep itself is not a failed tick: it ends the watcher.
        clock.sleep_error = CapabilityInvocationError("sleep broke")
        await clock.advance(1.0)
        await _settle()

        assert rig.host.stats()["status_watcher"] == "dead"
        (error,) = _records(caplog, logging.ERROR)
        assert "sleep broke" in _traceback_text(error)

    @pytest.mark.asyncio
    async def test_stop_is_orderly(self, caplog):
        rig = _rig(Tracked("lights"))
        caplog.set_level(logging.WARNING, logger=LOGGER)
        assert rig.host.stats()["status_watcher"] == "stopped"
        await _started(rig)
        assert rig.host.stats()["status_watcher"] == "running"

        await rig.host.stop()

        assert rig.host.stats()["status_watcher"] == "stopped"
        assert _records(caplog, logging.ERROR) == []
        assert _records(caplog, logging.WARNING) == []

    @pytest.mark.asyncio
    async def test_cancel_without_stop_is_a_death(self, caplog):
        rig = _rig(Tracked("lights"))
        caplog.set_level(logging.WARNING, logger=LOGGER)
        await _started(rig)
        (watcher,) = (t for t in asyncio.all_tasks() if t.get_name() == "capability-status-watcher")

        watcher.cancel()
        await _settle()

        assert rig.host.stats()["status_watcher"] == "dead"
        (error,) = _records(caplog, logging.ERROR)
        assert "cancelled without stop()" in error.getMessage()

    @pytest.mark.asyncio
    async def test_uncontained_exception_ends_the_watcher_visibly(self, caplog):
        lights = Tracked("lights")
        rig = _rig(lights)
        caplog.set_level(logging.WARNING, logger=LOGGER)
        await _started(rig)
        lights.status_errors.append(RuntimeError("bug in status"))

        await rig.clock.advance(1.0)
        await _settle()

        assert rig.host.stats()["status_watcher"] == "dead"
        (error,) = _records(caplog, logging.ERROR)
        assert "RuntimeError: bug in status" in _traceback_text(error)
        await rig.clock.advance(5.0)
        assert rig.clock.sleeps == [1.0]

    @pytest.mark.asyncio
    async def test_dead_watcher_leaves_transition_site_broadcasts_working(self):
        lights = Tracked("lights")
        rig = _rig(lights)
        await _started(rig)
        lights.status_errors.append(RuntimeError("bug in status"))
        await rig.clock.advance(1.0)
        await _settle()
        assert rig.host.stats()["status_watcher"] == "dead"
        count = len(rig.sink.of_type(STATUS))

        lights.set_status("degraded", "slow")
        rig.host.status_changed()
        await rig.host.stop()

        payloads = rig.sink.of_type(STATUS)[count:]
        assert [_entries(p)[0][1:3] for p in payloads] == [
            ("degraded", "slow"),
            ("unavailable", "stopped"),
        ]
        assert rig.host.stats()["status_watcher"] == "dead"


# -- snapshot_events -------------------------------------------------------------------------


class TestSnapshotEvents:
    @pytest.mark.asyncio
    async def test_with_alarm_control_gives_status_then_alarm_state(self):
        control = FakeAlarmControl()
        rig = _rig(FakeCapability("alarms"), alarm_control=control)

        events = await rig.host.snapshot_events()

        status, alarms = events
        assert CapabilitiesStatusEvent.model_validate(status).to_payload() == status
        assert AlarmStateEvent.model_validate(alarms).to_payload() == alarms
        assert alarms == _alarm_state()
        assert status["rev"] == 1
        # `status_changed` ran first, so the snapshot's rev is the one just broadcast.
        assert [p["rev"] for p in rig.sink.of_type(STATUS)] == [1]

    @pytest.mark.asyncio
    async def test_without_alarm_control_gives_status_only(self):
        rig = _rig(FakeCapability("lights"))
        rig.host.status_changed()

        (status,) = await rig.host.snapshot_events()

        assert status["type"] == STATUS
        assert status["rev"] == 1
        assert _entries(status) == [("lights", "available", None, ("lights.do",))]
        assert len(rig.sink.of_type(STATUS)) == 1

    @pytest.mark.asyncio
    async def test_snapshot_carries_a_change_since_the_last_broadcast(self):
        lights = FakeCapability("lights")
        rig = _rig(lights)
        rig.host.status_changed()
        lights.set_status("unavailable", "bridge down")

        (status,) = await rig.host.snapshot_events()

        assert status["rev"] == 2
        assert _entries(status) == [("lights", "unavailable", "bridge down", ("lights.do",))]

    @pytest.mark.parametrize(
        "error",
        [
            CapabilityStoreError(f"alarm store unreadable {TOKEN}"),
            CapabilityUnavailableError(f"scheduler failing {TOKEN}"),
        ],
        ids=["store", "unavailable"],
    )
    @pytest.mark.asyncio
    async def test_capability_error_from_the_alarm_side_leaves_alarm_state_out(self, caplog, error):
        control = FakeAlarmControl()
        control.snapshot_error = error
        rig = _rig(FakeCapability("alarms"), alarm_control=control)
        rig.secrets.register(Secret(TOKEN))
        caplog.set_level(logging.WARNING, logger=LOGGER)

        (status,) = await rig.host.snapshot_events()

        assert status["type"] == STATUS
        (warning,) = _records(caplog, logging.WARNING)
        message = warning.getMessage()
        assert "***" in message
        assert TOKEN not in message

    @pytest.mark.asyncio
    async def test_other_exception_from_the_alarm_side_propagates(self):
        control = FakeAlarmControl()
        control.snapshot_error = RuntimeError("bug in snapshot")
        rig = _rig(FakeCapability("alarms"), alarm_control=control)

        with pytest.raises(RuntimeError, match="bug in snapshot"):
            await rig.host.snapshot_events()


# -- alarm_action ----------------------------------------------------------------------------


class TestAlarmAction:
    @pytest.mark.asyncio
    async def test_unavailable_alarms_without_control_answer_with_the_status_reason(self):
        alarms = FakeCapability("alarms")
        alarms.set_status("unavailable", "MIST_TIMEZONE unset")
        rig = _rig(alarms, FakeCapability("clock"))

        outcome = await rig.host.alarm_action("a1", "stop", None)

        assert outcome.status == "unavailable"
        assert outcome.reason == "MIST_TIMEZONE unset"
        assert outcome.summary == "Alarms and timers are unavailable: MIST_TIMEZONE unset."
        assert "not enabled" not in outcome.summary + (outcome.reason or "")

    @pytest.mark.asyncio
    async def test_unavailable_alarms_with_control_answer_without_calling_act(self):
        alarms = FakeCapability("alarms")
        alarms.set_status("unavailable", "scheduler failing: disk I/O error")
        control = FakeAlarmControl()
        rig = _rig(alarms, alarm_control=control)

        outcome = await rig.host.alarm_action("a1", "snooze", 5)

        assert (outcome.status, outcome.reason) == (
            "unavailable",
            "scheduler failing: disk I/O error",
        )
        assert control.acts == []

    @pytest.mark.asyncio
    async def test_disabled_without_a_reason_answers_with_the_state(self):
        alarms = FakeCapability("alarms")
        alarms.set_status("disabled")
        control = FakeAlarmControl()
        rig = _rig(alarms, alarm_control=control)

        outcome = await rig.host.alarm_action("a1", "stop", None)

        assert (outcome.status, outcome.reason) == ("unavailable", "disabled")
        assert outcome.summary == "Alarms and timers are disabled: disabled."
        assert control.acts == []

    @pytest.mark.asyncio
    async def test_failed_start_of_alarms_is_the_gate_reason(self):
        alarms = FakeCapability("alarms")
        alarms.start_error = CapabilityStoreError("alarm database unwritable")
        control = FakeAlarmControl()
        rig = _rig(alarms, alarm_control=control)
        await _started(rig)

        outcome = await rig.host.alarm_action("a1", "stop", None)

        assert (outcome.status, outcome.reason) == ("unavailable", "alarm database unwritable")
        assert control.acts == []

    @pytest.mark.asyncio
    async def test_timers_gate_when_alarms_is_not_built(self):
        timers = FakeCapability("timers")
        timers.set_status("unavailable", "MIST_TIMEZONE unset")
        control = FakeAlarmControl()
        rig = _rig(timers, alarm_control=control)

        outcome = await rig.host.alarm_action("t1", "stop", None)

        assert (outcome.status, outcome.reason) == ("unavailable", "MIST_TIMEZONE unset")
        assert control.acts == []

    @pytest.mark.asyncio
    async def test_alarms_status_wins_over_timers(self):
        alarms = FakeCapability("alarms")
        alarms.set_status("unavailable", "alarms reason")
        timers = FakeCapability("timers")
        rig = _rig(timers, alarms, alarm_control=FakeAlarmControl())

        outcome = await rig.host.alarm_action("a1", "stop", None)

        assert outcome.reason == "alarms reason"

    @pytest.mark.asyncio
    async def test_neither_alarms_nor_timers_built(self):
        rig = _rig(FakeCapability("clock"))

        outcome = await rig.host.alarm_action("a1", "stop", None)

        assert outcome.status == "unavailable"
        assert outcome.reason == "alarms and timers are not in MIST_CAPABILITIES"
        assert "not enabled" not in outcome.summary

    @pytest.mark.asyncio
    async def test_available_alarms_with_no_control_is_not_wired(self, caplog):
        rig = _rig(FakeCapability("alarms"))
        caplog.set_level(logging.INFO, logger=LOGGER)

        outcome = await rig.host.alarm_action("a1", "stop", None)

        assert (outcome.status, outcome.reason) == ("unavailable", "alarm control is not wired")
        assert len(_records(caplog, logging.ERROR)) == 1

    @pytest.mark.parametrize(
        ("action", "minutes"), [("stop", None), ("snooze", 10), ("dismiss", None)]
    )
    @pytest.mark.asyncio
    async def test_delegates_to_act(self, action, minutes):
        control = FakeAlarmControl()
        control.outcome = ToolOutcome(status="ok", summary="Done.")
        rig = _rig(FakeCapability("alarms"), alarm_control=control)

        outcome = await rig.host.alarm_action("a1", action, minutes)

        assert outcome == ToolOutcome(status="ok", summary="Done.")
        assert control.acts == [("a1", action, minutes)]

    @pytest.mark.asyncio
    async def test_degraded_alarms_pass_the_gate(self):
        alarms = FakeCapability("alarms")
        alarms.set_status("degraded", "1 alarm quarantined")
        control = FakeAlarmControl()
        rig = _rig(alarms, alarm_control=control)

        outcome = await rig.host.alarm_action("a1", "dismiss", None)

        assert outcome.status == "ok"
        assert control.acts == [("a1", "dismiss", None)]

    @pytest.mark.asyncio
    async def test_outcome_from_act_is_redacted(self):
        control = FakeAlarmControl()
        control.outcome = ToolOutcome(
            status="invalid_args", summary=f"no alarm {TOKEN}", reason=f"unknown {TOKEN}"
        )
        rig = _rig(FakeCapability("alarms"), alarm_control=control)
        rig.secrets.register(Secret(TOKEN))

        outcome = await rig.host.alarm_action("a1", "stop", None)

        assert outcome.status == "invalid_args"
        assert outcome.summary == "no alarm ***"
        assert outcome.reason == "unknown ***"

    @pytest.mark.asyncio
    async def test_store_error_from_act_is_a_redacted_error_outcome(self):
        control = FakeAlarmControl()
        control.act_error = CapabilityStoreError(f"alarm store locked {TOKEN}")
        rig = _rig(FakeCapability("alarms"), alarm_control=control)
        rig.secrets.register(Secret(TOKEN))

        outcome = await rig.host.alarm_action("a1", "stop", None)

        assert outcome.status == "error"
        assert outcome.reason == "alarm store locked ***"
        assert TOKEN not in outcome.summary

    @pytest.mark.asyncio
    async def test_unavailable_error_from_act_is_an_unavailable_outcome(self):
        control = FakeAlarmControl()
        control.act_error = CapabilityUnavailableError("scheduler is dead")
        rig = _rig(FakeCapability("alarms"), alarm_control=control)

        outcome = await rig.host.alarm_action("a1", "snooze", 5)

        assert (outcome.status, outcome.reason) == ("unavailable", "scheduler is dead")

    @pytest.mark.asyncio
    async def test_other_exception_from_act_propagates(self):
        control = FakeAlarmControl()
        control.act_error = RuntimeError("bug in act")
        rig = _rig(FakeCapability("alarms"), alarm_control=control)

        with pytest.raises(RuntimeError, match="bug in act"):
            await rig.host.alarm_action("a1", "stop", None)

    @pytest.mark.asyncio
    async def test_logs_one_info_line_and_writes_no_action_log_row(self, caplog):
        control = FakeAlarmControl()
        rig = _rig(FakeCapability("alarms"), alarm_control=control)
        caplog.set_level(logging.INFO, logger=LOGGER)

        await rig.host.alarm_action("alarm-42", "snooze", 5)

        (info,) = _records(caplog, logging.INFO)
        assert "alarm-42" in info.getMessage()
        assert "snooze" in info.getMessage()
        assert await _log_rows(rig.db) == []
        assert rig.host.stats()["invocations"]["ok"] == 0
