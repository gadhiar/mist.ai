"""Tests for the alarms, timers and clock capabilities and `SchedulerAlarmControl`.

Each tool runs against real `AlarmRecords` on a temp-file database with a `FakeClock`. The
scheduler is `StubScheduler`: the real `AlarmScheduler` methods over those records, with its
health scripted and no task started, so a test drives each health state directly. One test runs
the real scheduler task end to end.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
import pytest_asyncio
from pydantic import ValidationError

from backend.capabilities.action_log import ActionLog
from backend.capabilities.alarms.capability import (
    TZ_UNSET_REASON,
    AlarmsCapability,
    ClockCapability,
    RingingTracker,
    SchedulerAlarmControl,
    TimersCapability,
    UserZone,
    resolve_zone,
)
from backend.capabilities.alarms.records import AlarmRecords, Occurrence
from backend.capabilities.alarms.ringing import RingDelivery
from backend.capabilities.alarms.scheduler import (
    AlarmScheduler,
    SchedulerHealth,
    SchedulerState,
)
from backend.capabilities.config import AlarmSettings, load_capabilities_config
from backend.capabilities.contracts import FastIntent, InvocationContext, ToolOutcome
from backend.capabilities.db import CapabilitiesDb
from backend.capabilities.decoder import DECODE_INSTRUCTIONS, ActionDecoder
from backend.capabilities.host import CapabilityHost, HostSettings
from backend.capabilities.intents import normalize_utterance
from backend.capabilities.policy import PolicyTable
from backend.capabilities.replies import is_brief_sentence
from backend.capabilities.router import RouterConfig, ToolRouter, count_catalog_tokens
from backend.capabilities.schema import to_args_schema, to_catalog_line
from backend.capabilities.secrets import SecretRegistry
from backend.chat.context_budget import ApproximateTokenCounter
from backend.errors import CapabilityStoreError, CapabilityUnavailableError
from tests.unit.capabilities.fakes import (
    DEFAULT_START,
    FakeCapability,
    FakeClock,
    FakePower,
    FakeTiers,
    RecordingEventSink,
    make_tool_spec,
)

T0 = DEFAULT_START
"""Monday 5 January 2026, 07:00 UTC."""
ZONE = UserZone(name="UTC", tz=UTC, reason=None)
UNSET = UserZone(name=None, tz=None, reason=TZ_UNSET_REASON)
SETTINGS = AlarmSettings(
    ring_timeout_s=600.0, snooze_minutes=9, missed_grace_s=900.0, fallback_after_s=30.0
)
COUNTER = ApproximateTokenCounter()
WAKE_WORDS = ("mist", "hey mist", "ok mist", "okay mist")
ENDPOINTS = ("desktop", "kitchen")


class RecordingNotifier:
    """The notifier `RingingTracker` wraps in these tests; it records every call."""

    def __init__(self) -> None:
        self.rings: list[str] = []
        self.stops: list[str] = []
        self.states: list[int] = []

    async def ring(self, occurrence: Occurrence, ring_timeout_s: float) -> RingDelivery:
        self.rings.append(occurrence.occurrence_id)
        return RingDelivery(sent_to=("desktop",), fallback=False)

    async def stopped(self, occurrence_id: str) -> None:
        self.stops.append(occurrence_id)

    async def state_changed(
        self, occurrences: Sequence[Occurrence], rev: int, at: datetime
    ) -> None:
        self.states.append(rev)


class StubScheduler(AlarmScheduler):
    """The real scheduler's methods with a scripted health; `start` and `stop` only count."""

    def __init__(self, records: AlarmRecords, clock: FakeClock, notifier: object) -> None:
        super().__init__(records, clock, notifier, SETTINGS, UTC)  # type: ignore[arg-type]
        self.state: SchedulerState = "running"
        self.reason: str | None = None
        self.dead = False
        self.starts = 0
        self.stops = 0

    def set_health(self, state: SchedulerState, reason: str | None = None) -> None:
        self.state = state
        self.reason = reason
        self.dead = state == "dead"

    def health(self) -> SchedulerHealth:
        return SchedulerHealth(state=self.state, reason=self.reason)

    def task_dead(self) -> bool:
        return self.dead

    async def start(self) -> None:
        self.starts += 1

    async def stop(self) -> None:
        self.stops += 1


class CountingRecords(AlarmRecords):
    """`AlarmRecords` counting the reads `SchedulerAlarmControl.snapshot` may make."""

    def __init__(self, db: CapabilitiesDb) -> None:
        super().__init__(db)
        self.snapshots = 0
        self.revs = 0

    async def snapshot(self) -> tuple[int, list[Occurrence]]:
        self.snapshots += 1
        return await super().snapshot()

    async def rev(self) -> int:
        self.revs += 1
        return await super().rev()


@dataclass
class Rig:
    db: CapabilitiesDb
    records: CountingRecords
    clock: FakeClock
    notifier: RecordingNotifier
    tracker: RingingTracker
    scheduler: StubScheduler
    alarms: AlarmsCapability
    timers: TimersCapability
    clock_capability: ClockCapability
    control: SchedulerAlarmControl
    endpoints: list[str] = field(default_factory=lambda: list(ENDPOINTS))

    async def ring(self, occurrence_id: str) -> None:
        """Fire an occurrence as ringing and publish the state, as a scheduler tick does."""
        assert await self.records.fire(
            occurrence_id, outcome="ringing", at=self.clock.now(), late_by_s=0.0, next_fire_at=None
        )
        rev, live = await self.records.snapshot()
        await self.tracker.state_changed(live, rev, self.clock.now())

    async def miss(self, occurrence_id: str) -> None:
        assert await self.records.fire(
            occurrence_id,
            outcome="missed",
            at=self.clock.now(),
            late_by_s=1000.0,
            next_fire_at=None,
        )

    async def live(self) -> list[Occurrence]:
        _, live = await self.records.snapshot()
        return live

    async def events(self) -> list[tuple[str, str | None, str | None]]:
        rows = await self.db.run(
            lambda conn: conn.execute(
                "SELECT kind, source, endpoint FROM alarm_events ORDER BY seq"
            ).fetchall()
        )
        return [tuple(row) for row in rows]


_DBS: list[CapabilitiesDb] = []


@pytest.fixture(autouse=True)
def _close_databases():
    yield
    while _DBS:
        _DBS.pop().close()


def _rig(path: Path, *, start: datetime = T0) -> Rig:
    db = CapabilitiesDb(str(path))
    _DBS.append(db)
    records = CountingRecords(db)
    clock = FakeClock(start)
    notifier = RecordingNotifier()
    tracker = RingingTracker(notifier)
    scheduler = StubScheduler(records, clock, tracker)
    endpoints = list(ENDPOINTS)

    def known() -> tuple[str, ...]:
        return tuple(endpoints)

    common = {
        "clock": clock,
        "zone": ZONE,
        "ringing": tracker,
        "endpoints": known,
        "snooze_minutes": SETTINGS.snooze_minutes,
    }
    return Rig(
        db=db,
        records=records,
        clock=clock,
        notifier=notifier,
        tracker=tracker,
        scheduler=scheduler,
        alarms=AlarmsCapability(records, scheduler, **common),  # type: ignore[arg-type]
        timers=TimersCapability(records, scheduler, **common),  # type: ignore[arg-type]
        clock_capability=ClockCapability(clock=clock, zone=ZONE),
        control=SchedulerAlarmControl(records, scheduler, clock, SETTINGS, UTC),
        endpoints=endpoints,
    )


@pytest_asyncio.fixture()
async def rig(tmp_path) -> Rig:
    built = _rig(tmp_path / "capabilities.db")
    await built.alarms.start()
    await built.timers.start()
    return built


def _unwritable(tmp_path: Path) -> Path:
    blocker = tmp_path / "blocker"
    blocker.write_text("a regular file where a directory should be")
    return blocker / "capabilities.db"


def _ctx(
    clock: FakeClock, source: str = "model", endpoint: str | None = "desktop"
) -> InvocationContext:
    return InvocationContext(
        session_id="s1",
        turn_id="t1",
        utterance="",
        tool_call_id="call-1",
        source=source,  # type: ignore[arg-type]
        now=clock.now(),
        endpoint=endpoint,
    )


def _spec(capability: object, tool: str):
    return next(spec for spec in capability.tools() if spec.name == tool)  # type: ignore[attr-defined]


async def _call(
    capability: object,
    tool: str,
    args: dict[str, object] | None = None,
    *,
    clock: FakeClock,
    source: str = "model",
    endpoint: str | None = "desktop",
) -> ToolOutcome:
    spec = _spec(capability, tool)
    validated = spec.args_model.model_validate(args or {})
    return await capability.invoke(tool, validated, _ctx(clock, source, endpoint))  # type: ignore[attr-defined]


def _norm(raw: str) -> str:
    return normalize_utterance(raw, WAKE_WORDS)


# -- Contracts ---------------------------------------------------------------------------------


EXPECTED_TOOLS = {
    "alarms.set": ("reversible", "allow", "alarms"),
    "alarms.list": ("read", "allow", "alarms"),
    "alarms.cancel": ("reversible", "allow", "alarms"),
    "alarms.ringing": ("reversible", "allow", "alarms"),
    "timers.start": ("reversible", "allow", "timers"),
    "timers.list": ("read", "allow", "timers"),
    "timers.cancel": ("reversible", "allow", "timers"),
    "timers.stop": ("reversible", "allow", "timers"),
    "clock.now": ("read", "allow", "clock"),
}


def _capabilities(rig: Rig) -> list[object]:
    return [rig.alarms, rig.timers, rig.clock_capability]


def _lines(capability: object) -> list[str]:
    return [to_catalog_line(spec, to_args_schema(spec)) for spec in capability.tools()]  # type: ignore[attr-defined]


class TestContracts:
    @pytest.mark.asyncio
    async def test_tools_carry_the_contract_risks_policies_acks_and_views(self, rig):
        specs = {s.name: s for c in _capabilities(rig) for s in c.tools()}  # type: ignore[attr-defined]

        assert set(specs) == set(EXPECTED_TOOLS)
        for name, (risk, policy, view) in EXPECTED_TOOLS.items():
            spec = specs[name]
            assert (spec.risk, spec.default_policy, spec.view) == (risk, policy, view), name
            assert spec.ack, name
        assert [c.name for c in _capabilities(rig)] == ["alarms", "timers", "clock"]  # type: ignore[attr-defined]

    @pytest.mark.parametrize(
        ("utterance", "capability"),
        [
            ("my alarm", "alarms"),
            ("alarms please", "alarms"),
            ("wake me", "alarms"),
            ("snooze that", "alarms"),
            ("a timer", "timers"),
            ("my timers", "timers"),
            ("countdown", "timers"),
            ("the time", "clock"),
            ("the date", "clock"),
            ("which day", "clock"),
            ("today", "clock"),
        ],
    )
    @pytest.mark.asyncio
    async def test_the_minimum_keywords_route_to_each_capability(self, rig, utterance, capability):
        router = ToolRouter(
            _capabilities(rig), RouterConfig(token_budget=1200, reserved_tokens=200), COUNTER, None
        )  # type: ignore[arg-type]

        assert router.select(utterance, "s1").capabilities == (capability,)

    @pytest.mark.asyncio
    async def test_each_catalog_fits_the_per_capability_budget_at_default_config(self, rig):
        config = load_capabilities_config({})
        router_config = RouterConfig(
            token_budget=config.tool_token_budget,
            reserved_tokens=COUNTER.count(DECODE_INSTRUCTIONS),
        )
        # Registration refuses a capability whose worst case exceeds the budget.
        router = ToolRouter(_capabilities(rig), router_config, COUNTER, None)  # type: ignore[arg-type]

        for capability in _capabilities(rig):
            worst = count_catalog_tokens(_lines(capability), COUNTER)
            worst += capability.refine_allowance_tokens  # type: ignore[attr-defined]
            assert worst <= router.per_capability_budget()

    @pytest.mark.asyncio
    async def test_arguments_schemas_are_byte_stable(self, rig, tmp_path):
        other = _rig(tmp_path / "other.db")
        for mine, theirs in zip(_capabilities(rig), _capabilities(other), strict=True):
            for left, right in zip(mine.tools(), theirs.tools(), strict=True):  # type: ignore[attr-defined]
                assert json.dumps(to_args_schema(left)) == json.dumps(to_args_schema(right))

    @pytest.mark.asyncio
    async def test_cancel_with_all_escalates_and_nothing_else_does(self, rig):
        cancel_all = _spec(rig.alarms, "alarms.cancel").args_model.model_validate({"all": True})
        cancel_one = _spec(rig.alarms, "alarms.cancel").args_model.model_validate({"time": "7"})
        timers_all = _spec(rig.timers, "timers.cancel").args_model.model_validate({"all": True})
        timers_one = _spec(rig.timers, "timers.cancel").args_model.model_validate({})

        assert rig.alarms.escalate("alarms.cancel", cancel_all) is True
        assert rig.alarms.escalate("alarms.cancel", cancel_one) is False
        assert rig.timers.escalate("timers.cancel", timers_all) is True
        assert rig.timers.escalate("timers.cancel", timers_one) is False
        set_args = _spec(rig.alarms, "alarms.set").args_model.model_validate({"time": "7"})
        assert rig.alarms.escalate("alarms.set", set_args) is False


# -- Time zone and status ----------------------------------------------------------------------


class TestZone:
    def test_an_unset_zone_names_the_variable(self):
        assert resolve_zone(None) == UserZone(name=None, tz=None, reason="MIST_TIMEZONE unset")
        assert resolve_zone("  ").reason == "MIST_TIMEZONE unset"

    def test_an_unknown_zone_names_it(self):
        zone = resolve_zone("Mars/Olympus_Mons")

        assert zone.tz is None
        assert zone.reason == "timezone Mars/Olympus_Mons not found (tzdata missing?)"

    def test_a_known_zone_resolves(self):
        zone = resolve_zone("UTC")
        if zone.tz is None:
            pytest.skip("tzdata absent until H1")

        assert (zone.name, zone.reason) == ("UTC", None)


class TestStatus:
    @pytest.mark.parametrize(
        "zone",
        [
            UNSET,
            UserZone(name="Mars/X", tz=None, reason="timezone Mars/X not found (tzdata missing?)"),
        ],
    )
    @pytest.mark.asyncio
    async def test_without_a_zone_each_capability_is_unavailable_with_the_reason(
        self, tmp_path, zone
    ):
        records = AlarmRecords(CapabilitiesDb(str(tmp_path / "capabilities.db")))
        clock = FakeClock()
        common = {
            "clock": clock,
            "zone": zone,
            "ringing": None,
            "endpoints": lambda: ("desktop",),
            "snooze_minutes": 9,
        }
        alarms = AlarmsCapability(records, None, **common)  # type: ignore[arg-type]
        timers = TimersCapability(records, None, **common)  # type: ignore[arg-type]
        clock_capability = ClockCapability(clock=clock, zone=zone)

        await alarms.start()
        await timers.start()
        for capability in (alarms, timers, clock_capability):
            status = capability.status()
            assert (status.state, status.reason) == ("unavailable", zone.reason)
        assert alarms.active() is False
        assert alarms.match_intent("set an alarm for 7") is None
        assert clock_capability.match_intent("what time is it") is None
        outcome = await _call(alarms, "alarms.set", {"time": "7"}, clock=clock)
        assert (outcome.status, outcome.reason) == ("unavailable", zone.reason)

    def test_a_scheduler_of_none_needs_a_zone_reason(self, tmp_path):
        records = AlarmRecords(CapabilitiesDb(str(tmp_path / "capabilities.db")))
        with pytest.raises(ValueError, match="scheduler"):
            AlarmsCapability(
                records,
                None,
                clock=FakeClock(),
                zone=ZONE,
                ringing=None,
                endpoints=lambda: ("desktop",),
                snooze_minutes=9,
            )

    @pytest.mark.parametrize(
        ("state", "reason", "expected_reason"),
        [
            (
                "failing",
                "alarm records failed (OperationalError)",
                "alarm records failed (OperationalError)",
            ),
            (
                "dead",
                "alarm scheduler task died: RuntimeError",
                "alarm scheduler task died: RuntimeError",
            ),
            ("stopped", None, "alarm scheduler stopped"),
        ],
    )
    @pytest.mark.asyncio
    async def test_alarms_and_timers_follow_the_scheduler_and_the_clock_does_not(
        self, rig, state, reason, expected_reason
    ):
        assert rig.alarms.status().state == "available"

        rig.scheduler.set_health(state, reason)

        for capability in (rig.alarms, rig.timers):
            status = capability.status()
            assert (status.state, status.reason) == ("unavailable", expected_reason)
            assert status.task_dead is (state == "dead")
        clock_status = rig.clock_capability.status()
        assert (clock_status.state, clock_status.reason) == ("available", None)

        rig.scheduler.set_health("running")
        assert rig.alarms.status().state == "available"
        assert rig.timers.status().task_dead is False

    @pytest.mark.asyncio
    async def test_a_quarantined_row_degrades_until_it_is_repaired(self, rig):
        await _call(rig.alarms, "alarms.set", {"time": "8am", "day": "tomorrow"}, clock=rig.clock)
        [occurrence] = await rig.live()
        await rig.db.run(
            lambda conn: conn.execute(
                "UPDATE alarm_occurrences SET late_by_s = -1 WHERE id = ?",
                (occurrence.occurrence_id,),
            )
        )
        await rig.records.snapshot()

        for capability in (rig.alarms, rig.timers):
            status = capability.status()
            assert (status.state, status.reason) == (
                "degraded",
                "1 stored alarm quarantined; see the log",
            )

        await rig.db.run(
            lambda conn: conn.execute(
                "UPDATE alarm_occurrences SET late_by_s = NULL WHERE id = ?",
                (occurrence.occurrence_id,),
            )
        )
        await rig.records.snapshot()

        assert (rig.alarms.status().state, rig.alarms.status().reason) == ("available", None)

    @pytest.mark.asyncio
    async def test_a_quarantined_row_that_is_cancelled_restores_available(self, rig):
        for when in ("8am", "9am"):
            await _call(
                rig.alarms, "alarms.set", {"time": when, "day": "tomorrow"}, clock=rig.clock
            )
        ids = [o.occurrence_id for o in await rig.live()]
        await rig.db.run(
            lambda conn: conn.executemany(
                "UPDATE alarm_occurrences SET late_by_s = -1 WHERE id = ?", [(i,) for i in ids]
            )
        )
        await rig.records.snapshot()
        assert rig.alarms.status().reason == "2 stored alarms quarantined; see the log"

        for occurrence_id in ids:
            outcome = await _call(
                rig.alarms, "alarms.cancel", {"alarm_id": occurrence_id}, clock=rig.clock
            )
            assert outcome.status == "ok"

        assert rig.alarms.status().state == "available"
        assert rig.records.quarantined() == ()

    @pytest.mark.asyncio
    async def test_start_migrates_starts_the_scheduler_and_stop_stops_it(self, tmp_path):
        built = _rig(tmp_path / "capabilities.db")

        await built.alarms.start()
        await built.timers.start()
        assert await built.records.saved() == []
        assert built.scheduler.starts == 2

        await built.alarms.stop()
        assert built.scheduler.stops == 1

    @pytest.mark.asyncio
    async def test_an_unwritable_database_fails_start_with_a_store_error_naming_it(self, tmp_path):
        path = _unwritable(tmp_path)
        built = _rig(path)

        for capability in (built.alarms, built.timers):
            with pytest.raises(CapabilityStoreError, match=str(path.parent).replace("\\", "\\\\")):
                await capability.start()
        assert built.scheduler.starts == 0


# -- alarms.set --------------------------------------------------------------------------------


class TestAlarmsSet:
    @pytest.mark.asyncio
    async def test_a_time_sets_an_alarm_with_a_speakable_summary(self, rig):
        outcome = await _call(rig.alarms, "alarms.set", {"time": "7am"}, clock=rig.clock)

        assert outcome.status == "ok"
        assert outcome.summary == "Alarm set for 7:00 AM tomorrow."
        [occurrence] = await rig.live()
        assert occurrence.next_fire_at == T0 + timedelta(days=1)
        assert occurrence.record.kind == "alarm"
        assert outcome.data is not None
        assert outcome.data["alarm_id"] == occurrence.occurrence_id

    @pytest.mark.parametrize(
        ("args", "summary"),
        [
            ({"time": "7"}, "Alarm set for 7:00 PM today."),
            ({"time": "6:30", "day": "wednesday"}, "Alarm set for 6:30 AM Wednesday."),
            ({"time": "9am", "day": "2026-01-20"}, "Alarm set for 9:00 AM Tuesday January 20."),
            ({"time": "7am", "repeat": "daily"}, "Alarm set for 7:00 AM every day."),
            ({"time": "6:15 am", "repeat": "weekdays"}, "Alarm set for 6:15 AM every weekday."),
            ({"time": "10am", "repeat": "weekends"}, "Alarm set for 10:00 AM on weekends."),
            (
                {"time": "7am", "repeat": "custom", "days": ["friday", "monday", "wednesday"]},
                "Alarm set for 7:00 AM every Monday, Wednesday and Friday.",
            ),
            (
                {
                    "time": "7am",
                    "repeat": "custom",
                    "days": ["monday", "tuesday", "wednesday", "thursday", "saturday", "sunday"],
                },
                "Alarm set for 7:00 AM every day except Friday.",
            ),
            ({"time": "noon", "label": "lunch"}, "Alarm set for 12:00 PM today."),
            ({"time": "in 20 minutes"}, "Alarm set for 7:20 AM today."),
        ],
    )
    @pytest.mark.asyncio
    async def test_each_summary_is_one_brief_sentence(self, rig, args, summary):
        outcome = await _call(rig.alarms, "alarms.set", args, clock=rig.clock)

        assert (outcome.status, outcome.summary) == ("ok", summary)
        assert is_brief_sentence(outcome.summary)

    @pytest.mark.asyncio
    async def test_a_time_the_parser_refuses_is_invalid(self, rig):
        outcome = await _call(rig.alarms, "alarms.set", {"time": "whenever"}, clock=rig.clock)

        assert outcome.status == "invalid_args"
        assert "whenever" in outcome.summary
        assert await rig.live() == []

    @pytest.mark.asyncio
    async def test_a_passed_time_today_is_invalid(self, rig):
        outcome = await _call(
            rig.alarms, "alarms.set", {"time": "6am", "day": "today"}, clock=rig.clock
        )

        assert outcome.status == "invalid_args"

    @pytest.mark.parametrize(
        "args",
        [
            {},
            {"day": "tomorrow"},
            {"time": "7", "days": ["monday"]},
            {"time": "7", "repeat": "custom"},
            {"time": "7", "repeat": "custom", "days": []},
            {"time": "7", "repeat": "daily", "day": "tomorrow"},
            {"label": "gym", "repeat": "daily"},
            {"time": "7", "ring_on": []},
            {"time": "7", "extra": 1},
        ],
    )
    def test_incoherent_arguments_fail_validation(self, rig, args):
        with pytest.raises(ValidationError):
            _spec(rig.alarms, "alarms.set").args_model.model_validate(args)

    @pytest.mark.asyncio
    async def test_the_same_one_off_time_on_two_days_is_one_record_used_twice(self, rig):
        await _call(rig.alarms, "alarms.set", {"time": "7am", "day": "tomorrow"}, clock=rig.clock)
        await _call(rig.alarms, "alarms.set", {"time": "7am", "day": "friday"}, clock=rig.clock)

        [record] = await rig.records.saved()
        assert (record.identity, record.use_count) == ("alarm:one_off:07:00", 2)
        assert len(await rig.live()) == 2

    @pytest.mark.asyncio
    async def test_custom_days_give_the_weekday_set(self, rig):
        outcome = await _call(
            rig.alarms,
            "alarms.set",
            {"time": "7am", "repeat": "custom", "days": ["monday", "thursday"]},
            clock=rig.clock,
        )

        assert outcome.status == "ok"
        [occurrence] = await rig.live()
        assert occurrence.record.days == frozenset({0, 3})
        assert occurrence.record.category == "recurring"
        # Monday 07:00 has just passed, so the series starts on Thursday.
        assert occurrence.next_fire_at == T0 + timedelta(days=3)

    @pytest.mark.asyncio
    async def test_a_repeat_needs_a_clock_time(self, rig):
        outcome = await _call(
            rig.alarms, "alarms.set", {"time": "in 20 minutes", "repeat": "daily"}, clock=rig.clock
        )

        assert outcome.status == "invalid_args"
        assert await rig.live() == []

    @pytest.mark.asyncio
    async def test_an_already_set_alarm_says_so(self, rig):
        await _call(rig.alarms, "alarms.set", {"time": "7am", "repeat": "daily"}, clock=rig.clock)
        outcome = await _call(
            rig.alarms, "alarms.set", {"time": "7am", "repeat": "daily"}, clock=rig.clock
        )

        assert outcome.summary == "Your 7:00 AM every day alarm is already set."
        assert outcome.data is not None and outcome.data["already_set"] is True

    @pytest.mark.asyncio
    async def test_a_saved_tag_alone_rearms_it_and_a_new_time_moves_it(self, rig):
        await _call(rig.alarms, "alarms.set", {"time": "6am", "label": "Gym"}, clock=rig.clock)
        [first] = await rig.live()
        await _call(rig.alarms, "alarms.cancel", {"alarm_id": first.occurrence_id}, clock=rig.clock)
        assert await rig.live() == []

        rearmed = await _call(rig.alarms, "alarms.set", {"label": "gym"}, clock=rig.clock)

        assert (rearmed.status, rearmed.summary) == ("ok", "Alarm set for 6:00 AM tomorrow.")
        [second] = await rig.live()
        assert second.record.record_id == first.record.record_id

        moved = await _call(
            rig.alarms, "alarms.set", {"time": "6:30am", "label": "GYM"}, clock=rig.clock
        )

        assert (moved.status, moved.summary) == ("ok", "Alarm moved to 6:30 AM tomorrow.")
        [third] = await rig.live()
        assert third.record.record_id == first.record.record_id
        assert third.next_fire_at == T0 + timedelta(days=1, minutes=-30)
        [record] = await rig.records.saved()
        assert record.use_count == 3

    @pytest.mark.asyncio
    async def test_an_unknown_tag_alone_is_invalid(self, rig):
        outcome = await _call(rig.alarms, "alarms.set", {"label": "gym"}, clock=rig.clock)

        assert outcome.status == "invalid_args"
        assert "gym" in outcome.summary

    @pytest.mark.asyncio
    async def test_an_unknown_ring_on_names_the_known_endpoints(self, rig):
        outcome = await _call(
            rig.alarms, "alarms.set", {"time": "7am", "ring_on": ["bedroom"]}, clock=rig.clock
        )

        assert outcome.status == "invalid_args"
        assert "bedroom" in outcome.summary
        assert "desktop" in outcome.summary and "kitchen" in outcome.summary
        assert outcome.data == {"known_endpoints": ["desktop", "kitchen"]}
        assert await rig.live() == []

    @pytest.mark.asyncio
    async def test_a_known_ring_on_is_armed(self, rig):
        await _call(
            rig.alarms, "alarms.set", {"time": "7am", "ring_on": ["kitchen"]}, clock=rig.clock
        )

        [occurrence] = await rig.live()
        assert occurrence.ring_on == ("kitchen",)

    @pytest.mark.asyncio
    async def test_the_call_source_is_recorded(self, rig):
        await _call(rig.alarms, "alarms.set", {"time": "7am"}, clock=rig.clock, source="fastpath")

        assert await rig.events() == [("set", "fastpath", None)]


# -- alarms.list -------------------------------------------------------------------------------


class TestAlarmsList:
    @pytest.mark.asyncio
    async def test_no_alarms(self, rig):
        outcome = await _call(rig.alarms, "alarms.list", clock=rig.clock)

        assert (outcome.status, outcome.summary) == ("ok", "You have no alarms set.")
        assert outcome.data == {"scope": "live", "alarms": []}

    @pytest.mark.asyncio
    async def test_live_alarms_are_read_in_firing_order_without_timers(self, rig):
        await _call(rig.alarms, "alarms.set", {"time": "9am", "day": "friday"}, clock=rig.clock)
        await _call(rig.alarms, "alarms.set", {"time": "6am", "label": "gym"}, clock=rig.clock)
        await _call(rig.timers, "timers.start", {"minutes": 10}, clock=rig.clock)

        outcome = await _call(rig.alarms, "alarms.list", clock=rig.clock)

        assert outcome.summary == ("You have 2 alarms. Gym, 6:00 AM tomorrow. 9:00 AM Friday.")
        assert outcome.data is not None
        assert [a["label"] for a in outcome.data["alarms"]] == ["gym", None]  # type: ignore[index]

    @pytest.mark.asyncio
    async def test_saved_scope_lists_every_record_by_use_count(self, rig):
        for _ in range(3):
            await _call(
                rig.alarms, "alarms.set", {"time": "7am", "repeat": "weekdays"}, clock=rig.clock
            )
            live = [o for o in await rig.live() if o.record.days is not None]
            await rig.records.cancel(
                [o.occurrence_id for o in live], at=rig.clock.now(), source="model"
            )
        await _call(
            rig.alarms, "alarms.set", {"time": "7am", "repeat": "weekdays"}, clock=rig.clock
        )
        await _call(rig.alarms, "alarms.set", {"time": "9pm", "day": "friday"}, clock=rig.clock)
        await _call(rig.timers, "timers.start", {"minutes": 5}, clock=rig.clock)

        outcome = await _call(rig.alarms, "alarms.list", {"scope": "saved"}, clock=rig.clock)

        assert outcome.status == "ok"
        assert outcome.summary == (
            "You have 2 alarms. 7:00 AM every weekday, used 4 times, set. "
            "9:00 PM once, used 1 time, set."
        )
        assert outcome.data is not None
        saved = outcome.data["alarms"]
        assert [(a["use_count"], a["active"]) for a in saved] == [(4, True), (1, True)]  # type: ignore[index]

    @pytest.mark.asyncio
    async def test_more_than_five_alarms_leave_the_rest_to_the_screen(self, rig):
        for hour in range(1, 8):
            await _call(rig.alarms, "alarms.set", {"time": f"{hour}pm"}, clock=rig.clock)

        outcome = await _call(rig.alarms, "alarms.list", clock=rig.clock)

        assert outcome.summary.startswith("You have 7 alarms. 1:00 PM today.")
        assert outcome.summary.endswith("5:00 PM today, and 2 more on screen.")
        assert len(outcome.data["alarms"]) == 7  # type: ignore[index]


# -- alarms.cancel -----------------------------------------------------------------------------


class TestAlarmsCancel:
    @pytest.mark.asyncio
    async def test_nothing_to_cancel(self, rig):
        outcome = await _call(rig.alarms, "alarms.cancel", {"all": True}, clock=rig.clock)

        assert (outcome.status, outcome.summary) == ("ok", "You have no alarms set.")

    @pytest.mark.asyncio
    async def test_all_cancels_every_alarm_and_no_timer(self, rig):
        await _call(rig.alarms, "alarms.set", {"time": "7am"}, clock=rig.clock)
        await _call(rig.alarms, "alarms.set", {"time": "8am", "repeat": "daily"}, clock=rig.clock)
        await _call(rig.timers, "timers.start", {"minutes": 5}, clock=rig.clock)

        outcome = await _call(rig.alarms, "alarms.cancel", {"all": True}, clock=rig.clock)

        assert (outcome.status, outcome.summary) == ("ok", "2 alarms cancelled.")
        assert [o.record.kind for o in await rig.live()] == ["timer"]

    @pytest.mark.asyncio
    async def test_by_id(self, rig):
        await _call(rig.alarms, "alarms.set", {"time": "7am"}, clock=rig.clock)
        [occurrence] = await rig.live()

        outcome = await _call(
            rig.alarms, "alarms.cancel", {"alarm_id": occurrence.occurrence_id}, clock=rig.clock
        )

        assert (outcome.status, outcome.summary) == ("ok", "Alarm for 7:00 AM tomorrow cancelled.")
        assert is_brief_sentence(outcome.summary)
        assert await rig.live() == []

    @pytest.mark.asyncio
    async def test_an_unknown_id_or_a_timers_id_is_invalid(self, rig):
        await _call(rig.timers, "timers.start", {"minutes": 5}, clock=rig.clock)
        [timer] = await rig.live()

        for alarm_id in ("nope", timer.occurrence_id):
            outcome = await _call(
                rig.alarms, "alarms.cancel", {"alarm_id": alarm_id}, clock=rig.clock
            )
            assert outcome.status == "invalid_args"
            assert alarm_id in outcome.summary
        assert len(await rig.live()) == 1

    @pytest.mark.asyncio
    async def test_a_time_matching_two_occurrences_cancels_the_next_and_names_its_day(self, rig):
        await _call(rig.alarms, "alarms.set", {"time": "7am", "day": "friday"}, clock=rig.clock)
        await _call(rig.alarms, "alarms.set", {"time": "7am", "day": "tomorrow"}, clock=rig.clock)

        outcome = await _call(rig.alarms, "alarms.cancel", {"time": "7am"}, clock=rig.clock)

        assert outcome.status == "ok"
        assert outcome.summary == "Alarm for 7:00 AM tomorrow cancelled; 1 more stays set."
        assert is_brief_sentence(outcome.summary)
        [left] = await rig.live()
        assert left.next_fire_at == T0 + timedelta(days=4)

    @pytest.mark.asyncio
    async def test_a_bare_hour_matches_the_evening_alarm_too(self, rig):
        await _call(rig.alarms, "alarms.set", {"time": "7pm", "day": "tomorrow"}, clock=rig.clock)

        outcome = await _call(rig.alarms, "alarms.cancel", {"time": "7"}, clock=rig.clock)

        assert (outcome.status, outcome.summary) == ("ok", "Alarm for 7:00 PM tomorrow cancelled.")

    @pytest.mark.asyncio
    async def test_a_repeating_alarm_loses_its_series_and_stays_saved(self, rig):
        await _call(
            rig.alarms, "alarms.set", {"time": "6am", "repeat": "weekdays"}, clock=rig.clock
        )

        outcome = await _call(rig.alarms, "alarms.cancel", {"time": "6am"}, clock=rig.clock)

        assert outcome.summary == "Alarm for 6:00 AM every weekday cancelled."
        assert await rig.live() == []
        [record] = await rig.records.saved()
        assert record.last_outcome == "cancelled"

    @pytest.mark.asyncio
    async def test_a_time_with_no_alarm_or_unreadable_is_invalid(self, rig):
        await _call(rig.alarms, "alarms.set", {"time": "7am"}, clock=rig.clock)

        for time_text in ("9am", "whenever"):
            outcome = await _call(rig.alarms, "alarms.cancel", {"time": time_text}, clock=rig.clock)
            assert outcome.status == "invalid_args"
        assert len(await rig.live()) == 1

    @pytest.mark.asyncio
    async def test_by_label(self, rig):
        await _call(rig.alarms, "alarms.set", {"time": "6am", "label": "gym"}, clock=rig.clock)
        await _call(rig.alarms, "alarms.set", {"time": "8am"}, clock=rig.clock)

        outcome = await _call(rig.alarms, "alarms.cancel", {"label": "Gym"}, clock=rig.clock)

        assert (outcome.status, outcome.summary) == ("ok", "Alarm for 6:00 AM tomorrow cancelled.")
        assert [o.record.label for o in await rig.live()] == [None]
        missing = await _call(rig.alarms, "alarms.cancel", {"label": "nap"}, clock=rig.clock)
        assert missing.status == "invalid_args"

    @pytest.mark.asyncio
    async def test_no_selector_cancels_the_only_alarm_and_asks_among_several(self, rig):
        await _call(rig.alarms, "alarms.set", {"time": "7am"}, clock=rig.clock)
        await _call(rig.alarms, "alarms.set", {"time": "8am"}, clock=rig.clock)

        several = await _call(rig.alarms, "alarms.cancel", clock=rig.clock)
        assert several.status == "invalid_args"
        assert len(await rig.live()) == 2

        await _call(rig.alarms, "alarms.cancel", {"time": "8am"}, clock=rig.clock)
        only = await _call(rig.alarms, "alarms.cancel", clock=rig.clock)
        assert (only.status, only.summary) == ("ok", "Alarm for 7:00 AM tomorrow cancelled.")


# -- Ringing -----------------------------------------------------------------------------------


async def _ringing_alarm(rig: Rig, time_text: str = "7:01am", day: str = "today") -> str:
    await _call(rig.alarms, "alarms.set", {"time": time_text, "day": day}, clock=rig.clock)
    occurrence = next(o for o in await rig.live() if o.record.kind == "alarm")
    await rig.clock.advance((occurrence.next_fire_at - rig.clock.now()).total_seconds())
    await rig.ring(occurrence.occurrence_id)
    return occurrence.occurrence_id


async def _ringing_timer(rig: Rig, seconds: int = 30) -> str:
    await _call(rig.timers, "timers.start", {"seconds": seconds}, clock=rig.clock)
    occurrence = next(
        o for o in await rig.live() if o.record.kind == "timer" and o.state == "scheduled"
    )
    await rig.clock.advance((occurrence.next_fire_at - rig.clock.now()).total_seconds())
    await rig.ring(occurrence.occurrence_id)
    return occurrence.occurrence_id


STOP_PHRASES = [
    "stop",
    "stop it",
    "turn it off",
    "turn off",
    "turn off the alarm",
    "turn the alarm off",
    "turn off the timer",
    "turn the timer off",
    "stop the alarm",
    "stop the timer",
    "alarm off",
    "timer off",
    "ok I'm up",
]


class TestRinging:
    @pytest.mark.asyncio
    async def test_active_while_one_of_its_items_rings(self, rig):
        assert (rig.alarms.active(), rig.timers.active()) == (False, False)

        alarm_id = await _ringing_alarm(rig)

        assert (rig.alarms.active(), rig.timers.active()) == (True, False)
        await _call(rig.alarms, "alarms.ringing", {"action": "stop"}, clock=rig.clock)
        assert rig.alarms.active() is False
        assert alarm_id in rig.notifier.stops

    @pytest.mark.asyncio
    async def test_the_tracker_follows_ring_stopped_and_loads_what_rings_at_start(
        self, rig, tmp_path
    ):
        alarm_id = await _ringing_alarm(rig)
        live = await rig.live()

        fresh = RingingTracker(RecordingNotifier())
        assert fresh.ringing("alarm") == ()
        await fresh.load(rig.records)
        assert fresh.ringing("alarm") == (alarm_id,)
        await fresh.stopped(alarm_id)
        assert fresh.ringing("alarm") == ()
        await fresh.ring(live[0], 600.0)
        assert fresh.ringing("alarm") == (alarm_id,)
        # A snapshot older than the state already seen is ignored.
        await fresh.state_changed([], 10_000, rig.clock.now())
        await fresh.load(rig.records)
        assert fresh.ringing("alarm") == ()

    @pytest.mark.asyncio
    async def test_ringing_stop_ends_the_alarm_recording_source_and_endpoint(self, rig):
        await _ringing_alarm(rig)

        outcome = await _call(
            rig.alarms, "alarms.ringing", {"action": "stop"}, clock=rig.clock, endpoint="kitchen"
        )

        assert (outcome.status, outcome.summary) == ("ok", "Alarm stopped.")
        assert await rig.live() == []
        assert (await rig.events())[-1] == ("stopped", "model", "kitchen")

    @pytest.mark.asyncio
    async def test_ringing_snooze_takes_minutes_or_the_default(self, rig):
        await _ringing_alarm(rig)

        outcome = await _call(
            rig.alarms, "alarms.ringing", {"action": "snooze", "minutes": 5}, clock=rig.clock
        )

        assert (outcome.status, outcome.summary) == ("ok", "Snoozed for 5 minutes.")
        [occurrence] = await rig.live()
        assert occurrence.state == "snoozed"
        assert occurrence.next_fire_at == rig.clock.now() + timedelta(minutes=5)

        await rig.clock.advance(300)
        await rig.ring(occurrence.occurrence_id)
        again = await _call(rig.alarms, "alarms.ringing", {"action": "snooze"}, clock=rig.clock)
        assert again.summary == "Snoozed for 9 minutes."

    @pytest.mark.asyncio
    async def test_ringing_with_nothing_ringing_is_invalid(self, rig):
        outcome = await _call(rig.alarms, "alarms.ringing", {"action": "stop"}, clock=rig.clock)

        assert (outcome.status, outcome.summary) == ("invalid_args", "No alarm is ringing.")

    @pytest.mark.parametrize(
        "args", [{"action": "stop", "minutes": 5}, {"action": "snooze", "minutes": 0}, {}]
    )
    def test_ringing_arguments_are_checked(self, rig, args):
        with pytest.raises(ValidationError):
            _spec(rig.alarms, "alarms.ringing").args_model.model_validate(args)

    @pytest.mark.parametrize("phrase", STOP_PHRASES)
    @pytest.mark.asyncio
    async def test_each_stop_phrase_stops_the_ringing_alarm(self, rig, phrase):
        await _ringing_alarm(rig)

        outcome = await rig.alarms.on_active_utterance(_norm(phrase), "s1")

        assert outcome is not None
        assert (outcome.status, outcome.summary) == ("ok", "Alarm stopped.")
        assert await rig.live() == []
        assert (await rig.events())[-1] == ("stopped", "fastpath", "desktop")

    @pytest.mark.parametrize(("phrase", "minutes"), [("snooze", 9), ("snooze 5 minutes", 5)])
    @pytest.mark.asyncio
    async def test_snooze_phrases_snooze(self, rig, phrase, minutes):
        await _ringing_alarm(rig)

        outcome = await rig.alarms.on_active_utterance(_norm(phrase), "s1")

        assert outcome is not None
        assert outcome.summary == f"Snoozed for {minutes} minutes."
        [occurrence] = await rig.live()
        assert occurrence.state == "snoozed"
        assert (await rig.events())[-1] == ("snoozed", "fastpath", "desktop")

    @pytest.mark.parametrize(
        "phrase", ["turn off the lights", "what time is it", "stop the music", "yes"]
    )
    @pytest.mark.asyncio
    async def test_anything_else_is_not_handled(self, rig, phrase):
        await _ringing_alarm(rig)

        assert await rig.alarms.on_active_utterance(_norm(phrase), "s1") is None
        assert [o.state for o in await rig.live()] == ["ringing"]

    @pytest.mark.asyncio
    async def test_a_phrase_naming_the_other_ringing_kind_is_left_to_it(self, rig):
        await _ringing_alarm(rig)
        await _ringing_timer(rig)

        assert await rig.alarms.on_active_utterance("stop the timer", "s1") is None
        stopped = await rig.timers.on_active_utterance("stop the timer", "s1")

        assert stopped is not None and stopped.summary == "Timer stopped."
        assert [o.record.kind for o in await rig.live()] == ["alarm"]

    @pytest.mark.asyncio
    async def test_a_ringing_alarm_takes_turn_off_before_the_fast_path(self, rig):
        lights = FakeCapability(
            "lights",
            tools=[make_tool_spec("lights.power")],
            intents={"turn off": FastIntent("lights.power", {})},
        )
        host = _host([rig.alarms, rig.timers, lights], rig.clock)
        await host.start()
        await host.wait_started()
        try:
            await _ringing_alarm(rig)

            resolution = await host.pre_llm("s1", "t1", "Turn off")

            assert resolution is not None and resolution.handled
            assert resolution.reply == "Alarm stopped."
            assert lights.invocations == []
            assert await rig.live() == []
        finally:
            await host.stop()

    @pytest.mark.asyncio
    async def test_the_real_scheduler_rings_a_timer_that_a_stop_phrase_ends(self, tmp_path):
        db = CapabilitiesDb(str(tmp_path / "capabilities.db"))
        _DBS.append(db)
        records = AlarmRecords(db)
        clock = FakeClock()
        notifier = RecordingNotifier()
        tracker = RingingTracker(notifier)
        scheduler = AlarmScheduler(records, clock, tracker, SETTINGS, UTC)
        timers = TimersCapability(
            records,
            scheduler,
            clock=clock,
            zone=ZONE,
            ringing=tracker,
            endpoints=lambda: ("desktop",),
            snooze_minutes=9,
        )
        await timers.start()
        try:
            await clock.wait_for_sleepers(1)
            outcome = await _call(timers, "timers.start", {"seconds": 10}, clock=clock)
            assert outcome.summary == "Timer set for 10 seconds."
            await clock.wait_for_sleepers(1)

            await clock.advance(10)
            await _eventually(timers.active)

            assert len(notifier.rings) == 1
            stopped = await timers.on_active_utterance("stop", "s1")
            assert stopped is not None and stopped.summary == "Timer stopped."
            assert timers.active() is False
        finally:
            await timers.stop()
        assert scheduler.health().state == "stopped"


async def _eventually(predicate: Callable[[], bool], timeout_s: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.01)


def _host(
    capabilities: list[object],
    clock: FakeClock,
    *,
    power: FakePower | None = None,
    tiers: FakeTiers | None = None,
) -> CapabilityHost:
    router = ToolRouter(
        capabilities, RouterConfig(token_budget=1200, reserved_tokens=100), COUNTER, None  # type: ignore[arg-type]
    )
    specs = [spec for capability in capabilities for spec in capability.tools()]  # type: ignore[attr-defined]
    db = CapabilitiesDb(":memory:")
    _DBS.append(db)
    return CapabilityHost(
        capabilities,  # type: ignore[arg-type]
        router=router,
        counter=COUNTER,
        policy=PolicyTable.from_specs(specs, None),
        decoder=ActionDecoder(),
        secrets=SecretRegistry(),
        action_log=ActionLog(db),
        event_sink=RecordingEventSink(),
        clock=clock,
        settings=HostSettings(),
        power=power or FakePower(),
        tiers=tiers,
    )


# -- Timers ------------------------------------------------------------------------------------


class TestTimers:
    @pytest.mark.parametrize(
        ("args", "summary"),
        [
            ({"minutes": 10}, "Timer set for 10 minutes."),
            ({"hours": 1, "minutes": 30}, "Timer set for 1 hour and 30 minutes."),
            ({"seconds": 1}, "Timer set for 1 second."),
            (
                {"hours": 2, "minutes": 5, "seconds": 10},
                "Timer set for 2 hours, 5 minutes and 10 seconds.",
            ),
            ({"minutes": 90}, "Timer set for 1 hour and 30 minutes."),
        ],
    )
    @pytest.mark.asyncio
    async def test_start_sets_a_timer_with_a_brief_summary(self, rig, args, summary):
        outcome = await _call(rig.timers, "timers.start", args, clock=rig.clock)

        assert (outcome.status, outcome.summary) == ("ok", summary)
        assert is_brief_sentence(outcome.summary)
        [occurrence] = await rig.live()
        assert occurrence.record.kind == "timer"
        total = occurrence.record.duration_s
        assert occurrence.next_fire_at == T0 + timedelta(seconds=total)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "args",
        [{}, {"minutes": 0}, {"hours": 25}, {"hours": 24, "seconds": 1}, {"minutes": -1}],
    )
    def test_a_duration_outside_one_second_to_a_day_fails_validation(self, rig, args):
        with pytest.raises(ValidationError):
            _spec(rig.timers, "timers.start").args_model.model_validate(args)

    @pytest.mark.asyncio
    async def test_a_saved_tag_alone_rearms_it(self, rig):
        await _call(rig.timers, "timers.start", {"minutes": 4, "label": "tea"}, clock=rig.clock)
        [first] = await rig.live()
        await _call(rig.timers, "timers.cancel", clock=rig.clock)

        outcome = await _call(rig.timers, "timers.start", {"label": "Tea"}, clock=rig.clock)

        assert (outcome.status, outcome.summary) == ("ok", "Timer set for 4 minutes.")
        [second] = await rig.live()
        assert second.record.record_id == first.record.record_id
        again = await _call(rig.timers, "timers.start", {"label": "tea"}, clock=rig.clock)
        assert (again.status, again.summary) == ("ok", "That timer is already running.")

    @pytest.mark.asyncio
    async def test_an_unknown_tag_alone_is_invalid(self, rig):
        outcome = await _call(rig.timers, "timers.start", {"label": "eggs"}, clock=rig.clock)

        assert outcome.status == "invalid_args"

    @pytest.mark.asyncio
    async def test_an_unknown_ring_on_names_the_known_endpoints(self, rig):
        outcome = await _call(
            rig.timers, "timers.start", {"minutes": 1, "ring_on": ["garage"]}, clock=rig.clock
        )

        assert outcome.status == "invalid_args"
        assert outcome.data == {"known_endpoints": ["desktop", "kitchen"]}

    @pytest.mark.asyncio
    async def test_list(self, rig):
        empty = await _call(rig.timers, "timers.list", clock=rig.clock)
        assert (empty.status, empty.summary) == ("ok", "No timers are running.")

        await _call(rig.timers, "timers.start", {"minutes": 10, "label": "pasta"}, clock=rig.clock)
        await _call(rig.timers, "timers.start", {"minutes": 3}, clock=rig.clock)
        await _call(rig.alarms, "alarms.set", {"time": "7am"}, clock=rig.clock)
        await rig.clock.advance(30)

        outcome = await _call(rig.timers, "timers.list", clock=rig.clock)

        assert outcome.summary == (
            "2 timers are running. 2 minutes and 30 seconds left. "
            "Pasta, 9 minutes and 30 seconds left."
        )
        assert [t["remaining_s"] for t in outcome.data["timers"]] == [150, 570]  # type: ignore[index]

    @pytest.mark.asyncio
    async def test_cancel_with_none_one_or_several(self, rig):
        none = await _call(rig.timers, "timers.cancel", clock=rig.clock)
        assert (none.status, none.summary) == ("ok", "No timer is running.")

        await _call(rig.timers, "timers.start", {"minutes": 10}, clock=rig.clock)
        one = await _call(rig.timers, "timers.cancel", clock=rig.clock)
        assert (one.status, one.summary) == ("ok", "Timer cancelled.")
        assert await rig.live() == []

        await _call(rig.timers, "timers.start", {"minutes": 10}, clock=rig.clock)
        await _call(rig.timers, "timers.start", {"minutes": 5, "label": "eggs"}, clock=rig.clock)
        several = await _call(rig.timers, "timers.cancel", clock=rig.clock)
        assert several.status == "invalid_args"
        assert several.summary == (
            "Which timer should I cancel? 2 timers are running. Eggs, 5 minutes left. "
            "10 minutes left."
        )
        assert len(await rig.live()) == 2

        labelled = await _call(rig.timers, "timers.cancel", {"label": "eggs"}, clock=rig.clock)
        assert (labelled.status, labelled.summary) == ("ok", "Timer cancelled.")
        everything = await _call(rig.timers, "timers.cancel", {"all": True}, clock=rig.clock)
        assert (everything.status, everything.summary) == ("ok", "1 timer cancelled.")
        assert await rig.live() == []

    @pytest.mark.asyncio
    async def test_cancel_by_id(self, rig):
        await _call(rig.timers, "timers.start", {"minutes": 10}, clock=rig.clock)
        [occurrence] = await rig.live()

        missing = await _call(rig.timers, "timers.cancel", {"timer_id": "nope"}, clock=rig.clock)
        assert missing.status == "invalid_args"
        outcome = await _call(
            rig.timers, "timers.cancel", {"timer_id": occurrence.occurrence_id}, clock=rig.clock
        )
        assert (outcome.status, outcome.summary) == ("ok", "Timer cancelled.")

    @pytest.mark.asyncio
    async def test_stop_ends_the_ringing_timer(self, rig):
        nothing = await _call(rig.timers, "timers.stop", clock=rig.clock)
        assert (nothing.status, nothing.summary) == ("invalid_args", "No timer is ringing.")

        await _ringing_timer(rig)
        assert rig.timers.active() is True

        outcome = await _call(rig.timers, "timers.stop", clock=rig.clock)

        assert (outcome.status, outcome.summary) == ("ok", "Timer stopped.")
        assert rig.timers.active() is False


# -- Clock -------------------------------------------------------------------------------------


class TestClock:
    @pytest.mark.asyncio
    async def test_now_reads_the_local_time_date_and_weekday(self):
        clock = FakeClock(datetime(2026, 10, 23, 23, 42, tzinfo=UTC))
        zone = UserZone(name="UTC-4", tz=timezone(timedelta(hours=-4)), reason=None)
        capability = ClockCapability(clock=clock, zone=zone)

        outcome = await _call(capability, "clock.now", clock=clock)

        assert (outcome.status, outcome.summary) == ("ok", "It's 7:42 PM, Friday October 23.")
        assert outcome.data == {
            "local_time": "2026-10-23T19:42:00-04:00",
            "time": "19:42",
            "date": "2026-10-23",
            "weekday": "Friday",
            "tz": "UTC-4",
        }

    @pytest.mark.parametrize(
        ("hour", "minute", "spoken"),
        [(0, 5, "12:05 AM"), (12, 0, "12:00 PM"), (9, 30, "9:30 AM"), (23, 59, "11:59 PM")],
    )
    @pytest.mark.asyncio
    async def test_the_spoken_clock_is_twelve_hour(self, hour, minute, spoken):
        clock = FakeClock(datetime(2026, 1, 5, hour, minute, tzinfo=UTC))
        capability = ClockCapability(clock=clock, zone=ZONE)

        outcome = await _call(capability, "clock.now", clock=clock)

        assert outcome.summary == f"It's {spoken}, Monday January 5."

    def test_the_clock_rows_match(self):
        capability = ClockCapability(clock=FakeClock(), zone=ZONE)

        assert capability.match_intent("whats the time") == FastIntent("clock.now", {})
        assert capability.match_intent("what time does it open") is None


# -- Fast path through the capabilities --------------------------------------------------------


class TestFastPath:
    @pytest.mark.asyncio
    async def test_alarms_and_timers_match_their_rows(self, rig):
        assert rig.alarms.match_intent(_norm("wake me up at 6:30 tomorrow")) == FastIntent(
            "alarms.set", {"time": "6:30", "day": "tomorrow"}
        )
        assert rig.alarms.match_intent(_norm("10 minute timer")) is None
        assert rig.timers.match_intent(_norm("10 minute timer")) == FastIntent(
            "timers.start", {"minutes": 10}
        )
        assert rig.timers.match_intent(_norm("set an alarm for 7")) is None

    @pytest.mark.asyncio
    async def test_a_fast_path_alarm_runs_through_the_host(self, rig):
        host = _host([rig.alarms, rig.timers, rig.clock_capability], rig.clock)
        await host.start()
        await host.wait_started()
        try:
            resolution = await host.pre_llm("s1", "t1", "Set an alarm for 7")
            assert resolution is not None and resolution.reply == "Alarm set for 7:00 PM today."
            timer = await host.pre_llm("s1", "t2", "10 minute timer")
            assert timer is not None and timer.reply == "Timer set for 10 minutes."
            clock = await host.pre_llm("s1", "t3", "what's the time")
            assert clock is not None and clock.reply == "It's 7:00 AM, Monday January 5."
        finally:
            await host.stop()

    @pytest.mark.asyncio
    async def test_the_tag_rows_match_only_the_tags_the_store_has_saved(self, rig):
        assert rig.alarms.match_intent(_norm("set my gym alarm")) is None
        assert rig.timers.match_intent(_norm("start my pasta timer")) is None

        await _call(rig.alarms, "alarms.set", {"time": "6am", "label": "Gym"}, clock=rig.clock)
        await _call(rig.timers, "timers.start", {"minutes": 9, "label": "Pasta"}, clock=rig.clock)

        assert rig.alarms.match_intent(_norm("set my GYM alarm for 6:30")) == FastIntent(
            "alarms.set", {"label": "gym", "time": "6:30"}
        )
        assert rig.alarms.match_intent(_norm("cancel the gym alarm")) == FastIntent(
            "alarms.cancel", {"label": "gym"}
        )
        assert rig.timers.match_intent(_norm("start my pasta timer")) == FastIntent(
            "timers.start", {"label": "pasta"}
        )
        assert rig.alarms.match_intent(_norm("set my pasta alarm")) is None
        assert rig.timers.match_intent(_norm("start my gym timer")) is None

    @pytest.mark.asyncio
    async def test_a_repeat_said_on_the_fast_path_arms_the_weekday_set(self, rig):
        host = _host(_capabilities(rig), rig.clock)
        await host.start()
        await host.wait_started()
        try:
            resolution = await host.pre_llm(
                "s1", "t1", "set an alarm for 7am every Monday and Thursday"
            )
        finally:
            await host.stop()

        assert resolution is not None and resolution.handled
        assert resolution.reply == "Alarm set for 7:00 AM every Monday and Thursday."
        [occurrence] = await rig.live()
        assert occurrence.record.days == frozenset({0, 3})

    @pytest.mark.asyncio
    async def test_reads_and_cancels_run_with_no_decode_at_the_off_level(self, rig):
        tiers = FakeTiers({})
        host = _host(_capabilities(rig), rig.clock, power=FakePower("off"), tiers=tiers)
        for hour in range(1, 8):
            await _call(rig.alarms, "alarms.set", {"time": f"{hour}pm"}, clock=rig.clock)
        await _call(rig.timers, "timers.start", {"minutes": 10}, clock=rig.clock)
        await host.start()
        await host.wait_started()
        # `pre_llm` takes no decode function, so a handled turn there ran no model; the status
        # broadcast reads the tiers for `decode_available`, which is why `asked` is not empty.
        assert host.decode_available() is False
        try:
            listed = await host.pre_llm("s1", "t1", "List my alarms")
            cancelled = await host.pre_llm("s1", "t2", "cancel the timer")
        finally:
            await host.stop()

        assert listed is not None and listed.handled
        assert listed.reply.startswith("You have 7 alarms. 1:00 PM today.")
        assert listed.reply.endswith("5:00 PM today, and 2 more on screen.")
        assert "6:00 PM" not in listed.reply
        assert cancelled is not None and cancelled.handled
        assert cancelled.reply == "Timer cancelled."
        assert [
            (r.tool, r.source, r.power_level) for r in (*listed.records, *cancelled.records)
        ] == [
            ("alarms.list", "fastpath", "off"),
            ("timers.cancel", "fastpath", "off"),
        ]
        assert [o.record.kind for o in await rig.live()] == ["alarm"] * 7
        assert set(tiers.asked) == {"interaction"}

    @pytest.mark.asyncio
    async def test_cancel_the_timer_takes_the_one_live_timer_or_asks_which(self, rig):
        host = _host(_capabilities(rig), rig.clock)
        await host.start()
        await host.wait_started()
        try:
            none = await host.pre_llm("s1", "t1", "cancel the timer")
            await _call(rig.timers, "timers.start", {"minutes": 10}, clock=rig.clock)
            one = await host.pre_llm("s1", "t2", "cancel my timer")
            await _call(rig.timers, "timers.start", {"minutes": 10}, clock=rig.clock)
            await _call(
                rig.timers, "timers.start", {"minutes": 5, "label": "eggs"}, clock=rig.clock
            )
            several = await host.pre_llm("s1", "t3", "cancel the timer")
        finally:
            await host.stop()

        assert none is not None and none.reply == "No timer is running."
        assert one is not None and one.reply == "Timer cancelled."
        assert several is not None and several.handled
        [record] = several.records
        assert record.outcome.status == "invalid_args"
        assert record.outcome.summary == (
            "Which timer should I cancel? 2 timers are running. Eggs, 5 minutes left. "
            "10 minutes left."
        )
        assert len(await rig.live()) == 2


# -- SchedulerAlarmControl ---------------------------------------------------------------------


NOT_RUNNING = [
    ("failing", "alarm records failed (OperationalError)"),
    ("dead", "alarm scheduler task died: RuntimeError"),
    ("stopped", None),
]


def _expected_reason(state: str, reason: str | None) -> str:
    return reason or f"alarm scheduler {state}"


class TestSchedulerAlarmControl:
    @pytest.mark.asyncio
    async def test_snapshot_reads_rev_and_alarms_in_one_read(self, rig):
        await _call(rig.alarms, "alarms.set", {"time": "7am"}, clock=rig.clock)
        await _call(rig.timers, "timers.start", {"minutes": 5}, clock=rig.clock)
        rig.records.snapshots = 0
        rig.records.revs = 0

        payload = await rig.control.snapshot()

        assert (rig.records.snapshots, rig.records.revs) == (1, 0)
        assert payload["type"] == "alarm_state"
        assert payload["rev"] == 2
        assert sorted(a["kind"] for a in payload["alarms"]) == ["alarm", "timer"]  # type: ignore[union-attr]
        assert payload["snooze_minutes"] == 9
        assert payload["tz"] == "UTC"

    @pytest.mark.parametrize(("state", "reason"), NOT_RUNNING)
    @pytest.mark.asyncio
    async def test_snapshot_is_unavailable_unless_the_scheduler_runs(self, rig, state, reason):
        rig.scheduler.set_health(state, reason)  # type: ignore[arg-type]

        with pytest.raises(CapabilityUnavailableError) as raised:
            await rig.control.snapshot()

        assert str(raised.value) == _expected_reason(state, reason)
        assert str(raised.value) == rig.alarms.status().reason

    @pytest.mark.asyncio
    async def test_snapshot_over_an_unwritable_database_is_a_store_error(self, tmp_path):
        built = _rig(_unwritable(tmp_path))

        with pytest.raises(CapabilityStoreError, match="alarm records"):
            await built.control.snapshot()

    @pytest.mark.asyncio
    async def test_stop_ends_a_ringing_alarm_with_the_endpoint_as_actor(self, rig):
        alarm_id = await _ringing_alarm(rig)

        outcome = await rig.control.act(alarm_id, "stop", None, endpoint="kitchen")

        assert (outcome.status, outcome.summary) == ("ok", "Alarm stopped.")
        assert await rig.live() == []
        assert (await rig.events())[-1] == ("stopped", "alarm_action", "kitchen")

    @pytest.mark.asyncio
    async def test_snooze_takes_minutes_or_the_configured_default(self, rig):
        alarm_id = await _ringing_alarm(rig)

        outcome = await rig.control.act(alarm_id, "snooze", None, endpoint="desktop")

        assert (outcome.status, outcome.summary) == ("ok", "Snoozed for 9 minutes.")
        [occurrence] = await rig.live()
        assert occurrence.next_fire_at == rig.clock.now() + timedelta(minutes=9)
        assert (await rig.events())[-1] == ("snoozed", "alarm_action", "desktop")

        await rig.clock.advance(9 * 60)
        await rig.ring(alarm_id)
        again = await rig.control.act(alarm_id, "snooze", 3, endpoint="desktop")
        assert again.summary == "Snoozed for 3 minutes."

    @pytest.mark.parametrize("action", ["stop", "snooze"])
    @pytest.mark.asyncio
    async def test_stop_or_snooze_on_an_alarm_not_ringing_is_invalid(self, rig, action):
        await _call(rig.alarms, "alarms.set", {"time": "7am"}, clock=rig.clock)
        [occurrence] = await rig.live()

        outcome = await rig.control.act(
            occurrence.occurrence_id, action, None, endpoint="desktop"  # type: ignore[arg-type]
        )

        assert outcome.status == "invalid_args"
        assert occurrence.occurrence_id in outcome.summary
        assert [o.state for o in await rig.live()] == ["scheduled"]

    @pytest.mark.asyncio
    async def test_a_snooze_length_out_of_range_is_invalid(self, rig):
        alarm_id = await _ringing_alarm(rig)

        outcome = await rig.control.act(alarm_id, "snooze", 0, endpoint="desktop")

        assert outcome.status == "invalid_args"
        assert [o.state for o in await rig.live()] == ["ringing"]

    @pytest.mark.asyncio
    async def test_dismiss_acknowledges_a_missed_alarm(self, rig):
        await _call(rig.alarms, "alarms.set", {"time": "7:01am", "day": "today"}, clock=rig.clock)
        [occurrence] = await rig.live()
        await rig.clock.advance(3600)
        await rig.miss(occurrence.occurrence_id)

        outcome = await rig.control.act(
            occurrence.occurrence_id, "dismiss", None, endpoint="desktop"
        )

        assert (outcome.status, outcome.summary) == ("ok", "Missed alarm dismissed.")
        assert await rig.live() == []
        payload = await rig.control.snapshot()
        assert payload["missed"] == []

    @pytest.mark.asyncio
    async def test_dismiss_on_a_scheduled_alarm_is_invalid(self, rig):
        await _call(rig.alarms, "alarms.set", {"time": "7am"}, clock=rig.clock)
        [occurrence] = await rig.live()

        outcome = await rig.control.act(
            occurrence.occurrence_id, "dismiss", None, endpoint="desktop"
        )

        assert outcome.status == "invalid_args"
        assert occurrence.occurrence_id in outcome.summary
        assert [o.state for o in await rig.live()] == ["scheduled"]

    @pytest.mark.parametrize("action", ["stop", "snooze", "dismiss"])
    @pytest.mark.parametrize(("state", "reason"), NOT_RUNNING)
    @pytest.mark.asyncio
    async def test_act_is_unavailable_unless_the_scheduler_runs(self, rig, action, state, reason):
        alarm_id = await _ringing_alarm(rig)
        rig.scheduler.set_health(state, reason)  # type: ignore[arg-type]

        with pytest.raises(CapabilityUnavailableError) as raised:
            await rig.control.act(alarm_id, action, None, endpoint="desktop")  # type: ignore[arg-type]

        assert str(raised.value) == rig.alarms.status().reason
        assert [o.state for o in await rig.live()] == ["ringing"]

    @pytest.mark.asyncio
    async def test_act_over_an_unwritable_database_is_a_store_error(self, tmp_path):
        built = _rig(_unwritable(tmp_path))

        with pytest.raises(CapabilityStoreError, match="alarm records"):
            await built.control.act("any", "stop", None, endpoint="desktop")

    @pytest.mark.asyncio
    async def test_report_goes_through_the_scheduler(self, rig):
        alarm_id = await _ringing_alarm(rig)
        at = rig.clock.now()

        result = await rig.control.report(
            alarm_id, endpoint="desktop", outcome="stopped", at=at, rang_at=at, snoozed_until=None
        )

        assert result == "applied"
        assert await rig.live() == []
        rig.scheduler.set_health("dead", "alarm scheduler task died: RuntimeError")
        with pytest.raises(CapabilityUnavailableError):
            await rig.control.report(
                alarm_id, endpoint="desktop", outcome="rang", at=at, rang_at=at, snoozed_until=None
            )
