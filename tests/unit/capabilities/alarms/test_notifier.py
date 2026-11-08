"""Tests for the alarm notifiers: the `alarm_fired` and `alarm_state` payloads and the fan-out."""

from __future__ import annotations

import dataclasses
import logging
import zoneinfo
from datetime import UTC, datetime, time, timedelta, timezone, tzinfo

import pytest

from backend.capabilities.alarms.notifier import (
    CompositeNotifier,
    WsAlarmNotifier,
    alarm_state_payload,
    effective_grace_s,
)
from backend.capabilities.alarms.records import AlarmRecord, Occurrence
from backend.capabilities.config import AlarmSettings
from backend.capabilities.events import AlarmFiredEvent, AlarmStateEvent
from backend.errors import CapabilityStoreError
from tests.unit.capabilities.fakes import RecordingEventSink

T0 = datetime(2026, 1, 5, 7, 0, tzinfo=UTC)
NOTIFIER_LOGGER = "backend.capabilities.alarms.notifier"
SETTINGS = AlarmSettings(ring_timeout_s=600.0, snooze_minutes=9, missed_grace_s=900.0)
FALLBACK_AFTER_S = 30.0


def _record(record_id: str = "r1", **changes: object) -> AlarmRecord:
    fields: dict[str, object] = {
        "record_id": record_id,
        "kind": "alarm",
        "category": "one_off",
        "identity": "alarm:one_off:07:00",
        "label": None,
        "wall_time": time(7, 0),
        "days": None,
        "duration_s": None,
        "ring_on": "all",
        "use_count": 1,
        "fire_count": 0,
        "created_at": T0 - timedelta(days=1),
        "last_set_at": T0 - timedelta(days=1),
        "last_fired_at": None,
        "last_outcome": None,
        "active": True,
    }
    fields.update(changes)
    return AlarmRecord(**fields)  # type: ignore[arg-type]


def _timer_record(record_id: str = "r2", **changes: object) -> AlarmRecord:
    fields: dict[str, object] = {
        "kind": "timer",
        "identity": "timer:one_off:600",
        "wall_time": None,
        "duration_s": 600,
    }
    fields.update(changes)
    return _record(record_id, **fields)


def _occurrence(
    occurrence_id: str,
    state: str = "scheduled",
    record: AlarmRecord | None = None,
    **changes: object,
) -> Occurrence:
    fields: dict[str, object] = {
        "occurrence_id": occurrence_id,
        "record": record if record is not None else _record(),
        "state": state,
        "ring_on": "all",
        "scheduled_for": T0,
        "next_fire_at": T0,
        "fired_at": None,
        "late_by_s": None,
        "missed_at": None,
        "missed_reason": None,
        "sent_to": frozenset(),
        "delivered_to": frozenset(),
    }
    fields.update(changes)
    return Occurrence(**fields)  # type: ignore[arg-type]


def _payload(
    occurrences: list[Occurrence],
    rev: int = 1,
    now: datetime = T0,
    *,
    settings: AlarmSettings = SETTINGS,
    tz: tzinfo = UTC,
) -> dict[str, object]:
    return alarm_state_payload(
        occurrences, rev, now, settings=settings, tz=tz, fallback_after_s=FALLBACK_AFTER_S
    )


def _ws(sink: RecordingEventSink, tz: tzinfo = UTC) -> WsAlarmNotifier:
    return WsAlarmNotifier(sink, SETTINGS, tz, fallback_after_s=FALLBACK_AFTER_S)


class RecordingNotifier:
    """An `AlarmNotifier` that records each call, or raises `error` when it is set."""

    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    async def ring(self, occurrence: Occurrence, ring_timeout_s: float) -> None:
        self.calls.append(("ring", (occurrence, ring_timeout_s)))
        if self.error is not None:
            raise self.error

    async def stopped(self, occurrence_id: str) -> None:
        self.calls.append(("stopped", (occurrence_id,)))
        if self.error is not None:
            raise self.error

    async def state_changed(self, occurrences: list[Occurrence], rev: int, at: datetime) -> None:
        self.calls.append(("state_changed", (occurrences, rev, at)))
        if self.error is not None:
            raise self.error


class TestAlarmStatePayload:
    def test_lists_live_occurrences_with_the_1_4_0_fields(self):
        gym = _record(
            "r-gym",
            category="tagged",
            identity="alarm:tagged:gym",
            label="Gym",
            wall_time=time(6, 30),
            days=frozenset({3, 0}),
            ring_on=("bedroom", "desktop"),
        )
        scheduled = _occurrence(
            "o1",
            record=gym,
            ring_on=("bedroom", "desktop"),
            scheduled_for=T0 + timedelta(hours=1),
            next_fire_at=T0 + timedelta(hours=1),
        )
        ringing = _occurrence(
            "o2",
            "ringing",
            record=_timer_record(),
            late_by_s=120.5,
            fired_at=T0,
            scheduled_for=T0 - timedelta(seconds=120.5),
            next_fire_at=T0 - timedelta(seconds=120.5),
        )
        pills = _record("r-pills", category="tagged", identity="alarm:tagged:pills", label="pills")
        missed = _occurrence(
            "o3",
            "missed",
            record=pills,
            scheduled_for=T0 - timedelta(hours=2),
            next_fire_at=T0 - timedelta(hours=2),
            missed_at=T0 - timedelta(hours=2),
            missed_reason="unheard",
        )
        snoozed = _occurrence("o4", "snoozed", next_fire_at=T0 + timedelta(minutes=9))

        payload = _payload([scheduled, ringing, missed, snoozed], 17)

        assert payload == {
            "type": "alarm_state",
            "rev": 17,
            "alarms": [
                {
                    "alarm_id": "o1",
                    "record_id": "r-gym",
                    "kind": "alarm",
                    "category": "tagged",
                    "label": "Gym",
                    "state": "scheduled",
                    "next_fire_at": "2026-01-05T08:00:00Z",
                    "wall_time": "06:30",
                    "repeat": "custom",
                    "days": [0, 3],
                    "duration_s": None,
                    "ring_on": ["bedroom", "desktop"],
                },
                {
                    "alarm_id": "o2",
                    "record_id": "r2",
                    "kind": "timer",
                    "category": "one_off",
                    "label": None,
                    "state": "ringing",
                    "next_fire_at": "2026-01-05T06:57:59.500000Z",
                    "wall_time": None,
                    "repeat": "none",
                    "days": None,
                    "duration_s": 600,
                    "ring_on": "all",
                },
                {
                    "alarm_id": "o3",
                    "record_id": "r-pills",
                    "kind": "alarm",
                    "category": "tagged",
                    "label": "pills",
                    "state": "missed",
                    "next_fire_at": "2026-01-05T05:00:00Z",
                    "wall_time": "07:00",
                    "repeat": "none",
                    "days": None,
                    "duration_s": None,
                    "ring_on": "all",
                },
                {
                    "alarm_id": "o4",
                    "record_id": "r1",
                    "kind": "alarm",
                    "category": "one_off",
                    "label": None,
                    "state": "snoozed",
                    "next_fire_at": "2026-01-05T07:09:00Z",
                    "wall_time": "07:00",
                    "repeat": "none",
                    "days": None,
                    "duration_s": None,
                    "ring_on": "all",
                },
            ],
            "ringing": [{"alarm_id": "o2", "late_by_s": 120.5}],
            "missed": [
                {
                    "alarm_id": "o3",
                    "record_id": "r-pills",
                    "label": "pills",
                    "scheduled_for": "2026-01-05T05:00:00Z",
                    "reason": "unheard",
                }
            ],
            "fallback_after_s": 30.0,
            "missed_grace_s": 900.0,
            "ring_timeout_s": 600.0,
            "snooze_minutes": 9,
            "tz": "UTC",
            "at": "2026-01-05T07:00:00Z",
        }

    @pytest.mark.parametrize(
        ("days", "repeat"),
        [
            (None, "none"),
            (frozenset(range(7)), "daily"),
            (frozenset(range(5)), "weekdays"),
            (frozenset({5, 6}), "weekends"),
            (frozenset({0, 3}), "custom"),
            (frozenset({6}), "custom"),
        ],
    )
    def test_names_the_repeat_of_each_weekday_set(self, days, repeat):
        record = _record(category="recurring" if days else "one_off", days=days)

        [entry] = _payload([_occurrence("o1", record=record)])["alarms"]  # type: ignore[misc]

        assert entry["repeat"] == repeat
        assert entry["days"] == (None if days is None else sorted(days))

    @pytest.mark.parametrize(("grace_s", "sent"), [(0.0, 60.0), (60.0, 60.0), (900.0, 900.0)])
    def test_sends_the_effective_grace_window(self, grace_s, sent):
        settings = dataclasses.replace(SETTINGS, missed_grace_s=grace_s)

        assert _payload([], settings=settings)["missed_grace_s"] == sent
        assert effective_grace_s(grace_s) == sent

    def test_names_the_time_zone_by_its_key(self):
        try:
            tz = zoneinfo.ZoneInfo("America/New_York")
        except zoneinfo.ZoneInfoNotFoundError:
            pytest.skip("tzdata absent")

        assert _payload([], tz=tz)["tz"] == "America/New_York"

    def test_validates_against_the_alarm_state_model(self):
        alarms = [
            _occurrence("o1", record=_record(category="recurring", days=frozenset(range(7)))),
            _occurrence("o2", "ringing", record=_timer_record(), late_by_s=12.0, fired_at=T0),
            _occurrence("o3", "missed", missed_at=T0, missed_reason="late"),
            _occurrence("o4", "snoozed"),
        ]

        payload = _payload(alarms, 9)

        assert AlarmStateEvent.model_validate(payload).to_payload() == payload

    def test_a_ringing_occurrence_without_late_by_lists_zero(self):
        payload = _payload([_occurrence("o1", "ringing", fired_at=T0)], 2)

        assert payload["ringing"] == [{"alarm_id": "o1", "late_by_s": 0.0}]

    def test_renders_times_in_utc(self):
        local = datetime(2026, 1, 5, 14, 30, tzinfo=timezone(timedelta(hours=5)))
        payload = _payload([_occurrence("o", next_fire_at=local)], 1, local)

        assert payload["at"] == "2026-01-05T09:30:00Z"
        assert payload["alarms"][0]["next_fire_at"] == "2026-01-05T09:30:00Z"  # type: ignore[index]


class TestWsAlarmNotifier:
    @pytest.mark.asyncio
    async def test_ring_emits_alarm_fired_with_the_1_4_0_fields(self):
        sink = RecordingEventSink()
        tea = _timer_record(category="tagged", identity="timer:tagged:tea", label="tea")
        occurrence = _occurrence(
            "o1",
            "ringing",
            record=tea,
            ring_on=("desktop",),
            scheduled_for=T0,
            next_fire_at=T0,
            fired_at=T0 + timedelta(seconds=90),
            late_by_s=90.0,
        )

        await _ws(sink).ring(occurrence, 600.0)

        assert sink.payloads == [
            {
                "type": "alarm_fired",
                "alarm_id": "o1",
                "record_id": "r2",
                "kind": "timer",
                "category": "tagged",
                "label": "tea",
                "scheduled_for": "2026-01-05T07:00:00Z",
                "fired_at": "2026-01-05T07:01:30Z",
                "late_by_s": 90.0,
                "ring_timeout_s": 600.0,
                "ring_on": ["desktop"],
                "fallback": False,
            }
        ]
        assert AlarmFiredEvent.model_validate(sink.payloads[0]).to_payload() == sink.payloads[0]

    @pytest.mark.asyncio
    async def test_ring_without_fired_at_or_late_by_reports_next_fire_at_and_zero(self):
        sink = RecordingEventSink()
        occurrence = _occurrence(
            "o1", "ringing", scheduled_for=T0, next_fire_at=T0 + timedelta(minutes=9)
        )

        await _ws(sink).ring(occurrence, 600.0)

        [payload] = sink.payloads
        assert payload["scheduled_for"] == "2026-01-05T07:00:00Z"
        assert payload["fired_at"] == "2026-01-05T07:09:00Z"
        assert payload["late_by_s"] == 0.0
        assert payload["ring_on"] == "all"

    @pytest.mark.asyncio
    async def test_state_changed_emits_exactly_the_shared_payload(self):
        sink = RecordingEventSink()
        alarms = [
            _occurrence("o1"),
            _occurrence("o2", "ringing", late_by_s=75.0, fired_at=T0 + timedelta(seconds=75)),
        ]

        await _ws(sink).state_changed(alarms, 42, T0)

        assert sink.payloads == [_payload(alarms, 42)]
        assert sink.payloads[0]["rev"] == 42
        assert sink.payloads[0]["ringing"] == [{"alarm_id": "o2", "late_by_s": 75.0}]

    @pytest.mark.asyncio
    async def test_stopped_emits_nothing_because_alarm_state_follows(self):
        sink = RecordingEventSink()

        await _ws(sink).stopped("o1")

        assert sink.payloads == []


class TestCompositeNotifier:
    @pytest.mark.asyncio
    async def test_fans_out_every_call_to_every_notifier(self):
        first, second = RecordingNotifier(), RecordingNotifier()
        composite = CompositeNotifier([first, second])
        occurrence = _occurrence("o1", "ringing", late_by_s=0.0, fired_at=T0)

        await composite.ring(occurrence, 600.0)
        await composite.stopped("o1")
        await composite.state_changed([occurrence], 5, T0)

        expected = [
            ("ring", (occurrence, 600.0)),
            ("stopped", ("o1",)),
            ("state_changed", ([occurrence], 5, T0)),
        ]
        assert first.calls == expected
        assert second.calls == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error", [OSError("socket gone"), CapabilityStoreError("sink store down")]
    )
    async def test_a_failing_notifier_does_not_stop_the_others(self, error, caplog):
        failing, healthy = RecordingNotifier(error=error), RecordingNotifier()
        composite = CompositeNotifier([failing, healthy])
        occurrence = _occurrence("o1", "ringing", late_by_s=0.0, fired_at=T0)

        with caplog.at_level(logging.ERROR, logger=NOTIFIER_LOGGER):
            await composite.ring(occurrence, 600.0)
            await composite.stopped("o1")
            await composite.state_changed([occurrence], 5, T0)

        assert [name for name, _ in healthy.calls] == ["ring", "stopped", "state_changed"]
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 3
        assert all("RecordingNotifier" in r.getMessage() for r in errors)

    @pytest.mark.asyncio
    async def test_an_error_outside_the_contained_set_propagates(self):
        composite = CompositeNotifier([RecordingNotifier(error=RuntimeError("bug"))])

        with pytest.raises(RuntimeError, match="bug"):
            await composite.stopped("o1")
