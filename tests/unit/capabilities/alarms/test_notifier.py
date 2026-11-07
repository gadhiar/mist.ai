"""Tests for the alarm notifiers: the `alarm_fired` and `alarm_state` payloads and the fan-out."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta, timezone

import pytest

from backend.capabilities.alarms.notifier import (
    CompositeNotifier,
    WsAlarmNotifier,
    alarm_state_payload,
)
from backend.capabilities.alarms.store import Alarm
from backend.capabilities.alarms.timeparse import RepeatRule
from backend.capabilities.events import AlarmFiredEvent, AlarmStateEvent
from backend.errors import CapabilityStoreError
from tests.unit.capabilities.fakes import RecordingEventSink

T0 = datetime(2026, 1, 5, 7, 0, tzinfo=UTC)
NOTIFIER_LOGGER = "backend.capabilities.alarms.notifier"


def _alarm(alarm_id: str, state: str = "scheduled", **changes: object) -> Alarm:
    fields: dict[str, object] = {
        "alarm_id": alarm_id,
        "kind": "alarm",
        "label": None,
        "state": state,
        "scheduled_for": T0,
        "next_fire_at": T0,
        "repeat": None,
        "late_by_s": None,
        "fired_at": None,
    }
    fields.update(changes)
    return Alarm(**fields)  # type: ignore[arg-type]


class RecordingNotifier:
    """An `AlarmNotifier` that records each call, or raises `error` when it is set."""

    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    async def ring(self, alarm: Alarm, ring_timeout_s: float) -> None:
        self.calls.append(("ring", (alarm, ring_timeout_s)))
        if self.error is not None:
            raise self.error

    async def stopped(self, alarm_id: str) -> None:
        self.calls.append(("stopped", (alarm_id,)))
        if self.error is not None:
            raise self.error

    async def state_changed(self, alarms: list[Alarm], rev: int, at: datetime) -> None:
        self.calls.append(("state_changed", (alarms, rev, at)))
        if self.error is not None:
            raise self.error


class TestAlarmStatePayload:
    def test_lists_live_alarms_with_ringing_and_missed_entries(self):
        rule = RepeatRule(kind="weekdays", hour=7, minute=0)
        scheduled = _alarm(
            "a1",
            label="gym",
            scheduled_for=T0 + timedelta(hours=1),
            next_fire_at=T0 + timedelta(hours=1),
            repeat=rule,
        )
        ringing = _alarm(
            "a2",
            "ringing",
            kind="timer",
            late_by_s=120.5,
            fired_at=T0,
            scheduled_for=T0 - timedelta(seconds=120.5),
            next_fire_at=T0 - timedelta(seconds=120.5),
        )
        missed = _alarm(
            "a3",
            "missed",
            label="pills",
            scheduled_for=T0 - timedelta(hours=2),
            next_fire_at=T0 - timedelta(hours=2),
        )
        snoozed = _alarm("a4", "snoozed", next_fire_at=T0 + timedelta(minutes=9))

        payload = alarm_state_payload([scheduled, ringing, missed, snoozed], 17, T0)

        assert payload == {
            "type": "alarm_state",
            "rev": 17,
            "alarms": [
                {
                    "alarm_id": "a1",
                    "kind": "alarm",
                    "label": "gym",
                    "state": "scheduled",
                    "next_fire_at": "2026-01-05T08:00:00Z",
                    "repeat": "weekdays",
                },
                {
                    "alarm_id": "a2",
                    "kind": "timer",
                    "label": None,
                    "state": "ringing",
                    "next_fire_at": "2026-01-05T06:57:59.500000Z",
                    "repeat": "none",
                },
                {
                    "alarm_id": "a3",
                    "kind": "alarm",
                    "label": "pills",
                    "state": "missed",
                    "next_fire_at": "2026-01-05T05:00:00Z",
                    "repeat": "none",
                },
                {
                    "alarm_id": "a4",
                    "kind": "alarm",
                    "label": None,
                    "state": "snoozed",
                    "next_fire_at": "2026-01-05T07:09:00Z",
                    "repeat": "none",
                },
            ],
            "ringing": [{"alarm_id": "a2", "late_by_s": 120.5}],
            "missed": [
                {"alarm_id": "a3", "label": "pills", "scheduled_for": "2026-01-05T05:00:00Z"}
            ],
            "at": "2026-01-05T07:00:00Z",
        }

    def test_validates_against_the_alarm_state_model(self):
        alarms = [
            _alarm("a1", repeat=RepeatRule(kind="daily", hour=7, minute=0)),
            _alarm("a2", "ringing", kind="timer", late_by_s=12.0, fired_at=T0),
            _alarm("a3", "missed", label="pills"),
            _alarm("a4", "snoozed"),
        ]

        payload = alarm_state_payload(alarms, 9, T0)

        assert AlarmStateEvent.model_validate(payload).to_payload() == payload

    def test_a_ringing_alarm_without_late_by_lists_zero(self):
        payload = alarm_state_payload([_alarm("a1", "ringing", fired_at=T0)], 2, T0)

        assert payload["ringing"] == [{"alarm_id": "a1", "late_by_s": 0.0}]

    def test_leaves_out_done_alarms(self):
        payload = alarm_state_payload([_alarm("old", "done"), _alarm("new")], 3, T0)

        assert [entry["alarm_id"] for entry in payload["alarms"]] == ["new"]  # type: ignore[index, union-attr]
        assert payload["ringing"] == []
        assert payload["missed"] == []

    def test_renders_times_in_utc(self):
        local = datetime(2026, 1, 5, 14, 30, tzinfo=timezone(timedelta(hours=5)))
        payload = alarm_state_payload([_alarm("a", next_fire_at=local)], 1, local)

        assert payload["at"] == "2026-01-05T09:30:00Z"
        assert payload["alarms"][0]["next_fire_at"] == "2026-01-05T09:30:00Z"  # type: ignore[index]


class TestWsAlarmNotifier:
    @pytest.mark.asyncio
    async def test_ring_emits_alarm_fired(self):
        sink = RecordingEventSink()
        alarm = _alarm(
            "a1",
            "ringing",
            kind="timer",
            label="tea",
            scheduled_for=T0,
            next_fire_at=T0,
            fired_at=T0 + timedelta(seconds=90),
            late_by_s=90.0,
        )

        await WsAlarmNotifier(sink).ring(alarm, 600.0)

        assert sink.payloads == [
            {
                "type": "alarm_fired",
                "alarm_id": "a1",
                "kind": "timer",
                "label": "tea",
                "scheduled_for": "2026-01-05T07:00:00Z",
                "fired_at": "2026-01-05T07:01:30Z",
                "late_by_s": 90.0,
                "ring_timeout_s": 600.0,
            }
        ]
        assert AlarmFiredEvent.model_validate(sink.payloads[0]).to_payload() == sink.payloads[0]

    @pytest.mark.asyncio
    async def test_ring_without_fired_at_or_late_by_reports_next_fire_at_and_zero(self):
        sink = RecordingEventSink()
        alarm = _alarm("a1", "ringing", scheduled_for=T0, next_fire_at=T0 + timedelta(minutes=9))

        await WsAlarmNotifier(sink).ring(alarm, 600.0)

        [payload] = sink.payloads
        assert payload["scheduled_for"] == "2026-01-05T07:00:00Z"
        assert payload["fired_at"] == "2026-01-05T07:09:00Z"
        assert payload["late_by_s"] == 0.0

    @pytest.mark.asyncio
    async def test_state_changed_emits_exactly_the_shared_payload(self):
        sink = RecordingEventSink()
        alarms = [
            _alarm("a1"),
            _alarm("a2", "ringing", late_by_s=75.0, fired_at=T0 + timedelta(seconds=75)),
        ]

        await WsAlarmNotifier(sink).state_changed(alarms, 42, T0)

        assert sink.payloads == [alarm_state_payload(alarms, 42, T0)]
        assert sink.payloads[0]["rev"] == 42
        assert sink.payloads[0]["ringing"] == [{"alarm_id": "a2", "late_by_s": 75.0}]

    @pytest.mark.asyncio
    async def test_stopped_emits_nothing_because_alarm_state_follows(self):
        sink = RecordingEventSink()

        await WsAlarmNotifier(sink).stopped("a1")

        assert sink.payloads == []


class TestCompositeNotifier:
    @pytest.mark.asyncio
    async def test_fans_out_every_call_to_every_notifier(self):
        first, second = RecordingNotifier(), RecordingNotifier()
        composite = CompositeNotifier([first, second])
        alarm = _alarm("a1", "ringing", late_by_s=0.0, fired_at=T0)

        await composite.ring(alarm, 600.0)
        await composite.stopped("a1")
        await composite.state_changed([alarm], 5, T0)

        expected = [
            ("ring", (alarm, 600.0)),
            ("stopped", ("a1",)),
            ("state_changed", ([alarm], 5, T0)),
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
        alarm = _alarm("a1", "ringing", late_by_s=0.0, fired_at=T0)

        with caplog.at_level(logging.ERROR, logger=NOTIFIER_LOGGER):
            await composite.ring(alarm, 600.0)
            await composite.stopped("a1")
            await composite.state_changed([alarm], 5, T0)

        assert [name for name, _ in healthy.calls] == ["ring", "stopped", "state_changed"]
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 3
        assert all("RecordingNotifier" in r.getMessage() for r in errors)

    @pytest.mark.asyncio
    async def test_an_error_outside_the_contained_set_propagates(self):
        composite = CompositeNotifier([RecordingNotifier(error=RuntimeError("bug"))])

        with pytest.raises(RuntimeError, match="bug"):
            await composite.stopped("a1")
