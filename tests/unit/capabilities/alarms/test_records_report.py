"""Tests for `AlarmRecords.report`: the D3 state table, duplicates and the lateness bound."""

from __future__ import annotations

import json
from datetime import UTC, datetime, time, timedelta

import pytest
import pytest_asyncio

from backend.capabilities.alarms.records import (
    REPORT_MAX_AGE,
    REPORT_MAX_AHEAD,
    AlarmRecords,
    ArmRequest,
    Occurrence,
    ReportResult,
)
from backend.capabilities.db import CapabilitiesDb

T0 = datetime(2026, 1, 5, 7, 0, tzinfo=UTC)
"""Monday 5 January 2026, 07:00 UTC: when the armed alarm is due."""
TUE_0700 = T0 + timedelta(days=1)
GRACE = timedelta(seconds=900)
"""The effective grace window, max(60 s, `missed_grace_s`) at its default."""
FALLBACK = timedelta(seconds=120)
"""How long an endpoint waits, disconnected, before it rings on its own."""
ENDPOINT = "desktop"


def alarm(fire_at: datetime = T0, *, days: frozenset[int] | None = None) -> ArmRequest:
    return ArmRequest(
        kind="alarm",
        fire_at=fire_at,
        wall_time=time(fire_at.hour, fire_at.minute),
        duration_s=None,
        label=None,
        days=days,
    )


def stamp(value: datetime) -> str:
    """A time as the store spells it."""
    return value.astimezone(UTC).isoformat(timespec="microseconds")


@pytest.fixture()
def db(tmp_path):
    database = CapabilitiesDb(str(tmp_path / "capabilities.db"))
    yield database
    database.close()


@pytest.fixture()
def records(db):
    store = AlarmRecords(db)
    store.migrate()
    return store


async def rows(db: CapabilitiesDb, sql: str, params: tuple[object, ...] = ()) -> list[tuple]:
    return await db.run(lambda conn: [tuple(r) for r in conn.execute(sql, params).fetchall()])


async def state(db: CapabilitiesDb) -> tuple:
    """Everything stored, so a test can tell that a report wrote nothing."""
    return (
        await rows(db, "SELECT * FROM alarm_records ORDER BY id"),
        await rows(db, "SELECT * FROM alarm_occurrences ORDER BY id"),
        await rows(db, "SELECT * FROM alarm_events ORDER BY seq"),
        await rows(db, "SELECT rev FROM alarm_records_rev"),
    )


async def kinds(db: CapabilitiesDb, occurrence_id: str) -> list[str]:
    found = await rows(
        db, "SELECT kind FROM alarm_events WHERE occurrence_id = ? ORDER BY seq", (occurrence_id,)
    )
    return [kind for (kind,) in found]


async def reported(db: CapabilitiesDb) -> list[tuple[str, str, str, str, dict]]:
    """Every `reported` event: occurrence id, `at`, source, endpoint and decoded detail."""
    found = await rows(
        db,
        "SELECT occurrence_id, at, source, endpoint, detail FROM alarm_events "
        "WHERE kind = 'reported' ORDER BY seq",
    )
    return [
        (o, at, source, endpoint, json.loads(detail)) for o, at, source, endpoint, detail in found
    ]


async def arm(records: AlarmRecords, request: ArmRequest | None = None) -> str:
    result = await records.arm(
        request or alarm(), at=T0 - timedelta(hours=8), tz=UTC, source="model"
    )
    return result.occurrence.occurrence_id


async def ring(records: AlarmRecords, occurrence_id: str, at: datetime = T0) -> None:
    assert await records.fire(
        occurrence_id, outcome="ringing", at=at, late_by_s=0.0, next_fire_at=None
    )


async def miss(records: AlarmRecords, occurrence_id: str, at: datetime) -> None:
    late = (at - T0).total_seconds()
    assert await records.fire(
        occurrence_id, outcome="missed", at=at, late_by_s=late, next_fire_at=None
    )


async def snooze(records: AlarmRecords, occurrence_id: str, until: datetime) -> None:
    assert await records.snooze(
        occurrence_id, until=until, at=T0, source="alarm_action", endpoint=ENDPOINT
    )


async def report(
    records: AlarmRecords,
    occurrence_id: str,
    outcome: str,
    at: datetime,
    *,
    rang_at: datetime | None = None,
    snoozed_until: datetime | None = None,
    now: datetime | None = None,
    next_fire_at: datetime | None = None,
    endpoint: str = ENDPOINT,
) -> ReportResult:
    return await records.report(
        occurrence_id,
        endpoint=endpoint,
        outcome=outcome,  # type: ignore[arg-type]
        at=at,
        rang_at=at if rang_at is None else rang_at,
        snoozed_until=snoozed_until,
        now=at if now is None else now,
        next_fire_at=next_fire_at,
        grace=GRACE,
    )


async def live(records: AlarmRecords) -> dict[str, Occurrence]:
    return {o.occurrence_id: o for o in (await records.snapshot())[1]}


async def assert_one_report(
    db: CapabilitiesDb,
    before: int,
    occurrence_id: str,
    *,
    outcome: str,
    at: datetime,
    rang_at: datetime,
    result: str,
) -> None:
    """The report appended exactly one `reported` event, keyed and described as D3 says."""
    after = await reported(db)
    assert len(after) == before + 1
    assert after[-1] == (
        occurrence_id,
        stamp(at),
        "alarm_report",
        ENDPOINT,
        {"outcome": outcome, "rang_at": stamp(rang_at), "result": result},
    )


class TestContract:
    def test_the_report_bounds_are_the_contract(self):
        assert timedelta(hours=24) == REPORT_MAX_AGE
        assert timedelta(seconds=60) == REPORT_MAX_AHEAD

    @pytest.mark.asyncio
    async def test_a_naive_time_is_refused_writing_nothing(self, records, db):
        occurrence_id = await arm(records)
        before = await state(db)

        with pytest.raises(ValueError):
            await report(records, occurrence_id, "rang", T0.replace(tzinfo=None))

        assert await state(db) == before

    @pytest.mark.asyncio
    async def test_an_unknown_outcome_is_refused(self, records):
        occurrence_id = await arm(records)

        with pytest.raises(ValueError):
            await report(records, occurrence_id, "dismissed", T0)

    @pytest.mark.asyncio
    async def test_snoozed_without_snoozed_until_is_refused(self, records):
        occurrence_id = await arm(records)

        with pytest.raises(ValueError):
            await report(records, occurrence_id, "snoozed", T0)


class TestScheduledAndDue:
    """The report counts as the firing, then its outcome applies."""

    @pytest.mark.asyncio
    async def test_rang_fires_it_and_acknowledges_the_endpoint(self, records, db):
        occurrence_id = await arm(records)
        rang_at = T0 + FALLBACK

        result = await report(records, occurrence_id, "rang", rang_at, now=rang_at + FALLBACK)

        assert result == "applied"
        occurrence = (await live(records))[occurrence_id]
        assert occurrence.state == "ringing"
        assert occurrence.delivered_to == frozenset({ENDPOINT})
        assert (occurrence.fired_at, occurrence.late_by_s) == (rang_at, FALLBACK.total_seconds())
        assert occurrence.record.fire_count == 1
        assert occurrence.record.last_fired_at == rang_at
        assert await kinds(db, occurrence_id) == ["set", "fired", "reported"]
        assert await records.rev() == 2
        await assert_one_report(
            db, 0, occurrence_id, outcome="rang", at=rang_at, rang_at=rang_at, result="applied"
        )

    @pytest.mark.asyncio
    async def test_a_repeating_record_gains_its_next_occurrence(self, records):
        occurrence_id = await arm(records, alarm(days=frozenset(range(7))))

        result = await report(records, occurrence_id, "rang", T0 + FALLBACK, next_fire_at=TUE_0700)

        assert result == "applied"
        occurrences = sorted((await live(records)).values(), key=lambda o: o.next_fire_at)
        assert [(o.state, o.scheduled_for) for o in occurrences] == [
            ("ringing", T0),
            ("scheduled", TUE_0700),
        ]
        assert occurrences[0].delivered_to == frozenset({ENDPOINT})
        assert occurrences[1].delivered_to == frozenset()

    @pytest.mark.asyncio
    async def test_a_next_occurrence_for_a_one_off_is_refused_writing_nothing(self, records, db):
        occurrence_id = await arm(records)
        before = await state(db)

        with pytest.raises(ValueError):
            await report(records, occurrence_id, "rang", T0 + FALLBACK, next_fire_at=TUE_0700)

        assert await state(db) == before

    @pytest.mark.asyncio
    async def test_stopped_fires_it_then_finishes_it(self, records, db):
        occurrence_id = await arm(records, alarm(days=frozenset(range(7))))
        rang_at, at = T0 + FALLBACK, T0 + timedelta(minutes=5)

        result = await report(
            records, occurrence_id, "stopped", at, rang_at=rang_at, next_fire_at=TUE_0700
        )

        assert result == "applied"
        [following] = (await live(records)).values()
        assert (following.state, following.scheduled_for) == ("scheduled", TUE_0700)
        assert following.record.fire_count == 1
        assert following.record.last_outcome == "stopped"
        assert await kinds(db, occurrence_id) == ["set", "fired", "stopped", "reported"]
        await assert_one_report(
            db, 0, occurrence_id, outcome="stopped", at=at, rang_at=rang_at, result="applied"
        )

    @pytest.mark.asyncio
    async def test_snoozed_fires_it_then_snoozes_it(self, records, db):
        occurrence_id = await arm(records)
        rang_at, at = T0 + FALLBACK, T0 + timedelta(minutes=3)
        until = at + timedelta(minutes=9)

        result = await report(
            records, occurrence_id, "snoozed", at, rang_at=rang_at, snoozed_until=until
        )

        assert result == "applied"
        occurrence = (await live(records))[occurrence_id]
        assert (occurrence.state, occurrence.next_fire_at) == ("snoozed", until)
        assert occurrence.record.fire_count == 1
        assert await kinds(db, occurrence_id) == ["set", "fired", "snoozed", "reported"]
        assert await records.rev() == 2


class TestScheduledNotDue:
    @pytest.mark.asyncio
    async def test_not_due_writes_nothing(self, records, db):
        occurrence_id = await arm(records)
        before = await state(db)

        result = await report(records, occurrence_id, "rang", T0 - timedelta(seconds=30))

        assert result == "not_due"
        assert await state(db) == before


class TestRinging:
    @pytest.mark.asyncio
    async def test_rang_adds_the_endpoint_to_delivered_to(self, records, db):
        occurrence_id = await arm(records)
        await ring(records, occurrence_id)
        rang_at = T0 + timedelta(seconds=1)

        result = await report(records, occurrence_id, "rang", rang_at)

        assert result == "applied"
        occurrence = (await live(records))[occurrence_id]
        assert (occurrence.state, occurrence.delivered_to) == ("ringing", frozenset({ENDPOINT}))
        assert occurrence.fired_at == T0
        assert await records.rev() == 3
        await assert_one_report(
            db, 0, occurrence_id, outcome="rang", at=rang_at, rang_at=rang_at, result="applied"
        )

    @pytest.mark.asyncio
    async def test_stopped_finishes_it_as_stopped(self, records, db):
        occurrence_id = await arm(records)
        await ring(records, occurrence_id)
        at = T0 + timedelta(seconds=20)

        result = await report(records, occurrence_id, "stopped", at, rang_at=T0)

        assert result == "applied"
        assert await live(records) == {}
        [record] = await records.saved()
        assert record.last_outcome == "stopped"
        assert await kinds(db, occurrence_id) == ["set", "fired", "stopped", "reported"]

    @pytest.mark.asyncio
    async def test_snoozed_snoozes_it(self, records, db):
        occurrence_id = await arm(records)
        await ring(records, occurrence_id)
        at = T0 + timedelta(seconds=20)
        until = at + timedelta(minutes=9)

        result = await report(
            records, occurrence_id, "snoozed", at, rang_at=T0, snoozed_until=until
        )

        assert result == "applied"
        occurrence = (await live(records))[occurrence_id]
        assert (occurrence.state, occurrence.next_fire_at) == ("snoozed", until)
        assert await kinds(db, occurrence_id) == ["set", "fired", "snoozed", "reported"]


class TestSnoozed:
    @pytest_asyncio.fixture()
    async def snoozed(self, records) -> str:
        occurrence_id = await arm(records)
        await ring(records, occurrence_id)
        await snooze(records, occurrence_id, T0 + timedelta(minutes=9))
        return occurrence_id

    @pytest.mark.asyncio
    async def test_rang_is_recorded_only(self, records, db, snoozed):
        before_occurrences = await rows(db, "SELECT * FROM alarm_occurrences")
        before_rev = await records.rev()
        rang_at = T0 + timedelta(seconds=1)

        result = await report(records, snoozed, "rang", rang_at)

        assert result == "applied"
        assert await rows(db, "SELECT * FROM alarm_occurrences") == before_occurrences
        assert await records.rev() == before_rev
        await assert_one_report(
            db, 0, snoozed, outcome="rang", at=rang_at, rang_at=rang_at, result="applied"
        )

    @pytest.mark.asyncio
    async def test_stopped_finishes_it(self, records, db, snoozed):
        result = await report(records, snoozed, "stopped", T0 + timedelta(minutes=2), rang_at=T0)

        assert result == "applied"
        assert await live(records) == {}
        assert (await kinds(db, snoozed))[-2:] == ["stopped", "reported"]

    @pytest.mark.asyncio
    async def test_snoozed_takes_the_later_until(self, records, db, snoozed):
        later = T0 + timedelta(minutes=15)

        result = await report(
            records, snoozed, "snoozed", T0 + timedelta(minutes=6), rang_at=T0, snoozed_until=later
        )

        assert result == "applied"
        assert (await live(records))[snoozed].next_fire_at == later
        assert (await kinds(db, snoozed))[-2:] == ["snoozed", "reported"]

    @pytest.mark.asyncio
    async def test_snoozed_keeps_the_later_until_it_has(self, records, db, snoozed):
        before_occurrences = await rows(db, "SELECT * FROM alarm_occurrences")
        before_rev = await records.rev()

        result = await report(
            records,
            snoozed,
            "snoozed",
            T0 + timedelta(minutes=1),
            rang_at=T0,
            snoozed_until=T0 + timedelta(minutes=5),
        )

        assert result == "applied"
        assert await rows(db, "SELECT * FROM alarm_occurrences") == before_occurrences
        assert await records.rev() == before_rev
        assert (await kinds(db, snoozed))[-1] == "reported"


class TestMissed:
    """The owner was down past the grace window, and the endpoint rang it on its own."""

    MISSED_AT = T0 + GRACE + timedelta(minutes=5)

    @pytest_asyncio.fixture()
    async def missed(self, records) -> str:
        occurrence_id = await arm(records)
        await miss(records, occurrence_id, self.MISSED_AT)
        return occurrence_id

    @pytest.mark.asyncio
    async def test_rang_ends_it_timed_out_as_heard(self, records, db, missed):
        rang_at = T0 + FALLBACK

        result = await report(records, missed, "rang", rang_at, now=self.MISSED_AT)

        assert result == "applied"
        assert await live(records) == {}
        [record] = await records.saved()
        assert record.last_outcome == "timed_out"
        assert (await kinds(db, missed))[-2:] == ["timed_out", "reported"]

    @pytest.mark.asyncio
    async def test_stopped_ends_it_stopped(self, records, db, missed):
        result = await report(
            records,
            missed,
            "stopped",
            T0 + timedelta(minutes=4),
            rang_at=T0 + FALLBACK,
            now=self.MISSED_AT,
        )

        assert result == "applied"
        assert await live(records) == {}
        [record] = await records.saved()
        assert record.last_outcome == "stopped"
        assert (await kinds(db, missed))[-2:] == ["stopped", "reported"]

    @pytest.mark.asyncio
    async def test_snoozed_until_a_later_time_snoozes_it(self, records, db, missed):
        rang_at, at = T0 + FALLBACK, T0 + timedelta(minutes=4)
        until = self.MISSED_AT + timedelta(minutes=10)

        result = await report(
            records,
            missed,
            "snoozed",
            at,
            rang_at=rang_at,
            snoozed_until=until,
            now=self.MISSED_AT,
        )

        assert result == "applied"
        occurrence = (await live(records))[missed]
        assert (occurrence.state, occurrence.next_fire_at) == ("snoozed", until)
        assert (occurrence.missed_at, occurrence.missed_reason) == (None, None)
        assert occurrence.fired_at == rang_at
        assert (await kinds(db, missed))[-2:] == ["snoozed", "reported"]

    @pytest.mark.asyncio
    async def test_snoozed_until_a_passed_time_ends_it_timed_out(self, records, db, missed):
        result = await report(
            records,
            missed,
            "snoozed",
            T0 + timedelta(minutes=4),
            rang_at=T0 + FALLBACK,
            snoozed_until=T0 + timedelta(minutes=13),
            now=self.MISSED_AT,
        )

        assert result == "applied"
        assert await live(records) == {}
        [record] = await records.saved()
        assert record.last_outcome == "timed_out"
        assert (await kinds(db, missed))[-2:] == ["timed_out", "reported"]


class TestEnded:
    @pytest.mark.asyncio
    async def test_already_ended_is_recorded(self, records, db):
        occurrence_id = await arm(records)
        await ring(records, occurrence_id)
        await records.finish(
            occurrence_id, outcome="stopped", at=T0, source="alarm_action", endpoint=ENDPOINT
        )
        records_before = await rows(db, "SELECT * FROM alarm_records")
        rev_before = await records.rev()
        at = T0 + timedelta(seconds=30)

        result = await report(records, occurrence_id, "stopped", at, rang_at=T0)

        assert result == "already_ended"
        assert await live(records) == {}
        assert await rows(db, "SELECT * FROM alarm_records") == records_before
        assert await records.rev() == rev_before
        await assert_one_report(
            db, 0, occurrence_id, outcome="stopped", at=at, rang_at=T0, result="already_ended"
        )
        [(record_id,)] = await rows(
            db, "SELECT record_id FROM alarm_events WHERE kind = 'reported'"
        )
        assert record_id == records_before[0][0]


class TestUnknown:
    @pytest.mark.asyncio
    async def test_unknown_writes_nothing(self, records, db):
        await arm(records)
        before = await state(db)

        result = await report(records, "no-such-occurrence", "rang", T0)

        assert result == "unknown"
        assert await state(db) == before


class TestDuplicates:
    @pytest.mark.asyncio
    async def test_a_resent_report_is_a_duplicate_writing_nothing(self, records, db):
        occurrence_id = await arm(records)
        await ring(records, occurrence_id)
        rang_at = T0 + timedelta(seconds=1)
        assert await report(records, occurrence_id, "rang", rang_at) == "applied"
        before = await state(db)

        result = await report(records, occurrence_id, "rang", rang_at, now=rang_at + FALLBACK)

        assert result == "duplicate"
        assert await state(db) == before

    @pytest.mark.asyncio
    async def test_another_endpoint_or_outcome_is_not_a_duplicate(self, records, db):
        occurrence_id = await arm(records)
        await ring(records, occurrence_id)
        rang_at = T0 + timedelta(seconds=1)
        assert await report(records, occurrence_id, "rang", rang_at) == "applied"

        assert await report(records, occurrence_id, "rang", rang_at, endpoint="room") == "applied"
        assert (await live(records))[occurrence_id].delivered_to == frozenset({ENDPOINT, "room"})

    @pytest.mark.asyncio
    async def test_a_re_ring_after_snooze_is_acknowledged_and_ends_timed_out(self, records, db):
        occurrence_id = await arm(records)
        await ring(records, occurrence_id)
        first = T0 + timedelta(seconds=1)
        assert await report(records, occurrence_id, "rang", first) == "applied"
        until = T0 + timedelta(minutes=9)
        await snooze(records, occurrence_id, until)
        await ring(records, occurrence_id, at=until)
        assert (await live(records))[occurrence_id].delivered_to == frozenset()
        second = until + timedelta(seconds=1)

        assert await report(records, occurrence_id, "rang", first) == "duplicate"
        result = await report(records, occurrence_id, "rang", second)

        assert result == "applied"
        assert (await live(records))[occurrence_id].delivered_to == frozenset({ENDPOINT})
        timeout = until + timedelta(minutes=5)
        assert await records.time_out(occurrence_id, at=timeout) == "timed_out"

    @pytest.mark.asyncio
    async def test_a_resent_stop_of_a_locally_snoozed_fallback_ring_is_a_duplicate(
        self, records, db
    ):
        occurrence_id = await arm(records)
        first_local_start = T0 + FALLBACK
        stopped_at = first_local_start + timedelta(minutes=18)  # after two local snoozes
        reconnected = stopped_at + timedelta(minutes=1)

        result = await report(
            records,
            occurrence_id,
            "stopped",
            stopped_at,
            rang_at=first_local_start,
            now=reconnected,
        )
        assert result == "applied"
        before = await state(db)

        resent = await report(
            records,
            occurrence_id,
            "stopped",
            stopped_at,
            rang_at=first_local_start,
            now=reconnected + timedelta(seconds=5),
        )

        assert resent == "duplicate"
        assert await state(db) == before


class TestLatenessBound:
    """A ring that started after `next_fire_at + grace` is recorded only, answered `late`."""

    @pytest.mark.asyncio
    async def test_a_late_rang_on_a_missed_occurrence_is_recorded_only(self, records, db):
        occurrence_id = await arm(records)
        await miss(records, occurrence_id, T0 + GRACE + timedelta(seconds=1))
        occurrences_before = await rows(db, "SELECT * FROM alarm_occurrences")
        rev_before = await records.rev()
        rang_at = T0 + GRACE + timedelta(seconds=1)

        result = await report(records, occurrence_id, "rang", rang_at)

        assert result == "late"
        assert await rows(db, "SELECT * FROM alarm_occurrences") == occurrences_before
        assert await records.rev() == rev_before
        assert (await kinds(db, occurrence_id))[-2:] == ["missed", "reported"]
        await assert_one_report(
            db, 0, occurrence_id, outcome="rang", at=rang_at, rang_at=rang_at, result="late"
        )

    @pytest.mark.asyncio
    async def test_a_late_stop_on_a_scheduled_due_occurrence_is_recorded_only(self, records, db):
        occurrence_id = await arm(records)
        occurrences_before = await rows(db, "SELECT * FROM alarm_occurrences")
        records_before = await rows(db, "SELECT * FROM alarm_records")
        rang_at = T0 + GRACE + timedelta(seconds=1)
        at = rang_at + timedelta(minutes=2)

        result = await report(records, occurrence_id, "stopped", at, rang_at=rang_at)

        assert result == "late"
        assert await rows(db, "SELECT * FROM alarm_occurrences") == occurrences_before
        assert await rows(db, "SELECT * FROM alarm_records") == records_before
        assert await kinds(db, occurrence_id) == ["set", "reported"]
        await assert_one_report(
            db, 0, occurrence_id, outcome="stopped", at=at, rang_at=rang_at, result="late"
        )

    @pytest.mark.asyncio
    async def test_a_rang_one_second_inside_the_bound_on_a_missed_occurrence_applies(
        self, records, db
    ):
        occurrence_id = await arm(records)
        await miss(records, occurrence_id, T0 + GRACE + timedelta(seconds=1))
        rang_at = T0 + GRACE - timedelta(seconds=1)

        result = await report(records, occurrence_id, "rang", rang_at, now=T0 + GRACE * 2)

        assert result == "applied"
        assert await live(records) == {}
        assert (await kinds(db, occurrence_id))[-2:] == ["timed_out", "reported"]

    @pytest.mark.asyncio
    async def test_a_stop_one_second_inside_the_bound_on_a_scheduled_occurrence_applies(
        self, records, db
    ):
        occurrence_id = await arm(records)
        rang_at = T0 + GRACE - timedelta(seconds=1)

        result = await report(
            records, occurrence_id, "stopped", rang_at + timedelta(minutes=2), rang_at=rang_at
        )

        assert result == "applied"
        assert await live(records) == {}
        assert await kinds(db, occurrence_id) == ["set", "fired", "stopped", "reported"]

    @pytest.mark.asyncio
    async def test_a_timely_ring_stopped_far_later_after_local_snoozes_applies(self, records, db):
        occurrence_id = await arm(records)
        await miss(records, occurrence_id, T0 + GRACE + timedelta(seconds=1))
        rang_at = T0 + FALLBACK
        at = T0 + timedelta(hours=3)

        result = await report(records, occurrence_id, "stopped", at, rang_at=rang_at)

        assert result == "applied"
        assert await live(records) == {}
        [record] = await records.saved()
        assert record.last_outcome == "stopped"
        await assert_one_report(
            db, 0, occurrence_id, outcome="stopped", at=at, rang_at=rang_at, result="applied"
        )

    @pytest.mark.asyncio
    async def test_the_bound_does_not_apply_to_a_ringing_occurrence(self, records):
        occurrence_id = await arm(records)
        late_start = T0 + GRACE + timedelta(minutes=1)
        await ring(records, occurrence_id, at=late_start)

        assert await report(records, occurrence_id, "rang", late_start) == "applied"


class TestOneReportedEvent:
    @pytest.mark.asyncio
    async def test_every_non_duplicate_report_appends_exactly_one_reported_event(self, records, db):
        """A run through every answer that writes: each adds one event, a resend adds none."""
        ringing_id = await arm(records)
        await ring(records, ringing_id)
        late_id = await arm(records, alarm(T0 + timedelta(minutes=1)))
        steps: list[tuple[str, str, datetime, datetime, str]] = [
            (ringing_id, "rang", T0 + timedelta(seconds=1), T0 + timedelta(seconds=1), "applied"),
            (ringing_id, "rang", T0 + timedelta(seconds=1), T0 + timedelta(seconds=1), "duplicate"),
            (ringing_id, "stopped", T0, T0 + timedelta(seconds=30), "applied"),
            (ringing_id, "stopped", T0, T0 + timedelta(seconds=30), "duplicate"),
            (
                ringing_id,
                "rang",
                T0 + timedelta(seconds=40),
                T0 + timedelta(seconds=40),
                "already_ended",
            ),
            (late_id, "rang", T0 + timedelta(hours=1), T0 + timedelta(hours=1), "late"),
        ]

        for occurrence_id, outcome, rang_at, at, expected in steps:
            before = await reported(db)
            result = await report(records, occurrence_id, outcome, at, rang_at=rang_at)
            assert result == expected
            if expected == "duplicate":
                assert await reported(db) == before
            else:
                await assert_one_report(
                    db,
                    len(before),
                    occurrence_id,
                    outcome=outcome,
                    at=at,
                    rang_at=rang_at,
                    result=expected,
                )
