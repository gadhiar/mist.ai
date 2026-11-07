"""Tests for `AlarmRecords`: identity, arming by category, firing, endings and the event trail."""

from __future__ import annotations

import itertools
import json
import logging
import sqlite3
import uuid
import zoneinfo
from collections.abc import Callable
from datetime import UTC, datetime, time, timedelta, timezone, tzinfo
from typing import TypeVar

import pytest

from backend.capabilities.alarms.records import (
    AlarmRecords,
    ArmRequest,
    identity_for,
    normalise_label,
)
from backend.capabilities.db import CapabilitiesDb
from backend.errors import CapabilityStoreError

T0 = datetime(2026, 1, 5, 7, 0, tzinfo=UTC)
"""Monday 5 January 2026, 07:00 UTC."""
TUE_0700 = T0 + timedelta(days=1)
SAT_0700 = T0 + timedelta(days=5)
RECORDS_LOGGER = "backend.capabilities.alarms.records"

R = TypeVar("R")


def new_york() -> tzinfo:
    try:
        return zoneinfo.ZoneInfo("America/New_York")
    except zoneinfo.ZoneInfoNotFoundError:
        pytest.skip("tzdata absent until H1")


def alarm(
    fire_at: datetime | None,
    hh: int = 7,
    mm: int = 0,
    *,
    label: str | None = None,
    days: frozenset[int] | None = None,
    ring_on: object = None,
) -> ArmRequest:
    return ArmRequest(
        kind="alarm",
        fire_at=fire_at,
        wall_time=time(hh, mm),
        duration_s=None,
        label=label,
        days=days,
        ring_on=ring_on,  # type: ignore[arg-type]
    )


def timer(
    fire_at: datetime | None, duration_s: int = 600, *, label: str | None = None
) -> ArmRequest:
    return ArmRequest(
        kind="timer",
        fire_at=fire_at,
        wall_time=None,
        duration_s=duration_s,
        label=label,
        days=None,
    )


def tag(kind: str, label: str, ring_on: object = None) -> ArmRequest:
    """The label alone: re-arm a tagged record as saved."""
    return ArmRequest(
        kind=kind,  # type: ignore[arg-type]
        fire_at=None,
        wall_time=None,
        duration_s=None,
        label=label,
        days=None,
        ring_on=ring_on,  # type: ignore[arg-type]
    )


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


async def arm(records: AlarmRecords, request: ArmRequest, at: datetime = T0, tz: tzinfo = UTC):
    return await records.arm(request, at=at, tz=tz, source="fastpath")


async def rows(db: CapabilitiesDb, sql: str, params: tuple[object, ...] = ()) -> list[tuple]:
    return await db.run(lambda conn: [tuple(r) for r in conn.execute(sql, params).fetchall()])


async def count(db: CapabilitiesDb, table: str) -> int:
    return (await rows(db, f"SELECT COUNT(*) FROM {table}"))[0][0]  # nosec B608


async def events(db: CapabilitiesDb) -> list[tuple]:
    return await rows(
        db,
        "SELECT kind, occurrence_id, source, endpoint, detail FROM alarm_events ORDER BY seq",
    )


async def state(db: CapabilitiesDb) -> tuple:
    """Everything stored, so a test can tell that a failed write changed nothing."""
    return (
        await rows(db, "SELECT * FROM alarm_records ORDER BY id"),
        await rows(db, "SELECT * FROM alarm_occurrences ORDER BY id"),
        await rows(db, "SELECT * FROM alarm_events ORDER BY seq"),
        await rows(db, "SELECT rev FROM alarm_records_rev"),
    )


async def ring(records: AlarmRecords, occurrence_id: str, at: datetime = T0, **kwargs) -> bool:
    kwargs.setdefault("next_fire_at", None)
    return await records.fire(occurrence_id, outcome="ringing", at=at, late_by_s=0.0, **kwargs)


async def live(records: AlarmRecords) -> list:
    return (await records.snapshot())[1]


async def record_of(records: AlarmRecords, identity: str):
    [record] = [r for r in await records.saved() if r.identity == identity]
    return record


class _FailingConnection:
    """A connection that counts each run's statements and fails the one numbered `fail_at`."""

    def __init__(self, conn: sqlite3.Connection, owner: FailingDb) -> None:
        self._conn = conn
        self._owner = owner

    @property
    def in_transaction(self) -> bool:
        return self._conn.in_transaction

    @property
    def text_factory(self) -> object:
        return self._conn.text_factory

    @text_factory.setter
    def text_factory(self, factory: object) -> None:
        self._conn.text_factory = factory  # type: ignore[assignment]

    def execute(self, sql: str, params: tuple[object, ...] = ()) -> sqlite3.Cursor:
        self._owner.statements.append(sql)
        if self._owner.fail_at == len(self._owner.statements):
            raise sqlite3.OperationalError("disk I/O error")
        return self._conn.execute(sql, params)


class FailingDb(CapabilitiesDb):
    """A `CapabilitiesDb` whose next `run` fails at statement `fail_at`, counted from 1."""

    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.fail_at: int | None = None
        self.statements: list[str] = []

    async def run(self, fn: Callable[[sqlite3.Connection], R]) -> R:
        self.statements = []
        return await super().run(lambda conn: fn(_FailingConnection(conn, self)))  # type: ignore[arg-type]


class InterleavingDb(CapabilitiesDb):
    """A `CapabilitiesDb` that, while `armed`, lands one arm behind the store before every `run`.

    Each one inserts a record and its occurrence and bumps `rev` once, so a store that holds only
    such arms always has `rev` equal to the number of live occurrences.
    """

    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.armed = False

    async def run(self, fn: Callable[[sqlite3.Connection], R]) -> R:
        if self.armed:
            await super().run(_arm_behind_the_store)
        return await super().run(fn)


def _arm_behind_the_store(conn: sqlite3.Connection) -> None:
    record_id, occurrence_id = uuid.uuid4().hex, uuid.uuid4().hex
    duration_s = next(_DURATIONS)
    when = (T0 + timedelta(days=1)).isoformat(timespec="microseconds")
    conn.execute("BEGIN IMMEDIATE")
    _insert_record_row(
        conn, record_id, identity=f"timer:one_off:{duration_s}", duration_s=duration_s
    )
    _insert_occurrence_row(conn, occurrence_id, record_id, scheduled_for=when, next_fire_at=when)
    conn.execute("UPDATE alarm_records_rev SET rev = rev + 1 WHERE id = 1")
    conn.execute("COMMIT")


_DURATIONS = itertools.count(1000)
"""Durations for raw timer records, above any a test arms through the store."""

_RECORD_DEFAULTS: dict[str, object] = {
    "kind": "timer",
    "category": "one_off",
    "identity": "timer:one_off:60",
    "label": None,
    "wall_time": None,
    "days": None,
    "duration_s": 60,
    "ring_on": "all",
    "use_count": 1,
    "fire_count": 0,
    "created_at": T0.isoformat(timespec="microseconds"),
    "last_set_at": T0.isoformat(timespec="microseconds"),
    "last_fired_at": None,
    "last_outcome": None,
}
_OCCURRENCE_DEFAULTS: dict[str, object] = {
    "state": "scheduled",
    "ring_on": "all",
    "scheduled_for": T0.isoformat(timespec="microseconds"),
    "next_fire_at": T0.isoformat(timespec="microseconds"),
    "fired_at": None,
    "late_by_s": None,
    "missed_at": None,
    "missed_reason": None,
    "sent_to": "",
    "delivered_to": "",
}


def _insert_record_row(conn: sqlite3.Connection, record_id: str, **overrides: object) -> None:
    values = {**_RECORD_DEFAULTS, **overrides}
    names = ", ".join(["id", *values])
    marks = ", ".join("?" * (len(values) + 1))
    conn.execute(
        f"INSERT INTO alarm_records ({names}) VALUES ({marks})",  # nosec B608
        (record_id, *values.values()),
    )


def _insert_occurrence_row(
    conn: sqlite3.Connection, occurrence_id: str, record_id: str, **overrides: object
) -> None:
    values = {**_OCCURRENCE_DEFAULTS, **overrides}
    names = ", ".join(["id", "record_id", *values])
    marks = ", ".join("?" * (len(values) + 2))
    conn.execute(
        f"INSERT INTO alarm_occurrences ({names}) VALUES ({marks})",  # nosec B608
        (occurrence_id, record_id, *values.values()),
    )


async def insert_record(db: CapabilitiesDb, **overrides: object) -> str:
    record_id = uuid.uuid4().hex
    if "identity" not in overrides:
        # A distinct duration keeps each raw record's identity unique and in agreement with it.
        duration_s = overrides.setdefault("duration_s", next(_DURATIONS))
        overrides["identity"] = f"timer:one_off:{duration_s}"
    await db.run(lambda conn: _insert_record_row(conn, record_id, **overrides))
    return record_id


async def insert_occurrence(db: CapabilitiesDb, record_id: str, **overrides: object) -> str:
    occurrence_id = uuid.uuid4().hex
    await db.run(lambda conn: _insert_occurrence_row(conn, occurrence_id, record_id, **overrides))
    return occurrence_id


# Identity and category come first: every arming rule is keyed on them.


class TestIdentity:
    @pytest.mark.parametrize(
        ("request_", "category", "identity"),
        [
            (alarm(T0), "one_off", "alarm:one_off:07:00"),
            (alarm(T0, 18, 5), "one_off", "alarm:one_off:18:05"),
            (timer(T0), "one_off", "timer:one_off:600"),
            (alarm(T0, days=frozenset({3, 0})), "recurring", "alarm:recurring:07:00:0,3"),
            (
                alarm(T0, days=frozenset(range(7))),
                "recurring",
                "alarm:recurring:07:00:0,1,2,3,4,5,6",
            ),
            (alarm(T0, label="Gym"), "tagged", "alarm:tagged:gym"),
            (alarm(T0, label="Gym", days=frozenset({0, 3})), "tagged", "alarm:tagged:gym"),
            (timer(T0, label="Pasta  Timer"), "tagged", "timer:tagged:pasta timer"),
            (tag("alarm", "gym"), "tagged", "alarm:tagged:gym"),
            (tag("timer", "gym"), "tagged", "timer:tagged:gym"),
        ],
    )
    def test_category_and_identity(self, request_, category, identity):
        assert identity_for(request_) == (category, identity)

    @pytest.mark.parametrize("label", ["my gym alarm", "My Gym Alarm", "  MY   gym\talarm  "])
    def test_labels_match_ignoring_case_and_repeated_spaces(self, label):
        assert normalise_label(label) == "my gym alarm"
        assert identity_for(tag("alarm", label)) == ("tagged", "alarm:tagged:my gym alarm")

    @pytest.mark.parametrize(
        "request_",
        [
            ArmRequest("timer", T0, None, 600, None, frozenset({0})),  # timers never repeat
            ArmRequest("alarm", T0, None, None, None, None),  # an alarm needs its time
            ArmRequest("timer", T0, None, None, None, None),  # a timer needs its duration
            ArmRequest("timer", T0, None, 0, None, None),
            ArmRequest("timer", T0, None, 86_401, None, None),
            ArmRequest("timer", T0, time(7), 600, None, None),
            ArmRequest("alarm", T0, time(7), 600, None, None),
            ArmRequest("alarm", T0, time(7, 0, 30), None, None, None),
            ArmRequest("alarm", T0, time(7), None, "   ", None),
            ArmRequest("alarm", T0, time(7), None, None, frozenset()),
            ArmRequest("alarm", T0, time(7), None, None, frozenset({7})),
            ArmRequest("alarm", None, time(7), None, None, None),  # a re-arm needs a label
            ArmRequest("alarm", None, time(7), None, "gym", None),  # a re-arm carries no values
            ArmRequest("clock", T0, time(7), None, None, None),  # type: ignore[arg-type]
            ArmRequest("alarm", datetime(2026, 1, 5, 7), time(7), None, None, None),
        ],
    )
    def test_an_incoherent_request_is_refused(self, request_):
        with pytest.raises(ValueError):
            identity_for(request_)

    @pytest.mark.asyncio
    async def test_the_tables_hold_no_enabled_flag(self, records, db):
        columns = {row[1] for row in await rows(db, "PRAGMA table_info(alarm_records)")}

        assert "enabled" not in columns
        assert columns == {
            "id",
            "kind",
            "category",
            "identity",
            "label",
            "wall_time",
            "days",
            "duration_s",
            "ring_on",
            "use_count",
            "fire_count",
            "created_at",
            "last_set_at",
            "last_fired_at",
            "last_outcome",
        }

    @pytest.mark.asyncio
    async def test_active_is_derived_from_live_occurrences(self, records):
        result = await arm(records, alarm(TUE_0700))
        assert (await record_of(records, "alarm:one_off:07:00")).active is True

        await records.cancel([result.occurrence.occurrence_id], at=T0, source="fastpath")

        assert (await record_of(records, "alarm:one_off:07:00")).active is False


class TestMigrate:
    @pytest.mark.asyncio
    async def test_a_second_migrate_is_a_no_op(self, records, db):
        await arm(records, alarm(TUE_0700))
        before = await state(db)

        records.migrate()
        AlarmRecords(db).migrate()

        assert await state(db) == before
        assert await rows(
            db, "SELECT version FROM schema_versions WHERE component = 'alarm_records'"
        ) == [(1,)]

    def test_the_constructor_touches_no_file(self, tmp_path):
        AlarmRecords(CapabilitiesDb(str(tmp_path / "absent" / "capabilities.db")))

        assert not (tmp_path / "absent").exists()

    def test_migrate_failure_names_the_store(self, tmp_path):
        parent = tmp_path / "not_a_directory"
        parent.write_text("", encoding="utf-8")
        store = AlarmRecords(CapabilitiesDb(str(parent / "capabilities.db")))

        with pytest.raises(CapabilityStoreError, match="alarm records"):
            store.migrate()

    @pytest.mark.parametrize("max_events", [0, -1, 1.5, True])
    def test_max_events_must_be_a_positive_integer(self, db, max_events):
        with pytest.raises(ValueError):
            AlarmRecords(db, max_events=max_events)


class TestArmOneOff:
    @pytest.mark.asyncio
    async def test_a_new_identity_inserts_a_record_with_its_first_occurrence(self, records, db):
        result = await arm(records, alarm(TUE_0700))

        assert result.created_record is True
        assert result.already_set is False
        assert result.moved == ()
        occurrence = result.occurrence
        assert occurrence.state == "scheduled"
        assert occurrence.scheduled_for == occurrence.next_fire_at == TUE_0700
        assert occurrence.ring_on == "all"
        assert occurrence.sent_to == occurrence.delivered_to == frozenset()
        record = occurrence.record
        assert (record.kind, record.category, record.identity) == (
            "alarm",
            "one_off",
            "alarm:one_off:07:00",
        )
        assert (record.wall_time, record.days, record.duration_s) == (time(7), None, None)
        assert (record.use_count, record.fire_count) == (1, 0)
        assert record.created_at == record.last_set_at == T0
        assert record.last_fired_at is None and record.last_outcome is None
        assert record.active is True
        assert [e[0] for e in await events(db)] == ["set"]
        assert await records.rev() == 1

    @pytest.mark.asyncio
    async def test_the_same_time_on_two_days_is_one_record_with_two_occurrences(self, records):
        first = await arm(records, alarm(TUE_0700, ring_on=("desktop",)))
        second = await arm(records, alarm(SAT_0700), at=T0 + timedelta(minutes=1))

        assert second.created_record is False and second.already_set is False
        record = await record_of(records, "alarm:one_off:07:00")
        assert record.use_count == 2
        assert record.last_set_at == T0 + timedelta(minutes=1)
        occurrences = {o.occurrence_id: o for o in await live(records)}
        assert set(occurrences) == {
            first.occurrence.occurrence_id,
            second.occurrence.occurrence_id,
        }
        assert occurrences[first.occurrence.occurrence_id].ring_on == ("desktop",)
        assert occurrences[first.occurrence.occurrence_id].scheduled_for == TUE_0700
        # The second arm took the record's last ring_on, and did not retarget the first.
        assert occurrences[second.occurrence.occurrence_id].ring_on == ("desktop",)
        assert occurrences[second.occurrence.occurrence_id].scheduled_for == SAT_0700

    @pytest.mark.asyncio
    async def test_each_occurrence_keeps_its_own_ring_on(self, records):
        first = await arm(records, alarm(TUE_0700, ring_on=("desktop",)))
        second = await arm(records, alarm(SAT_0700, ring_on=("room", "desktop")))

        occurrences = {o.occurrence_id: o for o in await live(records)}
        assert occurrences[first.occurrence.occurrence_id].ring_on == ("desktop",)
        assert occurrences[second.occurrence.occurrence_id].ring_on == ("desktop", "room")
        assert (await record_of(records, "alarm:one_off:07:00")).ring_on == ("desktop", "room")

    @pytest.mark.asyncio
    async def test_the_same_instant_again_is_already_set_and_only_retargets(self, records, db):
        first = await arm(records, alarm(TUE_0700))
        other = await arm(records, alarm(SAT_0700))
        rev = await records.rev()

        again = await arm(records, alarm(TUE_0700, ring_on=("desktop",)))

        assert again.already_set is True and again.created_record is False
        assert again.occurrence.occurrence_id == first.occurrence.occurrence_id
        assert again.occurrence.ring_on == ("desktop",)
        assert (await record_of(records, "alarm:one_off:07:00")).use_count == 2
        occurrences = {o.occurrence_id: o for o in await live(records)}
        assert len(occurrences) == 2
        assert occurrences[other.occurrence.occurrence_id].ring_on == "all"
        assert await records.rev() == rev + 1
        assert [e[0] for e in await events(db)] == ["set", "set"]

    @pytest.mark.asyncio
    async def test_already_set_with_no_target_writes_nothing(self, records, db):
        await arm(records, alarm(TUE_0700, ring_on=("desktop",)))
        before = await state(db)

        again = await arm(records, alarm(TUE_0700))

        assert again.already_set is True
        assert again.occurrence.ring_on == ("desktop",)
        assert await state(db) == before

    @pytest.mark.asyncio
    async def test_a_timer_started_twice_is_one_record_with_two_occurrences(self, records):
        first = await arm(records, timer(T0 + timedelta(seconds=600)))
        later = T0 + timedelta(minutes=3)
        second = await arm(records, timer(later + timedelta(seconds=600)), at=later)

        assert first.created_record is True and second.created_record is False
        assert second.already_set is False
        record = await record_of(records, "timer:one_off:600")
        assert (record.kind, record.duration_s, record.use_count) == ("timer", 600, 2)
        assert len(await live(records)) == 2

    @pytest.mark.asyncio
    async def test_a_ringing_or_missed_occurrence_does_not_count_as_set(self, records):
        first = await arm(records, alarm(T0))
        await ring(records, first.occurrence.occurrence_id)

        again = await arm(records, alarm(T0))

        assert again.already_set is False
        assert (await record_of(records, "alarm:one_off:07:00")).use_count == 2
        assert sorted(o.state for o in await live(records)) == ["ringing", "scheduled"]

    @pytest.mark.asyncio
    async def test_a_missed_occurrence_stays_beside_a_new_one(self, records):
        first = await arm(records, alarm(T0))
        await records.fire(
            first.occurrence.occurrence_id,
            outcome="missed",
            at=T0 + timedelta(hours=1),
            late_by_s=3600.0,
            next_fire_at=None,
        )

        await arm(records, alarm(TUE_0700))

        assert sorted(o.state for o in await live(records)) == ["missed", "scheduled"]


class TestArmRecurring:
    @pytest.mark.asyncio
    async def test_a_recurring_alarm_is_separate_from_a_one_off_at_the_same_time(self, records):
        await arm(records, alarm(TUE_0700))
        result = await arm(records, alarm(TUE_0700, days=frozenset({1, 3})))

        assert result.created_record is True
        identities = sorted(r.identity for r in await records.saved())
        assert identities == ["alarm:one_off:07:00", "alarm:recurring:07:00:1,3"]
        assert result.occurrence.record.days == frozenset({1, 3})

    @pytest.mark.asyncio
    async def test_set_again_while_scheduled_is_already_set(self, records):
        first = await arm(records, alarm(TUE_0700, days=frozenset({1, 3})))

        again = await arm(records, alarm(TUE_0700, days=frozenset({3, 1}), ring_on=("desktop",)))

        assert again.already_set is True
        assert again.occurrence.occurrence_id == first.occurrence.occurrence_id
        assert again.occurrence.ring_on == ("desktop",)
        assert again.occurrence.record.use_count == 1
        assert len(await live(records)) == 1

    @pytest.mark.asyncio
    async def test_set_again_after_a_cancel_arms_one_and_counts_it(self, records, db):
        first = await arm(records, alarm(TUE_0700, days=frozenset({1, 3})))
        await records.cancel([first.occurrence.occurrence_id], at=T0, source="fastpath")

        again = await arm(records, alarm(TUE_0700, days=frozenset({1, 3})))

        assert again.already_set is False and again.created_record is False
        assert again.occurrence.record.use_count == 2
        assert [o.occurrence_id for o in await live(records)] == [again.occurrence.occurrence_id]
        assert await count(db, "alarm_records") == 1


class TestArmTagged:
    @pytest.mark.asyncio
    async def test_a_tagged_alarm_is_separate_from_one_off_and_recurring(self, records):
        await arm(records, alarm(TUE_0700))
        await arm(records, alarm(TUE_0700, days=frozenset({1})))
        result = await arm(records, alarm(TUE_0700, label="Gym", days=frozenset({1})))

        assert result.created_record is True
        assert result.occurrence.record.category == "tagged"
        assert result.occurrence.record.label == "Gym"
        assert len(await records.saved()) == 3

    @pytest.mark.asyncio
    async def test_the_label_alone_arms_the_next_instant_of_its_saved_time_and_days(self, records):
        thursday_0600 = datetime(2026, 1, 8, 6, 0, tzinfo=UTC)
        first = await arm(records, alarm(thursday_0600, 6, 0, label="gym", days=frozenset({0, 3})))
        await records.cancel([first.occurrence.occurrence_id], at=T0, source="fastpath")
        # Monday 07:00 is past Monday's 06:00, so the next instant is Thursday's.

        result = await arm(records, tag("alarm", "GYM"))

        assert result.already_set is False and result.created_record is False
        assert result.occurrence.scheduled_for == thursday_0600
        assert result.occurrence.record.use_count == 2

    @pytest.mark.asyncio
    async def test_the_label_alone_resolves_in_the_given_zone(self, records):
        zone = new_york()
        # Monday 07:00 UTC is 02:00 in New York, before that day's 06:00.
        first = await arm(
            records,
            alarm(
                datetime(2026, 1, 8, 11, 0, tzinfo=UTC), 6, 0, label="gym", days=frozenset({0, 3})
            ),
            tz=zone,
        )
        await records.cancel([first.occurrence.occurrence_id], at=T0, source="fastpath")

        result = await arm(records, tag("alarm", "gym"), tz=zone)

        assert result.occurrence.scheduled_for == datetime(2026, 1, 5, 11, 0, tzinfo=UTC)

    @pytest.mark.asyncio
    async def test_the_label_alone_arms_a_saved_one_off_wall_time(self, records):
        first = await arm(records, alarm(T0 + timedelta(hours=2), 9, 0, label="nap"))
        await records.cancel([first.occurrence.occurrence_id], at=T0, source="fastpath")

        result = await arm(records, tag("alarm", "nap"), at=T0 + timedelta(hours=3))

        assert result.occurrence.scheduled_for == datetime(2026, 1, 6, 9, 0, tzinfo=UTC)

    @pytest.mark.asyncio
    async def test_the_label_alone_runs_a_saved_timer_for_its_duration(self, records):
        first = await arm(records, timer(T0 + timedelta(seconds=480), 480, label="pasta"))
        await records.cancel([first.occurrence.occurrence_id], at=T0, source="fastpath")
        later = T0 + timedelta(hours=5)

        result = await arm(records, tag("timer", "Pasta"), at=later)

        assert result.occurrence.scheduled_for == later + timedelta(seconds=480)
        assert result.occurrence.record.duration_s == 480

    @pytest.mark.asyncio
    async def test_the_label_alone_is_already_set_while_scheduled(self, records):
        first = await arm(records, alarm(TUE_0700, label="gym"))

        again = await arm(records, tag("alarm", "gym"))

        assert again.already_set is True
        assert again.occurrence.occurrence_id == first.occurrence.occurrence_id
        assert again.occurrence.record.use_count == 1

    @pytest.mark.asyncio
    async def test_with_one_ringing_the_label_alone_arms_the_next_instant(self, records):
        first = await arm(records, alarm(T0, label="gym"))
        await ring(records, first.occurrence.occurrence_id)

        again = await arm(records, tag("alarm", "gym"), at=T0 + timedelta(seconds=5))

        assert again.already_set is False
        assert again.occurrence.scheduled_for == TUE_0700
        assert again.occurrence.record.use_count == 2
        assert sorted(o.state for o in await live(records)) == ["ringing", "scheduled"]

    @pytest.mark.asyncio
    async def test_the_label_with_the_saved_values_and_instant_is_already_set(self, records):
        first = await arm(records, alarm(TUE_0700, label="gym"))

        again = await arm(records, alarm(TUE_0700, label="Gym"))

        assert again.already_set is True and again.moved == ()
        assert again.occurrence.occurrence_id == first.occurrence.occurrence_id

    @pytest.mark.asyncio
    async def test_the_label_with_a_new_time_moves_it(self, records, db):
        first = await arm(records, alarm(TUE_0700, label="gym"))
        old_id = first.occurrence.occurrence_id

        moved = await arm(records, alarm(TUE_0700 + timedelta(minutes=30), 7, 30, label="gym"))

        assert moved.moved == (old_id,)
        assert moved.already_set is False and moved.created_record is False
        record = moved.occurrence.record
        assert record.wall_time == time(7, 30)
        assert record.use_count == 2
        assert record.last_outcome == "moved"
        assert [o.occurrence_id for o in await live(records)] == [moved.occurrence.occurrence_id]
        assert [(e[0], e[1]) for e in await events(db)] == [
            ("set", old_id),
            ("moved", old_id),
            ("set", moved.occurrence.occurrence_id),
        ]

    @pytest.mark.asyncio
    async def test_the_label_with_new_days_moves_it(self, records):
        first = await arm(records, alarm(TUE_0700, label="gym", days=frozenset({1})))

        moved = await arm(records, alarm(TUE_0700, label="gym", days=frozenset({1, 4})))

        assert moved.moved == (first.occurrence.occurrence_id,)
        assert moved.occurrence.record.days == frozenset({1, 4})

    @pytest.mark.asyncio
    async def test_a_move_is_one_transaction(self, tmp_path):
        db = FailingDb(str(tmp_path / "capabilities.db"))
        try:
            records = AlarmRecords(db)
            records.migrate()
            first = await arm(records, alarm(TUE_0700, label="gym"))
            before = await state(db)
            request = alarm(TUE_0700 + timedelta(minutes=30), 7, 30, label="gym")

            fail_at = 2
            while True:
                db.fail_at = fail_at
                try:
                    result = await arm(records, request)
                except CapabilityStoreError:
                    db.fail_at = None
                    assert await state(db) == before, f"statement {fail_at} left a write"
                    fail_at += 1
                    continue
                break
            db.fail_at = None

            assert fail_at > 4
            assert result.moved == (first.occurrence.occurrence_id,)
            assert [o.occurrence_id for o in await live(records)] == [
                result.occurrence.occurrence_id
            ]
        finally:
            db.close()

    @pytest.mark.asyncio
    async def test_an_unknown_tag_raises_lookup_error_and_writes_nothing(self, records, db):
        await arm(records, alarm(TUE_0700, label="gym"))
        before = await state(db)

        with pytest.raises(LookupError):
            await arm(records, tag("alarm", "swim"))
        with pytest.raises(LookupError):
            await arm(records, tag("timer", "gym"))

        assert await state(db) == before

    @pytest.mark.asyncio
    async def test_find_tag_ignores_case_and_spacing(self, records):
        await arm(records, alarm(TUE_0700, label="Morning  Gym"))

        found = await records.find_tag("alarm", "morning gym")

        assert found is not None and found.label == "Morning  Gym"
        assert await records.find_tag("alarm", "  MORNING GYM ") == found
        assert await records.find_tag("timer", "morning gym") is None
        assert await records.find_tag("alarm", "evening gym") is None

    @pytest.mark.asyncio
    async def test_tag_labels_are_synchronous_and_include_one_armed_a_moment_before(
        self, records, db
    ):
        assert records.tag_labels("alarm") == frozenset()

        await arm(records, alarm(TUE_0700, label="Morning  Gym"))
        await arm(records, timer(T0 + timedelta(seconds=60), 60, label="Tea"))
        await arm(records, alarm(TUE_0700))

        assert records.tag_labels("alarm") == frozenset({"morning gym"})
        assert records.tag_labels("timer") == frozenset({"tea"})
        reopened = AlarmRecords(db)
        assert reopened.tag_labels("alarm") == frozenset()
        reopened.migrate()
        assert reopened.tag_labels("alarm") == frozenset({"morning gym"})

    @pytest.mark.asyncio
    async def test_a_ring_target_must_name_endpoints(self, records):
        for target in ((), ("",), ("all",), ("a,b",), "desktop"):
            with pytest.raises(ValueError):
                await arm(records, alarm(TUE_0700, ring_on=target))


class TestFire:
    @pytest.mark.asyncio
    async def test_scheduled_to_ringing_counts_the_fire(self, records, db):
        first = await arm(records, alarm(T0))
        at = T0 + timedelta(seconds=4)

        assert (
            await records.fire(
                first.occurrence.occurrence_id,
                outcome="ringing",
                at=at,
                late_by_s=4.0,
                next_fire_at=None,
            )
            is True
        )

        [occurrence] = await live(records)
        assert occurrence.state == "ringing"
        assert (occurrence.fired_at, occurrence.late_by_s) == (at, 4.0)
        assert occurrence.record.fire_count == 1
        assert occurrence.record.last_fired_at == at
        assert (await events(db))[-1][:3] == ("fired", first.occurrence.occurrence_id, "scheduler")
        assert await records.rev() == 2

    @pytest.mark.asyncio
    async def test_scheduled_to_missed_late(self, records, db):
        first = await arm(records, alarm(T0))
        at = T0 + timedelta(hours=2)

        assert (
            await records.fire(
                first.occurrence.occurrence_id,
                outcome="missed",
                at=at,
                late_by_s=7200.0,
                next_fire_at=None,
            )
            is True
        )

        [occurrence] = await live(records)
        assert (occurrence.state, occurrence.missed_reason, occurrence.missed_at) == (
            "missed",
            "late",
            at,
        )
        assert occurrence.record.fire_count == 1
        assert occurrence.record.last_outcome == "missed"
        kind, _, _, _, detail = (await events(db))[-1]
        assert kind == "missed" and json.loads(detail)["reason"] == "late"

    @pytest.mark.asyncio
    async def test_a_repeating_record_gains_its_next_occurrence(self, records):
        first = await arm(records, alarm(T0, days=frozenset(range(7))))

        await ring(records, first.occurrence.occurrence_id, next_fire_at=TUE_0700)

        occurrences = sorted(await live(records), key=lambda o: o.next_fire_at)
        assert [(o.state, o.scheduled_for) for o in occurrences] == [
            ("ringing", T0),
            ("scheduled", TUE_0700),
        ]
        assert occurrences[1].record.use_count == 1
        assert occurrences[1].ring_on == first.occurrence.ring_on

    @pytest.mark.asyncio
    async def test_a_next_occurrence_for_a_one_off_is_refused(self, records, db):
        first = await arm(records, alarm(T0))
        before = await state(db)

        with pytest.raises(ValueError):
            await ring(records, first.occurrence.occurrence_id, next_fire_at=TUE_0700)

        assert await state(db) == before

    @pytest.mark.asyncio
    async def test_from_snoozed_it_rings_again_without_a_count(self, records):
        first = await arm(records, alarm(T0, days=frozenset(range(7))))
        occurrence_id = first.occurrence.occurrence_id
        await ring(records, occurrence_id, next_fire_at=TUE_0700)
        await records.sent(occurrence_id, ["desktop"], fallback=False, at=T0)
        await records.acknowledge_ring(occurrence_id, "desktop", at=T0)
        until = T0 + timedelta(minutes=9)
        await records.snooze(
            occurrence_id, until=until, at=T0, source="alarm_action", endpoint="desktop"
        )

        assert await ring(records, occurrence_id, at=until, next_fire_at=TUE_0700) is True

        occurrences = {o.occurrence_id: o for o in await live(records)}
        assert len(occurrences) == 2
        rung = occurrences[occurrence_id]
        assert (rung.state, rung.fired_at) == ("ringing", until)
        assert rung.sent_to == rung.delivered_to == frozenset()
        assert rung.record.fire_count == 1

    @pytest.mark.asyncio
    async def test_it_returns_false_and_writes_nothing_when_no_longer_due(self, records, db):
        first = await arm(records, alarm(T0))
        await ring(records, first.occurrence.occurrence_id)
        before = await state(db)

        assert await ring(records, first.occurrence.occurrence_id) is False
        assert await ring(records, "no-such-occurrence") is False

        assert await state(db) == before

    @pytest.mark.parametrize("late_by_s", [-1.0, float("nan"), float("inf")])
    @pytest.mark.asyncio
    async def test_a_bad_lateness_is_refused(self, records, late_by_s):
        first = await arm(records, alarm(T0))

        with pytest.raises(ValueError):
            await records.fire(
                first.occurrence.occurrence_id,
                outcome="ringing",
                at=T0,
                late_by_s=late_by_s,
                next_fire_at=None,
            )

    @pytest.mark.asyncio
    async def test_a_failure_after_the_first_statement_leaves_nothing_written(self, tmp_path):
        db = FailingDb(str(tmp_path / "capabilities.db"))
        try:
            records = AlarmRecords(db)
            records.migrate()
            first = await arm(records, alarm(T0, days=frozenset(range(7))))
            occurrence_id = first.occurrence.occurrence_id
            before = await state(db)

            fail_at = 2
            while True:
                db.fail_at = fail_at
                try:
                    fired = await ring(records, occurrence_id, next_fire_at=TUE_0700)
                except CapabilityStoreError as exc:
                    assert "alarm records" in str(exc)
                    db.fail_at = None
                    assert await state(db) == before, f"statement {fail_at} left a write"
                    fail_at += 1
                    continue
                break
            db.fail_at = None

            # The transition, the count, the next occurrence, the event and rev each failed once.
            assert fired is True
            assert fail_at > 5
            occurrences = sorted(await live(records), key=lambda o: o.next_fire_at)
            assert [o.state for o in occurrences] == ["ringing", "scheduled"]
            assert occurrences[0].record.fire_count == 1
        finally:
            db.close()


class TestDelivery:
    @pytest.mark.asyncio
    async def test_sent_stores_the_sinks(self, records, db):
        first = await arm(records, alarm(T0))
        occurrence_id = first.occurrence.occurrence_id
        await ring(records, occurrence_id)

        assert await records.sent(occurrence_id, ["room", "desktop"], fallback=False, at=T0) is True

        [occurrence] = await live(records)
        assert occurrence.sent_to == frozenset({"desktop", "room"})
        assert "fallback" not in [e[0] for e in await events(db)]

    @pytest.mark.asyncio
    async def test_sent_records_a_fallback_when_asked(self, records, db):
        first = await arm(records, alarm(T0, ring_on=("room",)))
        await ring(records, first.occurrence.occurrence_id)

        await records.sent(first.occurrence.occurrence_id, ["desktop"], fallback=True, at=T0)

        kind, occurrence_id, _, _, detail = (await events(db))[-1]
        assert (kind, occurrence_id) == ("fallback", first.occurrence.occurrence_id)
        assert json.loads(detail)["sent_to"] == ["desktop"]

    @pytest.mark.asyncio
    async def test_sent_needs_a_ringing_occurrence(self, records, db):
        first = await arm(records, alarm(T0))
        before = await state(db)

        assert (
            await records.sent(first.occurrence.occurrence_id, ["desktop"], fallback=True, at=T0)
            is False
        )
        assert await state(db) == before

    @pytest.mark.asyncio
    async def test_acknowledge_ring_adds_the_endpoint_to_a_ringing_occurrence(self, records):
        first = await arm(records, alarm(T0))
        occurrence_id = first.occurrence.occurrence_id
        await ring(records, occurrence_id)

        assert await records.acknowledge_ring(occurrence_id, "desktop", at=T0) is True
        assert await records.acknowledge_ring(occurrence_id, "room", at=T0) is True

        [occurrence] = await live(records)
        assert occurrence.delivered_to == frozenset({"desktop", "room"})

    @pytest.mark.asyncio
    async def test_acknowledge_ring_ignores_an_occurrence_not_ringing(self, records):
        scheduled = await arm(records, alarm(TUE_0700))
        snoozed = await arm(records, alarm(T0))
        await ring(records, snoozed.occurrence.occurrence_id)
        await records.snooze(
            snoozed.occurrence.occurrence_id,
            until=T0 + timedelta(minutes=9),
            at=T0,
            source="alarm_action",
            endpoint="desktop",
        )

        for occurrence_id in (scheduled.occurrence.occurrence_id, snoozed.occurrence.occurrence_id):
            assert await records.acknowledge_ring(occurrence_id, "desktop", at=T0) is False
        assert all(o.delivered_to == frozenset() for o in await live(records))


class TestTimeOut:
    @pytest.mark.asyncio
    async def test_a_heard_ring_times_out(self, records, db):
        first = await arm(records, alarm(T0))
        occurrence_id = first.occurrence.occurrence_id
        await ring(records, occurrence_id)
        await records.sent(occurrence_id, ["desktop"], fallback=False, at=T0)
        await records.acknowledge_ring(occurrence_id, "desktop", at=T0)

        assert await records.time_out(occurrence_id, at=T0 + timedelta(minutes=5)) == "timed_out"

        assert await live(records) == []
        record = await record_of(records, "alarm:one_off:07:00")
        assert (record.last_outcome, record.active) == ("timed_out", False)
        assert (await events(db))[-1][:2] == ("timed_out", occurrence_id)

    @pytest.mark.parametrize("sent_to", [[], ["desktop"]])
    @pytest.mark.asyncio
    async def test_an_unheard_ring_is_missed_with_its_sinks(self, records, db, sent_to):
        first = await arm(records, alarm(T0))
        occurrence_id = first.occurrence.occurrence_id
        await ring(records, occurrence_id)
        if sent_to:
            await records.sent(occurrence_id, sent_to, fallback=False, at=T0)
        at = T0 + timedelta(minutes=5)

        assert await records.time_out(occurrence_id, at=at) == "missed"

        [occurrence] = await live(records)
        assert (occurrence.state, occurrence.missed_reason, occurrence.missed_at) == (
            "missed",
            "unheard",
            at,
        )
        assert occurrence.record.last_outcome == "missed"
        kind, _, _, _, detail = (await events(db))[-1]
        assert kind == "missed"
        assert json.loads(detail) == {"reason": "unheard", "sent_to": sent_to}

    @pytest.mark.asyncio
    async def test_it_returns_none_when_no_longer_ringing(self, records, db):
        first = await arm(records, alarm(T0))
        before = await state(db)

        assert await records.time_out(first.occurrence.occurrence_id, at=T0) is None
        assert await records.time_out("no-such-occurrence", at=T0) is None
        assert await state(db) == before


class TestEndings:
    @pytest.mark.asyncio
    async def test_finish_deletes_the_occurrence_and_keeps_the_record(self, records, db):
        first = await arm(records, alarm(T0))
        occurrence_id = first.occurrence.occurrence_id
        await ring(records, occurrence_id)

        assert (
            await records.finish(
                occurrence_id, outcome="stopped", at=T0, source="alarm_action", endpoint="desktop"
            )
            is True
        )

        assert await live(records) == []
        record = await record_of(records, "alarm:one_off:07:00")
        assert (record.last_outcome, record.last_fired_at, record.active) == ("stopped", T0, False)
        assert (await events(db))[-1][:4] == ("stopped", occurrence_id, "alarm_action", "desktop")

    @pytest.mark.asyncio
    async def test_finish_stops_a_snoozed_occurrence(self, records):
        first = await arm(records, alarm(T0))
        occurrence_id = first.occurrence.occurrence_id
        await ring(records, occurrence_id)
        await records.snooze(
            occurrence_id, until=T0 + timedelta(minutes=9), at=T0, source="fastpath", endpoint=None
        )

        assert (
            await records.finish(
                occurrence_id, outcome="stopped", at=T0, source="fastpath", endpoint=None
            )
            is True
        )
        assert await live(records) == []

    @pytest.mark.asyncio
    async def test_finish_ignores_a_scheduled_occurrence(self, records, db):
        first = await arm(records, alarm(T0))
        before = await state(db)

        assert (
            await records.finish(
                first.occurrence.occurrence_id,
                outcome="stopped",
                at=T0,
                source="fastpath",
                endpoint=None,
            )
            is False
        )
        assert await state(db) == before

    @pytest.mark.asyncio
    async def test_snooze_moves_a_ringing_occurrence(self, records, db):
        first = await arm(records, alarm(T0))
        occurrence_id = first.occurrence.occurrence_id
        await ring(records, occurrence_id)
        until = T0 + timedelta(minutes=9)

        assert (
            await records.snooze(
                occurrence_id, until=until, at=T0, source="fastpath", endpoint=None
            )
            is True
        )

        [occurrence] = await live(records)
        assert (occurrence.state, occurrence.next_fire_at, occurrence.scheduled_for) == (
            "snoozed",
            until,
            T0,
        )
        assert (await events(db))[-1][0] == "snoozed"
        assert await records.due(until - timedelta(seconds=1)) == []
        assert [o.occurrence_id for o in await records.due(until)] == [occurrence_id]

    @pytest.mark.parametrize(("by", "outcome"), [("dismiss", "dismissed"), ("expiry", "expired")])
    @pytest.mark.asyncio
    async def test_acknowledge_ends_a_missed_occurrence(self, records, db, by, outcome):
        first = await arm(records, alarm(T0))
        occurrence_id = first.occurrence.occurrence_id
        await records.fire(occurrence_id, outcome="missed", at=T0, late_by_s=0.0, next_fire_at=None)

        assert await records.acknowledge(occurrence_id, at=T0, by=by) is True

        assert await live(records) == []
        assert (await record_of(records, "alarm:one_off:07:00")).last_outcome == outcome
        assert (await events(db))[-1][:2] == (outcome, occurrence_id)

    @pytest.mark.asyncio
    async def test_acknowledge_ignores_an_occurrence_not_missed(self, records):
        first = await arm(records, alarm(T0))

        assert (
            await records.acknowledge(first.occurrence.occurrence_id, at=T0, by="dismiss") is False
        )
        assert len(await live(records)) == 1

    @pytest.mark.asyncio
    async def test_missed_occurrences_before_a_cutoff_expire(self, records, db):
        old = await arm(records, alarm(T0))
        new = await arm(records, alarm(TUE_0700))
        await records.fire(
            old.occurrence.occurrence_id, outcome="missed", at=T0, late_by_s=0.0, next_fire_at=None
        )
        await records.fire(
            new.occurrence.occurrence_id,
            outcome="missed",
            at=TUE_0700,
            late_by_s=0.0,
            next_fire_at=None,
        )

        expired = await records.acknowledge_missed_before(T0 + timedelta(hours=1), at=TUE_0700)

        assert expired == 1
        assert [o.occurrence_id for o in await live(records)] == [new.occurrence.occurrence_id]
        assert (await events(db))[-1][:2] == ("expired", old.occurrence.occurrence_id)

    @pytest.mark.asyncio
    async def test_cancel_one_occurrence_of_a_one_off_leaves_the_others(self, records, db):
        first = await arm(records, alarm(TUE_0700))
        second = await arm(records, alarm(SAT_0700))

        cancelled = await records.cancel([first.occurrence.occurrence_id], at=T0, source="fastpath")

        assert cancelled == 1
        assert [o.occurrence_id for o in await live(records)] == [second.occurrence.occurrence_id]
        record = await record_of(records, "alarm:one_off:07:00")
        assert (record.last_outcome, record.active, record.use_count) == ("cancelled", True, 2)
        assert (await events(db))[-1][:3] == (
            "cancelled",
            first.occurrence.occurrence_id,
            "fastpath",
        )

    @pytest.mark.asyncio
    async def test_cancel_any_occurrence_of_a_repeating_record_ends_its_series(self, records, db):
        first = await arm(records, alarm(T0, days=frozenset(range(7))))
        await ring(records, first.occurrence.occurrence_id, next_fire_at=TUE_0700)
        other = await arm(records, alarm(SAT_0700))
        [scheduled] = [
            o for o in await live(records) if o.state == "scheduled" and o.record.days is not None
        ]

        cancelled = await records.cancel([scheduled.occurrence_id], at=T0, source="model")

        assert cancelled == 2
        assert [o.occurrence_id for o in await live(records)] == [other.occurrence.occurrence_id]
        assert sorted(e[1] for e in await events(db) if e[0] == "cancelled") == sorted(
            [first.occurrence.occurrence_id, scheduled.occurrence_id]
        )
        record = await record_of(records, "alarm:recurring:07:00:0,1,2,3,4,5,6")
        assert (record.last_outcome, record.active) == ("cancelled", False)

    @pytest.mark.asyncio
    async def test_cancel_of_nothing_writes_nothing(self, records, db):
        await arm(records, alarm(TUE_0700))
        before = await state(db)

        assert await records.cancel(["no-such-occurrence"], at=T0, source="fastpath") == 0
        assert await records.cancel([], at=T0, source="fastpath") == 0
        assert await state(db) == before

    @pytest.mark.asyncio
    async def test_a_cancelled_record_stays_saved_and_rearms(self, records):
        first = await arm(records, alarm(TUE_0700, label="gym"))
        await records.cancel([first.occurrence.occurrence_id], at=T0, source="fastpath")

        assert [r.label for r in await records.saved()] == ["gym"]
        again = await arm(records, tag("alarm", "gym"))

        assert again.occurrence.scheduled_for == TUE_0700
        assert again.occurrence.record.use_count == 2


class TestReads:
    @pytest.mark.asyncio
    async def test_snapshot_returns_rev_and_every_live_occurrence(self, records):
        a = await arm(records, alarm(TUE_0700))
        b = await arm(records, timer(T0 + timedelta(seconds=60), 60))
        await ring(records, b.occurrence.occurrence_id)

        rev, occurrences = await records.snapshot()

        assert rev == await records.rev() == 3
        assert [o.occurrence_id for o in occurrences] == [
            b.occurrence.occurrence_id,
            a.occurrence.occurrence_id,
        ]

    @pytest.mark.asyncio
    async def test_a_write_between_runs_cannot_split_a_snapshot(self, tmp_path):
        db = InterleavingDb(str(tmp_path / "capabilities.db"))
        try:
            records = AlarmRecords(db)
            records.migrate()
            await arm(records, timer(T0 + timedelta(seconds=60), 60))
            db.armed = True

            for _ in range(3):
                rev, occurrences = await records.snapshot()
                assert rev == len(occurrences)
            db.armed = False
            # Each snapshot came after the arm that preceded it, never between its reads.
            assert len((await records.snapshot())[1]) == 4
        finally:
            db.close()

    @pytest.mark.asyncio
    async def test_due_and_next_due(self, records):
        assert await records.next_due() is None
        a = await arm(records, alarm(TUE_0700))
        await arm(records, alarm(SAT_0700))

        assert await records.next_due() == TUE_0700
        assert await records.due(TUE_0700 - timedelta(seconds=1)) == []
        assert [o.occurrence_id for o in await records.due(TUE_0700)] == [
            a.occurrence.occurrence_id
        ]

    @pytest.mark.asyncio
    async def test_saved_lists_every_record_by_use_count_then_last_set(self, records):
        one_off = await arm(records, alarm(TUE_0700))
        await arm(records, alarm(SAT_0700), at=T0 + timedelta(minutes=1))
        gym = await arm(records, alarm(TUE_0700, label="gym"), at=T0 + timedelta(minutes=2))
        await records.cancel([gym.occurrence.occurrence_id], at=T0, source="fastpath")
        await arm(records, timer(T0 + timedelta(seconds=60), 60), at=T0 + timedelta(minutes=3))

        saved = await records.saved()

        assert [r.identity for r in saved] == [
            "alarm:one_off:07:00",
            "timer:one_off:60",
            "alarm:tagged:gym",
        ]
        assert (saved[0].use_count, saved[0].active) == (2, True)
        assert saved[2].active is False
        assert saved[0].record_id == one_off.occurrence.record.record_id
        assert [r.identity for r in await records.saved(limit=1)] == ["alarm:one_off:07:00"]
        assert len(await records.saved(limit=None)) == 3


class TestEventTrail:
    @pytest.mark.asyncio
    async def test_the_trail_never_exceeds_max_events(self, db):
        records = AlarmRecords(db, max_events=5)
        records.migrate()
        stamps: list[datetime] = []
        for minute in range(4):
            at = T0 + timedelta(minutes=minute)
            result = await arm(records, alarm(TUE_0700), at=at)
            await records.cancel([result.occurrence.occurrence_id], at=at, source="fastpath")
            stamps += [at, at]
            assert await count(db, "alarm_events") <= 5

        assert await count(db, "alarm_events") == 5
        assert await records.oldest_event_at() == stamps[-5]
        record = await record_of(records, "alarm:one_off:07:00")
        assert record.use_count == 4
        assert record.last_outcome == "cancelled"

    @pytest.mark.asyncio
    async def test_oldest_event_at_with_no_events(self, records):
        assert await records.oldest_event_at() is None

    @pytest.mark.asyncio
    async def test_event_counts_group_by_local_date_and_split_missed(self, records):
        zone = timezone(timedelta(hours=-5))
        late = await arm(records, alarm(T0))
        await records.fire(
            late.occurrence.occurrence_id, outcome="missed", at=T0, late_by_s=0.0, next_fire_at=None
        )
        # 03:00 UTC on Tuesday is still Monday at UTC-5.
        night = datetime(2026, 1, 6, 3, 0, tzinfo=UTC)
        unheard = await arm(records, timer(night, 60), at=night - timedelta(seconds=60))
        await ring(records, unheard.occurrence.occurrence_id, at=night)
        await records.time_out(unheard.occurrence.occurrence_id, at=night)
        later = datetime(2026, 1, 6, 6, 0, tzinfo=UTC)
        await arm(records, alarm(SAT_0700), at=later)

        counts = await records.event_counts(T0 - timedelta(days=1), zone)

        assert counts == {
            "2026-01-05": {"set": 2, "missed:late": 1, "fired": 1, "missed:unheard": 1},
            "2026-01-06": {"set": 1},
        }
        assert await records.event_counts(later, zone) == {"2026-01-06": {"set": 1}}


class TestRev:
    @pytest.mark.asyncio
    async def test_every_write_that_changes_a_live_occurrence_bumps_rev_once(self, records):
        first = await arm(records, alarm(T0))
        assert await records.rev() == 1
        occurrence_id = first.occurrence.occurrence_id
        await ring(records, occurrence_id)
        assert await records.rev() == 2
        await records.sent(occurrence_id, ["desktop"], fallback=True, at=T0)
        assert await records.rev() == 3
        await records.acknowledge_ring(occurrence_id, "desktop", at=T0)
        assert await records.rev() == 4
        await records.finish(
            occurrence_id, outcome="stopped", at=T0, source="fastpath", endpoint=None
        )
        assert await records.rev() == 5
        await records.finish(
            occurrence_id, outcome="stopped", at=T0, source="fastpath", endpoint=None
        )
        assert await records.rev() == 5


class TestQuarantine:
    @pytest.mark.asyncio
    async def test_a_record_outside_its_domain_is_skipped_by_every_read(self, records, db, caplog):
        good = await arm(records, alarm(TUE_0700))
        bad = await insert_record(db, category="weekly")
        bad_occurrence = await insert_occurrence(db, bad)

        with caplog.at_level(logging.ERROR, logger=RECORDS_LOGGER):
            assert [o.occurrence_id for o in await live(records)] == [good.occurrence.occurrence_id]
            assert [r.identity for r in await records.saved()] == ["alarm:one_off:07:00"]
            assert [o.occurrence_id for o in await records.due(SAT_0700)] == [
                good.occurrence.occurrence_id
            ]
            assert await records.next_due() == TUE_0700

        assert records.quarantined() == (bad,)
        assert len([r for r in caplog.records if bad in r.getMessage()]) == 1
        assert bad_occurrence not in records.quarantined()

    @pytest.mark.parametrize(
        "overrides",
        [
            {"scheduled_for": "seven tomorrow"},
            {"next_fire_at": "2026-01-05T07:00:00"},  # no offset
            {"next_fire_at": "9999-01-01T00:00:00+00:00"},
            {"state": "done"},
            {"late_by_s": -1.0},
            {"sent_to": "b,a"},
            {"state": "missed"},  # missed without its reason
            {"state": "ringing"},  # ringing without fired_at
        ],
    )
    @pytest.mark.asyncio
    async def test_an_occurrence_outside_its_domain_is_quarantined(self, records, db, overrides):
        record_id = await insert_record(db)
        bad = await insert_occurrence(db, record_id, **overrides)

        assert await live(records) == []
        assert records.quarantined() == (bad,)

    @pytest.mark.parametrize(
        "overrides",
        [
            {
                "wall_time": "7am",
                "kind": "alarm",
                "duration_s": None,
                "identity": "alarm:one_off:7am",
            },
            {"days": "monday"},
            {"duration_s": 0},
            {"use_count": -1},
            {"ring_on": ""},
            {"last_outcome": "exploded"},
            {"identity": "timer:one_off:61"},  # disagrees with its duration
        ],
    )
    @pytest.mark.asyncio
    async def test_a_record_with_a_damaged_field_is_quarantined(self, records, db, overrides):
        overrides.setdefault("identity", "timer:one_off:60")
        bad = await insert_record(db, **overrides)

        assert await records.saved() == []
        assert records.quarantined() == (bad,)

    @pytest.mark.asyncio
    async def test_a_row_leaves_the_quarantine_when_it_is_changed(self, records, db):
        record_id = await insert_record(db)
        bad = await insert_occurrence(db, record_id, state="done")
        assert await live(records) == []

        await db.run(
            lambda conn: conn.execute(
                "UPDATE alarm_occurrences SET state = 'scheduled' WHERE id = ?", (bad,)
            )
        )

        assert [o.occurrence_id for o in await live(records)] == [bad]
        assert records.quarantined() == ()

    @pytest.mark.asyncio
    async def test_a_row_leaves_the_quarantine_when_it_is_deleted(self, records, db):
        record_id = await insert_record(db, category="weekly")
        assert await records.saved() == []
        assert records.quarantined() == (record_id,)

        await db.run(
            lambda conn: conn.execute("DELETE FROM alarm_records WHERE id = ?", (record_id,))
        )

        assert await records.saved() == []
        assert records.quarantined() == ()

    @pytest.mark.asyncio
    async def test_a_row_whose_text_is_not_utf8_costs_only_itself(self, records, db):
        good = await arm(records, alarm(TUE_0700))
        record_id = await insert_record(db)
        bad = await insert_occurrence(db, record_id, ring_on=b"\xff\xfe")

        assert [o.occurrence_id for o in await live(records)] == [good.occurrence.occurrence_id]
        assert records.quarantined() == (bad,)

    @pytest.mark.asyncio
    async def test_quarantine_by_id_hides_a_row_until_it_changes(self, records, db):
        first = await arm(records, alarm(T0))
        occurrence_id = first.occurrence.occurrence_id
        await live(records)

        records.quarantine(occurrence_id, "it could not be fired (ValueError)")

        assert await live(records) == []
        assert await records.due(T0) == []
        assert records.quarantined() == (occurrence_id,)
        await ring(records, occurrence_id)
        assert [o.state for o in await live(records)] == ["ringing"]
        assert records.quarantined() == ()

    @pytest.mark.asyncio
    async def test_quarantine_of_an_unread_id_holds_the_row_read_next(self, records, db):
        first = await arm(records, alarm(T0))
        reopened = AlarmRecords(db)
        reopened.migrate()

        reopened.quarantine(first.occurrence.occurrence_id, "set aside before any read")

        assert await reopened.snapshot() == (1, [])
        assert reopened.quarantined() == (first.occurrence.occurrence_id,)

    @pytest.mark.asyncio
    async def test_cancel_deletes_a_quarantined_row(self, records, db):
        record_id = await insert_record(db)
        bad = await insert_occurrence(db, record_id, state="done")
        await live(records)

        assert await records.cancel([bad], at=T0, source="fastpath") == 1

        assert await count(db, "alarm_occurrences") == 0
        assert records.quarantined() == ()
