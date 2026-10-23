"""Tests for `AlarmStore`: lifecycle, state transitions, due queries, `rev` and corrupt rows."""

from __future__ import annotations

import logging
import sqlite3
import uuid
from datetime import UTC, datetime, timedelta, timezone

import pytest

from backend.capabilities.alarms.store import AlarmStore
from backend.capabilities.alarms.timeparse import RepeatRule
from backend.capabilities.db import CapabilitiesDb
from backend.errors import CapabilityStoreError

T0 = datetime(2026, 1, 5, 7, 0, tzinfo=UTC)
STORE_LOGGER = "backend.capabilities.alarms.store"


@pytest.fixture()
def db(tmp_path):
    database = CapabilitiesDb(str(tmp_path / "capabilities.db"))
    yield database
    database.close()


@pytest.fixture()
def store(db):
    alarm_store = AlarmStore(db)
    alarm_store.migrate()
    return alarm_store


def _blocked_path(tmp_path) -> str:
    """A database path whose parent directory is a regular file, so no open can succeed."""
    parent = tmp_path / "not_a_directory"
    parent.write_text("", encoding="utf-8")
    return str(parent / "capabilities.db")


async def _raw_row(db: CapabilitiesDb, alarm_id: str) -> sqlite3.Row | None:
    def read(conn: sqlite3.Connection):
        conn.row_factory = sqlite3.Row
        try:
            return conn.execute("SELECT * FROM alarms WHERE id = ?", (alarm_id,)).fetchone()
        finally:
            conn.row_factory = None

    return await db.run(read)


async def _insert_raw(
    db: CapabilitiesDb,
    *,
    scheduled_for: object = "2026-01-05T06:00:00.000000+00:00",
    next_fire_at: object = "2026-01-05T06:00:00.000000+00:00",
    repeat: object = None,
    state: object = "scheduled",
    kind: object = "alarm",
    late_by_s: object = None,
) -> str:
    """Insert a row behind the store's back, as a damaged or hand-edited database would hold."""
    alarm_id = f"raw-{uuid.uuid4().hex}"

    def insert(conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO alarms (id, kind, label, state, scheduled_for, next_fire_at, repeat, "
            "late_by_s, fired_at) VALUES (?, ?, NULL, ?, ?, ?, ?, ?, NULL)",
            (alarm_id, kind, state, scheduled_for, next_fire_at, repeat, late_by_s),
        )

    await db.run(insert)
    return alarm_id


class TestLifecycle:
    def test_constructor_touches_no_file(self, tmp_path):
        path = tmp_path / "sub" / "capabilities.db"

        AlarmStore(CapabilitiesDb(str(path)))

        assert not path.parent.exists()

    def test_unwritable_path_constructs_and_migrate_raises(self, tmp_path):
        alarm_store = AlarmStore(CapabilitiesDb(_blocked_path(tmp_path)))

        with pytest.raises(CapabilityStoreError, match="capabilities.db") as raised:
            alarm_store.migrate()

        assert "alarm store" in str(raised.value)

    @pytest.mark.asyncio
    async def test_unwritable_path_on_first_use_names_the_alarm_store(self, tmp_path):
        alarm_store = AlarmStore(CapabilitiesDb(_blocked_path(tmp_path)))

        with pytest.raises(CapabilityStoreError, match="alarm store"):
            await alarm_store.create("alarm", T0)

    def test_migrate_twice_is_a_no_op(self, store):
        store.migrate()

    @pytest.mark.asyncio
    async def test_use_before_migrate_raises_a_store_error_naming_the_alarm_store(self, db):
        alarm_store = AlarmStore(db)

        with pytest.raises(CapabilityStoreError, match="alarm store"):
            await alarm_store.create("alarm", T0)

    @pytest.mark.asyncio
    async def test_read_before_migrate_raises_a_store_error(self, db):
        with pytest.raises(CapabilityStoreError, match="alarm store"):
            await AlarmStore(db).list()


class TestCreateAndList:
    @pytest.mark.asyncio
    async def test_create_returns_a_scheduled_alarm(self, store):
        rule = RepeatRule("daily", 7, 0)

        alarm = await store.create("alarm", T0, label="gym", repeat=rule)

        assert alarm.kind == "alarm"
        assert alarm.label == "gym"
        assert alarm.state == "scheduled"
        assert alarm.scheduled_for == T0
        assert alarm.next_fire_at == T0
        assert alarm.repeat == rule
        assert alarm.late_by_s is None
        assert alarm.fired_at is None

    @pytest.mark.asyncio
    async def test_ids_are_unique(self, store):
        first = await store.create("alarm", T0)
        second = await store.create("alarm", T0)

        assert first.alarm_id != second.alarm_id

    @pytest.mark.asyncio
    async def test_list_returns_what_was_created(self, store):
        alarm = await store.create("timer", T0, label="tea")

        assert await store.list() == [alarm]

    @pytest.mark.asyncio
    async def test_list_orders_by_next_fire_time(self, store):
        late = await store.create("alarm", T0 + timedelta(hours=2))
        early = await store.create("alarm", T0)

        assert [a.alarm_id for a in await store.list()] == [early.alarm_id, late.alarm_id]

    @pytest.mark.asyncio
    async def test_list_filters_by_state(self, store):
        ringing = await store.create("alarm", T0)
        scheduled = await store.create("alarm", T0 + timedelta(hours=1))
        await store.mark_ringing(ringing.alarm_id, T0, 0.0)

        result = await store.list(states=("scheduled",))

        assert [a.alarm_id for a in result] == [scheduled.alarm_id]

    @pytest.mark.asyncio
    async def test_times_are_stored_as_utc_iso_strings(self, store, db):
        plus_two = timezone(timedelta(hours=2))
        alarm = await store.create("alarm", datetime(2026, 1, 5, 9, 0, tzinfo=plus_two))

        row = await _raw_row(db, alarm.alarm_id)

        assert row["scheduled_for"] == "2026-01-05T07:00:00.000000+00:00"
        assert row["next_fire_at"] == "2026-01-05T07:00:00.000000+00:00"
        assert alarm.next_fire_at.tzinfo is UTC

    @pytest.mark.asyncio
    async def test_repeat_rule_is_stored_as_json(self, store, db):
        alarm = await store.create("alarm", T0, repeat=RepeatRule("weekends", 9, 30))

        row = await _raw_row(db, alarm.alarm_id)

        assert row["repeat"] == '{"kind": "weekends", "hour": 9, "minute": 30}'

    @pytest.mark.asyncio
    async def test_no_repeat_is_stored_as_null(self, store, db):
        alarm = await store.create("alarm", T0)

        row = await _raw_row(db, alarm.alarm_id)

        assert row["repeat"] is None

    @pytest.mark.asyncio
    async def test_naive_time_is_rejected(self, store):
        with pytest.raises(ValueError, match="aware"):
            await store.create("alarm", datetime(2026, 1, 5, 7, 0))

    @pytest.mark.asyncio
    async def test_unknown_kind_is_rejected(self, store):
        with pytest.raises(ValueError, match="reminder"):
            await store.create("reminder", T0)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_alarms_survive_a_new_connection(self, store, tmp_path):
        alarm = await store.create("alarm", T0, label="gym")
        reopened_db = CapabilitiesDb(str(tmp_path / "capabilities.db"))
        reopened = AlarmStore(reopened_db)
        reopened.migrate()

        assert await reopened.list() == [alarm]
        reopened_db.close()


class TestCancel:
    @pytest.mark.asyncio
    async def test_cancel_deletes_the_alarm(self, store, db):
        alarm = await store.create("alarm", T0)

        assert await store.cancel(alarm.alarm_id) is True
        assert await store.list() == []
        assert await _raw_row(db, alarm.alarm_id) is None

    @pytest.mark.asyncio
    async def test_cancel_unknown_id_returns_false(self, store):
        assert await store.cancel("no-such-alarm") is False

    @pytest.mark.asyncio
    async def test_cancel_a_ringing_alarm(self, store):
        alarm = await store.create("alarm", T0)
        await store.mark_ringing(alarm.alarm_id, T0, 0.0)

        assert await store.cancel(alarm.alarm_id) is True
        assert await store.list() == []


class TestTransitions:
    @pytest.mark.asyncio
    async def test_scheduled_to_ringing_stores_late_by_and_fired_at(self, store):
        alarm = await store.create("alarm", T0)
        fired_at = T0 + timedelta(seconds=95)

        assert await store.mark_ringing(alarm.alarm_id, fired_at, 95.0) is True

        [ringing] = await store.list()
        assert ringing.state == "ringing"
        assert ringing.late_by_s == 95.0
        assert ringing.fired_at == fired_at

    @pytest.mark.asyncio
    async def test_ringing_to_snoozed_moves_the_next_fire_time(self, store):
        alarm = await store.create("alarm", T0)
        await store.mark_ringing(alarm.alarm_id, T0, 0.0)
        until = T0 + timedelta(minutes=9)

        assert await store.snooze(alarm.alarm_id, until) is True

        [snoozed] = await store.list()
        assert snoozed.state == "snoozed"
        assert snoozed.next_fire_at == until
        assert snoozed.scheduled_for == T0

    @pytest.mark.asyncio
    async def test_snoozed_to_ringing(self, store):
        alarm = await store.create("alarm", T0)
        await store.mark_ringing(alarm.alarm_id, T0, 0.0)
        await store.snooze(alarm.alarm_id, T0 + timedelta(minutes=9))

        assert await store.mark_ringing(alarm.alarm_id, T0 + timedelta(minutes=9), 2.0) is True
        assert (await store.list())[0].state == "ringing"

    @pytest.mark.asyncio
    async def test_ringing_to_done(self, store):
        alarm = await store.create("alarm", T0)
        await store.mark_ringing(alarm.alarm_id, T0, 0.0)

        assert await store.mark_done(alarm.alarm_id) is True
        assert (await store.list())[0].state == "done"

    @pytest.mark.asyncio
    async def test_snoozed_to_done(self, store):
        alarm = await store.create("alarm", T0)
        await store.mark_ringing(alarm.alarm_id, T0, 0.0)
        await store.snooze(alarm.alarm_id, T0 + timedelta(minutes=9))

        assert await store.mark_done(alarm.alarm_id) is True
        assert (await store.list())[0].state == "done"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("before", ["scheduled", "snoozed", "ringing"])
    async def test_to_missed(self, store, before):
        alarm = await store.create("alarm", T0)
        if before in ("snoozed", "ringing"):
            await store.mark_ringing(alarm.alarm_id, T0, 0.0)
        if before == "snoozed":
            await store.snooze(alarm.alarm_id, T0 + timedelta(minutes=9))

        assert await store.mark_missed(alarm.alarm_id) is True
        assert (await store.list())[0].state == "missed"

    @pytest.mark.asyncio
    async def test_scheduled_cannot_snooze_or_finish(self, store):
        alarm = await store.create("alarm", T0)

        assert await store.snooze(alarm.alarm_id, T0 + timedelta(minutes=9)) is False
        assert await store.mark_done(alarm.alarm_id) is False
        assert (await store.list())[0].state == "scheduled"

    @pytest.mark.asyncio
    async def test_ringing_cannot_ring_again(self, store):
        alarm = await store.create("alarm", T0)
        await store.mark_ringing(alarm.alarm_id, T0, 0.0)

        assert await store.mark_ringing(alarm.alarm_id, T0, 5.0) is False
        assert (await store.list())[0].late_by_s == 0.0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("final", ["done", "missed"])
    async def test_finished_alarms_stay_finished(self, store, final):
        alarm = await store.create("alarm", T0)
        await store.mark_ringing(alarm.alarm_id, T0, 0.0)
        await (store.mark_done if final == "done" else store.mark_missed)(alarm.alarm_id)

        assert await store.mark_ringing(alarm.alarm_id, T0, 0.0) is False
        assert await store.snooze(alarm.alarm_id, T0) is False
        assert await store.mark_done(alarm.alarm_id) is False
        assert await store.mark_missed(alarm.alarm_id) is False
        assert (await store.list())[0].state == final

    @pytest.mark.asyncio
    async def test_unknown_id_transitions_nothing(self, store):
        assert await store.mark_ringing("no-such-alarm", T0, 0.0) is False

    @pytest.mark.asyncio
    async def test_negative_late_by_is_rejected(self, store):
        alarm = await store.create("alarm", T0)

        with pytest.raises(ValueError, match="late_by_s"):
            await store.mark_ringing(alarm.alarm_id, T0, -1.0)


class TestDue:
    @pytest.mark.asyncio
    async def test_due_returns_scheduled_alarms_at_or_before_now(self, store):
        past = await store.create("alarm", T0 - timedelta(minutes=1))
        exact = await store.create("timer", T0)
        await store.create("alarm", T0 + timedelta(seconds=1))

        due = await store.due(T0)

        assert [a.alarm_id for a in due] == [past.alarm_id, exact.alarm_id]

    @pytest.mark.asyncio
    async def test_due_includes_a_snooze_that_has_ended(self, store):
        alarm = await store.create("alarm", T0 - timedelta(minutes=10))
        await store.mark_ringing(alarm.alarm_id, T0 - timedelta(minutes=10), 0.0)
        await store.snooze(alarm.alarm_id, T0)

        assert [a.alarm_id for a in await store.due(T0)] == [alarm.alarm_id]
        assert await store.due(T0 - timedelta(seconds=1)) == []

    @pytest.mark.asyncio
    async def test_due_excludes_ringing_done_and_missed(self, store):
        for mark in ("ringing", "done", "missed"):
            alarm = await store.create("alarm", T0 - timedelta(minutes=5))
            await store.mark_ringing(alarm.alarm_id, T0, 0.0)
            if mark == "done":
                await store.mark_done(alarm.alarm_id)
            elif mark == "missed":
                await store.mark_missed(alarm.alarm_id)

        assert await store.due(T0) == []

    @pytest.mark.asyncio
    async def test_due_compares_instants_not_strings(self, store):
        plus_two = timezone(timedelta(hours=2))
        alarm = await store.create("alarm", datetime(2026, 1, 5, 8, 59, tzinfo=plus_two))

        assert [a.alarm_id for a in await store.due(T0)] == [alarm.alarm_id]

    @pytest.mark.asyncio
    async def test_naive_now_is_rejected(self, store):
        with pytest.raises(ValueError, match="aware"):
            await store.due(datetime(2026, 1, 5, 7, 0))

    @pytest.mark.asyncio
    async def test_next_due_is_the_earliest_pending_alarm(self, store):
        await store.create("alarm", T0 + timedelta(hours=2))
        soonest = await store.create("alarm", T0 + timedelta(hours=1))
        ringing = await store.create("alarm", T0)
        await store.mark_ringing(ringing.alarm_id, T0, 0.0)

        assert (await store.next_due()).alarm_id == soonest.alarm_id

    @pytest.mark.asyncio
    async def test_next_due_counts_a_snooze(self, store):
        await store.create("alarm", T0 + timedelta(hours=1))
        snoozed = await store.create("alarm", T0)
        await store.mark_ringing(snoozed.alarm_id, T0, 0.0)
        await store.snooze(snoozed.alarm_id, T0 + timedelta(minutes=9))

        assert (await store.next_due()).alarm_id == snoozed.alarm_id

    @pytest.mark.asyncio
    async def test_next_due_is_none_with_nothing_pending(self, store):
        alarm = await store.create("alarm", T0)
        await store.mark_ringing(alarm.alarm_id, T0, 0.0)

        assert await store.next_due() is None


class TestRev:
    @pytest.mark.asyncio
    async def test_starts_at_zero(self, store):
        assert await store.rev() == 0

    @pytest.mark.asyncio
    async def test_every_write_increments_it(self, store):
        alarm = await store.create("alarm", T0)
        assert await store.rev() == 1
        await store.mark_ringing(alarm.alarm_id, T0, 0.0)
        assert await store.rev() == 2
        await store.snooze(alarm.alarm_id, T0 + timedelta(minutes=9))
        assert await store.rev() == 3
        await store.mark_ringing(alarm.alarm_id, T0 + timedelta(minutes=9), 0.0)
        assert await store.rev() == 4
        await store.mark_done(alarm.alarm_id)
        assert await store.rev() == 5
        other = await store.create("timer", T0)
        await store.mark_missed(other.alarm_id)
        assert await store.rev() == 7
        await store.cancel(other.alarm_id)
        assert await store.rev() == 8

    @pytest.mark.asyncio
    async def test_a_write_that_changes_nothing_leaves_it(self, store):
        alarm = await store.create("alarm", T0)

        await store.mark_done(alarm.alarm_id)
        await store.cancel("no-such-alarm")

        assert await store.rev() == 1

    @pytest.mark.asyncio
    async def test_it_is_store_wide_and_durable(self, store, tmp_path):
        await store.create("alarm", T0)
        await store.create("timer", T0)
        reopened_db = CapabilitiesDb(str(tmp_path / "capabilities.db"))
        reopened = AlarmStore(reopened_db)
        reopened.migrate()

        assert await reopened.rev() == 2
        await reopened.create("alarm", T0)
        assert await reopened.rev() == 3
        reopened_db.close()


class TestCorruptRows:
    @staticmethod
    def _errors_naming(caplog, alarm_id: str) -> list[logging.LogRecord]:
        return [
            r
            for r in caplog.records
            if r.name == STORE_LOGGER and r.levelno == logging.ERROR and alarm_id in r.getMessage()
        ]

    @pytest.mark.asyncio
    async def test_malformed_time_is_skipped_logged_once_and_cancellable(self, store, db, caplog):
        good = await store.create("alarm", T0 - timedelta(minutes=1))
        bad = await _insert_raw(db, next_fire_at="seven tomorrow")
        caplog.set_level(logging.ERROR, logger=STORE_LOGGER)

        assert [a.alarm_id for a in await store.due(T0)] == [good.alarm_id]
        assert [a.alarm_id for a in await store.due(T0)] == [good.alarm_id]
        assert (await store.next_due()).alarm_id == good.alarm_id
        assert [a.alarm_id for a in await store.list()] == [good.alarm_id]
        assert len(self._errors_naming(caplog, bad)) == 1

        assert await store.cancel(bad) is True
        assert await _raw_row(db, bad) is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "damage",
        [
            {"scheduled_for": "2026-13-45T00:00:00+00:00"},
            {"next_fire_at": "2026-01-05T06:00:00"},
            {"next_fire_at": 12345},
            {"next_fire_at": "0001-01-01T00:00:00+05:00"},
            {"repeat": "{not json"},
            {"repeat": '{"kind": "hourly", "hour": 7, "minute": 0}'},
            {"repeat": '{"kind": ["daily"], "hour": 7, "minute": 0}'},
            {"repeat": '{"kind": {"daily": 1}, "hour": 7, "minute": 0}'},
            {"repeat": '{"kind": "daily", "hour": [7], "minute": 0}'},
            {"repeat": "null"},
            {"repeat": "7"},
            {"state": "exploded"},
            {"kind": "reminder"},
            {"late_by_s": "late"},
        ],
    )
    async def test_each_undecodable_field_skips_only_that_row(self, store, db, caplog, damage):
        good = await store.create("alarm", T0 - timedelta(minutes=1))
        bad = await _insert_raw(db, **damage)
        caplog.set_level(logging.ERROR, logger=STORE_LOGGER)

        assert [a.alarm_id for a in await store.list()] == [good.alarm_id]
        assert [a.alarm_id for a in await store.due(T0)] == [good.alarm_id]
        assert len(self._errors_naming(caplog, bad)) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("column", ["next_fire_at", "scheduled_for", "repeat", "label"])
    async def test_invalid_utf8_text_is_skipped_logged_once_and_cancellable(
        self, store, db, caplog, column
    ):
        good = await store.create("alarm", T0 - timedelta(minutes=1))
        bad = await _insert_raw(db)
        # Column names come from the parametrize list above, never from input.
        sql = f"UPDATE alarms SET {column} = CAST(X'C328FF' AS TEXT) WHERE id = ?"
        await db.run(lambda conn: conn.execute(sql, (bad,)))
        caplog.set_level(logging.ERROR, logger=STORE_LOGGER)

        assert [a.alarm_id for a in await store.due(T0)] == [good.alarm_id]
        assert (await store.next_due()).alarm_id == good.alarm_id
        assert [a.alarm_id for a in await store.list()] == [good.alarm_id]
        assert len(self._errors_naming(caplog, bad)) == 1

        assert await store.cancel(bad) is True
        assert await _raw_row(db, bad) is None

    @pytest.mark.asyncio
    async def test_wrong_typed_repeat_is_skipped_logged_once_and_cancellable(
        self, store, db, caplog
    ):
        good = await store.create("alarm", T0 - timedelta(minutes=1))
        bad = await _insert_raw(db, repeat='{"kind": ["daily"], "hour": 7, "minute": 0}')
        caplog.set_level(logging.ERROR, logger=STORE_LOGGER)

        assert [a.alarm_id for a in await store.due(T0)] == [good.alarm_id]
        assert (await store.next_due()).alarm_id == good.alarm_id
        assert [a.alarm_id for a in await store.list()] == [good.alarm_id]
        assert len(self._errors_naming(caplog, bad)) == 1

        assert await store.cancel(bad) is True
        assert await _raw_row(db, bad) is None

    @pytest.mark.asyncio
    async def test_reads_leave_the_shared_connection_returning_text(self, store, db):
        await store.create("alarm", T0, label="gym")
        await _insert_raw(db, next_fire_at="seven tomorrow")

        await store.list()

        assert await db.run(lambda conn: conn.execute("SELECT 'text'").fetchone()[0]) == "text"

    @pytest.mark.asyncio
    async def test_a_decodable_raw_row_is_read(self, store, db):
        alarm_id = await _insert_raw(db, repeat='{"kind": "daily", "hour": 6, "minute": 0}')

        [alarm] = await store.list()

        assert alarm.alarm_id == alarm_id
        assert alarm.repeat == RepeatRule("daily", 6, 0)
