"""Tests for `CapabilitiesDb`: lazy open, pragmas, serialised runs and migrations."""

from __future__ import annotations

import asyncio
import sqlite3
import threading

import pytest

from backend.capabilities.db import CapabilitiesDb
from backend.errors import CapabilityStoreError

ALARM_STEPS = (
    "CREATE TABLE alarms (id TEXT PRIMARY KEY, label TEXT)",
    "CREATE INDEX alarms_label ON alarms (label)",
)


@pytest.fixture()
def db(tmp_path):
    database = CapabilitiesDb(str(tmp_path / "capabilities.db"))
    yield database
    database.close()


def _blocked_path(tmp_path) -> str:
    """A database path whose parent directory is a regular file, so no open can succeed."""
    parent = tmp_path / "not_a_directory"
    parent.write_text("", encoding="utf-8")
    return str(parent / "capabilities.db")


class TestLazyOpen:
    def test_constructor_touches_no_file(self, tmp_path):
        path = tmp_path / "sub" / "capabilities.db"

        CapabilitiesDb(str(path))

        assert not path.exists()
        assert not path.parent.exists()

    def test_unopenable_path_constructs_and_the_first_migrate_raises(self, tmp_path):
        database = CapabilitiesDb(_blocked_path(tmp_path))

        with pytest.raises(CapabilityStoreError, match="capabilities.db"):
            database.migrate("alarms", ALARM_STEPS)

    @pytest.mark.asyncio
    async def test_unopenable_path_raises_on_the_first_run(self, tmp_path):
        database = CapabilitiesDb(_blocked_path(tmp_path))

        with pytest.raises(CapabilityStoreError):
            await database.run(lambda conn: conn.execute("SELECT 1").fetchone())

    @pytest.mark.asyncio
    async def test_first_run_creates_the_parent_directory_and_file(self, tmp_path):
        path = tmp_path / "sub" / "capabilities.db"
        database = CapabilitiesDb(str(path))

        await database.run(lambda conn: conn.execute("SELECT 1").fetchone())

        assert path.exists()
        database.close()


class TestPragmas:
    @pytest.mark.asyncio
    async def test_opens_in_wal_mode(self, db):
        mode = await db.run(lambda conn: conn.execute("PRAGMA journal_mode").fetchone()[0])

        assert mode == "wal"

    @pytest.mark.asyncio
    async def test_opens_with_foreign_keys_on(self, db):
        enabled = await db.run(lambda conn: conn.execute("PRAGMA foreign_keys").fetchone()[0])

        assert enabled == 1

    @pytest.mark.asyncio
    async def test_connection_is_usable_from_worker_and_caller_threads(self, db):
        worker_thread = await db.run(lambda conn: threading.get_ident())

        version = db.migrate("alarms", ALARM_STEPS)

        assert worker_thread != threading.get_ident()
        assert version == 2


class TestRun:
    @pytest.mark.asyncio
    async def test_returns_the_function_result(self, db):
        db.migrate("alarms", ALARM_STEPS)

        def insert_and_count(conn: sqlite3.Connection) -> int:
            conn.execute("INSERT INTO alarms (id, label) VALUES (?, ?)", ("a1", "wake"))
            return conn.execute("SELECT COUNT(*) FROM alarms").fetchone()[0]

        assert await db.run(insert_and_count) == 1

    @pytest.mark.asyncio
    async def test_concurrent_runs_are_serialised_and_both_complete(self, db):
        first_entered = threading.Event()
        release_first = threading.Event()
        inside = 0
        most_inside = 0
        counter_lock = threading.Lock()

        def enter() -> None:
            nonlocal inside, most_inside
            with counter_lock:
                inside += 1
                most_inside = max(most_inside, inside)

        def leave() -> None:
            nonlocal inside
            with counter_lock:
                inside -= 1

        def first(conn: sqlite3.Connection) -> str:
            enter()
            first_entered.set()
            release_first.wait(timeout=5)
            leave()
            return "first"

        def second(conn: sqlite3.Connection) -> str:
            enter()
            leave()
            return "second"

        first_task = asyncio.create_task(db.run(first))
        assert await asyncio.to_thread(first_entered.wait, 5)
        second_task = asyncio.create_task(db.run(second))
        await asyncio.sleep(0.05)
        second_ran_early = most_inside > 1 or second_task.done()
        release_first.set()
        results = await asyncio.gather(first_task, second_task)

        assert not second_ran_early
        assert results == ["first", "second"]
        assert most_inside == 1

    @pytest.mark.asyncio
    async def test_exception_from_the_function_propagates_unchanged(self, db):
        def broken(conn: sqlite3.Connection) -> None:
            conn.execute("SELECT * FROM no_such_table")

        with pytest.raises(sqlite3.OperationalError):
            await db.run(broken)

    @pytest.mark.asyncio
    async def test_run_after_close_reopens(self, db):
        db.migrate("alarms", ALARM_STEPS)
        db.close()

        count = await db.run(lambda conn: conn.execute("SELECT COUNT(*) FROM alarms").fetchone()[0])

        assert count == 0


class TestMigrate:
    def test_applies_every_step_and_returns_the_version(self, db):
        assert db.migrate("alarms", ALARM_STEPS) == 2

    def test_is_idempotent_per_component(self, db):
        db.migrate("alarms", ALARM_STEPS)

        assert db.migrate("alarms", ALARM_STEPS) == 2

    def test_applies_only_new_steps(self, db):
        db.migrate("alarms", ALARM_STEPS[:1])

        version = db.migrate("alarms", ALARM_STEPS)

        assert version == 2

    def test_components_are_versioned_separately(self, db):
        db.migrate("alarms", ALARM_STEPS)

        version = db.migrate("action_log", ("CREATE TABLE action_log (id INTEGER PRIMARY KEY)",))

        assert version == 1

    def test_failed_step_rolls_back_and_keeps_the_earlier_version(self, db):
        steps = (*ALARM_STEPS, "CREATE TABLE extra (x INTEGER); INSERT INTO nowhere VALUES (1)")

        with pytest.raises(CapabilityStoreError, match="alarms"):
            db.migrate("alarms", steps)

        assert db.migrate("alarms", ALARM_STEPS) == 2
        assert db.migrate("alarms", (*ALARM_STEPS, "CREATE TABLE extra (x INTEGER)")) == 3

    def test_database_newer_than_the_steps_is_refused(self, db):
        db.migrate("alarms", ALARM_STEPS)

        with pytest.raises(CapabilityStoreError, match="newer"):
            db.migrate("alarms", ALARM_STEPS[:1])

    def test_versions_survive_a_reopen(self, tmp_path):
        path = str(tmp_path / "capabilities.db")
        first = CapabilitiesDb(path)
        first.migrate("alarms", ALARM_STEPS)
        first.close()

        second = CapabilitiesDb(path)
        version = second.migrate("alarms", ALARM_STEPS)
        second.close()

        assert version == 2
