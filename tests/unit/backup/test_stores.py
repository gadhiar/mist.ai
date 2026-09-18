"""MIS-140 T1: the SQLite leg captures three NAMED stores and nothing else.

The central test here is the negative one. Live `./data` holds five `.db` files:
the three stores below plus `event_store.pre-r1.4-backup-2026-07-31.db` and
`event_store.pre-reset-backup-2026-06-09.db` (14331904 bytes). A `*.db` glob --
which is what `scripts/hydration/snapshot.py:519` does, correctly, against a
directory it owns -- sweeps both into the artifact, doubling it and giving a
hurried restore a second, older event store to choose from.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from scripts.backup.errors import BackupError
from scripts.backup.stores import (
    LIVE_STORE_FILENAMES,
    STORES_DIRNAME,
    capture_stores,
    copy_store,
    count_rows,
)

from .conftest import STALE_BACKUP_FILENAMES, make_store


class TestNamedNotGlobbed:
    def test_captures_exactly_the_three_named_stores(self, state_root, tmp_path):
        artifact = tmp_path / "artifact"
        artifact.mkdir()
        capture_stores(state_root, artifact)
        captured = sorted(p.name for p in (artifact / STORES_DIRNAME).iterdir())
        assert captured == sorted(LIVE_STORE_FILENAMES)

    @pytest.mark.parametrize("stale", STALE_BACKUP_FILENAMES)
    def test_stale_backup_files_are_not_swept_in(self, state_root, tmp_path, stale):
        # The mandated negative test. Both files exist in the fixture root and
        # neither may appear in the artifact.
        assert (state_root / stale).is_file()
        artifact = tmp_path / "artifact"
        artifact.mkdir()
        capture_stores(state_root, artifact)
        assert not (artifact / STORES_DIRNAME / stale).exists()
        assert stale not in {p.name for p in artifact.rglob("*")}

    def test_a_db_file_added_next_month_is_not_captured_by_accident(self, state_root, tmp_path):
        # Enumeration cuts both ways and this records the cost honestly: a NEW
        # store must be added to LIVE_STORE_FILENAMES, and until it is, it is
        # not in the artifact. That is a maintenance burden accepted in exchange
        # for never capturing junk.
        make_store(state_root / "some_future_store.db", table="rows", rows=1)
        artifact = tmp_path / "artifact"
        artifact.mkdir()
        capture_stores(state_root, artifact)
        assert not (artifact / STORES_DIRNAME / "some_future_store.db").exists()

    def test_the_derived_vector_store_is_not_captured(self, state_root, tmp_path):
        artifact = tmp_path / "artifact"
        artifact.mkdir()
        capture_stores(state_root, artifact)
        assert not (artifact / STORES_DIRNAME / "vector_store").exists()


class TestWalConsistency:
    def test_uncheckpointed_wal_rows_reach_the_copy(self, tmp_path):
        # The reason `Connection.backup()` exists here. Rows are committed and
        # the connection is held OPEN, so they are still in `-wal` when the
        # capture runs; a byte copy of the `.db` alone would miss them.
        source = tmp_path / "event_store.db"
        make_store(source, table="events", rows=2)
        live = sqlite3.connect(str(source))
        try:
            live.execute("PRAGMA journal_mode=WAL")
            live.executemany(
                "INSERT INTO events (payload) VALUES (?)",
                [("wal-only-1",), ("wal-only-2",)],
            )
            live.commit()
            assert Path(f"{source}-wal").exists()

            destination = tmp_path / "out" / "event_store.db"
            copy_store(source, destination)
            assert count_rows(destination) == {"events": 4}
        finally:
            live.close()

    def test_sidecars_are_stripped_after_the_row_count(self, state_root, tmp_path):
        # Reopening a WAL copy recreates `-wal` and `-shm`, so stripping them
        # before the count leaves them in the finished artifact.
        artifact = tmp_path / "artifact"
        artifact.mkdir()
        capture_stores(state_root, artifact)
        leftovers = [p.name for p in artifact.rglob("*") if p.name.endswith(("-wal", "-shm"))]
        assert leftovers == []

    def test_the_source_is_opened_read_only(self, tmp_path):
        source = tmp_path / "event_store.db"
        make_store(source, table="events", rows=1)
        before = source.read_bytes()
        copy_store(source, tmp_path / "out" / "event_store.db")
        assert source.read_bytes() == before


class TestRowCounts:
    def test_every_user_table_is_counted(self, state_root, tmp_path):
        artifact = tmp_path / "artifact"
        artifact.mkdir()
        captures = {c.filename: c for c in capture_stores(state_root, artifact)}
        assert captures["event_store.db"].row_counts == {"conversation_turn_events": 3}
        assert captures["extraction_cache.db"].row_counts == {"extraction_cache": 2}
        assert captures["vault_sidecar.db"].row_counts == {"vault_chunks": 5}

    def test_a_table_added_to_a_store_is_counted_without_a_code_change(self, state_root, tmp_path):
        conn = sqlite3.connect(str(state_root / "event_store.db"))
        try:
            conn.execute("CREATE TABLE epoch_ledger (id INTEGER PRIMARY KEY)")
            conn.execute("INSERT INTO epoch_ledger (id) VALUES (1)")
            conn.commit()
        finally:
            conn.close()
        artifact = tmp_path / "artifact"
        artifact.mkdir()
        captures = {c.filename: c for c in capture_stores(state_root, artifact)}
        assert captures["event_store.db"].row_counts["epoch_ledger"] == 1


class TestMissingStores:
    def test_an_absent_store_is_recorded_rather_than_skipped_quietly(self, state_root, tmp_path):
        (state_root / "extraction_cache.db").unlink()
        artifact = tmp_path / "artifact"
        artifact.mkdir()
        captures = {c.filename: c for c in capture_stores(state_root, artifact)}
        assert captures["extraction_cache.db"].present is False
        assert captures["extraction_cache.db"].to_manifest_entry()["present"] is False
        assert captures["event_store.db"].present is True

    def test_an_unreadable_store_fails_loudly(self, tmp_path):
        state = tmp_path / "live-data"
        state.mkdir()
        (state / "event_store.db").write_text("this is not a database", encoding="utf-8")
        with pytest.raises(BackupError):
            capture_stores(state, tmp_path / "artifact")
