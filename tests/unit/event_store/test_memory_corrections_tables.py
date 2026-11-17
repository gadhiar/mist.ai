"""The correction log's two event-store tables (mist-read-path T21, Design point 12).

`memory_corrections` holds one immutable row per forget or correct; `memory_corrections_applied`
holds whether each correction was applied under an epoch, keyed by (`correction_id`,
`epoch_id`) as `extraction_applied` is, so the log rows never change and a new epoch sees every
correction as unapplied.
"""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from backend.event_store.store import MEMORY_CORRECTION_COLUMNS, EventStore

_TABLES = {"memory_corrections", "memory_corrections_applied"}

_COLUMNS = (
    "correction_id",
    "recorded_at",
    "action",
    "fact_key",
    "subject_id",
    "subject_name",
    "relation",
    "object_id",
    "object_name",
    "replacement_object_name",
    "surface",
    "session_id",
    "conversation_id",
    "turn_index",
    "source_event_id",
    "origin",
)


def _store(path: object = ":memory:") -> EventStore:
    store = EventStore(db_path=str(path))
    store.initialize()
    return store


def _row(correction_id: str, recorded_at: str, **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "correction_id": correction_id,
        "recorded_at": recorded_at,
        "action": "forget",
        "fact_key": f"key-{correction_id}",
        "subject_id": "wren",
        "subject_name": "Wren",
        "relation": "MEMBER_OF",
        "object_id": "kiln-street-pottery",
        "object_name": "Kiln Street Pottery",
        "replacement_object_name": None,
        "surface": "ws",
        "session_id": "s1",
        "conversation_id": "c1",
        "turn_index": 3,
        "source_event_id": "evt-1",
        "origin": "real",
    }
    row.update(overrides)
    return row


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _ids(rows: list[dict[str, Any]]) -> list[str]:
    return [row["correction_id"] for row in rows]


class TestTables:
    def test_the_log_has_the_design_point_12_columns_keyed_by_correction_id(self) -> None:
        conn = _store()._get_connection()

        info = conn.execute("PRAGMA table_info(memory_corrections)").fetchall()

        assert tuple(row["name"] for row in info) == _COLUMNS == MEMORY_CORRECTION_COLUMNS
        assert [row["name"] for row in info if row["pk"]] == ["correction_id"]
        not_null = {row["name"] for row in info if row["notnull"]}
        nullable = {
            "replacement_object_name",
            "session_id",
            "conversation_id",
            "turn_index",
            "source_event_id",
        }
        assert not_null == set(_COLUMNS) - nullable - {"correction_id"}

    def test_the_applied_table_is_keyed_by_correction_and_epoch(self) -> None:
        conn = _store()._get_connection()

        info = conn.execute("PRAGMA table_info(memory_corrections_applied)").fetchall()

        keys = sorted((row["pk"], row["name"]) for row in info if row["pk"])
        assert [name for _, name in keys] == ["correction_id", "epoch_id"]
        assert {row["name"] for row in info} == {
            "correction_id",
            "epoch_id",
            "outcome",
            "applied_at",
        }

    def test_a_database_from_before_the_log_gains_both_tables_and_keeps_its_rows(
        self, tmp_path
    ) -> None:
        db = tmp_path / "events.db"
        old = _store(db)
        old.start_session("s1", input_modality="text")
        conn = old._get_connection()
        for table in _TABLES:
            conn.execute(f"DROP TABLE {table}")
        assert not (_tables(conn) & _TABLES)
        old.close()

        store = _store(db)
        store.initialize()  # idempotent

        assert _tables(store._get_connection()) >= _TABLES
        assert store.get_session("s1") is not None


class TestAppendOnly:
    def test_the_store_has_no_update_or_delete_path_for_the_log(self) -> None:
        names = [name for name in dir(EventStore) if "correction" in name]

        assert names, "the log has a store API"
        assert not [n for n in names if any(w in n for w in ("update", "delete", "remove"))]

    @pytest.mark.parametrize(
        "statement",
        [
            "UPDATE memory_corrections SET object_name = 'x'",
            "DELETE FROM memory_corrections",
        ],
    )
    def test_the_database_refuses_to_change_a_logged_row(self, statement: str) -> None:
        store = _store()
        store.append_memory_correction(_row("corr-a", "2026-03-10T21:04:59.000000+00:00"))

        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            store._get_connection().execute(statement)

        [row] = store.list_memory_corrections()
        assert row["object_name"] == "Kiln Street Pottery"

    def test_a_logged_id_cannot_be_appended_twice(self) -> None:
        store = _store()
        store.append_memory_correction(_row("corr-a", "2026-03-10T21:04:59.000000+00:00"))

        with pytest.raises(sqlite3.IntegrityError):
            store.append_memory_correction(_row("corr-a", "2026-03-11T00:00:00.000000+00:00"))

    def test_a_row_with_missing_or_extra_columns_is_refused(self) -> None:
        store = _store()
        row = _row("corr-a", "2026-03-10T21:04:59.000000+00:00")
        del row["origin"]

        with pytest.raises(ValueError, match="origin"):
            store.append_memory_correction(row)
        with pytest.raises(ValueError, match="stray"):
            store.append_memory_correction(
                {**_row("corr-b", "2026-03-10T21:04:59.000000+00:00"), "stray": 1}
            )
        assert store.list_memory_corrections() == []

    def test_a_correct_needs_its_replacement_and_a_forget_has_none(self) -> None:
        store = _store()

        with pytest.raises(sqlite3.IntegrityError):
            store.append_memory_correction(
                _row("corr-a", "2026-03-10T00:00:00.000000+00:00", action="correct")
            )
        with pytest.raises(sqlite3.IntegrityError):
            store.append_memory_correction(
                _row(
                    "corr-b",
                    "2026-03-10T00:00:00.000000+00:00",
                    replacement_object_name="Ledgerly",
                )
            )


class TestLogOrder:
    def test_the_log_lists_by_recorded_at_then_correction_id(self) -> None:
        store = _store()
        for correction_id, at in (
            ("corr-c", "2026-03-12T00:00:00.000000+00:00"),
            ("corr-b", "2026-03-10T00:00:00.000000+00:00"),
            ("corr-a", "2026-03-12T00:00:00.000000+00:00"),
        ):
            store.append_memory_correction(_row(correction_id, at))

        rows = store.list_memory_corrections()

        assert _ids(rows) == ["corr-b", "corr-a", "corr-c"]
        assert rows[0] == _row("corr-b", "2026-03-10T00:00:00.000000+00:00")

    def test_origins_filter_the_log_the_way_they_filter_turns(self) -> None:
        store = _store()
        store.append_memory_correction(_row("corr-a", "2026-03-10T00:00:00.000000+00:00"))
        store.append_memory_correction(
            _row("corr-b", "2026-03-11T00:00:00.000000+00:00", origin="test")
        )

        assert _ids(store.list_memory_corrections(origins=("real",))) == ["corr-a"]
        assert _ids(store.list_memory_corrections(origins=("real", "test"))) == [
            "corr-a",
            "corr-b",
        ]
        assert _ids(store.list_memory_corrections()) == ["corr-a", "corr-b"]
        with pytest.raises(ValueError, match="non-empty"):
            store.list_memory_corrections(origins=())


class TestApplied:
    def test_a_new_epoch_sees_every_correction_unapplied_and_the_old_rows_stay(self) -> None:
        store = _store()
        for correction_id, at in (
            ("corr-a", "2026-03-10T00:00:00.000000+00:00"),
            ("corr-b", "2026-03-11T00:00:00.000000+00:00"),
            ("corr-c", "2026-03-12T00:00:00.000000+00:00"),
        ):
            store.append_memory_correction(_row(correction_id, at))
        store.mark_memory_correction_applied(
            correction_id="corr-a", epoch_id=1, outcome="applied", applied_at="t1"
        )
        store.mark_memory_correction_applied(
            correction_id="corr-c", epoch_id=1, outcome="unresolved", applied_at="t2"
        )

        assert _ids(store.list_unapplied_memory_corrections(1)) == ["corr-b"]
        assert _ids(store.list_unapplied_memory_corrections(2)) == ["corr-a", "corr-b", "corr-c"]
        assert store.get_memory_corrections_applied(1) == {
            "corr-a": "applied",
            "corr-c": "unresolved",
        }
        assert store.get_memory_corrections_applied(2) == {}

        store.mark_memory_correction_applied(
            correction_id="corr-b", epoch_id=2, outcome="noop", applied_at="t3"
        )

        assert store.get_memory_corrections_applied(2) == {"corr-b": "noop"}
        assert store.get_memory_corrections_applied(1) == {
            "corr-a": "applied",
            "corr-c": "unresolved",
        }
        assert _ids(store.list_memory_corrections()) == ["corr-a", "corr-b", "corr-c"]

    def test_the_unapplied_listing_takes_the_origins_filter(self) -> None:
        store = _store()
        store.append_memory_correction(_row("corr-a", "2026-03-10T00:00:00.000000+00:00"))
        store.append_memory_correction(
            _row("corr-b", "2026-03-11T00:00:00.000000+00:00", origin="test")
        )

        assert _ids(store.list_unapplied_memory_corrections(1, origins=("real",))) == ["corr-a"]
        with pytest.raises(ValueError, match="non-empty"):
            store.list_unapplied_memory_corrections(1, origins=())

    def test_an_unknown_outcome_is_refused(self) -> None:
        store = _store()
        store.append_memory_correction(_row("corr-a", "2026-03-10T00:00:00.000000+00:00"))

        with pytest.raises(ValueError, match="outcome"):
            store.mark_memory_correction_applied(
                correction_id="corr-a", epoch_id=1, outcome="done", applied_at="t1"
            )
        assert store.get_memory_corrections_applied(1) == {}
