"""Event-store tables and queries behind the extraction backlog (T2a)."""

from __future__ import annotations

import sqlite3
from datetime import datetime

import pytest

from backend.event_store.models import ConversationTurnEvent
from backend.event_store.store import EventStore

_BACKLOG_TABLES = {
    "extraction_applied",
    "extraction_attempts",
    "extraction_activation",
    "extraction_legacy_turns",
}


def _store(path=":memory:") -> EventStore:
    store = EventStore(db_path=str(path))
    store.initialize()
    return store


def _append(store: EventStore, *, session: str, index: int, ts: str, event_id: str) -> None:
    if store.get_session(session) is None:
        store.start_session(session, input_modality="text", origin="test")
    store.append_turn(
        ConversationTurnEvent(
            session_id=session,
            turn_index=index,
            timestamp=datetime.fromisoformat(ts),
            user_utterance=f"utterance {event_id}",
            system_response="ok",
            event_id=event_id,
        )
    )


def _tables(conn) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


class TestMigration:
    def test_a_pre_t2a_database_gains_the_tables_and_keeps_its_rows(self, tmp_path):
        # Arrange: a database created by the schema as it was before T2a.
        db = tmp_path / "events.db"
        old = _store(db)
        _append(old, session="s", index=0, ts="2026-09-01T10:00:00+00:00", event_id="e0")
        conn = old._get_connection()
        for table in _BACKLOG_TABLES:
            conn.execute(f"DROP TABLE {table}")
        assert not (_tables(conn) & _BACKLOG_TABLES)
        old.close()

        # Act
        store = _store(db)
        store.initialize()  # idempotent

        # Assert
        assert _tables(store._get_connection()) >= _BACKLOG_TABLES
        assert [t["event_id"] for t in store.list_turn_keys_in_replay_order()] == ["e0"]


class TestReplayOrderKeys:
    def test_keys_come_back_in_timestamp_session_turn_order_for_every_origin(self):
        store = _store()
        _append(store, session="b", index=1, ts="2026-09-01T10:05:00+00:00", event_id="b1")
        _append(store, session="a", index=0, ts="2026-09-01T10:01:00+00:00", event_id="a0")
        _append(store, session="b", index=0, ts="2026-09-01T10:04:00+00:00", event_id="b0")
        _append(store, session="a", index=1, ts="2026-09-01T10:04:00+00:00", event_id="a1")

        keys = store.list_turn_keys_in_replay_order()

        assert [k["event_id"] for k in keys] == ["a0", "a1", "b0", "b1"]
        assert set(keys[0]) == {"event_id", "session_id", "turn_index", "timestamp"}

    def test_matches_get_all_turns_for_reextraction_order(self):
        store = _store()
        _append(store, session="b", index=0, ts="2026-09-01T10:02:00+00:00", event_id="b0")
        _append(store, session="a", index=0, ts="2026-09-01T10:02:00+00:00", event_id="a0")
        _append(store, session="a", index=1, ts="2026-09-01T10:01:00+00:00", event_id="a1")

        ours = [k["event_id"] for k in store.list_turn_keys_in_replay_order()]
        rebuild = [t["event_id"] for t in store.get_all_turns_for_reextraction()]

        assert ours == rebuild


class TestActivation:
    def test_records_floor_markers_and_legacy_rows_once(self):
        store = _store()

        created = store.record_extraction_activation(
            epoch_id=1,
            activated_at="2026-09-01T00:00:00+00:00",
            turns_at_activation=3,
            applied_event_ids=["e1", "e3"],
            legacy_event_ids=["e2"],
        )
        again = store.record_extraction_activation(
            epoch_id=1,
            activated_at="2026-09-02T00:00:00+00:00",
            turns_at_activation=9,
            applied_event_ids=["e9"],
            legacy_event_ids=["e8"],
        )

        assert (created, again) == (True, False)
        row = store.get_extraction_activation(1)
        assert row["activated_at"] == "2026-09-01T00:00:00+00:00"
        assert (row["marked_applied"], row["legacy_unextracted"]) == (2, 1)
        assert store.get_extraction_applied(1) == {"e1": "applied", "e3": "applied"}
        assert store.get_extraction_legacy_turns(1) == {"e2"}

    def test_floor_is_per_epoch(self):
        store = _store()
        store.record_extraction_activation(
            epoch_id=1,
            activated_at="t",
            turns_at_activation=0,
            applied_event_ids=[],
            legacy_event_ids=["e1"],
        )

        assert store.get_extraction_activation(2) is None
        assert store.get_extraction_legacy_turns(2) == set()


class TestAppliedMarkers:
    def test_stage_moves_from_curated_to_applied(self):
        store = _store()

        store.mark_extraction_stage(event_id="e1", epoch_id=1, stage="curated", updated_at="t1")
        after_curated = store.get_extraction_applied(1)
        store.mark_extraction_stage(event_id="e1", epoch_id=1, stage="applied", updated_at="t2")

        assert after_curated == {"e1": "curated"}
        assert store.get_extraction_applied(1) == {"e1": "applied"}

    def test_rejects_an_unknown_stage(self):
        store = _store()

        with pytest.raises(ValueError, match="unknown extraction apply stage"):
            store.mark_extraction_stage(event_id="e1", epoch_id=1, stage="done", updated_at="t")

    def test_delete_removes_only_that_epochs_marker(self):
        store = _store()
        store.mark_extraction_stage(event_id="e1", epoch_id=1, stage="applied", updated_at="t")
        store.mark_extraction_stage(event_id="e1", epoch_id=2, stage="applied", updated_at="t")

        assert store.delete_extraction_applied("e1", 1) is True
        assert store.delete_extraction_applied("e1", 1) is False
        assert store.get_extraction_applied(2) == {"e1": "applied"}


class TestAttempts:
    def _attempt(self, store, attempt: int, *, counted: bool, outcome: str = "failed") -> None:
        store.append_extraction_attempt(
            event_id="e1",
            epoch_id=1,
            attempt=attempt,
            job_id=f"job-{attempt}",
            request_id=f"req-{attempt}",
            turn_id="s:0",
            started_at="t0",
            finished_at="t1",
            duration_ms=12.5,
            error_code="upstream_llm" if counted else "unreachable",
            outcome=outcome,
            counted=counted,
        )

    def test_counts_only_counted_unretired_failures(self):
        store = _store()
        self._attempt(store, 1, counted=True)
        self._attempt(store, 2, counted=False, outcome="deferred")
        self._attempt(store, 3, counted=True)

        before = store.count_counted_extraction_failures("e1", 1)
        retired = store.retire_extraction_attempts("e1", 1)
        after = store.count_counted_extraction_failures("e1", 1)

        assert (before, retired, after) == (2, 3, 0)
        assert store.count_extraction_attempts("e1", 1) == 3

    def test_rows_keep_every_column_the_brief_names(self):
        store = _store()
        self._attempt(store, 1, counted=True)

        (row,) = store.list_extraction_attempts(event_id="e1")

        assert {
            "event_id",
            "epoch_id",
            "attempt",
            "job_id",
            "request_id",
            "turn_id",
            "started_at",
            "finished_at",
            "duration_ms",
            "error_code",
            "outcome",
        } <= set(row)
        assert (row["job_id"], row["request_id"], row["duration_ms"]) == ("job-1", "req-1", 12.5)


class TestActivationAtomicity:
    def test_activation_rolls_back_on_failure(self):
        store = _store()
        conn = store._get_connection()
        conn.execute("DROP TABLE extraction_legacy_turns")

        with pytest.raises(sqlite3.OperationalError):
            store.record_extraction_activation(
                epoch_id=1,
                activated_at="t",
                turns_at_activation=1,
                applied_event_ids=["e1"],
                legacy_event_ids=["e2"],
            )

        assert store.get_extraction_activation(1) is None
        assert store.get_extraction_applied(1) == {}
