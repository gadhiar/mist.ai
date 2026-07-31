"""Unit tests for EventStore.

Uses real in-memory SQLite (:memory:) -- fast enough and avoids
fake/real divergence.
"""

import sqlite3
from datetime import UTC, datetime

import pytest

from backend.event_store.models import ConversationTurnEvent
from backend.event_store.store import EventStore
from backend.knowledge.config import KnowledgeConfig
from backend.knowledge.ontologies import ONTOLOGY_V1_0_0


def _build_turn_event(
    *,
    session_id: str,
    turn_index: int = 0,
    user_utterance: str = "hello",
    system_response: str = "hi there",
    timestamp: datetime | None = None,
) -> ConversationTurnEvent:
    """Build a valid ConversationTurnEvent with overridable fields."""
    return ConversationTurnEvent(
        session_id=session_id,
        turn_index=turn_index,
        timestamp=timestamp or datetime.now(UTC),
        user_utterance=user_utterance,
        system_response=system_response,
    )


@pytest.fixture()
def store() -> EventStore:
    """Create an initialized in-memory EventStore."""
    s = EventStore(db_path=":memory:")
    s.initialize()
    return s


class TestInitialize:
    def test_initialize_creates_tables(self):
        s = EventStore(db_path=":memory:")
        s.initialize()

        conn = s._get_connection()
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        table_names = [row["name"] for row in cursor.fetchall()]

        assert "conversation_sessions" in table_names
        assert "conversation_turn_events" in table_names


class TestSessionLifecycle:
    def test_start_session_returns_the_id_passed_in(self, store: EventStore):
        """R1.3.1 fix round 1: the caller supplies the session id (the
        chat-layer id) rather than the store minting its own uuid4 -- the
        namespace collapse the id-mismatch bugs in catch-up motivated.
        """
        session_id = store.start_session("caller-supplied-id")

        assert session_id == "caller-supplied-id"

    def test_end_session_updates_ended_at(self, store: EventStore):
        session_id = store.start_session("s-end")

        store.end_session(session_id)

        session = store.get_session(session_id)
        assert session is not None
        assert session.ended_at is not None

    def test_get_session_returns_session_data(self, store: EventStore):
        session_id = store.start_session("s-info", input_modality="text")

        session = store.get_session(session_id)

        assert session is not None
        assert session.session_id == session_id
        assert session.input_modality == "text"
        assert session.turn_count == 0
        assert session.started_at is not None
        assert session.ended_at is None


class TestTurnEvents:
    def test_append_turn_stores_event(self, store: EventStore):
        session_id = store.start_session("s-append")
        event = _build_turn_event(
            session_id=session_id,
            user_utterance="what is the weather",
            system_response="I cannot check the weather yet",
        )

        event_id = store.append_turn(event)

        turns = store.get_turns(session_id)
        assert len(turns) == 1
        assert turns[0]["event_id"] == event_id
        assert turns[0]["user_utterance"] == "what is the weather"
        assert turns[0]["system_response"] == "I cannot check the weather yet"

    def test_append_turn_increments_turn_index(self, store: EventStore):
        session_id = store.start_session("s-increment")

        for i in range(3):
            event = _build_turn_event(session_id=session_id, turn_index=i)
            store.append_turn(event)

        turns = store.get_turns(session_id)

        assert len(turns) == 3
        assert turns[0]["turn_index"] == 0
        assert turns[1]["turn_index"] == 1
        assert turns[2]["turn_index"] == 2

    def test_get_turns_returns_all_turns_for_session(self, store: EventStore):
        session_id = store.start_session("s-get-turns")
        event_ids = []
        for i in range(3):
            event = _build_turn_event(session_id=session_id, turn_index=i)
            event_ids.append(store.append_turn(event))

        turns = store.get_turns(session_id)

        assert len(turns) == 3
        returned_ids = [t["event_id"] for t in turns]
        assert returned_ids == event_ids

    def test_get_turns_empty_for_unknown_session(self, store: EventStore):
        turns = store.get_turns("nonexistent-session-id")

        assert turns == []


class TestListSessionsWithTurns:
    def test_excludes_empty_sessions(self, store: EventStore):
        """A session row with no turns has nothing to synthesize."""
        empty = store.start_session("s-empty")
        populated = store.start_session("s-populated")
        store.append_turn(_build_turn_event(session_id=populated))

        result = store.list_sessions_with_turns()

        assert populated in result
        assert empty not in result

    def test_is_oldest_first(self, store: EventStore):
        """Catch-up drains oldest first so a long backlog makes visible progress.

        `first` gets a second turn after `second` starts, so its earliest
        turn (rowid 1) and latest turn (rowid 3) land on opposite sides of
        `second`'s only turn (rowid 2). Ordering by MIN(rowid) keeps
        [first, second]; a MIN -> MAX typo would flip it to [second, first],
        so this discriminates the two instead of passing under either.
        """
        first = store.start_session("s-first")
        store.append_turn(_build_turn_event(session_id=first, turn_index=0))
        second = store.start_session("s-second")
        store.append_turn(_build_turn_event(session_id=second, turn_index=0))
        store.append_turn(_build_turn_event(session_id=first, turn_index=1))

        result = store.list_sessions_with_turns()

        assert result.index(first) < result.index(second)


class TestOrigin:
    """Provenance discriminator on conversation_sessions -- see R1.4 Task 3.

    Uses `tmp_path` rather than the `:memory:` `store` fixture: the
    pre-existing-database test needs a file it can reopen with a second
    connection, which an in-memory database does not support.
    """

    def test_start_session_defaults_origin_to_real(self, tmp_path):
        """A caller that forgets to pass origin is counted as real usage,
        not silently excluded from a future rebuild.
        """
        store = EventStore(db_path=str(tmp_path / "es.db"))
        store.initialize()

        store.start_session("s-1", input_modality="text")

        conn = store._get_connection()
        row = conn.execute(
            "SELECT origin FROM conversation_sessions WHERE session_id = ?", ("s-1",)
        ).fetchone()
        assert row[0] == "real"

    def test_start_session_records_test_origin(self, tmp_path):
        store = EventStore(db_path=str(tmp_path / "es.db"))
        store.initialize()

        store.start_session("s-2", input_modality="text", origin="test")

        conn = store._get_connection()
        row = conn.execute(
            "SELECT origin FROM conversation_sessions WHERE session_id = ?", ("s-2",)
        ).fetchone()
        assert row[0] == "test"

    def test_origin_column_added_to_preexisting_db(self, tmp_path):
        """A database created before the column existed gains it on open."""
        db = tmp_path / "legacy.db"
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE conversation_sessions ("
            "session_id TEXT PRIMARY KEY, started_at TEXT NOT NULL, ended_at TEXT, "
            "turn_count INTEGER DEFAULT 0, input_modality TEXT DEFAULT 'voice')"
        )
        conn.commit()
        conn.close()

        store = EventStore(db_path=str(db))
        store.initialize()

        cols = {
            row[1]
            for row in store._get_connection().execute("PRAGMA table_info(conversation_sessions)")
        }
        assert "origin" in cols

    def test_migration_guard_is_idempotent_across_reopens(self, tmp_path):
        """The guarded ALTER must not raise `duplicate column name` the
        second time a database that already has `origin` is opened.
        """
        db_path = str(tmp_path / "es.db")
        first = EventStore(db_path=db_path)
        first.initialize()
        first.close()

        second = EventStore(db_path=db_path)
        second.initialize()  # must not raise

        cols = {
            row[1]
            for row in second._get_connection().execute("PRAGMA table_info(conversation_sessions)")
        }
        assert "origin" in cols


class TestEnsureInitialEpoch:
    """Provisional initial epoch -- see R1.4 Task 7 (spec 4.3, O2).

    Uses `tmp_path` rather than the `:memory:` `store` fixture for the same
    reason as `TestOrigin`: the pre-existing-database test needs a file it
    can reopen with a second connection.
    """

    def test_first_call_inserts_a_row_marked_provisional(self, tmp_path):
        """The provisional marking must be a real, queryable field."""
        store = EventStore(db_path=str(tmp_path / "es.db"))
        store.initialize()

        epoch = store.ensure_initial_epoch(
            now_iso="2026-07-31T00:00:00",
            ontology_version="onto-x",
            extraction_version="extract-x",
            model_hash="hash-x",
        )

        assert epoch["provisional"] == 1
        row = (
            store._get_connection()
            .execute("SELECT * FROM epoch_ledger WHERE epoch_id = ?", (epoch["epoch_id"],))
            .fetchone()
        )
        assert row["provisional"] == 1
        assert row["ontology_version"] == "onto-x"
        assert row["extraction_version"] == "extract-x"
        assert row["model_hash"] == "hash-x"
        assert row["activated_at"] == "2026-07-31T00:00:00"
        assert row["prev_epoch_id"] is None

    def test_first_call_adds_exactly_one_row(self, tmp_path):
        """Proves the first call genuinely inserts, not just that a later
        no-op call finds a row already there.
        """
        store = EventStore(db_path=str(tmp_path / "es.db"))
        store.initialize()

        store.ensure_initial_epoch(now_iso="2026-07-31T00:00:00")

        rows = store._get_connection().execute("SELECT * FROM epoch_ledger").fetchall()
        assert len(rows) == 1

    def test_second_call_does_not_insert_or_change_the_epoch(self, tmp_path):
        """A second call -- even with different arguments -- must be a
        pure no-op. Varying the arguments (rather than repeating the same
        call) discriminates a real no-op from one that happens to produce
        an identical row because the inputs matched.
        """
        store = EventStore(db_path=str(tmp_path / "es.db"))
        store.initialize()

        first = store.ensure_initial_epoch(
            now_iso="2026-07-31T00:00:00",
            ontology_version="onto-x",
            extraction_version="extract-x",
            model_hash="hash-x",
        )
        second = store.ensure_initial_epoch(
            now_iso="2099-01-01T00:00:00",
            ontology_version="onto-y",
            extraction_version="extract-y",
            model_hash="hash-y",
        )

        assert first == second
        rows = store._get_connection().execute("SELECT * FROM epoch_ledger").fetchall()
        assert len(rows) == 1, "a second call must not add a row"

    def test_default_stamps_come_from_knowledge_config_and_ontology_module(self, tmp_path):
        """A bare call (no override arguments) must not write placeholder
        strings -- it should read the same sources the rest of the system
        treats as the current ontology and extraction config.
        """
        store = EventStore(db_path=str(tmp_path / "es.db"))
        store.initialize()

        epoch = store.ensure_initial_epoch(now_iso="2026-07-31T00:00:00")

        expected_config = KnowledgeConfig.from_env()
        assert epoch["ontology_version"] == ONTOLOGY_V1_0_0.version
        assert epoch["extraction_version"] == expected_config.extraction_version
        assert epoch["model_hash"] == expected_config.model_hash

    def test_does_not_overwrite_an_existing_non_provisional_epoch(self, tmp_path):
        """A genuine epoch (written via `append_epoch`, not this method)
        must not be superseded by a provisional one -- 'ensure a reference
        exists' is not the same job as 'ensure a provisional epoch exists'.
        """
        store = EventStore(db_path=str(tmp_path / "es.db"))
        store.initialize()
        real_epoch_id = store.append_epoch(
            ontology_version="real-onto",
            extraction_version="real-extract",
            model_hash="real-hash",
            activated_at="2026-07-31T00:00:00",
        )

        result = store.ensure_initial_epoch(now_iso="2099-01-01T00:00:00")

        assert result["epoch_id"] == real_epoch_id
        assert result["ontology_version"] == "real-onto"
        assert result["provisional"] == 0
        rows = store._get_connection().execute("SELECT * FROM epoch_ledger").fetchall()
        assert len(rows) == 1

    def test_provisional_column_added_to_preexisting_db(self, tmp_path):
        """A database created before `provisional` existed gains it on open."""
        db = tmp_path / "legacy.db"
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE epoch_ledger ("
            "epoch_id INTEGER PRIMARY KEY AUTOINCREMENT, ontology_version TEXT NOT NULL, "
            "extraction_version TEXT NOT NULL, model_hash TEXT NOT NULL, "
            "activated_at TEXT NOT NULL, prev_epoch_id INTEGER)"
        )
        conn.commit()
        conn.close()

        store = EventStore(db_path=str(db))
        store.initialize()

        cols = {
            row[1] for row in store._get_connection().execute("PRAGMA table_info(epoch_ledger)")
        }
        assert "provisional" in cols

    def test_migration_guard_is_idempotent_across_reopens(self, tmp_path):
        """The guarded ALTER must not raise `duplicate column name` the
        second time a database that already has `provisional` is opened.
        """
        db_path = str(tmp_path / "es.db")
        first = EventStore(db_path=db_path)
        first.initialize()
        first.close()

        second = EventStore(db_path=db_path)
        second.initialize()  # must not raise

        cols = {
            row[1] for row in second._get_connection().execute("PRAGMA table_info(epoch_ledger)")
        }
        assert "provisional" in cols
