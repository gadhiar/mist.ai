"""Turn outcome, generated text and modality on `conversation_turn_events` (MIS-127).

An interrupted voice turn is logged as the part the user heard, with the full generated text
kept beside it, so the log keeps both what was said and what was produced. These tests cover the
schema migration of an existing database, the write and read path, and construction-time
validation of the two enumerated fields.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import pytest

from backend.event_store.models import ConversationTurnEvent
from backend.event_store.store import EventStore

_NEW_COLUMNS = ("turn_outcome", "generated_response", "turn_modality")

# The two tables as `schema.sql` created them before this change. Kept verbatim (comments
# aside) so the migration test exercises a database that really predates the new columns,
# rather than one built from today's schema with columns dropped.
_BASE_SCHEMA = """
CREATE TABLE conversation_sessions (
    session_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    turn_count INTEGER DEFAULT 0,
    input_modality TEXT DEFAULT 'voice',
    origin TEXT NOT NULL DEFAULT 'real'
);
CREATE TABLE conversation_turn_events (
    event_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES conversation_sessions(session_id),
    turn_index INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    user_utterance TEXT NOT NULL,
    system_response TEXT NOT NULL,
    context_window TEXT,
    retrieval_context TEXT,
    tool_calls TEXT,
    audio_hash TEXT,
    audio_format TEXT,
    audio_duration_ms INTEGER,
    audio_sample_rate INTEGER,
    stt_model TEXT,
    tts_model TEXT,
    llm_model TEXT,
    llm_parameters TEXT,
    ontology_version TEXT NOT NULL DEFAULT '1.0.0',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
INSERT INTO conversation_sessions (session_id, started_at)
VALUES ('legacy', '2026-09-01T10:00:00+00:00');
INSERT INTO conversation_turn_events (
    event_id, session_id, turn_index, timestamp, user_utterance, system_response
) VALUES ('e-legacy', 'legacy', 0, '2026-09-01T10:00:01+00:00', 'hello', 'hi');
"""


def _build_base_database(path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(_BASE_SCHEMA)
    finally:
        conn.close()


def _turn_columns(store: EventStore) -> list[str]:
    conn = store._get_connection()
    return [row[1] for row in conn.execute("PRAGMA table_info(conversation_turn_events)")]


def _event(**overrides) -> ConversationTurnEvent:
    fields = {
        "session_id": "s",
        "turn_index": 0,
        "timestamp": datetime(2026, 10, 1, 12, 0, tzinfo=UTC),
        "user_utterance": "tell me about the trip",
        "system_response": "The trip starts on",
    }
    fields.update(overrides)
    return ConversationTurnEvent(**fields)


class TestMigration:
    def test_a_base_database_gains_the_three_columns(self, tmp_path):
        db = tmp_path / "events.db"
        _build_base_database(db)

        store = EventStore(db_path=str(db))
        store.initialize()

        conn = store._get_connection()
        pragma = conn.execute("PRAGMA table_info(conversation_turn_events)")
        info = {row[1]: row for row in pragma}
        for name in _NEW_COLUMNS:
            assert name in info
            # PRAGMA table_info: (cid, name, type, notnull, dflt_value, pk)
            assert info[name][2] == "TEXT"
            assert info[name][3] == 0
            assert info[name][4] is None

    def test_initializing_twice_is_a_no_op(self, tmp_path):
        db = tmp_path / "events.db"
        _build_base_database(db)
        store = EventStore(db_path=str(db))
        store.initialize()
        before = _turn_columns(store)

        store.initialize()

        assert _turn_columns(store) == before
        assert [c for c in before if c in _NEW_COLUMNS] == list(_NEW_COLUMNS)

    def test_a_fresh_database_has_the_three_columns(self):
        store = EventStore(db_path=":memory:")
        store.initialize()

        columns = _turn_columns(store)

        for name in _NEW_COLUMNS:
            assert name in columns


class TestWriteAndRead:
    def test_append_turn_writes_the_three_fields_and_a_read_returns_them(self):
        store = EventStore(db_path=":memory:")
        store.initialize()
        store.start_session("s", input_modality="voice", origin="test")
        event = _event(
            turn_outcome="interrupted",
            generated_response="The trip starts on Monday and ends on Friday.",
            turn_modality="voice",
        )

        store.append_turn(event)

        (row,) = store.get_turns("s")
        assert row["turn_outcome"] == "interrupted"
        assert row["generated_response"] == "The trip starts on Monday and ends on Friday."
        assert row["turn_modality"] == "voice"
        assert row["system_response"] == "The trip starts on"
        restored = ConversationTurnEvent.from_row(row)
        assert restored.turn_outcome == "interrupted"
        assert restored.generated_response == "The trip starts on Monday and ends on Friday."
        assert restored.turn_modality == "voice"

    def test_a_turn_written_without_the_fields_stores_null(self):
        store = EventStore(db_path=":memory:")
        store.initialize()
        store.start_session("s", input_modality="text", origin="test")

        store.append_turn(_event())

        (row,) = store.get_turns("s")
        assert (row["turn_outcome"], row["generated_response"], row["turn_modality"]) == (
            None,
            None,
            None,
        )

    def test_a_legacy_row_reads_none_for_each_field(self, tmp_path):
        db = tmp_path / "events.db"
        _build_base_database(db)
        store = EventStore(db_path=str(db))
        store.initialize()

        (row,) = store.get_turns("legacy")
        restored = ConversationTurnEvent.from_row(row)

        for name in _NEW_COLUMNS:
            assert row[name] is None
            assert getattr(restored, name) is None


class TestValidation:
    def test_defaults_are_none(self):
        event = _event()

        assert (event.turn_outcome, event.generated_response, event.turn_modality) == (
            None,
            None,
            None,
        )

    @pytest.mark.parametrize("outcome", ["completed", "interrupted", "superseded", "error"])
    def test_each_known_outcome_is_accepted(self, outcome):
        assert _event(turn_outcome=outcome).turn_outcome == outcome

    @pytest.mark.parametrize("modality", ["voice", "text"])
    def test_each_known_modality_is_accepted(self, modality):
        assert _event(turn_modality=modality).turn_modality == modality

    def test_an_unknown_turn_outcome_raises_value_error(self):
        with pytest.raises(ValueError, match="turn_outcome"):
            _event(turn_outcome="cancelled")

    def test_an_unknown_turn_modality_raises_value_error(self):
        with pytest.raises(ValueError, match="turn_modality"):
            _event(turn_modality="api")
