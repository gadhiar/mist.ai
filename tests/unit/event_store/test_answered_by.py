"""`ConversationTurnEvent.answered_by`: who answered a turn, recorded with the turn.

A turn a capability answered alone carries `"capability"`, and the memory
writers leave it out (`backend.extraction_backlog.eligibility`). NULL means the
turn was recorded before the column existed, which reads as `"model"`.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime

import pytest

from backend.event_store.models import ANSWERED_BY_VALUES, ConversationTurnEvent
from backend.event_store.store import EventStore

from .test_turn_outcome import _build_base_database

_LOGGER = "backend.event_store.store"
_ADDED = "added `answered_by` column"

# The two tables exactly as `schema.sql` created them before `answered_by` (afb17c07),
# comments included: SQLite's ALTER TABLE edits the stored DDL text, comments and all, so a
# migration test that strips them, or that drops the column from today's table, tests a
# different text. Dropping it is also not portable: the SQLite on CI, dropping a last column,
# cuts from the nearest ',' in the raw text before it, which in today's DDL sits inside the
# `--` comment above `answered_by`, and fails with "incomplete input".
_SESSIONS_TABLE_BEFORE_ANSWERED_BY = """\
CREATE TABLE IF NOT EXISTS conversation_sessions (
    session_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,                -- ISO-8601
    ended_at TEXT,                           -- ISO-8601, NULL if still active
    turn_count INTEGER DEFAULT 0,
    input_modality TEXT DEFAULT 'voice',     -- 'voice', 'text', 'api'
    origin TEXT NOT NULL DEFAULT 'real'      -- 'real', 'test', 'seed'
);
"""

_TURN_TABLE_BEFORE_ANSWERED_BY = """\
CREATE TABLE IF NOT EXISTS conversation_turn_events (
    event_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES conversation_sessions(session_id),
    turn_index INTEGER NOT NULL,
    timestamp TEXT NOT NULL,                 -- ISO-8601

    -- Raw conversation data (NEVER changes)
    user_utterance TEXT NOT NULL,
    system_response TEXT NOT NULL,

    -- Context window (what the LLM actually saw)
    context_window TEXT,                     -- JSON: [{"role": "user", "content": "..."}, ...]

    -- RAG context (what retrieval injected)
    retrieval_context TEXT,                  -- JSON: {"doc_chunks": [...], "graph_facts": [...]}

    -- Tool usage
    tool_calls TEXT,                         -- JSON: [{"name": "...", "args": {...}, "result_summary": "..."}]

    -- Audio reference
    audio_hash TEXT,                         -- SHA-256 of audio file in archive
    audio_format TEXT,                       -- "wav", "opus"
    audio_duration_ms INTEGER,
    audio_sample_rate INTEGER,

    -- Model metadata
    stt_model TEXT,
    tts_model TEXT,
    llm_model TEXT,
    llm_parameters TEXT,                     -- JSON: {"temperature": 0.7, "top_p": 1.0, ...}

    -- Versioning
    ontology_version TEXT NOT NULL DEFAULT '1.0.0',

    -- Immutability timestamp
    created_at TEXT NOT NULL DEFAULT (datetime('now')),

    -- Turn outcome (MIS-127). NULL on every turn logged before these columns existed.
    -- Kept last so a fresh database and one migrated by EventStore.initialize(), whose
    -- ALTER TABLE ADD COLUMN appends, have the same column order.
    turn_outcome TEXT,                       -- 'completed', 'interrupted', 'superseded', 'error'
    generated_response TEXT,                 -- full generated text; longer than system_response
                                             -- when an interrupted turn logs only what was heard
    turn_modality TEXT                       -- 'voice', 'text'
);
"""

_OLD_TURN = """\
INSERT INTO conversation_sessions (session_id, started_at, input_modality)
VALUES ('s1', '2026-09-01T10:00:00+00:00', 'text');
INSERT INTO conversation_turn_events (
    event_id, session_id, turn_index, timestamp, user_utterance, system_response
) VALUES (
    'old-turn', 's1', 0, '2026-09-01T10:00:00+00:00', 'set a timer for ten minutes', 'Timer set.'
);
"""


def _event(**overrides) -> ConversationTurnEvent:
    values = {
        "session_id": "s1",
        "turn_index": 0,
        "timestamp": datetime(2026, 9, 1, 10, 0, tzinfo=UTC),
        "user_utterance": "set a timer for ten minutes",
        "system_response": "Timer set.",
    }
    values.update(overrides)
    return ConversationTurnEvent(**values)


def _columns(store: EventStore) -> list[str]:
    return [
        row[1]
        for row in store._get_connection().execute("PRAGMA table_info(conversation_turn_events)")
    ]


def _store(path) -> EventStore:
    store = EventStore(db_path=str(path))
    store.initialize()
    return store


class TestField:
    def test_values_are_model_and_capability(self):
        assert frozenset({"model", "capability"}) == ANSWERED_BY_VALUES

    def test_defaults_to_none(self):
        assert _event().answered_by is None

    @pytest.mark.parametrize("value", ["model", "capability"])
    def test_known_values_construct(self, value):
        assert _event(answered_by=value).answered_by == value

    @pytest.mark.parametrize("value", ["", "Model", "tool", "voice"])
    def test_an_unknown_value_raises_at_construction(self, value):
        with pytest.raises(ValueError, match="answered_by"):
            _event(answered_by=value)

    def test_to_dict_carries_it(self):
        assert _event(answered_by="capability").to_dict()["answered_by"] == "capability"
        assert _event().to_dict()["answered_by"] is None

    def test_from_row_reads_it_and_a_row_without_it(self):
        row = _event(answered_by="capability").to_dict()

        assert ConversationTurnEvent.from_row(row).answered_by == "capability"
        row.pop("answered_by")
        assert ConversationTurnEvent.from_row(row).answered_by is None

    def test_from_row_refuses_an_unknown_stored_value(self):
        row = _event().to_dict()
        row["answered_by"] = "oracle"

        with pytest.raises(ValueError, match="answered_by"):
            ConversationTurnEvent.from_row(row)


class TestRoundTrip:
    @pytest.mark.parametrize("value", ["model", "capability", None])
    def test_append_turn_get_turn_and_from_row(self, tmp_path, value):
        store = _store(tmp_path / "es.db")
        store.start_session("s1", input_modality="text")
        event_id = store.append_turn(_event(answered_by=value))

        row = store.get_turn(event_id)

        assert row is not None
        assert row["answered_by"] == value
        assert ConversationTurnEvent.from_row(row).answered_by == value
        assert store.get_turns("s1")[0]["answered_by"] == value

    def test_replay_keys_carry_it_in_replay_order(self, tmp_path):
        store = _store(tmp_path / "es.db")
        store.start_session("s1", input_modality="text")
        store.append_turn(
            _event(
                turn_index=1,
                timestamp=datetime(2026, 9, 1, 10, 5, tzinfo=UTC),
                answered_by="model",
                event_id="later",
            )
        )
        store.append_turn(_event(answered_by="capability", event_id="earlier"))

        keys = store.list_turn_keys_in_replay_order()

        assert [(k["event_id"], k["answered_by"]) for k in keys] == [
            ("earlier", "capability"),
            ("later", "model"),
        ]


class TestMigration:
    def _old_database(self, path) -> str:
        """A database whose turn table predates `answered_by`, holding one turn."""
        conn = sqlite3.connect(path)
        try:
            conn.executescript(
                _SESSIONS_TABLE_BEFORE_ANSWERED_BY + _TURN_TABLE_BEFORE_ANSWERED_BY + _OLD_TURN
            )
        finally:
            conn.close()
        return "old-turn"

    def test_the_old_database_has_no_answered_by_column(self, tmp_path):
        path = tmp_path / "old.db"
        self._old_database(path)
        conn = sqlite3.connect(path)
        try:
            columns = [r[1] for r in conn.execute("PRAGMA table_info(conversation_turn_events)")]
        finally:
            conn.close()

        assert "answered_by" not in columns
        assert columns[-1] == "turn_modality"

    def test_initialize_adds_the_column_and_leaves_old_rows_null(self, tmp_path, caplog):
        path = tmp_path / "old.db"
        event_id = self._old_database(path)

        store = EventStore(db_path=str(path))
        with caplog.at_level(logging.INFO, logger=_LOGGER):
            store.initialize()

        assert "answered_by" in _columns(store)
        row = store.get_turn(event_id)
        assert row is not None
        assert row["answered_by"] is None
        assert ConversationTurnEvent.from_row(row).answered_by is None
        assert sum(_ADDED in r.getMessage() for r in caplog.records) == 1

    def test_a_second_initialize_does_not_add_or_log_again(self, tmp_path, caplog):
        path = tmp_path / "old.db"
        self._old_database(path)
        first = EventStore(db_path=str(path))
        first.initialize()
        first.close()

        second = EventStore(db_path=str(path))
        with caplog.at_level(logging.INFO, logger=_LOGGER):
            second.initialize()

        assert not any(_ADDED in r.getMessage() for r in caplog.records)
        assert _columns(second).count("answered_by") == 1

    def test_a_fresh_database_does_not_log_the_migration(self, tmp_path, caplog):
        store = EventStore(db_path=str(tmp_path / "fresh.db"))
        with caplog.at_level(logging.INFO, logger=_LOGGER):
            store.initialize()

        assert not any(_ADDED in r.getMessage() for r in caplog.records)

    def test_fresh_and_migrated_databases_list_the_same_columns_in_order(self, tmp_path):
        old = tmp_path / "old.db"
        self._old_database(old)
        migrated = _store(old)
        fresh = _store(tmp_path / "fresh.db")

        assert _columns(migrated) == _columns(fresh)
        assert _columns(fresh)[-1] == "answered_by"

    def test_a_migrated_database_records_new_turns_with_the_field(self, tmp_path):
        old = tmp_path / "old.db"
        self._old_database(old)
        store = _store(old)

        event_id = store.append_turn(_event(turn_index=1, answered_by="capability"))

        row = store.get_turn(event_id)
        assert row is not None
        assert row["answered_by"] == "capability"

    def test_a_database_from_before_the_turn_outcome_columns_gains_all_four_in_order(
        self, tmp_path
    ):
        old = tmp_path / "pre-mis-127.db"
        _build_base_database(old)
        migrated = _store(old)
        fresh = _store(tmp_path / "fresh.db")

        assert _columns(migrated) == _columns(fresh)
        assert _columns(migrated)[-4:] == [
            "turn_outcome",
            "generated_response",
            "turn_modality",
            "answered_by",
        ]
