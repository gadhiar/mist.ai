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

_LOGGER = "backend.event_store.store"
_ADDED = "added `answered_by` column"


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
        store = _store(path)
        store.start_session("s1", input_modality="text")
        event_id = store.append_turn(_event(event_id="old-turn"))
        store.close()
        conn = sqlite3.connect(path)
        conn.execute("ALTER TABLE conversation_turn_events DROP COLUMN answered_by")
        conn.commit()
        conn.close()
        return event_id

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
