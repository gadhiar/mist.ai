"""Unit tests for EventStore.

Uses real in-memory SQLite (:memory:) -- fast enough and avoids
fake/real divergence.
"""

from datetime import UTC, datetime

import pytest

from backend.event_store.models import ConversationTurnEvent
from backend.event_store.store import EventStore


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
    def test_start_session_returns_session_id(self, store: EventStore):
        session_id = store.start_session()

        assert isinstance(session_id, str)
        assert len(session_id) == 36  # UUID format

    def test_end_session_updates_ended_at(self, store: EventStore):
        session_id = store.start_session()

        store.end_session(session_id)

        session = store.get_session(session_id)
        assert session is not None
        assert session.ended_at is not None

    def test_get_session_returns_session_data(self, store: EventStore):
        session_id = store.start_session(input_modality="text")

        session = store.get_session(session_id)

        assert session is not None
        assert session.session_id == session_id
        assert session.input_modality == "text"
        assert session.turn_count == 0
        assert session.started_at is not None
        assert session.ended_at is None


class TestTurnEvents:
    def test_append_turn_stores_event(self, store: EventStore):
        session_id = store.start_session()
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
        session_id = store.start_session()

        for i in range(3):
            event = _build_turn_event(session_id=session_id, turn_index=i)
            store.append_turn(event)

        turns = store.get_turns(session_id)

        assert len(turns) == 3
        assert turns[0]["turn_index"] == 0
        assert turns[1]["turn_index"] == 1
        assert turns[2]["turn_index"] == 2

    def test_get_turns_returns_all_turns_for_session(self, store: EventStore):
        session_id = store.start_session()
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
