"""Event store test fixtures and factory functions."""

from datetime import UTC, datetime

from tests.mocks.config import TEST_EVENT_ID, TEST_SESSION_ID


def make_turn_event(
    *,
    event_id: str = TEST_EVENT_ID,
    session_id: str = TEST_SESSION_ID,
    turn_index: int = 0,
    user_utterance: str = "test input",
    system_response: str = "test response",
    timestamp: datetime | None = None,
    llm_model: str = "test-model",
) -> dict:
    """Build a conversation turn event dict with sensible defaults.

    Returns a plain dict matching ConversationTurnEvent fields.
    Keyword-only args for type safety.
    """
    return {
        "event_id": event_id,
        "session_id": session_id,
        "turn_index": turn_index,
        "timestamp": timestamp or datetime.now(UTC),
        "user_utterance": user_utterance,
        "system_response": system_response,
        "llm_model": llm_model,
    }
