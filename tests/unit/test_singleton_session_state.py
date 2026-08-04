"""Per-session state must not live on the process-wide VoiceProcessor (T0).

`voice_processor` is a module-level singleton (`backend/server.py:75`) shared
by every entry in `active_connections`. Any instance attribute on it that is
semantically per-session is invisible while there is one connection and
becomes a data-crossing bug the moment there are two.

The member pinned here is `VoiceProcessor.latest_user_input`
(`backend/voice_processor.py:109`), the "latest-wins" pending-input slot:

  1. Session A's turn holds `generation_lock`.
  2. Session B sends text. `_process_conversation_turn` fails the non-blocking
     acquire, stashes the text in the shared slot, and returns.
  3. Session A's `finally` drains the slot and respawns the turn via
     `spawn_with_context`, which snapshots session A's context -- so session
     B's message is handled under session A's `current_session_id`.

The consequence is not a cosmetic mislabel: the session id selects the
conversation history, the EventStore session, the vault session note and the
graph provenance. Session B's utterance is recorded as session A's.

These tests drive the real `_process_conversation_turn` on both sides and
assert on the pair that ARRIVED at the handler, because the text alone
arrives correctly either way -- only the session id crosses.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest

from backend.chat.knowledge_integration import KnowledgeIntegration
from backend.chat.stream_events import Complete, Token
from backend.request_context import current_session_id
from backend.voice_models.model_manager import ModelManager
from backend.voice_processor import VoiceProcessor

TURN_TIMEOUT = 15.0


class GatedConversationHandler:
    """Terminal fake recording `(user_message, session_id)` per turn.

    The first turn parks inside `handle_message_streaming` until `release` is
    set, which is what holds `generation_lock` long enough for a second
    session's turn to lose the non-blocking acquire and take the pending-input
    path. Recording the pair rather than asserting keeps the failure message
    able to show which session the message was misattributed to.
    """

    def __init__(self) -> None:
        self.turns: list[tuple[str, str]] = []
        self.first_turn_entered = threading.Event()
        self.second_turn_recorded = threading.Event()
        self.release = threading.Event()
        self._lock = threading.Lock()

    async def handle_message_streaming(
        self, user_message: str, session_id: str
    ) -> AsyncIterator[Token | Complete]:
        with self._lock:
            self.turns.append((user_message, session_id))
            turn_number = len(self.turns)

        if turn_number == 1:
            self.first_turn_entered.set()
            # Park off-loop: this coroutine runs ON the bridge's event loop,
            # so a synchronous wait here would deadlock the second turn's
            # own bridge rather than merely delaying it.
            await asyncio.to_thread(self.release.wait, TURN_TIMEOUT)
        else:
            self.second_turn_recorded.set()

        yield Token(text="ok")
        yield Complete(final_response="ok")

    async def end_session(self, session_id: str | None = None) -> None:
        """Present so the endpoint's disconnect path has a real target."""

    def clear_session(self, session_id: str) -> None:
        """Present so `KnowledgeIntegration.clear_session` has a real target."""


def build_voice_processor(
    handler: GatedConversationHandler, event_loop: asyncio.AbstractEventLoop
) -> VoiceProcessor:
    """Real VoiceProcessor, TTS off, wired to the gated handler.

    `KnowledgeIntegration.__init__` and `ModelManager.__init__` construct live
    Neo4j / LLM / Whisper collaborators and swallow failures into a disabled
    state, which would short-circuit the code under test. Both are bypassed
    with `object.__new__` so every method on the path -- including
    `_process_conversation_turn`, `generate_llm_response` and
    `generate_response_streaming` -- stays production code.
    """
    ki = object.__new__(KnowledgeIntegration)
    ki.enabled = True
    ki.conversation_handler = handler
    ki.last_complete = None
    ki.last_error = None

    mm = object.__new__(ModelManager)
    mm.knowledge = ki
    mm.event_loop = event_loop

    vp = VoiceProcessor(
        config=SimpleNamespace(tts_enabled=False),
        message_queue=asyncio.Queue(),
    )
    vp.models = mm
    vp.loop = event_loop
    return vp


@pytest.fixture
def bridge_loop():
    """A background event loop for the sync->async bridge to run turns on."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5.0)
    loop.close()


@pytest.fixture(autouse=True)
def reset_session_context():
    """Keep a leaked context var from one test satisfying the next."""
    token = current_session_id.set(None)
    yield
    current_session_id.reset(token)


def run_interleaved_turns(
    vp: VoiceProcessor, handler: GatedConversationHandler
) -> list[BaseException]:
    """Drive the contended-lock interleave; return worker-thread exceptions.

    Ordering is enforced by events rather than sleeps: the second turn is not
    dispatched until the first is provably parked inside the handler (and so
    provably holding `generation_lock`), and the first is not released until
    the second has provably taken the pending-input path.
    """
    errors: list[BaseException] = []

    def turn(user_text: str, session_id: str) -> None:
        try:
            vp._process_conversation_turn(user_text, session_id=session_id)
        except BaseException as exc:  # noqa: BLE001 -- surfaced via assert
            errors.append(exc)

    alpha = threading.Thread(target=turn, args=("alpha message", "session-alpha"))
    alpha.start()

    assert handler.first_turn_entered.wait(TURN_TIMEOUT), "first turn never reached the handler"

    # Session beta loses the non-blocking acquire and stashes its text.
    beta = threading.Thread(target=turn, args=("beta message", "session-beta"))
    beta.start()
    beta.join(timeout=TURN_TIMEOUT)
    assert not beta.is_alive(), "queued turn should return immediately, not block"

    # Alpha's `finally` now drains the pending slot and respawns the turn.
    handler.release.set()
    handler.second_turn_recorded.wait(TURN_TIMEOUT)
    alpha.join(timeout=TURN_TIMEOUT)

    return errors


class TestPendingInputKeepsItsOwnSession:
    """A queued turn must run under the session that submitted it."""

    def test_queued_turn_is_handled_under_its_own_session(self, bridge_loop):
        # Arrange
        handler = GatedConversationHandler()
        vp = build_voice_processor(handler, bridge_loop)

        # Act
        errors = run_interleaved_turns(vp, handler)

        # Assert
        assert not errors, f"worker thread raised: {errors}"
        assert len(handler.turns) == 2, f"expected two turns at the handler, got {handler.turns!r}"
        assert handler.turns[1] == ("beta message", "session-beta"), (
            "the queued turn was handled under the wrong session. Got "
            f"{handler.turns[1]!r}. The pending-input slot carries only the "
            "text, so the respawn inherits the PREVIOUS turn's "
            "current_session_id -- session beta's message is recorded into "
            "session alpha's history, EventStore session and vault note."
        )

    def test_neither_session_absorbs_the_other_message(self, bridge_loop):
        """The property, stated over both turns rather than one index."""
        # Arrange
        handler = GatedConversationHandler()
        vp = build_voice_processor(handler, bridge_loop)

        # Act
        errors = run_interleaved_turns(vp, handler)

        # Assert
        assert not errors, f"worker thread raised: {errors}"
        # session id -> message handled under it
        by_session = {session_id: message for message, session_id in handler.turns}
        assert by_session == {
            "session-alpha": "alpha message",
            "session-beta": "beta message",
        }, f"interleaved sessions crossed: {handler.turns!r}"
