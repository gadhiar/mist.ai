"""Reachability tests for the WebSocket session id (P6).

The defect these pin: `server.websocket_endpoint` minted a fresh UUID per
connection and shipped it in the `session_started` handshake, but nothing
carried it inward. `KnowledgeIntegration` held a `current_session_id`
instance attribute initialised to `"default"`, so every turn of every
connection -- across restarts -- landed in one session literally named
`default`.

These tests assert the session id ARRIVES at `ConversationHandler
.handle_message_streaming`, driving the real chain end to end:

    server.websocket_endpoint          (real)
      -> VoiceProcessor.process_complete_audio / _process_conversation_turn
      -> ModelManager.generate_llm_response
      -> KnowledgeIntegration.generate_response_streaming
      -> ConversationHandler.handle_message_streaming     <- assertion point

Only the two genuine I/O boundaries are faked: the WebSocket itself, and
the ConversationHandler that terminates the chain. Everything between them
is production code, so a break anywhere in the plumbing fails these tests
rather than passing vacuously.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import WebSocketDisconnect

from backend.chat.knowledge_integration import KnowledgeIntegration
from backend.chat.stream_events import Complete, Token
from backend.request_context import current_session_id
from backend.voice_models.model_manager import ModelManager
from backend.voice_processor import VoiceProcessor

TRANSCRIBED_TEXT = "hello from the audio path"


class RecordingConversationHandler:
    """Terminal fake: records the session id each turn was handed.

    Stands in for `ConversationHandler`, the I/O-owning collaborator at the
    end of the chain. Records rather than asserts so tests can compare the
    arrived value against the id the handshake actually minted.
    """

    def __init__(self, before_yield: threading.Barrier | None = None) -> None:
        self.session_ids: list[str] = []
        self.ended_sessions: list[str | None] = []
        self.called = threading.Event()
        self._before_yield = before_yield
        self._lock = threading.Lock()

    async def handle_message_streaming(
        self, user_message: str, session_id: str
    ) -> AsyncIterator[Token | Complete]:
        with self._lock:
            self.session_ids.append(session_id)
        self.called.set()
        yield Token(text="ok")
        yield Complete(final_response="ok")

    async def end_session(self, session_id: str | None = None) -> None:
        """Called by the endpoint's disconnect path. Records the id it was scoped to.

        Records rather than asserts because the interesting value is WHICH
        session was ended, and `None` (end everything) must be distinguishable
        from a specific id. The endpoint wraps this call in
        `except Exception: logger.warning`, so a signature mismatch here would
        be swallowed silently rather than failing a test -- which is exactly
        why the assertion is on the recorded value, not on "it did not raise".
        """
        self.ended_sessions.append(session_id)

    def clear_session(self, session_id: str) -> None:
        """Present so `KnowledgeIntegration.clear_session` has a real target."""


class FakeWebSocket:
    """Scripted WebSocket: replays inbound frames, captures outbound ones.

    Raises `WebSocketDisconnect` once the script is exhausted, which is how
    the real endpoint's message loop terminates.
    """

    def __init__(self, inbound: list[dict]) -> None:
        self._inbound = list(inbound)
        self.sent: list[dict] = []

    async def accept(self, headers=None) -> None:
        return None

    async def send_json(self, payload: dict) -> None:
        self.sent.append(payload)

    async def receive_json(self) -> dict:
        if not self._inbound:
            raise WebSocketDisconnect(code=1000)
        return self._inbound.pop(0)

    @property
    def handshake_session_id(self) -> str:
        """The session id the server minted and published to the client."""
        for frame in self.sent:
            if frame.get("type") == "session_started":
                return frame["session_id"]
        raise AssertionError(f"no session_started frame was sent; got {self.sent}")


def build_knowledge_integration(handler: RecordingConversationHandler) -> KnowledgeIntegration:
    """Build a KnowledgeIntegration whose real methods run against a fake handler.

    Bypasses `__init__` deliberately: it constructs a live ConversationHandler
    (Neo4j, LLM, embeddings) and, when that fails, swallows the error and
    leaves `enabled = False` -- which would short-circuit the very method
    under test. `object.__new__` plus explicit attribute assignment keeps
    `generate_response_streaming` itself production code.
    """
    ki = object.__new__(KnowledgeIntegration)
    ki.enabled = True
    ki.conversation_handler = handler
    return ki


def build_model_manager(
    handler: RecordingConversationHandler, event_loop: asyncio.AbstractEventLoop
) -> ModelManager:
    """Build a ModelManager exposing the real `generate_llm_response`.

    `__init__` loads Whisper/TTS and builds a live KnowledgeIntegration, so it
    is bypassed for the same reason as above. `generate_llm_response` is the
    link that passes NO session id downstream -- keeping it real is the point,
    since the propagation must survive it.
    """
    mm = object.__new__(ModelManager)
    mm.knowledge = build_knowledge_integration(handler)
    mm.event_loop = event_loop
    # STT is an I/O boundary; stub it so the audio path needs no Whisper.
    mm.transcribe_audio = lambda audio_data, sample_rate: TRANSCRIBED_TEXT
    return mm


def build_voice_processor(
    handler: RecordingConversationHandler, event_loop: asyncio.AbstractEventLoop
) -> VoiceProcessor:
    """Real VoiceProcessor with TTS off, wired to the recording handler."""
    vp = VoiceProcessor(
        config=SimpleNamespace(tts_enabled=False),
        message_queue=asyncio.Queue(),
    )
    vp.models = build_model_manager(handler, event_loop)
    vp.loop = event_loop
    return vp


async def run_endpoint(monkeypatch, voice_processor: VoiceProcessor, inbound: list[dict]):
    """Drive the real `websocket_endpoint` over a scripted connection."""
    from backend import server

    monkeypatch.setattr(server, "voice_processor", voice_processor)
    monkeypatch.setattr(server, "log_handler", None)

    ws = FakeWebSocket(inbound)
    await server.websocket_endpoint(ws)
    return ws


@pytest.fixture(autouse=True)
def reset_session_context():
    """Keep a leaked context var from one test satisfying the next."""
    token = current_session_id.set(None)
    yield
    current_session_id.reset(token)


class TestDisconnectEndsOnlyItsOwnSession:
    """A disconnect must end THIS connection's session, not every session.

    `end_session(session_id=None)` ends every session the handler tracks. That
    was harmless while all connections shared the single "default" session, but
    giving each connection its own id turned that no-op into a live bug: one
    client disconnecting would write every other open connection's vault note
    early and evict their paths.

    The parameter was already implemented and documented; only the call site
    omitted it. So this is the reachability shape again -- an implemented
    parameter never passed -- and the assertion has to be on the value that
    ARRIVED, because the endpoint swallows exceptions from this call.
    """

    @pytest.mark.asyncio
    async def test_disconnect_scopes_end_session_to_the_handshake_id(self, monkeypatch):
        # Arrange
        handler = RecordingConversationHandler()
        vp = build_voice_processor(handler, asyncio.get_running_loop())

        # Act -- the FakeWebSocket raises WebSocketDisconnect once its script
        # is exhausted, which drives the endpoint's real disconnect path.
        ws = await run_endpoint(monkeypatch, vp, [{"type": "text", "text": "hi"}])

        # Assert
        assert handler.ended_sessions == [ws.handshake_session_id], (
            "disconnect did not scope end_session to this connection's session. "
            f"Got {handler.ended_sessions!r}; None means 'end every tracked "
            "session', which ends other live connections' sessions too."
        )

    @pytest.mark.asyncio
    async def test_disconnect_never_ends_every_session(self, monkeypatch):
        """The specific regression: `None` must never reach `end_session`."""
        handler = RecordingConversationHandler()
        vp = build_voice_processor(handler, asyncio.get_running_loop())

        await run_endpoint(monkeypatch, vp, [{"type": "text", "text": "hi"}])

        assert (
            None not in handler.ended_sessions
        ), "end_session was called with None, which ends EVERY tracked session"


class TestSessionIdReachesHandler:
    """The id minted at the WS handshake must arrive at the handler."""

    @pytest.mark.asyncio
    async def test_text_turn_delivers_handshake_session_id(self, monkeypatch):
        """A `text` frame's turn is handled under the handshake session id."""
        # Arrange
        handler = RecordingConversationHandler()
        vp = build_voice_processor(handler, asyncio.get_running_loop())

        # Act
        ws = await run_endpoint(monkeypatch, vp, [{"type": "text", "text": "hi"}])

        # Assert
        assert handler.session_ids == [ws.handshake_session_id], (
            "session id from the session_started handshake never reached "
            "handle_message_streaming"
        )

    @pytest.mark.asyncio
    async def test_audio_turn_delivers_handshake_session_id(self, monkeypatch):
        """The audio path crosses an extra thread hop and must still deliver.

        `process_complete_audio` hands off to `_process_user_speech` via
        `spawn_with_context`, so this also pins that the context var is set
        BEFORE the context snapshot is taken.
        """
        # Arrange
        handler = RecordingConversationHandler()
        vp = build_voice_processor(handler, asyncio.get_running_loop())
        frame = {
            "type": "audio",
            "audio": np.zeros(16, dtype=np.float32).tolist(),
            "sample_rate": 16000,
        }

        # Act
        ws = await run_endpoint(monkeypatch, vp, [frame])
        # The spawned transcription thread outlives the endpoint; wait off-loop
        # so the bridge's coroutines can still be serviced.
        arrived = await asyncio.to_thread(handler.called.wait, 10.0)

        # Assert
        assert arrived, "audio turn never reached handle_message_streaming"
        assert handler.session_ids == [ws.handshake_session_id]

    @pytest.mark.asyncio
    async def test_separate_connections_do_not_share_one_session(self, monkeypatch):
        """Two connections must produce two distinct session ids at the handler.

        This is the user-visible shape of the defect: every connection, and
        every restart, collapsed into a single session.
        """
        # Arrange
        handler = RecordingConversationHandler()
        vp = build_voice_processor(handler, asyncio.get_running_loop())

        # Act
        first = await run_endpoint(monkeypatch, vp, [{"type": "text", "text": "one"}])
        second = await run_endpoint(monkeypatch, vp, [{"type": "text", "text": "two"}])

        # Assert
        assert first.handshake_session_id != second.handshake_session_id
        assert handler.session_ids == [
            first.handshake_session_id,
            second.handshake_session_id,
        ], "connections collapsed into a shared session"


class TestConcurrentSessionsDoNotClobber:
    """The property instance state could not provide.

    `voice_processor` -- and through it a single `KnowledgeIntegration` -- is
    a module-level singleton shared by every entry in `active_connections`.
    Storing the session on the instance lets a second connection overwrite a
    first one's value between the write and the read.
    """

    def test_two_concurrent_turns_each_keep_their_own_session(self):
        """Both turns set their session id before either reads it.

        A `threading.Barrier` forces the interleave that makes shared mutable
        state fail: with `self.current_session_id = sid`, both reads would
        return whichever thread wrote last. With a context var, each thread
        reads its own.
        """
        # Arrange
        handler = RecordingConversationHandler()
        ki = build_knowledge_integration(handler)

        loop = asyncio.new_event_loop()
        loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
        loop_thread.start()

        barrier = threading.Barrier(2, timeout=10.0)
        observed: dict[str, str] = {}
        errors: list[BaseException] = []

        def turn(session_id: str) -> None:
            try:
                # Exactly what the executor boundary does per turn.
                current_session_id.set(session_id)
                barrier.wait()
                list(ki.generate_response_streaming("hi", event_loop=loop))
                observed[session_id] = current_session_id.get()
            except BaseException as exc:  # noqa: BLE001 -- re-raised in assert
                errors.append(exc)

        # Act
        threads = [
            threading.Thread(target=turn, args=("session-alpha",)),
            threading.Thread(target=turn, args=("session-beta",)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=20.0)

        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=5.0)

        # Assert
        assert not errors, f"worker thread raised: {errors}"
        assert observed == {
            "session-alpha": "session-alpha",
            "session-beta": "session-beta",
        }
        assert sorted(handler.session_ids) == [
            "session-alpha",
            "session-beta",
        ], "concurrent turns clobbered each other's session id at the handler"


class TestMissingSessionIdIsFatal:
    """An unset session id is a wiring defect, not a runtime condition."""

    def test_raises_when_no_session_id_is_available(self):
        """No argument and no context var must fail loudly, not default."""
        # Arrange
        handler = RecordingConversationHandler()
        ki = build_knowledge_integration(handler)
        current_session_id.set(None)

        # Act / Assert
        with pytest.raises(ValueError, match="No session id for this turn"):
            list(ki.generate_response_streaming("hi", event_loop=None))

    def test_does_not_reach_handler_when_session_id_is_missing(self):
        """The guard must stop the turn, not merely log on the way past."""
        # Arrange
        handler = RecordingConversationHandler()
        ki = build_knowledge_integration(handler)
        current_session_id.set(None)

        # Act
        with pytest.raises(ValueError):
            list(ki.generate_response_streaming("hi", event_loop=None))

        # Assert
        assert handler.session_ids == [], "turn reached the handler despite no session id"

    def test_explicit_argument_overrides_the_context_var(self):
        """An explicit session_id wins, so callers can still pin a session."""
        # Arrange
        handler = RecordingConversationHandler()
        ki = build_knowledge_integration(handler)
        current_session_id.set("ambient-session")

        # Act
        list(ki.generate_response_streaming("hi", session_id="explicit-session", event_loop=None))

        # Assert
        assert handler.session_ids == ["explicit-session"]
