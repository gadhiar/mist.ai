"""T3 acceptance test: the sync bridge preserves real-time streaming.

`KnowledgeIntegration.generate_response_streaming` bridges the async
`ConversationHandler.handle_message_streaming` generator to the sync
iteration the voice pipeline's worker thread needs
(backend/chat/knowledge_integration.py). This test proves that bridge
does not buffer: a sentence gated and yielded early by pass 1 reaches the
sync consumer before a later, event-gated sentence is even produced by
the fake provider -- the same "first token before completion" contract
`tests/unit/chat/test_handle_message_streaming.py` pins directly on
`handle_message_streaming`, exercised end-to-end through the thread/queue
bridge instead.
"""

from __future__ import annotations

import asyncio
import threading

from backend.chat.conversation_handler import ConversationHandler
from backend.chat.knowledge_integration import KnowledgeIntegration
from backend.errors import LLMConnectionError
from backend.knowledge.retrieval.knowledge_retriever import KnowledgeRetriever
from backend.knowledge.storage.graph_store import GraphStore
from backend.request_context import current_session_id, current_turn_complete, current_turn_error
from tests.mocks.config import build_test_config
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection
from tests.mocks.streaming_llm import FakeStreamingLLMProvider, ScriptedPass
from tests.unit.conftest import make_test_conventions_loader


class FakeExtractionPipeline:
    async def extract_from_utterance(self, **kwargs):
        from backend.knowledge.extraction.validator import ValidationResult

        return ValidationResult(valid=True, entities=[], relationships=[])


async def _set_event(event: asyncio.Event) -> None:
    event.set()


def test_first_token_reaches_sync_bridge_before_gated_sentence():
    """`generate_response_streaming`'s thread/queue bridge must forward the
    first gated sentence as soon as `handle_message_streaming` yields it,
    not after the whole turn (and its event-gated second sentence) has
    finished streaming.
    """
    conn = FakeNeo4jConnection()
    gs = GraphStore(conn, FakeEmbeddingGenerator())
    config = build_test_config()

    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()

    try:
        # The gate Event is only ever awaited/set by coroutines scheduled on
        # `loop` (both via run_coroutine_threadsafe below), so it is safe to
        # construct here despite `asyncio.new_event_loop()` not being the
        # thread-local running loop at construction time.
        gate_event = asyncio.Event()
        provider = FakeStreamingLLMProvider(
            [
                ScriptedPass(
                    chunks=[
                        "This is the first sentence. ",
                        gate_event,
                        "This is the second sentence.",
                    ]
                )
            ]
        )
        retriever = KnowledgeRetriever(config=config, graph_store=gs)
        handler = ConversationHandler(
            config=config,
            graph_store=gs,
            extraction_pipeline=FakeExtractionPipeline(),
            retriever=retriever,
            llm_provider=provider,
            conventions_loader=make_test_conventions_loader(),
        )

        # Bypass KnowledgeIntegration.__init__ (which would build a second,
        # real ConversationHandler via build_conversation_handler): wire the
        # already-built fake-backed handler directly, matching the pattern
        # in tests/unit/test_singleton_session_state.py.
        ki = object.__new__(KnowledgeIntegration)
        ki.enabled = True
        ki.conversation_handler = handler

        token = current_session_id.set("bridge-s1")
        try:
            gen = ki.generate_response_streaming("hi", session_id="bridge-s1", event_loop=loop)

            first_item = next(gen)
            while not isinstance(first_item, str):
                first_item = next(gen)

            assert first_item.strip() == "This is the first sentence."
            # Reaching this assertion at all proves the bridge delivered the
            # first sentence without waiting for the second -- if it had
            # buffered until end-of-turn, `next(gen)` above would still be
            # blocked on the queue, because nothing sets `gate_event` until
            # the lines below run.
            assert not gate_event.is_set()

            asyncio.run_coroutine_threadsafe(_set_event(gate_event), loop).result(timeout=5)

            remaining = list(gen)
            assert "second sentence" in "".join(item for item in remaining if isinstance(item, str))
        finally:
            current_session_id.reset(token)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=5)
        loop.close()


def test_error_turn_text_reaches_sync_bridge_and_matches_record():
    """Review finding 1, end to end: when the provider raises, the error text
    the handler records must be what the bridge forwards to the socket.

    The bridge forwards only `Token` events as `str` items (WSEvents go out as
    dicts, `Complete` only onto `current_turn_complete`), so before the fix --
    when the except path yielded no Token -- the consumer received nothing
    while session history and the event log recorded "I encountered an
    error: ...".
    """
    conn = FakeNeo4jConnection()
    gs = GraphStore(conn, FakeEmbeddingGenerator())
    config = build_test_config()

    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()

    try:
        provider = FakeStreamingLLMProvider(
            [ScriptedPass(chunks=[], error=LLMConnectionError("llama-server unreachable"))]
        )
        handler = ConversationHandler(
            config=config,
            graph_store=gs,
            extraction_pipeline=FakeExtractionPipeline(),
            retriever=KnowledgeRetriever(config=config, graph_store=gs),
            llm_provider=provider,
            conventions_loader=make_test_conventions_loader(),
        )
        recorded: list[str] = []
        original_record = handler._record_turn_event

        def _capture(*args, **kwargs):
            recorded.append(kwargs["assistant_message"])
            return original_record(*args, **kwargs)

        handler._record_turn_event = _capture

        ki = object.__new__(KnowledgeIntegration)
        ki.enabled = True
        ki.conversation_handler = handler

        token = current_session_id.set("bridge-err")
        try:
            items = list(
                ki.generate_response_streaming("hi there", session_id="bridge-err", event_loop=loop)
            )
            streamed = "".join(item for item in items if isinstance(item, str))
            complete = current_turn_complete.get()
            error = current_turn_error.get()
        finally:
            current_session_id.reset(token)

        assert streamed == "I encountered an error: llama-server unreachable"
        assert complete is not None
        assert complete.final_response == streamed
        assert recorded == [streamed]
        assert handler.sessions["bridge-err"].messages[-1].content == streamed
        # The handler recovered the turn itself; the bridge's side-channel
        # error is for a stream that failed to produce a Complete at all.
        assert error is None
    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=5)
        loop.close()
