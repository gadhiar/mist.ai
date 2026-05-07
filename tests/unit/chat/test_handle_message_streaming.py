"""Tests for ConversationHandler.handle_message_streaming (v1 fake-stream wrapper).

v1 wraps handle_message and yields Token events character-by-character then a
Complete event. Internal pipeline behavior (retrieval, tools, slop filter,
vault, extraction, EventStore) is delegated to handle_message and verified
by its existing test coverage. These tests only validate the streaming
contract: token sequence, terminal Complete event, equivalence to
handle_message's return.

v2 will invert the relationship — handle_message_streaming becomes the
canonical generator and handle_message wraps it. Until then, these tests
serve as the API surface contract.
"""

import pytest

from backend.chat.conversation_handler import ConversationHandler
from backend.chat.stream_events import Complete, StreamEvent, Token
from backend.knowledge.retrieval.knowledge_retriever import KnowledgeRetriever
from backend.knowledge.storage.graph_store import GraphStore
from tests.mocks.config import build_test_config
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection
from tests.mocks.ollama import FakeLLM


def _make_retriever(config, gs):
    return KnowledgeRetriever(config=config, graph_store=gs)


class FakeExtractionPipeline:
    async def extract_from_utterance(self, **kwargs):
        from backend.knowledge.extraction.validator import ValidationResult

        return ValidationResult(valid=True, entities=[], relationships=[])


def _make_handler(default_response: str = "Hello there.") -> ConversationHandler:
    conn = FakeNeo4jConnection()
    gs = GraphStore(conn, FakeEmbeddingGenerator())
    config = build_test_config()
    return ConversationHandler(
        config=config,
        graph_store=gs,
        extraction_pipeline=FakeExtractionPipeline(),
        retriever=_make_retriever(config, gs),
        llm_provider=FakeLLM(default_response=default_response),
    )


class TestStreamingContract:
    @pytest.mark.asyncio
    async def test_yields_at_least_one_event(self):
        handler = _make_handler("Hi.")
        events = []
        async for event in handler.handle_message_streaming(
            user_message="Tell me something useful", session_id="s1"
        ):
            events.append(event)
        assert len(events) >= 1, "stream must yield at least one event"
        assert all(isinstance(e, StreamEvent) for e in events)

    @pytest.mark.asyncio
    async def test_terminates_with_complete(self):
        handler = _make_handler("Hello.")
        events = []
        async for event in handler.handle_message_streaming(
            user_message="Tell me something useful", session_id="s2"
        ):
            events.append(event)
        assert isinstance(
            events[-1], Complete
        ), f"last event must be Complete, got {type(events[-1]).__name__}"

    @pytest.mark.asyncio
    async def test_only_one_complete_event(self):
        handler = _make_handler("Just one response.")
        complete_count = 0
        async for event in handler.handle_message_streaming(
            user_message="Tell me something useful", session_id="s3"
        ):
            if isinstance(event, Complete):
                complete_count += 1
        assert complete_count == 1, "exactly one Complete event must be yielded"

    @pytest.mark.asyncio
    async def test_token_texts_join_to_final_response(self):
        response = "This is the canonical response."
        handler = _make_handler(response)
        token_chars = []
        complete: Complete | None = None
        async for event in handler.handle_message_streaming(
            user_message="Tell me something useful", session_id="s4"
        ):
            if isinstance(event, Token):
                token_chars.append(event.text)
            elif isinstance(event, Complete):
                complete = event
        joined = "".join(token_chars)
        assert complete is not None
        assert joined == complete.final_response, (
            f"token concat ({joined!r}) must equal Complete.final_response "
            f"({complete.final_response!r})"
        )

    @pytest.mark.asyncio
    async def test_streaming_matches_non_streaming_response(self):
        """v1 contract: handle_message_streaming.Complete.final_response equals handle_message return."""
        response = "Identical content path."
        handler_a = _make_handler(response)
        non_streaming = await handler_a.handle_message(
            user_message="Tell me something useful", session_id="sa"
        )
        handler_b = _make_handler(response)
        streamed_complete: Complete | None = None
        async for event in handler_b.handle_message_streaming(
            user_message="Tell me something useful", session_id="sb"
        ):
            if isinstance(event, Complete):
                streamed_complete = event
        assert streamed_complete is not None
        assert streamed_complete.final_response == non_streaming, (
            "handle_message_streaming Complete.final_response must equal "
            "handle_message return for the same input"
        )

    @pytest.mark.asyncio
    async def test_token_count_equals_response_length(self):
        """v1 emits one Token per character; this validates the boundary case."""
        response = "A"  # single char, response.length == 1
        handler = _make_handler(response)
        token_count = 0
        complete: Complete | None = None
        async for event in handler.handle_message_streaming(
            user_message="Tell me something useful", session_id="s5"
        ):
            if isinstance(event, Token):
                token_count += 1
            elif isinstance(event, Complete):
                complete = event
        assert complete is not None
        assert token_count == len(complete.final_response), (
            f"Token count ({token_count}) must equal len(final_response) "
            f"({len(complete.final_response)})"
        )

    @pytest.mark.asyncio
    async def test_complete_carries_duration(self):
        handler = _make_handler("ok.")
        complete: Complete | None = None
        async for event in handler.handle_message_streaming(
            user_message="Tell me something useful", session_id="s6"
        ):
            if isinstance(event, Complete):
                complete = event
        assert complete is not None
        assert complete.duration_ms >= 0, "duration_ms must be non-negative"
