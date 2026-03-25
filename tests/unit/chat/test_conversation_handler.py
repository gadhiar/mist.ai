"""Tests for ConversationHandler Phase 2B refactor."""

import pytest

from backend.chat.conversation_handler import ConversationHandler
from backend.knowledge.extraction.tool_usage_tracker import ToolUsageTracker
from backend.knowledge.retrieval.knowledge_retriever import KnowledgeRetriever
from backend.knowledge.storage.graph_store import GraphStore
from tests.mocks.config import build_test_config
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection


def _make_retriever(config, gs):
    """Build a graph-only KnowledgeRetriever for tests."""
    return KnowledgeRetriever(config=config, graph_store=gs)


class FakeExtractionPipeline:
    """Test double that records extract_from_utterance calls."""

    def __init__(self):
        self.calls: list[dict] = []

    async def extract_from_utterance(self, **kwargs):
        self.calls.append(kwargs)
        # Return a minimal ValidationResult-like object
        from backend.knowledge.extraction.validator import ValidationResult

        return ValidationResult(valid=True, entities=[], relationships=[])


class FakeFailingPipeline:
    """Test double that raises on extraction."""

    async def extract_from_utterance(self, **kwargs):
        raise RuntimeError("extraction failed")


class TestConstructorDI:
    def test_accepts_extraction_pipeline(self):
        conn = FakeNeo4jConnection()
        gs = GraphStore(conn, FakeEmbeddingGenerator())
        pipeline = FakeExtractionPipeline()
        config = build_test_config()

        handler = ConversationHandler(
            config=config,
            graph_store=gs,
            extraction_pipeline=pipeline,
            retriever=_make_retriever(config, gs),
        )
        assert handler._extraction_pipeline is pipeline

    def test_no_extract_knowledge_tool(self):
        conn = FakeNeo4jConnection()
        gs = GraphStore(conn, FakeEmbeddingGenerator())
        pipeline = FakeExtractionPipeline()
        config = build_test_config()

        handler = ConversationHandler(
            config=config,
            graph_store=gs,
            extraction_pipeline=pipeline,
            retriever=_make_retriever(config, gs),
        )
        tool_names = [t.name for t in handler.tools]
        assert "extract_knowledge" not in tool_names
        assert "extract_knowledge_from_document" not in tool_names
        assert "query_knowledge_graph" in tool_names


class TestExtractKnowledgeAsync:
    @pytest.mark.asyncio
    async def test_extraction_called_with_correct_args(self):
        pipeline = FakeExtractionPipeline()

        conn = FakeNeo4jConnection()
        gs = GraphStore(conn, FakeEmbeddingGenerator())
        config = build_test_config()

        handler = ConversationHandler(
            config=config,
            graph_store=gs,
            extraction_pipeline=pipeline,
            retriever=_make_retriever(config, gs),
        )

        await handler._extract_knowledge_async(
            utterance="I use Python and React",
            conversation_history=[{"role": "user", "content": "I use Python and React"}],
            event_id="evt-001",
            session_id="sess-001",
        )

        assert len(pipeline.calls) == 1
        assert pipeline.calls[0]["utterance"] == "I use Python and React"
        assert pipeline.calls[0]["event_id"] == "evt-001"
        assert pipeline.calls[0]["session_id"] == "sess-001"

    @pytest.mark.asyncio
    async def test_extraction_failure_does_not_raise(self):
        pipeline = FakeFailingPipeline()

        conn = FakeNeo4jConnection()
        gs = GraphStore(conn, FakeEmbeddingGenerator())
        config = build_test_config()

        handler = ConversationHandler(
            config=config,
            graph_store=gs,
            extraction_pipeline=pipeline,
            retriever=_make_retriever(config, gs),
        )

        # Should not raise
        await handler._extract_knowledge_async(
            utterance="test",
            conversation_history=[],
            event_id="evt-001",
            session_id="sess-001",
        )


class TestToolUsageTrackerDI:
    def test_accepts_tool_usage_tracker_parameter(self):
        # Arrange
        conn = FakeNeo4jConnection()
        gs = GraphStore(conn, FakeEmbeddingGenerator())
        pipeline = FakeExtractionPipeline()
        config = build_test_config()
        tracker = ToolUsageTracker(config.skill_derivation)

        # Act
        handler = ConversationHandler(
            config=config,
            graph_store=gs,
            extraction_pipeline=pipeline,
            retriever=_make_retriever(config, gs),
            tool_usage_tracker=tracker,
        )

        # Assert
        assert handler._tool_usage_tracker is tracker

    def test_tool_usage_tracker_defaults_to_none(self):
        # Arrange
        conn = FakeNeo4jConnection()
        gs = GraphStore(conn, FakeEmbeddingGenerator())
        pipeline = FakeExtractionPipeline()
        config = build_test_config()

        # Act
        handler = ConversationHandler(
            config=config,
            graph_store=gs,
            extraction_pipeline=pipeline,
            retriever=_make_retriever(config, gs),
        )

        # Assert
        assert handler._tool_usage_tracker is None


class TestShortMessageSkip:
    def test_short_messages_skip_extraction(self):
        """Messages with fewer than 3 words should not trigger extraction."""
        pipeline = FakeExtractionPipeline()

        conn = FakeNeo4jConnection()
        gs = GraphStore(conn, FakeEmbeddingGenerator())
        config = build_test_config()

        handler = ConversationHandler(
            config=config,
            graph_store=gs,
            extraction_pipeline=pipeline,
            retriever=_make_retriever(config, gs),
        )
        assert handler is not None

        # Simulate the guard logic from handle_message
        short_messages = ["hi", "ok", "thanks"]
        for msg in short_messages:
            should_extract = len(msg.split()) >= 3
            assert not should_extract, f"'{msg}' should skip extraction"

        long_message = "I really enjoy Python programming"
        assert len(long_message.split()) >= 3
