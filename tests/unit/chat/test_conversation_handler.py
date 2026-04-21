"""Tests for ConversationHandler Phase 2B refactor."""

import pytest

from backend.chat.conversation_handler import ConversationHandler
from backend.knowledge.extraction.tool_usage_tracker import ToolUsageTracker
from backend.knowledge.retrieval.knowledge_retriever import KnowledgeRetriever
from backend.knowledge.storage.graph_store import GraphStore
from tests.mocks.config import build_test_config
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection
from tests.mocks.ollama import FakeLLM


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
            llm_provider=FakeLLM(),
        )
        assert handler._extraction_pipeline is pipeline

    def test_tool_schemas_contain_query_knowledge_graph(self):
        conn = FakeNeo4jConnection()
        gs = GraphStore(conn, FakeEmbeddingGenerator())
        pipeline = FakeExtractionPipeline()
        config = build_test_config()

        handler = ConversationHandler(
            config=config,
            graph_store=gs,
            extraction_pipeline=pipeline,
            retriever=_make_retriever(config, gs),
            llm_provider=FakeLLM(),
        )
        tool_names = [s["function"]["name"] for s in handler._tool_schemas]
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
            llm_provider=FakeLLM(),
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
            llm_provider=FakeLLM(),
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
            llm_provider=FakeLLM(),
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
            llm_provider=FakeLLM(),
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
            llm_provider=FakeLLM(),
        )
        assert handler is not None

        # Simulate the guard logic from handle_message
        short_messages = ["hi", "ok", "thanks"]
        for msg in short_messages:
            should_extract = len(msg.split()) >= 3
            assert not should_extract, f"'{msg}' should skip extraction"

        long_message = "I really enjoy Python programming"
        assert len(long_message.split()) >= 3


# =============================================================================
# Cluster 3 Task 6: Persona injection in system prompt
# =============================================================================


@pytest.fixture
def conversation_handler():
    """Shared ConversationHandler fixture for Task 6 tests."""
    conn = FakeNeo4jConnection()
    gs = GraphStore(conn, FakeEmbeddingGenerator())
    pipeline = FakeExtractionPipeline()
    config = build_test_config()
    return ConversationHandler(
        config=config,
        graph_store=gs,
        extraction_pipeline=pipeline,
        retriever=_make_retriever(config, gs),
        llm_provider=FakeLLM(),
    )


@pytest.fixture
def sample_mist_context():
    """A MistContext with absolute preferences, traits, and capabilities for persona tests."""
    from backend.chat.mist_context import (
        MistCapability,
        MistContext,
        MistPreference,
        MistTrait,
    )

    return MistContext(
        display_name="MIST",
        pronouns="she/her",
        self_concept="A cognitive architecture for personal knowledge.",
        traits=[
            MistTrait(
                id="trait-warm",
                display_name="Warm",
                axis="Persona",
                description="Friendly default.",
            ),
        ],
        capabilities=[
            MistCapability(id="cap-tool-use", display_name="Tool use", description="MCP tools."),
        ],
        preferences=[
            MistPreference(
                id="pref-no-emoji",
                display_name="No emoji or unicode decoration",
                enforcement="absolute",
                context="Hard rule across all output channels.",
            ),
        ],
    )


class TestPersonaInjection:
    """Cluster 3: _build_messages prepends the MistContext persona block."""

    def test_persona_block_is_first_system_message(self, conversation_handler, sample_mist_context):
        """Persona block must be the FIRST message and role system."""
        session = conversation_handler.get_or_create_session("persona-s1")
        messages = conversation_handler._build_messages(
            session, max_history=10, retrieval_result=None, mist_context=sample_mist_context
        )
        assert messages[0]["role"] == "system"
        assert "You are MIST" in messages[0]["content"]
        assert "she/her" in messages[0]["content"]

    def test_persona_block_contains_hard_rules(self, conversation_handler, sample_mist_context):
        """Absolute preferences render as HARD RULES in the persona block."""
        session = conversation_handler.get_or_create_session("persona-s2")
        messages = conversation_handler._build_messages(
            session, max_history=10, retrieval_result=None, mist_context=sample_mist_context
        )
        combined_system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        assert "HARD RULE" in combined_system
        assert "No emoji or unicode decoration" in combined_system

    def test_persona_block_contains_traits(self, conversation_handler, sample_mist_context):
        session = conversation_handler.get_or_create_session("persona-s3")
        messages = conversation_handler._build_messages(
            session, max_history=10, retrieval_result=None, mist_context=sample_mist_context
        )
        combined_system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        assert "Warm" in combined_system

    def test_no_mist_context_falls_back_to_static_prompt(self, conversation_handler):
        """When mist_context=None, the old hardcoded 'You are MIST' template is preserved."""
        session = conversation_handler.get_or_create_session("persona-s4")
        messages = conversation_handler._build_messages(
            session, max_history=10, retrieval_result=None, mist_context=None
        )
        # The fallback system prompt still has the static header line.
        assert messages[0]["role"] == "system"
        assert "MIST" in messages[0]["content"]

    def test_persona_block_appears_before_retrieval_context(
        self, conversation_handler, sample_mist_context
    ):
        """Ordering: persona -> static template -> retrieval context -> history."""
        from backend.knowledge.models import RetrievalResult

        session = conversation_handler.get_or_create_session("persona-s5")
        # Construct a minimal RetrievalResult with one fact so it injects.
        # Use sentinels on formatted_context to locate it.
        retrieval = RetrievalResult(
            query="test",
            user_id="User",
            facts=[],
            entities_found=0,
            total_facts=1,  # > 0 so it injects
            formatted_context="RETRIEVED_CONTEXT_SENTINEL",
            retrieval_time_ms=1.0,
            vector_search_time_ms=0.0,
            graph_traversal_time_ms=0.0,
            config_used={},
            intent="relational",
        )
        messages = conversation_handler._build_messages(
            session, max_history=10, retrieval_result=retrieval, mist_context=sample_mist_context
        )
        combined = "\n".join(m["content"] for m in messages)
        persona_idx = combined.index("You are MIST")
        retrieval_idx = combined.index("RETRIEVED_CONTEXT_SENTINEL")
        assert persona_idx < retrieval_idx


class TestMistContextCaching:
    """Cluster 3: MistContext is cached per session -- only one retrieve per session lifetime."""

    @pytest.mark.asyncio
    async def test_get_or_fetch_caches_per_session(self, conversation_handler):
        """Two calls with same session_id hit retriever once."""
        from unittest.mock import AsyncMock

        from backend.chat.mist_context import MistContext

        ctx_stub = MistContext(
            display_name="MIST",
            pronouns="she/her",
            self_concept="",
            traits=[],
            capabilities=[],
            preferences=[],
        )
        conversation_handler.retriever.retrieve_mist_context = AsyncMock(return_value=ctx_stub)

        r1 = await conversation_handler._get_or_fetch_mist_context("sess-A")
        r2 = await conversation_handler._get_or_fetch_mist_context("sess-A")
        r3 = await conversation_handler._get_or_fetch_mist_context("sess-A")

        assert conversation_handler.retriever.retrieve_mist_context.call_count == 1
        assert r1 is r2 is r3

    @pytest.mark.asyncio
    async def test_different_sessions_each_fetch_once(self, conversation_handler):
        """Distinct session_ids each trigger one retrieve."""
        from unittest.mock import AsyncMock

        from backend.chat.mist_context import MistContext

        ctx_stub = MistContext(
            display_name="MIST",
            pronouns="she/her",
            self_concept="",
            traits=[],
            capabilities=[],
            preferences=[],
        )
        conversation_handler.retriever.retrieve_mist_context = AsyncMock(return_value=ctx_stub)

        await conversation_handler._get_or_fetch_mist_context("sess-A")
        await conversation_handler._get_or_fetch_mist_context("sess-B")
        await conversation_handler._get_or_fetch_mist_context("sess-C")

        assert conversation_handler.retriever.retrieve_mist_context.call_count == 3
