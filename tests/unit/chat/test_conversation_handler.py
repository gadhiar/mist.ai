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
    # The word-count threshold is: len(user_message.split()) >= 3.
    # Messages with fewer than 3 words skip both auto-RAG retrieval AND
    # background extraction scheduling in handle_message.

    @pytest.mark.asyncio
    async def test_short_message_skips_extraction(self):
        """handle_message should NOT trigger extraction for messages < 3 words."""
        from unittest.mock import AsyncMock

        conn = FakeNeo4jConnection()
        gs = GraphStore(conn, FakeEmbeddingGenerator())
        config = build_test_config()
        pipeline = FakeExtractionPipeline()

        handler = ConversationHandler(
            config=config,
            graph_store=gs,
            extraction_pipeline=pipeline,
            retriever=_make_retriever(config, gs),
            llm_provider=FakeLLM(),
        )

        # Patch extraction pipeline to track calls
        handler._extraction_pipeline.extract_from_utterance = AsyncMock()

        await handler.handle_message(
            user_message="hi there",  # 2 words — below threshold
            session_id="short-s1",
        )

        # Give any scheduled tasks a moment to run
        import asyncio

        await asyncio.sleep(0.01)

        handler._extraction_pipeline.extract_from_utterance.assert_not_called()

    @pytest.mark.asyncio
    async def test_long_message_triggers_extraction(self):
        """handle_message SHOULD trigger extraction for messages >= 3 words.

        The extraction task is gated on (event_id AND word_count >= 3). The
        event store must be enabled so handle_message produces a non-None
        event_id; without it the task is never created regardless of word count.
        """
        import asyncio
        from unittest.mock import AsyncMock

        conn = FakeNeo4jConnection()
        gs = GraphStore(conn, FakeEmbeddingGenerator())
        # Enable in-memory event store so handle_message produces a non-None event_id.
        config = build_test_config(event_store_enabled=True, event_store_db_path=":memory:")
        pipeline = FakeExtractionPipeline()

        handler = ConversationHandler(
            config=config,
            graph_store=gs,
            extraction_pipeline=pipeline,
            retriever=_make_retriever(config, gs),
            llm_provider=FakeLLM(),
        )

        # Patch extraction pipeline to track calls
        handler._extraction_pipeline.extract_from_utterance = AsyncMock()

        await handler.handle_message(
            user_message="I use Python for data pipelines",  # 6 words — above threshold
            session_id="long-s1",
        )

        # Give the scheduled background task a moment to fire
        await asyncio.sleep(0.05)

        handler._extraction_pipeline.extract_from_utterance.assert_called()


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


class TestConversationTemperature:
    """Cluster 3: ConversationHandler uses conversation_temperature, not extraction."""

    @pytest.mark.asyncio
    async def test_invoke_uses_conversation_temperature_default(self, conversation_handler):
        """First-turn invoke must carry conversation_temperature (0.7), not extraction temp (0.0)."""
        from backend.llm.models import LLMResponse

        captured = []

        async def capture(request):
            captured.append(request)
            return LLMResponse(content="plain response", tool_calls=None)

        conversation_handler._provider.invoke = capture

        await conversation_handler.handle_message(user_message="hello", session_id="temp-s1")

        assert len(captured) >= 1, "invoke was not called"
        assert (
            captured[0].temperature == 0.7
        ), f"Expected conversation_temperature 0.7, got {captured[0].temperature}"
        # Guard: must not be the extraction default (0.0)
        assert captured[0].temperature != 0.0

    @pytest.mark.asyncio
    async def test_invoke_honors_config_override(self, conversation_handler):
        """Overriding config.llm.conversation_temperature flows through to invoke."""
        from backend.llm.models import LLMResponse

        conversation_handler.config.llm.conversation_temperature = 0.5

        captured = []

        async def capture(request):
            captured.append(request)
            return LLMResponse(content="plain response", tool_calls=None)

        conversation_handler._provider.invoke = capture

        await conversation_handler.handle_message(user_message="hello", session_id="temp-s2")
        assert captured[0].temperature == 0.5


# =============================================================================
# Cluster 3 Task 8: Response post-filter (slop regen + strip fallback)
# =============================================================================


class TestPostFilterRegeneration:
    """Cluster 3: response with critical slop triggers regeneration; fallback strips on cap."""

    @pytest.fixture
    def handler_with_queued_responses(self, conversation_handler):
        """Patch the fake provider's invoke to return a scripted queue."""

        def _builder(responses: list[str]):
            # Make a shallow shared queue on the provider
            conversation_handler._provider._scripted_queue = list(responses)

            from backend.llm.models import LLMResponse

            async def scripted_invoke(request):
                q = conversation_handler._provider._scripted_queue
                content = q.pop(0) if q else "fallback scripted response"
                return LLMResponse(content=content, tool_calls=None)

            conversation_handler._provider.invoke = scripted_invoke
            return conversation_handler

        return _builder

    @pytest.mark.asyncio
    async def test_clean_response_not_regenerated(self, handler_with_queued_responses):
        handler = handler_with_queued_responses(["This is a plain response with no slop."])
        result = await handler.handle_message(user_message="hello", session_id="pf-s1")
        assert result == "This is a plain response with no slop."
        # Queue fully consumed — exactly 1 invoke happened for this turn.
        assert handler._provider._scripted_queue == []

    @pytest.mark.asyncio
    async def test_slop_response_triggers_regeneration(self, handler_with_queued_responses):
        # Note: first response contains an emoji; post-filter should regenerate.
        handler = handler_with_queued_responses(
            [
                "Great work \U0001f389 ship it.",  # attempt 1: slop
                "Ship it.",  # attempt 2 (first regen): clean
            ]
        )
        result = await handler.handle_message(user_message="hello", session_id="pf-s2")
        assert "\U0001f389" not in result
        assert "Ship it" in result

    @pytest.mark.asyncio
    async def test_two_regen_cap_then_strip_fallback(self, handler_with_queued_responses):
        handler = handler_with_queued_responses(
            [
                "Great \U0001f389 work",  # attempt 1: slop
                "Amazing \U0001f680 output",  # attempt 2 (first regen): still slop
                "Even more \U0001f4af slop",  # attempt 3 (second regen): still slop — cap reached
                "never consumed",  # should not be popped
            ]
        )
        result = await handler.handle_message(user_message="hello", session_id="pf-s3")
        # After cap, strip_fixable runs on the last response; emojis removed.
        assert "\U0001f389" not in result
        assert "\U0001f680" not in result
        assert "\U0001f4af" not in result
        # The fourth queue item remains, proving we stopped at 2 regen attempts.
        assert "never consumed" in handler._provider._scripted_queue

    @pytest.mark.asyncio
    async def test_regen_rider_names_violation_patterns(self, handler_with_queued_responses):
        """The regeneration request's system message must name the detected slop patterns."""
        handler = handler_with_queued_responses(
            [
                "Great work \U0001f389",  # attempt 1: emoji
                "Ship it.",  # attempt 2: clean
            ]
        )
        captured_requests = []
        original_invoke = handler._provider.invoke

        async def capture(request):
            captured_requests.append(request)
            return await original_invoke(request)

        handler._provider.invoke = capture

        await handler.handle_message(user_message="hello", session_id="pf-s4")

        # At least one of the requests must be the regen request.
        assert len(captured_requests) >= 2
        regen_request = captured_requests[1]
        # Rider is appended as role=user (Fix H: role=system is non-standard after
        # an assistant turn per OpenAI spec). Check both roles for the violation text
        # so the test remains unambiguous about what it's asserting.
        regen_rider_content = "\n".join(
            m["content"] for m in regen_request.messages if m["role"] in ("user", "system")
        )
        # Rider should mention "emoji" as a detected violation type.
        assert "emoji" in regen_rider_content.lower()

    @pytest.mark.asyncio
    async def test_regen_uses_lower_temperature(self, handler_with_queued_responses):
        """Regeneration LLMRequest uses a tighter temperature (conversation_temp - 0.2 floor 0.3)."""
        handler = handler_with_queued_responses(
            [
                "Great work \U0001f389",
                "Ship it.",
            ]
        )
        captured = []
        original_invoke = handler._provider.invoke

        async def capture(request):
            captured.append(request)
            return await original_invoke(request)

        handler._provider.invoke = capture

        await handler.handle_message(user_message="hello", session_id="pf-s5")

        # First request: conversation_temperature (0.7 default)
        assert captured[0].temperature == 0.7
        # Second request (regen): conversation_temperature - 0.2 = 0.5
        assert captured[1].temperature == 0.5
        assert captured[1].temperature >= 0.3  # floor
