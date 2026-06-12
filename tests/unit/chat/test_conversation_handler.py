"""Tests for ConversationHandler Phase 2B refactor."""

import pytest
from pydantic import ValidationError

from backend.chat.conversation_handler import ConversationHandler
from backend.knowledge.extraction.tool_usage_tracker import ToolUsageTracker
from backend.knowledge.retrieval.knowledge_retriever import KnowledgeRetriever
from backend.knowledge.storage.graph_store import GraphStore
from tests.mocks.config import build_test_config
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection
from tests.mocks.ollama import FakeLLM
from tests.unit.conftest import make_test_conventions_loader


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
            conventions_loader=make_test_conventions_loader(),
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
            conventions_loader=make_test_conventions_loader(),
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
            conventions_loader=make_test_conventions_loader(),
        )

        await handler._extract_knowledge_async(
            utterance="I use Python and React",
            conversation_history=[{"role": "user", "content": "I use Python and React"}],
            event_id="evt-001",
            session_id="sess-001",
            recorded_at="2026-06-12T09:00:00+00:00",
        )

        assert len(pipeline.calls) == 1
        assert pipeline.calls[0]["utterance"] == "I use Python and React"
        assert pipeline.calls[0]["event_id"] == "evt-001"
        assert pipeline.calls[0]["session_id"] == "sess-001"
        # C1 fact-time threading: a regression here silently falls back to
        # wall-clock and rebuilds resolve relative dates differently than the
        # live turn did (deep review tests-quality-1).
        assert pipeline.calls[0]["recorded_at"] == "2026-06-12T09:00:00+00:00"

    @pytest.mark.asyncio
    async def test_extraction_call_propagates_session_and_event_to_llm_context(self):
        """Regression: extraction LLM calls must inherit session_id + event_id from caller.

        Pre-fix bug (2026-04-27): the extraction call sites
        (ontology_extractor, scope_classifier, internal_derivation) only set
        call_site in their llm_call_context blocks. The outer
        _extract_knowledge_async path did not wrap them in a context with
        session_id + event_id, so emitted phase=llm_call records had both
        IDs as None. The V8 scorer's event_id-based join broke as a result.

        Fix: _extract_knowledge_async wraps the extract_from_utterance call
        in llm_call_context(session_id=..., event_id=...). The inner blocks
        merge with inner-precedence so call_site is preserved while session
        and event are inherited from the outer context.
        """
        # Arrange -- pipeline that captures the LLM call context state at the
        # moment extract_from_utterance is invoked.
        from backend.llm.instrumented_provider import get_llm_call_context

        class CapturingPipeline:
            def __init__(self):
                self.captured_context: dict | None = None

            async def extract_from_utterance(self, **kwargs):
                self.captured_context = get_llm_call_context()
                from backend.knowledge.extraction.validator import ValidationResult

                return ValidationResult(valid=True, entities=[], relationships=[])

        pipeline = CapturingPipeline()
        conn = FakeNeo4jConnection()
        gs = GraphStore(conn, FakeEmbeddingGenerator())
        config = build_test_config()

        handler = ConversationHandler(
            config=config,
            graph_store=gs,
            extraction_pipeline=pipeline,
            retriever=_make_retriever(config, gs),
            llm_provider=FakeLLM(),
            conventions_loader=make_test_conventions_loader(),
        )

        # Act
        await handler._extract_knowledge_async(
            utterance="I use Python and React",
            conversation_history=[{"role": "user", "content": "..."}],
            event_id="evt-abc-123",
            session_id="sess-xyz-789",
        )

        # Assert -- both IDs propagated; ready for inner llm_call_context
        # blocks (call_site=...) to merge in.
        assert pipeline.captured_context is not None
        assert pipeline.captured_context.get("session_id") == "sess-xyz-789"
        assert pipeline.captured_context.get("event_id") == "evt-abc-123"

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
            conventions_loader=make_test_conventions_loader(),
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
            conventions_loader=make_test_conventions_loader(),
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
            conventions_loader=make_test_conventions_loader(),
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
            conventions_loader=make_test_conventions_loader(),
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
            conventions_loader=make_test_conventions_loader(),
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
# Auto-inject vault-only contract (ADR-010 invariant 4)
# =============================================================================


class TestAutoInjectVaultOnly:
    """Auto-inject must query vault sidecar only - never graph.

    Per ADR-010 invariant 4 ("No semantic read cross-pollination.
    Reasoning queries target graph. Prose/history queries target vault
    sidecar. Hybrid merges happen at the retriever layer, never inside
    a store."), the four-layer memory architecture establishes:
      - Graph (Neo4j) = MIST's reasoning substrate. Typed entities and
        relationships. Queried via the explicit `query_knowledge_graph`
        tool when the model needs structured user facts.
      - Vault sidecar (sqlite-vec + FTS5 over `mist-memory/`) = canonical
        prose/history. Queried for auto-inject context.

    Pre-fix, `handle_message` called `retriever.retrieve()` with the user
    message as query, which traversed graph + vector + sidecar and
    merged the result. Graph hits leaked into pass 1's system prompt as
    "Relevant knowledge from your graph (query: <user query>)" framing,
    which the model interpreted as a definitive search result. Two
    failure modes followed:
      - FN: when graph returned off-topic but personal-looking facts
        (e.g., Rust query -> trait-engineer-mindset), the model concluded
        "graph has no Rust info" and skipped the tool while textually
        claiming "I have checked the knowledge graph."
      - FP: when graph surfaced any user-related fact for an unrelated
        query (e.g., haiku -> "afternoon - related to user"), the model
        tool-called inappropriately.

    The architecturally-correct fix per ADR-010: auto-inject queries the
    vault sidecar only (force_intent="historical"). Graph is reserved
    for the explicit tool path. No cross-pollination, no bias.
    """

    @staticmethod
    def _spy_handler(call_order: list[str], retrieve_calls: list[dict]):
        """Build a ConversationHandler with spies on retriever.retrieve
        (capturing kwargs into retrieve_calls) and llm_provider.invoke
        (recording call sequence into call_order).
        """
        config = build_test_config()  # auto_inject_docs default True
        conn = FakeNeo4jConnection()
        gs = GraphStore(conn, FakeEmbeddingGenerator())
        pipeline = FakeExtractionPipeline()
        retriever = _make_retriever(config, gs)
        fake_llm = FakeLLM()

        original_invoke = fake_llm.invoke

        async def spy_invoke(request):
            call_order.append("provider")
            return await original_invoke(request)

        fake_llm.invoke = spy_invoke  # type: ignore[method-assign]

        original_retrieve = retriever.retrieve

        async def spy_retrieve(*args, **kwargs):
            call_order.append("retriever")
            retrieve_calls.append(dict(kwargs))
            return await original_retrieve(*args, **kwargs)

        retriever.retrieve = spy_retrieve  # type: ignore[method-assign]

        handler = ConversationHandler(
            config=config,
            graph_store=gs,
            extraction_pipeline=pipeline,
            retriever=retriever,
            llm_provider=fake_llm,
            conventions_loader=make_test_conventions_loader(),
        )
        return handler, fake_llm, retriever

    @pytest.mark.asyncio
    async def test_auto_inject_calls_retriever_with_historical_intent(self):
        """Auto-inject must invoke retriever.retrieve() with
        force_intent='historical' so only the vault sidecar is queried
        (per ADR-010 invariant 4). The graph remains reserved for the
        explicit query_knowledge_graph tool path.
        """
        call_order: list[str] = []
        retrieve_calls: list[dict] = []
        handler, _fake_llm, _retriever = self._spy_handler(call_order, retrieve_calls)

        await handler.handle_message(
            user_message="What's my experience level with Rust?",
            session_id="auto-inject-historical-1",
        )

        # The first retrieve call is the auto-inject one. Tool-dispatched
        # retrieves (when the model fires query_knowledge_graph) come
        # later in the sequence.
        assert retrieve_calls, "retriever.retrieve must be called for auto-inject"
        auto_inject_kwargs = retrieve_calls[0]
        assert auto_inject_kwargs.get("force_intent") == "historical", (
            "Auto-inject must pass force_intent='historical' to scope to "
            "vault sidecar only (per ADR-010 invariant 4); got kwargs="
            f"{auto_inject_kwargs}"
        )

    @pytest.mark.asyncio
    async def test_no_tool_path_single_provider_invocation(self):
        """When pass 1 produces a content response (no tool calls), only
        ONE provider call is needed. The vault-only auto-inject context
        was already injected for pass 1; no pass 2 is required because
        graph access (the only thing pass 2 could add) is gated behind
        the explicit tool path that pass 1 declined to take.
        """
        call_order: list[str] = []
        retrieve_calls: list[dict] = []
        handler, _fake_llm, _retriever = self._spy_handler(call_order, retrieve_calls)

        await handler.handle_message(
            user_message="What is the capital of Australia?",
            session_id="single-pass-1",
        )

        provider_calls = [c for c in call_order if c == "provider"]
        assert len(provider_calls) == 1, (
            f"expected 1 provider invocation on no-tool path; got "
            f"{len(provider_calls)}: {call_order}"
        )

    @pytest.mark.asyncio
    async def test_tool_path_two_provider_invocations(self):
        """When pass 1 fires a tool call, pass 2 follows with the tool
        result. Tool-decision pass + final-answer pass = 2 provider
        invocations. Unchanged from prior behavior; the architectural
        change is on the no-tool path only.
        """
        from backend.llm.models import LLMResponse, ToolCall

        call_order: list[str] = []
        retrieve_calls: list[dict] = []
        handler, fake_llm, _retriever = self._spy_handler(call_order, retrieve_calls)

        queued: list[LLMResponse] = [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="query_knowledge_graph",
                        arguments={"query": "user rust experience"},
                    )
                ],
                partial=False,
            ),
            LLMResponse(content="You have intermediate Rust experience.", partial=False),
        ]

        async def queued_invoke(request):
            call_order.append("provider")
            fake_llm.calls.append(request)
            return queued.pop(0)

        fake_llm.invoke = queued_invoke  # type: ignore[method-assign]

        await handler.handle_message(
            user_message="What's my experience level with Rust?",
            session_id="two-pass-tool-1",
        )

        provider_calls = [c for c in call_order if c == "provider"]
        assert len(provider_calls) == 2, (
            f"expected 2 provider invocations (initial+final), got "
            f"{len(provider_calls)}: {call_order}"
        )


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
        conventions_loader=make_test_conventions_loader(),
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


class TestBuildRequestPreValidationDump:
    """Cluster 5: _build_request wraps LLMRequest(**kwargs). On Pydantic
    ValidationError it emits a `phase: "llm_request_raw"` JSONL record via
    the debug_logger (gated on MIST_DEBUG_LLM_REQUESTS=1) and re-raises.
    """

    def _build_handler(self, *, debug_logger):
        from backend.debug_jsonl_logger import DebugJSONLLogger  # noqa: F401

        conn = FakeNeo4jConnection()
        gs = GraphStore(conn, FakeEmbeddingGenerator())
        config = build_test_config()
        return ConversationHandler(
            config=config,
            graph_store=gs,
            extraction_pipeline=FakeExtractionPipeline(),
            retriever=_make_retriever(config, gs),
            llm_provider=FakeLLM(),
            conventions_loader=make_test_conventions_loader(),
            debug_logger=debug_logger,
        )

    def test_valid_kwargs_return_llm_request_without_touching_logger(self, tmp_path, monkeypatch):
        from backend.debug_jsonl_logger import DebugJSONLLogger

        monkeypatch.setenv("MIST_DEBUG_LLM_REQUESTS", "1")
        path = tmp_path / "d.jsonl"
        debug_logger = DebugJSONLLogger(path)
        handler = self._build_handler(debug_logger=debug_logger)

        request = handler._build_request(
            call_site="test",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.5,
            max_tokens=128,
        )

        assert request.temperature == 0.5
        assert request.max_tokens == 128
        # No dump should be written on the happy path.
        assert not path.exists() or path.read_text() == ""

    def test_pydantic_validation_error_emits_dump_then_reraises(self, tmp_path, monkeypatch):
        import json as _json

        from backend.debug_jsonl_logger import DebugJSONLLogger

        monkeypatch.setenv("MIST_DEBUG_LLM_REQUESTS", "1")
        path = tmp_path / "d.jsonl"
        debug_logger = DebugJSONLLogger(path)
        handler = self._build_handler(debug_logger=debug_logger)

        with pytest.raises(ValidationError):
            # messages as str is invalid per LLMRequest schema.
            handler._build_request(
                call_site="chat.initial",
                session_id="sess-X",
                messages="not a list",
                temperature=0.7,
            )

        lines = path.read_text(encoding="utf-8").splitlines()
        records = [_json.loads(ln) for ln in lines if ln.strip()]
        assert len(records) == 1
        assert records[0]["phase"] == "llm_request_raw"
        assert records[0]["call_site"] == "chat.initial"
        assert records[0]["session_id"] == "sess-X"
        assert records[0]["request_dict"]["messages"] == "not a list"
        assert records[0]["error"]  # some error message present

    def test_dump_skipped_when_gate_closed_but_exception_still_propagates(
        self, tmp_path, monkeypatch
    ):
        from backend.debug_jsonl_logger import DebugJSONLLogger

        monkeypatch.delenv("MIST_DEBUG_LLM_REQUESTS", raising=False)
        path = tmp_path / "d.jsonl"
        debug_logger = DebugJSONLLogger(path)
        handler = self._build_handler(debug_logger=debug_logger)

        with pytest.raises(ValidationError):
            handler._build_request(
                call_site="chat.initial",
                messages=42,  # invalid
            )

        # Nothing written because gate is closed.
        assert not path.exists() or path.read_text() == ""

    def test_build_request_works_when_debug_logger_is_none(self):
        # Handler constructed without a debug logger should still build valid
        # requests and propagate exceptions without crashing on the missing dep.
        handler = self._build_handler(debug_logger=None)

        request = handler._build_request(
            call_site="test",
            messages=[{"role": "user", "content": "ok"}],
        )
        assert request.messages[0]["content"] == "ok"

        with pytest.raises(ValidationError):
            handler._build_request(call_site="test", messages="invalid")


class TestBudgetAwareBuildMessages:
    """Cluster 6: _build_messages consults the ContextBudgetPlanner (when
    enabled) to prune retrieval + history to fit a hard token budget.
    """

    def _handler_with_tiny_budget(self):
        """Build a handler whose context_window is small enough to force history pruning.

        The static system template is ~550 tokens on its own, so the budget
        must exceed that for the planner to exercise history pruning (rather
        than failing the fits=False degradation path).
        """
        from backend.chat.context_budget import ContextBudgetPlanner
        from backend.knowledge.config import ContextBudgetConfig

        conn = FakeNeo4jConnection()
        gs = GraphStore(conn, FakeEmbeddingGenerator())
        config = build_test_config()
        # 3000 total budget - 50 max_out - 50 reserve - 10 safety ~= 2890 usable
        # Static + persona + tool schemas ~= 1960 tokens (schemas grew with
        # Wave 2 tool catalog additions: tool_call observability, cards,
        # query_vault, switch_form), leaves ~930 for retrieval+history.
        # 50 history messages * ~36 tokens each (user + assistant pair * msg
        # body) -> well above 930 -> forces pruning.
        config.context_budget = ContextBudgetConfig(
            context_window=3000,
            output_reserve_tokens=50,
            safety_margin_tokens=10,
            retrieval_budget_ratio=0.3,
            enabled=True,
        )
        return ConversationHandler(
            config=config,
            graph_store=gs,
            extraction_pipeline=FakeExtractionPipeline(),
            retriever=_make_retriever(config, gs),
            llm_provider=FakeLLM(),
            conventions_loader=make_test_conventions_loader(),
            budget_planner=ContextBudgetPlanner(config.context_budget),
        )

    def test_long_history_is_pruned_to_recent_messages(self):
        from backend.knowledge.models import ConversationSession

        handler = self._handler_with_tiny_budget()
        session = ConversationSession(session_id="budget-s1", user_id="U")
        for i in range(50):
            session.add_message("user", f"message {i} " + "x" * 40)
            session.add_message("assistant", f"reply {i} " + "y" * 40)

        messages = handler._build_messages(
            session=session,
            max_history=100,  # upper bound; planner prunes further
            retrieval_result=None,
            mist_context=None,
            max_output_tokens=50,
        )

        # System prompt(s) still present.
        assert any(m["role"] == "system" for m in messages)
        # History (non-system messages) must be < 100 after pruning.
        history_msgs = [m for m in messages if m["role"] != "system"]
        assert len(history_msgs) < 100
        # Last history message must be the most recent.
        assert history_msgs[-1]["content"].startswith("reply 49")

    def test_disabled_budget_preserves_legacy_behavior(self):
        """When context_budget.enabled=False the handler skips pruning entirely."""
        from backend.knowledge.config import ContextBudgetConfig
        from backend.knowledge.models import ConversationSession

        conn = FakeNeo4jConnection()
        gs = GraphStore(conn, FakeEmbeddingGenerator())
        config = build_test_config()
        config.context_budget = ContextBudgetConfig(
            context_window=100,  # would force pruning if enabled
            output_reserve_tokens=10,
            safety_margin_tokens=5,
            enabled=False,
        )
        handler = ConversationHandler(
            config=config,
            graph_store=gs,
            extraction_pipeline=FakeExtractionPipeline(),
            retriever=_make_retriever(config, gs),
            llm_provider=FakeLLM(),
            conventions_loader=make_test_conventions_loader(),
        )
        # Budget planner must not be constructed when disabled.
        assert handler._budget_planner is None

        session = ConversationSession(session_id="legacy-s1", user_id="U")
        for i in range(5):
            session.add_message("user", f"message {i}")

        messages = handler._build_messages(
            session=session,
            max_history=10,
            retrieval_result=None,
            mist_context=None,
        )
        # All 5 history messages present (no pruning). Count by content so the
        # assertion is robust to always-present prefix blocks injected as
        # non-system messages (vault conventions / the curated user-profile
        # block), which are not conversation history.
        history_msgs = [
            m for m in messages if m["role"] == "user" and m["content"].startswith("message ")
        ]
        assert len(history_msgs) == 5


class TestToolCallObservability:
    """ADR-017 Wave 2: tool_call_started / tool_call_completed event emission.

    The dispatch wrap (_dispatch_tool_with_observability) buffers WS events
    into self._turn_ws_events. handle_message_streaming drains the buffer
    into WSEvent yields. These tests exercise the wrap directly for tight
    isolation and ordering guarantees.
    """

    def _build_handler(self):

        conn = FakeNeo4jConnection()
        gs = GraphStore(conn, FakeEmbeddingGenerator())
        config = build_test_config()
        return ConversationHandler(
            config=config,
            graph_store=gs,
            extraction_pipeline=FakeExtractionPipeline(),
            retriever=_make_retriever(config, gs),
            llm_provider=FakeLLM(),
            conventions_loader=make_test_conventions_loader(),
        )

    @pytest.mark.asyncio
    async def test_emits_started_before_dispatch(self):
        """tool_call_started must be appended BEFORE _dispatch_tool runs.

        Captures buffer state at dispatch time by patching _dispatch_tool.
        The captured snapshot must contain exactly one started event; the
        completed event lands only after dispatch returns.
        """
        from backend.llm.models import ToolCall as LLMToolCall

        handler = self._build_handler()
        captured_at_dispatch: list[dict] = []

        async def patched_dispatch(tc):
            captured_at_dispatch.extend(handler._turn_ws_events)
            return "test result"

        handler._dispatch_tool = patched_dispatch
        tc = LLMToolCall(id="oai-1", name="query_knowledge_graph", arguments={"query": "test"})

        await handler._dispatch_tool_with_observability(tc)

        assert (
            len(captured_at_dispatch) == 1
        ), f"buffer at dispatch time must have exactly tool_call_started; got {captured_at_dispatch}"
        assert captured_at_dispatch[0]["type"] == "tool_call_started"
        assert captured_at_dispatch[0]["name"] == "query_knowledge_graph"

    @pytest.mark.asyncio
    async def test_emits_completed_after_success(self):
        """tool_call_completed appended after successful dispatch.

        error=None, duration_ms is a non-negative integer, result_summary
        is non-empty.
        """
        from backend.llm.models import ToolCall as LLMToolCall

        handler = self._build_handler()

        async def patched_dispatch(tc):
            return "results retrieved"

        handler._dispatch_tool = patched_dispatch
        tc = LLMToolCall(id="oai-2", name="query_knowledge_graph", arguments={"query": "x"})

        result = await handler._dispatch_tool_with_observability(tc)

        assert result == "results retrieved"
        assert len(handler._turn_ws_events) == 2
        completed = handler._turn_ws_events[1]
        assert completed["type"] == "tool_call_completed"
        assert completed["error"] is None
        assert isinstance(completed["duration_ms"], int)
        assert completed["duration_ms"] >= 0
        assert completed["result_summary"] != ""

    @pytest.mark.asyncio
    async def test_emits_completed_after_failure(self):
        """When _dispatch_tool raises, completed event records error repr.

        Result becomes a "Tool error: ..." sentinel; tool_call_completed
        carries the exception repr in its error field.
        """
        from backend.llm.models import ToolCall as LLMToolCall

        handler = self._build_handler()

        async def patched_dispatch(tc):
            raise ValueError("simulated tool failure")

        handler._dispatch_tool = patched_dispatch
        tc = LLMToolCall(id="oai-3", name="query_knowledge_graph", arguments={"query": "x"})

        result = await handler._dispatch_tool_with_observability(tc)

        assert result.startswith("Tool error:")
        assert "simulated tool failure" in result
        completed = handler._turn_ws_events[1]
        assert completed["type"] == "tool_call_completed"
        assert completed["error"] is not None
        assert "simulated tool failure" in completed["error"]
        assert isinstance(completed["duration_ms"], int)
        assert completed["duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_tool_call_id_consistent_across_started_and_completed(self):
        """Same tool_call_id ties the started and completed events together."""
        from backend.llm.models import ToolCall as LLMToolCall

        handler = self._build_handler()

        async def patched_dispatch(tc):
            return "ok"

        handler._dispatch_tool = patched_dispatch
        tc = LLMToolCall(id="oai-4", name="query_knowledge_graph", arguments={"query": "x"})

        await handler._dispatch_tool_with_observability(tc)

        started, completed = handler._turn_ws_events
        assert started["tool_call_id"] == completed["tool_call_id"]
        # Deep review febe-observability-8: the FE-visible id PROPAGATES the
        # provider-assigned tool_call id so FE events, message history, and
        # JSONL records all join on one identifier.
        assert started["tool_call_id"] == tc.id

    @pytest.mark.asyncio
    async def test_args_summary_truncates_long_query(self):
        """args_summary truncates query strings to 40 chars.

        Uses the repr-formatted form so quoting is consistent with the
        rest of the summarizer surface.
        """
        from backend.chat.conversation_handler import _summarize_tool_args

        long_query = "x" * 100
        summary = _summarize_tool_args("query_knowledge_graph", {"query": long_query, "limit": 20})

        # The query portion of the summary should contain a 40-char excerpt
        # of the original (truncated via slicing). Length check: the inner
        # query string (with repr quotes) is the truncated portion.
        # Example: query='xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx' limit=20
        assert "x" * 40 in summary
        assert "x" * 41 not in summary
        assert "limit=20" in summary

    @pytest.mark.asyncio
    async def test_tool_not_found_emits_error_event(self):
        """An unregistered tool name surfaces as a tool_call_completed with
        ToolNotFound error label (not as a silent string sentinel).
        """
        from backend.llm.models import ToolCall as LLMToolCall

        handler = self._build_handler()
        tc = LLMToolCall(id="oai-nf", name="bogus.tool", arguments={})

        result = await handler._dispatch_tool_with_observability(tc)

        assert result.startswith("Tool not found:")
        assert "bogus.tool" in result
        events = handler._turn_ws_events
        assert len(events) == 2
        assert events[0]["type"] == "tool_call_started"
        completed = events[1]
        assert completed["type"] == "tool_call_completed"
        assert completed["error"] == "ToolNotFound: bogus.tool"

    @pytest.mark.asyncio
    async def test_error_field_does_not_leak_repr(self):
        """tool_call_completed.error is sanitized to type+message, not repr().

        repr(ValueError('x')) is "ValueError('x')" (with parens + quotes).
        The sanitized form is "ValueError: x" (cleaner for FE display, no
        leaked frame detail).
        """
        from backend.llm.models import ToolCall as LLMToolCall

        handler = self._build_handler()

        async def patched_dispatch(tc):
            raise ValueError("simulated failure")

        handler._dispatch_tool = patched_dispatch
        tc = LLMToolCall(id="oai-san", name="query_knowledge_graph", arguments={"query": "x"})

        await handler._dispatch_tool_with_observability(tc)

        completed = handler._turn_ws_events[1]
        assert completed["error"] == "ValueError: simulated failure"
        assert "(" not in completed["error"]
        assert "'" not in completed["error"]

    @pytest.mark.asyncio
    async def test_buffer_drained_by_streaming(self):
        """handle_message_streaming yields WSEvent for each buffered event
        and clears the buffer for the next turn.
        """
        from backend.chat.stream_events import WSEvent

        handler = self._build_handler()
        # Pre-populate the buffer as if a prior dispatch had run.
        handler._turn_ws_events = [
            {"type": "tool_call_started", "tool_call_id": "abc", "name": "x", "args_summary": ""},
            {
                "type": "tool_call_completed",
                "tool_call_id": "abc",
                "name": "x",
                "duration_ms": 1,
                "result_summary": "ok",
                "error": None,
            },
        ]

        # Patch handle_message to return immediately so streaming runs.
        async def stub_handle(**kwargs):
            return "hello"

        handler.handle_message = stub_handle

        emitted_events: list[dict] = []
        emitted_tokens: list[str] = []
        async for event in handler.handle_message_streaming(user_message="hi", session_id="s1"):
            if isinstance(event, WSEvent):
                emitted_events.append(event.payload)
            elif hasattr(event, "text"):
                emitted_tokens.append(event.text)

        assert len(emitted_events) == 2
        assert emitted_events[0]["type"] == "tool_call_started"
        assert emitted_events[1]["type"] == "tool_call_completed"
        # WS events must appear BEFORE Token chars (FE sees observability
        # before / alongside the response prose).
        assert "".join(emitted_tokens) == "hello"
        # Buffer cleared after drain.
        assert handler._turn_ws_events == []


class TestFrontendSummonCards:
    """ADR-017 Wave 2: frontend.summon_cards / frontend.dismiss_cards tools.

    The two tools are registered in KNOWLEDGE_TOOL_SCHEMAS and routed via
    _dispatch_tool. Handlers append cards_summon / cards_dismiss WS events
    to the per-turn buffer; the same drain mechanism as tool_call_*
    delivers them to the FE via the bridge.
    """

    def _build_handler(self):
        conn = FakeNeo4jConnection()
        gs = GraphStore(conn, FakeEmbeddingGenerator())
        config = build_test_config()
        return ConversationHandler(
            config=config,
            graph_store=gs,
            extraction_pipeline=FakeExtractionPipeline(),
            retriever=_make_retriever(config, gs),
            llm_provider=FakeLLM(),
            conventions_loader=make_test_conventions_loader(),
        )

    def test_summon_cards_tool_registered(self):
        """The tool name appears in the handler's tool_schemas catalog."""
        handler = self._build_handler()
        tool_names = [s["function"]["name"] for s in handler._tool_schemas]
        assert "frontend.summon_cards" in tool_names
        assert "frontend.dismiss_cards" in tool_names

    @pytest.mark.asyncio
    async def test_summon_cards_emits_panel_event(self):
        """Dispatch with valid args emits cards_summon with correct shape.

        cards_summon panel.cards each carries id, label, and pattern (default
        'lines' for any input card missing the field).
        """
        handler = self._build_handler()
        result = await handler._handle_summon_cards(
            header="Choices",
            cards=[
                {"id": "c1", "label": "Alpha"},
                {"id": "c2", "label": "Beta", "pattern": "dots"},
            ],
        )

        assert "Displayed 2 cards" in result
        assert len(handler._turn_ws_events) == 1
        event = handler._turn_ws_events[0]
        assert event["type"] == "cards_summon"
        assert event["panel"]["header"] == "Choices"
        assert len(event["panel"]["cards"]) == 2
        # Pattern default fills in for the card that omitted it.
        assert event["panel"]["cards"][0]["pattern"] == "lines"
        assert event["panel"]["cards"][1]["pattern"] == "dots"

    @pytest.mark.asyncio
    async def test_summon_cards_returns_summary_string(self):
        """Returns a short non-empty string to the LLM for the final pass."""
        handler = self._build_handler()
        result = await handler._handle_summon_cards(
            header="Test",
            cards=[{"id": "x", "label": "X"}],
        )
        assert isinstance(result, str)
        assert result != ""
        assert "1" in result

    @pytest.mark.asyncio
    async def test_summon_cards_validates_pattern_enum(self):
        """Invalid pattern raises ValueError; no WS event emitted."""
        handler = self._build_handler()
        with pytest.raises(ValueError, match="invalid card.pattern"):
            await handler._handle_summon_cards(
                header="Test",
                cards=[{"id": "c1", "label": "X", "pattern": "rainbow"}],
            )
        assert handler._turn_ws_events == []

    @pytest.mark.asyncio
    async def test_dismiss_cards_emits_event(self):
        """frontend.dismiss_cards appends cards_dismiss (empty payload)."""
        handler = self._build_handler()
        result = await handler._handle_dismiss_cards()
        assert result == "Dismissed cards panel"
        assert handler._turn_ws_events == [{"type": "cards_dismiss"}]

    @pytest.mark.asyncio
    async def test_summon_cards_dispatch_with_observability_full_event_chain(self):
        """End-to-end via _dispatch_tool_with_observability: started, cards_summon,
        completed in that order.
        """
        from backend.llm.models import ToolCall as LLMToolCall

        handler = self._build_handler()
        tc = LLMToolCall(
            id="oai-cards-1",
            name="frontend.summon_cards",
            arguments={
                "header": "Pick",
                "cards": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
            },
        )

        result = await handler._dispatch_tool_with_observability(tc)

        assert "Displayed 2 cards" in result
        event_types = [e["type"] for e in handler._turn_ws_events]
        assert event_types == ["tool_call_started", "cards_summon", "tool_call_completed"]
        # tool_call_started/completed share a tool_call_id distinct from tc.id.
        started = handler._turn_ws_events[0]
        completed = handler._turn_ws_events[2]
        assert started["tool_call_id"] == completed["tool_call_id"]
        assert started["name"] == "frontend.summon_cards"

    @pytest.mark.asyncio
    async def test_summon_cards_requires_non_empty_list(self):
        """Empty cards list rejected by the handler."""
        handler = self._build_handler()
        with pytest.raises(ValueError, match="non-empty"):
            await handler._handle_summon_cards(header="X", cards=[])
        assert handler._turn_ws_events == []


class TestGraphSubgraphEmit:
    """ADR-017 Wave 2: graph_subgraph emit after query_knowledge_graph success.

    The retriever drives the FE graph form via depth-1 focal+neighbors
    layout. Empty hits skip the emit so the FE keeps prior graphData.
    """

    def _build_handler(self):
        conn = FakeNeo4jConnection()
        gs = GraphStore(conn, FakeEmbeddingGenerator())
        config = build_test_config()
        return ConversationHandler(
            config=config,
            graph_store=gs,
            extraction_pipeline=FakeExtractionPipeline(),
            retriever=_make_retriever(config, gs),
            llm_provider=FakeLLM(),
            conventions_loader=make_test_conventions_loader(),
        )

    def _make_result(self, facts: list, query: str = "test"):
        from backend.knowledge.models import RetrievalResult

        return RetrievalResult(
            query=query,
            user_id="User",
            facts=facts,
            entities_found=len(facts),
            total_facts=len(facts),
            formatted_context="...",
            retrieval_time_ms=1.0,
            vector_search_time_ms=0.5,
            graph_traversal_time_ms=0.5,
            config_used={},
        )

    def _make_fact(
        self,
        subject: str = "User",
        predicate: str = "USES",
        obj: str = "Python",
        obj_type: str = "Technology",
        similarity: float = 0.9,
    ):
        from backend.knowledge.models import RetrievedFact

        return RetrievedFact(
            subject=subject,
            subject_type="Person",
            predicate=predicate,
            object=obj,
            object_type=obj_type,
            properties={},
            similarity_score=similarity,
            graph_distance=0,
        )

    @pytest.mark.asyncio
    async def test_emits_graph_subgraph_after_kg_query(self):
        """When retriever returns N>0 facts, the buffer carries graph_subgraph.

        Focal + neighbors populated; edges link focal id to each neighbor;
        the event is a dict matching the ADR-017 shape.
        """
        handler = self._build_handler()
        handler._current_session_id = "s1"
        facts = [
            self._make_fact(obj="Python", obj_type="Technology", similarity=0.95),
            self._make_fact(obj="Neo4j", obj_type="Technology", similarity=0.88),
            self._make_fact(obj="FastAPI", obj_type="Technology", similarity=0.85),
        ]

        async def stub_retrieve(**kwargs):
            return self._make_result(facts, query=kwargs.get("query", "test"))

        handler.retriever.retrieve = stub_retrieve

        result = await handler._handle_query_knowledge_graph(query="what do I use")

        # Tool returned the formatted context, not the placeholder
        assert "No information found" not in result

        # Buffer should have exactly one graph_subgraph event
        graph_events = [e for e in handler._turn_ws_events if e["type"] == "graph_subgraph"]
        assert len(graph_events) == 1
        event = graph_events[0]
        assert event["focal"]["label"] == "User"
        assert event["focal"]["kind"] == "Person"
        assert event["focal"]["x"] == 0.0 and event["focal"]["y"] == 0.0
        assert len(event["neighbors"]) == 3
        neighbor_ids = {n["id"] for n in event["neighbors"]}
        assert neighbor_ids == {"Python", "Neo4j", "FastAPI"}
        # Edges link focal -> each neighbor
        assert all(e["from"] == "User" for e in event["edges"])
        edge_targets = {e["to"] for e in event["edges"]}
        assert edge_targets == {"Python", "Neo4j", "FastAPI"}

    @pytest.mark.asyncio
    async def test_skips_emit_on_empty_hits(self):
        """Retriever returns 0 facts -> no graph_subgraph event."""
        handler = self._build_handler()
        handler._current_session_id = "s1"

        async def stub_retrieve(**kwargs):
            return self._make_result([], query="x")

        handler.retriever.retrieve = stub_retrieve

        result = await handler._handle_query_knowledge_graph(query="unknown thing")
        assert "No information found" in result
        graph_events = [e for e in handler._turn_ws_events if e["type"] == "graph_subgraph"]
        assert graph_events == []

    def test_neighbors_on_unit_circle(self):
        """Each neighbor is at distance ~1.0 from focal (origin)."""
        import math as _math

        from backend.chat.conversation_handler import _build_graph_subgraph_payload

        facts = [self._make_fact(obj=f"E{i}", similarity=0.9 - i * 0.01) for i in range(4)]
        result = self._make_result(facts)

        payload = _build_graph_subgraph_payload(result, seed="s1:1")
        assert payload is not None
        for n in payload["neighbors"]:
            distance = _math.sqrt(n["x"] ** 2 + n["y"] ** 2)
            assert abs(distance - 1.0) < 1e-9, f"neighbor not on unit circle: {n}"

    def test_distant_points_at_radius_1_6(self):
        """Each distant point sits at radius 1.6 from origin."""
        import math as _math

        from backend.chat.conversation_handler import _build_graph_subgraph_payload

        facts = [self._make_fact(obj="X")]
        result = self._make_result(facts)

        payload = _build_graph_subgraph_payload(result, seed="s1:1")
        assert payload is not None
        assert 5 <= len(payload["distant"]) <= 8
        for p in payload["distant"]:
            distance = _math.sqrt(p["x"] ** 2 + p["y"] ** 2)
            assert abs(distance - 1.6) < 1e-9, f"distant not at radius 1.6: {p}"

    def test_deterministic_distant_placement_for_same_seed(self):
        """Same seed produces identical distant placements; different seed differs.

        The Wave 2 prompt scoped this as "deterministic within a turn"
        which the seed parameter mechanism delivers.
        """
        from backend.chat.conversation_handler import _build_graph_subgraph_payload

        facts = [self._make_fact(obj="X")]
        result = self._make_result(facts)

        a = _build_graph_subgraph_payload(result, seed="sess-1:turn-7")
        b = _build_graph_subgraph_payload(result, seed="sess-1:turn-7")
        c = _build_graph_subgraph_payload(result, seed="sess-1:turn-8")

        assert a is not None and b is not None and c is not None
        assert a["distant"] == b["distant"], "same seed must produce identical distant"
        assert a["distant"] != c["distant"], "different seed should produce different distant"

    def test_neighbor_cap_enforced_at_six(self):
        """When the retriever returns more than 6 facts, only 6 neighbors render."""
        from backend.chat.conversation_handler import _build_graph_subgraph_payload

        facts = [self._make_fact(obj=f"E{i}", similarity=0.9 - i * 0.01) for i in range(10)]
        result = self._make_result(facts)

        payload = _build_graph_subgraph_payload(result, seed="s1:1")
        assert payload is not None
        assert len(payload["neighbors"]) == 6
        assert len(payload["edges"]) == 6

    def test_returns_none_for_empty_facts(self):
        """The helper returns None when no facts present; caller must skip emit."""
        from backend.chat.conversation_handler import _build_graph_subgraph_payload

        result = self._make_result([])
        payload = _build_graph_subgraph_payload(result, seed="s1:1")
        assert payload is None


class TestQueryKnowledgeGraphCompactLLMContext:
    """ADR-017 Wave 2 BE/FE differentiation: query_knowledge_graph trims
    the LLM-facing result to focal + neighbor labels (compact, default),
    while the graph_subgraph WS event still carries the full graph.

    verbosity='full' opt-in returns the legacy formatted_context with
    every fact verbatim for cases where one-turn detail is needed.
    """

    def _build_handler(self):
        conn = FakeNeo4jConnection()
        gs = GraphStore(conn, FakeEmbeddingGenerator())
        config = build_test_config()
        return ConversationHandler(
            config=config,
            graph_store=gs,
            extraction_pipeline=FakeExtractionPipeline(),
            retriever=_make_retriever(config, gs),
            llm_provider=FakeLLM(),
            conventions_loader=make_test_conventions_loader(),
        )

    def _make_facts(self, count: int = 3):
        from backend.knowledge.models import RetrievedFact

        names = ["Python", "Neo4j", "FastAPI", "MIST.AI", "llama-cpp", "Tauri", "PyTorch"]
        return [
            RetrievedFact(
                subject="User",
                subject_type="Person",
                predicate="USES",
                object=names[i],
                object_type="Technology",
                properties={},
                similarity_score=0.95 - i * 0.01,
                graph_distance=0,
            )
            for i in range(count)
        ]

    def _make_result(self, facts):
        from backend.knowledge.models import RetrievalResult

        return RetrievalResult(
            query="x",
            user_id="User",
            facts=facts,
            entities_found=len(facts),
            total_facts=len(facts),
            formatted_context="### FULL LEGACY FORMATTED CONTEXT\n- User USES Python\n- User USES Neo4j\n",
            retrieval_time_ms=1.0,
            vector_search_time_ms=0.5,
            graph_traversal_time_ms=0.5,
            config_used={},
        )

    @pytest.mark.asyncio
    async def test_default_compact_returns_focal_and_neighbors_only(self):
        """Default verbosity (omitted) returns the compact format."""
        handler = self._build_handler()
        handler._current_session_id = "s1"
        facts = self._make_facts(count=3)

        async def stub_retrieve(**kwargs):
            return self._make_result(facts)

        handler.retriever.retrieve = stub_retrieve

        result = await handler._handle_query_knowledge_graph(query="what do I use")

        assert result.startswith("Focal: User (Person)")
        assert "Related (3): Python, Neo4j, FastAPI" in result
        assert "Chain another query" in result
        # The full formatted_context must NOT leak into the compact LLM result.
        assert "FULL LEGACY FORMATTED CONTEXT" not in result

    @pytest.mark.asyncio
    async def test_compact_caps_neighbors_at_six(self):
        """Up to 6 neighbor labels appear; the rest are visible only in
        graph_subgraph for the FE.
        """
        handler = self._build_handler()
        handler._current_session_id = "s1"
        facts = self._make_facts(count=7)

        async def stub_retrieve(**kwargs):
            return self._make_result(facts)

        handler.retriever.retrieve = stub_retrieve

        result = await handler._handle_query_knowledge_graph(query="x")
        assert "Related (6)" in result
        # 7th name must not appear in the LLM result.
        assert "PyTorch" not in result

    @pytest.mark.asyncio
    async def test_full_verbosity_returns_formatted_context(self):
        """verbosity='full' returns the legacy verbose formatted_context."""
        handler = self._build_handler()
        handler._current_session_id = "s1"
        facts = self._make_facts(count=2)

        async def stub_retrieve(**kwargs):
            return self._make_result(facts)

        handler.retriever.retrieve = stub_retrieve

        result = await handler._handle_query_knowledge_graph(query="x", verbosity="full")
        assert "FULL LEGACY FORMATTED CONTEXT" in result
        assert "Focal:" not in result

    @pytest.mark.asyncio
    async def test_invalid_verbosity_raises(self):
        handler = self._build_handler()
        with pytest.raises(ValueError, match="invalid verbosity"):
            await handler._handle_query_knowledge_graph(query="x", verbosity="overview")

    @pytest.mark.asyncio
    async def test_graph_subgraph_emit_unchanged_by_verbosity(self):
        """Both compact and full verbosity emit the same graph_subgraph
        event to the FE. Only the LLM-facing result differs.
        """
        handler = self._build_handler()
        handler._current_session_id = "s1"
        facts = self._make_facts(count=3)

        async def stub_retrieve(**kwargs):
            return self._make_result(facts)

        handler.retriever.retrieve = stub_retrieve

        await handler._handle_query_knowledge_graph(query="x", verbosity="compact")
        compact_events = [e for e in handler._turn_ws_events if e["type"] == "graph_subgraph"]
        assert len(compact_events) == 1

        handler._turn_ws_events = []
        await handler._handle_query_knowledge_graph(query="x", verbosity="full")
        full_events = [e for e in handler._turn_ws_events if e["type"] == "graph_subgraph"]
        assert len(full_events) == 1

        # Same shape -- focal, neighbors, edges, distant identical.
        assert compact_events[0]["focal"]["label"] == full_events[0]["focal"]["label"]
        assert len(compact_events[0]["neighbors"]) == len(full_events[0]["neighbors"])

    @pytest.mark.asyncio
    async def test_zero_facts_short_circuit_unchanged(self):
        """Zero results -> 'No information found' regardless of verbosity."""
        handler = self._build_handler()
        handler._current_session_id = "s1"

        async def stub_retrieve(**kwargs):
            return self._make_result([])

        handler.retriever.retrieve = stub_retrieve

        result_compact = await handler._handle_query_knowledge_graph(query="x")
        result_full = await handler._handle_query_knowledge_graph(query="x", verbosity="full")
        assert "No information found" in result_compact
        assert "No information found" in result_full


class TestQueryVault:
    """ADR-017 Wave 2 (vault_results) -- query_vault tool emit contract.

    Tool routes to retriever.retrieve(force_intent='historical'). Tests
    stub the retriever to return controlled RetrievedFact lists so the
    vault_results emit can be exercised without a real vault sidecar.
    """

    def _build_handler(self):
        conn = FakeNeo4jConnection()
        gs = GraphStore(conn, FakeEmbeddingGenerator())
        config = build_test_config()
        return ConversationHandler(
            config=config,
            graph_store=gs,
            extraction_pipeline=FakeExtractionPipeline(),
            retriever=_make_retriever(config, gs),
            llm_provider=FakeLLM(),
            conventions_loader=make_test_conventions_loader(),
        )

    def _make_vault_fact(
        self,
        path: str = "sessions/2026-05-11-test.md",
        section: str | None = "Backend",
        text: str = "This is a vault chunk about the backend architecture.",
        similarity: float = 0.92,
        sources: list[str] | None = None,
        display_similarity: float | None = None,
    ):
        """Build a VaultNote RetrievedFact mirroring _vault_sidecar_retrieve.

        `similarity` populates `similarity_score` (the RRF fusion score).
        `display_similarity` is the distance-derived score the sidecar carries onto
        properties (Task 2) and is what the vault_results payload emits:
        a float for vector hits, None for FTS-only. It is independent of
        `similarity_score`, so set it explicitly when asserting the emit.
        """
        from backend.knowledge.models import RetrievedFact

        return RetrievedFact(
            subject="VaultNote",
            subject_type="VaultSession",
            predicate="MENTIONS",
            object=section or "(file)",
            object_type="VaultChunk",
            properties={
                "path": path,
                "text": text,
                "content": text,
                "sources": sources or ["vector", "fts"],
                "display_similarity": display_similarity,
            },
            similarity_score=similarity,
            graph_distance=99,
        )

    def _make_result(self, facts: list, query: str = "test query"):
        from backend.knowledge.models import RetrievalResult

        return RetrievalResult(
            query=query,
            user_id="User",
            facts=facts,
            entities_found=0,
            total_facts=len(facts),
            formatted_context="prose context for the LLM",
            retrieval_time_ms=1.0,
            vector_search_time_ms=0.5,
            graph_traversal_time_ms=0.0,
            config_used={},
        )

    @pytest.mark.asyncio
    async def test_returns_formatted_context_on_hits(self):
        """Tool returns the retriever's formatted_context on N>0 vault facts."""
        handler = self._build_handler()
        facts = [self._make_vault_fact()]

        async def stub_retrieve(**kwargs):
            assert kwargs.get("force_intent") == "historical"
            return self._make_result(facts)

        handler.retriever.retrieve = stub_retrieve

        result = await handler._handle_query_vault(query="backend")
        assert result == "prose context for the LLM"

    @pytest.mark.asyncio
    async def test_emits_vault_results_event(self):
        """On hits, vault_results event lands in the per-turn buffer."""
        handler = self._build_handler()
        facts = [
            # Vector hit: distance-derived score carried, emitted as the displayed score.
            self._make_vault_fact(
                path="sessions/2026-05-11-test.md",
                section="Backend",
                text="x" * 300,
                similarity=0.88,
                display_similarity=0.88,
                sources=["vector"],
            ),
            # FTS-only hit: no vector score, displayed similarity is None.
            self._make_vault_fact(
                path="decisions/DEC-001.md",
                section=None,
                text="A decision about Python.",
                similarity=0.71,
                display_similarity=None,
                sources=["fts"],
            ),
        ]

        async def stub_retrieve(**kwargs):
            return self._make_result(facts)

        handler.retriever.retrieve = stub_retrieve

        await handler._handle_query_vault(query="backend", limit=5, display_hint="panel")

        events = [e for e in handler._turn_ws_events if e["type"] == "vault_results"]
        assert len(events) == 1
        event = events[0]
        assert event["query"] == "backend"
        assert event["total_results"] == 2
        assert event["display_hint"] == "panel"
        assert len(event["results"]) == 2

        # First result: long content -> snippet truncated
        first = event["results"][0]
        assert first["note_path"] == "sessions/2026-05-11-test.md"
        assert first["section"] == "Backend"
        assert first["full_text"] == "x" * 300
        assert first["snippet"].endswith("...")
        assert len(first["snippet"]) == 203  # 200 chars + "..."
        # Displayed similarity is the distance-derived score (display_similarity), not
        # the RRF fusion score (similarity_score).
        assert first["similarity"] == 0.88
        assert "vector" in first["sources"]

        # Second result: '(file)' heading becomes None; FTS-only hit has no
        # vector score, so the emitted similarity is None (FE renders "lexical").
        second = event["results"][1]
        assert second["section"] is None
        assert second["full_text"] == "A decision about Python."
        assert second["similarity"] is None

    @pytest.mark.asyncio
    async def test_skips_emit_on_zero_hits(self):
        """0 vault facts -> no vault_results emit + descriptive tool result."""
        handler = self._build_handler()

        async def stub_retrieve(**kwargs):
            return self._make_result([])

        handler.retriever.retrieve = stub_retrieve

        result = await handler._handle_query_vault(query="unknown thing")
        assert "No vault content" in result
        events = [e for e in handler._turn_ws_events if e["type"] == "vault_results"]
        assert events == []

    @pytest.mark.asyncio
    async def test_validates_display_hint_enum(self):
        """Invalid display_hint raises ValueError; no event emitted."""
        handler = self._build_handler()
        with pytest.raises(ValueError, match="invalid display_hint"):
            await handler._handle_query_vault(query="x", display_hint="carousel")
        assert handler._turn_ws_events == []

    @pytest.mark.asyncio
    async def test_validates_query_not_empty(self):
        """Empty / whitespace query rejected."""
        handler = self._build_handler()
        with pytest.raises(ValueError, match="non-empty"):
            await handler._handle_query_vault(query="")
        with pytest.raises(ValueError, match="non-empty"):
            await handler._handle_query_vault(query="   ")

    @pytest.mark.asyncio
    async def test_validates_limit_bounds(self):
        """Limit must be 1-10."""
        handler = self._build_handler()
        with pytest.raises(ValueError, match="between 1 and 10"):
            await handler._handle_query_vault(query="x", limit=0)
        with pytest.raises(ValueError, match="between 1 and 10"):
            await handler._handle_query_vault(query="x", limit=11)

    def test_tool_registered_in_schema(self):
        """query_vault appears in the handler's tool_schemas catalog."""
        handler = self._build_handler()
        tool_names = [s["function"]["name"] for s in handler._tool_schemas]
        assert "query_vault" in tool_names

    def test_chunk_id_stable_across_calls(self):
        """Same (path, section) produces the same chunk_id."""
        from backend.chat.conversation_handler import _chunk_id_for

        a = _chunk_id_for("sessions/2026-05-11.md", "Backend")
        b = _chunk_id_for("sessions/2026-05-11.md", "Backend")
        c = _chunk_id_for("sessions/2026-05-11.md", "Frontend")
        d = _chunk_id_for("decisions/DEC-001.md", "Backend")
        assert a == b
        assert a != c
        assert a != d

    def test_note_title_derivation(self):
        """Title is derived from filename stem; date prefix stripped."""
        from backend.chat.conversation_handler import _derive_note_title_from_path

        assert _derive_note_title_from_path("sessions/2026-05-11-test-alpha.md") == "Test Alpha"
        assert _derive_note_title_from_path("decisions/DEC-001-foo.md") == "Dec 001 Foo"
        assert _derive_note_title_from_path("identity/mist.md") == "Mist"
        assert _derive_note_title_from_path("") == "Untitled"


class TestFrontendSwitchForm:
    """ADR-017 Wave 2 -- frontend.switch_form tool + form_switch event."""

    def _build_handler(self):
        conn = FakeNeo4jConnection()
        gs = GraphStore(conn, FakeEmbeddingGenerator())
        config = build_test_config()
        return ConversationHandler(
            config=config,
            graph_store=gs,
            extraction_pipeline=FakeExtractionPipeline(),
            retriever=_make_retriever(config, gs),
            llm_provider=FakeLLM(),
            conventions_loader=make_test_conventions_loader(),
        )

    def test_tool_registered_in_schema(self):
        handler = self._build_handler()
        tool_names = [s["function"]["name"] for s in handler._tool_schemas]
        assert "frontend.switch_form" in tool_names

    @pytest.mark.asyncio
    async def test_emits_form_switch_event(self):
        """Tool appends form_switch with the canonical ADR-017 shape."""
        handler = self._build_handler()
        result = await handler._handle_switch_form(
            form="graph", reason="rendering knowledge graph results"
        )
        assert "Switched" in result
        assert handler._turn_ws_events == [
            {
                "type": "form_switch",
                "form": "graph",
                "reason": "rendering knowledge graph results",
            }
        ]

    @pytest.mark.asyncio
    async def test_validates_form_enum(self):
        handler = self._build_handler()
        with pytest.raises(ValueError, match="invalid form"):
            await handler._handle_switch_form(form="invalid", reason="any")
        assert handler._turn_ws_events == []

    @pytest.mark.asyncio
    async def test_validates_reason_required(self):
        handler = self._build_handler()
        with pytest.raises(ValueError, match="non-empty"):
            await handler._handle_switch_form(form="graph", reason="")
        with pytest.raises(ValueError, match="non-empty"):
            await handler._handle_switch_form(form="graph", reason="   ")

    @pytest.mark.asyncio
    async def test_query_knowledge_graph_does_not_auto_switch_form(self):
        """Per user direction: no auto-form-switch when query_knowledge_graph
        fires. The LLM must chain frontend.switch_form explicitly.
        """
        from backend.knowledge.models import RetrievalResult, RetrievedFact

        handler = self._build_handler()
        handler._current_session_id = "s1"
        facts = [
            RetrievedFact(
                subject="User",
                subject_type="Person",
                predicate="USES",
                object="Python",
                object_type="Technology",
                properties={},
                similarity_score=0.95,
                graph_distance=0,
            )
        ]
        result = RetrievalResult(
            query="x",
            user_id="User",
            facts=facts,
            entities_found=1,
            total_facts=1,
            formatted_context="...",
            retrieval_time_ms=1.0,
            vector_search_time_ms=0.5,
            graph_traversal_time_ms=0.5,
            config_used={},
        )

        async def stub_retrieve(**kwargs):
            return result

        handler.retriever.retrieve = stub_retrieve
        await handler._handle_query_knowledge_graph(query="what do I use")

        form_switches = [e for e in handler._turn_ws_events if e["type"] == "form_switch"]
        assert form_switches == [], "query_knowledge_graph must not auto-emit form_switch"

    @pytest.mark.asyncio
    async def test_dispatch_chain_includes_observability(self):
        """End-to-end via _dispatch_tool_with_observability: started,
        form_switch, completed in that order.
        """
        from backend.llm.models import ToolCall as LLMToolCall

        handler = self._build_handler()
        tc = LLMToolCall(
            id="oai-fs-1",
            name="frontend.switch_form",
            arguments={"form": "graph", "reason": "showing graph data"},
        )
        await handler._dispatch_tool_with_observability(tc)
        event_types = [e["type"] for e in handler._turn_ws_events]
        assert event_types == ["tool_call_started", "form_switch", "tool_call_completed"]


class TestRecordTurnEventRecordedAt:
    """C1: the event timestamp is the fact-time authority for bitemporal edges."""

    def _handler(self, config):
        conn = FakeNeo4jConnection()
        gs = GraphStore(conn, FakeEmbeddingGenerator())
        return ConversationHandler(
            config=config,
            graph_store=gs,
            extraction_pipeline=FakeExtractionPipeline(),
            retriever=_make_retriever(config, gs),
            llm_provider=FakeLLM(),
            conventions_loader=make_test_conventions_loader(),
        )

    def test_returns_event_id_and_utc_aware_recorded_at(self):
        from datetime import datetime

        handler = self._handler(
            build_test_config(event_store_enabled=True, event_store_db_path=":memory:")
        )
        event_id, recorded_at = handler._record_turn_event(
            session_id="s1", user_message="I use Rust", assistant_message="Noted."
        )
        assert event_id is not None
        assert datetime.fromisoformat(recorded_at).tzinfo is not None
        turns = handler.event_store.get_all_turns_for_reextraction()
        assert turns[-1]["timestamp"] == recorded_at

    def test_disabled_event_store_returns_none_pair(self):
        handler = self._handler(build_test_config())
        assert handler._record_turn_event(
            session_id="s1", user_message="x", assistant_message="y"
        ) == (None, None)
