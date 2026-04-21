"""End-to-end integration reproducers for Cluster 6 (context budget).

Verifies that budget-aware context assembly works through a real
ConversationHandler + KnowledgeRetriever + ContextBudgetPlanner chain with
a FakeLLM. Two flows:

- Long-history path: 50-turn synthetic session with a tight context window.
  Planner must prune so the LLMRequest stays within budget and the handler
  does not crash.
- Max-tokens wiring: LLMRequest.max_tokens comes from config.llm.conversation
  _max_tokens, not the historical hardcoded 400.

Spec: cluster-execution-roadmap.md Cluster 6.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.chat.context_budget import ApproximateTokenCounter, ContextBudgetPlanner
from backend.chat.conversation_handler import ConversationHandler
from backend.knowledge.config import ContextBudgetConfig, KnowledgeConfig
from backend.knowledge.retrieval.knowledge_retriever import KnowledgeRetriever
from tests.mocks.config import build_test_config
from tests.mocks.ollama import FakeLLM

pytestmark = pytest.mark.integration


def _build_graph_store() -> MagicMock:
    graph_store = MagicMock()
    graph_store.get_mist_identity_context = MagicMock(
        return_value={
            "identity": {
                "id": "mist-identity",
                "display_name": "MIST",
                "pronouns": "she/her",
                "self_concept": "A cognitive architecture.",
            },
            "traits": [],
            "capabilities": [],
            "preferences": [],
        }
    )
    graph_store.search_similar_entities = MagicMock(return_value=[])
    graph_store.get_user_relationships_to_entities = MagicMock(return_value=[])
    graph_store.get_entity_neighborhood = MagicMock(return_value=[])
    return graph_store


def _build_handler(
    *,
    context_window: int,
    conversation_max_tokens: int = 1024,
    budget_enabled: bool = True,
) -> tuple[ConversationHandler, FakeLLM]:
    config: KnowledgeConfig = build_test_config()
    config.context_budget = ContextBudgetConfig(
        context_window=context_window,
        output_reserve_tokens=100,
        safety_margin_tokens=50,
        retrieval_budget_ratio=0.3,
        enabled=budget_enabled,
    )
    config.llm.conversation_max_tokens = conversation_max_tokens

    graph_store = _build_graph_store()
    retriever = KnowledgeRetriever(
        config=config,
        graph_store=graph_store,
        vector_store=None,
        query_classifier=None,
        embedding_provider=None,
    )
    fake_llm = FakeLLM(default_response="plain reply")

    handler = ConversationHandler(
        config=config,
        graph_store=graph_store,
        extraction_pipeline=MagicMock(extract_from_utterance=AsyncMock()),
        retriever=retriever,
        llm_provider=fake_llm,
    )
    return handler, fake_llm


# ---------------------------------------------------------------------------
# TestBudgetAwareLongSession
# ---------------------------------------------------------------------------


class TestBudgetAwareLongSession:
    @pytest.mark.asyncio
    async def test_long_session_prunes_history_and_handler_survives(self):
        """50 turns of ~30-token history — the planner must prune so the
        LLMRequest stays within budget and no crash occurs.
        """
        handler, fake_llm = _build_handler(context_window=2000)
        assert handler._budget_planner is not None

        # Pre-populate 50 turns of synthetic history.
        session = handler.get_or_create_session("long-s1")
        for i in range(50):
            session.add_message("user", f"user msg {i} " + "x" * 30)
            session.add_message("assistant", f"assistant msg {i} " + "y" * 30)

        result = await handler.handle_message(
            user_message="what did we talk about earlier",
            session_id="long-s1",
        )

        assert result == "plain reply"
        # The FakeLLM captured one invocation (no tool flow).
        assert len(fake_llm.calls) == 1
        sent_request = fake_llm.calls[0]
        # The sent history messages must be fewer than 100 (pruned).
        non_system = [m for m in sent_request.messages if m["role"] != "system"]
        assert len(non_system) < 100, f"expected pruning, got {len(non_system)} non-system messages"
        # The final user utterance must still be in the prompt.
        user_contents = [m["content"] for m in non_system if m["role"] == "user"]
        assert any("what did we talk about earlier" in c for c in user_contents)

    @pytest.mark.asyncio
    async def test_max_tokens_reads_from_config(self):
        """LLMRequest.max_tokens = config.llm.conversation_max_tokens."""
        handler, fake_llm = _build_handler(
            context_window=8192,
            conversation_max_tokens=2048,
        )
        await handler.handle_message(user_message="hello there", session_id="mt-s1")

        assert len(fake_llm.calls) == 1
        assert fake_llm.calls[0].max_tokens == 2048

    @pytest.mark.asyncio
    async def test_disabled_budget_preserves_full_history(self):
        """With context_budget.enabled=False the planner is inactive — history
        is bounded only by the caller's max_history parameter, not by budget.
        """
        handler, fake_llm = _build_handler(
            context_window=500,  # would force pruning if active
            budget_enabled=False,
        )
        assert handler._budget_planner is None

        session = handler.get_or_create_session("disabled-s1")
        for i in range(5):
            session.add_message("user", f"m{i}")

        # max_history default (10) accommodates 5 prior + 1 new = 6 user msgs.
        await handler.handle_message(user_message="final", session_id="disabled-s1")

        sent_request = fake_llm.calls[0]
        non_system = [m for m in sent_request.messages if m["role"] != "system"]
        user_messages = [m for m in non_system if m["role"] == "user"]
        assert len(user_messages) == 6


# ---------------------------------------------------------------------------
# TestBudgetIntegrationWithCounter
# ---------------------------------------------------------------------------


class TestBudgetIntegrationWithCustomCounter:
    @pytest.mark.asyncio
    async def test_custom_counter_via_injected_planner(self):
        """A ContextBudgetPlanner injected with a tighter counter is used."""
        config: KnowledgeConfig = build_test_config()
        config.context_budget = ContextBudgetConfig(
            context_window=2000,
            output_reserve_tokens=100,
            safety_margin_tokens=50,
            enabled=True,
        )

        # Wrap counter so tests can detect that it was called.
        base_counter = ApproximateTokenCounter()
        call_count = {"count": 0}

        class InstrumentedCounter(ApproximateTokenCounter):
            def count(self, text):
                call_count["count"] += 1
                return base_counter.count(text)

        planner = ContextBudgetPlanner(config.context_budget, counter=InstrumentedCounter())

        graph_store = _build_graph_store()
        retriever = KnowledgeRetriever(
            config=config,
            graph_store=graph_store,
            vector_store=None,
            query_classifier=None,
            embedding_provider=None,
        )
        handler = ConversationHandler(
            config=config,
            graph_store=graph_store,
            extraction_pipeline=MagicMock(extract_from_utterance=AsyncMock()),
            retriever=retriever,
            llm_provider=FakeLLM(),
            budget_planner=planner,
        )

        await handler.handle_message(user_message="hi there", session_id="cc-s1")

        assert call_count["count"] > 0  # counter was exercised by the planner
