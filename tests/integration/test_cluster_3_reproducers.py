"""End-to-end integration reproducers for Cluster 3.

Locks in combined behavior across Cluster 3 changes:
- Persona injection in system prompt (Task 6)
- Conversation temperature separation (Tasks 1 + 7)
- Identity-intent routing (Tasks 4 + 5)
- Response post-filter regeneration (Task 8)

These tests catch regressions when any subsystem's contract shifts.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.chat.conversation_handler import ConversationHandler
from backend.knowledge.retrieval.knowledge_retriever import KnowledgeRetriever
from backend.knowledge.retrieval.query_classifier import QueryClassifier
from backend.llm.models import LLMResponse
from tests.mocks.config import build_test_config
from tests.mocks.ollama import FakeLLM

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Shared seed data
# ---------------------------------------------------------------------------

_SEEDED_IDENTITY = {
    "identity": {
        "id": "mist-identity",
        "display_name": "MIST",
        "pronouns": "she/her",
        "self_concept": "A cognitive architecture for personal knowledge.",
    },
    "traits": [
        {
            "id": "trait-warm",
            "display_name": "Warm",
            "axis": "Persona",
            "description": "Friendly default tone.",
        },
        {
            "id": "trait-technical",
            "display_name": "Technical",
            "axis": "Persona",
            "description": "Precise language.",
        },
    ],
    "capabilities": [
        {
            "id": "cap-tool-use",
            "display_name": "Tool use",
            "description": "Invokes MCP tools.",
        },
    ],
    "preferences": [
        {
            "id": "pref-no-emoji",
            "display_name": "No emoji or unicode decoration",
            "enforcement": "absolute",
            "context": "Hard rule across all output channels.",
        },
        {
            "id": "pref-no-ai-slop",
            "display_name": "No AI-slop patterns",
            "enforcement": "absolute",
            "context": "No superlatives, hype, filler, exclamation spam.",
        },
    ],
}


def _build_graph_store() -> MagicMock:
    """Build a MagicMock graph_store whose get_mist_identity_context() returns seeded data.

    get_mist_identity_context is a sync method on GraphStore (Task 5B deviation
    from spec), so a synchronous MagicMock is correct here.
    """
    graph_store = MagicMock()
    graph_store.get_mist_identity_context = MagicMock(return_value=_SEEDED_IDENTITY)
    return graph_store


def _build_retriever(graph_store: MagicMock, query_classifier=None) -> KnowledgeRetriever:
    """Build a KnowledgeRetriever wired to the seeded graph_store."""
    return KnowledgeRetriever(
        config=build_test_config(),
        graph_store=graph_store,
        vector_store=None,
        query_classifier=query_classifier,
        embedding_provider=None,
    )


class _ScriptedFakeLLM(FakeLLM):
    """FakeLLM subclass that pops responses from a queue in order.

    Falls back to FakeLLM._resolve when the queue is exhausted.
    """

    def __init__(self, responses: list[str]):
        super().__init__()
        self._queue = list(responses)
        self.invocation_requests: list = []

    async def invoke(self, request):
        self.calls.append(request)
        self.invocation_requests.append(request)
        content = self._queue.pop(0) if self._queue else self._resolve(request)
        return LLMResponse(content=content, partial=False)


def _build_handler(
    graph_store: MagicMock,
    retriever: KnowledgeRetriever,
    responses: list[str] | None = None,
) -> tuple[ConversationHandler, _ScriptedFakeLLM]:
    """Build a ConversationHandler with a scripted FakeLLM."""
    fake_llm = _ScriptedFakeLLM(responses or ["plain response"])

    handler = ConversationHandler(
        config=build_test_config(),
        graph_store=graph_store,
        extraction_pipeline=MagicMock(extract_from_utterance=AsyncMock()),
        retriever=retriever,
        llm_provider=fake_llm,
    )
    return handler, fake_llm


# ---------------------------------------------------------------------------
# TestEndToEndPersonaInjection
# ---------------------------------------------------------------------------


class TestEndToEndPersonaInjection:
    """End-to-end: first handle_message call yields a system prompt with full persona."""

    @pytest.mark.asyncio
    async def test_first_turn_sees_persona_in_system_prompt(self):
        graph_store = _build_graph_store()
        retriever = _build_retriever(graph_store)
        handler, fake_llm = _build_handler(graph_store, retriever, responses=["plain response"])

        await handler.handle_message(user_message="I use Python.", session_id="e2e-s1")

        assert len(fake_llm.invocation_requests) >= 1
        first_request = fake_llm.invocation_requests[0]

        # Collect all system message content
        system_content = "\n".join(
            m["content"] for m in first_request.messages if m["role"] == "system"
        )

        # Persona header: display_name + pronouns
        assert "You are MIST" in system_content
        assert "she/her" in system_content

        # Absolute preferences rendered as HARD RULES (not as raw IDs)
        assert "HARD RULE" in system_content
        assert "pref-no-emoji" not in system_content
        assert "pref-no-ai-slop" not in system_content

        # Preference display names appear
        assert "No emoji or unicode decoration" in system_content
        assert "No AI-slop patterns" in system_content

        # Traits appear
        assert "Warm" in system_content
        assert "Technical" in system_content

        # Capabilities appear
        assert "Tool use" in system_content


# ---------------------------------------------------------------------------
# TestEndToEndPostFilter
# ---------------------------------------------------------------------------


class TestEndToEndPostFilter:
    """End-to-end: slop in LLM response triggers regeneration through the real pipeline."""

    @pytest.mark.asyncio
    async def test_emoji_response_gets_regenerated(self):
        graph_store = _build_graph_store()
        retriever = _build_retriever(graph_store)
        handler, fake_llm = _build_handler(
            graph_store,
            retriever,
            responses=[
                "Shipping this \U0001f389 is great.",  # attempt 1: contains emoji
                "Shipping this is great.",  # attempt 2: clean
            ],
        )

        result = await handler.handle_message(user_message="hello", session_id="e2e-s2")

        assert "\U0001f389" not in result
        assert result == "Shipping this is great."
        # Both queued responses consumed (first triggered regen, second passed)
        assert len(fake_llm.invocation_requests) == 2

    @pytest.mark.asyncio
    async def test_clean_response_no_regeneration(self):
        graph_store = _build_graph_store()
        retriever = _build_retriever(graph_store)
        handler, fake_llm = _build_handler(
            graph_store,
            retriever,
            responses=["Plain response, no slop."],
        )

        result = await handler.handle_message(user_message="hello", session_id="e2e-s3")

        assert result == "Plain response, no slop."
        # Only one invocation — no regen needed
        assert len(fake_llm.invocation_requests) == 1


# ---------------------------------------------------------------------------
# TestEndToEndIdentityIntent
# ---------------------------------------------------------------------------


class TestEndToEndIdentityIntent:
    """End-to-end: identity-classified query surfaces persona content via formatted_context."""

    @pytest.mark.asyncio
    async def test_identity_query_returns_persona_context(self):
        # Real QueryClassifier exercises the full intent-routing path.
        classifier = QueryClassifier()
        graph_store = _build_graph_store()
        retriever = _build_retriever(graph_store, query_classifier=classifier)

        result = await retriever.retrieve(
            query="what are your preferences?",
            user_id="User",
        )

        assert result.intent == "identity"

        # Preference display names surface in formatted_context
        assert "No emoji or unicode decoration" in result.formatted_context
        assert "No AI-slop patterns" in result.formatted_context

        # MIST name from identity block
        assert "MIST" in result.formatted_context

        # Identity path returns no graph facts
        assert result.facts == []
        assert result.requires_mcp is False

    @pytest.mark.asyncio
    async def test_identity_query_who_are_you(self):
        """Secondary guard: 'who are you' also routes identity."""
        classifier = QueryClassifier()
        graph_store = _build_graph_store()
        retriever = _build_retriever(graph_store, query_classifier=classifier)

        result = await retriever.retrieve(
            query="who are you?",
            user_id="User",
        )

        assert result.intent == "identity"
        assert "MIST" in result.formatted_context


# ---------------------------------------------------------------------------
# TestEndToEndTemperatureSplit
# ---------------------------------------------------------------------------


class TestEndToEndTemperatureSplit:
    """End-to-end: conversation vs extraction temperature split."""

    @pytest.mark.asyncio
    async def test_conversation_invoke_uses_conversation_temperature(self):
        graph_store = _build_graph_store()
        retriever = _build_retriever(graph_store)
        handler, fake_llm = _build_handler(graph_store, retriever, responses=["plain"])

        await handler.handle_message(user_message="hello", session_id="e2e-s4")

        assert len(fake_llm.invocation_requests) >= 1
        # First LLM call must use conversation_temperature (default 0.7)
        assert fake_llm.invocation_requests[0].temperature == 0.7
        # Extraction-side temperature unchanged at 0.0
        assert handler.config.llm.temperature == 0.0

    @pytest.mark.asyncio
    async def test_regen_uses_lower_temperature(self):
        graph_store = _build_graph_store()
        retriever = _build_retriever(graph_store)
        handler, fake_llm = _build_handler(
            graph_store,
            retriever,
            responses=[
                "Great work \U0001f389",  # triggers regen
                "Ship it.",  # clean
            ],
        )

        result = await handler.handle_message(user_message="hello", session_id="e2e-s5")

        assert result == "Ship it."
        assert len(fake_llm.invocation_requests) == 2

        # First invoke: conversation_temperature (0.7)
        assert fake_llm.invocation_requests[0].temperature == 0.7

        # Regen invoke: conversation_temperature - 0.2 = 0.5
        regen_temp = fake_llm.invocation_requests[1].temperature
        assert abs(regen_temp - 0.5) < 1e-9
