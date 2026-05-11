"""Phase 3 additions to ConversationHandler tests.

Tests wiring of ConventionsLoader into _build_messages (Task 9) and
InvalidationBus subscription for mist_context cache eviction (Task 21).
"""

import asyncio
from pathlib import Path

import pytest

from backend.chat.conversation_handler import ConversationHandler
from backend.knowledge.curation.graph_regenerator import RebuildResult
from backend.knowledge.models import ConversationSession
from backend.knowledge.retrieval.knowledge_retriever import KnowledgeRetriever
from backend.knowledge.storage.graph_store import GraphStore
from backend.vault.conventions import ConventionsLoader
from backend.vault.invalidation_bus import InvalidationBus
from tests.mocks.config import build_test_config
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection
from tests.mocks.ollama import FakeLLM


class FakeExtractionPipelineP3:
    """Minimal extraction pipeline test double."""

    async def extract_from_utterance(self, **kwargs):
        from backend.knowledge.extraction.validator import ValidationResult

        return ValidationResult(valid=True, entities=[], relationships=[])


def _make_handler(
    *,
    conventions_loader: ConventionsLoader,
    invalidation_bus: InvalidationBus | None = None,
) -> ConversationHandler:
    """Construct a ConversationHandler with the given ConventionsLoader."""
    conn = FakeNeo4jConnection()
    gs = GraphStore(conn, FakeEmbeddingGenerator())
    config = build_test_config()
    retriever = KnowledgeRetriever(config=config, graph_store=gs)
    return ConversationHandler(
        config=config,
        graph_store=gs,
        extraction_pipeline=FakeExtractionPipelineP3(),
        retriever=retriever,
        llm_provider=FakeLLM(),
        conventions_loader=conventions_loader,
        invalidation_bus=invalidation_bus,
    )


@pytest.fixture
def invalidation_handler(tmp_path: Path) -> ConversationHandler:
    """ConversationHandler wired to a fresh InvalidationBus for Task 21 tests."""
    loader = ConventionsLoader(tmp_path)
    bus = InvalidationBus()
    return _make_handler(conventions_loader=loader, invalidation_bus=bus)


class TestConventionsLoaderWiring:
    """Phase 3 Task 9: ConventionsLoader injected into ConversationHandler."""

    def test_build_messages_inserts_conventions_user_message_when_mist_md_present(
        self, tmp_path: Path
    ) -> None:
        """When MIST.md exists in vault root, _build_messages inserts a user message
        containing the conventions content, positioned after all system messages
        and before the actual conversation history.
        """
        (tmp_path / "MIST.md").write_text("VAULT CONVENTIONS BODY", encoding="utf-8")
        loader = ConventionsLoader(tmp_path)
        handler = _make_handler(conventions_loader=loader)

        session = ConversationSession(session_id="p3-s1", user_id="raj")
        session.add_message("user", "hello")

        messages = handler._build_messages(
            session=session,
            max_history=10,
            retrieval_result=None,
            mist_context=None,
        )

        # The conventions user message must exist.
        conventions_msgs = [
            m for m in messages if m["role"] == "user" and "VAULT CONVENTIONS" in m["content"]
        ]
        assert len(conventions_msgs) == 1

        # It must come AFTER all system messages.
        sys_indices = [i for i, m in enumerate(messages) if m["role"] == "system"]
        conv_index = messages.index(conventions_msgs[0])
        assert conv_index > max(sys_indices)

        # It must come BEFORE the actual user message ("hello").
        user_non_conv_indices = [
            i
            for i, m in enumerate(messages)
            if m["role"] == "user" and "VAULT CONVENTIONS" not in m["content"]
        ]
        assert conv_index < min(user_non_conv_indices)

    def test_build_messages_omits_conventions_user_message_when_no_mist_md(
        self, tmp_path: Path
    ) -> None:
        """When MIST.md does not exist in vault root, _build_messages adds
        no conventions user message.
        """
        # tmp_path is empty -- no MIST.md or CLAUDE.md.
        loader = ConventionsLoader(tmp_path)
        handler = _make_handler(conventions_loader=loader)

        session = ConversationSession(session_id="p3-s2", user_id="raj")
        session.add_message("user", "hello")

        messages = handler._build_messages(
            session=session,
            max_history=10,
            retrieval_result=None,
            mist_context=None,
        )

        assert not any("VAULT CONVENTIONS" in m.get("content", "") for m in messages)


# ---------------------------------------------------------------------------
# Task 12: Static template vocabulary and invariant assertions
# ---------------------------------------------------------------------------


from backend.chat.conversation_handler import _STATIC_SYSTEM_TEMPLATE_BODY


def test_static_template_uses_notes_and_knowledge_graph_vocabulary():
    body = _STATIC_SYSTEM_TEMPLATE_BODY
    assert "NOTES (vault prose" in body
    assert "KNOWLEDGE GRAPH (typed triples" in body
    assert "REASONING substrate" in body
    assert "HISTORICAL and FACTUAL substrate" in body


def test_static_template_has_negative_invariants():
    body = _STATIC_SYSTEM_TEMPLATE_BODY
    assert "DO NOT call query_knowledge_graph when" in body
    # Specific exclusions
    assert "greetings" in body.lower()
    assert "general-knowledge" in body.lower()
    assert "creative" in body.lower()


def test_static_template_does_not_use_stale_graph_facts_phrase():
    body = _STATIC_SYSTEM_TEMPLATE_BODY
    assert "graph facts, document excerpts, or both" not in body
    # And does not use the unbounded "personalize based on what you know"
    assert "personalize based on what you know" not in body


def test_static_template_includes_decision_rule():
    body = _STATIC_SYSTEM_TEMPLATE_BODY
    assert "does the answer depend on user-specific structured knowledge" in body.lower()


# ---------------------------------------------------------------------------
# Task 13: query_knowledge_graph tool description assertions
# ---------------------------------------------------------------------------


from backend.chat.conversation_handler import KNOWLEDGE_TOOL_SCHEMAS

# The query_knowledge_graph tool is the first (and only) entry in KNOWLEDGE_TOOL_SCHEMAS.
_QKG_FUNCTION = KNOWLEDGE_TOOL_SCHEMAS[0]["function"]


def test_tool_description_uses_reasoning_substrate_framing():
    desc = _QKG_FUNCTION["description"]
    assert "reasoning substrate" in desc.lower()
    assert "multi-hop" in desc.lower() or "multi hop" in desc.lower()


def test_tool_description_includes_negative_use_cases():
    desc = _QKG_FUNCTION["description"]
    assert "DO NOT USE" in desc
    assert "greeting" in desc.lower()
    assert "general-knowledge" in desc.lower() or "general knowledge" in desc.lower()


def test_tool_description_does_not_use_unbounded_personalize_trigger():
    desc = _QKG_FUNCTION["description"]
    assert "personalize based on what you know" not in desc


# ---------------------------------------------------------------------------
# Fix B (P1 #5): "Default to NOT calling the tool" must be absent
# ---------------------------------------------------------------------------


def test_static_template_does_not_contain_default_to_not_phrase():
    """The stale 'Default to NOT calling the tool' bullet must not appear.

    That phrase directly contradicts the 'ask yourself' decision rule above it.
    On Gemma 4 E4B the literal 'Default to NOT' beat the abstract decision rule
    and suppressed legitimate USE cases.
    """
    body = _STATIC_SYSTEM_TEMPLATE_BODY
    assert "Default to NOT calling the tool" not in body


def test_static_template_contains_fabrication_risk_replacement():
    """The replacement bullet must reference fabrication risk as the tie-breaker.

    This aligns the closing guideline with the 'ask yourself' decision rule:
    call the tool when prose would otherwise force the model to infer
    user-specific facts not stated in the available context.
    """
    body = _STATIC_SYSTEM_TEMPLATE_BODY
    assert "fabrication" in body.lower()


# ---------------------------------------------------------------------------
# Task 21: ConversationHandler subscribes to InvalidationBus
# ---------------------------------------------------------------------------


class TestInvalidationBusSubscription:
    """Task 21: _on_vault_rebuild evicts _mist_context_cache on rebuild events.

    Coordination guarantee: filewatcher publishes AFTER graph rebuild completes,
    so the next mist_context fetch reads correct re-derived state.
    """

    def _pre_populate(self, handler: ConversationHandler, session_id: str, user_id: str) -> None:
        """Register a session and warm the mist_context cache for it.

        Mirrors the production flow: handle_message calls get_or_create_session
        then _get_or_fetch_mist_context. We drive both here so the cache entry
        and the sessions map are both populated.
        """
        handler.get_or_create_session(session_id, user_id)
        asyncio.run(handler._get_or_fetch_mist_context(session_id))

    def test_identity_edit_invalidates_all_session_caches(
        self, invalidation_handler: ConversationHandler
    ) -> None:
        """identity/mist.md rebuild clears ALL active session caches."""
        self._pre_populate(invalidation_handler, "s1", "raj")
        self._pre_populate(invalidation_handler, "s2", "alice")
        assert len(invalidation_handler._mist_context_cache) == 2

        event = RebuildResult(
            path=Path("/app/mist-memory/identity/mist.md"),
            bucket="1",
            orphaned_triple_count=1,
            new_triple_count=1,
            ontology_version="1.1.0",
            deferred=False,
        )
        asyncio.run(invalidation_handler._on_vault_rebuild(event))
        assert invalidation_handler._mist_context_cache == {}

    def test_user_edit_invalidates_only_matching_user_caches(
        self, invalidation_handler: ConversationHandler
    ) -> None:
        """users/raj.md rebuild evicts only sessions belonging to user 'raj'."""
        self._pre_populate(invalidation_handler, "s1", "raj")
        self._pre_populate(invalidation_handler, "s2", "alice")

        event = RebuildResult(
            path=Path("/app/mist-memory/users/raj.md"),
            bucket="1",
            orphaned_triple_count=0,
            new_triple_count=0,
            ontology_version="1.1.0",
            deferred=False,
        )
        asyncio.run(invalidation_handler._on_vault_rebuild(event))

        # raj's session cache is evicted; alice's survives.
        assert "s1" not in invalidation_handler._mist_context_cache
        assert "s2" in invalidation_handler._mist_context_cache

    def test_unrelated_path_edit_does_not_invalidate_cache(
        self, invalidation_handler: ConversationHandler
    ) -> None:
        """sessions/* rebuild events are a no-op for the mist_context cache."""
        self._pre_populate(invalidation_handler, "s1", "raj")

        event = RebuildResult(
            path=Path("/app/mist-memory/sessions/2026-05-10-test.md"),
            bucket="2",
            orphaned_triple_count=1,
            new_triple_count=0,
            ontology_version="1.1.0",
            deferred=True,
        )
        asyncio.run(invalidation_handler._on_vault_rebuild(event))

        # Cache untouched.
        assert "s1" in invalidation_handler._mist_context_cache

    def test_handler_subscribes_to_bus_on_init(self, tmp_path: Path) -> None:
        """When invalidation_bus is provided, _on_vault_rebuild is registered as listener."""
        bus = InvalidationBus()
        loader = ConventionsLoader(tmp_path)
        handler = _make_handler(conventions_loader=loader, invalidation_bus=bus)

        # Subscribe count must be exactly 1 (the handler's subscription).
        assert len(bus._listeners) == 1
        assert bus._listeners[0] == handler._on_vault_rebuild

    def test_handler_without_bus_is_valid(self, tmp_path: Path) -> None:
        """invalidation_bus=None (default) creates a handler without any subscription."""
        loader = ConventionsLoader(tmp_path)
        # Must not raise; bus wiring is optional.
        handler = _make_handler(conventions_loader=loader, invalidation_bus=None)
        assert handler._invalidation_bus is None
