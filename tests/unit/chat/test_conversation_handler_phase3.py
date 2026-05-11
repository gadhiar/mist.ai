"""Phase 3 additions to ConversationHandler tests.

Tests wiring of ConventionsLoader into _build_messages (Task 9).
"""

from pathlib import Path

from backend.chat.conversation_handler import ConversationHandler
from backend.knowledge.models import ConversationSession
from backend.knowledge.retrieval.knowledge_retriever import KnowledgeRetriever
from backend.knowledge.storage.graph_store import GraphStore
from backend.vault.conventions import ConventionsLoader
from tests.mocks.config import build_test_config
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection
from tests.mocks.ollama import FakeLLM


class FakeExtractionPipelineP3:
    """Minimal extraction pipeline test double."""

    async def extract_from_utterance(self, **kwargs):
        from backend.knowledge.extraction.validator import ValidationResult

        return ValidationResult(valid=True, entities=[], relationships=[])


def _make_handler(*, conventions_loader: ConventionsLoader) -> ConversationHandler:
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
    )


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
