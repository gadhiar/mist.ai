"""R1.3.1 guard: the per-turn path performs no vault write.

Traps at the side-effect boundary (L19) rather than asserting a method does
not exist -- the latter passes if the behavior returns under a new name.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from backend.chat.conversation_handler import ConversationHandler
from backend.knowledge.extraction.validator import ValidationResult
from backend.knowledge.retrieval.knowledge_retriever import KnowledgeRetriever
from backend.knowledge.storage.graph_store import GraphStore
from tests.mocks.config import build_test_config
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection
from tests.mocks.ollama import FakeLLM
from tests.unit.conftest import make_test_conventions_loader

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _RecordingVaultWriter:
    """Records every write attempt. Never raises, so a swallowed failure
    cannot masquerade as 'no write occurred'.

    `session_path` is a genuinely synchronous, non-enqueued method on the
    real `VaultWriter` (pure path computation, no queue involved) -- built
    explicitly here rather than through `__getattr__` per L20, because the
    catch-all returns an async closure that `_get_or_allocate_vault_path`
    would call without awaiting, silently caching a coroutine object as the
    session's vault path instead of a string.
    """

    def __init__(self) -> None:
        self.writes: list[tuple[str, dict]] = []

    def session_path(self, session_date: str, session_slug: str) -> str:
        return f"/tmp/vault/sessions/{session_date}-{session_slug}.md"

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)

        async def _record(*args, **kwargs):
            self.writes.append((name, kwargs))
            return None

        return _record


class _FakeExtractionPipeline:
    """Minimal extraction pipeline double, shaped like the real
    `ValidationResult` (L20: built from the real collaborator's fields, not
    invented ones). Empty entities/relationships by default so the C-pattern
    user-snapshot trigger (`_maybe_refresh_user_vault`) never fires and
    cannot contaminate the write-count assertions below.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def extract_from_utterance(self, **kwargs):
        self.calls.append(kwargs)
        return ValidationResult(valid=True, entities=[], relationships=[])


# ---------------------------------------------------------------------------
# Fixture -- mirrors the make_handler() construction pattern from
# tests/unit/chat/test_conversation_handler_vault_integration.py
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def conversation_handler_with_recording_writer():
    conn = FakeNeo4jConnection()
    gs = GraphStore(conn, FakeEmbeddingGenerator())
    config = build_test_config(
        event_store_enabled=True,
        event_store_db_path=":memory:",
    )
    writer = _RecordingVaultWriter()
    handler = ConversationHandler(
        config=config,
        graph_store=gs,
        extraction_pipeline=_FakeExtractionPipeline(),
        retriever=KnowledgeRetriever(config=config, graph_store=gs),
        llm_provider=FakeLLM(),
        conventions_loader=make_test_conventions_loader(),
        vault_writer=writer,
    )
    yield handler, writer


# ---------------------------------------------------------------------------
# Guard tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_message_performs_no_vault_write(conversation_handler_with_recording_writer):
    """A conversational turn writes to the event store and the graph.
    The vault is written at session end only.
    """
    handler, writer = conversation_handler_with_recording_writer

    await handler.handle_message("I use Python for data-intensive systems", session_id="s-1")
    await handler._drain_extraction_tasks()

    assert (
        writer.writes == []
    ), f"R1.3.1: no vault write may occur on the per-turn path; got {writer.writes}"


@pytest.mark.asyncio
async def test_end_session_writes_exactly_one_session_note(
    conversation_handler_with_recording_writer,
):
    handler, writer = conversation_handler_with_recording_writer

    await handler.handle_message("first turn about Python", session_id="s-2")
    await handler.handle_message("second turn about Neo4j", session_id="s-2")
    await handler._drain_extraction_tasks()
    await handler.end_session(session_id="s-2")

    names = [name for name, _ in writer.writes]
    assert names.count("write_session_note") == 1
    assert "append_turn_to_session" not in names
    assert "append_session_synthesis" not in names
    assert "mark_session_completed" not in names
