"""ADR-009: provenance writers use :__Provenance__ base label, not :__Entity__."""

from __future__ import annotations

import pytest

from backend.knowledge.curation.confidence import ConfidenceManager
from backend.knowledge.curation.graph_writer import CurationGraphWriter
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeGraphExecutor, FakeNeo4jConnection


@pytest.mark.asyncio
async def test_ensure_conversation_context_uses_provenance_label() -> None:
    # Arrange
    conn = FakeNeo4jConnection()
    executor = FakeGraphExecutor(connection=conn)
    writer = CurationGraphWriter(executor, FakeEmbeddingGenerator(), ConfidenceManager())

    # Act
    await writer._ensure_conversation_context(
        session_id="test-session-123",
        now="2026-04-17T00:00:00Z",
    )

    # Assert
    ctx_merges = [q for q, _ in conn.writes if "ConversationContext" in q and "MERGE" in q]
    assert ctx_merges, f"Expected a ConversationContext MERGE, got writes: {conn.writes}"
    merge_query = ctx_merges[0]
    assert (
        "__Provenance__:ConversationContext" in merge_query
    ), f"ADR-009: ConversationContext must carry :__Provenance__, got: {merge_query}"
    assert (
        "__Entity__:ConversationContext" not in merge_query
    ), f"ADR-009: ConversationContext must not carry :__Entity__, got: {merge_query}"
