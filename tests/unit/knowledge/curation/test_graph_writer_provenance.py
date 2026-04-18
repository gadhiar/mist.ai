"""ADR-009: provenance writers use :__Provenance__ base label, not :__Entity__."""

from __future__ import annotations

import pytest

from backend.knowledge.curation.confidence import ConfidenceManager
from backend.knowledge.curation.conflict_resolver import SupersessionAction
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


@pytest.mark.asyncio
async def test_create_learning_event_uses_provenance_label() -> None:
    # Arrange
    conn = FakeNeo4jConnection()
    executor = FakeGraphExecutor(connection=conn)
    writer = CurationGraphWriter(executor, FakeEmbeddingGenerator(), ConfidenceManager())

    action = SupersessionAction(
        old_rel_type="KNOWS",
        old_target_id="entity-old-001",
        new_target_id="entity-new-002",
        reason="contradiction",
    )

    # Act
    await writer._create_learning_event(
        action=action,
        session_id="test-session-456",
        event_id="evt-001",
        now="2026-04-17T00:00:00Z",
        source_metadata=None,
    )

    # Assert: LearningEvent MERGE uses :__Provenance__, not :__Entity__
    le_merges = [q for q, _ in conn.writes if "LearningEvent" in q and "MERGE" in q]
    assert le_merges, f"Expected a LearningEvent MERGE, got writes: {conn.writes}"
    merge_query = le_merges[0]
    assert (
        "__Provenance__:LearningEvent" in merge_query
    ), f"ADR-009: LearningEvent must carry :__Provenance__, got: {merge_query}"
    assert (
        "__Entity__:LearningEvent" not in merge_query
    ), f"ADR-009: LearningEvent must not carry :__Entity__, got: {merge_query}"

    # Assert: ABOUT-edge target query keeps :__Entity__ (cross-layer edge semantics)
    about_matches = [q for q, _ in conn.writes if "ABOUT" in q and "MATCH" in q]
    assert about_matches, f"Expected an ABOUT MATCH query, got writes: {conn.writes}"
    about_query = about_matches[0]
    assert (
        ":__Entity__ {id:" in about_query or "target:__Entity__" in about_query
    ), f"ADR-009: ABOUT-edge target must retain :__Entity__ filter, got: {about_query}"


@pytest.mark.asyncio
async def test_create_new_fact_learning_event_uses_provenance_label() -> None:
    # Arrange
    conn = FakeNeo4jConnection()
    executor = FakeGraphExecutor(connection=conn)
    writer = CurationGraphWriter(executor, FakeEmbeddingGenerator(), ConfidenceManager())

    # Act
    await writer._create_new_fact_learning_event(
        entity_id="entity-new-001",
        session_id="test-session-789",
        event_id="evt-002",
        now="2026-04-17T00:00:00Z",
        source_type="conversation",
        source_metadata=None,
    )

    # Assert: LearningEvent MERGE uses :__Provenance__, not :__Entity__
    le_merges = [q for q, _ in conn.writes if "LearningEvent" in q and "MERGE" in q]
    assert le_merges, f"Expected a LearningEvent MERGE, got writes: {conn.writes}"
    merge_query = le_merges[0]
    assert (
        "__Provenance__:LearningEvent" in merge_query
    ), f"ADR-009: LearningEvent must carry :__Provenance__, got: {merge_query}"
    assert (
        "__Entity__:LearningEvent" not in merge_query
    ), f"ADR-009: LearningEvent must not carry :__Entity__, got: {merge_query}"

    # Assert: ABOUT-edge target query keeps :__Entity__ (cross-layer edge semantics)
    about_matches = [q for q, _ in conn.writes if "ABOUT" in q and "MATCH" in q]
    assert about_matches, f"Expected an ABOUT MATCH query, got writes: {conn.writes}"
    about_query = about_matches[0]
    assert (
        ":__Entity__ {id:" in about_query or "target:__Entity__" in about_query
    ), f"ADR-009: ABOUT-edge target must retain :__Entity__ filter, got: {about_query}"


@pytest.mark.asyncio
async def test_ensure_external_source_uses_provenance_label() -> None:
    # Arrange
    conn = FakeNeo4jConnection()
    executor = FakeGraphExecutor(connection=conn)
    writer = CurationGraphWriter(executor, FakeEmbeddingGenerator(), ConfidenceManager())

    from backend.knowledge.curation.graph_writer import SourceMetadata

    source = SourceMetadata(
        source_uri="https://example.com/doc.pdf",
        source_type="document",
        title="Test Document",
    )

    # Act
    await writer._ensure_external_source(
        source_metadata=source,
        now="2026-04-17T00:00:00Z",
    )

    # Assert
    es_merges = [q for q, _ in conn.writes if "ExternalSource" in q and "MERGE" in q]
    assert es_merges, f"Expected an ExternalSource MERGE, got writes: {conn.writes}"
    merge_query = es_merges[0]
    assert (
        "__Provenance__:ExternalSource" in merge_query
    ), f"ADR-009: ExternalSource must carry :__Provenance__, got: {merge_query}"
    assert (
        "__Entity__:ExternalSource" not in merge_query
    ), f"ADR-009: ExternalSource must not carry :__Entity__, got: {merge_query}"


@pytest.mark.asyncio
async def test_ensure_vector_chunks_uses_provenance_label() -> None:
    # Arrange
    conn = FakeNeo4jConnection()
    executor = FakeGraphExecutor(connection=conn)
    writer = CurationGraphWriter(executor, FakeEmbeddingGenerator(), ConfidenceManager())

    # Act
    await writer._ensure_vector_chunks(
        chunk_ids=["chunk-001", "chunk-002"],
        source_uri="https://example.com/doc.pdf",
        now="2026-04-17T00:00:00Z",
    )

    # Assert
    vc_merges = [q for q, _ in conn.writes if "VectorChunk" in q and "MERGE" in q]
    assert vc_merges, f"Expected a VectorChunk MERGE, got writes: {conn.writes}"
    merge_query = vc_merges[0]
    assert (
        "__Provenance__:VectorChunk" in merge_query
    ), f"ADR-009: VectorChunk must carry :__Provenance__, got: {merge_query}"
    assert (
        "__Entity__:VectorChunk" not in merge_query
    ), f"ADR-009: VectorChunk must not carry :__Entity__, got: {merge_query}"
