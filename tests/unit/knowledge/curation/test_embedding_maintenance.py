"""Tests for EmbeddingMaintenance."""

from __future__ import annotations

import pytest

from backend.knowledge.curation.embedding_maintenance import (
    EmbeddingMaintenance,
    EmbeddingMaintenanceResult,
    _build_embedding_text,
)
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeGraphExecutor, FakeNeo4jConnection


def _entity(
    *,
    entity_id: str = "e1",
    display_name: str = "Python",
    entity_type: str = "Technology",
    description: str = "A programming language",
) -> dict:
    return {
        "id": entity_id,
        "display_name": display_name,
        "entity_type": entity_type,
        "description": description,
    }


class TestEmbeddingText:
    def test_builds_text_from_all_fields(self):
        text = _build_embedding_text("Python", "Technology", "A language")
        assert text == "Python Technology A language"

    def test_omits_empty_description(self):
        text = _build_embedding_text("Python", "Technology", "")
        assert text == "Python Technology"


class TestRegenerateEmbeddings:
    @pytest.mark.asyncio
    async def test_regenerates_stale_embeddings(self):
        """Entities with stale embeddings get new vectors written."""
        rec = _entity()
        conn = FakeNeo4jConnection(query_results=[rec])
        executor = FakeGraphExecutor(connection=conn)
        embedder = FakeEmbeddingGenerator()
        job = EmbeddingMaintenance(executor, embedder)

        result = await job.run()

        assert result.entities_scanned == 1
        assert result.embeddings_regenerated == 1
        assert result.duration_ms > 0

        # Verify embedding was generated from correct text
        assert len(embedder.calls) == 1
        assert "Python" in embedder.calls[0]
        assert "Technology" in embedder.calls[0]

    @pytest.mark.asyncio
    async def test_writes_embedding_and_timestamp(self):
        """The write query should set both embedding and embedding_updated_at."""
        rec = _entity()
        conn = FakeNeo4jConnection(query_results=[rec])
        executor = FakeGraphExecutor(connection=conn)
        embedder = FakeEmbeddingGenerator()
        job = EmbeddingMaintenance(executor, embedder)

        await job.run()

        assert len(conn.writes) == 1
        query_str, params = conn.writes[0]
        assert "embedding" in query_str
        assert "embedding_updated_at" in query_str
        assert params["id"] == "e1"
        assert isinstance(params["embedding"], list)

    @pytest.mark.asyncio
    async def test_processes_multiple_entities(self):
        """All stale entities should be processed."""
        records = [
            _entity(entity_id="e1", display_name="Python"),
            _entity(entity_id="e2", display_name="JavaScript"),
            _entity(entity_id="e3", display_name="Rust"),
        ]
        conn = FakeNeo4jConnection(query_results=records)
        executor = FakeGraphExecutor(connection=conn)
        embedder = FakeEmbeddingGenerator()
        job = EmbeddingMaintenance(executor, embedder)

        result = await job.run()

        assert result.entities_scanned == 3
        assert result.embeddings_regenerated == 3
        assert len(conn.writes) == 3
        assert len(embedder.calls) == 3


class TestNoStaleEntities:
    @pytest.mark.asyncio
    async def test_empty_graph_returns_zero_counts(self):
        """When no entities need updates, counts should be zero."""
        conn = FakeNeo4jConnection(query_results=[])
        executor = FakeGraphExecutor(connection=conn)
        embedder = FakeEmbeddingGenerator()
        job = EmbeddingMaintenance(executor, embedder)

        result = await job.run()

        assert result == EmbeddingMaintenanceResult(
            entities_scanned=0,
            embeddings_regenerated=0,
            duration_ms=result.duration_ms,
        )
        conn.assert_no_writes()
        assert len(embedder.calls) == 0


class TestNullFields:
    @pytest.mark.asyncio
    async def test_handles_null_description(self):
        """Entity with None description should still generate embedding."""
        rec = _entity()
        rec["description"] = None
        conn = FakeNeo4jConnection(query_results=[rec])
        executor = FakeGraphExecutor(connection=conn)
        embedder = FakeEmbeddingGenerator()
        job = EmbeddingMaintenance(executor, embedder)

        result = await job.run()

        assert result.embeddings_regenerated == 1
        # Should use display_name + entity_type only
        assert embedder.calls[0] == "Python Technology"
