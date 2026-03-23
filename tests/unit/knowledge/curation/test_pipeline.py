"""Tests for CurationPipeline orchestrator."""

import pytest

from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeGraphExecutor, FakeNeo4jConnection
from tests.unit.knowledge.curation.conftest import (
    make_entity_dict,
    make_relationship_dict,
    make_validation_result,
)


def _build_pipeline(*, connection=None):
    """Build a CurationPipeline with fakes for testing."""
    from backend.knowledge.curation.confidence import ConfidenceManager
    from backend.knowledge.curation.conflict_resolver import ConflictResolver
    from backend.knowledge.curation.deduplication import EntityDeduplicator
    from backend.knowledge.curation.graph_writer import CurationGraphWriter
    from backend.knowledge.curation.pipeline import CurationPipeline

    conn = connection or FakeNeo4jConnection()
    executor = FakeGraphExecutor(connection=conn)
    embeddings = FakeEmbeddingGenerator()
    confidence = ConfidenceManager()

    return (
        CurationPipeline(
            deduplicator=EntityDeduplicator(executor, embeddings, confidence),
            conflict_resolver=ConflictResolver(executor),
            graph_writer=CurationGraphWriter(executor, embeddings, confidence),
        ),
        conn,
    )


class TestFullFlow:
    @pytest.mark.asyncio
    async def test_curate_and_store_full_pipeline(self):
        from backend.knowledge.curation.pipeline import CurationResult

        pipeline, conn = _build_pipeline()

        validation = make_validation_result(
            entities=[make_entity_dict()],
            relationships=[make_relationship_dict()],
        )

        result = await pipeline.curate_and_store(validation, "evt-001", "sess-001")

        assert isinstance(result, CurationResult)
        assert result.write_result.entities_created >= 1
        assert result.curation_time_ms > 0


class TestShortCircuit:
    @pytest.mark.asyncio
    async def test_short_circuits_on_empty_entities(self):
        from backend.knowledge.curation.pipeline import CurationResult

        pipeline, conn = _build_pipeline()

        validation = make_validation_result(entities=[], relationships=[])
        result = await pipeline.curate_and_store(validation, "evt-001", "sess-001")

        assert isinstance(result, CurationResult)
        assert result.write_result.entities_created == 0
        assert result.write_result.relationships_created == 0
        conn.assert_no_writes()


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_dedup_failure_logs_and_continues(self):
        """Dedup failure produces partial result -- pipeline does not crash."""
        from backend.knowledge.curation.confidence import ConfidenceManager
        from backend.knowledge.curation.conflict_resolver import ConflictResolver
        from backend.knowledge.curation.deduplication import EntityDeduplicator
        from backend.knowledge.curation.graph_writer import CurationGraphWriter
        from backend.knowledge.curation.pipeline import CurationPipeline

        conn = FakeNeo4jConnection()
        executor = FakeGraphExecutor(connection=conn)
        embeddings = FakeEmbeddingGenerator()
        confidence = ConfidenceManager()

        # Create a FakeEntityDeduplicator that raises
        class FailingDeduplicator(EntityDeduplicator):
            async def deduplicate(self, entities):
                raise RuntimeError("dedup failed")

        pipeline = CurationPipeline(
            deduplicator=FailingDeduplicator(executor, embeddings, confidence),
            conflict_resolver=ConflictResolver(executor),
            graph_writer=CurationGraphWriter(executor, embeddings, confidence),
        )

        validation = make_validation_result(entities=[make_entity_dict()])

        # Pipeline catches the error and returns a partial result
        result = await pipeline.curate_and_store(validation, "evt-001", "sess-001")
        assert result.dedup_result.entities_merged == 0
        assert len(result.stage_errors) == 1
        assert "dedup" in result.stage_errors[0].lower()
