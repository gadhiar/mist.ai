"""Tests for EntityNormalizer async migration."""

import pytest

from backend.knowledge.extraction.normalizer import EntityNormalizer
from backend.knowledge.extraction.ontology_extractor import ExtractionResult
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeGraphExecutor, FakeNeo4jConnection


class TestNormalizerAsync:
    @pytest.mark.asyncio
    async def test_normalize_is_async(self):
        normalizer = EntityNormalizer(
            embedding_generator=FakeEmbeddingGenerator(),
            executor=None,
        )
        extraction = ExtractionResult(
            entities=[{"id": "Python", "name": "Python", "type": "Technology"}],
            relationships=[],
        )
        result = await normalizer.normalize(extraction)
        assert result.entities[0]["id"] == "python"

    @pytest.mark.asyncio
    async def test_graph_dedup_via_executor(self):
        conn = FakeNeo4jConnection(
            query_responses={
                "toLower(e.id)": [{"id": "python", "entity_type": "Technology", "aliases": []}],
            }
        )
        executor = FakeGraphExecutor(connection=conn)
        normalizer = EntityNormalizer(
            embedding_generator=FakeEmbeddingGenerator(),
            executor=executor,
        )
        extraction = ExtractionResult(
            entities=[{"id": "Python", "name": "Python", "type": "Technology"}],
            relationships=[],
        )
        result = await normalizer.normalize(extraction)
        assert result.entities[0]["id"] == "python"
        assert len(conn.queries) >= 1

    @pytest.mark.asyncio
    async def test_normalize_without_executor_skips_graph(self):
        normalizer = EntityNormalizer(
            embedding_generator=FakeEmbeddingGenerator(),
            executor=None,
        )
        extraction = ExtractionResult(
            entities=[{"id": "JS", "name": "JS", "type": "Technology"}],
            relationships=[],
        )
        result = await normalizer.normalize(extraction)
        assert result.entities[0]["id"] == "javascript"
