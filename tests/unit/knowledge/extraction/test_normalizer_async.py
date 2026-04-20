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


class TestReservedNamespaceGuard:
    """Bug G: reserved names for the MIST system resolve to mist-identity."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "input_name,expected_canonical",
        [
            ("MIST", "mist-identity"),
            ("MIST.AI", "mist-identity"),
            ("MIST AI", "mist-identity"),
            ("mist", "mist-identity"),
            ("mist.ai", "mist-identity"),
            ("the AI", "mist-identity"),
            ("the assistant", "mist-identity"),
            ("The Assistant", "mist-identity"),
        ],
    )
    async def test_reserved_name_maps_to_mist_identity(self, input_name, expected_canonical):
        normalizer = EntityNormalizer(
            embedding_generator=FakeEmbeddingGenerator(),
            executor=None,
        )
        extraction = ExtractionResult(
            entities=[{"id": input_name, "name": input_name, "type": "Organization"}],
            relationships=[],
        )
        result = await normalizer.normalize(extraction)
        assert result.entities[0]["id"] == expected_canonical, (
            f"Expected {input_name} to canonicalize to {expected_canonical}, "
            f"got {result.entities[0]['id']}"
        )

    @pytest.mark.asyncio
    async def test_non_reserved_name_is_unchanged(self):
        """Sanity check: ordinary names do NOT get remapped."""
        normalizer = EntityNormalizer(
            embedding_generator=FakeEmbeddingGenerator(),
            executor=None,
        )
        extraction = ExtractionResult(
            entities=[{"id": "mistletoe", "name": "Mistletoe", "type": "Concept"}],
            relationships=[],
        )
        result = await normalizer.normalize(extraction)
        assert result.entities[0]["id"] == "mistletoe"

    @pytest.mark.asyncio
    async def test_reserved_name_logs_warning(self, caplog):
        import logging

        normalizer = EntityNormalizer(
            embedding_generator=FakeEmbeddingGenerator(),
            executor=None,
        )
        extraction = ExtractionResult(
            entities=[{"id": "MIST", "name": "MIST", "type": "Organization"}],
            relationships=[],
        )
        with caplog.at_level(logging.WARNING, logger="backend.knowledge.extraction.normalizer"):
            await normalizer.normalize(extraction)
        assert any(
            "reserved name" in r.message.lower() for r in caplog.records
        ), f"Expected reserved-name warning, got logs: {[r.message for r in caplog.records]}"
