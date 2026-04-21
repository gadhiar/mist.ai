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
            ("mist-ai", "mist-identity"),
            ("MIST-AI", "mist-identity"),
            ("the AI", "mist-identity"),
            ("the-ai", "mist-identity"),
            ("the assistant", "mist-identity"),
            ("The Assistant", "mist-identity"),
            ("the-assistant", "mist-identity"),
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


class TestReservedNameTypeRemap:
    """Cluster 1: reserved-name matches must override entity_type to MistIdentity.

    An LLM frequently labels 'MIST' or 'MIST.AI' as Organization. Cluster 1
    validator constraints require the mist-identity node to carry the
    MistIdentity label for IMPLEMENTED_WITH / MIST_HAS_CAPABILITY /
    MIST_HAS_TRAIT / MIST_HAS_PREFERENCE edges. The normalizer must rewrite
    both id AND type so the graph-writer produces a validator-compliant node.
    """

    @pytest.mark.asyncio
    async def test_mist_name_remaps_id_and_type(self):
        # Arrange
        normalizer = EntityNormalizer(
            embedding_generator=FakeEmbeddingGenerator(),
            executor=None,
        )
        extraction = ExtractionResult(
            entities=[{"id": "mist", "name": "MIST", "type": "Organization"}],
            relationships=[],
        )

        # Act
        result = await normalizer.normalize(extraction)

        # Assert
        assert result.entities[0]["id"] == "mist-identity"
        assert result.entities[0]["type"] == "MistIdentity"

    @pytest.mark.asyncio
    async def test_mist_dot_ai_remaps(self):
        # Arrange
        normalizer = EntityNormalizer(
            embedding_generator=FakeEmbeddingGenerator(),
            executor=None,
        )
        extraction = ExtractionResult(
            entities=[{"id": "mist-ai-1", "name": "MIST.AI", "type": "Organization"}],
            relationships=[],
        )

        # Act
        result = await normalizer.normalize(extraction)

        # Assert
        assert result.entities[0]["id"] == "mist-identity"
        assert result.entities[0]["type"] == "MistIdentity"

    @pytest.mark.asyncio
    async def test_the_ai_remaps(self):
        # Arrange
        normalizer = EntityNormalizer(
            embedding_generator=FakeEmbeddingGenerator(),
            executor=None,
        )
        extraction = ExtractionResult(
            entities=[{"id": "ai-agent", "name": "the AI", "type": "Concept"}],
            relationships=[],
        )

        # Act
        result = await normalizer.normalize(extraction)

        # Assert
        assert result.entities[0]["id"] == "mist-identity"
        assert result.entities[0]["type"] == "MistIdentity"

    @pytest.mark.asyncio
    async def test_relationships_forwarded_after_remap(self):
        # Arrange
        normalizer = EntityNormalizer(
            embedding_generator=FakeEmbeddingGenerator(),
            executor=None,
        )
        extraction = ExtractionResult(
            entities=[
                {"id": "mist", "name": "MIST", "type": "Organization"},
                {"id": "lancedb", "name": "LanceDB", "type": "Technology"},
            ],
            relationships=[
                {"source": "mist", "target": "lancedb", "type": "USES"},
            ],
        )

        # Act
        result = await normalizer.normalize(extraction)

        # Assert
        mist_entity = next(e for e in result.entities if e["id"] == "mist-identity")
        assert mist_entity["type"] == "MistIdentity"
        assert len(result.relationships) == 1
        assert result.relationships[0]["source"] == "mist-identity"
        assert result.relationships[0]["target"] == "lancedb"
        assert result.relationships[0]["type"] == "USES"

    @pytest.mark.asyncio
    async def test_non_reserved_entities_unchanged(self):
        # Arrange
        normalizer = EntityNormalizer(
            embedding_generator=FakeEmbeddingGenerator(),
            executor=None,
        )
        extraction = ExtractionResult(
            entities=[{"id": "python", "name": "Python", "type": "Technology"}],
            relationships=[],
        )

        # Act
        result = await normalizer.normalize(extraction)

        # Assert
        assert result.entities[0]["id"] == "python"
        assert result.entities[0]["type"] == "Technology"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "input_name",
        [
            pytest.param("Mist", id="title-case"),
            pytest.param("MIST", id="upper-case"),
            pytest.param("mist", id="lower-case"),
        ],
    )
    async def test_reserved_name_case_insensitive(self, input_name):
        # Arrange
        normalizer = EntityNormalizer(
            embedding_generator=FakeEmbeddingGenerator(),
            executor=None,
        )
        extraction = ExtractionResult(
            entities=[{"id": input_name, "name": input_name, "type": "Organization"}],
            relationships=[],
        )

        # Act
        result = await normalizer.normalize(extraction)

        # Assert
        assert result.entities[0]["id"] == "mist-identity"
        assert result.entities[0]["type"] == "MistIdentity"

    @pytest.mark.asyncio
    async def test_reserved_name_with_trailing_whitespace_remaps(self):
        # Arrange
        normalizer = EntityNormalizer(
            embedding_generator=FakeEmbeddingGenerator(),
            executor=None,
        )
        extraction = ExtractionResult(
            entities=[{"id": "whitespace-mist", "name": "  mist  ", "type": "Organization"}],
            relationships=[],
        )

        # Act
        result = await normalizer.normalize(extraction)

        # Assert
        assert result.entities[0]["id"] == "mist-identity"
        assert result.entities[0]["type"] == "MistIdentity"
