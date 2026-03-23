"""Tests for EntityDeduplicator."""

import pytest

from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeGraphExecutor, FakeNeo4jConnection
from tests.unit.knowledge.curation.conftest import make_entity_dict


class TestExactIdMatch:
    @pytest.mark.asyncio
    async def test_merges_on_exact_id_match(self):
        from backend.knowledge.curation.confidence import ConfidenceManager
        from backend.knowledge.curation.deduplication import EntityDeduplicator

        conn = FakeNeo4jConnection(
            query_responses={
                "toLower(e.id)": [
                    {
                        "id": "python",
                        "entity_type": "Technology",
                        "display_name": "Python",
                        "aliases": ["py"],
                        "description": "A language",
                        "confidence": 0.80,
                        "source_type": "extracted",
                    }
                ],
            }
        )
        executor = FakeGraphExecutor(connection=conn)
        dedup = EntityDeduplicator(executor, FakeEmbeddingGenerator(), ConfidenceManager())

        entities = [make_entity_dict(entity_id="python", display_name="Python 3")]
        result = await dedup.deduplicate(entities)

        assert result.entities_merged == 1
        assert len(result.merge_actions) == 1
        assert result.merge_actions[0].existing_entity_id == "python"

    @pytest.mark.asyncio
    async def test_no_match_passes_through(self):
        from backend.knowledge.curation.confidence import ConfidenceManager
        from backend.knowledge.curation.deduplication import EntityDeduplicator

        conn = FakeNeo4jConnection()  # Empty graph
        executor = FakeGraphExecutor(connection=conn)
        dedup = EntityDeduplicator(executor, FakeEmbeddingGenerator(), ConfidenceManager())

        entities = [make_entity_dict(entity_id="rust", display_name="Rust")]
        result = await dedup.deduplicate(entities)

        assert result.entities_merged == 0
        assert len(result.merge_actions) == 0
        assert len(result.entities) == 1
        assert result.entities[0]["id"] == "rust"


class TestPropertyMerge:
    @pytest.mark.asyncio
    async def test_keeps_longer_display_name(self):
        from backend.knowledge.curation.confidence import ConfidenceManager
        from backend.knowledge.curation.deduplication import EntityDeduplicator

        conn = FakeNeo4jConnection(
            query_responses={
                "toLower(e.id)": [
                    {
                        "id": "python",
                        "entity_type": "Technology",
                        "display_name": "Python",
                        "aliases": [],
                        "description": "",
                        "confidence": 0.80,
                        "source_type": "extracted",
                    }
                ],
            }
        )
        executor = FakeGraphExecutor(connection=conn)
        dedup = EntityDeduplicator(executor, FakeEmbeddingGenerator(), ConfidenceManager())

        entities = [
            make_entity_dict(entity_id="python", display_name="Python Programming Language")
        ]
        result = await dedup.deduplicate(entities)

        assert result.merge_actions[0].merge_instructions["display_name"] == "keep_incoming"

    @pytest.mark.asyncio
    async def test_aliases_union(self):
        from backend.knowledge.curation.confidence import ConfidenceManager
        from backend.knowledge.curation.deduplication import EntityDeduplicator

        conn = FakeNeo4jConnection(
            query_responses={
                "toLower(e.id)": [
                    {
                        "id": "python",
                        "entity_type": "Technology",
                        "display_name": "Python",
                        "aliases": ["py"],
                        "description": "",
                        "confidence": 0.80,
                        "source_type": "extracted",
                    }
                ],
            }
        )
        executor = FakeGraphExecutor(connection=conn)
        dedup = EntityDeduplicator(executor, FakeEmbeddingGenerator(), ConfidenceManager())

        entities = [make_entity_dict(entity_id="python", aliases=["python3", "py"])]
        result = await dedup.deduplicate(entities)

        assert result.merge_actions[0].merge_instructions["aliases"] == "merge"


class TestEmptyInput:
    @pytest.mark.asyncio
    async def test_empty_entities_returns_empty_result(self):
        from backend.knowledge.curation.confidence import ConfidenceManager
        from backend.knowledge.curation.deduplication import EntityDeduplicator

        executor = FakeGraphExecutor()
        dedup = EntityDeduplicator(executor, FakeEmbeddingGenerator(), ConfidenceManager())

        result = await dedup.deduplicate([])
        assert result.entities_merged == 0
        assert result.entities == []
        assert result.merge_actions == []
