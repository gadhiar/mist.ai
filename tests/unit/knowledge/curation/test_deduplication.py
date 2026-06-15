"""Tests for EntityDeduplicator."""

import pytest

from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeGraphExecutor, FakeNeo4jConnection
from tests.unit.knowledge.curation.conftest import make_entity_dict

# ---------------------------------------------------------------------------
# Helpers shared by the resolver shape tests
# ---------------------------------------------------------------------------


def _deduper(conn: FakeNeo4jConnection):
    from backend.knowledge.curation.confidence import ConfidenceManager
    from backend.knowledge.curation.deduplication import EntityDeduplicator

    emb = FakeEmbeddingGenerator()
    dd = EntityDeduplicator(
        executor=FakeGraphExecutor(connection=conn),
        embedding_provider=emb,
        confidence_manager=ConfidenceManager(),
    )
    return dd, emb


# ---------------------------------------------------------------------------
# Resolver determinism: ORDER BY on all tiers, exact-cosine (no ANN), probe
# ---------------------------------------------------------------------------


class TestDeterministicResolver:
    @pytest.mark.asyncio
    async def test_exact_and_alias_tiers_order_by_id_for_total_order(self):
        conn = FakeNeo4jConnection()
        dd, _ = _deduper(conn)

        await dd._find_existing("python", "Technology", "Python")

        exact_alias = [
            q for q, _ in conn.queries if "toLower(e.id)" in q or "IN [a IN e.aliases" in q
        ]
        assert exact_alias, "expected exact + alias tier queries"
        assert all("ORDER BY e.id ASC" in q for q in exact_alias), exact_alias

    @pytest.mark.asyncio
    async def test_similarity_tier_uses_exact_cosine_not_ann(self):
        conn = FakeNeo4jConnection()
        dd, _ = _deduper(conn)

        await dd._find_existing("python", "Technology", "Python")

        cosine_q = [q for q, _ in conn.queries if "vector.similarity.cosine" in q]
        assert cosine_q, "expected an exact-cosine query"
        q = cosine_q[0]
        assert "db.index.vector.queryNodes" not in q, "ANN must be gone from the resolver"
        assert "ORDER BY score DESC, e.id ASC" in q, q

    @pytest.mark.asyncio
    async def test_probe_embeds_display_name_not_id(self):
        conn = FakeNeo4jConnection()
        dd, emb = _deduper(conn)

        await dd._find_existing("py", "Technology", "Python")

        assert "Python" in emb.calls, f"probe must embed display_name, got calls: {emb.calls}"
        assert "py" not in emb.calls, "probe must NOT embed the id"


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

    @pytest.mark.asyncio
    async def test_rename_map_captures_old_id_before_rewrite(self):
        # deep review recon-engine-3(c): the in-place entity['id'] rewrite
        # destroys the old id, so relationships referencing it would point at
        # a nonexistent node; the rename map is the pipeline's remap source.
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
        dedup = EntityDeduplicator(
            FakeGraphExecutor(connection=conn), FakeEmbeddingGenerator(), ConfidenceManager()
        )

        entities = [make_entity_dict(entity_id="py", display_name="Python")]
        result = await dedup.deduplicate(entities)

        assert result.id_renames == {"py": "python"}
        assert result.entities[0]["id"] == "python"

    @pytest.mark.asyncio
    async def test_rename_map_empty_when_ids_already_match(self):
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
        dedup = EntityDeduplicator(
            FakeGraphExecutor(connection=conn), FakeEmbeddingGenerator(), ConfidenceManager()
        )

        result = await dedup.deduplicate([make_entity_dict(entity_id="python")])

        assert result.entities_merged == 1
        assert result.id_renames == {}


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
