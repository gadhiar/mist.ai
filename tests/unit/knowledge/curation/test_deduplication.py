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


# ---------------------------------------------------------------------------
# Tier 3: numeric/date name veto over the ordered candidate list
# ---------------------------------------------------------------------------


def _cand(entity_id: str, display_name: str | None, score: float) -> dict:
    return {
        "id": entity_id,
        "entity_type": "Technology",
        "display_name": display_name,
        "aliases": [],
        "description": "",
        "confidence": 0.8,
        "source_type": "extracted",
        "score": score,
    }


def _cosine_router(pool: list[dict]):
    """Answer the Tier-3 query the way Neo4j would, from the bound parameters.

    Filters on `score >= $threshold`, orders by (score DESC, id ASC) and applies
    `LIMIT $candidate_limit`. Tiers 1 and 2 fall through to the empty default.
    """

    def route(query, params):
        if "vector.similarity.cosine" not in query:
            return None
        kept = [c for c in pool if c["score"] >= params["threshold"]]
        kept.sort(key=lambda c: (-c["score"], c["id"]))
        return kept[: params["candidate_limit"]]

    return route


class TestTier3NameVeto:
    @pytest.mark.asyncio
    async def test_query_passes_threshold_and_bounded_limit(self):
        from backend.knowledge.curation.deduplication import (
            SIMILARITY_THRESHOLD,
            TIER3_CANDIDATE_LIMIT,
        )

        conn = FakeNeo4jConnection()
        dd, _ = _deduper(conn)

        await dd._find_existing("p95", "Technology", "P95")

        cosine = [(q, p) for q, p in conn.queries if "vector.similarity.cosine" in q]
        assert len(cosine) == 1
        query, params = cosine[0]
        assert SIMILARITY_THRESHOLD == 0.92
        assert params["threshold"] == 0.92
        assert "WHERE score >= $threshold" in query
        assert "ORDER BY score DESC, e.id ASC LIMIT $candidate_limit" in query
        assert params["candidate_limit"] == TIER3_CANDIDATE_LIMIT
        assert 1 <= TIER3_CANDIDATE_LIMIT <= 100

    @pytest.mark.asyncio
    async def test_top_candidate_vetoed_merges_into_second(self):
        conn = FakeNeo4jConnection(
            query_responses={
                "vector.similarity.cosine": [
                    _cand("p99-latency", "P99 latency", 0.98),
                    _cand("p95-latency", "P95 latency", 0.95),
                ]
            }
        )
        dd, _ = _deduper(conn)

        result = await dd.deduplicate(
            [make_entity_dict(entity_id="p95", display_name="P95", entity_type="Technology")]
        )

        # 'P95' vs 'P99 latency': {95} != {99} -> vetoed; 'P95 latency' -> {95} passes.
        assert result.entities_merged == 1
        assert result.merge_actions[0].existing_entity_id == "p95-latency"
        assert result.id_renames == {"p95": "p95-latency"}
        assert result.entities[0]["id"] == "p95-latency"

    @pytest.mark.asyncio
    async def test_candidates_are_taken_in_returned_order_not_resorted(self):
        conn = FakeNeo4jConnection(
            query_responses={
                "vector.similarity.cosine": [
                    _cand("b-first", "Q1 plan", 0.93),
                    _cand("a-second", "Q1 roadmap", 0.99),
                ]
            }
        )
        dd, _ = _deduper(conn)

        existing = await dd._find_existing("q1", "Technology", "Q1")

        assert existing is not None and existing["id"] == "b-first"

    @pytest.mark.asyncio
    async def test_all_candidates_vetoed_means_no_merge_and_no_rename(self):
        conn = FakeNeo4jConnection(
            query_responses={
                "vector.similarity.cosine": [
                    _cand("q2-2026", "Q2 2026", 0.99),
                    _cand("q1-2025", "Q1 2025", 0.97),
                    _cand("march-13-2026", "March 13, 2026", 0.96),
                ]
            }
        )
        dd, _ = _deduper(conn)

        entity = make_entity_dict(entity_id="q1", display_name="Q1")
        result = await dd.deduplicate([entity])

        assert result.entities_merged == 0
        assert result.merge_actions == []
        assert result.id_renames == {}
        assert result.entities[0]["id"] == "q1"

    @pytest.mark.asyncio
    async def test_null_display_name_falls_back_to_id(self):
        conn = FakeNeo4jConnection(
            query_responses={
                "vector.similarity.cosine": [
                    _cand("march-13", None, 0.99),
                    _cand("march-3", None, 0.95),
                ]
            }
        )
        dd, _ = _deduper(conn)

        existing = await dd._find_existing("mar-3", "Technology", "March 3")

        # id 'march-13' -> {13},{3} vetoed; id 'march-3' -> {3},{3} matches.
        assert existing is not None and existing["id"] == "march-3"

    @pytest.mark.asyncio
    async def test_score_ties_break_on_id_ascending(self):
        pool = [
            _cand("tie-c", "Grafana dashboards", 0.95),
            _cand("tie-a", "Grafana dashboard", 0.95),
            _cand("tie-b", "Grafana dashboard", 0.95),
        ]
        conn = FakeNeo4jConnection(query_router=_cosine_router(pool))
        dd, _ = _deduper(conn)

        existing = await dd._find_existing("grafana", "Technology", "Grafana")

        assert existing is not None and existing["id"] == "tie-a"

    @pytest.mark.asyncio
    async def test_vetoed_lowest_id_in_a_tie_passes_to_next_id(self):
        pool = [
            _cand("tie-c", "Sprint 12", 0.95),
            _cand("tie-a", "Sprint 13", 0.95),
            _cand("tie-b", "Sprint 12 review", 0.95),
            _cand("below", "Sprint 12", 0.91),
        ]
        conn = FakeNeo4jConnection(query_router=_cosine_router(pool))
        dd, _ = _deduper(conn)

        existing = await dd._find_existing("sprint-12", "Technology", "Sprint 12")

        # tie-a vetoed ({13} vs {12}); tie-b is next by id and passes.
        assert existing is not None and existing["id"] == "tie-b"

    @pytest.mark.asyncio
    async def test_candidate_below_threshold_is_never_considered(self):
        pool = [
            _cand("vetoed", "Sprint 13", 0.95),
            _cand("below", "Sprint 12", 0.9199),
        ]
        conn = FakeNeo4jConnection(query_router=_cosine_router(pool))
        dd, _ = _deduper(conn)

        existing = await dd._find_existing("sprint-12", "Technology", "Sprint 12")

        assert existing is None

    @pytest.mark.asyncio
    async def test_veto_does_not_apply_to_exact_id_tier(self):
        conn = FakeNeo4jConnection(
            query_responses={
                "toLower(e.id)": [_cand("p99", "P99 latency", 1.0)],
            }
        )
        dd, emb = _deduper(conn)

        existing = await dd._find_existing("p99", "Technology", "P95")

        assert existing is not None and existing["id"] == "p99"
        assert emb.calls == [], "Tier 3 must not run when Tier 1 matches"


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
