"""Tests for GraphHealthScorer."""

import pytest

from backend.knowledge.curation.health import (
    _TOTAL_ONTOLOGY_TYPES,
    _VALID_ENTITY_TYPES,
    _WEIGHTS,
    GraphHealthScorer,
    HealthScore,
)
from tests.mocks.neo4j import FakeGraphExecutor, FakeNeo4jConnection, FakeNeo4jRecord


def _build_connection(
    *,
    total: int = 0,
    fresh_count: int = 0,
    avg_confidence: float | None = None,
    entity_count: int = 0,
    rel_count: int = 0,
    type_counts: list[dict] | None = None,
    internal_count: int = 0,
) -> FakeNeo4jConnection:
    """Build a FakeNeo4jConnection pre-loaded with query responses."""
    responses: dict[str, list] = {
        "count(e) AS total": [FakeNeo4jRecord({"total": total})],
        "fresh_count": [FakeNeo4jRecord({"fresh_count": fresh_count})],
        "avg(e.confidence)": [FakeNeo4jRecord({"avg_confidence": avg_confidence})],
        "count(DISTINCT e)": [
            FakeNeo4jRecord({"entity_count": entity_count, "rel_count": rel_count})
        ],
        "e.entity_type AS entity_type": (
            [FakeNeo4jRecord(tc) for tc in type_counts] if type_counts else []
        ),
        "internal_count": [FakeNeo4jRecord({"internal_count": internal_count})],
    }
    return FakeNeo4jConnection(query_responses=responses)


class TestEmptyGraph:
    @pytest.mark.asyncio
    async def test_empty_graph_returns_zero(self):
        conn = _build_connection(total=0)
        scorer = GraphHealthScorer(FakeGraphExecutor(conn))

        result = await scorer.run()

        assert result.overall == 0.0
        assert result.freshness == 0.0
        assert result.confidence == 0.0
        assert result.connectivity == 0.0
        assert result.consistency == 0.0
        assert result.coverage == 0.0
        assert result.self_model == 0.0
        assert result.entity_count == 0
        assert result.relationship_count == 0


class TestWeights:
    def test_weights_sum_to_one(self):
        assert abs(sum(_WEIGHTS.values()) - 1.0) < 1e-9

    @pytest.mark.asyncio
    async def test_overall_is_weighted_average(self):
        type_counts = [
            {"entity_type": "Technology", "cnt": 5},
            {"entity_type": "Person", "cnt": 5},
        ]
        conn = _build_connection(
            total=10,
            fresh_count=10,
            avg_confidence=0.9,
            entity_count=10,
            rel_count=50,
            type_counts=type_counts,
            internal_count=5,
        )
        scorer = GraphHealthScorer(FakeGraphExecutor(conn))

        result = await scorer.run()

        expected = (
            _WEIGHTS["freshness"] * result.freshness
            + _WEIGHTS["confidence"] * result.confidence
            + _WEIGHTS["connectivity"] * result.connectivity
            + _WEIGHTS["consistency"] * result.consistency
            + _WEIGHTS["coverage"] * result.coverage
            + _WEIGHTS["self_model"] * result.self_model
        )
        assert abs(result.overall - expected) < 1e-9


class TestPerfectGraph:
    @pytest.mark.asyncio
    async def test_perfect_graph_near_100(self):
        all_types = list(_VALID_ENTITY_TYPES)
        type_counts = [{"entity_type": t, "cnt": 1} for t in all_types]
        total = len(all_types)

        conn = _build_connection(
            total=total,
            fresh_count=total,
            avg_confidence=1.0,
            entity_count=total,
            rel_count=total * 5,
            type_counts=type_counts,
            internal_count=5,
        )
        scorer = GraphHealthScorer(FakeGraphExecutor(conn))

        result = await scorer.run()

        assert result.freshness == 100.0
        assert result.confidence == 100.0
        assert result.connectivity == 100.0
        assert result.consistency == 100.0
        assert result.coverage == 100.0
        assert result.self_model == 100.0
        assert result.overall == 100.0


class TestIndividualScores:
    @pytest.mark.asyncio
    async def test_freshness_half(self):
        conn = _build_connection(
            total=10,
            fresh_count=5,
            avg_confidence=0.5,
            entity_count=10,
            rel_count=0,
            type_counts=[{"entity_type": "Technology", "cnt": 10}],
            internal_count=0,
        )
        scorer = GraphHealthScorer(FakeGraphExecutor(conn))

        result = await scorer.run()

        assert result.freshness == 50.0

    @pytest.mark.asyncio
    async def test_confidence_maps_to_percentage(self):
        conn = _build_connection(
            total=10,
            fresh_count=0,
            avg_confidence=0.75,
            entity_count=10,
            rel_count=0,
            type_counts=[{"entity_type": "Technology", "cnt": 10}],
            internal_count=0,
        )
        scorer = GraphHealthScorer(FakeGraphExecutor(conn))

        result = await scorer.run()

        assert result.confidence == 75.0

    @pytest.mark.asyncio
    async def test_connectivity_capped_at_100(self):
        conn = _build_connection(
            total=2,
            fresh_count=0,
            avg_confidence=0.5,
            entity_count=2,
            rel_count=100,
            type_counts=[{"entity_type": "Technology", "cnt": 2}],
            internal_count=0,
        )
        scorer = GraphHealthScorer(FakeGraphExecutor(conn))

        result = await scorer.run()

        assert result.connectivity == 100.0

    @pytest.mark.asyncio
    async def test_connectivity_scales_with_avg_rels(self):
        conn = _build_connection(
            total=4,
            fresh_count=0,
            avg_confidence=0.5,
            entity_count=4,
            rel_count=8,
            type_counts=[{"entity_type": "Technology", "cnt": 4}],
            internal_count=0,
        )
        scorer = GraphHealthScorer(FakeGraphExecutor(conn))

        result = await scorer.run()

        # avg_rels = 8/4 = 2, score = 2 * 20 = 40
        assert result.connectivity == 40.0

    @pytest.mark.asyncio
    async def test_consistency_with_invalid_types(self):
        type_counts = [
            {"entity_type": "Technology", "cnt": 6},
            {"entity_type": "BOGUS_TYPE", "cnt": 4},
        ]
        conn = _build_connection(
            total=10,
            fresh_count=0,
            avg_confidence=0.5,
            entity_count=10,
            rel_count=0,
            type_counts=type_counts,
            internal_count=0,
        )
        scorer = GraphHealthScorer(FakeGraphExecutor(conn))

        result = await scorer.run()

        # 6 valid out of 10
        assert result.consistency == 60.0

    @pytest.mark.asyncio
    async def test_coverage_partial(self):
        type_counts = [
            {"entity_type": "Technology", "cnt": 5},
            {"entity_type": "Person", "cnt": 5},
        ]
        conn = _build_connection(
            total=10,
            fresh_count=0,
            avg_confidence=0.5,
            entity_count=10,
            rel_count=0,
            type_counts=type_counts,
            internal_count=0,
        )
        scorer = GraphHealthScorer(FakeGraphExecutor(conn))

        result = await scorer.run()

        expected = 2 / _TOTAL_ONTOLOGY_TYPES * 100
        assert abs(result.coverage - expected) < 1e-9

    @pytest.mark.asyncio
    async def test_self_model_partial(self):
        conn = _build_connection(
            total=5,
            fresh_count=0,
            avg_confidence=0.5,
            entity_count=5,
            rel_count=0,
            type_counts=[{"entity_type": "Technology", "cnt": 5}],
            internal_count=3,
        )
        scorer = GraphHealthScorer(FakeGraphExecutor(conn))

        result = await scorer.run()

        # 3/5 * 100 = 60
        assert result.self_model == 60.0

    @pytest.mark.asyncio
    async def test_self_model_capped_at_100(self):
        conn = _build_connection(
            total=10,
            fresh_count=0,
            avg_confidence=0.5,
            entity_count=10,
            rel_count=0,
            type_counts=[{"entity_type": "Technology", "cnt": 10}],
            internal_count=20,
        )
        scorer = GraphHealthScorer(FakeGraphExecutor(conn))

        result = await scorer.run()

        assert result.self_model == 100.0


class TestScoreCapping:
    @pytest.mark.asyncio
    async def test_no_sub_score_exceeds_100(self):
        all_types = list(_VALID_ENTITY_TYPES)
        type_counts = [{"entity_type": t, "cnt": 10} for t in all_types]

        conn = _build_connection(
            total=len(all_types) * 10,
            fresh_count=len(all_types) * 10,
            avg_confidence=1.0,
            entity_count=len(all_types) * 10,
            rel_count=len(all_types) * 100,
            type_counts=type_counts,
            internal_count=50,
        )
        scorer = GraphHealthScorer(FakeGraphExecutor(conn))

        result = await scorer.run()

        assert result.freshness <= 100.0
        assert result.confidence <= 100.0
        assert result.connectivity <= 100.0
        assert result.consistency <= 100.0
        assert result.coverage <= 100.0
        assert result.self_model <= 100.0
        assert result.overall <= 100.0


class TestReturnType:
    @pytest.mark.asyncio
    async def test_returns_health_score_dataclass(self):
        conn = _build_connection(total=0)
        scorer = GraphHealthScorer(FakeGraphExecutor(conn))

        result = await scorer.run()

        assert isinstance(result, HealthScore)

    @pytest.mark.asyncio
    async def test_entity_and_rel_counts_populated(self):
        type_counts = [{"entity_type": "Technology", "cnt": 5}]
        conn = _build_connection(
            total=5,
            fresh_count=3,
            avg_confidence=0.8,
            entity_count=5,
            rel_count=10,
            type_counts=type_counts,
            internal_count=2,
        )
        scorer = GraphHealthScorer(FakeGraphExecutor(conn))

        result = await scorer.run()

        assert result.entity_count == 5
        assert result.relationship_count == 10
