"""Tests for StalenessDetector."""

import pytest

from backend.knowledge.curation.staleness import StalenessDetector, StalenessResult
from tests.mocks.neo4j import FakeGraphExecutor, FakeNeo4jConnection


def _entity_row(
    *,
    entity_id: str = "e1",
    display_name: str = "Entity",
    entity_type: str = "Technology",
    confidence: float = 0.85,
    last_updated: str = "2026-01-01T00:00:00",
) -> dict:
    """Build a row matching the staleness query projection."""
    return {
        "id": entity_id,
        "display_name": display_name,
        "entity_type": entity_type,
        "confidence": confidence,
        "last_updated": last_updated,
    }


class TestCategorizesByConfidenceTier:
    @pytest.mark.asyncio
    async def test_categorizes_by_confidence_tier(self):
        rows = [
            _entity_row(entity_id="active1", confidence=0.9),
            _entity_row(entity_id="active2", confidence=0.6),
            _entity_row(entity_id="stale1", confidence=0.55),
            _entity_row(entity_id="stale2", confidence=0.4),
            _entity_row(entity_id="very_stale1", confidence=0.35),
            _entity_row(entity_id="very_stale2", confidence=0.2),
            _entity_row(entity_id="moribund1", confidence=0.1),
        ]
        conn = FakeNeo4jConnection(query_results=rows)
        executor = FakeGraphExecutor(connection=conn)
        detector = StalenessDetector(executor)

        result = await detector.run()

        assert result.active_count == 2
        assert result.stale_count == 2
        assert result.very_stale_count == 2
        # Moribund (< 0.2) not counted in any of the three buckets

    @pytest.mark.asyncio
    async def test_boundary_060_is_active(self):
        rows = [_entity_row(confidence=0.6)]
        conn = FakeNeo4jConnection(query_results=rows)
        executor = FakeGraphExecutor(connection=conn)

        result = await StalenessDetector(executor).run()

        assert result.active_count == 1
        assert result.stale_count == 0

    @pytest.mark.asyncio
    async def test_boundary_040_is_stale(self):
        rows = [_entity_row(confidence=0.4)]
        conn = FakeNeo4jConnection(query_results=rows)
        executor = FakeGraphExecutor(connection=conn)

        result = await StalenessDetector(executor).run()

        assert result.stale_count == 1
        assert result.very_stale_count == 0

    @pytest.mark.asyncio
    async def test_boundary_020_is_very_stale(self):
        rows = [_entity_row(confidence=0.2)]
        conn = FakeNeo4jConnection(query_results=rows)
        executor = FakeGraphExecutor(connection=conn)

        result = await StalenessDetector(executor).run()

        assert result.very_stale_count == 1
        assert result.stale_count == 0


class TestConfirmationList:
    @pytest.mark.asyncio
    async def test_confirmation_list_ordered_by_confidence(self):
        rows = [
            _entity_row(entity_id="high_stale", display_name="HighStale", confidence=0.59),
            _entity_row(entity_id="low_stale", display_name="LowStale", confidence=0.41),
            _entity_row(entity_id="mid_stale", display_name="MidStale", confidence=0.50),
        ]
        conn = FakeNeo4jConnection(query_results=rows)
        executor = FakeGraphExecutor(connection=conn)
        detector = StalenessDetector(executor)

        result = await detector.run()

        assert len(result.confirmation_list) == 3
        confidences = [e["confidence"] for e in result.confirmation_list]
        assert confidences == [0.41, 0.50, 0.59]

    @pytest.mark.asyncio
    async def test_confirmation_list_contains_expected_fields(self):
        rows = [
            _entity_row(
                entity_id="py",
                display_name="Python",
                entity_type="Technology",
                confidence=0.45,
                last_updated="2026-02-01T00:00:00",
            ),
        ]
        conn = FakeNeo4jConnection(query_results=rows)
        executor = FakeGraphExecutor(connection=conn)

        result = await StalenessDetector(executor).run()

        entry = result.confirmation_list[0]
        assert entry["id"] == "py"
        assert entry["display_name"] == "Python"
        assert entry["entity_type"] == "Technology"
        assert entry["confidence"] == 0.45
        assert entry["last_updated"] == "2026-02-01T00:00:00"

    @pytest.mark.asyncio
    async def test_confirmation_list_excludes_non_stale_tiers(self):
        rows = [
            _entity_row(entity_id="active", confidence=0.7),
            _entity_row(entity_id="very_stale", confidence=0.25),
            _entity_row(entity_id="moribund", confidence=0.1),
            _entity_row(entity_id="stale", confidence=0.5),
        ]
        conn = FakeNeo4jConnection(query_results=rows)
        executor = FakeGraphExecutor(connection=conn)

        result = await StalenessDetector(executor).run()

        ids = [e["id"] for e in result.confirmation_list]
        assert ids == ["stale"]


class TestEmptyGraph:
    @pytest.mark.asyncio
    async def test_empty_graph(self):
        conn = FakeNeo4jConnection(query_results=[])
        executor = FakeGraphExecutor(connection=conn)
        detector = StalenessDetector(executor)

        result = await detector.run()

        assert result.active_count == 0
        assert result.stale_count == 0
        assert result.very_stale_count == 0
        assert result.confirmation_list == ()
        assert result.duration_ms >= 0


class TestResultDataclass:
    def test_staleness_result_is_frozen(self):
        result = StalenessResult(
            active_count=1,
            stale_count=0,
            very_stale_count=0,
            confirmation_list=(),
            duration_ms=1.0,
        )
        with pytest.raises(AttributeError):
            result.active_count = 99
