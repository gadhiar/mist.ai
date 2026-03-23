"""Tests for ConfidenceDecayJob."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from backend.knowledge.curation.confidence_decay import (
    ARCHIVE_THRESHOLD,
    ConfidenceDecayJob,
    DecayResult,
)
from backend.knowledge.ontologies.v1_0_0 import (
    CONFIDENCE_BRIDGING,
    CONFIDENCE_EXTERNAL,
)
from tests.mocks.neo4j import FakeGraphExecutor, FakeNeo4jConnection

MODULE = "backend.knowledge.curation.confidence_decay"


def _entity(
    *,
    entity_id: str = "e1",
    domain: str = "external",
    confidence: float = 0.8,
    updated_at: str = "2026-01-01T00:00:00+00:00",
) -> dict:
    return {
        "id": entity_id,
        "domain": domain,
        "confidence": confidence,
        "updated_at": updated_at,
    }


class TestExternalDecay:
    @pytest.mark.asyncio
    async def test_external_entity_decays_with_180_day_halflife(self):
        """After exactly one half-life (180 days), confidence should halve."""
        half_life = CONFIDENCE_EXTERNAL.decay_half_life_days
        assert half_life == 180

        rec = _entity(domain="external", confidence=0.8)
        conn = FakeNeo4jConnection(query_results=[rec])
        executor = FakeGraphExecutor(connection=conn)
        job = ConfidenceDecayJob(executor)

        with patch(f"{MODULE}._days_since_update", return_value=180.0):
            result = await job.run()

        assert result.entities_scanned == 1
        assert result.entities_decayed == 1

        # 0.8 * 0.5^(180/180) = 0.4
        _, params = conn.writes[0]
        assert abs(params["confidence"] - 0.4) < 1e-9

    @pytest.mark.asyncio
    async def test_recently_updated_entity_minimal_decay(self):
        """An entity updated 1 day ago should have negligible decay."""
        rec = _entity(domain="external", confidence=0.8)
        conn = FakeNeo4jConnection(query_results=[rec])
        executor = FakeGraphExecutor(connection=conn)
        job = ConfidenceDecayJob(executor)

        with patch(f"{MODULE}._days_since_update", return_value=1.0):
            result = await job.run()

        assert result.entities_scanned == 1
        # 0.8 * 0.5^(1/180) ~ 0.7969
        _, params = conn.writes[0]
        assert params["confidence"] > 0.79


class TestBridgingDecay:
    @pytest.mark.asyncio
    async def test_bridging_entity_decays_with_365_day_halflife(self):
        """After one half-life (365 days), bridging confidence should halve."""
        half_life = CONFIDENCE_BRIDGING.decay_half_life_days
        assert half_life == 365

        rec = _entity(domain="bridging", confidence=0.85)
        conn = FakeNeo4jConnection(query_results=[rec])
        executor = FakeGraphExecutor(connection=conn)
        job = ConfidenceDecayJob(executor)

        with patch(f"{MODULE}._days_since_update", return_value=365.0):
            result = await job.run()

        assert result.entities_decayed == 1
        _, params = conn.writes[0]
        assert abs(params["confidence"] - 0.425) < 1e-9


class TestInternalExempt:
    @pytest.mark.asyncio
    async def test_internal_entity_does_not_decay(self):
        """Internal entities should never be queried or decayed."""
        conn = FakeNeo4jConnection(query_results=[])
        executor = FakeGraphExecutor(connection=conn)
        job = ConfidenceDecayJob(executor)

        result = await job.run()

        # The query uses domains=['external', 'bridging'] -- internal excluded.
        assert result.entities_scanned == 0
        assert result.entities_decayed == 0
        conn.assert_no_writes()

        # Verify the query params excluded 'internal'.
        _, params = conn.queries[0]
        assert "internal" not in params["domains"]


class TestArchival:
    @pytest.mark.asyncio
    async def test_moribund_entity_archived(self):
        """Entity decayed below 0.2 threshold gets status='archived'."""
        # 0.3 * 0.5^(360/180) = 0.3 * 0.25 = 0.075 < 0.2
        rec = _entity(domain="external", confidence=0.3)
        conn = FakeNeo4jConnection(query_results=[rec])
        executor = FakeGraphExecutor(connection=conn)
        job = ConfidenceDecayJob(executor)

        with patch(f"{MODULE}._days_since_update", return_value=360.0):
            result = await job.run()

        assert result.entities_archived == 1
        assert result.entities_decayed == 1

        query_str, params = conn.writes[0]
        assert "archived" in query_str
        assert params["confidence"] < ARCHIVE_THRESHOLD


class TestEmptyGraph:
    @pytest.mark.asyncio
    async def test_empty_graph_returns_zero_counts(self):
        conn = FakeNeo4jConnection(query_results=[])
        executor = FakeGraphExecutor(connection=conn)
        job = ConfidenceDecayJob(executor)

        result = await job.run()

        assert result == DecayResult(
            entities_scanned=0,
            entities_decayed=0,
            entities_archived=0,
            duration_ms=result.duration_ms,
        )
        conn.assert_no_writes()
