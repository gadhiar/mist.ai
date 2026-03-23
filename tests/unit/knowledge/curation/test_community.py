"""Tests for CommunityDetector."""

from __future__ import annotations

import pytest

from backend.knowledge.curation.community import CommunityDetector, CommunityResult
from tests.mocks.neo4j import FakeGraphExecutor, FakeNeo4jConnection


class _GdsFailConnection(FakeNeo4jConnection):
    """Fake that raises on GDS projection queries to simulate missing GDS plugin."""

    def execute_write(self, query, params=None):
        if "gds.graph.project" in query:
            raise RuntimeError("There is no procedure with the name `gds.graph.project`")
        return super().execute_write(query, params)


class TestGdsAvailable:
    @pytest.mark.asyncio
    async def test_louvain_writes_community_ids(self):
        """When GDS is available, Louvain results are returned."""
        gds_result = {"communityCount": 3, "nodePropertiesWritten": 15}
        conn = FakeNeo4jConnection(write_results=[gds_result])
        executor = FakeGraphExecutor(connection=conn)
        detector = CommunityDetector(executor)

        result = await detector.run()

        assert result.communities_found == 3
        assert result.entities_labeled == 15
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_projection_and_drop_queries_executed(self):
        """The job should project, run Louvain, then drop the projection."""
        gds_result = {"communityCount": 1, "nodePropertiesWritten": 5}
        conn = FakeNeo4jConnection(write_results=[gds_result])
        executor = FakeGraphExecutor(connection=conn)
        detector = CommunityDetector(executor)

        await detector.run()

        write_queries = [q for q, _ in conn.writes]
        assert any("gds.graph.project" in q for q in write_queries)
        assert any("gds.louvain.write" in q for q in write_queries)
        assert any("gds.graph.drop" in q for q in write_queries)


class TestGdsFallback:
    @pytest.mark.asyncio
    async def test_returns_zero_when_gds_unavailable(self):
        """When GDS is not installed, return zero result without raising."""
        conn = _GdsFailConnection()
        executor = FakeGraphExecutor(connection=conn)
        detector = CommunityDetector(executor)

        result = await detector.run()

        assert result.communities_found == 0
        assert result.entities_labeled == 0
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_no_louvain_query_when_projection_fails(self):
        """When projection fails, Louvain should not be attempted."""
        conn = _GdsFailConnection()
        executor = FakeGraphExecutor(connection=conn)
        detector = CommunityDetector(executor)

        await detector.run()

        write_queries = [q for q, _ in conn.writes]
        assert not any("gds.louvain.write" in q for q in write_queries)


class TestEmptyResult:
    @pytest.mark.asyncio
    async def test_empty_gds_result_returns_zero(self):
        """When GDS returns empty result list, counts should be zero."""
        conn = FakeNeo4jConnection(write_results=[])
        executor = FakeGraphExecutor(connection=conn)
        detector = CommunityDetector(executor)

        result = await detector.run()

        assert result == CommunityResult(
            communities_found=0,
            entities_labeled=0,
            duration_ms=result.duration_ms,
        )
