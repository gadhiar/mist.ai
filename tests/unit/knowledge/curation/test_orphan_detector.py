"""Tests for OrphanDetector."""

import pytest

from tests.mocks.neo4j import FakeGraphExecutor, FakeNeo4jConnection, FakeNeo4jRecord


class TestArchivesOrphan:
    @pytest.mark.asyncio
    async def test_archives_orphan_with_low_confidence(self):
        """Entity with no relationships and low confidence gets archived."""
        from backend.knowledge.curation.orphan_detector import OrphanDetector

        conn = FakeNeo4jConnection(
            query_responses={
                "count(e)": [FakeNeo4jRecord({"total": 1})],
                "e.id AS id": [
                    FakeNeo4jRecord(
                        {"id": "old-tech", "entity_type": "Technology", "confidence": 0.2}
                    )
                ],
            }
        )
        executor = FakeGraphExecutor(connection=conn)
        detector = OrphanDetector(executor)

        result = await detector.run()

        assert result.entities_scanned == 1
        assert result.orphans_found == 1
        assert result.orphans_archived == 1
        conn.assert_write_executed("archived")


class TestPreservesConnected:
    @pytest.mark.asyncio
    async def test_preserves_connected_entity(self):
        """Entity with relationships is not flagged as orphan."""
        from backend.knowledge.curation.orphan_detector import OrphanDetector

        conn = FakeNeo4jConnection(
            query_responses={
                "count(e)": [FakeNeo4jRecord({"total": 1})],
                # Orphan query returns empty -- entity has relationships.
                "e.id AS id": [],
            }
        )
        executor = FakeGraphExecutor(connection=conn)
        detector = OrphanDetector(executor)

        result = await detector.run()

        assert result.entities_scanned == 1
        assert result.orphans_found == 0
        assert result.orphans_archived == 0
        conn.assert_no_writes()


class TestPreservesMistIdentity:
    @pytest.mark.asyncio
    async def test_preserves_mist_identity_singleton(self):
        """MistIdentity is never archived even if it has no relationships."""
        from backend.knowledge.curation.orphan_detector import OrphanDetector

        # The Cypher query excludes MistIdentity via the protected_types param,
        # so the orphan query returns nothing for MistIdentity.
        conn = FakeNeo4jConnection(
            query_responses={
                "count(e)": [FakeNeo4jRecord({"total": 1})],
                "e.id AS id": [],
            }
        )
        executor = FakeGraphExecutor(connection=conn)
        detector = OrphanDetector(executor)

        result = await detector.run()

        assert result.orphans_found == 0
        assert result.orphans_archived == 0
        conn.assert_no_writes()


class TestEmptyGraph:
    @pytest.mark.asyncio
    async def test_empty_graph(self):
        """No entities produces zero result."""
        from backend.knowledge.curation.orphan_detector import OrphanDetector

        conn = FakeNeo4jConnection(
            query_responses={
                "count(e)": [FakeNeo4jRecord({"total": 0})],
            }
        )
        executor = FakeGraphExecutor(connection=conn)
        detector = OrphanDetector(executor)

        result = await detector.run()

        assert result.entities_scanned == 0
        assert result.orphans_found == 0
        assert result.orphans_archived == 0
        assert result.duration_ms >= 0
        conn.assert_no_writes()
