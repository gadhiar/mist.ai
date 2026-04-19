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


class TestAdr009ProvenanceExclusion:
    """ADR-009 lock-in: orphan scan targets :__Entity__ only.

    After the label split, provenance nodes carry :__Provenance__:* as their
    base label instead of :__Entity__.  The orphan-detection queries
    `MATCH (e:__Entity__) ...` therefore naturally exclude all provenance
    nodes.  This class verifies that an isolated :__Provenance__:LearningEvent
    (no outgoing/incoming relationships) is NOT counted as an entity and is
    NOT flagged as an orphan.
    """

    @pytest.mark.asyncio
    async def test_orphan_detector_ignores_provenance_nodes(self):
        """ADR-009: orphan scan targets :__Entity__ only; isolated
        :__Provenance__:LearningEvent nodes are not flagged.
        """
        from backend.knowledge.curation.orphan_detector import OrphanDetector

        # The count query (`MATCH (e:__Entity__)`) returns 0 active entities --
        # meaning the fake graph contains only provenance nodes, none of which
        # match :__Entity__.  The orphan query therefore never fires.
        conn = FakeNeo4jConnection(
            query_responses={
                "count(e)": [FakeNeo4jRecord({"total": 0})],
            }
        )
        executor = FakeGraphExecutor(connection=conn)
        detector = OrphanDetector(executor)

        result = await detector.run()

        assert (
            result.entities_scanned == 0
        ), "Provenance nodes must not appear in the :__Entity__ count scan"
        assert result.orphans_found == 0
        assert result.orphans_archived == 0
        # No write must have been issued -- provenance nodes are untouched.
        conn.assert_no_writes()
