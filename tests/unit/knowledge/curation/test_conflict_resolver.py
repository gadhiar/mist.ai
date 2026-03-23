"""Tests for ConflictResolver."""

import pytest

from tests.mocks.neo4j import FakeGraphExecutor, FakeNeo4jConnection
from tests.unit.knowledge.curation.conftest import make_entity_dict, make_relationship_dict


class TestFunctionalSupersession:
    @pytest.mark.asyncio
    async def test_works_at_supersedes_old(self):
        from backend.knowledge.curation.conflict_resolver import ConflictResolver

        conn = FakeNeo4jConnection(
            query_responses={
                "t.id <> $new_target": [
                    {
                        "source_id": "user",
                        "target_id": "old-company",
                        "type": "WORKS_AT",
                        "status": "active",
                    }
                ],
            }
        )
        executor = FakeGraphExecutor(connection=conn)
        resolver = ConflictResolver(executor)

        entities = [
            make_entity_dict(entity_id="user", entity_type="User"),
            make_entity_dict(entity_id="new-company", entity_type="Organization"),
        ]
        relationships = [
            make_relationship_dict(source="user", target="new-company", rel_type="WORKS_AT"),
        ]

        result = await resolver.resolve(entities, relationships)
        assert result.conflicts_detected == 1
        assert result.conflicts_resolved == 1
        assert len(result.supersession_actions) == 1
        assert result.supersession_actions[0].old_target_id == "old-company"
        assert result.supersession_actions[0].new_target_id == "new-company"
        assert result.supersession_actions[0].reason == "functional_supersession"


class TestContradictionPairs:
    @pytest.mark.asyncio
    async def test_uses_contradicts_dislikes(self):
        from backend.knowledge.curation.conflict_resolver import ConflictResolver

        conn = FakeNeo4jConnection(
            query_responses={
                "$contra_type": [
                    {
                        "source_id": "user",
                        "target_id": "python",
                        "type": "DISLIKES",
                        "status": "active",
                    }
                ],
            }
        )
        executor = FakeGraphExecutor(connection=conn)
        resolver = ConflictResolver(executor)

        entities = [make_entity_dict(entity_id="user", entity_type="User")]
        relationships = [
            make_relationship_dict(source="user", target="python", rel_type="USES"),
        ]

        result = await resolver.resolve(entities, relationships)
        assert result.conflicts_detected == 1
        assert result.conflicts_resolved == 1
        action = result.supersession_actions[0]
        assert action.reason == "contradiction"


class TestIntraBatchContradiction:
    @pytest.mark.asyncio
    async def test_detects_contradiction_within_batch(self):
        from backend.knowledge.curation.conflict_resolver import ConflictResolver

        executor = FakeGraphExecutor()  # Empty graph
        resolver = ConflictResolver(executor)

        entities = [make_entity_dict(entity_id="user", entity_type="User")]
        relationships = [
            make_relationship_dict(source="user", target="python", rel_type="USES"),
            make_relationship_dict(source="user", target="python", rel_type="DISLIKES"),
        ]

        result = await resolver.resolve(entities, relationships)
        assert result.conflicts_detected == 1
        # Last one wins (newer), first is dropped
        assert len(result.relationships) == 1
        assert result.relationships[0]["type"] == "DISLIKES"


class TestNoConflict:
    @pytest.mark.asyncio
    async def test_passes_through_without_conflicts(self):
        from backend.knowledge.curation.conflict_resolver import ConflictResolver

        executor = FakeGraphExecutor()  # Empty graph
        resolver = ConflictResolver(executor)

        entities = [make_entity_dict(entity_id="user", entity_type="User")]
        relationships = [
            make_relationship_dict(source="user", target="python", rel_type="USES"),
            make_relationship_dict(source="user", target="rust", rel_type="LEARNING"),
        ]

        result = await resolver.resolve(entities, relationships)
        assert result.conflicts_detected == 0
        assert result.conflicts_resolved == 0
        assert len(result.relationships) == 2
