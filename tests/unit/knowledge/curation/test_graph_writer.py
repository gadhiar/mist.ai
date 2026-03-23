"""Tests for CurationGraphWriter."""

import pytest

from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeGraphExecutor, FakeNeo4jConnection
from tests.unit.knowledge.curation.conftest import make_entity_dict, make_relationship_dict


class TestEntityUpsert:
    @pytest.mark.asyncio
    async def test_creates_entity_with_merge(self):
        from backend.knowledge.curation.confidence import ConfidenceManager
        from backend.knowledge.curation.graph_writer import CurationGraphWriter

        conn = FakeNeo4jConnection()
        executor = FakeGraphExecutor(connection=conn)
        writer = CurationGraphWriter(executor, FakeEmbeddingGenerator(), ConfidenceManager())

        entities = [make_entity_dict(entity_id="python", display_name="Python")]
        result = await writer.write(
            entities=entities,
            relationships=[],
            merge_actions=[],
            supersession_actions=[],
            event_id="evt-001",
            session_id="sess-001",
        )

        assert result.entities_created + result.entities_updated >= 1
        conn.assert_write_executed("MERGE")
        conn.assert_write_executed("__Entity__")


class TestRelationshipUpsert:
    @pytest.mark.asyncio
    async def test_creates_relationship_with_merge(self):
        from backend.knowledge.curation.confidence import ConfidenceManager
        from backend.knowledge.curation.graph_writer import CurationGraphWriter

        conn = FakeNeo4jConnection()
        executor = FakeGraphExecutor(connection=conn)
        writer = CurationGraphWriter(executor, FakeEmbeddingGenerator(), ConfidenceManager())

        relationships = [make_relationship_dict(source="user", target="python", rel_type="USES")]
        result = await writer.write(
            entities=[],
            relationships=relationships,
            merge_actions=[],
            supersession_actions=[],
            event_id="evt-001",
            session_id="sess-001",
        )

        assert result.relationships_created + result.relationships_updated >= 1
        conn.assert_write_executed("MERGE")
        conn.assert_write_executed("USES")


class TestProvenance:
    @pytest.mark.asyncio
    async def test_creates_conversation_context(self):
        from backend.knowledge.curation.confidence import ConfidenceManager
        from backend.knowledge.curation.graph_writer import CurationGraphWriter

        conn = FakeNeo4jConnection()
        executor = FakeGraphExecutor(connection=conn)
        writer = CurationGraphWriter(executor, FakeEmbeddingGenerator(), ConfidenceManager())

        entities = [make_entity_dict(entity_id="python")]
        await writer.write(
            entities=entities,
            relationships=[],
            merge_actions=[],
            supersession_actions=[],
            event_id="evt-001",
            session_id="sess-001",
        )

        conn.assert_write_executed("ConversationContext")
        conn.assert_write_executed("EXTRACTED_FROM")


class TestSupersession:
    @pytest.mark.asyncio
    async def test_marks_old_relationship_superseded(self):
        from backend.knowledge.curation.confidence import ConfidenceManager
        from backend.knowledge.curation.conflict_resolver import SupersessionAction
        from backend.knowledge.curation.graph_writer import CurationGraphWriter

        conn = FakeNeo4jConnection()
        executor = FakeGraphExecutor(connection=conn)
        writer = CurationGraphWriter(executor, FakeEmbeddingGenerator(), ConfidenceManager())

        actions = [
            SupersessionAction(
                old_rel_type="WORKS_AT",
                old_target_id="old-company",
                new_target_id="new-company",
                reason="functional_supersession",
            )
        ]

        result = await writer.write(
            entities=[],
            relationships=[],
            merge_actions=[],
            supersession_actions=actions,
            event_id="evt-001",
            session_id="sess-001",
        )

        assert result.relationships_superseded == 1
        conn.assert_write_executed("superseded")


class TestEmptyInput:
    @pytest.mark.asyncio
    async def test_no_writes_on_empty_input(self):
        from backend.knowledge.curation.confidence import ConfidenceManager
        from backend.knowledge.curation.graph_writer import CurationGraphWriter

        conn = FakeNeo4jConnection()
        executor = FakeGraphExecutor(connection=conn)
        writer = CurationGraphWriter(executor, FakeEmbeddingGenerator(), ConfidenceManager())

        result = await writer.write(
            entities=[],
            relationships=[],
            merge_actions=[],
            supersession_actions=[],
            event_id="evt-001",
            session_id="sess-001",
        )

        assert result.entities_created == 0
        assert result.relationships_created == 0
        conn.assert_no_writes()
