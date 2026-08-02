"""Tests for CurationGraphWriter (entity + provenance writes).

Relationship writes and supersession moved to the ReconciliationEngine at
the C2 cutover -- see test_reconciliation_engine.py for that coverage
(including the Bug-A r.provenance='extraction' regression guard).
"""

import pytest

from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeGraphExecutor, FakeNeo4jConnection
from tests.unit.knowledge.curation._graph_writer_fakes import TEST_REBUILD_STAMPS
from tests.unit.knowledge.curation.conftest import make_entity_dict


class TestEntityUpsert:
    @pytest.mark.asyncio
    async def test_creates_entity_with_merge(self):
        from backend.knowledge.curation.confidence import ConfidenceManager
        from backend.knowledge.curation.graph_writer import CurationGraphWriter

        conn = FakeNeo4jConnection()
        executor = FakeGraphExecutor(connection=conn)
        writer = CurationGraphWriter(
            executor, FakeEmbeddingGenerator(), ConfidenceManager(), TEST_REBUILD_STAMPS
        )

        entities = [make_entity_dict(entity_id="python", display_name="Python")]
        result = await writer.write(
            entities=entities,
            merge_actions=[],
            event_id="evt-001",
            session_id="sess-001",
        )

        assert result.entities_created + result.entities_updated >= 1
        conn.assert_write_executed("MERGE")
        conn.assert_write_executed("__Entity__")

    @pytest.mark.asyncio
    async def test_entity_version_stamp_is_parameterized(self):
        """4.7 drift fix: no hardcoded ontology version literal in the MERGE."""
        from backend.knowledge.curation.confidence import ConfidenceManager
        from backend.knowledge.curation.graph_writer import CurationGraphWriter, RebuildStamps

        conn = FakeNeo4jConnection()
        executor = FakeGraphExecutor(connection=conn)
        writer = CurationGraphWriter(
            executor,
            FakeEmbeddingGenerator(),
            ConfidenceManager(),
            rebuild_stamps=RebuildStamps(
                ontology_version="9.9.9", extraction_version="x", model_hash="m"
            ),
        )

        await writer.write(
            entities=[make_entity_dict(entity_id="rust")],
            merge_actions=[],
            event_id="evt-001",
            session_id="sess-001",
        )

        entity_writes = [(q, p) for q, p in conn.writes if "MERGE (e:__Entity__" in q]
        assert len(entity_writes) == 1
        query, params = entity_writes[0]
        assert "e.ontology_version = $ontology_version" in query
        assert params["ontology_version"] == "9.9.9"


class TestProvenance:
    @pytest.mark.asyncio
    async def test_new_entity_gets_provenance_extraction(self):
        """Bug A: new entities from extraction must have provenance='extraction'."""
        from backend.knowledge.curation.confidence import ConfidenceManager
        from backend.knowledge.curation.graph_writer import CurationGraphWriter

        conn = FakeNeo4jConnection()
        executor = FakeGraphExecutor(connection=conn)
        writer = CurationGraphWriter(
            executor, FakeEmbeddingGenerator(), ConfidenceManager(), TEST_REBUILD_STAMPS
        )

        entities = [make_entity_dict(entity_id="rust", display_name="Rust")]
        await writer.write(
            entities=entities,
            merge_actions=[],
            event_id="evt-001",
            session_id="sess-001",
        )

        # conn.writes is list[tuple[str, dict | None]]; index 0 is the Cypher string.
        entity_writes = [q for q, _ in conn.writes if "MERGE (e:__Entity__" in q]
        assert len(entity_writes) == 1, f"Expected 1 entity MERGE, got {len(entity_writes)}"
        assert (
            "e.provenance = 'extraction'" in entity_writes[0]
        ), f"Expected provenance in ON CREATE SET clause, got:\n{entity_writes[0]}"

    @pytest.mark.asyncio
    async def test_creates_conversation_context(self):
        from backend.knowledge.curation.confidence import ConfidenceManager
        from backend.knowledge.curation.graph_writer import CurationGraphWriter

        conn = FakeNeo4jConnection()
        executor = FakeGraphExecutor(connection=conn)
        writer = CurationGraphWriter(
            executor, FakeEmbeddingGenerator(), ConfidenceManager(), TEST_REBUILD_STAMPS
        )

        entities = [make_entity_dict(entity_id="python")]
        await writer.write(
            entities=entities,
            merge_actions=[],
            event_id="evt-001",
            session_id="sess-001",
        )

        conn.assert_write_executed("ConversationContext")
        conn.assert_write_executed("EXTRACTED_FROM")


class TestBeliefChangeLearningEvent:
    @pytest.mark.asyncio
    async def test_creates_learning_event_for_belief_change(self):
        from backend.knowledge.curation.confidence import ConfidenceManager
        from backend.knowledge.curation.graph_writer import CurationGraphWriter

        conn = FakeNeo4jConnection()
        executor = FakeGraphExecutor(connection=conn)
        writer = CurationGraphWriter(
            executor, FakeEmbeddingGenerator(), ConfidenceManager(), TEST_REBUILD_STAMPS
        )

        await writer.create_belief_change_learning_event(
            reason="contradiction",
            predicate="DISLIKES",
            old_target_id="rust",
            session_id="sess-001",
            event_id="evt-001",
            now="2026-06-10T12:00:00+00:00",
        )

        learning_writes = [(q, p) for q, p in conn.writes if "LearningEvent" in q]
        assert len(learning_writes) == 1
        query, params = learning_writes[0]
        assert params["learning_id"] == "learning-evt-001-DISLIKES-rust"
        assert params["reason"] == "contradiction"
        assert "MERGE (le)-[:ABOUT]->(target)" in query


class TestEmptyInput:
    @pytest.mark.asyncio
    async def test_no_writes_on_empty_input(self):
        from backend.knowledge.curation.confidence import ConfidenceManager
        from backend.knowledge.curation.graph_writer import CurationGraphWriter

        conn = FakeNeo4jConnection()
        executor = FakeGraphExecutor(connection=conn)
        writer = CurationGraphWriter(
            executor, FakeEmbeddingGenerator(), ConfidenceManager(), TEST_REBUILD_STAMPS
        )

        result = await writer.write(
            entities=[],
            merge_actions=[],
            event_id="evt-001",
            session_id="sess-001",
        )

        assert result.entities_created == 0
        conn.assert_no_writes()
