"""Tests for GraphRegenerator ExtractionPipeline migration."""

import pytest

from backend.knowledge.extraction.validator import ValidationResult
from backend.knowledge.models import Utterance
from backend.knowledge.regeneration.graph_regenerator import GraphRegenerator
from tests.mocks.config import build_test_config
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection, FakeNeo4jRecord


class FakeExtractionPipeline:
    """Records calls for verification."""

    def __init__(self):
        self.calls: list[dict] = []

    async def extract_from_utterance(self, **kwargs):
        self.calls.append(kwargs)
        return ValidationResult(
            valid=True,
            entities=[
                {
                    "id": "python",
                    "type": "Technology",
                    "name": "Python",
                    "confidence": 0.85,
                    "description": "",
                }
            ],
            relationships=[],
        )


class TestExtractAndStore:
    @pytest.mark.asyncio
    async def test_uses_pipeline_with_synthetic_ids(self):
        from datetime import datetime

        from backend.knowledge.storage.graph_store import GraphStore

        conn = FakeNeo4jConnection()
        gs = GraphStore(conn, FakeEmbeddingGenerator())
        pipeline = FakeExtractionPipeline()

        regenerator = GraphRegenerator(
            config=build_test_config(),
            extraction_pipeline=pipeline,
            graph_store=gs,
        )

        utterance = Utterance(
            utterance_id="utt-001",
            conversation_id="conv-001",
            text="I use Python",
            timestamp=datetime.now(),
        )

        entities, rels = await regenerator._extract_and_store(utterance)

        assert len(pipeline.calls) == 1
        assert pipeline.calls[0]["event_id"] == "regen_utt-001"
        assert pipeline.calls[0]["session_id"] == "conv-001"
        assert entities == 1
        assert rels == 0


class TestConstructorDI:
    def test_accepts_pipeline_and_graph_store(self):
        from backend.knowledge.storage.graph_store import GraphStore

        conn = FakeNeo4jConnection()
        gs = GraphStore(conn, FakeEmbeddingGenerator())
        pipeline = FakeExtractionPipeline()

        regenerator = GraphRegenerator(
            config=build_test_config(),
            extraction_pipeline=pipeline,
            graph_store=gs,
        )
        assert regenerator._pipeline is pipeline
        assert regenerator.graph_store is gs


class TestAdr009DeleteEntitiesPreservesProvenance:
    """ADR-009 lock-in: _delete_graph_entities wipes :__Entity__ only.

    Provenance nodes (:__Provenance__:*) use a different base label and are
    therefore NOT matched by `MATCH (e:__Entity__) DETACH DELETE e`.  This
    test verifies that _delete_graph_entities:
      - issues a DETACH DELETE scoped to :__Entity__
      - does NOT issue any delete query targeting :__Provenance__
    """

    def test_delete_graph_entities_preserves_provenance_nodes(self):
        """ADR-009: _delete_graph_entities wipes :__Entity__ only; provenance
        nodes are not targeted.
        """
        from backend.knowledge.storage.graph_store import GraphStore

        # FakeNeo4jConnection returns empty results by default (delete count=0).
        conn = FakeNeo4jConnection(
            query_results=[FakeNeo4jRecord({"count": 0})],
        )
        gs = GraphStore(conn, FakeEmbeddingGenerator())
        pipeline = FakeExtractionPipeline()

        regenerator = GraphRegenerator(
            config=build_test_config(),
            extraction_pipeline=pipeline,
            graph_store=gs,
        )

        regenerator._delete_graph_entities()

        # Exactly one write must have been issued, and it must target :__Entity__.
        assert len(conn.writes) == 1, f"Expected 1 write (DETACH DELETE), got {len(conn.writes)}"
        delete_query, _ = conn.writes[0]
        assert "__Entity__" in delete_query, "Delete query must reference :__Entity__"
        assert (
            "__Provenance__" not in delete_query
        ), "Delete query must NOT reference :__Provenance__ -- provenance nodes are preserved"
