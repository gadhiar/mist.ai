"""Tests for the legacy utterance-based GraphRegenerator.

QUARANTINED per ADR-010, retired further by R1.3 (Inv-A1): vault markdown
is prose MIST reads, not a fact source, and a vault edit writes nothing to
the graph. `backend/knowledge/curation/graph_regenerator.py`, the module
this docstring used to point to, no longer exists (deleted by R1.3 Task
6). The legacy utterance-based regenerator in backend/knowledge/regeneration/
rebuilt the graph by replaying event-store utterances, which would
re-introduce synthetic eval pollution. Its public entry points
(`regenerate_all`, `regenerate_conversation`) now raise NotImplementedError
immediately.

What remains tested here:
  - The two public methods are quarantined (raise NotImplementedError
    pointing at the current replacement commands -- see
    backend/knowledge/regeneration/graph_regenerator.py's module docstring).
  - Constructor dependency injection still wires correctly (the class is
    not deleted, only its replay entry points are disabled).
  - The ADR-009 provenance-preservation invariant of the
    `_delete_graph_entities` helper is preserved as documentation of the
    dormant class's contract.

What was removed (meaningless post-quarantine): assertions that
`_extract_and_store` re-extracts entities and creates nodes during a
regeneration run. That path is no longer reachable from any public entry
point, so verifying node creation no longer documents live behavior.
"""

import asyncio

import pytest

from backend.knowledge.regeneration.graph_regenerator import GraphRegenerator
from tests.mocks.config import build_test_config
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection, FakeNeo4jRecord


class FakeExtractionPipeline:
    """Minimal pipeline stub for constructor-DI verification."""

    def __init__(self):
        self.calls: list[dict] = []

    async def extract_from_utterance(self, **kwargs):  # pragma: no cover - quarantined
        self.calls.append(kwargs)
        raise AssertionError("extract_from_utterance must not run on a quarantined regenerator")


class TestQuarantine:
    """The public replay entry points are disabled per ADR-010."""

    def test_regenerate_all_raises_quarantine(self):
        reg = GraphRegenerator.__new__(GraphRegenerator)

        with pytest.raises(NotImplementedError, match="ADR-010"):
            asyncio.run(reg.regenerate_all())

    def test_regenerate_conversation_raises_quarantine(self):
        reg = GraphRegenerator.__new__(GraphRegenerator)

        with pytest.raises(NotImplementedError, match="ADR-010"):
            asyncio.run(reg.regenerate_conversation("conv-001"))


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

    The helper is no longer reachable from a public entry point (the
    replay methods are quarantined), but the invariant is retained as
    documentation of the dormant class's contract.
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
