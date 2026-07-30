"""Phase 5.5 tie-up (P3 #1): Retrieval orphan filter tests.

Asserts that the retrieval-path orphan filter is respected end to end:
GraphStore.get_user_relationships_to_entities and get_entity_neighborhood
are expected to filter on edge status, and KnowledgeRetriever._gather_facts
must propagate whatever they return rather than re-filtering itself.

R1.3 retired the write side of this contract (GraphStore.mark_orphaned_by_
provenance_path, Task 5): nothing marks new edges orphaned going forward.
The filter on the read side stays load-bearing for pre-R1.3 data already in
the graph, so these tests stay; the former FakeGraphStore lifecycle
simulation (upsert -> mark_orphaned -> verify) tested only the fake's own
bookkeeping for methods with no real counterpart left and was removed.
"""

from __future__ import annotations

import pytest

from backend.knowledge.retrieval.knowledge_retriever import KnowledgeRetriever
from backend.knowledge.storage.graph_store import GraphStore
from tests.mocks.config import build_test_config
from tests.mocks.neo4j import FakeNeo4jConnection
from tests.unit.knowledge.conftest import FakeEmbeddingProvider, FakeVectorStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_retriever_with_stubs(
    *,
    search_similar_results: list[dict] | None = None,
    user_rels_results: list[dict] | None = None,
    neighborhood_results: list[dict] | None = None,
) -> tuple[KnowledgeRetriever, _RecordingStub, _RecordingStub, _RecordingStub]:
    """Build a KnowledgeRetriever with recording stubs on GraphStore methods."""
    conn = FakeNeo4jConnection()
    emb = FakeEmbeddingProvider()
    cfg = build_test_config()
    graph_store = GraphStore(connection=conn, embedding_generator=emb)

    similar_stub = _RecordingStub(search_similar_results or [])
    user_rels_stub = _RecordingStub(user_rels_results or [])
    neighborhood_stub = _RecordingStub(neighborhood_results or [])

    graph_store.search_similar_entities = similar_stub  # type: ignore[method-assign]
    graph_store.get_user_relationships_to_entities = user_rels_stub  # type: ignore[method-assign]
    graph_store.get_entity_neighborhood = neighborhood_stub  # type: ignore[method-assign]

    retriever = KnowledgeRetriever(
        config=cfg,
        graph_store=graph_store,
        vector_store=FakeVectorStore(),
        query_classifier=None,
        embedding_provider=emb,
    )
    return retriever, similar_stub, user_rels_stub, neighborhood_stub


class _RecordingStub:
    """Callable stub returning a fixed payload and recording all calls."""

    def __init__(self, payload: list[dict]) -> None:
        self._payload = payload
        self.calls: list[dict] = []

    def __call__(self, **kwargs: object) -> list[dict]:
        self.calls.append(dict(kwargs))
        return self._payload

    @property
    def call_count(self) -> int:
        return len(self.calls)


# ---------------------------------------------------------------------------
# TestGatherFactsOrphanFiltering
# ---------------------------------------------------------------------------


class TestGatherFactsOrphanFiltering:
    """_gather_facts must not surface facts whose underlying typed edges are orphaned.

    The retriever delegates to GraphStore.get_user_relationships_to_entities and
    GraphStore.get_entity_neighborhood. Both are expected to filter on edge status
    before returning results. If those methods return a row, _gather_facts converts
    it to a fact — so the filter MUST live in the GraphStore methods, not in
    _gather_facts itself.

    These tests verify the contract: stubs that return only active-edge rows produce
    facts; stubs that return no rows (as the real GraphStore will once the filter is
    in place) produce no facts.
    """

    @pytest.mark.asyncio
    async def test_gather_facts_returns_facts_when_active_edges_exist(self):
        """Baseline: _gather_facts produces facts when graph_store returns active results."""
        similar_entities = [{"entity_id": "python", "entity_type": "Technology", "similarity": 0.9}]
        user_rel = {
            "relationship_type": "USES",
            "entity_id": "python",
            "entity_type": "Technology",
            "properties": {},
        }
        retriever, _, _, _ = _make_retriever_with_stubs(
            search_similar_results=similar_entities,
            user_rels_results=[user_rel],
        )

        facts = await retriever._gather_facts(
            user_id="User",
            similar_entities=similar_entities,
            max_hops=1,
            filters=None,
        )

        assert any(
            f.predicate == "USES" and f.object == "python" for f in facts
        ), "Active edge must appear in retrieved facts"

    @pytest.mark.asyncio
    async def test_gather_facts_returns_empty_when_graph_store_returns_empty(self):
        """When graph_store returns no results (all filtered), _gather_facts returns no facts.

        This models the post-implementation state: mark_orphaned marks typed edges,
        the retrieval queries filter them out, so get_user_relationships_to_entities
        returns [] for the orphaned path's content.
        """
        similar_entities = [{"entity_id": "python", "entity_type": "Technology", "similarity": 0.9}]
        retriever, _, _, _ = _make_retriever_with_stubs(
            search_similar_results=similar_entities,
            user_rels_results=[],  # Simulates orphaned edges filtered at graph layer
        )

        facts = await retriever._gather_facts(
            user_id="User",
            similar_entities=similar_entities,
            max_hops=1,
            filters=None,
        )

        assert facts == [], (
            "When graph_store returns no edges (all orphaned/filtered), "
            "_gather_facts must return an empty list"
        )

    @pytest.mark.asyncio
    async def test_gather_facts_non_orphaned_path_still_surfaces(self):
        """Active edges from a non-orphaned path must appear in results.

        Simulates the post-implementation state: two users' data in graph; one
        path orphaned (filtered out at GraphStore layer), one active. The active
        one's edges must still appear.
        """
        similar_entities = [{"entity_id": "rust", "entity_type": "Technology", "similarity": 0.88}]
        active_rel = {
            "relationship_type": "USES",
            "entity_id": "rust",
            "entity_type": "Technology",
            "properties": {},
        }
        retriever, _, _, _ = _make_retriever_with_stubs(
            search_similar_results=similar_entities,
            user_rels_results=[active_rel],
        )

        facts = await retriever._gather_facts(
            user_id="User",
            similar_entities=similar_entities,
            max_hops=1,
            filters=None,
        )

        assert any(
            f.predicate == "USES" and f.object == "rust" for f in facts
        ), "Active edge from non-orphaned path must surface in facts"
