"""ADR-009 lock-in tests for KnowledgeRetriever provenance discipline.

Asserts that:
1. _vector_search targets the entity_embeddings index (covers :__Entity__ only).
2. _gather_facts delegates neighborhood expansion to GraphStore.get_entity_neighborhood
   (Task 8 hardened that method with :__Entity__-only / rel-type allowlist invariants).
3. Retrieval output never surfaces :__Provenance__ node IDs even when a fake
   graph-store returns provenance IDs in its result pool.

These tests are outcome-A lock-ins: the retriever already delegates cleanly;
this file ensures a regression cannot silently reintroduce inline Cypher that
would bypass the ADR-009 invariants baked into GraphStore.
"""

from __future__ import annotations

import pytest

from backend.knowledge.config import KnowledgeConfig
from backend.knowledge.retrieval.knowledge_retriever import KnowledgeRetriever
from backend.knowledge.storage.graph_store import GraphStore
from tests.mocks.config import build_test_config
from tests.mocks.neo4j import FakeNeo4jConnection
from tests.unit.knowledge.conftest import FakeEmbeddingProvider, FakeVectorStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENTITY_VECTOR_INDEX_NAME = "entity_embeddings"


def _make_retriever(
    *,
    search_similar_results: list[dict] | None = None,
    user_rels_results: list[dict] | None = None,
    neighborhood_results: list[dict] | None = None,
    vector_store: FakeVectorStore | None = None,
    config: KnowledgeConfig | None = None,
) -> tuple[KnowledgeRetriever, _CallRecorder, _CallRecorder, _CallRecorder]:
    """Build a KnowledgeRetriever whose GraphStore methods are replaced with
    recording stubs.

    Returns the retriever plus three call-recorders so tests can assert on
    whether and how each method was invoked.
    """
    conn = FakeNeo4jConnection()
    emb = FakeEmbeddingProvider()
    cfg = config or build_test_config()
    vs = vector_store or FakeVectorStore()

    graph_store = GraphStore(connection=conn, embedding_generator=emb)

    similar_recorder = _CallRecorder(search_similar_results or [])
    user_rels_recorder = _CallRecorder(user_rels_results or [])
    neighborhood_recorder = _CallRecorder(neighborhood_results or [])

    graph_store.search_similar_entities = similar_recorder  # type: ignore[method-assign]
    graph_store.get_user_relationships_to_entities = user_rels_recorder  # type: ignore[method-assign]
    graph_store.get_entity_neighborhood = neighborhood_recorder  # type: ignore[method-assign]

    retriever = KnowledgeRetriever(
        config=cfg,
        graph_store=graph_store,
        vector_store=vs,
        query_classifier=None,
        embedding_provider=emb,
    )
    return retriever, similar_recorder, user_rels_recorder, neighborhood_recorder


class _CallRecorder:
    """Callable stub that records every invocation and returns a fixed payload."""

    def __init__(self, payload: list[dict]) -> None:
        self._payload = payload
        self.calls: list[dict] = []

    def __call__(self, **kwargs: object) -> list[dict]:
        self.calls.append(dict(kwargs))
        return self._payload

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def was_called(self) -> bool:
        return bool(self.calls)


# ---------------------------------------------------------------------------
# TestVectorSearchEntityIndexOnly
# ---------------------------------------------------------------------------


class TestVectorSearchEntityIndexOnly:
    """_vector_search must query the entity_embeddings index.

    The entity_embeddings index is defined FOR (e:__Entity__) — it physically
    cannot return :__Provenance__ nodes because provenance nodes do not carry
    the embedding property and are not indexed by that index. This test asserts
    the structural invariant: _vector_search delegates to
    GraphStore.search_similar_entities, which is the only code path that issues
    the CALL db.index.vector.queryNodes('entity_embeddings', ...) Cypher.
    """

    @pytest.mark.asyncio
    async def test_vector_search_delegates_to_search_similar_entities(self):
        """_vector_search must call graph_store.search_similar_entities."""
        similar_entity = {
            "entity_id": "python",
            "entity_type": "Technology",
            "similarity": 0.91,
            "name": "Python",
        }
        retriever, similar_rec, _, _ = _make_retriever(
            search_similar_results=[similar_entity],
            # Relational intent skips vector store; graph-only so vector_search
            # is exercised directly via the graph path.
        )

        # Call _vector_search directly to isolate the method.
        results = await retriever._vector_search(
            query="Python programming language", similarity_threshold=0.7, limit=5
        )

        assert similar_rec.was_called(), (
            "_vector_search must delegate to GraphStore.search_similar_entities; "
            "it was not called"
        )
        assert results == [similar_entity]

    @pytest.mark.asyncio
    async def test_vector_search_passes_correct_index_params(self):
        """search_similar_entities must receive the caller-supplied limit and threshold."""
        retriever, similar_rec, _, _ = _make_retriever(search_similar_results=[])

        await retriever._vector_search(query="test query", similarity_threshold=0.75, limit=8)

        assert similar_rec.was_called()
        call_kwargs = similar_rec.calls[0]
        assert call_kwargs.get("limit") == 8
        assert call_kwargs.get("similarity_threshold") == 0.75


# ---------------------------------------------------------------------------
# TestGatherFactsDelegatesToGraphStore
# ---------------------------------------------------------------------------


class TestGatherFactsDelegatesToGraphStore:
    """_gather_facts must delegate expansion to GraphStore methods, not issue
    inline Cypher. GraphStore.get_entity_neighborhood (Task 8) enforces the
    ADR-009 v1.1 invariants (:__Entity__-only path nodes + rel-type allowlist).
    """

    @pytest.mark.asyncio
    async def test_gather_facts_calls_get_user_relationships_to_entities(self):
        """Direct user->entity path must be sourced from get_user_relationships_to_entities."""
        similar_entities = [
            {"entity_id": "neo4j", "entity_type": "Technology", "similarity": 0.88, "name": "Neo4j"}
        ]
        user_rel = {
            "relationship_type": "USES",
            "entity_id": "neo4j",
            "entity_type": "Technology",
            "properties": {},
        }
        retriever, _, user_rels_rec, _ = _make_retriever(
            search_similar_results=similar_entities,
            user_rels_results=[user_rel],
        )

        facts = await retriever._gather_facts(
            user_id="User",
            similar_entities=similar_entities,
            max_hops=1,
            filters=None,
        )

        assert user_rels_rec.was_called(), (
            "_gather_facts must call GraphStore.get_user_relationships_to_entities; "
            "it was not called"
        )
        assert any(f.predicate == "USES" and f.object == "neo4j" for f in facts)

    @pytest.mark.asyncio
    async def test_gather_facts_calls_get_entity_neighborhood_when_max_hops_gt_1(self):
        """Multi-hop expansion must go through GraphStore.get_entity_neighborhood."""
        similar_entities = [
            {
                "entity_id": "fastapi",
                "entity_type": "Technology",
                "similarity": 0.85,
                "name": "FastAPI",
            }
        ]
        neighborhood_rel = {
            "path_length": 1,
            "source": "fastapi",
            "source_type": "Technology",
            "relationship": "DEPENDS_ON",
            "target": "pydantic",
            "target_type": "Technology",
            "properties": {},
        }
        retriever, _, _, neighborhood_rec = _make_retriever(
            search_similar_results=similar_entities,
            neighborhood_results=[neighborhood_rel],
        )

        facts = await retriever._gather_facts(
            user_id="User",
            similar_entities=similar_entities,
            max_hops=2,
            filters=None,
        )

        assert neighborhood_rec.was_called(), (
            "_gather_facts must call GraphStore.get_entity_neighborhood for max_hops > 1; "
            "it was not called"
        )
        assert any(f.predicate == "DEPENDS_ON" and f.object == "pydantic" for f in facts)

    @pytest.mark.asyncio
    async def test_gather_facts_skips_neighborhood_when_max_hops_is_1(self):
        """When max_hops=1 there are no further hops; neighborhood call must be skipped."""
        similar_entities = [
            {"entity_id": "rust", "entity_type": "Technology", "similarity": 0.80, "name": "Rust"}
        ]
        retriever, _, _, neighborhood_rec = _make_retriever(
            search_similar_results=similar_entities,
        )

        await retriever._gather_facts(
            user_id="User",
            similar_entities=similar_entities,
            max_hops=1,
            filters=None,
        )

        assert (
            not neighborhood_rec.was_called()
        ), "_gather_facts must NOT call get_entity_neighborhood when max_hops=1"


# ---------------------------------------------------------------------------
# TestRetrievalNeverSurfacesProvenanceIds
# ---------------------------------------------------------------------------


class TestRetrievalNeverSurfacesProvenanceIds:
    """End-to-end: retrieval output must not expose :__Provenance__ node IDs.

    A compromised GraphStore stub is constructed that injects provenance-shaped
    IDs into every return value. The retriever must not forward those IDs as
    subject or object fields in its output facts.

    Provenance node ID convention: any ID prefixed with '__prov__' or
    containing ':__Provenance__' (node string representations). The real graph
    enforces this at the Neo4j label level; here we test the retriever's
    delegation boundary — it must go through GraphStore methods that already
    filter them, rather than issuing raw Cypher that could accidentally include
    provenance nodes.
    """

    def _is_provenance_id(self, value: str) -> bool:
        return "__prov__" in value or "__Provenance__" in value

    @pytest.mark.asyncio
    async def test_retrieval_never_surfaces_provenance_subject_or_object(self):
        """Facts produced by _gather_facts must not contain provenance IDs in
        subject or object fields, even when the stub returns provenance-shaped data.
        """
        similar_entities = [
            {
                "entity_id": "python",
                "entity_type": "Technology",
                "similarity": 0.9,
                "name": "Python",
            }
        ]

        # Stub injects a clean entity rel and a provenance-shaped one.
        # GraphStore's real implementation would never return the provenance row;
        # but if the retriever ever bypassed GraphStore and issued raw Cypher,
        # this test would catch the leakage.
        user_rels_with_prov = [
            {
                "relationship_type": "USES",
                "entity_id": "python",
                "entity_type": "Technology",
                "properties": {},
            },
            # Provenance-shaped ID that should never appear in retriever output.
            {
                "relationship_type": "SOURCED_FROM",
                "entity_id": "__prov__vc-abc123",
                "entity_type": "VectorChunk",
                "properties": {},
            },
        ]
        neighborhood_with_prov = [
            {
                "path_length": 1,
                "source": "python",
                "source_type": "Technology",
                "relationship": "RELATED_TO",
                "target": "fastapi",
                "target_type": "Technology",
                "properties": {},
            },
            {
                "path_length": 1,
                "source": "__prov__vc-xyz789",
                "source_type": "__Provenance__",
                "relationship": "HAS_ENTITY",
                "target": "python",
                "target_type": "Technology",
                "properties": {},
            },
        ]

        retriever, _, _, _ = _make_retriever(
            search_similar_results=similar_entities,
            user_rels_results=user_rels_with_prov,
            neighborhood_results=neighborhood_with_prov,
        )

        facts = await retriever._gather_facts(
            user_id="User",
            similar_entities=similar_entities,
            max_hops=2,
            filters=None,
        )

        for fact in facts:
            assert not self._is_provenance_id(
                fact.subject
            ), f"Provenance ID leaked into fact.subject: {fact.subject}"
            assert not self._is_provenance_id(
                fact.object
            ), f"Provenance ID leaked into fact.object: {fact.object}"

    @pytest.mark.asyncio
    async def test_graph_retrieval_path_produces_entity_only_facts(self):
        """Full relational retrieve() path must produce facts with entity IDs only.

        Uses the relational intent so the graph path is exercised end-to-end.
        """
        from tests.unit.knowledge.retrieval.test_knowledge_retriever import (
            StubQueryClassifier,
        )

        conn = FakeNeo4jConnection()
        emb = FakeEmbeddingProvider()
        cfg = build_test_config()

        graph_store = GraphStore(connection=conn, embedding_generator=emb)

        all_user_rels = [
            {
                "relationship_type": "KNOWS",
                "entity_id": "neo4j",
                "entity_type": "Technology",
                "properties": {"created_at": "2026-01-01"},
            },
        ]
        # get_all_user_relationships is used by the relational path
        graph_store.get_all_user_relationships = lambda **kwargs: all_user_rels  # type: ignore[method-assign]
        graph_store.get_entity_neighborhood = lambda **kwargs: []  # type: ignore[method-assign]
        graph_store.search_similar_entities = lambda **kwargs: []  # type: ignore[method-assign]

        retriever = KnowledgeRetriever(
            config=cfg,
            graph_store=graph_store,
            vector_store=FakeVectorStore(),
            query_classifier=StubQueryClassifier("relational"),
            embedding_provider=emb,
        )

        result = await retriever.retrieve(query="What databases do I know?", user_id="User")

        for fact in result.facts:
            assert not self._is_provenance_id(
                fact.subject
            ), f"Provenance ID in fact.subject: {fact.subject}"
            assert not self._is_provenance_id(
                fact.object
            ), f"Provenance ID in fact.object: {fact.object}"
