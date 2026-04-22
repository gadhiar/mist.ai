"""ADR-010 Cluster 8 Phase 9: vault-sidecar branch + three-way RRF.

Verifies that `KnowledgeRetriever` correctly routes the `historical` intent
to the vault sidecar, includes the sidecar as a third leg in the `hybrid`
RRF merge, and degrades cleanly when no sidecar is injected (Phase 6 / 8
behavior preserved).
"""

from __future__ import annotations

from typing import Any

import pytest

from backend.knowledge.config import KnowledgeConfig
from backend.knowledge.models import QueryIntent, RetrievedFact
from backend.knowledge.retrieval.knowledge_retriever import (
    _VECTOR_DISTANCE_SENTINEL,
    KnowledgeRetriever,
)
from backend.knowledge.storage.graph_store import GraphStore
from tests.mocks.config import build_test_config
from tests.mocks.neo4j import FakeNeo4jConnection
from tests.unit.knowledge.conftest import FakeEmbeddingProvider, FakeVectorStore

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class StubClassifier:
    """Returns a fixed intent for routing tests."""

    def __init__(self, intent: str) -> None:
        self._intent = intent

    def classify(self, query: str) -> QueryIntent:
        store_map: dict[str, tuple[str, ...]] = {
            "factual": ("vector",),
            "relational": ("graph",),
            "hybrid": ("vector", "graph", "vault"),
            "live": ("mcp",),
            "identity": ("mist",),
            "historical": ("vault",),
        }
        return QueryIntent(
            intent=self._intent,
            confidence=0.9,
            suggested_stores=store_map[self._intent],
        )


class FakeSidecar:
    """In-memory SidecarIndexProtocol stand-in.

    Records every query_hybrid call so tests can assert on the embedding,
    text, and k that arrived. Returns a configurable canned result list so
    tests can drive specific RRF outcomes.
    """

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self._rows = rows or []
        self.calls: list[dict[str, Any]] = []
        self.fail_on_query: bool = False

    def initialize(self) -> None:  # pragma: no cover -- protocol surface
        pass

    def close(self) -> None:  # pragma: no cover
        pass

    def upsert_file(self, *args, **kwargs) -> int:  # pragma: no cover
        return 0

    def delete_path(self, *args, **kwargs) -> int:  # pragma: no cover
        return 0

    def query_vector(self, *args, **kwargs) -> list[dict]:  # pragma: no cover
        return []

    def query_fts(self, *args, **kwargs) -> list[dict]:  # pragma: no cover
        return []

    def query_hybrid(
        self,
        embedding: list[float],
        text: str,
        k: int = 10,
        rrf_k: int = 60,
    ) -> list[dict]:
        self.calls.append({"embedding": embedding, "text": text, "k": k, "rrf_k": rrf_k})
        if self.fail_on_query:
            raise RuntimeError("simulated sidecar query failure")
        return list(self._rows)

    def chunk_count(self) -> int:  # pragma: no cover
        return len(self._rows)

    def health_check(self) -> bool:  # pragma: no cover
        return True


def _make_sidecar_rows(*paths_and_headings: tuple[str, str]) -> list[dict]:
    """Build sidecar rows for canned query_hybrid returns."""
    rows = []
    for i, (path, heading) in enumerate(paths_and_headings, start=1):
        rows.append(
            {
                "path": path,
                "heading": heading,
                "content": f"Content snippet {i} from {heading}",
                "score": 1.0 / i,  # decreasing score with rank
                "vector_rank": i,
                "fts_rank": i,
                "sources": ["vector", "fts"],
            }
        )
    return rows


def _make_retriever_with_sidecar(
    *,
    intent: str,
    sidecar: FakeSidecar | None = None,
    vector_store: FakeVectorStore | None = None,
    search_similar_results: list[dict] | None = None,
    config: KnowledgeConfig | None = None,
) -> tuple[KnowledgeRetriever, FakeSidecar | None]:
    cfg = config or build_test_config()
    conn = FakeNeo4jConnection()
    emb = FakeEmbeddingProvider()
    gs = GraphStore(connection=conn, embedding_generator=emb)

    similar = search_similar_results if search_similar_results is not None else []
    gs.search_similar_entities = lambda **kw: similar  # type: ignore[method-assign]
    gs.get_user_relationships_to_entities = lambda **kw: []  # type: ignore[method-assign]
    gs.get_entity_neighborhood = lambda **kw: []  # type: ignore[method-assign]

    retriever = KnowledgeRetriever(
        config=cfg,
        graph_store=gs,
        vector_store=vector_store,
        query_classifier=StubClassifier(intent),
        embedding_provider=emb,
        vault_sidecar=sidecar,
    )
    return retriever, sidecar


# ---------------------------------------------------------------------------
# TestHistoricalIntent -- vault sidecar is the sole retriever
# ---------------------------------------------------------------------------


class TestHistoricalIntentRouting:
    @pytest.mark.asyncio
    async def test_historical_intent_calls_sidecar(self) -> None:
        # Arrange
        sidecar = FakeSidecar(
            rows=_make_sidecar_rows(
                ("/vault/sessions/x.md", "Vault Architecture"),
                ("/vault/sessions/y.md", "Persistent Memory"),
            )
        )
        retriever, _ = _make_retriever_with_sidecar(intent="historical", sidecar=sidecar)

        # Act
        result = await retriever.retrieve(query="what did we discuss about memory?")

        # Assert -- sidecar called once with the query text + embedding
        assert len(sidecar.calls) == 1
        assert sidecar.calls[0]["text"] == "what did we discuss about memory?"
        assert sidecar.calls[0]["embedding"] is not None
        assert result.intent == "historical"
        assert result.total_facts == 2

    @pytest.mark.asyncio
    async def test_historical_intent_returns_vault_facts_with_path(self) -> None:
        # Arrange
        sidecar = FakeSidecar(
            rows=_make_sidecar_rows(("/vault/sessions/2026-04-22-vault.md", "Cluster 8"))
        )
        retriever, _ = _make_retriever_with_sidecar(intent="historical", sidecar=sidecar)

        # Act
        result = await retriever.retrieve(query="last time we discussed clusters")

        # Assert -- mapped to RetrievedFact with path in properties
        assert len(result.facts) == 1
        fact = result.facts[0]
        assert fact.subject == "VaultNote"
        assert fact.predicate == "MENTIONS"
        assert fact.object == "Cluster 8"
        assert fact.properties["path"] == "/vault/sessions/2026-04-22-vault.md"
        assert fact.properties["sources"] == ["vector", "fts"]
        assert fact.graph_distance == _VECTOR_DISTANCE_SENTINEL

    @pytest.mark.asyncio
    async def test_historical_intent_without_sidecar_returns_empty(self) -> None:
        # Arrange -- no sidecar injected
        retriever, _ = _make_retriever_with_sidecar(intent="historical", sidecar=None)

        # Act -- must not raise even though there's no backend to satisfy
        result = await retriever.retrieve(query="what did we discuss?")

        # Assert -- empty result, intent preserved
        assert result.intent == "historical"
        assert result.total_facts == 0

    @pytest.mark.asyncio
    async def test_historical_intent_swallows_sidecar_failure(self) -> None:
        # Arrange -- sidecar query raises
        sidecar = FakeSidecar()
        sidecar.fail_on_query = True
        retriever, _ = _make_retriever_with_sidecar(intent="historical", sidecar=sidecar)

        # Act -- error must not propagate; degrade to empty
        result = await retriever.retrieve(query="what did we discuss?")

        # Assert
        assert result.intent == "historical"
        assert result.total_facts == 0


# ---------------------------------------------------------------------------
# TestHybridThreeWayMerge -- graph + vector + vault legs
# ---------------------------------------------------------------------------


class TestHybridThreeWayMerge:
    @pytest.mark.asyncio
    async def test_hybrid_intent_calls_all_three_backends(self) -> None:
        # Arrange
        sidecar = FakeSidecar(rows=_make_sidecar_rows(("/v/s/a.md", "TopicA")))
        vector_store = FakeVectorStore()
        # Use search_similar to drive the graph leg's path
        retriever, _ = _make_retriever_with_sidecar(
            intent="hybrid",
            sidecar=sidecar,
            vector_store=vector_store,
            search_similar_results=[
                {"entity_id": "python", "entity_type": "Technology", "similarity": 0.9}
            ],
        )

        # Act
        await retriever.retrieve(query="What programming languages do I use?")

        # Assert -- sidecar fired once for the hybrid path
        assert len(sidecar.calls) == 1

    def test_three_way_rrf_merges_with_hybrid_weights(self) -> None:
        # Arrange -- distinct facts in each leg
        graph_facts = [
            RetrievedFact(
                subject="User",
                subject_type="Person",
                predicate="USES",
                object="Python",
                object_type="Technology",
                properties={},
                similarity_score=0.9,
                graph_distance=0,
            )
        ]
        vector_facts = [
            RetrievedFact(
                subject="Document",
                subject_type="DocumentChunk",
                predicate="CONTAINS",
                object="python-cheatsheet",
                object_type="markdown",
                properties={},
                similarity_score=0.85,
                graph_distance=_VECTOR_DISTANCE_SENTINEL,
            )
        ]
        vault_facts = [
            RetrievedFact(
                subject="VaultNote",
                subject_type="VaultSession",
                predicate="MENTIONS",
                object="Persistent Memory",
                object_type="VaultChunk",
                properties={"path": "/v/s/x.md"},
                similarity_score=0.7,
                graph_distance=_VECTOR_DISTANCE_SENTINEL,
            )
        ]
        config = build_test_config().query_intent

        # Act
        merged = KnowledgeRetriever._merge_rrf_three_way(
            graph_facts=graph_facts,
            vector_facts=vector_facts,
            vault_facts=vault_facts,
            config=config,
            intent="hybrid",
        )

        # Assert -- all three legs contribute distinct facts; total=3
        assert len(merged) == 3
        objects = {f.object for f in merged}
        assert objects == {"Python", "python-cheatsheet", "Persistent Memory"}

    def test_three_way_rrf_uses_historical_weights_when_intent_historical(self) -> None:
        # Arrange -- single fact per leg at rank 1 each. With historical
        # weights (graph=0.2, vector=0.1, vault=0.7), the vault fact must
        # outrank the others.
        graph_facts = [
            RetrievedFact(
                subject="g",
                subject_type="t",
                predicate="P",
                object="graph_obj",
                object_type="t",
                properties={},
                similarity_score=0.9,
                graph_distance=0,
            )
        ]
        vector_facts = [
            RetrievedFact(
                subject="v",
                subject_type="t",
                predicate="P",
                object="vector_obj",
                object_type="t",
                properties={},
                similarity_score=0.9,
                graph_distance=_VECTOR_DISTANCE_SENTINEL,
            )
        ]
        vault_facts = [
            RetrievedFact(
                subject="va",
                subject_type="t",
                predicate="P",
                object="vault_obj",
                object_type="t",
                properties={},
                similarity_score=0.9,
                graph_distance=_VECTOR_DISTANCE_SENTINEL,
            )
        ]
        config = build_test_config().query_intent

        # Act
        merged = KnowledgeRetriever._merge_rrf_three_way(
            graph_facts=graph_facts,
            vector_facts=vector_facts,
            vault_facts=vault_facts,
            config=config,
            intent="historical",
        )

        # Assert -- vault is heaviest for historical, so it wins rank 1
        assert merged[0].object == "vault_obj"

    def test_three_way_rrf_dedupes_overlapping_facts_across_legs(self) -> None:
        # Arrange -- same (subject, predicate, object) in two legs accumulates
        # weight without producing two rows.
        common_fact = RetrievedFact(
            subject="User",
            subject_type="Person",
            predicate="USES",
            object="Python",
            object_type="Technology",
            properties={},
            similarity_score=0.9,
            graph_distance=0,
        )
        config = build_test_config().query_intent

        # Act
        merged = KnowledgeRetriever._merge_rrf_three_way(
            graph_facts=[common_fact],
            vector_facts=[common_fact],
            vault_facts=[],
            config=config,
            intent="hybrid",
        )

        # Assert
        assert len(merged) == 1
        assert merged[0].object == "Python"

    @pytest.mark.asyncio
    async def test_hybrid_without_sidecar_falls_back_to_two_way_merge(self) -> None:
        # Arrange -- no sidecar injected; the hybrid path must still work
        # by falling back to the pre-Phase-9 two-way RRF.
        retriever, _ = _make_retriever_with_sidecar(
            intent="hybrid",
            sidecar=None,
            vector_store=FakeVectorStore(),
            search_similar_results=[
                {"entity_id": "python", "entity_type": "Technology", "similarity": 0.9}
            ],
        )

        # Act -- must not raise
        result = await retriever.retrieve(query="What languages do I know?")

        # Assert -- fell through, no crash
        assert result.intent == "hybrid"


# ---------------------------------------------------------------------------
# TestVaultSidecarRetrieve -- helper-level conversion semantics
# ---------------------------------------------------------------------------


class TestVaultSidecarRetrieve:
    def test_vault_sidecar_retrieve_returns_empty_when_sidecar_none(self) -> None:
        retriever, _ = _make_retriever_with_sidecar(intent="historical", sidecar=None)

        rows, facts = retriever._vault_sidecar_retrieve(query="x", embedding=[0.1] * 384, limit=5)

        assert rows == []
        assert facts == []

    def test_vault_sidecar_retrieve_handles_missing_heading(self) -> None:
        # Arrange -- a file-level chunk has heading=None in the schema
        sidecar = FakeSidecar(
            rows=[
                {
                    "path": "/v/s/x.md",
                    "heading": None,
                    "content": "full body content",
                    "score": 0.5,
                    "vector_rank": 1,
                    "fts_rank": 1,
                    "sources": ["vector"],
                }
            ]
        )
        retriever, _ = _make_retriever_with_sidecar(intent="historical", sidecar=sidecar)

        # Act
        rows, facts = retriever._vault_sidecar_retrieve(
            query="anything", embedding=[0.1] * 384, limit=5
        )

        # Assert -- file-level chunks render as "(file)" object
        assert len(facts) == 1
        assert facts[0].object == "(file)"

    def test_vault_sidecar_retrieve_passes_limit_as_k(self) -> None:
        sidecar = FakeSidecar()
        retriever, _ = _make_retriever_with_sidecar(intent="historical", sidecar=sidecar)

        retriever._vault_sidecar_retrieve(query="x", embedding=[0.1] * 384, limit=42)

        assert sidecar.calls[0]["k"] == 42
