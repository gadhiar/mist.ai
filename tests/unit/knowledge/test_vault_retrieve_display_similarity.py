"""Task 2 regression: rrf_k threading + display_similarity carry-through.

Verifies two things about KnowledgeRetriever._vault_sidecar_retrieve:

1. The retriever threads `rrf_k` from config into the sidecar's
   query_hybrid call. `rrf_k` lives on QueryIntentConfig
   (config.query_intent.rrf_k), so the retriever reads it via
   `self.config.query_intent.rrf_k`.
2. The per-row `display_similarity` value returned by query_hybrid
   (distance-derived score for vector hits, None for FTS-only) is carried onto
   the resulting RetrievedFact.properties so it can be surfaced to the
   vault_results FE payload as the displayed similarity.
"""

import pytest

from backend.knowledge.config import KnowledgeConfig
from backend.knowledge.retrieval.knowledge_retriever import KnowledgeRetriever


class _FakeSidecar:
    """Sidecar double mirroring the production worker API: run_async routes
    the call through (synchronously here), so the retriever's worker-routing
    branch is exercised rather than the test-double fallback.
    """

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows
        self.calls: list[dict] = []

    async def run_async(self, fn, /, *args):
        return fn(*args)

    def query_hybrid(self, *, embedding, text, k, rrf_k):
        self.calls.append({"embedding": embedding, "text": text, "k": k, "rrf_k": rrf_k})
        return self._rows


@pytest.mark.asyncio
async def test_retriever_threads_rrf_k_and_carries_display_similarity():
    # KnowledgeConfig requires neo4j/llm/embedding/extraction; from_env()
    # builds all sub-configs (including query_intent) from defaults.
    cfg = KnowledgeConfig.from_env()
    # rrf_k is nested on the query_intent sub-config (QueryIntentConfig).
    cfg.query_intent.rrf_k = 15
    sidecar = _FakeSidecar(
        rows=[
            {
                "path": "a.md",
                "heading": "A",
                "content": "real body text",
                "score": 0.01639,
                "display_similarity": 0.83,
                "vector_rank": 1,
                "fts_rank": None,
                "sources": ["vector"],
            },
        ]
    )
    r = KnowledgeRetriever.__new__(KnowledgeRetriever)
    r._vault_sidecar = sidecar
    r.config = cfg

    rows, facts = await r._vault_sidecar_retrieve(query="q", embedding=[0.1], limit=5)

    assert sidecar.calls[0]["rrf_k"] == 15
    assert facts[0].properties["display_similarity"] == 0.83


@pytest.mark.asyncio
async def test_retriever_carries_none_display_similarity_for_fts_only():
    """FTS-only hits carry display_similarity=None onto fact.properties."""
    cfg = KnowledgeConfig.from_env()
    sidecar = _FakeSidecar(
        rows=[
            {
                "path": "b.md",
                "heading": "B",
                "content": "lexical only body",
                "score": 0.00820,
                "display_similarity": None,
                "vector_rank": None,
                "fts_rank": 1,
                "sources": ["fts"],
            },
        ]
    )
    r = KnowledgeRetriever.__new__(KnowledgeRetriever)
    r._vault_sidecar = sidecar
    r.config = cfg

    _, facts = await r._vault_sidecar_retrieve(query="q", embedding=[0.1], limit=5)

    assert facts[0].properties["display_similarity"] is None
