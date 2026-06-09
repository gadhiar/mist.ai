"""query_hybrid must carry a real per-result display_similarity (cosine for
vector hits, None for FTS-only hits) distinct from the RRF ordering score.

The vault-results panel displays display_similarity as the human-readable
relevance percentage. Before this field existed, the panel showed the RRF
ordering score (1/(rrf_k+rank)), which collapses to a near-uniform ~2% for
every hit. These tests pin the contract that display_similarity carries the
true cosine (1/(1+distance), in (0, 1]) for vector hits and None for FTS-only
hits, and that it is distinct from the RRF score field.

Setup mirrors the real VaultSidecarIndex API used throughout
test_sidecar_index.py: SidecarIndexConfig + FakeEmbeddingGenerator, seeding
via upsert_file (embeddings generated internally from content), and querying
via query_hybrid(embedding, text, k).
"""

import pytest

from backend.knowledge.config import SidecarIndexConfig
from backend.vault.sidecar_index import VaultSidecarIndex
from tests.mocks.embeddings import FakeEmbeddingGenerator

_EMBEDDING_DIMENSION = 384


def _make_index(tmp_path):
    """Construct and initialize a fresh VaultSidecarIndex over tmp_path."""
    config = SidecarIndexConfig(
        enabled=True,
        db_path=str(tmp_path / "sidecar.db"),
        embedding_dimension=_EMBEDDING_DIMENSION,
        heading_context_weight=0.3,
        chunk_max_chars=6000,
        rebuild_on_startup=False,
    )
    idx = VaultSidecarIndex(config, FakeEmbeddingGenerator(dimension=_EMBEDDING_DIMENSION))
    idx.initialize()
    return idx


def test_vector_hit_carries_cosine_not_rrf(tmp_path):
    # Arrange -- two single-chunk notes; query with the embedding of the first.
    idx = _make_index(tmp_path)
    gen = FakeEmbeddingGenerator(dimension=_EMBEDDING_DIMENSION)
    idx.upsert_file("a.md", "alpha beta gamma about cocktails", 1000)
    idx.upsert_file("b.md", "totally unrelated lexical token zzz", 1000)
    query = gen.generate_embedding("alpha beta gamma about cocktails")

    # Act
    rows = idx.query_hybrid(embedding=query, text="cocktails", k=5)

    # Assert -- top hit carries a true cosine in (0, 1], not the RRF score.
    assert len(rows) >= 1
    top = rows[0]
    assert 0.0 < top["display_similarity"] <= 1.0
    # The RRF ordering score for rank 1 is 1/(60+1) ~= 0.0164; a real cosine is
    # not equal to it. This is the bug the field fixes (panel showed RRF as %).
    assert top["display_similarity"] != pytest.approx(top["score"])

    # Cosines vary per result (distinct distances), unlike the near-uniform RRF.
    sims = [r["display_similarity"] for r in rows if r["display_similarity"] is not None]
    assert len(sims) >= 2
    assert len({round(s, 4) for s in sims}) > 1

    idx.close()


def test_vector_hit_display_similarity_matches_query_vector_cosine(tmp_path):
    # Arrange -- a single note; the cosine carried through query_hybrid must be
    # exactly the score query_vector computes for the same chunk.
    idx = _make_index(tmp_path)
    gen = FakeEmbeddingGenerator(dimension=_EMBEDDING_DIMENSION)
    content = "exact cosine carry-through verification content"
    idx.upsert_file("exact.md", content, 1000)
    query = gen.generate_embedding(content)

    # Act
    vec_rows = idx.query_vector(query, k=5)
    hybrid_rows = idx.query_hybrid(embedding=query, text="cosine verification", k=5)

    # Assert -- the cosine surfaced by query_hybrid equals query_vector's score.
    vec_score_by_path = {r["path"]: r["score"] for r in vec_rows}
    found = next((r for r in hybrid_rows if r["path"] == "exact.md"), None)
    assert found is not None
    assert found["display_similarity"] == pytest.approx(vec_score_by_path["exact.md"])

    idx.close()


def test_fts_only_hit_has_none_display_similarity(tmp_path):
    # Arrange -- engineer a deterministic FTS-only hit. lex.md carries a unique
    # lexical token ("uniquelexicaltoken") that no filler note has, so it ranks
    # #1 in the FTS leg (high BM25 IDF) and survives the RRF top-k cut. The query
    # embedding targets a filler note, so lex.md falls outside the vector
    # overshoot window (k*2 nearest); with 20 fillers it is never a vector hit.
    # Result: lex.md appears in the merged output via the FTS leg only.
    idx = _make_index(tmp_path)
    gen = FakeEmbeddingGenerator(dimension=_EMBEDDING_DIMENSION)
    for i in range(20):
        idx.upsert_file(f"filler{i}.md", f"common ordinary padding text number {i}", 1000)
    idx.upsert_file("lex.md", "zzqxywv uniquelexicaltoken standalone phrase", 1000)
    query = gen.generate_embedding("common ordinary padding text number 0")

    # Act -- FTS matches only lex.md via its unique token.
    rows = idx.query_hybrid(embedding=query, text="uniquelexicaltoken", k=3)

    # Assert -- lex.md is present as an FTS-only hit (no vector rank), and every
    # FTS-only hit has display_similarity=None (FTS has no cosine).
    fts_only = [r for r in rows if r["vector_rank"] is None]
    assert fts_only, "expected at least one FTS-only hit"
    lex = next((r for r in rows if r["path"] == "lex.md"), None)
    assert lex is not None, "lex.md (unique FTS token) should survive the RRF cut"
    assert lex["vector_rank"] is None, "lex.md must be FTS-only (outside vector window)"
    assert all(r["display_similarity"] is None for r in fts_only)

    idx.close()


def test_vector_hit_keeps_cosine_even_when_also_fts_matched(tmp_path):
    # Arrange -- a chunk seen by BOTH legs (vector first, then FTS) must retain
    # its cosine; the FTS else-branch must not overwrite display_similarity.
    idx = _make_index(tmp_path)
    gen = FakeEmbeddingGenerator(dimension=_EMBEDDING_DIMENSION)
    content = "both sources match this retrieval document"
    idx.upsert_file("both.md", content, 1000)
    query = gen.generate_embedding(content)

    # Act
    rows = idx.query_hybrid(embedding=query, text="both sources match retrieval", k=5)

    # Assert -- the dual-source hit carries a real cosine, not None.
    found = next((r for r in rows if r["path"] == "both.md"), None)
    assert found is not None
    assert "vector" in found["sources"]
    assert "fts" in found["sources"]
    assert found["display_similarity"] is not None
    assert 0.0 < found["display_similarity"] <= 1.0

    idx.close()
