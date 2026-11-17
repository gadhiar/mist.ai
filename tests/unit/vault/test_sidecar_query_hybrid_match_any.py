"""`query_hybrid(..., match="any")`: the lexical arm ORs the quoted tokens.

A recall query is a sentence, and the default AND of its tokens empties the FTS leg as soon as
one word ("banana" below) appears in no note. The memory read path asks for `match="any"` so a
note sharing any query word still gets an FTS rank; every existing caller keeps the AND.

The existing calls, none of which passes `match` (`grep -rn 'query_hybrid(' backend scripts`
at the tip of this task's branch, line numbers left out because they drift):

    backend/interfaces.py:  def query_hybrid(
    backend/knowledge/retrieval/knowledge_retriever.py:  return sidecar.query_hybrid(
    backend/vault/sidecar_index.py:  results = idx.query_hybrid(embedding, text, k=5)
    backend/vault/sidecar_index.py:  def query_hybrid(

The two definitions, the retriever's call (keywords `embedding`, `text`, `k`, `rrf_k`) and
the class docstring's example. The memory read path calls `query_hybrid_readonly`, which the
pattern does not match. The tests below pin that the default stays `"all"`, so those calls
keep today's behaviour.
"""

import inspect

import pytest

from backend.interfaces import SidecarIndexProtocol
from backend.knowledge.config import SidecarIndexConfig
from backend.vault.sidecar_index import VaultSidecarIndex, _quote_fts5
from tests.mocks.embeddings import FakeEmbeddingGenerator

_EMBEDDING_DIMENSION = 384


@pytest.fixture()
def sidecar(tmp_path):
    """A fresh VaultSidecarIndex over a tmp database."""
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
    yield idx
    idx.close()


def _seed(idx: VaultSidecarIndex) -> list[float]:
    idx.upsert_file("projects/kestrel.md", "Kestrel is the rust rewrite of the parser.", 1)
    idx.upsert_file("notes/garden.md", "Tomatoes and basil in the raised bed.", 2)
    return [1.0] + [0.0] * (_EMBEDDING_DIMENSION - 1)


def _fts_hits(rows: list[dict]) -> list[str]:
    return [row["path"] for row in rows if row["fts_rank"] is not None]


def test_match_any_keeps_the_lexical_hit_of_a_query_with_an_unmatched_word(sidecar):
    embedding = _seed(sidecar)

    rows = sidecar.query_hybrid(embedding, "kestrel banana", k=10, match="any")

    assert "projects/kestrel.md" in _fts_hits(rows)
    kestrel = next(r for r in rows if r["path"] == "projects/kestrel.md" and r["fts_rank"])
    assert "fts" in kestrel["sources"]
    assert "notes/garden.md" not in _fts_hits(rows)


def test_match_all_drops_that_hit_so_the_default_is_unchanged(sidecar):
    embedding = _seed(sidecar)

    explicit = sidecar.query_hybrid(embedding, "kestrel banana", k=10, match="all")
    default = sidecar.query_hybrid(embedding, "kestrel banana", k=10)

    assert _fts_hits(explicit) == []
    assert default == explicit


def test_match_any_ors_the_quoted_tokens_and_all_keeps_the_implicit_and():
    assert _quote_fts5('not sure? "C++"', match="any") == '"not" OR "sure" OR "C"'
    assert _quote_fts5('not sure? "C++"', match="all") == '"not" "sure" "C"'
    assert _quote_fts5('not sure? "C++"') == '"not" "sure" "C"'
    assert _quote_fts5("???", match="any") == ""


def test_query_fts_accepts_match_any(sidecar):
    _seed(sidecar)

    assert [r["path"] for r in sidecar.query_fts("kestrel banana", match="any")] == [
        "projects/kestrel.md"
    ]
    assert sidecar.query_fts("kestrel banana") == []


def test_an_unknown_match_mode_is_refused(sidecar):
    embedding = _seed(sidecar)

    with pytest.raises(ValueError, match="match"):
        sidecar.query_hybrid(embedding, "kestrel", match="some")


@pytest.mark.parametrize("owner", [VaultSidecarIndex, SidecarIndexProtocol])
def test_match_defaults_to_all_on_the_class_and_the_protocol(owner):
    parameter = inspect.signature(owner.query_hybrid).parameters["match"]

    assert parameter.default == "all"
