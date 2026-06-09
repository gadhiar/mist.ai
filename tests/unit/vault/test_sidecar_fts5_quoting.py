"""Robustness tests for the FTS5 MATCH query builder in VaultSidecarIndex.

The vault sidecar's lexical (FTS5) leg builds a `MATCH` expression directly
from raw user text. Natural-language input routinely contains FTS5
metacharacters and operators -- most commonly a trailing `?` on a question --
which made `query_fts` raise `sqlite3.OperationalError: fts5: syntax error`.
That error was swallowed by `query_fts`'s except clause, so the lexical leg
silently returned `[]` and `query_hybrid` degraded to vector-only for EVERY
question. These tests pin the contract that the FTS builder is robust to
arbitrary natural-language punctuation/operators while still matching the real
terms, and that the existing term-combination semantics (implicit AND across
terms) is preserved.

Setup mirrors the real VaultSidecarIndex API used throughout
test_sidecar_index.py: SidecarIndexConfig + FakeEmbeddingGenerator, seeding via
upsert_file (embeddings generated internally from content), and querying via
query_fts / query_hybrid.
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


class TestNaturalLanguageQuestionDoesNotCrash:
    """The headline bug: a question ending in `?` must not raise and must still
    find a seeded doc containing the question's content words via the lexical
    leg.
    """

    def test_query_fts_question_mark_does_not_raise_and_matches(self, tmp_path):
        # Arrange -- a note whose body carries ALL of the question's content
        # words. The builder AND-s the terms (preserving original semantics), so
        # the doc must contain every word for the lexical leg to match it.
        idx = _make_index(tmp_path)
        idx.upsert_file(
            "about-me.md",
            "What do you actually know about me and my work and background?",
            1000,
        )
        idx.upsert_file("unrelated.md", "totally different lexical tokens zzqx", 1000)

        # Act -- the trailing '?' previously raised fts5: syntax error.
        results = idx.query_fts("what do you know about me?", k=5)

        # Assert -- the lexical leg returns the relevant doc (not swallowed []).
        paths = [r["path"] for r in results]
        assert "about-me.md" in paths

        idx.close()

    def test_query_hybrid_question_mark_keeps_fts_leg(self, tmp_path):
        # Arrange -- engineer a deterministic FTS-only hit so we can prove the
        # lexical leg actually fired through query_hybrid (vector-only
        # degradation would drop it). lex.md carries a unique token; the query
        # embedding targets the fillers, so lex.md is outside the vector window.
        idx = _make_index(tmp_path)
        gen = FakeEmbeddingGenerator(dimension=_EMBEDDING_DIMENSION)
        for i in range(20):
            idx.upsert_file(f"filler{i}.md", f"common ordinary padding text number {i}", 1000)
        idx.upsert_file("lex.md", "zzqxywv uniquelexicaltoken standalone phrase", 1000)
        query = gen.generate_embedding("common ordinary padding text number 0")

        # Act -- a natural-language question whose terms are all present in
        # lex.md (the builder AND-s them) plus a trailing '?'. If '?' crashed
        # the FTS leg, lex.md (an FTS-only hit) would be absent.
        rows = idx.query_hybrid(embedding=query, text="uniquelexicaltoken standalone phrase?", k=5)

        # Assert -- lex.md surfaced via the FTS leg; FTS leg was not swallowed.
        lex = next((r for r in rows if r["path"] == "lex.md"), None)
        assert lex is not None, "FTS-only hit must survive: lexical leg must not crash on '?'"
        assert "fts" in lex["sources"]

        idx.close()


class TestMetacharInputsDoNotCrashAndMatchRealTerms:
    """A spread of FTS5 metacharacters and bareword operators that appear in
    real user text. Each must (a) not raise, and (b) still match the real
    content words.
    """

    def test_parenthesized_phrase_with_question_mark(self, tmp_path):
        # Arrange
        idx = _make_index(tmp_path)
        idx.upsert_file("slalom.md", "Raj is a senior engineer working at Slalom consulting.", 1000)

        # Act -- parens and '?' are FTS5 metacharacters.
        results = idx.query_fts("Slalom (engineer)?", k=5)

        # Assert -- real terms still match.
        assert any(r["path"] == "slalom.md" for r in results)

        idx.close()

    def test_embedded_double_quotes(self, tmp_path):
        # Arrange
        idx = _make_index(tmp_path)
        idx.upsert_file("vault.md", "The vault layer indexes the sidecar for retrieval.", 1000)

        # Act -- a user quoting a word inside a sentence.
        results = idx.query_fts('the "vault" layer', k=5)

        # Assert -- terms match despite embedded quotes (no unterminated-string error).
        assert any(r["path"] == "vault.md" for r in results)

        idx.close()

    def test_plus_star_colon_metachars(self, tmp_path):
        # Arrange
        idx = _make_index(tmp_path)
        idx.upsert_file("cpp.md", "We discussed C and assorted stuff about the build.", 1000)

        # Act -- '+', '*', and trailing ':' are all FTS5 metacharacters; the
        # token 'C' should still match. (porter stemming keeps short alpha tokens.)
        results = idx.query_fts("C++ * stuff:", k=5)

        # Assert -- did not raise; 'stuff' (a content word) matches the doc.
        assert any(r["path"] == "cpp.md" for r in results)

        idx.close()

    def test_not_sure_is_literal_not_the_not_operator(self, tmp_path):
        # Arrange -- two docs. doc_both contains BOTH "not" and "sure"; doc_other
        # contains "sure" but NOT "not". If "not sure" were parsed as the NOT
        # operator (sure NOT ...), the result set would differ from a literal
        # AND of the two terms. We assert the literal-AND interpretation: only
        # the doc containing both words matches.
        idx = _make_index(tmp_path)
        idx.upsert_file("both.md", "I am not entirely sure about the plan we made.", 1000)
        idx.upsert_file("other.md", "She felt quite sure that the answer was correct.", 1000)

        # Act -- every content word is present in both.md so the implicit-AND
        # term set is satisfiable there; other.md lacks "not".
        results = idx.query_fts("not sure about", k=5)
        paths = {r["path"] for r in results}

        # Assert -- treated as literal terms (implicit AND), NOT the NOT operator.
        # both.md has all of {not, sure, about}; other.md lacks "not", so under
        # implicit AND it must not appear. (Under NOT-operator parsing of "not
        # sure", the discriminating outcome would differ.)
        assert "both.md" in paths
        assert "other.md" not in paths

        idx.close()

    def test_all_punctuation_query_returns_empty_without_crash(self, tmp_path):
        # Arrange
        idx = _make_index(tmp_path)
        idx.upsert_file("doc.md", "some ordinary content here", 1000)

        # Act -- a query with no real tokens at all.
        results = idx.query_fts("???", k=5)

        # Assert -- no crash, no spurious matches.
        assert results == []

        idx.close()

    @pytest.mark.parametrize(
        "query",
        [
            pytest.param("what do you know about me?", id="question-trailing-qmark"),
            pytest.param("Slalom (engineer)?", id="parens-and-qmark"),
            pytest.param('the "vault" layer', id="embedded-quotes"),
            pytest.param("C++ * stuff:", id="plus-star-colon"),
            pytest.param("not sure about this", id="not-operator-as-word"),
            pytest.param("???", id="all-punctuation"),
            pytest.param("AND OR NOT NEAR", id="bare-operators-only"),
            pytest.param("type:session memory", id="colon-field-like"),
            pytest.param("a.b.c -dash +plus ^caret", id="dot-dash-plus-caret"),
            pytest.param("it's a 'quoted' word", id="single-quotes"),
            pytest.param("", id="empty-string"),
        ],
    )
    def test_metachar_spread_never_raises(self, tmp_path, query):
        # Arrange
        idx = _make_index(tmp_path)
        idx.upsert_file("doc.md", "memory session vault content about work", 1000)

        # Act / Assert -- none of these may raise; all return a list.
        result = idx.query_fts(query, k=5)
        assert isinstance(result, list)

        idx.close()


class TestTermSemanticsPreserved:
    """The original builder combined bareword terms with implicit AND. The
    robust builder must keep that: a multi-word query matches only docs
    containing ALL the (real) terms.
    """

    def test_multi_word_query_still_returns_expected_hits(self, tmp_path):
        # Arrange
        idx = _make_index(tmp_path)
        idx.upsert_file("fox.md", "the quick brown fox jumped", 1000)
        idx.upsert_file("dog.md", "a lazy dog sleeps soundly", 1000)

        # Act -- a normal (no-metachar) multi-word query.
        results = idx.query_fts("quick brown fox", k=5)

        # Assert -- regression: normal lexical retrieval still works.
        assert any(r["path"] == "fox.md" for r in results)

        idx.close()

    def test_implicit_and_excludes_partial_matches(self, tmp_path):
        # Arrange -- only one doc contains BOTH terms; another contains one.
        idx = _make_index(tmp_path)
        idx.upsert_file("both.md", "alpha beta together in one note", 1000)
        idx.upsert_file("alpha_only.md", "alpha appears here but not the other", 1000)

        # Act -- implicit AND: only the doc with both 'alpha' and 'beta' matches.
        results = idx.query_fts("alpha beta", k=5)
        paths = {r["path"] for r in results}

        # Assert -- AND semantics preserved: alpha_only.md (missing 'beta') is excluded.
        assert "both.md" in paths
        assert "alpha_only.md" not in paths

        idx.close()
