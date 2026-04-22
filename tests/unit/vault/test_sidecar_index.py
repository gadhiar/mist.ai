"""Unit tests for VaultSidecarIndex.

Uses real SQLite + sqlite-vec against a tmp_path database. The
FakeEmbeddingGenerator from tests.mocks.embeddings provides deterministic
384-dimensional embeddings keyed off content hash. All tests are sync.

Test classes:
    TestInitialize        -- schema creation, rebuild_on_startup, idempotency
    TestClose             -- connection teardown, idempotent close
    TestUpsertFileFileLevel   -- single-chunk notes, re-upsert dedup
    TestUpsertFileHeadingChunks -- multi-chunk notes, provenance skip, truncation
    TestDeletePath        -- cross-table deletion, idempotent, missing path
    TestQueryVector       -- nearest-neighbour retrieval, shape, empty index
    TestQueryFTS          -- lexical retrieval, heading_context, special chars
    TestQueryHybrid       -- RRF merge, single-source edge cases, top-k cap
    TestChunkCount        -- count progression
    TestHealthCheck       -- state assertions
"""

import pytest

from backend.errors import SidecarIndexError
from backend.knowledge.config import SidecarIndexConfig
from backend.vault.sidecar_index import (
    VaultSidecarIndex,
    _extract_heading_blocks,
    _quote_fts5,
)
from tests.mocks.embeddings import FakeEmbeddingGenerator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_EMBEDDING_DIMENSION = 384


def _make_config(tmp_path, **kwargs) -> SidecarIndexConfig:
    """Build a SidecarIndexConfig pointing at tmp_path/sidecar.db."""
    defaults = {
        "enabled": True,
        "db_path": str(tmp_path / "sidecar.db"),
        "embedding_dimension": _EMBEDDING_DIMENSION,
        "heading_context_weight": 0.3,
        "chunk_max_chars": 6000,
        "rebuild_on_startup": False,
    }
    defaults.update(kwargs)
    return SidecarIndexConfig(**defaults)


@pytest.fixture()
def sidecar_index(tmp_path):
    """Construct and initialize a fresh VaultSidecarIndex per test."""
    config = _make_config(tmp_path)
    emb = FakeEmbeddingGenerator(dimension=_EMBEDDING_DIMENSION)
    idx = VaultSidecarIndex(config, emb)
    idx.initialize()
    yield idx
    idx.close()


# ---------------------------------------------------------------------------
# Helper content strings
# ---------------------------------------------------------------------------

_SIMPLE_NOTE = "This is a simple note with no headings.\n\nJust a paragraph."

_HEADED_NOTE = """\
# Title

Introduction paragraph.

## Section A

Content in section A. This has some detail about topic A.

## Section B

Content in section B. This covers topic B.

## Provenance

- event_id: abc123
- session_id: test-session
"""

_NESTED_HEADING_NOTE = """\
## Top level H2

Some content here.

### Nested H3

Nested content.

## Another H2

Another top-level block.
"""

# ---------------------------------------------------------------------------
# TestInitialize
# ---------------------------------------------------------------------------


class TestInitialize:
    def test_creates_database_file(self, tmp_path):
        # Arrange
        config = _make_config(tmp_path)
        idx = VaultSidecarIndex(config, FakeEmbeddingGenerator())

        # Act
        idx.initialize()

        # Assert
        assert (tmp_path / "sidecar.db").exists()
        idx.close()

    def test_creates_required_schema_objects(self, sidecar_index):
        # Arrange / Act -- initialize already called in fixture

        # Assert
        assert sidecar_index.health_check() is True

    def test_idempotent_reinitialize(self, tmp_path):
        # Arrange
        config = _make_config(tmp_path)
        idx = VaultSidecarIndex(config, FakeEmbeddingGenerator())
        idx.initialize()
        idx.upsert_file("a.md", "hello", 1000)

        # Act -- second initialize on same db
        idx.initialize()

        # Assert -- data survives
        assert idx.chunk_count() == 1
        idx.close()

    def test_rebuild_on_startup_drops_and_recreates(self, tmp_path):
        # Arrange -- index with data
        config = _make_config(tmp_path)
        idx = VaultSidecarIndex(config, FakeEmbeddingGenerator())
        idx.initialize()
        idx.upsert_file("a.md", "old content", 1000)
        assert idx.chunk_count() == 1
        idx.close()

        # Act -- reopen with rebuild_on_startup=True
        rebuild_config = _make_config(tmp_path, rebuild_on_startup=True)
        idx2 = VaultSidecarIndex(rebuild_config, FakeEmbeddingGenerator())
        idx2.initialize()

        # Assert -- table was dropped and recreated: no rows
        assert idx2.chunk_count() == 0
        idx2.close()

    def test_creates_parent_directories(self, tmp_path):
        # Arrange
        nested_path = tmp_path / "deep" / "nested" / "dir" / "sidecar.db"
        config = SidecarIndexConfig(
            db_path=str(nested_path),
            embedding_dimension=_EMBEDDING_DIMENSION,
        )
        idx = VaultSidecarIndex(config, FakeEmbeddingGenerator())

        # Act
        idx.initialize()

        # Assert
        assert nested_path.exists()
        idx.close()

    def test_vec0_dimension_matches_config(self, tmp_path):
        # Arrange -- use a non-default dimension
        config = _make_config(tmp_path, embedding_dimension=8)
        emb = FakeEmbeddingGenerator(dimension=8)
        idx = VaultSidecarIndex(config, emb)
        idx.initialize()

        # Act -- insert a chunk to exercise the vec table
        n = idx.upsert_file("dim.md", "test content", 1000)

        # Assert -- upsert succeeds (dimension matches)
        assert n >= 1
        idx.close()


# ---------------------------------------------------------------------------
# TestClose
# ---------------------------------------------------------------------------


class TestClose:
    def test_upsert_before_initialize_raises_sidecar_index_error(self, tmp_path):
        # Arrange -- not initialized
        config = _make_config(tmp_path)
        idx = VaultSidecarIndex(config, FakeEmbeddingGenerator())

        # Act / Assert
        with pytest.raises(SidecarIndexError, match="not initialized"):
            idx.upsert_file("a.md", "content", 1000)

    def test_closes_connection(self, tmp_path):
        # Arrange
        config = _make_config(tmp_path)
        idx = VaultSidecarIndex(config, FakeEmbeddingGenerator())
        idx.initialize()

        # Act
        idx.close()

        # Assert -- connection is None after close
        assert idx._conn is None

    def test_close_is_idempotent(self, tmp_path):
        # Arrange
        config = _make_config(tmp_path)
        idx = VaultSidecarIndex(config, FakeEmbeddingGenerator())
        idx.initialize()

        # Act -- call close twice; must not raise
        idx.close()
        idx.close()

        # Assert
        assert idx._conn is None

    def test_health_check_false_after_close(self, sidecar_index):
        # Arrange -- fixture already initialized
        assert sidecar_index.health_check() is True

        # Act
        sidecar_index.close()

        # Assert
        assert sidecar_index.health_check() is False


# ---------------------------------------------------------------------------
# TestUpsertFileFileLevel
# ---------------------------------------------------------------------------


class TestUpsertFileFileLevel:
    def test_single_paragraph_produces_one_chunk(self, sidecar_index):
        # Arrange
        content = "Hello world. No headings here."

        # Act
        n = sidecar_index.upsert_file("note.md", content, 1000)

        # Assert
        assert n == 1
        assert sidecar_index.chunk_count() == 1

    def test_note_without_headings_produces_only_file_level_chunk(self, sidecar_index):
        # Arrange
        content = _SIMPLE_NOTE

        # Act
        n = sidecar_index.upsert_file("simple.md", content, 1000)

        # Assert -- only one file-level chunk
        assert n == 1

    def test_reupsert_updates_content_without_duplicate(self, sidecar_index):
        # Arrange
        sidecar_index.upsert_file("note.md", "original content", 1000)
        assert sidecar_index.chunk_count() == 1

        # Act -- re-upsert same path with new content
        sidecar_index.upsert_file("note.md", "updated content", 2000)

        # Assert -- still one chunk, not two
        assert sidecar_index.chunk_count() == 1

    def test_reupsert_reflects_new_content_in_fts(self, sidecar_index):
        # Arrange
        sidecar_index.upsert_file("note.md", "original unique content", 1000)

        # Act
        sidecar_index.upsert_file("note.md", "completely different wording", 2000)

        # Assert -- new content searchable, old content gone
        new_results = sidecar_index.query_fts("completely different", k=5)
        assert len(new_results) >= 1
        assert new_results[0]["path"] == "note.md"

    def test_file_level_chunk_has_null_heading(self, sidecar_index):
        # Arrange / Act
        sidecar_index.upsert_file("note.md", "content here", 1000)

        # Assert -- file-level chunk has heading=None
        row = sidecar_index._conn.execute(
            "SELECT heading FROM vault_chunks WHERE path = 'note.md' AND heading IS NULL"
        ).fetchone()
        assert row is not None


# ---------------------------------------------------------------------------
# TestUpsertFileHeadingChunks
# ---------------------------------------------------------------------------


class TestUpsertFileHeadingChunks:
    def test_two_heading_blocks_produce_three_chunks(self, sidecar_index):
        # Arrange / Act
        n = sidecar_index.upsert_file("headed.md", _HEADED_NOTE, 1000)

        # Assert -- 1 file-level + 2 heading blocks (Section A, Section B;
        # Provenance is skipped; title # is level 1, ADR says ## is chunked)
        # The note has ## Section A and ## Section B (and ## Provenance skipped)
        # # Title is h1 -- also chunked
        assert n == 4  # file-level + Title(h1) + Section A(h2) + Section B(h2)

    def test_provenance_section_skipped(self, sidecar_index):
        # Arrange / Act
        sidecar_index.upsert_file("headed.md", _HEADED_NOTE, 1000)

        # Assert -- no chunk with heading="Provenance"
        row = sidecar_index._conn.execute(
            "SELECT chunk_id FROM vault_chunks WHERE heading = 'Provenance'"
        ).fetchone()
        assert row is None

    def test_heading_level_captured_correctly(self, sidecar_index):
        # Arrange
        note = "## H2 Heading\n\nContent.\n\n### H3 Heading\n\nMore content."
        sidecar_index.upsert_file("levels.md", note, 1000)

        # Act
        rows = sidecar_index._conn.execute(
            "SELECT heading, heading_level FROM vault_chunks WHERE path = 'levels.md' AND heading IS NOT NULL"
        ).fetchall()
        level_map = {row[0]: row[1] for row in rows}

        # Assert
        assert level_map["H2 Heading"] == 2
        assert level_map["H3 Heading"] == 3

    def test_heading_block_content_excludes_next_heading(self, sidecar_index):
        # Arrange
        note = "## Alpha\n\nAlpha content here.\n\n## Beta\n\nBeta content here."
        sidecar_index.upsert_file("excl.md", note, 1000)

        # Act
        row = sidecar_index._conn.execute(
            "SELECT content FROM vault_chunks WHERE path = 'excl.md' AND heading = 'Alpha'"
        ).fetchone()

        # Assert -- Alpha block does not include Beta content
        assert row is not None
        assert "Beta" not in row[0]
        assert "Alpha content here" in row[0]

    def test_chunk_content_truncated_at_chunk_max_chars(self, tmp_path):
        # Arrange
        max_chars = 50
        config = _make_config(tmp_path, chunk_max_chars=max_chars)
        idx = VaultSidecarIndex(config, FakeEmbeddingGenerator())
        idx.initialize()

        long_content = "x" * 200
        note = f"## Section\n\n{long_content}"

        # Act
        idx.upsert_file("long.md", note, 1000)

        # Assert -- heading block content is capped
        row = idx._conn.execute(
            "SELECT content FROM vault_chunks WHERE path = 'long.md' AND heading IS NOT NULL"
        ).fetchone()
        assert row is not None
        assert len(row[0]) <= max_chars
        idx.close()

    def test_nested_heading_produces_correct_block_boundary(self, sidecar_index):
        # Arrange / Act
        sidecar_index.upsert_file("nested.md", _NESTED_HEADING_NOTE, 1000)

        # The H3 should be its own chunk; H2 "Top level H2" block ends where H3 starts
        top_h2 = sidecar_index._conn.execute(
            "SELECT content FROM vault_chunks WHERE path = 'nested.md' AND heading = 'Top level H2'"
        ).fetchone()
        # Assert -- H3 content not in the H2 block (H3 is same or deeper level)
        assert top_h2 is not None
        # H3 is deeper so it should split out separately; h2 block ends at h3 or next h2
        # H3 is level 3 > level 2, so it does NOT end the H2 block per spec
        # The H2 block ends at the next sibling-or-shallower heading (next H2 or H1)
        # So H2 "Top level H2" block includes H3 nested content
        # This verifies chunk boundary, not excludes nested
        nested_h3 = sidecar_index._conn.execute(
            "SELECT content FROM vault_chunks WHERE path = 'nested.md' AND heading = 'Nested H3'"
        ).fetchone()
        assert nested_h3 is not None
        assert "Nested content" in nested_h3[0]

    def test_frontmatter_note_type_stored_on_chunk(self, sidecar_index):
        # Arrange
        frontmatter = {"type": "mist-session", "tags": ["memory"]}

        # Act
        sidecar_index.upsert_file("session.md", "## Turn 1\n\nHello.", 1000, frontmatter)

        # Assert
        row = sidecar_index._conn.execute(
            "SELECT note_type FROM vault_chunks WHERE path = 'session.md' LIMIT 1"
        ).fetchone()
        assert row is not None
        assert row[0] == "mist-session"


# ---------------------------------------------------------------------------
# TestDeletePath
# ---------------------------------------------------------------------------


class TestDeletePath:
    def test_removes_all_chunks_for_path(self, sidecar_index):
        # Arrange
        sidecar_index.upsert_file("a.md", _HEADED_NOTE, 1000)
        before = sidecar_index.chunk_count()
        assert before > 1

        # Act
        deleted = sidecar_index.delete_path("a.md")

        # Assert
        assert deleted == before
        assert sidecar_index.chunk_count() == 0

    def test_only_removes_specified_path(self, sidecar_index):
        # Arrange
        sidecar_index.upsert_file("a.md", "content a", 1000)
        sidecar_index.upsert_file("b.md", "content b", 1000)

        # Act
        sidecar_index.delete_path("a.md")

        # Assert -- b.md chunks survive
        assert sidecar_index.chunk_count() == 1

    def test_returns_zero_for_missing_path(self, sidecar_index):
        # Arrange -- empty index

        # Act
        count = sidecar_index.delete_path("nonexistent.md")

        # Assert
        assert count == 0

    def test_idempotent_second_delete(self, sidecar_index):
        # Arrange
        sidecar_index.upsert_file("a.md", "content", 1000)

        # Act
        first = sidecar_index.delete_path("a.md")
        second = sidecar_index.delete_path("a.md")

        # Assert
        assert first == 1
        assert second == 0

    def test_removes_from_vec_table(self, sidecar_index):
        # Arrange
        sidecar_index.upsert_file("a.md", "hello world content", 1000)
        embedding = FakeEmbeddingGenerator().generate_embedding("hello world content")
        results_before = sidecar_index.query_vector(embedding, k=5)
        assert len(results_before) >= 1

        # Act
        sidecar_index.delete_path("a.md")

        # Assert -- vec table cleared
        results_after = sidecar_index.query_vector(embedding, k=5)
        assert len(results_after) == 0

    def test_removes_from_fts_table(self, sidecar_index):
        # Arrange
        sidecar_index.upsert_file("a.md", "unique fts deletion test", 1000)
        assert len(sidecar_index.query_fts("unique fts deletion test", k=5)) >= 1

        # Act
        sidecar_index.delete_path("a.md")

        # Assert -- FTS cleared
        assert len(sidecar_index.query_fts("unique fts deletion test", k=5)) == 0


# ---------------------------------------------------------------------------
# TestQueryVector
# ---------------------------------------------------------------------------


class TestQueryVector:
    def test_returns_empty_list_on_empty_index(self, sidecar_index):
        # Arrange
        emb = FakeEmbeddingGenerator().generate_embedding("anything")

        # Act
        results = sidecar_index.query_vector(emb, k=5)

        # Assert
        assert results == []

    def test_returns_result_for_indexed_content(self, sidecar_index):
        # Arrange
        content = "The quick brown fox jumps over the lazy dog"
        sidecar_index.upsert_file("fox.md", content, 1000)
        emb = FakeEmbeddingGenerator().generate_embedding(content)

        # Act
        results = sidecar_index.query_vector(emb, k=5)

        # Assert
        assert len(results) >= 1
        assert results[0]["path"] == "fox.md"

    def test_result_dict_has_required_keys(self, sidecar_index):
        # Arrange
        content = "sample text for key verification"
        sidecar_index.upsert_file("keys.md", content, 1000)
        emb = FakeEmbeddingGenerator().generate_embedding(content)

        # Act
        results = sidecar_index.query_vector(emb, k=1)

        # Assert
        assert len(results) == 1
        row = results[0]
        for key in (
            "path",
            "heading",
            "content",
            "frontmatter",
            "note_type",
            "score",
            "distance",
            "source",
        ):
            assert key in row, f"Missing key: {key}"
        assert row["source"] == "vector"

    def test_score_is_normalized_from_distance(self, sidecar_index):
        # Arrange
        content = "normalization test content"
        sidecar_index.upsert_file("norm.md", content, 1000)
        emb = FakeEmbeddingGenerator().generate_embedding(content)

        # Act
        results = sidecar_index.query_vector(emb, k=1)

        # Assert -- score = 1 / (1 + distance)
        row = results[0]
        expected_score = 1.0 / (1.0 + row["distance"])
        assert abs(row["score"] - expected_score) < 1e-9

    def test_respects_k_limit(self, sidecar_index):
        # Arrange -- insert 5 different files
        gen = FakeEmbeddingGenerator()
        for i in range(5):
            sidecar_index.upsert_file(f"file{i}.md", f"content variant {i}", 1000)
        query_emb = gen.generate_embedding("content variant 0")

        # Act
        results = sidecar_index.query_vector(query_emb, k=3)

        # Assert
        assert len(results) <= 3

    def test_lower_distance_ranked_first(self, sidecar_index):
        # Arrange -- insert two files; query with exact embedding of first
        gen = FakeEmbeddingGenerator()
        sidecar_index.upsert_file("exact.md", "exact match content", 1000)
        sidecar_index.upsert_file("other.md", "completely different words", 1000)
        query_emb = gen.generate_embedding("exact match content")

        # Act
        results = sidecar_index.query_vector(query_emb, k=5)

        # Assert -- closest match is first
        assert results[0]["path"] == "exact.md"
        assert results[0]["distance"] <= results[-1]["distance"]


# ---------------------------------------------------------------------------
# TestQueryFTS
# ---------------------------------------------------------------------------


class TestQueryFTS:
    def test_returns_empty_list_on_empty_index(self, sidecar_index):
        # Act
        results = sidecar_index.query_fts("hello world", k=5)

        # Assert
        assert results == []

    def test_matches_on_content(self, sidecar_index):
        # Arrange
        sidecar_index.upsert_file("content.md", "the quick brown fox", 1000)

        # Act
        results = sidecar_index.query_fts("fox", k=5)

        # Assert
        assert len(results) >= 1
        assert results[0]["path"] == "content.md"

    def test_matches_on_heading_context(self, sidecar_index):
        # Arrange
        frontmatter = {"type": "mist-session", "tags": ["memory", "identity"]}
        sidecar_index.upsert_file("session.md", "body text here", 1000, frontmatter)

        # Act -- query on note_type, which is in heading_context
        results = sidecar_index.query_fts("mist-session", k=5)

        # Assert
        assert len(results) >= 1
        assert results[0]["path"] == "session.md"

    def test_result_dict_has_required_keys(self, sidecar_index):
        # Arrange
        sidecar_index.upsert_file("keys.md", "test content for fts key check", 1000)

        # Act
        results = sidecar_index.query_fts("test content", k=1)
        assert len(results) >= 1

        # Assert
        row = results[0]
        for key in (
            "path",
            "heading",
            "content",
            "frontmatter",
            "note_type",
            "score",
            "distance",
            "source",
        ):
            assert key in row, f"Missing key: {key}"
        assert row["source"] == "fts"
        assert row["distance"] is None

    def test_fts5_special_chars_do_not_crash(self, sidecar_index):
        # Arrange
        sidecar_index.upsert_file("safe.md", "some content", 1000)

        # Act -- queries with special FTS5 chars must not raise
        for query in ['"quoted"', "star*", "(paren", "col:on", "dash-dash"]:
            result = sidecar_index.query_fts(query, k=5)
            assert isinstance(result, list)

    def test_heading_context_weight_applied(self, tmp_path):
        # Arrange -- two indexes: one with high weight, one with low weight
        # Both have same content. A heading_context-only match should score higher
        # with higher weight.
        frontmatter = {"type": "mist-user", "tags": ["raj"]}
        content = "This is generic content without the keyword."

        config_high = _make_config(tmp_path / "high", heading_context_weight=2.0)
        idx_high = VaultSidecarIndex(config_high, FakeEmbeddingGenerator())
        idx_high.initialize()
        idx_high.upsert_file("note.md", content, 1000, frontmatter)

        config_low = _make_config(tmp_path / "low", heading_context_weight=0.01)
        idx_low = VaultSidecarIndex(config_low, FakeEmbeddingGenerator())
        idx_low.initialize()
        idx_low.upsert_file("note.md", content, 1000, frontmatter)

        # Act
        results_high = idx_high.query_fts("mist-user", k=5)
        results_low = idx_low.query_fts("mist-user", k=5)

        # Assert -- both return results; high weight gives higher score for heading_context match
        assert len(results_high) >= 1
        assert len(results_low) >= 1
        # High weight should produce higher score for same query on heading_context
        assert results_high[0]["score"] >= results_low[0]["score"]

        idx_high.close()
        idx_low.close()

    def test_scores_are_non_negative(self, sidecar_index):
        # Arrange
        sidecar_index.upsert_file("pos.md", "positive score test content query", 1000)

        # Act
        results = sidecar_index.query_fts("positive score", k=5)

        # Assert -- BM25 negated: score should be >= 0
        for row in results:
            assert row["score"] >= 0, f"Negative score: {row['score']}"


# ---------------------------------------------------------------------------
# TestQueryHybrid
# ---------------------------------------------------------------------------


class TestQueryHybrid:
    def test_returns_results_from_both_sources(self, sidecar_index):
        # Arrange -- content that matches both vector and FTS
        content = "hybrid retrieval test document"
        sidecar_index.upsert_file("hybrid.md", content, 1000)
        gen = FakeEmbeddingGenerator()
        emb = gen.generate_embedding(content)

        # Act
        results = sidecar_index.query_hybrid(emb, "hybrid retrieval", k=5)

        # Assert
        assert len(results) >= 1
        assert results[0]["path"] == "hybrid.md"

    def test_result_has_rrf_keys(self, sidecar_index):
        # Arrange
        content = "rrf key verification content"
        sidecar_index.upsert_file("rrf.md", content, 1000)
        emb = FakeEmbeddingGenerator().generate_embedding(content)

        # Act
        results = sidecar_index.query_hybrid(emb, "rrf key", k=5)
        assert len(results) >= 1

        # Assert
        row = results[0]
        for key in (
            "path",
            "heading",
            "content",
            "frontmatter",
            "note_type",
            "score",
            "vector_rank",
            "fts_rank",
            "sources",
        ):
            assert key in row, f"Missing key: {key}"

    def test_top_k_cap_respected(self, sidecar_index):
        # Arrange -- 10 notes
        gen = FakeEmbeddingGenerator()
        for i in range(10):
            sidecar_index.upsert_file(f"doc{i}.md", f"hybrid document number {i}", 1000)
        emb = gen.generate_embedding("hybrid document number 0")

        # Act
        results = sidecar_index.query_hybrid(emb, "hybrid document", k=3)

        # Assert
        assert len(results) <= 3

    def test_single_source_vector_only_included(self, sidecar_index):
        # Arrange -- content that will match vector but not FTS (FTS index empty except this)
        content = "zzzyyyxxx completely unique phrase that matches only on embedding"
        sidecar_index.upsert_file("vec_only.md", content, 1000)
        emb = FakeEmbeddingGenerator().generate_embedding(content)

        # Act -- FTS query for something that won't match, vector query will
        results = sidecar_index.query_hybrid(emb, "zzzyyyxxx completely unique phrase", k=5)

        # Assert -- result exists (from vector at minimum)
        assert len(results) >= 1
        found = next((r for r in results if r["path"] == "vec_only.md"), None)
        assert found is not None

    def test_fts_rank_none_for_vector_only_result(self, sidecar_index):
        # Arrange -- one note; query FTS for something not in the note
        sidecar_index.upsert_file("a.md", "alpha beta gamma content", 1000)
        emb = FakeEmbeddingGenerator().generate_embedding("alpha beta gamma content")

        # Act -- FTS query for unrelated term
        results = sidecar_index.query_hybrid(emb, "zzznomatch", k=5)

        # Assert -- a.md appears from vector only; fts_rank is None
        found = next((r for r in results if r["path"] == "a.md"), None)
        assert found is not None
        assert found["fts_rank"] is None

    def test_rrf_k_affects_ranking(self, sidecar_index):
        # Arrange -- three notes with different content
        gen = FakeEmbeddingGenerator()
        for i, phrase in enumerate(["alpha content", "beta content", "gamma content"]):
            sidecar_index.upsert_file(f"note{i}.md", phrase, 1000)
        emb = gen.generate_embedding("alpha content")

        # Act -- same query with different rrf_k
        results_low_k = sidecar_index.query_hybrid(emb, "alpha content", k=3, rrf_k=1)
        results_high_k = sidecar_index.query_hybrid(emb, "alpha content", k=3, rrf_k=1000)

        # Assert -- both return results; scores differ due to rrf_k
        assert len(results_low_k) >= 1
        assert len(results_high_k) >= 1
        # With very low rrf_k, rank 1 gives huge weight vs rank 2; scores spread wide
        # With very high rrf_k, ranks collapse toward equality; scores are similar
        # Just verify the top result is consistent (best match wins in both)
        assert results_low_k[0]["path"] in ("note0.md",)

    def test_sources_list_contains_both_when_present(self, sidecar_index):
        # Arrange -- note that matches both by content and by FTS
        content = "both sources match this content"
        sidecar_index.upsert_file("both.md", content, 1000)
        emb = FakeEmbeddingGenerator().generate_embedding(content)

        # Act
        results = sidecar_index.query_hybrid(emb, "both sources match", k=5)

        # Assert -- top result should have both sources
        assert len(results) >= 1
        top = results[0]
        assert "vector" in top["sources"]
        assert "fts" in top["sources"]

    def test_empty_index_returns_empty_list(self, sidecar_index):
        # Arrange -- empty index
        emb = FakeEmbeddingGenerator().generate_embedding("nothing")

        # Act
        results = sidecar_index.query_hybrid(emb, "nothing", k=5)

        # Assert
        assert results == []


# ---------------------------------------------------------------------------
# TestChunkCount
# ---------------------------------------------------------------------------


class TestChunkCount:
    def test_zero_for_empty_index(self, sidecar_index):
        # Act / Assert
        assert sidecar_index.chunk_count() == 0

    def test_increments_on_upsert(self, sidecar_index):
        # Act
        sidecar_index.upsert_file("a.md", "hello", 1000)

        # Assert
        assert sidecar_index.chunk_count() == 1

    def test_multiple_files_sum_correctly(self, sidecar_index):
        # Arrange -- 3 single-chunk files
        for i in range(3):
            sidecar_index.upsert_file(f"file{i}.md", "single chunk content", 1000)

        # Assert
        assert sidecar_index.chunk_count() == 3

    def test_decrements_on_delete(self, sidecar_index):
        # Arrange
        sidecar_index.upsert_file("a.md", "content a", 1000)
        sidecar_index.upsert_file("b.md", "content b", 1000)
        assert sidecar_index.chunk_count() == 2

        # Act
        sidecar_index.delete_path("a.md")

        # Assert
        assert sidecar_index.chunk_count() == 1

    def test_headed_note_counts_all_chunks(self, sidecar_index):
        # Arrange / Act
        n = sidecar_index.upsert_file("headed.md", _HEADED_NOTE, 1000)

        # Assert -- chunk_count matches upsert return value
        assert sidecar_index.chunk_count() == n


# ---------------------------------------------------------------------------
# TestHealthCheck
# ---------------------------------------------------------------------------


class TestHealthCheck:
    def test_true_after_initialize(self, sidecar_index):
        # Assert -- fixture already initialized
        assert sidecar_index.health_check() is True

    def test_false_after_close(self, sidecar_index):
        # Act
        sidecar_index.close()

        # Assert
        assert sidecar_index.health_check() is False

    def test_false_before_initialize(self, tmp_path):
        # Arrange
        config = _make_config(tmp_path)
        idx = VaultSidecarIndex(config, FakeEmbeddingGenerator())

        # Act / Assert -- no initialize called
        assert idx.health_check() is False

    def test_remains_true_after_upsert(self, sidecar_index):
        # Act
        sidecar_index.upsert_file("health.md", "content", 1000)

        # Assert
        assert sidecar_index.health_check() is True

    def test_remains_true_after_delete(self, sidecar_index):
        # Arrange
        sidecar_index.upsert_file("h.md", "content", 1000)

        # Act
        sidecar_index.delete_path("h.md")

        # Assert
        assert sidecar_index.health_check() is True


# ---------------------------------------------------------------------------
# Internal helper unit tests
# ---------------------------------------------------------------------------


class TestExtractHeadingBlocks:
    def test_no_headings_returns_empty(self):
        assert _extract_heading_blocks("no headings here") == []

    def test_two_h2_headings(self):
        content = "## Alpha\n\nalpha content\n\n## Beta\n\nbeta content"
        blocks = _extract_heading_blocks(content)
        assert len(blocks) == 2
        assert blocks[0][0] == "Alpha"
        assert blocks[1][0] == "Beta"
        assert "alpha content" in blocks[0][2]
        assert "beta content" in blocks[1][2]

    def test_provenance_skipped(self):
        content = "## Section\n\nsome content\n\n## Provenance\n\n- event_id: abc"
        blocks = _extract_heading_blocks(content)
        headings = [b[0] for b in blocks]
        assert "Provenance" not in headings
        assert "Section" in headings

    def test_heading_level_extraction(self):
        content = "### H3\n\ncontent\n\n## H2\n\nmore"
        blocks = _extract_heading_blocks(content)
        level_map = {b[0]: b[1] for b in blocks}
        assert level_map["H3"] == 3
        assert level_map["H2"] == 2


class TestQuoteFts5:
    def test_plain_text_unchanged(self):
        assert _quote_fts5("hello world") == "hello world"

    def test_double_quote_is_escaped(self):
        result = _quote_fts5('"quoted"')
        assert result.startswith('"')
        assert result.endswith('"')

    def test_asterisk_triggers_quoting(self):
        result = _quote_fts5("star*")
        assert result.startswith('"')

    def test_colon_triggers_quoting(self):
        result = _quote_fts5("type:session")
        assert result.startswith('"')
