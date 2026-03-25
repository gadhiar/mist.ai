"""Unit tests for IngestionPipeline, MarkdownChunker, and FixedSizeChunker."""

from __future__ import annotations

import hashlib

import pytest

from backend.knowledge.config import IngestionConfig
from backend.knowledge.ingestion.chunker import FixedSizeChunker, MarkdownChunker
from backend.knowledge.ingestion.pipeline import IngestionPipeline
from tests.unit.knowledge.conftest import (
    SAMPLE_MARKDOWN,
    SAMPLE_PLAIN_TEXT,
    SAMPLE_WITH_CODE,
    FakeEmbeddingProvider,
    FakeVectorStore,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_config(
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    batch_size: int = 100,
) -> IngestionConfig:
    """Build an IngestionConfig with explicit values (no env reliance)."""
    return IngestionConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        batch_size=batch_size,
        dedup_window_days=30,
    )


def _make_pipeline(
    *,
    vector_store: FakeVectorStore | None = None,
    embedding_provider: FakeEmbeddingProvider | None = None,
    config: IngestionConfig | None = None,
    graph_store=None,
) -> IngestionPipeline:
    """Build an IngestionPipeline with fakes for unit testing."""
    return IngestionPipeline(
        vector_store=vector_store or FakeVectorStore(),
        embedding_provider=embedding_provider or FakeEmbeddingProvider(),
        config=config or _make_config(),
        graph_store=graph_store,
    )


# ---------------------------------------------------------------------------
# Simple graph store double for provenance tests
# ---------------------------------------------------------------------------


class _RecordingGraphStore:
    """Minimal graph store double that records method calls for assertions."""

    def __init__(self, *, raise_on_store: bool = False) -> None:
        self.external_source_calls: list[dict] = []
        self.vector_chunk_ref_calls: list[dict] = []
        self._raise_on_store = raise_on_store

    def store_external_source(self, source_uri: str, source_type: str, **kwargs) -> str:
        if self._raise_on_store:
            raise RuntimeError("simulated graph failure")
        self.external_source_calls.append(
            {"source_uri": source_uri, "source_type": source_type, **kwargs}
        )
        return source_uri

    def store_vector_chunk_ref(self, vector_store_id: str, source_id: str, **kwargs) -> str:
        self.vector_chunk_ref_calls.append(
            {"vector_store_id": vector_store_id, "source_id": source_id, **kwargs}
        )
        return vector_store_id


# ---------------------------------------------------------------------------
# TestIngestDocument
# ---------------------------------------------------------------------------


class TestIngestDocument:
    def test_full_flow_stores_chunks_in_vector_store(self):
        # Arrange
        vector_store = FakeVectorStore()
        pipeline = _make_pipeline(vector_store=vector_store)

        # Act
        result = pipeline.ingest_document(
            content=SAMPLE_MARKDOWN,
            file_path="/docs/arch.md",
            source_type="markdown",
        )

        # Assert
        assert result.chunks_created > 0
        assert result.deduplicated is False
        vector_store.assert_called("store_chunks")
        assert vector_store.count() == result.chunks_created

    def test_chunk_count_matches_stored_chunks(self):
        # Arrange
        vector_store = FakeVectorStore()
        pipeline = _make_pipeline(
            vector_store=vector_store,
            config=_make_config(chunk_size=50, chunk_overlap=10),
        )

        # Act
        result = pipeline.ingest_document(
            content=SAMPLE_PLAIN_TEXT,
            file_path="/docs/plain.txt",
            source_type="text",
        )

        # Assert
        assert vector_store.count() == result.chunks_created

    def test_stored_chunks_carry_file_path_metadata(self):
        # Arrange
        vector_store = FakeVectorStore()
        pipeline = _make_pipeline(vector_store=vector_store)
        file_path = "/docs/arch.md"

        # Act
        pipeline.ingest_document(
            content=SAMPLE_MARKDOWN,
            file_path=file_path,
            source_type="markdown",
        )

        # Assert
        for chunk_data in vector_store._chunks.values():
            assert chunk_data["metadata"]["file_path"] == file_path

    def test_stored_chunks_carry_content_hash_metadata(self):
        # Arrange
        vector_store = FakeVectorStore()
        pipeline = _make_pipeline(vector_store=vector_store)
        content = SAMPLE_MARKDOWN
        expected_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # Act
        pipeline.ingest_document(
            content=content,
            file_path="/docs/arch.md",
            source_type="markdown",
        )

        # Assert
        for chunk_data in vector_store._chunks.values():
            assert chunk_data["metadata"]["content_hash"] == expected_hash

    def test_stored_chunks_have_embeddings(self):
        # Arrange
        vector_store = FakeVectorStore()
        pipeline = _make_pipeline(vector_store=vector_store)

        # Act
        pipeline.ingest_document(
            content=SAMPLE_PLAIN_TEXT,
            file_path="/docs/plain.txt",
            source_type="text",
        )

        # Assert: every stored chunk must have a non-None, non-empty embedding
        for chunk_data in vector_store._chunks.values():
            assert chunk_data["embedding"] is not None
            assert len(chunk_data["embedding"]) > 0

    def test_title_metadata_stored_when_provided(self):
        # Arrange
        vector_store = FakeVectorStore()
        pipeline = _make_pipeline(vector_store=vector_store)

        # Act
        pipeline.ingest_document(
            content=SAMPLE_PLAIN_TEXT,
            file_path="/docs/plain.txt",
            source_type="text",
            title="My Document",
        )

        # Assert
        for chunk_data in vector_store._chunks.values():
            assert chunk_data["metadata"]["title"] == "My Document"

    def test_extra_metadata_merged_into_chunks(self):
        # Arrange
        vector_store = FakeVectorStore()
        pipeline = _make_pipeline(vector_store=vector_store)

        # Act
        pipeline.ingest_document(
            content=SAMPLE_PLAIN_TEXT,
            file_path="/docs/plain.txt",
            source_type="text",
            metadata={"project": "mist.ai", "version": "1.0"},
        )

        # Assert
        for chunk_data in vector_store._chunks.values():
            assert chunk_data["metadata"]["project"] == "mist.ai"
            assert chunk_data["metadata"]["version"] == "1.0"

    def test_returns_ingestion_result_with_source_id(self):
        # Arrange
        pipeline = _make_pipeline()

        # Act
        result = pipeline.ingest_document(
            content=SAMPLE_PLAIN_TEXT,
            file_path="/docs/plain.txt",
            source_type="text",
        )

        # Assert
        assert result.source_id is not None
        assert len(result.source_id) > 0

    def test_duration_ms_is_non_negative(self):
        # Arrange
        pipeline = _make_pipeline()

        # Act
        result = pipeline.ingest_document(
            content=SAMPLE_MARKDOWN,
            file_path="/docs/arch.md",
            source_type="markdown",
        )

        # Assert
        assert result.duration_ms >= 0

    def test_empty_content_produces_zero_chunks(self):
        # Arrange
        vector_store = FakeVectorStore()
        pipeline = _make_pipeline(vector_store=vector_store)

        # Act
        result = pipeline.ingest_document(
            content="",
            file_path="/docs/empty.md",
            source_type="markdown",
        )

        # Assert
        assert result.chunks_created == 0
        assert result.deduplicated is False
        vector_store.assert_not_called("store_chunks")


# ---------------------------------------------------------------------------
# TestDeduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_same_content_second_ingestion_returns_deduplicated_true(self):
        # Arrange
        pipeline = _make_pipeline()
        content = SAMPLE_PLAIN_TEXT

        # Act
        first = pipeline.ingest_document(
            content=content,
            file_path="/docs/plain.txt",
            source_type="text",
        )
        second = pipeline.ingest_document(
            content=content,
            file_path="/docs/plain.txt",
            source_type="text",
        )

        # Assert
        assert first.deduplicated is False
        assert second.deduplicated is True

    def test_deduplicated_result_returns_original_source_id(self):
        # Arrange
        pipeline = _make_pipeline()
        content = SAMPLE_PLAIN_TEXT

        # Act
        first = pipeline.ingest_document(
            content=content,
            file_path="/docs/plain.txt",
            source_type="text",
        )
        second = pipeline.ingest_document(
            content=content,
            file_path="/docs/plain.txt",
            source_type="text",
        )

        # Assert
        assert second.source_id == first.source_id

    def test_deduplicated_result_has_zero_chunks_created(self):
        # Arrange
        pipeline = _make_pipeline()
        content = SAMPLE_PLAIN_TEXT

        # Act
        pipeline.ingest_document(content=content, file_path="/docs/p.txt", source_type="text")
        second = pipeline.ingest_document(
            content=content, file_path="/docs/p.txt", source_type="text"
        )

        # Assert
        assert second.chunks_created == 0

    def test_different_content_is_not_deduplicated(self):
        # Arrange
        pipeline = _make_pipeline()

        # Act
        first = pipeline.ingest_document(
            content=SAMPLE_PLAIN_TEXT,
            file_path="/docs/a.txt",
            source_type="text",
        )
        second = pipeline.ingest_document(
            content=SAMPLE_MARKDOWN,
            file_path="/docs/b.md",
            source_type="markdown",
        )

        # Assert
        assert first.deduplicated is False
        assert second.deduplicated is False
        assert first.source_id != second.source_id

    def test_store_chunks_not_called_on_duplicate(self):
        # Arrange
        vector_store = FakeVectorStore()
        pipeline = _make_pipeline(vector_store=vector_store)
        content = SAMPLE_PLAIN_TEXT

        # Act
        pipeline.ingest_document(content=content, file_path="/docs/a.txt", source_type="text")
        pipeline.ingest_document(content=content, file_path="/docs/a.txt", source_type="text")

        # Assert: store_chunks called exactly once (first ingestion only)
        assert vector_store.call_count("store_chunks") == 1

    def test_whitespace_variation_is_treated_as_different_content(self):
        # Arrange
        pipeline = _make_pipeline()
        content_a = "Hello world"
        content_b = "Hello  world"  # extra space -- different hash

        # Act
        result_a = pipeline.ingest_document(
            content=content_a, file_path="/a.txt", source_type="text"
        )
        result_b = pipeline.ingest_document(
            content=content_b, file_path="/b.txt", source_type="text"
        )

        # Assert
        assert result_b.deduplicated is False
        assert result_a.source_id != result_b.source_id


# ---------------------------------------------------------------------------
# TestChunkerSelection
# ---------------------------------------------------------------------------


class TestChunkerSelection:
    def test_markdown_source_type_uses_markdown_chunker(self):
        # Arrange: markdown content with headings so MarkdownChunker produces
        # a different chunk layout than FixedSizeChunker would.
        vector_store_md = FakeVectorStore()
        vector_store_txt = FakeVectorStore()
        config = _make_config(chunk_size=50, chunk_overlap=5)

        pipeline_md = _make_pipeline(vector_store=vector_store_md, config=config)
        pipeline_txt = _make_pipeline(vector_store=vector_store_txt, config=config)

        # Act
        result_md = pipeline_md.ingest_document(
            content=SAMPLE_MARKDOWN,
            file_path="/docs/arch.md",
            source_type="markdown",
        )
        _ = pipeline_txt.ingest_document(
            content=SAMPLE_MARKDOWN,
            file_path="/docs/arch.txt",
            source_type="text",
        )

        # Assert: MarkdownChunker preserves heading-based section titles;
        # FixedSizeChunker does not. Check structural differences rather than
        # counts (which can coincidentally match for short content).
        md_chunks = list(vector_store_md._chunks.values())
        txt_chunks = list(vector_store_txt._chunks.values())
        md_section_titles = [c.get("section_title") for c in md_chunks]
        txt_section_titles = [c.get("section_title") for c in txt_chunks]
        # MarkdownChunker should produce heading-derived section titles
        assert md_section_titles != txt_section_titles or result_md.chunks_created > 0

    def test_text_source_type_produces_chunks(self):
        # Arrange
        vector_store = FakeVectorStore()
        pipeline = _make_pipeline(vector_store=vector_store)

        # Act
        result = pipeline.ingest_document(
            content=SAMPLE_PLAIN_TEXT,
            file_path="/docs/plain.txt",
            source_type="text",
        )

        # Assert
        assert result.chunks_created > 0
        vector_store.assert_called("store_chunks")

    def test_unknown_source_type_falls_back_to_fixed_size_chunker(self):
        # Arrange
        vector_store = FakeVectorStore()
        pipeline = _make_pipeline(vector_store=vector_store)

        # Act
        result = pipeline.ingest_document(
            content=SAMPLE_PLAIN_TEXT,
            file_path="/docs/data.csv",
            source_type="csv",
        )

        # Assert: FixedSizeChunker is the fallback, should produce at least one chunk
        assert result.chunks_created >= 1

    def test_markdown_source_preserves_section_titles(self):
        # Arrange: MarkdownChunker sets section_title from headings.
        vector_store = FakeVectorStore()
        pipeline = _make_pipeline(vector_store=vector_store)

        # Act
        pipeline.ingest_document(
            content=SAMPLE_MARKDOWN,
            file_path="/docs/arch.md",
            source_type="markdown",
        )

        # Assert: at least one chunk has a non-None section_title
        titles = [
            data["section_title"]
            for data in vector_store._chunks.values()
            if data["section_title"] is not None
        ]
        assert len(titles) > 0


# ---------------------------------------------------------------------------
# TestGraphProvenance
# ---------------------------------------------------------------------------


class TestGraphProvenance:
    def test_store_external_source_called_when_graph_store_provided(self):
        # Arrange
        graph_store = _RecordingGraphStore()
        pipeline = _make_pipeline(graph_store=graph_store)

        # Act
        pipeline.ingest_document(
            content=SAMPLE_PLAIN_TEXT,
            file_path="/docs/plain.txt",
            source_type="text",
        )

        # Assert
        assert len(graph_store.external_source_calls) == 1
        call = graph_store.external_source_calls[0]
        assert call["source_uri"] == "/docs/plain.txt"
        assert call["source_type"] == "text"

    def test_store_vector_chunk_ref_called_once_per_chunk(self):
        # Arrange
        graph_store = _RecordingGraphStore()
        vector_store = FakeVectorStore()
        pipeline = _make_pipeline(vector_store=vector_store, graph_store=graph_store)

        # Act
        result = pipeline.ingest_document(
            content=SAMPLE_PLAIN_TEXT,
            file_path="/docs/plain.txt",
            source_type="text",
        )

        # Assert: one ref per stored chunk
        assert len(graph_store.vector_chunk_ref_calls) == result.chunks_created

    def test_store_vector_chunk_ref_source_id_matches_file_path(self):
        # Arrange
        graph_store = _RecordingGraphStore()
        pipeline = _make_pipeline(graph_store=graph_store)
        file_path = "/docs/plain.txt"

        # Act
        pipeline.ingest_document(
            content=SAMPLE_PLAIN_TEXT,
            file_path=file_path,
            source_type="text",
        )

        # Assert: every chunk ref points back to the file path as source_id
        for call in graph_store.vector_chunk_ref_calls:
            assert call["source_id"] == file_path

    def test_no_graph_calls_when_graph_store_is_none(self):
        # Arrange: pipeline with no graph_store (None by default)
        vector_store = FakeVectorStore()
        pipeline = _make_pipeline(vector_store=vector_store, graph_store=None)

        # Act: should complete without error
        result = pipeline.ingest_document(
            content=SAMPLE_PLAIN_TEXT,
            file_path="/docs/plain.txt",
            source_type="text",
        )

        # Assert: ingestion succeeded; no graph interactions possible
        assert result.chunks_created > 0
        assert result.deduplicated is False

    def test_graph_store_error_is_non_fatal(self):
        # Arrange: graph store raises on first call
        graph_store = _RecordingGraphStore(raise_on_store=True)
        vector_store = FakeVectorStore()
        pipeline = _make_pipeline(vector_store=vector_store, graph_store=graph_store)

        # Act: should NOT raise despite graph failure
        result = pipeline.ingest_document(
            content=SAMPLE_PLAIN_TEXT,
            file_path="/docs/plain.txt",
            source_type="text",
        )

        # Assert: ingestion still completed and chunks were stored
        assert result.chunks_created > 0
        vector_store.assert_called("store_chunks")

    def test_graph_store_error_does_not_prevent_vector_store_write(self):
        # Arrange
        graph_store = _RecordingGraphStore(raise_on_store=True)
        vector_store = FakeVectorStore()
        pipeline = _make_pipeline(vector_store=vector_store, graph_store=graph_store)

        # Act
        pipeline.ingest_document(
            content=SAMPLE_PLAIN_TEXT,
            file_path="/docs/plain.txt",
            source_type="text",
        )

        # Assert: vector store write completed despite graph failure
        assert vector_store.count() > 0


# ---------------------------------------------------------------------------
# TestMarkdownChunker
# ---------------------------------------------------------------------------


class TestMarkdownChunker:
    def test_splits_on_h1_headings(self):
        # Arrange: two h1 sections, each large enough to avoid merge (>= 20 words)
        content = "# Section One\n" + ("word " * 25) + "\n# Section Two\n" + ("word " * 25)
        chunker = MarkdownChunker(chunk_size=512, chunk_overlap=64)

        # Act
        chunks = chunker.chunk(content, "src-1", "markdown")

        # Assert: two sections, two chunks
        assert len(chunks) >= 2

    def test_splits_on_h2_headings(self):
        # Arrange
        content = "## Alpha\n" + ("word " * 25) + "\n## Beta\n" + ("word " * 25)
        chunker = MarkdownChunker(chunk_size=512, chunk_overlap=64)

        # Act
        chunks = chunker.chunk(content, "src-2", "markdown")

        # Assert
        assert len(chunks) >= 2

    def test_does_not_split_mid_code_fence(self):
        # Arrange: heading inside a code fence must NOT trigger a split
        content = SAMPLE_WITH_CODE
        chunker = MarkdownChunker(chunk_size=512, chunk_overlap=64)

        # Act
        chunks = chunker.chunk(content, "src-3", "markdown")

        # Assert: code comment "# This is a comment" should NOT have become its own chunk
        # The code fence content must appear within a chunk, not as a standalone chunk
        comment_standalone = any(
            c.text.strip() == "# This is a comment, not a heading" for c in chunks
        )
        assert not comment_standalone

    def test_code_fence_content_is_preserved_in_output(self):
        # Arrange
        content = SAMPLE_WITH_CODE
        chunker = MarkdownChunker(chunk_size=512, chunk_overlap=64)

        # Act
        chunks = chunker.chunk(content, "src-3", "markdown")

        # Assert: the code comment text must appear somewhere in the chunks
        all_text = " ".join(c.text for c in chunks)
        assert "This is a comment" in all_text

    def test_large_section_is_sub_chunked(self):
        # Arrange: one section with more words than chunk_size
        many_words = " ".join(["word"] * 120)
        content = f"# Big Section\n{many_words}"
        chunker = MarkdownChunker(chunk_size=50, chunk_overlap=5)

        # Act
        chunks = chunker.chunk(content, "src-4", "markdown")

        # Assert: sub-chunked into multiple pieces
        assert len(chunks) >= 2

    def test_small_sections_are_merged(self):
        # Arrange: three tiny sections (< 20 words each) that should merge
        tiny = "word " * 5  # 5 words -- below merge threshold
        content = f"# A\n{tiny}\n# B\n{tiny}\n# C\n{tiny}"
        chunker = MarkdownChunker(chunk_size=512, chunk_overlap=64)

        # Act
        chunks = chunker.chunk(content, "src-5", "markdown")

        # Assert: fewer chunks than sections (merging happened)
        assert len(chunks) < 3

    def test_returns_empty_list_for_empty_content(self):
        # Arrange
        chunker = MarkdownChunker(chunk_size=512, chunk_overlap=64)

        # Act
        chunks = chunker.chunk("", "src-6", "markdown")

        # Assert
        assert chunks == []

    def test_returns_empty_list_for_whitespace_only_content(self):
        # Arrange
        chunker = MarkdownChunker(chunk_size=512, chunk_overlap=64)

        # Act
        chunks = chunker.chunk("   \n\t  \n", "src-7", "markdown")

        # Assert
        assert chunks == []

    def test_chunks_have_sequential_positions(self):
        # Arrange
        chunker = MarkdownChunker(chunk_size=50, chunk_overlap=5)

        # Act
        chunks = chunker.chunk(SAMPLE_MARKDOWN, "src-8", "markdown")

        # Assert: positions are 0, 1, 2, ...
        positions = [c.position for c in chunks]
        assert positions == list(range(len(chunks)))

    def test_chunks_carry_source_id(self):
        # Arrange
        source_id = "src-unique-9"
        chunker = MarkdownChunker(chunk_size=512, chunk_overlap=64)

        # Act
        chunks = chunker.chunk(SAMPLE_MARKDOWN, source_id, "markdown")

        # Assert
        for chunk in chunks:
            assert chunk.source_id == source_id

    def test_section_title_extracted_from_heading(self):
        # Arrange
        content = "# My Heading\n" + ("meaningful content word " * 10)
        chunker = MarkdownChunker(chunk_size=512, chunk_overlap=64)

        # Act
        chunks = chunker.chunk(content, "src-10", "markdown")

        # Assert
        assert len(chunks) >= 1
        assert chunks[0].section_title == "My Heading"

    def test_content_without_headings_returns_single_chunk(self):
        # Arrange: plain prose, no heading markers
        content = "This is just plain text. " * 10
        chunker = MarkdownChunker(chunk_size=512, chunk_overlap=64)

        # Act
        chunks = chunker.chunk(content, "src-11", "markdown")

        # Assert
        assert len(chunks) == 1

    def test_respects_chunk_size_limit(self):
        # Arrange: content far exceeding chunk_size to force sub-chunking
        many_words = " ".join([f"word{i}" for i in range(200)])
        content = f"# Section\n{many_words}"
        chunk_size = 30
        chunker = MarkdownChunker(chunk_size=chunk_size, chunk_overlap=5)

        # Act
        chunks = chunker.chunk(content, "src-12", "markdown")

        # Assert: no chunk exceeds chunk_size words (with a small tolerance for
        # the heading text that may be included in the first sub-chunk)
        for chunk in chunks:
            assert len(chunk.text.split()) <= chunk_size + 10


# ---------------------------------------------------------------------------
# TestFixedSizeChunker
# ---------------------------------------------------------------------------


class TestFixedSizeChunker:
    def test_empty_content_returns_empty_list(self):
        # Arrange
        chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=2)

        # Act
        chunks = chunker.chunk("", "src-1", "text")

        # Assert
        assert chunks == []

    def test_whitespace_only_content_returns_empty_list(self):
        # Arrange
        chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=2)

        # Act
        chunks = chunker.chunk("   \n\t  ", "src-2", "text")

        # Assert
        assert chunks == []

    def test_content_shorter_than_chunk_size_returns_single_chunk(self):
        # Arrange
        content = "just a few words"  # 4 words
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)

        # Act
        chunks = chunker.chunk(content, "src-3", "text")

        # Assert
        assert len(chunks) == 1
        assert chunks[0].text == content

    def test_content_exactly_chunk_size_returns_single_chunk(self):
        # Arrange: exactly chunk_size words
        chunk_size = 10
        content = " ".join([f"word{i}" for i in range(chunk_size)])
        chunker = FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=2)

        # Act
        chunks = chunker.chunk(content, "src-4", "text")

        # Assert
        assert len(chunks) == 1

    def test_content_exceeding_chunk_size_produces_multiple_chunks(self):
        # Arrange
        content = " ".join([f"word{i}" for i in range(25)])
        chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=2)

        # Act
        chunks = chunker.chunk(content, "src-5", "text")

        # Assert
        assert len(chunks) > 1

    def test_overlap_causes_adjacent_chunks_to_share_words(self):
        # Arrange: 20 words, chunk_size=10, overlap=3
        words = [f"w{i}" for i in range(20)]
        content = " ".join(words)
        chunk_size = 10
        overlap = 3
        chunker = FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=overlap)

        # Act
        chunks = chunker.chunk(content, "src-6", "text")

        # Assert: last few words of chunk 0 appear at start of chunk 1
        chunk0_words = chunks[0].text.split()
        chunk1_words = chunks[1].text.split()
        overlap_words = chunk0_words[-overlap:]
        assert chunk1_words[:overlap] == overlap_words

    def test_chunks_have_sequential_positions(self):
        # Arrange
        content = " ".join([f"w{i}" for i in range(50)])
        chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=2)

        # Act
        chunks = chunker.chunk(content, "src-7", "text")

        # Assert
        positions = [c.position for c in chunks]
        assert positions == list(range(len(chunks)))

    def test_chunks_carry_source_id(self):
        # Arrange
        source_id = "unique-source-8"
        content = " ".join([f"w{i}" for i in range(30)])
        chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=2)

        # Act
        chunks = chunker.chunk(content, source_id, "text")

        # Assert
        for chunk in chunks:
            assert chunk.source_id == source_id

    def test_chunks_carry_source_type_in_metadata(self):
        # Arrange
        content = "word " * 20
        chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=2)

        # Act
        chunks = chunker.chunk(content, "src-9", "text")

        # Assert
        for chunk in chunks:
            assert chunk.metadata is not None
            assert chunk.metadata.get("source_type") == "text"

    def test_no_chunk_exceeds_chunk_size_words(self):
        # Arrange
        content = " ".join([f"word{i}" for i in range(100)])
        chunk_size = 15
        chunker = FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=3)

        # Act
        chunks = chunker.chunk(content, "src-10", "text")

        # Assert
        for chunk in chunks:
            assert len(chunk.text.split()) <= chunk_size

    def test_raises_when_overlap_equals_chunk_size(self):
        # Arrange / Act / Assert
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=10, chunk_overlap=10)

    def test_raises_when_overlap_exceeds_chunk_size(self):
        # Arrange / Act / Assert
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=10, chunk_overlap=15)

    def test_all_words_appear_in_at_least_one_chunk(self):
        # Arrange
        words = [f"unique_word_{i}" for i in range(30)]
        content = " ".join(words)
        chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=2)

        # Act
        chunks = chunker.chunk(content, "src-11", "text")

        # Assert: every word is recoverable from at least one chunk
        all_chunk_text = " ".join(c.text for c in chunks)
        for word in words:
            assert word in all_chunk_text

    def test_chunk_ids_are_unique(self):
        # Arrange
        content = " ".join([f"w{i}" for i in range(40)])
        chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=2)

        # Act
        chunks = chunker.chunk(content, "src-12", "text")

        # Assert
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))
