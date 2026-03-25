"""Unit tests for LanceDBVectorStore.

Tests cover the public VectorStoreProvider API and the internal filter
builder. LanceDB is mocked at the module boundary -- all tests verify
our code's behaviour in response to the fake database's output, not
LanceDB's own correctness.
"""

from unittest.mock import MagicMock, patch

import pytest

from backend.errors import VectorStoreError
from backend.knowledge.config import VectorStoreConfig
from backend.knowledge.models import DocumentChunk
from backend.knowledge.storage.vector_store import LanceDBVectorStore

MODULE = "backend.knowledge.storage.vector_store"

EMBEDDING_DIM = 384


# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------


def build_vector_store_config(
    *,
    data_dir: str = "/tmp/test_vector_store",
    similarity_threshold: float = 0.6,
    batch_size: int = 100,
    collection_name: str = "test_chunks",
) -> VectorStoreConfig:
    """Build a VectorStoreConfig with test defaults."""
    return VectorStoreConfig(
        backend="lancedb",
        data_dir=data_dir,
        similarity_threshold=similarity_threshold,
        batch_size=batch_size,
        collection_name=collection_name,
    )


_SENTINEL = object()


def build_chunk(
    *,
    chunk_id: str = "chunk-001",
    source_id: str = "src-001",
    text: str = "Test chunk text.",
    position: int = 0,
    embedding: list[float] | None = _SENTINEL,
    section_title: str | None = "Intro",
    metadata: dict | None = _SENTINEL,
) -> DocumentChunk:
    """Build a DocumentChunk with sensible test defaults.

    Embedding defaults to a 384-dimensional zero vector so that a
    zero-argument call produces a storable chunk. Pass ``embedding=None``
    explicitly to create a chunk without an embedding (for error-path tests).
    Same for ``metadata=None``.
    """
    if embedding is _SENTINEL:
        embedding = [0.0] * EMBEDDING_DIM
    if metadata is _SENTINEL:
        metadata = {"source_type": "markdown"}
    return DocumentChunk(
        chunk_id=chunk_id,
        source_id=source_id,
        text=text,
        position=position,
        embedding=embedding,
        section_title=section_title,
        metadata=metadata,
    )


def build_search_row(
    *,
    chunk_id: str = "chunk-001",
    text: str = "Test chunk text.",
    source_id: str = "src-001",
    source_type: str = "markdown",
    distance: float = 0.1,
    metadata: str | None = None,
) -> dict:
    """Build a raw row dict as returned by lancedb table.search().to_list()."""
    return {
        "chunk_id": chunk_id,
        "text": text,
        "source_id": source_id,
        "source_type": source_type,
        "_distance": distance,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Shared fixture: patched lancedb + connected store
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_table():
    """A mock LanceDB table with sensible defaults."""
    table = MagicMock()
    table.count_rows.return_value = 0
    # Chain: table.search(...).metric(...).limit(...).to_list()
    search_chain = MagicMock()
    search_chain.metric.return_value = search_chain
    search_chain.limit.return_value = search_chain
    search_chain.where.return_value = search_chain
    search_chain.to_list.return_value = []
    table.search.return_value = search_chain
    return table


@pytest.fixture
def mock_db(mock_table):
    """A mock LanceDB connection that returns mock_table for any table operation."""
    db = MagicMock()
    db.table_names.return_value = []
    db.create_table.return_value = mock_table
    db.open_table.return_value = mock_table
    return db


@pytest.fixture
def connected_store(mock_db):
    """A LanceDBVectorStore that has been connected via a patched lancedb.connect."""
    config = build_vector_store_config()
    store = LanceDBVectorStore(config=config)

    with patch(f"{MODULE}.lancedb.connect", return_value=mock_db), patch("pathlib.Path.mkdir"):
        store.connect()

    return store


# ---------------------------------------------------------------------------
# TestConnect
# ---------------------------------------------------------------------------


class TestConnect:
    def test_creates_new_table_when_collection_absent(self, mock_db):
        # Arrange
        config = build_vector_store_config(collection_name="chunks")
        mock_db.table_names.return_value = []
        store = LanceDBVectorStore(config=config)

        # Act
        with patch(f"{MODULE}.lancedb.connect", return_value=mock_db), patch("pathlib.Path.mkdir"):
            store.connect()

        # Assert
        mock_db.create_table.assert_called_once()
        call_kwargs = mock_db.create_table.call_args
        assert call_kwargs[0][0] == "chunks"

    def test_opens_existing_table_when_collection_present(self, mock_db, mock_table):
        # Arrange
        config = build_vector_store_config(collection_name="chunks")
        mock_db.table_names.return_value = ["chunks"]
        mock_db.open_table.return_value = mock_table
        store = LanceDBVectorStore(config=config)

        # Act
        with patch(f"{MODULE}.lancedb.connect", return_value=mock_db), patch("pathlib.Path.mkdir"):
            store.connect()

        # Assert
        mock_db.open_table.assert_called_once_with("chunks")
        mock_db.create_table.assert_not_called()

    def test_raises_vector_store_error_when_connect_fails(self):
        # Arrange
        config = build_vector_store_config()
        store = LanceDBVectorStore(config=config)

        # Act / Assert
        with (
            patch(f"{MODULE}.lancedb.connect", side_effect=OSError("disk error")),
            patch("pathlib.Path.mkdir"),
            pytest.raises(VectorStoreError, match="Failed to connect"),
        ):
            store.connect()

    def test_raises_when_not_connected(self):
        # Arrange - store created but connect() never called
        config = build_vector_store_config()
        store = LanceDBVectorStore(config=config)

        # Act / Assert
        with pytest.raises(VectorStoreError, match="not connected"):
            store.store_chunks([build_chunk()])


# ---------------------------------------------------------------------------
# TestStoreChunks
# ---------------------------------------------------------------------------


class TestStoreChunks:
    def test_empty_list_is_noop(self, connected_store, mock_table):
        # Arrange / Act
        result = connected_store.store_chunks([])

        # Assert
        assert result == []
        mock_table.add.assert_not_called()

    def test_chunk_without_embedding_raises_vector_store_error(self, connected_store):
        # Arrange
        chunk = build_chunk(chunk_id="chunk-no-emb", embedding=None)

        # Act / Assert
        with pytest.raises(VectorStoreError, match="chunk-no-emb"):
            connected_store.store_chunks([chunk])

    def test_valid_chunk_is_forwarded_to_table(self, connected_store, mock_table):
        # Arrange
        chunk = build_chunk(chunk_id="chunk-abc", source_id="src-xyz")

        # Act
        result = connected_store.store_chunks([chunk])

        # Assert
        assert result == ["chunk-abc"]
        mock_table.add.assert_called_once()
        added_batch = mock_table.add.call_args[0][0]
        assert len(added_batch) == 1
        assert added_batch[0]["chunk_id"] == "chunk-abc"
        assert added_batch[0]["source_id"] == "src-xyz"

    def test_multiple_chunks_return_all_ids(self, connected_store, mock_table):
        # Arrange
        chunks = [build_chunk(chunk_id=f"chunk-{i}", position=i) for i in range(3)]

        # Act
        result = connected_store.store_chunks(chunks)

        # Assert
        assert result == ["chunk-0", "chunk-1", "chunk-2"]

    def test_chunks_batched_according_to_config(self, mock_db, mock_table):
        # Arrange -- batch_size=2 with 3 chunks should produce two add() calls
        config = build_vector_store_config(batch_size=2)
        store = LanceDBVectorStore(config=config)
        mock_db.table_names.return_value = []
        mock_db.create_table.return_value = mock_table

        with patch(f"{MODULE}.lancedb.connect", return_value=mock_db), patch("pathlib.Path.mkdir"):
            store.connect()

        chunks = [build_chunk(chunk_id=f"chunk-{i}", position=i) for i in range(3)]

        # Act
        store.store_chunks(chunks)

        # Assert: first batch of 2, then flush of 1 = 2 total add() calls
        assert mock_table.add.call_count == 2

    def test_metadata_serialised_as_json(self, connected_store, mock_table):
        # Arrange
        chunk = build_chunk(metadata={"source_type": "pdf", "author": "Raj"})

        # Act
        connected_store.store_chunks([chunk])

        # Assert
        added_batch = mock_table.add.call_args[0][0]
        import json

        stored_meta = json.loads(added_batch[0]["metadata"])
        assert stored_meta["source_type"] == "pdf"
        assert stored_meta["author"] == "Raj"

    def test_source_type_extracted_from_metadata(self, connected_store, mock_table):
        # Arrange
        chunk = build_chunk(metadata={"source_type": "pdf"})

        # Act
        connected_store.store_chunks([chunk])

        # Assert
        added_batch = mock_table.add.call_args[0][0]
        assert added_batch[0]["source_type"] == "pdf"

    def test_source_type_defaults_to_unknown_when_metadata_absent(
        self, connected_store, mock_table
    ):
        # Arrange
        chunk = build_chunk(metadata=None)

        # Act
        connected_store.store_chunks([chunk])

        # Assert
        added_batch = mock_table.add.call_args[0][0]
        assert added_batch[0]["source_type"] == "unknown"

    def test_table_add_failure_raises_vector_store_error(self, connected_store, mock_table):
        # Arrange
        mock_table.add.side_effect = RuntimeError("write failed")
        chunk = build_chunk()

        # Act / Assert
        with pytest.raises(VectorStoreError, match="Failed to store"):
            connected_store.store_chunks([chunk])


# ---------------------------------------------------------------------------
# TestSearch
# ---------------------------------------------------------------------------


class TestSearch:
    def test_returns_empty_list_when_no_results(self, connected_store, mock_table):
        # Arrange
        mock_table.search.return_value.metric.return_value.limit.return_value.to_list.return_value = (
            []
        )
        query = [0.1] * EMBEDDING_DIM

        # Act
        results = connected_store.search(query_embedding=query, limit=5)

        # Assert
        assert results == []

    def test_results_above_threshold_are_returned(self, connected_store, mock_table):
        # Arrange -- similarity = 1.0 - 0.1 = 0.9, threshold is 0.6
        row = build_search_row(chunk_id="chunk-001", distance=0.1)
        mock_table.search.return_value.metric.return_value.limit.return_value.to_list.return_value = [
            row
        ]
        query = [0.1] * EMBEDDING_DIM

        # Act
        results = connected_store.search(query_embedding=query, limit=5)

        # Assert
        assert len(results) == 1
        assert results[0].chunk_id == "chunk-001"
        assert abs(results[0].similarity - 0.9) < 1e-9

    def test_results_below_threshold_are_filtered(self, connected_store, mock_table):
        # Arrange -- similarity = 1.0 - 0.8 = 0.2, below default threshold 0.6
        row = build_search_row(chunk_id="chunk-low", distance=0.8)
        mock_table.search.return_value.metric.return_value.limit.return_value.to_list.return_value = [
            row
        ]
        query = [0.1] * EMBEDDING_DIM

        # Act
        results = connected_store.search(query_embedding=query, limit=5)

        # Assert
        assert results == []

    def test_results_exactly_at_threshold_are_included(self, connected_store, mock_table):
        # Arrange -- similarity = 1.0 - 0.4 = 0.6, exactly at threshold 0.6
        row = build_search_row(chunk_id="chunk-exact", distance=0.4)
        mock_table.search.return_value.metric.return_value.limit.return_value.to_list.return_value = [
            row
        ]
        query = [0.1] * EMBEDDING_DIM

        # Act
        results = connected_store.search(query_embedding=query, limit=5)

        # Assert
        assert len(results) == 1
        assert results[0].chunk_id == "chunk-exact"

    def test_mixed_threshold_filters_only_low_results(self, connected_store, mock_table):
        # Arrange: one above (distance=0.1 -> similarity=0.9), one below (distance=0.8 -> similarity=0.2)
        rows = [
            build_search_row(chunk_id="chunk-high", distance=0.1),
            build_search_row(chunk_id="chunk-low", distance=0.8),
        ]
        mock_table.search.return_value.metric.return_value.limit.return_value.to_list.return_value = (
            rows
        )
        query = [0.1] * EMBEDDING_DIM

        # Act
        results = connected_store.search(query_embedding=query, limit=5)

        # Assert
        assert len(results) == 1
        assert results[0].chunk_id == "chunk-high"

    def test_search_with_valid_filter_calls_where(self, connected_store, mock_table):
        # Arrange
        query = [0.1] * EMBEDDING_DIM
        search_chain = mock_table.search.return_value.metric.return_value.limit.return_value
        search_chain.to_list.return_value = []

        # Act
        connected_store.search(
            query_embedding=query,
            limit=5,
            filters={"source_id": "src-001"},
        )

        # Assert: where() was called with a filter string containing source_id
        search_chain.where.assert_called_once()
        filter_str = search_chain.where.call_args[0][0]
        assert "source_id" in filter_str
        assert "src-001" in filter_str

    def test_metadata_json_deserialised_in_result(self, connected_store, mock_table):
        # Arrange
        import json

        row = build_search_row(
            chunk_id="chunk-meta",
            distance=0.1,
            metadata=json.dumps({"author": "Raj"}),
        )
        mock_table.search.return_value.metric.return_value.limit.return_value.to_list.return_value = [
            row
        ]
        query = [0.1] * EMBEDDING_DIM

        # Act
        results = connected_store.search(query_embedding=query, limit=5)

        # Assert
        assert results[0].metadata == {"author": "Raj"}

    def test_search_failure_raises_vector_store_error(self, connected_store, mock_table):
        # Arrange
        mock_table.search.side_effect = RuntimeError("index corrupt")
        query = [0.1] * EMBEDDING_DIM

        # Act / Assert
        with pytest.raises(VectorStoreError, match="Vector search failed"):
            connected_store.search(query_embedding=query, limit=5)


# ---------------------------------------------------------------------------
# TestFilterBuilder
# ---------------------------------------------------------------------------


class TestFilterBuilder:
    def test_valid_string_key_produces_quoted_clause(self):
        # Arrange / Act
        result = LanceDBVectorStore._build_filter_string({"source_id": "doc-abc"})

        # Assert
        assert result == "source_id = 'doc-abc'"

    def test_valid_int_key_produces_unquoted_clause(self):
        # Arrange / Act
        result = LanceDBVectorStore._build_filter_string({"position": 3})

        # Assert
        assert result == "position = 3"

    def test_multiple_valid_keys_joined_with_and(self):
        # Arrange / Act -- use a single key pair to keep assertion deterministic
        result = LanceDBVectorStore._build_filter_string({"source_type": "pdf", "position": 1})

        # Assert: both clauses present, joined by AND
        assert "source_type = 'pdf'" in result
        assert "position = 1" in result
        assert " AND " in result

    def test_unknown_key_is_skipped_and_no_exception_raised(self):
        # Arrange / Act -- injection_attempt is not in the allowlist
        result = LanceDBVectorStore._build_filter_string({"injection_attempt": "x OR 1=1"})

        # Assert
        assert result == ""

    def test_unknown_key_does_not_suppress_valid_keys(self):
        # Arrange / Act
        result = LanceDBVectorStore._build_filter_string(
            {"source_id": "src-001", "unknown_field": "evil"}
        )

        # Assert: valid clause present, unknown key absent
        assert "source_id = 'src-001'" in result
        assert "unknown_field" not in result

    def test_single_quotes_in_string_value_are_escaped(self):
        # Arrange / Act
        result = LanceDBVectorStore._build_filter_string({"source_id": "it's"})

        # Assert
        assert "it''s" in result

    def test_empty_filters_returns_empty_string(self):
        # Arrange / Act
        result = LanceDBVectorStore._build_filter_string({})

        # Assert
        assert result == ""

    def test_unsupported_value_type_is_skipped(self):
        # Arrange / Act -- list is not str or int
        result = LanceDBVectorStore._build_filter_string({"source_id": ["a", "b"]})

        # Assert
        assert result == ""

    @pytest.mark.parametrize(
        "key",
        [
            pytest.param("source_id", id="source_id"),
            pytest.param("source_type", id="source_type"),
            pytest.param("section_title", id="section_title"),
            pytest.param("chunk_id", id="chunk_id"),
        ],
    )
    def test_all_allowed_string_keys_produce_clause(self, key):
        # Arrange / Act
        result = LanceDBVectorStore._build_filter_string({key: "value"})

        # Assert
        assert key in result
        assert "'value'" in result

    @pytest.mark.parametrize(
        "key",
        [
            pytest.param("position", id="position"),
            pytest.param("word_count", id="word_count"),
        ],
    )
    def test_all_allowed_int_keys_produce_clause(self, key):
        # Arrange / Act
        result = LanceDBVectorStore._build_filter_string({key: 42})

        # Assert
        assert key in result
        assert "42" in result


# ---------------------------------------------------------------------------
# TestDeleteBySource
# ---------------------------------------------------------------------------


class TestDeleteBySource:
    def test_calls_table_delete_with_correct_filter(self, connected_store, mock_table):
        # Arrange
        mock_table.count_rows.return_value = 3

        # Act
        count = connected_store.delete_by_source("src-001")

        # Assert
        assert count == 3
        mock_table.delete.assert_called_once()
        filter_expr = mock_table.delete.call_args[0][0]
        assert "src-001" in filter_expr
        assert "source_id" in filter_expr

    def test_single_quotes_in_source_id_are_escaped(self, connected_store, mock_table):
        # Arrange
        mock_table.count_rows.return_value = 1

        # Act
        connected_store.delete_by_source("it's-tricky")

        # Assert
        filter_expr = mock_table.delete.call_args[0][0]
        assert "it''s-tricky" in filter_expr

    def test_delete_failure_raises_vector_store_error(self, connected_store, mock_table):
        # Arrange
        mock_table.count_rows.return_value = 1
        mock_table.delete.side_effect = RuntimeError("delete failed")

        # Act / Assert
        with pytest.raises(VectorStoreError, match="Failed to delete"):
            connected_store.delete_by_source("src-001")

    def test_returns_zero_when_no_matching_rows(self, connected_store, mock_table):
        # Arrange
        mock_table.count_rows.return_value = 0

        # Act
        count = connected_store.delete_by_source("nonexistent")

        # Assert
        assert count == 0


# ---------------------------------------------------------------------------
# TestHealthCheck
# ---------------------------------------------------------------------------


class TestHealthCheck:
    def test_returns_true_when_table_responds(self, connected_store, mock_table):
        # Arrange
        mock_table.count_rows.return_value = 5

        # Act
        result = connected_store.health_check()

        # Assert
        assert result is True

    def test_returns_false_when_table_is_none(self):
        # Arrange -- store never connected; _table is None
        config = build_vector_store_config()
        store = LanceDBVectorStore(config=config)

        # Act
        result = store.health_check()

        # Assert
        assert result is False

    def test_returns_false_when_count_rows_raises(self, connected_store, mock_table):
        # Arrange
        mock_table.count_rows.side_effect = RuntimeError("table unavailable")

        # Act
        result = connected_store.health_check()

        # Assert
        assert result is False


# ---------------------------------------------------------------------------
# TestCount
# ---------------------------------------------------------------------------


class TestCount:
    def test_returns_row_count_from_table(self, connected_store, mock_table):
        # Arrange
        mock_table.count_rows.return_value = 42

        # Act
        result = connected_store.count()

        # Assert
        assert result == 42

    def test_returns_zero_when_table_is_empty(self, connected_store, mock_table):
        # Arrange
        mock_table.count_rows.return_value = 0

        # Act
        result = connected_store.count()

        # Assert
        assert result == 0

    def test_count_failure_raises_vector_store_error(self, connected_store, mock_table):
        # Arrange
        mock_table.count_rows.side_effect = RuntimeError("count failed")

        # Act / Assert
        with pytest.raises(VectorStoreError, match="Failed to count"):
            connected_store.count()
