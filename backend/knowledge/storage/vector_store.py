"""LanceDB-backed vector store for document chunk storage and retrieval.

Implements VectorStoreProvider protocol using LanceDB as the storage backend.
Handles chunk persistence, cosine similarity search, and filtered queries.

Table schema uses PyArrow for explicit column typing. Embeddings are stored
as fixed-size float32 lists (384 dimensions, matching all-MiniLM-L6-v2).
"""

import json
import logging
from pathlib import Path

import lancedb
import pyarrow as pa

from backend.errors import VectorStoreError
from backend.knowledge.config import VectorStoreConfig
from backend.knowledge.models import DocumentChunk, VectorSearchResult

logger = logging.getLogger(__name__)

_TABLE_SCHEMA = pa.schema(
    [
        pa.field("chunk_id", pa.string(), nullable=False),
        pa.field("text", pa.large_string(), nullable=False),
        pa.field("vector", pa.list_(pa.float32(), 384), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("source_type", pa.string(), nullable=False),
        pa.field("position", pa.int32(), nullable=False),
        pa.field("section_title", pa.string(), nullable=True),
        pa.field("word_count", pa.int32(), nullable=False),
        pa.field("metadata", pa.string(), nullable=True),
    ]
)

# Keys allowed in filter queries. Reject anything outside this set to prevent
# injection of arbitrary LanceDB filter expressions.
_ALLOWED_FILTER_KEYS = frozenset(
    {
        "source_id",
        "source_type",
        "section_title",
        "position",
        "word_count",
        "chunk_id",
    }
)


class LanceDBVectorStore:
    """LanceDB implementation of VectorStoreProvider.

    Lifecycle: create with config, call connect() to open the database,
    use store/search/delete methods, then close() when done. The factory
    function in `backend/factories.py` handles connect() automatically.
    """

    def __init__(self, config: VectorStoreConfig) -> None:
        """Store configuration only. No connection is made here.

        Args:
            config: Vector store configuration.
        """
        self._config = config
        self._db: lancedb.DBConnection | None = None
        self._table: lancedb.table.Table | None = None

    def connect(self) -> None:
        """Open LanceDB database and create or open the chunk table.

        Creates the data directory if it does not exist. If the table
        already exists it is opened; otherwise a new empty table is
        created with the predefined schema.

        Raises:
            VectorStoreError: If the database cannot be opened.
        """
        try:
            data_path = Path(self._config.data_dir)
            data_path.mkdir(parents=True, exist_ok=True)

            self._db = lancedb.connect(str(data_path))

            existing_tables = self._db.table_names()
            if self._config.collection_name in existing_tables:
                self._table = self._db.open_table(self._config.collection_name)
                logger.info(
                    "Opened existing LanceDB table '%s' (%d rows)",
                    self._config.collection_name,
                    self._table.count_rows(),
                )
            else:
                self._table = self._db.create_table(
                    self._config.collection_name,
                    schema=_TABLE_SCHEMA,
                )
                logger.info(
                    "Created new LanceDB table '%s'",
                    self._config.collection_name,
                )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to connect to LanceDB at {self._config.data_dir}: {exc}"
            ) from exc

    def close(self) -> None:
        """Release database and table references."""
        self._table = None
        self._db = None
        logger.info("LanceDB vector store closed")

    def _ensure_table(self) -> lancedb.table.Table:
        """Return the active table or raise if not connected.

        Raises:
            VectorStoreError: If connect() has not been called.
        """
        if self._table is None:
            raise VectorStoreError("Vector store not connected. Call connect() first.")
        return self._table

    # -- Public API (VectorStoreProvider) --------------------------------------

    def store_chunks(self, chunks: list[DocumentChunk]) -> list[str]:
        """Persist document chunks with their embeddings.

        Chunks are appended in batches of `config.batch_size`. Deduplication
        is not performed here -- that responsibility belongs to the ingestion
        pipeline (K-03).

        Args:
            chunks: Document chunks with embeddings already computed.

        Returns:
            List of stored chunk IDs.

        Raises:
            VectorStoreError: If any chunk lacks an embedding or the write fails.
        """
        table = self._ensure_table()

        stored_ids: list[str] = []
        batch: list[dict] = []

        for chunk in chunks:
            if chunk.embedding is None:
                raise VectorStoreError(
                    f"Chunk {chunk.chunk_id} has no embedding. "
                    "Embeddings must be computed before storage."
                )

            source_type = (
                chunk.metadata.get("source_type", "unknown") if chunk.metadata else "unknown"
            )

            record = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "vector": chunk.embedding,
                "source_id": chunk.source_id,
                "source_type": source_type,
                "position": chunk.position,
                "section_title": chunk.section_title,
                "word_count": chunk.word_count,
                "metadata": json.dumps(chunk.metadata) if chunk.metadata else None,
            }
            batch.append(record)
            stored_ids.append(chunk.chunk_id)

            if len(batch) >= self._config.batch_size:
                try:
                    table.add(batch)
                except Exception as exc:
                    raise VectorStoreError(
                        f"Failed to store batch of {len(batch)} chunks: {exc}"
                    ) from exc
                batch = []

        # Flush remaining
        if batch:
            try:
                table.add(batch)
            except Exception as exc:
                raise VectorStoreError(
                    f"Failed to store final batch of {len(batch)} chunks: {exc}"
                ) from exc

        logger.info("Stored %d chunks in vector store", len(stored_ids))
        return stored_ids

    def search(
        self,
        query_embedding: list[float],
        limit: int,
        filters: dict | None = None,
    ) -> list[VectorSearchResult]:
        """Find chunks most similar to the query embedding.

        Uses cosine distance. LanceDB returns ``_distance`` in [0, 2] for
        cosine metric where 0 means identical vectors. Similarity is computed
        as ``1.0 - distance``, so the result range is [-1, 1] with 1 being
        most similar.

        Results below ``config.similarity_threshold`` are discarded.

        Args:
            query_embedding: Query vector (384 dimensions).
            limit: Maximum number of results to return.
            filters: Optional column filters (allowlisted keys only).

        Returns:
            Sorted list of VectorSearchResult (highest similarity first).

        Raises:
            VectorStoreError: If the search fails.
        """
        table = self._ensure_table()

        try:
            query = table.search(query_embedding).metric("cosine").limit(limit)

            if filters:
                filter_str = self._build_filter_string(filters)
                if filter_str:
                    query = query.where(filter_str)

            raw_results = query.to_list()
        except Exception as exc:
            raise VectorStoreError(f"Vector search failed: {exc}") from exc

        results: list[VectorSearchResult] = []
        for row in raw_results:
            # Cosine distance -> similarity conversion.
            # LanceDB cosine metric: _distance in [0, 2], 0 = identical.
            similarity = 1.0 - row["_distance"]

            if similarity < self._config.similarity_threshold:
                continue

            metadata = {}
            if row.get("metadata"):
                try:
                    metadata = json.loads(row["metadata"])
                except (json.JSONDecodeError, TypeError):
                    metadata = {}

            results.append(
                VectorSearchResult(
                    chunk_id=row["chunk_id"],
                    text=row["text"],
                    similarity=similarity,
                    source_id=row["source_id"],
                    source_type=row["source_type"],
                    metadata=metadata,
                )
            )

        return results

    def delete_by_source(self, source_id: str) -> int:
        """Delete all chunks belonging to a source document.

        Args:
            source_id: The source document identifier.

        Returns:
            Approximate number of rows deleted (counted before deletion).

        Raises:
            VectorStoreError: If the deletion fails.
        """
        table = self._ensure_table()

        escaped = source_id.replace("'", "''")
        filter_expr = f"source_id = '{escaped}'"

        try:
            count_before = table.count_rows(filter=filter_expr)
            table.delete(filter_expr)
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to delete chunks for source {source_id}: {exc}"
            ) from exc

        logger.info("Deleted ~%d chunks for source_id='%s'", count_before, source_id)
        return count_before

    def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        """Retrieve a single chunk by ID.

        Args:
            chunk_id: The chunk identifier.

        Returns:
            DocumentChunk if found, None otherwise.

        Raises:
            VectorStoreError: If the query fails.
        """
        table = self._ensure_table()

        escaped = chunk_id.replace("'", "''")
        filter_expr = f"chunk_id = '{escaped}'"

        try:
            rows = table.search().where(filter_expr).limit(1).to_list()
        except Exception as exc:
            raise VectorStoreError(f"Failed to get chunk {chunk_id}: {exc}") from exc

        if not rows:
            return None

        row = rows[0]
        metadata = None
        if row.get("metadata"):
            try:
                metadata = json.loads(row["metadata"])
            except (json.JSONDecodeError, TypeError):
                metadata = None

        return DocumentChunk(
            chunk_id=row["chunk_id"],
            source_id=row["source_id"],
            text=row["text"],
            position=row["position"],
            embedding=row.get("vector"),
            word_count=row["word_count"],
            section_title=row.get("section_title"),
            metadata=metadata,
        )

    def count(self) -> int:
        """Return the total number of chunks in the store.

        Raises:
            VectorStoreError: If the count query fails.
        """
        table = self._ensure_table()

        try:
            return table.count_rows()
        except Exception as exc:
            raise VectorStoreError(f"Failed to count rows: {exc}") from exc

    def health_check(self) -> bool:
        """Check whether the vector store is operational.

        Returns:
            True if the store responds to a count query, False otherwise.
        """
        try:
            table = self._ensure_table()
            table.count_rows()
            return True
        except Exception:
            return False

    # -- Internal helpers ------------------------------------------------------

    @staticmethod
    def _build_filter_string(filters: dict) -> str:
        """Build a LanceDB SQL filter string from a dict of column filters.

        Only keys in the allowlist are accepted. Values must be str or int.
        Unknown keys are logged as warnings and skipped. Single quotes in
        string values are escaped by doubling.

        Args:
            filters: Mapping of column name to desired value.

        Returns:
            AND-joined SQL filter expression, or empty string if no valid
            filters remain.
        """
        clauses: list[str] = []

        for key, value in filters.items():
            if key not in _ALLOWED_FILTER_KEYS:
                logger.warning(
                    "Ignoring unknown filter key '%s'. " "Allowed keys: %s",
                    key,
                    sorted(_ALLOWED_FILTER_KEYS),
                )
                continue

            if isinstance(value, str):
                escaped = value.replace("'", "''")
                clauses.append(f"{key} = '{escaped}'")
            elif isinstance(value, int):
                clauses.append(f"{key} = {value}")
            else:
                logger.warning(
                    "Ignoring filter key '%s': value must be str or int, " "got %s",
                    key,
                    type(value).__name__,
                )

        return " AND ".join(clauses)
