"""Document ingestion pipeline.

Orchestrates chunking, embedding, and storage of documents into the
vector store. Provides session-scoped content-hash deduplication and
batch embedding for throughput.

For cross-session re-ingestion of a previously ingested document, the
caller should first call `vector_store.delete_by_source(source_id)` to
remove stale chunks before re-ingesting.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from backend.errors import IngestionError
from backend.interfaces import EmbeddingProvider, VectorStoreProvider
from backend.knowledge.config import IngestionConfig
from backend.knowledge.ingestion.chunker import FixedSizeChunker, MarkdownChunker

if TYPE_CHECKING:
    from backend.knowledge.storage.graph_store import GraphStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Outcome of a single document ingestion."""

    source_id: str
    chunks_created: int
    chunks_skipped: int
    deduplicated: bool
    duration_ms: float


# Map file extensions to source types.
_EXTENSION_MAP: dict[str, str] = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".text": "text",
    ".rst": "text",
    ".csv": "text",
    ".json": "text",
    ".yaml": "text",
    ".yml": "text",
}

# Regex for extracting the first markdown heading from content.
_FIRST_HEADING_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)


class IngestionPipeline:
    """Chunking, embedding, and storage pipeline for documents.

    Accepts raw document content or file paths, splits them into chunks
    using a strategy appropriate for the source type, generates
    embeddings in batches, and stores the results in the vector store.

    Session-scoped dedup prevents the same content from being ingested
    twice within one pipeline instance lifetime.

    Args:
        vector_store: Vector store for persisting chunks.
        embedding_provider: Embedding generator for batch vectorization.
        config: Ingestion configuration (chunk size, overlap, batch size).
    """

    def __init__(
        self,
        vector_store: VectorStoreProvider,
        embedding_provider: EmbeddingProvider,
        config: IngestionConfig,
        graph_store: GraphStore | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider
        self._config = config
        self._graph_store = graph_store
        self._ingested_hashes: dict[str, str] = {}  # content_hash -> source_id

    def ingest_document(
        self,
        content: str,
        file_path: str,
        source_type: str = "markdown",
        title: str | None = None,
        metadata: dict | None = None,
    ) -> IngestionResult:
        """Ingest a document from raw content.

        Flow:
            1. Dedup check via SHA-256 content hash
            2. Select chunking strategy based on source_type
            3. Chunk content
            4. Enrich chunk metadata
            5. Batch embed chunks
            6. Store in vector store

        For re-ingestion of a previously stored document, call
        `vector_store.delete_by_source(source_id)` before calling this
        method to remove stale data.

        Args:
            content: Raw document text.
            file_path: Path or URI to the source document (for provenance).
            source_type: Content type hint ("markdown", "text", etc.).
            title: Optional document title. Extracted from content if None.
            metadata: Optional additional metadata merged into chunk metadata.

        Returns:
            IngestionResult with chunk counts and timing.

        Raises:
            IngestionError: If chunking, embedding, or storage fails.
        """
        start = time.perf_counter()

        # 1. Dedup check
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if content_hash in self._ingested_hashes:
            duration_ms = (time.perf_counter() - start) * 1000
            existing_id = self._ingested_hashes[content_hash]
            logger.info(
                "Duplicate content detected (hash=%s), skipping ingestion "
                "(existing source_id=%s)",
                content_hash[:12],
                existing_id,
            )
            return IngestionResult(
                source_id=existing_id,
                chunks_created=0,
                chunks_skipped=0,
                deduplicated=True,
                duration_ms=duration_ms,
            )

        # 2. Generate source_id
        source_id = str(uuid.uuid4())

        # 3. Select chunker
        if source_type == "markdown":
            chunker = MarkdownChunker(
                chunk_size=self._config.chunk_size,
                chunk_overlap=self._config.chunk_overlap,
            )
        else:
            chunker = FixedSizeChunker(
                chunk_size=self._config.chunk_size,
                chunk_overlap=self._config.chunk_overlap,
            )

        # 4. Chunk content
        try:
            chunks = chunker.chunk(content, source_id, source_type)
        except Exception as exc:
            raise IngestionError(f"Chunking failed for '{file_path}': {exc}") from exc

        if not chunks:
            duration_ms = (time.perf_counter() - start) * 1000
            self._ingested_hashes[content_hash] = source_id
            return IngestionResult(
                source_id=source_id,
                chunks_created=0,
                chunks_skipped=0,
                deduplicated=False,
                duration_ms=duration_ms,
            )

        # 5. Enrich metadata
        extra_meta = {
            "content_hash": content_hash,
            "file_path": file_path,
        }
        if title:
            extra_meta["title"] = title
        if metadata:
            extra_meta.update(metadata)

        for chunk in chunks:
            if chunk.metadata is not None:
                chunk.metadata.update(extra_meta)
            else:
                # DocumentChunk.__post_init__ sets word/char counts but
                # metadata may be None; replace with enriched dict.
                object.__setattr__(chunk, "metadata", dict(extra_meta))

        # 6. Batch embed
        try:
            texts = [c.text for c in chunks]
            batch_size = self._config.batch_size

            for batch_start in range(0, len(texts), batch_size):
                batch_end = batch_start + batch_size
                batch_texts = texts[batch_start:batch_end]

                batch_t = time.perf_counter()
                embeddings = self._embedding_provider.generate_embeddings(batch_texts)
                batch_ms = (time.perf_counter() - batch_t) * 1000
                logger.debug(
                    "Embedded batch %d-%d (%d texts) in %.1fms",
                    batch_start,
                    batch_end,
                    len(batch_texts),
                    batch_ms,
                )

                for i, emb in enumerate(embeddings):
                    chunks[batch_start + i].embedding = emb
        except Exception as exc:
            raise IngestionError(f"Embedding generation failed for '{file_path}': {exc}") from exc

        # 7. Store
        try:
            self._vector_store.store_chunks(chunks)
        except Exception as exc:
            raise IngestionError(f"Vector store write failed for '{file_path}': {exc}") from exc

        # 8. Register hash
        self._ingested_hashes[content_hash] = source_id

        # 9. Create graph provenance nodes (ExternalSource + VectorChunk refs)
        if self._graph_store is not None:
            try:
                self._graph_store.store_external_source(
                    source_uri=file_path,
                    source_type=source_type,
                    title=title or "",
                )
                for chunk in chunks:
                    self._graph_store.store_vector_chunk_ref(
                        vector_store_id=chunk.chunk_id,
                        source_id=file_path,
                    )
                logger.info(
                    "Created graph provenance for %d chunks from %s",
                    len(chunks),
                    file_path,
                )
            except Exception as e:
                logger.warning("Graph provenance creation failed (non-fatal): %s", e)

        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "Ingested '%s': %d chunks in %.1fms (source_id=%s)",
            title or file_path,
            len(chunks),
            duration_ms,
            source_id,
        )

        return IngestionResult(
            source_id=source_id,
            chunks_created=len(chunks),
            chunks_skipped=0,
            deduplicated=False,
            duration_ms=duration_ms,
        )

    def ingest_file(
        self,
        file_path: str,
        source_type: str | None = None,
        **kwargs,
    ) -> IngestionResult:
        """Ingest a document from a file path.

        Reads the file as UTF-8, auto-detects source_type from the file
        extension if not provided, and extracts a title from the first
        markdown heading (for markdown files) or uses the filename.

        Args:
            file_path: Path to the file to ingest.
            source_type: Override for auto-detected source type.
            **kwargs: Additional arguments forwarded to `ingest_document`.

        Returns:
            IngestionResult with chunk counts and timing.

        Raises:
            IngestionError: If the file cannot be read or ingestion fails.
        """
        path = Path(file_path)

        # Read file
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as exc:
            raise IngestionError(f"Failed to read file '{file_path}': {exc}") from exc

        # Auto-detect source type
        if source_type is None:
            source_type = _EXTENSION_MAP.get(path.suffix.lower(), "text")

        # Extract title
        title = kwargs.pop("title", None)
        if title is None:
            if source_type == "markdown":
                match = _FIRST_HEADING_RE.search(content)
                if match:
                    title = match.group(1).strip()
            if title is None:
                title = path.stem

        return self.ingest_document(
            content=content,
            file_path=str(path),
            source_type=source_type,
            title=title,
            **kwargs,
        )
