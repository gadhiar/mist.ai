"""I/O boundary contracts for the MIST.AI backend.

These Protocol classes define the interfaces for external dependencies
(graph database, embeddings, LLM, event store) so that concrete
implementations can be swapped for test doubles via dependency injection.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from backend.knowledge.models import DocumentChunk, VectorSearchResult
    from backend.llm.models import LLMRequest, LLMResponse


class GraphConnection(Protocol):
    """Contract for Neo4j graph database access."""

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def is_connected(self) -> bool: ...
    def execute_query(self, query: str, params: dict | None = None) -> list[dict]: ...
    def execute_write(self, query: str, params: dict | None = None) -> list[dict]: ...


class EmbeddingProvider(Protocol):
    """Contract for text embedding generation."""

    def generate_embedding(self, text: str) -> list[float]: ...
    def generate_embeddings(self, texts: list[str]) -> list[list[float]]: ...


class LLMProvider(Protocol):
    """Contract for large language model inference.

    Implemented by StreamingLLMProvider (production) and FakeLLM (tests).
    """

    async def invoke(self, request: LLMRequest) -> LLMResponse: ...


class EventStoreProvider(Protocol):
    """Contract for conversation event storage."""

    def initialize(self) -> None: ...
    def start_session(self, input_modality: str = "voice") -> str: ...
    def end_session(self, session_id: str) -> None: ...
    def append_turn(self, event: Any) -> str: ...
    def get_turns(self, session_id: str) -> list: ...
    def get_turns_since(self, since: datetime) -> list[dict]: ...


class VectorStoreProvider(Protocol):
    """Contract for vector similarity search and chunk storage."""

    def store_chunks(self, chunks: list[DocumentChunk]) -> list[str]: ...
    def search(
        self,
        query_embedding: list[float],
        limit: int,
        filters: dict | None = None,
    ) -> list[VectorSearchResult]: ...
    def delete_by_source(self, source_id: str) -> int: ...
    def get_chunk(self, chunk_id: str) -> DocumentChunk | None: ...
    def count(self) -> int: ...
    def health_check(self) -> bool: ...


class VaultWriterProtocol(Protocol):
    """Contract for serialized vault writes (ADR-010 Cluster 8 / Layer 2).

    All writes are atomic and serialized through a single VaultWriter
    instance owned by the backend process. `start()` boots the consumer
    coroutine; `stop()` drains and shuts it down. Caller-facing methods
    enqueue a write request and await completion.

    Implemented by `backend.vault.writer.VaultWriter` (production).
    """

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def append_turn_to_session(
        self,
        session_id: str,
        turn_index: int,
        user_text: str,
        mist_text: str,
        vault_note_path: str | None = None,
    ) -> str: ...
    async def update_entities_extracted(
        self,
        vault_note_path: str,
        turn_index: int,
        entity_slugs: list[str],
    ) -> None: ...
    async def upsert_identity(
        self,
        traits: list[dict],
        capabilities: list[dict],
        preferences: list[dict],
        rendered_at: str | None = None,
    ) -> str: ...
    async def upsert_user(
        self, user_id: str, body_markdown: str, rendered_at: str | None = None
    ) -> str: ...
    async def upsert_user_snapshot(
        self, user_id: str, body_markdown: str, rendered_at: str | None = None
    ) -> str: ...
    def session_path(self, session_date: str, session_slug: str) -> str: ...


class SidecarIndexProtocol(Protocol):
    """Contract for the vault sidecar index (ADR-010 Cluster 8 / Layer 4).

    SQLite database with sqlite-vec `vec0` + FTS5 virtual tables over
    vault note chunks. Indexes file-level (full body) and heading-block
    (per `##` heading) chunks for two-tier retrieval. Read paths return
    rows with `path`, `heading`, `content`, `score`, `frontmatter_json`.

    Implemented by `backend.vault.sidecar_index.VaultSidecarIndex`.
    """

    def initialize(self) -> None: ...
    def close(self) -> None: ...
    def upsert_file(
        self,
        path: str,
        content: str,
        mtime: int,
        frontmatter: dict | None = None,
    ) -> int: ...
    def delete_path(self, path: str) -> int: ...
    def query_vector(self, embedding: list[float], k: int = 10) -> list[dict]: ...
    def query_fts(self, text: str, k: int = 10) -> list[dict]: ...
    def query_hybrid(
        self,
        embedding: list[float],
        text: str,
        k: int = 10,
        rrf_k: int = 60,
    ) -> list[dict]: ...
    def chunk_count(self) -> int: ...
    def distinct_paths(self) -> list[str]: ...
    def health_check(self) -> bool: ...


class GraphStoreProtocol(Protocol):
    """Contract for graph store operations used by GraphRegenerator.

    Extends the base graph storage surface with methods required for
    ADR-010 invariant 5 closure: orphan-marking DERIVED_FROM-scoped
    triples and re-deriving them deterministically or via LLM extraction.

    Implemented by `backend.knowledge.storage.graph_store.GraphStore`
    (production). Test doubles implement this protocol directly without
    inheriting from it.
    """

    async def mark_orphaned_by_provenance_path(self, path: str) -> int:
        """Mark all triples with DERIVED_FROM.path == path as status='orphaned'.

        Preserves the triples per ADR-010 (no hard-delete). Returns the
        count of triples marked.
        """
        ...

    def current_ontology_version(self) -> str:
        """Return the current ontology version string (e.g. '1.1.0').

        Used by GraphRegenerator to stamp re-derived triples with the
        version active at rebuild time.
        """
        ...

    async def get_orphaned_provenance_paths(self) -> list[str]:
        """Return the list of distinct DERIVED_FROM.path values for orphaned triples.

        Used by GraphRegenerator.retry_orphaned to enumerate provenance paths
        whose async re-extraction previously failed so they can be retried.

        Returns:
            List of absolute path strings for which orphaned triples exist.
        """
        ...


class ExtractionPipelineProtocol(Protocol):
    """Contract for the extraction pipeline surface used by GraphRegenerator.

    Covers the vault-file re-extraction path (Bucket 2/3). The full
    ExtractionPipeline satisfies this protocol via duck-typing; only the
    method required by GraphRegenerator is listed here.
    """

    async def extract_from_file(
        self,
        content: str,
        vault_note_path: str,
        ontology_version: str,
    ) -> Any:
        """Extract entities and relationships from vault file content.

        Args:
            content: Full text of the vault file.
            vault_note_path: Absolute path string for provenance tracking.
            ontology_version: Ontology version string to stamp on extracted
                triples.

        Returns:
            Implementation-defined extraction result.
        """
        ...
