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
    ) -> str: ...
    async def upsert_user(self, user_id: str, body_markdown: str) -> str: ...
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
    def health_check(self) -> bool: ...
