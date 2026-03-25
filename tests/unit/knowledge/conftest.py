"""Shared test fixtures for the knowledge system.

Provides fake implementations of VectorStoreProvider and EmbeddingProvider,
factory functions for ExtractionResult and graph data, sample document
content, and pytest fixtures for common test objects.

K-16 / MIS-56
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

import pytest

from backend.knowledge.extraction.ontology_extractor import ExtractionResult
from backend.knowledge.models import DocumentChunk, SourceDocument, VectorSearchResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 384

_ENTITY_TYPES = [
    "User",
    "Person",
    "Organization",
    "Technology",
    "Skill",
    "Project",
    "Concept",
    "Topic",
    "Event",
    "Goal",
    "Preference",
    "Location",
]

_RELATIONSHIP_TYPES = [
    "USES",
    "KNOWS",
    "WORKS_ON",
    "WORKS_AT",
    "INTERESTED_IN",
    "HAS_GOAL",
    "PREFERS",
    "EXPERT_IN",
    "LEARNING",
    "RELATED_TO",
    "DEPENDS_ON",
    "IS_A",
    "PART_OF",
    "MEMBER_OF",
    "KNOWS_PERSON",
]

_ENTITY_NAMES: dict[str, list[str]] = {
    "User": ["raj"],
    "Person": ["Alice", "Bob", "Carol", "Dave"],
    "Organization": ["Anthropic", "Google", "Mozilla"],
    "Technology": ["Python", "FastAPI", "Neo4j", "Flutter", "Ollama", "PyTorch"],
    "Skill": ["system-design", "graph-databases", "NLP"],
    "Project": ["MIST.AI", "knowledge-vault", "hana"],
    "Concept": ["event-sourcing", "dependency-injection", "ontology"],
    "Topic": ["machine-learning", "knowledge-graphs"],
    "Event": ["sprint-review", "deploy-v2"],
    "Goal": ["ship-knowledge-system", "learn-rust"],
    "Preference": ["dark-mode", "vim-keybindings"],
    "Location": ["Toronto", "San Francisco"],
}


# ---------------------------------------------------------------------------
# FakeVectorStore
# ---------------------------------------------------------------------------


class FakeVectorStore:
    """In-memory VectorStoreProvider for unit tests.

    Stores chunks in a dict keyed by chunk_id. Records every method call
    in ``_calls`` so tests can assert on interaction patterns.
    """

    def __init__(self) -> None:
        self._chunks: dict[str, dict[str, Any]] = {}
        self._calls: list[tuple[str, Any]] = []

    # -- VectorStoreProvider protocol methods --------------------------------

    def store_chunks(self, chunks: list[DocumentChunk]) -> list[str]:
        """Store chunks and return their IDs."""
        self._calls.append(("store_chunks", chunks))
        ids: list[str] = []
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = {
                "chunk_id": chunk.chunk_id,
                "source_id": chunk.source_id,
                "text": chunk.text,
                "position": chunk.position,
                "embedding": chunk.embedding,
                "word_count": chunk.word_count,
                "char_count": chunk.char_count,
                "section_title": chunk.section_title,
                "metadata": chunk.metadata or {},
            }
            ids.append(chunk.chunk_id)
        return ids

    def search(
        self,
        query_embedding: list[float],
        limit: int,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Return chunks sorted by position-based fake similarity.

        If *filters* is provided, only chunks whose metadata contains
        matching key-value pairs are included.
        """
        self._calls.append(("search", (query_embedding, limit, filters)))
        candidates = list(self._chunks.values())

        if filters:
            candidates = [
                c
                for c in candidates
                if all(c.get("metadata", {}).get(k) == v for k, v in filters.items())
            ]

        # Deterministic scoring: lower position -> higher similarity.
        candidates.sort(key=lambda c: c["position"])
        results: list[VectorSearchResult] = []
        for i, c in enumerate(candidates[:limit]):
            score = round(1.0 - (i * 0.05), 4)
            results.append(
                VectorSearchResult(
                    chunk_id=c["chunk_id"],
                    text=c["text"],
                    similarity=score,
                    source_id=c["source_id"],
                    source_type=c.get("metadata", {}).get("source_type", "markdown"),
                    metadata=c.get("metadata", {}),
                )
            )
        return results

    def delete_by_source(self, source_id: str) -> int:
        """Delete all chunks belonging to *source_id*. Return count removed."""
        self._calls.append(("delete_by_source", source_id))
        to_remove = [cid for cid, data in self._chunks.items() if data["source_id"] == source_id]
        for cid in to_remove:
            del self._chunks[cid]
        return len(to_remove)

    def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        """Retrieve a single chunk by ID, or None."""
        self._calls.append(("get_chunk", chunk_id))
        data = self._chunks.get(chunk_id)
        if data is None:
            return None
        return DocumentChunk(
            chunk_id=data["chunk_id"],
            source_id=data["source_id"],
            text=data["text"],
            position=data["position"],
            embedding=data["embedding"],
            section_title=data["section_title"],
            metadata=data["metadata"],
        )

    def count(self) -> int:
        """Return the number of stored chunks."""
        self._calls.append(("count", None))
        return len(self._chunks)

    def health_check(self) -> bool:
        """Always healthy in tests."""
        self._calls.append(("health_check", None))
        return True

    # -- Assertion helpers ---------------------------------------------------

    def assert_called(self, method: str) -> None:
        """Assert that *method* was called at least once."""
        if not any(name == method for name, _ in self._calls):
            called = [name for name, _ in self._calls]
            raise AssertionError(
                f"Expected call to {method!r}, but only these were called: {called}"
            )

    def assert_not_called(self, method: str) -> None:
        """Assert that *method* was never called."""
        if any(name == method for name, _ in self._calls):
            raise AssertionError(f"Expected no call to {method!r}, but it was called")

    def call_count(self, method: str) -> int:
        """Return the number of times *method* was called."""
        return sum(1 for name, _ in self._calls if name == method)


# ---------------------------------------------------------------------------
# FakeEmbeddingProvider
# ---------------------------------------------------------------------------


class FakeEmbeddingProvider:
    """Deterministic embedding provider for tests.

    Produces 384-dimensional vectors derived from the SHA-256 hash of the
    input text. Same input always yields the same output.
    """

    def __init__(self, *, dimension: int = EMBEDDING_DIM) -> None:
        self._dimension = dimension
        self.calls: list[str] = []

    def generate_embedding(self, text: str) -> list[float]:
        """Return a deterministic vector based on text hash."""
        self.calls.append(text)
        digest = hashlib.sha256(text.encode()).digest()
        vec = [float(b) / 255.0 for b in digest]
        # Extend to required dimension by repeating the hash bytes.
        while len(vec) < self._dimension:
            vec.extend([float(b) / 255.0 for b in digest])
        return vec[: self._dimension]

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        return [self.generate_embedding(t) for t in texts]


# ---------------------------------------------------------------------------
# ExtractionResult factory
# ---------------------------------------------------------------------------


def make_extraction_result(
    *,
    entity_count: int = 2,
    relationship_count: int = 1,
    confidence: float = 0.8,
    entity_types: list[str] | None = None,
) -> ExtractionResult:
    """Build a valid ExtractionResult with configurable parameters.

    Generates realistic entity and relationship dicts using ontology types
    (User, Technology, Skill, etc.). Entities are given sequential names
    drawn from ``_ENTITY_NAMES`` where available.

    Parameters
    ----------
    entity_count:
        Number of entities to generate.
    relationship_count:
        Number of relationships to generate. Relationships reference
        generated entities by their ``id`` field. Clamped so there are
        enough entities to form distinct pairs.
    confidence:
        Confidence score assigned to every entity and relationship.
    entity_types:
        Explicit list of entity types to cycle through. Defaults to a
        selection from the ontology.
    """
    types = entity_types or ["Technology", "Skill", "Person", "Project"]
    entities: list[dict[str, Any]] = []
    for i in range(entity_count):
        etype = types[i % len(types)]
        names = _ENTITY_NAMES.get(etype, [etype.lower()])
        name = names[i % len(names)]
        entities.append(
            {
                "id": name.lower().replace(" ", "-"),
                "type": etype,
                "name": name,
                "confidence": confidence,
                "source_type": "extracted",
                "aliases": [],
                "description": f"A {etype.lower()} entity for testing.",
            }
        )

    relationships: list[dict[str, Any]] = []
    max_rels = max(0, entity_count - 1)
    for i in range(min(relationship_count, max_rels)):
        src = entities[i]
        tgt = entities[(i + 1) % entity_count]
        rel_type = _RELATIONSHIP_TYPES[i % len(_RELATIONSHIP_TYPES)]
        relationships.append(
            {
                "source": src["id"],
                "target": tgt["id"],
                "type": rel_type,
                "confidence": confidence,
                "source_type": "extracted",
                "temporal_status": "current",
                "context": f"{src['name']} {rel_type.lower()} {tgt['name']}",
            }
        )

    return ExtractionResult(
        entities=entities,
        relationships=relationships,
        raw_llm_output="{}",
        extraction_time_ms=12.5,
        source_utterance="I use Python and FastAPI for building APIs.",
    )


# ---------------------------------------------------------------------------
# Sample document content
# ---------------------------------------------------------------------------

SAMPLE_MARKDOWN = """# Architecture Overview
This document describes the system architecture for MIST.AI.

## Backend
The backend uses Python with FastAPI for the WebSocket server.
Ollama runs the Qwen 2.5 7B model for inference.

## Frontend
The frontend uses Flutter for desktop applications.
Riverpod handles state management.

## Knowledge System
Neo4j stores the knowledge graph.
Sentence Transformers produce 384-dimensional embeddings.
"""

SAMPLE_PLAIN_TEXT = (
    "Python was created by Guido van Rossum and first released in 1991. "
    "It emphasizes code readability and supports multiple programming "
    "paradigms including procedural, object-oriented, and functional "
    "programming. Python is widely used in web development, data science, "
    "artificial intelligence, and scientific computing."
)

SAMPLE_WITH_CODE = """# Setup Guide
Install dependencies:
```bash
# This is a comment, not a heading
pip install -r requirements.txt
```
## Configuration
Set the environment variables in `.env`:
```python
NEO4J_URI = "bolt://localhost:7687"
OLLAMA_HOST = "http://localhost:11434"
```
## Running
Start the backend server with `python backend/server.py`.
"""


# ---------------------------------------------------------------------------
# Graph state factories
# ---------------------------------------------------------------------------


def make_graph_entities(count: int = 10) -> list[dict[str, Any]]:
    """Return a list of entity dicts with realistic types and properties.

    Cycles through ontology entity types and draws display names from
    ``_ENTITY_NAMES``.
    """
    entities: list[dict[str, Any]] = []
    flat_names: list[tuple[str, str]] = []
    for etype, names in _ENTITY_NAMES.items():
        for name in names:
            flat_names.append((etype, name))

    for i in range(count):
        etype, name = flat_names[i % len(flat_names)]
        entities.append(
            {
                "id": name.lower().replace(" ", "-"),
                "type": etype,
                "name": name,
                "confidence": round(0.7 + (i % 4) * 0.05, 2),
                "source_type": "extracted",
                "aliases": [],
                "created_at": "2026-01-15T10:00:00",
            }
        )
    return entities


def make_graph_relationships(
    entities: list[dict[str, Any]],
    count: int = 15,
) -> list[dict[str, Any]]:
    """Return relationship dicts connecting the provided entities.

    Distributes relationships evenly across entity pairs, cycling through
    ontology relationship types.
    """
    if len(entities) < 2:
        return []

    relationships: list[dict[str, Any]] = []
    for i in range(count):
        src = entities[i % len(entities)]
        tgt = entities[(i + 1) % len(entities)]
        if src["id"] == tgt["id"]:
            tgt = entities[(i + 2) % len(entities)]
        rel_type = _RELATIONSHIP_TYPES[i % len(_RELATIONSHIP_TYPES)]
        relationships.append(
            {
                "source": src["id"],
                "target": tgt["id"],
                "type": rel_type,
                "confidence": round(0.75 + (i % 3) * 0.05, 2),
                "source_type": "extracted",
                "temporal_status": "current",
                "context": f"{src['name']} {rel_type.lower()} {tgt['name']}",
            }
        )
    return relationships


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_vector_store() -> FakeVectorStore:
    """A fresh FakeVectorStore instance."""
    return FakeVectorStore()


@pytest.fixture
def fake_embedding_provider() -> FakeEmbeddingProvider:
    """A fresh FakeEmbeddingProvider instance."""
    return FakeEmbeddingProvider()


@pytest.fixture
def sample_source_document() -> SourceDocument:
    """A SourceDocument representing a markdown file."""
    return SourceDocument(
        source_id="src-test-0001",
        file_path="/docs/architecture.md",
        source_type="markdown",
        content_hash=hashlib.sha256(SAMPLE_MARKDOWN.encode()).hexdigest(),
        title="Architecture Overview",
        file_size=len(SAMPLE_MARKDOWN.encode()),
        metadata={"project": "mist.ai"},
    )


@pytest.fixture
def sample_chunks(fake_embedding_provider: FakeEmbeddingProvider) -> list[DocumentChunk]:
    """Five DocumentChunk instances with deterministic embeddings.

    Chunks are derived from ``SAMPLE_MARKDOWN`` sections. Each chunk has
    a pre-computed embedding from ``FakeEmbeddingProvider``.
    """
    source_id = "src-test-0001"
    texts = [
        "This document describes the system architecture for MIST.AI.",
        "The backend uses Python with FastAPI for the WebSocket server.",
        "Ollama runs the Qwen 2.5 7B model for inference.",
        "The frontend uses Flutter for desktop applications.",
        "Neo4j stores the knowledge graph.",
    ]
    sections = [
        "Architecture Overview",
        "Backend",
        "Backend",
        "Frontend",
        "Knowledge System",
    ]
    chunks: list[DocumentChunk] = []
    for i, (text, section) in enumerate(zip(texts, sections)):
        chunk_id = f"chunk-{uuid.UUID(int=i)}"
        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                source_id=source_id,
                text=text,
                position=i,
                embedding=fake_embedding_provider.generate_embedding(text),
                section_title=section,
                metadata={"source_type": "markdown"},
            )
        )
    return chunks
