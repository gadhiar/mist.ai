"""Knowledge System Data Models.

Core data structures for working with the knowledge graph.

Includes both the original retrieval/session models and Phase 1B
extraction pipeline models (re-exported from their source modules).
"""

import enum
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


class RoutingDestination(str, enum.Enum):
    """Where a piece of content should be routed for storage."""

    VECTOR_ONLY = "vector_only"
    GRAPH_AND_VECTOR = "graph_and_vector"
    MCP_ONLY = "mcp_only"
    DISCARD = "discard"


class ContentSourceType(str, enum.Enum):
    """Origin type of incoming content."""

    CONVERSATION = "conversation"
    DOCUMENT_CHUNK = "document_chunk"
    MCP_TOOL_OUTPUT = "mcp_tool_output"
    REFERENCE_LOOKUP = "reference_lookup"
    SYSTEM = "system"


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """Result of knowledge routing classification.

    Determines whether content belongs in the graph, vector store,
    both, or should be discarded entirely.
    """

    destination: str  # RoutingDestination value
    reason: str
    confidence: float


@dataclass(frozen=True, slots=True)
class ToolClassification:
    """Result of tool output classification for MCP tool results.

    Provides granular per-provider routing decisions that supplement
    the generic MCP fast-path in KnowledgeRouter. The classifier
    parses tool names, applies provider-specific rules, and enforces
    anti-bloat gates (dedup, rate limiting, content length).

    Collapse of suppressed/suppression_reason into destination/reason:
    suppression is expressed as destination=DISCARD with reason describing
    the suppression cause (e.g. "rate_limit_exceeded", "duplicate_within_1h").
    """

    destination: str  # RoutingDestination value
    reason: str
    confidence: float
    tool_provider: str  # "linear", "github", "filesystem", "web", "unknown"
    tool_action: str  # parsed action from tool name
    source_level: str  # "orchestrator" or "agent"
    content_hash: str  # SHA-256 of tool output


@dataclass(frozen=True, slots=True)
class QueryIntent:
    """Result of query intent classification for hybrid retrieval.

    Determines which retrieval backends should handle a query based
    on pattern-matched intent signals.
    """

    intent: str  # "factual" | "relational" | "hybrid" | "live" | "identity"
    confidence: float  # 0.0-1.0
    suggested_stores: tuple[
        str, ...
    ]  # ("vector",), ("graph",), ("vector", "graph"), ("mcp",), ("mist",)


# Re-export Phase 1B extraction models for convenience.
# Canonical definitions live in their respective modules.
from backend.knowledge.extraction.ontology_extractor import ExtractionResult  # noqa: F401
from backend.knowledge.extraction.preprocessor import PreProcessedInput  # noqa: F401
from backend.knowledge.extraction.validator import ValidationResult  # noqa: F401


@dataclass
class Utterance:
    """Represents an immutable utterance from conversation.

    This is the source of truth - the knowledge graph can always
    be regenerated from utterances.
    """

    utterance_id: str
    conversation_id: str
    text: str
    timestamp: datetime
    metadata: dict | None = None

    def __str__(self):
        return f"Utterance({self.utterance_id[:8]}...): {self.text[:50]}..."


@dataclass
class RegenerationReport:
    """Statistics and results from graph regeneration.

    Tracks what happened during regeneration process.
    """

    total_utterances: int
    processed: int
    failed: int
    entities_created: int
    relationships_created: int
    duration_seconds: float
    errors: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_utterances == 0:
            return 0.0
        return (self.processed / self.total_utterances) * 100

    def __str__(self):
        return (
            f"RegenerationReport:\n"
            f"  Utterances: {self.processed}/{self.total_utterances} "
            f"({self.success_rate:.1f}% success)\n"
            f"  Entities: {self.entities_created}\n"
            f"  Relationships: {self.relationships_created}\n"
            f"  Duration: {self.duration_seconds:.2f}s\n"
            f"  Errors: {len(self.errors)}"
        )


@dataclass
class SearchResult:
    """Result from vector similarity search.

    Contains entity information and similarity score.
    """

    entity_id: str
    entity_type: str
    similarity: float
    properties: dict | None = None

    def __str__(self):
        return f"{self.entity_id} ({self.entity_type}) - {self.similarity:.3f}"


@dataclass
class RetrievalFilters:
    """Optional filters for knowledge retrieval.

    Designed for LLM to specify what it wants to retrieve.
    Default behavior: Return all relevant context (no filtering).
    """

    entity_types: list[str] | None = None
    relationship_types: list[str] | None = None
    exclude_entity_types: list[str] | None = None
    exclude_relationship_types: list[str] | None = None
    min_similarity: float | None = None
    max_age_days: int | None = None  # Filter by creation date


@dataclass
class RetrievedFact:
    """A single fact (relationship) retrieved from knowledge graph.

    Represents a subject-predicate-object triple with metadata.
    """

    # Core relationship
    subject: str  # Entity ID (e.g., "User")
    subject_type: str  # Entity type (e.g., "Person")
    predicate: str  # Relationship type (e.g., "USES")
    object: str  # Target entity ID (e.g., "Python")
    object_type: str  # Target entity type (e.g., "Technology")

    # Properties and metadata
    properties: dict[str, Any]  # Relationship properties (duration, proficiency, etc.)
    similarity_score: float  # Vector similarity to query (0-1)
    graph_distance: int  # Hops from search center (0 = direct vector match)

    # Provenance (optional)
    source_utterance_id: str | None = None
    created_at: datetime | None = None

    def to_triple_string(self) -> str:
        """Format as subject-predicate-object triple."""
        return f"{self.subject} -[{self.predicate}]-> {self.object}"

    def to_natural_language(self) -> str:
        """Format as natural language sentence.

        Example: "You use Python (expert proficiency, 5 years)"
        """
        props_str = ""
        if self.properties:
            prop_parts = []
            for k, v in self.properties.items():
                if k not in [
                    "created_at",
                    "ontology_version",
                    "created_from_utterance",
                    "embedding",
                ]:
                    prop_parts.append(f"{k}: {v}")
            if prop_parts:
                props_str = f" ({', '.join(prop_parts)})"

        return f"{self.subject} {self.predicate.lower().replace('_', ' ')} {self.object}{props_str}"

    def __str__(self):
        return self.to_triple_string()


RetrievalSource = Literal["vault", "graph", "vector"]

# The (subject, subject_type) pairs `KnowledgeRetriever` stamps on the facts it
# maps from vault sidecar rows and vector store chunks. Every other fact is a
# graph relationship. The subject is part of the mark because the graph also
# holds `DocumentChunk` nodes, so the type alone does not identify a vector fact.
_VAULT_FACT_MARK = ("VaultNote", "VaultSession")
_VECTOR_FACT_MARK = ("Document", "DocumentChunk")

# Hex digits of the SHA-256 kept in a ref, fixed by the turn-trace contract.
_FACT_REF_HEX_DIGITS = 20


def _fact_source(fact: RetrievedFact) -> RetrievalSource:
    mark = (fact.subject, fact.subject_type)
    if mark == _VAULT_FACT_MARK:
        return "vault"
    if mark == _VECTOR_FACT_MARK:
        return "vector"
    return "graph"


def fact_ref(fact: RetrievedFact) -> str:
    """Return an opaque, stable ref for a retrieved fact: `<source>:<20 hex digits>`.

    The turn trace records which memories a turn saw without writing their
    text, so the ref is a hash of the fact's natural key. mist-read-path maps a
    ref back by hashing the vault and graph it is granted with the same keys:

    - vault: the path, the heading and the SHA-256 hex of `properties["text"]`,
      joined by newlines. The heading is `fact.object`; a file-level chunk,
      whose sidecar heading is null, is keyed with the heading `(file)`, the
      object the retriever gives it. The text digest is part of the key because
      the sidecar writes one chunk per heading block, so two sections of one
      note can share a path and a heading.
    - graph: `<subject>|<predicate>|<object>`.
    - vector: `properties["chunk_id"]`.

    Args:
        fact: A fact produced by `KnowledgeRetriever`, or one shaped like it.

    Returns:
        The ref, matching `^(vault|graph|vector):[0-9a-f]{20}$`.
    """
    source = _fact_source(fact)
    properties = fact.properties or {}
    if source == "vault":
        text = str(properties.get("text") or "")
        text_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        key = f"{properties.get('path') or ''}\n{fact.object}\n{text_digest}"
    elif source == "vector":
        key = str(properties.get("chunk_id") or "")
    else:
        key = f"{fact.subject}|{fact.predicate}|{fact.object}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return f"{source}:{digest[:_FACT_REF_HEX_DIGITS]}"


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    """One row a retrieval leg returned, before the merge and the limit.

    The vault and vector legs report every row they returned, before any
    filtering, so an empty vault chunk the retriever drops still appears. The
    graph leg reports its facts after `RetrievalFilters` and triple dedup: a
    row the caller's filters exclude has no disposition in the turn trace, and
    a duplicate triple has the same ref as the fact it repeats.

    `rank` is the 1-based position within its leg, `score` the leg's own score
    for the row, `selected` whether the fact reached `RetrievalResult.facts`,
    and `empty` whether the row was a vault chunk with no body, which the
    retriever drops before formatting.
    """

    source: RetrievalSource
    ref: str
    rank: int
    score: float | None
    selected: bool
    empty: bool


@dataclass
class RetrievalResult:
    """Complete result from knowledge retrieval.

    Contains all retrieved facts and formatted context for LLM.
    """

    query: str  # Original query
    user_id: str  # User who made the query

    # Retrieved data
    facts: list[RetrievedFact]  # All retrieved facts
    entities_found: int  # Number of entities matched by vector search
    total_facts: int  # Total number of facts retrieved

    # Formatted output
    formatted_context: str  # Natural language context for LLM

    # Metadata
    retrieval_time_ms: float  # How long retrieval took
    vector_search_time_ms: float  # Time for vector search
    graph_traversal_time_ms: float  # Time for graph queries

    # Configuration used
    config_used: dict[str, Any]  # What parameters were used

    # Hybrid retrieval metadata (all defaulted for backwards compat)
    intent: str | None = None  # Classified query intent
    requires_mcp: bool = False  # True when live data needed
    suggested_tools: tuple[str, ...] = ()  # MCP tools for live intent
    document_chunks_used: int = 0  # Number of vector store chunks in results

    # Each leg's rows in leg order (graph, vector, vault) and rank: every vault
    # and vector row, and the graph facts after filters and dedup (see
    # `RetrievalCandidate`).
    candidates: tuple[RetrievalCandidate, ...] = ()

    def get_top_facts(self, n: int = 10) -> list[RetrievedFact]:
        """Get top N facts by score."""
        return self.facts[:n]

    def get_facts_by_entity(self, entity_id: str) -> list[RetrievedFact]:
        """Get all facts involving a specific entity."""
        return [f for f in self.facts if f.subject == entity_id or f.object == entity_id]

    def summary(self) -> str:
        """Brief summary of retrieval.

        The vault-only path (intent == "historical") has no graph entities by
        design, so it reports chunk count rather than "0 entities" to avoid
        reading like a failed retrieval.
        """
        if self.intent == "historical":
            return f"Retrieved {self.total_facts} vault chunks in {self.retrieval_time_ms:.1f}ms"
        return f"Retrieved {self.total_facts} facts from {self.entities_found} entities in {self.retrieval_time_ms:.1f}ms"

    def __str__(self):
        return self.summary()


@dataclass
class Message:
    """A single message in a conversation.

    Represents either user input or assistant response.
    """

    role: str  # "user" or "assistant"
    content: str  # Message text
    timestamp: datetime = field(default_factory=datetime.now)

    # Tool usage tracking (for assistant messages)
    tool_calls: list[dict[str, Any]] | None = None
    tool_results: list[dict[str, Any]] | None = None

    def __str__(self):
        return f"[{self.role}] {self.content[:50]}..."


@dataclass
class ConversationSession:
    """Tracks a conversation session with context.

    Models human memory: One event per session with timestamps.
    Links to ConversationEvent in Neo4j.
    """

    session_id: str  # Unique session identifier
    user_id: str  # User participating in conversation
    started_at: datetime = field(default_factory=datetime.now)

    # Conversation history
    messages: list[Message] = field(default_factory=list)

    # Neo4j tracking
    conversation_event_id: str | None = None  # Links to ConversationEvent node

    def add_message(self, role: str, content: str, tool_calls=None, tool_results=None):
        """Add a message to the conversation."""
        msg = Message(role=role, content=content, tool_calls=tool_calls, tool_results=tool_results)
        self.messages.append(msg)
        return msg

    def get_history(self, max_messages: int | None = None) -> list[dict[str, str]]:
        """Get conversation history formatted for LLM.

        Returns list of {"role": "user/assistant", "content": "..."}
        """
        messages = self.messages
        if max_messages:
            messages = messages[-max_messages:]

        return [{"role": msg.role, "content": msg.content} for msg in messages]

    def __str__(self):
        return f"Session {self.session_id}: {len(self.messages)} messages"


@dataclass
class SourceDocument:
    """Immutable source document for RAG corpus.

    Represents external knowledge sources:
    - Markdown files
    - PDFs
    - MCP-fetched web pages
    - Uploaded documents

    These are the provenance layer for DocumentChunks.
    """

    source_id: str  # Unique identifier (UUID)
    file_path: str  # Path or URI to source
    source_type: str  # "markdown", "pdf", "web", "upload"
    content_hash: str  # SHA256 hash of content for deduplication
    ingested_at: datetime = field(default_factory=datetime.now)

    # Metadata
    title: str | None = None
    author: str | None = None
    file_size: int | None = None  # bytes
    metadata: dict[str, Any] | None = None  # Flexible metadata storage

    def __str__(self):
        return f"SourceDocument({self.source_type}): {self.title or self.file_path}"


@dataclass
class DocumentChunk:
    """A chunk of a source document for RAG retrieval.

    Serves dual purpose:
    1. RAG Corpus: Vector search retrieves verbatim text
    2. Entity Provenance: Entities link back to chunks

    Immutable - chunks don't change unless source is re-ingested.
    """

    chunk_id: str  # Unique identifier (UUID)
    source_id: str  # Links to SourceDocument
    text: str  # The actual chunk text
    position: int  # Position in document (0-indexed)

    # Vector search
    embedding: list[float] | None = None  # 384-dim vector for semantic search

    # Metadata
    word_count: int = 0
    char_count: int = 0
    section_title: str | None = None  # Section/header this chunk belongs to
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        """Calculate word and char counts if not provided."""
        if self.word_count == 0:
            self.word_count = len(self.text.split())
        if self.char_count == 0:
            self.char_count = len(self.text)

    def __str__(self):
        return f"Chunk {self.chunk_id[:8]}... ({self.word_count} words)"


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    """Result from vector similarity search in the document store.

    Returned by VectorStoreProvider.search(). Immutable value object.
    """

    chunk_id: str
    text: str
    similarity: float
    source_id: str
    source_type: str
    metadata: dict
