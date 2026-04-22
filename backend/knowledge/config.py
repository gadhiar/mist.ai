"""Knowledge System Configuration.

Central configuration for Neo4j connection, embedding models, and extraction settings.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Neo4jConfig:
    """Neo4j database configuration."""

    uri: str = "bolt://localhost:7687"  # Default Neo4j Desktop URI
    username: str = "neo4j"
    password: str = os.getenv("NEO4J_PASSWORD")
    database: str = "neo4j"  # Default database name

    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        """Load configuration from environment variables."""
        return cls(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD"),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
        )


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""

    model_name: str = "all-MiniLM-L6-v2"  # Fast, local model
    dimension: int = 384  # MiniLM embedding dimension
    device: str = "cpu"  # or "cuda" if GPU available

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        """Load configuration from environment variables."""
        return cls(
            model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            dimension=int(os.getenv("EMBEDDING_DIMENSION", "384")),
            device=os.getenv("EMBEDDING_DEVICE", "cpu"),
        )


@dataclass
class LLMConfig:
    """LLM model configuration.

    Temperature is split into two scopes:
    - `temperature` (default 0.0) is used by the extraction pipeline for
      JSON reliability. Extraction call sites read this field.
    - `conversation_temperature` (default 0.7) is used by ConversationHandler
      for natural assistant voice. Conversation call sites read this field.

    Keeping the original `temperature` field as extraction-default avoids
    a rename that would churn through 10+ extraction call sites.
    """

    model: str = "qwen2.5:7b-instruct"
    base_url: str = "http://localhost:8080"  # llama-server default
    temperature: float = 0.0  # Deterministic for extraction
    conversation_temperature: float = 0.7  # Natural voice for conversation
    # Cluster 6: max_tokens for conversation calls. Defaulted to 1024 (up from
    # the historical 400) after the 2026-04-21 V6 diagnostic showed 15/30 V6
    # turns were "GHOST" — completion_tokens=400 with empty content and no
    # tool_calls, indicating truncated tool-call JSON cut off at max_tokens.
    # 1024 gives tool-call emission room to complete.
    conversation_max_tokens: int = 1024
    backend: str = "llamacpp"  # "llamacpp" or "ollama"

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load configuration from environment variables."""
        backend = os.getenv("LLM_BACKEND", "llamacpp")
        default_url = "http://localhost:11434" if backend == "ollama" else "http://localhost:8080"
        return cls(
            model=os.getenv("MODEL", "qwen2.5:7b-instruct"),
            base_url=os.getenv("LLM_SERVER_URL", default_url),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
            conversation_temperature=float(os.getenv("LLM_CONVERSATION_TEMPERATURE", "0.7")),
            conversation_max_tokens=int(os.getenv("LLM_CONVERSATION_MAX_TOKENS", "1024")),
            backend=backend,
        )


@dataclass
class ExtractionConfig:
    """Entity extraction configuration."""

    # Extraction settings
    extract_node_properties: bool = True  # Extract node descriptions
    extract_relationship_properties: bool = True  # Extract edge descriptions

    # Ontology constraints (None = dynamic extraction)
    allowed_nodes: list[str] | None = None
    allowed_relationships: list[str] | None = None

    # LLM instructions for extraction
    additional_instructions: str = """
    Extract entities and relationships from this conversational statement.

    CRITICAL RULES:
    1. ALWAYS create a "User" entity for first-person pronouns (I, me, my, mine)
    2. The User is the SUBJECT performing actions - relationships flow FROM User TO other entities
    3. Relationship direction MUST be correct: subject -[VERB]-> object
       - "Sarah taught me React" -> Sarah -[TAUGHT]-> User (subject is Sarah), User -[LEARNED]-> React
       - "I use Python" -> User -[USES]-> Python (NOT Python uses anything)
       - "I work on Project" -> User -[WORKS_ON]-> Project

    Entity Guidelines:
    - Extract: tools, technologies, projects, topics, skills, people, organizations
    - Do NOT extract: dates, durations, numbers, quantities as entities (use as relationship properties)
    - Avoid creating "X of Y" entities - extract the main entity only (not "packaging system of Python")
    - Use consistent entity types: Tool, Technology, Project, Topic, Skill, Person, Organization

    Relationship Guidelines:
    - Core relationships: USES, PREFERS, WORKS_ON, KNOWS_ABOUT, INTERESTED_IN, DISLIKES, STRUGGLES_WITH
    - Teaching: TAUGHT, LEARNED, TEACHES, LEARNS_FROM
    - Skill level: EXPERT_IN, PROFICIENT_IN, BEGINNER_IN, LEARNING
    - Temporal: USED_TO_USE, PLANS_TO_LEARN (capture time in properties)
    - Negation: DOES_NOT_USE, AVOIDS, HAS_NOT_WORKED_WITH
    - Multi-entity: "I use X with Y" -> User -[USES]-> X, User -[USES]-> Y, X -[WORKS_WITH]-> Y

    Property Capture:
    - Temporal info: "for 3 months" -> {{duration: "3 months"}}
    - Time: "next month" -> {{when: "next month"}}
    - Proficiency: "expert", "beginner" -> {{proficiency: "expert"}}
    - Context: "in production", "daily" -> {{context: "in production"}}

    Common Patterns:
    - "I have trouble with X" -> User -[STRUGGLES_WITH]-> X
    - "I'm good at X" -> User -[PROFICIENT_IN]-> X
    - "X for Y" when both are tools -> User -[USES]-> X, X -[USED_FOR]-> Y
    - "X with Y" -> create relationships to BOTH, plus X -[WORKS_WITH]-> Y if applicable
    - "prefer X over Y" -> User -[PREFERS]-> X (do NOT create relationship with Y)
    """

    # Confidence thresholds
    min_extraction_confidence: float = 0.5  # Don't store extractions below this

    # Significance gate -- utterances scoring below this are skipped
    significance_threshold: float = 0.3

    # Rate limiting -- max extraction calls per minute
    rate_limit_max_per_minute: int = 30

    # Input deduplication -- skip near-duplicate utterances
    dedup_similarity_threshold: float = 0.95
    dedup_cache_size: int = 200
    dedup_cache_ttl_seconds: int = 300

    @classmethod
    def from_env(cls) -> "ExtractionConfig":
        """Load configuration from environment variables."""
        return cls(
            extract_node_properties=os.getenv("EXTRACT_NODE_PROPERTIES", "true").lower() == "true",
            extract_relationship_properties=os.getenv("EXTRACT_REL_PROPERTIES", "true").lower()
            == "true",
            min_extraction_confidence=float(os.getenv("MIN_EXTRACTION_CONFIDENCE", "0.5")),
            significance_threshold=float(os.getenv("SIGNIFICANCE_THRESHOLD", "0.3")),
            rate_limit_max_per_minute=int(os.getenv("RATE_LIMIT_MAX_PER_MINUTE", "30")),
            dedup_similarity_threshold=float(os.getenv("DEDUP_SIMILARITY_THRESHOLD", "0.95")),
            dedup_cache_size=int(os.getenv("DEDUP_CACHE_SIZE", "200")),
            dedup_cache_ttl_seconds=int(os.getenv("DEDUP_CACHE_TTL_SECONDS", "300")),
        )


@dataclass
class VectorStoreConfig:
    """Vector store configuration."""

    backend: str = "lancedb"
    data_dir: str = "data/vector_store"
    similarity_threshold: float = 0.6
    batch_size: int = 100
    collection_name: str = "mist_knowledge"

    @classmethod
    def from_env(cls) -> "VectorStoreConfig":
        """Load configuration from environment variables."""
        return cls(
            backend=os.getenv("VECTOR_STORE_BACKEND", "lancedb"),
            data_dir=os.getenv("VECTOR_STORE_DATA_DIR", "data/vector_store"),
            similarity_threshold=float(os.getenv("VECTOR_STORE_SIMILARITY_THRESHOLD", "0.6")),
            batch_size=int(os.getenv("VECTOR_STORE_BATCH_SIZE", "100")),
            collection_name=os.getenv("VECTOR_STORE_COLLECTION", "mist_knowledge"),
        )


@dataclass
class EventStoreConfig:
    """Event store (Layer 1) configuration."""

    enabled: bool = True  # Master switch for event store recording
    db_path: str | None = None  # Defaults to ~/.mist/event_store.db
    audio_dir: str | None = None  # Defaults to ~/.mist/audio/

    @classmethod
    def from_env(cls) -> "EventStoreConfig":
        """Load configuration from environment variables."""
        return cls(
            enabled=os.getenv("EVENT_STORE_ENABLED", "true").lower() == "true",
            db_path=os.getenv("EVENT_STORE_DB_PATH"),
            audio_dir=os.getenv("EVENT_STORE_AUDIO_DIR"),
        )


@dataclass
class IngestionConfig:
    """Document ingestion pipeline configuration."""

    chunk_size: int = 512
    chunk_overlap: int = 64
    batch_size: int = 100
    dedup_window_days: int = 30

    @classmethod
    def from_env(cls) -> "IngestionConfig":
        """Load configuration from environment variables."""
        return cls(
            chunk_size=int(os.getenv("INGESTION_CHUNK_SIZE", "512")),
            chunk_overlap=int(os.getenv("INGESTION_CHUNK_OVERLAP", "64")),
            batch_size=int(os.getenv("INGESTION_BATCH_SIZE", "100")),
            dedup_window_days=int(os.getenv("INGESTION_DEDUP_WINDOW_DAYS", "30")),
        )


@dataclass
class QueryIntentConfig:
    """Configuration for query intent classification.

    Includes RRF (Reciprocal Rank Fusion) parameters used by
    KnowledgeRetriever when merging graph and vector results
    during hybrid retrieval.
    """

    hybrid_threshold: float = 0.4
    confidence_base: float = 0.5
    confidence_per_match: float = 0.1
    min_confidence: float = 0.3

    # RRF merge parameters
    rrf_k: int = 60
    rrf_vector_weight: float = 0.5
    rrf_graph_weight: float = 0.5
    max_vector_results: int = 20
    max_graph_results: int = 20

    # ADR-010 Cluster 8 Phase 9: vault sidecar as a third RRF retriever.
    # Default weights mirror the ADR table:
    #   historical 0.2/0.7/0.1, structural 0.7/0.2/0.1,
    #   identity 0.5/0.1/0.4, hybrid 0.4/0.4/0.2.
    # We expose hybrid + historical defaults here. structural maps to
    # the existing `relational` intent (graph-only path, no merge).
    rrf_vault_weight: float = 0.4  # hybrid-intent default
    max_vault_results: int = 20
    rrf_historical_graph_weight: float = 0.2
    rrf_historical_vector_weight: float = 0.1
    rrf_historical_vault_weight: float = 0.7

    @classmethod
    def from_env(cls) -> "QueryIntentConfig":
        """Load configuration from environment variables."""
        return cls(
            hybrid_threshold=float(os.getenv("QUERY_HYBRID_THRESHOLD", "0.4")),
            confidence_base=float(os.getenv("QUERY_CONFIDENCE_BASE", "0.5")),
            confidence_per_match=float(os.getenv("QUERY_CONFIDENCE_PER_MATCH", "0.1")),
            min_confidence=float(os.getenv("QUERY_MIN_CONFIDENCE", "0.3")),
            rrf_k=int(os.getenv("RRF_K", "60")),
            rrf_vector_weight=float(os.getenv("RRF_VECTOR_WEIGHT", "0.5")),
            rrf_graph_weight=float(os.getenv("RRF_GRAPH_WEIGHT", "0.5")),
            max_vector_results=int(os.getenv("MAX_VECTOR_RESULTS", "20")),
            max_graph_results=int(os.getenv("MAX_GRAPH_RESULTS", "20")),
            rrf_vault_weight=float(os.getenv("RRF_VAULT_WEIGHT", "0.4")),
            max_vault_results=int(os.getenv("MAX_VAULT_RESULTS", "20")),
            rrf_historical_graph_weight=float(os.getenv("RRF_HISTORICAL_GRAPH_WEIGHT", "0.2")),
            rrf_historical_vector_weight=float(os.getenv("RRF_HISTORICAL_VECTOR_WEIGHT", "0.1")),
            rrf_historical_vault_weight=float(os.getenv("RRF_HISTORICAL_VAULT_WEIGHT", "0.7")),
        )


@dataclass
class ContextBudgetConfig:
    """Cluster 6: budget-aware context assembly for ConversationHandler._build_messages.

    The model's usable attention window is usually smaller than its configured
    context window. Gemma 4 E4B at 32K ctx_size still degrades quality past
    its trained context (~8K). This config caps the total prompt budget and
    splits the remainder (after persona + tools + static) between retrieval
    and history via weight-based allocation.

    - `context_window` is the hard ceiling; output `max_tokens` is subtracted.
    - `output_reserve_tokens` additionally reserves headroom for the completion.
    - `safety_margin_tokens` is extra headroom for tokenizer inaccuracy.
    - `retrieval_budget_ratio` allocates share of the remaining budget to
      retrieval-context; history gets the rest.
    - `history_strategy` names a strategy registered in
      `backend.chat.context_budget` (defaults to sliding-window).
    """

    context_window: int = 8192  # Effective usable window for Gemma 4 E4B
    output_reserve_tokens: int = 512  # Headroom for completion output
    safety_margin_tokens: int = 256  # Absorbs tokenizer estimation error
    retrieval_budget_ratio: float = 0.4  # 40% of flex budget to retrieval
    history_strategy: str = "sliding_window"  # sliding_window | (future) summarized
    enabled: bool = True  # Master switch; False keeps legacy behavior

    @classmethod
    def from_env(cls) -> "ContextBudgetConfig":
        """Load context-budget configuration from environment variables."""
        return cls(
            context_window=int(os.getenv("MIST_CTX_BUDGET_WINDOW", "8192")),
            output_reserve_tokens=int(os.getenv("MIST_CTX_BUDGET_OUTPUT_RESERVE", "512")),
            safety_margin_tokens=int(os.getenv("MIST_CTX_BUDGET_SAFETY", "256")),
            retrieval_budget_ratio=float(os.getenv("MIST_CTX_BUDGET_RETRIEVAL_RATIO", "0.4")),
            history_strategy=os.getenv("MIST_CTX_BUDGET_HISTORY_STRATEGY", "sliding_window"),
            enabled=os.getenv("MIST_CTX_BUDGET_ENABLED", "true").lower() == "true",
        )


@dataclass(frozen=True, slots=True)
class ScopeClassifierConfig:
    """Configuration for Stage 1.5 subject-scope classifier (Cluster 1).

    The classifier is a small LLM call that tags each utterance as
    user-scope, system-scope, or third-party so the extraction prompt
    can weight its output correctly. This addresses Bug I/J extraction
    drift where MIST-self-reference utterances were mislabeled as
    user-scope facts.

    - `enabled` master switch. When False the pipeline skips Stage 1.5
      and the extractor treats subject_scope as "unknown".
    - `temperature` kept at 0.0 for deterministic classification.
    - `max_tokens` capped at 96 -- the expected JSON payload is ~40 tokens.
    - `timeout_seconds` soft upper bound before the call is considered
      failed and scope is returned as "unknown".
    """

    enabled: bool = True
    temperature: float = 0.0
    max_tokens: int = 96
    timeout_seconds: float = 5.0

    @classmethod
    def from_env(cls) -> "ScopeClassifierConfig":
        """Load scope-classifier configuration from environment variables."""
        return cls(
            enabled=os.getenv("MIST_SCOPE_CLASSIFIER_ENABLED", "true").lower() == "true",
            temperature=float(os.getenv("MIST_SCOPE_CLASSIFIER_TEMPERATURE", "0.0")),
            max_tokens=int(os.getenv("MIST_SCOPE_CLASSIFIER_MAX_TOKENS", "96")),
            timeout_seconds=float(os.getenv("MIST_SCOPE_CLASSIFIER_TIMEOUT_SECONDS", "5.0")),
        )


@dataclass(frozen=True, slots=True)
class VaultConfig:
    """Cluster 8 / ADR-010: vault layer (canonical user-approved history).

    The vault is a markdown corpus rendered to disk at `root`. MIST appends
    session-note turns and the user can edit prose directly with Obsidian or
    any text editor. The sidecar index (`SidecarIndexConfig`) and filewatcher
    (`FilewatcherConfig`) are paired subsystems that complete the layer.

    Defaults match ADR-010 implementation defaults section: container-relative
    `mist-memory/` root with host-side bind mount controlled by docker-compose
    `VAULT_HOST_PATH`. Append sentinel is the canonical literal documented in
    the ADR. Session soft caps drive auto-split behavior.

    The default user id (`raj`) is used to bootstrap `users/<id>.md` from
    `scripts/seed_data.yaml` on first run; MIST currently only models a single
    user, so this is not yet user-selectable per request.
    """

    enabled: bool = True
    root: str = "mist-memory"
    default_user_id: str = "raj"
    git_auto_init: bool = True
    session_soft_cap_turns: int = 20
    session_soft_cap_tokens: int = 6000
    append_sentinel: str = "<!-- MIST_APPEND_HERE -->"
    writer_queue_max_depth: int = 100

    @classmethod
    def from_env(cls) -> "VaultConfig":
        """Load vault configuration from environment variables."""
        return cls(
            enabled=os.getenv("MIST_VAULT_ENABLED", "true").lower() == "true",
            root=os.getenv("MIST_VAULT_ROOT", "mist-memory"),
            default_user_id=os.getenv("MIST_VAULT_DEFAULT_USER_ID", "raj"),
            git_auto_init=os.getenv("MIST_VAULT_GIT_AUTO_INIT", "true").lower() == "true",
            session_soft_cap_turns=int(os.getenv("MIST_VAULT_SESSION_SOFT_CAP_TURNS", "20")),
            session_soft_cap_tokens=int(os.getenv("MIST_VAULT_SESSION_SOFT_CAP_TOKENS", "6000")),
            append_sentinel=os.getenv("MIST_VAULT_APPEND_SENTINEL", "<!-- MIST_APPEND_HERE -->"),
            writer_queue_max_depth=int(os.getenv("MIST_VAULT_WRITER_QUEUE_MAX_DEPTH", "100")),
        )


@dataclass(frozen=True, slots=True)
class SidecarIndexConfig:
    """Cluster 8 / ADR-010: SQLite-backed sidecar index for the vault.

    Holds two-tier chunks (file-level and heading-block-level) with paired
    FTS5 + vec0 virtual tables. The index is rebuilt on vault file changes
    via the filewatcher and is the retrieval substrate for the `historical`
    query intent.

    `embedding_dimension` MUST match `EmbeddingConfig.dimension` -- the vec0
    table is created with a fixed schema and a mismatch produces silent
    retrieval failure. Defaulted here to keep the layer self-contained;
    cross-validated at builder wiring time.

    `heading_context_weight` is the BM25 weight for the heading-context FTS
    column relative to content (1.0). 0.3 is the Crosley pattern default
    (frontmatter-as-heading-context, hybrid-retriever-obsidian post).
    """

    enabled: bool = True
    db_path: str = "data/vault_sidecar.db"
    embedding_dimension: int = 384
    heading_context_weight: float = 0.3
    chunk_max_chars: int = 6000
    rebuild_on_startup: bool = False

    @classmethod
    def from_env(cls) -> "SidecarIndexConfig":
        """Load sidecar-index configuration from environment variables."""
        return cls(
            enabled=os.getenv("MIST_SIDECAR_ENABLED", "true").lower() == "true",
            db_path=os.getenv("MIST_SIDECAR_DB_PATH", "data/vault_sidecar.db"),
            embedding_dimension=int(os.getenv("MIST_SIDECAR_EMBEDDING_DIMENSION", "384")),
            heading_context_weight=float(os.getenv("MIST_SIDECAR_HEADING_CONTEXT_WEIGHT", "0.3")),
            chunk_max_chars=int(os.getenv("MIST_SIDECAR_CHUNK_MAX_CHARS", "6000")),
            rebuild_on_startup=os.getenv("MIST_SIDECAR_REBUILD_ON_STARTUP", "false").lower()
            == "true",
        )


@dataclass(frozen=True, slots=True)
class FilewatcherConfig:
    """Cluster 8 / ADR-010: watchdog-based filewatcher for vault edits.

    Runs in a dedicated daemon thread inside the backend process. Bridges
    to async via `loop.call_soon_threadsafe`. On vault file changes, schedules
    a debounced sidecar reindex; on vault-note edits made outside MIST, marks
    the corresponding `mist-session` / `mist-user` note as `authored_by:
    user-edit` and queues affected graph subgraph rebuild.

    `observer_type` selects the watchdog observer backend:
      - `auto`: PollingObserver on Windows/WSL2 bind mounts; Inotify on
        native Linux; FSEvents on macOS. Resolution lives in
        `backend/vault/filewatcher.py`.
      - `polling` / `inotify` / `fsevents`: explicit override for deployments
        where auto-detection is wrong.

    `debounce_ms` collapses bursts (e.g. Obsidian's atomic-replace-on-save
    delete+create within ~100ms). 500ms is the ADR-010 default.

    `staleness_slo_seconds` is the budget for sidecar lag behind vault content
    under normal load. The mtime audit job runs every `audit_interval_seconds`
    to catch dropped events (Windows ReadDirectoryChangesW overflow case).
    """

    enabled: bool = True
    observer_type: str = "auto"
    debounce_ms: int = 500
    staleness_slo_seconds: int = 5
    audit_interval_seconds: int = 60

    @classmethod
    def from_env(cls) -> "FilewatcherConfig":
        """Load filewatcher configuration from environment variables."""
        return cls(
            enabled=os.getenv("MIST_FILEWATCHER_ENABLED", "true").lower() == "true",
            observer_type=os.getenv("MIST_FILEWATCHER_OBSERVER_TYPE", "auto"),
            debounce_ms=int(os.getenv("MIST_FILEWATCHER_DEBOUNCE_MS", "500")),
            staleness_slo_seconds=int(os.getenv("MIST_FILEWATCHER_STALENESS_SLO_SECONDS", "5")),
            audit_interval_seconds=int(os.getenv("MIST_FILEWATCHER_AUDIT_INTERVAL_SECONDS", "60")),
        )


@dataclass
class SkillDerivationConfig:
    """Configuration for skill derivation from tool usage patterns."""

    window_size: int = 50
    skill_threshold: int = 3
    capability_threshold: int = 5
    lookback_days: int = 7
    similarity_threshold: float = 0.7
    enabled: bool = True

    @classmethod
    def from_env(cls) -> "SkillDerivationConfig":
        """Load configuration from environment variables."""
        return cls(
            window_size=int(os.getenv("SKILL_WINDOW_SIZE", "50")),
            skill_threshold=int(os.getenv("SKILL_MIN_SESSION_USES", "3")),
            capability_threshold=int(os.getenv("SKILL_CAPABILITY_THRESHOLD", "5")),
            lookback_days=int(os.getenv("SKILL_LOOKBACK_DAYS", "7")),
            similarity_threshold=float(os.getenv("SKILL_SIMILARITY_THRESHOLD", "0.7")),
            enabled=os.getenv("SKILL_DERIVATION_ENABLED", "true").lower() == "true",
        )


@dataclass
class KnowledgeConfig:
    """Complete knowledge system configuration."""

    neo4j: Neo4jConfig
    llm: LLMConfig
    embedding: EmbeddingConfig
    extraction: ExtractionConfig

    # Event store (Layer 1)
    event_store: EventStoreConfig = None  # type: ignore[assignment]

    # Vector store (Layer 2)
    vector_store: VectorStoreConfig = None  # type: ignore[assignment]

    # Ingestion pipeline
    ingestion: IngestionConfig = None  # type: ignore[assignment]

    # Query intent classification
    query_intent: QueryIntentConfig = None  # type: ignore[assignment]

    # Skill derivation from tool usage
    skill_derivation: SkillDerivationConfig = None  # type: ignore[assignment]

    # Cluster 6: budget-aware context assembly
    context_budget: ContextBudgetConfig = None  # type: ignore[assignment]

    # Cluster 1: Stage 1.5 subject-scope classifier
    scope_classifier: ScopeClassifierConfig = None  # type: ignore[assignment]

    # Cluster 8 / ADR-010: vault-native memory architecture (Phase 1)
    vault: VaultConfig = None  # type: ignore[assignment]
    sidecar_index: SidecarIndexConfig = None  # type: ignore[assignment]
    filewatcher: FilewatcherConfig = None  # type: ignore[assignment]

    # Feature flags
    enable_knowledge_integration: bool = True  # Master switch for knowledge system

    # System settings
    ontology_version: str = "1.0.0"  # Current ontology version
    enable_versioning: bool = True  # Track ontology versions
    enable_provenance: bool = True  # Track extraction provenance

    # ADR-010 Cluster 8 Phase 8: rebuild determinism stamps. Stamped on every
    # DERIVED_FROM->VaultNote edge so vault-rebuild can detect version drift
    # and migrate forward when ontology / extraction prompt / model changes.
    # `extraction_version` should bump when EXTRACTION_SYSTEM_PROMPT or the
    # ontology contract changes. `model_hash` is an immutable identifier for
    # the LLM binary actually running extraction (recipe + quantization +
    # weights). Free-form strings; the only contract is that they change
    # when the underlying parameter changes.
    extraction_version: str = "2026-04-17-r1"
    model_hash: str = "gemma-4-e4b-q5-k-m-carteakey-full-v1"

    # Auto-RAG configuration
    auto_inject_docs: bool = True  # Enable/disable auto-injection
    auto_inject_limit: int = 3  # Number of chunks to auto-inject
    auto_inject_threshold: float = (
        0.4  # Similarity threshold for auto-injection (lowered for permissive matching)
    )

    def __post_init__(self):
        """Ensure sub-configs are never None."""
        if self.event_store is None:
            self.event_store = EventStoreConfig()
        if self.vector_store is None:
            self.vector_store = VectorStoreConfig()
        if self.ingestion is None:
            self.ingestion = IngestionConfig()
        if self.query_intent is None:
            self.query_intent = QueryIntentConfig()
        if self.skill_derivation is None:
            self.skill_derivation = SkillDerivationConfig()
        if self.context_budget is None:
            self.context_budget = ContextBudgetConfig()
        if self.scope_classifier is None:
            self.scope_classifier = ScopeClassifierConfig()
        if self.vault is None:
            self.vault = VaultConfig()
        if self.sidecar_index is None:
            self.sidecar_index = SidecarIndexConfig()
        if self.filewatcher is None:
            self.filewatcher = FilewatcherConfig()

    @classmethod
    def from_env(cls) -> "KnowledgeConfig":
        """Load complete configuration from environment."""
        return cls(
            neo4j=Neo4jConfig.from_env(),
            llm=LLMConfig.from_env(),
            embedding=EmbeddingConfig.from_env(),
            extraction=ExtractionConfig.from_env(),
            event_store=EventStoreConfig.from_env(),
            vector_store=VectorStoreConfig.from_env(),
            ingestion=IngestionConfig.from_env(),
            query_intent=QueryIntentConfig.from_env(),
            skill_derivation=SkillDerivationConfig.from_env(),
            context_budget=ContextBudgetConfig.from_env(),
            scope_classifier=ScopeClassifierConfig.from_env(),
            vault=VaultConfig.from_env(),
            sidecar_index=SidecarIndexConfig.from_env(),
            filewatcher=FilewatcherConfig.from_env(),
            enable_knowledge_integration=os.getenv("ENABLE_KNOWLEDGE_INTEGRATION", "true").lower()
            == "true",
            ontology_version=os.getenv("ONTOLOGY_VERSION", "1.0.0"),
            enable_versioning=os.getenv("ENABLE_VERSIONING", "true").lower() == "true",
            enable_provenance=os.getenv("ENABLE_PROVENANCE", "true").lower() == "true",
            extraction_version=os.getenv("EXTRACTION_VERSION", "2026-04-17-r1"),
            model_hash=os.getenv("MIST_MODEL_HASH", "gemma-4-e4b-q5-k-m-carteakey-full-v1"),
            auto_inject_docs=os.getenv("AUTO_INJECT_DOCS", "true").lower() == "true",
            auto_inject_limit=int(os.getenv("AUTO_INJECT_LIMIT", "3")),
            auto_inject_threshold=float(os.getenv("AUTO_INJECT_THRESHOLD", "0.4")),
        )


# Global config instance
_config: KnowledgeConfig | None = None


def get_config() -> KnowledgeConfig:
    """Get global knowledge system configuration."""
    global _config
    if _config is None:
        _config = KnowledgeConfig.from_env()
    return _config


def set_config(config: KnowledgeConfig):
    """Set global knowledge system configuration."""
    global _config
    _config = config
