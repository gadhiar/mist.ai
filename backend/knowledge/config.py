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
    """LLM model configuration."""

    model: str = "qwen2.5:7b-instruct"
    base_url: str = "http://localhost:8080"  # llama-server default
    temperature: float = 0.0  # Deterministic for extraction
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

    # Feature flags
    enable_knowledge_integration: bool = True  # Master switch for knowledge system

    # System settings
    ontology_version: str = "1.0.0"  # Current ontology version
    enable_versioning: bool = True  # Track ontology versions
    enable_provenance: bool = True  # Track extraction provenance

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
            enable_knowledge_integration=os.getenv("ENABLE_KNOWLEDGE_INTEGRATION", "true").lower()
            == "true",
            ontology_version=os.getenv("ONTOLOGY_VERSION", "1.0.0"),
            enable_versioning=os.getenv("ENABLE_VERSIONING", "true").lower() == "true",
            enable_provenance=os.getenv("ENABLE_PROVENANCE", "true").lower() == "true",
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
