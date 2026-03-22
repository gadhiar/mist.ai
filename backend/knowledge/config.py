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

    model: str = "qwen2.5:7b-instruct"  # Default Ollama model
    base_url: str = "http://localhost:11434"  # Ollama default
    temperature: float = 0.0  # Deterministic for extraction

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load configuration from environment variables."""
        return cls(
            model=os.getenv("MODEL", "qwen2.5:7b-instruct"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
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

    @classmethod
    def from_env(cls) -> "ExtractionConfig":
        """Load configuration from environment variables."""
        return cls(
            extract_node_properties=os.getenv("EXTRACT_NODE_PROPERTIES", "true").lower() == "true",
            extract_relationship_properties=os.getenv("EXTRACT_REL_PROPERTIES", "true").lower()
            == "true",
            min_extraction_confidence=float(os.getenv("MIN_EXTRACTION_CONFIDENCE", "0.5")),
        )


@dataclass
class KnowledgeConfig:
    """Complete knowledge system configuration."""

    neo4j: Neo4jConfig
    llm: LLMConfig
    embedding: EmbeddingConfig
    extraction: ExtractionConfig

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

    @classmethod
    def from_env(cls) -> "KnowledgeConfig":
        """Load complete configuration from environment."""
        return cls(
            neo4j=Neo4jConfig.from_env(),
            llm=LLMConfig.from_env(),
            embedding=EmbeddingConfig.from_env(),
            extraction=ExtractionConfig.from_env(),
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
