"""Knowledge Graph Configuration.

Configuration for knowledge graph integration with voice system.
"""

import os
from dataclasses import dataclass


@dataclass
class KnowledgeConfig:
    """Configuration for knowledge graph integration."""

    # Neo4j connection
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "your_password")

    # LLM for knowledge operations
    knowledge_model: str = os.getenv("MODEL", "qwen2.5:7b-instruct").strip('"')

    # Feature flags
    enable_knowledge_integration: bool = True
    enable_auto_extraction: bool = True  # Automatically extract from user messages
    enable_auto_retrieval: bool = True  # Automatically retrieve context

    # Performance settings
    max_retrieval_facts: int = 20
    max_conversation_history: int = 10
    retrieval_similarity_threshold: float = 0.6


# Default configuration
DEFAULT_KNOWLEDGE_CONFIG = KnowledgeConfig()
