"""Test configuration factory.

Provides build_test_config() for creating KnowledgeConfig with test
defaults. Always use this instead of KnowledgeConfig.from_env() to
avoid .env bleed into test isolation.
"""

from backend.knowledge.config import (
    EmbeddingConfig,
    EventStoreConfig,
    ExtractionConfig,
    KnowledgeConfig,
    LLMConfig,
    Neo4jConfig,
)

# Standard test constants -- use these instead of magic strings
TEST_USER_ID = "user-test-001"
TEST_SESSION_ID = "session-test-001"
TEST_EVENT_ID = "event-test-001"


def build_test_config(
    *,
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "test",
    embedding_model: str = "test-model",
    llm_model: str = "test-model",
    llm_backend: str = "llamacpp",
    min_extraction_confidence: float = 0.5,
    event_store_enabled: bool = False,
    event_store_db_path: str = ":memory:",
) -> KnowledgeConfig:
    """Build a KnowledgeConfig with test defaults.

    All parameters are keyword-only for type safety and IDE autocomplete.
    """
    return KnowledgeConfig(
        neo4j=Neo4jConfig(uri=neo4j_uri, username=neo4j_user, password=neo4j_password),
        embedding=EmbeddingConfig(model_name=embedding_model),
        llm=LLMConfig(model=llm_model, backend=llm_backend),
        extraction=ExtractionConfig(min_extraction_confidence=min_extraction_confidence),
        event_store=EventStoreConfig(enabled=event_store_enabled, db_path=event_store_db_path),
    )
