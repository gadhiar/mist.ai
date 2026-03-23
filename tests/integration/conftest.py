"""Integration test fixtures.

These fixtures provide real Neo4j and Ollama connections for
integration testing. Requires running services:
  - Neo4j on bolt://localhost:7687 (neo4j/password)
  - Ollama on http://localhost:11434 with qwen2.5:7b

Run with: pytest tests/integration/ -v
Start stack first: ./scripts/start-dev.sh
"""

import pytest

from backend.knowledge.config import (
    EmbeddingConfig,
    EventStoreConfig,
    ExtractionConfig,
    KnowledgeConfig,
    LLMConfig,
    Neo4jConfig,
)


def _services_available() -> bool:
    """Check if Neo4j and Ollama are reachable."""
    import socket

    for host, port in [("localhost", 7687), ("localhost", 11434)]:
        try:
            sock = socket.create_connection((host, port), timeout=2)
            sock.close()
        except (OSError, ConnectionRefusedError):
            return False
    return True


skipif_no_services = pytest.mark.skipif(
    not _services_available(),
    reason="Neo4j and/or Ollama not running (start with ./scripts/start-dev.sh)",
)


@pytest.fixture
def integration_config() -> KnowledgeConfig:
    """Real config pointing at local Neo4j + Ollama."""
    return KnowledgeConfig(
        neo4j=Neo4jConfig(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="password",
        ),
        embedding=EmbeddingConfig(model_name="all-MiniLM-L6-v2"),
        llm=LLMConfig(model="qwen2.5:7b-instruct"),
        extraction=ExtractionConfig(min_extraction_confidence=0.5),
        event_store=EventStoreConfig(enabled=False),
    )
