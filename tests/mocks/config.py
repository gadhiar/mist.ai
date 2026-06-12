"""Test configuration factory.

Provides build_test_config() for creating KnowledgeConfig with test
defaults. Always use this instead of KnowledgeConfig.from_env() to
avoid .env bleed into test isolation.
"""

import os

from backend.knowledge.config import (
    EmbeddingConfig,
    EventStoreConfig,
    ExtractionConfig,
    KnowledgeConfig,
    LLMConfig,
    Neo4jConfig,
    VaultConfig,
)

# Standard test constants -- use these instead of magic strings
TEST_USER_ID = "user-test-001"
TEST_SESSION_ID = "session-test-001"
TEST_EVENT_ID = "event-test-001"

# Test-vault baseline default user_id. Matches users/test-user.md in
# tests/fixtures/test-vault/. Override via build_test_config(vault_user_id=...)
# if a test requires a different user identity.
TEST_VAULT_USER_ID = "test-user"


def build_test_config(
    *,
    neo4j_uri: str | None = None,
    neo4j_user: str | None = None,
    neo4j_password: str | None = None,
    embedding_model: str = "test-model",
    llm_model: str = "test-model",
    llm_backend: str = "llamacpp",
    min_extraction_confidence: float = 0.5,
    event_store_enabled: bool = False,
    event_store_db_path: str = ":memory:",
    vault_root: str | None = None,
    vault_enabled: bool = False,
    vault_user_id: str = TEST_VAULT_USER_ID,
    vault_git_auto_init: bool = False,
) -> KnowledgeConfig:
    """Build a KnowledgeConfig with test defaults.

    All parameters are keyword-only for type safety and IDE autocomplete.

    Vault parameters:
        vault_root: Path to the vault directory. If None, an explicitly
            DISABLED VaultConfig with a placeholder root is constructed --
            the production VaultConfig() defaults to enabled=True with
            root="mist-memory", which silently leaked LIVE-vault reads
            (curated profile injection) into vault-less unit tests.
        vault_enabled: Whether vault code paths execute. Defaults False so
            most unit tests do not exercise vault writes. Set True for
            integration tests that need the real vault flow.
        vault_user_id: Default user_id for vault writes. Defaults to
            TEST_VAULT_USER_ID (matches tests/fixtures/test-vault/users/test-user.md).
        vault_git_auto_init: Whether to git-init the vault on first use.
            Defaults False for tests (ephemeral vaults do not need git);
            production default is True.
    """
    # Honor container env vars so the same test config works inside
    # docker (where Neo4j is at mist-neo4j:7687) and on host (where the
    # docker-compose port mapping exposes Neo4j at localhost:7687).
    resolved_neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
    resolved_neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
    resolved_neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "test")

    vault_config = VaultConfig(
        enabled=vault_enabled if vault_root is not None else False,
        root=vault_root or "test-vault-unset",
        # Vault-less tests keep the dataclass-default user id so
        # _user_id_for_vault resolves to "user" exactly as before; an
        # explicit vault_root opts into TEST_VAULT_USER_ID (matches
        # tests/fixtures/test-vault/users/test-user.md).
        default_user_id=vault_user_id if vault_root is not None else "raj",
        git_auto_init=vault_git_auto_init,
    )
    return KnowledgeConfig(
        neo4j=Neo4jConfig(
            uri=resolved_neo4j_uri,
            username=resolved_neo4j_user,
            password=resolved_neo4j_password,
        ),
        embedding=EmbeddingConfig(model_name=embedding_model),
        llm=LLMConfig(model=llm_model, backend=llm_backend),
        extraction=ExtractionConfig(min_extraction_confidence=min_extraction_confidence),
        event_store=EventStoreConfig(enabled=event_store_enabled, db_path=event_store_db_path),
        vault=vault_config,
    )
