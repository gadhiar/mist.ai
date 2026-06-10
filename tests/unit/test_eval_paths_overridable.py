"""Guardrail: the three live memory-store paths are env-overridable.

Eval/gauntlet runs that exercise the memory layer (via
`scripts/mist_admin.py` replay/chat -> `build_conversation_handler` ->
full retrieval + extraction + graph + vault writes) must be able to
redirect the three stores to throwaway locations so a run cannot pollute
live memory (the synthetic 37A8 corpus incident). Isolation works by
overriding three env vars at the container/process level, which the
config layer reads in its `from_env` constructors:

| Store        | Config class           | Env var               | Attribute |
|--------------|------------------------|-----------------------|-----------|
| Sidecar DB   | `SidecarIndexConfig`   | `MIST_SIDECAR_DB_PATH`| `db_path` |
| Event store  | `EventStoreConfig`     | `EVENT_STORE_DB_PATH` | `db_path` |
| Vault root   | `VaultConfig`          | `MIST_VAULT_ROOT`     | `root`    |

This test locks that contract: if any of these reads is replaced by a
hardcoded literal (regression), the corresponding assertion fails. It
passes against current code because the env reads already exist; it is a
characterization/guardrail test, not a driver of new behavior.

Each test uses `monkeypatch.setenv` so the override is scoped to the test
and reverted on teardown. These assert on `from_env()` directly (not
`get_config()`), so the module-global `_config` memoization and import-time
`load_dotenv()` are irrelevant here -- the contract under test is the
constructor's read of the env var, which is the same read an
override-before-import eval invocation relies on.
"""

from backend.knowledge.config import (
    EventStoreConfig,
    Neo4jConfig,
    SidecarIndexConfig,
    VaultConfig,
)


class TestSidecarDbPathOverridable:
    """SidecarIndexConfig.from_env honors MIST_SIDECAR_DB_PATH."""

    def test_from_env_honors_sidecar_db_path_override(self, monkeypatch):
        # Arrange
        monkeypatch.setenv("MIST_SIDECAR_DB_PATH", "/app/data/eval-run/vault_sidecar.db")

        # Act
        config = SidecarIndexConfig.from_env()

        # Assert
        assert config.db_path == "/app/data/eval-run/vault_sidecar.db"

    def test_from_env_uses_live_default_when_unset(self, monkeypatch):
        # Arrange: ensure no override leaks in from the ambient process env.
        monkeypatch.delenv("MIST_SIDECAR_DB_PATH", raising=False)

        # Act
        config = SidecarIndexConfig.from_env()

        # Assert: the live default is unchanged (non-breaking).
        assert config.db_path == "data/vault_sidecar.db"


class TestEventStoreDbPathOverridable:
    """EventStoreConfig.from_env honors EVENT_STORE_DB_PATH."""

    def test_from_env_honors_event_store_db_path_override(self, monkeypatch):
        # Arrange
        monkeypatch.setenv("EVENT_STORE_DB_PATH", "/app/data/eval-run/event_store.db")

        # Act
        config = EventStoreConfig.from_env()

        # Assert
        assert config.db_path == "/app/data/eval-run/event_store.db"

    def test_from_env_db_path_is_none_when_unset(self, monkeypatch):
        # Arrange: EventStoreConfig has no string literal default -- db_path is
        # None when the env var is absent (resolved downstream to ~/.mist/).
        monkeypatch.delenv("EVENT_STORE_DB_PATH", raising=False)

        # Act
        config = EventStoreConfig.from_env()

        # Assert
        assert config.db_path is None


class TestVaultRootOverridable:
    """VaultConfig.from_env honors MIST_VAULT_ROOT."""

    def test_from_env_honors_vault_root_override(self, monkeypatch):
        # Arrange
        monkeypatch.setenv("MIST_VAULT_ROOT", "/app/data/eval-run/vault")

        # Act
        config = VaultConfig.from_env()

        # Assert
        assert config.root == "/app/data/eval-run/vault"

    def test_from_env_uses_live_default_when_unset(self, monkeypatch):
        # Arrange
        monkeypatch.delenv("MIST_VAULT_ROOT", raising=False)

        # Act
        config = VaultConfig.from_env()

        # Assert: the live default is unchanged (non-breaking).
        assert config.root == "mist-memory"


class TestThrowawayTrioIsolatedTogether:
    """All three overrides applied at once redirect every store to one
    per-run throwaway dir -- the exact pattern the eval runbook documents.
    """

    def test_all_three_paths_redirect_under_one_eval_dir(self, monkeypatch):
        # Arrange: the throwaway-trio as the runbook prescribes.
        monkeypatch.setenv("MIST_SIDECAR_DB_PATH", "/app/data/eval-run/vault_sidecar.db")
        monkeypatch.setenv("EVENT_STORE_DB_PATH", "/app/data/eval-run/event_store.db")
        monkeypatch.setenv("MIST_VAULT_ROOT", "/app/data/eval-run/vault")

        # Act
        sidecar = SidecarIndexConfig.from_env()
        event_store = EventStoreConfig.from_env()
        vault = VaultConfig.from_env()

        # Assert: every store points under the throwaway dir, none at live paths.
        assert sidecar.db_path == "/app/data/eval-run/vault_sidecar.db"
        assert event_store.db_path == "/app/data/eval-run/event_store.db"
        assert vault.root == "/app/data/eval-run/vault"


class TestNeo4jUriOverridable:
    """Neo4jConfig.from_env honors NEO4J_URI / NEO4J_DATABASE (the 4th leg).

    Neo4j Community has one user database, so eval isolation selects a separate
    INSTANCE via NEO4J_URI rather than a database name. This guardrail locks
    that the URI/database are env-overridable; a regression to a hardcoded URI
    would break eval isolation silently.
    """

    def test_from_env_honors_neo4j_uri_override(self, monkeypatch):
        monkeypatch.setenv("NEO4J_URI", "bolt://mist-neo4j-eval:7687")
        config = Neo4jConfig.from_env()
        assert config.uri == "bolt://mist-neo4j-eval:7687"

    def test_from_env_honors_neo4j_database_override(self, monkeypatch):
        monkeypatch.setenv("NEO4J_DATABASE", "evaldb")
        config = Neo4jConfig.from_env()
        assert config.database == "evaldb"

    def test_from_env_uses_live_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("NEO4J_URI", raising=False)
        config = Neo4jConfig.from_env()
        assert config.uri == "bolt://localhost:7687"
