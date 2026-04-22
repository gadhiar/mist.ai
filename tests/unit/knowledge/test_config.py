"""Tests for backend.knowledge.config."""

import os
from contextlib import contextmanager

from backend.knowledge.config import (
    FilewatcherConfig,
    KnowledgeConfig,
    LLMConfig,
    SidecarIndexConfig,
    VaultConfig,
)


@contextmanager
def _env(**values):
    original = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in original.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestLLMConfigTemperatureSplit:
    """Cluster 3: LLMConfig has separate extraction and conversation temperatures."""

    def test_default_conversation_temperature_is_07(self):
        config = LLMConfig()
        assert config.conversation_temperature == 0.7

    def test_default_extraction_temperature_is_00(self):
        config = LLMConfig()
        assert config.temperature == 0.0

    def test_from_env_reads_llm_conversation_temperature(self):
        with _env(LLM_CONVERSATION_TEMPERATURE="0.5", LLM_TEMPERATURE=None):
            config = LLMConfig.from_env()
        assert config.conversation_temperature == 0.5
        assert config.temperature == 0.0  # extraction default unchanged

    def test_from_env_conversation_defaults_when_unset(self):
        with _env(LLM_CONVERSATION_TEMPERATURE=None, LLM_TEMPERATURE=None):
            config = LLMConfig.from_env()
        assert config.conversation_temperature == 0.7
        assert config.temperature == 0.0

    def test_fields_are_independent(self):
        with _env(LLM_TEMPERATURE="0.2", LLM_CONVERSATION_TEMPERATURE="0.9"):
            config = LLMConfig.from_env()
        assert config.temperature == 0.2
        assert config.conversation_temperature == 0.9


_VAULT_ENV_KEYS = (
    "MIST_VAULT_ENABLED",
    "MIST_VAULT_ROOT",
    "MIST_VAULT_DEFAULT_USER_ID",
    "MIST_VAULT_GIT_AUTO_INIT",
    "MIST_VAULT_SESSION_SOFT_CAP_TURNS",
    "MIST_VAULT_SESSION_SOFT_CAP_TOKENS",
    "MIST_VAULT_APPEND_SENTINEL",
    "MIST_VAULT_WRITER_QUEUE_MAX_DEPTH",
)

_SIDECAR_ENV_KEYS = (
    "MIST_SIDECAR_ENABLED",
    "MIST_SIDECAR_DB_PATH",
    "MIST_SIDECAR_EMBEDDING_DIMENSION",
    "MIST_SIDECAR_HEADING_CONTEXT_WEIGHT",
    "MIST_SIDECAR_CHUNK_MAX_CHARS",
    "MIST_SIDECAR_REBUILD_ON_STARTUP",
)

_FILEWATCHER_ENV_KEYS = (
    "MIST_FILEWATCHER_ENABLED",
    "MIST_FILEWATCHER_OBSERVER_TYPE",
    "MIST_FILEWATCHER_DEBOUNCE_MS",
    "MIST_FILEWATCHER_STALENESS_SLO_SECONDS",
    "MIST_FILEWATCHER_AUDIT_INTERVAL_SECONDS",
)


def _clear(*keys):
    return {k: None for k in keys}


class TestVaultConfigDefaults:
    """Cluster 8 / ADR-010: VaultConfig defaults match the ADR implementation defaults."""

    def test_defaults_match_adr_010(self):
        # Arrange / Act
        config = VaultConfig()

        # Assert
        assert config.enabled is True
        assert config.root == "mist-memory"
        assert config.default_user_id == "raj"
        assert config.git_auto_init is True
        assert config.session_soft_cap_turns == 20
        assert config.session_soft_cap_tokens == 6000
        assert config.append_sentinel == "<!-- MIST_APPEND_HERE -->"
        assert config.writer_queue_max_depth == 100

    def test_is_frozen(self):
        config = VaultConfig()
        try:
            config.root = "other"  # type: ignore[misc]
        except (AttributeError, Exception) as exc:
            assert "frozen" in str(exc).lower() or "cannot assign" in str(exc).lower()
        else:
            raise AssertionError("VaultConfig should be frozen but allowed mutation")


class TestVaultConfigFromEnv:
    """VaultConfig.from_env reads MIST_VAULT_* environment variables."""

    def test_from_env_uses_defaults_when_unset(self):
        with _env(**_clear(*_VAULT_ENV_KEYS)):
            config = VaultConfig.from_env()
        assert config == VaultConfig()

    def test_from_env_reads_root_override(self):
        with _env(**{**_clear(*_VAULT_ENV_KEYS), "MIST_VAULT_ROOT": "/custom/vault"}):
            config = VaultConfig.from_env()
        assert config.root == "/custom/vault"

    def test_from_env_reads_disabled_flag(self):
        with _env(**{**_clear(*_VAULT_ENV_KEYS), "MIST_VAULT_ENABLED": "false"}):
            config = VaultConfig.from_env()
        assert config.enabled is False

    def test_from_env_reads_session_soft_caps_as_int(self):
        with _env(
            **{
                **_clear(*_VAULT_ENV_KEYS),
                "MIST_VAULT_SESSION_SOFT_CAP_TURNS": "30",
                "MIST_VAULT_SESSION_SOFT_CAP_TOKENS": "9000",
            }
        ):
            config = VaultConfig.from_env()
        assert config.session_soft_cap_turns == 30
        assert config.session_soft_cap_tokens == 9000

    def test_from_env_reads_default_user_id(self):
        with _env(**{**_clear(*_VAULT_ENV_KEYS), "MIST_VAULT_DEFAULT_USER_ID": "alice"}):
            config = VaultConfig.from_env()
        assert config.default_user_id == "alice"

    def test_from_env_reads_writer_queue_max_depth(self):
        with _env(
            **{
                **_clear(*_VAULT_ENV_KEYS),
                "MIST_VAULT_WRITER_QUEUE_MAX_DEPTH": "250",
            }
        ):
            config = VaultConfig.from_env()
        assert config.writer_queue_max_depth == 250


class TestSidecarIndexConfigDefaults:
    """SidecarIndexConfig defaults match the ADR-010 sidecar schema section."""

    def test_defaults_match_adr_010(self):
        config = SidecarIndexConfig()

        assert config.enabled is True
        assert config.db_path == "data/vault_sidecar.db"
        assert config.embedding_dimension == 384
        assert config.heading_context_weight == 0.3
        assert config.chunk_max_chars == 6000
        assert config.rebuild_on_startup is False

    def test_embedding_dimension_matches_minilm_default(self):
        # The SidecarIndexConfig.embedding_dimension MUST match
        # EmbeddingConfig.dimension (384 for all-MiniLM-L6-v2). vec0 schema is
        # fixed at table-create time; mismatch produces silent retrieval failure.
        from backend.knowledge.config import EmbeddingConfig

        assert SidecarIndexConfig().embedding_dimension == EmbeddingConfig().dimension


class TestSidecarIndexConfigFromEnv:
    """SidecarIndexConfig.from_env reads MIST_SIDECAR_* environment variables."""

    def test_from_env_uses_defaults_when_unset(self):
        with _env(**_clear(*_SIDECAR_ENV_KEYS)):
            config = SidecarIndexConfig.from_env()
        assert config == SidecarIndexConfig()

    def test_from_env_reads_db_path(self):
        with _env(**{**_clear(*_SIDECAR_ENV_KEYS), "MIST_SIDECAR_DB_PATH": "/data/sidecar.db"}):
            config = SidecarIndexConfig.from_env()
        assert config.db_path == "/data/sidecar.db"

    def test_from_env_reads_embedding_dimension_as_int(self):
        with _env(**{**_clear(*_SIDECAR_ENV_KEYS), "MIST_SIDECAR_EMBEDDING_DIMENSION": "768"}):
            config = SidecarIndexConfig.from_env()
        assert config.embedding_dimension == 768

    def test_from_env_reads_heading_context_weight_as_float(self):
        with _env(
            **{
                **_clear(*_SIDECAR_ENV_KEYS),
                "MIST_SIDECAR_HEADING_CONTEXT_WEIGHT": "0.5",
            }
        ):
            config = SidecarIndexConfig.from_env()
        assert config.heading_context_weight == 0.5

    def test_from_env_reads_rebuild_on_startup(self):
        with _env(
            **{
                **_clear(*_SIDECAR_ENV_KEYS),
                "MIST_SIDECAR_REBUILD_ON_STARTUP": "true",
            }
        ):
            config = SidecarIndexConfig.from_env()
        assert config.rebuild_on_startup is True


class TestFilewatcherConfigDefaults:
    """FilewatcherConfig defaults match the ADR-010 filewatcher strategy section."""

    def test_defaults_match_adr_010(self):
        config = FilewatcherConfig()

        assert config.enabled is True
        assert config.observer_type == "auto"
        assert config.debounce_ms == 500
        assert config.staleness_slo_seconds == 5
        assert config.audit_interval_seconds == 60


class TestFilewatcherConfigFromEnv:
    """FilewatcherConfig.from_env reads MIST_FILEWATCHER_* environment variables."""

    def test_from_env_uses_defaults_when_unset(self):
        with _env(**_clear(*_FILEWATCHER_ENV_KEYS)):
            config = FilewatcherConfig.from_env()
        assert config == FilewatcherConfig()

    def test_from_env_reads_observer_type(self):
        with _env(
            **{
                **_clear(*_FILEWATCHER_ENV_KEYS),
                "MIST_FILEWATCHER_OBSERVER_TYPE": "polling",
            }
        ):
            config = FilewatcherConfig.from_env()
        assert config.observer_type == "polling"

    def test_from_env_reads_debounce_ms_as_int(self):
        with _env(**{**_clear(*_FILEWATCHER_ENV_KEYS), "MIST_FILEWATCHER_DEBOUNCE_MS": "1000"}):
            config = FilewatcherConfig.from_env()
        assert config.debounce_ms == 1000

    def test_from_env_reads_staleness_slo(self):
        with _env(
            **{
                **_clear(*_FILEWATCHER_ENV_KEYS),
                "MIST_FILEWATCHER_STALENESS_SLO_SECONDS": "10",
            }
        ):
            config = FilewatcherConfig.from_env()
        assert config.staleness_slo_seconds == 10

    def test_from_env_reads_audit_interval(self):
        with _env(
            **{
                **_clear(*_FILEWATCHER_ENV_KEYS),
                "MIST_FILEWATCHER_AUDIT_INTERVAL_SECONDS": "120",
            }
        ):
            config = FilewatcherConfig.from_env()
        assert config.audit_interval_seconds == 120

    def test_from_env_reads_disabled_flag(self):
        with _env(**{**_clear(*_FILEWATCHER_ENV_KEYS), "MIST_FILEWATCHER_ENABLED": "false"}):
            config = FilewatcherConfig.from_env()
        assert config.enabled is False


class TestKnowledgeConfigVaultWiring:
    """KnowledgeConfig __post_init__ and from_env wire vault subsystems."""

    def test_post_init_defaults_vault_subconfigs(self):
        # Arrange: pass only required sub-configs, leave Cluster 8 trio as None.
        from backend.knowledge.config import (
            EmbeddingConfig,
            ExtractionConfig,
            Neo4jConfig,
        )

        # Act
        config = KnowledgeConfig(
            neo4j=Neo4jConfig(),
            llm=LLMConfig(),
            embedding=EmbeddingConfig(),
            extraction=ExtractionConfig(),
        )

        # Assert
        assert isinstance(config.vault, VaultConfig)
        assert isinstance(config.sidecar_index, SidecarIndexConfig)
        assert isinstance(config.filewatcher, FilewatcherConfig)

    def test_from_env_loads_all_three_vault_configs(self):
        cleared = {
            **_clear(*_VAULT_ENV_KEYS),
            **_clear(*_SIDECAR_ENV_KEYS),
            **_clear(*_FILEWATCHER_ENV_KEYS),
        }
        with _env(**cleared):
            config = KnowledgeConfig.from_env()

        assert config.vault == VaultConfig()
        assert config.sidecar_index == SidecarIndexConfig()
        assert config.filewatcher == FilewatcherConfig()

    def test_post_init_preserves_explicit_vault_config(self):
        # Arrange
        from backend.knowledge.config import (
            EmbeddingConfig,
            ExtractionConfig,
            Neo4jConfig,
        )

        custom = VaultConfig(root="/tmp/custom-vault", enabled=False)

        # Act
        config = KnowledgeConfig(
            neo4j=Neo4jConfig(),
            llm=LLMConfig(),
            embedding=EmbeddingConfig(),
            extraction=ExtractionConfig(),
            vault=custom,
        )

        # Assert
        assert config.vault is custom
        assert config.vault.root == "/tmp/custom-vault"
        assert config.vault.enabled is False
