"""Tests for ServiceSettings.from_env()."""

from backend.extraction_service.settings import ServiceSettings

_ALL_ENV_VARS = [
    "EXTRACTION_LLM_BASE_URL",
    "EXTRACTION_MODEL_HASH",
    "EXTRACTION_MODEL_FILE",
    "EXTRACTION_ADAPTER",
    "EXTRACTION_REASONING_EFFORT",
    "EXTRACTION_REASONING_BUDGET_TOKENS",
    "EXTRACTION_CONSTRAINED_MODE",
    "EXTRACTION_LOCATION_LABEL",
    "EXTRACTION_LLM_TIMEOUT_SECONDS",
    "EXTRACTION_MAX_ATTEMPTS",
    "EXTRACTION_IDEMPOTENCY_CACHE_SIZE",
    "EXTRACTION_SCOPE_ENABLED",
    "EXTRACTION_TEMPERATURE",
    "LLAMA_CPP_BUILD",
    "EXTRACTION_PORT",
]


class TestServiceSettingsFromEnv:
    def test_defaults_when_no_env_set(self, monkeypatch):
        for key in _ALL_ENV_VARS:
            monkeypatch.delenv(key, raising=False)

        settings = ServiceSettings.from_env()

        assert settings.adapter_name == "gptoss"
        assert settings.reasoning_effort == "low"
        assert settings.reasoning_budget_tokens is None
        assert settings.constrained_mode is None
        assert settings.max_attempts == 2
        assert settings.idempotency_cache_size == 256
        assert settings.scope_enabled is True
        assert settings.temperature == 0.0
        assert settings.llama_cpp_build == "b11151"
        assert settings.port == 8090

    def test_reads_overrides_from_env(self, monkeypatch):
        monkeypatch.setenv("EXTRACTION_ADAPTER", "qwen")
        monkeypatch.setenv("EXTRACTION_REASONING_BUDGET_TOKENS", "128")
        monkeypatch.setenv("EXTRACTION_CONSTRAINED_MODE", "json_object")
        monkeypatch.setenv("EXTRACTION_SCOPE_ENABLED", "false")
        monkeypatch.setenv("EXTRACTION_MAX_ATTEMPTS", "3")
        monkeypatch.setenv("EXTRACTION_MODEL_HASH", "qwen-3.5-9b")

        settings = ServiceSettings.from_env()

        assert settings.adapter_name == "qwen"
        assert settings.reasoning_budget_tokens == 128
        assert settings.constrained_mode == "json_object"
        assert settings.scope_enabled is False
        assert settings.max_attempts == 3
        assert settings.model_hash == "qwen-3.5-9b"

    def test_reasoning_budget_tokens_unset_is_none(self, monkeypatch):
        monkeypatch.delenv("EXTRACTION_REASONING_BUDGET_TOKENS", raising=False)

        settings = ServiceSettings.from_env()

        assert settings.reasoning_budget_tokens is None

    def test_scope_enabled_accepts_common_truthy_and_falsy_spellings(self, monkeypatch):
        monkeypatch.setenv("EXTRACTION_SCOPE_ENABLED", "0")
        assert ServiceSettings.from_env().scope_enabled is False

        monkeypatch.setenv("EXTRACTION_SCOPE_ENABLED", "yes")
        assert ServiceSettings.from_env().scope_enabled is True
