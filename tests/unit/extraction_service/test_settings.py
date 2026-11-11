"""Tests for ServiceSettings.from_env()."""

import os

import pytest

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
    "EXTRACTION_TURN_PLAN",
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
        assert settings.turn_plan is False

    @pytest.mark.parametrize("value", ["", "  ", "0", "false", "off", "no", "maybe"])
    def test_the_turn_plan_stays_off_for_an_empty_or_falsy_value(self, monkeypatch, value):
        """Compose forwards an unset `${EXTRACTION_TURN_PLAN:-}` as an empty string."""
        monkeypatch.setenv("EXTRACTION_TURN_PLAN", value)

        assert ServiceSettings.from_env().turn_plan is False

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", " on ", "yes"])
    def test_the_turn_plan_is_on_for_a_truthy_value(self, monkeypatch, value):
        monkeypatch.setenv("EXTRACTION_TURN_PLAN", value)

        assert ServiceSettings.from_env().turn_plan is True

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


def _clear_server_env(monkeypatch) -> None:
    for key in list(os.environ):
        if key.startswith("EXTRACTION_SERVER"):
            monkeypatch.delenv(key)


class TestServerConfigFromEnv:
    """The declared llama-server settings, `EXTRACTION_SERVER_<NAME>` (MIS-196)."""

    def test_none_declared_is_empty(self, monkeypatch):
        _clear_server_env(monkeypatch)

        assert ServiceSettings.from_env().server_config == ()

    def test_every_declared_setting_is_read_lower_cased_stripped_and_sorted(self, monkeypatch):
        _clear_server_env(monkeypatch)
        monkeypatch.setenv("EXTRACTION_SERVER_N_CPU_MOE", " 24 ")
        monkeypatch.setenv("EXTRACTION_SERVER_DEVICE", "rtx4070s")
        monkeypatch.setenv("EXTRACTION_SERVER_IMAGE", "ghcr.io/ggml-org/llama.cpp@sha256:ab\n")
        monkeypatch.setenv("EXTRACTION_SERVER_CTX_SIZE", "16384")

        assert ServiceSettings.from_env().server_config == (
            ("ctx_size", "16384"),
            ("device", "rtx4070s"),
            ("image", "ghcr.io/ggml-org/llama.cpp@sha256:ab"),
            ("n_cpu_moe", "24"),
        )

    def test_an_empty_declaration_is_kept_as_declared(self, monkeypatch):
        """An unset `${VAR:-}` arrives as an empty string; the service reports it so."""
        _clear_server_env(monkeypatch)
        monkeypatch.setenv("EXTRACTION_SERVER_DEVICE", "  ")

        assert ServiceSettings.from_env().server_config == (("device", ""),)

    @pytest.mark.parametrize(
        "name",
        [
            pytest.param("EXTRACTION_SERVER", id="the-prefix-without-a-name"),
            pytest.param("EXTRACTION_SERVER_", id="the-prefix-with-an-empty-name"),
            pytest.param("EXTRACTION_SERVERS_DEVICE", id="another-prefix"),
            pytest.param("EXTRACTION_LLM_BASE_URL", id="another-service-setting"),
        ],
    )
    def test_other_variables_are_not_declarations(self, monkeypatch, name):
        _clear_server_env(monkeypatch)
        monkeypatch.setenv(name, "x")

        assert ServiceSettings.from_env().server_config == ()

    def test_a_setting_declared_twice_is_refused(self):
        with pytest.raises(ValueError, match="device"):
            ServiceSettings(
                llm_base_url="http://x",
                model_hash="m",
                model_file="f",
                server_config=(("device", "a"), ("device", "b")),
            )
