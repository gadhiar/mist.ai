"""Tests for backend.knowledge.config."""

import os
from contextlib import contextmanager

from backend.knowledge.config import LLMConfig


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
