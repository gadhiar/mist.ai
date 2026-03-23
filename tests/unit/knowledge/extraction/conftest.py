"""Extraction test fixtures."""

import pytest

from tests.mocks.ollama import FakeLLM


@pytest.fixture
def fake_llm():
    """A FakeLLM with default empty extraction response."""
    return FakeLLM()
