"""Unit tests for the model-family adapters.

Covers the Harmony raw-markup fallback and the Qwen <think> strip named in
the T0 brief's acceptance criteria, plus the thinking-control wiring each
adapter's `prepare()` is responsible for.
"""

import pytest

from backend.extraction_service.adapters import (
    GemmaAdapter,
    GptOssAdapter,
    QwenAdapter,
    get_adapter,
)
from backend.llm.models import LLMRequest, LLMResponse


def _request() -> LLMRequest:
    return LLMRequest(messages=[{"role": "user", "content": "hi"}])


class TestGptOssAdapter:
    def test_prepare_sets_effort_and_omits_budget_when_unset(self):
        adapter = GptOssAdapter(reasoning_effort="medium")

        prepared = adapter.prepare(_request(), stage="extract")

        assert prepared.thinking.effort == "medium"
        assert prepared.thinking.budget_tokens is None

    def test_prepare_forwards_reasoning_budget_tokens_when_set(self):
        adapter = GptOssAdapter(reasoning_effort="high", reasoning_budget_tokens=512)

        prepared = adapter.prepare(_request(), stage="extract")

        assert prepared.thinking.budget_tokens == 512

    def test_final_text_returns_plain_content_unchanged(self):
        adapter = GptOssAdapter()
        response = LLMResponse(content='{"entities": []}')

        assert adapter.final_text(response) == '{"entities": []}'

    def test_final_text_extracts_final_channel_from_raw_harmony_markup(self):
        """b11151 may return the raw Harmony transcript instead of just the
        final-channel text; the adapter must recover the final message.
        """
        adapter = GptOssAdapter()
        raw = (
            "<|channel|>analysis<|message|>thinking about it<|end|>"
            '<|start|>assistant<|channel|>final<|message|>{"entities": []}<|end|>'
        )
        response = LLMResponse(content=raw)

        assert adapter.final_text(response) == '{"entities": []}'

    def test_final_text_handles_missing_end_marker(self):
        adapter = GptOssAdapter()
        raw = '<|channel|>final<|message|>{"entities": []}'
        response = LLMResponse(content=raw)

        assert adapter.final_text(response) == '{"entities": []}'

    def test_default_constrained_mode_is_schema(self):
        # "none" makes llama.cpp b11151's peg parser reject gpt-oss's
        # `<|constrain|>JSON` Harmony wrapper (HTTP 500 on the scope call).
        assert GptOssAdapter().default_constrained_mode == "schema"


class TestQwenAdapter:
    def test_prepare_disables_thinking(self):
        prepared = QwenAdapter().prepare(_request(), stage="extract")

        assert prepared.thinking.enabled is False

    def test_final_text_strips_leading_think_block(self):
        adapter = QwenAdapter()
        response = LLMResponse(content='<think>reasoning here</think>\n{"entities": []}')

        assert adapter.final_text(response) == '{"entities": []}'

    def test_final_text_passes_through_when_no_think_block(self):
        adapter = QwenAdapter()
        response = LLMResponse(content='{"entities": []}')

        assert adapter.final_text(response) == '{"entities": []}'

    def test_default_constrained_mode_is_schema(self):
        assert QwenAdapter().default_constrained_mode == "schema"


class TestGemmaAdapter:
    def test_prepare_disables_thinking(self):
        prepared = GemmaAdapter().prepare(_request(), stage="extract")

        assert prepared.thinking.enabled is False

    def test_final_text_returns_content_unchanged(self):
        adapter = GemmaAdapter()
        response = LLMResponse(content='{"entities": []}')

        assert adapter.final_text(response) == '{"entities": []}'

    def test_default_constrained_mode_is_schema(self):
        assert GemmaAdapter().default_constrained_mode == "schema"


class TestGetAdapterRegistry:
    def test_gptoss_forwards_reasoning_kwargs(self):
        adapter = get_adapter("gptoss", reasoning_effort="high", reasoning_budget_tokens=64)

        assert isinstance(adapter, GptOssAdapter)
        assert adapter.reasoning_effort == "high"
        assert adapter.reasoning_budget_tokens == 64

    def test_qwen_and_gemma_ignore_reasoning_kwargs(self):
        assert isinstance(get_adapter("qwen", reasoning_effort="high"), QwenAdapter)
        assert isinstance(get_adapter("gemma", reasoning_effort="high"), GemmaAdapter)

    def test_unknown_adapter_raises_value_error(self):
        with pytest.raises(ValueError):
            get_adapter("not-a-real-adapter")
