"""Tests for LLM data models."""

import pytest

from backend.llm.models import LLMRequest, LLMResponse, ToolCall, UsageMetadata


class TestLLMRequest:
    def test_default_values(self):
        request = LLMRequest(
            messages=[{"role": "user", "content": "hello"}],
        )

        assert request.temperature == 0.7
        assert request.max_tokens == 400
        assert request.top_p == 0.9
        assert request.json_mode is False
        assert request.tools is None

    def test_explicit_values(self):
        tools = [{"type": "function", "function": {"name": "test"}}]
        request = LLMRequest(
            messages=[{"role": "user", "content": "hello"}],
            tools=tools,
            temperature=0.0,
            max_tokens=1000,
            json_mode=True,
        )

        assert request.temperature == 0.0
        assert request.max_tokens == 1000
        assert request.json_mode is True
        assert request.tools == tools


class TestLLMResponse:
    def test_partial_response(self):
        response = LLMResponse(content="hello", partial=True)

        assert response.content == "hello"
        assert response.partial is True
        assert response.tool_calls is None
        assert response.usage is None

    def test_complete_response_with_tool_calls(self):
        tc = ToolCall(id="call_1", name="search", arguments={"query": "test"})
        response = LLMResponse(
            content="",
            tool_calls=[tc],
            partial=False,
        )

        assert response.tool_calls[0].name == "search"
        assert response.tool_calls[0].arguments == {"query": "test"}
        assert response.tool_calls[0].id == "call_1"

    def test_response_with_usage(self):
        usage = UsageMetadata(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        response = LLMResponse(content="ok", usage=usage)

        assert response.usage.prompt_tokens == 10
        assert response.usage.total_tokens == 30


class TestToolCall:
    def test_frozen(self):
        tc = ToolCall(id="call_1", name="fn", arguments={"a": 1})

        with pytest.raises(AttributeError):
            tc.name = "other"

    def test_to_openai_dict(self):
        tc = ToolCall(id="call_1", name="search", arguments={"q": "test"})
        d = tc.to_openai_dict()

        assert d == {
            "id": "call_1",
            "type": "function",
            "function": {"name": "search", "arguments": '{"q": "test"}'},
        }
