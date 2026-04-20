"""Tests for LLM data models."""

import json

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


class TestLLMRequestToolCallMessages:
    """Bug C: LLMRequest.messages must accept assistant messages with tool_calls."""

    def test_accepts_assistant_message_with_tool_calls(self):
        """An assistant message with a tool_calls list-of-dicts must validate."""
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": json.dumps({"expression": "2+2"}),
                        },
                    }
                ],
            },
            {"role": "tool", "content": "4", "tool_call_id": "call_abc123"},
        ]

        request = LLMRequest(messages=messages, temperature=0.7, max_tokens=400)

        assert len(request.messages) == 3
        assert request.messages[1]["tool_calls"][0]["function"]["name"] == "calculator"

    def test_accepts_many_tool_call_turns_without_validation_error(self):
        """Simulate the 6+ turn accumulation that triggers Bug C in gauntlet."""
        messages = []
        for turn in range(8):
            messages.append({"role": "user", "content": f"turn {turn}"})
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call_{turn}",
                            "type": "function",
                            "function": {"name": "fn", "arguments": json.dumps({})},
                        }
                    ],
                }
            )
            messages.append({"role": "tool", "content": "ok", "tool_call_id": f"call_{turn}"})

        # Pre-Cluster-4: raises pydantic.ValidationError on the dict-with-list-value.
        request = LLMRequest(messages=messages, temperature=0.7, max_tokens=400)
        assert len(request.messages) == 24  # 8 turns x (user + assistant + tool)
