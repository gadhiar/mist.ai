"""Data models for the LLM provider abstraction.

LLMRequest and LLMResponse are the wire types for all inference calls.
ToolCall captures structured tool invocations returned by the model.
"""

import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A structured tool invocation returned by the model."""

    id: str
    name: str
    arguments: dict

    def to_openai_dict(self) -> dict:
        """Serialize to OpenAI-compatible tool_call dict for message history."""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments),
            },
        }


class UsageMetadata(BaseModel):
    """Token usage statistics from a completed generation."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class LLMRequest(BaseModel):
    """Parameters for an LLM generation call."""

    messages: list[dict[str, Any]]
    tools: list[dict] | None = None
    temperature: float = 0.7
    max_tokens: int = 400
    top_p: float = 0.9
    json_mode: bool = False


class LLMResponse(BaseModel):
    """A (possibly partial) response from the LLM."""

    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    partial: bool = False
    usage: UsageMetadata | None = None
