"""Data models for the LLM provider abstraction.

LLMRequest and LLMResponse are the wire types for all inference calls.
ToolCall captures structured tool invocations returned by the model.
ThinkingConfig carries model-agnostic reasoning/thinking controls that
LlamaServerProvider translates into llama.cpp's specific request fields.
"""

import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, field_validator


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
    """Token usage statistics from a completed generation.

    `cached_prompt_tokens` is how many prompt tokens the server took from its
    KV cache instead of processing again; None when the server did not say.
    """

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cached_prompt_tokens: int | None = None


class ThinkingConfig(BaseModel):
    """Model-agnostic reasoning/thinking controls for an LLM request.

    `budget_tokens=None` means unbudgeted: the field is omitted from the
    outgoing request entirely, letting llama-server fall back to its own
    `--reasoning-budget` default. llama.cpp b11151 treats a per-request
    `-1` as "not provided" and silently falls back to that same server
    default -- so `-1` can never mean "unbudgeted" here, it would just be a
    confusing spelling of `None`. Use `budget_tokens=0` to end thinking
    immediately, and leave it `None` for "unbudgeted".

    `effort` and `enabled` are gpt-oss/Qwen chat-template hints
    (`reasoning_effort`, `enable_thinking`) forwarded via
    `chat_template_kwargs`; either, both, or neither may be set.
    """

    budget_tokens: int | None = None
    effort: Literal["low", "medium", "high"] | None = None
    enabled: bool | None = None

    @field_validator("budget_tokens")
    @classmethod
    def _reject_unprovided_sentinel(cls, value: int | None) -> int | None:
        """Reject -1: llama.cpp's own "not provided" sentinel, never ours."""
        if value is not None and value < 0:
            raise ValueError(
                "budget_tokens must be >= 0 when set; -1 is llama.cpp's "
                "'not provided' sentinel and would silently fall back to "
                "the server's --reasoning-budget default instead of "
                "applying a zero-length budget -- omit the field "
                "(budget_tokens=None) to mean 'unbudgeted' instead"
            )
        return value


class LLMRequest(BaseModel):
    """Parameters for an LLM generation call.

    `cache_prompt` and `current_date` are llama-server request fields, sent
    only when set, so a caller that sets neither sends the body it sent before
    they existed. `cache_prompt` False has llama-server process the prompt from
    its first token instead of reusing a cached prefix, whose values depend on
    the batch shape that computed them. `current_date` is a chat template
    variable (`chat_template_kwargs`): the extraction service's shipped gpt-oss
    template writes it as the system message's date in place of the server's
    clock.
    """

    messages: list[dict[str, Any]]
    tools: list[dict] | None = None
    temperature: float = 0.7
    max_tokens: int = 400
    top_p: float = 0.9
    json_mode: bool = False
    thinking: ThinkingConfig | None = None
    response_schema: dict[str, Any] | None = None
    cache_prompt: bool | None = None
    current_date: str | None = None


class LLMResponse(BaseModel):
    """A (possibly partial) response from the LLM."""

    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    partial: bool = False
    usage: UsageMetadata | None = None
    reasoning_content: str | None = None
    finish_reason: str | None = None
