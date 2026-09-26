"""LlamaServerProvider -- wraps llama-server's OpenAI-compatible API.

Uses the openai Python package (both async and sync clients) to
communicate with llama-server. The sync path is used by the voice
pipeline; the async path by conversation handling and extraction.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator, Generator
from typing import Any

import httpx
from openai import AsyncOpenAI, OpenAI

from backend.errors import LLMResponseError
from backend.llm.models import LLMRequest, LLMResponse, ToolCall, UsageMetadata
from backend.llm.provider import StreamingLLMProvider

logger = logging.getLogger(__name__)


class LlamaServerProvider(StreamingLLMProvider):
    """StreamingLLMProvider backed by llama-server's OpenAI-compatible API."""

    def __init__(self, base_url: str, model: str) -> None:
        self.model = model
        self._base_url = base_url
        self._async_client = AsyncOpenAI(base_url=f"{base_url}/v1", api_key="not-needed")
        self._sync_client = OpenAI(base_url=f"{base_url}/v1", api_key="not-needed")

    def _build_kwargs(self, request: LLMRequest, stream: bool) -> dict:
        """Build kwargs dict for the OpenAI chat completions API."""
        kwargs: dict = {
            "model": self.model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "stream": stream,
        }
        if request.response_schema is not None:
            # Constrained decoding takes precedence over plain json_mode.
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": request.response_schema,
                    "strict": True,
                },
            }
        elif request.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if request.tools:
            kwargs["tools"] = request.tools

        extra_body = self._build_extra_body(request)
        if extra_body:
            kwargs["extra_body"] = extra_body

        if stream:
            # Ask llama-server to emit a final usage-only chunk so streamed
            # calls can still report token counts (that chunk has no
            # choices -- see the empty-choices guard in generate()).
            kwargs["stream_options"] = {"include_usage": True}

        return kwargs

    def _build_extra_body(self, request: LLMRequest) -> dict:
        """Build the `extra_body` dict carrying llama.cpp's thinking controls.

        `reasoning_budget_tokens` is omitted entirely when
        `thinking.budget_tokens` is None (llama.cpp b11151: a request value
        of -1 means "not provided" and falls back to the server's
        `--reasoning-budget`, so "unbudgeted" must mean omitting the field,
        never sending -1 -- `ThinkingConfig` already rejects -1 at
        construction time). `chat_template_kwargs` is omitted entirely when
        neither `effort` nor `enabled` is set.
        """
        extra_body: dict = {}
        thinking = request.thinking
        if thinking is None:
            return extra_body

        if thinking.budget_tokens is not None:
            extra_body["reasoning_budget_tokens"] = thinking.budget_tokens

        chat_template_kwargs: dict = {}
        if thinking.effort is not None:
            chat_template_kwargs["reasoning_effort"] = thinking.effort
        if thinking.enabled is not None:
            chat_template_kwargs["enable_thinking"] = thinking.enabled
        if chat_template_kwargs:
            extra_body["chat_template_kwargs"] = chat_template_kwargs

        return extra_body

    def _parse_tool_calls(self, raw_tool_calls) -> list[ToolCall] | None:
        """Parse OpenAI tool_calls into our ToolCall dataclass."""
        if not raw_tool_calls:
            return None
        return [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments),
            )
            for tc in raw_tool_calls
        ]

    def _parse_usage(self, raw_usage) -> UsageMetadata | None:
        """Parse OpenAI usage into our UsageMetadata."""
        if not raw_usage:
            return None
        return UsageMetadata(
            prompt_tokens=raw_usage.prompt_tokens,
            completion_tokens=raw_usage.completion_tokens,
            total_tokens=raw_usage.total_tokens,
        )

    def _accumulate_tool_call_deltas(self, acc: dict[int, dict[str, Any]], deltas) -> None:
        """Fold one chunk's tool_call delta fragments into the running accumulator.

        OpenAI-compatible streaming splits each tool call across multiple
        chunks, correlated by `index`. `id` and `function.name` typically
        arrive once (on the first fragment for that index); `function.arguments`
        arrives incrementally and must be concatenated, not overwritten.
        """
        for tc_delta in deltas:
            entry = acc.setdefault(tc_delta.index, {"id": None, "name": None, "arguments": ""})
            if getattr(tc_delta, "id", None):
                entry["id"] = tc_delta.id
            func = getattr(tc_delta, "function", None)
            if func is not None:
                if getattr(func, "name", None):
                    entry["name"] = func.name
                if getattr(func, "arguments", None):
                    entry["arguments"] += func.arguments

    def _finalize_tool_calls(self, acc: dict[int, dict[str, Any]]) -> list[ToolCall] | None:
        """Parse accumulated tool-call fragments into ToolCall instances.

        Raises:
            LLMResponseError: A tool call's concatenated `arguments` string
                is not valid JSON.
        """
        if not acc:
            return None
        tool_calls: list[ToolCall] = []
        for index in sorted(acc):
            entry = acc[index]
            raw_arguments = entry["arguments"]
            try:
                arguments = json.loads(raw_arguments) if raw_arguments else {}
            except json.JSONDecodeError as exc:
                raise LLMResponseError(
                    f"Malformed tool-call arguments JSON at index {index}: {exc}"
                ) from exc
            tool_calls.append(
                ToolCall(
                    id=entry["id"] or "",
                    name=entry["name"] or "",
                    arguments=arguments,
                )
            )
        return tool_calls

    async def generate(
        self, request: LLMRequest, *, stream: bool = False
    ) -> AsyncGenerator[LLMResponse, None]:
        """Generate via async OpenAI client. Yields partial or complete responses."""
        kwargs = self._build_kwargs(request, stream)

        if stream:
            response = await self._async_client.chat.completions.create(**kwargs)
            tool_call_acc: dict[int, dict[str, Any]] = {}
            reasoning_parts: list[str] = []
            finish_reason: str | None = None
            usage: UsageMetadata | None = None
            async for chunk in response:
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage:
                    usage = self._parse_usage(chunk_usage)
                if not chunk.choices:
                    # The final usage-only chunk (stream_options.include_usage)
                    # has no choices at all.
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                content = getattr(delta, "content", None)
                if content:
                    yield LLMResponse(content=content, partial=True)
                reasoning_piece = getattr(delta, "reasoning_content", None)
                if reasoning_piece:
                    reasoning_parts.append(reasoning_piece)
                tool_call_deltas = getattr(delta, "tool_calls", None)
                if tool_call_deltas:
                    self._accumulate_tool_call_deltas(tool_call_acc, tool_call_deltas)
                choice_finish_reason = getattr(choice, "finish_reason", None)
                if choice_finish_reason:
                    finish_reason = choice_finish_reason
            yield LLMResponse(
                content=None,
                tool_calls=self._finalize_tool_calls(tool_call_acc),
                reasoning_content="".join(reasoning_parts) or None,
                finish_reason=finish_reason,
                usage=usage,
                partial=False,
            )
        else:
            response = await self._async_client.chat.completions.create(**kwargs)
            message = response.choices[0].message
            yield LLMResponse(
                content=message.content or "",
                tool_calls=self._parse_tool_calls(message.tool_calls),
                reasoning_content=getattr(message, "reasoning_content", None),
                finish_reason=getattr(response.choices[0], "finish_reason", None),
                partial=False,
                usage=self._parse_usage(response.usage),
            )

    def generate_sync(
        self, request: LLMRequest, *, stream: bool = False
    ) -> Generator[LLMResponse, None, None]:
        """Generate via sync OpenAI client. Yields partial or complete responses."""
        kwargs = self._build_kwargs(request, stream)

        if stream:
            response = self._sync_client.chat.completions.create(**kwargs)
            tool_call_acc: dict[int, dict[str, Any]] = {}
            reasoning_parts: list[str] = []
            finish_reason: str | None = None
            usage: UsageMetadata | None = None
            for chunk in response:
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage:
                    usage = self._parse_usage(chunk_usage)
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                content = getattr(delta, "content", None)
                if content:
                    yield LLMResponse(content=content, partial=True)
                reasoning_piece = getattr(delta, "reasoning_content", None)
                if reasoning_piece:
                    reasoning_parts.append(reasoning_piece)
                tool_call_deltas = getattr(delta, "tool_calls", None)
                if tool_call_deltas:
                    self._accumulate_tool_call_deltas(tool_call_acc, tool_call_deltas)
                choice_finish_reason = getattr(choice, "finish_reason", None)
                if choice_finish_reason:
                    finish_reason = choice_finish_reason
            yield LLMResponse(
                content=None,
                tool_calls=self._finalize_tool_calls(tool_call_acc),
                reasoning_content="".join(reasoning_parts) or None,
                finish_reason=finish_reason,
                usage=usage,
                partial=False,
            )
        else:
            response = self._sync_client.chat.completions.create(**kwargs)
            message = response.choices[0].message
            yield LLMResponse(
                content=message.content or "",
                tool_calls=self._parse_tool_calls(message.tool_calls),
                reasoning_content=getattr(message, "reasoning_content", None),
                finish_reason=getattr(response.choices[0], "finish_reason", None),
                partial=False,
                usage=self._parse_usage(response.usage),
            )

    async def health_check(self) -> bool:
        """Check llama-server /health endpoint."""
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(f"{self._base_url}/health")
                return r.status_code == 200
        except Exception:
            return False
