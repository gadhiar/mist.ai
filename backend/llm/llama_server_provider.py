"""LlamaServerProvider -- wraps llama-server's OpenAI-compatible API.

Uses the openai Python package (both async and sync clients) to
communicate with llama-server. The sync path is used by the voice
pipeline; the async path by conversation handling and extraction.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator, Generator

import httpx
from openai import AsyncOpenAI, OpenAI

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
        if request.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if request.tools:
            kwargs["tools"] = request.tools
        return kwargs

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

    async def generate(
        self, request: LLMRequest, *, stream: bool = False
    ) -> AsyncGenerator[LLMResponse, None]:
        """Generate via async OpenAI client. Yields partial or complete responses."""
        kwargs = self._build_kwargs(request, stream)

        if stream:
            response = await self._async_client.chat.completions.create(**kwargs)
            async for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield LLMResponse(content=delta.content, partial=True)
        else:
            response = await self._async_client.chat.completions.create(**kwargs)
            message = response.choices[0].message
            yield LLMResponse(
                content=message.content or "",
                tool_calls=self._parse_tool_calls(message.tool_calls),
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
            for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield LLMResponse(content=delta.content, partial=True)
        else:
            response = self._sync_client.chat.completions.create(**kwargs)
            message = response.choices[0].message
            yield LLMResponse(
                content=message.content or "",
                tool_calls=self._parse_tool_calls(message.tool_calls),
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
