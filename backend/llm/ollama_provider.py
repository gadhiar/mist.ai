"""OllamaProvider -- fallback provider wrapping the ollama Python client.

Selected via LLM_BACKEND=ollama. Useful for quick model testing via
Ollama's automatic model management (pull, evict, keep-alive).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Generator

import ollama

from backend.llm.models import LLMRequest, LLMResponse
from backend.llm.provider import StreamingLLMProvider

logger = logging.getLogger(__name__)


class OllamaProvider(StreamingLLMProvider):
    """StreamingLLMProvider backed by the Ollama Python client."""

    def __init__(self, base_url: str, model: str) -> None:
        self.model = model
        self._base_url = base_url

    def _build_options(self, request: LLMRequest) -> dict:
        """Build Ollama options dict from LLMRequest."""
        return {
            "num_predict": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }

    async def generate(
        self, request: LLMRequest, *, stream: bool = False
    ) -> AsyncGenerator[LLMResponse, None]:
        """Generate a response via ollama AsyncClient."""
        client = ollama.AsyncClient(host=self._base_url)
        kwargs: dict = {
            "model": self.model,
            "messages": request.messages,
            "stream": stream,
            "options": self._build_options(request),
        }
        if request.json_mode:
            kwargs["format"] = "json"

        if stream:
            async for chunk in await client.chat(**kwargs):
                content = chunk.message.content
                if content:
                    yield LLMResponse(content=content, partial=True)
        else:
            response = await client.chat(**kwargs)
            yield LLMResponse(
                content=response.message.content or "",
                partial=False,
            )

    def generate_sync(
        self, request: LLMRequest, *, stream: bool = False
    ) -> Generator[LLMResponse, None, None]:
        """Generate a response via ollama sync Client."""
        client = ollama.Client(host=self._base_url)
        kwargs: dict = {
            "model": self.model,
            "messages": request.messages,
            "stream": stream,
            "options": self._build_options(request),
        }
        if request.json_mode:
            kwargs["format"] = "json"

        if stream:
            for chunk in client.chat(**kwargs):
                content = chunk.message.content
                if content:
                    yield LLMResponse(content=content, partial=True)
        else:
            response = client.chat(**kwargs)
            yield LLMResponse(
                content=response.message.content or "",
                partial=False,
            )

    async def health_check(self) -> bool:
        """Check Ollama availability via model list."""
        try:
            client = ollama.AsyncClient(host=self._base_url)
            await client.list()
            return True
        except Exception:
            return False
