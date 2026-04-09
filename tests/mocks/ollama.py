"""Test double for LLM operations.

FakeLLM satisfies the StreamingLLMProvider ABC from backend.llm.
Configure responses by pattern-matching on prompt content.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator

from backend.llm.models import LLMRequest, LLMResponse
from backend.llm.provider import StreamingLLMProvider


class FakeLLM(StreamingLLMProvider):
    """Configurable LLM test double. Satisfies StreamingLLMProvider ABC."""

    model: str = "fake-model"

    def __init__(
        self,
        *,
        responses: dict[str, str] | None = None,
        default_response: str | None = None,
        streaming_chunks: list[str] | None = None,
    ):
        self._responses = responses or {}
        self._default = default_response or '{"entities": [], "relationships": []}'
        self._streaming_chunks = streaming_chunks
        self.calls: list[LLMRequest] = []

    def _resolve(self, request: LLMRequest) -> str:
        """Pick the response string for a request."""
        prompt = str(request.messages[-1].get("content", ""))
        for pattern, response in self._responses.items():
            if pattern in prompt:
                return response
        return self._default

    async def generate(
        self, request: LLMRequest, *, stream: bool = False
    ) -> AsyncGenerator[LLMResponse, None]:
        """Async generate -- yields partial chunks when streaming."""
        self.calls.append(request)
        if stream and self._streaming_chunks is not None:
            for chunk in self._streaming_chunks:
                yield LLMResponse(content=chunk, partial=True)
        else:
            yield LLMResponse(content=self._resolve(request), partial=False)

    def generate_sync(
        self, request: LLMRequest, *, stream: bool = False
    ) -> Generator[LLMResponse, None, None]:
        """Sync generate -- yields partial chunks when streaming."""
        self.calls.append(request)
        if stream and self._streaming_chunks is not None:
            for chunk in self._streaming_chunks:
                yield LLMResponse(content=chunk, partial=True)
        else:
            yield LLMResponse(content=self._resolve(request), partial=False)

    async def invoke(self, request: LLMRequest) -> LLMResponse:
        """Return configured response based on last message content."""
        self.calls.append(request)
        return LLMResponse(content=self._resolve(request), partial=False)

    def assert_called(self):
        """Assert the LLM was invoked at least once."""
        if not self.calls:
            raise AssertionError("FakeLLM was never called")

    def assert_not_called(self):
        """Assert the LLM was never invoked."""
        if self.calls:
            raise AssertionError(f"FakeLLM was called {len(self.calls)} times")
