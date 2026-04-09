"""Test double for LLM operations.

FakeLLM satisfies the LLMProvider protocol from backend.interfaces.
Configure responses by pattern-matching on prompt content.
"""

from __future__ import annotations

from backend.llm.models import LLMRequest, LLMResponse


class FakeLLM:
    """Configurable LLM test double. Satisfies LLMProvider protocol."""

    def __init__(
        self,
        *,
        responses: dict[str, str] | None = None,
        default_response: str | None = None,
    ):
        self._responses = responses or {}
        self._default = default_response or '{"entities": [], "relationships": []}'
        self.calls: list[LLMRequest] = []

    async def invoke(self, request: LLMRequest) -> LLMResponse:
        """Return configured response based on last message content."""
        self.calls.append(request)
        prompt = str(request.messages[-1].get("content", ""))
        for pattern, response in self._responses.items():
            if pattern in prompt:
                return LLMResponse(content=response, partial=False)
        return LLMResponse(content=self._default, partial=False)

    def assert_called(self):
        """Assert the LLM was invoked at least once."""
        if not self.calls:
            raise AssertionError("FakeLLM was never called")

    def assert_not_called(self):
        """Assert the LLM was never invoked."""
        if self.calls:
            raise AssertionError(f"FakeLLM was called {len(self.calls)} times")
