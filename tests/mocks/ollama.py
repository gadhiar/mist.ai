"""Test double for LLM (Ollama) operations.

FakeLLM satisfies the LLMProvider protocol from backend.interfaces.
Configure responses by pattern-matching on prompt content.
"""

from __future__ import annotations


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
        self.calls: list[tuple[list, dict]] = []

    async def ainvoke(self, messages, **kwargs):
        """Return configured response based on prompt content."""
        self.calls.append((messages, kwargs))
        prompt = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
        for pattern, response in self._responses.items():
            if pattern in prompt:
                return _make_ai_message(response)
        return _make_ai_message(self._default)

    def assert_called(self):
        """Assert the LLM was invoked at least once."""
        if not self.calls:
            raise AssertionError("FakeLLM was never called")

    def assert_not_called(self):
        """Assert the LLM was never invoked."""
        if self.calls:
            raise AssertionError(f"FakeLLM was called {len(self.calls)} times")


def _make_ai_message(content: str):
    """Build a LangChain-compatible AIMessage."""
    from langchain_core.messages import AIMessage

    return AIMessage(content=content)
