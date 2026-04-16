"""Abstract base class for LLM inference providers.

StreamingLLMProvider defines the contract for all LLM backends.
Implementations must provide both async (generate) and sync
(generate_sync) streaming methods. Convenience methods invoke()
and invoke_sync() handle single-shot non-streaming calls.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Generator

from backend.llm.models import LLMRequest, LLMResponse


class StreamingLLMProvider(ABC):
    """Contract for LLM inference -- streaming, batch, and tool-calling.

    Subclasses implement generate() (async) and generate_sync() (sync).
    The async path is used by the conversation handler and extraction
    pipeline. The sync path is used by the voice pipeline which runs
    LLM streaming in a dedicated thread.
    """

    model: str

    @abstractmethod
    async def generate(
        self, request: LLMRequest, *, stream: bool = False
    ) -> AsyncGenerator[LLMResponse, None]:
        """Generate a response (async).

        When stream=True, yields partial LLMResponse chunks (partial=True)
        followed by a final chunk (partial=False) with aggregated content.

        When stream=False, yields a single LLMResponse (partial=False).
        """
        ...

    @abstractmethod
    def generate_sync(
        self, request: LLMRequest, *, stream: bool = False
    ) -> Generator[LLMResponse, None, None]:
        """Generate a response (sync).

        Same contract as generate() but synchronous. Used by the voice
        pipeline which runs LLM streaming in a worker thread.
        """
        ...

    async def invoke(self, request: LLMRequest) -> LLMResponse:
        """Non-streaming async convenience. Returns the complete response."""
        async for response in self.generate(request, stream=False):
            if not response.partial:
                return response
        raise RuntimeError("generate() yielded no non-partial response")

    def invoke_sync(self, request: LLMRequest) -> LLMResponse:
        """Non-streaming sync convenience. Returns the complete response."""
        for response in self.generate_sync(request, stream=False):
            if not response.partial:
                return response
        raise RuntimeError("generate_sync() yielded no non-partial response")

    async def health_check(self) -> bool:
        """Check if the LLM backend is reachable. Default returns True."""
        return True
