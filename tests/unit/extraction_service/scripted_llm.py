"""A base for providers scripted in Python: one reply per call, streamed or not.

The engine sends its Stage 2 calls as streaming requests and the others not,
so a fake provider has to answer both ways. A subclass states `reply`, the
whole answer to one request, and this base serves it in either shape.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import AsyncGenerator, Generator

from backend.llm.models import LLMRequest, LLMResponse
from backend.llm.provider import StreamingLLMProvider


class ScriptedProvider(StreamingLLMProvider):
    """Answers every call from `reply`, in the shape the provider contract gives it.

    A streamed call yields the reply's content as one partial chunk, then the
    final chunk with the rest and no content, as
    `StreamingLLMProvider.generate` promises. A call that is not streamed
    yields the reply whole.
    """

    model = "scripted"

    @abstractmethod
    async def reply(self, request: LLMRequest) -> LLMResponse:
        """The whole answer to `request`."""

    @property
    def streamed(self) -> list[bool]:
        """Per call, in call order: whether it was a streaming request."""
        return self.__dict__.setdefault("_streamed", [])

    async def generate(
        self, request: LLMRequest, *, stream: bool = False
    ) -> AsyncGenerator[LLMResponse, None]:
        self.streamed.append(stream)
        response = await self.reply(request)
        if not stream:
            yield response
            return
        if response.content:
            yield LLMResponse(content=response.content, partial=True)
        yield response.model_copy(update={"content": None})

    def generate_sync(
        self, request: LLMRequest, *, stream: bool = False
    ) -> Generator[LLMResponse, None, None]:
        raise NotImplementedError("the extraction service makes no synchronous call")
