"""Cancelling a consumer closes the HTTP stream through the whole provider chain.

The chain is built in production order (`backend/factories.py`,
`build_llm_provider`): the base provider is wrapped in
`InstrumentedStreamingLLMProvider` first and `AdaptiveThinkingProvider` last.
If any layer iterates its inner generator bare, a cancellation leaves the
inner generators open until the event loop's asyncgen finalizer runs them,
several loop ticks later, and the llama-server stream keeps generating.
"""

from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace

import pytest

from backend.debug_jsonl_logger import DebugJSONLLogger
from backend.llm.adaptive_thinking import AdaptiveThinkingProvider
from backend.llm.instrumented_provider import InstrumentedStreamingLLMProvider
from backend.llm.llama_server_provider import LlamaServerProvider
from backend.llm.models import LLMRequest


class FakeAsyncStream:
    """Stands in for `openai.AsyncStream`: async-iterable chunks plus async `close()`."""

    def __init__(self, contents: list[str]) -> None:
        self._chunks = [
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])
            for text in contents
        ]
        self.closed = False

    def __aiter__(self) -> FakeAsyncStream:
        return self

    async def __anext__(self) -> SimpleNamespace:
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)

    async def close(self) -> None:
        self.closed = True


def _production_chain(stream: FakeAsyncStream) -> AdaptiveThinkingProvider:
    base = LlamaServerProvider(base_url="http://localhost:8080", model="test-model")

    async def create(**kwargs: object) -> FakeAsyncStream:
        return stream

    base._async_client.chat.completions.create = create
    instrumented = InstrumentedStreamingLLMProvider(base, DebugJSONLLogger(None))
    return AdaptiveThinkingProvider(instrumented, budget_tokens=256)


@pytest.mark.asyncio
async def test_cancelled_consumer_has_closed_the_stream_when_await_task_raises():
    stream = FakeAsyncStream(["one", "two", "three"])
    chain = _production_chain(stream)
    first_token = asyncio.Event()
    never = asyncio.Event()

    async def consume() -> None:
        request = LLMRequest(messages=[{"role": "user", "content": "hi"}])
        async with contextlib.aclosing(chain.generate(request, stream=True)) as responses:
            async for _ in responses:
                first_token.set()
                await never.wait()

    task = asyncio.create_task(consume())
    await first_token.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # No `await asyncio.sleep(0)` between the cancel and this check: the close
    # must happen inside the cancelled task, not in a later finalizer tick.
    assert stream.closed is True
