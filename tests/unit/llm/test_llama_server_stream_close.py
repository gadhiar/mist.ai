"""`LlamaServerProvider.generate` closes its HTTP stream when its consumer stops.

A consumer that stops early (a cancelled turn, a `break`, an explicit
`aclose()`) must release the llama-server connection at once; otherwise the
server keeps generating tokens nobody reads until the event loop's asyncgen
finalizer gets round to the generator.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest

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


def _provider_streaming(stream: FakeAsyncStream) -> LlamaServerProvider:
    provider = LlamaServerProvider(base_url="http://localhost:8080", model="test-model")

    async def create(**kwargs: object) -> FakeAsyncStream:
        return stream

    provider._async_client.chat.completions.create = create
    return provider


def _request() -> LLMRequest:
    return LLMRequest(messages=[{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_closing_generator_mid_stream_closes_the_http_stream():
    stream = FakeAsyncStream(["one", "two", "three"])
    provider = _provider_streaming(stream)
    generator = provider.generate(_request(), stream=True)

    first = await generator.__anext__()
    await generator.aclose()

    assert first.content == "one"
    assert stream.closed is True


@pytest.mark.asyncio
async def test_exhausted_generator_closes_the_http_stream():
    stream = FakeAsyncStream(["one"])
    provider = _provider_streaming(stream)

    async with contextlib.aclosing(provider.generate(_request(), stream=True)) as generator:
        responses = [response async for response in generator]

    assert [r.partial for r in responses] == [True, False]
    assert stream.closed is True
