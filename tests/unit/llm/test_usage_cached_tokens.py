"""`LlamaServerProvider` reports how many prompt tokens llama-server took from its cache.

llama-server can report the count two ways: `usage.prompt_tokens_details.cached_tokens`
(the OpenAI field) and `timings.cache_n`, a field the OpenAI schema does not name, which
the client keeps on the parsed model as an extra. The chunks below are built through the
client's own model classes so the tests see what `generate` sees at runtime.
"""

from __future__ import annotations

from typing import Any

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from backend.llm.llama_server_provider import LlamaServerProvider
from backend.llm.models import LLMRequest, UsageMetadata


def _chunk(
    *,
    content: str | None = None,
    finish_reason: str | None = None,
    usage: dict[str, Any] | None = None,
    timings: dict[str, Any] | None = None,
    no_choices: bool = False,
) -> ChatCompletionChunk:
    payload: dict[str, Any] = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "test-model",
        "choices": (
            []
            if no_choices
            else [{"index": 0, "delta": {"content": content}, "finish_reason": finish_reason}]
        ),
    }
    if usage is not None:
        payload["usage"] = usage
    if timings is not None:
        payload["timings"] = timings
    return ChatCompletionChunk.model_validate(payload)


def _usage(cached: int | None = None) -> dict[str, Any]:
    usage: dict[str, Any] = {"prompt_tokens": 1200, "completion_tokens": 30, "total_tokens": 1230}
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    return usage


_TIMINGS = {"cache_n": 850, "prompt_n": 350, "predicted_n": 30}


class _Stream:
    def __init__(self, chunks: list[ChatCompletionChunk]) -> None:
        self._chunks = list(chunks)

    def __aiter__(self) -> _Stream:
        return self

    async def __anext__(self) -> ChatCompletionChunk:
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)

    async def close(self) -> None:
        return None


def _provider(response: object) -> LlamaServerProvider:
    provider = LlamaServerProvider(base_url="http://localhost:8080", model="test-model")

    async def create(**kwargs: object) -> object:
        return response

    provider._async_client.chat.completions.create = create
    return provider


async def _final_usage(chunks: list[ChatCompletionChunk]) -> UsageMetadata | None:
    provider = _provider(_Stream(chunks))
    request = LLMRequest(messages=[{"role": "user", "content": "hi"}])
    responses = [r async for r in provider.generate(request, stream=True)]
    assert responses[-1].partial is False
    return responses[-1].usage


def test_usage_metadata_defaults_to_no_cached_count():
    assert UsageMetadata(prompt_tokens=10).cached_prompt_tokens is None


@pytest.mark.asyncio
async def test_streamed_usage_details_give_the_cached_count():
    usage = await _final_usage(
        [
            _chunk(content="Hello"),
            _chunk(finish_reason="stop"),
            _chunk(no_choices=True, usage=_usage(cached=900)),
        ]
    )
    assert usage is not None
    assert usage.prompt_tokens == 1200
    assert usage.cached_prompt_tokens == 900


@pytest.mark.asyncio
async def test_streamed_timings_fill_the_count_when_usage_has_none():
    usage = await _final_usage(
        [
            _chunk(content="Hello"),
            _chunk(finish_reason="stop", timings=_TIMINGS),
            _chunk(no_choices=True, usage=_usage()),
        ]
    )
    assert usage is not None
    assert usage.prompt_tokens == 1200
    assert usage.cached_prompt_tokens == 850


@pytest.mark.asyncio
async def test_streamed_usage_details_win_over_timings():
    usage = await _final_usage(
        [
            _chunk(content="Hello"),
            _chunk(no_choices=True, usage=_usage(cached=900), timings=_TIMINGS),
        ]
    )
    assert usage is not None
    assert usage.cached_prompt_tokens == 900


@pytest.mark.asyncio
async def test_streamed_count_is_none_with_neither():
    usage = await _final_usage(
        [
            _chunk(content="Hello"),
            _chunk(finish_reason="stop"),
            _chunk(no_choices=True, usage=_usage()),
        ]
    )
    assert usage is not None
    assert usage.prompt_tokens == 1200
    assert usage.cached_prompt_tokens is None


def _completion(*, usage: dict[str, Any], timings: dict[str, Any] | None) -> ChatCompletion:
    payload: dict[str, Any] = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
    }
    if timings is not None:
        payload["timings"] = timings
    return ChatCompletion.model_validate(payload)


async def _completion_usage(completion: ChatCompletion) -> UsageMetadata | None:
    provider = _provider(completion)
    request = LLMRequest(messages=[{"role": "user", "content": "hi"}])
    responses = [r async for r in provider.generate(request, stream=False)]
    return responses[-1].usage


@pytest.mark.asyncio
async def test_non_streamed_usage_details_give_the_cached_count():
    usage = await _completion_usage(_completion(usage=_usage(cached=640), timings=None))
    assert usage is not None
    assert usage.cached_prompt_tokens == 640


@pytest.mark.asyncio
async def test_non_streamed_timings_fill_the_count_when_usage_has_none():
    usage = await _completion_usage(_completion(usage=_usage(), timings=_TIMINGS))
    assert usage is not None
    assert usage.cached_prompt_tokens == 850


@pytest.mark.asyncio
async def test_non_streamed_count_is_none_with_neither():
    usage = await _completion_usage(_completion(usage=_usage(), timings=None))
    assert usage is not None
    assert usage.cached_prompt_tokens is None


def _sync_usage(response: object, *, stream: bool) -> UsageMetadata | None:
    provider = LlamaServerProvider(base_url="http://localhost:8080", model="test-model")
    provider._sync_client.chat.completions.create = lambda **kwargs: response
    request = LLMRequest(messages=[{"role": "user", "content": "hi"}])
    return list(provider.generate_sync(request, stream=stream))[-1].usage


def test_sync_stream_reads_both_shapes():
    by_details = _sync_usage(
        iter([_chunk(content="Hi"), _chunk(no_choices=True, usage=_usage(cached=900))]),
        stream=True,
    )
    by_timings = _sync_usage(
        iter(
            [
                _chunk(finish_reason="stop", timings=_TIMINGS),
                _chunk(no_choices=True, usage=_usage()),
            ]
        ),
        stream=True,
    )
    neither = _sync_usage(iter([_chunk(no_choices=True, usage=_usage())]), stream=True)

    assert by_details is not None and by_details.cached_prompt_tokens == 900
    assert by_timings is not None and by_timings.cached_prompt_tokens == 850
    assert neither is not None and neither.cached_prompt_tokens is None


def test_sync_completion_reads_timings():
    usage = _sync_usage(_completion(usage=_usage(), timings=_TIMINGS), stream=False)
    assert usage is not None
    assert usage.cached_prompt_tokens == 850
