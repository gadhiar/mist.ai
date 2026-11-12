"""Tests for LlamaServerProvider."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import DEFAULT_TIMEOUT, AsyncOpenAI
from pydantic import ValidationError

from backend.errors import LLMConnectionError, LLMResponseError
from backend.llm.llama_server_provider import LlamaServerProvider
from backend.llm.models import LLMRequest, ThinkingConfig, ToolCall, UsageMetadata

MODULE = "backend.llm.llama_server_provider"

# ---------------------------------------------------------------------------
# Helpers -- build mock OpenAI response objects
# ---------------------------------------------------------------------------


def _make_message(
    *,
    content: str = "hello",
    tool_calls: list | None = None,
) -> SimpleNamespace:
    """Build a mock ChatCompletionMessage."""
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _make_usage(
    *, prompt_tokens: int = 10, completion_tokens: int = 20, total_tokens: int = 30
) -> SimpleNamespace:
    """Build a mock CompletionUsage."""
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def _make_tool_call(
    *, tc_id: str = "call_1", name: str = "search", arguments: str = '{"q": "test"}'
) -> SimpleNamespace:
    """Build a mock ChatCompletionMessageToolCall."""
    return SimpleNamespace(
        id=tc_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _make_completion(
    *,
    content: str = "hello",
    tool_calls: list | None = None,
    usage: SimpleNamespace | None = None,
) -> SimpleNamespace:
    """Build a mock ChatCompletion (non-streaming)."""
    message = _make_message(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=usage or _make_usage(),
    )


def _make_chunk(*, content: str | None = None) -> SimpleNamespace:
    """Build a mock ChatCompletionChunk (streaming)."""
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content))],
    )


def _make_tool_call_delta(
    *,
    index: int,
    tc_id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> SimpleNamespace:
    """Build a mock streaming tool_call delta fragment."""
    function = None
    if name is not None or arguments is not None:
        function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=tc_id, function=function)


def _make_tool_call_chunk(*, deltas: list[SimpleNamespace]) -> SimpleNamespace:
    """Build a mock ChatCompletionChunk carrying tool_call delta fragments."""
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=deltas))],
    )


def _make_reasoning_chunk(*, reasoning_content: str) -> SimpleNamespace:
    """Build a mock ChatCompletionChunk carrying a reasoning_content delta."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=None, reasoning_content=reasoning_content)
            )
        ],
    )


async def _empty_async_stream():
    """An async generator that yields nothing (empty stream)."""
    return
    yield  # pragma: no cover -- makes this an async generator


def _default_request() -> LLMRequest:
    return LLMRequest(messages=[{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------------------
# Provider fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def provider() -> LlamaServerProvider:
    return LlamaServerProvider(base_url="http://localhost:8080", model="test-model")


# ---------------------------------------------------------------------------
# The client timeout
# ---------------------------------------------------------------------------


class TestClientTimeout:
    def test_a_timeout_reaches_both_clients_with_a_short_connect(self):
        provider = LlamaServerProvider(
            base_url="http://localhost:8080", model="test-model", timeout=900.0
        )

        expected = httpx.Timeout(900.0, connect=5.0)
        assert provider._async_client.timeout == expected
        assert provider._sync_client.timeout == expected

    def test_no_timeout_keeps_the_sdk_default(self, provider: LlamaServerProvider):
        assert provider._async_client.timeout == DEFAULT_TIMEOUT
        assert provider._sync_client.timeout == DEFAULT_TIMEOUT

    def test_the_timeout_is_keyword_only(self):
        with pytest.raises(TypeError):
            LlamaServerProvider("http://localhost:8080", "test-model", 900.0)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _build_kwargs
# ---------------------------------------------------------------------------


class TestBuildKwargs:
    def test_basic_kwargs(self, provider: LlamaServerProvider):
        request = _default_request()

        kwargs = provider._build_kwargs(request, stream=False)

        assert kwargs["model"] == "test-model"
        assert kwargs["messages"] == [{"role": "user", "content": "hi"}]
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 400
        assert kwargs["top_p"] == 0.9
        assert kwargs["stream"] is False
        assert "response_format" not in kwargs
        assert "tools" not in kwargs

    def test_no_thinking_no_response_schema_pins_exact_dict(self, provider: LlamaServerProvider):
        """thinking=None and response_schema=None must produce exactly today's
        (pre-T0) kwargs dict for a non-streaming request -- no extra_body,
        no response_format, no stream_options.
        """
        request = _default_request()

        kwargs = provider._build_kwargs(request, stream=False)

        assert kwargs == {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.7,
            "max_tokens": 400,
            "top_p": 0.9,
            "stream": False,
        }

    def test_json_mode_adds_response_format(self, provider: LlamaServerProvider):
        request = LLMRequest(
            messages=[{"role": "user", "content": "hi"}],
            json_mode=True,
        )

        kwargs = provider._build_kwargs(request, stream=False)

        assert kwargs["response_format"] == {"type": "json_object"}

    def test_tools_forwarded(self, provider: LlamaServerProvider):
        tools = [{"type": "function", "function": {"name": "search"}}]
        request = LLMRequest(
            messages=[{"role": "user", "content": "hi"}],
            tools=tools,
        )

        kwargs = provider._build_kwargs(request, stream=True)

        assert kwargs["tools"] == tools
        assert kwargs["stream"] is True


class TestBuildKwargsThinking:
    def test_budget_tokens_sets_reasoning_budget(self, provider: LlamaServerProvider):
        request = LLMRequest(
            messages=[{"role": "user", "content": "hi"}],
            thinking=ThinkingConfig(budget_tokens=256),
        )

        kwargs = provider._build_kwargs(request, stream=False)

        assert kwargs["extra_body"] == {"reasoning_budget_tokens": 256}

    def test_budget_tokens_none_omits_reasoning_budget_field(self, provider: LlamaServerProvider):
        request = LLMRequest(
            messages=[{"role": "user", "content": "hi"}],
            thinking=ThinkingConfig(budget_tokens=None, effort="low"),
        )

        kwargs = provider._build_kwargs(request, stream=False)

        assert "reasoning_budget_tokens" not in kwargs["extra_body"]

    def test_effort_only_sets_chat_template_kwargs(self, provider: LlamaServerProvider):
        request = LLMRequest(
            messages=[{"role": "user", "content": "hi"}],
            thinking=ThinkingConfig(effort="high"),
        )

        kwargs = provider._build_kwargs(request, stream=False)

        assert kwargs["extra_body"] == {"chat_template_kwargs": {"reasoning_effort": "high"}}

    def test_enabled_only_sets_chat_template_kwargs(self, provider: LlamaServerProvider):
        request = LLMRequest(
            messages=[{"role": "user", "content": "hi"}],
            thinking=ThinkingConfig(enabled=True),
        )

        kwargs = provider._build_kwargs(request, stream=False)

        assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}

    def test_effort_and_enabled_both_set(self, provider: LlamaServerProvider):
        request = LLMRequest(
            messages=[{"role": "user", "content": "hi"}],
            thinking=ThinkingConfig(effort="medium", enabled=False),
        )

        kwargs = provider._build_kwargs(request, stream=False)

        assert kwargs["extra_body"]["chat_template_kwargs"] == {
            "reasoning_effort": "medium",
            "enable_thinking": False,
        }

    def test_no_effort_no_enabled_omits_chat_template_kwargs_key(
        self, provider: LlamaServerProvider
    ):
        request = LLMRequest(
            messages=[{"role": "user", "content": "hi"}],
            thinking=ThinkingConfig(budget_tokens=10),
        )

        kwargs = provider._build_kwargs(request, stream=False)

        assert "chat_template_kwargs" not in kwargs["extra_body"]

    def test_negative_one_budget_tokens_raises(self):
        with pytest.raises(ValidationError):
            ThinkingConfig(budget_tokens=-1)


class TestBuildKwargsResponseSchema:
    def test_response_schema_sets_json_schema_response_format(self, provider: LlamaServerProvider):
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        request = LLMRequest(
            messages=[{"role": "user", "content": "hi"}],
            response_schema=schema,
        )

        kwargs = provider._build_kwargs(request, stream=False)

        assert kwargs["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": schema, "strict": True},
        }

    def test_response_schema_takes_precedence_over_json_mode(self, provider: LlamaServerProvider):
        schema = {"type": "object"}
        request = LLMRequest(
            messages=[{"role": "user", "content": "hi"}],
            json_mode=True,
            response_schema=schema,
        )

        kwargs = provider._build_kwargs(request, stream=False)

        assert kwargs["response_format"]["type"] == "json_schema"


class TestBuildKwargsStreamOptions:
    def test_stream_true_requests_usage(self, provider: LlamaServerProvider):
        request = _default_request()

        kwargs = provider._build_kwargs(request, stream=True)

        assert kwargs["stream_options"] == {"include_usage": True}

    def test_stream_false_omits_stream_options(self, provider: LlamaServerProvider):
        request = _default_request()

        kwargs = provider._build_kwargs(request, stream=False)

        assert "stream_options" not in kwargs


# ---------------------------------------------------------------------------
# Async generate -- batch (non-streaming)
# ---------------------------------------------------------------------------


class TestAsyncBatch:
    @pytest.mark.asyncio
    async def test_returns_content(self, provider: LlamaServerProvider):
        mock_create = AsyncMock(return_value=_make_completion(content="world"))
        provider._async_client.chat.completions.create = mock_create

        responses = [r async for r in provider.generate(_default_request())]

        assert len(responses) == 1
        assert responses[0].content == "world"
        assert responses[0].partial is False

    @pytest.mark.asyncio
    async def test_returns_tool_calls(self, provider: LlamaServerProvider):
        tc = _make_tool_call(tc_id="call_42", name="lookup", arguments='{"key": "val"}')
        mock_create = AsyncMock(return_value=_make_completion(content="", tool_calls=[tc]))
        provider._async_client.chat.completions.create = mock_create

        responses = [r async for r in provider.generate(_default_request())]

        assert len(responses) == 1
        assert responses[0].tool_calls is not None
        assert len(responses[0].tool_calls) == 1
        assert responses[0].tool_calls[0] == ToolCall(
            id="call_42", name="lookup", arguments={"key": "val"}
        )

    @pytest.mark.asyncio
    async def test_returns_usage(self, provider: LlamaServerProvider):
        usage = _make_usage(prompt_tokens=5, completion_tokens=15, total_tokens=20)
        mock_create = AsyncMock(return_value=_make_completion(content="ok", usage=usage))
        provider._async_client.chat.completions.create = mock_create

        responses = [r async for r in provider.generate(_default_request())]

        assert responses[0].usage == UsageMetadata(
            prompt_tokens=5, completion_tokens=15, total_tokens=20
        )

    @pytest.mark.asyncio
    async def test_none_content_becomes_empty_string(self, provider: LlamaServerProvider):
        mock_create = AsyncMock(return_value=_make_completion(content=None))
        provider._async_client.chat.completions.create = mock_create

        responses = [r async for r in provider.generate(_default_request())]

        assert responses[0].content == ""

    @pytest.mark.asyncio
    async def test_returns_reasoning_content_and_finish_reason(self, provider: LlamaServerProvider):
        message = SimpleNamespace(
            content="answer", tool_calls=None, reasoning_content="because reasons"
        )
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=_make_usage(),
        )
        mock_create = AsyncMock(return_value=completion)
        provider._async_client.chat.completions.create = mock_create

        responses = [r async for r in provider.generate(_default_request())]

        assert responses[0].reasoning_content == "because reasons"
        assert responses[0].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_missing_reasoning_content_defaults_to_none(self, provider: LlamaServerProvider):
        mock_create = AsyncMock(return_value=_make_completion(content="plain"))
        provider._async_client.chat.completions.create = mock_create

        responses = [r async for r in provider.generate(_default_request())]

        assert responses[0].reasoning_content is None
        assert responses[0].finish_reason is None

    @pytest.mark.asyncio
    async def test_json_mode_forwarded(self, provider: LlamaServerProvider):
        mock_create = AsyncMock(return_value=_make_completion(content="{}"))
        provider._async_client.chat.completions.create = mock_create
        request = LLMRequest(
            messages=[{"role": "user", "content": "hi"}],
            json_mode=True,
        )

        _ = [r async for r in provider.generate(request)]

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}


# ---------------------------------------------------------------------------
# Async generate -- streaming
# ---------------------------------------------------------------------------


class TestAsyncStreaming:
    @pytest.mark.asyncio
    async def test_yields_partial_chunks(self, provider: LlamaServerProvider):
        chunks = [
            _make_chunk(content="hell"),
            _make_chunk(content="o"),
            _make_chunk(content=" world"),
        ]

        async def mock_stream(**kwargs):
            for c in chunks:
                yield c

        mock_create = AsyncMock(side_effect=mock_stream)
        provider._async_client.chat.completions.create = mock_create

        responses = [r async for r in provider.generate(_default_request(), stream=True)]

        partials = responses[:-1]
        assert len(partials) == 3
        assert all(r.partial is True for r in partials)
        assert "".join(r.content for r in partials) == "hello world"
        # Exactly one final aggregated chunk, content None by contract.
        final = responses[-1]
        assert final.partial is False
        assert final.content is None

    @pytest.mark.asyncio
    async def test_skips_empty_content_chunks(self, provider: LlamaServerProvider):
        chunks = [
            _make_chunk(content=None),
            _make_chunk(content="data"),
            _make_chunk(content=None),
        ]

        async def mock_stream(**kwargs):
            for c in chunks:
                yield c

        mock_create = AsyncMock(side_effect=mock_stream)
        provider._async_client.chat.completions.create = mock_create

        responses = [r async for r in provider.generate(_default_request(), stream=True)]

        partials = [r for r in responses if r.partial]
        assert len(partials) == 1
        assert partials[0].content == "data"

    @pytest.mark.asyncio
    async def test_final_chunk_has_no_content_field(self, provider: LlamaServerProvider):
        """The final aggregated chunk carries content=None on purpose -- see
        the StreamingLLMProvider.generate docstring in provider.py.
        """
        chunks = [_make_chunk(content="hi")]

        async def mock_stream(**kwargs):
            for c in chunks:
                yield c

        mock_create = AsyncMock(side_effect=mock_stream)
        provider._async_client.chat.completions.create = mock_create

        responses = [r async for r in provider.generate(_default_request(), stream=True)]

        assert len(responses) == 2
        assert responses[0].content == "hi"
        assert responses[0].partial is True
        assert responses[1].content is None
        assert responses[1].partial is False

    @pytest.mark.asyncio
    async def test_reassembles_tool_call_split_across_fragments(
        self, provider: LlamaServerProvider
    ):
        chunks = [
            _make_tool_call_chunk(
                deltas=[_make_tool_call_delta(index=0, tc_id="call_1", name="search")]
            ),
            _make_tool_call_chunk(deltas=[_make_tool_call_delta(index=0, arguments='{"q": ')]),
            _make_tool_call_chunk(deltas=[_make_tool_call_delta(index=0, arguments='"test"}')]),
        ]

        async def mock_stream(**kwargs):
            for c in chunks:
                yield c

        mock_create = AsyncMock(side_effect=mock_stream)
        provider._async_client.chat.completions.create = mock_create

        responses = [r async for r in provider.generate(_default_request(), stream=True)]

        final = responses[-1]
        assert final.partial is False
        assert final.tool_calls == [ToolCall(id="call_1", name="search", arguments={"q": "test"})]

    @pytest.mark.asyncio
    async def test_malformed_tool_call_json_raises_llm_response_error(
        self, provider: LlamaServerProvider
    ):
        chunks = [
            _make_tool_call_chunk(
                deltas=[
                    _make_tool_call_delta(
                        index=0, tc_id="call_1", name="search", arguments="{not json"
                    )
                ]
            ),
        ]

        async def mock_stream(**kwargs):
            for c in chunks:
                yield c

        mock_create = AsyncMock(side_effect=mock_stream)
        provider._async_client.chat.completions.create = mock_create

        with pytest.raises(LLMResponseError):
            [r async for r in provider.generate(_default_request(), stream=True)]

    @pytest.mark.asyncio
    async def test_empty_choices_usage_chunk_does_not_raise(self, provider: LlamaServerProvider):
        usage_only_chunk = SimpleNamespace(choices=[], usage=_make_usage())
        chunks = [_make_chunk(content="hi"), usage_only_chunk]

        async def mock_stream(**kwargs):
            for c in chunks:
                yield c

        mock_create = AsyncMock(side_effect=mock_stream)
        provider._async_client.chat.completions.create = mock_create

        responses = [r async for r in provider.generate(_default_request(), stream=True)]

        final = responses[-1]
        assert final.partial is False
        assert final.usage == UsageMetadata(prompt_tokens=10, completion_tokens=20, total_tokens=30)

    @pytest.mark.asyncio
    async def test_stream_options_include_usage_requested(self, provider: LlamaServerProvider):
        mock_create = AsyncMock(side_effect=lambda **kwargs: _empty_async_stream())
        provider._async_client.chat.completions.create = mock_create

        _ = [r async for r in provider.generate(_default_request(), stream=True)]

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["stream_options"] == {"include_usage": True}

    @pytest.mark.asyncio
    async def test_reasoning_content_joined_on_final_chunk(self, provider: LlamaServerProvider):
        chunks = [
            _make_reasoning_chunk(reasoning_content="think "),
            _make_reasoning_chunk(reasoning_content="more"),
        ]

        async def mock_stream(**kwargs):
            for c in chunks:
                yield c

        mock_create = AsyncMock(side_effect=mock_stream)
        provider._async_client.chat.completions.create = mock_create

        responses = [r async for r in provider.generate(_default_request(), stream=True)]

        final = responses[-1]
        assert final.reasoning_content == "think more"


# ---------------------------------------------------------------------------
# Async generate -- the stream is closed however the call ends
# ---------------------------------------------------------------------------


class _RecordingStream:
    """A stand-in for the SDK's `AsyncStream`: chunks to iterate and a recorded `close`.

    After its chunks it waits on `hold` when given one, then raises `error`
    when given one.
    """

    def __init__(
        self,
        chunks: list[SimpleNamespace],
        *,
        hold: asyncio.Event | None = None,
        error: Exception | None = None,
    ) -> None:
        self._chunks = chunks
        self._hold = hold
        self._error = error
        self.closed = 0

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for chunk in self._chunks:
            yield chunk
        if self._hold is not None:
            await self._hold.wait()
        if self._error is not None:
            raise self._error

    async def close(self) -> None:
        self.closed += 1


def _serve(provider: LlamaServerProvider, stream: object) -> None:
    provider._async_client.chat.completions.create = AsyncMock(return_value=stream)


_FIRST_EVENT = (
    b'data: {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m", '
    b'"choices": [{"index": 0, "delta": {"role": "assistant", "content": "par"}, '
    b'"finish_reason": null}]}\n\n'
)


class _HeldEventStream(httpx.AsyncByteStream):
    """An HTTP response body: one server-sent event, then nothing until it is closed."""

    def __init__(self) -> None:
        self.closed = False
        self._hold = asyncio.Event()

    async def __aiter__(self):
        yield _FIRST_EVENT
        await self._hold.wait()

    async def aclose(self) -> None:
        self.closed = True


def _provider_over(body: httpx.AsyncByteStream) -> LlamaServerProvider:
    """A provider whose real SDK client reads `body` as the reply to any request."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body)

    provider = LlamaServerProvider(base_url="http://llama.test", model="test-model")
    provider._async_client = AsyncOpenAI(
        base_url="http://llama.test/v1",
        api_key="not-needed",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return provider


class TestAsyncStreamIsClosed:
    @pytest.mark.asyncio
    async def test_a_stream_read_to_its_end_is_closed(self, provider: LlamaServerProvider):
        stream = _RecordingStream([_make_chunk(content="hi")])
        _serve(provider, stream)

        responses = [r async for r in provider.generate(_default_request(), stream=True)]

        assert [r.partial for r in responses] == [True, False]
        assert stream.closed == 1

    @pytest.mark.asyncio
    async def test_a_consumer_that_leaves_early_closes_the_stream(
        self, provider: LlamaServerProvider
    ):
        stream = _RecordingStream([_make_chunk(content="a"), _make_chunk(content="b")])
        _serve(provider, stream)
        responses = provider.generate(_default_request(), stream=True)

        first = await responses.__anext__()
        await responses.aclose()

        assert first.content == "a"
        assert stream.closed == 1

    @pytest.mark.asyncio
    async def test_a_cancelled_consumer_closes_the_stream(self, provider: LlamaServerProvider):
        stream = _RecordingStream([_make_chunk(content="a")], hold=asyncio.Event())
        _serve(provider, stream)
        seen: list[str | None] = []

        async def consume() -> None:
            async for response in provider.generate(_default_request(), stream=True):
                seen.append(response.content)

        consumer = asyncio.create_task(consume())
        while not seen:
            await asyncio.sleep(0)
        await asyncio.sleep(0)

        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer

        assert seen == ["a"]
        assert stream.closed == 1

    @pytest.mark.asyncio
    async def test_a_transport_failure_mid_stream_is_a_connection_error(
        self, provider: LlamaServerProvider
    ):
        """The SDK wraps a failed request, not a failure while the body is read."""
        stream = _RecordingStream(
            [_make_chunk(content="a")], error=httpx.ReadError("connection reset")
        )
        _serve(provider, stream)

        with pytest.raises(LLMConnectionError, match="connection reset"):
            [r async for r in provider.generate(_default_request(), stream=True)]

        assert stream.closed == 1

    @pytest.mark.asyncio
    async def test_a_stream_that_is_an_async_generator_is_closed_too(
        self, provider: LlamaServerProvider
    ):
        """An async generator has `aclose` and no `close`; it is finished, not left to the GC."""
        finished: list[bool] = []

        async def mock_stream(**kwargs):
            try:
                yield _make_chunk(content="a")
                yield _make_chunk(content="b")
            finally:
                finished.append(True)

        provider._async_client.chat.completions.create = AsyncMock(side_effect=mock_stream)
        responses = provider.generate(_default_request(), stream=True)

        await responses.__anext__()
        await responses.aclose()

        assert finished == [True]

    @pytest.mark.asyncio
    async def test_leaving_early_closes_the_http_response_under_the_sdk_client(self):
        """Through the real SDK client: the response body is closed, so the connection is."""
        body = _HeldEventStream()
        responses = _provider_over(body).generate(_default_request(), stream=True)

        first = await responses.__anext__()
        await responses.aclose()

        assert (first.content, first.partial) == ("par", True)
        assert body.closed is True

    @pytest.mark.asyncio
    async def test_cancelling_closes_the_http_response_under_the_sdk_client(self):
        body = _HeldEventStream()
        provider = _provider_over(body)
        seen: list[str | None] = []

        async def consume() -> None:
            async for response in provider.generate(_default_request(), stream=True):
                seen.append(response.content)

        consumer = asyncio.create_task(consume())
        while not seen:
            await asyncio.sleep(0)
        await asyncio.sleep(0)

        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer

        assert seen == ["par"]
        assert body.closed is True


# ---------------------------------------------------------------------------
# Sync generate_sync -- batch
# ---------------------------------------------------------------------------


class TestSyncBatch:
    def test_returns_content(self, provider: LlamaServerProvider):
        provider._sync_client.chat.completions.create = MagicMock(
            return_value=_make_completion(content="sync reply")
        )

        responses = list(provider.generate_sync(_default_request()))

        assert len(responses) == 1
        assert responses[0].content == "sync reply"
        assert responses[0].partial is False

    def test_returns_tool_calls(self, provider: LlamaServerProvider):
        tc = _make_tool_call(tc_id="call_99", name="exec", arguments='{"cmd": "ls"}')
        provider._sync_client.chat.completions.create = MagicMock(
            return_value=_make_completion(content="", tool_calls=[tc])
        )

        responses = list(provider.generate_sync(_default_request()))

        assert responses[0].tool_calls is not None
        assert responses[0].tool_calls[0] == ToolCall(
            id="call_99", name="exec", arguments={"cmd": "ls"}
        )

    def test_returns_usage(self, provider: LlamaServerProvider):
        usage = _make_usage(prompt_tokens=8, completion_tokens=12, total_tokens=20)
        provider._sync_client.chat.completions.create = MagicMock(
            return_value=_make_completion(content="ok", usage=usage)
        )

        responses = list(provider.generate_sync(_default_request()))

        assert responses[0].usage == UsageMetadata(
            prompt_tokens=8, completion_tokens=12, total_tokens=20
        )

    def test_returns_reasoning_content_and_finish_reason(self, provider: LlamaServerProvider):
        message = SimpleNamespace(
            content="answer", tool_calls=None, reasoning_content="because reasons"
        )
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=_make_usage(),
        )
        provider._sync_client.chat.completions.create = MagicMock(return_value=completion)

        responses = list(provider.generate_sync(_default_request()))

        assert responses[0].reasoning_content == "because reasons"
        assert responses[0].finish_reason == "stop"


# ---------------------------------------------------------------------------
# Sync generate_sync -- streaming
# ---------------------------------------------------------------------------


class TestSyncStreaming:
    def test_yields_partial_chunks(self, provider: LlamaServerProvider):
        chunks = [
            _make_chunk(content="a"),
            _make_chunk(content="b"),
            _make_chunk(content="c"),
        ]
        provider._sync_client.chat.completions.create = MagicMock(return_value=iter(chunks))

        responses = list(provider.generate_sync(_default_request(), stream=True))

        partials = responses[:-1]
        assert len(partials) == 3
        assert all(r.partial is True for r in partials)
        assert "".join(r.content for r in partials) == "abc"
        final = responses[-1]
        assert final.partial is False
        assert final.content is None

    def test_skips_empty_content_chunks(self, provider: LlamaServerProvider):
        chunks = [
            _make_chunk(content=None),
            _make_chunk(content="x"),
        ]
        provider._sync_client.chat.completions.create = MagicMock(return_value=iter(chunks))

        responses = list(provider.generate_sync(_default_request(), stream=True))

        partials = [r for r in responses if r.partial]
        assert len(partials) == 1
        assert partials[0].content == "x"

    def test_reassembles_tool_call_split_across_fragments(self, provider: LlamaServerProvider):
        chunks = [
            _make_tool_call_chunk(
                deltas=[_make_tool_call_delta(index=0, tc_id="call_1", name="search")]
            ),
            _make_tool_call_chunk(deltas=[_make_tool_call_delta(index=0, arguments='{"q": ')]),
            _make_tool_call_chunk(deltas=[_make_tool_call_delta(index=0, arguments='"test"}')]),
        ]
        provider._sync_client.chat.completions.create = MagicMock(return_value=iter(chunks))

        responses = list(provider.generate_sync(_default_request(), stream=True))

        final = responses[-1]
        assert final.partial is False
        assert final.tool_calls == [ToolCall(id="call_1", name="search", arguments={"q": "test"})]

    def test_malformed_tool_call_json_raises_llm_response_error(
        self, provider: LlamaServerProvider
    ):
        chunks = [
            _make_tool_call_chunk(
                deltas=[
                    _make_tool_call_delta(
                        index=0, tc_id="call_1", name="search", arguments="{not json"
                    )
                ]
            ),
        ]
        provider._sync_client.chat.completions.create = MagicMock(return_value=iter(chunks))

        with pytest.raises(LLMResponseError):
            list(provider.generate_sync(_default_request(), stream=True))

    def test_empty_choices_usage_chunk_does_not_raise(self, provider: LlamaServerProvider):
        usage_only_chunk = SimpleNamespace(choices=[], usage=_make_usage())
        chunks = [_make_chunk(content="hi"), usage_only_chunk]
        provider._sync_client.chat.completions.create = MagicMock(return_value=iter(chunks))

        responses = list(provider.generate_sync(_default_request(), stream=True))

        final = responses[-1]
        assert final.partial is False
        assert final.usage == UsageMetadata(prompt_tokens=10, completion_tokens=20, total_tokens=30)

    def test_stream_options_include_usage_requested(self, provider: LlamaServerProvider):
        provider._sync_client.chat.completions.create = MagicMock(return_value=iter([]))

        list(provider.generate_sync(_default_request(), stream=True))

        call_kwargs = provider._sync_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["stream_options"] == {"include_usage": True}


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_healthy(self, provider: LlamaServerProvider):
        mock_response = MagicMock(status_code=200)
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch(f"{MODULE}.httpx.AsyncClient", return_value=mock_client):
            result = await provider.health_check()

        assert result is True
        mock_client.get.assert_called_once_with("http://localhost:8080/health")

    @pytest.mark.asyncio
    async def test_unhealthy_status(self, provider: LlamaServerProvider):
        mock_response = MagicMock(status_code=503)
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch(f"{MODULE}.httpx.AsyncClient", return_value=mock_client):
            result = await provider.health_check()

        assert result is False

    @pytest.mark.asyncio
    async def test_connection_error_returns_false(self, provider: LlamaServerProvider):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=ConnectionError("refused"))

        with patch(f"{MODULE}.httpx.AsyncClient", return_value=mock_client):
            result = await provider.health_check()

        assert result is False


# ---------------------------------------------------------------------------
# server_context_size
# ---------------------------------------------------------------------------


def _mock_async_client(*, get_return=None, get_side_effect=None) -> AsyncMock:
    """Build an AsyncMock standing in for httpx.AsyncClient's `async with` usage."""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    if get_side_effect is not None:
        mock_client.get = AsyncMock(side_effect=get_side_effect)
    else:
        mock_client.get = AsyncMock(return_value=get_return)
    return mock_client


class TestServerContextSize:
    """MIS-171 T4 (A): GET /props -> default_generation_settings.n_ctx."""

    @pytest.mark.asyncio
    async def test_parses_n_ctx_from_default_generation_settings(
        self, provider: LlamaServerProvider
    ):
        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = {"default_generation_settings": {"n_ctx": 32768}}
        mock_client = _mock_async_client(get_return=mock_response)

        with patch(f"{MODULE}.httpx.AsyncClient", return_value=mock_client):
            result = await provider.server_context_size()

        assert result == 32768
        mock_client.get.assert_called_once_with("http://localhost:8080/props")

    @pytest.mark.asyncio
    async def test_non_200_status_returns_none(self, provider: LlamaServerProvider):
        mock_response = MagicMock(status_code=503)
        mock_client = _mock_async_client(get_return=mock_response)

        with patch(f"{MODULE}.httpx.AsyncClient", return_value=mock_client):
            result = await provider.server_context_size()

        assert result is None

    @pytest.mark.asyncio
    async def test_malformed_json_body_returns_none(self, provider: LlamaServerProvider):
        mock_response = MagicMock(status_code=200)
        mock_response.json.side_effect = ValueError("not json")
        mock_client = _mock_async_client(get_return=mock_response)

        with patch(f"{MODULE}.httpx.AsyncClient", return_value=mock_client):
            result = await provider.server_context_size()

        assert result is None

    @pytest.mark.asyncio
    async def test_non_dict_json_body_returns_none(self, provider: LlamaServerProvider):
        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = ["not", "a", "dict"]
        mock_client = _mock_async_client(get_return=mock_response)

        with patch(f"{MODULE}.httpx.AsyncClient", return_value=mock_client):
            result = await provider.server_context_size()

        assert result is None

    @pytest.mark.asyncio
    async def test_missing_n_ctx_returns_none(self, provider: LlamaServerProvider):
        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = {"default_generation_settings": {}}
        mock_client = _mock_async_client(get_return=mock_response)

        with patch(f"{MODULE}.httpx.AsyncClient", return_value=mock_client):
            result = await provider.server_context_size()

        assert result is None

    @pytest.mark.asyncio
    async def test_non_int_n_ctx_returns_none(self, provider: LlamaServerProvider):
        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = {"default_generation_settings": {"n_ctx": "not-a-number"}}
        mock_client = _mock_async_client(get_return=mock_response)

        with patch(f"{MODULE}.httpx.AsyncClient", return_value=mock_client):
            result = await provider.server_context_size()

        assert result is None

    @pytest.mark.asyncio
    async def test_connection_error_returns_none(self, provider: LlamaServerProvider):
        mock_client = _mock_async_client(get_side_effect=httpx.ConnectError("refused"))

        with patch(f"{MODULE}.httpx.AsyncClient", return_value=mock_client):
            result = await provider.server_context_size()

        assert result is None
