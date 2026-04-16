"""Tests for LlamaServerProvider."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.llm.llama_server_provider import LlamaServerProvider
from backend.llm.models import LLMRequest, ToolCall, UsageMetadata

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


def _default_request() -> LLMRequest:
    return LLMRequest(messages=[{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------------------
# Provider fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def provider() -> LlamaServerProvider:
    return LlamaServerProvider(base_url="http://localhost:8080", model="test-model")


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

        assert len(responses) == 3
        assert all(r.partial is True for r in responses)
        assert "".join(r.content for r in responses) == "hello world"

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

        assert len(responses) == 1
        assert responses[0].content == "data"


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

        assert len(responses) == 3
        assert all(r.partial is True for r in responses)
        assert "".join(r.content for r in responses) == "abc"

    def test_skips_empty_content_chunks(self, provider: LlamaServerProvider):
        chunks = [
            _make_chunk(content=None),
            _make_chunk(content="x"),
        ]
        provider._sync_client.chat.completions.create = MagicMock(return_value=iter(chunks))

        responses = list(provider.generate_sync(_default_request(), stream=True))

        assert len(responses) == 1
        assert responses[0].content == "x"


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
