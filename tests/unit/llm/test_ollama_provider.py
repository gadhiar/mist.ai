"""Tests for OllamaProvider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.llm.models import LLMRequest, LLMResponse
from backend.llm.ollama_provider import OllamaProvider

MODULE = "backend.llm.ollama_provider"


def _make_request(**overrides) -> LLMRequest:
    """Build a minimal LLMRequest with overridable fields."""
    defaults = {"messages": [{"role": "user", "content": "hello"}]}
    defaults.update(overrides)
    return LLMRequest(**defaults)


def _make_chat_response(content: str = "response text") -> MagicMock:
    """Build a mock matching ollama ChatResponse structure."""
    resp = MagicMock()
    resp.message.content = content
    return resp


# ---------------------------------------------------------------------------
# Sync batch
# ---------------------------------------------------------------------------


class TestGenerateSyncBatch:
    def test_returns_content(self):
        provider = OllamaProvider(base_url="http://localhost:11434", model="qwen2.5:7b")
        mock_client = MagicMock()
        mock_client.chat.return_value = _make_chat_response("hello world")

        with patch(f"{MODULE}.ollama.Client", return_value=mock_client):
            results = list(provider.generate_sync(_make_request(), stream=False))

        assert len(results) == 1
        assert results[0].content == "hello world"
        assert results[0].partial is False

    def test_forwards_options(self):
        provider = OllamaProvider(base_url="http://localhost:11434", model="qwen2.5:7b")
        mock_client = MagicMock()
        mock_client.chat.return_value = _make_chat_response("ok")

        request = _make_request(temperature=0.3, max_tokens=200, top_p=0.8)

        with patch(f"{MODULE}.ollama.Client", return_value=mock_client):
            list(provider.generate_sync(request, stream=False))

        call_kwargs = mock_client.chat.call_args.kwargs
        assert call_kwargs["options"]["temperature"] == 0.3
        assert call_kwargs["options"]["num_predict"] == 200
        assert call_kwargs["options"]["top_p"] == 0.8

    def test_forwards_model(self):
        provider = OllamaProvider(base_url="http://localhost:11434", model="qwen2.5:7b")
        mock_client = MagicMock()
        mock_client.chat.return_value = _make_chat_response("ok")

        with patch(f"{MODULE}.ollama.Client", return_value=mock_client):
            list(provider.generate_sync(_make_request(), stream=False))

        call_kwargs = mock_client.chat.call_args.kwargs
        assert call_kwargs["model"] == "qwen2.5:7b"

    def test_passes_base_url_to_client(self):
        provider = OllamaProvider(base_url="http://myhost:9999", model="m")
        mock_client = MagicMock()
        mock_client.chat.return_value = _make_chat_response("ok")

        with patch(f"{MODULE}.ollama.Client", return_value=mock_client) as mock_cls:
            list(provider.generate_sync(_make_request(), stream=False))

        mock_cls.assert_called_once_with(host="http://myhost:9999")

    def test_none_content_yields_empty_string(self):
        provider = OllamaProvider(base_url="http://localhost:11434", model="m")
        mock_client = MagicMock()
        mock_client.chat.return_value = _make_chat_response(None)

        with patch(f"{MODULE}.ollama.Client", return_value=mock_client):
            results = list(provider.generate_sync(_make_request(), stream=False))

        assert results[0].content == ""


# ---------------------------------------------------------------------------
# Sync streaming
# ---------------------------------------------------------------------------


class TestGenerateSyncStreaming:
    def test_yields_partial_chunks(self):
        provider = OllamaProvider(base_url="http://localhost:11434", model="m")
        chunks = [_make_chat_response("hello "), _make_chat_response("world")]
        mock_client = MagicMock()
        mock_client.chat.return_value = iter(chunks)

        with patch(f"{MODULE}.ollama.Client", return_value=mock_client):
            results = list(provider.generate_sync(_make_request(), stream=True))

        assert len(results) == 2
        assert results[0] == LLMResponse(content="hello ", partial=True)
        assert results[1] == LLMResponse(content="world", partial=True)

    def test_skips_empty_content_chunks(self):
        provider = OllamaProvider(base_url="http://localhost:11434", model="m")
        chunks = [_make_chat_response("hi"), _make_chat_response(""), _make_chat_response("there")]
        mock_client = MagicMock()
        mock_client.chat.return_value = iter(chunks)

        with patch(f"{MODULE}.ollama.Client", return_value=mock_client):
            results = list(provider.generate_sync(_make_request(), stream=True))

        assert len(results) == 2
        assert results[0].content == "hi"
        assert results[1].content == "there"

    def test_passes_stream_true(self):
        provider = OllamaProvider(base_url="http://localhost:11434", model="m")
        mock_client = MagicMock()
        mock_client.chat.return_value = iter([])

        with patch(f"{MODULE}.ollama.Client", return_value=mock_client):
            list(provider.generate_sync(_make_request(), stream=True))

        call_kwargs = mock_client.chat.call_args.kwargs
        assert call_kwargs["stream"] is True


# ---------------------------------------------------------------------------
# json_mode
# ---------------------------------------------------------------------------


class TestJsonMode:
    def test_sync_passes_format_json(self):
        provider = OllamaProvider(base_url="http://localhost:11434", model="m")
        mock_client = MagicMock()
        mock_client.chat.return_value = _make_chat_response('{"key": "val"}')

        with patch(f"{MODULE}.ollama.Client", return_value=mock_client):
            list(provider.generate_sync(_make_request(json_mode=True), stream=False))

        call_kwargs = mock_client.chat.call_args.kwargs
        assert call_kwargs["format"] == "json"

    def test_sync_omits_format_when_not_json_mode(self):
        provider = OllamaProvider(base_url="http://localhost:11434", model="m")
        mock_client = MagicMock()
        mock_client.chat.return_value = _make_chat_response("text")

        with patch(f"{MODULE}.ollama.Client", return_value=mock_client):
            list(provider.generate_sync(_make_request(json_mode=False), stream=False))

        call_kwargs = mock_client.chat.call_args.kwargs
        assert "format" not in call_kwargs

    @pytest.mark.asyncio
    async def test_async_passes_format_json(self):
        provider = OllamaProvider(base_url="http://localhost:11434", model="m")
        mock_client = AsyncMock()
        mock_client.chat.return_value = _make_chat_response('{"key": "val"}')

        with patch(f"{MODULE}.ollama.AsyncClient", return_value=mock_client):
            [r async for r in provider.generate(_make_request(json_mode=True))]

        call_kwargs = mock_client.chat.call_args.kwargs
        assert call_kwargs["format"] == "json"


# ---------------------------------------------------------------------------
# Async batch
# ---------------------------------------------------------------------------


class TestGenerateAsyncBatch:
    @pytest.mark.asyncio
    async def test_returns_content(self):
        provider = OllamaProvider(base_url="http://localhost:11434", model="qwen2.5:7b")
        mock_client = AsyncMock()
        mock_client.chat.return_value = _make_chat_response("async response")

        with patch(f"{MODULE}.ollama.AsyncClient", return_value=mock_client):
            results = [r async for r in provider.generate(_make_request(), stream=False)]

        assert len(results) == 1
        assert results[0].content == "async response"
        assert results[0].partial is False

    @pytest.mark.asyncio
    async def test_forwards_options(self):
        provider = OllamaProvider(base_url="http://localhost:11434", model="m")
        mock_client = AsyncMock()
        mock_client.chat.return_value = _make_chat_response("ok")

        request = _make_request(temperature=0.5, max_tokens=100, top_p=0.95)

        with patch(f"{MODULE}.ollama.AsyncClient", return_value=mock_client):
            [r async for r in provider.generate(request, stream=False)]

        call_kwargs = mock_client.chat.call_args.kwargs
        assert call_kwargs["options"]["temperature"] == 0.5
        assert call_kwargs["options"]["num_predict"] == 100
        assert call_kwargs["options"]["top_p"] == 0.95

    @pytest.mark.asyncio
    async def test_passes_base_url_to_async_client(self):
        provider = OllamaProvider(base_url="http://remote:11434", model="m")
        mock_client = AsyncMock()
        mock_client.chat.return_value = _make_chat_response("ok")

        with patch(f"{MODULE}.ollama.AsyncClient", return_value=mock_client) as mock_cls:
            [r async for r in provider.generate(_make_request(), stream=False)]

        mock_cls.assert_called_once_with(host="http://remote:11434")


# ---------------------------------------------------------------------------
# Async streaming
# ---------------------------------------------------------------------------


class TestGenerateAsyncStreaming:
    @pytest.mark.asyncio
    async def test_yields_partial_chunks(self):
        provider = OllamaProvider(base_url="http://localhost:11434", model="m")
        chunks = [_make_chat_response("one "), _make_chat_response("two")]

        async def _async_iter():
            for c in chunks:
                yield c

        mock_client = AsyncMock()
        mock_client.chat.return_value = _async_iter()

        with patch(f"{MODULE}.ollama.AsyncClient", return_value=mock_client):
            results = [r async for r in provider.generate(_make_request(), stream=True)]

        assert len(results) == 2
        assert results[0] == LLMResponse(content="one ", partial=True)
        assert results[1] == LLMResponse(content="two", partial=True)

    @pytest.mark.asyncio
    async def test_skips_empty_content(self):
        provider = OllamaProvider(base_url="http://localhost:11434", model="m")
        chunks = [_make_chat_response("a"), _make_chat_response(""), _make_chat_response("b")]

        async def _async_iter():
            for c in chunks:
                yield c

        mock_client = AsyncMock()
        mock_client.chat.return_value = _async_iter()

        with patch(f"{MODULE}.ollama.AsyncClient", return_value=mock_client):
            results = [r async for r in provider.generate(_make_request(), stream=True)]

        assert len(results) == 2
        assert results[0].content == "a"
        assert results[1].content == "b"


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_returns_true_when_ollama_available(self):
        provider = OllamaProvider(base_url="http://localhost:11434", model="m")
        mock_client = AsyncMock()
        mock_client.list.return_value = MagicMock()

        with patch(f"{MODULE}.ollama.AsyncClient", return_value=mock_client):
            assert await provider.health_check() is True

    @pytest.mark.asyncio
    async def test_returns_false_on_connection_error(self):
        provider = OllamaProvider(base_url="http://localhost:11434", model="m")
        mock_client = AsyncMock()
        mock_client.list.side_effect = ConnectionError("refused")

        with patch(f"{MODULE}.ollama.AsyncClient", return_value=mock_client):
            assert await provider.health_check() is False
