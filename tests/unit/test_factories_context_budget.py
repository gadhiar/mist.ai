"""Unit tests for MIS-171 T4: factory-level context-budget window resolution
and the AdaptiveThinkingProvider wiring in build_llm_provider.

Dependency notes (mirrors tests/unit/test_factories_phase3.py):
- TestResolveContextBudgetWindow and TestBuildLlmProviderWrapping are
  platform-neutral: they import backend.factories but only exercise
  resolve_context_budget_window / build_llm_provider, neither of which
  builds an EmbeddingGenerator.
- TestBuildConversationHandlerContextWindowWiring calls the real
  build_conversation_handler and is marked @requires_sentence_transformers,
  injecting a GraphStore wrapping FakeNeo4jConnection (+ FakeEmbeddingGenerator),
  a FakeVectorStore, and a FakeLLM so it never opens a real Neo4j connection
  or builds a real LanceDB vector store.
"""

from __future__ import annotations

import logging

import httpx
import pytest

from backend.knowledge.config import ContextBudgetConfig, LLMConfig
from tests.mocks.config import build_test_config

# ---------------------------------------------------------------------------
# Platform-availability marker (mirrors test_factories_phase3.py)
# ---------------------------------------------------------------------------

_SENTENCE_TRANSFORMERS_AVAILABLE = False
try:
    import sentence_transformers as _st  # noqa: F401

    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

requires_sentence_transformers = pytest.mark.skipif(
    not _SENTENCE_TRANSFORMERS_AVAILABLE,
    reason="sentence_transformers not available on this platform",
)


def _mock_sync_client_returning(response: httpx.Response):
    """Build a MagicMock standing in for `with httpx.Client(...) as client:`."""
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.get = MagicMock(return_value=response)
    return mock_client


def _mock_sync_client_raising(exc: Exception):
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.get = MagicMock(side_effect=exc)
    return mock_client


# ---------------------------------------------------------------------------
# resolve_context_budget_window
# ---------------------------------------------------------------------------


class TestResolveContextBudgetWindow:
    def test_explicit_int_wins_no_http_call(self, monkeypatch):
        from backend import factories

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("httpx.Client must not be constructed for an explicit int")

        monkeypatch.setattr(factories.httpx, "Client", _fail_if_called)

        context_budget = ContextBudgetConfig(context_window=8192)
        llm_config = LLMConfig()

        result = factories.resolve_context_budget_window(context_budget, llm_config)

        assert result == 8192

    def test_auto_resolves_from_props(self, monkeypatch):
        from backend import factories

        response = httpx.Response(
            200,
            json={"default_generation_settings": {"n_ctx": 32768}},
            request=httpx.Request("GET", "http://llm/props"),
        )
        monkeypatch.setattr(
            factories.httpx, "Client", lambda **kw: _mock_sync_client_returning(response)
        )

        context_budget = ContextBudgetConfig(context_window="auto")
        llm_config = LLMConfig(base_url="http://llm")

        result = factories.resolve_context_budget_window(context_budget, llm_config)

        assert result == 32768

    def test_auto_falls_back_on_connection_error(self, monkeypatch, caplog):
        from backend import factories

        monkeypatch.setattr(
            factories.httpx,
            "Client",
            lambda **kw: _mock_sync_client_raising(httpx.ConnectError("refused")),
        )
        monkeypatch.delenv("LLM_CTX_SIZE", raising=False)

        context_budget = ContextBudgetConfig(context_window="auto")
        llm_config = LLMConfig(base_url="http://llm")

        with caplog.at_level(logging.WARNING):
            result = factories.resolve_context_budget_window(context_budget, llm_config)

        assert result == 32768
        assert any("falling back" in r.message for r in caplog.records)

    def test_auto_falls_back_to_llm_ctx_size_env(self, monkeypatch, caplog):
        from backend import factories

        monkeypatch.setattr(
            factories.httpx,
            "Client",
            lambda **kw: _mock_sync_client_raising(httpx.ConnectError("refused")),
        )
        monkeypatch.setenv("LLM_CTX_SIZE", "16384")

        context_budget = ContextBudgetConfig(context_window="auto")
        llm_config = LLMConfig(base_url="http://llm")

        with caplog.at_level(logging.WARNING):
            result = factories.resolve_context_budget_window(context_budget, llm_config)

        assert result == 16384

    def test_auto_falls_back_on_non_200(self, monkeypatch):
        from backend import factories

        response = httpx.Response(503, request=httpx.Request("GET", "http://llm/props"))
        monkeypatch.setattr(
            factories.httpx, "Client", lambda **kw: _mock_sync_client_returning(response)
        )
        monkeypatch.delenv("LLM_CTX_SIZE", raising=False)

        context_budget = ContextBudgetConfig(context_window="auto")
        llm_config = LLMConfig(base_url="http://llm")

        result = factories.resolve_context_budget_window(context_budget, llm_config)

        assert result == 32768

    def test_auto_falls_back_on_malformed_json(self, monkeypatch):
        from backend import factories

        response = httpx.Response(
            200, content=b"not json", request=httpx.Request("GET", "http://llm/props")
        )
        monkeypatch.setattr(
            factories.httpx, "Client", lambda **kw: _mock_sync_client_returning(response)
        )
        monkeypatch.delenv("LLM_CTX_SIZE", raising=False)

        context_budget = ContextBudgetConfig(context_window="auto")
        llm_config = LLMConfig(base_url="http://llm")

        result = factories.resolve_context_budget_window(context_budget, llm_config)

        assert result == 32768


# ---------------------------------------------------------------------------
# build_llm_provider -- AdaptiveThinkingProvider wiring
# ---------------------------------------------------------------------------


class TestBuildLlmProviderAdaptiveThinkingWiring:
    def test_default_wraps_outermost_with_adaptive_thinking(self):
        from backend.factories import build_llm_provider
        from backend.llm.adaptive_thinking import AdaptiveThinkingProvider
        from backend.llm.llama_server_provider import LlamaServerProvider

        config = build_test_config()
        assert config.llm.tool_thinking_budget_tokens == 1024  # default, not "off"

        provider = build_llm_provider(config)

        assert isinstance(provider, AdaptiveThinkingProvider)
        assert isinstance(provider.inner, LlamaServerProvider)

    def test_off_disables_wrapper_entirely(self):
        from backend.factories import build_llm_provider
        from backend.llm.adaptive_thinking import AdaptiveThinkingProvider
        from backend.llm.llama_server_provider import LlamaServerProvider

        config = build_test_config()
        config.llm.tool_thinking_budget_tokens = None

        provider = build_llm_provider(config)

        assert not isinstance(provider, AdaptiveThinkingProvider)
        assert isinstance(provider, LlamaServerProvider)

    def test_adaptive_thinking_wraps_outside_instrumented_provider(self):
        from backend.factories import build_llm_provider
        from backend.llm.adaptive_thinking import AdaptiveThinkingProvider
        from backend.llm.instrumented_provider import InstrumentedStreamingLLMProvider
        from backend.llm.llama_server_provider import LlamaServerProvider

        class _FakeDebugLogger:
            llm_call_enabled = True

        config = build_test_config()

        provider = build_llm_provider(config, debug_logger=_FakeDebugLogger())

        assert isinstance(provider, AdaptiveThinkingProvider)
        assert isinstance(provider.inner, InstrumentedStreamingLLMProvider)
        assert isinstance(provider.inner.inner, LlamaServerProvider)

    def test_budget_tokens_forwarded_from_config(self):
        from backend.factories import build_llm_provider

        config = build_test_config()
        config.llm.tool_thinking_budget_tokens = 2048

        provider = build_llm_provider(config)

        assert provider._budget_tokens == 2048


# ---------------------------------------------------------------------------
# build_conversation_handler -- resolved context-window wiring
# ---------------------------------------------------------------------------


class TestBuildConversationHandlerContextWindowWiring:
    @requires_sentence_transformers
    def test_auto_window_resolved_via_props_reaches_planner(self, monkeypatch):
        from backend import factories
        from backend.factories import build_conversation_handler
        from backend.knowledge.storage.graph_store import GraphStore
        from tests.mocks.embeddings import FakeEmbeddingGenerator
        from tests.mocks.neo4j import FakeNeo4jConnection
        from tests.mocks.ollama import FakeLLM
        from tests.unit.knowledge.conftest import FakeVectorStore

        monkeypatch.delenv("MIST_HYDRATION_ISOLATION", raising=False)

        response = httpx.Response(
            200,
            json={"default_generation_settings": {"n_ctx": 32768}},
            request=httpx.Request("GET", "http://llm/props"),
        )
        monkeypatch.setattr(
            factories.httpx, "Client", lambda **kw: _mock_sync_client_returning(response)
        )

        config = build_test_config()
        config.context_budget = ContextBudgetConfig(
            context_window="auto",
            output_reserve_tokens=512,
            safety_margin_tokens=256,
            enabled=True,
        )

        graph_store = GraphStore(FakeNeo4jConnection(), FakeEmbeddingGenerator())

        handler = build_conversation_handler(
            config=config,
            graph_store=graph_store,
            vector_store=FakeVectorStore(),
            llm_provider=FakeLLM(),
        )

        assert handler._budget_planner is not None
        assert handler._budget_planner._config.context_window == 32768

    @requires_sentence_transformers
    def test_explicit_int_window_wins_no_http_call(self, monkeypatch):
        from backend import factories
        from backend.factories import build_conversation_handler
        from backend.knowledge.storage.graph_store import GraphStore
        from tests.mocks.embeddings import FakeEmbeddingGenerator
        from tests.mocks.neo4j import FakeNeo4jConnection
        from tests.mocks.ollama import FakeLLM
        from tests.unit.knowledge.conftest import FakeVectorStore

        monkeypatch.delenv("MIST_HYDRATION_ISOLATION", raising=False)

        def _fail_if_called(**kw):
            raise AssertionError("httpx.Client must not be constructed for an explicit int")

        monkeypatch.setattr(factories.httpx, "Client", _fail_if_called)

        config = build_test_config()
        config.context_budget = ContextBudgetConfig(context_window=8192, enabled=True)

        graph_store = GraphStore(FakeNeo4jConnection(), FakeEmbeddingGenerator())

        handler = build_conversation_handler(
            config=config,
            graph_store=graph_store,
            vector_store=FakeVectorStore(),
            llm_provider=FakeLLM(),
        )

        assert handler._budget_planner._config.context_window == 8192

    @requires_sentence_transformers
    def test_disabled_budgeting_keeps_planner_none(self, monkeypatch):
        from backend.factories import build_conversation_handler
        from backend.knowledge.storage.graph_store import GraphStore
        from tests.mocks.embeddings import FakeEmbeddingGenerator
        from tests.mocks.neo4j import FakeNeo4jConnection
        from tests.mocks.ollama import FakeLLM
        from tests.unit.knowledge.conftest import FakeVectorStore

        monkeypatch.delenv("MIST_HYDRATION_ISOLATION", raising=False)

        config = build_test_config()
        config.context_budget = ContextBudgetConfig(context_window="auto", enabled=False)

        graph_store = GraphStore(FakeNeo4jConnection(), FakeEmbeddingGenerator())

        handler = build_conversation_handler(
            config=config,
            graph_store=graph_store,
            vector_store=FakeVectorStore(),
            llm_provider=FakeLLM(),
        )

        assert handler._budget_planner is None
