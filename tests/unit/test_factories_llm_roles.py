"""T39: the conversation role's endpoint, and which builder's provider follows it.

The role moves the conversation, and only the conversation. Extraction and
curation stay on `LLM_SERVER_URL` and `MODEL` whatever the role says, and with
the role unset every builder is what it was before the role existed: the
handler and the extraction pipeline hold one provider object.

The configs are read from the environment through `LLMConfig.from_env`, the
production path, and dropped into `build_test_config()`'s config so nothing
else in it comes from `.env`. Graph and vector stores are fakes; no test here
reaches a live service (`LlamaServerProvider` opens no connection until a call).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from backend import factories
from backend.knowledge.config import ContextBudgetConfig, LLMConfig, LLMEndpoint
from backend.knowledge.storage.graph_store import GraphStore
from tests.mocks.config import build_test_config
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeGraphExecutor, FakeNeo4jConnection
from tests.mocks.ollama import FakeLLM
from tests.unit.knowledge.conftest import FakeVectorStore

BASE_URL = "http://base-llm:8080"
BASE_MODEL = "base-model"
CONVERSATION_URL = "http://conversation-llm:8081"
CONVERSATION_MODEL = "conversation-model"

BASE = LLMEndpoint(base_url=BASE_URL, model=BASE_MODEL)


@pytest.fixture
def llm_env(monkeypatch):
    """Pin the base endpoint and clear the role, the debug sink and the thinking budget."""
    monkeypatch.setenv("LLM_BACKEND", "llamacpp")
    monkeypatch.setenv("LLM_SERVER_URL", BASE_URL)
    monkeypatch.setenv("MODEL", BASE_MODEL)
    for name in (
        "MIST_LLM_CONVERSATION_URL",
        "MIST_LLM_CONVERSATION_MODEL",
        "MIST_TOOL_THINKING_BUDGET",
        "MIST_DEBUG_JSONL",
        "MIST_DEBUG_LLM_JSONL",
        "MIST_HYDRATION_ISOLATION",
    ):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _config():
    """A test config whose `llm` section is read from the environment."""
    config = build_test_config()
    config.llm = LLMConfig.from_env()
    # An "auto" window would probe the base URL's /props; the window is not under test.
    config.context_budget = ContextBudgetConfig(context_window=8192, enabled=False)
    return config


def _set_role(env, *, model: str | None = None) -> None:
    env.setenv("MIST_LLM_CONVERSATION_URL", CONVERSATION_URL)
    if model is not None:
        env.setenv("MIST_LLM_CONVERSATION_MODEL", model)


def _endpoint_of(provider) -> LLMEndpoint:
    """Unwrap the instrumented and adaptive-thinking wrappers to the concrete provider."""
    while hasattr(provider, "inner"):
        provider = provider.inner
    return LLMEndpoint(base_url=provider._base_url, model=provider.model)


def _wrapper_chain(provider) -> list[type]:
    chain = [type(provider)]
    while hasattr(provider, "inner"):
        provider = provider.inner
        chain.append(type(provider))
    return chain


def _handler(config, **injected):
    return factories.build_conversation_handler(
        config=config,
        graph_store=GraphStore(FakeNeo4jConnection(), FakeEmbeddingGenerator()),
        vector_store=FakeVectorStore(),
        **injected,
    )


def _pipeline_provider(handler):
    return handler._extraction_pipeline._extractor._llm


class TestBuildLlmProviderRole:
    def test_no_role_targets_the_base(self, llm_env):
        assert _endpoint_of(factories.build_llm_provider(_config())) == BASE

    def test_conversation_role_unset_targets_the_base(self, llm_env):
        provider = factories.build_llm_provider(_config(), role="conversation")
        assert _endpoint_of(provider) == BASE

    def test_conversation_role_set_takes_its_url_and_the_base_model(self, llm_env):
        _set_role(llm_env)
        provider = factories.build_llm_provider(_config(), role="conversation")
        assert _endpoint_of(provider) == LLMEndpoint(CONVERSATION_URL, BASE_MODEL)

    def test_conversation_role_model_variable_wins(self, llm_env):
        _set_role(llm_env, model=CONVERSATION_MODEL)
        provider = factories.build_llm_provider(_config(), role="conversation")
        assert _endpoint_of(provider) == LLMEndpoint(CONVERSATION_URL, CONVERSATION_MODEL)

    def test_role_set_leaves_the_default_provider_on_the_base(self, llm_env):
        _set_role(llm_env, model=CONVERSATION_MODEL)
        assert _endpoint_of(factories.build_llm_provider(_config())) == BASE

    def test_role_provider_is_wrapped_exactly_as_the_base(self, llm_env):
        from backend.llm.adaptive_thinking import AdaptiveThinkingProvider
        from backend.llm.instrumented_provider import InstrumentedStreamingLLMProvider
        from backend.llm.llama_server_provider import LlamaServerProvider

        class _GateOnLogger:
            llm_call_enabled = True

        _set_role(llm_env)
        config = _config()
        assert config.llm.tool_thinking_budget_tokens is not None

        base = factories.build_llm_provider(config, debug_logger=_GateOnLogger())
        role = factories.build_llm_provider(
            config, debug_logger=_GateOnLogger(), role="conversation"
        )

        expected = [AdaptiveThinkingProvider, InstrumentedStreamingLLMProvider, LlamaServerProvider]
        assert _wrapper_chain(base) == expected
        assert _wrapper_chain(role) == expected


class TestBuildConversationHandlerProviderSplit:
    """Design review finding 1: the pipeline keeps the base, the handler takes the role."""

    def test_role_unset_built_provider_is_one_object_on_the_base(self, llm_env):
        handler = _handler(_config())

        assert handler._provider is _pipeline_provider(handler)
        assert _endpoint_of(handler._provider) == BASE

    def test_role_unset_injected_provider_is_one_object(self, llm_env):
        injected = FakeLLM()
        handler = _handler(_config(), llm_provider=injected)

        assert handler._provider is injected
        assert _pipeline_provider(handler) is injected

    @pytest.mark.parametrize(
        ("model_var", "expected_model"),
        [(None, BASE_MODEL), (CONVERSATION_MODEL, CONVERSATION_MODEL)],
        ids=["model-from-MODEL", "model-from-role-variable"],
    )
    def test_role_set_nothing_injected(self, llm_env, model_var, expected_model):
        _set_role(llm_env, model=model_var)

        handler = _handler(_config())

        assert _endpoint_of(handler._provider) == LLMEndpoint(CONVERSATION_URL, expected_model)
        assert _endpoint_of(_pipeline_provider(handler)) == BASE
        assert handler._provider is not _pipeline_provider(handler)

    def test_role_set_injected_base_stays_the_pipelines(self, llm_env):
        """The live server's path: `voice_processor` injects the base provider."""
        _set_role(llm_env, model=CONVERSATION_MODEL)
        injected = FakeLLM()

        handler = _handler(_config(), llm_provider=injected)

        assert _pipeline_provider(handler) is injected
        assert _endpoint_of(handler._provider) == LLMEndpoint(CONVERSATION_URL, CONVERSATION_MODEL)

    @pytest.mark.parametrize("role_set", [False, True], ids=["role-unset", "role-set"])
    def test_injected_conversation_provider_is_the_handlers(self, llm_env, role_set):
        if role_set:
            _set_role(llm_env)
        base, conversation = FakeLLM(), FakeLLM()

        handler = _handler(_config(), llm_provider=base, conversation_llm_provider=conversation)

        assert handler._provider is conversation
        assert _pipeline_provider(handler) is base


class _FakeGraphStore:
    """The attributes `build_extraction_pipeline` reads with curation and Stage 9 off."""

    def __init__(self):
        self.connection = FakeNeo4jConnection()
        self.embedding_generator = None


@pytest.mark.parametrize("role_set", [False, True], ids=["role-unset", "role-set"])
class TestOtherBuildersKeepTheBase:
    def test_extraction_pipeline_alone(self, llm_env, role_set):
        if role_set:
            _set_role(llm_env, model=CONVERSATION_MODEL)

        pipeline = factories.build_extraction_pipeline(
            _config(),
            graph_store=_FakeGraphStore(),
            include_curation=False,
            include_internal_derivation=False,
        )

        assert _endpoint_of(pipeline._extractor._llm) == BASE

    def test_curation_scheduler_alone(self, llm_env, role_set):
        if role_set:
            _set_role(llm_env, model=CONVERSATION_MODEL)
        connection = FakeNeo4jConnection()

        with (
            patch.object(
                factories,
                "build_graph_store",
                return_value=GraphStore(connection, FakeEmbeddingGenerator()),
            ),
            patch.object(
                factories, "build_graph_executor", return_value=FakeGraphExecutor(connection)
            ),
        ):
            scheduler = factories.build_curation_scheduler(_config())

        (reflection,) = (job for cfg, job in scheduler._jobs if cfg.name == "self_reflection")
        assert _endpoint_of(reflection._deriver._llm) == BASE


def test_writer_stamps_do_not_depend_on_the_conversation_role(llm_env):
    without_role = factories.writer_stamps_from_config(_config())
    _set_role(llm_env, model=CONVERSATION_MODEL)
    config = _config()
    assert "conversation" in config.llm.roles, "precondition: the role is set"
    with_role = factories.writer_stamps_from_config(config)

    assert with_role == without_role
