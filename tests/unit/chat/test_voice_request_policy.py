"""Voice turn request policy: no thinking on spoken turns, first clause released early.

A spoken reply cannot start until the model's first words reach TTS, so a voice turn asks
for no reasoning at all and a `tts` turn releases its first clause before the sentence ends.
Typed turns keep the adaptive thinking budget; one without playback (protocol 2.1.0) is on
`text`, where Tokens break at sentence boundaries, and one with playback is on `tts`, so it
releases its first clause early too.
"""

from __future__ import annotations

import asyncio

import pytest

from backend.chat.conversation_handler import ConversationHandler
from backend.chat.stream_events import Complete, Token
from backend.chat.turn_scope import TurnRequest, TurnScope
from backend.knowledge.config import LLMConfig, LLMEndpoint
from backend.knowledge.retrieval.knowledge_retriever import KnowledgeRetriever
from backend.knowledge.storage.graph_store import GraphStore
from backend.llm.adaptive_thinking import (
    AdaptiveThinkingProvider,
    get_adaptive_thinking_retry_count,
    reset_adaptive_thinking_retry_count,
)
from backend.llm.models import ToolCall as LLMToolCall
from backend.sentence_detector import find_clause_boundary
from tests.mocks.config import build_test_config
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection
from tests.mocks.streaming_llm import FakeStreamingLLMProvider, ScriptedPass
from tests.unit.conftest import make_test_conventions_loader

# The comma after "right now" is the first clause boundary at or past 40 characters; the one
# after "Sure thing" is too early.
REPLY = (
    "Sure thing, I can look into the weather for you right now, and then I will tell you "
    "more. Second sentence here."
)
FIRST_CLAUSE = "Sure thing, I can look into the weather for you right now,"


class _NoopExtractionPipeline:
    async def extract_from_utterance(self, **kwargs):
        from backend.knowledge.extraction.validator import ValidationResult

        return ValidationResult(valid=True, entities=[], relationships=[])


class _StubDispatch:
    """Stands in for `_dispatch_tool` so a tool pass needs no tool registry."""

    async def __call__(self, scope: TurnScope, tool_call: LLMToolCall) -> str:
        return "stub tool result"


def _handler(provider, *, min_chars: int | None = None, voice_budget: int | None = None):
    config = build_test_config()
    if min_chars is not None:
        config.llm.voice_first_segment_min_chars = min_chars
    if voice_budget is not None:
        config.llm.voice_thinking_budget_tokens = voice_budget
    graph_store = GraphStore(FakeNeo4jConnection(), FakeEmbeddingGenerator())
    handler = ConversationHandler(
        config=config,
        graph_store=graph_store,
        extraction_pipeline=_NoopExtractionPipeline(),
        retriever=KnowledgeRetriever(config=config, graph_store=graph_store),
        llm_provider=provider,
        conventions_loader=make_test_conventions_loader(),
    )
    handler._dispatch_tool = _StubDispatch()
    return handler


def _open(handler: ConversationHandler, *, origin: str, channel: str) -> TurnScope:
    return handler.open_turn(
        TurnRequest(
            session_id=f"s-{origin}-{channel}",
            user_text="What is the weather like today?",
            origin=origin,
            channel=channel,
        )
    )


async def _tokens(handler: ConversationHandler, *, origin: str, channel: str) -> list[Token]:
    scope = _open(handler, origin=origin, channel=channel)
    tokens: list[Token] = []
    async for event in handler.stream_turn(scope):
        if isinstance(event, Token):
            tokens.append(event)
    return tokens


def _tool_call() -> list[LLMToolCall]:
    return [LLMToolCall(id="call_1", name="query_knowledge_graph", arguments={"query": "x"})]


@pytest.fixture(autouse=True)
def _reset_retry_count():
    reset_adaptive_thinking_retry_count()
    yield
    reset_adaptive_thinking_retry_count()


class TestLLMConfigVoiceFields:
    def test_defaults(self):
        config = LLMConfig()
        assert config.voice_thinking_budget_tokens == 0
        assert config.voice_first_segment_min_chars == 40

    def test_from_env_defaults_when_unset(self, monkeypatch):
        monkeypatch.delenv("MIST_VOICE_THINKING_BUDGET_TOKENS", raising=False)
        monkeypatch.delenv("MIST_VOICE_FIRST_SEGMENT_MIN_CHARS", raising=False)
        config = LLMConfig.from_env()
        assert config.voice_thinking_budget_tokens == 0
        assert config.voice_first_segment_min_chars == 40

    def test_from_env_reads_both_variables(self, monkeypatch):
        monkeypatch.setenv("MIST_VOICE_THINKING_BUDGET_TOKENS", "64")
        monkeypatch.setenv("MIST_VOICE_FIRST_SEGMENT_MIN_CHARS", "25")
        config = LLMConfig.from_env()
        assert config.voice_thinking_budget_tokens == 64
        assert config.voice_first_segment_min_chars == 25

    @pytest.mark.parametrize(
        "variable", ["MIST_VOICE_THINKING_BUDGET_TOKENS", "MIST_VOICE_FIRST_SEGMENT_MIN_CHARS"]
    )
    def test_negative_value_raises_naming_the_variable(self, monkeypatch, variable):
        monkeypatch.setenv(variable, "-1")
        with pytest.raises(ValueError, match=variable):
            LLMConfig.from_env()

    @pytest.mark.parametrize(
        "variable", ["MIST_VOICE_THINKING_BUDGET_TOKENS", "MIST_VOICE_FIRST_SEGMENT_MIN_CHARS"]
    )
    def test_non_integer_value_raises_naming_the_variable(self, monkeypatch, variable):
        monkeypatch.setenv(variable, "lots")
        with pytest.raises(ValueError, match=variable):
            LLMConfig.from_env()

    def test_endpoint_resolution_is_unchanged(self, monkeypatch):
        monkeypatch.setenv("MIST_VOICE_THINKING_BUDGET_TOKENS", "64")
        monkeypatch.setenv("MIST_VOICE_FIRST_SEGMENT_MIN_CHARS", "25")
        monkeypatch.delenv("MIST_LLM_CONVERSATION_URL", raising=False)
        monkeypatch.delenv("MIST_LLM_CONVERSATION_MODEL", raising=False)
        config = LLMConfig.from_env()
        assert config.endpoint("conversation") == LLMEndpoint(
            base_url=config.base_url, model=config.model
        )


class TestVoiceTurnThinking:
    """No-thinking follows the origin: a voice turn; never a typed one, spoken or not (5 (b))."""

    @pytest.mark.asyncio
    async def test_both_passes_reach_the_inner_provider_with_budget_zero(self):
        inner = FakeStreamingLLMProvider(
            [
                ScriptedPass(chunks=[], tool_calls=_tool_call()),
                ScriptedPass(chunks=["It is sunny today."]),
            ]
        )
        handler = _handler(AdaptiveThinkingProvider(inner, budget_tokens=1024))

        await _tokens(handler, origin="voice", channel="tts")

        assert len(inner.calls) == 2
        assert [call.thinking.budget_tokens for call in inner.calls] == [0, 0]
        assert get_adaptive_thinking_retry_count() == 0

    @pytest.mark.asyncio
    async def test_a_failing_first_attempt_is_not_retried_unbudgeted(self):
        # Length-exhausted with no output is the shape the adaptive wrapper retries on when
        # it applied its own budget. A voice turn set the budget itself, so nothing retries.
        inner = FakeStreamingLLMProvider([ScriptedPass(chunks=[], finish_reason="length")])
        handler = _handler(AdaptiveThinkingProvider(inner, budget_tokens=1024))

        await _tokens(handler, origin="voice", channel="tts")

        assert len(inner.calls) == 1
        assert inner.calls[0].thinking.budget_tokens == 0
        assert get_adaptive_thinking_retry_count() == 0

    @pytest.mark.asyncio
    async def test_configured_budget_is_carried(self):
        inner = FakeStreamingLLMProvider(
            [
                ScriptedPass(chunks=[], tool_calls=_tool_call()),
                ScriptedPass(chunks=["It is sunny today."]),
            ]
        )
        handler = _handler(inner, voice_budget=64)

        await _tokens(handler, origin="voice", channel="tts")

        assert [call.thinking.budget_tokens for call in inner.calls] == [64, 64]

    @pytest.mark.asyncio
    async def test_text_turn_leaves_thinking_unset_so_the_adaptive_budget_applies(self):
        """A typed turn without playback, on `text`; with playback, see the next test."""
        inner = FakeStreamingLLMProvider(
            [
                ScriptedPass(chunks=[], tool_calls=_tool_call()),
                ScriptedPass(chunks=["It is sunny today."]),
            ]
        )
        recorder = FakeStreamingLLMProvider(
            [
                ScriptedPass(chunks=[], tool_calls=_tool_call()),
                ScriptedPass(chunks=["It is sunny today."]),
            ]
        )
        await _tokens(
            _handler(AdaptiveThinkingProvider(inner, budget_tokens=1024)),
            origin="text",
            channel="text",
        )
        await _tokens(_handler(recorder), origin="text", channel="text")

        assert [call.thinking for call in recorder.calls] == [None, None]
        # Pass 1 carries tools, so the wrapper budgets it; pass 2 has none and passes through.
        assert inner.calls[0].thinking.budget_tokens == 1024
        assert inner.calls[1].thinking is None

    @pytest.mark.asyncio
    async def test_a_typed_turn_with_playback_on_tts_leaves_thinking_unset(self):
        """Protocol 2.1.0's spoken typed reply keeps a typed turn's reasoning."""
        recorder = FakeStreamingLLMProvider(
            [
                ScriptedPass(chunks=[], tool_calls=_tool_call()),
                ScriptedPass(chunks=["It is sunny today."]),
            ]
        )
        await _tokens(_handler(recorder, voice_budget=64), origin="text", channel="tts")

        assert [call.thinking for call in recorder.calls] == [None, None]


class TestFirstClauseReleasedEarly:
    @pytest.mark.asyncio
    async def test_tts_turn_releases_the_first_clause_before_the_sentence_ends(self):
        later = asyncio.Event()
        provider = FakeStreamingLLMProvider([ScriptedPass(chunks=[REPLY[:70], later, REPLY[70:]])])
        handler = _handler(provider)
        scope = _open(handler, origin="voice", channel="tts")
        stream = handler.stream_turn(scope)

        # The fake parks on `later` until the test sets it, so a handler that held the clause
        # back to the sentence end would never yield; the timeout turns that into a failure.
        # `asyncio.timeout`, not `wait_for`: the stream must run in this task, because the
        # LLM call context it enters is a context variable reset in the same context.
        async with asyncio.timeout(5):
            first = await anext(stream)
            while not isinstance(first, Token):
                first = await anext(stream)
        assert first.text == FIRST_CLAUSE
        assert first.text == REPLY[: find_clause_boundary(REPLY, 40)]
        assert not later.is_set()

        later.set()
        rest: list[Token] = []
        complete: Complete | None = None
        async for event in stream:
            if isinstance(event, Token):
                rest.append(event)
            elif isinstance(event, Complete):
                complete = event

        # Later emissions break at sentence boundaries, as before; the clause's trailing
        # space becomes the next Token's separator.
        assert [token.text for token in rest] == [
            " and then I will tell you more.",
            " Second sentence here.",
        ]
        assert complete is not None
        assert complete.final_response == REPLY

    @pytest.mark.asyncio
    async def test_min_chars_comes_from_config(self):
        provider = FakeStreamingLLMProvider([ScriptedPass(chunks=[REPLY])])
        tokens = await _tokens(_handler(provider, min_chars=5), origin="voice", channel="tts")
        assert tokens[0].text == "Sure thing,"

    @pytest.mark.asyncio
    async def test_first_counts_across_passes_when_pass_one_only_called_a_tool(self):
        provider = FakeStreamingLLMProvider(
            [
                ScriptedPass(chunks=[], tool_calls=_tool_call()),
                ScriptedPass(chunks=[REPLY]),
            ]
        )
        tokens = await _tokens(_handler(provider), origin="voice", channel="tts")
        assert [(token.text, token.pass_num) for token in tokens] == [
            (FIRST_CLAUSE, 2),
            (" and then I will tell you more.", 2),
            (" Second sentence here.", 2),
        ]

    @pytest.mark.asyncio
    async def test_pass_two_breaks_at_sentences_once_pass_one_released_the_clause(self):
        provider = FakeStreamingLLMProvider(
            [
                ScriptedPass(chunks=[REPLY], tool_calls=_tool_call()),
                ScriptedPass(chunks=[REPLY]),
            ]
        )
        tokens = await _tokens(_handler(provider), origin="voice", channel="tts")
        pass_two = [token.text for token in tokens if token.pass_num == 2]
        assert tokens[0].text == FIRST_CLAUSE
        assert pass_two == [
            " Sure thing, I can look into the weather for you right now, and then I will tell "
            "you more.",
            " Second sentence here.",
        ]

    @pytest.mark.asyncio
    async def test_slop_gate_runs_on_the_clause(self):
        reply = "Great news \U0001f389 on that front for the whole team today, the build passed."
        provider = FakeStreamingLLMProvider([ScriptedPass(chunks=[reply])])
        tokens = await _tokens(_handler(provider), origin="voice", channel="tts")
        assert len(tokens) == 2
        assert "\U0001f389" not in tokens[0].text
        assert tokens[0].text.startswith("Great news")
        assert tokens[0].text.endswith("the whole team today,")
        assert tokens[1].text == " the build passed."

    @pytest.mark.asyncio
    async def test_a_reply_with_no_clause_boundary_flushes_at_the_end(self):
        provider = FakeStreamingLLMProvider([ScriptedPass(chunks=["Sunny today."])])
        tokens = await _tokens(_handler(provider), origin="voice", channel="tts")
        assert [token.text for token in tokens] == ["Sunny today."]

    @pytest.mark.asyncio
    async def test_a_typed_turn_with_playback_releases_the_first_clause_early_too(self):
        """The early clause follows the channel: a spoken typed reply starts its audio sooner."""
        provider = FakeStreamingLLMProvider([ScriptedPass(chunks=[REPLY])])
        tokens = await _tokens(_handler(provider), origin="text", channel="tts")
        assert [token.text for token in tokens] == [
            FIRST_CLAUSE,
            " and then I will tell you more.",
            " Second sentence here.",
        ]


class TestTextChannelUnchanged:
    """The `text` channel: a typed turn without playback, or a voice turn nothing speaks."""

    @pytest.mark.asyncio
    async def test_text_channel_tokens_break_at_sentences(self):
        later = asyncio.Event()
        provider = FakeStreamingLLMProvider([ScriptedPass(chunks=[REPLY[:70], later, REPLY[70:]])])
        handler = _handler(provider)
        scope = _open(handler, origin="text", channel="text")
        later.set()
        tokens = [event async for event in handler.stream_turn(scope) if isinstance(event, Token)]
        assert tokens == [
            Token(
                text="Sure thing, I can look into the weather for you right now, and then I will "
                "tell you more.",
                pass_num=1,
            ),
            Token(text=" Second sentence here.", pass_num=1),
        ]

    @pytest.mark.asyncio
    async def test_voice_origin_on_the_text_channel_breaks_at_sentences(self):
        provider = FakeStreamingLLMProvider([ScriptedPass(chunks=[REPLY])])
        tokens = await _tokens(_handler(provider), origin="voice", channel="text")
        assert [token.text for token in tokens] == [
            "Sure thing, I can look into the weather for you right now, and then I will tell you "
            "more.",
            " Second sentence here.",
        ]
