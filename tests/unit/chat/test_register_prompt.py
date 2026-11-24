"""The register in the prompt: a fixed prefix line, and a spoken-reply part in the delta.

A spoken reply should be brief, a typed one need not be. The request for brevity rides in
the per-turn delta of voice turns on `tts` only, so switching between typed and spoken turns
never moves the cached prefix; the register line every turn shares sits in the stable prefix.
A typed reply spoken on request (protocol 2.1.0) is shown as text too, so it keeps the written
register.
"""

from __future__ import annotations

import json
from typing import Literal

import pytest

from backend.chat.context_budget import TurnDeltaPart
from backend.chat.conversation_handler import SPOKEN_REGISTER_TEXT, ConversationHandler
from backend.chat.mist_context import MistContext
from backend.chat.turn_scope import TurnDelivery, TurnRequest, TurnScope
from backend.knowledge.config import ContextBudgetConfig
from backend.knowledge.models import RetrievalResult
from backend.knowledge.storage.graph_store import GraphStore
from tests.mocks.config import build_test_config
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection
from tests.mocks.streaming_llm import FakeStreamingLLMProvider, ScriptedPass
from tests.unit.conftest import make_test_conventions_loader

REGISTER_LINE = "Keep a casual but respectful register."


class _NoopExtractionPipeline:
    async def extract_from_utterance(self, **kwargs):
        from backend.knowledge.extraction.validator import ValidationResult

        return ValidationResult(valid=True, entities=[], relationships=[])


class _EmptyRetriever:
    """No retrieval part, so the register part is the only thing a delta can carry."""

    async def retrieve(self, *, query: str, user_id: str, **_kwargs) -> RetrievalResult:
        return RetrievalResult(
            query=query,
            user_id=user_id,
            facts=[],
            entities_found=0,
            total_facts=0,
            formatted_context="",
            retrieval_time_ms=1.0,
            vector_search_time_ms=0.0,
            graph_traversal_time_ms=0.0,
            config_used={},
            intent="historical",
        )

    async def retrieve_mist_context(self) -> MistContext:
        return MistContext(display_name="MIST", pronouns="she/her", self_concept="")


class _PlanSpy:
    """Wraps the planner's `plan` to keep each call's arguments and result."""

    def __init__(self, plan) -> None:
        self._plan = plan
        self.calls: list[tuple[dict, object]] = []

    def __call__(self, **kwargs):
        result = self._plan(**kwargs)
        self.calls.append((kwargs, result))
        return result


def _handler(
    passes: int = 2, *, register_prompt: str | None = None
) -> tuple[ConversationHandler, FakeStreamingLLMProvider, _PlanSpy]:
    config = build_test_config()
    config.context_budget = ContextBudgetConfig(context_window=32768)
    provider = FakeStreamingLLMProvider(
        [ScriptedPass(chunks=[f"Reply {n} is here."]) for n in range(1, passes + 1)]
    )
    handler = ConversationHandler(
        config=config,
        graph_store=GraphStore(FakeNeo4jConnection(), FakeEmbeddingGenerator()),
        extraction_pipeline=_NoopExtractionPipeline(),
        retriever=_EmptyRetriever(),
        llm_provider=provider,
        conventions_loader=make_test_conventions_loader(),
        register_prompt=register_prompt,
    )
    spy = _PlanSpy(handler._budget_planner.plan)
    handler._budget_planner.plan = spy
    return handler, provider, spy


async def _run_turn(
    handler: ConversationHandler,
    text: str,
    *,
    spoken: bool,
    origin: Literal["voice", "text"] | None = None,
    channel: Literal["tts", "text"] | None = None,
) -> TurnScope:
    """Run one turn: `spoken` is a voice turn on `tts`; `origin` and `channel` override it."""
    scope = handler.open_turn(
        TurnRequest(
            session_id="s1",
            user_text=text,
            origin=origin or ("voice" if spoken else "text"),
            channel=channel or ("tts" if spoken else "text"),
        )
    )
    async for _event in handler.stream_turn(scope):
        pass
    handler.commit_turn(
        scope,
        TurnDelivery(
            outcome="completed",
            delivered_text=scope.generated_text,
            stt_model=None,
            tts_model=None,
        ),
    )
    return scope


def test_spoken_register_text_is_the_fixed_sentence():
    assert SPOKEN_REGISTER_TEXT == (
        "This reply is spoken. Answer in one to three short sentences, and offer more on "
        "screen or on request."
    )


class TestRegisterPromptInThePrefix:
    @pytest.mark.asyncio
    async def test_sits_after_the_persona_block(self):
        handler, provider, _ = _handler(1, register_prompt=REGISTER_LINE)
        await _run_turn(handler, "what is on today", spoken=False)
        messages = provider.calls[0].messages
        persona = MistContext(
            display_name="MIST", pronouns="she/her", self_concept=""
        ).as_system_prompt_block()
        assert messages[0] == {"role": "system", "content": persona}
        assert messages[1] == {"role": "system", "content": REGISTER_LINE}
        assert sum(REGISTER_LINE in m["content"] for m in messages) == 1

    @pytest.mark.asyncio
    async def test_absent_when_not_given(self):
        with_line, with_provider, _ = _handler(1, register_prompt=REGISTER_LINE)
        without, without_provider, _ = _handler(1)
        await _run_turn(with_line, "what is on today", spoken=False)
        await _run_turn(without, "what is on today", spoken=False)
        with_messages = with_provider.calls[0].messages
        assert without_provider.calls[0].messages == with_messages[:1] + with_messages[2:]

    @pytest.mark.asyncio
    async def test_is_charged_as_fixed_cost(self):
        with_line, _, with_spy = _handler(1, register_prompt=REGISTER_LINE)
        without, _, without_spy = _handler(1)
        await _run_turn(with_line, "what is on today", spoken=False)
        await _run_turn(without, "what is on today", spoken=False)
        with_kwargs, with_plan = with_spy.calls[0]
        _, without_plan = without_spy.calls[0]
        assert REGISTER_LINE in with_kwargs["extra_fixed_texts"]
        assert with_plan.fixed_cost > without_plan.fixed_cost

    @pytest.mark.asyncio
    async def test_is_part_of_the_prefix_fingerprint_and_estimate(self):
        with_line, _, _ = _handler(1, register_prompt=REGISTER_LINE)
        without, _, _ = _handler(1)
        with_scope = await _run_turn(with_line, "what is on today", spoken=False)
        without_scope = await _run_turn(without, "what is on today", spoken=False)
        assert with_scope.prompt_stats.prefix_tokens_est > (
            without_scope.prompt_stats.prefix_tokens_est
        )


class TestSpokenRegisterInTheDelta:
    """Decision 5 (b): the spoken register is for a voice turn on `tts` only.

    A typed reply with playback (protocol 2.1.0) is on `tts` too, but keeps the written register:
    its client shows the text, so the delta stays off it.
    """

    @pytest.mark.asyncio
    async def test_tts_turn_carries_one_register_part(self):
        """A voice turn on `tts`: the one turn that carries the spoken register."""
        handler, provider, spy = _handler(1)
        await _run_turn(handler, "what is on today", spoken=True)
        kwargs, _ = spy.calls[0]
        register_parts = [p for p in kwargs["delta_parts"] if p.kind == "register"]
        assert register_parts == [
            TurnDeltaPart(kind="register", text=SPOKEN_REGISTER_TEXT, prunable=False)
        ]
        messages = provider.calls[0].messages
        assert SPOKEN_REGISTER_TEXT in messages[-1]["content"]
        assert messages[-1]["content"].endswith("what is on today")
        assert all(SPOKEN_REGISTER_TEXT not in m["content"] for m in messages[:-1])

    @pytest.mark.asyncio
    async def test_text_turn_carries_none(self):
        """A typed turn without playback, on `text`."""
        handler, provider, spy = _handler(1)
        await _run_turn(handler, "what is on today", spoken=False)
        kwargs, _ = spy.calls[0]
        assert [p for p in kwargs["delta_parts"] if p.kind == "register"] == []
        messages = provider.calls[0].messages
        assert all(SPOKEN_REGISTER_TEXT not in m["content"] for m in messages)
        assert messages[-1] == {"role": "user", "content": "what is on today"}

    @pytest.mark.asyncio
    async def test_a_typed_turn_with_playback_on_tts_carries_none(self):
        handler, provider, spy = _handler(1)
        await _run_turn(handler, "what is on today", spoken=False, origin="text", channel="tts")
        kwargs, _ = spy.calls[0]
        assert [p for p in kwargs["delta_parts"] if p.kind == "register"] == []
        messages = provider.calls[0].messages
        assert all(SPOKEN_REGISTER_TEXT not in m["content"] for m in messages)
        assert messages[-1] == {"role": "user", "content": "what is on today"}

    @pytest.mark.asyncio
    async def test_a_voice_turn_on_text_carries_none(self):
        """No synthesizer, or playback off: the reply is read, so the delta stays off, as before."""
        handler, provider, spy = _handler(1)
        await _run_turn(handler, "what is on today", spoken=True, channel="text")
        kwargs, _ = spy.calls[0]
        assert [p for p in kwargs["delta_parts"] if p.kind == "register"] == []
        messages = provider.calls[0].messages
        assert all(SPOKEN_REGISTER_TEXT not in m["content"] for m in messages)


class TestSpokenAndTypedTurnsShareThePrefix:
    @pytest.mark.asyncio
    async def test_prefix_messages_and_tools_match_byte_for_byte(self):
        handler, provider, _ = _handler(2, register_prompt=REGISTER_LINE)
        await _run_turn(handler, "what is on today", spoken=True)
        typed = await _run_turn(handler, "and tomorrow", spoken=False)

        spoken_call, typed_call = provider.calls
        # Turn 1 has no history, so everything before its final message is the prefix.
        prefix_len = len(spoken_call.messages) - 1
        assert json.dumps(typed_call.messages[:prefix_len]) == json.dumps(
            spoken_call.messages[:prefix_len]
        )
        assert json.dumps(typed_call.tools, sort_keys=True) == json.dumps(
            spoken_call.tools, sort_keys=True
        )
        assert typed.prompt_stats.prefix_changed is False

        # Only the delta differs: the spoken turn's register part, stored nowhere else.
        assert SPOKEN_REGISTER_TEXT in spoken_call.messages[-1]["content"]
        assert all(SPOKEN_REGISTER_TEXT not in m["content"] for m in typed_call.messages)
        assert typed_call.messages[prefix_len] == {"role": "user", "content": "what is on today"}
        assert typed_call.messages[-1] == {"role": "user", "content": "and tomorrow"}
