"""Cache-stable prompt layout: stable prefix, stepped history, per-turn delta last.

llama-server reuses its KV cache up to the first token that differs from the
previous prompt. These tests pin the layout that makes that prefix long:
nothing that varies per turn sits before the history, the history window
moves its start only in steps, and the per-turn context (retrieval, the live
advisory, the interruption note, a clock) goes into the final user message.
The central test drives a session through 30 turns and checks that each
prompt extends the previous one up to the previous turn's user message,
except on the turns where the window steps.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from backend.chat.context_budget import PromptStats, TurnDeltaPart
from backend.chat.conversation_handler import (
    ConversationHandler,
    _render_turn_delta,
)
from backend.chat.mist_context import MistContext
from backend.chat.turn_scope import TurnDelivery, TurnRequest, TurnScope
from backend.knowledge.config import ContextBudgetConfig
from backend.knowledge.models import ConversationSession, RetrievalResult
from backend.knowledge.storage.graph_store import GraphStore
from backend.llm.models import UsageMetadata
from tests.mocks.config import build_test_config
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection
from tests.mocks.streaming_llm import FakeStreamingLLMProvider, ScriptedPass
from tests.unit.conftest import make_test_conventions_loader

MID_REPLY_NOTE = (
    "The user interrupted your previous reply; the assistant message above is the part "
    "they heard."
)
BEFORE_REPLY_NOTE = "The user interrupted your previous turn before you said anything."


class _NoopExtractionPipeline:
    async def extract_from_utterance(self, **kwargs):
        from backend.knowledge.extraction.validator import ValidationResult

        return ValidationResult(valid=True, entities=[], relationships=[])


class _ScriptedRetriever:
    """Returns a different retrieval block for every query, with an advisory on request."""

    def __init__(self, *, advisory_queries: set[str] | None = None) -> None:
        self.advisory_queries = advisory_queries or set()
        self.persona = MistContext(display_name="MIST", pronouns="she/her", self_concept="")

    async def retrieve(self, *, query: str, user_id: str, **_kwargs) -> RetrievalResult:
        advisory = query in self.advisory_queries
        return RetrievalResult(
            query=query,
            user_id=user_id,
            facts=[],
            entities_found=0,
            total_facts=1,
            formatted_context=f"Relevant prose from your vault (query: '{query}'):\nnote on {query}",
            retrieval_time_ms=1.0,
            vector_search_time_ms=0.0,
            graph_traversal_time_ms=0.0,
            config_used={},
            intent="historical",
            requires_mcp=advisory,
            suggested_tools=("weather",) if advisory else (),
        )

    async def retrieve_mist_context(self) -> MistContext:
        return self.persona


def _handler(
    passes: int,
    *,
    retriever: _ScriptedRetriever | None = None,
    budget: ContextBudgetConfig | None = None,
    vault_root: Path | None = None,
) -> tuple[ConversationHandler, FakeStreamingLLMProvider]:
    if vault_root is not None:
        config = build_test_config(
            vault_root=str(vault_root), vault_enabled=True, vault_user_id="user"
        )
    else:
        config = build_test_config()
    config.context_budget = budget or ContextBudgetConfig(context_window=32768)
    provider = FakeStreamingLLMProvider(
        [
            ScriptedPass(
                chunks=[f"Reply {n} is here."], usage=UsageMetadata(prompt_tokens=1000 + n)
            )
            for n in range(1, passes + 1)
        ]
    )
    handler = ConversationHandler(
        config=config,
        graph_store=GraphStore(FakeNeo4jConnection(), FakeEmbeddingGenerator()),
        extraction_pipeline=_NoopExtractionPipeline(),
        retriever=retriever or _ScriptedRetriever(),
        llm_provider=provider,
        conventions_loader=make_test_conventions_loader(),
    )
    return handler, provider


async def _run_turn(
    handler: ConversationHandler,
    text: str,
    *,
    note: str | None = None,
    outcome: str = "completed",
    delivered: str | None = None,
    session_id: str = "s1",
) -> TurnScope:
    scope = handler.open_turn(
        TurnRequest(
            session_id=session_id,
            user_text=text,
            origin="voice",
            channel="tts",
            max_history=10,
            interruption_note=note,
        )
    )
    async for _event in handler.stream_turn(scope):
        pass
    handler.commit_turn(
        scope,
        TurnDelivery(
            outcome=outcome,
            delivered_text=scope.generated_text if delivered is None else delivered,
            stt_model=None,
            tts_model=None,
        ),
    )
    return scope


def _question(turn: int) -> str:
    return f"question number {turn} for you"


class TestThirtyTurnPrefixInvariant:
    @pytest.mark.asyncio
    async def test_each_prompt_extends_the_previous_except_on_window_steps(self):
        advisory_turns = {4, 9, 13, 22}
        interrupted_turn = 17
        retriever = _ScriptedRetriever(advisory_queries={_question(t) for t in advisory_turns})
        handler, provider = _handler(30, retriever=retriever)

        prompts: list[list[dict]] = []
        for turn in range(1, 31):
            interrupted = turn == interrupted_turn
            await _run_turn(
                handler,
                _question(turn),
                note="mid_reply" if turn == interrupted_turn + 1 else None,
                outcome="interrupted" if interrupted else "completed",
                delivered="Reply" if interrupted else None,
            )
            prompts.append(provider.calls[-1].messages)

        prefix_len = len(prompts[0]) - 1  # turn 1: stable prefix + its user message
        broken = []
        for turn in range(2, 31):
            previous, current = prompts[turn - 2], prompts[turn - 1]
            shared = len(previous) - 1  # up to and excluding the previous user message
            assert current[:prefix_len] == previous[:prefix_len]
            if current[:shared] != previous[:shared]:
                broken.append(turn)
        assert broken == [11, 16, 21, 26]

        # Retrieval differed every turn and stayed out of everything but the
        # final user message.
        for turn, prompt in enumerate(prompts, start=1):
            assert _question(turn) in prompt[-1]["content"]
            assert f"note on {_question(turn)}" in prompt[-1]["content"]
            assert all("note on" not in m["content"] for m in prompt[:-1])
            has_advisory = "LIVE DATA ADVISORY" in prompt[-1]["content"]
            assert has_advisory is (turn in advisory_turns)

        # The interruption note rode on the next turn only, and history holds
        # what was heard, not the full reply.
        assert MID_REPLY_NOTE in prompts[interrupted_turn][-1]["content"]
        assert sum(MID_REPLY_NOTE in p[-1]["content"] for p in prompts) == 1
        assert {"role": "assistant", "content": "Reply"} in prompts[interrupted_turn]

    @pytest.mark.asyncio
    async def test_stored_history_keeps_the_raw_user_text(self):
        handler, provider = _handler(2, retriever=_ScriptedRetriever())
        await _run_turn(handler, _question(1))
        await _run_turn(handler, _question(2))

        session = handler.sessions["s1"]
        assert [m.content for m in session.messages if m.role == "user"] == [
            _question(1),
            _question(2),
        ]
        # The second prompt shows turn 1's user message raw, without its delta.
        assert {"role": "user", "content": _question(1)} in provider.calls[1].messages


class TestDeltaPlacement:
    def _build(self, handler: ConversationHandler, **kwargs):
        session = handler.get_or_create_session("p1")
        session.add_message("user", "earlier question here")
        session.add_message("assistant", "earlier answer")
        retrieval = RetrievalResult(
            query="q",
            user_id="User",
            facts=[],
            entities_found=0,
            total_facts=1,
            formatted_context="RETRIEVAL_SENTINEL",
            retrieval_time_ms=1.0,
            vector_search_time_ms=0.0,
            graph_traversal_time_ms=0.0,
            config_used={},
            intent="historical",
            requires_mcp=True,
            suggested_tools=("weather",),
        )
        return handler._build_messages(
            session,
            10,
            retrieval_result=retrieval,
            pending_user_message="what is the weather now",
            **kwargs,
        )

    def test_user_prefix_puts_every_part_in_the_final_user_message(self):
        handler, _ = _handler(0)
        layout = self._build(handler, interruption_note="before_reply")
        messages = layout.messages

        assert messages[-3:-1] == [
            {"role": "user", "content": "earlier question here"},
            {"role": "assistant", "content": "earlier answer"},
        ]
        final = messages[-1]
        assert final["role"] == "user"
        content = final["content"]
        assert content.endswith("what is the weather now")
        note_at = content.index(BEFORE_REPLY_NOTE)
        retrieval_at = content.index("RETRIEVAL_SENTINEL")
        advisory_at = content.index("LIVE DATA ADVISORY")
        assert note_at < retrieval_at < advisory_at < content.index("what is the weather now")
        assert all("RETRIEVAL_SENTINEL" not in m["content"] for m in messages[:-1])

    def test_system_tail_puts_the_parts_in_one_system_message_before_the_user(self):
        handler, _ = _handler(0, budget=ContextBudgetConfig(delta_placement="system_tail"))
        layout = self._build(handler, interruption_note="mid_reply")
        messages = layout.messages

        assert messages[-1] == {"role": "user", "content": "what is the weather now"}
        tail = messages[-2]
        assert tail["role"] == "system"
        assert MID_REPLY_NOTE in tail["content"]
        assert "RETRIEVAL_SENTINEL" in tail["content"]
        assert "LIVE DATA ADVISORY" in tail["content"]
        assert messages[-4:-2] == [
            {"role": "user", "content": "earlier question here"},
            {"role": "assistant", "content": "earlier answer"},
        ]

    def test_no_parts_leaves_the_user_text_unchanged(self):
        handler, _ = _handler(0)
        session = handler.get_or_create_session("p2")
        layout = handler._build_messages(session, 10, pending_user_message="hello there")
        assert layout.messages[-1] == {"role": "user", "content": "hello there"}

    def test_clock_part_renders_first_as_a_labelled_block(self):
        parts = [
            TurnDeltaPart(kind="retrieval", text="RETRIEVED", prunable=True),
            TurnDeltaPart(kind="clock", text="It is 09:30 on Friday.", prunable=False),
        ]
        messages, delta_text = _render_turn_delta(parts, "good morning", "user_prefix")

        assert len(messages) == 1
        content = messages[0]["content"]
        assert content.index("It is 09:30 on Friday.") < content.index("RETRIEVED")
        assert "=== CURRENT TIME ===" in content
        assert content.endswith("good morning")
        assert delta_text in content
        assert "good morning" not in delta_text


class TestInterruptionNote:
    @pytest.mark.parametrize(
        "note, line",
        [
            pytest.param("mid_reply", MID_REPLY_NOTE, id="mid_reply"),
            pytest.param("before_reply", BEFORE_REPLY_NOTE, id="before_reply"),
            pytest.param(None, None, id="none"),
        ],
    )
    def test_renders_the_fixed_line_and_never_the_reply(self, note, line):
        handler, _ = _handler(0)
        session = handler.get_or_create_session("n1")
        session.add_message("user", "tell me a long story")
        session.add_message("assistant", "Once upon a time UNIQUE_REPLY_TEXT")

        layout = handler._build_messages(
            session, 10, pending_user_message="stop that", interruption_note=note
        )
        final = layout.messages[-1]["content"]

        assert "UNIQUE_REPLY_TEXT" not in final
        if line is None:
            assert final == "stop that"
            assert "interrupted" not in final
        else:
            assert line in final
            other = BEFORE_REPLY_NOTE if line == MID_REPLY_NOTE else MID_REPLY_NOTE
            assert other not in final


class TestHistoryWindowSelection:
    def test_planner_disabled_still_selects_the_stepped_window(self):
        handler, _ = _handler(0, budget=ContextBudgetConfig(enabled=False))
        assert handler._budget_planner is None
        session = handler.get_or_create_session("w1")
        for i in range(10):
            session.add_message("user", f"user turn {i}")
            session.add_message("assistant", f"assistant turn {i}")

        layout = handler._build_messages(session, 10, pending_user_message="the next one")

        # 21 messages counting the pending one pass the 20 high-water mark, so
        # the window starts at message 10 (user turn 5).
        history = [m["content"] for m in layout.messages if m["role"] in ("user", "assistant")]
        assert history[0] == "user turn 5"
        assert history[-1] == "the next one"
        assert len(history) == 11
        assert layout.history_compacted == "step"

    def test_planner_receives_the_full_history_and_max_history(self):
        handler, _ = _handler(0)
        captured = {}
        real_plan = handler._budget_planner.plan

        def spy(**kwargs):
            captured.update(kwargs)
            return real_plan(**kwargs)

        handler._budget_planner.plan = spy  # type: ignore[method-assign]
        session = handler.get_or_create_session("w2")
        for i in range(15):
            session.add_message("user", f"u{i}")
            session.add_message("assistant", f"a{i}")

        handler._build_messages(session, 10, pending_user_message="now")

        assert len(captured["history"]) == 31
        assert captured["history"][-1] == {"role": "user", "content": "now"}
        assert captured["max_history"] == 10


class TestPrefixChangeAndStats:
    @pytest.mark.asyncio
    async def test_prompt_stats_carry_every_field(self):
        handler, _ = _handler(1)
        scope = await _run_turn(handler, _question(1))

        stats = scope.prompt_stats
        assert isinstance(stats, PromptStats)
        assert stats.prefix_tokens_est > 0
        assert stats.history_tokens_est == 0
        assert stats.delta_tokens_est > 0
        assert stats.prompt_tokens == 1001
        assert stats.prefix_changed is False
        assert stats.history_compacted == "none"
        assert hasattr(stats, "cached_tokens")

    @pytest.mark.asyncio
    async def test_unchanged_prefix_is_not_flagged(self):
        handler, _ = _handler(2)
        await _run_turn(handler, _question(1))
        scope = await _run_turn(handler, _question(2))
        assert scope.prompt_stats.prefix_changed is False
        assert scope.prompt_stats.history_tokens_est > 0

    @pytest.mark.asyncio
    async def test_persona_eviction_is_named(self, caplog):
        retriever = _ScriptedRetriever()
        handler, _ = _handler(2, retriever=retriever)
        await _run_turn(handler, _question(1))
        retriever.persona = MistContext(
            display_name="MIST", pronouns="she/her", self_concept="A changed self concept."
        )
        handler._mist_context_cache.pop("s1")

        with caplog.at_level(logging.INFO, logger="backend.chat.conversation_handler"):
            scope = await _run_turn(handler, _question(2))

        assert scope.prompt_stats.prefix_changed is True
        lines = [r.getMessage() for r in caplog.records if "prefix changed" in r.getMessage()]
        assert len(lines) == 1
        assert "persona" in lines[0]
        assert "user profile" not in lines[0]

    @pytest.mark.asyncio
    async def test_profile_file_change_is_named(self, tmp_path, caplog):
        users = tmp_path / "users"
        users.mkdir()
        profile = users / "user.md"
        profile.write_text("---\ntype: mist-user\n---\n\nLikes tea.\n", encoding="utf-8")
        handler, _ = _handler(2, vault_root=tmp_path)
        await _run_turn(handler, _question(1))
        profile.write_text("---\ntype: mist-user\n---\n\nLikes coffee.\n", encoding="utf-8")

        with caplog.at_level(logging.INFO, logger="backend.chat.conversation_handler"):
            scope = await _run_turn(handler, _question(2))

        assert scope.prompt_stats.prefix_changed is True
        lines = [r.getMessage() for r in caplog.records if "prefix changed" in r.getMessage()]
        assert len(lines) == 1
        assert "user profile" in lines[0]
        assert "persona" not in lines[0]

    @pytest.mark.asyncio
    async def test_prefix_is_tracked_per_session(self):
        retriever = _ScriptedRetriever()
        handler, _ = _handler(2, retriever=retriever)
        await _run_turn(handler, _question(1), session_id="a")
        retriever.persona = MistContext(display_name="MIST", pronouns="", self_concept="Other.")
        scope = await _run_turn(handler, _question(1), session_id="b")
        # Session b's first turn has nothing to compare against.
        assert scope.prompt_stats.prefix_changed is False


def test_conversation_session_is_untouched_by_build(tmp_path):
    """`_build_messages` reads history; it never appends the pending message."""
    handler, _ = _handler(0)
    session = ConversationSession(session_id="ro", user_id="User")
    handler._build_messages(session, 10, pending_user_message="hi there you")
    assert session.messages == []
