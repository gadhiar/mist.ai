"""Unit tests for backend.chat.context_budget.

Covers:
- ApproximateTokenCounter: char/3.6 heuristic, per-message overhead,
  tool_calls overhead, empty text handling.
- SlidingWindowStrategy: most-recent-fits semantics, chronological ordering
  preserved, zero-budget edge case.
- score_fact_for_retrieval_pruning: similarity × confidence × recency with
  exponential decay; missing fields fall back to safe defaults.
- ContextBudgetPlanner: fixed-cost arithmetic, flex-budget split, retrieval
  full-text happy path, retrieval priority pruning when over budget, history
  strategy invocation, hard-stop graceful degradation when fixed > total.

Spec: ~/.claude/plans/cluster-execution-roadmap.md Cluster 6.
"""

from __future__ import annotations

import os
import random
from datetime import UTC, datetime, timedelta
from unittest import mock

import pytest

from backend.chat.context_budget import (
    ApproximateTokenCounter,
    BudgetPlan,
    ContextBudgetPlanner,
    HistorySelection,
    SlidingWindowStrategy,
    StableWindowStrategy,
    TurnDeltaPart,
    get_history_strategy,
    score_fact_for_retrieval_pruning,
)
from backend.knowledge.config import ContextBudgetConfig
from backend.knowledge.models import (
    ConversationSession,
    RetrievalResult,
    RetrievedFact,
    fact_ref,
)

# ---------------------------------------------------------------------------
# Plan identity invariants, checked on every plan() in this module
# ---------------------------------------------------------------------------


def _assert_plan_identities(plan: BudgetPlan, retrieval_result: RetrievalResult | None) -> None:
    assert plan.facts_kept == len(plan.kept_fact_refs)
    assert plan.facts_dropped == len(plan.dropped_fact_refs)
    # The planner re-renders exactly when the scored path ran, which is when it
    # emits text other than the retriever's own block. Passing the block
    # through is not a re-render; emitting "" because no fact fits is one.
    passed_through = (
        retrieval_result is not None
        and plan.pruned_retrieval_text == retrieval_result.formatted_context
    )
    assert plan.retrieval_rerendered == (
        plan.pruned_retrieval_text is not None and not passed_through
    )


@pytest.fixture(autouse=True)
def _check_every_plan(monkeypatch):
    """Spy on `plan()` so every planner test in this module, old and new, also
    proves the kept/dropped refs agree with the counts. Removing the fixture
    drops that coverage from the pre-existing tests silently.
    """
    original = ContextBudgetPlanner.plan

    def checked(self, **kwargs):
        plan = original(self, **kwargs)
        _assert_plan_identities(plan, kwargs.get("retrieval_result"))
        return plan

    monkeypatch.setattr(ContextBudgetPlanner, "plan", checked)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _make_fact(
    *,
    subject: str = "User",
    predicate: str = "USES",
    object: str = "Python",
    similarity: float = 0.8,
    graph_distance: int = 0,
    properties: dict | None = None,
    created_at=None,
) -> RetrievedFact:
    return RetrievedFact(
        subject=subject,
        subject_type="Person",
        predicate=predicate,
        object=object,
        object_type="Technology",
        properties=properties or {},
        similarity_score=similarity,
        graph_distance=graph_distance,
        created_at=created_at,
    )


def _make_retrieval_result(
    facts: list[RetrievedFact],
    *,
    query: str = "test query",
    formatted: str | None = None,
    intent: str = "hybrid",
) -> RetrievalResult:
    if formatted is None:
        lines = [f"Relevant knowledge from your graph (query: '{query}'):"]
        for f in facts:
            lines.append(f"- {f.subject} -[{f.predicate}]-> {f.object}")
        formatted = "\n".join(lines)
    return RetrievalResult(
        query=query,
        user_id="User",
        facts=facts,
        entities_found=len(facts),
        total_facts=len(facts),
        formatted_context=formatted,
        retrieval_time_ms=1.0,
        vector_search_time_ms=0.0,
        graph_traversal_time_ms=0.0,
        config_used={},
        intent=intent,
    )


# ---------------------------------------------------------------------------
# ApproximateTokenCounter
# ---------------------------------------------------------------------------


class TestApproximateTokenCounter:
    def test_empty_text_counts_zero(self):
        counter = ApproximateTokenCounter()
        assert counter.count("") == 0
        assert counter.count(None or "") == 0

    def test_short_text_uses_char_ratio(self):
        counter = ApproximateTokenCounter()
        # 36 chars / 3.6 = 10 tokens exactly
        assert counter.count("a" * 36) == 10

    def test_non_round_chars_rounds_up(self):
        counter = ApproximateTokenCounter()
        # 10 chars / 3.6 = 2.78 -> ceil = 3
        assert counter.count("a" * 10) == 3

    def test_count_message_includes_overhead(self):
        counter = ApproximateTokenCounter()
        # Content=36 chars -> 10 content tokens + 4 overhead = 14
        message = {"role": "user", "content": "a" * 36}
        assert counter.count_message(message) == 14

    def test_count_message_handles_none_content(self):
        counter = ApproximateTokenCounter()
        message = {"role": "assistant", "content": None}
        # 0 content tokens + 4 overhead = 4
        assert counter.count_message(message) == 4

    def test_tool_calls_add_overhead_per_call(self):
        counter = ApproximateTokenCounter()
        message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "tc-1", "function": {"name": "a", "arguments": "{}"}},
                {"id": "tc-2", "function": {"name": "b", "arguments": "{}"}},
            ],
        }
        # 0 content + 4 base + 20*2 tool overhead = 44
        assert counter.count_message(message) == 44

    def test_count_messages_sums_across_list(self):
        counter = ApproximateTokenCounter()
        messages = [
            {"role": "user", "content": "a" * 36},  # 14
            {"role": "assistant", "content": "b" * 18},  # 9
        ]
        assert counter.count_messages(messages) == 14 + 9


# ---------------------------------------------------------------------------
# SlidingWindowStrategy
# ---------------------------------------------------------------------------


class TestSlidingWindowStrategy:
    def test_empty_history_returns_empty(self):
        strategy = SlidingWindowStrategy()
        counter = ApproximateTokenCounter()
        assert strategy.select([], 1000, counter) == HistorySelection([], 0, "none")

    def test_zero_budget_returns_empty(self):
        strategy = SlidingWindowStrategy()
        counter = ApproximateTokenCounter()
        history = [{"role": "user", "content": "hi"}]
        assert strategy.select(history, 0, counter) == HistorySelection([], 1, "budget")

    def test_fits_returns_all_in_order(self):
        strategy = SlidingWindowStrategy()
        counter = ApproximateTokenCounter()
        history = [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
        ]
        result = strategy.select(history, 1000, counter)
        assert [m["content"] for m in result.messages] == ["one", "two", "three"]
        assert result.start_index == 0
        assert result.compacted == "none"

    def test_keeps_most_recent_fits_preserve_order(self):
        strategy = SlidingWindowStrategy()
        counter = ApproximateTokenCounter()
        # Each message = ~14 tokens (10 content + 4 overhead).
        history = [
            {"role": "user", "content": "a" * 36},  # 14
            {"role": "assistant", "content": "b" * 36},  # 14
            {"role": "user", "content": "c" * 36},  # 14
            {"role": "assistant", "content": "d" * 36},  # 14
        ]
        # Budget = 30 -> fits last 2 messages (28 tokens)
        result = strategy.select(history, 30, counter)
        assert len(result.messages) == 2
        assert [m["content"][0] for m in result.messages] == ["c", "d"]  # chronological
        assert result.start_index == 2
        assert result.compacted == "budget"

    def test_keeps_only_last_when_budget_too_small_for_older(self):
        strategy = SlidingWindowStrategy()
        counter = ApproximateTokenCounter()
        history = [
            {"role": "user", "content": "a" * 36},  # 14
            {"role": "assistant", "content": "b" * 36},  # 14
        ]
        # Budget = 15 -> one message (14) fits; adding second would exceed
        result = strategy.select(history, 15, counter)
        assert len(result.messages) == 1
        assert result.messages[0]["content"][0] == "b"
        assert result.start_index == 1

    def test_max_messages_caps_before_the_budget_without_counting_as_compaction(self):
        strategy = SlidingWindowStrategy()
        counter = ApproximateTokenCounter()
        history = _alternating_history(9)

        result = strategy.select(history, 10_000, counter, max_messages=4)

        assert result.messages == history[-4:]
        assert result.start_index == 5
        assert result.compacted == "none"

    def test_full_history_reproduces_the_pre_sliced_selection(self):
        """Given the session's full history and `max_messages`, the selection
        equals what the old list-returning strategy chose from
        `get_history(max_messages)`, over 40 random histories and budgets.
        """
        rng = random.Random(1717)
        strategy = SlidingWindowStrategy()
        counter = ApproximateTokenCounter()
        for _ in range(40):
            session = ConversationSession(session_id="eq", user_id="U")
            for i in range(rng.randint(0, 60)):
                role = "user" if i % 2 == 0 else "assistant"
                session.add_message(role, "w" * rng.randint(0, 400))
            max_messages = rng.choice([None, 0, 1, 5, 10, 20])
            budget = rng.randint(0, 3000)

            expected = _old_sliding_select(session.get_history(max_messages), budget, counter)
            result = strategy.select(
                session.get_history(None), budget, counter, max_messages=max_messages
            )

            assert result.messages == expected
            full = session.get_history(None)
            assert result.messages == full[result.start_index :]


def _alternating_history(count: int, *, size: int = 36) -> list[dict]:
    """`count` messages alternating user, assistant, each `size` characters."""
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"{i:03d}" + "x" * (size - 3)}
        for i in range(count)
    ]


def _old_sliding_select(history: list[dict], budget: int, counter) -> list[dict]:
    """The sliding-window selection as it was before `HistorySelection`."""
    if budget <= 0 or not history:
        return []
    selected: list[dict] = []
    used = 0
    for msg in reversed(history):
        cost = counter.count_message(msg)
        if used + cost > budget:
            break
        selected.append(msg)
        used += cost
    selected.reverse()
    return selected


# ---------------------------------------------------------------------------
# StableWindowStrategy
# ---------------------------------------------------------------------------


class TestStableWindowStrategy:
    def _select(self, history, budget=1_000_000, *, max_messages=10, factor=2.0):
        return StableWindowStrategy().select(
            history,
            budget,
            ApproximateTokenCounter(),
            max_messages=max_messages,
            high_water_factor=factor,
        )

    def test_window_steps_only_past_the_high_water_mark(self):
        """Turn n holds 2n-1 messages (its own user message last). At 10
        messages and factor 2.0 the start moves on turns 11, 16, 21 and 26,
        in steps of 10 counted from the first message.
        """
        starts = {}
        for turn in range(1, 31):
            history = _alternating_history(2 * turn - 1)
            selection = self._select(history)
            assert selection.messages == history[selection.start_index :]
            starts[turn] = selection.start_index
        moved = [t for t in range(2, 31) if starts[t] != starts[t - 1]]
        assert moved == [11, 16, 21, 26]
        assert starts[10] == 0
        assert starts[11] == 10
        assert starts[16] == 20
        assert starts[30] == 40

    def test_window_never_exceeds_high_water_nor_drops_below_max_messages(self):
        for count in range(1, 80):
            selection = self._select(_alternating_history(count))
            assert min(count, 10) <= len(selection.messages) <= 20

    def test_compacted_reports_none_then_step(self):
        assert self._select(_alternating_history(19)).compacted == "none"
        assert self._select(_alternating_history(21)).compacted == "step"

    def test_starts_on_a_whole_exchange_with_odd_max_messages(self):
        for count in range(1, 60):
            selection = self._select(_alternating_history(count), max_messages=5)
            assert selection.start_index % 2 == 0
            assert len(selection.messages) >= min(count, 5)

    def test_factor_one_keeps_the_last_max_messages(self):
        # The step rule alone would start at 20 (5 messages); the floor of 10
        # messages moves it back to 15, and the exchange that 15 belongs to
        # starts at 14.
        history = _alternating_history(25)
        selection = self._select(history, factor=1.0)
        assert selection.start_index == 14
        assert selection.messages == history[14:]

    def test_over_budget_jumps_further_steps(self):
        # Factor 3.0, 35 messages of 14 tokens each: the step rule starts the
        # window at 10 (25 messages, 350 tokens). One more step, to 20, keeps
        # 15 messages (210 tokens), which fits the budget of 215.
        history = _alternating_history(35)
        selection = self._select(history, budget=215, factor=3.0)
        assert selection.start_index == 20
        assert selection.compacted == "budget"

    def test_drops_whole_exchanges_when_one_step_does_not_fit(self):
        history = _alternating_history(21)  # step start 10, 11 messages
        # 7 messages fit (98 tokens); whole exchanges keep 7 (start at 14).
        selection = self._select(history, budget=100)
        assert selection.start_index == 14
        assert selection.messages == history[14:]
        assert selection.compacted == "budget"

    def test_budget_compaction_without_max_messages(self):
        history = _alternating_history(9)
        selection = self._select(history, budget=60, max_messages=None)
        # 4 messages fit (56 tokens); the front exchange boundary at 6
        # leaves 3, the boundary at 4 would need 5.
        assert selection.start_index == 6
        assert selection.compacted == "budget"
        assert self._select(history, max_messages=None).compacted == "none"

    def test_empty_and_zero_budget(self):
        assert self._select([]) == HistorySelection([], 0, "none")
        assert self._select(_alternating_history(3), budget=0) == HistorySelection([], 3, "budget")

    def test_is_a_pure_function_of_history_and_budget(self):
        history = _alternating_history(47)
        first = self._select(history, budget=300)
        assert self._select(list(history), budget=300) == first
        assert StableWindowStrategy().select(
            history, 300, ApproximateTokenCounter(), max_messages=10
        ) == self._select(history, budget=300)


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------


class TestHistoryStrategyRegistry:
    def test_sliding_window_returns_sliding_strategy(self):
        assert isinstance(get_history_strategy("sliding_window"), SlidingWindowStrategy)

    def test_unknown_name_falls_back_to_sliding_window(self):
        # Should log a warning and return the default.
        strategy = get_history_strategy("nonexistent_strategy")
        assert isinstance(strategy, SlidingWindowStrategy)

    def test_stable_window_is_registered(self):
        assert isinstance(get_history_strategy("stable_window"), StableWindowStrategy)


# ---------------------------------------------------------------------------
# ContextBudgetConfig prompt-layout settings
# ---------------------------------------------------------------------------


class TestContextBudgetConfigPromptLayout:
    def test_defaults(self):
        config = ContextBudgetConfig()
        assert config.turn_delta_max_tokens == 1000
        assert config.delta_placement == "user_prefix"
        assert config.history_high_water_factor == 2.0
        assert config.history_strategy == "stable_window"

    def test_from_env_defaults(self):
        keys = (
            "MIST_CTX_BUDGET_TURN_DELTA_MAX_TOKENS",
            "MIST_CTX_BUDGET_DELTA_PLACEMENT",
            "MIST_CTX_BUDGET_HISTORY_HIGH_WATER",
            "MIST_CTX_BUDGET_HISTORY_STRATEGY",
        )
        env = {k: v for k, v in os.environ.items() if k not in keys}
        with mock.patch.dict(os.environ, env, clear=True):
            config = ContextBudgetConfig.from_env()
        assert config.turn_delta_max_tokens == 1000
        assert config.delta_placement == "user_prefix"
        assert config.history_high_water_factor == 2.0
        assert config.history_strategy == "stable_window"

    def test_from_env_reads_each_variable(self):
        with mock.patch.dict(
            os.environ,
            {
                "MIST_CTX_BUDGET_TURN_DELTA_MAX_TOKENS": "600",
                "MIST_CTX_BUDGET_DELTA_PLACEMENT": "system_tail",
                "MIST_CTX_BUDGET_HISTORY_HIGH_WATER": "3.5",
                "MIST_CTX_BUDGET_HISTORY_STRATEGY": "sliding_window",
            },
        ):
            config = ContextBudgetConfig.from_env()
        assert config.turn_delta_max_tokens == 600
        assert config.delta_placement == "system_tail"
        assert config.history_high_water_factor == 3.5
        assert config.history_strategy == "sliding_window"

    def test_invalid_placement_raises(self):
        with pytest.raises(ValueError, match="delta_placement"):
            ContextBudgetConfig(delta_placement="before_history")

    def test_invalid_placement_from_env_raises(self):
        with (
            mock.patch.dict(os.environ, {"MIST_CTX_BUDGET_DELTA_PLACEMENT": "middle"}),
            pytest.raises(ValueError, match="delta_placement"),
        ):
            ContextBudgetConfig.from_env()

    def test_factor_below_one_raises(self):
        with pytest.raises(ValueError, match="history_high_water_factor"):
            ContextBudgetConfig(history_high_water_factor=0.99)

    def test_factor_of_exactly_one_is_accepted(self):
        assert ContextBudgetConfig(history_high_water_factor=1.0).history_high_water_factor == 1.0


# ---------------------------------------------------------------------------
# score_fact_for_retrieval_pruning
# ---------------------------------------------------------------------------


class TestScoreFactForPruning:
    def test_similarity_alone_when_other_fields_absent(self):
        fact = _make_fact(similarity=0.7)
        # confidence=1.0, recency=1.0 (no created_at) -> score = 0.7
        assert score_fact_for_retrieval_pruning(fact) == pytest.approx(0.7)

    def test_confidence_multiplied_in(self):
        fact = _make_fact(similarity=0.8, properties={"confidence": 0.5})
        # 0.8 × 0.5 × 1.0 = 0.4
        assert score_fact_for_retrieval_pruning(fact) == pytest.approx(0.4)

    def test_recency_decay_at_one_half_life(self):
        now = datetime(2026, 4, 21, tzinfo=UTC)
        one_half_life_ago = now - timedelta(days=30)
        fact = _make_fact(similarity=1.0, created_at=one_half_life_ago.isoformat())
        # similarity × confidence × 0.5 = 0.5
        score = score_fact_for_retrieval_pruning(fact, now=now, recency_half_life_days=30.0)
        assert score == pytest.approx(0.5, rel=0.01)

    def test_recency_decay_for_fresh_fact_is_near_one(self):
        now = datetime(2026, 4, 21, tzinfo=UTC)
        fact = _make_fact(similarity=0.8, created_at=now.isoformat())
        score = score_fact_for_retrieval_pruning(fact, now=now, recency_half_life_days=30.0)
        assert score == pytest.approx(0.8, rel=0.01)

    def test_invalid_created_at_falls_back_to_recency_one(self):
        fact = _make_fact(similarity=0.9, created_at="not a date")
        score = score_fact_for_retrieval_pruning(fact)
        assert score == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# ContextBudgetPlanner
# ---------------------------------------------------------------------------


def _default_config(**overrides) -> ContextBudgetConfig:
    defaults = {
        "context_window": 1000,
        "output_reserve_tokens": 100,
        "safety_margin_tokens": 50,
        "retrieval_budget_ratio": 0.4,
        "turn_delta_max_tokens": 200,
    }
    defaults.update(overrides)
    return ContextBudgetConfig(**defaults)


def _advisory(text: str = "Suggested tools: weather") -> TurnDeltaPart:
    return TurnDeltaPart(kind="live_advisory", text=text, prunable=False)


class TestPlannerTurnDelta:
    """The stable prefix and tools are fixed cost; the delta has its own reserve."""

    def _plan(self, planner, **overrides):
        kwargs = {
            "persona_text": "p" * 360,
            "static_text": "s" * 360,
            "retrieval_result": None,
            "history": _alternating_history(30),
            "tools": None,
            "max_output_tokens": 100,
            "max_history": 10,
        }
        kwargs.update(overrides)
        return planner.plan(**kwargs)

    def test_history_budget_ignores_advisory_and_retrieval(self):
        planner = ContextBudgetPlanner(config=_default_config(context_window=2000))
        bare = self._plan(planner)
        busy = self._plan(
            planner,
            retrieval_result=_make_retrieval_result([_make_fact()]),
            delta_parts=[
                _advisory("x" * 300),
                TurnDeltaPart(kind="interruption", text="cut off", prunable=False),
            ],
        )

        assert busy.history_budget == bare.history_budget
        assert busy.fixed_cost == bare.fixed_cost
        assert busy.pruned_history == bare.pruned_history
        # The prefix alone is fixed cost: 100 + 100 content tokens + 2 * 4 overhead.
        assert bare.fixed_cost == 208
        assert bare.delta_budget == 200
        assert bare.history_budget == bare.total_budget - 208 - 200

    def test_retrieval_gets_the_delta_left_after_the_fixed_parts(self):
        planner = ContextBudgetPlanner(
            config=_default_config(
                context_window=4000, turn_delta_max_tokens=100, retrieval_budget_ratio=0.9
            )
        )
        facts = [_make_fact(subject=f"S{i}", similarity=1.0 - i / 100) for i in range(40)]
        plan = self._plan(
            planner,
            retrieval_result=_make_retrieval_result(facts, formatted="BIG " * 400),
            delta_parts=[_advisory("a" * 108)],  # 30 content + 4 overhead
        )

        assert plan.retrieval_budget == 100 - 34
        assert 0 < plan.retrieval_used <= plan.retrieval_budget
        assert plan.facts_dropped > 0
        assert "S0" in plan.pruned_retrieval_text

    def test_live_advisory_keyword_charges_like_a_delta_part(self):
        planner = ContextBudgetPlanner(
            config=_default_config(
                context_window=4000, turn_delta_max_tokens=100, retrieval_budget_ratio=0.9
            )
        )
        retrieval = _make_retrieval_result([_make_fact()])
        by_keyword = self._plan(planner, retrieval_result=retrieval, live_advisory_text="a" * 108)
        by_part = self._plan(
            planner, retrieval_result=retrieval, delta_parts=[_advisory("a" * 108)]
        )

        assert by_keyword.retrieval_budget == by_part.retrieval_budget == 100 - 34
        assert by_keyword.history_budget == by_part.history_budget

    def test_retrieval_is_also_capped_by_the_flex_ratio(self):
        planner = ContextBudgetPlanner(
            config=_default_config(
                context_window=1000, turn_delta_max_tokens=500, retrieval_budget_ratio=0.1
            )
        )
        plan = self._plan(planner, retrieval_result=_make_retrieval_result([_make_fact()]))

        flex = plan.total_budget - plan.fixed_cost
        assert plan.retrieval_budget == int(flex * 0.1)

    def test_delta_reserve_never_exceeds_the_flex_budget(self):
        planner = ContextBudgetPlanner(
            config=_default_config(context_window=1000, turn_delta_max_tokens=5000)
        )
        plan = self._plan(planner)
        assert plan.delta_budget == plan.total_budget - plan.fixed_cost
        assert plan.history_budget == 0


class TestContextBudgetPlannerRejectsUnresolvedWindow:
    """MIS-171 T4: an unresolved 'auto' sentinel must never reach plan() arithmetic."""

    def test_string_auto_context_window_raises_value_error(self):
        config = _default_config(context_window="auto")
        with pytest.raises(ValueError, match="resolved int"):
            ContextBudgetPlanner(config=config)

    def test_int_context_window_constructs_fine(self):
        config = _default_config(context_window=1000)
        ContextBudgetPlanner(config=config)  # must not raise


class TestContextBudgetPlanner:
    def test_fits_with_retrieval_and_history_inside_budget(self):
        planner = ContextBudgetPlanner(config=_default_config())
        # Fixed: persona 36 chars (10) + static 36 (10) + overhead 4*2 = ~28
        # Total budget: 1000 - 100 (max_out) - 100 - 50 = 750
        # flex = 750 - ~28 = ~722
        # retrieval = ~288, history = ~434
        result = _make_retrieval_result([_make_fact()])
        history = [{"role": "user", "content": "hi"}]

        plan = planner.plan(
            persona_text="p" * 36,
            static_text="s" * 36,
            retrieval_result=result,
            history=history,
            tools=None,
            max_output_tokens=100,
        )

        assert plan.fits is True
        assert plan.total_budget == 750
        assert plan.retrieval_budget > 0
        assert plan.history_budget > 0
        assert plan.pruned_retrieval_text == result.formatted_context
        assert plan.pruned_history == history
        assert plan.facts_dropped == 0

    def test_no_retrieval_result_produces_none_pruned_text(self):
        planner = ContextBudgetPlanner(config=_default_config())
        plan = planner.plan(
            persona_text="p",
            static_text="s",
            retrieval_result=None,
            history=[],
            tools=None,
            max_output_tokens=100,
        )
        assert plan.pruned_retrieval_text is None
        assert plan.facts_kept == 0
        assert plan.facts_dropped == 0

    def test_extra_fixed_texts_are_charged_as_fixed_cost(self):
        # deep review phase1-conversation-1: the always-injected conventions
        # and user-profile blocks bypassed fixed-cost accounting, eating the
        # quality envelope while the planner believed pruned content fit.
        planner = ContextBudgetPlanner(config=_default_config())
        kwargs = {
            "persona_text": "p",
            "static_text": "s",
            "retrieval_result": None,
            "history": [],
            "tools": None,
            "max_output_tokens": 100,
        }

        base = planner.plan(**kwargs)
        with_extra = planner.plan(**kwargs, extra_fixed_texts=["X" * 400, "Y" * 200])

        assert with_extra.fixed_cost > base.fixed_cost
        assert with_extra.history_budget <= base.history_budget

    def test_fixed_over_budget_returns_not_fits(self):
        # Huge persona + tiny window forces overflow.
        planner = ContextBudgetPlanner(config=_default_config(context_window=200))
        plan = planner.plan(
            persona_text="p" * 5000,
            static_text="s",
            retrieval_result=_make_retrieval_result([_make_fact()]),
            history=[{"role": "user", "content": "x"}],
            tools=None,
            max_output_tokens=50,
        )
        assert plan.fits is False
        assert plan.retrieval_budget == 0
        assert plan.history_budget == 0
        assert plan.pruned_retrieval_text is None
        assert plan.pruned_history == []
        # facts_dropped reports the full retrieval pool we had to discard.
        assert plan.facts_dropped == 1

    def test_retrieval_pruning_keeps_top_scored_facts(self):
        # Very small retrieval budget forces pruning.
        cfg = _default_config(
            context_window=500,
            output_reserve_tokens=50,
            safety_margin_tokens=10,
            retrieval_budget_ratio=0.1,
        )
        planner = ContextBudgetPlanner(config=cfg)

        high = _make_fact(subject="H", similarity=0.95)
        low_1 = _make_fact(subject="L1", similarity=0.1)
        low_2 = _make_fact(subject="L2", similarity=0.1)
        low_3 = _make_fact(subject="L3", similarity=0.1)

        # Create a formatted context longer than the small retrieval budget
        # so the planner falls into the pruning branch.
        facts = [high, low_1, low_2, low_3]
        long_formatted = "BIG HEADER " * 200  # force full_cost > budget
        result = _make_retrieval_result(facts, formatted=long_formatted)

        plan = planner.plan(
            persona_text="p",
            static_text="s",
            retrieval_result=result,
            history=[],
            tools=None,
            max_output_tokens=50,
        )

        assert plan.fits is True
        # The highest-scoring fact (subject="H") must survive.
        assert plan.pruned_retrieval_text is not None
        assert "H" in plan.pruned_retrieval_text
        # At least one lower-scoring fact got dropped.
        assert plan.facts_dropped >= 1

    def test_pruner_uses_vault_header_for_historical_intent(self):
        """Mirror the retriever-level header reframe at the budget-pruner
        level. When intent='historical', the budget pruner must NOT emit
        'Relevant knowledge from your graph' framing — that biases the
        model into treating vault chunks as authoritative graph results
        per ADR-010 invariant 4.
        """
        cfg = _default_config(
            context_window=500,
            output_reserve_tokens=50,
            safety_margin_tokens=10,
            retrieval_budget_ratio=0.1,
        )
        planner = ContextBudgetPlanner(config=cfg)

        high = _make_fact(subject="H", similarity=0.95)
        low_1 = _make_fact(subject="L1", similarity=0.1)
        long_formatted = "BIG HEADER " * 200  # force pruning branch
        result = _make_retrieval_result(
            [high, low_1], formatted=long_formatted, intent="historical"
        )

        plan = planner.plan(
            persona_text="p",
            static_text="s",
            retrieval_result=result,
            history=[],
            tools=None,
            max_output_tokens=50,
        )

        assert plan.pruned_retrieval_text is not None
        assert "from your graph" not in plan.pruned_retrieval_text, (
            f"vault-only retrieval must not emit graph framing in budget "
            f"pruner; got: {plan.pruned_retrieval_text!r}"
        )
        assert "vault" in plan.pruned_retrieval_text.lower(), (
            f"expected vault framing in pruner header; got: " f"{plan.pruned_retrieval_text!r}"
        )

    def test_history_strategy_invoked_with_budget(self):
        planner = ContextBudgetPlanner(config=_default_config())
        many_messages = [{"role": "user", "content": "a" * 100} for _ in range(50)]

        plan = planner.plan(
            persona_text="p",
            static_text="s",
            retrieval_result=None,
            history=many_messages,
            tools=None,
            max_output_tokens=100,
        )

        # History must have been pruned (can't fit 50 * ~32 tokens = 1600).
        assert len(plan.pruned_history) < len(many_messages)
        # Chronological order preserved.
        assert plan.pruned_history == many_messages[-len(plan.pruned_history) :]

    def test_plan_passes_max_history_and_factor_to_the_strategy(self):
        planner = ContextBudgetPlanner(
            config=_default_config(context_window=100_000, history_high_water_factor=3.0)
        )
        history = _alternating_history(31)

        plan = planner.plan(
            persona_text="p",
            static_text="s",
            retrieval_result=None,
            history=history,
            tools=None,
            max_output_tokens=100,
            max_history=10,
        )

        # Factor 3.0 at 10 messages: 31 messages exceed 30, so one step.
        assert plan.pruned_history == history[10:]
        assert plan.history_compacted == "step"

    def test_tools_schema_charged_to_fixed_cost(self):
        config = _default_config(
            context_window=400,
            output_reserve_tokens=50,
            safety_margin_tokens=10,
        )
        planner = ContextBudgetPlanner(config=config)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "x" * 50,
                    "description": "d" * 200,
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        plan = planner.plan(
            persona_text="",
            static_text="",
            retrieval_result=None,
            history=[],
            tools=tools,
            max_output_tokens=50,
        )
        # The fixed_cost should include the serialized tool schema.
        assert plan.fixed_cost > 50  # tool JSON alone exceeds ~50 tokens


class TestBudgetPlanDataclass:
    def test_plan_is_frozen(self):
        plan = BudgetPlan(
            fits=True,
            total_budget=100,
            fixed_cost=10,
            retrieval_budget=30,
            history_budget=60,
            pruned_retrieval_text="ok",
            pruned_history=[],
            retrieval_used=5,
            history_used=0,
            facts_kept=1,
            facts_dropped=0,
        )
        with pytest.raises((AttributeError, TypeError)):
            plan.fits = False  # type: ignore[misc]

    def test_identity_fields_default_to_empty_and_false(self):
        plan = BudgetPlan(
            fits=True,
            total_budget=100,
            fixed_cost=10,
            retrieval_budget=30,
            history_budget=60,
            pruned_retrieval_text=None,
            pruned_history=[],
            retrieval_used=0,
            history_used=0,
            facts_kept=0,
            facts_dropped=0,
        )

        assert plan.kept_fact_refs == ()
        assert plan.dropped_fact_refs == ()
        assert plan.retrieval_rerendered is False


# ---------------------------------------------------------------------------
# Kept and dropped fact refs
# ---------------------------------------------------------------------------


def _pruning_config(retrieval_budget_ratio: float = 0.1) -> ContextBudgetConfig:
    return _default_config(
        context_window=500,
        output_reserve_tokens=50,
        safety_margin_tokens=10,
        retrieval_budget_ratio=retrieval_budget_ratio,
    )


def _plan(planner: ContextBudgetPlanner, result: RetrievalResult | None, **overrides):
    kwargs = {
        "persona_text": "p",
        "static_text": "s",
        "retrieval_result": result,
        "live_advisory_text": None,
        "history": [],
        "tools": None,
        "max_output_tokens": 50,
    }
    kwargs.update(overrides)
    return planner.plan(**kwargs)


def _ranked_facts() -> list[RetrievedFact]:
    return [
        _make_fact(subject="L1", similarity=0.1),
        _make_fact(subject="H", similarity=0.95),
        _make_fact(subject="L2", similarity=0.2),
        _make_fact(subject="L3", similarity=0.3),
    ]


class TestBudgetPlanFactRefs:
    def test_pass_through_keeps_every_fact_in_retrieval_order(self):
        planner = ContextBudgetPlanner(config=_default_config())
        facts = _ranked_facts()
        result = _make_retrieval_result(facts)

        plan = _plan(planner, result, max_output_tokens=100)

        assert plan.kept_fact_refs == tuple(fact_ref(f) for f in facts)
        assert plan.dropped_fact_refs == ()
        assert plan.retrieval_rerendered is False

    def test_scored_path_reports_kept_in_score_order_and_the_rest_dropped(self):
        planner = ContextBudgetPlanner(config=_pruning_config())
        facts = _ranked_facts()
        result = _make_retrieval_result(facts, formatted="BIG HEADER " * 200)

        plan = _plan(planner, result)

        by_score = [facts[1], facts[3], facts[2], facts[0]]
        kept = plan.facts_kept
        assert 0 < kept < len(facts)
        assert plan.kept_fact_refs == tuple(fact_ref(f) for f in by_score[:kept])
        assert plan.dropped_fact_refs == tuple(fact_ref(f) for f in by_score[kept:])
        assert plan.retrieval_rerendered is True

    def test_scored_path_that_keeps_nothing_still_counts_as_rerendered(self):
        planner = ContextBudgetPlanner(config=_pruning_config(retrieval_budget_ratio=0.01))
        facts = _ranked_facts()
        result = _make_retrieval_result(facts, formatted="BIG HEADER " * 200)

        plan = _plan(planner, result)

        assert plan.pruned_retrieval_text == ""
        assert plan.kept_fact_refs == ()
        assert len(plan.dropped_fact_refs) == len(facts)
        assert plan.retrieval_rerendered is True

    def test_plan_that_does_not_fit_drops_every_fact(self):
        planner = ContextBudgetPlanner(config=_default_config(context_window=200))
        facts = _ranked_facts()
        result = _make_retrieval_result(facts)

        plan = _plan(planner, result, persona_text="p" * 5000)

        assert plan.fits is False
        assert plan.kept_fact_refs == ()
        assert plan.dropped_fact_refs == tuple(fact_ref(f) for f in facts)
        assert plan.retrieval_rerendered is False

    def test_no_retrieval_reports_no_refs(self):
        planner = ContextBudgetPlanner(config=_default_config())

        plan = _plan(planner, None)

        assert plan.kept_fact_refs == ()
        assert plan.dropped_fact_refs == ()
        assert plan.retrieval_rerendered is False

    def test_rerender_leaves_the_prompt_text_unchanged(self):
        planner = ContextBudgetPlanner(config=_pruning_config())
        result = _make_retrieval_result(_ranked_facts(), formatted="BIG HEADER " * 200)

        plan = _plan(planner, result)

        assert plan.pruned_retrieval_text == (
            "Relevant knowledge from your graph (query: 'test query'):\n"
            + "".join(
                f"- {s} -[USES]-> Python\n" for s in ("H", "L3", "L2", "L1")[: plan.facts_kept]
            )
        )
