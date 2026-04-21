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

from datetime import UTC, datetime, timedelta

import pytest

from backend.chat.context_budget import (
    ApproximateTokenCounter,
    BudgetPlan,
    ContextBudgetPlanner,
    SlidingWindowStrategy,
    get_history_strategy,
    score_fact_for_retrieval_pruning,
)
from backend.knowledge.config import ContextBudgetConfig
from backend.knowledge.models import RetrievalResult, RetrievedFact

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
        intent="hybrid",
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
        assert strategy.select([], 1000, counter) == []

    def test_zero_budget_returns_empty(self):
        strategy = SlidingWindowStrategy()
        counter = ApproximateTokenCounter()
        history = [{"role": "user", "content": "hi"}]
        assert strategy.select(history, 0, counter) == []

    def test_fits_returns_all_in_order(self):
        strategy = SlidingWindowStrategy()
        counter = ApproximateTokenCounter()
        history = [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
        ]
        result = strategy.select(history, 1000, counter)
        assert [m["content"] for m in result] == ["one", "two", "three"]

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
        assert len(result) == 2
        assert [m["content"][0] for m in result] == ["c", "d"]  # chronological

    def test_keeps_only_last_when_budget_too_small_for_older(self):
        strategy = SlidingWindowStrategy()
        counter = ApproximateTokenCounter()
        history = [
            {"role": "user", "content": "a" * 36},  # 14
            {"role": "assistant", "content": "b" * 36},  # 14
        ]
        # Budget = 15 -> one message (14) fits; adding second would exceed
        result = strategy.select(history, 15, counter)
        assert len(result) == 1
        assert result[0]["content"][0] == "b"


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
    }
    defaults.update(overrides)
    return ContextBudgetConfig(**defaults)


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
            live_advisory_text=None,
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
            live_advisory_text=None,
            history=[],
            tools=None,
            max_output_tokens=100,
        )
        assert plan.pruned_retrieval_text is None
        assert plan.facts_kept == 0
        assert plan.facts_dropped == 0

    def test_fixed_over_budget_returns_not_fits(self):
        # Huge persona + tiny window forces overflow.
        planner = ContextBudgetPlanner(config=_default_config(context_window=200))
        plan = planner.plan(
            persona_text="p" * 5000,
            static_text="s",
            retrieval_result=_make_retrieval_result([_make_fact()]),
            live_advisory_text=None,
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
            live_advisory_text=None,
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

    def test_history_strategy_invoked_with_budget(self):
        planner = ContextBudgetPlanner(config=_default_config())
        many_messages = [{"role": "user", "content": "a" * 100} for _ in range(50)]

        plan = planner.plan(
            persona_text="p",
            static_text="s",
            retrieval_result=None,
            live_advisory_text=None,
            history=many_messages,
            tools=None,
            max_output_tokens=100,
        )

        # History must have been pruned (can't fit 50 * ~32 tokens = 1600).
        assert len(plan.pruned_history) < len(many_messages)
        # Chronological order preserved.
        assert plan.pruned_history == many_messages[-len(plan.pruned_history) :]

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
            live_advisory_text=None,
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
