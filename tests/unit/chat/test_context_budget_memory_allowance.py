"""Tests for `ContextBudgetPlanner.retrieval_allowance` (T32).

The memory block is sized before the fetch from the retrieval share the turn's plan would
give, so the allowance must equal `plan(...).retrieval_budget` for the same inputs. The floor
tests run the real static template and tool schemas through the planner, because the 8K E4B
envelope is only meaningful against the prompt the handler sends; a fixture-sized template
would let the template or the tool schemas grow until a 500-token block no longer fits, with
no test noticing.
"""

from __future__ import annotations

import pytest

from backend.chat.context_budget import ApproximateTokenCounter, ContextBudgetPlanner
from backend.chat.conversation_handler import (
    _STATIC_SYSTEM_TEMPLATE_BODY,
    KNOWLEDGE_TOOL_SCHEMAS,
)
from backend.knowledge.config import ContextBudgetConfig

# `conversation_max_tokens`, the reply limit the handler plans every turn with.
_REPLY_LIMIT_TOKENS = 1024

# The memory block's default budget (`MIST_MEMORY_BLOCK_TOKENS`, Design point 7).
_DEFAULT_BLOCK_TOKENS = 500

_SYNTHETIC_SENTENCE = "The quick synthetic sentence fills a fixed block for budget tests. "


def _synthetic_text(tokens: int) -> str:
    """Return synthetic prose that the approximate counter counts as exactly `tokens`.

    The counter divides by 3.6 in floating point, so the length is searched rather than
    computed, to keep each floor's input at its stated size.
    """
    counter = ApproximateTokenCounter()
    source = _SYNTHETIC_SENTENCE * (tokens // 10 + 2)
    length = int(tokens * ApproximateTokenCounter.CHARS_PER_TOKEN)
    while counter.count(source[:length]) > tokens:
        length -= 1
    while counter.count(source[:length]) < tokens:
        length += 1
    text = source[:length]
    assert counter.count(text) == tokens
    return text


def _variable_blocks(total_tokens: int) -> tuple[str | None, list[str]]:
    """Split `total_tokens` across persona, MIST.md conventions and the user profile.

    Returns the persona text and the two extra fixed texts, in the shape the handler passes
    them to the planner.
    """
    if total_tokens == 0:
        return None, []
    persona = total_tokens * 2 // 5
    conventions = total_tokens * 7 // 20
    profile = total_tokens - persona - conventions
    return _synthetic_text(persona), [_synthetic_text(conventions), _synthetic_text(profile)]


def _real_planner(window: int) -> ContextBudgetPlanner:
    """Planner at `window` with the shipped reserves and retrieval ratio.

    The reserves and ratio are left at the config's defaults on purpose: the floors must
    move if those defaults move.
    """
    return ContextBudgetPlanner(config=ContextBudgetConfig(context_window=window))


def _real_allowance(window: int, variable_tokens: int) -> int:
    persona_text, extra_fixed_texts = _variable_blocks(variable_tokens)
    return _real_planner(window).retrieval_allowance(
        persona_text=persona_text,
        static_text=_STATIC_SYSTEM_TEMPLATE_BODY,
        live_advisory_text=None,
        tools=KNOWLEDGE_TOOL_SCHEMAS,
        max_output_tokens=_REPLY_LIMIT_TOKENS,
        extra_fixed_texts=extra_fixed_texts,
    )


class TestRetrievalAllowanceMatchesPlan:
    @pytest.mark.parametrize(
        ("window", "variable_tokens"),
        [
            pytest.param(4096, 0, id="4096-bare"),
            pytest.param(8192, 0, id="8192-bare"),
            pytest.param(8192, 2000, id="8192-2000"),
            pytest.param(8192, 3000, id="8192-3000"),
            pytest.param(16384, 1000, id="16384-1000"),
            pytest.param(32768, 0, id="32768-bare"),
            pytest.param(2048, 0, id="2048-fixed-cost-overflows"),
        ],
    )
    def test_allowance_equals_plan_retrieval_budget(self, window, variable_tokens):
        planner = _real_planner(window)
        persona_text, extra_fixed_texts = _variable_blocks(variable_tokens)
        inputs = {
            "persona_text": persona_text,
            "static_text": _STATIC_SYSTEM_TEMPLATE_BODY,
            "live_advisory_text": None,
            "tools": KNOWLEDGE_TOOL_SCHEMAS,
            "max_output_tokens": _REPLY_LIMIT_TOKENS,
            "extra_fixed_texts": extra_fixed_texts,
        }

        allowance = planner.retrieval_allowance(**inputs)
        plan = planner.plan(**inputs, retrieval_result=None, history=[])

        assert allowance == plan.retrieval_budget


class TestRetrievalAllowanceArithmetic:
    def test_allowance_is_the_ratio_share_of_the_flex_budget(self):
        config = ContextBudgetConfig(
            context_window=1000,
            output_reserve_tokens=100,
            safety_margin_tokens=50,
            retrieval_budget_ratio=0.3,
        )
        planner = ContextBudgetPlanner(config=config)

        allowance = planner.retrieval_allowance(
            persona_text="p" * 36,
            static_text="s" * 36,
            live_advisory_text="a" * 36,
            tools=None,
            max_output_tokens=100,
            extra_fixed_texts=["c" * 36],
        )

        # total = 1000 - 100 - 100 - 50 = 750; fixed = 4 x (10 + 4) = 56; flex = 694.
        assert allowance == int(694 * 0.3)

    def test_allowance_is_zero_when_the_fixed_cost_does_not_fit(self):
        config = ContextBudgetConfig(
            context_window=200, output_reserve_tokens=100, safety_margin_tokens=50
        )
        planner = ContextBudgetPlanner(config=config)

        allowance = planner.retrieval_allowance(
            persona_text="p" * 5000,
            static_text="s",
            live_advisory_text=None,
            tools=None,
            max_output_tokens=50,
        )

        assert allowance == 0


class TestRetrievalAllowanceFloors:
    """G10: the 8K E4B envelope leaves room for the default block, and shrinks it rather
    than dropping it when persona, conventions and profile grow.
    """

    @pytest.mark.parametrize(
        "variable_tokens",
        [
            pytest.param(0, id="no-persona-conventions-profile"),
            pytest.param(2000, id="2000-tokens-of-them"),
        ],
    )
    def test_8192_window_fits_the_default_block(self, variable_tokens):
        allowance = _real_allowance(8192, variable_tokens)

        assert allowance >= _DEFAULT_BLOCK_TOKENS

    def test_8192_window_shrinks_the_block_at_3000_tokens_of_them(self):
        allowance = _real_allowance(8192, 3000)

        assert 0 < allowance < _DEFAULT_BLOCK_TOKENS

    def test_32768_window_fits_the_default_block(self):
        allowance = _real_allowance(32768, 0)

        assert allowance > _DEFAULT_BLOCK_TOKENS
