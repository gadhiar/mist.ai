"""Cluster 6 — budget-aware context assembly for ConversationHandler.

The conversation prompt is built from five ordered segments (persona block,
static system template, retrieval context, optional live-data advisory,
conversation history). Before Cluster 6 every segment was dumped in full and
trusted to fit the model's context window. Bug E (43% empty V6 responses)
traced back to this — persona + retrieval + 30 turns of history consumed
the model's usable attention past Gemma 4 E4B's effective 8K context, and
responses went empty.

Cluster 6 enforces a hard prompt-token budget. Fixed segments (persona,
tools, static system) consume from the top. The remainder is split between
retrieval and history using a configurable ratio. Retrieval is pruned by
priority score; history is pruned by a pluggable strategy (sliding-window
default). All arithmetic happens BEFORE LLMRequest construction so the
model never sees oversized prompts.

Components:
- `TokenCounter`: protocol for token accounting. `ApproximateTokenCounter`
  (char/3.6 heuristic, fast, zero-IO) is the default. `LlamaServerTokenCounter`
  calls llama-server's /tokenize endpoint for exact counts (slow — reserved
  for validation, not inner loop).
- `HistoryStrategy`: protocol for history pruning. `SlidingWindowStrategy`
  keeps the most recent turns that fit; future strategies (summarization)
  plug into the same interface.
- `ContextBudgetPlanner`: owns the arithmetic. Computes fixed cost, splits
  remainder, invokes strategies, returns the pruned retrieval context string
  and pruned history list.
- `score_fact_for_retrieval_pruning`: priority = similarity x confidence x
  recency-decay.

Spec: ~/.claude/plans/cluster-execution-roadmap.md Cluster 6.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from backend.knowledge.config import ContextBudgetConfig
from backend.knowledge.models import RetrievalResult, RetrievedFact

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TokenCounter
# ---------------------------------------------------------------------------


class TokenCounter(Protocol):
    """Counts tokens in text and message lists.

    Implementations may be approximate (char-based heuristic) or exact
    (tokenizer API). Budget planning uses this to decide what fits.
    """

    def count(self, text: str) -> int:
        """Return the estimated token count of `text`."""
        ...

    def count_message(self, message: dict[str, Any]) -> int:
        """Return the estimated token count of an OpenAI-format message dict.

        Includes fixed overhead for message boundary tokens (role, delimiters).
        """
        ...

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        """Return total token count for a list of messages."""
        ...


class ApproximateTokenCounter:
    """Char/3.6 heuristic. Overestimates by ~5-10% for English text.

    Default for budget planning because it's zero-IO and fast. The safety
    margin in `ContextBudgetConfig` absorbs the ~10% estimation error.
    """

    # 3.6 chars/token is conservative (typical English is 3.8-4.2).
    # Overestimating is safe — budget planner prunes earlier than strictly
    # necessary rather than overflowing the real context window.
    CHARS_PER_TOKEN = 3.6

    # Each message has fixed overhead for role + delimiters (OpenAI format).
    PER_MESSAGE_OVERHEAD_TOKENS = 4

    def count(self, text: str) -> int:
        """Approximate token count = ceil(len(text) / CHARS_PER_TOKEN)."""
        if not text:
            return 0
        return math.ceil(len(text) / self.CHARS_PER_TOKEN)

    def count_message(self, message: dict[str, Any]) -> int:
        """Content tokens + PER_MESSAGE_OVERHEAD + 20 per tool_call if present."""
        content = str(message.get("content") or "")
        overhead = self.PER_MESSAGE_OVERHEAD_TOKENS
        # Tool-calls bundled into assistant messages add extra tokens.
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            # Each tool call roughly 15-30 tokens for name + id + args wrapper.
            overhead += 20 * len(tool_calls)
        return self.count(content) + overhead

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        """Sum of count_message across the list."""
        return sum(self.count_message(m) for m in messages)


# ---------------------------------------------------------------------------
# HistoryStrategy
# ---------------------------------------------------------------------------


class HistoryStrategy(Protocol):
    """Chooses which history messages fit the budget."""

    def select(
        self,
        history: list[dict[str, Any]],
        budget_tokens: int,
        counter: TokenCounter,
    ) -> list[dict[str, Any]]:
        """Return a subset of `history` whose total token count ≤ `budget_tokens`.

        Must return messages in original chronological order. May return an
        empty list when the budget is too small for any message.
        """
        ...


class SlidingWindowStrategy:
    """Keeps the most recent N turns that fit the budget.

    Walks history in reverse (newest first) and accumulates until adding the
    next message would exceed the budget. Preserves chronological order in
    the returned list.

    This is the simplest viable strategy for MVP. Summarization and
    hierarchical strategies plug into the same interface post-Cluster-6.
    """

    def select(
        self,
        history: list[dict[str, Any]],
        budget_tokens: int,
        counter: TokenCounter,
    ) -> list[dict[str, Any]]:
        """Keep the most recent history messages that fit the token budget."""
        if budget_tokens <= 0 or not history:
            return []

        selected_reverse: list[dict[str, Any]] = []
        used = 0
        for msg in reversed(history):
            cost = counter.count_message(msg)
            if used + cost > budget_tokens:
                break
            selected_reverse.append(msg)
            used += cost

        selected_reverse.reverse()
        return selected_reverse


_HISTORY_STRATEGIES: dict[str, HistoryStrategy] = {
    "sliding_window": SlidingWindowStrategy(),
}


def get_history_strategy(name: str) -> HistoryStrategy:
    """Look up a registered history strategy by name. Falls back to sliding_window."""
    strategy = _HISTORY_STRATEGIES.get(name)
    if strategy is None:
        logger.warning("Unknown history_strategy %r; falling back to sliding_window", name)
        return _HISTORY_STRATEGIES["sliding_window"]
    return strategy


# ---------------------------------------------------------------------------
# Retrieval priority scoring
# ---------------------------------------------------------------------------


def score_fact_for_retrieval_pruning(
    fact: RetrievedFact,
    *,
    now: datetime | None = None,
    recency_half_life_days: float = 30.0,
) -> float:
    """Priority score for retrieval pruning = similarity x confidence x recency.

    - similarity: `fact.similarity_score` (0-1, from vector search or RRF merge)
    - confidence: `fact.properties["confidence"]` if present, else 1.0
    - recency: exponential decay from `fact.created_at` with the given
      half-life; facts with no created_at get recency=1.0 (assume fresh)

    Higher score = keep. Pruning sorts descending and truncates.
    """
    similarity = float(fact.similarity_score or 0.0)

    confidence_raw = (fact.properties or {}).get("confidence")
    try:
        confidence = float(confidence_raw) if confidence_raw is not None else 1.0
    except (TypeError, ValueError):
        confidence = 1.0

    recency = 1.0
    created_at_raw = getattr(fact, "created_at", None)
    if created_at_raw:
        try:
            if isinstance(created_at_raw, str):
                created_at = datetime.fromisoformat(created_at_raw.replace("Z", "+00:00"))
            elif isinstance(created_at_raw, datetime):
                created_at = created_at_raw
            else:
                created_at = None
            if created_at is not None:
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=UTC)
                now_aware = now or datetime.now(UTC)
                age_days = max(0.0, (now_aware - created_at).total_seconds() / 86400.0)
                recency = 0.5 ** (age_days / recency_half_life_days)
        except (ValueError, TypeError) as exc:
            logger.debug("Recency parse failed for fact %s: %s", fact.object, exc)

    return similarity * confidence * recency


# ---------------------------------------------------------------------------
# Budget plan and planner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BudgetPlan:
    """Result of budget planning for a single turn.

    Returned by `ContextBudgetPlanner.plan()`. Callers use `pruned_retrieval_text`
    in place of the raw retrieval block and `pruned_history` in place of the
    full history list. `fits` is False when even the minimum (persona + static)
    exceeds context_window — a hard stop that should log + degrade gracefully.

    `retrieval_budget` and `history_budget` are the allocated token budgets;
    `retrieval_used` and `history_used` are what the pruned output actually
    consumed.
    """

    fits: bool
    total_budget: int
    fixed_cost: int
    retrieval_budget: int
    history_budget: int
    pruned_retrieval_text: str | None
    pruned_history: list[dict[str, Any]]
    retrieval_used: int
    history_used: int
    facts_kept: int
    facts_dropped: int


class ContextBudgetPlanner:
    """Computes per-turn budget allocation and prunes retrieval + history.

    Constructor takes the config and a TokenCounter. `plan()` is called once
    per turn BEFORE `LLMRequest` construction with the already-serialized
    persona block, static template, tools schema, retrieval result, and
    conversation history.

    Separation from `_build_messages`: the planner owns arithmetic, the
    handler owns message composition. This lets us unit-test the math with
    trivial fakes instead of wiring a full handler.
    """

    def __init__(
        self,
        config: ContextBudgetConfig,
        counter: TokenCounter | None = None,
        history_strategy: HistoryStrategy | None = None,
    ) -> None:
        """Initialize the planner with config and optional counter/strategy injections."""
        self._config = config
        self._counter = counter or ApproximateTokenCounter()
        self._history_strategy = history_strategy or get_history_strategy(config.history_strategy)

    @property
    def counter(self) -> TokenCounter:
        """Return the injected or default TokenCounter."""
        return self._counter

    @property
    def history_strategy(self) -> HistoryStrategy:
        """Return the injected or default HistoryStrategy."""
        return self._history_strategy

    def _fixed_segment_tokens(self, segments: list[str]) -> int:
        """Per-message overhead + content tokens for the fixed-cost segments."""
        total = 0
        for seg in segments:
            if seg:
                total += self._counter.count(seg)
                total += ApproximateTokenCounter.PER_MESSAGE_OVERHEAD_TOKENS
        return total

    def _tools_schema_tokens(self, tools: list[dict] | None) -> int:
        """Approximate tokens consumed by the OpenAI tool schema array.

        The serialized JSON lives outside the `messages` array in the wire
        format but still counts against the prompt window. Char/3.6 estimate
        over the JSON serialization is close enough for budget purposes.
        """
        if not tools:
            return 0
        import json

        serialized = json.dumps(tools)
        return self._counter.count(serialized)

    def plan(
        self,
        *,
        persona_text: str | None,
        static_text: str,
        retrieval_result: RetrievalResult | None,
        live_advisory_text: str | None,
        history: list[dict[str, Any]],
        tools: list[dict] | None,
        max_output_tokens: int,
    ) -> BudgetPlan:
        """Plan the budget allocation for a single turn.

        Args:
            persona_text: Serialized persona block (MistContext.as_system_prompt_block).
            static_text: Static system-template body.
            retrieval_result: Auto-RAG result (or None if retrieval was skipped).
            live_advisory_text: Optional live-data advisory body.
            history: Conversation history messages in chronological order.
            tools: Tool schema array passed alongside messages.
            max_output_tokens: `LLMRequest.max_tokens` for the coming call.

        Returns:
            BudgetPlan describing what survives the budget, what got pruned,
            and whether anything fits at all.
        """
        total_budget = max(
            0,
            self._config.context_window
            - max_output_tokens
            - self._config.output_reserve_tokens
            - self._config.safety_margin_tokens,
        )

        # Fixed segments that never get pruned.
        fixed_texts = [t for t in (persona_text, static_text, live_advisory_text) if t]
        fixed_cost = self._fixed_segment_tokens(fixed_texts)
        fixed_cost += self._tools_schema_tokens(tools)

        flex_budget = max(0, total_budget - fixed_cost)
        if flex_budget <= 0:
            logger.warning(
                "ContextBudgetPlanner: fixed_cost %d exceeds total_budget %d "
                "(ctx=%d, max_out=%d, reserve=%d, safety=%d). Emitting empty "
                "retrieval + history.",
                fixed_cost,
                total_budget,
                self._config.context_window,
                max_output_tokens,
                self._config.output_reserve_tokens,
                self._config.safety_margin_tokens,
            )
            return BudgetPlan(
                fits=False,
                total_budget=total_budget,
                fixed_cost=fixed_cost,
                retrieval_budget=0,
                history_budget=0,
                pruned_retrieval_text=None,
                pruned_history=[],
                retrieval_used=0,
                history_used=0,
                facts_kept=0,
                facts_dropped=(len(retrieval_result.facts) if retrieval_result else 0),
            )

        # Allocate flex budget by ratio.
        retrieval_budget = int(flex_budget * self._config.retrieval_budget_ratio)
        history_budget = flex_budget - retrieval_budget

        # Prune retrieval by priority score.
        pruned_retrieval_text: str | None = None
        retrieval_used = 0
        facts_kept = 0
        facts_dropped = 0
        if retrieval_result is not None and retrieval_result.total_facts > 0:
            pruned_retrieval_text, retrieval_used, facts_kept, facts_dropped = (
                self._prune_retrieval(retrieval_result, retrieval_budget)
            )

        # Prune history by strategy.
        pruned_history = self._history_strategy.select(history, history_budget, self._counter)
        history_used = self._counter.count_messages(pruned_history)

        return BudgetPlan(
            fits=True,
            total_budget=total_budget,
            fixed_cost=fixed_cost,
            retrieval_budget=retrieval_budget,
            history_budget=history_budget,
            pruned_retrieval_text=pruned_retrieval_text,
            pruned_history=pruned_history,
            retrieval_used=retrieval_used,
            history_used=history_used,
            facts_kept=facts_kept,
            facts_dropped=facts_dropped,
        )

    def _prune_retrieval(
        self,
        retrieval_result: RetrievalResult,
        budget_tokens: int,
    ) -> tuple[str, int, int, int]:
        """Score facts, sort descending, emit as formatted context until budget hits.

        Returns `(text, used_tokens, kept_count, dropped_count)`. When the
        full formatted context already fits, emits it verbatim (no scoring
        overhead).
        """
        full_text = retrieval_result.formatted_context or ""
        full_cost = (
            self._counter.count(full_text) + ApproximateTokenCounter.PER_MESSAGE_OVERHEAD_TOKENS
        )
        if full_cost <= budget_tokens:
            return full_text, full_cost, len(retrieval_result.facts), 0

        scored = sorted(
            retrieval_result.facts,
            key=score_fact_for_retrieval_pruning,
            reverse=True,
        )

        kept: list[RetrievedFact] = []
        used = 0
        header = f"Relevant knowledge from your graph (query: '{retrieval_result.query}'):\n"
        used += self._counter.count(header) + ApproximateTokenCounter.PER_MESSAGE_OVERHEAD_TOKENS

        for fact in scored:
            line = f"- {fact.subject} -[{fact.predicate}]-> {fact.object}\n"
            cost = self._counter.count(line)
            if used + cost > budget_tokens:
                break
            kept.append(fact)
            used += cost

        if not kept:
            return "", 0, 0, len(retrieval_result.facts)

        body = "".join(f"- {f.subject} -[{f.predicate}]-> {f.object}\n" for f in kept)
        text = header + body
        return text, used, len(kept), len(retrieval_result.facts) - len(kept)
