"""Per-turn retrieval trace: every candidate a turn saw and what happened to it.

`ConversationHandler` makes one `TurnTraceBuilder` per turn, feeds it at each
point where a candidate can be kept or lost (auto-inject, the profile dedups,
the budget plan, the two retrieval tools), and writes `build()` to the turn's
`retrieval_context["trace"]`. mist-read-path measures the read path later
from those rows: a candidate that was retrieved and not injected is a missed
memory.

The trace holds refs, ranks, scores and dispositions only. A ref is the
opaque hash `fact_ref` makes of a fact's natural key, so the event store gains
no utterance, chunk or fact text, no vault path and no heading from it.

The builder only reads what it is given: it never touches the retrieval
result, the plan or the prompt, so feeding it cannot change what the model
is sent.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from backend.knowledge.models import RetrievalCandidate, RetrievedFact, fact_ref

from .context_budget import BudgetPlan

TRACE_VERSION = 1

# Candidates written per retrieval; the rest are counted in `truncated`, which
# keeps a turn's row bounded however many rows the legs return.
MAX_CANDIDATES_PER_RETRIEVAL = 50

Disposition = Literal[
    "injected",
    "injected_label_only",
    "returned",
    "dropped_empty",
    "dropped_profile_dedup",
    "dropped_budget",
    "dropped_no_fit",
    "dropped_compact",
    "not_selected",
]
AutoInjectSkipReason = Literal["disabled", "short_message", "error"]
TraceTool = Literal["query_vault", "query_knowledge_graph"]

_SKIP_REASONS: frozenset[str] = frozenset({"disabled", "short_message", "error"})
_TRACE_TOOLS: frozenset[str] = frozenset({"query_vault", "query_knowledge_graph"})

# A planner re-render writes each kept fact as `subject -[predicate]-> object`.
# For these sources that line is a label (`VaultNote -[MENTIONS]-> <heading>`,
# `Document -[CONTAINS]-> <title>`) without the fact's text; a graph fact's
# triple is the fact itself.
_LABEL_ONLY_SOURCES: frozenset[str] = frozenset({"vault", "vector"})


def trace_failed() -> dict[str, Any]:
    """Return the marker recorded in place of a trace the builder could not make."""
    return {"v": TRACE_VERSION, "error": "trace_failed"}


def refs_removed(
    before: Sequence[RetrievedFact], after: Sequence[RetrievedFact]
) -> tuple[str, ...]:
    """Return the refs of the facts in `before` that are no longer in `after`.

    Compared by identity, because the profile dedups build `after` by
    filtering `before`, and two distinct facts can compare equal.
    """
    remaining = {id(fact) for fact in after}
    return tuple(fact_ref(fact) for fact in before if id(fact) not in remaining)


@dataclass(frozen=True, slots=True)
class TracedCandidate:
    """One candidate as written to the trace."""

    source: str
    ref: str
    rank: int
    score: float | None
    disposition: Disposition

    def to_json(self) -> dict[str, Any]:
        """Render as the contract's `CANDIDATE` object."""
        return {
            "source": self.source,
            "ref": self.ref,
            "rank": self.rank,
            "score": self.score,
            "disposition": self.disposition,
        }


@dataclass(frozen=True, slots=True)
class _AutoInject:
    ran: bool
    skip_reason: AutoInjectSkipReason | None
    limit: int | None
    candidates: tuple[RetrievalCandidate, ...]


@dataclass(frozen=True, slots=True)
class _Budget:
    facts_kept: int
    facts_dropped: int
    history_kept: int
    history_raw: int
    fits: bool
    kept_refs: frozenset[str]
    dropped_refs: frozenset[str]
    rerendered: bool


@dataclass(frozen=True, slots=True)
class _ToolRetrieval:
    tool: TraceTool
    call: int
    limit: int
    candidates: tuple[RetrievalCandidate, ...]
    deduped_refs: frozenset[str]
    unlisted_refs: frozenset[str]


_NOT_RUN = _AutoInject(ran=False, skip_reason=None, limit=None, candidates=())


def _json_score(score: float | None) -> float | None:
    # The event store serializes with `json.dumps`, which would write a NaN
    # as a bare `NaN` token that strict JSON readers refuse.
    if score is None:
        return None
    value = float(score)
    return value if math.isfinite(value) else None


class TurnTraceBuilder:
    """Collects one turn's retrieval decisions and renders the trace contract.

    Each feed method records what one stage decided; dispositions are worked
    out in `build()`, once every stage has reported, because a candidate's
    fate depends on stages after its retrieval (the dedup, then the plan).
    Feed methods raise `ValueError` on input outside the contract, and
    `build()` raises `KeyError` when a selected auto-inject fact is in
    neither the plan's kept nor its dropped refs; the handler records
    `trace_failed()` for either rather than a trace that would be wrong.
    """

    __slots__ = ("_auto_inject", "_deduped", "_budget", "_tools")

    def __init__(self) -> None:
        self._auto_inject: _AutoInject = _NOT_RUN
        self._deduped: frozenset[str] = frozenset()
        self._budget: _Budget | None = None
        self._tools: list[_ToolRetrieval] = []

    def auto_inject_skipped(self, reason: AutoInjectSkipReason, *, limit: int | None) -> None:
        """Record that auto-inject retrieved nothing this turn, and why."""
        if reason not in _SKIP_REASONS:
            raise ValueError(f"unknown auto-inject skip reason: {reason!r}")
        self._auto_inject = _AutoInject(ran=False, skip_reason=reason, limit=limit, candidates=())

    def auto_inject_ran(self, candidates: Iterable[RetrievalCandidate], *, limit: int) -> None:
        """Record the candidates of the auto-inject retrieval."""
        self._auto_inject = _AutoInject(
            ran=True, skip_reason=None, limit=limit, candidates=tuple(candidates)
        )

    def auto_inject_profile_dedup(self, refs: Iterable[str]) -> None:
        """Record the auto-inject facts removed because the profile is injected on its own."""
        self._deduped = self._deduped | frozenset(refs)

    def budget_planned(self, plan: BudgetPlan, *, history_raw: int) -> None:
        """Record the budget plan, with the history length it was given."""
        self._budget = _Budget(
            facts_kept=plan.facts_kept,
            facts_dropped=plan.facts_dropped,
            history_kept=len(plan.pruned_history),
            history_raw=history_raw,
            fits=plan.fits,
            kept_refs=frozenset(plan.kept_fact_refs),
            dropped_refs=frozenset(plan.dropped_fact_refs),
            rerendered=plan.retrieval_rerendered,
        )

    def tool_retrieved(
        self,
        tool: TraceTool,
        *,
        call: int,
        limit: int,
        candidates: Iterable[RetrievalCandidate],
        deduped_refs: Iterable[str] = (),
        unlisted_refs: Iterable[str] = (),
    ) -> None:
        """Record one retrieval made by a tool call.

        Args:
            tool: The tool that retrieved.
            call: The tool call's 1-based position among the turn's tool calls.
            limit: The `limit` the tool passed to the retriever.
            candidates: The retrieval's candidates.
            deduped_refs: Refs the profile dedup removed from the tool result.
            unlisted_refs: Refs of selected facts the compact graph output did
                not list.
        """
        if tool not in _TRACE_TOOLS:
            raise ValueError(f"tool is not a retrieval tool: {tool!r}")
        self._tools.append(
            _ToolRetrieval(
                tool=tool,
                call=call,
                limit=limit,
                candidates=tuple(candidates),
                deduped_refs=frozenset(deduped_refs),
                unlisted_refs=frozenset(unlisted_refs),
            )
        )

    def build(self) -> dict[str, Any]:
        """Render the trace as the turn-trace JSON contract."""
        truncated = 0

        auto_candidates, overflow = self._capped(self._auto_inject.candidates)
        truncated += overflow
        auto_json = [
            self._traced(c, self._auto_inject_disposition(c)).to_json() for c in auto_candidates
        ]

        tools_json: list[dict[str, Any]] = []
        for retrieval in self._tools:
            tool_candidates, overflow = self._capped(retrieval.candidates)
            truncated += overflow
            tools_json.append(
                {
                    "tool": retrieval.tool,
                    "call": retrieval.call,
                    "limit": retrieval.limit,
                    "candidates": [
                        self._traced(c, self._tool_disposition(c, retrieval)).to_json()
                        for c in tool_candidates
                    ],
                }
            )

        budget = self._budget
        budget_json = (
            None
            if budget is None
            else {
                "facts_kept": budget.facts_kept,
                "facts_dropped": budget.facts_dropped,
                "history_kept": budget.history_kept,
                "history_raw": budget.history_raw,
                "fits": budget.fits,
            }
        )

        return {
            "v": TRACE_VERSION,
            "auto_inject": {
                "ran": self._auto_inject.ran,
                "skip_reason": self._auto_inject.skip_reason,
                "limit": self._auto_inject.limit,
                "candidates": auto_json,
            },
            "tools": tools_json,
            "budget": budget_json,
            "truncated": truncated,
        }

    @staticmethod
    def _capped(
        candidates: tuple[RetrievalCandidate, ...],
    ) -> tuple[tuple[RetrievalCandidate, ...], int]:
        """Keep at most the cap, dropping what the turn's outcome least depends on.

        Unselected rows go first, then selected ones, each from the end of
        the leg-and-rank order, so a selected row is never lost while an
        unselected one remains. The kept rows stay in their original order.
        """
        overflow = max(0, len(candidates) - MAX_CANDIDATES_PER_RETRIEVAL)
        if overflow == 0:
            return candidates, 0
        unselected = [i for i, c in enumerate(candidates) if not c.selected]
        selected = [i for i, c in enumerate(candidates) if c.selected]
        drop_order = unselected[::-1] + selected[::-1]
        dropped = set(drop_order[:overflow])
        kept = tuple(c for i, c in enumerate(candidates) if i not in dropped)
        return kept, overflow

    @staticmethod
    def _traced(candidate: RetrievalCandidate, disposition: Disposition) -> TracedCandidate:
        return TracedCandidate(
            source=candidate.source,
            ref=candidate.ref,
            rank=candidate.rank,
            score=_json_score(candidate.score),
            disposition=disposition,
        )

    @staticmethod
    def _retrieval_disposition(candidate: RetrievalCandidate) -> Disposition | None:
        # The first two checks of the contract's order. Every later value
        # describes what happened to a fact the retriever returned, so it is
        # reached only by a selected candidate.
        if candidate.empty:
            return "dropped_empty"
        if not candidate.selected:
            return "not_selected"
        return None

    def _auto_inject_disposition(self, candidate: RetrievalCandidate) -> Disposition:
        retrieval = self._retrieval_disposition(candidate)
        if retrieval is not None:
            return retrieval
        if candidate.ref in self._deduped:
            return "dropped_profile_dedup"
        budget = self._budget
        if budget is None:
            # No planner: the handler injects the retriever's block whole.
            return "injected"
        if not budget.fits:
            return "dropped_no_fit"
        if candidate.ref in budget.dropped_refs:
            return "dropped_budget"
        if candidate.ref in budget.kept_refs:
            if budget.rerendered and candidate.source in _LABEL_ONLY_SOURCES:
                return "injected_label_only"
            return "injected"
        raise KeyError(candidate.ref)

    def _tool_disposition(
        self, candidate: RetrievalCandidate, retrieval: _ToolRetrieval
    ) -> Disposition:
        disposition = self._retrieval_disposition(candidate)
        if disposition is not None:
            return disposition
        if candidate.ref in retrieval.deduped_refs:
            return "dropped_profile_dedup"
        if candidate.ref in retrieval.unlisted_refs:
            return "dropped_compact"
        return "returned"
