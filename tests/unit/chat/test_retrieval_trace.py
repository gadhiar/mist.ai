"""Unit tests for `backend.chat.retrieval_trace`, the per-turn retrieval trace builder.

The trace is the JSON of the turn-trace contract (`retrieval_context["trace"]`):
every retrieval candidate a turn saw, with its ref, rank, score and one
disposition, plus the budget planner's counts. It must hold no utterance,
chunk or fact text, so these tests build every fixture from synthetic facts
and assert that none of their strings reach the serialized trace.
"""

from __future__ import annotations

import json
import math

import pytest

from backend.chat.context_budget import BudgetPlan
from backend.chat.retrieval_trace import (
    MAX_CANDIDATES_PER_RETRIEVAL,
    TRACE_VERSION,
    TurnTraceBuilder,
    refs_removed,
    trace_failed,
)
from backend.knowledge.models import RetrievalCandidate, RetrievedFact, fact_ref

_TOP_KEYS = {"v", "auto_inject", "tools", "budget", "truncated"}
_AUTO_INJECT_KEYS = {"ran", "skip_reason", "limit", "candidates"}
_TOOL_KEYS = {"tool", "call", "limit", "candidates"}
_BUDGET_KEYS = {"facts_kept", "facts_dropped", "history_kept", "history_raw", "fits"}
_CANDIDATE_KEYS = {"source", "ref", "rank", "score", "disposition"}
_DISPOSITIONS = {
    "injected",
    "injected_label_only",
    "returned",
    "dropped_empty",
    "dropped_profile_dedup",
    "dropped_budget",
    "dropped_no_fit",
    "dropped_compact",
    "not_selected",
}

# Synthetic fixture strings. None of them may appear in a serialized trace.
_VAULT_PATH = "sessions/zebra-orchard-notes.md"
_VAULT_HEADING = "Quokka harbour planning"
_VAULT_TEXT = "The heliotrope lantern festival moved to the marmalade pier."
_VAULT_PATH_2 = "users/velvet-tangerine.md"
_VAULT_HEADING_2 = "Saffron kite ledger"
_VAULT_TEXT_2 = "Cobalt walrus enjoys the periwinkle observatory."
_GRAPH_SUBJECT = "Wombatina"
_GRAPH_PREDICATE = "PREFERS"
_GRAPH_OBJECT = "Juniperscript"
_VECTOR_TITLE = "Narwhal carousel handbook"
_VECTOR_TEXT = "Tamarind gears hum beneath the lavender viaduct."
_VECTOR_CHUNK_ID = "chunk-ochre-falcon-17"

_FIXTURE_STRINGS = (
    _VAULT_PATH,
    _VAULT_HEADING,
    _VAULT_TEXT,
    _VAULT_PATH_2,
    _VAULT_HEADING_2,
    _VAULT_TEXT_2,
    _GRAPH_SUBJECT,
    _GRAPH_OBJECT,
    _VECTOR_TITLE,
    _VECTOR_TEXT,
    _VECTOR_CHUNK_ID,
)


def _vault_fact(path: str = _VAULT_PATH, heading: str = _VAULT_HEADING, text: str = _VAULT_TEXT):
    return RetrievedFact(
        subject="VaultNote",
        subject_type="VaultSession",
        predicate="MENTIONS",
        object=heading,
        object_type="VaultChunk",
        properties={"path": path, "text": text, "content": text},
        similarity_score=0.71,
        graph_distance=999,
    )


def _graph_fact(obj: str = _GRAPH_OBJECT):
    return RetrievedFact(
        subject=_GRAPH_SUBJECT,
        subject_type="Person",
        predicate=_GRAPH_PREDICATE,
        object=obj,
        object_type="Technology",
        properties={},
        similarity_score=0.83,
        graph_distance=1,
    )


def _vector_fact():
    return RetrievedFact(
        subject="Document",
        subject_type="DocumentChunk",
        predicate="CONTAINS",
        object=_VECTOR_TITLE,
        object_type="markdown",
        properties={"chunk_id": _VECTOR_CHUNK_ID, "text": _VECTOR_TEXT, "source_id": "doc-1"},
        similarity_score=0.64,
        graph_distance=999,
    )


def _candidate(
    fact: RetrievedFact,
    *,
    source: str,
    rank: int = 1,
    score: float | None = 0.5,
    selected: bool = True,
    empty: bool = False,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        source=source,  # type: ignore[arg-type]
        ref=fact_ref(fact),
        rank=rank,
        score=score,
        selected=selected,
        empty=empty,
    )


def _plan(
    *,
    fits: bool = True,
    kept: tuple[str, ...] = (),
    dropped: tuple[str, ...] = (),
    rerendered: bool = False,
    history_kept: int = 1,
) -> BudgetPlan:
    return BudgetPlan(
        fits=fits,
        total_budget=1000,
        fixed_cost=400,
        retrieval_budget=240,
        history_budget=360,
        pruned_retrieval_text=None,
        pruned_history=[{"role": "user", "content": "x"}] * history_kept,
        retrieval_used=0,
        history_used=0,
        facts_kept=len(kept),
        facts_dropped=len(dropped),
        kept_fact_refs=kept,
        dropped_fact_refs=dropped,
        retrieval_rerendered=rerendered,
    )


def _dispositions(entry: dict) -> list[str]:
    return [c["disposition"] for c in entry["candidates"]]


class TestShape:
    def test_skipped_auto_inject_builds_the_contract_shape(self) -> None:
        builder = TurnTraceBuilder()
        builder.auto_inject_skipped("short_message", limit=None)

        trace = builder.build()

        assert trace == {
            "v": 1,
            "auto_inject": {
                "ran": False,
                "skip_reason": "short_message",
                "limit": None,
                "candidates": [],
            },
            "tools": [],
            "budget": None,
            "truncated": 0,
        }
        assert TRACE_VERSION == 1

    def test_every_key_set_matches_the_contract(self) -> None:
        vault = _vault_fact()
        graph = _graph_fact()
        builder = TurnTraceBuilder()
        builder.auto_inject_ran([_candidate(vault, source="vault", score=0.42)], limit=3)
        builder.budget_planned(_plan(kept=(fact_ref(vault),)), history_raw=4)
        builder.tool_retrieved(
            "query_knowledge_graph",
            call=1,
            limit=20,
            candidates=[_candidate(graph, source="graph")],
        )

        trace = builder.build()

        assert set(trace) == _TOP_KEYS
        assert set(trace["auto_inject"]) == _AUTO_INJECT_KEYS
        assert set(trace["budget"]) == _BUDGET_KEYS
        assert len(trace["tools"]) == 1
        assert set(trace["tools"][0]) == _TOOL_KEYS
        for candidate in trace["auto_inject"]["candidates"] + trace["tools"][0]["candidates"]:
            assert set(candidate) == _CANDIDATE_KEYS
            assert candidate["disposition"] in _DISPOSITIONS
        assert trace["auto_inject"] == {
            "ran": True,
            "skip_reason": None,
            "limit": 3,
            "candidates": [
                {
                    "source": "vault",
                    "ref": fact_ref(vault),
                    "rank": 1,
                    "score": 0.42,
                    "disposition": "injected",
                }
            ],
        }
        assert trace["budget"] == {
            "facts_kept": 1,
            "facts_dropped": 0,
            "history_kept": 1,
            "history_raw": 4,
            "fits": True,
        }
        assert trace["tools"][0]["tool"] == "query_knowledge_graph"
        assert trace["tools"][0]["call"] == 1
        assert trace["tools"][0]["limit"] == 20

    @pytest.mark.parametrize("reason", ["disabled", "short_message", "error"])
    def test_each_skip_reason_is_recorded(self, reason: str) -> None:
        builder = TurnTraceBuilder()
        builder.auto_inject_skipped(reason, limit=3 if reason == "error" else None)  # type: ignore[arg-type]
        auto = builder.build()["auto_inject"]
        assert auto["ran"] is False
        assert auto["skip_reason"] == reason

    def test_unknown_skip_reason_is_refused(self) -> None:
        with pytest.raises(ValueError):
            TurnTraceBuilder().auto_inject_skipped("bored", limit=None)  # type: ignore[arg-type]

    def test_unknown_tool_is_refused(self) -> None:
        with pytest.raises(ValueError):
            TurnTraceBuilder().tool_retrieved(
                "frontend.switch_form", call=1, limit=1, candidates=()  # type: ignore[arg-type]
            )

    def test_non_finite_score_is_written_as_null(self) -> None:
        builder = TurnTraceBuilder()
        builder.auto_inject_ran(
            [_candidate(_vault_fact(), source="vault", score=math.nan)], limit=3
        )
        trace = builder.build()
        assert trace["auto_inject"]["candidates"][0]["score"] is None
        json.dumps(trace, allow_nan=False)

    def test_trace_failed_is_the_contract_marker(self) -> None:
        assert trace_failed() == {"v": 1, "error": "trace_failed"}
        assert trace_failed() is not trace_failed()


class TestAutoInjectDispositions:
    def test_no_budget_plan_means_every_selected_fact_was_injected(self) -> None:
        vault = _vault_fact()
        builder = TurnTraceBuilder()
        builder.auto_inject_ran([_candidate(vault, source="vault")], limit=3)
        trace = builder.build()
        assert _dispositions(trace["auto_inject"]) == ["injected"]
        assert trace["budget"] is None

    def test_empty_chunk_is_dropped_empty(self) -> None:
        empty = _vault_fact(text="   ")
        builder = TurnTraceBuilder()
        builder.auto_inject_ran(
            [_candidate(empty, source="vault", selected=False, empty=True)], limit=3
        )
        builder.budget_planned(_plan(), history_raw=1)
        assert _dispositions(builder.build()["auto_inject"]) == ["dropped_empty"]

    def test_profile_chunk_is_dropped_profile_dedup(self) -> None:
        profile = _vault_fact(_VAULT_PATH_2, _VAULT_HEADING_2, _VAULT_TEXT_2)
        kept = _vault_fact()
        builder = TurnTraceBuilder()
        builder.auto_inject_ran(
            [_candidate(profile, source="vault"), _candidate(kept, source="vault", rank=2)],
            limit=3,
        )
        builder.auto_inject_profile_dedup([fact_ref(profile)])
        builder.budget_planned(_plan(kept=(fact_ref(kept),)), history_raw=1)
        assert _dispositions(builder.build()["auto_inject"]) == [
            "dropped_profile_dedup",
            "injected",
        ]

    def test_planner_drop_is_dropped_budget(self) -> None:
        first = _vault_fact()
        second = _vault_fact(_VAULT_PATH_2, _VAULT_HEADING_2, _VAULT_TEXT_2)
        builder = TurnTraceBuilder()
        builder.auto_inject_ran(
            [_candidate(first, source="vault"), _candidate(second, source="vault", rank=2)],
            limit=3,
        )
        builder.budget_planned(
            _plan(kept=(fact_ref(first),), dropped=(fact_ref(second),), rerendered=True),
            history_raw=1,
        )
        assert _dispositions(builder.build()["auto_inject"]) == [
            "injected_label_only",
            "dropped_budget",
        ]

    def test_rerender_keeps_vault_and_vector_as_label_only_and_graph_as_injected(self) -> None:
        vault, vector, graph = _vault_fact(), _vector_fact(), _graph_fact()
        refs = (fact_ref(vault), fact_ref(vector), fact_ref(graph))
        builder = TurnTraceBuilder()
        builder.auto_inject_ran(
            [
                _candidate(graph, source="graph"),
                _candidate(vector, source="vector"),
                _candidate(vault, source="vault"),
            ],
            limit=3,
        )
        builder.budget_planned(_plan(kept=refs, rerendered=True), history_raw=1)
        assert _dispositions(builder.build()["auto_inject"]) == [
            "injected",
            "injected_label_only",
            "injected_label_only",
        ]

    def test_pass_through_plan_keeps_every_fact_injected(self) -> None:
        vault, vector = _vault_fact(), _vector_fact()
        builder = TurnTraceBuilder()
        builder.auto_inject_ran(
            [_candidate(vault, source="vault"), _candidate(vector, source="vector", rank=2)],
            limit=3,
        )
        builder.budget_planned(
            _plan(kept=(fact_ref(vault), fact_ref(vector)), rerendered=False), history_raw=1
        )
        assert _dispositions(builder.build()["auto_inject"]) == ["injected", "injected"]

    def test_plan_that_does_not_fit_is_dropped_no_fit(self) -> None:
        vault = _vault_fact()
        builder = TurnTraceBuilder()
        builder.auto_inject_ran([_candidate(vault, source="vault")], limit=3)
        builder.budget_planned(
            _plan(fits=False, dropped=(fact_ref(vault),), history_kept=0), history_raw=3
        )
        trace = builder.build()
        assert _dispositions(trace["auto_inject"]) == ["dropped_no_fit"]
        assert trace["budget"]["fits"] is False
        assert trace["budget"]["history_kept"] == 0
        assert trace["budget"]["history_raw"] == 3

    def test_candidate_the_retriever_did_not_select_is_not_selected(self) -> None:
        vault = _vault_fact()
        builder = TurnTraceBuilder()
        builder.auto_inject_ran([_candidate(vault, source="vault", selected=False)], limit=3)
        builder.budget_planned(_plan(), history_raw=1)
        assert _dispositions(builder.build()["auto_inject"]) == ["not_selected"]

    @pytest.mark.parametrize(
        "plan_kwargs",
        [
            {"kept": "ref", "rerendered": True},
            {"dropped": "ref"},
            {"fits": False, "dropped": "ref"},
        ],
    )
    @pytest.mark.parametrize("deduped", [False, True])
    def test_not_selected_is_checked_before_every_selected_value(
        self, plan_kwargs: dict, deduped: bool
    ) -> None:
        # The ref appears in the dedup set and in the plan's refs; an
        # unselected row still reads `not_selected`, because the checks
        # after it apply only to selected candidates.
        vault = _vault_fact()
        ref = fact_ref(vault)
        kwargs = {k: ((ref,) if v == "ref" else v) for k, v in plan_kwargs.items()}
        builder = TurnTraceBuilder()
        builder.auto_inject_ran([_candidate(vault, source="vault", selected=False)], limit=3)
        if deduped:
            builder.auto_inject_profile_dedup([ref])
        builder.budget_planned(_plan(**kwargs), history_raw=1)
        assert _dispositions(builder.build()["auto_inject"]) == ["not_selected"]

    def test_empty_is_checked_before_not_selected(self) -> None:
        empty = _vault_fact(text="")
        builder = TurnTraceBuilder()
        builder.auto_inject_ran(
            [_candidate(empty, source="vault", selected=False, empty=True)], limit=3
        )
        assert _dispositions(builder.build()["auto_inject"]) == ["dropped_empty"]

    def test_profile_dedup_is_checked_before_the_plan(self) -> None:
        vault = _vault_fact()
        ref = fact_ref(vault)
        builder = TurnTraceBuilder()
        builder.auto_inject_ran([_candidate(vault, source="vault")], limit=3)
        builder.auto_inject_profile_dedup([ref])
        builder.budget_planned(_plan(fits=False, dropped=(ref,)), history_raw=1)
        assert _dispositions(builder.build()["auto_inject"]) == ["dropped_profile_dedup"]

    def test_no_fit_is_checked_before_dropped_budget(self) -> None:
        vault = _vault_fact()
        builder = TurnTraceBuilder()
        builder.auto_inject_ran([_candidate(vault, source="vault")], limit=3)
        builder.budget_planned(_plan(fits=False, dropped=(fact_ref(vault),)), history_raw=1)
        assert _dispositions(builder.build()["auto_inject"]) == ["dropped_no_fit"]

    def test_selected_fact_missing_from_the_plan_is_an_error(self) -> None:
        vault = _vault_fact()
        builder = TurnTraceBuilder()
        builder.auto_inject_ran([_candidate(vault, source="vault")], limit=3)
        builder.budget_planned(_plan(), history_raw=1)
        with pytest.raises(KeyError):
            builder.build()


class TestToolDispositions:
    def test_vault_tool_dispositions(self) -> None:
        returned = _vault_fact()
        profile = _vault_fact(_VAULT_PATH_2, _VAULT_HEADING_2, _VAULT_TEXT_2)
        empty = _vault_fact(text="")
        builder = TurnTraceBuilder()
        builder.auto_inject_skipped("disabled", limit=None)
        builder.tool_retrieved(
            "query_vault",
            call=2,
            limit=5,
            candidates=[
                _candidate(returned, source="vault", rank=1),
                _candidate(profile, source="vault", rank=2),
                _candidate(empty, source="vault", rank=3, selected=False, empty=True),
            ],
            deduped_refs=[fact_ref(profile)],
        )
        entry = builder.build()["tools"][0]
        assert entry["tool"] == "query_vault"
        assert entry["call"] == 2
        assert entry["limit"] == 5
        assert _dispositions(entry) == ["returned", "dropped_profile_dedup", "dropped_empty"]
        assert [c["rank"] for c in entry["candidates"]] == [1, 2, 3]

    def test_graph_tool_compact_and_unselected_dispositions(self) -> None:
        listed = _graph_fact("Alpha")
        unlisted = _graph_fact("Beta")
        unselected = _graph_fact("Gamma")
        builder = TurnTraceBuilder()
        builder.tool_retrieved(
            "query_knowledge_graph",
            call=1,
            limit=2,
            candidates=[
                _candidate(listed, source="graph", rank=1),
                _candidate(unlisted, source="graph", rank=2),
                _candidate(unselected, source="graph", rank=3, selected=False),
            ],
            unlisted_refs=[fact_ref(unlisted)],
        )
        assert _dispositions(builder.build()["tools"][0]) == [
            "returned",
            "dropped_compact",
            "not_selected",
        ]

    def test_tool_not_selected_is_checked_before_dedup_and_compact(self) -> None:
        folded = _vault_fact()
        cut = _graph_fact("Delta")
        builder = TurnTraceBuilder()
        builder.tool_retrieved(
            "query_knowledge_graph",
            call=1,
            limit=1,
            candidates=[
                _candidate(folded, source="vault", selected=False),
                _candidate(cut, source="graph", rank=2, selected=False),
            ],
            deduped_refs=[fact_ref(folded)],
            unlisted_refs=[fact_ref(cut)],
        )
        assert _dispositions(builder.build()["tools"][0]) == ["not_selected", "not_selected"]

    def test_tool_dedup_is_checked_before_compact(self) -> None:
        vault = _vault_fact()
        builder = TurnTraceBuilder()
        builder.tool_retrieved(
            "query_vault",
            call=1,
            limit=1,
            candidates=[_candidate(vault, source="vault")],
            deduped_refs=[fact_ref(vault)],
            unlisted_refs=[fact_ref(vault)],
        )
        assert _dispositions(builder.build()["tools"][0]) == ["dropped_profile_dedup"]

    def test_tools_keep_their_call_order(self) -> None:
        builder = TurnTraceBuilder()
        builder.tool_retrieved("query_vault", call=1, limit=5, candidates=())
        builder.tool_retrieved("query_knowledge_graph", call=3, limit=20, candidates=())
        tools = builder.build()["tools"]
        assert [(t["tool"], t["call"]) for t in tools] == [
            ("query_vault", 1),
            ("query_knowledge_graph", 3),
        ]


class TestCap:
    def _many(self, n: int) -> list[RetrievalCandidate]:
        return [
            _candidate(_graph_fact(f"Obj{i}"), source="graph", rank=i + 1, selected=False)
            for i in range(n)
        ]

    def test_at_most_fifty_candidates_per_retrieval_with_overflow_counted(self) -> None:
        assert MAX_CANDIDATES_PER_RETRIEVAL == 50
        builder = TurnTraceBuilder()
        builder.auto_inject_ran(self._many(53), limit=60)
        builder.tool_retrieved("query_knowledge_graph", call=1, limit=60, candidates=self._many(61))
        builder.tool_retrieved("query_vault", call=2, limit=5, candidates=self._many(50))
        trace = builder.build()
        assert len(trace["auto_inject"]["candidates"]) == 50
        assert [c["rank"] for c in trace["auto_inject"]["candidates"]] == list(range(1, 51))
        assert len(trace["tools"][0]["candidates"]) == 50
        assert len(trace["tools"][1]["candidates"]) == 50
        assert trace["truncated"] == 3 + 11

    def test_unselected_rows_are_dropped_before_any_selected_row(self) -> None:
        # 55 unselected graph rows lead the leg order; the selected vector
        # and vault rows come after them and must all survive the cap.
        unselected = self._many(55)
        selected = [
            _candidate(_vector_fact(), source="vector", rank=1),
            _candidate(_vault_fact(), source="vault", rank=1),
            _candidate(
                _vault_fact(_VAULT_PATH_2, _VAULT_HEADING_2, _VAULT_TEXT_2), source="vault", rank=2
            ),
        ]
        builder = TurnTraceBuilder()
        builder.tool_retrieved(
            "query_knowledge_graph", call=1, limit=60, candidates=unselected + selected
        )
        trace = builder.build()
        kept = trace["tools"][0]["candidates"]
        assert len(kept) == 50
        assert trace["truncated"] == 8
        # The last unselected rows go first; the kept rows keep leg-and-rank order.
        assert [(c["source"], c["rank"]) for c in kept] == (
            [("graph", r) for r in range(1, 48)] + [("vector", 1), ("vault", 1), ("vault", 2)]
        )
        assert [c["disposition"] for c in kept[-3:]] == ["returned"] * 3

    def test_selected_rows_are_dropped_last_first_only_when_no_unselected_row_remains(
        self,
    ) -> None:
        selected = [
            _candidate(_graph_fact(f"Sel{i}"), source="graph", rank=i + 1) for i in range(52)
        ]
        unselected = [
            _candidate(_vault_fact(heading=f"Unsel{i}"), source="vault", rank=i + 1, selected=False)
            for i in range(3)
        ]
        builder = TurnTraceBuilder()
        builder.tool_retrieved(
            "query_knowledge_graph", call=1, limit=60, candidates=selected + unselected
        )
        trace = builder.build()
        kept = trace["tools"][0]["candidates"]
        assert trace["truncated"] == 5
        assert [(c["source"], c["rank"]) for c in kept] == [("graph", r) for r in range(1, 51)]


class TestNoContent:
    def test_serialized_trace_holds_no_fixture_text(self) -> None:
        vault = _vault_fact()
        profile = _vault_fact(_VAULT_PATH_2, _VAULT_HEADING_2, _VAULT_TEXT_2)
        vector, graph = _vector_fact(), _graph_fact()
        builder = TurnTraceBuilder()
        builder.auto_inject_ran(
            [_candidate(profile, source="vault"), _candidate(vault, source="vault", rank=2)],
            limit=3,
        )
        builder.auto_inject_profile_dedup([fact_ref(profile)])
        builder.budget_planned(_plan(kept=(fact_ref(vault),), rerendered=True), history_raw=2)
        builder.tool_retrieved(
            "query_knowledge_graph",
            call=1,
            limit=20,
            candidates=[_candidate(graph, source="graph"), _candidate(vector, source="vector")],
        )
        serialized = json.dumps(builder.build())
        for text in _FIXTURE_STRINGS:
            assert text not in serialized, text
        for text in _FIXTURE_STRINGS:
            for i in range(len(text) - 5):
                assert text[i : i + 6] not in serialized, text[i : i + 6]


class TestRefsRemoved:
    def test_reports_the_refs_of_facts_no_longer_present(self) -> None:
        a, b, c = _vault_fact(), _graph_fact(), _vector_fact()
        assert refs_removed([a, b, c], [a, c]) == (fact_ref(b),)

    def test_compares_by_identity_not_equality(self) -> None:
        a = _vault_fact()
        twin = _vault_fact()
        assert refs_removed([a, twin], [a]) == (fact_ref(twin),)

    def test_nothing_removed(self) -> None:
        a = _vault_fact()
        assert refs_removed([a], [a]) == ()
