"""The extraction retrieval scorer, on hand-built query sets and run files.

Each run here is written by the test, in the run-file schema the scorer's module docstring
states, so these tests also pin that schema for the runner that will write it. The one test
that opens a repository file loads the published query set through the scorer's own loader.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval_harness import extraction_retrieval_score as ers  # noqa: E402

_QUERIES = _REPO_ROOT / "data" / "gold" / "retrieval" / "queries-2026-11.jsonl"

A = {"source": "user", "predicate": "KNOWS", "target": "kotlin"}
B = {"source": "user", "predicate": "KNOWS", "target": "lua"}
C = {"source": "user", "predicate": "USES", "target": "rust"}
NOISE = [{"source": "noise", "predicate": "RELATED_TO", "target": f"n{i}"} for i in range(6)]


def _fact(fact: dict, grade: int) -> dict:
    return {**fact, "grade": grade}


def _query(qid: str, kind: str, relevant: list[dict], text: str = "q") -> dict:
    return {"id": qid, "query": text, "kind": kind, "relevant": relevant}


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _score(tmp_path: Path, queries: list[dict], run: list[dict]) -> ers.RetrievalSummary:
    qpath = _write_jsonl(tmp_path / "queries.jsonl", queries)
    rpath = _write_jsonl(tmp_path / "run.jsonl", run)
    return ers.score_run(ers.load_queries(qpath), ers.load_run(rpath))


def _graded() -> dict:
    # Two facts answer the query, one supports it.
    return _query("g", "type_scoped", [_fact(A, 2), _fact(B, 2), _fact(C, 1)])


def _ndcg(gains: list[int], ideal: list[int]) -> float:
    def dcg(gs: list[int]) -> float:
        return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gs))

    return dcg(gains) / dcg(ideal)


# -- whole runs -------------------------------------------------------------


def test_all_miss(tmp_path: Path) -> None:
    queries = [
        _query("d", "direct", [_fact(A, 2)]),
        _query("p", "paraphrase", [_fact(B, 2), _fact(C, 1)]),
    ]
    run = [{"id": "d", "retrieved": NOISE[:5]}, {"id": "p", "retrieved": NOISE[:5]}]
    summary = _score(tmp_path, queries, run)
    overall = summary.answerable
    assert (overall.hit_at_5, overall.recall_at_5, overall.mrr, overall.ndcg_at_5) == (
        0.0,
        0.0,
        0.0,
        0.0,
    )
    assert summary.by_kind["direct"].hit_at_5 == 0.0
    assert summary.missing == ()


def test_all_hit(tmp_path: Path) -> None:
    queries = [
        _query("d", "direct", [_fact(A, 2)]),
        _graded(),
        _query("n", "negative", []),
    ]
    run = [
        {"id": "d", "retrieved": [A]},
        {"id": "g", "retrieved": [A, B, C]},
        {"id": "n", "retrieved": []},
    ]
    summary = _score(tmp_path, queries, run)
    overall = summary.answerable
    assert overall.queries == 2
    assert (overall.hit_at_5, overall.recall_at_5, overall.mrr, overall.ndcg_at_5) == (
        1.0,
        1.0,
        1.0,
        1.0,
    )
    assert summary.negative_empty_share == 1.0
    assert summary.meets_target is True


def test_per_kind_means_and_overall(tmp_path: Path) -> None:
    queries = [
        _query("d1", "direct", [_fact(A, 2)]),
        _query("d2", "direct", [_fact(B, 2)]),
        _query("l1", "list_item", [_fact(C, 2)]),
    ]
    run = [
        {"id": "d1", "retrieved": [A]},
        {"id": "d2", "retrieved": [*NOISE[:1], B]},
        {"id": "l1", "retrieved": NOISE[:5]},
    ]
    summary = _score(tmp_path, queries, run)
    assert summary.by_kind["direct"].hit_at_5 == 1.0
    assert summary.by_kind["direct"].mrr == pytest.approx((1.0 + 0.5) / 2)
    assert summary.by_kind["list_item"].hit_at_5 == 0.0
    assert summary.answerable.hit_at_5 == pytest.approx(2 / 3)
    assert summary.meets_target is False
    # A kind with no query in the set reports no value, not a vacuous 1.0 or 0.0.
    assert summary.by_kind["paraphrase"].queries == 0
    assert summary.by_kind["paraphrase"].hit_at_5 is None


# -- graded relevance ---------------------------------------------------------


def test_graded_ties_in_either_order_score_the_same(tmp_path: Path) -> None:
    first = _score(tmp_path, [_graded()], [{"id": "g", "retrieved": [A, B, C]}])
    second = _score(tmp_path, [_graded()], [{"id": "g", "retrieved": [B, A, C]}])
    assert first.per_query[0].ndcg == pytest.approx(1.0)
    assert second.per_query[0].ndcg == pytest.approx(1.0)


def test_a_supporting_fact_ranked_first_lowers_ndcg_and_mrr(tmp_path: Path) -> None:
    summary = _score(tmp_path, [_graded()], [{"id": "g", "retrieved": [C, A, B]}])
    score = summary.per_query[0]
    assert score.hit is True
    assert score.recall == pytest.approx(1.0)
    assert score.reciprocal_rank == pytest.approx(0.5)
    assert score.ndcg == pytest.approx(_ndcg([1, 2, 2], [2, 2, 1]))
    assert score.ndcg < 1.0


def test_only_a_supporting_fact_is_not_a_hit(tmp_path: Path) -> None:
    summary = _score(tmp_path, [_graded()], [{"id": "g", "retrieved": [C]}])
    score = summary.per_query[0]
    assert score.hit is False
    assert score.reciprocal_rank == 0.0
    assert score.recall == pytest.approx(1 / 3)
    assert score.ndcg == pytest.approx(_ndcg([1], [2, 2, 1]))


def test_only_the_top_five_count(tmp_path: Path) -> None:
    queries = [_query("d", "direct", [_fact(A, 2)])]
    summary = _score(tmp_path, queries, [{"id": "d", "retrieved": [*NOISE[:5], A]}])
    score = summary.per_query[0]
    assert score.retrieved == 6
    assert (score.hit, score.recall, score.reciprocal_rank, score.ndcg) == (False, 0.0, 0.0, 0.0)


def test_a_repeated_fact_counts_once(tmp_path: Path) -> None:
    queries = [_query("g", "type_scoped", [_fact(A, 2), _fact(B, 2)])]
    summary = _score(tmp_path, queries, [{"id": "g", "retrieved": [A, A, B]}])
    score = summary.per_query[0]
    assert score.recall == pytest.approx(1.0)
    assert score.ndcg == pytest.approx(_ndcg([2, 0, 2], [2, 2]))


# -- negatives and missing queries -------------------------------------------


def test_an_empty_negative_counts_as_returning_nothing(tmp_path: Path) -> None:
    queries = [
        _query("n1", "negative", []),
        _query("n2", "negative", []),
        _query("n3", "negative", []),
        _query("d", "direct", [_fact(A, 2)]),
    ]
    run = [
        {"id": "n1", "retrieved": []},
        {"id": "n2", "retrieved": [NOISE[0]]},
        {"id": "d", "retrieved": [A]},
    ]
    summary = _score(tmp_path, queries, run)
    # n3 never ran: it counts against the share rather than as an empty answer.
    assert summary.negative_empty_share == pytest.approx(1 / 3)
    assert summary.by_kind["negative"].empty_share == pytest.approx(1 / 3)
    assert summary.by_kind["negative"].hit_at_5 is None
    # Negatives sit outside the answerable means.
    assert summary.answerable.queries == 1
    assert summary.answerable.hit_at_5 == 1.0
    assert summary.missing == ("n3",)


def test_a_query_missing_from_the_run_is_a_miss(tmp_path: Path) -> None:
    queries = [_query("d1", "direct", [_fact(A, 2)]), _query("d2", "direct", [_fact(B, 2)])]
    run = [{"id": "d1", "retrieved": [A]}, {"id": "stray", "retrieved": [A]}]
    summary = _score(tmp_path, queries, run)
    assert summary.answerable.hit_at_5 == pytest.approx(0.5)
    assert summary.ran == 1
    assert summary.missing == ("d2",)
    assert summary.unknown == ("stray",)
    assert [s.ran for s in summary.per_query] == [True, False]


# -- id normalisation ---------------------------------------------------------


def test_produced_ids_are_normalised_like_the_extraction_scorer(tmp_path: Path) -> None:
    queries = [
        _query(
            "d",
            "direct",
            [_fact({"source": "rust", "predicate": "USED_FOR", "target": "backend-work"}, 2)],
        ),
    ]
    run = [
        {
            "id": "d",
            "retrieved": [{"source": " Rust ", "predicate": "used_for", "target": "Backend_Work!"}],
        }
    ]
    assert _score(tmp_path, queries, run).per_query[0].hit is True


def test_a_metric_target_is_rebuilt_value_first(tmp_path: Path) -> None:
    gold = {"source": "api", "predicate": "HAS_METRIC", "target": "12000-requests-per-second"}
    queries = [_query("m", "direct", [_fact(gold, 2)])]
    # HAS_METRIC admits only a Metric target, so the type is implied when the run omits it.
    implied = {"source": "API", "predicate": "HAS_METRIC", "target": "requests-per-second-12000"}
    assert _score(tmp_path, queries, [{"id": "m", "retrieved": [implied]}]).per_query[0].hit
    stated = {
        "source": "api",
        "source_type": "Technology",
        "predicate": "HAS_METRIC",
        "target": "Requests per second 12000",
        "target_type": "Metric",
    }
    assert _score(tmp_path, queries, [{"id": "m", "retrieved": [stated]}]).per_query[0].hit


def test_a_non_metric_id_keeps_its_word_order() -> None:
    fact = ers.normalise_fact({"source": "user", "predicate": "USES", "target": "python-3"})
    assert fact == ("user", "USES", "python-3")


def test_extra_fields_on_a_retrieved_fact_are_ignored(tmp_path: Path) -> None:
    queries = [_query("d", "direct", [_fact(A, 2)])]
    run = [{"id": "d", "retrieved": [{**A, "score": 0.42, "text": "user knows kotlin"}]}]
    assert _score(tmp_path, queries, run).per_query[0].hit is True


# -- loaders ------------------------------------------------------------------


@pytest.mark.parametrize(
    "row",
    [
        _query("x", "sideways", [_fact(A, 2)]),
        _query("x", "direct", [_fact(A, 3)]),
        _query("x", "direct", [_fact(A, 1)]),
        _query("x", "direct", []),
        _query("x", "negative", [_fact(A, 2)]),
        {"id": "x", "kind": "direct", "relevant": [_fact(A, 2)]},
        _query("x", "direct", [{"source": "user", "predicate": "KNOWS", "grade": 2}]),
    ],
    ids=[
        "unknown-kind",
        "grade-out-of-range",
        "no-answering-fact",
        "answerable-without-facts",
        "negative-with-a-fact",
        "no-query-text",
        "fact-without-target",
    ],
)
def test_load_queries_refuses_a_malformed_row(tmp_path: Path, row: dict) -> None:
    with pytest.raises(ValueError):
        ers.load_queries(_write_jsonl(tmp_path / "q.jsonl", [row]))


def test_load_queries_refuses_a_duplicate_id(tmp_path: Path) -> None:
    rows = [_query("x", "direct", [_fact(A, 2)]), _query("x", "direct", [_fact(B, 2)])]
    with pytest.raises(ValueError, match="x"):
        ers.load_queries(_write_jsonl(tmp_path / "q.jsonl", rows))


@pytest.mark.parametrize(
    "rows",
    [
        [{"id": "x", "retrieved": []}, {"id": "x", "retrieved": []}],
        [{"id": "x"}],
        [{"id": "x", "retrieved": [{"source": "user", "predicate": "KNOWS"}]}],
        [{"retrieved": []}],
    ],
    ids=["duplicate-id", "no-retrieved", "fact-without-target", "no-id"],
)
def test_load_run_refuses_a_malformed_row(tmp_path: Path, rows: list[dict]) -> None:
    with pytest.raises(ValueError):
        ers.load_run(_write_jsonl(tmp_path / "run.jsonl", rows))


def test_the_published_query_set_loads() -> None:
    queries = ers.load_queries(_QUERIES)
    assert len(queries) == 50
    assert {q.kind for q in queries} == set(ers.KINDS)


# -- output and command line --------------------------------------------------


def test_summary_dict_carries_counts_and_the_target(tmp_path: Path) -> None:
    summary = _score(
        tmp_path,
        [_query("d", "direct", [_fact(A, 2)]), _query("n", "negative", [])],
        [{"id": "d", "retrieved": [A]}, {"id": "n", "retrieved": []}],
    )
    data = ers.summary_to_dict(summary)
    assert data["k"] == 5
    assert data["target_hit_at_5"] == 0.90
    assert data["queries"] == 2 and data["ran"] == 2
    assert data["answerable"]["queries"] == 1
    assert data["by_kind"]["negative"]["empty_share"] == 1.0
    assert data["per_query"][0]["id"] == "d"
    json.dumps(data)


def test_markdown_names_the_target_and_every_kind(tmp_path: Path) -> None:
    summary = _score(
        tmp_path, [_query("d", "direct", [_fact(A, 2)])], [{"id": "d", "retrieved": [A]}]
    )
    text = ers.render_markdown(summary)
    assert "0.90" in text
    for kind in ers.KINDS:
        assert kind in text


def test_main_writes_both_outputs(tmp_path: Path) -> None:
    qpath = _write_jsonl(tmp_path / "q.jsonl", [_query("d", "direct", [_fact(A, 2)])])
    rpath = _write_jsonl(tmp_path / "run.jsonl", [{"id": "d", "retrieved": [A]}])
    out_json = tmp_path / "out" / "retrieval_summary.json"
    out_md = tmp_path / "out" / "retrieval.md"
    code = ers.main(
        [
            "--queries",
            str(qpath),
            "--run",
            str(rpath),
            "--json-out",
            str(out_json),
            "--out",
            str(out_md),
        ]
    )
    assert code == 0
    assert json.loads(out_json.read_text(encoding="utf-8"))["answerable"]["hit_at_5"] == 1.0
    assert "hit at 5" in out_md.read_text(encoding="utf-8").lower()


def test_main_exits_2_on_an_unreadable_input(tmp_path: Path) -> None:
    qpath = _write_jsonl(tmp_path / "q.jsonl", [_query("d", "direct", [_fact(A, 2)])])
    bad = tmp_path / "run.jsonl"
    bad.write_text("{not json\n", encoding="utf-8")
    assert ers.main(["--queries", str(qpath), "--run", str(tmp_path / "absent.jsonl")]) == 2
    assert ers.main(["--queries", str(qpath), "--run", str(bad)]) == 2
