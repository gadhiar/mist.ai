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


# -- the build's merge map ------------------------------------------------------


def _relevant(source: str, predicate: str, target: str, grade: int) -> ers.RelevantFact:
    return ers.RelevantFact(source=source, predicate=predicate, target=target, grade=grade)


def _retrieval_query(*relevant: ers.RelevantFact, kind: str = "direct") -> ers.RetrievalQuery:
    return ers.RetrievalQuery(id="q", query="q", kind=kind, relevant=tuple(relevant))


def _merge(incoming: str, graph: str, graph_type: str = "Technology") -> dict:
    return {
        "tag": "t",
        "event_id": "golden-t",
        "incoming_id": incoming,
        "incoming_type": graph_type,
        "graph_id": graph,
        "graph_type": graph_type,
    }


PYTHON_MERGE = [_merge("python", "tech-python")]


def test_merge_aliases_map_a_gold_id_to_its_graph_ids() -> None:
    aliases = ers.merge_aliases([_merge("Python", "Tech_Python"), _merge("user", "user")])
    # Keys take the relevant facts' form, values the retrieved facts'; a self-mapping drops.
    assert aliases == {"python": frozenset({"tech-python"})}


def test_a_metric_merge_value_is_rebuilt_value_first() -> None:
    aliases = ers.merge_aliases(
        [_merge("api-throughput", "requests-per-second-12000", graph_type="Metric")]
    )
    assert aliases == {"api-throughput": frozenset({"12000-requests-per-second"})}


def test_accepted_keys_cross_both_endpoints_with_their_aliases() -> None:
    aliases = {"user": frozenset({"raj"}), "python": frozenset({"tech-python"})}
    keys = ers.accepted_keys(("user", "EXPERT_IN", "python"), aliases)
    assert keys == {
        ("user", "EXPERT_IN", "python"),
        ("user", "EXPERT_IN", "tech-python"),
        ("raj", "EXPERT_IN", "python"),
        ("raj", "EXPERT_IN", "tech-python"),
    }
    assert ers.accepted_keys(_relevant("lua", "USED_FOR", "modding", 2), aliases) == {
        ("lua", "USED_FOR", "modding")
    }


def test_a_merged_target_is_a_full_hit_through_the_merge_map() -> None:
    query = _retrieval_query(_relevant("user", "EXPERT_IN", "python", 2))
    retrieved = (("user", "EXPERT_IN", "tech-python"),)

    without = ers.score_query(query, retrieved)
    score = ers.score_query(query, retrieved, aliases=ers.merge_aliases(PYTHON_MERGE))

    assert without.hit is False
    assert (score.hit, score.recall, score.reciprocal_rank, score.ndcg) == (True, 1.0, 1.0, 1.0)


def test_a_fact_with_both_endpoints_merged_is_a_full_hit() -> None:
    aliases = ers.merge_aliases(
        [_merge("ingest-service", "svc-ingest"), _merge("clickhouse", "db-clickhouse")]
    )
    query = _retrieval_query(_relevant("ingest-service", "DEPENDS_ON", "clickhouse", 2))
    score = ers.score_query(query, (("svc-ingest", "DEPENDS_ON", "db-clickhouse"),), aliases)
    assert (score.hit, score.recall, score.reciprocal_rank, score.ndcg) == (True, 1.0, 1.0, 1.0)


def test_two_facts_resolving_to_one_key_collapse_into_one_item_at_the_higher_grade() -> None:
    query = _retrieval_query(
        _relevant("user", "USES", "python", 1),
        _relevant("user", "USES", "tech-python", 2),
        kind="type_scoped",
    )
    score = ers.score_query(
        query, (("user", "USES", "tech-python"),), aliases=ers.merge_aliases(PYTHON_MERGE)
    )
    # One item, credited once at grade 2: the ideal is that one item, so nDCG reaches 1.
    assert (score.hit, score.recall, score.reciprocal_rank) == (True, 1.0, 1.0)
    assert score.ndcg == pytest.approx(1.0)


def test_an_id_merged_into_two_graph_ids_is_credited_through_either() -> None:
    aliases = ers.merge_aliases(PYTHON_MERGE + [_merge("python", "python-lang")])
    query = _retrieval_query(_relevant("user", "EXPERT_IN", "python", 2))
    for graph_id in ("tech-python", "python-lang"):
        assert ers.score_query(query, (("user", "EXPERT_IN", graph_id),), aliases).hit is True


def test_a_key_listed_twice_in_one_query_takes_its_higher_grade() -> None:
    key = ("user", "KNOWS", "kotlin")
    query = _retrieval_query(_relevant(*key, 2), _relevant(*key, 1))
    score = ers.score_query(query, (key,))
    assert (score.hit, score.recall, score.reciprocal_rank) == (True, 1.0, 1.0)
    assert score.ndcg == pytest.approx(1.0)


# -- the built graph's edge directions -----------------------------------------

DEPENDS = ("ingest-service", "DEPENDS_ON", "clickhouse")
REVERSED = ("clickhouse", "DEPENDS_ON", "ingest-service")


def _record(merges: list[dict], edges: list[tuple[str, str, str]]) -> ers.BuildRecord:
    return ers.build_record({"merges": merges, "edges": [list(e) for e in edges]})


def _key(fact: dict) -> tuple[str, str, str]:
    return (fact["source"], fact["predicate"], fact["target"])


def test_orient_flips_a_key_whose_only_edge_is_its_reverse() -> None:
    oriented, flipped = ers.orient((REVERSED,), frozenset({DEPENDS}))
    assert (oriented, flipped) == ((DEPENDS,), 1)


@pytest.mark.parametrize(
    "edges",
    [
        pytest.param({DEPENDS}, id="an-edge-as-reported"),
        pytest.param({DEPENDS, REVERSED}, id="both-directions"),
        pytest.param(set(), id="neither-direction"),
    ],
)
def test_orient_keeps_a_key_unless_only_its_reverse_is_an_edge(edges: set) -> None:
    oriented, flipped = ers.orient((DEPENDS,), frozenset(edges))
    assert (oriented, flipped) == ((DEPENDS,), 0)


def test_a_record_with_no_merges_and_every_key_an_edge_scores_as_without_one(
    tmp_path: Path,
) -> None:
    queries = [_graded(), _query("d", "direct", [_fact(A, 2)]), _query("n", "negative", [])]
    run = [
        {"id": "g", "retrieved": [C, *NOISE[:2], A]},
        {"id": "d", "retrieved": NOISE[:3]},
        {"id": "n", "retrieved": [NOISE[4]]},
    ]
    qs = ers.load_queries(_write_jsonl(tmp_path / "q.jsonl", queries))
    loaded = ers.load_run(_write_jsonl(tmp_path / "r.jsonl", run))
    edges = {key for facts in loaded.values() for key in facts}

    plain = ers.score_run(qs, loaded)
    built = ers.score_run(qs, loaded, build=_record([], sorted(edges)))

    assert plain.build is None
    assert built.per_query == plain.per_query
    assert built.answerable == plain.answerable
    assert built.by_kind == plain.by_kind
    assert built.build is not None and built.build.reoriented == 0


def test_a_fact_the_retriever_reported_reversed_is_flipped_and_credited(tmp_path: Path) -> None:
    fact = dict(zip(("source", "predicate", "target"), DEPENDS, strict=True))
    reported = dict(zip(("source", "predicate", "target"), REVERSED, strict=True))
    qs = ers.load_queries(
        _write_jsonl(tmp_path / "q.jsonl", [_query("t", "type_scoped", [_fact(fact, 2)])])
    )
    loaded = ers.load_run(
        _write_jsonl(tmp_path / "r.jsonl", [{"id": "t", "retrieved": [NOISE[0], reported]}])
    )

    summary = ers.score_run(qs, loaded, build=_record([], [DEPENDS, _key(NOISE[0])]))

    assert summary.per_query[0].hit is True
    assert summary.per_query[0].reciprocal_rank == pytest.approx(0.5)
    assert (summary.build.retrieved, summary.build.reoriented) == (2, 1)


def test_an_extraction_direction_error_is_not_credited(tmp_path: Path) -> None:
    # Extraction stored the edge reversed: the graph holds REVERSED only.
    fact = dict(zip(("source", "predicate", "target"), DEPENDS, strict=True))
    qs = ers.load_queries(
        _write_jsonl(tmp_path / "q.jsonl", [_query("t", "type_scoped", [_fact(fact, 2)])])
    )
    record = _record([], [REVERSED])
    for reported in (REVERSED, DEPENDS):
        row = dict(zip(("source", "predicate", "target"), reported, strict=True))
        loaded = ers.load_run(_write_jsonl(tmp_path / "r.jsonl", [{"id": "t", "retrieved": [row]}]))
        assert ers.score_run(qs, loaded, build=record).per_query[0].hit is False


def test_the_build_block_counts_merges_translations_and_flips(tmp_path: Path) -> None:
    python = {"source": "user", "predicate": "EXPERT_IN", "target": "python"}
    merged = {"source": "user", "predicate": "EXPERT_IN", "target": "tech-python"}
    queries = [
        _query("d", "direct", [_fact(python, 2), _fact(merged, 1)]),
        _query("p", "paraphrase", [_fact(B, 2)]),
    ]
    run = [{"id": "d", "retrieved": [merged]}, {"id": "p", "retrieved": [A, B]}]
    qs = ers.load_queries(_write_jsonl(tmp_path / "q.jsonl", queries))
    loaded = ers.load_run(_write_jsonl(tmp_path / "r.jsonl", run))
    record = _record(
        PYTHON_MERGE + [_merge("python", "python-lang")],
        [_key(merged), ("kotlin", "KNOWS", "user"), _key(B)],
    )

    summary = ers.score_run(qs, loaded, build=record)

    block = summary.build
    assert block == ers.BuildSummary(
        merged_ids=1,
        several_targets=("python",),
        facts_translated=1,
        facts_collapsed=1,
        retrieved=3,
        reoriented=1,
    )
    assert summary.per_query[0].ndcg == pytest.approx(1.0)
    data = ers.summary_to_dict(summary)
    assert data["build"] == {
        "merged_ids": 1,
        "several_targets": ["python"],
        "facts_translated": 1,
        "facts_collapsed": 1,
        "retrieved": 3,
        "reoriented": 1,
    }
    text = ers.render_markdown(summary)
    assert "1 merged ids" in text and "1 of 3 retrieved facts reoriented" in text


def test_a_summary_without_a_record_says_so(tmp_path: Path) -> None:
    summary = _score(
        tmp_path, [_query("d", "direct", [_fact(A, 2)])], [{"id": "d", "retrieved": [A]}]
    )
    assert ers.summary_to_dict(summary)["build"] is None
    assert "scored without a build report" in ers.render_markdown(summary)


@pytest.mark.parametrize(
    "report",
    [
        pytest.param({"edges": []}, id="no-merges"),
        pytest.param({"merges": []}, id="no-edges"),
        pytest.param({"merges": {}, "edges": []}, id="merges-not-a-list"),
        pytest.param({"merges": [], "edges": [["a", "B"]]}, id="edge-not-a-triple"),
        pytest.param({"merges": [{"graph_id": "x"}], "edges": []}, id="merge-without-id"),
        pytest.param([], id="not-an-object"),
    ],
)
def test_load_build_record_refuses_a_report_it_cannot_score_with(
    tmp_path: Path, report: object
) -> None:
    path = tmp_path / "build_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        ers.load_build_record(path)


def test_load_build_record_reads_merges_and_edges(tmp_path: Path) -> None:
    path = tmp_path / "build_report.json"
    path.write_text(
        json.dumps({"merges": PYTHON_MERGE, "edges": [list(DEPENDS)], "other": 1}),
        encoding="utf-8",
    )
    assert ers.load_build_record(path) == ers.BuildRecord(
        aliases={"python": frozenset({"tech-python"})}, edges=frozenset({DEPENDS})
    )


def test_main_scores_through_a_build_report(tmp_path: Path) -> None:
    python = {"source": "user", "predicate": "EXPERT_IN", "target": "python"}
    merged = {"source": "user", "predicate": "EXPERT_IN", "target": "tech-python"}
    qpath = _write_jsonl(tmp_path / "q.jsonl", [_query("d", "direct", [_fact(python, 2)])])
    rpath = _write_jsonl(tmp_path / "run.jsonl", [{"id": "d", "retrieved": [merged]}])
    report = tmp_path / "build_report.json"
    report.write_text(
        json.dumps({"merges": PYTHON_MERGE, "edges": [list(_key(merged))]}), encoding="utf-8"
    )
    out_json = tmp_path / "summary.json"
    args = ["--queries", str(qpath), "--run", str(rpath), "--json-out", str(out_json)]

    assert ers.main([*args, "--out", str(tmp_path / "a.md")]) == 0
    assert json.loads(out_json.read_text(encoding="utf-8"))["answerable"]["hit_at_5"] == 0.0
    assert ers.main([*args, "--out", str(tmp_path / "b.md"), "--build-report", str(report)]) == 0
    written = json.loads(out_json.read_text(encoding="utf-8"))
    assert written["answerable"]["hit_at_5"] == 1.0
    assert written["build"]["merged_ids"] == 1


def test_main_exits_2_on_a_build_report_without_merges_or_edges(tmp_path: Path) -> None:
    qpath = _write_jsonl(tmp_path / "q.jsonl", [_query("d", "direct", [_fact(A, 2)])])
    rpath = _write_jsonl(tmp_path / "run.jsonl", [{"id": "d", "retrieved": [A]}])
    report = tmp_path / "build_report.json"
    report.write_text(json.dumps({"missing": []}), encoding="utf-8")
    args = ["--queries", str(qpath), "--run", str(rpath), "--build-report", str(report)]
    assert ers.main(args) == 2
    assert ers.main([*args[:4], "--build-report", str(tmp_path / "absent.json")]) == 2


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
