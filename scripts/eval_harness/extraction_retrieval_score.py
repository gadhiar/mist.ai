"""Score retrieval of extracted facts: does what extraction stored come back in the top five?

The query set (`data/gold/retrieval/queries-2026-11.jsonl`) names, per query, the gold facts
that answer it (grade 2) or support it (grade 1). A run file records what a retriever returned
for each query over a graph built from extraction's output for the dev turns. This module
compares the two; it reads no graph and calls no retriever, so a run file written by a runner
that builds the graph and queries it and one built by hand are scored the same way.

Query file, one JSON object per line:

    {"id": str, "query": str, "kind": "direct" | "paraphrase" | "list_item" | "type_scoped"
     | "negative", "relevant": [{"source": str, "predicate": str, "target": str,
     "grade": 1 | 2}]}

A negative query has no relevant fact; every other query has at least one grade-2 fact.

Run file, one JSON object per line, one line per query that was run:

    {"id": <query id>, "retrieved": [{"source": str, "predicate": str, "target": str,
     "source_type"?: str, "target_type"?: str, ...}, ...]}

`retrieved` is in rank order, best first, and may be empty. Other keys on a row or a fact
(a score, the rendered text) are ignored. A query with no line was not run: it scores as a
miss, and a negative query that was not run does not count as returning nothing.

Produced ids are matched with the extraction scorer's own normalisation, imported from
`score_extraction_run` rather than copied: `canonical_id` for every endpoint, and the
value-first Metric id when the endpoint is a Metric. An endpoint is a Metric when the fact
says so in `source_type` or `target_type`, or when the predicate admits only a Metric on that
side (`HAS_METRIC`'s target), so a run that omits types still matches. Predicates are compared
upper-cased, the form Neo4j relationship types take.

Metrics at k = 5, per query, then averaged per kind and over every non-negative query:

- hit at 5: 1 when any grade-2 fact is in the top five;
- recall at 5: the share of the query's relevant facts, of either grade, in the top five;
- reciprocal rank: 1 / the rank of the first grade-2 fact in the top five, else 0 (the mean
  is MRR);
- nDCG at 5: gain 2^grade - 1 at a log2(rank + 1) discount, over the ideal ordering of the
  query's relevant facts.

A fact retrieved twice earns its gain once, at its first rank. For negative queries the one
number is the share that returned nothing. A kind with no query reports no value rather than
a vacuous 0 or 1.

    python -m scripts.eval_harness.extraction_retrieval_score
        --queries data/gold/retrieval/queries-2026-11.jsonl --run <run.jsonl>
        [--json-out retrieval_summary.json] [--out retrieval.md]

Exit 0, or 2 when an input is missing or malformed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.knowledge.extraction.validator import RELATIONSHIP_CONSTRAINTS  # noqa: E402
from scripts.eval_harness.score_extraction_run import (  # noqa: E402
    _produced_entity_match_id,
    canonical_id,
)

K = 5
HIT_AT_5_TARGET = 0.90
KINDS = ("direct", "paraphrase", "list_item", "type_scoped", "negative")
ANSWERABLE_KINDS = tuple(kind for kind in KINDS if kind != "negative")
GRADES = (1, 2)
ANSWER_GRADE = 2

FactKey = tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class RelevantFact:
    """A gold fact relevant to a query, with its grade (2 answers, 1 supports)."""

    source: str
    predicate: str
    target: str
    grade: int

    @property
    def key(self) -> FactKey:
        """The normalised `(source, predicate, target)` a retrieved fact is matched on."""
        return (self.source, self.predicate, self.target)


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    """One query of the set and the facts relevant to it."""

    id: str
    query: str
    kind: str
    relevant: tuple[RelevantFact, ...]


@dataclass(frozen=True, slots=True)
class QueryScore:
    """One query's scores; the metric fields are None for a negative query."""

    id: str
    kind: str
    ran: bool
    retrieved: int
    hit: bool | None
    recall: float | None
    reciprocal_rank: float | None
    ndcg: float | None
    empty: bool | None


@dataclass(frozen=True, slots=True)
class KindSummary:
    """Means over a group of queries; None where the group has no query to average."""

    queries: int
    ran: int
    hit_at_5: float | None
    recall_at_5: float | None
    mrr: float | None
    ndcg_at_5: float | None
    empty_share: float | None


@dataclass(frozen=True, slots=True)
class RetrievalSummary:
    """A scored run: every non-negative query together, each kind, and each query."""

    queries: int
    ran: int
    missing: tuple[str, ...]
    unknown: tuple[str, ...]
    answerable: KindSummary
    by_kind: dict[str, KindSummary]
    per_query: tuple[QueryScore, ...]

    @property
    def negative_empty_share(self) -> float | None:
        """The share of negative queries that returned nothing."""
        return self.by_kind["negative"].empty_share

    @property
    def meets_target(self) -> bool:
        """True when hit at 5 over the non-negative queries reaches the target."""
        hit = self.answerable.hit_at_5
        return hit is not None and hit >= HIT_AT_5_TARGET


def _implied_type(predicate: str, side: int) -> str:
    """The one type a predicate admits on a side (0 source, 1 target), else empty."""
    constraint = RELATIONSHIP_CONSTRAINTS.get(predicate)
    if constraint is None or constraint[side] is None or len(constraint[side]) != 1:
        return ""
    return next(iter(constraint[side]))


def normalise_fact(raw: dict[str, Any]) -> FactKey:
    """Normalise a retrieved fact's ids and predicate the way the extraction scorer does."""
    predicate = str(raw["predicate"]).strip().upper()
    source_type = str(raw.get("source_type") or _implied_type(predicate, 0))
    target_type = str(raw.get("target_type") or _implied_type(predicate, 1))
    return (
        _produced_entity_match_id(str(raw["source"]), source_type, {}),
        predicate,
        _produced_entity_match_id(str(raw["target"]), target_type, {}),
    )


def _iter_rows(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append((line_no, obj))
    return rows


def _require(obj: dict[str, Any], fields: tuple[str, ...], where: str) -> None:
    missing = [name for name in fields if name not in obj]
    if missing:
        raise ValueError(f"{where}: missing {', '.join(missing)}")


def load_queries(path: Path) -> list[RetrievalQuery]:
    """Load and validate the query set, so a malformed query fails before any scoring."""
    queries: list[RetrievalQuery] = []
    seen: set[str] = set()
    for line_no, obj in _iter_rows(path):
        where = f"{path}:{line_no}"
        _require(obj, ("id", "query", "kind", "relevant"), where)
        qid = str(obj["id"])
        if qid in seen:
            raise ValueError(f"{where}: duplicate query id {qid!r}")
        seen.add(qid)
        kind = obj["kind"]
        if kind not in KINDS:
            raise ValueError(f"{where}: unknown kind {kind!r} (expected one of {KINDS})")
        relevant = []
        for fact in obj["relevant"]:
            _require(fact, ("source", "predicate", "target", "grade"), where)
            if fact["grade"] not in GRADES:
                raise ValueError(f"{where}: grade {fact['grade']!r} not in {GRADES}")
            relevant.append(
                RelevantFact(
                    source=canonical_id(str(fact["source"])),
                    predicate=str(fact["predicate"]).strip().upper(),
                    target=canonical_id(str(fact["target"])),
                    grade=int(fact["grade"]),
                )
            )
        if kind == "negative" and relevant:
            raise ValueError(f"{where}: a negative query has no relevant fact")
        if kind != "negative" and not any(f.grade == ANSWER_GRADE for f in relevant):
            raise ValueError(f"{where}: query {qid!r} has no grade-2 fact")
        queries.append(
            RetrievalQuery(id=qid, query=str(obj["query"]), kind=kind, relevant=tuple(relevant))
        )
    return queries


def load_run(path: Path) -> dict[str, tuple[FactKey, ...]]:
    """Load a run file into normalised, ranked facts per query id."""
    run: dict[str, tuple[FactKey, ...]] = {}
    for line_no, obj in _iter_rows(path):
        where = f"{path}:{line_no}"
        _require(obj, ("id", "retrieved"), where)
        qid = str(obj["id"])
        if qid in run:
            raise ValueError(f"{where}: duplicate query id {qid!r}")
        facts = []
        for fact in obj["retrieved"]:
            _require(fact, ("source", "predicate", "target"), where)
            facts.append(normalise_fact(fact))
        run[qid] = tuple(facts)
    return run


def _gain(grade: int) -> float:
    return float(2**grade - 1)


def _dcg(grades: list[int]) -> float:
    return sum(_gain(g) / math.log2(rank + 1) for rank, g in enumerate(grades, start=1))


def score_query(query: RetrievalQuery, retrieved: tuple[FactKey, ...] | None) -> QueryScore:
    """Score one query against its ranked facts, or None when it was not run."""
    ran = retrieved is not None
    facts = retrieved or ()
    if query.kind == "negative":
        return QueryScore(
            id=query.id,
            kind=query.kind,
            ran=ran,
            retrieved=len(facts),
            hit=None,
            recall=None,
            reciprocal_rank=None,
            ndcg=None,
            empty=ran and not facts,
        )
    grade_by_key = {fact.key: fact.grade for fact in query.relevant}
    credited: set[FactKey] = set()
    ranked_grades: list[int] = []
    first_answer = 0
    for rank, key in enumerate(facts[:K], start=1):
        grade = grade_by_key.get(key, 0)
        if key in credited:
            grade = 0
        elif grade:
            credited.add(key)
        ranked_grades.append(grade)
        if grade == ANSWER_GRADE and not first_answer:
            first_answer = rank
    ideal = sorted(grade_by_key.values(), reverse=True)[:K]
    return QueryScore(
        id=query.id,
        kind=query.kind,
        ran=ran,
        retrieved=len(facts),
        hit=bool(first_answer),
        recall=len(credited) / len(grade_by_key),
        reciprocal_rank=1.0 / first_answer if first_answer else 0.0,
        ndcg=_dcg(ranked_grades) / _dcg(ideal),
        empty=None,
    )


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _summarise(scores: list[QueryScore]) -> KindSummary:
    answerable = [s for s in scores if s.kind != "negative"]
    negative = [s for s in scores if s.kind == "negative"]
    return KindSummary(
        queries=len(scores),
        ran=sum(s.ran for s in scores),
        hit_at_5=_mean([float(bool(s.hit)) for s in answerable]),
        recall_at_5=_mean([s.recall or 0.0 for s in answerable]),
        mrr=_mean([s.reciprocal_rank or 0.0 for s in answerable]),
        ndcg_at_5=_mean([s.ndcg or 0.0 for s in answerable]),
        empty_share=_mean([float(bool(s.empty)) for s in negative]),
    )


def score_run(
    queries: list[RetrievalQuery], run: dict[str, tuple[FactKey, ...]]
) -> RetrievalSummary:
    """Score every query of the set against a loaded run."""
    per_query = tuple(score_query(q, run.get(q.id)) for q in queries)
    known = {q.id for q in queries}
    return RetrievalSummary(
        queries=len(queries),
        ran=sum(s.ran for s in per_query),
        missing=tuple(s.id for s in per_query if not s.ran),
        unknown=tuple(sorted(qid for qid in run if qid not in known)),
        answerable=_summarise([s for s in per_query if s.kind in ANSWERABLE_KINDS]),
        by_kind={kind: _summarise([s for s in per_query if s.kind == kind]) for kind in KINDS},
        per_query=per_query,
    )


def _kind_dict(summary: KindSummary) -> dict[str, Any]:
    return {
        "queries": summary.queries,
        "ran": summary.ran,
        "hit_at_5": summary.hit_at_5,
        "recall_at_5": summary.recall_at_5,
        "mrr": summary.mrr,
        "ndcg_at_5": summary.ndcg_at_5,
        "empty_share": summary.empty_share,
    }


def summary_to_dict(summary: RetrievalSummary) -> dict[str, Any]:
    """The machine-readable summary, counts beside every mean."""
    return {
        "k": K,
        "target_hit_at_5": HIT_AT_5_TARGET,
        "meets_target": summary.meets_target,
        "queries": summary.queries,
        "ran": summary.ran,
        "missing": list(summary.missing),
        "unknown": list(summary.unknown),
        "answerable": _kind_dict(summary.answerable),
        "by_kind": {kind: _kind_dict(s) for kind, s in summary.by_kind.items()},
        "negative_empty_share": summary.negative_empty_share,
        "per_query": [
            {
                "id": s.id,
                "kind": s.kind,
                "ran": s.ran,
                "retrieved": s.retrieved,
                "hit": s.hit,
                "recall": s.recall,
                "reciprocal_rank": s.reciprocal_rank,
                "ndcg": s.ndcg,
                "empty": s.empty,
            }
            for s in summary.per_query
        ],
    }


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def render_markdown(summary: RetrievalSummary) -> str:
    """One table, a row per kind, with the hit-at-5 target stated beside the overall row."""
    lines = [
        "# Extraction retrieval",
        "",
        f"- Queries: {summary.queries} (ran: {summary.ran})",
        f"- Target: hit at 5 >= {HIT_AT_5_TARGET:.2f} over non-negative queries: "
        f"{'PASS' if summary.meets_target else 'FAIL'}",
        f"- Negative queries that returned nothing: {_fmt(summary.negative_empty_share)}",
        "",
        "| Kind | Queries | Ran | Hit at 5 | Recall at 5 | MRR | nDCG at 5 | Empty |",
        "|---|---|---|---|---|---|---|---|",
    ]
    rows = [("all non-negative", summary.answerable)] + list(summary.by_kind.items())
    for name, s in rows:
        lines.append(
            f"| {name} | {s.queries} | {s.ran} | {_fmt(s.hit_at_5)} | {_fmt(s.recall_at_5)} "
            f"| {_fmt(s.mrr)} | {_fmt(s.ndcg_at_5)} | {_fmt(s.empty_share)} |"
        )
    if summary.missing:
        lines += ["", f"Not run (scored as misses): {', '.join(summary.missing)}"]
    if summary.unknown:
        lines += ["", f"Run ids not in the query set (ignored): {', '.join(summary.unknown)}"]
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score retrieval of extracted facts at k = 5.")
    p.add_argument("--queries", required=True, type=Path, help="Retrieval query set JSONL.")
    p.add_argument("--run", required=True, type=Path, help="Run file JSONL.")
    p.add_argument("--json-out", default=None, type=Path, help="Write the JSON summary.")
    p.add_argument("--out", default=None, type=Path, help="Write markdown (default stdout).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        queries = load_queries(args.queries)
        run = load_run(args.run)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    summary = score_run(queries, run)
    text = render_markdown(summary)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary_to_dict(summary), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
