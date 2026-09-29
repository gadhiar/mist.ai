"""Aggregate scored pairs into per-slice results, a JSON document and a text table."""

from __future__ import annotations

from .dataset import LABEL_DISTINCT, LABEL_DUPLICATE, SLICES
from .metrics import (
    best_f1,
    cosine_from_neo4j_score,
    distribution,
    point_at,
    point_to_dict,
    zero_false_merge,
)
from .scoring import ScoredPair

OVERALL = "overall"
SCHEMA_VERSION = 1
_TOP_N = 5


def _pair_record(sp: ScoredPair) -> dict:
    return {
        "slice": sp.pair.slice,
        "entity_type": sp.pair.entity_type,
        "label": sp.pair.label,
        "hard": sp.pair.hard,
        "a": sp.pair.a,
        "b": sp.pair.b,
        "cosine": sp.cosine,
        "neo4j_score": sp.neo4j_score,
    }


def analyze_slice(scored: list[ScoredPair], current_threshold: float) -> dict:
    """Compute every reported statistic for one set of scored pairs.

    Args:
        scored: Scored pairs, all of one slice (or all slices for overall).
        current_threshold: The live threshold, on the Neo4j-score scale.
    """
    dup = [s for s in scored if s.pair.label == LABEL_DUPLICATE]
    distinct = [s for s in scored if s.pair.label == LABEL_DISTINCT]
    dup_scores = [s.neo4j_score for s in dup]
    distinct_scores = [s.neo4j_score for s in distinct]

    zfm = zero_false_merge(dup_scores, distinct_scores)
    hardest_distinct = sorted(distinct, key=lambda s: s.neo4j_score, reverse=True)[:_TOP_N]
    weakest_dup = sorted(dup, key=lambda s: s.neo4j_score)[:_TOP_N]

    return {
        "n_duplicate": len(dup),
        "n_distinct": len(distinct),
        "duplicate": {
            "cosine": distribution([s.cosine for s in dup]),
            "neo4j_score": distribution(dup_scores),
        },
        "distinct": {
            "cosine": distribution([s.cosine for s in distinct]),
            "neo4j_score": distribution(distinct_scores),
        },
        "zero_false_merge": {
            "max_distinct_score": zfm.max_distinct_score,
            "max_distinct_raw_cosine": (
                None
                if zfm.max_distinct_score is None
                else cosine_from_neo4j_score(zfm.max_distinct_score)
            ),
            "point": point_to_dict(zfm.point),
        },
        "best_f1": point_to_dict(best_f1(dup_scores, distinct_scores)),
        "at_current_threshold": point_to_dict(
            point_at(dup_scores, distinct_scores, current_threshold)
        ),
        "highest_scoring_distinct": [_pair_record(s) for s in hardest_distinct],
        "lowest_scoring_duplicates": [_pair_record(s) for s in weakest_dup],
    }


def build_report(
    scored: list[ScoredPair],
    *,
    current_threshold: float,
    model_name: str,
    dataset_path: str,
) -> dict:
    """Build the full JSON-serialisable report."""
    slices = {OVERALL: analyze_slice(scored, current_threshold)}
    for slice_name in SLICES:
        in_slice = [s for s in scored if s.pair.slice == slice_name]
        if in_slice:
            slices[slice_name] = analyze_slice(in_slice, current_threshold)
    return {
        "schema_version": SCHEMA_VERSION,
        "model": model_name,
        "dataset": dataset_path,
        "score_scales": {
            "cosine": "raw cosine similarity in [-1, 1]",
            "neo4j_score": "(1 + cosine) / 2, what vector.similarity.cosine returns; "
            "a pair merges when neo4j_score >= threshold",
        },
        "current_threshold": {
            "neo4j_score": current_threshold,
            "raw_cosine": cosine_from_neo4j_score(current_threshold),
        },
        "slices": slices,
        "pairs": [_pair_record(s) for s in scored],
    }


def _fmt(value: float | None, width: int = 7) -> str:
    return f"{'n/a':>{width}}" if value is None else f"{value:>{width}.4f}"


def _dist_row(label: str, dist: dict) -> str:
    return (
        f"  {label:<22}{dist['n']:>4} "
        f"{_fmt(dist['min'])}{_fmt(dist['p10'])}{_fmt(dist['p25'])}{_fmt(dist['median'])}"
        f"{_fmt(dist['p75'])}{_fmt(dist['p90'])}{_fmt(dist['max'])}{_fmt(dist['mean'])}"
    )


def _point_line(label: str, point: dict | None) -> str:
    if point is None:
        return f"  {label:<26} none (no duplicate scores above every distinct pair)"
    return (
        f"  {label:<26} threshold={point['threshold']:.4f} (cos {point['raw_cosine']:.4f})  "
        f"P={_fmt(point['precision'], 6).strip()} R={_fmt(point['recall'], 6).strip()} "
        f"F1={point['f1']:.4f}  TP={point['tp']} FP={point['fp']} FN={point['fn']} TN={point['tn']}"
    )


def format_table(report: dict) -> str:
    """Render the report as a human-readable text table."""
    cur = report["current_threshold"]
    lines = [
        "MIST dedup threshold calibration",
        f"model:   {report['model']}",
        f"dataset: {report['dataset']}",
        "scores:  neo4j_score = (1 + cosine) / 2; a pair merges when neo4j_score >= threshold",
        f"current threshold: {cur['neo4j_score']:.4f} neo4j score (raw cosine {cur['raw_cosine']:.4f})",
    ]
    header = f"  {'':<22}{'n':>4} " + "".join(
        f"{name:>7}" for name in ("min", "p10", "p25", "median", "p75", "p90", "max", "mean")
    )
    for name, block in report["slices"].items():
        lines.extend(
            [
                "",
                f"=== {name}: {block['n_duplicate']} duplicate, {block['n_distinct']} distinct ===",
                header,
                _dist_row("duplicate raw cosine", block["duplicate"]["cosine"]),
                _dist_row("distinct raw cosine", block["distinct"]["cosine"]),
                _dist_row("duplicate neo4j score", block["duplicate"]["neo4j_score"]),
                _dist_row("distinct neo4j score", block["distinct"]["neo4j_score"]),
            ]
        )
        zfm = block["zero_false_merge"]
        if zfm["max_distinct_score"] is not None:
            lines.append(
                f"  highest distinct score: {zfm['max_distinct_score']:.4f} "
                f"(cos {zfm['max_distinct_raw_cosine']:.4f}); zero false merges needs threshold above it"
            )
        lines.append(_point_line("zero-false-merge point", zfm["point"]))
        lines.append(_point_line("best-F1 point", block["best_f1"]))
        lines.append(
            _point_line(f"at current {cur['neo4j_score']:.2f}", block["at_current_threshold"])
        )
        if name == "overall":
            lines.append("  highest-scoring distinct pairs (closest to a false merge):")
            for rec in block["highest_scoring_distinct"]:
                lines.append(
                    f"    {rec['neo4j_score']:.4f}  [{rec['slice']}/{rec['entity_type']}] "
                    f"{rec['a']!r} vs {rec['b']!r}"
                )
            lines.append("  lowest-scoring duplicate pairs (closest to a missed merge):")
            for rec in block["lowest_scoring_duplicates"]:
                lines.append(
                    f"    {rec['neo4j_score']:.4f}  [{rec['slice']}/{rec['entity_type']}] "
                    f"{rec['a']!r} vs {rec['b']!r}"
                )
    return "\n".join(lines) + "\n"
