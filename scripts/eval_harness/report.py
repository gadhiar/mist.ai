"""Markdown report generator for eval harness runs.

Consumes a RunScores structure from scorers.py and writes a human-readable
comparison report to disk. The report is designed to be read in Obsidian
and to inform the Phase 2 A/B decision for MIST's model backend.

Sections:
    1. Run metadata (timestamp, candidates, tests, iterations)
    2. Headline table: aggregate quality vs mean tokens/second
    3. Per-test score matrix (rows = candidates, columns = tests)
    4. Speed breakdown (tps, p50, p95)
    5. Winners per category
    6. Per-candidate failure drill-down (first 5 failing cases)
    7. Raw data pointer (JSONL paths)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from scripts.eval_harness.scorers import RunScores

logger = logging.getLogger(__name__)

MAX_FAIL_EXAMPLES = 5
RESPONSE_PREVIEW_CHARS = 280


def generate_markdown_report(
    output_path: Path,
    run_scores: RunScores,
    *,
    candidates: list[Any],
    test_files: list[Any],
    run_timestamp: str,
) -> None:
    """Write a comparison report. candidates and test_files pass through from run.py."""
    candidate_by_id = {c.id: c for c in candidates}
    test_order = [tf.name for tf in test_files]

    lines: list[str] = []
    _write_header(lines, run_timestamp, candidates, test_files)
    _write_headline_table(lines, run_scores, candidate_by_id, test_order)
    _write_per_test_matrix(lines, run_scores, candidate_by_id, test_order)
    _write_speed_breakdown(lines, run_scores, candidate_by_id)
    _write_winners(lines, run_scores, candidate_by_id, test_order)
    _write_failure_drill_down(lines, run_scores, candidate_by_id)
    _write_raw_data_pointer(lines, output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote markdown report: %s", output_path)


# ---------------------------------------------------------------------------
# Section writers
# ---------------------------------------------------------------------------


def _write_header(
    lines: list[str],
    run_timestamp: str,
    candidates: list[Any],
    test_files: list[Any],
) -> None:
    lines.append("# MIST Model Backend A/B Evaluation Report")
    lines.append("")
    lines.append(f"**Run timestamp:** {run_timestamp}")
    lines.append(f"**Candidates:** {len(candidates)}")
    lines.append(f"**Tests:** {len(test_files)}")
    lines.append("")
    lines.append("## Candidates")
    lines.append("")
    lines.append("| ID | Display Name | Tier | Arch | Quant | Size (GB) |")
    lines.append("|---|---|---|---|---|---|")
    for c in candidates:
        lines.append(
            f"| `{c.id}` | {c.display_name} | {c.tier} | {c.architecture} "
            f"| {c.quant} | {c.size_gb:.1f} |"
        )
    lines.append("")

    lines.append("## Tests")
    lines.append("")
    for tf in test_files:
        case_count = len(tf.cases)
        lines.append(f"- **{tf.name}** ({case_count} cases) — {tf.description}")
    lines.append("")


def _write_headline_table(
    lines: list[str],
    run_scores: RunScores,
    candidate_by_id: dict[str, Any],
    test_order: list[str],
) -> None:
    lines.append("## Headline: Quality vs Speed")
    lines.append("")
    lines.append(
        "| Candidate | Quality (mean) | Tokens/s | p50 latency (ms) | p95 latency (ms) | Errors |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    sorted_rows = sorted(
        run_scores.per_candidate.items(),
        key=lambda kv: kv[1].aggregate_quality_score(),
        reverse=True,
    )
    for candidate_id, scores in sorted_rows:
        display = _display_name(candidate_id, candidate_by_id)
        quality = scores.aggregate_quality_score()
        tps = scores.mean_tokens_per_second
        p50 = scores.p50_total_time_ms
        p95 = scores.p95_total_time_ms
        lines.append(
            f"| {display} | {quality:.3f} | {tps:.1f} | {p50:.0f} | "
            f"{p95:.0f} | {scores.error_count} |"
        )
    lines.append("")


def _write_per_test_matrix(
    lines: list[str],
    run_scores: RunScores,
    candidate_by_id: dict[str, Any],
    test_order: list[str],
) -> None:
    lines.append("## Per-Test Score Matrix")
    lines.append("")
    header = "| Candidate |"
    divider = "|---|"
    for name in test_order:
        header += f" {name} |"
        divider += "---:|"
    lines.append(header)
    lines.append(divider)

    for candidate_id, scores in run_scores.per_candidate.items():
        display = _display_name(candidate_id, candidate_by_id)
        row = f"| {display} |"
        for name in test_order:
            ts = scores.per_test.get(name)
            if ts is None:
                row += " n/a |"
            else:
                row += f" {ts.mean_score:.2f} ({ts.pass_count}/{ts.pass_count + ts.fail_count}) |"
        lines.append(row)
    lines.append("")


def _write_speed_breakdown(
    lines: list[str],
    run_scores: RunScores,
    candidate_by_id: dict[str, Any],
) -> None:
    lines.append("## Speed Breakdown")
    lines.append("")
    lines.append(
        "| Candidate | Mean tokens/s | p50 latency | p95 latency "
        "| Prompt tokens | Completion tokens |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for candidate_id, scores in run_scores.per_candidate.items():
        display = _display_name(candidate_id, candidate_by_id)
        lines.append(
            f"| {display} | {scores.mean_tokens_per_second:.1f} "
            f"| {scores.p50_total_time_ms:.0f} ms "
            f"| {scores.p95_total_time_ms:.0f} ms "
            f"| {scores.prompt_tokens_total} "
            f"| {scores.completion_tokens_total} |"
        )
    lines.append("")


def _write_winners(
    lines: list[str],
    run_scores: RunScores,
    candidate_by_id: dict[str, Any],
    test_order: list[str],
) -> None:
    lines.append("## Winners per Category")
    lines.append("")
    for name in test_order:
        best_id: str | None = None
        best_score = -1.0
        for candidate_id, scores in run_scores.per_candidate.items():
            ts = scores.per_test.get(name)
            if ts is None:
                continue
            if ts.mean_score > best_score:
                best_score = ts.mean_score
                best_id = candidate_id
        if best_id is None:
            continue
        display = _display_name(best_id, candidate_by_id)
        lines.append(f"- **{name}**: {display} ({best_score:.3f})")

    best_overall_id: str | None = None
    best_overall_score = -1.0
    for candidate_id, scores in run_scores.per_candidate.items():
        q = scores.aggregate_quality_score()
        if q > best_overall_score:
            best_overall_score = q
            best_overall_id = candidate_id
    if best_overall_id is not None:
        lines.append("")
        lines.append(
            f"**Overall quality leader:** "
            f"{_display_name(best_overall_id, candidate_by_id)} "
            f"({best_overall_score:.3f})"
        )

    best_speed_id: str | None = None
    best_speed = -1.0
    for candidate_id, scores in run_scores.per_candidate.items():
        if scores.mean_tokens_per_second > best_speed:
            best_speed = scores.mean_tokens_per_second
            best_speed_id = candidate_id
    if best_speed_id is not None:
        lines.append(
            f"**Speed leader:** {_display_name(best_speed_id, candidate_by_id)} "
            f"({best_speed:.1f} tokens/s)"
        )
    lines.append("")


def _write_failure_drill_down(
    lines: list[str],
    run_scores: RunScores,
    candidate_by_id: dict[str, Any],
) -> None:
    lines.append("## Failure Drill-Down")
    lines.append("")
    for candidate_id, scores in run_scores.per_candidate.items():
        display = _display_name(candidate_id, candidate_by_id)
        lines.append(f"### {display}")
        lines.append("")
        any_failure = False
        for test_name, ts in scores.per_test.items():
            failures = [cs for cs in ts.case_scores if not cs.passed][:MAX_FAIL_EXAMPLES]
            if not failures:
                continue
            any_failure = True
            lines.append(f"**{test_name}** — {ts.fail_count} failures")
            lines.append("")
            for cs in failures:
                lines.append(f"- `{cs.case_id}` score={cs.score:.2f}")
                if cs.error:
                    lines.append(f"  - error: `{cs.error}`")
                errors = cs.breakdown.get("errors")
                if errors:
                    for err in errors[:3]:
                        lines.append(f"  - {err}")
            lines.append("")
        if not any_failure:
            lines.append("_No failures._")
            lines.append("")


def _write_raw_data_pointer(lines: list[str], report_path: Path) -> None:
    lines.append("## Raw Data")
    lines.append("")
    lines.append(f"Per-candidate JSONL results live in: `{report_path.parent}`")
    lines.append("")
    lines.append(
        "Each JSONL line is one `CaseResult` with the full request/response payload, "
        "tool calls, token usage, and timing. Score breakdowns are re-derivable by "
        "rerunning `scorers.score_run()` against the JSONL files."
    )
    lines.append("")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _display_name(candidate_id: str, candidate_by_id: dict[str, Any]) -> str:
    c = candidate_by_id.get(candidate_id)
    return c.display_name if c is not None else candidate_id


# ---------------------------------------------------------------------------
# Standalone helper: score existing results without rerunning models
# ---------------------------------------------------------------------------


def rescore_from_jsonl(
    jsonl_dir: Path,
    test_files: list[Any],
    candidates: list[Any],
) -> RunScores:
    """Rescore from an existing results directory without rerunning the models.

    Useful when scoring logic changes after a run has already executed.
    """
    from scripts.eval_harness.scorers import score_run

    jsonl_paths = sorted(jsonl_dir.glob("*.jsonl"))
    return score_run(jsonl_paths, test_files)


def dump_run_scores_json(run_scores: RunScores, path: Path) -> None:
    """Dump RunScores to a JSON file for offline analysis."""
    payload: dict[str, Any] = {"per_candidate": {}}
    for candidate_id, scores in run_scores.per_candidate.items():
        payload["per_candidate"][candidate_id] = {
            "mean_tokens_per_second": scores.mean_tokens_per_second,
            "p50_total_time_ms": scores.p50_total_time_ms,
            "p95_total_time_ms": scores.p95_total_time_ms,
            "aggregate_quality_score": scores.aggregate_quality_score(),
            "error_count": scores.error_count,
            "per_test": {
                name: {
                    "mean_score": ts.mean_score,
                    "pass_count": ts.pass_count,
                    "fail_count": ts.fail_count,
                    "error_count": ts.error_count,
                }
                for name, ts in scores.per_test.items()
            },
        }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
