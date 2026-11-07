"""Dev and held-out extraction scores side by side, from extraction service gauntlet runs.

A candidate tuned on the dev probes is judged on probes it was never tuned on, so the
held-out column has to sit next to the dev one, with the run identity of each, before the
numbers mean anything. This command prints that one table, a column per run.

It reads only what `scripts.model_bench.probes.extraction_service` wrote into each run
directory: `extraction_service_summary.json` and `extraction_service.jsonl`. It never opens
a gold corpus or any probe text; the gold corpus is shown by the file name the summary
records. Every score and interval is the summary's own value. Two cells are derived from
what the gauntlet wrote, because it writes no value for them:

- entity F1, as 2PR/(P+R) of the summary's entity precision and recall, the formula the
  gauntlet uses for relation F1;
- the list-only row, the relation precision, recall and F1 pooled over the rows of probes
  whose tag slug starts with `list-` (tp is `gold_relationships - rel_fn`, the derivation
  the gauntlet's bootstrap uses), with the scorer's vacuous default of 1.0 for an empty
  denominator.

Of the F1 cells, relation F1 carries the gauntlet's cluster bootstrap when the run recorded
one (a summary written before the gauntlet recorded it shows `[no interval]`); the gauntlet
computes no interval for a derived cell, so entity F1 and the list-only row carry none.

A run with no run identity (a service from before contract 1.2.0, or a gauntlet from
before it recorded one) shows as "unrecorded". Runs whose recorded identities differ, a
run whose identity is unrecorded beside others, and a run with results stamped with an
identity other than its own each get a `[WARN]` line on stderr and in the report.

Usage:
    python -m scripts.eval_harness.extraction_report --run DIR [--run DIR ...] --out FILE.md

Exit status: 0 when the report is written; 2 when a run directory is unreadable, in which
case nothing is written.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

SUMMARY_FILE = "extraction_service_summary.json"
ROWS_FILE = "extraction_service.jsonl"

EXIT_OK = 0
EXIT_UNREADABLE = 2

UNRECORDED = "unrecorded"
NOT_AVAILABLE = "n/a"
NO_INTERVAL = "[no interval]"

LIST_SLUG_PREFIX = "list-"
# Gold tags are `<set>-<number>-<slug>`: `ext-01-uses`, `dvl-03-list-ref-second-database`.
_TAG_RE = re.compile(r"^[a-z]+-\d+-(?P<slug>.+)$")


class RunUnreadableError(Exception):
    """A run directory whose gauntlet output cannot be read as the gauntlet writes it."""


@dataclass(frozen=True, slots=True)
class Run:
    """One gauntlet run directory, as read from disk."""

    label: str
    summary: dict[str, Any]
    rows: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise RunUnreadableError(f"cannot read {path.name}: {exc}") from exc


def load_run(run_dir: Path) -> Run:
    """Read one run directory's summary and rows.

    Raises:
        RunUnreadableError: a file is missing, is not UTF-8 JSON, or is not the shape the
            gauntlet writes (an object, and one object per non-blank line).
    """
    try:
        summary = json.loads(_read_text(run_dir / SUMMARY_FILE))
    except json.JSONDecodeError as exc:
        raise RunUnreadableError(f"{SUMMARY_FILE} is not JSON: {exc}") from exc
    if not isinstance(summary, dict):
        raise RunUnreadableError(f"{SUMMARY_FILE} is not a JSON object")

    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(_read_text(run_dir / ROWS_FILE).splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RunUnreadableError(f"{ROWS_FILE} line {line_no} is not JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise RunUnreadableError(f"{ROWS_FILE} line {line_no} is not a JSON object")
        rows.append(row)
    return Run(label=str(run_dir), summary=summary, rows=rows)


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _interval(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    lo, hi = _number(value[0]), _number(value[1])
    if lo is None or hi is None:
        return None
    return lo, hi


def _fmt(value: float | None) -> str:
    return NOT_AVAILABLE if value is None else f"{value:.4f}"


def _with_interval(value: Any, interval: Any, suffix: str = "") -> str:
    point = _number(value)
    if point is None:
        return NOT_AVAILABLE
    bounds = _interval(interval)
    if bounds is None:
        return f"{point:.4f} {NO_INTERVAL}"
    return f"{point:.4f} [{bounds[0]:.4f}, {bounds[1]:.4f}]{suffix}"


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def run_identity(run: Run) -> str | None:
    """The run's identity as the summary records it, or None when unrecorded."""
    value = run.summary.get("run_identity")
    return value if isinstance(value, str) and value else None


def _corpus_cell(summary: dict[str, Any]) -> str:
    corpus = summary.get("gold_corpus")
    if not isinstance(corpus, str) or not corpus:
        return NOT_AVAILABLE
    # The summary may have been written on another OS; take the last part either way.
    return PureWindowsPath(PurePosixPath(corpus).name).name


def _probes_cell(summary: dict[str, Any]) -> str:
    matched, total = summary.get("matched_probes"), summary.get("total_probes")
    if not isinstance(matched, int) or not isinstance(total, int):
        return NOT_AVAILABLE
    return f"{matched}/{total}"


def _outcomes_cell(summary: dict[str, Any]) -> str:
    classes = summary.get("failure_classes")
    if not isinstance(classes, dict):
        return NOT_AVAILABLE
    parts = [f"{name} {count}" for name, count in classes.items() if count]
    return ", ".join(parts) if parts else "none"


def _bootstrap_cell(summary: dict[str, Any]) -> str:
    boot = summary.get("bootstrap")
    if not isinstance(boot, dict) or "B" not in boot:
        return NOT_AVAILABLE
    return f"B={boot['B']}, confidence {boot.get('confidence', NOT_AVAILABLE)}"


def _stage2_cell(rows: list[dict[str, Any]]) -> str:
    calls = [
        r["stage2_calls"]
        for r in rows
        if isinstance(r.get("stage2_calls"), int) and not isinstance(r["stage2_calls"], bool)
    ]
    return str(sum(calls)) if calls else UNRECORDED


def is_list_probe(tag: Any) -> bool:
    """True for a probe whose tag slug starts with `list-`."""
    if not isinstance(tag, str):
        return False
    match = _TAG_RE.match(tag)
    return match is not None and match.group("slug").startswith(LIST_SLUG_PREFIX)


def _list_only_cell(rows: list[dict[str, Any]]) -> str:
    list_rows = [r for r in rows if is_list_probe(r.get("id"))]
    if not list_rows:
        return "no list probes"
    tp = fp = fn = 0
    for row in list_rows:
        gold, rel_fp, rel_fn = (row.get(k) for k in ("gold_relationships", "rel_fp", "rel_fn"))
        if not all(isinstance(v, int) for v in (gold, rel_fp, rel_fn)):
            return NOT_AVAILABLE
        tp += gold - rel_fn
        fp += rel_fp
        fn += rel_fn
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    n = len(list_rows)
    return (
        f"{precision:.4f} / {recall:.4f} / {_fmt(_f1(precision, recall))} "
        f"({n} probe{'' if n == 1 else 's'})"
    )


def _column(run: Run) -> dict[str, str]:
    s = run.summary
    boot = s.get("bootstrap") if isinstance(s.get("bootstrap"), dict) else {}
    wilson = s.get("wilson") if isinstance(s.get("wilson"), dict) else {}
    identity = run_identity(run)
    return {
        "gold corpus": _corpus_cell(s),
        "run identity": UNRECORDED if identity is None else f"`{identity}`",
        "probes matched": _probes_cell(s),
        "entity precision": _with_interval(s.get("entity_precision"), boot.get("entity_precision")),
        "entity recall": _with_interval(s.get("entity_recall"), boot.get("entity_recall")),
        "entity F1": _fmt(_f1(_number(s.get("entity_precision")), _number(s.get("entity_recall")))),
        "relation precision": _with_interval(s.get("rel_precision"), boot.get("rel_precision")),
        "relation recall": _with_interval(s.get("rel_recall"), boot.get("rel_recall")),
        "relation F1": _with_interval(s.get("rel_f1"), boot.get("rel_f1")),
        "typing accuracy": _with_interval(
            s.get("typing_accuracy"), wilson.get("typing_accuracy"), " Wilson"
        ),
        "RELATED_TO rate": _fmt(_number(s.get("related_to_rate"))),
        "outcomes": _outcomes_cell(s),
        "cases not run": str(s.get("cases_not_run", NOT_AVAILABLE)),
        "bootstrap": _bootstrap_cell(s),
        "Stage 2 calls": _stage2_cell(run.rows),
        "list-only relation P / R / F1": _list_only_cell(run.rows),
    }


# ---------------------------------------------------------------------------
# Warnings and rendering
# ---------------------------------------------------------------------------


def identity_warnings(runs: list[Run]) -> list[str]:
    """`[WARN]` lines for runs that cannot be shown to share one run identity."""
    warnings: list[str] = []
    recorded = [(run.label, rid) for run in runs if (rid := run_identity(run)) is not None]
    if len({rid for _, rid in recorded}) > 1:
        pairs = ", ".join(f"{label}={rid}" for label, rid in recorded)
        warnings.append(f"[WARN] run identities differ: {pairs}")
    if len(runs) > 1:
        for run in runs:
            if run_identity(run) is None:
                warnings.append(
                    f"[WARN] run identity {UNRECORDED} for {run.label}: "
                    "cannot confirm the runs share one"
                )
    for run in runs:
        mismatches = run.summary.get("run_identity_mismatches")
        if isinstance(mismatches, list) and mismatches:
            warnings.append(
                f"[WARN] {run.label}: {len(mismatches)} result(s) stamped with a run identity "
                "other than the run's"
            )
    return warnings


def render_report(runs: list[Run], warnings: list[str]) -> str:
    """The Markdown report: one table, a column per run, then the notes and warnings."""
    columns = [_column(run) for run in runs]
    lines = [
        "# Extraction scores by run",
        "",
        "| metric | " + " | ".join(run.label for run in runs) + " |",
        "|---" * (len(runs) + 1) + "|",
    ]
    for label in columns[0]:
        lines.append(f"| {label} | " + " | ".join(col[label] for col in columns) + " |")
    lines += [
        "",
        "Precision and recall intervals are the gauntlet's cluster bootstrap by probe id; "
        "typing carries its Wilson interval. Relation F1 carries the gauntlet's cluster "
        "bootstrap when the run recorded one. Entity F1 and the list-only row carry no "
        "interval. Entity F1 is 2PR/(P+R) of the precision and recall shown. The list-only "
        "row pools the relation counts of probes whose tag slug starts with `list-`.",
    ]
    if warnings:
        lines += ["", "## Warnings", ""]
        lines += [f"- {w}" for w in warnings]
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.eval_harness.extraction_report",
        description="Print extraction gauntlet runs side by side, a column per run.",
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=Path,
        help="a gauntlet output directory; repeat for each run, in column order",
    )
    parser.add_argument("--out", required=True, type=Path, help="the Markdown file to write")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs: list[Run] = []
    for run_dir in args.run:
        try:
            runs.append(load_run(run_dir))
        except RunUnreadableError as exc:
            print(f"[ERROR] run {run_dir}: {exc}", file=sys.stderr)
            return EXIT_UNREADABLE
    warnings = identity_warnings(runs)
    for warning in warnings:
        print(warning, file=sys.stderr)
    report = render_report(runs, warnings)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(report, end="")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
