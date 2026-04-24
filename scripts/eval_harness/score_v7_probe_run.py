"""V7 tool-heavy probe set scorer.

Joins the V7 input JSONL against MIST's per-turn debug JSONL
(MIST_DEBUG_JSONL) on utterance text within a session_id filter, then
computes precision / recall / per-bucket recall and checks the design-doc
acceptance thresholds.

Companion to:
    data/ingest/v7-tool-heavy-inputs.jsonl       (probe set)
    scripts/eval_harness/v7_probe_set_design.md  (acceptance criteria)

Usage:
    python scripts/eval_harness/score_v7_probe_run.py \
        --input data/ingest/v7-tool-heavy-inputs.jsonl \
        --debug-jsonl data/runtime/mist-debug.jsonl \
        --session-id v7-probe-20260423

Default output is a markdown report on stdout. `--output` writes to a file;
`--json-output` emits a machine-readable dump for downstream consumers.
`--strict` exits non-zero when the design-doc thresholds (precision >= 0.90,
recall >= 0.90, 0/5 false positives on negatives) are not met.

Spec: scripts/eval_harness/v7_probe_set_design.md ("Acceptance criteria",
"How to run") + mist-ai-tool-calling-production-rigor workstream
next_actions[0].
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Acceptance thresholds -- mirror scripts/eval_harness/v7_probe_set_design.md
# "Acceptance criteria" section. The design doc is the source of truth; if
# thresholds change there, change them here too. The intent is parity, not
# a separate scoring policy.
# ---------------------------------------------------------------------------

PRECISION_THRESHOLD = 0.90
RECALL_THRESHOLD = 0.90
NEGATIVE_FALSE_POSITIVE_LIMIT = 0  # 0/5 -- discrete pass/fail

# Tag schema convention: negative-control probes carry the "-neg-" infix in
# their tag (see data/ingest/v7-tool-heavy-inputs.jsonl rows 21-25). If the
# tag schema changes, update this constant + tests/unit/test_score_v7_probe_run.py.
NEGATIVE_TAG_INFIX = "-neg-"


# ---------------------------------------------------------------------------
# Domain dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class V7Probe:
    """One probe row from data/ingest/v7-tool-heavy-inputs.jsonl."""

    tag: str
    utterance: str
    expected_tool: str | None
    rationale: str | None

    @property
    def is_negative(self) -> bool:
        """True for negative-control probes (tag carries the -neg- infix)."""
        return NEGATIVE_TAG_INFIX in self.tag

    @property
    def bucket(self) -> str:
        """Strip the v7-NN- prefix to expose the bucket label.

        v7-01-direct-tech-recall    -> direct-tech-recall
        v7-21-neg-general-knowledge -> neg-general-knowledge
        """
        parts = self.tag.split("-", 2)
        return parts[2] if len(parts) >= 3 else self.tag


@dataclass(frozen=True, slots=True)
class TurnObservation:
    """A single phase=turn record from the MIST_DEBUG_JSONL stream."""

    utterance: str
    session_id: str | None
    event_id: str | None
    tool_calls: tuple[str, ...]

    @property
    def first_tool_call(self) -> str | None:
        """Name of the first tool the model fired across all passes, or None."""
        return self.tool_calls[0] if self.tool_calls else None


@dataclass(frozen=True, slots=True)
class ProbeOutcome:
    """A probe joined to its observation and its verdict."""

    probe: V7Probe
    observation: TurnObservation | None
    verdict: str  # one of: "tp" | "fn" | "tn" | "fp" | "missing"


@dataclass(slots=True)
class V7Report:
    """Aggregated outcomes + headline metrics for one V7 run."""

    outcomes: list[ProbeOutcome] = field(default_factory=list)

    @property
    def positives(self) -> list[ProbeOutcome]:
        """Outcomes for probes labelled positive (tool expected to fire)."""
        return [o for o in self.outcomes if not o.probe.is_negative]

    @property
    def negatives(self) -> list[ProbeOutcome]:
        """Outcomes for probes labelled negative (no tool expected)."""
        return [o for o in self.outcomes if o.probe.is_negative]

    @property
    def true_positives(self) -> int:
        """Count of probes that fired the expected tool."""
        return sum(1 for o in self.outcomes if o.verdict == "tp")

    @property
    def false_negatives(self) -> int:
        """Count of positive probes that fired no tool or the wrong tool."""
        return sum(1 for o in self.outcomes if o.verdict == "fn")

    @property
    def true_negatives(self) -> int:
        """Count of negative probes that correctly stayed silent."""
        return sum(1 for o in self.outcomes if o.verdict == "tn")

    @property
    def false_positives(self) -> int:
        """Count of negative probes that erroneously fired a tool."""
        return sum(1 for o in self.outcomes if o.verdict == "fp")

    @property
    def missing(self) -> int:
        """Count of probes with no matching turn record in the debug JSONL."""
        return sum(1 for o in self.outcomes if o.verdict == "missing")

    @property
    def precision(self) -> float:
        """TP / (TP + FP). 0.0 when no tools fired (no decisions to score)."""
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        """TP / (TP + FN). 0.0 when there are no expected positives."""
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    def acceptance_pass(self) -> bool:
        """True when precision, recall, and the negative-FP rule all clear."""
        return (
            self.precision >= PRECISION_THRESHOLD
            and self.recall >= RECALL_THRESHOLD
            and self.false_positives <= NEGATIVE_FALSE_POSITIVE_LIMIT
        )


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def iter_probes(input_path: Path) -> Iterator[V7Probe]:
    """Stream probes from the V7 input JSONL.

    Raises ValueError on malformed JSON to fail loudly: V7 input is curated
    research data, not log output, so any decode error is a real bug.
    """
    with input_path.open("r", encoding="utf-8") as fh:
        for line_num, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{input_path}:{line_num}: invalid JSON: {exc}") from exc
            expected_behavior = rec.get("expected_behavior") or {}
            yield V7Probe(
                tag=rec.get("tag", ""),
                utterance=rec.get("utterance", ""),
                expected_tool=expected_behavior.get("tool_call"),
                rationale=expected_behavior.get("rationale"),
            )


def iter_turn_observations(
    debug_path: Path,
    session_id: str | None = None,
) -> Iterator[TurnObservation]:
    """Stream phase=turn records from MIST_DEBUG_JSONL.

    Lines that fail to parse are reported to stderr and skipped. The debug
    JSONL is append-only and may have a truncated tail from a mid-write
    crash; surfacing the warn line lets a caller spot real corruption while
    not blocking on transient writer failures.

    When `session_id` is provided, restrict to records with matching
    session_id. Pass None to ingest every turn record in the file.
    """
    with debug_path.open("r", encoding="utf-8") as fh:
        for line_num, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"WARN: {debug_path}:{line_num}: malformed JSON, skipping",
                    file=sys.stderr,
                )
                continue
            if rec.get("phase") != "turn":
                continue
            if session_id is not None and rec.get("session_id") != session_id:
                continue
            tool_calls: list[str] = []
            for pass_rec in rec.get("llm_passes") or []:
                for tc in pass_rec.get("tool_calls") or []:
                    name = tc.get("name") if isinstance(tc, dict) else None
                    if name:
                        tool_calls.append(name)
            yield TurnObservation(
                utterance=rec.get("utterance", ""),
                session_id=rec.get("session_id"),
                event_id=rec.get("event_id"),
                tool_calls=tuple(tool_calls),
            )


# ---------------------------------------------------------------------------
# Join + verdict
# ---------------------------------------------------------------------------


def index_observations_by_utterance(
    observations: Iterator[TurnObservation],
) -> dict[str, TurnObservation]:
    """Index observations by utterance, keeping the FIRST occurrence.

    Replay is sequential, so the first turn record for a given utterance
    captures the model's actual decision for that probe. A later duplicate
    necessarily belongs to a separate turn (a manual retry or test rerun)
    and is not the canonical answer for the V7 run.
    """
    index: dict[str, TurnObservation] = {}
    for obs in observations:
        if obs.utterance and obs.utterance not in index:
            index[obs.utterance] = obs
    return index


def verdict_for(probe: V7Probe, observation: TurnObservation | None) -> str:
    """Compute the verdict per design-doc semantics."""
    if observation is None:
        return "missing"
    actual = observation.first_tool_call
    if probe.is_negative:
        return "tn" if actual is None else "fp"
    return "tp" if actual == probe.expected_tool else "fn"


def score_run(
    probes: list[V7Probe],
    observation_index: dict[str, TurnObservation],
) -> V7Report:
    """Join probes against observations and emit a V7Report."""
    outcomes: list[ProbeOutcome] = []
    for probe in probes:
        obs = observation_index.get(probe.utterance)
        outcomes.append(
            ProbeOutcome(
                probe=probe,
                observation=obs,
                verdict=verdict_for(probe, obs),
            )
        )
    return V7Report(outcomes=outcomes)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render_markdown(report: V7Report) -> str:
    """Render a one-page markdown summary suitable for stdout or PR review."""
    lines: list[str] = [
        "# V7 Probe Set -- Tool-Selection Score Report",
        "",
        f"**Probes total:** {len(report.outcomes)} "
        f"({len(report.positives)} positive + {len(report.negatives)} negative)",
    ]
    matched = len(report.outcomes) - report.missing
    lines.append(f"**Matched against debug JSONL:** {matched}/{len(report.outcomes)}")
    if report.missing:
        lines.append(f"**Missing (no debug record for utterance):** {report.missing}")
    lines.append("")
    lines.append("## Headline Metrics")
    lines.append("")
    fired_total = report.true_positives + report.false_positives
    expected_pos_total = report.true_positives + report.false_negatives
    lines.append(
        f"- Precision: {report.precision:.3f}  "
        f"({report.true_positives} TP / {fired_total} fired)"
    )
    lines.append(
        f"- Recall:    {report.recall:.3f}  "
        f"({report.true_positives} TP / {expected_pos_total} expected positives)"
    )
    lines.append(
        f"- False positives on negatives: " f"{report.false_positives} / {len(report.negatives)}"
    )
    lines.append("")
    lines.append("## Acceptance Criteria")
    lines.append("")
    p_ok = report.precision >= PRECISION_THRESHOLD
    r_ok = report.recall >= RECALL_THRESHOLD
    n_ok = report.false_positives <= NEGATIVE_FALSE_POSITIVE_LIMIT
    lines.append(f"- [{'PASS' if p_ok else 'FAIL'}] Precision >= {PRECISION_THRESHOLD}")
    lines.append(f"- [{'PASS' if r_ok else 'FAIL'}] Recall >= {RECALL_THRESHOLD}")
    lines.append(
        f"- [{'PASS' if n_ok else 'FAIL'}] "
        f"False positives on negatives <= {NEGATIVE_FALSE_POSITIVE_LIMIT}"
    )
    lines.append("")
    lines.append(f"**Verdict:** {'PASS' if report.acceptance_pass() else 'FAIL'}")
    lines.append("")
    lines.append("## Confusion Matrix")
    lines.append("")
    lines.append("|                   | Predicted: tool | Predicted: no-tool | Missing |")
    lines.append("|-------------------|----------------:|-------------------:|--------:|")
    pos_missing = sum(1 for o in report.positives if o.verdict == "missing")
    neg_missing = sum(1 for o in report.negatives if o.verdict == "missing")
    lines.append(
        f"| Expected: tool    | {report.true_positives} | "
        f"{report.false_negatives} | {pos_missing} |"
    )
    lines.append(
        f"| Expected: no-tool | {report.false_positives} | "
        f"{report.true_negatives} | {neg_missing} |"
    )
    lines.append("")
    lines.append("## Per-Bucket Recall (Positives)")
    lines.append("")
    lines.append("| Bucket | Probes | TP | FN | Missing | Recall |")
    lines.append("|--------|-------:|---:|---:|--------:|-------:|")
    bucket_groups: dict[str, list[ProbeOutcome]] = {}
    for o in report.positives:
        bucket_groups.setdefault(o.probe.bucket, []).append(o)
    for bucket in sorted(bucket_groups):
        outs = bucket_groups[bucket]
        tp = sum(1 for o in outs if o.verdict == "tp")
        fn = sum(1 for o in outs if o.verdict == "fn")
        miss = sum(1 for o in outs if o.verdict == "missing")
        denom = tp + fn
        recall = tp / denom if denom else 0.0
        lines.append(f"| {bucket} | {len(outs)} | {tp} | {fn} | {miss} | {recall:.2f} |")
    lines.append("")
    lines.append("## Failures and Misses")
    lines.append("")
    failures = [o for o in report.outcomes if o.verdict in ("fn", "fp", "missing")]
    if not failures:
        lines.append("(none)")
    else:
        for o in failures:
            actual = o.observation.first_tool_call if o.observation else "<no debug record>"
            lines.append(f"- **{o.probe.tag}** [{o.verdict.upper()}]")
            lines.append(f"    - Utterance: {o.probe.utterance!r}")
            lines.append(f"    - Expected: {o.probe.expected_tool!r}")
            lines.append(f"    - Actual:   {actual!r}")
    lines.append("")
    return "\n".join(lines)


def render_json(report: V7Report) -> str:
    """Render a machine-readable JSON dump of the report."""
    payload = {
        "totals": {
            "probes": len(report.outcomes),
            "positives": len(report.positives),
            "negatives": len(report.negatives),
            "matched": len(report.outcomes) - report.missing,
            "missing": report.missing,
            "tp": report.true_positives,
            "fn": report.false_negatives,
            "tn": report.true_negatives,
            "fp": report.false_positives,
        },
        "headline": {
            "precision": report.precision,
            "recall": report.recall,
            "false_positives_on_negatives": report.false_positives,
        },
        "acceptance": {
            "precision_threshold": PRECISION_THRESHOLD,
            "recall_threshold": RECALL_THRESHOLD,
            "negative_fp_limit": NEGATIVE_FALSE_POSITIVE_LIMIT,
            "passed": report.acceptance_pass(),
        },
        "outcomes": [
            {
                "tag": o.probe.tag,
                "bucket": o.probe.bucket,
                "is_negative": o.probe.is_negative,
                "expected_tool": o.probe.expected_tool,
                "actual_tool": (o.observation.first_tool_call if o.observation else None),
                "verdict": o.verdict,
                "utterance": o.probe.utterance,
                "session_id": (o.observation.session_id if o.observation else None),
                "event_id": (o.observation.event_id if o.observation else None),
            }
            for o in report.outcomes
        ],
    }
    return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score a V7 tool-heavy probe run against expected_behavior labels. "
            "See scripts/eval_harness/v7_probe_set_design.md for the underlying "
            "probe design and acceptance criteria."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="V7 input JSONL (typically data/ingest/v7-tool-heavy-inputs.jsonl)",
    )
    parser.add_argument(
        "--debug-jsonl",
        required=True,
        type=Path,
        help="Per-turn debug JSONL produced when MIST_DEBUG_JSONL is set",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Restrict to debug JSONL records with this session_id",
    )
    parser.add_argument(
        "--output",
        default=None,
        type=Path,
        help="Write the markdown report to this path (default: stdout)",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        type=Path,
        help="Optionally write a machine-readable JSON report to this path",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if acceptance criteria fail",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        return 2
    if not args.debug_jsonl.exists():
        print(f"ERROR: debug JSONL not found: {args.debug_jsonl}", file=sys.stderr)
        return 2

    probes = list(iter_probes(args.input))
    observation_index = index_observations_by_utterance(
        iter_turn_observations(args.debug_jsonl, session_id=args.session_id),
    )
    report = score_run(probes, observation_index)

    md = render_markdown(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md, encoding="utf-8")
    else:
        sys.stdout.write(md)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(render_json(report), encoding="utf-8")

    if args.strict and not report.acceptance_pass():
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
