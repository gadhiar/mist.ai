"""V8 edge-production probe set scorer.

Joins the V8 input JSONL against MIST's MIST_DEBUG_JSONL `phase: llm_call`
records (filtered to `call_site = "extraction.ontology"`) and computes
per-edge recall against the design-doc acceptance thresholds.

Companion to:
    data/ingest/v8-edge-production-inputs.jsonl       (probe set)
    scripts/eval_harness/v8_probe_set_design.md       (acceptance criteria)

Usage:
    python scripts/eval_harness/score_v8_probe_run.py \
        --input data/ingest/v8-edge-production-inputs.jsonl \
        --debug-jsonl data/runtime/mist-debug.jsonl \
        --session-id v8-probe-20260427-200530

Prerequisites for the debug JSONL (set as env vars on the backend process):
    MIST_DEBUG_JSONL=<path>          # base sink (per-turn records)
    MIST_DEBUG_LLM_JSONL=1            # phase: llm_call gate

The scorer reads BOTH `phase: turn` records (for utterance -> event_id lookup)
AND `phase: llm_call` records (for the extraction JSON). The chain is:

    probe.utterance
        -> turn.event_id           (utterance match)
            -> llm_call records    (event_id match, call_site filter)
                -> response.content -> parsed entities + relationships

Default output is a markdown report on stdout. `--output` writes to a file;
`--json-output` emits a machine-readable dump. `--strict` exits non-zero
when the design-doc thresholds are not met.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Acceptance thresholds -- mirror scripts/eval_harness/v8_probe_set_design.md
# "Acceptance criteria". The design doc is source of truth; if thresholds
# move there, move them here too.
# ---------------------------------------------------------------------------

PER_BUCKET_RECALL_THRESHOLD = 0.75
OVERALL_RECALL_THRESHOLD = 0.70
NEGATIVE_FALSE_POSITIVE_LIMIT = 0  # discrete pass/fail

# Tag schema convention -- matches V7. Negative-control probes carry "-neg-".
NEGATIVE_TAG_INFIX = "-neg-"

# Only the ontology extractor produces the structured {"entities": [...],
# "relationships": [...]} JSON we score against. scope_classifier returns a
# single label; internal_derivation runs on a different code path (MIST
# learning events, not user-facing extraction).
EXTRACTION_CALL_SITE = "extraction.ontology"

# The four post-MVP additive edges this probe set measures end-to-end
# production for. Mirrors NEW edges added to backend/knowledge/ontologies/
# v1_0_0.py on 2026-04-22 (OCCURRED_ON, HAS_METRIC, REFERENCES_DOCUMENT,
# PRECEDED_BY).
NEW_EDGE_TYPES: frozenset[str] = frozenset(
    {"OCCURRED_ON", "HAS_METRIC", "REFERENCES_DOCUMENT", "PRECEDED_BY"}
)


# ---------------------------------------------------------------------------
# Domain dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class V8Probe:
    """One probe row from data/ingest/v8-edge-production-inputs.jsonl."""

    tag: str
    utterance: str
    expected_edges: tuple[str, ...]
    expected_entities: tuple[str, ...]
    rationale: str | None

    @property
    def is_negative(self) -> bool:
        """True for negative-control probes (tag carries the -neg- infix)."""
        return NEGATIVE_TAG_INFIX in self.tag


@dataclass(frozen=True, slots=True)
class TurnIndexEntry:
    """A phase=turn record's identifying fields, for joining to extraction."""

    event_id: str | None
    session_id: str | None
    utterance: str


@dataclass(frozen=True, slots=True)
class ExtractionRecord:
    """A phase=llm_call record with call_site=extraction.ontology, parsed."""

    event_id: str | None
    session_id: str | None
    extracted_entity_types: frozenset[str]
    extracted_relationship_types: frozenset[str]
    parse_ok: bool
    raw_response: str


@dataclass(frozen=True, slots=True)
class ProbeOutcome:
    """A probe joined to its aggregated extraction observations."""

    probe: V8Probe
    extracted_edges: frozenset[str]
    extracted_entities: frozenset[str]
    matched: bool
    parse_ok: bool


@dataclass(slots=True)
class V8Report:
    """Aggregated outcomes + headline metrics for one V8 run."""

    outcomes: list[ProbeOutcome] = field(default_factory=list)

    @property
    def positives(self) -> list[ProbeOutcome]:
        """Outcomes for probes labelled positive (edge expected to fire)."""
        return [o for o in self.outcomes if not o.probe.is_negative]

    @property
    def negatives(self) -> list[ProbeOutcome]:
        """Outcomes for probes labelled negative (no new edge expected)."""
        return [o for o in self.outcomes if o.probe.is_negative]

    @property
    def missing(self) -> int:
        """Count of probes with no matching extraction record."""
        return sum(1 for o in self.outcomes if not o.matched)

    def per_bucket_stats(self) -> dict[str, dict[str, int]]:
        """For each new edge type, count expected/produced across positives.

        A probe contributes to bucket X's expected count when X appears in
        its expected_edges. A probe contributes to bucket X's produced count
        when X also appears in extracted_edges.
        """
        stats: dict[str, dict[str, int]] = {
            edge: {"expected": 0, "produced": 0} for edge in sorted(NEW_EDGE_TYPES)
        }
        for o in self.positives:
            for edge in o.probe.expected_edges:
                if edge in NEW_EDGE_TYPES:
                    stats[edge]["expected"] += 1
                    if edge in o.extracted_edges:
                        stats[edge]["produced"] += 1
        return stats

    def per_bucket_recall(self) -> dict[str, float]:
        """Per-edge recall = produced / expected, 0 when no probes expected it."""
        return {
            edge: (s["produced"] / s["expected"]) if s["expected"] else 0.0
            for edge, s in self.per_bucket_stats().items()
        }

    @property
    def overall_recall(self) -> float:
        """Sum-of-produced over sum-of-expected across all positives."""
        total_expected = 0
        total_produced = 0
        for o in self.positives:
            for edge in o.probe.expected_edges:
                if edge in NEW_EDGE_TYPES:
                    total_expected += 1
                    if edge in o.extracted_edges:
                        total_produced += 1
        return total_produced / total_expected if total_expected else 0.0

    @property
    def negative_false_positives(self) -> int:
        """Count of negative probes that produced any of the four new edges.

        A negative probe is engineered such that NO new edge should land. The
        discrete 0/4 rule means a single false positive fails acceptance; this
        property counts the violations across all negative probes.
        """
        return sum(1 for o in self.negatives if o.matched and (o.extracted_edges & NEW_EDGE_TYPES))

    def acceptance_pass(self) -> bool:
        """True when per-bucket, overall, and negative-FP gates all clear."""
        per_bucket_ok = all(
            r >= PER_BUCKET_RECALL_THRESHOLD for r in self.per_bucket_recall().values()
        )
        overall_ok = self.overall_recall >= OVERALL_RECALL_THRESHOLD
        neg_ok = self.negative_false_positives <= NEGATIVE_FALSE_POSITIVE_LIMIT
        return per_bucket_ok and overall_ok and neg_ok


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def iter_probes(input_path: Path) -> Iterator[V8Probe]:
    """Stream probes from the V8 input JSONL.

    Raises ValueError on malformed JSON to fail loudly: the V8 input is
    curated research data, not log output, so any decode error is a real bug.
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
            eb = rec.get("expected_behavior") or {}
            yield V8Probe(
                tag=rec.get("tag", ""),
                utterance=rec.get("utterance", ""),
                expected_edges=tuple(eb.get("expected_edges", []) or []),
                expected_entities=tuple(eb.get("expected_entities", []) or []),
                rationale=eb.get("rationale"),
            )


def iter_debug_records(
    debug_path: Path,
    session_id: str | None = None,
) -> Iterator[dict]:
    """Yield every record in the debug JSONL with optional session_id filter.

    Lines that fail to parse are reported to stderr and skipped. The debug
    JSONL is append-only and may have a truncated tail from a mid-write
    crash; surfacing the warn line lets a caller spot real corruption while
    not blocking on transient writer failures.
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
            if session_id is not None and rec.get("session_id") != session_id:
                continue
            yield rec


def parse_extraction_json(
    content: str,
) -> tuple[bool, frozenset[str], frozenset[str]]:
    """Parse an extraction LLM response content.

    Returns (parse_ok, entity_types, relationship_types). Tries strict JSON
    first, falls back to regex-extracted-first-object. Mirrors the lenient
    parsing strategy in scripts/eval_harness/scorers.py:_parse_json_lenient.
    """
    parsed: object = None
    if content:
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            parsed = json.loads(content)
    if not isinstance(parsed, dict):
        match = re.search(r"\{.*\}", content or "", flags=re.DOTALL)
        if match is not None:
            with contextlib.suppress(json.JSONDecodeError):
                parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        return False, frozenset(), frozenset()
    entity_types: set[str] = set()
    for ent in parsed.get("entities", []) or []:
        if isinstance(ent, dict):
            t = ent.get("type")
            if isinstance(t, str):
                entity_types.add(t)
    rel_types: set[str] = set()
    for rel in parsed.get("relationships", []) or []:
        if isinstance(rel, dict):
            t = rel.get("type")
            if isinstance(t, str):
                rel_types.add(t)
    return True, frozenset(entity_types), frozenset(rel_types)


# ---------------------------------------------------------------------------
# Indexing + join + verdict
# ---------------------------------------------------------------------------


def build_indices(
    records: Iterator[dict],
) -> tuple[dict[str, list[TurnIndexEntry]], dict[str, list[ExtractionRecord]]]:
    """Single-pass index of turn + extraction records from the debug JSONL.

    Returns:
        utterance_index: utterance -> list of turn entries (for join). Multiple
            entries are kept; first-occurrence wins at lookup time.
        extraction_index: event_id -> list of extraction records. Multiple
            extraction calls per event_id (rare; retries) are kept and the
            entity/relationship types are unioned at scoring time.
    """
    utterance_index: dict[str, list[TurnIndexEntry]] = {}
    extraction_index: dict[str, list[ExtractionRecord]] = {}
    for rec in records:
        phase = rec.get("phase")
        if phase == "turn":
            utterance = rec.get("utterance", "")
            if utterance:
                utterance_index.setdefault(utterance, []).append(
                    TurnIndexEntry(
                        event_id=rec.get("event_id"),
                        session_id=rec.get("session_id"),
                        utterance=utterance,
                    )
                )
        elif phase == "llm_call":
            if rec.get("call_site") != EXTRACTION_CALL_SITE:
                continue
            event_id = rec.get("event_id")
            if event_id is None:
                continue
            response = rec.get("response") or {}
            content = response.get("content") or ""
            parse_ok, entity_types, rel_types = parse_extraction_json(content)
            extraction_index.setdefault(event_id, []).append(
                ExtractionRecord(
                    event_id=event_id,
                    session_id=rec.get("session_id"),
                    extracted_entity_types=entity_types,
                    extracted_relationship_types=rel_types,
                    parse_ok=parse_ok,
                    raw_response=content,
                )
            )
    return utterance_index, extraction_index


def score_run(
    probes: list[V8Probe],
    utterance_index: dict[str, list[TurnIndexEntry]],
    extraction_index: dict[str, list[ExtractionRecord]],
) -> V8Report:
    """Join probes against indices and emit a V8Report.

    For each probe, looks up all turn entries by utterance, then aggregates
    extracted entity/relationship types from the extraction records linked
    by event_id. A probe is matched=True iff at least one extraction record
    was found.
    """
    outcomes: list[ProbeOutcome] = []
    for probe in probes:
        turn_entries = utterance_index.get(probe.utterance, [])
        all_edges: set[str] = set()
        all_entities: set[str] = set()
        any_parse_ok = False
        matched = False
        for entry in turn_entries:
            if entry.event_id is None:
                continue
            extractions = extraction_index.get(entry.event_id, [])
            for ext in extractions:
                matched = True
                if ext.parse_ok:
                    any_parse_ok = True
                all_edges |= ext.extracted_relationship_types
                all_entities |= ext.extracted_entity_types
        outcomes.append(
            ProbeOutcome(
                probe=probe,
                extracted_edges=frozenset(all_edges),
                extracted_entities=frozenset(all_entities),
                matched=matched,
                parse_ok=any_parse_ok if matched else False,
            )
        )
    return V8Report(outcomes=outcomes)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render_markdown(report: V8Report) -> str:
    """Render a one-page markdown summary suitable for stdout or PR review."""
    lines: list[str] = [
        "# V8 Probe Set -- Edge-Production Score Report",
        "",
        f"**Probes total:** {len(report.outcomes)} "
        f"({len(report.positives)} positive + {len(report.negatives)} negative)",
    ]
    matched = len(report.outcomes) - report.missing
    lines.append(f"**Matched against debug JSONL:** {matched}/{len(report.outcomes)}")
    if report.missing:
        lines.append(f"**Missing (no extraction record for utterance):** {report.missing}")
    lines.append("")
    lines.append("## Headline Metrics")
    lines.append("")
    lines.append(f"- Overall recall: {report.overall_recall:.3f}")
    lines.append(
        f"- Negative false positives (any new edge on a negative probe): "
        f"{report.negative_false_positives} / {len(report.negatives)}"
    )
    lines.append("")
    lines.append("## Per-Edge Recall")
    lines.append("")
    lines.append("| Edge type | Expected | Produced | Recall | Pass |")
    lines.append("|-----------|---------:|---------:|-------:|:----:|")
    for edge in sorted(NEW_EDGE_TYPES):
        stats = report.per_bucket_stats()[edge]
        recall = stats["produced"] / stats["expected"] if stats["expected"] else 0.0
        passed = recall >= PER_BUCKET_RECALL_THRESHOLD
        lines.append(
            f"| {edge} | {stats['expected']} | {stats['produced']} | "
            f"{recall:.2f} | {'PASS' if passed else 'FAIL'} |"
        )
    lines.append("")
    lines.append("## Acceptance Criteria")
    lines.append("")
    bucket_recalls = report.per_bucket_recall()
    bucket_ok = all(r >= PER_BUCKET_RECALL_THRESHOLD for r in bucket_recalls.values())
    overall_ok = report.overall_recall >= OVERALL_RECALL_THRESHOLD
    neg_ok = report.negative_false_positives <= NEGATIVE_FALSE_POSITIVE_LIMIT
    lines.append(
        f"- [{'PASS' if bucket_ok else 'FAIL'}] "
        f"Per-bucket recall >= {PER_BUCKET_RECALL_THRESHOLD} (each)"
    )
    lines.append(
        f"- [{'PASS' if overall_ok else 'FAIL'}] " f"Overall recall >= {OVERALL_RECALL_THRESHOLD}"
    )
    lines.append(
        f"- [{'PASS' if neg_ok else 'FAIL'}] "
        f"Negative false positives <= {NEGATIVE_FALSE_POSITIVE_LIMIT}"
    )
    lines.append("")
    lines.append(f"**Verdict:** {'PASS' if report.acceptance_pass() else 'FAIL'}")
    lines.append("")
    lines.append("## Per-Probe Outcomes")
    lines.append("")
    for o in report.outcomes:
        if not o.matched:
            status = "MISSING"
        elif not o.parse_ok:
            status = "PARSE_FAIL"
        elif o.probe.is_negative:
            offenders = sorted(o.extracted_edges & NEW_EDGE_TYPES)
            status = f"FP {offenders}" if offenders else "TN"
        else:
            expected_new = set(o.probe.expected_edges) & NEW_EDGE_TYPES
            missing_subset = expected_new - o.extracted_edges
            status = "TP" if not missing_subset else f"PARTIAL (missing: {sorted(missing_subset)})"
        lines.append(f"- **{o.probe.tag}** [{status}]")
        lines.append(f"    - Utterance: {o.probe.utterance!r}")
        lines.append(f"    - Expected edges: {list(o.probe.expected_edges)}")
        lines.append(f"    - Produced edges: {sorted(o.extracted_edges)}")
    lines.append("")
    return "\n".join(lines)


def render_json(report: V8Report) -> str:
    """Render a machine-readable JSON dump of the report."""
    payload = {
        "totals": {
            "probes": len(report.outcomes),
            "positives": len(report.positives),
            "negatives": len(report.negatives),
            "matched": len(report.outcomes) - report.missing,
            "missing": report.missing,
        },
        "headline": {
            "overall_recall": report.overall_recall,
            "negative_false_positives": report.negative_false_positives,
        },
        "per_bucket": {
            edge: {
                "expected": stats["expected"],
                "produced": stats["produced"],
                "recall": (stats["produced"] / stats["expected"] if stats["expected"] else 0.0),
            }
            for edge, stats in report.per_bucket_stats().items()
        },
        "acceptance": {
            "per_bucket_recall_threshold": PER_BUCKET_RECALL_THRESHOLD,
            "overall_recall_threshold": OVERALL_RECALL_THRESHOLD,
            "negative_fp_limit": NEGATIVE_FALSE_POSITIVE_LIMIT,
            "passed": report.acceptance_pass(),
        },
        "outcomes": [
            {
                "tag": o.probe.tag,
                "is_negative": o.probe.is_negative,
                "expected_edges": list(o.probe.expected_edges),
                "expected_entities": list(o.probe.expected_entities),
                "extracted_edges": sorted(o.extracted_edges),
                "extracted_entities": sorted(o.extracted_entities),
                "matched": o.matched,
                "parse_ok": o.parse_ok,
                "utterance": o.probe.utterance,
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
            "Score a V8 edge-production probe run against expected_edges. "
            "See scripts/eval_harness/v8_probe_set_design.md for the "
            "underlying probe design and acceptance criteria."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="V8 input JSONL (typically data/ingest/v8-edge-production-inputs.jsonl)",
    )
    parser.add_argument(
        "--debug-jsonl",
        required=True,
        type=Path,
        help=(
            "Per-turn debug JSONL produced when MIST_DEBUG_JSONL is set. "
            "Requires MIST_DEBUG_LLM_JSONL=1 to capture extraction response content."
        ),
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
    utterance_index, extraction_index = build_indices(
        iter_debug_records(args.debug_jsonl, session_id=args.session_id),
    )
    report = score_run(probes, utterance_index, extraction_index)

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
