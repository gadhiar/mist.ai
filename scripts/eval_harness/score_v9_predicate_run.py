"""V9 v1.1.0 predicate-coverage probe set scorer.

Joins the V9 input JSONL against MIST's MIST_DEBUG_JSONL `phase: llm_call`
records (filtered to `call_site = "extraction.ontology"`) and computes
per-predicate recall against the design-doc acceptance threshold.

Companion to:
    data/ingest/v9-v1-1-0-predicate-coverage.jsonl  (probe set)
    scripts/eval_harness/v9_probe_set_design.md     (acceptance criteria)

Usage:
    python scripts/eval_harness/score_v9_predicate_run.py \
        --input data/ingest/v9-v1-1-0-predicate-coverage.jsonl \
        --debug-jsonl data/runtime/mist-debug.jsonl \
        --session-id v9-replay-20260508

Prerequisites for the debug JSONL (set as env vars on the backend process):
    MIST_DEBUG_JSONL=<path>          # base sink (per-turn records)
    MIST_DEBUG_LLM_JSONL=1            # phase: llm_call gate

The scorer reads `phase: llm_call` records with
`call_site = "extraction.ontology"` and recovers the utterance from the
extraction request's user message (the extraction llm_call records do
not currently propagate session_id/event_id; same workaround as the V8
scorer's join strategy).

Default output is a markdown report on stdout. `--output` writes to a
file; `--strict` exits non-zero when any predicate fails the per-predicate
recall threshold.
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
# Acceptance threshold -- mirror scripts/eval_harness/v9_probe_set_design.md
# "Acceptance criteria". The design doc is source of truth.
# ---------------------------------------------------------------------------

PER_PREDICATE_RECALL_THRESHOLD = 0.25  # >= 1 of 4 probes must fire

EXTRACTION_CALL_SITE = "extraction.ontology"

EXTRACTION_UTTERANCE_PATTERN = re.compile(r'Utterance:\s*"(.+?)"\s*\n\s*Output:', re.DOTALL)


def _normalize_utterance(s: str) -> str:
    """Normalize utterance text for join purposes.

    Strips surrounding whitespace and collapses internal whitespace runs
    to single spaces. The probe utterance and the extracted utterance
    must agree on this normalization for the join to succeed; without
    it, any prompt-template change that adds/removes whitespace around
    or inside the utterance drops every probe to MISSING.
    """
    return " ".join(s.split())


# The three v1.1.0 predicates that did NOT fire on V6 conversational sample.
# This probe set isolates the question: do they fire under engineered
# triggers? V9 acceptance is presence-of-life, not production quality.
TARGET_PREDICATES: frozenset[str] = frozenset({"MECHANISM_OF", "INPUT_TO", "COMPRISES"})


# ---------------------------------------------------------------------------
# Domain dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class V9Probe:
    """One probe row from data/ingest/v9-v1-1-0-predicate-coverage.jsonl."""

    tag: str
    utterance: str
    expected_edges: tuple[str, ...]
    expected_entities: tuple[str, ...]
    rationale: str | None


@dataclass(frozen=True, slots=True)
class ExtractionRecord:
    """A phase=llm_call record with call_site=extraction.ontology, parsed."""

    extracted_entity_types: frozenset[str]
    extracted_relationship_types: frozenset[str]
    parse_ok: bool


@dataclass(frozen=True, slots=True)
class ProbeOutcome:
    """A probe joined to its aggregated extraction observations."""

    probe: V9Probe
    extracted_edges: frozenset[str]
    extracted_entities: frozenset[str]
    matched: bool
    parse_ok: bool

    @property
    def fired(self) -> bool:
        """True iff every expected predicate appears in the extracted edges."""
        if not self.expected_target_predicates:
            return False
        return all(p in self.extracted_edges for p in self.expected_target_predicates)

    @property
    def expected_target_predicates(self) -> frozenset[str]:
        """The expected_edges intersected with the V9 target set."""
        return frozenset(self.probe.expected_edges) & TARGET_PREDICATES


@dataclass(slots=True)
class V9Report:
    """Aggregated outcomes + headline metrics for one V9 run."""

    outcomes: list[ProbeOutcome] = field(default_factory=list)

    @property
    def missing(self) -> int:
        """Count of probes with no matching extraction record."""
        return sum(1 for o in self.outcomes if not o.matched)

    def per_predicate_stats(self) -> dict[str, dict[str, int]]:
        """Per-predicate counts of expected vs fired across all probes."""
        stats: dict[str, dict[str, int]] = {
            p: {"expected": 0, "fired": 0} for p in sorted(TARGET_PREDICATES)
        }
        for o in self.outcomes:
            for predicate in o.expected_target_predicates:
                stats[predicate]["expected"] += 1
                if predicate in o.extracted_edges:
                    stats[predicate]["fired"] += 1
        return stats

    def per_predicate_recall(self) -> dict[str, float]:
        """Per-predicate recall = fired / expected, 0 when no probes expected it."""
        return {
            p: (s["fired"] / s["expected"]) if s["expected"] else 0.0
            for p, s in self.per_predicate_stats().items()
        }

    def acceptance_pass(self) -> bool:
        """True iff every target predicate cleared the per-predicate recall threshold."""
        return all(
            r >= PER_PREDICATE_RECALL_THRESHOLD for r in self.per_predicate_recall().values()
        )


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def iter_probes(input_path: Path) -> Iterator[V9Probe]:
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
            yield V9Probe(
                tag=rec.get("tag", ""),
                utterance=rec.get("utterance", ""),
                expected_edges=tuple(eb.get("expected_edges", []) or []),
                expected_entities=tuple(eb.get("expected_entities", []) or []),
                rationale=eb.get("rationale"),
            )


def iter_debug_records(debug_path: Path, session_id: str | None = None) -> Iterator[dict]:
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


def extract_utterance_from_request(request: dict) -> str | None:
    if not isinstance(request, dict):
        return None
    messages = request.get("messages") or []
    if not messages:
        return None
    last = messages[-1]
    if not isinstance(last, dict):
        return None
    content = last.get("content") or ""
    match = EXTRACTION_UTTERANCE_PATTERN.search(content)
    return match.group(1) if match else None


def parse_extraction_json(content: str) -> tuple[bool, frozenset[str], frozenset[str]]:
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


# Buffer to add after the latest TURN ts_iso when scoping extraction records
# by time window. Extraction llm_call typically completes within seconds of
# its triggering turn; 60 seconds is generous and accommodates retries +
# slow LLM responses without leaking subsequent sessions' extractions.
_EXTRACTION_TS_BUFFER_SECONDS = 60


def get_session_ts_window(records: Iterator[dict], session_id: str) -> tuple[str, str] | None:
    """Return (earliest, latest) ts_iso for the session's TURN records.

    Used to scope extraction llm_call records (which have session_id=None
    because the extraction call site does not propagate the conversation
    context) to a single replay run by time range. Returns None if no
    TURN records match the requested session_id.

    The session's TURN ts_iso bounds are used directly as the lower
    bound; the caller adds an extraction-lag buffer to the upper bound
    before filtering records.
    """
    ts_values: list[str] = []
    for rec in records:
        if rec.get("phase") != "turn":
            continue
        if rec.get("session_id") != session_id:
            continue
        ts = rec.get("ts_iso")
        if isinstance(ts, str) and ts:
            ts_values.append(ts)
    if not ts_values:
        return None
    return (min(ts_values), max(ts_values))


def build_extraction_index(
    records: Iterator[dict], ts_range: tuple[str, str] | None = None
) -> dict[str, list[ExtractionRecord]]:
    """Index extraction llm_call records by utterance recovered from request.

    When `ts_range=(min_ts, max_ts)` is supplied, only records with
    `ts_iso` lexicographically within `[min_ts, max_ts]` are indexed.
    ISO 8601 timestamps with consistent timezone are lexically ordered,
    so string comparison is correct here. Caller is responsible for
    extending `max_ts` with an extraction-lag buffer if needed.
    """
    by_utterance: dict[str, list[ExtractionRecord]] = {}
    for rec in records:
        if rec.get("phase") != "llm_call":
            continue
        if rec.get("call_site") != EXTRACTION_CALL_SITE:
            continue
        if ts_range is not None:
            ts = rec.get("ts_iso")
            if not isinstance(ts, str) or not (ts_range[0] <= ts <= ts_range[1]):
                continue
        utterance = extract_utterance_from_request(rec.get("request") or {})
        if not utterance:
            continue
        utterance = _normalize_utterance(utterance)
        response = rec.get("response") or {}
        content = response.get("content") or ""
        parse_ok, entity_types, rel_types = parse_extraction_json(content)
        by_utterance.setdefault(utterance, []).append(
            ExtractionRecord(
                extracted_entity_types=entity_types,
                extracted_relationship_types=rel_types,
                parse_ok=parse_ok,
            )
        )
    return by_utterance


def score_run(
    probes: list[V9Probe],
    extraction_by_utterance: dict[str, list[ExtractionRecord]],
) -> V9Report:
    outcomes: list[ProbeOutcome] = []
    for probe in probes:
        extractions = extraction_by_utterance.get(_normalize_utterance(probe.utterance), [])
        all_edges: set[str] = set()
        all_entities: set[str] = set()
        any_parse_ok = False
        for ext in extractions:
            if ext.parse_ok:
                any_parse_ok = True
            all_edges |= ext.extracted_relationship_types
            all_entities |= ext.extracted_entity_types
        outcomes.append(
            ProbeOutcome(
                probe=probe,
                extracted_edges=frozenset(all_edges),
                extracted_entities=frozenset(all_entities),
                matched=bool(extractions),
                parse_ok=any_parse_ok if extractions else False,
            )
        )
    return V9Report(outcomes=outcomes)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render_markdown(report: V9Report) -> str:
    lines: list[str] = [
        "# V9 Probe Set -- v1.1.0 Predicate-Coverage Score Report",
        "",
        f"**Probes total:** {len(report.outcomes)}",
    ]
    matched = len(report.outcomes) - report.missing
    lines.append(f"**Matched against debug JSONL:** {matched}/{len(report.outcomes)}")
    if report.missing:
        lines.append(f"**Missing (no extraction record for utterance):** {report.missing}")
    lines.append("")
    lines.append("## Per-Predicate Recall")
    lines.append("")
    lines.append("| Predicate | Probes expected | Fired | Recall | Pass |")
    lines.append("|-----------|---------:|------:|-------:|:----:|")
    for predicate in sorted(TARGET_PREDICATES):
        stats = report.per_predicate_stats()[predicate]
        recall = stats["fired"] / stats["expected"] if stats["expected"] else 0.0
        passed = recall >= PER_PREDICATE_RECALL_THRESHOLD
        lines.append(
            f"| {predicate} | {stats['expected']} | {stats['fired']} | "
            f"{recall:.2f} | {'PASS' if passed else 'FAIL'} |"
        )
    lines.append("")
    lines.append("## Acceptance Criteria")
    lines.append("")
    for predicate in sorted(TARGET_PREDICATES):
        recall = report.per_predicate_recall()[predicate]
        passed = recall >= PER_PREDICATE_RECALL_THRESHOLD
        marker = "PASS" if passed else "FAIL"
        lines.append(
            f"- [{marker}] {predicate} recall >= {PER_PREDICATE_RECALL_THRESHOLD:.2f} "
            f"(actual {recall:.2f})"
        )
    verdict = "PASS" if report.acceptance_pass() else "FAIL"
    lines.append("")
    lines.append(f"**Verdict:** {verdict}")
    lines.append("")
    lines.append("## Per-Probe Outcomes")
    lines.append("")
    lines.append("| Tag | Utterance | Expected | Extracted edges | Fired |")
    lines.append("|-----|-----------|----------|-----------------|:-----:|")
    for o in report.outcomes:
        utt_preview = o.probe.utterance[:60] + ("..." if len(o.probe.utterance) > 60 else "")
        expected_in_target = sorted(o.expected_target_predicates)
        extracted_in_target = sorted(o.extracted_edges & TARGET_PREDICATES)
        fired_marker = "OK" if o.fired else ("MISS" if o.matched else "MISSING")
        lines.append(
            f"| {o.probe.tag} | {utt_preview!r} | {expected_in_target} | "
            f"{extracted_in_target} | {fired_marker} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V9 v1.1.0 predicate-coverage scorer")
    parser.add_argument("--input", type=Path, required=True, help="V9 input JSONL")
    parser.add_argument("--debug-jsonl", type=Path, required=True, help="MIST_DEBUG_JSONL path")
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help=(
            "Scope the score to a single replay run. Extraction llm_call "
            "records have session_id=None, so the filter applies via the "
            "per-session TURN utterance set: only extractions whose "
            "utterance appeared in TURN records of this session are "
            "counted. When omitted, all extractions in the JSONL are "
            "counted (multi-run aggregation -- caller's responsibility "
            "to ensure single-run input)."
        ),
    )
    parser.add_argument("--output", type=Path, default=None, help="Write markdown to file")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 when any predicate fails its recall threshold",
    )
    args = parser.parse_args(argv)

    probes = list(iter_probes(args.input))
    # Extraction llm_call records have session_id=None, so they cannot be
    # filtered directly by session. We instead derive the time window of
    # the requested session from its TURN records (which DO carry
    # session_id) and bound the extraction index by ts_iso. This properly
    # scopes when two replay runs land in the same JSONL file (the most
    # likely multi-run accident).
    ts_range: tuple[str, str] | None = None
    if args.session_id is not None:
        window = get_session_ts_window(iter_debug_records(args.debug_jsonl), args.session_id)
        if window is None:
            print(
                f"WARN: session_id={args.session_id!r} matched no TURN records; "
                "scoring will aggregate the entire JSONL.",
                file=sys.stderr,
            )
        else:
            from datetime import datetime, timedelta

            min_ts, max_ts = window
            try:
                upper = (
                    datetime.fromisoformat(max_ts)
                    + timedelta(seconds=_EXTRACTION_TS_BUFFER_SECONDS)
                ).isoformat()
            except (TypeError, ValueError):
                # Malformed ts_iso -- fall back to no buffer
                upper = max_ts
            ts_range = (min_ts, upper)
    extraction_by_utterance = build_extraction_index(
        iter_debug_records(args.debug_jsonl), ts_range=ts_range
    )
    report = score_run(probes, extraction_by_utterance)
    md = render_markdown(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md, encoding="utf-8")
    print(md, end="")

    if args.strict and not report.acceptance_pass():
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
