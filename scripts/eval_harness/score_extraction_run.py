"""Score a MIST extraction run against a hand-labeled gold corpus (F2).

Runs the gold corpus through `mist_admin replay` (which emits a MIST_DEBUG_JSONL
with per-turn extraction `llm_call` records), joins each gold probe to its
produced extraction by utterance text, and computes entity/relationship P/R,
relationship-typing accuracy, RELATED_TO rate, and valid-time accuracy.

Reconciliation-action accuracy is NOT scored: the reconciliation engine (C2)
does not yet emit reconciliation telemetry. A hook reports SKIPPED until then.

Standalone, mirrors scripts/eval_harness/score_v8_probe_run.py.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Running this script by path puts scripts/eval_harness/ on sys.path[0], NOT the
# repo root -- so `import backend` would fail and typing accuracy would silently
# report 0.000. Add the repo root first (mirrors scripts/mist_admin.py:51-53).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Intentionally NOT guarded: a missing backend must fail loudly, never silently
# zero the typing metric.
from backend.knowledge.extraction.validator import (  # noqa: E402
    RELATIONSHIP_CONSTRAINTS,
)

EXTRACTION_CALL_SITE = "extraction.ontology"
EXTRACTION_UTTERANCE_PATTERN = re.compile(r'Utterance:\s*"(.+?)"\s*\n\s*Output:', re.DOTALL)


def canonical_id(name: str) -> str:
    """Canonical entity id: lowercase, hyphenated, alnum+hyphen only.

    Mirrors the core of EntityNormalizer._canonicalize. Version collapsing is
    omitted -- the extraction prompt (Rule 3) already collapses versions, so
    produced ids arrive version-free; gold ids are authored canonical.
    """
    s = name.lower().strip()
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"[^a-z0-9\-]", "", s)
    s = re.sub(r"-{2,}", "-", s)
    s = s.strip("-")
    return s or name.lower().replace(" ", "-")


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True, slots=True)
class GoldEntity:
    """A gold-labeled entity: canonical id plus ontology type."""

    id: str
    type: str


@dataclass(frozen=True, slots=True)
class GoldRel:
    """A gold-labeled relationship with optional valid-time bounds."""

    source: str
    source_type: str
    predicate: str
    target: str
    target_type: str
    valid_from: str | None = None
    valid_to: str | None = None


@dataclass(frozen=True, slots=True)
class GoldProbe:
    """One gold corpus probe: utterance plus expected entities and relationships."""

    tag: str
    utterance: str
    entities: tuple[GoldEntity, ...]
    relationships: tuple[GoldRel, ...]


@dataclass(frozen=True, slots=True)
class Produced:
    """The extraction the pipeline produced for a single utterance."""

    utterance: str
    parse_ok: bool
    entities: tuple[GoldEntity, ...]
    entity_type_by_id: dict[str, str]
    relationships: tuple[dict[str, Any], ...]


def iter_gold_probes(path: Path) -> list[GoldProbe]:
    probes: list[GoldProbe] = []
    with path.open(encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            entities = tuple(
                GoldEntity(id=canonical_id(e["id"]), type=e["type"])
                for e in obj.get("expected_entities", [])
            )
            rels = tuple(
                GoldRel(
                    source=canonical_id(r["source"]),
                    source_type=r["source_type"],
                    predicate=r["predicate"],
                    target=canonical_id(r["target"]),
                    target_type=r["target_type"],
                    valid_from=r.get("valid_from"),
                    valid_to=r.get("valid_to"),
                )
                for r in obj.get("expected_relationships", [])
            )
            probes.append(
                GoldProbe(
                    tag=obj.get("tag", f"probe-{line_no}"),
                    utterance=obj["utterance"],
                    entities=entities,
                    relationships=rels,
                )
            )
    return probes


def iter_debug_records(path: Path, session_id: str | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"warn: skipping malformed debug line: {exc}", file=sys.stderr)
                continue
            if session_id and rec.get("session_id") != session_id:
                continue
            records.append(rec)
    return records


def _recover_utterance(request: dict[str, Any]) -> str | None:
    messages = request.get("messages") or []
    if not messages:
        return None
    content = messages[-1].get("content") or ""
    m = EXTRACTION_UTTERANCE_PATTERN.search(content)
    return _norm_ws(m.group(1)) if m else None


def parse_produced(
    content: str,
) -> tuple[bool, tuple[GoldEntity, ...], dict[str, str], tuple[dict[str, Any], ...]]:
    """Parse a raw extraction-LLM JSON string into canonicalized entities + rels."""
    parsed: Any = None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except json.JSONDecodeError:
                parsed = None
    if not isinstance(parsed, dict):
        return False, (), {}, ()
    entities: list[GoldEntity] = []
    type_by_id: dict[str, str] = {}
    for e in parsed.get("entities", []):
        if not isinstance(e, dict):
            continue
        cid = canonical_id(str(e.get("id") or e.get("name") or ""))
        etype = str(e.get("type", ""))
        if cid:
            entities.append(GoldEntity(id=cid, type=etype))
            type_by_id[cid] = etype
    rels: list[dict[str, Any]] = []
    for r in parsed.get("relationships", []):
        if not isinstance(r, dict):
            continue
        rels.append(
            {
                "source": canonical_id(str(r.get("source", ""))),
                "target": canonical_id(str(r.get("target", ""))),
                "predicate": str(r.get("type", "")),
                "properties": r.get("properties") or {},
            }
        )
    return True, tuple(entities), type_by_id, tuple(rels)


def build_produced_index(records: list[dict[str, Any]]) -> dict[str, Produced]:
    """Index produced extractions by recovered (normalized) utterance."""
    index: dict[str, Produced] = {}
    for rec in records:
        if rec.get("phase") != "llm_call" or rec.get("call_site") != EXTRACTION_CALL_SITE:
            continue
        utterance = _recover_utterance(rec.get("request") or {})
        if not utterance:
            continue
        content = (rec.get("response") or {}).get("content") or ""
        ok, entities, type_by_id, rels = parse_produced(content)
        if utterance in index and not ok:
            continue
        index[utterance] = Produced(
            utterance=utterance,
            parse_ok=ok,
            entities=entities,
            entity_type_by_id=type_by_id,
            relationships=rels,
        )
    return index


@dataclass(slots=True)
class Report:
    """Accumulated extraction-accuracy tallies and the derived metrics."""

    total_probes: int = 0
    matched_probes: int = 0
    entity_tp: int = 0
    entity_fp: int = 0
    entity_fn: int = 0
    rel_tp: int = 0
    rel_fp: int = 0
    rel_fn: int = 0
    typing_total: int = 0
    typing_ok: int = 0
    related_to_count: int = 0
    produced_rel_total: int = 0
    valid_time_total: int = 0
    valid_time_ok: int = 0
    negative_probes: int = 0
    negative_violations: int = 0
    per_probe: list[dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def _pr(tp: int, fp: int, fn: int) -> tuple[float, float]:
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 1.0
        return precision, recall

    @property
    def entity_precision(self) -> float:
        """Entity precision across all probes."""
        return self._pr(self.entity_tp, self.entity_fp, self.entity_fn)[0]

    @property
    def entity_recall(self) -> float:
        """Entity recall across all probes."""
        return self._pr(self.entity_tp, self.entity_fp, self.entity_fn)[1]

    @property
    def rel_precision(self) -> float:
        """Relationship precision across all probes."""
        return self._pr(self.rel_tp, self.rel_fp, self.rel_fn)[0]

    @property
    def rel_recall(self) -> float:
        """Relationship recall across all probes."""
        return self._pr(self.rel_tp, self.rel_fp, self.rel_fn)[1]

    @property
    def typing_accuracy(self) -> float:
        """Fraction of produced relationships that are constraint-valid."""
        return self.typing_ok / self.typing_total if self.typing_total else 1.0

    @property
    def related_to_rate(self) -> float:
        """Fraction of produced relationships typed RELATED_TO."""
        return self.related_to_count / self.produced_rel_total if self.produced_rel_total else 0.0

    @property
    def valid_time_accuracy(self) -> float:
        """Fraction of valid-time gold relationships whose dates matched."""
        return self.valid_time_ok / self.valid_time_total if self.valid_time_total else 1.0


def _date_matches(expected: str | None, produced: Any) -> bool:
    if expected is None:
        return True
    if not produced:
        return False
    p = str(produced)
    return p.startswith(expected) or expected.startswith(p)


def _typing_ok(predicate: str, s_type: str | None, t_type: str | None) -> bool:
    constraint = RELATIONSHIP_CONSTRAINTS.get(predicate)
    if constraint is None:
        return False  # unknown predicate -> typing fail
    allowed_src, allowed_tgt = constraint
    if allowed_src is not None and s_type not in allowed_src:
        return False
    if allowed_tgt is not None and t_type not in allowed_tgt:
        return False
    return True


def score_run(probes: list[GoldProbe], produced_index: dict[str, Produced]) -> Report:
    report = Report(total_probes=len(probes))
    for probe in probes:
        produced = produced_index.get(_norm_ws(probe.utterance))
        if produced is not None:
            report.matched_probes += 1

        gold_entities = set(probe.entities)
        prod_entities = set(produced.entities) if produced else set()
        report.entity_tp += len(gold_entities & prod_entities)
        report.entity_fp += len(prod_entities - gold_entities)
        report.entity_fn += len(gold_entities - prod_entities)

        gold_rel_keys = {(r.source, r.predicate, r.target) for r in probe.relationships}
        prod_rel_keys: set[tuple[str, str, str]] = set()
        if produced:
            for r in produced.relationships:
                prod_rel_keys.add((r["source"], r["predicate"], r["target"]))
                report.produced_rel_total += 1
                if r["predicate"] == "RELATED_TO":
                    report.related_to_count += 1
                report.typing_total += 1
                if _typing_ok(
                    r["predicate"],
                    produced.entity_type_by_id.get(r["source"]),
                    produced.entity_type_by_id.get(r["target"]),
                ):
                    report.typing_ok += 1
        report.rel_tp += len(gold_rel_keys & prod_rel_keys)
        report.rel_fp += len(prod_rel_keys - gold_rel_keys)
        report.rel_fn += len(gold_rel_keys - prod_rel_keys)

        for gr in probe.relationships:
            if gr.valid_from is None and gr.valid_to is None:
                continue
            report.valid_time_total += 1
            if not produced:
                continue
            match = next(
                (
                    r
                    for r in produced.relationships
                    if (r["source"], r["predicate"], r["target"])
                    == (gr.source, gr.predicate, gr.target)
                ),
                None,
            )
            if (
                match
                and _date_matches(gr.valid_from, match["properties"].get("start_date"))
                and _date_matches(gr.valid_to, match["properties"].get("end_date"))
            ):
                report.valid_time_ok += 1

        if not probe.entities and not probe.relationships:
            report.negative_probes += 1
            if produced is not None and (produced.entities or produced.relationships):
                report.negative_violations += 1

        if produced is not None:
            entity_fps = sorted([e.id, e.type] for e in prod_entities - gold_entities)
            entity_fns = sorted([e.id, e.type] for e in gold_entities - prod_entities)
            rel_fps = sorted(list(t) for t in prod_rel_keys - gold_rel_keys)
            rel_fns = sorted(list(t) for t in gold_rel_keys - prod_rel_keys)
        else:
            entity_fps = []
            entity_fns = sorted([e.id, e.type] for e in gold_entities)
            rel_fps = []
            rel_fns = sorted(list(t) for t in gold_rel_keys)
        report.per_probe.append(
            {
                "tag": probe.tag,
                "matched": produced is not None,
                "gold_entities": len(gold_entities),
                "gold_relationships": len(gold_rel_keys),
                "entity_fps": entity_fps,
                "entity_fns": entity_fns,
                "rel_fps": rel_fps,
                "rel_fns": rel_fns,
            }
        )
    return report


ENTITY_PRECISION_GATE = 0.90
ENTITY_RECALL_GATE = 0.80
REL_PRECISION_GATE = 0.90
REL_RECALL_GATE = 0.80
TYPING_ACCURACY_GATE = 0.90
RELATED_TO_RATE_LIMIT = 0.10


def score_reconciliation() -> None:
    """Hook: reconciliation-action accuracy.

    The reconciliation engine (C2) does not yet emit reconciliation telemetry to
    the debug stream. Until C2 lands this reports SKIPPED.
    """
    print("reconciliation-action accuracy: SKIPPED (requires C2 telemetry)", file=sys.stderr)


def _row(name: str, value: float, gate: float | None, op: str) -> str:
    if gate is None:
        return f"| {name} | {value:.3f} | - | - |"
    ok = value >= gate if op == ">=" else value <= gate
    return f"| {name} | {value:.3f} | {op} {gate:.2f} | {'PASS' if ok else 'FAIL'} |"


def render_markdown(report: Report) -> str:
    lines = [
        "# Extraction Accuracy Report (F2)",
        "",
        f"- Probes: {report.total_probes} (matched in debug log: {report.matched_probes})",
        "",
        "| Metric | Value | Gate | Pass |",
        "|---|---|---|---|",
        _row("Entity precision", report.entity_precision, ENTITY_PRECISION_GATE, ">="),
        _row("Entity recall", report.entity_recall, ENTITY_RECALL_GATE, ">="),
        _row("Relationship precision", report.rel_precision, REL_PRECISION_GATE, ">="),
        _row("Relationship recall", report.rel_recall, REL_RECALL_GATE, ">="),
        _row("Typing accuracy", report.typing_accuracy, TYPING_ACCURACY_GATE, ">="),
        _row("RELATED_TO rate", report.related_to_rate, RELATED_TO_RATE_LIMIT, "<="),
        _row("Valid-time accuracy", report.valid_time_accuracy, None, ""),
        f"| Negative-control violations | {report.negative_violations} | == 0 | "
        f"{'PASS' if report.negative_violations == 0 else 'FAIL'} |",
        "",
        f"Matched probes: {report.matched_probes}/{report.total_probes}.",
        "Reconciliation-action accuracy: SKIPPED (requires C2 telemetry).",
    ]
    return "\n".join(lines) + "\n"


def render_json(report: Report) -> str:
    return json.dumps(
        {
            "total_probes": report.total_probes,
            "matched_probes": report.matched_probes,
            "entity_precision": report.entity_precision,
            "entity_recall": report.entity_recall,
            "rel_precision": report.rel_precision,
            "rel_recall": report.rel_recall,
            "typing_accuracy": report.typing_accuracy,
            "related_to_rate": report.related_to_rate,
            "valid_time_accuracy": report.valid_time_accuracy,
            "per_probe": report.per_probe,
        },
        indent=2,
    )


def _gates_pass(report: Report) -> bool:
    return (
        report.matched_probes == report.total_probes  # a broken join must not pass
        and report.negative_violations == 0  # no hallucinated facts on negative controls
        and report.entity_precision >= ENTITY_PRECISION_GATE
        and report.entity_recall >= ENTITY_RECALL_GATE
        and report.rel_precision >= REL_PRECISION_GATE
        and report.rel_recall >= REL_RECALL_GATE
        and report.typing_accuracy >= TYPING_ACCURACY_GATE
        and report.related_to_rate <= RELATED_TO_RATE_LIMIT
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score a MIST extraction run against a gold corpus.")
    p.add_argument("--gold", required=True, type=Path, help="Gold corpus JSONL.")
    p.add_argument(
        "--debug-jsonl", required=True, type=Path, help="MIST_DEBUG_JSONL from the replay run."
    )
    p.add_argument("--session-id", default=None, help="Filter debug records by session_id.")
    p.add_argument(
        "--output", default=None, type=Path, help="Write markdown report (default stdout)."
    )
    p.add_argument(
        "--json-output", default=None, type=Path, help="Write a machine-readable JSON report."
    )
    p.add_argument("--strict", action="store_true", help="Exit 1 if any core gate fails.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.gold.exists():
        print(f"error: gold corpus not found: {args.gold}", file=sys.stderr)
        return 2
    if not args.debug_jsonl.exists():
        print(f"error: debug jsonl not found: {args.debug_jsonl}", file=sys.stderr)
        return 2
    probes = iter_gold_probes(args.gold)
    records = iter_debug_records(args.debug_jsonl, session_id=args.session_id)
    report = score_run(probes, build_produced_index(records))
    score_reconciliation()
    md = render_markdown(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md, encoding="utf-8")
    else:
        print(md)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(render_json(report), encoding="utf-8")
    if args.strict and not _gates_pass(report):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
