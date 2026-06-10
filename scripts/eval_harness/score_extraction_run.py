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

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
