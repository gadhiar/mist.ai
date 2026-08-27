"""Canonical graph serialization for determinism checks (F3).

`dump_graph_cypher` (admin.py) is a snapshot tool: it stamps a wall-clock
header and emits node/edge audit timestamps, so two dumps of the same graph
differ. This module produces a CANONICAL form -- deterministic given graph
content, excluding wall-clock/audit fields, embeddings, epoch stamps
(ontology_version/extraction_version/model_hash), and wall-clock-decayed
node confidence, with stable ordering -- so two rebuilds of the same
(log, epoch) can be diffed for byte-identity (Inv-A4). Deterministic
provenance (source_utterance_id,
recorded_at, evidence, event_id) is RETAINED -- it is tied to the immutable
log, not to wall-clock.
"""

from __future__ import annotations

import json
from typing import Any

from backend.knowledge.admin import dump_graph_json

# Wall-clock audit fields excluded from the canonical form. Log-tied provenance
# (event_id, source_utterance_id, recorded_at, evidence, first_event_id,
# last_event_id, plus the legacy source_event_id where it survives) is RETAINED
# -- it is deterministic given the immutable log.
AUDIT_FIELDS = frozenset(
    {
        "created_at",
        "updated_at",
        "derived_at",
        "first_seen_at",
        "last_seen_at",
    }
)

# Epoch-derived metadata. Present-day stamps have a single authority
# (`backend/knowledge/version_stamps.py`, collapsed 2026-08-02), so a fresh write
# and a fresh rebuild already agree. The exclusion is therefore about HISTORICAL
# DATA COMPATIBILITY, not about reconciling authorities: rows written under
# earlier stamp values persist in the graph, and a rebuild re-stamps them with
# today's triple. Do NOT un-exclude these on the grounds that the authorities now
# agree -- comparing the stamps makes `live == rebuilt` fail on every legacy row,
# over a difference that says nothing about whether the FACTS were reproduced.
# The proof is deliberately "same log + same epoch => same facts", not
# "same stamps".
EPOCH_STAMP_FIELDS = frozenset({"ontology_version", "extraction_version", "model_hash"})

# Derived artifacts: reproducible in principle, not compared in practice.
# `embedding` is large and float-noisy, and excluding it has a documented cost
# -- `seed/gates.py:264-268` records that a canonical form is byte-identical
# whether embeddings are present, absent, or all-zero, so a seed-apply that
# skips the backfill yields a graph nothing can retrieve from AND certifies
# clean. MIS-130 carries a separate presence-and-dimension assertion because
# THIS set cannot cover it.
DERIVED_ARTIFACT_FIELDS = frozenset({"embedding"})

# Excluded on NODES only; the same property on an edge is compared.
#
# Edge `confidence` is reinforce-only -- `graph_writer.py:251` takes a monotonic
# max on write -- and therefore log-deterministic. Node `confidence` is
# additionally written by `ConfidenceDecayJob` (`confidence_decay.py:34,39`)
# off the wall clock, which is the entire reason for the asymmetry.
#
# That reason is CONDITIONAL, and the condition is now controllable: the decay
# job is a scheduler job, and `CurationScheduler.start()` refuses to run under
# MIST_HYDRATION_ISOLATION (B1). With the scheduler off, node confidence has
# only log-deterministic writers and this exclusion costs coverage rather than
# buying determinism -- MIS-131's exclusion decision, closed 2026-08-26. Do not
# simply delete it: the un-exclusion and the scheduler-off knob are one change,
# because the exclusion is correct whenever the job CAN run.
NODE_ONLY_EXCLUDED_FIELDS = frozenset({"confidence"})


def _canon_props(props: dict[str, Any], *, is_node: bool) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in props.items():
        if k in AUDIT_FIELDS or k in EPOCH_STAMP_FIELDS or k in DERIVED_ARTIFACT_FIELDS:
            continue
        # See NODE_ONLY_EXCLUDED_FIELDS: node `confidence` is wall-clock-decayed
        # while the decay job can run; edge `confidence` is reinforce-only
        # (log-deterministic) and is retained.
        if is_node and k in NODE_ONLY_EXCLUDED_FIELDS:
            continue
        out[k] = sorted(v) if isinstance(v, list) else v
    return out


def _node(n: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": n["id"],
        "labels": sorted(n["labels"]),
        "properties": _canon_props(n["properties"], is_node=True),
    }


def _rel(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": r["source"],
        "type": r["type"],
        "target": r["target"],
        "properties": _canon_props(r["properties"], is_node=False),
    }


def _node_key(n: dict[str, Any]) -> str:
    return str(n["id"])


def _rel_key(r: dict[str, Any]) -> tuple[str, str, str, str, str, str, str]:
    props = r.get("properties") or {}
    # Every relationship write path persists `source_utterance_id`
    # (`graph_writer._ensure_extracted_from`, reconciliation's fact-edge MERGEs).
    # `source_event_id` is the pre-C1 name: still declared in the ontology's
    # universal relationship properties, but written by no relationship path, and
    # carried by 0 live edges. It is read FIRST only so pre-C1 rows keep the same
    # tiebreak they had before the rename.
    utt = props.get("source_event_id") or props.get("source_utterance_id") or ""
    # version_key (= event_id|valid_from|valid_to, the C2 MERGE identity) makes
    # the key total over valid-time-distinct versions. DURABLE/provenance edges
    # have null version_key; the leading (source,type,target) disambiguates them
    # (unique per triple), so the trailing components are inert "" -- still total.
    return (
        str(r["source"]),
        str(r["type"]),
        str(r["target"]),
        str(utt),
        str(props.get("version_key") or ""),
        str(props.get("valid_from") or ""),
        str(props.get("valid_to") or ""),
    )


def canonical_graph_form(connection, *, include_provenance: bool = False) -> str:
    """Return a deterministic canonical string for the graph.

    Excludes wall-clock/audit fields + embeddings; sorts nodes by id, edges by
    (source, type, target, source_utterance_id, version_key, valid_from, valid_to),
    property keys, and list values.
    Two graphs with identical content produce identical strings regardless of
    write wall-clock time or write order.
    """
    payload = dump_graph_json(connection, include_provenance=include_provenance)

    canon: dict[str, Any] = {
        "nodes": [_node(n) for n in sorted(payload["nodes"], key=_node_key)],
        "relationships": [_rel(r) for r in sorted(payload["relationships"], key=_rel_key)],
    }
    if include_provenance:
        canon["provenance"] = {
            "nodes": [_node(n) for n in sorted(payload["provenance"]["nodes"], key=_node_key)],
            "relationships": [
                _rel(r) for r in sorted(payload["provenance"]["relationships"], key=_rel_key)
            ],
        }
        canon["cross_layer_edges"] = [
            _rel(r) for r in sorted(payload["cross_layer_edges"], key=_rel_key)
        ]
    return json.dumps(canon, sort_keys=True, indent=2) + "\n"
