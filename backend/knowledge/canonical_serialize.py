"""Canonical graph serialization for determinism checks (F3).

`dump_graph_cypher` (admin.py) is a snapshot tool: it stamps a wall-clock
header and emits node/edge audit timestamps, so two dumps of the same graph
differ. This module produces a CANONICAL form -- deterministic given graph
content, excluding wall-clock/audit fields and embeddings, with stable
ordering -- so two rebuilds of the same (log, epoch) can be diffed for
byte-identity (Inv-A4). Deterministic provenance (source_utterance_id,
recorded_at, evidence, event_id) is RETAINED -- it is tied to the immutable
log, not to wall-clock.
"""

from __future__ import annotations

import json
from typing import Any

from backend.knowledge.admin import dump_graph_json

# Wall-clock audit fields excluded from the canonical form. Log-tied provenance
# (event_id, source_event_id, recorded_at, evidence, first_event_id,
# last_event_id) is RETAINED -- it is deterministic given the immutable log.
AUDIT_FIELDS = frozenset(
    {
        "created_at",
        "updated_at",
        "derived_at",
        "first_seen_at",
        "last_seen_at",
    }
)


def _canon_props(props: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in props.items():
        if k in AUDIT_FIELDS or k == "embedding":
            continue
        out[k] = sorted(v) if isinstance(v, list) else v
    return out


def _node(n: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": n["id"],
        "labels": sorted(n["labels"]),
        "properties": _canon_props(n["properties"]),
    }


def _rel(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": r["source"],
        "type": r["type"],
        "target": r["target"],
        "properties": _canon_props(r["properties"]),
    }


def _node_key(n: dict[str, Any]) -> str:
    return str(n["id"])


def _rel_key(r: dict[str, Any]) -> tuple[str, str, str, str]:
    props = r.get("properties") or {}
    # The live graph persists `source_event_id` (graph_writer.py:314); C1 renames
    # provenance to `source_utterance_id`. Read either so the multi-edge tiebreak
    # is stable on both today's and the post-C1 schema.
    utt = props.get("source_event_id") or props.get("source_utterance_id") or ""
    return (str(r["source"]), str(r["type"]), str(r["target"]), str(utt))


def canonical_graph_form(connection, *, include_provenance: bool = False) -> str:
    """Return a deterministic canonical string for the graph.

    Excludes wall-clock/audit fields + embeddings; sorts nodes by id, edges by
    (source, type, target, source_utterance_id), property keys, and list values.
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
