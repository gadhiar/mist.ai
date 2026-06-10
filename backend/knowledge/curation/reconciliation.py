"""Schema-driven bitemporal reconciliation (C2, design 5).

One generic processor replaces `conflict_resolver.py` and the relationship
write/supersede half of `graph_writer.py`. It contains no predicate names:
all semantics (cardinality, temporal_class, contradicts,
progression_supersedes, directional) come from the ontology (Inv-A6).

Split into a PURE planner (`plan_edge`: assertion + definition + pre-fetched
beliefs -> action list) and a Neo4j adapter (`ReconciliationEngine`: fetch,
plan, apply, emit telemetry). The planner carries the semantic battery in
unit tests; the adapter is exercised with graph fakes.

Append-only contract (Inv-A3): the only in-place updates ever applied to an
existing version are transaction closure (recorded_until + is_latest_belief)
and reinforcement (confidence max + evidence set-append). Valid-time edits
happen by appending a clamped copy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from backend.knowledge.curation.intervals import (
    ALWAYS,
    Interval,
    overlaps,
    parse_from_bound,
    parse_to_bound,
)
from backend.knowledge.ontologies import Cardinality, EdgeTypeDefinition, TemporalClass

logger = logging.getLogger(__name__)


class AssertionKind(StrEnum):
    """What an extracted edge claims (design 6.2). Default ASSERT until C3."""

    ASSERT = "assert"
    CEASE = "cease"
    RETRACT = "retract"


class ActionKind(StrEnum):
    """Planner output vocabulary; the adapter maps each to Cypher."""

    APPEND_VERSION = "append_version"
    APPEND_CLOSED_COPY = "append_closed_copy"
    CLOSE_TRANSACTION = "close_transaction"
    REINFORCE = "reinforce"
    STRUCTURAL_UPSERT = "structural_upsert"
    FLAG_AMBIGUOUS = "flag_ambiguous"
    NOOP_IDEMPOTENT = "noop_idempotent"


@dataclass(frozen=True, slots=True)
class EdgeAssertion:
    """One validated extracted relationship, normalized for reconciliation."""

    source: str
    predicate: str
    target: str
    confidence: float
    source_type: str
    context: str
    temporal_status: str
    valid_from_stated: str | None
    valid_to_stated: str | None
    assertion_kind: AssertionKind

    @classmethod
    def from_rel_dict(cls, rel: dict[str, Any], defn: EdgeTypeDefinition) -> EdgeAssertion:
        """Build from the pipeline's rel dict; canonicalize undirected order.

        Temporal bounds come from properties.start_date/end_date (where the
        temporal resolver writes them); temporal_status checks properties
        first, then the legacy top-level key the old writer read.
        """
        props = rel.get("properties") or {}
        source, target = rel.get("source", ""), rel.get("target", "")
        if not defn.directional and target < source:
            source, target = target, source
        raw_kind = str(rel.get("assertion_kind") or props.get("assertion_kind") or "assert")
        try:
            kind = AssertionKind(raw_kind.lower())
        except ValueError:
            kind = AssertionKind.ASSERT  # deterministic fallback, never destructive
        return cls(
            source=source,
            predicate=rel.get("type", ""),
            target=target,
            confidence=float(rel.get("confidence", 0.8)),
            source_type=str(rel.get("source_type", "extracted")),
            context=str(rel.get("context", "") or props.get("context", "") or ""),
            temporal_status=str(
                props.get("temporal_status") or rel.get("temporal_status") or "current"
            ),
            valid_from_stated=props.get("start_date"),
            valid_to_stated=props.get("end_date"),
            assertion_kind=kind,
        )


@dataclass(frozen=True, slots=True)
class BeliefRow:
    """An existing latest-belief edge version fetched from the graph."""

    edge_ref: str  # Neo4j elementId(r)
    predicate: str
    source: str
    target: str
    valid_from: str | None
    valid_to: str | None
    recorded_at: str
    confidence: float
    source_type: str
    context: str
    temporal_status: str
    evidence: list[str]
    source_utterance_id: str = ""

    def interval(self) -> Interval:
        """Valid-time interval; legacy rows (no bounds) read as fully open."""
        return Interval(parse_from_bound(self.valid_from), parse_from_bound(self.valid_to))


@dataclass(frozen=True, slots=True)
class ExistingBeliefs:
    """Pre-fetched graph state relevant to one assertion."""

    turn_already_applied: bool
    same_fact: list[BeliefRow]
    single_conflicts: list[BeliefRow]
    contradictions: list[BeliefRow]
    progressions: list[BeliefRow]


@dataclass(frozen=True, slots=True)
class ReconcileAction:
    """One planned graph effect."""

    kind: ActionKind
    predicate: str
    source: str
    target: str
    reason: str = ""
    edge_ref: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    correction: bool = False
    copy_of: BeliefRow | None = None


@dataclass(slots=True)
class ReconcileTurnResult:
    """Aggregate of one turn's reconciliation."""

    actions: list[ReconcileAction] = field(default_factory=list)
    appended: int = 0
    closed: int = 0
    reinforced: int = 0
    structural: int = 0
    flags: list[str] = field(default_factory=list)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


def plan_edge(
    assertion: EdgeAssertion,
    defn: EdgeTypeDefinition,
    existing: ExistingBeliefs,
    recorded_at: datetime,
    event_id: str,
) -> list[ReconcileAction]:
    """Decide the graph effects for one assertion (design 5.2, pure logic)."""
    base = {
        "predicate": assertion.predicate,
        "source": assertion.source,
        "target": assertion.target,
    }
    if existing.turn_already_applied:
        return [ReconcileAction(kind=ActionKind.NOOP_IDEMPOTENT, reason="turn_replay", **base)]

    if defn.temporal_class is TemporalClass.DURABLE:
        return [ReconcileAction(kind=ActionKind.STRUCTURAL_UPSERT, reason="durable", **base)]

    # EVENT carries no special branch in C1+C2: its semantics flow entirely
    # from the table (no contradicts/progression on event predicates;
    # OCCURRED_ON's SINGLE supersession is intentional data-correction).
    # The 4.5 stated-occurrence default degrades to recorded_at when unstated.
    if assertion.valid_from_stated == ALWAYS:
        new_start = None  # stated "always" -- open lower bound, NOT recorded_at
    else:
        new_start = parse_from_bound(assertion.valid_from_stated) or recorded_at
    new_end = parse_to_bound(assertion.valid_to_stated)
    new_iv = Interval(new_start, new_end)

    if assertion.assertion_kind is AssertionKind.RETRACT:
        actions = [
            ReconcileAction(
                kind=ActionKind.CLOSE_TRANSACTION, reason="retract", edge_ref=b.edge_ref, **base
            )
            for b in existing.same_fact
        ]
        actions.append(
            ReconcileAction(
                kind=ActionKind.APPEND_VERSION,
                reason="retract",
                valid_from=recorded_at.isoformat(),
                valid_to=recorded_at.isoformat(),  # [t, t): empty -- never valid-current
                correction=True,
                **base,
            )
        )
        return actions

    if assertion.assertion_kind is AssertionKind.CEASE:
        stop = (
            parse_to_bound(assertion.valid_to_stated)
            or parse_to_bound(assertion.valid_from_stated)
            or recorded_at
        )
        # Only priors the stop actually SHORTENS: a cease must never extend an
        # already-closed historical interval (review fix).
        hit = [
            b
            for b in existing.same_fact
            if overlaps(b.interval(), Interval(None, stop))
            and (b.interval().end is None or b.interval().end > stop)
        ]
        if not hit:
            return [
                ReconcileAction(
                    kind=ActionKind.FLAG_AMBIGUOUS, reason="cease_without_prior", **base
                )
            ]
        # Close-before-copy: if the process dies between the two writes, replay
        # re-plans against the closed prior and flags cease_without_prior --
        # non-destructive convergence (copy-first would leave a permanently
        # un-closeable open prior because the copy satisfies the eid probe).
        actions = []
        for b in hit:
            actions.append(
                ReconcileAction(
                    kind=ActionKind.CLOSE_TRANSACTION, reason="cease", edge_ref=b.edge_ref, **base
                )
            )
            actions.append(
                ReconcileAction(
                    kind=ActionKind.APPEND_CLOSED_COPY,
                    reason="cease",
                    edge_ref=b.edge_ref,
                    valid_to=stop.isoformat(),
                    copy_of=b,
                    **base,
                )
            )
        return actions

    # ASSERT (default)
    overlapping_same = [b for b in existing.same_fact if overlaps(b.interval(), new_iv)]
    if overlapping_same:
        # Deterministic tie-break (never edge_ref/elementId -- not rebuild-stable):
        best = max(
            overlapping_same,
            key=lambda b: (b.recorded_at, b.valid_from or "", b.source_utterance_id),
        )
        # Design 5.2: re-assertion reinforces and returns -- no supersessions.
        return [
            ReconcileAction(
                kind=ActionKind.REINFORCE, reason="reassertion", edge_ref=best.edge_ref, **base
            )
        ]

    actions = [
        ReconcileAction(
            kind=ActionKind.APPEND_VERSION,
            reason="assert",
            valid_from=ALWAYS if new_start is None else _iso(new_start),
            valid_to=_iso(new_end),
            **base,
        )
    ]

    candidates: list[tuple[BeliefRow, str]] = []
    if defn.cardinality is Cardinality.SINGLE:
        candidates += [(b, "single_supersession") for b in existing.single_conflicts]
    candidates += [(b, "contradiction") for b in existing.contradictions]
    candidates += [(b, "progression") for b in existing.progressions]

    overlapping = [(b, r) for b, r in candidates if overlaps(b.interval(), new_iv)]
    if new_end is not None:
        # A closed (historical) assertion never retires anything. If it
        # overlaps an open prior the situation is ambiguous -- accumulate
        # and flag (design 11: never a destructive guess).
        open_overlaps = [b for b, _ in overlapping if b.interval().end is None]
        if open_overlaps:
            actions.append(
                ReconcileAction(
                    kind=ActionKind.FLAG_AMBIGUOUS,
                    reason="closed_assertion_overlaps_open_prior",
                    **base,
                )
            )
        return actions

    for b, reason in overlapping:
        b_start = b.interval().start
        bounds = [dt for dt in (b_start, new_start) if dt is not None]
        close_at = max(bounds) if bounds else recorded_at  # both-open: close now
        actions.append(
            ReconcileAction(
                kind=ActionKind.APPEND_CLOSED_COPY,
                reason=reason,
                edge_ref=b.edge_ref,
                valid_to=close_at.isoformat(),
                copy_of=b,
                predicate=b.predicate,
                source=b.source,
                target=b.target,
            )
        )
        actions.append(
            ReconcileAction(
                kind=ActionKind.CLOSE_TRANSACTION,
                reason=reason,
                edge_ref=b.edge_ref,
                predicate=b.predicate,
                source=b.source,
                target=b.target,
            )
        )
    return actions
