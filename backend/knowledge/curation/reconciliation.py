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
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
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


def derive_assertion_kind(rel: dict[str, Any]) -> tuple[AssertionKind, bool]:
    """Single source of truth for assertion-kind derivation: (kind, past_mapped).

    Explicit field wins -- rel-level first, then properties, case-insensitive --
    INCLUDING explicit 'assert' (the past-tense mapping below must not coerce a
    deliberately asserted past-tense EVENT). The interim past->CEASE mapping
    applies only when the payload carried no parseable explicit kind (pre-r2
    log events, parse drops). Consumed by EdgeAssertion.from_rel_dict, the
    intra-turn sort key, and the F2 scorer -- change it in one place only.
    """
    props = rel.get("properties") or {}
    raw = rel.get("assertion_kind") or props.get("assertion_kind")
    if raw is not None:
        try:
            return AssertionKind(str(raw).lower()), False
        except ValueError:
            pass  # invalid explicit value: treat as absent
    temporal_status = str(props.get("temporal_status") or rel.get("temporal_status") or "current")
    if temporal_status == "past" and not props.get("end_date"):
        return AssertionKind.CEASE, True
    return AssertionKind.ASSERT, False


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
    past_mapped: bool = False  # True when temporal_status='past' coerced ASSERT->CEASE

    @classmethod
    def from_rel_dict(cls, rel: dict[str, Any], defn: EdgeTypeDefinition) -> EdgeAssertion:
        """Build from the pipeline's rel dict; canonicalize undirected order.

        Temporal bounds come from properties.start_date/end_date (where the
        temporal resolver writes them); temporal_status checks properties
        first, then the legacy top-level key the old writer read. Confidence
        and source_type follow the same props-first pattern -- validator
        output carries them ONLY under properties, and a top-level-only read
        would flatten every edge to the 0.8 default.
        """
        props = rel.get("properties") or {}
        source, target = rel.get("source", ""), rel.get("target", "")
        if not defn.directional and target < source:
            source, target = target, source
        # Derivation + interim past-tense mapping live in derive_assertion_kind
        # (the single source of truth shared with the intra-turn sort and the
        # F2 scorer). The past->CEASE coercion fires only for payloads with no
        # explicit kind -- an explicit 'assert' on a past-tense EVENT must
        # accumulate, not be coerced to CEASE and flagged away (C3 gate).
        kind, past_mapped = derive_assertion_kind(rel)
        temporal_status = str(
            props.get("temporal_status") or rel.get("temporal_status") or "current"
        )
        return cls(
            source=source,
            predicate=rel.get("type", ""),
            target=target,
            confidence=float(props.get("confidence", rel.get("confidence", 0.8))),
            source_type=str(props.get("source_type") or rel.get("source_type") or "extracted"),
            context=str(rel.get("context", "") or props.get("context", "") or ""),
            temporal_status=temporal_status,
            valid_from_stated=props.get("start_date"),
            valid_to_stated=props.get("end_date"),
            assertion_kind=kind,
            past_mapped=past_mapped,
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
        cease_reason = "cease_past_tense" if assertion.past_mapped else "cease"
        actions = []
        if assertion.past_mapped:
            # Surface model-inferred cessation in flags + telemetry (spec 6.2
            # accuracy gating): the closes are append-only and transaction-
            # recoverable if the 'past' reading was wrong.
            actions.append(
                ReconcileAction(kind=ActionKind.FLAG_AMBIGUOUS, reason="past_tense_cease", **base)
            )
        for b in hit:
            actions.append(
                ReconcileAction(
                    kind=ActionKind.CLOSE_TRANSACTION,
                    reason=cease_reason,
                    edge_ref=b.edge_ref,
                    **base,
                )
            )
            actions.append(
                ReconcileAction(
                    kind=ActionKind.APPEND_CLOSED_COPY,
                    reason=cease_reason,
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

    # A predicate listed in both contradicts and progression_supersedes (e.g.
    # EXPERT_IN / STRUGGLES_WITH) yields the same belief row twice; collapse
    # to one close per edge version so counters and telemetry match actual
    # writes. First reason wins (single_supersession > contradiction >
    # progression by append order above).
    seen_refs: set[str] = set()
    unique: list[tuple[BeliefRow, str]] = []
    for b, reason in candidates:
        if b.edge_ref in seen_refs:
            continue
        seen_refs.add(b.edge_ref)
        unique.append((b, reason))
    candidates = unique

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


_REL_TYPE_SAFE = re.compile(r"[^A-Z_]")


def _sanitize(rel_type: str) -> str:
    return _REL_TYPE_SAFE.sub("", rel_type.upper())


def _version_key(event_id: str, valid_from: str | None, valid_to: str | None) -> str:
    """Deterministic, rebuild-stable MERGE key for one edge version.

    `source_utterance_id` alone collides when one turn appends two versions of
    the same (s,type,t) (cease over a true->false->true history; same-turn
    contradiction+progression copies). MERGE keys must be non-null, so open
    bounds use the 'open' token. Never derived from elementId (not stable
    across rebuilds).
    """
    return f"{event_id}|{valid_from or 'open'}|{valid_to or 'open'}"


_BELIEF_RETURN = (
    "RETURN elementId(r) AS edge_ref, t.id AS target, r.valid_from AS valid_from, "
    "r.valid_to AS valid_to, coalesce(r.recorded_at, r.created_at, '') AS recorded_at, "
    "coalesce(r.confidence, 0.8) AS confidence, "
    "coalesce(r.source_type, 'extracted') AS source_type, "
    "coalesce(r.context, '') AS context, "
    "coalesce(r.temporal_status, 'current') AS temporal_status, "
    "coalesce(r.evidence, []) AS evidence, "
    "coalesce(r.source_utterance_id, '') AS source_utterance_id "
    "ORDER BY recorded_at, valid_from, source_utterance_id"
)


class ReconciliationEngine:
    """Fetch -> plan -> apply -> telemetry for one turn's relationships.

    Idempotency + crash convergence (Inv-A5): appended versions MERGE on a
    deterministic `version_key`; transaction closes are guarded by
    `is_latest_belief`; reinforce/structural use append-if-absent evidence.
    Per edge, the apply order is a commit protocol -- closes and clamped
    copies first, the probe-bearing primary APPEND_VERSION LAST -- so a crash
    mid-edge replays convergently: the eid probe only hits once the edge's
    final write landed; earlier partial writes re-MERGE/no-op on replay.
    Residual: a CEASE that dies between its close and its clamped copy
    replays as a flagged cease_without_prior (non-destructive; the precise
    stop date is recoverable via R1 rebuild). The CALLER (CurationPipeline)
    serializes invocations (Inv-A9); the engine itself is not re-entrant-safe.

    Legacy edges (no bitemporal props) are read via coalesce(is_latest_belief,
    true) so the engine can supersede pre-C1 facts before backfill runs.
    """

    def __init__(
        self,
        executor: Any,
        rebuild_stamps: Any,
        edge_types_by_name: dict[str, EdgeTypeDefinition] | None = None,
        debug_logger: Any | None = None,
    ) -> None:
        from backend.knowledge.ontologies import EDGE_TYPES_BY_NAME

        self._executor = executor
        self._stamps = rebuild_stamps
        self._edges = edge_types_by_name or EDGE_TYPES_BY_NAME
        self._debug_logger = debug_logger

    async def reconcile_turn(
        self,
        relationships: list[dict[str, Any]],
        recorded_at: str,
        event_id: str,
        session_id: str,
    ) -> ReconcileTurnResult:
        """Reconcile one turn's validated relationships (deterministic order)."""
        result = ReconcileTurnResult()
        recorded_dt = parse_from_bound(recorded_at)
        if recorded_dt is None:
            raise ValueError(f"unparsable recorded_at: {recorded_at!r}")

        ordered = sorted(
            relationships,
            key=lambda r: (r.get("type", ""), r.get("source", ""), r.get("target", "")),
        )
        for rel in ordered:
            defn = self._edges.get(rel.get("type", ""))
            if defn is None:
                result.flags.append(f"unknown_predicate:{rel.get('type', '')}")
                continue
            assertion = EdgeAssertion.from_rel_dict(rel, defn)
            existing = await self._fetch_existing(assertion, defn, event_id)
            actions = plan_edge(assertion, defn, existing, recorded_dt, event_id)
            await self._apply(actions, assertion, recorded_at, event_id, result)
            self._emit(actions, event_id, session_id)
        return result

    # -- fetch -----------------------------------------------------------------

    async def _fetch_latest(
        self, source: str, predicate: str, *, target: str | None, exclude_target: str | None
    ) -> list[BeliefRow]:
        stype = _sanitize(predicate)
        clauses = ["coalesce(r.is_latest_belief, true)"]
        params: dict[str, Any] = {"source": source}
        if target is not None:
            clauses.append("t.id = $target")
            params["target"] = target
        if exclude_target is not None:
            clauses.append("t.id <> $target")
            params["target"] = exclude_target
        rows = await self._executor.execute_query(
            f"MATCH (s:__Entity__ {{id: $source}})-[r:{stype}]->(t:__Entity__) "
            f"WHERE {' AND '.join(clauses)} {_BELIEF_RETURN}",
            params,
        )
        return [
            BeliefRow(
                edge_ref=str(row["edge_ref"]),
                predicate=predicate,
                source=source,
                target=str(row["target"]),
                valid_from=row.get("valid_from"),
                valid_to=row.get("valid_to"),
                recorded_at=str(row.get("recorded_at") or ""),
                confidence=float(row.get("confidence", 0.8)),
                source_type=str(row.get("source_type", "extracted")),
                context=str(row.get("context", "")),
                temporal_status=str(row.get("temporal_status", "current")),
                evidence=list(row.get("evidence") or []),
                source_utterance_id=str(row.get("source_utterance_id", "")),
            )
            for row in rows
        ]

    async def _fetch_existing(
        self, a: EdgeAssertion, defn: EdgeTypeDefinition, event_id: str
    ) -> ExistingBeliefs:
        stype = _sanitize(a.predicate)
        applied = await self._executor.execute_query(
            f"MATCH (s:__Entity__ {{id: $source}})"
            f"-[r:{stype} {{source_utterance_id: $eid}}]->"
            f"(t:__Entity__ {{id: $target}}) RETURN count(r) AS n",
            {"source": a.source, "target": a.target, "eid": event_id},
        )
        turn_already_applied = bool(applied and applied[0].get("n", 0))
        if turn_already_applied:
            # Short-circuit: replay of an applied edge needs no belief fetches.
            return ExistingBeliefs(
                turn_already_applied=True,
                same_fact=[],
                single_conflicts=[],
                contradictions=[],
                progressions=[],
            )

        same_fact = await self._fetch_latest(
            a.source, a.predicate, target=a.target, exclude_target=None
        )
        single_conflicts: list[BeliefRow] = []
        if defn.cardinality is Cardinality.SINGLE:
            single_conflicts = await self._fetch_latest(
                a.source, a.predicate, target=None, exclude_target=a.target
            )
        contradictions: list[BeliefRow] = []
        for pred in defn.contradicts:
            contradictions += await self._fetch_latest(
                a.source, pred, target=a.target, exclude_target=None
            )
        progressions: list[BeliefRow] = []
        for pred in defn.progression_supersedes:
            if pred in defn.contradicts:
                # Already fetched as a contradiction; the planner dedups by
                # edge_ref, so a second identical fetch is pure waste.
                continue
            progressions += await self._fetch_latest(
                a.source, pred, target=a.target, exclude_target=None
            )
        return ExistingBeliefs(
            turn_already_applied=False,
            same_fact=same_fact,
            single_conflicts=single_conflicts,
            contradictions=contradictions,
            progressions=progressions,
        )

    # -- apply -----------------------------------------------------------------

    async def _apply(
        self,
        actions: list[ReconcileAction],
        assertion: EdgeAssertion,
        recorded_at: str,
        event_id: str,
        result: ReconcileTurnResult,
    ) -> None:
        now = datetime.now(UTC).isoformat()  # wall-clock audit only (not canonical)
        # Commit-protocol ordering: the primary APPEND_VERSION (which makes the
        # idempotency probe hit) is applied LAST so partial application replays
        # convergently. Closes/copies keep their planner-relative order.
        ordered = [a for a in actions if a.kind is not ActionKind.APPEND_VERSION] + [
            a for a in actions if a.kind is ActionKind.APPEND_VERSION
        ]
        for act in ordered:
            result.actions.append(act)
            if act.kind is ActionKind.NOOP_IDEMPOTENT:
                continue
            if act.kind is ActionKind.FLAG_AMBIGUOUS:
                result.flags.append(f"{act.reason}:{act.predicate}:{act.source}->{act.target}")
                continue
            if act.kind is ActionKind.STRUCTURAL_UPSERT:
                if await self._apply_structural(act, assertion, recorded_at, event_id, now):
                    result.structural += 1
                else:
                    result.flags.append(
                        f"structural_dangling:{act.predicate}:{act.source}->{act.target}"
                    )
                continue
            if act.kind in (ActionKind.APPEND_VERSION, ActionKind.APPEND_CLOSED_COPY):
                applied = await self._apply_append(
                    act, assertion, recorded_at, event_id, now, copy_of=act.copy_of
                )
                if applied:
                    result.appended += 1
                else:
                    # MATCH..MATCH..MERGE silently writes zero rows when an
                    # endpoint node is missing; counting that as applied would
                    # report success while the fact vanished.
                    result.flags.append(
                        f"append_dangling:{act.predicate}:{act.source}->{act.target}"
                    )
                continue
            if act.kind is ActionKind.CLOSE_TRANSACTION:
                await self._executor.execute_write(
                    "MATCH ()-[r]->() WHERE elementId(r) = $ref "
                    "AND coalesce(r.is_latest_belief, true) "
                    "SET r.recorded_until = $recorded_at, r.is_latest_belief = false, "
                    "r.updated_at = $now",
                    {"ref": act.edge_ref, "recorded_at": recorded_at, "now": now},
                )
                result.closed += 1
                continue
            if act.kind is ActionKind.REINFORCE:
                await self._executor.execute_write(
                    "MATCH ()-[r]->() WHERE elementId(r) = $ref "
                    "SET r.confidence = CASE WHEN coalesce(r.confidence, 0) < $conf "
                    "THEN $conf ELSE r.confidence END, "
                    "r.evidence = CASE WHEN $eid IN coalesce(r.evidence, []) "
                    "THEN r.evidence ELSE coalesce(r.evidence, []) + [$eid] END, "
                    "r.updated_at = $now",
                    {
                        "ref": act.edge_ref,
                        "conf": assertion.confidence,
                        "eid": event_id,
                        "now": now,
                    },
                )
                result.reinforced += 1

    async def _apply_append(
        self,
        act: ReconcileAction,
        assertion: EdgeAssertion,
        recorded_at: str,
        event_id: str,
        now: str,
        copy_of: BeliefRow | None,
    ) -> bool:
        stype = _sanitize(act.predicate)
        src = copy_of or assertion
        vf = act.valid_from if copy_of is None else copy_of.valid_from
        rows = await self._executor.execute_write(
            f"MATCH (s:__Entity__ {{id: $source}}) "
            f"MATCH (t:__Entity__ {{id: $target}}) "
            f"MERGE (s)-[r:{stype} {{version_key: $vk}}]->(t) "
            "ON CREATE SET r.source_utterance_id = $eid, "
            "r.valid_from = $valid_from, r.valid_to = $valid_to, "
            "r.recorded_at = $recorded_at, r.recorded_until = NULL, "
            "r.is_latest_belief = true, r.correction = $correction, "
            "r.confidence = $confidence, r.source_type = $source_type, "
            "r.context = $context, r.temporal_status = $temporal_status, "
            "r.evidence = $evidence, r.supersession_reason = $reason, "
            "r.provenance = 'extraction', r.ontology_version = $ontology_version, "
            "r.extraction_version = $extraction_version, r.model_hash = $model_hash, "
            "r.created_at = $now, r.updated_at = $now "
            "RETURN count(r) AS n",
            {
                "source": act.source,
                "target": act.target,
                "eid": event_id,
                "vk": _version_key(event_id, vf, act.valid_to),
                "valid_from": vf,
                "valid_to": act.valid_to,
                "recorded_at": recorded_at,
                "correction": act.correction,
                "confidence": src.confidence,
                "source_type": src.source_type,
                "context": src.context,
                # A version with a closed valid_to is by definition not a
                # current state; copying the prior's 'current' verbatim onto
                # clamped copies and retract markers contradicts their own
                # interval and poisons any consumer keying on the legacy field.
                "temporal_status": ("past" if act.valid_to is not None else src.temporal_status),
                "evidence": list(copy_of.evidence) if copy_of else [event_id],
                "reason": act.reason if copy_of is not None else "",
                "ontology_version": self._stamps.ontology_version,
                "extraction_version": self._stamps.extraction_version,
                "model_hash": self._stamps.model_hash,
                "now": now,
            },
        )
        # Real Neo4j always returns one row for RETURN count(r); legacy fakes
        # returning no rows are treated as applied.
        return bool(int(rows[0].get("n", 1)) if rows else 1)

    async def _apply_structural(
        self,
        act: ReconcileAction,
        assertion: EdgeAssertion,
        recorded_at: str,
        event_id: str,
        now: str,
    ) -> bool:
        stype = _sanitize(act.predicate)
        rows = await self._executor.execute_write(
            f"MATCH (s:__Entity__ {{id: $source}}) "
            f"MATCH (t:__Entity__ {{id: $target}}) "
            f"MERGE (s)-[r:{stype}]->(t) "
            "ON CREATE SET r.source_utterance_id = $eid, r.recorded_at = $recorded_at, "
            "r.recorded_until = NULL, r.is_latest_belief = true, r.correction = false, "
            "r.valid_from = NULL, r.valid_to = NULL, "
            "r.confidence = $confidence, r.source_type = $source_type, "
            "r.context = $context, r.evidence = [$eid], r.provenance = 'extraction', "
            "r.ontology_version = $ontology_version, "
            "r.extraction_version = $extraction_version, r.model_hash = $model_hash, "
            "r.created_at = $now, r.updated_at = $now "
            "ON MATCH SET r.confidence = CASE WHEN coalesce(r.confidence, 0) < $confidence "
            "THEN $confidence ELSE r.confidence END, "
            "r.evidence = CASE WHEN $eid IN coalesce(r.evidence, []) "
            "THEN r.evidence ELSE coalesce(r.evidence, []) + [$eid] END, "
            "r.updated_at = $now "
            "RETURN count(r) AS n",
            {
                "source": act.source,
                "target": act.target,
                "eid": event_id,
                "recorded_at": recorded_at,
                "confidence": assertion.confidence,
                "source_type": assertion.source_type,
                "context": assertion.context,
                "ontology_version": self._stamps.ontology_version,
                "extraction_version": self._stamps.extraction_version,
                "model_hash": self._stamps.model_hash,
                "now": now,
            },
        )
        return bool(int(rows[0].get("n", 1)) if rows else 1)

    # -- telemetry ---------------------------------------------------------------

    def _emit(self, actions: list[ReconcileAction], event_id: str, session_id: str) -> None:
        if self._debug_logger is None:
            return
        for act in actions:
            self._debug_logger.record_reconciliation(
                event_id=event_id,
                session_id=session_id,
                predicate=act.predicate,
                source=act.source,
                target=act.target,
                action=act.kind.value,
                reason=act.reason,
                edge_ref=act.edge_ref,
                valid_from=act.valid_from,
                valid_to=act.valid_to,
            )
