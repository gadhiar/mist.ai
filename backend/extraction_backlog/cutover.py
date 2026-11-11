"""Epoch cutover: re-extract the whole log under a new model, check it, promote it.

Decision 7 (MIS-171): changing the extraction model means a NEW epoch and
re-extraction of the WHOLE log through the backlog, with the old graph kept
until the new one passes a check. So does a change of a serving setting the
extraction service keys, or of a backend dispatch setting
(mist-extraction-epoch-identity): such a candidate keeps the active stamp
triple (`begin_cutover`'s `settings_change`) and is told apart by the identity
it binds to at its first fill contact. The lifecycle, one `epoch_cutover` row:

    begin     -> filling   the candidate stamps are recorded; nothing is live
    (fill)    -> ready     the dispatcher has a candidate cache row for every
                           logged turn (it keeps filling new turns after this)
    rebuild   -> checked   a staging graph built twice from the log under the
                           candidate passed the rebuild gates
    promote --graph-swapped
              checked -> promoted
                           the candidate is appended to `epoch_ledger` with a
                           first-hand activation, the turns the swapped-in graph
                           holds marked applied, in one SQLite transaction
    promote --seed-only-graph
              ready | checked -> promoted
                           the conversation log is empty, no apply marker
                           exists, and a read-only probe finds no stamped and
                           no unseeded element in the live graph: the
                           candidate is appended with a first-hand activation
                           that marks nothing, in one SQLite transaction; the
                           dispatcher then extracts and applies the turns
                           logged after it. Only when no extraction has ever
                           run against this live graph (CUTOVER.md 6A)
    abandon   -> abandoned deletes nothing

Filling is the dispatcher's job (`ExtractionDispatcher._fill_step`); this
module holds the operator steps the admin CLI runs, the rebuild check, and the
seed-only checks (`check_seed_only`, which `cutover probe` runs read-only).

What is live while a cutover is open: the OLD graph, frozen. The dispatcher
applies nothing while filling (it only caches under the candidate's stamps), so
the active epoch's apply-pending turns stay pending; the service now serves the
candidate model and cannot extract for the active epoch.

The code never writes the live Neo4j graph. `promote` takes exactly one of two
statements about it. `--graph-swapped`: the operator did the host swap in
`CUTOVER.md`. `--seed-only-graph`: the operator's statement that no extraction
has ever run against this live graph (CUTOVER.md 6A states the precondition and
why a reseed after any log reset is part of it). `promote_seed_only_cutover`
checks what it can before promoting: the conversation log holds no turn,
`extraction_applied` is empty, and one read-only Cypher statement
(`SEED_ONLY_PROBE_CYPHER`) finds no element carrying an extraction stamp and no
element lacking `seed_version`. The probe cannot see every extraction write;
the comment above `SEED_ONLY_PROBE_CYPHER` lists what it misses.

Every transition logs one structured INFO line (`log_transition`).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from types import ModuleType
from typing import TYPE_CHECKING, Any

from backend.errors import MistError
from backend.event_store.store import EpochCutoverStateError

from .store import (
    BacklogStore,
    CarriesPackHashes,
    Cutover,
    PackHashes,
    compose_epoch_model_hash,
    describe_pack_hashes,
    pack_hash_mismatches,
)

if TYPE_CHECKING:
    from backend.knowledge.extraction.normalizer import EntityNormalizer
    from backend.knowledge.extraction.validator import ExtractionValidator
    from backend.knowledge.ontologies.pack import Pack

logger = logging.getLogger(__name__)

# Exit codes of `cutover rebuild`, the same families as
# `scripts/mist_admin.py::cmd_graph_rebuild_from_log`: 0 every gate green,
# 1 rebuild-twice disagreed (non-determinism), 2 refused before measuring,
# 4 a non-vacuity floor or the self-model gate failed. 3 (live != rebuilt) is
# not used: a new model is expected to produce a different graph, so that
# comparison is reported as information and gates nothing.
EXIT_OK = 0
EXIT_NONDETERMINISTIC = 1
EXIT_REFUSED = 2
EXIT_VACUOUS = 4

_FAILURE_TEXT_LIMIT = 4000


class CutoverRefusedError(MistError):
    """An operator cutover step was refused; nothing was changed."""


def log_transition(cutover: Cutover, from_state: str, to_state: str, **fields: Any) -> None:
    """The one structured INFO line every cutover transition logs."""
    extra = " ".join(f"{name}={value}" for name, value in fields.items())
    logger.info(
        "epoch_cutover transition cutover_id=%d from=%s to=%s ontology_version=%s "
        "extraction_version=%s model_hash=%s%s",
        cutover.cutover_id,
        from_state,
        to_state,
        cutover.ontology_version,
        cutover.extraction_version,
        cutover.model_hash,
        f" {extra}" if extra else "",
    )


# ---------------------------------------------------------------------------
# begin / abandon / promote
# ---------------------------------------------------------------------------


def begin_cutover(
    store: BacklogStore,
    *,
    bare_model_hash: str,
    extraction_version: str,
    ontology_version: str,
    embedding_model_name: str,
    now_iso: str,
    settings_change: bool = False,
    refill: bool = False,
    pack_prompt_hash: str | None = None,
    pack_rails_hash: str | None = None,
) -> Cutover:
    """Open a cutover candidate in state 'filling'.

    The candidate's `model_hash` is composed from the bare service hash and the
    backend's embedding model through `compose_model_hash`, never inline.

    A settings change (`settings_change`) keeps the active epoch's stamp
    triple: what changed is a serving setting the service keys or a backend
    dispatch setting. Nothing records the flag; a candidate whose triple equals
    its source's is a settings change, and the dispatcher binds it at its first
    fill contact (`binding.decide_binding`): rule 1 refuses a service that has
    not changed, rule 4 or 5a gives it a keyed namespace, and rule 3 gives a
    return to a configuration that claims the pair its 3-part rows.

    `pack_prompt_hash` and `pack_rails_hash` are the candidate's ontology pack
    hashes. They are compared here with the active epoch's, beside the triple
    (`pack_hash_mismatches`): a candidate that differs from the epoch in a hash
    the epoch holds does not equal it, so it is a cutover of its own and never
    a settings change. An epoch that holds no hash compares the triple only.
    The two values are compared and not stored: `BacklogStore.begin_cutover`
    takes no pack hash, so the candidate row holds none.

    Two candidates need the active epoch bound, because what decides their
    rows is a comparison with its identity. A settings change while it is
    unbound would fill keyed by rule 5a even against an unchanged service, so
    it needs `refill`, the operator's choice to re-extract the log. An
    ontology-only candidate (the active extraction version and model hash)
    while it is unbound would take the pair's 3-part rows on the recorded
    fields alone (rule 5c); it waits for the binding, or for an acceptance.

    Raises:
        CutoverRefusedError: The ledger is empty; a cutover is already open;
            the candidate's stamps equal the active epoch's without
            `settings_change`, or differ with it; `refill` without
            `settings_change`; or one of the two candidates above while the
            active epoch is unbound.
    """
    epoch = store.active_epoch()
    if epoch is None:
        raise CutoverRefusedError("the epoch ledger is empty; there is no epoch to cut over from")
    if refill and not settings_change:
        raise CutoverRefusedError(
            "--refill applies only with --settings-change: any other candidate's rows are "
            "decided by its stamps"
        )
    existing = store.open_cutover()
    if existing is not None:
        raise CutoverRefusedError(
            f"cutover {existing.cutover_id} is already open ({existing.state}); "
            "promote or abandon it first"
        )
    model_hash = compose_epoch_model_hash(bare_model_hash, embedding_model_name)
    candidate_pack = PackHashes(pack_prompt_hash=pack_prompt_hash, pack_rails_hash=pack_rails_hash)
    differing_hashes = pack_hash_mismatches(epoch, candidate_pack)
    # "The same stamps": the triple, and each pack hash the active epoch holds.
    same_stamps = (
        (ontology_version, extraction_version, model_hash)
        == (epoch.ontology_version, epoch.extraction_version, epoch.model_hash)
    ) and not differing_hashes
    if same_stamps and not settings_change:
        raise CutoverRefusedError(
            f"the candidate stamps equal the active epoch {epoch.epoch_id}'s; nothing to cut "
            "over. A change of serving or dispatch settings alone is cut over with "
            "--settings-change"
        )
    if settings_change and not same_stamps:
        pack_detail = ""
        if differing_hashes:
            pack_detail = (
                f"; the epoch's ontology pack is "
                f"({describe_pack_hashes(epoch, differing_hashes)}) and the candidate's is "
                f"({describe_pack_hashes(candidate_pack, differing_hashes)})"
            )
        raise CutoverRefusedError(
            f"--settings-change keeps the active epoch {epoch.epoch_id}'s stamps "
            f"(ontology_version={epoch.ontology_version}, "
            f"extraction_version={epoch.extraction_version}, model_hash={epoch.model_hash})"
            f"{pack_detail}; "
            "this candidate's differ, so it is a cutover of its own: omit the flag"
        )
    if settings_change and not epoch.bound and not refill:
        raise CutoverRefusedError(
            f"epoch {epoch.epoch_id} has no identity yet, so this change cannot be checked "
            "against it; wait until admin status shows it bound, or pass --refill to "
            "re-extract the log anyway"
        )
    ontology_only = (
        not same_stamps
        and extraction_version == epoch.extraction_version
        and model_hash == epoch.model_hash
    )
    if ontology_only and not epoch.bound:
        raise CutoverRefusedError(
            f"epoch {epoch.epoch_id} has no identity yet, so this candidate cannot be shown "
            "to share its rows; wait until admin status shows it bound, or run admin epoch "
            "accept"
        )
    cutover = store.begin_cutover(
        ontology_version=ontology_version,
        extraction_version=extraction_version,
        model_hash=model_hash,
        bare_model_hash=bare_model_hash,
        source_epoch_id=epoch.epoch_id,
        requested_at=now_iso,
    )
    if cutover is None:
        existing = store.open_cutover()
        raise CutoverRefusedError(
            f"cutover {existing.cutover_id if existing else '?'} is already open "
            f"({existing.state if existing else '?'}); promote or abandon it first"
        )
    log_transition(
        cutover,
        "none",
        "filling",
        source_epoch_id=epoch.epoch_id,
        settings_change=settings_change,
        refill=refill,
    )
    return cutover


def abandon_cutover(store: BacklogStore, *, now_iso: str) -> Cutover:
    """Close the open cutover as 'abandoned'. Deletes nothing.

    The candidate's cache rows stay: they are keyed by the candidate's stamps
    and identity namespace, so no other epoch reads them, and a later cutover
    to the same stamps that binds to the same namespace reuses them. A keyed
    namespace is one identity's (`keyed_namespace`), so only a candidate bound
    to that identity reads it; the 3-part rows are the claim's.

    Raises:
        CutoverRefusedError: No cutover is open.
    """
    cutover = store.open_cutover()
    if cutover is None:
        raise CutoverRefusedError("no cutover is open")
    if not store.transition_cutover(
        cutover, from_states=(cutover.state,), to_state="abandoned", updated_at=now_iso
    ):
        raise CutoverRefusedError(f"cutover {cutover.cutover_id} changed state; re-run status")
    log_transition(cutover, cutover.state, "abandoned")
    return cutover


PROMOTION_GRAPH_SWAPPED = "graph_swapped"
PROMOTION_SEED_ONLY_GRAPH = "seed_only_graph"


@dataclass(frozen=True, slots=True)
class Promotion:
    """What `promote_cutover` or `promote_seed_only_cutover` did.

    Attributes:
        mode: `PROMOTION_GRAPH_SWAPPED` or `PROMOTION_SEED_ONLY_GRAPH`.
        graph_probe: The live-graph probe the seed-only path passed; None on
            the graph-swapped path, which runs no probe.
    """

    cutover: Cutover
    epoch_id: int
    marked_applied: int
    turns_at_activation: int
    mode: str = PROMOTION_GRAPH_SWAPPED
    graph_probe: GraphProbeReport | None = None


def promote_cutover(store: BacklogStore, *, graph_swapped: bool, now_iso: str) -> Promotion:
    """Make the checked candidate the active epoch.

    Requires state 'checked' AND `graph_swapped`: the operator's statement that
    the live graph has been replaced by the checked staging graph (CUTOVER.md).
    The ledger append, the first-hand activation, the applied markers through
    `rebuilt_through_event_id` and the state change are one SQLite transaction
    (`EventStore.promote_epoch_cutover`).

    Raises:
        CutoverRefusedError: No cutover is open, it is not 'checked',
            `graph_swapped` is False, it is bound keyed and its check report
            does not record that namespace, or the store refused the promotion.
    """
    cutover = store.open_cutover()
    if cutover is None:
        raise CutoverRefusedError("no cutover is open")
    if cutover.state != "checked":
        raise CutoverRefusedError(
            f"cutover {cutover.cutover_id} is {cutover.state!r}; run `cutover rebuild` until "
            "it records 'checked' (or, when the live graph holds seed data only, promote "
            "with --seed-only-graph instead)"
        )
    if not graph_swapped:
        raise CutoverRefusedError(
            "--graph-swapped is required: pass it only after the live graph has been "
            "replaced by the checked staging graph; or pass --seed-only-graph instead when "
            "the live graph holds seed data only (backend/extraction_backlog/CUTOVER.md)"
        )
    # A keyed candidate's check must have read its keyed rows. A check by code
    # from before bindings records no namespace and reads the 3-part rows; the
    # candidate may have been bound keyed and refilled to 'ready' meanwhile,
    # and that check's final transition compares only the state. A 3-part or
    # unbound candidate reads the 3-part rows either way.
    if cutover.identity_namespace is not None:
        recorded = (cutover.check_report or {}).get("identity_namespace")
        if recorded != cutover.identity_namespace:
            raise CutoverRefusedError(
                f"cutover {cutover.cutover_id} is bound to the keyed namespace "
                f"{cutover.identity_namespace}, but its check report records "
                f"{'no namespace' if recorded is None else recorded}: the check did not "
                "read the candidate's rows; run `cutover rebuild` again"
            )
    try:
        result = store.promote_cutover(cutover, activated_at=now_iso)
    except EpochCutoverStateError as exc:
        raise CutoverRefusedError(str(exc)) from exc
    log_transition(
        cutover,
        "checked",
        "promoted",
        epoch_id=result["epoch_id"],
        marked_applied=result["marked_applied"],
        turns_at_activation=result["turns_at_activation"],
        rebuilt_through_event_id=cutover.rebuilt_through_event_id,
    )
    return Promotion(
        cutover=cutover,
        epoch_id=int(result["epoch_id"]),
        marked_applied=int(result["marked_applied"]),
        turns_at_activation=int(result["turns_at_activation"]),
    )


# ---------------------------------------------------------------------------
# promote --seed-only-graph
# ---------------------------------------------------------------------------

# What the probe proves: the live graph has at least one node, no node or
# relationship WITHOUT `seed_version`, and no node or relationship WITH
# `ontology_version`, `extraction_version` or `model_hash`. Nothing more.
# Code anchors below are symbols, each with the grep that finds it (run from
# the repository root); line numbers drift.
#
# The stamps it relies on: the seed applier (`apply_seed_documents`) sets
# `seed_version` on every node it writes (the node `properties`) and every edge
# (`_MERGE_EDGE`):
#     grep -nE '"seed_version": seed_version,|SET r\.seed_version' \
#         backend/knowledge/seed/applier.py
# The extraction writers that CREATE elements stamp some of them:
# `CurationGraphWriter._upsert_entity` (nodes, ON CREATE, `ontology_version`),
# `CurationGraphWriter._extracted_from_clause` (EXTRACTED_FROM edges, all three),
# `ReconciliationEngine._apply_append` and `_apply_structural` (relationships,
# ON CREATE only), `SkillDerivationJob._create_skill` / `_ensure_capability`
# (nodes, ON CREATE, `ontology_version`):
#     grep -rn '[a-z]\.ontology_version = ' backend/knowledge/curation
# An element such a writer creates lacks `seed_version` when it is written (no
# writer outside the seed applier and the staging seeder sets it:
# `grep -rln seed_version backend/knowledge/curation backend/knowledge/extraction
# backend/chat` lists only `curation/reconciliation.py`, which reads it and
# never sets it), so the probe sees it. A later reseed refuses to run over most
# such elements and can adopt the rest (see "adopted elements" below).
#
# Unstamped CREATES are seen: an edge or node an extraction writer creates
# without a stamp (e.g. the KNOWS and HAS_CAPABILITY edges
# `SkillDerivationJob` MERGEs: `grep -nE 'MERGE \((u|m)\)-'
# backend/knowledge/curation/skill_derivation.py`) still lacks `seed_version`,
# and the probe counts it.
#
# What the probe CANNOT see: an unstamped SET on an existing SEEDED element.
#   - nodes: `InternalKnowledgeDeriver._apply_operation` (Stage 9, and the
#     curation scheduler's `SelfReflectionJob` through `derive`): the UPDATE
#     and DEPRECATE branches, and the CREATE-op MERGE's ON MATCH (updated_at,
#     confidence, and a type label) when it matches a seeded `:__SelfModel__`
#     node:
#         grep -nE 'op_type == "(UPDATE|DEPRECATE)"|ON MATCH SET' \
#             backend/knowledge/extraction/internal_derivation.py
#     `SkillDerivationJob._update_skill` and `_ensure_capability`'s update
#     branch: `grep -n 'SET e.proficiency'
#     backend/knowledge/curation/skill_derivation.py`;
#     `CurationGraphWriter._upsert_entity`'s ON MATCH when the statement
#     carries no EXTRACTED_FROM clause (`with_conversation_provenance` false;
#     with it, the EXTRACTED_FROM edge and its ConversationContext node lack
#     `seed_version` and are counted): `grep -n 'ON MATCH SET e.confidence'
#     backend/knowledge/curation/graph_writer.py`;
#   - edges: `ReconciliationEngine._apply`'s CLOSE_TRANSACTION branch
#     (recorded_until, is_latest_belief = false, updated_at: a seed edge closed
#     by supersession) and REINFORCE branch (confidence, evidence,
#     updated_at), both by elementId, and `_apply_structural`'s ON MATCH
#     (confidence, evidence, updated_at) when its MERGE matches a seeded edge:
#         grep -nE 'is ActionKind\.(CLOSE_TRANSACTION|REINFORCE):|ON MATCH SET' \
#             backend/knowledge/curation/reconciliation.py
#     Seed edges join `:__Entity__` and `:__SelfModel__` nodes
#     (`_MERGE_EDGE`'s label union: `grep -n 'MATCH (s:'
#     backend/knowledge/seed/applier.py`).
#   - adopted elements: the seed applier's `_MERGE_NODE` MERGEs on the node id
#     and `_MERGE_EDGE` on (subject, type, object), neither on `seed_version`,
#     so a reseed would give `seed_version` to a non-seed element that survived
#     its wipe and matches a seed node or fact:
#         grep -nE '"MERGE \((n|s)' backend/knowledge/seed/applier.py
#     MIS-177 D1: `reseed` (and `apply_seed_documents`) now refuses, before
#     its wipe, a graph holding anything `admin.RESET_GUARD_CYPHER` counts
#     (an `:__Entity__` node lacking `provenance = 'seed'` or `seed_version`,
#     or carrying an extraction stamp; a relationship touching an
#     `:__Entity__` node without `seed_version` or with a stamp), any node or
#     relationship carrying `extraction_version` or `model_hash`, or any with
#     `provenance = 'extraction'`:
#         grep -n '_assert_seed_target_holds_only_seed(' backend/knowledge/seed/applier.py
#     That covers the clamped seed copy `ReconciliationEngine._apply_append`
#     writes: its target is always an `:__Entity__` node and it has no
#     `seed_version` (`grep -nE 'MATCH \(t:__Entity__|r.seed_origin_version = '
#     backend/knowledge/curation/reconciliation.py`),
#     and the KNOWS edge and `user` node `SkillDerivationJob` MERGEs
#     (`grep -nE 'MERGE \(u:|MERGE \(u\)-' backend/knowledge/curation/skill_derivation.py`).
#     It does not cover an unstamped `:__SelfModel__`-only element: the edge
#     Stage 9 MERGEs from MistIdentity carries no property (`grep -n
#     'MERGE (m)-\[:{rel_type}\]->(e)' backend/knowledge/extraction/internal_derivation.py`),
#     nor does `SkillDerivationJob`'s HAS_CAPABILITY edge (`grep -n
#     'MERGE (m)-\[:HAS_CAPABILITY\]' backend/knowledge/curation/skill_derivation.py`).
#     A reseed can still adopt one of those, and it then passes this probe
#     with any earlier write on it. That needs a seed source defining a fact
#     of the same type between the same two nodes; whether the current one
#     does is not established here (the source, `mist-memory/seed/`, is not in
#     the repository). The nodes those two writers create carry
#     `ontology_version`, which this probe refuses whether or not a reseed
#     adopted them; the seed guard omits it (see `SEED_GUARD_STAMP_PROPERTIES`
#     in seed/applier.py for why).
#   - curation-scheduler jobs, which run inside the live backend and need no
#     logged turn: `ConfidenceDecayJob` (confidence, status), `OrphanDetector`
#     (status) and `EmbeddingMaintenance` (embedding; registered disabled):
#         grep -n 'SET e\.' backend/knowledge/curation/confidence_decay.py \
#             backend/knowledge/curation/orphan_detector.py \
#             backend/knowledge/curation/embedding_maintenance.py
# The apply-marker check (`extraction_applied` empty) does not close the gap:
# the in-process extraction path (`ConversationHandler`, the `elif event_id:`
# branch) runs whenever no dispatcher is attached and writes no marker; and a
# turn recorded as legacy at the backlog's first activation
# (`BacklogStore.ensure_activation`) gets no marker whether or not an earlier
# path applied it.
#
# The log-empty check narrows it for the extraction writers, given the
# CUTOVER.md 6A procedure: both extraction paths act only on a turn that is in
# the log (the dispatcher reads the log; the in-process branch needs the
# `event_id` `append_turn` returned), and `SelfReflectionJob` reads its turns
# from the log. The operator reseeds the live graph after any log reset and
# with the backend stopped; the reseed deletes every current-version seed edge
# and every seed node left without a relationship, then re-creates them, so a
# pre-reseed SET survives only on a seed node that still has a non-seed
# relationship, which this probe counts, or on an unstamped `:__SelfModel__`-only
# element the reseed adopted (above), which it does not. An empty log at promotion, checked
# again under the write lock, then shows that no turn existed after the
# reseed to be extracted. The curation-scheduler writers need no turn: that is
# why the backend stays stopped from the reseed through promotion.
#
# ONE statement, read-only: MATCH, WITH, OPTIONAL MATCH, RETURN and
# aggregations only (a unit test refuses any write clause). The relationship
# CASEs test `r IS NOT NULL` first because OPTIONAL MATCH yields one NULL `r`
# on a graph with no relationships, and `NULL.seed_version IS NULL` is true.
SEED_ONLY_PROBE_CYPHER = (
    "MATCH (n) "
    "WITH count(n) AS node_count, "
    "count(CASE WHEN n.seed_version IS NULL THEN 1 END) AS nodes_without_seed_version, "
    "count(CASE WHEN n.ontology_version IS NOT NULL OR n.extraction_version IS NOT NULL "
    "OR n.model_hash IS NOT NULL THEN 1 END) AS nodes_with_extraction_stamp "
    "OPTIONAL MATCH ()-[r]->() "
    "RETURN node_count, nodes_without_seed_version, nodes_with_extraction_stamp, "
    "count(r) AS relationship_count, "
    "count(CASE WHEN r IS NOT NULL AND r.seed_version IS NULL THEN 1 END) "
    "AS relationships_without_seed_version, "
    "count(CASE WHEN r IS NOT NULL AND (r.ontology_version IS NOT NULL "
    "OR r.extraction_version IS NOT NULL OR r.model_hash IS NOT NULL) THEN 1 END) "
    "AS relationships_with_extraction_stamp"
)


class GraphProbeError(MistError):
    """The live-graph probe could not connect, failed, or returned an unexpected shape."""


@dataclass(frozen=True, slots=True)
class GraphProbeReport:
    """What `SEED_ONLY_PROBE_CYPHER` counted in the live graph."""

    node_count: int
    nodes_without_seed_version: int
    nodes_with_extraction_stamp: int
    relationship_count: int
    relationships_without_seed_version: int
    relationships_with_extraction_stamp: int

    def as_dict(self) -> dict[str, int]:
        """The counts by name, for the check report and the transition log line."""
        return {
            "node_count": self.node_count,
            "nodes_without_seed_version": self.nodes_without_seed_version,
            "nodes_with_extraction_stamp": self.nodes_with_extraction_stamp,
            "relationship_count": self.relationship_count,
            "relationships_without_seed_version": self.relationships_without_seed_version,
            "relationships_with_extraction_stamp": self.relationships_with_extraction_stamp,
        }

    def violations(self) -> list[str]:
        """The probe conditions these counts fail; empty when they pass them.

        Passing does not prove no extraction write reached the graph: see the
        comment above `SEED_ONLY_PROBE_CYPHER` for the writes it cannot see.
        """
        found: list[str] = []
        if self.node_count < 1:
            found.append("the live graph has no nodes (the seed has not been applied)")
        if self.nodes_without_seed_version:
            found.append(f"{self.nodes_without_seed_version} node(s) without seed_version")
        if self.nodes_with_extraction_stamp:
            found.append(
                f"{self.nodes_with_extraction_stamp} node(s) carrying ontology_version, "
                "extraction_version or model_hash"
            )
        if self.relationships_without_seed_version:
            found.append(
                f"{self.relationships_without_seed_version} relationship(s) without seed_version"
            )
        if self.relationships_with_extraction_stamp:
            found.append(
                f"{self.relationships_with_extraction_stamp} relationship(s) carrying "
                "ontology_version, extraction_version or model_hash"
            )
        return found


# The probe seam: returns the live graph's counts, or raises a `MistError`
# (`GraphProbeError` for a connection or query failure). Unit tests inject a
# fake; `probe_live_graph_from_env` is the real one.
GraphProbe = Callable[[], GraphProbeReport]


def probe_graph(connection: Any) -> GraphProbeReport:
    """Run `SEED_ONLY_PROBE_CYPHER` once over `connection` and parse its one row.

    Uses `execute_query`, never `execute_write`. `Neo4jConnection` offers no
    read-access-mode session (its only two paths are `execute_query`, an
    auto-commit session in the driver's default access mode, and
    `execute_write`), so read-only is guaranteed by the statement itself.

    Raises:
        GraphProbeError: The statement did not return exactly one row with
            every count.
    """
    rows = connection.execute_query(SEED_ONLY_PROBE_CYPHER, {})
    if len(rows) != 1:
        raise GraphProbeError(f"the seed-only probe returned {len(rows)} rows, expected 1")
    row = rows[0]
    names = GraphProbeReport.__dataclass_fields__
    missing = [name for name in names if row.get(name) is None]
    if missing:
        raise GraphProbeError(f"the seed-only probe returned no value for {missing}")
    return GraphProbeReport(**{name: int(row[name]) for name in names})


def probe_live_graph_from_env() -> GraphProbeReport:
    """Open the LIVE graph named by the environment, probe it read-only, disconnect.

    Needs Neo4j; not exercised by the unit tier (tests inject a fake probe).

    Raises:
        GraphProbeError: Connecting or querying failed, in any form: a
            `MistError` from `Neo4jConnection`, an eval-isolation refusal, or a
            neo4j driver error the connection does not wrap.
    """
    from neo4j.exceptions import DriverError, Neo4jError

    from backend.knowledge.config import get_config
    from backend.knowledge.eval_isolation import EvalIsolationError
    from backend.knowledge.storage.neo4j_connection import Neo4jConnection

    config = get_config()
    connection = Neo4jConnection(config.neo4j)
    try:
        connection.connect()
        return probe_graph(connection)
    except GraphProbeError:
        raise
    except (MistError, EvalIsolationError, DriverError, Neo4jError) as exc:
        raise GraphProbeError(
            f"could not probe the live graph at {config.neo4j.uri}: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        connection.disconnect()


def log_not_empty_reason(logged_turns: int) -> str | None:
    """Why a log holding `logged_turns` turns refuses the seed-only path; None when empty.

    `logged_turns` is every row of `conversation_turn_events`, whatever the
    session's origin (`EventStore.get_turn_count`). The in-transaction
    re-check in `EventStore.promote_epoch_cutover_seed_only` words its refusal
    the same way.
    """
    if logged_turns == 0:
        return None
    return (
        f"the conversation log holds {logged_turns} logged turn(s) "
        "(conversation_turn_events, any session origin): seed-only promotion requires an "
        "empty log, because a logged turn may have been extracted into the live graph by a "
        "write the probe cannot see; use the graph-swapped path"
    )


@dataclass(frozen=True, slots=True)
class SeedOnlyCheck:
    """What `check_seed_only` observed: the log, and the live graph if the probe ran.

    Attributes:
        logged_turns: Rows in `conversation_turn_events`, any session origin.
        graph_probe: The probe's counts; None when the probe could not run.
        probe_error: Why the probe could not run (`<ErrorType>: <message>`);
            None when it ran.
    """

    logged_turns: int
    graph_probe: GraphProbeReport | None
    probe_error: str | None

    @property
    def log_violation(self) -> str | None:
        """The log-empty refusal, or None when the log is empty."""
        return log_not_empty_reason(self.logged_turns)

    @property
    def graph_violations(self) -> list[str]:
        """The probe conditions the graph fails; empty when it passed or did not run."""
        return [] if self.graph_probe is None else self.graph_probe.violations()


def check_seed_only(store: BacklogStore, *, graph_probe: GraphProbe) -> SeedOnlyCheck:
    """Run the log-empty check and the live-graph probe, read-only, and report both.

    `cutover probe` runs this; it neither needs nor looks at an open cutover,
    and writes nothing: one `SELECT COUNT(*)` on the event store and one call
    of `graph_probe` (read-only by construction, `probe_graph`). The probe
    runs even when the log check fails, so the operator sees every violation
    at once. A probe that raises `MistError` (`GraphProbeError` for any
    connection or query failure of the real probe) is reported in
    `probe_error`, not raised.
    """
    logged_turns = store.event_store.get_turn_count()
    try:
        probe = graph_probe()
    except MistError as exc:
        return SeedOnlyCheck(logged_turns, None, f"{type(exc).__name__}: {exc}")
    return SeedOnlyCheck(logged_turns, probe, None)


def promote_seed_only_cutover(
    store: BacklogStore, *, graph_probe: GraphProbe, now_iso: str
) -> Promotion:
    """Make the filled candidate the active epoch without a staging rebuild or graph swap.

    Only for a live graph no extraction has ever run against (CUTOVER.md 6A):
    then it holds only what the seed applier wrote, and the candidate epoch
    can start from it as it is. That precondition is the operator's, and 6A's
    procedure (reseed with the backend stopped, then probe) is how it is
    shown; this function checks what it can observe, and the probe cannot see
    every extraction write (the comment above `SEED_ONLY_PROBE_CYPHER` lists
    what it misses). Refused, with nothing written, unless ALL of:

    a. a cutover is open and is 'ready' or 'checked';
    b. its fill is complete (`fill_scan(...).head is None`);
    c. `extraction_applied` has no row under ANY epoch;
    d. the active epoch is still the cutover's source epoch;
    e. the conversation log is empty: `conversation_turn_events` has no row,
       from any session origin (`log_not_empty_reason`);
    f. `graph_probe` succeeds and reports at least one node, every node and
       relationship carrying `seed_version`, and none carrying
       `ontology_version`, `extraction_version` or `model_hash`.

    c, d and e are pre-checked here (so the probe is not run for nothing: it
    runs only after a to e pass) and re-checked, with a state recheck, inside
    the promotion transaction (`EventStore.promote_epoch_cutover_seed_only`)
    under `BEGIN IMMEDIATE`, which writes the ledger row and a first-hand
    activation that marks nothing. A turn logged after the pre-check (during
    the probe, say) is refused there and nothing is written.

    b is checked here only, not inside the transaction (the candidate cache is
    a separate database); with e holding there is no logged turn for it to
    cover, so it is a guard kept, not the evidence.

    Raises:
        CutoverRefusedError: Any refusal above, or the store refused.
    """
    cutover = store.open_cutover()
    if cutover is None:
        raise CutoverRefusedError("no cutover is open")
    if cutover.state not in ("ready", "checked"):
        raise CutoverRefusedError(
            f"cutover {cutover.cutover_id} is {cutover.state!r}; wait until the dispatcher "
            "has filled it ('ready')"
        )
    fill = store.fill_scan(cutover)
    if fill.head is not None:
        raise CutoverRefusedError(
            f"the candidate covers {fill.covered} of {fill.total} logged turns; wait until "
            "the dispatcher has filled the rest"
        )
    markers = store.event_store.count_extraction_applied_by_stage()
    if markers:
        counts = ", ".join(f"{stage}={n}" for stage, n in markers.items())
        raise CutoverRefusedError(
            f"extraction_applied holds apply markers ({counts}): a turn has been applied to "
            "a graph under some epoch, so the live graph is not seed-only; use the "
            "graph-swapped path"
        )
    epoch = store.active_epoch()
    if epoch is None or epoch.epoch_id != cutover.source_epoch_id:
        raise CutoverRefusedError(
            f"the active epoch is {None if epoch is None else epoch.epoch_id}, but cutover "
            f"{cutover.cutover_id} began from epoch {cutover.source_epoch_id}; abandon it and "
            "begin again"
        )
    log_refusal = log_not_empty_reason(store.event_store.get_turn_count())
    if log_refusal is not None:
        raise CutoverRefusedError(log_refusal)

    try:
        probe = graph_probe()
    except MistError as exc:
        raise CutoverRefusedError(
            f"the live graph probe failed: {type(exc).__name__}: {exc}"
        ) from exc
    violations = probe.violations()
    if violations:
        raise CutoverRefusedError(
            "the live graph is not seed-only: " + "; ".join(violations) + f" ({probe.as_dict()})"
        )

    try:
        result = store.event_store.promote_epoch_cutover_seed_only(
            cutover_id=cutover.cutover_id, activated_at=now_iso, probe_report=probe.as_dict()
        )
    except EpochCutoverStateError as exc:
        raise CutoverRefusedError(str(exc)) from exc
    log_transition(
        cutover,
        cutover.state,
        "promoted",
        mode=PROMOTION_SEED_ONLY_GRAPH,
        epoch_id=result["epoch_id"],
        marked_applied=result["marked_applied"],
        turns_at_activation=result["turns_at_activation"],
        **probe.as_dict(),
    )
    return Promotion(
        cutover=cutover,
        epoch_id=int(result["epoch_id"]),
        marked_applied=int(result["marked_applied"]),
        turns_at_activation=int(result["turns_at_activation"]),
        mode=PROMOTION_SEED_ONLY_GRAPH,
        graph_probe=probe,
    )


# ---------------------------------------------------------------------------
# The check: `cutover rebuild`
# ---------------------------------------------------------------------------


@dataclass
class RebuildDeps:
    """What the check needs from the outside world; faked in unit tests.

    Attributes:
        live_uri: The live graph's bolt URI, a guard VALUE only.
        wipe_staging: Empties the staging graph before each build.
        build_regenerator: A fresh `LogRegenerator` over the staging graph.
        staging_form: The staging graph's canonical form, read after a build.
        live_form: The live graph's canonical form (read-only), for the
            self-model gate and the information-only live-vs-rebuilt diff.
        close: Releases connections.
    """

    live_uri: str
    wipe_staging: Callable[[], None]
    build_regenerator: Callable[[], Any]
    staging_form: Callable[[], str]
    live_form: Callable[[], str]
    close: Callable[[], None] = field(default=lambda: None)


@dataclass
class CheckResult:
    """The outcome of `check_cutover`."""

    exit_code: int
    report: dict[str, Any] = field(default_factory=dict)


def _in_scope_event_ids(store: BacklogStore, cutover: Cutover) -> list[str]:
    """The turns `LogRegenerator.rebuild` selects for this candidate, in replay order.

    The same call `rebuild` makes (`get_all_turns_for_reextraction` with the
    epoch's `ontology_version` and `CANONICAL_ORIGINS`), so its last element is
    the last turn the rebuild replays.
    """
    from backend.knowledge.regeneration.log_regenerator import CANONICAL_ORIGINS

    turns = store.event_store.get_all_turns_for_reextraction(
        ontology_version=cutover.ontology_version, origins=CANONICAL_ORIGINS
    )
    return [str(t["event_id"]) for t in turns]


class _ScopeChangedError(MistError):
    """A turn entered the rebuild's selection while the check ran."""


async def check_cutover(
    store: BacklogStore,
    deps_factory: Callable[[Cutover], RebuildDeps],
    *,
    staging_uri: str,
    min_seed_nodes: int,
    expect_turns: int,
    min_replay_edges: int,
    now_iso: str,
    gates: ModuleType | None = None,
) -> CheckResult:
    """Build the staging graph twice under the candidate and gate it.

    Gates, all from `backend.knowledge.regeneration.rebuild_gate` (none
    re-implemented): rebuild-twice identical, turns processed == expected (both
    builds), canonical form non-vacuous, replay-derived edges non-vacuous, and
    the self-model applied (present on both sides and equal). `live == rebuilt`
    is NOT gated -- a different model is expected to produce a different graph
    -- and its summary line is recorded as information.

    Every refusal `assert_rebuild_target_not_live` makes stands: it runs here
    before any staging write, and again inside every `LogRegenerator.rebuild`.
    `deps_factory` (which opens the Neo4j connections) is called only after the
    cutover's state, binding and coverage have been checked. An unbound
    candidate is refused: which rows a check certifies depends on the
    namespace the binding names, and a bind during the check could change it.

    One check beyond the gates: the rebuild's turn selection is read before and
    after the two builds and must be unchanged and equal to what each build
    processed, because `rebuilt_through_event_id` (the last turn replayed) is
    what promotion marks applied through. A turn logged into the selection
    mid-check would otherwise be in the graph but not marked, and be applied a
    second time after promotion.

    On a pass the cutover goes to 'checked' with the report, the second build's
    job id and `rebuilt_through_event_id`. On any failure or in-run refusal it
    goes (or stays) 'ready' with the report and those two columns cleared.

    Returns:
        The exit code (EXIT_*) and the report.
    """
    from backend.knowledge.eval_isolation import (
        RebuildTargetError,
        assert_rebuild_target_not_live,
    )
    from backend.knowledge.regeneration import rebuild_gate
    from backend.knowledge.regeneration.log_regenerator import (
        ColdCacheError,
        RebuildScopeError,
    )
    from backend.knowledge.regeneration.rebuild_gate import (
        RebuildDeterminismError,
        RebuildVacuityError,
    )

    gate = gates if gates is not None else rebuild_gate
    cutover = store.open_cutover()
    if cutover is None:
        return CheckResult(EXIT_REFUSED, {"refused": "no cutover is open"})
    if cutover.state not in ("ready", "checked"):
        return CheckResult(
            EXIT_REFUSED,
            {"refused": f"cutover {cutover.cutover_id} is {cutover.state!r}; wait for 'ready'"},
        )
    # An unbound candidate (one open at upgrade) reads the 3-part rows. If the
    # dispatcher bound it keyed during the check, the bind would send it back
    # to 'filling' and a keyed refill could bring it to 'ready' again before
    # the final transition, which compares only the state: the stale check
    # would be recorded on a keyed candidate. A binding is written once, so a
    # check that starts on a bound candidate cannot race a bind.
    if not cutover.bound:
        return CheckResult(
            EXIT_REFUSED,
            {
                "refused": (
                    f"cutover {cutover.cutover_id} is not bound yet, so its check cannot be "
                    "tied to its rows; wait until admin status shows it bound"
                )
            },
        )
    fill = store.fill_scan(cutover)
    if fill.head is not None:
        return CheckResult(
            EXIT_REFUSED,
            {
                "refused": (
                    f"the candidate covers {fill.covered} of {fill.total} logged turns; "
                    "wait until the dispatcher has filled the rest"
                )
            },
        )
    missing = [
        name
        for name, value in (
            ("--min-seed-nodes", min_seed_nodes),
            ("--expect-turns", expect_turns),
            ("--min-replay-edges", min_replay_edges),
        )
        if value < 1
    ]
    if missing:
        return CheckResult(
            EXIT_REFUSED,
            {"refused": f"{', '.join(missing)} must be >= 1, sized from the corpus"},
        )

    # A re-check wipes the staging graph a previous pass certified, so the
    # previous 'checked' must not survive it: demote first. A check that then
    # dies part way (a Neo4j error propagates from here) leaves 'ready', and
    # promotion refuses, instead of 'checked' over a half-built staging graph.
    if cutover.state == "checked":
        if not store.transition_cutover(
            cutover,
            from_states=("checked",),
            to_state="ready",
            updated_at=now_iso,
            check_report={"in_progress": True, "started_at": now_iso},
            write_check=True,
        ):
            return CheckResult(EXIT_REFUSED, {"refused": "the cutover changed state; re-run"})
        log_transition(cutover, "checked", "ready", reason="re-check started")

    report: dict[str, Any] = {
        "cutover_id": cutover.cutover_id,
        "checked_at": now_iso,
        "staging_uri": staging_uri,
        "expect_turns": expect_turns,
        "min_replay_edges": min_replay_edges,
        "min_seed_nodes": min_seed_nodes,
        # The namespace both builds read the candidate's rows from (None: the
        # 3-part one), so the report says which rows it certified.
        "identity_namespace": cutover.identity_namespace,
        "gates": {},
        "passed": False,
    }
    exit_code = EXIT_REFUSED
    rebuilt_through: str | None = None
    job_id: str | None = None
    deps: RebuildDeps | None = None
    try:
        deps = deps_factory(cutover)
        assert_rebuild_target_not_live(staging_uri, deps.live_uri)
        # Carries the candidate's `identity_namespace`, which `LogRegenerator`
        # reads its rows from: a keyed candidate is rebuilt from its own rows,
        # never from the 3-part rows of the same pair.
        epoch = cutover.epoch_dict()
        scope_before = _in_scope_event_ids(store, cutover)
        builds: list[tuple[str, Any]] = []
        for _ in range(2):
            deps.wipe_staging()
            regenerator = deps.build_regenerator()
            rebuild_report = await regenerator.rebuild(
                staging_uri=staging_uri,
                live_uri=deps.live_uri,
                epoch=epoch,
                min_seed_nodes=min_seed_nodes,
            )
            builds.append((deps.staging_form(), rebuild_report))
        scope_after = _in_scope_event_ids(store, cutover)
        (form_a, report_a), (form_b, report_b) = builds
        report["turns_processed"] = [report_a.turns_processed, report_b.turns_processed]
        report["total_logged"] = report_b.total_logged
        report["ontology_version"] = report_b.ontology_version
        report["origins"] = list(report_b.origins)
        report["seed_nodes_written"] = report_b.seed_nodes_written

        if scope_before != scope_after or any(
            r.turns_processed != len(scope_after) for r in (report_a, report_b)
        ):
            raise _ScopeChangedError(
                f"the rebuild's turn selection changed during the check ({len(scope_before)} "
                f"-> {len(scope_after)} turns; builds processed "
                f"{report_a.turns_processed} and {report_b.turns_processed}); re-run it"
            )
        report["gates"]["scope_stable"] = "passed"

        gate.assert_rebuild_twice_identical(form_a, form_b)
        report["gates"]["rebuild_twice_identical"] = "passed"
        for label, form, rebuild_report in (
            ("rebuild_1", form_a, report_a),
            ("rebuild_2", form_b, report_b),
        ):
            gate.assert_turns_processed(
                processed=rebuild_report.turns_processed, expected=expect_turns
            )
            gate.assert_canonical_form_non_vacuous(form, minimum_nodes=1)
            gate.assert_replay_derived_non_vacuous(form, minimum_edges=min_replay_edges)
            report["gates"][f"non_vacuous_{label}"] = "passed"

        live_form = deps.live_form()
        # Information only: a new model is expected to differ from live.
        report["live_vs_rebuilt"] = gate.live_vs_rebuilt_report(live_form, form_b).splitlines()[0]
        gate.assert_self_model_applied(live_form, form_b)
        report["gates"]["self_model_applied"] = "passed"

        rebuilt_through = scope_after[-1] if scope_after else None
        job_id = str(report_b.job_id)
        report["rebuilt_through_event_id"] = rebuilt_through
        report["rebuild_job_id"] = job_id
        report["passed"] = True
        exit_code = EXIT_OK
    except (
        RebuildTargetError,
        ColdCacheError,
        RebuildScopeError,
        CutoverRefusedError,
        _ScopeChangedError,
    ) as exc:
        exit_code = EXIT_REFUSED
        report["failure"] = _failure(exc)
    except RebuildDeterminismError as exc:
        exit_code = EXIT_NONDETERMINISTIC
        report["failure"] = _failure(exc)
    except RebuildVacuityError as exc:
        exit_code = EXIT_VACUOUS
        report["failure"] = _failure(exc)
    finally:
        if deps is not None:
            deps.close()

    report["exit_code"] = exit_code
    to_state = "checked" if exit_code == EXIT_OK else "ready"
    if not store.transition_cutover(
        cutover,
        from_states=("ready",),
        to_state=to_state,
        updated_at=now_iso,
        rebuild_job_id=job_id,
        rebuilt_through_event_id=rebuilt_through,
        check_report=report,
        write_check=True,
    ):
        report["recorded"] = False
        logger.warning(
            "cutover %d changed state during the check; its report was not recorded",
            cutover.cutover_id,
        )
        return CheckResult(EXIT_REFUSED, report)
    report["recorded"] = True
    log_transition(
        cutover,
        "ready",
        to_state,
        passed=report["passed"],
        exit_code=exit_code,
        rebuilt_through_event_id=rebuilt_through,
    )
    return CheckResult(exit_code, report)


def _failure(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= _FAILURE_TEXT_LIMIT else text[:_FAILURE_TEXT_LIMIT] + " [truncated]"


def format_report(report: dict[str, Any]) -> str:
    """The check report as indented JSON, for the CLI."""
    return json.dumps(report, indent=2, sort_keys=True, default=str)


# ---------------------------------------------------------------------------
# Real wiring (needs Neo4j; not exercised by the unit tier)
# ---------------------------------------------------------------------------


def candidate_pack_refusal(cutover: Cutover, pack: CarriesPackHashes) -> str | None:
    """Why `pack` may not rebuild the candidate's staging graph, or None when it may.

    The rebuild runs `pack`'s rails over the candidate's cached rows and stamps
    the staging edges with `pack`'s hashes, so a pack whose hashes differ from
    the candidate's would build a graph the candidate's stamps do not describe.
    A candidate that holds no hash is not compared (`pack_hash_mismatches`).

    Args:
        cutover: The candidate.
        pack: The pack the rebuild would run: anything with `pack_prompt_hash`
            and `pack_rails_hash`.
    """
    differing = pack_hash_mismatches(cutover, pack)
    if not differing:
        return None
    return (
        f"cutover {cutover.cutover_id} was begun under the ontology pack "
        f"({describe_pack_hashes(cutover, differing)}), but this environment's active pack "
        f"is ({describe_pack_hashes(pack, differing)}); run the rebuild from the code whose "
        "pack the candidate names, or abandon and begin again"
    )


def replay_pack_consumers(
    pack: Pack, *, min_confidence: float
) -> tuple[EntityNormalizer, ExtractionValidator]:
    """The rebuild's normalizer and validator, both over `pack`.

    One pack for both, and the one `candidate_pack_refusal` compared: the
    rails that run and the types that validate are the pack whose hashes the
    staging edges are stamped with.

    Returns:
        `(EntityNormalizer, ExtractionValidator)`.
    """
    from backend.knowledge.extraction.normalizer import EntityNormalizer
    from backend.knowledge.extraction.validator import ExtractionValidator

    return (
        EntityNormalizer(embedding_generator=None, executor=None, pack=pack),
        ExtractionValidator(min_confidence=min_confidence, pack=pack),
    )


def build_rebuild_deps_from_env(
    store: BacklogStore, staging_uri: str, cutover: Cutover
) -> RebuildDeps:
    """Wire the real staging rebuild, as `scripts/mist_admin.py::_build_log_regenerator` does.

    Same construction, from the same backend factories (nothing is imported
    from `scripts/`), with these deliberate differences:

    - the replay's normalizer and validator, and the curation pipeline's
      stamps, are built over one `registry.active_pack()`, which is refused
      first if its hashes differ from those the candidate holds
      (`candidate_pack_refusal`);

    - the curation pipeline is built from a config whose stamps are the
      CANDIDATE's (`dataclasses.replace` of `ontology_version`,
      `extraction_version` and the bare `model_hash`), so the staging graph's
      edges carry the stamps of the epoch it will become. The composed value is
      re-checked against the cutover row and refused if the embedding model has
      changed since `begin`;
    - the replay source is this admin's own event store and cache
      (`store.event_store`, `store.cache`); the regenerator writes no row to
      either (`NullRebuildJournal`).

    `assert_rebuild_target_not_live` runs before any connection is opened.

    Raises:
        CutoverRefusedError: The composed candidate hash no longer matches, or
            the active pack's hashes differ from the candidate's.
        RebuildTargetError: The staging URI is not an allowlisted staging
            endpoint, or resolves to live.
    """
    from dataclasses import replace
    from pathlib import Path

    from backend.factories import build_curation_pipeline
    from backend.knowledge.canonical_serialize import canonical_graph_form
    from backend.knowledge.config import Neo4jConfig, get_config
    from backend.knowledge.embeddings.embedding_generator import EmbeddingGenerator
    from backend.knowledge.eval_isolation import assert_rebuild_target_not_live
    from backend.knowledge.extraction.confidence import ConfidenceScorer
    from backend.knowledge.extraction.temporal import TemporalResolver
    from backend.knowledge.ontologies import registry
    from backend.knowledge.regeneration.log_regenerator import LogRegenerator
    from backend.knowledge.regeneration.rebuild_journal import NullRebuildJournal
    from backend.knowledge.regeneration.staging_seeder import StagingSeeder
    from backend.knowledge.seed.loader import load_seed_documents
    from backend.knowledge.storage.graph_executor import GraphExecutor
    from backend.knowledge.storage.graph_store import GraphStore
    from backend.knowledge.storage.neo4j_connection import Neo4jConnection
    from backend.knowledge.version_stamps import compose_model_hash

    config = get_config()
    candidate_config = replace(
        config,
        ontology_version=cutover.ontology_version,
        extraction_version=cutover.extraction_version,
        model_hash=cutover.bare_model_hash,
    )
    if compose_model_hash(candidate_config) != cutover.model_hash:
        raise CutoverRefusedError(
            f"the candidate's composed model hash is {cutover.model_hash!r}, but this "
            f"environment composes {compose_model_hash(candidate_config)!r} (the embedding "
            "model changed since `cutover begin`); abandon and begin again"
        )
    pack = registry.active_pack()
    pack_refusal = candidate_pack_refusal(cutover, pack)
    if pack_refusal is not None:
        raise CutoverRefusedError(pack_refusal)

    live_uri = config.neo4j.uri
    assert_rebuild_target_not_live(staging_uri, live_uri)

    seed_documents = load_seed_documents(Path(config.vault.root) / "seed")
    embedding_provider = EmbeddingGenerator(
        config.embedding.model_name, device=config.embedding.resolved_device()
    )

    staging_conn = Neo4jConnection(
        Neo4jConfig(uri=staging_uri, username=config.neo4j.username, password=config.neo4j.password)
    )
    staging_conn.connect()
    live_conn = Neo4jConnection(config.neo4j)  # read only: canonical_graph_form
    try:
        live_conn.connect()
    except MistError:
        staging_conn.disconnect()
        raise

    def build_regenerator() -> LogRegenerator:
        GraphStore(
            connection=staging_conn, embedding_generator=embedding_provider
        ).initialize_schema()
        pipeline = build_curation_pipeline(
            candidate_config,
            GraphExecutor(staging_conn),
            embedding_provider=embedding_provider,
            pack=pack,
        )
        normalizer, validator = replay_pack_consumers(
            pack, min_confidence=config.extraction.min_extraction_confidence
        )
        return LogRegenerator(
            event_store=store.event_store,
            extraction_cache=store.cache,
            staging_curation_pipeline=pipeline,
            confidence_scorer=ConfidenceScorer(),
            temporal_resolver=TemporalResolver(),
            normalizer=normalizer,
            validator=validator,
            journal=NullRebuildJournal(),
            staging_seeder=StagingSeeder(
                connection=staging_conn,
                documents=seed_documents,
                seed_version=seed_documents[0].seed_version,
                embedding_generator=embedding_provider,
                expected_dimension=config.embedding.dimension,
            ),
        )

    def close() -> None:
        staging_conn.disconnect()
        live_conn.disconnect()

    return RebuildDeps(
        live_uri=live_uri,
        wipe_staging=lambda: staging_conn.execute_write("MATCH (n) DETACH DELETE n", {}),
        build_regenerator=build_regenerator,
        staging_form=lambda: canonical_graph_form(
            staging_conn, include_provenance=True, include_self_model=True
        ),
        live_form=lambda: canonical_graph_form(
            live_conn, include_provenance=True, include_self_model=True
        ),
        close=close,
    )
