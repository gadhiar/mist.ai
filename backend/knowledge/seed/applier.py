"""Apply the versioned seed source to the graph, deterministically.

No LLM is involved: seed facts are authored, so they are written as given
rather than inferred from prose (R1.4 spec 2.0). Every node and edge carries
`seed_version`, which is what makes the wipe applied elsewhere for a given
version exact -- a node or edge written without the stamp is un-wipeable and
becomes permanent graph litter that no gate can detect.

R1.4 Task 12 (addendum): every node also gets every descriptive property the
source defines (`entity_type` plus whatever `SeedNode` carries beyond
`id`/`type`). Before that task, the node MERGE set only
`seed_version`/`created_at`/`updated_at` -- Task 10's live run proved that a
wipe-and-recreate cycle then leaves a node with no type and no descriptive
properties at all, since `MERGE` preserves untouched properties on a MATCH
but a fresh CREATE gets nothing beyond what the query explicitly sets. See
the Task 10 report for the live consequence.

MIS-177 (i115): the node's type is the `entity_type` PROPERTY, not a graph
label. Task 12 also wrote the type as a label on every node; on
`:__Entity__` nodes that departed from the documented convention
(`scripts/migrations/ontology_v1_4_0.py` module docstring: types are stored
only as `entity_type`, with `:User` and `:MistIdentity` as the label
invariants), and no reader used it -- every type reader filters on the
property (`grep -rn 'entity_type' backend/knowledge/curation/deduplication.py
backend/knowledge/storage/graph_store.py backend/vault/user_snapshot.py`;
`count_nodes_by_type` in admin.py). `node_type_label` below states which
nodes still get one.
"""

import difflib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from backend.errors import Neo4jQueryError, SeedSourceError, SeedTargetNotSeedOnlyError
from backend.interfaces import GraphConnection
from backend.knowledge.admin import _EXTRACTION_STAMP_PROPERTIES, count_reset_guard_elements
from backend.knowledge.eval_isolation import assert_neo4j_uri_not_live
from backend.knowledge.ontologies import ALL_NODE_TYPE_NAMES
from backend.knowledge.ontologies.v1_0_0 import ALL_EDGE_TYPE_NAMES
from backend.knowledge.storage.partitions import ENTITY_LABEL, SELF_MODEL_LABEL

from .models import (
    SEED_CONFIDENCE,
    SEED_PROVENANCE,
    SEED_SOURCE_TYPE,
    SeedDocument,
    SeedNode,
)

logger = logging.getLogger(__name__)

# Partition label and ontology type label are both interpolated (by
# `_merge_node_query`), never fixed constants. Partition: the graph has two
# id-scoped, constraint-isolated partitions (`entity_id_unique` on
# :__Entity__, `selfmodel_id_unique` on :__SelfModel__) and a hardcoded
# label here would create a duplicate :__Entity__ copy of every live
# :__SelfModel__ node the self-model seed content (`seed/mist.md`)
# references, silently orphaning the real self-model (R1.4 Task 4 rework,
# found during Task 8). No runtime allowlist check guards the partition
# interpolation the way `_validate_predicates` guards the edge type below:
# `SeedDocument.partition` is `Literal`-typed against exactly
# `ENTITY_LABEL`/`SELF_MODEL_LABEL`, which makes constructing a document
# with any other value impossible, so the type-level closure IS the guard.
# Type: `SeedNode.type` has no equivalent type-level closure (the
# ontology's node types are too numerous and version-dependent to
# enumerate as a `Literal`, same reasoning as `predicate`) -- guarded by
# `_validate_node_types` below, at this exact interpolation point, for the
# identical reason `_validate_predicates` guards `_MERGE_EDGE`'s `%s`
# rather than trusting Task 11's loader-level check alone (a caller that
# constructs `SeedDocument`s directly bypasses the loader entirely).
#
# `n += $properties` on BOTH branches (not just ON MATCH) makes re-seeding
# enforce the source as ground truth for every property it defines, without
# touching properties the applier does not own (e.g. `embedding`). Only
# `created_at` is create-only, mirroring `backend/knowledge/admin.py`'s
# `_seed_internal_nodes` (the established production precedent for this
# MERGE shape). The label clauses and the RETURN are appended by
# `_merge_node_query`.
_MERGE_NODE = (
    "MERGE (n:%s {id: $id}) "
    "ON CREATE SET n.created_at = $now, n += $properties "
    "ON MATCH SET n += $properties "
)

# The only ontology type label an `:__Entity__` node keeps as a graph label.
# `:User` is an invariant of the user node, set by every writer that writes it
# (`grep -n 'SET e:User' backend/knowledge/curation/graph_writer.py
# backend/knowledge/storage/graph_store.py`); every other `:__Entity__` type
# lives in `entity_type` only (MIS-177 D2, module docstring above).
ENTITY_TYPE_LABELS_KEPT: frozenset[str] = frozenset({"User"})

# A label is interpolated into Cypher, so each name is checked to be a plain
# identifier before it is used -- `ALL_NODE_TYPE_NAMES` is authored ontology
# data, not user input, but a name that needed backtick-quoting would turn a
# REMOVE into a syntax error or a different statement.
_LABEL_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def node_type_label(partition: str, node_type: str) -> str | None:
    """Return the ontology type label a seed node carries, or None for none.

    `:__SelfModel__` nodes keep their type label (`:MistIdentity` is an
    invariant, and `ensure_mist_identity` MERGEs on
    `(:__SelfModel__:MistIdentity {id})`: `grep -n 'MERGE (m:__SelfModel__'
    backend/knowledge/storage/graph_store.py`). An `:__Entity__` node keeps
    only `User` (`ENTITY_TYPE_LABELS_KEPT`).

    Args:
        partition: `ENTITY_LABEL` or `SELF_MODEL_LABEL`.
        node_type: The node's validated ontology type.

    Returns:
        The label to SET, or None when the node carries no type label.
    """
    if partition == SELF_MODEL_LABEL or node_type in ENTITY_TYPE_LABELS_KEPT:
        return node_type
    return None


def _require_label_identifier(name: str, clause: str) -> None:
    """Raise ValueError unless `name` can be interpolated as a bare Cypher label."""
    if not _LABEL_IDENTIFIER.match(name):
        raise ValueError(
            f"ontology node type {name!r} is not a plain identifier; refusing to "
            f"interpolate it into a {clause} clause"
        )


def entity_type_labels_removed(kept: str | None) -> list[str]:
    """Return every ontology type label an `:__Entity__` seed node must not carry.

    Built from `ALL_NODE_TYPE_NAMES` (the validated ontology), never from the
    seed source, minus the one label the node keeps (if any).

    Raises:
        ValueError: An ontology type name is not a plain Cypher identifier.
    """
    removed = [name for name in ALL_NODE_TYPE_NAMES if name != kept]
    for name in removed:
        _require_label_identifier(name, "REMOVE")
    return removed


def _merge_node_query(partition: str, node_type: str) -> str:
    """Build the node MERGE for one seed node.

    An `:__Entity__` node ends with no ontology type label except `User`,
    whether the MERGE created it or matched an existing node: the REMOVE
    strips any type label an earlier writer (a pre-MIS-177 seed, the retired
    `admin.apply_seed`) left on it. A `:__SelfModel__` node gets its type
    label SET and nothing removed.

    Every label it interpolates, SET and REMOVE, is checked here, and
    `_build_node_queries` calls this for every node before the first write,
    so the check never fires mid-write.

    Raises:
        ValueError: A label it would interpolate is not a plain identifier.
    """
    label = node_type_label(partition, node_type)
    query = _MERGE_NODE % partition
    if label is not None:
        _require_label_identifier(label, "SET")
        query += f"SET n:{label} "
    if partition == ENTITY_LABEL:
        query += "REMOVE n:" + ":".join(entity_type_labels_removed(label)) + " "
    return query + "RETURN n.id AS id"


# The label union (`:A|B`) matches a node in EITHER partition -- this MATCH
# must find self-model nodes as readily as entity nodes, since a fact's
# subject/object may resolve to either. Mirrors the existing production
# precedent at backend/knowledge/admin.py's edge-merge helper, which solves
# the identical two-partition matching problem for the older seed_data.yaml
# path.
#
# KG-125 (MIS-175, Option A): every seed edge also carries what it IS --
# `provenance='seed'`, `source_type='stated'`, `confidence=1.0` (the
# `SEED_*` constants in models.py, passed as parameters). Before this, a seed
# edge carried only `seed_version`, so the reconciliation engine read it
# through its extraction defaults (0.8, 'extracted') and stamped any clamped
# copy of it `provenance='extraction'`: the seed origin was lost the first time
# a conversation retired a seed belief. This statement never WRITES
# `ontology_version`/`extraction_version`/`model_hash`: those are extraction
# stamps, and the seed-only cutover probe (`extraction_backlog/cutover.py`,
# `SEED_ONLY_PROBE_CYPHER`) refuses any element carrying one.
#
# The MERGE is keyed only on (s, type, o), so it matches EVERY existing edge
# of that type between the two nodes, not just the one it wrote last time.
# Adopting a non-seed edge would set `seed_version` on it, reset
# `valid_from`/`valid_to` to the fact's (reopening a retired belief) and
# overwrite provenance/source_type/confidence, and the next reseed's wipe would
# delete it. MIS-177 D1 (the former D11 follow-up) closes that for the edges it
# can see: `_assert_seed_target_holds_only_seed` refuses, before any write or
# wipe, a graph holding any edge touching an `:__Entity__` node without
# `seed_version` -- which includes every clamped copy
# `curation/reconciliation.py` `_apply_append` appends (`grep -nE 'MATCH
# \(t:__Entity__|r.seed_origin_version = '
# backend/knowledge/curation/reconciliation.py`) -- any edge carrying
# `extraction_version` or `model_hash`, and any `provenance='extraction'` edge. What it still cannot see -- unstamped edges between two
# `:__SelfModel__` nodes -- is named above `SEED_GUARD_STAMP_PROPERTIES`.
_MERGE_EDGE = (
    f"MATCH (s:{ENTITY_LABEL}|{SELF_MODEL_LABEL} {{id: $subject}}) "
    f"MATCH (o:{ENTITY_LABEL}|{SELF_MODEL_LABEL} {{id: $object}}) "
    "MERGE (s)-[r:%s]->(o) "
    "SET r.seed_version = $seed_version, r.valid_from = $valid_from, "
    "    r.valid_to = $valid_to, r.provenance = $provenance, "
    "    r.source_type = $source_type, r.confidence = $confidence, "
    "    r.updated_at = $now "
    "RETURN type(r) AS t"
)

_WIPE_EDGES = "MATCH ()-[r]->() WHERE r.seed_version = $seed_version DELETE r RETURN count(r) AS n"
_WIPE_NODES = (
    "MATCH (n) WHERE n.seed_version = $seed_version "
    "AND NOT (n)--() "
    "DELETE n RETURN count(n) AS n"
)

# ---------------------------------------------------------------------------
# The seed guard (MIS-177 D1)
#
# A seed or reseed refuses, before any write or wipe, a graph that holds:
#   (a) anything `admin.RESET_GUARD_CYPHER` counts: an `:__Entity__` node whose
#       provenance is not 'seed', or with no `seed_version`, or with any of
#       `admin._EXTRACTION_STAMP_PROPERTIES` (its own `ontology_version` arm
#       included); or a relationship touching an `:__Entity__` node with no
#       `seed_version` or with such a stamp. Reused through
#       `admin.count_reset_guard_elements`, not restated;
#   (b) any node or relationship, in ANY partition, carrying one of
#       `SEED_GUARD_STAMP_PROPERTIES`;
#   (c) any node or relationship, in any partition, with
#       `provenance = 'extraction'` (writers: `grep -n "provenance = 'extraction'"
#       backend/knowledge/curation/graph_writer.py
#       backend/knowledge/curation/reconciliation.py`).
# No override. Adopting such an element is what this guard exists to stop:
# the seed MERGEs (`_MERGE_NODE`, `_MERGE_EDGE`) match on id and on
# (subject, type, object) alone.
# ---------------------------------------------------------------------------

# The stamps (b) tests in every partition. `ontology_version` is deliberately
# NOT one of them, although `admin._EXTRACTION_STAMP_PROPERTIES` lists it:
# `GraphStore.ensure_mist_identity` writes `ontology_version` on the
# `:__SelfModel__:MistIdentity` node it creates
# (`grep -n 'm.ontology_version = ' backend/knowledge/storage/graph_store.py`),
# and the backend calls it at startup whenever internal derivation is enabled
# (`grep -n 'gs.ensure_mist_identity()' backend/factories.py`). With
# `ontology_version` here, a graph whose backend started before its first seed
# could never be seeded, and no command clears the property.
#
# The gap this leaves, deliberately not closed here (a follow-up decision):
# Stage 9 self-model nodes carry `ontology_version` as their only stamp and no
# provenance (`grep -n 'e.ontology_version = \$ontology_version'
# backend/knowledge/extraction/internal_derivation.py`), and the edge it
# MERGEs from MistIdentity to them carries no property at all (`grep -n
# 'MERGE (m)-\[:{rel_type}\]->(e)'
# backend/knowledge/extraction/internal_derivation.py`).
# `SkillDerivationJob._ensure_capability` writes the same shape
# (`grep -nE 'e.ontology_version = |MERGE \(m\)-\[:HAS_CAPABILITY\]'
# backend/knowledge/curation/skill_derivation.py`). Both are `:__SelfModel__`
# only, so (a) never sees them either; a seed node or fact with the same id or
# (subject, type, object) would adopt them.
SEED_GUARD_STAMP_PROPERTIES = ("extraction_version", "model_hash")


def _seed_guard_stamped(var: str) -> str:
    """Cypher: `var` carries at least one of `SEED_GUARD_STAMP_PROPERTIES`."""
    return "(" + " OR ".join(f"{var}.{p} IS NOT NULL" for p in SEED_GUARD_STAMP_PROPERTIES) + ")"


# ONE read-only statement for (b) and (c). `MATCH (n) WITH count(...)` yields
# one row even on an empty graph, and the OPTIONAL MATCH keeps that row when
# the graph has no relationship; the relationship CASEs test `r IS NOT NULL`
# first because OPTIONAL MATCH then binds one NULL `r`.
SEED_GUARD_CYPHER = (
    "MATCH (n) "
    "WITH "
    f"count(CASE WHEN {_seed_guard_stamped('n')} THEN 1 END) AS stamped_nodes, "
    "count(CASE WHEN n.provenance = 'extraction' THEN 1 END) AS extraction_nodes "
    "OPTIONAL MATCH ()-[r]->() "
    "RETURN stamped_nodes, extraction_nodes, "
    f"count(CASE WHEN r IS NOT NULL AND {_seed_guard_stamped('r')} THEN 1 END) "
    "AS stamped_relationships, "
    "count(CASE WHEN r IS NOT NULL AND r.provenance = 'extraction' THEN 1 END) "
    "AS extraction_relationships"
)

SEED_GUARD_COLUMNS = (
    "stamped_nodes",
    "extraction_nodes",
    "stamped_relationships",
    "extraction_relationships",
)


@dataclass(frozen=True, slots=True)
class SeedGuardCounts:
    """What the seed guard counted in the target graph. Every field must be 0 to seed."""

    reset_guard_nodes: int
    reset_guard_relationships: int
    stamped_nodes: int
    stamped_relationships: int
    extraction_nodes: int
    extraction_relationships: int

    def blocking(self) -> bool:
        """True when any count is nonzero."""
        return any(
            (
                self.reset_guard_nodes,
                self.reset_guard_relationships,
                self.stamped_nodes,
                self.stamped_relationships,
                self.extraction_nodes,
                self.extraction_relationships,
            )
        )

    def describe(self) -> str:
        """Every count, by name, for the refusal message."""
        return (
            f"{self.reset_guard_nodes} non-seed :__Entity__ node(s) and "
            f"{self.reset_guard_relationships} non-seed relationship(s) touching one "
            "(admin.RESET_GUARD_CYPHER); "
            f"{self.stamped_nodes} node(s) and {self.stamped_relationships} relationship(s) "
            "in any partition carrying extraction_version or model_hash; "
            f"{self.extraction_nodes} node(s) and {self.extraction_relationships} "
            "relationship(s) with provenance='extraction'"
        )


def count_seed_guard_elements(connection: GraphConnection) -> SeedGuardCounts:
    """Run the seed guard's two read-only statements and return their counts.

    Fails closed, like `admin.count_reset_guard_elements` (which it calls for
    (a)): a missing row, or a row without every column, raises rather than
    reading as zero.

    Raises:
        Neo4jQueryError: Either statement returned no row or an unexpected row.
    """
    reset_nodes, reset_relationships = count_reset_guard_elements(connection)
    rows = connection.execute_query(SEED_GUARD_CYPHER)
    if not rows:
        raise Neo4jQueryError("seed guard query returned no row")
    row = rows[0]
    try:
        values = {name: int(row[name]) for name in SEED_GUARD_COLUMNS}
    except (KeyError, TypeError, ValueError) as exc:
        raise Neo4jQueryError(f"seed guard query returned an unexpected row: {row!r}") from exc
    return SeedGuardCounts(
        reset_guard_nodes=reset_nodes,
        reset_guard_relationships=reset_relationships,
        **values,
    )


def _assert_seed_target_holds_only_seed(connection: GraphConnection, *, action: str) -> None:
    """Refuse to seed a graph holding non-seed or extraction-written elements.

    Raises:
        SeedTargetNotSeedOnlyError: Any seed guard count is nonzero; the
            message names every count.
        Neo4jQueryError: A guard statement failed or returned no usable row.
    """
    counts = count_seed_guard_elements(connection)
    if counts.blocking():
        raise SeedTargetNotSeedOnlyError(
            f"Refusing {action}: the target graph holds {counts.describe()}. Seeding "
            "would adopt such elements and a later reseed's wipe would delete them. "
            "Seed an empty or seed-only graph (graph-reset --include-derived removes "
            ":__Entity__ and :__Provenance__ data; :__SelfModel__ data is not reset)."
        )


def _assert_seed_target_permitted(
    connection: GraphConnection, *, allow_live: bool, action: str
) -> None:
    """Refuse a live target unless the caller said `allow_live` (F1).

    Default-CLOSED, and at the WRITE site rather than the call site. Every
    isolation guard in the repo reasons about URI strings; these two functions
    take a connection OBJECT, so none of them could be pointed at the thing
    issuing the writes. A guard the caller must remember to add is absent
    exactly when it matters -- at the R1.7 seed-apply insertion point
    (`log_regenerator.py:445`), `source_conn` and `staging_conn` are both in
    scope and differ by six characters.

    A connection with no readable `.config.uri` is NOT a real `Neo4jConnection`
    and cannot reach live, so it passes. That is the test-double case, and it is
    sound rather than a loophole: the threat model is a real connection aimed at
    the canonical graph, and a real one always exposes its URI.
    """
    if allow_live:
        return
    uri = getattr(getattr(connection, "config", None), "uri", None)
    if uri is None:
        return
    assert_neo4j_uri_not_live(uri, action=action)


def apply_seed_documents(
    connection: GraphConnection,
    documents: list[SeedDocument],
    *,
    seed_version: str,
    now_iso: str,
    allow_live: bool = False,
) -> dict[str, int]:
    """Write every fact in `documents` to the graph, stamped with `seed_version`.

    Predicates are validated against the ontology's known relationship types
    before any write happens (see `_validate_predicates`), so a single typo
    anywhere in the seed source aborts the whole application rather than
    leaving a partial write -- some nodes and edges stamped, others not.
    Node types are validated the same way (`_validate_node_types`).

    Every node referenced by a fact (`_assign_node_partitions`'s output --
    unchanged from before Task 12; which ids get written is still driven by
    fact references, not by `doc.nodes` membership) is written with its full
    `SeedNode` definition: `entity_type` property and every other
    descriptive property the source defines (R1.4 Task 12). Type labels
    follow `node_type_label`: a `:__SelfModel__` node and a `User` node get
    their type label; any other `:__Entity__` node ends with none (MIS-177).

    Args:
        connection: Sync graph connection. Callers in async contexts must
            offload -- see the root `CLAUDE.md` Async Boundaries rule (never
            call sync Neo4j from async code; use `GraphExecutor`).
        documents: Parsed seed documents, in application order.
        seed_version: The one global version (spec O10). Passed explicitly
            rather than read off the documents so the caller cannot apply a
            different version than it wiped.
        now_iso: Timestamp for `created_at` / `updated_at`. Passed in rather
            than read from the clock so application is byte-reproducible --
            two calls with identical input must produce identical writes.
        allow_live: Permit a connection pointed at the canonical graph (F1).
            Defaults to False so the dangerous call is the one that has to be
            spelled out; `cmd_seed` is the only production caller that sets it.

    Returns:
        Counts keyed `nodes` and `facts`.

    Raises:
        SeedSourceError: A fact's predicate is not a recognized ontology
            relationship type, a node's type is not a recognized ontology
            node type, the same node id is assigned to two different
            partitions by different documents, the same node id is defined
            more than once, or a fact references a node id with no matching
            `SeedNode` definition (this last case is Task 11's
            referential-integrity check re-asserted here as the applier's
            own defense -- a caller that constructs `SeedDocument`s
            directly, bypassing `load_seed_documents`, is not protected by
            a loader-only check). Also raised, before any write, for a
            node carrying an extraction stamp
            (`_validate_no_extraction_stamps`) or a label that is not a plain
            identifier (`_build_node_queries`).
        SeedTargetNotSeedOnlyError: The target graph holds an element the
            seed guard counts (`_assert_seed_target_holds_only_seed`).
            Raised before any write.
        Neo4jQueryError: A seed guard statement failed or returned no
            usable row (fail closed). Raised before any write.
    """
    _assert_seed_target_permitted(
        connection, allow_live=allow_live, action="applying seed documents"
    )
    _validate_predicates(documents)
    _validate_node_types(documents)
    _validate_no_extraction_stamps(documents)
    node_partitions = _assign_node_partitions(documents)
    node_definitions = _collect_node_definitions(documents)
    node_queries = _build_node_queries(node_partitions, node_definitions)
    # MIS-177 D1: after the source checks (which need no query), before the
    # first write.
    _assert_seed_target_holds_only_seed(connection, action="applying seed documents")

    for node_id in sorted(node_partitions):
        node = node_definitions.get(node_id)
        if node is None:
            raise SeedSourceError(
                f"fact references node id {node_id!r}, which has no matching "
                "`SeedNode` definition -- every fact's subject and object must "
                "have a node definition (R1.4 Task 11/12)"
            )
        # Authored properties spread FIRST, applier-owned stamps LAST -- a
        # dict literal's later keys win. R1.4 whole-branch review, I4: the
        # original ordering put the spread last, so an authored
        # `seed_version`/`entity_type`/`updated_at` (SeedNode's extra="allow"
        # lets any name through) silently overrode the applier's own stamp.
        # `SeedNode._no_applier_owned_extras` (models.py) now rejects that at
        # construction time, but this ordering is an independent second
        # layer: it holds even for a `SeedNode` that reached this function
        # without going through that validator (`model_construct`, or a
        # future caller that constructs `SeedDocument`s directly) -- the
        # same "loader check doesn't protect a direct constructor" reasoning
        # this file already applies to `_validate_node_types`/
        # `_validate_predicates`.
        #
        # `created_at` is excluded from the spread entirely, not merely
        # ordered to lose: unlike the three keys above, it is never supposed
        # to be a member of `properties` at all -- it is set exclusively by
        # `_MERGE_NODE`'s own `ON CREATE SET n.created_at = $now` clause. An
        # authored `created_at` slipping into `properties` would reach the
        # graph via `n += $properties` on BOTH branches, corrupting the
        # create-only guarantee on every future ON MATCH re-seed, not merely
        # losing a values comparison on write.
        #
        # `provenance` (KG-125): a seed node is seed-authored. The graph-reset
        # guard (`admin.RESET_GUARD_CYPHER`) counts an `:__Entity__` node
        # whose provenance is not 'seed' as derived data, so a node without
        # it blocks a reset; `provenance` alone does not pass one, since the
        # guard also needs `seed_version` and no extraction stamp.
        # Applier-owned like the other stamps, so it is in
        # `_APPLIER_OWNED_NODE_PROPERTIES` and sits after the spread.
        properties = {
            **{k: v for k, v in node.model_dump().items() if k not in ("id", "type", "created_at")},
            "entity_type": node.type,
            "seed_version": seed_version,
            "provenance": SEED_PROVENANCE,
            "updated_at": now_iso,
        }
        connection.execute_write(
            node_queries[node_id],
            {"id": node_id, "now": now_iso, "properties": properties},
        )

    fact_count = 0
    for doc in documents:
        for fact in doc.facts:
            connection.execute_write(
                _MERGE_EDGE % fact.predicate,
                {
                    "subject": fact.subject,
                    "object": fact.object,
                    "predicate": fact.predicate,
                    "seed_version": seed_version,
                    "valid_from": fact.valid_from,
                    "valid_to": fact.valid_to,
                    "provenance": SEED_PROVENANCE,
                    "source_type": SEED_SOURCE_TYPE,
                    "confidence": SEED_CONFIDENCE,
                    "now": now_iso,
                },
            )
            fact_count += 1

    logger.info(
        "Seed applied: %d nodes, %d facts at version %s",
        len(node_partitions),
        fact_count,
        seed_version,
    )
    return {"nodes": len(node_partitions), "facts": fact_count}


def wipe_seed_version(connection: GraphConnection, seed_version: str) -> dict[str, int]:
    """Remove everything stamped with `seed_version`.

    Scoped entirely on the `seed_version` property -- never on label or id
    patterns. Real conversation-derived facts share `__Entity__` and the
    ontology's relationship types with seeded ones, so an unscoped delete,
    or one scoped on anything broader than the stamp, would destroy the
    user's actual memory alongside the seed content.

    Edges are deleted first, then nodes left with no remaining
    relationship. Order matters: reversed, `NOT (n)--()` would find nothing
    orphaned (the seeded edges are still attached) and the node delete
    would silently no-op.

    A seeded node that has since acquired a conversation-derived edge is
    deliberately kept -- `NOT (n)--()` excludes any node still holding a
    relationship, seeded or not. Dropping it would delete a
    conversation-derived fact, which the seed layer has no authority to do.

    Args:
        connection: Sync graph connection.
        seed_version: The exact stamp to remove.

    Returns:
        Counts keyed `edges` and `nodes`.
    """
    edge_result = connection.execute_write(_WIPE_EDGES, {"seed_version": seed_version})
    node_result = connection.execute_write(_WIPE_NODES, {"seed_version": seed_version})
    edges_removed = _count(edge_result)
    nodes_removed = _count(node_result)

    logger.info(
        "Seed wiped: %d edges, %d nodes at version %s",
        edges_removed,
        nodes_removed,
        seed_version,
    )
    return {"edges": edges_removed, "nodes": nodes_removed}


def reseed(
    connection: GraphConnection,
    documents: list[SeedDocument],
    *,
    seed_version: str,
    now_iso: str,
    allow_live: bool = False,
) -> dict[str, int]:
    """Wipe `seed_version` and re-apply `documents` under the same version.

    MERGE alone cannot remove a fact that was deleted from the source: a
    fact written by a prior application but absent from `documents` would
    otherwise persist in the graph forever, silently, and no gate catches
    it -- Gate 2 checks that authored facts are present, never that
    unauthored ones are absent. Wiping first is what makes the graph
    actually track the source rather than only ever accumulate it.

    Predicates, node types, and node-partition/definition assignment are all
    validated before the wipe runs, not just before the re-apply's writes
    (`apply_seed_documents` already guards every one of these; this call is
    deliberately redundant -- see the identical redundancy for
    `_validate_predicates`, established before this function existed).
    Without this, a typo, an unknown node type, or a partition/duplicate-id
    conflict introduced in a source edit would empty a previously-good graph
    via the wipe and then abort the re-apply, leaving a real data-loss
    window open until the source is fixed.

    Args:
        connection: Sync graph connection.
        documents: Parsed seed documents to apply after the wipe.
        seed_version: The one global version wiped and re-applied together
            -- a caller cannot wipe one version and apply another.
        now_iso: Timestamp forwarded to `apply_seed_documents`. Required,
            not read from the clock, so re-seeding is byte-reproducible.
        allow_live: Permit a connection pointed at the canonical graph (F1).
            Checked here BEFORE the wipe, not only in the delegate -- a guard
            that fired after `wipe_seed_version` would refuse an already-empty
            graph.

    Returns:
        Counts keyed `nodes` and `facts`, from the re-apply.

    Raises:
        SeedSourceError: A fact's predicate is not a recognized ontology
            relationship type, a node's type is not a recognized ontology
            node type, the same node id is assigned to two different
            partitions by different documents, the same node id is defined
            more than once, or a fact references an undefined node id.
            Raised before the wipe runs. Also raised before the wipe for a
            node carrying an extraction stamp or a label its MERGE would
            interpolate that is not a plain identifier.
        SeedTargetNotSeedOnlyError: The target graph holds an element the
            seed guard counts. Raised before the wipe runs.
        Neo4jQueryError: A seed guard statement failed or returned no
            usable row (fail closed). Raised before the wipe runs.
    """
    # BEFORE the wipe, and independently of the delegate's own guard: by the
    # time `apply_seed_documents` refused, `wipe_seed_version` would already
    # have emptied the graph. This is the 2026-07-31 loss path.
    _assert_seed_target_permitted(connection, allow_live=allow_live, action="re-seeding")
    _validate_predicates(documents)
    _validate_node_types(documents)
    _validate_no_extraction_stamps(documents)
    _build_node_queries(_assign_node_partitions(documents), _collect_node_definitions(documents))
    # MIS-177 D1: BEFORE the wipe. The delegate's own check runs after the
    # wipe, on a graph the wipe has already changed; a refusal there would
    # leave the seed content deleted.
    _assert_seed_target_holds_only_seed(connection, action="re-seeding")
    wipe_seed_version(connection, seed_version)
    return apply_seed_documents(
        connection,
        documents,
        seed_version=seed_version,
        now_iso=now_iso,
        allow_live=allow_live,
    )


def _count(results: list[dict]) -> int:
    """Extract the `n` count from a `RETURN count(...) AS n` result.

    `FakeNeo4jConnection.execute_write` returns an empty list unless a test
    pre-configures `write_results`, which real Neo4j never does for an
    aggregation query -- `count()` always yields exactly one row, even over
    zero matches. Guarding the empty case keeps unit tests that are not
    exercising this return value from raising `IndexError`.
    """
    if not results:
        return 0
    return int(results[0]["n"])


def _assign_node_partitions(documents: list[SeedDocument]) -> dict[str, str]:
    """Map every subject/object id referenced in `documents` to its partition.

    A document's `partition` applies to every subject and object its facts
    reference. `SeedDocument.partition` is `Literal`-typed against the
    graph's two valid partition labels, so a single document can never
    carry an invalid one -- what this function additionally catches is a
    node id claimed by two DIFFERENT documents under different partitions,
    which no single document's type validation can see. That case is a
    genuine authoring conflict (the same id cannot mean two different
    partitioned things), not a typo class covered elsewhere.

    Args:
        documents: Parsed seed documents to map.

    Returns:
        Every referenced node id mapped to the partition label
        (`ENTITY_LABEL` or `SELF_MODEL_LABEL`) it belongs to.

    Raises:
        SeedSourceError: The same node id is assigned different partitions
            by different documents.
    """
    partitions: dict[str, str] = {}
    for doc in documents:
        for fact in doc.facts:
            for node_id in (fact.subject, fact.object):
                claimed = partitions.get(node_id)
                if claimed is not None and claimed != doc.partition:
                    raise SeedSourceError(
                        f"{doc.source_path}: {node_id!r} is claimed by partition "
                        f"{claimed!r} elsewhere in the seed source and "
                        f"{doc.partition!r} here -- a node cannot live in two "
                        "graph partitions"
                    )
                partitions[node_id] = doc.partition
    return partitions


def _collect_node_definitions(documents: list[SeedDocument]) -> dict[str, SeedNode]:
    """Map every defined node id to its `SeedNode`.

    Task 11's loader already rejects a duplicate node id at load time
    (`_validate_unique_node_ids`); this is the applier's own defense, the
    same posture `_validate_node_types` takes for `type` -- a caller that
    constructs `SeedDocument`s directly, bypassing `load_seed_documents`,
    is not protected by a loader-only check.

    Args:
        documents: Parsed seed documents to collect from.

    Returns:
        Every defined node id mapped to its `SeedNode`.

    Raises:
        SeedSourceError: The same node id is defined more than once, within
            or across documents.
    """
    definitions: dict[str, SeedNode] = {}
    defined_in: dict[str, Path] = {}
    for doc in documents:
        for node in doc.nodes:
            first_seen = defined_in.get(node.id)
            if first_seen is not None:
                raise SeedSourceError(
                    f"{doc.source_path}: node id {node.id!r} is already defined in "
                    f"{first_seen} -- node ids must be unique across the whole seed source"
                )
            definitions[node.id] = node
            defined_in[node.id] = doc.source_path
    return definitions


def _validate_node_types(documents: list[SeedDocument]) -> None:
    """Reject any node whose `type` is not a known ontology node type.

    Neo4j cannot parameterize a label, so `apply_seed_documents` interpolates
    `node.type` directly into the Cypher string (`_merge_node_query`, for a
    `:__SelfModel__` or `User` node). That interpolation point is where this
    check belongs --
    Task 11's loader-level `_validate_node_types` (same name, different
    module) already rejects an unknown type at load time, but mirrors
    `_validate_predicates`'s reasoning below: a loader check does not
    protect a caller that constructs `SeedDocument`s directly, so the
    injection boundary needs its own guard regardless.

    Args:
        documents: Parsed seed documents to validate.

    Raises:
        SeedSourceError: A node's `type` is not in `ALL_NODE_TYPE_NAMES`,
            naming the type, the node id, the source file, and the closest
            allowed type if there is an obvious near-match.
    """
    allowed = set(ALL_NODE_TYPE_NAMES)
    for doc in documents:
        for node in doc.nodes:
            if node.type in allowed:
                continue
            suggestion = difflib.get_close_matches(node.type, ALL_NODE_TYPE_NAMES, n=1)
            hint = f" Closest allowed type: {suggestion[0]!r}." if suggestion else ""
            raise SeedSourceError(
                f"{doc.source_path}: node {node.id!r} has unknown type {node.type!r}, "
                f"not a recognized ontology node type.{hint}"
            )


def _validate_no_extraction_stamps(documents: list[SeedDocument]) -> None:
    """Reject any node whose properties carry an extraction stamp.

    `SeedNode._no_applier_owned_extras` (models.py) refuses these names when a
    node is built; this is the applier's independent second layer, for a
    `SeedNode` that skipped that validator (`model_construct`). Ordering the
    spread cannot neutralise a stamp the way it does an applier-owned key,
    because the applier writes no stamp of its own for it to lose to: an
    authored one would reach the graph through `n += $properties`. Which guard
    then refuses the seed's own node is stated above
    `_EXTRACTION_STAMP_NODE_PROPERTIES` in models.py.

    Checks every name in `admin._EXTRACTION_STAMP_PROPERTIES` against the keys
    `apply_seed_documents` spreads (`node.model_dump()`).

    Raises:
        SeedSourceError: A node carries `ontology_version`,
            `extraction_version` or `model_hash`.
    """
    stamps = frozenset(_EXTRACTION_STAMP_PROPERTIES)
    for doc in documents:
        for node in doc.nodes:
            found = stamps & node.model_dump().keys()
            if found:
                raise SeedSourceError(
                    f"{doc.source_path}: node {node.id!r} carries extraction stamp(s) "
                    f"{sorted(found)}; a seed node must not carry one, or the seed "
                    "guard or graph-reset guard can refuse the graph this seed wrote "
                    "(MIS-177)"
                )


def _build_node_queries(
    node_partitions: dict[str, str], node_definitions: dict[str, SeedNode]
) -> dict[str, str]:
    """Build the MERGE for every node the applier will write, before any write.

    `_merge_node_query` interpolates labels (SET and REMOVE) and refuses one
    that is not a plain identifier; building every query here first means that
    refusal happens before the first write, and in `reseed` before the wipe,
    instead of after some nodes are already written. A node id with no
    definition is skipped; the write loop reports it.

    Returns:
        Each defined, fact-referenced node id mapped to its MERGE statement.

    Raises:
        SeedSourceError: A label a node's MERGE would interpolate is not a
            plain Cypher identifier.
    """
    queries: dict[str, str] = {}
    for node_id in sorted(node_partitions):
        node = node_definitions.get(node_id)
        if node is None:
            continue
        try:
            queries[node_id] = _merge_node_query(node_partitions[node_id], node.type)
        except ValueError as exc:
            raise SeedSourceError(f"node {node_id!r}: {exc}") from exc
    return queries


def _validate_predicates(documents: list[SeedDocument]) -> None:
    """Reject any fact whose predicate is not a known ontology relationship type.

    Neo4j cannot parameterize a relationship type, so `apply_seed_documents`
    interpolates `fact.predicate` directly into the Cypher string (`_MERGE_EDGE
    % fact.predicate`). That interpolation point is where this check belongs --
    not at YAML-read time in the loader, which would duplicate the check while
    leaving the actual injection boundary unguarded. Runs over every document
    before any `execute_write` call, so one bad predicate anywhere aborts the
    whole application rather than leaving a partial write.

    Args:
        documents: Parsed seed documents to validate.

    Raises:
        SeedSourceError: A fact uses a predicate outside `ALL_EDGE_TYPE_NAMES`,
            naming the predicate, the source file, and the closest allowed
            predicate if there is an obvious near-match.
    """
    allowed = set(ALL_EDGE_TYPE_NAMES)
    for doc in documents:
        for fact in doc.facts:
            if fact.predicate in allowed:
                continue
            suggestion = difflib.get_close_matches(fact.predicate, ALL_EDGE_TYPE_NAMES, n=1)
            hint = f" Closest allowed predicate: {suggestion[0]!r}." if suggestion else ""
            raise SeedSourceError(
                f"{doc.source_path}: unknown predicate {fact.predicate!r} is not a "
                f"recognized ontology relationship type.{hint}"
            )
