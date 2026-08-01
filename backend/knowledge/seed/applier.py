"""Apply the versioned seed source to the graph, deterministically.

No LLM is involved: seed facts are authored, so they are written as given
rather than inferred from prose (R1.4 spec 2.0). Every node and edge carries
`seed_version`, which is what makes the wipe applied elsewhere for a given
version exact -- a node or edge written without the stamp is un-wipeable and
becomes permanent graph litter that no gate can detect.

R1.4 Task 12 (addendum): every node also gets its ontology type label and
every descriptive property the source defines (`entity_type` plus whatever
`SeedNode` carries beyond `id`/`type`). Before this task, `_MERGE_NODE` set
only `seed_version`/`created_at`/`updated_at` -- Task 10's live run proved
that a wipe-and-recreate cycle then leaves a node with no ontology label and
no descriptive properties at all, since `MERGE` preserves untouched
properties on a MATCH but a fresh CREATE gets nothing beyond what the query
explicitly sets. See the Task 10 report for the live consequence.
"""

import difflib
import logging
from pathlib import Path

from backend.errors import SeedSourceError
from backend.interfaces import GraphConnection
from backend.knowledge.ontologies import ALL_NODE_TYPE_NAMES
from backend.knowledge.ontologies.v1_0_0 import ALL_EDGE_TYPE_NAMES
from backend.knowledge.storage.partitions import ENTITY_LABEL, SELF_MODEL_LABEL

from .models import SeedDocument, SeedNode

logger = logging.getLogger(__name__)

# Partition label (`%s` #1) and ontology type label (`%s` #2) are both
# interpolated, never fixed constants. Partition: the graph has two
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
# exact shape: `MERGE (n:{partition} {id: $id}) ... SET n:{label}`).
_MERGE_NODE = (
    "MERGE (n:%s {id: $id}) "
    "ON CREATE SET n.created_at = $now, n += $properties "
    "ON MATCH SET n += $properties "
    "SET n:%s "
    "RETURN n.id AS id"
)

# The label union (`:A|B`) matches a node in EITHER partition -- this MATCH
# must find self-model nodes as readily as entity nodes, since a fact's
# subject/object may resolve to either. Mirrors the existing production
# precedent at backend/knowledge/admin.py's edge-merge helper, which solves
# the identical two-partition matching problem for the older seed_data.yaml
# path.
_MERGE_EDGE = (
    f"MATCH (s:{ENTITY_LABEL}|{SELF_MODEL_LABEL} {{id: $subject}}) "
    f"MATCH (o:{ENTITY_LABEL}|{SELF_MODEL_LABEL} {{id: $object}}) "
    "MERGE (s)-[r:%s]->(o) "
    "SET r.seed_version = $seed_version, r.valid_from = $valid_from, "
    "    r.valid_to = $valid_to, r.updated_at = $now "
    "RETURN type(r) AS t"
)

_WIPE_EDGES = "MATCH ()-[r]->() WHERE r.seed_version = $seed_version DELETE r RETURN count(r) AS n"
_WIPE_NODES = (
    "MATCH (n) WHERE n.seed_version = $seed_version "
    "AND NOT (n)--() "
    "DELETE n RETURN count(n) AS n"
)


def apply_seed_documents(
    connection: GraphConnection,
    documents: list[SeedDocument],
    *,
    seed_version: str,
    now_iso: str,
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
    `SeedNode` definition: ontology type label, `entity_type` property, and
    every other descriptive property the source defines (R1.4 Task 12).

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
            a loader-only check).
    """
    _validate_predicates(documents)
    _validate_node_types(documents)
    node_partitions = _assign_node_partitions(documents)
    node_definitions = _collect_node_definitions(documents)

    for node_id in sorted(node_partitions):
        node = node_definitions.get(node_id)
        if node is None:
            raise SeedSourceError(
                f"fact references node id {node_id!r}, which has no matching "
                "`SeedNode` definition -- every fact's subject and object must "
                "have a node definition (R1.4 Task 11/12)"
            )
        properties = {
            "entity_type": node.type,
            "seed_version": seed_version,
            "updated_at": now_iso,
            **{k: v for k, v in node.model_dump().items() if k not in ("id", "type")},
        }
        connection.execute_write(
            _MERGE_NODE % (node_partitions[node_id], node.type),
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

    Returns:
        Counts keyed `nodes` and `facts`, from the re-apply.

    Raises:
        SeedSourceError: A fact's predicate is not a recognized ontology
            relationship type, a node's type is not a recognized ontology
            node type, the same node id is assigned to two different
            partitions by different documents, the same node id is defined
            more than once, or a fact references an undefined node id.
            Raised before the wipe runs.
    """
    _validate_predicates(documents)
    _validate_node_types(documents)
    _assign_node_partitions(documents)
    _collect_node_definitions(documents)
    wipe_seed_version(connection, seed_version)
    return apply_seed_documents(connection, documents, seed_version=seed_version, now_iso=now_iso)


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
    `node.type` directly into the Cypher string (`_MERGE_NODE % (partition,
    node.type)`). That interpolation point is where this check belongs --
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
