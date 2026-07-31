"""Apply the versioned seed source to the graph, deterministically.

No LLM is involved: seed facts are authored, so they are written as given
rather than inferred from prose (R1.4 spec 2.0). Every node and edge carries
`seed_version`, which is what makes the wipe applied elsewhere for a given
version exact -- a node or edge written without the stamp is un-wipeable and
becomes permanent graph litter that no gate can detect.
"""

import difflib
import logging

from backend.errors import SeedSourceError
from backend.interfaces import GraphConnection
from backend.knowledge.ontologies.v1_0_0 import ALL_EDGE_TYPE_NAMES

from .models import SeedDocument

logger = logging.getLogger(__name__)

_MERGE_NODE = (
    "MERGE (n:__Entity__ {id: $id}) "
    "ON CREATE SET n.created_at = $now "
    "SET n.seed_version = $seed_version, n.updated_at = $now "
    "RETURN n.id AS id"
)

_MERGE_EDGE = (
    "MATCH (s:__Entity__ {id: $subject}) "
    "MATCH (o:__Entity__ {id: $object}) "
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
            relationship type.
    """
    _validate_predicates(documents)

    subjects_and_objects: set[str] = set()
    for doc in documents:
        for fact in doc.facts:
            subjects_and_objects.add(fact.subject)
            subjects_and_objects.add(fact.object)

    for node_id in sorted(subjects_and_objects):
        connection.execute_write(
            _MERGE_NODE, {"id": node_id, "seed_version": seed_version, "now": now_iso}
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
        len(subjects_and_objects),
        fact_count,
        seed_version,
    )
    return {"nodes": len(subjects_and_objects), "facts": fact_count}


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

    Predicates are validated before the wipe runs, not just before the
    re-apply's writes (`apply_seed_documents` already guards that point).
    Without this, a typo introduced in a source edit would empty a
    previously-good graph via the wipe and then abort the re-apply,
    leaving a real data-loss window open until the typo is fixed.

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
            relationship type. Raised before the wipe runs.
    """
    _validate_predicates(documents)
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
