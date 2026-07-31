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
