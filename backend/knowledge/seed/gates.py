"""Verification gates for the versioned seed source.

Four independent gates check different failure classes after a seed
source is loaded (R1.4 spec section 5, `check_node_definitions` added by
the Task 11-14 addendum):

- `check_facts_present` -- the graph actually holds what the source says
  it should. This is the gate that cannot be satisfied vacuously: it
  compares the live graph against the authored source, not against
  another copy of the same rebuild. An equality-between-two-runs check
  would hold just as well if both sides were empty, which is exactly how
  this sub-project lost 32 nodes / 30 relationships with zero provenance
  in the first place.
- `check_node_definitions` -- every seeded NODE (not just every fact)
  carries its ontology type label and a display name in the live graph.
  This is the gate Task 10's live defect needed and did not have: Gate 2
  checks that authored facts are present, and the wipe-and-recreate cycle
  that stripped every node's ontology label and descriptive property left
  the edges intact (MERGE recreated them from the source's facts), so
  Gate 2 passed on a graph that had lost everything else.
- `check_containment` -- the prose and the frontmatter facts agree on
  which entities they mention.
- `check_negation_proximity` -- the prose does not obviously contradict
  a fact near where that fact's object is mentioned.

None of these alone is sufficient, and none of them together proves
semantic agreement between prose and facts -- see each gate's own
docstring for exactly what it does and does not catch. The real
backstops are that the seed is small and human-reviewed, and that
bitemporal `valid_to` (already on `SeedFact`) gives semantic change a
structured home so inversions rarely become prose-only edits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from backend.interfaces import GraphConnection
from backend.knowledge.seed.models import SeedDocument
from backend.knowledge.storage.partitions import ENTITY_LABEL, SELF_MODEL_LABEL

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GateResult:
    """Outcome of one verification gate.

    `passed` is `True` exactly when `failures` is empty. `failures`
    carries one human-readable line per problem found, naming the source
    document and the specific fact involved -- see each `check_*`
    function for the exact message shape.
    """

    passed: bool
    failures: list[str]


# The subject/object MATCH clauses use the label union
# `:{ENTITY_LABEL}|{SELF_MODEL_LABEL}` because the graph has two
# id-scoped partitions with label-scoped uniqueness constraints that
# cannot see each other -- a MATCH restricted to `:__Entity__` alone
# returns no bind for a `:__SelfModel__` node such as `mist-identity`,
# which would make every self-model fact report as missing. Verified
# live against Neo4j during Task 9 (see the report).
_CHECK_FACT_QUERY = (
    f"MATCH (s:{ENTITY_LABEL}|{SELF_MODEL_LABEL} {{id: $subject}}) "
    f"MATCH (o:{ENTITY_LABEL}|{SELF_MODEL_LABEL} {{id: $object}}) "
    "MATCH (s)-[r:%s]->(o) "
    "WHERE r.seed_version = $seed_version "
    "RETURN count(r) AS n"
)


def check_facts_present(
    connection: GraphConnection,
    documents: list[SeedDocument],
    *,
    seed_version: str,
) -> GateResult:
    """Verify every fact in `documents` exists in the graph at `seed_version`.

    Queries the live graph once per fact -- deliberately not a comparison
    between two in-memory representations of the same rebuild, which
    would pass even if both sides were empty. The authored source is
    ground truth; the graph is checked against it, not the other way
    round.

    Read-only: issues `execute_query` only, never `execute_write`.

    Args:
        connection: Sync graph connection.
        documents: Parsed seed documents to check.
        seed_version: The version stamp every present fact must carry.

    Returns:
        `GateResult` with one failure line per fact not found in the
        graph, naming the source document and the subject/predicate/
        object that is missing.
    """
    failures: list[str] = []
    for doc in documents:
        for fact in doc.facts:
            query = _CHECK_FACT_QUERY % fact.predicate
            results = connection.execute_query(
                query,
                {
                    "subject": fact.subject,
                    "object": fact.object,
                    "seed_version": seed_version,
                },
            )
            if _count(results) < 1:
                failures.append(
                    f"{doc.source_path}: missing fact {fact.subject} {fact.predicate} "
                    f"{fact.object} at seed_version={seed_version!r}"
                )
    return GateResult(passed=not failures, failures=failures)


# One MATCH clause carrying BOTH the document's partition label and the
# node's ontology type label (`:{partition}:{type}`) -- a node missing
# EITHER label fails to bind and is reported, rather than needing two
# separate checks. This is deliberately the same interpolate-and-pin
# discipline as `_CHECK_FACT_QUERY`'s predicate: Neo4j cannot parameterize
# a label, and a query that only checked `display_name IS NOT NULL`
# without also re-asserting the labels in the MATCH pattern would still
# pass on a node holding the right property under the wrong label.
_CHECK_NODE_QUERY = (
    "MATCH (n:%s:%s {id: $id}) "
    "WHERE n.seed_version = $seed_version AND n.display_name IS NOT NULL "
    "RETURN count(n) AS n"
)


def check_node_definitions(
    connection: GraphConnection,
    documents: list[SeedDocument],
    *,
    seed_version: str,
) -> GateResult:
    """Verify every seeded node carries its ontology type label and a display name.

    This is the gate R1.4 Task 10's live defect needed and did not have.
    `reseed()`'s wipe-then-apply cycle stripped every node's ontology type
    label (`MistIdentity`, `MistTrait`, `User`, `Organization`, ...) and
    every descriptive property (including `display_name`) down to a bare
    partition label plus `id`/timestamps/`seed_version` -- and
    `check_facts_present` (Gate 2) passed throughout, because the edges
    those facts describe were recreated correctly from the source; only
    the NODES lost their identity. This gate checks the nodes directly:
    for every `SeedNode` the source defines, the live graph node must
    match on both its document's partition label and its ontology type
    label in one MATCH clause (a node missing either fails to bind) and
    carry a non-null `display_name`.

    Read-only: issues `execute_query` only, never `execute_write`.

    Args:
        connection: Sync graph connection.
        documents: Parsed seed documents to check.
        seed_version: The version stamp every present node must carry.

    Returns:
        `GateResult` with one failure line per node whose live graph
        counterpart is missing its partition label, its ontology type
        label, or a non-null `display_name`.
    """
    failures: list[str] = []
    for doc in documents:
        for node in doc.nodes:
            query = _CHECK_NODE_QUERY % (doc.partition, node.type)
            results = connection.execute_query(
                query,
                {"id": node.id, "seed_version": seed_version},
            )
            if _count(results) < 1:
                failures.append(
                    f"{doc.source_path}: node {node.id!r} is missing its "
                    f"{doc.partition!r} partition label, its {node.type!r} ontology "
                    f"type label, or a non-null display_name at "
                    f"seed_version={seed_version!r}"
                )
    return GateResult(passed=not failures, failures=failures)


def check_containment(documents: list[SeedDocument]) -> GateResult:
    """Verify every fact's object is mentioned by display name in its document body.

    R1.4 Task 14: matches on the object node's `SeedNode.display_name`
    (Task 11), not the raw `fact.object` id. The original Task 9
    implementation checked the raw id as a literal substring, which is
    structurally unable to pass against real prose -- `fact.object` is a
    kebab id (`trait-transparent`); the prose describes it by display
    name (`**Transparent**`), a string the id never equals. 29 of 30 real
    facts failed under that check. A prefix-strip/hyphen-collapse
    normalization was scoped as the fix during Task 10 but never
    implemented; by the time this landed, Task 11 had given every node an
    exact `display_name`, which is strictly better than a heuristic
    reconstruction of one -- use it directly instead.

    Falls back to the raw `fact.object` id when the object has no
    matching `SeedNode` (referential integrity is `load_seed_documents`'s
    job, not this gate's -- see Task 11) or the node defines no
    `display_name`, so a fact is never silently skipped over.

    Case-insensitive: `slalom` must find `Slalom`.

    Does NOT prove semantic agreement. It proves the prose mentions the
    same entities the frontmatter asserts. Semantic inversion is the job
    of `check_negation_proximity` (partial) and the advisory extraction
    audit (spec 5.3); neither is complete, and the real backstops are
    that the seed is small and human-reviewed, and that bitemporal
    `valid_to` gives semantic change a structured home.

    Args:
        documents: Parsed seed documents to check.

    Returns:
        `GateResult` with one failure line per fact whose object's display
        name (or raw id, if undefined) does not appear in its own
        document's body.
    """
    node_by_id = {node.id: node for doc in documents for node in doc.nodes}
    failures: list[str] = []
    for doc in documents:
        body_lower = doc.body.lower()
        for fact in doc.facts:
            node = node_by_id.get(fact.object)
            display_name = getattr(node, "display_name", None) if node is not None else None
            search_term = display_name or fact.object
            if search_term.lower() not in body_lower:
                failures.append(
                    f"{doc.source_path}: fact object {fact.object!r} "
                    f"(searched for {search_term!r}) "
                    f"({fact.subject} {fact.predicate} {fact.object}) not found in "
                    "document body"
                )
    return GateResult(passed=not failures, failures=failures)


# Marker phrases that, near a fact's object, suggest the prose may have
# inverted or retired that fact rather than asserting it. Deliberately a
# flat set of literal substrings, not a parser: `"ex-"` and `"left"` are
# short enough to false-positive on unrelated words ("flex-time", "left
# side of the diagram"). The gate is biased toward over-flagging rather
# than missing a real inversion -- a false positive costs a human a
# second look; a false negative ships a fact the prose contradicts.
_NEGATION_MARKERS = {
    "no longer",
    "former",
    "formerly",
    "ex-",
    "left",
    "used to",
    "previously",
}

# Characters scanned on either side of a fact object's occurrence for a
# negation marker. Widening this trades precision for recall -- a wider
# window catches negations phrased further from the object at the cost
# of more false positives from unrelated markers elsewhere in the
# sentence. 60 was chosen to comfortably span one sentence clause without
# reaching into neighboring sentences.
_PROXIMITY_WINDOW = 60


def check_negation_proximity(documents: list[SeedDocument]) -> GateResult:
    """Flag a fact whose object occurs near a negation marker in the body.

    For every occurrence of a fact's object in its document body, scans
    `_PROXIMITY_WINDOW` characters on either side (case-insensitive) for
    one of `_NEGATION_MARKERS`. A marker inside that window flags the
    fact; a marker elsewhere in the document does not -- an unrelated
    negation about a different fact must not fail this one.

    Partial, like `check_containment`: proximity is not parsing, so this
    cannot tell "Raj no longer works at Slalom" (a real inversion) apart
    from an unrelated marker that happens to land in the window by
    coincidence. See `check_containment`'s docstring for the full
    limitation statement and the actual backstops.

    Args:
        documents: Parsed seed documents to check.

    Returns:
        `GateResult` with one failure line per fact with a negation
        marker near an occurrence of its object.
    """
    failures: list[str] = []
    for doc in documents:
        body_lower = doc.body.lower()
        for fact in doc.facts:
            object_lower = fact.object.lower()
            if not object_lower:
                continue
            for start in _find_all(body_lower, object_lower):
                end = start + len(object_lower)
                window = body_lower[max(0, start - _PROXIMITY_WINDOW) : end + _PROXIMITY_WINDOW]
                marker = next((m for m in _NEGATION_MARKERS if m in window), None)
                if marker is not None:
                    failures.append(
                        f"{doc.source_path}: possible negation {marker!r} near fact object "
                        f"{fact.object!r} ({fact.subject} {fact.predicate} {fact.object})"
                    )
                    break
    return GateResult(passed=not failures, failures=failures)


def _find_all(haystack: str, needle: str) -> list[int]:
    """Return every start index of `needle` in `haystack`, left to right."""
    if not needle:
        return []
    indices = []
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            break
        indices.append(idx)
        start = idx + 1
    return indices


def _count(results: list[dict]) -> int:
    """Extract the `n` count from a `RETURN count(...) AS n` result.

    Mirrors `applier._count`: `FakeNeo4jConnection.execute_query` returns
    an empty list unless a test pre-configures `query_results`, which
    real Neo4j never does for an aggregation query -- `count()` always
    yields exactly one row, even over zero matches. Guarding the empty
    case keeps unit tests that leave `query_results` unset from raising
    `IndexError` instead of exercising the intended "fact missing" path.
    """
    if not results:
        return 0
    return int(results[0]["n"])
