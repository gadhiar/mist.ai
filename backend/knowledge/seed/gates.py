"""Verification gates for the versioned seed source.

Five independent gates check different failure classes after a seed
source is loaded (R1.4 spec section 5, `check_node_definitions` added by
the Task 11-14 addendum, `check_embeddings` by I7):

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
  a fact near where that fact's object is mentioned. Shares
  `_search_term_for` with `check_containment` (fixed together as C1 of
  the R1.4 whole-branch review, after Task 14 fixed containment's half of
  this and left negation-proximity searching the raw kebab id -- which
  real prose never contains, so the gate passed having scanned nothing).
- `check_embeddings` -- every seeded node carries a vector that is
  present, correctly shaped, non-zero, and computed from the text the
  source currently authors. This is the only gate that can see anything
  about embeddings at all: `canonical_serialize` excludes `embedding`
  from the canonical form, so `assert_rebuild_twice_identical` and
  `live_vs_rebuilt_report` are byte-identical whether embeddings are
  present, absent, or all-zero -- the blindness is structural, not an
  oversight. Two live losses had already happened before this gate
  existed, both invisible to all four gates above.

None of these alone is sufficient, and none of them together proves
semantic agreement between prose and facts -- see each gate's own
docstring for exactly what it does and does not catch. The real
backstops are that the seed is small and human-reviewed, and that
bitemporal `valid_to` (already on `SeedFact`) gives semantic change a
structured home so inversions rarely become prose-only edits.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from backend.interfaces import EmbeddingProvider, GraphConnection
from backend.knowledge.embeddings.embedding_text import embedding_text_for
from backend.knowledge.seed.models import SeedDocument, SeedNode
from backend.knowledge.storage.partitions import ENTITY_LABEL, SELF_MODEL_LABEL

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GateResult:
    """Outcome of one verification gate.

    `passed` is `True` exactly when `failures` is empty. `failures`
    carries one human-readable line per problem found, naming the source
    document and the specific fact involved -- see each `check_*`
    function for the exact message shape.

    `examined` is how many units of work the gate actually inspected, so
    that "did it look at anything?" is a query rather than a hope. R1.4's
    C1: `check_negation_proximity` reported `passed=True` having examined
    0 of 20 self-model facts, and nothing in the result could say so --
    the pass and the vacuous pass were indistinguishable.

    It defaults to 0 and ONLY `check_embeddings` populates it. Reading
    `examined` off any of the other four gates therefore tells you
    nothing: `check_containment(...).examined == 0` means "this gate does
    not report a count", not "this gate examined nothing". The default is
    deliberate -- it leaves the four pre-existing gates and every one of
    their tests untouched -- but it is a footgun for anyone who later
    treats `examined == 0` as a universal vacuity check, so populate it
    in any gate that grows one rather than inferring it here.
    """

    passed: bool
    failures: list[str]
    examined: int = 0


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


# Returns the vector and NOTHING else. `check_embeddings` must not be
# able to read the graph's own `display_name`/`description` even by
# accident: it recomputes from the authored seed source, and a gate that
# recomputed from the graph's properties would agree with itself by
# construction whenever the applier and the backfill were both wrong in
# the same way. Stale text is precisely the case where the graph's
# properties are RIGHT and only its vector is old, so reading them back
# would turn the comparison into a tautology. Withholding the columns is
# stronger than declining to use them -- see Gate 2's docstring for the
# same "the authored source is ground truth" discipline.
_CHECK_EMBEDDING_QUERY = (
    "MATCH (n:%s {id: $id}) "
    "WHERE n.seed_version = $seed_version "
    "RETURN n.embedding AS embedding"
)

# Below this L2 norm a stored vector is treated as the zero vector.
# `EmbeddingGenerator.generate_embedding` returns `[0.0] * 384` for empty
# or whitespace-only text -- right width, not null, and matching nothing
# at query time, so only a norm check can see it. Not exact `== 0.0`:
# float round-tripping through Neo4j and back is exact for zero today,
# but an epsilon costs nothing and a near-zero vector is just as useless.
_MIN_L2_NORM = 1e-6

# Cosine floor for "this vector was computed from this text".
# sentence-transformers is deterministic in eval mode, so the honest
# comparison is equality -- but pinning bit-identity across processes,
# library versions and hardware would make the gate brittle without
# detecting anything more. Every real failure mode this catches (stale
# text, a cross-wired vector, a different model) moves cosine far below
# this, not marginally: two unrelated MiniLM sentence vectors sit around
# 0.0-0.3, and even a one-word edit moves well past 0.001.
_MIN_COSINE = 0.999


def check_embeddings(
    connection: GraphConnection,
    documents: list[SeedDocument],
    *,
    seed_version: str,
    embedding_generator: EmbeddingProvider,
    expected_dimension: int,
) -> GateResult:
    """Verify every seeded node's stored vector matches its authored source text.

    The gate the two live embedding losses needed and did not have. It is
    the only check in the codebase that can see anything about embeddings
    at all: `canonical_serialize` excludes `embedding` from the canonical
    form, so every determinism and equality check in this sub-project is
    byte-identical whether embeddings are present, absent, or all-zero.

    Four conditions, in order, at most one failure line per node (a
    dimension mismatch makes the cosine comparison meaningless, so later
    conditions do not run once an earlier one fires):

    1. The node is absent from the graph, or its `embedding` is null.
       Reached by `--no-embeddings`, by a backfill that raised after the
       graph writes had already committed, and by a post-wipe recreate.
    2. The stored vector is not `expected_dimension` wide. Reached by an
       `EMBEDDING_DIMENSION` change without a reindex.
    3. Its L2 norm is essentially zero -- `generate_embedding`'s
       empty-text branch.
    4. Its cosine against a vector recomputed from the AUTHORED source
       node falls below `_MIN_COSINE`.

    Condition 4 is why this gate exists. `_WIPE_NODES` only deletes
    seed-stamped nodes matching `NOT (n)--()`, so a seeded node that has
    acquired a conversation-derived edge survives the wipe with its old
    vector intact; the applier's `ON MATCH SET n += $properties` then
    refreshes its `display_name` and `description` but not its
    `embedding`, and `_backfill_embeddings_for_seed` skips it because its
    `WHERE n.embedding IS NULL` guard is false. The node keeps,
    indefinitely, a vector computed from text that is no longer authored
    anywhere -- present, correctly shaped, unit-norm, and wrong. No
    count-based or presence-based check can ever see that.

    Recomputation goes through `embedding_text_for`, the same builder
    both production backfills use (I7 Task 1), so the gate compares
    against what the backfill would produce today rather than against a
    third private copy of the join -- which is the shape C1 came in
    through.

    Fails closed: a run that examined zero nodes returns `passed=False`.
    `check_negation_proximity` returning `passed=True` having examined 0
    of 20 facts is a defect that already shipped once in this exact
    module; a gate that examined nothing has proven nothing.

    Read-only: issues `execute_query` only, never `execute_write`.

    Args:
        connection: Sync graph connection.
        documents: Parsed seed documents to check.
        seed_version: The version stamp every checked node must carry.
        embedding_generator: Provider used to recompute each node's
            vector from its authored text. Injected rather than
            constructed here (no hidden construction in a callee either),
            and it MUST be the same model the stored vectors were written
            with -- a different model fails condition 4 for every node,
            which is a true report of a real mismatch, not a false alarm.
        expected_dimension: Width every stored vector must have.

    Returns:
        `GateResult` with one failure line per problem node, naming the
        source document and the node id, and `examined` set to the number
        of nodes inspected.
    """
    failures: list[str] = []
    examined = 0
    for doc in documents:
        query = _CHECK_EMBEDDING_QUERY % doc.partition
        for node in doc.nodes:
            examined += 1
            rows = connection.execute_query(
                query,
                {"id": node.id, "seed_version": seed_version},
            )
            failure = _embedding_failure(
                doc=doc,
                node=node,
                rows=rows,
                seed_version=seed_version,
                embedding_generator=embedding_generator,
                expected_dimension=expected_dimension,
            )
            if failure is not None:
                failures.append(failure)
    if examined == 0:
        failures.append(
            f"check_embeddings examined 0 nodes across {len(documents)} document(s) "
            "-- refusing to report a pass on an empty examination set (a gate that "
            "examined nothing has proven nothing; see check_negation_proximity, "
            "which shipped reporting passed=True having examined 0 of 20 facts)"
        )
    return GateResult(passed=not failures, failures=failures, examined=examined)


def _embedding_failure(
    *,
    doc: SeedDocument,
    node: SeedNode,
    rows: list[dict],
    seed_version: str,
    embedding_generator: EmbeddingProvider,
    expected_dimension: int,
) -> str | None:
    """Return the first problem with one node's stored vector, or None.

    Split out of `check_embeddings` so the four conditions read as an
    ordered list rather than as nesting, and so the early-return
    discipline (a later condition is meaningless once an earlier one
    fires) is structural rather than a `continue` a future edit can drop.
    """
    prefix = f"{doc.source_path}: node {node.id!r}"
    if not rows:
        return (
            f"{prefix} has no node in the graph under its {doc.partition!r} "
            f"partition label at seed_version={seed_version!r}"
        )
    stored = rows[0].get("embedding")
    if stored is None:
        return f"{prefix} has a null embedding at seed_version={seed_version!r}"
    if not isinstance(stored, list):
        return (
            f"{prefix} has an embedding of type {type(stored).__name__}, expected a "
            "list of floats"
        )
    if len(stored) != expected_dimension:
        return (
            f"{prefix} has an embedding of {len(stored)} dimensions, expected "
            f"{expected_dimension}"
        )
    # Element type, not just container type. `_l2_norm` calls `float(x)` per
    # element, so a correctly-shaped list of non-numeric values reaches it and
    # raises ValueError -- a traceback out of `seed-verify` / `cmd_seed` instead
    # of the clean per-node failure line every other condition here produces.
    # Reachable in exactly one way, which is why this is a condition and not a
    # comment: `SeedNode` is `extra="allow"` and `embedding` is not applier-owned,
    # so an authored `embedding:` key in `mist-memory/seed/*.md` flows straight
    # through `$properties` to the graph. The Neo4j driver itself always returns
    # lists of floats for array properties, so the driver path cannot produce it.
    bad = next((x for x in stored if not isinstance(x, int | float) or isinstance(x, bool)), None)
    if bad is not None:
        return (
            f"{prefix} has a non-numeric value of type {type(bad).__name__} in its "
            "embedding, expected a list of floats"
        )
    norm = _l2_norm(stored)
    if norm < _MIN_L2_NORM:
        return (
            f"{prefix} has a zero vector as its embedding (L2 norm {norm:.3g}) -- "
            "the empty-text branch of generate_embedding, which matches nothing "
            "at query time"
        )
    text = embedding_text_for(
        getattr(node, "display_name", None), getattr(node, "description", None), node.id
    )
    recomputed = embedding_generator.generate_embedding(text)
    if len(recomputed) != len(stored):
        return (
            f"{prefix} recomputed to {len(recomputed)} dimensions but the stored "
            f"embedding has {len(stored)} -- the injected embedding_generator "
            f"disagrees with expected_dimension={expected_dimension}"
        )
    similarity = _cosine(stored, recomputed)
    if similarity < _MIN_COSINE:
        return (
            f"{prefix} has an embedding that does not match its authored source text "
            f"(cosine {similarity:.4f} < {_MIN_COSINE}, authored text {text!r}) -- the "
            "stored vector was computed from different text and no presence, "
            "dimension or norm check can see it"
        )
    return None


def _node_by_id(documents: list[SeedDocument]) -> dict[str, object]:
    """Index every `SeedNode` across all documents by id.

    Shared by `check_containment` and `check_negation_proximity` -- both
    ask the same question ("how does this fact's object appear in
    prose?") and answering it independently in two places is exactly
    what let R1.4's C1 regression through: Task 14 fixed containment's
    id-vs-display-name resolution and left negation-proximity on the raw
    id, so the gate that should have caught an inversion near a trait's
    prose mention (`**Transparent**`) was scanning for a string
    (`trait-transparent`) that string never contains.
    """
    return {node.id: node for doc in documents for node in doc.nodes}


def _search_term_for(fact_object: str, node_by_id: dict[str, object]) -> str:
    """Resolve how a fact's object should be searched for in a document body.

    A `SeedNode`'s `display_name` is what a human author actually writes
    in prose (`**Transparent**`); the raw kebab id (`trait-transparent`)
    is not prose at all and a literal-substring search for it against
    real content fails (containment, Task 9) or silently never runs
    (negation-proximity, C1) depending on which side of the check it
    breaks. Falls back to the raw id when the object has no matching
    `SeedNode` (referential integrity is `load_seed_documents`'s job, not
    a gate's -- Task 11) or the node defines no `display_name`, so a
    fact's object is never silently skipped over by either caller.
    """
    node = node_by_id.get(fact_object)
    display_name = getattr(node, "display_name", None) if node is not None else None
    return display_name or fact_object


def check_containment(documents: list[SeedDocument]) -> GateResult:
    """Verify every fact's object is mentioned by display name in its document body.

    R1.4 Task 14: matches on the object node's `SeedNode.display_name`
    (Task 11), not the raw `fact.object` id, via the shared
    `_search_term_for` helper. The original Task 9 implementation checked
    the raw id as a literal substring, which is structurally unable to
    pass against real prose -- `fact.object` is a kebab id
    (`trait-transparent`); the prose describes it by display name
    (`**Transparent**`), a string the id never equals. 29 of 30 real
    facts failed under that check. A prefix-strip/hyphen-collapse
    normalization was scoped as the fix during Task 10 but never
    implemented; by the time this landed, Task 11 had given every node an
    exact `display_name`, which is strictly better than a heuristic
    reconstruction of one -- use it directly instead.

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
    node_by_id = _node_by_id(documents)
    failures: list[str] = []
    for doc in documents:
        body_lower = doc.body.lower()
        for fact in doc.facts:
            search_term = _search_term_for(fact.object, node_by_id)
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

    For every occurrence of a fact's object's search term (Task 14's C1
    fix: resolved via the same `_search_term_for` helper `check_containment`
    uses, not the raw `fact.object` id -- see below) in its document body,
    scans `_PROXIMITY_WINDOW` characters on either side (case-insensitive)
    for one of `_NEGATION_MARKERS`. A marker inside that window flags the
    fact; a marker elsewhere in the document does not -- an unrelated
    negation about a different fact must not fail this one.

    C1 (R1.4 whole-branch review): this gate previously searched the raw
    `fact.object` kebab id (`trait-transparent`), which real prose never
    contains -- it describes entities by `SeedNode.display_name`
    (`**Transparent**`). `_find_all` therefore returned an empty list for
    almost every real fact, the scan loop never ran, and the gate reported
    `passed=True` having examined nothing: silent, not loud, which is why
    it survived Task 14 (which fixed `check_containment`'s identical
    defect, the loud-failing half of the same bug) -- live measurement
    against the real seed source found only 4 of 30 facts scannable under
    the raw-id search, 0 of 20 in `seed/mist.md` specifically, the entire
    persona layer. Fixed by resolving the same display-name search term
    `check_containment` resolves, via the shared `_search_term_for`.

    Partial, like `check_containment`: proximity is not parsing, so this
    cannot tell "Raj no longer works at Slalom" (a real inversion) apart
    from an unrelated marker that happens to land in the window by
    coincidence. See `check_containment`'s docstring for the full
    limitation statement and the actual backstops.

    Args:
        documents: Parsed seed documents to check.

    Returns:
        `GateResult` with one failure line per fact with a negation
        marker near an occurrence of its object's search term.
    """
    node_by_id = _node_by_id(documents)
    failures: list[str] = []
    for doc in documents:
        body_lower = doc.body.lower()
        for fact in doc.facts:
            search_term = _search_term_for(fact.object, node_by_id).lower()
            if not search_term:
                continue
            for start in _find_all(body_lower, search_term):
                end = start + len(search_term)
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


def _l2_norm(vector: list[float]) -> float:
    """Euclidean length of a stored embedding."""
    return math.sqrt(sum(float(x) * float(x) for x in vector))


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors.

    Pure Python rather than numpy: the gate compares at most a few dozen
    384-d vectors per run, and `gates.py` is imported by every
    `mist_admin.py` subcommand, so it stays dependency-light on purpose
    (the same reason `embedding_text_for` lives in a module that pulls in
    no model layer). Callers must have already established equal lengths;
    `zip(strict=True)` turns a violation into a loud error rather than a
    silently truncated -- and therefore wrong -- similarity.
    """
    dot = sum(float(x) * float(y) for x, y in zip(a, b, strict=True))
    magnitude = _l2_norm(a) * _l2_norm(b)
    if magnitude == 0.0:
        return 0.0
    return dot / magnitude


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
