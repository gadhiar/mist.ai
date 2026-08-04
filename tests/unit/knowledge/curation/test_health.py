"""Tests for GraphHealthScorer.

The fake here answers each query from a node population keyed by PARTITION
LABEL, then by RETURN alias. The label half is the point. The previous version
of this file dispatched on the RETURN alias alone, which cannot see which
partition a query reads: the self-model sub-score was pointed at :__Entity__
(disjoint from :__SelfModel__ since the R1.0 relabel, so a permanent count of
zero) and all five self-model assertions stayed green anyway. A fake that
cannot distinguish `MATCH (e:__SelfModel__)` from `MATCH (e:__Nonexistent__)`
verifies the scorer's arithmetic and nothing about where its numbers come from.
"""

import re
from collections import Counter
from datetime import UTC, datetime, timedelta

import pytest

from backend.knowledge.curation.health import (
    _TOTAL_ONTOLOGY_TYPES,
    _VALID_ENTITY_TYPES,
    _WEIGHTS,
    GraphHealthScorer,
    HealthScore,
)
from backend.knowledge.storage.partitions import (
    ENTITY_LABEL,
    PROVENANCE_LABEL,
    SELF_MODEL_LABEL,
    SELF_MODEL_TYPES,
)
from tests.mocks.neo4j import FakeGraphExecutor, FakeNeo4jConnection, FakeNeo4jRecord

# Ontology types that graph_writer.py writes into the :__Provenance__ partition
# rather than :__Entity__ (ADR-009). Named here so the coverage tests can put a
# type in the partition production actually puts it in.
_PROVENANCE_TYPES: frozenset[str] = frozenset(
    {"ConversationContext", "LearningEvent", "ExternalSource", "VectorChunk"}
)

# First `MATCH (alias:Label)` in a query, capturing a `A|B|C` label union whole.
_MATCH_LABELS = re.compile(r"MATCH \(\w+:([\w|]+)\)")

# Everything from the first WHERE to the final RETURN -- the predicate text the
# fake is obliged to account for.
_WHERE_BLOCK = re.compile(r"\bWHERE\b(.*)\bRETURN\b", re.DOTALL)

# Node predicates the fake evaluates against the population, longest first so
# stripping one never leaves a fragment of another behind.
_MODELLED_PREDICATES: tuple[str, ...] = (
    "coalesce(e.status, 'active') = 'active'",
    "e.updated_at >= $cutoff",
    "e.status = 'active'",
)

# Relationship-currency predicates on the OPTIONAL MATCH. The fake takes
# `rel_count` as a given rather than evaluating these -- whether an edge is a
# current belief is Neo4j's evaluation, not a property of the node population.
# Listed explicitly so they are ACKNOWLEDGED rather than silently skipped; a
# new predicate outside this tuple still trips the completeness check below.
_UNMODELLED_PREDICATES: tuple[str, ...] = (
    "coalesce(r.is_latest_belief, true)",
    "r.valid_to IS NULL OR r.valid_to > $now",
    "r.valid_from IS NULL OR r.valid_from = '-inf' OR r.valid_from <= $now",
)


def _assert_every_predicate_accounted_for(query: str) -> None:
    """Fail loudly if the query filters on something the fake does not evaluate.

    Without this the fake silently ignores any predicate it was not written
    for, which reproduces the exact defect this file exists to prevent one
    level down: `AND e.knowledge_domain = 'internal'` was dropped on the floor,
    so a mutation reintroducing it still scored as though it were not there.
    """
    block = _WHERE_BLOCK.search(query)
    if block is None:
        return
    # An OPTIONAL MATCH between the two WHEREs is a traversal, not a filter --
    # drop the clause and its own WHERE keyword before weighing predicates.
    residue = re.sub(r"OPTIONAL MATCH.*?WHERE", "", block.group(1), flags=re.DOTALL)
    for predicate in (*_MODELLED_PREDICATES, *_UNMODELLED_PREDICATES):
        residue = residue.replace(predicate, "")
    residue = re.sub(r"\b(AND|OR)\b|[()\s]", "", residue)
    if residue:
        raise AssertionError(
            f"Query filters on {residue!r}, which the fake does not evaluate. "
            "Add it to _MODELLED_PREDICATES and honor it, or the assertions "
            f"below are blind to it. Query: {query!r}"
        )


def _node(
    entity_type: str,
    *,
    confidence: float = 1.0,
    fresh: bool = True,
    status: str | None = "active",
) -> dict:
    """Build one node for the fake population.

    `status=None` models a node carrying no `status` property at all -- the
    real shape of 20 of the 21 seeded self-model nodes, since the seed applier
    writes only the properties `mist-memory/seed/mist.md` authors and it
    authors `status` on `mist-identity` alone.
    """
    age = timedelta(days=1) if fresh else timedelta(days=90)
    node = {
        "entity_type": entity_type,
        "confidence": confidence,
        "updated_at": (datetime.now(UTC) - age).isoformat(),
    }
    if status is not None:
        node["status"] = status
    return node


class _PartitionedGraph:
    """Answer health.py's Cypher from a population keyed by partition label.

    Resolves the query's MATCH label expression against the population FIRST,
    so a query aimed at a partition that holds no such nodes -- or at a label
    no node carries -- reads zero rows, exactly as Neo4j would answer it. Only
    then does it dispatch on the RETURN alias to compute the aggregate.

    The `status` predicate is honored as written rather than assumed: a query
    using `coalesce(e.status, 'active')` sees status-less nodes, one using a
    bare `e.status = 'active'` does not. That keeps the difference between the
    two visible to assertions instead of smoothing it away.
    """

    def __init__(
        self,
        *,
        population: dict[str, list[dict]] | None = None,
        rel_count: int = 0,
    ) -> None:
        self._population = population or {}
        self._rel_count = rel_count

    def __call__(self, query: str, params: dict | None) -> list:
        match = _MATCH_LABELS.search(query)
        if match is None:
            raise AssertionError(f"Fake cannot parse a MATCH partition from query: {query!r}")
        _assert_every_predicate_accounted_for(query)
        nodes = self._active(self._nodes_for(match.group(1)), query)
        return self._answer(query, params or {}, nodes)

    def _nodes_for(self, label_expression: str) -> list[dict]:
        """Resolve an `A|B|C` label union to its nodes.

        A label with no population contributes nothing -- which is the whole
        mechanism by which a mis-pointed query is caught.
        """
        nodes: list[dict] = []
        for label in label_expression.split("|"):
            nodes.extend(self._population.get(label, []))
        return nodes

    def _active(self, nodes: list[dict], query: str) -> list[dict]:
        """Apply whichever `status` predicate the query actually carries."""
        if "coalesce(e.status, 'active') = 'active'" in query:
            return [n for n in nodes if n.get("status", "active") == "active"]
        if "e.status = 'active'" in query:
            return [n for n in nodes if n.get("status") == "active"]
        return list(nodes)

    def _answer(self, query: str, params: dict, nodes: list[dict]) -> list:
        if "count(e) AS total" in query:
            return [FakeNeo4jRecord({"total": len(nodes)})]
        if "count(e) AS self_model_count" in query:
            return [FakeNeo4jRecord({"self_model_count": len(nodes)})]
        if "fresh_count" in query:
            cutoff = params["cutoff"]
            fresh = [n for n in nodes if n["updated_at"] >= cutoff]
            return [FakeNeo4jRecord({"fresh_count": len(fresh)})]
        if "avg(e.confidence)" in query:
            avg = sum(n["confidence"] for n in nodes) / len(nodes) if nodes else None
            return [FakeNeo4jRecord({"avg_confidence": avg})]
        if "count(DISTINCT e) AS entity_count" in query:
            return [FakeNeo4jRecord({"entity_count": len(nodes), "rel_count": self._rel_count})]
        if "e.entity_type AS entity_type" in query:
            counts = Counter(n["entity_type"] for n in nodes)
            return [
                FakeNeo4jRecord({"entity_type": t, "cnt": c}) for t, c in sorted(counts.items())
            ]
        raise AssertionError(f"Fake does not model this query: {query!r}")


def _build_scorer(
    *,
    entities: list[dict] | None = None,
    self_model: list[dict] | None = None,
    provenance: list[dict] | None = None,
    rel_count: int = 0,
) -> GraphHealthScorer:
    """Build a scorer over a three-partition node population."""
    population = {
        ENTITY_LABEL: entities or [],
        SELF_MODEL_LABEL: self_model or [],
        PROVENANCE_LABEL: provenance or [],
    }
    connection = FakeNeo4jConnection(
        query_router=_PartitionedGraph(population=population, rel_count=rel_count)
    )
    return GraphHealthScorer(FakeGraphExecutor(connection))


def _self_model_nodes(count: int) -> list[dict]:
    """`count` self-model nodes, cycling through the five self-model types."""
    types = sorted(SELF_MODEL_TYPES)
    return [_node(types[i % len(types)]) for i in range(count)]


class TestEmptyGraph:
    @pytest.mark.asyncio
    async def test_empty_graph_returns_zero(self):
        scorer = _build_scorer()

        result = await scorer.run()

        assert result.overall == 0.0
        assert result.freshness == 0.0
        assert result.confidence == 0.0
        assert result.connectivity == 0.0
        assert result.consistency == 0.0
        assert result.coverage == 0.0
        assert result.self_model == 0.0
        assert result.entity_count == 0
        assert result.relationship_count == 0


class TestWeights:
    def test_weights_sum_to_one(self):
        assert abs(sum(_WEIGHTS.values()) - 1.0) < 1e-9

    @pytest.mark.asyncio
    async def test_overall_is_weighted_average(self):
        scorer = _build_scorer(
            entities=[_node("Technology")] * 5 + [_node("Person")] * 5,
            self_model=_self_model_nodes(5),
            rel_count=50,
        )

        result = await scorer.run()

        expected = (
            _WEIGHTS["freshness"] * result.freshness
            + _WEIGHTS["confidence"] * result.confidence
            + _WEIGHTS["connectivity"] * result.connectivity
            + _WEIGHTS["consistency"] * result.consistency
            + _WEIGHTS["coverage"] * result.coverage
            + _WEIGHTS["self_model"] * result.self_model
        )
        assert abs(result.overall - expected) < 1e-9


class TestPerfectGraph:
    @pytest.mark.asyncio
    async def test_perfect_graph_reaches_100(self):
        """Every ontology type present, each in the partition that holds it.

        Coverage can only reach 100 if its query spans all three partitions:
        five of the thirty ontology types live in :__SelfModel__ and four more
        in :__Provenance__, so an :__Entity__-only scan caps this at 83.3.
        """
        entity_types = sorted(_VALID_ENTITY_TYPES - SELF_MODEL_TYPES - _PROVENANCE_TYPES)
        entities = [_node(t) for t in entity_types]

        scorer = _build_scorer(
            entities=entities,
            self_model=[_node(t) for t in sorted(SELF_MODEL_TYPES)],
            provenance=[_node(t) for t in sorted(_PROVENANCE_TYPES)],
            rel_count=len(entities) * 5,
        )

        result = await scorer.run()

        assert result.freshness == 100.0
        assert result.confidence == 100.0
        assert result.connectivity == 100.0
        assert result.consistency == 100.0
        assert result.coverage == 100.0
        assert result.self_model == 100.0
        assert result.overall == 100.0


class TestPartitionScoping:
    """The partition each query reads is load-bearing, not incidental."""

    @pytest.mark.asyncio
    async def test_self_model_score_reads_the_selfmodel_partition(self):
        scorer = _build_scorer(
            entities=[_node("Technology")] * 5,
            self_model=_self_model_nodes(3),
        )

        result = await scorer.run()

        assert result.self_model == 60.0

    @pytest.mark.asyncio
    async def test_self_model_score_ignores_selfmodel_types_left_in_entity_partition(self):
        """The R1.0 regression, asserted directly.

        Self-model-typed nodes stranded in :__Entity__ are not the self-model:
        the partition label is what identifies it, not `entity_type` and not
        `knowledge_domain`. A scorer reading :__Entity__ would report 100 here.
        """
        scorer = _build_scorer(
            entities=[_node(t) for t in sorted(SELF_MODEL_TYPES)] + [_node("Technology")],
            self_model=[],
        )

        result = await scorer.run()

        assert result.self_model == 0.0

    @pytest.mark.asyncio
    async def test_self_model_counts_nodes_carrying_no_status_property(self):
        """The live seed population: 21 nodes, exactly one with `status`.

        A bare `e.status = 'active'` predicate would count that one node and
        score 20.0.
        """
        seeded = [_node("MistIdentity", status="active")]
        seeded += [_node("MistTrait", status=None) for _ in range(20)]

        scorer = _build_scorer(entities=[_node("Technology")], self_model=seeded)

        result = await scorer.run()

        assert result.self_model == 100.0

    @pytest.mark.asyncio
    async def test_self_model_excludes_explicitly_inactive_nodes(self):
        scorer = _build_scorer(
            entities=[_node("Technology")],
            self_model=_self_model_nodes(2) + [_node("MistTrait", status="archived")] * 8,
        )

        result = await scorer.run()

        assert result.self_model == 40.0

    @pytest.mark.asyncio
    async def test_coverage_counts_types_outside_the_entity_partition(self):
        """A type present only in :__SelfModel__ or :__Provenance__ still counts."""
        scorer = _build_scorer(
            entities=[_node("Technology")],
            self_model=[_node("MistTrait")],
            provenance=[_node("LearningEvent")],
        )

        result = await scorer.run()

        expected = 3 / _TOTAL_ONTOLOGY_TYPES * 100
        assert abs(result.coverage - expected) < 1e-9

    @pytest.mark.asyncio
    async def test_consistency_denominator_stays_entity_scoped(self):
        """Consistency divides by the :__Entity__ count, so it must count only
        :__Entity__ types. Widening it alongside coverage would let self-model
        and provenance nodes inflate the numerator past its own denominator.
        """
        scorer = _build_scorer(
            entities=[_node("Technology")] * 5 + [_node("BOGUS_TYPE")] * 5,
            self_model=_self_model_nodes(10),
            provenance=[_node("LearningEvent")] * 10,
        )

        result = await scorer.run()

        assert result.consistency == 50.0

    @pytest.mark.asyncio
    async def test_entity_count_stays_entity_scoped(self):
        scorer = _build_scorer(
            entities=[_node("Technology")] * 4,
            self_model=_self_model_nodes(9),
            provenance=[_node("LearningEvent")] * 7,
        )

        result = await scorer.run()

        assert result.entity_count == 4


class TestIndividualScores:
    @pytest.mark.asyncio
    async def test_freshness_half(self):
        scorer = _build_scorer(
            entities=[_node("Technology", fresh=True)] * 5 + [_node("Technology", fresh=False)] * 5,
        )

        result = await scorer.run()

        assert result.freshness == 50.0

    @pytest.mark.asyncio
    async def test_confidence_maps_to_percentage(self):
        scorer = _build_scorer(entities=[_node("Technology", confidence=0.75)] * 10)

        result = await scorer.run()

        assert result.confidence == 75.0

    @pytest.mark.asyncio
    async def test_connectivity_capped_at_100(self):
        scorer = _build_scorer(entities=[_node("Technology")] * 2, rel_count=100)

        result = await scorer.run()

        assert result.connectivity == 100.0

    @pytest.mark.asyncio
    async def test_connectivity_scales_with_avg_rels(self):
        scorer = _build_scorer(entities=[_node("Technology")] * 4, rel_count=8)

        result = await scorer.run()

        # avg_rels = 8/4 = 2, score = 2 * 20 = 40
        assert result.connectivity == 40.0

    @pytest.mark.asyncio
    async def test_consistency_with_invalid_types(self):
        scorer = _build_scorer(
            entities=[_node("Technology")] * 6 + [_node("BOGUS_TYPE")] * 4,
        )

        result = await scorer.run()

        # 6 valid out of 10
        assert result.consistency == 60.0

    @pytest.mark.asyncio
    async def test_coverage_partial(self):
        scorer = _build_scorer(entities=[_node("Technology")] * 5 + [_node("Person")] * 5)

        result = await scorer.run()

        expected = 2 / _TOTAL_ONTOLOGY_TYPES * 100
        assert abs(result.coverage - expected) < 1e-9

    @pytest.mark.asyncio
    async def test_self_model_partial(self):
        scorer = _build_scorer(
            entities=[_node("Technology")] * 5,
            self_model=_self_model_nodes(3),
        )

        result = await scorer.run()

        # 3/5 * 100 = 60
        assert result.self_model == 60.0

    @pytest.mark.asyncio
    async def test_self_model_capped_at_100(self):
        scorer = _build_scorer(
            entities=[_node("Technology")] * 10,
            self_model=_self_model_nodes(20),
        )

        result = await scorer.run()

        assert result.self_model == 100.0


class TestScoreCapping:
    @pytest.mark.asyncio
    async def test_no_sub_score_exceeds_100(self):
        entity_types = sorted(_VALID_ENTITY_TYPES - SELF_MODEL_TYPES - _PROVENANCE_TYPES)
        entities = [_node(t) for t in entity_types for _ in range(10)]

        scorer = _build_scorer(
            entities=entities,
            self_model=_self_model_nodes(50),
            provenance=[_node(t) for t in sorted(_PROVENANCE_TYPES)],
            rel_count=len(entities) * 100,
        )

        result = await scorer.run()

        assert result.freshness <= 100.0
        assert result.confidence <= 100.0
        assert result.connectivity <= 100.0
        assert result.consistency <= 100.0
        assert result.coverage <= 100.0
        assert result.self_model <= 100.0
        assert result.overall <= 100.0


class TestReturnType:
    @pytest.mark.asyncio
    async def test_returns_health_score_dataclass(self):
        scorer = _build_scorer()

        result = await scorer.run()

        assert isinstance(result, HealthScore)

    @pytest.mark.asyncio
    async def test_entity_and_rel_counts_populated(self):
        scorer = _build_scorer(
            entities=[_node("Technology", fresh=True)] * 3 + [_node("Technology", fresh=False)] * 2,
            rel_count=10,
        )

        result = await scorer.run()

        assert result.entity_count == 5
        assert result.relationship_count == 10
