"""Graph health scoring with a 6-dimension composite score.

Computes a 0-100 overall health score from freshness, confidence,
connectivity, consistency, coverage, and self-model sub-scores.
Each sub-score is 0-100, combined via a weighted average.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from backend.knowledge.ontologies.v1_0_0 import ONTOLOGY_V1_0_0
from backend.knowledge.storage.graph_executor import GraphExecutor

logger = logging.getLogger(__name__)

# Weights must sum to 1.0.
_WEIGHTS = {
    "freshness": 0.20,
    "confidence": 0.20,
    "connectivity": 0.15,
    "consistency": 0.15,
    "coverage": 0.15,
    "self_model": 0.15,
}

_VALID_ENTITY_TYPES: frozenset[str] = frozenset(nt.type_name for nt in ONTOLOGY_V1_0_0.node_types)

_TOTAL_ONTOLOGY_TYPES = len(_VALID_ENTITY_TYPES)

# -- Cypher queries ----------------------------------------------------------

_COUNT_QUERY = """\
MATCH (e:__Entity__)
WHERE e.status = 'active'
RETURN count(e) AS total
"""

_FRESHNESS_QUERY = """\
MATCH (e:__Entity__)
WHERE e.status = 'active'
  AND e.updated_at >= $cutoff
RETURN count(e) AS fresh_count
"""

_CONFIDENCE_QUERY = """\
MATCH (e:__Entity__)
WHERE e.status = 'active'
RETURN avg(e.confidence) AS avg_confidence
"""

_CONNECTIVITY_QUERY = """\
MATCH (e:__Entity__)
WHERE e.status = 'active'
OPTIONAL MATCH (e)-[r]-()
RETURN count(DISTINCT e) AS entity_count,
       count(r) AS rel_count
"""

_CONSISTENCY_QUERY = """\
MATCH (e:__Entity__)
WHERE e.status = 'active'
RETURN e.entity_type AS entity_type, count(e) AS cnt
"""

_SELF_MODEL_QUERY = """\
MATCH (e:__Entity__)
WHERE e.status = 'active'
  AND e.knowledge_domain = 'internal'
RETURN count(e) AS internal_count
"""

_REL_COUNT_QUERY = """\
MATCH ()-[r]->()
RETURN count(r) AS total_rels
"""


@dataclass(frozen=True, slots=True)
class HealthScore:
    """Composite graph health score."""

    overall: float
    freshness: float
    confidence: float
    connectivity: float
    consistency: float
    coverage: float
    self_model: float
    entity_count: int
    relationship_count: int


class GraphHealthScorer:
    """Compute a composite 0-100 graph health score."""

    def __init__(self, executor: GraphExecutor) -> None:
        self._executor = executor

    async def run(self) -> HealthScore:
        """Compute composite graph health score."""
        start = time.perf_counter()

        total = await self._count_entities()
        if total == 0:
            elapsed = (time.perf_counter() - start) * 1000
            logger.info("Health score: empty graph (%.1fms)", elapsed)
            return HealthScore(
                overall=0.0,
                freshness=0.0,
                confidence=0.0,
                connectivity=0.0,
                consistency=0.0,
                coverage=0.0,
                self_model=0.0,
                entity_count=0,
                relationship_count=0,
            )

        freshness = await self._score_freshness(total)
        confidence = await self._score_confidence()
        connectivity, rel_count = await self._score_connectivity()
        consistency = await self._score_consistency(total)
        coverage = await self._score_coverage()
        self_model = await self._score_self_model()

        overall = (
            _WEIGHTS["freshness"] * freshness
            + _WEIGHTS["confidence"] * confidence
            + _WEIGHTS["connectivity"] * connectivity
            + _WEIGHTS["consistency"] * consistency
            + _WEIGHTS["coverage"] * coverage
            + _WEIGHTS["self_model"] * self_model
        )

        elapsed = (time.perf_counter() - start) * 1000

        logger.info(
            "Health score: overall=%.1f freshness=%.1f confidence=%.1f "
            "connectivity=%.1f consistency=%.1f coverage=%.1f "
            "self_model=%.1f entities=%d rels=%d (%.1fms)",
            overall,
            freshness,
            confidence,
            connectivity,
            consistency,
            coverage,
            self_model,
            total,
            rel_count,
            elapsed,
        )

        return HealthScore(
            overall=overall,
            freshness=freshness,
            confidence=confidence,
            connectivity=connectivity,
            consistency=consistency,
            coverage=coverage,
            self_model=self_model,
            entity_count=total,
            relationship_count=rel_count,
        )

    async def _count_entities(self) -> int:
        records = await self._executor.execute_query(_COUNT_QUERY)
        if not records:
            return 0
        return records[0]["total"]

    async def _score_freshness(self, total: int) -> float:
        """Percentage of entities updated in last 30 days."""
        import datetime as _dt

        cutoff = (_dt.datetime.now(_dt.UTC) - _dt.timedelta(days=30)).isoformat()

        records = await self._executor.execute_query(_FRESHNESS_QUERY, {"cutoff": cutoff})
        if not records:
            return 0.0
        fresh = records[0]["fresh_count"]
        return min(100.0, fresh / total * 100)

    async def _score_confidence(self) -> float:
        """Average confidence * 100."""
        records = await self._executor.execute_query(_CONFIDENCE_QUERY)
        if not records or records[0]["avg_confidence"] is None:
            return 0.0
        return min(100.0, records[0]["avg_confidence"] * 100)

    async def _score_connectivity(self) -> tuple[float, int]:
        """min(100, avg_rels_per_entity * 20). Returns (score, rel_count)."""
        records = await self._executor.execute_query(_CONNECTIVITY_QUERY)
        if not records or records[0]["entity_count"] == 0:
            return 0.0, 0

        entity_count = records[0]["entity_count"]
        rel_count = records[0]["rel_count"]
        avg_rels = rel_count / entity_count
        score = min(100.0, avg_rels * 20)
        return score, rel_count

    async def _score_consistency(self, total: int) -> float:
        """Percentage of entities with a valid entity_type per ontology."""
        records = await self._executor.execute_query(_CONSISTENCY_QUERY)
        if not records:
            return 0.0
        valid_count = sum(r["cnt"] for r in records if r["entity_type"] in _VALID_ENTITY_TYPES)
        return min(100.0, valid_count / total * 100)

    async def _score_coverage(self) -> float:
        """Distinct entity types used / total ontology types * 100."""
        records = await self._executor.execute_query(_CONSISTENCY_QUERY)
        if not records:
            return 0.0
        distinct_types = {
            r["entity_type"] for r in records if r["entity_type"] in _VALID_ENTITY_TYPES
        }
        return min(100.0, len(distinct_types) / _TOTAL_ONTOLOGY_TYPES * 100)

    async def _score_self_model(self) -> float:
        """min(100, internal_entity_count / 5 * 100)."""
        records = await self._executor.execute_query(_SELF_MODEL_QUERY)
        if not records:
            return 0.0
        internal_count = records[0]["internal_count"]
        return min(100.0, internal_count / 5 * 100)
