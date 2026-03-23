"""Orphan entity detection and archival.

Finds entities with zero meaningful relationships (excluding EXTRACTED_FROM
provenance edges) and low confidence, then archives them. Never archives
User or MistIdentity nodes regardless of state.
"""

import logging
import time
from dataclasses import dataclass

from backend.knowledge.storage.graph_executor import GraphExecutor

logger = logging.getLogger(__name__)

# Entity types that must never be archived, regardless of orphan status.
_PROTECTED_TYPES = frozenset({"User", "MistIdentity"})

# Orphans are only archived when confidence falls below this threshold.
_CONFIDENCE_THRESHOLD = 0.4


@dataclass(frozen=True, slots=True)
class OrphanResult:
    """Summary of an orphan detection run."""

    entities_scanned: int
    orphans_found: int
    orphans_archived: int
    duration_ms: float


class OrphanDetector:
    """Find and archive orphan entities in the knowledge graph.

    An orphan is an active entity with zero relationships (excluding
    EXTRACTED_FROM provenance edges), confidence below 0.4, and an
    entity_type that is not protected (User, MistIdentity).
    """

    def __init__(self, executor: GraphExecutor) -> None:
        self._executor = executor

    async def run(self) -> OrphanResult:
        """Find and archive orphan entities."""
        start = time.perf_counter()

        # Count all active entities.
        count_records = await self._executor.execute_query(
            "MATCH (e:__Entity__) WHERE e.status = 'active' " "RETURN count(e) AS total"
        )
        entities_scanned = count_records[0]["total"] if count_records else 0

        if entities_scanned == 0:
            elapsed = (time.perf_counter() - start) * 1000
            return OrphanResult(
                entities_scanned=0,
                orphans_found=0,
                orphans_archived=0,
                duration_ms=elapsed,
            )

        # Find orphans: active entities with zero non-EXTRACTED_FROM relationships
        # and confidence below threshold, excluding protected types.
        orphan_records = await self._executor.execute_query(
            "MATCH (e:__Entity__) "
            "WHERE e.status = 'active' "
            "AND e.confidence < $threshold "
            "AND NOT e.entity_type IN $protected_types "
            "AND NOT EXISTS { "
            "  MATCH (e)-[r]-() "
            "  WHERE type(r) <> 'EXTRACTED_FROM' "
            "} "
            "RETURN e.id AS id, e.entity_type AS entity_type, "
            "e.confidence AS confidence",
            {"threshold": _CONFIDENCE_THRESHOLD, "protected_types": list(_PROTECTED_TYPES)},
        )

        orphans_found = len(orphan_records)
        orphans_archived = 0

        if orphans_found > 0:
            orphan_ids = [r["id"] for r in orphan_records]
            await self._executor.execute_write(
                "MATCH (e:__Entity__) " "WHERE e.id IN $ids " "SET e.status = 'archived'",
                {"ids": orphan_ids},
            )
            orphans_archived = orphans_found
            logger.info(
                "Archived %d orphan entities out of %d scanned",
                orphans_archived,
                entities_scanned,
            )

        elapsed = (time.perf_counter() - start) * 1000
        return OrphanResult(
            entities_scanned=entities_scanned,
            orphans_found=orphans_found,
            orphans_archived=orphans_archived,
            duration_ms=elapsed,
        )
