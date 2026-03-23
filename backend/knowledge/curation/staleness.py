"""Staleness detection and proactive confirmation list generation.

Scans active entities in the knowledge graph, categorizes them by
confidence tier, and builds a confirmation list of stale entities
that MIST should proactively ask the user about.
"""

import logging
import time
from dataclasses import dataclass

from backend.knowledge.storage.graph_executor import GraphExecutor

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StalenessResult:
    """Outcome of a staleness detection run.

    Attributes:
        active_count: Entities with confidence >= 0.6.
        stale_count: Entities with confidence 0.4-0.59.
        very_stale_count: Entities with confidence 0.2-0.39.
        confirmation_list: Stale-tier entities ordered by confidence
            ascending (most stale first) for proactive user confirmation.
        duration_ms: Wall-clock time of the scan in milliseconds.
    """

    active_count: int
    stale_count: int
    very_stale_count: int
    confirmation_list: tuple[dict, ...]
    duration_ms: float


class StalenessDetector:
    """Categorize active entities by confidence staleness tier.

    Tiers:
        >= 0.6   active     -- no action
        0.4-0.59 stale      -- add to proactive confirmation list
        0.2-0.39 very stale -- flag for review
        < 0.2    moribund   -- handled by ConfidenceDecayJob archival
    """

    def __init__(self, executor: GraphExecutor) -> None:
        """Initialize with a GraphExecutor for async Neo4j access.

        Args:
            executor: A GraphExecutor (or async-compatible fake).
        """
        self._executor = executor

    async def run(self) -> StalenessResult:
        """Scan entities and categorize by staleness tier."""
        start = time.perf_counter()

        rows = await self._executor.execute_query(
            "MATCH (n:__Entity__) "
            "WHERE n.status = 'active' AND n.confidence IS NOT NULL "
            "RETURN n.id AS id, n.display_name AS display_name, "
            "       n.entity_type AS entity_type, n.confidence AS confidence, "
            "       n.updated_at AS last_updated"
        )

        active_count = 0
        stale_count = 0
        very_stale_count = 0
        stale_entities: list[dict] = []

        for row in rows:
            conf = row["confidence"]
            if conf >= 0.6:
                active_count += 1
            elif conf >= 0.4:
                stale_count += 1
                stale_entities.append(
                    {
                        "id": row["id"],
                        "display_name": row["display_name"],
                        "entity_type": row["entity_type"],
                        "confidence": conf,
                        "last_updated": row["last_updated"],
                    }
                )
            elif conf >= 0.2:
                very_stale_count += 1

            # < 0.2 is moribund -- not counted here, handled by decay job

        # Order by confidence ascending so most-stale items come first
        stale_entities.sort(key=lambda e: e["confidence"])

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "Staleness scan: active=%d stale=%d very_stale=%d (%.1fms)",
            active_count,
            stale_count,
            very_stale_count,
            elapsed_ms,
        )

        return StalenessResult(
            active_count=active_count,
            stale_count=stale_count,
            very_stale_count=very_stale_count,
            confirmation_list=tuple(stale_entities),
            duration_ms=elapsed_ms,
        )
