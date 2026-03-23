"""Community detection via Neo4j GDS Louvain algorithm.

Writes a `community_id` property on each entity node. Used for retrieval
boosting and missing relationship suggestion. Requires GDS plugin -- falls
back gracefully if GDS is not installed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from backend.knowledge.storage.graph_executor import GraphExecutor

logger = logging.getLogger(__name__)

_PROJECT_QUERY = """\
CALL gds.graph.project(
    'mist_community',
    '__Entity__',
    {RELATES_TO: {orientation: 'UNDIRECTED'}}
)
"""

_LOUVAIN_QUERY = """\
CALL gds.louvain.write('mist_community', {writeProperty: 'community_id'})
YIELD communityCount, nodePropertiesWritten
RETURN communityCount, nodePropertiesWritten
"""

_DROP_PROJECTION_QUERY = """\
CALL gds.graph.drop('mist_community', false)
"""


@dataclass(frozen=True, slots=True)
class CommunityResult:
    """Outcome of a community detection run."""

    communities_found: int
    entities_labeled: int
    duration_ms: float


class CommunityDetector:
    """Detect communities in the knowledge graph using Louvain modularity.

    Requires Neo4j GDS plugin. If GDS is unavailable the job logs a warning
    and returns a zero result instead of raising.
    """

    def __init__(self, executor: GraphExecutor) -> None:
        self._executor = executor

    async def run(self) -> CommunityResult:
        """Run Louvain community detection via GDS.

        Returns zero result when GDS is not available.
        """
        start = time.perf_counter()

        try:
            await self._executor.execute_write(_PROJECT_QUERY)
        except Exception:
            logger.warning(
                "GDS graph projection failed -- GDS plugin may not be installed. "
                "Skipping community detection."
            )
            elapsed = (time.perf_counter() - start) * 1000
            return CommunityResult(communities_found=0, entities_labeled=0, duration_ms=elapsed)

        try:
            records = await self._executor.execute_write(_LOUVAIN_QUERY)

            communities = 0
            labeled = 0
            if records:
                communities = records[0].get("communityCount", 0)
                labeled = records[0].get("nodePropertiesWritten", 0)

            elapsed = (time.perf_counter() - start) * 1000

            logger.info(
                "Community detection complete: communities=%d labeled=%d (%.1fms)",
                communities,
                labeled,
                elapsed,
            )

            return CommunityResult(
                communities_found=communities,
                entities_labeled=labeled,
                duration_ms=elapsed,
            )
        finally:
            try:
                await self._executor.execute_write(_DROP_PROJECTION_QUERY)
            except Exception:
                logger.debug("Failed to drop GDS projection 'mist_community'")
