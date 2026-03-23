"""Centrality analysis via Neo4j GDS PageRank algorithm.

Writes a `pagerank` property on each entity node for importance ranking.
Used for retrieval boosting and curation priority. Requires GDS plugin --
falls back gracefully if GDS is not installed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from backend.knowledge.storage.graph_executor import GraphExecutor

logger = logging.getLogger(__name__)

_PROJECT_QUERY = """\
CALL gds.graph.project(
    'mist_centrality',
    '__Entity__',
    'RELATES_TO'
)
"""

_PAGERANK_QUERY = """\
CALL gds.pageRank.write('mist_centrality', {writeProperty: 'pagerank'})
YIELD nodePropertiesWritten
RETURN nodePropertiesWritten
"""

_DROP_PROJECTION_QUERY = """\
CALL gds.graph.drop('mist_centrality', false)
"""


@dataclass(frozen=True, slots=True)
class CentralityResult:
    """Outcome of a centrality analysis run."""

    entities_scored: int
    duration_ms: float


class CentralityAnalyzer:
    """Compute PageRank centrality scores for knowledge graph entities.

    Requires Neo4j GDS plugin. If GDS is unavailable the job logs a warning
    and returns a zero result instead of raising.
    """

    def __init__(self, executor: GraphExecutor) -> None:
        self._executor = executor

    async def run(self) -> CentralityResult:
        """Run PageRank centrality analysis via GDS.

        Returns zero result when GDS is not available.
        """
        start = time.perf_counter()

        try:
            await self._executor.execute_write(_PROJECT_QUERY)
        except Exception:
            logger.warning(
                "GDS graph projection failed -- GDS plugin may not be installed. "
                "Skipping centrality analysis."
            )
            elapsed = (time.perf_counter() - start) * 1000
            return CentralityResult(entities_scored=0, duration_ms=elapsed)

        try:
            records = await self._executor.execute_write(_PAGERANK_QUERY)

            scored = 0
            if records:
                scored = records[0].get("nodePropertiesWritten", 0)

            elapsed = (time.perf_counter() - start) * 1000

            logger.info(
                "Centrality analysis complete: scored=%d (%.1fms)",
                scored,
                elapsed,
            )

            return CentralityResult(entities_scored=scored, duration_ms=elapsed)
        finally:
            try:
                await self._executor.execute_write(_DROP_PROJECTION_QUERY)
            except Exception:
                logger.debug("Failed to drop GDS projection 'mist_centrality'")
