"""Embedding maintenance for knowledge graph entities.

Finds entities whose text content has changed since their embedding was
last generated, then regenerates the embedding vector from the current
display_name, entity_type, and description.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from backend.interfaces import EmbeddingProvider
from backend.knowledge.storage.graph_executor import GraphExecutor

logger = logging.getLogger(__name__)

_STALE_EMBEDDINGS_QUERY = """\
MATCH (e:__Entity__)
WHERE e.status = 'active'
  AND (e.embedding_updated_at IS NULL OR e.updated_at > e.embedding_updated_at)
RETURN e.id AS id,
       e.display_name AS display_name,
       e.entity_type AS entity_type,
       e.description AS description
"""

_UPDATE_EMBEDDING_QUERY = """\
MATCH (e:__Entity__ {id: $id})
SET e.embedding = $embedding,
    e.embedding_updated_at = datetime()
"""


@dataclass(frozen=True, slots=True)
class EmbeddingMaintenanceResult:
    """Outcome of an embedding maintenance run."""

    entities_scanned: int
    embeddings_regenerated: int
    duration_ms: float


def _build_embedding_text(display_name: str, entity_type: str, description: str) -> str:
    """Compose the text input for embedding generation."""
    parts = [display_name, entity_type]
    if description:
        parts.append(description)
    return " ".join(parts)


class EmbeddingMaintenance:
    """Scan for stale embeddings and regenerate them.

    An embedding is considered stale when `updated_at > embedding_updated_at`
    or when `embedding_updated_at` is null (entity was never embedded).
    """

    def __init__(
        self,
        executor: GraphExecutor,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self._executor = executor
        self._embedding_provider = embedding_provider

    async def run(self) -> EmbeddingMaintenanceResult:
        """Find stale embeddings and regenerate them."""
        start = time.perf_counter()

        records = await self._executor.execute_query(_STALE_EMBEDDINGS_QUERY)

        scanned = len(records)
        regenerated = 0

        for record in records:
            entity_id = record["id"]
            display_name = record.get("display_name", "") or ""
            entity_type = record.get("entity_type", "") or ""
            description = record.get("description", "") or ""

            text = _build_embedding_text(display_name, entity_type, description)
            embedding = self._embedding_provider.generate_embedding(text)

            await self._executor.execute_write(
                _UPDATE_EMBEDDING_QUERY,
                {"id": entity_id, "embedding": embedding},
            )
            regenerated += 1

        elapsed = (time.perf_counter() - start) * 1000

        logger.info(
            "Embedding maintenance complete: scanned=%d regenerated=%d (%.1fms)",
            scanned,
            regenerated,
            elapsed,
        )

        return EmbeddingMaintenanceResult(
            entities_scanned=scanned,
            embeddings_regenerated=regenerated,
            duration_ms=elapsed,
        )
