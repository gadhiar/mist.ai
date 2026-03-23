"""Temporal confidence decay for knowledge graph entities.

Applies exponential half-life decay to entity confidence scores based on
ontology policies. External and bridging entities decay over time; internal
entities are exempt. Entities whose confidence drops below the archive
threshold are marked as archived.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from backend.knowledge.ontologies.base import ConfidencePolicy
from backend.knowledge.ontologies.v1_0_0 import ONTOLOGY_V1_0_0
from backend.knowledge.storage.graph_executor import GraphExecutor

logger = logging.getLogger(__name__)

ARCHIVE_THRESHOLD = 0.2

_FETCH_QUERY = """\
MATCH (e:__Entity__)
WHERE e.knowledge_domain IN $domains AND e.status = 'active'
RETURN e.id AS id,
       e.knowledge_domain AS domain,
       e.confidence AS confidence,
       e.updated_at AS updated_at
"""

_UPDATE_CONFIDENCE_QUERY = """\
MATCH (e:__Entity__ {id: $id})
SET e.confidence = $confidence
"""

_ARCHIVE_QUERY = """\
MATCH (e:__Entity__ {id: $id})
SET e.confidence = $confidence, e.status = 'archived'
"""


@dataclass(frozen=True, slots=True)
class DecayResult:
    """Outcome of a confidence decay run."""

    entities_scanned: int
    entities_decayed: int
    entities_archived: int
    duration_ms: float


def _build_policy_map(
    policies: tuple[ConfidencePolicy, ...],
) -> dict[str, ConfidencePolicy]:
    """Map domain name -> policy for decay-enabled domains only."""
    return {
        str(p.domain): p for p in policies if p.decay_enabled and p.decay_half_life_days is not None
    }


class ConfidenceDecayJob:
    """Scan entities and apply temporal confidence decay per ontology policies."""

    def __init__(self, executor: GraphExecutor) -> None:
        self._executor = executor
        self._policy_map = _build_policy_map(ONTOLOGY_V1_0_0.confidence_policies)

    async def run(self) -> DecayResult:
        """Scan all entities, apply temporal decay, archive moribund ones."""
        start = time.perf_counter()

        domains = list(self._policy_map.keys())
        if not domains:
            elapsed = (time.perf_counter() - start) * 1000
            return DecayResult(
                entities_scanned=0,
                entities_decayed=0,
                entities_archived=0,
                duration_ms=elapsed,
            )

        records = await self._executor.execute_query(_FETCH_QUERY, {"domains": domains})

        scanned = len(records)
        decayed = 0
        archived = 0

        for record in records:
            entity_id = record["id"]
            domain = record["domain"]
            confidence = record["confidence"]
            updated_at = record["updated_at"]

            policy = self._policy_map.get(domain)
            if policy is None or policy.decay_half_life_days is None:
                continue

            days_since = _days_since_update(updated_at)
            if days_since <= 0:
                continue

            new_confidence = confidence * (0.5 ** (days_since / policy.decay_half_life_days))

            if new_confidence < ARCHIVE_THRESHOLD:
                await self._executor.execute_write(
                    _ARCHIVE_QUERY,
                    {"id": entity_id, "confidence": new_confidence},
                )
                archived += 1
                decayed += 1
            elif new_confidence < confidence:
                await self._executor.execute_write(
                    _UPDATE_CONFIDENCE_QUERY,
                    {"id": entity_id, "confidence": new_confidence},
                )
                decayed += 1

        elapsed = (time.perf_counter() - start) * 1000

        logger.info(
            "Confidence decay complete: scanned=%d decayed=%d archived=%d (%.1fms)",
            scanned,
            decayed,
            archived,
            elapsed,
        )

        return DecayResult(
            entities_scanned=scanned,
            entities_decayed=decayed,
            entities_archived=archived,
            duration_ms=elapsed,
        )


def _days_since_update(updated_at: str | float) -> float:
    """Compute days between `updated_at` and now.

    Accepts either an ISO-8601 string or an epoch timestamp (float).
    """
    import datetime as _dt

    now = _dt.datetime.now(_dt.UTC)

    if isinstance(updated_at, int | float):
        then = _dt.datetime.fromtimestamp(updated_at, tz=_dt.UTC)
    else:
        raw = str(updated_at)
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        then = _dt.datetime.fromisoformat(raw)
        if then.tzinfo is None:
            then = then.replace(tzinfo=_dt.UTC)

    delta = now - then
    return delta.total_seconds() / 86400
