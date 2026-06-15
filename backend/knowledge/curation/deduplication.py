"""Entity deduplication against existing graph state.

Stage 7a: 3-tier dedup (exact ID -> alias -> embedding similarity).
Produces merge instructions for the graph writer.
"""

import asyncio
import logging
from dataclasses import dataclass, field

from backend.interfaces import EmbeddingProvider
from backend.knowledge.curation.confidence import ConfidenceManager
from backend.knowledge.ontologies.hierarchy import dedup_type_filter
from backend.knowledge.storage.graph_executor import GraphExecutor

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.92
MAX_ALIASES = 20


@dataclass(frozen=True, slots=True)
class MergeAction:
    """Instructions for merging an incoming entity with an existing one."""

    existing_entity_id: str
    incoming_entity: dict
    merge_instructions: dict


@dataclass(frozen=True, slots=True)
class DeduplicationResult:
    """Result of entity deduplication."""

    entities: list[dict]
    merge_actions: list[MergeAction]
    entities_merged: int
    # old incoming id -> existing graph id, captured BEFORE the in-place
    # entity['id'] rewrite destroys the old id. The pipeline uses this to
    # remap relationship endpoints; without it, rels referencing a merged
    # id point at a node that does not exist and drop silently.
    id_renames: dict[str, str] = field(default_factory=dict)


class EntityDeduplicator:
    """Deduplicates extracted entities against existing graph entities.

    3-tier matching: exact ID -> alias -> embedding similarity (0.92 threshold).
    Same entity_type required for all tiers.
    """

    def __init__(
        self,
        executor: GraphExecutor,
        embedding_provider: EmbeddingProvider,
        confidence_manager: ConfidenceManager,
    ) -> None:
        self._executor = executor
        self._embedding_provider = embedding_provider
        self._confidence_manager = confidence_manager

    async def deduplicate(self, entities: list[dict]) -> DeduplicationResult:
        """Deduplicate entities against the graph.

        Args:
            entities: List of entity dicts from extraction pipeline.

        Returns:
            DeduplicationResult with deduplicated entities and merge actions.
        """
        if not entities:
            return DeduplicationResult(entities=[], merge_actions=[], entities_merged=0)

        result_entities: list[dict] = []
        merge_actions: list[MergeAction] = []
        id_renames: dict[str, str] = {}
        merged_count = 0

        for entity in entities:
            entity_id = entity.get("id", "")
            entity_type = entity.get("type", "")
            display_name = entity.get("name") or entity.get("display_name") or entity_id

            existing = await self._find_existing(entity_id, entity_type, display_name)

            if existing is not None:
                instructions = self._build_merge_instructions(existing, entity)
                merge_actions.append(
                    MergeAction(
                        existing_entity_id=existing["id"],
                        incoming_entity=entity,
                        merge_instructions=instructions,
                    )
                )
                # Capture the rename BEFORE the in-place rewrite (the rewrite
                # destroys the old id, and incoming_entity references the
                # same mutated dict).
                if existing["id"] != entity_id:
                    id_renames[entity_id] = existing["id"]
                # Rewrite entity ID to existing
                entity["id"] = existing["id"]
                merged_count += 1

            result_entities.append(entity)

        return DeduplicationResult(
            entities=result_entities,
            merge_actions=merge_actions,
            entities_merged=merged_count,
            id_renames=id_renames,
        )

    async def _find_existing(
        self, entity_id: str, entity_type: str, display_name: str
    ) -> dict | None:
        """Deterministic 3-tier resolver: exact id -> alias -> exact cosine.

        Every tier has a total order (ORDER BY id; the cosine tier breaks ties on
        id) so live and a replay rebuild make identical merge decisions on the
        same input -- no ANN, no insertion-order sensitivity.
        """
        _RET = (
            "RETURN e.id AS id, e.entity_type AS entity_type, "
            "e.display_name AS display_name, e.aliases AS aliases, "
            "e.description AS description, e.confidence AS confidence, "
            "e.source_type AS source_type"
        )

        # Tier 1: exact id (ORDER BY id makes case-fold collisions deterministic)
        results = await self._executor.execute_query(
            "MATCH (e:__Entity__) WHERE toLower(e.id) = $entity_id "
            f"AND e.entity_type = $entity_type {_RET} ORDER BY e.id ASC LIMIT 1",
            {"entity_id": entity_id.lower(), "entity_type": entity_type},
        )
        if results:
            return results[0]

        # Tier 2: alias (same total order)
        results = await self._executor.execute_query(
            "MATCH (e:__Entity__) WHERE $entity_id IN [a IN e.aliases | toLower(a)] "
            f"AND e.entity_type = $entity_type {_RET} ORDER BY e.id ASC LIMIT 1",
            {"entity_id": entity_id.lower(), "entity_type": entity_type},
        )
        if results:
            return results[0]

        # Tier 3: exact cosine over the widened-type candidate set. No ANN: a full
        # exact-cosine scan is a recall superset of HNSW top-k and is deterministic.
        # Probe embeds the display_name (the text the node's embedding is built
        # from), compared against stored node embeddings. Winner = argmax over
        # (score, id) as a total order.
        try:
            probe = await asyncio.get_running_loop().run_in_executor(
                None, self._embedding_provider.generate_embedding, display_name
            )
            results = await self._executor.execute_query(
                "MATCH (e:__Entity__) WHERE e.entity_type IN $types AND e.embedding IS NOT NULL "
                "WITH e, vector.similarity.cosine(e.embedding, $embedding) AS score "
                f"WHERE score >= $threshold {_RET}, score "
                "ORDER BY score DESC, e.id ASC LIMIT 1",
                {
                    "embedding": probe,
                    "threshold": SIMILARITY_THRESHOLD,
                    "types": dedup_type_filter(entity_type),
                },
            )
            if results:
                return results[0]
        except Exception:
            logger.debug("Cosine similarity search failed (expected if no embeddings present)")

        return None

    def _build_merge_instructions(self, existing: dict, incoming: dict) -> dict:
        """Determine how to merge incoming entity properties with existing.

        Args:
            existing: The existing entity from the graph.
            incoming: The incoming entity from extraction.

        Returns:
            Dict mapping field names to merge strategies.
        """
        instructions: dict[str, str] = {}

        # display_name: keep longer
        existing_name = existing.get("display_name") or ""
        incoming_name = incoming.get("name") or ""
        if len(incoming_name) > len(existing_name):
            instructions["display_name"] = "keep_incoming"
        else:
            instructions["display_name"] = "keep_existing"

        # description: keep longer
        existing_desc = existing.get("description") or ""
        incoming_desc = incoming.get("description") or ""
        if len(incoming_desc) > len(existing_desc):
            instructions["description"] = "keep_incoming"
        else:
            instructions["description"] = "keep_existing"

        # aliases: set union
        instructions["aliases"] = "merge"

        # confidence: reinforcement
        instructions["confidence"] = "reinforce"

        # entity_type: keep existing unless existing is unknown
        existing_type = existing.get("entity_type") or ""
        if existing_type.lower() == "unknown":
            instructions["entity_type"] = "keep_incoming"
        else:
            instructions["entity_type"] = "keep_existing"

        # source_type: keep existing unless incoming is stated or corrected
        incoming_source = incoming.get("source_type") or ""
        if incoming_source in ("stated", "corrected"):
            instructions["source_type"] = "keep_incoming"
        else:
            instructions["source_type"] = "keep_existing"

        # embedding: regenerate when display_name or description changed
        if (
            instructions["display_name"] == "keep_incoming"
            or instructions["description"] == "keep_incoming"
        ):
            instructions["embedding"] = "regenerate"
        else:
            instructions["embedding"] = "keep_existing"

        return instructions
