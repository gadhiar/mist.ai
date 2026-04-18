"""Curated knowledge graph writer with provenance tracking.

Stage 8: Writes deduplicated, conflict-resolved entities and relationships
to Neo4j using MERGE semantics. Creates ConversationContext provenance
anchors, EXTRACTED_FROM edges, and LearningEvent entities.
"""

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime

from backend.interfaces import EmbeddingProvider
from backend.knowledge.curation.confidence import ConfidenceManager
from backend.knowledge.curation.conflict_resolver import SupersessionAction
from backend.knowledge.curation.deduplication import MergeAction
from backend.knowledge.storage.graph_executor import GraphExecutor

logger = logging.getLogger(__name__)

PROPERTY_KEY_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


@dataclass(frozen=True, slots=True)
class SourceMetadata:
    """Metadata for external (non-conversation) knowledge sources.

    When provided to `CurationGraphWriter.write`, provenance edges target
    an ExternalSource node instead of a ConversationContext node.

    Attributes:
        source_uri: Unique URI identifying the external source.
        source_type: Category of the source (document, mcp, web, etc.).
        title: Optional human-readable title for the source.
        chunk_ids: Optional vector-store chunk IDs associated with the source.
        synthesis: When True, entity-to-chunk edges use DERIVED_FROM instead
            of REFERENCES (indicates LLM synthesis rather than direct extraction).
    """

    source_uri: str
    source_type: str
    title: str | None = None
    chunk_ids: list[str] | None = None
    synthesis: bool = False


@dataclass(slots=True)
class WriteResult:
    """Counts of graph write operations performed."""

    entities_created: int = 0
    entities_updated: int = 0
    relationships_created: int = 0
    relationships_updated: int = 0
    relationships_superseded: int = 0
    learning_events_created: int = 0
    provenance_edges_created: int = 0
    source_nodes_created: int = 0
    document_provenance_edges: int = 0


class CurationGraphWriter:
    """Writes curated entities and relationships to Neo4j.

    Uses MERGE for idempotent upserts. Creates ConversationContext
    provenance anchors and EXTRACTED_FROM edges for every entity.
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

    async def write(
        self,
        entities: list[dict],
        relationships: list[dict],
        merge_actions: list[MergeAction],
        supersession_actions: list[SupersessionAction],
        event_id: str,
        session_id: str,
        source_metadata: SourceMetadata | None = None,
    ) -> WriteResult:
        """Write curated knowledge to the graph with provenance.

        Args:
            entities: Deduplicated entity list.
            relationships: Conflict-resolved relationship list.
            merge_actions: Merge instructions from deduplication.
            supersession_actions: Supersession instructions from conflict resolution.
            event_id: Source event ID for provenance.
            session_id: Conversation session ID.
            source_metadata: Optional external source metadata. When provided,
                provenance targets an ExternalSource node instead of
                ConversationContext.

        Returns:
            WriteResult with operation counts.
        """
        if not entities and not relationships and not supersession_actions:
            return WriteResult()

        result = WriteResult()
        now = datetime.now(UTC).isoformat()

        # Create/update provenance anchor (external source or conversation)
        if entities or relationships:
            if source_metadata is not None:
                await self._ensure_external_source(source_metadata, now)
                result.source_nodes_created += 1
                if source_metadata.chunk_ids:
                    await self._ensure_vector_chunks(
                        source_metadata.chunk_ids, source_metadata.source_uri, now
                    )
            else:
                await self._ensure_conversation_context(session_id, now)

        # Upsert entities
        merge_lookup = {a.existing_entity_id: a for a in merge_actions}
        for entity in entities:
            entity_id = entity.get("id", "")
            is_update = entity_id in merge_lookup
            await self._upsert_entity(entity, merge_lookup.get(entity_id), now, event_id)
            if is_update:
                result.entities_updated += 1
            else:
                result.entities_created += 1
                # LearningEvent for new facts (first-time entity creation)
                source_type = entity.get("source_type", "extracted")
                if source_type in ("stated", "corrected", "extracted"):
                    await self._create_new_fact_learning_event(
                        entity_id,
                        session_id,
                        event_id,
                        now,
                        source_type,
                        source_metadata=source_metadata,
                    )
                    result.learning_events_created += 1

            # Provenance edges
            if source_metadata is not None:
                edges = await self._create_document_provenance(
                    entity_id, source_metadata, event_id, now
                )
                result.document_provenance_edges += edges
            else:
                await self._create_provenance_edge(entity_id, session_id, event_id, now)
                result.provenance_edges_created += 1

        if source_metadata is not None and result.document_provenance_edges > 0:
            logger.info(
                "Tagged %d entities with source provenance: %s (%s)",
                result.document_provenance_edges,
                source_metadata.source_uri,
                source_metadata.source_type,
            )

        # Upsert relationships
        for rel in relationships:
            await self._upsert_relationship(rel, now, event_id)
            result.relationships_created += 1

        # Apply supersession actions
        for action in supersession_actions:
            await self._apply_supersession(action, now)
            result.relationships_superseded += 1

            # LearningEvent for contradictions, progressions, and corrections
            if action.reason in ("contradiction", "progression", "correction"):
                await self._create_learning_event(
                    action,
                    session_id,
                    event_id,
                    now,
                    source_metadata=source_metadata,
                )
                result.learning_events_created += 1

        return result

    async def _ensure_conversation_context(self, session_id: str, now: str) -> None:
        """Create or update the ConversationContext provenance node."""
        await self._executor.execute_write(
            "MERGE (ctx:__Provenance__:ConversationContext {conversation_id: $session_id}) "
            "ON CREATE SET ctx.id = $session_id, ctx.entity_type = 'ConversationContext', "
            "ctx.created_at = $now, ctx.updated_at = $now, ctx.status = 'active' "
            "ON MATCH SET ctx.updated_at = $now",
            {"session_id": session_id, "now": now},
        )

    async def _upsert_entity(
        self, entity: dict, merge_action: MergeAction | None, now: str, event_id: str
    ) -> None:
        """MERGE an entity into the graph."""
        entity_id = entity.get("id", "")
        entity_type = entity.get("type", "")
        display_name = entity.get("name", entity_id)
        confidence = entity.get("confidence", 0.8)
        source_type = entity.get("source_type", "extracted")
        aliases = entity.get("aliases") or []
        description = entity.get("description") or ""
        domain = self._confidence_manager.determine_domain(entity_type)

        # Generate embedding
        embedding = entity.get("embedding")
        if embedding is None:
            embedding = self._embedding_provider.generate_embedding(display_name)

        await self._executor.execute_write(
            "MERGE (e:__Entity__ {id: $entity_id}) "
            "ON CREATE SET e.entity_type = $entity_type, e.display_name = $display_name, "
            "e.knowledge_domain = $domain, e.confidence = $confidence, "
            "e.source_type = $source_type, e.created_at = $now, e.updated_at = $now, "
            "e.ontology_version = '1.0.0', e.embedding = $embedding, "
            "e.description = $description, e.aliases = $aliases, e.status = 'active' "
            "ON MATCH SET e.confidence = CASE WHEN e.confidence < $reinforced "
            "THEN $reinforced ELSE e.confidence END, "
            "e.updated_at = $now, "
            "e.display_name = CASE WHEN size(e.display_name) < size($display_name) "
            "THEN $display_name ELSE e.display_name END, "
            "e.description = CASE WHEN size(coalesce(e.description, '')) < size($description) "
            "THEN $description ELSE e.description END",
            {
                "entity_id": entity_id,
                "entity_type": entity_type,
                "display_name": display_name,
                "domain": domain.value,
                "confidence": confidence,
                "reinforced": self._confidence_manager.reinforced_confidence(confidence, domain),
                "source_type": source_type,
                "now": now,
                "embedding": embedding,
                "description": description,
                "aliases": aliases,
            },
        )

    async def _upsert_relationship(self, rel: dict, now: str, event_id: str) -> None:
        """MERGE a relationship into the graph."""
        source = rel.get("source", "")
        target = rel.get("target", "")
        rel_type = rel.get("type", "")
        confidence = rel.get("confidence", 0.8)
        source_type = rel.get("source_type", "extracted")
        temporal_status = rel.get("temporal_status", "current")
        context = rel.get("context", "")

        # Sanitize rel_type for use in Cypher
        sanitized_type = re.sub(r"[^A-Z_]", "", rel_type.upper())
        if not sanitized_type:
            logger.warning("Invalid relationship type '%s', skipping", rel_type)
            return

        await self._executor.execute_write(
            f"MATCH (s:__Entity__ {{id: $source}}) "
            f"MATCH (t:__Entity__ {{id: $target}}) "
            f"MERGE (s)-[r:{sanitized_type}]->(t) "
            "ON CREATE SET r.confidence = $confidence, r.source_event_id = $event_id, "
            "r.source_type = $source_type, r.created_at = $now, r.updated_at = $now, "
            "r.ontology_version = '1.0.0', r.status = 'active', "
            "r.evidence = [$event_id], r.temporal_status = $temporal_status, "
            "r.context = $context "
            "ON MATCH SET r.confidence = $confidence, r.updated_at = $now, "
            "r.evidence = r.evidence + [$event_id]",
            {
                "source": source,
                "target": target,
                "confidence": confidence,
                "event_id": event_id,
                "source_type": source_type,
                "now": now,
                "temporal_status": temporal_status,
                "context": context,
            },
        )

    async def _create_provenance_edge(
        self, entity_id: str, session_id: str, event_id: str, now: str
    ) -> None:
        """Create EXTRACTED_FROM edge between entity and ConversationContext."""
        await self._executor.execute_write(
            "MATCH (e:__Entity__ {id: $entity_id}) "
            "MATCH (ctx:ConversationContext {conversation_id: $session_id}) "
            "MERGE (e)-[r:EXTRACTED_FROM]->(ctx) "
            "ON CREATE SET r.event_id = $event_id, r.created_at = $now "
            "ON MATCH SET r.event_id = $event_id",
            {"entity_id": entity_id, "session_id": session_id, "event_id": event_id, "now": now},
        )

    async def _ensure_external_source(self, source_metadata: SourceMetadata, now: str) -> None:
        """Create or update an ExternalSource provenance node."""
        await self._executor.execute_write(
            "MERGE (es:ExternalSource {source_uri: $source_uri}) "
            "ON CREATE SET es.source_type = $source_type, es.created_at = $now, "
            "es.title = $title, es.status = 'active' "
            "ON MATCH SET es.updated_at = $now",
            {
                "source_uri": source_metadata.source_uri,
                "source_type": source_metadata.source_type,
                "now": now,
                "title": source_metadata.title,
            },
        )

    async def _ensure_vector_chunks(self, chunk_ids: list[str], source_uri: str, now: str) -> None:
        """Create or update VectorChunk nodes linked to an ExternalSource."""
        await self._executor.execute_write(
            "UNWIND $chunk_ids AS cid "
            "MERGE (vc:VectorChunk {vector_store_id: cid}) "
            "ON CREATE SET vc.source_id = $source_uri, vc.created_at = $now "
            "ON MATCH SET vc.updated_at = $now",
            {"chunk_ids": chunk_ids, "source_uri": source_uri, "now": now},
        )

    async def _create_document_provenance(
        self,
        entity_id: str,
        source_metadata: SourceMetadata,
        event_id: str,
        now: str,
    ) -> int:
        """Create provenance edges from an entity to its external source and chunks.

        Always creates a SOURCED_FROM edge to the ExternalSource. When chunk_ids
        are present, creates REFERENCES edges (direct extraction) or DERIVED_FROM
        edges (LLM synthesis) to VectorChunk nodes.

        Returns:
            Count of provenance edges created.
        """
        edges = 0

        # SOURCED_FROM -> ExternalSource
        await self._executor.execute_write(
            "MATCH (e:__Entity__ {id: $entity_id}) "
            "MATCH (es:ExternalSource {source_uri: $source_uri}) "
            "MERGE (e)-[r:SOURCED_FROM]->(es) "
            "ON CREATE SET r.event_id = $event_id, r.created_at = $now "
            "ON MATCH SET r.event_id = $event_id, r.updated_at = $now",
            {
                "entity_id": entity_id,
                "source_uri": source_metadata.source_uri,
                "event_id": event_id,
                "now": now,
            },
        )
        edges += 1

        # Chunk-level provenance
        if source_metadata.chunk_ids:
            rel_type = "DERIVED_FROM" if source_metadata.synthesis else "REFERENCES"
            await self._executor.execute_write(
                "UNWIND $chunk_ids AS cid "
                f"MATCH (e:__Entity__ {{id: $entity_id}}) "
                f"MATCH (vc:VectorChunk {{vector_store_id: cid}}) "
                f"MERGE (e)-[r:{rel_type}]->(vc) "
                "ON CREATE SET r.event_id = $event_id, r.created_at = $now "
                "ON MATCH SET r.event_id = $event_id, r.updated_at = $now",
                {
                    "entity_id": entity_id,
                    "chunk_ids": source_metadata.chunk_ids,
                    "event_id": event_id,
                    "now": now,
                },
            )
            edges += len(source_metadata.chunk_ids)

        return edges

    async def _apply_supersession(self, action: SupersessionAction, now: str) -> None:
        """Mark an existing relationship as superseded."""
        sanitized_type = re.sub(r"[^A-Z_]", "", action.old_rel_type.upper())
        await self._executor.execute_write(
            f"MATCH (s:__Entity__)-[r:{sanitized_type}]->(t:__Entity__ {{id: $old_target}}) "
            "WHERE r.status = 'active' "
            "SET r.status = 'superseded', r.updated_at = $now, "
            "r.superseded_by = $new_target, r.supersession_reason = $reason",
            {
                "old_target": action.old_target_id,
                "new_target": action.new_target_id,
                "now": now,
                "reason": action.reason,
            },
        )

    def _learned_from_clause(
        self, source_metadata: SourceMetadata | None, session_id: str
    ) -> tuple[str, dict]:
        """Build the LEARNED_FROM MATCH/MERGE clause and params.

        Returns:
            A (cypher_fragment, params_dict) tuple. The fragment expects to
            start after a ``WITH le`` clause.
        """
        if source_metadata is not None:
            return (
                "MATCH (src:ExternalSource {source_uri: $source_uri}) "
                "MERGE (le)-[:LEARNED_FROM]->(src) ",
                {"source_uri": source_metadata.source_uri},
            )
        return (
            "MATCH (ctx:ConversationContext {conversation_id: $session_id}) "
            "MERGE (le)-[:LEARNED_FROM]->(ctx) ",
            {"session_id": session_id},
        )

    async def _create_learning_event(
        self,
        action: SupersessionAction,
        session_id: str,
        event_id: str,
        now: str,
        source_metadata: SourceMetadata | None = None,
    ) -> None:
        """Create a LearningEvent entity for a contradiction or progression."""
        learning_id = f"learning-{event_id}-{action.old_rel_type}-{action.old_target_id}"
        learned_clause, learned_params = self._learned_from_clause(source_metadata, session_id)
        await self._executor.execute_write(
            "MERGE (le:__Entity__:LearningEvent {id: $learning_id}) "
            "ON CREATE SET le.entity_type = 'LearningEvent', "
            "le.display_name = $display_name, le.knowledge_domain = 'bridging', "
            "le.learning_type = $reason, le.old_relationship = $old_rel, "
            "le.old_target = $old_target, le.new_target = $new_target, "
            "le.created_at = $now, le.status = 'active' "
            "WITH le " + learned_clause + "WITH le "
            "MATCH (target:__Entity__ {id: $about_target}) "
            "MERGE (le)-[:ABOUT]->(target)",
            {
                "learning_id": learning_id,
                "display_name": f"{action.reason}: {action.old_rel_type} "
                f"{action.old_target_id} -> {action.new_target_id}",
                "reason": action.reason,
                "old_rel": action.old_rel_type,
                "old_target": action.old_target_id,
                "new_target": action.new_target_id,
                "now": now,
                "about_target": action.new_target_id,
                **learned_params,
            },
        )

    async def _create_new_fact_learning_event(
        self,
        entity_id: str,
        session_id: str,
        event_id: str,
        now: str,
        source_type: str,
        source_metadata: SourceMetadata | None = None,
    ) -> None:
        """Create a LearningEvent for a newly created entity (new_fact)."""
        learning_id = f"learning-{event_id}-new_fact-{entity_id}"
        learned_clause, learned_params = self._learned_from_clause(source_metadata, session_id)
        await self._executor.execute_write(
            "MERGE (le:__Entity__:LearningEvent {id: $learning_id}) "
            "ON CREATE SET le.entity_type = 'LearningEvent', "
            "le.display_name = $display_name, le.knowledge_domain = 'bridging', "
            "le.learning_type = 'new_fact', le.source_type = $source_type, "
            "le.created_at = $now, le.status = 'active' "
            "WITH le " + learned_clause + "WITH le "
            "MATCH (target:__Entity__ {id: $entity_id}) "
            "MERGE (le)-[:ABOUT]->(target)",
            {
                "learning_id": learning_id,
                "display_name": f"new_fact: {entity_id}",
                "source_type": source_type,
                "now": now,
                "entity_id": entity_id,
                **learned_params,
            },
        )
