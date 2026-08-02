"""Curated knowledge graph writer with provenance tracking.

Stage 8 (entities): Writes deduplicated entities to Neo4j using MERGE
semantics. Creates ConversationContext provenance anchors, EXTRACTED_FROM
edges, and LearningEvent entities. Relationship writes moved to the
bitemporal ReconciliationEngine at the C2 cutover (curation/reconciliation.py).
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime

from backend.interfaces import EmbeddingProvider
from backend.knowledge.curation.confidence import ConfidenceManager
from backend.knowledge.curation.deduplication import MergeAction
from backend.knowledge.storage.graph_executor import GraphExecutor

logger = logging.getLogger(__name__)

PROPERTY_KEY_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


@dataclass(frozen=True, slots=True)
class RebuildStamps:
    """Per-deployment rebuild-determinism stamps for EXTRACTED_FROM edges and
    reconciled fact edges (reconciliation.py stamps both from this object).

    ADR-010 "Rebuild Determinism Model" requires every entity-provenance edge
    to carry the ontology, extraction-prompt, and model identifiers that were
    active when the entity was extracted. R1.3 moved this anchor from
    DERIVED_FROM->VaultNote onto EXTRACTED_FROM->ConversationContext, and the
    stamps' purpose is unchanged: they let a future consumer detect drift
    against current config values. `mist_admin vault-rebuild` no longer reads
    them -- R1.3 (Task 8) made it a sidecar-only reindex with no graph-side
    comparison; drift consumption is not wired to any command today.

    Stable for the lifetime of the writer -- the LLM binary and ontology
    version do not change mid-process. Constructed from `KnowledgeConfig`
    in the factory and injected into `CurationGraphWriter` as a required
    dependency; `ontology_version` and `extraction_version` trace back to
    `backend.knowledge.version_stamps`, the single authority for both.
    """

    ontology_version: str
    extraction_version: str
    model_hash: str


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
    """Counts of graph write operations performed.

    Relationship counts live on ReconcileTurnResult since the C2 cutover.
    """

    entities_created: int = 0
    entities_updated: int = 0
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
        rebuild_stamps: RebuildStamps,
    ) -> None:
        self._executor = executor
        self._embedding_provider = embedding_provider
        self._confidence_manager = confidence_manager
        # ADR-010 Phase 8, re-anchored by R1.3: stamps ride every entity write
        # and every EXTRACTED_FROM edge. Required, not optional: `ontology_version`
        # is a required universal entity property, so a writer without stamps
        # could only ever emit a wrong-by-default literal (it used to emit
        # "1.2.1") or an invalid entity. `build_curation_pipeline` injects it.
        self._rebuild_stamps = rebuild_stamps

    async def write(
        self,
        entities: list[dict],
        merge_actions: list[MergeAction],
        event_id: str,
        session_id: str,
        source_metadata: SourceMetadata | None = None,
    ) -> WriteResult:
        """Write curated entities to the graph with provenance.

        Relationships are written by the ReconciliationEngine (C2 cutover),
        not here.

        Args:
            entities: Deduplicated entity list.
            merge_actions: Merge instructions from deduplication.
            event_id: Source event ID for provenance.
            session_id: Conversation session ID.
            source_metadata: Optional external source metadata. When provided,
                provenance targets an ExternalSource node instead of
                ConversationContext.

        Returns:
            WriteResult with operation counts.
        """
        if not entities:
            return WriteResult()

        result = WriteResult()
        now = datetime.now(UTC).isoformat()

        # Create/update provenance anchor (external source or conversation)
        if entities:
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

        # Generate embedding off-loop: model.encode blocks ~10ms warm
        # (seconds cold) and this runs under the curation write lock.
        embedding = entity.get("embedding")
        if embedding is None:
            embedding = await asyncio.get_running_loop().run_in_executor(
                None, self._embedding_provider.generate_embedding, display_name
            )

        # :User label is an invariant of the user node (persona/identity
        # reads anchor on it); the extraction path must stamp it, not just
        # seed/rebuild (deep review cypher-data-integrity-2a). Idempotent.
        user_label_set = " SET e:User" if entity_id == "user" else ""
        await self._executor.execute_write(
            "MERGE (e:__Entity__ {id: $entity_id}) "
            "ON CREATE SET e.entity_type = $entity_type, e.display_name = $display_name, "
            "e.knowledge_domain = $domain, e.confidence = $confidence, "
            "e.source_type = $source_type, e.created_at = $now, e.updated_at = $now, "
            "e.ontology_version = $ontology_version, e.embedding = $embedding, "
            "e.description = $description, e.aliases = $aliases, e.status = 'active', "
            "e.provenance = 'extraction' "
            "ON MATCH SET e.confidence = CASE WHEN e.confidence < $reinforced "
            "THEN $reinforced ELSE e.confidence END, "
            "e.updated_at = $now, "
            "e.display_name = CASE WHEN size(e.display_name) < size($display_name) "
            "THEN $display_name ELSE e.display_name END, "
            "e.description = CASE WHEN size(coalesce(e.description, '')) < size($description) "
            "THEN $description ELSE e.description END" + user_label_set,
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
                # 4.7 drift fix: stamped from config via RebuildStamps, no
                # hardcoded version literal.
                "ontology_version": self._rebuild_stamps.ontology_version,
            },
        )

    async def _create_provenance_edge(
        self, entity_id: str, session_id: str, event_id: str, now: str
    ) -> None:
        """Anchor an entity to the utterance it was extracted from.

        R1.3: this is the sole entity-level provenance anchor on the
        conversational path. `source_utterance_id` names the MOST RECENT
        utterance in this session that produced (or re-produced, via
        re-extraction) the entity: the edge MERGEs on (entity,
        ConversationContext) and the property is set on both ON CREATE and
        ON MATCH, so a later turn's re-extraction overwrites it --
        last-writer-wins, not append-only.

        This is NOT the same guarantee as the identically-named property C2
        stamps on reconciled relationship edges (`reconciliation.py`), which
        MERGEs on `{version_key: $vk}` and sets the property ON CREATE only,
        pinning it permanently to the originating utterance. Do not assume
        the two are interchangeable for provenance tracing back to a single
        log row. The vault is not a fact source under Inv-A1, so no
        `DERIVED_FROM -> VaultNote` edge is written.

        Epoch stamps always ride this edge (`rebuild_stamps` is a required
        constructor dependency), keeping the per-turn (ontology, extraction,
        model) triple auditable in the graph now that the VaultNote anchor that
        used to carry them is retired.
        """
        params: dict[str, str] = {
            "entity_id": entity_id,
            "session_id": session_id,
            "event_id": event_id,
            "now": now,
        }
        params["ontology_version"] = self._rebuild_stamps.ontology_version
        params["extraction_version"] = self._rebuild_stamps.extraction_version
        params["model_hash"] = self._rebuild_stamps.model_hash
        stamp_clause = (
            ", r.ontology_version = $ontology_version"
            ", r.extraction_version = $extraction_version"
            ", r.model_hash = $model_hash"
            ", r.derived_at = $now"
        )
        create_set = (
            "r.source_utterance_id = $event_id, r.created_at = $now, "
            "r.status = 'active'" + stamp_clause
        )
        match_set = (
            "r.source_utterance_id = $event_id, r.updated_at = $now, "
            "r.status = 'active'" + stamp_clause
        )

        await self._executor.execute_write(
            "MATCH (e:__Entity__ {id: $entity_id}) "
            "MATCH (ctx:ConversationContext {conversation_id: $session_id}) "
            "MERGE (e)-[r:EXTRACTED_FROM]->(ctx) "
            f"ON CREATE SET {create_set} "
            f"ON MATCH SET {match_set}",
            params,
        )

    async def _ensure_external_source(self, source_metadata: SourceMetadata, now: str) -> None:
        """Create or update an ExternalSource provenance node."""
        await self._executor.execute_write(
            "MERGE (es:__Provenance__:ExternalSource {source_uri: $source_uri}) "
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
            "MERGE (vc:__Provenance__:VectorChunk {vector_store_id: cid}) "
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

    async def create_belief_change_learning_event(
        self,
        reason: str,
        predicate: str,
        old_target_id: str,
        session_id: str,
        event_id: str,
        now: str,
        source_metadata: SourceMetadata | None = None,
    ) -> None:
        """Create a LearningEvent for a reconciliation belief change (C2).

        Called by CurationPipeline for every close-bearing engine action
        (single_supersession / contradiction / progression / cease / retract).
        Unlike the legacy supersession variant, ABOUT points at the OLD
        (closed) target -- close actions do not carry the superseding target.
        """
        learning_id = f"learning-{event_id}-{predicate}-{old_target_id}"
        learned_clause, learned_params = self._learned_from_clause(source_metadata, session_id)
        await self._executor.execute_write(
            "MERGE (le:__Provenance__:LearningEvent {id: $learning_id}) "
            "ON CREATE SET le.entity_type = 'LearningEvent', "
            "le.display_name = $display_name, le.knowledge_domain = 'bridging', "
            "le.learning_type = $reason, le.old_relationship = $predicate, "
            "le.old_target = $old_target, "
            "le.created_at = $now, le.status = 'active' "
            "WITH le " + learned_clause + "WITH le "
            "MATCH (target:__Entity__ {id: $old_target}) "
            "MERGE (le)-[:ABOUT]->(target)",
            {
                "learning_id": learning_id,
                "display_name": f"{reason}: {predicate} {old_target_id}",
                "reason": reason,
                "predicate": predicate,
                "old_target": old_target_id,
                "now": now,
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
            "MERGE (le:__Provenance__:LearningEvent {id: $learning_id}) "
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
