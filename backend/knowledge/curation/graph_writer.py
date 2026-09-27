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

# ON CREATE properties of a `new_fact` LearningEvent, shared by the
# conversational path (inside the `_upsert_entity` statement) and document
# ingest (`_create_new_fact_learning_event`), so the two cannot drift. Expects
# `le` bound by `MERGE (le:__Provenance__:LearningEvent {id: $learning_id})`.
_NEW_FACT_ON_CREATE = (
    "ON CREATE SET le.entity_type = 'LearningEvent', "
    "le.display_name = $learning_display_name, le.knowledge_domain = 'bridging', "
    "le.learning_type = 'new_fact', le.source_type = $source_type, "
    "le.created_at = $now, le.status = 'active'"
)


def _new_fact_learning_id(event_id: str, entity_id: str) -> str:
    """Deterministic id of the `new_fact` LearningEvent for one entity and event."""
    return f"learning-{event_id}-new_fact-{entity_id}"


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

    `backend/factories.py` constructs this at two sites (one per consumer --
    `build_curation_pipeline` and `build_extraction_pipeline`) from the same
    `KnowledgeConfig`; a cross-factory test
    (`tests/unit/test_factories_rebuild_stamps.py::TestCrossFactoryStampAgreement`)
    asserts the two outputs are `==`. That coverage relies on every field here
    keeping the dataclass default `compare=True` -- a field declared with
    `field(compare=False)` would be silently excluded from `__eq__` and could
    diverge between the two sites undetected.
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
        conversational = source_metadata is None
        for entity in entities:
            entity_id = entity.get("id", "")
            is_update = entity_id in merge_lookup
            # LearningEvent for new facts (first-time entity creation)
            new_fact = not is_update and (
                entity.get("source_type", "extracted") in ("stated", "corrected", "extracted")
            )
            # Conversational path: ONE statement per entity writes the entity,
            # its guarded reinforce, its EXTRACTED_FROM edge and (for a new
            # fact) its LearningEvent, so a kill cannot land between them.
            await self._upsert_entity(
                entity,
                now,
                event_id,
                session_id,
                with_conversation_provenance=conversational,
                with_new_fact_learning_event=new_fact and conversational,
            )
            if is_update:
                result.entities_updated += 1
            else:
                result.entities_created += 1
            if new_fact:
                if not conversational:
                    await self._create_new_fact_learning_event(
                        entity_id,
                        session_id,
                        event_id,
                        now,
                        entity.get("source_type", "extracted"),
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
                # Written by the `_upsert_entity` statement above.
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
        self,
        entity: dict,
        now: str,
        event_id: str,
        session_id: str,
        *,
        with_conversation_provenance: bool,
        with_new_fact_learning_event: bool,
    ) -> None:
        """MERGE an entity into the graph, reinforcing confidence once per event.

        With `with_conversation_provenance` (the conversational path), the same
        statement also MERGEs the entity's EXTRACTED_FROM edge to this session's
        ConversationContext (`_extracted_from_clause`), and with
        `with_new_fact_learning_event` the entity's `new_fact` LearningEvent
        with its LEARNED_FROM and ABOUT edges (`_NEW_FACT_ON_CREATE`). Each
        `execute_write` runs one statement in its own managed write transaction
        (`grep -n 'session.execute_write' backend/knowledge/storage/neo4j_connection.py`
        -> 116), so entity, reinforce, edge and LearningEvent commit together or
        not at all. The statement order:

        1. the replay guard (`OPTIONAL MATCH` + `count`), before anything is
           written, so it sees the graph as the previous statement left it;
        2. the entity MERGE with its ON CREATE / ON MATCH;
        3. the LearningEvent node MERGE;
        4. `MATCH` the ConversationContext, then the EXTRACTED_FROM MERGE and
           the LearningEvent's LEARNED_FROM and ABOUT edges.

        The ConversationContext `MATCH` sits AFTER the entity and LearningEvent
        node MERGEs on purpose. `write()` ensures the context one statement
        earlier, but were it absent, a `MATCH` ahead of the entity MERGE would
        yield no row and silently drop the entity. Placed after them, a missing
        context leaves what the separate statements this fold replaced left: the
        entity and the LearningEvent node, with no EXTRACTED_FROM, LEARNED_FROM
        or ABOUT edge (the old LearningEvent statement also `MATCH`ed the
        context before its ABOUT edge).

        Document ingest (`with_conversation_provenance` False) issues the entity
        statement alone, unchanged; `write()` then writes its SOURCED_FROM /
        chunk edges and `new_fact` LearningEvent as separate statements. That
        path is not replayed by the extraction backlog. The dispatcher applies
        through `apply_cached_turn`, called from `ExtractionDispatcher._apply`
        (`grep -n 'self._pipeline.apply_cached_turn' backend/extraction_backlog/dispatcher.py`),
        whose `curate_and_store` call passes event, session and
        `recorded_at` and no `source_metadata`
        (`grep -n 'recorded_at=turn.recorded_at' backend/knowledge/extraction/pipeline.py`
        -> 1206; the call opens at 1202).

        Replay guard (MIS-171, plan v2). The extraction backlog re-applies a
        turn whose curation a crash interrupted, so this statement can run
        twice for one event. Without a guard the second run takes the
        `ON MATCH` branch and raises `confidence` to `$reinforced` on an entity
        the first run CREATED at `$confidence`, which one apply never does.
        The statement therefore first looks for THIS event's EXTRACTED_FROM
        edge from this entity to this session's ConversationContext
        (`source_utterance_id = $event_id`) and, when it exists, leaves
        `e.confidence` unchanged: this event has already reinforced (or
        created) the entity. The lookup is an `OPTIONAL MATCH` in the same
        statement as the MERGE, not a separate read.

        Why `source_utterance_id` identifies "this event". The edge carries no
        append-only per-event record -- its properties are
        `source_utterance_id`, `created_at`/`updated_at`, `status`, the three
        epoch stamps and `derived_at` (`_extracted_from_clause`) -- and
        `source_utterance_id` is last-writer-wins: it is set on both ON CREATE
        and ON MATCH, so a later turn overwrites it. Matching on it is still
        sound for crash replay because the backlog re-applies turn N before any
        turn that sorts after N:
        - the dispatcher processes only the backlog head, taken in
          `ExtractionDispatcher._step`:
          `grep -n 'head = scan.head' backend/extraction_backlog/dispatcher.py`;
        - the head is the first pending turn in replay order:
          `grep -n 'return self.pending.0. if' backend/extraction_backlog/store.py`
          -> 217 (`BacklogScan.head`);
        - only an `applied` marker takes a turn out of the pending list, so a
          crashed turn N stays pending and ahead of every later turn:
          `grep -n 'if stage == STAGE_APPLIED' backend/extraction_backlog/store.py`
          -> 475.

        Why the edge MERGE is in this statement (plan v2 follow-up). When the
        edge was a separate statement after this one, a kill between the two
        left the entity reinforced (or created) with no edge for this event, so
        the replay's guard found nothing and reinforced again. Now the guard's
        evidence (the edge) commits in the same transaction as the reinforce it
        suppresses: after any kill, either both are in the graph or neither is.

        Why the `new_fact` LearningEvent is in this statement too. `write()`
        writes it only for an entity dedup did NOT map onto an existing node
        (`grep -n 'is_update = entity_id in merge_lookup' backend/knowledge/curation/graph_writer.py`
        -> 186, plus this citation's own line). On a replay, dedup finds the entity the crashed run created,
        by exact id first
        (`grep -n 'existing = await self._find_existing' backend/knowledge/curation/deduplication.py`
        -> 84), and emits a MergeAction for it, so the replay never writes the
        LearningEvent. As a separate statement, a kill after the entity and
        before the LearningEvent therefore lost it for good. In this statement,
        the LearningEvent exists whenever the entity this event created does.

        What the guard does NOT cover:
        - A turn that sorts BEFORE N but is logged after N crashed becomes the
          head first and overwrites `source_utterance_id` before N replays.
        - Document ingest (`source_metadata` set): it writes SOURCED_FROM, not
          EXTRACTED_FROM, so the guard never matches and behaviour is
          unchanged there. Its separate statements keep the kill windows this
          fold closes on the conversational path.
        For an entity that already existed before this event, a second
        reinforce is harmless anyway: `$reinforced` is computed from the
        incoming `confidence` param alone
        (`ConfidenceManager.reinforced_confidence(confidence, domain)` below),
        and `max(max(c, r), r) == max(c, r)`. Only the create-then-match
        transition diverges, and that is what the guard closes.

        Only the confidence reinforce is guarded. `updated_at`, `display_name`
        and `description` keep their semantics on every `ON MATCH`:
        `display_name` and `description` are longest-wins, which is already
        idempotent for the same input, and `updated_at` is an audit timestamp
        that differs on every run regardless. `canonical_graph_form` compares
        neither: `updated_at` is an audit field
        (`grep -n '"updated_at",' backend/knowledge/canonical_serialize.py` -> 29,
        inside `AUDIT_FIELDS`), and node `confidence` is excluded
        (`grep -n 'NODE_ONLY_EXCLUDED_FIELDS = ' backend/knowledge/canonical_serialize.py`
        -> 71). That exclusion is why the canonical-form crash test cannot see
        this defect and a direct confidence assertion is needed.
        """
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
        query = (
            # Replay guard: evaluated BEFORE the MERGE, in the same statement.
            # count() makes exactly one row whether or not the edge exists.
            "OPTIONAL MATCH (:__Entity__ {id: $entity_id})-[seen:EXTRACTED_FROM]->"
            "(:ConversationContext {conversation_id: $session_id}) "
            "WHERE seen.source_utterance_id = $event_id "
            "WITH count(seen) > 0 AS event_already_applied "
            "MERGE (e:__Entity__ {id: $entity_id}) "
            "ON CREATE SET e.entity_type = $entity_type, e.display_name = $display_name, "
            "e.knowledge_domain = $domain, e.confidence = $confidence, "
            "e.source_type = $source_type, e.created_at = $now, e.updated_at = $now, "
            "e.ontology_version = $ontology_version, e.embedding = $embedding, "
            "e.description = $description, e.aliases = $aliases, e.status = 'active', "
            "e.provenance = 'extraction' "
            "ON MATCH SET e.confidence = CASE WHEN event_already_applied THEN e.confidence "
            "WHEN e.confidence < $reinforced THEN $reinforced ELSE e.confidence END, "
            "e.updated_at = $now, "
            "e.display_name = CASE WHEN size(e.display_name) < size($display_name) "
            "THEN $display_name ELSE e.display_name END, "
            "e.description = CASE WHEN size(coalesce(e.description, '')) < size($description) "
            "THEN $description ELSE e.description END" + user_label_set
        )
        params: dict = {
            "entity_id": entity_id,
            "event_id": event_id,
            "session_id": session_id,
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
        }
        if with_conversation_provenance:
            if with_new_fact_learning_event:
                # The entity's own source_type is the LearningEvent's
                # ($source_type), as in `_create_new_fact_learning_event`.
                query += (
                    " WITH e "
                    "MERGE (le:__Provenance__:LearningEvent {id: $learning_id}) "
                    + _NEW_FACT_ON_CREATE
                    + " WITH e, le"
                )
                params["learning_id"] = _new_fact_learning_id(event_id, entity_id)
                params["learning_display_name"] = f"new_fact: {entity_id}"
            else:
                query += " WITH e"
            edge_clause, edge_params = self._extracted_from_clause()
            query += " MATCH (ctx:ConversationContext {conversation_id: $session_id}) " + (
                edge_clause
            )
            params.update(edge_params)
            if with_new_fact_learning_event:
                # Same order as `_create_new_fact_learning_event`.
                query += " MERGE (le)-[:LEARNED_FROM]->(ctx) MERGE (le)-[:ABOUT]->(e)"
        await self._executor.execute_write(query, params)

    def _extracted_from_clause(self) -> tuple[str, dict[str, str]]:
        """Build the EXTRACTED_FROM MERGE that anchors an entity to its utterance.

        Returns a (cypher_fragment, params) pair. The fragment expects `e` (the
        entity) and `ctx` (this session's ConversationContext) bound, and uses
        `$event_id` and `$now` from the enclosing `_upsert_entity` statement;
        the params are the epoch stamps. It is written inside that statement,
        not on its own, so the edge commits atomically with the entity and its
        guarded reinforce (see `_upsert_entity`).

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
            "ontology_version": self._rebuild_stamps.ontology_version,
            "extraction_version": self._rebuild_stamps.extraction_version,
            "model_hash": self._rebuild_stamps.model_hash,
        }
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

        return (
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
        """Create a LearningEvent for a newly created entity (new_fact).

        Document ingest only. On the conversational path `_upsert_entity`
        writes the same LearningEvent (`_NEW_FACT_ON_CREATE`) inside the
        entity statement.
        """
        learning_id = _new_fact_learning_id(event_id, entity_id)
        learned_clause, learned_params = self._learned_from_clause(source_metadata, session_id)
        await self._executor.execute_write(
            "MERGE (le:__Provenance__:LearningEvent {id: $learning_id}) "
            + _NEW_FACT_ON_CREATE
            + " WITH le "
            + learned_clause
            + "WITH le "
            "MATCH (target:__Entity__ {id: $entity_id}) "
            "MERGE (le)-[:ABOUT]->(target)",
            {
                "learning_id": learning_id,
                "learning_display_name": f"new_fact: {entity_id}",
                "source_type": source_type,
                "now": now,
                "entity_id": entity_id,
                **learned_params,
            },
        )
