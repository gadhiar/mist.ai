"""Curation pipeline orchestrator.

Stages 7-8: Coordinates deduplication, conflict resolution, and graph
writing. Called by ExtractionPipeline after Stage 6 validation when
curation is enabled.
"""

import logging
import time
from dataclasses import dataclass, field

from backend.knowledge.curation.conflict_resolver import ConflictResolutionResult, ConflictResolver
from backend.knowledge.curation.deduplication import DeduplicationResult, EntityDeduplicator
from backend.knowledge.curation.graph_writer import CurationGraphWriter, SourceMetadata, WriteResult
from backend.knowledge.extraction.validator import ValidationResult

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CurationResult:
    """Combined result of the curation pipeline."""

    write_result: WriteResult
    dedup_result: DeduplicationResult
    conflict_result: ConflictResolutionResult
    curation_time_ms: float
    stage_errors: list[str] = field(default_factory=list)


class CurationPipeline:
    """Orchestrates stages 7-8 of the knowledge pipeline.

    Stage 7a: EntityDeduplicator -- match against existing graph entities.
    Stage 7b: ConflictResolver -- detect supersessions and contradictions.
    Stage 8:  CurationGraphWriter -- MERGE entities/relationships + provenance.
    """

    def __init__(
        self,
        deduplicator: EntityDeduplicator,
        conflict_resolver: ConflictResolver,
        graph_writer: CurationGraphWriter,
    ) -> None:
        self._deduplicator = deduplicator
        self._conflict_resolver = conflict_resolver
        self._graph_writer = graph_writer

    async def curate_and_store(
        self,
        validation_result: ValidationResult,
        event_id: str,
        session_id: str,
        source_metadata: SourceMetadata | None = None,
        vault_note_path: str | None = None,
    ) -> CurationResult:
        """Run curation stages and write to graph.

        Short-circuits on empty entities. Each stage wrapped in
        try/except -- failures are logged and pipeline continues with
        best-effort data. No rollback -- MERGE is idempotent, partial
        writes are recoverable via regeneration.

        Args:
            validation_result: Output of Stage 6 validation.
            event_id: Source event ID for provenance.
            session_id: Conversation session ID.
            source_metadata: Optional external source metadata. Forwarded
                to `CurationGraphWriter.write` for document provenance.
            vault_note_path: Optional vault session-note path (ADR-010
                Cluster 8 Phase 6). Forwarded to `CurationGraphWriter.write`
                so every upserted entity gets a DERIVED_FROM edge to its
                source vault note.

        Returns:
            CurationResult with combined stage results (may be partial
            if a stage failed).
        """
        start = time.perf_counter()
        stage_errors: list[str] = []

        entities = validation_result.entities
        relationships = validation_result.relationships

        # Defaults for graceful degradation
        empty_dedup = DeduplicationResult(entities=[], merge_actions=[], entities_merged=0)
        empty_conflict = ConflictResolutionResult(relationships=[])

        # Short-circuit on empty
        if not entities and not relationships:
            elapsed = (time.perf_counter() - start) * 1000
            return CurationResult(
                write_result=WriteResult(),
                dedup_result=empty_dedup,
                conflict_result=empty_conflict,
                curation_time_ms=elapsed,
            )

        # Stage 7a: Deduplication
        try:
            dedup_result = await self._deduplicator.deduplicate(entities)
            logger.debug("Stage 7a (dedup): %d merged", dedup_result.entities_merged)
        except Exception as e:
            logger.error("Stage 7a (dedup) failed, continuing with raw entities: %s", e)
            stage_errors.append(f"Dedup failed: {e}")
            dedup_result = DeduplicationResult(
                entities=entities, merge_actions=[], entities_merged=0
            )

        # Stage 7b: Conflict resolution
        try:
            conflict_result = await self._conflict_resolver.resolve(
                dedup_result.entities, relationships
            )
            logger.debug(
                "Stage 7b (conflict): %d detected, %d resolved",
                conflict_result.conflicts_detected,
                conflict_result.conflicts_resolved,
            )
        except Exception as e:
            logger.error("Stage 7b (conflict) failed, continuing with raw relationships: %s", e)
            stage_errors.append(f"Conflict resolution failed: {e}")
            conflict_result = ConflictResolutionResult(relationships=relationships)

        # Stage 8: Graph write + provenance
        try:
            write_result = await self._graph_writer.write(
                entities=dedup_result.entities,
                relationships=conflict_result.relationships,
                merge_actions=dedup_result.merge_actions,
                supersession_actions=conflict_result.supersession_actions,
                event_id=event_id,
                session_id=session_id,
                source_metadata=source_metadata,
                vault_note_path=vault_note_path,
            )
            doc_prov_msg = ""
            if write_result.document_provenance_edges > 0:
                doc_prov_msg = (
                    ", %d document provenance edges" % write_result.document_provenance_edges
                )
            logger.debug(
                "Stage 8 (write): %d entities, %d relationships, %d provenance edges%s",
                write_result.entities_created + write_result.entities_updated,
                write_result.relationships_created,
                write_result.provenance_edges_created,
                doc_prov_msg,
            )
        except Exception as e:
            logger.error("Stage 8 (write) failed: %s", e)
            stage_errors.append(f"Graph write failed: {e}")
            write_result = WriteResult()

        elapsed = (time.perf_counter() - start) * 1000
        if stage_errors:
            logger.warning(
                "Curation completed with %d stage errors in %.1fms",
                len(stage_errors),
                elapsed,
            )
        else:
            logger.info("Curation complete in %.1fms", elapsed)

        return CurationResult(
            write_result=write_result,
            dedup_result=dedup_result,
            conflict_result=conflict_result,
            curation_time_ms=elapsed,
            stage_errors=stage_errors,
        )
