"""Curation pipeline orchestrator (C2 cutover).

Stage 7a: EntityDeduplicator -- match against existing graph entities.
Stage 7b+8 (relationships): ReconciliationEngine -- schema-driven bitemporal
reconcile + write (replaces the deleted ConflictResolver and the writer's
relationship path).
Stage 8 (entities): CurationGraphWriter -- entity MERGE + provenance.

All graph mutation for a turn runs inside one asyncio.Lock acquisition
(Inv-A9): extraction tasks are fire-and-forget concurrent, so the
dedup-read -> reconcile-read-modify-write sequence must serialize.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

from backend.knowledge.curation.deduplication import DeduplicationResult, EntityDeduplicator
from backend.knowledge.curation.graph_writer import CurationGraphWriter, SourceMetadata, WriteResult
from backend.knowledge.curation.reconciliation import (
    ActionKind,
    ReconcileTurnResult,
    ReconciliationEngine,
)
from backend.knowledge.extraction.validator import ValidationResult

logger = logging.getLogger(__name__)

_LEARNING_REASONS = {"single_supersession", "contradiction", "progression", "cease", "retract"}


@dataclass(slots=True)
class CurationResult:
    """Combined result of the curation pipeline.

    `validated_entities`/`validated_relationships` carry forward post-validation
    inputs for downstream trigger detection (e.g. user.md C-pattern render).
    """

    write_result: WriteResult
    dedup_result: DeduplicationResult
    reconcile_result: ReconcileTurnResult
    curation_time_ms: float
    stage_errors: list[str] = field(default_factory=list)
    validated_entities: list[dict] = field(default_factory=list)
    validated_relationships: list[dict] = field(default_factory=list)


class CurationPipeline:
    """Orchestrates stages 7-8 of the knowledge pipeline (bitemporal)."""

    def __init__(
        self,
        deduplicator: EntityDeduplicator,
        reconciliation_engine: ReconciliationEngine,
        graph_writer: CurationGraphWriter,
        write_lock: asyncio.Lock | None = None,
    ) -> None:
        self._deduplicator = deduplicator
        self._engine = reconciliation_engine
        self._graph_writer = graph_writer
        self._write_lock = write_lock or asyncio.Lock()

    async def curate_and_store(
        self,
        validation_result: ValidationResult,
        event_id: str,
        session_id: str,
        source_metadata: SourceMetadata | None = None,
        vault_note_path: str | None = None,
        recorded_at: str | None = None,
    ) -> CurationResult:
        """Run curation stages and write to graph, serialized per turn.

        recorded_at: the source event's timestamp (fact-time, C1). Defaults to
        wall-clock only for non-conversation paths (document ingest) that
        carry no event.
        """
        start = time.perf_counter()
        stage_errors: list[str] = []
        recorded_at = recorded_at or datetime.now(UTC).isoformat()

        entities = validation_result.entities
        relationships = validation_result.relationships
        empty_dedup = DeduplicationResult(entities=[], merge_actions=[], entities_merged=0)

        if not entities and not relationships:
            return CurationResult(
                write_result=WriteResult(),
                dedup_result=empty_dedup,
                reconcile_result=ReconcileTurnResult(),
                curation_time_ms=(time.perf_counter() - start) * 1000,
                validated_entities=entities,
                validated_relationships=relationships,
            )

        async with self._write_lock:  # Inv-A9: one turn's read-modify-write at a time
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

            # Stage 8 (entities + provenance)
            entity_write_failed = False
            try:
                write_result = await self._graph_writer.write(
                    entities=dedup_result.entities,
                    merge_actions=dedup_result.merge_actions,
                    event_id=event_id,
                    session_id=session_id,
                    source_metadata=source_metadata,
                    vault_note_path=vault_note_path,
                )
            except Exception as e:
                logger.error("Stage 8 (entity write) failed: %s", e)
                stage_errors.append(f"Graph write failed: {e}")
                write_result = WriteResult()
                entity_write_failed = True

            # Stage 7b+8 (relationships): bitemporal reconcile + write
            if entity_write_failed:
                # Closes must never land without their replacement appends: the
                # engine's CLOSE_TRANSACTION matches priors by elementId and
                # succeeds even when the turn's new entity nodes were never
                # created, so reconciling after an entity-write failure turns
                # a transient Neo4j error into net silent belief deletion.
                # The turn's writes are MERGE-idempotent and replay
                # convergently if re-run later.
                logger.error(
                    "Stage 7b/8 (reconcile) skipped: entity write failed for event %s", event_id
                )
                stage_errors.append("Reconciliation skipped: entity write failed")
                reconcile_result = ReconcileTurnResult()
            else:
                # Stage 7a rewrote merged entity ids in place; relationships
                # still reference the incoming ids, which may have no node.
                # Remap endpoints the same way the Stage-5 normalizer does.
                rels_for_engine = relationships
                if dedup_result.id_renames:
                    renames = dedup_result.id_renames
                    rels_for_engine = [
                        {
                            **rel,
                            "source": renames.get(rel.get("source", ""), rel.get("source", "")),
                            "target": renames.get(rel.get("target", ""), rel.get("target", "")),
                        }
                        for rel in relationships
                    ]
                try:
                    reconcile_result = await self._engine.reconcile_turn(
                        rels_for_engine,
                        recorded_at=recorded_at,
                        event_id=event_id,
                        session_id=session_id,
                    )
                    logger.debug(
                        "Stage 7b/8 (reconcile): %d appended, %d closed, %d reinforced, %d flags",
                        reconcile_result.appended,
                        reconcile_result.closed,
                        reconcile_result.reinforced,
                        len(reconcile_result.flags),
                    )
                except Exception as e:
                    logger.error("Stage 7b/8 (reconcile) failed: %s", e)
                    stage_errors.append(f"Reconciliation failed: {e}")
                    reconcile_result = ReconcileTurnResult()

            # LearningEvents for belief changes (audit layer)
            try:
                now_iso = datetime.now(UTC).isoformat()
                for act in reconcile_result.actions:
                    if act.kind is ActionKind.CLOSE_TRANSACTION and act.reason in _LEARNING_REASONS:
                        await self._graph_writer.create_belief_change_learning_event(
                            reason=act.reason,
                            predicate=act.predicate,
                            old_target_id=act.target,
                            session_id=session_id,
                            event_id=event_id,
                            now=now_iso,
                            source_metadata=source_metadata,
                        )
                        write_result.learning_events_created += 1
            except Exception as e:
                logger.error("LearningEvent emission failed: %s", e)
                stage_errors.append(f"LearningEvent emission failed: {e}")

        elapsed = (time.perf_counter() - start) * 1000
        if stage_errors:
            logger.warning(
                "Curation completed with %d stage errors in %.1fms", len(stage_errors), elapsed
            )
        else:
            logger.info("Curation complete in %.1fms", elapsed)

        return CurationResult(
            write_result=write_result,
            dedup_result=dedup_result,
            reconcile_result=reconcile_result,
            curation_time_ms=elapsed,
            stage_errors=stage_errors,
            validated_entities=entities,
            validated_relationships=relationships,
        )
