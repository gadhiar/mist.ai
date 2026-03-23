"""Extraction pipeline orchestrator.

Orchestrates the 6-stage extraction pipeline (Phase 1B):
  Stage 1: Pre-processing (context assembly, no LLM)
  Stage 2: Extraction (single LLM call, ontology-constrained)
  Stage 3: Confidence scoring (hedge detection, third-party cap)
  Stage 4: Temporal resolution (relative -> absolute dates)
  Stage 5: Normalization + deduplication (canonical IDs, graph matching)
  Stage 6: Validation (schema + constraint checks)

Stages 7-8 (curation) are Phase 2. Stage 9 (internal reasoning) is Phase 3.
Stage 10 (cloud validation) is optional/future.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING

from backend.event_store.models import ConversationTurnEvent
from backend.interfaces import EventStoreProvider
from backend.knowledge.extraction.confidence import ConfidenceScorer
from backend.knowledge.extraction.normalizer import EntityNormalizer
from backend.knowledge.extraction.ontology_extractor import OntologyConstrainedExtractor
from backend.knowledge.extraction.preprocessor import PreProcessor
from backend.knowledge.extraction.temporal import TemporalResolver
from backend.knowledge.extraction.validator import ExtractionValidator, ValidationResult
from backend.knowledge.storage.graph_store import GraphStore

if TYPE_CHECKING:
    from backend.knowledge.curation.pipeline import CurationPipeline, CurationResult
    from backend.knowledge.extraction.internal_derivation import InternalKnowledgeDeriver

logger = logging.getLogger(__name__)


class ExtractionPipeline:
    """Orchestrates the full extraction pipeline (stages 1-8).

    Stages 1-6 produce validated entities and relationships. When a
    CurationPipeline is provided, stages 7-8 deduplicate, resolve
    conflicts, and write to Neo4j with provenance tracking.
    """

    def __init__(
        self,
        preprocessor: PreProcessor,
        extractor: OntologyConstrainedExtractor,
        confidence_scorer: ConfidenceScorer,
        temporal_resolver: TemporalResolver,
        normalizer: EntityNormalizer,
        validator: ExtractionValidator,
        graph_store: GraphStore,
        event_store: EventStoreProvider | None = None,
        curation_pipeline: CurationPipeline | None = None,
        internal_deriver: InternalKnowledgeDeriver | None = None,
    ) -> None:
        """Initialize the extraction pipeline with injected stage processors.

        Args:
            preprocessor: Stage 1 -- context assembly.
            extractor: Stage 2 -- ontology-constrained LLM extraction.
            confidence_scorer: Stage 3 -- hedge detection and scoring.
            temporal_resolver: Stage 4 -- relative to absolute date conversion.
            normalizer: Stage 5 -- canonical ID generation and dedup.
            validator: Stage 6 -- schema and constraint validation.
            graph_store: Neo4j graph store for normalization lookups.
            event_store: Optional event store for re-extraction workflows.
            curation_pipeline: Optional stages 7-8 curation. When None,
                pipeline stops at Stage 6 (test/regeneration mode).
            internal_deriver: Optional stage 9 internal knowledge derivation.
                When None, self-model updates are skipped.
        """
        self.graph_store = graph_store
        self.event_store = event_store
        self._preprocessor = preprocessor
        self._extractor = extractor
        self._confidence_scorer = confidence_scorer
        self._temporal_resolver = temporal_resolver
        self._normalizer = normalizer
        self._validator = validator
        self._curation_pipeline = curation_pipeline
        self._internal_deriver = internal_deriver
        max_stage = "9" if internal_deriver else ("8" if curation_pipeline else "6")
        logger.info("ExtractionPipeline initialized (stages 1-%s)", max_stage)

    async def extract_from_utterance(
        self,
        utterance: str,
        conversation_history: list[dict[str, str]],
        event_id: str,
        session_id: str,
        reference_date: datetime | None = None,
    ) -> ValidationResult | CurationResult:
        """Main entry point for live extraction.

        Runs stages 1-6 on a single utterance and returns a
        ValidationResult with validated entities and relationships.

        Args:
            utterance: The user utterance to extract from.
            conversation_history: Recent conversation as list of
                {"role": str, "content": str} dicts.
            event_id: The event store event ID for provenance.
            session_id: The conversation session ID.
            reference_date: Reference date for temporal resolution.
                Defaults to datetime.now().

        Returns:
            ValidationResult with validated entities and relationships.
        """
        if reference_date is None:
            reference_date = datetime.now()

        pipeline_start = time.perf_counter()

        # Stage 1: Pre-processing
        stage_start = time.perf_counter()
        pre_processed = self._preprocessor.pre_process(
            utterance=utterance,
            conversation_history=conversation_history,
            reference_date=reference_date,
        )
        stage_1_ms = (time.perf_counter() - stage_start) * 1000
        logger.debug("Stage 1 (pre-processing): %.1fms", stage_1_ms)

        # Stage 2: Extraction (LLM call)
        stage_start = time.perf_counter()
        extraction = await self._extractor.extract(pre_processed)
        stage_2_ms = (time.perf_counter() - stage_start) * 1000
        logger.debug("Stage 2 (extraction): %.1fms", stage_2_ms)

        # Short-circuit if nothing was extracted
        if not extraction.entities and not extraction.relationships:
            total_ms = (time.perf_counter() - pipeline_start) * 1000
            logger.info(
                "Pipeline complete in %.1fms: no entities extracted from '%s'",
                total_ms,
                utterance[:60],
            )
            return ValidationResult(valid=True)

        # Stage 3: Confidence scoring
        stage_start = time.perf_counter()
        extraction = self._confidence_scorer.adjust_confidence(extraction)
        stage_3_ms = (time.perf_counter() - stage_start) * 1000
        logger.debug("Stage 3 (confidence): %.1fms", stage_3_ms)

        # Stage 4: Temporal resolution
        stage_start = time.perf_counter()
        extraction = self._temporal_resolver.resolve(extraction, reference_date)
        stage_4_ms = (time.perf_counter() - stage_start) * 1000
        logger.debug("Stage 4 (temporal): %.1fms", stage_4_ms)

        # Stage 5: Normalization + deduplication
        stage_start = time.perf_counter()
        extraction = await self._normalizer.normalize(extraction)
        stage_5_ms = (time.perf_counter() - stage_start) * 1000
        logger.debug("Stage 5 (normalization): %.1fms", stage_5_ms)

        # Stage 6: Validation
        stage_start = time.perf_counter()
        result = self._validator.validate(extraction)
        stage_6_ms = (time.perf_counter() - stage_start) * 1000
        logger.debug("Stage 6 (validation): %.1fms", stage_6_ms)

        # Stages 7-8: Curation (if enabled and entities present)
        if self._curation_pipeline is not None and result.entities:
            stage_start = time.perf_counter()
            curation_result = await self._curation_pipeline.curate_and_store(
                result, event_id=event_id, session_id=session_id
            )
            stage_78_ms = (time.perf_counter() - stage_start) * 1000
            logger.debug("Stages 7-8 (curation): %.1fms", stage_78_ms)

            total_ms = (time.perf_counter() - pipeline_start) * 1000
            logger.info(
                "Pipeline complete in %.1fms: %d entities, %d relationships "
                "(curation: %d merged, %d conflicts) from '%s'",
                total_ms,
                len(result.entities),
                len(result.relationships),
                curation_result.dedup_result.entities_merged,
                curation_result.conflict_result.conflicts_detected,
                utterance[:60],
            )

            # Stage 9: Internal knowledge derivation (if enabled)
            await self._run_internal_derivation(utterance, event_id, session_id)

            return curation_result

        total_ms = (time.perf_counter() - pipeline_start) * 1000
        logger.info(
            "Pipeline complete in %.1fms: %d entities, %d relationships "
            "(%d warnings, %d errors) from '%s'",
            total_ms,
            len(result.entities),
            len(result.relationships),
            len(result.warnings),
            len(result.errors),
            utterance[:60],
        )

        # Stage 9: Internal knowledge derivation (if enabled)
        await self._run_internal_derivation(utterance, event_id, session_id)

        return result

    async def _run_internal_derivation(
        self, utterance: str, event_id: str, session_id: str
    ) -> None:
        """Run Stage 9 internal derivation if enabled and signals detected."""
        if self._internal_deriver is None:
            return

        try:
            signals = self._internal_deriver._signal_detector.detect(utterance)
            if signals.has_signals:
                stage_start = time.perf_counter()
                # TODO: assistant_response is empty at extraction time. For richer
                # self-model derivation, ConversationHandler can call the deriver
                # separately with full turn context in a future enhancement.
                internal_result = await self._internal_deriver.derive(
                    utterance=utterance,
                    assistant_response="",
                    signals=signals,
                    session_id=session_id,
                    event_id=event_id,
                )
                if internal_result.llm_called:
                    stage_9_ms = (time.perf_counter() - stage_start) * 1000
                    logger.debug(
                        "Stage 9 (internal): %.1fms, %d operations",
                        stage_9_ms,
                        len(internal_result.operations),
                    )
        except Exception as e:
            logger.error("Stage 9 (internal derivation) failed: %s", e)

    async def extract_from_event(
        self,
        event: ConversationTurnEvent,
        conversation_context: list[dict[str, str]],
    ) -> ValidationResult | CurationResult:
        """Entry point for re-extraction from the event store.

        Used during graph regeneration to re-extract knowledge from
        previously recorded conversation turns.

        Args:
            event: A ConversationTurnEvent from the event store.
            conversation_context: Conversation context assembled from
                surrounding events.

        Returns:
            ValidationResult with validated entities and relationships.
        """
        reference_date = (
            event.timestamp if isinstance(event.timestamp, datetime) else datetime.now()
        )

        return await self.extract_from_utterance(
            utterance=event.user_utterance,
            conversation_history=conversation_context,
            event_id=event.event_id,
            session_id=event.session_id,
            reference_date=reference_date,
        )
