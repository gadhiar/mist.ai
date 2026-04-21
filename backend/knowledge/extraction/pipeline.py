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

import hashlib
import logging
import time
from collections import OrderedDict
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from backend.event_store.models import ConversationTurnEvent
from backend.interfaces import EmbeddingProvider, EventStoreProvider
from backend.knowledge.config import ExtractionConfig
from backend.knowledge.extraction.confidence import ConfidenceScorer
from backend.knowledge.extraction.normalizer import EntityNormalizer
from backend.knowledge.extraction.ontology_extractor import OntologyConstrainedExtractor
from backend.knowledge.extraction.preprocessor import PreProcessor
from backend.knowledge.extraction.scope_classifier import SubjectScopeClassifier
from backend.knowledge.extraction.temporal import TemporalResolver
from backend.knowledge.extraction.validator import ExtractionValidator, ValidationResult
from backend.knowledge.storage.graph_store import GraphStore

if TYPE_CHECKING:
    from backend.knowledge.curation.graph_writer import SourceMetadata
    from backend.knowledge.curation.pipeline import CurationPipeline, CurationResult
    from backend.knowledge.extraction.internal_derivation import InternalKnowledgeDeriver

logger = logging.getLogger(__name__)

# Relative importance of entity types for significance scoring.
ENTITY_TYPE_IMPORTANCE: dict[str, float] = {
    "User": 1.0,
    "Person": 0.9,
    "Organization": 0.8,
    "Project": 0.85,
    "Skill": 0.7,
    "Technology": 0.7,
    "Goal": 0.75,
    "Preference": 0.65,
    "Event": 0.6,
    "Concept": 0.5,
    "Topic": 0.4,
    "Location": 0.5,
}

# Significance thresholds per extraction source.
_SOURCE_THRESHOLDS: dict[str, float] = {
    "conversation": 0.3,
    "orchestrator_summary": 0.2,
    "agent_tool_output": 0.5,
}

# Common English stopwords used for information density scoring.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "need",
        "dare",
        "ought",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "because",
        "but",
        "and",
        "or",
        "if",
        "while",
        "about",
        "up",
        "it",
        "its",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "him",
        "his",
        "she",
        "her",
        "they",
        "them",
        "their",
        "this",
        "that",
        "these",
        "those",
        "am",
        "what",
        "which",
        "who",
        "whom",
    }
)


class ExtractionPipeline:
    """Orchestrates the full extraction pipeline (stages 1-8).

    Stages 1-6 produce validated entities and relationships. When a
    CurationPipeline is provided, stages 7-8 deduplicate, resolve
    conflicts, and write to Neo4j with provenance tracking.

    Pre-extraction gates (rate limiting, significance scoring, and input
    deduplication) prevent low-value or redundant utterances from reaching
    the LLM extraction stage.
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
        embedding_provider: EmbeddingProvider | None = None,
        extraction_config: ExtractionConfig | None = None,
        scope_classifier: SubjectScopeClassifier | None = None,
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
            embedding_provider: Optional embedding provider for significance
                scoring and input deduplication. When None, novelty scoring
                and dedup are disabled.
            extraction_config: Optional extraction config for gate thresholds.
                When None, default ExtractionConfig values are used.
            scope_classifier: Optional Stage 1.5 subject-scope classifier
                (Cluster 1). When provided, runs between Stage 1 and
                Stage 2 and writes the subject_scope + confidence into
                PreProcessedInput.metadata. When None, Stage 1.5 is
                skipped and Stage 2 treats scope as "unknown".
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
        self._embedding_provider = embedding_provider
        self._config = extraction_config or ExtractionConfig()
        self._scope_classifier = scope_classifier

        # Rate limiter state
        self._extraction_timestamps: list[float] = []

        # Dedup cache: SHA-256 hash -> (embedding vector, insertion timestamp)
        self._dedup_cache: OrderedDict[str, tuple[list[float], float]] = OrderedDict()

        max_stage = "9" if internal_deriver else ("8" if curation_pipeline else "6")
        logger.info("ExtractionPipeline initialized (stages 1-%s)", max_stage)

    # ------------------------------------------------------------------
    # Pre-extraction gates
    # ------------------------------------------------------------------

    def _check_rate_limit(self) -> bool:
        """Check whether the extraction rate limit has been exceeded.

        Returns:
            True if extraction is allowed, False if rate-limited.
        """
        now = time.monotonic()
        cutoff = now - 60.0
        self._extraction_timestamps = [ts for ts in self._extraction_timestamps if ts > cutoff]
        if len(self._extraction_timestamps) >= self._config.rate_limit_max_per_minute:
            logger.debug(
                "Rate limit hit: %d extractions in last 60s (max %d)",
                len(self._extraction_timestamps),
                self._config.rate_limit_max_per_minute,
            )
            return False
        return True

    def _compute_significance(
        self,
        utterance: str,
        embedding: list[float] | None,
    ) -> float:
        """Compute a significance score for the utterance.

        The score is a weighted sum of three components:
        - Content length (0.3 weight): longer utterances are more likely
          to contain extractable knowledge.
        - Information density (0.4 weight): ratio of non-stopword tokens.
        - Novelty (0.3 weight): 1 minus max similarity to recent cache
          entries. Falls back to 1.0 when no embedding provider or empty
          cache.

        Args:
            utterance: The raw user utterance.
            embedding: Pre-computed embedding vector, or None.

        Returns:
            Float between 0.0 and 1.0.
        """
        words = utterance.split()
        word_count = len(words)

        # Content length component
        length_score = min(word_count / 20.0, 1.0)

        # Information density component
        if word_count == 0:
            density_score = 0.0
        else:
            non_stop = sum(1 for w in words if w.lower() not in _STOPWORDS)
            density_score = non_stop / word_count

        # Novelty component
        novelty_score = 1.0
        if embedding is not None and self._dedup_cache:
            max_sim = 0.0
            emb_array = np.array(embedding)
            emb_norm = np.linalg.norm(emb_array)
            if emb_norm > 0:
                for _hash, (cached_emb, _ts) in self._dedup_cache.items():
                    cached_array = np.array(cached_emb)
                    cached_norm = np.linalg.norm(cached_array)
                    if cached_norm > 0:
                        sim = float(np.dot(emb_array, cached_array) / (emb_norm * cached_norm))
                        if sim > max_sim:
                            max_sim = sim
            novelty_score = 1.0 - max_sim

        return (length_score * 0.3) + (density_score * 0.4) + (novelty_score * 0.3)

    def _check_dedup(self, utterance: str, embedding: list[float]) -> bool:
        """Check whether the utterance is a near-duplicate of a recent one.

        Uses SHA-256 of the utterance text as the cache key and compares
        the embedding against cached embeddings via cosine similarity.

        Args:
            utterance: The raw user utterance.
            embedding: Pre-computed embedding vector.

        Returns:
            True if the utterance is a duplicate and should be skipped.
        """
        content_hash = hashlib.sha256(utterance.encode("utf-8")).hexdigest()

        # Exact match by hash
        if content_hash in self._dedup_cache:
            logger.debug("Dedup: exact hash match for '%s'", utterance[:60])
            return True

        # Semantic similarity check against cached embeddings
        threshold = self._config.dedup_similarity_threshold
        emb_array = np.array(embedding)
        emb_norm = np.linalg.norm(emb_array)
        if emb_norm > 0:
            now = time.monotonic()
            ttl = self._config.dedup_cache_ttl_seconds
            for _hash, (cached_emb, ts) in self._dedup_cache.items():
                if now - ts > ttl:
                    continue
                cached_array = np.array(cached_emb)
                cached_norm = np.linalg.norm(cached_array)
                if cached_norm > 0:
                    sim = float(np.dot(emb_array, cached_array) / (emb_norm * cached_norm))
                    if sim >= threshold:
                        logger.debug(
                            "Dedup: similarity %.3f >= %.3f for '%s'",
                            sim,
                            threshold,
                            utterance[:60],
                        )
                        return True

        return False

    def _add_to_dedup_cache(self, utterance: str, embedding: list[float]) -> None:
        """Add an utterance embedding to the dedup cache.

        Evicts the oldest entry when the cache exceeds the configured
        size, and prunes entries older than the TTL.

        Args:
            utterance: The raw user utterance.
            embedding: Pre-computed embedding vector.
        """
        content_hash = hashlib.sha256(utterance.encode("utf-8")).hexdigest()
        now = time.monotonic()

        # Prune expired entries
        ttl = self._config.dedup_cache_ttl_seconds
        expired = [h for h, (_emb, ts) in self._dedup_cache.items() if now - ts > ttl]
        for h in expired:
            del self._dedup_cache[h]

        # Add new entry
        self._dedup_cache[content_hash] = (embedding, now)

        # Evict oldest if over capacity
        while len(self._dedup_cache) > self._config.dedup_cache_size:
            self._dedup_cache.popitem(last=False)

    # ------------------------------------------------------------------
    # Main extraction entry points
    # ------------------------------------------------------------------

    async def extract_from_utterance(
        self,
        utterance: str,
        conversation_history: list[dict[str, str]],
        event_id: str,
        session_id: str,
        reference_date: datetime | None = None,
        source_metadata: SourceMetadata | None = None,
        extraction_source: str = "conversation",
    ) -> ValidationResult | CurationResult:
        """Main entry point for live extraction.

        Runs pre-extraction gates (rate limit, significance, dedup) then
        stages 1-6 on a single utterance and returns a ValidationResult
        with validated entities and relationships.

        Args:
            utterance: The user utterance to extract from.
            conversation_history: Recent conversation as list of
                {"role": str, "content": str} dicts.
            event_id: The event store event ID for provenance.
            session_id: The conversation session ID.
            reference_date: Reference date for temporal resolution.
                Defaults to datetime.now().
            source_metadata: Optional external source metadata. Forwarded
                to the curation pipeline for document provenance tracking.
            extraction_source: Source type for threshold lookup. One of
                "conversation" (default), "orchestrator_summary", or
                "agent_tool_output".

        Returns:
            ValidationResult with validated entities and relationships.
        """
        if reference_date is None:
            reference_date = datetime.now()

        pipeline_start = time.perf_counter()

        # -- Gate 1: Rate limit (before any processing) --
        if not self._check_rate_limit():
            logger.info("Extraction skipped (rate-limited) for '%s'", utterance[:60])
            return ValidationResult(valid=True)

        # Stage 1: Pre-processing
        stage_start = time.perf_counter()
        pre_processed = self._preprocessor.pre_process(
            utterance=utterance,
            conversation_history=conversation_history,
            reference_date=reference_date,
        )
        stage_1_ms = (time.perf_counter() - stage_start) * 1000
        logger.debug("Stage 1 (pre-processing): %.1fms", stage_1_ms)

        # -- Generate embedding for significance + dedup gates --
        embedding: list[float] | None = None
        if self._embedding_provider is not None:
            embedding = self._embedding_provider.generate_embedding(utterance)

        # -- Gate 2: Significance scoring --
        sig_threshold = _SOURCE_THRESHOLDS.get(
            extraction_source,
            self._config.significance_threshold,
        )
        significance = self._compute_significance(utterance, embedding)
        if significance < sig_threshold:
            logger.info(
                "Extraction skipped (significance %.3f < %.3f) for '%s'",
                significance,
                sig_threshold,
                utterance[:60],
            )
            return ValidationResult(valid=True)

        # -- Gate 3: Input deduplication --
        if embedding is not None and self._check_dedup(utterance, embedding):
            logger.info("Extraction skipped (duplicate) for '%s'", utterance[:60])
            return ValidationResult(valid=True)

        # Stage 1.5: Subject-scope classification (Cluster 1).
        # Terse LLM call that tags the utterance as user-scope, system-scope,
        # or third-party so Stage 2 can weight the extraction prompt. Writes
        # results into pre_processed.metadata. Never gates the pipeline --
        # on any failure the scope is "unknown" and Stage 2 proceeds as if
        # Stage 1.5 were disabled. Positioned AFTER rate-limit, significance,
        # and dedup gates so dropped utterances never spawn a classifier LLM
        # call; positioned BEFORE Stage 2 so the extractor can read the scope
        # metadata in its prompt.
        if self._scope_classifier is not None:
            stage_start = time.perf_counter()
            scope_result = await self._scope_classifier.classify(pre_processed)
            stage_1_5_ms = (time.perf_counter() - stage_start) * 1000
            pre_processed.metadata["subject_scope"] = scope_result.scope
            pre_processed.metadata["subject_scope_confidence"] = scope_result.confidence
            logger.debug(
                "Stage 1.5 (scope classifier): %.1fms scope=%s confidence=%.2f",
                stage_1_5_ms,
                scope_result.scope,
                scope_result.confidence,
            )

        # Record timestamp for rate limiter (extraction proceeding)
        self._extraction_timestamps.append(time.monotonic())

        # Stage 2: Extraction (LLM call).
        # Note on Bug K two-layer defense: pre_processed.metadata may carry an
        # "injection_warning" flag from the preprocessor (backend/knowledge/extraction/
        # preprocessor.py _detect_injection). We intentionally do NOT gate here on that
        # flag — enforcement lives in EXTRACTION_SYSTEM_PROMPT Rule 10, which instructs
        # the LLM to return empty extraction on directive utterances. The metadata is a
        # reserved signal for a future drop-on-flag policy; flipping to hard-drop here
        # would require re-tuning the prompt rule and re-running Phase A gauntlet.
        stage_start = time.perf_counter()
        extraction = await self._extractor.extract(pre_processed)
        stage_2_ms = (time.perf_counter() - stage_start) * 1000
        logger.debug("Stage 2 (extraction): %.1fms", stage_2_ms)

        # Stamp source provenance onto the extraction result
        if source_metadata is not None:
            extraction.source_metadata = source_metadata

        # Short-circuit if nothing was extracted
        if not extraction.entities and not extraction.relationships:
            # Cache the utterance so repeated identical inputs are deduped and
            # do not trigger another LLM call (K-12 fix).
            if embedding is not None:
                self._add_to_dedup_cache(utterance, embedding)
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

        # Add to dedup cache after successful extraction
        if embedding is not None:
            self._add_to_dedup_cache(utterance, embedding)

        # Stages 7-8: Curation (if enabled and entities present)
        if self._curation_pipeline is not None and result.entities:
            stage_start = time.perf_counter()
            curation_result = await self._curation_pipeline.curate_and_store(
                result,
                event_id=event_id,
                session_id=session_id,
                source_metadata=source_metadata,
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
