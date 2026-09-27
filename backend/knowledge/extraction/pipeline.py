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
import sqlite3
import time
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from backend.event_store.models import ConversationTurnEvent
from backend.interfaces import EmbeddingProvider, EventStoreProvider
from backend.knowledge.config import ExtractionConfig
from backend.knowledge.extraction.confidence import ConfidenceScorer
from backend.knowledge.extraction.normalizer import EntityNormalizer
from backend.knowledge.extraction.ontology_extractor import (
    ExtractionResult,
    OntologyConstrainedExtractor,
)
from backend.knowledge.extraction.preprocessor import PreProcessor
from backend.knowledge.extraction.scope_classifier import SubjectScopeClassifier
from backend.knowledge.extraction.signal_detector import SignalDetector
from backend.knowledge.extraction.temporal import TemporalResolver
from backend.knowledge.extraction.validator import ExtractionValidator, ValidationResult
from backend.knowledge.extraction_cache import (
    OUTCOME_EXTRACTED,
    OUTCOME_SKIPPED,
    SKIP_BELOW_SIGNIFICANCE,
    SKIP_DUPLICATE,
    SKIP_RATE_LIMITED,
    SKIP_TOO_SHORT,
)
from backend.knowledge.storage.graph_store import GraphStore

if TYPE_CHECKING:
    from backend.knowledge.curation.graph_writer import RebuildStamps, SourceMetadata
    from backend.knowledge.curation.pipeline import CurationPipeline, CurationResult
    from backend.knowledge.extraction.internal_derivation import InternalKnowledgeDeriver
    from backend.knowledge.extraction_cache import ExtractionCache

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

# Significance thresholds for sources that deliberately DEVIATE from the
# configured baseline. A source absent from this table resolves to
# `ExtractionConfig.significance_threshold` (env: SIGNIFICANCE_THRESHOLD) at
# the lookup below.
#
# "conversation" is deliberately ABSENT. It is the default extraction_source
# and the dominant production path, so an entry here would shadow the
# configured threshold on the one path that matters most -- which is exactly
# the bug this table used to carry. It resolves through config instead.
# `test_pipeline_significance_threshold.py` guards the absence.
#
# The two entries that remain encode an intended ORDERING against that
# configured baseline: a pre-digested orchestrator summary clears a LOWER bar
# than ordinary conversation, and noisy agent tool output a HIGHER one. Both
# are Command Center ingest sources with no caller yet.
_SOURCE_THRESHOLDS: dict[str, float] = {
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


# ----------------------------------------------------------------------
# Backlog dispatch surface (T2a). The extraction-backlog dispatcher
# (`backend/extraction_backlog/dispatcher.py`) sends Stages 1.5 / 2 / 9 to the
# out-of-process extraction service and uses the pieces below for everything
# the backend still owns: gates 0/2/3, the Stage 9 context, and the apply step.
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DispatchGateDecision:
    """Outcome of the dispatch-time gates (0, 2, 3) for one turn.

    `skip_reason` is one of the `SKIP_*` constants when a gate declined the
    turn, None when it may be dispatched. `embedding` is the utterance
    embedding the gates computed (None when there is no embedding provider or
    Gate 0 fired first); the caller hands it back to
    `note_extraction_completed` so Gate 3's dedup cache learns the turn.
    """

    skip_reason: str | None
    embedding: list[float] | None


@dataclass(frozen=True, slots=True)
class DerivationContext:
    """Stage 9 context the backend gathers because the service cannot.

    Mirrors the fields of `backend.extraction_contract.models.DerivationInput`
    except `assistant_response`, which the dispatcher takes from the logged
    turn.
    """

    signal_types: tuple[str, ...]
    matched_patterns: tuple[str, ...]
    existing_internal_entities: str


@dataclass(frozen=True, slots=True)
class TurnToApply:
    """The logged-turn fields the apply step reads.

    `recorded_at` is the turn's LOGGED timestamp (`conversation_turn_events.
    timestamp`, UTC-normalised by `ConversationTurnEvent.to_dict`), the value
    `LogRegenerator.rebuild` passes as both reference date and fact-time.
    """

    event_id: str
    session_id: str
    user_utterance: str
    recorded_at: str


class ApplyProgress(Protocol):
    """Durable per-turn apply progress, owned by the caller of `apply_cached_turn`.

    Two markers rather than one. The graph write (Stages 7-8) and the Stage 9
    operations are separate Neo4j writes, and neither is atomic with the SQLite
    marker, so a crash can land between any two of them. Recording `curated`
    between the two lets a restart skip a curation that already landed and
    re-run only the Stage 9 operations, which are MERGE/SET-idempotent
    (`internal_derivation.py`, `_apply_operation`). The window that remains --
    a crash inside `curate_and_store` or between its return and
    `mark_curated` -- re-runs curation; see the dispatcher module docstring for
    what that does and does not reproduce.
    """

    @property
    def curated(self) -> bool:
        """True when Stages 3-8 already completed for this turn."""
        ...

    def mark_curated(self) -> None:
        """Durably record that Stages 3-8 completed."""
        ...

    def mark_applied(self) -> None:
        """Durably record that the turn is fully applied."""
        ...


@dataclass(frozen=True, slots=True)
class ApplyReport:
    """What `apply_cached_turn` did for one turn."""

    skipped: bool
    curation_result: Any
    curation_resumed: bool
    derivation_operations_applied: int

    @property
    def stage_errors(self) -> list[str]:
        """Curation stage errors, empty when curation did not run or was clean."""
        if self.curation_result is None:
            return []
        return list(getattr(self.curation_result, "stage_errors", []) or [])


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
        extraction_cache: ExtractionCache | None = None,
        rebuild_stamps: RebuildStamps | None = None,
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
            extraction_cache: Optional cache of Stage-2 extraction decisions
                (F3, extraction-cache-phase-1). When provided, pre-extraction
                gates record their skip via `_record_skip`, and the
                post-Stage-2 site records the raw extraction (empty or not)
                via `_record_extraction`. When None, gates still
                short-circuit the pipeline exactly as before -- no row is
                written, which is the status quo this phase incrementally
                replaces gate by gate, not a new failure mode.
            rebuild_stamps: Optional (ontology_version, extraction_version,
                model_hash) triple to stamp on cache rows. Must be provided
                together with `extraction_cache`, or not at all -- see Raises.

        Raises:
            ValueError: `extraction_cache` and `rebuild_stamps` were not both
                provided or both omitted. A pipeline wired with only one of
                the two would have `_record_skip` silently write no row on
                every gate, with no error and no log -- the mis-wire would
                surface only much later as a `ColdCacheError` from a rebuild,
                pointing at the rebuild rather than at this construction
                site. Caught here instead.
        """
        if extraction_cache is not None and rebuild_stamps is None:
            raise ValueError(
                "rebuild_stamps is required when extraction_cache is provided -- "
                "without it _record_skip cannot stamp a cache row and would "
                "silently write nothing"
            )
        if rebuild_stamps is not None and extraction_cache is None:
            raise ValueError(
                "extraction_cache is required when rebuild_stamps is provided -- "
                "without it there is nothing for _record_skip to write to"
            )

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
        self._extraction_cache = extraction_cache
        self._rebuild_stamps = rebuild_stamps

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

    # The three helpers below are the gate logic shared by the in-process entry
    # point (`extract_from_utterance`) and the backlog dispatcher
    # (`evaluate_dispatch_gates`). One definition each, so the two paths cannot
    # drift on what "too short", "significant" or "embedded" means.

    @staticmethod
    def _is_too_short(utterance: str) -> bool:
        """Gate 0: fewer than three words cannot carry a fact."""
        return len(utterance.split()) < 3

    def _significance_threshold(self, extraction_source: str) -> float:
        """Gate 2 threshold for a source (see `_SOURCE_THRESHOLDS`).

        Kept as a named assignment so `grep -n "sig_threshold = _SOURCE_THRESHOLDS.get"
        backend/knowledge/extraction/pipeline.py`, cited by
        test_pipeline_significance_threshold.py and KNOWN_ISSUES.md, still finds
        the one place the threshold is resolved.
        """
        sig_threshold = _SOURCE_THRESHOLDS.get(
            extraction_source, self._config.significance_threshold
        )
        return sig_threshold

    def _embed(self, utterance: str) -> list[float] | None:
        """Utterance embedding for Gates 2/3, None when no provider is wired."""
        if self._embedding_provider is None:
            return None
        return self._embedding_provider.generate_embedding(utterance)

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

    def _record_skip(self, event_id: str, skip_reason: str, created_at: str) -> None:
        """Record a pre-extraction gate skip to the cache.

        No-ops when `extraction_cache` and `rebuild_stamps` are both None
        (the constructor default) -- a pipeline built without either simply
        writes no row, which is the status quo this phase incrementally
        replaces, not a new hazard (extraction-cache-phase-1 Task 3 ruling).
        A pipeline wired with only one of the two cannot reach this method:
        `__init__` rejects that combination at construction time.

        Failure-isolated by design, but only for operational storage
        failures -- a full disk or a lost connection degrades
        REBUILDABILITY, never the conversation. `sqlite3.Error` and
        `OSError` are caught for that reason. `ValueError` (raised by
        `ExtractionCache.put`'s own fail-closed guards on an inconsistent
        outcome/skip_reason pair -- `extraction_cache.py:173,176,178,180`)
        and `TypeError` (from `json.dumps` on a non-serializable payload)
        are deliberately NOT caught: those mean the CALLER passed something
        wrong, and swallowing them would reproduce the exact silent-failure
        mode Task 3's constructor pairing guard exists to prevent --
        surfacing only later as a `ColdCacheError` pointing at a rebuild
        rather than at this call.

        Args:
            event_id: The event store event ID this turn belongs to.
            skip_reason: One of the `SKIP_*` constants in extraction_cache.py.
            created_at: The recorded_at timestamp for this turn (C1
                bitemporal recorded_at, not wall-clock now()).
        """
        if self._extraction_cache is None:
            # __init__ guarantees extraction_cache and rebuild_stamps are
            # both None or both set -- checking one is sufficient.
            return
        try:
            self._extraction_cache.put(
                event_id,
                self._rebuild_stamps.ontology_version,
                self._rebuild_stamps.extraction_version,
                self._rebuild_stamps.model_hash,
                outcome=OUTCOME_SKIPPED,
                created_at=created_at,
                skip_reason=skip_reason,
            )
        except (sqlite3.Error, OSError):
            logger.warning(
                "[WARNING] extraction cache write failed for event %s (skip=%s); "
                "this turn will not be rebuildable",
                event_id,
                skip_reason,
                exc_info=True,
            )

    def _record_extraction(
        self,
        event_id: str,
        extraction: ExtractionResult,
        scope: str | None,
        scope_confidence: float | None,
        created_at: str,
    ) -> None:
        """Record the RAW Stage-2 output -- before Stages 3-6 touch it.

        The boundary is deliberate (spec D2). Stages 3-6 are pure and a
        rebuild re-runs them, so caching their output instead would freeze
        the ontology's effects into the row and force a full LLM re-run on
        every ontology bump.

        No-ops when `extraction_cache` and `rebuild_stamps` are both None,
        mirroring `_record_skip`. Failure-isolated the same way and for the
        same reason: `sqlite3.Error` / `OSError` (operational storage
        failures) are caught; `ValueError` (the `put` guards) and
        `TypeError` (non-serializable payload) propagate, because both mean
        this call passed something wrong rather than that storage failed.

        Args:
            event_id: The event store event ID this turn belongs to.
            extraction: The Stage-2 ExtractionResult, before confidence
                scoring, temporal resolution, normalization, or validation.
            scope: The Stage 1.5 subject-scope classification, or None when
                the scope classifier is disabled.
            scope_confidence: Confidence for `scope`, or None to match.
            created_at: The recorded_at timestamp for this turn (C1
                bitemporal recorded_at, not wall-clock now()).
        """
        if self._extraction_cache is None:
            return
        try:
            self._extraction_cache.put(
                event_id,
                self._rebuild_stamps.ontology_version,
                self._rebuild_stamps.extraction_version,
                self._rebuild_stamps.model_hash,
                outcome=OUTCOME_EXTRACTED,
                created_at=created_at,
                entities=extraction.entities,
                relationships=extraction.relationships,
                scope=scope,
                scope_confidence=scope_confidence,
            )
        except (sqlite3.Error, OSError):
            logger.warning(
                "[WARNING] extraction cache write failed for event %s; "
                "this turn will not be rebuildable",
                event_id,
                exc_info=True,
            )

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
        recorded_at: str | None = None,
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
            recorded_at: Fact-time ISO-8601 timestamp of the source event
                (C1). Defaults to now(UTC). Anchors `reference_date` so a
                rebuild resolves relative dates identically to the live turn,
                and flows to the reconciliation engine as the bitemporal
                recorded_at.

        Returns:
            ValidationResult with validated entities and relationships.
        """
        if recorded_at is None:
            recorded_at = datetime.now(UTC).isoformat()
        if reference_date is None:
            # Temporal resolution anchors to the event's fact-time so a rebuild
            # resolves "last year" identically to the live turn (C1).
            reference_date = datetime.fromisoformat(recorded_at)

        # -- Gate 0: too short to carry a fact --
        # Moved here from conversation_handler.py in phase 1. Left in the
        # handler it prevented this function from being called at all, so a
        # gated turn produced NO cache row -- making "the pipeline skipped it"
        # indistinguishable from "the row was lost". One place decides, one
        # place records. Runs before Gate 1 (rate limit): a turn that is both
        # too short and rate-limited records "too_short", not
        # "rate_limited" -- the utterance carries no fact either way, so the
        # more specific, cheaper-to-check reason wins.
        if self._is_too_short(utterance):
            self._record_skip(event_id, SKIP_TOO_SHORT, recorded_at)
            logger.info("Extraction skipped (too short) for '%s'", utterance[:60])
            return ValidationResult(valid=True)

        pipeline_start = time.perf_counter()

        # -- Gate 1: Rate limit (before any processing) --
        if not self._check_rate_limit():
            self._record_skip(event_id, SKIP_RATE_LIMITED, recorded_at)
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
        embedding = self._embed(utterance)

        # -- Gate 2: Significance scoring --
        sig_threshold = self._significance_threshold(extraction_source)
        significance = self._compute_significance(utterance, embedding)
        if significance < sig_threshold:
            self._record_skip(event_id, SKIP_BELOW_SIGNIFICANCE, recorded_at)
            logger.info(
                "Extraction skipped (significance %.3f < %.3f) for '%s'",
                significance,
                sig_threshold,
                utterance[:60],
            )
            return ValidationResult(valid=True)

        # -- Gate 3: Input deduplication --
        if embedding is not None and self._check_dedup(utterance, embedding):
            self._record_skip(event_id, SKIP_DUPLICATE, recorded_at)
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

        # Site 5 of 5. Placed here rather than in each downstream branch so the
        # empty short-circuit and the full Stages 3-6 path share ONE write.
        # Stage 2 ran in both cases, so both are outcome='extracted'; an empty
        # payload means the model looked and found nothing, which is a different
        # fact from a 'skipped' row where it never looked.
        self._record_extraction(
            event_id,
            extraction,
            pre_processed.metadata.get("subject_scope"),
            pre_processed.metadata.get("subject_scope_confidence"),
            recorded_at,
        )

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
                recorded_at=recorded_at,
            )
            stage_78_ms = (time.perf_counter() - stage_start) * 1000
            logger.debug("Stages 7-8 (curation): %.1fms", stage_78_ms)

            total_ms = (time.perf_counter() - pipeline_start) * 1000
            logger.info(
                "Pipeline complete in %.1fms: %d entities, %d relationships "
                "(curation: %d merged, %d closed) from '%s'",
                total_ms,
                len(result.entities),
                len(result.relationships),
                curation_result.dedup_result.entities_merged,
                curation_result.reconcile_result.closed,
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
        # Fact-time for bitemporal edges: the stored event timestamp (C1).
        recorded_at = (
            event.timestamp.isoformat()
            if isinstance(event.timestamp, datetime)
            else str(event.timestamp)
        )

        return await self.extract_from_utterance(
            utterance=event.user_utterance,
            conversation_history=conversation_context,
            event_id=event.event_id,
            session_id=event.session_id,
            reference_date=reference_date,
            recorded_at=recorded_at,
        )

    # ------------------------------------------------------------------
    # Backlog dispatch surface (T2a)
    # ------------------------------------------------------------------

    def evaluate_dispatch_gates(
        self, utterance: str, extraction_source: str = "conversation"
    ) -> DispatchGateDecision:
        """Run Gates 0, 2 and 3 for a turn the backlog is about to dispatch.

        The same gate logic as `extract_from_utterance` (shared through
        `_is_too_short`, `_significance_threshold`, `_embed`,
        `_compute_significance` and `_check_dedup`), minus Gate 1. The queued
        rate limit is retired on the dispatcher path because a queue does not
        rate-limit: the backlog dispatches one job at a time, so there is no
        burst to shed, and a turn Gate 1 declined would be lost from the graph
        rather than merely delayed. `extract_from_utterance` keeps Gate 1.

        Pure decision: nothing is written here. The dispatcher records the skip
        in the extraction cache under the ACTIVE EPOCH's stamps, which is what
        a rebuild looks the row up by (`LogRegenerator.rebuild` keys the cache
        off the epoch row).

        Args:
            utterance: The logged user utterance.
            extraction_source: Source for the Gate 2 threshold lookup.

        Returns:
            The decision, carrying the embedding for `note_extraction_completed`.
        """
        if self._is_too_short(utterance):
            logger.info("Extraction skipped (too short) for '%s'", utterance[:60])
            return DispatchGateDecision(skip_reason=SKIP_TOO_SHORT, embedding=None)

        embedding = self._embed(utterance)

        sig_threshold = self._significance_threshold(extraction_source)
        significance = self._compute_significance(utterance, embedding)
        if significance < sig_threshold:
            logger.info(
                "Extraction skipped (significance %.3f < %.3f) for '%s'",
                significance,
                sig_threshold,
                utterance[:60],
            )
            return DispatchGateDecision(skip_reason=SKIP_BELOW_SIGNIFICANCE, embedding=embedding)

        if embedding is not None and self._check_dedup(utterance, embedding):
            logger.info("Extraction skipped (duplicate) for '%s'", utterance[:60])
            return DispatchGateDecision(skip_reason=SKIP_DUPLICATE, embedding=embedding)

        return DispatchGateDecision(skip_reason=None, embedding=embedding)

    def note_extraction_completed(self, utterance: str, embedding: list[float] | None) -> None:
        """Teach Gate 3 about a turn whose Stage 2 completed (empty result or not).

        `extract_from_utterance` adds to the dedup cache on both the empty
        short-circuit and the full path; the dispatcher calls this once the
        service's result is durably cached, so the two paths feed Gate 3 the
        same turns.
        """
        if embedding is not None:
            self._add_to_dedup_cache(utterance, embedding)

    @property
    def extraction_cache(self) -> ExtractionCache | None:
        """The cache this pipeline records into (None when built without one).

        Exposed so `backend.factories.build_extraction_dispatcher` hands the
        dispatcher the SAME cache instance, and so the same SQLite connection,
        the in-process path writes through.
        """
        return self._extraction_cache

    @property
    def derivation_enabled(self) -> bool:
        """True when Stage 9 is wired (`resolve_internal_derivation` allowed it)."""
        return self._internal_deriver is not None

    async def build_derivation_context(self, utterance: str) -> DerivationContext | None:
        """Gather the Stage 9 context for a turn, or None when Stage 9 will not run.

        None when Stage 9 is not wired, or when `SignalDetector` finds no signal
        in the utterance -- the same gate `_run_internal_derivation` applies on
        the in-process path (`detect(utterance)` with no tool calls), so both
        paths send derivation for the same turns.

        `existing_internal_entities` is read from the graph HERE, at dispatch
        time, which is why the dispatcher must not dispatch turn N+1 before turn
        N's operations are applied: the next turn's prompt context depends on
        them.
        """
        if self._internal_deriver is None:
            return None
        signals = SignalDetector().detect(utterance)
        if not signals.has_signals:
            return None
        existing = await self._internal_deriver.fetch_existing_internal_entities()
        return DerivationContext(
            signal_types=tuple(sorted(signals.signal_types)),
            matched_patterns=tuple(signals.matched_patterns),
            existing_internal_entities=existing,
        )

    async def apply_cached_turn(
        self,
        turn: TurnToApply,
        cached: Mapping[str, Any],
        progress: ApplyProgress,
    ) -> ApplyReport:
        r"""Apply one turn's cached extraction decision to the graph.

        THE apply step of the extraction backlog: the dispatcher calls this for
        every turn, both for a result that just arrived from the service and
        for a turn a crashed process cached but never finished applying. One
        code path for live and for recovery.

        Stages 3-8 are the per-turn body of `LogRegenerator.rebuild`'s replay
        loop (`log_regenerator.py`, the `for turn in turns:` block), kept
        semantically identical so the live graph equals a rebuild:

        - a `skipped` row is a recorded decision and applies as a no-op;
        - an `extracted` row becomes `ExtractionResult(entities, relationships,
          source_utterance=<logged utterance>)`, Stages 3-6 run with
          `reference_date = <logged timestamp>`, and `curate_and_store` runs
          UNCONDITIONALLY with `event_id`, `session_id` and
          `recorded_at=<logged timestamp>` -- even when validation left no
          entities, because the rebuild does too. (The in-process path skips
          curation when `result.entities` is empty; see
          `extract_from_utterance`. That is a live-vs-rebuild difference this
          function deliberately resolves toward the rebuild.)

        Then, beyond the rebuild: Stage 9 operations persisted on the cache row
        (`cached["derivation"]["operations"]`) are applied through
        `InternalKnowledgeDeriver.apply_operations`. The rebuild does not do
        this (it has no Stage 9 at all -- `grep -c "apply_operations\|derive("
        backend/knowledge/regeneration/log_regenerator.py` is 0); the
        operations land in the `:__SelfModel__` partition, which
        `canonical_graph_form` does not compare by default
        (`include_self_model=False`).

        Markers: `progress.mark_curated()` after Stages 3-8, then
        `progress.mark_applied()` after the operations. A turn whose progress
        already says `curated` skips straight to the operations.

        Args:
            turn: The logged turn.
            cached: The row `ExtractionCache.get` returned for this turn under
                the active epoch.
            progress: The turn's durable apply markers.

        Returns:
            What was done.
        """
        if cached["outcome"] == OUTCOME_SKIPPED:
            progress.mark_applied()
            return ApplyReport(
                skipped=True,
                curation_result=None,
                curation_resumed=False,
                derivation_operations_applied=0,
            )

        curation_result = None
        curation_resumed = progress.curated
        if not curation_resumed:
            extraction = ExtractionResult(
                entities=cached["entities"],
                relationships=cached["relationships"],
                source_utterance=turn.user_utterance,
            )
            reference_date = datetime.fromisoformat(turn.recorded_at)
            extraction = self._confidence_scorer.adjust_confidence(extraction)
            extraction = self._temporal_resolver.resolve(extraction, reference_date)
            extraction = await self._normalizer.normalize(extraction)
            validation = self._validator.validate(extraction)
            if self._curation_pipeline is not None:
                curation_result = await self._curation_pipeline.curate_and_store(
                    validation,
                    event_id=turn.event_id,
                    session_id=turn.session_id,
                    recorded_at=turn.recorded_at,
                )
            progress.mark_curated()

        operations = list((cached.get("derivation") or {}).get("operations") or [])
        applied_ops = 0
        if operations:
            if self._internal_deriver is None:
                logger.warning(
                    "Stage 9 is disabled; not applying %d cached derivation operation(s) "
                    "for event %s",
                    len(operations),
                    turn.event_id,
                )
            else:
                applied = await self._internal_deriver.apply_operations(
                    operations, session_id=turn.session_id, event_id=turn.event_id
                )
                applied_ops = len(applied)

        progress.mark_applied()
        return ApplyReport(
            skipped=False,
            curation_result=curation_result,
            curation_resumed=curation_resumed,
            derivation_operations_applied=applied_ops,
        )
