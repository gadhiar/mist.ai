"""Tests for ExtractionPipeline + Stage 1.5 integration (Cluster 1).

Verifies that:
  - when scope_classifier is injected, the pipeline writes
    `subject_scope` + `subject_scope_confidence` into
    PreProcessedInput.metadata before Stage 2 runs.
  - when scope_classifier is None, legacy behavior is preserved.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from backend.knowledge.config import ExtractionConfig, ScopeClassifierConfig
from backend.knowledge.extraction.confidence import ConfidenceScorer
from backend.knowledge.extraction.normalizer import EntityNormalizer
from backend.knowledge.extraction.ontology_extractor import (
    ExtractionResult,
    OntologyConstrainedExtractor,
)
from backend.knowledge.extraction.pipeline import ExtractionPipeline
from backend.knowledge.extraction.preprocessor import PreProcessor
from backend.knowledge.extraction.scope_classifier import SubjectScopeClassifier
from backend.knowledge.extraction.temporal import TemporalResolver
from backend.knowledge.extraction.validator import ExtractionValidator
from backend.knowledge.storage.graph_store import GraphStore
from tests.mocks.config import TEST_EVENT_ID, TEST_SESSION_ID
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection
from tests.mocks.ollama import FakeLLM


def _scope_response(scope: str, confidence: float = 0.9) -> str:
    return (
        '{"scope": "' + scope + '", '
        '"confidence": ' + str(confidence) + ", "
        '"reasoning": "stub"}'
    )


def _empty_extractor() -> AsyncMock:
    """Return a stubbed extractor that returns an empty ExtractionResult.

    Captures pre_processed to let tests assert metadata was set before
    Stage 2 ran.
    """
    mock = AsyncMock(spec=OntologyConstrainedExtractor)
    mock.extract.return_value = ExtractionResult(
        entities=[],
        relationships=[],
        raw_llm_output="{}",
        extraction_time_ms=1.0,
        source_utterance="",
    )
    return mock


def _build_pipeline(
    *,
    extractor,
    scope_classifier,
    extraction_config: ExtractionConfig | None = None,
) -> ExtractionPipeline:
    """Build a minimal ExtractionPipeline suitable for Stage 1.5 integration tests."""
    conn = FakeNeo4jConnection()
    embeddings = FakeEmbeddingGenerator()
    graph_store = GraphStore(connection=conn, embedding_generator=embeddings)

    if extraction_config is None:
        extraction_config = ExtractionConfig(
            significance_threshold=0.0,
            rate_limit_max_per_minute=1000,
            dedup_similarity_threshold=0.95,
            dedup_cache_size=200,
            dedup_cache_ttl_seconds=300,
        )

    return ExtractionPipeline(
        preprocessor=PreProcessor(),
        extractor=extractor,
        confidence_scorer=ConfidenceScorer(),
        temporal_resolver=TemporalResolver(),
        normalizer=EntityNormalizer(embedding_generator=embeddings, executor=None),
        validator=ExtractionValidator(min_confidence=0.0),
        graph_store=graph_store,
        event_store=None,
        curation_pipeline=None,
        internal_deriver=None,
        embedding_provider=FakeEmbeddingGenerator(),
        extraction_config=extraction_config,
        scope_classifier=scope_classifier,
    )


class _CountingClassifier:
    """Captures every call to classify() so tests can assert call counts.

    Mirrors SubjectScopeClassifier's public surface but does not hit any LLM;
    every call returns a deterministic ScopeResult and increments `call_count`.
    Lets ordering tests verify that gates skip Stage 1.5 without patching.
    """

    def __init__(self, scope: str = "user-scope", confidence: float = 0.9) -> None:
        from backend.knowledge.extraction.scope_classifier import ScopeResult

        self._scope = scope
        self._confidence = confidence
        self._result_cls = ScopeResult
        self.call_count = 0

    async def classify(self, pre_processed):  # noqa: ANN001, ANN201
        self.call_count += 1
        return self._result_cls(
            scope=self._scope,  # type: ignore[arg-type]
            confidence=self._confidence,
            reasoning="stub",
            elapsed_ms=0.0,
        )


class TestScopeClassifierInjection:
    """Pipeline accepts scope_classifier kwarg and wires it into Stage 1.5."""

    def test_accepts_scope_classifier_none(self):
        """Backward-compatible path -- None is a valid value for the kwarg."""
        # Arrange + Act
        pipeline = _build_pipeline(extractor=_empty_extractor(), scope_classifier=None)

        # Assert
        assert pipeline._scope_classifier is None

    def test_accepts_scope_classifier_instance(self):
        """Injecting a classifier stores it on the pipeline."""
        # Arrange
        llm = FakeLLM(default_response=_scope_response("user-scope"))
        classifier = SubjectScopeClassifier(llm=llm, config=ScopeClassifierConfig())

        # Act
        pipeline = _build_pipeline(extractor=_empty_extractor(), scope_classifier=classifier)

        # Assert
        assert pipeline._scope_classifier is classifier


class TestStage15Writes:
    """Stage 1.5 writes subject_scope metadata before Stage 2 sees the input."""

    @pytest.mark.asyncio
    async def test_writes_subject_scope_to_metadata_before_extractor_runs(self):
        """Stage 1.5 runs before Stage 2 and populates the scope metadata."""
        # Arrange
        extractor = _empty_extractor()
        llm = FakeLLM(default_response=_scope_response("user-scope", 0.95))
        classifier = SubjectScopeClassifier(llm=llm)
        pipeline = _build_pipeline(extractor=extractor, scope_classifier=classifier)

        # Act
        await pipeline.extract_from_utterance(
            utterance="I am learning Rust.",
            conversation_history=[],
            event_id=TEST_EVENT_ID,
            session_id=TEST_SESSION_ID,
        )

        # Assert -- inspect the PreProcessedInput the extractor received
        extractor.extract.assert_awaited_once()
        pre_processed = extractor.extract.await_args.args[0]
        assert pre_processed.metadata["subject_scope"] == "user-scope"
        assert pre_processed.metadata["subject_scope_confidence"] == pytest.approx(0.95)

    @pytest.mark.asyncio
    async def test_forwards_system_scope_result_to_metadata(self):
        """A system-scope classification propagates to the extractor input."""
        # Arrange
        extractor = _empty_extractor()
        llm = FakeLLM(default_response=_scope_response("system-scope", 0.88))
        classifier = SubjectScopeClassifier(llm=llm)
        pipeline = _build_pipeline(extractor=extractor, scope_classifier=classifier)

        # Act
        await pipeline.extract_from_utterance(
            utterance="MIST uses LanceDB for vector search.",
            conversation_history=[],
            event_id=TEST_EVENT_ID,
            session_id=TEST_SESSION_ID,
        )

        # Assert
        pre_processed = extractor.extract.await_args.args[0]
        assert pre_processed.metadata["subject_scope"] == "system-scope"
        assert pre_processed.metadata["subject_scope_confidence"] == pytest.approx(0.88)


class TestStage15Disabled:
    """When scope_classifier is None, pipeline runs unchanged and metadata is absent."""

    @pytest.mark.asyncio
    async def test_pipeline_runs_without_scope_classifier(self):
        """Legacy path: scope_classifier=None means Stage 1.5 is skipped."""
        # Arrange
        extractor = _empty_extractor()
        pipeline = _build_pipeline(extractor=extractor, scope_classifier=None)

        # Act
        result = await pipeline.extract_from_utterance(
            utterance="I use Python.",
            conversation_history=[],
            event_id=TEST_EVENT_ID,
            session_id=TEST_SESSION_ID,
        )

        # Assert
        extractor.extract.assert_awaited_once()
        pre_processed = extractor.extract.await_args.args[0]
        assert "subject_scope" not in pre_processed.metadata
        assert result is not None  # ValidationResult

    @pytest.mark.asyncio
    async def test_classifier_failure_falls_through_as_unknown(self):
        """Even when classifier returns 'unknown', the pipeline proceeds."""
        # Arrange
        extractor = _empty_extractor()
        llm = FakeLLM(default_response="not-json")  # forces parse failure
        classifier = SubjectScopeClassifier(llm=llm)
        pipeline = _build_pipeline(extractor=extractor, scope_classifier=classifier)

        # Act
        await pipeline.extract_from_utterance(
            utterance="Ambiguous.",
            conversation_history=[],
            event_id=TEST_EVENT_ID,
            session_id=TEST_SESSION_ID,
        )

        # Assert -- metadata is set, value is 'unknown', extractor still ran
        pre_processed = extractor.extract.await_args.args[0]
        assert pre_processed.metadata["subject_scope"] == "unknown"


class TestStage15OrderingAfterGates:
    """P0 #3: scope classifier runs AFTER rate-limit, significance, and dedup gates.

    The previous ordering ran Stage 1.5 between pre-processing and the
    significance / dedup gates, spawning a classifier LLM call for every
    non-rate-limited utterance even when the next gate dropped it.
    Moving Stage 1.5 to after the gates means dropped utterances never
    hit the classifier.
    """

    @pytest.mark.asyncio
    async def test_classifier_skipped_when_rate_limited(self):
        """A rate-limited utterance returns early without invoking the classifier."""
        # Arrange -- rate limit max of 1, fill the window so the next call trips it.
        extractor = _empty_extractor()
        classifier = _CountingClassifier()
        config = ExtractionConfig(
            significance_threshold=0.0,
            rate_limit_max_per_minute=1,
            dedup_similarity_threshold=0.95,
            dedup_cache_size=200,
            dedup_cache_ttl_seconds=300,
        )
        pipeline = _build_pipeline(
            extractor=extractor, scope_classifier=classifier, extraction_config=config
        )
        # Fill the rate-limiter window so the very next extraction is dropped.
        import time as _time

        pipeline._extraction_timestamps.append(_time.monotonic())

        # Act
        result = await pipeline.extract_from_utterance(
            utterance="I use Rust.",
            conversation_history=[],
            event_id=TEST_EVENT_ID,
            session_id=TEST_SESSION_ID,
        )

        # Assert
        assert result is not None
        assert classifier.call_count == 0
        extractor.extract.assert_not_called()

    @pytest.mark.asyncio
    async def test_classifier_skipped_when_significance_below_threshold(self):
        """A below-threshold utterance returns early without invoking the classifier."""
        # Arrange -- threshold 2.0 is unreachable; pass a non-registered extraction_source
        # so the config threshold is used instead of the _SOURCE_THRESHOLDS default.
        extractor = _empty_extractor()
        classifier = _CountingClassifier()
        config = ExtractionConfig(
            significance_threshold=2.0,
            rate_limit_max_per_minute=1000,
            dedup_similarity_threshold=0.95,
            dedup_cache_size=200,
            dedup_cache_ttl_seconds=300,
        )
        pipeline = _build_pipeline(
            extractor=extractor, scope_classifier=classifier, extraction_config=config
        )

        # Act -- extraction_source="test_unregistered" falls back to config.significance_threshold.
        result = await pipeline.extract_from_utterance(
            utterance="I use Rust.",
            conversation_history=[],
            event_id=TEST_EVENT_ID,
            session_id=TEST_SESSION_ID,
            extraction_source="test_unregistered",
        )

        # Assert
        assert result is not None
        assert classifier.call_count == 0
        extractor.extract.assert_not_called()

    @pytest.mark.asyncio
    async def test_classifier_skipped_when_duplicate_detected(self):
        """A duplicate utterance returns early without invoking the classifier."""
        # Arrange -- dedup threshold at 0.0 means every utterance once cached is a dup.
        extractor = _empty_extractor()
        classifier = _CountingClassifier()
        config = ExtractionConfig(
            significance_threshold=0.0,
            rate_limit_max_per_minute=1000,
            dedup_similarity_threshold=0.0,
            dedup_cache_size=200,
            dedup_cache_ttl_seconds=300,
        )
        pipeline = _build_pipeline(
            extractor=extractor, scope_classifier=classifier, extraction_config=config
        )
        # Prime the dedup cache with an initial call; that first call passes.
        await pipeline.extract_from_utterance(
            utterance="I use Rust.",
            conversation_history=[],
            event_id=TEST_EVENT_ID,
            session_id=TEST_SESSION_ID,
        )
        calls_after_first = classifier.call_count

        # Act -- second identical call must be caught by the dedup gate.
        result = await pipeline.extract_from_utterance(
            utterance="I use Rust.",
            conversation_history=[],
            event_id=TEST_EVENT_ID,
            session_id=TEST_SESSION_ID,
        )

        # Assert -- classifier count did not change on the second (dropped) call.
        assert result is not None
        assert classifier.call_count == calls_after_first

    @pytest.mark.asyncio
    async def test_classifier_invoked_when_all_gates_pass(self):
        """Sanity companion: when gates pass, classifier IS invoked exactly once."""
        # Arrange
        extractor = _empty_extractor()
        classifier = _CountingClassifier()
        pipeline = _build_pipeline(extractor=extractor, scope_classifier=classifier)

        # Act
        await pipeline.extract_from_utterance(
            utterance="I am learning Rust.",
            conversation_history=[],
            event_id=TEST_EVENT_ID,
            session_id=TEST_SESSION_ID,
        )

        # Assert
        assert classifier.call_count == 1
        pre_processed = extractor.extract.await_args.args[0]
        assert pre_processed.metadata["subject_scope"] == "user-scope"
