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


def _build_pipeline(*, extractor, scope_classifier) -> ExtractionPipeline:
    """Build a minimal ExtractionPipeline suitable for Stage 1.5 integration tests."""
    conn = FakeNeo4jConnection()
    embeddings = FakeEmbeddingGenerator()
    graph_store = GraphStore(connection=conn, embedding_generator=embeddings)

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
