"""Regression tests for ExtractionPipeline dedup cache on empty extraction (K-12).

K-12 bug: when extract_from_utterance yields zero entities/relationships,
the utterance is not added to _dedup_cache, causing repeated LLM calls for
the same input.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from backend.knowledge.config import ExtractionConfig
from backend.knowledge.extraction.confidence import ConfidenceScorer
from backend.knowledge.extraction.normalizer import EntityNormalizer
from backend.knowledge.extraction.ontology_extractor import (
    ExtractionResult,
    OntologyConstrainedExtractor,
)
from backend.knowledge.extraction.pipeline import ExtractionPipeline
from backend.knowledge.extraction.preprocessor import PreProcessor
from backend.knowledge.extraction.temporal import TemporalResolver
from backend.knowledge.extraction.validator import ExtractionValidator
from backend.knowledge.storage.graph_store import GraphStore
from tests.mocks.config import TEST_EVENT_ID, TEST_SESSION_ID
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline(extractor) -> ExtractionPipeline:
    """Build an ExtractionPipeline suitable for dedup tests.

    Uses real stage instances (no LLM needed for stages 3-6) and injects the
    supplied extractor at stage 2. ExtractionConfig uses significance_threshold=0.0
    so utterances are never skipped by the significance gate, and
    dedup_similarity_threshold=0.95 (default) for the dedup check.
    """
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
    )


def _empty_extractor() -> AsyncMock:
    """Return an AsyncMock that mimics OntologyConstrainedExtractor.extract.

    Always returns an ExtractionResult with no entities and no relationships,
    simulating the case where the LLM finds nothing to extract.
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDedupCacheOnEmptyExtraction:
    """Verify that empty extraction results are cached to prevent repeated LLM calls."""

    @pytest.mark.asyncio
    async def test_extractor_not_called_again_after_empty_result(self):
        """After an empty extraction, the same utterance must be deduped.

        Arrange: pipeline with an extractor that always returns empty results.
        Act: call extract_from_utterance twice with the same utterance.
        Assert: extractor.extract was called exactly once (second call deduped).
        """
        # Arrange
        extractor = _empty_extractor()
        pipeline = _make_pipeline(extractor)
        utterance = "I spent the weekend hiking in the mountains."

        # Act
        await pipeline.extract_from_utterance(
            utterance=utterance,
            conversation_history=[],
            event_id=TEST_EVENT_ID,
            session_id=TEST_SESSION_ID,
        )
        await pipeline.extract_from_utterance(
            utterance=utterance,
            conversation_history=[],
            event_id=TEST_EVENT_ID,
            session_id=TEST_SESSION_ID,
        )

        # Assert
        assert extractor.extract.call_count == 1, (
            f"extractor.extract called {extractor.extract.call_count} times; "
            "expected 1 -- second call should have been deduped after empty result"
        )

    @pytest.mark.asyncio
    async def test_dedup_cache_populated_after_empty_result(self):
        """After an empty extraction, the utterance hash must appear in _dedup_cache.

        This is the direct state-based complement to the call-count test above.
        """
        # Arrange
        extractor = _empty_extractor()
        pipeline = _make_pipeline(extractor)
        utterance = "Nothing extractable here at all."

        # Act
        await pipeline.extract_from_utterance(
            utterance=utterance,
            conversation_history=[],
            event_id=TEST_EVENT_ID,
            session_id=TEST_SESSION_ID,
        )

        # Assert
        import hashlib

        content_hash = hashlib.sha256(utterance.encode("utf-8")).hexdigest()
        assert (
            content_hash in pipeline._dedup_cache
        ), "Expected utterance hash in _dedup_cache after empty extraction result"

    @pytest.mark.asyncio
    async def test_successful_extraction_still_deduped(self):
        """Baseline: successful extraction (entities present) deduplication is unaffected.

        After a non-empty extraction, the same utterance must also be deduped on
        the second call. This confirms the existing path still works correctly
        alongside the new empty-result path.
        """
        # Arrange
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

        extractor = AsyncMock(spec=OntologyConstrainedExtractor)
        extractor.extract.return_value = ExtractionResult(
            entities=[
                {
                    "id": "python",
                    "type": "Technology",
                    "name": "Python",
                    "confidence": 0.9,
                    "source_type": "extracted",
                    "aliases": [],
                    "description": "Programming language.",
                }
            ],
            relationships=[],
            raw_llm_output='{"entities": [{"id": "python"}], "relationships": []}',
            extraction_time_ms=1.0,
            source_utterance="I use Python.",
        )

        pipeline = ExtractionPipeline(
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
        )

        utterance = "I use Python."

        # Act
        await pipeline.extract_from_utterance(
            utterance=utterance,
            conversation_history=[],
            event_id=TEST_EVENT_ID,
            session_id=TEST_SESSION_ID,
        )
        await pipeline.extract_from_utterance(
            utterance=utterance,
            conversation_history=[],
            event_id=TEST_EVENT_ID,
            session_id=TEST_SESSION_ID,
        )

        # Assert
        assert extractor.extract.call_count == 1, (
            f"extractor.extract called {extractor.extract.call_count} times after "
            "non-empty extraction; expected 1 on second call deduped"
        )
