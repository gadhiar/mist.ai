"""Tests for ExtractionPipeline writing decisions to the extraction cache.

Gate 0 (this file's first test): the too-short guard used to live in
conversation_handler.py, where it prevented extract_from_utterance from being
called at all -- so a gated turn produced no cache row. Moved into the
pipeline (Task 3) so the pipeline itself can record the skip.
"""

from unittest.mock import AsyncMock

import pytest

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
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection


def _empty_result() -> ExtractionResult:
    return ExtractionResult(
        entities=[],
        relationships=[],
        raw_llm_output="{}",
        extraction_time_ms=1.0,
        source_utterance="",
    )


class TestGate0:
    """The <3-word gate, relocated from the handler into the pipeline."""

    @pytest.mark.asyncio
    async def test_short_utterance_is_gated_inside_the_pipeline(self, pipeline_factory):
        """Gate 0: the pipeline itself decides, so the pipeline itself can record."""
        pipeline, spy_cache = pipeline_factory()
        result = await pipeline.extract_from_utterance(
            utterance="ok sure",  # two words
            conversation_history=[],
            event_id="evt-short",
            session_id="sess-1",
            recorded_at="2026-08-18T00:00:00+00:00",
        )
        assert result.entities == []
        assert spy_cache.calls == [
            ("evt-short", "skipped", "too_short", "2026-08-18T00:00:00+00:00"),
        ]

    @pytest.mark.asyncio
    async def test_three_word_utterance_is_not_gated(self, pipeline_factory):
        """Boundary: exactly 3 words must pass Gate 0 (the gate is `< 3`, not `<= 3`).

        Kills a widening mutant (`< 3` -> `<= 3`, or any threshold above 3)
        that would silently stop extracting from ordinary three-word
        utterances -- the dangerous over-gating direction, since nothing
        else in this file's word counts exercises the boundary itself.
        """
        pipeline, spy_cache = pipeline_factory(extractor_returns=_empty_result())

        await pipeline.extract_from_utterance(
            utterance="I use Python",  # exactly three words
            conversation_history=[],
            event_id="evt-three",
            session_id="sess-1",
            recorded_at="2026-08-18T00:00:00+00:00",
        )

        assert spy_cache.calls == []
        pipeline._extractor.extract.assert_awaited_once()


class TestConstructorPairing:
    """extraction_cache and rebuild_stamps must be wired together or not at all.

    pipeline_factory always supplies both, so no test built on it can
    distinguish a correctly wired pipeline from a half-wired one -- these
    tests construct ExtractionPipeline directly instead.
    """

    def _kwargs(self, **overrides):
        embeddings = FakeEmbeddingGenerator()
        kwargs = {
            "preprocessor": PreProcessor(),
            "extractor": AsyncMock(spec=OntologyConstrainedExtractor),
            "confidence_scorer": ConfidenceScorer(),
            "temporal_resolver": TemporalResolver(),
            "normalizer": EntityNormalizer(embedding_generator=embeddings, executor=None),
            "validator": ExtractionValidator(min_confidence=0.0),
            "graph_store": GraphStore(
                connection=FakeNeo4jConnection(), embedding_generator=embeddings
            ),
        }
        kwargs.update(overrides)
        return kwargs

    def test_rejects_extraction_cache_without_rebuild_stamps(self):
        """Half-wiring either one silently no-ops every skip write forever.

        No error, no log -- the mis-wire would surface only much later as a
        ColdCacheError from a rebuild, pointing at the rebuild rather than
        at this construction site.
        """
        with pytest.raises(ValueError, match="rebuild_stamps is required"):
            ExtractionPipeline(
                **self._kwargs(extraction_cache=object(), rebuild_stamps=None),
            )

    def test_rejects_rebuild_stamps_without_extraction_cache(self):
        """Mirror of the above: rebuild_stamps without extraction_cache."""
        with pytest.raises(ValueError, match="extraction_cache is required"):
            ExtractionPipeline(
                **self._kwargs(extraction_cache=None, rebuild_stamps=object()),
            )
