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
from backend.knowledge.extraction_cache import (
    OUTCOME_EXTRACTED,
    OUTCOME_SKIPPED,
    SKIP_BELOW_SIGNIFICANCE,
    SKIP_DUPLICATE,
    SKIP_RATE_LIMITED,
)
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

        Asserts outcome='extracted', not an empty call list: Task 4 wires
        Site 5 (post-Stage-2), which records every completed Stage-2 pass,
        empty payload or not (`grep -n "self._record_extraction(" backend/
        knowledge/extraction/pipeline.py` -- one call site, right after the
        Stage 2 debug log). This is that write, not the too_short skip --
        the boundary claim below still lives in `extract.assert_awaited_once()`.
        """
        pipeline, spy_cache = pipeline_factory(extractor_returns=_empty_result())

        await pipeline.extract_from_utterance(
            utterance="I use Python",  # exactly three words
            conversation_history=[],
            event_id="evt-three",
            session_id="sess-1",
            recorded_at="2026-08-18T00:00:00+00:00",
        )

        assert spy_cache.calls == [
            ("evt-three", "extracted", None, "2026-08-18T00:00:00+00:00"),
        ]
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


# ---------------------------------------------------------------------------
# Task 4: the four remaining decision sites (Gates 1-3 plus post-Stage-2).
#
# task-4-brief.md's own test snippets were written against a pipeline_factory
# contract this file's actual fixture does not implement: they assert 3-tuple
# spy_cache.calls (the real SpyCache.put records a 4-tuple including
# created_at -- see conftest.py), pass bare tuples like `("user", "python")`
# where an ExtractionResult is required, and pass force_gate values
# ("rate_limit", "significance", "dedup") the fixture never accepted. The
# fixture docstring in conftest.py is authoritative on what is actually
# settable and why; the tests below are written against it, not against the
# brief's snippets.
# ---------------------------------------------------------------------------

UTTERANCE = "I have used Python for about five years now"


def _extraction_with_python_entity() -> ExtractionResult:
    """A minimal valid ExtractionResult: one Technology entity, no relationships.

    Same entity shape as test_pipeline_dedup.py's
    test_successful_extraction_still_deduped, which already proves this shape
    clears real Stage 3-6 (ConfidenceScorer, TemporalResolver, EntityNormalizer,
    ExtractionValidator) unmocked.
    """
    return ExtractionResult(
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
        source_utterance=UTTERANCE,
    )


class TestGates1Through3RecordTheirReason:
    """One case per pre-extraction gate that survives Gate 0.

    Each case drives its gate by a different mechanism because the fixture
    only lets construction force SKIP_RATE_LIMITED and SKIP_DUPLICATE (see
    conftest.py's pipeline_factory docstring):
      - rate_limit: force_gate=SKIP_RATE_LIMITED (rate_limit_max_per_minute=0).
      - significance: NOT forceable at construction (the "conversation"
        source hardcodes a 0.3 threshold that never reads config -- see the
        fixture docstring). Driven instead by extraction_source=
        "agent_tool_output" (threshold 0.5) plus an all-stopword utterance,
        so density=0 and even a maxed novelty score (1.0, since the dedup
        cache is empty) can't clear 0.5.
      - dedup: force_gate=SKIP_DUPLICATE with dedup_utterance matching the
        call's utterance, so Gate 3's exact-hash branch trips immediately.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "event_id, utterance, force_gate, dedup_utterance, extraction_source, expected_reason",
        [
            pytest.param(
                "evt-rate-limit",
                UTTERANCE,
                SKIP_RATE_LIMITED,
                None,
                "conversation",
                SKIP_RATE_LIMITED,
                id="rate-limit",
            ),
            pytest.param(
                "evt-significance",
                "it is not that",
                None,
                None,
                "agent_tool_output",
                SKIP_BELOW_SIGNIFICANCE,
                id="significance",
            ),
            pytest.param(
                "evt-dedup",
                UTTERANCE,
                SKIP_DUPLICATE,
                UTTERANCE,
                "conversation",
                SKIP_DUPLICATE,
                id="dedup",
            ),
        ],
    )
    async def test_each_gate_records_its_own_reason(
        self,
        pipeline_factory,
        event_id,
        utterance,
        force_gate,
        dedup_utterance,
        extraction_source,
        expected_reason,
    ):
        pipeline, spy_cache = pipeline_factory(
            force_gate=force_gate, dedup_utterance=dedup_utterance
        )

        await pipeline.extract_from_utterance(
            utterance=utterance,
            conversation_history=[],
            event_id=event_id,
            session_id="sess-1",
            extraction_source=extraction_source,
            recorded_at="2026-08-18T00:00:00+00:00",
        )

        assert spy_cache.calls == [
            (event_id, OUTCOME_SKIPPED, expected_reason, "2026-08-18T00:00:00+00:00"),
        ]


class TestSiteFiveRecordsExtractedOutcome:
    """Post-Stage-2: both the empty and non-empty extraction paths write
    outcome='extracted', never 'skipped' -- Stage 2 ran either way.
    """

    @pytest.mark.asyncio
    async def test_successful_extraction_records_outcome_extracted(self, pipeline_factory):
        pipeline, spy_cache = pipeline_factory(extractor_returns=_extraction_with_python_entity())

        await pipeline.extract_from_utterance(
            utterance=UTTERANCE,
            conversation_history=[],
            event_id="evt-ok",
            session_id="sess-1",
            recorded_at="2026-08-18T00:00:00+00:00",
        )

        assert spy_cache.calls == [
            ("evt-ok", OUTCOME_EXTRACTED, None, "2026-08-18T00:00:00+00:00"),
        ]

    @pytest.mark.asyncio
    async def test_empty_extraction_is_extracted_not_skipped(self, pipeline_factory):
        """Stage 2 ran and found nothing. That is NOT the same as never running."""
        pipeline, spy_cache = pipeline_factory(extractor_returns=_empty_result())

        await pipeline.extract_from_utterance(
            utterance=UTTERANCE,
            conversation_history=[],
            event_id="evt-empty",
            session_id="sess-1",
            recorded_at="2026-08-18T00:00:00+00:00",
        )

        assert spy_cache.calls == [
            ("evt-empty", OUTCOME_EXTRACTED, None, "2026-08-18T00:00:00+00:00"),
        ]


class TestFiveSiteCountGuard:
    """The count guard: every decision path writes exactly one row.

    Branch 1 on 2026-08-04 (842bb90) applied a guard to one of five dispatch
    sites; only the whole-branch gate caught it. If a new early return is
    added to extract_from_utterance without a paired cache write, or an
    existing write is deleted, this fails.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "label, utterance, force_gate, dedup_utterance, extraction_source, extractor_returns",
        [
            pytest.param("too_short", "ok", None, None, "conversation", None, id="too-short"),
            pytest.param(
                "rate_limit",
                UTTERANCE,
                SKIP_RATE_LIMITED,
                None,
                "conversation",
                None,
                id="rate-limit",
            ),
            pytest.param(
                "significance",
                "it is not that",
                None,
                None,
                "agent_tool_output",
                None,
                id="significance",
            ),
            pytest.param(
                "dedup",
                UTTERANCE,
                SKIP_DUPLICATE,
                UTTERANCE,
                "conversation",
                None,
                id="dedup",
            ),
            pytest.param(
                "extracted",
                UTTERANCE,
                None,
                None,
                "conversation",
                _extraction_with_python_entity(),
                id="extracted",
            ),
        ],
    )
    async def test_every_decision_path_writes_exactly_one_row(
        self,
        pipeline_factory,
        label,
        utterance,
        force_gate,
        dedup_utterance,
        extraction_source,
        extractor_returns,
    ):
        pipeline, spy_cache = pipeline_factory(
            force_gate=force_gate,
            dedup_utterance=dedup_utterance,
            extractor_returns=extractor_returns,
        )

        await pipeline.extract_from_utterance(
            utterance=utterance,
            conversation_history=[],
            event_id=f"evt-{label}",
            session_id="sess-1",
            extraction_source=extraction_source,
            recorded_at="2026-08-18T00:00:00+00:00",
        )

        assert len(spy_cache.calls) == 1, f"gate={label} wrote {len(spy_cache.calls)} rows"


class TestCacheWriteFailureIsolation:
    """A cache write that raises must degrade rebuildability, never the turn."""

    @pytest.mark.asyncio
    async def test_a_failing_cache_write_never_breaks_the_turn(self, pipeline_factory):
        class ExplodingCache:
            def put(self, *a, **kw):
                raise RuntimeError("disk full")

            def get(self, *a, **kw):
                return None

        pipeline, _ = pipeline_factory(
            cache=ExplodingCache(), extractor_returns=_extraction_with_python_entity()
        )

        result = await pipeline.extract_from_utterance(
            utterance=UTTERANCE,
            conversation_history=[],
            event_id="evt-boom",
            session_id="sess-1",
            recorded_at="2026-08-18T00:00:00+00:00",
        )

        assert result is not None  # the turn completed
