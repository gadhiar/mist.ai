"""Tests for ExtractionPipeline writing decisions to the extraction cache.

Gate 0 (this file's first test): the too-short guard used to live in
conversation_handler.py, where it prevented extract_from_utterance from being
called at all -- so a gated turn produced no cache row. Moved into the
pipeline (Task 3) so the pipeline itself can record the skip.
"""

import inspect
import sqlite3
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
from backend.knowledge.extraction.scope_classifier import ScopeResult
from backend.knowledge.extraction.temporal import TemporalResolver
from backend.knowledge.extraction.validator import ExtractionValidator
from backend.knowledge.extraction_cache import (
    OUTCOME_EXTRACTED,
    OUTCOME_SKIPPED,
    SKIP_BELOW_SIGNIFICANCE,
    SKIP_DUPLICATE,
    SKIP_RATE_LIMITED,
    SKIP_TOO_SHORT,
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
            ("evt-short", "skipped", "too_short", "2026-08-18T00:00:00+00:00", None, None),
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
            ("evt-three", "extracted", None, "2026-08-18T00:00:00+00:00", None, None),
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
            (event_id, OUTCOME_SKIPPED, expected_reason, "2026-08-18T00:00:00+00:00", None, None),
        ]


class TestSkipReasonVocabularyIsPinned:
    """The SKIP_*/OUTCOME_* values are the on-disk vocabulary -- the literal
    strings written into the extraction_cache table's `outcome` and
    `skip_reason` columns. Every other assertion in this file compares a
    constant to itself (`expected_reason=SKIP_DUPLICATE` against
    `spy_cache.calls`, which is built from the SAME constant the production
    code imports), so a change to a constant's VALUE -- not its existence --
    would silently split old and new cache rows with nothing here noticing.
    This is the one place that pins the literals.
    """

    def test_skip_reason_literal_values(self):
        assert SKIP_TOO_SHORT == "too_short"
        assert SKIP_RATE_LIMITED == "rate_limited"
        assert SKIP_BELOW_SIGNIFICANCE == "below_significance"
        assert SKIP_DUPLICATE == "duplicate"

    def test_outcome_literal_values(self):
        assert OUTCOME_EXTRACTED == "extracted"
        assert OUTCOME_SKIPPED == "skipped"


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
            ("evt-ok", OUTCOME_EXTRACTED, None, "2026-08-18T00:00:00+00:00", None, None),
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
            ("evt-empty", OUTCOME_EXTRACTED, None, "2026-08-18T00:00:00+00:00", None, None),
        ]


class _FixedScopeClassifier:
    """Minimal Stage 1.5 double: always returns the same scope/confidence.

    A real SubjectScopeClassifier + FakeLLM (see test_pipeline_scope_classifier.py)
    would work too, but this file only needs a non-None value to reach the
    cache row, not to test the classifier itself.
    """

    def __init__(self, scope: str, confidence: float) -> None:
        self._scope = scope
        self._confidence = confidence

    async def classify(self, pre_processed) -> ScopeResult:
        return ScopeResult(scope=self._scope, confidence=self._confidence, reasoning="stub")


class TestSiteFiveRecordsScopeMetadata:
    """`scope` and `scope_confidence` are the two columns Task 1 added
    specifically for this write. Every other test in this file drives a
    pipeline with no scope_classifier, so both are always None there --
    a mutant transposing the two arguments at the `_record_extraction` call
    site, or reading the wrong `pre_processed.metadata` key, would be
    invisible without a test that drives a non-None value through.
    """

    @pytest.mark.asyncio
    async def test_extraction_records_scope_and_confidence_from_stage_1_5(self, pipeline_factory):
        pipeline, spy_cache = pipeline_factory(
            extractor_returns=_extraction_with_python_entity(),
            scope_classifier=_FixedScopeClassifier("user-scope", 0.87),
        )

        await pipeline.extract_from_utterance(
            utterance=UTTERANCE,
            conversation_history=[],
            event_id="evt-scope",
            session_id="sess-1",
            recorded_at="2026-08-18T00:00:00+00:00",
        )

        assert spy_cache.calls == [
            (
                "evt-scope",
                OUTCOME_EXTRACTED,
                None,
                "2026-08-18T00:00:00+00:00",
                "user-scope",
                0.87,
            ),
        ]


class TestFiveSiteCountGuard:
    """The count guard: every KNOWN decision path writes exactly one row.

    Branch 1 on 2026-08-04 (842bb90) applied a guard to one of five dispatch
    sites; only the whole-branch gate caught it. This test proves that
    deleting an existing write breaks its case (mutation-proved -- see
    task-4-report.md "Fix round 1"). It does NOT prove that a SIXTH,
    unenumerated early return would be caught: enumeration is exactly what a
    new site escapes, since nothing here would construct a case for it.
    TestNoUnguardedEarlyReturn below closes that gap structurally.
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


class TestNoUnguardedEarlyReturn:
    """Source-level guard, in the spirit of TestNoBackendModuleRestatesAStamp
    (tests/unit/knowledge/test_version_stamps.py): a SIXTH early return added
    to extract_from_utterance without a paired recorder call is exactly the
    842bb90 shape (a guard applied to one of five sites), and it is
    invisible to TestFiveSiteCountGuard above by construction -- that test
    only enumerates the five sites known when it was written.

    Today, every "the pipeline decided not to produce a result this turn"
    exit is spelled `return ValidationResult(valid=True)`: the four gates
    use it directly, and the post-Stage-2 empty short-circuit reuses it
    (an empty Stage-2 result is still valid=True with nothing extracted).
    `grep -c "return ValidationResult(valid=True)"` against this method's
    source is 5; `grep -c` for `self._record_skip(` + `self._record_extraction(`
    is 4 + 1 = 5. That equality is not "each exit is individually paired"
    -- the empty short-circuit shares Site 5's single _record_extraction
    call with the two success-path returns below it (`return curation_result`,
    `return result`), which are not `ValidationResult(valid=True)`-shaped
    and so are not in this count at all. It holds because right now no
    gate-shaped exit exists without a recorder call, and no recorder call
    exists without at least one gate-shaped exit.

    Adding a new gate-shaped exit WITHOUT a paired call changes the exit
    count but not the call count, breaking the equality -- this test fails.
    Adding one correctly, WITH a call, moves both counts together and this
    test stays silent -- it exists to catch omission, not to tax every
    future correct addition.

    Known limitation, stated rather than hidden: this only catches early
    returns spelled exactly `return ValidationResult(valid=True)`. A new
    exit using a differently-shaped literal (extra kwargs, a different
    return type) would not change either count and would not be caught
    here -- the same limitation the reviewed fix for this gap named. The
    call side carries the worse-direction risk: `call_count` is a textual
    count over the WHOLE method body, so a future comment or docstring line
    that happens to contain the literal text `self._record_skip(` or
    `self._record_extraction(` would inflate it and mask a genuinely
    missing write behind a false-passing equality -- do not write either
    token in a comment or docstring inside this method.
    """

    def test_gate_shaped_exit_count_matches_recorder_call_count(self):
        source = inspect.getsource(ExtractionPipeline.extract_from_utterance)
        exit_count = source.count("return ValidationResult(valid=True)")
        call_count = source.count("self._record_skip(") + source.count("self._record_extraction(")

        assert exit_count == call_count, (
            f"extract_from_utterance has {exit_count} gate-shaped early returns "
            f"(`return ValidationResult(valid=True)`) but {call_count} recorder "
            "calls (_record_skip + _record_extraction) -- a new early return "
            "of this shape needs a paired _record_skip call, or must be "
            "provably covered by an earlier _record_extraction the way the "
            "post-Stage-2 empty short-circuit is. Update this test's "
            "docstring to explain the new pairing once verified."
        )


class _ExplodingCache:
    """A cache double raising the exception type production code isolates.

    sqlite3.OperationalError, not a placeholder RuntimeError: it is the real
    exception type a SQLite-backed ExtractionCache raises for an actual
    "disk full" condition, and it is exactly what `_record_skip` /
    `_record_extraction`'s narrowed `except (sqlite3.Error, OSError)` is
    written to catch (see TestCacheWriteExceptionNarrowing below for the
    complementary case: what must NOT be caught).
    """

    def put(self, *a, **kw):
        raise sqlite3.OperationalError("disk full")

    def get(self, *a, **kw):
        return None


class TestCacheWriteFailureIsolation:
    """A cache write that raises an operational storage error must degrade
    rebuildability, never the turn -- for both recorder methods.
    """

    @pytest.mark.asyncio
    async def test_a_failing_extraction_write_never_breaks_the_turn(self, pipeline_factory):
        """_record_extraction's isolation (Site 5, the outcome='extracted' path)."""
        pipeline, _ = pipeline_factory(
            cache=_ExplodingCache(), extractor_returns=_extraction_with_python_entity()
        )

        result = await pipeline.extract_from_utterance(
            utterance=UTTERANCE,
            conversation_history=[],
            event_id="evt-boom",
            session_id="sess-1",
            recorded_at="2026-08-18T00:00:00+00:00",
        )

        # The turn completed AND produced the extraction -- not merely a
        # non-None placeholder, which a mutant short-circuiting to an empty
        # ValidationResult on cache failure would also satisfy.
        assert result.valid is True
        assert result.entities == _extraction_with_python_entity().entities
        assert result.relationships == []

    @pytest.mark.asyncio
    async def test_a_failing_skip_write_never_breaks_the_turn(self, pipeline_factory):
        """_record_skip's isolation (the four gate paths) -- the
        extraction-path test above only exercises _record_extraction's
        try/except; a narrowing mutant applied to only one of the two
        methods would be invisible without this.
        """
        pipeline, _ = pipeline_factory(cache=_ExplodingCache())

        result = await pipeline.extract_from_utterance(
            utterance="ok",  # two words -- trips Gate 0
            conversation_history=[],
            event_id="evt-boom-skip",
            session_id="sess-1",
            recorded_at="2026-08-18T00:00:00+00:00",
        )

        assert result.valid is True
        assert result.entities == []


class TestCacheWriteExceptionNarrowing:
    """ValueError/TypeError from a cache write must NOT be isolated.

    Both mean the CALLER passed something wrong -- ExtractionCache.put's own
    fail-closed guards on an inconsistent outcome/skip_reason pair
    (extraction_cache.py:173,176,178,180), or json.dumps on a
    non-serializable payload. Swallowing either reproduces the exact
    silent-failure mode Task 3's constructor pairing guard exists to
    prevent: no error, no log beyond a warning that hides the real cause,
    surfacing only much later as a ColdCacheError pointing at a rebuild
    rather than at this call.

    Each test below drives a DIFFERENT recorder method (_record_extraction
    via extractor_returns, _record_skip via force_gate) with its own cache
    double and its own distinct exception type -- per this phase's standing
    pytest.raises(match=...) rule, two adjacent checks must key on what
    DIFFERS, so a mutant narrowing only one of the two methods' except
    clauses cannot pass both.
    """

    @pytest.mark.asyncio
    async def test_extraction_recorder_propagates_value_error(self, pipeline_factory):
        class BadCache:
            def put(self, *a, **kw):
                raise ValueError("outcome='skipped' requires a skip_reason")

            def get(self, *a, **kw):
                return None

        pipeline, _ = pipeline_factory(
            cache=BadCache(), extractor_returns=_extraction_with_python_entity()
        )

        with pytest.raises(ValueError, match="skip_reason"):
            await pipeline.extract_from_utterance(
                utterance=UTTERANCE,
                conversation_history=[],
                event_id="evt-bad-cache-extraction",
                session_id="sess-1",
                recorded_at="2026-08-18T00:00:00+00:00",
            )

    @pytest.mark.asyncio
    async def test_skip_recorder_propagates_type_error(self, pipeline_factory):
        class BadCache:
            def put(self, *a, **kw):
                raise TypeError("Object of type bytes is not JSON serializable")

            def get(self, *a, **kw):
                return None

        pipeline, _ = pipeline_factory(cache=BadCache(), force_gate=SKIP_RATE_LIMITED)

        with pytest.raises(TypeError, match="JSON serializable"):
            await pipeline.extract_from_utterance(
                utterance=UTTERANCE,
                conversation_history=[],
                event_id="evt-bad-cache-skip",
                session_id="sess-1",
                recorded_at="2026-08-18T00:00:00+00:00",
            )
