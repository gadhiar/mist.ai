"""Extraction test fixtures."""

import hashlib
import time
from unittest.mock import AsyncMock

import pytest

from backend.knowledge.config import ExtractionConfig
from backend.knowledge.curation.graph_writer import RebuildStamps
from backend.knowledge.extraction.confidence import ConfidenceScorer
from backend.knowledge.extraction.normalizer import EntityNormalizer
from backend.knowledge.extraction.ontology_extractor import OntologyConstrainedExtractor
from backend.knowledge.extraction.pipeline import ExtractionPipeline
from backend.knowledge.extraction.preprocessor import PreProcessor
from backend.knowledge.extraction.temporal import TemporalResolver
from backend.knowledge.extraction.validator import ExtractionValidator
from backend.knowledge.extraction_cache import SKIP_DUPLICATE, SKIP_RATE_LIMITED
from backend.knowledge.storage.graph_store import GraphStore
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection
from tests.mocks.ollama import FakeLLM

_FORCEABLE_GATES = (SKIP_RATE_LIMITED, SKIP_DUPLICATE)


@pytest.fixture
def fake_llm():
    """A FakeLLM with default empty extraction response."""
    return FakeLLM()


class SpyCache:
    """Records what the pipeline decided, without a database.

    Records `created_at` alongside outcome/skip_reason: it must be the
    caller's C1 bitemporal `recorded_at`, never wall-clock `now()` -- a
    wall-clock value would make the cache row non-reproducible across a
    rebuild of the same log. A mutant that swaps in `datetime.now(UTC)`
    would satisfy every other assertion in this suite silently; only
    checking the recorded value here catches it.
    """

    def __init__(self):
        self.calls: list[tuple[str, str, str | None, str]] = []

    def put(self, event_id, ontology_version, extraction_version, model_hash, **kw):
        self.calls.append((event_id, kw["outcome"], kw.get("skip_reason"), kw["created_at"]))

    def get(self, *a, **kw):
        return None


# Fixed stamps for pipeline_factory-built pipelines. Values are arbitrary --
# tests assert on cache *calls*, never on the stamp values themselves -- and
# stable so a stamp typo can't masquerade as a passing test.
_TEST_STAMPS = RebuildStamps(
    ontology_version="test-ontology-v0",
    extraction_version="test-extraction-v0",
    model_hash="test-model-hash-v0",
)


def _unreachable_extractor() -> AsyncMock:
    """Extractor stub that fails loudly if Stage 2 is ever reached.

    Default for pipeline_factory(): every gated-skip test (Gate 0 included)
    must never call Stage 2, so a passing test that silently reached the LLM
    call would be a false green. Callers that need Stage 2 to run pass
    `extractor_returns=` explicitly instead.
    """
    mock = AsyncMock(spec=OntologyConstrainedExtractor)
    mock.extract.side_effect = AssertionError(
        "extractor.extract called -- a pre-extraction gate should have "
        "short-circuited before Stage 2 ran"
    )
    return mock


@pytest.fixture
def pipeline_factory():
    """Build an ExtractionPipeline wired to a SpyCache (or an injected `cache`).

    Collaborators mirror backend/factories.py build_extraction_pipeline, using
    the same real-stage-instances-plus-fakes pattern as
    tests/unit/knowledge/extraction/test_pipeline_dedup.py: FakeNeo4jConnection
    + FakeEmbeddingGenerator + GraphStore for storage/embeddings, real Stage
    1/3/4/5/6 processors, and a mocked Stage 2 extractor.

    Keyword args (stable for the whole extraction-cache-phase-1 phase -- later
    tasks add test functions that call this fixture, not new fixture params):
        force_gate: preset pipeline/config state so a specific pre-extraction
            gate trips on the next call. Settable values are
            `SKIP_RATE_LIMITED` and `SKIP_DUPLICATE` (both from
            backend.knowledge.extraction_cache). `SKIP_RATE_LIMITED` maps
            onto `rate_limit_max_per_minute=0` at construction time.
            `SKIP_DUPLICATE` (Task 4 addition) pre-seeds `_dedup_cache` with
            the exact-hash key of `dedup_utterance` -- the caller's first
            `extract_from_utterance(utterance=dedup_utterance, ...)` then
            trips Gate 3 via the exact-hash branch of `_check_dedup`
            (`grep -n "content_hash in self._dedup_cache" backend/knowledge/
            extraction/pipeline.py`), which runs before any embedding-based
            similarity comparison, so the seeded embedding value never
            matters. None (default) leaves every gate open. Two SKIP_*
            reasons remain deliberately NOT settable here:
            `SKIP_TOO_SHORT` depends on the utterance text passed to
            extract_from_utterance at call time (an under-3-word utterance),
            not on anything fixable at construction time.
            `SKIP_BELOW_SIGNIFICANCE` looks config-settable but is not:
            verified via a throwaway pipeline_factory(force_gate=...) run
            that `_SOURCE_THRESHOLDS["conversation"]` in pipeline.py
            (`grep -n "_SOURCE_THRESHOLDS: dict" backend/knowledge/extraction/pipeline.py`)
            hardcodes 0.3 for the default extraction_source and is read via
            `_SOURCE_THRESHOLDS.get(extraction_source, self._config.significance_threshold)`
            -- so `self._config.significance_threshold` is only consulted for
            an extraction_source absent from that table, never for the
            "conversation" default this fixture builds pipelines for. Drive
            it instead by passing a non-"conversation" `extraction_source`
            (its threshold IS in that table) at call time together with a
            low information-density utterance.
        dedup_utterance: required when `force_gate=SKIP_DUPLICATE`; the
            exact utterance text the caller will pass to
            `extract_from_utterance` next. Ignored otherwise.
        extractor_returns: an ExtractionResult for the mocked Stage-2
            extractor to return when reached. When None (default), Stage 2 is
            wired to fail the test if it is ever reached -- see
            `_unreachable_extractor`.
        cache: a cache double to inject in place of the default SpyCache.

    Returns:
        A `(pipeline, spy_cache)` tuple, where `spy_cache` is either the
        injected `cache` or a fresh `SpyCache`.
    """

    def _build(
        *,
        force_gate: str | None = None,
        dedup_utterance: str | None = None,
        extractor_returns=None,
        cache=None,
    ):
        spy_cache = cache if cache is not None else SpyCache()

        if force_gate is not None and force_gate not in _FORCEABLE_GATES:
            raise ValueError(
                f"force_gate={force_gate!r} is not settable via pipeline_factory -- "
                "see the fixture docstring for why"
            )
        if force_gate == SKIP_DUPLICATE and dedup_utterance is None:
            raise ValueError(
                "force_gate=SKIP_DUPLICATE requires dedup_utterance -- the exact "
                "utterance text the next extract_from_utterance call will use, so "
                "the fixture can pre-seed _dedup_cache with its exact-hash key"
            )

        embeddings = FakeEmbeddingGenerator()
        graph_store = GraphStore(connection=FakeNeo4jConnection(), embedding_generator=embeddings)

        extraction_config = ExtractionConfig(
            significance_threshold=0.0,
            rate_limit_max_per_minute=(0 if force_gate == SKIP_RATE_LIMITED else 1000),
            dedup_similarity_threshold=0.95,
            dedup_cache_size=200,
            dedup_cache_ttl_seconds=300,
        )

        if extractor_returns is not None:
            extractor = AsyncMock(spec=OntologyConstrainedExtractor)
            extractor.extract.return_value = extractor_returns
        else:
            extractor = _unreachable_extractor()

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
            embedding_provider=embeddings,
            extraction_config=extraction_config,
            extraction_cache=spy_cache,
            rebuild_stamps=_TEST_STAMPS,
        )

        if force_gate == SKIP_DUPLICATE:
            # Exact-hash seed only -- _check_dedup's hash branch runs before
            # any embedding comparison, so the paired "embedding" here is a
            # placeholder never read for this path.
            content_hash = hashlib.sha256(dedup_utterance.encode("utf-8")).hexdigest()
            pipeline._dedup_cache[content_hash] = ([], time.monotonic())

        return pipeline, spy_cache

    return _build
