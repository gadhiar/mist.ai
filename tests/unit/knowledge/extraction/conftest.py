"""Extraction test fixtures."""

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
from backend.knowledge.extraction_cache import SKIP_RATE_LIMITED
from backend.knowledge.storage.graph_store import GraphStore
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection
from tests.mocks.ollama import FakeLLM


@pytest.fixture
def fake_llm():
    """A FakeLLM with default empty extraction response."""
    return FakeLLM()


class SpyCache:
    """Records what the pipeline decided, without a database."""

    def __init__(self):
        self.calls: list[tuple[str, str, str | None]] = []

    def put(self, event_id, ontology_version, extraction_version, model_hash, **kw):
        self.calls.append((event_id, kw["outcome"], kw.get("skip_reason")))

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
            gate trips on the next call. Only `SKIP_RATE_LIMITED` (from
            backend.knowledge.extraction_cache) is settable this way -- it
            maps cleanly onto `rate_limit_max_per_minute=0` at construction
            time. None (default) leaves every gate open. The other three
            SKIP_* reasons are deliberately NOT settable here:
            `SKIP_TOO_SHORT` and `SKIP_DUPLICATE` depend on the utterance
            text passed to extract_from_utterance at call time (an
            under-3-word utterance; a second call repeating the first's
            utterance), not on anything fixable at construction time.
            `SKIP_BELOW_SIGNIFICANCE` looks config-settable but is not:
            verified via a throwaway pipeline_factory(force_gate=...) run
            that `_SOURCE_THRESHOLDS["conversation"]` in pipeline.py
            (`grep -n "_SOURCE_THRESHOLDS: dict" backend/knowledge/extraction/pipeline.py`)
            hardcodes 0.3 for the default extraction_source and is read via
            `_SOURCE_THRESHOLDS.get(extraction_source, self._config.significance_threshold)`
            -- so `self._config.significance_threshold` is only consulted for
            an extraction_source absent from that table, never for the
            "conversation" default this fixture builds pipelines for.
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
        extractor_returns=None,
        cache=None,
    ):
        spy_cache = cache if cache is not None else SpyCache()

        if force_gate is not None and force_gate != SKIP_RATE_LIMITED:
            raise ValueError(
                f"force_gate={force_gate!r} is not settable via pipeline_factory -- "
                "see the fixture docstring for why"
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

        return pipeline, spy_cache

    return _build
