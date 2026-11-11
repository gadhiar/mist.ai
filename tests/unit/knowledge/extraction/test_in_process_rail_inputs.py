"""The in-process apply hands the rails the scope it recorded on the cache row.

`extract_from_utterance` records the Stage 1.5 scope on the turn's cache row and then
runs Stages 3-6 itself. A replay of that row hands its scope to `normalize`
(`rail_inputs`), so the first apply has to make the same call: a scope-gated rule then
fires at both or at neither. `live_rail_inputs` is that call's keyword arguments.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from backend.knowledge.config import ExtractionConfig
from backend.knowledge.curation.graph_writer import RebuildStamps
from backend.knowledge.extraction.confidence import ConfidenceScorer
from backend.knowledge.extraction.normalizer import live_rail_inputs, rail_inputs
from backend.knowledge.extraction.ontology_extractor import (
    ExtractionResult,
    OntologyConstrainedExtractor,
)
from backend.knowledge.extraction.pipeline import ExtractionPipeline, TurnToApply
from backend.knowledge.extraction.preprocessor import PreProcessedInput, PreProcessor
from backend.knowledge.extraction.scope_classifier import ScopeResult
from backend.knowledge.extraction.temporal import TemporalResolver
from backend.knowledge.extraction.validator import ExtractionValidator
from backend.knowledge.extraction_cache import ExtractionCache
from backend.knowledge.storage.graph_store import GraphStore
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection
from tests.unit.extraction_backlog.fakes import FakeGraphCuration

_STAMPS = RebuildStamps(
    ontology_version="1.4.0",
    extraction_version="2026-06-14-r5",
    model_hash="test-model-hash",
)
_STAMP = "2026-07-01T09:00:00+00:00"
_EVENT_ID = "evt-s-a-0"
_SESSION_ID = "s-a"
_UTTERANCE = "Ilse keeps a boat in the harbour."
_SCOPE = "user-scope"


class _RecordingNormalizer:
    """Stands in for `EntityNormalizer` and keeps the keyword arguments of each call."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def normalize(self, extraction, **row):
        self.calls.append(row)
        return extraction


class _Classifier:
    """A Stage 1.5 double whose every result carries the scope it was built with."""

    def __init__(self, scope: str | None) -> None:
        self._scope = scope

    async def classify(self, pre_processed) -> ScopeResult:
        return ScopeResult(scope=self._scope, confidence=0.9)  # type: ignore[arg-type]


class _NoProgress:
    curated = False

    def mark_curated(self) -> None:
        pass

    def mark_applied(self) -> None:
        pass


def _extractor() -> AsyncMock:
    """A Stage 2 double with one entity, so the turn reaches Stage 5."""
    extractor = AsyncMock(spec=OntologyConstrainedExtractor)
    extractor.extract.return_value = ExtractionResult(
        entities=[{"id": "user", "type": "User", "name": "User"}],
        relationships=[],
        raw_llm_output="{}",
        extraction_time_ms=1.0,
        source_utterance=_UTTERANCE,
    )
    return extractor


def _pipeline(normalizer, cache: ExtractionCache, scope_classifier) -> ExtractionPipeline:
    embeddings = FakeEmbeddingGenerator()
    cache.initialize()
    return ExtractionPipeline(
        preprocessor=PreProcessor(),
        extractor=_extractor(),
        confidence_scorer=ConfidenceScorer(),
        temporal_resolver=TemporalResolver(),
        normalizer=normalizer,
        validator=ExtractionValidator(),
        graph_store=GraphStore(FakeNeo4jConnection(), embeddings),
        curation_pipeline=FakeGraphCuration(),  # type: ignore[arg-type]
        embedding_provider=embeddings,
        extraction_config=ExtractionConfig(
            significance_threshold=0.0,
            rate_limit_max_per_minute=1000,
            dedup_similarity_threshold=0.95,
            dedup_cache_size=200,
            dedup_cache_ttl_seconds=300,
        ),
        extraction_cache=cache,
        rebuild_stamps=_STAMPS,
        scope_classifier=scope_classifier,
    )


async def _apply_in_process(pipeline: ExtractionPipeline) -> None:
    await pipeline.extract_from_utterance(
        utterance=_UTTERANCE,
        conversation_history=[],
        event_id=_EVENT_ID,
        session_id=_SESSION_ID,
        recorded_at=_STAMP,
    )


@pytest.mark.parametrize(
    ("metadata", "scope"),
    [
        ({"subject_scope": _SCOPE, "subject_scope_confidence": 0.9}, _SCOPE),
        ({"subject_scope": None, "subject_scope_confidence": 0.9}, None),
        ({}, None),
    ],
    ids=["a_scope", "a_scope_of_none", "no_classifier"],
)
def test_live_rail_inputs_are_the_rail_inputs_of_the_row_the_turn_records(metadata, scope):
    pre_processed = PreProcessedInput(
        original_text=_UTTERANCE,
        resolved_text=_UTTERANCE,
        conversation_context=[],
        reference_date=datetime.fromisoformat(_STAMP),
        turn_index=0,
        metadata=metadata,
    )

    handed = live_rail_inputs(pre_processed)

    assert handed == rail_inputs({"scope": scope, "batch_record": None, "turn_plan": None})
    assert handed == ({"scope": scope} if scope is not None else {})


@pytest.mark.asyncio
async def test_the_in_process_apply_hands_the_classified_scope_to_the_normalizer():
    normalizer = _RecordingNormalizer()
    pipeline = _pipeline(normalizer, ExtractionCache(":memory:"), _Classifier(_SCOPE))

    await _apply_in_process(pipeline)

    assert normalizer.calls == [{"scope": _SCOPE}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scope_classifier", [None, _Classifier(None)], ids=["no_classifier", "scope_of_none"]
)
async def test_the_in_process_apply_without_a_scope_calls_the_normalizer_as_before(
    scope_classifier,
):
    normalizer = _RecordingNormalizer()
    pipeline = _pipeline(normalizer, ExtractionCache(":memory:"), scope_classifier)

    await _apply_in_process(pipeline)

    assert normalizer.calls == [{}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope_classifier", "handed"),
    [(_Classifier(_SCOPE), {"scope": _SCOPE}), (None, {})],
    ids=["with_a_scope", "without_one"],
)
async def test_a_replay_of_the_row_an_in_process_apply_wrote_makes_the_same_call(
    scope_classifier, handed
):
    normalizer = _RecordingNormalizer()
    cache = ExtractionCache(":memory:")
    pipeline = _pipeline(normalizer, cache, scope_classifier)

    await _apply_in_process(pipeline)
    row = cache.get(_EVENT_ID, _STAMPS.extraction_version, _STAMPS.model_hash)
    assert row is not None, "the in-process apply left no cache row"
    await pipeline.apply_cached_turn(
        TurnToApply(
            event_id=_EVENT_ID,
            session_id=_SESSION_ID,
            user_utterance=_UTTERANCE,
            recorded_at=_STAMP,
        ),
        row,
        _NoProgress(),
    )

    in_process, replayed = normalizer.calls
    assert in_process == replayed == handed
