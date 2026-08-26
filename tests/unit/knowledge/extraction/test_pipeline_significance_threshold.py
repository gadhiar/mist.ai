"""Tests for Gate 2's significance threshold resolution.

Subject: the `sig_threshold = _SOURCE_THRESHOLDS.get(extraction_source,
self._config.significance_threshold)` lookup in `ExtractionPipeline`
(`grep -n "sig_threshold = _SOURCE_THRESHOLDS.get" backend/knowledge/extraction/pipeline.py`).

These tests exist because `"conversation"` -- the DEFAULT extraction_source
(`grep -n 'extraction_source: str = "conversation"' backend/knowledge/extraction/pipeline.py`)
and the dominant production path -- used to carry a hardcoded entry in
`_SOURCE_THRESHOLDS`, so the env-settable `SIGNIFICANCE_THRESHOLD` /
`ExtractionConfig.significance_threshold` was never consulted for it. The
shadowing was silent because every authority happened to agree at 0.3
(`config.py` dataclass default, `config.py` from_env default, and `.env`),
so only a differing override could reveal it -- and the extraction-cache
phase made its consequence durable, since the significance decision is now
written to the extraction cache and inherited verbatim by every rebuild.

The per-source table itself is deliberately retained: `orchestrator_summary`
and `agent_tool_output` are the Command Center's planned ingest sources, and
their values encode an intended ORDERING (a pre-digested summary clears a
lower bar than conversation; noisy tool output a higher one). What changed is
that `"conversation"` is no longer a key, so it resolves through config like
any other unregistered source.
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
from backend.knowledge.extraction.pipeline import _SOURCE_THRESHOLDS, ExtractionPipeline
from backend.knowledge.extraction.preprocessor import PreProcessor
from backend.knowledge.extraction.temporal import TemporalResolver
from backend.knowledge.extraction.validator import ExtractionValidator
from backend.knowledge.storage.graph_store import GraphStore
from tests.mocks.config import TEST_EVENT_ID, TEST_SESSION_ID
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection

# NOT a module-level pytestmark: two of the tests below are synchronous, and a
# blanket asyncio mark makes pytest-asyncio warn on each of them.

# An utterance that clears the significance gate at the historical 0.3
# hardcode. Tests below drive the gate by moving the CONFIG threshold around
# this utterance's score, never by changing the utterance.
_SIGNIFICANT_UTTERANCE = "I use Rust for the embedded firmware at work."


def _empty_extractor() -> AsyncMock:
    mock = AsyncMock(spec=OntologyConstrainedExtractor)
    mock.extract.return_value = ExtractionResult(
        entities=[],
        relationships=[],
        raw_llm_output="{}",
        extraction_time_ms=1.0,
        source_utterance="",
    )
    return mock


def _build_pipeline(*, extractor, significance_threshold: float) -> ExtractionPipeline:
    """Build a pipeline whose ONLY interesting knob is the config threshold."""
    conn = FakeNeo4jConnection()
    embeddings = FakeEmbeddingGenerator()
    graph_store = GraphStore(connection=conn, embedding_generator=embeddings)

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
        extraction_config=ExtractionConfig(
            significance_threshold=significance_threshold,
            rate_limit_max_per_minute=1000,
            dedup_similarity_threshold=0.95,
            dedup_cache_size=200,
            dedup_cache_ttl_seconds=300,
        ),
        scope_classifier=None,
    )


@pytest.mark.asyncio
async def test_conversation_source_honors_configured_significance_threshold():
    """An unreachable config threshold skips extraction on the DEFAULT source.

    Fails before the fix: `_SOURCE_THRESHOLDS["conversation"] = 0.3` shadowed
    the config, so a 2.0 threshold was ignored and Stage 2 ran anyway.
    """
    extractor = _empty_extractor()
    pipeline = _build_pipeline(extractor=extractor, significance_threshold=2.0)

    await pipeline.extract_from_utterance(
        utterance=_SIGNIFICANT_UTTERANCE,
        conversation_history=[],
        event_id=TEST_EVENT_ID,
        session_id=TEST_SESSION_ID,
        # extraction_source deliberately omitted -- exercises the "conversation"
        # default, which is the path the shadowing bug lived on.
    )

    extractor.extract.assert_not_called()


@pytest.mark.asyncio
async def test_conversation_source_extracts_when_config_threshold_is_permissive():
    """The same utterance and source DO reach Stage 2 under a 0.0 threshold.

    Guards the pairing: without this, a fix that hardcoded "always skip"
    would satisfy the test above.
    """
    extractor = _empty_extractor()
    pipeline = _build_pipeline(extractor=extractor, significance_threshold=0.0)

    await pipeline.extract_from_utterance(
        utterance=_SIGNIFICANT_UTTERANCE,
        conversation_history=[],
        event_id=TEST_EVENT_ID,
        session_id=TEST_SESSION_ID,
    )

    extractor.extract.assert_called_once()


def test_conversation_is_absent_from_the_source_threshold_table():
    """Structural guard: re-adding a "conversation" key reintroduces the bug.

    The behavioural tests above would also catch it, but this one names the
    cause rather than the symptom, so a regression reads as "the key came
    back" instead of "extraction ran when it should not have".
    """
    assert "conversation" not in _SOURCE_THRESHOLDS


def test_source_thresholds_bracket_the_configured_conversation_threshold():
    """The Command Center ordering holds AGAINST CONFIG, not against 0.3.

    orchestrator_summary (pre-digested -> lower bar)
        < conversation (from config)
        < agent_tool_output (noisy -> higher bar)

    Fails before the fix at a non-default config value: conversation was
    pinned to the hardcoded 0.3 and did not track config at all, so the
    bracket silently described a constant rather than the live threshold.
    """
    conversation_threshold = ExtractionConfig(significance_threshold=0.35).significance_threshold
    resolved = _SOURCE_THRESHOLDS.get("conversation", conversation_threshold)

    assert _SOURCE_THRESHOLDS["orchestrator_summary"] < resolved
    assert resolved < _SOURCE_THRESHOLDS["agent_tool_output"]
    assert resolved == 0.35


def test_default_config_threshold_matches_the_retired_hardcode():
    """Pins the claim that this fix is a no-op on current production data.

    The dataclass default, the from_env default and `.env` all say 0.3, which
    is exactly the value `_SOURCE_THRESHOLDS["conversation"]` used to hardcode
    -- so removing that entry changes no effective threshold today, invalidates
    no cached extraction decision, and cannot diverge a rebuild. This test is a
    characterisation test: it passed before the fix and after, deliberately.
    It fails the moment someone changes the dataclass default, which is exactly
    when the no-op claim above stops being true.
    """
    assert ExtractionConfig().significance_threshold == 0.3
