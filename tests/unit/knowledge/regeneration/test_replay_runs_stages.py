"""Replay must re-run Stages 3-6, not treat the cached payload as validated.

Spec D2: the cache holds RAW Stage-2 output. If replay skipped these stages the
ontology's effects would never be applied and every ontology bump would silently
produce the old graph.

The task brief's Step 1 draft included a standalone `RecordingStage` class (a
sync `__call__` that appends a name to a sink and returns its argument
unchanged) as an illustration of the fake shape `regenerator_factory` needs.
It is not defined here: none of the three tests below reference it by name,
and Stage 5's `normalize()` is async (`await self._normalizer.normalize(...)`
in `LogRegenerator.rebuild`), so a single sync callable cannot stand in for
all four stages regardless. The recording fakes actually used --
`_RecordingConfidenceScorer`, `_RecordingTemporalResolver`,
`_RecordingNormalizer`, `_RecordingValidator` -- live in `conftest.py` next to
`regenerator_factory`, the fixture that builds them.
"""

import copy

import pytest

from backend.knowledge.extraction.confidence import ConfidenceScorer
from backend.knowledge.extraction.ontology_extractor import ExtractionResult

# `rebuild()` is keyword-only and `live_uri` and `epoch` have NO defaults
# (`grep -n "async def rebuild" backend/knowledge/regeneration/log_regenerator.py`, one
# hit -- read the signature there). Calling it with staging_uri alone raises TypeError
# before any assertion runs. `epoch` must carry the same stamp fields the cache is
# keyed on, or _assert_cache_coverage misses on every turn.
REBUILD_ARGS = {
    "staging_uri": "bolt://mist-neo4j-staging:7687",
    "live_uri": "bolt://mist-neo4j:7687",
    "epoch": {
        "epoch_id": 1,
        "ontology_version": "1.4.0",
        "extraction_version": "2026-06-14-r5",
        "model_hash": "test-model-hash",
        "activated_at": "2026-08-18T00:00:00+00:00",
    },
}


@pytest.mark.asyncio
async def test_replay_runs_every_post_llm_stage_in_order(regenerator_factory):
    order: list[str] = []
    regenerator = regenerator_factory(stage_sink=order)
    await regenerator.rebuild(**REBUILD_ARGS)
    assert order == ["confidence", "temporal", "normalize", "validate"]


@pytest.mark.asyncio
async def test_a_skipped_row_replays_as_a_no_op_not_a_failure(regenerator_factory):
    """D1: a 'skipped' row is a recorded decision. Replay honours it silently.

    `report.turns_processed`/`turns_failed` alone cannot distinguish "short-circuited
    before curation" from "ran an empty-but-valid extraction through curation": the
    fake curation pipeline records zero stage_errors either way, so both paths give
    turns_processed=1, turns_failed=0. The extra assertion on `event_ids` is what
    actually proves curate_and_store was never reached for a skipped turn -- reverting
    the skip short-circuit in `LogRegenerator.rebuild` turns this line red while
    leaving the two counters above green.
    """
    regenerator = regenerator_factory(cached_outcome="skipped", skip_reason="rate_limited")
    report = await regenerator.rebuild(**REBUILD_ARGS)
    assert report.turns_processed == 1
    assert report.turns_failed == 0
    assert regenerator._curation.event_ids == []


@pytest.mark.asyncio
async def test_replay_needs_no_model(regenerator_factory):
    """Stages 3-6 are pure (Task 2). If a model is reached, this raises."""

    class ExplodingLLM:
        def __getattr__(self, _):
            raise AssertionError("replay reached a model")

    regenerator = regenerator_factory(llm=ExplodingLLM())
    await regenerator.rebuild(**REBUILD_ARGS)


@pytest.mark.asyncio
async def test_a_hedged_utterance_gets_the_same_confidence_penalty_on_replay(regenerator_factory):
    """Fix round 1: source_utterance must flow from the replayed turn into Stage 3.

    Live: `ConfidenceScorer.adjust_confidence` reads `extraction.source_utterance` to
    detect hedge words ("I think"/"I guess"/...) and apply a penalty
    (`ontology_extractor.py:135,159` set `source_utterance=pre_processed.original_text`,
    and `PreProcessor.pre_process` sets `original_text=utterance` unconditionally --
    `preprocessor.py:132`, no branch transforms it first). The cache holds no
    `source_utterance` at all (spec D2 -- it's raw Stage-2 output: entities and
    relationships only). Without it, replay's Stage 3 call sees `source_utterance=""`
    and the hedge penalty never fires, so a rebuilt graph would disagree with the live
    one on every hedged relationship -- exactly what R1.6's `live == rebuilt` gate
    exists to catch.

    `turn["user_utterance"]` is the fix: `conversation_handler.py`'s
    `_record_turn_event` writes `user_utterance=user_message` to the event store, and
    the SAME `user_message` variable is passed as `utterance=` to
    `_extract_knowledge_async` -> `extract_from_utterance` -> `pre_process`, so the
    replayed turn's `user_utterance` is byte-identical to what fed the live
    extraction.
    """
    utterance = "I think Alice uses Rust."
    base_confidence = 0.9
    entities = [
        {"id": "user", "type": "User"},
        {"id": "rust", "type": "Technology"},
    ]
    relationship = {
        "source": "user",
        "target": "rust",
        "type": "USES",
        "properties": {"confidence": base_confidence},
    }

    # Oracle: the exact pure function the live pipeline calls, given the exact
    # source_utterance the live pipeline would have had -- a deep copy, since
    # adjust_confidence mutates in place and the original must stay pristine
    # (D2: the cache holds the RAW, pre-confidence-adjustment relationship).
    expected = ConfidenceScorer().adjust_confidence(
        ExtractionResult(relationships=[copy.deepcopy(relationship)], source_utterance=utterance)
    )
    expected_confidence = expected.relationships[0]["properties"]["confidence"]
    assert expected_confidence == pytest.approx(0.75), "sanity: the hedge penalty must fire"

    regenerator = regenerator_factory(
        utterance=utterance, entities=entities, relationships=[relationship]
    )
    await regenerator.rebuild(**REBUILD_ARGS)

    replayed_relationships = regenerator._curation.relationships[0]
    assert replayed_relationships, "the USES relationship must have survived validation"
    assert replayed_relationships[0]["properties"]["confidence"] == expected_confidence
