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

import pytest

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
