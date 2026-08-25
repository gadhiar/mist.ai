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
from datetime import datetime

import pytest

from backend.knowledge.extraction.confidence import ConfidenceScorer
from backend.knowledge.extraction.normalizer import EntityNormalizer
from backend.knowledge.extraction.ontology_extractor import ExtractionResult
from backend.knowledge.extraction.temporal import TemporalResolver
from backend.knowledge.extraction.validator import ExtractionValidator

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
async def test_replay_never_touches_the_normalizers_dead_dependencies(regenerator_factory):
    """Stages 3-6 are pure (Task 2). If a model is reached, this raises.

    Narrower than it sounds: `EntityNormalizer` is the ONLY one of the four Stage 3-6
    components with a constructor parameter that could hold anything model- or
    graph-shaped (`embedding_generator`, `executor`) -- `ConfidenceScorer()` and
    `TemporalResolver()` take no arguments at all, and `ExtractionValidator()`'s only
    argument is a float threshold. So this test exercises purity for exactly one slot
    pair; for the other three components the property holds by construction (there is
    nothing to reach), not because this test ran anything against them.
    """

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


# Fields `LogRegenerator.rebuild()`'s replay loop actually populates on the
# `ExtractionResult` it builds from the cached row -- must be kept in sync with that
# constructor call (`entities=`, `relationships=`, `source_utterance=`). Everything
# else `ExtractionResult` carries (`raw_llm_output`, `extraction_time_ms`,
# `source_metadata`) is something replay does NOT supply; a stage reading one of
# those would silently diverge live from rebuilt exactly the way `source_utterance`
# did before fix round 1.
_FIELDS_REPLAY_SUPPLIES = frozenset({"entities", "relationships", "source_utterance"})


class _FieldAccessGuardedExtractionResult(ExtractionResult):
    """Raises the instant a field replay does not supply is read.

    Structural guard against the defect class fix round 1 closed recurring: nothing
    enforces that Stages 3-6 only read the three fields above. Before that fix, a
    stage reading `source_utterance` got `""` silently -- no exception, no test
    failure, just a rebuilt graph that quietly disagreed with the live one. A future
    stage reading `raw_llm_output`, `extraction_time_ms`, or `source_metadata` (none
    of which replay supplies AT ALL, not even as an empty default standing in for a
    real value) would fail the identical way. This guard turns that into an
    immediate, loud failure at the read site instead of a silent divergence
    discovered later by comparing graphs.

    Deliberately a RUNTIME behavioral check (`__getattribute__`), not a source-text
    scan. `TestNoUnguardedEarlyReturn` (test_pipeline_cache_writes.py) counts a
    literal string across `inspect.getsource(...)` and documents its own failure
    mode: a comment or docstring containing that literal token inflates the count
    and masks a real violation behind a false-passing equality. Intercepting the
    actual attribute read has no equivalent hole -- a comment mentioning
    `raw_llm_output` cannot trigger `__getattribute__`; only real code reading
    `extraction.raw_llm_output` can.

    Known non-coverage, stated rather than hidden (Task 4's lesson: say what a guard
    does and does not catch):
    - `object.__getattribute__(extraction, "raw_llm_output")`, called directly on
      the base class rather than through the instance, bypasses this override. Not
      a realistic shape for the four stage methods this guards.
    - `vars(extraction)`, `extraction.__dict__[...]`, and `dataclasses.asdict(extraction)`
      all read `__dict__` directly and never invoke `__getattribute__`.
    - A field added to `ExtractionResult` in the future, under a name that happens
      to already be in `_FIELDS_REPLAY_SUPPLIES`, would not be caught -- only the
      three fields known today to be unsupplied are guarded, by name.
    """

    def __getattribute__(self, name: str):
        if name in {"raw_llm_output", "extraction_time_ms", "source_metadata"}:
            raise AssertionError(
                f"Stage 3-6 code read ExtractionResult.{name}, which "
                "LogRegenerator.rebuild()'s replay loop does not supply. Either the "
                "read is a bug, or replay must start supplying this field from the "
                "cached row or the replayed turn -- otherwise a rebuilt graph will "
                "silently disagree with the live one, the way source_utterance did "
                "before fix round 1."
            )
        return super().__getattribute__(name)

    def __repr__(self) -> str:
        # The dataclass-generated __repr__ reads every field, including the three
        # this class exists to forbid -- so a failure's traceback trying to render
        # `self` would re-trigger the guard mid-repr (pytest catches this and prints
        # "raised in repr()", which obscures the real failure). Report only the
        # fields this class allows a read of.
        return (
            f"{type(self).__name__}(entities={self.entities!r}, "
            f"relationships={self.relationships!r}, source_utterance={self.source_utterance!r})"
        )


@pytest.mark.asyncio
async def test_stages_3_to_6_only_read_fields_replay_actually_supplies():
    """The structural guard: run the REAL Stages 3-6 against a field-access trap.

    Not a LogRegenerator test -- it exercises the four stage components directly,
    the same way `LogRegenerator.rebuild()`'s loop calls them (same method names,
    same argument shapes), against an `ExtractionResult` that raises the instant
    anything outside `_FIELDS_REPLAY_SUPPLIES` is read. No assertion beyond "this
    completes without raising": the guard class itself is the assertion mechanism.
    """
    guarded = _FieldAccessGuardedExtractionResult(
        entities=[
            {"id": "user", "type": "User"},
            {"id": "rust", "type": "Technology"},
        ],
        relationships=[
            {"source": "user", "target": "rust", "type": "USES", "properties": {"confidence": 0.9}}
        ],
        source_utterance="I use Rust.",
    )

    extraction = ConfidenceScorer().adjust_confidence(guarded)
    extraction = TemporalResolver().resolve(
        extraction, datetime.fromisoformat("2026-07-01T09:00:00+00:00")
    )
    extraction = await EntityNormalizer(embedding_generator=None, executor=None).normalize(
        extraction
    )
    ExtractionValidator().validate(extraction)
