"""The golden log replays: 100% cache coverage, turns processed > 0, deterministic stream.

## What this proves, and what it does not

The live log is empty (0 sessions, 0 turns), so before this fixture existed the replay loop
in `LogRegenerator.rebuild` never executed, the cold-cache coverage gate was satisfied
vacuously (100% coverage of an empty set), and the determinism gate compared seed to seed.
These tests exercise the real `LogRegenerator` against a real `EventStore` and a real
`ExtractionCache` -- both isolated SQLite under `tmp_path` -- over 87 authored turns.

The determinism assertion here is over the PAYLOAD STREAM the replay feeds to curation, not
over a rebuilt graph. Reconciling into a graph needs a staging Neo4j, which is out of scope
for a unit test; the graph-level `assert_rebuild_twice_identical` lives in
`tests/integration/knowledge/test_golden_log_rebuild.py` and needs the staging profile up.
So: this proves the replay is deterministic in what it hands forward. It does not prove the
reconciliation of that stream is deterministic -- that is the integration test's job.

`RecordingCurationPipeline` stands in for the graph-backed pipeline. It is an explicit fake
(never MagicMock) and returns a real `CurationResult`, so a signature change in
`curate_and_store` fails here rather than being silently absorbed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from backend.knowledge.curation.deduplication import DeduplicationResult
from backend.knowledge.curation.graph_writer import WriteResult
from backend.knowledge.curation.pipeline import CurationResult
from backend.knowledge.curation.reconciliation import ReconcileTurnResult
from backend.knowledge.extraction.validator import ValidationResult
from backend.knowledge.regeneration.log_regenerator import ColdCacheError, LogRegenerator
from backend.knowledge.regeneration.rebuild_gate import (
    RebuildDeterminismError,
    assert_rebuild_twice_identical,
)
from scripts.golden_log.generate import build_golden_turns, materialize_isolated
from scripts.golden_log.native_shape import native_predicate, native_properties

EXPECTED_TURN_COUNT = 87
STAGING_URI = "bolt://golden-log-staging:7687"
LIVE_URI = "bolt://mist-neo4j:7687"

# `generate.SESSION_ORIGIN` is "test": the golden log is fixture traffic and is
# "never counted as genuine usage by anything that filters on origin". Since
# `rebuild` now reads that guard and defaults to `('real',)`, replaying the
# fixture requires saying so. Passing this explicitly is the point -- a default
# that quietly swept fixture traffic into a canonical rebuild is the L2 bug.
GOLDEN_LOG_ORIGINS = ("test",)


@dataclass(slots=True)
class ReplayedTurn:
    """One `curate_and_store` call, captured verbatim."""

    event_id: str
    session_id: str
    recorded_at: str
    entities: list[dict[str, Any]]
    relationships: list[dict[str, Any]]


@dataclass(slots=True)
class RecordingCurationPipeline:
    """Records what the replay loop hands to curation. Substitutes for the graph pipeline."""

    calls: list[ReplayedTurn] = field(default_factory=list)

    async def curate_and_store(
        self,
        validation_result: ValidationResult,
        event_id: str,
        session_id: str,
        source_metadata: Any = None,
        recorded_at: str | None = None,
    ) -> CurationResult:
        self.calls.append(
            ReplayedTurn(
                event_id=event_id,
                session_id=session_id,
                recorded_at=recorded_at or "",
                entities=validation_result.entities,
                relationships=validation_result.relationships,
            )
        )
        return CurationResult(
            write_result=WriteResult(),
            dedup_result=DeduplicationResult(entities=[], merge_actions=[], entities_merged=0),
            reconcile_result=ReconcileTurnResult(),
            curation_time_ms=0.0,
            validated_entities=validation_result.entities,
            validated_relationships=validation_result.relationships,
        )


def canonical_replay_form(calls: list[ReplayedTurn]) -> str:
    """Deterministic serialization of a replay stream, one turn per line.

    Same shape of guarantee `canonical_graph_form` gives for a graph: stable ordering, no
    wall-clock, so two replays of the same (log, epoch) can be diffed for byte-identity.
    """
    return "\n".join(
        json.dumps(
            {
                "event_id": call.event_id,
                "session_id": call.session_id,
                "recorded_at": call.recorded_at,
                "entities": call.entities,
                "relationships": call.relationships,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        for call in calls
    )


async def replay_golden_log(root, turns=None) -> tuple[Any, RecordingCurationPipeline, Any]:
    """Materialize the golden log under `root` and rebuild from it. Returns the report."""
    materialized = materialize_isolated(turns or build_golden_turns(), root=root)
    recorder = RecordingCurationPipeline()
    regenerator = LogRegenerator(
        event_store=materialized.event_store,
        extraction_cache=materialized.extraction_cache,
        staging_curation_pipeline=recorder,
    )
    report = await regenerator.rebuild(
        staging_uri=STAGING_URI,
        live_uri=LIVE_URI,
        epoch=materialized.epoch,
        origins=GOLDEN_LOG_ORIGINS,
    )
    return report, recorder, materialized


@pytest.fixture(scope="module")
def golden_turns():
    return build_golden_turns()


class TestTheOriginGuardIsLoadBearing:
    """`origins=GOLDEN_LOG_ORIGINS` must be a guard, not a decoration.

    Every other test in this file passes `origins=("test",)` explicitly. If the default were
    later widened back to include fixture traffic, all of them would keep passing -- the
    argument still present, still spelled correctly, and no longer doing anything. That is
    the same green-mutation shape as the L2 bug this argument exists to fix: the `origin`
    discriminator was written, read back, and tested for two rounds while no selection ever
    consulted it.

    This is the only test that fails when the default stops excluding fixture traffic.
    """

    @pytest.mark.asyncio
    async def test_the_default_origins_replay_none_of_the_golden_log(self, tmp_path, golden_turns):
        # Arrange: the same 87-turn fixture every other test in this file replays, with the
        # cache warmed under the epoch triple -- so nothing but the origin guard can keep a
        # turn out of the replay.
        materialized = materialize_isolated(golden_turns, root=tmp_path / "iso")
        recorder = RecordingCurationPipeline()
        regenerator = LogRegenerator(
            event_store=materialized.event_store,
            extraction_cache=materialized.extraction_cache,
            staging_curation_pipeline=recorder,
        )

        # Act: no `origins` -- whatever a rebuild of the CANONICAL graph does by default.
        report = await regenerator.rebuild(
            staging_uri=STAGING_URI, live_uri=LIVE_URI, epoch=materialized.epoch
        )

        # Assert: the log is populated and the canonical rebuild replayed none of it.
        # `generate.SESSION_ORIGIN` is "test"; a canonical rebuild is not fixture traffic's.
        assert materialized.event_store.get_turn_count() == EXPECTED_TURN_COUNT
        assert report.turns_processed == 0
        assert recorder.calls == []


class TestReplayIsNotVacuous:
    """The vacuity guard. Everything else here would pass over an empty log."""

    @pytest.mark.asyncio
    async def test_replay_loop_actually_executes(self, tmp_path, golden_turns):
        # Act
        report, recorder, _ = await replay_golden_log(tmp_path / "iso", golden_turns)

        # Assert: the failure this fixture exists to make impossible.
        assert report.turns_processed > 0
        assert report.turns_processed == EXPECTED_TURN_COUNT
        assert len(recorder.calls) == EXPECTED_TURN_COUNT

    @pytest.mark.asyncio
    async def test_no_turn_fails_curation(self, tmp_path, golden_turns):
        report, _recorder, _ = await replay_golden_log(tmp_path / "iso", golden_turns)
        assert report.turns_failed == 0

    @pytest.mark.asyncio
    async def test_job_ledger_records_a_completed_run_over_every_turn(self, tmp_path, golden_turns):
        # Assert: the checkpoint cursor advanced across the whole log, not just row 1.
        report, _recorder, materialized = await replay_golden_log(tmp_path / "iso", golden_turns)
        job = materialized.event_store.get_reextraction_job(report.job_id)

        assert job is not None
        assert job["status"] == "completed"
        assert job["processed"] == EXPECTED_TURN_COUNT
        assert job["total_events"] == EXPECTED_TURN_COUNT


class TestCacheCoverage:
    @pytest.mark.asyncio
    async def test_rebuild_does_not_raise_cold_cache(self, tmp_path, golden_turns):
        # Assert: 100% coverage over 87 real turns, not over an empty set.
        report, _recorder, _ = await replay_golden_log(tmp_path / "iso", golden_turns)
        assert report.turns_processed == EXPECTED_TURN_COUNT

    @pytest.mark.asyncio
    async def test_a_hole_in_the_cache_still_refuses(self, tmp_path, golden_turns):
        # Arrange: prove the coverage gate is live over this corpus rather than assumed.
        # A stamp triple the cache was never warmed for is exactly a cold cache.
        materialized = materialize_isolated(golden_turns, root=tmp_path / "iso")
        drifted_epoch = {**materialized.epoch, "extraction_version": "2026-01-01-r0"}
        regenerator = LogRegenerator(
            event_store=materialized.event_store,
            extraction_cache=materialized.extraction_cache,
            staging_curation_pipeline=RecordingCurationPipeline(),
        )

        # Act / Assert
        with pytest.raises(ColdCacheError, match="uncached"):
            await regenerator.rebuild(
                staging_uri=STAGING_URI,
                live_uri=LIVE_URI,
                epoch=drifted_epoch,
                origins=GOLDEN_LOG_ORIGINS,
            )

    @pytest.mark.asyncio
    async def test_a_stamp_mismatch_is_distinguishable_from_a_genuinely_cold_turn(
        self, tmp_path, golden_turns
    ):
        # A stamp mismatch and a real cache hole raise the same ColdCacheError, so a proof
        # that cannot tell them apart will misattribute the failure. Two things separate
        # them here. First, `materialize_isolated` verifies coverage under the triple it
        # just wrote, so after it returns, a hole cannot be the cause. Second, the error
        # names the stamps it looked under.
        materialized = materialize_isolated(golden_turns, root=tmp_path / "iso")
        drifted = {**materialized.epoch, "extraction_version": "2026-01-01-r0"}
        regenerator = LogRegenerator(
            event_store=materialized.event_store,
            extraction_cache=materialized.extraction_cache,
            staging_curation_pipeline=RecordingCurationPipeline(),
        )

        with pytest.raises(ColdCacheError) as excinfo:
            await regenerator.rebuild(
                staging_uri=STAGING_URI,
                live_uri=LIVE_URI,
                epoch=drifted,
                origins=GOLDEN_LOG_ORIGINS,
            )

        message = str(excinfo.value)
        # ALL turns cold, not some: the signature of a stamp mismatch rather than a hole.
        assert f"{EXPECTED_TURN_COUNT} of {EXPECTED_TURN_COUNT} turns are uncached" in message
        assert "extraction=2026-01-01-r0" in message

    @pytest.mark.asyncio
    async def test_the_cold_cache_message_does_not_name_model_hash(self, tmp_path, golden_turns):
        # Known diagnostic gap, pinned rather than left to be rediscovered. `model_hash` is
        # a third of the cache key and the two triples in this system DIFFER in it
        # (`factories.py` composes "|emb:<embedding model>"; epoch rows carry the bare
        # value). A model_hash mismatch therefore produces a ColdCacheError that reports
        # matching ontology and extraction versions and gives no hint at the real cause.
        materialized = materialize_isolated(golden_turns, root=tmp_path / "iso")
        drifted = {
            **materialized.epoch,
            "model_hash": f"{materialized.epoch['model_hash']}|emb:all-MiniLM-L6-v2",
        }
        regenerator = LogRegenerator(
            event_store=materialized.event_store,
            extraction_cache=materialized.extraction_cache,
            staging_curation_pipeline=RecordingCurationPipeline(),
        )

        with pytest.raises(ColdCacheError) as excinfo:
            await regenerator.rebuild(
                staging_uri=STAGING_URI,
                live_uri=LIVE_URI,
                epoch=drifted,
                origins=GOLDEN_LOG_ORIGINS,
            )

        assert "emb:" not in str(excinfo.value), (
            "ColdCacheError now names model_hash -- delete this test and stop warning about "
            "the blind spot in the report"
        )


class TestReplayedContent:
    @pytest.mark.asyncio
    async def test_turns_arrive_in_log_order_with_authored_timestamps(self, tmp_path, golden_turns):
        # Assert: recorded_at is the authored fact-time, not a wall-clock read. R1.5 reads
        # elapsed time off these, so a wall-clock substitution would silently destroy the
        # entire gap schedule.
        _report, recorder, _ = await replay_golden_log(tmp_path / "iso", golden_turns)

        assert [c.event_id for c in recorder.calls] == [t.event_id for t in golden_turns]
        assert [c.recorded_at for c in recorder.calls] == [t.timestamp for t in golden_turns]

    @pytest.mark.asyncio
    async def test_gold_facts_reach_curation_in_the_native_shape(self, tmp_path, golden_turns):
        # Assert: `ext-01-uses` asserts user USES rust; it must arrive readable by the
        # engine's own accessors, with provenance rather than gold's entity type.
        _report, recorder, _ = await replay_golden_log(tmp_path / "iso", golden_turns)
        call = next(c for c in recorder.calls if c.event_id == "golden-ext-01-uses")

        rel = next(r for r in call.relationships if r["target"] == "rust")
        assert native_predicate(rel) == "USES"
        assert native_properties(rel)["source_type"] == "extracted"
        assert {e["id"]: e["type"] for e in call.entities}["rust"] == "Technology"

    @pytest.mark.asyncio
    async def test_the_cease_control_arrives_with_its_assertion_kind(self, tmp_path, golden_turns):
        # Assert: negative control B survives the round trip through cache and replay.
        _report, recorder, _ = await replay_golden_log(tmp_path / "iso", golden_turns)
        call = next(c for c in recorder.calls if c.event_id == "golden-ext-45-cease-learning")

        assert native_properties(call.relationships[0])["assertion_kind"] == "cease"

    @pytest.mark.asyncio
    async def test_every_fact_bearing_turn_carries_facts(self, tmp_path, golden_turns):
        # Assert: replay did not quietly deliver 87 empty payloads.
        _report, recorder, _ = await replay_golden_log(tmp_path / "iso", golden_turns)

        with_facts = [c for c in recorder.calls if c.relationships]
        assert len(with_facts) == sum(1 for t in golden_turns if t.relationships)
        assert len(with_facts) > 0


class TestReplayDeterminism:
    """Stream-level determinism. The graph-level gate is the integration test."""

    @pytest.mark.asyncio
    async def test_two_replays_feed_curation_a_byte_identical_stream(self, tmp_path, golden_turns):
        # Act: two independent materializations and rebuilds, not one run compared to itself.
        _report_a, recorder_a, _ = await replay_golden_log(tmp_path / "run-a", golden_turns)
        _report_b, recorder_b, _ = await replay_golden_log(tmp_path / "run-b", golden_turns)

        # Assert: the project's own determinism gate, over real replayed turns.
        assert_rebuild_twice_identical(
            canonical_replay_form(recorder_a.calls), canonical_replay_form(recorder_b.calls)
        )
        assert len(recorder_a.calls) == EXPECTED_TURN_COUNT

    @pytest.mark.asyncio
    async def test_the_determinism_gate_would_catch_a_divergence(self, tmp_path, golden_turns):
        # Assert: the gate is not passing because both sides are empty or equal by
        # construction. Perturb one stream and it must fail.
        _report, recorder, _ = await replay_golden_log(tmp_path / "iso", golden_turns)
        perturbed = list(recorder.calls)
        perturbed[0] = ReplayedTurn(
            event_id=perturbed[0].event_id,
            session_id=perturbed[0].session_id,
            recorded_at="1999-01-01T00:00:00+00:00",
            entities=perturbed[0].entities,
            relationships=perturbed[0].relationships,
        )

        with pytest.raises(RebuildDeterminismError):
            assert_rebuild_twice_identical(
                canonical_replay_form(recorder.calls), canonical_replay_form(perturbed)
            )
