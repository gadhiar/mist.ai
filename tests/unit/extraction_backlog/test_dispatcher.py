"""ExtractionDispatcher behaviour against a fake extraction service (T2a acceptance)."""

from __future__ import annotations

import asyncio
import io
import logging

import pytest

from backend.extraction_backlog import admin
from backend.extraction_contract.models import ErrorCode
from backend.knowledge.extraction_cache import (
    OUTCOME_EXTRACTED,
    OUTCOME_SKIPPED,
    SKIP_EXTRACTION_FAILED,
    SKIP_TOO_SHORT,
)
from tests.unit.extraction_backlog.conftest import ONTOLOGY_VERSION, composed, wait_until
from tests.unit.extraction_backlog.fakes import SimulatedCrashError


def _put_cached(world, event_id: str, *, entity: str, created_at: str) -> None:
    epoch = world.store.active_epoch()
    world.cache.put(
        event_id,
        ONTOLOGY_VERSION,
        epoch.extraction_version,
        epoch.model_hash,
        outcome=OUTCOME_EXTRACTED,
        entities=[{"id": entity, "type": "Technology", "name": entity}],
        relationships=[],
        created_at=created_at,
    )


class TestServiceDown:
    @pytest.mark.asyncio
    async def test_five_logged_turns_stay_in_the_backlog_while_the_service_is_down(
        self, backlog_world, ts
    ):
        # Arrange
        world = backlog_world
        world.service.down = True
        for i in range(5):
            world.log_turn(
                session_id="s1", turn_index=i, timestamp=ts(i), utterance=f"I really use tool{i}"
            )
        dispatcher = world.build_dispatcher()

        # Act
        await dispatcher.start()
        await wait_until(lambda: dispatcher.state == "unreachable")
        status = dispatcher.snapshot()

        # Assert
        assert status.state == "unreachable"
        assert status.backlog_depth == 5
        assert status.apply_pending == 0
        assert status.service.reachable is False
        assert world.main_llm.calls == []
        assert world.curation.calls == []

    @pytest.mark.asyncio
    async def test_backlog_drains_once_the_service_comes_back(self, backlog_world, ts):
        # Arrange
        world = backlog_world
        world.service.down = True
        for i in range(3):
            world.log_turn(
                session_id="s1", turn_index=i, timestamp=ts(i), utterance=f"I really use tool{i}"
            )
        dispatcher = world.build_dispatcher()
        await dispatcher.start()
        await wait_until(lambda: dispatcher.state == "unreachable")

        # Act
        world.service.down = False
        dispatcher.wake()
        await wait_until(lambda: dispatcher.snapshot().backlog_depth == 0)
        drained = await dispatcher.drain(timeout=5.0)

        # Assert
        assert drained is True
        assert len(world.curation.calls) == 3
        assert world.main_llm.calls == []
        assert dispatcher.snapshot().state == "idle"

    @pytest.mark.asyncio
    async def test_model_loading_defers_without_counting_an_attempt(self, backlog_world, ts):
        # Arrange
        world = backlog_world
        world.service.fail_script["I really use rust"] = [(ErrorCode.MODEL_LOADING, True)] * 3
        world.log_turn(
            session_id="s1", turn_index=0, timestamp=ts(0), utterance="I really use rust"
        )
        dispatcher = world.build_dispatcher()

        # Act
        await dispatcher.start()
        drained = await dispatcher.drain(timeout=5.0)
        if not drained:
            await wait_until(lambda: dispatcher.snapshot().backlog_depth == 0)

        # Assert
        rows = world.attempts()
        assert [r["outcome"] for r in rows] == ["deferred", "deferred", "deferred", "extracted"]
        assert all(r["counted"] == 0 for r in rows)
        assert [r["error_code"] for r in rows[:3]] == ["model_loading"] * 3


class TestLogOrder:
    @pytest.mark.asyncio
    async def test_turns_inserted_out_of_order_drain_in_replay_order(self, backlog_world, ts):
        # Arrange: insertion order differs from (timestamp, session_id, turn_index).
        world = backlog_world
        inserted = [
            ("b", 1, ts(5), "I really like zig"),
            ("a", 0, ts(1), "I really like rust"),
            ("b", 0, ts(2), "I really like go"),
            ("a", 2, ts(6), "I really like ocaml"),
            ("a", 1, ts(4), "I really like elixir"),  # same stamp as b/2: session 'a' first
            ("b", 2, ts(4), "I really like haskell"),
        ]
        ids = {}
        for session, index, stamp, utterance in inserted:
            ids[utterance] = world.log_turn(
                session_id=session, turn_index=index, timestamp=stamp, utterance=utterance
            )
        expected = [
            "I really like rust",
            "I really like go",
            "I really like elixir",
            "I really like haskell",
            "I really like zig",
            "I really like ocaml",
        ]
        dispatcher = world.build_dispatcher()

        # Act
        await dispatcher.start()
        assert await dispatcher.drain(timeout=5.0)

        # Assert
        assert world.service.received_utterances == expected
        assert world.curation.event_ids == [ids[u] for u in expected]
        assert [c.recorded_at for c in world.curation.calls] == [
            ts(1),
            ts(2),
            ts(4),
            ts(4),
            ts(5),
            ts(6),
        ]

    @pytest.mark.asyncio
    async def test_gate_one_is_retired_on_the_dispatcher_path(self, backlog_world, ts):
        """`rate_limit_max_per_minute=0` would refuse every in-process extraction."""
        world = backlog_world
        for i in range(3):
            world.log_turn(
                session_id="s1", turn_index=i, timestamp=ts(i), utterance=f"I really use tool{i}"
            )
        dispatcher = world.build_dispatcher()

        await dispatcher.start()
        assert await dispatcher.drain(timeout=5.0)

        assert len(world.service.received) == 3
        epoch = world.store.active_epoch()
        reasons = world.cache.event_ids_for(epoch.extraction_version, epoch.model_hash)
        assert set(reasons.values()) == {None}

    @pytest.mark.asyncio
    async def test_a_too_short_turn_is_recorded_as_a_skip_and_never_dispatched(
        self, backlog_world, ts
    ):
        world = backlog_world
        short = world.log_turn(session_id="s1", turn_index=0, timestamp=ts(0), utterance="ok sure")
        world.log_turn(session_id="s1", turn_index=1, timestamp=ts(1), utterance="I use rust daily")
        dispatcher = world.build_dispatcher()

        await dispatcher.start()
        assert await dispatcher.drain(timeout=5.0)

        epoch = world.store.active_epoch()
        row = world.cache.get(short, epoch.extraction_version, epoch.model_hash)
        assert row["outcome"] == OUTCOME_SKIPPED
        assert row["skip_reason"] == SKIP_TOO_SHORT
        assert world.service.received_utterances == ["I use rust daily"]

    @pytest.mark.asyncio
    async def test_request_carries_bare_model_hash_and_log_history(self, backlog_world, ts):
        world = backlog_world
        world.log_turn(
            session_id="s1",
            turn_index=0,
            timestamp=ts(0),
            utterance="I use rust daily",
            response="R0",
        )
        world.log_turn(
            session_id="s1",
            turn_index=1,
            timestamp=ts(1),
            utterance="I also use zig",
            response="R1",
        )
        dispatcher = world.build_dispatcher()

        await dispatcher.start()
        assert await dispatcher.drain(timeout=5.0)

        second = world.service.received[1]
        assert second.expect.model_hash == world.service.model_hash
        assert second.expect.extraction_version == "ev-test"
        assert second.turn_id == "s1:1"
        assert [(m.role, m.content) for m in second.conversation_history] == [
            ("user", "I use rust daily"),
            ("assistant", "R0"),
            ("user", "I also use zig"),
            ("assistant", "R1"),
        ]
        assert second.request_id != world.service.received[0].request_id
        assert second.job_id != world.service.received[0].job_id


class TestCrashRecovery:
    @pytest.mark.asyncio
    async def test_crash_after_graph_write_reapplies_from_cache_without_the_service(
        self, backlog_world, ts
    ):
        # Arrange: reference run, uninterrupted, in a separate world.
        world = backlog_world
        first = world.log_turn(
            session_id="s1", turn_index=0, timestamp=ts(0), utterance="I really use rust"
        )
        world.log_turn(session_id="s1", turn_index=1, timestamp=ts(1), utterance="I really use zig")
        world.curation.crash_after_write_for = first
        dispatcher = world.build_dispatcher()

        # Act 1: the fault fires after the graph write, before any marker.
        await dispatcher.start()
        await wait_until(lambda: not dispatcher.running)
        received_before_restart = len(world.service.received)
        epoch = world.store.active_epoch()
        assert world.event_store.get_extraction_applied(epoch.epoch_id) == {}
        assert world.cache.get(first, epoch.extraction_version, epoch.model_hash) is not None

        # Act 2: a fresh dispatcher on the same stores.
        restarted = world.build_dispatcher()
        await restarted.start()
        assert await restarted.drain(timeout=5.0)

        # Assert: turn 1 re-applied from the cache; the service saw it once.
        assert received_before_restart == 1
        assert world.service.received_utterances == ["I really use rust", "I really use zig"]
        assert world.curation.event_ids.count(first) == 2
        assert world.curation.graph_state() == (
            {
                "rust": {"type": "Technology", "name": "Rust"},
                "zig": {"type": "Technology", "name": "Zig"},
            },
            {},
        )

    @pytest.mark.asyncio
    async def test_crash_after_curation_marker_skips_curation_on_restart(
        self, backlog_world_with_deriver, ts
    ):
        # Arrange: a turn with a Stage 9 signal, whose operations crash on apply.
        world = backlog_world_with_deriver
        utterance = "I prefer you always answer briefly"
        event_id = world.log_turn(
            session_id="s1", turn_index=0, timestamp=ts(0), utterance=utterance, response="Sure."
        )
        world.service.derivation_ops[utterance] = [
            {"op": "CREATE_PREFERENCE", "id": "pref-brief", "display_name": "Brief answers"}
        ]
        world.deriver.crash_after_apply_for = event_id
        dispatcher = world.build_dispatcher()

        # Act
        await dispatcher.start()
        await wait_until(lambda: not dispatcher.running)
        epoch = world.store.active_epoch()
        stage_after_crash = world.event_store.get_extraction_applied(epoch.epoch_id)
        restarted = world.build_dispatcher()
        await restarted.start()
        assert await restarted.drain(timeout=5.0)

        # Assert
        assert stage_after_crash == {event_id: "curated"}
        assert world.curation.event_ids == [event_id]
        assert [call[0] for call in world.deriver.apply_calls] == [event_id, event_id]
        assert world.deriver.self_model == {
            "pref-brief": {
                "op": "CREATE_PREFERENCE",
                "id": "pref-brief",
                "display_name": "Brief answers",
            }
        }
        assert len(world.service.received) == 1
        assert world.event_store.get_extraction_applied(epoch.epoch_id) == {event_id: "applied"}

    @pytest.mark.asyncio
    async def test_derivation_input_carries_the_logged_reply_and_graph_context(
        self, backlog_world_with_deriver, ts
    ):
        world = backlog_world_with_deriver
        world.log_turn(
            session_id="s1",
            turn_index=0,
            timestamp=ts(0),
            utterance="I prefer you always answer briefly",
            response="Understood, I will keep it short.",
        )
        world.log_turn(
            session_id="s1", turn_index=1, timestamp=ts(1), utterance="I use rust at work"
        )
        dispatcher = world.build_dispatcher()

        await dispatcher.start()
        assert await dispatcher.drain(timeout=5.0)

        with_signal, without_signal = world.service.received
        assert with_signal.derivation is not None
        assert with_signal.derivation.assistant_response == "Understood, I will keep it short."
        assert with_signal.derivation.existing_internal_entities == world.deriver.existing
        # "I prefer you" trips both the feedback and the preference patterns.
        assert with_signal.derivation.signal_types == ["feedback", "preference"]
        assert without_signal.derivation is None


class TestDeadLetter:
    @pytest.mark.asyncio
    async def test_five_upstream_failures_dead_letter_and_the_next_turn_proceeds(
        self, backlog_world, ts
    ):
        # Arrange
        world = backlog_world
        bad = world.log_turn(
            session_id="s1", turn_index=0, timestamp=ts(0), utterance="I really use cobol"
        )
        good = world.log_turn(
            session_id="s1", turn_index=1, timestamp=ts(1), utterance="I really use rust"
        )
        world.service.fail_script["I really use cobol"] = [(ErrorCode.UPSTREAM_LLM, True)] * 5
        dispatcher = world.build_dispatcher()

        # Act
        await dispatcher.start()
        assert await dispatcher.drain(timeout=5.0)
        status = dispatcher.snapshot()

        # Assert
        epoch = world.store.active_epoch()
        row = world.cache.get(bad, epoch.extraction_version, epoch.model_hash)
        assert row["outcome"] == OUTCOME_SKIPPED
        assert row["skip_reason"] == SKIP_EXTRACTION_FAILED
        assert status.dead_lettered == 1
        assert status.backlog_depth == 0
        assert world.curation.event_ids == [good]
        rows = [r for r in world.attempts() if r["event_id"] == bad]
        assert [r["attempt"] for r in rows] == [1, 2, 3, 4, 5]
        assert [r["outcome"] for r in rows] == ["failed"] * 4 + ["dead_lettered"]
        assert all(r["counted"] == 1 and r["error_code"] == "upstream_llm" for r in rows)
        assert status.last_job is not None

    @pytest.mark.asyncio
    async def test_retry_dead_letters_puts_the_turn_back_and_it_is_applied(self, backlog_world, ts):
        # Arrange: dead-letter one turn, then heal the service.
        world = backlog_world
        bad = world.log_turn(
            session_id="s1", turn_index=0, timestamp=ts(0), utterance="I really use cobol"
        )
        good = world.log_turn(
            session_id="s1", turn_index=1, timestamp=ts(1), utterance="I really use rust"
        )
        world.service.fail_script["I really use cobol"] = [(ErrorCode.TIMEOUT, True)] * 5
        dispatcher = world.build_dispatcher()
        await dispatcher.start()
        assert await dispatcher.drain(timeout=5.0)
        assert dispatcher.snapshot().dead_lettered == 1
        out = io.StringIO()

        # Act
        code = admin.main(["retry-dead-letters"], store=world.store, out=out)
        dispatcher.wake()
        assert await dispatcher.drain(timeout=5.0)

        # Assert
        assert code == 0
        assert f"retried event_id={bad}" in out.getvalue()
        assert "OUT OF LOG ORDER" in out.getvalue()
        status = dispatcher.snapshot()
        assert status.dead_lettered == 0
        assert status.backlog_depth == 0
        assert world.curation.event_ids == [good, bad]  # applied after its successor
        epoch = world.store.active_epoch()
        assert world.store.counted_failures(bad, epoch) == 0
        assert world.store.next_attempt_number(bad, epoch) == 7  # history kept, count reset

    @pytest.mark.asyncio
    async def test_a_non_retryable_upstream_failure_dead_letters_at_once(self, backlog_world, ts):
        world = backlog_world
        world.log_turn(
            session_id="s1", turn_index=0, timestamp=ts(0), utterance="I really use cobol"
        )
        world.service.fail_script["I really use cobol"] = [(ErrorCode.UPSTREAM_LLM, False)]
        dispatcher = world.build_dispatcher()

        await dispatcher.start()
        assert await dispatcher.drain(timeout=5.0)

        assert dispatcher.snapshot().dead_lettered == 1
        assert [r["outcome"] for r in world.attempts()] == ["dead_lettered"]


class TestEpochAndContract:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "field, value",
        [
            pytest.param("extraction_version", "ev-other", id="extraction-version"),
            pytest.param("model_hash", "other-model", id="model-hash"),
        ],
    )
    async def test_info_epoch_mismatch_stalls_without_dispatching(
        self, backlog_world, ts, field, value
    ):
        # Arrange
        world = backlog_world
        setattr(world.service, field, value)
        world.log_turn(session_id="s1", turn_index=0, timestamp=ts(0), utterance="I use rust daily")
        dispatcher = world.build_dispatcher()

        # Act
        await dispatcher.start()
        await wait_until(lambda: dispatcher.state == "epoch_mismatch")
        drained = await dispatcher.drain(timeout=0.5)

        # Assert
        assert drained is False
        assert world.service.received == []
        assert world.attempts() == []
        assert dispatcher.snapshot().backlog_depth == 1

    @pytest.mark.asyncio
    async def test_epoch_comparison_uses_the_composed_model_hash(self, backlog_world, ts):
        """The epoch holds `compose(bare)`; the service reports `bare`. They match."""
        world = backlog_world
        epoch = world.store.active_epoch()
        assert epoch.model_hash == composed(world.service.model_hash)
        assert epoch.model_hash != world.service.model_hash

        world.log_turn(session_id="s1", turn_index=0, timestamp=ts(0), utterance="I use rust daily")
        dispatcher = world.build_dispatcher()
        await dispatcher.start()

        assert await dispatcher.drain(timeout=5.0)

    @pytest.mark.asyncio
    async def test_incompatible_contract_version_stalls(self, backlog_world, ts):
        world = backlog_world
        world.service.contract_version = "2.0.0"
        world.log_turn(session_id="s1", turn_index=0, timestamp=ts(0), utterance="I use rust daily")
        dispatcher = world.build_dispatcher()

        await dispatcher.start()
        await wait_until(lambda: dispatcher.state == "stalled")

        assert world.service.received == []
        assert world.attempts() == []

    @pytest.mark.asyncio
    async def test_epoch_mismatch_recovers_when_the_service_is_corrected(self, backlog_world, ts):
        world = backlog_world
        world.service.extraction_version = "ev-other"
        world.log_turn(session_id="s1", turn_index=0, timestamp=ts(0), utterance="I use rust daily")
        dispatcher = world.build_dispatcher()
        await dispatcher.start()
        await wait_until(lambda: dispatcher.state == "epoch_mismatch")

        world.service.extraction_version = "ev-test"
        await wait_until(lambda: dispatcher.snapshot().backlog_depth == 0)

        assert len(world.service.received) == 1


class TestFirstActivation:
    @pytest.mark.asyncio
    async def test_cached_turns_are_marked_applied_and_uncached_ones_are_legacy(
        self, unactivated_backlog_world, ts, caplog
    ):
        # Arrange: three turns logged before the backlog ever ran.
        world = unactivated_backlog_world
        t1 = world.log_turn(
            session_id="s1", turn_index=0, timestamp=ts(0), utterance="I use rust daily"
        )
        t2 = world.log_turn(
            session_id="s1", turn_index=1, timestamp=ts(1), utterance="I use zig daily"
        )
        t3 = world.log_turn(
            session_id="s1", turn_index=2, timestamp=ts(2), utterance="I use go daily"
        )
        _put_cached(world, t1, entity="rust", created_at=ts(0))
        _put_cached(world, t3, entity="go", created_at=ts(2))
        dispatcher = world.build_dispatcher()

        # Act
        with caplog.at_level(logging.INFO, logger="backend.extraction_backlog.dispatcher"):
            await dispatcher.start()
            assert await dispatcher.drain(timeout=5.0)

        # Assert
        epoch = world.store.active_epoch()
        assert world.event_store.get_extraction_applied(epoch.epoch_id) == {
            t1: "applied",
            t3: "applied",
        }
        assert world.event_store.get_extraction_legacy_turns(epoch.epoch_id) == {t2}
        assert dispatcher.legacy_unextracted == 1
        assert world.service.received == []
        assert world.curation.calls == []
        status = dispatcher.snapshot()
        assert (status.backlog_depth, status.apply_pending) == (0, 0)
        assert status.legacy_unextracted == 1
        assert "1 legacy-unextracted" in caplog.text

    @pytest.mark.asyncio
    async def test_the_floor_is_stable_across_a_restart(self, unactivated_backlog_world, ts):
        # Arrange: activate once over one cached and one uncached turn.
        world = unactivated_backlog_world
        t1 = world.log_turn(
            session_id="s1", turn_index=0, timestamp=ts(0), utterance="I use rust daily"
        )
        t2 = world.log_turn(
            session_id="s1", turn_index=1, timestamp=ts(1), utterance="I use zig daily"
        )
        _put_cached(world, t1, entity="rust", created_at=ts(0))
        first = world.build_dispatcher()
        await first.start()
        assert await first.drain(timeout=5.0)
        await first.stop()
        epoch = world.store.active_epoch()
        floor_before = world.event_store.get_extraction_activation(epoch.epoch_id)

        # Act: a turn logged after activation, then a restart.
        t3 = world.log_turn(
            session_id="s1", turn_index=2, timestamp=ts(2), utterance="I use go daily"
        )
        second = world.build_dispatcher()
        await second.start()
        assert await second.drain(timeout=5.0)

        # Assert
        assert world.event_store.get_extraction_activation(epoch.epoch_id) == floor_before
        assert world.event_store.get_extraction_legacy_turns(epoch.epoch_id) == {t2}
        assert world.service.received_utterances == ["I use go daily"]
        assert world.curation.event_ids == [t3]
        assert second.legacy_unextracted == 1


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_stop_abandons_an_in_flight_job_and_restart_redispatches_it(
        self, backlog_world, ts
    ):
        # Arrange: the fake service holds the job.
        world = backlog_world
        world.service.hold = asyncio.Event()
        event_id = world.log_turn(
            session_id="s1", turn_index=0, timestamp=ts(0), utterance="I use rust daily"
        )
        dispatcher = world.build_dispatcher()
        await dispatcher.start()
        await wait_until(lambda: len(world.service.received) == 1)

        # Act
        await dispatcher.stop(timeout=2.0)
        epoch = world.store.active_epoch()
        cached_after_stop = world.cache.get(event_id, epoch.extraction_version, epoch.model_hash)
        world.service.hold.set()
        restarted = world.build_dispatcher()
        await restarted.start()
        assert await restarted.drain(timeout=5.0)

        # Assert
        assert cached_after_stop is None
        assert len(world.service.received) == 2
        assert world.service.received[0].job_id != world.service.received[1].job_id
        assert world.curation.event_ids == [event_id]

    @pytest.mark.asyncio
    async def test_off_mode_dispatches_nothing_and_reports_disabled(self, backlog_world, ts):
        world = backlog_world
        world.log_turn(session_id="s1", turn_index=0, timestamp=ts(0), utterance="I use rust daily")
        dispatcher = world.build_dispatcher(mode="off")

        await dispatcher.start()

        assert dispatcher.running is False
        assert dispatcher.snapshot().state == "disabled"
        assert await dispatcher.drain(timeout=0.1) is False
        assert world.service.info_calls == 0
        epoch = world.store.active_epoch()
        assert world.event_store.get_extraction_activation(epoch.epoch_id) is not None

    @pytest.mark.asyncio
    async def test_apply_listener_receives_the_curation_result(self, backlog_world, ts):
        world = backlog_world
        world.log_turn(session_id="s1", turn_index=0, timestamp=ts(0), utterance="I use rust daily")
        dispatcher = world.build_dispatcher()
        reports = []

        async def listener(report):
            reports.append(report)

        dispatcher.add_apply_listener(listener)

        await dispatcher.start()
        assert await dispatcher.drain(timeout=5.0)

        assert len(reports) == 1
        # The fake service names the entity after the utterance's last word.
        assert reports[0].curation_result.validated_entities[0]["id"] == "daily"

    @pytest.mark.asyncio
    async def test_a_simulated_crash_kills_the_loop_and_reports_stalled(self, backlog_world, ts):
        world = backlog_world
        event_id = world.log_turn(
            session_id="s1", turn_index=0, timestamp=ts(0), utterance="I use rust daily"
        )
        world.curation.crash_after_write_for = event_id
        dispatcher = world.build_dispatcher()

        await dispatcher.start()
        await wait_until(lambda: not dispatcher.running)

        assert dispatcher.state == "stalled"
        assert isinstance(dispatcher._task.exception(), SimulatedCrashError)
