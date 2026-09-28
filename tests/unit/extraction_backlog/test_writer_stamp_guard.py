"""The dispatcher's writer-stamp guard.

Curation stamps the relationship edges reconciliation creates, and
EXTRACTED_FROM edges, with a triple from `KnowledgeConfig`, not from the
epoch ledger, so a backend whose
writer stamps differ from the active epoch's must neither dispatch nor apply:
it stalls with a reason naming both triples.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging

import pytest

from backend.knowledge.curation.graph_writer import RebuildStamps
from backend.knowledge.extraction_cache import OUTCOME_EXTRACTED
from tests.unit.extraction_backlog.conftest import (
    ONTOLOGY_VERSION,
    composed,
    wait_until,
)

DISPATCHER_LOGGER = "backend.extraction_backlog.dispatcher"
CANDIDATE = "svc-model-2"

_OTHER_VALUE = {
    "ontology_version": "9.9.9",
    "extraction_version": "ev-stale",
    "model_hash": composed("stale-model"),
}


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


def _apply_pending_and_inference_pending(world, ts) -> tuple[str, str]:
    """Two turns: the first already cached (apply-pending), the second not."""
    cached = world.log_turn(
        session_id="s1", turn_index=0, timestamp=ts(0), utterance="I really use rust"
    )
    _put_cached(world, cached, entity="rust", created_at=ts(0))
    uncached = world.log_turn(
        session_id="s1", turn_index=1, timestamp=ts(1), utterance="I really use zig"
    )
    return cached, uncached


def _mismatched(world, field: str) -> RebuildStamps:
    return dataclasses.replace(world.active_epoch_stamps(), **{field: _OTHER_VALUE[field]})


def _triple(stamps) -> str:
    return (
        f"(ontology_version={stamps.ontology_version!r}, "
        f"extraction_version={stamps.extraction_version!r}, "
        f"model_hash={stamps.model_hash!r})"
    )


class TestMismatchStalls:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("field", ["ontology_version", "extraction_version", "model_hash"])
    async def test_one_differing_field_stalls_with_no_service_call_and_no_apply(
        self, backlog_world, ts, caplog, field
    ):
        # Arrange
        world = backlog_world
        cached, uncached = _apply_pending_and_inference_pending(world, ts)
        writer = _mismatched(world, field)
        epoch = world.store.active_epoch()
        dispatcher = world.build_dispatcher(writer_stamps=writer)

        # Act
        with caplog.at_level(logging.INFO, logger=DISPATCHER_LOGGER):
            await dispatcher.start()
            await wait_until(lambda: dispatcher.state == "stalled")
            # Several recheck periods (stall_recheck_s=0.02): the guard holds.
            await asyncio.sleep(0.2)

        # Assert
        assert dispatcher.state == "stalled"
        assert world.service.received == []
        assert world.service.info_calls == 0
        assert world.curation.calls == []
        assert world.main_llm.calls == []
        assert world.event_store.get_extraction_applied(epoch.epoch_id) == {}
        assert world.attempts() == []
        assert world.cache.get(uncached, epoch.extraction_version, epoch.model_hash) is None
        status = dispatcher.snapshot()
        # One inference-pending turn in the backlog, one apply-pending turn.
        assert (status.backlog_depth, status.apply_pending) == (1, 1)
        assert status.state == "stalled"
        assert cached not in world.event_store.get_extraction_applied(epoch.epoch_id)
        expected = (
            f"backend writer stamps {_triple(writer)} != "
            f"epoch {epoch.epoch_id} {_triple(epoch)}"
        )
        assert expected in caplog.text
        assert "MIST_MODEL_HASH" in caplog.text
        # Logged at ERROR once per epoch, not once per recheck.
        assert caplog.text.count("refuses to apply") == 1

    @pytest.mark.asyncio
    async def test_drain_returns_false_promptly_while_stalled_by_the_guard(self, backlog_world, ts):
        world = backlog_world
        _apply_pending_and_inference_pending(world, ts)
        dispatcher = world.build_dispatcher(writer_stamps=_mismatched(world, "model_hash"))
        await dispatcher.start()
        await wait_until(lambda: dispatcher.state == "stalled")
        loop = asyncio.get_running_loop()

        started = loop.time()
        drained = await dispatcher.drain(timeout=5.0)

        assert drained is False
        assert loop.time() - started < 1.0

    @pytest.mark.asyncio
    async def test_an_empty_backlog_also_stalls(self, backlog_world):
        """The guard runs before the scan: nothing to do is still not 'idle'."""
        world = backlog_world
        dispatcher = world.build_dispatcher(writer_stamps=_mismatched(world, "model_hash"))

        await dispatcher.start()
        await wait_until(lambda: dispatcher.state == "stalled")

        assert world.service.info_calls == 0


class TestRecovery:
    @pytest.mark.asyncio
    async def test_a_restart_with_matching_stamps_drains_the_stalled_backlog(
        self, backlog_world, ts
    ):
        # Arrange: stalled by a stale model hash.
        world = backlog_world
        cached, uncached = _apply_pending_and_inference_pending(world, ts)
        stale = world.build_dispatcher(writer_stamps=_mismatched(world, "model_hash"))
        await stale.start()
        await wait_until(lambda: stale.state == "stalled")
        await stale.stop(timeout=2.0)

        # Act: the operator recreates the backend with a matching MIST_MODEL_HASH.
        restarted = world.build_dispatcher()
        await restarted.start()
        drained = await restarted.drain(timeout=5.0)

        # Assert
        epoch = world.store.active_epoch()
        assert drained is True
        assert world.curation.event_ids == [cached, uncached]
        assert [r.event_id for r in world.service.received] == [uncached]
        assert world.event_store.get_extraction_applied(epoch.epoch_id) == {
            cached: "applied",
            uncached: "applied",
        }
        assert restarted.state == "idle"


class TestMatchingStamps:
    @pytest.mark.asyncio
    async def test_explicit_matching_stamps_drain_without_stalling(self, backlog_world, ts):
        world = backlog_world
        cached, uncached = _apply_pending_and_inference_pending(world, ts)
        epoch = world.store.active_epoch()
        matching = RebuildStamps(
            ontology_version=epoch.ontology_version,
            extraction_version=epoch.extraction_version,
            model_hash=epoch.model_hash,
        )
        dispatcher = world.build_dispatcher(writer_stamps=matching)
        states: list[str] = []
        dispatcher.add_state_listener(lambda _previous, current: states.append(current))

        await dispatcher.start()

        assert await dispatcher.drain(timeout=5.0)
        assert world.curation.event_ids == [cached, uncached]
        assert "stalled" not in states


class TestCutover:
    @pytest.mark.asyncio
    async def test_the_fill_is_unaffected_by_writer_stamps_that_differ_from_the_active_epoch(
        self, backlog_world, ts
    ):
        # Arrange: two logged turns, a cutover open, and a backend whose writer
        # stamps are already the CANDIDATE's (so they differ from the active epoch).
        world = backlog_world
        ids = [
            world.log_turn(
                session_id="s1", turn_index=i, timestamp=ts(i), utterance=f"I really use tool{i}"
            )
            for i in range(2)
        ]
        cutover = _begin(world)
        world.service.model_hash = CANDIDATE
        candidate_stamps = RebuildStamps(
            ontology_version=cutover.ontology_version,
            extraction_version=cutover.extraction_version,
            model_hash=cutover.model_hash,
        )
        dispatcher = world.build_dispatcher(writer_stamps=candidate_stamps)
        states: list[str] = []
        dispatcher.add_state_listener(lambda _previous, current: states.append(current))

        # Act
        await dispatcher.start()
        await wait_until(lambda: world.store.open_cutover().state == "ready")

        # Assert: the fill ran exactly as without the guard.
        assert [r.event_id for r in world.service.received] == ids
        assert world.curation.calls == []
        for event_id in ids:
            row = world.cache.get(event_id, cutover.extraction_version, cutover.model_hash)
            assert row is not None and row["outcome"] == OUTCOME_EXTRACTED
        assert "stalled" not in states
        assert dispatcher.state == "idle"

    @pytest.mark.asyncio
    async def test_after_promotion_a_dispatcher_with_the_old_stamps_stalls_until_restarted(
        self, backlog_world, ts
    ):
        # Arrange: one turn applied under epoch 1, a cutover filled and checked.
        world = backlog_world
        first = world.log_turn(
            session_id="s1", turn_index=0, timestamp=ts(0), utterance="I really use tool0"
        )
        dispatcher = world.build_dispatcher()  # writer stamps = epoch 1
        await dispatcher.start()
        assert await dispatcher.drain(timeout=5.0)
        cutover = _begin(world)
        world.service.model_hash = CANDIDATE
        dispatcher.wake()
        await wait_until(lambda: world.store.open_cutover().state == "ready")
        store = world.store
        assert store.transition_cutover(
            store.open_cutover(),
            from_states=("ready",),
            to_state="checked",
            updated_at="2026-09-02T00:00:00+00:00",
            rebuild_job_id="rebuild-job",
            rebuilt_through_event_id=first,
            check_report={"passed": True},
            write_check=True,
        )

        # Act: promote under the running dispatcher, then log a new turn.
        new_epoch = store.promote_cutover(
            store.get_cutover(cutover.cutover_id), activated_at="2026-09-02T00:00:00+00:00"
        )
        received_before = len(world.service.received)
        late = world.log_turn(
            session_id="s1", turn_index=1, timestamp=ts(5), utterance="I really use tool5"
        )
        dispatcher.wake()
        await wait_until(lambda: dispatcher.state == "stalled")
        await asyncio.sleep(0.1)

        # Assert: nothing applied under the new epoch with the old stamps.
        assert world.curation.event_ids == [first]
        assert len(world.service.received) == received_before
        assert late not in world.event_store.get_extraction_applied(int(new_epoch["epoch_id"]))

        # Act: restart with the new epoch's stamps.
        await dispatcher.stop(timeout=2.0)
        restarted = world.build_dispatcher()
        await restarted.start()

        # Assert
        assert await restarted.drain(timeout=5.0)
        assert world.curation.event_ids == [first, late]


def _begin(world):
    """Open a cutover whose candidate differs from the active epoch in model hash only."""
    store = world.store
    epoch = store.active_epoch()
    cutover = store.begin_cutover(
        ontology_version=epoch.ontology_version,
        extraction_version=epoch.extraction_version,
        model_hash=composed(CANDIDATE),
        bare_model_hash=CANDIDATE,
        source_epoch_id=epoch.epoch_id,
        requested_at="2026-09-01T12:00:00+00:00",
    )
    assert cutover is not None
    return cutover
