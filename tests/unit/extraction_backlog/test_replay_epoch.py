"""The replay epoch: a change of the ontology pack's rails alone, cut over with no refill.

A candidate that keeps the active epoch's triple and `pack_prompt_hash` and
changes `pack_rails_hash` is not the same stamps as its source
(`has_same_stamps`), so it is not a settings change. What feeds the model is
unchanged, so the rows the active epoch cached are the candidate's too: the
candidate binds in the namespace its source's rows are in, the fill finds every
turn covered and sends no job, and the staging rebuild replays those rows under
the new rails.

The dispatcher runs against the fake extraction service (`fakes.py`). The event
store is a file under `tmp_path`, and the pack hashes are invented values.
"""

from __future__ import annotations

import dataclasses

import pytest
import pytest_asyncio

from backend.extraction_backlog.cutover import abandon_cutover
from backend.knowledge.extraction_cache import OUTCOME_EXTRACTED, keyed_namespace
from tests.unit.extraction_backlog.conftest import (
    ONTOLOGY_VERSION,
    _build_world,
    wait_until,
)
from tests.unit.extraction_backlog.fakes import fake_pack_resolver, serving_identity_of
from tests.unit.extraction_backlog.test_identity_cutover import (
    _check_async,
    _cli,
    _log,
    _RealRebuild,
)

PROMPT = "a" * 64
RAILS = "b" * 64
NEXT_RAILS = "d" * 64
# The pack this code would hold once its rails changed: the active prompt hash,
# another rails hash.
REPLAY_PACK = fake_pack_resolver({ONTOLOGY_VERSION: (PROMPT, NEXT_RAILS)})


def _hashes(carrier) -> tuple[str | None, str | None]:
    return (carrier.pack_prompt_hash, carrier.pack_rails_hash)


def _row_hashes(row: dict) -> tuple[str | None, str | None]:
    return (row["pack_prompt_hash"], row["pack_rails_hash"])


def _cache_rows(world) -> list[tuple]:
    """Every cache row's event, key and namespace, in the order they were written."""
    rows = world.cache._get_connection().execute(
        "SELECT event_id, cache_key, identity_namespace FROM extraction_cache ORDER BY rowid"
    )
    return [tuple(row) for row in rows]


def _begin_replay(world) -> tuple[int, str]:
    """`cutover begin` with the active triple, no flag, from code that holds `REPLAY_PACK`."""
    return _cli(
        world,
        "cutover",
        "begin",
        "--model-hash",
        world.service.model_hash,
        "--extraction-version",
        world.service.extraction_version,
        "--ontology-version",
        ONTOLOGY_VERSION,
        pack_for_version=REPLAY_PACK,
    )


@pytest_asyncio.fixture
async def replay_world(tmp_path):
    """A world whose active epoch stores (PROMPT, RAILS), which the fake service reports."""
    world = _build_world(
        with_deriver=False,
        db_path=str(tmp_path / "events.db"),
        pack_prompt_hash=PROMPT,
        pack_rails_hash=RAILS,
    )
    yield world
    for dispatcher in world.dispatchers:
        await dispatcher.stop(timeout=2.0)
    for client in world.clients:
        await client.aclose()


class TestReplayEpochEndToEnd:
    @pytest.mark.asyncio
    async def test_a_rails_only_change_binds_3_part_covers_every_row_and_promotes_with_no_job(
        self, replay_world, ts
    ):
        world = replay_world
        # Arrange: three turns extracted and applied under epoch 1, which stores
        # (PROMPT, RAILS) and which the dispatcher binds at first contact
        # (3-part, a claim on the pair).
        ids = [_log(world, ts, index) for index in range(3)]
        dispatcher = world.build_dispatcher()
        await dispatcher.start()
        assert await dispatcher.drain(timeout=5.0)
        await dispatcher.stop(timeout=2.0)
        epoch_1 = world.store.active_epoch()
        assert epoch_1.bound and epoch_1.identity_namespace is None
        assert _hashes(epoch_1) == (PROMPT, RAILS)
        identity = world.service.serving_identity
        assert len(world.service.received) == 3
        rows_before = _cache_rows(world)
        assert [row[0] for row in rows_before] == ids

        # 1. Begin with (PROMPT, NEXT_RAILS) and no flag: the candidate opens
        # with the active triple, stores its own two hashes, and already counts
        # the pair's cached rows.
        code, text = _begin_replay(world)
        assert code == 0, text
        candidate = world.store.open_cutover()
        assert (
            candidate.ontology_version,
            candidate.extraction_version,
            candidate.model_hash,
        ) == (epoch_1.ontology_version, epoch_1.extraction_version, epoch_1.model_hash)
        assert _hashes(candidate) == (PROMPT, NEXT_RAILS)
        (cutover_row,) = world.event_store.list_epoch_cutovers()
        assert _row_hashes(cutover_row) == (PROMPT, NEXT_RAILS)
        assert not candidate.bound
        assert "covered=3 total=3" in text and "binding: unbound" in text

        # 2. While the service still reports the active epoch's rails hash, the
        # fill refuses it, naming the hash, and binds nothing.
        dispatcher = world.build_dispatcher()
        await dispatcher.start()
        await wait_until(lambda: dispatcher.state == "epoch_mismatch")
        await dispatcher.stop(timeout=2.0)
        assert f"pack_rails_hash={RAILS!r}" in dispatcher.reason
        assert f"pack_rails_hash={NEXT_RAILS!r}" in dispatcher.reason
        assert not world.store.open_cutover().bound
        assert len(world.service.received) == 3

        # 3. The service reports (PROMPT, NEXT_RAILS) and its serving identity
        # is unchanged: the candidate binds in the 3-part namespace (rule 3, its
        # source's claim), every cached row of the pair counts as covered, no
        # job is sent, and the cutover reaches `ready`.
        world.service.pack_rails_hash = NEXT_RAILS
        assert world.service.serving_identity == identity
        dispatcher = world.build_dispatcher()
        await dispatcher.start()
        await wait_until(lambda: world.store.open_cutover().state == "ready")
        await dispatcher.stop(timeout=2.0)
        ready = world.store.open_cutover()
        assert (ready.identity_namespace, ready.serving_identity) == (None, identity)
        assert world.event_store.get_cutover_identity(ready.cutover_id)["evidence"]["rule"] == "3"
        fill = world.store.fill_scan(ready)
        assert (fill.head, fill.covered, fill.total, fill.bound) == (None, 3, 3, True)
        assert len(world.service.received) == 3
        assert _cache_rows(world) == rows_before

        # 4. The check rebuilds from the rows epoch 1 cached.
        rebuild = _RealRebuild(world)
        result = await _check_async(world, rebuild, expect_turns=3)
        assert result.exit_code == 0, result.report
        assert world.store.open_cutover().state == "checked"
        assert rebuild.curation.calls[:3] == [
            (event_id, [f"tool{index}"]) for index, event_id in enumerate(ids)
        ]

        # 5. Promotion: the new ledger row stores (PROMPT, NEXT_RAILS).
        code, text = _cli(world, "cutover", "promote", "--graph-swapped")
        assert code == 0, text
        epoch_2 = world.store.active_epoch()
        assert epoch_2.epoch_id != epoch_1.epoch_id
        assert _row_hashes(world.event_store.get_current_epoch()) == (PROMPT, NEXT_RAILS)
        assert _hashes(epoch_2) == (PROMPT, NEXT_RAILS)
        assert (epoch_2.identity_source, epoch_2.identity_namespace) == ("promoted", None)

        # 6. The writer guard refuses a backend whose pack still holds RAILS,
        # and passes for one that holds NEXT_RAILS, which then extracts and
        # applies the turn logged since.
        late = _log(world, ts, 3)
        stale_stamps = dataclasses.replace(world.active_epoch_stamps(), pack_rails_hash=RAILS)
        stale = world.build_dispatcher(writer_stamps=stale_stamps)
        await stale.start()
        await wait_until(lambda: stale.state == "stalled")
        await stale.stop(timeout=2.0)
        assert f"pack_rails_hash={RAILS!r}" in stale.reason
        assert f"pack_rails_hash={NEXT_RAILS!r}" in stale.reason
        assert len(world.service.received) == 3

        assert _hashes(world.active_epoch_stamps()) == (PROMPT, NEXT_RAILS)
        current = world.build_dispatcher()
        assert current._writer_stamp_mismatch(epoch_2) is None
        await current.start()
        assert await current.drain(timeout=5.0)
        await current.stop(timeout=2.0)
        assert [request.event_id for request in world.service.received[3:]] == [late]
        assert world.curation.event_ids[-1] == late


class TestAReplayCandidateOfAKeyedEpoch:
    @pytest.mark.asyncio
    async def test_it_binds_in_the_epochs_keyed_namespace_and_covers_its_rows_with_no_job(
        self, replay_world, ts
    ):
        world = replay_world
        # Arrange: epoch 1 is bound keyed to the live identities and holds its
        # rows under that key. The pair's 3-part namespace is claimed by another
        # configuration: an abandoned candidate that was bound there.
        dispatch = world.dispatch_identity()
        keyed = keyed_namespace(world.service.serving_identity, dispatch.identity)
        world.bind_active_epoch(namespace=keyed, source="first-contact")
        epoch = world.store.active_epoch()
        ids = [_log(world, ts, index) for index in range(2)]
        for event_id in ids:
            world.cache.put(
                event_id,
                ONTOLOGY_VERSION,
                epoch.extraction_version,
                epoch.model_hash,
                outcome=OUTCOME_EXTRACTED,
                entities=[{"id": "x", "type": "Technology", "name": "X"}],
                created_at=ts(0),
                identity_namespace=keyed,
            )
        other_components = {**world.service.serving_components, "server.n_cpu_moe": "13"}
        closed_id = world.event_store.begin_epoch_cutover(
            ontology_version=epoch.ontology_version,
            extraction_version=epoch.extraction_version,
            model_hash=epoch.model_hash,
            bare_model_hash=world.service.model_hash,
            source_epoch_id=epoch.epoch_id,
            requested_at=ts(1),
            pack_prompt_hash=PROMPT,
            pack_rails_hash=RAILS,
        )
        assert world.event_store.bind_cutover_identity(
            closed_id,
            serving_identity=serving_identity_of(other_components),
            serving_components=other_components,
            dispatch_identity=dispatch.identity,
            dispatch_components=dict(dispatch.components),
            identity_namespace=None,
            source="first-contact",
            evidence={},
            bound_at=ts(1),
        )
        abandon_cutover(world.store, now_iso=ts(2))
        assert world.event_store.legacy_claims(epoch.extraction_version, epoch.model_hash) == [
            (serving_identity_of(other_components), dispatch.identity)
        ]

        # Act: a replay candidate, and a service on its pack whose serving
        # identity is still the epoch's.
        code, text = _begin_replay(world)
        assert code == 0, text
        world.service.pack_rails_hash = NEXT_RAILS
        dispatcher = world.build_dispatcher()
        await dispatcher.start()
        await wait_until(lambda: world.store.open_cutover().state == "ready")
        await dispatcher.stop(timeout=2.0)

        # Assert
        ready = world.store.open_cutover()
        assert ready.identity_namespace == keyed == epoch.identity_namespace
        assert world.event_store.get_cutover_identity(ready.cutover_id)["evidence"]["rule"] == "4"
        fill = world.store.fill_scan(ready)
        assert (fill.head, fill.covered, fill.total) == (None, 2, 2)
        assert world.service.received == []
