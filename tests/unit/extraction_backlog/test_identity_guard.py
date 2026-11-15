"""The dispatcher refuses a service or backend whose identity is not the epoch's.

A bound epoch (or candidate) names the serving identity it reads and writes
under and the dispatch identity of the backend settings that gate it. The
dispatcher checks the first at every `/v1/info` contact, sends it on every job
(`ExpectStamps.serving_identity`) and checks the reply's stamp; it checks the
second before it applies or dispatches anything. An unbound target is bound
first, at the first successful contact, and waits (stalled) while the service
cannot be read.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from backend.knowledge.curation.graph_writer import RebuildStamps
from backend.knowledge.extraction.serving_identity import diff_components
from backend.knowledge.extraction_cache import OUTCOME_EXTRACTED, keyed_namespace
from tests.unit.extraction_backlog.conftest import ONTOLOGY_VERSION, composed, wait_until
from tests.unit.extraction_backlog.fakes import serving_identity_of

DISPATCHER_LOGGER = "backend.extraction_backlog.dispatcher"


def _log(world, ts, i: int) -> str:
    return world.log_turn(
        session_id="s1", turn_index=i, timestamp=ts(i), utterance=f"I really use tool{i}"
    )


def _cache_extracted(world, event_id: str, created_at: str) -> None:
    """A cached, unapplied row under the active epoch: the head is applied, not dispatched."""
    epoch = world.store.active_epoch()
    world.cache.put(
        event_id,
        ONTOLOGY_VERSION,
        epoch.extraction_version,
        epoch.model_hash,
        outcome=OUTCOME_EXTRACTED,
        entities=[{"id": "rust", "type": "Technology", "name": "Rust"}],
        relationships=[],
        created_at=created_at,
    )


def _settings_change_cutover(world, ts):
    """A candidate with its source's triple, as `--settings-change` opens one."""
    epoch = world.store.active_epoch()
    return world.store.begin_cutover(
        ontology_version=epoch.ontology_version,
        extraction_version=epoch.extraction_version,
        model_hash=epoch.model_hash,
        bare_model_hash=world.service.model_hash,
        source_epoch_id=epoch.epoch_id,
        requested_at=ts(30),
    )


class TestTheServingIdentity:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "name,value",
        [
            ("n_ctx", 32768),
            ("server.device", "rtx-4070"),
            ("request.extract.top_p", 0.95),
            ("server.n_cpu_moe", "12"),
        ],
    )
    async def test_a_changed_component_stalls_a_bound_epoch_naming_it(
        self, backlog_world, ts, name, value
    ):
        # Arrange: bound to the service as it was, then the service restarted.
        world = backlog_world
        world.bind_active_epoch()
        bound = dict(world.service.serving_components)
        _log(world, ts, 0)
        world.service.serving_components[name] = value
        dispatcher = world.build_dispatcher()

        # Act
        await dispatcher.start()
        await wait_until(lambda: dispatcher.state == "epoch_mismatch")

        # Assert
        diffs = diff_components(bound, world.service.serving_components)
        assert len(diffs) == 1 and diffs[0].startswith(f"{name}: ")
        assert dispatcher.reason == (
            f"service serving identity {world.service.serving_identity} != epoch 1's "
            f"{serving_identity_of(bound)}: {diffs[0]}"
        )
        assert world.service.received == []
        assert world.attempts() == []
        assert world.curation.calls == []

    @pytest.mark.asyncio
    async def test_restoring_the_component_resumes(self, backlog_world, ts):
        world = backlog_world
        world.bind_active_epoch()
        event_id = _log(world, ts, 0)
        world.service.serving_components["n_ctx"] = 32768
        dispatcher = world.build_dispatcher()
        await dispatcher.start()
        await wait_until(lambda: dispatcher.state == "epoch_mismatch")

        world.service.serving_components["n_ctx"] = 8192
        await wait_until(lambda: world.curation.event_ids == [event_id])

        assert world.curation.event_ids == [event_id]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bound", [False, True], ids=["unbound", "bound"])
    async def test_a_pre_1_3_0_service_stalls_with_its_contract_named(
        self, backlog_world, ts, bound
    ):
        world = backlog_world
        if bound:
            world.bind_active_epoch()
        _log(world, ts, 0)
        world.service.contract_version = "1.2.0"
        world.service.reports_serving_identity = False
        dispatcher = world.build_dispatcher()

        await dispatcher.start()
        await wait_until(lambda: dispatcher.state == "stalled")

        assert dispatcher.reason == (
            "service reports no serving identity: contract 1.2.0 predates 1.3.0"
        )
        assert (world.event_store.get_epoch_identity(world.epoch_id) is not None) is bound
        assert world.service.received == []

    @pytest.mark.asyncio
    async def test_a_service_that_cannot_compute_its_identity_stalls_with_its_error(
        self, backlog_world, ts
    ):
        world = backlog_world
        _log(world, ts, 0)
        world.service.serving_identity_error = "llama-server /props unreachable"
        dispatcher = world.build_dispatcher()

        await dispatcher.start()
        await wait_until(lambda: dispatcher.state == "stalled")

        assert dispatcher.reason == (
            "service reports no serving identity: llama-server /props unreachable"
        )
        assert world.event_store.get_epoch_identity(world.epoch_id) is None
        assert world.service.received == []


class TestTheJob:
    @pytest.mark.asyncio
    async def test_the_expect_carries_the_epochs_serving_identity(self, backlog_world, ts):
        world = backlog_world
        _log(world, ts, 0)
        dispatcher = world.build_dispatcher()

        await dispatcher.start()
        assert await dispatcher.drain(timeout=5.0)

        bound = world.event_store.get_epoch_identity(world.epoch_id)["serving_identity"]
        assert [r.expect.serving_identity for r in world.service.received] == [bound]

    @pytest.mark.asyncio
    async def test_a_reply_stamped_with_another_identity_is_deferred_not_counted(
        self, backlog_world, ts
    ):
        world = backlog_world
        world.bind_active_epoch()
        event_id = _log(world, ts, 0)
        world.service.result_serving_identity = "si1-changed"
        dispatcher = world.build_dispatcher()

        await dispatcher.start()
        await wait_until(lambda: dispatcher.state == "epoch_mismatch")

        assert dispatcher.reason == (
            f"reply serving identity si1-changed != epoch 1's {world.service.serving_identity}"
        )
        rows = world.attempts()
        assert rows and all(
            (r["outcome"], r["error_code"], r["counted"]) == ("deferred", "epoch_mismatch", 0)
            for r in rows
        )
        epoch = world.store.active_epoch()
        assert world.cache.get(event_id, epoch.extraction_version, epoch.model_hash) is None
        assert world.curation.calls == []

    @pytest.mark.asyncio
    async def test_a_service_changed_mid_job_answers_409_and_the_next_contact_names_it(
        self, backlog_world, ts
    ):
        # Arrange: one job held in flight.
        world = backlog_world
        world.bind_active_epoch()
        event_id = _log(world, ts, 0)
        world.service.hold = asyncio.Event()
        dispatcher = world.build_dispatcher()
        await dispatcher.start()
        await wait_until(lambda: len(world.service.received) == 1)

        # Act: the service restarts with another context size, then answers.
        world.service.serving_components["n_ctx"] = 32768
        world.service.hold.set()
        await wait_until(lambda: (dispatcher.reason or "").startswith("service serving identity "))

        # Assert
        first = world.attempts()[0]
        assert (first["outcome"], first["error_code"], first["counted"]) == (
            "deferred",
            "epoch_mismatch",
            0,
        )
        assert dispatcher.state == "epoch_mismatch"
        assert "n_ctx: " in dispatcher.reason
        assert world.curation.calls == []

        # Act: restored.
        world.service.serving_components["n_ctx"] = 8192
        await wait_until(lambda: world.curation.event_ids == [event_id])
        assert world.curation.event_ids == [event_id]


class TestTheDispatchIdentity:
    @pytest.mark.asyncio
    async def test_other_backend_settings_stall_before_any_contact_or_apply(
        self, backlog_world, ts, caplog
    ):
        # Arrange: bound with history_messages=10; the backend now runs with 4,
        # and the head is cached, so only the guard keeps it from the graph.
        world = backlog_world
        world.bind_active_epoch()
        event_id = _log(world, ts, 0)
        _cache_extracted(world, event_id, ts(0))
        bound = world.dispatch_identity()
        dispatcher = world.build_dispatcher(history_messages=4)

        # Act
        with caplog.at_level(logging.ERROR, logger=DISPATCHER_LOGGER):
            await dispatcher.start()
            await wait_until(lambda: dispatcher.state == "stalled")
            await asyncio.sleep(0.1)

        # Assert
        diffs = diff_components(
            dict(bound.components), dict(dispatcher.dispatch_identity.components)
        )
        assert diffs and all(d.startswith("history_messages: ") for d in diffs)
        assert dispatcher.reason == (
            f"backend dispatch settings {dispatcher.dispatch_identity.identity} != epoch 1's "
            f"{bound.identity}: {'; '.join(diffs)}; restore them or begin a cutover with "
            "--settings-change"
        )
        assert world.service.info_calls == 0
        assert world.curation.calls == []
        assert world.event_store.get_extraction_applied(world.epoch_id) == {}
        refusals = [r for r in caplog.records if "refuses to dispatch" in r.getMessage()]
        assert len(refusals) == 1 and refusals[0].levelno == logging.ERROR

    @pytest.mark.asyncio
    async def test_a_bound_candidate_is_not_filled_under_other_backend_settings(
        self, backlog_world, ts
    ):
        # Arrange: an open candidate bound (keyed) under history_messages=10.
        world = backlog_world
        world.bind_active_epoch()
        _log(world, ts, 0)
        cutover = _settings_change_cutover(world, ts)
        world.service.serving_components["n_ctx"] = 32768
        bound = world.dispatch_identity()
        si = world.service.serving_identity
        assert world.event_store.bind_cutover_identity(
            cutover.cutover_id,
            serving_identity=si,
            serving_components=dict(world.service.serving_components),
            dispatch_identity=bound.identity,
            dispatch_components=dict(bound.components),
            identity_namespace=keyed_namespace(si, bound.identity),
            source="first-contact",
            evidence={},
            bound_at=ts(31),
        )
        dispatcher = world.build_dispatcher(history_messages=4)

        # Act
        await dispatcher.start()
        await wait_until(lambda: dispatcher.state == "stalled")
        await asyncio.sleep(0.1)

        # Assert
        assert dispatcher.reason.startswith(
            f"backend dispatch settings {dispatcher.dispatch_identity.identity} != "
            f"cutover {cutover.cutover_id} candidate's {bound.identity}: history_messages: "
        )
        assert world.service.info_calls == 0
        assert world.service.received == []
        assert world.store.get_cutover(cutover.cutover_id).state == "filling"


class TestBindingWaitsForTheService:
    @pytest.mark.asyncio
    async def test_an_unbound_epoch_stalls_and_applies_nothing_until_it_can_bind(
        self, backlog_world, ts
    ):
        # Arrange: a cached head, an unbound epoch, the service down.
        world = backlog_world
        event_id = _log(world, ts, 0)
        _cache_extracted(world, event_id, ts(0))
        world.service.down = True
        dispatcher = world.build_dispatcher()

        # Act
        await dispatcher.start()
        await wait_until(lambda: dispatcher.state == "stalled")
        await asyncio.sleep(0.1)

        # Assert
        assert dispatcher.reason.startswith(
            "epoch 1 is not bound yet: waiting for the extraction service ("
        )
        assert dispatcher.reason.endswith(")")
        assert dispatcher.snapshot().service.reachable is False
        assert world.curation.calls == []
        assert world.event_store.get_epoch_identity(world.epoch_id) is None

        # Act: the service comes back.
        world.service.down = False
        await wait_until(lambda: world.curation.event_ids == [event_id])

        # Assert: bound first, then the cached head applied with no inference.
        binding = world.event_store.get_epoch_identity(world.epoch_id)
        assert binding["serving_identity"] == world.service.serving_identity
        assert world.curation.event_ids == [event_id]
        assert world.service.received == []

    @pytest.mark.asyncio
    async def test_the_writer_guard_runs_before_binding(self, backlog_world, ts):
        world = backlog_world
        _log(world, ts, 0)
        other = RebuildStamps(
            ontology_version=ONTOLOGY_VERSION,
            extraction_version="ev-test",
            model_hash=composed("another-model"),
        )
        dispatcher = world.build_dispatcher(writer_stamps=other)

        await dispatcher.start()
        await wait_until(lambda: dispatcher.state == "stalled")
        await asyncio.sleep(0.1)

        assert dispatcher.reason.startswith("backend writer stamps ")
        assert world.service.info_calls == 0
        assert world.event_store.get_epoch_identity(world.epoch_id) is None

    @pytest.mark.asyncio
    async def test_activation_is_recorded_while_the_service_is_down(
        self, unactivated_backlog_world
    ):
        world = unactivated_backlog_world
        world.service.down = True
        dispatcher = world.build_dispatcher()

        await dispatcher.start()
        await wait_until(lambda: dispatcher.state == "stalled")

        assert world.event_store.get_extraction_activation(world.epoch_id) is not None
        assert world.event_store.get_epoch_identity(world.epoch_id) is None


class TestTheStatus:
    @pytest.mark.asyncio
    async def test_the_service_and_cutover_report_their_serving_identities(self, backlog_world, ts):
        world = backlog_world
        _log(world, ts, 0)
        dispatcher = world.build_dispatcher()
        await dispatcher.start()
        assert await dispatcher.drain(timeout=5.0)
        assert dispatcher.snapshot().service.serving_identity == world.service.serving_identity

        cutover = _settings_change_cutover(world, ts)
        world.service.serving_components["n_ctx"] = 32768
        dispatcher.wake()
        await wait_until(lambda: world.store.get_cutover(cutover.cutover_id).bound)

        status = dispatcher.snapshot()
        assert status.cutover.target_serving_identity == world.service.serving_identity
        assert status.service.serving_identity == world.service.serving_identity

    @pytest.mark.asyncio
    async def test_an_unbound_cutover_reports_no_target_identity(self, backlog_world, ts):
        world = backlog_world
        world.bind_active_epoch()
        _settings_change_cutover(world, ts)
        world.service.down = True
        dispatcher = world.build_dispatcher()

        await dispatcher.start()
        await wait_until(lambda: dispatcher.state == "stalled")

        assert dispatcher.snapshot().cutover.target_serving_identity is None


class TestTheFill:
    @pytest.mark.asyncio
    async def test_rule_1_refuses_until_the_service_changes_then_binds_keyed_by_rule_4(
        self, backlog_world, ts, caplog
    ):
        # Arrange: a settings-change candidate; the service not yet restarted.
        world = backlog_world
        world.bind_active_epoch()
        event_id = _log(world, ts, 0)
        cutover = _settings_change_cutover(world, ts)
        dispatcher = world.build_dispatcher()

        # Act
        await dispatcher.start()
        await wait_until(lambda: dispatcher.state == "epoch_mismatch")
        await asyncio.sleep(0.1)

        # Assert
        assert dispatcher.reason == (
            f"cutover {cutover.cutover_id}: the service still reports the active epoch's "
            "identity; restart it with the candidate settings"
        )
        assert not world.store.get_cutover(cutover.cutover_id).bound
        assert world.service.received == []

        # Act: the service restarted with the candidate settings.
        world.service.serving_components["n_ctx"] = 32768
        with caplog.at_level(logging.INFO, logger=DISPATCHER_LOGGER):
            await wait_until(lambda: world.store.get_cutover(cutover.cutover_id).state == "ready")

        # Assert: the active epoch's 3-part rows are claimed, so it is keyed.
        bound = world.store.get_cutover(cutover.cutover_id)
        si, di = world.service.serving_identity, world.dispatch_identity().identity
        assert bound.identity_namespace == keyed_namespace(si, di)
        assert world.event_store.get_cutover_identity(cutover.cutover_id)["evidence"]["rule"] == "4"
        assert [r.event_id for r in world.service.received] == [event_id]
        assert [r.expect.serving_identity for r in world.service.received] == [si]
        line = next(r for r in caplog.records if r.getMessage().startswith("epoch_identity"))
        assert line.levelno == logging.WARNING
        assert f"target=cutover:{cutover.cutover_id} " in line.getMessage()
        assert "rule=4" in line.getMessage() and "namespace=keyed" in line.getMessage()

    @pytest.mark.asyncio
    async def test_an_unbound_candidate_is_never_ready_even_with_an_empty_log(
        self, backlog_world, ts
    ):
        # Arrange: nothing logged, so the fill has no head; the service is down.
        world = backlog_world
        world.bind_active_epoch()
        cutover = _settings_change_cutover(world, ts)
        world.service.serving_components["n_ctx"] = 32768
        world.service.down = True
        dispatcher = world.build_dispatcher()

        # Act
        await dispatcher.start()
        await wait_until(lambda: dispatcher.state == "stalled")
        await asyncio.sleep(0.2)

        # Assert
        assert world.store.get_cutover(cutover.cutover_id).state == "filling"
        assert dispatcher.reason.startswith(
            f"cutover {cutover.cutover_id} candidate is not bound yet: waiting for the "
            "extraction service ("
        )

        # Act: reachable again.
        world.service.down = False
        await wait_until(lambda: world.store.get_cutover(cutover.cutover_id).state == "ready")

        # Assert
        assert world.store.get_cutover(cutover.cutover_id).bound
