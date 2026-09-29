"""Extraction status: the `extraction_status` push, `GET /extraction/status`, `/health`,
and the admin CLI's `status` command (`TestAdminStatusCli`).
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import sqlite3
from types import SimpleNamespace

import pytest

from backend import server
from backend.extraction_backlog import admin, telemetry
from backend.extraction_backlog.status import extraction_status, health_block
from backend.knowledge.version_stamps import EXTRACTION_VERSION as CODE_EXTRACTION_VERSION
from backend.knowledge.version_stamps import ONTOLOGY_VERSION as CODE_ONTOLOGY_VERSION
from tests.unit.extraction_backlog.conftest import (
    EMBEDDING_MODEL,
    ONTOLOGY_VERSION,
    composed,
    wait_until,
)


@pytest.fixture(autouse=True)
def _zero_unrecorded():
    telemetry.reset_unrecorded_turns()
    yield
    telemetry.reset_unrecorded_turns()


def _drain(queue: asyncio.Queue) -> list[dict]:
    messages = []
    while not queue.empty():
        messages.append(json.loads(queue.get_nowait()))
    return messages


class TestProducer:
    def test_no_dispatcher_is_disabled_with_zero_counts_and_unrecorded_turns(self):
        # Arrange
        telemetry.record_unrecorded_turn("turn-1")
        telemetry.record_unrecorded_turn("turn-2")

        # Act
        status = extraction_status(None)

        # Assert
        assert status.state == "disabled"
        assert (status.backlog_depth, status.apply_pending, status.dead_lettered) == (0, 0, 0)
        assert status.unrecorded_turns == 2
        assert status.legacy_unextracted == 0
        assert status.service.reachable is False
        assert status.model_dump(mode="json")["type"] == "extraction_status"

    def test_off_mode_dispatcher_is_disabled_with_zero_counts(self, backlog_world, ts):
        world = backlog_world
        world.log_turn(session_id="s1", turn_index=0, timestamp=ts(0), utterance="I use rust")
        dispatcher = world.build_dispatcher(mode="off")

        status = extraction_status(dispatcher)

        assert status.state == "disabled"
        assert status.backlog_depth == 0

    def test_a_running_dispatcher_reports_its_backlog(self, backlog_world, ts):
        world = backlog_world
        world.log_turn(session_id="s1", turn_index=0, timestamp=ts(0), utterance="I use rust")
        telemetry.record_unrecorded_turn(None)
        dispatcher = world.build_dispatcher()

        status = extraction_status(dispatcher)

        assert status.state == "idle"
        assert status.backlog_depth == 1
        assert status.unrecorded_turns == 1

    @pytest.mark.asyncio
    async def test_a_running_dispatcher_reports_its_legacy_unextracted_count(
        self, unactivated_backlog_world, ts
    ):
        world = unactivated_backlog_world
        world.log_turn(session_id="s1", turn_index=0, timestamp=ts(0), utterance="I use rust")
        world.log_turn(session_id="s1", turn_index=1, timestamp=ts(1), utterance="I use zig")
        dispatcher = world.build_dispatcher()
        await dispatcher.start()
        assert await dispatcher.drain(timeout=5.0)

        status = extraction_status(dispatcher)

        assert status.legacy_unextracted == dispatcher.legacy_unextracted == 2

    def test_a_snapshot_storage_failure_reports_the_state_with_zero_counts(self):
        class _Broken:
            mode = "service"
            state = "working"

            def snapshot(self):
                raise sqlite3.OperationalError("database is locked")

        status = extraction_status(_Broken())  # type: ignore[arg-type]

        assert status.state == "working"
        assert status.backlog_depth == 0

    def test_health_block_is_a_subset_of_the_status(self):
        status = extraction_status(None)

        block = health_block(status)

        assert block == {
            "state": "disabled",
            "backlog_depth": 0,
            "dead_lettered": 0,
            "unrecorded_turns": 0,
            "service": {"reachable": False},
        }


class TestWebSocketPush:
    @pytest.mark.asyncio
    async def test_the_timer_pushes_extraction_status_with_no_dispatcher(self, monkeypatch):
        # Arrange
        captured: asyncio.Queue = asyncio.Queue()
        monkeypatch.setattr(server, "message_queue", captured)
        monkeypatch.setattr(server, "extraction_dispatcher", None)
        telemetry.record_unrecorded_turn("turn-9")

        # Act
        task = asyncio.create_task(server.extraction_status_loop(interval_seconds=0.02))
        await asyncio.sleep(0.07)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        # Assert
        messages = _drain(captured)
        assert len(messages) >= 2
        for message in messages:
            assert message["type"] == "extraction_status"
            assert message["state"] == "disabled"
            assert message["unrecorded_turns"] == 1

    @pytest.mark.asyncio
    async def test_the_timer_reports_the_running_dispatcher(self, monkeypatch, backlog_world, ts):
        world = backlog_world
        world.log_turn(session_id="s1", turn_index=0, timestamp=ts(0), utterance="I use rust")
        captured: asyncio.Queue = asyncio.Queue()
        monkeypatch.setattr(server, "message_queue", captured)
        monkeypatch.setattr(server, "extraction_dispatcher", world.build_dispatcher())

        task = asyncio.create_task(server.extraction_status_loop(interval_seconds=0.02))
        await asyncio.sleep(0.05)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        message = _drain(captured)[0]
        assert message["state"] == "idle"
        assert message["backlog_depth"] == 1

    @pytest.mark.asyncio
    async def test_a_state_transition_pushes_at_once(self, monkeypatch, backlog_world, ts):
        # Arrange: the service is down, so start() moves idle -> unreachable.
        world = backlog_world
        world.service.down = True
        world.log_turn(session_id="s1", turn_index=0, timestamp=ts(0), utterance="I use rust")
        captured: asyncio.Queue = asyncio.Queue()
        monkeypatch.setattr(server, "message_queue", captured)
        dispatcher = world.build_dispatcher()
        dispatcher.add_state_listener(lambda _p, _c: server.push_extraction_status(dispatcher))

        # Act
        await dispatcher.start()
        await wait_until(lambda: not captured.empty())

        # Assert: pushed by the transition, no timer running.
        message = json.loads(captured.get_nowait())
        assert message["type"] == "extraction_status"
        assert message["state"] == "unreachable"
        assert message["backlog_depth"] == 1

    @pytest.mark.asyncio
    async def test_server_wiring_registers_the_transition_push(
        self, monkeypatch, backlog_world, ts
    ):
        """`_start_extraction_dispatcher` itself wires the listener.

        The service URL is a closed loopback port, so the first `/v1/info` is
        refused locally and the dispatcher moves idle -> unreachable.
        """
        world = backlog_world
        world.log_turn(session_id="s1", turn_index=0, timestamp=ts(0), utterance="I use rust")
        monkeypatch.setenv("MIST_EXTRACTION_INFERENCE", "service")
        monkeypatch.setenv("MIST_EXTRACTION_SERVICE_URL", "http://127.0.0.1:9")
        monkeypatch.setenv("MIST_EXTRACTION_STALL_RECHECK_S", "0.05")
        captured: asyncio.Queue = asyncio.Queue()
        monkeypatch.setattr(server, "message_queue", captured)
        handler = SimpleNamespace(
            event_store=world.event_store,
            _extraction_pipeline=world.build_pipeline(),
            attach_extraction_dispatcher=lambda d: None,
        )
        voice_processor = SimpleNamespace(
            models=SimpleNamespace(knowledge=SimpleNamespace(conversation_handler=handler))
        )
        from tests.mocks.config import build_test_config

        config = build_test_config(embedding_model="test-emb")
        # The backend under test is configured to match the active epoch;
        # the writer-stamp guard would otherwise stall it.
        config.model_hash = world.service.model_hash
        config.extraction_version = world.service.extraction_version
        config.ontology_version = ONTOLOGY_VERSION
        dispatcher = await server._start_extraction_dispatcher(voice_processor, config)
        world.dispatchers.append(dispatcher)
        await wait_until(lambda: not captured.empty())

        message = json.loads(captured.get_nowait())
        assert message["state"] == "unreachable"
        assert message["backlog_depth"] == 1


def _matching_config(world):
    """A backend config whose writer stamps equal the world's active epoch."""
    from tests.mocks.config import build_test_config

    config = build_test_config(embedding_model=EMBEDDING_MODEL)
    config.model_hash = world.service.model_hash
    config.extraction_version = world.service.extraction_version
    config.ontology_version = ONTOLOGY_VERSION
    return config


def _admin_status(world, config) -> tuple[int, str]:
    out = io.StringIO()
    code = admin.main(["status"], store=world.store, out=out, knowledge_config=config)
    return code, out.getvalue()


class TestAdminStatusCli:
    """`python -m backend.extraction_backlog.admin status`: the stamp lines."""

    def test_it_prints_the_epoch_and_code_stamps_and_a_writer_stamp_match(self):
        from tests.unit.extraction_backlog.conftest import _build_world

        world = _build_world(with_deriver=False)
        config = _matching_config(world)

        code, text = _admin_status(world, config)

        assert code == 0, text
        assert (
            f"epoch {world.epoch_id}: ontology_version={ONTOLOGY_VERSION} "
            f"extraction_version={world.service.extraction_version} "
            f"model_hash={composed(world.service.model_hash)}"
        ) in text
        assert (
            f"code: ONTOLOGY_VERSION={CODE_ONTOLOGY_VERSION} "
            f"EXTRACTION_VERSION={CODE_EXTRACTION_VERSION} "
            f"configured model_hash={world.service.model_hash} (MIST_MODEL_HASH; composed "
            f"{composed(world.service.model_hash)})"
        ) in text
        assert "writer stamps: match\n" in text
        assert "MISMATCH" not in text

    def test_a_different_model_hash_is_a_mismatch_naming_the_field(self):
        from tests.unit.extraction_backlog.conftest import _build_world

        world = _build_world(with_deriver=False)
        config = _matching_config(world)
        config.model_hash = "some-other-model"

        code, text = _admin_status(world, config)

        assert code == 0, text
        assert "writer stamps: MISMATCH (model_hash)\n" in text
        # The dispatcher guard's own reason follows, with both triples.
        assert "nothing is applied or dispatched until they match" in text
        assert "some-other-model" in text

    def test_every_differing_field_is_named(self):
        from tests.unit.extraction_backlog.conftest import _build_world

        world = _build_world(with_deriver=False)
        config = _matching_config(world)
        config.ontology_version = "9.9.9"
        config.extraction_version = "other-ev"
        config.model_hash = "some-other-model"

        _, text = _admin_status(world, config)

        assert "writer stamps: MISMATCH (ontology_version, extraction_version, model_hash)" in text

    def test_the_verdict_is_the_dispatcher_guards(self, monkeypatch):
        # `status` does not compare the stamps itself: whatever the guard
        # returns decides the line.
        from backend.extraction_backlog.dispatcher import ExtractionDispatcher
        from tests.unit.extraction_backlog.conftest import _build_world

        world = _build_world(with_deriver=False)
        monkeypatch.setattr(
            ExtractionDispatcher, "_writer_stamp_mismatch", lambda self, epoch: "guard says no"
        )

        _, text = _admin_status(world, _matching_config(world))

        assert "writer stamps: MISMATCH (see reason)\n  guard says no" in text

    def test_an_open_cutover_prints_its_candidate_versions(self, ts):
        from tests.unit.extraction_backlog.conftest import _build_world

        world = _build_world(with_deriver=False)
        store = world.store
        store.begin_cutover(
            ontology_version="1.5.0",
            extraction_version="candidate-ev",
            model_hash=composed("svc-model-2"),
            bare_model_hash="svc-model-2",
            source_epoch_id=world.epoch_id,
            requested_at=ts(0),
        )

        code, text = _admin_status(world, _matching_config(world))

        assert code == 0, text
        assert (
            "  candidate: ontology_version=1.5.0 extraction_version=candidate-ev "
            f"model_hash={composed('svc-model-2')}"
        ) in text
        # Before promotion the writer stamps are still compared with the ACTIVE epoch.
        assert "writer stamps: match" in text

    def test_an_empty_ledger_exits_1(self):
        from backend.event_store.store import EventStore
        from backend.extraction_backlog.store import BacklogStore
        from backend.knowledge.extraction_cache import ExtractionCache

        event_store = EventStore(db_path=":memory:")
        event_store.initialize()
        cache = ExtractionCache(":memory:")
        cache.initialize()
        out = io.StringIO()

        code = admin.main(
            ["status"],
            store=BacklogStore(event_store, cache),
            out=out,
            knowledge_config=SimpleNamespace(),
        )

        assert code == 1
        assert "No epoch in the ledger" in out.getvalue()


class TestHttp:
    @pytest.mark.asyncio
    async def test_get_extraction_status_returns_the_snapshot(self, monkeypatch, backlog_world, ts):
        world = backlog_world
        world.log_turn(session_id="s1", turn_index=0, timestamp=ts(0), utterance="I use rust")
        monkeypatch.setattr(server, "extraction_dispatcher", world.build_dispatcher())

        body = await server.get_extraction_status()

        assert body["type"] == "extraction_status"
        assert body["state"] == "idle"
        assert body["backlog_depth"] == 1
        assert body["cutover"] is None

    @pytest.mark.asyncio
    async def test_get_extraction_status_without_a_dispatcher_is_disabled(self, monkeypatch):
        monkeypatch.setattr(server, "extraction_dispatcher", None)
        telemetry.record_unrecorded_turn("turn-3")

        body = await server.get_extraction_status()

        assert body["state"] == "disabled"
        assert body["unrecorded_turns"] == 1

    @pytest.mark.asyncio
    async def test_health_carries_the_extraction_block(self, monkeypatch, backlog_world, ts):
        world = backlog_world
        world.log_turn(session_id="s1", turn_index=0, timestamp=ts(0), utterance="I use rust")
        monkeypatch.setattr(server, "extraction_dispatcher", world.build_dispatcher())

        payload = await server.health()

        # No `_health_registry` exists in this unit context (no lifespan ran), so
        # `_health_snapshot` reports the honest "nothing measured" answer, `unhealthy`
        # -- see `server.health`'s docstring and `_health_snapshot`. This test predates
        # that measured-report wiring and asserted the old hardcoded "healthy" literal;
        # it checks the extraction block only now, like its
        # `test_health_without_a_dispatcher_reports_disabled` sibling below.
        assert payload["extraction"]["state"] == "idle"
        assert payload["extraction"]["backlog_depth"] == 1
        assert payload["extraction"]["service"] == {"reachable": False}

    @pytest.mark.asyncio
    async def test_health_without_a_dispatcher_reports_disabled(self, monkeypatch):
        monkeypatch.setattr(server, "extraction_dispatcher", None)

        payload = await server.health()

        assert payload["extraction"]["state"] == "disabled"
