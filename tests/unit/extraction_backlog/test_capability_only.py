"""Capability-only turns: covered by a `capability_only` skip row, never extracted (T31).

A turn a capability answered alone is logged with `answered_by="capability"`.
`eligibility.policy_skip_reason` turns that into a skip reason, the scan flags
the turn `policy_skip` and leaves it out of the counts, and the dispatcher
covers it with the same kind of row a gate skip writes, with no job and no
service call. These tests drive the real dispatcher against the fake service.
"""

from __future__ import annotations

import ast
import asyncio
import io
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from backend.event_store.models import ConversationTurnEvent
from backend.event_store.store import EventStore
from backend.extraction_backlog import admin, eligibility
from backend.extraction_backlog.eligibility import policy_skip_reason
from backend.extraction_backlog.store import BacklogStore
from backend.knowledge.extraction_cache import (
    OUTCOME_EXTRACTED,
    OUTCOME_SKIPPED,
    SKIP_CAPABILITY_ONLY,
    SKIP_EXTRACTION_FAILED,
    SKIP_TOO_SHORT,
    VALID_SKIP_REASONS,
    ExtractionCache,
)
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.ollama import FakeLLM
from tests.unit.extraction_backlog.conftest import (
    EMBEDDING_MODEL,
    ONTOLOGY_VERSION,
    BacklogWorld,
    _build_world,
    composed,
)
from tests.unit.extraction_backlog.fakes import FakeGraphCuration, FakeServiceState

CANDIDATE = "svc-model-2"
_CAPABILITY_UTTERANCE = "set a timer for ten minutes"


def _ts(minute: int) -> str:
    return f"2026-09-01T10:{minute:02d}:00+00:00"


def _log(
    world: BacklogWorld,
    *,
    index: int,
    minute: int,
    answered_by: str | None,
    utterance: str | None = None,
    session_id: str = "s1",
    event_id: str | None = None,
) -> str:
    if world.event_store.get_session(session_id) is None:
        world.event_store.start_session(session_id, input_modality="text")
    event = ConversationTurnEvent(
        session_id=session_id,
        turn_index=index,
        timestamp=datetime.fromisoformat(_ts(minute)),
        user_utterance=utterance
        or (_CAPABILITY_UTTERANCE if answered_by == "capability" else f"I really use tool{index}"),
        system_response="Timer set." if answered_by == "capability" else "Noted.",
        ontology_version=ONTOLOGY_VERSION,
        answered_by=answered_by,  # type: ignore[arg-type]
    )
    if event_id is not None:
        event.event_id = event_id
    return world.event_store.append_turn(event)


def _active_put(world: BacklogWorld, event_id: str, **kwargs: Any) -> None:
    epoch = world.store.active_epoch()
    world.cache.put(
        event_id,
        epoch.ontology_version,
        epoch.extraction_version,
        epoch.model_hash,
        **kwargs,
    )


def _raw_row(world: BacklogWorld, event_id: str, epoch) -> dict:
    """The stored cache row, every column (`ExtractionCache.get` omits `created_at`)."""
    row = (
        world.cache._get_connection()
        .execute(
            "SELECT * FROM extraction_cache WHERE event_id = ? AND extraction_version = ? "
            "AND model_hash = ?",
            (event_id, epoch.extraction_version, epoch.model_hash),
        )
        .fetchone()
    )
    assert row is not None
    return dict(row)


def _applied(world: BacklogWorld) -> dict[str, str]:
    return world.event_store.get_extraction_applied(world.epoch_id)


async def _settle(dispatcher, limit: int = 50) -> None:
    """Run `_step` on this task until one would sleep (idle or backoff)."""
    for _ in range(limit):
        if await dispatcher._step() is not None:
            return
    raise AssertionError("dispatcher did not settle")


@pytest.fixture
def sync_world():
    return _build_world(with_deriver=False)


# --- The predicate --------------------------------------------------------------


class TestPredicate:
    def test_capability_gives_the_capability_only_reason(self):
        assert policy_skip_reason("capability") == "capability_only" == SKIP_CAPABILITY_ONLY

    @pytest.mark.parametrize("value", ["model", None])
    def test_model_and_null_give_none(self, value):
        assert policy_skip_reason(value) is None

    def test_the_reason_is_a_valid_cache_skip_reason(self):
        assert SKIP_CAPABILITY_ONLY in VALID_SKIP_REASONS

    def test_the_module_imports_only_the_extraction_cache(self):
        tree = ast.parse(Path(eligibility.__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)

        assert imported - {"__future__"} == {"backend.knowledge.extraction_cache"}


# --- The scan -------------------------------------------------------------------


class TestScan:
    def test_an_uncached_capability_turn_is_the_head_and_not_counted(self, sync_world):
        world = sync_world
        capability = _log(world, index=0, minute=0, answered_by="capability")
        model = _log(world, index=1, minute=1, answered_by="model")

        scan = world.store.scan(world.store.active_epoch())

        assert [t.event_id for t in scan.pending] == [capability, model]
        assert scan.head is not None
        assert scan.head.event_id == capability
        assert scan.head.policy_skip == SKIP_CAPABILITY_ONLY
        assert scan.head.cached is False
        assert scan.pending[1].policy_skip is None
        assert scan.backlog_depth == 1
        assert scan.apply_pending == 0
        assert scan.oldest_pending_timestamp == _ts(1)
        assert scan.policy_skipped == {SKIP_CAPABILITY_ONLY: 1}

    def test_only_capability_turns_leave_no_lag(self, sync_world):
        world = sync_world
        _log(world, index=0, minute=0, answered_by="capability")

        scan = world.store.scan(world.store.active_epoch())

        assert scan.head is not None
        assert scan.backlog_depth == 0
        assert scan.apply_pending == 0
        assert scan.oldest_pending_timestamp is None

    def test_a_capability_only_row_with_no_marker_is_pending_and_not_counted(self, sync_world):
        world = sync_world
        capability = _log(world, index=0, minute=0, answered_by="capability")
        _active_put(
            world,
            capability,
            outcome=OUTCOME_SKIPPED,
            skip_reason=SKIP_CAPABILITY_ONLY,
            created_at=_ts(0),
        )

        scan = world.store.scan(world.store.active_epoch())

        assert scan.head is not None
        assert scan.head.event_id == capability
        assert scan.head.cached is True
        assert scan.head.policy_skip == SKIP_CAPABILITY_ONLY
        assert scan.apply_pending == 0
        assert scan.backlog_depth == 0
        assert scan.policy_skipped == {SKIP_CAPABILITY_ONLY: 1}

    def test_a_capability_turn_with_an_extracted_row_is_counted_as_today(self, sync_world):
        world = sync_world
        capability = _log(world, index=0, minute=0, answered_by="capability")
        _active_put(
            world,
            capability,
            outcome=OUTCOME_EXTRACTED,
            entities=[{"id": "timer", "type": "Technology", "name": "Timer"}],
            relationships=[],
            created_at=_ts(0),
        )

        scan = world.store.scan(world.store.active_epoch())

        assert scan.head is not None
        assert scan.head.policy_skip is None
        assert scan.apply_pending == 1
        assert scan.oldest_pending_timestamp == _ts(0)
        assert scan.policy_skipped == {}

    def test_a_capability_turn_with_another_skip_reason_is_counted_as_today(self, sync_world):
        world = sync_world
        capability = _log(world, index=0, minute=0, answered_by="capability")
        _active_put(
            world,
            capability,
            outcome=OUTCOME_SKIPPED,
            skip_reason=SKIP_TOO_SHORT,
            created_at=_ts(0),
        )

        scan = world.store.scan(world.store.active_epoch())

        assert scan.head is not None
        assert scan.head.policy_skip is None
        assert scan.apply_pending == 1
        assert scan.policy_skipped == {}

    def test_policy_skipped_counts_pending_and_applied_and_leaves_legacy_out(self):
        world = _build_world(with_deriver=False, activated=False)
        legacy = _log(world, index=0, minute=0, answered_by="capability")
        store = world.store
        epoch = store.active_epoch()
        store.ensure_activation(epoch, now_iso="2026-09-01T10:00:30+00:00")
        applied = _log(world, index=1, minute=1, answered_by="capability")
        _active_put(
            world,
            applied,
            outcome=OUTCOME_SKIPPED,
            skip_reason=SKIP_CAPABILITY_ONLY,
            created_at=_ts(1),
        )
        world.event_store.mark_extraction_stage(
            event_id=applied, epoch_id=epoch.epoch_id, stage="applied", updated_at=_ts(1)
        )
        pending = _log(world, index=2, minute=2, answered_by="capability")

        scan = store.scan(epoch)

        assert [t.event_id for t in scan.pending] == [pending]
        assert legacy not in {t.event_id for t in scan.pending}
        assert scan.legacy_unextracted == 1
        assert scan.policy_skipped == {SKIP_CAPABILITY_ONLY: 2}

    def test_an_old_null_turn_is_counted_as_today(self, sync_world):
        world = sync_world
        _log(world, index=0, minute=0, answered_by=None)

        scan = world.store.scan(world.store.active_epoch())

        assert scan.head is not None
        assert scan.head.policy_skip is None
        assert scan.backlog_depth == 1
        assert scan.oldest_pending_timestamp == _ts(0)
        assert scan.policy_skipped == {}


# --- The dispatcher, active epoch ----------------------------------------------


class TestActiveEpoch:
    @pytest.mark.asyncio
    async def test_an_uncached_capability_head_is_covered_with_no_job(
        self, backlog_world, monkeypatch
    ):
        world = backlog_world
        capability = _log(world, index=0, minute=0, answered_by="capability")
        dispatcher = world.build_dispatcher()

        async def _no_service(_epoch):
            raise AssertionError("_check_service must not run for a capability head")

        monkeypatch.setattr(dispatcher, "_check_service", _no_service)

        wait = await dispatcher._step()

        assert wait is None
        epoch = world.store.active_epoch()
        row = _raw_row(world, capability, epoch)
        assert row["outcome"] == OUTCOME_SKIPPED
        assert row["skip_reason"] == SKIP_CAPABILITY_ONLY
        assert row["created_at"] == _ts(0)
        assert row["ontology_version"] == epoch.ontology_version
        assert _applied(world) == {capability: "applied"}
        assert world.service.received == []
        assert world.service.info_calls == 0
        assert world.curation.calls == []
        assert world.main_llm.calls == []

    @pytest.mark.asyncio
    async def test_a_stale_marker_is_cleared_before_the_row_is_written(self, backlog_world):
        world = backlog_world
        capability = _log(world, index=0, minute=0, answered_by="capability")
        world.event_store.mark_extraction_stage(
            event_id=capability, epoch_id=world.epoch_id, stage="applied", updated_at=_ts(0)
        )
        dispatcher = world.build_dispatcher()
        store = dispatcher._store
        markers_at_write: list[dict[str, str]] = []
        real_put_skip = store.put_skip

        def _put_skip(*args: Any, **kwargs: Any) -> None:
            markers_at_write.append(dict(_applied(world)))
            real_put_skip(*args, **kwargs)

        store.put_skip = _put_skip  # type: ignore[method-assign]

        await dispatcher._step()

        assert markers_at_write == [{}]
        assert _applied(world) == {capability: "applied"}

    @pytest.mark.asyncio
    async def test_an_existing_capability_only_row_is_applied_with_no_second_write(
        self, backlog_world
    ):
        world = backlog_world
        capability = _log(world, index=0, minute=0, answered_by="capability")
        _active_put(
            world,
            capability,
            outcome=OUTCOME_SKIPPED,
            skip_reason=SKIP_CAPABILITY_ONLY,
            created_at=_ts(0),
        )
        dispatcher = world.build_dispatcher()
        writes: list[tuple] = []
        dispatcher._store.put_skip = lambda *a, **k: writes.append((a, k))  # type: ignore[method-assign]

        assert await dispatcher._step() is None

        assert writes == []
        assert _applied(world) == {capability: "applied"}
        assert world.service.info_calls == 0

    @pytest.mark.asyncio
    async def test_covering_keeps_the_unreachable_state_and_the_model_head_backs_off(
        self, backlog_world
    ):
        world = backlog_world
        world.service.down = True
        model = _log(world, index=1, minute=5, answered_by="model")
        dispatcher = world.build_dispatcher()
        first = await dispatcher._step()
        assert first is not None
        assert dispatcher.state == "unreachable"
        capability = _log(world, index=0, minute=1, answered_by="capability")
        transitions: list[tuple[str, str]] = []
        dispatcher.add_state_listener(
            lambda previous, current: transitions.append((previous, current))
        )

        covered = await dispatcher._step()
        backoff = await dispatcher._step()

        assert covered is None
        assert _applied(world) == {capability: "applied"}
        assert transitions == []
        assert dispatcher.state == "unreachable"
        assert backoff is not None
        assert backoff.seconds == dispatcher._settings.backoff_s(2)
        assert backoff.interruptible is True
        scan = world.store.scan(world.store.active_epoch())
        assert scan.head is not None
        assert scan.head.event_id == model
        assert world.service.received == []

    @pytest.mark.asyncio
    async def test_a_run_of_capability_heads_emits_no_transition(self, backlog_world):
        world = backlog_world
        ids = [_log(world, index=i, minute=i, answered_by="capability") for i in range(3)]
        dispatcher = world.build_dispatcher()
        transitions: list[tuple[str, str]] = []
        dispatcher.add_state_listener(
            lambda previous, current: transitions.append((previous, current))
        )

        await _settle(dispatcher)

        assert transitions == []
        assert dispatcher.state == "idle"
        assert _applied(world) == {event_id: "applied" for event_id in ids}

    @pytest.mark.asyncio
    async def test_an_old_null_turn_is_still_dispatched(self, backlog_world):
        world = backlog_world
        null_turn = _log(world, index=0, minute=0, answered_by=None, utterance="I really use rust")
        dispatcher = world.build_dispatcher()

        await _settle(dispatcher)

        assert world.service.received_utterances == ["I really use rust"]
        assert world.curation.event_ids == [null_turn]


# --- The dispatcher, cutover fill ------------------------------------------------


def _begin_cutover(world: BacklogWorld) -> None:
    out = io.StringIO()
    code = admin.main(
        [
            "cutover",
            "begin",
            "--model-hash",
            CANDIDATE,
            "--extraction-version",
            world.service.extraction_version,
            "--ontology-version",
            ONTOLOGY_VERSION,
        ],
        store=world.store,
        out=out,
        embedding_model_name=EMBEDDING_MODEL,
    )
    assert code == 0, out.getvalue()
    world.service.model_hash = CANDIDATE


def _capability_rows(cache: ExtractionCache) -> list[dict]:
    rows = (
        cache._get_connection()
        .execute(
            "SELECT * FROM extraction_cache WHERE skip_reason = ? ORDER BY event_id",
            (SKIP_CAPABILITY_ONLY,),
        )
        .fetchall()
    )
    return [dict(row) for row in rows]


class TestCutoverFill:
    @pytest.mark.asyncio
    async def test_the_fill_writes_the_row_under_the_candidate_and_applies_nothing(
        self, backlog_world, caplog
    ):
        world = backlog_world
        capability = _log(world, index=0, minute=0, answered_by="capability")
        _begin_cutover(world)
        dispatcher = world.build_dispatcher()
        cutover = world.store.open_cutover()
        assert cutover is not None

        with caplog.at_level(logging.INFO, logger="backend.extraction_backlog.dispatcher"):
            assert await dispatcher._step() is None

        row = _raw_row(world, capability, cutover.target)
        assert row["outcome"] == OUTCOME_SKIPPED
        assert row["skip_reason"] == SKIP_CAPABILITY_ONLY
        assert row["created_at"] == _ts(0)
        assert world.store.get_cached(capability, world.store.active_epoch()) is None
        assert _applied(world) == {}
        assert world.event_store.get_extraction_applied(cutover.target.epoch_id) == {}
        assert world.service.received == []
        assert world.service.info_calls == 0
        lines = [r.getMessage() for r in caplog.records if "extraction job" in r.getMessage()]
        assert len(lines) == 1
        assert "error_code=capability_only outcome=skipped" in lines[0]

    @pytest.mark.asyncio
    async def test_filling_reaches_ready_with_capability_turns_in_the_log(self, backlog_world):
        world = backlog_world
        _log(world, index=0, minute=0, answered_by="capability")
        _log(world, index=1, minute=1, answered_by="model")
        _log(world, index=2, minute=2, answered_by="capability")
        _begin_cutover(world)
        dispatcher = world.build_dispatcher()

        await _settle(dispatcher)

        cutover = world.store.open_cutover()
        assert cutover is not None
        assert cutover.state == "ready"
        fill = world.store.fill_scan(cutover)
        assert (fill.covered, fill.total) == (3, 3)
        assert world.service.received_utterances == ["I really use tool1"]

    @pytest.mark.asyncio
    async def test_two_fills_into_fresh_caches_give_identical_rows(self):
        rows = []
        for _ in range(2):
            world = _build_world(with_deriver=False)
            _log(world, index=0, minute=0, answered_by="capability", event_id="cap-0")
            _log(world, index=1, minute=1, answered_by="capability", event_id="cap-1")
            _begin_cutover(world)
            dispatcher = world.build_dispatcher()
            try:
                await _settle(dispatcher)
            finally:
                await dispatcher.stop(timeout=2.0)
                for client in world.clients:
                    await client.aclose()
            assert world.store.open_cutover().state == "ready"
            rows.append(_capability_rows(world.cache))

        assert len(rows[0]) == 2
        assert rows[0] == rows[1]


# --- Drain and status -------------------------------------------------------------


class TestDrainAndStatus:
    @pytest.mark.asyncio
    async def test_drain_returns_true_at_once_with_only_capability_turns_uncovered(
        self, backlog_world
    ):
        world = backlog_world
        _log(world, index=0, minute=0, answered_by="capability")
        _log(world, index=1, minute=1, answered_by="capability")
        dispatcher = world.build_dispatcher()

        started = time.perf_counter()
        drained = await dispatcher.drain(0.1)
        elapsed = time.perf_counter() - started

        assert drained is True
        assert elapsed < 0.1
        assert (
            world.cache.event_ids_for(world.service.extraction_version, composed("svc-model")) == {}
        )

    @pytest.mark.asyncio
    async def test_drain_still_waits_on_an_uncovered_model_turn(self, backlog_world):
        world = backlog_world
        _log(world, index=0, minute=0, answered_by="capability")
        _log(world, index=1, minute=1, answered_by="model")
        dispatcher = world.build_dispatcher()

        assert await dispatcher.drain(0.1) is False

    @pytest.mark.asyncio
    async def test_snapshot_carries_policy_skipped(self, backlog_world):
        world = backlog_world
        for i in range(3):
            _log(world, index=i, minute=i, answered_by="capability")
        dispatcher = world.build_dispatcher()

        before = dispatcher.snapshot()
        await _settle(dispatcher)
        after = dispatcher.snapshot()

        assert before.policy_skipped == {SKIP_CAPABILITY_ONLY: 3}
        assert before.backlog_depth == 0
        assert before.oldest_pending_age_ms is None
        assert after.policy_skipped == {SKIP_CAPABILITY_ONLY: 3}
        assert after.model_dump(mode="json")["policy_skipped"] == {"capability_only": 3}

    def test_admin_status_prints_the_count(self, sync_world):
        world = sync_world
        _log(world, index=0, minute=0, answered_by="capability")
        _log(world, index=1, minute=1, answered_by="model")
        out = io.StringIO()

        code = admin.main(["status"], store=world.store, out=out)

        assert code == 0
        assert "policy_skipped=capability_only:1" in out.getvalue().splitlines()


# --- Never a failure ----------------------------------------------------------------


class TestNeverAFailure:
    @pytest.mark.asyncio
    async def test_an_applied_capability_only_row_is_not_a_failure(self, backlog_world):
        world = backlog_world
        capability = _log(world, index=0, minute=0, answered_by="capability")
        dispatcher = world.build_dispatcher()

        await _settle(dispatcher)

        epoch = world.store.active_epoch()
        scan = world.store.scan(epoch)
        status = dispatcher.snapshot()
        assert scan.dead_lettered == 0
        assert status.dead_lettered == 0
        assert world.store.list_dead_letters(epoch) == []
        assert world.store.retry_dead_letter(capability, epoch) is False
        assert world.store.get_cached(capability, epoch)["skip_reason"] == SKIP_CAPABILITY_ONLY
        assert status.last_job is not None
        assert status.last_job.event_id == capability
        assert status.last_job.outcome == "skipped"
        assert status.oldest_pending_age_ms is None
        assert SKIP_CAPABILITY_ONLY != SKIP_EXTRACTION_FAILED


# --- End to end -----------------------------------------------------------------------


def _tmp_world(tmp_path: Path) -> BacklogWorld:
    """`_build_world`, with the event store and the cache in files under `tmp_path`."""
    event_store = EventStore(db_path=str(tmp_path / "event_store.db"))
    event_store.initialize()
    service = FakeServiceState()
    epoch = event_store.ensure_initial_epoch(
        now_iso="2026-09-01T00:00:00+00:00",
        ontology_version=ONTOLOGY_VERSION,
        extraction_version=service.extraction_version,
        model_hash=composed(service.model_hash),
    )
    cache = ExtractionCache(str(tmp_path / "extraction_cache.db"))
    cache.initialize()
    store = BacklogStore(event_store, cache)
    store.ensure_activation(store.active_epoch(), now_iso="2026-09-01T00:00:00+00:00")
    return BacklogWorld(
        event_store=event_store,
        cache=cache,
        service=service,
        main_llm=FakeLLM(),
        curation=FakeGraphCuration(),
        deriver=None,
        epoch_id=int(epoch["epoch_id"]),
        embeddings=FakeEmbeddingGenerator(),
    )


class _CapabilityPort:
    """A `CapabilityTurnPort` that answers the timer command and leaves the rest to the model."""

    def __init__(self) -> None:
        from backend.capabilities.contracts import ActionRecord, PreTurnResolution, ToolOutcome

        self._resolution = PreTurnResolution(
            handled=True,
            reply="Timer set.",
            remainder=None,
            records=(
                ActionRecord(
                    tool_call_id="call-1",
                    tool="timers.start",
                    capability="timers",
                    risk="reversible",
                    args_summary="minutes=10",
                    outcome=ToolOutcome(status="ok", summary="Timer set.", data=None, reason=None),
                    duration_ms=12,
                    power_level="full",
                    ack="Setting a timer.",
                    view="timers",
                    source="fastpath",
                ),
            ),
            events=(),
        )

    async def pre_llm(
        self, session_id: str, turn_id: str, utterance: str, *, received_at: float | None = None
    ) -> Any:
        return self._resolution if utterance == _CAPABILITY_UTTERANCE else None

    def plan_turn(self, *args: Any, **kwargs: Any):  # noqa: ANN201
        raise AssertionError("no tiers are attached, so the turn is not routed")

    async def run_actions(self, *args: Any, **kwargs: Any):  # noqa: ANN201
        raise AssertionError("no tiers are attached, so no action decode runs")

    async def decide(self, *args: Any, **kwargs: Any):  # noqa: ANN201
        raise AssertionError("no proposal is decided here")

    def static_sentence(self) -> str:
        return "Actions run only through the action system."

    def redact(self, text: str) -> str:
        return text

    def record_turn(self, session_id: str, used: Any) -> None:
        return None

    def drain_events(self, session_id: str) -> list[dict[str, object]]:
        return []


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_a_handled_capability_turn_sends_no_job_and_a_model_turn_does(self, tmp_path):
        from backend.chat.conversation_handler import ConversationHandler
        from backend.knowledge.retrieval.knowledge_retriever import KnowledgeRetriever
        from backend.knowledge.storage.graph_store import GraphStore
        from tests.mocks.config import build_test_config
        from tests.mocks.neo4j import FakeNeo4jConnection
        from tests.mocks.streaming_llm import FakeStreamingLLMProvider, ScriptedPass
        from tests.unit.conftest import make_test_conventions_loader

        world = _tmp_world(tmp_path)
        config = build_test_config()
        graph_store = GraphStore(FakeNeo4jConnection(), FakeEmbeddingGenerator())
        handler = ConversationHandler(
            config=config,
            graph_store=graph_store,
            extraction_pipeline=world.build_pipeline(),
            retriever=KnowledgeRetriever(config=config, graph_store=graph_store),
            llm_provider=FakeStreamingLLMProvider([ScriptedPass(chunks=["Rust it is."])]),
            conventions_loader=make_test_conventions_loader(),
        )
        handler.event_store = world.event_store
        dispatcher = world.build_dispatcher()
        handler.attach_extraction_dispatcher(dispatcher)
        handler.attach_capability_host(_CapabilityPort())  # type: ignore[arg-type]
        try:
            async for _event in handler.handle_message_streaming(
                user_message=_CAPABILITY_UTTERANCE, session_id="e2e"
            ):
                pass
            await _settle(dispatcher)

            epoch = world.store.active_epoch()
            turns = world.event_store.get_turns("e2e")
            assert [t["answered_by"] for t in turns] == ["capability"]
            capability = turns[0]["event_id"]
            assert world.service.received == []
            assert world.store.get_cached(capability, epoch)["skip_reason"] == SKIP_CAPABILITY_ONLY
            assert _applied(world) == {capability: "applied"}
            status = dispatcher.snapshot()
            assert status.backlog_depth == 0
            assert status.policy_skipped == {SKIP_CAPABILITY_ONLY: 1}

            async for _event in handler.handle_message_streaming(
                user_message="I really use rust", session_id="e2e"
            ):
                pass
            await _settle(dispatcher)

            turns = world.event_store.get_turns("e2e")
            assert [t["answered_by"] for t in turns] == ["capability", "model"]
            assert world.service.received_utterances == ["I really use rust"]
            assert world.curation.event_ids == [turns[1]["event_id"]]
            assert dispatcher.snapshot().policy_skipped == {SKIP_CAPABILITY_ONLY: 1}
        finally:
            await dispatcher.stop(timeout=2.0)
            for client in world.clients:
                await client.aclose()
            await asyncio.sleep(0)
