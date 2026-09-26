"""T2b part C: epoch cutover -- fill, check (rebuild wiring), promote, abandon.

The fill and promote tests drive the real dispatcher against the fake
extraction service (`fakes.py`); the check tests fake `LogRegenerator` and
`rebuild_gate` (a real run needs Neo4j and is the lead's to do).
"""

from __future__ import annotations

import asyncio
import io
import json
from types import SimpleNamespace

import pytest

from backend.extraction_backlog import admin
from backend.extraction_backlog.cutover import RebuildDeps
from backend.knowledge.extraction_cache import OUTCOME_EXTRACTED
from backend.knowledge.regeneration.rebuild_gate import (
    RebuildDeterminismError,
    RebuildVacuityError,
)
from tests.unit.extraction_backlog.conftest import (
    EMBEDDING_MODEL,
    ONTOLOGY_VERSION,
    composed,
    wait_until,
)

CANDIDATE = "svc-model-2"
STAGING_URI = "bolt://localhost:7689"
LIVE_URI = "bolt://mist-neo4j:7687"


@pytest.fixture
def sync_world():
    """A backlog world for synchronous tests (no dispatcher is started)."""
    from tests.unit.extraction_backlog.conftest import _build_world

    return _build_world(with_deriver=False)


def _cli(world, *argv: str) -> tuple[int, str]:
    out = io.StringIO()
    code = admin.main(list(argv), store=world.store, out=out, embedding_model_name=EMBEDDING_MODEL)
    return code, out.getvalue()


def _begin(world) -> tuple[int, str]:
    return _cli(
        world,
        "cutover",
        "begin",
        "--model-hash",
        CANDIDATE,
        "--extraction-version",
        world.service.extraction_version,
        "--ontology-version",
        ONTOLOGY_VERSION,
    )


def _log(world, ts, index: int, minute: int | None = None) -> str:
    return world.log_turn(
        session_id="s1",
        turn_index=index,
        timestamp=ts(index if minute is None else minute),
        utterance=f"I really use tool{index}",
    )


async def _two_applied_then_two_logged(world, ts) -> list[str]:
    """Four logged turns; the first two applied under epoch 1 by a real dispatcher."""
    ids = [_log(world, ts, 0), _log(world, ts, 1)]
    dispatcher = world.build_dispatcher()
    await dispatcher.start()
    assert await dispatcher.drain(timeout=5.0)
    await dispatcher.stop(timeout=2.0)
    ids += [_log(world, ts, 2), _log(world, ts, 3)]
    assert world.curation.event_ids == ids[:2]
    return ids


async def _filled(world, ts) -> list[str]:
    """Four turns (two applied), a cutover begun and filled to 'ready', dispatcher stopped."""
    ids = await _two_applied_then_two_logged(world, ts)
    assert _begin(world)[0] == 0
    world.service.model_hash = CANDIDATE
    dispatcher = world.build_dispatcher()
    await dispatcher.start()
    await wait_until(lambda: world.store.open_cutover().state == "ready")
    await dispatcher.stop(timeout=2.0)
    return ids


def _mark_checked(world, through: str) -> None:
    store = world.store
    cutover = store.open_cutover()
    assert store.transition_cutover(
        cutover,
        from_states=("ready",),
        to_state="checked",
        updated_at="2026-09-02T00:00:00+00:00",
        rebuild_job_id="rebuild-job",
        rebuilt_through_event_id=through,
        check_report={"passed": True},
        write_check=True,
    )


def _applied_rows(world, epoch_id: int) -> dict[str, str]:
    rows = (
        world.event_store._get_connection()
        .execute(
            "SELECT event_id, source FROM extraction_applied WHERE epoch_id = ? AND stage = ?",
            (epoch_id, "applied"),
        )
        .fetchall()
    )
    return {row[0]: row[1] for row in rows}


class TestBegin:
    def test_begin_records_a_candidate_not_a_ledger_row(self, sync_world, ts):
        world = sync_world
        _log(world, ts, 0)

        code, text = _begin(world)

        assert code == 0
        cutover = world.store.open_cutover()
        assert cutover.state == "filling"
        assert cutover.model_hash == composed(CANDIDATE)
        assert cutover.bare_model_hash == CANDIDATE
        assert cutover.source_epoch_id == world.epoch_id
        assert len(world.event_store.list_epochs()) == 1
        assert world.store.active_epoch().epoch_id == world.epoch_id
        assert "covered=0 total=1" in text

    def test_a_second_begin_is_refused(self, sync_world):
        world = sync_world
        assert _begin(world)[0] == 0

        code, text = _begin(world)

        assert code == 2
        assert "already open" in text

    def test_the_active_epochs_own_stamps_are_refused(self, sync_world):
        world = sync_world

        code, text = _cli(
            world,
            "cutover",
            "begin",
            "--model-hash",
            world.service.model_hash,
            "--extraction-version",
            world.service.extraction_version,
            "--ontology-version",
            ONTOLOGY_VERSION,
        )

        assert code == 2
        assert "nothing to cut over" in text
        assert world.store.open_cutover() is None

    def test_abandon_closes_it_and_deletes_nothing(self, sync_world, ts):
        world = sync_world
        event_id = _log(world, ts, 0)
        assert _begin(world)[0] == 0
        world.cache.put(
            event_id,
            ONTOLOGY_VERSION,
            world.service.extraction_version,
            composed(CANDIDATE),
            outcome=OUTCOME_EXTRACTED,
            entities=[],
            created_at=ts(0),
        )

        code, _text = _cli(world, "cutover", "abandon")

        assert code == 0
        assert world.store.open_cutover() is None
        assert world.store.list_cutovers()[-1].state == "abandoned"
        assert (
            world.cache.get(event_id, world.service.extraction_version, composed(CANDIDATE))
            is not None
        )
        assert _cli(world, "cutover", "abandon")[0] == 2


class TestFill:
    @pytest.mark.asyncio
    async def test_the_candidate_service_gets_every_turn_in_log_order_and_live_gets_nothing(
        self, backlog_world, ts
    ):
        # Arrange: 4 logged turns, 2 applied under epoch 1.
        world = backlog_world
        ids = await _two_applied_then_two_logged(world, ts)
        assert _begin(world)[0] == 0
        world.service.model_hash = CANDIDATE
        world.service.received.clear()
        curation_before = list(world.curation.event_ids)
        dispatcher = world.build_dispatcher()

        # Act
        await dispatcher.start()
        await wait_until(lambda: world.store.open_cutover().state == "ready")

        # Assert
        assert [r.event_id for r in world.service.received] == ids
        assert {r.expect.model_hash for r in world.service.received} == {CANDIDATE}
        assert world.curation.event_ids == curation_before  # zero live curation calls
        assert world.main_llm.calls == []
        cutover = world.store.open_cutover()
        for event_id in ids:
            row = world.cache.get(event_id, cutover.extraction_version, cutover.model_hash)
            assert row is not None and row["outcome"] == OUTCOME_EXTRACTED
        fill_attempts = [a for a in world.attempts() if a["epoch_id"] == -cutover.cutover_id]
        assert [a["event_id"] for a in fill_attempts] == ids
        status = dispatcher.snapshot()
        assert status.cutover is not None
        assert (status.cutover.state, status.cutover.covered, status.cutover.total) == (
            "ready",
            4,
            4,
        )
        assert status.cutover.target_model_hash == composed(CANDIDATE)
        # The active epoch cannot progress: turns 2 and 3 stay pending under it.
        assert status.backlog_depth == 2
        assert await dispatcher.drain(timeout=0.2) is False

    @pytest.mark.asyncio
    async def test_state_is_working_while_filling_and_a_mid_fill_turn_is_inferred(
        self, backlog_world, ts
    ):
        world = backlog_world
        ids = await _two_applied_then_two_logged(world, ts)
        assert _begin(world)[0] == 0
        world.service.model_hash = CANDIDATE
        world.service.received.clear()
        world.service.hold = asyncio.Event()
        dispatcher = world.build_dispatcher()

        await dispatcher.start()
        await wait_until(lambda: len(world.service.received) == 1)
        in_flight = dispatcher.snapshot()
        late = _log(world, ts, 4, minute=9)
        world.service.hold.set()
        await wait_until(lambda: world.store.open_cutover().state == "ready")

        assert in_flight.state == "working"
        assert in_flight.cutover.state == "filling"
        assert (in_flight.cutover.covered, in_flight.cutover.total) == (0, 4)
        assert [r.event_id for r in world.service.received] == ids + [late]
        assert len(world.curation.calls) == 2

    @pytest.mark.asyncio
    async def test_filling_continues_after_ready(self, backlog_world, ts):
        world = backlog_world
        await _filled(world, ts)
        world.service.received.clear()
        dispatcher = world.build_dispatcher()
        await dispatcher.start()

        later = _log(world, ts, 4, minute=9)
        dispatcher.wake()
        await wait_until(lambda: len(world.service.received) == 1)
        await wait_until(lambda: world.store.fill_scan(world.store.open_cutover()).head is None)

        assert world.service.received[0].event_id == later
        assert world.store.open_cutover().state == "ready"
        assert len(world.curation.calls) == 2

    @pytest.mark.asyncio
    async def test_a_service_still_on_the_old_model_is_epoch_mismatch_and_gets_no_jobs(
        self, backlog_world, ts
    ):
        world = backlog_world
        await _two_applied_then_two_logged(world, ts)
        assert _begin(world)[0] == 0
        world.service.received.clear()
        dispatcher = world.build_dispatcher()

        await dispatcher.start()
        await wait_until(lambda: dispatcher.state == "epoch_mismatch")

        assert world.service.received == []
        assert world.store.open_cutover().state == "filling"
        assert dispatcher.snapshot().cutover.covered == 0

    @pytest.mark.asyncio
    async def test_the_ready_transition_is_pushed_to_state_listeners(self, backlog_world, ts):
        world = backlog_world
        await _two_applied_then_two_logged(world, ts)
        assert _begin(world)[0] == 0
        world.service.model_hash = CANDIDATE
        dispatcher = world.build_dispatcher()
        seen: list[str | None] = []
        dispatcher.add_state_listener(
            lambda _p, _c: seen.append(
                None
                if dispatcher.snapshot().cutover is None
                else dispatcher.snapshot().cutover.state
            )
        )

        await dispatcher.start()
        await wait_until(lambda: world.store.open_cutover().state == "ready")
        await wait_until(lambda: "ready" in seen)

        assert "filling" in seen

    @pytest.mark.asyncio
    async def test_a_checked_cutover_reports_ready_on_the_wire(self, backlog_world, ts):
        world = backlog_world
        ids = await _filled(world, ts)
        _mark_checked(world, ids[-1])

        status = world.build_dispatcher().snapshot()

        assert status.cutover.state == "ready"
        assert world.store.open_cutover().state == "checked"


class TestPromote:
    @pytest.mark.asyncio
    async def test_promotion_is_refused_unless_checked(self, backlog_world, ts):
        world = backlog_world
        await _filled(world, ts)

        code, text = _cli(world, "cutover", "promote", "--graph-swapped")

        assert code == 2
        assert "'ready'" in text
        assert len(world.event_store.list_epochs()) == 1

    @pytest.mark.asyncio
    async def test_promotion_is_refused_without_graph_swapped(self, backlog_world, ts):
        world = backlog_world
        ids = await _filled(world, ts)
        _mark_checked(world, ids[2])

        code, text = _cli(world, "cutover", "promote")

        assert code == 2
        assert "--graph-swapped is required" in text
        assert len(world.event_store.list_epochs()) == 1
        assert world.store.open_cutover().state == "checked"

    @pytest.mark.asyncio
    async def test_promotion_marks_exactly_through_the_rebuilt_turn_and_the_rest_apply_after(
        self, backlog_world, ts
    ):
        # Arrange: filled, checked through the THIRD of four turns.
        world = backlog_world
        ids = await _filled(world, ts)
        _mark_checked(world, ids[2])
        received_before = len(world.service.received)
        curation_before = list(world.curation.event_ids)

        # Act
        code, text = _cli(world, "cutover", "promote", "--graph-swapped")

        # Assert: one transaction appended the epoch and wrote its activation.
        assert code == 0, text
        epochs = world.event_store.list_epochs()
        assert len(epochs) == 2
        new_epoch = epochs[-1]
        assert new_epoch["model_hash"] == composed(CANDIDATE)
        assert new_epoch["prev_epoch_id"] == world.epoch_id
        activation = world.event_store.get_extraction_activation(new_epoch["epoch_id"])
        assert (
            activation["turns_at_activation"],
            activation["marked_applied"],
            activation["legacy_unextracted"],
        ) == (4, 3, 0)
        assert _applied_rows(world, new_epoch["epoch_id"]) == {
            event_id: "cutover" for event_id in ids[:3]
        }
        promoted = world.store.list_cutovers()[-1]
        assert (promoted.state, promoted.promoted_epoch_id) == ("promoted", new_epoch["epoch_id"])
        assert f"MIST_MODEL_HASH={CANDIDATE}" in text

        # Act 2: the dispatcher applies the fourth turn under the new epoch.
        dispatcher = world.build_dispatcher()
        await dispatcher.start()
        assert await dispatcher.drain(timeout=5.0)

        assert world.curation.event_ids == curation_before + [ids[3]]
        assert len(world.service.received) == received_before  # applied from the cache
        assert _applied_rows(world, new_epoch["epoch_id"])[ids[3]] == "dispatcher"
        assert world.store.get_activation(world.store.active_epoch()).marked_applied == 3
        assert dispatcher.snapshot().cutover is None


class _FakeRegenerator:
    def __init__(self, world, calls: list, *, on_rebuild=None) -> None:
        self._world = world
        self._calls = calls
        self._on_rebuild = on_rebuild

    async def rebuild(self, **kwargs):
        self._calls.append(kwargs)
        if self._on_rebuild is not None:
            self._on_rebuild(len(self._calls))
        epoch = kwargs["epoch"]
        turns = self._world.event_store.get_all_turns_for_reextraction(
            ontology_version=epoch["ontology_version"], origins=("real",)
        )
        return SimpleNamespace(
            job_id=f"rebuild-job-{len(self._calls)}",
            turns_processed=len(turns),
            total_logged=self._world.event_store.get_turn_count(),
            ontology_version=epoch["ontology_version"],
            origins=("real",),
            seed_nodes_written=5,
        )


def _fake_gates(**raises) -> tuple[SimpleNamespace, list[str]]:
    called: list[str] = []

    def gate(name):
        def check(*_args, **_kwargs):
            called.append(name)
            if name in raises:
                raise raises[name]

        return check

    names = (
        "assert_rebuild_twice_identical",
        "assert_turns_processed",
        "assert_canonical_form_non_vacuous",
        "assert_replay_derived_non_vacuous",
        "assert_self_model_applied",
    )
    gates = SimpleNamespace(**{name: gate(name) for name in names})
    gates.live_vs_rebuilt_report = lambda live, rebuilt: "live != rebuilt: 7 differing lines\n..."
    return gates, called


def _fill_directly(world, ts, n: int = 4) -> list[str]:
    """Log `n` turns, begin a cutover, write candidate rows, and set state 'ready'."""
    ids = [_log(world, ts, i) for i in range(n)]
    assert _begin(world)[0] == 0
    for event_id in ids:
        world.cache.put(
            event_id,
            ONTOLOGY_VERSION,
            world.service.extraction_version,
            composed(CANDIDATE),
            outcome=OUTCOME_EXTRACTED,
            entities=[{"id": "x", "type": "Technology", "name": "X"}],
            created_at=ts(0),
        )
    store = world.store
    assert store.transition_cutover(
        store.open_cutover(), from_states=("filling",), to_state="ready", updated_at=ts(10)
    )
    return ids


class _Harness:
    """Fake RebuildDeps plus a record of what the check did with them."""

    def __init__(self, world, *, live_uri: str = LIVE_URI, on_rebuild=None) -> None:
        self.calls: list[dict] = []
        self.wipes = 0
        self.closed = False
        self.factory_calls = 0
        self._world = world
        self._live_uri = live_uri
        self._on_rebuild = on_rebuild

    def factory(self, _cutover) -> RebuildDeps:
        self.factory_calls += 1

        def wipe():
            self.wipes += 1

        def close():
            self.closed = True

        return RebuildDeps(
            live_uri=self._live_uri,
            wipe_staging=wipe,
            build_regenerator=lambda: _FakeRegenerator(
                self._world, self.calls, on_rebuild=self._on_rebuild
            ),
            staging_form=lambda: '{"nodes": [], "relationships": []}',
            live_form=lambda: '{"nodes": [], "relationships": [], "self_model": {"nodes": []}}',
            close=close,
        )


def _run_check(world, harness, gates, *, staging_uri=STAGING_URI, expect_turns=4):
    from backend.extraction_backlog.cutover import check_cutover

    return asyncio.run(
        check_cutover(
            world.store,
            harness.factory,
            staging_uri=staging_uri,
            min_seed_nodes=1,
            expect_turns=expect_turns,
            min_replay_edges=1,
            now_iso="2026-09-02T00:00:00+00:00",
            gates=gates,
        )
    )


class TestRebuildCheck:
    def test_a_passing_check_records_checked_and_the_last_replayed_turn(self, sync_world, ts):
        world = sync_world
        ids = _fill_directly(world, ts)
        harness = _Harness(world)
        gates, called = _fake_gates()

        result = _run_check(world, harness, gates)

        assert result.exit_code == 0
        cutover = world.store.open_cutover()
        assert cutover.state == "checked"
        assert cutover.rebuilt_through_event_id == ids[-1]
        assert cutover.rebuild_job_id == "rebuild-job-2"
        assert cutover.check_report["passed"] is True
        assert cutover.check_report["live_vs_rebuilt"] == "live != rebuilt: 7 differing lines"
        # Built twice, each over a wiped staging graph, under the CANDIDATE epoch.
        assert harness.wipes == 2 and harness.closed
        assert len(harness.calls) == 2
        epoch = harness.calls[0]["epoch"]
        assert epoch["model_hash"] == composed(CANDIDATE)
        assert epoch["epoch_id"] == -cutover.cutover_id
        assert epoch["activated_at"] == cutover.requested_at
        assert harness.calls[0]["live_uri"] == LIVE_URI
        assert harness.calls[0]["staging_uri"] == STAGING_URI
        assert called.count("assert_rebuild_twice_identical") == 1
        assert called.count("assert_turns_processed") == 2
        assert called.count("assert_replay_derived_non_vacuous") == 2
        assert "assert_self_model_applied" in called

    def test_the_admin_command_wires_the_check(self, sync_world, ts):
        world = sync_world
        _fill_directly(world, ts)
        harness = _Harness(world)
        out = io.StringIO()

        code = admin.main(
            [
                "cutover",
                "rebuild",
                "--staging-uri",
                STAGING_URI,
                "--min-seed-nodes",
                "1",
                "--expect-turns",
                "4",
                "--min-replay-edges",
                "1",
            ],
            store=world.store,
            out=out,
            rebuild_deps_factory=harness.factory,
        )

        # The real rebuild_gate runs here, over two empty forms: vacuity fails.
        assert code == 4
        assert world.store.open_cutover().state == "ready"
        report = world.store.open_cutover().check_report
        assert report["passed"] is False
        assert "non-vacuity" in report["failure"]
        assert json.loads(out.getvalue().rsplit("[cutover]", 1)[0])["exit_code"] == 4

    @pytest.mark.parametrize(
        ("gate", "error", "exit_code"),
        [
            ("assert_rebuild_twice_identical", RebuildDeterminismError("two rebuilds differ"), 1),
            ("assert_replay_derived_non_vacuous", RebuildVacuityError("0 replay edges"), 4),
            ("assert_self_model_applied", RebuildVacuityError("self-model gate FAILED"), 4),
        ],
    )
    def test_a_gate_failure_leaves_ready_with_the_report(
        self, sync_world, ts, gate, error, exit_code
    ):
        world = sync_world
        _fill_directly(world, ts)
        gates, _called = _fake_gates(**{gate: error})

        result = _run_check(world, _Harness(world), gates)

        assert result.exit_code == exit_code
        cutover = world.store.open_cutover()
        assert cutover.state == "ready"
        assert cutover.rebuilt_through_event_id is None
        assert cutover.check_report["passed"] is False
        assert str(error) in cutover.check_report["failure"]

    def test_a_failed_recheck_demotes_a_checked_cutover(self, sync_world, ts):
        world = sync_world
        ids = _fill_directly(world, ts)
        _mark_checked(world, ids[-1])
        gates, _called = _fake_gates(assert_rebuild_twice_identical=RebuildDeterminismError("x"))

        result = _run_check(world, _Harness(world), gates)

        assert result.exit_code == 1
        cutover = world.store.open_cutover()
        assert (cutover.state, cutover.rebuilt_through_event_id) == ("ready", None)
        assert _cli(world, "cutover", "promote", "--graph-swapped")[0] == 2

    def test_a_filling_cutover_is_refused_before_any_connection(self, sync_world, ts):
        world = sync_world
        _log(world, ts, 0)
        assert _begin(world)[0] == 0
        harness = _Harness(world)
        gates, called = _fake_gates()

        result = _run_check(world, harness, gates, expect_turns=1)

        assert result.exit_code == 2
        assert harness.factory_calls == 0
        assert called == []
        assert world.store.open_cutover().state == "filling"

    def test_a_live_staging_uri_is_refused_before_any_write(self, sync_world, ts):
        world = sync_world
        _fill_directly(world, ts)
        harness = _Harness(world)
        gates, called = _fake_gates()

        result = _run_check(world, harness, gates, staging_uri=LIVE_URI)

        assert result.exit_code == 2
        assert harness.wipes == 0
        assert harness.calls == []
        assert "RebuildTargetError" in result.report["failure"]
        assert world.store.open_cutover().state == "ready"

    def test_a_turn_logged_into_scope_mid_check_is_refused(self, sync_world, ts):
        world = sync_world
        _fill_directly(world, ts)

        def log_during_first_build(call_number: int) -> None:
            if call_number == 1:
                _log(world, ts, 7, minute=20)

        harness = _Harness(world, on_rebuild=log_during_first_build)
        gates, _called = _fake_gates()

        result = _run_check(world, harness, gates)

        assert result.exit_code == 2
        assert "selection changed" in result.report["failure"]
        assert world.store.open_cutover().state == "ready"
