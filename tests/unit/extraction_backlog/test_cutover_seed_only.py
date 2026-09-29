"""`cutover promote --seed-only-graph` and `cutover probe`: the seed-only path.

Seed-only promotion needs an EMPTY conversation log (any session origin), so
every promotion here is over an empty log, and a logged turn is a refusal: before
the probe runs, and again inside the promotion transaction when the turn is
logged after that first check.

Every refusal is asserted write-free: the ledger, the activations, the apply
markers, the legacy list and the whole `epoch_cutover` row are compared before
and after. The live-graph probe is always a fake (`FakeProbe`), or the real
probe over a fake `Neo4jConnection`; the unit tier never connects to Neo4j. The
probe's Cypher is checked textually for write clauses, and `probe_graph` is run
over a recording fake connection.
"""

from __future__ import annotations

import io
import json
import re
from datetime import datetime

import pytest

from backend.errors import Neo4jConnectionError, Neo4jQueryError
from backend.event_store.store import EpochCutoverStateError
from backend.extraction_backlog import admin
from backend.extraction_backlog.cutover import (
    SEED_ONLY_PROBE_CYPHER,
    GraphProbeError,
    GraphProbeReport,
    log_not_empty_reason,
    probe_graph,
)
from backend.knowledge.extraction_cache import OUTCOME_EXTRACTED
from tests.unit.extraction_backlog.conftest import (
    EMBEDDING_MODEL,
    ONTOLOGY_VERSION,
    composed,
    wait_until,
)

CANDIDATE = "svc-model-2"
NOW = "2026-09-03T00:00:00+00:00"

SEED_ONLY = GraphProbeReport(
    node_count=32,
    nodes_without_seed_version=0,
    nodes_with_extraction_stamp=0,
    relationship_count=30,
    relationships_without_seed_version=0,
    relationships_with_extraction_stamp=0,
)


class FakeProbe:
    """A `GraphProbe` that records its calls and can act on the world mid-probe.

    `during` runs inside the probe call, i.e. after the pre-checks and before
    the promotion transaction: the window a concurrent writer could use.
    `snapshot_after` is the world as the probe left it, so a race test can
    assert the transaction then wrote nothing on top.
    """

    def __init__(self, world, result=SEED_ONLY, *, raises=None, during=None) -> None:
        self.calls = 0
        self.snapshot_after: dict | None = None
        self._world = world
        self._result = result
        self._raises = raises
        self._during = during

    def __call__(self) -> GraphProbeReport:
        self.calls += 1
        if self._during is not None:
            self._during(self._world)
        self.snapshot_after = _snapshot(self._world)
        if self._raises is not None:
            raise self._raises
        return self._result


@pytest.fixture
def sync_world():
    """A backlog world for synchronous tests (no dispatcher is started)."""
    from tests.unit.extraction_backlog.conftest import _build_world

    return _build_world(with_deriver=False)


def _cli(world, *argv: str, probe=None) -> tuple[int, str]:
    out = io.StringIO()
    code = admin.main(
        list(argv),
        store=world.store,
        out=out,
        embedding_model_name=EMBEDDING_MODEL,
        graph_probe=probe,
        clock=lambda: datetime.fromisoformat(NOW),
    )
    return code, out.getvalue()


def _promote(world, probe) -> tuple[int, str]:
    return _cli(world, "cutover", "promote", "--seed-only-graph", probe=probe)


def _begin(world) -> None:
    code, text = _cli(
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
    assert code == 0, text


def _log(world, ts, index: int, *, session_id: str = "s1") -> str:
    return world.log_turn(
        session_id=session_id,
        turn_index=index,
        timestamp=ts(index),
        utterance=f"I really use tool{index}",
    )


def _cache_candidate(world, ts, event_id: str) -> None:
    world.cache.put(
        event_id,
        ONTOLOGY_VERSION,
        world.service.extraction_version,
        composed(CANDIDATE),
        outcome=OUTCOME_EXTRACTED,
        entities=[{"id": "x", "type": "Technology", "name": "X"}],
        created_at=ts(0),
    )


def _ready(world, ts, n: int = 0) -> list[str]:
    """Log `n` turns (none applied), begin, cache every turn under the candidate, set 'ready'.

    `n` = 0, the default, is the only log seed-only promotion accepts.
    """
    ids = [_log(world, ts, i) for i in range(n)]
    _begin(world)
    for event_id in ids:
        _cache_candidate(world, ts, event_id)
    store = world.store
    assert store.transition_cutover(
        store.open_cutover(), from_states=("filling",), to_state="ready", updated_at=ts(10)
    )
    return ids


def _snapshot(world) -> dict[str, list[tuple]]:
    """Every row a promotion could write, in a stable order, every column, plus the log."""
    conn = world.event_store._get_connection()
    queries = {
        "ledger": "SELECT * FROM epoch_ledger ORDER BY epoch_id",
        "activation": "SELECT * FROM extraction_activation ORDER BY epoch_id",
        "applied": "SELECT * FROM extraction_applied ORDER BY epoch_id, event_id",
        "legacy": "SELECT * FROM extraction_legacy_turns ORDER BY epoch_id, event_id",
        "cutover": "SELECT * FROM epoch_cutover ORDER BY cutover_id",
        "turns": "SELECT * FROM conversation_turn_events ORDER BY event_id",
        "sessions": "SELECT * FROM conversation_sessions ORDER BY session_id",
    }
    return {name: [tuple(row) for row in conn.execute(sql)] for name, sql in queries.items()}


def _assert_refused(code: int, text: str, reason: str) -> None:
    assert code == 2, text
    assert "[cutover] REFUSED: " in text
    assert reason in text, text
    assert "promoted cutover" not in text


# ---------------------------------------------------------------------------
# Refusals, each write-free
# ---------------------------------------------------------------------------


class TestRefusals:
    def test_a_no_open_cutover(self, sync_world):
        world = sync_world
        probe = FakeProbe(world)
        before = _snapshot(world)

        code, text = _promote(world, probe)

        _assert_refused(code, text, "no cutover is open")
        assert _snapshot(world) == before
        assert probe.calls == 0

    def test_a_a_filling_cutover(self, sync_world, ts):
        world = sync_world
        _log(world, ts, 0)
        _begin(world)
        probe = FakeProbe(world)
        before = _snapshot(world)

        code, text = _promote(world, probe)

        _assert_refused(code, text, "is 'filling'")
        assert _snapshot(world) == before
        assert probe.calls == 0

    def test_b_an_incomplete_fill(self, sync_world, ts):
        # 'ready', then a turn logged later is not yet covered.
        world = sync_world
        _ready(world, ts, 3)
        _log(world, ts, 5)
        probe = FakeProbe(world)
        before = _snapshot(world)

        code, text = _promote(world, probe)

        _assert_refused(code, text, "covers 3 of 4 logged turns")
        assert _snapshot(world) == before
        assert probe.calls == 0

    @pytest.mark.parametrize("stage", ["applied", "curated"])
    def test_c_an_apply_marker_under_the_source_epoch(self, sync_world, ts, stage):
        world = sync_world
        ids = _ready(world, ts, 3)
        world.event_store.mark_extraction_stage(
            event_id=ids[0], epoch_id=world.epoch_id, stage=stage, updated_at=ts(1)
        )
        probe = FakeProbe(world)
        before = _snapshot(world)

        code, text = _promote(world, probe)

        _assert_refused(code, text, f"extraction_applied holds apply markers ({stage}=1)")
        assert _snapshot(world) == before
        assert probe.calls == 0

    def test_c_an_apply_marker_under_an_epoch_other_than_the_source(self, sync_world, ts):
        # The source epoch is 2; the marker is epoch 1's, from before it.
        world = sync_world
        ids = [_log(world, ts, i) for i in range(2)]
        world.event_store.mark_extraction_stage(
            event_id=ids[0], epoch_id=world.epoch_id, stage="applied", updated_at=ts(1)
        )
        world.event_store.append_epoch(
            ONTOLOGY_VERSION, world.service.extraction_version, "older-model|x", activated_at=ts(2)
        )
        _begin(world)
        for event_id in ids:
            _cache_candidate(world, ts, event_id)
        store = world.store
        cutover = store.open_cutover()
        assert cutover.source_epoch_id != world.epoch_id
        assert store.transition_cutover(
            cutover, from_states=("filling",), to_state="ready", updated_at=ts(10)
        )
        probe = FakeProbe(world)
        before = _snapshot(world)

        code, text = _promote(world, probe)

        _assert_refused(code, text, "extraction_applied holds apply markers (applied=1)")
        assert _snapshot(world) == before
        assert probe.calls == 0

    def test_c_a_marker_written_during_the_probe_is_caught_inside_the_transaction(
        self, sync_world, ts
    ):
        # The log is empty, so the marker names a turn that is not in it:
        # `extraction_applied` has no foreign key, and any row refuses.
        world = sync_world
        _ready(world, ts)
        probe = FakeProbe(
            world,
            during=lambda w: w.event_store.mark_extraction_stage(
                event_id="evt-not-logged", epoch_id=w.epoch_id, stage="curated", updated_at=ts(1)
            ),
        )

        code, text = _promote(world, probe)

        _assert_refused(code, text, "extraction_applied holds apply markers (curated=1)")
        assert probe.calls == 1
        assert _snapshot(world) == probe.snapshot_after

    def test_d_the_active_epoch_moved_since_begin(self, sync_world, ts):
        world = sync_world
        _ready(world, ts, 2)
        world.event_store.append_epoch(
            ONTOLOGY_VERSION, world.service.extraction_version, "other|x", activated_at=ts(11)
        )
        probe = FakeProbe(world)
        before = _snapshot(world)

        code, text = _promote(world, probe)

        _assert_refused(code, text, f"began from epoch {world.epoch_id}")
        assert _snapshot(world) == before
        assert probe.calls == 0

    def test_d_a_ledger_move_during_the_probe_is_caught_inside_the_transaction(
        self, sync_world, ts
    ):
        world = sync_world
        _ready(world, ts)
        probe = FakeProbe(
            world,
            during=lambda w: w.event_store.append_epoch(
                ONTOLOGY_VERSION, w.service.extraction_version, "other|x", activated_at=ts(11)
            ),
        )

        code, text = _promote(world, probe)

        _assert_refused(code, text, f"began from epoch {world.epoch_id}")
        assert probe.calls == 1
        assert _snapshot(world) == probe.snapshot_after

    @pytest.mark.parametrize("origin", ["real", "test", "seed"])
    def test_e_a_logged_turn_refuses_before_the_probe_runs(self, sync_world, ts, origin):
        # Every turn is cached under the candidate, so a to d all pass; only
        # the log refuses. Any session origin counts.
        world = sync_world
        world.event_store.start_session("s-origin", input_modality="text", origin=origin)
        ids = [_log(world, ts, i, session_id="s-origin") for i in range(3)]
        _begin(world)
        for event_id in ids:
            _cache_candidate(world, ts, event_id)
        store = world.store
        assert store.transition_cutover(
            store.open_cutover(), from_states=("filling",), to_state="ready", updated_at=ts(10)
        )
        assert store.fill_scan(store.open_cutover()).head is None
        probe = FakeProbe(world)
        before = _snapshot(world)

        code, text = _promote(world, probe)

        _assert_refused(code, text, "the conversation log holds 3 logged turn(s)")
        assert "use the graph-swapped path" in text
        assert probe.calls == 0
        assert _snapshot(world) == before

    def test_e_a_turn_logged_during_the_probe_is_caught_inside_the_transaction(
        self, sync_world, ts
    ):
        world = sync_world
        _ready(world, ts)
        probe = FakeProbe(world, during=lambda w: _log(w, ts, 0))

        code, text = _promote(world, probe)

        _assert_refused(code, text, "the conversation log holds 1 logged turn(s)")
        assert probe.calls == 1
        # The transaction wrote nothing on top of the logged turn.
        assert _snapshot(world) == probe.snapshot_after
        assert len(_snapshot(world)["ledger"]) == 1
        assert world.store.open_cutover().state == "ready"

    def test_e_a_turn_logged_after_the_probe_is_caught_inside_the_transaction(
        self, sync_world, ts, monkeypatch
    ):
        # A store hook: the turn lands after the probe returns and immediately
        # before the promotion transaction opens, the last window a concurrent
        # writer has. Only the in-transaction re-check can see it.
        world = sync_world
        _ready(world, ts)
        event_store = world.event_store
        real_promote = event_store.promote_epoch_cutover_seed_only
        seen: dict = {}

        def log_then_promote(**kwargs):
            _log(world, ts, 0)
            seen["before_txn"] = _snapshot(world)
            return real_promote(**kwargs)

        monkeypatch.setattr(event_store, "promote_epoch_cutover_seed_only", log_then_promote)
        probe = FakeProbe(world)

        code, text = _promote(world, probe)

        _assert_refused(code, text, "the conversation log holds 1 logged turn(s)")
        assert probe.calls == 1
        assert _snapshot(world) == seen["before_txn"]
        assert len(seen["before_txn"]["ledger"]) == 1
        assert world.store.open_cutover().state == "ready"

    def test_e_the_pre_check_and_the_transaction_word_the_refusal_alike(self, sync_world, ts):
        world = sync_world
        _ready(world, ts, 1)

        with pytest.raises(EpochCutoverStateError) as exc:
            world.event_store.promote_epoch_cutover_seed_only(
                cutover_id=world.store.open_cutover().cutover_id,
                activated_at=NOW,
                probe_report=SEED_ONLY.as_dict(),
            )

        assert str(exc.value) == log_not_empty_reason(1)
        assert log_not_empty_reason(0) is None

    @pytest.mark.parametrize(
        ("field", "value", "reason"),
        [
            ("node_count", 0, "the live graph has no nodes"),
            ("nodes_without_seed_version", 1, "1 node(s) without seed_version"),
            ("nodes_with_extraction_stamp", 2, "2 node(s) carrying ontology_version"),
            ("relationships_without_seed_version", 3, "3 relationship(s) without seed_version"),
            ("relationships_with_extraction_stamp", 1, "1 relationship(s) carrying"),
        ],
    )
    def test_f_a_graph_that_is_not_seed_only(self, sync_world, ts, field, value, reason):
        world = sync_world
        _ready(world, ts)
        counts = SEED_ONLY.as_dict() | {field: value}
        probe = FakeProbe(world, GraphProbeReport(**counts))
        before = _snapshot(world)

        code, text = _promote(world, probe)

        _assert_refused(code, text, "the live graph is not seed-only: " + reason)
        assert probe.calls == 1
        assert _snapshot(world) == before

    @pytest.mark.parametrize(
        "error",
        [
            Neo4jConnectionError("Failed to connect to Neo4j: refused"),
            Neo4jQueryError("Query execution failed: boom"),
            GraphProbeError("the seed-only probe returned 0 rows, expected 1"),
        ],
    )
    def test_f_a_probe_that_cannot_connect_or_errors(self, sync_world, ts, error):
        world = sync_world
        _ready(world, ts)
        probe = FakeProbe(world, raises=error)
        before = _snapshot(world)

        code, text = _promote(world, probe)

        _assert_refused(code, text, f"the live graph probe failed: {type(error).__name__}")
        assert _snapshot(world) == before

    def test_a_state_change_during_the_probe_is_caught_inside_the_transaction(self, sync_world, ts):
        world = sync_world
        _ready(world, ts)

        def abandon(w) -> None:
            store = w.store
            assert store.transition_cutover(
                store.open_cutover(), from_states=("ready",), to_state="abandoned", updated_at=NOW
            )

        probe = FakeProbe(world, during=abandon)

        code, text = _promote(world, probe)

        _assert_refused(code, text, "is 'abandoned'; only a 'ready' or 'checked' cutover")
        assert probe.calls == 1
        assert _snapshot(world) == probe.snapshot_after

    def test_both_flags_are_refused_by_argparse_before_anything_is_opened(
        self, sync_world, ts, capsys
    ):
        world = sync_world
        _ready(world, ts, 2)
        probe = FakeProbe(world)
        before = _snapshot(world)

        with pytest.raises(SystemExit) as exc:
            admin.main(
                ["cutover", "promote", "--graph-swapped", "--seed-only-graph"],
                store=world.store,
                out=io.StringIO(),
                graph_probe=probe,
            )

        assert exc.value.code == 2
        assert "not allowed with argument --graph-swapped" in capsys.readouterr().err
        assert _snapshot(world) == before
        assert probe.calls == 0

    def test_neither_flag_is_refused_naming_both(self, sync_world, ts):
        world = sync_world
        _ready(world, ts, 2)
        before = _snapshot(world)

        code, text = _cli(world, "cutover", "promote")

        _assert_refused(code, text, "--seed-only-graph")
        assert _snapshot(world) == before


# ---------------------------------------------------------------------------
# Success
# ---------------------------------------------------------------------------


def _assert_seed_only_promoted(world, text: str, *, prior=None) -> int:
    """The new epoch, its activation over an empty log, the record, and the CLI text."""
    epochs = world.event_store.list_epochs()
    assert len(epochs) == 2
    new_epoch = epochs[-1]
    assert (
        new_epoch["ontology_version"],
        new_epoch["extraction_version"],
        new_epoch["model_hash"],
        new_epoch["prev_epoch_id"],
        new_epoch["provisional"],
        new_epoch["activated_at"],
    ) == (
        ONTOLOGY_VERSION,
        world.service.extraction_version,
        composed(CANDIDATE),
        world.epoch_id,
        0,
        NOW,
    )
    activation = world.event_store.get_extraction_activation(new_epoch["epoch_id"])
    assert (
        activation["turns_at_activation"],
        activation["marked_applied"],
        activation["legacy_unextracted"],
    ) == (0, 0, 0)
    assert _snapshot(world)["applied"] == []
    assert _snapshot(world)["legacy"] == []
    assert _snapshot(world)["turns"] == []
    promoted = world.store.list_cutovers()[-1]
    assert (promoted.state, promoted.promoted_epoch_id) == ("promoted", new_epoch["epoch_id"])
    assert promoted.check_report == {
        "promotion_mode": "seed_only_graph",
        "promoted_at": NOW,
        "graph_probe": SEED_ONLY.as_dict(),
        "prior_check_report": prior,
    }
    assert f"MIST_MODEL_HASH={CANDIDATE}" in text
    assert "seed-only live graph (32 node(s), 30 relationship(s)" in text
    assert "and an empty conversation log: no turn was marked applied" in text
    assert "logged turn(s) marked applied" not in text
    return int(new_epoch["epoch_id"])


class TestSuccess:
    def test_a_ready_cutover_over_an_empty_log_is_promoted_marking_nothing(self, sync_world, ts):
        world = sync_world
        _ready(world, ts)
        probe = FakeProbe(world)

        code, text = _promote(world, probe)

        assert code == 0, text
        assert probe.calls == 1
        _assert_seed_only_promoted(world, text)
        assert world.store.open_cutover() is None

    def test_a_checked_cutover_keeps_its_check_report_nested(self, sync_world, ts):
        # A 'checked' cutover over an empty log cannot come from `cutover
        # rebuild` (its floors are >= 1); the state is set directly here to
        # show the seed-only path still accepts it and keeps the report.
        world = sync_world
        _ready(world, ts)
        store = world.store
        assert store.transition_cutover(
            store.open_cutover(),
            from_states=("ready",),
            to_state="checked",
            updated_at=ts(11),
            rebuild_job_id="job",
            rebuilt_through_event_id="evt-none",
            check_report={"passed": True},
            write_check=True,
        )

        code, text = _promote(world, FakeProbe(world))

        assert code == 0, text
        _assert_seed_only_promoted(world, text, prior={"passed": True})

    def test_a_crash_at_the_fault_point_writes_nothing_and_a_rerun_succeeds(
        self, sync_world, ts, monkeypatch
    ):
        class SimulatedCrashError(Exception):
            pass

        world = sync_world
        _ready(world, ts)
        before = _snapshot(world)

        def crash(step: str) -> None:
            assert step == "ledger_appended"
            raise SimulatedCrashError(step)

        monkeypatch.setattr(world.event_store, "_promotion_fault_point", crash)
        with pytest.raises(SimulatedCrashError):
            _promote(world, FakeProbe(world))
        assert _snapshot(world) == before
        assert world.store.open_cutover().state == "ready"

        monkeypatch.undo()
        code, text = _promote(world, FakeProbe(world))

        assert code == 0, text
        _assert_seed_only_promoted(world, text)

    def test_the_transition_is_logged_with_the_probe_counts(self, sync_world, ts, caplog):
        world = sync_world
        _ready(world, ts)

        with caplog.at_level("INFO", logger="backend.extraction_backlog.cutover"):
            assert _promote(world, FakeProbe(world))[0] == 0

        lines = [r.getMessage() for r in caplog.records if "to=promoted" in r.getMessage()]
        assert len(lines) == 1
        assert "from=ready" in lines[0]
        assert "mode=seed_only_graph" in lines[0]
        assert "node_count=32" in lines[0] and "relationship_count=30" in lines[0]


class TestAfterPromotion:
    @pytest.mark.asyncio
    async def test_a_filled_log_is_refused_even_with_every_turn_cached(self, backlog_world, ts):
        # Was: 4 logged turns filled, promoted seed-only, then applied from
        # the candidate cache. Under the empty-log rule the same world is a
        # refusal: complete fill and no apply marker do not make it seed-only.
        world = backlog_world
        for i in range(4):
            _log(world, ts, i)
        _begin(world)
        world.service.model_hash = CANDIDATE
        filler = world.build_dispatcher()
        await filler.start()
        await wait_until(lambda: world.store.open_cutover().state == "ready")
        await filler.stop(timeout=2.0)
        assert world.curation.event_ids == []
        probe = FakeProbe(world)
        before = _snapshot(world)

        code, text = _promote(world, probe)

        _assert_refused(code, text, "the conversation log holds 4 logged turn(s)")
        assert probe.calls == 0
        assert _snapshot(world) == before

    @pytest.mark.asyncio
    async def test_turns_logged_after_promotion_are_extracted_and_applied_in_log_order(
        self, backlog_world, ts
    ):
        # Arrange: an empty log, filled to 'ready' by the dispatcher.
        world = backlog_world
        _begin(world)
        world.service.model_hash = CANDIDATE
        filler = world.build_dispatcher()
        await filler.start()
        await wait_until(lambda: world.store.open_cutover().state == "ready")
        await filler.stop(timeout=2.0)
        received_before = len(world.service.received)

        # Act 1: seed-only promotion.
        code, text = _promote(world, FakeProbe(world))
        assert code == 0, text
        epoch_id = _assert_seed_only_promoted(world, text)

        # Act 2: a FRESH dispatcher, built after promotion (the backend
        # recreated with the new MIST_MODEL_HASH); turns are logged and drained.
        ids = [_log(world, ts, i) for i in range(3)]
        dispatcher = world.build_dispatcher()
        await dispatcher.start()
        assert await dispatcher.drain(timeout=5.0)

        # Assert: every turn inferred under the new epoch and applied in log order.
        assert world.curation.event_ids == ids
        assert len(world.service.received) == received_before + 3
        applied = (
            world.event_store._get_connection()
            .execute(
                "SELECT event_id, stage, source FROM extraction_applied WHERE epoch_id = ?",
                (epoch_id,),
            )
            .fetchall()
        )
        assert {row[0]: (row[1], row[2]) for row in applied} == {
            event_id: ("applied", "dispatcher") for event_id in ids
        }
        assert world.store.get_activation(world.store.active_epoch()).marked_applied == 0
        assert dispatcher.snapshot().cutover is None

    @pytest.mark.asyncio
    async def test_an_empty_log_fills_to_ready_and_promotes_with_zero_turns(self, backlog_world):
        world = backlog_world
        _begin(world)
        world.service.model_hash = CANDIDATE
        filler = world.build_dispatcher()
        await filler.start()
        await wait_until(lambda: world.store.open_cutover().state == "ready")
        await filler.stop(timeout=2.0)

        code, text = _promote(world, FakeProbe(world))

        assert code == 0, text
        _assert_seed_only_promoted(world, text)


# ---------------------------------------------------------------------------
# `cutover probe`: the same checks, read-only
# ---------------------------------------------------------------------------


def _probe_cli(world, probe) -> tuple[int, str]:
    return _cli(world, "cutover", "probe", probe=probe)


class TestProbeCommand:
    def test_an_empty_log_and_a_seed_only_graph_pass_with_exit_0(self, sync_world, ts):
        world = sync_world
        _ready(world, ts)
        probe = FakeProbe(world)
        before = _snapshot(world)

        code, text = _probe_cli(world, probe)

        assert code == 0, text
        assert probe.calls == 1
        assert "[probe] conversation log: PASS (0 logged turns)" in text
        assert "[probe] live graph: PASS (node_count=32 " in text
        assert "violation" not in text
        assert _snapshot(world) == before
        assert world.store.open_cutover().state == "ready"

    def test_it_runs_without_an_open_cutover(self, sync_world):
        world = sync_world
        assert world.store.open_cutover() is None
        before = _snapshot(world)

        code, text = _probe_cli(world, FakeProbe(world))

        assert code == 0, text
        assert _snapshot(world) == before

    def test_a_logged_turn_refuses_with_exit_2_and_the_graph_is_still_probed(self, sync_world, ts):
        world = sync_world
        world.event_store.start_session("s-test", input_modality="text", origin="test")
        _log(world, ts, 0, session_id="s-test")
        _log(world, ts, 1, session_id="s-test")
        probe = FakeProbe(world)
        before = _snapshot(world)

        code, text = _probe_cli(world, probe)

        assert code == 2, text
        assert probe.calls == 1
        assert "[probe] conversation log: REFUSED (2 logged turn(s))" in text
        assert f"  violation: {log_not_empty_reason(2)}" in text
        assert "[probe] live graph: PASS" in text
        assert "would be REFUSED (exit 2)" in text
        assert _snapshot(world) == before

    def test_every_graph_violation_is_printed_with_exit_2(self, sync_world):
        world = sync_world
        bad = GraphProbeReport(
            **(
                SEED_ONLY.as_dict()
                | {"nodes_without_seed_version": 4, "relationships_with_extraction_stamp": 5}
            )
        )
        probe = FakeProbe(world, bad)
        before = _snapshot(world)

        code, text = _probe_cli(world, probe)

        assert code == 2, text
        assert "[probe] conversation log: PASS" in text
        assert "[probe] live graph: REFUSED (" in text
        assert "nodes_without_seed_version=4" in text
        for violation in bad.violations():
            assert f"  violation: {violation}" in text
        assert len(bad.violations()) == 2
        assert _snapshot(world) == before

    def test_both_checks_failing_print_both_with_exit_2(self, sync_world, ts):
        world = sync_world
        _log(world, ts, 0)
        probe = FakeProbe(world, GraphProbeReport(**(SEED_ONLY.as_dict() | {"node_count": 0})))

        code, text = _probe_cli(world, probe)

        assert code == 2, text
        assert "conversation log: REFUSED (1 logged turn(s))" in text
        assert "  violation: the live graph has no nodes" in text

    @pytest.mark.parametrize(
        "error",
        [
            Neo4jConnectionError("Failed to connect to Neo4j: refused"),
            GraphProbeError("could not probe the live graph at bolt://x:7687: ServiceUnavailable"),
        ],
    )
    @pytest.mark.parametrize("logged", [0, 1])
    def test_a_probe_that_could_not_run_exits_1(self, sync_world, ts, error, logged):
        # Exit 1 even when the log check already refuses: the graph result is
        # unknown, and the operator must fix the probe to see it.
        world = sync_world
        for i in range(logged):
            _log(world, ts, i)
        probe = FakeProbe(world, raises=error)
        before = _snapshot(world)

        code, text = _probe_cli(world, probe)

        assert code == 1, text
        assert f"[probe] live graph: NOT RUN: {type(error).__name__}: " in text
        assert "could not complete" in text
        assert _snapshot(world) == before

    def test_the_real_probe_over_a_fake_connection_writes_nothing(self, sync_world, monkeypatch):
        # The CLI's default probe (`probe_live_graph_from_env`), with
        # `Neo4jConnection` replaced: one read query, no write, disconnected.
        from backend.knowledge.storage import neo4j_connection

        events: list[str] = []

        class ReadOnlyConnection:
            def __init__(self, config) -> None:
                events.append("init")

            def connect(self) -> None:
                events.append("connect")

            def execute_query(self, query, params=None):
                events.append("query")
                assert query == SEED_ONLY_PROBE_CYPHER
                return [SEED_ONLY.as_dict()]

            def execute_write(self, query, params=None):
                events.append("WRITE")
                raise AssertionError("cutover probe must never write the graph")

            def disconnect(self) -> None:
                events.append("disconnect")

        monkeypatch.setattr(neo4j_connection, "Neo4jConnection", ReadOnlyConnection)
        world = sync_world
        before = _snapshot(world)

        code, text = _probe_cli(world, None)

        assert code == 0, text
        assert events == ["init", "connect", "query", "disconnect"]
        assert _snapshot(world) == before


# ---------------------------------------------------------------------------
# The probe itself
# ---------------------------------------------------------------------------

_WRITE_CLAUSES = (
    r"\bCREATE\b",
    r"\bMERGE\b",
    r"\bSET\b",
    r"\bDELETE\b",
    r"\bDETACH\b",
    r"\bREMOVE\b",
    r"\bDROP\b",
    r"\bLOAD\s+CSV\b",
    r"\bFOREACH\b",
    r"\bCALL\s*\{",
    r"\bIN\s+TRANSACTIONS\b",
    r"\bCALL\b",
    r"\bapoc\b",
)


def _write_clauses_in(statement: str) -> list[str]:
    return [p for p in _WRITE_CLAUSES if re.search(p, statement, flags=re.IGNORECASE)]


class TestProbe:
    def test_the_probe_cypher_has_no_write_clause(self):
        assert _write_clauses_in(SEED_ONLY_PROBE_CYPHER) == []

    @pytest.mark.parametrize(
        "statement",
        [
            "MATCH (n) DETACH DELETE n",
            "create (n:X)",
            "MATCH (n) set n.x = 1",
            "MERGE (n:X {id: 1})",
            "MATCH (n) REMOVE n.x",
            "DROP INDEX foo",
            "LOAD CSV FROM 'file:///x' AS row RETURN row",
            "FOREACH (x IN [1] | CREATE ())",
            "CALL { MATCH (n) RETURN n } IN TRANSACTIONS",
            "CALL apoc.create.node(['X'], {})",
            "RETURN apoc.periodic.iterate('a', 'b', {})",
        ],
    )
    def test_the_write_clause_check_is_not_vacuous(self, statement):
        assert _write_clauses_in(statement) != []

    def test_the_probe_is_one_read_query_parsed_into_the_report(self):
        class RecordingConnection:
            def __init__(self) -> None:
                self.queries: list[tuple[str, dict]] = []

            def execute_query(self, query, params=None):
                self.queries.append((query, params))
                return [SEED_ONLY.as_dict()]

            def execute_write(self, query, params=None):  # pragma: no cover -- must not run
                raise AssertionError("the probe must never use execute_write")

        connection = RecordingConnection()

        report = probe_graph(connection)

        assert report == SEED_ONLY
        assert connection.queries == [(SEED_ONLY_PROBE_CYPHER, {})]

    @pytest.mark.parametrize(
        "rows",
        [[], [SEED_ONLY.as_dict(), SEED_ONLY.as_dict()], [{"node_count": 3}]],
        ids=["no-row", "two-rows", "missing-counts"],
    )
    def test_an_unexpected_result_shape_is_a_probe_error(self, rows):
        class Connection:
            def execute_query(self, query, params=None):
                return rows

        with pytest.raises(GraphProbeError):
            probe_graph(Connection())

    @pytest.mark.parametrize(
        ("fail_on", "error_name"),
        [(None, None), ("connect", "ServiceUnavailable"), ("query", "ServiceUnavailable")],
    )
    def test_the_real_probe_translates_driver_errors_and_always_disconnects(
        self, monkeypatch, fail_on, error_name
    ):
        # The real wiring, with `Neo4jConnection` replaced: nothing connects.
        from neo4j.exceptions import ServiceUnavailable

        from backend.extraction_backlog.cutover import probe_live_graph_from_env
        from backend.knowledge.storage import neo4j_connection

        events: list[str] = []

        class FakeConnection:
            def __init__(self, config) -> None:
                events.append(f"init {config.uri}")

            def connect(self) -> None:
                events.append("connect")
                if fail_on == "connect":
                    raise ServiceUnavailable("down")

            def execute_query(self, query, params=None):
                events.append("query")
                if fail_on == "query":
                    raise ServiceUnavailable("dropped")  # a DriverError, not a Neo4jError
                return [SEED_ONLY.as_dict()]

            def execute_write(self, query, params=None):  # pragma: no cover -- must not run
                raise AssertionError("the probe must never use execute_write")

            def disconnect(self) -> None:
                events.append("disconnect")

        monkeypatch.setattr(neo4j_connection, "Neo4jConnection", FakeConnection)

        if fail_on is None:
            assert probe_live_graph_from_env() == SEED_ONLY
        else:
            with pytest.raises(GraphProbeError, match=error_name):
                probe_live_graph_from_env()
        assert events[-1] == "disconnect"
        assert events.count("query") <= 1

    def test_the_report_names_every_count(self):
        assert json.loads(json.dumps(SEED_ONLY.as_dict())) == {
            "node_count": 32,
            "nodes_without_seed_version": 0,
            "nodes_with_extraction_stamp": 0,
            "relationship_count": 30,
            "relationships_without_seed_version": 0,
            "relationships_with_extraction_stamp": 0,
        }
        assert SEED_ONLY.violations() == []
