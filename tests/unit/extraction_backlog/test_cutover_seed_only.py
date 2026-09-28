"""`cutover promote --seed-only-graph`: promotion over a live graph that holds seed data only.

Every refusal is asserted write-free: the ledger, the activations, the apply
markers, the legacy list and the whole `epoch_cutover` row are compared before
and after. The live-graph probe is always a fake (`FakeProbe`); the unit tier
never connects to Neo4j. The probe's Cypher is checked textually for write
clauses, and `probe_graph` is run over a recording fake connection.
"""

from __future__ import annotations

import io
import json
import re
from datetime import datetime

import pytest

from backend.errors import Neo4jConnectionError, Neo4jQueryError
from backend.extraction_backlog import admin
from backend.extraction_backlog.cutover import (
    SEED_ONLY_PROBE_CYPHER,
    GraphProbeError,
    GraphProbeReport,
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


def _log(world, ts, index: int) -> str:
    return world.log_turn(
        session_id="s1",
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


def _ready(world, ts, n: int = 3) -> list[str]:
    """Log `n` turns (none applied), begin, cache every turn under the candidate, set 'ready'."""
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
    """Every row a promotion could write, in a stable order, every column."""
    conn = world.event_store._get_connection()
    queries = {
        "ledger": "SELECT * FROM epoch_ledger ORDER BY epoch_id",
        "activation": "SELECT * FROM extraction_activation ORDER BY epoch_id",
        "applied": "SELECT * FROM extraction_applied ORDER BY epoch_id, event_id",
        "legacy": "SELECT * FROM extraction_legacy_turns ORDER BY epoch_id, event_id",
        "cutover": "SELECT * FROM epoch_cutover ORDER BY cutover_id",
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
        world = sync_world
        ids = _ready(world, ts, 3)
        probe = FakeProbe(
            world,
            during=lambda w: w.event_store.mark_extraction_stage(
                event_id=ids[0], epoch_id=w.epoch_id, stage="curated", updated_at=ts(1)
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
        _ready(world, ts, 2)
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
    def test_e_a_graph_that_is_not_seed_only(self, sync_world, ts, field, value, reason):
        world = sync_world
        _ready(world, ts, 2)
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
    def test_e_a_probe_that_cannot_connect_or_errors(self, sync_world, ts, error):
        world = sync_world
        _ready(world, ts, 2)
        probe = FakeProbe(world, raises=error)
        before = _snapshot(world)

        code, text = _promote(world, probe)

        _assert_refused(code, text, f"the live graph probe failed: {type(error).__name__}")
        assert _snapshot(world) == before

    def test_f_a_state_change_during_the_probe_is_caught_inside_the_transaction(
        self, sync_world, ts
    ):
        world = sync_world
        _ready(world, ts, 2)

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


def _assert_seed_only_promoted(world, text: str, *, turns: int, prior=None) -> int:
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
    ) == (turns, 0, 0)
    assert _snapshot(world)["applied"] == []
    assert _snapshot(world)["legacy"] == []
    promoted = world.store.list_cutovers()[-1]
    assert (promoted.state, promoted.promoted_epoch_id) == ("promoted", new_epoch["epoch_id"])
    assert promoted.check_report == {
        "promotion_mode": "seed_only_graph",
        "promoted_at": NOW,
        "graph_probe": SEED_ONLY.as_dict(),
        "prior_check_report": prior,
    }
    assert f"MIST_MODEL_HASH={CANDIDATE}" in text
    assert "0 of" in text and "seed-only live graph (32 node(s), 30 relationship(s)" in text
    return int(new_epoch["epoch_id"])


class TestSuccess:
    def test_a_ready_cutover_is_promoted_with_an_activation_that_marks_nothing(
        self, sync_world, ts
    ):
        world = sync_world
        _ready(world, ts, 3)
        probe = FakeProbe(world)

        code, text = _promote(world, probe)

        assert code == 0, text
        assert probe.calls == 1
        _assert_seed_only_promoted(world, text, turns=3)
        assert world.store.open_cutover() is None

    def test_a_checked_cutover_keeps_its_check_report_nested(self, sync_world, ts):
        world = sync_world
        ids = _ready(world, ts, 2)
        store = world.store
        assert store.transition_cutover(
            store.open_cutover(),
            from_states=("ready",),
            to_state="checked",
            updated_at=ts(11),
            rebuild_job_id="job",
            rebuilt_through_event_id=ids[-1],
            check_report={"passed": True},
            write_check=True,
        )

        code, text = _promote(world, FakeProbe(world))

        assert code == 0, text
        _assert_seed_only_promoted(world, text, turns=2, prior={"passed": True})

    def test_a_crash_at_the_fault_point_writes_nothing_and_a_rerun_succeeds(
        self, sync_world, ts, monkeypatch
    ):
        class SimulatedCrashError(Exception):
            pass

        world = sync_world
        _ready(world, ts, 3)
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
        _assert_seed_only_promoted(world, text, turns=3)

    def test_the_transition_is_logged_with_the_probe_counts(self, sync_world, ts, caplog):
        world = sync_world
        _ready(world, ts, 1)

        with caplog.at_level("INFO", logger="backend.extraction_backlog.cutover"):
            assert _promote(world, FakeProbe(world))[0] == 0

        lines = [r.getMessage() for r in caplog.records if "to=promoted" in r.getMessage()]
        assert len(lines) == 1
        assert "from=ready" in lines[0]
        assert "mode=seed_only_graph" in lines[0]
        assert "node_count=32" in lines[0] and "relationship_count=30" in lines[0]


class TestAfterPromotion:
    @pytest.mark.asyncio
    async def test_every_turn_is_applied_in_log_order_from_the_candidate_cache(
        self, backlog_world, ts
    ):
        # Arrange: 4 turns logged, NONE applied under the source epoch; filled.
        world = backlog_world
        ids = [_log(world, ts, i) for i in range(4)]
        _begin(world)
        world.service.model_hash = CANDIDATE
        filler = world.build_dispatcher()
        await filler.start()
        await wait_until(lambda: world.store.open_cutover().state == "ready")
        await filler.stop(timeout=2.0)
        assert world.curation.event_ids == []
        received_before = len(world.service.received)

        # Act 1: seed-only promotion.
        code, text = _promote(world, FakeProbe(world))
        assert code == 0, text
        epoch_id = _assert_seed_only_promoted(world, text, turns=4)

        # Act 2: a FRESH dispatcher, built after promotion (the backend
        # recreated with the new MIST_MODEL_HASH), drains the log.
        dispatcher = world.build_dispatcher()
        await dispatcher.start()
        assert await dispatcher.drain(timeout=5.0)

        # Assert: every turn applied, in log order, from the cache.
        assert world.curation.event_ids == ids
        assert len(world.service.received) == received_before
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
        _assert_seed_only_promoted(world, text, turns=0)


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
