"""The run modes: `--cpu-load-procs`, `--placement`, `--threads`, `--rescan-facts`, `--trace`.

The busy-loop processes start for the run and stop after it; the block builder gets the
placement asked for; torch's thread count is set before any model loads; forced rescans of the
fact index run for the whole run and are logged; and the trace writes a run header and one
record per probe that agrees with the run file.
"""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import pytest

import scripts.eval_harness.memory_recall_run as harness
from scripts.eval_harness.memory_recall import load_probes
from scripts.eval_harness.memory_recall_run import cpu_load, main


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class FakeEvent:
    def __init__(self) -> None:
        self.flag = False

    def set(self) -> None:
        self.flag = True

    def is_set(self) -> bool:
        return self.flag


class FakeProcess:
    """Exits on join once the stop event is set, unless `stuck`; a join costs its timeout."""

    def __init__(
        self,
        *,
        target: object,
        args: tuple[FakeEvent],
        name: str,
        daemon: bool,
        clock: FakeClock,
        stuck: bool,
    ) -> None:
        self.stop = args[0]
        self.name = name
        self.daemon = daemon
        self.clock = clock
        self.stuck = stuck
        self.started = False
        self.alive = False
        self.joins: list[float | None] = []
        self.terminated = False

    def start(self) -> None:
        self.started = True
        self.alive = True

    def join(self, timeout: float | None = None) -> None:
        self.joins.append(timeout)
        if not self.alive:
            return
        if self.stop.is_set() and not self.stuck:
            self.alive = False
        elif timeout:
            self.clock.now += timeout

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.terminated = True
        self.alive = False


def _factory(clock: FakeClock, made: list[FakeProcess], *, stuck: set[int] = frozenset()):  # type: ignore[assignment]
    def make(**kwargs: object) -> FakeProcess:
        process = FakeProcess(clock=clock, stuck=len(made) in stuck, **kwargs)  # type: ignore[arg-type]
        made.append(process)
        return process

    return make


def test_the_processes_run_for_the_body_and_stop_after() -> None:
    clock, made = FakeClock(), []
    event = FakeEvent()

    with cpu_load(
        3, process_factory=_factory(clock, made), event_factory=lambda: event, clock=clock
    ):
        assert [p.started and p.alive for p in made] == [True, True, True]
        assert not event.is_set()

    assert event.is_set()
    assert [p.alive for p in made] == [False, False, False]
    assert not any(p.terminated for p in made)
    assert all(p.daemon for p in made)
    assert [p.name for p in made] == ["memory-load-0", "memory-load-1", "memory-load-2"]


def test_a_process_that_will_not_stop_is_terminated_within_the_join_timeout() -> None:
    clock, made = FakeClock(), []

    with cpu_load(
        3,
        process_factory=_factory(clock, made, stuck={0}),
        event_factory=FakeEvent,
        clock=clock,
        join_timeout_s=5.0,
    ):
        pass

    assert made[0].terminated and not made[0].alive
    assert made[0].joins == [pytest.approx(5.0), 1.0]  # the bounded wait, then the reap
    # The stuck join used the whole timeout, so the others get no more time by the clock.
    assert made[1].joins == [0.0] and made[2].joins == [0.0]
    assert clock.now == pytest.approx(105.0)


def test_the_processes_stop_when_the_body_raises() -> None:
    clock, made = FakeClock(), []

    load = cpu_load(2, process_factory=_factory(clock, made), event_factory=FakeEvent, clock=clock)
    with pytest.raises(RuntimeError, match="run failed"), load:
        raise RuntimeError("run failed")

    assert [p.alive for p in made] == [False, False]


def test_no_processes_and_a_negative_count() -> None:
    with cpu_load(0) as started:
        assert started == []
    with pytest.raises(ValueError, match="negative"), cpu_load(-1):
        pass


def test_a_short_run_with_load_writes_every_probe_and_leaves_no_process(tmp_path: Path) -> None:
    output = tmp_path / "run.jsonl"

    assert main(["--output", str(output), "--cpu-load-procs", "2", "--deadline-ms", "10000"]) == 0

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [r["probe_id"] for r in records] == [p.probe_id for p in load_probes()]
    assert {r["cpu_load_procs"] for r in records} == {2}
    assert all(r["critical_path_ms"] >= 0 and "delta_tokens_est" in r for r in records)
    assert multiprocessing.active_children() == []


# -- the run modes ------------------------------------------------------------------------------


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def placements(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The placement every block builder of the run is built with."""
    seen: list[str] = []
    base = harness.MemoryBlockBuilder

    class RecordingBuilder(base):  # type: ignore[misc, valid-type]
        def __init__(self, *, counter: object, config: object, placement: str) -> None:
            seen.append(placement)
            super().__init__(counter=counter, config=config, placement=placement)  # type: ignore[arg-type]

    monkeypatch.setattr(harness, "MemoryBlockBuilder", RecordingBuilder)
    return seen


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, object]]:
    """`torch.set_num_threads` calls and embedder loads, in the order the run makes them."""
    seen: list[tuple[str, object]] = []
    build_embedder = harness.build_embedder

    def embedder(choice: str, **kwargs: object) -> object:
        seen.append(("embedder", choice))
        return build_embedder(choice, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(harness.torch, "set_num_threads", lambda n: seen.append(("threads", n)))
    monkeypatch.setattr(harness, "build_embedder", embedder)
    return seen


@pytest.mark.parametrize(
    ("argv", "expected"),
    [([], "user_prefix"), (["--placement", "system_tail"], "system_tail")],
    ids=["default", "system-tail"],
)
def test_the_block_is_built_for_the_placement_asked_for(
    tmp_path: Path, placements: list[str], argv: list[str], expected: str
) -> None:
    assert main(["--output", str(tmp_path / "run.jsonl"), *argv]) == 0

    assert placements == [expected]


def test_threads_are_set_before_any_model_loads(
    tmp_path: Path, events: list[tuple[str, object]]
) -> None:
    assert main(["--output", str(tmp_path / "run.jsonl"), "--threads", "3"]) == 0

    assert events == [("threads", 3), ("embedder", "fake")]


def test_without_threads_torch_keeps_its_default(
    tmp_path: Path, events: list[tuple[str, object]]
) -> None:
    assert main(["--output", str(tmp_path / "run.jsonl")]) == 0

    assert events == [("embedder", "fake")]


def test_rescan_facts_needs_a_log(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--output", str(tmp_path / "run.jsonl"), "--rescan-facts", "3"])

    assert "--rescan-facts needs --rescan-log" in capsys.readouterr().err


@pytest.fixture
def every_mode(
    tmp_path: Path, placements: list[str], events: list[tuple[str, object]]
) -> dict[str, Path]:
    """One fake run with every mode on: the run file, the rescan log and the trace."""
    paths = {
        "output": tmp_path / "run.jsonl",
        "log": tmp_path / "rescan.json",
        "trace": tmp_path / "trace.jsonl",
    }
    code = main(
        [
            "--output",
            str(paths["output"]),
            "--placement",
            "system_tail",
            "--threads",
            "2",
            "--rescan-facts",
            "3",
            "--rescan-log",
            str(paths["log"]),
            "--trace",
            str(paths["trace"]),
            "--deadline-ms",
            "10000",
        ]
    )
    assert code == 0
    return paths


def test_a_run_in_every_mode_writes_every_probe_and_logs_the_rescans(
    every_mode: dict[str, Path], placements: list[str], events: list[tuple[str, object]]
) -> None:
    output, log = every_mode["output"], every_mode["log"]

    records = _records(output)
    assert [r["probe_id"] for r in records] == [p.probe_id for p in load_probes()]
    assert all(isinstance(r["shortlisted"], int) and r["shortlisted"] >= 0 for r in records)
    assert any(r["shortlisted"] > 0 for r in records)
    assert placements == ["system_tail"] and events[0] == ("threads", 2)
    rescans = json.loads(log.read_text(encoding="utf-8"))
    assert rescans["rescan_facts"] == 3
    assert rescans["cycles_completed"] == len(rescans["cycles"]) >= 1
    versions = [cycle["version"] for cycle in rescans["cycles"]]
    assert versions == list(range(1, len(versions) + 1))
    for cycle in rescans["cycles"]:
        assert cycle["began_s"] >= 0 and cycle["seconds"] >= 0
        # Each cycle moves the synthetic facts to a new revision: three re-embedded rows.
        assert (cycle["report"]["updated"], cycle["report"]["embedded"]) == (3, 3)
        assert cycle["report"]["inserted"] == cycle["report"]["deleted"] == 0


# -- the trace ----------------------------------------------------------------------------------


def test_the_trace_opens_with_the_run_header(every_mode: dict[str, Path]) -> None:
    header = _records(every_mode["trace"])[0]

    assert header["record"] == "run"
    assert (header["device"], header["placement"], header["rescan_facts"]) == (
        "cpu",
        "system_tail",
        3,
    )
    assert isinstance(header["threads"], int) and header["threads"] > 0
    assert (header["embedder_path"], header["reranker_path"]) == (None, None)
    assert header["cost_per_pair_ms"] is None and header["call_overhead_ms"] is None
    assert header["config"]["deadline_ms"] == 10000
    assert {"min_score", "rerank_top_n", "block_tokens", "rerank_weight"} <= set(header["config"])


def test_the_trace_has_one_record_per_probe_matching_its_run_record(
    every_mode: dict[str, Path],
) -> None:
    traces = _records(every_mode["trace"])[1:]
    records = {r["probe_id"]: r for r in _records(every_mode["output"])}
    top_n = _records(every_mode["trace"])[0]["config"]["rerank_top_n"]

    assert sorted(t["probe_id"] for t in traces) == sorted(records)
    for trace in traces:
        record = records[trace["probe_id"]]
        assert trace["record"] == "probe"
        assert trace["shortlisted"] == record["shortlisted"] == len(trace["scored"])
        assert trace["reranked"] == record["reranked"]
        assert trace["rerank_matches_shortlist"] is True
        assert len(trace["embed_calls"]) == 1 and trace["embed_calls"][0]["wall_ms"] >= 0
        positions = [entry["shortlist"] for entry in trace["fused"]]
        assert positions == [n if n <= top_n else None for n in range(1, len(positions) + 1)]
        assert [s["rank"] for s in trace["scored"]] == list(range(1, len(trace["scored"]) + 1))
        for scored in trace["scored"]:
            assert (scored["logit"] is None) == (not scored["reranked"])
        for call in trace["rerank_calls"]:
            assert call["pairs_scored"] == record["reranked"] == len(call["pair_tokens"])
            assert call["pairs_within"] >= call["pairs_scored"] and call["discarded"] == 0


def test_the_trace_follows_each_target_through_the_stages(every_mode: dict[str, Path]) -> None:
    traces = _records(every_mode["trace"])[1:]
    targets = [target for trace in traces for target in trace["targets"]]

    traced = [target for target in targets if target["traced"]]
    assert {target["kind"] for target in traced} == {"fact", "note"}
    assert all(target["kind"] not in ("fact", "note") for target in targets if not target["traced"])
    in_block = [target for target in traced if target["in_block"]]
    assert in_block and all(target["indexed_at_now"] for target in in_block)
    assert any(target["disposition"] == "injected" for target in in_block)
    for target in traced:
        if target["shortlist"] is not None:
            assert target["fused"] == target["shortlist"]
        if not target["indexed_at_now"]:
            continue
        oracle = target["oracle"]
        if target["kind"] == "fact":
            assert set(target["ranks"]) == {"fts", "vector", "entity"}
            assert oracle["text"] and isinstance(oracle["logit"], float)
        else:
            assert set(target["ranks"]) == {"note_vector", "note_fts"}
            logits = [chunk["logit"] for chunk in oracle["chunks"]]
            assert logits and oracle["max"] == max(logits)
            assert all(chunk["heading"] and chunk["text"] for chunk in oracle["chunks"])
