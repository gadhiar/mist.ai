"""The run modes: `--cpu-load-procs`, `--placement`, `--threads`, `--rescan-facts`, `--trace`,
`--probes`, `--gpu-load-procs`, `--vram-log` and `--vram-stt`; the need gate's flags and
`--fact-view`; and `run_settings`.

The busy-loop processes start for the run and stop after it; the block builder gets the
placement asked for; torch's thread count is set before any model loads; forced rescans of the
fact index run for the whole run and are logged; and the trace writes a run header and one
record per probe that agrees with the run file. The probe file can be chosen; the GPU load
needs CUDA; and the VRAM log reads its points in order through injected NVML, torch and Whisper
fakes, so no test here needs a GPU or Whisper's weights.

The need gate's model files are written here with a zero weight vector, so every query gets
`sigmoid(b)` and a test picks the band by `b`. A gate embedder given by path is a recording fake,
so no test loads a model.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import multiprocessing
import sys
import time
import weakref
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import backend.memory.need_gate as need_gate_module
import scripts.eval_harness.memory_recall_run as harness
from backend.memory.config import MemoryConfig
from backend.memory.models import MemoryRequest
from backend.memory.need_gate import (
    NEED_EMBEDDERS,
    NeedGate,
    NeedKey,
    load_bank,
    load_models,
    need_embedder_kwargs,
    need_model_key,
)
from backend.memory.retrieval_view import FACT_VIEWS
from scripts.eval_harness.memory_recall import FIXTURE_DIR, load_probes
from scripts.eval_harness.memory_recall_run import (
    FAKE_EMBEDDER_NAME,
    FAKE_EMBEDDER_REVISION,
    Fixture,
    RunSetting,
    TokenEmbedder,
    TokenReranker,
    cpu_load,
    gpu_load,
    main,
    run_fixture,
    run_settings,
)


@pytest.fixture(autouse=True)
def _gate_off_unless_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    # A run without `--need-gate` takes the config's mode: `off` by default, or whatever
    # `MIST_MEMORY_NEED_GATE` says. The shipped gate model's record is keyed to the real fact
    # embedder, so a token-embedder run with the gate on or shadow finds no record for its key;
    # the tests of the other flags pin the gate off through the environment, which
    # `MemoryConfig.from_env` reads. A `--need-gate` flag overrides it.
    monkeypatch.setenv("MIST_MEMORY_NEED_GATE", "off")


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
        args: tuple[Any, ...],
        name: str,
        daemon: bool,
        clock: FakeClock,
        stuck: bool,
    ) -> None:
        self.target = target
        self.args = args
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
    config = set(header["config"])
    assert {"select_k", "facets", "rerank_top_n", "block_tokens", "list_max_items"} <= config
    assert not {"min_score", "rerank_weight", "recency_weight", "confidence_weight"} & config


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
        # The record also counts the candidates dropped before the order (suppressed facts,
        # facets without members), which the trace's ordered shortlist leaves out.
        assert trace["shortlisted"] == len(trace["scored"]) <= record["shortlisted"]
        assert trace["reranked"] == record["reranked"]
        assert trace["rerank_matches_shortlist"] is True
        assert len(trace["embed_calls"]) == 1 and trace["embed_calls"][0]["wall_ms"] >= 0
        positions = [entry["shortlist"] for entry in trace["fused"]]
        assert positions == [n if n <= top_n else None for n in range(1, len(positions) + 1)]
        assert [s["rank"] for s in trace["scored"]] == list(range(1, len(trace["scored"]) + 1))
        for scored in trace["scored"]:
            assert scored["position"] == scored["rank"] and scored["pool_rank"] >= 1
            assert (scored["logit"] is None) == (scored["ce_rank"] is None)
            assert (scored["p_ce"] is None) == (scored["ce_rank"] is None)
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


# -- selection flags ----------------------------------------------------------------------------


def test_min_score_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--output", str(tmp_path / "run.jsonl"), "--min-score", "0.3"])

    assert exit_info.value.code == 2
    assert not (tmp_path / "run.jsonl").exists()


@pytest.mark.parametrize(("flag", "facets"), [("on", True), ("off", False)])
def test_select_k_and_facets_reach_the_config(flag: str, facets: bool) -> None:
    args = harness._parser().parse_args(
        ["--output", "run.jsonl", "--select-k", "5", "--facets", flag]
    )

    config = harness.config_from_args(args)

    assert (config.select_k, config.facets) == (5, facets)


def test_facets_takes_only_on_or_off() -> None:
    with pytest.raises(SystemExit) as exit_info:
        harness._parser().parse_args(["--output", "run.jsonl", "--facets", "yes"])

    assert exit_info.value.code == 2


# -- the probe file -----------------------------------------------------------------------------


def test_a_chosen_probe_file_writes_exactly_its_probes(tmp_path: Path) -> None:
    lines = (FIXTURE_DIR / "probes.jsonl").read_text(encoding="utf-8").splitlines()
    chosen = [lines[5], lines[0]]
    probes = tmp_path / "two_probes.jsonl"
    probes.write_text("\n".join(chosen) + "\n", encoding="utf-8")
    output = tmp_path / "run.jsonl"

    assert main(["--output", str(output), "--probes", str(probes)]) == 0

    expected = [json.loads(line)["probe_id"] for line in chosen]
    assert [r["probe_id"] for r in _records(output)] == expected


# -- the GPU load -------------------------------------------------------------------------------


def test_the_gpu_load_runs_matmul_processes_on_cuda_and_stops_them() -> None:
    clock, made = FakeClock(), []
    event = FakeEvent()

    with gpu_load(
        2, process_factory=_factory(clock, made), event_factory=lambda: event, clock=clock
    ):
        assert [p.started and p.alive for p in made] == [True, True]

    assert event.is_set() and [p.alive for p in made] == [False, False]
    assert [p.name for p in made] == ["memory-gpu-load-0", "memory-gpu-load-1"]
    assert all(p.target is harness.matmul_loop and p.args == (event, "cuda") for p in made)
    with gpu_load(0) as started:
        assert started == []


class CountingEvent:
    """Set once it has been asked `after` times."""

    def __init__(self, after: int) -> None:
        self.after = after
        self.asked = 0

    def set(self) -> None:
        self.asked = self.after

    def is_set(self) -> bool:
        self.asked += 1
        return self.asked > self.after


def test_the_matmul_loop_runs_until_stopped() -> None:
    stop = CountingEvent(after=3)

    harness.matmul_loop(stop, "cpu", size=4)

    assert stop.asked == 4


# -- the GPU fakes ------------------------------------------------------------------------------

_MIB = 1024 * 1024
_CUDA_CALLS = {
    "is_available",
    "init",
    "ones",
    "synchronize",
    "mem_get_info",
    "memory_allocated",
    "max_memory_allocated",
    "memory_reserved",
    "max_memory_reserved",
    "empty_cache",
}


@dataclass
class GpuState:
    """Everything the fakes saw, in one ordered list, and what they answer."""

    nvml_mib: list[float]
    available: bool = True
    nvml_ok: bool = True
    events: list[tuple[Any, ...]] = field(default_factory=list)
    whisper: weakref.ref[Any] | None = None

    def whisper_alive(self) -> bool:
        return self.whisper is not None and self.whisper() is not None

    def names(self) -> list[str]:
        return [str(event[0]) for event in self.events]

    def nvml_reads(self) -> list[int]:
        return [n for n, event in enumerate(self.events) if event[0] == "nvml"]


class FakeTorchCuda:
    """`torch.cuda` as the harness reads it: 12 GiB total, 6 GiB free, fixed allocator counts."""

    def __init__(self, state: GpuState) -> None:
        self.state = state

    def _log(self, name: str) -> None:
        self.state.events.append((name,))

    def is_available(self) -> bool:
        self._log("is_available")
        return self.state.available

    def init(self) -> None:
        self._log("init")

    def synchronize(self, device: object = None) -> None:
        self._log("synchronize")

    def mem_get_info(self, device: object = None) -> tuple[int, int]:
        self._log("mem_get_info")
        return 6144 * _MIB, 12288 * _MIB

    def memory_allocated(self, device: object = None) -> int:
        self._log("memory_allocated")
        return 100 * _MIB

    def max_memory_allocated(self, device: object = None) -> int:
        self._log("max_memory_allocated")
        return 150 * _MIB

    def memory_reserved(self, device: object = None) -> int:
        self._log("memory_reserved")
        return 200 * _MIB

    def max_memory_reserved(self, device: object = None) -> int:
        self._log("max_memory_reserved")
        return 250 * _MIB

    def empty_cache(self) -> None:
        self._log("empty_cache")


class FakeTorch:
    def __init__(self, state: GpuState) -> None:
        self.state = state
        self.cuda = FakeTorchCuda(state)

    def ones(self, *shape: int, device: object = None) -> object:
        self.state.events.append(("ones", shape, device))
        return object()


class FakeNvml:
    def __init__(self, state: GpuState) -> None:
        self.state = state

    def used_bytes(self) -> int:
        self.state.events.append(("nvml", self.state.whisper_alive()))
        return int(self.state.nvml_mib.pop(0) * _MIB)

    def close(self) -> None:
        self.state.events.append(("nvml_close",))


class FakeWhisperModel:
    def __init__(self, state: GpuState) -> None:
        self.state = state

    def transcribe(self, audio: np.ndarray, **kwargs: object) -> dict[str, str]:
        self.state.events.append(("transcribe", audio.shape, str(audio.dtype), kwargs))
        return {"text": ""}


def _access(state: GpuState) -> harness.GpuAccess:
    def open_nvml() -> FakeNvml:
        state.events.append(("nvml_open",))
        if not state.nvml_ok:
            raise harness.GpuUnavailableError("NVML is not available")
        return FakeNvml(state)

    def load_whisper(name: str) -> FakeWhisperModel:
        state.events.append(("whisper_load", name))
        model = FakeWhisperModel(state)
        state.whisper = weakref.ref(model)  # weak, so the release can be seen
        return model

    @contextlib.contextmanager
    def load_procs(procs: int) -> Iterator[list[object]]:
        # As `gpu_load`: no process, and nothing to record, for zero.
        if procs == 0:
            yield []
            return
        state.events.append(("gpu_load_start", procs))
        try:
            yield []
        finally:
            state.events.append(("gpu_load_stop",))

    return harness.GpuAccess(
        cuda=harness.TorchCuda(FakeTorch(state)),  # type: ignore[arg-type]
        open_nvml=open_nvml,
        load_whisper=load_whisper,
        load_procs=load_procs,
    )


class ReleaseRecording:
    """The run's reranker, with its `release` recorded among the GPU events."""

    def __init__(self, inner: Any, state: GpuState) -> None:
        self.inner = inner
        self.state = state

    def score(self, query: str, texts: Any, budget_ms: float | None = None) -> list[float]:
        return self.inner.score(query, texts, budget_ms)

    def pairs_within(self, budget_ms: float) -> int:
        return self.inner.pairs_within(budget_ms)

    def release(self) -> None:
        self.state.events.append(("reranker_release",))
        self.inner.release()

    def reload(self) -> bool:
        return self.inner.reload()


@pytest.fixture
def gpu_reranker(monkeypatch: pytest.MonkeyPatch) -> list[GpuState]:
    """Records each reranker build and release into the state the test registers first."""
    states: list[GpuState] = []
    build_reranker = harness.build_reranker

    def build(choice: str, *, device: str, config: object) -> harness.BuiltReranker:
        built = build_reranker(choice, device=device, config=config)  # type: ignore[arg-type]
        states[0].events.append(("reranker_load", device))
        return replace(built, reranker=ReleaseRecording(built.reranker, states[0]))

    monkeypatch.setattr(harness, "build_reranker", build)
    return states


def _vram_run(
    tmp_path: Path, state: GpuState, gpu_reranker: list[GpuState], *extra: str
) -> dict[str, Any]:
    gpu_reranker.append(state)
    log = tmp_path / "vram.json"
    argv = [
        "--output",
        str(tmp_path / "run.jsonl"),
        "--device",
        "cuda",
        "--embedder-device",
        "cpu",
        "--vram-log",
        str(log),
        "--deadline-ms",
        "10000",
        *extra,
    ]
    assert main(argv, gpu=_access(state)) == 0
    return json.loads(log.read_text(encoding="utf-8"))


_READING_KEYS = {
    "nvml_used",
    "mem_get_info_used",
    "memory_allocated",
    "max_memory_allocated",
    "memory_reserved",
    "max_memory_reserved",
}


# -- the VRAM log -------------------------------------------------------------------------------


def test_without_stt_the_points_come_in_order_with_their_derived_figures(
    tmp_path: Path, gpu_reranker: list[GpuState]
) -> None:
    state = GpuState(nvml_mib=[1000, 1300, 1500, 1600, 1350])

    log = _vram_run(tmp_path, state, gpu_reranker)

    assert list(log["points"]) == ["before_cuda", "context", "loaded", "after_run", "released"]
    assert log["points"]["before_cuda"] == {"nvml_used": 1000.0}
    assert log["points"]["context"] == {
        "nvml_used": 1300.0,
        "mem_get_info_used": 6144.0,
        "memory_allocated": 100.0,
        "max_memory_allocated": 150.0,
        "memory_reserved": 200.0,
        "max_memory_reserved": 250.0,
    }
    assert all(set(log["points"][p]) == _READING_KEYS for p in list(log["points"])[1:])
    assert log["derived"] == {
        "process_mib": 600.0,
        "context_mib": 300.0,
        "stt_mib": None,
        "marginal_mib": 300.0,
        "residual_mib": 350.0,
    }
    assert (log["unit"], log["device"], log["embedder_device"], log["vram_stt"]) == (
        "MiB",
        "cuda",
        "cpu",
        None,
    )
    names = state.names()
    reads = state.nvml_reads()
    # The reranker loads between the context reading and the loaded reading.
    assert reads[1] < names.index("reranker_load") < reads[2]
    assert "whisper_load" not in names and "transcribe" not in names


def test_with_stt_the_whisper_point_comes_before_the_reranker(
    tmp_path: Path, gpu_reranker: list[GpuState]
) -> None:
    state = GpuState(nvml_mib=[1000, 1300, 1700, 1900, 2000, 1350])

    log = _vram_run(tmp_path, state, gpu_reranker, "--vram-stt", "base")

    assert list(log["points"]) == [
        "before_cuda",
        "context",
        "stt",
        "loaded",
        "after_run",
        "released",
    ]
    assert log["derived"] == {
        "process_mib": 1000.0,
        "context_mib": 300.0,
        "stt_mib": 400.0,
        "marginal_mib": 300.0,
        "residual_mib": 350.0,
    }
    assert log["vram_stt"] == "base"
    names, reads = state.names(), state.nvml_reads()
    assert reads[1] < names.index("whisper_load") < reads[2] < names.index("reranker_load")


def test_before_cuda_is_read_before_any_cuda_call(
    tmp_path: Path, gpu_reranker: list[GpuState]
) -> None:
    state = GpuState(nvml_mib=[1000, 1300, 1700, 1900, 2000, 1350])

    _vram_run(tmp_path, state, gpu_reranker, "--vram-stt", "base")

    first = state.nvml_reads()[0]
    assert not any(name in _CUDA_CALLS for name in state.names()[:first])


def test_the_context_is_made_by_a_tensor_and_a_synchronize(
    tmp_path: Path, gpu_reranker: list[GpuState]
) -> None:
    state = GpuState(nvml_mib=[1000, 1300, 1500, 1600, 1350])

    _vram_run(tmp_path, state, gpu_reranker)

    reads = state.nvml_reads()
    between = state.events[reads[0] + 1 : reads[1]]
    tensor = between.index(("ones", (1,), "cuda"))
    assert ("synchronize",) in between[tensor + 1 :]
    assert ("init",) not in state.events


def test_whisper_transcribes_one_clip_without_fp16_before_the_stt_reading(
    tmp_path: Path, gpu_reranker: list[GpuState]
) -> None:
    state = GpuState(nvml_mib=[1000, 1300, 1700, 1900, 2000, 1350])

    _vram_run(tmp_path, state, gpu_reranker, "--vram-stt", "base")

    names, reads = state.names(), state.nvml_reads()
    transcribes = [event for event in state.events if event[0] == "transcribe"]
    assert transcribes == [("transcribe", (30 * 16000,), "float32", {"fp16": False})]
    assert names.index("whisper_load") < names.index("transcribe") < reads[2]
    assert ("whisper_load", "base") in state.events


def test_released_is_read_after_both_releases(tmp_path: Path, gpu_reranker: list[GpuState]) -> None:
    state = GpuState(nvml_mib=[1000, 1300, 1700, 1900, 2000, 1350])

    _vram_run(tmp_path, state, gpu_reranker, "--vram-stt", "base")

    names, reads = state.names(), state.nvml_reads()
    released = reads[-1]
    assert state.events[reads[-2]] == ("nvml", True)  # after_run: Whisper still held
    assert state.events[released] == ("nvml", False)  # the Whisper model is gone
    assert reads[-2] < names.index("reranker_release") < released
    assert "empty_cache" in names[names.index("reranker_release") : released]
    assert names[-1] == "nvml_close"


def test_the_stt_clip_is_fixed() -> None:
    clip = harness.stt_clip()

    assert clip.shape == (30 * 16000,) and clip.dtype == np.float32
    assert np.array_equal(clip, harness.stt_clip())


_VRAM = ["--vram-log", "vram.json"]
_SPLIT = ["--device", "cuda", "--embedder-device", "cpu"]


@pytest.mark.parametrize(
    ("argv", "available", "nvml_ok"),
    [
        pytest.param([*_SPLIT, *_VRAM], False, True, id="no-cuda"),
        pytest.param([*_SPLIT, *_VRAM], True, False, id="no-nvml"),
        pytest.param(["--embedder-device", "cuda", *_VRAM], True, True, id="embedder-cuda"),
        pytest.param(["--device", "cuda", *_VRAM], True, True, id="embedder-from-device"),
        pytest.param([*_SPLIT, "--gpu-load-procs", "1", *_VRAM], True, True, id="gpu-load"),
        pytest.param(["--device", "cuda", "--vram-stt", "base"], True, True, id="stt-alone"),
    ],
)
def test_the_vram_log_is_refused_with_exit_2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    available: bool,
    nvml_ok: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    state = GpuState(nvml_mib=[1000], available=available, nvml_ok=nvml_ok)

    with pytest.raises(SystemExit) as exited:
        main(["--output", "run.jsonl", *argv], gpu=_access(state))

    assert exited.value.code == 2
    assert not (tmp_path / "run.jsonl").exists() and not (tmp_path / "vram.json").exists()
    assert not {"ones", "whisper_load", "gpu_load_start"} & set(state.names())


def test_the_gpu_load_is_refused_with_exit_2_without_cuda(tmp_path: Path) -> None:
    state = GpuState(nvml_mib=[], available=False)
    output = tmp_path / "run.jsonl"

    with pytest.raises(SystemExit) as exited:
        main(["--output", str(output), "--gpu-load-procs", "2"], gpu=_access(state))

    assert exited.value.code == 2
    assert "gpu_load_start" not in state.names() and not output.exists()


def test_the_gpu_load_runs_for_the_whole_run(tmp_path: Path) -> None:
    state = GpuState(nvml_mib=[])
    output = tmp_path / "run.jsonl"

    assert main(["--output", str(output), "--gpu-load-procs", "2"], gpu=_access(state)) == 0

    assert [e for e in state.events if e[0].startswith("gpu_load")] == [
        ("gpu_load_start", 2),
        ("gpu_load_stop",),
    ]
    assert [r["probe_id"] for r in _records(output)] == [p.probe_id for p in load_probes()]


# -- NVML through pynvml ------------------------------------------------------------------------


class FakeNVMLError(Exception):
    pass


def _pynvml(*, init_fails: bool) -> SimpleNamespace:
    calls: list[str] = []

    def init() -> None:
        calls.append("init")
        if init_fails:
            raise FakeNVMLError("driver not loaded")

    return SimpleNamespace(
        calls=calls,
        NVMLError=FakeNVMLError,
        nvmlInit=init,
        nvmlDeviceGetHandleByIndex=lambda index: ("handle", index),
        nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(used=512 * _MIB),
        nvmlShutdown=lambda: calls.append("shutdown"),
    )


def test_nvml_reads_device_wide_used_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _pynvml(init_fails=False)
    monkeypatch.setitem(sys.modules, "pynvml", module)

    reader = harness.PynvmlReader.open()
    used = reader.used_bytes()
    reader.close()

    assert used == 512 * _MIB and module.calls == ["init", "shutdown"]


@pytest.mark.parametrize("module", [None, _pynvml(init_fails=True)], ids=["absent", "no-driver"])
def test_nvml_unavailable_is_reported(monkeypatch: pytest.MonkeyPatch, module: object) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", module)

    with pytest.raises(harness.GpuUnavailableError, match="NVML"):
        harness.PynvmlReader.open()


# -- the need gate ------------------------------------------------------------------------------

_SHA = "a" * 40
_BGE = "bge-small-en-v1.5"
_GRANITE = "granite-embedding-small-english-r2"
_MINILM = "all-MiniLM-L6-v2"
_LATENCY = ("critical_path_ms", "timings_ms")
_NEED_FIELDS = {
    "mode",
    "band",
    "p",
    "theta_low",
    "theta_high",
    "band_k",
    "need_ids",
    "no_need_ids",
    "context",
    "context_p",
    "degraded",
}


def _model_file(
    path: Path,
    *,
    name: str,
    revision: str,
    prefix: str = "",
    b: float = 0.0,
    theta_low: float = 0.3,
    theta_high: float = 0.7,
) -> Path:
    """A gate model file of one logistic record keyed to the shipped bank.

    Its weights are zero, so every query's `p` is `sigmoid(b)`: 0.5 at `b` 0 (band), above
    0.99 at 5 (open) and below 0.01 at -5 (closed).
    """
    key = NeedKey(load_bank().sha256, name, revision, prefix)
    record = {
        "key": need_model_key(key),
        "bank_sha256": key.bank_sha256,
        "embedder": {"name": name, "revision": revision},
        "query_prefix": prefix,
        "normalisation": "l2",
        "form": "logistic",
        "params": {"c": 1.0, "w": [0.0] * 384, "b": b},
        "theta_low": theta_low,
        "theta_high": theta_high,
        "band_k": 2,
        "recall_target": 0.98,
        "source": "test",
    }
    path.write_text(json.dumps({"records": [record]}), encoding="utf-8")
    return path


def _fake_model(tmp_path: Path, *, b: float, revision: str = FAKE_EMBEDDER_REVISION) -> Path:
    return _model_file(
        tmp_path / "need_model.json", name=FAKE_EMBEDDER_NAME, revision=revision, b=b
    )


def _lock(tmp_path: Path, *names: str) -> Path:
    """A models.lock holding one entry per name, each at revision `_SHA`."""
    tables = [
        "\n".join(
            [
                "[[model]]",
                f'name = "{name}"',
                f'repo = "owner/{name}"',
                f'revision = "{_SHA}"',
                'license = "Apache-2.0"',
                f'target = "/opt/models/{name}"',
            ]
        )
        for name in names
    ]
    path = tmp_path / "models.lock"
    path.write_text("\n\n".join(tables) + "\n", encoding="utf-8")
    return path


def _probe_file(tmp_path: Path) -> Path:
    """Three fact probes, two note probes and two noise probes of the fixture."""
    lines = (FIXTURE_DIR / "probes.jsonl").read_text(encoding="utf-8").splitlines()
    chosen: list[str] = []
    for kind, count in (("fact", 3), ("note", 2), ("noise", 2)):
        chosen += [line for line in lines if json.loads(line)["kind"] == kind][:count]
    path = tmp_path / "gate_probes.jsonl"
    path.write_text("\n".join(chosen) + "\n", encoding="utf-8")
    return path


def _gate_argv(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--output",
        str(tmp_path / "run.jsonl"),
        "--probes",
        str(_probe_file(tmp_path)),
        "--deadline-ms",
        "10000",
        *extra,
    ]


def _exit_code(argv: list[str]) -> int | str | None:
    with pytest.raises(SystemExit) as exited:
        main(argv)
    return exited.value.code


def test_the_need_flags_and_the_fact_view_reach_the_config() -> None:
    args = harness._parser().parse_args(
        [
            "--output",
            "run.jsonl",
            "--need-gate",
            "shadow",
            "--need-context",
            "off",
            "--fact-view",
            "F1",
            "--need-embedder",
            f"/opt/models/{_BGE}",
            "--need-embedder-name",
            _BGE,
            "--need-embedder-revision",
            _SHA,
        ]
    )

    config = harness.config_from_args(args)

    assert (config.need_gate, config.need_context, config.fact_view, config.need_embedder) == (
        "shadow",
        False,
        "F1",
        _BGE,
    )


def test_without_need_flags_the_config_keeps_the_environments_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MIST_MEMORY_NEED_GATE", "on")
    monkeypatch.setenv("MIST_MEMORY_NEED_CONTEXT", "false")
    args = harness._parser().parse_args(["--output", "run.jsonl"])

    config = harness.config_from_args(args)

    assert (config.need_gate, config.need_context, config.need_embedder) == (
        "on",
        False,
        "shared",
    )


def test_the_fact_view_reaches_the_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    views: list[object] = []
    base = harness.FactIndex

    class RecordingIndex(base):  # type: ignore[misc, valid-type]
        def __init__(self, *args: object, **kwargs: object) -> None:
            views.append(kwargs.get("view"))
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(harness, "FactIndex", RecordingIndex)

    assert main(_gate_argv(tmp_path, "--fact-view", "F1")) == 0

    assert views == [FACT_VIEWS["F1"]]


@pytest.mark.parametrize("name", [_BGE, _GRANITE])
def test_a_gate_embedder_is_built_with_exactly_its_specs_keywords(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    made: list[tuple[tuple[object, ...], dict[str, object], list[list[str]]]] = []

    class RecordingGateEmbedder(TokenEmbedder):
        """`EmbeddingGenerator` as the harness calls it; vectors are the token embedder's."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            self.batches: list[list[str]] = []
            made.append((args, kwargs, self.batches))

        def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
            self.batches.append(list(texts))
            return super().generate_embeddings(texts)

    monkeypatch.setattr(harness, "EmbeddingGenerator", RecordingGateEmbedder)
    spec = NEED_EMBEDDERS[name]
    path = f"/opt/models/{name}"
    model = _model_file(
        tmp_path / "model.json", name=name, revision=_SHA, prefix=spec.query_prefix, b=5.0
    )

    code = main(
        _gate_argv(
            tmp_path,
            "--need-gate",
            "on",
            "--need-embedder",
            path,
            "--need-embedder-name",
            name,
            "--need-embedder-revision",
            _SHA,
            "--need-model",
            str(model),
        )
    )

    assert code == 0
    assert [(args, kwargs) for args, kwargs, _ in made] == [
        ((path,), {"device": "cpu", **need_embedder_kwargs(spec)})
    ]
    # bge-small has no attention setting, so its load gets no keyword; granite-r2 gets sdpa.
    assert ("attn_implementation" in made[0][1]) == (spec.attn_implementation is not None)
    bank_texts = made[0][2][0]
    assert len(bank_texts) == len(load_bank().exemplars)
    assert all(text.startswith(spec.query_prefix) for text in bank_texts)
    records = _records(tmp_path / "run.jsonl")
    assert {(r["need"]["mode"], r["need"]["band"]) for r in records} == {("on", "open")}


def test_shared_takes_its_name_and_revision_from_the_models_lock_with_the_real_embedder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The real embedder is not loaded: the lock's entry is what the test is about.
    monkeypatch.setattr(harness, "build_embedder", lambda choice, **kwargs: TokenEmbedder())
    lock = _lock(tmp_path, "ms-marco-MiniLM-L-6-v2", _MINILM)
    keyed_to_lock = _model_file(tmp_path / "lock_model.json", name=_MINILM, revision=_SHA, b=5.0)
    keyed_to_fake = _fake_model(tmp_path, b=5.0)

    def argv(model: Path) -> list[str]:
        return _gate_argv(
            tmp_path,
            "--embedder",
            "real",
            "--models-lock",
            str(lock),
            "--need-gate",
            "on",
            "--need-model",
            str(model),
        )

    assert main(argv(keyed_to_lock)) == 0
    assert {r["need"]["band"] for r in _records(tmp_path / "run.jsonl")} == {"open"}
    assert _exit_code(argv(keyed_to_fake)) == 2
    assert "--need-model" in capsys.readouterr().err


def test_shared_takes_the_fake_constants_with_the_fake_embedder(tmp_path: Path) -> None:
    # The lock is not read for the fake embedder: it does not exist.
    code = main(
        _gate_argv(
            tmp_path,
            "--models-lock",
            str(tmp_path / "absent.lock"),
            "--need-gate",
            "on",
            "--need-model",
            str(_fake_model(tmp_path, b=5.0)),
        )
    )

    assert code == 0
    assert {r["need"]["band"] for r in _records(tmp_path / "run.jsonl")} == {"open"}


@pytest.mark.parametrize("names", [("ms-marco-MiniLM-L-6-v2",), None], ids=["no-entry", "absent"])
def test_a_lock_without_the_fact_embedder_exits_2_naming_models_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    names: tuple[str, ...] | None,
) -> None:
    monkeypatch.setattr(harness, "build_embedder", lambda choice, **kwargs: TokenEmbedder())
    lock = _lock(tmp_path, *names) if names is not None else tmp_path / "absent.lock"
    model = _model_file(tmp_path / "lock_model.json", name=_MINILM, revision=_SHA, b=5.0)

    code = _exit_code(
        _gate_argv(
            tmp_path,
            "--embedder",
            "real",
            "--models-lock",
            str(lock),
            "--need-gate",
            "on",
            "--need-model",
            str(model),
        )
    )

    assert code == 2
    assert "--models-lock" in capsys.readouterr().err
    assert not (tmp_path / "run.jsonl").exists()


@pytest.mark.parametrize(
    "extra",
    [
        ["--need-embedder-name", _BGE],
        ["--need-embedder-revision", _SHA],
        ["--need-embedder", "shared", "--need-embedder-name", _BGE],
    ],
    ids=["name", "revision", "explicit-shared"],
)
def test_a_name_or_revision_with_shared_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extra: list[str]
) -> None:
    assert _exit_code(_gate_argv(tmp_path, *extra)) == 2
    assert "shared" in capsys.readouterr().err
    assert not (tmp_path / "run.jsonl").exists()


@pytest.mark.parametrize(
    "extra",
    [[], ["--need-embedder-name", _BGE], ["--need-embedder-revision", _SHA]],
    ids=["neither", "no-revision", "no-name"],
)
def test_a_path_without_its_name_and_revision_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extra: list[str]
) -> None:
    argv = _gate_argv(tmp_path, "--need-embedder", f"/opt/models/{_BGE}", *extra)

    assert _exit_code(argv) == 2
    err = capsys.readouterr().err
    assert "--need-embedder-name" in err and "--need-embedder-revision" in err
    assert not (tmp_path / "run.jsonl").exists()


@pytest.mark.parametrize("mode", ["on", "shadow"])
@pytest.mark.parametrize("model", ["other-key", "absent"])
def test_a_gate_without_a_record_for_its_key_exits_2_naming_need_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], mode: str, model: str
) -> None:
    path = (
        _fake_model(tmp_path, b=5.0, revision="another-revision")
        if model == "other-key"
        else tmp_path / "absent.json"
    )

    code = _exit_code(_gate_argv(tmp_path, "--need-gate", mode, "--need-model", str(path)))

    assert code == 2
    assert "--need-model" in capsys.readouterr().err
    assert not (tmp_path / "run.jsonl").exists()


def test_with_the_gate_off_neither_the_bank_nor_the_model_file_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    absent = tmp_path / "absent"
    monkeypatch.setattr(
        need_gate_module,
        "DEFAULT_BANK_PATHS",
        (absent / "writer_a.jsonl", absent / "writer_b.jsonl"),
    )
    model = absent / "need_model.json"

    assert main(_gate_argv(tmp_path, "--need-gate", "off", "--need-model", str(model))) == 0

    assert all(r["need"] is None for r in _records(tmp_path / "run.jsonl"))
    # The same paths are read, and refused, once the gate is on.
    assert _exit_code(_gate_argv(tmp_path, "--need-gate", "on", "--need-model", str(model))) == 2


def test_a_record_carries_the_need_trace_and_the_stage_ids(tmp_path: Path) -> None:
    model = _fake_model(tmp_path, b=0.0)

    assert main(_gate_argv(tmp_path, "--need-gate", "on", "--need-model", str(model))) == 0

    records = _records(tmp_path / "run.jsonl")
    for record in records:
        need = record["need"]
        assert set(need) == _NEED_FIELDS
        assert (need["mode"], need["band"], need["band_k"]) == ("on", "band", 2)
        assert need["p"] == pytest.approx(0.5)
        assert (need["theta_low"], need["theta_high"]) == (0.3, 0.7)
        assert len(need["need_ids"]) == len(need["no_need_ids"]) == 3
        assert (need["context"], need["context_p"], need["degraded"]) == (False, None, None)
        # A band turn is served at most `band_k` block entries.
        assert len({item["ref"] for item in record["items"]}) <= 2
        assert "cuts" not in record
        stage_ids = record["stage_ids"]
        assert set(stage_ids) == {"fused", "shortlisted", "selected"}
        for ids in stage_ids.values():
            assert all(
                isinstance(i, str) and i.split(":", 1)[0] in {"fact", "facet", "note"} for i in ids
            )
        assert set(stage_ids["shortlisted"]) <= set(stage_ids["fused"])
    assert any(r["stage_ids"]["selected"] for r in records)


def test_a_closed_gate_serves_no_block(tmp_path: Path) -> None:
    model = _fake_model(tmp_path, b=-5.0)

    assert main(_gate_argv(tmp_path, "--need-gate", "on", "--need-model", str(model))) == 0

    for record in _records(tmp_path / "run.jsonl"):
        assert record["need"]["band"] == "closed"
        assert record["items"] == [] and record["tokens"] == 0
        assert record["stage_ids"] == {"fused": [], "shortlisted": [], "selected": []}


def test_a_shadow_record_carries_each_cut_with_its_items_and_tokens(tmp_path: Path) -> None:
    model = _fake_model(tmp_path, b=0.0)

    assert main(_gate_argv(tmp_path, "--need-gate", "shadow", "--need-model", str(model))) == 0

    records = _records(tmp_path / "run.jsonl")
    assert {(r["need"]["mode"], r["need"]["band"]) for r in records} == {("shadow", "band")}
    with_block = [r for r in records if r["tokens"] > 0]
    assert with_block
    # Served as open: some shadow band turn holds more than `band_k` entries.
    assert any(len({item["ref"] for item in r["items"]}) > 2 for r in with_block)
    for record in records:
        if record["tokens"] == 0:
            assert "cuts" not in record
            continue
        cuts = record["cuts"]
        assert set(cuts) == {"1", "2"}
        for k, cut in cuts.items():
            assert set(cut) == {"items", "tokens"}
            # Not bounded by the open block's count: a cut that drops an entry adds a line.
            assert isinstance(cut["tokens"], int) and cut["tokens"] > 0
            assert 1 <= len({item["ref"] for item in cut["items"]}) <= int(k)
            assert all(set(item) == {"ref", "kind", "id", "score", "rank"} for item in cut["items"])


# -- run_settings ------------------------------------------------------------------------------

_BASE = MemoryConfig(deadline_ms=10_000, need_gate="off")


def _fake_gate(tmp_path: Path, *, b: float) -> NeedGate:
    model = load_models(_fake_model(tmp_path, b=b))[0]
    return NeedGate(
        TokenEmbedder(),
        load_bank(),
        model,
        embedder_name=FAKE_EMBEDDER_NAME,
        embedder_revision=FAKE_EMBEDDER_REVISION,
        query_prefix="",
        shares_query_vector=True,
    )


def _settings(tmp_path: Path) -> tuple[RunSetting, RunSetting]:
    """An ungated setting and a gated one that selects fewer candidates."""
    gated = replace(_BASE, need_gate="on", select_k=5)
    return RunSetting(_BASE, None), RunSetting(gated, _fake_gate(tmp_path, b=0.0))


def _some_probes(fixture: Fixture) -> tuple[object, ...]:
    """Every tenth probe and every probe at a later `now` than the first, without repeats."""
    first = fixture.probes[0].now
    chosen = [*fixture.probes[::10], *(p for p in fixture.probes if p.now != first)]
    return tuple({probe.probe_id: probe for probe in chosen}.values())


def _strip(records: list[dict[str, object]]) -> list[dict[str, object]]:
    return [{k: v for k, v in record.items() if k not in _LATENCY} for record in records]


def _walk(tmp_path: Path, name: str, settings: tuple[RunSetting, ...], **kwargs: Any) -> Any:
    workdir = tmp_path / name
    workdir.mkdir()
    fixture = kwargs.pop("fixture")
    return asyncio.run(
        run_settings(
            fixture,
            settings=settings,
            embedder=TokenEmbedder(),
            reranker=TokenReranker(),
            workdir=workdir,
            **kwargs,
        )
    )


def _fetches(monkeypatch: pytest.MonkeyPatch) -> list[tuple[object, MemoryRequest]]:
    """Every `fetch_outcome` call of the run: the fetcher and its request, in order."""
    seen: list[tuple[object, MemoryRequest]] = []
    original = harness.MemoryFetcher.fetch_outcome

    async def fetch_outcome(self: object, request: MemoryRequest) -> object:
        seen.append((self, request))
        return await original(self, request)  # type: ignore[arg-type]

    monkeypatch.setattr(harness.MemoryFetcher, "fetch_outcome", fetch_outcome)
    return seen


def test_two_settings_in_one_walk_equal_two_separate_runs(tmp_path: Path) -> None:
    fixture = Fixture.load()
    probes = _some_probes(fixture)
    settings = _settings(tmp_path)

    together = _walk(tmp_path, "together", settings, fixture=fixture, probes=probes)
    apart = []
    for n, setting in enumerate(settings):
        workdir = tmp_path / f"apart-{n}"
        workdir.mkdir()
        apart.append(
            asyncio.run(
                run_fixture(
                    fixture,
                    config=setting.config,
                    need_gate=setting.need_gate,
                    embedder=TokenEmbedder(),
                    reranker=TokenReranker(),
                    workdir=workdir,
                    probes=probes,
                )
            )
        )

    assert [_strip(records) for records in together] == [_strip(records) for records in apart]
    assert [len(records) for records in together] == [len(probes)] * 2
    # The settings differ in what they serve, so the equality is not of two copies of one.
    assert _strip(together[0]) != _strip(together[1])


def test_settings_of_two_fact_views_are_refused(tmp_path: Path) -> None:
    settings = (RunSetting(_BASE, None), RunSetting(replace(_BASE, fact_view="F2"), None))

    with pytest.raises(ValueError, match="fact view"):
        _walk(tmp_path, "views", settings, fixture=Fixture.load())


def test_a_setting_has_a_gate_exactly_when_its_config_asks_for_one(tmp_path: Path) -> None:
    gate = _fake_gate(tmp_path, b=0.0)

    with pytest.raises(ValueError, match="need_gate"):
        RunSetting(_BASE, gate)
    for mode in ("on", "shadow"):
        with pytest.raises(ValueError, match="need_gate"):
            RunSetting(replace(_BASE, need_gate=mode), None)
        assert RunSetting(replace(_BASE, need_gate=mode), gate).need_gate is gate


def test_warm_up_runs_before_each_now_and_records_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = Fixture.load()
    probes = _some_probes(fixture)
    settings = _settings(tmp_path)
    seen = _fetches(monkeypatch)

    warm = _walk(tmp_path, "warm", settings, fixture=fixture, probes=probes, warm_up=True)
    with_warm_up = list(seen)
    seen.clear()
    cold = _walk(tmp_path, "cold", settings, fixture=fixture, probes=probes)

    assert [_strip(records) for records in warm] == [_strip(records) for records in cold]
    ups = [
        (fetcher, request)
        for fetcher, request in with_warm_up
        if request.query_text == harness.WARM_UP_QUERY
    ]
    nows = sorted({probe.now for probe in probes})
    assert [request.now for _, request in ups] == nows
    assert len(with_warm_up) == len(seen) + len(nows)
    # Each warm-up goes through the first setting's fetcher, before that `now`'s probes.
    first_fetcher = next(f for f, r in with_warm_up if r.query_text != harness.WARM_UP_QUERY)
    assert all(fetcher is first_fetcher for fetcher, _ in ups)
    assert with_warm_up[0][1].query_text == harness.WARM_UP_QUERY
    assert len({request.session_id for _, request in ups}) == len(nows)


def test_each_followup_runs_two_fetches_per_setting_in_its_own_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = Fixture.load()
    probes = _some_probes(fixture)[:5]
    followups = (
        ("cat", "I adopted a cat called Miso last week", "what does she like to eat?"),
        ("trip", "We are planning the coast trip again", "where did we stay last time?"),
    )
    seen = _fetches(monkeypatch)

    runs = _walk(
        tmp_path,
        "followups",
        _settings(tmp_path),
        fixture=fixture,
        probes=probes,
        followups=followups,
    )

    last = max(probe.now for probe in fixture.probes)
    for n, records in enumerate(runs):
        assert [r["probe_id"] for r in records[: len(probes)]] == [p.probe_id for p in probes]
        tail = records[len(probes) :]
        assert [(r["followup_id"], r["turn"]) for r in tail] == [
            ("cat", "previous"),
            ("cat", "query"),
            ("trip", "previous"),
            ("trip", "query"),
        ]
        for record in tail:
            assert set(record) == {"followup_id", "turn", "need", *_LATENCY, *_FETCH_LATENCY}
            assert (record["need"] is None) == (n == 0)
    for followup_id, previous, query in followups:
        calls = [(f, r) for f, r in seen if r.session_id == f"followup-{followup_id}"]
        assert len(calls) == 4
        fetchers = [f for f, _ in calls]
        assert fetchers[0] is fetchers[1] and fetchers[2] is fetchers[3]
        assert fetchers[0] is not fetchers[2]
        for pair in (calls[:2], calls[2:]):
            (_, first), (_, second) = pair
            assert (first.query_text, first.recent_user_turns) == (previous, ())
            assert (second.query_text, second.recent_user_turns) == (query, (previous,))
            assert first.now == second.now == last


_FETCH_LATENCY = ("degraded_reason", "reranked", "shortlisted")
_LATE_S = 3.0


class _LateFor:
    """A gate embedder that is late for one text: it sleeps past the fetch's deadline."""

    def __init__(self, inner: TokenEmbedder, late_text: str) -> None:
        self._inner = inner
        self._late_text = late_text

    def generate_embedding(self, text: str) -> list[float]:
        if text == self._late_text:
            time.sleep(_LATE_S)
        return self._inner.generate_embedding(text)

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [self.generate_embedding(text) for text in texts]


def test_a_followup_record_carries_the_latency_fields_run_record_gives_its_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The selection grid's load check takes the deadline share and the rerank depth over
    # follow-up fetches too (v7:416-418), so a follow-up record holds what `run_record` holds.
    fixture = Fixture.load()
    previous = next(p.query for p in fixture.probes if p.kind == "fact" and p.lexical)
    late = "and the one before that?"
    gate = NeedGate(
        _LateFor(TokenEmbedder(), late),
        load_bank(),
        load_models(_fake_model(tmp_path, b=5.0))[0],
        embedder_name=FAKE_EMBEDDER_NAME,
        embedder_revision=FAKE_EMBEDDER_REVISION,
        query_prefix="",
    )
    config = replace(_BASE, need_gate="on", deadline_ms=int(_LATE_S * 1000) - 1000)
    outcomes: list[tuple[MemoryRequest, object]] = []
    original = harness.MemoryFetcher.fetch_outcome

    async def fetch_outcome(self: object, request: MemoryRequest) -> object:
        outcome = await original(self, request)  # type: ignore[arg-type]
        outcomes.append((request, outcome))
        return outcome

    monkeypatch.setattr(harness.MemoryFetcher, "fetch_outcome", fetch_outcome)

    (records,) = _walk(
        tmp_path,
        "followup-latency",
        (RunSetting(config, gate),),
        fixture=fixture,
        probes=fixture.probes[:1],
        followups=(("late", previous, late),),
    )

    tail = records[1:]
    fetched = [outcome for request, outcome in outcomes if request.session_id == "followup-late"]
    assert [record["turn"] for record in tail] == ["previous", "query"]
    for record, outcome in zip(tail, fetched, strict=True):
        expected = harness.outcome_record(
            fixture.probes[0],
            outcome,  # type: ignore[arg-type]
            counter=harness.ApproximateTokenCounter(),
            cpu_load_procs=0,
            embedder_device="cpu",
        )
        assert {k: record[k] for k in _FETCH_LATENCY} == {k: expected[k] for k in _FETCH_LATENCY}
    shortlist, late_fetch = tail
    # The open previous turn reranks a shortlist; the query's gate vector misses the deadline.
    assert shortlist["degraded_reason"] is None
    assert shortlist["reranked"] > 0 and shortlist["shortlisted"] > 0
    assert late_fetch["degraded_reason"] == "deadline"
    assert late_fetch["need"]["degraded"] == "deadline"
    assert late_fetch["shortlisted"] == 0
