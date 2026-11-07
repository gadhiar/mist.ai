"""`--cpu-load-procs`: the busy-loop processes start for the run and stop after it."""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import pytest

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
