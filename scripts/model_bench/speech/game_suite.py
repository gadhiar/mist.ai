"""Runs 8 and 9: the guided `game-session`, what MIST's low level costs a game Raj plays.

Raj runs it in an administrator terminal. The session asks for the game's process name, makes a
bench run id of its own (`<run>-game-<UTC time>`, refused if its `session/` exists), stops MIST
with bench_host's `snapshot` and `stop --with-neo4j`, and walks the conditions, 3 passes each:

- With `--with-1070`, first `G1-gtx1070-shared` and `G2-gtx1070-shared`, against host-1070's game
  alias, skipped when host-1070 is not serving it.
- `G0`: the game alone.
- `G1-<slice>` and `G2-<slice>`: the largest of `gpu-full`, `gpu-4g` and `gpu-2g` whose run 7
  footprint leaves 512 MiB beside G0's peak memory, then `G1-cpu` and `G2-cpu`.

G1 loads the low level (the model, Parakeet and Pocket TTS on the CPU) and leaves it idle; G2 adds
10 chained turns, one every 12 s, while Raj plays. Before each pass Raj is prompted to start the
scene, then 120 s are captured: PresentMon (or the game's own benchmark figures with
`--manual-fps`, or a typed frame rate when a 10 s PresentMon test capture records nothing),
nvidia-smi at 2 Hz with `utilization.gpu` (this module's own query; `probes/nvidia_smi.py` is
left as it is), and `typeperf` at 1 Hz, with the server process's counters for run 9's
`--server-process`. MIST is restored at the end, and E4B's return is timed from the start of the
restore to its first warm-up token.

stdlib only: it runs on the Windows host. Nothing about other players is recorded.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import io
import os
import subprocess
import time
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from .. import bench_host
from . import chain, core, frametimes, host1070_script, lowtier_suite, tts_suite

SUITE = "game"
PASSES = 3
CAPTURE_S = 120
TEST_CAPTURE_S = 10
TURN_EVERY_S = 12.0
GAME_TURNS: tuple[str, ...] = chain.CHAIN_UTTERANCES[:10]
MARGIN_MIB = 512
SLICE_ORDER = ("gpu-full", "gpu-4g", "gpu-2g")
GPU_PLACEMENT_GROUP = "gpu"
CPU_PLACEMENT = "cpu"
REMOTE_PLACEMENT = host1070_script.SHARED
SKIPPABLE = (GPU_PLACEMENT_GROUP, CPU_PLACEMENT, REMOTE_PLACEMENT)
GPU_QUERY_FIELDS = (
    "timestamp", "memory.used", "memory.total", "utilization.gpu", "clocks.sm", "clocks.mem",
    "power.draw",
)  # fmt: skip
GPU_SAMPLE_MS = 500
TOTAL_CPU_COUNTER = r"\Processor(_Total)\% Processor Time"
E4B_URL = "http://127.0.0.1:8080"
E4B_TIMEOUT_S = 900.0
DEFAULT_REMOTE_URL = "http://mist-extraction-gtx1070:8090"
PROCESS_GRACE_S = 30.0
ASK_ATTEMPTS = 3

Clock = Callable[[], float]


class GameSessionError(core.SpeechBenchError):
    """The session cannot start, or MIST did not come back cleanly."""


# ---------------------------------------------------------------------------
# argv for the three capture tools
# ---------------------------------------------------------------------------


def new_bench_run(run: str, now: datetime) -> str:
    """The session's own bench run id, from the UTC time to the second."""
    return f"{core.validate_name(run)}-game-{now.astimezone(UTC):%Y%m%dT%H%M%SZ}"


def presentmon_argv(presentmon: Path, process: str, out: Path, seconds: int) -> list[str]:
    """PresentMon's timed capture of one process, ending by itself."""
    return [
        str(presentmon), "--process_name", process, "--output_file", str(out), "--timed", str(seconds),
        "--terminate_after_timed",
    ]  # fmt: skip


def process_counters(server_process: str) -> tuple[str, str]:
    """Typeperf's CPU and private memory counters of one process (its name without `.exe`)."""
    name = server_process.removesuffix(".exe")
    return (rf"\Process({name})\% Processor Time", rf"\Process({name})\Working Set - Private")


def typeperf_argv(out: Path, seconds: int, server_process: str | None = None) -> list[str]:
    """Typeperf sampling total CPU (and a server's counters) at 1 Hz for `seconds` samples."""
    counters = [TOTAL_CPU_COUNTER]
    if server_process:
        counters += process_counters(server_process)
    return [
        "typeperf",
        *counters,
        "-si",
        "1",
        "-sc",
        str(seconds),
        "-f",
        "CSV",
        "-o",
        str(out),
        "-y",
    ]


def nvidia_smi_argv(out: Path) -> list[str]:
    """nvidia-smi at 2 Hz with utilisation, written to `out` until it is stopped."""
    return [
        "nvidia-smi", f"--query-gpu={','.join(GPU_QUERY_FIELDS)}",
        "--format=csv,noheader,nounits", "-lms", str(GPU_SAMPLE_MS), "-f", str(out),
    ]  # fmt: skip


# ---------------------------------------------------------------------------
# Parsing the captures
# ---------------------------------------------------------------------------


def _number(value: object) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True, slots=True)
class GpuSample:
    """One nvidia-smi line: memory used and total, utilisation and power."""

    used_mib: float
    total_mib: float
    util_pct: float | None
    power_w: float | None


def parse_gpu_csv(text: str) -> list[GpuSample]:
    """The samples of `nvidia_smi_argv`'s file; a line that does not parse is skipped."""
    samples = []
    for row in csv.reader(io.StringIO(text)):
        if len(row) != len(GPU_QUERY_FIELDS):
            continue
        used, total = _number(row[1]), _number(row[2])
        if used is None or total is None:
            continue
        samples.append(GpuSample(used, total, _number(row[3]), _number(row[6])))
    return samples


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def gpu_summary(samples: Sequence[GpuSample]) -> dict[str, object]:
    utils = [s.util_pct for s in samples if s.util_pct is not None]
    powers = [s.power_w for s in samples if s.power_w is not None]
    return {
        "samples": len(samples),
        "peak_used_mib": max((s.used_mib for s in samples), default=None),
        "total_mib": max((s.total_mib for s in samples), default=None),
        "mean_util_pct": _mean(utils),
        "mean_power_w": _mean(powers),
    }


@dataclass(frozen=True, slots=True)
class CounterCapture:
    """A typeperf CSV: the counter columns and the rows, timestamp first."""

    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


def parse_typeperf_csv(text: str) -> CounterCapture:
    rows = [tuple(r) for r in csv.reader(io.StringIO(text.lstrip("﻿"))) if r]
    if not rows:
        return CounterCapture((), ())
    return CounterCapture(rows[0], tuple(rows[1:]))


def counter_summary(capture: CounterCapture, counter: str) -> dict[str, object]:
    """Mean and maximum of the column whose path ends with `counter`; blank samples skipped."""
    index = next(
        (i for i, name in enumerate(capture.columns) if name.lower().endswith(counter.lower())),
        None,
    )
    values = []
    if index is not None:
        for row in capture.rows:
            value = _number(row[index]) if index < len(row) else None
            if value is not None:
                values.append(value)
    return {"n": len(values), "mean": _mean(values), "max": max(values, default=None)}


# ---------------------------------------------------------------------------
# The 4070 slice
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SliceOption:
    """A placement's footprint from run 7, and the layer count it serves with."""

    footprint_mib: float
    ngl: int | None


def slice_footprints(speech: Path, model: str) -> dict[str, SliceOption]:
    """Run 7's footprints: `gpu-full` by difference, the slices from their recorded search.

    A placement run 7 left no record for is not a candidate.
    """
    options: dict[str, SliceOption] = {}
    memory = speech / "lowtier" / model / "gpu-full" / "memory.json"
    if memory.is_file():
        found = _number(core.load_json(memory).get("gpu_by_difference_mib"))
        if found is not None:
            options["gpu-full"] = SliceOption(found, None)
    for placement in SLICE_ORDER[1:]:
        record = speech / "lowtier" / "slice" / model / placement / "slice.json"
        if not record.is_file():
            continue
        doc = core.load_json(record)
        ngl = doc.get("chosen_ngl")
        if ngl is None:
            continue
        probe = next((p for p in doc.get("probes", []) if p.get("ngl") == ngl), {})
        footprint = _number(probe.get("footprint_mib"))
        if footprint is None:
            footprint = _number(doc.get("slice_mib"))
        if footprint is not None:
            options[placement] = SliceOption(footprint, int(ngl))
    return options


def choose_slice(
    card_mib: float,
    g0_peak_mib: float,
    footprints: Mapping[str, SliceOption],
    margin_mib: float = MARGIN_MIB,
) -> str | None:
    """The first of gpu-full, gpu-4g and gpu-2g that fits beside the game with the margin."""
    for placement in SLICE_ORDER:
        option = footprints.get(placement)
        if option is not None and g0_peak_mib + option.footprint_mib + margin_mib <= card_mib:
            return placement
    return None


# ---------------------------------------------------------------------------
# What the session runs through (injected, so tests replace every outside thing)
# ---------------------------------------------------------------------------


class Handle(Protocol):
    """A started capture tool."""

    def wait(self, timeout: float) -> int | None:
        """The exit code, or None if the process is still running after `timeout`."""
        ...

    def terminate(self) -> None:
        """Stop the process."""
        ...


class Runner(Protocol):
    """Starts the capture tools; `SubprocessRunner` is the real one."""

    def start(self, argv: list[str], *, log_path: Path) -> Handle:
        """Start `argv` with its output in `log_path`."""
        ...


class Services(Protocol):
    """The chain's two services; `chain.ChainServices` is the real one."""

    def start(self) -> None:
        """Start the chain services."""
        ...

    def stop(self, log_dir: Path) -> None:
        """Stop them, keeping their logs in `log_dir`."""
        ...


class _Process:
    def __init__(self, proc: subprocess.Popen, log: object):
        self._proc = proc
        self._log = log

    def wait(self, timeout: float) -> int | None:
        try:
            code = self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
        self._log.close()
        return code

    def terminate(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()


class SubprocessRunner:
    """Starts a capture tool as a child process: an argv list, never a shell."""

    def start(self, argv: list[str], *, log_path: Path) -> Handle:
        """Start `argv` with its output in `log_path`, refusing to overwrite a log."""
        log = log_path.open("x", encoding="utf-8")
        try:
            proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, shell=False)
        except OSError as exc:
            log.close()
            raise GameSessionError(f"cannot start {argv[0]}: {exc}") from exc
        return _Process(proc, log)


def is_admin() -> bool:
    """Whether this terminal has administrator rights, which PresentMon's capture needs."""
    if os.name == "nt":
        try:
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        except (AttributeError, OSError):
            return False
    return hasattr(os, "geteuid") and os.geteuid() == 0


@dataclass(frozen=True)
class GameEnv:
    """Every outside thing the session touches."""

    ask: Callable[[str], str]
    say: Callable[[str], None]
    clock: Clock
    sleep: Callable[[float], None]
    now: Callable[[], datetime]
    runner: Runner
    bench_main: Callable[[list[str]], int]
    is_admin: Callable[[], bool]
    opener: Callable[..., object]
    serve: Callable[..., object]
    unserve: Callable[[Path], object]
    services_factory: Callable[[], Services]


@dataclass(frozen=True, slots=True)
class GameRequest:
    """One session: the model, how frames are read, and which conditions run."""

    results_root: Path
    run: str
    model: str
    material_dir: Path
    speech_models_dir: Path | None
    presentmon: Path | None = None
    manual_fps: bool = False
    with_1070: bool = False
    remote_url: str = DEFAULT_REMOTE_URL
    server_process: str | None = None
    skip_placements: tuple[str, ...] = ()
    pins: Path | None = None
    models_dir: Path | None = None
    small_models_dir: Path | None = None
    git_sha: str | None = None


# ---------------------------------------------------------------------------
# The session
# ---------------------------------------------------------------------------


def _check_request(request: GameRequest, env: GameEnv) -> None:
    """Refuse a session that cannot run, before any prompt, docker call or write."""
    core.speech_root(request.results_root, request.run)
    models = core.load_small_arms()["models"]
    if request.model not in models:
        raise core.PlacementError(f"unknown small model {request.model!r}; known: {list(models)}")
    if (request.presentmon is None) == (not request.manual_fps):
        raise GameSessionError("give --presentmon <path> or --manual-fps, one of the two")
    if request.presentmon is not None:
        if not env.is_admin():
            raise GameSessionError(
                "PresentMon needs administrator rights; open an administrator terminal, "
                "or use --manual-fps with the game's own benchmark"
            )
        if not request.presentmon.is_file():
            raise GameSessionError(f"--presentmon {request.presentmon} is not a file")
    unknown = [p for p in request.skip_placements if p not in SKIPPABLE]
    if unknown:
        raise GameSessionError(f"--skip-placement {unknown}: choose from {list(SKIPPABLE)}")


def run_game_session(request: GameRequest, env: GameEnv) -> Path:
    """Run the guided session; return its directory, `<speech>/game/<bench run>/`."""
    _check_request(request, env)
    inputs = chain.load_chain_inputs(request.material_dir, GAME_TURNS)
    speech = core.speech_root(request.results_root, request.run)
    bench_run = new_bench_run(request.run, env.now())
    core.begin_bench_window(request.results_root, bench_run)

    process = env.ask(
        "The game's process name, as Task Manager shows it (for example examplegame.exe): "
    ).strip()
    if not process:
        raise GameSessionError("no process name given")
    if not process.lower().endswith(".exe"):
        process += ".exe"
    game = env.ask("The game's name: ").strip()
    version = env.ask("The game's version: ").strip()
    settings = env.ask("The settings you play at (resolution, preset, frame cap, upscaling): ")

    claim = core.claim_output(
        request.results_root,
        request.run,
        SUITE,
        bench_run,
        git_sha=request.git_sha or core.current_git_sha(),
    )
    session = _Session(request, env, claim, inputs, speech, bench_run, process)
    session.frames_mode = session.test_capture()
    core.write_json_exclusive(
        claim.path / "meta.json",
        {
            **claim.stamp(),
            "bench_run": bench_run,
            "process": process,
            "game": game,
            "version": version,
            "settings": settings.strip(),
            "model": request.model,
            "frames_mode": session.frames_mode,
            "with_1070": request.with_1070,
            "server_process": request.server_process,
            "skip_placements": list(request.skip_placements),
            "passes": PASSES,
            "capture_s": CAPTURE_S,
            "turns": list(GAME_TURNS),
            "turn_every_s": TURN_EVERY_S,
            "margin_mib": MARGIN_MIB,
        },
    )
    session.bench("snapshot")
    try:
        session.bench("stop", "--with-neo4j")
        session.walk()
    except BaseException:
        session.restore(raise_on_failure=False)
        raise
    session.restore(raise_on_failure=True)
    return claim.path


@dataclass
class _Chain:
    """Where G2's turns go: the alias and its server."""

    alias: str
    llm: chain.LlmClient


class _Session:
    def __init__(
        self,
        request: GameRequest,
        env: GameEnv,
        claim: core.OutputClaim,
        inputs: chain.ChainInputs,
        speech: Path,
        bench_run: str,
        process: str,
    ):
        self.request = request
        self.env = env
        self.claim = claim
        self.inputs = inputs
        self.speech = speech
        self.bench_run = bench_run
        self.process = process
        self.frames_mode = "game_benchmark"
        self.stt = chain.SttClient(chain.STT_URL, opener=env.opener, clock=env.clock)
        self.tts = chain.TtsClient(chain.TTS_URL, opener=env.opener, clock=env.clock)
        self.passes: core.JsonlWriter | None = None
        self.status: core.JsonlWriter | None = None

    # -- bench_host ---------------------------------------------------------------------------

    def bench(self, command: str, *extra: str) -> int:
        argv = [command, "--run", self.bench_run, "--results-root", str(self.request.results_root)]
        code = self.env.bench_main([*argv, *extra])
        if code != 0 and command != "restore":
            raise GameSessionError(f"bench_host {command} exited {code}")
        return code

    def restore(self, *, raise_on_failure: bool) -> None:
        """Restore MIST and time E4B's return to its first warm-up token."""
        t0 = self.env.clock()
        code = self.bench("restore")
        back_s: float | None = None
        error: str | None = None
        try:
            result = core.stream_chat(
                E4B_URL,
                core.WARMUP_BODY,
                clock=self.env.clock,
                timeout=E4B_TIMEOUT_S,
                opener=self.env.opener,
            )
            if result.t_first_delta is not None:
                back_s = result.t_first_delta - t0
        except core.StreamError as exc:
            error = str(exc)
        core.write_json_exclusive(
            self.claim.path / "e4b_return.json",
            {
                **self.claim.stamp(),
                "restore_exit": code,
                "e4b_return_s": back_s,
                "method": "from the start of bench_host restore to E4B's first warm-up token",
                "error": error,
            },
        )
        if code != 0 and raise_on_failure:
            raise GameSessionError(
                f"bench_host restore exited {code}; see {self.request.results_root}/"
                f"{self.bench_run}/session for the inspect diff"
            )

    # -- the walk -----------------------------------------------------------------------------

    def walk(self) -> None:
        with self.claim.jsonl("passes.jsonl") as passes, self.claim.jsonl("conditions.jsonl") as st:
            self.passes, self.status = passes, st
            skip = set(self.request.skip_placements)
            if self.request.with_1070:
                if REMOTE_PLACEMENT in skip:
                    self.skipped(REMOTE_PLACEMENT, "dropped by --skip-placement")
                else:
                    self.remote()
            g0 = [self.one_pass("G0", None, n) for n in range(1, PASSES + 1)]
            self.status.write({"placement": None, "conditions": ["G0"], "status": "done"})
            if GPU_PLACEMENT_GROUP in skip:
                self.skipped(GPU_PLACEMENT_GROUP, "dropped by --skip-placement")
            else:
                chosen = self.choose(g0)
                if chosen is None:
                    self.skipped(GPU_PLACEMENT_GROUP, "no slice fits beside G0's peak memory")
                else:
                    self.local(chosen)
            if CPU_PLACEMENT in skip:
                self.skipped(CPU_PLACEMENT, "dropped by --skip-placement")
            else:
                self.local(CPU_PLACEMENT)

    def skipped(self, placement: str, reason: str) -> None:
        conditions = [f"G1-{placement}", f"G2-{placement}"]
        self.env.say(f"[skip] {', '.join(conditions)}: {reason}")
        self.status.write(
            {
                "placement": placement,
                "conditions": conditions,
                "status": "skipped",
                "reason": reason,
            }
        )

    def choose(self, g0: Sequence[Mapping[str, object]]) -> str | None:
        peaks = [r["gpu"]["peak_used_mib"] for r in g0 if r["gpu"]["peak_used_mib"] is not None]
        cards = [r["gpu"]["total_mib"] for r in g0 if r["gpu"]["total_mib"] is not None]
        footprints = slice_footprints(self.speech, self.request.model)
        if not peaks or not cards:
            chosen = None
        else:
            chosen = choose_slice(max(cards), max(peaks), footprints)
        core.write_json_exclusive(
            self.claim.path / "slice.json",
            {
                **self.claim.stamp(),
                "chosen": chosen,
                "g0_peak_mib": max(peaks, default=None),
                "card_mib": max(cards, default=None),
                "margin_mib": MARGIN_MIB,
                "footprints": {p: asdict(o) for p, o in footprints.items()},
            },
        )
        return chosen

    def placement_passes(self, placement: str, target: _Chain) -> None:
        for condition, turns in ((f"G1-{placement}", None), (f"G2-{placement}", target)):
            for n in range(1, PASSES + 1):
                self.one_pass(condition, placement, n, turns)
        self.status.write(
            {
                "placement": placement,
                "conditions": [f"G1-{placement}", f"G2-{placement}"],
                "status": "done",
                "alias": target.alias,
            }
        )

    def remote(self) -> None:
        alias = host1070_script.game_alias(self.request.model)
        llm = chain.LlmClient(self.request.remote_url, opener=self.env.opener, clock=self.env.clock)
        served = llm.served_alias()
        if served != alias:
            self.skipped(REMOTE_PLACEMENT, f"host-1070 serves {served!r}, not {alias!r}")
            return
        services = self.env.services_factory()
        try:
            services.start()
            self.placement_passes(REMOTE_PLACEMENT, _Chain(alias, llm))
        finally:
            services.stop(self.log_dir(REMOTE_PLACEMENT))

    def local(self, placement: str) -> None:
        model = self.request.model
        out_dir = core.small_serve_dir(self.speech, model, placement, tag=self.bench_run)
        footprint = slice_footprints(self.speech, model).get(placement)
        ngl = footprint.ngl if footprint is not None else None
        services = self.env.services_factory()
        served = False
        try:
            services.start()
            self.env.serve(
                model,
                placement,
                out_dir=out_dir,
                pins_path=self.request.pins or self.speech / "host" / "pins.json",
                models_dirs={
                    "main": self.request.models_dir,
                    "small": self.request.small_models_dir,
                },
                ngl=ngl,
            )
            served = True
            alias = core.small_alias(model, placement)
            llm = chain.LlmClient(core.SMALL_BASE_URL, opener=self.env.opener, clock=self.env.clock)
            if llm.served_alias() != alias:
                raise lowtier_suite.AliasError(f"{core.SMALL_BASE_URL} does not serve {alias}")
            self.placement_passes(placement, _Chain(alias, llm))
        finally:
            if served:
                self.env.unserve(out_dir)
            services.stop(self.log_dir(placement))

    def log_dir(self, placement: str) -> Path:
        path = self.claim.path / f"services-{placement}"
        path.mkdir(exist_ok=True)
        return path

    # -- one pass -----------------------------------------------------------------------------

    def test_capture(self) -> str:
        """The frames source: a 10 s PresentMon capture that records nothing means typed."""
        if self.request.presentmon is None:
            return "game_benchmark"
        test_dir = core.claim_dir(self.claim.path / "presentmon-test")
        self.env.ask(
            f"PresentMon test: start the game, then press Enter to capture {TEST_CAPTURE_S} s"
        )
        out = test_dir / "presentmon.csv"
        argv = presentmon_argv(self.request.presentmon, self.process, out, TEST_CAPTURE_S)
        handle = self.env.runner.start(argv, log_path=test_dir / "presentmon.log")
        self.env.sleep(TEST_CAPTURE_S)
        self._finish(handle)
        try:
            frames = frametimes.summarize_presentmon(out).frames or 0
        except frametimes.FrameTimeError:
            frames = 0
        if frames == 0:
            self.env.say(
                "[warn] PresentMon recorded no frames of this game; each pass asks for the "
                "frame rate the game shows instead"
            )
            return "typed"
        return "presentmon"

    def _finish(self, handle: Handle) -> None:
        if handle.wait(PROCESS_GRACE_S) is None:
            handle.terminate()
            handle.wait(PROCESS_GRACE_S)

    def one_pass(
        self, condition: str, placement: str | None, n: int, turns: _Chain | None = None
    ) -> dict[str, object]:
        env, request = self.env, self.request
        pass_dir = core.claim_dir(self.claim.path / condition / f"pass{n}")
        env.ask(
            f"{condition}, pass {n} of {PASSES}: start the scene, then press Enter to capture "
            f"{CAPTURE_S} s"
        )
        t_start = env.clock()
        handles: list[Handle] = []
        if self.frames_mode == "presentmon":
            argv = presentmon_argv(
                request.presentmon, self.process, pass_dir / "presentmon.csv", CAPTURE_S
            )
            handles.append(env.runner.start(argv, log_path=pass_dir / "presentmon.log"))
        cpu_argv = typeperf_argv(pass_dir / "cpu.csv", CAPTURE_S, request.server_process)
        handles.append(env.runner.start(cpu_argv, log_path=pass_dir / "typeperf.log"))
        gpu = env.runner.start(
            nvidia_smi_argv(pass_dir / "gpu.csv"), log_path=pass_dir / "nvidia_smi.log"
        )
        turn_rows: list[dict[str, object]] = []
        try:
            if turns is not None:
                turn_rows = self.chain_turns(pass_dir, condition, n, t_start, turns)
            remaining = t_start + CAPTURE_S - env.clock()
            if remaining > 0:
                env.sleep(remaining)
            for handle in handles:
                self._finish(handle)
        finally:
            gpu.terminate()
            gpu.wait(PROCESS_GRACE_S)
            for handle in handles:
                handle.terminate()
        capture_s = env.clock() - t_start

        frames = self.frames(pass_dir)
        gpu_stats = gpu_summary(parse_gpu_csv(_read(pass_dir / "gpu.csv")))
        capture = parse_typeperf_csv(_read(pass_dir / "cpu.csv"))
        server = None
        if request.server_process:
            cpu_counter, memory_counter = process_counters(request.server_process)
            memory = counter_summary(capture, memory_counter)
            server = {
                "process": request.server_process,
                "cpu": counter_summary(capture, cpu_counter),
                "memory_max_mib": memory["max"] / 2**20 if memory["max"] is not None else None,
            }
        note = env.ask(
            "Anything you noticed in this pass, in one line (a stutter, a delay; blank for none): "
        ).strip()
        tick = None
        if request.server_process:
            tick = env.ask("The server's tick reading from its console (for example /tick query): ")
            tick = tick.strip()
        turn_ms = [r["turn_ms"] for r in turn_rows if r.get("turn_ms") is not None]
        row = {
            "condition": condition,
            "placement": placement,
            "pass": n,
            "capture_s": capture_s,
            "frames": frames.as_row(),
            "gpu": gpu_stats,
            "cpu": counter_summary(capture, TOTAL_CPU_COUNTER),
            "server": server,
            "note": note,
            "tick": tick,
            "turns": len(turn_rows),
            "turns_failed": sum(1 for r in turn_rows if r.get("error")),
            "turn_ms": core.summarize(turn_ms).as_dict() if turn_ms else None,
        }
        core.write_json_exclusive(pass_dir / "pass.json", {**self.claim.stamp(), **row})
        self.passes.write(row)
        return row

    def chain_turns(
        self, pass_dir: Path, condition: str, n: int, t_start: float, target: _Chain
    ) -> list[dict[str, object]]:
        """G2's turns, one every 12 s from the start of the capture."""
        env = self.env
        rows = []
        with core.JsonlWriter(pass_dir / "chain.jsonl", stamp=self.claim.stamp()) as out:
            for index, uid in enumerate(GAME_TURNS):
                due = TURN_EVERY_S * index
                wait = t_start + due - env.clock()
                if wait > 0:
                    env.sleep(wait)
                late = max(0.0, env.clock() - t_start - due)
                row = chain.run_turn(
                    self.inputs.items[uid], rep=n, alias=target.alias, inputs=self.inputs,
                    llm=target.llm, stt=self.stt, tts=self.tts, clock=env.clock,
                )  # fmt: skip
                row.update(condition=condition, due_s=due, late_s=late)
                row["pass"] = n
                out.write(row)
                rows.append(row)
        return rows

    def frames(self, pass_dir: Path) -> frametimes.FrameSummary:
        if self.frames_mode == "presentmon":
            try:
                summary = frametimes.summarize_presentmon(pass_dir / "presentmon.csv")
            except frametimes.FrameTimeError as exc:
                self.env.say(f"[warn] {exc}")
                summary = None
            if summary is not None and summary.frames:
                return summary
            self.env.say("[warn] PresentMon recorded no frames in this pass")
            return frametimes.from_typed_fps(self.ask_number(_TYPED_PROMPT, "frame rate"))
        if self.frames_mode == "typed":
            return frametimes.from_typed_fps(self.ask_number(_TYPED_PROMPT, "frame rate"))
        return frametimes.from_game_benchmark(
            self.ask_number("The benchmark's average frame rate (fps): ", "average frame rate"),
            self.ask_number(
                "The benchmark's 1% low (fps; blank if it gives none): ", "1% low", optional=True
            ),
            self.ask_number(
                "The benchmark's 99th-percentile frame time (ms; blank if it gives none): ",
                "99th-percentile frame time",
                optional=True,
            ),
        )

    def ask_number(self, prompt: str, what: str, *, optional: bool = False) -> float | None:
        for attempt in range(1, ASK_ATTEMPTS + 1):
            try:
                return frametimes.parse_positive(self.env.ask(prompt), what, optional=optional)
            except frametimes.FrameTimeError as exc:
                if attempt == ASK_ATTEMPTS:
                    raise
                self.env.say(f"[again] {exc}")
        return None


_TYPED_PROMPT = "The frame rate the game shows, averaged over the pass (fps): "


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# CLI and the suite contract
# ---------------------------------------------------------------------------


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "game-session", help="runs 8 and 9: a guided game session, in an administrator terminal"
    )
    core.add_results_args(parser)
    parser.add_argument("--model", required=True, help="the low-level model run 7 picked")
    frames = parser.add_mutually_exclusive_group(required=True)
    frames.add_argument("--presentmon", type=Path, help="PresentMon's executable")
    frames.add_argument(
        "--manual-fps", action="store_true", help="type the game's own benchmark figures instead"
    )
    parser.add_argument(
        "--with-1070", action="store_true", help="run the gtx1070-shared conditions first"
    )
    parser.add_argument("--remote-url", default=DEFAULT_REMOTE_URL, help="host-1070's server")
    parser.add_argument(
        "--server-process", help="run 9: the game server's process, for its CPU and memory"
    )
    parser.add_argument(
        "--skip-placement",
        action="append",
        default=[],
        choices=SKIPPABLE,
        help="drop a placement run 7 already ruled out",
    )
    parser.add_argument(
        "--material-dir", type=Path, required=True, help="material directory with manifest.json"
    )
    env_models = os.environ.get(tts_suite.SPEECH_MODELS_DIR_ENV)
    parser.add_argument(
        "--speech-models-dir",
        type=Path,
        default=Path(env_models) if env_models else None,
        help=f"host speech models directory (env {tts_suite.SPEECH_MODELS_DIR_ENV})",
    )
    parser.add_argument("--pins", type=Path, help="pin file (default <speech>/host/pins.json)")
    small_env = os.environ.get(lowtier_suite.SMALL_MODELS_DIR_ENV)
    parser.add_argument(
        "--small-models-dir",
        type=Path,
        default=Path(small_env) if small_env else None,
        help=f"small models directory (env {lowtier_suite.SMALL_MODELS_DIR_ENV})",
    )
    main_env = os.environ.get("MODELS_DIR")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(main_env) if main_env else None,
        help="main models directory (env MODELS_DIR)",
    )
    parser.set_defaults(func=cmd_game_session)


def _print(text: str) -> None:
    print(text, flush=True)


def default_env(request: GameRequest) -> GameEnv:
    if request.speech_models_dir is None:
        raise GameSessionError(
            f"--speech-models-dir (or {tts_suite.SPEECH_MODELS_DIR_ENV}) names the speech models"
        )
    speech_models = request.speech_models_dir.resolve().as_posix()

    def services() -> Services:
        return chain.ChainServices.default(
            repo_dir=core.REPO_ROOT.resolve().as_posix(), speech_models_dir=speech_models
        )

    return GameEnv(
        ask=input,
        say=_print,
        clock=time.perf_counter,
        sleep=time.sleep,
        now=lambda: datetime.now(UTC),
        runner=SubprocessRunner(),
        bench_main=bench_host.main,
        is_admin=is_admin,
        opener=urllib.request.urlopen,
        serve=core.serve_small,
        unserve=core.unserve_small,
        services_factory=services,
    )


def cmd_game_session(args: argparse.Namespace) -> int:
    request = GameRequest(
        results_root=args.results_root,
        run=args.run,
        model=args.model,
        material_dir=args.material_dir,
        speech_models_dir=args.speech_models_dir,
        presentmon=args.presentmon,
        manual_fps=args.manual_fps,
        with_1070=args.with_1070,
        remote_url=args.remote_url,
        server_process=args.server_process,
        skip_placements=tuple(args.skip_placement),
        pins=args.pins,
        models_dir=args.models_dir,
        small_models_dir=args.small_models_dir,
    )
    out = run_game_session(request, default_env(request))
    print(f"game session recorded at {out}; MIST restored")
    return 0


def container_argv(ctx: core.PlanContext) -> list[tuple[str, list[str]]]:
    """The session runs no container of its own; it starts `chain`'s services and small-serve."""
    return []


def validate_inputs(repo_root: Path) -> list[str]:
    """G2's turns are chain recordings, and every 4070 slice placement exists."""
    problems = [
        f"game: turn {u} is not a chain recording"
        for u in GAME_TURNS
        if u not in chain.CHAIN_UTTERANCES
    ]
    if len(GAME_TURNS) != 10:
        problems.append(f"game: G2 runs 10 turns, not {len(GAME_TURNS)}")
    arms = core.load_json(repo_root / core.SPEECH_DIR_REL / "small_arms.json")
    for placement in SLICE_ORDER:
        if placement not in arms.get("placements", {}):
            problems.append(f"small_arms.json: no {placement} placement for the 4070 slice")
    return problems


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


_SELFTEST_PRESENTMON = (
    "Application,ProcessID,MsBetweenPresents,MsBetweenDisplayChange\n"
    + "x.exe,1,10.0,10.0\n" * 99
    + "x.exe,1,50.0,50.0\n"
)


def selftest() -> None:
    times, displayed = frametimes.frame_times(_SELFTEST_PRESENTMON)
    summary = frametimes.summarize_frame_times(times, displayed=displayed)
    _check(summary.frames == 100 and displayed == 100, "frame count wrong")
    _check(abs(summary.low_1pct_fps - 20.0) < 1e-9, "1% low is not 1000 over the slowest 1%")
    _check(summary.p99_frame_ms == 10.0, "99th-percentile frame time wrong")
    capture = parse_typeperf_csv('"(PDH-CSV 4.0)","\\\\PC\\Processor(_Total)\\% Processor Time"\n'
                                 '"t1","10.0"\n"t2"," "\n"t3","30.0"\n')  # fmt: skip
    _check(counter_summary(capture, TOTAL_CPU_COUNTER)["mean"] == 20.0, "typeperf mean wrong")
    options = {"gpu-full": SliceOption(3500.0, None), "gpu-2g": SliceOption(2000.0, 12)}
    _check(choose_slice(12282, 6000, options) == "gpu-full", "slice choice wrong at 6000 MiB")
    _check(choose_slice(12282, 9000, options) == "gpu-2g", "slice choice wrong at 9000 MiB")
    when = datetime(2026, 11, 14, 19, 30, 5, tzinfo=UTC)
    _check(new_bench_run("mh1", when) == "mh1-game-20261114T193005Z", "bench run id format")
    _check("utilization.gpu" in nvidia_smi_argv(Path("g.csv"))[1], "no utilisation in the query")
