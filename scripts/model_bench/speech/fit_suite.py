"""Run 5, memory fit: the `fit` subcommand (mist-model-host plan v2, task T4).

`fit` measures what fits beside E4B on the 12 GB card, against `c0` already served as
`mist-bench-llm` and checked as bench_host's `run` checks it. It runs in two parts, each in its
own stop-and-restore window:

- `--part speech` (window `mh1-h4`): E4B alone, then the 12 combinations of E4B with one speech
  engine (`qwen3`, `turbo`, `original`) and one recognition engine (`whisper-base`,
  `parakeet-gpu`, `nemotron-160`, `parakeet-cpu`);
- `--part small` (window `mh1-h5`): for each small model of run 7 other than the E4B reference,
  served through `small-serve` as `mist-bench-small` at full offload (context 8,192, q8_0 cache),
  three steps: E4B and the small model; the same with `qwen3` and `parakeet-gpu`; the same with
  `turbo` and `parakeet-gpu`. Nine steps.

In each step the speech engine and the recognition engine run their probes' `loop` modes (T2's
and T3's loop argv), which load, warm up, print `READY` and then synthesize one line or transcribe
one utterance over and over. Once both are ready, 5 rounds run while `bench_host vram-step`
samples GPU memory for 10 s under the window's `--bench-run`: each round one 256-token completion
on E4B and, in the small part, one tool prompt on the small model at the same time.

A step whose engine or small model fails to load, runs out of memory or stops mid-run is a row
with `fits: false` and its message, never an error that ends the suite. Only a lost E4B server
ends it, since no later step could be measured. Each step's entry goes into `fit/steps.json` as
soon as it is measured: the `vram_steps.json` step plus `fits`, `headroom_mib` and what ran.

Stdlib only: the host runs it under a plain Python.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import json
import os
import re
import shlex
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from .. import bench_host
from . import cache_suite, core, lowtier_suite, stt_suite, toolcall, tts_suite
from .probes import load_gen

SUITE = "fit"
PARTS: tuple[str, ...] = ("speech", "small")
E4B_ARM = "c0"
E4B_LABEL = "e4b"
FIT_TTS: tuple[str, ...] = ("qwen3", "turbo", "original")
FIT_STT: tuple[str, ...] = ("whisper-base", "parakeet-gpu", "nemotron-160", "parakeet-cpu")
# The speech load beside each small model: none, then each of two speech engines with Parakeet.
SMALL_SPEECH: tuple[tuple[str | None, str | None], ...] = (
    (None, None),
    ("qwen3", "parakeet-gpu"),
    ("turbo", "parakeet-gpu"),
)
# Run 7's E4B reference is today's E4B, which already runs in every step.
REFERENCE_SMALL_MODEL = "e4b-ref"
SMALL_PLACEMENT = "gpu-full"
SMALL_TAG = "fit"
SAMPLE_S = 10.0
ROUNDS = 5
COMPLETION_TOKENS = 256
TOOL_SEED = 1
DEFAULT_LINE = "L01"
DEFAULT_UTT = "U01"
LOOP_SECONDS = 900.0
READY_TIMEOUT_S = 900.0
STOP_TIMEOUT_S = 60.0
TTS_LOOP_NAME = "mist-bench-fit-tts"
STT_LOOP_NAME = "mist-bench-fit-stt"
VRAM_LABEL_PREFIX = "fit-"
STEPS_FILE = "steps.json"
BENCH_LLM_URL = cache_suite.BENCH_LLM_URL
SPEECH_MODELS_DIR_ENV = tts_suite.SPEECH_MODELS_DIR_ENV
SMALL_MODELS_DIR_ENV = lowtier_suite.SMALL_MODELS_DIR_ENV
TAIL_LINES = 40
_OOM_RE = re.compile(
    r"out of memory|OutOfMemory|cudaMalloc failed|failed to allocate", re.IGNORECASE
)


def is_oom(text: str) -> bool:
    """Whether an engine's or server's output reports running out of memory."""
    return bool(_OOM_RE.search(text))


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FitStep:
    """One measured combination: E4B plus what runs beside it."""

    part: str
    tts: str | None = None
    stt: str | None = None
    small: str | None = None

    @property
    def label(self) -> str:
        """`e4b`, then the small model, the speech engine and the recognition engine, by `+`."""
        return "+".join(p for p in (E4B_LABEL, self.small, self.tts, self.stt) if p)

    @property
    def vram_label(self) -> str:
        """The step's label in the window's `vram_steps.json`."""
        return VRAM_LABEL_PREFIX + self.label

    @property
    def slug(self) -> str:
        """The step's results directory name, a plain path segment."""
        return self.label.replace("+", "_")

    def components(self) -> dict[str, str | None]:
        """What the step loads beside the card's desktop use."""
        return {"llm": E4B_ARM, "tts": self.tts, "stt": self.stt, "small": self.small}


def fit_small_models(arms: Mapping[str, object] | None = None) -> tuple[str, ...]:
    doc = arms if arms is not None else core.load_small_arms()
    return tuple(m for m in doc["models"] if m != REFERENCE_SMALL_MODEL)


def build_steps(part: str, *, small_models: Sequence[str] | None = None) -> list[FitStep]:
    """The part's steps in run order."""
    if part == "speech":
        return [FitStep(part)] + [FitStep(part, tts=t, stt=s) for t in FIT_TTS for s in FIT_STT]
    if part == "small":
        models = small_models if small_models is not None else fit_small_models()
        return [FitStep(part, tts=t, stt=s, small=m) for m in models for t, s in SMALL_SPEECH]
    raise core.SpeechBenchError(f"unknown fit part {part!r}; known: {list(PARTS)}")


def fit_entry(
    step: FitStep,
    summary: Mapping[str, object] | None,
    *,
    fits: bool,
    message: str | None,
    oom: bool,
    desktop_mib: float | None,
) -> dict[str, object]:
    """A `steps.json` entry: the `vram_steps.json` summary plus `fits` and `headroom_mib`.

    Headroom is the card's total less the step's peak; `by_difference_mib` is the peak less the
    window's `desktop` step, the memory the step's models took.
    """
    data = dict(summary or {})
    peak = data.get("max_mib")
    total = data.get("total_mib")
    numbers = isinstance(peak, int | float)
    return {
        "label": step.label,
        "part": step.part,
        "components": step.components(),
        "median_mib": data.get("median_mib"),
        "max_mib": peak,
        "total_mib": total,
        "samples": data.get("samples", 0),
        "t_utc": data.get("t_utc") or datetime.now(UTC).isoformat(),
        "fits": fits,
        "headroom_mib": total - peak if numbers and isinstance(total, int | float) else None,
        "desktop_mib": desktop_mib,
        "by_difference_mib": (
            peak - desktop_mib if numbers and isinstance(desktop_mib, int | float) else None
        ),
        "message": message,
        "oom": oom,
    }


# ---------------------------------------------------------------------------
# The loop containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LoopOutcome:
    """How a loop container ended: its exit code, whether it got ready or finished, its tail."""

    exit_code: int | None
    ready: bool
    done: bool
    tail: str


class LoopHandle(Protocol):
    """One started loop container."""

    def wait_ready(self, timeout: float) -> bool:
        """Wait for `READY`; False when the container ended first or the time ran out."""
        ...

    def running(self) -> bool:
        """Whether the container is still running."""
        ...

    def stop(self) -> None:
        """Stop the container and wait for it to end."""
        ...

    def outcome(self) -> LoopOutcome:
        """How it ended; read after `stop`."""
        ...


class LoopRunner(Protocol):
    """Starts loop containers; tests pass a fake."""

    def start(self, argv: list[str], *, log_path: Path) -> LoopHandle:
        """Start `argv` in the background, saving its output to `log_path`."""
        ...


class _DockerLoop:
    """A foreground `docker run` in a background process, its output pumped to a log."""

    def __init__(self, argv: list[str], log_path: Path):
        self.name = argv[argv.index("--name") + 1]
        self._tail: collections.deque[str] = collections.deque(maxlen=TAIL_LINES)
        self._ready = threading.Event()
        self._ended = threading.Event()
        self._done = False
        try:
            self._log = log_path.open("x", encoding="utf-8", newline="\n")
        except FileExistsError as exc:
            raise core.OutputExistsError(f"{log_path} already exists") from exc
        self._log.write(f"$ {shlex.join(argv)}\n")
        try:
            self._proc = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                shell=False,
            )
        except OSError as exc:
            self._log.close()
            raise core.DockerError(f"docker run failed to start: {exc}") from exc
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            self._log.write(line)
            self._log.flush()
            text = line.rstrip("\r\n")
            self._tail.append(text)
            if text == tts_suite.tts_probe.READY:
                self._ready.set()
            elif text == tts_suite.tts_probe.DONE:
                self._done = True
        self._ended.set()

    def wait_ready(self, timeout: float) -> bool:
        """Wait for `READY`, giving up early when the output ends."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._ready.wait(0.5):
                return True
            if self._ended.is_set():
                break
        return self._ready.is_set()

    def running(self) -> bool:
        """Whether the `docker run` process is still alive."""
        return self._proc.poll() is None

    def stop(self) -> None:
        """`docker stop` the container (it was started with `--rm`) and close the log."""
        if self._proc.poll() is None:
            # A failed `docker stop` falls through to the wait below, which kills the client.
            with contextlib.suppress(OSError, subprocess.TimeoutExpired):
                subprocess.run(
                    ["docker", "stop", "-t", "5", self.name],
                    capture_output=True,
                    text=True,
                    shell=False,
                    timeout=STOP_TIMEOUT_S,
                )
        try:
            self._proc.wait(timeout=STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        self._thread.join(timeout=5)
        if not self._log.closed:
            self._log.close()

    def outcome(self) -> LoopOutcome:
        """The exit code, the `READY` and `DONE` marks and the last lines of output."""
        return LoopOutcome(
            self._proc.returncode, self._ready.is_set(), self._done, "\n".join(self._tail)
        )


class DockerLoops:
    """Loop containers through the docker CLI on the host (argv lists, no shell)."""

    def start(self, argv: list[str], *, log_path: Path) -> LoopHandle:
        """Start one loop container."""
        return _DockerLoop(argv, log_path)


def _container_state(name: str) -> str:
    return bench_host.probe_container_state(name).state


# ---------------------------------------------------------------------------
# Requests and dependencies
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FitRequest:
    """One `fit` invocation: one part, under that part's window."""

    results_root: Path
    run: str
    bench_run: str
    part: str
    material_dir: Path
    speech_models_dir: Path
    line_id: str = DEFAULT_LINE
    utt_id: str = DEFAULT_UTT
    only: tuple[str, ...] = ()
    sample_s: float = SAMPLE_S
    rounds: int = ROUNDS
    loop_seconds: float = LOOP_SECONDS
    ready_timeout_s: float = READY_TIMEOUT_S
    pins_path: Path | None = None
    models_dirs: Mapping[str, Path | None] = field(default_factory=dict)


@dataclass
class FitDeps:
    """Everything `fit` reaches outside the process; tests pass fakes."""

    loops: LoopRunner
    served_check: Callable[[], object]
    bench_main: Callable[[list[str]], int] = bench_host.main
    chat: Callable[..., core.ChatStreamResult] = core.stream_chat
    container_state: Callable[[str], str] = _container_state
    serve: Callable[..., core.ServeResult] = core.serve_small
    unserve: Callable[..., float] = core.unserve_small
    clock: core.Clock = time.perf_counter


@dataclass(frozen=True, slots=True)
class SmallInputs:
    """The small model's system prompt, tool set and the command prompts the rounds send."""

    prompt: str
    tools: list
    prompts: tuple[toolcall.ToolPrompt, ...]

    @classmethod
    def load(cls) -> SmallInputs:
        """Run 7's low-level prompt and tools, and its command prompts in file order."""
        tools = toolcall.load_tools()
        commands = tuple(p for p in toolcall.load_prompts(tools=tools) if p.kind == "command")
        return cls(lowtier_suite.load_lowtier_prompt(), tools, commands)


# ---------------------------------------------------------------------------
# The rounds
# ---------------------------------------------------------------------------


class _Worker(threading.Thread):
    """A thread whose exception is re-raised by `result`."""

    def __init__(self, target: Callable[[], object]):
        super().__init__(daemon=True)
        self._target_fn = target
        self._value: object = None
        self._error: BaseException | None = None

    def run(self) -> None:
        try:
            self._value = self._target_fn()
        except BaseException as exc:  # noqa: BLE001 -- handed to the joining thread unchanged
            self._error = exc

    def result(self) -> object:
        self.join()
        if self._error is not None:
            raise self._error
        return self._value


def e4b_body() -> dict:
    """T2's load-generator completion, cut to 256 tokens; `ignore_eos` keeps it decoding."""
    return {**load_gen.DEFAULT_BODY, "max_tokens": COMPLETION_TOKENS}


def tool_body(alias: str, inputs: SmallInputs, prompt: toolcall.ToolPrompt) -> dict:
    return {
        "model": alias,
        "messages": [
            {"role": "system", "content": inputs.prompt},
            {"role": "user", "content": prompt.text},
        ],
        "tools": inputs.tools,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "max_tokens": lowtier_suite.TOOL_REPLY_TOKENS,
        "seed": TOOL_SEED,
        "cache_prompt": True,
    }


def _tool_round(
    deps: FitDeps, alias: str, inputs: SmallInputs, prompt: toolcall.ToolPrompt
) -> dict[str, object]:
    try:
        result = deps.chat(core.SMALL_BASE_URL, tool_body(alias, inputs, prompt), clock=deps.clock)
    except core.StreamError as exc:
        return {"small_error": str(exc)}
    score = toolcall.score_calls(prompt, result.tool_calls, inputs.tools)
    return {
        **score.as_row(),
        "small_ttft_ms": None if result.ttft_s is None else result.ttft_s * 1000.0,
        "small_error": None,
    }


def run_rounds(
    rounds: int,
    deps: FitDeps,
    *,
    small_alias: str | None = None,
    small_inputs: SmallInputs | None = None,
) -> list[dict[str, object]]:
    """Each round: one 256-token completion on E4B and, beside it, one small-model tool prompt.

    A failed request is recorded in its round and the rounds go on, so the sample keeps its load.
    """
    out = []
    for index in range(1, rounds + 1):
        row: dict[str, object] = {"round": index}
        tool_worker = None
        if small_alias is not None and small_inputs is not None:
            prompt = small_inputs.prompts[(index - 1) % len(small_inputs.prompts)]
            row["prompt_id"] = prompt.id
            tool_worker = _Worker(
                lambda prompt=prompt: _tool_round(deps, small_alias, small_inputs, prompt)
            )
            tool_worker.start()
        try:
            result = deps.chat(BENCH_LLM_URL, e4b_body(), clock=deps.clock)
            now = time.time()
            done = load_gen.completion_from(result, seq=index, t_start_unix=now, t_end_unix=now)
        except core.StreamError as exc:
            row.update(e4b_predicted_n=None, e4b_predicted_per_second=None, e4b_error=str(exc))
        else:
            row.update(
                e4b_predicted_n=done.predicted_n,
                e4b_predicted_per_second=done.predicted_per_second,
                e4b_error=None,
            )
        if tool_worker is not None:
            row.update(tool_worker.result())
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Running a part
# ---------------------------------------------------------------------------


def _vram_steps_path(results_root: Path, bench_run: str) -> Path:
    return results_root.resolve() / bench_run / "session" / "vram_steps.json"


def _vram_labels(results_root: Path, bench_run: str) -> set[str]:
    path = _vram_steps_path(results_root, bench_run)
    if not path.is_file():
        return set()
    return {str(s.get("label")) for s in core.load_json(path).get("steps", [])}


def read_vram_step(results_root: Path, bench_run: str, label: str) -> dict:
    """The step `bench_host vram-step --label <label>` wrote into the window's file."""
    path = _vram_steps_path(results_root, bench_run)
    steps = core.load_json(path).get("steps", []) if path.is_file() else []
    for step in steps:
        if step.get("label") == label:
            return step
    raise core.BenchRunError(f"{path} has no step {label!r} after vram-step ran")


def _recorded_labels(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {str(s.get("label")) for s in core.load_json(path).get("steps", [])}


def append_step(path: Path, entry: Mapping[str, object]) -> None:
    """Add `entry` to `fit/steps.json`, refusing a label it already holds; replace atomically."""
    doc = core.load_json(path) if path.is_file() else {"schema": 1, "steps": []}
    if any(s.get("label") == entry["label"] for s in doc.get("steps", [])):
        raise core.OutputExistsError(f"{path} already holds step {entry['label']!r}")
    doc.setdefault("steps", []).append(dict(entry))
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _select(steps: list[FitStep], only: Sequence[str], part: str) -> list[FitStep]:
    if not only:
        return steps
    labels = [s.label for s in steps]
    unknown = [o for o in only if o not in labels]
    if unknown:
        raise core.SpeechBenchError(f"unknown fit step(s) {unknown}; the {part} part runs {labels}")
    return [s for s in steps if s.label in only]


def _loop_argvs(
    step: FitStep, request: FitRequest, material: Path, models_dir: Path
) -> list[tuple[str, str, list[str]]]:
    repo = core.REPO_ROOT.resolve().as_posix()
    argvs = []
    if step.tts is not None:
        argv = tts_suite.loop_container_argv(
            step.tts,
            line_id=request.line_id,
            seconds=request.loop_seconds,
            repo_dir=repo,
            speech_models_dir=models_dir.as_posix(),
            name=TTS_LOOP_NAME,
        )
        argvs.append(("tts", step.tts, argv))
    if step.stt is not None:
        argv = stt_suite.loop_container_argv(
            step.stt,
            utt_id=request.utt_id,
            seconds=request.loop_seconds,
            repo_dir=repo,
            material_dir=material.as_posix(),
            weights_dir=models_dir.as_posix(),
            name=STT_LOOP_NAME,
        )
        argvs.append(("stt", step.stt, argv))
    return argvs


def _require_llm(deps: FitDeps, label: str) -> None:
    state = deps.container_state(bench_host.BENCH_LLM_CONTAINER)
    if state != "running":
        raise core.BenchRunError(
            f"{bench_host.BENCH_LLM_CONTAINER} is {state} before step {label}: E4B is gone, so "
            f"no later step can be measured; serve {E4B_ARM} again and run the remaining steps "
            f"with --step"
        )


def _last_line(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "no output"


@dataclass(frozen=True, slots=True)
class _StepContext:
    request: FitRequest
    deps: FitDeps
    argvs: Mapping[str, list[tuple[str, str, list[str]]]]
    desktop_mib: float | None
    stamp: Mapping[str, str]
    small_inputs: SmallInputs | None


def _run_step(
    step: FitStep,
    ctx: _StepContext,
    log_dir: Path,
    *,
    small_alias: str | None,
    small_failure: tuple[str, bool] | None,
) -> dict[str, object]:
    request, deps = ctx.request, ctx.deps
    extra: dict[str, object] = {
        **ctx.stamp,
        "bench_run": request.bench_run,
        "vram_step_label": step.vram_label,
        "sample_s": request.sample_s,
        "line_id": request.line_id if step.tts else None,
        "utt_id": request.utt_id if step.stt else None,
        "small_alias": small_alias,
        "loops": [],
        "rounds": [],
        "rounds_completed": 0,
        "load_covered_sample": None,
    }
    if small_failure is not None:
        message, oom = small_failure
        entry = fit_entry(
            step, None, fits=False, message=message, oom=oom, desktop_mib=ctx.desktop_mib
        )
        return {**entry, **extra}

    handles: list[tuple[str, str, LoopHandle, float | None]] = []
    failed: tuple[str, str, bool] | None = None
    running: dict[str, bool] = {}
    rounds: list[dict[str, object]] = []
    summary = None
    rc = 0
    t_sample_end = t_rounds_end = None
    try:
        for kind, engine, argv in ctx.argvs[step.label]:
            handle = deps.loops.start(argv, log_path=log_dir / f"{kind}_loop.log")
            t0 = deps.clock()
            ready = handle.wait_ready(request.ready_timeout_s)
            handles.append((kind, engine, handle, deps.clock() - t0 if ready else None))
            if not ready:
                failed = (kind, engine, handle.running())
                break
        if failed is None:
            worker = _Worker(
                lambda: (
                    run_rounds(
                        request.rounds,
                        deps,
                        small_alias=small_alias,
                        small_inputs=ctx.small_inputs,
                    ),
                    deps.clock(),
                )
            )
            worker.start()
            rc = deps.bench_main(
                [
                    "vram-step",
                    "--results-root",
                    str(request.results_root),
                    "--run",
                    request.bench_run,
                    "--label",
                    step.vram_label,
                    "--seconds",
                    f"{request.sample_s:g}",
                ]
            )
            t_sample_end = deps.clock()
            rounds, t_rounds_end = worker.result()
            running = {engine: handle.running() for _, engine, handle, _ in handles}
    finally:
        for _, _, handle, _ in handles:
            handle.stop()
    if rc != 0:
        raise core.BenchRunError(f"bench_host vram-step {step.vram_label} exited {rc}")

    messages: list[str] = []
    oom = False
    for kind, engine, handle, ready_s in handles:
        out = handle.outcome()
        info = {
            "kind": kind,
            "engine": engine,
            "ready": out.ready,
            "ready_s": ready_s,
            "exit_code": out.exit_code,
            "running_after_sample": running.get(engine),
            "ended_early": False,
        }
        if failed is not None and failed[1] == engine:
            if failed[2]:
                messages.append(
                    f"{kind} engine {engine} did not report READY within "
                    f"{request.ready_timeout_s:g} s: {_last_line(out.tail)}"
                )
            else:
                messages.append(
                    f"{kind} engine {engine} did not load (exit {out.exit_code}): "
                    f"{_last_line(out.tail)}"
                )
            oom = oom or is_oom(out.tail)
        elif failed is None and running.get(engine) is False:
            if out.done and out.exit_code == 0:
                info["ended_early"] = True
            else:
                messages.append(
                    f"{kind} engine {engine} stopped during the sample (exit {out.exit_code}): "
                    f"{_last_line(out.tail)}"
                )
                oom = oom or is_oom(out.tail)
        extra["loops"].append(info)

    for row in rounds:
        for who, key in (
            ("E4B completion", "e4b_error"),
            ("small-model tool prompt", "small_error"),
        ):
            error = row.get(key)
            if error:
                messages.append(f"{who} in round {row['round']} failed: {error}")
                oom = oom or is_oom(str(error))
    extra["rounds"] = rounds
    extra["rounds_completed"] = sum(
        1 for r in rounds if not r.get("e4b_error") and not r.get("small_error")
    )
    if t_sample_end is not None and t_rounds_end is not None:
        extra["load_covered_sample"] = t_rounds_end >= t_sample_end
    if failed is None:
        summary = read_vram_step(request.results_root, request.bench_run, step.vram_label)
    entry = fit_entry(
        step,
        summary,
        fits=not messages,
        message="; ".join(messages) or None,
        oom=oom,
        desktop_mib=ctx.desktop_mib,
    )
    return {**entry, **extra}


def _serve_failure(model_id: str, exc: core.ServeFailedError) -> tuple[str, bool]:
    text = str(exc)
    log = (
        exc.log_path.read_text(encoding="utf-8") if exc.log_path and exc.log_path.is_file() else ""
    )
    message = f"small model {model_id} did not load beside E4B at {SMALL_PLACEMENT}: {text}"
    return message, is_oom(text) or is_oom(log)


def _desktop(results_root: Path, bench_run: str) -> float | None:
    try:
        return lowtier_suite.desktop_baseline_mib(results_root, bench_run)
    except core.SpeechBenchError:
        return None


def run_fit(request: FitRequest, deps: FitDeps, *, git_sha: str | None = None) -> Path:
    """Run one part's steps and return `fit/steps.json`.

    Every refusal comes before anything is written or started: the window's snapshot, the
    served E4B, the material and weights, the small models' pins, the container names, and any
    step already recorded in `steps.json`, in the window's `vram_steps.json` (which `vram-step`
    would silently replace) or in its own results directory.
    """
    if request.rounds < 1 or request.sample_s <= 0:
        raise core.SpeechBenchError("--rounds and --sample-seconds must be positive")
    steps = _select(build_steps(request.part), request.only, request.part)
    speech = core.speech_root(request.results_root, request.run)
    core.validate_name(request.bench_run, "bench run")
    core.bench_run_snapshot(request.results_root, request.bench_run)
    deps.served_check()
    material = request.material_dir.resolve()
    stt_suite.material_manifest(material, [request.utt_id])
    models_dir = request.speech_models_dir.resolve()
    if not models_dir.is_dir():
        raise core.SpeechBenchError(f"speech models directory {models_dir} does not exist")
    argvs = {step.label: _loop_argvs(step, request, material, models_dir) for step in steps}

    small_inputs = None
    serve_dirs: dict[str, Path] = {}
    names = [TTS_LOOP_NAME, STT_LOOP_NAME]
    pins_path = request.pins_path or speech / "host" / "pins.json"
    if request.part == "small":
        small_inputs = SmallInputs.load()
        pins = core.load_pins(pins_path)
        arms = core.load_small_arms()
        for model_id in dict.fromkeys(s.small for s in steps if s.small):
            core.pinned_model(pins, model_id)
            where = arms["models"][model_id]["models_dir"]
            if request.models_dirs.get(where) is None:
                raise core.SpeechBenchError(
                    f"model {model_id!r} lives in the {where!r} models directory, which is not set"
                )
            serve_dirs[model_id] = core.small_serve_dir(
                speech, model_id, SMALL_PLACEMENT, SMALL_TAG
            )
            if serve_dirs[model_id].exists():
                raise core.OutputExistsError(f"{serve_dirs[model_id]} already exists")
        names.append(core.SMALL_CONTAINER)
    for name in names:
        state = deps.container_state(name)
        if state != "absent":
            raise core.SpeechBenchError(f"{name} already exists ({state}); remove it first")

    suite_dir = speech / SUITE
    steps_path = suite_dir / STEPS_FILE
    recorded = _recorded_labels(steps_path)
    vram_labels = _vram_labels(request.results_root, request.bench_run)
    for step in steps:
        if step.label in recorded:
            raise core.OutputExistsError(f"{steps_path} already holds step {step.label!r}")
        if step.vram_label in vram_labels:
            raise core.OutputExistsError(
                f"the window's vram_steps.json already holds {step.vram_label!r}, which vram-step "
                f"would replace; use a new --bench-run window"
            )
        if (suite_dir / step.slug).exists():
            raise core.OutputExistsError(f"{suite_dir / step.slug} already exists")

    sha = git_sha or core.current_git_sha()
    ctx = _StepContext(
        request=request,
        deps=deps,
        argvs=argvs,
        desktop_mib=_desktop(request.results_root, request.bench_run),
        stamp={"run": request.run, "suite": SUITE, "git_sha": sha},
        small_inputs=small_inputs,
    )

    def measure(step: FitStep, alias: str | None, failure: tuple[str, bool] | None) -> None:
        _require_llm(deps, step.label)
        claim = core.claim_output(request.results_root, request.run, SUITE, step.slug, git_sha=sha)
        entry = _run_step(step, ctx, claim.path, small_alias=alias, small_failure=failure)
        append_step(steps_path, entry)
        verdict = "fits" if entry["fits"] else f"does not fit: {entry['message']}"
        print(f"[fit] {step.label}: {verdict}")

    if request.part == "speech":
        for step in steps:
            measure(step, None, None)
        return steps_path

    for model_id, out_dir in serve_dirs.items():
        group = [s for s in steps if s.small == model_id]
        _require_llm(deps, group[0].label)
        served = None
        failure = None
        try:
            served = deps.serve(
                model_id,
                SMALL_PLACEMENT,
                out_dir=out_dir,
                pins_path=pins_path,
                models_dirs=request.models_dirs,
            )
        except core.ServeFailedError as exc:
            failure = _serve_failure(model_id, exc)
            if deps.container_state(core.SMALL_CONTAINER) != "absent" and out_dir.is_dir():
                deps.unserve(out_dir)
        try:
            for step in group:
                measure(step, served.alias if served is not None else None, failure)
        finally:
            if served is not None:
                deps.unserve(out_dir)
    return steps_path


# ---------------------------------------------------------------------------
# CLI, plan hooks and selftest
# ---------------------------------------------------------------------------


def _env_path(name: str) -> Path | None:
    raw = os.environ.get(name)
    return Path(raw) if raw else None


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "fit", help="run 5: GPU memory of E4B beside speech, recognition and small models"
    )
    core.add_results_args(parser, bench_run=True)
    parser.add_argument(
        "--part",
        required=True,
        choices=PARTS,
        help="speech: E4B alone and the 12 combinations; small: the 9 small-model steps",
    )
    parser.add_argument(
        "--material-dir", type=Path, required=True, help="material directory with manifest.json"
    )
    parser.add_argument(
        "--speech-models-dir",
        type=Path,
        default=_env_path(SPEECH_MODELS_DIR_ENV),
        help=f"speech and recognition weights, <org>/<repo>/ layout (env {SPEECH_MODELS_DIR_ENV})",
    )
    parser.add_argument(
        "--step", action="append", default=[], dest="only", help="run only this step label"
    )
    parser.add_argument("--line", default=DEFAULT_LINE, help="the line the speech loop speaks")
    parser.add_argument(
        "--utt", default=DEFAULT_UTT, help="the utterance the recognition loop hears"
    )
    parser.add_argument("--sample-seconds", type=float, default=SAMPLE_S)
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--loop-seconds", type=float, default=LOOP_SECONDS)
    parser.add_argument("--ready-timeout", type=float, default=READY_TIMEOUT_S)
    parser.add_argument("--pins", type=Path, help="pin file (default <speech>/host/pins.json)")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=_env_path("MODELS_DIR"),
        help="main models directory (env MODELS_DIR)",
    )
    parser.add_argument(
        "--small-models-dir",
        type=Path,
        default=_env_path(SMALL_MODELS_DIR_ENV),
        help=f"small models directory (env {SMALL_MODELS_DIR_ENV})",
    )
    parser.set_defaults(func=cmd_fit)


def cmd_fit(args: argparse.Namespace) -> int:
    if args.speech_models_dir is None:
        raise core.SpeechBenchError(
            f"--speech-models-dir (or {SPEECH_MODELS_DIR_ENV}) names the speech models directory"
        )
    request = FitRequest(
        results_root=args.results_root,
        run=args.run,
        bench_run=args.bench_run,
        part=args.part,
        material_dir=args.material_dir,
        speech_models_dir=args.speech_models_dir,
        line_id=args.line,
        utt_id=args.utt,
        only=tuple(args.only),
        sample_s=args.sample_seconds,
        rounds=args.rounds,
        loop_seconds=args.loop_seconds,
        ready_timeout_s=args.ready_timeout,
        pins_path=args.pins,
        models_dirs={"main": args.models_dir, "small": args.small_models_dir},
    )
    deps = FitDeps(
        loops=DockerLoops(),
        served_check=lambda: cache_suite.check_served(
            E4B_ARM, results_root=args.results_root, bench_run=args.bench_run
        ),
    )
    path = run_fit(request, deps)
    doc = core.load_json(path)
    fitting = sum(1 for s in doc["steps"] if s.get("fits"))
    print(f"fit {args.part}: {fitting} of {len(doc['steps'])} recorded steps fit; see {path}")
    return 0


def container_argv(ctx: core.PlanContext) -> list[tuple[str, list[str]]]:
    """The loop container of each engine `fit` starts; small models go through small-serve."""
    blocks = []
    for engine in FIT_TTS:
        argv = tts_suite.loop_container_argv(
            engine,
            line_id=DEFAULT_LINE,
            seconds=LOOP_SECONDS,
            repo_dir=ctx.repo_dir,
            speech_models_dir=ctx.speech_models_dir,
            name=TTS_LOOP_NAME,
        )
        blocks.append((f"fit loop {engine}", argv))
    for engine in FIT_STT:
        argv = stt_suite.loop_container_argv(
            engine,
            utt_id=DEFAULT_UTT,
            seconds=LOOP_SECONDS,
            repo_dir=ctx.repo_dir,
            material_dir=ctx.material_dir,
            weights_dir=ctx.speech_models_dir,
            name=STT_LOOP_NAME,
        )
        blocks.append((f"fit loop {engine}", argv))
    return blocks


def validate_inputs(repo_root: Path) -> list[str]:
    """Problems with what run 5 reads: its engines, line, utterance, small models and prompts."""
    problems = []
    speech = repo_root / core.SPEECH_DIR_REL
    combos = [s for s in build_steps("speech") if s.tts]
    if len(combos) != 12:
        problems.append(f"fit: the speech part has {len(combos)} combinations, not 12")
    try:
        small = build_steps(
            "small", small_models=fit_small_models(core.load_small_arms(speech / "small_arms.json"))
        )
    except (core.InputError, KeyError) as exc:
        problems.append(f"fit: small_arms.json: {exc}")
    else:
        if len(small) != 9:
            problems.append(f"fit: the small part has {len(small)} steps, not 9")
    try:
        container_argv(core.PlanContext(cuda_llama_image="cuda"))
    except (core.SpeechBenchError, KeyError) as exc:
        problems.append(f"fit: a loop argv fails: {exc}")
    try:
        utterances = core.load_json(speech / "utterances.json")["utterances"]
    except (core.InputError, KeyError) as exc:
        problems.append(f"fit: utterances.json: {exc}")
    else:
        if DEFAULT_UTT not in [u.get("id") for u in utterances]:
            problems.append(f"fit: utterances.json lists no {DEFAULT_UTT}")
    try:
        tools = toolcall.load_tools(speech / toolcall.TOOLS_SCHEMA_PATH.name)
        prompts = toolcall.load_prompts(speech / toolcall.TOOL_PROMPTS_PATH.name, tools=tools)
    except core.InputError as exc:
        problems.append(f"fit: tool prompts: {exc}")
    else:
        if sum(1 for p in prompts if p.kind == "command") < ROUNDS:
            problems.append(f"fit: fewer than {ROUNDS} command prompts for the rounds")
    return problems


def _check(condition: bool, message: str) -> None:
    # Explicit raise rather than `assert`, which `python -O` strips.
    if not condition:
        raise AssertionError(message)


def selftest() -> None:
    """The steps, the entry arithmetic, the out-of-memory check and the loop argv names."""
    speech = build_steps("speech")
    small = build_steps("small")
    _check(len(speech) == 13 and speech[0].label == E4B_LABEL, f"speech steps {len(speech)}")
    _check(len(small) == 9, f"small steps {len(small)}")
    labels = [s.label for s in (*speech, *small)]
    _check(len(set(labels)) == len(labels), "fit step labels repeat")
    _check(all(core.validate_name(s.slug, "step") for s in (*speech, *small)), "step slugs")

    summary = {"median_mib": 9000, "max_mib": 9500, "total_mib": 12282, "samples": 50}
    entry = fit_entry(speech[1], summary, fits=True, message=None, oom=False, desktop_mib=1000)
    _check(entry["headroom_mib"] == 2782 and entry["by_difference_mib"] == 8500, f"{entry}")
    failed = fit_entry(speech[1], None, fits=False, message="m", oom=True, desktop_mib=1000)
    _check(failed["headroom_mib"] is None and failed["fits"] is False, f"{failed}")

    _check(is_oom("torch.OutOfMemoryError: CUDA out of memory"), "torch out of memory")
    _check(is_oom("cudaMalloc failed: out of memory"), "llama.cpp out of memory")
    _check(not is_oom("No module named 'nemo'"), "an import error is not out of memory")

    blocks = dict(container_argv(core.PlanContext(cuda_llama_image="cuda")))
    for engine in FIT_TTS:
        _check(TTS_LOOP_NAME in blocks[f"fit loop {engine}"], f"{engine} loop name")
    for engine in FIT_STT:
        _check(STT_LOOP_NAME in blocks[f"fit loop {engine}"], f"{engine} loop name")
