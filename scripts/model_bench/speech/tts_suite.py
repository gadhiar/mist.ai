"""Run 1, speech synthesis: the `tts` subcommand (mist-model-host plan v2, task T2).

Per engine, in `speech_arms.json` order, each step its own container of the engine's bench
image: the timed pass with the bench model idle; a 60-second decode-rate baseline with no
speech; the timed pass again while the load generator keeps one 512-token completion streaming
from `mist-bench-llm` without a break; then, with the load generator stopped, the untimed
listening clips (each listening-only voice, each `--extra-voice`, and `original`'s emotive lines
in its `expressive` setting). The GPU sampler writes the engine's `vram.csv` throughout.

Every engine speaks with a stock voice of its own and nothing is cloned. An `--extra-voice` name
the engine does not list is refused here before any write, and the probe refuses it again in the
container. Containers mount the repository at `/work:ro` and the speech models directory
read-only, run with `HF_HUB_OFFLINE=1` and publish no port; GPU engines share the bench server's
network namespace, as bench_host's extraction probe does, and `pocket` runs with no GPU, no
network and at most 4 threads. The probe's row files and clips land in the engine's claimed
results directory, mounted at `/out`; the host stamps the rows and merges them into
`tts/samples.jsonl` and `tts/load.jsonl`.

Stdlib only: the host runs it under a plain Python.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shlex
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .. import bench_host
from . import core, docker_cmds
from .probes import load_gen, tts_engines, tts_probe

SUITE = "tts"
PROBE_MODULE = "scripts.model_bench.speech.probes.tts_probe"
CONTAINER_OUT = "/out"
DEFAULT_SEED = 20261106
DEFAULT_REPEATS = tts_probe.DEFAULT_REPEATS
BASELINE_S = 60.0
MAX_CPU_THREADS = 4
LOAD_GEN_BASE_URL = f"http://{bench_host.BENCH_LLM_HOST_BIND}:{bench_host.BENCH_LLM_PORT}"
SPEECH_MODELS_DIR_ENV = "MODEL_BENCH_SPEECH_MODELS_DIR"
SAMPLES_FILE = "samples.jsonl"
LOAD_FILE = "load.jsonl"
DECODE_FILE = "decode.jsonl"
PLAN_LOOP_LINE = "L01"
PLAN_LOOP_SECONDS = 60.0


def _tts_arms(arms: Mapping[str, object] | None = None) -> dict:
    doc = arms if arms is not None else core.load_json(core.SPEECH_ARMS_PATH)
    return dict(doc["tts"])


def _engine_cfg(engine_id: str, tts: Mapping[str, object]) -> Mapping[str, object]:
    if engine_id not in tts or engine_id not in tts_engines.ADAPTERS:
        raise core.SpeechBenchError(f"unknown speech engine {engine_id!r}; known: {list(tts)}")
    return tts[engine_id]


def _item_ids(lines_doc: Mapping[str, object]) -> list[str]:
    return [str(i["id"]) for i in [*lines_doc["lines"], *lines_doc["acknowledgements"]]]


# ---------------------------------------------------------------------------
# Container argv
# ---------------------------------------------------------------------------


def _probe_argv(
    engine_id: str,
    cfg: Mapping[str, object],
    *,
    repo_dir: str,
    speech_models_dir: str,
    out_dir: str | None = None,
    name: str | None = None,
) -> list[str]:
    """`docker run` up to and including the probe's `--engine` and `--models-dir`."""
    argv = ["docker", "run", "--rm"]
    if name is not None:
        argv += ["--name", name]
    env = {"PYTHONDONTWRITEBYTECODE": "1"}
    if cfg["device"] == "cuda":
        argv += docker_cmds.gpus_all()
        argv += ["--network", f"container:{bench_host.BENCH_LLM_CONTAINER}"]
    else:
        threads = min(int(cfg.get("threads", MAX_CPU_THREADS)), MAX_CPU_THREADS)
        argv += ["--network", "none", "--cpus", str(threads)]
        env.update({"OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)})
    argv += docker_cmds.engine_env(env)
    argv += docker_cmds.ro_mount(repo_dir, docker_cmds.REPO_MOUNT)
    argv += docker_cmds.ro_mount(speech_models_dir, tts_engines.CONTAINER_MODELS_DIR)
    if out_dir is not None:
        argv += ["-v", f"{out_dir}:{CONTAINER_OUT}"]
    argv += ["-w", docker_cmds.REPO_MOUNT, str(cfg["image"]), "python", "-m", PROBE_MODULE]
    argv += ["--engine", engine_id, "--models-dir", tts_engines.CONTAINER_MODELS_DIR]
    return argv


def _line_flags(lines: Sequence[str] | None) -> list[str]:
    return [flag for line in lines or () for flag in ("--lines", line)]


def timed_argv(
    engine_id: str,
    condition: str,
    *,
    out_dir: str,
    repo_dir: str,
    speech_models_dir: str,
    seed: int = DEFAULT_SEED,
    repeats: int = DEFAULT_REPEATS,
    lines: Sequence[str] | None = None,
    arms: Mapping[str, object] | None = None,
) -> list[str]:
    """The container argv of one timed pass, `idle` or `decoding`."""
    if condition not in tts_probe.CONDITIONS:
        raise core.SpeechBenchError(f"condition {condition!r} is not one of {tts_probe.CONDITIONS}")
    cfg = _engine_cfg(engine_id, _tts_arms(arms))
    argv = _probe_argv(
        engine_id, cfg, repo_dir=repo_dir, speech_models_dir=speech_models_dir, out_dir=out_dir
    )
    argv += ["--mode", "timed", "--condition", condition, "--seed", str(seed)]
    argv += ["--repeats", str(repeats), "--out", CONTAINER_OUT, *_line_flags(lines)]
    return argv


def render_argv(
    engine_id: str,
    voices: Sequence[str],
    *,
    expressive: bool,
    out_dir: str,
    repo_dir: str,
    speech_models_dir: str,
    lines: Sequence[str] | None = None,
    arms: Mapping[str, object] | None = None,
) -> list[str]:
    """The container argv of the untimed listening clips."""
    cfg = _engine_cfg(engine_id, _tts_arms(arms))
    argv = _probe_argv(
        engine_id, cfg, repo_dir=repo_dir, speech_models_dir=speech_models_dir, out_dir=out_dir
    )
    argv += ["--mode", "render", "--out", CONTAINER_OUT]
    argv += [flag for voice in voices for flag in ("--voice", voice)]
    if expressive:
        argv.append("--expressive")
    return argv + _line_flags(lines)


def loop_container_argv(
    engine_id: str,
    *,
    line_id: str,
    seconds: float,
    repo_dir: str,
    speech_models_dir: str,
    arms: Mapping[str, object] | None = None,
    lines_doc: Mapping[str, object] | None = None,
    name: str | None = None,
) -> list[str]:
    """The argv of run 5's synthesis loop: load, warm up, `READY`, synthesize, `DONE`.

    The container writes nothing and mounts nothing writable; it is named
    `mist-bench-tts-loop-<engine>` unless `name` is given, so the caller can stop it.
    """
    cfg = _engine_cfg(engine_id, _tts_arms(arms))
    known = _item_ids(lines_doc if lines_doc is not None else core.load_json(core.LINES_PATH))
    if line_id not in known:
        raise core.SpeechBenchError(f"unknown line id {line_id!r}")
    if seconds <= 0:
        raise core.SpeechBenchError(f"--seconds must be positive, got {seconds}")
    argv = _probe_argv(
        engine_id,
        cfg,
        repo_dir=repo_dir,
        speech_models_dir=speech_models_dir,
        name=name or f"mist-bench-tts-loop-{engine_id}",
    )
    return argv + ["--mode", "loop", "--line-id", line_id, "--seconds", f"{seconds:g}"]


# ---------------------------------------------------------------------------
# Voices
# ---------------------------------------------------------------------------


def parse_extra_voices(
    values: Sequence[str], tts: Mapping[str, object]
) -> dict[str, tuple[str, ...]]:
    """Group `<engine>:<voice>` specs by engine, refusing any name the engine does not list."""
    grouped: dict[str, list[str]] = {}
    for spec in values:
        engine_id, sep, voice = spec.partition(":")
        if not sep or not engine_id or not voice:
            raise tts_engines.VoiceError(f"--extra-voice {spec!r} is not <engine>:<voice>")
        if engine_id not in tts:
            raise tts_engines.VoiceError(
                f"--extra-voice {spec!r}: unknown engine {engine_id!r}; known: {list(tts)}"
            )
        stock = [str(v) for v in tts[engine_id]["voices"]]
        if voice not in stock:
            raise tts_engines.VoiceError(
                f"--extra-voice {spec!r}: {engine_id} has no stock voice {voice!r}; "
                f"it lists {stock}"
            )
        if voice not in grouped.setdefault(engine_id, []):
            grouped[engine_id].append(voice)
    return {engine_id: tuple(voices) for engine_id, voices in grouped.items()}


def render_plan(
    engine_id: str, cfg: Mapping[str, object], extra: Sequence[str]
) -> tuple[tuple[str, ...], bool]:
    """The voices to render untimed (listening-only, then extra) and whether `expressive` is."""
    voices = [str(v) for v in cfg.get("listening_voices") or []]
    voices += [v for v in extra if v not in voices]
    return tuple(voices), tts_probe.EXPRESSIVE in (cfg.get("settings") or {})


# ---------------------------------------------------------------------------
# Running the suite
# ---------------------------------------------------------------------------


class ContainerRunner(Protocol):
    """Runs one container to its end; tests pass a fake that runs the probe in-process."""

    def run(self, argv: list[str], *, log_path: Path, on_line: Callable[[str], None]) -> int:
        """Run `argv`, save its output to `log_path`, pass each output line to `on_line`."""
        ...


class Sampler(Protocol):
    """The GPU sampler (bench_host's `GpuSampler`)."""

    def start(self) -> None:
        """Start sampling."""
        ...

    def stop(self) -> None:
        """Stop sampling."""
        ...


class DockerRunner:
    """`docker run` on the host as an argv list, its merged output streamed line by line."""

    def run(self, argv: list[str], *, log_path: Path, on_line: Callable[[str], None]) -> int:
        """Run the container in the foreground and return its exit code."""
        try:
            log = log_path.open("x", encoding="utf-8", newline="\n")
        except FileExistsError as exc:
            raise core.OutputExistsError(f"{log_path} already exists") from exc
        with log:
            log.write(f"$ {shlex.join(argv)}\n")
            try:
                proc = subprocess.Popen(
                    argv,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    shell=False,
                )
            except OSError as exc:
                raise core.DockerError(f"docker run failed to start: {exc}") from exc
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                log.flush()
                on_line(line.rstrip("\r\n"))
            return proc.wait()


@dataclass(frozen=True, slots=True)
class TtsRequest:
    """One `tts` invocation."""

    results_root: Path
    run: str
    engines: tuple[str, ...]
    speech_models_dir: Path
    extra_voices: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    seed: int = DEFAULT_SEED
    repeats: int = DEFAULT_REPEATS
    lines: tuple[str, ...] | None = None
    baseline_s: float = BASELINE_S


def _read_rows(path: Path) -> list[dict]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise core.SpeechBenchError(f"the probe wrote no {path.name}: {exc}") from exc
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _refuse_before_writing(request: TtsRequest, tts: Mapping[str, object], lines_doc) -> None:
    for engine_id, voices in request.extra_voices.items():
        stock = [str(v) for v in _engine_cfg(engine_id, tts)["voices"]]
        refused = [v for v in voices if v not in stock]
        if refused:
            raise tts_engines.VoiceError(
                f"{engine_id} has no stock voices {refused}; it lists {stock}"
            )
        if engine_id not in request.engines:
            raise core.SpeechBenchError(
                f"--extra-voice names {engine_id}, which this run does not include"
            )
    if not request.engines or len(set(request.engines)) != len(request.engines):
        raise core.SpeechBenchError(f"engines {list(request.engines)} must be distinct, not none")
    for engine_id in request.engines:
        _engine_cfg(engine_id, tts)
    if request.lines is not None:
        unknown = [i for i in request.lines if i not in _item_ids(lines_doc)]
        if unknown:
            raise core.SpeechBenchError(f"unknown line ids {unknown}")
    if request.repeats < 1 or request.baseline_s <= 0:
        raise core.SpeechBenchError("--repeats must be at least 1 and --baseline-s positive")


@dataclass(slots=True)
class _EngineRun:
    """One engine's steps, sharing its claim, its paths and the injected dependencies."""

    engine_id: str
    cfg: Mapping[str, object]
    claim: core.OutputClaim
    request: TtsRequest
    runner: ContainerRunner
    clock: Callable[[], float]
    repo_dir: str
    models_dir: str
    arms: Mapping[str, object]

    def timed(self, condition: str, on_line: Callable[[str], None]) -> dict:
        argv = timed_argv(
            self.engine_id,
            condition,
            out_dir=self.claim.path.as_posix(),
            repo_dir=self.repo_dir,
            speech_models_dir=self.models_dir,
            seed=self.request.seed,
            repeats=self.request.repeats,
            lines=self.request.lines,
            arms=self.arms,
        )
        self._run(argv, f"{condition}.log", on_line)
        rows = _read_rows(self.claim.path / tts_probe.probe_file(condition))
        loads = [r for r in rows if r.get("row_type") == "load"]
        samples = [r for r in rows if r.get("row_type") == "sample"]
        if len(loads) != 1 or not samples:
            raise core.SpeechBenchError(
                f"tts {self.engine_id} {condition}: expected one load row and samples, got "
                f"{len(loads)} and {len(samples)}"
            )
        with self.claim.shared_jsonl(SAMPLES_FILE) as writer:
            for row in samples:
                writer.write({k: v for k, v in row.items() if k != "row_type"})
        return loads[0]

    def render(self, voices: Sequence[str], expressive: bool) -> None:
        argv = render_argv(
            self.engine_id,
            voices,
            expressive=expressive,
            out_dir=self.claim.path.as_posix(),
            repo_dir=self.repo_dir,
            speech_models_dir=self.models_dir,
            lines=self.request.lines,
            arms=self.arms,
        )
        self._run(argv, "render.log", lambda line: None)

    def _run(self, argv: list[str], log_name: str, on_line: Callable[[str], None]) -> None:
        log_path = self.claim.path / log_name
        code = self.runner.run(argv, log_path=log_path, on_line=on_line)
        if code != 0:
            raise core.SpeechBenchError(
                f"tts {self.engine_id}: container exited {code}; log at {log_path}"
            )


def _run_engine(run: _EngineRun, *, load_gen_factory, sampler_factory) -> None:
    sampler = sampler_factory(run.claim.path / "vram.csv")
    sampler.start()
    try:
        idle = run.timed("idle", lambda line: None)
        alone = load_gen_factory().run_for(run.request.baseline_s)

        marks: dict[str, float] = {}

        def mark(line: str) -> None:
            if line.strip() in (tts_probe.TIMED_START, tts_probe.TIMED_END):
                marks[line.strip()] = run.clock()

        generator = load_gen_factory()
        generator.start()
        try:
            decoding = run.timed("decoding", mark)
        except BaseException:
            with contextlib.suppress(core.SpeechBenchError):
                generator.stop()
            raise
        busy = generator.stop()
        if set(marks) != {tts_probe.TIMED_START, tts_probe.TIMED_END}:
            raise core.SpeechBenchError(
                f"tts {run.engine_id} decoding: the probe printed {sorted(marks)}, "
                f"not both window markers"
            )
        inside = load_gen.within(busy, marks[tts_probe.TIMED_START], marks[tts_probe.TIMED_END])

        voices, expressive = render_plan(
            run.engine_id, run.cfg, run.request.extra_voices.get(run.engine_id, ())
        )
        if voices or expressive:
            run.render(voices, expressive)
    finally:
        sampler.stop()

    with run.claim.jsonl(DECODE_FILE) as writer:
        for done in alone:
            writer.write({"engine": run.engine_id, "phase": "alone", **done.as_row()})
        for done in busy:
            writer.write(
                {
                    "engine": run.engine_id,
                    "phase": "with_tts",
                    "in_window": done in inside,
                    **done.as_row(),
                }
            )
    with run.claim.shared_jsonl(LOAD_FILE) as writer:
        writer.write(
            {
                "engine": run.engine_id,
                "voice": run.cfg["timing_voice"],
                "device": run.cfg["device"],
                "image": run.cfg["image"],
                "streaming": idle["streaming"],
                "sample_rate": idle["sample_rate"],
                "load_s": idle["load_s"],
                "load_s_decoding": decoding["load_s"],
                "decode_tps_alone": load_gen.decode_tps(alone),
                "decode_tps_with_tts": load_gen.decode_tps(inside),
                "decode_completions_alone": len(alone),
                "decode_completions_with_tts": len(inside),
                "decode_completions_overlapping": len(busy),
                "baseline_s": run.request.baseline_s,
                "seed": run.request.seed,
                "repeats": run.request.repeats,
                "lines": list(run.request.lines) if run.request.lines is not None else None,
            }
        )


def run_tts(
    request: TtsRequest,
    *,
    runner: ContainerRunner,
    container_state: Callable[[str], str],
    load_gen_factory: Callable[[], load_gen.LoadGenerator],
    sampler_factory: Callable[[Path], Sampler],
    clock: Callable[[], float] = time.perf_counter,
    arms: Mapping[str, object] | None = None,
    lines_doc: Mapping[str, object] | None = None,
    git_sha: str | None = None,
    repo_root: Path = core.REPO_ROOT,
) -> Path:
    """Run every requested engine in turn; return the `tts/` results directory.

    Refuses before any write: an unlisted extra voice, an unknown engine or line, a results root
    inside the repository, a missing speech models directory, a bench server that is not
    running, and an engine whose results directory already exists. `clock` marks the decoding
    window and must be the clock the load generator stamps its completions with.
    """
    arms_doc = arms if arms is not None else core.load_json(core.SPEECH_ARMS_PATH)
    tts = _tts_arms(arms_doc)
    lines = lines_doc if lines_doc is not None else core.load_json(core.LINES_PATH)
    _refuse_before_writing(request, tts, lines)
    speech = core.speech_root(request.results_root, request.run, repo_root=repo_root)
    if not request.speech_models_dir.is_dir():
        raise core.SpeechBenchError(
            f"speech models directory {request.speech_models_dir} does not exist"
        )
    state = container_state(bench_host.BENCH_LLM_CONTAINER)
    if state != "running":
        raise core.SpeechBenchError(
            f"{bench_host.BENCH_LLM_CONTAINER} is {state}, not running; serve the bench arm "
            f"with bench_host before timing speech beside it"
        )
    tts_dir = speech / SUITE
    existing = [e for e in request.engines if (tts_dir / e).exists()]
    if existing:
        raise core.OutputExistsError(
            f"{tts_dir} already holds {existing}; pick a new --run instead of overwriting"
        )

    sha = git_sha or core.current_git_sha(repo_root)
    claims = [
        core.claim_output(
            request.results_root, request.run, SUITE, e, git_sha=sha, repo_root=repo_root
        )
        for e in request.engines
    ]
    for engine_id, claim in zip(request.engines, claims, strict=True):
        run = _EngineRun(
            engine_id=engine_id,
            cfg=tts[engine_id],
            claim=claim,
            request=request,
            runner=runner,
            clock=clock,
            repo_dir=repo_root.resolve().as_posix(),
            models_dir=request.speech_models_dir.resolve().as_posix(),
            arms=arms_doc,
        )
        _run_engine(run, load_gen_factory=load_gen_factory, sampler_factory=sampler_factory)
    return tts_dir


def cmd_tts(args: argparse.Namespace) -> int:
    tts = _tts_arms()
    extra = parse_extra_voices(args.extra_voice, tts)
    models_dir = args.speech_models_dir
    if models_dir is None:
        raise core.SpeechBenchError(
            f"--speech-models-dir (or {SPEECH_MODELS_DIR_ENV}) names the speech models directory"
        )
    request = TtsRequest(
        results_root=args.results_root,
        run=args.run,
        engines=tuple(args.engine or tts),
        speech_models_dir=models_dir,
        extra_voices=extra,
        seed=args.seed,
        repeats=args.repeats,
        lines=tuple(args.lines) if args.lines else None,
        baseline_s=args.baseline_s,
    )
    tts_dir = run_tts(
        request,
        runner=DockerRunner(),
        container_state=lambda name: bench_host.probe_container_state(name).state,
        load_gen_factory=lambda: load_gen.LoadGenerator(LOAD_GEN_BASE_URL),
        sampler_factory=bench_host.GpuSampler,
    )
    print(f"tts ok: {', '.join(request.engines)}; results in {tts_dir}")
    return 0


# ---------------------------------------------------------------------------
# Registry contract: register, container_argv, validate_inputs, selftest
# ---------------------------------------------------------------------------


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "tts", help="run 1: speech synthesis, each engine in its own stock voice"
    )
    core.add_results_args(parser)
    parser.add_argument(
        "--engine", action="append", help="an engine id from speech_arms.json (default: all)"
    )
    parser.add_argument(
        "--extra-voice",
        action="append",
        default=[],
        metavar="ENGINE:VOICE",
        help="also render this stock voice, untimed, after the engine's timed passes",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="seed of the line order")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument(
        "--lines", action="append", metavar="ID", help="restrict to these line ids (smoke pass)"
    )
    parser.add_argument("--baseline-s", type=float, default=BASELINE_S)
    env_models = os.environ.get(SPEECH_MODELS_DIR_ENV)
    parser.add_argument(
        "--speech-models-dir",
        type=Path,
        default=Path(env_models) if env_models else None,
        help=f"host speech models directory, <org>/<repo>/ layout (env {SPEECH_MODELS_DIR_ENV})",
    )
    parser.set_defaults(func=cmd_tts)


def container_argv(ctx: core.PlanContext) -> list[tuple[str, list[str]]]:
    """Every engine's timed, render and loop argv on the plan's placeholder paths."""
    blocks = []
    paths = {"repo_dir": ctx.repo_dir, "speech_models_dir": ctx.speech_models_dir}
    for engine_id, cfg in _tts_arms().items():
        out = f"{ctx.speech_dir}/{SUITE}/{engine_id}"
        for condition in tts_probe.CONDITIONS:
            argv = timed_argv(engine_id, condition, out_dir=out, **paths)
            blocks.append((f"tts {engine_id} {condition}", argv))
        voices, expressive = render_plan(engine_id, cfg, ())
        if voices or expressive:
            argv = render_argv(engine_id, voices, expressive=expressive, out_dir=out, **paths)
            blocks.append((f"tts {engine_id} render", argv))
        argv = loop_container_argv(
            engine_id, line_id=PLAN_LOOP_LINE, seconds=PLAN_LOOP_SECONDS, **paths
        )
        blocks.append((f"tts {engine_id} loop", argv))
    return blocks


def validate_inputs(repo_root: Path) -> list[str]:
    """Check what run 1 reads from the committed inputs; an empty list means usable."""
    speech = repo_root / core.SPEECH_DIR_REL
    try:
        tts = _tts_arms(core.load_json(speech / "speech_arms.json"))
        lines_doc = core.load_json(speech / "lines.json")
    except (core.InputError, KeyError, TypeError) as exc:
        return [f"tts inputs unreadable: {exc}"]
    problems = []
    if set(tts) != set(tts_engines.ADAPTERS):
        problems.append(
            f"tts engines {sorted(tts)} differ from the adapters {sorted(tts_engines.ADAPTERS)}"
        )
    lines = list(lines_doc.get("lines") or [])
    ids = _item_ids(lines_doc)
    if len(ids) != len(set(ids)):
        problems.append("lines.json: a line or acknowledgement id repeats")
    if not any(line.get("emotive") for line in lines):
        problems.append("lines.json: no emotive line for the expressive setting")
    for engine_id, cfg in tts.items():
        where = f"speech_arms.json: tts.{engine_id}"
        if not isinstance(cfg.get("streaming"), bool):
            problems.append(f"{where}: streaming must be true or false")
        if cfg.get("device") == "cpu":
            threads = cfg.get("threads")
            if not isinstance(threads, int) or not 1 <= threads <= MAX_CPU_THREADS:
                problems.append(f"{where}: a CPU engine needs threads from 1 to {MAX_CPU_THREADS}")
        repository = str((cfg.get("model") or {}).get("repository", ""))
        if len([p for p in repository.split("/") if p]) != 2:
            problems.append(f"{where}: model repository {repository!r} is not <org>/<repo>")
        text_field = cfg.get("text_field")
        if text_field is not None and not any(text_field in line for line in lines):
            problems.append(f"{where}: no line carries its text field {text_field!r}")
        for name, setting in (cfg.get("settings") or {}).items():
            if name != "default" and setting.get("timed") is not False:
                problems.append(f"{where}: setting {name!r} is listening-only, so timed is false")
    expressive = ((tts.get("original") or {}).get("settings") or {}).get(tts_probe.EXPRESSIVE)
    if not expressive or expressive.get("lines") != "emotive":
        problems.append("speech_arms.json: original's expressive setting must cover emotive lines")
    return problems


def _check(condition: bool, message: str) -> None:
    # Explicit raise rather than `assert`, which `python -O` strips.
    if not condition:
        raise AssertionError(message)


class _SelftestEngine:
    """A stand-in engine yielding plain lists, so the selftest needs no numpy."""

    name = "selftest"
    device = "cpu"
    sample_rate = 10

    def __init__(self, streaming: bool, lengths: Sequence[int]):
        self.streaming = streaming
        self.lengths = tuple(lengths)
        self.texts: list[str] = []

    def stream(self, text: str):
        self.texts.append(text)
        for n in self.lengths:
            yield [0.0] * n


def selftest() -> None:
    """Check argv rules, the cut, the order, timing arithmetic, refusals and the load parser."""
    tts = _tts_arms()
    for engine_id, cfg in tts.items():
        argv = timed_argv(
            engine_id, "idle", out_dir="/r/tts/x", repo_dir="/repo", speech_models_dir="/m"
        )
        mounts = [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]
        _check(
            [m for m in mounts if not m.endswith(":ro")] == ["/r/tts/x:/out"], f"{engine_id} mounts"
        )
        _check("HF_HUB_OFFLINE=1" in argv, f"{engine_id} is not offline")
        _check("-p" not in argv, f"{engine_id} publishes a port")
        gpu = cfg["device"] == "cuda"
        _check(("--gpus" in argv) == gpu, f"{engine_id} GPU rule")
        network = argv[argv.index("--network") + 1]
        _check(network == ("container:mist-bench-llm" if gpu else "none"), f"{engine_id} network")
        loop = loop_container_argv(
            engine_id, line_id="L01", seconds=5, repo_dir="/repo", speech_models_dir="/m"
        )
        _check(":/out" not in " ".join(loop), f"{engine_id} loop mounts an output")

    _check(tts_probe.first_segment("a" * 39 + ", b") == "a" * 39 + ",", "cut at 40")
    _check(tts_probe.first_segment("a" * 38 + ", b") == "a" * 38 + ", b", "no cut before 40")
    order = tts_probe.timed_order(["a", "b", "c"], repeats=2, seed=1)
    _check(order == tts_probe.timed_order(["a", "b", "c"], repeats=2, seed=1), "seeded order")
    _check(sorted(order) == sorted((i, r) for r in (1, 2) for i in "abc"), "order coverage")

    ticks = iter([10.0, 10.25, 10.75, 11.0])
    streamed = tts_probe.time_item(
        _SelftestEngine(True, (5, 15)),
        "x",
        clock=lambda: next(ticks),
        wall=lambda: 0.0,
        device=tts_probe.NoDevice(),
    )
    _check(abs(streamed.first_audio_ms - 250.0) < 1e-6, f"first audio {streamed.first_audio_ms}")
    _check(abs(streamed.compute_per_speech_s - 0.5) < 1e-9, "compute per second of speech")
    whole_text = "a" * 39 + ", then the rest."
    ticks = iter([5.0, 5.3, 6.5])
    whole = tts_probe.time_item(
        _SelftestEngine(False, (30,)),
        whole_text,
        clock=lambda: next(ticks),
        wall=lambda: 0.0,
        device=tts_probe.NoDevice(),
    )
    _check(abs((whole.first_segment_ms or 0) - 300.0) < 1e-6, "first segment time")
    _check(abs(whole.total_ms - 1200.0) < 1e-6, "whole line time")

    try:
        parse_extra_voices(["qwen3:nobody"], tts)
    except tts_engines.VoiceError:
        pass
    else:
        raise AssertionError("an extra voice the engine does not list was accepted")

    fixture = core.SPEECH_FIXTURES_DIR / "tts" / "sse_long_completion.txt"
    _check(fixture.is_file(), f"load generator fixture missing at {fixture}")
    counter = iter(range(1, 1000))
    parsed = core.parse_chat_stream(
        fixture.read_text(encoding="utf-8").splitlines(keepends=True),
        clock=lambda: float(next(counter)),
        t_request=0.0,
    )
    done = load_gen.completion_from(parsed, seq=0, t_start_unix=0.0, t_end_unix=0.0)
    _check(done.predicted_per_second == 75.0, f"decode rate {done.predicted_per_second}")
