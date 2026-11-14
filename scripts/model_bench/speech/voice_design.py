"""The voice design tool: the `voice-design` subcommand (mist-realtime-voice plan v6, task T40).

    python -m scripts.model_bench.speech_bench voice-design --descriptions <file> --lines <file>
        --seeds N --out <dir> [--models-dir <dir>]

It renders candidate voices from text descriptions with the Qwen3-TTS VoiceDesign model, for
Raj's blind listening (`listen-pack --from-dir`). One container of `mist-speech-bench-qwen3`
runs `probes/design_probe.py` with the GPU, requested as the `qwen3` arm's runs request it, and
no network. It mounts the repository, the VoiceDesign weights from
`<models-dir>/speech/<org>/<repo>` and both input files read-only, and only `--out` writable.
Each description and seed gives one candidate, `<out>/cand-<description id>-s<seed>.wav`, and one
row of `<out>/design-log.jsonl`; the probe's docstring has the details.

Descriptions and lines are private text, so nothing here prints either: the container's output
goes only to `<out>/design-container.log`, and stdout carries ids, counts and paths. Exit 2,
before any container starts, refuses an input or `--out` inside the git checkout, an input that
is not a JSON list of `{"id", "text"}` with unique ids matching `[a-z0-9][a-z0-9-]*`, a seed
count below 1, missing weights, and an `--out` that already holds a candidate or a design log.
Stdlib only: the host runs it under a plain Python.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import core, docker_cmds, tts_suite
from .probes import design_probe

SUBCOMMAND = "voice-design"
PROBE_MODULE = "scripts.model_bench.speech.probes.design_probe"
IMAGE = "mist-speech-bench-qwen3"

# The VoiceDesign model and the revision host step H12 downloaded.
MODEL_PIN = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign@5ecdb67327fd37bb2e042aab12ff7391903235d3"
MODEL_REPOSITORY, MODEL_REVISION = MODEL_PIN.split("@")

MODELS_DIR_ENV = "MODELS_DIR"
SPEECH_SUBDIR = "speech"
CONTAINER_MODEL = "/models/voice-design"
CONTAINER_DESCRIPTIONS = "/inputs/descriptions.json"
CONTAINER_LINES = "/inputs/lines.json"
CONTAINER_OUT = "/out"
CONTAINER_LOG = "design-container.log"
PLAN_SEEDS = 5


class DesignRefusedError(core.SpeechBenchError):
    """A `voice-design` request refused before any container starts (exit 2)."""


def weights_dir(models_dir: str) -> str:
    """`<models dir>/speech/<org>/<repo>`, as a string so the plan's placeholders pass through."""
    return "/".join([models_dir.rstrip("/"), SPEECH_SUBDIR, *MODEL_REPOSITORY.split("/")])


def design_argv(
    *, repo_dir: str, models_dir: str, descriptions: str, lines: str, out_dir: str, seeds: int
) -> list[str]:
    """`docker run` of the probe: qwen3 image, the GPU, no network, only `--out` writable."""
    argv = ["docker", "run", "--rm", *docker_cmds.gpus_all(), "--network", "none"]
    argv += docker_cmds.engine_env({"PYTHONDONTWRITEBYTECODE": "1"})
    argv += docker_cmds.ro_mount(repo_dir, docker_cmds.REPO_MOUNT)
    argv += docker_cmds.ro_mount(weights_dir(models_dir), CONTAINER_MODEL)
    argv += docker_cmds.ro_mount(descriptions, CONTAINER_DESCRIPTIONS)
    argv += docker_cmds.ro_mount(lines, CONTAINER_LINES)
    argv += ["-v", f"{out_dir}:{CONTAINER_OUT}"]
    argv += ["-w", docker_cmds.REPO_MOUNT, IMAGE, "python", "-m", PROBE_MODULE]
    argv += ["--model", CONTAINER_MODEL, "--model-pin", MODEL_PIN]
    argv += ["--descriptions", CONTAINER_DESCRIPTIONS, "--lines", CONTAINER_LINES]
    argv += ["--seeds", str(seeds), "--out", CONTAINER_OUT]
    return argv


# ---------------------------------------------------------------------------
# Running a design session
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DesignRequest:
    """One `voice-design` invocation."""

    descriptions: Path
    lines: Path
    seeds: int
    out: Path
    models_dir: Path | None


@dataclass(frozen=True, slots=True)
class DesignResult:
    """What a finished session wrote: counts and paths, never text."""

    descriptions: int
    seeds: int
    candidates: int
    out: Path


def default_runner() -> tts_suite.ContainerRunner:
    return tts_suite.DockerRunner()


def _outside_repo(path: Path, what: str, repo_root: Path) -> Path:
    try:
        return core.refuse_path_inside_repo(path, what=what, repo_root=repo_root)
    except core.RepoPathError as exc:
        raise DesignRefusedError(str(exc)) from None


def _input_file(path: Path, what: str, repo_root: Path) -> tuple[Path, int]:
    resolved = _outside_repo(path, what, repo_root)
    if not resolved.is_file():
        raise DesignRefusedError(f"{what} {resolved} is not a file")
    try:
        items = design_probe.load_items(resolved, what)
    except design_probe.RefusedError as exc:
        raise DesignRefusedError(str(exc)) from None
    return resolved, len(items)


def check_request(request: DesignRequest, *, repo_root: Path) -> tuple[Path, Path, Path, Path, int]:
    """Every refusal, before anything is written: resolved inputs, weights, out, description count."""
    out = _outside_repo(request.out, "--out", repo_root)
    descriptions, count = _input_file(request.descriptions, "--descriptions", repo_root)
    lines, _ = _input_file(request.lines, "--lines", repo_root)
    try:
        design_probe.seeds_for(request.seeds)
    except design_probe.RefusedError as exc:
        raise DesignRefusedError(str(exc)) from None
    if request.models_dir is None:
        raise DesignRefusedError(f"--models-dir (or {MODELS_DIR_ENV}) names the models directory")
    weights = Path(weights_dir(request.models_dir.resolve().as_posix()))
    if not weights.is_dir():
        raise DesignRefusedError(f"the {MODEL_REPOSITORY} weights are not at {weights}")
    if out.exists() and not out.is_dir():
        raise DesignRefusedError(f"--out {out} is not a directory")
    existing = design_probe.existing_outputs(out)
    if (out / CONTAINER_LOG).exists():
        existing.append(CONTAINER_LOG)
    if existing:
        raise DesignRefusedError(f"--out {out} already holds {existing}; pick a new directory")
    return descriptions, lines, weights, out, count


def run_design(
    request: DesignRequest,
    *,
    runner: tts_suite.ContainerRunner,
    repo_root: Path = core.REPO_ROOT,
) -> DesignResult:
    """Check the request, run the probe's container, then check what it wrote."""
    descriptions, lines, _, out, count = check_request(request, repo_root=repo_root)
    out.mkdir(parents=True, exist_ok=True)
    argv = design_argv(
        repo_dir=repo_root.resolve().as_posix(),
        models_dir=request.models_dir.resolve().as_posix(),
        descriptions=descriptions.as_posix(),
        lines=lines.as_posix(),
        out_dir=out.as_posix(),
        seeds=request.seeds,
    )
    log_path = out / CONTAINER_LOG
    reported: list[str] = []

    def keep(line: str) -> None:
        # The container's lines are kept for counting only, never echoed.
        if line.startswith(design_probe.CANDIDATE_PREFIX):
            reported.append(line[len(design_probe.CANDIDATE_PREFIX) :])

    code = runner.run(argv, log_path=log_path, on_line=keep)
    if code != 0:
        raise core.SpeechBenchError(f"{SUBCOMMAND}: the container exited {code}; log at {log_path}")
    expected = count * request.seeds
    written = design_probe.existing_outputs(out)
    candidates = [name for name in written if name != design_probe.LOG_FILE]
    if len(reported) != expected or len(candidates) != expected:
        raise core.SpeechBenchError(
            f"{SUBCOMMAND}: expected {expected} candidates, the container reported "
            f"{len(reported)} and wrote {len(candidates)}; log at {log_path}"
        )
    return DesignResult(count, request.seeds, expected, out)


def cmd_voice_design(args: argparse.Namespace, *, emit: Callable[[str], None] = print) -> int:
    request = DesignRequest(
        descriptions=args.descriptions,
        lines=args.lines,
        seeds=args.seeds,
        out=args.out,
        models_dir=args.models_dir,
    )
    try:
        result = run_design(request, runner=default_runner())
    except DesignRefusedError as exc:
        emit(f"[REFUSED] {exc}")
        return 2
    emit(
        f"{SUBCOMMAND} ok: {result.descriptions} descriptions x {result.seeds} seeds = "
        f"{result.candidates} candidates in {result.out}; log {result.out / design_probe.LOG_FILE}"
    )
    return 0


# ---------------------------------------------------------------------------
# Registry contract: register, container_argv, validate_inputs, selftest
# ---------------------------------------------------------------------------


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        SUBCOMMAND, help="render candidate voices from text descriptions (VoiceDesign, GPU)"
    )
    parser.add_argument(
        "--descriptions",
        type=Path,
        required=True,
        help='JSON list of {"id", "text"} voice descriptions, outside the repository',
    )
    parser.add_argument(
        "--lines",
        type=Path,
        required=True,
        help='JSON list of {"id", "text"} lines every candidate reads, outside the repository',
    )
    parser.add_argument("--seeds", type=int, required=True, help="seeds 1 to N per description")
    parser.add_argument(
        "--out", type=Path, required=True, help="directory for the candidates and the design log"
    )
    env_models = os.environ.get(MODELS_DIR_ENV)
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(env_models) if env_models else None,
        help=f"models directory holding speech/<org>/<repo>/ (env {MODELS_DIR_ENV})",
    )
    parser.set_defaults(func=cmd_voice_design)


def container_argv(ctx: core.PlanContext) -> list[tuple[str, list[str]]]:
    argv = design_argv(
        repo_dir=ctx.repo_dir,
        models_dir=ctx.models_dir,
        descriptions="<descriptions>",
        lines="<lines>",
        out_dir="<out>",
        seeds=PLAN_SEEDS,
    )
    return [(SUBCOMMAND, argv)]


def validate_inputs(repo_root: Path) -> list[str]:
    """The pin's shape; the inputs themselves are private and checked at run time."""
    problems = []
    if len([p for p in MODEL_REPOSITORY.split("/") if p]) != 2:
        problems.append(f"model pin repository {MODEL_REPOSITORY!r} is not <org>/<repo>")
    if len(MODEL_REVISION) != 40 or any(c not in "0123456789abcdef" for c in MODEL_REVISION):
        problems.append(f"model pin revision {MODEL_REVISION!r} is not a full commit sha")
    return problems


def _check(condition: bool, message: str) -> None:
    # Explicit raise rather than `assert`, which `python -O` strips.
    if not condition:
        raise AssertionError(message)


def selftest() -> None:
    """Check the argv rules, the candidate name and the input refusals."""
    argv = design_argv(
        repo_dir="/repo", models_dir="/m", descriptions="/d", lines="/l", out_dir="/o", seeds=2
    )
    specs = [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]
    _check([s for s in specs if not s.endswith(":ro")] == [f"/o:{CONTAINER_OUT}"], "mounts")
    _check(f"/m/speech/{MODEL_REPOSITORY}:{CONTAINER_MODEL}:ro" in specs, "weights mount")
    _check(argv[argv.index("--gpus") + 1] == "all", "GPU request")
    _check(argv[argv.index("--network") + 1] == "none", "network")
    _check("HF_HUB_OFFLINE=1" in argv and "-p" not in argv, "offline, no port")
    _check(design_probe.candidate_name("warm", 3) == "cand-warm-s3.wav", "candidate name")
    _check(list(design_probe.seeds_for(2)) == [1, 2], "seeds")
    _check(validate_inputs(core.REPO_ROOT) == [], "model pin")

    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "items.json"
        for doc in ('[{"id": "Bad", "text": "x"}]', '[{"id": "a", "text": "x"}] x', "{}"):
            path.write_text(doc, encoding="utf-8")
            try:
                design_probe.load_items(path, "--lines")
            except design_probe.RefusedError:
                continue
            raise AssertionError(f"accepted {doc!r}")
        path.write_text('[{"id": "a-1", "text": "x"}]', encoding="utf-8")
        _check([i.id for i in design_probe.load_items(path, "--lines")] == ["a-1"], "items")
