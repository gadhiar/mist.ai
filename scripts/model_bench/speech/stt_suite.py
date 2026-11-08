"""Run 3, speech to text, and `prepare`, the step that turns Raj's recordings into material.

`stt` runs one container per engine (`probes/stt_probe.py`), each writing its raw rows into the
engine's claimed directory; the host then scores each row's text against the script with
`wer.py` and appends it to `stt/samples.jsonl`. `prepare` runs one base-image container with
ffmpeg (`probes/prepare_material.py`). `loop_container_argv` is run 5's way in: the memory-fit
suite starts the probe's `loop` mode with it and waits for `READY`.

Container rules (mist-model-host plan v2): every container runs with `--network none`, the
repository at `/work:ro`, the material and the weights read-only and `HF_HUB_OFFLINE=1`; GPU
engines get `--gpus all`, and CPU engines no `--gpus`, at most 4 CPUs and 4 threads. The images
hold packages only; the probe code is the mounted repository.

Stdlib only, like `bench_host.py`: nothing here imports numpy or an engine package.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from . import core, docker_cmds, wer

STT_ENGINES: tuple[str, ...] = ("whisper-base", "parakeet-gpu", "parakeet-cpu", "nemotron-160")
BASE_IMAGE = "mist-speech-bench-base"
STT_PROBE = "scripts.model_bench.speech.probes.stt_probe"
PREPARE_PROBE = "scripts.model_bench.speech.probes.prepare_material"
CONTAINER_MATERIAL = "/material"
CONTAINER_WEIGHTS = "/weights"
CONTAINER_OUT = "/out"
MAX_CPU_THREADS = 4
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
# The probe's exit code for an engine that did not load (stt_probe.NOT_LOADED_EXIT).
NOT_LOADED_EXIT = 3
STREAMING_CHUNK_MS = 160


# ---------------------------------------------------------------------------
# argv pieces shared with the turn suite
# ---------------------------------------------------------------------------


def docker_head(name: str) -> list[str]:
    """A foreground, self-removing, offline container: its output is captured and saved."""
    return ["docker", "run", "--rm", "--name", name, "--network", "none"]


def resource_args(cfg: Mapping[str, object], what: str) -> list[str]:
    """`--gpus all` for a CUDA engine; for a CPU engine a CPU limit and its thread environment."""
    if cfg.get("device") == "cuda":
        return docker_cmds.gpus_all() + docker_cmds.engine_env()
    threads = cpu_threads(cfg, what)
    env = docker_cmds.engine_env({name: str(threads) for name in THREAD_ENV})
    return ["--cpus", str(threads), *env]


def cpu_threads(cfg: Mapping[str, object], what: str) -> int:
    threads = int(str(cfg.get("threads", MAX_CPU_THREADS)))
    if not 1 <= threads <= MAX_CPU_THREADS:
        raise core.SpeechBenchError(
            f"{what} asks for {threads} threads; CPU engines run on 1 to {MAX_CPU_THREADS}"
        )
    return threads


def probe_mounts(
    *, repo_dir: str, material_dir: str, weights_dir: str, out_dir: str | None
) -> list[str]:
    """The repository, material and weights read-only; only the claimed output writable."""
    mounts = docker_cmds.ro_mount(repo_dir, docker_cmds.REPO_MOUNT)
    mounts += docker_cmds.ro_mount(material_dir, CONTAINER_MATERIAL)
    mounts += docker_cmds.ro_mount(weights_dir, CONTAINER_WEIGHTS)
    if out_dir is not None:
        mounts += ["-v", f"{out_dir}:{CONTAINER_OUT}"]
    return mounts + ["-w", docker_cmds.REPO_MOUNT]


def run_container(argv: list[str]) -> subprocess.CompletedProcess:
    """Run a foreground container to completion, capturing its output (argv list, no shell)."""
    return subprocess.run(
        argv, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False
    )


def save_container_log(path: Path, proc: subprocess.CompletedProcess) -> None:
    core.write_text_exclusive(
        path,
        f"exit {proc.returncode}\n=== stdout ===\n{proc.stdout}\n=== stderr ===\n{proc.stderr}\n",
    )


def add_pins_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--pins",
        type=Path,
        help="H2's pin file (default <results-root>/<run>/speech/host/pins.json)",
    )
    group.add_argument(
        "--no-pins", action="store_true", help="skip the weights check (smoke passes only)"
    )


def check_pins(
    args: argparse.Namespace, speech: Path, engine_ids: Sequence[str], weights_dir: Path
) -> list[str]:
    """Refuse weights that differ from the pin file; return a warning per engine it does not list."""
    if args.no_pins:
        return [f"weights not checked against a pin file for {', '.join(engine_ids)}"]
    path = args.pins if args.pins is not None else speech / "host" / "pins.json"
    if not path.is_file():
        raise core.PinError(f"pin file {path} not found; pass --pins, or --no-pins for a smoke run")
    pins = core.load_pins(path)
    warnings = []
    for engine_id in engine_ids:
        if engine_id not in pins["models"]:
            warnings.append(f"the pin file lists no model {engine_id!r}; its weights are unchecked")
            continue
        pin = core.pinned_model(pins, engine_id)
        core.verify_model_file(pin, weights_dir / pin.file)
    return warnings


def material_manifest(material_dir: Path, utt_ids: Sequence[str]) -> core.Manifest:
    """Verify the material against its manifest and refuse an utterance it does not list."""
    manifest = core.verify_material(material_dir / "manifest.json")
    missing = sorted(set(utt_ids) - {item.id for item in manifest.items})
    if missing:
        raise core.ManifestError(f"the manifest lists no utterance {missing}")
    return manifest


def refuse_existing(paths: Sequence[Path]) -> None:
    for path in paths:
        if path.exists():
            raise core.OutputExistsError(
                f"{path} already exists; pick a new run instead of overwriting"
            )


def read_raw_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def read_load(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# argv
# ---------------------------------------------------------------------------


def _engine(arms: Mapping[str, object] | None, engine_id: str) -> dict:
    doc = arms if arms is not None else core.load_json(core.SPEECH_ARMS_PATH)
    engines = doc["stt"]
    if engine_id not in engines:
        raise core.SpeechBenchError(f"unknown stt engine {engine_id!r}; known: {list(engines)}")
    return engines[engine_id]


def _probe_tail(cfg: Mapping[str, object], engine_id: str, mode: str) -> list[str]:
    tail = [str(cfg["image"]), "python", "-m", STT_PROBE, mode, "--engine", engine_id]
    tail += ["--material", CONTAINER_MATERIAL, "--weights", CONTAINER_WEIGHTS]
    if cfg.get("device") == "cpu":
        tail += ["--threads", str(cpu_threads(cfg, engine_id))]
    return tail


def stt_container_argv(
    engine_id: str,
    *,
    repo_dir: str,
    material_dir: str,
    weights_dir: str,
    out_dir: str,
    reps: int = 3,
    warmups: int = 3,
    utt_ids: Sequence[str] = (),
    arms: Mapping[str, object] | None = None,
) -> list[str]:
    """`docker run` argv timing one engine over the material (run 3)."""
    cfg = _engine(arms, engine_id)
    argv = docker_head(f"mist-bench-stt-{engine_id}") + resource_args(cfg, engine_id)
    argv += probe_mounts(
        repo_dir=repo_dir, material_dir=material_dir, weights_dir=weights_dir, out_dir=out_dir
    )
    argv += _probe_tail(cfg, engine_id, "run")
    argv += ["--out", CONTAINER_OUT, "--reps", str(reps), "--warmups", str(warmups)]
    for uid in utt_ids:
        argv += ["--utt", uid]
    return argv


def loop_container_argv(
    engine_id: str,
    *,
    utt_id: str,
    seconds: float | str,
    repo_dir: str,
    material_dir: str,
    weights_dir: str,
    warmups: int = 3,
    name: str | None = None,
    arms: Mapping[str, object] | None = None,
) -> list[str]:
    """`docker run` argv for run 5: one engine transcribing `utt_id` for `seconds`.

    The probe prints `READY` on its own flushed stdout line once the engine is loaded and warm,
    and `DONE` when the time is up, then exits 0. Nothing is written: no output mount.
    `seconds` may be a placeholder string, for `plan`'s printout.
    """
    if not isinstance(seconds, str) and not seconds > 0:
        raise core.SpeechBenchError(f"loop seconds must be positive, not {seconds!r}")
    cfg = _engine(arms, engine_id)
    argv = docker_head(name or f"mist-bench-stt-loop-{engine_id}") + resource_args(cfg, engine_id)
    argv += probe_mounts(
        repo_dir=repo_dir, material_dir=material_dir, weights_dir=weights_dir, out_dir=None
    )
    argv += _probe_tail(cfg, engine_id, "loop")
    argv += ["--utt", utt_id, "--seconds", str(seconds), "--warmups", str(warmups)]
    return argv


def prepare_container_argv(
    *, repo_dir: str, material_dir: str, image: str = BASE_IMAGE
) -> list[str]:
    """`docker run` argv for `prepare`: the material directory writable, its recordings not.

    The read-only recordings mount sits inside the writable material mount, so the container can
    write the converted WAVs and the manifest but cannot change a recording.
    """
    recordings = f"{material_dir}/recordings"
    argv = docker_head("mist-bench-prepare") + docker_cmds.engine_env()
    argv += docker_cmds.ro_mount(repo_dir, docker_cmds.REPO_MOUNT)
    argv += ["-v", f"{material_dir}:{CONTAINER_MATERIAL}"]
    argv += docker_cmds.ro_mount(recordings, f"{CONTAINER_MATERIAL}/recordings")
    argv += ["-w", docker_cmds.REPO_MOUNT, image, "python", "-m", PREPARE_PROBE]
    argv += ["--recordings", f"{CONTAINER_MATERIAL}/recordings", "--out", CONTAINER_MATERIAL]
    return argv


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def _host_path(path: Path) -> str:
    return path.resolve().as_posix()


def score_row(engine_id: str, device: str, raw: Mapping[str, object], script: str) -> dict:
    """A `stt/samples.jsonl` row: the probe's timing plus the word error rate of its text."""
    result = wer.word_error_rate(script, str(raw.get("text", "")))
    row = {
        "engine": engine_id,
        "device": device,
        "utt_id": raw["utt_id"],
        "rep": raw["rep"],
        "audio_s": raw["audio_s"],
        "final_ms": raw["final_ms"],
        "wer": result.wer,
        "wer_errors": result.errors,
        "wer_reference_words": result.reference_words,
        "torch_peak_reserved_mib": raw.get("torch_peak_reserved_mib"),
        "text": raw.get("text", ""),
    }
    if "feed_lag_ms" in raw:
        row["feed_lag_ms"] = raw["feed_lag_ms"]
    return row


def cmd_stt(args: argparse.Namespace) -> int:
    speech = core.speech_root(args.results_root, args.run, repo_root=core.REPO_ROOT)
    engines = list(dict.fromkeys(args.engine or STT_ENGINES))
    arms = core.load_json(core.SPEECH_ARMS_PATH)
    configs = {e: _engine(arms, e) for e in engines}
    material = args.material_dir.resolve()
    manifest = material_manifest(material, args.utt)
    weights = args.weights_dir.resolve()
    if not weights.is_dir():
        raise core.SpeechBenchError(f"weights directory {weights} does not exist")
    warnings = check_pins(args, speech, engines, weights)
    refuse_existing([speech / "stt" / e for e in engines])
    for warning in warnings:
        print(f"[warn] {warning}")

    scripts = {item.id: item.text for item in manifest.items}
    git_sha = core.current_git_sha(core.REPO_ROOT)
    failures = 0
    for engine_id in engines:
        claim = core.claim_output(args.results_root, args.run, "stt", engine_id, git_sha=git_sha)
        argv = stt_container_argv(
            engine_id,
            repo_dir=_host_path(core.REPO_ROOT),
            material_dir=_host_path(material),
            weights_dir=_host_path(weights),
            out_dir=_host_path(claim.path),
            reps=args.reps,
            warmups=args.warmups,
            utt_ids=args.utt,
            arms=arms,
        )
        print(f"[run] stt {engine_id}")
        proc = run_container(argv)
        save_container_log(claim.path / "container.log", proc)
        load = read_load(claim.path / "load.json")
        if load.get("loaded") is False or proc.returncode == NOT_LOADED_EXIT:
            print(f"[not loaded] {engine_id}: {load.get('error', proc.stderr.strip())}")
            continue
        if proc.returncode != 0:
            print(
                f"[FAIL] stt {engine_id}: exit {proc.returncode}; log {claim.path / 'container.log'}"
            )
            failures += 1
            continue
        rows = [
            score_row(
                engine_id, str(configs[engine_id]["device"]), raw, scripts[str(raw["utt_id"])]
            )
            for raw in read_raw_rows(claim.path / "raw.jsonl")
        ]
        with claim.shared_jsonl("samples.jsonl") as writer:
            for row in rows:
                writer.write(row)
        final = core.summarize(r["final_ms"] for r in rows)
        mean_wer = sum(r["wer"] for r in rows) / len(rows) if rows else None
        print(
            f"[ok] stt {engine_id}: {final.n} rows, final text median {final.median} ms, "
            f"p90 {final.p90} ms, mean word error rate {mean_wer}"
        )
    return 1 if failures else 0


def cmd_prepare(args: argparse.Namespace) -> int:
    material = core.refuse_path_inside_repo(
        args.material_dir, what="--material-dir", repo_root=core.REPO_ROOT
    )
    if not (material / "recordings").is_dir():
        raise core.SpeechBenchError(
            f"{material / 'recordings'} does not exist; record into it first"
        )
    refuse_existing([material / "manifest.json"])
    argv = prepare_container_argv(
        repo_dir=_host_path(core.REPO_ROOT), material_dir=material.as_posix()
    )
    print("[run] prepare")
    proc = run_container(argv)
    print(proc.stdout, end="")
    if proc.returncode != 0:
        print(proc.stderr, end="")
        print(f"[FAIL] prepare: exit {proc.returncode}; nothing was written")
        return 1
    manifest = core.verify_material(material / "manifest.json")
    print(f"[ok] prepare: {len(manifest.items)} recordings verified against {manifest.path}")
    return 0


def _add_material_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--material-dir", type=Path, required=True, help="material directory with manifest.json"
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        required=True,
        help="speech weights directory, mounted read-only",
    )
    parser.add_argument("--utt", action="append", default=[], help="limit to these utterance ids")
    add_pins_args(parser)


def register(subparsers: argparse._SubParsersAction) -> None:
    stt = subparsers.add_parser("stt", help="run 3: final-text latency and word error rate")
    core.add_results_args(stt)
    _add_material_args(stt)
    stt.add_argument("--engine", action="append", choices=STT_ENGINES, help="default: all four")
    stt.add_argument("--reps", type=int, default=3)
    stt.add_argument("--warmups", type=int, default=3)
    stt.set_defaults(func=cmd_stt)

    prepare = subparsers.add_parser(
        "prepare", help="convert the recordings to 16 kHz mono WAV and write manifest.json"
    )
    prepare.add_argument(
        "--material-dir",
        type=Path,
        required=True,
        help="material directory outside the repository, with the recordings in recordings/",
    )
    prepare.set_defaults(func=cmd_prepare)


# ---------------------------------------------------------------------------
# plan and selftest
# ---------------------------------------------------------------------------


def container_argv(ctx: core.PlanContext) -> list[tuple[str, list[str]]]:
    """`prepare`'s argv, each engine's run-3 argv and each engine's run-5 loop argv."""
    blocks = [
        ("prepare", prepare_container_argv(repo_dir=ctx.repo_dir, material_dir=ctx.material_dir))
    ]
    paths = {
        "repo_dir": ctx.repo_dir,
        "material_dir": ctx.material_dir,
        "weights_dir": ctx.speech_models_dir,
    }
    for engine_id in STT_ENGINES:
        out_dir = f"{ctx.speech_dir}/stt/{engine_id}"
        blocks.append((f"stt {engine_id}", stt_container_argv(engine_id, out_dir=out_dir, **paths)))
    for engine_id in STT_ENGINES:
        argv = loop_container_argv(engine_id, utt_id="<utt-id>", seconds="<seconds>", **paths)
        blocks.append((f"stt loop {engine_id}", argv))
    return blocks


def validate_inputs(repo_root: Path) -> list[str]:
    """Check the `stt` section of speech_arms.json against what run 3 needs."""
    try:
        arms = core.load_json(repo_root / core.SPEECH_DIR_REL / "speech_arms.json")
        section = arms["stt"]
    except (core.InputError, KeyError) as exc:
        return [f"speech_arms.json: no stt section: {exc}"]
    problems = []
    if tuple(section) != STT_ENGINES:
        problems.append(
            f"speech_arms.json: stt engines are {list(section)}, expected {STT_ENGINES}"
        )
    for engine_id, cfg in section.items():
        if not str(cfg.get("image", "")).startswith("mist-speech-bench-"):
            problems.append(f"speech_arms.json: stt.{engine_id} image {cfg.get('image')!r}")
        try:
            stt_container_argv(
                engine_id, repo_dir="r", material_dir="m", weights_dir="w", out_dir="o", arms=arms
            )
        except (core.SpeechBenchError, KeyError, TypeError, ValueError) as exc:
            problems.append(f"speech_arms.json: stt.{engine_id} argv fails: {exc}")
    streaming = section.get("nemotron-160", {})
    if not streaming.get("streaming") or streaming.get("chunk_ms") != STREAMING_CHUNK_MS:
        problems.append("speech_arms.json: nemotron-160 must stream 160 ms chunks")
    return problems


def _pairs(argv: Sequence[str], flag: str) -> list[str]:
    return [argv[i + 1] for i, arg in enumerate(argv[:-1]) if arg == flag]


def _check(condition: bool, message: str) -> None:
    # An explicit raise rather than `assert`, which `python -O` strips.
    if not condition:
        raise AssertionError(message)


def selftest() -> None:
    """Check the argv rules, the loop argv and the word error rate without docker."""
    for engine_id in STT_ENGINES:
        cfg = _engine(None, engine_id)
        argv = stt_container_argv(
            engine_id, repo_dir="/r", material_dir="/m", weights_dir="/w", out_dir="/o"
        )
        docker_part = argv[: argv.index(str(cfg["image"]))]
        _check(_pairs(docker_part, "--network") == ["none"], f"{engine_id}: network")
        mounts = _pairs(docker_part, "-v")
        _check(
            all(m.endswith(":ro") for m in mounts if not m.endswith(f":{CONTAINER_OUT}")),
            f"{engine_id}: a writable mount besides the output: {mounts}",
        )
        _check("HF_HUB_OFFLINE=1" in _pairs(docker_part, "-e"), f"{engine_id}: offline")
        if cfg["device"] == "cuda":
            _check(_pairs(argv, "--gpus") == ["all"], f"{engine_id}: --gpus all")
        else:
            _check("--gpus" not in argv, f"{engine_id}: a CPU engine got --gpus")
            _check(_pairs(argv, "--cpus") == ["4"], f"{engine_id}: --cpus 4")
            _check("OMP_NUM_THREADS=4" in _pairs(argv, "-e"), f"{engine_id}: 4 threads")

    loop = loop_container_argv(
        "parakeet-cpu", utt_id="U01", seconds=30, repo_dir="/r", material_dir="/m", weights_dir="/w"
    )
    _check(loop[loop.index(STT_PROBE) + 1] == "loop", "loop argv runs the probe's loop mode")
    _check(_pairs(loop, "--seconds") == ["30"], "loop argv passes --seconds")
    _check(not any(m.endswith(CONTAINER_OUT) for m in _pairs(loop, "-v")), "loop writes")

    prepare = prepare_container_argv(repo_dir="/r", material_dir="/m")
    _check(
        "/m/recordings:/material/recordings:ro" in _pairs(prepare, "-v"),
        "prepare mounts the recordings read-only",
    )

    result = wer.word_error_rate("Set a timer for ten minutes.", "set a timer for 10 minutes")
    _check(result.wer == 0.0, f"word error rate of a digit spelling: {result}")
    result = wer.word_error_rate("what time is it", "what is it")
    _check(result.deletions == 1 and result.wer == 0.25, f"one deletion: {result}")
