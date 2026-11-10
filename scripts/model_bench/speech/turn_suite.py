"""Run 4, end of turn: Smart Turn v3.2 at every decision point of Raj's recordings.

One base-image container (`probes/turn_probe.py`) finds each utterance's decision points with the
material's frame detector, infers each point 5 times after 3 warm-ups and writes its raw rows into
the claimed `turn/smart-turn-v3.2/` directory; the host appends them, stamped, to
`turn/samples.jsonl`. The container gets no `--gpus`, at most 4 CPUs and 4 threads,
`--network none`, and the repository, material and weights read-only.

Stdlib only, like `bench_host.py`.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

from . import core, stt_suite

TURN_ENGINE = "smart-turn-v3.2"
TURN_PROBE = "scripts.model_bench.speech.probes.turn_probe"
TURN_FIELDS = (
    "utt_id",
    "kind",
    "cut_ms",
    "truth",
    "p_finished",
    "called_finished",
    "inference_ms",
)


def _engine(arms: Mapping[str, object] | None) -> dict:
    doc = arms if arms is not None else core.load_json(core.SPEECH_ARMS_PATH)
    turn = doc["turn"]
    if TURN_ENGINE not in turn:
        raise core.SpeechBenchError(f"speech_arms.json lists no turn engine {TURN_ENGINE!r}")
    return turn[TURN_ENGINE]


def turn_container_argv(
    *,
    repo_dir: str,
    material_dir: str,
    weights_dir: str,
    out_dir: str,
    reps: int = 5,
    warmups: int = 3,
    utt_ids: Sequence[str] = (),
    arms: Mapping[str, object] | None = None,
) -> list[str]:
    """`docker run` argv for run 4: Smart Turn on the CPU over every decision point."""
    cfg = _engine(arms)
    if cfg.get("device") != "cpu":
        raise core.SpeechBenchError(f"{TURN_ENGINE} runs on the CPU, not {cfg.get('device')!r}")
    threads = stt_suite.cpu_threads(cfg, TURN_ENGINE)
    argv = stt_suite.docker_head("mist-bench-turn") + stt_suite.resource_args(cfg, TURN_ENGINE)
    argv += stt_suite.probe_mounts(
        repo_dir=repo_dir, material_dir=material_dir, weights_dir=weights_dir, out_dir=out_dir
    )
    argv += [str(cfg["image"]), "python", "-m", TURN_PROBE]
    argv += ["--material", stt_suite.CONTAINER_MATERIAL, "--weights", stt_suite.CONTAINER_WEIGHTS]
    argv += ["--out", stt_suite.CONTAINER_OUT, "--threads", str(threads)]
    argv += ["--reps", str(reps), "--warmups", str(warmups)]
    for uid in utt_ids:
        argv += ["--utt", uid]
    return argv


def turn_row(raw: Mapping[str, object]) -> dict:
    """A `turn/samples.jsonl` row: the plan's fields first, then the probe's extras."""
    row = {key: raw[key] for key in TURN_FIELDS}
    row.update({key: value for key, value in raw.items() if key not in row})
    return row


def cmd_turn(args: argparse.Namespace) -> int:
    speech = core.speech_root(args.results_root, args.run, repo_root=core.REPO_ROOT)
    arms = core.load_json(core.SPEECH_ARMS_PATH)
    _engine(arms)
    material = args.material_dir.resolve()
    stt_suite.material_manifest(material, args.utt)
    weights = args.weights_dir.resolve()
    if not weights.is_dir():
        raise core.SpeechBenchError(f"weights directory {weights} does not exist")
    warnings = stt_suite.check_pins(args, speech, [TURN_ENGINE], weights)
    stt_suite.refuse_existing([speech / "turn" / TURN_ENGINE])
    for warning in warnings:
        print(f"[warn] {warning}")

    git_sha = core.current_git_sha(core.REPO_ROOT)
    claim = core.claim_output(args.results_root, args.run, "turn", TURN_ENGINE, git_sha=git_sha)
    argv = turn_container_argv(
        repo_dir=core.REPO_ROOT.resolve().as_posix(),
        material_dir=material.as_posix(),
        weights_dir=weights.as_posix(),
        out_dir=claim.path.resolve().as_posix(),
        reps=args.reps,
        warmups=args.warmups,
        utt_ids=args.utt,
        arms=arms,
    )
    print(f"[run] turn {TURN_ENGINE}")
    proc = run_container(argv)
    stt_suite.save_container_log(claim.path / "container.log", proc)
    if proc.returncode != 0:
        print(f"[FAIL] turn: exit {proc.returncode}; log {claim.path / 'container.log'}")
        return 1
    rows = [turn_row(raw) for raw in stt_suite.read_raw_rows(claim.path / "raw.jsonl")]
    with claim.shared_jsonl("samples.jsonl") as writer:
        for row in rows:
            writer.write(row)
    right = sum(1 for r in rows if bool(r["called_finished"]) == (r["truth"] == "finished"))
    times = core.summarize(float(str(r["inference_ms"])) for r in rows)
    print(
        f"[ok] turn {TURN_ENGINE}: {right} of {len(rows)} points called right; "
        f"inference median {times.median} ms, p90 {times.p90} ms"
    )
    return 0


def run_container(argv: list[str]):
    """The container runner; a seam the tests replace."""
    return stt_suite.run_container(argv)


def register(subparsers: argparse._SubParsersAction) -> None:
    turn = subparsers.add_parser("turn", help="run 4: Smart Turn at every decision point")
    core.add_results_args(turn)
    turn.add_argument(
        "--material-dir", type=Path, required=True, help="material directory with manifest.json"
    )
    turn.add_argument(
        "--weights-dir",
        type=Path,
        required=True,
        help="speech weights directory, mounted read-only",
    )
    turn.add_argument("--utt", action="append", default=[], help="limit to these utterance ids")
    turn.add_argument("--reps", type=int, default=5)
    turn.add_argument("--warmups", type=int, default=3)
    stt_suite.add_pins_args(turn)
    turn.set_defaults(func=cmd_turn)


def container_argv(ctx: core.PlanContext) -> list[tuple[str, list[str]]]:
    """Run 4's one container argv."""
    argv = turn_container_argv(
        repo_dir=ctx.repo_dir,
        material_dir=ctx.material_dir,
        weights_dir=ctx.speech_models_dir,
        out_dir=f"{ctx.speech_dir}/turn/{TURN_ENGINE}",
    )
    return [(f"turn {TURN_ENGINE}", argv)]


def validate_inputs(repo_root: Path) -> list[str]:
    """Check the `turn` section of speech_arms.json against what run 4 needs."""
    try:
        arms = core.load_json(repo_root / core.SPEECH_DIR_REL / "speech_arms.json")
        cfg = _engine(arms)
    except (core.InputError, core.SpeechBenchError, KeyError) as exc:
        return [f"speech_arms.json: {exc}"]
    problems = []
    model = cfg.get("model") or {}
    if model.get("file") != "smart-turn-v3.2-cpu.onnx" or model.get("quant") != "int8":
        problems.append("speech_arms.json: turn model must be the int8 smart-turn-v3.2-cpu.onnx")
    try:
        turn_container_argv(repo_dir="r", material_dir="m", weights_dir="w", out_dir="o", arms=arms)
    except (core.SpeechBenchError, KeyError, ValueError) as exc:
        problems.append(f"speech_arms.json: turn argv fails: {exc}")
    return problems


def selftest() -> None:
    """Check run 4's argv rules without docker: no GPU, 4 threads, offline, read-only inputs."""
    argv = turn_container_argv(repo_dir="/r", material_dir="/m", weights_dir="/w", out_dir="/o")

    def pairs(flag: str) -> list[str]:
        return [argv[i + 1] for i, arg in enumerate(argv[:-1]) if arg == flag]

    mounts = sorted(pairs("-v"))
    checks = (
        ("--gpus" not in argv, "the Smart Turn container got --gpus"),
        (pairs("--network") == ["none"], "the Smart Turn container has a network"),
        (pairs("--cpus") == ["4"], "the Smart Turn container is not limited to 4 CPUs"),
        ("OMP_NUM_THREADS=4" in pairs("-e"), "the Smart Turn container has no thread limit"),
        ("HF_HUB_OFFLINE=1" in pairs("-e"), "the Smart Turn container is not offline"),
        (pairs("--threads") == ["4"], "the Smart Turn probe is not given 4 threads"),
        (
            mounts == ["/m:/material:ro", "/o:/out", "/r:/work:ro", "/w:/weights:ro"],
            f"the Smart Turn mounts are {mounts}",
        ),
    )
    for ok, message in checks:
        if not ok:
            raise AssertionError(message)
