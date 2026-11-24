"""`student probe` and `student train`: fit, fine-tune and distill the Pocket TTS student.

`student probe --corpus C --run R [--batch N]` runs one GPU job in the `train-pocket` family
(`stages/student_train.py`) that trains the 24-layer base for `PROBE_STEPS` optimizer steps at
batch 2, then 4, then 8, each with accumulation to `EFFECTIVE_BATCH` (upstream requires it),
and each next batch only when the last one fitted (with `--batch N`, at N alone). A batch fits
when its peak `torch.cuda.max_memory_reserved` stays at or below the free VRAM the job saw
minus `HEADROOM_MIB`. The job writes `students/<R>/probe.json`: the largest batch that fits,
its accumulation, its peak and free VRAM in MiB, its optimizer steps/s and the hours the
pre-registered steps would take at that rate. The `--run` names the run that `student train`
then trains at that batch. When the smallest batch does not fit,
the command exits 3. The job asks the preflight for `PROBE_VRAM_GB` (7.5: batch 2 by a linear
fit of upstream's VRAM table, 6 GiB plus 0.75 GiB a sample) rather than the family's 16 GB,
which the 12 GB card cannot give.

`student train --corpus C --run R [--resume] [--stage finetune|distill] [--steps-finetune N]
[--steps-distill N] [--until HH:MM]` runs one GPU job that fine-tunes the 24-layer base with
upstream's `finetune.yaml`, then distills the 6-layer student with `depth_distill.yaml`, the
student warm-started from the released 6-layer student (`student_init_from`). It trains at the
probe's batch with accumulation to `EFFECTIVE_BATCH`, asks the preflight for the probe's peak
plus `HEADROOM_MIB`, and sets the checkpoint interval from the probe's rate so a checkpoint
lands about every `CHECKPOINT_TARGET_MINUTES`; the job also has the trainer save and restart
whenever a checkpoint is late, so one lands at least every `CHECKPOINT_MAX_MINUTES`. The steps are pre-registered here:
`STEPS_FINETUNE` and `STEPS_DISTILL`; `--steps-*` override them for a stand-in run, and a
resumed run keeps the steps it was planned with unless they are given again. `--until HH:MM` is
host local time; the job receives it as `--deadline-utc` (`deadline.py`). At the deadline, or on
SIGTERM, the job has the trainer save a checkpoint and exits 0 with `status: paused`;
`--resume` continues from the newest checkpoint.

Everything lands in `students/<R>/`: `probe.json`, `train-plan.json` (the host's plan),
`train.json` (the job's status, steps, losses, versions and file sha256s), the corpus copy the
trainer reads under `data/`, the model and trainer configs under `configs/`, and the trainer's
checkpoints under `checkpoints/finetune/` and `checkpoints/distill/`. Each distill checkpoint
the trainer keeps has its student weights, alone and in the runtime's key layout, at
`checkpoints/weights/<checkpoint>/model.safetensors` (`weights_rel`), which `student score`
and `student export` read.

Nothing printed carries a transcript, a line's text or a clip's path: only ids, counts, sizes,
rates and hours. The stage imports the names and helpers below, so host and job share one
definition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import PurePosixPath

from .. import deadline as deadline_mod
from .. import jobs
from .. import root as root_mod
from ..errors import GateError, InputError, JobError
from ..records import JobInput, read_json, sha256_file, write_json
from . import clean as clean_cmd
from . import corpus as corpus_cmd
from . import extract as extract_cmd
from . import teacher as teacher_cmd

SCHEMA = 1
FAMILY = "train-pocket"
MODULE = "scripts.voice_data.stages.student_train"
# The probe's preflight need: batch 2 by the linear fit of upstream's VRAM table (E).
PROBE_VRAM_GB = 7.5
PROBE_STEPS = 50
PROBE_BATCHES = (2, 4, 8)
HEADROOM_MIB = 512
EFFECTIVE_BATCH = 64
# Pre-registered steps (E): the probe's projection of them decides whether training moves to a
# rented GPU.
STEPS_FINETUNE = 8000
STEPS_DISTILL = 20000
CHECKPOINT_TARGET_MINUTES = 20
CHECKPOINT_MAX_MINUTES = 30
FINETUNE, DISTILL = "finetune", "distill"
STAGES = (FINETUNE, DISTILL)
RUNNING, PAUSED, DONE = "running", "paused", "done"
DEADLINE, SIGNAL = "deadline", "signal"
PROBE_NAME = "probe.json"
PLAN_NAME = "train-plan.json"
TRAIN_NAME = "train.json"
OUTPUTS_NAME = "outputs.json"
DATA_DIR = "data"
CONFIGS_DIR = "configs"
CHECKPOINTS_DIR = "checkpoints"
WEIGHTS_DIR = "weights"
WEIGHTS_NAME = "model.safetensors"
# Where a job mounts the root: corpus rows name their clips under it.
JOB_DATA = "/data"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_CHECKPOINT = re.compile(r"checkpoint_[0-9]{8}")


def run_rel(run_id: str) -> str:
    """Return a student run's folder, relative to the root."""
    return f"students/{run_id}"


def stage_rel(run_id: str, stage: str) -> str:
    """Return the trainer's run_dir for a stage, relative to the root."""
    return f"{run_rel(run_id)}/{CHECKPOINTS_DIR}/{stage}"


def checkpoint_name(step: int) -> str:
    """Return upstream's name for the checkpoint of `step`, without its `.pt`."""
    return f"checkpoint_{step:08d}"


def weights_rel(run_id: str, checkpoint: str) -> str:
    """Return a distill checkpoint's runtime-layout student weights, relative to the root.

    They sit in their own tree rather than beside the trainer's `checkpoint_*.pt` files, so a
    folder named like a checkpoint never meets upstream's checkpoint search or pruning.

    Raises:
        InputError: `checkpoint` is not a `checkpoint_<8 digits>` name.
    """
    if not _CHECKPOINT.fullmatch(checkpoint):
        raise InputError("refused: a checkpoint is named checkpoint_<8 digits>")
    return f"{run_rel(run_id)}/{CHECKPOINTS_DIR}/{WEIGHTS_DIR}/{checkpoint}/{WEIGHTS_NAME}"


def accum_for(batch: int, effective: int = EFFECTIVE_BATCH) -> int:
    """Return the gradient accumulation that brings `batch` to at least `effective`."""
    return max(1, math.ceil(effective / batch))


def projected_hours(steps_per_s: float, steps: int) -> float:
    """Return the hours `steps` optimizer steps take at the probe's rate.

    The probe measures optimizer steps at its batch with accumulation to the effective batch,
    as training runs them. Both stages are projected at the fine-tune's rate; distillation
    backpropagates through 6 layers rather than 24, so this errs long.
    """
    return round(steps / steps_per_s / 3600.0, 3)


def checkpoint_every(steps_per_s: float) -> int:
    """Return the optimizer steps between checkpoints for about `CHECKPOINT_TARGET_MINUTES`."""
    return max(1, int(steps_per_s * CHECKPOINT_TARGET_MINUTES * 60))


def vram_gb_for(peak_vram_mib: float) -> float:
    """Return the preflight need of a training job: the probe's peak plus the headroom."""
    return math.ceil((peak_vram_mib + HEADROOM_MIB) / 1024 * 10) / 10


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add the `student` command with its `probe` and `train` steps."""
    parser = subparsers.add_parser(
        "student",
        help="the Pocket TTS student: fit probe and training",
        description=(
            "Probe how large a batch fits the card, then fine-tune the 24-layer Pocket TTS base "
            "on a corpus and distill the 6-layer student, resumably."
        ),
    )
    steps = parser.add_subparsers(dest="step", metavar="STEP", required=True)
    probe = steps.add_parser("probe", help="measure the batch that fits (a GPU job)")
    probe.add_argument("--corpus", required=True, help="the corpus version under corpus/")
    probe.add_argument("--run", required=True, help="the run id, a folder under students/")
    probe.add_argument("--batch", type=int, help="probe this batch alone")
    probe.set_defaults(func=run_probe)
    train = steps.add_parser("train", help="fine-tune, then distill (a GPU job)")
    train.add_argument("--corpus", required=True, help="the corpus version under corpus/")
    train.add_argument("--run", required=True, help="the probed run id under students/")
    train.add_argument("--resume", action="store_true", help="continue an existing run")
    train.add_argument("--stage", choices=STAGES, help="run this stage alone")
    train.add_argument("--steps-finetune", type=int, help=f"default {STEPS_FINETUNE}")
    train.add_argument("--steps-distill", type=int, help=f"default {STEPS_DISTILL}")
    train.add_argument("--until", metavar="HH:MM", help="checkpoint and pause then (local)")
    train.set_defaults(func=run_train)


def make_docker() -> jobs.DockerOps:
    """Return the docker CLI; tests replace this function."""
    return jobs.HostDocker()


def image_tag(family: str) -> str:
    """Return the local image of `family` at HEAD, as `images build` tags it."""
    return extract_cmd.image_tag(family)


def utc_now() -> datetime:
    """Return the current time in UTC; tests replace this function."""
    return datetime.now(UTC)


def _check_id(value: str, what: str) -> None:
    if not _ID.fullmatch(value) or ".." in value:
        raise InputError(f"refused: a {what} is one folder name")


def _job_id(root: root_mod.Root) -> str:
    now = utc_now()
    number = 1
    while (root.path / "runs" / root_mod.job_id(now, number)).exists():
        number += 1
    return root_mod.job_id(now, number)


# The corpus the job reads.


def _clip_rel(path: object, number: int, rel: str) -> str:
    prefix = f"{JOB_DATA}/"
    if not isinstance(path, str) or not path.startswith(prefix):
        raise InputError(f"refused: {rel} row {number} names no clip under {JOB_DATA}")
    pure = PurePosixPath(path.removeprefix(prefix))
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise InputError(f"refused: {rel} row {number} names no clip under {JOB_DATA}")
    return pure.as_posix()


def corpus_inputs(root: root_mod.Root, version: str) -> tuple[JobInput, ...]:
    """Return a built corpus's job inputs: its two splits and the run folders of their clips.

    The corpus rows name each clip as `/data/<root-relative path>`, so the job mounts the
    teacher run and the seed run the manifest names, and every row's clip must lie in one of
    them. A folder's sha256 is a digest of the sha256s of the clips the splits name in it.

    Raises:
        InputError: The corpus has no manifest (its gate did not pass), a split changed since
            the manifest, or a row names a clip outside the mounted runs or missing. Errors name
            row numbers, never text or paths.
    """
    _check_id(version, "corpus version")
    folder = corpus_cmd.corpus_rel(version)
    manifest_path = root.path / folder / corpus_cmd.MANIFEST_NAME
    if not manifest_path.is_file():
        raise InputError(f"refused: corpus {version} has no manifest; run corpus build")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("version") != version:
        raise InputError(f"refused: corpus {version}'s manifest is not this version's")
    recorded = {
        f.get("rel"): f.get("sha256") for f in manifest.get("files") or [] if isinstance(f, dict)
    }
    splits = [f"{folder}/{name}" for name in (corpus_cmd.TRAIN_NAME, corpus_cmd.VALID_NAME)]
    inputs: list[JobInput] = []
    for rel in splits:
        path = root.path / rel
        if not path.is_file() or sha256_file(path) != recorded.get(rel):
            raise InputError(f"refused: {rel} is missing or changed since its manifest")
        inputs.append(JobInput(rel=rel, sha256=recorded[rel]))  # type: ignore[arg-type]
    dirs = [teacher_cmd.run_rel(str(manifest.get("takes_run")))]
    if manifest.get("seed_run") is not None:
        dirs.append(clean_cmd.run_rel(str(manifest["seed_run"])))
    clips: dict[str, list[str]] = {d: [] for d in dirs}
    for rel in splits:
        text = (root.path / rel).read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InputError(f"refused: {rel} row {number} is not JSON") from exc
            clip = _clip_rel(row.get("path") if isinstance(row, dict) else None, number, rel)
            owner = next((d for d in dirs if clip.startswith(f"{d}/")), None)
            if owner is None or not (root.path / clip).is_file():
                raise InputError(f"refused: {rel} row {number}'s clip is missing or unmounted")
            if clip not in clips[owner]:
                clips[owner].append(clip)
    for folder_rel in dirs:
        parts = [sha256_file(root.path / clip) for clip in sorted(clips[folder_rel])]
        digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
        inputs.append(JobInput(rel=folder_rel, sha256=digest))
    return tuple(inputs)


def _run_job(root: root_mod.Root, job: jobs.JobSpec, name: str) -> Mapping[str, object]:
    jobs.executor_for("local", root=root, docker=make_docker()).run(job)
    rel = f"{job.outputs}/{name}"
    try:
        doc = read_json(root.path / rel)
    except InputError as exc:
        raise JobError(f"job {job.job_id}: {rel} is missing") from exc
    if not isinstance(doc, dict):
        raise JobError(f"job {job.job_id}: {rel} is not a JSON object")
    return doc


def _num(value: object, digits: int = 3) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "none" if value is None else "?"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}" if math.isfinite(value) else "?"


# `student probe`.


def run_probe(args: argparse.Namespace) -> int:
    """Run the fit probe and print its result; exit 3 when the smallest batch does not fit."""
    root = root_mod.open_root(args.root)
    _check_id(args.run, "run id")
    if args.batch is not None and args.batch < 1:
        raise InputError("refused: --batch is at least 1")
    folder = run_rel(args.run)
    if (root.path / folder / TRAIN_NAME).exists():
        raise InputError(f"refused: run {args.run} has started training; probe a new run")
    inputs = corpus_inputs(root, args.corpus)
    batches = (args.batch,) if args.batch is not None else PROBE_BATCHES
    job = jobs.JobSpec(
        job_id=_job_id(root),
        family=FAMILY,
        image=image_tag(FAMILY),
        module=MODULE,
        args=(
            "probe",
            "--run",
            args.run,
            "--corpus",
            args.corpus,
            "--batches",
            *(str(b) for b in batches),
        ),
        inputs=inputs,
        outputs=folder,
        gpu=True,
        network=False,
        vram_gb=PROBE_VRAM_GB,
    )
    probe = _run_job(root, job, PROBE_NAME)
    for line in probe_lines(args.run, probe):
        print(line)
    if probe.get("batch") is None:
        raise GateError(
            f"student probe {args.run}: batch {batches[0]} does not fit the free VRAM less "
            f"{HEADROOM_MIB} MiB"
        )
    return 0


def probe_lines(run_id: str, probe: Mapping[str, object]) -> list[str]:
    """Return what `student probe` prints: each batch's fit, then the batch chosen."""
    lines = [f"student probe {run_id}: free {_num(probe.get('free_vram_mib'), 0)} MiB"]
    trials = probe.get("trials")
    for trial in trials if isinstance(trials, list) else []:
        if not isinstance(trial, Mapping):
            continue
        fit = "fits" if trial.get("fits") else "out of memory" if trial.get("oom") else "too big"
        lines.append(
            f"  batch {_num(trial.get('batch'))}: peak {_num(trial.get('peak_vram_mib'), 0)} "
            f"MiB, {_num(trial.get('steps_per_s'))} steps/s, {fit}"
        )
    if probe.get("batch") is None:
        lines.append("  no batch fits")
    else:
        lines.append(
            f"  batch {_num(probe.get('batch'))} x accum {_num(probe.get('accum'))}: "
            f"{_num(probe.get('projected_hours'))} h projected"
        )
    return lines


# `student train`.


def _read_probe(root: root_mod.Root, run_id: str, corpus: str) -> Mapping[str, object]:
    path = root.path / run_rel(run_id) / PROBE_NAME
    if not path.is_file():
        raise InputError(f"refused: run {run_id} has no probe; run student probe first")
    probe = read_json(path)
    if not isinstance(probe, dict) or probe.get("schema") != SCHEMA:
        raise InputError(f"refused: run {run_id}'s probe is not a schema {SCHEMA} probe")
    if probe.get("corpus") != corpus:
        raise InputError(f"refused: run {run_id} was probed on another corpus")
    batch, rate, peak = probe.get("batch"), probe.get("steps_per_s"), probe.get("peak_vram_mib")
    if batch is None:
        raise InputError(f"refused: run {run_id}'s probe found no batch that fits")
    for value in (batch, rate, peak):
        if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
            raise InputError(f"refused: run {run_id}'s probe is incomplete")
    return probe


def _steps(value: int | None, planned: object, default: int, flag: str) -> int:
    if value is not None:
        if value < 1:
            raise InputError(f"refused: {flag} is at least 1")
        return value
    return int(planned) if isinstance(planned, int) and not isinstance(planned, bool) else default


def run_train(args: argparse.Namespace) -> int:
    """Plan the run, run the training job, and print its status."""
    root = root_mod.open_root(args.root)
    _check_id(args.run, "run id")
    folder = run_rel(args.run)
    deadline = deadline_mod.deadline_utc(args.until, now=utc_now()) if args.until else None
    probe = _read_probe(root, args.run, args.corpus)
    plan_path = root.path / folder / PLAN_NAME
    started = (root.path / folder / TRAIN_NAME).exists() or plan_path.exists()
    if started and not args.resume:
        raise InputError(f"refused: run {args.run} exists; pass --resume to continue it")
    if args.resume and not plan_path.exists():
        raise InputError(f"refused: run {args.run} has no training to resume")
    old: Mapping[str, object] = {}
    if args.resume:
        doc = read_json(plan_path)
        if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
            raise InputError(f"refused: run {args.run}'s plan is not a schema {SCHEMA} plan")
        old = doc
    old_steps = old.get("steps") if isinstance(old.get("steps"), Mapping) else {}
    steps = {
        FINETUNE: _steps(
            args.steps_finetune, old_steps.get(FINETUNE), STEPS_FINETUNE, "--steps-finetune"  # type: ignore[union-attr]
        ),
        DISTILL: _steps(
            args.steps_distill, old_steps.get(DISTILL), STEPS_DISTILL, "--steps-distill"  # type: ignore[union-attr]
        ),
    }
    batch = int(probe["batch"])  # type: ignore[call-overload]
    accum = accum_for(batch)
    rate = float(probe["steps_per_s"])  # type: ignore[arg-type]
    every = checkpoint_every(rate)
    vram = vram_gb_for(float(probe["peak_vram_mib"]))  # type: ignore[arg-type]
    inputs = corpus_inputs(root, args.corpus)
    plan = {
        "schema": SCHEMA,
        "run": args.run,
        "corpus": args.corpus,
        "batch": batch,
        "accum": accum,
        "effective_batch": batch * accum,
        "ckpt_freq": every,
        "vram_gb": vram,
        "steps": steps,
    }
    write_json(root, f"{folder}/{PLAN_NAME}", plan)
    extra: list[str] = []
    if args.stage:
        extra += ["--stage", args.stage]
    if deadline is not None:
        extra += ["--deadline-utc", deadline_mod.format_utc(deadline)]
    job = jobs.JobSpec(
        job_id=_job_id(root),
        family=FAMILY,
        image=image_tag(FAMILY),
        module=MODULE,
        args=(
            "train",
            "--run",
            args.run,
            "--corpus",
            args.corpus,
            "--batch",
            str(batch),
            "--ckpt-freq",
            str(every),
            "--steps-finetune",
            str(steps[FINETUNE]),
            "--steps-distill",
            str(steps[DISTILL]),
            *extra,
        ),
        inputs=inputs,
        outputs=folder,
        gpu=True,
        network=False,
        vram_gb=vram,
    )
    train = _run_job(root, job, TRAIN_NAME)
    for line in train_lines(args.run, train):
        print(line)
    return 0


def train_lines(run_id: str, train: Mapping[str, object]) -> list[str]:
    """Return what `student train` prints: the status and each stage's steps and last loss."""
    status = train.get("status") if train.get("status") in (RUNNING, PAUSED, DONE) else "?"
    by = train.get("stopped_by")
    reason = f" ({by})" if by in (DEADLINE, SIGNAL) else ""
    lines = [f"student train {run_id}: {status}{reason}"]
    steps = train.get("steps")
    losses = train.get("losses")
    for stage in STAGES:
        entry = steps.get(stage) if isinstance(steps, Mapping) else None
        if not isinstance(entry, Mapping):
            continue
        trail = losses.get(stage) if isinstance(losses, Mapping) else None
        last = trail[-1] if isinstance(trail, list) and trail else None
        loss = last[1] if isinstance(last, list) and len(last) == 2 else None
        lines.append(
            f"  {stage}: step {_num(entry.get('done'))} of {_num(entry.get('max'))}, "
            f"loss {_num(loss, 4)}"
        )
    return lines
