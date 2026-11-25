"""`student probe`, `train`, `score` and `export`: the Pocket TTS student, trained and exported.

`student probe --corpus C --run R [--batch N]` runs one GPU job in the `train-pocket` family
(`stages/student_train.py`) that trains the 24-layer base for `PROBE_STEPS` optimizer steps at
batch 2, then 4, then 8, each with accumulation to `EFFECTIVE_BATCH` (upstream requires it),
and each next batch only when the last one fitted (with `--batch N`, at N alone). A batch fits
when its peak `torch.cuda.max_memory_reserved` stays at or below the free VRAM the job saw
minus `HEADROOM_MIB`. The job writes `students/<R>/probe.json`: the largest batch that fits,
its accumulation, its peak and free VRAM in MiB, its optimizer steps/s, the hours the
pre-registered steps would take at that rate, and `save_s`, the longest checkpoint save the
trials timed, in seconds to one decimal. The `--run` names the run that `student train`
then trains at that batch. When the smallest batch does not fit,
the command exits 3. The job asks the preflight for `PROBE_VRAM_GB` (7.5: batch 2 by a linear
fit of upstream's VRAM table, 6 GiB plus 0.75 GiB a sample) rather than the family's 16 GB,
which the 12 GB card cannot give.

`student train --corpus C --run R [--resume] [--stage finetune|distill] [--steps-finetune N]
[--steps-distill N] [--keep-last N] [--until HH:MM]` runs one GPU job that fine-tunes the
24-layer base with upstream's `finetune.yaml`, then distills the 6-layer student with
`depth_distill.yaml`, the student warm-started from the released 6-layer student
(`student_init_from`). It trains at the
probe's batch with accumulation to `EFFECTIVE_BATCH`, asks the preflight for the probe's peak
plus `HEADROOM_MIB`, and sets the checkpoint interval from the probe's rate so a checkpoint
lands about every `CHECKPOINT_TARGET_MINUTES`; the job also has the trainer save and restart
whenever a checkpoint is late, so one lands at least every `CHECKPOINT_MAX_MINUTES`. The steps are pre-registered here:
`STEPS_FINETUNE` and `STEPS_DISTILL`; `--steps-*` override them for a stand-in run, and a
resumed run keeps the steps it was planned with unless they are given again. `--until HH:MM` is
host local time; the job receives it as `--deadline-utc` (`deadline.py`). At the deadline, or on
SIGTERM, the job has the trainer save a checkpoint and exits 0 with `status: paused`;
`--resume` continues from the newest checkpoint. Each stage keeps its last `--keep-last`
checkpoints (default `KEEP_LAST`, upstream's `num_ckpt_keep`, which replaces
`depth_distill.yaml`'s 999), and a resumed run keeps the number it was planned with unless it is
given again. Validation runs at every checkpoint interval over the validation split's whole
batches, so each interval's checkpoint has a validation loss; the distillation also keeps the
weights of the checkpoint with the lowest, `train.json`'s `best`.

Everything lands in `students/<R>/`: `probe.json`, `train-plan.json` (the host's plan),
`train.json` (the job's status, steps, losses, best, versions and file sha256s), the corpus copy
the trainer reads under `data/`, the model and trainer configs under `configs/`, and the trainer's
checkpoints under `checkpoints/finetune/` and `checkpoints/distill/`. Each distill checkpoint
the trainer keeps, and the best, has its student weights, alone and in the runtime's layout, at
`checkpoints/weights/<checkpoint>/model.safetensors` (`weights_rel`), which `student score`
and `student export` read.

`student score --run R [--checkpoint C]` gates a distill checkpoint (default: the newest with
weights) against the anchor, the identity whose neutral clip the run's teacher takes were
conditioned on (corpus manifest, then the teacher run's plan), and against the seed target those
takes were scored with (the teacher run's score plan; takes scored under the seed fallback have
none, and are refused). It runs two jobs of `stages/student_score.py` on
`students/<R>/scores/<C>/`: one in `train-pocket` renders the corpus's validation lines and the
speech bench's 26 items (`BENCH_LINES_REL`: its lines and acknowledgements) with the checkpoint,
at `RENDER_VRAM_GB` (E: inference of the 6-layer student) rather than the family's 16, and one in
the `score` family, at that family's own VRAM need, scores them. The gates are the bench's, as
`gates_default.json` notes them: the highest line's CER at most `bench.cer_max` (the mean is
recorded too);
on each embedder, the renders' median SIM to the anchor clip over the seed target's
leave-one-out SIM median (`distribution[embedder].median`), the lower of the two at least
`bench.sim_median_ratio_min`; and the renders' median F0 standard deviation over the seed's for
the rendered state, neutral (the target's `profile.f0_std_hz`), within `bench.f0_std_tolerance`
of 1. The result is `students/<R>/score.json` {cer, sim_ratio, f0_sd_ratio, pass}; a failed
gate exits 3.

`student export --run R --checkpoint C --voice-id V --anchor A [--status placeholder|frozen]`
writes the checkpoint as the schema-2 identity `voices/<V>/` on the host, with no job: the
checkpoint's tensor names and shapes must equal the stock `languages/english/model.safetensors`
train-pocket fetches, both read from their safetensors headers. The identity holds identity A's
neutral clip and transcript, the weights with their sha256, `train.json`'s tokenizer sha256, and
the training tokenizer only when it is not the stock runtime tokenizer. Its base is the realtime
adapter's `MODEL_PIN`, read from `backend/realtime/engines/pocket_tts.py`'s source rather than
imported, so this module stays standard-library only. The cached files are found under
`cache/hf/hub/` by the Hugging Face cache layout at the pinned revision, the folder
`stages/fetch.py::snapshot_dir` resolves inside a job, never by repository id.

Nothing printed carries a transcript, a line's text or a clip's path: only ids, counts, sizes,
rates, hours and scores. The stages import the names and helpers below, so host and job share
one definition.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from .. import deadline as deadline_mod
from .. import gates as gates_mod
from .. import identity as identity_mod
from .. import jobs
from .. import root as root_mod
from .. import target as target_mod
from ..errors import GateError, InputError, JobError
from ..records import JobInput, read_json, sha256_file, write_json
from ..states import NEUTRAL_STATE
from . import clean as clean_cmd
from . import corpus as corpus_cmd
from . import extract as extract_cmd
from . import fetch as fetch_cmd
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
# Checkpoints each stage keeps (upstream's `num_ckpt_keep`); the best distill weights are kept too.
KEEP_LAST = 3
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
_SHA256 = re.compile(r"[0-9a-f]{64}")

# `student score`: two jobs of one stage module on `students/<R>/scores/<C>/`.
SCORE_MODULE = "scripts.voice_data.stages.student_score"
SCORE_FAMILY = teacher_cmd.SCORE_FAMILY
# The render job's preflight need (E: inference of the 6-layer student), not train-pocket's 16.
RENDER_VRAM_GB = 4.0
RENDER, SCORE = "render", "score"
DEVICE = "cuda"
SCORES_DIR = "scores"
SCORE_NAME = "score.json"
SCORE_PLAN_NAME = "score-plan.json"
RENDERS_NAME = "renders.json"
RENDERS_DIR = "renders"
LINE_SCORES_NAME = "lines.jsonl"
VALID, BENCH = "valid", "bench"
# The speech bench's 26 items, relative to the `scripts` folder (the image carries it too):
# its lines and its acknowledgements, as the bench speaks them.
BENCH_LINES_REL = "model_bench/speech/lines.json"
BENCH_SECTIONS = ("lines", "acknowledgements")
# Each gate's name in `score.json` and its key in `gates_default.json`.
GATE_KEYS = {
    "cer_max": "bench.cer_max",
    "sim_median_ratio_min": "bench.sim_median_ratio_min",
    "f0_std_tolerance": "bench.f0_std_tolerance",
}
CER, SIM_RATIO, F0_SD_RATIO = "cer", "sim_ratio", "f0_sd_ratio"

# `student export`. Cached train-pocket files: the `families.json` model and the path in its
# snapshot (the stage's `STUDENT_WEIGHTS`, `STUDENT_TOKENIZER` and `BASE_TOKENIZER`).
STOCK_WEIGHTS = ("pocket-tts", "languages/english/model.safetensors")
RUNTIME_TOKENIZER = ("pocket-tts-runtime-tokenizer", "languages/english/tokenizer.json")
BASE_TOKENIZER = ("pocket-tts", "languages/english_2026-04_24l/tokenizer.json")
# Where a tokenizer the student was trained with may be cached, the runtime's first.
TOKENIZERS = (RUNTIME_TOKENIZER, BASE_TOKENIZER)
# The Hugging Face hub cache under the root, mounted at `/cache` with `HF_HOME=/cache/hf`.
HUB_REL = "cache/hf/hub"
# The realtime Pocket adapter, whose `ENGINE_ID`, `REPOSITORY` and `REVISION` name the base.
POCKET_ADAPTER = (
    Path(__file__).resolve().parents[3] / "backend" / "realtime" / "engines" / "pocket_tts.py"
)
# A safetensors header larger than this is refused rather than read.
HEADER_MAX_BYTES = 100 << 20


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
    """Add the `student` command with its `probe`, `train`, `score` and `export` steps."""
    parser = subparsers.add_parser(
        "student",
        help="the Pocket TTS student: fit probe, training, checkpoint gate and export",
        description=(
            "Probe how large a batch fits the card, then fine-tune the 24-layer Pocket TTS base "
            "on a corpus and distill the 6-layer student, resumably; gate a checkpoint against "
            "the anchor, and export it as a voice identity."
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
    train.add_argument(
        "--keep-last", type=int, metavar="N",
        help=f"checkpoints each stage keeps, besides the best distill one (default {KEEP_LAST})",
    )  # fmt: skip
    train.add_argument("--until", metavar="HH:MM", help="checkpoint and pause then (local)")
    train.set_defaults(func=run_train)
    sco = steps.add_parser("score", help="gate a checkpoint by CER, SIM and F0 (two GPU jobs)")
    sco.add_argument("--run", required=True, help="the trained run id under students/")
    sco.add_argument("--checkpoint", help="a distill checkpoint (default: the newest)")
    sco.set_defaults(func=run_score)
    exp = steps.add_parser("export", help="write a checkpoint as a schema-2 voice identity")
    exp.add_argument("--run", required=True, help="the trained run id under students/")
    exp.add_argument("--checkpoint", required=True, help="the distill checkpoint to export")
    exp.add_argument("--voice-id", required=True, help="the new identity under voices/")
    exp.add_argument("--anchor", required=True, help="the identity whose neutral clip to copy")
    exp.add_argument(
        "--status", choices=identity_mod.STATUSES, default=identity_mod.PLACEHOLDER,
        help="the identity's status (default placeholder)",
    )  # fmt: skip
    exp.set_defaults(func=run_export)


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


def _planned(value: int | None, planned: object, default: int, flag: str) -> int:
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
        FINETUNE: _planned(
            args.steps_finetune, old_steps.get(FINETUNE), STEPS_FINETUNE, "--steps-finetune"  # type: ignore[union-attr]
        ),
        DISTILL: _planned(
            args.steps_distill, old_steps.get(DISTILL), STEPS_DISTILL, "--steps-distill"  # type: ignore[union-attr]
        ),
    }
    keep_last = _planned(args.keep_last, old.get("keep_last"), KEEP_LAST, "--keep-last")
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
        "keep_last": keep_last,
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
            "--keep-last",
            str(keep_last),
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
    """Return what `student train` prints: the status, each stage's steps and last loss, the best.

    The best is the distill checkpoint with the lowest validation loss, once there is one.
    """
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
    best = train.get("best")
    if isinstance(best, Mapping) and _CHECKPOINT.fullmatch(str(best.get("checkpoint"))):
        lines.append(
            f"  best: {best['checkpoint']} at step {_num(best.get('step'))}, "
            f"valid loss {_num(best.get('valid_loss'), 4)}"
        )
    return lines


# What `student score` and `student export` read: the run, its checkpoints and the cache.


def scores_rel(run_id: str, checkpoint: str) -> str:
    """Return the folder a checkpoint's scoring writes, relative to the root."""
    return f"{run_rel(run_id)}/{SCORES_DIR}/{checkpoint}"


def newest_checkpoint(root: root_mod.Root, run_id: str) -> str:
    """Return the newest distill checkpoint of the run that has runtime-layout weights.

    Raises:
        InputError: The run has none.
    """
    folder = root.path / run_rel(run_id) / CHECKPOINTS_DIR / WEIGHTS_DIR
    names = (
        sorted(
            p.name
            for p in folder.iterdir()
            if _CHECKPOINT.fullmatch(p.name) and (p / WEIGHTS_NAME).is_file()
        )
        if folder.is_dir()
        else []
    )
    if not names:
        raise InputError(f"refused: run {run_id} has no distill checkpoint weights")
    return names[-1]


def checkpoint_weights(root: root_mod.Root, run_id: str, checkpoint: str) -> str:
    """Return a checkpoint's weights, relative to the root, refusing one that is missing.

    Raises:
        InputError: The name is not a checkpoint's, or its weights are missing.
    """
    rel = weights_rel(run_id, checkpoint)
    if not (root.path / rel).is_file():
        raise InputError(f"refused: run {run_id} has no weights for {checkpoint}")
    return rel


def read_train(root: root_mod.Root, run_id: str) -> Mapping[str, object]:
    """Return the run's `train.json`, which names the corpus and the tokenizer's sha256.

    Raises:
        InputError: It is missing, not schema 1, or lacks a corpus or a tokenizer sha256.
    """
    path = root.path / run_rel(run_id) / TRAIN_NAME
    if not path.is_file():
        raise InputError(f"refused: run {run_id} has no {TRAIN_NAME}; run student train")
    train = read_json(path)
    if not isinstance(train, dict) or train.get("schema") != SCHEMA:
        raise InputError(f"refused: run {run_id}'s {TRAIN_NAME} is not schema {SCHEMA}")
    corpus, sha = train.get("corpus"), train.get("tokenizer_sha256")
    if not isinstance(corpus, str) or not isinstance(sha, str) or not _SHA256.fullmatch(sha):
        raise InputError(f"refused: run {run_id}'s {TRAIN_NAME} lacks its corpus or tokenizer")
    _check_id(corpus, "corpus version")
    return train


def run_anchor(root: root_mod.Root, corpus: str) -> teacher_cmd.Anchor:
    """Return the anchor of the teacher run a corpus was built from, unchanged since then.

    Raises:
        InputError: The corpus or its teacher run's plan is missing, or the anchor's neutral
            clip changed since the teacher run (`teacher_cmd.read_anchor` names the rest).
    """
    manifest = read_json(root.path / corpus_cmd.corpus_rel(corpus) / corpus_cmd.MANIFEST_NAME)
    takes_run = manifest.get("takes_run") if isinstance(manifest, dict) else None
    if not isinstance(takes_run, str):
        raise InputError(f"refused: corpus {corpus}'s manifest names no teacher run")
    _check_id(takes_run, "teacher run id")
    plan = read_json(root.path / teacher_cmd.run_rel(takes_run) / teacher_cmd.PLAN_NAME)
    doc = plan.get("anchor") if isinstance(plan, dict) else None
    voice = doc.get("voice") if isinstance(doc, dict) else None
    if not isinstance(voice, str):
        raise InputError(f"refused: teacher run {takes_run}'s plan names no anchor")
    anchor = teacher_cmd.read_anchor(root, voice)
    if anchor.to_json() != doc:
        raise InputError(f"refused: identity {voice}'s neutral clip changed since run {takes_run}")
    return anchor


def _number(value: object) -> float | None:
    found = _finite(value)
    return found if found is not None and found > 0.0 else None


def seed_reference(root: root_mod.Root, corpus: str, voice_id: str) -> dict[str, object]:
    """Return what the seed target gives the gates: its SIM medians and its neutral F0 SD.

    The chain is `commands/corpus.py`'s: the corpus's teacher run, that run's score plan, and
    the seed target its takes were scored against. `sim_median` is each embedder's leave-one-out
    SIM median over the seed's confirmed segments (`distribution`, pooled over states);
    `f0_std_hz` is the median F0 standard deviation of the seed's neutral segments (`profile`),
    and `f0_std_spread_hz` that measure's standard deviation across them (`spread`), recorded.

    Raises:
        InputError: The takes were scored under the seed fallback (no target), the target is
            missing, not the anchor's voice, or not a seed target, or it lacks a SIM median on
            either embedder or a neutral F0 SD.
    """
    manifest = read_json(root.path / corpus_cmd.corpus_rel(corpus) / corpus_cmd.MANIFEST_NAME)
    takes_run = str(manifest.get("takes_run")) if isinstance(manifest, dict) else ""
    _check_id(takes_run, "teacher run id")
    plan_path = root.path / teacher_cmd.run_rel(takes_run) / teacher_cmd.SCORE_PLAN_NAME
    if not plan_path.is_file():
        raise InputError(f"refused: teacher run {takes_run} has no score plan")
    plan = read_json(plan_path)
    floor = plan.get("floor") if isinstance(plan, dict) else None
    doc = floor if isinstance(floor, Mapping) else {}
    target_id = doc.get("target")
    if doc.get("from") != teacher_cmd.FLOOR_SEED or not isinstance(target_id, str):
        raise InputError(
            f"refused: teacher run {takes_run}'s takes were scored under the seed fallback, "
            "with no seed target; student score gates against a seed target's numbers"
        )
    _check_id(target_id, "target id")
    record = target_mod.read_target(root, target_id)
    if record.voice_id != voice_id:
        raise InputError(f"refused: target {target_id} is not the voice of identity {voice_id}")
    if record.seed_run is None or record.mode == target_mod.IDENTITY:
        raise InputError(f"refused: target {target_id} has no seed distribution")
    sim_median: dict[str, float] = {}
    for name in teacher_cmd.EMBEDDERS:
        entry = record.distribution.get(name)
        median = _number(entry.get("median")) if isinstance(entry, Mapping) else None
        if median is None:
            raise InputError(f"refused: target {target_id} has no {name} SIM median")
        sim_median[name] = median
    state = record.states.get(NEUTRAL_STATE)
    neutral = state if isinstance(state, Mapping) else {}
    profile, spread = neutral.get("profile"), neutral.get("spread")
    f0_sd = _number(profile.get("f0_std_hz")) if isinstance(profile, Mapping) else None
    if f0_sd is None:
        raise InputError(f"refused: target {target_id} has no {NEUTRAL_STATE} F0 SD")
    f0_spread = _finite(spread.get("f0_std_hz")) if isinstance(spread, Mapping) else None
    return {
        "target": target_id,
        "state": NEUTRAL_STATE,
        "sim_median": sim_median,
        "f0_std_hz": f0_sd,
        "f0_std_spread_hz": f0_spread,
    }


def hub_file(root: root_mod.Root, model: tuple[str, str]) -> Path:
    """Return a cached train-pocket file on the host, by the hub cache's layout.

    `<root>/cache/hf/hub/<type>s--<org>--<name>/snapshots/<revision>/<file>` is the folder
    `fetch.snapshot_dir` resolves offline inside a job; the host reads it with the standard
    library, at the pinned revision and never through a `refs/` name.

    Raises:
        InputError: The model is not a pinned Hugging Face model of train-pocket, or the file is
            not in the cache.
    """
    models = jobs.load_families().get(FAMILY, {}).get("models", {})
    entry = models.get(model[0]) if isinstance(models, dict) else None
    if not isinstance(entry, dict) or entry.get("kind") != "hf":
        raise InputError(f"refused: {FAMILY} has no Hugging Face model {model[0]}")
    revision = entry.get("revision")
    if not fetch_cmd.pinned(revision):
        raise InputError(f"refused: {model[0]} has no pinned revision")
    repo_type = str(entry.get("repo_type") or "model")
    folder = f"{repo_type}s--{str(entry['repository']).replace('/', '--')}"
    path = root.path / HUB_REL / folder / "snapshots" / str(revision) / model[1]
    if not path.is_file():
        raise InputError(
            f"refused: {model[0]} {model[1]} is not in the cache; run fetch --family {FAMILY}"
        )
    return path


def training_tokenizer(root: root_mod.Root, sha256: str) -> tuple[tuple[str, str], Path]:
    """Return the cached tokenizer whose sha256 is the one the student was trained with.

    Raises:
        InputError: No cached train-pocket tokenizer has that sha256.
    """
    for model in TOKENIZERS:
        try:
            path = hub_file(root, model)
        except InputError:
            continue
        if sha256_file(path) == sha256:
            return model, path
    raise InputError("refused: the tokenizer the student was trained with is not in the cache")


def tensor_layout(path: Path, what: str) -> dict[str, tuple[int, ...]]:
    """Return each tensor's shape from a safetensors file's header, read with the stdlib.

    The format is an 8-byte little-endian header length, then that many bytes of JSON mapping
    each tensor's name to its dtype, shape and offsets, with an optional `__metadata__` entry.

    Raises:
        InputError: The file is missing or not a safetensors file, or holds no tensor.
    """
    refused = f"refused: {what} is not a safetensors file"
    try:
        with path.open("rb") as handle:
            head = handle.read(8)
            size = int.from_bytes(head, "little")
            if len(head) != 8 or size > HEADER_MAX_BYTES or 8 + size > path.stat().st_size:
                raise InputError(refused)
            header = json.loads(handle.read(size).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputError(refused) from exc
    if not isinstance(header, dict):
        raise InputError(refused)
    layout: dict[str, tuple[int, ...]] = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        shape = entry.get("shape") if isinstance(entry, dict) else None
        if not isinstance(shape, list) or not all(
            isinstance(d, int) and not isinstance(d, bool) and d >= 0 for d in shape
        ):
            raise InputError(refused)
        layout[name] = tuple(shape)
    if not layout:
        raise InputError(f"refused: {what} holds no tensor")
    return layout


def layout_differences(
    found: Mapping[str, tuple[int, ...]], stock: Mapping[str, tuple[int, ...]]
) -> dict[str, int]:
    """Return how many stock tensors are missing, extra or of another shape in `found`."""
    return {
        "missing": len(stock.keys() - found.keys()),
        "extra": len(found.keys() - stock.keys()),
        "reshaped": sum(found[k] != stock[k] for k in found.keys() & stock.keys()),
    }


def runtime_pin(path: Path = POCKET_ADAPTER) -> tuple[str, str, str]:
    """Return the realtime Pocket adapter's `ENGINE_ID`, `REPOSITORY` and `REVISION`.

    Read from the module's source as string constants, so the host imports no backend code.

    Raises:
        InputError: The file is missing, or one of the three is not a string constant there.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        raise InputError("refused: the realtime Pocket adapter is missing or unreadable") from exc
    values: dict[str, str] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            values[node.targets[0].id] = node.value.value
    names = ("ENGINE_ID", "REPOSITORY", "REVISION")
    if not all(name in values for name in names):
        raise InputError("refused: the realtime Pocket adapter does not name its pin")
    return values["ENGINE_ID"], values["REPOSITORY"], values["REVISION"]


# The lines `student score` renders.


def bench_lines_path() -> Path:
    """Return the speech bench's lines file beside this package, in the repository or image."""
    return Path(__file__).resolve().parents[2] / BENCH_LINES_REL


def read_bench_lines(path: Path) -> list[tuple[str, str]]:
    """Return the speech bench's items as (id, text): its lines, then its acknowledgements.

    The bench speaks both lists (`model_bench/speech/tts_suite.py::_item_ids`), 26 items in all,
    so the student is gated on the same items the realtime engine is benched on.

    Raises:
        InputError: The file is missing, a list is missing or empty, an item lacks a string id
            or text, or an id repeats.
    """
    doc = read_json(path)
    lines: list[tuple[str, str]] = []
    for section in BENCH_SECTIONS:
        rows = doc.get(section) if isinstance(doc, dict) else None
        if not isinstance(rows, list) or not rows:
            raise InputError(f"refused: the bench lines file has no {section}")
        for number, row in enumerate(rows, start=1):
            if not isinstance(row, dict) or not all(
                isinstance(row.get(k), str) and row.get(k) for k in ("id", "text")
            ):
                raise InputError(f"refused: bench {section} item {number} lacks an id or text")
            lines.append((row["id"], row["text"]))
    if len({line_id for line_id, _ in lines}) != len(lines):
        raise InputError("refused: the bench lines repeat an id")
    return lines


def read_valid_rows(path: Path) -> list[Mapping[str, object]]:
    """Return a corpus validation split's rows, each with a clip `path` and a `transcript`.

    Raises:
        InputError: The file is missing, or a row is not JSON or lacks either field. Errors
            name row numbers, never text or paths.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise InputError("refused: the validation split is missing or unreadable") from exc
    rows: list[Mapping[str, object]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InputError(f"refused: validation row {number} is not JSON") from exc
        if not isinstance(row, dict) or not all(
            isinstance(row.get(k), str) and row.get(k) for k in ("path", "transcript")
        ):
            raise InputError(f"refused: validation row {number} lacks a path or transcript")
        rows.append(row)
    return rows


def valid_id(number: int) -> str:
    """Return the id of the validation split's row `number` (from 1)."""
    return f"{VALID}-{number:04d}"


def bench_id(line_id: str) -> str:
    """Return the id of a bench line."""
    return f"{BENCH}-{line_id}"


def render_seed(line_id: str) -> int:
    """Return a line's render seed: 31 bits of its id's sha256, the same for every checkpoint."""
    return int(hashlib.sha256(line_id.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return float(value)


def gate_failures(metrics: Mapping[str, object], gates: Mapping[str, float]) -> list[str]:
    """Return the gates a checkpoint fails, in `score.json`'s order; a missing value fails.

    `cer`, the highest line's CER, is at most `cer_max`, `sim_ratio` at least `sim_median_ratio_min`, and `f0_sd_ratio`
    within `f0_std_tolerance` of 1 either way.
    """
    cer, sim, f0 = (_finite(metrics.get(k)) for k in (CER, SIM_RATIO, F0_SD_RATIO))
    failed = []
    if cer is None or not cer <= gates["cer_max"]:
        failed.append(CER)
    if sim is None or not sim >= gates["sim_median_ratio_min"]:
        failed.append(SIM_RATIO)
    if f0 is None or not abs(f0 - 1.0) <= gates["f0_std_tolerance"]:
        failed.append(F0_SD_RATIO)
    return failed


# `student score`.


def run_score(args: argparse.Namespace) -> int:
    """Render and score a checkpoint; write `score.json`; exit 3 when a gate fails."""
    root = root_mod.open_root(args.root)
    _check_id(args.run, "run id")
    checkpoint = args.checkpoint or newest_checkpoint(root, args.run)
    weights = checkpoint_weights(root, args.run, checkpoint)
    train = read_train(root, args.run)
    corpus = str(train["corpus"])
    anchor = run_anchor(root, corpus)
    seed = seed_reference(root, corpus, anchor.voice_id)
    gates = gates_mod.load(root.path)
    inputs = corpus_inputs(root, corpus)
    valid_rel = f"{corpus_cmd.corpus_rel(corpus)}/{corpus_cmd.VALID_NAME}"
    valid_input = next(i for i in inputs if i.rel == valid_rel)
    valid_n = len(read_valid_rows(root.path / valid_rel))
    if valid_n == 0:
        raise InputError(f"refused: corpus {corpus} has no validation line")
    bench_path = bench_lines_path()
    bench = read_bench_lines(bench_path)
    tokenizer, tokenizer_path = training_tokenizer(root, str(train["tokenizer_sha256"]))
    folder = scores_rel(args.run, checkpoint)
    weights_input = JobInput(rel=weights, sha256=sha256_file(root.path / weights))
    lines = [{"id": valid_id(n), "set": VALID} for n in range(1, valid_n + 1)]
    lines += [{"id": bench_id(line_id), "set": BENCH} for line_id, _ in bench]
    plan = {
        "schema": SCHEMA,
        "run": args.run,
        "checkpoint": checkpoint,
        "weights": weights,
        "weights_sha256": weights_input.sha256,
        "corpus": corpus,
        "valid": valid_rel,
        "valid_sha256": valid_input.sha256,
        "bench_lines": BENCH_LINES_REL,
        "bench_sha256": sha256_file(bench_path),
        "lines": lines,
        "anchor": anchor.to_json(),
        "seed": seed,
        "tokenizer": {
            "model": tokenizer[0],
            "file": tokenizer[1],
            "sha256": sha256_file(tokenizer_path),
        },
        "embedders": list(teacher_cmd.EMBEDDERS),
        "gates": {name: float(gates.value(key)) for name, key in GATE_KEYS.items()},
    }
    write_json(root, f"{folder}/{SCORE_PLAN_NAME}", plan)
    anchor_input = JobInput(rel=anchor.audio, sha256=anchor.audio_sha256)
    job_args = ("--run", args.run, "--checkpoint", checkpoint, "--device", DEVICE)
    render = jobs.JobSpec(
        job_id=_job_id(root),
        family=FAMILY,
        image=image_tag(FAMILY),
        module=SCORE_MODULE,
        args=(RENDER, *job_args),
        inputs=(weights_input, valid_input, anchor_input),
        outputs=folder,
        gpu=True,
        network=False,
        vram_gb=RENDER_VRAM_GB,
    )
    renders = _run_job(root, render, RENDERS_NAME)
    rendered = renders.get("lines")
    if not isinstance(rendered, list) or len(rendered) != len(lines):
        raise JobError(f"job {render.job_id}: rendered another number of lines than planned")
    scoring = jobs.JobSpec(
        job_id=_job_id(root),
        family=SCORE_FAMILY,
        image=image_tag(SCORE_FAMILY),
        module=SCORE_MODULE,
        args=(SCORE, *job_args),
        inputs=(valid_input, anchor_input),
        outputs=folder,
        gpu=True,
        network=False,
    )
    doc = _run_job(root, scoring, SCORE_NAME)
    write_json(root, f"{run_rel(args.run)}/{SCORE_NAME}", doc)
    for line in score_lines(args.run, checkpoint, doc):
        print(line)
    if doc.get("pass") is not True:
        names = ", ".join(_failed(doc)) or "the gate"
        raise GateError(f"student score {args.run} {checkpoint}: failed {names}")
    return 0


def score_lines(run_id: str, checkpoint: str, doc: Mapping[str, object]) -> list[str]:
    """Return what `student score` prints: line counts, each measure with its gate, the verdict."""
    counts, gates = _mapping(doc.get("lines")), _mapping(doc.get("gates"))
    lines = [
        f"student score {run_id} {checkpoint}: {_num(counts.get(VALID))} validation and "
        f"{_num(counts.get(BENCH))} bench lines",
        f"  cer {_num(doc.get(CER))}, highest line (at most {_num(gates.get('cer_max'))}); "
        f"mean {_num(doc.get('cer_mean'))}",
        f"  sim ratio {_num(doc.get(SIM_RATIO))} "
        f"(at least {_num(gates.get('sim_median_ratio_min'))})",
        f"  f0 sd ratio {_num(doc.get(F0_SD_RATIO))} "
        f"(within {_num(gates.get('f0_std_tolerance'))} of 1)",
    ]
    names = _failed(doc)
    lines.append("  pass" if doc.get("pass") is True else f"  fail: {', '.join(names) or '?'}")
    return lines


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _failed(doc: Mapping[str, object]) -> list[str]:
    failed = doc.get("failed")
    known = (CER, SIM_RATIO, F0_SD_RATIO)
    return [n for n in failed if n in known] if isinstance(failed, list) else []


# `student export`.


def run_export(args: argparse.Namespace) -> int:
    """Check a checkpoint's layout against the stock weights and write it as an identity."""
    root = root_mod.open_root(args.root)
    _check_id(args.run, "run id")
    identity_mod.check_voice_id(args.voice_id)
    weights = root.path / checkpoint_weights(root, args.run, args.checkpoint)
    train = read_train(root, args.run)
    engine, repository, revision = runtime_pin()
    stock_entry = jobs.load_families()[FAMILY]["models"][STOCK_WEIGHTS[0]]  # type: ignore[index]
    if (repository, revision) != (stock_entry["repository"], stock_entry["revision"]):  # type: ignore[index]
        raise InputError(
            "refused: the realtime Pocket adapter pins another revision than train-pocket fetches"
        )
    found = tensor_layout(weights, f"{args.checkpoint}'s weights")
    stock = tensor_layout(hub_file(root, STOCK_WEIGHTS), "the stock weights")
    differences = layout_differences(found, stock)
    if any(differences.values()):
        detail = ", ".join(f"{count} {kind}" for kind, count in differences.items())
        raise InputError(
            f"refused: {args.checkpoint}'s tensors are not the stock student's ({detail})"
        )
    tokenizer_sha = str(train["tokenizer_sha256"])
    stock_tokenizer = hub_file(root, RUNTIME_TOKENIZER)
    tokenizer = (
        None
        if sha256_file(stock_tokenizer) == tokenizer_sha
        else training_tokenizer(root, tokenizer_sha)[1]
    )
    anchor = teacher_cmd.read_anchor(root, args.anchor)
    clip = identity_mod.ExportClip(
        state=NEUTRAL_STATE,
        audio=root.path / anchor.audio,
        transcript=root.path / anchor.transcript,
        transcript_sha256=anchor.transcript_sha256,
    )
    model = identity_mod.ExportModel(
        engine=engine,
        base=f"{repository}@{revision}",
        weights=weights,
        tokenizer=tokenizer,
        tokenizer_sha256=tokenizer_sha,
    )
    identity = identity_mod.export(
        root, args.voice_id, [clip], status=args.status, model=model, rate=None
    )
    for line in export_lines(args.run, args.checkpoint, identity, len(found)):
        print(line)
    return 0


def export_lines(
    run_id: str, checkpoint: str, identity: identity_mod.Identity, tensors: int
) -> list[str]:
    """Return what `student export` prints: the identity, its status and the weights' pins."""
    model = identity.model
    if model is None:
        raise InputError(f"refused: identity {identity.voice_id} was written without its model")
    return [
        f"student export {run_id} {checkpoint}: wrote "
        f"{identity_mod.voice_rel(identity.voice_id)}, schema {identity_mod.MODEL_SCHEMA}, "
        f"status {identity.status}",
        f"  weights sha256 {model.sha256[:12]}, {tensors} tensors as the stock student's",
        f"  tokenizer sha256 {model.tokenizer_sha256[:12]}, "
        f"{'stock' if model.tokenizer is None else 'copied into the identity'}",
    ]
