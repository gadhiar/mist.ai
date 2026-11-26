"""The Pocket TTS student's fit probe and training, run as `train-pocket` jobs.

Runs as `/opt/venv-train-pocket/bin/python -I -u -m scripts.voice_data.stages.student_train
--data /data probe --run R --corpus C --batches 2 4 8`, or with `train --run R --corpus C
--batch B --ckpt-freq K --steps-finetune N --steps-distill M [--stage S] [--deadline-utc T]`.
The corpus splits and the run folders of their clips are mounted read-only, `students/<R>/`
writable, and the weight cache at `/cache`. On a rented host (H6) the same module runs
standalone with the root at `--data`, the cache under `HF_HOME`, and `--upstream` naming the
pinned clone when it is not at `/opt/mist-voice/upstream/pocket-tts`.

The trainer is upstream's own, `training/train.py` of the ee2ebc89 clone, run once per stretch
in a child process (`trainer` mode of this module). The child puts the clone first on
`sys.path`, because `-I` keeps it off, and checks that `pocket_tts` resolves inside it, so
`training` and the `pocket_tts` it imports come from one commit and not from the installed 4.0.1
wheel. It then runs `training.train` as `__main__` with the config as its one argument, from the
clone's root, and writes what it measured to a metrics file: the free VRAM before the run, the
peak `torch.cuda.max_memory_reserved`, the optimizer steps and their rate (a global optimizer
post-step hook; the first `WARMUP_STEPS` are left out), the loss of each step (the scalar
`backward` received, summed over the step's micro-batches, averaged per `LOSS_WINDOW` steps) and
the longest `save_checkpoint` call, timed on its own clock (`save_clock`). Upstream ends a run,
at its last step and on a stop, with `training.distributed.exit_training`, whose `os._exit(0)`
skips every `finally`, so the child wraps that function to write the metrics before the exit.

Everything the trainer reads is local, because jobs run with `HF_HUB_OFFLINE=1` and the configs'
`hf://` URIs name a revision (`49252265`) the cache does not hold:

- The corpus splits are copied to `students/<R>/data/`, each row's `path` moved from `/data` to
  `--data`, because upstream writes `<manifest>.idx` and the precomputed Mimi latents beside the
  manifest. The copy is kept across jobs, so the latents are computed once per run.
- The two model configs are copied from the clone with `weights_path` and `tokenizer_path`
  pointing at the cached files: the 24-layer teacher (`english_2026-04_24l.yaml`, its weights
  and tokenizer from the `kyutai/pocket-tts` snapshot) and the 6-layer student (`english.yaml`,
  its tokenizer the runtime's, from `kyutai/pocket-tts-without-voice-cloning`). The student's
  config is the runtime's, so the student is trained with the runtime's tokenizer. Upstream
  distils with one tokenizer for teacher and student, so training refuses to start when the two
  tokenizer files differ.
- The trainer configs are upstream's `finetune.yaml` and `depth_distill.yaml` with the paths,
  batch, accumulation, steps and checkpoint interval set here, `compile: false` and no sample
  sentences. Compilation costs about 9 minutes per process start, which every resumed window
  and every checkpoint restart would pay again, and the probe must measure the memory and rate
  of what training runs. Sample WAVs are off: nothing here listens to them.

`probe` trains `PROBE_STEPS` optimizer steps at each batch in turn, with accumulation to the
effective batch upstream requires, stopping at the first that does not fit (its peak above the
free VRAM less `HEADROOM_MIB`, or out of memory), and writes `probe.json`, with the longest
checkpoint save of any trial as `save_s`, from which a stop's grace is sized. `train` runs the
fine-tune from the 24-layer base and then the distillation, the student warm-started from the
released 6-layer student (`student_init_from`) and taught by the fine-tune's newest checkpoint.
Upstream resumes from its newest checkpoint by itself, saves at `ckpt_freq` and at `max_steps`,
and, when a `STOP` file appears in its run_dir, saves after the current step and exits. This
stage writes that file at the deadline, on SIGTERM, and when `WATCHDOG` passes without a new
checkpoint (it then starts the trainer again, which resumes), so a checkpoint lands at least
every `CHECKPOINT_MAX_MINUTES` even when the probe's rate overstates the run's. `train.json` holds
the status (`running`, `paused` or `done`), the steps, the losses, the pocket-tts versions and
the sha256 of every config and cached file; the log stays in `runs/<job>/log.txt`.

In the distill child, `training.checkpointing.save_checkpoint` is wrapped: after each save, the
`model.safetensors` that upstream's `export_pocket_safetensors` refreshes in the run_dir (the
EMA student's `flow_lm.*` and `mimi.*` tensors, no optimizer state) is checked for that layout
and written to `checkpoints/weights/<checkpoint>/model.safetensors`. Each stage keeps its last
`--keep-last` checkpoints (upstream's `num_ckpt_keep`) and validates at every checkpoint
interval over the validation split's whole batches (`Trainer.retention`), logging the loss to
`progress.jsonl`. A distill checkpoint upstream prunes loses its weights unless its validation
loss is the lowest so far (`best_checkpoint`), which `train.json` records as `best`. Nothing
this stage prints carries a transcript, a line's text or a clip's path.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import importlib.metadata
import json
import math
import os
import re
import runpy
import shutil
import signal
import subprocess
import sys
import time
import tomllib
import traceback
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import monotonic as _monotonic
from types import FrameType
from typing import Protocol

from .. import deadline as deadline_mod
from ..commands import corpus as corpus_contract
from ..commands import student as contract
from ..errors import JobError
from . import fetch

SCHEMA = 1
UPSTREAM = fetch.UPSTREAM / "pocket-tts"
PACKAGE = "pocket-tts"
TRAIN_MODULE = "training.train"
CHECKPOINTING_MODULE = "training.checkpointing"
# Home of `exit_training`, upstream's hard exit (`os._exit(0)`) at the end of a run and on a stop.
DISTRIBUTED_MODULE = "training.distributed"
TEACHER_MODEL_CONFIG = "pocket_tts/config/english_2026-04_24l.yaml"
STUDENT_MODEL_CONFIG = "pocket_tts/config/english.yaml"
FINETUNE_CONFIG = "training/configs/finetune.yaml"
DISTILL_CONFIG = "training/configs/depth_distill.yaml"
# Cached files: the `families.json` train-pocket model and the path in its snapshot.
BASE_WEIGHTS = ("pocket-tts", "languages/english_2026-04_24l/model.safetensors")
BASE_TOKENIZER = ("pocket-tts", "languages/english_2026-04_24l/tokenizer.json")
STUDENT_WEIGHTS = ("pocket-tts", "languages/english/model.safetensors")
STUDENT_TOKENIZER = ("pocket-tts-runtime-tokenizer", "languages/english/tokenizer.json")
POLL_S = 5.0
# With no checkpoint for this long, the trainer is stopped (it saves) and started again. It is
# under `CHECKPOINT_MAX_MINUTES` by a step, the save and a restart's model load.
WATCHDOG = timedelta(minutes=contract.CHECKPOINT_MAX_MINUTES - 5)
WARMUP_STEPS = 5
LOSS_WINDOW = 50
# The probe validates never: a validation pass would be timed as training.
PROBE_VALID_FREQ = 10**9
CHILD_OOM = 3
STOP_NAME = "STOP"
SOURCE_NAME = "source.json"
# Upstream's log in its run_dir, one JSON object a line, written by `ProgressLog.log(event, step,
# metrics)` (training/train_utils.py at ee2ebc89) as `{"time", "type": event, "step", "metrics"}`.
# train.py logs `progress.log("valid", step + 1, valid_metrics)` after each validation, and every
# forward branch's metrics carry `loss`: `{"type": "valid", "step": S, "metrics": {"loss": L}}`.
PROGRESS_NAME = "progress.jsonl"
VALID_MARK = ("type", "valid")
CHECKPOINT = "checkpoint"
MIB = 1 << 20
_CHECKPOINT_PT = re.compile(r"checkpoint_([0-9]{8})\.pt")
_SHA = re.compile(r"[0-9a-f]{40}")


class StageError(RuntimeError):
    """The probe or the training cannot proceed; the message names ids and counts only."""


def utc_now() -> datetime:
    """Return the current time in UTC; tests replace this function."""
    return datetime.now(UTC)


def sleep(seconds: float) -> None:
    """Wait between polls of the trainer; tests replace this function."""
    time.sleep(seconds)


def save_clock() -> float:
    """Return a monotonic reading in seconds, to time a checkpoint save; tests replace this."""
    return _monotonic()


class Child(Protocol):
    """A running trainer: `subprocess.Popen` or a test's fake."""

    def poll(self) -> int | None:
        """Return the exit code once the trainer has exited, else `None`."""
        ...


def spawn(
    data: Path, upstream: Path, config: Path, metrics: Path, export_dir: Path | None
) -> Child:
    """Start the trainer child: this module in `trainer` mode; tests replace this function.

    Its output goes to this job's own, which the executor writes to the job log.
    """
    argv = [
        sys.executable, "-I", "-u", "-m", "scripts.voice_data.stages.student_train",
        "--data", str(data), "--upstream", str(upstream),
        "trainer", "--config", str(config), "--metrics", str(metrics),
    ]  # fmt: skip
    if export_dir is not None:
        argv += ["--export-dir", str(export_dir)]
    return subprocess.Popen(argv)


class Stop:
    """When to have the trainer save and stop: the deadline has come, or SIGTERM arrived."""

    def __init__(self, deadline: datetime | None, clock: Callable[[], datetime]) -> None:
        self._deadline = deadline
        self._clock = clock
        self.signalled = False

    def on_signal(self, signum: int, frame: FrameType | None) -> None:
        """The SIGTERM handler: record it; the trainer saves after its current step."""
        self.signalled = True

    def reason(self) -> str | None:
        """Return `signal` or `deadline` when training should pause now, else `None`."""
        if self.signalled:
            return contract.SIGNAL
        if self._deadline is not None and self._clock() >= self._deadline:
            return contract.DEADLINE
        return None


# Files.


def _sha256(path: Path) -> str:
    return fetch.sha256_file(path)


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_bytes(data)
    os.replace(part, path)


def _write_json(path: Path, payload: object) -> None:
    _write_bytes(path, (json.dumps(payload, indent=2) + "\n").encode("utf-8"))


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _read_yaml(path: Path) -> dict[str, object]:
    import yaml

    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise StageError(f"{path.name} is missing or not YAML") from exc
    if not isinstance(doc, dict):
        raise StageError(f"{path.name} is not a mapping")
    return doc


def _write_yaml(path: Path, doc: Mapping[str, object]) -> None:
    import yaml

    _write_bytes(path, yaml.safe_dump(dict(doc), sort_keys=False).encode("utf-8"))


def newest_step(run_dir: Path) -> int:
    """Return the step of the trainer's newest `checkpoint_<step>.pt` in `run_dir`, or 0."""
    if not run_dir.is_dir():
        return 0
    steps = [int(m[1]) for p in run_dir.iterdir() if (m := _CHECKPOINT_PT.fullmatch(p.name))]
    return max(steps, default=0)


# The corpus copy, the cached files and the configs.


def prepare_corpus(data: Path, run_id: str, corpus: str) -> tuple[Path, Path]:
    """Copy the corpus splits into the run's writable `data/`; return the train and valid copies.

    Each row's `path` moves from the job mount (`/data`) to `data`, so the copy also serves a
    standalone run whose root is elsewhere. A copy already made from the same splits is kept,
    with the index and latents upstream wrote beside it.

    Raises:
        StageError: A split is missing, the run's copy came from other splits, or a row's clip
            is missing.
    """
    source_dir = data / corpus_contract.corpus_rel(corpus)
    dest_dir = data / contract.run_rel(run_id) / contract.DATA_DIR
    names = (corpus_contract.TRAIN_NAME, corpus_contract.VALID_NAME)
    shas: dict[str, str] = {}
    for name in names:
        if not (source_dir / name).is_file():
            raise StageError(f"corpus {corpus} has no {name}")
        shas[name] = _sha256(source_dir / name)
    record = {"corpus": corpus, "sha256": shas}
    old = _read_json(dest_dir / SOURCE_NAME)
    if old is not None and old != record:
        raise StageError(f"run {run_id}'s corpus copy is of other splits; train a new run")
    if old == record and all((dest_dir / name).is_file() for name in names):
        return dest_dir / names[0], dest_dir / names[1]
    prefix = f"{contract.JOB_DATA}/"
    missing = 0
    for name in names:
        rows = []
        for line in (source_dir / name).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            path = str(row.get("path", ""))
            if not path.startswith(prefix):
                raise StageError(f"corpus {corpus} has a row with no clip under /data")
            local = data / path.removeprefix(prefix)
            missing += not local.is_file()
            rows.append(json.dumps({**row, "path": str(local)}) + "\n")
        _write_bytes(dest_dir / name, "".join(rows).encode("utf-8"))
    if missing:
        raise StageError(f"{missing} corpus clips are missing")
    _write_json(dest_dir / SOURCE_NAME, record)
    return dest_dir / names[0], dest_dir / names[1]


def cached_file(model: tuple[str, str]) -> Path:
    """Return a cached file of a train-pocket model, offline.

    Raises:
        StageError: The snapshot or the file is not in the cache.
    """
    try:
        return fetch.model_file(contract.FAMILY, model[0], model[1])
    except JobError as exc:
        raise StageError(f"{model[0]} {model[1]} is not in the cache; run fetch") from exc


def _replace_key(node: object, key: str, value: str) -> int:
    count = 0
    if isinstance(node, dict):
        for name in list(node):
            if name == key:
                node[name] = value
                count += 1
            else:
                count += _replace_key(node[name], key, value)
    elif isinstance(node, list):
        for item in node:
            count += _replace_key(item, key, value)
    return count


def local_model_config(source: Path, dest: Path, weights: Path, tokenizer: Path) -> Path:
    """Write a copy of a model config whose `weights_path` and `tokenizer_path` are local files.

    Each key is replaced wherever it sits in the config; other `hf://` values are left as they
    are (the student's `weights_path_without_voice_cloning` is one that training never loads).

    Raises:
        StageError: The config is missing, or does not hold each key exactly once.
    """
    doc = _read_yaml(source)
    for key, value in (("weights_path", weights), ("tokenizer_path", tokenizer)):
        if _replace_key(doc, key, str(value)) != 1:
            raise StageError(f"{source.name} does not name {key} exactly once")
    _write_yaml(dest, doc)
    return dest


def trainer_config(
    base: Mapping[str, object],
    *,
    model_config: Path,
    train_jsonl: Path,
    valid_jsonl: Path,
    run_dir: Path,
    batch: int,
    max_steps: int,
    ckpt_freq: int,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return upstream's trainer config `base` with this run's paths, batch and steps set.

    The accumulation brings `batch` to the config's own effective batch; the checkpoint interval
    is the smaller of `ckpt_freq` and the config's, and validation runs at the same interval, so
    upstream logs a validation loss at the step of each interval checkpoint.
    """
    config = copy.deepcopy(dict(base))
    effective = int(config.get("batch_size", contract.EFFECTIVE_BATCH)) * int(
        config.get("grad_accum_steps", 1)  # type: ignore[call-overload]
    )
    data_section = dict(config.get("data") or {})  # type: ignore[call-overload]
    data_section.update(train_jsonl=str(train_jsonl), valid_jsonl=str(valid_jsonl))
    base_freq = config.get("ckpt_freq")
    every = min(ckpt_freq, int(base_freq)) if isinstance(base_freq, int) else ckpt_freq  # type: ignore[arg-type]
    config.update(
        model_config=str(model_config),
        data=data_section,
        run_dir=str(run_dir),
        batch_size=batch,
        grad_accum_steps=contract.accum_for(batch, effective),
        max_steps=max_steps,
        ckpt_freq=max(1, every),
        valid_freq=max(1, every),
        compile=False,
        sample_sentences=[],
    )
    config.update(extra or {})
    return config


class Workspace:
    """A run's folders, its local corpus and configs, and the sha256 of each file they name."""

    def __init__(self, data: Path, upstream: Path, run_id: str, corpus: str) -> None:
        self.data = data
        self.upstream = upstream
        self.run_id = run_id
        self.corpus = corpus
        self.folder = data / contract.run_rel(run_id)
        self.configs = self.folder / contract.CONFIGS_DIR
        self.files: dict[str, str] = {}

    def stage_dir(self, stage: str) -> Path:
        """Return the trainer's run_dir for a stage."""
        return self.data / contract.stage_rel(self.run_id, stage)

    def weights_dir(self) -> Path:
        """Return the folder of the distill checkpoints' runtime-layout weights."""
        return self.folder / contract.CHECKPOINTS_DIR / contract.WEIGHTS_DIR

    def prepare(self, *, student: bool) -> None:
        """Copy the corpus and write the teacher's (and with `student`, the student's) config."""
        self.train_jsonl, self.valid_jsonl = prepare_corpus(self.data, self.run_id, self.corpus)
        lines = self.valid_jsonl.read_text(encoding="utf-8").splitlines()
        self.valid_rows = sum(1 for line in lines if line.strip())
        base, base_tokenizer = cached_file(BASE_WEIGHTS), cached_file(BASE_TOKENIZER)
        self.teacher_config = local_model_config(
            self.upstream / TEACHER_MODEL_CONFIG, self.configs / "teacher-model.yaml",
            base, base_tokenizer,
        )  # fmt: skip
        named = {
            "teacher_model_config": self.teacher_config,
            "base_weights": base,
            "base_tokenizer": base_tokenizer,
        }
        if student:
            self.student_init, tokenizer = cached_file(STUDENT_WEIGHTS), cached_file(
                STUDENT_TOKENIZER
            )
            self.student_config = local_model_config(
                self.upstream / STUDENT_MODEL_CONFIG, self.configs / "student-model.yaml",
                self.student_init, tokenizer,
            )  # fmt: skip
            named.update(
                student_model_config=self.student_config,
                student_init_weights=self.student_init,
                student_tokenizer=tokenizer,
            )
        self.files = {name: _sha256(path) for name, path in named.items()}

    def config(self, name: str, doc: Mapping[str, object]) -> Path:
        """Write a trainer config under `configs/` and record its sha256."""
        path = self.configs / f"{name}.yaml"
        _write_yaml(path, doc)
        self.files[f"{name}_config"] = _sha256(path)
        return path

    def rel(self, path: Path) -> str:
        """Return a path under `--data` as root-relative posix."""
        return path.relative_to(self.data).as_posix()


def installed_version() -> str | None:
    """Return the installed pocket-tts wheel's version; tests replace this function."""
    try:
        return importlib.metadata.version(PACKAGE)
    except importlib.metadata.PackageNotFoundError:
        return None


def versions(upstream: Path) -> dict[str, str | None]:
    """Return the installed pocket-tts version and the clone's version and commit."""
    installed = installed_version()
    clone: str | None = None
    try:
        project = tomllib.loads((upstream / "pyproject.toml").read_text(encoding="utf-8"))
        value = project.get("project", {}).get("version")
        clone = value if isinstance(value, str) else None
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        clone = None
    try:
        head = (upstream / ".git" / "HEAD").read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        head = ""
    return {
        "pocket_tts_installed": installed,
        "pocket_tts_clone": clone,
        "clone_commit": head if _SHA.fullmatch(head) else None,
    }


# Watching a trainer.


def read_metrics(path: Path) -> dict[str, object]:
    """Return a trainer's metrics file, or an empty mapping when it wrote none (it was killed)."""
    doc = _read_json(path)
    return doc if isinstance(doc, dict) else {}


def watch(
    child: Child,
    run_dir: Path,
    stop: Stop,
    *,
    checkpoint_max: timedelta | None,
    clock: Callable[[], datetime],
) -> tuple[int, str | None]:
    """Wait for the trainer; write `STOP` once a stop reason arises; return its exit code and why.

    With `checkpoint_max`, a stretch that long without a new checkpoint is a reason too
    (`checkpoint`): the trainer saves, exits, and the caller starts it again.
    """
    seen, since = newest_step(run_dir), clock()
    stopped: str | None = None
    while (code := child.poll()) is None:
        step = newest_step(run_dir)
        if step != seen:
            seen, since = step, clock()
        if stopped is None:
            reason = stop.reason()
            if reason is None and checkpoint_max is not None and clock() - since >= checkpoint_max:
                reason = CHECKPOINT
            if reason is not None:
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / STOP_NAME).touch()
                stopped = reason
        sleep(POLL_S)
    (run_dir / STOP_NAME).unlink(missing_ok=True)
    return code, stopped


# `probe`.


def _mib(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return round(value / MIB, 1)


def _rate(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not value > 0:
        return None
    return float(value)


def _seconds(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not value >= 0:
        return None
    return float(value) if math.isfinite(value) else None


def run_probe(
    data: Path, upstream: Path, run_id: str, corpus: str, batches: Sequence[int], stop: Stop
) -> tuple[list[str], dict[str, object]]:
    """Train `PROBE_STEPS` steps at each batch until one does not fit; write `probe.json`."""
    space = Workspace(data, upstream, run_id, corpus)
    space.prepare(student=False)
    base = _read_yaml(upstream / FINETUNE_CONFIG)
    trials: list[dict[str, object]] = []
    for batch in batches:
        run_dir = space.folder / "probe" / f"batch-{batch}"
        shutil.rmtree(run_dir, ignore_errors=True)
        doc = trainer_config(
            base, model_config=space.teacher_config, train_jsonl=space.train_jsonl,
            valid_jsonl=space.valid_jsonl, run_dir=run_dir, batch=batch,
            max_steps=contract.PROBE_STEPS, ckpt_freq=contract.PROBE_STEPS,
            extra={"num_ckpt_keep": 1, "valid_freq": PROBE_VALID_FREQ},
        )  # fmt: skip
        config = space.config(f"probe-batch-{batch}", doc)
        metrics_path = space.folder / "probe" / f"batch-{batch}-metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.unlink(missing_ok=True)
        child = spawn(data, upstream, config, metrics_path, None)
        code, stopped = watch(child, run_dir, stop, checkpoint_max=None, clock=utc_now)
        if stopped is not None:
            raise StageError(f"the probe was stopped by a {stopped} before it finished")
        metrics = read_metrics(metrics_path)
        oom = code == CHILD_OOM and bool(metrics.get("oom"))
        if code != 0 and not oom:
            raise StageError(f"the trainer exited {code} at batch {batch}")
        peak, free = _mib(metrics.get("peak_reserved_bytes")), _mib(metrics.get("free_bytes"))
        rate = _rate(metrics.get("steps_per_s"))
        if free is None:
            raise StageError(f"the trainer measured no free VRAM at batch {batch}")
        fits = (
            not oom and peak is not None and rate is not None
            and peak <= free - contract.HEADROOM_MIB
        )  # fmt: skip
        trials.append(
            {"batch": batch, "accum": int(doc["grad_accum_steps"]), "peak_vram_mib": peak,  # type: ignore[call-overload]
             "free_vram_mib": free, "steps_per_s": rate,
             "save_s": _seconds(metrics.get("save_s")), "oom": oom, "fits": fits}
        )  # fmt: skip
        shutil.rmtree(run_dir, ignore_errors=True)
        print(f"student_train: probe batch {batch}: {'fits' if fits else 'does not fit'}")
        if not fits:
            break
    fitting = [t for t in trials if t["fits"]]
    best = fitting[-1] if fitting else None
    saves = [float(t["save_s"]) for t in trials if t["save_s"] is not None]  # type: ignore[arg-type]
    total = contract.STEPS_FINETUNE + contract.STEPS_DISTILL
    probe: dict[str, object] = {
        "schema": SCHEMA,
        "run": run_id,
        "corpus": corpus,
        "batch": best["batch"] if best else None,
        "accum": best["accum"] if best else None,
        "peak_vram_mib": best["peak_vram_mib"] if best else None,
        "free_vram_mib": (best or trials[0])["free_vram_mib"],
        "steps_per_s": best["steps_per_s"] if best else None,
        # The longest save of any trial: the model saved is the same at every batch.
        "save_s": round(max(saves), 1) if saves else None,
        "projected_hours": (
            contract.projected_hours(float(best["steps_per_s"]), total) if best else None  # type: ignore[arg-type]
        ),
        "steps": {
            contract.FINETUNE: contract.STEPS_FINETUNE,
            contract.DISTILL: contract.STEPS_DISTILL,
        },
        "trials": trials,
    }
    _write_json(space.folder / contract.PROBE_NAME, probe)
    written = [
        contract.PROBE_NAME,
        *(f"{contract.CONFIGS_DIR}/{p.name}" for p in sorted(space.configs.iterdir())),
    ]
    return [f"{contract.run_rel(run_id)}/{rel}" for rel in written], probe


# `train`.


class TrainRecord:
    """`train.json`: the status, steps, losses, versions and file sha256s, rewritten as it goes."""

    def __init__(self, path: Path, run_id: str, corpus: str) -> None:
        self.path = path
        old = _read_json(path)
        doc = old if isinstance(old, dict) and old.get("schema") == SCHEMA else {}
        self.doc: dict[str, object] = {
            "schema": SCHEMA,
            "run": run_id,
            "corpus": corpus,
            "status": contract.RUNNING,
            "stopped_by": None,
            "stage": doc.get("stage"),
            "steps": dict(doc.get("steps") or {}),  # type: ignore[call-overload]
            "losses": {k: list(v) for k, v in dict(doc.get("losses") or {}).items()},  # type: ignore[call-overload]
            "best": doc.get("best"),
            "versions": doc.get("versions"),
            "tokenizer_sha256": doc.get("tokenizer_sha256"),
            "files": doc.get("files"),
        }

    def save(self) -> None:
        """Write the record atomically."""
        _write_json(self.path, self.doc)

    def steps(self, stage: str, done: int, maximum: int) -> None:
        """Record a stage's newest checkpoint step and its last step."""
        self.doc["steps"][stage] = {"done": done, "max": maximum}  # type: ignore[index]

    def losses(self, stage: str, start: int, metrics: Mapping[str, object]) -> None:
        """Append a trainer's loss windows, numbered from the step it resumed at."""
        windows = metrics.get("losses")
        trail = self.doc["losses"].setdefault(stage, [])  # type: ignore[union-attr]
        for entry in windows if isinstance(windows, list) else []:
            if isinstance(entry, list) and len(entry) == 2 and isinstance(entry[0], int):
                trail.append([start + entry[0], entry[1]])


class Trainer:
    """One `train` job: each stage's trainer, restarted until it reaches its last step or stops."""

    def __init__(
        self,
        space: Workspace,
        record: TrainRecord,
        stop: Stop,
        batch: int,
        ckpt_freq: int | None,
        keep_last: int = contract.KEEP_LAST,
    ) -> None:
        self.space = space
        self.record = record
        self.stop = stop
        self.batch = batch
        self.ckpt_freq = ckpt_freq
        self.keep_last = keep_last

    def retention(self) -> dict[str, object]:
        """Return the checkpoints a stage keeps and the validation batches each pass runs.

        Upstream's `validate` reads the split in file order in whole batches, dropping a tail
        shorter than one, and starts again from the top after the last; `rows // batch` batches
        validate each row of the whole batches once. The corpus writes its held-out seed rows
        first, so a dropped tail is teacher rows.
        """
        return {
            "num_ckpt_keep": self.keep_last,
            "num_valid_batches": max(1, self.space.valid_rows // self.batch),
        }

    def finetune_config(self, max_steps: int) -> Path:
        """Write the fine-tune's trainer config: the 24-layer base, upstream's `finetune.yaml`."""
        base = _read_yaml(self.space.upstream / FINETUNE_CONFIG)
        doc = trainer_config(
            base, model_config=self.space.teacher_config, train_jsonl=self.space.train_jsonl,
            valid_jsonl=self.space.valid_jsonl, run_dir=self.space.stage_dir(contract.FINETUNE),
            batch=self.batch, max_steps=max_steps,
            ckpt_freq=self.ckpt_freq or int(base.get("ckpt_freq", 1)),  # type: ignore[call-overload]
            extra=self.retention(),
        )  # fmt: skip
        return self.space.config(contract.FINETUNE, doc)

    def distill_config(self, max_steps: int, teacher_steps: int) -> Path:
        """Write the distillation's trainer config, taught by the fine-tune's newest checkpoint.

        The teacher's config is the 24-layer one it was fine-tuned with; the student's is the
        runtime's 6-layer config, warm-started from the released student.
        """
        base = _read_yaml(self.space.upstream / DISTILL_CONFIG)
        finetune_dir = self.space.stage_dir(contract.FINETUNE)
        teacher = finetune_dir / f"{contract.checkpoint_name(newest_step(finetune_dir))}.pt"
        if newest_step(finetune_dir) < teacher_steps or not teacher.is_file():
            raise StageError(
                f"the fine-tune has not reached step {teacher_steps}; finish it before distilling"
            )
        doc = trainer_config(
            base, model_config=self.space.student_config, train_jsonl=self.space.train_jsonl,
            valid_jsonl=self.space.valid_jsonl, run_dir=self.space.stage_dir(contract.DISTILL),
            batch=self.batch, max_steps=max_steps,
            ckpt_freq=self.ckpt_freq or int(base.get("ckpt_freq", 1)),  # type: ignore[call-overload]
            extra={
                **self.retention(),
                "student_init_from": str(self.space.student_init),
                "distill_teacher_config": str(self.space.teacher_config),
                "distill_teacher_weights": str(teacher),
            },
        )  # fmt: skip
        return self.space.config(contract.DISTILL, doc)

    def run_stage(self, stage: str, config: Path, max_steps: int) -> str | None:
        """Run a stage's trainer until its last step (return `None`) or a pause (its reason)."""
        run_dir = self.space.stage_dir(stage)
        export_dir = self.space.weights_dir() if stage == contract.DISTILL else None
        self.record.doc["stage"] = stage
        while True:
            start = newest_step(run_dir)
            self.record.steps(stage, start, max_steps)
            self.record.save()
            if start >= max_steps:
                return None
            reason = self.stop.reason()
            if reason is not None:
                return reason
            metrics_path = self.space.folder / f"{stage}-metrics.json"
            metrics_path.unlink(missing_ok=True)
            child = spawn(self.space.data, self.space.upstream, config, metrics_path, export_dir)
            code, stopped = watch(child, run_dir, self.stop, checkpoint_max=WATCHDOG, clock=utc_now)
            self.record.losses(stage, start, read_metrics(metrics_path))
            done = newest_step(run_dir)
            self.record.steps(stage, done, max_steps)
            self.record.save()
            if code != 0:
                raise StageError(f"the {stage} trainer exited {code}")
            if stopped is None and done < max_steps:
                raise StageError(f"the {stage} trainer stopped before step {max_steps}")
            print(f"student_train: {stage} at step {done} of {max_steps}")

    def run(self, stages: Sequence[str], steps: Mapping[str, int]) -> str | None:
        """Run the stages in order; return the pause reason, or `None` when all are done."""
        for stage in stages:
            if stage == contract.FINETUNE:
                config = self.finetune_config(steps[stage])
            else:
                config = self.distill_config(steps[stage], steps[contract.FINETUNE])
            reason = self.run_stage(stage, config, steps[stage])
            if reason is not None:
                return reason
        return None


def run_train(
    data: Path,
    upstream: Path,
    run_id: str,
    corpus: str,
    batch: int,
    ckpt_freq: int | None,
    steps: Mapping[str, int],
    stage: str | None,
    stop: Stop,
    keep_last: int = contract.KEEP_LAST,
) -> tuple[list[str], dict[str, object]]:
    """Fine-tune, then distill (or the one `stage`); return the files written and `train.json`.

    Raises:
        StageError: `keep_last` is below 1, the two tokenizers differ, a trainer failed, or
            distillation was asked for before the fine-tune finished.
    """
    if keep_last < 1:
        raise StageError("a stage keeps at least 1 checkpoint")
    space = Workspace(data, upstream, run_id, corpus)
    weights_before = set(space.weights_dir().glob(f"*/{contract.WEIGHTS_NAME}"))
    space.prepare(student=True)
    if space.files["base_tokenizer"] != space.files["student_tokenizer"]:
        raise StageError(
            "the 24-layer teacher's tokenizer differs from the student's; distillation needs one"
        )
    record = TrainRecord(space.folder / contract.TRAIN_NAME, run_id, corpus)
    record.doc["versions"] = versions(upstream)
    record.doc["tokenizer_sha256"] = space.files["student_tokenizer"]
    stages = (stage,) if stage else contract.STAGES
    trainer = Trainer(space, record, stop, batch, ckpt_freq, keep_last)
    reason: str | None = None
    try:
        reason = trainer.run(stages, steps)
    finally:
        record.doc["files"] = dict(space.files)
        record.doc["best"] = best_checkpoint(space.stage_dir(contract.DISTILL), space.weights_dir())
        record.save()
    finished = all(newest_step(space.stage_dir(name)) >= steps[name] for name in contract.STAGES)
    record.doc["status"] = (
        contract.PAUSED if reason is not None else contract.DONE if finished else contract.RUNNING
    )
    record.doc["stopped_by"] = reason
    record.save()
    print(f"student_train: {record.doc['status']}")
    new = sorted(set(space.weights_dir().glob(f"*/{contract.WEIGHTS_NAME}")) - weights_before)
    written = [space.folder / contract.TRAIN_NAME, *sorted(space.configs.iterdir()), *new]
    return [space.rel(path) for path in written], record.doc


# `trainer`: the child that runs upstream's trainer.


def first_on_path(folder: Path) -> None:
    """Put `folder` first on `sys.path`, moving it there if it is already further back."""
    entry = str(folder)
    while entry in sys.path:
        sys.path.remove(entry)
    sys.path.insert(0, entry)


def check_clone(upstream: Path) -> None:
    """Import `pocket_tts` with the clone first on `sys.path` and check that it is the clone's.

    Raises:
        StageError: `pocket_tts` resolves outside the clone (the installed wheel).
    """
    first_on_path(upstream)
    module = importlib.import_module("pocket_tts")
    origin = Path(str(getattr(module, "__file__", "") or "")).resolve()
    if not origin.is_relative_to(upstream.resolve()):
        raise StageError("pocket_tts does not resolve to the pinned clone")


class StepMonitor:
    """Times the optimizer's steps and sums the loss `backward` received in each.

    Only the first optimizer that steps is followed, so a trainer with two optimizers (Muon
    and AdamW) counts each step once.
    """

    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self._tracked: int | None = None
        self._pending: list[float] = []
        self.times: list[float] = []
        self.step_losses: list[float | None] = []
        self._restore: list[Callable[[], None]] = []

    def on_backward(self, value: float) -> None:
        """Add one micro-batch's loss to the step under way."""
        self._pending.append(value)

    def on_step(self, optimizer: object) -> None:
        """Close a step of the followed optimizer: its time and its loss."""
        if self._tracked is None:
            self._tracked = id(optimizer)
        if id(optimizer) != self._tracked:
            return
        self.times.append(self._clock())
        total = math.fsum(self._pending) if self._pending else None
        self.step_losses.append(total if total is None or math.isfinite(total) else None)
        self._pending.clear()

    def install(self, torch: object) -> None:
        """Hook the optimizer's post-step and `Tensor.backward`; `remove` undoes both.

        The hook registration lives in the `torch.optim.optimizer` submodule, which `import
        torch` loads but `torch.optim` neither binds as an attribute nor re-exports (torch 2.6
        and 2.8), so it is looked up as a module under the injected torch's name.
        """
        name = torch.__name__  # type: ignore[attr-defined]
        hooks = importlib.import_module(f"{name}.optim.optimizer")
        handle = hooks.register_optimizer_step_post_hook(lambda opt, a, k: self.on_step(opt))
        self._restore.append(handle.remove)
        tensor = torch.Tensor  # type: ignore[attr-defined]
        original = tensor.backward

        def backward(this: object, *args: object, **kwargs: object) -> object:
            if this.numel() == 1:  # type: ignore[attr-defined]
                self.on_backward(float(this.detach().float().item()))  # type: ignore[attr-defined]
            return original(this, *args, **kwargs)

        tensor.backward = backward

        def put_back() -> None:
            tensor.backward = original

        self._restore.append(put_back)

    def remove(self) -> None:
        """Undo `install`."""
        while self._restore:
            self._restore.pop()()

    def steps_per_s(self) -> float | None:
        """Return the optimizer steps per second after the first `WARMUP_STEPS`."""
        timed = self.times[WARMUP_STEPS:]
        if len(timed) < 2 or timed[-1] <= timed[0]:
            return None
        return (len(timed) - 1) / (timed[-1] - timed[0])

    def windows(self) -> list[list[object]]:
        """Return `[last step, mean loss]` per `LOSS_WINDOW` steps, the last window partial."""
        out: list[list[object]] = []
        for start in range(0, len(self.step_losses), LOSS_WINDOW):
            chunk = [v for v in self.step_losses[start : start + LOSS_WINDOW] if v is not None]
            end = min(start + LOSS_WINDOW, len(self.step_losses))
            out.append([end, round(math.fsum(chunk) / len(chunk), 6) if chunk else None])
        return out


def runtime_weights(state: Mapping[str, object]) -> dict[str, object]:
    """Return the tensors of an exported state dict, checked for the runtime's key layout.

    The runtime loads `flow_lm.*` and `mimi.*` tensors (`export_pocket_safetensors`); anything
    else, optimizer state included, means the file is not a student export.

    Raises:
        StageError: A key is outside `flow_lm.` and `mimi.`, or either prefix has no tensor.
    """
    stray = [k for k in state if not k.startswith(("flow_lm.", "mimi."))]
    if stray:
        raise StageError(f"the export holds {len(stray)} tensors outside flow_lm and mimi")
    for prefix in ("flow_lm.", "mimi."):
        if not any(k.startswith(prefix) for k in state):
            raise StageError(f"the export holds no {prefix.rstrip('.')} tensor")
    return {key: state[key] for key in sorted(state)}


def export_weights(source: Path, dest: Path) -> None:
    """Write the checked runtime-layout copy of upstream's export at `dest`, atomically."""
    import safetensors.torch

    weights = runtime_weights(safetensors.torch.load_file(str(source)))
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_name(dest.name + ".part")
    safetensors.torch.save_file(weights, str(temp))
    os.replace(temp, dest)


def valid_losses(run_dir: Path) -> dict[int, float]:
    """Return the validation loss of each step upstream logged in `<run_dir>/progress.jsonl`.

    A step logged twice (a stretch was killed after validating and run again) keeps its last
    line; a line that is not a whole JSON object (a torn last line), or has no `metrics` mapping
    holding a finite `loss`, is passed over.
    """
    try:
        text = (run_dir / PROGRESS_NAME).read_text(encoding="utf-8")
    except (FileNotFoundError, NotADirectoryError, UnicodeDecodeError):
        return {}
    key, mark = VALID_MARK
    losses: dict[int, float] = {}
    for line in text.splitlines():
        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(doc, dict) or doc.get(key) != mark:
            continue
        metrics = doc.get("metrics")
        if not isinstance(metrics, dict):
            continue
        step, loss = doc.get("step"), metrics.get("loss")
        if isinstance(step, bool) or not isinstance(step, int):
            continue
        if isinstance(loss, bool) or not isinstance(loss, int | float) or not math.isfinite(loss):
            continue
        losses[step] = float(loss)
    return losses


def best_checkpoint(run_dir: Path, export_dir: Path) -> dict[str, object] | None:
    """Return the checkpoint with weights whose validation loss is lowest, or `None`.

    Ties go to the earlier step. A checkpoint saved by a stop, the deadline or the watchdog has
    no validation line, and a validated step whose save never landed has no weights; neither
    can be the best.
    """
    found = [
        (loss, step)
        for step, loss in valid_losses(run_dir).items()
        if (export_dir / contract.checkpoint_name(step) / contract.WEIGHTS_NAME).is_file()
    ]
    if not found:
        return None
    loss, step = min(found)
    return {"checkpoint": contract.checkpoint_name(step), "step": step, "valid_loss": loss}


def prune_exports(run_dir: Path, export_dir: Path, keep: str | None) -> None:
    """Remove the weights of each checkpoint upstream no longer keeps, except `keep`'s."""
    if not export_dir.is_dir():
        return
    for folder in export_dir.iterdir():
        if (
            re.fullmatch(r"checkpoint_[0-9]{8}", folder.name)
            and folder.name != keep
            and not (run_dir / f"{folder.name}.pt").is_file()
        ):
            shutil.rmtree(folder)


def _stamp(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_ino, stat.st_mtime_ns


def install_export(module: object, export_dir: Path) -> None:
    """Wrap `module.save_checkpoint` so each save also writes its checkpoint's student weights.

    Upstream calls `save_checkpoint(run_dir, step, ...)`; with Mimi it refreshes
    `run_dir/model.safetensors`, which the next save overwrites, so the copy is made at once.
    The weights of the checkpoints upstream keeps (its last `num_ckpt_keep`) remain, and so do
    those of the best validated checkpoint (`best_checkpoint`); the others are removed.

    Raises:
        StageError: (from the wrapper) A save refreshed no `model.safetensors`.
    """
    original = module.save_checkpoint  # type: ignore[attr-defined]

    def save_checkpoint(*args: object, **kwargs: object) -> object:
        run_dir = Path(str(kwargs.get("run_dir", args[0] if args else "")))
        step = int(kwargs.get("step", args[1] if len(args) > 1 else -1))  # type: ignore[call-overload]
        exported = run_dir / contract.WEIGHTS_NAME
        before = _stamp(exported)
        result = original(*args, **kwargs)
        after = _stamp(exported)
        if after is None or after == before:
            raise StageError(f"the save at step {step} refreshed no {contract.WEIGHTS_NAME}")
        name = contract.checkpoint_name(step)
        export_weights(exported, export_dir / name / contract.WEIGHTS_NAME)
        best = best_checkpoint(run_dir, export_dir)
        prune_exports(run_dir, export_dir, str(best["checkpoint"]) if best else None)
        return result

    module.save_checkpoint = save_checkpoint  # type: ignore[attr-defined]


class SaveTimer:
    """Times each `save_checkpoint` call, the export copy included when it is installed first.

    The longest is what a stop's save costs on top of the step under way.
    """

    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self.durations: list[float] = []

    def install(self, module: object) -> None:
        """Wrap `module.save_checkpoint`; a save that raises is timed too."""
        original = module.save_checkpoint  # type: ignore[attr-defined]

        def save_checkpoint(*args: object, **kwargs: object) -> object:
            start = self._clock()
            try:
                return original(*args, **kwargs)
            finally:
                self.durations.append(self._clock() - start)

        module.save_checkpoint = save_checkpoint  # type: ignore[attr-defined]

    def longest(self) -> float | None:
        """Return the longest save in seconds, or `None` before the first."""
        return max(self.durations, default=None)


def install_exit_hook(module: object, before_exit: Callable[[], None]) -> Callable[[], None]:
    """Wrap `module.exit_training` so `before_exit` runs first; return the call that undoes it.

    Upstream's `exit_training` ends with `os._exit(0)`, which runs no `finally` and no `atexit`
    handler, so whatever the child must leave behind has to be written before it. The exit still
    happens when `before_exit` raises: a raise out of the wrapper would propagate through
    `train.py` past upstream's cleanup, and the loader processes `exit_training` kills could then
    keep the child alive. `os._exit` drops the exception in flight, so any failure of
    `before_exit` is printed with its traceback first, and the log says why the parent then
    finds no metrics. The flag, not `sys.exc_info()` alone, decides that it failed: a `finally`
    also sees an exception an enclosing `except` is handling.
    """
    original = module.exit_training  # type: ignore[attr-defined]

    def exit_training(*args: object, **kwargs: object) -> None:
        written = False
        try:
            before_exit()
            written = True
        finally:
            if not written:
                print("error: the trainer's metrics could not be written", file=sys.stderr)
                traceback.print_exc()
            original(*args, **kwargs)

    module.exit_training = exit_training  # type: ignore[attr-defined]

    def restore() -> None:
        module.exit_training = original  # type: ignore[attr-defined]

    return restore


def run_trainer(config: Path, metrics_path: Path, upstream: Path, export_dir: Path | None) -> int:
    """Run upstream's trainer on `config` in this process; write the metrics; return the code.

    Upstream ends a run, at its last step or on a stop (a `STOP` file or its own signal handler),
    with `training.distributed.exit_training`, which hard-exits with `os._exit(0)` and so skips
    this function's `finally`. `train.py` binds that function by name when it is imported, so the
    wrapper that writes the metrics first (`install_exit_hook`) is installed on the module before
    `runpy` imports `train.py`, and removed again once the trainer returns or raises. The
    `finally` writes the same metrics on the paths that do not hard-exit: `CHILD_OOM` is returned
    when CUDA ran out of memory, and upstream's own `SystemExit` (a refused config, no CUDA
    device, a non-finite gradient) passes through after the metrics are written.
    """
    check_clone(upstream)
    import torch

    free = int(torch.cuda.mem_get_info()[0])
    monitor = StepMonitor(time.perf_counter)
    checkpointing = importlib.import_module(CHECKPOINTING_MODULE)
    if export_dir is not None:
        install_export(checkpointing, export_dir)
    saves = SaveTimer(save_clock)
    saves.install(checkpointing)

    def write_metrics(oom: bool) -> None:
        _write_json(
            metrics_path,
            {
                "schema": SCHEMA,
                "free_bytes": free,
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "steps": len(monitor.times),
                "steps_per_s": monitor.steps_per_s(),
                "save_s": saves.longest(),
                "losses": monitor.windows(),
                "oom": oom,
            },
        )

    distributed = importlib.import_module(DISTRIBUTED_MODULE)
    saved_argv, saved_cwd = sys.argv, os.getcwd()
    oom = False
    monitor.install(torch)
    # The last statement before `try`, so the `finally` that restores it covers every raise.
    restore_exit = install_exit_hook(distributed, lambda: write_metrics(False))
    try:
        os.chdir(upstream)
        sys.argv = [TRAIN_MODULE, str(config)]
        runpy.run_module(TRAIN_MODULE, run_name="__main__", alter_sys=True)
    except torch.cuda.OutOfMemoryError:
        oom = True
    finally:
        restore_exit()
        sys.argv = saved_argv
        os.chdir(saved_cwd)
        monitor.remove()
        write_metrics(oom)
    return CHILD_OOM if oom else 0


# The command line.


def _outputs(data: Path, run_id: str, written: Sequence[str]) -> None:
    files = [{"rel": rel, "sha256": _sha256(data / rel)} for rel in written]
    path = data / contract.run_rel(run_id) / contract.OUTPUTS_NAME
    _write_json(path, {"schema": SCHEMA, "files": files})


def main(argv: Sequence[str] | None = None) -> int:
    """Run the probe, the training (under a SIGTERM handler) or the trainer child."""
    parser = argparse.ArgumentParser(prog="scripts.voice_data.stages.student_train")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--upstream", type=Path, help="the pinned pocket-tts clone")
    modes = parser.add_subparsers(dest="mode", required=True)
    probe = modes.add_parser("probe")
    probe.add_argument("--run", required=True)
    probe.add_argument("--corpus", required=True)
    probe.add_argument("--batches", type=int, nargs="+", default=list(contract.PROBE_BATCHES))
    train = modes.add_parser("train")
    train.add_argument("--run", required=True)
    train.add_argument("--corpus", required=True)
    train.add_argument("--batch", type=int, required=True)
    train.add_argument("--ckpt-freq", type=int)
    train.add_argument("--steps-finetune", type=int, default=contract.STEPS_FINETUNE)
    train.add_argument("--steps-distill", type=int, default=contract.STEPS_DISTILL)
    train.add_argument("--keep-last", type=int, default=contract.KEEP_LAST)
    train.add_argument("--stage", choices=contract.STAGES)
    train.add_argument("--deadline-utc", type=deadline_mod.parse_utc)
    trainer = modes.add_parser("trainer")
    trainer.add_argument("--config", required=True, type=Path)
    trainer.add_argument("--metrics", required=True, type=Path)
    trainer.add_argument("--export-dir", type=Path)
    args = parser.parse_args(argv)
    upstream = args.upstream or UPSTREAM
    if args.mode == "trainer":
        try:
            return run_trainer(args.config, args.metrics, upstream, args.export_dir)
        except StageError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    fetch.prepare("cuda")
    stop = Stop(getattr(args, "deadline_utc", None), utc_now)
    previous = signal.signal(signal.SIGTERM, stop.on_signal)
    try:
        if args.mode == "probe":
            written, _doc = run_probe(
                args.data, upstream, args.run, args.corpus, args.batches, stop
            )
        else:
            steps = {
                contract.FINETUNE: args.steps_finetune,
                contract.DISTILL: args.steps_distill,
            }
            written, _doc = run_train(
                args.data, upstream, args.run, args.corpus, args.batch, args.ckpt_freq, steps,
                args.stage, stop, args.keep_last,
            )  # fmt: skip
    except StageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)
    _outputs(args.data, args.run, written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
