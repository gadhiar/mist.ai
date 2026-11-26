"""`student probe` and `student train`: the Pocket student's fit probe and resumable training.

The commands run through the CLI, and their job's stage runs in this process (`StageDocker`), as
`test_corpus.py` does. Upstream's trainer is replaced by `FakeTrainer`, whose children behave as
`training/train.py` is documented to: each poll is one optimizer step on a shared fake clock;
it resumes from its run_dir's newest `checkpoint_<step>.pt`, saves at `ckpt_freq` and at
`max_steps`, keeps `num_ckpt_keep` checkpoints, refreshes `model.safetensors` at each save, and
when a `STOP` file appears saves after the current step and exits 0. A fake `huggingface_hub`
holds the cached files, and a fake clone holds upstream's configs. The trainer child itself
(`trainer` mode) runs against a second fake clone whose `training/train.py` is a small script,
with fake `torch` and `safetensors` modules, so no test needs CUDA.

The canary transcript is in every corpus row; it must reach no printed line, and no printed line
names a clip.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import os
import signal
import subprocess
import sys
import time
import types
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from scripts.voice_data import cli, jobs
from scripts.voice_data import root as root_mod
from scripts.voice_data.commands import student as student_cmd
from scripts.voice_data.records import write_json
from scripts.voice_data.stages import student_train as stage

from .conftest import Canaries, _assert_no_canary
from .test_identity import _Capture

RUN = "s1"
CORPUS = "c1"
TAKES_RUN = "g1"
SEED_RUN = "r1"
IMAGE = "mist-voice-train-pocket:0123456789ab"
CANARY = Canaries()
MIB = 1 << 20
# Free VRAM the fake nvidia-smi reports: under the family's 16 GB, over the probe's 7.5.
FREE_MIB = 10240
T0 = datetime(2026, 11, 24, 15, 0, tzinfo=UTC)
CLONE_COMMIT = "ee2ebc89aba9d0121d04d18e29c20eea75bb624e"
POCKET = jobs.load_families()["train-pocket"]["models"]["pocket-tts"]
RUNTIME_TOKENIZER = jobs.load_families()["train-pocket"]["models"]["pocket-tts-runtime-tokenizer"]
TOKENIZER = b'{"model": "a tokenizer"}'
NEW_YORK = "EST5EDT,M3.2.0,M11.1.0"  # a POSIX rule: the image has no zoneinfo database

FINETUNE_YAML = """\
model_config: pocket_tts/config/english_2026-04_24l.yaml
start_from_pretrained: true
data:
  train_jsonl: data/train_aligned.jsonl
  valid_jsonl: data/valid_aligned.jsonl
  max_duration_sec: 30.0
  max_voice_prompt_sec: 5.0
flow:
  type: lsd
  kwargs: {}
flow_batch_multiplier: 4
eos_loss_weight: 0.1
text_dropout: 0.2
voice_dropout: 0.2
optim:
  lr: 2e-5
  muon_lr: 5e-4
  schedule: constant
  warmup_steps: 1000
run_dir: runs/finetune
batch_size: 64
grad_accum_steps: 1
max_steps: 20000
stats_ema_steps: 1000
ema_decay: 0.999
log_freq: 50
valid_freq: 2500
ckpt_freq: 2500
num_ckpt_keep: 3
sample_cfg_coef: 2.0
"""

DISTILL_YAML = """\
model_config: pocket_tts/config/english.yaml   # 6-layer student
start_from_pretrained: false
distill_cfg_coef: 2.0
distill_teacher_config: pocket_tts/config/english.yaml
distill_teacher_overrides:
  flow_lm.transformer.num_layers: 24
distill_teacher_weights: runs/scratch/checkpoint_00400000.pt
data:
  train_jsonl: data/train_aligned.jsonl
  valid_jsonl: data/valid_aligned.jsonl
flow:
  type: lsd
  kwargs: {}
text_dropout: 0.0
voice_dropout: 0.0
optim:
  lr: 4e-4
  schedule: cosine
  warmup_steps: 1000
run_dir: runs/depth_distill
batch_size: 64
grad_accum_steps: 1
max_steps: 200000
ema_decay: 0.9999
log_freq: 50
valid_freq: 2500
ckpt_freq: 2500
num_ckpt_keep: 999
"""

TEACHER_MODEL_YAML = """\
weights_path: hf://kyutai/pocket-tts/languages/english_2026-04_24l/model.safetensors@492522650173a0653b7575cdc25ae09810e5d741
flow_lm:
  transformer:
    num_layers: 24
  lookup_table:
    tokenizer: tokenizers
    tokenizer_path: hf://kyutai/pocket-tts-without-voice-cloning/languages/english_2026-04_24l/tokenizer.json@00eac05ed3d16bdc3f6b5d598874019c34a89214
"""

STUDENT_MODEL_YAML = """\
weights_path: hf://kyutai/pocket-tts/languages/english/model.safetensors@db04f0e653473e5c7333c69686ac92b7460b0c1d
weights_path_without_voice_cloning: hf://kyutai/pocket-tts-without-voice-cloning/languages/english/model.safetensors@a7056f505a4a8ca72d8b5fb1440071d6f1a35739
flow_lm:
  transformer:
    num_layers: 6
  lookup_table:
    tokenizer: tokenizers
    tokenizer_path: hf://kyutai/pocket-tts-without-voice-cloning/languages/english/tokenizer.json@00eac05ed3d16bdc3f6b5d598874019c34a89214
"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# The corpus T4 builds, the cache and the clone.


def write_corpus(root: root_mod.Root, runs: Sequence[str] = (TAKES_RUN,)) -> dict[str, list[str]]:
    """Write a built corpus: splits naming teacher takes and seed segments, and its manifest.

    One run gives a schema-1 manifest, as `corpus build` wrote before corpora combined runs;
    more give a schema-2 manifest, its takes spread over the runs in turn.
    """
    takes = [f"teachers/omnivoice/{runs[k % len(runs)]}/takes/take-{k:04d}.flac" for k in range(6)]
    segments = [f"seed/runs/{SEED_RUN}/audio/seg-{k:06d}.flac" for k in range(2)]
    for rel in takes + segments:
        path = root.path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"clip {rel}".encode())

    def row(rel: str) -> str:
        words = [{"word": w, "start": 0.1 * i, "end": 0.1 * i + 0.1}
                 for i, w in enumerate(CANARY.transcript.split())]  # fmt: skip
        doc = {"path": f"/data/{rel}", "duration": 2.0, "transcript": CANARY.transcript,
               "words": words}  # fmt: skip
        return json.dumps(doc) + "\n"

    folder = root.path / "corpus" / CORPUS
    folder.mkdir(parents=True, exist_ok=True)
    train = takes[:5] + segments * 3
    (folder / "train_aligned.jsonl").write_text("".join(map(row, train)), encoding="utf-8")
    (folder / "valid_aligned.jsonl").write_text(row(takes[5]), encoding="utf-8")
    files = [
        {"rel": f"corpus/{CORPUS}/{name}", "sha256": sha256(folder / name)}
        for name in ("train_aligned.jsonl", "valid_aligned.jsonl")
    ]
    named = {"schema": 1, "takes_run": runs[0]} if len(runs) == 1 else {
        "schema": 2, "takes_runs": list(runs)
    }  # fmt: skip
    manifest = {**named, "version": CORPUS, "seed_run": SEED_RUN, "files": files}
    write_json(root, f"corpus/{CORPUS}/manifest.json", manifest)
    return {"takes": takes, "segments": segments}


@pytest.fixture
def root(private_root: root_mod.Root) -> root_mod.Root:
    write_corpus(private_root)
    return private_root


@dataclass
class Hub:
    folders: dict[str, Path]
    calls: list[tuple[str, str | None, bool]] = field(default_factory=list)

    def file(self, model: tuple[str, str]) -> Path:
        repository = {"pocket-tts": POCKET, "pocket-tts-runtime-tokenizer": RUNTIME_TOKENIZER}
        return self.folders[repository[model[0]]["repository"]] / model[1]


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Hub:
    """A fake `huggingface_hub` whose cache holds train-pocket's two snapshots."""
    folders = {}
    for model in (POCKET, RUNTIME_TOKENIZER):
        repo = model["repository"]
        folders[repo] = (
            tmp_path / "hf" / f"models--{repo.replace('/', '--')}" / "snapshots" / model["revision"]
        )
    seen = Hub(folders)
    files = {
        stage.BASE_WEIGHTS: b"24-layer base weights",
        stage.BASE_TOKENIZER: TOKENIZER,
        stage.STUDENT_WEIGHTS: b"released 6-layer student",
        stage.STUDENT_TOKENIZER: TOKENIZER,
    }
    for model, content in files.items():
        path = seen.file(model)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def snapshot_download(
        repo_id: str, *, revision: str | None = None, repo_type: str | None = None,
        local_files_only: bool = False, **kwargs: object,
    ) -> str:  # fmt: skip
        seen.calls.append((repo_id, revision, local_files_only))
        return str(folders[repo_id])

    module = types.ModuleType("huggingface_hub")
    module.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return seen


@pytest.fixture
def clone(tmp_path: Path) -> Path:
    """The pinned pocket-tts clone's configs, pyproject and detached HEAD."""
    folder = tmp_path / "upstream" / "pocket-tts"
    texts = {
        stage.TEACHER_MODEL_CONFIG: TEACHER_MODEL_YAML,
        stage.STUDENT_MODEL_CONFIG: STUDENT_MODEL_YAML,
        stage.FINETUNE_CONFIG: FINETUNE_YAML,
        stage.DISTILL_CONFIG: DISTILL_YAML,
        "pyproject.toml": '[project]\nname = "pocket-tts"\nversion = "4.1.0.dev0"\n',
        ".git/HEAD": CLONE_COMMIT + "\n",
    }
    for rel, text in texts.items():
        path = folder / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return folder


class FakeSafetensorsTorch(types.ModuleType):
    """`safetensors.torch` over JSON files: a "tensor" is a list of numbers."""

    def __init__(self) -> None:
        super().__init__("safetensors.torch")

    def save_file(self, tensors: dict, path: str) -> None:
        Path(path).write_text(json.dumps(tensors), encoding="utf-8")

    def load_file(self, path: str) -> dict:
        return json.loads(Path(path).read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def safetensors_fake(monkeypatch: pytest.MonkeyPatch) -> FakeSafetensorsTorch:
    package = types.ModuleType("safetensors")
    module = FakeSafetensorsTorch()
    package.torch = module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "safetensors", package)
    monkeypatch.setitem(sys.modules, "safetensors.torch", module)
    return module


# The trainer.


def write_valid_line(run_dir: Path, step: int, loss: float) -> None:
    """Append the line upstream logs after validating (`ProgressLog.log("valid", ...)`)."""
    doc = {"time": "2026-11-24 15:00:00+0000", "type": "valid", "step": step,
           "metrics": {"flow_loss": loss - 0.1, "loss": loss}}  # fmt: skip
    write_progress_line(run_dir, doc)


def write_progress_line(run_dir: Path, doc: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(doc) + "\n")


def upstream_save(keep: int) -> object:
    """`save_checkpoint` as upstream's: a `.pt`, pruned to `keep`, and a refreshed export."""

    def save_checkpoint(run_dir: Path, step: int) -> None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / f"checkpoint_{step:08d}.pt").write_bytes(b"model ema")
        for old in sorted(run_dir.glob("checkpoint_*.pt"))[:-keep]:
            old.unlink()
        state = {"flow_lm.w": [float(step)], "mimi.w": [1.0]}
        sys.modules["safetensors.torch"].save_file(state, str(run_dir / "model.tmp"))
        (run_dir / "model.tmp").rename(run_dir / "model.safetensors")

    return save_checkpoint


class FakeClock:
    def __init__(self, now: datetime = T0) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


@dataclass
class Start:
    name: str
    step: int
    config: dict
    export_dir: Path | None


@dataclass
class FakeTrainer:
    """Upstream's trainer in behaviour; `kill_at`, `signal_at` and `jump_at` name `(run, step)`."""

    clock: FakeClock = field(default_factory=FakeClock)
    seconds_per_step: float = 10.0
    peaks_mib: dict[int, int] = field(default_factory=lambda: {2: 6100, 4: 7600, 8: 9900})
    rates: dict[int, float] = field(default_factory=lambda: {2: 0.02, 4: 0.025, 8: 0.03})
    save_s: dict[int, float] = field(default_factory=dict)
    # Validation losses by `(run, step)`; a step not named falls with the step.
    valid_losses: dict[tuple[str, int], float] = field(default_factory=dict)
    free_mib: int = FREE_MIB
    oom: set[int] = field(default_factory=set)
    kill_at: tuple[str, int] | None = None
    signal_at: tuple[str, int] | None = None
    jump_at: tuple[str, int] | None = None
    starts: list[Start] = field(default_factory=list)
    saves: list[tuple[str, int, datetime]] = field(default_factory=list)

    def __call__(
        self, data: Path, upstream: Path, config: Path, metrics: Path, export_dir: Path | None
    ) -> FakeChild:
        return FakeChild(self, yaml.safe_load(config.read_text(encoding="utf-8")), metrics,
                         export_dir)  # fmt: skip

    def starts_of(self, name: str) -> list[int]:
        return [s.step for s in self.starts if s.name == name]


class FakeChild:
    def __init__(
        self, trainer: FakeTrainer, config: dict, metrics: Path, export_dir: Path | None
    ) -> None:
        self.trainer, self.config, self.metrics = trainer, config, metrics
        self.run_dir = Path(config["run_dir"])
        self.name = self.run_dir.name
        self.code: int | None = None
        self.local = 0
        if (self.run_dir / "STOP").exists():  # upstream renames a STOP left from a stop
            (self.run_dir / "STOP").rename(self.run_dir / "STOP.done")
        self.step = stage.newest_step(self.run_dir)
        trainer.starts.append(Start(self.name, self.step, config, export_dir))
        self.module = types.SimpleNamespace(save_checkpoint=self._save)
        if export_dir is not None:
            stage.install_export(self.module, export_dir)

    def _save(self, run_dir: Path, step: int, **kwargs: object) -> None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / f"checkpoint_{step:08d}.pt").write_bytes(b"model ema")
        (run_dir / f"optim_{step:08d}.pt").write_bytes(b"optimizer")
        for old in sorted(run_dir.glob("checkpoint_*.pt"))[: -self.config["num_ckpt_keep"]]:
            old.unlink()
        state = {"flow_lm.layer.weight": [float(step)], "mimi.encoder.weight": [0.5]}
        # As `export_pocket_safetensors` does: a temporary file, then a rename over the export.
        sys.modules["safetensors.torch"].save_file(state, str(run_dir / "model.tmp"))
        (run_dir / "model.tmp").rename(run_dir / "model.safetensors")
        self.trainer.saves.append((self.name, step, self.trainer.clock()))

    def _write_metrics(self, oom: bool) -> None:
        batch = self.config["batch_size"]
        doc = {
            "free_bytes": self.trainer.free_mib * MIB,
            "peak_reserved_bytes": self.trainer.peaks_mib.get(batch, 0) * MIB,
            "steps": self.local,
            "steps_per_s": None if oom else self.trainer.rates.get(batch),
            "save_s": None if oom else self.trainer.save_s.get(batch),
            "losses": [[self.local, 2.0 - 0.01 * self.step]] if self.local else [],
            "oom": oom,
        }
        self.metrics.write_text(json.dumps(doc), encoding="utf-8")

    def poll(self) -> int | None:
        if self.code is not None:
            return self.code
        if self.config["batch_size"] in self.trainer.oom:
            self._write_metrics(oom=True)
            self.code = stage.CHILD_OOM
            return self.code
        self.step += 1
        self.local += 1
        self.trainer.clock.advance(self.trainer.seconds_per_step)
        here = (self.name, self.step)
        if here == self.trainer.kill_at:
            self.code = -9
            return self.code
        if here == self.trainer.jump_at:
            self.trainer.clock.advance(25 * 3600)
        if here == self.trainer.signal_at:
            signal.raise_signal(signal.SIGTERM)
        stop = (self.run_dir / "STOP").exists()
        last = self.step >= self.config["max_steps"]
        if self.step % self.config["valid_freq"] == 0:
            loss = self.trainer.valid_losses.get(here, 2.0 - 0.01 * self.step)
            write_valid_line(self.run_dir, self.step, loss)
        if stop or last or self.step % self.config["ckpt_freq"] == 0:
            self.module.save_checkpoint(self.run_dir, self.step)
        if stop or last:
            self._write_metrics(oom=False)
            self.code = 0
        return self.code


class StageDocker:
    """A `DockerOps` whose `run_logged` runs the job's stage in this process."""

    def __init__(self, root: root_mod.Root, trainer: FakeTrainer, upstream: Path) -> None:
        self.root = root
        self.trainer = trainer
        self.upstream = upstream
        self.argvs: list[list[str]] = []

    def capture(self, argv: Sequence[str]) -> str:
        if list(argv[:3]) == ["docker", "ps", "-q"]:
            return ""
        assert "nvidia-smi" in argv
        return f"{FREE_MIB}\n"

    def check(self, argv: Sequence[str]) -> None:
        raise AssertionError("student builds no image")

    def run_logged(self, argv: Sequence[str], log: Path) -> int:
        self.argvs.append(list(argv))
        at = list(argv).index("-m")
        module, rest = argv[at + 1], list(argv[at + 2 :])
        assert rest[:2] == ["--data", "/data"]
        rest[1] = str(self.root.path)
        module_obj = importlib.import_module(module)
        with (
            pytest.MonkeyPatch.context() as mp,
            log.open("w", encoding="utf-8") as handle,
            contextlib.redirect_stdout(handle),
            contextlib.redirect_stderr(handle),
        ):
            mp.setattr(module_obj, "spawn", self.trainer)
            mp.setattr(module_obj, "utc_now", self.trainer.clock)
            mp.setattr(module_obj, "sleep", lambda seconds: None)
            mp.setattr(module_obj, "UPSTREAM", self.upstream)
            mp.setattr(module_obj, "installed_version", lambda: "4.0.1")
            mp.setattr(module_obj.fetch, "prepare", lambda *names: None)
            return module_obj.main(rest)


@dataclass
class Result:
    code: int
    out: str
    err: str
    docker: StageDocker

    @property
    def job(self) -> dict:
        latest = sorted((self.docker.root.path / "runs").glob("job-*/job.json"))[-1]
        return json.loads(latest.read_text(encoding="utf-8"))

    @property
    def log(self) -> str:
        latest = sorted((self.docker.root.path / "runs").glob("job-*/log.txt"))[-1]
        return latest.read_text(encoding="utf-8")


def student(
    root: root_mod.Root,
    clone: Path,
    *args: str,
    trainer: FakeTrainer | None = None,
    host_now: datetime | None = None,
) -> Result:
    docker = StageDocker(root, trainer or FakeTrainer(), clone)
    out, err = _Capture(), _Capture()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(student_cmd, "make_docker", lambda: docker)
        mp.setattr(student_cmd, "image_tag", lambda family: IMAGE)
        if host_now is not None:
            mp.setattr(student_cmd, "utc_now", lambda: host_now)
        mp.delenv(jobs.GPU_PEERS_ENV, raising=False)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(["--root", str(root.path), "student", *args])
    result = Result(code, out.getvalue(), err.getvalue(), docker)
    _assert_no_canary(result.out + result.err)
    assert ".wav" not in result.out + result.err and ".flac" not in result.out + result.err
    return result


def probe(root: root_mod.Root, clone: Path, *args: str, **kwargs: object) -> Result:
    return student(root, clone, "probe", "--corpus", CORPUS, "--run", RUN, *args, **kwargs)  # type: ignore[arg-type]


def train(root: root_mod.Root, clone: Path, *args: str, **kwargs: object) -> Result:
    return student(root, clone, "train", "--corpus", CORPUS, "--run", RUN, *args, **kwargs)  # type: ignore[arg-type]


def run_dir(root: root_mod.Root) -> Path:
    return root.path / "students" / RUN


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


ABSENT = object()


def write_probe(
    root: root_mod.Root, *, batch: int | None = 4, steps_per_s: float = 0.025,
    peak: float = 7600.0, corpus: str = CORPUS, save_s: object = 2.0,
) -> None:  # fmt: skip
    doc = {
        "schema": 1, "run": RUN, "corpus": corpus, "batch": batch,
        "accum": None if batch is None else 64 // batch, "peak_vram_mib": peak,
        "free_vram_mib": float(FREE_MIB), "steps_per_s": steps_per_s, "projected_hours": 1.0,
        "steps": {"finetune": 8000, "distill": 20000}, "trials": [],
    }  # fmt: skip
    if save_s is not ABSENT:
        doc["save_s"] = save_s
    write_json(root, f"students/{RUN}/probe.json", doc)


def stage_config(root: root_mod.Root, name: str) -> dict:
    return yaml.safe_load((run_dir(root) / "configs" / f"{name}.yaml").read_text("utf-8"))


# `student probe`.


def test_probe_is_a_gpu_job_on_the_corpus_and_its_clips_at_7_5_gb(
    root: root_mod.Root, hub: Hub, clone: Path
) -> None:
    result = probe(root, clone)
    assert result.code == 0, result.err + result.log
    job = result.job
    assert (job["family"], job["module"], job["gpu"], job["network"]) == (
        "train-pocket", "scripts.voice_data.stages.student_train", True, False
    )  # fmt: skip
    assert job["vram_gb"] == 7.5
    assert job["args"] == ["probe", "--run", RUN, "--corpus", CORPUS, "--batches", "2", "4", "8"]
    assert [i["rel"] for i in job["inputs"]] == [
        f"corpus/{CORPUS}/train_aligned.jsonl", f"corpus/{CORPUS}/valid_aligned.jsonl",
        f"teachers/omnivoice/{TAKES_RUN}", f"seed/runs/{SEED_RUN}",
    ]  # fmt: skip
    assert job["outputs"] == f"students/{RUN}"
    argv = result.docker.argvs[0]
    assert argv[argv.index("--network") + 1] == "none"


def test_the_family_vram_estimate_would_refuse_the_probe(root: root_mod.Root, clone: Path) -> None:
    executor = jobs.LocalDocker(root, StageDocker(root, FakeTrainer(), clone), jobs.load_families())
    spec = jobs.JobSpec(
        job_id="job-x", family="train-pocket", image=IMAGE, module=student_cmd.MODULE, args=(),
        inputs=(), outputs=f"students/{RUN}", gpu=True, network=False,
    )  # fmt: skip
    with pytest.raises(jobs.PreflightError):
        executor.preflight(spec)
    executor.preflight(replace(spec, vram_gb=student_cmd.PROBE_VRAM_GB))


def test_probe_trains_50_steps_per_batch_and_keeps_the_largest_that_fits(
    root: root_mod.Root, hub: Hub, clone: Path
) -> None:
    # 2 and 4 fit under 10240 - 512 MiB; 8 peaks at 9900 MiB, inside the card but not the margin.
    trainer = FakeTrainer()
    result = probe(root, clone, trainer=trainer)
    assert result.code == 0, result.err + result.log
    doc = read(run_dir(root) / "probe.json")
    assert {k: doc[k] for k in ("batch", "accum", "peak_vram_mib", "free_vram_mib")} == {
        "batch": 4, "accum": 16, "peak_vram_mib": 7600.0, "free_vram_mib": float(FREE_MIB)
    }  # fmt: skip
    assert doc["steps_per_s"] == 0.025
    assert doc["projected_hours"] == pytest.approx((8000 + 20000) / 0.025 / 3600, abs=1e-3)
    assert [(t["batch"], t["fits"]) for t in doc["trials"]] == [(2, True), (4, True), (8, False)]
    assert [s.name for s in trainer.starts] == ["batch-2", "batch-4", "batch-8"]
    for start, batch in zip(trainer.starts, (2, 4, 8), strict=True):
        config = start.config
        assert (config["batch_size"], config["grad_accum_steps"]) == (batch, 64 // batch)
        assert (config["max_steps"], config["ckpt_freq"]) == (50, 50)
        assert config["compile"] is False and config["sample_sentences"] == []
        assert config["model_config"] == str(run_dir(root) / "configs" / "teacher-model.yaml")
    # Each trial's checkpoints are removed once measured.
    assert not any((run_dir(root) / "probe").glob("batch-*/checkpoint_*.pt"))
    assert "batch 4 x accum 16" in result.out and "too big" in result.out


def test_probe_exits_3_when_batch_2_exceeds_free_vram_less_512_mib(
    root: root_mod.Root, hub: Hub, clone: Path
) -> None:
    trainer = FakeTrainer(peaks_mib={2: FREE_MIB - 500})
    result = probe(root, clone, trainer=trainer)
    assert result.code == 3
    assert "no batch fits" in result.out and "batch 2 does not fit" in result.err
    doc = read(run_dir(root) / "probe.json")
    assert doc["batch"] is None and doc["projected_hours"] is None
    assert [s.name for s in trainer.starts] == ["batch-2"]
    refused = train(root, clone)
    assert refused.code == 2 and "no batch that fits" in refused.err


def test_probe_counts_out_of_memory_as_not_fitting(
    root: root_mod.Root, hub: Hub, clone: Path
) -> None:
    result = probe(root, clone, trainer=FakeTrainer(oom={8}))
    assert result.code == 0, result.err + result.log
    doc = read(run_dir(root) / "probe.json")
    assert doc["batch"] == 4
    assert doc["trials"][-1]["batch"] == 8 and doc["trials"][-1]["oom"] is True
    assert "out of memory" in result.out


def test_probe_records_its_longest_checkpoint_save(
    root: root_mod.Root, hub: Hub, clone: Path
) -> None:
    # Batch 8 does not fit, but its save was timed too; the longest is kept, to one decimal.
    trainer = FakeTrainer(save_s={2: 1.24, 4: 2.36, 8: 1.9})
    result = probe(root, clone, trainer=trainer)
    assert result.code == 0, result.err + result.log
    doc = read(run_dir(root) / "probe.json")
    assert doc["save_s"] == 2.4
    assert [t["save_s"] for t in doc["trials"]] == [1.24, 2.36, 1.9]


def test_probe_save_s_is_null_when_no_save_was_timed(
    root: root_mod.Root, hub: Hub, clone: Path
) -> None:
    result = probe(root, clone, trainer=FakeTrainer(oom={2}))
    assert result.code == 3
    doc = read(run_dir(root) / "probe.json")
    assert doc["save_s"] is None and doc["trials"][0]["save_s"] is None


def test_probe_batch_probes_that_batch_alone(root: root_mod.Root, hub: Hub, clone: Path) -> None:
    trainer = FakeTrainer(peaks_mib={16: 9000}, rates={16: 0.1})
    result = probe(root, clone, "--batch", "16", trainer=trainer)
    assert result.code == 0, result.err + result.log
    assert result.job["args"][-2:] == ["--batches", "16"]
    doc = read(run_dir(root) / "probe.json")
    assert (doc["batch"], doc["accum"]) == (16, 4)


def test_probe_resolves_the_cached_files_and_writes_local_model_configs(
    root: root_mod.Root, hub: Hub, clone: Path
) -> None:
    assert probe(root, clone).code == 0
    config = stage_config(root, "teacher-model")
    assert config["weights_path"] == str(hub.file(stage.BASE_WEIGHTS))
    assert config["flow_lm"]["lookup_table"]["tokenizer_path"] == str(
        hub.file(stage.BASE_TOKENIZER)
    )
    assert config["flow_lm"]["transformer"]["num_layers"] == 24
    assert {(repo, rev) for repo, rev, _ in hub.calls} == {
        (POCKET["repository"], POCKET["revision"])
    }
    assert all(offline for _, _, offline in hub.calls)


def test_the_corpus_copy_is_writable_and_points_at_the_mounted_clips(
    root: root_mod.Root, hub: Hub, clone: Path
) -> None:
    trainer = FakeTrainer()
    assert probe(root, clone, trainer=trainer).code == 0
    data = trainer.starts[0].config["data"]
    copy = run_dir(root) / "data"
    assert data["train_jsonl"] == str(copy / "train_aligned.jsonl")
    assert data["valid_jsonl"] == str(copy / "valid_aligned.jsonl")
    assert data["max_duration_sec"] == 30.0
    source = root.path / "corpus" / CORPUS / "train_aligned.jsonl"
    rows = [json.loads(r) for r in (copy / "train_aligned.jsonl").read_text("utf-8").splitlines()]
    originals = [json.loads(r) for r in source.read_text("utf-8").splitlines()]
    assert len(rows) == len(originals)
    for row, original in zip(rows, originals, strict=True):
        assert row["path"] == str(root.path / original["path"].removeprefix("/data/"))
        assert {k: v for k, v in row.items() if k != "path"} == {
            k: v for k, v in original.items() if k != "path"
        }


def test_probe_refuses_a_corpus_without_a_manifest(
    root: root_mod.Root, hub: Hub, clone: Path
) -> None:
    (root.path / "corpus" / CORPUS / "manifest.json").unlink()
    result = probe(root, clone)
    assert result.code == 2 and "corpus build" in result.err
    assert not result.docker.argvs


TAKES_RUN2 = "g2"


def test_probe_mounts_every_teacher_run_of_a_two_run_corpus(
    root: root_mod.Root, hub: Hub, clone: Path
) -> None:
    clips = write_corpus(root, runs=(TAKES_RUN, TAKES_RUN2))
    result = probe(root, clone)
    assert result.code == 0, result.err + result.log
    inputs = {i["rel"]: i["sha256"] for i in result.job["inputs"]}
    folders = [f"teachers/omnivoice/{TAKES_RUN}", f"teachers/omnivoice/{TAKES_RUN2}",
               f"seed/runs/{SEED_RUN}"]  # fmt: skip
    assert list(inputs) == [
        f"corpus/{CORPUS}/train_aligned.jsonl", f"corpus/{CORPUS}/valid_aligned.jsonl", *folders
    ]  # fmt: skip
    for folder in folders:
        mine = sorted(rel for rel in clips["takes"] + clips["segments"] if rel.startswith(folder))
        assert mine
        parts = [sha256(root.path / rel) for rel in mine]
        assert inputs[folder] == hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _manifest(root: root_mod.Root, change: dict) -> None:
    path = root.path / "corpus" / CORPUS / "manifest.json"
    doc = read(path)
    doc.update(change)
    path.write_text(json.dumps({k: v for k, v in doc.items() if v is not None}), encoding="utf-8")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"takes_runs": [TAKES_RUN]}, "unmounted"),
        ({"takes_runs": None, "takes_run": TAKES_RUN}, "names no teacher run"),
        ({"takes_runs": [TAKES_RUN, "../x"]}, "one folder name"),
    ],
    ids=["a-run-left-out", "schema-1-key", "bad-id"],
)
def test_probe_refuses_a_two_run_corpus_whose_manifest_does_not_match_its_rows(
    root: root_mod.Root, hub: Hub, clone: Path, change: dict, message: str
) -> None:
    write_corpus(root, runs=(TAKES_RUN, TAKES_RUN2))
    _manifest(root, change)
    result = probe(root, clone)
    assert result.code == 2 and message in result.err, result.err
    assert not result.docker.argvs


def test_probe_refuses_a_split_changed_since_its_manifest(
    root: root_mod.Root, hub: Hub, clone: Path
) -> None:
    with (root.path / "corpus" / CORPUS / "valid_aligned.jsonl").open("a") as handle:
        handle.write("\n")
    result = probe(root, clone)
    assert result.code == 2 and "changed since its manifest" in result.err


# `student train`.


def test_train_finetunes_then_distills_from_the_local_files(
    root: root_mod.Root, hub: Hub, clone: Path
) -> None:
    write_probe(root, batch=4, steps_per_s=0.025, peak=7600.0)
    trainer = FakeTrainer()
    result = train(root, clone, "--steps-finetune", "6", "--steps-distill", "8", trainer=trainer)
    assert result.code == 0, result.err + result.log
    job = result.job
    # 0.025 steps/s for 20 minutes is 30 steps; 7600 + 512 MiB is 7.93 GiB.
    assert job["args"] == [
        "train", "--run", RUN, "--corpus", CORPUS, "--batch", "4", "--ckpt-freq", "30",
        "--steps-finetune", "6", "--steps-distill", "8", "--keep-last", "3",
    ]  # fmt: skip
    assert job["vram_gb"] == 8.0
    assert [s.name for s in trainer.starts] == ["finetune", "distill"]
    finetune, distill = (s.config for s in trainer.starts)
    configs = run_dir(root) / "configs"
    assert finetune["model_config"] == str(configs / "teacher-model.yaml")
    assert finetune["start_from_pretrained"] is True
    assert (finetune["batch_size"], finetune["grad_accum_steps"]) == (4, 16)
    assert (finetune["max_steps"], finetune["ckpt_freq"]) == (6, 30)
    assert finetune["run_dir"] == str(run_dir(root) / "checkpoints" / "finetune")
    assert distill["model_config"] == str(configs / "student-model.yaml")
    assert distill["student_init_from"] == str(hub.file(stage.STUDENT_WEIGHTS))
    assert distill["distill_teacher_config"] == str(configs / "teacher-model.yaml")
    assert distill["distill_teacher_weights"] == str(
        run_dir(root) / "checkpoints" / "finetune" / "checkpoint_00000006.pt"
    )
    assert (distill["batch_size"], distill["grad_accum_steps"], distill["max_steps"]) == (4, 16, 8)
    assert distill["start_from_pretrained"] is False
    assert distill["optim"] == yaml.safe_load(DISTILL_YAML)["optim"]
    for config in (finetune, distill):
        assert config["compile"] is False and config["sample_sentences"] == []
    student_model = stage_config(root, "student-model")
    assert student_model["weights_path"] == str(hub.file(stage.STUDENT_WEIGHTS))
    assert student_model["flow_lm"]["lookup_table"]["tokenizer_path"] == str(
        hub.file(stage.STUDENT_TOKENIZER)
    )
    assert student_model["weights_path_without_voice_cloning"].startswith("hf://")
    record = read(run_dir(root) / "train.json")
    assert (record["status"], record["stopped_by"]) == ("done", None)
    assert record["steps"] == {"finetune": {"done": 6, "max": 6}, "distill": {"done": 8, "max": 8}}
    assert record["losses"]["finetune"] == [[6, pytest.approx(1.94)]]
    assert record["versions"] == {
        "pocket_tts_installed": "4.0.1", "pocket_tts_clone": "4.1.0.dev0",
        "clone_commit": CLONE_COMMIT,
    }  # fmt: skip
    assert record["tokenizer_sha256"] == hashlib.sha256(TOKENIZER).hexdigest()
    assert record["files"]["base_weights"] == sha256(hub.file(stage.BASE_WEIGHTS))
    assert record["files"]["student_model_config"] == sha256(configs / "student-model.yaml")
    assert record["files"]["distill_config"] == sha256(configs / "distill.yaml")
    assert "student train s1: done" in result.out and "distill: step 8 of 8" in result.out


def test_train_steps_are_pre_registered_at_8k_and_20k(
    root: root_mod.Root, hub: Hub, clone: Path
) -> None:
    write_probe(root)
    trainer = FakeTrainer(kill_at=("finetune", 1))
    result = train(root, clone, trainer=trainer)
    assert result.code == 4
    args = result.job["args"]
    assert args[args.index("--steps-finetune") + 1] == "8000"
    assert args[args.index("--steps-distill") + 1] == "20000"
    assert trainer.starts[0].config["max_steps"] == 8000
    plan = read(run_dir(root) / "train-plan.json")
    assert (plan["effective_batch"], plan["steps"]) == (64, {"finetune": 8000, "distill": 20000})


def test_distill_checkpoints_keep_runtime_layout_student_weights(
    root: root_mod.Root, hub: Hub, clone: Path
) -> None:
    write_probe(root, steps_per_s=0.002)  # a checkpoint every 2 steps
    # The fake's validation loss falls with the step, so the best is among the last two.
    result = train(root, clone, "--steps-finetune", "4", "--steps-distill", "7", "--keep-last", "2")
    assert result.code == 0, result.err + result.log
    distill = run_dir(root) / "checkpoints" / "distill"
    kept = sorted(p.stem for p in distill.glob("checkpoint_*.pt"))
    assert kept == ["checkpoint_00000006", "checkpoint_00000007"]
    weights = run_dir(root) / "checkpoints" / "weights"
    assert sorted(p.name for p in weights.iterdir()) == kept
    for name in kept:
        rel = student_cmd.weights_rel(RUN, name)
        state = read(root.path / rel)
        assert state == {"flow_lm.layer.weight": [float(name[-1])], "mimi.encoder.weight": [0.5]}
    listed = {f["rel"] for f in read(run_dir(root) / "outputs.json")["files"]}
    assert {student_cmd.weights_rel(RUN, name) for name in kept} <= listed


def rewrite_valid(root: root_mod.Root, rows: int) -> None:
    """Make the corpus's validation split `rows` rows long, its manifest updated."""
    folder = root.path / "corpus" / CORPUS
    line = (folder / "valid_aligned.jsonl").read_text(encoding="utf-8").splitlines()[0]
    (folder / "valid_aligned.jsonl").write_text((line + "\n") * rows, encoding="utf-8")
    manifest = read(folder / "manifest.json")
    manifest["files"] = [
        {"rel": f"corpus/{CORPUS}/{name}", "sha256": sha256(folder / name)}
        for name in ("train_aligned.jsonl", "valid_aligned.jsonl")
    ]
    write_json(root, f"corpus/{CORPUS}/manifest.json", manifest)


@pytest.mark.parametrize(
    ("args", "rows", "keep", "batches"),
    [
        pytest.param((), 1, 3, 1, id="default-one-row"),
        pytest.param(("--keep-last", "5"), 9, 5, 2, id="keep-5-nine-rows"),
    ],
)
def test_train_sets_retention_and_validation_for_both_stages(
    root: root_mod.Root, hub: Hub, clone: Path, args: tuple[str, ...], rows: int, keep: int,
    batches: int,
) -> None:  # fmt: skip
    rewrite_valid(root, rows)
    write_probe(root, batch=4, steps_per_s=0.025)  # a checkpoint every 30 steps
    trainer = FakeTrainer()
    result = train(
        root, clone, "--steps-finetune", "2", "--steps-distill", "2", *args, trainer=trainer
    )
    assert result.code == 0, result.err + result.log
    assert result.job["args"][-2:] == ["--keep-last", str(keep)]
    assert read(run_dir(root) / "train-plan.json")["keep_last"] == keep
    assert [s.name for s in trainer.starts] == ["finetune", "distill"]
    for start in trainer.starts:
        config = start.config
        # The validation split's whole batches of 4, each validated once at every checkpoint.
        assert (config["num_ckpt_keep"], config["valid_freq"], config["ckpt_freq"]) == (
            keep, 30, 30
        )  # fmt: skip
        assert config["num_valid_batches"] == batches


def test_train_refuses_keep_last_below_1(root: root_mod.Root, hub: Hub, clone: Path) -> None:
    write_probe(root)
    result = train(root, clone, "--keep-last", "0")
    assert result.code == 2 and "--keep-last is at least 1" in result.err
    assert not result.docker.argvs


def test_a_resumed_run_keeps_its_planned_keep_last(
    root: root_mod.Root, hub: Hub, clone: Path
) -> None:
    write_probe(root)
    killed = train(root, clone, "--keep-last", "5", trainer=FakeTrainer(kill_at=("finetune", 1)))
    assert killed.code == 4
    trainer = FakeTrainer()
    resumed = train(root, clone, "--resume", "--steps-finetune", "1", "--steps-distill", "1",
                    trainer=trainer)  # fmt: skip
    assert resumed.code == 0, resumed.err + resumed.log
    assert resumed.job["args"][-2:] == ["--keep-last", "5"]
    assert {s.config["num_ckpt_keep"] for s in trainer.starts} == {5}


def test_distill_keeps_the_last_n_and_the_best_validated_weights(
    root: root_mod.Root, hub: Hub, clone: Path
) -> None:
    write_probe(root, steps_per_s=0.002)  # a checkpoint and a validation every 2 steps
    losses = {2: 0.9, 4: 0.4, 6: 0.7, 8: 0.6}
    trainer = FakeTrainer(valid_losses={("distill", s): v for s, v in losses.items()})
    result = train(
        root, clone, "--steps-finetune", "2", "--steps-distill", "9", "--keep-last", "2",
        trainer=trainer,
    )  # fmt: skip
    assert result.code == 0, result.err + result.log
    distill = run_dir(root) / "checkpoints" / "distill"
    # Upstream keeps the last two; step 9's save (the last step) had no validation.
    assert sorted(p.stem for p in distill.glob("checkpoint_*.pt")) == [
        "checkpoint_00000008", "checkpoint_00000009"
    ]  # fmt: skip
    weights = run_dir(root) / "checkpoints" / "weights"
    kept = ["checkpoint_00000004", "checkpoint_00000008", "checkpoint_00000009"]
    assert sorted(p.name for p in weights.iterdir()) == kept
    record = read(run_dir(root) / "train.json")
    assert record["best"] == {"checkpoint": "checkpoint_00000004", "step": 4, "valid_loss": 0.4}
    listed = {f["rel"] for f in read(run_dir(root) / "outputs.json")["files"]}
    assert {student_cmd.weights_rel(RUN, name) for name in kept} <= listed
    assert student_cmd.weights_rel(RUN, "checkpoint_00000006") not in listed
    assert "best: checkpoint_00000004 at step 4, valid loss 0.4000" in result.out


def test_train_json_best_is_null_before_the_first_validation(
    root: root_mod.Root, hub: Hub, clone: Path
) -> None:
    write_probe(root)  # a checkpoint every 30 steps: none is validated in 2
    result = train(root, clone, "--steps-finetune", "2", "--steps-distill", "2")
    assert result.code == 0, result.err + result.log
    assert read(run_dir(root) / "train.json")["best"] is None
    assert "best:" not in result.out


def test_retention_breaks_ties_by_the_earlier_step_and_skips_unvalidated_saves(
    tmp_path: Path,
) -> None:
    run, weights = tmp_path / "distill", tmp_path / "weights"
    module = types.SimpleNamespace(save_checkpoint=upstream_save(keep=2))
    stage.install_export(module, weights)

    def save(step: int, loss: float | None) -> list[int]:
        if loss is not None:
            write_valid_line(run, step, loss)
        module.save_checkpoint(run, step)
        return sorted(int(p.name[-8:]) for p in weights.iterdir())

    assert save(2, 0.5) == [2]
    assert save(4, 0.5) == [2, 4]
    # Steps 2 and 4 tie: the earlier is the best, kept beside the last two.
    assert save(6, 0.6) == [2, 4, 6]
    # A stop's save has no figure: it is one of the last two, and never the best.
    assert save(7, None) == [2, 6, 7]
    assert stage.best_checkpoint(run, weights) == {
        "checkpoint": "checkpoint_00000002", "step": 2, "valid_loss": 0.5
    }  # fmt: skip
    # A validation whose save never landed (the trainer was killed) and a torn last line
    # are passed over.
    write_valid_line(run, 10, 0.1)
    with (run / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"kind": "valid", "st')
    assert stage.best_checkpoint(run, weights)["step"] == 2  # type: ignore[index]
    with (run / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    # A new best within the last two releases the old one.
    assert save(8, 0.3) == [7, 8]
    assert stage.best_checkpoint(run, weights)["checkpoint"] == "checkpoint_00000008"  # type: ignore[index]


def test_only_a_valid_line_with_a_metrics_loss_is_a_validation_loss(tmp_path: Path) -> None:
    run, weights = tmp_path / "distill", tmp_path / "weights"
    for step in (2, 4, 6, 8):
        (weights / f"checkpoint_{step:08d}").mkdir(parents=True)
        (weights / f"checkpoint_{step:08d}" / "model.safetensors").write_text("{}", "utf-8")
    write_valid_line(run, 2, 0.9)
    # A top-level `loss` with no `metrics`, metrics that are not a mapping, a checkpoint line.
    write_progress_line(run, {"type": "valid", "step": 4, "loss": 0.1})
    write_progress_line(run, {"type": "valid", "step": 6, "metrics": [0.1]})
    write_progress_line(run, {"type": "checkpoint", "step": 8, "metrics": {"loss": 0.1}})
    assert stage.valid_losses(run) == {2: 0.9}
    assert stage.best_checkpoint(run, weights)["step"] == 2  # type: ignore[index]


def test_best_checkpoint_is_none_without_progress(tmp_path: Path) -> None:
    assert stage.best_checkpoint(tmp_path / "distill", tmp_path / "weights") is None


def test_a_checkpoint_lands_at_least_every_30_minutes(
    root: root_mod.Root, hub: Hub, clone: Path
) -> None:
    # The probe's rate says a checkpoint every 1200 steps, but the run makes one step a minute.
    write_probe(root, steps_per_s=1.0)
    trainer = FakeTrainer(seconds_per_step=60.0)
    result = train(root, clone, "--steps-finetune", "70", "--stage", "finetune", trainer=trainer)
    assert result.code == 0, result.err + result.log
    times = [T0] + [when for name, _, when in trainer.saves if name == "finetune"]
    assert max(b - a for a, b in zip(times, times[1:], strict=False)) <= timedelta(minutes=30)
    assert trainer.starts_of("finetune")[0] == 0 and len(trainer.starts_of("finetune")) > 1
    # Each restart resumed from the checkpoint the stop saved.
    restarts = trainer.starts_of("finetune")[1:]
    assert (
        restarts == [step for name, step, _ in trainer.saves if name == "finetune"][: len(restarts)]
    )
    assert read(run_dir(root) / "train.json")["steps"]["finetune"] == {"done": 70, "max": 70}


def test_a_killed_trainer_resumes_from_its_newest_checkpoint(
    root: root_mod.Root, hub: Hub, clone: Path
) -> None:
    write_probe(root, steps_per_s=0.002)  # a checkpoint every 2 steps
    trainer = FakeTrainer(kill_at=("finetune", 5))
    killed = train(root, clone, "--steps-finetune", "6", "--steps-distill", "4", trainer=trainer)
    assert killed.code == 4
    assert "exited 1" in killed.err and "the finetune trainer exited -9" in killed.log
    record = read(run_dir(root) / "train.json")
    assert record["status"] == "running"
    assert record["steps"]["finetune"] == {"done": 4, "max": 6}

    again = train(root, clone)
    assert again.code == 2 and "--resume" in again.err

    trainer = FakeTrainer()
    resumed = train(root, clone, "--resume", trainer=trainer)
    assert resumed.code == 0, resumed.err + resumed.log
    assert [(s.name, s.step) for s in trainer.starts] == [("finetune", 4), ("distill", 0)]
    assert (trainer.starts[0].config["max_steps"], trainer.starts[1].config["max_steps"]) == (6, 4)
    record = read(run_dir(root) / "train.json")
    assert record["status"] == "done"
    assert record["steps"] == {"finetune": {"done": 6, "max": 6}, "distill": {"done": 4, "max": 4}}


def test_deadline_saves_a_checkpoint_and_pauses_then_resume_finishes(
    root: root_mod.Root, hub: Hub, clone: Path
) -> None:
    write_probe(root)
    # Whatever the host's zone, the deadline is 10 minutes to 24 h after T0; the trainer's clock
    # jumps 25 h at fine-tune step 3.
    trainer = FakeTrainer(jump_at=("finetune", 3))
    first = train(
        root, clone, "--until", "11:59", "--steps-finetune", "6", "--steps-distill", "4",
        trainer=trainer, host_now=T0,
    )  # fmt: skip
    assert first.code == 0, first.err + first.log
    assert "--deadline-utc" in first.job["args"]
    record = read(run_dir(root) / "train.json")
    assert (record["status"], record["stopped_by"], record["stage"]) == (
        "paused", "deadline", "finetune"
    )  # fmt: skip
    assert [(n, s) for n, s, _ in trainer.saves] == [("finetune", 4)]
    assert record["steps"]["finetune"] == {"done": 4, "max": 6}
    assert "paused (deadline)" in first.out
    assert not (run_dir(root) / "checkpoints" / "finetune" / "STOP").exists()

    trainer = FakeTrainer()
    done = train(root, clone, "--resume", trainer=trainer)
    assert done.code == 0, done.err + done.log
    assert [(s.name, s.step) for s in trainer.starts] == [("finetune", 4), ("distill", 0)]
    assert read(run_dir(root) / "train.json")["status"] == "done"


def test_sigterm_saves_a_checkpoint_and_pauses(root: root_mod.Root, hub: Hub, clone: Path) -> None:
    write_probe(root)
    previous = signal.getsignal(signal.SIGTERM)
    trainer = FakeTrainer(signal_at=("distill", 2))
    result = train(root, clone, "--steps-finetune", "3", "--steps-distill", "9", trainer=trainer)
    assert result.code == 0, result.err + result.log
    record = read(run_dir(root) / "train.json")
    assert (record["status"], record["stopped_by"], record["stage"]) == (
        "paused", "signal", "distill"
    )  # fmt: skip
    assert [(n, s) for n, s, _ in trainer.saves] == [("finetune", 3), ("distill", 3)]
    assert (run_dir(root) / "checkpoints" / "weights" / "checkpoint_00000003").is_dir()
    assert signal.getsignal(signal.SIGTERM) == previous


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="needs time.tzset (POSIX)")
def test_until_reaches_the_job_as_a_utc_deadline(
    root: root_mod.Root, hub: Hub, clone: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TZ", NEW_YORK)
    time.tzset()
    try:
        write_probe(root)
        now = datetime(2026, 11, 24, 3, 30, tzinfo=UTC)  # 22:30 EST
        result = train(
            root, clone, "--until", "01:15", "--steps-finetune", "2", "--steps-distill", "2",
            trainer=FakeTrainer(clock=FakeClock(now)), host_now=now,
        )  # fmt: skip
        assert result.code == 0, result.err + result.log
        assert result.job["args"][-2:] == ["--deadline-utc", "2026-11-24T06:15:00Z"]
        refused = train(root, clone, "--resume", "--until", "22:35", host_now=now)
        assert refused.code == 2 and "10 minutes to 24 h ahead" in refused.err
        assert not refused.docker.argvs
    finally:
        monkeypatch.undo()
        time.tzset()


def test_stage_distill_alone_needs_a_finished_finetune(
    root: root_mod.Root, hub: Hub, clone: Path
) -> None:
    write_probe(root)
    early = train(root, clone, "--stage", "distill", "--steps-finetune", "3")
    assert early.code == 4 and "has not reached step 3" in early.log
    trainer = FakeTrainer()
    first = train(root, clone, "--resume", "--stage", "finetune", trainer=trainer)
    assert first.code == 0, first.err + first.log
    assert [s.name for s in trainer.starts] == ["finetune"]
    assert read(run_dir(root) / "train.json")["status"] == "running"
    trainer = FakeTrainer()
    second = train(root, clone, "--resume", "--stage", "distill", "--steps-distill", "2",
                   trainer=trainer)  # fmt: skip
    assert second.code == 0, second.err + second.log
    assert [s.name for s in trainer.starts] == ["distill"]
    assert read(run_dir(root) / "train.json")["status"] == "done"


def test_train_refuses_differing_tokenizers(root: root_mod.Root, hub: Hub, clone: Path) -> None:
    hub.file(stage.STUDENT_TOKENIZER).write_bytes(b'{"model": "another tokenizer"}')
    write_probe(root)
    trainer = FakeTrainer()
    result = train(root, clone, trainer=trainer)
    assert result.code == 4 and "tokenizer differs" in result.log
    assert not trainer.starts


@pytest.mark.parametrize(
    ("setup", "args", "message"),
    [
        pytest.param(lambda r: None, (), "has no probe", id="no-probe"),
        pytest.param(lambda r: write_probe(r, corpus="c2"), (), "another corpus", id="corpus"),
        pytest.param(
            lambda r: write_probe(r), ("--resume",), "no training to resume", id="resume-nothing"
        ),
        pytest.param(lambda r: write_probe(r), ("--steps-distill", "0"), "at least 1", id="steps"),
        pytest.param(
            lambda r: write_probe(r, save_s=ABSENT), (), f"students/{RUN}/probe.json has no save_s",
            id="no-save-s",
        ),  # fmt: skip
        pytest.param(
            lambda r: write_probe(r, save_s=None), (), f"students/{RUN}/probe.json has no save_s",
            id="null-save-s",
        ),  # fmt: skip
    ],
)
def test_train_refusals_exit_2(
    root: root_mod.Root, hub: Hub, clone: Path, setup: object, args: tuple[str, ...],
    message: str,
) -> None:  # fmt: skip
    setup(root)  # type: ignore[operator]
    result = train(root, clone, *args)
    assert result.code == 2
    assert message in result.err
    assert not result.docker.argvs


@pytest.mark.parametrize(
    ("steps_per_s", "save_s", "grace"),
    [
        # 2 x (1 / 0.025 + 2.0) + 30 = 114, under the floor.
        pytest.param(0.025, 2.0, 120, id="floor"),
        # ceil(2 x (100 + 3.4)) + 30 = 207 + 30.
        pytest.param(0.01, 3.4, 237, id="measured"),
    ],
)
def test_train_job_stop_grace_comes_from_the_probe(
    root: root_mod.Root, hub: Hub, clone: Path, steps_per_s: float, save_s: float, grace: int
) -> None:
    write_probe(root, steps_per_s=steps_per_s, save_s=save_s)
    result = train(root, clone, "--steps-finetune", "1", "--steps-distill", "1")
    assert result.code == 0, result.err + result.log
    assert result.job["stop_timeout_s"] == grace
    argv = result.docker.argvs[0]
    assert argv[argv.index("--stop-timeout") + 1] == str(grace)


def test_probe_job_has_no_stop_grace(root: root_mod.Root, hub: Hub, clone: Path) -> None:
    result = probe(root, clone)
    assert result.code == 0, result.err + result.log
    assert result.job["stop_timeout_s"] is None
    assert "--stop-timeout" not in result.docker.argvs[0]


def test_probe_refuses_a_run_that_started_training(
    root: root_mod.Root, hub: Hub, clone: Path
) -> None:
    write_probe(root)
    assert train(root, clone, "--steps-finetune", "1", "--steps-distill", "1").code == 0
    result = probe(root, clone)
    assert result.code == 2 and "started training" in result.err


# The helpers both sides share.


def test_contract_helpers() -> None:
    assert [student_cmd.accum_for(b) for b in (2, 4, 8, 16, 3)] == [32, 16, 8, 4, 22]
    assert student_cmd.projected_hours(0.5, 28000) == pytest.approx(15.556, abs=1e-3)
    assert student_cmd.checkpoint_every(0.1) == 120
    assert student_cmd.checkpoint_every(0.0001) == 1
    assert student_cmd.vram_gb_for(7000.0) == 7.4
    assert student_cmd.stop_timeout_for(0.025, 2.0) == 120
    assert student_cmd.stop_timeout_for(0.01, 3.4) == 237
    assert student_cmd.stop_timeout_for(0.005, 0.0) == 430
    assert student_cmd.weights_rel(RUN, "checkpoint_00002500") == (
        f"students/{RUN}/checkpoints/weights/checkpoint_00002500/model.safetensors"
    )
    with pytest.raises(student_cmd.InputError):
        student_cmd.weights_rel(RUN, "../x")


def test_runtime_weights_keep_flow_lm_and_mimi_only() -> None:
    state = {"mimi.b": 2, "flow_lm.a": 1}
    assert list(stage.runtime_weights(state)) == ["flow_lm.a", "mimi.b"]
    with pytest.raises(stage.StageError, match="1 tensors outside"):
        stage.runtime_weights({**state, "optim.state.0.exp_avg": 3})
    with pytest.raises(stage.StageError, match="no mimi tensor"):
        stage.runtime_weights({"flow_lm.a": 1})


def test_export_refuses_a_save_that_refreshed_no_export(tmp_path: Path) -> None:
    module = types.SimpleNamespace(save_checkpoint=lambda run_dir, step: None)
    stage.install_export(module, tmp_path / "weights")
    with pytest.raises(stage.StageError, match="refreshed no model.safetensors"):
        module.save_checkpoint(tmp_path, 5)


def test_stop_reads_the_signal_before_the_clock() -> None:
    deadline = datetime(2026, 11, 23, 12, 0, tzinfo=UTC)
    stop = stage.Stop(deadline, lambda: deadline - timedelta(seconds=1))
    assert stop.reason() is None
    stop.on_signal(signal.SIGTERM, None)
    assert stop.reason() == "signal"
    assert stage.Stop(deadline, lambda: deadline).reason() == "deadline"


def test_the_stage_runs_standalone() -> None:
    proc = subprocess.run(  # noqa: S603 - a fixed argv of this interpreter
        [sys.executable, "-m", "scripts.voice_data.stages.student_train", "--data", "/data",
         "train", "--help"],
        capture_output=True, text=True, check=False, cwd=Path(__file__).resolve().parents[3],
    )  # fmt: skip
    assert proc.returncode == 0 and "--deadline-utc" in proc.stdout


# The trainer child: upstream's trainer run in-process from the clone.

# Like upstream's train.py, it binds `exit_training` by name at import and ends through it, at
# the last step and on a stop (`stop_at`: the stop request arrives during that step). A non-finite
# gradient (`nonfinite`) raises `SystemExit` before that step's optimizer step, as upstream does.
TRAIN_PY = """\
import json
import sys

import torch

import pocket_tts
from training.checkpointing import save_checkpoint
from training.distributed import exit_training


def main():
    config = json.loads(open(sys.argv[1], encoding="utf-8").read())
    if config.get("oom"):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")
    adam, muon = torch.optim.Optimizer(), torch.optim.Optimizer()
    for step in range(1, config["steps"] + 1):
        for _ in range(2):
            torch.tensor(0.25 * step).backward()
        if config.get("nonfinite") == step:
            raise SystemExit(f"non-finite gradient at step {step}")
        adam.step()
        muon.step()
        if config.get("stop_at") == step:
            save_checkpoint(config["run_dir"], step)
            exit_training()
        if step % config["ckpt_freq"] == 0:
            save_checkpoint(config["run_dir"], step)
    with open(config["seen"], "w", encoding="utf-8") as handle:
        json.dump({"pocket_tts": pocket_tts.__file__, "argv": sys.argv[1:]}, handle)
    exit_training()


if __name__ == "__main__":
    main()
"""

# Upstream's `exit_training` kills the loader processes, shuts logging down and calls
# `os._exit(0)`; only the hard exit matters here, and the tests replace `os._exit` (`HardExit`).
DISTRIBUTED_PY = """\
import os


def exit_training():
    os._exit(0)
"""

CHECKPOINTING_PY = """\
from pathlib import Path

import safetensors.torch


def save_checkpoint(run_dir, step, ema=None, mimi=None):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"checkpoint_{step:08d}.pt").write_bytes(b"model ema")
    for old in sorted(run_dir.glob("checkpoint_*.pt"))[:-1]:
        old.unlink()
    state = {"flow_lm.w": [float(step)], "mimi.w": [1.0]}
    safetensors.torch.save_file(state, str(run_dir / "model.tmp"))
    (run_dir / "model.tmp").rename(run_dir / "model.safetensors")
"""


def make_torch(free: int, peak: int) -> types.ModuleType:
    """A fake `torch`: CUDA memory figures, a step-post hook registry and scalar tensors."""
    hooks: list[object] = []

    class Handle:
        def __init__(self, fn: object) -> None:
            self.fn = fn

        def remove(self) -> None:
            hooks.remove(self.fn)

    def register(fn: object) -> Handle:
        hooks.append(fn)
        return Handle(fn)

    class Optimizer:
        def step(self) -> None:
            for hook in list(hooks):
                hook(self, (), {})  # type: ignore[operator]

    class Tensor:
        def __init__(self, value: float) -> None:
            self.value = value

        def numel(self) -> int:
            return 1

        def detach(self) -> Tensor:
            return self

        def float(self) -> Tensor:
            return self

        def item(self) -> float:
            return self.value

        def backward(self) -> None:
            pass

    class OutOfMemoryError(RuntimeError):
        pass

    # Real torch's layout (2.6 and 2.8): `torch.optim.optimizer` is imported but `torch.optim`
    # does not bind it, and `torch.optim` does not re-export the hook registration.
    hook_module = types.ModuleType("torch.optim.optimizer")
    hook_module.register_optimizer_step_post_hook = register  # type: ignore[attr-defined]
    optim = types.ModuleType("torch.optim")
    optim.Optimizer = Optimizer  # type: ignore[attr-defined]
    torch = types.ModuleType("torch")
    torch.optim = optim  # type: ignore[attr-defined]
    torch.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        mem_get_info=lambda: (free, 2 * free), max_memory_reserved=lambda: peak,
        OutOfMemoryError=OutOfMemoryError,
    )  # fmt: skip
    torch.Tensor = Tensor  # type: ignore[attr-defined]
    torch.tensor = Tensor  # type: ignore[attr-defined]
    torch.hooks = hooks  # type: ignore[attr-defined]
    torch.hook_module = hook_module  # type: ignore[attr-defined]
    return torch


def use_torch(monkeypatch: pytest.MonkeyPatch, torch: types.ModuleType) -> types.ModuleType:
    """Put a `make_torch` fake in `sys.modules` under the three names real torch puts there.

    The monitor looks up `torch.optim.optimizer` by name, so replacing that entry is what keeps
    a real torch imported earlier in the session from lending the fake its global hook registry.
    """
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.optim", torch.optim)  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules, "torch.optim.optimizer", torch.hook_module  # type: ignore[attr-defined]
    )
    return torch


def test_step_monitor_reaches_the_hook_module_torch_optim_does_not_bind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = use_torch(monkeypatch, make_torch(free=0, peak=0))
    assert not hasattr(torch.optim, "optimizer")
    ticks = iter(range(100))
    monitor = stage.StepMonitor(lambda: float(next(ticks)))
    monitor.install(torch)
    optimizer = torch.optim.Optimizer()
    torch.tensor(0.25).backward()
    torch.tensor(0.5).backward()
    optimizer.step()
    monitor.remove()
    assert monitor.times == [0.0] and monitor.step_losses == [pytest.approx(0.75)]
    assert torch.hooks == []


def sgd_step(torch: types.ModuleType) -> Callable[[], None]:
    """Return one CPU SGD step on the scalar loss 2w, w starting at 1.0 with lr 0.1."""
    weight = torch.nn.Parameter(torch.tensor(1.0))  # type: ignore[attr-defined]
    optimizer = torch.optim.SGD([weight], lr=0.1)  # type: ignore[attr-defined]

    def step() -> None:
        optimizer.zero_grad()
        (weight * 2.0).backward()
        optimizer.step()

    return step


def test_step_monitor_counts_real_torch_steps_and_their_losses() -> None:
    torch = pytest.importorskip("torch")
    step = sgd_step(torch)
    ticks = iter(range(100))
    monitor = stage.StepMonitor(lambda: float(next(ticks)))
    monitor.install(torch)
    try:
        for _ in range(3):
            step()
    finally:
        monitor.remove()
    assert monitor.times == [0.0, 1.0, 2.0]
    # The loss is 2w, and each step moves w down by 0.1 * 2.
    assert monitor.step_losses == [
        pytest.approx(2.0), pytest.approx(1.6), pytest.approx(1.2)
    ]  # fmt: skip


def test_step_monitor_remove_restores_real_torch() -> None:
    torch = pytest.importorskip("torch")
    original = torch.Tensor.backward
    step = sgd_step(torch)
    monitor = stage.StepMonitor(lambda: 0.0)
    monitor.install(torch)
    try:
        step()
    finally:
        monitor.remove()
    assert torch.Tensor.backward is original
    step()
    assert len(monitor.times) == 1 and len(monitor.step_losses) == 1


class HardExit(BaseException):
    """Raised by the `os._exit` stub, so the process upstream would end is the test's."""


@dataclass
class ExitStub:
    """Stands in for `os._exit`: records the code and the metrics file as they were at the call.

    The record is taken at the call because `run_trainer`'s `finally` writes the metrics while
    `HardExit` unwinds, which upstream's real `os._exit` would have skipped.
    """

    metrics: Path
    calls: list[tuple[int, dict | None]] = field(default_factory=list)

    def __call__(self, code: int) -> None:
        self.calls.append((code, read(self.metrics) if self.metrics.is_file() else None))
        raise HardExit(code)


@pytest.fixture
def hard_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ExitStub:
    """Replace `os._exit`; the trainer tests' metrics file is `tmp_path / "metrics.json"`."""
    stub = ExitStub(tmp_path / "metrics.json")
    monkeypatch.setattr(os, "_exit", stub)
    return stub


@pytest.fixture
def trainer_clone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hard_exit: ExitStub
) -> Iterator[Path]:
    """A clone whose trainer is `TRAIN_PY`, a decoy installed `pocket_tts`, and a fake torch.

    It depends on `hard_exit`, so the fake upstream's `os._exit` never ends the test process.
    """
    folder = tmp_path / "trainer-clone"
    for rel, text in {
        "pocket_tts/__init__.py": "",
        "training/__init__.py": "",
        "training/train.py": TRAIN_PY,
        "training/checkpointing.py": CHECKPOINTING_PY,
        "training/distributed.py": DISTRIBUTED_PY,
    }.items():
        path = folder / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    decoy = tmp_path / "site-packages"
    (decoy / "pocket_tts").mkdir(parents=True)
    (decoy / "pocket_tts" / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(sys, "path", [str(decoy), *sys.path])
    use_torch(monkeypatch, make_torch(free=10 * 1024 * MIB, peak=6 * 1024 * MIB))
    ticks = iter(range(10_000))
    monkeypatch.setattr(
        stage, "time", types.SimpleNamespace(perf_counter=lambda: next(ticks) * 0.5)
    )
    names = (
        "pocket_tts", "training", "training.train", "training.checkpointing",
        "training.distributed",
    )  # fmt: skip
    for name in names:
        monkeypatch.delitem(sys.modules, name, raising=False)
    yield folder
    for name in names:
        sys.modules.pop(name, None)


def trainer_config_file(tmp_path: Path, **values: object) -> Path:
    config = {"steps": 12, "ckpt_freq": 4, "run_dir": str(tmp_path / "distill"),
              "seen": str(tmp_path / "seen.json"), **values}  # fmt: skip
    path = tmp_path / "config.yaml"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def trainer_argv(clone: Path, tmp_path: Path, config: Path, *extra: str) -> list[str]:
    """The `trainer` mode's command line, its metrics at `tmp_path / "metrics.json"`."""
    return [
        "--data", str(tmp_path), "--upstream", str(clone), "trainer", "--config", str(config),
        "--metrics", str(tmp_path / "metrics.json"), *extra,
    ]  # fmt: skip


def metrics_at_exit(hard_exit: ExitStub) -> dict:
    """Return the metrics file as it was when upstream's one `os._exit(0)` was called."""
    [(code, doc)] = hard_exit.calls
    assert code == 0
    assert doc is not None, "no metrics file when upstream's exit_training hard-exited"
    return doc


def upstream_exit_is_bound(clone: Path) -> bool:
    """Whether `training.distributed.exit_training` is the clone's own function again."""
    function = sys.modules["training.distributed"].exit_training  # type: ignore[attr-defined]
    return Path(function.__code__.co_filename).resolve().is_relative_to(clone.resolve())


def test_trainer_runs_upstream_from_the_clone_and_measures_it(
    trainer_clone: Path, tmp_path: Path, hard_exit: ExitStub
) -> None:
    config = trainer_config_file(tmp_path)
    weights = tmp_path / "weights"
    with pytest.raises(HardExit):
        stage.main(trainer_argv(trainer_clone, tmp_path, config, "--export-dir", str(weights)))
    seen = read(tmp_path / "seen.json")
    # `pocket_tts` came from the clone, not from the installed copy earlier on the path.
    assert Path(seen["pocket_tts"]).resolve().is_relative_to(trainer_clone.resolve())
    assert seen["argv"] == [str(config)]
    assert sys.path[0] == str(trainer_clone)
    doc = metrics_at_exit(hard_exit)
    assert (doc["free_bytes"], doc["peak_reserved_bytes"]) == (10 * 1024 * MIB, 6 * 1024 * MIB)
    # One timed step per step of the first optimizer; the first five are warm-up.
    assert doc["steps"] == 12 and doc["steps_per_s"] == pytest.approx(2.0)
    # Each step's loss is what `backward` saw, summed over its two micro-batches.
    assert doc["losses"] == [[12, pytest.approx(0.5 * 6.5)]]
    assert doc["oom"] is False
    # Each save left the student's runtime-layout weights; the pruned checkpoints lost theirs.
    assert sorted(p.name for p in weights.iterdir()) == ["checkpoint_00000012"]
    assert read(weights / "checkpoint_00000012" / "model.safetensors") == {
        "flow_lm.w": [12.0], "mimi.w": [1.0]
    }  # fmt: skip
    # The hooks are gone once the trainer returns.
    assert sys.modules["torch"].hooks == []  # type: ignore[attr-defined]


def test_trainer_writes_its_metrics_before_upstreams_hard_exit_at_the_last_step(
    trainer_clone: Path, tmp_path: Path, hard_exit: ExitStub
) -> None:
    config = trainer_config_file(tmp_path)

    with pytest.raises(HardExit):
        stage.main(trainer_argv(trainer_clone, tmp_path, config))

    doc = metrics_at_exit(hard_exit)
    assert (doc["free_bytes"], doc["peak_reserved_bytes"]) == (10 * 1024 * MIB, 6 * 1024 * MIB)
    assert doc["steps"] == 12 and doc["steps_per_s"] == pytest.approx(2.0)
    assert doc["oom"] is False
    assert upstream_exit_is_bound(trainer_clone)


def test_trainer_writes_its_metrics_before_upstreams_hard_exit_on_a_stop(
    trainer_clone: Path, tmp_path: Path, hard_exit: ExitStub
) -> None:
    config = trainer_config_file(tmp_path, stop_at=7)

    with pytest.raises(HardExit):
        stage.main(trainer_argv(trainer_clone, tmp_path, config))

    doc = metrics_at_exit(hard_exit)
    assert (doc["free_bytes"], doc["peak_reserved_bytes"]) == (10 * 1024 * MIB, 6 * 1024 * MIB)
    assert doc["steps"] == 7 and doc["steps_per_s"] == pytest.approx(2.0)
    assert doc["losses"] == [[7, pytest.approx(0.5 * 4.0)]]
    assert doc["oom"] is False
    # The stop saved after its step and ended the trainer there.
    assert stage.newest_step(tmp_path / "distill") == 7
    assert not (tmp_path / "seen.json").exists()
    assert upstream_exit_is_bound(trainer_clone)


def _metrics_path_is_a_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A folder where the metrics file belongs: writing it raises `IsADirectoryError`.
    (tmp_path / "metrics.json").mkdir()


def _metrics_json_raises_value_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(path: Path, payload: object) -> None:
        raise ValueError("not serializable")

    monkeypatch.setattr(stage, "_write_json", refuse)


@pytest.mark.parametrize(
    ("break_write", "error"),
    [
        pytest.param(_metrics_path_is_a_folder, IsADirectoryError, id="os-error"),
        pytest.param(_metrics_json_raises_value_error, ValueError, id="value-error"),
    ],
)
def test_trainer_hard_exits_and_reports_why_when_its_metrics_cannot_be_written(
    trainer_clone: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hard_exit: ExitStub,
    capsys: pytest.CaptureFixture[str],
    break_write: Callable[[Path, pytest.MonkeyPatch], None],
    error: type[Exception],
) -> None:
    break_write(tmp_path, monkeypatch)
    config = trainer_config_file(tmp_path)

    # Upstream's `os._exit` would end the process here, dropping the exception in flight; the
    # stub lets the `finally` run, and its own write fails the same way.
    with pytest.raises(error):
        stage.main(trainer_argv(trainer_clone, tmp_path, config))

    assert hard_exit.calls == [(0, None)]
    # The failure reached stderr, traceback included, before the hard exit.
    err = capsys.readouterr().err
    assert "metrics could not be written" in err
    assert "Traceback (most recent call last)" in err and error.__name__ in err
    assert upstream_exit_is_bound(trainer_clone)


def test_trainer_passes_upstreams_system_exit_through_after_the_metrics(
    trainer_clone: Path, tmp_path: Path, hard_exit: ExitStub
) -> None:
    config = trainer_config_file(tmp_path, nonfinite=9)

    with pytest.raises(SystemExit, match="non-finite gradient at step 9"):
        stage.main(trainer_argv(trainer_clone, tmp_path, config))

    assert hard_exit.calls == []
    doc = read(tmp_path / "metrics.json")
    assert doc["steps"] == 8 and doc["oom"] is False
    assert upstream_exit_is_bound(trainer_clone)


def test_trainer_times_each_checkpoint_save(
    trainer_clone: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hard_exit: ExitStub
) -> None:
    # Saves at steps 4, 8 and 12, each read twice: they take 1.25, 2.5 and 0.5 seconds.
    readings = iter([0.0, 1.25, 10.0, 12.5, 20.0, 20.5])
    monkeypatch.setattr(stage, "save_clock", lambda: next(readings))
    config = trainer_config_file(tmp_path)
    with pytest.raises(HardExit):
        stage.main(trainer_argv(trainer_clone, tmp_path, config))
    assert metrics_at_exit(hard_exit)["save_s"] == pytest.approx(2.5)
    assert next(readings, None) is None


def test_save_timer_records_a_save_that_raised() -> None:
    def save(run_dir: Path, step: int) -> None:
        raise OSError("disk full")

    module = types.SimpleNamespace(save_checkpoint=save)
    readings = iter([5.0, 8.0])
    timer = stage.SaveTimer(lambda: next(readings))
    timer.install(module)
    assert timer.longest() is None
    with pytest.raises(OSError, match="disk full"):
        module.save_checkpoint(Path("x"), 1)
    assert timer.longest() == pytest.approx(3.0)


def test_trainer_reports_out_of_memory(
    trainer_clone: Path, tmp_path: Path, hard_exit: ExitStub
) -> None:
    config = trainer_config_file(tmp_path, oom=True)
    code = stage.main(trainer_argv(trainer_clone, tmp_path, config))
    assert code == stage.CHILD_OOM
    assert hard_exit.calls == []
    doc = read(tmp_path / "metrics.json")
    assert doc["oom"] is True and doc["steps"] == 0 and doc["steps_per_s"] is None
    assert upstream_exit_is_bound(trainer_clone)


def test_trainer_refuses_a_pocket_tts_outside_the_clone(
    trainer_clone: Path, tmp_path: Path
) -> None:
    (trainer_clone / "pocket_tts" / "__init__.py").unlink()
    (trainer_clone / "pocket_tts").rmdir()
    config = trainer_config_file(tmp_path)
    code = stage.main(trainer_argv(trainer_clone, tmp_path, config))
    assert code == 1
