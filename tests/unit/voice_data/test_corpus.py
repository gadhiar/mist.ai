"""`corpus build`: the student's training corpus in the pinned Pocket TTS training layout.

The command runs through the CLI, and its alignment job's stage runs in this process
(`StageDocker`), as `test_teacher.py` does. The stage's aligner is replaced by `FakeAligner`,
which writes upstream's row shape (`words` of `{"word", "start", "end"}`) and leaves out the
rows a test names, as `training/scripts/align_data.py` leaves out what it cannot align. A fake
`huggingface_hub` resolves the aligner's pinned snapshot, so the test sees the folder the job
passes as `--model`.

The canary transcript is in every line's text and every seed transcript; it must reach no
printed line, and no printed line names a clip.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import sys
import types
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts.voice_data import cli, jobs, text_corpus
from scripts.voice_data import root as root_mod
from scripts.voice_data.commands import corpus as corpus_cmd
from scripts.voice_data.commands import teacher as teacher_cmd
from scripts.voice_data.errors import InputError
from scripts.voice_data.records import (
    Prosody,
    SegmentRecord,
    TakeRecord,
    TargetRecord,
    write_json,
    write_jsonl,
)
from scripts.voice_data.stages import corpus_build as stage
from scripts.voice_data.stages import fetch

from .conftest import Canaries, _assert_no_canary
from .test_identity import _Capture

RATE = 24000
VOICE = "anchor-voice"
TEXT = "v1"
RUN = "g1"
TARGET = "t1"
SEED_RUN = "c1"
VERSION = "s1"
IMAGE = "mist-voice-train-pocket:0123456789ab"
CANARY = Canaries()
LINES = tuple(f"line-{n:04d}" for n in range(1, 13))
TWO_SELECTED = "line-0001"
ALIGNER = jobs.load_families()["train-pocket"]["models"]["wav2vec2-base-960h"]
PROSODY = Prosody(f0_mean_hz=200.0, f0_std_hz=4.0, rate_sps=4.0, energy_db=-20.0)
EAR_LINE = text_corpus.load_ear_test()[0]
# Free VRAM the fake nvidia-smi reports: under the family's 16 GB, over the job's 3.
FREE_MIB = "10240\n"


def line_text(line_id: str) -> str:
    return f"{CANARY.transcript}, {line_id.replace('-', ' ')}."


def tone(seconds: float, hz: float = 220.0) -> np.ndarray:
    t = np.arange(int(round(seconds * RATE))) / RATE
    return (0.3 * np.sin(2.0 * np.pi * hz * t)).astype(np.float32)


def write_audio(path: Path, seconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), tone(seconds), RATE, format="FLAC", subtype="PCM_16")


def write_lines(
    root: root_mod.Root,
    texts: dict[str, str] | None = None,
    *,
    text: str = TEXT,
    lines: Sequence[str] = LINES,
) -> None:
    texts = texts or {}
    rows = [{"line_id": i, "state": "neutral", "text": texts.get(i, line_text(i))} for i in lines]
    path = root.path / "text" / text / "lines.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def take_seconds(index: int, take: int) -> float:
    return 2.0 + 0.25 * (index % 3) + 0.1 * take


def write_takes(
    root: root_mod.Root,
    *,
    floor_from: str = "seed",
    run: str = RUN,
    text: str = TEXT,
    lines: Sequence[str] = LINES,
) -> list[TakeRecord]:
    """Write a scored run: per line a selected take, a kept unselected one and a rejected one."""
    takes = []
    for index, line in enumerate(lines):
        for k in range(3):
            seed = teacher_cmd.take_seed(line, 1, k)
            take = teacher_cmd.take_id(line, seed)
            audio = teacher_cmd.take_audio_rel(run, take)
            write_audio(root.path / audio, take_seconds(index, k))
            selected = k == 0 or (line == TWO_SELECTED and k == 1)
            takes.append(
                TakeRecord(
                    take_id=take, line_id=line, state="neutral", teacher="omnivoice", seed=seed,
                    round=1, audio=audio, sim={"wavlm-sv": 0.9, "ecapa": 0.8},
                    cer=0.0 if k < 2 else 0.2, prosody=PROSODY, arousal=0.4, kept=k < 2,
                    selected=selected, reasons=() if k < 2 else ("cer",),
                )  # fmt: skip
            )
    write_jsonl(root, f"{teacher_cmd.run_rel(run)}/takes.jsonl", takes)
    floor: dict[str, object] = (
        {"from": "seed", "target": TARGET, "values": {"wavlm-sv": 0.5, "ecapa": 0.5},
         "centroid_sha256": {"wavlm-sv": "a" * 64, "ecapa": "b" * 64}}
        if floor_from == "seed"
        else {"from": "round-1-takes", "target": None, "percentile": 10.0}
    )  # fmt: skip
    plan = {
        "schema": 1, "run": run, "teacher": "omnivoice", "state": "neutral",
        "lines_file": f"text/{text}/lines.jsonl", "anchor": {"voice": VOICE}, "embedders": [],
        "cer_max": 0.03, "floor": floor, "centroids": None, "profile": None, "spread": None,
    }  # fmt: skip
    write_json(root, f"{teacher_cmd.run_rel(run)}/score-plan.json", plan)
    return takes


def seg_id(k: int) -> str:
    return f"seg-{k + 1:06d}"


def seg_seconds(k: int) -> float:
    return 3.0 + 0.1 * k


# Seed segments: four trusted, one whose two ASRs disagree, one dropped, one speaking an
# ear-test line.
TRUSTED = (0, 1, 2, 3)
DISAGREE = 4
DROPPED = 5
EAR = 6


ONE_SOURCE = "src-0001"


def segment(k: int, source: str = ONE_SOURCE) -> SegmentRecord:
    kept = k != DROPPED
    text = EAR_LINE if k == EAR else CANARY.transcript
    reasons = ("asr_disagree",) if k == DISAGREE else () if kept else ("not-her",)
    return SegmentRecord(
        segment_id=seg_id(k), source_id=source, start_s=10.0 * k,
        end_s=10.0 * k + seg_seconds(k), channel="FC", channel_reason=None, separator="none",
        audio=f"seed/runs/{SEED_RUN}/audio/{seg_id(k)}.flac" if kept else "",
        diar_cluster="spk-01", cluster="primary", flag_group=None,
        sim={"wavlm-sv": 0.9, "ecapa": 0.8}, overlap_ms=0, snr_db=30.0, c50_db=40.0,
        transcripts={"whisperx": text, "crisper": text}, asr_agreement_cer=0.0,
        prosody=PROSODY, arousal=0.4, state_proposed="neutral",
        state="neutral" if kept else None, set="short", kept=kept, reasons=reasons,
    )  # fmt: skip


def write_seed(
    root: root_mod.Root, *, extra: int = 0, source_of: Callable[[int], str] = lambda k: ONE_SOURCE
) -> list[SegmentRecord]:
    """Write the seed: the seven segments above plus `extra` trusted ones, each in `source_of(k)`."""
    records = [segment(k, source_of(k)) for k in range(7 + extra)]
    for k, record in enumerate(records):
        if record.audio:
            write_audio(root.path / record.audio, seg_seconds(k))
    write_jsonl(root, f"seed/runs/{SEED_RUN}/segments.jsonl", records)
    run = root.path / "seed" / "runs" / SEED_RUN
    plan = {"schema": 1, "run": SEED_RUN, "from": "r1", "target": TARGET, "thresholds": {},
            "labels": None}  # fmt: skip
    (run / "clean-plan.json").write_text(json.dumps(plan), encoding="utf-8")
    refs = {"schema": 1, "run": SEED_RUN, "target": TARGET, "medians": {}, "states": {}}
    (run / "refs.json").write_text(json.dumps(refs), encoding="utf-8")
    target = TargetRecord(
        schema=1, target_id=TARGET, voice_id=VOICE, mode="seed", seed_run=SEED_RUN,
        identity=None, embedders={}, centroids={}, distribution={}, windows={}, mode_check=(),
        states={},
    )  # fmt: skip
    write_json(root, f"targets/{TARGET}/target.json", target)
    return records


def set_gates(root: root_mod.Root, **values: float) -> None:
    gates = {"corpus.teacher_hours_min": 0.001, "corpus.seed_share_max": 0.9}
    gates.update({f"corpus.{k}": v for k, v in values.items()})
    (root.path / "config" / "gates.json").write_text(json.dumps(gates), encoding="utf-8")


@pytest.fixture
def root(private_root: root_mod.Root) -> root_mod.Root:
    write_lines(private_root)
    write_takes(private_root)
    write_seed(private_root)
    set_gates(private_root)
    return private_root


# The aligner's model folder and the aligner.


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, object]:
    """A fake `huggingface_hub` whose cache holds the aligner's snapshot at its pinned revision."""
    folder = (
        tmp_path / "hf" / "hub" / "models--facebook--wav2vec2-base-960h" / "snapshots"
        / ALIGNER["revision"]
    )  # fmt: skip
    folder.mkdir(parents=True)
    seen: dict[str, object] = {"folder": folder, "calls": []}

    def snapshot_download(
        repo_id: str, *, revision: str | None = None, repo_type: str | None = None,
        local_files_only: bool = False, **kwargs: object,
    ) -> str:  # fmt: skip
        seen["calls"].append((repo_id, revision, repo_type, local_files_only))  # type: ignore[union-attr]
        return str(folder)

    module = types.ModuleType("huggingface_hub")
    module.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return seen


@dataclass
class FakeAligner:
    """Upstream's aligner in shape: the same rows with `words`, unalignable rows left out."""

    skip: set[str] = field(default_factory=set)
    calls: list[list[str]] = field(default_factory=list)
    inputs: list[dict] = field(default_factory=list)

    def __call__(self, source: Path, dest: Path, options: Sequence[str]) -> None:
        self.calls.append(list(options))
        rows = [json.loads(r) for r in source.read_text(encoding="utf-8").splitlines() if r]
        self.inputs = rows
        out = []
        for row in rows:
            assert set(row) == {"path", "duration", "transcript"}
            assert Path(row["path"]).is_file()
            if Path(row["path"]).stem in self.skip:
                continue
            words = row["transcript"].split()
            step = row["duration"] / len(words)
            timed = [
                {"word": w, "start": round(i * step, 3), "end": round((i + 1) * step, 3)}
                for i, w in enumerate(words)
            ]
            out.append({**row, "words": timed})
        dest.write_text("".join(json.dumps(r) + "\n" for r in out), encoding="utf-8")


class StageDocker:
    """A `DockerOps` whose `run_logged` runs the job's stage in this process."""

    def __init__(self, root: root_mod.Root, aligner: FakeAligner) -> None:
        self.root = root
        self.aligner = aligner
        self.argvs: list[list[str]] = []

    def capture(self, argv: Sequence[str]) -> str:
        if list(argv[:3]) == ["docker", "ps", "-q"]:
            return ""
        assert "nvidia-smi" in argv
        return FREE_MIB

    def check(self, argv: Sequence[str]) -> None:
        raise AssertionError("corpus build builds no image")

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
            mp.setattr(module_obj, "align", self.aligner)
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


def build(
    root: root_mod.Root,
    *args: str,
    aligner: FakeAligner | None = None,
    version: str = VERSION,
    runs: Sequence[str] = (RUN,),
) -> Result:
    docker = StageDocker(root, aligner or FakeAligner())
    out, err = _Capture(), _Capture()
    takes = [arg for run in runs for arg in ("--takes", run)]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(corpus_cmd, "make_docker", lambda: docker)
        mp.setattr(corpus_cmd, "image_tag", lambda family: IMAGE)
        mp.delenv(jobs.GPU_PEERS_ENV, raising=False)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(
                ["--root", str(root.path), "corpus", "build", *takes, "--version", version,
                 *args]
            )  # fmt: skip
    result = Result(code, out.getvalue(), err.getvalue(), docker)
    _assert_no_canary(result.out + result.err)
    assert ".wav" not in result.out + result.err and ".flac" not in result.out + result.err
    return result


def corpus_dir(root: root_mod.Root) -> Path:
    return root.path / "corpus" / VERSION


def rows_of(root: root_mod.Root, name: str) -> list[dict]:
    text = (corpus_dir(root) / name).read_text(encoding="utf-8")
    return [json.loads(r) for r in text.splitlines() if r]


def manifest_of(root: root_mod.Root) -> dict:
    return json.loads((corpus_dir(root) / "manifest.json").read_text(encoding="utf-8"))


def selected_takes(takes: Sequence[TakeRecord]) -> list[TakeRecord]:
    return [t for t in takes if t.selected]


def seconds_of(root: root_mod.Root, rel: str) -> float:
    info = sf.info(str(root.path / rel))
    return info.frames / info.samplerate


def by_path(root: root_mod.Root, rels: Sequence[str]) -> dict[str, str]:
    return {str(root.path / rel): rel for rel in rels}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# The corpus and its layout.


def test_build_writes_the_upstream_training_layout(root: root_mod.Root, hub: dict) -> None:
    takes = selected_takes(write_takes(root))
    result = build(root)
    assert result.code == 0, result.err
    train, valid = rows_of(root, "train_aligned.jsonl"), rows_of(root, "valid_aligned.jsonl")
    for row in train + valid:
        assert set(row) == {"path", "duration", "transcript", "words"}
        assert row["words"] and all(set(w) == {"word", "start", "end"} for w in row["words"])
    take_paths = by_path(root, [t.audio for t in takes])
    teacher_rows = [r for r in train + valid if r["path"] in take_paths]
    assert sorted(r["path"] for r in teacher_rows) == sorted(take_paths)
    texts = {t.audio: line_text(t.line_id) for t in takes}
    for row in teacher_rows:
        rel = take_paths[row["path"]]
        assert row["transcript"] == texts[rel]
        assert row["duration"] == pytest.approx(seconds_of(root, rel), abs=1e-6)
    seed_paths = by_path(root, [segment(k).audio for k in TRUSTED])
    seed_rows = [r for r in train if r["path"] in seed_paths]
    assert sorted({r["path"] for r in seed_rows}) == sorted(seed_paths)
    for path in seed_paths:
        assert sum(r["path"] == path for r in seed_rows) == 3
        assert not any(r["path"] == path for r in valid)
    assert {r["transcript"] for r in seed_rows} == {CANARY.transcript}
    assert len(train) + len(valid) == len(teacher_rows) + len(seed_rows)


def test_alignment_is_one_gpu_job_on_the_pinned_aligner_snapshot(
    root: root_mod.Root, hub: dict
) -> None:
    aligner = FakeAligner()
    result = build(root, aligner=aligner)
    assert result.code == 0, result.err
    job = result.job
    assert (job["family"], job["module"], job["gpu"], job["network"]) == (
        "train-pocket", "scripts.voice_data.stages.corpus_build", True, False
    )  # fmt: skip
    assert job["vram_gb"] == 3
    assert job["args"] == ["--version", VERSION, "--device", "cuda", "--batch-size", "8"]
    assert [i["rel"] for i in job["inputs"]] == [
        f"text/{TEXT}/lines.jsonl", f"teachers/omnivoice/{RUN}", f"seed/runs/{SEED_RUN}"
    ]  # fmt: skip
    assert job["outputs"] == f"corpus/{VERSION}"
    argv = result.docker.argvs[0]
    assert argv[argv.index("--gpus") + 1] == "all"
    assert argv[argv.index("--network") + 1] == "none"
    # One alignment pass over every candidate, the model resolved from its pinned snapshot.
    assert aligner.calls == [
        ["--model", str(hub["folder"]), "--device", "cuda", "--batch-size", "8"]
    ]
    assert hub["calls"] == [
        ("facebook/wav2vec2-base-960h", "22aad52d435eb6dbaf354bdad9b0da84ce7d6156", "model", True)
    ]


def test_the_family_vram_estimate_would_refuse_the_job(root: root_mod.Root) -> None:
    docker = StageDocker(root, FakeAligner())
    executor = jobs.LocalDocker(root, docker, jobs.load_families())
    spec = jobs.JobSpec(
        job_id="job-x", family="train-pocket", image=IMAGE, module=corpus_cmd.MODULE, args=(),
        inputs=(), outputs="corpus/x", gpu=True, network=False,
    )  # fmt: skip
    with pytest.raises(jobs.PreflightError):
        executor.preflight(spec)
    executor.preflight(replace(spec, vram_gb=corpus_cmd.VRAM_GB))


def test_unaligned_rows_are_dropped_and_counted(root: root_mod.Root, hub: dict) -> None:
    takes = selected_takes(write_takes(root))
    lost_take = next(t for t in takes if t.line_id == "line-0005")
    lost_seed = segment(1).audio
    aligner = FakeAligner(skip={Path(lost_take.audio).stem, Path(lost_seed).stem})
    result = build(root, aligner=aligner)
    assert result.code == 0, result.err
    paths = {r["path"] for r in rows_of(root, "train_aligned.jsonl")}
    paths |= {r["path"] for r in rows_of(root, "valid_aligned.jsonl")}
    assert str(root.path / lost_take.audio) not in paths
    assert str(root.path / lost_seed) not in paths
    assert manifest_of(root)["unaligned"] == {"teacher": 1, "seed": 1}
    assert "unaligned: teacher 1, seed 1" in result.out


def test_validation_holds_out_whole_lines(root: root_mod.Root, hub: dict) -> None:
    set_gates(root, val_fraction=0.25)
    takes = selected_takes(write_takes(root))
    result = build(root)
    assert result.code == 0, result.err
    line_of = {str(root.path / t.audio): t.line_id for t in takes}
    train = {
        line_of[r["path"]] for r in rows_of(root, "train_aligned.jsonl") if r["path"] in line_of
    }
    valid = {line_of[r["path"]] for r in rows_of(root, "valid_aligned.jsonl")}
    assert len(valid) == 3 and train.isdisjoint(valid) and train | valid == set(LINES)
    manifest = manifest_of(root)
    assert manifest["val_fraction"] == 0.25
    assert manifest["valid_lines"] == 3


def test_holdout_is_stable_and_keeps_a_line_in_each_split() -> None:
    lines = [f"l{n}" for n in range(100)]
    held = corpus_cmd.holdout_lines(lines, 0.02, seed=0)
    assert len(held) == 2
    assert held == corpus_cmd.holdout_lines(list(reversed(lines)), 0.02, seed=0)
    assert len(corpus_cmd.holdout_lines(lines[:10], 0.02, seed=0)) == 1
    assert corpus_cmd.holdout_lines(lines[:1], 0.02, seed=0) == set()
    assert corpus_cmd.holdout_lines(lines[:10], 0.0, seed=0) == set()
    assert len(corpus_cmd.holdout_lines(lines[:3], 0.9, seed=0)) == 2


def test_the_seed_orders_the_line_hold_out_and_the_seed_cap() -> None:
    lines = [f"l{n}" for n in range(100)]
    first = corpus_cmd.holdout_lines(lines, 0.1, seed=1)
    assert first == corpus_cmd.holdout_lines(lines, 0.1, seed=1)
    assert first != corpus_cmd.holdout_lines(lines, 0.1, seed=2)
    segments = [(f"s{n}", 1.0) for n in range(20)]
    kept = corpus_cmd.cap_seed(segments, 14.0, 0.3, 3, seed=1)
    assert len(kept) == 2 and kept == corpus_cmd.cap_seed(segments[::-1], 14.0, 0.3, 3, seed=1)
    assert kept != corpus_cmd.cap_seed(segments, 14.0, 0.3, 3, seed=2)
    assert corpus_cmd.cap_seed(segments, 14.0, 1.0, 3, seed=1) != corpus_cmd.cap_seed(
        segments, 14.0, 1.0, 3, seed=2
    )


def test_the_seed_is_recorded_and_its_splits_reproduce(root: root_mod.Root, hub: dict) -> None:
    set_gates(root, val_fraction=0.25, seed_share_max=0.3)
    assert build(root).code == 0
    assert manifest_of(root)["random_seed"] == 0
    plan = json.loads((corpus_dir(root) / "corpus-plan.json").read_text(encoding="utf-8"))
    assert plan["random_seed"] == 0
    names = ("train_aligned.jsonl", "valid_aligned.jsonl")
    default = {n: (corpus_dir(root) / n).read_bytes() for n in names}
    assert build(root, version="s2").code == 0
    again = root.path / "corpus" / "s2"
    assert {n: (again / n).read_bytes() for n in names} == default
    result = build(root, "--seed", "7", version="s3")
    assert result.code == 0, result.err
    other = root.path / "corpus" / "s3"
    assert json.loads((other / "manifest.json").read_text(encoding="utf-8"))["random_seed"] == 7
    assert (other / "valid_aligned.jsonl").read_bytes() != default["valid_aligned.jsonl"]
    assert (other / "train_aligned.jsonl").read_bytes() != default["train_aligned.jsonl"]
    assert build(root, "--seed", "7", version="s4").code == 0
    for name in names:
        assert (root.path / "corpus" / "s4" / name).read_bytes() == (other / name).read_bytes()


def edit_plan(root: root_mod.Root, change: dict) -> None:
    path = corpus_dir(root) / "corpus-plan.json"
    plan = json.loads(path.read_text(encoding="utf-8"))
    plan.update(change)
    path.write_text(json.dumps({k: v for k, v in plan.items() if v is not None}), encoding="utf-8")


@pytest.mark.parametrize("value", [None, "7", True])
def test_the_stage_refuses_a_plan_without_an_integer_seed(
    root: root_mod.Root, hub: dict, value: object
) -> None:
    assert build(root).code == 0
    edit_plan(root, {"random_seed": value})
    with pytest.raises(stage.StageError, match="random_seed"):
        stage.run(root.path, VERSION, "cpu", 8)


def test_a_seed_that_is_not_an_integer_is_refused(root: root_mod.Root, hub: dict) -> None:
    with pytest.raises(SystemExit) as exc:
        build(root, "--seed", "x")
    assert exc.value.code == 2
    assert not corpus_dir(root).exists()


def test_the_seed_share_is_capped_after_oversampling(root: root_mod.Root, hub: dict) -> None:
    set_gates(root, seed_share_max=0.3)
    result = build(root)
    assert result.code == 0, result.err
    manifest = manifest_of(root)
    train = rows_of(root, "train_aligned.jsonl")
    seed_paths = by_path(root, [segment(k).audio for k in TRUSTED])
    kept = {r["path"] for r in train if r["path"] in seed_paths}
    assert 0 < len(kept) < len(TRUSTED)
    teacher_s = sum(r["duration"] for r in train if r["path"] not in seed_paths)
    seed_s = sum(r["duration"] for r in train if r["path"] in seed_paths)
    assert seed_s / (teacher_s + seed_s) <= 0.3
    for path in set(seed_paths) - kept:  # every segment left out would break the cap
        extra = 3 * seconds_of(root, seed_paths[path])
        assert (seed_s + extra) / (teacher_s + seed_s + extra) > 0.3
    assert manifest["seed_hours"] == pytest.approx(seed_s / 3600, abs=1e-6)
    assert manifest["oversample"] == 3
    assert manifest["seed_share_max"] == 0.3
    assert manifest["seed_left_out"]["share_cap"] == len(TRUSTED) - len(kept)


def test_cap_seed_fills_greedily_in_a_stable_order() -> None:
    segments = [("a", 1.0), ("b", 1.0), ("c", 1.0), ("d", 1.0)]
    kept = corpus_cmd.cap_seed(segments, teacher_train_s=14.0, share_max=0.3, oversample=3, seed=0)
    assert len(kept) == 2  # 3 x 2 s of 20 s is 0.3; a third would be 9 of 23
    assert kept == corpus_cmd.cap_seed(segments[::-1], 14.0, 0.3, 3, seed=0)
    assert corpus_cmd.cap_seed(segments, 14.0, 0.0, 3, seed=0) == []
    assert len(corpus_cmd.cap_seed(segments, 14.0, 1.0, 3, seed=0)) == 4


def test_refuses_below_teacher_hours_min_counted_after_alignment(
    root: root_mod.Root, hub: dict
) -> None:
    takes = selected_takes(write_takes(root))
    total = sum(seconds_of(root, t.audio) for t in takes)
    lost = [t for t in takes if t.line_id in ("line-0002", "line-0003")]
    lost_s = sum(seconds_of(root, t.audio) for t in lost)
    set_gates(root, teacher_hours_min=(total - lost_s / 2) / 3600)
    aligner = FakeAligner(skip={Path(t.audio).stem for t in lost})
    result = build(root, aligner=aligner)
    assert result.code == 3
    assert "corpus.teacher_hours_min" in result.err
    assert not (corpus_dir(root) / "manifest.json").exists()
    report = json.loads((corpus_dir(root) / "report.json").read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert report["teacher_hours"] == pytest.approx((total - lost_s) / 3600, abs=1e-6)
    # The same takes all aligned pass, and the refused version can be built again.
    assert build(root).code == 0


def test_manifest_records_hours_shares_and_file_hashes(root: root_mod.Root, hub: dict) -> None:
    takes = selected_takes(write_takes(root))
    result = build(root)
    assert result.code == 0, result.err
    manifest = manifest_of(root)
    teacher_s = sum(seconds_of(root, t.audio) for t in takes)
    seed_s = 3 * sum(seg_seconds(k) for k in TRUSTED)
    assert manifest["teacher_hours"] == pytest.approx(teacher_s / 3600, abs=1e-6)
    assert manifest["seed_hours"] == pytest.approx(seed_s / 3600, abs=1e-6)
    assert manifest["oversample"] == 3
    assert manifest["val_fraction"] == 0.02
    assert manifest["unaligned"] == {"teacher": 0, "seed": 0}
    assert manifest["schema"] == 2 and "takes_run" not in manifest
    assert manifest["takes_runs"] == [RUN] and manifest["seed_run"] == SEED_RUN
    assert manifest["teacher_hours_by_run"] == {RUN: manifest["teacher_hours"]}
    assert manifest["aligner"] == {
        "repository": "facebook/wav2vec2-base-960h", "revision": ALIGNER["revision"],
        "device": "cuda", "batch_size": 8,
    }  # fmt: skip
    files = {f["rel"]: f["sha256"] for f in manifest["files"]}
    assert set(files) == {f"corpus/{VERSION}/train_aligned.jsonl",
                          f"corpus/{VERSION}/valid_aligned.jsonl"}  # fmt: skip
    for rel, digest in files.items():
        assert sha256(root.path / rel) == digest
    assert manifest["seed_left_out"] == {"untrusted_text": 1, "ear_test": 1, "share_cap": 0}
    assert f"corpus build {VERSION}" in result.out


def test_ear_test_lines_count_zero(root: root_mod.Root, hub: dict) -> None:
    result = build(root)
    assert result.code == 0, result.err
    hold_out = text_corpus.HoldOut(text_corpus.load_ear_test())
    rows = rows_of(root, "train_aligned.jsonl") + rows_of(root, "valid_aligned.jsonl")
    assert sum(hold_out.holds(r["transcript"]) for r in rows) == 0
    assert manifest_of(root)["ear_test_lines"] == 0
    assert str(root.path / segment(EAR).audio) not in {r["path"] for r in rows}


def test_a_take_of_an_ear_test_line_is_refused(root: root_mod.Root, hub: dict) -> None:
    write_lines(root, {"line-0004": EAR_LINE})
    result = build(root)
    assert result.code == 2
    assert "ear-test" in result.err
    assert not result.docker.argvs


def test_untrusted_and_dropped_seed_segments_stay_out(root: root_mod.Root, hub: dict) -> None:
    aligner = FakeAligner()
    assert build(root, aligner=aligner).code == 0
    aligned = {row["path"] for row in aligner.inputs}
    for k in (DISAGREE, EAR):
        assert str(root.path / segment(k).audio) not in aligned
    assert not any(seg_id(DROPPED) in p for p in aligned)


def test_takes_scored_without_a_seed_target_build_a_teacher_only_corpus(
    root: root_mod.Root, hub: dict
) -> None:
    takes = selected_takes(write_takes(root, floor_from="round-1-takes"))
    result = build(root)
    assert result.code == 0, result.err
    manifest = manifest_of(root)
    assert manifest["seed_run"] is None and manifest["seed_hours"] == 0
    rows = rows_of(root, "train_aligned.jsonl") + rows_of(root, "valid_aligned.jsonl")
    assert sorted(r["path"] for r in rows) == sorted(str(root.path / t.audio) for t in takes)
    assert [i["rel"] for i in result.job["inputs"]] == [
        f"text/{TEXT}/lines.jsonl", f"teachers/omnivoice/{RUN}"
    ]  # fmt: skip


def test_refusals_exit_2(root: root_mod.Root, hub: dict) -> None:
    assert build(root, version="../x").code == 2
    assert build(root).code == 0
    again = build(root)
    assert again.code == 2 and "exists" in again.err
    other = root.path / "teachers" / "omnivoice" / RUN / "score-plan.json"
    other.unlink()
    assert build(root, version="s2").code == 2
    write_takes(root)
    unselected = [
        TakeRecord.from_json({**t.to_json(), "selected": False}) for t in write_takes(root)
    ]
    write_jsonl(root, f"{teacher_cmd.run_rel(RUN)}/takes.jsonl", unselected)
    result = build(root, version="s3")
    assert result.code == 2 and "selected" in result.err


def test_missing_run_is_refused(private_root: root_mod.Root, hub: dict) -> None:
    assert build(private_root).code == 2


# The seed's hold-out by whole source.

FOUR_SOURCES = tuple(f"src-{n:04d}" for n in range(1, 5))


def four_sources(k: int) -> str:
    return FOUR_SOURCES[k % 4]


def trusted_ks(extra: int) -> list[int]:
    return [*TRUSTED, *range(7, 7 + extra)]


def test_holdout_sources_takes_whole_sources_to_the_share() -> None:
    even = {f"s{n}": 10.0 for n in range(4)}
    one = corpus_cmd.holdout_sources(even, 0.15, seed=0)
    assert len(one.sources) == 1 and one.reason is None  # 10 s of 40 reaches 6 s
    two = corpus_cmd.holdout_sources(even, 0.5, seed=0)
    assert len(two.sources) == 2  # 20 s of 40 reaches 20 s
    assert one.sources < two.sources  # the same seeded order, taken further
    assert len(corpus_cmd.holdout_sources(even, 0.99, seed=0).sources) == 3  # one stays
    assert len(corpus_cmd.holdout_sources(even, 0.001, seed=0).sources) == 1  # one is held
    pair = corpus_cmd.holdout_sources({"a": 1.0, "b": 100.0}, 0.15, seed=0)
    assert len(pair.sources) == 1 and pair.reason is None
    uneven = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0, "e": 96.0}
    for seed in range(20):
        held = corpus_cmd.holdout_sources(uneven, 0.15, seed=seed).sources
        seconds = sum(uneven[s] for s in held)
        assert 1 <= len(held) <= 4
        # Taken in order until the share is reached: the last one taken was still needed.
        assert seconds >= 15.0 or len(held) == 4
        assert min(seconds - uneven[s] for s in held) < 15.0


def test_holdout_sources_is_seeded_and_reproducible() -> None:
    sources = {f"source-{n}": 5.0 for n in range(20)}
    first = corpus_cmd.holdout_sources(sources, 0.15, seed=1)
    assert len(first.sources) == 3
    assert first == corpus_cmd.holdout_sources(dict(reversed(sources.items())), 0.15, seed=1)
    assert first.sources != corpus_cmd.holdout_sources(sources, 0.15, seed=2).sources


@pytest.mark.parametrize(
    "sources", [{}, {"src-only": 30.0}], ids=["no trusted segment", "one source"]
)
def test_holdout_sources_holds_none_out_below_two_sources(sources: dict[str, float]) -> None:
    held = corpus_cmd.holdout_sources(sources, 0.15, seed=0)
    assert held.sources == frozenset()
    assert held.reason and "src" not in held.reason


def test_held_out_sources_have_no_training_row(root: root_mod.Root, hub: dict) -> None:
    records = write_seed(root, extra=8, source_of=four_sources)
    trusted = {str(root.path / records[k].audio): records[k] for k in trusted_ks(8)}
    result = build(root)
    assert result.code == 0, result.err
    train, valid = rows_of(root, "train_aligned.jsonl"), rows_of(root, "valid_aligned.jsonl")
    seed_valid = [r for r in valid if r["path"] in trusted]
    seed_train = [r for r in train if r["path"] in trusted]
    held = {trusted[r["path"]].source_id for r in seed_valid}
    kept = {trusted[r["path"]].source_id for r in seed_train}
    assert held and kept and held.isdisjoint(kept) and held | kept == set(FOUR_SOURCES)
    # Whole sources, written once each, ahead of every teacher validation row.
    assert sorted(r["path"] for r in seed_valid) == sorted(
        path for path, record in trusted.items() if record.source_id in held
    )
    assert valid[: len(seed_valid)] == seed_valid
    assert all(r["path"] not in trusted for r in valid[len(seed_valid) :])
    seconds = dict.fromkeys(FOUR_SOURCES, 0.0)
    for record in trusted.values():
        seconds[record.source_id] += seconds_of(root, record.audio)
    expected = corpus_cmd.holdout_sources(seconds, 0.15, seed=0)
    assert held == expected.sources
    manifest = manifest_of(root)
    assert manifest["seed_val_share"] == 0.15
    assert manifest["seed_val_sources"] == len(held)
    assert manifest["seed_val_segments"] == len(seed_valid)
    held_s = sum(r["duration"] for r in seed_valid)
    assert manifest["seed_val_hours"] == pytest.approx(held_s / 3600, abs=1e-6)
    assert manifest["seed_val_reason"] is None
    # The share cap and the seed hours count the training seed only.
    assert manifest["seed_hours"] == pytest.approx(
        sum(r["duration"] for r in seed_train) / 3600, abs=1e-6
    )
    assert manifest["seed_segments"] == len({r["path"] for r in seed_train})
    hours = f"{manifest['seed_val_hours']:.3f}"
    assert (
        f"seed validation: {len(held)} sources held out whole, {len(seed_valid)} segments, "
        f"{hours} h" in result.out
    )
    assert "src-" not in result.out + result.err


def test_two_seeds_hold_out_different_sources(root: root_mod.Root, hub: dict) -> None:
    records = write_seed(root, extra=8, source_of=four_sources)
    source_of_path = {str(root.path / r.audio): r.source_id for r in records if r.audio}

    def held(version: str, *args: str) -> set[str]:
        assert build(root, *args, version=version).code == 0
        valid = (root.path / "corpus" / version / "valid_aligned.jsonl").read_text("utf-8")
        rows = [json.loads(r) for r in valid.splitlines() if r]
        return {source_of_path[r["path"]] for r in rows if r["path"] in source_of_path}

    first = held("s1")
    assert held("s2") == first
    assert held("s3", "--seed", "3") != first


def test_one_seed_source_is_kept_for_training(root: root_mod.Root, hub: dict) -> None:
    result = build(root)
    assert result.code == 0, result.err
    manifest = manifest_of(root)
    assert (manifest["seed_val_sources"], manifest["seed_val_segments"]) == (0, 0)
    assert manifest["seed_val_hours"] == 0
    assert manifest["seed_val_reason"] == "one seed source, which training keeps"
    assert "seed validation: none held out (one seed source, which training keeps)" in result.out
    seed_paths = {str(root.path / segment(k).audio) for k in TRUSTED}
    assert not any(r["path"] in seed_paths for r in rows_of(root, "valid_aligned.jsonl"))


def test_dropped_segments_never_count_toward_the_hold_out(root: root_mod.Root, hub: dict) -> None:
    # The untrusted, dropped and ear-test segments are a second source of their own.
    write_seed(
        root, source_of=lambda k: "src-0009" if k in (DISAGREE, DROPPED, EAR) else ONE_SOURCE
    )
    result = build(root)
    assert result.code == 0, result.err
    manifest = manifest_of(root)
    assert manifest["seed_val_sources"] == 0
    assert manifest["seed_val_reason"] == "one seed source, which training keeps"


def test_a_teacher_only_corpus_holds_out_no_seed(root: root_mod.Root, hub: dict) -> None:
    write_takes(root, floor_from="round-1-takes")
    assert build(root).code == 0
    manifest = manifest_of(root)
    assert manifest["seed_val_sources"] == 0
    assert manifest["seed_val_reason"] == "no trusted seed segment"


def test_plan_segments_carry_their_source(root: root_mod.Root, hub: dict) -> None:
    records = write_seed(root, extra=8, source_of=four_sources)
    assert build(root).code == 0
    plan = json.loads((corpus_dir(root) / "corpus-plan.json").read_text(encoding="utf-8"))
    sources = {r.segment_id: r.source_id for r in records}
    assert plan["segments"]
    for entry in plan["segments"]:
        assert set(entry) == {"segment_id", "source_id", "audio"}
        assert entry["source_id"] == sources[entry["segment_id"]]
    assert plan["seed_val_share"] == 0.15


@pytest.mark.parametrize("source", [None, "", "src-0002"])
def test_the_stage_refuses_a_plan_segment_without_its_source(
    root: root_mod.Root, hub: dict, source: object
) -> None:
    assert build(root).code == 0
    path = corpus_dir(root) / "corpus-plan.json"
    plan = json.loads(path.read_text(encoding="utf-8"))
    first = plan["segments"][0]
    first.pop("source_id")
    if source is not None:
        first["source_id"] = source
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(stage.StageError, match=first["segment_id"]):
        stage.run(root.path, VERSION, "cpu", 8)


@pytest.mark.parametrize("value", [None, "0.15", True])
def test_the_stage_refuses_a_plan_without_a_seed_val_share(
    root: root_mod.Root, hub: dict, value: object
) -> None:
    assert build(root).code == 0
    edit_plan(root, {"seed_val_share": value})
    with pytest.raises(stage.StageError, match="seed_val_share"):
        stage.run(root.path, VERSION, "cpu", 8)


# A corpus from several teacher runs.

RUN2 = "g2"
TEXT2 = "v2"
LINES2 = tuple(f"line-{n:04d}" for n in range(101, 109))
THRESHOLDS = {"take.cer_max": 0.03, "take.rounds_max": 3}


def write_generate_plan(root: root_mod.Root, run: str, text: str, **change: object) -> None:
    plan = {
        "schema": 1, "run": run, "teacher": "omnivoice", "state": "neutral", "text": text,
        "lines_file": f"text/{text}/lines.jsonl", "line_ids": [], "anchor": {"voice": VOICE},
        "take_n": 3, "rounds_max": 3, "thresholds": THRESHOLDS,
    }  # fmt: skip
    plan.update(change)
    write_json(root, f"{teacher_cmd.run_rel(run)}/generate-plan.json", plan)


def add_second_run(
    root: root_mod.Root, *, texts: dict[str, str] | None = None, floor_from: str = "seed"
) -> list[TakeRecord]:
    """Add run g2 over text v2, made as g1 was but for one round and the newer text gate."""
    write_lines(root, texts, text=TEXT2, lines=LINES2)
    takes = write_takes(root, run=RUN2, text=TEXT2, lines=LINES2, floor_from=floor_from)
    write_generate_plan(root, RUN, TEXT)
    newer = {**THRESHOLDS, "text.line_chars_min": 80}
    write_generate_plan(root, RUN2, TEXT2, rounds_max=1, thresholds=newer)
    return takes


TWO = (RUN, RUN2)


def edit_json(root: root_mod.Root, rel: str, change: Callable[[dict], None]) -> None:
    path = root.path / rel
    doc = json.loads(path.read_text(encoding="utf-8"))
    change(doc)
    path.write_text(json.dumps(doc), encoding="utf-8")


def plan_of(root: root_mod.Root, version: str = VERSION) -> dict:
    path = root.path / "corpus" / version / "corpus-plan.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_two_runs_of_two_text_versions_combine(root: root_mod.Root, hub: dict) -> None:
    first = selected_takes(write_takes(root))
    second = selected_takes(add_second_run(root))
    result = build(root, runs=TWO)
    assert result.code == 0, result.err
    rows = rows_of(root, "train_aligned.jsonl") + rows_of(root, "valid_aligned.jsonl")
    take_paths = by_path(root, [t.audio for t in first + second])
    assert sorted({r["path"] for r in rows if r["path"] in take_paths}) == sorted(take_paths)
    texts = {str(root.path / t.audio): line_text(t.line_id) for t in first + second}
    assert all(r["transcript"] == texts[r["path"]] for r in rows if r["path"] in texts)
    plan = plan_of(root)
    assert plan["schema"] == 2
    assert "takes_run" not in plan and "lines_file" not in plan
    assert plan["takes_runs"] == [
        {"run": RUN, "lines_file": f"text/{TEXT}/lines.jsonl"},
        {"run": RUN2, "lines_file": f"text/{TEXT2}/lines.jsonl"},
    ]
    assert plan["takes"] == [
        {"take_id": t.take_id, "line_id": t.line_id, "audio": t.audio, "run": run}
        for run, takes in ((RUN, first), (RUN2, second))
        for t in takes
    ]
    assert plan["seed_run"] == SEED_RUN
    manifest = manifest_of(root)
    assert manifest["schema"] == 2 and "takes_run" not in manifest
    assert manifest["takes_runs"] == [RUN, RUN2]
    by_run = manifest["teacher_hours_by_run"]
    assert list(by_run) == [RUN, RUN2]
    for run, takes in ((RUN, first), (RUN2, second)):
        seconds = sum(seconds_of(root, t.audio) for t in takes)
        assert by_run[run] == pytest.approx(seconds / 3600, abs=1e-6)
    assert manifest["teacher_hours"] == pytest.approx(sum(by_run.values()), abs=1e-5)
    assert manifest["teacher_takes"] == len(first) + len(second)
    assert manifest["teacher_lines"] == len(LINES) + len(LINES2)


def test_runs_are_read_in_the_order_given(root: root_mod.Root, hub: dict) -> None:
    add_second_run(root)
    assert build(root, runs=(RUN2, RUN)).code == 0
    assert [r["run"] for r in plan_of(root)["takes_runs"]] == [RUN2, RUN]
    assert manifest_of(root)["takes_runs"] == [RUN2, RUN]


def test_two_runs_hold_out_lines_from_their_union(root: root_mod.Root, hub: dict) -> None:
    set_gates(root, val_fraction=0.25)
    takes = selected_takes(write_takes(root)) + selected_takes(add_second_run(root))
    line_of = {str(root.path / t.audio): t.line_id for t in takes}

    def splits(version: str, *args: str) -> tuple[set[str], set[str]]:
        result = build(root, *args, version=version, runs=TWO)
        assert result.code == 0, result.err
        folder = root.path / "corpus" / version
        found = []
        for name in ("train_aligned.jsonl", "valid_aligned.jsonl"):
            rows = [json.loads(r) for r in (folder / name).read_text("utf-8").splitlines() if r]
            found.append({line_of[r["path"]] for r in rows if r["path"] in line_of})
        return found[0], found[1]

    train, valid = splits("s1")
    union = set(LINES) | set(LINES2)
    assert valid == corpus_cmd.holdout_lines(union, 0.25, seed=0)
    assert len(valid) == 5 and train.isdisjoint(valid) and train | valid == union
    assert splits("s2") == (train, valid)
    seven = splits("s3", "--seed", "7")
    assert seven[1] == corpus_cmd.holdout_lines(union, 0.25, seed=7) != valid
    assert seven[0].isdisjoint(seven[1])
    assert splits("s4", "--seed", "7") == seven


def test_two_runs_cap_the_seed_on_their_combined_training_hours(
    root: root_mod.Root, hub: dict
) -> None:
    set_gates(root, seed_share_max=0.3)
    add_second_run(root)
    result = build(root, runs=TWO)
    assert result.code == 0, result.err
    train = rows_of(root, "train_aligned.jsonl")
    seed_paths = by_path(root, [segment(k).audio for k in TRUSTED])
    kept = {r["path"] for r in train if r["path"] in seed_paths}
    teacher_s = sum(r["duration"] for r in train if r["path"] not in seed_paths)
    seed_s = sum(r["duration"] for r in train if r["path"] in seed_paths)
    first_s = sum(r["duration"] for r in train if f"/{RUN}/" in r["path"])
    assert 0 < first_s < teacher_s  # both runs' takes are in training
    assert 0 < len(kept) < len(TRUSTED)
    assert seed_s / (teacher_s + seed_s) <= 0.3
    for path in set(seed_paths) - kept:  # every segment left out would break the combined cap
        extra = 3 * seconds_of(root, seed_paths[path])
        assert (seed_s + extra) / (teacher_s + seed_s + extra) > 0.3
    # The cap read the training seconds of both runs; one run's alone would fit fewer segments.
    offered = [(seg_id(k), seg_seconds(k)) for k in TRUSTED]
    combined = corpus_cmd.cap_seed(offered, teacher_s, 0.3, 3, seed=0)
    assert {str(root.path / segment(int(s[4:]) - 1).audio) for s in combined} == kept
    assert len(corpus_cmd.cap_seed(offered, first_s, 0.3, 3, seed=0)) < len(kept)


def test_job_inputs_are_each_lines_file_and_run_folder(root: root_mod.Root, hub: dict) -> None:
    first = selected_takes(write_takes(root))
    second = selected_takes(add_second_run(root))
    result = build(root, runs=TWO)
    assert result.code == 0, result.err
    inputs = {i["rel"]: i["sha256"] for i in result.job["inputs"]}
    assert list(inputs) == [
        f"text/{TEXT}/lines.jsonl", f"teachers/omnivoice/{RUN}",
        f"text/{TEXT2}/lines.jsonl", f"teachers/omnivoice/{RUN2}", f"seed/runs/{SEED_RUN}",
    ]  # fmt: skip
    for text in (TEXT, TEXT2):
        rel = f"text/{text}/lines.jsonl"
        assert inputs[rel] == sha256(root.path / rel)
    for run, takes in ((RUN, first), (RUN2, second)):
        folder = root.path / teacher_cmd.run_rel(run)
        parts = [sha256(folder / "takes.jsonl"), *(sha256(root.path / t.audio) for t in takes)]
        digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
        assert inputs[teacher_cmd.run_rel(run)] == digest


def test_the_host_prints_each_runs_counts_and_hours(root: root_mod.Root, hub: dict) -> None:
    second = add_second_run(root)
    unscored = [t for t in second if not t.kept][:2]
    records = [
        (
            TakeRecord.from_json({**t.to_json(), "reasons": [teacher_cmd.UNSCORED]})
            if t in unscored
            else t
        )
        for t in second
    ]
    write_jsonl(root, f"{teacher_cmd.run_rel(RUN2)}/takes.jsonl", records)
    result = build(root, runs=TWO)
    assert result.code == 0, result.err
    by_run = manifest_of(root)["teacher_hours_by_run"]
    assert (
        f"  run {RUN}: {len(LINES) + 1} selected takes, 0 not yet scored, "
        f"{by_run[RUN]:.3f} h aligned" in result.out
    )
    assert (
        f"  run {RUN2}: {len(LINES2)} selected takes, 2 not yet scored, "
        f"{by_run[RUN2]:.3f} h aligned" in result.out
    )


def test_the_gate_holds_the_aligned_hours_of_every_run_together(
    root: root_mod.Root, hub: dict
) -> None:
    first = selected_takes(write_takes(root))
    second = selected_takes(add_second_run(root))
    first_s = sum(seconds_of(root, t.audio) for t in first)
    both_s = first_s + sum(seconds_of(root, t.audio) for t in second)
    set_gates(root, teacher_hours_min=(first_s + both_s) / 2 / 3600)
    assert build(root).code == 3
    result = build(root, runs=TWO, version="s2")
    assert result.code == 0, result.err


def _named_twice(root: root_mod.Root) -> tuple[str, ...]:
    return (RUN, RUN)


def _score_plan(key: str, value: object) -> Callable[[root_mod.Root], tuple[str, ...]]:
    def breaks(root: root_mod.Root) -> tuple[str, ...]:
        def change(plan: dict) -> None:
            *path, last = key.split(".")
            doc = plan
            for part in path:
                doc = doc[part]
            doc[last] = value

        edit_json(root, f"{teacher_cmd.run_rel(RUN2)}/score-plan.json", change)
        return TWO

    return breaks


def _own_floors(root: root_mod.Root) -> tuple[str, ...]:
    write_takes(root, floor_from="round-1-takes")
    add_second_run(root, floor_from="round-1-takes")
    return TWO


def _take_n(root: root_mod.Root) -> tuple[str, ...]:
    write_generate_plan(root, RUN2, TEXT2, take_n=4)
    return TWO


def _same_lines_file(root: root_mod.Root) -> tuple[str, ...]:
    write_takes(root, run=RUN2, text=TEXT, lines=LINES)
    write_generate_plan(root, RUN2, TEXT)
    return TWO


def _shared_line_id(root: root_mod.Root) -> tuple[str, ...]:
    path = root.path / "text" / TEXT2 / "lines.jsonl"
    extra = {"line_id": LINES[0], "state": "neutral", "text": f"{CANARY.transcript} elsewhere."}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(extra) + "\n")
    return TWO


def _shared_text(root: root_mod.Root) -> tuple[str, ...]:
    # The same line as g1's first, in other case and punctuation.
    write_lines(root, {LINES2[0]: line_text(LINES[0]).upper() + "!"}, text=TEXT2, lines=LINES2)
    return TWO


@pytest.mark.parametrize(
    ("breaks", "said", "hidden"),
    [
        (_named_twice, "named twice", None),
        (_score_plan("anchor.voice", "another-voice"), "anchor", "another-voice"),
        # The ecapa value: "wavlm-sv" holds ".wav", which `build` refuses to see printed.
        (_score_plan("floor.values.ecapa", 0.625), "floor.values.ecapa", "0.625"),
        (_score_plan("floor.centroid_sha256.ecapa", "c" * 64), "centroid_sha256", "c" * 64),
        (_score_plan("cer_max", 0.0625), "cer_max", "0.0625"),
        (_own_floors, "seed target", None),
        (_take_n, "take_n", None),
        (_same_lines_file, "same lines file", None),
        (_shared_line_id, "1 line id", None),
        (_shared_text, "1 line", None),
    ],
    ids=[
        "named-twice", "anchor", "floor-value", "centroid-hash", "cer-max", "own-floors",
        "take-n", "same-lines-file", "shared-line-id", "shared-text",
    ],
)  # fmt: skip
def test_runs_made_differently_are_refused(
    root: root_mod.Root,
    hub: dict,
    breaks: Callable[[root_mod.Root], tuple[str, ...]],
    said: str,
    hidden: str | None,
) -> None:
    add_second_run(root)
    runs = breaks(root)
    result = build(root, runs=runs)
    assert result.code == 2, result.out + result.err
    assert said in result.err and RUN in result.err, result.err
    if len(set(runs)) > 1:
        assert RUN2 in result.err
    if hidden is not None:
        assert hidden not in result.err
    assert "elsewhere" not in result.err
    assert not result.docker.argvs
    assert not (corpus_dir(root) / "corpus-plan.json").exists()


def test_a_shared_text_refusal_counts_by_normalised_text(root: root_mod.Root, hub: dict) -> None:
    add_second_run(root)
    _shared_text(root)
    result = build(root, runs=TWO)
    assert result.code == 2 and "share 1 line by normalised text" in result.err, result.err
    assert text_corpus.normalise(line_text(LINES[0])) not in result.err


def test_one_run_needs_no_generation_plan_and_keeps_an_own_floor(
    root: root_mod.Root, hub: dict
) -> None:
    write_takes(root, floor_from="round-1-takes")
    assert not (root.path / teacher_cmd.run_rel(RUN) / "generate-plan.json").exists()
    assert build(root).code == 0
    assert manifest_of(root)["takes_runs"] == [RUN]


def test_the_stage_refuses_a_schema_1_plan(root: root_mod.Root, hub: dict) -> None:
    assert build(root).code == 0
    edit_plan(root, {"schema": 1})
    with pytest.raises(stage.StageError, match="schema 2"):
        stage.run(root.path, VERSION, "cpu", 8)


def test_the_stage_refuses_a_take_whose_run_is_not_in_the_plan(
    root: root_mod.Root, hub: dict
) -> None:
    add_second_run(root)
    assert build(root, runs=TWO).code == 0
    plan = plan_of(root)
    take = plan["takes"][0]["take_id"]
    edit_json(root, f"corpus/{VERSION}/corpus-plan.json",
              lambda doc: doc["takes"][0].update(run="g9"))  # fmt: skip
    with pytest.raises(stage.StageError, match=f"take {take}.*not in the plan"):
        stage.run(root.path, VERSION, "cpu", 8)


def test_the_stage_checks_each_take_against_its_own_run(root: root_mod.Root, hub: dict) -> None:
    add_second_run(root)
    assert build(root, runs=TWO).code == 0
    take = plan_of(root)["takes"][0]["take_id"]
    edit_json(root, f"corpus/{VERSION}/corpus-plan.json",
              lambda doc: doc["takes"][0].update(run=RUN2))  # fmt: skip
    with pytest.raises(stage.StageError, match=f"take {take} is not a selected take of run {RUN2}"):
        stage.run(root.path, VERSION, "cpu", 8)


def test_teacher_runs_reads_schema_1_and_schema_2_manifests() -> None:
    assert corpus_cmd.teacher_runs({"schema": 1, "takes_run": RUN}) == (RUN,)
    assert corpus_cmd.teacher_runs({"schema": 2, "takes_runs": [RUN, RUN2]}) == (RUN, RUN2)


@pytest.mark.parametrize(
    "manifest",
    [
        {"schema": 2},
        {"schema": 1},
        {"schema": 2, "takes_run": RUN},
        {"schema": 2, "takes_runs": []},
        {"schema": 2, "takes_runs": RUN},
        {"schema": 2, "takes_runs": [RUN, "../x"]},
        {"schema": 1, "takes_run": "../x"},
        {"schema": 1, "takes_run": 7},
        [],
    ],
    ids=["v2-none", "v1-none", "v2-old-key", "empty", "a-string", "bad-id-2", "bad-id-1",
         "not-a-string", "not-a-dict"],
)  # fmt: skip
def test_teacher_runs_refuses_a_manifest_without_valid_runs(manifest: object) -> None:
    with pytest.raises(InputError, match="refused"):
        corpus_cmd.teacher_runs(manifest)


# The upstream aligner, run in-process from the clone.


@pytest.fixture
def clone(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """A stand-in clone whose `training.scripts.align_data` records its argv and exit code."""
    folder = tmp_path / "pocket-tts"
    scripts = folder / "training" / "scripts"
    scripts.mkdir(parents=True)
    (folder / "training" / "__init__.py").write_text("", encoding="utf-8")
    (scripts / "__init__.py").write_text("", encoding="utf-8")
    (scripts / "align_data.py").write_text(
        "import json, os, sys\n"
        "if __name__ == '__main__':\n"
        "    out = os.environ['FAKE_ALIGN_RECORD']\n"
        "    open(out, 'w').write(json.dumps({'argv': sys.argv[1:], 'path0': sys.path[0]}))\n"
        "    sys.exit(int(os.environ.get('FAKE_ALIGN_EXIT', '0')))\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(stage, "UPSTREAM", folder)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", ["pytest"])
    for name in fetch.QUIET_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FAKE_ALIGN_RECORD", str(tmp_path / "record.json"))
    yield folder
    for name in [n for n in sys.modules if n == "training" or n.startswith("training.")]:
        del sys.modules[name]


def test_align_runs_upstreams_script_from_the_clone(clone: Path, tmp_path: Path) -> None:
    stage.align(tmp_path / "in.jsonl", tmp_path / "out.jsonl", ["--device", "cuda"])
    record = json.loads((tmp_path / "record.json").read_text(encoding="utf-8"))
    assert record["argv"] == [str(tmp_path / "in.jsonl"), str(tmp_path / "out.jsonl"),
                              "--device", "cuda"]  # fmt: skip
    assert record["path0"] == str(clone)
    assert sys.argv == ["pytest"]


def test_a_failing_aligner_fails_the_stage(
    clone: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_ALIGN_EXIT", "2")
    with pytest.raises(stage.StageError):
        stage.align(tmp_path / "in.jsonl", tmp_path / "out.jsonl", [])
    assert sys.argv == ["pytest"]
