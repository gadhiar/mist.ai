"""`labels prefill`: the host command's plan and job, and the stage that pre-fills each clip (T15).

A synthetic extract run of four kept segments and one dropped one goes through the command, whose
job runs the stage in this process with a scripted `EmotionDimensions` (as `test_clean.py` runs
its stage), and the tests read what it wrote. The stage's refusals are tested on the stage
itself, after the host has written the plan and a file has been changed or removed.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts.voice_data import cli, jobs
from scripts.voice_data import labels as labels_mod
from scripts.voice_data import root as root_mod
from scripts.voice_data.commands import labels as labels_cmd
from scripts.voice_data.commands.extract import TurnRecord
from scripts.voice_data.records import Prosody, SegmentRecord, read_jsonl, write_jsonl
from scripts.voice_data.stages import prefill as prefill_stage

from .fakes import FakeArousal

RATE = 16000
IMAGE = "mist-voice-base:0123456789ab"
RUN = "r1"
RUN_DIR = f"seed/runs/{RUN}"
OUT_DIR = f"seed/prefill/{RUN}"


# The extract run.


@dataclass(frozen=True)
class Seg:
    sid: str
    source: str
    start: float
    end: float
    diar_cluster: str | None
    overlap_ms: int
    values: tuple[float, float, float]


# Each kept segment has its own length, so the scripted model tells them apart by sample count.
SEGS = (
    Seg("seg-000001", "src-0001", 10.0, 14.0, "spk-01", 0, (0.125, 0.3, 0.625)),
    Seg("seg-000002", "src-0001", 20.0, 25.0, None, 120, (0.9, 0.5, 0.1)),
    Seg("seg-000003", "src-0001", 30.0, 36.0, None, 0, (0.51234, 0.5, 0.49996)),
    Seg("seg-000004", "src-0002", 40.0, 43.5, None, 0, (0.6, 0.0, 1.0)),
)
DROPPED = "seg-000005"


def turn(source: str, start: float, end: float, role: str = "other") -> TurnRecord:
    return TurnRecord(
        source_id=source, turn=0, start_s=start, end_s=end, diar_cluster="spk-02", role=role,
        embedded=True, touches_overlap=False, sim_anchor=None, sim_target=None, gaps=None,
        passed=None, reasons=(),
    )  # fmt: skip


TURNS = (
    turn("src-0001", 9.0, 11.0),
    turn("src-0001", 11.0, 30.0, role="primary"),
    turn("src-0001", 30.0, 31.5),
    turn("src-0001", 40.0, 43.5),
    turn("src-0002", 40.0, 40.05),
)


def segment(seg: Seg, run: str = RUN, kept: bool = True) -> SegmentRecord:
    return SegmentRecord(
        segment_id=seg.sid, source_id=seg.source, start_s=seg.start, end_s=seg.end,
        channel="FC", channel_reason=None, separator="none",
        audio=f"seed/runs/{run}/audio/{seg.sid}.flac" if kept else "",
        diar_cluster=seg.diar_cluster, cluster="primary", flag_group=None,
        sim={"wavlm-sv": 0.9, "ecapa": 0.8}, overlap_ms=seg.overlap_ms, snr_db=30.0, c50_db=40.0,
        transcripts={}, asr_agreement_cer=0.0,
        prosody=Prosody(f0_mean_hz=210.0, f0_std_hz=4.0, rate_sps=4.0, energy_db=-20.0),
        arousal=0.4, state_proposed="neutral", state=None, set="short", kept=kept,
        reasons=() if kept else ("overlap",),
    )  # fmt: skip


def tone(seconds: float, hz: float = 220.0) -> np.ndarray:
    t = np.arange(int(round(seconds * RATE)), dtype=np.float64) / RATE
    return (0.1 * np.sin(2.0 * np.pi * hz * t)).astype(np.float32)


def make_root(base: Path, segs: Sequence[Seg] = SEGS, run: str = RUN) -> root_mod.Root:
    """A private root holding an extract run of `segs`, each a FLAC, and one dropped segment."""
    root = root_mod.init_root(base / "voice-root", "private")
    records = []
    for seg in segs:
        record = segment(seg, run)
        path = root.path / record.audio
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), tone(seg.end - seg.start), RATE, format="FLAC", subtype="PCM_16")
        records.append(record)
    records.append(segment(Seg(DROPPED, "src-0002", 50.0, 53.0, None, 0, (0, 0, 0)), run, False))
    write_jsonl(root, f"seed/runs/{run}/segments.jsonl", records)
    write_jsonl(root, f"seed/runs/{run}/turns.jsonl", TURNS)
    return root


class ScriptedDimensions:
    """An `EmotionDimensions` that returns each segment's scripted values, found by clip length."""

    def __init__(self, segs: Sequence[Seg] = SEGS) -> None:
        self.values = {round(s.end - s.start, 2): s.values for s in segs}
        self.rates: list[int] = []

    def dimensions(self, audio: np.ndarray, sample_rate: int) -> tuple[float, float, float]:
        self.rates.append(sample_rate)
        return self.values[round(len(audio) / sample_rate, 2)]


# The command, with a job that runs the stage in this process.


class StageDocker:
    """A `DockerOps` whose `run_logged` runs the stage here with the given model."""

    def __init__(self, root: root_mod.Root, model: object, free_mib: int = 12000) -> None:
        self.root = root
        self.model = model
        self.free_mib = free_mib
        self.argvs: list[list[str]] = []
        self.captures: list[list[str]] = []
        self.devices: list[str] = []

    def capture(self, argv: Sequence[str]) -> str:
        self.captures.append(list(argv))
        if list(argv) == ["docker", "ps", "-q"]:
            return ""
        assert "nvidia-smi" in argv
        return f"{self.free_mib}\n"

    def check(self, argv: Sequence[str]) -> None:
        raise AssertionError("labels prefill builds nothing")

    def run_logged(self, argv: Sequence[str], log: Path) -> int:
        self.argvs.append(list(argv))
        at = list(argv).index("-m")
        module, rest = argv[at + 1], list(argv[at + 2 :])
        assert rest[:2] == ["--data", "/data"]
        rest[1] = str(self.root.path)
        stage = importlib.import_module(module)

        def load_model(device: str) -> object:
            self.devices.append(device)
            return self.model

        with (
            pytest.MonkeyPatch.context() as mp,
            log.open("w", encoding="utf-8") as handle,
            contextlib.redirect_stdout(handle),
            contextlib.redirect_stderr(handle),
        ):
            mp.setattr(stage, "load_model", load_model)
            return stage.main(rest)


class _Capture:
    def __init__(self) -> None:
        self.parts: list[str] = []

    def write(self, text: str) -> int:
        self.parts.append(text)
        return len(text)

    def flush(self) -> None:
        pass

    def getvalue(self) -> str:
        return "".join(self.parts)


@contextlib.contextmanager
def _patched(docker: StageDocker, families: list[str]) -> Iterator[None]:
    def image_tag(family: str) -> str:
        families.append(family)
        return IMAGE

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(labels_cmd, "make_docker", lambda: docker)
        mp.setattr(labels_cmd, "image_tag", image_tag)
        mp.delenv(jobs.GPU_PEERS_ENV, raising=False)
        yield


@dataclass
class Ran:
    root: root_mod.Root
    code: int
    out: str
    err: str
    docker: StageDocker
    families: list[str]

    def json(self, name: str) -> dict:
        return json.loads((self.root.path / OUT_DIR / name).read_text(encoding="utf-8"))

    def records(self) -> list[labels_mod.PrefillRecord]:
        return read_jsonl(self.root.path / OUT_DIR / "prefill.jsonl", labels_mod.PrefillRecord)

    def job(self) -> dict:
        (path,) = (self.root.path / "runs").glob("job-*/job.json")
        return json.loads(path.read_text(encoding="utf-8"))


def run_prefill(root: root_mod.Root, docker: StageDocker, *args: str) -> Ran:
    out, err = _Capture(), _Capture()
    families: list[str] = []
    argv = ["--root", str(root.path), "labels", "prefill", *args]
    with (
        _patched(docker, families),
        contextlib.redirect_stdout(out),
        contextlib.redirect_stderr(err),
    ):
        code = cli.main(argv)
    return Ran(root, code, out.getvalue(), err.getvalue(), docker, families)


@pytest.fixture(scope="module")
def main_run(tmp_path_factory: pytest.TempPathFactory) -> Ran:
    root = make_root(tmp_path_factory.mktemp("prefill"))
    return run_prefill(root, StageDocker(root, ScriptedDimensions()), "--run", RUN)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# The host command: plan, job and prints.


def test_the_prefill_run_exits_0(main_run: Ran) -> None:
    assert main_run.code == 0, main_run.err


def test_the_plan_pins_segments_turns_and_each_kept_clip(main_run: Ran) -> None:
    plan = main_run.json("prefill-plan.json")
    assert (plan["schema"], plan["run"], plan["min_s"]) == (1, RUN, 3)
    clips = [f"{RUN_DIR}/audio/{s.sid}.flac" for s in SEGS]
    expected = [f"{RUN_DIR}/segments.jsonl", f"{RUN_DIR}/turns.jsonl", *clips]
    assert sorted(plan["files"]) == sorted(expected)
    for rel, digest in plan["files"].items():
        assert digest == sha(main_run.root.path / rel)


def test_the_job_runs_the_stage_in_the_score_image_with_the_run_folder_read_only(
    main_run: Ran,
) -> None:
    (argv,) = main_run.docker.argvs
    assert main_run.families == ["score"]
    assert IMAGE in argv and "--gpus" not in argv
    assert argv[argv.index("--network") + 1] == "none"
    at = argv.index(IMAGE)
    assert argv[at + 1 : at + 6] == [
        "/opt/venv-score/bin/python", "-I", "-u", "-m", "scripts.voice_data.stages.prefill"
    ]  # fmt: skip
    assert argv[-4:] == ["--run", RUN, "--device", "cpu"]
    mounts = [argv[k + 1] for k, a in enumerate(argv) if a == "-v"]
    assert f"{main_run.root.path / 'seed' / 'runs' / RUN}:/data/{RUN_DIR}:ro" in mounts
    assert [m for m in mounts if m.endswith(f":/data/{OUT_DIR}")]
    assert len([m for m in mounts if ":/data/seed/" in m]) == 2
    assert main_run.docker.devices == ["cpu"]


def test_job_json_pins_the_digest_of_the_plan_shas_in_path_order(main_run: Ran) -> None:
    job = main_run.job()
    assert job["family"] == "score" and job["module"] == "scripts.voice_data.stages.prefill"
    assert job["gpu"] is False and job["network"] is False and job["vram_gb"] is None
    assert job["outputs"] == OUT_DIR
    files = main_run.json("prefill-plan.json")["files"]
    digest = hashlib.sha256("\n".join(files[k] for k in sorted(files)).encode()).hexdigest()
    assert job["inputs"] == [{"rel": RUN_DIR, "sha256": digest}]


def test_the_host_prints_counts_seconds_and_voices_only(main_run: Ran) -> None:
    lines = main_run.out.splitlines()
    for line in lines:
        assert line.startswith(("job ", "prefill run ", "  voices:")), line
    assert f"prefill run {RUN}: 4 clips, 18.50 s" in lines
    assert "  voices: her 2, her_overlap 1, her_nearby 1" in lines
    assert "seed/" not in main_run.out and ".flac" not in main_run.out


def test_the_command_is_discovered_without_registration() -> None:
    assert "labels" in cli.command_names()


# The stage's outputs.


def test_one_prefill_record_per_kept_segment_in_segment_order(main_run: Ran) -> None:
    rows = [r.to_json() for r in main_run.records()]
    common = {"source_id": "src-0001"}
    assert rows == [
        {**common, "segment_id": "seg-000001", "start_s": 10.0, "end_s": 14.0, "duration_s": 4.0,
         "arousal": 0.125, "dominance": 0.3, "valence": 0.625, "energy": 2, "mood": 4,
         "voices": "her", "keep_from_s": None, "keep_to_s": None, "other_s": 1.0, "overlap_ms": 0},
        {**common, "segment_id": "seg-000002", "start_s": 20.0, "end_s": 25.0, "duration_s": 5.0,
         "arousal": 0.9, "dominance": 0.5, "valence": 0.1, "energy": 5, "mood": 1,
         "voices": "her_overlap", "keep_from_s": None, "keep_to_s": None, "other_s": 0.0,
         "overlap_ms": 120},
        {**common, "segment_id": "seg-000003", "start_s": 30.0, "end_s": 36.0, "duration_s": 6.0,
         "arousal": 0.5123, "dominance": 0.5, "valence": 0.5, "energy": 3, "mood": 3,
         "voices": "her_nearby", "keep_from_s": 1.5, "keep_to_s": 6.0, "other_s": 1.5,
         "overlap_ms": 0},
        {"source_id": "src-0002", "segment_id": "seg-000004", "start_s": 40.0, "end_s": 43.5,
         "duration_s": 3.5, "arousal": 0.6, "dominance": 0.0, "valence": 1.0, "energy": 3,
         "mood": 5, "voices": "her", "keep_from_s": None, "keep_to_s": None, "other_s": 0.05,
         "overlap_ms": 0},
    ]  # fmt: skip
    assert main_run.docker.model.rates == [RATE] * 4


def test_energy_and_mood_are_scale_from_of_arousal_and_valence(main_run: Ran) -> None:
    for record in main_run.records():
        assert record.energy == labels_mod.scale_from(record.arousal)
        assert record.mood == labels_mod.scale_from(record.valence)


def test_summary_and_outputs_are_written(main_run: Ran) -> None:
    assert main_run.json("summary.json") == {
        "schema": 1, "run": RUN, "clips": 4, "seconds": 18.5, "device": "cpu",
        "voices": {"her": 2, "her_overlap": 1, "her_nearby": 1},
    }  # fmt: skip
    files = main_run.json("outputs.json")["files"]
    assert [f["rel"] for f in files] == [f"{OUT_DIR}/prefill.jsonl", f"{OUT_DIR}/summary.json"]
    for item in files:
        assert item["sha256"] == sha(main_run.root.path / item["rel"])
    job_id = main_run.job()["job_id"]
    assert (main_run.root.path / "runs" / job_id / "outputs.json").is_file()


def test_the_extract_run_is_left_as_it_was(main_run: Ran) -> None:
    plan = main_run.json("prefill-plan.json")
    for rel, digest in plan["files"].items():
        assert sha(main_run.root.path / rel) == digest
    assert sorted(p.name for p in (main_run.root.path / RUN_DIR).iterdir()) == [
        "audio", "segments.jsonl", "turns.jsonl"
    ]  # fmt: skip


def test_the_stage_docstring_names_its_command_line() -> None:
    venv = jobs.load_families()["score"]["venv"]
    doc = " ".join((prefill_stage.__doc__ or "").split())
    assert f"{venv}/bin/python -I -u -m scripts.voice_data.stages.prefill" in doc


def test_the_score_family_already_holds_the_emotion_model() -> None:
    family = jobs.load_families()[labels_cmd.FAMILY]
    assert labels_cmd.FAMILY == "score" and "arousal" in family["models"]
    assert "cpu" in family["devices"] and "cuda" in family["devices"]


# On cuda.


def test_on_cuda_the_job_sets_gpu_and_keeps_the_family_vram(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    docker = StageDocker(root, FakeArousal(), free_mib=12000)
    ran = run_prefill(root, docker, "--run", RUN, "--device", "cuda")
    assert ran.code == 0, ran.err
    (argv,) = docker.argvs
    assert argv[argv.index("--gpus") + 1] == "all"
    assert argv[-4:] == ["--run", RUN, "--device", "cuda"]
    job = ran.job()
    assert job["gpu"] is True and job["vram_gb"] is None
    assert docker.devices == ["cuda"] and ran.json("summary.json")["device"] == "cuda"
    # FakeArousal's valence is one minus its arousal, so mood is read from the third output.
    for record in ran.records():
        assert record.dominance == record.arousal
        assert record.valence == pytest.approx(1.0 - record.arousal, abs=1e-4)
        assert record.arousal != record.valence


def test_on_cuda_the_preflight_asks_for_the_family_vram(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    docker = StageDocker(root, FakeArousal(), free_mib=5 * 1024)
    ran = run_prefill(root, docker, "--run", RUN, "--device", "cuda")
    assert ran.code == 2
    assert "family score needs 6" in ran.err
    assert docker.argvs == []


# The host's refusals, each before any job.


def _refused(ran: Ran, message: str) -> None:
    assert ran.code == 2, ran.out
    assert message in ran.err
    assert ran.docker.argvs == []


def test_an_unknown_run_is_refused(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    ran = run_prefill(root, StageDocker(root, ScriptedDimensions()), "--run", "r9")
    _refused(ran, "run extract first")
    assert not (root.path / "seed" / "prefill" / "r9").exists()


def test_a_run_id_that_is_not_one_folder_is_refused(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    _refused(run_prefill(root, StageDocker(root, ScriptedDimensions()), "--run", "../x"), "run id")


def test_a_run_with_no_kept_segment_is_refused(tmp_path: Path) -> None:
    root = make_root(tmp_path, segs=())
    ran = run_prefill(root, StageDocker(root, ScriptedDimensions()), "--run", RUN)
    _refused(ran, "no kept segment")


def test_a_run_without_turns_is_refused(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    (root.path / RUN_DIR / "turns.jsonl").unlink()
    ran = run_prefill(root, StageDocker(root, ScriptedDimensions()), "--run", RUN)
    _refused(ran, "turns.jsonl")


@pytest.mark.parametrize("name", ["prefill.jsonl", "summary.json"])
def test_an_output_folder_that_holds_a_prefill_is_refused(tmp_path: Path, name: str) -> None:
    root = make_root(tmp_path)
    root.write_bytes(f"{OUT_DIR}/{name}", b"{}\n")
    ran = run_prefill(root, StageDocker(root, ScriptedDimensions()), "--run", RUN)
    _refused(ran, "already holds a pre-fill")
    assert not (root.path / OUT_DIR / "prefill-plan.json").exists()


def test_a_second_prefill_of_the_same_run_is_refused(main_run: Ran) -> None:
    ran = run_prefill(main_run.root, StageDocker(main_run.root, ScriptedDimensions()), "--run", RUN)
    _refused(ran, "already holds a pre-fill")


# The stage's checks of the plan's files.


@pytest.fixture
def planned(tmp_path: Path) -> root_mod.Root:
    """A root whose run has its plan written and no job run yet."""
    root = make_root(tmp_path)
    labels_cmd.write_plan(root, RUN)
    return root


def run_stage(root: root_mod.Root) -> list[str]:
    return prefill_stage.run(root.path, RUN, ScriptedDimensions(), "cpu")


def assert_refused(root: root_mod.Root, kind: str) -> None:
    with pytest.raises(prefill_stage.StageError) as info:
        run_stage(root)
    message = str(info.value)
    assert kind in message
    assert "seed/" not in message and ".flac" not in message and "seg-" not in message
    assert not (root.path / OUT_DIR / "prefill.jsonl").exists()
    assert not (root.path / OUT_DIR / "summary.json").exists()


def test_the_stage_runs_on_a_planned_run(planned: root_mod.Root) -> None:
    written = run_stage(planned)
    assert written == [f"{OUT_DIR}/prefill.jsonl", f"{OUT_DIR}/summary.json"]


def test_a_changed_flac_is_refused_as_a_kept_clip(planned: root_mod.Root) -> None:
    path = planned.path / RUN_DIR / "audio" / "seg-000003.flac"
    sf.write(str(path), tone(6.0, hz=330.0), RATE, format="FLAC", subtype="PCM_16")
    assert_refused(planned, "a kept clip")


def test_a_missing_flac_is_refused_as_a_kept_clip(planned: root_mod.Root) -> None:
    (planned.path / RUN_DIR / "audio" / "seg-000002.flac").unlink()
    assert_refused(planned, "a kept clip")


def test_a_changed_turns_file_is_refused(planned: root_mod.Root) -> None:
    write_jsonl(planned, f"{RUN_DIR}/turns.jsonl", TURNS[:-1])
    assert_refused(planned, "turns.jsonl")


def test_a_missing_turns_file_is_refused(planned: root_mod.Root) -> None:
    (planned.path / RUN_DIR / "turns.jsonl").unlink()
    assert_refused(planned, "turns.jsonl")


def test_a_changed_segments_file_is_refused(planned: root_mod.Root) -> None:
    records = read_jsonl(planned.path / RUN_DIR / "segments.jsonl", SegmentRecord)
    write_jsonl(planned, f"{RUN_DIR}/segments.jsonl", [replace(records[0], overlap_ms=5)])
    assert_refused(planned, "segments.jsonl")


def test_a_plan_whose_clips_are_not_the_kept_segments_is_refused(planned: root_mod.Root) -> None:
    path = planned.path / OUT_DIR / "prefill-plan.json"
    plan = json.loads(path.read_text(encoding="utf-8"))
    del plan["files"][f"{RUN_DIR}/audio/seg-000004.flac"]
    path.write_text(json.dumps(plan), encoding="utf-8")
    assert_refused(planned, "kept")


@pytest.mark.parametrize(
    "change",
    [{"schema": 2}, {"run": "r2"}, {"min_s": "3"}, {"files": []}],
    ids=["schema", "run", "min_s", "files"],
)
def test_a_plan_that_is_not_this_runs_is_refused(
    planned: root_mod.Root, change: Mapping[str, object]
) -> None:
    path = planned.path / OUT_DIR / "prefill-plan.json"
    plan = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**plan, **change}), encoding="utf-8")
    assert_refused(planned, "plan")


def test_a_missing_plan_is_refused(planned: root_mod.Root) -> None:
    (planned.path / OUT_DIR / "prefill-plan.json").unlink()
    assert_refused(planned, "plan")


def test_main_reports_a_refusal_as_exit_1_naming_the_file_kind(
    planned: root_mod.Root, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    (planned.path / RUN_DIR / "turns.jsonl").unlink()
    monkeypatch.setattr(prefill_stage, "load_model", lambda device: ScriptedDimensions())
    code = prefill_stage.main(["--data", str(planned.path), "--run", RUN, "--device", "cpu"])
    captured = capsys.readouterr()
    assert code == 1
    assert "turns.jsonl" in captured.err and "seed/" not in captured.err + captured.out
