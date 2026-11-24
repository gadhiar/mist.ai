"""`student score` and `student export`: a checkpoint gated against the anchor, then an identity.

The commands run through the CLI, and each job's stage runs in this process (`StageDocker`), as
`test_student_train.py` does. The render job's student is `FakeSynth`, which speaks each line as
a synthetic voice whose length is unique to the line, so the scripted ASR hears a line by its
length; the scoring job gets the fake embedders of `fakes.py` and a prosody whose F0 SD a test
sets. The gates' references are a hand-written seed target's numbers, reached as the command
reaches them: the corpus, its teacher run, that run's score plan, the target. The weights are
safetensors files with a stdlib-written header, and the hub cache is a folder laid out as
Hugging Face's, with a fake `huggingface_hub` for the job's lookups.

The export's contract test loads the identity with the realtime loader
(`backend/realtime/voice_assets.py::load_identity`) and the Pocket adapter's model pin check.

The canary transcript is the anchor's transcript and every validation line's text; it must reach
no printed line or job log, and no printed line names a clip.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import math
import sys
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

from backend.realtime import voice_assets
from backend.realtime.engines import pocket_tts
from scripts.voice_data import cli, jobs
from scripts.voice_data import root as root_mod
from scripts.voice_data.bindings.protocols import Transcript
from scripts.voice_data.commands import student as student_cmd
from scripts.voice_data.commands import teacher as teacher_cmd
from scripts.voice_data.records import Prosody, write_json
from scripts.voice_data.stages import fetch, student_train
from scripts.voice_data.stages import student_score as stage
from scripts.voice_data.stages import teacher_score as score_stage

from .conftest import Canaries, _assert_no_canary, _synth_voice
from .fakes import FakeArousal, FakeEmbedder
from .test_identity import _Capture, make_identity

RATE = 24000
RUN = "s1"
CORPUS = "c1"
TAKES_RUN = "g1"
ANCHOR = "anchor-voice"
VOICE = "student-voice"
CHECKPOINT = "checkpoint_00002500"
NEWER = "checkpoint_00005000"
CANARY = Canaries()
FORMANTS = (700.0, 1200.0, 2600.0)
TARGET = "t1"
# Well under the SIM a render in the anchor's own voice reaches on the fake embedders, so the
# ratio passes; a median above 1 makes any cosine fail it.
SEED_SIM_MEDIANS = {"wavlm-sv": 0.5, "ecapa": 0.5}
SEED_F0_SD = 20.0
SEED_F0_SPREAD = 3.0
ANCHOR_SECONDS = 4.0
VALID_N = 3
FAMILIES = jobs.load_families()
POCKET = FAMILIES["train-pocket"]["models"]["pocket-tts"]
RUNTIME_TOKENIZER = FAMILIES["train-pocket"]["models"]["pocket-tts-runtime-tokenizer"]
STOCK_TOKENIZER = b'{"model": "the stock runtime tokenizer"}'
BASE_TOKENIZER = b'{"model": "the 24-layer base tokenizer"}'
LAYOUT = {"flow_lm.layer.weight": (4, 2), "flow_lm.layer.bias": (4,), "mimi.conv.weight": (2, 2)}
BENCH_DOC = json.loads(student_cmd.bench_lines_path().read_text(encoding="utf-8"))
# The bench speaks its lines and its acknowledgements (`tts_suite._item_ids`).
BENCH_N = len(BENCH_DOC["lines"]) + len(BENCH_DOC["acknowledgements"])


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safetensors_bytes(layout: Mapping[str, Sequence[int]], fill: int = 0) -> bytes:
    """A safetensors file of F32 tensors, its header written as the format specifies."""
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, shape in layout.items():
        size = 4 * math.prod(shape)
        header[name] = {
            "dtype": "F32",
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    head = json.dumps(header).encode("utf-8")
    return len(head).to_bytes(8, "little") + head + bytes([fill]) * offset


# The run `student train` leaves, the anchor, the corpus and the cache.


def valid_text(number: int) -> str:
    return f"{CANARY.transcript}, validation {number}."


def take_rel(number: int) -> str:
    return f"{teacher_cmd.run_rel(TAKES_RUN)}/takes/take-{number:04d}.flac"


def write_anchor(root: root_mod.Root, voice: str = ANCHOR) -> teacher_cmd.Anchor:
    clip = _synth_voice(205.0, FORMANTS, ANCHOR_SECONDS, RATE)
    make_identity(root.path / "voices", voice, {"neutral": clip})
    return teacher_cmd.read_anchor(root, voice)


def write_corpus(root: root_mod.Root) -> None:
    """A built corpus of teacher takes in the anchor's voice, and its manifest."""
    for number in range(4 + VALID_N):
        path = root.path / take_rel(number)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), _synth_voice(205.0, FORMANTS, 2.0, RATE), RATE, format="FLAC")

    def row(number: int, text: str) -> str:
        doc = {"path": f"/data/{take_rel(number)}", "duration": 2.0, "transcript": text,
               "words": []}  # fmt: skip
        return json.dumps(doc) + "\n"

    folder = root.path / "corpus" / CORPUS
    folder.mkdir(parents=True, exist_ok=True)
    train = "".join(row(n, CANARY.transcript) for n in range(4))
    valid = "".join(row(4 + k, valid_text(k + 1)) for k in range(VALID_N))
    (folder / "train_aligned.jsonl").write_text(train, encoding="utf-8")
    (folder / "valid_aligned.jsonl").write_text(valid, encoding="utf-8")
    files = [
        {"rel": f"corpus/{CORPUS}/{name}", "sha256": sha256((folder / name).read_bytes())}
        for name in ("train_aligned.jsonl", "valid_aligned.jsonl")
    ]
    manifest = {"schema": 1, "version": CORPUS, "takes_run": TAKES_RUN, "seed_run": None,
                "files": files}  # fmt: skip
    write_json(root, f"corpus/{CORPUS}/manifest.json", manifest)


def snapshot(root: root_mod.Root, model: Mapping[str, object]) -> Path:
    repository = str(model["repository"]).replace("/", "--")
    return (
        root.path / "cache/hf/hub" / f"models--{repository}" / "snapshots" / str(model["revision"])
    )


def write_cache(root: root_mod.Root, *, base_tokenizer: bool = True) -> None:
    files = {
        snapshot(root, POCKET) / student_train.STUDENT_WEIGHTS[1]: safetensors_bytes(LAYOUT),
        snapshot(root, RUNTIME_TOKENIZER) / student_train.STUDENT_TOKENIZER[1]: STOCK_TOKENIZER,
    }
    if base_tokenizer:
        files[snapshot(root, POCKET) / student_train.BASE_TOKENIZER[1]] = BASE_TOKENIZER
    for path, data in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def write_checkpoint(
    root: root_mod.Root, checkpoint: str = CHECKPOINT, layout: Mapping = LAYOUT, fill: int = 7
) -> Path:
    path = root.path / student_cmd.weights_rel(RUN, checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(safetensors_bytes(layout, fill))
    return path


def write_train(root: root_mod.Root, tokenizer: bytes = STOCK_TOKENIZER) -> None:
    doc = {"schema": 1, "run": RUN, "corpus": CORPUS, "status": "done",
           "tokenizer_sha256": sha256(tokenizer)}  # fmt: skip
    write_json(root, f"students/{RUN}/train.json", doc)


def write_target(
    root: root_mod.Root,
    *,
    medians: Mapping[str, float] = SEED_SIM_MEDIANS,
    f0_std_hz: float = SEED_F0_SD,
    mode: str = "seed",
    voice_id: str = ANCHOR,
) -> None:
    """A seed target's `target.json`, with the numbers `student score` reads from it."""
    prosody = {"f0_mean_hz": 200.0, "rate_sps": 4.0, "energy_db": -20.0}
    neutral = {
        "floor": {name: 0.5 for name in medians},
        "floor_from": "own",
        "n": 12,
        "profile": {**prosody, "f0_std_hz": f0_std_hz},
        "spread": {**prosody, "f0_std_hz": SEED_F0_SPREAD},
    }
    record = {
        "schema": 1, "target_id": TARGET, "voice_id": voice_id, "mode": mode, "seed_run": "r1",
        "identity": None, "embedders": {name: "rev" for name in medians},
        "centroids": {name: f"centroid-{name}.npy" for name in medians},
        "distribution": {"kind": "loo-segments",
                         **{name: {"median": m, "p02": 0.1, "p10": 0.2}
                            for name, m in medians.items()}},
        "windows": {}, "mode_check": [], "states": {"neutral": neutral},
    }  # fmt: skip
    write_json(root, f"targets/{TARGET}/target.json", record)


def write_teacher_score_plan(root: root_mod.Root, target: str | None = TARGET) -> None:
    """The teacher run's score plan: its floor from the seed target, or the seed fallback's."""
    floor = (
        {"from": teacher_cmd.FLOOR_SEED, "target": target}
        if target is not None
        else {"from": teacher_cmd.FLOOR_TAKES, "target": None, "percentile": 10.0}
    )
    plan = {"schema": 1, "run": TAKES_RUN, "lines_file": "text/v1/lines.jsonl", "floor": floor}
    write_json(root, f"{teacher_cmd.run_rel(TAKES_RUN)}/{teacher_cmd.SCORE_PLAN_NAME}", plan)


@pytest.fixture
def root(private_root: root_mod.Root) -> root_mod.Root:
    anchor = write_anchor(private_root)
    plan = {"schema": 1, "run": TAKES_RUN, "anchor": anchor.to_json()}
    write_json(private_root, f"{teacher_cmd.run_rel(TAKES_RUN)}/generate-plan.json", plan)
    write_teacher_score_plan(private_root)
    write_target(private_root)
    write_corpus(private_root)
    write_cache(private_root)
    write_checkpoint(private_root)
    write_train(private_root)
    return private_root


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch, root: root_mod.Root) -> list[tuple[str, str | None]]:
    """A fake `huggingface_hub` over the root's cache, as a job sees it at `/cache`."""
    calls: list[tuple[str, str | None]] = []

    def snapshot_download(
        repo_id: str, *, revision: str | None = None, local_files_only: bool = False,
        **kwargs: object,
    ) -> str:  # fmt: skip
        assert local_files_only
        calls.append((repo_id, revision))
        model = {"repository": repo_id, "revision": revision}
        return str(snapshot(root, model))

    module = types.ModuleType("huggingface_hub")
    module.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return calls


# The student, the ASR and the prosody.


@dataclass
class FakeSynth:
    """A loaded student: each line is a synthetic voice whose length is unique to the line."""

    f0: float = 205.0
    formants: Sequence[float] = FORMANTS
    sample_rate: int = RATE
    loaded: list[tuple[Path, Path, str]] = field(default_factory=list)
    voices: list[Path] = field(default_factory=list)
    seeds: dict[str, int] = field(default_factory=dict)
    by_length: dict[int, str] = field(default_factory=dict)

    def load(self, weights: Path, tokenizer: Path, device: str) -> FakeSynth:
        self.loaded.append((weights, tokenizer, device))
        return self

    def voice(self, clip: Path) -> object:
        self.voices.append(clip)
        return ("voice", clip.name)

    def render(self, voice: object, text: str, seed: int) -> np.ndarray:
        assert voice == ("voice", "neutral.wav")
        seconds = 1.0 + 0.01 * len(self.by_length)
        audio = _synth_voice(self.f0, self.formants, seconds, RATE)
        self.by_length[len(audio)] = text
        self.seeds[text] = seed
        return audio


class LengthAsr:
    """An ASR that hears each render's line by its length; it mishears the lines `misheard` has.

    `misheard` is a set of line texts, or `ALL` for every line.
    """

    name = "whisper-large-v3-turbo"

    def __init__(self, synth: FakeSynth, misheard: frozenset[str] = frozenset()) -> None:
        self.synth = synth
        self.misheard = misheard

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Transcript:
        text = self.synth.by_length[len(audio)]
        wrong = self.misheard is ALL or text in self.misheard
        return Transcript(text=MISHEARD if wrong else text, words=())


ALL: frozenset[str] = frozenset({"every line"})
MISHEARD = "a different sentence altogether"


def prosody_for(render_sd: float):
    def measure(audio: np.ndarray, sample_rate: int) -> Prosody:
        return Prosody(f0_mean_hz=200.0, f0_std_hz=render_sd, rate_sps=4.0, energy_db=-20.0)

    return measure


EMBEDDERS = (FakeEmbedder("wavlm-sv"), FakeEmbedder("ecapa"))


def models(synth: FakeSynth, misheard: frozenset[str] = frozenset(), render_sd: float = 20.0):
    return score_stage.Models(
        embedders=EMBEDDERS,
        asr=LengthAsr(synth, misheard),
        arousal=FakeArousal(),
        prosody=prosody_for(render_sd),
    )


# The jobs and the command line.


class StageDocker:
    """A `DockerOps` whose `run_logged` runs the job's stage in this process."""

    def __init__(self, root: root_mod.Root, synth: FakeSynth, scoring: score_stage.Models) -> None:
        self.root = root
        self.synth = synth
        self.scoring = scoring
        self.argvs: list[list[str]] = []
        self.free_mib = 16000

    def capture(self, argv: Sequence[str]) -> str:
        if list(argv[:3]) == ["docker", "ps", "-q"]:
            return ""
        assert "nvidia-smi" in argv
        return f"{self.free_mib}\n"

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
            mp.setattr(module_obj, "load_synth", self.synth.load)
            mp.setattr(module_obj, "load_models", lambda device: self.scoring)
            mp.setattr(module_obj.fetch, "prepare", lambda *names: None)
            return module_obj.main(rest)


@dataclass
class Result:
    code: int
    out: str
    err: str
    docker: StageDocker

    def jobs(self) -> list[dict]:
        folders = sorted((self.docker.root.path / "runs").glob("job-*/job.json"))
        return [json.loads(p.read_text(encoding="utf-8")) for p in folders]

    def logs(self) -> str:
        folders = sorted((self.docker.root.path / "runs").glob("job-*/log.txt"))
        return "".join(p.read_text(encoding="utf-8") for p in folders)


def student(
    root: root_mod.Root, *args: str, synth: FakeSynth | None = None, **model_args: object
) -> Result:
    synth = synth or FakeSynth()
    docker = StageDocker(root, synth, models(synth, **model_args))  # type: ignore[arg-type]
    out, err = _Capture(), _Capture()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(student_cmd, "make_docker", lambda: docker)
        mp.setattr(student_cmd, "image_tag", lambda family: f"mist-voice-{family}:0123456789ab")
        mp.delenv(jobs.GPU_PEERS_ENV, raising=False)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(["--root", str(root.path), "student", *args])
    result = Result(code, out.getvalue(), err.getvalue(), docker)
    printed = result.out + result.err
    _assert_no_canary(printed + result.logs())
    assert ".wav" not in printed and ".flac" not in printed
    return result


def score(root: root_mod.Root, *args: str, **kwargs: object) -> Result:
    return student(root, "score", "--run", RUN, *args, **kwargs)  # type: ignore[arg-type]


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def scores_dir(root: root_mod.Root, checkpoint: str = CHECKPOINT) -> Path:
    return root.path / student_cmd.scores_rel(RUN, checkpoint)


# `student score`.


def test_score_renders_every_line_then_scores_and_passes(
    root: root_mod.Root, hub: list[tuple[str, str | None]]
) -> None:
    synth = FakeSynth()
    result = score(root, "--checkpoint", CHECKPOINT, synth=synth)
    assert result.code == 0, result.err
    doc = read(root.path / "students" / RUN / "score.json")
    assert doc == read(scores_dir(root) / "score.json")
    assert doc["pass"] is True and doc["failed"] == []
    assert doc["cer"] == 0.0
    assert doc["lines"] == {"valid": VALID_N, "bench": BENCH_N}
    assert BENCH_N == 26
    assert doc["cer_mean"] == 0.0
    # The F0 SD ratio is the renders' median over the seed target's neutral profile.
    assert doc["f0_sd_ratio"] == 1.0
    assert doc["f0"] == {"seed_sd_hz": SEED_F0_SD, "seed_sd_spread_hz": SEED_F0_SPREAD,
                         "render_sd_median_hz": 20.0}  # fmt: skip
    # The SIM ratio is the renders' median SIM to the anchor over the seed target's median.
    lines = [json.loads(x) for x in (scores_dir(root) / "lines.jsonl").read_text().splitlines()]
    for name in ("wavlm-sv", "ecapa"):
        render_median = float(np.median([line["sim"][name] for line in lines]))
        entry = doc["sim"][name]
        assert entry["seed_median"] == SEED_SIM_MEDIANS[name]
        assert entry["render_median"] == pytest.approx(render_median, abs=1e-6)
        assert entry["ratio"] == pytest.approx(render_median / SEED_SIM_MEDIANS[name], abs=1e-5)
    assert doc["sim_ratio"] == min(doc["sim"][name]["ratio"] for name in ("wavlm-sv", "ecapa"))
    assert doc["sim_ratio"] >= 0.9
    assert doc["seed"] == {"target": TARGET, "state": "neutral"}
    assert doc["gates"] == {"cer_max": 0.03, "sim_median_ratio_min": 0.9, "f0_std_tolerance": 0.2}
    assert doc["checkpoint"] == CHECKPOINT
    weights = root.path / student_cmd.weights_rel(RUN, CHECKPOINT)
    assert doc["weights_sha256"] == sha256(weights.read_bytes())
    # Every validation line and bench line rendered once, in the anchor's voice, seeded by id.
    texts = sorted(synth.by_length.values())
    assert len(texts) == VALID_N + BENCH_N
    assert {valid_text(k) for k in range(1, VALID_N + 1)} <= set(texts)
    assert synth.seeds[valid_text(1)] == student_cmd.render_seed("valid-0001")
    assert synth.voices == [root.path / "voices" / ANCHOR / "neutral.wav"]
    assert synth.loaded[0][0] == weights
    assert (
        synth.loaded[0][1] == snapshot(root, RUNTIME_TOKENIZER) / "languages/english/tokenizer.json"
    )
    assert len(lines) == VALID_N + BENCH_N
    assert "pass" in result.out and "26 bench lines" in result.out


def test_score_runs_a_4_gb_render_in_train_pocket_and_scores_in_the_score_family(
    root: root_mod.Root, hub: list[tuple[str, str | None]]
) -> None:
    result = score(root, "--checkpoint", CHECKPOINT)
    assert result.code == 0, result.err
    render, scoring = result.jobs()
    assert (render["family"], render["vram_gb"]) == ("train-pocket", 4.0)
    assert render["module"] == scoring["module"] == "scripts.voice_data.stages.student_score"
    assert render["args"][0] == "render" and scoring["args"][0] == "score"
    assert render["gpu"] and scoring["gpu"] and not render["network"] and not scoring["network"]
    # The scoring job keeps the score family's own VRAM need.
    assert scoring["family"] == "score" and scoring["vram_gb"] is None
    assert FAMILIES["score"]["vram_gb"] == 6
    rendered = {i["rel"] for i in render["inputs"]}
    assert rendered == {
        student_cmd.weights_rel(RUN, CHECKPOINT),
        f"corpus/{CORPUS}/valid_aligned.jsonl",
        f"voices/{ANCHOR}/neutral.wav",
    }
    # The scoring reads the lines' texts and the anchor's clip; the seed's numbers are in the plan.
    scored = {i["rel"] for i in scoring["inputs"]}
    assert scored == {f"corpus/{CORPUS}/valid_aligned.jsonl", f"voices/{ANCHOR}/neutral.wav"}
    plan = read(scores_dir(root) / "score-plan.json")
    assert plan["seed"] == {
        "target": TARGET, "state": "neutral", "sim_median": SEED_SIM_MEDIANS,
        "f0_std_hz": SEED_F0_SD, "f0_std_spread_hz": SEED_F0_SPREAD,
    }  # fmt: skip
    assert render["outputs"] == scoring["outputs"] == student_cmd.scores_rel(RUN, CHECKPOINT)
    # The render's tokenizer was looked up offline at the pinned revision.
    assert hub == [(RUNTIME_TOKENIZER["repository"], RUNTIME_TOKENIZER["revision"])]


def test_score_preflight_needs_only_4_gb_for_the_render(
    root: root_mod.Root, hub: list[tuple[str, str | None]]
) -> None:
    synth = FakeSynth()
    docker = StageDocker(root, synth, models(synth))
    docker.free_mib = 5 * 1024  # over the render's 4 GB, under the score family's 6
    out, err = _Capture(), _Capture()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(student_cmd, "make_docker", lambda: docker)
        mp.setattr(student_cmd, "image_tag", lambda family: f"mist-voice-{family}:0123456789ab")
        mp.delenv(jobs.GPU_PEERS_ENV, raising=False)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(["--root", str(root.path), "student", "score", "--run", RUN])
    assert code != 0
    assert "family score needs 6" in err.getvalue()
    assert (scores_dir(root) / "renders.json").is_file()


def test_score_defaults_to_the_newest_checkpoint_with_weights(
    root: root_mod.Root, hub: list[tuple[str, str | None]]
) -> None:
    write_checkpoint(root, NEWER, fill=9)
    (root.path / f"students/{RUN}/checkpoints/weights/checkpoint_00009999").mkdir(parents=True)
    result = score(root)
    assert result.code == 0, result.err
    assert read(root.path / "students" / RUN / "score.json")["checkpoint"] == NEWER
    assert (scores_dir(root, NEWER) / "score.json").is_file()
    assert NEWER in result.out


@pytest.mark.parametrize(
    ("kwargs", "medians", "failed"),
    [
        ({"misheard": ALL}, SEED_SIM_MEDIANS, ["cer"]),
        # A seed median above 1 is one no cosine reaches, on one embedder only.
        ({}, {"wavlm-sv": 0.5, "ecapa": 1.5}, ["sim_ratio"]),
        ({"render_sd": 30.0}, SEED_SIM_MEDIANS, ["f0_sd_ratio"]),
        ({"render_sd": 15.0}, SEED_SIM_MEDIANS, ["f0_sd_ratio"]),
    ],
    ids=["cer", "sim", "f0-high", "f0-low"],
)
def test_a_failed_gate_writes_the_score_and_exits_3(
    root: root_mod.Root,
    hub: list[tuple[str, str | None]],
    kwargs: dict,
    medians: dict[str, float],
    failed: list[str],
) -> None:
    write_target(root, medians=medians)
    result = score(root, "--checkpoint", CHECKPOINT, **kwargs)
    assert result.code == 3, result.err
    doc = read(root.path / "students" / RUN / "score.json")
    assert doc["pass"] is False and doc["failed"] == failed
    assert f"fail: {failed[0]}" in result.out
    assert failed[0] in result.err


def test_the_cer_gate_holds_the_highest_lines_cer_and_records_the_mean(
    root: root_mod.Root, hub: list[tuple[str, str | None]]
) -> None:
    # `bench.cer_max` is the highest CER (its note in gates_default.json): one misheard line of
    # 29 fails the gate though the mean stays under it.
    result = score(root, "--checkpoint", CHECKPOINT, misheard=frozenset({valid_text(2)}))
    assert result.code == 3, result.err
    doc = read(root.path / "students" / RUN / "score.json")
    assert doc["failed"] == ["cer"]
    assert doc["cer_line_max_id"] == "valid-0002"
    lines = [json.loads(x) for x in (scores_dir(root) / "lines.jsonl").read_text().splitlines()]
    assert doc["cer"] == max(line["cer"] for line in lines) > 0.03
    mean = sum(line["cer"] for line in lines) / len(lines)
    assert doc["cer_mean"] == pytest.approx(mean, abs=1e-6)
    assert doc["cer"] > doc["cer_mean"] > 0.0


def test_the_bench_lines_are_its_lines_and_acknowledgements(tmp_path: Path) -> None:
    lines = student_cmd.read_bench_lines(student_cmd.bench_lines_path())
    ids = [line_id for line_id, _ in lines]
    expected = [str(item["id"]) for item in [*BENCH_DOC["lines"], *BENCH_DOC["acknowledgements"]]]
    assert ids == expected and len(ids) == 26
    path = tmp_path / "lines.json"
    path.write_text(json.dumps({"lines": [{"id": "L01", "text": "one"}]}), encoding="utf-8")
    with pytest.raises(student_cmd.InputError, match="acknowledgements"):
        student_cmd.read_bench_lines(path)
    repeated = {
        "lines": [{"id": "L01", "text": "one"}],
        "acknowledgements": [{"id": "L01", "text": "two"}],
    }
    path.write_text(json.dumps(repeated), encoding="utf-8")
    with pytest.raises(student_cmd.InputError, match="repeat"):
        student_cmd.read_bench_lines(path)


def test_gate_failures_holds_each_gate_at_its_boundary() -> None:
    gates = {"cer_max": 0.03, "sim_median_ratio_min": 0.9, "f0_std_tolerance": 0.2}
    edge = {"cer": 0.03, "sim_ratio": 0.9, "f0_sd_ratio": 1.2}
    assert student_cmd.gate_failures(edge, gates) == []
    assert student_cmd.gate_failures({**edge, "f0_sd_ratio": 0.8}, gates) == []
    over = {"cer": 0.031, "sim_ratio": 0.899, "f0_sd_ratio": 1.21}
    assert student_cmd.gate_failures(over, gates) == ["cer", "sim_ratio", "f0_sd_ratio"]
    missing = {"cer": None, "sim_ratio": float("nan"), "f0_sd_ratio": True}
    assert student_cmd.gate_failures(missing, gates) == ["cer", "sim_ratio", "f0_sd_ratio"]


def _no_train(root: root_mod.Root) -> None:
    (root.path / "students" / RUN / "train.json").unlink()


def _no_weights(root: root_mod.Root) -> None:
    (root.path / student_cmd.weights_rel(RUN, CHECKPOINT)).unlink()


def _tokenizer_not_cached(root: root_mod.Root) -> None:
    write_train(root, tokenizer=b"a tokenizer the cache does not hold")


def _anchor_changed(root: root_mod.Root) -> None:
    # The anchor's clip is rewritten, consistently, after the teacher run planned it.
    clip = _synth_voice(180.0, FORMANTS, ANCHOR_SECONDS, RATE)
    make_identity(root.path / "voices", ANCHOR, {"neutral": clip})


def _no_validation_lines(root: root_mod.Root) -> None:
    folder = root.path / "corpus" / CORPUS
    (folder / "valid_aligned.jsonl").write_text("", encoding="utf-8")
    manifest = read(folder / "manifest.json")
    manifest["files"][1]["sha256"] = sha256(b"")
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _seed_fallback(root: root_mod.Root) -> None:
    write_teacher_score_plan(root, target=None)


def _not_scored(root: root_mod.Root) -> None:
    (root.path / teacher_cmd.run_rel(TAKES_RUN) / teacher_cmd.SCORE_PLAN_NAME).unlink()


def _identity_target(root: root_mod.Root) -> None:
    write_target(root, mode="identity")


def _another_voice(root: root_mod.Root) -> None:
    write_target(root, voice_id="another-voice")


def _no_sim_median(root: root_mod.Root) -> None:
    write_target(root, medians={"wavlm-sv": 0.5})


def _no_f0_sd(root: root_mod.Root) -> None:
    write_target(root, f0_std_hz=0.0)


@pytest.mark.parametrize(
    ("breaks", "message"),
    [
        (_no_train, "train.json"),
        (_no_weights, "no weights"),
        (_tokenizer_not_cached, "tokenizer"),
        (_anchor_changed, "changed"),
        (_no_validation_lines, "no validation line"),
        (_seed_fallback, "seed fallback"),
        (_not_scored, "no score plan"),
        (_identity_target, "no seed distribution"),
        (_another_voice, "not the voice"),
        (_no_sim_median, "no ecapa SIM median"),
        (_no_f0_sd, "no neutral F0 SD"),
    ],
    ids=[
        "train", "weights", "tokenizer", "anchor", "validation", "seed-fallback", "unscored",
        "identity-target", "other-voice", "sim-median", "f0-sd",
    ],
)  # fmt: skip
def test_score_refusals_exit_2_before_any_job(
    root: root_mod.Root, hub: list[tuple[str, str | None]], breaks, message: str
) -> None:
    breaks(root)
    result = score(root, "--checkpoint", CHECKPOINT)
    assert result.code == 2 and message in result.err, result.err
    assert result.docker.argvs == []


def test_score_refuses_a_run_without_checkpoints(
    root: root_mod.Root, hub: list[tuple[str, str | None]]
) -> None:
    _no_weights(root)
    result = score(root)
    assert result.code == 2 and "no distill checkpoint" in result.err


def test_the_render_refuses_weights_that_changed_since_the_plan(
    root: root_mod.Root, hub: list[tuple[str, str | None]], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The stage checks the weights against the host's plan before it loads them.
    weights = root.path / student_cmd.weights_rel(RUN, CHECKPOINT)
    write_plan = student_cmd.write_json

    def write_then_change(root_arg: root_mod.Root, rel: str, payload: object) -> None:
        write_plan(root_arg, rel, payload)
        if rel.endswith(student_cmd.SCORE_PLAN_NAME):
            weights.write_bytes(safetensors_bytes(LAYOUT, fill=8))

    monkeypatch.setattr(student_cmd, "write_json", write_then_change)
    synth = FakeSynth()
    result = score(root, "--checkpoint", CHECKPOINT, synth=synth)
    assert result.code == 4, result.err
    assert synth.loaded == []
    assert "weights" in result.logs()
    assert not (scores_dir(root) / "renders.json").exists()


def test_runtime_config_loads_the_student_offline() -> None:
    installed = yaml.safe_load(
        """\
weights_path: hf://kyutai/pocket-tts/languages/english/model.safetensors@db04f0e6
weights_path_without_voice_cloning: hf://kyutai/pocket-tts-without-voice-cloning/languages/english/model.safetensors@a7056f50
flow_lm:
  lookup_table:
    tokenizer: sentencepiece
    tokenizer_path: hf://kyutai/pocket-tts-without-voice-cloning/languages/english/tokenizer.json@00eac05e
mimi:
  channels: 1
"""
    )
    config = stage.runtime_config(installed, Path("/data/w.safetensors"), Path("/cache/t.json"))
    assert config["weights_path"] == str(Path("/data/w.safetensors"))
    assert config["weights_path_without_voice_cloning"] is None
    assert config["flow_lm"]["lookup_table"]["tokenizer_path"] == str(Path("/cache/t.json"))
    assert config["mimi"] == {"channels": 1}
    assert installed["weights_path"].startswith("hf://")  # the installed copy is untouched
    installed["mimi"]["weights"] = "hf://kyutai/another/file.safetensors@0"
    with pytest.raises(stage.StageError, match="remote"):
        stage.runtime_config(installed, Path("/data/w"), Path("/cache/t"))
    with pytest.raises(stage.StageError, match="tokenizer_path"):
        stage.runtime_config({"weights_path": "x"}, Path("/data/w"), Path("/cache/t"))


def test_the_stage_writes_nothing_and_exits_1_without_a_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fetch, "prepare", lambda *names: None)
    code = stage.main(["--data", str(tmp_path), "render", "--run", RUN, "--checkpoint", CHECKPOINT,
                       "--device", "cpu"])  # fmt: skip
    assert code == 1
    assert not any(tmp_path.rglob("outputs.json"))


# `student export`.


def export(root: root_mod.Root, *args: str) -> Result:
    return student(
        root, "export", "--run", RUN, "--checkpoint", CHECKPOINT, "--voice-id", VOICE,
        "--anchor", ANCHOR, *args,
    )  # fmt: skip


def stage_models_dir(tmp_path: Path, tokenizer: bytes = STOCK_TOKENIZER) -> Path:
    """The realtime models folder holding the stock tokenizer where the adapter stages it."""
    models_dir = tmp_path / "models"
    path = pocket_tts._local_path(pocket_tts.TOKENIZER_URI, models_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(tokenizer)
    return models_dir


def test_an_exported_identity_loads_in_the_realtime_loader(
    root: root_mod.Root, tmp_path: Path
) -> None:
    result = export(root)
    assert result.code == 0, result.err
    directory = root.path / "voices" / VOICE
    weights = root.path / student_cmd.weights_rel(RUN, CHECKPOINT)
    identity = voice_assets.load_identity(root.path / "voices", VOICE)
    assert identity.status == "placeholder"
    assert [clip.state for clip in identity.clips] == ["neutral"]
    model = identity.model
    assert model is not None
    assert (model.engine, model.base) == (pocket_tts.ENGINE_ID, pocket_tts.MODEL_PIN)
    assert model.sha256 == sha256(weights.read_bytes())
    assert model.tokenizer_path is None
    assert model.tokenizer_sha256 == sha256(STOCK_TOKENIZER)
    pin = pocket_tts._model_pin(identity, stage_models_dir(tmp_path))
    assert pin == f"{pocket_tts.MODEL_PIN}+{model.sha256}"
    # The anchor's neutral clip and transcript are copied in unchanged.
    anchor = root.path / "voices" / ANCHOR
    assert (directory / "neutral.wav").read_bytes() == (anchor / "neutral.wav").read_bytes()
    assert (directory / "neutral.private.txt").read_bytes() == CANARY.transcript.encode()
    assert not (directory / "tokenizer.json").exists()
    assert "schema 2" in result.out and "status placeholder" in result.out
    assert "3 tensors" in result.out and "stock" in result.out


def test_export_writes_a_frozen_identity_when_asked(root: root_mod.Root) -> None:
    result = export(root, "--status", "frozen")
    assert result.code == 0, result.err
    assert voice_assets.load_identity(root.path / "voices", VOICE).status == "frozen"


def test_export_copies_a_training_tokenizer_that_is_not_the_stock_one(
    root: root_mod.Root, tmp_path: Path
) -> None:
    write_train(root, tokenizer=BASE_TOKENIZER)
    result = export(root)
    assert result.code == 0, result.err
    identity = voice_assets.load_identity(root.path / "voices", VOICE)
    model = identity.model
    assert model is not None and model.tokenizer_path is not None
    assert model.tokenizer_path.read_bytes() == BASE_TOKENIZER
    assert model.tokenizer_sha256 == sha256(BASE_TOKENIZER)
    # The identity carries its tokenizer, so the stock one is not consulted.
    pin = pocket_tts._model_pin(identity, tmp_path / "no-models")
    assert pin == f"{pocket_tts.MODEL_PIN}+{model.sha256}"
    assert "copied into the identity" in result.out


def test_the_realtime_adapter_refuses_an_identity_whose_tokenizer_it_lacks(
    root: root_mod.Root, tmp_path: Path
) -> None:
    # The stock tokenizer staged for the runtime is another file than the one trained with.
    assert export(root).code == 0
    identity = voice_assets.load_identity(root.path / "voices", VOICE)
    with pytest.raises(voice_assets.VoiceAssetError, match="tokenizer_sha256"):
        pocket_tts._model_pin(identity, stage_models_dir(tmp_path, b"another tokenizer"))


def test_export_refuses_a_tokenizer_the_cache_does_not_hold(root: root_mod.Root) -> None:
    write_train(root, tokenizer=b"a tokenizer the cache does not hold")
    result = export(root)
    assert result.code == 2 and "tokenizer" in result.err
    assert not (root.path / "voices" / VOICE).exists()


@pytest.mark.parametrize(
    ("layout", "detail"),
    [
        ({k: v for k, v in LAYOUT.items() if k != "mimi.conv.weight"}, "1 missing"),
        ({**LAYOUT, "flow_lm.extra.weight": (2,)}, "1 extra"),
        ({**LAYOUT, "flow_lm.layer.weight": (2, 4)}, "1 reshaped"),
    ],
    ids=["missing", "extra", "reshaped"],
)
def test_export_refuses_weights_whose_layout_is_not_the_stock_students(
    root: root_mod.Root, layout: dict, detail: str
) -> None:
    write_checkpoint(root, layout=layout)
    result = export(root)
    assert result.code == 2 and detail in result.err, result.err
    assert not (root.path / "voices" / VOICE).exists()


def test_export_refuses_a_file_that_is_not_safetensors(root: root_mod.Root) -> None:
    (root.path / student_cmd.weights_rel(RUN, CHECKPOINT)).write_bytes(b"\xff" * 16)
    result = export(root)
    assert result.code == 2 and "not a safetensors file" in result.err


def test_export_refuses_an_existing_identity_and_a_missing_checkpoint(
    root: root_mod.Root,
) -> None:
    (root.path / "voices" / VOICE).mkdir(parents=True)
    result = export(root)
    assert result.code == 2 and "already exists" in result.err
    result = student(
        root, "export", "--run", RUN, "--checkpoint", NEWER, "--voice-id", "other-voice",
        "--anchor", ANCHOR,
    )  # fmt: skip
    assert result.code == 2 and "no weights" in result.err
    assert not (root.path / "voices" / "other-voice").exists()


def test_export_refuses_stock_weights_that_are_not_cached(root: root_mod.Root) -> None:
    (snapshot(root, POCKET) / student_train.STUDENT_WEIGHTS[1]).unlink()
    result = export(root)
    assert result.code == 2 and "not in the cache" in result.err


# The names host and runtime share.


def test_runtime_pin_reads_the_realtime_adapters_pin() -> None:
    expected = (pocket_tts.ENGINE_ID, pocket_tts.REPOSITORY, pocket_tts.REVISION)
    assert student_cmd.runtime_pin() == expected
    assert (POCKET["repository"], POCKET["revision"]) == expected[1:]


def test_the_cached_files_are_the_ones_training_used() -> None:
    assert student_cmd.STOCK_WEIGHTS == student_train.STUDENT_WEIGHTS
    assert student_cmd.RUNTIME_TOKENIZER == student_train.STUDENT_TOKENIZER
    assert student_cmd.BASE_TOKENIZER == student_train.BASE_TOKENIZER


def test_tensor_layout_skips_the_metadata_entry(tmp_path: Path) -> None:
    path = tmp_path / "w.safetensors"
    path.write_bytes(safetensors_bytes(LAYOUT))
    assert student_cmd.tensor_layout(path, "w") == {k: tuple(v) for k, v in LAYOUT.items()}
    assert fetch.sha256_file(path) == sha256(path.read_bytes())
