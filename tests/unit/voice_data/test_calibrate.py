"""`calibrate run|apply|check|lower`: the six retrieval values on public speakers (T34).

The rehearsal root is synthetic: a build plan whose `sources` list LibriTTS-R speakers, a build
record naming the speakers the episodes use, and a RAVDESS archive. One fake `DockerOps` answers
the job with a canned `calibration.json`; another runs `rehearsal/calibrate.py`'s `main` in this
process, with cheap fake embedders, on the plan the command wrote.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import io
import json
import re
import zipfile
import zlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import soundfile

from scripts.voice_data import cli, gates, jobs
from scripts.voice_data import root as root_mod
from scripts.voice_data.commands import calibrate as calibrate_cmd
from scripts.voice_data.records import write_json

IMAGE = "mist-voice-base:0123456789ab"
NAMES = ("wavlm-sv", "ecapa")
GATES = ("anchors.sim_min", "retrieval.anchor_floor", "retrieval.margin")
KEYS = tuple(f"{g}.{n}" for g in GATES for n in NAMES)
FLOORS = tuple(f"retrieval.anchor_floor.{n}" for n in NAMES)
FOLDER = "rehearsal/calibration/c1"
PLAN_REL = f"{FOLDER}/calibration-plan.json"
RESULT_REL = f"{FOLDER}/calibration.json"
OVERLAY_REL = "config/gates.json"
ARCHIVE_REL = "public/ravdess/Audio_Speech_Actors_01-24.zip"

# Calibration speakers: none appears in the build record, so each is listed with its files.
FREE = {"female": ("1111", "2222", "3333"), "male": ("4444", "5555", "6666")}
# Speakers the build record names: the impostor and the episodes' other speakers.
IMPOSTOR = "237"
EPISODE_OTHERS = (("121", "1580"), ("3575", "8463", "61"))
USED_SEX = {"237": "female", "121": "female", "1580": "female", "3575": "female"}
USED_SEX |= {"8463": "female", "61": "male"}
TARGET = "ex02"
LIBRI_FILES = 36
LIBRI_RATE = 4000

CANNED = {
    "anchors.sim_min.wavlm-sv": 0.7123,
    "anchors.sim_min.ecapa": 0.5512,
    "retrieval.anchor_floor.wavlm-sv": 0.6631,
    "retrieval.anchor_floor.ecapa": 0.4407,
    "retrieval.margin.wavlm-sv": 0.0215,
    "retrieval.margin.ecapa": -0.0132,
}


# ---------------------------------------------------------------------------
# The synthetic rehearsal root
# ---------------------------------------------------------------------------


def _tone(freq: float, seconds: float, rate: int, seed: int) -> np.ndarray:
    t = np.arange(int(round(seconds * rate))) / rate
    noise = np.random.default_rng(seed).normal(0, 0.02, len(t))
    return (0.3 * np.sin(2 * np.pi * freq * t) + noise).astype(np.float32)


def _fake_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _speaker(public: Path, speaker: str, sex: str, f0: float, audio: bool, freq: float) -> dict:
    folder = f"LibriTTS_R/test-clean/{speaker}"
    files = []
    count = LIBRI_FILES if audio else 3
    for index in range(count):
        rel = f"{speaker}_{index:03d}.wav"
        if audio:
            path = public / "librittsr" / folder / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            tone = _tone(freq, 3.0, LIBRI_RATE, seed=int(speaker) + index)
            soundfile.write(str(path), tone, LIBRI_RATE, subtype="PCM_16")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            digest = _fake_sha(f"{speaker}/{rel}")
        files.append([rel, digest])
    return {"speaker": speaker, "sex": sex, "f0_median_hz": f0, "dir": folder, "files": files}


def make_sources(public: Path, audio: bool) -> dict:
    """A `sources.json` that `parse_sources` accepts; only the free speakers get real audio."""
    styles = {}
    states = {"default": "neutral", "enunciated": "calm", "happy": "awake", "sad": "sleepy"}
    states["laughing"] = "excited"
    for style in states:
        name = f"base/{TARGET}_{style}_00001.wav"
        styles[style] = {"dir": f"expresso/read/{TARGET}/{style}", "files": [[name, "a" * 64]]}
    speakers = []
    for sex, ids in FREE.items():
        for number, speaker in enumerate(ids):
            freq = 200 + 90 * number + (40 if sex == "male" else 0)
            f0 = 210.0 if sex == "female" else 115.0
            speakers.append(_speaker(public, speaker, sex, f0, audio, freq))
    for speaker, sex in USED_SEX.items():
        f0 = 205.0 if sex == "female" else 120.0
        speakers.append(_speaker(public, speaker, sex, f0, False, 0.0))
    musan = {
        kind: {
            "dataset": "musan",
            "licence": "CC or public domain per file",
            "subset": kind,
            "folder": "public/musan",
            "collections": [
                {
                    "name": "x",
                    "licence_file": f"musan/{kind}/x/LICENSE",
                    "dir": f"musan/{kind}/x",
                    "files": [[f"{kind}-x-0000.wav", _fake_sha(kind)]],
                }
            ],
        }
        for kind in ("music", "noise")
    }
    return {
        "schema": 1,
        "file_fields": ["path", "sha256"],
        "target": {
            "dataset": "expresso",
            "licence": "CC BY-NC 4.0",
            "subset": f"read-{TARGET}",
            "folder": "public/expresso",
            "speaker": TARGET,
            "sex": "female",
            "f0_median_hz": 217,
            "states": states,
            "styles": styles,
        },
        "others": {
            "dataset": "librittsr",
            "licence": "CC BY 4.0",
            "subset": "test-clean",
            "folder": "public/librittsr",
            "clear_f0_min_hz": 168,
            "speakers": speakers,
        },
        **musan,
    }


def ravdess_names(actor: int) -> list[str]:
    names = []
    for emotion in range(1, 9):
        for intensity in (1,) if emotion == 1 else (1, 2):
            for statement in (1, 2):
                for repetition in (1, 2):
                    names.append(
                        f"Actor_{actor:02d}/03-01-{emotion:02d}-{intensity:02d}-"
                        f"{statement:02d}-{repetition:02d}-{actor:02d}.wav"
                    )
    return names


def write_ravdess(path: Path, actors: Sequence[int]) -> str:
    """Write a synthetic speech archive of the given actors; return its sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rate = 2000
    with zipfile.ZipFile(path, "w") as archive:
        for actor in actors:
            archive.writestr(f"Actor_{actor:02d}/", b"")
            for index, name in enumerate(ravdess_names(actor)):
                buffer = io.BytesIO()
                tone = _tone(150 + 60 * actor, 1.5, rate, seed=actor * 100 + index)
                soundfile.write(buffer, tone, rate, format="WAV", subtype="PCM_16")
                archive.writestr(name, buffer.getvalue())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_root(tmp_path: Path, audio: bool = False) -> root_mod.Root:
    """A rehearsal root holding the build plan, the build record and the RAVDESS archive."""
    root = root_mod.init_root(tmp_path / "stand-in", "rehearsal")
    sources = make_sources(root.path / "public", audio)
    write_json(root, "rehearsal/build-plan.json", {"schema": 1, "seed": 7, "sources": sources})
    build = {
        "schema": 1,
        "target": {"speaker": TARGET, "sex": "female"},
        "impostor": {"speaker": IMPOSTOR, "ecapa_sim": 0.81, "candidates": {IMPOSTOR: 0.81}},
        "episodes": [
            {"episode": f"ep{i + 1:02d}", "other_speakers": list(others)}
            for i, others in enumerate(EPISODE_OTHERS)
        ],
    }
    write_json(root, "rehearsal/build.json", build)
    archive = root.path / ARCHIVE_REL
    if audio:
        write_ravdess(archive, range(1, 7))
    else:
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(b"stand-in archive")
    return root


# ---------------------------------------------------------------------------
# The fake container boundary
# ---------------------------------------------------------------------------


class CannedDocker:
    """A `DockerOps` whose job prints two lines and writes a canned `calibration.json`."""

    def __init__(self, root: root_mod.Root, code: int = 0) -> None:
        self.root = root
        self.code = code
        self.argvs: list[list[str]] = []

    def capture(self, argv: Sequence[str]) -> str:
        raise AssertionError("the calibration job runs on the CPU, with no GPU preflight")

    def check(self, argv: Sequence[str]) -> None:
        raise AssertionError("calibrate builds no image")

    def run_logged(self, argv: Sequence[str], log: Path) -> int:
        self.argvs.append(list(argv))
        if self.code:
            log.write_text("error: librittsr has 2 eligible male speakers\n", encoding="utf-8")
            return self.code
        result = {"schema": 1, "id": "c1", "keys": dict(CANNED)}
        path = self.root.path / RESULT_REL
        path.write_text(json.dumps(result), encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        outputs = {"schema": 1, "files": [{"rel": RESULT_REL, "sha256": digest}]}
        (self.root.path / FOLDER / "outputs.json").write_text(json.dumps(outputs), "utf-8")
        log.write_text(
            "some library warning\ncalibrate: 42 files checked against the plan's sha256\n"
            "calibrate: key retrieval.anchor_floor.ecapa = 0.4407\n",
            encoding="utf-8",
        )
        return 0


class ToneEmbedder:
    """A cheap deterministic embedder: a fixed projection of the clip's log band spectrum."""

    def __init__(self, name: str) -> None:
        self.name = name
        rng = np.random.default_rng(zlib.crc32(name.encode("utf-8")))
        self._projection = rng.standard_normal((12, 32))

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        spectrum = np.abs(np.fft.rfft(np.asarray(audio, dtype=np.float64)[:8192], n=8192))
        bands = np.log(spectrum[:4096].reshape(32, 128).sum(axis=1) + 1e-9)
        vector = self._projection @ (bands - bands.mean())
        return (vector / np.linalg.norm(vector)).astype(np.float32)


class JobDocker(CannedDocker):
    """A `DockerOps` that runs the job module's `main` in this process, `--data` at the root."""

    def run_logged(self, argv: Sequence[str], log: Path) -> int:
        self.argvs.append(list(argv))
        at = list(argv).index("-m")
        module, rest = argv[at + 1], list(argv[at + 2 :])
        rest[1] = str(self.root.path)
        job = importlib.import_module(module)
        with (
            pytest.MonkeyPatch.context() as mp,
            log.open("w", encoding="utf-8") as handle,
            contextlib.redirect_stdout(handle),
            contextlib.redirect_stderr(handle),
        ):
            mp.setattr(job, "load_embedders", lambda: [ToneEmbedder(n) for n in NAMES])
            return job.main(rest)


@pytest.fixture
def docker_for(monkeypatch: pytest.MonkeyPatch):
    def install(docker: CannedDocker) -> CannedDocker:
        monkeypatch.setattr(calibrate_cmd, "make_docker", lambda: docker)
        monkeypatch.setattr(calibrate_cmd, "image_tag", lambda family: IMAGE)
        return docker

    return install


def run_cli(argv: Sequence[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


def calibrate(root: root_mod.Root, *args: str) -> tuple[int, str, str]:
    return run_cli(["--root", str(root.path), "calibrate", *args])


def overlay(root: root_mod.Root) -> dict:
    return json.loads((root.path / OVERLAY_REL).read_text(encoding="utf-8"))


def no_file_names(text: str) -> bool:
    return re.search(r"\.(wav|zip)\b|public/", text) is None


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def test_run_plans_public_speakers_runs_the_job_and_applies_its_values(
    tmp_path: Path, docker_for
) -> None:
    root = make_root(tmp_path)
    docker = docker_for(CannedDocker(root))
    code, out, err = calibrate(root, "run", "--id", "c1")
    assert code == 0, err
    plan = json.loads((root.path / PLAN_REL).read_text(encoding="utf-8"))
    sources = json.loads((root.path / "rehearsal/build-plan.json").read_text("utf-8"))["sources"]
    assert (plan["schema"], plan["kind"], plan["id"]) == (1, "calibration-plan", "c1")
    assert (plan["alpha"], plan["seed"]) == (0.01, calibrate_cmd.SEED)
    families = jobs.load_families()["score"]["models"]
    assert plan["embedders"] == {n: {"revision": families[n]["revision"]} for n in NAMES}
    assert plan["trials"] == {
        "piece_s": [3.0, 1.5, 6.0],
        "max_pieces": 100,
        "model_pieces": 30,
        "rival_pieces": 20,
        "cluster_n": [10, 20, 30],
        "reported_n": [5],
        "draws": 10,
        "anchor_k": [1, 2, 4],
        "bootstrap": 1000,
        "anchor_sets": 10,
    }
    libri, ravdess = plan["datasets"]
    assert (libri["name"], libri["licence"], libri["subset"]) == (
        "librittsr",
        "CC BY 4.0",
        "test-clean",
    )
    excluded = {row["speaker"]: (row["sex"], row["reason"]) for row in libri["excluded"]}
    used = {TARGET, IMPOSTOR} | {s for others in EPISODE_OTHERS for s in others}
    assert set(excluded) == used
    assert excluded[TARGET][0] == "female" and excluded["61"][0] == "male"
    assert "impostor" in excluded[IMPOSTOR][1] and "target" in excluded[TARGET][1]
    listed = {entry["speaker"]: entry for entry in libri["speakers"]}
    assert set(listed) == {s for ids in FREE.values() for s in ids}
    assert not set(listed) & used
    expected = {
        s["speaker"]: [[f"public/librittsr/{s['dir']}/{rel}", sha] for rel, sha in s["files"]]
        for s in sources["others"]["speakers"]
    }
    for speaker, entry in listed.items():
        assert entry["sex"] == ("female" if speaker in FREE["female"] else "male")
        assert entry["files"] == sorted(expected[speaker])
    assert ravdess == {
        "name": "ravdess",
        "licence": "CC BY-NC-SA 4.0",
        "subset": ravdess["subset"],
        "archive": [ARCHIVE_REL, calibrate_cmd.RAVDESS_ARCHIVE_SHA256],
    }
    assert calibrate_cmd.RAVDESS_ARCHIVE_SHA256 == (
        "5d208e01632cc3e5242106fa2af3273e6dc5239fb8143131979ac74c4aa40657"
    )
    [argv] = docker.argvs
    assert argv[argv.index("--network") + 1] == "none" and "--gpus" not in argv
    assert any(a.endswith(":/data/public:ro") for a in argv)
    at = argv.index("-m")
    assert argv[at + 1] == "scripts.voice_data.rehearsal.calibrate"
    assert argv[at + 2 :] == ["--data", "/data", "--plan", PLAN_REL]
    assert IMAGE in argv and "/opt/venv-score/bin/python" in argv
    assert overlay(root) == CANNED
    assert "calibrate: key retrieval.anchor_floor.ecapa = 0.4407" in out
    assert "library warning" not in out
    assert no_file_names(out + err)


def test_run_refuses_an_existing_calibration_folder(tmp_path: Path, docker_for) -> None:
    root = make_root(tmp_path)
    docker = docker_for(CannedDocker(root))
    (root.path / FOLDER).mkdir(parents=True)
    code, _, err = calibrate(root, "run", "--id", "c1")
    assert code == 2 and "exists" in err
    assert docker.argvs == []


def test_run_refuses_a_missing_archive(tmp_path: Path, docker_for) -> None:
    root = make_root(tmp_path)
    (root.path / ARCHIVE_REL).unlink()
    docker = docker_for(CannedDocker(root))
    code, _, err = calibrate(root, "run", "--id", "c1")
    assert code == 2 and docker.argvs == []
    assert not (root.path / FOLDER).exists()
    assert no_file_names(err)


def test_run_exits_4_and_prints_the_refusal_when_the_job_fails(tmp_path: Path, docker_for) -> None:
    root = make_root(tmp_path)
    docker_for(CannedDocker(root, code=1))
    code, out, _ = calibrate(root, "run", "--id", "c1")
    assert code == 4
    assert "2 eligible male speakers" in out
    assert not (root.path / OVERLAY_REL).exists()


def test_run_refuses_a_build_speaker_the_sources_do_not_list(tmp_path: Path, docker_for) -> None:
    root = make_root(tmp_path)
    build = json.loads((root.path / "rehearsal/build.json").read_text(encoding="utf-8"))
    build["episodes"][0]["other_speakers"].append("9999")
    write_json(root, "rehearsal/build.json", build)
    docker = docker_for(CannedDocker(root))
    code, _, err = calibrate(root, "run", "--id", "c1")
    assert code == 2 and "9999" in err and docker.argvs == []


@pytest.mark.parametrize("bad", ["-x", "a/b", "..", "a..b", "x" * 65])
def test_a_malformed_id_is_refused(tmp_path: Path, docker_for, bad: str) -> None:
    root = make_root(tmp_path)
    docker = docker_for(CannedDocker(root))
    code, _, _ = calibrate(root, "run", f"--id={bad}")
    assert code == 2 and docker.argvs == []


def test_the_job_runs_end_to_end_on_the_plan_run_wrote(
    tmp_path: Path, docker_for, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_root(tmp_path, audio=True)
    digest = hashlib.sha256((root.path / ARCHIVE_REL).read_bytes()).hexdigest()
    monkeypatch.setattr(calibrate_cmd, "RAVDESS_ARCHIVE_SHA256", digest)
    monkeypatch.setattr(calibrate_cmd, "TRIALS", {**calibrate_cmd.TRIALS, "bootstrap": 5})
    docker_for(JobDocker(root))
    code, out, err = calibrate(root, "run", "--id", "c1")
    assert code == 0, err + out
    result = json.loads((root.path / RESULT_REL).read_text(encoding="utf-8"))
    libri = result["datasets"][0]
    assert {sex: row["eligible"] for sex, row in libri["speakers"].items()} == {
        "female": 3,
        "male": 3,
    }
    assert set(result["keys"]) == set(KEYS)
    assert overlay(root) == result["keys"]
    for key in KEYS:
        assert f"calibrate: key {key} = " in out
    assert no_file_names(out + err)
    code, _, err = calibrate(root, "check", "--id", "c1")
    assert code == 0, err


# ---------------------------------------------------------------------------
# apply and check
# ---------------------------------------------------------------------------


def write_result(root: root_mod.Root, keys: dict | None = None) -> None:
    write_json(root, RESULT_REL, {"schema": 1, "id": "c1", "keys": keys or dict(CANNED)})


def test_apply_creates_a_missing_overlay(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    write_result(root)
    assert not (root.path / OVERLAY_REL).exists()
    code, out, err = calibrate(root, "apply", "--id", "c1")
    assert code == 0, err
    assert overlay(root) == CANNED
    loaded = gates.load(root.path)
    assert {k: loaded.value(k) for k in KEYS} == CANNED
    assert "retrieval.margin.ecapa = -0.0132" in out


def test_apply_keeps_every_other_overlay_key(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    write_result(root)
    write_json(root, OVERLAY_REL, {"segment.min_s": 1.25, "retrieval.margin.ecapa": 0.3})
    code, _, err = calibrate(root, "apply", "--id", "c1")
    assert code == 0, err
    assert overlay(root) == {**CANNED, "segment.min_s": 1.25}
    loaded = gates.load(root.path)
    assert loaded.value("segment.min_s") == 1.25
    assert loaded.value("retrieval.margin.ecapa") == -0.0132


def test_apply_refuses_a_missing_calibration(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    code, _, err = calibrate(root, "apply", "--id", "c1")
    assert code == 2 and "c1" in err
    assert not (root.path / OVERLAY_REL).exists()


def test_apply_refuses_a_calibration_without_the_six_keys(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    keys = dict(CANNED)
    del keys["retrieval.margin.ecapa"]
    write_result(root, keys)
    code, _, _ = calibrate(root, "apply", "--id", "c1")
    assert code == 2
    assert not (root.path / OVERLAY_REL).exists()


def test_check_exits_0_when_the_overlay_holds_the_six_values(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    write_result(root)
    write_json(root, OVERLAY_REL, {**CANNED, "segment.min_s": 1.25})
    code, out, err = calibrate(root, "check", "--id", "c1")
    assert code == 0, err
    assert "c1" in out


def test_check_exits_3_naming_each_differing_key(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    write_result(root)
    found = dict(CANNED)
    found["retrieval.anchor_floor.ecapa"] = 0.41
    del found["anchors.sim_min.wavlm-sv"]
    write_json(root, OVERLAY_REL, found)
    code, _, err = calibrate(root, "check", "--id", "c1")
    assert code == 3
    assert "retrieval.anchor_floor.ecapa" in err and "anchors.sim_min.wavlm-sv" in err
    for key in set(KEYS) - {"retrieval.anchor_floor.ecapa", "anchors.sim_min.wavlm-sv"}:
        assert key not in err


def test_check_exits_3_when_the_overlay_is_missing(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    write_result(root)
    code, _, err = calibrate(root, "check", "--id", "c1")
    assert code == 3
    for key in KEYS:
        assert key in err


# ---------------------------------------------------------------------------
# lower
# ---------------------------------------------------------------------------


def write_score(
    root: root_mod.Root,
    flag_floor: dict | None,
    thresholds: dict | None = None,
    run: str = "r3",
) -> None:
    retrieval = {
        "thresholds": dict(CANNED) if thresholds is None else thresholds,
        "impostor_episodes": {"primary": 0, "flagged": 0, "other": 3, "none": 0},
        "flag_floor": flag_floor,
    }
    write_json(root, f"rehearsal/scores/{run}/score.json", {"schema": 1, "retrieval": retrieval})


def calibrated(tmp_path: Path) -> root_mod.Root:
    root = make_root(tmp_path)
    write_result(root)
    write_json(root, OVERLAY_REL, {**CANNED, "segment.min_s": 1.25})
    return root


def test_lower_writes_the_lower_floor_and_touches_no_other_key(tmp_path: Path) -> None:
    root = calibrated(tmp_path)
    # WavLM-SV's floor falls to the flag floor; ECAPA's already sits under it and is kept.
    write_score(root, {"wavlm-sv": 0.612, "ecapa": 0.47})
    code, out, err = calibrate(root, "lower", "--score", "r3", "--id", "c1")
    assert code == 0, err
    assert overlay(root) == {
        **CANNED,
        "segment.min_s": 1.25,
        "retrieval.anchor_floor.wavlm-sv": 0.612,
    }
    assert "0.6631" in out and "0.612" in out and "0.4407" in out
    gates.load(root.path)


def test_lower_writes_both_floors_when_both_fall(tmp_path: Path) -> None:
    root = calibrated(tmp_path)
    write_score(root, {"wavlm-sv": 0.5, "ecapa": 0.3})
    code, _, err = calibrate(root, "lower", "--score", "r3", "--id", "c1")
    assert code == 0, err
    found = overlay(root)
    assert {k: found[k] for k in FLOORS} == {FLOORS[0]: 0.5, FLOORS[1]: 0.3}
    assert {k: v for k, v in found.items() if k not in FLOORS} == {
        **{k: v for k, v in CANNED.items() if k not in FLOORS},
        "segment.min_s": 1.25,
    }


def test_lower_refuses_a_null_flag_floor(tmp_path: Path) -> None:
    root = calibrated(tmp_path)
    write_score(root, None)
    before = overlay(root)
    code, _, err = calibrate(root, "lower", "--score", "r3", "--id", "c1")
    assert code == 2 and "flag floor" in err
    assert overlay(root) == before


def test_lower_refuses_a_score_run_at_other_values(tmp_path: Path) -> None:
    root = calibrated(tmp_path)
    write_score(root, {"wavlm-sv": 0.5, "ecapa": 0.3}, {**CANNED, "retrieval.margin.ecapa": 0.1})
    before = overlay(root)
    code, _, err = calibrate(root, "lower", "--score", "r3", "--id", "c1")
    assert code == 2 and "c1" in err
    assert overlay(root) == before


def test_lower_refuses_when_neither_floor_would_fall(tmp_path: Path) -> None:
    root = calibrated(tmp_path)
    # At or above both floors: one equal, one above.
    write_score(root, {"wavlm-sv": 0.6631, "ecapa": 0.5})
    before = overlay(root)
    code, _, err = calibrate(root, "lower", "--score", "r3", "--id", "c1")
    assert code == 2 and "neither" in err
    assert overlay(root) == before


def test_lower_refuses_a_missing_score(tmp_path: Path) -> None:
    root = calibrated(tmp_path)
    code, _, err = calibrate(root, "lower", "--score", "r9", "--id", "c1")
    assert code == 2 and "r9" in err


# ---------------------------------------------------------------------------
# Roots and the CLI
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        ("run", "--id", "c1"),
        ("apply", "--id", "c1"),
        ("check", "--id", "c1"),
        ("lower", "--score", "r3", "--id", "c1"),
    ],
)
def test_a_private_root_is_refused(
    private_root: root_mod.Root, docker_for, args: tuple[str, ...]
) -> None:
    docker = docker_for(CannedDocker(private_root))
    code, _, err = calibrate(private_root, *args)
    assert code == 2 and "rehearsal" in err
    assert docker.argvs == []
    assert not (private_root.path / "rehearsal").exists()


def test_the_cli_help_lists_calibrate() -> None:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    assert "calibrate" in out.getvalue()
