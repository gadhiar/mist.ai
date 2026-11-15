"""Tests for the rehearsal: `sources.json`, `rehearsal build`, `rehearsal answer`, `rehearsal score`.

Public sources are synthetic: harmonic pulse-train voices through formant resonators, written
where H5's downloads sit (`public/expresso`, `public/librittsr`, `public/musan`), with a
`sources.json` made from them. The job runs through the real `LocalDocker` with a fake `DockerOps`
that runs `rehearsal/mix.py` in this process (`--data` pointed at the root, where the container's
`/data` mounts point), with fake embedders in place of ECAPA and WavLM-SV. ffmpeg is real: the
episodes are encoded and decoded as H6 will.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib
import io
import json
import math
import shutil
import subprocess
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import lfilter

from scripts.voice_data import cli, gates, packs
from scripts.voice_data import rehearsal as pkg
from scripts.voice_data import root as root_mod
from scripts.voice_data.commands import rehearsal as rehearsal_cmd
from scripts.voice_data.errors import InputError
from scripts.voice_data.records import (
    Prosody,
    SegmentRecord,
    SourceRecord,
    sha256_file,
    write_json,
    write_jsonl,
)
from scripts.voice_data.rehearsal import mix
from scripts.voice_data.stages import ingest as ingest_stage
from scripts.voice_data.stages.score import spectrum_distance_db
from scripts.voice_data.states import SPEAKING_STATES

from .conftest import KEY_ENV_NAMES, Canaries
from .fakes import FakeEmbedder

IMAGE = "mist-voice-base:0123456789ab"
EPISODES = 3
MINUTES = 1.0
SEED = 7
STYLES = {
    "default": ("neutral", 215.0),
    "enunciated": ("calm", 205.0),
    "happy": ("awake", 235.0),
    "sad": ("sleepy", 195.0),
    "laughing": ("excited", 250.0),
}
TARGET_FORMANTS = ((700.0, 90.0), (1220.0, 110.0), (2600.0, 160.0))
# id: (sex, F0, formants, clips). 1221 is the target's twin with too few clips to be eligible.
OTHERS = {
    "121": ("female", 212.0, ((720.0, 90.0), (1250.0, 110.0), (2650.0, 160.0)), 4),
    "1221": ("female", 215.0, TARGET_FORMANTS, 2),
    "237": ("female", 206.0, ((500.0, 80.0), (1750.0, 120.0), (2950.0, 170.0)), 4),
    "1580": ("female", 213.0, ((880.0, 100.0), (1100.0, 100.0), (2400.0, 150.0)), 4),
    "1995": ("female", 174.0, ((400.0, 70.0), (2100.0, 130.0), (3100.0, 180.0)), 4),
    "3570": ("female", 174.0, ((620.0, 90.0), (1500.0, 110.0), (3300.0, 180.0)), 4),
    "3575": ("female", 207.0, ((950.0, 110.0), (1400.0, 120.0), (2300.0, 150.0)), 4),
    "3729": ("female", 183.0, ((450.0, 70.0), (1000.0, 100.0), (2800.0, 160.0)), 4),
    "6930": ("female", 138.0, ((560.0, 80.0), (1900.0, 120.0), (2700.0, 160.0)), 4),
    "61": ("male", 117.0, ((550.0, 80.0), (1100.0, 100.0), (2400.0, 150.0)), 4),
    "260": ("male", 120.0, ((650.0, 90.0), (1300.0, 110.0), (2300.0, 150.0)), 4),
    "672": ("male", 110.0, ((480.0, 70.0), (1600.0, 120.0), (2500.0, 160.0)), 4),
}
CANARY_SPEAKER = "237"
BREATH = 0.1


# ---------------------------------------------------------------------------
# Synthetic public sources
# ---------------------------------------------------------------------------


def voice(
    f0: float,
    formants: Sequence[tuple[float, float]],
    seconds: float,
    rate: int,
    seed: int,
) -> np.ndarray:
    """A pulse-train voice through formant resonators, with vibrato, breath noise and syllables.

    Linear filtering keeps it cheap at 48 kHz. The breath noise, at about twice the pulses' power,
    fills the spectrum between harmonics: without it the geometry test's deconvolution divides
    codec and music noise by near-empty bins, and its noise floor buries the reverb it measures.
    """
    n = int(seconds * rate)
    t = np.arange(n) / rate
    rng = np.random.default_rng(seed)
    phase = np.cumsum(f0 * (1.0 + 0.02 * np.sin(2.0 * np.pi * 5.0 * t + seed))) / rate
    source = np.diff(np.floor(phase), prepend=0.0) + BREATH * rng.standard_normal(n)
    out = source
    for centre, bandwidth in formants:
        r = math.exp(-math.pi * bandwidth / rate)
        out = lfilter(
            [1.0 - r], [1.0, -2.0 * r * math.cos(2.0 * math.pi * centre / rate), r * r], out
        )
    out = out * (0.6 + 0.4 * np.sin(2.0 * np.pi * 4.0 * t) ** 2)
    out = mix.fade(out, rate)
    return (0.3 * out / np.max(np.abs(out))).astype(np.float32)


def _write(path: Path, audio: np.ndarray, rate: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, rate, subtype="PCM_16", format="WAV")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_public(root: Path) -> dict:
    """Write the synthetic public sources under `root/public` and return their `sources.json`."""
    public = root / "public"
    exp_dir = "expresso/audio_48khz/read/ex02"
    styles = {}
    for s_index, (style, (_, f0)) in enumerate(STYLES.items()):
        files = []
        for i in range(10):
            rel = f"base/ex02_{style}_{i:05d}.wav"
            audio = voice(f0, TARGET_FORMANTS, 2.6 + 0.08 * i, 48000, 100 * s_index + i)
            files.append([rel, _write(public / "expresso" / exp_dir / style / rel, audio, 48000)])
        styles[style] = {"dir": f"{exp_dir}/{style}", "files": sorted(files)}
    speakers = []
    for n, (spk, (sex, f0, formants, count)) in enumerate(OTHERS.items()):
        chapter = Canaries().source_name if spk == CANARY_SPEAKER else "100"
        files = []
        for i in range(count):
            rel = f"{chapter}/{spk}_100_{i:06d}_000000.wav"
            audio = voice(f0, formants, 2.4 + 0.1 * i, 24000, 1000 + 10 * n + i)
            path = public / "librittsr" / "LibriTTS_R/test-clean" / spk / rel
            files.append([rel, _write(path, audio, 24000)])
        speakers.append(
            {
                "speaker": spk,
                "sex": sex,
                "f0_median_hz": f0,
                "dir": f"LibriTTS_R/test-clean/{spk}",
                "files": sorted(files),
            }
        )
    rng = np.random.default_rng(3)
    musan = {}
    for kind, count, seconds in (("music", 2, 20.0), ("noise", 3, 3.0)):
        files = []
        for i in range(count):
            t = np.arange(int(seconds * 16000)) / 16000
            if kind == "music":
                audio = sum(np.sin(2 * np.pi * f * t) for f in (196.0, 247.0, 294.0)) * 0.1
                audio = audio * (0.7 + 0.3 * np.sin(2 * np.pi * 0.5 * t))
            else:
                audio = 0.1 * rng.standard_normal(len(t))
            rel = f"{kind}-x-{i:04d}.wav"
            files.append([rel, _write(public / "musan" / "musan" / kind / "x" / rel, audio, 16000)])
        musan[kind] = {
            "dataset": "musan",
            "licence": "CC or public domain per file",
            "subset": kind,
            "folder": "public/musan",
            "collections": [
                {
                    "name": "x",
                    "licence_file": f"musan/{kind}/x/LICENSE",
                    "dir": f"musan/{kind}/x",
                    "files": files,
                }
            ],
        }
    return {
        "schema": 1,
        "file_fields": ["path", "sha256"],
        "target": {
            "dataset": "expresso",
            "licence": "CC BY-NC 4.0",
            "subset": "read-ex02",
            "folder": "public/expresso",
            "speaker": "ex02",
            "sex": "female",
            "f0_median_hz": 217,
            "states": {style: state for style, (state, _) in STYLES.items()},
            "unused_styles": ["confused", "longform", "narration", "singing", "whisper"],
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


# ---------------------------------------------------------------------------
# The fake container boundary
# ---------------------------------------------------------------------------


class StageDocker:
    """A `DockerOps` whose `run_logged` runs the job module in this process."""

    def __init__(self, root: root_mod.Root) -> None:
        self.root = root
        self.argvs: list[list[str]] = []

    def capture(self, argv: Sequence[str]) -> str:
        raise AssertionError("rehearsal jobs run on the CPU, with no GPU preflight")

    def check(self, argv: Sequence[str]) -> None:
        raise AssertionError("rehearsal builds no image")

    def run_logged(self, argv: Sequence[str], log: Path) -> int:
        self.argvs.append(list(argv))
        assert "--gpus" not in argv and argv[argv.index("--network") + 1] == "none"
        at = list(argv).index("-m")
        module, rest = argv[at + 1], list(argv[at + 2 :])
        rest[1] = str(self.root.path)
        with (
            log.open("w", encoding="utf-8") as handle,
            contextlib.redirect_stdout(handle),
            contextlib.redirect_stderr(handle),
        ):
            return importlib.import_module(module).main(rest)


@contextlib.contextmanager
def patched(docker: StageDocker, sources: Path) -> Iterator[None]:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(rehearsal_cmd, "make_docker", lambda: docker)
        mp.setattr(rehearsal_cmd, "image_tag", lambda family: IMAGE)
        mp.setattr(rehearsal_cmd, "SOURCES_PATH", sources)
        mp.setattr(mix, "load_ecapa", lambda: FakeEmbedder("ecapa"))
        mp.setattr(mix, "load_embedders", lambda: [FakeEmbedder("wavlm-sv"), FakeEmbedder("ecapa")])
        for name in KEY_ENV_NAMES:
            mp.setenv(name, Canaries().key)
        yield


def run_cli(argv: Sequence[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


@dataclass
class Built:
    root: root_mod.Root
    sources: Path
    docker: StageDocker
    code: int
    out: str
    err: str
    build: dict
    truth: list[pkg.TruthLine]


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Built:
    tmp = tmp_path_factory.mktemp("rehearsal")
    root = root_mod.init_root(tmp / "stand-in", "rehearsal")
    (root.path / "guard" / "terms.txt").write_text(Canaries().term + "\n", encoding="utf-8")
    sources = tmp / "sources.json"
    sources.write_text(json.dumps(make_public(root.path)), encoding="utf-8")
    docker = StageDocker(root)
    with patched(docker, sources):
        code, out, err = run_cli(
            ["--root", str(root.path), "rehearsal", "build", "--episodes", str(EPISODES),
             "--minutes", str(MINUTES), "--seed", str(SEED)]
        )  # fmt: skip
    assert code == 0, err + (root.path / "runs").as_posix()
    build = json.loads((root.path / pkg.BUILD_REL).read_text(encoding="utf-8"))
    return Built(
        root, sources, docker, code, out, err, build, pkg.read_truth(root.path / pkg.TRUTH_REL)
    )


def decode(path: Path, codec: str, channel: str) -> np.ndarray:
    argv = ingest_stage.decode_argv(str(path), codec, channel)
    proc = subprocess.run(argv, capture_output=True, check=True)
    return np.frombuffer(proc.stdout, dtype="<f4").astype(np.float64)


def dry_line(root: Path, clip: str) -> np.ndarray:
    audio, rate = mix.read_audio(root / clip)
    return mix.normalise(mix.in_show(mix.trim(mix.resample(audio, rate), mix.RATE), mix.RATE), 0.05)


# ---------------------------------------------------------------------------
# sources.json
# ---------------------------------------------------------------------------


def test_repository_sources_json_holds_the_selection() -> None:
    size = pkg.SOURCES_PATH.stat().st_size
    assert size < 600_000
    sources = pkg.load_sources()
    raw = sources.raw
    assert sources.target_speaker == "ex02" and sources.target_sex == "female"
    assert sources.states == {
        "default": "neutral",
        "enunciated": "calm",
        "happy": "awake",
        "sad": "sleepy",
        "laughing": "excited",
    }
    assert {len(files) for files in sources.target.values()} == {200}
    assert all("/longform/" not in p for files in sources.target.values() for p, _ in files)
    for group in ("target", "others", "music", "noise"):
        assert raw[group]["licence"] and raw[group]["dataset"] and raw[group]["subset"]
    female = [s for s in sources.others if s.sex == "female"]
    male = [s for s in sources.others if s.sex == "male"]
    assert len(female) == 20 and len(male) == 19
    assert all(len(s.files) <= 120 for s in female) and all(len(s.files) <= 40 for s in male)
    assert len(sources.clear_same_sex()) == 15
    assert len(sources.music) == 60 and len(sources.noise) == 120
    assert len(raw["music"]["collections"]) == 5 and len(raw["noise"]["collections"]) == 2
    assert all(c["licence_file"].endswith("/LICENSE") for c in raw["music"]["collections"])
    assert [
        s for s in mix.impostor_candidates(sources, pkg.DEFAULT_MINUTES) if s.speaker == "1221"
    ] == []


def _bad(spec: dict, edit: Callable[[dict], None]) -> dict:
    copy = json.loads(json.dumps(spec))
    edit(copy)
    return copy


@pytest.mark.parametrize(
    "edit",
    [
        lambda s: s["target"]["states"].update(default="calm"),
        lambda s: s["target"]["styles"]["sad"]["files"][0].__setitem__(1, "f" * 63),
        lambda s: s["target"]["styles"]["sad"]["files"].reverse(),
        lambda s: s["others"]["speakers"].__setitem__(
            slice(None), [x for x in s["others"]["speakers"] if x["speaker"] != "121"]
        ),
        lambda s: s["others"]["speakers"][0].pop("sex"),
        lambda s: s["music"].pop("licence"),
        lambda s: s["noise"]["collections"][0].pop("licence_file"),
        lambda s: s["others"].update(folder="elsewhere"),
        lambda s: s["target"]["styles"]["sad"]["files"][0].__setitem__(0, "../x.wav"),
    ],
    ids=["states", "sha256", "order", "too-few-clear", "sex", "licence", "licence-file",
         "folder", "path"],  # fmt: skip
)
def test_sources_check_refuses(built: Built, edit: Callable[[dict], None]) -> None:
    spec = json.loads(built.sources.read_text(encoding="utf-8"))
    pkg.parse_sources(spec)
    with pytest.raises(InputError, match="sources.json"):
        pkg.parse_sources(_bad(spec, edit))


# ---------------------------------------------------------------------------
# The impostor
# ---------------------------------------------------------------------------


def test_impostor_eligibility_rule() -> None:
    assert pkg.impostor_turns(10.0) == 20 and pkg.impostor_turns(1.0) == 3
    assert pkg.eligible(20, 10.0) and not pkg.eligible(19, 10.0)
    assert pkg.eligible(29, 10.0) and not pkg.eligible(7, 10.0)


def test_impostor_is_the_nearest_eligible_speaker_of_the_target_sex() -> None:
    target = [np.array([1.0, 0.0, 0.0]), np.array([0.9, 0.1, 0.0])]
    vectors = {
        "a": [np.array([0.6, 0.8, 0.0])] * 3,
        "b": [np.array([0.9, 0.2, 0.0]), np.array([0.8, 0.3, 0.1])],
        "c": [np.array([0.0, 0.0, 1.0])],
    }
    scores = mix.impostor_scores(target, vectors)
    assert mix.choose_impostor(scores) == "b"
    assert scores["b"] > scores["a"] > scores["c"]
    with pytest.raises(mix.StageError):
        mix.choose_impostor({})


def test_candidates_are_same_sex_and_eligible(built: Built) -> None:
    sources = pkg.load_sources(built.sources)
    names = {s.speaker for s in mix.impostor_candidates(sources, MINUTES)}
    assert "1221" not in names  # the target's twin, with 2 clips for 3 turns
    assert names == {s for s, v in OTHERS.items() if v[0] == "female" and v[3] >= 3}
    imp = built.build["impostor"]
    assert set(imp["candidates"]) == names
    assert imp["speaker"] == max(imp["candidates"], key=imp["candidates"].get)


# ---------------------------------------------------------------------------
# rehearsal build
# ---------------------------------------------------------------------------


def test_build_writes_episodes_truth_and_anchor(built: Built) -> None:
    root = built.root.path
    episodes = built.build["episodes"]
    assert [e["format"] for e in episodes] == ["eac3-mkv", "aac-mkv", "stereo-mp4"]
    assert not list((root / "seed" / "raw").iterdir())
    by_sha = {}
    for ep in episodes:
        path = root / ep["file"]
        assert path.parent == root / pkg.EPISODES_REL and sha256_file(path) == ep["sha256"]
        by_sha[ep["sha256"]] = ep
        assert abs(ep["duration_s"] - MINUTES * 60) < 15
        assert ep["fallback_lines"] >= 3 and ep["off_centre_lines"] >= 2
        assert ep["impostor_lines"] >= 1 and ep["overlapping_lines"] >= 0
    for line in built.truth:
        assert line.episode_sha256 in by_sha
    truth_imp = {(ln.episode, ln.speaker_id) for ln in built.truth if ln.speaker == "impostor"}
    assert {e for e, _ in truth_imp} == {e["episode"] for e in episodes}
    assert {s for _, s in truth_imp} == {built.build["impostor"]["speaker"]}
    states = {ln.state for ln in built.truth if ln.speaker == "target"}
    assert states == set(SPEAKING_STATES)
    styles = {ln.style: ln.state for ln in built.truth if ln.speaker == "target"}
    assert all(STYLES[style][0] == state for style, state in styles.items())
    assert any(ln.overlap for ln in built.truth)
    used = {ln.clip for ln in built.truth if ln.speaker == "target"}
    target_lines = [ln for ln in built.truth if ln.speaker == "target"]
    assert len(used) == len(target_lines)  # no target clip twice
    anchor_rels = set(built.build["anchor"]["clips"])
    assert anchor_rels and not anchor_rels & used
    info = sf.info(str(root / pkg.SEED_ANCHOR_REL))
    assert 10.0 <= info.duration <= 30.0
    assert (root / pkg.SEED_ANCHOR_REL).read_bytes() == (root / pkg.ANCHOR_REL).read_bytes()


def test_build_encodes_like_raj_files(built: Built) -> None:
    codecs = {e["video_codec"] for e in built.build["episodes"]}
    assert len(codecs) == 1
    listed = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
    assert codecs == {"hevc" if " libx265 " in listed.stdout else "mpeg4"}
    expected = {
        "eac3-mkv": ("matroska", "eac3", 6),
        "aac-mkv": ("matroska", "aac", 6),
        "stereo-mp4": ("mov", "aac", 2),
    }
    for ep in built.build["episodes"]:
        probe = ingest_stage.probe_file(built.root.path / ep["file"], ep["episode"])
        container, codec, channels = expected[ep["format"]]
        assert probe["container"] == container
        kinds = [(s["type"], s["codec"]) for s in probe["streams"]]
        assert kinds == [("video", ep["video_codec"]), ("audio", codec)]
        assert probe["streams"][1]["channels"] == channels


def _episode(built: Built, fmt: str) -> tuple[dict, list[pkg.TruthLine]]:
    ep = next(e for e in built.build["episodes"] if e["format"] == fmt)
    return ep, [ln for ln in built.truth if ln.episode == ep["episode"]]


def test_fallback_geometry_fails_the_centre_and_passes_the_downmix(built: Built) -> None:
    ep, lines = _episode(built, "eac3-mkv")
    path = built.root.path / ep["file"]
    fc, down = decode(path, "eac3", "FC"), decode(path, "eac3", "downmix")
    g = gates.load(None)
    snr_gate, c50_gate = g.value("segment.snr_min_db"), g.value("segment.c50_min_db")
    fallback = [ln for ln in lines if ln.geometry == "fallback"]
    assert len(fallback) >= 3
    assert built.build["fallback"]["copy_db"] == -15.0  # the plan's "a copy 15 dB down"
    # The build measures each fallback line's centre cut as this test does, for the score.
    recorded = {r["line"]: r for r in ep["fallback_fc"]}
    assert sorted(recorded) == sorted(ln.line for ln in fallback)
    for line in fallback:
        dry = dry_line(built.root.path, line.clip)
        a = int(line.start_s * mix.RATE)
        b = a + len(dry)
        tail = b + int(mix.TAIL_S * mix.RATE)
        snr = (mix.snr_db(fc[a:b]), mix.snr_db(down[a:b]))
        c50 = (mix.c50_db(dry, fc[a:tail]), mix.c50_db(dry, down[a:tail]))
        assert snr[0] < snr_gate < snr[1], snr
        assert c50[0] < c50_gate < c50[1], c50
        assert recorded[line.line]["snr_db"] == pytest.approx(snr[0], abs=0.01)
        assert recorded[line.line]["c50_db"] == pytest.approx(c50[0], abs=0.01)
    for line in (ln for ln in lines if ln.geometry == "off-centre"):
        a, b = int(line.start_s * mix.RATE), int(line.end_s * mix.RATE)
        assert mix.band_power(down[a:b], mix.SPEECH_BAND_HZ) > 100 * mix.band_power(
            fc[a:b], mix.SPEECH_BAND_HZ
        )
    assert _episode(built, "aac-mkv")[0]["fallback_fc"]
    assert _episode(built, "stereo-mp4")[0]["fallback_fc"] == []  # no centre channel


def test_target_lines_carry_both_effects(built: Built) -> None:
    ep, lines = _episode(built, "eac3-mkv")
    fc = decode(built.root.path / ep["file"], "eac3", "FC")
    plain = [ln for ln in lines if ln.geometry == "centre" and not ln.overlap]
    assert plain
    emb = FakeEmbedder("ecapa")

    def sim(a: np.ndarray, b: np.ndarray, rate: int) -> float:
        return float(emb.embed(a, rate) @ emb.embed(b, rate))

    for line in plain[:3]:
        audio, rate = mix.read_audio(built.root.path / line.clip)
        raw = mix.trim(audio, rate)
        processed = mix.in_show(raw, rate)
        heard = fc[int(line.start_s * rate) : int(line.start_s * rate) + len(raw)]
        assert spectrum_distance_db(heard, processed, rate) < spectrum_distance_db(heard, raw, rate)
        assert sim(heard, processed, rate) > sim(heard, raw, rate)


def test_build_output_names_no_file(built: Built) -> None:
    text = built.out + built.err
    assert "rehearsal/episodes/ep01.mkv" in text and pkg.SEED_ANCHOR_REL in text
    sources = pkg.load_sources(built.sources)
    for rel in sources.all_files():
        assert Path(rel).name not in text
    for folder in ("public/expresso", "public/librittsr", "public/musan", "LibriTTS_R", "ex02_"):
        assert folder not in text


def test_build_refuses_a_second_build(built: Built) -> None:
    with patched(built.docker, built.sources):
        code, _, err = run_cli(["--root", str(built.root.path), "rehearsal", "build"])
    assert code == 2 and "already holds a build" in err


def test_build_checks_each_sha256_before_use(built: Built, tmp_path: Path) -> None:
    root = root_mod.init_root(tmp_path / "stand-in", "rehearsal")
    shutil.copytree(built.root.path / "public", root.path / "public", dirs_exist_ok=True)
    spec = json.loads(built.sources.read_text(encoding="utf-8"))
    first = spec["target"]["styles"]["happy"]
    rel = f"public/expresso/{first['dir']}/{first['files'][0][0]}"
    data = bytearray((root.path / rel).read_bytes())
    data[-1] ^= 0xFF
    (root.path / rel).write_bytes(bytes(data))
    sources = tmp_path / "sources.json"
    sources.write_text(json.dumps(spec), encoding="utf-8")
    docker = StageDocker(root)
    with patched(docker, sources):
        code, out, err = run_cli(["--root", str(root.path), "rehearsal", "build", "--minutes", "1"])
    assert code == 4 and "log runs/" in err
    log = next((root.path / "runs").glob("*/log.txt")).read_text(encoding="utf-8")
    assert "sha256 does not match" in log and "episode" not in log
    assert not (root.path / pkg.EPISODES_REL).exists()
    assert not (root.path / pkg.BUILD_REL).exists() and not (root.path / pkg.TRUTH_REL).exists()


def test_build_checked_every_public_file(built: Built) -> None:
    sources = pkg.load_sources(built.sources)
    assert built.build["files_checked"] == len(sources.all_files())


# ---------------------------------------------------------------------------
# A fake extraction of the built episodes
# ---------------------------------------------------------------------------


def _flac(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    noise = 0.01 * np.random.default_rng(seed).standard_normal(4800)
    sf.write(str(path), noise, 24000, format="FLAC", subtype="PCM_16")


def _segment(n: int, sid: str, line: pkg.TruthLine, **kw: object) -> SegmentRecord:
    values: dict[str, object] = {
        "segment_id": f"seg-{n:06d}",
        "source_id": sid,
        "start_s": line.start_s + 0.1,
        "end_s": line.end_s - 0.1,
        "channel": "FC",
        "channel_reason": None,
        "separator": "none",
        "audio": f"seed/runs/r1/audio/seg-{n:06d}.flac",
        "diar_cluster": "spk-01",
        "cluster": "primary",
        "flag_group": None,
        "sim": {"wavlm-sv": 0.8 + 0.001 * (n % 7), "ecapa": 0.7 + 0.002 * (n % 5)},
        "overlap_ms": 0,
        "snr_db": 30.0 + n % 3,
        "c50_db": 40.0,
        "transcripts": {"whisperx": Canaries().transcript, "crisper": Canaries().transcript},
        "asr_agreement_cer": 0.0,
        "prosody": Prosody(200.0, 20.0, 4.0, -20.0),
        "arousal": 0.5,
        "state_proposed": line.state or "neutral",
        "state": None,
        "set": "short",
        "kept": True,
        "reasons": (),
    }
    values.update(kw)
    return SegmentRecord(**values)  # type: ignore[arg-type]


@dataclass
class Extracted:
    built: Built
    segments: list[SegmentRecord]
    expected: dict


@pytest.fixture(scope="module")
def extracted(built: Built) -> Extracted:
    """Episodes moved to `seed/raw/rehearsal/` (the lead's H6 step), an inventory, run r1, clean c1."""
    root = built.root
    raw = root.path / "seed" / "raw" / "rehearsal"
    raw.mkdir(parents=True)
    records, segments = [], []
    n = 0
    wrong_state = 0
    expected = {"primary": 0, "true_primary": 0, "impostor": 0, "off": 0, "fallback": 0}
    for index, ep in enumerate(built.build["episodes"]):
        shutil.copyfile(root.path / ep["file"], raw / Path(ep["file"]).name)
        sid = f"src-{index + 1:04d}"
        records.append(
            SourceRecord(
                source_id=sid, path=f"seed/raw/rehearsal/{Path(ep['file']).name}",
                container="matroska", size_bytes=1, mtime_ns=0, streams=(), stream_index=1,
                audio_index=0, language="eng", codec=ep["audio_codec"], profile=None,
                bitrate_bps=1, bitrate_from="packets", sample_rate=48000,
                channels=6 if ep["layout"] == "5.1" else 2, layout=ep["layout"],
                duration_s=ep["duration_s"], lossless=False, effective_bandwidth_hz=20000,
                sha256=ep["sha256"], track=f"seed/tracks/{sid}.mka",
                primary_channel="FC" if ep["layout"] == "5.1" else "downmix",
            )
        )  # fmt: skip
        surround = ep["layout"] == "5.1"
        first = len(segments)
        lines = [ln for ln in built.truth if ln.episode == ep["episode"]]
        for line in lines:
            if line.overlap or line.end_s - line.start_s < 1.0:
                continue
            n += 1
            if line.speaker == "target" and line.geometry == "off-centre" and surround:
                segments.append(
                    _segment(n, sid, line, channel="downmix", channel_reason="off-centre",
                             diar_cluster=None, cluster="flagged", flag_group=f"seg-{n:06d}")
                )  # fmt: skip
                expected["off"] += 1
            elif line.speaker == "target" and line.geometry == "fallback" and surround:
                # Extract writes one record, the downmix's, when the centre cut fails and the
                # downmix passes; the centre cut's SNR and C50 are not recorded.
                segments.append(
                    _segment(n, sid, line, channel="downmix", channel_reason="fc-failed:snr")
                )
                expected["fallback"] += 1
                expected["primary"] += 1
                expected["true_primary"] += 1
            elif line.speaker == "target":
                proposed = line.state
                if n % 6 == 0:
                    proposed = next(s for s in SPEAKING_STATES if s != line.state)
                    wrong_state += 1
                segments.append(_segment(n, sid, line, state_proposed=proposed))
                expected["primary"] += 1
                expected["true_primary"] += 1
            elif line.speaker == "impostor":
                segments.append(
                    _segment(n, sid, line, diar_cluster="spk-02", cluster="flagged",
                             flag_group=f"{sid}/spk-02")
                )  # fmt: skip
                expected["impostor"] += 1
            elif (
                line.speaker == "other"
                and index == 0
                and not any(s.diar_cluster == "spk-03" for s in segments)
            ):
                segments.append(_segment(n, sid, line, diar_cluster="spk-01"))
                expected["primary"] += 1
        if not surround:  # a stereo source has no centre: extract cuts its downmix, no reason
            segments[first:] = [replace(s, channel="downmix") for s in segments[first:]]
    expected["wrong_state"] = wrong_state
    write_jsonl(root, "seed/inventory.jsonl", records)
    write_jsonl(root, "seed/runs/r1/segments.jsonl", segments)
    for seg in segments:
        if seg.audio:
            _flac(root.path / seg.audio, int(seg.segment_id[4:]))
    clusters = [
        {"diar_cluster": "spk-01", "role": "primary", "turns": 9,
         "median_sim": {"wavlm-sv": 0.81, "ecapa": 0.74}},
        {"diar_cluster": "spk-02", "role": "flagged", "turns": 3,
         "median_sim": {"wavlm-sv": 0.66, "ecapa": 0.62}},
        {"diar_cluster": "spk-03", "role": "other", "turns": 5,
         "median_sim": {"wavlm-sv": 0.31, "ecapa": 0.22}},
    ]  # fmt: skip
    write_json(
        root,
        "seed/runs/r1/report.json",
        {
            "run": "r1",
            "sources": [{"source_id": r.source_id, "clusters": clusters} for r in records],
        },
    )
    # The clean run: the kept target segments, cleaned audio being the processed clip at 24 kHz.
    by_id = {(ln.episode, ln.start_s): ln for ln in built.truth}
    clean = []
    for seg in segments:
        if not seg.kept or seg.cluster != "primary" or seg.channel != "FC":
            continue
        sid_index = int(seg.source_id[4:]) - 1
        line = by_id[(built.build["episodes"][sid_index]["episode"], round(seg.start_s - 0.1, 3))]
        if line.speaker != "target":
            continue
        audio, rate = mix.read_audio(root.path / line.clip)
        cleaned = mix.resample(mix.in_show(mix.trim(audio, rate), rate), rate, 24000)
        rel = f"seed/runs/c1/audio/{seg.segment_id}.flac"
        (root.path / rel).parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(root.path / rel), cleaned, 24000, format="FLAC", subtype="PCM_24")
        clean.append(_segment(0, seg.source_id, line, segment_id=seg.segment_id, audio=rel))
    write_jsonl(root, "seed/runs/c1/segments.jsonl", clean)
    write_json(root, "seed/runs/c1/clean-plan.json", {"schema": 1, "run": "c1", "from": "r1"})
    return Extracted(built, segments, expected)


@dataclass
class Answered:
    pack: str
    code: int
    out: str
    err: str
    import_code: int
    import_err: str
    labels: dict[str, str]


@pytest.fixture(scope="module")
def answered(extracted: Extracted) -> Answered:
    root = extracted.built.root
    built = packs.build_pack(root, "labels", {"run": "r1"}, seed=11)
    with patched(extracted.built.docker, extracted.built.sources):
        code, out, err = run_cli(
            ["--root", str(root.path), "rehearsal", "answer", "--pack", built.pack_id]
        )
        # The path answer itself wrote to and printed, as the lead follows it in H6.
        rel = rehearsal_cmd.ratings_rel(built.pack_id)
        assert f"ratings written to {rel};" in out
        ratings = root.path / rel
        icode, _, ierr = run_cli(
            [
                "--root",
                str(root.path),
                "pack",
                "import",
                "--pack",
                built.pack_id,
                "--ratings",
                str(ratings),
            ]
        )
    text = (root.path / packs.LABELS_REL).read_text(encoding="utf-8") if icode == 0 else ""
    labels = {row["segment_id"]: row["state"] for row in csv.DictReader(io.StringIO(text))}
    return Answered(built.pack_id, code, out, err, icode, ierr, labels)


def test_answer_writes_ratings_that_pack_import_accepts(
    extracted: Extracted, answered: Answered
) -> None:
    assert answered.code == 0, answered.err
    assert answered.import_code == 0, answered.import_err
    rel = f"{pkg.ANSWERS_REL}/{answered.pack}/{packs.RATINGS_FILE}"
    assert rehearsal_cmd.ratings_rel(answered.pack) == rel and rel in answered.out
    assert (extracted.built.root.path / rel).is_file()
    eps = rehearsal_cmd._episodes(extracted.built.root)[0]
    kept = [s for s in extracted.segments if s.kept]
    target = {s.segment_id: eps.target_line(s).state for s in kept if eps.who(s) == "target"}  # type: ignore[union-attr]
    assert answered.labels == target
    impostor = [s.segment_id for s in kept if eps.who(s) == "impostor"]
    assert impostor and not set(impostor) & set(answered.labels)


# ---------------------------------------------------------------------------
# rehearsal score
# ---------------------------------------------------------------------------


@dataclass
class Scored:
    code: int
    out: str
    err: str
    score: dict


@pytest.fixture(scope="module")
def scored(extracted: Extracted, answered: Answered) -> Scored:
    root = extracted.built.root
    with patched(extracted.built.docker, extracted.built.sources):
        code, out, err = run_cli(["--root", str(root.path), "rehearsal", "score", "--run", "r1"])
    path = root.path / pkg.SCORES_REL / "r1" / pkg.SCORE_NAME
    score = json.loads(path.read_text(encoding="utf-8")) if code == 0 else {}
    return Scored(code, out, err, score)


def test_score_reports_extraction_against_the_truth(extracted: Extracted, scored: Scored) -> None:
    assert scored.code == 0, scored.err
    s, exp = scored.score, extracted.expected
    assert s["clean_run"] == "c1" and s["episodes"] == EPISODES
    ex = s["extraction"]
    assert ex["kept_primary"] == exp["primary"] and ex["true_primary"] == exp["true_primary"]
    assert ex["precision"] == round(exp["true_primary"] / exp["primary"], 4) < 1.0
    assert ex["min_s"] == gates.load(None).value("segment.min_s")
    assert 0.0 < ex["keepable_s"] < ex["target_s"]
    assert 0.5 < ex["recall"] <= 1.0 and 0.0 < ex["recall_all_target"] < ex["recall"]
    assert "recall" in scored.out and "all-target recall" in scored.out
    acc = s["state_accuracy"]
    assert acc["n"] - sum(r["correct"] for r in acc["per_state"].values()) == exp["wrong_state"]
    assert s["impostor"] == {
        "lines": sum(ln.speaker == "impostor" for ln in extracted.built.truth),
        "kept_primary": 0,
        "kept_flagged": exp["impostor"],
        "in_labels": 0,
    }
    assert s["downmix"]["off-centre"]["recovered"] == exp["off"] > 0
    assert s["downmix"]["off-centre"]["by_reason"] == {"off-centre": exp["off"]}
    assert s["downmix"]["fallback"]["recovered"] == exp["fallback"] >= 3
    assert s["downmix"]["fallback"]["by_reason"] == {"fc-failed:snr": exp["fallback"]}
    for row in s["impostor_clusters"]:
        assert row["status"] == "flagged" and row["identified_by"] == "segments"
        assert row["median_sim"] == {"wavlm-sv": 0.66, "ecapa": 0.62}
        assert row["stray_records"] == []


def test_score_reports_whether_each_effect_survived(scored: Scored) -> None:
    comb, eq = scored.score["effects"]["comb"], scored.score["effects"]["eq"]
    assert comb["survived"] is True and eq["survived"] is True
    assert all(comb["sim_processed"][k] > comb["sim_unprocessed"][k] for k in comb["sim_processed"])
    assert eq["distance_processed_db"] < eq["distance_unprocessed_db"]
    assert "comb filter survived: yes" in scored.out and "EQ survived: yes" in scored.out


def test_score_reports_every_h6_gate(built: Built, scored: Scored) -> None:
    h6 = {k for k, d in gates.load_defaults().items() if d.h6}
    entries = scored.score["gates"]
    assert set(entries) == h6
    for key, entry in entries.items():
        assert (entry["value"] is None) == bool(entry.get("reason")), key
        assert key in scored.out
    for key, field in (("segment.snr_min_db", "snr_db"), ("segment.c50_min_db", "c50_db")):
        measured = [r[field] for e in built.build["episodes"] for r in e["fallback_fc"]]
        assert len(measured) >= 6  # both 5.1 episodes' fallback lines
        assert entries[key]["fallback_fc_max"] == round(max(measured), 3)
        assert entries[key]["fallback_fc_n"] == len(measured)
        assert entries[key]["separates"] is True
    assert entries["segment.c50_min_db"]["value"] == 40.0
    assert entries["retrieval.anchor_floor"]["value"] is None
    assert entries["text.reject_max"]["value"] is None
    # The encodes are band-limited by their codecs (E-AC-3 near 20.5 kHz, AAC near 18 kHz).
    bandwidth = entries["ingest.bandwidth_cutoff_db"]["reason"]
    assert "full band" not in bandwidth and "no known band limit" in bandwidth
    for key in ("anchors.sim_min", "cleaning.spectrum_max_db", "state_floor.offset.neutral",
                "state_floor.rehearsal_identity_window_floor"):  # fmt: skip
        assert isinstance(entries[key]["value"], float), key


def _probe_line(n: int, speaker: str, start: float) -> pkg.TruthLine:
    target = speaker == "target"
    return pkg.TruthLine(
        "ep01", "a" * 64, n, speaker, "" if target else "121", start, start + 3.5,
        "neutral" if target else None, "default" if target else None,
        "centre" if target else None, "x", False,
    )  # fmt: skip


PROBE_LINES = [
    _probe_line(1, "target", 1.0),
    _probe_line(2, "impostor", 5.0),
    _probe_line(3, "target", 9.0),
    _probe_line(4, "impostor", 13.0),
    _probe_line(5, "target", 17.0),
    _probe_line(6, "impostor", 21.0),
]


def _probe(
    placed: Sequence[tuple[int, str, str, bool]], spk02_role: str = "other"
) -> tuple[list[dict], dict[str, dict]]:
    """Score one episode whose records are (truth line, diarizer label, cluster, kept)."""
    record = SourceRecord(
        "src-0001", "p", "matroska", 1, 0, (), 1, 0, None, "eac3", None, 1, "packets", 48000, 6,
        "5.1", 30.0, False, 1, "a" * 64, "t", "FC",
    )  # fmt: skip
    score_mod = rehearsal_cmd.score_mod
    eps = score_mod.Episodes(PROBE_LINES, [record])
    segments = [
        _segment(i, "src-0001", PROBE_LINES[n - 1], diar_cluster=label, cluster=cluster, kept=kept)
        for i, (n, label, cluster, kept) in enumerate(placed, start=1)
    ]
    report = {"sources": [{"source_id": "src-0001", "clusters": [
        {"diar_cluster": "spk-01", "role": "primary", "median_sim": {"ecapa": 0.8}},
        {"diar_cluster": "spk-02", "role": spk02_role, "median_sim": {"ecapa": 0.55}},
        {"diar_cluster": "spk-03", "role": "other", "median_sim": {"ecapa": 0.2}},
    ]}]}  # fmt: skip
    rows = score_mod.impostor_clusters(eps, segments, report)
    gates_found = score_mod.gate_values(
        eps, segments, rows, None, {"retrieval.anchor_floor": 0.6}, {}
    )
    return rows, gates_found


def test_a_stray_impostor_record_in_the_primary_cluster_is_not_its_cluster() -> None:
    # The review's probe: one dropped impostor record diarized into the target's primary cluster,
    # and the impostor's own cluster (spk-02) not admitted, so it left no records.
    rows, found = _probe(
        [(1, "spk-01", "primary", True), (3, "spk-01", "primary", True),
         (5, "spk-01", "primary", True), (2, "spk-01", "primary", False)]
    )  # fmt: skip
    assert rows == [
        {"episode": "ep01", "source_id": "src-0001", "diar_cluster": "spk-02",
         "status": "not admitted", "median_sim": {"ecapa": 0.55},
         "identified_by": "upper bound: highest non-admitted cluster",
         "stray_records": [{"diar_cluster": "spk-01", "role": "primary", "records": 1}]}
    ]  # fmt: skip
    floor = found["retrieval.anchor_floor"]
    assert floor["impostor_primary_episodes"] == 0
    assert floor["impostor_median_max"] is None
    assert floor["non_admitted_median_upper_bound_max"] == 0.55
    assert floor["impostor_stray_records"] == 1
    lines = rehearsal_cmd.score_mod.impostor_cluster_lines(rows)
    assert lines == [
        "  ep01 (src-0001) impostor cluster: not admitted, median ecapa 0.550 (upper bound: "
        "highest non-admitted cluster); stray impostor records in admitted clusters: primary 1"
    ]


def test_the_impostor_cluster_is_the_admitted_one_its_records_hold_the_majority_of() -> None:
    rows, found = _probe(
        [(1, "spk-01", "primary", True), (3, "spk-01", "primary", True),
         (2, "spk-01", "primary", False), (4, "spk-02", "flagged", True),
         (6, "spk-02", "flagged", True), (5, "spk-02", "flagged", True)],
        spk02_role="flagged",
    )  # fmt: skip
    assert len(rows) == 1
    row = rows[0]
    assert (row["diar_cluster"], row["status"], row["identified_by"]) == (
        "spk-02", "flagged", "segments",
    )  # fmt: skip
    assert row["median_sim"] == {"ecapa": 0.55}
    assert row["stray_records"] == [{"diar_cluster": "spk-01", "role": "primary", "records": 1}]
    floor = found["retrieval.anchor_floor"]
    assert floor["impostor_median_max"] == 0.55 and floor["impostor_primary_episodes"] == 0
    assert floor["non_admitted_median_upper_bound_max"] is None


def test_recall_is_over_the_speech_extraction_can_keep_with_all_target_recall_beside() -> None:
    def line(n: int, speaker: str, start: float, end: float) -> pkg.TruthLine:
        target = speaker == "target"
        return replace(
            _probe_line(n, speaker, start), end_s=end, speaker_id="" if target else "260"
        )

    lines = [
        line(1, "target", 1.0, 5.0),  # 4 s, keepable whole
        line(2, "target", 6.0, 8.0),  # 2 s, below segment.min_s: dropped by rule
        line(3, "target", 10.0, 15.0),  # overlapped from 14 s: 10-14 keepable
        line(4, "other", 14.0, 16.0),
        line(5, "other", 19.0, 21.5),
        line(6, "target", 20.0, 23.0),  # 21.5-23 left after the overlap: 1.5 s, dropped
    ]
    record = SourceRecord(
        "src-0001", "p", "matroska", 1, 0, (), 1, 0, None, "eac3", None, 1, "packets", 48000, 6,
        "5.1", 30.0, False, 1, "a" * 64, "t", "FC",
    )  # fmt: skip
    eps = rehearsal_cmd.score_mod.Episodes(lines, [record])
    segments = [
        _segment(1, "src-0001", lines[0]),  # 1.1-4.9
        _segment(2, "src-0001", lines[2], start_s=10.1, end_s=13.9),
    ]
    ex = rehearsal_cmd.score_mod.extraction(eps, segments, min_s=3.0)
    assert (ex["keepable_s"], ex["keepable_covered_s"]) == (8.0, 7.6)
    assert (ex["target_s"], ex["target_covered_s"]) == (14.0, 7.6)
    assert ex["recall"] == 0.95 and ex["recall_all_target"] == round(7.6 / 14.0, 4)
    assert ex["min_s"] == 3.0


def test_an_impostor_majority_primary_cluster_reports_primary() -> None:
    rows, found = _probe(
        [(2, "spk-01", "primary", True), (4, "spk-01", "primary", True),
         (1, "spk-01", "primary", True), (3, "spk-02", "flagged", True)],
        spk02_role="flagged",
    )  # fmt: skip
    assert rows[0]["status"] == "primary" and rows[0]["median_sim"] == {"ecapa": 0.8}
    assert rows[0]["stray_records"] == []
    assert found["retrieval.anchor_floor"]["impostor_primary_episodes"] == 1


# ---------------------------------------------------------------------------
# Refusals and privacy
# ---------------------------------------------------------------------------

ACTIONS = (["build"], ["answer", "--pack", "0123456789abcdef"], ["score", "--run", "r1"])


@pytest.mark.parametrize("action", ACTIONS, ids=lambda a: a[0])
def test_refused_unless_the_root_is_a_rehearsal_root(
    private_root: root_mod.Root, action: list[str]
) -> None:
    docker = StageDocker(private_root)
    with patched(docker, pkg.SOURCES_PATH):
        code, out, err = run_cli(["--root", str(private_root.path), "rehearsal", *action])
    assert code == 2 and "kind rehearsal" in err and not docker.argvs


@pytest.mark.parametrize("as_file", [False, True], ids=["git-dir", "git-file"])
@pytest.mark.parametrize("action", ACTIONS, ids=lambda a: a[0])
def test_refused_inside_a_git_work_tree(tmp_path: Path, action: list[str], as_file: bool) -> None:
    root = root_mod.init_root(tmp_path / "stand-in", "rehearsal")
    git = tmp_path / ".git"
    if as_file:
        git.write_text("gitdir: /elsewhere\n", encoding="utf-8")
    else:
        git.mkdir()
    docker = StageDocker(root)
    with patched(docker, pkg.SOURCES_PATH):
        code, _, err = run_cli(["--root", str(root.path), "rehearsal", *action])
    assert code == 2 and ".git" in err and not docker.argvs


def test_no_canary_reaches_output_argv_or_records(
    built: Built, answered: Answered, scored: Scored, assert_no_canary: Callable[[str], None]
) -> None:
    assert_no_canary(built.out + built.err + answered.out + answered.err + scored.out + scored.err)
    for argv in built.docker.argvs:
        assert_no_canary(" ".join(argv))
    root = built.root.path
    for rel in (pkg.BUILD_REL, f"{pkg.SCORES_REL}/r1/{pkg.SCORE_NAME}"):
        text = (root / rel).read_text(encoding="utf-8")
        assert Canaries().transcript not in text and Canaries().key not in text
    for job in (root / "runs").glob("*/job.json"):
        assert_no_canary(job.read_text(encoding="utf-8"))
