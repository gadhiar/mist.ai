"""Tests for `scripts/voice_bench/voice_ab.py`: the ear test's two voices, rendered offline.

The renderer loads each identity through the realtime adapter's `load` and streams every line
through her character stage, as she sounds live, then brings both arms of each line to the same
loudness and writes FLAC clips and one manifest into the private root's bench run. Here the
adapter is a fake that speaks a synthetic vowel at a level set per identity, the clock is a
counter, and the private root lives in pytest's temporary directory: no model, no identity and no
private data is read.
"""

from __future__ import annotations

import io
import json
import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from backend.realtime.budget import CharacterConfig, PlacementConfig
from backend.realtime.character import CharacterStage
from backend.realtime.engines.audio import TARGET_LUFS, integrated_loudness
from backend.realtime.errors import SpeechEngineError
from backend.realtime.speech import CancelToken, SynthChunk, TextSegment
from scripts.voice_bench import voice_ab
from scripts.voice_data import packs
from scripts.voice_data import root as root_mod

FS = 24000
BASELINE = "mist"
TRAINED = "mist-trained-r1"
LEVELS = {BASELINE: 0.05, TRAINED: 0.2}
LINES_FILE = voice_ab.REPO_ROOT / "scripts" / "model_bench" / "speech" / "lines.json"
CHUNK_SAMPLES = 4800


def _vowel(seconds: float, level: float) -> np.ndarray:
    t = np.arange(round(seconds * FS)) / FS
    x = sum(np.sin(2 * np.pi * k * 200.0 * t) / k**2 for k in range(1, 30))
    return (level * x / np.max(np.abs(x))).astype(np.float32)


class FakeSynthesizer:
    """Speaks a 1.2 s vowel per segment, in 200 ms chunks, at its identity's level."""

    sample_rate = FS

    def __init__(self, voice: str, log: list[tuple[str, str]]) -> None:
        self.voice = voice
        self.model_id = f"kyutai/pocket-tts@pin+{voice}"
        self._log = log

    def synthesize(self, segment: TextSegment, token: CancelToken) -> Iterator[SynthChunk]:
        self._log.append(("synthesize", self.voice))
        assert segment.state == "neutral" and segment.text
        pcm = (_vowel(1.2, LEVELS[self.voice]) * 32767).astype("<i2").tobytes()
        for start in range(0, len(pcm), 2 * CHUNK_SAMPLES):
            piece = pcm[start : start + 2 * CHUNK_SAMPLES]
            yield SynthChunk(pcm16=piece, samples=len(piece) // 2, segment=segment)

    def release(self) -> None:
        self._log.append(("release", self.voice))


class Harness:
    """The loader, clock and log a run is given."""

    def __init__(self) -> None:
        self.log: list[tuple[str, str]] = []
        self.placements: list[PlacementConfig] = []
        self.ticks = 0.0

    def load(self, placement: PlacementConfig, profile: str) -> FakeSynthesizer:
        self.placements.append(placement)
        self.log.append(("load", str(placement.tts_voice)))
        return FakeSynthesizer(str(placement.tts_voice), self.log)

    def clock(self) -> float:
        # Each reading is 0.125 s after the last: a line's first chunk lands one reading after
        # the call, so every `first_chunk_ms` is 125.0.
        self.ticks += 0.125
        return self.ticks


@pytest.fixture
def private_root(tmp_path: Path) -> root_mod.Root:
    return root_mod.init_root(tmp_path / "voice-root", "private")


def _lines_file(tmp_path: Path, lines: list[dict]) -> Path:
    path = tmp_path / "lines.json"
    path.write_text(json.dumps({"schema": 1, "lines": lines}), encoding="utf-8")
    return path


def _argv(out: Path, lines: Path = LINES_FILE, **overrides: str) -> list[str]:
    args = {"--baseline": BASELINE, "--trained": TRAINED, "--lines": str(lines), "--out": str(out)}
    args.update(overrides)
    return [part for pair in args.items() for part in pair]


def _run(argv: list[str], harness: Harness) -> int:
    return voice_ab.main(argv, load=harness.load, clock=harness.clock)


def _out(root: root_mod.Root, run: str = "r1") -> Path:
    return root.path / "bench" / "voice-ab" / run


def _read_flac(path: Path) -> np.ndarray:
    soundfile = pytest.importorskip("soundfile")
    audio, rate = soundfile.read(io.BytesIO(path.read_bytes()), dtype="float64")
    assert rate == FS
    return audio


# --- a run ---------------------------------------------------------------------------------


def test_renders_every_line_for_both_identities_into_the_bench_run(
    private_root: root_mod.Root, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("soundfile")
    harness = Harness()
    out = _out(private_root)
    assert _run(_argv(out), harness) == 0
    doc = json.loads(LINES_FILE.read_text(encoding="utf-8"))
    public = doc["lines"] + doc["acknowledgements"]  # L01-L20, then A01-A06
    manifest = json.loads((out / voice_ab.MANIFEST).read_text(encoding="utf-8"))
    assert manifest["schema"] == 1
    assert (manifest["baseline"], manifest["trained"]) == (BASELINE, TRAINED)
    assert manifest["target_lufs"] == TARGET_LUFS
    assert manifest["sample_rate"] == FS
    assert manifest["character"] == {"double_gain": 0.35, "limiter": True}
    assert [row["line_id"] for row in manifest["lines"]] == [row["id"] for row in public]
    assert len(manifest["lines"]) == 26
    for row, line in zip(manifest["lines"], public, strict=True):
        assert row["text"] == line["text"]
        assert set(row["arms"]) == {"baseline", "trained"}
        for arm, info in row["arms"].items():
            assert info["clip"] == f"bench/voice-ab/r1/clips/{line['id']}-{arm}.flac"
            assert info["first_chunk_ms"] == 125.0
            assert set(info) == {
                "clip",
                "first_chunk_ms",
                "loudness_lufs",
                "gain_db",
                "clipped_samples",
                "model_id",
            }
    first = manifest["lines"][0]["arms"]
    assert first["trained"]["model_id"].endswith(TRAINED)
    # The louder identity gets the smaller gain; the stage's compressor narrows the gap first.
    assert first["trained"]["loudness_lufs"] > first["baseline"]["loudness_lufs"] + 1
    assert first["trained"]["gain_db"] < first["baseline"]["gain_db"]
    for info in first.values():
        assert isinstance(info["clipped_samples"], int) and info["clipped_samples"] >= 0
    out_text, err = capsys.readouterr()
    assert err == ""
    for secret in (BASELINE, TRAINED, public[0]["text"], "L01"):
        assert secret not in out_text


def test_each_identity_loads_once_through_the_runtime_placement_and_is_released(
    private_root: root_mod.Root, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("soundfile")
    monkeypatch.setenv("MIST_VOICE_IDENTITY_ROOT", "/srv/voices")
    monkeypatch.setenv("MIST_VOICE_MODELS_DIR", "/srv/models")
    monkeypatch.setenv("MIST_VOICE_CACHE_DIR", "/srv/voice-cache")
    harness = Harness()
    assert _run(_argv(_out(private_root)), harness) == 0
    assert [p.tts_voice for p in harness.placements] == [BASELINE, TRAINED]
    for placement in harness.placements:
        assert placement.voice_root == "/srv/voices"
        assert placement.models_dir == "/srv/models"
        assert placement.voice_cache_dir == "/srv/voice-cache"
    loads = [i for i, entry in enumerate(harness.log) if entry[0] == "load"]
    releases = [i for i, entry in enumerate(harness.log) if entry[0] == "release"]
    # The baseline is released before the trained identity loads: one model in memory at a time.
    assert len(loads) == len(releases) == 2
    assert loads[0] < releases[0] < loads[1] < releases[1]
    assert sum(entry == ("synthesize", BASELINE) for entry in harness.log) == 26


def test_clips_are_untagged_pcm16_flac_at_the_target_loudness(
    private_root: root_mod.Root,
) -> None:
    pytest.importorskip("soundfile")
    out = _out(private_root)
    assert _run(_argv(out), Harness()) == 0
    manifest = json.loads((out / voice_ab.MANIFEST).read_text(encoding="utf-8"))
    paths = [
        private_root.path / info["clip"]
        for row in manifest["lines"]
        for info in row["arms"].values()
    ]
    assert len(paths) == 52
    # The pack copies these byte for byte, so its format check must pass on them as written.
    packs.check_formats(paths)
    info = packs.flac_info(paths[0])
    assert (info.sample_rate, info.channels, info.bits_per_sample) == (FS, 1, 16)
    for row in manifest["lines"][:2]:
        levels = []
        for arm in row["arms"].values():
            audio = _read_flac(private_root.path / arm["clip"])
            levels.append(integrated_loudness(audio, FS))
            assert arm["loudness_lufs"] + arm["gain_db"] == pytest.approx(TARGET_LUFS, abs=1e-6)
        assert levels == pytest.approx([TARGET_LUFS, TARGET_LUFS], abs=0.1)


def test_the_audio_is_her_character_stage_streamed_then_gained(
    private_root: root_mod.Root,
) -> None:
    pytest.importorskip("soundfile")
    out = _out(private_root)
    assert _run(_argv(out), Harness()) == 0
    manifest = json.loads((out / voice_ab.MANIFEST).read_text(encoding="utf-8"))
    arm = manifest["lines"][0]["arms"]["trained"]
    stage = CharacterStage(CharacterConfig(double_gain=0.35), sample_rate=FS)
    pcm = (_vowel(1.2, LEVELS[TRAINED]) * 32767).astype("<i2").tobytes()
    chunks = [pcm[s : s + 2 * CHUNK_SAMPLES] for s in range(0, len(pcm), 2 * CHUNK_SAMPLES)]
    staged = b"".join([*(stage.process(c) for c in chunks), stage.end_segment()])
    expected = np.frombuffer(staged, dtype="<i2").astype(np.float64) / 32768.0
    assert arm["loudness_lufs"] == pytest.approx(integrated_loudness(expected, FS), abs=1e-6)
    gained = _read_flac(private_root.path / arm["clip"])
    assert len(gained) == len(expected)
    scale = 10 ** (arm["gain_db"] / 20)
    assert np.max(np.abs(gained - np.clip(expected * scale, -1, 1))) < 2 / 32768


def test_the_character_stage_is_on_at_double_gain_0_35() -> None:
    assert CharacterConfig(double_gain=0.35) == voice_ab.CHARACTER
    assert voice_ab.CHARACTER.limiter is not None


def test_first_chunk_ms_times_the_first_chunk_of_each_line(
    private_root: root_mod.Root,
) -> None:
    pytest.importorskip("soundfile")
    harness = Harness()
    ticks = iter([0.0, 0.25, 1.0, 1.4] + [10.0 + n for n in range(400)])
    harness.clock = lambda: next(ticks)  # type: ignore[method-assign]
    out = _out(private_root)
    assert _run(_argv(out), harness) == 0
    manifest = json.loads((out / voice_ab.MANIFEST).read_text(encoding="utf-8"))
    assert manifest["lines"][0]["arms"]["baseline"]["first_chunk_ms"] == 250.0
    assert manifest["lines"][1]["arms"]["baseline"]["first_chunk_ms"] == 400.0


def test_prints_counts_and_first_chunk_figures_per_arm(
    private_root: root_mod.Root, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("soundfile")
    assert _run(_argv(_out(private_root)), Harness()) == 0
    printed = capsys.readouterr().out.splitlines()
    assert printed[0].startswith("baseline: 26 lines, first chunk median 125 ms, p90 125 ms")
    assert printed[1].startswith("trained: 26 lines, first chunk median 125 ms, p90 125 ms")
    assert "LUFS" in printed[0]
    assert printed[-1] == "voice-ab: written to bench/voice-ab/r1/"


# --- refusals --------------------------------------------------------------------------------


def _refusal(
    argv: list[str], private_root: root_mod.Root, capsys: pytest.CaptureFixture[str]
) -> str:
    harness = Harness()
    assert _run(argv, harness) == 2
    assert harness.placements == []
    err = capsys.readouterr().err
    assert err.startswith("voice_ab: ")
    return err


@pytest.mark.parametrize(
    ("overrides", "words"),
    [
        ({"--trained": BASELINE}, "two different identities"),
        ({"--trained": "Mist_Trained"}, "--trained"),
        ({"--baseline": "../mist"}, "--baseline"),
    ],
    ids=["same", "bad-trained", "bad-baseline"],
)
def test_refuses_identity_ids_before_loading(
    private_root: root_mod.Root,
    capsys: pytest.CaptureFixture[str],
    overrides: dict[str, str],
    words: str,
) -> None:
    err = _refusal(_argv(_out(private_root), **overrides), private_root, capsys)
    assert words in err
    assert not _out(private_root).exists()


def test_refuses_an_out_folder_that_exists(
    private_root: root_mod.Root, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _out(private_root)
    out.mkdir(parents=True)
    assert "already exists" in _refusal(_argv(out), private_root, capsys)
    assert list(out.iterdir()) == []


@pytest.mark.parametrize(
    "where", ["bench/other/r1", "voice-ab/r1", "bench/voice-ab/Bad Run"], ids=str
)
def test_refuses_an_out_folder_outside_bench_voice_ab(
    private_root: root_mod.Root, capsys: pytest.CaptureFixture[str], where: str
) -> None:
    err = _refusal(_argv(private_root.path / where), private_root, capsys)
    assert "--out" in err and str(private_root.path) not in err


def test_refuses_an_out_folder_outside_a_private_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "unmarked" / "bench" / "voice-ab" / "r1"
    harness = Harness()
    assert _run(_argv(out), harness) == 2
    assert "marker" in capsys.readouterr().err
    assert not out.exists() and harness.placements == []


def test_refuses_an_out_folder_inside_the_checkout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = voice_ab.REPO_ROOT / "bench" / "voice-ab" / "r1"
    harness = Harness()
    assert _run(_argv(out), harness) == 2
    assert "inside the checkout" in capsys.readouterr().err
    assert not out.exists() and harness.placements == []


@pytest.mark.parametrize(
    "lines",
    [
        [],
        [{"id": "L01"}],
        [{"id": "L01", "text": "One."}, {"id": "L01", "text": "Two."}],
        [{"id": 5, "text": "One."}],
    ],
    ids=["empty", "no-text", "duplicate", "bad-id"],
)
def test_refuses_a_malformed_lines_file(
    private_root: root_mod.Root,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    lines: list[dict],
) -> None:
    err = _refusal(_argv(_out(private_root), _lines_file(tmp_path, lines)), private_root, capsys)
    assert "--lines" in err
    assert not _out(private_root).exists()


def test_an_engine_failure_exits_1_and_writes_nothing(
    private_root: root_mod.Root, capsys: pytest.CaptureFixture[str]
) -> None:
    harness = Harness()

    def load(placement: PlacementConfig, profile: str) -> FakeSynthesizer:
        if placement.tts_voice == TRAINED:
            raise SpeechEngineError("pocket-clone: the model loaded without voice cloning")
        return harness.load(placement, profile)

    assert voice_ab.main(_argv(_out(private_root)), load=load, clock=harness.clock) == 1
    assert "without voice cloning" in capsys.readouterr().err
    assert not _out(private_root).exists()
    assert harness.log[-1] == ("release", BASELINE)


def test_a_short_lines_file_renders_only_its_lines(
    private_root: root_mod.Root, tmp_path: Path
) -> None:
    pytest.importorskip("soundfile")
    lines = [{"id": "L01", "text": "One line."}, {"id": "A01", "text": "Okay."}]
    out = _out(private_root, "short")
    assert _run(_argv(out, _lines_file(tmp_path, lines)), Harness()) == 0
    manifest = json.loads((out / voice_ab.MANIFEST).read_text(encoding="utf-8"))
    assert [row["line_id"] for row in manifest["lines"]] == ["L01", "A01"]
    assert math.isfinite(manifest["lines"][1]["arms"]["trained"]["loudness_lufs"])
