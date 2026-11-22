"""Tests for the verifier's data tools: staging, synthesis, mining and the leakage guard.

Every input is generated here: tone bursts and low noise as WAV and FLAC files, a recording
manifest with train, tune and test entries, an AMI annotations zip, ambient directories with
their `SHA256SUMS`, and synthetic clips at the rates the synthesisers write. The spotter and the
VAD are fakes that score a 20 ms frame by its loudness, so a burst is a hit at every cutoff and
low noise is none. No test reads a recording, needs a network or the GPU.
"""

import ast
import io
import json
import math
import subprocess
import sys
import types
import wave
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from backend.realtime.engines.verifier import GRID
from scripts.wake import manifest as wake_manifest
from scripts.wake.record import FOG, MISSED, NAME_ALONE, WAKE_PHRASES
from scripts.wake.verifier import (
    build,
    chatterbox_driver,
    dataset,
    mine,
    piper_driver,
    stage,
    synth,
)

REPO = Path(__file__).resolve().parents[3]
RATE = 16000
FRAME = 320


# -- Audio ------------------------------------------------------------------------------------


def quiet(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(-50, 51, int(round(seconds * RATE))).astype("<i2")


def burst(seconds: float, rate: int = RATE) -> np.ndarray:
    times = np.arange(int(round(seconds * rate))) / rate
    return (16000 * np.sin(2 * np.pi * 440 * times)).astype("<i2")


def spoken(before: float, name: float, after: float, seed: int = 0) -> np.ndarray:
    """Quiet, a loud burst the fake spotter hits on, quiet: the shape of a recording."""
    return np.concatenate([quiet(before, seed), burst(name), quiet(after, seed + 1)])


def write(path: Path, samples: np.ndarray, rate: int = RATE) -> Path:
    dataset.write_wav(path, samples, rate)
    return path


# -- Fake engines -----------------------------------------------------------------------------


def _rms(pcm: bytes) -> np.ndarray:
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
    frames = len(samples) // FRAME
    return np.sqrt(np.mean(samples[: frames * FRAME].reshape(frames, FRAME) ** 2, axis=1))


@dataclass
class FakeSpotter:
    """A 20 ms step scored by its loudness: a burst scores 1.0, low noise under 0.01."""

    stride: int = 2
    cooldown_steps: int = 5
    series_calls: int = 0

    def score_series(self, pcm: bytes) -> list[float]:
        self.series_calls += 1
        return [float(value) for value in np.minimum(1.0, _rms(pcm) / 8000.0)]

    def probabilities(self, pcm: bytes) -> list[float]:
        return [0.0] * (len(pcm) // 320 // self.stride)


def fake_segments(pcm: bytes) -> list[tuple[int, int]]:
    loud = _rms(pcm) > 1000
    segments: list[tuple[int, int]] = []
    for index, flag in enumerate(loud):
        if flag and segments and segments[-1][1] == index - 1:
            segments[-1] = (segments[-1][0], index)
        elif flag:
            segments.append((index, index))
    return segments


def fake_engines(training: str) -> Callable[[], mine.Engines]:
    def load() -> mine.Engines:
        return mine.Engines(
            spotter=FakeSpotter(),
            segments=fake_segments,
            warmup_samples=0,
            model_id="mist-mww-test",
            model_sha256="cd" * 32,
            training=training,
        )

    return load


# -- The generated world ----------------------------------------------------------------------

PARTITION = stage.Partition(
    train=("ES2002a", "ES2002b", "ES2005a", "IS1000a", "IB4005", "TS3005a"),
    dev=("IB4001",),
    eval=("EN2002a",),
)
SPEAKERS = {
    "ES2002a": ("MEE001", "FEE002"),
    "ES2002b": ("MEE001", "FEE002"),
    "ES2005a": ("MEE003", "FEE002"),
    "IS1000a": ("MIO010",),
    "IB4005": ("MIO020", "FIE030"),
    "TS3005a": ("MTD040",),
    "IB4001": ("FIE030",),
    "EN2002a": ("MEO050",),
}
SECONDS = {"ES2002a": 4, "ES2002b": 4, "ES2005a": 4, "IS1000a": 6, "TS3005a": 10}
TINY = stage.Minimums(calibration_hours=10 / 3600, check_hours=5 / 3600)


def meetings_xml(speakers: dict[str, tuple[str, ...]], namespace: str = "") -> bytes:
    """`meetings.xml` as lhotse reads it: meetings with `observation`, speakers with
    `global_name`; the tag names are not read, so a namespace on them changes nothing.
    """
    ns = f"{{{namespace}}}" if namespace else ""
    lines = [f'<{ns}meetings xmlns="{namespace}">' if namespace else "<meetings>"]
    for meeting, names in speakers.items():
        lines.append(f'<meeting observation="{meeting}" type="test">')
        for channel, name in enumerate(names):
            lines.append(
                f'<speaker nxt_agent="{"ABCD"[channel]}" global_name="{name}" '
                f'channel="{channel}"/>'
            )
        lines.append("</meeting>")
    lines.append("</meetings>")
    return "\n".join(line.replace(ns, "") for line in lines).encode("utf-8")


def annotations_zip(path: Path, speakers: dict[str, tuple[str, ...]], namespace: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("corpusResources/meetings.xml", meetings_xml(speakers, namespace))
        archive.writestr("words/ES2002a.A.words.xml", "<words/>")
    return path


def held_out_dir(directory: Path, meetings: Sequence[str]) -> Path:
    """An AMI dev or eval directory as v14 H28 staged it: the WAVs and their `SHA256SUMS`."""
    sums = {}
    for number, meeting in enumerate(meetings):
        path = write(directory / f"{meeting}.Mix-Headset.wav", spoken(1.0, 0.3, 1.0, 50 + number))
        sums[path.name] = dataset.sha256_file(path)
    (directory / "SHA256SUMS").write_bytes(dataset.sums_bytes(sums))
    return directory


def multichannel(meeting: str, channels: int) -> np.ndarray:
    """A meeting's samples by channel, led by the frames whose mean rounds, ties or clips."""
    seconds = SECONDS[meeting]
    base = spoken(seconds / 2, 0.4, seconds / 2 - 0.4, seed=7).astype(np.int32)
    columns = [base, base * 3 // 4 + 1, -base // 2, base // 3 - 5][:channels]
    frames = np.stack(columns, axis=1)
    counting = np.arange(1, channels + 1)
    edges = [
        np.full(channels, 32767),
        np.full(channels, -32768),
        np.resize([32767, -32768], channels),
        counting,
        counting + 1,
        -counting,
    ]
    frames[: len(edges)] = np.stack(edges)
    return frames.astype("<i2")


def write_channels(path: Path, frames: np.ndarray, rate: int = RATE, width: int = 2) -> Path:
    """A WAV of `frames` (samples by channels) at `width` bytes a sample, as a mirror serves one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.asarray(frames, dtype="<i2")
    if width == 1:
        data = ((samples.astype(np.int32) >> 8) + 128).astype(np.uint8).tobytes()
    elif width == 3:
        data = (samples.astype("<i4") << 8).view(np.uint8).reshape(-1, 4)[:, 1:].tobytes()
    else:
        data = samples.tobytes()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(samples.shape[1])
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(data)
    return path


@dataclass
class FakeFetch:
    """Writes each meeting's WAV: `SECONDS[meeting]` of quiet with a burst in the middle.

    A meeting in `served` is written by `write_channels` in `channels`, `width` and `rate`, the
    same bytes on every call; any other is mono PCM16 at `rate`.
    """

    rate: int = RATE
    served: tuple[str, ...] = ()
    channels: int = 2
    width: int = 2
    urls: list[str] = field(default_factory=list)

    def __call__(self, url: str, dest: Path) -> None:
        self.urls.append(url)
        meeting = url.rsplit("/", 1)[1].split(".")[0]
        if meeting in self.served:
            write_channels(dest, multichannel(meeting, self.channels), self.rate, self.width)
            return
        seconds = SECONDS[meeting]
        samples = spoken(seconds / 2, 0.4, seconds / 2 - 0.4, seed=len(self.urls))
        write(dest, samples, self.rate)


def run_stage(ami: Path, fetch: FakeFetch) -> int:
    """`stage.stage` over `<ami>/{annotations/ami.zip,dev,eval}` into `<ami>/train`."""
    return stage.stage(
        ami / "annotations" / "ami.zip",
        ami / "dev",
        ami / "eval",
        ami / "train",
        fetch=fetch,
        partition=PARTITION,
        minimums=TINY,
    )


def staged_ami(root: Path, fetch: FakeFetch | None = None) -> Path:
    """`<root>/ami/{annotations,dev,eval,train}` staged by `stage.stage`."""
    ami = root / "ami"
    annotations_zip(ami / "annotations" / "ami.zip", SPEAKERS)
    held_out_dir(ami / "dev", PARTITION.dev)
    held_out_dir(ami / "eval", PARTITION.eval)
    assert run_stage(ami, fetch or FakeFetch()) == 0
    return ami


@dataclass
class World:
    root: Path
    manifest: Path
    train_manifest: Path
    recordings: Path
    ami: Path
    librispeech: Path
    musan: Path
    rirs: Path
    synth: Path
    models: Path
    training: str

    def mine_args(self, out: Path, *, manifest: Path | None = None, workers: int = 1) -> list[str]:
        return [
            "--manifest",
            str(manifest or self.train_manifest),
            "--root-map",
            f"{self.recordings}={self.recordings}",
            "--models-dir",
            str(self.models),
            "--spotter",
            "mist-mww-test",
            "--ami",
            str(self.ami / "train"),
            "--librispeech",
            str(self.librispeech),
            "--musan",
            str(self.musan),
            "--rirs",
            str(self.rirs),
            "--synth",
            str(self.synth),
            "--out",
            str(out),
            "--workers",
            str(workers),
        ]

    def build_args(self, mined: Path, out: Path) -> list[str]:
        return [
            "--mined",
            str(mined),
            "--manifest",
            str(self.manifest),
            "--root-map",
            f"{self.recordings}={self.recordings}",
            "--ami-root",
            str(self.ami),
            "--out",
            str(out),
        ]


ENTRIES = (
    # (file, label, split, phrase, session)
    ("rec/train-pos-1.wav", "positive", "train", "mist", "s1"),
    ("rec/train-pos-2.wav", "positive", "train", "hey mist", "s2"),
    ("rec/train-hom-1.wav", "homophone", "train", "a homophone sentence one", "s1"),
    ("rec/train-media-1.wav", "media", "train", "", "m1"),
    ("rec/tune-pos-1.wav", "positive", "tune", "mist", "s3"),
    ("rec/tune-hom-1.wav", "homophone", "tune", "a homophone sentence two", "s3"),
    ("rec/test-pos-1.wav", "positive", "test", "mist", "s4"),
    ("rec/test-media-1.wav", "media", "test", "", "m2"),
)


def write_manifest(root: Path, recordings: Path, *, written_root: str | None = None) -> Path:
    entries = []
    for number, (file, label, split, phrase, session) in enumerate(ENTRIES):
        samples = spoken(1.5, 0.4, 0.6, seed=number * 3)
        path = write(recordings / file, samples)
        del path
        entries.append(
            {
                "file": file,
                "label": label,
                "split": split,
                "phrase": phrase,
                "condition": "headset",
                "duration_ms": len(samples) * 1000 // RATE,
                "session": session,
            }
        )
    manifest = root / "manifest.json"
    payload = {"root": written_root or str(recordings), "entries": entries}
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest


def write_clips(directory: Path, name: str, rate: int, labels: Sequence[str]) -> None:
    rows = []
    for number, label in enumerate(labels):
        file = f"t{number:04d}/{number}.wav"
        clip = np.concatenate([quiet(0.1, number), burst(0.4, rate=rate), quiet(0.1, number + 9)])
        if rate != RATE:
            clip = burst(0.6, rate=rate)
        write(directory / file, clip, rate)
        row = {
            "file": file,
            "text_id": f"t{number:04d}",
            "label": label,
            "speaker": f"{19 + number}",
        }
        rows.append(json.dumps(row, sort_keys=True))
    (directory / dataset.CLIPS_FILE).write_text("\n".join(rows) + "\n", encoding="utf-8")
    meta = {"synthesiser": name, "revision": "5bb1f6ee", "package_version": "0.1.7"}
    (directory / dataset.SYNTH_META).write_text(json.dumps(meta), encoding="utf-8")


@pytest.fixture
def world(tmp_path: Path) -> World:
    recordings = tmp_path / "wake"
    manifest = write_manifest(tmp_path, recordings)
    full = wake_manifest.load_manifest(manifest)
    train_manifest = tmp_path / "train_manifest.json"
    assert (
        wake_manifest.main(
            ["split", "--manifest", str(manifest), "--split", "train", "--out", str(train_manifest)]
        )
        == 0
    )
    training = wake_manifest.subset_sha256(full, "train")
    assert dataset.sha256_file(train_manifest) == training

    ambient = tmp_path / "ambient"
    ami = staged_ami(ambient)
    librispeech = ambient / "librispeech"
    flac = librispeech / "LibriSpeech" / "train-clean-100" / "19" / "198" / "19-198-0001.flac"
    flac.parent.mkdir(parents=True)
    import soundfile

    soundfile.write(str(flac), spoken(1.2, 0.4, 0.6, seed=70), RATE, subtype="PCM_16")
    musan = ambient / "musan"
    write(musan / "noise" / "free-sound" / "noise-free-sound-0001.wav", quiet(3.0, 80))
    write(musan / "music" / "fma" / "music-fma-0001.wav", spoken(1.0, 0.4, 1.6, seed=81))
    rirs = ambient / "rirs"
    impulse = np.zeros(800, dtype=np.float64)
    impulse[0] = 0.9
    impulse[1:] = 0.2 * np.exp(-np.arange(799) / 80.0)
    write(
        rirs / "RIRS_NOISES" / "simulated_rirs" / "smallroom" / "Room001" / "Room001-00001.wav",
        dataset.to_pcm16(impulse),
    )
    write(rirs / "RIRS_NOISES" / "pointsource_noises" / "noise-0001.wav", quiet(1.0, 82))
    synth_root = tmp_path / "synth"
    write_clips(synth_root / "piper", "piper", 22050, ["positive", "negative"] * 2)
    write_clips(synth_root / "chatterbox", "chatterbox", 24000, ["positive", "negative"] * 2)
    models = tmp_path / "models"
    models.mkdir()
    return World(
        root=tmp_path,
        manifest=manifest,
        train_manifest=train_manifest,
        recordings=recordings,
        ami=ami,
        librispeech=librispeech,
        musan=musan,
        rirs=rirs,
        synth=synth_root,
        models=models,
        training=training,
    )


def run_mine(world: World, out: Path, *, workers: int = 1, manifest: Path | None = None) -> int:
    return mine.main(
        world.mine_args(out, manifest=manifest, workers=workers),
        engines=fake_engines(world.training),
    )


@pytest.fixture
def mined(world: World) -> Path:
    out = world.root / "mined"
    assert run_mine(world, out) == 0
    return out


def rewrite(path: Path, change: Callable[[list[dict]], None]) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    change(rows)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


# -- The leakage guard ------------------------------------------------------------------------


class TestLeakageGuard:
    def test_the_index_holds_train_entries_only(self, world: World, mined: Path) -> None:
        out = world.root / "dataset"
        assert build.main(world.build_args(mined, out)) == 0
        windows = dataset.read_windows(out / dataset.INDEX_FILE)
        full = wake_manifest.load_manifest(world.manifest)
        recorded = [window for window in windows if window.file.source == "manifest"]
        assert recorded
        assert {full.entries[w.file.entry].split for w in recorded} == {"train"}
        assert {w.kind for w in recorded} == {"real_positive", "pre_name", "homophone", "media"}
        held_out = {item.sha256 for item in full.entries if item.split != "train"}
        assert not held_out & {ref.sha256 for w in windows for ref in w.refs()}
        sources = {w.file.source for w in windows}
        assert sources == {"manifest", "ami", "librispeech", "musan", "mixed"}

    def test_a_window_from_a_tune_or_test_entry_is_refused(
        self, world: World, mined: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        full = wake_manifest.load_manifest(world.manifest)
        for split in ("tune", "test"):
            entry = next(item for item in full.entries if item.split == split)

            def change(rows: list[dict], entry: wake_manifest.Entry = entry) -> None:
                row = next(r for r in rows if r["file"]["source"] == "manifest")
                row["file"].update(entry=entry.index, sha256=entry.sha256)

            rewrite(mined / dataset.MINED_FILE, change)
            out = world.root / f"dataset-{split}"
            assert build.main(world.build_args(mined, out)) == 2
            assert not out.exists()
        assert "manifest's train split" in capsys.readouterr().err

    @pytest.mark.parametrize(
        ("meeting", "reason"),
        [
            ("IB4001", "AMI dev meeting"),
            ("EN2002a", "AMI eval meeting"),
            ("IB4005", "speaker filter"),
            ("ES2002a", "not in the mining share"),
        ],
    )
    def test_an_ami_window_outside_the_mining_share_is_refused(
        self,
        world: World,
        mined: Path,
        capsys: pytest.CaptureFixture[str],
        meeting: str,
        reason: str,
    ) -> None:
        def change(rows: list[dict]) -> None:
            row = next(r for r in rows if r["file"]["source"] == "ami")
            row["file"]["path"] = f"{meeting}.Mix-Headset.wav"

        rewrite(mined / dataset.MINED_FILE, change)
        out = world.root / "dataset"
        assert build.main(world.build_args(mined, out)) == 2
        assert reason in capsys.readouterr().err
        assert not out.exists()

    @pytest.mark.parametrize("held_out", ["dev", "eval"])
    def test_a_file_listed_in_a_held_out_sha256sums_is_refused(
        self, world: World, mined: Path, capsys: pytest.CaptureFixture[str], held_out: str
    ) -> None:
        windows = dataset.read_windows(mined / dataset.MINED_FILE)
        musan = next(w for w in windows if w.file.source == "musan")
        sums_path = world.ami / held_out / "SHA256SUMS"
        sums = dataset.read_sums(sums_path)
        sums["copied.wav"] = musan.file.sha256
        sums_path.write_bytes(dataset.sums_bytes(sums))
        assert build.main(world.build_args(mined, world.root / "dataset")) == 2
        assert f"ami/{held_out}/SHA256SUMS" in capsys.readouterr().err

    def test_a_mixed_files_background_is_guarded_too(
        self, world: World, mined: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def change(rows: list[dict]) -> None:
            row = next(r for r in rows if any(i["source"] == "ami" for i in r["inputs"]))
            ref = next(i for i in row["inputs"] if i["source"] == "ami")
            ref["path"] = "EN2002a.Mix-Headset.wav"

        rewrite(mined / dataset.MINED_FILE, change)
        assert build.main(world.build_args(mined, world.root / "dataset")) == 2
        assert "AMI eval meeting" in capsys.readouterr().err

    def test_mined_output_from_another_manifest_is_refused(
        self, world: World, mined: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        meta_path = mined / dataset.MINED_META
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["manifest_sha256"] = "0" * 64
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        assert build.main(world.build_args(mined, world.root / "dataset")) == 2
        assert "subset_sha256" in capsys.readouterr().err

    def test_mine_refuses_a_manifest_with_a_tune_or_test_entry(
        self, world: World, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = world.root / "mined-full"
        assert run_mine(world, out, manifest=world.manifest) == 2
        assert "train-only" in capsys.readouterr().err
        assert not out.exists()

    def test_mine_refuses_a_spotter_trained_on_another_split(self, world: World) -> None:
        out = world.root / "mined"
        code = mine.main(world.mine_args(out), engines=fake_engines("e" * 64))
        assert code == 2

    def test_mine_records_the_train_only_manifests_sha256(self, world: World, mined: Path) -> None:
        meta = json.loads((mined / dataset.MINED_META).read_text(encoding="utf-8"))
        assert meta["manifest_sha256"] == dataset.sha256_file(world.train_manifest)
        assert meta["manifest_sha256"] == world.training


# -- Provenance and determinism ---------------------------------------------------------------


class TestProvenance:
    def test_every_indexed_file_is_listed(self, world: World, mined: Path) -> None:
        out = world.root / "dataset"
        assert build.main(world.build_args(mined, out)) == 0
        provenance = json.loads((out / dataset.PROVENANCE_FILE).read_text(encoding="utf-8"))
        assert set(provenance) == {"version", "training", "file_sha256", "sources", "excluded"}
        assert provenance["version"] == 1
        assert provenance["training"] == world.training
        listed = provenance["file_sha256"]
        assert listed == sorted(set(listed))
        windows = dataset.read_windows(out / dataset.INDEX_FILE)
        refs = {ref.sha256 for window in windows for ref in window.refs()}
        assert refs == set(listed)
        mixed = [w for w in windows if w.file.source == "mixed"]
        assert mixed and all(w.inputs for w in mixed)
        for window in mixed:
            assert dataset.sha256_file(mined / window.file.path) == window.file.sha256
        assert provenance["excluded"] == {
            "ami_dev_sha256sums": dataset.sha256_file(world.ami / "dev" / "SHA256SUMS"),
            "ami_eval_sha256sums": dataset.sha256_file(world.ami / "eval" / "SHA256SUMS"),
        }
        for name, source in provenance["sources"].items():
            assert set(source) == {"licence", "version", "files", "hours"}, name
            assert source["files"] >= 1 and source["hours"] > 0
        assert "5bb1f6ee" in provenance["sources"]["chatterbox"]["version"]
        assert set(provenance["sources"]) >= {"manifest", "ami", "piper", "chatterbox", "rirs"}

    def test_mixing_is_deterministic_by_seed(self, world: World, mined: Path) -> None:
        again = world.root / "mined-again"
        assert run_mine(world, again, workers=2) == 0
        assert (again / dataset.MINED_FILE).read_bytes() == (
            mined / dataset.MINED_FILE
        ).read_bytes()
        reseeded = world.root / "mined-reseeded"
        code = mine.main(
            [*world.mine_args(reseeded), "--seed", "18"], engines=fake_engines(world.training)
        )
        assert code == 0
        first = {w.file.sha256 for w in dataset.read_windows(mined / dataset.MINED_FILE)}
        other = {w.file.sha256 for w in dataset.read_windows(reseeded / dataset.MINED_FILE)}
        assert first != other

    def test_hits_are_windows_at_their_cutoffs(self, world: World, mined: Path) -> None:
        windows = dataset.read_windows(mined / dataset.MINED_FILE)
        positives = [w for w in windows if w.kind == "real_positive"]
        # A 0.4 s burst at 1.5 s is frames 75 to 94; a cooldown of 5 steps hits 75, 80, 85, 90.
        assert sorted({w.end for w in positives if w.file.entry == 0}) == [
            (min(124, pos + 15) + 1) * FRAME for pos in (75, 80, 85, 90)
        ]
        assert all(w.cutoffs == GRID for w in positives)
        pre_name = [w for w in windows if w.kind == "pre_name"]
        assert pre_name and all(w.end <= 75 * FRAME and w.label == "negative" for w in pre_name)
        randoms = [w for w in windows if w.kind == "random"]
        assert randoms and all(w.cutoffs == () for w in randoms)
        assert {w.label for w in windows if w.kind == "synthetic"} == {"positive", "negative"}

    def test_build_refuses_an_out_that_exists(self, world: World, mined: Path) -> None:
        out = world.root / "dataset"
        out.mkdir()
        assert build.main(world.build_args(mined, out)) == 2


# -- The manifest's root map ------------------------------------------------------------------


class TestRootMap:
    def test_a_mapped_root_loads_and_keeps_subset_sha256(self, tmp_path: Path) -> None:
        original = tmp_path / "written"
        manifest = write_manifest(tmp_path, original)
        unmapped = wake_manifest.load_manifest(manifest)
        mounted = tmp_path / "mounted"
        original.rename(mounted)
        mapped = wake_manifest.load_manifest(manifest, root_map={str(original): str(mounted)})
        assert mapped.root == str(original)
        assert all(item.path.is_relative_to(mounted.resolve()) for item in mapped.entries)
        for split in wake_manifest.SPLITS:
            assert wake_manifest.subset_sha256(mapped, split) == wake_manifest.subset_sha256(
                unmapped, split
            )

    @pytest.mark.parametrize(
        ("written", "key"),
        [
            ("D:\\Users\\raj\\wake", "D:/Users/raj/wake"),
            ("D:/Users/raj/wake", "d:/users/raj/wake/"),
            ("D:/Users/raj/wake/sub", "D:/Users/raj/wake"),
        ],
    )
    def test_a_foreign_absolute_root_loads_under_the_mapped_root(
        self, tmp_path: Path, written: str, key: str
    ) -> None:
        mounted = tmp_path / "mounted"
        target = mounted / "sub" if written.endswith("sub") else mounted
        manifest = write_manifest(tmp_path, target, written_root=written)
        loaded = wake_manifest.load_manifest(manifest, root_map={key: str(mounted)})
        assert loaded.root == written
        assert all(item.path.is_relative_to(target.resolve()) for item in loaded.entries)

    def test_a_root_no_key_matches_is_refused(self, tmp_path: Path) -> None:
        manifest = write_manifest(tmp_path, tmp_path / "wake", written_root="D:/elsewhere")
        with pytest.raises(wake_manifest.ManifestError, match="root-map"):
            wake_manifest.load_manifest(manifest, root_map={"D:/wake": str(tmp_path / "wake")})

    def test_parse_root_map(self) -> None:
        assert dataset.parse_root_map(["D:/a=/wake"]) == {"D:/a": "/wake"}
        with pytest.raises(dataset.DatasetError):
            dataset.parse_root_map(["D:/a"])


# -- Staging AMI ------------------------------------------------------------------------------


class TestStage:
    def test_the_partition_lists_are_the_full_corpus_asr_partition(self) -> None:
        assert (len(stage.AMI_TRAIN), len(stage.AMI_DEV), len(stage.AMI_EVAL)) == (137, 18, 16)
        lists = (set(stage.AMI_TRAIN), set(stage.AMI_DEV), set(stage.AMI_EVAL))
        assert sum(map(len, lists)) == 171 == len(set.union(*lists))
        assert {"ES2003a", "ES2014d", "TS3006a", "TS3007d", "IB4005"} <= lists[0]
        assert "groups.inf.ed.ac.uk/ami/corpus/datasets.shtml" in stage.AMI_PARTITION_SOURCE

    def test_meetings_xml_is_read_by_position_and_attribute(self, tmp_path: Path) -> None:
        plain = stage.read_meetings(annotations_zip(tmp_path / "a.zip", SPEAKERS))
        spaced = stage.read_meetings(
            annotations_zip(tmp_path / "b.zip", SPEAKERS, namespace="http://nite.sourceforge.net/")
        )
        assert plain == spaced == {meeting: frozenset(names) for meeting, names in SPEAKERS.items()}

    def test_the_speaker_filter_and_the_groups(self) -> None:
        meetings = {meeting: frozenset(names) for meeting, names in SPEAKERS.items()}
        kept, excluded = stage.speaker_filter(meetings, PARTITION)
        assert kept == ("ES2002a", "ES2002b", "ES2005a", "IS1000a", "TS3005a")
        assert excluded == {"IB4005": ("FIE030",)}
        groups = stage.speaker_groups(kept, meetings)
        # ES2002 and ES2005 share FEE002, so the two series are one group.
        assert groups == [("ES2002a", "ES2002b", "ES2005a"), ("IS1000a",), ("TS3005a",)]

    def test_a_meeting_missing_from_meetings_xml_is_refused(self) -> None:
        meetings = {m: frozenset(n) for m, n in SPEAKERS.items() if m != "TS3005a"}
        with pytest.raises(stage.StageError, match="TS3005a"):
            stage.speaker_filter(meetings, PARTITION)

    def test_stage_downloads_the_kept_meetings_and_splits_them(self, tmp_path: Path) -> None:
        fetch = FakeFetch()
        ami = staged_ami(tmp_path, fetch)
        assert fetch.urls == [
            stage.MIRROR.format(meeting=m)
            for m in ("ES2002a", "ES2002b", "ES2005a", "IS1000a", "TS3005a")
        ]
        assert all(url.startswith("https://") for url in fetch.urls)
        train = ami / "train"
        sums = dataset.read_sums(train / "SHA256SUMS")
        assert sorted(sums) == sorted(f"{m}.Mix-Headset.wav" for m in SECONDS)
        assert all(dataset.sha256_file(train / name) == sha for name, sha in sums.items())
        split = json.loads((train / "split.json").read_text(encoding="utf-8"))
        assert split["shares"] == {
            "calibration": ["ES2002a", "ES2002b", "ES2005a"],
            "check": ["IS1000a"],
            "mining": ["TS3005a"],
        }
        assert split["excluded"] == {"IB4005": ["FIE030"]}
        assert split["hours"]["calibration"] == pytest.approx(12 / 3600)
        assert not list(train.glob("IB4005*"))

    def test_a_share_under_its_minimum_exits_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ami = tmp_path / "ami"
        zip_path = annotations_zip(ami / "ann.zip", SPEAKERS)
        held_out_dir(ami / "dev", PARTITION.dev)
        held_out_dir(ami / "eval", PARTITION.eval)
        minimums = stage.Minimums(calibration_hours=10 / 3600, check_hours=7 / 3600)
        code = stage.stage(
            zip_path,
            ami / "dev",
            ami / "eval",
            ami / "train",
            fetch=FakeFetch(),
            partition=PARTITION,
            minimums=minimums,
        )
        assert code == 1
        assert "check" in capsys.readouterr().out
        assert not (ami / "train" / "split.json").exists()

    def test_a_rerun_refuses_a_file_whose_sha256_differs(self, tmp_path: Path) -> None:
        ami = staged_ami(tmp_path)
        fetch = FakeFetch()
        rerun = stage.stage(
            ami / "annotations" / "ami.zip",
            ami / "dev",
            ami / "eval",
            ami / "train",
            fetch=fetch,
            partition=PARTITION,
            minimums=TINY,
        )
        assert rerun == 0 and fetch.urls == []
        write(ami / "train" / "IS1000a.Mix-Headset.wav", quiet(6.0, 99))
        refused = stage.stage(
            ami / "annotations" / "ami.zip",
            ami / "dev",
            ami / "eval",
            ami / "train",
            fetch=fetch,
            partition=PARTITION,
            minimums=TINY,
        )
        assert refused == 2

    @pytest.mark.parametrize(
        ("rate", "channels", "width"),
        [(44100, 1, 2), (44100, 2, 2), (RATE, 1, 1), (RATE, 2, 1), (RATE, 2, 3)],
    )
    def test_a_download_in_another_format_is_refused(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        rate: int,
        channels: int,
        width: int,
    ) -> None:
        ami = tmp_path / "ami"
        annotations_zip(ami / "annotations" / "ami.zip", SPEAKERS)
        held_out_dir(ami / "dev", PARTITION.dev)
        held_out_dir(ami / "eval", PARTITION.eval)
        fetch = FakeFetch(rate=rate, served=tuple(SECONDS), channels=channels, width=width)
        assert run_stage(ami, fetch) == 2
        assert f"is {rate} Hz, {channels} channels, {8 * width}-bit" in capsys.readouterr().err
        assert len(fetch.urls) == 1
        assert not list((ami / "train").iterdir())

    @pytest.mark.parametrize("channels", [2, 3])
    def test_a_multichannel_download_is_staged_as_the_mean_of_its_channels(
        self, tmp_path: Path, channels: int
    ) -> None:
        ami = staged_ami(tmp_path, FakeFetch(served=("ES2005a",), channels=channels))
        train = ami / "train"
        name = "ES2005a.Mix-Headset.wav"
        frames = multichannel("ES2005a", channels)
        original = write_channels(tmp_path / "original.wav", frames)
        staged = train / name
        info = dataset.wav_info(staged)
        assert dataset.is_pcm16_16k_mono(info) and info.frames == len(frames)
        mono = dataset.read_wav_segment(staged, 0, info.frames)
        rounded = np.round(frames.astype(np.float64).mean(axis=1))
        np.testing.assert_array_equal(mono, np.clip(rounded, -32768, 32767).astype("<i2"))
        # The rule the dataset reads any other multichannel file by.
        np.testing.assert_array_equal(mono, dataset.to_pcm16(dataset.read_audio(original)[0]))
        assert dataset.read_sums(train / "SHA256SUMS")[name] == dataset.sha256_file(staged)
        source = {name: {"channels": channels, "sha256": dataset.sha256_file(original)}}
        recorded = json.loads((train / "SOURCES.json").read_text(encoding="utf-8"))
        assert recorded == {"version": 1, "files": source}
        split = json.loads((train / "split.json").read_text(encoding="utf-8"))
        assert split["sources"] == source
        assert sorted(path.name for path in train.iterdir()) == sorted(
            [*(f"{m}.Mix-Headset.wav" for m in SECONDS), "SHA256SUMS", "SOURCES.json", "split.json"]
        )

    def test_a_rerun_downloads_only_what_is_missing_and_clears_nothing(
        self, tmp_path: Path
    ) -> None:
        ami = staged_ami(tmp_path)
        train = ami / "train"
        name = "TS3005a.Mix-Headset.wav"
        # What a run leaves when a multichannel download is refused: the files before it staged
        # and listed, the refused one and split.json absent.
        (train / name).unlink()
        (train / "split.json").unlink()
        listed = dataset.read_sums(train / "SHA256SUMS")
        del listed[name]
        (train / "SHA256SUMS").write_bytes(dataset.sums_bytes(listed))
        kept = {path.name: path.read_bytes() for path in train.glob("*.wav")}
        fetch = FakeFetch(served=("TS3005a",))
        assert run_stage(ami, fetch) == 0
        assert fetch.urls == [stage.MIRROR.format(meeting="TS3005a")]
        assert {n: (train / n).read_bytes() for n in kept} == kept
        sums = dataset.read_sums(train / "SHA256SUMS")
        assert sums == listed | {name: dataset.sha256_file(train / name)}
        original = write_channels(tmp_path / "original.wav", multichannel("TS3005a", 2))
        source = {name: {"channels": 2, "sha256": dataset.sha256_file(original)}}
        # A run after that downloads nothing and keeps the record.
        again = FakeFetch()
        assert run_stage(ami, again) == 0 and again.urls == []
        assert dataset.read_sums(train / "SHA256SUMS") == sums
        recorded = json.loads((train / "SOURCES.json").read_text(encoding="utf-8"))
        assert recorded["files"] == source
        split = json.loads((train / "split.json").read_text(encoding="utf-8"))
        assert split["sources"] == source

    @pytest.mark.parametrize("moment", ["download", "rerun"])
    @pytest.mark.parametrize("kind", ["downloaded", "staged"])
    def test_a_held_out_sha256_as_downloaded_or_as_staged_is_refused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], kind: str, moment: str
    ) -> None:
        name = "ES2005a.Mix-Headset.wav"
        clean = staged_ami(tmp_path / "clean", FakeFetch(served=("ES2005a",)))
        if kind == "downloaded":
            files = json.loads((clean / "train" / "SOURCES.json").read_text(encoding="utf-8"))
            digest = files["files"][name]["sha256"]
        else:
            digest = dataset.read_sums(clean / "train" / "SHA256SUMS")[name]
        ami = clean
        if moment == "download":
            ami = tmp_path / "fresh" / "ami"
            annotations_zip(ami / "annotations" / "ami.zip", SPEAKERS)
            held_out_dir(ami / "dev", PARTITION.dev)
            held_out_dir(ami / "eval", PARTITION.eval)
        sums_path = ami / "eval" / "SHA256SUMS"
        held_out = dataset.read_sums(sums_path) | {"EN2002b.Mix-Headset.wav": digest}
        sums_path.write_bytes(dataset.sums_bytes(held_out))
        capsys.readouterr()
        assert run_stage(ami, FakeFetch(served=("ES2005a",))) == 2
        err = capsys.readouterr().err
        assert name in err and "ami/eval/EN2002b.Mix-Headset.wav" in err
        if moment == "download":
            assert not (ami / "train" / name).exists()
            assert name not in dataset.read_sums(ami / "train" / "SHA256SUMS")
            assert not list((ami / "train").glob("*.part"))

    @pytest.mark.parametrize(
        "files",
        [
            {"IB4005.Mix-Headset.wav": {"channels": 2, "sha256": "a" * 64}},
            {"ES2005a.Mix-Headset.wav": {"channels": 1, "sha256": "a" * 64}},
            {"ES2005a.Mix-Headset.wav": {"channels": 2, "sha256": "a"}},
            {"ES2005a.Mix-Headset.wav": {"channels": 2}},
            [],
        ],
    )
    def test_a_sources_json_that_is_malformed_or_names_another_meeting_is_refused(
        self, tmp_path: Path, files: object
    ) -> None:
        ami = staged_ami(tmp_path)
        sources = ami / "train" / "SOURCES.json"
        sources.write_text(json.dumps({"version": 1, "files": files}), encoding="utf-8")
        assert run_stage(ami, FakeFetch()) == 2

    def test_held_out_dirs_without_sha256sums_are_refused(self, tmp_path: Path) -> None:
        ami = tmp_path / "ami"
        zip_path = annotations_zip(ami / "ann.zip", SPEAKERS)
        held_out_dir(ami / "dev", PARTITION.dev)
        (ami / "eval").mkdir()
        code = stage.stage(
            zip_path,
            ami / "dev",
            ami / "eval",
            ami / "train",
            fetch=FakeFetch(),
            partition=PARTITION,
            minimums=TINY,
        )
        assert code == 2

    def test_the_https_fetch_checks_each_status(self, tmp_path: Path) -> None:
        class Response(io.BytesIO):
            def __init__(self, status: int, body: bytes) -> None:
                super().__init__(body)
                self.status = status

        def opener(status: int) -> Callable[..., Response]:
            return lambda url, timeout: Response(status, b"RIFF")

        dest = tmp_path / "a.wav"
        stage.https_fetch("https://example.org/a.wav", dest, opener=opener(200))
        assert dest.read_bytes() == b"RIFF"
        with pytest.raises(stage.StageError, match="404"):
            stage.https_fetch("https://example.org/b.wav", tmp_path / "b.wav", opener=opener(404))
        with pytest.raises(stage.StageError, match="https"):
            stage.https_fetch("http://example.org/c.wav", tmp_path / "c.wav", opener=opener(200))


# -- Synthesis --------------------------------------------------------------------------------


class TestSynth:
    def test_the_texts_are_disjoint_from_the_recorded_sentences(self) -> None:
        texts = synth.make_texts(seed=17)
        recorded = {
            wake_manifest.phrase_key(text)
            for text in (*MISSED, *FOG, *(phrase for pair in WAKE_PHRASES for phrase in pair))
        }
        names = {wake_manifest.phrase_key(name) for name in NAME_ALONE}
        keys = [wake_manifest.phrase_key(item.text) for item in texts]
        assert not (set(keys) - names) & recorded
        assert len(keys) == len(set(keys))
        assert {item.form for item in texts if item.label == "positive"} >= {
            "alone",
            "first",
            "last",
            "after_greeting",
            "mid",
        }
        assert {item.form for item in texts if item.label == "negative"} >= {
            "sound_alike",
            "fog",
            "other_name",
        }

    def test_the_classes_are_balanced_and_seeded(self, tmp_path: Path) -> None:
        texts = synth.make_texts(seed=17)
        labels = [item.label for item in texts]
        assert labels.count("positive") == labels.count("negative") > 100
        assert synth.make_texts(seed=17) == texts
        assert synth.make_texts(seed=18) != texts
        out = tmp_path / "texts.json"
        assert synth.main(["texts", "--out", str(out), "--seed", "17"]) == 0
        assert synth.load_texts(out) == texts
        assert synth.main(["texts", "--out", str(out), "--seed", "17"]) == 2

    @pytest.mark.parametrize("rate", [16000, 22050, 24000])
    def test_a_clip_is_checked_or_resampled_to_16k_pcm16(self, tmp_path: Path, rate: int) -> None:
        path = write(tmp_path / "clip.wav", burst(0.5, rate=rate), rate)
        clip = synth.load_clip(path)
        assert clip.dtype == np.dtype("<i2")
        assert len(clip) == pytest.approx(8000, abs=1)

    def test_mixing_is_a_function_of_its_plan(self) -> None:
        clip = burst(0.4)
        background = quiet(5.0, 3)
        rir = np.array([1.0, 0.5, 0.25], dtype=np.float32)
        rng = np.random.default_rng([17, 3])
        plan = synth.plan_mix(rng, len(clip), [len(background)], 1)
        assert 0 <= plan.snr_db <= 20
        first = synth.mix(clip, background, rir, plan)
        again = synth.mix(clip, background, rir, plan)
        assert first.dtype == np.dtype("<i2")
        np.testing.assert_array_equal(first, again)
        assert len(first) == plan.lead + len(clip) + synth.TAIL_SAMPLES


class TestPiperDriver:
    def test_it_imports_the_standard_library_only(self) -> None:
        tree = ast.parse(Path(piper_driver.__file__).read_text(encoding="utf-8"))
        modules = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert modules and modules <= set(sys.stdlib_module_names)

    @staticmethod
    def _runner(help_text: str, calls: list[list[str]]) -> Callable[..., object]:
        def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
            calls.append(command)
            if command[-1] == "--help":
                return subprocess.CompletedProcess(command, 0, help_text, "")
            count = int(command[command.index("--max-samples") + 1])
            out = Path(command[command.index("--output-dir") + 1])
            for number in range(count):
                write(out / f"{number}.wav", burst(0.3, rate=22050), 22050)
            return subprocess.CompletedProcess(command, 0, "", "")

        return run

    def _texts(self, tmp_path: Path) -> Path:
        out = tmp_path / "texts.json"
        assert synth.main(["texts", "--out", str(out), "--seed", "17", "--per-class", "3"]) == 0
        return out

    def test_it_generates_each_text_with_the_checked_flags(self, tmp_path: Path) -> None:
        calls: list[list[str]] = []
        run = self._runner("usage: ... --max-samples N --output-dir DIR", calls)
        out = tmp_path / "piper"
        args = [
            "--texts",
            str(self._texts(tmp_path)),
            "--generator",
            str(tmp_path / "gen"),
            "--python",
            "/usr/bin/python3",
            "--per-text",
            "2",
            "--out",
            str(out),
        ]
        assert piper_driver.main(args, run=run) == 0
        assert calls[0][-1] == "--help"
        assert len(calls) == 1 + 6
        rows = [json.loads(line) for line in (out / "clips.jsonl").read_text().splitlines()]
        assert len(rows) == 12
        labels = [row["label"] for row in rows]
        assert labels.count("positive") == labels.count("negative") == 6
        assert {row["rate"] for row in rows} == {22050}
        meta = json.loads((out / "meta.json").read_text())
        assert meta["synthesiser"] == "piper" and meta["per_text"] == 2

    def test_a_generator_without_the_flags_is_refused(self, tmp_path: Path) -> None:
        calls: list[list[str]] = []
        run = self._runner("usage: generate_samples.py text --output-dir DIR", calls)
        args = [
            "--texts",
            str(self._texts(tmp_path)),
            "--generator",
            str(tmp_path / "gen"),
            "--python",
            "python3",
            "--per-text",
            "2",
            "--out",
            str(tmp_path / "piper"),
        ]
        assert piper_driver.main(args, run=run) == 2
        assert len(calls) == 1


class TestChatterboxDriver:
    @pytest.fixture
    def fake_chatterbox(self, monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
        loads: list[tuple] = []

        class FakeTTS:
            sr = 24000

            @classmethod
            def from_local(cls, ckpt_dir: object, device: str) -> "FakeTTS":
                loads.append((Path(ckpt_dir), device))
                return cls()

            @classmethod
            def from_pretrained(cls, device: str) -> "FakeTTS":
                raise AssertionError("from_pretrained fetches with no revision")

            def generate(self, text: str, audio_prompt_path: str | None = None) -> np.ndarray:
                assert audio_prompt_path and Path(audio_prompt_path).is_file()
                return (burst(0.5, rate=24000).astype(np.float32) / 32768.0)[np.newaxis, :]

        package = types.ModuleType("chatterbox")
        module = types.ModuleType("chatterbox.tts")
        module.ChatterboxTTS = FakeTTS  # type: ignore[attr-defined]
        package.tts = module  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "chatterbox", package)
        monkeypatch.setitem(sys.modules, "chatterbox.tts", module)
        return loads

    def _setup(self, tmp_path: Path) -> tuple[Path, Path, Path]:
        texts = tmp_path / "texts.json"
        assert synth.main(["texts", "--out", str(texts), "--seed", "17", "--per-class", "4"]) == 0
        references = tmp_path / "train-clean-100"
        for speaker in ("19", "26"):
            flac = references / speaker / "198" / f"{speaker}-198-0001.flac"
            flac.parent.mkdir(parents=True)
            flac.write_bytes(b"a reference the fake model never decodes")
        snapshot = tmp_path / "snapshots" / "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
        snapshot.mkdir(parents=True)
        for name in chatterbox_driver.SNAPSHOT_FILES:
            (snapshot / name).write_bytes(b"x")
        return texts, references, snapshot

    def test_it_loads_from_local_and_balances_the_classes(
        self, tmp_path: Path, fake_chatterbox: list[tuple]
    ) -> None:
        texts, references, snapshot = self._setup(tmp_path)
        seeds: list[int] = []
        out = tmp_path / "chatterbox"
        args = [
            "--texts",
            str(texts),
            "--references",
            str(references),
            "--snapshot",
            str(snapshot),
            "--count",
            "6",
            "--seed",
            "17",
            "--out",
            str(out),
            "--device",
            "cpu",
        ]
        assert chatterbox_driver.main(args, seed_all=seeds.append) == 0
        assert fake_chatterbox == [(snapshot, "cpu")]
        rows = [json.loads(line) for line in (out / "clips.jsonl").read_text().splitlines()]
        labels = [row["label"] for row in rows]
        assert labels.count("positive") == labels.count("negative") == 3
        assert {row["rate"] for row in rows} == {24000}
        assert len(set(seeds)) == 6
        meta = json.loads((out / "meta.json").read_text())
        assert meta["revision"] == snapshot.name
        for row in rows:
            assert dataset.wav_info(out / row["file"]).rate == 24000

    def test_an_odd_count_or_an_incomplete_snapshot_is_refused(
        self, tmp_path: Path, fake_chatterbox: list[tuple]
    ) -> None:
        texts, references, snapshot = self._setup(tmp_path)
        base = [
            "--texts",
            str(texts),
            "--references",
            str(references),
            "--snapshot",
            str(snapshot),
            "--seed",
            "17",
            "--device",
            "cpu",
        ]
        assert (
            chatterbox_driver.main(
                [*base, "--count", "5", "--out", str(tmp_path / "a")], seed_all=lambda seed: None
            )
            == 2
        )
        (snapshot / "conds.pt").unlink()
        assert (
            chatterbox_driver.main(
                [*base, "--count", "4", "--out", str(tmp_path / "b")], seed_all=lambda seed: None
            )
            == 2
        )
        assert fake_chatterbox == []

    def test_the_docstring_states_the_api_and_never_from_pretrained_is_called(self) -> None:
        source = Path(chatterbox_driver.__file__).read_text(encoding="utf-8")
        assert "from_local" in chatterbox_driver.__doc__
        assert "0.1.7" in chatterbox_driver.__doc__
        assert "from_pretrained(" not in source.replace("`from_pretrained(", "")
        assert "import chatterbox" not in source.split("def load_chatterbox")[0]


# -- The boundary -----------------------------------------------------------------------------


def test_nothing_in_backend_imports_the_verifier_data_tools() -> None:
    hits = [
        str(path)
        for path in (REPO / "backend").rglob("*.py")
        if "scripts.wake.verifier" in path.read_text(encoding="utf-8")
    ]
    assert hits == []


def test_window_bounds_follow_the_verifiers_window() -> None:
    assert dataset.window_bounds(150, 1000) == (50 * FRAME, 166 * FRAME)
    assert dataset.window_bounds(30, 1000) == (0, 46 * FRAME)
    assert dataset.window_bounds(10, 1000) is None
    assert dataset.window_bounds(10, 20) is None
    assert dataset.window_bounds(1000, 1000) is None
    assert math.isclose(dataset.BEFORE_MS + dataset.AFTER_MS, 2300)
