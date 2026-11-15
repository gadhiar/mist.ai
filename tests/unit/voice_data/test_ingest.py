"""Tests for `ingest`: probing, stream choice, the lossless track copy, decoding and bandwidth.

Sources are made with ffmpeg in tmp. The job runs through the real `LocalDocker` with a fake
`DockerOps` that runs the stage module in this process with `--data` pointed at the root, which
is exactly where the container's `/data` mounts point, so the argv is the real one and only the
container boundary is faked.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import os
import shutil
import subprocess
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts.voice_data import cli
from scripts.voice_data import root as root_mod
from scripts.voice_data.commands import ingest as ingest_cmd
from scripts.voice_data.errors import InputError, RootError
from scripts.voice_data.records import SourceRecord, read_jsonl
from scripts.voice_data.stages import bandwidth
from scripts.voice_data.stages import ingest as ingest_stage

from .conftest import KEY_ENV_NAMES, Canaries

RATE = 48000
SECONDS = 1.0
IMAGE = "mist-voice-base:0123456789ab"


# Signals and sources.


def _noise(channels: int, band_hz: float | None = None, seed: int = 0) -> np.ndarray:
    """White noise, brick-wall low-passed at `band_hz` when given, shape (samples, channels)."""
    n = int(RATE * SECONDS)
    x = np.random.default_rng(seed).standard_normal((n, channels)) * 0.1
    if band_hz is not None:
        spectrum = np.fft.rfft(x, axis=0)
        spectrum[np.fft.rfftfreq(n, 1.0 / RATE) > band_hz] = 0
        x = np.fft.irfft(spectrum, n, axis=0)
    return x.astype(np.float32)


def _tones(freqs: Sequence[float]) -> np.ndarray:
    t = np.arange(int(RATE * SECONDS)) / RATE
    return np.stack([0.2 * np.sin(2 * np.pi * f * t) for f in freqs], axis=1).astype(np.float32)


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *args], check=True)


def _wav(path: Path, samples: np.ndarray) -> Path:
    sf.write(str(path), samples, RATE, subtype="FLOAT")
    return path


VIDEO = ("-f", "lavfi", "-i", f"testsrc=size=64x64:rate=5:duration={SECONDS}")


def _make_sources(work: Path, raw: Path, canary_name: str, canary_title: str) -> None:
    """The five source kinds of the acceptance: E-AC-3 5.1 MKV with video, AAC 5.1 MKV, AAC
    stereo MP4, FLAC and WAV, plus files ingest must ignore.
    """
    (raw / "show").mkdir(parents=True)
    six = _wav(work / "six.wav", _noise(6, seed=1))
    stereo = _wav(work / "two.wav", _noise(2, seed=2))
    side = ("-af", "channelmap=channel_layout=5.1(side)")
    _ffmpeg(
        *VIDEO, "-i", str(six), "-map", "0:v", "-map", "1:a", "-c:v", "mpeg4", *side,
        "-c:a", "eac3", "-b:a", "384k", "-metadata", f"title={canary_title}",
        str(raw / "show" / canary_name),
    )  # fmt: skip
    _ffmpeg("-i", str(six), *side, "-c:a", "aac", "-b:a", "256k", str(raw / "show" / "b.mkv"))
    _ffmpeg("-i", str(stereo), "-c:a", "aac", "-b:a", "128k", str(raw / "c.mp4"))
    _ffmpeg("-i", str(stereo), "-c:a", "flac", str(raw / "d.flac"))
    lowpassed = _wav(work / "lp.wav", _noise(2, band_hz=8000, seed=3))
    _ffmpeg("-i", str(lowpassed), "-c:a", "pcm_s16le", str(raw / "e.wav"))
    (raw / "notes.txt").write_text("not a source\n", encoding="utf-8")
    (raw / ".hidden.mkv").write_bytes(b"")


# The fake container boundary.


class StageDocker:
    """A `DockerOps` whose `run_logged` runs the stage module in this process."""

    def __init__(self, root: root_mod.Root, before_step: Callable[[str], None] | None = None):
        self.root = root
        self.before_step = before_step
        self.argvs: list[list[str]] = []

    def capture(self, argv: Sequence[str]) -> str:
        raise AssertionError("ingest runs no GPU preflight")

    def check(self, argv: Sequence[str]) -> None:
        raise AssertionError("ingest builds nothing")

    def run_logged(self, argv: Sequence[str], log: Path) -> int:
        self.argvs.append(list(argv))
        at = list(argv).index("-m")
        module, rest = argv[at + 1], list(argv[at + 2 :])
        assert rest[:2] == ["--data", "/data"]
        rest[1] = str(self.root.path)
        if self.before_step is not None:
            self.before_step(rest[rest.index("--step") + 1])
        with (
            log.open("w", encoding="utf-8") as handle,
            contextlib.redirect_stdout(handle),
            contextlib.redirect_stderr(handle),
        ):
            return importlib.import_module(module).main(rest)


@contextlib.contextmanager
def _patched(docker: StageDocker) -> Iterator[None]:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ingest_cmd, "make_docker", lambda: docker)
        mp.setattr(ingest_cmd, "image_tag", lambda family: IMAGE)
        yield


@dataclass
class Ingested:
    root: root_mod.Root
    tmp: Path
    code: int
    out: str
    err: str
    argvs: list[list[str]]
    before: dict[str, tuple[int, int, str]]
    records: list[SourceRecord]


def _fingerprints(raw: Path) -> dict[str, tuple[int, int, str]]:
    found = {}
    for path in sorted(p for p in raw.rglob("*") if p.is_file()):
        stat = path.stat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        found[path.relative_to(raw).as_posix()] = (stat.st_size, stat.st_mtime_ns, digest)
    return found


def _ingest(root: root_mod.Root, docker: StageDocker, *extra: str) -> tuple[int, str, str]:
    out, err = _Capture(), _Capture()
    with _patched(docker), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(["--root", str(root.path), "ingest", *extra])
    return code, out.getvalue(), err.getvalue()


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


@pytest.fixture(scope="module")
def ingested(tmp_path_factory: pytest.TempPathFactory) -> Ingested:
    canaries = Canaries()
    tmp = tmp_path_factory.mktemp("ingest")
    root = root_mod.init_root(tmp / "voice-root", "private")
    (root.path / "guard" / "terms.txt").write_text(canaries.term + "\n", encoding="utf-8")
    work = tmp / "work"
    work.mkdir()
    _make_sources(work, root.path / "seed" / "raw", canaries.source_name, canaries.transcript)
    shutil.rmtree(work)
    before = _fingerprints(root.path / "seed" / "raw")
    docker = StageDocker(root)
    with pytest.MonkeyPatch.context() as mp:
        for name in KEY_ENV_NAMES:
            mp.setenv(name, canaries.key)
        code, out, err = _ingest(root, docker)
    inventory = root.path / "seed" / "inventory.jsonl"
    records = read_jsonl(inventory, SourceRecord) if inventory.exists() else []
    return Ingested(root, tmp, code, out, err, docker.argvs, before, records)


def _probe(path: Path) -> dict[str, object]:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def _by_suffix(ingested: Ingested, name: str) -> SourceRecord:
    matches = [r for r in ingested.records if r.path.endswith(name)]
    assert len(matches) == 1
    return matches[0]


# End to end on the five source kinds.


def test_ingest_exits_0_with_one_record_per_source(ingested: Ingested) -> None:
    assert ingested.code == 0, ingested.err
    paths = [r.path for r in ingested.records]
    assert len(paths) == 5
    assert all(p.startswith("seed/raw/") for p in paths)
    by_path = sorted(paths)
    assert [r.source_id for r in sorted(ingested.records, key=lambda r: r.path)] == [
        root_mod.source_id(n) for n in range(1, len(by_path) + 1)
    ]


def test_subfolders_are_walked_and_non_media_ignored(ingested: Ingested) -> None:
    paths = {r.path for r in ingested.records}
    assert "seed/raw/show/b.mkv" in paths
    assert not any(p.endswith((".txt", ".hidden.mkv")) for p in paths)


def test_every_stream_is_recorded_with_both_indexes(ingested: Ingested) -> None:
    mkv = [r for r in ingested.records if r.streams and r.streams[0]["type"] == "video"]
    assert len(mkv) == 1
    record = mkv[0]
    assert [s["type"] for s in record.streams] == ["video", "audio"]
    assert record.streams[0]["codec"] == "mpeg4"
    assert record.streams[0]["width"] == 64
    assert record.stream_index == 1
    assert record.audio_index == 0
    assert record.codec == "eac3"
    assert record.channels == 6
    assert record.layout == "5.1(side)"
    assert record.container == "matroska"
    assert record.primary_channel == "FC"
    assert record.lossless is False
    assert record.bitrate_from == "packets"


def test_the_copy_keeps_codec_and_layout_and_drops_video(ingested: Ingested) -> None:
    for record in ingested.records:
        source = _probe(ingested.root.path / record.path)["streams"]
        audio = [s for s in source if s["codec_type"] == "audio"][record.audio_index]
        assert record.track == f"seed/tracks/{record.source_id}.mka"
        copy = _probe(ingested.root.path / record.track)["streams"]
        assert [s["codec_type"] for s in copy] == ["audio"]
        assert copy[0]["codec_name"] == audio["codec_name"] == record.codec
        assert copy[0]["channels"] == audio["channels"] == record.channels
        assert copy[0].get("channel_layout") == audio.get("channel_layout")


def test_packet_bitrate_of_the_copy_matches_the_source(ingested: Ingested) -> None:
    for record in ingested.records:
        source_bps, _ = ingest_stage.packet_bitrate(
            ingested.root.path / record.path, record.audio_index
        )
        assert source_bps > 0
        assert abs(record.bitrate_bps - source_bps) / source_bps <= 0.01


def test_sources_are_unchanged(ingested: Ingested) -> None:
    assert _fingerprints(ingested.root.path / "seed" / "raw") == ingested.before
    for record in ingested.records:
        size, mtime, digest = ingested.before[record.path.removeprefix("seed/raw/")]
        assert (record.size_bytes, record.mtime_ns, record.sha256) == (size, mtime, digest)


def test_codec_facts_per_kind(ingested: Ingested) -> None:
    assert _by_suffix(ingested, "b.mkv").codec == "aac"
    # ffprobe names no layout for AAC 5.1 in Matroska; ffmpeg's default for 6 channels is 5.1.
    assert _by_suffix(ingested, "b.mkv").layout == "5.1"
    assert _by_suffix(ingested, "b.mkv").primary_channel == "FC"
    assert _by_suffix(ingested, "c.mp4").layout == "stereo"
    assert _by_suffix(ingested, "c.mp4").primary_channel == "downmix"
    assert _by_suffix(ingested, "d.flac").lossless is True
    assert _by_suffix(ingested, "e.wav").codec == "pcm_s16le"
    assert _by_suffix(ingested, "e.wav").lossless is True


def test_bandwidth_of_a_lowpassed_source(ingested: Ingested) -> None:
    assert abs(_by_suffix(ingested, "e.wav").effective_bandwidth_hz - 8000) <= 500
    assert _by_suffix(ingested, "d.flac").effective_bandwidth_hz >= 20000


def test_the_job_mounts_seed_raw_read_only(ingested: Ingested) -> None:
    assert len(ingested.argvs) == 2
    raw = f"{ingested.root.path / 'seed' / 'raw'}:/data/seed/raw:ro"
    for argv in ingested.argvs:
        assert argv[:2] == ["docker", "run"]
        assert raw in argv
        assert IMAGE in argv
        assert "/opt/venv-score/bin/python" in argv
        assert "scripts.voice_data.stages.ingest" in argv
        writable = [a for a in argv if a.endswith(":/data/seed/tracks")]
        assert len(writable) == 1


def test_no_canary_in_output_argv_or_files(
    ingested: Ingested, assert_no_canary: Callable[[str], None]
) -> None:
    assert_no_canary(ingested.out + ingested.err)
    for argv in ingested.argvs:
        assert_no_canary("\n".join(argv))
    inventory = ingested.root.path / "seed" / "inventory.jsonl"
    raw = ingested.root.path / "seed" / "raw"
    terms = ingested.root.path / "guard" / "terms.txt"
    for path in ingested.tmp.rglob("*"):
        if not path.is_file() or path in (inventory, terms) or path.is_relative_to(raw):
            continue
        # Binary files too: a track copied with the source's tags would carry its title.
        assert_no_canary(path.read_bytes().decode("latin-1"))


def test_output_names_ids_not_files(ingested: Ingested) -> None:
    assert "src-0001" in ingested.out
    assert "unchanged" in ingested.out


# Stable ids, changes during the run, refusals.


def _flac(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), _noise(1, seed=seed)[: RATE // 4], RATE, format="FLAC")


def test_a_rerun_with_one_file_added_keeps_ids(private_root: root_mod.Root) -> None:
    raw = private_root.path / "seed" / "raw"
    _flac(raw / "m.flac", 1)
    _flac(raw / "z.flac", 2)
    assert _ingest(private_root, StageDocker(private_root))[0] == 0
    inventory = private_root.path / "seed" / "inventory.jsonl"
    first = {r.path: r.source_id for r in read_jsonl(inventory, SourceRecord)}
    assert first == {"seed/raw/m.flac": "src-0001", "seed/raw/z.flac": "src-0002"}
    _flac(raw / "a.flac", 3)
    assert _ingest(private_root, StageDocker(private_root))[0] == 0
    second = {r.path: r.source_id for r in read_jsonl(inventory, SourceRecord)}
    assert second == dict(first, **{"seed/raw/a.flac": "src-0003"})


@pytest.mark.parametrize("change", ["content", "mtime"])
def test_a_source_changed_during_ingest_exits_4(
    private_root: root_mod.Root, change: str, capsys: pytest.CaptureFixture[str]
) -> None:
    source = private_root.path / "seed" / "raw" / "m.flac"
    _flac(source, 1)

    def tamper(step: str) -> None:
        if step != "copy":
            return
        if change == "content":
            with source.open("ab") as handle:
                handle.write(b"\0")
        else:
            stat = source.stat()
            os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    code, _, err = _ingest(private_root, StageDocker(private_root, tamper))
    assert code == 4
    assert "src-0001" in err
    assert not (private_root.path / "seed" / "inventory.jsonl").exists()


def test_no_audio_is_reported_by_id_and_skipped(private_root: root_mod.Root) -> None:
    raw = private_root.path / "seed" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    _ffmpeg(*VIDEO, "-c:v", "mpeg4", str(raw / "a.mkv"))
    _flac(raw / "b.flac", 1)
    code, out, _ = _ingest(private_root, StageDocker(private_root))
    assert code == 0
    assert "src-0001: no audio stream; skipped" in out
    records = read_jsonl(private_root.path / "seed" / "inventory.jsonl", SourceRecord)
    assert [r.source_id for r in records] == ["src-0002"]


def _two_tracks(raw: Path, name: str, *, default: bool, tmp: Path) -> Path:
    raw.mkdir(parents=True, exist_ok=True)
    wav = _wav(tmp / "two.wav", _noise(2, seed=4))
    disposition = ("-disposition:a:0", "default") if default else ("-disposition:a:0", "0")
    target = raw / name
    _ffmpeg(
        *VIDEO, "-i", str(wav), "-i", str(wav), "-map", "0:v", "-map", "1:a", "-map", "2:a",
        "-c:v", "mpeg4", "-c:a", "flac", "-metadata:s:a:0", "language=eng",
        "-metadata:s:a:1", "language=fra", *disposition, "-disposition:a:1", "0",
        # Without passthrough the Matroska muxer marks the first audio track default itself.
        "-default_mode", "passthrough", str(target),
    )  # fmt: skip
    return target


def test_language_picks_the_track(private_root: root_mod.Root, tmp_path: Path) -> None:
    _two_tracks(private_root.path / "seed" / "raw", "a.mkv", default=True, tmp=tmp_path)
    assert _ingest(private_root, StageDocker(private_root), "--language", "fra")[0] == 0
    (record,) = read_jsonl(private_root.path / "seed" / "inventory.jsonl", SourceRecord)
    assert (record.stream_index, record.audio_index, record.language) == (2, 1, "fra")


def test_ambiguous_tracks_exit_2_without_the_file_name(
    private_root: root_mod.Root,
    tmp_path: Path,
    canaries,
    assert_no_canary: Callable[[str], None],
) -> None:
    name = canaries.source_name
    _two_tracks(private_root.path / "seed" / "raw", name, default=False, tmp=tmp_path)
    code, out, err = _ingest(private_root, StageDocker(private_root))
    assert code == 2
    assert "src-0001" in err
    assert "1" in err and "2" in err and "eng" in err and "fra" in err
    assert_no_canary(out + err)


def test_no_sources_exit_2(private_root: root_mod.Root) -> None:
    assert _ingest(private_root, StageDocker(private_root))[0] == 2


@pytest.mark.parametrize("kind", ["dir", "file"])
def test_a_root_in_a_git_work_tree_is_refused(tmp_path: Path, kind: str) -> None:
    root = root_mod.init_root(tmp_path / "work" / "voice-root", "private")
    _flac(root.path / "seed" / "raw" / "a.flac", 1)
    git = tmp_path / "work" / ".git"
    if kind == "dir":
        git.mkdir()
    else:
        git.write_text("gitdir: elsewhere\n", encoding="utf-8")
    docker = StageDocker(root)
    assert _ingest(root, docker)[0] == 2
    assert docker.argvs == []
    with pytest.raises(RootError):
        root.write_bytes("seed/tracks/x.mka", b"")


def test_writers_refuse_seed_raw(private_root: root_mod.Root) -> None:
    with pytest.raises(RootError):
        private_root.write_bytes("seed/raw/x.mka", b"")


# Stream choice, a pure function on the probe's stream list.


def _audio(index: int, language: str | None, default: bool) -> dict[str, object]:
    stream: dict[str, object] = {"index": index, "type": "audio", "codec": "eac3"}
    if language is not None:
        stream["language"] = language
    stream["default"] = default
    return stream


VIDEO_STREAM = {"index": 0, "type": "video", "codec": "hevc"}


@pytest.mark.parametrize(
    ("streams", "language", "chosen"),
    [
        ([VIDEO_STREAM, _audio(1, "fra", False)], None, 1),
        ([VIDEO_STREAM, _audio(1, "fra", False)], "eng", 1),
        ([_audio(0, "eng", False), _audio(1, "fra", True)], None, 1),
        ([_audio(0, "eng", False), _audio(1, "fra", True)], "eng", 0),
        ([_audio(0, "eng", False), _audio(1, "fra", True)], "ENG", 0),
        ([_audio(0, "eng", True), _audio(1, None, False)], "deu", 0),
        ([_audio(0, "eng", False), _audio(1, "eng", True), _audio(2, "fra", False)], "eng", 1),
    ],
)
def test_choose_audio(streams: list[dict[str, object]], language: str | None, chosen: int) -> None:
    stream, audio_index = ingest_cmd.choose_audio("src-0001", streams, language)
    assert stream["index"] == chosen
    assert audio_index == [s["index"] for s in streams if s["type"] == "audio"].index(chosen)


def test_choose_audio_with_no_audio_is_none() -> None:
    assert ingest_cmd.choose_audio("src-0001", [VIDEO_STREAM], None) is None


@pytest.mark.parametrize(
    ("streams", "language"),
    [
        ([_audio(1, "eng", False), _audio(2, "fra", False)], None),
        ([_audio(1, "eng", False), _audio(2, "eng", False)], "eng"),
        ([_audio(1, "eng", True), _audio(2, "fra", True)], None),
    ],
)
def test_choose_audio_refuses_ambiguity(
    streams: list[dict[str, object]], language: str | None
) -> None:
    with pytest.raises(InputError) as caught:
        ingest_cmd.choose_audio("src-0007", streams, language)
    message = str(caught.value)
    assert "src-0007" in message
    assert "stream 1" in message and "stream 2" in message


def test_ids_are_assigned_in_sorted_order_after_existing() -> None:
    paths = ["seed/raw/a.mkv", "seed/raw/b.mkv", "seed/raw/c.mkv"]
    assert ingest_cmd.assign_ids(paths, {}) == {
        "seed/raw/a.mkv": "src-0001",
        "seed/raw/b.mkv": "src-0002",
        "seed/raw/c.mkv": "src-0003",
    }
    existing = {"seed/raw/b.mkv": "src-0001", "seed/raw/c.mkv": "src-0004"}
    assert ingest_cmd.assign_ids(paths, existing) == {
        "seed/raw/a.mkv": "src-0005",
        "seed/raw/b.mkv": "src-0001",
        "seed/raw/c.mkv": "src-0004",
    }


# Decoding: DRC off, the centre channel alone, and the downmix.


@pytest.mark.parametrize("codec", ["ac3", "eac3"])
def test_decode_turns_drc_off_for_ac3_family(codec: str) -> None:
    argv = ingest_stage.decode_argv("t.mka", codec, "FC")
    at = argv.index("-drc_scale")
    assert argv[at + 1] == "0"
    assert at < argv.index("-i")


def test_decode_argv_channels() -> None:
    fc = ingest_stage.decode_argv("t.mka", "aac", "FC")
    assert fc[fc.index("-af") + 1] == "pan=mono|c0=FC"
    assert "-ac" not in fc
    down = ingest_stage.decode_argv("t.mka", "aac", "downmix")
    assert down[down.index("-ac") + 1] == "1"
    assert "-af" not in down


@pytest.mark.parametrize(
    ("layout", "primary"),
    [
        ("5.1(side)", "FC"),
        ("5.1", "FC"),
        ("7.1", "FC"),
        ("3.0", "FC"),
        ("stereo", "downmix"),
        ("mono", "downmix"),
        ("quad", "downmix"),
        ("FL+FR+FC", "FC"),
    ],
)
def test_primary_channel(layout: str, primary: str) -> None:
    assert ingest_stage.primary_channel(layout) == primary


def _band_power(samples: np.ndarray, freq: float) -> float:
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples)))) ** 2
    freqs = np.fft.rfftfreq(len(samples), 1.0 / RATE)
    return float(spectrum[(freqs > freq - 20) & (freqs < freq + 20)].sum())


def test_fc_decode_holds_only_the_centre_tone(tmp_path: Path) -> None:
    # WAV order is FL FR FC LFE BL BR; channelmap names them as the real files' 5.1(side).
    tones = {"FL": 300.0, "FR": 500.0, "FC": 1000.0, "LFE": 60.0, "SL": 700.0, "SR": 1500.0}
    wav = _wav(tmp_path / "tones.wav", _tones(list(tones.values())))
    track = tmp_path / "tones.mka"
    _ffmpeg(
        "-i", str(wav), "-af", "channelmap=channel_layout=5.1(side)",
        "-c:a", "eac3", "-b:a", "384k", str(track),
    )  # fmt: skip
    fc = bandwidth.read_decode(ingest_stage.decode_argv(str(track), "eac3", "FC"))
    centre = _band_power(fc, tones["FC"])
    for name, freq in tones.items():
        if name != "FC":
            assert 10 * np.log10(centre / max(_band_power(fc, freq), 1e-30)) > 40, name
    down = bandwidth.read_decode(ingest_stage.decode_argv(str(track), "eac3", "downmix"))
    assert _band_power(down, tones["FL"]) > _band_power(down, tones["FC"]) * 1e-2


# Effective bandwidth on synthetic signals.


@pytest.mark.parametrize(("band", "expected"), [(8000.0, 8000), (16000.0, 16000)])
def test_bandwidth_of_a_band_limited_signal(band: float, expected: int) -> None:
    signal = _noise(1, band_hz=band, seed=5)[:, 0]
    assert abs(bandwidth.effective_bandwidth(signal, RATE, 40.0) - expected) <= 500


def test_bandwidth_of_a_full_band_signal() -> None:
    assert bandwidth.effective_bandwidth(_noise(1, seed=6)[:, 0], RATE, 40.0) >= 20000


def test_bandwidth_ignores_silent_frames() -> None:
    # A quiet full-band noise floor between loud 8 kHz passages must not widen the band.
    loud = _noise(1, band_hz=8000.0, seed=7)[:, 0]
    quiet = _noise(1, seed=8)[:, 0] * 1e-3
    signal = np.concatenate([loud, quiet, loud])
    assert abs(bandwidth.effective_bandwidth(signal, RATE, 40.0) - 8000) <= 500


def test_bandwidth_streams_in_blocks() -> None:
    signal = _noise(1, band_hz=8000.0, seed=9)[:, 0]
    ltas = bandwidth.Ltas(RATE)
    for start in range(0, len(signal), 1000):
        ltas.add(signal[start : start + 1000])
    assert ltas.cutoff_hz(40.0) == bandwidth.effective_bandwidth(signal, RATE, 40.0)
