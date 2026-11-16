"""Tests for `extract`: the command, the stage and Design 4 on synthetic episodes.

The job runs through the real `LocalDocker` with a fake `DockerOps` that runs the stage module in
this process, `--data` pointed at the root (where the container's `/data` mounts point), and the
stage's `load_models` replaced by fakes. Tracks are float WAV files that the stage decodes with
the real ffmpeg argv, so the centre channel and the downmix are taken exactly as in the image.

Two fakes live here rather than in `fakes.py`:

- `PrototypeEmbedder` names each clip's voice by its nearest prototype in a voice-band log
  spectrum (80-3800 Hz) and returns that voice's designed vector, so every cosine of the
  acceptance (an impostor at 0.8, a damaged centre-channel cut at 0.3) is set by design while
  which voice a clip holds is still read from the decoded audio. Both registers of the target
  map to the target's vector, as a split by register leaves the voice unchanged.
- `BandSnrQuality` measures SNR as the energy in the voices' band (100-3800 Hz) against the
  effects burst's band (5-7.8 kHz), per 50 ms frame, so the 5.1 fallback passes or fails on what
  the mix holds.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import json
import subprocess
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from scripts.voice_data import cli, jobs, seed_gate
from scripts.voice_data import gates as gates_mod
from scripts.voice_data import root as root_mod
from scripts.voice_data.bindings.protocols import QualityTrack, Span, Turn
from scripts.voice_data.commands import extract as extract_cmd
from scripts.voice_data.errors import RootError
from scripts.voice_data.records import Prosody, SegmentRecord, SourceRecord, read_jsonl, write_jsonl
from scripts.voice_data.stages import extract as extract_stage
from scripts.voice_data.stages import prosody as prosody_mod
from scripts.voice_data.stages import retrieve
from scripts.voice_data.stages import sdh as sdh_stage
from scripts.voice_data.stages.ingest import decode_argv
from scripts.voice_data.stages.score import spectrum_distance_db
from scripts.voice_data.stages.segments import span_quality
from scripts.voice_data.states import SPEAKING_STATES

from .conftest import KEY_ENV_NAMES, Canaries, _synth_voice
from .fakes import FakeArousal, FakeDiarizer, FakeEmbedder, FakeQualityEstimator, FakeVad

RATE = 16000
IMAGE = "mist-voice-extract:0123456789ab"
GAP = 0.4
BED_DB = -26.0

# Voices: F0 and formants; the target and its excited register carry the in-show comb effect.
VOICES = {
    "target": (210.0, (700.0, 1200.0, 2600.0)),
    "excited": (245.0, (700.0, 1200.0, 2600.0)),
    "impostor": (185.0, (500.0, 1700.0, 2900.0)),
    "other_a": (120.0, (400.0, 2000.0, 3200.0)),
    "other_b": (160.0, (850.0, 1500.0, 3000.0)),
}
LABELS = {"target": "spk-01", "excited": "spk-02", "impostor": "spk-03", "other_a": "spk-04",
          "other_b": "spk-05"}  # fmt: skip


# Signals.


def comb(audio: np.ndarray) -> np.ndarray:
    """The fixed in-show effect: a 2 ms feed-forward comb, notches every 500 Hz."""
    delay = int(0.002 * RATE)
    out = audio.astype(np.float64).copy()
    out[delay:] += 0.7 * audio[:-delay]
    return (0.5 * out / np.max(np.abs(out))).astype(np.float32)


def lowpass(audio: np.ndarray, cutoff_hz: float) -> np.ndarray:
    spectrum = np.fft.rfft(audio.astype(np.float64))
    spectrum[np.fft.rfftfreq(len(audio), 1.0 / RATE) > cutoff_hz] = 0.0
    return np.fft.irfft(spectrum, len(audio)).astype(np.float32)


def band_noise(seconds: float, low: float, high: float, seed: int) -> np.ndarray:
    n = int(round(seconds * RATE))
    spectrum = np.fft.rfft(np.random.default_rng(seed).standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1.0 / RATE)
    spectrum[(freqs < low) | (freqs > high)] = 0.0
    out = np.fft.irfft(spectrum, n)
    return (out / np.sqrt(np.mean(out**2))).astype(np.float32)


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float64) ** 2)))


VOICE_S = 9.0
BED_LOOP_S = 10.0


@functools.cache
def source_voice(name: str, effect: bool) -> np.ndarray:
    """9 s of a voice, from which lines of up to 8 s are taken at different offsets."""
    f0, formants = VOICES[name]
    audio = _synth_voice(f0, formants, VOICE_S, RATE)
    return comb(audio) if effect else audio


def take(name: str, seconds: float, k: int, effect: bool = True) -> np.ndarray:
    """A line of `seconds` of a voice; `k` picks the offset, so lines differ."""
    use_effect = effect and name in ("target", "excited")
    voice = source_voice(name, use_effect)
    offset = int(((k * 3.7) % (VOICE_S - seconds - 0.5)) * RATE)
    return voice[offset : offset + int(round(seconds * RATE))].copy()


@functools.cache
def _bed_loop() -> np.ndarray:
    # Every note and the 0.3 Hz swell complete whole cycles in 10 s, so the loop tiles seamlessly.
    t = np.arange(int(round(BED_LOOP_S * RATE))) / RATE
    notes = (196.0, 246.9, 293.7, 392.0)
    bed = sum(np.sin(2 * np.pi * f * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.3 * t + i))
              for i, f in enumerate(notes))  # fmt: skip
    bed = np.asarray(bed) / np.max(np.abs(bed))
    return (0.5 * 10 ** (BED_DB / 20.0) * bed).astype(np.float32)


def music_bed(seconds: float) -> np.ndarray:
    n = int(round(seconds * RATE))
    loop = _bed_loop()
    return np.tile(loop, n // len(loop) + 1)[:n].copy()


# Fakes local to these tests.


N_FFT = 2048


@functools.cache
def _band_edges(rate: int) -> tuple[np.ndarray, np.ndarray]:
    # A Hann window, and the first FFT bin of each of 30 log-spaced bands over 80-3800 Hz plus the
    # bin past the last, so a band's power is a difference of cumulative sums (no BLAS call,
    # whose threads oversubscribe a loaded host).
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / rate)
    return np.hanning(N_FFT), np.searchsorted(freqs, np.geomspace(80.0, 3800.0, 31))


def _band_features(audio: np.ndarray, rate: int) -> np.ndarray:
    window, edges = _band_edges(rate)
    signal = np.asarray(audio, dtype=np.float64)
    count = max(1, len(signal) // N_FFT)
    signal = np.pad(signal, (0, max(0, count * N_FFT - len(signal))))[: count * N_FFT]
    frames = signal.reshape(count, N_FFT) * window
    power = (np.abs(np.fft.rfft(frames, axis=1)) ** 2).sum(axis=0)
    total = np.concatenate(([0.0], np.cumsum(power)))
    bands = total[edges[1:]] - total[edges[:-1]]
    if bands.max() <= 0.0:
        return np.zeros(len(bands))
    levels = 10.0 * np.log10(bands + 1e-20)
    levels = np.maximum(levels, levels.max() - 40.0)
    return levels - levels.mean()


DIM = 8


def _designed(cos: float, axis: int) -> np.ndarray:
    v = np.zeros(DIM)
    v[0] = cos
    v[axis] = np.sqrt(1.0 - cos**2)
    return v


DESIGNED = {
    "target": _designed(1.0, 1),
    "impostor": _designed(0.8, 2),
    "other_a": _designed(0.1, 3),
    "other_b": _designed(0.05, 4),
    "damaged": _designed(0.3, 5),
    "music": _designed(0.0, 6),
    "silence": _designed(0.0, 7),
    # The target and the impostor through the low-pass separator of the separator-choice cases.
    "target_dme": _designed(0.9, 6),
    "impostor_dme": _designed(0.85, 7),
}


class Voices:
    """Names a clip's voice by its nearest prototype in the band spectrum.

    The stage hands the same array to each embedder in turn, so the last answer is kept and
    reused while the array is the same object.
    """

    def __init__(self, prototypes: Mapping[str, Sequence[np.ndarray]]) -> None:
        self._keys: list[str] = []
        rows = []
        for key, clips in prototypes.items():
            for clip in clips:
                feature = _band_features(clip, RATE)
                rows.append(feature / np.linalg.norm(feature))
                self._keys.append(key)
        self._features = np.array(rows)
        self._last: tuple[np.ndarray, str] | None = None

    def voice(self, audio: np.ndarray, sample_rate: int) -> str:
        if self._last is not None and self._last[0] is audio:
            return self._last[1]
        feature = _band_features(audio, sample_rate)
        norm = float(np.linalg.norm(feature))
        key = (
            "silence"
            if norm == 0.0
            else self._keys[int(np.argmax(self._features @ (feature / norm)))]
        )
        self._last = (audio, key)
        return key


class PrototypeEmbedder:
    """Returns the designed vector of the clip's voice, in this embedder's own rotation."""

    def __init__(self, name: str, voices: Voices, seed: int) -> None:
        self.name = name
        self.voices = voices
        rotation = np.linalg.qr(np.random.default_rng(seed).standard_normal((DIM, DIM)))[0]
        self._vectors = {k: (rotation @ v).astype(np.float32) for k, v in DESIGNED.items()}

    def voice(self, audio: np.ndarray, sample_rate: int) -> str:
        return self.voices.voice(audio, sample_rate)

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        key = self.voice(audio, sample_rate)
        return self._vectors["target" if key == "excited" else key]


class BandSnrQuality:
    """SNR from the voices' band against the burst's band, per 50 ms frame; C50 fixed at 40 dB."""

    hop_s = 0.05

    def frames(self, audio: np.ndarray, sample_rate: int) -> QualityTrack:
        size = int(self.hop_s * sample_rate)
        count = int(np.ceil(len(audio) / size))
        signal = np.pad(np.asarray(audio, dtype=np.float64), (0, count * size - len(audio)))
        power = np.abs(np.fft.rfft(signal.reshape(count, size), axis=1)) ** 2
        freqs = np.fft.rfftfreq(size, 1.0 / sample_rate)
        voice = power[:, (freqs >= 100.0) & (freqs <= 3800.0)].sum(axis=1)
        burst = power[:, (freqs >= 5000.0) & (freqs <= 7800.0)].sum(axis=1)
        snr = np.clip(10.0 * np.log10((voice + 1e-12) / (burst + 1e-12)), -20.0, 60.0)
        return QualityTrack(hop_s=self.hop_s, snr_db=snr, c50_db=np.full(count, 40.0))


def quick_prosody(audio: np.ndarray, sample_rate: int) -> Prosody:
    """A fast prosody: F0 from the autocorrelation peak, nuclei rate and active energy."""
    signal = np.asarray(audio[:sample_rate], dtype=np.float64)
    spectrum = np.fft.rfft(signal, 2 * len(signal))
    corr = np.fft.irfft(np.abs(spectrum) ** 2)[: len(signal)]
    lo, hi = sample_rate // 400, sample_rate // 60
    lag = lo + int(np.argmax(corr[lo:hi]))
    return Prosody(
        f0_mean_hz=float(sample_rate / lag),
        f0_std_hz=0.0,
        rate_sps=prosody_mod.rate_from_nuclei(audio, sample_rate),
        energy_db=prosody_mod.active_energy_db(audio, sample_rate),
    )


class Distorter:
    """A separator that colours the voice: a low-pass at `cutoff_hz`."""

    def __init__(self, name: str, cutoff_hz: float) -> None:
        self.name = name
        self.cutoff_hz = cutoff_hz

    def dialogue(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        return lowpass(audio, self.cutoff_hz)


class BedRemover:
    """A separator that removes the music bed exactly and leaves the voices untouched."""

    def __init__(self, name: str, bed: np.ndarray) -> None:
        self.name = name
        self.bed = bed

    def dialogue(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        return (np.asarray(audio) - self.bed[: len(audio)]).astype(np.float32)


# Episodes.


@dataclass
class Episode:
    """A synthetic episode: its channels, the diarizer's script, and the ground truth."""

    audio: np.ndarray
    turns: list[Turn] = field(default_factory=list)
    overlaps: list[Span] = field(default_factory=list)
    groups: dict[str, list[Span]] = field(default_factory=dict)


class Builder:
    """Places lines on a timeline and scripts the diarizer's turns."""

    def __init__(self, channels: int = 1) -> None:
        self.parts: list[tuple[int, np.ndarray, float]] = []
        self.turns: list[Turn] = []
        self.overlaps: list[Span] = []
        self.groups: dict[str, list[Span]] = {}
        self.channels = channels
        self.t = 0.5
        self.k = 0

    def place(self, channel: int, audio: np.ndarray, start: float) -> None:
        self.parts.append((channel, audio, start))

    def group(self, name: str, span: Span) -> None:
        self.groups.setdefault(name, []).append(span)

    def line(self, voice: str, seconds: float, group: str, channel: int = 0, gain_db: float = 0.0):
        audio = take(voice, seconds, self.k) * np.float32(10 ** (gain_db / 20.0))
        self.k += 1
        self.place(channel, audio, self.t)
        self.turns.append(Turn(self.t, self.t + seconds, LABELS[voice]))
        self.group(group, Span(self.t, self.t + seconds))
        self.t += seconds + GAP

    def build(self, tail: float = 1.0) -> Episode:
        length = int(round((self.t + tail) * RATE))
        audio = np.zeros((length, self.channels), dtype=np.float64)
        for channel, part, start in self.parts:
            at = int(round(start * RATE))
            audio[at : at + len(part), channel] += part
        out = audio[:, 0] if self.channels == 1 else audio
        return Episode(out.astype(np.float32), self.turns, self.overlaps, self.groups)


def main_episode() -> Episode:
    """Mono: the target (calm and excited), an impostor, two others, overlaps, a bed."""
    # 17 calm lines of three 1.5 s tiles each, plus the mixed turn's six, put the window model's
    # p02 above its one other-voice tile.
    b = Builder()
    for i in range(17):
        b.line("target", 5.0, "calm")
        if i % 3 == 2:
            b.line("excited", 4.0, "excited", gain_db=3.0)
        if i % 4 == 3:
            b.line("impostor", 4.0, "impostor")
        if i % 4 == 1:
            b.line("other_a" if i % 8 == 1 else "other_b", 4.0, "other")
        if i == 4:
            t = b.t  # the target, then another voice talking over its last 2 s
            b.place(0, take("target", 5.0, 900), t)
            b.place(0, take("other_a", 5.0, 901), t + 3.0)
            b.turns += [Turn(t, t + 5.0, "spk-01"), Turn(t + 5.0, t + 8.0, "spk-04")]
            b.overlaps.append(Span(t + 3.0, t + 5.0))
            b.group("overlap", Span(t, t + 5.0))
            b.t = t + 8.0 + GAP
        if i == 9:
            t = b.t  # another voice, then the target talking over its last 2 s
            b.place(0, take("other_b", 5.0, 902), t)
            b.place(0, take("target", 5.0, 903), t + 3.0)
            b.turns += [Turn(t, t + 3.0, "spk-05"), Turn(t + 3.0, t + 8.0, "spk-01")]
            b.overlaps.append(Span(t + 3.0, t + 5.0))
            b.group("overlap", Span(t + 3.0, t + 8.0))
            b.t = t + 8.0 + GAP
        if i == 13:
            t = b.t  # one diarized turn with another voice inside it, missed by the diarizer
            b.place(0, take("target", 4.5, 904), t)
            b.place(0, take("other_a", 1.6, 905), t + 4.5)
            b.place(0, take("target", 3.5, 906), t + 6.1)
            b.turns.append(Turn(t, t + 9.6, "spk-01"))
            b.group("mixed", Span(t, t + 9.6))
            b.t = t + 9.6 + GAP
    episode = b.build()
    episode.audio = (episode.audio + music_bed(len(episode.audio) / RATE)).astype(np.float32)
    return episode


FL, FR, FC = 0, 1, 2
SNR_FC_DB = 6.0


def surround_episode() -> Episode:
    """5.1: dialogue in FC, an off-centre line, an SNR fallback line and a SIM fallback line."""
    b = Builder(channels=6)
    # 38 calm lines and the two fallback lines put p02 at about 0.85 over the one 0.3 cut.
    for i in range(38):
        b.line("target", 3.2, "calm", channel=FC)
        if i % 11 == 4:
            b.line("impostor", 3.2, "impostor", channel=FC)
        if i % 11 == 8:
            b.line("other_a" if i % 2 else "other_b", 3.2, "other", channel=FC)
        if i == 10:  # off-centre: the target in FL and FR only
            line = take("target", 4.0, 950, effect=False)
            b.place(FL, line, b.t)
            b.place(FR, line, b.t)
            b.group("offcentre", Span(b.t, b.t + 4.0))
            b.t += 4.0 + GAP
        if i == 20:  # SNR fallback: dry in FL and FR; in FC a copy 15 dB down under a burst
            line = take("target", 4.0, 951, effect=False)
            copy = line * np.float32(10 ** (-15.0 / 20.0))
            burst = band_noise(4.0, 5000.0, 7800.0, seed=3) * np.float32(
                rms(copy) * 10 ** (-SNR_FC_DB / 20.0)
            )
            b.place(FL, line, b.t)
            b.place(FR, line, b.t)
            b.place(FC, copy + burst, b.t)
            b.turns.append(Turn(b.t, b.t + 4.0, "spk-01"))
            b.group("snr", Span(b.t, b.t + 4.0))
            b.t += 4.0 + GAP
        if i == 30:  # SIM fallback: dry in FL and FR; in FC a muffled rendering at full level
            line = take("target", 4.0, 952, effect=False)
            b.place(FL, line, b.t)
            b.place(FR, line, b.t)
            muffled = lowpass(line, 700.0)
            b.place(FC, muffled * np.float32(0.5 / np.max(np.abs(muffled))), b.t)
            b.turns.append(Turn(b.t, b.t + 4.0, "spk-01"))
            b.group("sim", Span(b.t, b.t + 4.0))
            b.t += 4.0 + GAP
    return b.build()


def main_prototypes() -> dict[str, list[np.ndarray]]:
    return {
        "target": [take("target", 6.0, 500)],
        "excited": [take("excited", 6.0, 501)],
        "impostor": [take("impostor", 6.0, 502)],
        "other_a": [take("other_a", 6.0, 503)],
        "other_b": [take("other_b", 6.0, 504)],
        "music": [music_bed(6.0)],
    }


def surround_prototypes() -> dict[str, list[np.ndarray]]:
    dry = take("target", 6.0, 510, effect=False)
    return {
        "target": [dry],
        "impostor": [take("impostor", 6.0, 511)],
        "other_a": [take("other_a", 6.0, 512)],
        "other_b": [take("other_b", 6.0, 513)],
        "damaged": [lowpass(dry, 700.0)],
    }


def models_for(
    episode: Episode,
    prototypes: Mapping[str, Sequence[np.ndarray]],
    *,
    quality: object | None = None,
    separators: tuple[object, ...] = (),
) -> extract_stage.Models:
    voices = Voices(prototypes)
    return extract_stage.Models(
        separators=separators,  # type: ignore[arg-type]
        embedders=(
            PrototypeEmbedder("wavlm-sv", voices, seed=1),
            PrototypeEmbedder("ecapa", voices, seed=2),
        ),
        diarizer=FakeDiarizer(episode.turns, episode.overlaps),
        vad=FakeVad(range_db=20.0),
        quality=quality or FakeQualityEstimator(),  # type: ignore[arg-type]
        arousal=FakeArousal(),
        prosody=quick_prosody,
    )


# The root and the fake container boundary.


TRACK_REL = "seed/tracks/src-0001.wav"
CODEC = "pcm_f32le"


def write_track(path: Path, audio: np.ndarray) -> None:
    """Write the episode as a float WAV track; ffmpeg reads six channels as 5.1, FC third."""
    sf.write(str(path), audio, RATE, format="WAV", subtype="FLOAT")


def make_root(
    base: Path,
    episode: Episode,
    *,
    source_name: str = "show/episode one.mkv",
    clips: Mapping[str, np.ndarray] | None = None,
    anchors_csv: str | None = None,
) -> root_mod.Root:
    """A private root with one ingested source, its track, and the anchors given."""
    root = root_mod.init_root(base / "voice-root", "private")
    surround = episode.audio.ndim == 2
    write_track(root.path / TRACK_REL, episode.audio)
    record = SourceRecord(
        source_id="src-0001", path=f"seed/raw/{source_name}", container="wav",
        size_bytes=1, mtime_ns=0, streams=(), stream_index=0, audio_index=0, language="eng",
        codec=CODEC, profile=None, bitrate_bps=512000, bitrate_from="packets",
        sample_rate=RATE, channels=6 if surround else 1, layout="5.1" if surround else "mono",
        duration_s=len(episode.audio) / RATE, lossless=True, effective_bandwidth_hz=8000,
        sha256="0" * 64, track=TRACK_REL,
        primary_channel="FC" if surround else "downmix",
    )  # fmt: skip
    write_jsonl(root, "seed/inventory.jsonl", [record])
    for name, clip in (clips or {}).items():
        sf.write(str(root.path / "seed" / "reference" / name), clip, RATE, subtype="FLOAT")
    if anchors_csv is not None:
        (root.path / "seed" / "anchors.csv").write_text(anchors_csv, encoding="utf-8")
    return root


class StageDocker:
    """A `DockerOps` whose `run_logged` runs the stage in this process with the given models."""

    def __init__(self, root: root_mod.Root, models: extract_stage.Models) -> None:
        self.root = root
        self.models = models
        self.argvs: list[list[str]] = []

    def capture(self, argv: Sequence[str]) -> str:
        raise AssertionError("a cpu extract job runs no GPU preflight")

    def check(self, argv: Sequence[str]) -> None:
        raise AssertionError("extract builds nothing")

    def run_logged(self, argv: Sequence[str], log: Path) -> int:
        self.argvs.append(list(argv))
        at = list(argv).index("-m")
        module, rest = argv[at + 1], list(argv[at + 2 :])
        assert rest[:2] == ["--data", "/data"]
        rest[1] = str(self.root.path)
        stage = importlib.import_module(module)
        with (
            pytest.MonkeyPatch.context() as mp,
            log.open("w", encoding="utf-8") as handle,
            contextlib.redirect_stdout(handle),
            contextlib.redirect_stderr(handle),
        ):
            mp.setattr(stage, "load_models", lambda device: self.models)
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
def _patched(docker: StageDocker) -> Iterator[None]:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(extract_cmd, "make_docker", lambda: docker)
        mp.setattr(extract_cmd, "image_tag", lambda family: IMAGE)
        yield


def extract(root: root_mod.Root, docker: StageDocker, *args: str) -> tuple[int, str, str]:
    out, err = _Capture(), _Capture()
    argv = ["--root", str(root.path), "extract", "--run", "r1", *args]
    with _patched(docker), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(argv)
    return code, out.getvalue(), err.getvalue()


@dataclass
class Ran:
    root: root_mod.Root
    episode: Episode
    code: int
    out: str
    err: str
    records: list[SegmentRecord]
    report: dict[str, object]
    docker: StageDocker


def run_episode(
    base: Path, episode: Episode, models: extract_stage.Models, clip: np.ndarray
) -> Ran:
    root = make_root(base, episode, clips={"solo.wav": clip})
    docker = StageDocker(root, models)
    code, out, err = extract(root, docker)
    run_dir = root.path / "seed" / "runs" / "r1"
    records = read_jsonl(run_dir / "segments.jsonl", SegmentRecord) if code == 0 else []
    report = json.loads((run_dir / "report.json").read_text()) if code == 0 else {}
    return Ran(root, episode, code, out, err, records, report, docker)


def _within(span: Span, record: SegmentRecord, slack: float = 0.05) -> bool:
    return record.start_s >= span.start_s - slack and record.end_s <= span.end_s + slack


def _hits(spans: Sequence[Span], records: Sequence[SegmentRecord]) -> list[SegmentRecord]:
    return [r for r in records if any(_within(s, r) for s in spans)]


# The main synthetic episode.


@pytest.fixture(scope="module")
def main_run(tmp_path_factory: pytest.TempPathFactory) -> Ran:
    episode = main_episode()
    models = models_for(episode, main_prototypes())
    return run_episode(tmp_path_factory.mktemp("main"), episode, models, take("target", 8.0, 700))


def test_the_main_episode_exits_0(main_run: Ran) -> None:
    assert main_run.code == 0, main_run.err


def test_target_turns_are_retrieved_with_precision_1_and_recall_0_9(main_run: Ran) -> None:
    kept = [r for r in main_run.records if r.kept]
    truth = main_run.episode.groups["calm"] + main_run.episode.groups["excited"]
    on_target = _hits(truth, kept)
    assert kept and len(on_target) == len(kept)
    found = [s for s in truth if any(_within(s, r) for r in kept)]
    assert len(found) / len(truth) >= 0.9


def test_a_split_register_is_kept_as_flagged(main_run: Ran) -> None:
    excited = _hits(main_run.episode.groups["excited"], main_run.records)
    assert len(excited) == 5
    assert all(r.kept and r.cluster == "flagged" for r in excited)
    assert {r.flag_group for r in excited} == {"src-0001/spk-02"}
    calm = _hits(main_run.episode.groups["calm"], main_run.records)
    assert all(r.cluster == "primary" and r.flag_group is None for r in calm)
    roles = {c["diar_cluster"]: c["role"] for c in main_run.report["sources"][0]["clusters"]}
    assert roles["spk-01"] == "primary"
    assert roles["spk-02"] == "flagged"


def test_the_impostor_is_flagged_not_admitted_and_its_turns_dropped(main_run: Ran) -> None:
    roles = {c["diar_cluster"]: c for c in main_run.report["sources"][0]["clusters"]}
    assert roles["spk-03"]["role"] == "flagged"
    assert roles["spk-03"]["median_sim"]["wavlm-sv"] == pytest.approx(0.8, abs=0.01)
    impostor = _hits(main_run.episode.groups["impostor"], main_run.records)
    assert len(impostor) == 4
    for record in impostor:
        assert not record.kept
        assert "floor:wavlm-sv" in record.reasons and "floor:ecapa" in record.reasons


def test_other_voices_leave_no_candidate(main_run: Ran) -> None:
    roles = {c["diar_cluster"]: c["role"] for c in main_run.report["sources"][0]["clusters"]}
    assert roles["spk-04"] == roles["spk-05"] == "other"
    assert not [r for r in main_run.records if r.diar_cluster in ("spk-04", "spk-05")]


def test_the_centroid_is_unmoved_by_the_impostor(main_run: Ran) -> None:
    # The embedder gives every target turn the same vector, so a centroid built from primary
    # turns only is that vector exactly: the target's turns sit at 1.0 and the impostor's at its
    # designed 0.8. One impostor turn in the centroid would pull both off those values.
    primary = [r for r in main_run.records if r.kept and r.cluster == "primary"]
    assert primary
    for record in primary:
        assert record.sim == {"wavlm-sv": pytest.approx(1.0), "ecapa": pytest.approx(1.0)}
    for record in _hits(main_run.episode.groups["impostor"], main_run.records):
        assert record.sim == {"wavlm-sv": pytest.approx(0.8), "ecapa": pytest.approx(0.8)}
    for name in ("wavlm-sv", "ecapa"):
        assert main_run.report["distribution"][name]["median"] == pytest.approx(1.0)


def test_every_segment_touching_an_overlap_is_dropped(main_run: Ran) -> None:
    overlaps = main_run.episode.overlaps
    for record in main_run.records:
        if any(max(record.start_s, o.start_s) < min(record.end_s, o.end_s) for o in overlaps):
            assert not record.kept
    on_overlap = _hits(main_run.episode.groups["overlap"], main_run.records)
    assert len(on_overlap) == 2
    assert all(not r.kept and r.reasons for r in on_overlap)
    assert any(r.reasons == ("overlap",) and r.overlap_ms == 2000 for r in on_overlap)


def test_a_segment_with_another_voice_inside_is_dropped_by_the_window_check(main_run: Ran) -> None:
    (mixed,) = _hits(main_run.episode.groups["mixed"], main_run.records)
    assert not mixed.kept
    assert mixed.reasons == ("window",)


def test_the_in_show_effect_is_kept(main_run: Ran) -> None:
    kept = [r for r in main_run.records if r.kept and r.cluster == "primary"]
    audio, rate = sf.read(str(main_run.root.path / kept[0].audio), dtype="float32")
    assert rate == RATE
    n = len(audio)
    effected = take("target", n / RATE + 0.1, 77)[:n]
    dry = take("target", n / RATE + 0.1, 77, effect=False)[:n]
    assert spectrum_distance_db(audio, effected, RATE) < spectrum_distance_db(audio, dry, RATE)
    assert spectrum_distance_db(audio, effected, RATE) < 2.0


def test_every_drop_records_a_reason(main_run: Ran) -> None:
    dropped = [r for r in main_run.records if not r.kept]
    assert dropped
    assert all(r.reasons for r in dropped)
    assert all(r.audio == "" and r.state_proposed == "" for r in dropped)
    kept = [r for r in main_run.records if r.kept]
    assert all(r.reasons == () for r in kept)


def test_kept_segments_have_audio_states_and_ids(main_run: Ran) -> None:
    ids = [r.segment_id for r in main_run.records]
    assert ids == [f"seg-{i:06d}" for i in range(1, len(ids) + 1)]
    run_dir = main_run.root.path / "seed" / "runs" / "r1"
    proposals = [
        json.loads(line) for line in (run_dir / "proposals.jsonl").read_text().splitlines()
    ]
    kept = [r for r in main_run.records if r.kept]
    assert [p["segment_id"] for p in proposals] == [r.segment_id for r in kept]
    for record, proposal in zip(kept, proposals, strict=True):
        assert record.state_proposed == proposal["state_proposed"] in SPEAKING_STATES
        assert proposal["margin"] >= 0.0
        assert (main_run.root.path / record.audio).is_file()
        assert record.channel == "FC" or record.channel == "downmix"
        assert 3.0 <= record.end_s - record.start_s <= 12.0 and record.set == "short"
        assert record.separator == "none"


def test_the_report_feeds_the_seed_gate(main_run: Ran) -> None:
    gates = gates_mod.load(main_run.root.path).effective()
    result = seed_gate.evaluate(main_run.report, gates)
    names = [c.name for c in result.criteria]
    assert "one speaker" in names and "clean minutes" in names
    one = next(c for c in result.criteria if c.name == "one speaker")
    assert one.passed is True
    assert main_run.report["thresholds"] == dict(sorted(gates.items()))


def test_the_report_holds_minutes_distributions_and_purity(main_run: Ran) -> None:
    report = main_run.report
    kept = [r for r in main_run.records if r.kept and r.cluster == "primary"]
    minutes = sum(r.end_s - r.start_s for r in kept) / 60.0
    assert report["clean_minutes"] == pytest.approx(minutes, abs=1e-3)
    assert set(report["minutes_per_state"]) == set(SPEAKING_STATES)
    assert sum(report["minutes_per_state"].values()) == pytest.approx(minutes, abs=1e-3)
    assert report["minutes_per_channel"]["FC"] == 0.0
    assert report["minutes_per_channel"]["downmix"] == pytest.approx(minutes, abs=1e-3)
    for key in ("distribution", "windows"):
        for name in ("wavlm-sv", "ecapa"):
            assert set(report[key][name]) == {"n", "median", "p02", "p10", "floor"}
    assert report["distribution"]["wavlm-sv"]["n"] == 18
    assert isinstance(report["purity_p10"], float) and report["purity_p10"] > 0.5
    assert report["flagged"]["segments"] == 5 and report["flagged"]["groups"] == 1
    source = report["sources"][0]
    assert source["source_id"] == "src-0001" and source["codec"] == CODEC
    assert "path" not in source


def test_the_command_prints_numbers_and_ids(main_run: Ran) -> None:
    assert "extract run r1:" in main_run.out
    assert "clean minutes" in main_run.out
    assert "src-0001: separator none" in main_run.out
    assert "episode one" not in main_run.out


def test_extract_builds_a_local_job_in_the_extract_image(main_run: Ran) -> None:
    (argv,) = main_run.docker.argvs
    assert argv[:2] == ["docker", "run"]
    assert "--network" in argv and "none" in argv
    assert "--gpus" not in argv
    assert IMAGE in argv
    assert "/opt/venv-extract/bin/python" in argv
    assert argv[-4:] == ["--run", "r1", "--device", "cpu"]
    assert f"{main_run.root.path / TRACK_REL}:/data/{TRACK_REL}:ro" in argv
    assert f"{main_run.root.path / 'seed' / 'reference'}:/data/seed/reference:ro" in argv
    assert [a for a in argv if a.endswith(":/data/seed/runs/r1")]
    job = json.loads(next((main_run.root.path / "runs").glob("job-*/job.json")).read_text())
    assert job["family"] == "extract" and job["gpu"] is False


def test_extract_has_no_executor_option(main_run: Ran, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["extract", "--run", "r2", "--executor", "remote:x"])
    names: list[str] = []
    real = jobs.executor_for

    def spy(name: str, **kwargs: object) -> jobs.Executor:
        names.append(name)
        return real(name, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(jobs, "executor_for", spy)
    code, _, _ = extract(main_run.root, main_run.docker)
    assert code == 2  # run r1 exists; refused before any job, after nothing was chosen
    assert names == []
    small = small_episode()
    root = make_root(
        main_run.root.path.parent / "spy", small, clips={"a.wav": take("target", 6.0, 1)}
    )
    docker = StageDocker(root, models_for(small, main_prototypes()))
    code, _, err = extract(root, docker)
    assert code == 0, err
    assert names == ["local"]
    code, _, err = extract(root, docker)  # a finished run is never overwritten
    assert code == 2
    assert "already holds a run" in err
    assert names == ["local"] and len(docker.argvs) == 1


# The 5.1 episode.


@pytest.fixture(scope="module")
def surround_run(tmp_path_factory: pytest.TempPathFactory) -> Ran:
    episode = surround_episode()
    models = models_for(episode, surround_prototypes(), quality=BandSnrQuality())
    clip = take("target", 8.0, 701, effect=False)
    return run_episode(tmp_path_factory.mktemp("surround"), episode, models, clip)


def test_the_surround_episode_exits_0(surround_run: Ran) -> None:
    assert surround_run.code == 0, surround_run.err


def test_a_damaged_centre_cut_is_kept_from_the_downmix_as_fc_failed_sim(surround_run: Ran) -> None:
    (record,) = _hits(surround_run.episode.groups["sim"], surround_run.records)
    assert record.kept
    assert (record.channel, record.channel_reason, record.cluster) == (
        "downmix",
        "fc-failed:sim",
        "primary",
    )
    voices = Voices(surround_prototypes())
    fc = decode(surround_run.root, "FC")
    span = surround_run.episode.groups["sim"][0]
    assert voices.voice(fc[int(span.start_s * RATE) : int(span.end_s * RATE)], RATE) == "damaged"


def test_the_impostors_downmix_cuts_fail_too(surround_run: Ran) -> None:
    impostor = _hits(surround_run.episode.groups["impostor"], surround_run.records)
    assert len(impostor) == 4
    for record in impostor:
        assert not record.kept
        assert "floor:wavlm-sv" in record.reasons
        assert "downmix:floor:wavlm-sv" in record.reasons


def test_an_off_centre_line_is_kept_as_flagged(surround_run: Ran) -> None:
    (record,) = _hits(surround_run.episode.groups["offcentre"], surround_run.records)
    assert record.kept
    assert (record.channel, record.channel_reason, record.cluster) == (
        "downmix",
        "off-centre",
        "flagged",
    )
    assert record.diar_cluster is None
    assert record.flag_group == record.segment_id


def decode(root: root_mod.Root, channel: str) -> np.ndarray:
    return _decode(str(root.path / TRACK_REL), channel)


@functools.cache
def _decode(track: str, channel: str) -> np.ndarray:
    proc = subprocess.run(decode_argv(track, CODEC, channel), capture_output=True, check=True)
    return np.frombuffer(proc.stdout, dtype="<f4")


def test_a_buried_centre_line_is_kept_from_the_downmix_as_fc_failed_snr(surround_run: Ran) -> None:
    (record,) = _hits(surround_run.episode.groups["snr"], surround_run.records)
    assert record.kept
    assert (record.channel, record.channel_reason) == ("downmix", "fc-failed:snr")
    floor = gates_mod.load(None).value("segment.snr_min_db")
    assert record.snr_db >= floor
    span = Span(record.start_s, record.end_s)
    quality = BandSnrQuality()
    fc_snr, _ = span_quality(quality.frames(decode(surround_run.root, "FC"), RATE), span)
    dm_snr, _ = span_quality(quality.frames(decode(surround_run.root, "downmix"), RATE), span)
    assert fc_snr < floor < dm_snr


def test_every_other_kept_surround_segment_is_fc(surround_run: Ran) -> None:
    groups = surround_run.episode.groups
    special = [span for g in ("sim", "snr", "offcentre") for span in groups[g]]
    kept = [r for r in surround_run.records if r.kept]
    rest = [r for r in kept if not any(_within(s, r) for s in special)]
    assert len(rest) == 38
    assert all(r.channel == "FC" and r.channel_reason is None for r in rest)
    channels = surround_run.report["minutes_per_channel"]
    assert channels["FC"] > 0.0 and channels["downmix"] > 0.0
    assert surround_run.report["channel_reasons"] == {
        "fc-failed:sim": 1,
        "fc-failed:snr": 1,
        "off-centre": 1,
    }


# The turn record, on a three-cluster episode: the target (primary), the impostor (flagged at its
# designed 0.8) and another voice (other).

TURN_FIELDS = {
    "source_id", "turn", "start_s", "end_s", "diar_cluster", "role", "embedded",
    "touches_overlap", "sim_anchor", "sim_target", "gaps", "passed", "reasons",
}  # fmt: skip


def turn_episode() -> Episode:
    """Six target lines, two impostor and two other-voice lines, then a 0.4 s target turn.

    The diarizer reports an overlap on the last second of the first other-voice turn.
    """
    b = Builder()
    for i in range(6):
        b.line("target", 4.0, "calm")
        if i % 3 == 1:
            b.line("impostor", 4.0, "impostor")
        if i % 3 == 2:
            b.line("other_a", 4.0, "other")
    b.line("target", 0.4, "short")
    first = b.groups["other"][0]
    b.overlaps.append(Span(first.end_s - 1.0, first.end_s))
    return b.build()


@dataclass
class TurnRun:
    ran: Ran
    rows: list[dict]
    outputs: list[str]


def _turn_run(base: Path, overlay: Mapping[str, float] | None = None) -> TurnRun:
    episode = turn_episode()
    root = make_root(base, episode, clips={"solo.wav": take("target", 6.0, 1)})
    if overlay is not None:
        (root.path / "config" / "gates.json").write_text(json.dumps(overlay), encoding="utf-8")
    docker = StageDocker(root, models_for(episode, main_prototypes()))
    code, out, err = extract(root, docker)
    run_dir = root.path / "seed" / "runs" / "r1"
    assert code == 0, err
    report = json.loads((run_dir / "report.json").read_text())
    text = (run_dir / extract_cmd.TURNS_NAME).read_text(encoding="utf-8")
    rows = [json.loads(line) for line in text.splitlines()]
    outputs = json.loads((run_dir / "outputs.json").read_text())["files"]
    ran = Ran(root, episode, code, out, err, [], report, docker)
    return TurnRun(ran, rows, [f["rel"] for f in outputs])


@pytest.fixture(scope="module")
def turn_run(tmp_path_factory: pytest.TempPathFactory) -> TurnRun:
    return _turn_run(tmp_path_factory.mktemp("turns"))


@pytest.fixture(scope="module")
def no_target_run(tmp_path_factory: pytest.TempPathFactory) -> TurnRun:
    # No cluster reaches an anchor floor above 1, so none is primary and no target model exists.
    return _turn_run(tmp_path_factory.mktemp("no-target"), {"retrieval.anchor_floor.ecapa": 1.5})


def test_every_turn_of_a_three_cluster_source_has_a_row(turn_run: TurnRun) -> None:
    turns = turn_run.ran.episode.turns
    rows = turn_run.rows
    assert extract_cmd.TURNS_NAME == "turns.jsonl"
    assert "seed/runs/r1/turns.jsonl" in turn_run.outputs
    assert len(rows) == len(turns) == 11
    assert all(set(row) == TURN_FIELDS for row in rows)
    assert [row["turn"] for row in rows] == list(range(len(turns)))
    for row, turn in zip(rows, sorted(turns, key=lambda t: t.start_s), strict=True):
        assert row["source_id"] == "src-0001"
        assert (row["start_s"], row["end_s"]) == (round(turn.start_s, 3), round(turn.end_s, 3))
        assert row["diar_cluster"] == turn.speaker
    roles = {row["diar_cluster"]: row["role"] for row in rows}
    assert roles == {"spk-01": "primary", "spk-03": "flagged", "spk-04": "other"}
    others = [row for row in rows if row["role"] == "other"]
    assert len(others) == 2
    for row in others:
        assert row["embedded"] is True
        assert row["sim_anchor"] == {"wavlm-sv": pytest.approx(0.1), "ecapa": pytest.approx(0.1)}
        assert row["sim_target"] == {"wavlm-sv": pytest.approx(0.1), "ecapa": pytest.approx(0.1)}
        # The sole rival is the other voice's own cluster, so its gap is 0.1 less 1.0.
        assert row["gaps"] == {"wavlm-sv": pytest.approx(-0.9), "ecapa": pytest.approx(-0.9)}
        assert row["passed"] is False and "margin:ecapa" in row["reasons"]
    for row in rows:
        for key in ("sim_anchor", "sim_target", "gaps"):
            assert all(round(v, 4) == v for v in (row[key] or {}).values())
    target = [row for row in rows if row["role"] == "primary" and row["embedded"]]
    assert len(target) == 6
    assert all(row["passed"] is True and row["reasons"] == [] for row in target)
    impostor = [row for row in rows if row["role"] == "flagged"]
    assert [row["sim_target"]["wavlm-sv"] for row in impostor] == [pytest.approx(0.8)] * 2


def test_a_turn_under_half_a_second_is_not_embedded(turn_run: TurnRun) -> None:
    (short,) = (row for row in turn_run.rows if row["end_s"] - row["start_s"] < 0.5)
    assert (short["diar_cluster"], short["role"], short["embedded"]) == ("spk-01", "primary", False)
    assert short["sim_anchor"] is None and short["sim_target"] is None and short["gaps"] is None
    assert short["passed"] is None and short["reasons"] == []


def test_a_turn_meeting_an_overlap_is_marked(turn_run: TurnRun) -> None:
    first = turn_run.ran.episode.groups["other"][0]
    touching = [row for row in turn_run.rows if row["touches_overlap"]]
    assert [(row["start_s"], row["diar_cluster"]) for row in touching] == [
        (round(first.start_s, 3), "spk-04")
    ]


def test_with_no_target_model_the_target_sims_gaps_and_verdicts_are_null(
    no_target_run: TurnRun,
) -> None:
    rows = no_target_run.rows
    assert no_target_run.ran.report["distribution"] is None
    assert len(rows) == 11 and {row["role"] for row in rows} == {"other"}
    for row in rows:
        assert row["sim_target"] is None and row["gaps"] is None
        assert row["passed"] is None and row["reasons"] == []
        assert (row["sim_anchor"] is not None) == row["embedded"]
    (source,) = no_target_run.ran.report["sources"]
    assert source["primary_s"] == 0


def test_the_report_gives_each_clusters_seconds_and_the_primary_seconds(
    turn_run: TurnRun,
) -> None:
    (source,) = turn_run.ran.report["sources"]
    seconds = {c["diar_cluster"]: c["seconds"] for c in source["clusters"]}
    # A cluster's seconds count every one of its turns, the 0.4 s one too.
    assert seconds == {"spk-01": 24.4, "spk-03": 8.0, "spk-04": 8.0}
    assert source["primary_s"] == 24.4
    out = turn_run.ran.out
    assert "src-0001: separator none" in out and "primary 24.400 s" in out
    assert "spk-01 primary: 6 turns, 24.400 s, median" in out
    assert "spk-04 other: 2 turns, 8.000 s, median" in out


def test_anchor_lines_mark_each_embedder_below_its_value() -> None:
    rows = [
        {"anchor": "clip-1", "sim": {"wavlm-sv": 0.85, "ecapa": 0.40}},
        {"anchor": "time-1", "sim": {"wavlm-sv": 0.79, "ecapa": 0.50}},
    ]
    thresholds = {"anchors.sim_min.wavlm-sv": 0.8, "anchors.sim_min.ecapa": 0.45}
    assert extract_cmd.anchor_lines(rows, thresholds) == [
        "  anchor clip-1: wavlm-sv 0.850, ecapa 0.400 below",
        "  anchor time-1: wavlm-sv 0.790 below, ecapa 0.500",
    ]


# Separator choice, on a synthetic episode with the spectral fake embedder.


def small_episode(bed_db: float | None = None) -> Episode:
    b = Builder()
    for i in range(6):
        b.line("target", 4.0, "calm")
        if i % 3 == 1:
            b.line("other_a", 4.0, "other")
    episode = b.build()
    return episode


def _spectral_models(separators: tuple[object, ...]) -> extract_stage.Models:
    return extract_stage.Models(
        separators=separators,  # type: ignore[arg-type]
        embedders=(FakeEmbedder("wavlm-sv"), FakeEmbedder("ecapa")),
        diarizer=FakeDiarizer(),
        vad=FakeVad(range_db=20.0),
        quality=FakeQualityEstimator(),
        arousal=FakeArousal(),
    )


def _scores(audio: np.ndarray, models: extract_stage.Models) -> dict[str, float | None]:
    outputs = extract_stage.separate(audio, RATE, models.separators)
    solo = take("target", 8.0, 700)
    reference = retrieve.centroids([extract_stage.embed(models.embedders, solo, RATE)])
    return extract_stage.candidate_scores(outputs, RATE, models, dict.fromkeys(outputs, reference))


def test_the_separator_that_preserves_the_target_is_chosen() -> None:
    episode = small_episode()
    bed = 4.0 * music_bed(len(episode.audio) / RATE) * np.float32(10 ** (20.0 / 20.0)) / 4.0
    audio = (episode.audio + bed).astype(np.float32)
    models = _spectral_models((BedRemover("dme", bed), Distorter("roformer", 1000.0)))
    scores = _scores(audio, models)
    assert scores["dme"] > scores["none"] > scores["roformer"]  # type: ignore[operator]
    assert retrieve.choose_separator(scores) == "dme"


def test_none_is_chosen_when_every_separator_colours_a_clean_channel() -> None:
    episode = small_episode()
    models = _spectral_models((Distorter("dme", 2500.0), Distorter("roformer", 1000.0)))
    scores = _scores(episode.audio, models)
    assert retrieve.choose_separator(scores) == "none"
    assert scores["none"] > 0.95  # type: ignore[operator]


def test_a_candidate_without_a_reference_scores_none() -> None:
    models = _spectral_models((Distorter("dme", 2500.0),))
    audio = small_episode().audio
    outputs = extract_stage.separate(audio, RATE, models.separators)
    scores = extract_stage.candidate_scores(outputs, RATE, models, {"none": None, "dme": None})
    assert scores == {"none": None, "dme": None}


def test_a_separator_named_none_is_refused() -> None:
    with pytest.raises(extract_stage.StageError):
        extract_stage.separate(np.zeros(RATE, dtype=np.float32), RATE, (Distorter("none", 1.0),))


# Separator choice on the provisional target turns. Through the low-pass `dme`, the target sits at
# 0.9 to the solo clip and the impostor at 0.85; through `none` they sit at 1.0 and 0.8. When the
# impostor holds most windows, the median of the best windows is the impostor's, so `dme` wins on
# windows, while on the target's own turns `none` wins.

DME_CUTOFF_HZ = 1500.0


class CountingDiarizer(FakeDiarizer):
    """The scripted diarizer, keeping the audio of each `turns` call."""

    def __init__(self, turns: Sequence[Turn] = (), overlaps: Sequence[Span] = ()) -> None:
        super().__init__(turns, overlaps)
        self.heard: list[np.ndarray] = []

    def turns(self, audio: np.ndarray, sample_rate: int) -> list[Turn]:
        self.heard.append(audio)
        return super().turns(audio, sample_rate)


def choice_episode(target_lines: int, impostor_lines: int) -> Episode:
    """The target's and the impostor's 4 s lines, interleaved, the impostor's first."""
    b = Builder()
    for i in range(max(target_lines, impostor_lines)):
        if i < impostor_lines:
            b.line("impostor", 4.0, "impostor")
        if i < target_lines:
            b.line("target", 4.0, "calm")
    return b.build()


def choice_models(episode: Episode) -> extract_stage.Models:
    target, impostor = take("target", 6.0, 500), take("impostor", 6.0, 502)
    prototypes = {
        "target": [target],
        "impostor": [impostor],
        "target_dme": [lowpass(target, DME_CUTOFF_HZ)],
        "impostor_dme": [lowpass(impostor, DME_CUTOFF_HZ)],
    }
    voices = Voices(prototypes)
    return extract_stage.Models(
        separators=(Distorter("dme", DME_CUTOFF_HZ),),  # type: ignore[arg-type]
        embedders=(
            PrototypeEmbedder("wavlm-sv", voices, seed=1),
            PrototypeEmbedder("ecapa", voices, seed=2),
        ),
        diarizer=CountingDiarizer(episode.turns, episode.overlaps),
        vad=FakeVad(range_db=20.0),
        quality=FakeQualityEstimator(),
        arousal=FakeArousal(),
        prosody=quick_prosody,
    )


def _choice_run(base: Path, target_lines: int, impostor_lines: int) -> Ran:
    episode = choice_episode(target_lines, impostor_lines)
    return run_episode(base, episode, choice_models(episode), take("target", 8.0, 700))


def _job_log(ran: Ran) -> str:
    (log,) = (ran.root.path / "runs").glob("job-*/log.txt")
    return log.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def flip_run(tmp_path_factory: pytest.TempPathFactory) -> Ran:
    return _choice_run(tmp_path_factory.mktemp("flip"), target_lines=4, impostor_lines=12)


def test_a_window_winner_on_another_speakers_windows_loses_on_the_target_turns(
    flip_run: Ran,
) -> None:
    assert flip_run.code == 0, flip_run.err
    (source,) = flip_run.report["sources"]
    assert source["separator_window_scores"] == {
        "none": pytest.approx(0.8),
        "dme": pytest.approx(0.85),
    }
    assert source["separator_provisional"] == "dme"
    assert source["separator_basis"] == "turns"
    assert source["separator_turns"] == 4
    assert source["separator_scores"] == {"none": pytest.approx(1.0), "dme": pytest.approx(0.9)}
    assert source["separator"] == "none"
    assert source["separator_rediarized"] is True
    assert all(r.separator == "none" for r in flip_run.records)


def test_the_diarizer_runs_again_on_the_chosen_output_when_the_choice_flips(flip_run: Ran) -> None:
    diarizer = flip_run.docker.models.diarizer
    assert isinstance(diarizer, CountingDiarizer)
    first, second = diarizer.heard
    voices = Voices(
        {
            "target": [take("target", 6.0, 500)],
            "target_dme": [lowpass(take("target", 6.0, 500), DME_CUTOFF_HZ)],
        }
    )
    span = flip_run.episode.groups["calm"][0]
    at = slice(int(span.start_s * RATE), int(span.end_s * RATE))
    assert voices.voice(first[at], RATE) == "target_dme"
    assert voices.voice(second[at], RATE) == "target"
    # The kept segments are cut from the second diarization, on `none`, at the target's 1.0.
    kept = [r for r in flip_run.records if r.kept and r.cluster == "primary"]
    assert kept and all(r.sim == {"wavlm-sv": pytest.approx(1.0), "ecapa": pytest.approx(1.0)}
                        for r in kept)  # fmt: skip


def test_the_log_and_the_report_lines_name_both_candidates_and_the_basis(flip_run: Ran) -> None:
    assert (
        "extract: src-0001 separator none (provisional dme, basis turns, 4 turns scored, "
        "diarized again)" in _job_log(flip_run)
    )
    assert "src-0001: separator none (none 1.000, dme 0.900)" in flip_run.out
    assert (
        "    separator basis turns: provisional dme, chosen none, 4 turns scored, diarized again"
        in flip_run.out
    )
    assert "    window scores: none 0.800, dme 0.850" in flip_run.out


def test_when_the_turn_scores_agree_the_provisional_turns_are_kept(tmp_path: Path) -> None:
    ran = _choice_run(tmp_path, target_lines=12, impostor_lines=4)
    assert ran.code == 0, ran.err
    (source,) = ran.report["sources"]
    assert source["separator_window_scores"] == {
        "none": pytest.approx(1.0),
        "dme": pytest.approx(0.9),
    }
    assert (source["separator_provisional"], source["separator"]) == ("none", "none")
    assert (source["separator_basis"], source["separator_turns"]) == ("turns", 12)
    assert source["separator_scores"] == {"none": pytest.approx(1.0), "dme": pytest.approx(0.9)}
    assert source["separator_rediarized"] is False
    diarizer = ran.docker.models.diarizer
    assert isinstance(diarizer, CountingDiarizer) and len(diarizer.heard) == 1
    assert "basis turns, 12 turns scored, diarized once)" in _job_log(ran)


def _direct_choice(
    diarizer: CountingDiarizer, references: Mapping[str, Mapping[str, np.ndarray] | None]
) -> tuple[extract_stage.SeparatorChoice, dict[str, float | None]]:
    episode = choice_episode(4, 12)
    models = choice_models(episode)
    models = extract_stage.Models(
        separators=models.separators, embedders=models.embedders, diarizer=diarizer,
        vad=models.vad, quality=models.quality, arousal=models.arousal,
    )  # fmt: skip
    outputs = extract_stage.separate(episode.audio, RATE, models.separators)
    windows = extract_stage.candidate_scores(outputs, RATE, models, references)
    choice = extract_stage.separator_choice(outputs, RATE, models, references)
    return choice, windows


def _solo_reference(models: extract_stage.Models) -> dict[str, np.ndarray]:
    return retrieve.centroids(
        [extract_stage.embed(models.embedders, take("target", 8.0, 700), RATE)]
    )


def _no_span_diarizers() -> dict[str, CountingDiarizer]:
    turns = choice_episode(4, 12).turns
    every = [Span(t.start_s, t.end_s) for t in turns]
    return {
        "no-turn": CountingDiarizer(),
        "every-turn-overlapped": CountingDiarizer(turns, every),
        "every-turn-too-short": CountingDiarizer(
            [Turn(t.start_s, t.start_s + 0.4, t.speaker) for t in turns]
        ),
    }


@pytest.mark.parametrize("case", sorted(_no_span_diarizers()))
def test_with_no_span_the_window_scores_decide(case: str) -> None:
    diarizer = _no_span_diarizers()[case]
    reference = _solo_reference(choice_models(choice_episode(4, 12)))
    choice, windows = _direct_choice(diarizer, {"none": reference, "dme": reference})
    assert (choice.basis, choice.spans, choice.rediarized) == ("windows", 0, False)
    assert choice.provisional == choice.separator == "dme"
    assert choice.scores == choice.window_scores == windows
    assert len(diarizer.heard) == 1


def test_with_no_reference_the_window_scores_decide() -> None:
    diarizer = CountingDiarizer(choice_episode(4, 12).turns)
    choice, windows = _direct_choice(diarizer, {"none": None, "dme": None})
    assert windows == {"none": None, "dme": None}
    assert (choice.basis, choice.spans, choice.rediarized) == ("windows", 0, False)
    assert choice.provisional == choice.separator == "none"
    assert choice.scores == choice.window_scores == windows
    assert len(diarizer.heard) == 1


def test_report_lines_print_the_basis_both_candidates_and_both_score_sets() -> None:
    source = {
        "source_id": "src-0002", "separator": "roformer", "primary_s": 12.0,
        "separator_scores": {"none": 0.61, "dme": 0.7, "roformer": 0.72},
        "separator_basis": "turns",
        "separator_window_scores": {"none": 0.5, "dme": 0.9, "roformer": 0.8},
        "separator_provisional": "dme", "separator_turns": 7, "separator_rediarized": True,
        "kept": 3, "flagged": 0, "dropped": 1, "clusters": [],
    }  # fmt: skip
    lines = extract_cmd.report_lines({"run": "r1", "sources": [source]})
    at = lines.index(
        "  src-0002: separator roformer (none 0.610, dme 0.700, roformer 0.720); primary 12.000 "
        "s; kept 3, flagged 0, dropped 1"
    )
    assert lines[at + 1 : at + 3] == [
        "    separator basis turns: provisional dme, chosen roformer, 7 turns scored, "
        "diarized again",
        "    window scores: none 0.500, dme 0.900, roformer 0.800",
    ]


def test_report_lines_still_print_a_report_without_the_basis() -> None:
    source = {"source_id": "src-0001", "separator": "none", "separator_scores": {"none": 0.9}}
    lines = extract_cmd.report_lines({"run": "r1", "sources": [source]})
    (line,) = (x for x in lines if x.startswith("  src-0001"))
    assert line.startswith("  src-0001: separator none (none 0.900); primary none s")
    assert not [x for x in lines if "separator basis" in x or "window scores" in x]


# Per-embedder gate values: the plan's thresholds hold one value per embedder for the anchor
# check, the anchor floor and the margin, since the two embedders' cosines sit on different
# scales.

EMBEDDER_NAMES = ("wavlm-sv", "ecapa")


def test_limits_read_one_value_per_embedder() -> None:
    values = gates_mod.load(None).effective()
    values["retrieval.anchor_floor.ecapa"] = 0.3
    values["retrieval.margin.wavlm-sv"] = -0.02
    limits = extract_stage.Limits.from_thresholds(values, EMBEDDER_NAMES)
    assert limits.anchor_floor == {"wavlm-sv": 0.6, "ecapa": 0.3}
    assert limits.margin == {"wavlm-sv": -0.02, "ecapa": 0.1}
    assert limits.anchor_sim_min == {"wavlm-sv": 0.6, "ecapa": 0.6}


def test_thresholds_lacking_an_embedders_value_are_refused_by_name() -> None:
    values = gates_mod.load(None).effective()
    del values["retrieval.margin.ecapa"]
    with pytest.raises(extract_stage.StageError, match=r"retrieval\.margin\.ecapa"):
        extract_stage.Limits.from_thresholds(values, EMBEDDER_NAMES)
    del values["anchors.sim_min.wavlm-sv"]
    with pytest.raises(extract_stage.StageError) as caught:
        extract_stage.Limits.from_thresholds(values, EMBEDDER_NAMES)
    assert "anchors.sim_min.wavlm-sv" in str(caught.value)
    assert "retrieval.margin.ecapa" in str(caught.value)


def _at(wavlm: float, ecapa: float) -> retrieve.Embedding:
    """A clip at exactly these cosines to the unit vector on axis 0, one per embedder."""

    def unit(cos: float) -> np.ndarray:
        v = np.zeros(8)
        v[0], v[1] = cos, np.sqrt(1.0 - cos**2)
        return v

    return {"wavlm-sv": unit(wavlm), "ecapa": unit(ecapa)}


def test_an_sdh_anchor_below_its_value_on_either_embedder_is_dropped() -> None:
    # Raj's anchor sits on axis 0. The first SDH anchor passes ECAPA's lower value although a
    # single value of 0.6 on both would drop it; the second is below ECAPA's, the third below
    # WavLM-SV's.
    span = sdh_stage.SdhSpan(start_s=0.0, end_s=4.0, offscreen=False, cues=1)
    offscreen = sdh_stage.SdhSpan(start_s=5.0, end_s=9.0, offscreen=True, cues=1)
    anchors = (span, offscreen, span)
    spans = sdh_stage.SourceSpans(
        found=3, found_offscreen=1, multi=0, merged=anchors, kept=anchors, held_out=(),
        anchors=anchors,
    )  # fmt: skip
    work = SimpleNamespace(
        source=SimpleNamespace(source_id="src-0001"),
        sdh_embs=[_at(0.85, 0.40), _at(0.85, 0.30), _at(0.75, 0.90)],
    )
    stage = SimpleNamespace(
        sdh={"src-0001": spans},
        limits=SimpleNamespace(anchor_sim_min={"wavlm-sv": 0.8, "ecapa": 0.35}),
    )
    own = _at(1.0, 1.0)
    centre, counts = extract_stage.Extraction._sdh_anchors(stage, [work], [own])  # type: ignore[arg-type]
    assert counts is not None
    assert (counts["joined"], counts["dropped"], counts["dropped_offscreen"]) == (1, 2, 1)
    expected = retrieve.centroids([own, _at(0.85, 0.40)])
    for name in EMBEDDER_NAMES:
        np.testing.assert_allclose(centre[name], expected[name])


# Anchors and refusals, on a small episode.


def _small_root(tmp_path: Path, **kwargs: object) -> tuple[root_mod.Root, StageDocker]:
    episode = small_episode()
    root = make_root(tmp_path, episode, **kwargs)  # type: ignore[arg-type]
    return root, StageDocker(root, models_for(episode, main_prototypes()))


def test_no_anchor_exits_2_before_any_job(tmp_path: Path) -> None:
    root, docker = _small_root(tmp_path)
    code, _, err = extract(root, docker)
    assert code == 2
    assert "no anchor" in err
    assert docker.argvs == []


def test_anchors_that_disagree_exit_3(tmp_path: Path) -> None:
    csv = "source,start_s,end_s\nshow/episode one.mkv,9.3,13.3\n"  # the first other-voice line
    root, docker = _small_root(
        tmp_path, clips={"solo.wav": take("target", 8.0, 1)}, anchors_csv=csv
    )
    code, out, err = extract(root, docker)
    assert code == 3, err
    assert "anchors disagree" in err
    assert "anchor clip-1" in out and "anchor time-1" in out
    # Two anchors: each one's SIM is to the other, so both fall below the default 0.6.
    assert "wavlm-sv 0.100 below, ecapa 0.100 below" in out
    assert not (root.path / "seed" / "runs" / "r1" / "segments.jsonl").exists()


def test_timestamp_anchors_alone_are_enough(tmp_path: Path) -> None:
    csv = "source,start_s,end_s\nshow/episode one.mkv,0.5,4.5\nshow/episode one.mkv,4.9,8.9\n"
    root, docker = _small_root(tmp_path, anchors_csv=csv)
    code, out, err = extract(root, docker)
    assert code == 0, err
    report = json.loads((root.path / "seed" / "runs" / "r1" / "report.json").read_text())
    assert report["anchors"]["clips"] == 0 and report["anchors"]["timestamps"] == 2


@pytest.mark.parametrize("seconds", [2.0, 31.0])
def test_a_solo_clip_out_of_range_exits_2(tmp_path: Path, seconds: float) -> None:
    root, docker = _small_root(tmp_path, clips={"solo.wav": take("target", 6.0, 1)})
    long = np.tile(take("target", 6.0, 1), 6)[: int(seconds * RATE)]
    sf.write(str(root.path / "seed" / "reference" / "solo.wav"), long, RATE)
    code, _, err = extract(root, docker)
    assert code == 2
    assert "clip-1" in err


def test_a_timestamp_anchor_out_of_range_exits_2_before_any_job(tmp_path: Path) -> None:
    csv = "source,start_s,end_s\nshow/episode one.mkv,0.5,2.0\n"
    root, docker = _small_root(tmp_path, anchors_csv=csv)
    code, _, err = extract(root, docker)
    assert code == 2
    assert "row 2" in err
    assert docker.argvs == []


def test_an_anchor_row_naming_an_unknown_source_exits_2_without_the_name(
    tmp_path: Path, canaries: Canaries, assert_no_canary: Callable[[str], None]
) -> None:
    csv = f"source,start_s,end_s\n{canaries.source_name},0.5,4.5\n"
    root, docker = _small_root(tmp_path, anchors_csv=csv)
    code, out, err = extract(root, docker)
    assert code == 2
    assert "row 2" in err and "not in the inventory" in err
    assert_no_canary(out + err)


def test_a_bad_anchor_header_exits_2(tmp_path: Path) -> None:
    root, docker = _small_root(tmp_path, anchors_csv="file,from,to\nx,1,5\n")
    assert extract(root, docker)[0] == 2


@pytest.mark.parametrize("run_id", ["../x", "a/b", ".hidden"])
def test_a_run_id_must_be_one_folder_name(tmp_path: Path, run_id: str) -> None:
    root, docker = _small_root(tmp_path, clips={"solo.wav": take("target", 6.0, 1)})
    out, err = _Capture(), _Capture()
    with _patched(docker), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(["--root", str(root.path), "extract", "--run", run_id])
    assert code == 2
    assert docker.argvs == []


def test_a_missing_inventory_exits_2(tmp_path: Path) -> None:
    root, docker = _small_root(tmp_path, clips={"solo.wav": take("target", 6.0, 1)})
    (root.path / "seed" / "inventory.jsonl").unlink()
    code, _, err = extract(root, docker)
    assert code == 2 and "ingest" in err


@pytest.mark.parametrize("kind", ["dir", "file"])
def test_a_root_in_a_git_work_tree_is_refused(tmp_path: Path, kind: str) -> None:
    root, docker = _small_root(tmp_path / "work", clips={"solo.wav": take("target", 6.0, 1)})
    git = tmp_path / "work" / ".git"
    if kind == "dir":
        git.mkdir()
    else:
        git.write_text("gitdir: elsewhere\n", encoding="utf-8")
    assert extract(root, docker)[0] == 2
    assert docker.argvs == []
    with pytest.raises(RootError):
        root.write_bytes("seed/runs/r1/report.json", b"{}")


def test_no_canary_in_output_argv_or_written_files(tmp_path: Path) -> None:
    canaries = Canaries()
    csv = f"source,start_s,end_s\n{canaries.source_name},0.5,4.5\n"
    clip_name = f"{canaries.transcript}.wav"
    episode = small_episode()
    root = make_root(tmp_path, episode, source_name=canaries.source_name,
                     clips={clip_name: take("target", 6.0, 1)}, anchors_csv=csv)  # fmt: skip
    terms = root.path / "guard" / "terms.txt"
    terms.write_text(canaries.term + "\n", encoding="utf-8")
    docker = StageDocker(root, models_for(episode, main_prototypes()))
    with pytest.MonkeyPatch.context() as mp:
        for name in KEY_ENV_NAMES:
            mp.setenv(name, canaries.key)
        code, out, err = extract(root, docker)
    assert code == 0, err
    planted = {
        root.path / "seed" / "inventory.jsonl",
        root.path / "seed" / "anchors.csv",
        root.path / "seed" / "reference" / clip_name,
        terms,
    }
    checked = 0
    for text in [out, err, *("\n".join(a) for a in docker.argvs)]:
        _assert_clean(text, canaries)
    for path in tmp_path.rglob("*"):
        if not path.is_file() or path in planted:
            continue
        _assert_clean(path.relative_to(tmp_path).as_posix(), canaries)
        _assert_clean(path.read_bytes().decode("latin-1"), canaries)
        checked += 1
    assert checked >= 6


# SDH anchors (T27), on the small episode. Her cues go through the scan stage's own parser from an
# SRT text built here, so a planted cue text and label meet the whole path to the report.

SDH_LABEL = "SPEAKER-QZV"
# (start, end, text): target lines sit at 0.5-4.5, 4.9-8.9, 13.7-17.7, 18.1-22.1, 22.5-26.5 and
# 31.3-35.3 s, the other voice at 9.3-13.3 and 26.9-30.9 s.
SDH_TRACK = (
    (0.6, 4.4, "[{label}] {text} one"),
    (4.9, 8.9, "-[{label}] {text} two\n- Someone else."),  # multi: never a span
    (9.4, 13.2, "[{label}]: {text} three"),  # mislabelled other voice: dropped by the check
    (13.8, 17.6, "<i>[{label}] {text} four</i>"),  # offscreen
    (18.2, 22.0, "[{label}] {text} five"),
    (27.0, 30.8, "[{label}] {text} six"),  # the 5th kept span, held out: the other voice
    (31.4, 35.2, "[{label}] {text} seven"),
    (35.9, 36.4, "[{label}] {text} eight"),  # too short to keep
)


def _stamp(seconds: float) -> str:
    return f"00:00:{seconds:06.3f}".replace(".", ",")


def write_sdh(root: root_mod.Root, label: str = SDH_LABEL) -> None:
    """Write `config/sdh.json` and src-0001's cue file as `sdh scan` would."""
    text = Canaries().transcript
    srt = "\n".join(
        f"{n}\n{_stamp(a)} --> {_stamp(b)}\n{line.format(label=label, text=text)}\n"
        for n, (a, b, line) in enumerate(SDH_TRACK, 1)
    )
    cues, total = sdh_stage.her_cues(srt, [label])
    assert (total, len(cues)) == (8, 8)
    (root.path / "config" / "sdh.json").write_text(json.dumps({"labels": [label]}), "utf-8")
    write_jsonl(root, "seed/sdh/src-0001.jsonl", [c.to_json() for c in cues])


@pytest.fixture(scope="module")
def sdh_run(tmp_path_factory: pytest.TempPathFactory) -> Ran:
    base = tmp_path_factory.mktemp("sdh")
    episode = small_episode()
    root = make_root(base, episode, clips={"solo.wav": take("target", 6.0, 1)})
    write_sdh(root)
    docker = StageDocker(root, models_for(episode, main_prototypes()))
    code, out, err = extract(root, docker)
    run_dir = root.path / "seed" / "runs" / "r1"
    report = json.loads((run_dir / "report.json").read_text()) if code == 0 else {}
    return Ran(root, episode, code, out, err, [], report, docker)


def test_sdh_spans_become_checked_anchors_with_every_fifth_held_out(sdh_run: Ran) -> None:
    assert sdh_run.code == 0, sdh_run.err
    # Held out, the other voice at 27.0 s is never an anchor, so only the one at 9.4 s is dropped.
    assert sdh_run.report["sdh"] == {
        "sources": 1, "found": 8, "found_offscreen": 1, "multi": 1, "merged": 7,
        "merged_offscreen": 1, "kept": 6, "kept_offscreen": 1, "held_out": 1,
        "held_out_offscreen": 0, "anchors": 5, "dropped": 1, "dropped_offscreen": 0, "joined": 4,
    }  # fmt: skip
    assert [row["anchor"] for row in sdh_run.report["anchors"]["sim"]] == ["clip-1"]
    assert "sdh spans: sources 1, found 8, found_offscreen 1, multi 1, merged 7" in sdh_run.out
    assert "held_out 1, held_out_offscreen 0, anchors 5, dropped 1" in sdh_run.out
    (argv,) = sdh_run.docker.argvs
    assert f"{sdh_run.root.path / 'seed' / 'sdh'}:/data/seed/sdh:ro" in argv


def test_no_sdh_cue_text_or_label_leaves_the_extract_path(sdh_run: Ran) -> None:
    planted = {
        sdh_run.root.path / "config" / "sdh.json",
        sdh_run.root.path / "seed" / "reference" / "solo.wav",
    }
    texts = [sdh_run.out, sdh_run.err, *("\n".join(a) for a in sdh_run.docker.argvs)]
    for path in sdh_run.root.path.rglob("*"):
        if path.is_file() and path not in planted:
            texts += [path.as_posix(), path.read_bytes().decode("latin-1")]
    assert len(texts) > 8
    for text in texts:
        assert SDH_LABEL.lower() not in text.lower()
        assert Canaries().transcript not in text.lower()


def test_anchors_that_disagree_still_exit_3_with_sdh_spans(tmp_path: Path) -> None:
    csv = "source,start_s,end_s\nshow/episode one.mkv,9.3,13.3\n"
    root, docker = _small_root(
        tmp_path, clips={"solo.wav": take("target", 8.0, 1)}, anchors_csv=csv
    )
    write_sdh(root)
    code, out, err = extract(root, docker)
    assert code == 3, err
    assert "anchor clip-1" in out and "anchor time-1" in out and "anchor time-2" not in out


def test_sdh_spans_without_seed_anchors_still_exit_2(tmp_path: Path) -> None:
    root, docker = _small_root(tmp_path)
    write_sdh(root)
    code, _, err = extract(root, docker)
    assert code == 2 and "no anchor" in err
    assert docker.argvs == []


def _assert_clean(text: str, canaries: Canaries) -> None:
    lowered = text.lower()
    for index, value in enumerate(canaries.all()):
        if value.lower() in lowered:
            pytest.fail(f"canary #{index} appears in the checked text")
