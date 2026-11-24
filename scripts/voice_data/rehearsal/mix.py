"""The rehearsal's audio work, run as a job in the `score` image (Design 12).

Runs as `/opt/venv-score/bin/python -I -u -m scripts.voice_data.rehearsal.mix --data /data
--step S`, with `public/` mounted read-only. Two steps:

- `build` (`rehearsal build`): reads `rehearsal/build-plan.json`, checks every public file's
  sha256 against `sources.json` before it reads any, chooses the impostor with ECAPA, mixes
  the episodes in 5.1 at 48 kHz, encodes them with ffmpeg and writes `rehearsal/episodes/`,
  `truth.jsonl`, `build.json` and the solo anchor clip `reference/anchor.wav`. It decodes each
  5.1 episode's centre channel and records every fallback line's centre-cut SNR and C50 in
  `build.json` (`measure_fallback`).
- `effects` (`rehearsal score --run R`): reads `rehearsal/scores/<R>/effects-plan.json` and
  measures whether each in-show effect survived cleaning, and the identity-window floor of the
  stand-in target.

The mix, per episode:

- turns alternate between the target and the others; one non-target turn in three is the
  impostor's, up to `impostor_turns`; some turns overlap their neighbour;
- every target line carries the two in-show effects, a comb filter (it moves the speaker
  embedding) and an EQ, a high-pass and a presence peak (the embedders largely ignore it, the
  cleaning gate's spectrum check sees it);
- dialogue sits in the centre (FC), except the off-centre lines (front left and right only, each
  3 dB down) and the fallback lines: the dry line in FL and FR at full level, and in FC only a
  copy 15 dB down (`FALLBACK_COPY_DB`, the plan's figure) convolved with a reverb tail (RT60
  1.5 s, 6 dB below the direct sound) under a burst of MUSAN noise band-passed to
  `BURST_BAND_HZ`, 20 dB below the line. The centre cut fails SNR and C50 whether or not a
  separator removes the burst, and the `-ac 1` downmix, dominated by the dry left and right
  speech, passes. On the synthetic voices of `test_rehearsal.py`, at 15 dB down, the geometry
  test measured the FC cut at SNR -2.0 to 0.4 dB and C50 -3.6 to -3.4 dB, and the downmix at
  SNR about 23.3 dB and C50 23.2 to 25.1 dB: 3 to 5 dB above the default 20 dB C50 gate, the
  narrowest margin;
- a MUSAN music bed in FL, FR and the surrounds (low-passed into the LFE), ducked under dialogue,
  and MUSAN noise effects between lines.

Output names ids and counts only; public file names stay in the records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..states import SPEAKING_STATES
from . import (
    AAC_MKV,
    ANCHOR_REL,
    BUILD_REL,
    EAC3_MKV,
    EFFECTS_NAME,
    EFFECTS_PLAN_NAME,
    EPISODES_REL,
    OUTPUTS_NAME,
    PLAN_REL,
    REHEARSAL_REL,
    SCHEMA,
    SCORES_REL,
    STEPS,
    STEREO_MP4,
    TRUTH_REL,
    Sources,
    Speaker,
    TruthLine,
    eligible,
    episode_formats,
    episode_name,
    impostor_turns,
    parse_sources,
)

RATE = 48000
FL, FR, FC, LFE, SL, SR = range(6)
CHANNELS = 6
DIALOGUE_RMS = 0.05
MUSIC_DB = -24.0
MUSIC_DUCK_DB = -12.0
SURROUND_DB = -3.0
EFFECTS_DB = -18.0
EFFECTS_EVERY_S = 30.0
OFF_CENTRE_GAIN = math.sqrt(0.5)
FALLBACK_COPY_DB = -15.0
TAIL_DB = -6.0
RT60_S = 1.5
TAIL_S = 1.6
BURST_DB = -20.0
BURST_BAND_HZ = (4000.0, 7500.0)
SPEECH_BAND_HZ = (100.0, 3500.0)
LFE_HZ = 120.0
COMB = {"delay_ms": 3.0, "gain": 0.6}
EQ = {
    "highpass_hz": 150.0,
    "highpass_order": 2,
    "presence_hz": 3000.0,
    "presence_db": 6.0,
    "presence_q": 1.0,
}
OTHER_MAX_S = 12.0
OVERLAP_P = 0.15
GAP_S = (0.3, 1.0)
FALLBACK_GAP_BEFORE_S = 0.6
FALLBACK_GAP_AFTER_S = 1.8
OFF_CENTRE_GAP_S = 0.4
ANCHOR_MIN_S = 12.0
ANCHOR_MAX_S = 20.0
ANCHOR_GAP_S = 0.25
TARGET_SAMPLE = 20
CANDIDATE_SAMPLE = 10
OTHERS_PER_EPISODE = 3
CHUNK_S = 10.0
ENCODE_WORKERS = 3
PEAK_MAX = 0.95
VIDEO_SOURCE = "color=c=black:s=128x72:r=1"
WINDOW_S = 3.0
IDENTITY_S = 9.0
EFFECTS_MAX_CLIPS = 200


class StageError(RuntimeError):
    """The step cannot proceed; the message names ids and counts, never a file."""


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Return a file's sha256, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(part, path)


class Verifier:
    """Hands out public files, each checked against its `sources.json` sha256 before it is read."""

    def __init__(self, data: Path, expected: Mapping[str, str]) -> None:
        self._data = data
        self._expected = dict(expected)
        self._checked: set[str] = set()

    def path(self, rel: str) -> Path:
        """Return the file's path once its sha256 matches.

        Raises:
            StageError: The file is not in `sources.json`, is missing, or its sha256 differs.
        """
        if rel not in self._expected:
            raise StageError("a file outside sources.json was asked for")
        path = self._data / rel
        if rel not in self._checked:
            if not path.is_file():
                raise StageError("a public file sources.json names is missing; the build stops")
            if sha256_file(path) != self._expected[rel]:
                raise StageError(
                    "a public file's sha256 does not match sources.json; the build stops"
                )
            self._checked.add(rel)
        return path

    def check_all(self) -> None:
        """Check every file up front, so a mismatch stops the build before it writes anything.

        Raises:
            StageError: A file is missing or its sha256 differs.
        """
        for rel in sorted(self._expected):
            self.path(rel)

    @property
    def checked(self) -> int:
        """How many files were checked."""
        return len(self._checked)


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    """Return a file as float64 mono and its rate."""
    import soundfile

    audio, rate = soundfile.read(str(path), dtype="float64", always_2d=True)
    return audio.mean(axis=1), int(rate)


def resample(audio: np.ndarray, rate_in: int, rate_out: int = RATE) -> np.ndarray:
    """Return the clip at `rate_out` by polyphase resampling."""
    if rate_in == rate_out:
        return np.asarray(audio, dtype=np.float64)
    from scipy.signal import resample_poly

    common = math.gcd(rate_in, rate_out)
    return resample_poly(np.asarray(audio, dtype=np.float64), rate_out // common, rate_in // common)


def trim(audio: np.ndarray, rate: int, range_db: float = 40.0) -> np.ndarray:
    """Cut leading and trailing silence: 20 ms frames more than `range_db` below the loudest."""
    frame = max(1, int(0.02 * rate))
    count = len(audio) // frame
    if count == 0:
        return audio
    power = np.mean(audio[: count * frame].reshape(count, frame) ** 2, axis=1)
    if power.max() <= 0:
        return audio
    active = np.flatnonzero(power >= power.max() * 10.0 ** (-range_db / 10.0))
    pad = int(0.05 * rate)
    start = max(0, active[0] * frame - pad)
    end = min(len(audio), (active[-1] + 1) * frame + pad)
    return audio[start:end]


def rms(audio: np.ndarray) -> float:
    """Return the clip's RMS level."""
    return float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0


def normalise(audio: np.ndarray, level: float) -> np.ndarray:
    """Return the clip scaled to RMS `level`."""
    current = rms(audio)
    return audio * (level / current) if current > 0 else audio


def fade(audio: np.ndarray, rate: int, seconds: float = 0.05) -> np.ndarray:
    """Return the clip with linear fades in and out."""
    n = min(len(audio) // 2, int(seconds * rate))
    out = np.array(audio, dtype=np.float64, copy=True)
    if n > 0:
        ramp = np.linspace(0.0, 1.0, n)
        out[:n] *= ramp
        out[-n:] *= ramp[::-1]
    return out


def comb(audio: np.ndarray, rate: int) -> np.ndarray:
    """The first in-show effect: a feed-forward comb filter, at the input's RMS."""
    delay = int(round(COMB["delay_ms"] * rate / 1000.0))
    out = np.array(audio, dtype=np.float64, copy=True)
    out[delay:] += COMB["gain"] * audio[: len(audio) - delay]
    return normalise(out, rms(audio))


def eq(audio: np.ndarray, rate: int) -> np.ndarray:
    """The second in-show effect: a Butterworth high-pass and an RBJ presence peak."""
    from scipy.signal import butter, lfilter, sosfilt

    sos = butter(
        int(EQ["highpass_order"]), EQ["highpass_hz"], btype="highpass", fs=rate, output="sos"
    )
    out = sosfilt(sos, audio)
    gain = 10.0 ** (EQ["presence_db"] / 40.0)
    w0 = 2.0 * math.pi * EQ["presence_hz"] / rate
    alpha = math.sin(w0) / (2.0 * EQ["presence_q"])
    b = np.array([1.0 + alpha * gain, -2.0 * math.cos(w0), 1.0 - alpha * gain])
    a = np.array([1.0 + alpha / gain, -2.0 * math.cos(w0), 1.0 - alpha / gain])
    return lfilter(b / a[0], a / a[0], out)


def in_show(audio: np.ndarray, rate: int) -> np.ndarray:
    """Apply both in-show effects, as every target line in every episode carries them."""
    return eq(comb(audio, rate), rate)


def bandpass(audio: np.ndarray, rate: int, band: tuple[float, float]) -> np.ndarray:
    """Return the clip band-passed (4th-order Butterworth)."""
    from scipy.signal import butter, sosfilt

    return sosfilt(butter(4, band, btype="bandpass", fs=rate, output="sos"), audio)


def lowpass(audio: np.ndarray, rate: int, hz: float) -> np.ndarray:
    """Return the clip low-passed (4th-order Butterworth)."""
    from scipy.signal import butter, sosfilt

    return sosfilt(butter(4, hz, btype="lowpass", fs=rate, output="sos"), audio)


def reverb_tail(rng: np.random.Generator, rate: int = RATE) -> np.ndarray:
    """Return an exponentially decaying noise tail of RT60 `RT60_S`, `TAIL_DB` in energy."""
    n = int(TAIL_S * rate)
    t = np.arange(n) / rate
    tail = rng.standard_normal(n) * np.exp(-3.0 * math.log(10.0) * t / RT60_S)
    tail /= math.sqrt(float(np.sum(tail**2)))
    return tail * 10.0 ** (TAIL_DB / 20.0)


def fallback_copy(dry: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return a fallback line's centre copy: `FALLBACK_COPY_DB` down, convolved with the tail."""
    from scipy.signal import fftconvolve

    return fftconvolve(dry * 10.0 ** (FALLBACK_COPY_DB / 20.0), reverb_tail(rng))


# ---------------------------------------------------------------------------
# The fallback lines' centre cut, measured
# ---------------------------------------------------------------------------


def band_power(audio: np.ndarray, band: tuple[float, float], rate: int = RATE) -> float:
    """Return the clip's spectral power within `band`."""
    spectrum = np.abs(np.fft.rfft(audio)) ** 2
    freqs = np.fft.rfftfreq(len(audio), 1.0 / rate)
    return float(spectrum[(freqs >= band[0]) & (freqs < band[1])].sum())


def snr_db(window: np.ndarray) -> float:
    """Return the speech band's power against the burst band's, in dB."""
    return 10 * math.log10(band_power(window, SPEECH_BAND_HZ) / band_power(window, BURST_BAND_HZ))


def c50_db(dry: np.ndarray, wet: np.ndarray) -> float:
    """C50 of the response from the known `dry` line to `wet`, weighted by the line's spectrum.

    The response is deconvolved in the speech band. Its early part runs from 10 ms before the
    direct path (the codecs delay it by under 25 ms) to 50 ms after, its late part from there to
    the end of the reverb tail; each part's energy is that of `dry` through it, so a bin weighs
    what the line puts in it and the line's weak bins do not inflate the estimate's noise.
    """
    n = len(wet)
    x = np.fft.rfft(dry, n)
    y = np.fft.rfft(wet, n)
    freqs = np.fft.rfftfreq(n, 1.0 / RATE)
    band = (freqs >= SPEECH_BAND_HZ[0]) & (freqs < SPEECH_BAND_HZ[1])
    power = np.abs(x) ** 2
    eps = 1e-4 * power[band].mean()
    h = np.fft.irfft(np.where(band, y * np.conj(x) / (power + eps), 0.0), n)
    ms = RATE // 1000

    def through(lo_ms: int, hi_ms: int) -> float:
        lags = np.arange(lo_ms * ms, hi_ms * ms) % n
        part = np.zeros(n)
        part[lags] = h[lags]
        return float(np.sum(np.fft.irfft(x * np.fft.rfft(part), n) ** 2))

    return 10 * math.log10(through(-10, 50) / through(50, int(TAIL_S * 1000)))


def decode_fc(path: Path, codec: str) -> np.ndarray:
    """Return an encoded episode's centre channel at `RATE`, decoded as ingest decodes it.

    Raises:
        StageError: ffmpeg exited non-zero.
    """
    from ..stages.ingest import decode_argv

    proc = subprocess.run(decode_argv(str(path), codec, "FC"), capture_output=True, check=False)
    if proc.returncode != 0:
        raise StageError(f"decoding an episode's centre channel: ffmpeg exited {proc.returncode}")
    return np.frombuffer(proc.stdout, dtype="<f4").astype(np.float64)


def measure_fallback(fc: np.ndarray, turns: Sequence[Turn]) -> list[dict[str, float]]:
    """Return each fallback line's centre-cut SNR and C50 (`snr_db`, `c50_db`), by line number.

    Extract records no centre-cut values for a fallback line whose downmix passes (it writes the
    downmix record only), so the build measures them on the decoded episode. These are the
    build's own estimators, not extract's quality model: `rehearsal score` reports them beside
    the calibrated floors as a check that the floors still fail the fallback lines' centre cut.
    """
    found = []
    for number, turn in enumerate(turns, start=1):
        if turn.geometry != "fallback":
            continue
        a = int(turn.start_s * RATE)
        b = a + len(turn.audio)
        tail = b + int(TAIL_S * RATE)
        found.append(
            {
                "line": number,
                "snr_db": round(snr_db(fc[a:b]), 2),
                "c50_db": round(c50_db(turn.audio, fc[a:tail]), 2),
            }
        )
    return found


# ---------------------------------------------------------------------------
# The impostor
# ---------------------------------------------------------------------------


def _unit(vector: np.ndarray) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float64).reshape(-1)
    return v / np.linalg.norm(v)


def centroid(embeddings: Sequence[np.ndarray]) -> np.ndarray:
    """Return the normalised mean of the normalised embeddings."""
    return _unit(np.mean([_unit(e) for e in embeddings], axis=0))


def impostor_scores(
    target: Sequence[np.ndarray], candidates: Mapping[str, Sequence[np.ndarray]]
) -> dict[str, float]:
    """Return each candidate's median SIM to the target centroid, as a cluster's is scored."""
    centre = centroid(target)
    return {
        name: float(np.median([float(_unit(e) @ centre) for e in embs]))
        for name, embs in candidates.items()
    }


def impostor_candidates(sources: Sources, minutes: float) -> list[Speaker]:
    """Return the others who may be the impostor: the target's sex, and `eligible` by clips."""
    return [
        s for s in sources.others if s.sex == sources.target_sex and eligible(len(s.files), minutes)
    ]


def choose_impostor(scores: Mapping[str, float]) -> str:
    """Return the nearest candidate; ties go to the lower speaker id, so a rebuild agrees.

    Raises:
        StageError: There is no candidate.
    """
    if not scores:
        raise StageError("no other speaker of the target's sex has enough clips to be the impostor")
    return min(scores, key=lambda name: (-scores[name], name))


# ---------------------------------------------------------------------------
# Episode plan
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    """One line of an episode before it is placed."""

    speaker: str
    speaker_id: str
    clip: str
    audio: np.ndarray
    state: str | None = None
    style: str | None = None
    geometry: str | None = None
    start_s: float = 0.0
    overlap: bool = False

    @property
    def seconds(self) -> float:
        """The line's length."""
        return len(self.audio) / RATE

    @property
    def end_s(self) -> float:
        """Where the line ends."""
        return self.start_s + self.seconds


@dataclass
class Library:
    """Clip loading with a cache, and the target's unused clips per style."""

    verifier: Verifier
    pools: dict[str, list[str]]
    styles: dict[str, str]
    cache: dict[str, np.ndarray] = field(default_factory=dict)

    def raw(self, rel: str) -> np.ndarray:
        """Return a clip at 48 kHz mono, leading and trailing silence cut."""
        if rel not in self.cache:
            audio, rate = read_audio(self.verifier.path(rel))
            self.cache[rel] = trim(resample(audio, rate), RATE)
        return self.cache[rel]

    def target(self, rel: str) -> np.ndarray:
        """Return a target clip as every episode carries it: both effects, at dialogue level."""
        return normalise(in_show(self.raw(rel), RATE), DIALOGUE_RMS)

    def other(self, rel: str) -> np.ndarray:
        """Return another speaker's clip at dialogue level, at most `OTHER_MAX_S` long."""
        audio = self.raw(rel)[: int(OTHER_MAX_S * RATE)]
        return normalise(fade(audio, RATE), DIALOGUE_RMS)

    def take_target(self, state: str) -> tuple[str, str] | None:
        """Take the next unused target clip of `state`, else of any state; `None` when empty."""
        order = [s for s, st in self.styles.items() if st == state]
        order += [s for s in self.styles if s not in order]
        for style in order:
            if self.pools[style]:
                return style, self.pools[style].pop(0)
        return None


def plan_turns(
    rng: np.random.Generator,
    minutes: float,
    library: Library,
    impostor: Speaker,
    others: Sequence[Speaker],
    state_offset: int,
) -> list[Turn]:
    """Return an episode's turns in order: target and non-target alternate.

    Every third non-target turn is the impostor's, up to `impostor_turns(minutes)`, with no clip
    repeated; the other speakers take the rest in rotation.

    Raises:
        StageError: The target's clips run out.
    """
    total = minutes * 60.0
    cap = impostor_turns(minutes)
    imp_clips = [impostor.files[i][0] for i in rng.permutation(len(impostor.files))][:cap]
    other_clips = {
        s.speaker: [s.files[i][0] for i in rng.permutation(len(s.files))] for s in others
    }
    turns: list[Turn] = []
    elapsed = 1.0
    k_target = non_target = used_imp = 0
    while elapsed < total - 1.0:
        if not turns or turns[-1].speaker != "target":
            state = SPEAKING_STATES[(k_target + state_offset) % len(SPEAKING_STATES)]
            taken = library.take_target(state)
            if taken is None:
                raise StageError("the target's clips ran out; build fewer or shorter episodes")
            style, rel = taken
            turn = Turn(
                "target", "", rel, library.target(rel), library.styles[style], style, "centre"
            )
            k_target += 1
        elif non_target % 3 == 1 and used_imp < len(imp_clips):
            rel = imp_clips[used_imp]
            turn = Turn("impostor", impostor.speaker, rel, library.other(rel))
            used_imp += 1
            non_target += 1
        else:
            speaker = others[non_target % len(others)]
            pool = other_clips[speaker.speaker]
            rel = pool[(non_target // len(others)) % len(pool)]
            turn = Turn("other", speaker.speaker, rel, library.other(rel))
            non_target += 1
        turns.append(turn)
        elapsed += turn.seconds + sum(GAP_S) / 2.0
    return turns


def assign_geometry(turns: Sequence[Turn]) -> None:
    """Mark the fallback and off-centre target lines, spread evenly over the episode.

    At least three fallback lines (one per 15 target lines) and two off-centre ones (one per 20),
    never the first or last target line.

    Raises:
        StageError: The episode holds too few target lines for them.
    """
    targets = [i for i, t in enumerate(turns) if t.speaker == "target"]
    inner = targets[1:-1]
    n_fb = max(3, len(targets) // 15)
    n_oc = max(2, len(targets) // 20)
    if len(inner) < n_fb + n_oc + 1:
        raise StageError("an episode is too short for its fallback and off-centre lines")
    count = n_fb + n_oc
    picks = [inner[int((i + 0.5) * len(inner) / count)] for i in range(count)]
    kinds = ["fallback"] * n_fb + ["off-centre"] * n_oc
    order = [kinds[i] for i in sorted(range(count), key=lambda i: (i % 2, i))]
    for index, kind in zip(picks, order, strict=True):
        turns[index].geometry = kind


def place(rng: np.random.Generator, turns: Sequence[Turn]) -> float:
    """Set each turn's start, with gaps and overlaps; return the episode's length in seconds.

    Plain turns overlap their predecessor with probability `OVERLAP_P`. Fallback and off-centre
    lines never overlap, a fallback line leaves its reverb tail clear of the next line, and the
    overlap flags are set on both turns of each overlapping pair.
    """
    t = 1.0
    previous: Turn | None = None
    for turn in turns:
        gap = float(rng.uniform(*GAP_S))
        plain = turn.geometry in (None, "centre")
        prev_plain = previous is None or previous.geometry in (None, "centre")
        if previous is None:
            turn.start_s = t
        elif plain and prev_plain and rng.random() < OVERLAP_P:
            amount = min(float(rng.uniform(0.4, 1.2)), 0.4 * min(previous.seconds, turn.seconds))
            turn.start_s = previous.end_s - amount
            turn.overlap = previous.overlap = True
        else:
            if turn.geometry == "fallback":
                gap = max(gap, FALLBACK_GAP_BEFORE_S)
            if not plain or not prev_plain:
                gap = max(gap, OFF_CENTRE_GAP_S)
            if previous.geometry == "fallback":
                gap = max(gap, FALLBACK_GAP_AFTER_S)
            turn.start_s = previous.end_s + gap
        turn.start_s = round(turn.start_s, 3)
        previous = turn
    return round(max(t.end_s for t in turns) + 1.0, 3)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Event:
    """A signal placed in the mix: start sample, samples and one gain per channel."""

    start: int
    signal: np.ndarray
    gains: tuple[float, ...]


def _gains(**channels: float) -> tuple[float, ...]:
    gains = [0.0] * CHANNELS
    for name, value in channels.items():
        gains[{"FL": FL, "FR": FR, "FC": FC, "LFE": LFE, "SL": SL, "SR": SR}[name]] = value
    return tuple(gains)


def _looped(signal: np.ndarray, length: int) -> np.ndarray:
    reps = int(math.ceil(length / max(1, len(signal))))
    return np.tile(signal, reps)[:length]


def dialogue_events(
    turns: Sequence[Turn], rng: np.random.Generator, library: Library, noise: Sequence[str]
) -> tuple[list[Event], int]:
    """Return the dialogue events (with each fallback line's copy and burst) and burst count."""
    events = []
    bursts = 0
    for turn in turns:
        start = int(round(turn.start_s * RATE))
        if turn.geometry == "off-centre":
            events.append(Event(start, turn.audio, _gains(FL=OFF_CENTRE_GAIN, FR=OFF_CENTRE_GAIN)))
        elif turn.geometry == "fallback":
            events.append(Event(start, turn.audio, _gains(FL=1.0, FR=1.0)))
            events.append(Event(start, fallback_copy(turn.audio, rng), _gains(FC=1.0)))
            pad = int(0.3 * RATE)
            source = library.raw(noise[int(rng.integers(len(noise)))])
            burst = bandpass(_looped(source, len(turn.audio) + 2 * pad), RATE, BURST_BAND_HZ)
            burst = fade(normalise(burst, rms(turn.audio) * 10.0 ** (BURST_DB / 20.0)), RATE)
            events.append(Event(max(0, start - pad), burst, _gains(FC=1.0)))
            bursts += 1
        else:
            events.append(Event(start, turn.audio, _gains(FC=1.0)))
    return events, bursts


def effects_events(
    turns: Sequence[Turn],
    seconds: float,
    rng: np.random.Generator,
    library: Library,
    noise: Sequence[str],
) -> list[Event]:
    """Return MUSAN noise effects between lines, kept 2 s clear of fallback and off-centre lines."""
    guarded = [
        (t.start_s - 2.0, t.end_s + 2.0) for t in turns if t.geometry != "centre" and t.geometry
    ]
    events = []
    for _ in range(int(seconds // EFFECTS_EVERY_S)):
        length = float(rng.uniform(1.0, 3.0))
        at = float(rng.uniform(1.0, max(1.5, seconds - length - 1.0)))
        if any(a < at + length and at < b for a, b in guarded):
            continue
        source = library.raw(noise[int(rng.integers(len(noise)))])
        clip = fade(normalise(_looped(source, int(length * RATE)), DIALOGUE_RMS), RATE)
        clip = clip * 10.0 ** (EFFECTS_DB / 20.0)
        pair = _gains(FL=OFF_CENTRE_GAIN, FR=OFF_CENTRE_GAIN)
        if rng.random() < 0.5:
            pair = _gains(SL=OFF_CENTRE_GAIN, SR=OFF_CENTRE_GAIN)
        events.append(Event(int(at * RATE), clip, pair))
    return events


def episode_samples(seconds: float) -> int:
    """Return the episode's sample count: the one rule `music_bed` and `render` both size by.

    `render` indexes the bed with its own chunk bounds, so a bed sized by a different rounding
    rule is a sample short whenever the fraction rounds up, and the last chunk cannot broadcast.
    """
    return int(round(seconds * RATE))


def music_bed(
    seconds: float, rng: np.random.Generator, library: Library, music: Sequence[str]
) -> np.ndarray:
    """Return a mono music bed for the episode at `MUSIC_DB` below dialogue, before ducking."""
    length = episode_samples(seconds)
    picks = [music[int(i)] for i in rng.permutation(len(music))[:4]]
    bed = np.concatenate([library.raw(rel) for rel in picks])
    return (normalise(_looped(bed, length), DIALOGUE_RMS) * 10.0 ** (MUSIC_DB / 20.0)).astype(
        np.float32
    )


def duck_envelope(turns: Sequence[Turn], seconds: float) -> tuple[np.ndarray, np.ndarray]:
    """Return a 100 Hz gain envelope that ducks the music under every line, with 0.3 s ramps.

    A fallback line keeps the music ducked through its reverb tail, which is part of the line's
    sound: music returning under the tail would count against the downmix's C50.
    """
    times = np.arange(0.0, seconds + 0.01, 0.01)
    duck = 10.0 ** (MUSIC_DUCK_DB / 20.0)
    gain = np.ones_like(times)
    for t in turns:
        end = t.end_s + (TAIL_S if t.geometry == "fallback" else 0.0)
        inside = np.clip(np.minimum(times - (t.start_s - 0.3), (end + 0.3) - times) / 0.3, 0.0, 1.0)
        gain = np.minimum(gain, 1.0 - (1.0 - duck) * inside)
    return times, gain


def render(
    seconds: float,
    events: Sequence[Event],
    bed: np.ndarray,
    envelope: tuple[np.ndarray, np.ndarray],
    emit: Callable[[np.ndarray], None],
    master: float = 1.0,
) -> float:
    """Render the 5.1 mix chunk by chunk into `emit`, scaled by `master`; return its peak."""
    total = episode_samples(seconds)
    chunk = int(CHUNK_S * RATE)
    surround = 10.0 ** (SURROUND_DB / 20.0)
    lfe = lowpass(bed.astype(np.float64), RATE, LFE_HZ).astype(np.float32)
    peak = 0.0
    for c0 in range(0, total, chunk):
        c1 = min(total, c0 + chunk)
        out = np.zeros((CHANNELS, c1 - c0), dtype=np.float64)
        duck = np.interp(np.arange(c0, c1) / RATE, *envelope)
        music = bed[c0:c1] * duck
        out[FL] += music
        out[FR] += music
        out[SL] += surround * music
        out[SR] += surround * music
        out[LFE] += lfe[c0:c1] * duck
        for event in events:
            a, b = max(c0, event.start), min(c1, event.start + len(event.signal))
            if a >= b:
                continue
            part = event.signal[a - event.start : b - event.start]
            for channel, gain in enumerate(event.gains):
                if gain:
                    out[channel, a - c0 : b - c0] += gain * part
        out *= master
        peak = max(peak, float(np.max(np.abs(out))))
        emit(out)
    return peak


def stereo_fold(block: np.ndarray) -> np.ndarray:
    """Fold a 5.1 block to stereo: each front, plus the centre and its surround 3 dB down."""
    c = OFF_CENTRE_GAIN
    left = block[FL] + c * block[FC] + c * block[SL]
    right = block[FR] + c * block[FC] + c * block[SR]
    return np.stack([left, right])


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def video_codec() -> str:
    """Return `hevc` when ffmpeg lists libx265, else `mpeg4`."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, check=False
    )
    return "hevc" if " libx265 " in proc.stdout else "mpeg4"


def encode_argv(fmt: str, video: str, seconds: float, output: Path) -> list[str]:
    """Return the ffmpeg argv that encodes raw float32 audio on stdin with a black video."""
    layout, channels, audio = {
        EAC3_MKV: ("5.1(side)", 6, ["-c:a", "eac3", "-b:a", "640k"]),
        AAC_MKV: ("5.1", 6, ["-c:a", "aac", "-b:a", "384k"]),
        STEREO_MP4: ("stereo", 2, ["-c:a", "aac", "-b:a", "192k"]),
    }[fmt]
    if video == "hevc":
        vargs = ["-c:v", "libx265", "-preset", "ultrafast", "-x265-params", "log-level=error"]
        if fmt == STEREO_MP4:
            vargs += ["-tag:v", "hvc1"]
    else:
        vargs = ["-c:v", "mpeg4", "-q:v", "10"]
    container = "mp4" if fmt == STEREO_MP4 else "matroska"
    return [
        "ffmpeg", "-v", "error", "-y",
        "-f", "f32le", "-ar", str(RATE), "-ac", str(channels), "-channel_layout", layout,
        "-i", "pipe:0",
        "-f", "lavfi", "-i", f"{VIDEO_SOURCE}:d={seconds:.3f}",
        "-map", "1:v", "-map", "0:a", *vargs, *audio,
        "-metadata:s:a:0", "language=eng", "-f", container, str(output),
    ]  # fmt: skip


class Encoder:
    """An ffmpeg process fed raw interleaved float32 blocks on its stdin."""

    def __init__(self, argv: Sequence[str], stereo: bool) -> None:
        self._log = tempfile.TemporaryFile()
        self._proc = subprocess.Popen(
            list(argv), stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._log
        )
        self._stereo = stereo

    def write(self, block: np.ndarray) -> None:
        """Write one (channels, samples) block."""
        data = stereo_fold(block) if self._stereo else block
        assert self._proc.stdin is not None
        self._proc.stdin.write(np.ascontiguousarray(data.T, dtype="<f4").tobytes())

    def close(self, what: str) -> None:
        """Finish the encode.

        Raises:
            StageError: ffmpeg exited non-zero; its stderr goes to the job log.
        """
        assert self._proc.stdin is not None
        self._proc.stdin.close()
        code = self._proc.wait()
        if code != 0:
            self._log.seek(0)
            print(self._log.read().decode("utf-8", "replace")[-2000:], file=sys.stderr)
            raise StageError(f"{what}: ffmpeg exited {code}")
        self._log.close()


# ---------------------------------------------------------------------------
# The build step
# ---------------------------------------------------------------------------


def _read_plan(path: Path) -> dict:
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StageError("the plan is missing or not JSON") from exc
    if not isinstance(plan, dict) or plan.get("schema") != SCHEMA:
        raise StageError("the plan is not a schema 1 plan")
    return plan


def load_ecapa() -> object:
    """Return the ECAPA embedder on the CPU; tests replace this function."""
    from ..bindings import embedders

    return embedders.load_ecapa("cpu")


def load_embedders() -> list:
    """Return both embedders on the CPU, WavLM-SV then ECAPA; tests replace this function."""
    from ..bindings import embedders

    return list(embedders.load("cpu"))


def _anchor(library: Library, sources: Sources) -> tuple[list[str], np.ndarray]:
    """Reserve the solo anchor: the last neutral clips in path order, 12-20 s with both effects."""
    neutral = next(s for s, st in sources.states.items() if st == "neutral")
    chosen: list[str] = []
    parts: list[np.ndarray] = []
    seconds = 0.0
    gap = np.zeros(int(ANCHOR_GAP_S * RATE))
    for rel in reversed(library.pools[neutral]):
        clip = library.target(rel)
        if seconds + len(clip) / RATE > ANCHOR_MAX_S:
            continue
        chosen.append(rel)
        parts += [clip, gap]
        seconds += len(clip) / RATE + ANCHOR_GAP_S
        if seconds >= ANCHOR_MIN_S:
            break
    if seconds < ANCHOR_MIN_S:
        raise StageError("the neutral clips cannot make a 12-20 s anchor")
    library.pools[neutral] = [rel for rel in library.pools[neutral] if rel not in chosen]
    return chosen, np.concatenate(parts[:-1])


def _choose_impostor(
    library: Library, sources: Sources, minutes: float
) -> tuple[Speaker, dict[str, float]]:
    ecapa = load_ecapa()
    sample: list[str] = []
    for style in library.styles:
        sample += library.pools[style][: TARGET_SAMPLE // len(library.styles)]
    target = [ecapa.embed(library.target(rel), RATE) for rel in sample]  # type: ignore[attr-defined]
    candidates = {
        s.speaker: [
            ecapa.embed(library.raw(rel), RATE)  # type: ignore[attr-defined]
            for rel, _ in s.files[:CANDIDATE_SAMPLE]
        ]
        for s in impostor_candidates(sources, minutes)
    }
    scores = impostor_scores(target, candidates)
    name = choose_impostor(scores)
    return next(s for s in sources.others if s.speaker == name), scores


def _others(rng: np.random.Generator, sources: Sources, impostor: Speaker) -> list[Speaker]:
    pool = [s for s in sources.others if s.speaker != impostor.speaker]
    same = [s for s in pool if s.sex == sources.target_sex]
    other = [s for s in pool if s.sex != sources.target_sex]
    picks: list[Speaker] = []
    if same:
        picks.append(same[int(rng.integers(len(same)))])
    rest = other or same
    order = rng.permutation(len(rest))
    picks += [rest[int(i)] for i in order if rest[int(i)] not in picks]
    return picks[:OTHERS_PER_EPISODE]


def build(data: Path) -> list[str]:
    """Run the build step; return the root-relative files written."""
    plan = _read_plan(data / PLAN_REL)
    sources = parse_sources(plan["sources"])
    episodes, minutes, seed = int(plan["episodes"]), float(plan["minutes"]), int(plan["seed"])
    verifier = Verifier(data, sources.all_files())
    verifier.check_all()
    rng = np.random.default_rng(seed)
    pools = {
        style: [files[i][0] for i in rng.permutation(len(files))]
        for style, files in sources.target.items()
    }
    library = Library(verifier, pools, dict(sources.states))
    # The anchor takes its clips from the sorted list's end, before episodes take any.
    library.pools = {style: sorted(p) for style, p in pools.items()}
    anchor_clips, anchor = _anchor(library, sources)
    library.pools = {
        style: [rel for rel in pools[style] if rel not in anchor_clips] for style in pools
    }
    impostor, scores = _choose_impostor(library, sources, minutes)
    print(f"rehearsal build: impostor chosen from {len(scores)} candidates")
    video = video_codec()
    noise = [rel for rel, _ in sources.noise]
    music = [rel for rel, _ in sources.music]
    written: list[str] = []
    truth: list[TruthLine] = []
    records = []
    # Episodes are planned in order (the library hands out clips), then encoded in parallel.
    planned: list[tuple] = []
    pool = ThreadPoolExecutor(max_workers=ENCODE_WORKERS)
    for index, fmt in enumerate(episode_formats(episodes)):
        if index >= ENCODE_WORKERS:  # at most ENCODE_WORKERS mixes held at once
            planned[index - ENCODE_WORKERS][-1].result()
        ep_rng = np.random.default_rng([seed, index])
        others = _others(ep_rng, sources, impostor)
        turns = plan_turns(ep_rng, minutes, library, impostor, others, index)
        assign_geometry(turns)
        seconds = place(ep_rng, turns)
        events, bursts = dialogue_events(turns, ep_rng, library, noise)
        events += effects_events(turns, seconds, ep_rng, library, noise)
        bed = music_bed(seconds, ep_rng, library, music)
        envelope = duck_envelope(turns, seconds)
        peak = render(seconds, events, bed, envelope, lambda block: None)
        master = min(1.0, PEAK_MAX / peak) if peak > 0 else 1.0
        rel = f"{EPISODES_REL}/{episode_name(index, fmt)}"
        job = (fmt, video, seconds, events, bed, envelope, master, data / rel, index + 1)
        future = pool.submit(_encode, *job)
        planned.append(
            (index, fmt, rel, turns, others, seconds, master, bursts, len(events), future)
        )
    for index, fmt, rel, turns, others, seconds, master, bursts, n_events, future in planned:
        digest = future.result()
        name = f"ep{index + 1:02d}"
        for number, turn in enumerate(turns, start=1):
            truth.append(
                TruthLine(
                    episode=name,
                    episode_sha256=digest,
                    line=number,
                    speaker=turn.speaker,
                    speaker_id=turn.speaker_id or sources.target_speaker,
                    start_s=round(turn.start_s, 3),
                    end_s=round(turn.end_s, 3),
                    state=turn.state,
                    style=turn.style,
                    geometry=turn.geometry if turn.speaker == "target" else None,
                    clip=turn.clip,
                    overlap=turn.overlap,
                )
            )
        target_turns = [t for t in turns if t.speaker == "target"]
        codec = "eac3" if fmt == EAC3_MKV else "aac"
        fallback_fc = []
        if fmt != STEREO_MP4:
            fallback_fc = measure_fallback(decode_fc(data / rel, codec), turns)
        records.append(
            {
                "episode": name,
                "file": rel,
                "sha256": digest,
                "format": fmt,
                "video_codec": video,
                "audio_codec": codec,
                "layout": "stereo" if fmt == STEREO_MP4 else "5.1",
                "duration_s": seconds,
                "master_gain": round(master, 6),
                "lines": len(turns),
                "target_lines": len(target_turns),
                "impostor_lines": sum(t.speaker == "impostor" for t in turns),
                "other_speakers": [s.speaker for s in others],
                "fallback_lines": sum(t.geometry == "fallback" for t in target_turns),
                "fallback_fc": fallback_fc,
                "off_centre_lines": sum(t.geometry == "off-centre" for t in target_turns),
                "overlapping_lines": sum(t.overlap for t in turns),
                "bursts": bursts,
                "effects_events": n_events - len(turns) - 2 * bursts,
                "states": {s: sum(t.state == s for t in target_turns) for s in SPEAKING_STATES},
            }
        )
        written.append(rel)
        print(f"rehearsal build: episode {index + 1} of {episodes} written ({seconds:.0f} s)")
    pool.shutdown()
    _write_anchor(data / ANCHOR_REL, anchor)
    truth_path = data / TRUTH_REL
    part = truth_path.with_name(truth_path.name + ".part")
    part.write_text("".join(json.dumps(t.to_json()) + "\n" for t in truth), encoding="utf-8")
    os.replace(part, truth_path)
    _write_json(
        data / BUILD_REL,
        {
            "schema": SCHEMA,
            "seed": seed,
            "minutes": minutes,
            "target": {"speaker": sources.target_speaker, "sex": sources.target_sex},
            "impostor": {
                "speaker": impostor.speaker,
                "ecapa_sim": round(scores[impostor.speaker], 6),
                "candidates": {k: round(v, 6) for k, v in sorted(scores.items())},
                "rule": "nearest by ECAPA median SIM to the processed target's centroid, among "
                "the target's sex with clips for impostor_turns(minutes) turns",
                "turns_max": impostor_turns(minutes),
            },
            "effects": {"comb": COMB, "eq": EQ},
            "fallback": {
                "copy_db": FALLBACK_COPY_DB,
                "tail_db": TAIL_DB,
                "rt60_s": RT60_S,
                "burst_db": BURST_DB,
                "burst_band_hz": list(BURST_BAND_HZ),
                "speech_band_hz": list(SPEECH_BAND_HZ),
            },
            "anchor": {
                "file": ANCHOR_REL,
                "clips": anchor_clips,
                "duration_s": round(len(anchor) / RATE, 3),
            },
            "files_checked": verifier.checked,
            "episodes": records,
        },
    )
    return [*written, ANCHOR_REL, TRUTH_REL, BUILD_REL]


def _encode(
    fmt: str,
    video: str,
    seconds: float,
    events: Sequence[Event],
    bed: np.ndarray,
    envelope: tuple[np.ndarray, np.ndarray],
    master: float,
    out: Path,
    number: int,
) -> str:
    """Render and encode one planned episode to `out`; return its sha256."""
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    encoder = Encoder(encode_argv(fmt, video, seconds, part), fmt == STEREO_MP4)
    render(seconds, events, bed, envelope, encoder.write, master)
    encoder.close(f"episode {number}")
    os.replace(part, out)
    return sha256_file(out)


def _write_anchor(path: Path, audio: np.ndarray) -> None:
    import soundfile

    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    soundfile.write(str(part), audio.astype(np.float32), RATE, format="WAV", subtype="PCM_24")
    os.replace(part, path)


# ---------------------------------------------------------------------------
# The effects step
# ---------------------------------------------------------------------------


def _embed_all(embedders: Sequence[object], clips: Sequence[np.ndarray], rate: int) -> dict:
    return {e.name: [np.asarray(e.embed(c, rate)) for c in clips] for e in embedders}  # type: ignore[attr-defined]


def _medians(segments: Mapping[str, list], centre: Mapping[str, np.ndarray]) -> dict[str, float]:
    return {
        name: float(np.median([float(_unit(v) @ centre[name]) for v in vectors]))
        for name, vectors in segments.items()
    }


def identity_windows(
    embedders: Sequence[object], per_state: Mapping[str, np.ndarray], rate: int
) -> dict[str, object]:
    """Score the stand-in target's identity windows as `target build` does in identity mode.

    One processed clip per state (`IDENTITY_S` long) is cut into 3 s windows; each window is
    scored against the centroid of the other clips' windows. Returns the count, median and p10
    per embedder.
    """
    size = int(WINDOW_S * rate)
    windows = [
        (state, clip[i : i + size])
        for state, clip in per_state.items()
        for i in range(0, len(clip) - size + 1, size)
    ]
    if len({s for s, _ in windows}) < 2:
        return {"n": len(windows), "median": None, "p10": None}
    embs = _embed_all(embedders, [w for _, w in windows], rate)
    result: dict[str, dict[str, float]] = {"median": {}, "p10": {}}
    for name, vectors in embs.items():
        scores = []
        for i, (state, _) in enumerate(windows):
            others = [v for j, v in enumerate(vectors) if windows[j][0] != state]
            scores.append(float(_unit(vectors[i]) @ centroid(others)))
        result["median"][name] = round(float(np.median(scores)), 6)
        result["p10"][name] = round(float(np.percentile(scores, 10)), 6)
    return {"n": len(windows), **result}


def effects(data: Path, run: str) -> list[str]:
    """Run the effects step for score run `run`; return the root-relative files written."""
    from ..stages.score import spectrum_distance_db

    folder = f"{SCORES_REL}/{run}"
    plan = _read_plan(data / folder / EFFECTS_PLAN_NAME)
    verifier = Verifier(data, plan["sources"])
    rate = 24000
    embedders = load_embedders()

    def clip(rel: str, processed: bool) -> np.ndarray:
        audio, native = read_audio(verifier.path(rel))
        audio = trim(audio, native)
        if processed:
            audio = in_show(audio, native)
        return resample(audio, native, rate)

    result: dict[str, object] = {"schema": SCHEMA}
    clips = list(plan["clips"])[:EFFECTS_MAX_CLIPS]
    segments = list(plan["segments"])[:EFFECTS_MAX_CLIPS]
    if segments and clips:
        cleaned = []
        for rel in segments:
            audio, native = read_audio(data / rel)
            cleaned.append(resample(audio, native, rate))
        processed = [clip(rel, True) for rel in clips]
        raw = [clip(rel, False) for rel in clips]
        seg_embs = _embed_all(embedders, cleaned, rate)
        cent_p = {n: centroid(v) for n, v in _embed_all(embedders, processed, rate).items()}
        cent_u = {n: centroid(v) for n, v in _embed_all(embedders, raw, rate).items()}
        sim_p, sim_u = _medians(seg_embs, cent_p), _medians(seg_embs, cent_u)
        result["comb"] = {
            "segments": len(cleaned),
            "clips": len(clips),
            "sim_processed": {k: round(v, 6) for k, v in sim_p.items()},
            "sim_unprocessed": {k: round(v, 6) for k, v in sim_u.items()},
            "survived": all(sim_p[k] > sim_u[k] for k in sim_p),
        }
        joined = {
            name: normalise(np.concatenate(parts), DIALOGUE_RMS)
            for name, parts in (("cleaned", cleaned), ("processed", processed), ("raw", raw))
        }
        d_p = spectrum_distance_db(joined["cleaned"], joined["processed"], rate)
        d_u = spectrum_distance_db(joined["cleaned"], joined["raw"], rate)
        own = spectrum_distance_db(joined["processed"], joined["raw"], rate)
        result["eq"] = {
            "distance_processed_db": round(d_p, 4),
            "distance_unprocessed_db": round(d_u, 4),
            "eq_own_distance_db": round(own, 4),
            "survived": bool(d_p < d_u),
        }
    else:
        result["comb"] = result["eq"] = None
    per_state = {}
    for state, rels in plan["identity"].items():
        parts = []
        for rel in rels:
            parts.append(clip(rel, True))
            if sum(len(p) for p in parts) >= IDENTITY_S * rate:
                break
        joined_state = np.concatenate(parts) if parts else np.zeros(0)
        per_state[state] = joined_state[: int(IDENTITY_S * rate)]
    result["identity_windows"] = identity_windows(embedders, per_state, rate)
    rel = f"{folder}/{EFFECTS_NAME}"
    _write_json(data / rel, result)
    print(f"rehearsal effects: {len(segments)} cleaned segments, {len(clips)} target clips")
    return [rel]


def main(argv: Sequence[str] | None = None) -> int:
    """Run one step; write `outputs.json` for the executor to verify."""
    parser = argparse.ArgumentParser(prog="scripts.voice_data.rehearsal.mix")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--step", required=True, choices=STEPS)
    parser.add_argument("--run", help="the score run of the effects step")
    args = parser.parse_args(argv)
    try:
        if args.step == "build":
            written, folder = build(args.data), REHEARSAL_REL
        else:
            if not args.run:
                raise StageError("the effects step needs --run")
            written, folder = effects(args.data, args.run), f"{SCORES_REL}/{args.run}"
    except StageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    files = [{"rel": rel, "sha256": sha256_file(args.data / rel)} for rel in written]
    _write_json(args.data / folder / OUTPUTS_NAME, {"schema": SCHEMA, "files": files})
    return 0


if __name__ == "__main__":
    sys.exit(main())
