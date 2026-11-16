"""Deterministic fakes of the model protocols in `scripts.voice_data.bindings.protocols`.

Stage tests pass these where the images pass T11's bindings, so no test needs weights, a GPU or
the network. Every fake is a pure function of its constructor arguments and its input audio:
the same call gives the same result in every run and on every machine.

- `FakeEmbedder`: a projection of the log spectrum. The long-term power spectrum is summed into
  log-spaced bands from 80 Hz to 8 kHz, taken in dB, floored 40 dB below its peak band, has its
  mean removed (so gain does not move it) and is projected by a fixed Gaussian matrix seeded
  from the embedder's name, then L2-normalised. With conftest's `synth_voice`, one voice at
  two lengths or levels scores above 0.99, the same formants 5 Hz apart in F0 about 0.98, and
  formants (800, 1200, 2800) against (400, 2000, 3200) Hz at one F0 about 0.65.
- `FakeSeparator`: the identity; it returns a copy of its input.
- `FakeDiarizer`: returns the turns and overlap spans it was built with.
- `FakeVad`: frames within `range_db` of the loudest 20 ms frame are speech; runs of speech
  closer than `min_gap_s` are merged.
- `FakeQualityEstimator`: a constant SNR and C50 track, one frame per `hop_s`.
- `FakeEnhancer`: applies the function it was built with, or returns a copy.
- `FakeAsr`: returns the text it was built with, its words spread evenly over the audio.
- `FakeArousal`: the clip's RMS level mapped linearly from -40..-10 dBFS onto 0..1.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable, Sequence

import numpy as np

from scripts.voice_data.bindings.protocols import QualityTrack, Span, Transcript, Turn, Word

EMBED_BANDS = 40
EMBED_LOW_HZ = 80.0
EMBED_HIGH_HZ = 8000.0
EMBED_FLOOR_DB = 40.0


def _power_spectrum(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    # About 6 Hz bins, so even the lowest band holds several of them.
    n_fft = int(2 ** np.ceil(np.log2(sample_rate / 6.0)))
    signal = np.asarray(audio, dtype=np.float64)
    if len(signal) < n_fft:
        signal = np.pad(signal, (0, n_fft - len(signal)))
    hop = n_fft // 2
    starts = range(0, len(signal) - n_fft + 1, hop)
    window = np.hanning(n_fft)
    power = np.zeros(n_fft // 2 + 1)
    for start in starts:
        power += np.abs(np.fft.rfft(signal[start : start + n_fft] * window)) ** 2
    return np.fft.rfftfreq(n_fft, 1.0 / sample_rate), power


class FakeEmbedder:
    """A speaker embedder whose embedding is a fixed projection of the clip's log spectrum."""

    def __init__(self, name: str = "wavlm-sv", dim: int = 32, seed: int | None = None) -> None:
        self.name = name
        rng_seed = zlib.crc32(name.encode("utf-8")) if seed is None else seed
        rng = np.random.default_rng(rng_seed)
        self._projection = rng.standard_normal((dim, EMBED_BANDS)) / np.sqrt(EMBED_BANDS)
        self._edges = np.geomspace(EMBED_LOW_HZ, EMBED_HIGH_HZ, EMBED_BANDS + 1)

    def log_spectrum(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return the mean-removed band levels in dB that `embed` projects."""
        freqs, power = _power_spectrum(audio, sample_rate)
        bands = np.empty(EMBED_BANDS)
        for i in range(EMBED_BANDS):
            mask = (freqs >= self._edges[i]) & (freqs < self._edges[i + 1])
            bands[i] = power[mask].sum() if mask.any() else 0.0
        levels = 10.0 * np.log10(bands + 1e-20)
        levels = np.maximum(levels, levels.max() - EMBED_FLOOR_DB)
        return levels - levels.mean()

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        vector = self._projection @ self.log_spectrum(audio, sample_rate)
        norm = float(np.linalg.norm(vector))
        return (vector / norm if norm > 0 else vector).astype(np.float32)


class FakeSeparator:
    """A separator that returns its input unchanged, like the `none` candidate."""

    def __init__(self, name: str = "none") -> None:
        self.name = name

    def dialogue(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        return np.array(audio, dtype=np.float32, copy=True)


class FakeDiarizer:
    """A diarizer that returns the turns and overlaps a test scripted."""

    def __init__(self, turns: Sequence[Turn] = (), overlaps: Sequence[Span] = ()) -> None:
        self._turns = list(turns)
        self._overlaps = list(overlaps)

    def turns(self, audio: np.ndarray, sample_rate: int) -> list[Turn]:
        return list(self._turns)

    def overlaps(self, audio: np.ndarray, sample_rate: int) -> list[Span]:
        return list(self._overlaps)


class FakeVad:
    """An energy VAD: 20 ms frames within `range_db` of the loudest frame are speech."""

    def __init__(self, range_db: float = 30.0, min_gap_s: float = 0.2) -> None:
        self.range_db = range_db
        self.min_gap_s = min_gap_s

    def speech(self, audio: np.ndarray, sample_rate: int) -> list[Span]:
        frame = max(1, int(round(0.02 * sample_rate)))
        count = len(audio) // frame
        if count == 0:
            return []
        frames = np.asarray(audio[: count * frame], dtype=np.float64).reshape(count, frame)
        power = np.mean(frames**2, axis=1)
        if power.max() <= 0.0:
            return []
        active = power >= power.max() * 10.0 ** (-self.range_db / 10.0)
        spans: list[Span] = []
        start: int | None = None
        for index, flag in enumerate([*active, False]):
            if flag and start is None:
                start = index
            elif not flag and start is not None:
                begin, end = start * frame / sample_rate, index * frame / sample_rate
                if spans and begin - spans[-1].end_s < self.min_gap_s:
                    spans[-1] = Span(spans[-1].start_s, end)
                else:
                    spans.append(Span(begin, end))
                start = None
        return spans


class FakeQualityEstimator:
    """A quality estimator that reports the same SNR and C50 for every frame."""

    def __init__(self, snr_db: float = 30.0, c50_db: float = 40.0, hop_s: float = 0.01) -> None:
        self.snr_db = snr_db
        self.c50_db = c50_db
        self.hop_s = hop_s

    def frames(self, audio: np.ndarray, sample_rate: int) -> QualityTrack:
        count = int(np.ceil(len(audio) / (self.hop_s * sample_rate)))
        return QualityTrack(
            hop_s=self.hop_s,
            snr_db=np.full(count, self.snr_db),
            c50_db=np.full(count, self.c50_db),
        )


class FakeEnhancer:
    """An enhancer step that applies `fn(audio, sample_rate)`, or returns a copy without one."""

    def __init__(
        self,
        name: str = "identity",
        fn: Callable[[np.ndarray, int], np.ndarray] | None = None,
    ) -> None:
        self.name = name
        self._fn = fn

    def apply(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if self._fn is None:
            return np.array(audio, dtype=np.float32, copy=True)
        return np.asarray(self._fn(audio, sample_rate), dtype=np.float32)


class FakeAsr:
    """An ASR that returns a scripted text; word `i` of `n` takes the first 90% of slot `i`."""

    def __init__(self, name: str = "whisperx", text: str = "") -> None:
        self.name = name
        self.text = text

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Transcript:
        tokens = self.text.split()
        duration = len(audio) / sample_rate
        slot = duration / len(tokens) if tokens else 0.0
        words = tuple(
            Word(text=token, start_s=i * slot, end_s=i * slot + 0.9 * slot)
            for i, token in enumerate(tokens)
        )
        return Transcript(text=self.text, words=words)


class FakeArousal:
    """An arousal monitor that rises with level: -40 dBFS RMS or below is 0, -10 or above is 1."""

    def arousal(self, audio: np.ndarray, sample_rate: int) -> float:
        rms = float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float64) ** 2))) if len(audio) else 0
        level = 20.0 * np.log10(rms) if rms > 0 else -120.0
        return float(np.clip((level + 40.0) / 30.0, 0.0, 1.0))
