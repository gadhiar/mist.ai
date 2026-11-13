"""A clip's prosody summary: F0 mean and spread, active-speech energy and speech rate.

State proposal (T7), reference choice (T8), take selection (T15) and the bench compare these
numbers per state, so every stage computes them here, the same way. Energy is measured over
active frames only, so the pauses in a line do not lower it, and it moves exactly with gain,
which `score.teacher_gain` relies on.

`librosa` is imported inside `f0_stats` only: its pitch tracker compiles numba kernels on first
use, which callers that need energy alone should not pay for.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

from ..bindings.protocols import Word
from ..records import Prosody

FRAME_S = 0.025
HOP_S = 0.010
ACTIVE_RANGE_DB = 30.0
ACTIVE_PERCENTILE = 99.0
NUCLEUS_DIP_DB = 2.0
F0_MIN_HZ = 60.0
F0_MAX_HZ = 500.0

_VOWEL_GROUP = re.compile(r"[aeiouy]+")


def _frame_power(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    frame = max(1, int(round(FRAME_S * sample_rate)))
    hop = max(1, int(round(HOP_S * sample_rate)))
    signal = np.asarray(audio, dtype=np.float64)
    if len(signal) < frame:
        signal = np.pad(signal, (0, frame - len(signal)))
    windows = np.lib.stride_tricks.sliding_window_view(signal, frame)[::hop]
    return np.mean(windows**2, axis=1)


def _active(power: np.ndarray) -> np.ndarray:
    # Frames within ACTIVE_RANGE_DB of a high percentile of the clip, so one click cannot set the
    # reference and long silences cannot lower it below the speech.
    if power.size == 0 or power.max() <= 0.0:
        return np.zeros(power.shape, dtype=bool)
    reference = float(np.percentile(power, ACTIVE_PERCENTILE))
    return (power > 0.0) & (power >= reference * 10.0 ** (-ACTIVE_RANGE_DB / 10.0))


def active_energy_db(audio: np.ndarray, sample_rate: int) -> float:
    """Return the mean power of the active 25 ms frames in dB relative to full scale.

    A frame is active within 30 dB of the clip's 99th-percentile frame power. Silence gives
    `-inf`.
    """
    power = _frame_power(audio, sample_rate)
    active = _active(power)
    if not active.any():
        return float("-inf")
    return float(10.0 * np.log10(np.mean(power[active])))


def f0_stats(
    audio: np.ndarray, sample_rate: int, fmin: float = F0_MIN_HZ, fmax: float = F0_MAX_HZ
) -> tuple[float, float]:
    """Return the mean and standard deviation of F0 in Hz over `librosa.pyin`'s voiced frames.

    A clip with no voiced frame returns `(0.0, 0.0)`, which keeps the record valid JSON.
    """
    import librosa

    f0, voiced, _ = librosa.pyin(
        np.asarray(audio, dtype=np.float32), fmin=fmin, fmax=fmax, sr=sample_rate
    )
    values = f0[voiced & np.isfinite(f0)]
    if values.size == 0:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))


def syllables(word: str) -> int:
    """Return an English syllable count from vowel groups, `y` included.

    A final silent `e` is dropped unless it ends `-le` or is the only vowel group. A word with no
    letter counts zero; any other word at least one.
    """
    letters = re.sub(r"[^a-z]", "", word.lower())
    if not letters:
        return 0
    count = len(_VOWEL_GROUP.findall(letters))
    if count > 1 and letters.endswith("e") and not letters.endswith("le"):
        count -= 1
    return max(count, 1)


def rate_from_words(words: Sequence[Word]) -> float:
    """Return syllables per second from the first word's start to the last word's end."""
    if not words:
        return 0.0
    span = max(w.end_s for w in words) - min(w.start_s for w in words)
    if span <= 0.0:
        return 0.0
    return sum(syllables(w.text) for w in words) / span


def rate_from_nuclei(audio: np.ndarray, sample_rate: int) -> float:
    """Return syllable nuclei per second over the active span, from the energy contour.

    A nucleus is a local energy peak in an active frame that rises at least 2 dB above the dip
    since the previous nucleus; a peak without that dip belongs to the previous syllable.
    """
    power = _frame_power(audio, sample_rate)
    active = _active(power)
    if not active.any():
        return 0.0
    level = 10.0 * np.log10(power + 1e-20)
    nuclei = 0
    last: int | None = None
    for i in range(1, len(level) - 1):
        if not (active[i] and level[i] > level[i - 1] and level[i] >= level[i + 1]):
            continue
        if last is None or level[i] - level[last : i + 1].min() >= NUCLEUS_DIP_DB:
            nuclei += 1
            last = i
        elif level[i] > level[last]:
            last = i
    frames = np.flatnonzero(active)
    span = (frames[-1] - frames[0]) * HOP_S + FRAME_S
    return nuclei / span


def prosody(audio: np.ndarray, sample_rate: int, words: Sequence[Word] | None = None) -> Prosody:
    """Return the clip's prosody; the rate comes from word times when any are given."""
    mean, std = f0_stats(audio, sample_rate)
    rate = rate_from_words(words) if words else rate_from_nuclei(audio, sample_rate)
    return Prosody(
        f0_mean_hz=mean,
        f0_std_hz=std,
        rate_sps=rate,
        energy_db=active_energy_db(audio, sample_rate),
    )
