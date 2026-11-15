"""Tests for the prosody summary: F0 by `librosa.pyin`, energy and speech rate (T6)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from scripts.voice_data.bindings.protocols import Word
from scripts.voice_data.records import Prosody
from scripts.voice_data.stages import prosody


def _keep_numba_cache_in_the_worktree() -> None:
    # pyin's numba kernels take about 20 s to compile in a fresh container, and the runner starts
    # one per run. A cache under this folder's gitignored `__pycache__` survives between runs;
    # where the folder is read-only (a reviewer's run) numba keeps its default and compiles.
    cache = Path(__file__).resolve().parent / "__pycache__" / "numba"
    try:
        cache.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    if os.access(cache, os.W_OK):
        os.environ.setdefault("NUMBA_CACHE_DIR", str(cache))


_keep_numba_cache_in_the_worktree()

RATE = 16000


def _harmonic(f0_contour: np.ndarray, rate: int = RATE, harmonics: int = 12) -> np.ndarray:
    # A voiced signal that follows `f0_contour` (one F0 value per sample), 1/k harmonic amplitudes.
    phase = 2.0 * np.pi * np.cumsum(f0_contour) / rate
    signal = sum(np.sin(k * phase) / k for k in range(1, harmonics + 1))
    return (0.3 * signal / np.max(np.abs(signal))).astype(np.float32)


SYLLABLE_S = 0.15


def _syllable_train(count: int, per_second: float, rate: int = RATE) -> np.ndarray:
    # `count` Hann-shaped syllables on a steady 150 Hz voice, one every 1/per_second s.
    period = int(rate / per_second)
    length = period * count
    carrier = _harmonic(np.full(length, 150.0), rate)
    envelope = np.zeros(length)
    burst = np.hanning(int(SYLLABLE_S * rate))
    for i in range(count):
        envelope[i * period : i * period + len(burst)] = burst
    return (carrier * envelope).astype(np.float32)


@pytest.mark.parametrize(
    "contour",
    [
        pytest.param(lambda t: 150.0 + 50.0 * t, id="glide-150-250"),
        pytest.param(lambda t: 210.0 + 40.0 * np.sin(2.0 * np.pi * 1.5 * t), id="sine-170-250"),
        pytest.param(lambda t: 110.0 + 15.0 * np.sin(2.0 * np.pi * 0.7 * t), id="low-95-125"),
    ],
)
def test_f0_mean_and_std_are_within_three_percent(contour) -> None:
    t = np.arange(2 * RATE) / RATE
    f0 = contour(t)
    mean, std = prosody.f0_stats(_harmonic(f0), RATE)
    assert mean == pytest.approx(float(np.mean(f0)), rel=0.03)
    assert std == pytest.approx(float(np.std(f0)), rel=0.03)


def test_silence_has_no_f0() -> None:
    assert prosody.f0_stats(np.zeros(RATE, dtype=np.float32), RATE) == (0.0, 0.0)


def test_energy_follows_gain_exactly() -> None:
    voice = _harmonic(np.full(RATE, 180.0))
    quiet = prosody.active_energy_db(voice, RATE)
    assert prosody.active_energy_db(2.0 * voice, RATE) - quiet == pytest.approx(6.0206, abs=1e-3)


def test_energy_counts_active_speech_only() -> None:
    voice = _harmonic(np.full(RATE, 180.0))
    with_pauses = np.concatenate(
        [np.zeros(RATE, np.float32), voice, np.zeros(2 * RATE, np.float32)]
    )
    level = 10.0 * np.log10(np.mean(voice.astype(np.float64) ** 2))
    assert prosody.active_energy_db(voice, RATE) == pytest.approx(level, abs=0.1)
    assert prosody.active_energy_db(with_pauses, RATE) == pytest.approx(level, abs=0.1)
    assert prosody.active_energy_db(np.zeros(RATE, np.float32), RATE) == float("-inf")


@pytest.mark.parametrize(
    ("word", "count"),
    [
        ("the", 1),
        ("hello", 2),
        ("make", 1),
        ("table", 2),
        ("beautiful", 3),
        ("rhythm", 1),
        ("Okay,", 2),
        ("42", 0),
    ],
)
def test_syllables_count_vowel_groups(word: str, count: int) -> None:
    assert prosody.syllables(word) == count


def test_rate_from_word_times_is_syllables_over_the_spoken_span() -> None:
    words = [
        Word("hello", 0.5, 0.9),  # 2
        Word("beautiful", 1.0, 1.6),  # 3
        Word("table", 1.7, 2.5),  # 2
    ]
    assert prosody.rate_from_words(words) == pytest.approx(7 / 2.0)
    assert prosody.rate_from_words([]) == 0.0


def _train_rate(count: int, per_second: float) -> float:
    # Syllables over the span from the first syllable's start to the last one's end.
    return count / ((count - 1) / per_second + SYLLABLE_S)


@pytest.mark.parametrize(("count", "per_second"), [(12, 4.0), (6, 2.0), (15, 6.0)])
def test_rate_from_syllable_nuclei_counts_energy_peaks(count: int, per_second: float) -> None:
    audio = _syllable_train(count, per_second)
    rate = prosody.rate_from_nuclei(audio, RATE)
    assert rate == pytest.approx(_train_rate(count, per_second), rel=0.05)


def test_prosody_uses_word_times_when_given_and_nuclei_otherwise() -> None:
    audio = _syllable_train(8, 4.0)
    words = [Word("hello", 0.0, 1.0), Word("there", 1.0, 2.0)]  # 3 syllables over 2 s
    with_words = prosody.prosody(audio, RATE, words)
    without = prosody.prosody(audio, RATE)
    assert isinstance(with_words, Prosody)
    assert with_words.rate_sps == pytest.approx(1.5)
    assert without.rate_sps == pytest.approx(_train_rate(8, 4.0), rel=0.05)
    assert with_words.f0_mean_hz == pytest.approx(150.0, rel=0.03)
    assert with_words.energy_db == pytest.approx(prosody.active_energy_db(audio, RATE))
