"""Shared fixtures for the voice-data pipeline's unit tests.

Every later voice-data task reuses these, so each is documented here rather than in the tests:

- `private_root`: an initialised private root (a `scripts.voice_data.root.Root`) under the test's
  tmp directory, marker and layout written, with no `.git` above it. Use `private_root.path` for
  `--root` on the command line.
- `synth_voice(f0, formants, seconds, rate)`: a deterministic harmonic "voice" as float32 mono.
  Distinct F0 and formants give distinct speakers to the fakes of the embedder protocols, so
  retrieval and gate tests need no weights.
- `mix(parts, rate, seconds=None, noise_snr_db=None, seed=0)`: overlays signals at offsets, with
  optional white noise, for overlap, music-bed and noisy-segment cases.
- `canaries`: the canary transcript, source file name, private term and key value of the privacy
  tests (Test strategy). It sets every key env var to the canary key value, so a command that
  prints a value is caught.
- `assert_no_canary(text)`: fails when any canary appears in `text`, case-insensitively, without
  printing the canary itself, so a failing test does not repeat the leak in its own output.

`synth_voice`, `mix` and `assert_no_canary` are fixtures that return the function, so a test asks
for them by name and needs no import from this file.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from scripts.voice_data import root as root_mod

KEY_ENV_NAMES = ("HF_TOKEN", "ELEVENLABS_API_KEY", "MIST_VOICE_TEXT_LLM_KEY")


@dataclass(frozen=True, slots=True)
class Canaries:
    """The four planted strings of the privacy tests.

    Each is unique enough that a hit can only come from the plant, never from ordinary output.
    """

    transcript: str = "the quartz canary whispers seven lanterns"
    source_name: str = "Canary Episode S01E01 zqxv.mkv"
    term: str = "zqxvcanaryterm"
    key: str = "sk-canary-key-7f3a9b1c"

    def all(self) -> tuple[str, ...]:
        """Return every canary, for checks that must cover each one."""
        return (self.transcript, self.source_name, self.term, self.key)


def _assert_no_canary(text: str) -> None:
    lowered = text.lower()
    for index, value in enumerate(Canaries().all()):
        if value.lower() in lowered:
            pytest.fail(f"canary #{index} appears in the checked text")


def _resonance(freqs: np.ndarray, centre: float, bandwidth: float) -> np.ndarray:
    # A second-order resonance's magnitude response, enough to shape harmonic amplitudes.
    return 1.0 / np.sqrt((1.0 - (freqs / centre) ** 2) ** 2 + (freqs * bandwidth / centre**2) ** 2)


def _synth_voice(
    f0: float,
    formants: Sequence[float | tuple[float, float]],
    seconds: float,
    rate: int,
) -> np.ndarray:
    t = np.arange(int(round(seconds * rate)), dtype=np.float64) / rate
    # A slow 5 Hz vibrato of 2 percent keeps the pitch track from being a single line.
    inst_f0 = f0 * (1.0 + 0.02 * np.sin(2.0 * np.pi * 5.0 * t))
    phase = 2.0 * np.pi * np.cumsum(inst_f0) / rate
    harmonics = np.arange(1, max(1, int((rate / 2.0) // (f0 * 1.04))) + 1)
    amps = 1.0 / harmonics
    for formant in formants:
        centre, bandwidth = formant if isinstance(formant, tuple) else (formant, 80.0)
        amps = amps * _resonance(harmonics * f0, centre, bandwidth)
    signal = np.zeros_like(t)
    for k, amp in zip(harmonics, amps, strict=True):
        signal += amp * np.sin(k * phase)
    # A 4 Hz syllable-rate envelope makes energy and voicing vary like speech.
    signal *= 0.6 + 0.4 * np.sin(2.0 * np.pi * 4.0 * t) ** 2
    peak = float(np.max(np.abs(signal))) or 1.0
    return (0.5 * signal / peak).astype(np.float32)


def _mix(
    parts: Sequence[tuple[np.ndarray, float] | tuple[np.ndarray, float, float]],
    rate: int,
    seconds: float | None = None,
    noise_snr_db: float | None = None,
    seed: int = 0,
) -> np.ndarray:
    ends = [int(round(part[1] * rate)) + len(part[0]) for part in parts]
    length = int(round(seconds * rate)) if seconds is not None else max(ends, default=0)
    out = np.zeros(length, dtype=np.float64)
    for part in parts:
        signal, start_s = part[0], part[1]
        gain = 10.0 ** (part[2] / 20.0) if len(part) == 3 else 1.0
        start = int(round(start_s * rate))
        if start >= length:
            continue
        chunk = signal[: length - start].astype(np.float64)
        out[start : start + len(chunk)] += gain * chunk
    if noise_snr_db is not None:
        rms = float(np.sqrt(np.mean(out**2))) if length else 0.0
        noise_rms = rms / (10.0 ** (noise_snr_db / 20.0))
        out += np.random.default_rng(seed).normal(0.0, noise_rms, length)
    return out.astype(np.float32)


@pytest.fixture
def private_root(tmp_path: Path) -> root_mod.Root:
    """An initialised private root under the test's tmp directory."""
    return root_mod.init_root(tmp_path / "voice-root", "private")


@pytest.fixture
def synth_voice() -> Callable[..., np.ndarray]:
    """The harmonic voice synthesiser; formants are Hz or (Hz, bandwidth) pairs."""
    return _synth_voice


@pytest.fixture
def mix() -> Callable[..., np.ndarray]:
    """The overlay helper; each part is `(signal, start_s)` or `(signal, start_s, gain_db)`."""
    return _mix


@pytest.fixture
def canaries(monkeypatch: pytest.MonkeyPatch) -> Canaries:
    """Plant the canary key value in every key env var and return the four canaries."""
    planted = Canaries()
    for name in KEY_ENV_NAMES:
        monkeypatch.setenv(name, planted.key)
    return planted


@pytest.fixture
def assert_no_canary() -> Callable[[str], None]:
    """The canary check, as a function of the text to check."""
    return _assert_no_canary
