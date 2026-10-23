"""Deterministic speech-like audio for realtime voice tests.

Ingress, endpointing and barge-in tests need audio with a known shape: bursts that an energy VAD
scores as speech, silences it scores as silence, at exact lengths. Bursts are band-limited noise
(300 to 3400 Hz, the telephone speech band) under a syllable-rate envelope, and silence is
low-level white noise, so a test exercises real thresholds rather than all-zero input. All of it
comes from one seeded generator: the same parts and seed give the same samples on every run.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 512

_BAND_HZ = (300.0, 3400.0)
_SYLLABLE_HZ = 4.0
_EDGE_MS = 10


@dataclass(frozen=True, slots=True)
class AudioPart:
    """One stretch of scripted audio: a speech-like burst or silence, in milliseconds."""

    kind: Literal["speech", "silence"]
    ms: int


def speech_burst(ms: int) -> AudioPart:
    return AudioPart("speech", ms)


def silence(ms: int) -> AudioPart:
    return AudioPart("silence", ms)


def scripted_audio(
    parts: Sequence[AudioPart],
    *,
    seed: int = 0,
    sample_rate: int = SAMPLE_RATE,
    speech_rms: float = 0.1,
    noise_rms: float = 0.0005,
) -> np.ndarray:
    """Render parts in order as mono float32 audio.

    Each part is `round(ms * sample_rate / 1000)` samples long, so lengths add up exactly.
    """
    rng = np.random.default_rng(seed)
    pieces = []
    for part in parts:
        count = round(part.ms * sample_rate / 1000)
        if part.kind == "speech":
            pieces.append(_burst(rng, count, sample_rate, speech_rms))
        else:
            pieces.append(rng.standard_normal(count) * noise_rms)
    if not pieces:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(pieces).astype(np.float32)


def _burst(rng: np.random.Generator, count: int, sample_rate: int, rms: float) -> np.ndarray:
    if count == 0:
        return np.zeros(0)
    spectrum = np.fft.rfft(rng.standard_normal(count))
    freqs = np.fft.rfftfreq(count, d=1.0 / sample_rate)
    spectrum[(freqs < _BAND_HZ[0]) | (freqs > _BAND_HZ[1])] = 0
    band = np.fft.irfft(spectrum, n=count)
    band /= max(float(np.sqrt(np.mean(band**2))), 1e-12)
    t = np.arange(count) / sample_rate
    phase = rng.uniform(0, np.pi)
    envelope = 0.6 + 0.4 * np.sin(np.pi * _SYLLABLE_HZ * t + phase) ** 2
    edge = min(count // 2, _EDGE_MS * sample_rate // 1000)
    if edge:
        ramp = np.linspace(0.0, 1.0, edge)
        envelope[:edge] *= ramp
        envelope[-edge:] *= ramp[::-1]
    return band * envelope * rms


def windows(audio: np.ndarray, size: int = WINDOW_SAMPLES) -> list[np.ndarray]:
    """Cut audio into whole VAD windows, dropping a shorter tail."""
    return [audio[start : start + size] for start in range(0, len(audio) - size + 1, size)]


def to_pcm16(audio: np.ndarray) -> bytes:
    """Convert float32 audio to little-endian PCM16 with clipping, as the uplink carries it."""
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767).astype("<i2").tobytes()
