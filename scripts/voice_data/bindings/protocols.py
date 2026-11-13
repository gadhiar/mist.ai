"""The model protocols the seed stages call, and the values they exchange.

Stages take every model as one of these protocols, so their unit tests pass the deterministic
fakes in `tests/unit/voice_data/fakes.py` and only the bindings beside this file load weights.
Audio is always a 1-D float32 mono array at the `sample_rate` passed with it, and times are
seconds from the start of that array. The `Teacher` protocol and `RefClip` live in
`scripts/voice_data/teachers/base.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class Span:
    """A stretch of a recording: speech from a VAD, or overlapped speech from a diarizer."""

    start_s: float
    end_s: float


@dataclass(frozen=True, slots=True)
class Turn:
    """One speaker's exclusive turn; `speaker` is the diarizer's label, unique per recording only."""

    start_s: float
    end_s: float
    speaker: str


@dataclass(frozen=True, slots=True, eq=False)
class QualityTrack:
    """Per-frame SNR and C50 in dB; frame `i` starts at `i * hop_s` seconds.

    Compared by identity, not value: the fields are arrays, whose `==` is elementwise.
    """

    hop_s: float
    snr_db: np.ndarray
    c50_db: np.ndarray


@dataclass(frozen=True, slots=True)
class Word:
    """One recognised word and its time in the audio, which the prosody speech rate reads."""

    text: str
    start_s: float
    end_s: float


@dataclass(frozen=True, slots=True)
class Transcript:
    """An ASR result: the full text, and its words with times where the engine gives them."""

    text: str
    words: tuple[Word, ...]


class SpeakerEmbedder(Protocol):
    """A speaker-verification embedder; WavLM-SV and ECAPA are the two the gates use."""

    name: str

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return the clip's 1-D, L2-normalised speaker embedding."""
        ...


class Separator(Protocol):
    """A dialogue separator that removes music and effects from a channel."""

    name: str

    def dialogue(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return the dialogue stem, the same length and rate as the input."""
        ...


class Diarizer(Protocol):
    """A speaker diarizer."""

    def turns(self, audio: np.ndarray, sample_rate: int) -> list[Turn]:
        """Return the exclusive (single-speaker) turns, in time order."""
        ...

    def overlaps(self, audio: np.ndarray, sample_rate: int) -> list[Span]:
        """Return the spans where two or more speakers talk at once."""
        ...


class Vad(Protocol):
    """A voice activity detector."""

    def speech(self, audio: np.ndarray, sample_rate: int) -> list[Span]:
        """Return the speech spans, in time order."""
        ...


class QualityEstimator(Protocol):
    """A non-intrusive estimator of noise and reverberation."""

    def frames(self, audio: np.ndarray, sample_rate: int) -> QualityTrack:
        """Return the SNR and C50 of each frame, in dB."""
        ...


class Enhancer(Protocol):
    """One cleaning step (denoising or dereverberation), kept only if the cleaning gate passes."""

    name: str

    def apply(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return the cleaned audio at the same rate."""
        ...


class Asr(Protocol):
    """A speech recogniser; WhisperX gives word times, CrisperWhisper verbatim text."""

    name: str

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Transcript:
        """Return the transcript, with word times where the engine gives them."""
        ...


class ArousalMonitor(Protocol):
    """An arousal estimator: a monitor reported beside takes and segments, never a gate."""

    def arousal(self, audio: np.ndarray, sample_rate: int) -> float:
        """Return the clip's arousal from 0 (low) to 1 (high)."""
        ...
