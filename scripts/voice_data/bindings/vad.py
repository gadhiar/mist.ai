"""Silero VAD behind the `Vad` protocol.

The `silero-vad` package carries its own JIT weights, so this model has no `families.json` entry
and needs no cache. It runs on the CPU whatever device the stage names: it is a small recurrent
model that processes 32 ms windows one after another, where a GPU adds transfer cost and no speed.
"""

from __future__ import annotations

import numpy as np

from ..stages.clean import resample
from ..stages.fetch import library_errors, prepare
from .protocols import Span

MODEL = "silero-vad"
RATE = 16000


class SileroVad:
    """Speech spans from Silero's timestamps, read in samples and returned in seconds."""

    def __init__(self, model: object, timestamps: object) -> None:
        self._model = model
        self._timestamps = timestamps

    def speech(self, audio: np.ndarray, sample_rate: int) -> list[Span]:
        """Return the speech spans of the clip, in seconds from its start, in time order."""
        import torch

        clip = np.ascontiguousarray(resample(audio, sample_rate, RATE), dtype=np.float32)
        end_s = len(clip) / RATE
        with library_errors(MODEL), torch.inference_mode():
            stamps = self._timestamps(  # type: ignore[operator]
                torch.from_numpy(clip), self._model, sampling_rate=RATE, return_seconds=False
            )
        return [Span(s["start"] / RATE, min(s["end"] / RATE, end_s)) for s in stamps]


def load(device: str) -> SileroVad:
    """Load Silero VAD from the package's own weights."""
    prepare(device, "silero_vad")
    with library_errors(MODEL):
        from silero_vad import get_speech_timestamps, load_silero_vad

        model = load_silero_vad()
    return SileroVad(model, get_speech_timestamps)
