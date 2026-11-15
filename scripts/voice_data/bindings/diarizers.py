"""pyannote's speaker-diarization-community-1 behind the `Diarizer` protocol.

The pipeline gives two annotations: `exclusive_speaker_diarization`, where no two turns overlap,
is read as the turns; `speaker_diarization`, which keeps overlapped speech, is where the overlaps
are measured, as every stretch two or more of its tracks cover. The overlaps come from the tracks
here rather than from the annotation's own overlap method, so the binding relies on one
annotation call, `itertracks`.

The extract stage asks for turns and overlaps of the same audio in turn, so the last result is
kept and reused when the next call passes identical samples.
"""

from __future__ import annotations

import hashlib

import numpy as np

from ..stages.clean import resample
from ..stages.fetch import EXTRACT, library_errors, model_dir, prepare
from .protocols import Span, Turn

MODEL = "diarizer-community-1"
RATE = 16000
LOGGERS = ("pyannote", "speechbrain", "lightning", "pytorch_lightning", "lightning_fabric")

Track = tuple[float, float, str]


def overlapped(tracks: list[Track]) -> list[Span]:
    """Return the stretches covered by two or more tracks, merged, in time order."""
    # At one instant, ends sort before starts, so touching tracks do not count as overlapping.
    events = sorted([(s, 1) for s, _e, _l in tracks] + [(e, -1) for _s, e, _l in tracks])
    spans: list[Span] = []
    active, opened = 0, 0.0
    for time, step in events:
        before, active = active, active + step
        if before < 2 <= active:
            opened = time
        elif active < 2 <= before and time > opened:
            if spans and spans[-1].end_s >= opened:
                spans[-1] = Span(spans[-1].start_s, time)
            else:
                spans.append(Span(opened, time))
    return spans


def _tracks(annotation: object) -> list[Track]:
    return [
        (float(segment.start), float(segment.end), str(label))
        for segment, _track, label in annotation.itertracks(yield_label=True)  # type: ignore[attr-defined]
    ]


class CommunityDiarizer:
    """Exclusive turns and overlaps from one pipeline run per clip."""

    def __init__(self, pipeline: object) -> None:
        self._pipeline = pipeline
        self._last: tuple[tuple[object, ...], list[Track], list[Track]] | None = None

    def _run(self, audio: np.ndarray, sample_rate: int) -> tuple[list[Track], list[Track]]:
        import torch

        samples = np.ascontiguousarray(audio, dtype=np.float32)
        digest = hashlib.blake2b(samples.tobytes(), digest_size=16).hexdigest()
        key = (sample_rate, samples.shape, digest)
        if self._last is not None and self._last[0] == key:
            return self._last[1], self._last[2]
        clip = np.ascontiguousarray(resample(samples, sample_rate, RATE), dtype=np.float32)
        file = {"waveform": torch.from_numpy(clip)[None], "sample_rate": RATE}
        with library_errors(MODEL):
            output = self._pipeline(file)  # type: ignore[operator]
            exclusive = _tracks(output.exclusive_speaker_diarization)
            regular = _tracks(output.speaker_diarization)
        self._last = (key, exclusive, regular)
        return exclusive, regular

    def turns(self, audio: np.ndarray, sample_rate: int) -> list[Turn]:
        """Return the exclusive turns, in time order."""
        exclusive, _ = self._run(audio, sample_rate)
        return [Turn(s, e, label) for s, e, label in sorted(exclusive)]

    def overlaps(self, audio: np.ndarray, sample_rate: int) -> list[Span]:
        """Return the spans where two or more speakers talk at once."""
        _, regular = self._run(audio, sample_rate)
        return overlapped(regular)


def load(device: str) -> CommunityDiarizer:
    """Load the pipeline from its pinned snapshot folder, as the card's offline use does."""
    prepare(device, *LOGGERS)
    folder = model_dir(EXTRACT, MODEL)
    with library_errors(MODEL):
        import torch
        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(str(folder))
        if pipeline is None:
            raise RuntimeError("the pipeline did not load")
        pipeline.to(torch.device(device))
    return CommunityDiarizer(pipeline)
