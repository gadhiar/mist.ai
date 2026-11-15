"""Brouhaha behind the `QualityEstimator` protocol: SNR and C50 per frame.

Brouhaha's checkpoint names its model class in the brouhaha-vad repository, which is not on PyPI
and is cloned into the extract image, so that clone goes on `sys.path` before pyannote loads the
checkpoint. `strict=False` follows the model card; the repository's own task heads are not part
of the checkpoint's state.
"""

from __future__ import annotations

import numpy as np

from ..stages.clean import resample
from ..stages.fetch import EXTRACT, UPSTREAM, library_errors, model_dir, on_path, prepare
from .protocols import QualityTrack

MODEL = "quality-brouhaha"
RATE = 16000
BROUHAHA_DIR = UPSTREAM / "brouhaha-vad"
CHECKPOINT = "pytorch_model.bin"
LOGGERS = ("pyannote", "lightning", "pytorch_lightning", "lightning_fabric")


class Brouhaha:
    """Frames of (VAD, SNR dB, C50 dB), as the card iterates them, with the window's step."""

    def __init__(self, inference: object) -> None:
        self._inference = inference

    def frames(self, audio: np.ndarray, sample_rate: int) -> QualityTrack:
        """Return the SNR and C50 of each frame; frame `i` starts at `i` times the step."""
        import torch

        clip = np.ascontiguousarray(resample(audio, sample_rate, RATE), dtype=np.float32)
        file = {"waveform": torch.from_numpy(clip)[None], "sample_rate": RATE}
        with library_errors(MODEL):
            output = self._inference(file)  # type: ignore[operator]
            rows = [np.asarray(values, dtype=np.float64) for _frame, values in output]
            hop_s = float(output.sliding_window.step)
        data = np.stack(rows) if rows else np.zeros((0, 3))
        return QualityTrack(hop_s=hop_s, snr_db=data[:, 1].copy(), c50_db=data[:, 2].copy())


def load(device: str) -> Brouhaha:
    """Load Brouhaha from its pinned snapshot, on `device`."""
    prepare(device, *LOGGERS)
    folder = model_dir(EXTRACT, MODEL)
    checkpoint = folder / CHECKPOINT
    on_path(BROUHAHA_DIR)
    with library_errors(MODEL):
        import torch
        from pyannote.audio import Inference, Model

        model = Model.from_pretrained(
            str(checkpoint if checkpoint.is_file() else folder), strict=False
        )
        if model is None:
            raise RuntimeError("the checkpoint did not load")
        inference = Inference(model, device=torch.device(device))
    return Brouhaha(inference)
