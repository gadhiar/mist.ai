"""audeering's dimensional emotion model behind the `ArousalMonitor` protocol.

The checkpoint is not a stock transformers class: its card defines `EmotionModel`, a wav2vec 2.0
encoder whose hidden states are mean-pooled into a regression head (dense, tanh, out_proj), and
that definition is reproduced here. The head's three outputs are arousal, dominance and valence,
each about 0 to 1; arousal is the first, clipped to 0..1. A checkpoint that leaves any of these
weights missing would give an untrained head, so missing keys are refused rather than warned.
"""

from __future__ import annotations

import numpy as np

from ..errors import JobError
from ..stages.clean import resample
from ..stages.fetch import SCORE, library_errors, model_dir, prepare

MODEL = "arousal"
RATE = 16000
AROUSAL_INDEX = 0


def emotion_model_class() -> type:
    """Return the card's `EmotionModel`, built on transformers' wav2vec 2.0 classes."""
    import torch
    from torch import nn
    from transformers.models.wav2vec2.modeling_wav2vec2 import (
        Wav2Vec2Model,
        Wav2Vec2PreTrainedModel,
    )

    class RegressionHead(nn.Module):
        """The card's head: dropout, dense, tanh, dropout, out_proj."""

        def __init__(self, config: object) -> None:
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # type: ignore[attr-defined]
            self.dropout = nn.Dropout(config.final_dropout)  # type: ignore[attr-defined]
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)  # type: ignore[attr-defined]

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            x = self.dropout(features)
            x = torch.tanh(self.dense(x))
            return self.out_proj(self.dropout(x))

    class EmotionModel(Wav2Vec2PreTrainedModel):  # type: ignore[misc, valid-type]
        """The card's model: the encoder's mean-pooled hidden states into the head."""

        def __init__(self, config: object) -> None:
            super().__init__(config)
            self.config = config
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = RegressionHead(config)
            self.post_init()

        def forward(self, input_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            hidden = self.wav2vec2(input_values)[0]
            pooled = torch.mean(hidden, dim=1)
            return pooled, self.classifier(pooled)

    return EmotionModel


class Arousal:
    """The model's arousal output, clipped to 0..1."""

    def __init__(self, extractor: object, model: object, device: str) -> None:
        self._extractor = extractor
        self._model = model
        self._device = device

    def arousal(self, audio: np.ndarray, sample_rate: int) -> float:
        """Return the clip's arousal from 0 (low) to 1 (high)."""
        import torch

        clip = np.ascontiguousarray(resample(audio, sample_rate, RATE), dtype=np.float32)
        with library_errors(MODEL), torch.inference_mode():
            features = self._extractor(  # type: ignore[operator]
                clip, sampling_rate=RATE, return_tensors="np"
            )
            values = np.asarray(features["input_values"], dtype=np.float32).reshape(1, -1)
            _pooled, logits = self._model(torch.from_numpy(values).to(self._device))  # type: ignore[operator]
            value = float(logits[0][AROUSAL_INDEX])
        return min(1.0, max(0.0, value))


def load(device: str) -> Arousal:
    """Load the card's model and the feature extractor from the pinned snapshot."""
    prepare(device, "transformers")
    folder = str(model_dir(SCORE, MODEL))
    with library_errors(MODEL):
        from transformers import Wav2Vec2FeatureExtractor

        model, info = emotion_model_class().from_pretrained(
            folder, local_files_only=True, output_loading_info=True
        )
        missing = list(info.get("missing_keys") or [])
        if missing:
            raise JobError(f"{MODEL}: {len(missing)} weights are missing from the checkpoint")
        model = model.to(device).eval()
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(folder, local_files_only=True)
    return Arousal(extractor, model, device)
