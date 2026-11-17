"""The two speaker embedders behind the `SpeakerEmbedder` protocol: WavLM-SV and ECAPA.

They load in the order `target.EMBEDDERS` names them, and every gate holds on both. WavLM-SV is
transformers' `WavLMForXVector` with its feature extractor; its x-vector is L2-normalised here.
ECAPA is speechbrain's `EncoderClassifier`. speechbrain copies a model's files into a `savedir`
before loading; the cache is read-only to every job but `fetch`, so `savedir` is a fresh
temporary folder, and `pretrained_path` is overridden to the snapshot so the checkpoint is read
from the cache rather than looked up on the hub. Both models take 16 kHz audio.
"""

from __future__ import annotations

import tempfile

import numpy as np

from ..errors import JobError
from ..stages.clean import resample
from ..stages.fetch import SCORE, library_errors, model_dir, prepare

WAVLM_MODEL = "wavlm-sv"
ECAPA_MODEL = "ecapa"
RATE = 16000


def unit(vector: object, name: str) -> np.ndarray:
    """Return the vector flattened and L2-normalised.

    Raises:
        JobError: The vector is zero or not finite, which no real embedding is.
    """
    out = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(out))
    if not np.isfinite(norm) or norm == 0.0:
        raise JobError(f"{name}: the embedding is zero or not finite")
    return out / norm


def _clip(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    return np.ascontiguousarray(resample(audio, sample_rate, RATE), dtype=np.float32)


class WavLmSv:
    """WavLM-SV's x-vector."""

    name = WAVLM_MODEL

    def __init__(self, extractor: object, model: object, device: str) -> None:
        self._extractor = extractor
        self._model = model
        self._device = device

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return the clip's L2-normalised x-vector."""
        import torch

        with library_errors(self.name), torch.inference_mode():
            features = self._extractor(  # type: ignore[operator]
                _clip(audio, sample_rate), sampling_rate=RATE, return_tensors="pt"
            )
            values = features["input_values"].to(self._device)
            vector = self._model(input_values=values).embeddings[0]  # type: ignore[operator]
            data = vector.detach().float().cpu().numpy()
        return unit(data, self.name)


class Ecapa:
    """speechbrain's ECAPA-TDNN embedding."""

    name = ECAPA_MODEL

    def __init__(self, classifier: object, device: str) -> None:
        self._classifier = classifier
        self._device = device

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return the clip's L2-normalised ECAPA embedding."""
        import torch

        with library_errors(self.name), torch.inference_mode():
            wavs = torch.from_numpy(_clip(audio, sample_rate))[None].to(self._device)
            vector = self._classifier.encode_batch(wavs)  # type: ignore[attr-defined]
            data = vector.detach().float().cpu().numpy()
        return unit(data, self.name)


def load_wavlm(device: str) -> WavLmSv:
    """Load WavLM-SV and its feature extractor from the pinned snapshot."""
    prepare(device, "transformers")
    folder = str(model_dir(SCORE, WAVLM_MODEL))
    with library_errors(WAVLM_MODEL):
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

        extractor = Wav2Vec2FeatureExtractor.from_pretrained(folder, local_files_only=True)
        model = WavLMForXVector.from_pretrained(folder, local_files_only=True)
        model = model.to(device).eval()
    return WavLmSv(extractor, model, device)


def load_ecapa(device: str) -> Ecapa:
    """Load ECAPA from the pinned snapshot, through a temporary `savedir`."""
    prepare(device, "speechbrain")
    folder = str(model_dir(SCORE, ECAPA_MODEL))
    savedir = tempfile.mkdtemp(prefix="mvd-ecapa-")
    with library_errors(ECAPA_MODEL):
        from speechbrain.inference.speaker import EncoderClassifier

        classifier = EncoderClassifier.from_hparams(
            source=folder,
            savedir=savedir,
            overrides={"pretrained_path": folder},
            run_opts={"device": device},
        )
        if classifier is None:
            raise RuntimeError("the classifier did not load")
    return Ecapa(classifier, device)


def load(device: str) -> list[WavLmSv | Ecapa]:
    """Return the embedders in the order the gates name them: WavLM-SV, then ECAPA."""
    return [load_wavlm(device), load_ecapa(device)]
