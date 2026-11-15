"""The cleaning steps behind the `Enhancer` protocol: Sidon, then Mel-Band RoFormer dereverb.

The clean stage runs them in this order and keeps each only if its gate passes.

Sidon ships as two TorchScript files built for CUDA (a w2v-BERT feature extractor and a vocoder
decoder), so it loads on the GPU only and `load` refuses `cpu`. `apply` follows the recipe of
the authors' demo Space: a 50 Hz high-pass, 16 kHz, 24000 samples of padding at the end, chunks
of 96 s each padded by 160 samples on both sides, the encoder's `last_hidden_state` per chunk,
the last frame repeated once, then the decoder, whose 48 kHz output loses the repeated frame's
960 samples. The output is resampled to the input's rate and cut to its length. The features come
from Sidon's own `extract_seamless_m4t_features` instead of the Space's transformers extractor,
which would need w2v-BERT's preprocessor files from the hub.

The dereverb model is MSST's Mel-Band RoFormer with anvuew's dereverb weights; `separators`
builds it.
"""

from __future__ import annotations

import numpy as np

from ..errors import JobError
from ..stages.clean import resample
from ..stages.fetch import EXTRACT, UPSTREAM, library_errors, model_file, on_path, prepare
from .separators import MelBandRoformer, fit, load_melband

SIDON_MODEL = "enhancer-sidon"
DEREVERB_MODEL = "dereverb-melband"
SIDON_DIR = UPSTREAM / "Sidon" / "src"
SIDON_FILES = ("feature_extractor_cuda.pt", "decoder_cuda.pt")
SIDON_HIGHPASS_HZ = 50.0
SIDON_IN_RATE = 16000
SIDON_OUT_RATE = 48000
SIDON_PAD = 24000
SIDON_CHUNK = 16000 * 96
SIDON_CHUNK_PAD = 160
SIDON_TRIM = 960
LOGGERS = ("sidon",)


class Sidon:
    """Sidon's speech restoration, run as the demo Space runs it."""

    name = SIDON_MODEL

    def __init__(self, feature_extractor: object, decoder: object, extract: object, device: str):
        self._fe = feature_extractor
        self._decoder = decoder
        self._extract = extract
        self._device = device

    def apply(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return the restored clip at the input's rate and length."""
        import torch
        import torch.nn.functional as nnf
        import torchaudio.functional as taf

        with library_errors(self.name), torch.inference_mode():
            wave = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None]
            wave = taf.highpass_biquad(wave, sample_rate, SIDON_HIGHPASS_HZ)
            clip = resample(wave[0].numpy(), sample_rate, SIDON_IN_RATE)
            clip16 = nnf.pad(
                torch.from_numpy(np.ascontiguousarray(clip, dtype=np.float32)), (0, SIDON_PAD)
            )
            features = []
            for chunk in clip16.split(SIDON_CHUNK):
                padded = nnf.pad(chunk, (SIDON_CHUNK_PAD, SIDON_CHUNK_PAD))
                inputs = self._extract(  # type: ignore[operator]
                    padded, sampling_rate=SIDON_IN_RATE, device=self._device
                )
                hidden = self._fe(inputs["input_features"].to(self._device))  # type: ignore[operator]
                features.append(hidden["last_hidden_state"])
            feature = torch.cat(features, dim=1)
            feature = torch.cat([feature, feature[:, -1:]], dim=1)
            restored = self._decoder(feature.transpose(1, 2)).view(-1)[:-SIDON_TRIM]  # type: ignore[operator]
            out = restored.float().cpu().numpy()
        return fit(resample(out, SIDON_OUT_RATE, sample_rate), len(audio))


def load_sidon(device: str) -> Sidon:
    """Load Sidon's two TorchScript files onto the GPU.

    Raises:
        JobError: `device` is `cpu`; the files are built for CUDA.
    """
    prepare(device, *LOGGERS)
    if device != "cuda":
        raise JobError(f"{SIDON_MODEL} runs on cuda only")
    fe_path, decoder_path = (model_file(EXTRACT, SIDON_MODEL, name) for name in SIDON_FILES)
    on_path(SIDON_DIR)
    with library_errors(SIDON_MODEL):
        import torch
        from sidon.cleansing.audio import extract_seamless_m4t_features

        fe = torch.jit.load(str(fe_path), map_location=device).eval()
        decoder = torch.jit.load(str(decoder_path), map_location=device).eval()
    return Sidon(fe, decoder, extract_seamless_m4t_features, device)


def load(device: str) -> list[Sidon | MelBandRoformer]:
    """Return the cleaning steps in the order they run: Sidon, then dereverb."""
    sidon = load_sidon(device)
    return [sidon, load_melband(DEREVERB_MODEL, device)]
