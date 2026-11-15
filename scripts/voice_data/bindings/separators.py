"""The dialogue separators behind the `Separator` protocol: TIGER-DnR and Mel-Band RoFormer.

TIGER-DnR (the dialogue, music and effects separator) loads through `TIGERDNR.from_pretrained`
from TIGER's clone in the image; its first output is the dialogue stem. Mel-Band RoFormer loads
the Music-Source-Separation-Training (MSST) way: the model built from its YAML config, then its
state dict read with `torch.load(..., weights_only=True)`, then MSST's `demix`. The dereverb
enhancer is the same architecture, so `enhancers` builds it with `load_melband` from here.

Both models work at 44.1 kHz, so the channel is resampled in and the stem back out, and the stem
is trimmed or padded to the input's length. MSST is loaded before TIGER: both clones are put at
the front of `sys.path`, and MSST imports top-level packages named `utils` and `models`.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..errors import JobError
from ..stages.clean import resample
from ..stages.fetch import (
    EXTRACT,
    UPSTREAM,
    library_errors,
    model_dir,
    model_entry,
    model_file,
    on_path,
    prepare,
)

TIGER_MODEL = "separator-dme"
MELBAND_MODEL = "separator-melband-vocals"
TIGER_DIR = UPSTREAM / "TIGER"
MSST_DIR = UPSTREAM / "Music-Source-Separation-Training"
TIGER_RATE = 44100
MSST_TYPE = "mel_band_roformer"
MSST_RATE = 44100
MSST_CHANNELS = 2
LOGGERS = ("look2hear",)


def fit(audio: np.ndarray, length: int) -> np.ndarray:
    """Return the clip trimmed or zero-padded to `length` samples, as float32."""
    out = np.asarray(audio, dtype=np.float32).reshape(-1)[:length]
    if len(out) < length:
        out = np.concatenate([out, np.zeros(length - len(out), dtype=np.float32)])
    return out


class TigerDnr:
    """TIGER-DnR's dialogue stem."""

    name = TIGER_MODEL

    def __init__(self, model: object, device: str) -> None:
        self._model = model
        self._device = device

    def dialogue(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return the dialogue stem at the input's rate and length."""
        import torch

        clip = np.ascontiguousarray(resample(audio, sample_rate, TIGER_RATE), dtype=np.float32)
        mixture = torch.from_numpy(clip)[None, None].to(self._device)
        with library_errors(self.name), torch.inference_mode():
            stems = self._model(mixture)  # type: ignore[operator]
            dialog = stems[0].detach().float().cpu().numpy().reshape(-1)
        return fit(resample(dialog, TIGER_RATE, sample_rate), len(audio))


class MelBandRoformer:
    """An MSST Mel-Band RoFormer; `dialogue` and `apply` both return its target stem."""

    def __init__(
        self, name: str, model: object, config: object, demix: object, stem: str, device: object
    ) -> None:
        self.name = name
        self._model = model
        self._config = config
        self._demix = demix
        self._stem = stem
        self._device = device

    def _audio_setting(self, key: str, default: int) -> int:
        audio = getattr(self._config, "audio", None)
        return int(getattr(audio, key, default) or default)

    def run(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return the target stem, mono, at the input's rate and length."""
        import torch

        rate = self._audio_setting("sample_rate", MSST_RATE)
        channels = self._audio_setting("num_channels", MSST_CHANNELS)
        clip = np.asarray(resample(audio, sample_rate, rate), dtype=np.float32)
        mix = torch.from_numpy(np.ascontiguousarray(np.stack([clip] * channels)))
        with library_errors(self.name), torch.inference_mode():
            result = self._demix(  # type: ignore[operator]
                self._config, self._model, mix, self._device, model_type=MSST_TYPE, pbar=False
            )
        if isinstance(result, Mapping):
            if self._stem not in result:
                raise JobError(f"{self.name}: demix returned no {self._stem} stem")
            result = result[self._stem]
        stem = np.asarray(result, dtype=np.float32)
        if stem.ndim == 3:
            stem = stem[0]
        mono = stem.mean(axis=0) if stem.ndim == 2 else stem
        return fit(resample(mono.astype(np.float32), rate, sample_rate), len(audio))

    def dialogue(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """The `Separator` call."""
        return self.run(audio, sample_rate)

    def apply(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """The `Enhancer` call."""
        return self.run(audio, sample_rate)


def _pair(model_id: str) -> tuple[str, str]:
    files = [str(f) for f in model_entry(EXTRACT, model_id).get("files", [])]  # type: ignore[union-attr]
    configs = [f for f in files if f.endswith((".yaml", ".yml"))]
    checkpoints = [f for f in files if f.endswith((".ckpt", ".pth", ".pt"))]
    if len(configs) != 1 or len(checkpoints) != 1:
        raise JobError(f"{model_id}: families.json must list one config and one checkpoint")
    return configs[0], checkpoints[0]


def load_melband(model_id: str, device: str) -> MelBandRoformer:
    """Build an MSST Mel-Band RoFormer from the config and checkpoint its model lists."""
    config_name, checkpoint_name = _pair(model_id)
    config_path = model_file(EXTRACT, model_id, config_name)
    checkpoint = model_file(EXTRACT, model_id, checkpoint_name)
    on_path(MSST_DIR)
    with library_errors(model_id):
        import torch
        from utils.model_utils import demix, prefer_target_instrument
        from utils.settings import get_model_from_config

        model, config = get_model_from_config(MSST_TYPE, str(config_path))
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        if isinstance(state, Mapping) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        target = torch.device(device)
        model = model.to(target).eval()
        stem = prefer_target_instrument(config)[0]
    return MelBandRoformer(model_id, model, config, demix, stem, target)


def load_tiger(device: str) -> TigerDnr:
    """Load TIGER-DnR from its pinned snapshot folder."""
    folder = model_dir(EXTRACT, TIGER_MODEL)
    on_path(TIGER_DIR)
    with library_errors(TIGER_MODEL):
        from look2hear.models import TIGERDNR

        model = TIGERDNR.from_pretrained(str(folder), local_files_only=True)
        model = model.to(device).eval()
    return TigerDnr(model, device)


def load(device: str) -> list[TigerDnr | MelBandRoformer]:
    """Return the separator candidates, TIGER-DnR first, on `device`."""
    prepare(device, *LOGGERS)
    melband = load_melband(MELBAND_MODEL, device)
    return [load_tiger(device), melband]
