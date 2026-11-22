"""The wake verifier's network: WavLM Base+ with its last four layers and a head fine-tuned.

The encoder is built as torchaudio's `WAVLM_BASE_PLUS.get_model()` builds it, less the download:
`wavlm_model(**WAVLM_BASE_PLUS._params)` and then a strict `load_state_dict` of the staged
`wavlm_base_plus.pth` (MIT, Microsoft; the weights torchaudio publishes). `wavlm_base()` is not
used, because its dropout and layer-drop defaults are not the bundle's. The bundle applies no
waveform normalisation for Base+ (`_normalize_waveform` is False), and neither does the exported
graph, so the trainer refuses to start if the installed torchaudio says otherwise.

The convolutional feature encoder and transformer layers 1 to 8 are frozen and kept in eval
mode even while the model trains, so their dropout and layer drop stay off; layers 9 to 12 and
the head train. The head takes a softmax-weighted sum of the 13 hidden states (the transformer's
input after the positional convolution, then each layer's output), pools it over time by
attentive statistics (a weighted mean and standard deviation, 1536 values), and maps that through
an MLP 1536 -> 256 -> 1. The model returns a logit; `ScoreGraph` adds the sigmoid for export.

The input is the waveform `verifier.pcm_to_waveform` makes: int16 over 32768, never padded.
"""

import hashlib
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor, nn
from torchaudio.models import Wav2Vec2Model, wavlm_model
from torchaudio.pipelines import WAVLM_BASE_PLUS

FIRST_TRAINABLE_LAYER = 9
"""Layers are counted from 1, as the design counts them: 9 to 12 train, 1 to 8 stay frozen."""
POOL_ATTENTION = 128
HEAD_HIDDEN = 256
HEAD_DROPOUT = 0.1
_VARIANCE_FLOOR = 1e-6


class ModelError(ValueError):
    """Weights or a bundle the verifier refuses to build from; the CLIs exit 2 with it."""


def base_plus_params() -> dict[str, object]:
    """A copy of the bundle's `wavlm_model` arguments, the model the weights were trained as."""
    return dict(WAVLM_BASE_PLUS._params)


def normalizes_waveform() -> bool:
    """The bundle's `_normalize_waveform`: True would mean the weights expect normalised input."""
    return bool(WAVLM_BASE_PLUS._normalize_waveform)


def build_wavlm(params: Mapping[str, object]) -> Wav2Vec2Model:
    """A random-initialised WavLM with `params`, ready for a strict load."""
    return wavlm_model(**params)


def load_wavlm(path: Path, params: Mapping[str, object]) -> Wav2Vec2Model:
    """`build_wavlm(params)` with the state dict at `path` loaded strictly; nothing downloads.

    Raises:
        ModelError: The file cannot be read as a state dict, or its keys or shapes are not the
            model's.
    """
    model = build_wavlm(params)
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ModelError(f"{path} could not be read as a state dict: {exc}") from exc
    if not isinstance(state, Mapping):
        raise ModelError(f"{path} holds a {type(state).__name__}, not a state dict")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ModelError(f"{path} is not a WavLM state dict for these params: {exc}") from exc
    return model


def sha256_path(path: Path) -> str:
    """The file's sha256, so a run records which weights it started from."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class StateEncoder(nn.Module):
    """An encoder that returns its hidden states and says which of its modules train.

    Every parameter outside `trainable_modules()` is frozen, and those modules alone follow
    `train()`; the rest stay in eval mode.
    """

    dim: int
    num_states: int

    def hidden_states(self, waveform: Tensor) -> list[Tensor]:
        """`num_states` tensors of shape `[batch, time, dim]`."""
        raise NotImplementedError

    def trainable_modules(self) -> list[nn.Module]:
        """The modules whose parameters train."""
        raise NotImplementedError


class WavLMEncoder(StateEncoder):
    """torchaudio's WavLM read layer by layer, so every hidden state reaches the head.

    The loop is `Transformer.forward`'s: the positional convolution and its normalisation, then
    each layer with the relative position bias the first one computes. Layer drop applies only
    to the trainable layers while training, as the transformer would apply it.
    """

    def __init__(self, wavlm: Wav2Vec2Model, first_trainable: int = FIRST_TRAINABLE_LAYER) -> None:
        super().__init__()
        layers = wavlm.encoder.transformer.layers
        if not 1 <= first_trainable <= len(layers):
            raise ModelError(
                f"the first trainable layer must be 1 to {len(layers)}, got {first_trainable}"
            )
        self.wavlm = wavlm
        self.first_trainable = first_trainable
        self.dim = int(wavlm.encoder.feature_projection.projection.out_features)
        self.num_states = len(layers) + 1
        self.layer_drop = float(wavlm.encoder.transformer.layer_drop)

    def trainable_modules(self) -> list[nn.Module]:
        """Transformer layers `first_trainable` to the last."""
        return list(self.wavlm.encoder.transformer.layers[self.first_trainable - 1 :])

    def hidden_states(self, waveform: Tensor) -> list[Tensor]:
        """The transformer's input and each layer's output, `[batch, time, dim]` each."""
        features, _ = self.wavlm.feature_extractor(waveform, None)
        transformer = self.wavlm.encoder.transformer
        x = transformer._preprocess(self.wavlm.encoder.feature_projection(features))
        states = [x]
        position_bias = None
        for number, layer in enumerate(transformer.layers, start=1):
            dropped = (
                self.training
                and number >= self.first_trainable
                and self.layer_drop > 0
                and torch.rand(1).item() <= self.layer_drop
            )
            if not dropped:
                x, position_bias = layer(x, None, position_bias=position_bias)
            states.append(x)
        return states


class AttentiveStatsPool(nn.Module):
    """A learned weighting over time, then the weighted mean and standard deviation."""

    def __init__(self, dim: int, attention: int = POOL_ATTENTION) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, attention), nn.Tanh(), nn.Linear(attention, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        """`[batch, time, dim]` to `[batch, 2 * dim]`, in float32 so fp16 cannot cancel."""
        x = x.float()
        weights = torch.softmax(self.attention(x).float().squeeze(-1), dim=1).unsqueeze(-1)
        mean = (weights * x).sum(dim=1)
        variance = (weights * x * x).sum(dim=1) - mean * mean
        std = torch.sqrt(variance.clamp(min=_VARIANCE_FLOOR))
        return torch.cat([mean, std], dim=-1)


class VerifierHead(nn.Module):
    """Softmax-weighted hidden states, attentive statistics pooling and the MLP to one logit."""

    def __init__(self, num_states: int, dim: int) -> None:
        super().__init__()
        self.layer_weights = nn.Parameter(torch.zeros(num_states))
        self.pool = AttentiveStatsPool(dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, HEAD_HIDDEN),
            nn.ReLU(),
            nn.Dropout(HEAD_DROPOUT),
            nn.Linear(HEAD_HIDDEN, 1),
        )

    def forward(self, states: list[Tensor]) -> Tensor:
        """The logit of each window, `[batch]`."""
        if len(states) != self.layer_weights.numel():
            raise ModelError(f"the head takes {self.layer_weights.numel()} states")
        weights = torch.softmax(self.layer_weights.float(), dim=0).view(-1, 1, 1, 1)
        mixed = (weights * torch.stack(states).float()).sum(dim=0)
        return self.mlp(self.pool(mixed)).squeeze(-1)


class VerifierModel(nn.Module):
    """The encoder and the head; the encoder's frozen part stays in eval mode under `train()`."""

    def __init__(self, encoder: StateEncoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = VerifierHead(encoder.num_states, encoder.dim)
        trainable = {id(p) for m in encoder.trainable_modules() for p in m.parameters()}
        for parameter in encoder.parameters():
            parameter.requires_grad_(id(parameter) in trainable)
        self.train(False)

    def train(self, mode: bool = True) -> "VerifierModel":
        """Set the mode of the head and the trainable layers; the frozen part stays in eval."""
        super().train(mode)
        self.encoder.eval()
        self.encoder.training = mode
        for module in self.encoder.trainable_modules():
            module.train(mode)
        return self

    def frozen_modules(self) -> list[nn.Module]:
        """Every encoder submodule outside the trainable ones, the encoder itself excluded."""
        inside = {id(sub) for m in self.encoder.trainable_modules() for sub in m.modules()}
        return [
            module
            for module in self.encoder.modules()
            if id(module) not in inside and module is not self.encoder
        ]

    def parameter_groups(self, *, layers_lr: float, head_lr: float) -> list[dict[str, object]]:
        """AdamW's two groups: the trainable layers at `layers_lr`, the head at `head_lr`."""
        layers = [p for m in self.encoder.trainable_modules() for p in m.parameters()]
        return [
            {"params": layers, "lr": layers_lr},
            {"params": list(self.head.parameters()), "lr": head_lr},
        ]

    def forward(self, waveform: Tensor) -> Tensor:
        """Logits `[batch]` for waveforms `[batch, samples]` of one length."""
        return self.head(self.encoder.hidden_states(waveform))


class ScoreGraph(nn.Module):
    """The exported graph: `waveform` float32 `[1, n]` to `score` float32 `[1]`, a sigmoid."""

    def __init__(self, model: VerifierModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, waveform: Tensor) -> Tensor:
        """The window's score in [0, 1]."""
        return torch.sigmoid(self.model(waveform))


def build_verifier(
    params: Mapping[str, object],
    *,
    weights: Path | None = None,
    first_trainable: int = FIRST_TRAINABLE_LAYER,
) -> VerifierModel:
    """The verifier over WavLM: loaded strictly from `weights`, or random-initialised for a load.

    Raises:
        ModelError: The weights are refused (`load_wavlm`) or the layer is out of range.
    """
    wavlm = build_wavlm(params) if weights is None else load_wavlm(weights, params)
    return VerifierModel(WavLMEncoder(wavlm, first_trainable))
