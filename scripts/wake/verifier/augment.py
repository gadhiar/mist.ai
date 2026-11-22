"""Augment the verifier's training batches on the device the model trains on.

`train.py` augmented each window in its DataLoader workers in numpy, and the room response's
`np.convolve` against 8,000 to 32,000 taps held the workers long enough that the GPU sat idle.
Here a worker only makes the random choices: `draw` takes an item's choices from its seed on
the CPU, and `batch_draws` turns a batch's choices into tensors, the low-pass's Butterworth
sections included. `augment_batch` then applies them to the whole batch on the device.

It keeps the old path's steps and their order (gain, room response, low-pass, noise, AGC,
clip), each step's probability and range, and its float32 output; no step reads a label. What
differs:

- The room response and the low-pass are one multiplication in the frequency domain instead of
  `np.convolve` and `sosfilt`. The low-pass is the sixth-order Butterworth's frequency response
  evaluated from the same sections, so the product is a circular convolution with the room's
  taps and the low-pass's whole impulse response. The room-convolved signal is 2L - 1 samples
  long, and the FFT is longer than that by `lowpass_tail` samples, the point past which the
  impulse response's absolute sum is under `TAIL_BOUND`, at the cutoff in the settings' range
  that rings longest. What wraps into a kept sample is therefore under `TAIL_BOUND` times the
  room-convolved signal's peak.
- The work is float32 on the device, where the old path was float64.
- `draw` makes every draw whether or not its step applies, in a fixed order, and the noise
  comes from a torch generator seeded per item. A seed therefore gives other values than
  `train.augment` gave it; each step's distribution is the old one.
"""

import functools
import math
from collections.abc import Sequence
from dataclasses import dataclass, fields

import numpy as np
import torch
from scipy.fft import next_fast_len
from scipy.signal import butter, sosfilt

from .dataset import FRAME_SAMPLES, SAMPLE_RATE

LOWPASS_ORDER = 6
LOWPASS_SECTIONS = LOWPASS_ORDER // 2
CLIP_LOW = -1.0
CLIP_HIGH = 32767 / 32768
_IDENTITY_SECTION = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
TAIL_BOUND = 2.0**-24
"""The low-pass's impulse response's absolute sum left past `lowpass_tail`: float32's unit
roundoff, so what wraps into a kept sample is below float32 resolution of the signal's peak."""
TAIL_CUTOFFS = 65
"""Cutoffs `lowpass_tail` evaluates across the settings' range, both ends included."""


@dataclass(frozen=True, slots=True)
class Augment:
    """The augmentation's probabilities and ranges, the same for both classes."""

    truncate_p: float = 0.5
    gain_db: tuple[float, float] = (-6.0, 6.0)
    noise_p: float = 0.5
    noise_snr_db: tuple[float, float] = (10.0, 40.0)
    rir_p: float = 0.25
    lowpass_p: float = 0.3
    lowpass_hz: tuple[float, float] = (3000.0, 7500.0)
    agc_p: float = 0.3
    agc_target_rms: float = 0.1
    agc_max_gain_db: float = 20.0


@dataclass(frozen=True, slots=True)
class Draws:
    """One item's choices; `None` (or False) where its step does not apply.

    `rir` indexes the run's room responses; the caller puts that response in the item's row of
    `augment_batch`'s `responses`.
    """

    gain_db: float
    rir: int | None
    cutoff_hz: float | None
    snr_db: float | None
    agc: bool
    noise_seed: int


def draw(seed: int, settings: Augment, rirs: int) -> Draws:
    """An item's choices from its seed: a fixed number of draws in a fixed order.

    Every draw is made whether or not its step applies, so changing one step's probability
    leaves the values every other step draws from the same seed unchanged.

    Args:
        seed: The item's augmentation seed.
        settings: The probabilities and ranges.
        rirs: How many room responses the run has; with none, no item gets one.
    """
    rng = np.random.default_rng(seed)
    gain_db = float(rng.uniform(*settings.gain_db))
    room = bool(rng.random() < settings.rir_p) and rirs > 0
    room_index = int(rng.integers(max(rirs, 1)))
    lowpass = bool(rng.random() < settings.lowpass_p)
    cutoff = float(rng.uniform(*settings.lowpass_hz))
    noise = bool(rng.random() < settings.noise_p)
    snr = float(rng.uniform(*settings.noise_snr_db))
    agc = bool(rng.random() < settings.agc_p)
    noise_seed = int(rng.integers(2**63))
    return Draws(
        gain_db=gain_db,
        rir=room_index if room else None,
        cutoff_hz=cutoff if lowpass else None,
        snr_db=snr if noise else None,
        agc=agc,
        noise_seed=noise_seed,
    )


@dataclass(frozen=True, slots=True)
class BatchDraws:
    """A batch's choices as tensors, one row per item.

    `noise_seeds` stays on the host through `to`: each seeds a generator from Python, and
    reading it back from the device would wait for the device every batch.

    Attributes:
        gain: [B] float32, linear.
        room: [B] bool, the item's row of `responses` is applied.
        lowpass: [B] bool.
        noise: [B] bool.
        agc: [B] bool.
        snr_db: [B] float32, zero where `noise` is False.
        noise_seeds: [B] int64.
        lowpass_sos: [B, 3, 6] float64 second-order sections, zero where `lowpass` is False.
    """

    gain: torch.Tensor
    room: torch.Tensor
    lowpass: torch.Tensor
    noise: torch.Tensor
    agc: torch.Tensor
    snr_db: torch.Tensor
    noise_seeds: torch.Tensor
    lowpass_sos: torch.Tensor

    def to(self, device: torch.device | str, non_blocking: bool = False) -> "BatchDraws":
        """The same choices on `device`, `noise_seeds` left on the host."""
        return BatchDraws(
            **{
                item.name: (
                    getattr(self, item.name)
                    if item.name == "noise_seeds"
                    else getattr(self, item.name).to(device, non_blocking=non_blocking)
                )
                for item in fields(self)
            }
        )

    def pin_memory(self) -> "BatchDraws":
        """The same choices in pinned memory; DataLoader's `pin_memory` calls this."""
        return BatchDraws(
            **{item.name: getattr(self, item.name).pin_memory() for item in fields(self)}
        )


def batch_draws(draws: Sequence[Draws]) -> BatchDraws:
    """A batch's choices as tensors, made in a loader worker on the CPU.

    The low-pass's sections are designed here with `scipy.signal.butter`, as the old path did
    per item, so the device only evaluates them.
    """
    sos = np.zeros((len(draws), LOWPASS_SECTIONS, 6))
    for row, item in enumerate(draws):
        if item.cutoff_hz is not None:
            sos[row] = butter(LOWPASS_ORDER, item.cutoff_hz, fs=SAMPLE_RATE, output="sos")
    return BatchDraws(
        gain=torch.tensor([10 ** (item.gain_db / 20) for item in draws], dtype=torch.float32),
        room=torch.tensor([item.rir is not None for item in draws], dtype=torch.bool),
        lowpass=torch.tensor([item.cutoff_hz is not None for item in draws], dtype=torch.bool),
        noise=torch.tensor([item.snr_db is not None for item in draws], dtype=torch.bool),
        agc=torch.tensor([item.agc for item in draws], dtype=torch.bool),
        snr_db=torch.tensor(
            [item.snr_db if item.snr_db is not None else 0.0 for item in draws],
            dtype=torch.float32,
        ),
        noise_seeds=torch.tensor([item.noise_seed for item in draws], dtype=torch.int64),
        lowpass_sos=torch.from_numpy(sos),
    )


def augment_batch(
    audio: torch.Tensor, draws: BatchDraws, responses: torch.Tensor, settings: Augment
) -> torch.Tensor:
    """Gain, a room response, a band limit, noise and AGC on the batch, then the clip.

    Args:
        audio: [B, L] int16 on the device, one kept length a batch, `L` whole frames.
        draws: The batch's choices on the same device.
        responses: [B, R] float32 on the device, row i item i's peak-normalised room response
            (any row zero where the item has none); only the first `L` taps can reach the
            samples kept, so the rest are ignored.
        settings: The AGC's target and gain limit.

    Returns:
        [B, L] float32 in [-1, 32767/32768].

    Raises:
        ValueError: The shapes do not fit, or `L` is not whole frames.
    """
    if audio.dim() != 2 or audio.dtype != torch.int16:
        raise ValueError(f"audio must be [B, L] int16, not {tuple(audio.shape)} {audio.dtype}")
    batch, length = audio.shape
    if length == 0 or length % FRAME_SAMPLES:
        raise ValueError(f"a window of {length} samples is not whole {FRAME_SAMPLES}-sample frames")
    if responses.dim() != 2 or responses.shape[0] != batch:
        raise ValueError(f"responses must be [{batch}, R], not {tuple(responses.shape)}")
    out = audio.to(torch.float32) / 32768.0
    out = out * draws.gain[:, None]
    out = _filter(out, draws, responses, lowpass_tail(*settings.lowpass_hz))
    out = _add_noise(out, draws)
    out = _agc(out, draws.agc, settings)
    return out.clamp(CLIP_LOW, CLIP_HIGH)


@functools.lru_cache(maxsize=8)
def lowpass_tail(low_hz: float, high_hz: float) -> int:
    """Samples after which the low-pass's impulse response sums under `TAIL_BOUND` in absolute
    value, at whichever of `TAIL_CUTOFFS` cutoffs across `[low_hz, high_hz]` rings longest.

    The ringing is not longest at the lowest cutoff: at 16 kHz the largest pole radius is
    0.78 at 3 kHz and 0.95 at 7.5 kHz, and the tail 72 samples at 3 kHz and 345 at 7.5 kHz.
    The impulse response is computed long enough that the tail ends in its first half.
    """
    longest = 0
    for cutoff in np.linspace(low_hz, high_hz, TAIL_CUTOFFS):
        sos = butter(LOWPASS_ORDER, cutoff, fs=SAMPLE_RATE, output="sos")
        count = 1024
        while True:
            impulse = np.zeros(count)
            impulse[0] = 1.0
            remaining = np.cumsum(np.abs(sosfilt(sos, impulse))[::-1])[::-1]
            below = np.flatnonzero(remaining < TAIL_BOUND)
            if len(below) and below[0] < count // 2:
                longest = max(longest, int(below[0]))
                break
            count *= 2
    return longest


def _filter(
    out: torch.Tensor, draws: BatchDraws, responses: torch.Tensor, tail: int
) -> torch.Tensor:
    """The room response and the low-pass as one product of spectra.

    The transform holds the room-convolved signal, 2L - 1 samples, and `tail` more, so the
    low-pass's ringing past them is all that wraps into the samples kept.
    """
    length = out.shape[1]
    size = next_fast_len(2 * length + tail, real=True)
    taps = responses[:, :length].to(torch.float32)
    taps = torch.nn.functional.pad(taps, (0, size - taps.shape[1]))
    room = torch.fft.rfft(taps)
    response = torch.where(draws.room[:, None], room, torch.ones_like(room))
    response = response * _lowpass_response(draws.lowpass_sos, draws.lowpass, size)
    filtered = torch.fft.irfft(torch.fft.rfft(out, n=size) * response, n=size)[:, :length]
    return torch.where((draws.room | draws.lowpass)[:, None], filtered, out)


def _lowpass_response(sos: torch.Tensor, applies: torch.Tensor, size: int) -> torch.Tensor:
    """The sections' frequency response at a `size`-point rfft's bins; 1 where none applies.

    Evaluated in float64, as `sosfilt` ran, and cast once at the end.
    """
    identity = torch.tensor(_IDENTITY_SECTION, dtype=torch.float64, device=sos.device)
    sos = torch.where(applies[:, None, None], sos.to(torch.float64), identity)
    angle = torch.arange(size // 2 + 1, dtype=torch.float64, device=sos.device) * (
        2 * math.pi / size
    )
    z1 = torch.polar(torch.ones_like(angle), -angle)
    z2 = z1 * z1
    response = torch.ones(sos.shape[0], angle.shape[0], dtype=torch.complex128, device=sos.device)
    for section in range(sos.shape[1]):
        b0, b1, b2, a0, a1, a2 = (sos[:, section, k, None] for k in range(6))
        response = response * (b0 + b1 * z1 + b2 * z2) / (a0 + a1 * z1 + a2 * z2)
    return response.to(torch.complex64)


def _add_noise(out: torch.Tensor, draws: BatchDraws) -> torch.Tensor:
    """White noise at each noisy item's SNR against its own mean power.

    Every row draws its normals from a generator seeded with its own seed, so an item's noise
    does not depend on its batch; rows without noise are scaled by zero, which leaves them
    unchanged without reading the flags back from the device.
    """
    normals = torch.empty_like(out)
    for row, seed in enumerate(draws.noise_seeds.tolist()):
        generator = torch.Generator(device=out.device)
        generator.manual_seed(seed)
        normals[row].normal_(generator=generator)
    power = out.square().mean(dim=1) + 1e-10
    std = torch.sqrt(power / 10 ** (draws.snr_db / 10))
    std = torch.where(draws.noise, std, torch.zeros_like(std))
    return out + std[:, None] * normals


def _agc(out: torch.Tensor, applies: torch.Tensor, settings: Augment) -> torch.Tensor:
    """`train._agc` on whole frames: a gain per frame from a peak-holding level envelope."""
    batch, length = out.shape
    frames = out.reshape(batch, length // FRAME_SAMPLES, FRAME_SAMPLES)
    level = torch.sqrt(frames.square().mean(dim=2) + 1e-10)
    envelope = torch.cummax(level, dim=1).values * 0.5 + level * 0.5
    gain = torch.clamp(
        settings.agc_target_rms / envelope.clamp_min(1e-6),
        max=10 ** (settings.agc_max_gain_db / 20),
    )
    gain = torch.where(applies[:, None], gain, torch.ones_like(gain))
    return (frames * gain[:, :, None]).reshape(batch, length)
