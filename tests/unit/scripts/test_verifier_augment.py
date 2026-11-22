"""Tests for the verifier's batched augmentation against the CPU path it replaces.

The oracle is `augment` and `_agc` as `scripts/wake/verifier/train.py` had them at commit
dc98960c, copied verbatim below because the trainer drops them once it augments on the device.
With noise off, a scripted generator feeds the oracle the same choices the batched path gets,
so the two must agree sample for sample; with noise on, the two draw from different streams,
so they are compared as distributions over a thousand fixed seeds. Every input is synthetic:
a tone under noise as int16, room responses as decaying noise. Everything runs on the CPU.
"""

import inspect
import math
from collections.abc import Sequence

import numpy as np
import pytest
import torch
from scipy.signal import butter
from scipy.stats import ks_2samp

from scripts.wake.verifier.augment import (
    Augment,
    BatchDraws,
    Draws,
    augment_batch,
    batch_draws,
    draw,
)
from scripts.wake.verifier.dataset import FRAME_SAMPLES, SAMPLE_RATE

CLIP_HIGH = 32767 / 32768
PARITY = 1e-4

# -- The oracle: train.py at dc98960c, lines 327-359, verbatim --------------------------------


def _agc(samples: np.ndarray, target: float, max_gain_db: float) -> np.ndarray:
    frames = max(1, len(samples) // FRAME_SAMPLES)
    level = np.sqrt(
        np.mean(samples[: frames * FRAME_SAMPLES].reshape(frames, -1) ** 2, axis=1) + 1e-10
    )
    envelope = np.maximum.accumulate(level) * 0.5 + level * 0.5
    gain = np.minimum(target / np.maximum(envelope, 1e-6), 10 ** (max_gain_db / 20))
    per_sample = np.repeat(gain, FRAME_SAMPLES)[: len(samples)]
    per_sample = np.pad(per_sample, (0, len(samples) - len(per_sample)), mode="edge")
    return samples * per_sample


def augment(
    samples: np.ndarray, rng: np.random.Generator, settings: Augment, rirs: Sequence[np.ndarray]
) -> np.ndarray:
    """Gain, noise, a room response, a band limit and AGC, each drawn from `rng`; no label read."""
    out = samples.astype(np.float64)
    out = out * 10 ** (rng.uniform(*settings.gain_db) / 20)
    if rirs and rng.random() < settings.rir_p:
        response = rirs[int(rng.integers(len(rirs)))]
        out = np.convolve(out, response)[: len(out)]
    if rng.random() < settings.lowpass_p:
        from scipy.signal import butter, sosfilt  # noqa: PLC0415 - training only

        cutoff = rng.uniform(*settings.lowpass_hz)
        out = sosfilt(butter(6, cutoff, fs=SAMPLE_RATE, output="sos"), out)
    if rng.random() < settings.noise_p:
        power = float(np.mean(out**2)) + 1e-10
        snr = rng.uniform(*settings.noise_snr_db)
        out = out + rng.normal(0.0, math.sqrt(power / 10 ** (snr / 10)), len(out))
    if rng.random() < settings.agc_p:
        out = _agc(out, settings.agc_target_rms, settings.agc_max_gain_db)
    return np.clip(out, -1.0, 32767 / 32768).astype(np.float32)


# -- Synthetic inputs -------------------------------------------------------------------------


def speech_like(
    frames: int, seed: int, level: float = 1500.0, silent_frames: int = 0
) -> np.ndarray:
    """A 440 Hz tone under white noise with a slow swell, so AGC sees a moving level."""
    rng = np.random.default_rng(seed)
    count = frames * FRAME_SAMPLES
    t = np.arange(count) / SAMPLE_RATE
    carrier = rng.normal(0.0, 1.0, count) + 2.0 * np.sin(2 * np.pi * 440.0 * t)
    swell = 0.2 + 0.8 * (0.5 + 0.5 * np.sin(2 * np.pi * 1.5 * t)) ** 2
    samples = np.round(level * carrier * swell / 2.0)
    if silent_frames:
        samples[-silent_frames * FRAME_SAMPLES :] = 0.0
    return np.clip(samples, -32768, 32767).astype(np.int16)


def room(taps: int, seed: int) -> np.ndarray:
    """A direct path and a decaying noise tail, peak-normalised as `WindowReader.rir` returns."""
    rng = np.random.default_rng(seed)
    tail = 0.1 * rng.normal(0.0, 1.0, taps) * np.exp(-np.arange(taps) / (taps / 8))
    tail[0] = 1.0
    return (tail / np.max(np.abs(tail))).astype(np.float32)


class Scripted:
    """A stand-in for `np.random.Generator` that returns the given values in order."""

    def __init__(self, uniform: list[float], random: list[float], integers: list[int]) -> None:
        self._uniform = list(uniform)
        self._random = list(random)
        self._integers = list(integers)

    def uniform(self, low: float, high: float) -> float:
        return self._uniform.pop(0)

    def random(self) -> float:
        return self._random.pop(0)

    def integers(self, high: int) -> int:
        return self._integers.pop(0)

    def normal(self, *args: object) -> np.ndarray:
        raise AssertionError("noise is off in the parity cases")

    @property
    def spent(self) -> bool:
        return not (self._uniform or self._random or self._integers)


def oracle_fixed(
    samples: np.ndarray,
    gain_db: float,
    response: np.ndarray | None,
    cutoff: float | None,
    agc: bool,
    settings: Augment,
) -> np.ndarray:
    """The oracle with its choices fixed and noise off."""
    rng = Scripted(
        uniform=[gain_db] + ([cutoff] if cutoff is not None else []),
        random=([0.0] if response is not None else [])
        + [0.0 if cutoff is not None else 1.0, 1.0, 0.0 if agc else 1.0],
        integers=[0] if response is not None else [],
    )
    audio = samples.astype(np.float32) / np.float32(32768.0)
    rirs = [response] if response is not None else []
    out = augment(audio, rng, settings, rirs)  # type: ignore[arg-type]
    assert rng.spent
    return out


# -- Deterministic parity ---------------------------------------------------------------------


@pytest.mark.parametrize("frames", [40, 116])
def test_matches_the_oracle_step_by_step_with_noise_off(frames: int) -> None:
    length = frames * FRAME_SAMPLES
    settings = Augment()
    # (name, gain dB, room taps, low-pass cutoff, AGC)
    cases = [
        ("gain up", 4.5, None, None, False),
        ("gain down", -5.0, None, None, False),
        ("room 400", 0.0, 400, None, False),
        ("room 8000", 0.0, 8000, None, False),
        ("room 32000", 0.0, 32000, None, False),
        ("room longer than L", 0.0, length + 3200, None, False),
        ("low-pass 3000", 0.0, None, 3000.0, False),
        ("low-pass 7500", 0.0, None, 7500.0, False),
        ("agc", 0.0, None, None, True),
        ("all four", 3.0, 8000, 3000.0, True),
        ("nothing", 0.0, None, None, False),
    ]
    inputs = [speech_like(frames, seed=row) for row in range(len(cases))]
    responses = [
        room(taps, seed=100 + row) if taps else None for row, (_, _, taps, _, _) in enumerate(cases)
    ]
    width = max(len(r) for r in responses if r is not None)
    bank = np.zeros((len(cases), width), dtype=np.float32)
    for row, response in enumerate(responses):
        if response is not None:
            bank[row, : len(response)] = response
    draws = [
        Draws(
            gain_db=gain,
            rir=0 if taps else None,
            cutoff_hz=cutoff,
            snr_db=None,
            agc=agc,
            noise_seed=row,
        )
        for row, (_, gain, taps, cutoff, agc) in enumerate(cases)
    ]
    out = augment_batch(
        torch.from_numpy(np.stack(inputs)),
        batch_draws(draws).to("cpu"),
        torch.from_numpy(bank),
        settings,
    )
    assert out.dtype == torch.float32
    assert out.shape == (len(cases), length)
    for row, (name, gain, _, cutoff, agc) in enumerate(cases):
        expected = oracle_fixed(inputs[row], gain, responses[row], cutoff, agc, settings)
        error = float(np.max(np.abs(out[row].numpy() - expected)))
        assert error < PARITY, f"{name}: max abs {error:.3g}"


def test_a_batch_without_responses_still_runs() -> None:
    samples = speech_like(40, seed=1)
    draws = [Draws(2.0, None, 5000.0, None, False, 3)]
    out = augment_batch(
        torch.from_numpy(samples[None]), batch_draws(draws), torch.zeros(1, 0), Augment()
    )
    expected = oracle_fixed(samples, 2.0, None, 5000.0, False, Augment())
    assert float(np.max(np.abs(out[0].numpy() - expected))) < PARITY


def test_refuses_a_length_that_is_not_whole_frames() -> None:
    draws = batch_draws([Draws(0.0, None, None, None, False, 0)])
    with pytest.raises(ValueError, match="frame"):
        augment_batch(torch.zeros(1, 1000, dtype=torch.int16), draws, torch.zeros(1, 0), Augment())


# -- The draws --------------------------------------------------------------------------------

SEEDS = range(1000)


def test_each_step_applies_at_its_probability() -> None:
    settings = Augment()
    draws = [draw(seed, settings, 4) for seed in SEEDS]
    count = len(draws)
    for name, p, hits in [
        ("room", settings.rir_p, sum(d.rir is not None for d in draws)),
        ("low-pass", settings.lowpass_p, sum(d.cutoff_hz is not None for d in draws)),
        ("noise", settings.noise_p, sum(d.snr_db is not None for d in draws)),
        ("agc", settings.agc_p, sum(d.agc for d in draws)),
    ]:
        error = math.sqrt(p * (1 - p) / count)
        assert abs(hits / count - p) <= 3 * error, f"{name}: {hits}/{count} against {p}"
    assert all(settings.gain_db[0] <= d.gain_db <= settings.gain_db[1] for d in draws)
    assert all(d.rir in range(4) for d in draws if d.rir is not None)
    assert all(
        settings.lowpass_hz[0] <= d.cutoff_hz <= settings.lowpass_hz[1]
        for d in draws
        if d.cutoff_hz is not None
    )
    assert all(
        settings.noise_snr_db[0] <= d.snr_db <= settings.noise_snr_db[1]
        for d in draws
        if d.snr_db is not None
    )


def test_every_draw_is_made_whether_or_not_its_step_applies() -> None:
    never = Augment(rir_p=0.0, lowpass_p=0.0, noise_p=0.0, agc_p=0.0)
    always = Augment(rir_p=1.0, lowpass_p=1.0, noise_p=1.0, agc_p=1.0)
    for seed in range(50):
        off, on = draw(seed, never, 3), draw(seed, always, 3)
        assert (off.rir, off.cutoff_hz, off.snr_db, off.agc) == (None, None, None, False)
        assert on.rir is not None and on.cutoff_hz is not None and on.snr_db is not None
        assert on.agc
        assert off.gain_db == on.gain_db
        assert off.noise_seed == on.noise_seed
        assert draw(seed, always, 0).rir is None


def test_batch_draws_shapes_and_sections() -> None:
    draws = [
        Draws(6.0, 1, 3000.0, 20.0, True, 2**63 - 1),
        Draws(-6.0, None, None, None, False, 0),
    ]
    batch = batch_draws(draws)
    assert batch.gain.dtype == torch.float32
    assert torch.allclose(batch.gain, torch.tensor([10 ** (6 / 20), 10 ** (-6 / 20)]))
    assert batch.room.tolist() == [True, False]
    assert batch.lowpass.tolist() == [True, False]
    assert batch.noise.tolist() == [True, False]
    assert batch.agc.tolist() == [True, False]
    assert batch.room.dtype == torch.bool
    assert batch.snr_db.dtype == torch.float32
    assert batch.snr_db[0].item() == 20.0
    assert batch.noise_seeds.dtype == torch.int64
    assert batch.noise_seeds.tolist() == [2**63 - 1, 0]
    assert batch.lowpass_sos.shape == (2, 3, 6)
    assert batch.lowpass_sos.dtype == torch.float64
    np.testing.assert_array_equal(
        batch.lowpass_sos[0].numpy(), butter(6, 3000.0, fs=SAMPLE_RATE, output="sos")
    )
    assert not batch.lowpass_sos[1].any()
    moved = batch.to("cpu", non_blocking=True)
    assert isinstance(moved, BatchDraws)
    assert moved.noise_seeds.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinning needs an accelerator")
def test_pin_memory_pins_every_tensor() -> None:
    pinned = batch_draws([Draws(0.0, None, None, None, False, 0)]).pin_memory()
    assert pinned.gain.is_pinned() and pinned.lowpass_sos.is_pinned()


def test_no_step_takes_a_label() -> None:
    for function in (draw, batch_draws, augment_batch):
        assert "label" not in inspect.signature(function).parameters


# -- Seeded runs against the oracle -----------------------------------------------------------


def batched(
    samples: np.ndarray, seeds: range, settings: Augment, bank: list[np.ndarray], size: int = 200
) -> np.ndarray:
    """`draw`, `batch_draws` and `augment_batch` over `seeds`, `size` items a batch."""
    width = max(len(response) for response in bank)
    outputs = []
    for start in range(0, len(seeds), size):
        draws = [draw(seed, settings, len(bank)) for seed in seeds[start : start + size]]
        responses = np.zeros((len(draws), width), dtype=np.float32)
        for row, item in enumerate(draws):
            if item.rir is not None:
                responses[row, : len(bank[item.rir])] = bank[item.rir]
        audio = torch.from_numpy(np.repeat(samples[None], len(draws), axis=0))
        out = augment_batch(audio, batch_draws(draws), torch.from_numpy(responses), settings)
        outputs.append(out.numpy())
    return np.concatenate(outputs)


def statistics(outputs: np.ndarray) -> dict[str, np.ndarray]:
    """Per-output RMS, share of energy above 3 kHz and the level of the silent tail.

    Rounded, so a value several items share (gain moves no share, silence stays at the floor)
    is one value on both sides, not two that differ by rounding error.
    """
    power = np.mean(outputs.astype(np.float64) ** 2, axis=1)
    spectrum = np.abs(np.fft.rfft(outputs.astype(np.float64), axis=1)) ** 2
    bins = np.fft.rfftfreq(outputs.shape[1], 1 / SAMPLE_RATE)
    high = spectrum[:, bins > 3000.0].sum(axis=1) / spectrum.sum(axis=1)
    tail = np.mean(outputs[:, -5 * FRAME_SAMPLES :].astype(np.float64) ** 2, axis=1)
    return {
        "rms dB": np.round(10 * np.log10(power + 1e-12), 2),
        "share above 3 kHz": np.round(high, 4),
        "noise floor dB": np.round(10 * np.log10(tail + 1e-9), 2),
    }


def test_agrees_with_the_oracle_in_distribution() -> None:
    settings = Augment()
    samples = speech_like(40, seed=7, silent_frames=10)
    bank = [room(400, seed) for seed in range(4)]
    audio = samples.astype(np.float32) / np.float32(32768.0)
    old = np.stack([augment(audio, np.random.default_rng(seed), settings, bank) for seed in SEEDS])
    new = batched(samples, SEEDS, settings, bank)
    old_stats, new_stats = statistics(old), statistics(new)
    for name in old_stats:
        result = ks_2samp(old_stats[name], new_stats[name])
        assert result.pvalue > 0.01, f"{name}: KS {result.statistic:.3f}, p {result.pvalue:.3g}"


def test_the_same_seeds_give_the_same_output_within_the_clip_range() -> None:
    settings = Augment(gain_db=(10.0, 12.0), noise_p=1.0)
    samples = speech_like(40, seed=3, level=8000.0)
    bank = [room(400, 0)]
    first = batched(samples, range(32), settings, bank, size=16)
    second = batched(samples, range(32), settings, bank, size=16)
    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.float32
    assert first.min() >= -1.0 and first.max() <= CLIP_HIGH
    assert first.max() == np.float32(CLIP_HIGH)
    assert not np.array_equal(first[0], first[1])
