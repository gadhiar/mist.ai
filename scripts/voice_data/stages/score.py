"""Speaker-similarity scoring, state floors, the cleaning spectrum check and take selection.

"SIM" is always one clip's cosine to a target centroid. A gate compares it with the
leave-one-out distribution of that same quantity: each real clip against the centroid of the
others. Pairwise clip-to-clip cosines run lower and would let a near voice through (Design 4).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..records import Prosody

THIRD_OCTAVE_CENTRES_HZ = 1000.0 * 2.0 ** (np.arange(-10, 10) / 3.0)
"""The 20 third-octave band centres whose nominal values run from 100 Hz to 8 kHz."""

VOICED_PERIODICITY_MIN = 0.5
VOICED_RANGE_DB = 40.0
SPECTRUM_RANGE_DB = 30.0

StateProfile = Prosody
"""A state's prosody profile (medians) or spread (standard deviations), as in `target.json`."""


@dataclass(frozen=True, slots=True)
class Distribution:
    """A SIM distribution's size, median and the percentiles the gates use."""

    n: int
    median: float
    p02: float
    p10: float


@dataclass(frozen=True, slots=True)
class StateFloor:
    """A state's SIM floor on one embedder and where it came from.

    `floor_from` is `state` (its own p10), `thin` (5-19 segments: the lower of its own and the
    pooled p10) or `pooled-offset` (under 5: the pooled p10 minus the state's offset).
    """

    floor: float
    floor_from: str
    n: int

    @property
    def flagged(self) -> bool:
        """Whether the state is thin, so its floor does not rest on its own segments alone."""
        return self.floor_from != "state"


@dataclass(frozen=True, slots=True)
class TakeScores:
    """A generated take's gate inputs: SIM per embedder, CER and prosody."""

    take_id: str
    sim: dict[str, float]
    cer: float
    prosody: Prosody


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Return the cosine of two vectors."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def centroid(embeddings: Sequence[np.ndarray]) -> np.ndarray:
    """Return the L2-normalised mean of the L2-normalised embeddings.

    Each clip is normalised first, so a clip's weight does not depend on its embedding's norm.
    """
    stacked = np.asarray(embeddings, dtype=np.float64)
    stacked = stacked / np.linalg.norm(stacked, axis=1, keepdims=True)
    mean = stacked.mean(axis=0)
    return mean / np.linalg.norm(mean)


def distribution(scores: Sequence[float]) -> Distribution:
    """Return the size, median, 2nd and 10th percentiles of `scores` (linear interpolation).

    Raises:
        ValueError: `scores` is empty.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        raise ValueError("a distribution needs at least one score")
    return Distribution(
        n=int(values.size),
        median=float(np.median(values)),
        p02=float(np.percentile(values, 2)),
        p10=float(np.percentile(values, 10)),
    )


def leave_one_out_scores(embeddings: Sequence[np.ndarray], sources: Sequence[str]) -> np.ndarray:
    """Return each clip's SIM to the centroid of the others.

    "The others" are the clips from other sources when there are any, so clips of one episode
    do not vouch for each other through a shared channel; with a single source, every other clip.

    Raises:
        ValueError: Fewer than two clips, or `sources` is not one per clip.
    """
    if len(embeddings) < 2:
        raise ValueError("leave-one-out needs at least two clips")
    if len(sources) != len(embeddings):
        raise ValueError("leave-one-out needs one source per clip")
    scores = np.empty(len(embeddings))
    for i, clip in enumerate(embeddings):
        others = [e for j, e in enumerate(embeddings) if sources[j] != sources[i]]
        if not others:
            others = [e for j, e in enumerate(embeddings) if j != i]
        scores[i] = cosine(clip, centroid(others))
    return scores


def leave_one_out(embeddings: Sequence[np.ndarray], sources: Sequence[str]) -> Distribution:
    """Return the distribution of `leave_one_out_scores`.

    Raises:
        ValueError: Fewer than two clips, or `sources` is not one per clip.
    """
    return distribution(leave_one_out_scores(embeddings, sources).tolist())


def state_floors(
    scores: Mapping[str, Sequence[float]],
    pooled: Distribution,
    offsets: Mapping[str, float],
    min_n: int = 20,
    thin_n: int = 5,
) -> dict[str, StateFloor]:
    """Return each state's floor on one embedder from its segments' leave-one-out SIMs.

    A state with `min_n` or more segments takes its own p10. A thin one is never held to a floor
    stricter than its segments support: from `thin_n` it takes the lower of its own and the
    pooled p10, and below `thin_n` the pooled p10 minus its offset (0 when `offsets` has none,
    the default H6 starts from).
    """
    floors: dict[str, StateFloor] = {}
    for state, values in scores.items():
        n = len(values)
        if n >= min_n:
            floors[state] = StateFloor(float(np.percentile(values, 10)), "state", n)
        elif n >= thin_n:
            own = float(np.percentile(values, 10))
            floors[state] = StateFloor(min(own, pooled.p10), "thin", n)
        else:
            floors[state] = StateFloor(pooled.p10 - offsets.get(state, 0.0), "pooled-offset", n)
    return floors


def _frames(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    # 40 ms rounded up to a power of two: 16-24 Hz per bin at 16, 24, 44.1 and 48 kHz. Short
    # frames keep each harmonic's main lobe wide enough to reach the third-octave bands between
    # a low voice's harmonics, so those bands hold voice rather than only the noise floor.
    size = int(2 ** np.ceil(np.log2(0.04 * sample_rate)))
    signal = np.asarray(audio, dtype=np.float64)
    if len(signal) < size:
        signal = np.pad(signal, (0, size - len(signal)))
    return np.lib.stride_tricks.sliding_window_view(signal, size)[:: size // 2]


def _voiced(frames: np.ndarray, sample_rate: int) -> np.ndarray:
    # Voiced: within VOICED_RANGE_DB of the loudest frame, and periodic, with a normalised
    # autocorrelation peak of at least VOICED_PERIODICITY_MIN at a lag between 2.5 and 16.7 ms
    # (60-400 Hz). Noise and silence fail the second test.
    power = np.mean(frames**2, axis=1)
    if power.max() <= 0.0:
        return np.zeros(len(frames), dtype=bool)
    loud = power >= power.max() * 10.0 ** (-VOICED_RANGE_DB / 10.0)
    size = frames.shape[1]
    centred = frames - frames.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(centred, n=2 * size, axis=1)
    autocorr = np.fft.irfft(np.abs(spectrum) ** 2, axis=1)[:, :size]
    lo, hi = int(sample_rate / 400.0), min(int(sample_rate / 60.0), size - 1)
    energy = autocorr[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        peak = np.where(energy > 0, autocorr[:, lo : hi + 1].max(axis=1) / energy, 0.0)
    return loud & (peak >= VOICED_PERIODICITY_MIN)


def _third_octave_levels_db(audio: np.ndarray, sample_rate: int) -> np.ndarray | None:
    frames = _frames(audio, sample_rate)
    voiced = _voiced(frames, sample_rate)
    if not voiced.any():
        return None
    window = np.hanning(frames.shape[1])
    power = np.mean(np.abs(np.fft.rfft(frames[voiced] * window, axis=1)) ** 2, axis=0)
    freqs = np.fft.rfftfreq(frames.shape[1], 1.0 / sample_rate)
    levels = np.empty(len(THIRD_OCTAVE_CENTRES_HZ))
    for i, centre in enumerate(THIRD_OCTAVE_CENTRES_HZ):
        mask = (freqs >= centre * 2.0 ** (-1 / 6)) & (freqs < centre * 2.0 ** (1 / 6))
        band = power[mask].sum() if mask.any() else power[np.argmin(np.abs(freqs - centre))]
        levels[i] = 10.0 * np.log10(band + 1e-20)
    return levels


def spectrum_distance_db(a: np.ndarray, b: np.ndarray, sample_rate: int) -> float:
    """Return the RMS difference in dB of two clips' voiced long-term third-octave spectra.

    Each clip's spectrum is the mean power of its voiced frames in the 20 third-octave bands from
    100 Hz to 8 kHz. Band levels more than 30 dB below the stronger clip's strongest band are
    raised to that floor in both clips: bands the voice does not reach (below a high F0, or above
    its bandwidth) hold only noise, and removing that noise is not a change of voice. Speaker
    embedders are trained to ignore the channel, so this is the check
    that catches a cleaning step stripping an EQ-type effect. A clip with no voiced frame (a step
    that silenced it) is infinitely far.
    """
    levels_a = _third_octave_levels_db(a, sample_rate)
    levels_b = _third_octave_levels_db(b, sample_rate)
    if levels_a is None or levels_b is None:
        return float("inf")
    floor = max(levels_a.max(), levels_b.max()) - SPECTRUM_RANGE_DB
    levels_a = np.maximum(levels_a, floor)
    levels_b = np.maximum(levels_b, floor)
    return float(np.sqrt(np.mean((levels_a - levels_b) ** 2)))


def keep(
    take: TakeScores, floors: Mapping[str, float], cer_max: float = 0.03
) -> tuple[bool, tuple[str, ...]]:
    """Apply the take gate (VC-044): SIM at each state floor or above, CER at most `cer_max`.

    `floors` holds the take's state floor per embedder. Returns whether the take is kept, and a reason per failed check: `sim:<embedder>` in the
    order of `floors`, then `cer`. A take with no SIM for an embedder fails that embedder.
    """
    reasons = [
        f"sim:{name}"
        for name, floor in floors.items()
        if not take.sim.get(name, float("-inf")) >= floor
    ]
    if not take.cer <= cer_max:
        reasons.append("cer")
    return not reasons, tuple(reasons)


def _z_distance(values: Prosody, profile: StateProfile, spread: StateProfile) -> float:
    total = 0.0
    for field in ("f0_mean_hz", "f0_std_hz", "rate_sps", "energy_db"):
        scale = getattr(spread, field)
        if scale > 0.0:
            total += ((getattr(values, field) - getattr(profile, field)) / scale) ** 2
    return float(np.sqrt(total))


def select(kept: Sequence[TakeScores], profile: StateProfile, spread: StateProfile) -> TakeScores:
    """Return the kept take nearest the state's prosody profile, whatever its SIM.

    The distance is Euclidean over the four prosody fields, each as a z-score against the
    state's spread; a field with no spread is left out. Ties go to the earlier take.

    Raises:
        ValueError: No take was kept.
    """
    if not kept:
        raise ValueError("select needs at least one kept take")
    return min(kept, key=lambda take: _z_distance(take.prosody, profile, spread))


def teacher_gain(energies_db: Sequence[float], reference_db: float) -> float:
    """Return the gain in dB that brings a teacher's median take energy to `reference_db`.

    Fitted once on the teacher's bake-off takes across every state it covers, and applied
    unchanged to every later take: a gain refitted on a run of one or two states would pull
    their energy to the reference and erase the difference between states.

    Raises:
        ValueError: `energies_db` is empty.
    """
    if len(energies_db) == 0:
        raise ValueError("a teacher gain needs at least one take")
    return float(reference_db - np.median(np.asarray(energies_db, dtype=np.float64)))
