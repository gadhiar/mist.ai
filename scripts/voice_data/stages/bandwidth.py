"""Effective bandwidth: the spectral cutoff of a copied track's dialogue channel.

A lossy web release keeps its 48 kHz sample rate while its encoder may have discarded everything
above 16 kHz, so the sample rate alone overstates what a source holds. The cutoff is the highest
frequency above which the long-term average spectrum (LTAS) of active frames stays at least
`below_db` under its 1-4 kHz mean (`ingest.bandwidth_cutoff_db` in `gates.py`, calibrated by H6).

Active frames stand in for voiced ones: frames within `ACTIVE_DB` of the 95th-percentile frame
energy. Silence and a quiet noise floor between lines are left out, so they cannot widen the band.
Spectra are summed per 1 dB frame-energy bin as they stream in, so an episode-length track needs
memory for a few hundred spectra, not for every frame.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence

import numpy as np

FRAME = 4096
ACTIVE_DB = 30.0
ACTIVE_PERCENTILE = 95.0
REFERENCE_BAND_HZ = (1000.0, 4000.0)
# Frame energies are binned in 1 dB steps from -150 dB to +50 dB relative to full scale.
_LOWEST_DB = -150
_BINS = 200
_READ_BYTES = FRAME * 4 * 64


class DecodeError(RuntimeError):
    """ffmpeg could not decode a track; its stderr is not kept, as it names the input path."""


class Ltas:
    """A streaming long-term average spectrum, binned by frame energy.

    Frames are non-overlapping and Hann-windowed; a trailing part frame is ignored.
    """

    def __init__(self, rate: int, frame: int = FRAME) -> None:
        self.rate = rate
        self.frame = frame
        self._window = np.hanning(frame)
        self._sums = np.zeros((_BINS, frame // 2 + 1))
        self._counts = np.zeros(_BINS, dtype=np.int64)
        self._pending = np.zeros(0, dtype=np.float64)

    def add(self, samples: np.ndarray) -> None:
        """Add mono samples; any length, carried over between calls."""
        data = np.concatenate([self._pending, np.asarray(samples, dtype=np.float64).ravel()])
        whole = len(data) // self.frame
        self._pending = data[whole * self.frame :]
        if whole == 0:
            return
        frames = data[: whole * self.frame].reshape(whole, self.frame)
        energy_db = 10.0 * np.log10(np.mean(frames**2, axis=1) + 1e-20)
        bins = np.clip(np.floor(energy_db).astype(np.int64) - _LOWEST_DB, 0, _BINS - 1)
        power = np.abs(np.fft.rfft(frames * self._window, axis=1)) ** 2
        np.add.at(self._sums, bins, power)
        np.add.at(self._counts, bins, 1)

    def cutoff_hz(self, below_db: float) -> int:
        """Return the effective bandwidth in Hz, or 0 when no frame or no reference energy."""
        total = int(self._counts.sum())
        if total == 0:
            return 0
        loud_bin = int(np.searchsorted(np.cumsum(self._counts), ACTIVE_PERCENTILE / 100 * total))
        active = np.arange(_BINS) >= loud_bin - ACTIVE_DB
        ltas = self._sums[active].sum(axis=0) / int(self._counts[active].sum())
        freqs = np.fft.rfftfreq(self.frame, 1.0 / self.rate)
        low, high = REFERENCE_BAND_HZ
        reference = ltas[(freqs >= low) & (freqs <= high)]
        if reference.size == 0 or reference.mean() <= 0:
            return 0
        above = np.nonzero(ltas >= reference.mean() * 10.0 ** (-below_db / 10.0))[0]
        return int(round(float(freqs[above[-1]])))


def effective_bandwidth(samples: np.ndarray, rate: int, below_db: float) -> int:
    """Return the effective bandwidth of mono samples held in memory."""
    ltas = Ltas(rate)
    ltas.add(samples)
    return ltas.cutoff_hz(below_db)


def stream_bandwidth(argv: Sequence[str], rate: int, below_db: float) -> int:
    """Run a decode argv that writes mono f32le to stdout and measure it as it streams.

    Raises:
        DecodeError: ffmpeg exited non-zero.
    """
    ltas = Ltas(rate)
    carry = b""
    with subprocess.Popen(list(argv), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as proc:
        stdout = proc.stdout
        if stdout is None:
            raise DecodeError("decode has no output pipe")
        while block := stdout.read(_READ_BYTES):
            data = carry + block
            usable = len(data) - len(data) % 4
            carry = data[usable:]
            ltas.add(np.frombuffer(data[:usable], dtype="<f4"))
    if proc.returncode != 0:
        raise DecodeError(f"decode exited {proc.returncode}")
    return ltas.cutoff_hz(below_db)


def read_decode(argv: Sequence[str]) -> np.ndarray:
    """Run a decode argv that writes mono f32le to stdout and return the samples.

    Raises:
        DecodeError: ffmpeg exited non-zero.
    """
    proc = subprocess.run(list(argv), capture_output=True, check=False)
    if proc.returncode != 0:
        raise DecodeError(f"decode exited {proc.returncode}")
    return np.frombuffer(proc.stdout, dtype="<f4").astype(np.float64)
