"""Her character stage: the one place the sound of her spoken voice is shaped.

Raj chose her character by ear: a film-style mix and a slight but audible detuned double, so she
sounds like a cyber entity while her intonation stays hers. Engines, voice identities and training
targets all stay clean; every reply she speaks passes through this stage before playback, so the
character lives here and nowhere else. One `CharacterStage` serves one reply, and its state runs
from silence at the reply's start, as the ear test's clips did. All processing is causal, in
float64, on 24 kHz mono PCM16 with full scale 1.0 at 32768.

Each part, in order, and why it is there:

- High-pass, a 2nd-order Butterworth at 100 Hz: removes rumble and the proximity weight below
  her voice, so the compressor responds to her voice rather than to low-frequency energy.
- Three peaking biquads (RBJ Audio EQ Cookbook `peakingEQ`): 300 Hz -3 dB thins the boxy
  low mids, 4 kHz +4 dB brings presence forward, and 9 kHz +2 dB adds air. With the high-pass
  they run as one `sosfilt` cascade whose state carries across chunks.
- Compressor, 4:1 above -40.5 dBFS with no knee and no makeup gain: evens out her level the way
  a film mix does. Its level is the RMS of the EQ output over the trailing 8 ms, and its gain is
  smoothed by a trailing 15 ms boxcar on linear gain, the same up and down. The ear test's
  window was centred and its threshold relative to each clip's loudest 8 ms; a stream can see
  neither, so both are made causal and the threshold is the clips' median, fixed.
- Detuned double: two `DetunedVoice` copies of the compressor's output, +15 cents centred 9 ms
  behind and -15 cents centred 14 ms behind, mixed at `double_gain` each. This is the character's
  synthetic edge, and the ear test made it with offline PSOLA. A stream cannot look ahead the
  two pitch periods PSOLA needs, so each copy is a pitch-synchronous delay-line shifter: its read
  delay moves by `1 - r` per sample, so it plays at exactly r = 2^(cents/1200) times the input's
  pitch, and when the delay nears one side of its range it jumps back by a whole number of the
  voice's periods, found by normalised cross-correlation, through a 4 ms crossfade whose gains
  sum to 1. Because the jump lands in phase, a splice neither combs nor dips the level, which a
  fixed two-tap delay line would.
- Fractional reads use a Kaiser-windowed sinc interpolator of `INTERPOLATOR_TAPS` (32) taps with
  `INTERPOLATOR_BETA` (5.0), its cutoff at the Nyquist frequency, each phase normalised to unit DC
  gain, tabulated at 1024 phases and linearly interpolated between them. Its magnitude error at
  9 kHz, the top EQ band, is under 0.002 dB at every fractional delay; a cubic interpolator would
  lose several dB there.
- Mix and output gain: `out = G_out * (y + g * (c1 + c2))`, rounded and clipped to PCM16, with the
  clipped samples counted. `G_out` keeps her loudness where the engine left it (`output_gain_db`).

The dry path adds no latency: the first output sample is the first input sample, processed. The
copies only ever read at least `delay_ms - sweep_ms / 2` behind, so they are causal too. At the
end of each segment `end_segment` runs `tail_samples` of silence through the stage, so the
copies finish the segment's last syllable inside that segment. Through the tail the copies are
fed silence rather than the dry path, which still rings (the EQ is recursive), so after it they
hold nothing of the segment and a later segment, after any wait, carries no scrap of it.
"""

import math

import numpy as np
from scipy.signal import butter, sosfilt

from .budget import CharacterConfig, DetunedCopy, PeakingBand

PCM16_FULL_SCALE = 32768.0
INTERPOLATOR_TAPS = 32
INTERPOLATOR_BETA = 5.0
_INTERPOLATOR_PHASES = 1024
# Tap offsets around `floor(position)`: 15 before it, itself, and 16 after it.
_TAP_OFFSETS = np.arange(1 - INTERPOLATOR_TAPS // 2, INTERPOLATOR_TAPS // 2 + 1)
# How far behind a read position the interpolator reaches, in whole samples.
_LOOKBACK = INTERPOLATOR_TAPS // 2

# The splice search: jumps of at least 2.5 ms (below that a jump cannot be a period of her voice),
# matched over 12 ms windows. Multiples of the period score alike, so the smallest jump within
# `_TIE_TOLERANCE` of the best score wins, and a rounding-level difference cannot flip the choice.
# Where both windows are below `_SILENCE_DBFS` there is nothing to match, and the jump closest to
# the last one is taken.
_MIN_JUMP_MS = 2.5
_MATCH_WINDOW_MS = 12.0
_TIE_TOLERANCE = 1e-3
_SILENCE_DBFS = -60.0


def _peaking_section(band: PeakingBand, sample_rate: int) -> np.ndarray:
    """One RBJ `peakingEQ` biquad as a second-order section, normalised by `a0`."""
    gain = 10.0 ** (band.gain_db / 40.0)
    w0 = 2.0 * math.pi * band.freq_hz / sample_rate
    alpha = math.sin(w0) / (2.0 * band.q)
    b = [1.0 + alpha * gain, -2.0 * math.cos(w0), 1.0 - alpha * gain]
    a = [1.0 + alpha / gain, -2.0 * math.cos(w0), 1.0 - alpha / gain]
    return np.array([*b, *a]) / a[0]


def dry_chain_sos(config: CharacterConfig, *, sample_rate: int) -> np.ndarray:
    """The EQ cascade the stage runs: the high-pass, then each peaking band in order.

    Raises:
        ValueError: when a frequency is not below the Nyquist frequency of `sample_rate`.
    """
    nyquist = sample_rate / 2.0
    for freq in (config.highpass_hz, *(band.freq_hz for band in config.bands)):
        if freq >= nyquist:
            raise ValueError(f"{freq} Hz is not below the Nyquist frequency {nyquist} Hz")
    highpass = butter(2, config.highpass_hz, "highpass", fs=sample_rate, output="sos")
    peaks = [_peaking_section(band, sample_rate) for band in config.bands]
    return np.vstack([highpass, *peaks])


def _trailing_mean(history: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The mean of each sample of `x` and the `len(history)` samples before it, and the new history.

    The same sums run whatever the chunking, so the streamed result equals the whole-array one.
    """
    padded = np.concatenate((history, x))
    length = len(history) + 1
    mean = np.convolve(padded, np.ones(length), "valid") / length
    return mean, padded[len(padded) - len(history) :]


class Compressor:
    """A causal 8 ms RMS compressor; float64 in and out, state carried across chunks.

    Level is `10 * log10` of the mean square of the trailing `rms_window_ms`, from silence. Above
    `threshold_dbfs` the gain falls by `1 - 1 / ratio` dB per dB, with no knee and no makeup gain.
    The linear gain is then averaged over the trailing `gain_smoothing_ms`, from unity, so the
    first sample passes at 0 dB.
    """

    def __init__(self, config: CharacterConfig, *, sample_rate: int) -> None:
        self._threshold = config.threshold_dbfs
        self._ratio = config.ratio
        window = max(1, round(config.rms_window_ms * sample_rate / 1000.0))
        smoothing = max(1, round(config.gain_smoothing_ms * sample_rate / 1000.0))
        self._squares = np.zeros(window - 1)
        self._gains = np.ones(smoothing - 1)

    def process(self, x: np.ndarray) -> np.ndarray:
        """Compress `x`; the output has the same length."""
        x = np.asarray(x, dtype=np.float64)
        if len(x) == 0:
            return np.zeros(0)
        mean_square, self._squares = _trailing_mean(self._squares, x * x)
        with np.errstate(divide="ignore"):
            level = 10.0 * np.log10(mean_square)
        over = np.maximum(level - self._threshold, 0.0)
        gain_db = -(over - over / self._ratio)
        gain, self._gains = _trailing_mean(self._gains, 10.0 ** (gain_db / 20.0))
        return x * gain


def _interpolator_table() -> np.ndarray:
    """Kaiser-windowed sinc taps for fractional delays 0, 1/P, ..., 1, one row per phase."""
    fractions = np.arange(_INTERPOLATOR_PHASES + 1) / _INTERPOLATOR_PHASES
    t = _TAP_OFFSETS[None, :] - fractions[:, None]
    half = INTERPOLATOR_TAPS / 2.0
    window = np.i0(INTERPOLATOR_BETA * np.sqrt(np.clip(1.0 - (t / half) ** 2, 0.0, None)))
    taps = np.sinc(t) * window / np.i0(INTERPOLATOR_BETA)
    return taps / taps.sum(axis=1, keepdims=True)


_TABLE = _interpolator_table()


def _read(buffer: np.ndarray, base: int, positions: np.ndarray) -> np.ndarray:
    """`buffer` read at fractional absolute `positions`; `buffer[0]` is absolute sample `base`."""
    whole = np.floor(positions)
    phase = (positions - whole) * _INTERPOLATOR_PHASES
    row = np.minimum(phase.astype(np.int64), _INTERPOLATOR_PHASES - 1)
    weight = (phase - row)[:, None]
    taps = _TABLE[row] * (1.0 - weight) + _TABLE[row + 1] * weight
    samples = buffer[(whole.astype(np.int64) - base)[:, None] + _TAP_OFFSETS[None, :]]
    return np.einsum("ij,ij->i", samples, taps)


def _parabolic_peak(scores: np.ndarray, index: int) -> tuple[float, float]:
    """The offset from `index` and the height of the parabola through its neighbours' scores."""
    if index == 0 or index == len(scores) - 1:
        return 0.0, float(scores[index])
    left, centre, right = scores[index - 1], scores[index], scores[index + 1]
    curvature = left - 2.0 * centre + right
    if curvature >= 0.0:
        return 0.0, float(centre)
    offset = float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5))
    return offset, float(centre - 0.25 * (left - right) * offset)


class DetunedVoice:
    """One copy of the double: a pitch-synchronous delay-line pitch shifter, causal.

    The copy reads its input at delay d(n), which moves by `1 - r` per sample between splices,
    so it plays at r times the input's pitch. d stays within `delay_ms +- sweep_ms / 2`: when it
    is half the last jump past the centre, it jumps back across the centre by J, a whole number
    of the input's periods. J is the jump in [2.5 ms, `sweep_ms`] whose landing lies in range and
    whose 12 ms window before the new read point best matches (normalised cross-correlation,
    refined by a parabola) the 12 ms before the current one; of jumps scoring within 1e-3 of the
    best, the smallest. The old and new reads crossfade over `splice_ms` with raised-cosine gains
    that sum to 1. A splice fires only after the previous crossfade ends, and early enough that
    the old read never leaves the range while it fades out.
    """

    def __init__(
        self, copy: DetunedCopy, *, sweep_ms: float, splice_ms: float, sample_rate: int
    ) -> None:
        ratio = 2.0 ** (copy.cents / 1200.0)
        self._rate = 1.0 - ratio
        self._direction = 1.0 if self._rate >= 0.0 else -1.0
        self._centre = copy.delay_ms * sample_rate / 1000.0
        self._half_sweep = sweep_ms * sample_rate / 2000.0
        lowest = self._centre - self._half_sweep
        highest = self._centre + self._half_sweep
        if lowest < _LOOKBACK:
            raise ValueError(
                f"the copy's shortest delay, {lowest:.1f} samples, is within the interpolator's "
                f"{_LOOKBACK} forward taps"
            )
        fade = max(1, round(splice_ms * sample_rate / 1000.0))
        self._fade_gain = 0.5 - 0.5 * np.cos(np.pi * (np.arange(fade) + 0.5) / fade)
        self._min_jump = _MIN_JUMP_MS * sample_rate / 1000.0
        self._max_jump = sweep_ms * sample_rate / 1000.0
        self._window = max(2, round(_MATCH_WINDOW_MS * sample_rate / 1000.0))
        # A splice fires at most this far past the centre, so the old read, still drifting
        # through the crossfade, stays inside the range.
        self._max_offset = self._half_sweep - fade * abs(self._rate) - 1.0
        self._silence = 10.0 ** (_SILENCE_DBFS / 10.0) * self._window  # energy over a window
        self._history_length = math.ceil(highest) + max(self._window, _LOOKBACK) + 4
        self._history = np.zeros(self._history_length)
        self._next = 0  # absolute index of the next input sample
        self._anchor = (0, self._centre)  # d(n) = delay + rate * (n - sample) from here
        self._faded_out: tuple[int, float] | None = None  # the old read's anchor while fading
        self._fade_start = 0
        self._last_jump = self._half_sweep
        self._fire_at = self._next_fire(earliest=1)

    @property
    def delay_samples(self) -> float:
        """The read delay at the last sample processed; the centre before any."""
        if self._next == 0:
            return self._centre
        return self._delay(self._anchor, self._next - 1)

    def process(self, x: np.ndarray) -> np.ndarray:
        """Shift `x`; the output has the same length."""
        x = np.asarray(x, dtype=np.float64)
        count = len(x)
        if count == 0:
            return np.zeros(0)
        first = self._next
        buffer = np.concatenate((self._history, x))
        base = first - self._history_length
        out = np.empty(count)
        done = 0
        while done < count:
            n = first + done
            if self._faded_out is not None:
                end = min(count, self._fade_start + len(self._fade_gain) - first)
                times = np.arange(n, first + end)
                gain = self._fade_gain[times - self._fade_start]
                old = _read(buffer, base, times - self._delay(self._faded_out, times))
                new = _read(buffer, base, times - self._delay(self._anchor, times))
                out[done:end] = (1.0 - gain) * old + gain * new
                if first + end == self._fade_start + len(self._fade_gain):
                    self._faded_out = None
                done = end
            elif n == self._fire_at:
                self._splice(buffer, base, n)
            else:
                end = min(count, self._fire_at - first)
                times = np.arange(n, first + end)
                out[done:end] = _read(buffer, base, times - self._delay(self._anchor, times))
                done = end
        self._history = buffer[len(buffer) - self._history_length :]
        self._next = first + count
        return out

    def _delay(self, anchor: tuple[int, float], times: np.ndarray | int) -> np.ndarray | float:
        sample, delay = anchor
        return delay + self._rate * (times - sample)

    def _next_fire(self, *, earliest: int) -> int:
        """The first sample, not before `earliest`, at which d is far enough past the centre."""
        if self._rate == 0.0:
            return np.iinfo(np.int64).max
        sample, delay = self._anchor
        limit = min(self._last_jump / 2.0, self._max_offset)
        offset = self._direction * (delay - self._centre)
        due = sample + math.ceil((limit - offset) / abs(self._rate))
        return max(earliest, due)

    def _splice(self, buffer: np.ndarray, base: int, n: int) -> None:
        delay = self._delay(self._anchor, n)
        offset = self._direction * (delay - self._centre)
        max_jump = min(self._max_jump, offset + self._half_sweep)
        min_jump = min(self._min_jump, max_jump)
        jump = self._choose_jump(buffer, base, n - delay, min_jump, max_jump)
        self._faded_out = self._anchor
        self._fade_start = n
        self._anchor = (n, delay - self._direction * jump)
        self._last_jump = jump
        self._fire_at = self._next_fire(earliest=n + len(self._fade_gain))

    def _choose_jump(
        self, buffer: np.ndarray, base: int, position: float, min_jump: float, max_jump: float
    ) -> float:
        """The jump J for a read at `position`: the read moves by `direction * J` samples."""
        fallback = float(np.clip(self._last_jump, min_jump, max_jump))
        window = self._window
        end = math.floor(position) - base + 1  # one past the current window, in `buffer`
        current = buffer[end - window : end]
        step = int(self._direction)  # the window's move per unit of jump
        lags = np.arange(math.ceil(min_jump), math.floor(max_jump) + 1)
        held = buffer[end + step * round(fallback) - window : end + step * round(fallback)]
        if len(lags) == 0 or (current @ current < self._silence and held @ held < self._silence):
            return fallback
        if step < 0:
            span = buffer[end - lags[-1] - window : end - lags[0]]
            views = np.lib.stride_tricks.sliding_window_view(span, window)[::-1]
        else:
            span = buffer[end + lags[0] - window : end + lags[-1]]
            views = np.lib.stride_tricks.sliding_window_view(span, window)
        dots = np.correlate(span, current, "valid")
        dots = dots[::-1] if step < 0 else dots
        energies = np.einsum("ij,ij->i", views, views) * (current @ current)
        scores = np.divide(dots, np.sqrt(energies), out=np.zeros(len(lags)), where=energies > 0.0)
        peaks = []
        for index in range(len(scores)):
            left = scores[index - 1] if index > 0 else -np.inf
            right = scores[index + 1] if index + 1 < len(scores) else -np.inf
            if scores[index] >= left and scores[index] >= right:
                shift, height = _parabolic_peak(scores, index)
                peaks.append((float(lags[index]) + shift, height))
        best = max(height for _, height in peaks)
        jump = min(lag for lag, height in peaks if height >= best - _TIE_TOLERANCE)
        return float(np.clip(jump, min_jump, max_jump))


def character_tail_samples(config: CharacterConfig, *, sample_rate: int) -> int:
    """Samples of silence after which no copy reads any of the audio before them; 0 if no double."""
    if config.double_gain == 0.0 or not config.copies:
        return 0
    longest = max(copy.delay_ms for copy in config.copies) + config.sweep_ms / 2.0
    return math.ceil(longest * sample_rate / 1000.0) + _LOOKBACK


class CharacterStage:
    """One reply's character stage; state carried across chunks and segments.

    `process` takes mono little-endian PCM16 and returns the same number of samples, processed.
    `end_segment` returns the stage run on `tail_samples` of silence, so the copies finish the
    segment's last syllable; it returns `b""` when nothing was processed since the last call.
    The stage goes on after it, so the next segment continues the same reply.
    """

    def __init__(self, config: CharacterConfig, *, sample_rate: int) -> None:
        self._sos = dry_chain_sos(config, sample_rate=sample_rate)
        self._state = np.zeros((len(self._sos), 2))
        self._compressor = Compressor(config, sample_rate=sample_rate)
        self._double_gain = config.double_gain
        self._voices = (
            tuple(
                DetunedVoice(
                    copy,
                    sweep_ms=config.sweep_ms,
                    splice_ms=config.splice_ms,
                    sample_rate=sample_rate,
                )
                for copy in config.copies
            )
            if config.double_gain > 0.0
            else ()
        )
        self._output_gain = 10.0 ** (config.output_gain_db / 20.0)
        self._tail = character_tail_samples(config, sample_rate=sample_rate)
        self._clipped = 0
        self._pending = False

    @property
    def tail_samples(self) -> int:
        """The length of each segment's tail; 0 when the double is off."""
        return self._tail

    @property
    def clipped_samples(self) -> int:
        """Output samples clipped to the PCM16 range so far."""
        return self._clipped

    def process(self, pcm16: bytes) -> bytes:
        """Process one chunk of mono little-endian PCM16; the output has the same length.

        Raises:
            ValueError: when `pcm16` holds an odd number of bytes.
        """
        if len(pcm16) % 2:
            raise ValueError(f"PCM16 chunk of {len(pcm16)} bytes is not whole samples")
        if not pcm16:
            return b""
        self._pending = True
        x = np.frombuffer(pcm16, dtype="<i2").astype(np.float64) / PCM16_FULL_SCALE
        return self._run(x)

    def end_segment(self) -> bytes:
        """The segment's tail: `tail_samples` of silence through the stage, or `b""`."""
        if not self._pending:
            return b""
        self._pending = False
        if self._tail == 0:
            return b""
        return self._run(np.zeros(self._tail), tail=True)

    def _run(self, x: np.ndarray, *, tail: bool = False) -> bytes:
        equalised, self._state = sosfilt(self._sos, x, zi=self._state)
        y = self._compressor.process(equalised)
        if self._voices:
            # Through the tail the dry path still rings (the EQ is recursive), so the copies are
            # fed silence instead: after the tail they hold nothing of the segment, and a later
            # segment, after any wait, starts with only the dry ring-down.
            heard = np.zeros(len(y)) if tail else y
            copies = self._voices[0].process(heard)
            for voice in self._voices[1:]:
                copies = copies + voice.process(heard)
            y = y + self._double_gain * copies
        scaled = np.rint(self._output_gain * y * PCM16_FULL_SCALE)
        self._clipped += int(np.count_nonzero((scaled > 32767.0) | (scaled < -32768.0)))
        return np.clip(scaled, -32768.0, 32767.0).astype("<i2").tobytes()
