"""Tests for her character stage: the EQ, the compressor, the detuned double and the stage.

Every signal here is synthesized in the test: a steady harmonic vowel (harmonics of f0 up to
11 kHz, amplitudes falling 12 dB per octave), sines, DC and impulses. No audio file is read or
written. Each reference is computed offline over the whole array, from the definitions the plan
and the ear test give, and the streamed stage must match it:

- the dry chain against `scipy.signal.sosfilt` with coefficients built here from the RBJ Audio EQ
  Cookbook's peaking formula and a 2nd-order Butterworth high-pass;
- the compressor against a whole-array trailing 8 ms RMS, the static 4:1 law above
  `threshold_dbfs`, and a trailing 15 ms boxcar on linear gain;
- each copy against what a perfect pitch shifter would do: every harmonic moved by the copy's
  ratio with its level kept, and nothing else in the spectrum.

The tolerances are the plan's Test strategy's: 1 LSB where only float rounding can differ, and
measurement bounds elsewhere that a wrong ratio, a wrong coefficient or a comb would fail by a
wide margin.
"""

import math
import time

import numpy as np
import pytest
from scipy.signal import butter, sosfilt, sosfreqz

from backend.realtime.budget import CharacterConfig, DetunedCopy, PeakingBand
from backend.realtime.character import (
    CharacterStage,
    Compressor,
    DetunedVoice,
    dry_chain_sos,
)

FS = 24000
FULL_SCALE = 32768.0
# H20 section 3: the median, over the clips Raj heard, of the maximum of the 8 ms RMS envelope of
# the compressor's input (the EQ output).
REFERENCE_DBFS = -14.47
CLIP_CEILING = round(FULL_SCALE * 10 ** (-1.0 / 20.0))  # -1 dBFS in PCM16 steps


def _vowel(f0: float, seconds: float) -> np.ndarray:
    """A steady vowel: harmonics of `f0` up to 11 kHz, amplitude 1/k^2 (-12 dB per octave)."""
    t = np.arange(round(seconds * FS)) / FS
    harmonics = np.arange(1, int(11000.0 // f0) + 1)
    return sum(np.sin(2 * np.pi * k * f0 * t) / k**2 for k in harmonics)


def _pcm(x: np.ndarray) -> bytes:
    return np.clip(np.rint(x * FULL_SCALE), -32768, 32767).astype("<i2").tobytes()


def _samples(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.int64)


def _reference_sos(config: CharacterConfig) -> np.ndarray:
    """The dry chain's cascade, built here: Butterworth high-pass, then RBJ peakingEQ biquads."""
    rows = [butter(2, config.highpass_hz, "highpass", fs=FS, output="sos")]
    for band in config.bands:
        gain = 10 ** (band.gain_db / 40)
        w0 = 2 * math.pi * band.freq_hz / FS
        alpha = math.sin(w0) / (2 * band.q)
        b = [1 + alpha * gain, -2 * math.cos(w0), 1 - alpha * gain]
        a = [1 + alpha / gain, -2 * math.cos(w0), 1 - alpha / gain]
        rows.append(np.array([[*b, *a]]) / a[0])
    return np.vstack(rows)


def _trailing_mean(x: np.ndarray, length: int, history: float) -> np.ndarray:
    """The mean of each sample and the `length - 1` before it, with `history` before the start."""
    padded = np.concatenate((np.full(length - 1, history), x))
    return np.convolve(padded, np.ones(length), "valid") / length


def _offline_compressor(e: np.ndarray, config: CharacterConfig) -> np.ndarray:
    """H20 sections 3 to 5, made causal as the brief's Decision 3c to 3e reads them."""
    mean_square = _trailing_mean(e * e, round(config.rms_window_ms * FS / 1000), 0.0)
    with np.errstate(divide="ignore"):
        level = 10 * np.log10(mean_square)
    over = np.maximum(level - config.threshold_dbfs, 0.0)
    gain_db = -(over - over / config.ratio)
    gain = _trailing_mean(10 ** (gain_db / 20), round(config.gain_smoothing_ms * FS / 1000), 1.0)
    return e * gain


def _offline_dry(x: np.ndarray, config: CharacterConfig) -> np.ndarray:
    """The dry path in PCM16 steps: EQ, compressor and output gain, no copies."""
    y = _offline_compressor(sosfilt(_reference_sos(config), x), config)
    return np.rint(10 ** (config.output_gain_db / 20) * y * FULL_SCALE)


def _max_trailing_rms_dbfs(x: np.ndarray) -> float:
    return 10 * math.log10(np.max(_trailing_mean(x * x, 192, 0.0)))


def _at_reference_level(x: np.ndarray) -> np.ndarray:
    """`x` scaled so its EQ output's 8 ms RMS envelope peaks at H20's reference level."""
    level = _max_trailing_rms_dbfs(sosfilt(_reference_sos(CharacterConfig()), x))
    return x * 10 ** ((REFERENCE_DBFS - level) / 20)


def _calibration_proxy() -> np.ndarray:
    """Decision 3h's proxy for typical TTS output, at H20's reference level.

    The 200 Hz vowel, 3 s, under 12 syllables: each a 200 ms Hann window and 50 ms of silence,
    with peaks cycling 0, -6 and -12 dB.
    """
    vowel = _vowel(200.0, 3.0)
    syllable, gap = round(0.2 * FS), round(0.05 * FS)
    envelope = np.zeros(len(vowel))
    for index in range(12):
        start = index * (syllable + gap)
        envelope[start : start + syllable] = np.hanning(syllable) * 10 ** (-6 * (index % 3) / 20)
    return _at_reference_level(vowel * envelope)


def _run(stage: CharacterStage, pcm: bytes, sizes: list[int]) -> bytes:
    """`pcm` through `stage` in chunks of `sizes` samples (the last repeating), then the tail."""
    out = []
    position = 0
    index = 0
    while position < len(pcm) // 2:
        size = sizes[min(index, len(sizes) - 1)]
        out.append(stage.process(pcm[2 * position : 2 * (position + size)]))
        position += size
        index += 1
    out.append(stage.end_segment())
    return b"".join(out)


def _dry_only(**overrides: float) -> CharacterConfig:
    return CharacterConfig(double_gain=0.0, **overrides)


def _neutral_compressor(**overrides: float) -> CharacterConfig:
    """A configuration whose compressor passes everything at unity: ratio 1."""
    return CharacterConfig(ratio=1.0, threshold_dbfs=0.0, output_gain_db=0.0, **overrides)


def _run_voice(voice: DetunedVoice, x: np.ndarray, chunk: int) -> tuple[np.ndarray, np.ndarray]:
    """Run `voice` over `x` in chunks; return its output and its delay after each chunk."""
    out = []
    delays = []
    for start in range(0, len(x), chunk):
        out.append(voice.process(x[start : start + chunk]))
        delays.append(voice.delay_samples)
    return np.concatenate(out), np.array(delays)


def _voice(copy: DetunedCopy) -> DetunedVoice:
    config = CharacterConfig()
    return DetunedVoice(copy, sweep_ms=config.sweep_ms, splice_ms=config.splice_ms, sample_rate=FS)


COPIES = [
    pytest.param(DetunedCopy(15.0, 9.0), id="plus15-9ms"),
    pytest.param(DetunedCopy(-15.0, 14.0), id="minus15-14ms"),
]
F0S = [pytest.param(f0, id=f"f0-{f0:.0f}") for f0 in (120.0, 200.0, 280.0)]


def _spectrum(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Amplitude spectrum with a Kaiser (beta 20) window and 8x zero padding."""
    window = np.kaiser(len(x), 20.0)
    size = 8 * len(x)
    amplitude = np.abs(np.fft.rfft(x * window, size)) * 2 / window.sum()
    return np.fft.rfftfreq(size, 1 / FS), amplitude


def _peak(
    freqs: np.ndarray, amplitude: np.ndarray, near_hz: float, search_hz: float
) -> tuple[float, float]:
    """The interpolated frequency and level (dB) of the largest peak within `search_hz`."""
    lo, hi = np.searchsorted(freqs, [near_hz - search_hz, near_hz + search_hz])
    index = lo + int(np.argmax(amplitude[lo:hi]))
    a, b, c = 20 * np.log10(amplitude[index - 1 : index + 2])
    offset = 0.5 * (a - c) / (a - 2 * b + c)
    return freqs[index] + offset * (freqs[1] - freqs[0]), b - 0.25 * (a - c) * offset


# --- configuration ---------------------------------------------------------------------------


class TestCharacterConfig:
    def test_defaults_are_the_ear_tests_definitions(self) -> None:
        config = CharacterConfig()

        assert config.highpass_hz == 100.0
        assert config.bands == (
            PeakingBand(300.0, -3.0, 1.0),
            PeakingBand(4000.0, 4.0, 1.0),
            PeakingBand(9000.0, 2.0, 0.8),
        )
        assert config.rms_window_ms == 8.0
        assert config.threshold_dbfs == -40.5
        assert config.ratio == 4.0
        assert config.gain_smoothing_ms == 15.0
        assert config.copies == (DetunedCopy(15.0, 9.0), DetunedCopy(-15.0, 14.0))
        assert config.double_gain == 0.35

    @pytest.mark.parametrize(
        "overrides",
        [
            pytest.param({"highpass_hz": 12000.0}, id="highpass-at-12k"),
            pytest.param({"highpass_hz": 0.0}, id="highpass-not-positive"),
            pytest.param({"bands": (PeakingBand(12000.0, 2.0, 1.0),)}, id="band-at-12k"),
            pytest.param({"bands": (PeakingBand(4000.0, 2.0, 0.0),)}, id="q-zero"),
            pytest.param({"bands": (PeakingBand(4000.0, 2.0, -1.0),)}, id="q-negative"),
            pytest.param({"ratio": 0.99}, id="ratio-below-1"),
            pytest.param({"double_gain": -0.01}, id="double-gain-below-0"),
            pytest.param({"double_gain": 1.01}, id="double-gain-above-1"),
            pytest.param({"copies": (DetunedCopy(15.0, 6.9),)}, id="copy-shortest-delay-0.9ms"),
            pytest.param({"splice_ms": 6.0}, id="splice-not-below-half-sweep"),
        ],
    )
    def test_refuses_a_value_the_stage_cannot_run(self, overrides: dict) -> None:
        with pytest.raises(ValueError):
            CharacterConfig(**overrides)

    def test_accepts_the_edges_of_each_range(self) -> None:
        CharacterConfig(ratio=1.0, double_gain=0.0, copies=(DetunedCopy(15.0, 7.0),))
        CharacterConfig(double_gain=1.0, splice_ms=5.99)


# --- dry chain -------------------------------------------------------------------------------


class TestDryChain:
    @pytest.mark.parametrize(
        ("section", "freq_hz", "gain_db"),
        [
            pytest.param(0, 100.0, -3.01, id="highpass-100Hz"),
            pytest.param(1, 300.0, -3.0, id="peak-300Hz"),
            pytest.param(2, 4000.0, 4.0, id="peak-4kHz"),
            pytest.param(3, 9000.0, 2.0, id="peak-9kHz"),
        ],
    )
    def test_each_section_has_its_gain_at_its_own_frequency(
        self, section: int, freq_hz: float, gain_db: float
    ) -> None:
        sos = dry_chain_sos(CharacterConfig(), sample_rate=FS)

        _, response = sosfreqz(sos[section : section + 1], worN=[freq_hz], fs=FS)

        assert 20 * math.log10(abs(response[0])) == pytest.approx(gain_db, abs=0.01)

    def test_streamed_eq_matches_the_offline_cascade(self) -> None:
        config = _neutral_compressor(double_gain=0.0)
        x = 0.3 * _vowel(200.0, 1.0)
        stage = CharacterStage(config, sample_rate=FS)

        out = _samples(_run(stage, _pcm(x), [480]))

        quantized = _samples(_pcm(x)) / FULL_SCALE
        reference = np.rint(sosfilt(_reference_sos(config), quantized) * FULL_SCALE)
        assert np.max(np.abs(out[: len(x)] - reference)) <= 1


# --- compressor ------------------------------------------------------------------------------


class TestCompressor:
    def test_streamed_stage_matches_the_offline_compressor(self) -> None:
        config = _dry_only()
        x = _calibration_proxy()
        stage = CharacterStage(config, sample_rate=FS)

        out = _samples(_run(stage, _pcm(x), [480]))

        reference = _offline_dry(_samples(_pcm(x)) / FULL_SCALE, config)
        assert np.max(np.abs(out[: len(x)] - reference)) <= 1

    @pytest.mark.parametrize("level_dbfs", [float(level) for level in range(-50, 1, 5)])
    def test_static_curve_on_a_steady_1khz_sine(self, level_dbfs: float) -> None:
        config = CharacterConfig()
        amplitude = math.sqrt(2) * 10 ** (level_dbfs / 20)
        x = amplitude * np.sin(2 * np.pi * 1000.0 * np.arange(round(0.3 * FS)) / FS + 0.3)
        compressor = Compressor(config, sample_rate=FS)

        y = np.concatenate([compressor.process(x[i : i + 480]) for i in range(0, len(x), 480)])

        settled = slice(round(0.2 * FS), len(x))
        gain_db = 20 * np.log10(np.abs(y[settled]) / np.abs(x[settled]))
        over = max(level_dbfs - config.threshold_dbfs, 0.0)
        expected = -over * (1 - 1 / config.ratio)
        assert np.max(np.abs(gain_db - expected)) <= 0.1

    def test_a_quiet_first_sample_passes_at_unity(self) -> None:
        compressor = Compressor(CharacterConfig(), sample_rate=FS)

        y = compressor.process(np.array([1e-3, 0.9]))

        assert y[0] == 1e-3

    def test_a_loud_first_sample_passes_within_a_twentieth_of_a_db(self) -> None:
        compressor = Compressor(CharacterConfig(), sample_rate=FS)

        y = compressor.process(np.array([0.9, 0.9]))

        assert abs(20 * math.log10(y[0] / 0.9)) <= 0.05


# --- detuned double --------------------------------------------------------------------------


class TestDetunedVoice:
    @pytest.mark.parametrize("copy", COPIES)
    @pytest.mark.parametrize("f0", F0S)
    def test_pitch_moves_by_the_copys_cents(self, copy: DetunedCopy, f0: float) -> None:
        x = _vowel(f0, 3.0)

        y, _ = _run_voice(_voice(copy), x, 480)

        steady = slice(FS, 2 * FS)
        f_in, _ = _peak(*_spectrum(x[steady]), 10 * f0, 0.02 * 10 * f0)
        f_out, _ = _peak(*_spectrum(y[steady]), 10 * f0, 0.02 * 10 * f0)
        assert 1200 * math.log2(f_out / f_in) == pytest.approx(copy.cents, abs=1.0)

    @pytest.mark.parametrize("copy", COPIES)
    @pytest.mark.parametrize("f0", F0S)
    def test_harmonics_keep_their_level_and_nothing_else_appears(
        self, copy: DetunedCopy, f0: float
    ) -> None:
        ratio = 2 ** (copy.cents / 1200)
        x = _vowel(f0, 3.0)

        y, _ = _run_voice(_voice(copy), x, 480)

        steady = slice(FS, 2 * FS)
        freqs, spec_in = _spectrum(x[steady])
        _, spec_out = _spectrum(y[steady])
        harmonics = np.arange(1, int(9000.0 // f0) + 1)
        errors = [
            _peak(freqs, spec_out, k * f0 * ratio, 3.0)[1] - _peak(freqs, spec_in, k * f0, 3.0)[1]
            for k in harmonics
        ]
        assert max(abs(error) for error in errors) <= 0.5
        strongest = 20 * np.log10(np.max(spec_out))
        others = (freqs > 20.0) & (freqs <= 9000.0)
        for k in np.arange(1, int(11000.0 // f0) + 2):
            others &= np.abs(freqs - k * f0 * ratio) > 25.0
        assert 20 * np.log10(np.max(spec_out[others])) - strongest < -50.0

    @pytest.mark.parametrize("copy", COPIES)
    def test_dc_passes_unchanged_through_every_splice(self, copy: DetunedCopy) -> None:
        x = np.full(3 * FS, 0.5)

        y, delays = _run_voice(_voice(copy), x, 240)

        assert np.max(np.abs(np.diff(delays))) > 50.0, "no splice happened"
        filled = round(FS * (copy.delay_ms + 6.0) / 1000) + 16
        assert np.max(np.abs(y[filled:] - 0.5)) <= 1e-12

    @pytest.mark.parametrize("copy", COPIES)
    def test_a_sines_level_holds_through_every_splice(self, copy: DetunedCopy) -> None:
        amplitude = 0.5
        x = amplitude * np.sin(2 * np.pi * 200.0 * np.arange(3 * FS) / FS)

        y, delays = _run_voice(_voice(copy), x, 240)

        assert np.max(np.abs(np.diff(delays))) > 50.0, "no splice happened"
        rms = np.sqrt(_trailing_mean(y * y, round(0.005 * FS), 0.0))
        filled = round(FS * (copy.delay_ms + 6.0) / 1000) + 16 + round(0.005 * FS)
        level_db = 20 * np.log10(rms[filled:] / (amplitude / math.sqrt(2)))
        assert np.max(np.abs(level_db)) <= 0.5

    @pytest.mark.parametrize("copy", COPIES)
    @pytest.mark.parametrize("f0", F0S)
    def test_delay_centres_on_the_copys_delay_and_stays_in_range(
        self, copy: DetunedCopy, f0: float
    ) -> None:
        x = _vowel(f0, 2.0)

        _, delays = _run_voice(_voice(copy), x, 8)

        delays_ms = delays * 1000 / FS
        assert np.mean(delays_ms) == pytest.approx(copy.delay_ms, abs=1.5)
        assert np.min(delays_ms) >= copy.delay_ms - 6.0
        assert np.max(delays_ms) <= copy.delay_ms + 6.0

    @pytest.mark.parametrize("copy", COPIES)
    def test_after_the_tail_a_copy_outputs_exact_zeros(self, copy: DetunedCopy) -> None:
        tail = CharacterStage(CharacterConfig(), sample_rate=FS).tail_samples
        voice = _voice(copy)
        voice.process(_vowel(200.0, 1.0))
        voice.process(np.zeros(tail))

        later = np.concatenate([voice.process(np.zeros(480)) for _ in range(50)])

        assert np.all(later == 0.0)


# --- the stage -------------------------------------------------------------------------------


class TestCharacterStage:
    def test_dry_path_adds_no_latency(self) -> None:
        stage = CharacterStage(_dry_only(), sample_rate=FS)
        impulse = np.zeros(1000)
        impulse[137] = 0.5

        out = _samples(stage.process(_pcm(impulse)))

        assert np.all(out[:137] == 0)
        assert out[137] != 0

    @pytest.mark.parametrize("size", [0, 1, 7, 480, 4097])
    def test_process_returns_its_inputs_length(self, size: int) -> None:
        stage = CharacterStage(CharacterConfig(), sample_rate=FS)

        out = stage.process(_pcm(0.1 * _vowel(200.0, 1.0)[:size]))

        assert len(out) == 2 * size

    def test_process_refuses_an_odd_byte_count(self) -> None:
        stage = CharacterStage(CharacterConfig(), sample_rate=FS)

        with pytest.raises(ValueError):
            stage.process(b"\x00\x00\x00")

    def test_tail_covers_the_longest_read_of_either_copy(self) -> None:
        stage = CharacterStage(CharacterConfig(), sample_rate=FS)

        assert stage.tail_samples == math.ceil((14.0 + 6.0) * FS / 1000) + 16

    def test_end_segment_returns_the_tail_once_per_segment(self) -> None:
        stage = CharacterStage(CharacterConfig(), sample_rate=FS)
        assert stage.end_segment() == b""

        stage.process(_pcm(0.1 * _vowel(200.0, 0.1)))
        first = stage.end_segment()
        repeated = stage.end_segment()
        stage.process(b"")
        after_empty = stage.end_segment()
        stage.process(_pcm(0.1 * _vowel(200.0, 0.1)))
        second = stage.end_segment()

        assert len(first) == 2 * stage.tail_samples
        assert repeated == b""
        assert after_empty == b""
        assert len(second) == 2 * stage.tail_samples

    def test_without_the_double_there_is_no_tail(self) -> None:
        stage = CharacterStage(_dry_only(), sample_rate=FS)
        stage.process(_pcm(0.1 * _vowel(200.0, 0.1)))

        assert stage.tail_samples == 0
        assert stage.end_segment() == b""

    @pytest.mark.parametrize(
        "sizes",
        [
            pytest.param([1], id="1"),
            pytest.param([7], id="7"),
            pytest.param([480], id="480"),
            pytest.param([960], id="960"),
            pytest.param([4096], id="4096"),
            pytest.param(
                [int(n) for n in np.random.default_rng(50).integers(1, 3000, size=200)],
                id="seeded-random",
            ),
        ],
    )
    def test_output_does_not_depend_on_chunking(self, sizes: list[int]) -> None:
        pcm = _pcm(_at_reference_level(_vowel(200.0, 1.2)))
        whole = _samples(_run(CharacterStage(CharacterConfig(), sample_rate=FS), pcm, [len(pcm)]))

        chunked = _samples(_run(CharacterStage(CharacterConfig(), sample_rate=FS), pcm, sizes))

        assert len(chunked) == len(whole)
        assert np.max(np.abs(chunked - whole)) <= 1

    def test_the_chunking_signal_crosses_splices(self) -> None:
        for copy in CharacterConfig().copies:
            _, delays = _run_voice(_voice(copy), _vowel(200.0, 1.2), 240)

            assert np.max(np.abs(np.diff(delays))) > 50.0

    def test_after_a_segments_tail_only_the_dry_ring_down_follows(self) -> None:
        """The gap after a tail is the dry ring-down; the copies hold nothing of the segment.

        The EQ is recursive, so the dry path still rings in the gap, and the copies, which read
        the gap's own dry signal, echo that ring-down at `double_gain` each. So the gap may differ
        from the dry-only reference by at most `2 * double_gain` of the ring-down's own peak, plus
        rounding: a few LSB here, where a scrap of the segment's last syllable would be hundreds.
        """
        config = CharacterConfig()
        stage = CharacterStage(config, sample_rate=FS)
        vowel = _pcm(_at_reference_level(_vowel(200.0, 1.0)))
        silence = bytes(2 * round(0.5 * FS))
        stage.process(vowel)
        stage.end_segment()

        gap = _samples(stage.process(silence))

        whole = np.concatenate((_samples(vowel) / FULL_SCALE, np.zeros(stage.tail_samples)))
        whole = np.concatenate((whole, np.zeros(len(silence) // 2)))
        reference = _offline_dry(whole, config)[-len(gap) :]
        echo_bound = 2 * config.double_gain * np.max(np.abs(reference))
        assert np.max(np.abs(gap - reference)) <= 1 + echo_bound
        # The first 3 ms (72 samples, the 9 ms copy's shortest delay) hold to 1 LSB because the
        # copies add less than 1 LSB there, not because they read only the tail: the
        # interpolator's 16 forward taps reach the gap from sample 72 - 16 = 56 at that delay.
        copies_below_one_lsb = round(FS * (9.0 - 6.0) / 1000)
        assert np.max(np.abs(gap[:copies_below_one_lsb] - reference[:copies_below_one_lsb])) <= 1

    def test_before_the_forward_taps_reach_the_gap_the_copies_add_nothing(self) -> None:
        """Through the gap's first 56 samples every copy read lies in the tail's silence.

        The 9 ms copy's shortest delay is 72 samples and the interpolator reads 16 samples ahead
        of its read point, so the first gap sample a copy can touch is sample 56. Before it the
        copies are exactly 0, and the stage equals the same stage without the double.
        """
        vowel = _pcm(_at_reference_level(_vowel(200.0, 1.0)))
        silence = bytes(2 * round(0.5 * FS))
        stage = CharacterStage(CharacterConfig(), sample_rate=FS)
        stage.process(vowel)
        dry = CharacterStage(_dry_only(), sample_rate=FS)
        dry.process(vowel)
        dry.process(bytes(2 * stage.tail_samples))
        stage.end_segment()

        gap = _samples(stage.process(silence))

        dry_gap = _samples(dry.process(silence))
        tail_only = round(FS * (9.0 - 6.0) / 1000) - 16
        assert np.array_equal(gap[:tail_only], dry_gap[:tail_only])

    def test_counts_the_samples_it_clips(self) -> None:
        config = CharacterConfig(double_gain=0.0, ratio=1.0, output_gain_db=12.0)
        x = 0.5 * np.sin(2 * np.pi * 1000.0 * np.arange(FS // 10) / FS)
        stage = CharacterStage(config, sample_rate=FS)

        out = _samples(stage.process(_pcm(x)))

        unclipped = np.rint(
            10 ** (12.0 / 20)
            * sosfilt(_reference_sos(config), _samples(_pcm(x)) / FULL_SCALE)
            * FULL_SCALE
        )
        expected = int(np.sum((unclipped > 32767) | (unclipped < -32768)))
        assert expected > 0
        assert stage.clipped_samples == pytest.approx(expected, abs=2)
        assert out.max() == 32767
        assert out.min() == -32768

    def test_refuses_a_band_at_or_above_nyquist(self) -> None:
        with pytest.raises(ValueError):
            CharacterStage(CharacterConfig(), sample_rate=16000)


# --- level, calibration and cost ---------------------------------------------------------------


class TestLevel:
    def test_typical_speech_at_the_reference_level_does_not_clip(self) -> None:
        stage = CharacterStage(CharacterConfig(), sample_rate=FS)

        out = _samples(_run(stage, _pcm(_calibration_proxy()), [1920]))

        assert stage.clipped_samples == 0
        assert np.max(np.abs(out)) <= CLIP_CEILING

    @pytest.mark.parametrize("f0", F0S)
    def test_a_steady_vowel_at_the_reference_level_peaks_below_1_dbfs_once_settled(
        self, f0: float
    ) -> None:
        """A vowel that starts at full level is the design doc's loud first syllable.

        The compressor's gain starts at unity, so its first 15 ms may clip, and the stage counts
        that; once the gain has settled nothing reaches -1 dBFS.
        """
        stage = CharacterStage(CharacterConfig(), sample_rate=FS)

        out = _samples(_run(stage, _pcm(_at_reference_level(_vowel(f0, 3.0))), [1920]))

        settled = round(0.03 * FS)
        assert np.max(np.abs(out[settled:])) <= CLIP_CEILING

    def test_default_output_gain_keeps_the_input_loudness(self) -> None:
        pcm = _pcm(_calibration_proxy())
        stage = CharacterStage(CharacterConfig(output_gain_db=0.0), sample_rate=FS)

        out = _samples(_run(stage, pcm, [1920]))[: len(pcm) // 2]

        x = _samples(pcm).astype(float)
        derived = 20 * math.log10(math.sqrt(np.mean(x * x)) / math.sqrt(np.mean(out * out)))
        default = CharacterConfig().output_gain_db
        assert abs(derived - default) <= 0.1, f"derived output_gain_db {derived:.3f}"
        assert default == round(default, 1)

    def test_twenty_seconds_cost_under_two_seconds_of_cpu(self) -> None:
        pcm = _pcm(_at_reference_level(_vowel(200.0, 20.0)))
        chunk = 2 * round(0.08 * FS)

        timings = []
        for _ in range(3):
            stage = CharacterStage(CharacterConfig(), sample_rate=FS)
            start = time.process_time()
            for offset in range(0, len(pcm), chunk):
                stage.process(pcm[offset : offset + chunk])
            stage.end_segment()
            timings.append(time.process_time() - start)

        assert min(timings) < 2.0, f"best of 3: {min(timings):.3f} s"
