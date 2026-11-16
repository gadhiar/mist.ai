"""Tests for speaker-similarity scoring, state floors, the spectrum check and take selection (T6)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from scripts.voice_data.bindings import protocols
from scripts.voice_data.records import Prosody
from scripts.voice_data.stages import prosody, score

from .fakes import (
    FakeArousal,
    FakeAsr,
    FakeDiarizer,
    FakeEmbedder,
    FakeEnhancer,
    FakeQualityEstimator,
    FakeSeparator,
    FakeVad,
)

RATE = 16000


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def _orthogonal_noise(rng: np.random.Generator, axis: np.ndarray) -> np.ndarray:
    noise = rng.standard_normal(axis.shape)
    noise -= (noise @ axis) * axis
    return _unit(noise)


def _cluster(rng: np.random.Generator, axis: np.ndarray, count: int, along: float) -> list:
    # Each clip has cosine `along` to the cluster axis and the rest in a random direction.
    spread = np.sqrt(1.0 - along**2)
    return [along * axis + spread * _orthogonal_noise(rng, axis) for _ in range(count)]


# --- cosine, centroid, leave-one-out -------------------------------------------------------------


def test_cosine_is_scale_free() -> None:
    a = np.array([1.0, 2.0, 0.0])
    assert score.cosine(a, 3.0 * a) == pytest.approx(1.0)
    assert score.cosine(a, np.array([-2.0, 1.0, 5.0])) == pytest.approx(0.0)
    assert score.cosine(a, -a) == pytest.approx(-1.0)


def test_centroid_is_the_normalised_mean_of_normalised_clips() -> None:
    clips = [np.array([2.0, 0.0]), np.array([0.0, 0.5])]
    centre = score.centroid(clips)
    assert np.linalg.norm(centre) == pytest.approx(1.0)
    assert centre == pytest.approx(_unit(np.array([1.0, 1.0])))


def test_leave_one_out_with_one_source_uses_every_other_clip() -> None:
    rng = np.random.default_rng(1)
    clips = [rng.standard_normal(8) for _ in range(5)]
    scores = score.leave_one_out_scores(clips, ["src-1"] * 5)
    for i, clip in enumerate(clips):
        others = [c for j, c in enumerate(clips) if j != i]
        assert scores[i] == pytest.approx(score.cosine(clip, score.centroid(others)))


def test_leave_one_out_scores_each_clip_against_other_sources_only() -> None:
    rng = np.random.default_rng(2)
    axis = _unit(rng.standard_normal(16))
    clips = _cluster(rng, axis, 6, 0.8)
    sources = ["src-1", "src-1", "src-1", "src-2", "src-2", "src-3"]
    scores = score.leave_one_out_scores(clips, sources)
    for i, clip in enumerate(clips):
        others = [c for j, c in enumerate(clips) if sources[j] != sources[i]]
        assert scores[i] == pytest.approx(score.cosine(clip, score.centroid(others)))


def test_leave_one_out_reports_n_median_p02_and_p10() -> None:
    rng = np.random.default_rng(3)
    clips = [rng.standard_normal(8) for _ in range(30)]
    sources = [f"src-{i % 3}" for i in range(30)]
    values = score.leave_one_out_scores(clips, sources)
    dist = score.leave_one_out(clips, sources)
    assert dist == score.Distribution(
        n=30,
        median=pytest.approx(float(np.median(values))),
        p02=pytest.approx(float(np.percentile(values, 2))),
        p10=pytest.approx(float(np.percentile(values, 10))),
    )


def test_leave_one_out_needs_two_clips() -> None:
    with pytest.raises(ValueError):
        score.leave_one_out([np.ones(4)], ["src-1"])


def test_near_voice_falls_below_the_leave_one_out_p10_but_passes_a_pairwise_p10() -> None:
    rng = np.random.default_rng(4)
    dim = 64
    target_axis = _unit(rng.standard_normal(dim))
    # The near voice's axis sits at cosine 0.8 to the target's.
    near_axis = 0.8 * target_axis + 0.6 * _orthogonal_noise(rng, target_axis)
    target = _cluster(rng, target_axis, 40, 0.7)
    sources = [f"src-{i % 4}" for i in range(40)]
    near = _cluster(rng, near_axis, 10, 0.7)

    loo = score.leave_one_out(target, sources)
    pairwise = [score.cosine(target[i], target[j]) for i in range(40) for j in range(40) if i != j]
    pairwise_p10 = float(np.percentile(pairwise, 10))
    centre = score.centroid(target)
    near_sims = [score.cosine(clip, centre) for clip in near]

    assert all(sim < loo.p10 for sim in near_sims)
    assert all(sim >= pairwise_p10 for sim in near_sims)


# --- state floors --------------------------------------------------------------------------------

POOLED = score.Distribution(n=200, median=0.80, p02=0.66, p10=0.70)


def _scores(count: int, centre: float) -> list[float]:
    return list(np.linspace(centre - 0.05, centre + 0.05, count))


def test_a_state_with_twenty_segments_takes_its_own_p10() -> None:
    floors = score.state_floors({"neutral": _scores(20, 0.85)}, POOLED, {"neutral": 0.0})
    expected = float(np.percentile(_scores(20, 0.85), 10))
    assert floors["neutral"] == score.StateFloor(
        floor=pytest.approx(expected), floor_from="state", n=20
    )
    assert expected > POOLED.p10
    assert not floors["neutral"].flagged


def test_a_thin_sleepy_below_the_pooled_p10_gets_its_own() -> None:
    sleepy = _scores(12, 0.66)
    own = float(np.percentile(sleepy, 10))
    assert own < POOLED.p10
    floors = score.state_floors({"sleepy": sleepy}, POOLED, {"sleepy": 0.0})
    assert floors["sleepy"].floor == pytest.approx(own)
    assert floors["sleepy"].floor_from == "thin"
    assert floors["sleepy"].flagged


def test_a_thin_state_above_the_pooled_p10_is_held_to_the_pooled_p10() -> None:
    floors = score.state_floors({"calm": _scores(5, 0.85)}, POOLED, {"calm": 0.0})
    assert floors["calm"].floor == pytest.approx(POOLED.p10)
    assert floors["calm"].floor_from == "thin"


def test_a_three_segment_state_gets_the_pooled_p10_minus_its_offset() -> None:
    floors = score.state_floors(
        {"excited": [0.9, 0.91, 0.92], "awake": [0.5, 0.6, 0.7, 0.8]},
        POOLED,
        {"excited": 0.04, "awake": 0.02},
    )
    assert floors["excited"] == score.StateFloor(
        floor=pytest.approx(0.66), floor_from="pooled-offset", n=3
    )
    assert floors["awake"].floor == pytest.approx(0.68)
    assert floors["excited"].flagged


def test_a_state_without_an_offset_uses_zero() -> None:
    floors = score.state_floors({"calm": [0.9]}, POOLED, {})
    assert floors["calm"].floor == pytest.approx(POOLED.p10)


def test_state_floor_thresholds_are_parameters() -> None:
    floors = score.state_floors({"calm": _scores(10, 0.85)}, POOLED, {}, min_n=10, thin_n=3)
    assert floors["calm"].floor_from == "state"


# --- spectrum distance ---------------------------------------------------------------------------


def _presence_peak(audio: np.ndarray, rate: int, gain_db: float = 6.0) -> np.ndarray:
    # A bell in log frequency around 3 kHz, 0.6 octave standard deviation, applied by FFT.
    spectrum = np.fft.rfft(audio.astype(np.float64))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / rate)
    octaves = np.log2(np.maximum(freqs, 1.0) / 3000.0)
    gain = 10.0 ** (gain_db * np.exp(-0.5 * (octaves / 0.6) ** 2) / 20.0)
    return np.fft.irfft(spectrum * gain, n=len(audio)).astype(np.float32)


def _speechlike(f0: float, seconds: float = 3.0, rate: int = RATE) -> np.ndarray:
    # A harmonic voice with speech's long-term tilt (about -6 dB per octave above 500 Hz) and
    # formant bumps at 700, 1200 and 2600 Hz. Conftest's `synth_voice` falls off far faster
    # above its last formant, leaving the 4-8 kHz bands with no voice to measure an EQ by.
    t = np.arange(int(seconds * rate)) / rate
    phase = 2.0 * np.pi * np.cumsum(f0 * (1.0 + 0.02 * np.sin(2.0 * np.pi * 5.0 * t))) / rate
    harmonics = np.arange(1, int(rate / 2 / (f0 * 1.04)) + 1)
    freqs = harmonics * f0
    amps = 1.0 / (1.0 + freqs / 500.0)
    for formant in (700.0, 1200.0, 2600.0):
        amps = amps * (1.0 + 3.0 * np.exp(-0.5 * (np.log2(freqs / formant) / 0.25) ** 2))
    signal = sum(a * np.sin(k * phase) for k, a in zip(harmonics, amps, strict=True))
    signal = signal * (0.6 + 0.4 * np.sin(2.0 * np.pi * 4.0 * t) ** 2)
    return (0.5 * signal / np.max(np.abs(signal))).astype(np.float32)


def test_spectrum_distance_of_a_clip_to_itself_is_zero() -> None:
    voice = _speechlike(110.0)
    assert score.spectrum_distance_db(voice, voice, RATE) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("f0", [110.0, 220.0])
def test_a_six_db_presence_peak_moves_the_spectrum_more_than_two_db(f0: float) -> None:
    voice = _speechlike(f0)
    assert score.spectrum_distance_db(voice, _presence_peak(voice, RATE), RATE) > 2.0


@pytest.mark.parametrize("f0", [110.0, 220.0])
def test_removing_a_noise_floor_thirty_db_down_moves_it_less_than_two_db(
    f0: float, mix: Callable
) -> None:
    # At 220 Hz the bands below F0 hold only noise; the floor keeps them out of the distance.
    voice = _speechlike(f0)
    noisy = mix([(voice, 0.0)], RATE, noise_snr_db=30.0, seed=5)
    assert score.spectrum_distance_db(noisy, voice, RATE) < 2.0


def test_an_output_with_no_voiced_frames_is_infinitely_far() -> None:
    voice = _speechlike(110.0)
    assert score.spectrum_distance_db(voice, np.zeros_like(voice), RATE) == float("inf")
    noise_only = np.random.default_rng(6).normal(0.0, 0.05, len(voice)).astype(np.float32)
    assert score.spectrum_distance_db(voice, noise_only, RATE) == float("inf")


# --- keep and select -----------------------------------------------------------------------------

FLOORS = {"wavlm-sv": 0.70, "ecapa": 0.60}
PROFILE = Prosody(f0_mean_hz=200.0, f0_std_hz=30.0, rate_sps=4.0, energy_db=-24.0)
SPREAD = Prosody(f0_mean_hz=20.0, f0_std_hz=5.0, rate_sps=0.5, energy_db=3.0)


def _take(take_id: str, wavlm: float, ecapa: float, cer: float, **prosody_kw: float):
    fields = {**PROFILE.to_json(), **prosody_kw}
    return score.TakeScores(
        take_id=take_id,
        sim={"wavlm-sv": wavlm, "ecapa": ecapa},
        cer=cer,
        prosody=Prosody(**fields),
    )


def test_keep_passes_a_take_at_its_floors() -> None:
    assert score.keep(_take("t", 0.70, 0.60, 0.03), FLOORS) == (True, ())


@pytest.mark.parametrize(
    ("wavlm", "ecapa", "cer", "reasons"),
    [
        (0.69, 0.65, 0.0, ("sim:wavlm-sv",)),
        (0.80, 0.59, 0.0, ("sim:ecapa",)),
        (0.80, 0.65, 0.031, ("cer",)),
        (0.60, 0.50, 0.2, ("sim:wavlm-sv", "sim:ecapa", "cer")),
    ],
)
def test_keep_names_each_failed_gate(
    wavlm: float, ecapa: float, cer: float, reasons: tuple[str, ...]
) -> None:
    assert score.keep(_take("t", wavlm, ecapa, cer), FLOORS) == (False, reasons)


def test_keep_fails_a_take_missing_an_embedder() -> None:
    take = score.TakeScores(take_id="t", sim={"wavlm-sv": 0.9}, cer=0.0, prosody=PROFILE)
    assert score.keep(take, FLOORS) == (False, ("sim:ecapa",))


def test_keep_uses_the_given_cer_ceiling() -> None:
    assert score.keep(_take("t", 0.9, 0.9, 0.08), FLOORS, cer_max=0.10) == (True, ())


def test_select_picks_the_nearest_profile_over_a_higher_sim() -> None:
    loud = _take("loud", 0.95, 0.93, 0.0, f0_mean_hz=240.0, energy_db=-15.0)
    near = _take("near", 0.74, 0.65, 0.0, f0_mean_hz=205.0, energy_db=-23.0)
    fast = _take("fast", 0.90, 0.88, 0.0, rate_sps=5.5)
    assert score.select([loud, near, fast], PROFILE, SPREAD).take_id == "near"


def test_select_needs_a_kept_take() -> None:
    with pytest.raises(ValueError):
        score.select([], PROFILE, SPREAD)


# --- teacher gain --------------------------------------------------------------------------------

STATE_LEVELS_DB = {
    "neutral": -20.0,
    "calm": -24.0,
    "awake": -18.0,
    "sleepy": -28.0,
    "excited": -14.0,
}
REFERENCE_DB = -23.0


def _bakeoff(synth_voice: Callable) -> dict[str, list[np.ndarray]]:
    # The teacher's raw output: three takes per state, each state at its own level.
    voice = synth_voice(190.0, [600.0, 1500.0, 2700.0], 1.0, RATE)
    unit = voice / np.sqrt(np.mean(voice.astype(np.float64) ** 2))
    return {
        state: [unit * 10.0 ** ((level + 0.5 * k) / 20.0) for k in range(3)]
        for state, level in STATE_LEVELS_DB.items()
    }


def _apply(audio: np.ndarray, gain_db: float) -> np.ndarray:
    return audio * 10.0 ** (gain_db / 20.0)


def test_teacher_gain_brings_the_bakeoff_median_to_the_reference(synth_voice: Callable) -> None:
    takes = [take for state in _bakeoff(synth_voice).values() for take in state]
    energies = [prosody.active_energy_db(take, RATE) for take in takes]
    gain = score.teacher_gain(energies, REFERENCE_DB)
    after = [prosody.active_energy_db(_apply(take, gain), RATE) for take in takes]
    assert float(np.median(after)) == pytest.approx(REFERENCE_DB, abs=0.01)


def test_teacher_gain_keeps_the_energy_order_of_the_states(synth_voice: Callable) -> None:
    bakeoff = _bakeoff(synth_voice)
    energies = [prosody.active_energy_db(t, RATE) for takes in bakeoff.values() for t in takes]
    gain = score.teacher_gain(energies, REFERENCE_DB)
    medians = {
        state: float(np.median([prosody.active_energy_db(_apply(t, gain), RATE) for t in takes]))
        for state, takes in bakeoff.items()
    }
    by_level = sorted(STATE_LEVELS_DB, key=STATE_LEVELS_DB.__getitem__)
    assert sorted(medians, key=medians.__getitem__) == by_level


def test_a_stored_gain_keeps_a_one_state_run_off_the_reference(synth_voice: Callable) -> None:
    bakeoff = _bakeoff(synth_voice)
    energies = [prosody.active_energy_db(t, RATE) for takes in bakeoff.values() for t in takes]
    stored = score.teacher_gain(energies, REFERENCE_DB)
    run = bakeoff["sleepy"]
    run_energies = [prosody.active_energy_db(t, RATE) for t in run]
    offset_in_bakeoff = float(np.median(run_energies)) + stored - REFERENCE_DB

    with_stored = [prosody.active_energy_db(_apply(t, stored), RATE) for t in run]
    refit = score.teacher_gain(run_energies, REFERENCE_DB)
    with_refit = [prosody.active_energy_db(_apply(t, refit), RATE) for t in run]

    assert offset_in_bakeoff < -3.0
    assert float(np.median(with_stored)) - REFERENCE_DB == pytest.approx(offset_in_bakeoff)
    assert float(np.median(with_refit)) == pytest.approx(REFERENCE_DB, abs=0.01)


def test_teacher_gain_needs_takes() -> None:
    with pytest.raises(ValueError):
        score.teacher_gain([], REFERENCE_DB)


# --- fakes ---------------------------------------------------------------------------------------


def test_fake_embedder_separates_voices_with_different_formants(synth_voice: Callable) -> None:
    embedder = FakeEmbedder()

    def embed(f0: float, formants: list[float], seconds: float = 2.0) -> np.ndarray:
        return embedder.embed(synth_voice(f0, formants, seconds, RATE), RATE)

    first = embed(200.0, [800.0, 1200.0, 2800.0])
    assert score.cosine(first, embed(200.0, [800.0, 1200.0, 2800.0], 3.0)) > 0.99
    assert score.cosine(first, embed(205.0, [800.0, 1200.0, 2800.0])) > 0.95
    assert score.cosine(first, embed(200.0, [400.0, 2000.0, 3200.0])) < 0.75


def test_fake_embedder_ignores_level_and_is_deterministic(synth_voice: Callable) -> None:
    voice = synth_voice(180.0, [700.0, 1100.0, 2500.0], 2.0, RATE)
    a = FakeEmbedder().embed(voice, RATE)
    assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-5)
    assert score.cosine(a, FakeEmbedder().embed(0.1 * voice, RATE)) == pytest.approx(1.0)
    assert np.array_equal(a, FakeEmbedder().embed(voice, RATE))


def test_fakes_return_the_protocol_types(synth_voice: Callable) -> None:
    voice = synth_voice(180.0, [700.0, 1100.0, 2500.0], 1.0, RATE)
    padded = np.concatenate([np.zeros(RATE // 2, np.float32), voice, np.zeros(RATE, np.float32)])
    turn = protocols.Turn(start_s=0.5, end_s=1.5, speaker="spk-00")
    overlap = protocols.Span(start_s=1.0, end_s=1.2)

    assert np.array_equal(FakeSeparator().dialogue(voice, RATE), voice)
    diarizer = FakeDiarizer([turn], [overlap])
    assert diarizer.turns(voice, RATE) == [turn]
    assert diarizer.overlaps(voice, RATE) == [overlap]
    (span,) = FakeVad().speech(padded, RATE)
    assert span.start_s == pytest.approx(0.5, abs=0.03)
    assert span.end_s == pytest.approx(1.5, abs=0.03)
    track = FakeQualityEstimator(snr_db=22.0).frames(voice, RATE)
    assert len(track.snr_db) == 100
    assert set(track.snr_db.tolist()) == {22.0}
    assert np.array_equal(FakeEnhancer(fn=lambda a, r: 0.5 * a).apply(voice, RATE), 0.5 * voice)
    transcript = FakeAsr(text="one two").transcribe(voice, RATE)
    assert isinstance(transcript, protocols.Transcript)
    assert [w.text for w in transcript.words] == ["one", "two"]
    times = [t for w in transcript.words for t in (w.start_s, w.end_s)]
    assert times == pytest.approx([0.0, 0.45, 0.5, 0.95])
    monitor = FakeArousal()
    assert monitor.arousal(0.01 * voice, RATE) < monitor.arousal(voice, RATE) <= 1.0
