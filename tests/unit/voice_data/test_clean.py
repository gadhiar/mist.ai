"""SIM-gated cleaning, loudness, transcripts and references through `clean` (T8).

The step gate is tested on its own first, with the fake embedders of `fakes.py`: a hum remover
keeps the embedding and the spectrum, a comb remover moves the embedding, and an EQ remover is
caught by the spectrum check alone once the embedder ignores the channel. A synthetic extract run
then goes through the command, whose job runs the stage in this process with fake models (as
`test_extract.py` does), and the tests read what it wrote.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import lfilter

from scripts.voice_data import cli, jobs
from scripts.voice_data import root as root_mod
from scripts.voice_data.bindings.protocols import Transcript, Word
from scripts.voice_data.commands import clean as clean_cmd
from scripts.voice_data.records import Prosody, SegmentRecord, read_jsonl, write_jsonl
from scripts.voice_data.stages import clean as clean_stage
from scripts.voice_data.stages import prosody as prosody_mod
from scripts.voice_data.stages import score, transcribe
from scripts.voice_data.stages.clean import RATE

from .conftest import Canaries, _assert_no_canary, _synth_voice
from .fakes import FakeAsr, FakeEmbedder, FakeEnhancer

SOURCE_RATE = 48000
IMAGE = "mist-voice-extract:0123456789ab"
FORMANTS = (700.0, 1200.0, 2600.0)
NOISE_SNR_DB = 20.0
HUM_LEVEL = 0.01
SPECTRUM_MAX_DB = 2.0
CANARY = Canaries()


# Signals and steps.


def comb(audio: np.ndarray, rate: int) -> np.ndarray:
    """The in-show effect: a 2 ms feed-forward comb, notches every 500 Hz."""
    delay = int(0.002 * rate)
    out = audio.astype(np.float64).copy()
    out[delay:] += 0.7 * audio[:-delay]
    return out.astype(np.float32)


def decomb(audio: np.ndarray, rate: int) -> np.ndarray:
    """The comb's exact inverse, as a cleaning step that removes the effect."""
    delay = int(0.002 * rate)
    den = np.zeros(delay + 1)
    den[0], den[delay] = 1.0, 0.7
    return lfilter([1.0], den, audio.astype(np.float64)).astype(np.float32)


def _shape(audio: np.ndarray, rate: int, gain: float) -> np.ndarray:
    spectrum = np.fft.rfft(audio.astype(np.float64))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / rate)
    spectrum *= np.where((freqs > 1500.0) & (freqs < 4000.0), gain, 1.0)
    return np.fft.irfft(spectrum, len(audio)).astype(np.float32)


def eq(audio: np.ndarray, rate: int) -> np.ndarray:
    """An EQ-type in-show effect: +12 dB from 1.5 to 4 kHz."""
    return _shape(audio, rate, 4.0)


def de_eq(audio: np.ndarray, rate: int) -> np.ndarray:
    """A step that strips the EQ effect."""
    return _shape(audio, rate, 0.25)


def hum(samples: int, rate: int) -> np.ndarray:
    return (HUM_LEVEL * np.sin(2.0 * np.pi * 50.0 * np.arange(samples) / rate)).astype(np.float32)


def dehum(audio: np.ndarray, rate: int) -> np.ndarray:
    """A denoiser of mains hum: everything below 70 Hz removed, below both measures' bands."""
    spectrum = np.fft.rfft(audio.astype(np.float64))
    spectrum[np.fft.rfftfreq(len(audio), 1.0 / rate) < 70.0] = 0.0
    return np.fft.irfft(spectrum, len(audio)).astype(np.float32)


def noise(samples: int, level: float, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, level, samples).astype(np.float32)


@functools.cache
def base_voice(rate: int) -> np.ndarray:
    """14 s of the target with the comb effect, from which segments are cut."""
    return comb(_synth_voice(210.0, FORMANTS, 14.0, rate), rate)


def cut(rate: int, seconds: float, k: int, gain_db: float = 0.0) -> np.ndarray:
    """A segment of `seconds`: hum and noise at a fixed SNR added, then `gain_db` applied."""
    voice = base_voice(rate)
    offset = int(((k * 1.3) % (14.0 - seconds - 0.1)) * rate)
    clip = voice[offset : offset + int(round(seconds * rate))].astype(np.float64)
    level = float(np.sqrt(np.mean(clip**2))) * 10.0 ** (-NOISE_SNR_DB / 20.0)
    clip = clip + hum(len(clip), rate) + noise(len(clip), level, seed=k)
    return (clip * 10.0 ** (gain_db / 20.0)).astype(np.float32)


def energy(audio: np.ndarray, rate: int) -> float:
    return prosody_mod.active_energy_db(audio, rate)


def embedders() -> tuple[FakeEmbedder, FakeEmbedder]:
    return FakeEmbedder("wavlm-sv"), FakeEmbedder("ecapa")


class PitchEmbedder:
    """A channel-invariant embedder: the angle of its 2-D embedding is the clip's pitch period.

    EQ changes the spectral envelope and not the period, so every EQ step keeps this embedding.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        signal = np.asarray(audio[:sample_rate], dtype=np.float64)
        spectrum = np.fft.rfft(signal, 2 * len(signal))
        corr = np.fft.irfft(np.abs(spectrum) ** 2)[: len(signal)]
        lo, hi = sample_rate // 400, sample_rate // 60
        angle = (lo + int(np.argmax(corr[lo:hi]))) / 100.0
        return np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)


# The step gate on its own.


SOURCES = ("a", "a", "b", "b", "c", "c")


@dataclass(frozen=True)
class Inputs:
    audios: list[np.ndarray]
    embeddings: list[dict[str, np.ndarray]]
    thresholds: dict[str, float]
    models: tuple[object, ...]


def inputs(audios: list[np.ndarray], models: tuple[object, ...]) -> Inputs:
    embs = [clean_stage.embed(models, a, RATE) for a in audios]  # type: ignore[arg-type]
    return Inputs(audios, embs, clean_stage.sim_thresholds(embs, SOURCES, 50.0), models)


def gate(step: FakeEnhancer, given: Inputs) -> clean_stage.StepResult:
    return clean_stage.gate_step(
        step, given.audios, given.embeddings, RATE, given.models,  # type: ignore[arg-type]
        given.thresholds, SPECTRUM_MAX_DB,
    )  # fmt: skip


@pytest.fixture(scope="module")
def comb_inputs() -> Inputs:
    return inputs([cut(RATE, 3.0 + 0.5 * k, k) for k in range(6)], embedders())


def test_a_denoiser_that_preserves_embedding_and_spectrum_is_kept(comb_inputs: Inputs) -> None:
    result = gate(FakeEnhancer("dehum", dehum), comb_inputs)
    decision = result.decision
    assert decision.kept, decision
    assert decision.reasons == ()
    assert all(decision.sim[n] >= decision.threshold[n] for n in ("wavlm-sv", "ecapa"))
    assert decision.spectrum_db <= SPECTRUM_MAX_DB
    assert len(result.audios) == len(comb_inputs.audios)
    first = dehum(comb_inputs.audios[0], RATE)
    assert float(np.max(np.abs(result.audios[0] - first))) < 1e-6


def test_a_step_that_removes_the_comb_effect_moves_the_embedding_and_is_dropped(
    comb_inputs: Inputs,
) -> None:
    result = gate(FakeEnhancer("decomb", decomb), comb_inputs)
    decision = result.decision
    assert not decision.kept
    assert "sim:wavlm-sv" in decision.reasons and "sim:ecapa" in decision.reasons
    assert all(decision.sim[n] < decision.threshold[n] for n in ("wavlm-sv", "ecapa"))
    # A dropped step changes nothing: the next step starts from the same audio and embeddings.
    assert result.audios[0] is comb_inputs.audios[0]
    assert result.embeddings[0] is comb_inputs.embeddings[0]


def test_with_a_channel_invariant_embedder_an_eq_removal_is_dropped_by_the_spectrum_alone() -> None:
    segments = [
        eq(_synth_voice(f0, FORMANTS, 3.0, RATE), RATE)
        for f0 in (200.0, 204.0, 208.0, 212.0, 216.0, 220.0)
    ]
    given = inputs(segments, (PitchEmbedder("wavlm-sv"), PitchEmbedder("ecapa")))
    decision = gate(FakeEnhancer("de-eq", de_eq), given).decision
    assert not decision.kept
    assert decision.reasons == ("spectrum",)
    assert all(decision.sim[n] >= decision.threshold[n] for n in ("wavlm-sv", "ecapa"))
    assert decision.spectrum_db > SPECTRUM_MAX_DB


def test_a_step_that_silences_the_audio_fails_the_spectrum_check(comb_inputs: Inputs) -> None:
    silence = FakeEnhancer("mute", lambda a, r: np.zeros_like(a))
    decision = gate(silence, comb_inputs).decision
    assert not decision.kept and "spectrum" in decision.reasons
    assert decision.to_json()["spectrum_db"] is None


def test_sim_thresholds_are_the_leave_one_out_percentile(comb_inputs: Inputs) -> None:
    embs = comb_inputs.embeddings
    for percentile in (50.0, 10.0):
        found = clean_stage.sim_thresholds(embs, SOURCES, percentile)
        for name in ("wavlm-sv", "ecapa"):
            loo = score.leave_one_out_scores([e[name] for e in embs], SOURCES)
            assert found[name] == pytest.approx(float(np.percentile(loo, percentile)))


# Loudness on its own.


def test_one_gain_per_source_from_its_median_energy() -> None:
    gains = clean_stage.source_gains(
        ["s1", "s1", "s1", "s2"], [-20.0, -30.0, -26.0, -10.0], [-6.0, -16.0, -12.0, -3.0],
        target_db=-23.0, ceiling_dbfs=-1.0,
    )  # fmt: skip
    assert gains == {"s1": pytest.approx(3.0), "s2": pytest.approx(-13.0)}


def test_a_source_gain_never_lifts_its_loudest_peak_above_the_ceiling() -> None:
    gains = clean_stage.source_gains(
        ["s1", "s1"], [-40.0, -40.0], [-2.0, -20.0], target_db=-23.0, ceiling_dbfs=-1.0
    )
    assert gains == {"s1": pytest.approx(1.0)}


# The command, on a synthetic extract run.


@dataclass(frozen=True, slots=True)
class Seg:
    sid: str
    source: str
    state: str
    seconds: float
    gain_db: float = 0.0
    cluster: str = "primary"


SEGS = (
    Seg("seg-000001", "src-0001", "neutral", 5.0),
    Seg("seg-000002", "src-0002", "neutral", 5.5),
    Seg("seg-000003", "src-0003", "neutral", 6.0),
    Seg("seg-000004", "src-0001", "neutral", 6.5, gain_db=-12.0),
    Seg("seg-000005", "src-0002", "neutral", 4.5),
    Seg("seg-000006", "src-0001", "neutral", 2.0),
    Seg("seg-000007", "src-0001", "calm", 4.2),
    Seg("seg-000008", "src-0003", "calm", 4.8),
    Seg("seg-000009", "src-0002", "awake", 3.6),
    Seg("seg-000010", "src-0003", "awake", 3.8),
    Seg("seg-000011", "src-0002", "sleepy", 5.2),
    Seg("seg-000012", "src-0001", "neutral", 4.0, cluster="flagged"),
)
DROPPED = "seg-000013"
DISAGREE = "seg-000005"
FLAGGED = "seg-000012"
LOUD, QUIET = "seg-000001", "seg-000004"


def from_record(seg: Seg, run: str, start: float) -> SegmentRecord:
    return SegmentRecord(
        segment_id=seg.sid, source_id=seg.source, start_s=start, end_s=start + seg.seconds,
        channel="FC", channel_reason=None, separator="none",
        audio=f"seed/runs/{run}/audio/{seg.sid}.flac", diar_cluster="spk-01",
        cluster=seg.cluster, flag_group=None if seg.cluster == "primary" else "src-0001/spk-02",
        sim={"wavlm-sv": 0.9, "ecapa": 0.8}, overlap_ms=0, snr_db=30.0, c50_db=40.0,
        transcripts={}, asr_agreement_cer=0.0,
        prosody=Prosody(f0_mean_hz=210.0, f0_std_hz=4.0, rate_sps=4.0, energy_db=-20.0),
        arousal=0.4, state_proposed=seg.state, state=None, set="short", kept=True, reasons=(),
    )  # fmt: skip


def dropped_record(run: str) -> SegmentRecord:
    base = from_record(Seg(DROPPED, "src-0002", "neutral", 3.0), run, 100.0)
    return replace(
        base, audio="", state_proposed="", kept=False, reasons=("overlap",),
        prosody=Prosody(f0_mean_hz=0.0, f0_std_hz=0.0, rate_sps=0.0, energy_db=0.0),
    )  # fmt: skip


def make_root(base: Path, segs: Sequence[Seg], run: str = "r1") -> root_mod.Root:
    """A private root holding an extract run of `segs`, FLAC PCM_24 at the source rate."""
    root = root_mod.init_root(base / "voice-root", "private")
    records = []
    for k, seg in enumerate(segs):
        record = from_record(seg, run, 10.0 * k)
        path = root.path / record.audio
        path.parent.mkdir(parents=True, exist_ok=True)
        clip = cut(SOURCE_RATE, seg.seconds, k + 1, seg.gain_db)
        sf.write(str(path), clip, SOURCE_RATE, format="FLAC", subtype="PCM_24")
        records.append(record)
    records.append(dropped_record(run))
    write_jsonl(root, f"seed/runs/{run}/segments.jsonl", records)
    return root


class ScriptedAsr:
    """`FakeAsr` with a text per clip length: the clip of `seconds` gets `texts[seconds]`."""

    def __init__(self, name: str, text: str, texts: Mapping[float, str] | None = None) -> None:
        self.name = name
        self.text = text
        self.texts = dict(texts or {})

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Transcript:
        text = self.texts.get(round(len(audio) / sample_rate, 1), self.text)
        return FakeAsr(self.name, text).transcribe(audio, sample_rate)


def quick_prosody(audio: np.ndarray, sample_rate: int, words: Sequence[Word]) -> Prosody:
    rate = prosody_mod.rate_from_words(words) if words else -1.0
    return Prosody(
        f0_mean_hz=210.0, f0_std_hz=4.0, rate_sps=rate, energy_db=energy(audio, sample_rate)
    )


def models(
    steps: Sequence[FakeEnhancer] = (), disagree_s: float | None = None
) -> clean_stage.Models:
    texts = {disagree_s: "an entirely different line of words"} if disagree_s else {}
    return clean_stage.Models(
        embedders=embedders(),
        enhancers=tuple(steps),
        timing=ScriptedAsr("whisperx-large-v3", CANARY.transcript),
        verbatim=ScriptedAsr("crisperwhisper", CANARY.transcript, texts),
        prosody=quick_prosody,
    )


class StageDocker:
    """A `DockerOps` whose `run_logged` runs the stage in this process with the given models."""

    def __init__(self, root: root_mod.Root, stage_models: clean_stage.Models) -> None:
        self.root = root
        self.models = stage_models
        self.argvs: list[list[str]] = []

    def capture(self, argv: Sequence[str]) -> str:
        raise AssertionError("a cpu clean job runs no GPU preflight")

    def check(self, argv: Sequence[str]) -> None:
        raise AssertionError("clean builds nothing")

    def run_logged(self, argv: Sequence[str], log: Path) -> int:
        self.argvs.append(list(argv))
        at = list(argv).index("-m")
        module, rest = argv[at + 1], list(argv[at + 2 :])
        assert rest[:2] == ["--data", "/data"]
        rest[1] = str(self.root.path)
        stage = importlib.import_module(module)
        with (
            pytest.MonkeyPatch.context() as mp,
            log.open("w", encoding="utf-8") as handle,
            contextlib.redirect_stdout(handle),
            contextlib.redirect_stderr(handle),
        ):
            mp.setattr(stage, "load_models", lambda device: self.models)
            return stage.main(rest)


class _Capture:
    def __init__(self) -> None:
        self.parts: list[str] = []

    def write(self, text: str) -> int:
        self.parts.append(text)
        return len(text)

    def flush(self) -> None:
        pass

    def getvalue(self) -> str:
        return "".join(self.parts)


@contextlib.contextmanager
def _patched(docker: StageDocker, executors: list[str]) -> Iterator[None]:
    real = jobs.executor_for

    def spy(name: str, **kwargs: object) -> jobs.Executor:
        executors.append(name)
        return real(name, **kwargs)  # type: ignore[arg-type]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(clean_cmd, "make_docker", lambda: docker)
        mp.setattr(clean_cmd, "image_tag", lambda family: IMAGE)
        mp.setattr(jobs, "executor_for", spy)
        yield


def run_clean(
    root: root_mod.Root, docker: StageDocker, *args: str, executors: list[str] | None = None
) -> tuple[int, str, str]:
    out, err = _Capture(), _Capture()
    argv = ["--root", str(root.path), "clean", *args]
    with (
        _patched(docker, [] if executors is None else executors),
        contextlib.redirect_stdout(out),
        contextlib.redirect_stderr(err),
    ):
        code = cli.main(argv)
    return code, out.getvalue(), err.getvalue()


ARGS = ("--run", "c1", "--from", "r1", "--target", "t1")


@dataclass
class Ran:
    root: root_mod.Root
    code: int
    out: str
    err: str
    docker: StageDocker
    executors: list[str]

    @property
    def run_dir(self) -> Path:
        return self.root.path / "seed" / "runs" / "c1"

    def records(self) -> dict[str, SegmentRecord]:
        rows = read_jsonl(self.run_dir / "segments.jsonl", SegmentRecord)
        return {r.segment_id: r for r in rows}

    def json(self, name: str) -> dict:
        return json.loads((self.run_dir / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def main_run(tmp_path_factory: pytest.TempPathFactory) -> Ran:
    root = make_root(tmp_path_factory.mktemp("clean"), SEGS)
    steps = (FakeEnhancer("dehum", dehum), FakeEnhancer("decomb", decomb))
    docker = StageDocker(root, models(steps, disagree_s=4.5))
    executors: list[str] = []
    code, out, err = run_clean(root, docker, *ARGS, executors=executors)
    return Ran(root, code, out, err, docker, executors)


def source_clip(ran: Ran, sid: str) -> np.ndarray:
    audio, rate = sf.read(str(ran.root.path / "seed" / "runs" / "r1" / "audio" / f"{sid}.flac"))
    return clean_stage.resample(audio.astype(np.float32), rate, RATE)


def clean_clip(ran: Ran, sid: str) -> np.ndarray:
    audio, rate = sf.read(str(ran.root.path / ran.records()[sid].audio), dtype="float32")
    assert rate == RATE
    return audio


def test_the_clean_run_exits_0(main_run: Ran) -> None:
    assert main_run.code == 0, main_run.err


def test_gates_json_records_each_step_decision_with_both_measures(main_run: Ran) -> None:
    steps = main_run.json("gates.json")["steps"]
    assert [(s["step"], s["kept"]) for s in steps] == [("dehum", True), ("decomb", False)]
    for step in steps:
        assert set(step["sim"]) == set(step["threshold"]) == {"wavlm-sv", "ecapa"}
        assert isinstance(step["spectrum_db"], float)
        assert step["spectrum_max_db"] == SPECTRUM_MAX_DB
    assert steps[0]["reasons"] == []
    assert "sim:wavlm-sv" in steps[1]["reasons"]


def test_the_kept_step_is_applied_and_the_dropped_one_is_not(main_run: Ran) -> None:
    clip = clean_clip(main_run, "seg-000002")
    before = source_clip(main_run, "seg-000002")
    gain = 10.0 ** (main_run.json("gates.json")["loudness"]["gains_db"]["src-0002"] / 20.0)
    expected = dehum(before, RATE) * gain
    assert float(np.max(np.abs(clip - expected))) < 1e-4


def test_loudness_gain_is_one_value_per_source(main_run: Ran) -> None:
    gains = main_run.json("gates.json")["loudness"]["gains_db"]
    assert set(gains) == {"src-0001", "src-0002", "src-0003"}
    records = main_run.records()
    for seg in SEGS:
        moved = energy(clean_clip(main_run, seg.sid), RATE) - energy(
            source_clip(main_run, seg.sid), RATE
        )
        assert moved == pytest.approx(gains[records[seg.sid].source_id], abs=0.05)


def test_the_energy_difference_of_a_loud_and_a_quiet_segment_survives(main_run: Ran) -> None:
    before = energy(source_clip(main_run, LOUD), RATE) - energy(source_clip(main_run, QUIET), RATE)
    after = energy(clean_clip(main_run, LOUD), RATE) - energy(clean_clip(main_run, QUIET), RATE)
    assert before == pytest.approx(12.0, abs=0.5)
    assert after == pytest.approx(before, abs=0.05)


def test_output_is_24khz_mono_flac(main_run: Ran) -> None:
    kept = [r for r in main_run.records().values() if r.kept]
    assert len(kept) == len(SEGS)
    for record in kept:
        assert record.audio == f"seed/runs/c1/audio/{record.segment_id}.flac"
        info = sf.info(str(main_run.root.path / record.audio))
        assert (info.format, info.channels, info.samplerate) == ("FLAC", 1, 24000)


def test_the_extract_run_is_left_untouched_and_dropped_rows_are_carried(main_run: Ran) -> None:
    before = read_jsonl(main_run.root.path / "seed/runs/r1/segments.jsonl", SegmentRecord)
    assert all(r.transcripts == {} for r in before)
    after = main_run.records()
    assert list(after) == [r.segment_id for r in before]
    assert after[DROPPED] == next(r for r in before if r.segment_id == DROPPED)


def test_two_asr_transcripts_and_their_agreement_are_recorded(main_run: Ran) -> None:
    record = main_run.records()["seg-000001"]
    assert record.transcripts == {"whisperx": CANARY.transcript, "crisper": CANARY.transcript}
    assert record.asr_agreement_cer == 0.0
    assert transcribe.training_text(record) == CANARY.transcript


def test_a_disagreeing_segment_is_marked_and_kept_out_of_training_text(main_run: Ran) -> None:
    record = main_run.records()[DISAGREE]
    assert record.asr_agreement_cer > transcribe.AGREEMENT_CER_MAX
    assert record.reasons == (transcribe.ASR_DISAGREE,)
    assert record.kept
    assert transcribe.training_text(record) is None
    assert main_run.json("report.json")["segments"]["asr_disagree"] == 1


def test_word_times_feed_the_prosody_rate(main_run: Ran) -> None:
    for record in main_run.records().values():
        if not record.kept:
            continue
        words = (
            FakeAsr("x", CANARY.transcript)
            .transcribe(clean_clip(main_run, record.segment_id), RATE)
            .words
        )
        assert record.prosody.rate_sps == pytest.approx(prosody_mod.rate_from_words(words))


def test_references_are_written_under_targets_as_wav_pcm16_mono_24khz(main_run: Ran) -> None:
    refs = main_run.json("refs.json")
    medians = refs["medians"]
    written = 0
    for state, entry in refs["states"].items():
        folder = main_run.root.path / "targets" / "t1" / "refs" / state
        if entry["short"]:
            assert entry["refs"] == [] and not folder.exists()
            continue
        assert 2 <= len(entry["refs"]) <= 4
        assert sorted(p.name for p in folder.iterdir()) == sorted(
            Path(r["audio"]).name for r in entry["refs"]
        )
        for ref in entry["refs"]:
            assert ref["audio"].startswith(f"refs/{state}/")
            path = main_run.root.path / "targets" / "t1" / ref["audio"]
            info = sf.info(str(path))
            assert (info.format, info.subtype, info.channels, info.samplerate) == (
                "WAV", "PCM_16", 1, 24000,
            )  # fmt: skip
            assert 3.0 <= info.duration <= 12.0
            assert all(ref["sim"][n] >= medians[n] for n in medians)
            written += 1
    assert written >= 2


def test_states_without_two_candidates_are_reported_short(main_run: Ran) -> None:
    refs = main_run.json("refs.json")["states"]
    assert refs["sleepy"]["short"] and refs["sleepy"]["members"] == 1
    assert refs["excited"]["short"] and refs["excited"]["members"] == 0
    assert set(main_run.json("report.json")["short_states"]) >= {"sleepy", "excited"}
    assert re.search(r"short states: .*sleepy", main_run.out)


def test_a_flagged_segment_is_cleaned_but_never_a_reference(main_run: Ran) -> None:
    assert main_run.records()[FLAGGED].kept
    refs = main_run.json("refs.json")["states"]
    assert FLAGGED not in {r["segment_id"] for e in refs.values() for r in e["refs"]}
    assert main_run.json("gates.json")["distribution"]["wavlm-sv"]["n"] == len(SEGS) - 1


def test_clean_prints_step_decisions_and_counts_only(main_run: Ran) -> None:
    _assert_no_canary(main_run.out + main_run.err)
    assert "step dehum: kept" in main_run.out
    assert "step decomb: dropped (sim:wavlm-sv" in main_run.out
    allowed = (
        "job ", "clean run ", "  step ", "  segments:", "  references:", "  short states:",
        "  minutes per state:", "  references written to ",
    )  # fmt: skip
    for line in main_run.out.splitlines():
        assert line.startswith(allowed), line


def test_clean_builds_one_local_job_in_the_extract_image(main_run: Ran) -> None:
    (argv,) = main_run.docker.argvs
    assert argv[:2] == ["docker", "run"]
    assert "--network" in argv and "none" in argv
    assert "--gpus" not in argv
    assert IMAGE in argv and "/opt/venv-extract/bin/python" in argv
    assert argv[argv.index("-m") + 1] == "scripts.voice_data.stages.clean"
    assert argv[-4:] == ["--run", "c1", "--device", "cpu"]
    assert f"{main_run.root.path / 'seed' / 'runs' / 'r1'}:/data/seed/runs/r1:ro" in argv
    assert [a for a in argv if a.endswith(":/data/seed/runs/c1")]
    job = json.loads(next((main_run.root.path / "runs").glob("job-*/job.json")).read_text())
    assert job["family"] == "extract" and job["gpu"] is False and job["network"] is False


def test_clean_runs_on_the_local_executor_only(main_run: Ran) -> None:
    assert main_run.executors == ["local"]
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["clean", *ARGS, "--executor", "remote:x"])


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (ARGS, "already holds a run"),
        (("--run", "c2", "--from", "c2", "--target", "t2"), "differ"),
        (("--run", "c2", "--from", "missing", "--target", "t2"), "segments.jsonl"),
        (("--run", "c2", "--from", "r1", "--target", "t1"), "already holds references"),
        (("--run", "../x", "--from", "r1", "--target", "t2"), "run id"),
        (("--run", "c2", "--from", "r1", "--target", "../t"), "target id"),
    ],
)
def test_clean_refuses_before_any_job(main_run: Ran, args: tuple[str, ...], message: str) -> None:
    executors: list[str] = []
    code, out, err = run_clean(main_run.root, main_run.docker, *args, executors=executors)
    assert code == 2
    assert message in err
    assert executors == [] and len(main_run.docker.argvs) == 1


# Raj's labels.


def test_labels_set_the_state_and_drop_unlisted_segments(tmp_path: Path) -> None:
    segs = SEGS[:3] + (SEGS[6],)
    root = make_root(tmp_path, segs)
    labels = "segment_id,state\nseg-000001,calm\nseg-000002,neutral\nseg-000007,calm\n"
    (root.path / "seed" / "labels.csv").write_text(labels, encoding="utf-8")
    docker = StageDocker(root, models())
    code, out, err = run_clean(root, docker, *ARGS)
    assert code == 0, err
    rows = {
        r.segment_id: r
        for r in read_jsonl(root.path / "seed/runs/c1/segments.jsonl", SegmentRecord)
    }
    assert rows["seg-000001"].state == "calm" and rows["seg-000001"].kept
    assert rows["seg-000002"].state == "neutral"
    assert not rows["seg-000003"].kept and rows["seg-000003"].reasons == ("not-her",)
    assert rows["seg-000003"].audio == ""
    refs = json.loads((root.path / "seed/runs/c1/refs.json").read_text())["states"]
    assert refs["calm"]["members"] == 2 and refs["neutral"]["members"] == 1
    assert "not her 1" in out


def test_labels_that_name_no_segment_of_the_run_are_refused(tmp_path: Path) -> None:
    root = make_root(tmp_path, SEGS[:3])
    (root.path / "seed" / "labels.csv").write_text(
        "segment_id,state\nseg-009999,calm\n", encoding="utf-8"
    )
    code, _, err = run_clean(root, StageDocker(root, models()), *ARGS)
    assert code == 2 and "labels" in err


def test_fewer_than_two_confirmed_segments_are_refused(tmp_path: Path) -> None:
    root = make_root(tmp_path, (SEGS[0], SEGS[-1]))
    code, _, err = run_clean(root, StageDocker(root, models()), *ARGS)
    assert code == 2 and "two" in err


def test_resample_keeps_duration_and_level() -> None:
    clip = _synth_voice(210.0, FORMANTS, 2.0, SOURCE_RATE)
    out = clean_stage.resample(clip, SOURCE_RATE, RATE)
    assert len(out) == 2 * RATE and out.dtype == np.float32
    assert energy(out, RATE) == pytest.approx(energy(clip, SOURCE_RATE), abs=0.1)
    assert clean_stage.resample(clip, RATE, RATE) is clip
