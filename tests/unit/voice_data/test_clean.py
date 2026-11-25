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
import hashlib
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
from scripts.voice_data import labels as labels_mod
from scripts.voice_data import root as root_mod
from scripts.voice_data.bindings.protocols import Transcript, Word
from scripts.voice_data.commands import clean as clean_cmd
from scripts.voice_data.errors import JobError
from scripts.voice_data.labels import LabelRecord
from scripts.voice_data.records import Prosody, SegmentRecord, read_jsonl, write_jsonl
from scripts.voice_data.stages import clean as clean_stage
from scripts.voice_data.stages import prosody as prosody_mod
from scripts.voice_data.stages import references, score, transcribe
from scripts.voice_data.stages.clean import RATE
from scripts.voice_data.states import SPEAKING_STATES

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
            Path(r[key]).name for r in entry["refs"] for key in ("audio", "transcript")
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


def all_refs(ran: Ran) -> list[tuple[str, dict]]:
    states = ran.json("refs.json")["states"]
    return [(state, ref) for state, entry in states.items() for ref in entry["refs"]]


def assert_transcripts_beside_references(ran: Ran) -> None:
    """Each reference WAV has its segment's `crisper` text beside it, in the run and the target."""
    records = ran.records()
    refs = all_refs(ran)
    assert len(refs) >= 2
    for state, ref in refs:
        audio = Path(ref["audio"])
        assert ref["transcript"] == f"refs/{state}/{audio.stem}.private.txt"
        expected = records[ref["segment_id"]].transcripts["crisper"].encode("utf-8")
        assert expected.strip()
        for folder in (ran.run_dir, ran.root.path / "targets" / "t1"):
            assert (folder / ref["audio"]).is_file()
            assert (folder / ref["transcript"]).read_bytes() == expected


def assert_no_transcript_outside_its_files(ran: Ran) -> None:
    """The planted transcript is in no printed line and no file but the two that hold it."""
    _assert_no_canary(ran.out + ran.err)
    needle = CANARY.transcript.lower().encode("utf-8")
    checked = 0
    for path in ran.root.path.rglob("*"):
        if not path.is_file() or path.name == "segments.jsonl":
            continue
        if path.name.endswith(".private.txt"):
            continue
        assert needle not in path.read_bytes().lower(), path.relative_to(ran.root.path)
        checked += 1
    assert list((ran.root.path / "runs").glob("job-*/*")), "the job log was not scanned"
    assert checked > 0


def test_each_reference_has_its_verbatim_transcript_beside_it(main_run: Ran) -> None:
    assert_transcripts_beside_references(main_run)


def test_reference_transcripts_are_listed_in_the_job_outputs(main_run: Ran) -> None:
    outputs = main_run.json("outputs.json")["files"]
    listed = {f["rel"]: f["sha256"] for f in outputs}
    for _, ref in all_refs(main_run):
        rel = f"seed/runs/c1/{ref['transcript']}"
        data = (main_run.root.path / rel).read_bytes()
        assert listed[rel] == hashlib.sha256(data).hexdigest()


def test_a_disagreeing_segment_is_never_a_reference(main_run: Ran) -> None:
    assert transcribe.ASR_DISAGREE in main_run.records()[DISAGREE].reasons
    assert DISAGREE not in {ref["segment_id"] for _, ref in all_refs(main_run)}
    neutral = main_run.json("refs.json")["states"]["neutral"]
    assert neutral["members"] == 6


def test_the_transcript_appears_only_in_segments_and_reference_transcripts(main_run: Ran) -> None:
    assert_no_transcript_outside_its_files(main_run)


TEXT_SEGS = SEGS[:5] + SEGS[6:8]


@pytest.fixture(scope="module")
def text_run(tmp_path_factory: pytest.TempPathFactory) -> Ran:
    """A run whose segments each have their own text, non-ASCII included, both ASRs agreeing."""
    root = make_root(tmp_path_factory.mktemp("clean-texts"), TEXT_SEGS)
    texts = {s.seconds: f"{CANARY.transcript} café — {s.sid}" for s in TEXT_SEGS}
    docker = StageDocker(
        root,
        clean_stage.Models(
            embedders=embedders(),
            enhancers=(FakeEnhancer("dehum", dehum),),
            timing=ScriptedAsr("whisperx-large-v3", CANARY.transcript, texts),
            verbatim=ScriptedAsr("crisperwhisper", CANARY.transcript, texts),
            prosody=quick_prosody,
        ),
    )
    executors: list[str] = []
    code, out, err = run_clean(root, docker, *ARGS, executors=executors)
    return Ran(root, code, out, err, docker, executors)


def test_each_reference_transcript_is_its_own_segments_text(text_run: Ran) -> None:
    assert text_run.code == 0, text_run.err
    assert_transcripts_beside_references(text_run)
    for _, ref in all_refs(text_run):
        text = (text_run.run_dir / ref["transcript"]).read_text(encoding="utf-8")
        assert text.endswith(ref["segment_id"])


def test_per_segment_transcripts_appear_only_where_they_belong(text_run: Ran) -> None:
    assert_no_transcript_outside_its_files(text_run)


def test_placing_references_refuses_a_transcript_path_off_the_layout(tmp_path: Path) -> None:
    root = root_mod.init_root(tmp_path / "voice-root", "private")
    run_dir = "seed/runs/c1"
    for name in ("1.wav", "1.private.txt", "2.private.txt"):
        root.write_bytes(f"{run_dir}/refs/calm/{name}", b"x")
    for transcript in (None, "refs/calm/2.private.txt", "refs/neutral/1.private.txt", "../x.txt"):
        ref = {"audio": "refs/calm/1.wav", "segment_id": "seg-000001"}
        if transcript is not None:
            ref["transcript"] = transcript
        refs = {"states": {"calm": {"refs": [ref]}}}
        with pytest.raises(JobError):
            clean_cmd.place_references(root, run_dir, "targets/t1", refs)
    assert not (root.path / "targets" / "t1").exists()


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
        "  minutes per state:", "  references written to ", "  clean minutes ",
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


# Raj's labels: `seed/labels/<run>/labels.jsonl`, read through `labels.read_labels`.


def seg_label(
    seg: Seg,
    k: int,
    *,
    run: str = "r1",
    voices: str = "her",
    energy: int = 3,
    mood: int = 3,
    tags: tuple[str, ...] = (),
    keep: tuple[float, float] | None = None,
    state: str | None = None,
    removed: str | None = None,
    carried_from: tuple[str, ...] = (),
    sid: str | None = None,
) -> LabelRecord:
    """The label of `seg`, the `k`th segment `make_root` wrote (its span starts at 10 k s)."""
    start = 10.0 * k
    return LabelRecord(
        run=run, segment_id=sid or seg.sid, source_id=seg.source, start_s=start,
        end_s=start + seg.seconds, channel="FC", separator="none",
        duration_s=round(seg.seconds, 2), voices=voices, energy=energy, mood=mood, tags=tags,
        keep_from_s=None if keep is None else keep[0], keep_to_s=None if keep is None else keep[1],
        has_note=False, state=state, removed=removed, carried_from=carried_from,
    )  # fmt: skip


def write_run_labels(root: root_mod.Root, run: str, records: Sequence[LabelRecord]) -> None:
    labels_mod.write_labels(root, run, records, {}, {}, replace=True)


# Ten segments of SEGS, one per removal reason, a trim, an unlabelled one, and a labelled clip
# whose projection is None although extract proposed `calm` for it.
LABELLED_SEGS = SEGS[:10]
TRIM = (1.0, 4.5)


def labelled_labels() -> list[LabelRecord]:
    s = LABELLED_SEGS
    return [
        seg_label(s[0], 0, state="neutral"),
        seg_label(s[1], 1, energy=2, state="calm"),
        # s[2] (seg-000003) has no label.
        seg_label(s[3], 3, voices="her_nearby", keep=TRIM, state="neutral"),
        seg_label(s[4], 4, voices="not_her", removed="not-her"),
        seg_label(s[5], 5, voices="her_overlap", removed="overlap"),
        seg_label(s[6], 6, voices="her_nearby", removed="nearby"),
        seg_label(s[7], 7, mood=5, tags=("warm",), state=None),
        seg_label(s[8], 8, removed="flag-group"),
        seg_label(s[9], 9, energy=4, state="awake"),
    ]


SURVIVORS = ("seg-000001", "seg-000002", "seg-000004", "seg-000008", "seg-000010")
UNPROJECTED = "seg-000008"


@pytest.fixture(scope="module")
def labelled_run(tmp_path_factory: pytest.TempPathFactory) -> Ran:
    root = make_root(tmp_path_factory.mktemp("labelled"), LABELLED_SEGS)
    write_run_labels(root, "r1", labelled_labels())
    docker = StageDocker(root, models())
    code, out, err = run_clean(root, docker, *ARGS)
    return Ran(root, code, out, err, docker, [])


def test_a_labelled_clean_run_exits_0(labelled_run: Ran) -> None:
    assert labelled_run.code == 0, labelled_run.err


def test_the_plan_is_schema_2_with_every_label_but_the_note(labelled_run: Ran) -> None:
    plan = labelled_run.json("clean-plan.json")
    assert plan["schema"] == 2
    labels = plan["labels"]
    assert set(labels) == {r.segment_id for r in labelled_labels()}
    assert labels["seg-000004"] == {
        "voices": "her_nearby", "energy": 3, "mood": 3, "tags": [], "keep_from_s": 1.0,
        "keep_to_s": 4.5, "state": "neutral", "removed": None, "carried_from": [],
    }  # fmt: skip
    assert labels["seg-000008"]["tags"] == ["warm"] and labels["seg-000008"]["state"] is None
    assert all("note" not in label and "has_note" not in label for label in labels.values())


@pytest.mark.parametrize(
    ("sid", "reason"),
    [
        ("seg-000003", "unlabelled"),
        ("seg-000005", "not-her"),
        ("seg-000006", "overlap"),
        ("seg-000007", "nearby"),
        ("seg-000009", "flag-group"),
    ],
)
def test_each_removal_reason_drops_its_segment(labelled_run: Ran, sid: str, reason: str) -> None:
    row = labelled_run.records()[sid]
    assert not row.kept and row.reasons == (reason,) and row.audio == ""


def test_only_the_labelled_survivors_are_cleaned(labelled_run: Ran) -> None:
    kept = {sid for sid, r in labelled_run.records().items() if r.kept}
    assert kept == set(SURVIVORS)
    assert labelled_run.json("gates.json")["distribution"]["wavlm-sv"]["n"] == len(SURVIVORS)


def test_a_her_nearby_clip_is_cut_to_its_keep_range(labelled_run: Ran) -> None:
    row = labelled_run.records()["seg-000004"]
    assert (row.start_s, row.end_s) == pytest.approx((30.0 + TRIM[0], 30.0 + TRIM[1]))
    assert len(clean_clip(labelled_run, "seg-000004")) == round((TRIM[1] - TRIM[0]) * RATE)


def test_the_state_is_the_projection_with_no_fallback(labelled_run: Ran) -> None:
    rows = labelled_run.records()
    assert rows["seg-000001"].state == "neutral" and rows["seg-000002"].state == "calm"
    assert rows["seg-000010"].state == "awake"
    assert rows[UNPROJECTED].state is None and rows[UNPROJECTED].state_proposed == "calm"


def test_a_null_state_is_never_a_reference(labelled_run: Ran) -> None:
    refs = labelled_run.json("refs.json")["states"]
    assert refs["calm"]["members"] == 1 and refs["neutral"]["members"] == 2
    assert refs["awake"]["members"] == 1
    assert UNPROJECTED not in {r["segment_id"] for e in refs.values() for r in e["refs"]}


def test_the_report_lists_all_five_states_and_the_new_keys(labelled_run: Ran) -> None:
    report = labelled_run.json("report.json")
    minutes = report["minutes_per_state"]
    assert list(minutes) == sorted(SPEAKING_STATES)
    assert minutes["excited"] == 0 and minutes["sleepy"] == 0
    assert minutes["neutral"] == pytest.approx((5.0 + 3.5) / 60, abs=1e-3)
    assert minutes["calm"] == pytest.approx(5.5 / 60, abs=1e-3)
    assert report["clean_minutes"] == pytest.approx((5.0 + 5.5 + 3.5 + 4.8 + 3.8) / 60, abs=1e-3)
    segments = report["segments"]
    assert segments["kept"] == len(SURVIVORS) and segments["trimmed"] == 1
    assert segments["removed_by_label"] == {
        "not-her": 1, "flag-group": 1, "overlap": 1, "nearby": 1
    }  # fmt: skip


def test_the_report_tables_are_the_labels_of_the_survivors(labelled_run: Ran) -> None:
    survivors = [r for r in labelled_labels() if r.segment_id in SURVIVORS]
    expected = labels_mod.tables(survivors, None, state_seconds_min=30.0)
    tables = labelled_run.json("report.json")["labels"]
    assert tables == json.loads(json.dumps(expected))
    assert "agreement" not in tables
    assert tables["state_none"] == {"count": 1, "seconds": 4.8}
    assert tables["trimmed_s"] == pytest.approx(3.0)


def test_the_host_prints_the_label_tables_after_its_lines(labelled_run: Ran) -> None:
    _assert_no_canary(labelled_run.out + labelled_run.err)
    lines = labelled_run.out.splitlines()
    tables = labelled_run.json("report.json")["labels"]
    table = labels_mod.table_lines(tables)
    at = lines.index(table[0])
    assert lines[at : at + len(table)] == table
    assert at > max(i for i, line in enumerate(lines) if line.startswith("  short states:"))


def test_the_tables_flag_thin_states_at_the_roots_gate(tmp_path: Path) -> None:
    root = make_root(tmp_path, LABELLED_SEGS)
    (root.path / "config" / "gates.json").write_text(
        json.dumps({"seed.state_seconds_min": 1}), encoding="utf-8"
    )
    write_run_labels(root, "r1", labelled_labels())
    code, _, err = run_clean(root, StageDocker(root, models()), *ARGS)
    assert code == 0, err
    tables = json.loads((root.path / "seed/runs/c1/report.json").read_text())["labels"]
    assert tables["state_seconds_min"] == 1
    assert not tables["states"]["neutral"]["thin"] and tables["states"]["excited"]["thin"]


def test_without_labels_the_report_has_no_tables(main_run: Ran) -> None:
    report = main_run.json("report.json")
    assert report["labels"] is None
    assert report["segments"]["removed_by_label"] == {
        "not-her": 0, "flag-group": 0, "overlap": 0, "nearby": 0
    }  # fmt: skip
    assert report["segments"]["trimmed"] == 0
    assert "excited" not in report["minutes_per_state"]
    seconds = sum(s.seconds for s in SEGS)
    assert report["clean_minutes"] == pytest.approx(seconds / 60, abs=1e-3)


def test_labels_that_name_no_segment_of_the_run_are_refused(tmp_path: Path) -> None:
    root = make_root(tmp_path, SEGS[:3])
    write_run_labels(root, "r1", [seg_label(SEGS[0], 0, sid="seg-009999", state="neutral")])
    executors: list[str] = []
    code, _, err = run_clean(root, StageDocker(root, models()), *ARGS, executors=executors)
    assert code == 2 and "labels" in err and executors == []


def test_the_old_labels_csv_is_refused(tmp_path: Path) -> None:
    root = make_root(tmp_path, SEGS[:3])
    (root.path / labels_mod.LEGACY_LABELS_REL).write_text(
        "segment_id,state\nseg-000001,calm\n", encoding="utf-8"
    )
    executors: list[str] = []
    code, _, err = run_clean(root, StageDocker(root, models()), *ARGS, executors=executors)
    assert code == 2 and labels_mod.LEGACY_LABELS_REL in err and executors == []


def test_a_stale_carried_label_is_refused(tmp_path: Path) -> None:
    segs = SEGS[:3]
    root = make_root(tmp_path, segs)
    earlier = seg_label(segs[0], 0, run="r0", state="neutral")
    write_run_labels(root, "r0", [earlier])
    carried = [
        seg_label(segs[0], 0, state="neutral", carried_from=("r0/seg-000001",)),
        seg_label(segs[1], 1, state="neutral"),
        seg_label(segs[2], 2, state="neutral"),
    ]
    write_run_labels(root, "r1", carried)
    # r0 is imported again with a changed label after r1 carried it.
    write_run_labels(root, "r0", [replace(earlier, energy=4, state="awake")])
    executors: list[str] = []
    code, _, err = run_clean(root, StageDocker(root, models()), *ARGS, executors=executors)
    assert code == 2 and executors == []
    assert "1 carried label" in err and "pack labels --run r1 --replace" in err


def test_the_stage_refuses_a_schema_1_plan(tmp_path: Path) -> None:
    plan = {"schema": 1, "from": "r1", "target": "t1", "thresholds": {}, "labels": None}
    folder = tmp_path / "seed" / "runs" / "c1"
    folder.mkdir(parents=True)
    (folder / "clean-plan.json").write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(clean_stage.StageError, match="schema 2"):
        clean_stage.Cleaning(tmp_path, "c1", models())


def test_references_skip_a_null_state_and_one_outside_the_vocabulary() -> None:
    def candidate(sid: str, state: str | None) -> references.Candidate:
        return references.Candidate(
            segment_id=sid, source_id=sid, state=state, duration_s=5.0,  # type: ignore[arg-type]
            snr_db=30.0, prosody=Prosody(210.0, 4.0, 4.0, -20.0), sim={"e": 0.9},
            transcript="", asr_disagree=False,
        )  # fmt: skip

    choices = references.choose(
        [candidate("a", None), candidate("b", None), candidate("c", "bogus")], {"e": 0.5}
    )
    assert set(choices) == set(SPEAKING_STATES)
    assert all(choice.members == 0 for choice in choices.values())


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
