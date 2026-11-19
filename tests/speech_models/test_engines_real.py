"""The picked speech engines on the host, against the generated fixtures.

Each test checks what it needs before any work and skips naming it: the package, the GPU, the
weight file or directory under `MIST_VOICE_MODELS_DIR`, the voice identity variables, or the
generated manifest. Engine libraries are imported in the test bodies only, so an image without
them collects the tier and skips it. A weight that is present but fails to load fails the test:
that is a broken host, not a missing engine.

Asserted: Silero's onset within 100 ms of the manifest's, Smart Turn right on at least 80% of
the manifest's end-of-turn probes, a word error rate of at most 0.15 per STT engine, onnxruntime
1.31.1, and each picked arm's `packages` in `speech_arms.json` installed at their versions.
Recorded without a verdict (the `speech_models records` section): STT partial and final decode
times, from which the caption lag follows; each GPU engine's VRAM as `build_speech_engines`
measured it; what STT made of the noise bursts and whether ingress's drop rule drops it; the
keyword-over-echo transcripts; and Pocket's first-chunk time, real-time factor and peak per
segment length.

Word timings, per STT engine. On the generated `wake` fixtures, the final's first word must
overlap the fixture's name span within 160 ms. `MIST_VOICE_WORDS_DIR`, when set, names a directory
of 16 kHz mono PCM16 WAVs, each a name said alone, synthetic or public only; the test that reads
it skips naming the variable when it is unset. Each clip's name span runs from its first to its
last 10 ms frame within 30 dB of its loudest. Each clip is decoded alone, with 500 ms of silence
before it and 1000 ms after, and followed 150 ms later by the next clip (the last by the first),
padded the same. The word overlapping the name span most must overlap the 400 ms alias window
before a hit 100 ms and a hit 180 ms after the name's end. Its records hold counts and the name
word's start and end errors in ms, never a transcript. Parakeet's last word ends where the
decoded audio ends, so a lone clip's end error is about its 1000 ms of padding; the pairs' first
names measure its word ends.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import importlib.util
import json
import os
import re
import statistics
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest

from backend.realtime.budget import EndpointingConfig, EngineOption, VoiceSettings
from backend.realtime.errors import SpeechEngineError
from backend.realtime.speech import CancelToken, StreamingTranscriber, TextSegment, WordTiming
from scripts.model_bench.speech.wer import word_error_rate
from scripts.voice_fixtures.generate_fixtures import (
    FixtureEntry,
    FixtureError,
    Manifest,
    Span,
    read_wav,
    samples,
)

pytestmark = pytest.mark.speech_models

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEECH_ARMS = REPO_ROOT / "scripts" / "model_bench" / "speech" / "speech_arms.json"

ONNXRUNTIME_PIN = "1.31.1"
ONSET_TOLERANCE_MS = 100
EOT_MIN_ACCURACY = 0.8
MAX_WER = 0.15

RATE = 16000
FRAME_SAMPLES = RATE // 50
VAD_WINDOW = 512
WER_SCENARIOS = ("complete", "pause", "hesitation", "wake")

# Word timings. A wake fixture's first word must overlap its name span widened by this much.
WAKE_NAME_TOLERANCE_MS = 160
# The synthetic name clips (H33): the variable, the name-span rule, and how each is decoded.
WORDS_DIR_VARIABLE = "MIST_VOICE_WORDS_DIR"
NAME_SPAN_FRAME_MS = 10
NAME_SPAN_DB = 30.0
NAME_LEAD_MS = 500
NAME_TAIL_MS = 1000
NAME_PAIR_GAP_MS = 150
# The measured median and maximum lag of a spotter hit after the name's speech ended.
HIT_LAGS_MS = (100, 180)
# Design 4's alias window, the stretch before a hit where the gate reads a near spelling of the
# name. T69 adds it as the wake setting `alias_window_ms`, default 400; this literal stands in.
ALIAS_WINDOW_MS = 400

# An invented passage, cut at word boundaries into segments of about each length.
TTS_PASSAGE = (
    "The kettle is on and the rain has eased, so the walk can start once the bread is out of "
    "the oven and the table by the window has been cleared of maps and string and pencils."
)
TTS_LENGTHS = (10, 20, 30, 40, 120)
TTS_REPEATS = 3

# The picked arms and the module each one's library imports as.
PICKED_ARMS = (
    ("tts", "pocket-clone", "pocket_tts"),
    ("stt", "nemotron-160", "nemo"),
    ("stt", "parakeet-cpu", "onnx_asr"),
)


def _require_module(module: str, distribution: str) -> None:
    """Skip naming the package when `module` is not installed, without importing it."""
    if importlib.util.find_spec(module) is None:
        pytest.skip(f"package {distribution} is not installed (no module {module!r})")


def _require_onnx_model(settings: VoiceSettings, setting: str, variable: str) -> None:
    """Skip naming the weight file when the ONNX model a setting names is not staged."""
    from backend.realtime.engines.silero_vad import resolve_model_path

    try:
        resolve_model_path(settings.placement.models_dir, setting, variable)
    except SpeechEngineError as exc:
        pytest.skip(f"weight missing: {exc}")


def _entries(manifest: Callable[[], Manifest], *scenarios: str) -> list[FixtureEntry]:
    entries = [entry for entry in manifest().fixtures if entry.scenario in scenarios]
    if not entries:
        pytest.fail(f"the manifest has no {', '.join(scenarios)} fixtures")
    return entries


def _peak_dbfs(audio: np.ndarray) -> float:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return round(20 * np.log10(peak), 2) if peak > 0 else float("-inf")


# -- Packages ---------------------------------------------------------------------------------


def test_onnxruntime_is_the_pinned_version():
    onnxruntime = pytest.importorskip("onnxruntime", reason="package onnxruntime is not installed")

    assert onnxruntime.__version__ == ONNXRUNTIME_PIN


@pytest.mark.parametrize(("slot", "arm", "module"), PICKED_ARMS, ids=[a for _, a, _ in PICKED_ARMS])
def test_picked_arm_packages_are_installed_at_their_versions(slot, arm, module):
    pytest.importorskip(module, reason=f"package for {arm} is not installed (module {module!r})")
    packages = json.loads(SPEECH_ARMS.read_text(encoding="utf-8"))[slot][arm]["packages"]

    installed = {}
    for requirement in packages:
        # `nemo_toolkit[asr]` is installed as `nemo_toolkit`; the extra is not a distribution.
        name = re.sub(r"\[.*\]$", "", requirement)
        try:
            installed[requirement] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed[requirement] = None

    assert installed == packages


# -- Silero VAD and Smart Turn ----------------------------------------------------------------


def _onset_ms(stream, audio: np.ndarray, endpointing: EndpointingConfig) -> int | None:
    """The first window of the first run of speech windows long enough to open a segment.

    The run rule is ingress's (`SpeechSegmentTracker`): `speech_start_windows` consecutive
    windows at or above `vad_threshold`, the segment starting at the run's first window.
    """
    run = 0
    run_start = 0
    for index in range(len(audio) // VAD_WINDOW):
        window = audio[index * VAD_WINDOW : (index + 1) * VAD_WINDOW]
        if stream.score(window) >= endpointing.vad_threshold:
            if run == 0:
                run_start = index
            run += 1
            if run >= endpointing.speech_start_windows:
                return run_start * VAD_WINDOW * 1000 // RATE
        else:
            run = 0
    return None


def test_silero_onset_is_within_100_ms_of_the_manifest(
    voice_settings, manifest, fixture_audio, speech_record
):
    _require_module("onnxruntime", "onnxruntime")
    placement = voice_settings.placement
    _require_onnx_model(voice_settings, placement.vad_model, "MIST_VOICE_VAD_MODEL")
    fixtures = manifest().fixtures
    from backend.realtime.engines.silero_vad import (
        SileroVad,
        onnxruntime_providers,
        open_onnx_session,
    )

    vad = SileroVad.load(
        placement, session_factory=open_onnx_session, available_providers=onnxruntime_providers
    )
    errors = {}
    for entry in fixtures:
        if entry.onset_ms is None:
            continue
        detected = _onset_ms(vad.open_stream(), fixture_audio(entry), voice_settings.endpointing)
        error = None if detected is None else detected - entry.onset_ms
        errors[entry.id] = error
        speech_record("silero_onset", fixture=entry.id, expected_ms=entry.onset_ms, error_ms=error)

    assert errors, "the manifest has no fixture with an onset"
    late = {
        key: error
        for key, error in errors.items()
        if error is None or abs(error) > ONSET_TOLERANCE_MS
    }
    assert not late, f"onset off by more than {ONSET_TOLERANCE_MS} ms (None: not detected): {late}"


def test_smart_turn_is_right_on_at_least_80_percent_of_the_probes(
    voice_settings, manifest, fixture_audio, speech_record
):
    _require_module("onnxruntime", "onnxruntime")
    _require_module("transformers", "transformers")
    placement = voice_settings.placement
    _require_onnx_model(voice_settings, placement.eot_model, "MIST_VOICE_EOT_MODEL")
    fixtures = manifest().fixtures
    from backend.realtime.engines.silero_vad import onnxruntime_providers, open_onnx_session
    from backend.realtime.engines.smart_turn import SmartTurnDetector, whisper_features

    detector = SmartTurnDetector.load(
        placement,
        voice_settings.endpointing,
        session_factory=open_onnx_session,
        available_providers=onnxruntime_providers,
        feature_extractor_factory=whisper_features,
    )
    verdicts = []
    for entry in fixtures:
        audio = fixture_audio(entry)
        for probe in entry.eot_probes:
            start = time.perf_counter()
            probability = detector.probability_complete(audio[: samples(probe.at_ms)])
            elapsed_ms = (time.perf_counter() - start) * 1000
            complete = probability > voice_settings.endpointing.eot_threshold
            verdicts.append(complete == probe.complete)
            speech_record(
                "smart_turn",
                fixture=entry.id,
                at_ms=probe.at_ms,
                expected_complete=probe.complete,
                probability=round(probability, 4),
                inference_ms=round(elapsed_ms, 1),
            )

    assert verdicts, "the manifest has no end-of-turn probe"
    accuracy = sum(verdicts) / len(verdicts)
    speech_record("smart_turn_accuracy", accuracy=round(accuracy, 4), probes=len(verdicts))
    assert accuracy >= EOT_MIN_ACCURACY


# -- STT --------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class LoadedStt:
    engine: str
    transcriber: StreamingTranscriber
    vram_mib: int | None


def _load_nemotron(settings: VoiceSettings) -> LoadedStt:
    """Nemotron through `build_speech_engines`, so its VRAM is the backend's own measure."""
    _require_module("torch", "torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("nemotron-160 needs a CUDA GPU, and torch sees none")
    _require_module("nemo", "nemo_toolkit[asr]")
    from backend.realtime.engines.nemotron_stt import WEIGHT_PATH

    weight = Path(settings.placement.models_dir, "speech", *WEIGHT_PATH.split("/"))
    if not weight.is_file():
        pytest.skip(f"weight missing: {weight}")
    from backend.factories import build_speech_engines

    placement = dataclasses.replace(
        settings.placement,
        stt_chain=(EngineOption("nemotron-160", "cuda"),),
        tts_chain=(),
        kws_model=None,
    )
    engines = build_speech_engines(dataclasses.replace(settings, placement=placement))
    if engines.transcriber is None:
        pytest.fail("nemotron-160 did not load; the backend's WARNING above says why")
    return LoadedStt("nemotron-160", engines.transcriber, engines.statuses.stt.vram_mib)


def _load_parakeet(settings: VoiceSettings) -> LoadedStt:
    _require_module("onnx_asr", "onnx-asr")
    _require_module("onnxruntime", "onnxruntime")
    from backend.realtime.engines.parakeet_stt import (
        WEIGHT_PATH,
        ParakeetTranscriber,
        import_onnx_asr,
    )

    weights = Path(settings.placement.models_dir, "speech", *WEIGHT_PATH.split("/"))
    if not weights.is_dir() or not any(weights.iterdir()):
        pytest.skip(f"weight missing: directory {weights} is missing or empty")
    transcriber = ParakeetTranscriber.load(settings.placement, library_loader=import_onnx_asr)
    return LoadedStt("parakeet-cpu", transcriber, None)


@pytest.fixture(scope="module", params=["nemotron-160", "parakeet-cpu"])
def stt(request, voice_settings) -> Iterator[LoadedStt]:
    """Each picked STT engine, loaded once for the module and released after it."""
    if request.param == "nemotron-160":
        loaded = _load_nemotron(voice_settings)
    else:
        loaded = _load_parakeet(voice_settings)
    yield loaded
    loaded.transcriber.release()


@dataclasses.dataclass(frozen=True, slots=True)
class Transcription:
    text: str
    no_speech_prob: float | None
    avg_logprob: float | None
    partial_ms: list[float]
    final_ms: float
    words: tuple[WordTiming, ...] | None = None


def _transcribe(
    transcriber: StreamingTranscriber, audio: np.ndarray, partial_interval_ms: int
) -> Transcription:
    """Push the audio in 20 ms frames, as ingress does, with a partial every interval."""
    stream = transcriber.open_stream()
    every = partial_interval_ms // 20
    partial_ms = []
    try:
        for index in range(0, len(audio), FRAME_SAMPLES):
            stream.push(audio[index : index + FRAME_SAMPLES])
            if (index // FRAME_SAMPLES + 1) % every == 0:
                start = time.perf_counter()
                stream.decode_partial(CancelToken())
                partial_ms.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        final = stream.decode_final(CancelToken())
        final_ms = (time.perf_counter() - start) * 1000
    finally:
        stream.close()
    return Transcription(
        final.text, final.no_speech_prob, final.avg_logprob, partial_ms, final_ms, final.words
    )


def _timing_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "p50_ms": round(statistics.median(ordered), 1),
        "p90_ms": round(ordered[max(0, int(np.ceil(0.9 * len(ordered))) - 1)], 1),
        "max_ms": round(ordered[-1], 1),
    }


def test_stt_word_error_rate_is_at_most_0_15(
    stt, voice_settings, manifest, fixture_audio, speech_record
):
    engine, transcriber = stt.engine, stt.transcriber
    if stt.vram_mib is not None:
        speech_record("stt_vram", engine=engine, vram_mib=stt.vram_mib)
    interval = voice_settings.endpointing.stt_partial_interval_ms
    errors = words = 0
    partials: list[float] = []
    finals: list[float] = []
    for entry in _entries(manifest, *WER_SCENARIOS):
        result = _transcribe(transcriber, fixture_audio(entry), interval)
        score = word_error_rate(entry.text, result.text)
        errors += score.errors
        words += score.reference_words
        partials += result.partial_ms
        finals.append(result.final_ms)
        speech_record(
            "stt_fixture",
            engine=engine,
            fixture=entry.id,
            wer=round(score.wer, 4),
            final_ms=round(result.final_ms, 1),
        )

    wer = errors / words
    speech_record("stt_partial_decode", engine=engine, **_timing_summary(partials))
    speech_record("stt_final_decode", engine=engine, **_timing_summary(finals))
    speech_record("stt_wer", engine=engine, wer=round(wer, 4), errors=errors, words=words)
    assert wer <= MAX_WER


def _ingress_drop(text: str, no_speech_prob, avg_logprob, cfg: EndpointingConfig) -> str | None:
    """Ingress's rule for a final (`ingress.py`, `_on_final`): the drop reason, or None."""
    if not text.strip():
        return "noise"
    if (
        no_speech_prob is not None
        and avg_logprob is not None
        and no_speech_prob > cfg.no_speech_threshold
        and avg_logprob <= cfg.logprob_threshold
    ):
        return "no_speech"
    return None


def test_stt_noise_bursts_are_recorded_with_the_ingress_verdict(
    stt, voice_settings, manifest, fixture_audio, speech_record
):
    engine, transcriber = stt.engine, stt.transcriber
    cfg = voice_settings.endpointing
    for entry in _entries(manifest, "noise"):
        result = _transcribe(transcriber, fixture_audio(entry), cfg.stt_partial_interval_ms)
        speech_record(
            "stt_noise_final",
            engine=engine,
            fixture=entry.id,
            text=result.text,
            no_speech_prob=result.no_speech_prob,
            avg_logprob=result.avg_logprob,
            dropped=_ingress_drop(result.text, result.no_speech_prob, result.avg_logprob, cfg),
        )


def test_stt_keyword_over_echo_transcripts_are_recorded(
    stt, voice_settings, manifest, fixture_audio, speech_record
):
    engine, transcriber = stt.engine, stt.transcriber
    interval = voice_settings.endpointing.stt_partial_interval_ms
    for entry in _entries(manifest, "keyword_echo"):
        result = _transcribe(transcriber, fixture_audio(entry), interval)
        keyword = entry.echo.keyword if entry.echo is not None else entry.text
        speech_record(
            "stt_keyword_over_echo",
            engine=engine,
            fixture=entry.id,
            keyword=keyword,
            text=result.text,
            keyword_heard=keyword.lower() in result.text.lower(),
        )


# -- STT word timings -------------------------------------------------------------------------


def _overlap_ms(word: WordTiming, span: Span) -> int:
    return min(word.end_ms, span.end_ms) - max(word.start_ms, span.start_ms)


def _near(word: WordTiming, span: Span, tolerance_ms: int) -> bool:
    """Whether the word overlaps the span widened by `tolerance_ms` on each side."""
    return word.start_ms < span.end_ms + tolerance_ms and word.end_ms > span.start_ms - tolerance_ms


def _timing_outcome(result: Transcription) -> str | None:
    """`no_timings` for text without words, `no_text` for an empty final, else None."""
    if not result.text.strip():
        return "no_text"
    if result.words is None:
        return "no_timings"
    return None


def test_stt_word_timings_on_the_wake_fixtures(
    stt, voice_settings, manifest, fixture_audio, speech_record
):
    engine, transcriber = stt.engine, stt.transcriber
    interval = voice_settings.endpointing.stt_partial_interval_ms
    outcomes: dict[str, str] = {}
    for entry in _entries(manifest, "wake"):
        result = _transcribe(transcriber, fixture_audio(entry), interval)
        name = entry.speech[0]
        outcome = _timing_outcome(result)
        first = result.words[0] if result.words else None
        fields: dict[str, object] = {}
        if outcome is None and first is None:
            outcome = "no_word"
        elif outcome is None and first is not None:
            outcome = "near" if _near(first, name, WAKE_NAME_TOLERANCE_MS) else "far"
            fields = {
                "start_error_ms": first.start_ms - name.start_ms,
                "end_error_ms": first.end_ms - name.end_ms,
            }
        outcomes[entry.id] = outcome
        speech_record(
            "stt_wake_name_word", engine=engine, fixture=entry.id, outcome=outcome, **fields
        )

    counts = {kind: list(outcomes.values()).count(kind) for kind in sorted(set(outcomes.values()))}
    speech_record("stt_wake_name_words", engine=engine, **counts)
    assert outcomes, "the manifest has no wake fixtures"
    misses = {key: kind for key, kind in outcomes.items() if kind != "near"}
    assert (
        not misses
    ), f"{engine}: the first word is not within {WAKE_NAME_TOLERANCE_MS} ms of the name: {misses}"


def name_span(audio: np.ndarray) -> Span | None:
    """From the first to the last 10 ms frame within `NAME_SPAN_DB` of the loudest frame's RMS.

    None for a clip with no frame above digital silence.
    """
    frame = samples(NAME_SPAN_FRAME_MS)
    count = len(audio) // frame
    if count == 0:
        return None
    frames = audio[: count * frame].astype(np.float64).reshape(count, frame)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    loudest = float(rms.max())
    if loudest <= 0.0:
        return None
    loud = np.flatnonzero(rms >= loudest * 10 ** (-NAME_SPAN_DB / 20))
    return Span(int(loud[0]) * NAME_SPAN_FRAME_MS, (int(loud[-1]) + 1) * NAME_SPAN_FRAME_MS)


def name_word(words: tuple[WordTiming, ...], span: Span) -> WordTiming | None:
    """The word overlapping the name span most, or None when no word overlaps it."""
    best = max(words, key=lambda word: _overlap_ms(word, span), default=None)
    if best is None or _overlap_ms(best, span) <= 0:
        return None
    return best


def in_alias_window(word: WordTiming, hit_ms: int) -> bool:
    """Whether the word overlaps Design 4's alias window, `ALIAS_WINDOW_MS` up to the hit."""
    return word.start_ms <= hit_ms and word.end_ms > hit_ms - ALIAS_WINDOW_MS


def _silence(ms: int) -> np.ndarray:
    return np.zeros(samples(ms), dtype=np.float32)


def _name_clips(root: Path) -> list[tuple[np.ndarray, Span]]:
    """Each WAV under `root` with its name span; a file that does not read fails the test."""
    paths = sorted(root.glob("*.wav"))
    if not paths:
        pytest.fail(f"{WORDS_DIR_VARIABLE} names {root}, which holds no .wav file")
    clips = []
    for path in paths:
        try:
            audio = read_wav(path)
        except (FixtureError, OSError, EOFError) as exc:
            pytest.fail(f"{path.name} does not read as 16 kHz mono PCM16: {exc}")
        span = name_span(audio)
        if span is None:
            pytest.fail(f"{path.name} is silent: it has no name span")
        clips.append((audio, span))
    return clips


def _shifted(span: Span, ms: int) -> Span:
    return Span(span.start_ms + ms, span.end_ms + ms)


def test_stt_word_timings_on_synthetic_names(stt, voice_settings, speech_record):
    root = os.environ.get(WORDS_DIR_VARIABLE)
    if not root:
        pytest.skip(f"variable {WORDS_DIR_VARIABLE} is not set: it names the synthetic name clips")
    engine, transcriber = stt.engine, stt.transcriber
    interval = voice_settings.endpointing.stt_partial_interval_ms
    clips = _name_clips(Path(root))
    lead, tail = _silence(NAME_LEAD_MS), _silence(NAME_TAIL_MS)
    decodes = []
    for index, (clip, _) in enumerate(clips):
        following = clips[(index + 1) % len(clips)][0]
        decodes.append(("alone", index, np.concatenate([lead, clip, tail])))
        decodes.append(
            (
                "pair",
                index,
                np.concatenate([lead, clip, _silence(NAME_PAIR_GAP_MS), following, tail]),
            )
        )

    outcomes: list[tuple[str, str]] = []
    errors: dict[str, list[int]] = {"start": [], "end": []}
    for case, index, audio in decodes:
        span = _shifted(clips[index][1], NAME_LEAD_MS)
        result = _transcribe(transcriber, audio, interval)
        outcome = _timing_outcome(result)
        fields: dict[str, object] = {}
        word = None if outcome is not None else name_word(result.words or (), span)
        if outcome is None and word is None:
            outcome = "no_word"
        elif word is not None:
            hits = [span.end_ms + lag for lag in HIT_LAGS_MS]
            outcome = "in_window" if all(in_alias_window(word, hit) for hit in hits) else "outside"
            fields = {
                "start_error_ms": word.start_ms - span.start_ms,
                "end_error_ms": word.end_ms - span.end_ms,
            }
            errors["start"].append(word.start_ms - span.start_ms)
            errors["end"].append(word.end_ms - span.end_ms)
        outcomes.append((case, outcome))
        speech_record(
            "stt_name_word", engine=engine, case=case, clip=index, outcome=outcome, **fields
        )

    summary: dict[str, object] = {"clips": len(clips), "decodes": len(decodes)}
    for case, outcome in sorted(set(outcomes)):
        summary[f"{case}_{outcome}"] = outcomes.count((case, outcome))
    for edge, values in errors.items():
        if values:
            summary[f"{edge}_error_ms_median"] = round(statistics.median(values), 1)
            summary[f"{edge}_error_ms_max_abs"] = max(abs(value) for value in values)
    speech_record("stt_name_words", engine=engine, **summary)
    assert not [
        kind for _, kind in outcomes if kind == "no_timings"
    ], f"{engine} gave a final with text and no word timings: {summary}"
    assert all(kind == "in_window" for _, kind in outcomes), (
        f"{engine}: a name word misses the alias window of a hit {HIT_LAGS_MS} ms after the "
        f"name: {summary}"
    )


def test_the_name_span_runs_from_the_first_to_the_last_frame_within_30_db_of_the_loudest():
    loud = np.full(samples(200), 0.5, dtype=np.float32)
    within = np.full(samples(100), 0.5 * 10 ** (-29 / 20), dtype=np.float32)
    below = np.full(samples(100), 0.5 * 10 ** (-31 / 20), dtype=np.float32)
    audio = np.concatenate([below, within, _silence(50), loud, below])

    assert name_span(audio) == Span(100, 450)
    assert name_span(_silence(300)) is None
    assert name_span(np.zeros(0, dtype=np.float32)) is None


def test_the_name_word_is_the_word_overlapping_the_name_span_most():
    words = (WordTiming("hey", 0, 300), WordTiming("mist", 300, 800), WordTiming("go", 800, 900))

    assert name_word(words, Span(250, 820)) == words[1]
    assert name_word(words, Span(950, 1000)) is None
    assert name_word((), Span(0, 100)) is None


@pytest.mark.parametrize(
    ("word", "inside"),
    [
        # A name ending at 1000 ms: hits at 1100 and 1180 ms open windows from 700 and 780 ms.
        (WordTiming("mist", 600, 1000), True),
        (WordTiming("mist", 600, 781), True),
        (WordTiming("mist", 600, 780), False),
        (WordTiming("mist", 1100, 1500), True),
        (WordTiming("mist", 1101, 1500), False),
    ],
)
def test_the_alias_window_bound_holds_for_both_hits(word, inside):
    hits = [1000 + lag for lag in HIT_LAGS_MS]

    assert all(in_alias_window(word, hit) for hit in hits) is inside


# -- TTS --------------------------------------------------------------------------------------


def _segments() -> list[str]:
    """The passage's longest prefix at a word boundary not over each length."""
    words = TTS_PASSAGE.split()
    segments = []
    for length in TTS_LENGTHS:
        text = words[0]
        for word in words[1:]:
            if len(text) + 1 + len(word) > length:
                break
            text = f"{text} {word}"
        segments.append(text)
    return segments


def test_pocket_clone_first_chunk_time_per_segment_length(voice_settings, speech_record):
    for variable in ("MIST_VOICE_IDENTITY_ROOT", "MIST_VOICE_TTS_VOICE"):
        if not os.environ.get(variable):
            pytest.skip(f"variable {variable} is not set: pocket-clone speaks a voice identity")
    _require_module("pocket_tts", "pocket-tts")
    placement = voice_settings.placement
    weights = Path(placement.models_dir, "speech", "kyutai")
    if not weights.is_dir():
        pytest.skip(f"weight missing: directory {weights} is missing")
    from backend.realtime.engines.pocket_tts import PocketCloneSynthesizer, import_pocket

    synthesizer = PocketCloneSynthesizer.load(
        placement, voice_settings.profile, library_loader=import_pocket
    )
    try:
        # The first call warms the model; it is recorded apart, as the cold start.
        for repeat, text in [(-1, _segments()[0])] + [
            (n, text) for text in _segments() for n in range(TTS_REPEATS)
        ]:
            start = time.perf_counter()
            first_chunk_ms = None
            chunks = []
            for chunk in synthesizer.synthesize(TextSegment(text, 0, len(text)), CancelToken()):
                if first_chunk_ms is None:
                    first_chunk_ms = (time.perf_counter() - start) * 1000
                chunks.append(chunk.pcm16)
            synthesis_ms = (time.perf_counter() - start) * 1000
            audio = np.frombuffer(b"".join(chunks), dtype="<i2").astype(np.float32) / 32767
            audio_ms = len(audio) * 1000 / synthesizer.sample_rate
            assert audio_ms > 0, f"pocket-clone produced no audio for {text!r}"
            speech_record(
                "tts_segment",
                engine="pocket-clone",
                chars=len(text),
                repeat="cold" if repeat < 0 else repeat,
                first_chunk_ms=round(first_chunk_ms or 0.0, 1),
                synthesis_ms=round(synthesis_ms, 1),
                audio_ms=round(audio_ms, 1),
                rtf=round(synthesis_ms / audio_ms, 3),
                peak_dbfs=_peak_dbfs(audio),
            )
    finally:
        synthesizer.release()
