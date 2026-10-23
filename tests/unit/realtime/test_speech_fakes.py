"""Tests for the speech engine contracts and the deterministic fakes every later test uses.

The engine protocols are the seam between the realtime layer and the model adapters that later
tasks add in parallel, so their method names, parameters and value-type fields are pinned here:
an adapter written against a drifted contract would only fail when wired together. The fakes
replace real models in every ingress, controller and session test, so each one is pinned to be
deterministic: the same input or seed gives the same output, run after run.
"""

import ast
import dataclasses
import inspect
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from backend.realtime import speech
from backend.realtime.speech import (
    CancelToken,
    SynthChunk,
    TextSegment,
    TranscriptUpdate,
    VoiceInfo,
    VoiceReference,
)
from tests.fakes.scripted_audio import (
    SAMPLE_RATE,
    scripted_audio,
    silence,
    speech_burst,
    to_pcm16,
    windows,
)
from tests.fakes.speech import (
    BUILTIN_VOICE,
    EnergyVad,
    FakeSynthesizer,
    ScriptedTranscriber,
    ScriptedTurnEnd,
    ScriptedUtterance,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _fields(cls: type) -> list[tuple[str, object]]:
    return [(field.name, field.type) for field in dataclasses.fields(cls)]


def _params(method: object) -> list[str]:
    return list(inspect.signature(method).parameters)


class TestEnginesPackage:
    def test_engines_init_holds_a_docstring_only(self):
        source = (REPO_ROOT / "backend/realtime/engines/__init__.py").read_text(encoding="utf-8")

        body = ast.parse(source).body

        assert len(body) == 1
        assert isinstance(body[0], ast.Expr)
        assert isinstance(body[0].value, ast.Constant)
        assert isinstance(body[0].value.value, str)


class TestValueTypes:
    def test_transcript_update_fields_match_the_design_doc(self):
        assert _fields(TranscriptUpdate) == [
            ("text", str),
            ("stable_chars", int),
            ("is_final", bool),
            ("audio_ms", int),
            ("no_speech_prob", float | None),
            ("avg_logprob", float | None),
        ]

    def test_transcript_update_whisper_scores_default_to_none(self):
        update = TranscriptUpdate(text="hi", stable_chars=0, is_final=False, audio_ms=320)

        assert update.no_speech_prob is None
        assert update.avg_logprob is None

    def test_text_segment_and_synth_chunk_fields_match_the_design_doc(self):
        assert _fields(TextSegment) == [("text", str), ("start_char", int), ("end_char", int)]
        assert _fields(SynthChunk) == [
            ("pcm16", bytes),
            ("samples", int),
            ("segment", TextSegment),
        ]

    def test_voice_reference_fields_match_the_design_doc(self):
        assert _fields(VoiceReference) == [
            ("profile", str),
            ("wav", bytes),
            ("sha256", str),
            ("seconds", float),
            ("exaggeration", float),
            ("cfg_weight", float),
        ]

    def test_voice_reference_repr_leaves_out_the_reference_audio(self):
        reference = VoiceReference(
            profile="p",
            wav=b"RIFF-secret",
            sha256="ab",
            seconds=6.0,
            exaggeration=0.5,
            cfg_weight=0.5,
        )

        assert "RIFF-secret" not in repr(reference)

    def test_voice_info_fields_match_the_design_doc(self):
        assert _fields(VoiceInfo) == [
            ("profile", str | None),
            ("source", Literal["profile", "builtin"]),
            ("seconds", float | None),
            ("sha256_prefix", str | None),
        ]

    @pytest.mark.parametrize(
        "cls",
        [TranscriptUpdate, TextSegment, SynthChunk, VoiceReference, VoiceInfo],
        ids=lambda cls: cls.__name__,
    )
    def test_value_types_are_frozen_and_slotted(self, cls):
        params = cls.__dataclass_params__

        assert params.frozen
        assert "__slots__" in cls.__dict__

    def test_phase_names_the_five_controller_phases(self):
        assert speech.Phase == Literal["idle", "held", "thinking", "speaking", "text"]


class TestProtocols:
    def test_vad_protocols_match_the_design_doc(self):
        assert _params(speech.VoiceActivityDetector.open_stream) == ["self"]
        assert _params(speech.VadStream.score) == ["self", "window"]
        assert _params(speech.VadStream.reset) == ["self"]

    def test_turn_end_protocol_matches_the_design_doc(self):
        assert _params(speech.TurnEndDetector.probability_complete) == ["self", "audio"]

    def test_transcriber_protocols_match_the_design_doc(self):
        assert speech.StreamingTranscriber.__annotations__ == {"model_id": str}
        assert _params(speech.StreamingTranscriber.open_stream) == ["self"]
        assert _params(speech.TranscriptStream.push) == ["self", "pcm"]
        assert _params(speech.TranscriptStream.decode_partial) == ["self", "token"]
        assert _params(speech.TranscriptStream.decode_final) == ["self", "token"]

    def test_synthesizer_protocol_matches_the_design_doc(self):
        assert speech.SpeechSynthesizer.__annotations__ == {
            "model_id": str,
            "variant": str,
            "sample_rate": int,
            "voice": VoiceInfo,
        }
        assert _params(speech.SpeechSynthesizer.synthesize) == ["self", "segment", "token"]


class TestCancelToken:
    def test_a_new_token_is_clear_and_set_is_sticky(self):
        token = CancelToken()
        clear_before = token.is_set()

        token.set()
        token.set()

        assert clear_before is False
        assert token.is_set() is True


class TestScriptedAudio:
    def test_the_same_seed_gives_identical_audio(self):
        parts = [silence(200), speech_burst(600), silence(400)]

        assert np.array_equal(scripted_audio(parts, seed=7), scripted_audio(parts, seed=7))

    def test_a_different_seed_gives_different_audio(self):
        parts = [speech_burst(300)]

        assert not np.array_equal(scripted_audio(parts, seed=1), scripted_audio(parts, seed=2))

    def test_parts_have_their_requested_length_at_16_khz_as_float32(self):
        audio = scripted_audio([speech_burst(250), silence(125)], seed=0)

        assert audio.dtype == np.float32
        assert audio.shape == (int(0.375 * SAMPLE_RATE),)

    def test_bursts_are_louder_than_silence(self):
        audio = scripted_audio([silence(320), speech_burst(320)], seed=3)
        quiet, loud = audio[:5120], audio[5120:]

        assert np.sqrt(np.mean(loud**2)) > 20 * np.sqrt(np.mean(quiet**2))

    def test_windows_cut_whole_512_sample_windows(self):
        audio = scripted_audio([silence(100)], seed=0)

        cut = windows(audio)

        assert [len(window) for window in cut] == [512, 512, 512]

    def test_pcm16_conversion_clips_and_is_little_endian(self):
        pcm = to_pcm16(np.array([0.0, 1.5, -1.5], dtype=np.float32))

        assert np.frombuffer(pcm, dtype="<i2").tolist() == [0, 32767, -32767]


class TestEnergyVad:
    def test_the_same_window_scores_the_same_every_time(self):
        audio = scripted_audio([speech_burst(320)], seed=5)
        window = windows(audio)[2]
        first, second = EnergyVad().open_stream(), EnergyVad().open_stream()

        assert first.score(window) == second.score(window) == first.score(window)

    def test_bursts_score_above_and_silence_below_the_default_threshold(self):
        audio = scripted_audio([silence(320), speech_burst(320)], seed=5)
        stream = EnergyVad().open_stream()

        scores = [stream.score(window) for window in windows(audio)]

        assert all(score < 0.5 for score in scores[:10])
        assert all(score >= 0.5 for score in scores[10:])

    def test_streams_record_scores_and_resets(self):
        stream = EnergyVad().open_stream()

        stream.score(np.zeros(512, dtype=np.float32))
        stream.reset()

        assert stream.scores == [0.0]
        assert stream.resets == 1


class TestScriptedTranscriber:
    def test_two_runs_over_the_same_audio_give_the_same_updates(self):
        def run() -> list[TranscriptUpdate]:
            transcriber = ScriptedTranscriber(["turn on the lights"], ms_per_word=200)
            stream = transcriber.open_stream()
            updates = []
            for _ in range(3):
                stream.push(np.zeros(3200, dtype=np.float32))
                updates.append(stream.decode_partial(CancelToken()))
            updates.append(stream.decode_final(CancelToken()))
            return updates

        assert run() == run()

    def test_partials_reveal_words_with_audio_and_mark_the_stable_prefix(self):
        transcriber = ScriptedTranscriber(["turn on the lights"], ms_per_word=200)
        stream = transcriber.open_stream()

        stream.push(np.zeros(6400, dtype=np.float32))
        first = stream.decode_partial(CancelToken())
        stream.push(np.zeros(3200, dtype=np.float32))
        second = stream.decode_partial(CancelToken())

        assert first == TranscriptUpdate("turn on", 0, False, 400)
        assert second == TranscriptUpdate("turn on the", len("turn on"), False, 600)

    def test_the_final_carries_the_utterances_scripted_whisper_scores(self):
        utterance = ScriptedUtterance("thanks for watching", no_speech_prob=0.9, avg_logprob=-1.4)
        stream = ScriptedTranscriber([utterance]).open_stream()
        stream.push(np.zeros(1600, dtype=np.float32))

        final = stream.decode_final(CancelToken())

        assert final == TranscriptUpdate("thanks for watching", 19, True, 100, 0.9, -1.4)

    def test_each_stream_takes_the_next_utterance_and_then_empty_text(self):
        transcriber = ScriptedTranscriber(["one", "two"])

        finals = [transcriber.open_stream().decode_final(CancelToken()).text for _ in range(3)]

        assert finals == ["one", "two", ""]
        assert len(transcriber.streams) == 3

    def test_streams_record_what_was_pushed(self):
        stream = ScriptedTranscriber(["hi"]).open_stream()

        stream.push(np.zeros(512, dtype=np.float32))
        stream.push(np.zeros(256, dtype=np.float32))

        assert stream.pushed == [512, 256]


class TestScriptedTurnEnd:
    def test_returns_the_scripted_probabilities_in_order_then_the_last_again(self):
        detector = ScriptedTurnEnd([0.2, 0.9])
        audio = np.zeros(16000, dtype=np.float32)

        results = [detector.probability_complete(audio) for _ in range(4)]

        assert results == [0.2, 0.9, 0.9, 0.9]
        assert detector.calls == [16000] * 4

    def test_two_detectors_with_the_same_script_agree(self):
        audio = np.zeros(800, dtype=np.float32)
        first, second = ScriptedTurnEnd([0.1, 0.4, 0.8]), ScriptedTurnEnd([0.1, 0.4, 0.8])

        assert [first.probability_complete(audio) for _ in range(3)] == [
            second.probability_complete(audio) for _ in range(3)
        ]

    def test_an_empty_script_is_refused(self):
        with pytest.raises(ValueError):
            ScriptedTurnEnd([])


class TestFakeSynthesizer:
    def test_the_same_segment_gives_identical_chunks(self):
        segment = TextSegment("Hello there, Raj.", 0, 17)

        first = list(FakeSynthesizer().synthesize(segment, CancelToken()))
        second = list(FakeSynthesizer().synthesize(segment, CancelToken()))

        assert first == second
        assert first

    def test_chunks_cover_the_segment_in_200_ms_pieces_at_24_khz(self):
        segment = TextSegment("x" * 10, 0, 10)
        synthesizer = FakeSynthesizer(ms_per_char=50)

        chunks = list(synthesizer.synthesize(segment, CancelToken()))

        assert [chunk.samples for chunk in chunks] == [4800, 4800, 2400]
        assert all(len(chunk.pcm16) == 2 * chunk.samples for chunk in chunks)
        assert all(chunk.segment == segment for chunk in chunks)

    def test_a_set_token_stops_synthesis_between_chunks(self):
        token = CancelToken()
        chunks = FakeSynthesizer(ms_per_char=50).synthesize(TextSegment("x" * 20, 0, 20), token)

        first = next(chunks)
        token.set()

        assert first.samples == 4800
        assert list(chunks) == []

    def test_a_token_set_before_synthesis_yields_nothing(self):
        token = CancelToken()
        token.set()

        assert list(FakeSynthesizer().synthesize(TextSegment("hello", 0, 5), token)) == []

    def test_the_per_chunk_delay_is_slept_before_each_chunk(self):
        slept: list[float] = []
        synthesizer = FakeSynthesizer(ms_per_char=50, chunk_delay_s=0.05, sleep=slept.append)

        chunks = list(synthesizer.synthesize(TextSegment("x" * 10, 0, 10), CancelToken()))

        assert slept == [0.05] * len(chunks)

    def test_defaults_are_turbo_and_the_builtin_voice_and_both_are_settable(self):
        synthesizer = FakeSynthesizer()
        defaults = (synthesizer.variant, synthesizer.voice, synthesizer.sample_rate)
        profile_voice = VoiceInfo("jarvis", "profile", 12.5, "0123abcd")

        synthesizer.variant = "original"
        synthesizer.voice = profile_voice

        assert defaults == ("turbo", BUILTIN_VOICE, 24000)
        assert VoiceInfo(None, "builtin", None, None) == BUILTIN_VOICE
        assert (synthesizer.variant, synthesizer.voice) == ("original", profile_voice)

    def test_records_every_requested_segment(self):
        synthesizer = FakeSynthesizer()
        segments = [TextSegment("one", 0, 3), TextSegment("two", 4, 7)]

        for segment in segments:
            list(synthesizer.synthesize(segment, CancelToken()))

        assert synthesizer.segments == segments
