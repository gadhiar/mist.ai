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
import logging
import math
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from backend.realtime import speech
from backend.realtime.errors import RealtimeError, SpeechEngineError
from backend.realtime.lanes import LaneJobCancelled
from backend.realtime.protocol import Spotter, TtsVoice
from backend.realtime.speech import (
    NEUTRAL_STATE,
    SPEAKING_STATES,
    CancelToken,
    NameHit,
    SpotterCalibration,
    SynthChunk,
    TextSegment,
    TranscriptUpdate,
    VoiceInfo,
    arbitration_key,
    false_accepts_per_hour,
    positive_quantile,
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
    FAKE_VOICE,
    EnergyVad,
    FakeSynthesizer,
    ScriptedNameSpotter,
    ScriptedTranscriber,
    ScriptedTranscriptStream,
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

    def test_text_segment_and_synth_chunk_fields_match_the_interfaces(self):
        assert _fields(TextSegment) == [
            ("text", str),
            ("start_char", int),
            ("end_char", int),
            ("state", str),
        ]
        assert _fields(SynthChunk) == [
            ("pcm16", bytes),
            ("samples", int),
            ("segment", TextSegment),
            ("state", str),
        ]

    def test_a_segment_built_without_a_state_is_neutral(self):
        assert TextSegment("a", 0, 1).state == "neutral"

    def test_a_positional_synth_chunk_still_constructs_and_is_neutral(self):
        segment = TextSegment("a", 0, 1, "calm")

        chunk = SynthChunk(b"\x00\x00", 1, segment)

        assert chunk.state == NEUTRAL_STATE
        assert chunk.segment.state == "calm"

    def test_voice_info_names_an_engine_a_voice_identity_and_the_profile(self):
        assert _fields(VoiceInfo) == [("engine", str), ("voice", str), ("profile", str)]
        assert "identity" in VoiceInfo.__doc__
        assert "stock" not in VoiceInfo.__doc__

    def test_the_tts_voice_wire_model_reports_the_voice_identity_id(self):
        assert "identity" in TtsVoice.__doc__
        assert "stock" not in TtsVoice.__doc__

    def test_the_cloned_voice_reference_is_gone(self):
        assert not hasattr(speech, "VoiceReference")

    def test_name_hit_fields_match_the_interfaces(self):
        assert _fields(NameHit) == [("end_pos", int), ("keyword", str), ("score", float)]

    def test_spotter_calibration_holds_the_two_tables(self):
        table = tuple[tuple[float, float], ...]

        assert _fields(SpotterCalibration) == [
            ("false_accepts", table),
            ("positive_quantile", table),
        ]

    @pytest.mark.parametrize(
        "cls",
        [TranscriptUpdate, TextSegment, SynthChunk, VoiceInfo, NameHit, SpotterCalibration],
        ids=lambda cls: cls.__name__,
    )
    def test_value_types_are_frozen_and_slotted(self, cls):
        params = cls.__dataclass_params__

        assert params.frozen
        assert "__slots__" in cls.__dict__

    def test_phase_names_the_five_controller_phases(self):
        assert speech.Phase == Literal["idle", "held", "thinking", "speaking", "text"]


class TestSpeakingStates:
    def test_the_vocabulary_is_exactly_the_five_states(self):
        assert frozenset({"neutral", "calm", "awake", "sleepy", "excited"}) == SPEAKING_STATES
        assert isinstance(SPEAKING_STATES, frozenset)

    def test_neutral_is_the_fallback_and_in_the_vocabulary(self):
        assert NEUTRAL_STATE == "neutral"
        assert NEUTRAL_STATE in SPEAKING_STATES

    def test_the_segment_and_synthesizer_docstrings_state_the_speaking_state_rules(self):
        for doc in (TextSegment.__doc__, speech.SpeechSynthesizer.__doc__):
            assert "neutral" in doc
            assert "SPEAKING_STATES" in doc

    @pytest.mark.parametrize(
        "protocol",
        [speech.SpeechSynthesizer, speech.StreamingTranscriber],
        ids=lambda protocol: protocol.__name__,
    )
    def test_each_release_docstring_states_the_release_rules(self, protocol):
        doc = protocol.release.__doc__

        assert "second call does nothing" in doc
        assert "SpeechEngineError" in doc

    def test_the_module_docstring_keeps_the_placement_by_configuration_rule(self):
        assert "by configuration alone" in speech.__doc__


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
        assert _params(speech.StreamingTranscriber.release) == ["self"]
        assert _params(speech.TranscriptStream.push) == ["self", "pcm"]
        assert _params(speech.TranscriptStream.decode_partial) == ["self", "token"]
        assert _params(speech.TranscriptStream.decode_final) == ["self", "token"]
        assert _params(speech.TranscriptStream.close) == ["self"]

    def test_synthesizer_protocol_matches_the_design_doc(self):
        assert speech.SpeechSynthesizer.__annotations__ == {
            "model_id": str,
            "engine_id": str,
            "sample_rate": int,
            "voice": VoiceInfo,
            "states": frozenset[str],
        }
        assert _params(speech.SpeechSynthesizer.synthesize) == ["self", "segment", "token"]
        assert (
            _params(speech.SpeechSynthesizer.release)
            == _params(speech.StreamingTranscriber.release)
            == ["self"]
        )

    def test_the_fakes_match_the_engine_protocols(self):
        synthesizer = FakeSynthesizer()
        transcriber = ScriptedTranscriber([])

        for name in speech.SpeechSynthesizer.__annotations__:
            assert hasattr(synthesizer, name)
        assert _params(FakeSynthesizer.synthesize) == ["self", "segment", "token"]
        assert _params(FakeSynthesizer.release) == ["self"]
        assert _params(ScriptedTranscriber.release) == ["self"]
        assert hasattr(transcriber, "model_id")

    def test_name_spotter_protocols_match_the_interfaces(self):
        assert speech.NameSpotter.__annotations__ == {
            "model_id": str,
            "names": tuple[str, ...],
            "window_ms": int,
            "hit_lag_ms": int,
            "calibration": Mapping[Spotter, SpotterCalibration],
        }
        assert _params(speech.NameSpotter.open_stream) == ["self", "threshold"]
        assert _params(speech.NameSpotter.score_window) == ["self", "pcm"]
        assert _params(speech.NameSpotterStream.push) == ["self", "pcm", "pos"]
        assert _params(speech.NameSpotterStream.reset) == ["self", "warm_up"]
        warm_up = inspect.signature(speech.NameSpotterStream.reset).parameters["warm_up"]
        assert (warm_up.kind, warm_up.default) == (inspect.Parameter.KEYWORD_ONLY, True)

    def test_the_scripted_name_spotter_matches_the_protocols(self):
        spotter = ScriptedNameSpotter([NameHit(end_pos=3, keyword="mist", score=0.9)])
        stream = spotter.open_stream(0.5)

        hits = stream.push(bytes(640), 3)

        for name in speech.NameSpotter.__annotations__:
            assert hasattr(spotter, name)
        assert _params(ScriptedNameSpotter.open_stream) == ["self", "threshold"]
        assert _params(ScriptedNameSpotter.score_window) == ["self", "pcm"]
        assert _params(type(stream).push) == ["self", "pcm", "pos"]
        assert _params(type(stream).reset) == ["self", "warm_up"]
        warm_up = inspect.signature(type(stream).reset).parameters["warm_up"]
        assert (warm_up.kind, warm_up.default) == (inspect.Parameter.KEYWORD_ONLY, True)
        assert hits and all(isinstance(hit, NameHit) for hit in hits)


# Two points of a table, so a score between them interpolates and one outside clamps.
TWO_POINTS = ((0.5, 2.0), (0.9, 0.0))
RISING = ((0.5, 0.2), (0.9, 0.6))


class TestCalibrationTables:
    @pytest.mark.parametrize(
        ("lookup", "table", "score", "expected"),
        [
            pytest.param(false_accepts_per_hour, TWO_POINTS, 0.7, 1.0, id="false-accepts-mid"),
            pytest.param(false_accepts_per_hour, TWO_POINTS, 0.6, 1.5, id="false-accepts-quarter"),
            pytest.param(positive_quantile, RISING, 0.7, 0.4, id="quantile-mid"),
            pytest.param(positive_quantile, RISING, 0.8, 0.5, id="quantile-three-quarters"),
        ],
    )
    def test_a_score_between_two_points_interpolates_linearly(self, lookup, table, score, expected):
        assert lookup(score, table) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("lookup", "table", "score", "expected"),
        [
            pytest.param(false_accepts_per_hour, TWO_POINTS, 0.1, 2.0, id="false-accepts-below"),
            pytest.param(false_accepts_per_hour, TWO_POINTS, 0.99, 0.0, id="false-accepts-above"),
            pytest.param(positive_quantile, RISING, 0.1, 0.2, id="quantile-below"),
            pytest.param(positive_quantile, RISING, 0.99, 0.6, id="quantile-above"),
        ],
    )
    def test_a_score_outside_the_table_clamps_to_the_nearest_end(
        self, lookup, table, score, expected
    ):
        assert lookup(score, table) == expected

    @pytest.mark.parametrize(
        "lookup",
        [false_accepts_per_hour, positive_quantile],
        ids=lambda lookup: lookup.__name__,
    )
    def test_an_empty_table_gives_none(self, lookup):
        assert lookup(0.7, ()) is None

    @pytest.mark.parametrize(
        "lookup",
        [false_accepts_per_hour, positive_quantile],
        ids=lambda lookup: lookup.__name__,
    )
    def test_a_score_on_a_table_point_returns_that_point(self, lookup):
        table = ((0.3, 5.0), (0.6, 1.0), (0.9, 0.5))

        assert lookup(0.6, table) == 1.0


# One spotter's calibration: false accepts reach their floor of 0 per hour at 0.8, and the
# quantile keeps rising past it, as tune positives score above every false accept.
SERVER_CAL = SpotterCalibration(
    false_accepts=((0.5, 3.0), (0.8, 0.0)),
    positive_quantile=((0.5, 0.0), (1.0, 1.0)),
)
# A second spotter on a different score scale.
ENDPOINT_CAL = SpotterCalibration(
    false_accepts=((0.2, 6.0), (0.6, 0.0)),
    positive_quantile=((0.2, 0.0), (0.8, 1.0)),
)


class TestArbitrationKey:
    def test_two_scores_at_the_false_accept_floor_order_by_quantile_higher_first(self):
        clear = arbitration_key(0.95, 0.5, SERVER_CAL)
        marginal = arbitration_key(0.65, 0.5, ENDPOINT_CAL)

        ranked = sorted([marginal, clear])

        assert clear[0] == marginal[0] == 0.0
        assert ranked == [clear, marginal]
        assert clear == pytest.approx((0.0, 0.1))

    def test_a_lower_false_accept_estimate_wins_whatever_the_quantiles(self):
        # The first has the lower quantile but the lower false-accept estimate.
        cal = SpotterCalibration(
            false_accepts=((0.5, 4.0), (0.9, 1.0)),
            positive_quantile=((0.5, 0.0), (0.9, 0.1)),
        )
        steep = SpotterCalibration(
            false_accepts=((0.5, 4.0), (0.9, 2.0)),
            positive_quantile=((0.5, 0.9), (0.9, 1.0)),
        )
        fewer = arbitration_key(0.9, 0.5, cal)
        more = arbitration_key(0.9, 0.5, steep)

        assert fewer < more
        assert fewer[1] > more[1]

    def test_any_calibrated_key_sorts_before_any_uncalibrated_one(self):
        # The calibrated candidate is weak; the uncalibrated one has a large margin.
        weak = arbitration_key(0.5, 0.5, SERVER_CAL)
        strong_uncalibrated = arbitration_key(1.0, 0.1, None)

        assert weak < strong_uncalibrated
        assert strong_uncalibrated[0] == math.inf

    def test_a_calibration_with_an_empty_table_is_uncalibrated(self):
        partial = SpotterCalibration(false_accepts=((0.5, 1.0),), positive_quantile=())

        assert arbitration_key(0.8, 0.5, partial) == arbitration_key(0.8, 0.5, None)

    def test_two_uncalibrated_keys_order_by_margin_larger_first(self):
        wide = arbitration_key(0.9, 0.5, None)
        narrow = arbitration_key(0.7, 0.6, None)

        assert sorted([narrow, wide]) == [wide, narrow]
        assert wide == pytest.approx((math.inf, -0.4))


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


class TestScriptedTranscriberRelease:
    def test_release_is_counted_and_a_second_call_does_nothing_more(self):
        transcriber = ScriptedTranscriber(["hi"])

        transcriber.release()
        transcriber.release()

        assert transcriber.releases == 2
        assert transcriber.streams == []

    def test_open_stream_after_release_raises_speech_engine_error(self):
        transcriber = ScriptedTranscriber(["hi"])
        transcriber.release()

        with pytest.raises(SpeechEngineError, match="released"):
            transcriber.open_stream()

        assert transcriber.streams == []

    def test_release_can_be_told_to_raise_and_still_counts_as_released(self):
        transcriber = ScriptedTranscriber(["hi"], release_error=SpeechEngineError("unload failed"))

        with pytest.raises(SpeechEngineError, match="unload failed"):
            transcriber.release()

        assert transcriber.releases == 1
        with pytest.raises(SpeechEngineError, match="released"):
            transcriber.open_stream()


class TestTranscriptStreamClose:
    """The owner closes a stream once; a release or a cut may close it under a running decode."""

    def test_a_second_close_does_nothing(self):
        stream = ScriptedTranscriber(["hi"]).open_stream()
        stream.push(np.zeros(512, dtype=np.float32))

        stream.close()
        stream.close()

        assert stream.closed is True
        assert stream.pushed == [512]
        assert stream.partials == []
        assert stream.finals == []

    def test_a_partial_decode_in_flight_at_close_has_its_result_discarded(self):
        streams: list[ScriptedTranscriptStream] = []
        transcriber = ScriptedTranscriber(
            ["turn on the lights"], on_decode=lambda: streams[0].close()
        )
        streams.append(transcriber.open_stream())
        streams[0].push(np.zeros(6400, dtype=np.float32))

        with pytest.raises(LaneJobCancelled, match="closed"):
            streams[0].decode_partial(CancelToken())

        assert streams[0].partials == []

    def test_a_final_decode_in_flight_at_close_has_its_result_discarded(self):
        streams: list[ScriptedTranscriptStream] = []
        transcriber = ScriptedTranscriber(["hi"], on_decode=lambda: streams[0].close())
        streams.append(transcriber.open_stream())

        with pytest.raises(LaneJobCancelled, match="closed"):
            streams[0].decode_final(CancelToken())

        assert streams[0].finals == []

    def test_a_partial_decode_in_flight_on_another_thread_is_discarded(self):
        started, release = threading.Event(), threading.Event()

        def hold() -> None:
            started.set()
            release.wait(timeout=5)

        stream = ScriptedTranscriber(["turn on"], on_decode=hold).open_stream()
        stream.push(np.zeros(6400, dtype=np.float32))
        outcome: list[object] = []

        def decode() -> None:
            try:
                outcome.append(stream.decode_partial(CancelToken()))
            except RealtimeError as exc:
                outcome.append(exc)

        worker = threading.Thread(target=decode)
        worker.start()
        assert started.wait(timeout=5)
        stream.close()
        release.set()
        worker.join(timeout=5)

        assert len(outcome) == 1
        assert isinstance(outcome[0], LaneJobCancelled)
        assert stream.partials == []

    def test_a_push_after_close_raises_realtime_error(self):
        stream = ScriptedTranscriber(["hi"]).open_stream()
        stream.close()

        with pytest.raises(RealtimeError, match="closed") as caught:
            stream.push(np.zeros(512, dtype=np.float32))

        assert not isinstance(caught.value, LaneJobCancelled)
        assert stream.pushed == []

    def test_a_decode_started_after_close_is_cut_with_lane_job_cancelled(self):
        stream = ScriptedTranscriber(["hi"]).open_stream()
        stream.close()

        with pytest.raises(LaneJobCancelled, match="closed"):
            stream.decode_partial(CancelToken())
        with pytest.raises(LaneJobCancelled, match="closed"):
            stream.decode_final(CancelToken())


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

    def test_defaults_are_the_fake_engine_and_its_voice_identity_and_both_are_settable(self):
        synthesizer = FakeSynthesizer()
        defaults = (synthesizer.engine_id, synthesizer.voice, synthesizer.sample_rate)
        other_voice = VoiceInfo("pocket-clone", "test-voice", "studio")

        synthesizer.engine_id = "pocket-clone"
        synthesizer.voice = other_voice

        assert defaults == ("fake", VoiceInfo("fake", "fake-voice", "mist"), 24000)
        assert VoiceInfo("fake", "fake-voice", "mist") == FAKE_VOICE
        assert (synthesizer.engine_id, synthesizer.voice) == ("pocket-clone", other_voice)

    def test_engine_id_and_voice_are_constructor_arguments(self):
        voice = VoiceInfo("pocket-clone", "test-voice", "mist")

        synthesizer = FakeSynthesizer(engine_id="pocket-clone", voice=voice)

        assert (synthesizer.engine_id, synthesizer.voice) == ("pocket-clone", voice)

    def test_records_every_requested_segment(self):
        synthesizer = FakeSynthesizer()
        segments = [TextSegment("one", 0, 3), TextSegment("two", 4, 7)]

        for segment in segments:
            list(synthesizer.synthesize(segment, CancelToken()))

        assert synthesizer.segments == segments


FAKES_LOGGER = "tests.fakes.speech"


def _voice(synthesizer: FakeSynthesizer, state: str) -> list[SynthChunk]:
    return list(synthesizer.synthesize(TextSegment("x" * 10, 0, 10, state), CancelToken()))


class TestFakeSynthesizerStates:
    """The fake voices by the speaking-state rules every adapter follows."""

    def test_states_default_to_neutral_only(self):
        assert FakeSynthesizer().states == frozenset({"neutral"})

    def test_a_supported_state_is_carried_on_every_chunk(self, caplog):
        synthesizer = FakeSynthesizer(states=frozenset({"neutral", "calm"}))

        with caplog.at_level(logging.DEBUG, logger=FAKES_LOGGER):
            chunks = _voice(synthesizer, "calm")

        assert chunks
        assert all(chunk.state == "calm" for chunk in chunks)
        assert caplog.records == []

    def test_a_segment_without_a_state_is_voiced_neutral(self):
        chunks = list(FakeSynthesizer().synthesize(TextSegment("hello", 0, 5), CancelToken()))

        assert chunks
        assert all(chunk.state == "neutral" for chunk in chunks)

    def test_an_unvoiced_state_in_the_vocabulary_falls_back_to_neutral_with_one_info(self, caplog):
        synthesizer = FakeSynthesizer()

        with caplog.at_level(logging.DEBUG, logger=FAKES_LOGGER):
            chunks = _voice(synthesizer, "sleepy")

        assert chunks
        assert all(chunk.state == "neutral" for chunk in chunks)
        assert all(chunk.segment.state == "sleepy" for chunk in chunks)
        assert [record.levelno for record in caplog.records] == [logging.INFO]
        message = caplog.records[0].getMessage()
        assert "fake" in message and "fake-voice" in message and "sleepy" in message

    def test_a_state_outside_the_vocabulary_falls_back_to_neutral_with_one_warning(self, caplog):
        synthesizer = FakeSynthesizer(states=frozenset({"neutral", "calm"}))

        with caplog.at_level(logging.DEBUG, logger=FAKES_LOGGER):
            chunks = _voice(synthesizer, "furious")

        assert chunks
        assert all(chunk.state == "neutral" for chunk in chunks)
        assert [record.levelno for record in caplog.records] == [logging.WARNING]
        assert "furious" in caplog.records[0].getMessage()

    def test_a_repeated_fallback_logs_nothing_more(self, caplog):
        synthesizer = FakeSynthesizer()

        with caplog.at_level(logging.DEBUG, logger=FAKES_LOGGER):
            for state in ("sleepy", "furious", "sleepy", "furious", "sleepy"):
                _voice(synthesizer, state)

        assert [record.levelno for record in caplog.records] == [logging.INFO, logging.WARNING]

    def test_a_fallback_log_never_holds_the_segment_text(self, caplog):
        synthesizer = FakeSynthesizer()
        text = "test line zebra quartz"

        with caplog.at_level(logging.DEBUG, logger=FAKES_LOGGER):
            list(synthesizer.synthesize(TextSegment(text, 0, len(text), "awake"), CancelToken()))

        assert caplog.records
        assert all("zebra quartz" not in record.getMessage() for record in caplog.records)

    def test_the_segment_is_recorded_with_its_requested_state(self):
        synthesizer = FakeSynthesizer()
        segment = TextSegment("hello", 0, 5, "excited")

        list(synthesizer.synthesize(segment, CancelToken()))

        assert synthesizer.segments == [segment]


class TestFakeSynthesizerRelease:
    def test_release_is_counted_and_a_second_call_does_nothing_more(self):
        synthesizer = FakeSynthesizer()

        synthesizer.release()
        synthesizer.release()

        assert synthesizer.releases == 2
        assert synthesizer.segments == []

    def test_synthesize_after_release_raises_speech_engine_error(self):
        synthesizer = FakeSynthesizer()
        synthesizer.release()

        with pytest.raises(SpeechEngineError, match="released"):
            synthesizer.synthesize(TextSegment("hello", 0, 5), CancelToken())

        assert synthesizer.segments == []

    def test_release_can_be_told_to_raise_and_still_counts_as_released(self):
        synthesizer = FakeSynthesizer(release_error=SpeechEngineError("unload failed"))

        with pytest.raises(SpeechEngineError, match="unload failed"):
            synthesizer.release()

        assert synthesizer.releases == 1
        with pytest.raises(SpeechEngineError, match="released"):
            synthesizer.synthesize(TextSegment("hello", 0, 5), CancelToken())


FRAME = bytes(640)
"""One 20 ms PCM16 frame of silence at 16 kHz."""


class TestScriptedNameSpotter:
    def test_a_hit_is_returned_at_its_end_position_and_nowhere_else(self):
        hit = NameHit(end_pos=4, keyword="mist", score=0.8)
        stream = ScriptedNameSpotter([hit]).open_stream(0.5)

        results = [stream.push(FRAME, pos) for pos in range(7)]

        assert results == [[], [], [], [], [hit], [], []]

    def test_a_hit_at_the_threshold_counts_and_one_below_it_does_not(self):
        at = NameHit(end_pos=2, keyword="mist", score=0.5)
        below = NameHit(end_pos=3, keyword="mist", score=0.49)
        stream = ScriptedNameSpotter([at, below]).open_stream(0.5)

        assert stream.push(FRAME, 2) == [at]
        assert stream.push(FRAME, 3) == []

    def test_the_threshold_is_the_streams_own(self):
        hit = NameHit(end_pos=1, keyword="mist", score=0.6)
        spotter = ScriptedNameSpotter([hit])
        strict = spotter.open_stream(0.7)
        lenient = spotter.open_stream(0.3)

        assert strict.push(FRAME, 1) == []
        assert lenient.push(FRAME, 1) == [hit]

    def test_hit_lag_and_calibration_default_to_zero_and_empty(self):
        spotter = ScriptedNameSpotter([])

        assert spotter.hit_lag_ms == 0
        assert dict(spotter.calibration) == {}

    def test_hit_lag_and_calibration_are_constructor_arguments(self):
        calibration = {"server": SERVER_CAL}

        spotter = ScriptedNameSpotter([], hit_lag_ms=240, calibration=calibration)

        assert spotter.hit_lag_ms == 240
        assert dict(spotter.calibration) == calibration

    def test_score_window_returns_the_score_scripted_for_a_marker_in_the_buffer(self):
        marker = b"NAME-MARKER"
        spotter = ScriptedNameSpotter([], markers={marker: 0.83})
        buffer = FRAME * 3 + marker + FRAME * 2

        assert spotter.score_window(buffer) == 0.83

    def test_score_window_without_a_marker_returns_zero(self):
        spotter = ScriptedNameSpotter([], markers={b"NAME-MARKER": 0.83})

        assert spotter.score_window(FRAME * 10) == 0.0

    def test_score_window_with_two_markers_returns_the_higher_score(self):
        spotter = ScriptedNameSpotter([], markers={b"FAINT": 0.3, b"CLEAR": 0.9})

        assert spotter.score_window(b"FAINT" + FRAME + b"CLEAR") == 0.9

    def test_an_empty_marker_is_refused(self):
        with pytest.raises(ValueError, match="marker"):
            ScriptedNameSpotter([], markers={b"": 0.5})

    def test_a_hit_already_reported_in_a_context_is_not_reported_again(self):
        hit = NameHit(end_pos=5, keyword="mist", score=0.9)
        stream = ScriptedNameSpotter([hit]).open_stream(0.5)

        first = stream.push(FRAME, 5)
        replayed = stream.push(FRAME, 5)

        assert first == [hit]
        assert replayed == []

    def test_reset_forgets_partial_context(self):
        hit = NameHit(end_pos=5, keyword="mist", score=0.9)
        stream = ScriptedNameSpotter([hit]).open_stream(0.5)
        for pos in range(3, 6):
            stream.push(FRAME, pos)

        stream.reset()

        assert stream.context == []
        assert stream.resets == 1
        assert stream.push(FRAME, 5) == [hit]
        assert stream.context == [5]

    def test_the_spotter_records_its_streams_and_scored_buffers(self):
        spotter = ScriptedNameSpotter([])

        stream = spotter.open_stream(0.4)
        spotter.score_window(FRAME * 2)
        stream.push(FRAME, 0)

        assert spotter.streams == [stream]
        assert stream.threshold == 0.4
        assert spotter.scored == [len(FRAME) * 2]
        assert stream.pushed == [0]
