"""Tests for the voice fixture generator, `scripts.voice_fixtures.generate_fixtures`.

Every clip is a synthetic tone from `ToneSource`, a fake clip source, so no TTS engine, voice
identity or weight is touched. The tones carry silence at both ends, as engine output does, so
the tests also check that the generator trims it before it places a clip. Files are written only
under `tmp_path`: `main` takes the repository root as an argument, so the refusal of any other
`--out` is tested against a temporary root rather than the checkout.
"""

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from backend.realtime.addressee import normalize
from backend.realtime.budget import BargeInConfig, EndpointingConfig, EngineOption, WakeConfig
from backend.realtime.speech import VoiceInfo
from scripts.voice_fixtures import generate_fixtures as gen

REPO = Path(__file__).resolve().parents[3]
RATE = gen.RATE
PAD_MS = 60


def ms(n: int) -> int:
    return n * RATE // 1000


def tone(duration_ms: int, freq: float = 440.0, amplitude: float = 0.5) -> np.ndarray:
    # A cosine, so the first sample is already non-zero and a clip's start is its first sample.
    t = np.arange(ms(duration_ms)) / RATE
    return (amplitude * np.cos(2 * np.pi * freq * t)).astype(np.float32)


def padded(clip: np.ndarray, pad_ms: int = PAD_MS) -> np.ndarray:
    return np.concatenate(
        [np.zeros(ms(pad_ms), np.float32), clip, np.zeros(ms(pad_ms), np.float32)]
    )


class ToneSource:
    """A clip source whose clip is a tone, 30 ms per character, inside 60 ms of silence."""

    def __init__(self) -> None:
        self.voice = VoiceInfo(engine="fake-tts", voice="fake-voice", profile="mist")
        self.texts: list[str] = []

    def clip(self, text: str) -> np.ndarray:
        self.texts.append(text)
        return padded(tone(30 * len(text), freq=300.0 + 10 * (len(text) % 30)))


def first_nonzero(audio: np.ndarray) -> int:
    return int(np.flatnonzero(np.abs(audio) > 1e-6)[0])


def last_nonzero(audio: np.ndarray) -> int:
    return int(np.flatnonzero(np.abs(audio) > 1e-6)[-1])


def rms_dbfs(samples: np.ndarray) -> float:
    return 20 * math.log10(float(np.sqrt(np.mean(np.square(samples.astype(np.float64))))))


# -- Signal helpers --------------------------------------------------------------------------


class TestTrimSilence:
    def test_trims_the_silence_around_a_tone(self):
        clip = padded(tone(400), pad_ms=120)

        trimmed = gen.trim_silence(clip)

        assert abs(len(trimmed) - ms(400)) <= ms(10)

    def test_keeps_a_clip_with_no_silence(self):
        clip = tone(300)

        assert len(gen.trim_silence(clip)) == len(clip)

    def test_an_all_silent_clip_is_refused(self):
        with pytest.raises(gen.FixtureError, match="silent"):
            gen.trim_silence(np.zeros(ms(200), np.float32))


class TestNoiseBurst:
    def test_burst_rms_is_its_level(self):
        burst = gen.noise_burst(300, -20.0, seed=1)

        assert len(burst) == ms(300)
        assert rms_dbfs(burst) == pytest.approx(-20.0, abs=0.01)

    def test_same_seed_gives_the_same_burst(self):
        assert np.array_equal(gen.noise_burst(150, -12.0, seed=2), gen.noise_burst(150, -12.0, 2))

    def test_different_seeds_give_different_bursts(self):
        assert not np.array_equal(gen.noise_burst(150, -12.0, 2), gen.noise_burst(150, -12.0, 3))


# -- Scenario composition --------------------------------------------------------------------


class TestCompleteFixture:
    def test_places_the_clip_after_the_lead_and_labels_it_complete(self):
        clip = tone(700)

        fixture = gen.complete_fixture("complete-01", "What is it?", clip)

        entry = fixture.entry
        assert entry.scenario == "complete"
        assert entry.text == "What is it?"
        assert entry.onset_ms == gen.LEAD_MS
        assert entry.speech == (gen.Span(gen.LEAD_MS, gen.LEAD_MS + 700),)
        assert entry.eot_probes == (
            gen.EotProbe(gen.LEAD_MS + 700 + EndpointingConfig().endpoint_silence_ms, True),
        )
        assert first_nonzero(fixture.audio) == ms(gen.LEAD_MS)
        assert len(fixture.audio) == ms(gen.LEAD_MS + 700 + gen.TAIL_MS)
        assert entry.duration_ms == gen.LEAD_MS + 700 + gen.TAIL_MS


class TestPauseFixture:
    def test_pause_is_silent_and_labelled_incomplete_then_complete(self):
        before, after = tone(500), tone(600)

        fixture = gen.pause_fixture("pause-01", "Could you", "do this?", before, after)

        entry = fixture.entry
        first_end = gen.LEAD_MS + 500
        second_start = first_end + gen.PAUSE_MS
        second_end = second_start + 600
        probe_ms = EndpointingConfig().endpoint_silence_ms
        assert entry.text == "Could you do this?"
        assert entry.speech == (
            gen.Span(gen.LEAD_MS, first_end),
            gen.Span(second_start, second_end),
        )
        assert entry.eot_probes == (
            gen.EotProbe(first_end + probe_ms, False),
            gen.EotProbe(second_end + probe_ms, True),
        )
        assert not np.any(fixture.audio[ms(first_end) : ms(second_start)])

    def test_pause_is_longer_than_endpoint_silence_and_shorter_than_the_cap(self):
        config = EndpointingConfig()

        assert config.endpoint_silence_ms < gen.PAUSE_MS < config.eot_max_silence_ms


class TestHesitationFixture:
    def test_trailing_hesitation_is_labelled_incomplete(self):
        fixture = gen.hesitation_fixture("hesitation-01", "So maybe we...", tone(800))

        entry = fixture.entry
        end = gen.LEAD_MS + 800
        assert entry.scenario == "hesitation"
        assert entry.eot_probes == (
            gen.EotProbe(end + EndpointingConfig().endpoint_silence_ms, False),
        )


class TestKeywordEchoFixture:
    def test_echo_is_attenuated_20_db_and_delayed_50_ms_under_the_keyword(self):
        keyword, echo = tone(300, freq=800.0), tone(2000, freq=200.0)

        fixture = gen.keyword_echo_fixture("echo-01", "stop", keyword, "An echo line.", echo)

        entry = fixture.entry
        echo_start = gen.LEAD_MS + gen.ECHO_DELAY_MS
        keyword_start = echo_start + gen.KEYWORD_OFFSET_MS
        assert gen.ECHO_ATTENUATION_DB == 20.0
        assert gen.ECHO_DELAY_MS == 50
        assert entry.text == "stop"
        assert entry.onset_ms is None
        assert entry.echo == gen.EchoInfo(
            keyword="stop",
            keyword_span=gen.Span(keyword_start, keyword_start + 300),
            echo_text="An echo line.",
            echo_span=gen.Span(echo_start, echo_start + 2000),
            playback_start_ms=gen.LEAD_MS,
            attenuation_db=20.0,
            delay_ms=50,
        )
        assert first_nonzero(fixture.audio) == ms(echo_start)
        expected = np.zeros(len(fixture.audio), np.float32)
        expected[ms(echo_start) : ms(echo_start) + len(echo)] += echo * 0.1
        expected[ms(keyword_start) : ms(keyword_start) + len(keyword)] += keyword
        np.testing.assert_allclose(fixture.audio, expected, atol=1e-6)

    def test_a_keyword_past_the_echo_end_lengthens_the_fixture(self):
        fixture = gen.keyword_echo_fixture("echo-02", "stop", tone(300), "Short.", tone(200))

        keyword_end = gen.LEAD_MS + gen.ECHO_DELAY_MS + gen.KEYWORD_OFFSET_MS + 300
        assert fixture.entry.duration_ms == keyword_end + gen.TAIL_MS
        assert last_nonzero(fixture.audio) < ms(keyword_end)


class TestNoiseFixture:
    def test_burst_sits_between_silences_and_expects_no_text(self):
        fixture = gen.noise_fixture("noise-01", burst_ms=300, level_dbfs=-20.0, seed=1)

        entry = fixture.entry
        assert entry.text == ""
        assert entry.speech == ()
        assert entry.onset_ms is None
        assert entry.eot_probes == ()
        assert entry.noise == gen.NoiseInfo(
            span=gen.Span(gen.LEAD_MS, gen.LEAD_MS + 300), level_dbfs=-20.0, seed=1
        )
        burst = fixture.audio[ms(gen.LEAD_MS) : ms(gen.LEAD_MS + 300)]
        assert rms_dbfs(burst) == pytest.approx(-20.0, abs=0.01)
        assert not np.any(fixture.audio[: ms(gen.LEAD_MS)])
        assert not np.any(fixture.audio[ms(gen.LEAD_MS + 300) :])


class TestWakeFixture:
    def test_backlog_to_the_name_end_meets_the_endpoint_minimum(self):
        fixture = gen.wake_fixture(
            "wake-01", "Hey Mist,", "what time is it?", "tool", tone(400), tone(900)
        )

        entry = fixture.entry
        wake = entry.wake
        assert wake is not None
        assert wake.name_end_ms >= WakeConfig().endpoint_min_backlog_ms
        assert wake.name_end_ms % gen.FRAME_MS == 0
        assert wake.kind == "tool"
        assert entry.onset_ms == wake.name_end_ms - 400
        assert entry.text == "Hey Mist, what time is it?"
        request_start = wake.name_end_ms + gen.WAKE_GAP_MS
        assert entry.speech == (
            gen.Span(wake.name_end_ms - 400, wake.name_end_ms),
            gen.Span(request_start, request_start + 900),
        )
        assert first_nonzero(fixture.audio) == ms(entry.onset_ms)

    def test_a_long_name_clip_keeps_the_usual_lead(self):
        fixture = gen.wake_fixture("wake-02", "Hey Mist,", "hi.", "plain", tone(1800), tone(300))

        assert fixture.entry.onset_ms is not None
        assert fixture.entry.onset_ms >= gen.LEAD_MS


# -- Phrases and the whole set ---------------------------------------------------------------


class TestPhrases:
    def test_the_committed_phrases_file_loads(self):
        phrases = gen.load_phrases(gen.PHRASES_PATH)

        assert phrases.complete
        assert phrases.pause
        assert phrases.hesitation
        assert phrases.keyword_echo
        assert phrases.wake
        assert phrases.noise

    def test_every_echo_keyword_is_a_barge_in_keyword(self):
        phrases = gen.load_phrases(gen.PHRASES_PATH)

        keywords = {item.keyword for item in phrases.keyword_echo}

        assert keywords <= set(BargeInConfig().keywords)

    def test_every_wake_name_says_one_of_her_names(self):
        phrases = gen.load_phrases(gen.PHRASES_PATH)

        for item in phrases.wake:
            assert set(normalize(item.name)) & set(WakeConfig().names), item.name

    def test_a_keyword_outside_the_barge_in_set_is_refused(self, tmp_path):
        raw = json.loads(gen.PHRASES_PATH.read_text(encoding="utf-8"))
        raw["keyword_echo"][0]["keyword"] = "banana"
        path = tmp_path / "phrases.json"
        path.write_text(json.dumps(raw), encoding="utf-8")

        with pytest.raises(gen.FixtureError, match="banana"):
            gen.load_phrases(path)


class TestBuildFixtures:
    def test_builds_every_scenario_with_unique_ids_from_trimmed_clips(self):
        phrases = gen.load_phrases(gen.PHRASES_PATH)
        source = ToneSource()

        fixtures = gen.build_fixtures(phrases, source)

        ids = [fixture.entry.id for fixture in fixtures]
        assert len(ids) == len(set(ids))
        scenarios = {fixture.entry.scenario for fixture in fixtures}
        assert scenarios == {"complete", "pause", "hesitation", "keyword_echo", "noise", "wake"}
        complete = next(f for f in fixtures if f.entry.scenario == "complete")
        # The tone's own 60 ms of leading silence was trimmed, so speech starts at the lead.
        assert abs(first_nonzero(complete.audio) - ms(gen.LEAD_MS)) <= ms(10)

    def test_counts_follow_the_phrases_file(self):
        phrases = gen.load_phrases(gen.PHRASES_PATH)

        fixtures = gen.build_fixtures(phrases, ToneSource())

        counts = {}
        for fixture in fixtures:
            counts[fixture.entry.scenario] = counts.get(fixture.entry.scenario, 0) + 1
        assert counts == {
            "complete": len(phrases.complete),
            "pause": len(phrases.pause),
            "hesitation": len(phrases.hesitation),
            "keyword_echo": len(phrases.keyword_echo),
            "noise": len(phrases.noise),
            "wake": len(phrases.wake),
        }


class TestManifest:
    def test_written_set_reads_back_with_its_audio(self, tmp_path):
        source = ToneSource()
        fixtures = gen.build_fixtures(gen.load_phrases(gen.PHRASES_PATH), source)

        gen.write_fixtures(tmp_path, fixtures, source.voice)
        manifest = gen.read_manifest(tmp_path)

        assert manifest.sample_rate == RATE
        assert manifest.voice == source.voice
        assert manifest.fixtures == tuple(fixture.entry for fixture in fixtures)
        first = fixtures[0]
        audio = gen.read_wav(tmp_path / first.entry.file)
        np.testing.assert_allclose(audio, first.audio, atol=1.0 / 32767)

    def test_manifest_is_json_with_expected_text_and_boundaries(self, tmp_path):
        fixture = gen.complete_fixture("complete-01", "What is it?", tone(700))

        gen.write_fixtures(tmp_path, [fixture], VoiceInfo("fake-tts", "fake-voice", "mist"))

        raw = json.loads((tmp_path / gen.MANIFEST_NAME).read_text(encoding="utf-8"))
        (entry,) = raw["fixtures"]
        assert entry["text"] == "What is it?"
        assert entry["speech"] == [{"start_ms": gen.LEAD_MS, "end_ms": gen.LEAD_MS + 700}]
        assert entry["file"] == "complete-01.wav"


# -- The engine the generator speaks through --------------------------------------------------


class TestFirstTtsOption:
    def test_takes_the_chains_first_option_with_the_named_identity(self):
        environ = {"MIST_VOICE_TTS_CHAIN": "pocket-clone@cpu,other-engine@cuda"}

        settings, option = gen.first_tts_option(environ, "/voices-test", "voice-a")

        assert option == EngineOption("pocket-clone", "cpu")
        assert settings.placement.tts_chain == (option,)
        assert settings.placement.stt_chain == ()
        assert settings.placement.voice_root == "/voices-test"
        assert settings.placement.tts_voice == "voice-a"

    def test_the_arguments_override_the_environment_identity(self):
        environ = {"MIST_VOICE_IDENTITY_ROOT": "/other", "MIST_VOICE_TTS_VOICE": "other-voice"}

        settings, _ = gen.first_tts_option(environ, "/voices-test", "voice-a")

        assert settings.placement.voice_root == "/voices-test"
        assert settings.placement.tts_voice == "voice-a"

    def test_a_chain_that_is_off_is_refused(self):
        with pytest.raises(gen.FixtureError, match="MIST_VOICE_TTS_CHAIN"):
            gen.first_tts_option({"MIST_VOICE_TTS_CHAIN": "off"}, "/voices-test", "voice-a")

    def test_a_malformed_voice_id_is_refused_naming_it(self):
        with pytest.raises(gen.FixtureError, match="MIST_VOICE_TTS_VOICE"):
            gen.first_tts_option({}, "/voices-test", "Not An Id")


# -- The command line -------------------------------------------------------------------------


class RecordingLoader:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.source = ToneSource()

    def __call__(self, voice_root: str, voice_id: str, environ):
        self.calls.append((voice_root, voice_id))
        return self.source


def run_main(repo_root: Path, *args: str, loader: RecordingLoader) -> int:
    return gen.main(
        ["--voice-root", "/voices-test", "--voice-id", "voice-a", *args],
        repo_root=repo_root,
        environ={},
        load_source=loader,
    )


def files_under(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


class TestMain:
    def test_writes_the_set_under_the_generated_directory(self, tmp_path):
        loader = RecordingLoader()

        code = run_main(tmp_path, loader=loader)

        assert code == 0
        out = tmp_path / gen.GENERATED_DIR
        assert loader.calls == [("/voices-test", "voice-a")]
        manifest = gen.read_manifest(out)
        assert {path.name for path in out.iterdir()} == {gen.MANIFEST_NAME} | {
            entry.file for entry in manifest.fixtures
        }
        assert all(path.is_relative_to(out) for path in files_under(tmp_path))

    def test_accepts_the_generated_directory_named_explicitly(self, tmp_path):
        loader = RecordingLoader()

        code = run_main(tmp_path, "--out", str(tmp_path / gen.GENERATED_DIR), loader=loader)

        assert code == 0
        assert (tmp_path / gen.GENERATED_DIR / gen.MANIFEST_NAME).is_file()

    @pytest.mark.parametrize(
        "out",
        [
            pytest.param("elsewhere", id="outside-the-repository-tree"),
            pytest.param("tests/fixtures/voice", id="the-parent-directory"),
            pytest.param("tests/fixtures/voice/generated/sub", id="a-subdirectory"),
            pytest.param("tests/fixtures/voice/generated/../other", id="dot-dot-out"),
        ],
    )
    def test_refuses_any_other_out_before_loading_the_engine(self, tmp_path, capsys, out):
        loader = RecordingLoader()

        code = run_main(tmp_path, "--out", str(tmp_path / out), loader=loader)

        assert code == 2
        assert loader.calls == []
        assert files_under(tmp_path) == []
        assert "tests/fixtures/voice/generated" in capsys.readouterr().err

    @pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
    def test_refuses_a_generated_directory_that_links_outside(self, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        link = tmp_path / "repo" / gen.GENERATED_DIR
        link.parent.mkdir(parents=True)
        link.symlink_to(outside, target_is_directory=True)
        loader = RecordingLoader()

        code = run_main(tmp_path / "repo", loader=loader)

        assert code == 2
        assert loader.calls == []
        assert list(outside.iterdir()) == []

    def test_voice_root_and_voice_id_are_required(self, tmp_path):
        with pytest.raises(SystemExit) as excinfo:
            gen.main([], repo_root=tmp_path, environ={}, load_source=RecordingLoader())

        assert excinfo.value.code == 2

    def test_a_voice_that_fails_to_load_exits_1_and_writes_nothing(self, tmp_path, capsys):
        def failing(voice_root: str, voice_id: str, environ):
            raise gen.FixtureError("pocket-clone@cpu did not load")

        code = gen.main(
            ["--voice-root", "/voices-test", "--voice-id", "voice-a"],
            repo_root=tmp_path,
            environ={},
            load_source=failing,
        )

        assert code == 1
        assert files_under(tmp_path) == []
        assert "pocket-clone@cpu did not load" in capsys.readouterr().err


def test_gitignore_ignores_the_generated_directory():
    lines = (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert "tests/fixtures/voice/generated/" in lines
