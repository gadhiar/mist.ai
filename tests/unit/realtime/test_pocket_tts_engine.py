"""Tests for the cloned-voice Pocket TTS adapter and the shared text and audio rules.

pocket-tts is not installed where these tests run, so every test hands the adapter a fake
library: a model factory that records the config it was given, a voice-state export that writes
a small file, and a model that returns sine tones and records the voice state and text of each
call. That pins what the adapter owns: the config it builds from the installed one, where the
weights must be, the voice prompts it builds and keeps per speaking state, the state fallback,
the text it sends, the audio it yields, release, and that no log line or error holds text.

Identities are synthetic: sine-tone WAVs written with `wave` into `tmp_path`, with transcripts
carrying a distinctive mark that must never reach a log record or an error.
"""

import gc
import hashlib
import importlib
import json
import logging
import shutil
import sys
import wave
import weakref
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import yaml

from backend.realtime.budget import PlacementConfig
from backend.realtime.engines import audio, pocket_tts
from backend.realtime.engines.audio import (
    LoudnessNormalizer,
    integrated_loudness,
    pcm16_chunks,
    resample,
    speakable_text,
    to_pcm16,
)
from backend.realtime.engines.pocket_tts import (
    CLONE_MODEL_URI,
    CONFIG_NAME,
    MODEL_URI,
    TOKENIZER_URI,
    VOICE_STATE_FILE,
    PocketCloneSynthesizer,
    PocketLibrary,
    PocketVoiceBuilder,
    local_config,
)
from backend.realtime.errors import SpeechEngineError
from backend.realtime.speech import (
    CancelToken,
    SpeechSynthesizer,
    SynthChunk,
    TextSegment,
    VoiceInfo,
)
from backend.realtime.voice_assets import (
    VOICE_MANIFEST,
    VoiceArtefactBuilder,
    VoiceAssetError,
    load_identity,
)

RATE = 24000
VOICE_ID = "test-voice"
PROFILE = "test-profile"
TRANSCRIPT_MARK = "zebra quartz"
"""A distinctive transcript string no log record or error text may hold."""
SEGMENT_MARK = "violet harbour"
"""A distinctive segment string no log record or error text may hold."""
TONES = {"neutral": 220.0, "calm": 330.0, "sleepy": 440.0}
LOGGERS = (pocket_tts.__name__, audio.__name__, "backend.realtime.voice_assets")


# ---------------------------------------------------------------------------
# Synthetic identities and models directory
# ---------------------------------------------------------------------------


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_wav(path: Path, frequency: float, seconds: float = 4.0, rate: int = RATE) -> None:
    t = np.arange(int(seconds * rate)) / rate
    tone = (0.3 * np.sin(2 * np.pi * frequency * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(tone.tobytes())


def make_identity(voice_root: Path, states: tuple[str, ...]) -> Path:
    identity = voice_root / VOICE_ID
    identity.mkdir(parents=True)
    clips = []
    for index, state in enumerate(states):
        _write_wav(identity / f"{state}.wav", TONES.get(state, 550.0))
        line = f"test line {index} {TRANSCRIPT_MARK}"
        (identity / f"{state}.txt").write_text(line, encoding="utf-8")
        clips.append(
            {
                "state": state,
                "audio": f"{state}.wav",
                "transcript": f"{state}.txt",
                "audio_sha256": _sha(identity / f"{state}.wav"),
                "transcript_sha256": _sha(identity / f"{state}.txt"),
            }
        )
    manifest = {"schema": 1, "voice_id": VOICE_ID, "status": "placeholder", "clips": clips}
    (identity / VOICE_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    return identity


def installed_config() -> dict:
    """The shape of pocket-tts 4.0.1's `english.yaml`, with its three `hf://` URIs."""
    return {
        "weights_path": CLONE_MODEL_URI,
        "weights_path_without_voice_cloning": MODEL_URI,
        "flow_lm": {
            "dim": 8,
            "lookup_table": {
                "n_bins": 4,
                "tokenizer": "sentencepiece",
                "tokenizer_path": TOKENIZER_URI,
            },
        },
        "mimi": {"sample_rate": RATE, "channels": 1},
    }


def mapped_files(models_dir: Path) -> dict[str, Path]:
    speech = models_dir / "speech" / "kyutai"
    return {
        CLONE_MODEL_URI: speech / "pocket-tts" / "languages" / "english" / "model.safetensors",
        MODEL_URI: speech
        / "pocket-tts-without-voice-cloning"
        / "languages"
        / "english"
        / "model.safetensors",
        TOKENIZER_URI: speech
        / "pocket-tts-without-voice-cloning"
        / "languages"
        / "english"
        / "tokenizer.json",
    }


def stage_weights(models_dir: Path) -> dict[str, Path]:
    files = mapped_files(models_dir)
    for path in files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"weights")
    return files


# ---------------------------------------------------------------------------
# The fake library
# ---------------------------------------------------------------------------


class FakeState:
    """A voice state: which clip it came from, and whether it was read back from the cache."""

    def __init__(self, source: str, loaded: bool) -> None:
        self.source = source
        self.loaded = loaded


@dataclass
class FakeModel:
    """A Pocket model: voice states from clips and files, sine tones for text."""

    sample_rate: int = RATE
    has_voice_cloning: bool = True
    seconds: float = 1.0
    amplitude: float = 0.1
    fail_prompt: Exception | None = None
    fail_generate: Exception | None = None
    stream_pieces: int = 0
    prompt_calls: list[Path] = field(default_factory=list)
    generated: list[tuple[str, str]] = field(default_factory=list)
    pulled: int = 0

    def get_state_for_audio_prompt(self, audio_conditioning: Path, truncate: bool = False):
        assert isinstance(audio_conditioning, Path)
        self.prompt_calls.append(audio_conditioning)
        if self.fail_prompt is not None:
            raise self.fail_prompt
        if audio_conditioning.suffix == ".safetensors":
            return FakeState(audio_conditioning.read_text(encoding="utf-8"), loaded=True)
        return FakeState(audio_conditioning.stem, loaded=False)

    def _tone(self, seconds: float) -> np.ndarray:
        t = np.arange(int(seconds * self.sample_rate)) / self.sample_rate
        return (self.amplitude * np.sin(2 * np.pi * 300.0 * t)).astype(np.float32)

    def generate_audio(self, model_state, text_to_generate: str):
        self.generated.append((model_state.source, text_to_generate))
        if self.fail_generate is not None:
            raise self.fail_generate
        return self._tone(self.seconds)


@dataclass
class StreamingModel(FakeModel):
    """A model with the streaming call, yielding `stream_pieces` pieces."""

    def generate_audio_stream(self, model_state, text_to_generate: str) -> Iterator[np.ndarray]:
        self.generated.append((model_state.source, text_to_generate))
        for _ in range(self.stream_pieces):
            self.pulled += 1
            yield self._tone(self.seconds / self.stream_pieces)


@dataclass
class FakeLibrary:
    """Builds the `PocketLibrary` the adapter loads with, and records every call into it."""

    configs_dir: Path
    model: FakeModel = field(default_factory=FakeModel)
    configs: list[dict] = field(default_factory=list)
    threads: list[int] = field(default_factory=list)
    exports: list[Path] = field(default_factory=list)
    loads: int = 0
    fail_load: Exception | None = None
    fail_export: Exception | None = None
    models: list[FakeModel] = field(default_factory=list)

    def load_model(self, config: Path) -> FakeModel:
        assert isinstance(config, Path)
        self.configs.append(yaml.safe_load(config.read_text(encoding="utf-8")))
        if self.fail_load is not None:
            raise self.fail_load
        model, self.model = self.model, None
        self.models.append(model)
        return model

    def export_model_state(self, model_state: FakeState, dest: Path) -> None:
        self.exports.append(dest)
        if self.fail_export is not None:
            raise self.fail_export
        dest.write_text(model_state.source, encoding="utf-8")

    def __call__(self) -> PocketLibrary:
        self.loads += 1
        return PocketLibrary(
            load_model=self.load_model,
            export_model_state=self.export_model_state,
            configs_dir=self.configs_dir,
            set_num_threads=self.threads.append,
        )


@dataclass
class Env:
    placement: PlacementConfig
    library: FakeLibrary
    models_dir: Path
    voice_root: Path
    cache: Path

    def load(self, **kwargs) -> PocketCloneSynthesizer:
        return PocketCloneSynthesizer.load(
            self.placement, PROFILE, library_loader=self.library, **kwargs
        )


def make_env(
    tmp_path: Path,
    states: tuple[str, ...] = ("neutral",),
    model: FakeModel | None = None,
    config: dict | None = None,
) -> Env:
    models_dir = tmp_path / "models"
    stage_weights(models_dir)
    configs_dir = tmp_path / "pocket_configs"
    configs_dir.mkdir(parents=True)
    (configs_dir / CONFIG_NAME).write_text(
        yaml.safe_dump(config if config is not None else installed_config()), encoding="utf-8"
    )
    voice_root = tmp_path / "voices"
    make_identity(voice_root, states)
    cache = tmp_path / "cache"
    placement = PlacementConfig(
        tts_voice=VOICE_ID,
        voice_root=str(voice_root),
        voice_cache_dir=str(cache),
        models_dir=str(models_dir),
    )
    library = FakeLibrary(configs_dir=configs_dir, model=model or FakeModel())
    return Env(placement, library, models_dir, voice_root, cache)


def collect(
    synth: PocketCloneSynthesizer, segment: TextSegment, token: CancelToken | None = None
) -> list[SynthChunk]:
    return list(synth.synthesize(segment, token or CancelToken()))


def segment(text: str = "Hello there, test line.", state: str = "neutral") -> TextSegment:
    return TextSegment(text, 0, len(text), state)


def pcm_samples(chunks: list[SynthChunk]) -> np.ndarray:
    raw = b"".join(chunk.pcm16 for chunk in chunks)
    return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32767


def _listing(root: Path) -> list[tuple[str, int, int]]:
    return sorted(
        (str(path.relative_to(root)), path.lstat().st_size, path.lstat().st_mtime_ns)
        for path in root.rglob("*")
    )


def _chain_text(exc: BaseException) -> str:
    """The text of an exception and of every exception a traceback of it would print."""
    parts = []
    seen: BaseException | None = exc
    while seen is not None:
        parts.append(f"{type(seen).__name__}: {seen}")
        seen = seen.__cause__ or (None if seen.__suppress_context__ else seen.__context__)
    return "\n".join(parts)


def _assert_private(records: list[logging.LogRecord], *marks: str) -> None:
    for record in records:
        message = record.getMessage()
        for mark in marks:
            assert mark not in message, message
        args = record.args if isinstance(record.args, tuple) else (record.args,)
        assert not [arg for arg in args if isinstance(arg, bytes | bytearray | np.ndarray)]
        if record.exc_info and record.exc_info[1] is not None:
            for mark in marks:
                assert mark not in _chain_text(record.exc_info[1])


# ---------------------------------------------------------------------------
# Text and audio rules (`engines/audio.py`)
# ---------------------------------------------------------------------------


class TestSpeakableText:
    @pytest.mark.parametrize(
        ("text", "keep_tags", "expected"),
        [
            ("Hello [laugh] NASA said: yes.", True, "Hello [laugh] NASA said: yes."),
            ("Hello [laugh] NASA said: yes.", False, "Hello NASA said: yes."),
            ("**Bold** and *it*", False, "Bold and it"),
            ("__Bold__ and _it_ in snake_case", False, "Bold and it in snake_case"),
            ("Run `pip list` now", False, "Run pip list now"),
            ("See [the docs](http://x) first", False, "See the docs first"),
            ("See [the docs](http://x) first", True, "See the docs first"),
            ("## Heading\n- one\n* two\n+ three", False, "Heading one two three"),
            ("Case, 42 digits; and: punctuation!?", False, "Case, 42 digits; and: punctuation!?"),
            ("**", False, ""),
            ("  [sigh]  ", False, ""),
        ],
    )
    def test_removes_only_what_is_not_speech(self, text, keep_tags, expected):
        assert speakable_text(text, keep_tags=keep_tags) == expected


def _sine(amplitude: float, seconds: float, rate: int = RATE, frequency: float = 997.0):
    t = np.arange(int(seconds * rate)) / rate
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


class TestLoudness:
    def test_meter_reads_a_minus_20_dbfs_tone_as_bs1770_does(self):
        # BS.1770: a 997 Hz sine at -20 dBFS peak reads -23.01 LUFS.
        assert integrated_loudness(_sine(0.1, 2.0, rate=48000), 48000) == pytest.approx(
            -23.01, abs=0.15
        )

    def test_silence_is_minus_infinity(self):
        assert integrated_loudness(np.zeros(RATE, dtype=np.float32), RATE) == float("-inf")

    @pytest.mark.parametrize("amplitude", [0.02, 0.6])
    def test_quiet_and_loud_segments_reach_the_target(self, amplitude):
        out = LoudnessNormalizer()(_sine(amplitude, 1.0), RATE)

        assert integrated_loudness(out, RATE) == pytest.approx(-20.0, abs=1.0)

    def test_gain_is_capped_at_20_db(self):
        quiet = _sine(0.001, 1.0)
        out = LoudnessNormalizer()(quiet, RATE)

        np.testing.assert_allclose(out, quiet * 10.0, rtol=1e-4)

    def test_a_short_segment_takes_the_previous_gain(self):
        normalizer = LoudnessNormalizer()
        first = _sine(0.05, 1.0)
        gain = float(np.max(normalizer(first, RATE)) / np.max(first))
        short = _sine(0.3, 0.3)

        np.testing.assert_allclose(normalizer(short, RATE), short * gain, rtol=1e-4)

    def test_a_first_short_segment_keeps_unity_gain(self):
        short = _sine(0.3, 0.3)

        np.testing.assert_allclose(LoudnessNormalizer()(short, RATE), short)


class TestPcmAndChunks:
    def test_pcm16_clips(self):
        pcm = np.frombuffer(to_pcm16(np.array([1.5, -1.5, 0.5], dtype=np.float32)), "<i2")

        assert pcm.tolist() == [32767, -32767, 16383]

    def test_resample_reaches_24_khz(self):
        out = resample(_sine(0.5, 1.0, rate=16000), 16000, RATE)

        assert len(out) == RATE
        assert out.dtype == np.float32

    def test_resample_at_the_same_rate_is_unchanged(self):
        samples = _sine(0.5, 0.5)

        assert resample(samples, RATE, RATE) is samples

    def test_chunks_are_at_most_200_ms(self):
        pcm = bytes(2 * (RATE + 100))
        chunks = list(pcm16_chunks(pcm, RATE))

        assert [len(chunk) // 2 for chunk in chunks] == [4800] * 5 + [100]


# ---------------------------------------------------------------------------
# Config and weights
# ---------------------------------------------------------------------------


class TestLocalConfig:
    def test_maps_the_three_uris_under_models_dir_with_cloning_on(self, tmp_path):
        files = stage_weights(tmp_path)

        config = local_config(installed_config(), tmp_path)

        assert config["weights_path"] == str(files[CLONE_MODEL_URI])
        assert config["weights_path_without_voice_cloning"] == str(files[MODEL_URI])
        assert config["flow_lm"]["lookup_table"]["tokenizer_path"] == str(files[TOKENIZER_URI])
        assert config["mimi"] == installed_config()["mimi"]
        assert "hf://" not in yaml.safe_dump(config)

    def test_leaves_the_installed_config_untouched(self, tmp_path):
        stage_weights(tmp_path)
        installed = installed_config()

        local_config(installed, tmp_path)

        assert installed == installed_config()

    @pytest.mark.parametrize(
        "key", ["weights_path", "weights_path_without_voice_cloning", "tokenizer_path"]
    )
    def test_refuses_another_uri_naming_it(self, tmp_path, key):
        stage_weights(tmp_path)
        other = "hf://kyutai/pocket-tts/languages/english/model.safetensors@0000000"
        config = installed_config()
        if key == "tokenizer_path":
            config["flow_lm"]["lookup_table"][key] = other
        else:
            config[key] = other

        with pytest.raises(SpeechEngineError, match="0000000"):
            local_config(config, tmp_path)

    def test_refuses_a_remote_file_anywhere_else_naming_it(self, tmp_path):
        stage_weights(tmp_path)
        config = installed_config()
        config["mimi"]["weights_path_extra"] = "hf://kyutai/other/mimi.safetensors@1234567"

        with pytest.raises(SpeechEngineError, match="kyutai/other/mimi.safetensors@1234567"):
            local_config(config, tmp_path)

    @pytest.mark.parametrize("uri", [CLONE_MODEL_URI, MODEL_URI, TOKENIZER_URI])
    def test_a_missing_mapped_file_raises_naming_its_path(self, tmp_path, uri):
        files = stage_weights(tmp_path)
        files[uri].unlink()

        with pytest.raises(SpeechEngineError) as caught:
            local_config(installed_config(), tmp_path)

        assert str(files[uri]) in str(caught.value)

    def test_a_config_without_its_sections_raises(self, tmp_path):
        stage_weights(tmp_path)

        with pytest.raises(SpeechEngineError, match="flow_lm"):
            local_config({"weights_path": CLONE_MODEL_URI}, tmp_path)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


class TestLoad:
    def test_meets_the_contract(self, tmp_path):
        env = make_env(tmp_path, ("neutral", "calm", "sleepy"))

        synth: SpeechSynthesizer = env.load()

        assert synth.engine_id == "pocket-clone"
        assert synth.model_id == "kyutai/pocket-tts@db04f0e653473e5c7333c69686ac92b7460b0c1d"
        assert synth.sample_rate == RATE
        assert synth.voice == VoiceInfo("pocket-clone", VOICE_ID, PROFILE)
        assert synth.states == frozenset({"neutral", "calm", "sleepy"})

    def test_loads_the_mapped_config_on_four_threads(self, tmp_path):
        env = make_env(tmp_path)

        env.load()

        (config,) = env.library.configs
        assert config == local_config(installed_config(), env.models_dir)
        assert env.library.threads == [4]

    def test_weights_load_only_under_models_dir_and_no_hub_download_runs(self, tmp_path):
        env = make_env(tmp_path)
        hub = pytest.importorskip("huggingface_hub")
        refuse = mock.Mock(side_effect=AssertionError("hub download"))
        with (
            mock.patch.object(hub, "hf_hub_download", refuse),
            mock.patch.object(hub, "snapshot_download", refuse),
        ):
            env.load()

        (config,) = env.library.configs
        paths = [
            config["weights_path"],
            config["weights_path_without_voice_cloning"],
            config["flow_lm"]["lookup_table"]["tokenizer_path"],
        ]
        assert all(Path(path).is_relative_to(env.models_dir) for path in paths)
        refuse.assert_not_called()

    def test_a_missing_weight_file_raises_naming_its_path(self, tmp_path):
        env = make_env(tmp_path)
        missing = mapped_files(env.models_dir)[CLONE_MODEL_URI]
        missing.unlink()

        with pytest.raises(SpeechEngineError) as caught:
            env.load()

        assert str(missing) in str(caught.value)
        assert env.library.configs == []

    def test_builds_one_voice_state_per_state_into_the_cache(self, tmp_path):
        env = make_env(tmp_path, ("neutral", "sleepy"))

        env.load()

        for state in ("neutral", "sleepy"):
            (key_dir,) = (env.cache / "pocket-clone" / VOICE_ID / state).iterdir()
            assert (key_dir / VOICE_STATE_FILE).read_text(encoding="utf-8") == state

    def test_identity_errors_come_before_the_library_loads(self, tmp_path):
        env = make_env(tmp_path)
        (env.voice_root / VOICE_ID / "neutral.wav").write_bytes(b"not a wav")

        with pytest.raises(VoiceAssetError, match="audio_sha256"):
            env.load()

        assert env.library.loads == 0

    def test_no_identity_configured_is_a_voice_asset_error(self, tmp_path):
        env = make_env(tmp_path)
        env.placement = PlacementConfig(models_dir=str(env.models_dir))

        with pytest.raises(VoiceAssetError, match="no voice identity"):
            env.load()

    def test_writes_nothing_under_the_identity(self, tmp_path):
        env = make_env(tmp_path, ("neutral", "calm"))
        before = _listing(env.voice_root)

        synth = env.load()
        collect(synth, segment(state="calm"))

        assert _listing(env.voice_root) == before

    def test_reads_nothing_from_the_cache_after_load(self, tmp_path):
        env = make_env(tmp_path, ("neutral", "calm", "sleepy"))
        synth = env.load()
        model = env.library.models[0]
        calls = list(model.prompt_calls)
        shutil.rmtree(env.cache)

        for state in ("neutral", "calm", "sleepy"):
            assert collect(synth, segment(state=state))

        assert model.prompt_calls == calls

    def test_keeps_the_prompts_ensure_artefacts_loaded(self, tmp_path):
        env = make_env(tmp_path, ("neutral", "calm"))
        synth = env.load()
        model = env.library.models[0]

        collect(synth, segment(state="calm"))

        loaded = [path for path in model.prompt_calls if path.suffix == ".safetensors"]
        assert sorted(path.parent.parent.name for path in loaded) == ["calm", "neutral"]
        assert model.generated[-1][0] == "calm"

    def test_a_cpu_engine_refuses_cuda(self, tmp_path):
        env = make_env(tmp_path)

        with pytest.raises(SpeechEngineError, match="cuda"):
            env.load(device="cuda")

        assert env.library.loads == 0

    def test_a_model_without_cloning_is_refused(self, tmp_path):
        env = make_env(tmp_path, model=FakeModel(has_voice_cloning=False))

        with pytest.raises(SpeechEngineError, match="voice cloning"):
            env.load()

    @pytest.mark.parametrize("error", [RuntimeError("boom"), OSError("gone"), ValueError("bad")])
    def test_library_load_errors_become_speech_engine_errors(self, tmp_path, error):
        env = make_env(tmp_path)
        env.library.fail_load = error

        with pytest.raises(SpeechEngineError) as caught:
            env.load()

        assert not isinstance(caught.value, VoiceAssetError)

    def test_a_build_failure_is_a_voice_asset_error_naming_engine_and_state(self, tmp_path):
        env = make_env(tmp_path, ("neutral", "calm"))
        env.library.fail_export = OSError("disk full")

        with pytest.raises(VoiceAssetError, match="pocket-clone") as caught:
            env.load()

        assert "neutral" in str(caught.value) or "calm" in str(caught.value)

    def test_a_voice_state_failure_is_a_voice_asset_error(self, tmp_path):
        env = make_env(tmp_path, model=FakeModel(fail_prompt=RuntimeError("bad clip")))

        with pytest.raises(VoiceAssetError, match="pocket-clone"):
            env.load()


class TestVoiceBuilder:
    def test_meets_the_builder_contract(self):
        builder: VoiceArtefactBuilder = PocketVoiceBuilder(FakeModel(), lambda state, dest: None)

        assert builder.engine_id == "pocket-clone"
        assert builder.model_pin == "kyutai/pocket-tts@db04f0e653473e5c7333c69686ac92b7460b0c1d"
        assert isinstance(builder.artefact_version, int)

    def test_build_computes_from_the_clip_and_exports_and_never_reads_the_transcript(
        self, tmp_path
    ):
        make_identity(tmp_path / "voices", ("neutral",))
        clip = load_identity(tmp_path / "voices", VOICE_ID).clip("neutral")
        clip.transcript.unlink()
        model = FakeModel()
        exported: list[tuple[str, Path]] = []
        out = tmp_path / "out"
        out.mkdir()

        PocketVoiceBuilder(model, lambda state, dest: exported.append((state.source, dest))).build(
            clip, out
        )

        assert model.prompt_calls == [clip.audio]
        assert exported == [("neutral", out / VOICE_STATE_FILE)]

    def test_load_reads_the_exported_state_file(self, tmp_path):
        (tmp_path / VOICE_STATE_FILE).write_text("calm", encoding="utf-8")
        model = FakeModel()

        state = PocketVoiceBuilder(model, lambda state, dest: None).load(tmp_path)

        assert model.prompt_calls == [tmp_path / VOICE_STATE_FILE]
        assert (state.source, state.loaded) == ("calm", True)

    def test_load_of_a_missing_state_file_raises_naming_it(self, tmp_path):
        with pytest.raises(SpeechEngineError, match=VOICE_STATE_FILE):
            PocketVoiceBuilder(FakeModel(), lambda state, dest: None).load(tmp_path)


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


class TestSpeakingStates:
    def test_a_state_with_a_clip_uses_its_prompt(self, tmp_path):
        env = make_env(tmp_path, ("neutral", "calm", "sleepy"))
        synth = env.load()

        chunks = collect(synth, segment(state="sleepy"))

        assert env.library.models[0].generated[-1][0] == "sleepy"
        assert {chunk.state for chunk in chunks} == {"sleepy"}

    def test_a_state_without_a_clip_falls_back_to_neutral_with_one_info_line(
        self, tmp_path, caplog
    ):
        env = make_env(tmp_path, ("neutral", "calm", "sleepy"))
        synth = env.load()
        caplog.set_level(logging.DEBUG)
        caplog.clear()

        first = collect(synth, segment(state="excited"))
        second = collect(synth, segment(state="excited"))

        assert [state for state, _text in env.library.models[0].generated[-2:]] == [
            "neutral",
            "neutral",
        ]
        assert {chunk.state for chunk in first + second} == {"neutral"}
        assert {chunk.segment.state for chunk in first} == {"excited"}
        fallbacks = [r for r in caplog.records if "excited" in r.getMessage()]
        assert [r.levelno for r in fallbacks] == [logging.INFO]
        assert "pocket-clone" in fallbacks[0].getMessage()
        assert VOICE_ID in fallbacks[0].getMessage()

    def test_a_state_outside_the_vocabulary_warns_once_and_never_raises(self, tmp_path, caplog):
        env = make_env(tmp_path, ("neutral", "calm"))
        synth = env.load()
        caplog.set_level(logging.DEBUG)
        caplog.clear()

        chunks = collect(synth, segment(state="grumpy")) + collect(synth, segment(state="grumpy"))

        assert {chunk.state for chunk in chunks} == {"neutral"}
        fallbacks = [r for r in caplog.records if "grumpy" in r.getMessage()]
        assert [r.levelno for r in fallbacks] == [logging.WARNING]

    def test_each_adapter_logs_its_own_first_fallback(self, tmp_path, caplog):
        env = make_env(tmp_path, ("neutral",))
        first = env.load()
        env.library.model = FakeModel()
        second = env.load()
        caplog.set_level(logging.DEBUG)
        caplog.clear()

        collect(first, segment(state="calm"))
        collect(second, segment(state="calm"))

        assert len([r for r in caplog.records if "calm" in r.getMessage()]) == 2


class TestSynthesis:
    def test_text_is_cleaned_without_tags(self, tmp_path):
        env = make_env(tmp_path)
        synth = env.load()

        collect(synth, segment("Hello [laugh] **NASA** said: yes."))

        assert env.library.models[0].generated[-1][1] == "Hello NASA said: yes."

    def test_a_segment_that_cleans_to_empty_makes_no_engine_call(self, tmp_path):
        env = make_env(tmp_path)
        synth = env.load()

        assert collect(synth, segment("** [sigh] `` ")) == []
        assert env.library.models[0].generated == []

    def test_chunks_are_pcm16_at_24_khz_of_at_most_200_ms_and_carry_the_segment(self, tmp_path):
        env = make_env(tmp_path, model=FakeModel(seconds=1.05))
        synth = env.load()
        seg = segment()

        chunks = collect(synth, seg)

        assert [chunk.samples for chunk in chunks] == [4800] * 5 + [1200]
        assert all(len(chunk.pcm16) == 2 * chunk.samples for chunk in chunks)
        assert all(chunk.segment is seg for chunk in chunks)

    def test_audio_at_another_rate_is_resampled_to_24_khz(self, tmp_path):
        env = make_env(tmp_path, model=FakeModel(sample_rate=16000, seconds=1.0))
        synth = env.load()

        chunks = collect(synth, segment())

        assert sum(chunk.samples for chunk in chunks) == RATE

    def test_loudness_is_normalized_to_minus_20_lufs(self, tmp_path):
        env = make_env(tmp_path, model=FakeModel(amplitude=0.02, seconds=1.0))
        synth = env.load()

        out = pcm_samples(collect(synth, segment()))

        assert integrated_loudness(out, RATE) == pytest.approx(-20.0, abs=1.0)

    def test_a_set_token_stops_between_chunks(self, tmp_path):
        env = make_env(tmp_path)
        synth = env.load()
        token = CancelToken()
        stream = synth.synthesize(segment(), token)

        first = next(stream)
        token.set()

        assert first.samples == 4800
        assert list(stream) == []

    def test_a_token_set_before_the_call_makes_no_engine_call(self, tmp_path):
        env = make_env(tmp_path)
        synth = env.load()
        token = CancelToken()
        token.set()

        assert list(synth.synthesize(segment(), token)) == []
        assert env.library.models[0].generated == []

    def test_a_streaming_model_is_read_piece_by_piece_and_stops_on_the_token(self, tmp_path):
        model = StreamingModel(stream_pieces=4, seconds=1.0)
        env = make_env(tmp_path, model=model)
        synth = env.load()
        token = CancelToken()

        assert sum(chunk.samples for chunk in collect(synth, segment())) == RATE
        pulled = model.pulled

        original = model._tone

        def tone_then_cancel(seconds: float) -> np.ndarray:
            token.set()
            return original(seconds)

        model._tone = tone_then_cancel
        assert list(synth.synthesize(segment(), token)) == []
        assert model.pulled == pulled + 1

    @pytest.mark.parametrize("error", [RuntimeError, ValueError, IndexError, TypeError])
    def test_library_errors_become_speech_engine_errors_without_text(self, tmp_path, error):
        text = f"say {SEGMENT_MARK}"
        model = FakeModel(fail_generate=error(f"cannot say {text}"))
        env = make_env(tmp_path, model=model)
        synth = env.load()

        with pytest.raises(SpeechEngineError) as caught:
            collect(synth, segment(text))

        assert SEGMENT_MARK not in _chain_text(caught.value)


class TestRelease:
    def test_release_drops_the_model_and_the_prompts(self, tmp_path):
        env = make_env(tmp_path, ("neutral", "calm"))
        synth = env.load()
        model_ref = weakref.ref(env.library.models.pop())
        prompts = [weakref.ref(state) for state in synth._prompts.values()]
        gc.collect()
        assert model_ref() is not None

        synth.release()
        gc.collect()

        assert model_ref() is None
        assert [ref() for ref in prompts] == [None, None]

    def test_a_second_release_does_nothing(self, tmp_path):
        synth = make_env(tmp_path).load()

        synth.release()
        synth.release()

    def test_synthesize_after_release_raises(self, tmp_path):
        synth = make_env(tmp_path).load()
        synth.release()

        with pytest.raises(SpeechEngineError, match="released"):
            synth.synthesize(segment(), CancelToken())


# ---------------------------------------------------------------------------
# Privacy and lazy import
# ---------------------------------------------------------------------------


class TestPrivacy:
    def test_no_record_or_error_holds_text_over_load_failure_and_synthesis(self, tmp_path, caplog):
        caplog.set_level(logging.DEBUG)
        env = make_env(tmp_path, ("neutral", "calm"))
        synth = env.load()
        collect(synth, segment(f"Say {SEGMENT_MARK} please.", state="excited"))
        collect(synth, segment(f"Say {SEGMENT_MARK} again.", state="grumpy"))
        collect(synth, segment(f"Say {SEGMENT_MARK} calmly.", state="calm"))

        failing = make_env(tmp_path / "failing", ("neutral",))
        failing.library.fail_export = OSError("disk full")
        with pytest.raises(VoiceAssetError) as build_error:
            failing.load()

        broken = make_env(tmp_path / "broken", model=FakeModel(fail_generate=RuntimeError("x")))
        with pytest.raises(SpeechEngineError) as synth_error:
            collect(broken.load(), segment(f"Say {SEGMENT_MARK}."))

        assert caplog.records
        _assert_private(caplog.records, TRANSCRIPT_MARK, SEGMENT_MARK)
        for caught in (build_error, synth_error):
            text = _chain_text(caught.value)
            assert TRANSCRIPT_MARK not in text
            assert SEGMENT_MARK not in text


class TestLazyImport:
    def test_importing_the_modules_loads_no_engine_package(self, monkeypatch):
        for blocked in ("pocket_tts", "pocket_tts.utils", "pocket_tts.utils.config", "torch"):
            monkeypatch.setitem(sys.modules, blocked, None)
        for name in (pocket_tts.__name__, audio.__name__):
            monkeypatch.delitem(sys.modules, name)

        fresh = importlib.import_module(pocket_tts.__name__)

        assert fresh.PocketCloneSynthesizer.__name__ == "PocketCloneSynthesizer"
        with pytest.raises(SpeechEngineError, match="pocket-tts"):
            fresh.import_pocket()

    def test_the_real_loader_is_never_called_by_load_with_a_fake(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "pocket_tts", None)
        monkeypatch.setitem(sys.modules, "torch", None)

        assert make_env(tmp_path).load().engine_id == "pocket-clone"
