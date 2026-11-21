"""Tests for the Silero VAD and Smart Turn adapters over an injected onnxruntime session factory.

The unit tier has no onnxruntime and no weights, so every test hands the adapters a fake session
factory that records what it was asked to open and returns a scripted session. That pins what the
adapters own: the model inputs they check, the tensors they feed, the state they carry, the
device, threads and path they ask for, and that they never reach for a hub download. The real
sessions run in T15's host tier.
"""

import importlib
import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from backend.realtime.budget import EndpointingConfig, EngineOption, PlacementConfig, VoiceSettings
from backend.realtime.engines import silero_vad, smart_turn
from backend.realtime.engines.silero_vad import SessionSpec, SileroVad
from backend.realtime.engines.smart_turn import SmartTurnDetector
from backend.realtime.errors import SpeechEngineError
from backend.realtime.runtime import EngineLoaders, load_speech_engines
from backend.realtime.speech import NameSpotter, TurnEndDetector, VoiceActivityDetector
from tests.fakes.speech import ScriptedTranscriber

SILERO_INPUTS = ("input", "state", "sr")
SMART_TURN_INPUTS = ("input_features",)
VAD_DEFAULT_RELPATH = Path("speech/onnx-community/silero-vad/onnx/model.onnx")
EOT_DEFAULT_RELPATH = Path("speech/pipecat-ai/smart-turn-v3/smart-turn-v3.2-cpu.onnx")
CPU = "CPUExecutionProvider"
CUDA = "CUDAExecutionProvider"
EIGHT_SECONDS = 8 * 16000


@dataclass(frozen=True, slots=True)
class FakeInput:
    name: str


@dataclass
class FakeSession:
    """A scripted onnxruntime session: records every feed and answers from `respond`."""

    input_names: Sequence[str]
    respond: Callable[[Mapping[str, np.ndarray]], list[np.ndarray]]
    providers: Sequence[str] = (CPU,)
    feeds: list[dict[str, np.ndarray]] = field(default_factory=list)

    def get_inputs(self) -> list[FakeInput]:
        return [FakeInput(name) for name in self.input_names]

    def get_providers(self) -> list[str]:
        return list(self.providers)

    def run(
        self, output_names: Sequence[str] | None, input_feed: Mapping[str, np.ndarray]
    ) -> list[np.ndarray]:
        assert output_names is None
        self.feeds.append({name: np.array(value, copy=True) for name, value in input_feed.items()})
        return self.respond(input_feed)


@dataclass
class FakeFactory:
    """A session factory that records each `SessionSpec` it receives."""

    session: FakeSession
    specs: list[SessionSpec] = field(default_factory=list)

    def __call__(self, spec: SessionSpec) -> FakeSession:
        self.specs.append(spec)
        return self.session


def silero_respond(feed: Mapping[str, np.ndarray]) -> list[np.ndarray]:
    """Speech probability is the window's peak; the next state is the old one plus one."""
    probability = float(np.max(np.abs(feed["input"][:, 64:])))
    return [np.array([[probability]], dtype=np.float32), feed["state"] + 1.0]


def smart_turn_respond(probability: float) -> Callable[[Mapping[str, np.ndarray]], list]:
    def respond(feed: Mapping[str, np.ndarray]) -> list[np.ndarray]:
        return [np.array([[probability]], dtype=np.float32)]

    return respond


def only_cpu() -> list[str]:
    return [CPU]


def with_cuda() -> list[str]:
    return [CUDA, CPU]


def stage(models_dir: Path, relpath: Path) -> Path:
    path = models_dir / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"onnx")
    return path


@pytest.fixture
def models_dir(tmp_path: Path) -> Path:
    stage(tmp_path, VAD_DEFAULT_RELPATH)
    stage(tmp_path, EOT_DEFAULT_RELPATH)
    return tmp_path


def placement(models_dir: Path, **changes: object) -> PlacementConfig:
    return PlacementConfig(models_dir=str(models_dir), **changes)


def load_vad(
    place: PlacementConfig,
    session: FakeSession | None = None,
    providers: Callable[[], list[str]] = only_cpu,
) -> tuple[SileroVad, FakeFactory]:
    factory = FakeFactory(session or FakeSession(SILERO_INPUTS, silero_respond))
    vad = SileroVad.load(place, session_factory=factory, available_providers=providers)
    return vad, factory


def load_eot(
    place: PlacementConfig,
    session: FakeSession | None = None,
    providers: Callable[[], list[str]] = only_cpu,
    endpointing: EndpointingConfig | None = None,
    feature_extractor: Callable[..., object] | None = None,
) -> tuple[SmartTurnDetector, FakeFactory]:
    factory = FakeFactory(session or FakeSession(SMART_TURN_INPUTS, smart_turn_respond(0.5)))
    detector = SmartTurnDetector.load(
        place,
        endpointing or EndpointingConfig(),
        session_factory=factory,
        available_providers=providers,
        feature_extractor_factory=(
            (lambda: feature_extractor) if feature_extractor else smart_turn.whisper_features
        ),
    )
    return detector, factory


def load_either(kind: str, place: PlacementConfig, **kwargs: object) -> FakeFactory:
    if kind == "vad":
        return load_vad(place, **kwargs)[1]
    return load_eot(place, **kwargs)[1]


def window(value: float) -> np.ndarray:
    return np.full(512, value, dtype=np.float32)


# --- Silero VAD -----------------------------------------------------------------------------


class TestSileroInputs:
    def test_load_accepts_the_v5_inputs(self, models_dir: Path) -> None:
        vad, _ = load_vad(placement(models_dir))
        detector: VoiceActivityDetector = vad
        assert detector.open_stream() is not None

    def test_load_names_each_missing_and_unexpected_input(self, models_dir: Path) -> None:
        legacy = FakeSession(("input", "sr", "h", "c"), silero_respond)
        with pytest.raises(SpeechEngineError) as caught:
            load_vad(placement(models_dir), session=legacy)
        message = str(caught.value)
        assert "state" in message
        assert "'h'" in message and "'c'" in message
        assert str(models_dir / VAD_DEFAULT_RELPATH) in message


class TestSileroStream:
    def test_first_window_feeds_zero_context_zero_state_and_the_rate(
        self, models_dir: Path
    ) -> None:
        session = FakeSession(SILERO_INPUTS, silero_respond)
        vad, _ = load_vad(placement(models_dir), session=session)
        probability = vad.open_stream().score(window(0.25))

        assert probability == pytest.approx(0.25)
        feed = session.feeds[0]
        assert feed["input"].shape == (1, 576)
        assert feed["input"].dtype == np.float32
        assert np.all(feed["input"][:, :64] == 0.0)
        assert np.all(feed["input"][:, 64:] == np.float32(0.25))
        assert feed["state"].shape == (2, 1, 128)
        assert feed["state"].dtype == np.float32
        assert np.all(feed["state"] == 0.0)
        assert feed["sr"].dtype == np.int64
        assert int(feed["sr"]) == 16000

    def test_state_and_context_carry_across_windows(self, models_dir: Path) -> None:
        session = FakeSession(SILERO_INPUTS, silero_respond)
        vad, _ = load_vad(placement(models_dir), session=session)
        stream = vad.open_stream()
        first = np.arange(512, dtype=np.float32) / 512.0
        stream.score(first)
        stream.score(window(0.5))
        stream.score(window(0.1))

        assert [float(feed["state"].flat[0]) for feed in session.feeds] == [0.0, 1.0, 2.0]
        np.testing.assert_array_equal(session.feeds[1]["input"][0, :64], first[-64:])
        np.testing.assert_array_equal(session.feeds[2]["input"][0, :64], window(0.5)[-64:])

    def test_reset_clears_state_and_context(self, models_dir: Path) -> None:
        session = FakeSession(SILERO_INPUTS, silero_respond)
        vad, _ = load_vad(placement(models_dir), session=session)
        stream = vad.open_stream()
        stream.score(window(0.5))
        stream.score(window(0.5))
        stream.reset()
        stream.score(window(0.5))

        last = session.feeds[-1]
        assert np.all(last["state"] == 0.0)
        assert np.all(last["input"][:, :64] == 0.0)

    def test_each_stream_starts_fresh_and_keeps_its_own_state(self, models_dir: Path) -> None:
        session = FakeSession(SILERO_INPUTS, silero_respond)
        vad, _ = load_vad(placement(models_dir), session=session)
        first = vad.open_stream()
        first.score(window(0.5))
        first.score(window(0.5))
        second = vad.open_stream()
        second.score(window(0.5))
        first.score(window(0.5))

        assert [float(feed["state"].flat[0]) for feed in session.feeds] == [0.0, 1.0, 0.0, 2.0]

    @pytest.mark.parametrize("samples", [256, 513, 1024])
    def test_a_window_that_is_not_512_samples_raises(self, models_dir: Path, samples: int) -> None:
        vad, _ = load_vad(placement(models_dir))
        with pytest.raises(SpeechEngineError, match="512"):
            vad.open_stream().score(np.zeros(samples, dtype=np.float32))

    def test_vad_session_runs_single_threaded(self, models_dir: Path) -> None:
        _, factory = load_vad(placement(models_dir))
        assert factory.specs[0].intra_op_num_threads == 1
        assert factory.specs[0].inter_op_num_threads == 1


# --- Smart Turn ------------------------------------------------------------------------------


@dataclass
class RecordingExtractor:
    """Stands in for `WhisperFeatureExtractor`: records its audio and keyword arguments."""

    calls: list[tuple[np.ndarray, dict[str, object]]] = field(default_factory=list)

    def __call__(self, audio: np.ndarray, **kwargs: object) -> object:
        self.calls.append((np.array(audio, copy=True), kwargs))
        return _Features(np.zeros((1, 80, 800), dtype=np.float32))


@dataclass(frozen=True, slots=True)
class _Features:
    input_features: np.ndarray


class TestSmartTurnInputs:
    def test_load_names_a_missing_input_features(self, models_dir: Path) -> None:
        session = FakeSession(("audio",), smart_turn_respond(0.5))
        with pytest.raises(SpeechEngineError) as caught:
            load_eot(placement(models_dir), session=session)
        assert "input_features" in str(caught.value)
        assert "'audio'" in str(caught.value)


class TestSmartTurnFeatures:
    def test_whisper_features_reach_the_session_as_one_by_80_by_800(self, models_dir: Path) -> None:
        session = FakeSession(SMART_TURN_INPUTS, smart_turn_respond(0.8))
        detector, _ = load_eot(placement(models_dir), session=session)
        turn_end: TurnEndDetector = detector
        rng = np.random.default_rng(7)
        probability = turn_end.probability_complete(
            rng.uniform(-0.1, 0.1, 3 * 16000).astype(np.float32)
        )

        assert probability == pytest.approx(0.8)
        features = session.feeds[0]["input_features"]
        assert features.shape == (1, 80, 800)
        assert features.dtype == np.float32

    def test_extractor_is_called_as_inference_py_calls_it(self, models_dir: Path) -> None:
        extractor = RecordingExtractor()
        detector, _ = load_eot(placement(models_dir), feature_extractor=extractor)
        detector.probability_complete(np.ones(16000, dtype=np.float32))

        _, kwargs = extractor.calls[0]
        assert kwargs == {
            "sampling_rate": 16000,
            "return_tensors": "np",
            "padding": "max_length",
            "max_length": EIGHT_SECONDS,
            "truncation": True,
            "do_normalize": True,
        }

    def test_only_the_last_eight_seconds_are_kept(self, models_dir: Path) -> None:
        extractor = RecordingExtractor()
        detector, _ = load_eot(placement(models_dir), feature_extractor=extractor)
        audio = np.arange(10 * 16000, dtype=np.float32)
        detector.probability_complete(audio)

        seen, _ = extractor.calls[0]
        assert seen.dtype == np.float32
        np.testing.assert_array_equal(seen, audio[-EIGHT_SECONDS:])

    def test_short_audio_is_zero_padded_at_the_front_to_eight_seconds(
        self, models_dir: Path
    ) -> None:
        extractor = RecordingExtractor()
        detector, _ = load_eot(placement(models_dir), feature_extractor=extractor)
        audio = np.linspace(0.1, 0.9, 2 * 16000, dtype=np.float32)
        detector.probability_complete(audio)

        seen, _ = extractor.calls[0]
        assert seen.shape == (EIGHT_SECONDS,)
        assert np.all(seen[: EIGHT_SECONDS - len(audio)] == 0.0)
        np.testing.assert_array_equal(seen[-len(audio) :], audio)

    def test_multichannel_audio_is_refused(self, models_dir: Path) -> None:
        detector, _ = load_eot(placement(models_dir), feature_extractor=RecordingExtractor())
        with pytest.raises(SpeechEngineError, match="mono"):
            detector.probability_complete(np.zeros((2, 16000), dtype=np.float32))


class TestSmartTurnThreshold:
    @pytest.mark.parametrize(
        ("threshold", "probability", "complete"),
        [(0.5, 0.5, False), (0.5, 0.51, True), (0.5, 0.2, False), (0.3, 0.35, True)],
    )
    def test_complete_only_above_the_endpointing_threshold(
        self, models_dir: Path, threshold: float, probability: float, complete: bool
    ) -> None:
        session = FakeSession(SMART_TURN_INPUTS, smart_turn_respond(probability))
        detector, _ = load_eot(
            placement(models_dir),
            session=session,
            endpointing=EndpointingConfig(eot_threshold=threshold),
            feature_extractor=RecordingExtractor(),
        )
        assert detector.is_complete(np.zeros(16000, dtype=np.float32)) is complete


class TestSmartTurnThreads:
    @pytest.mark.parametrize("threads", [1, 4, 6])
    def test_intra_op_threads_equal_eot_threads(self, models_dir: Path, threads: int) -> None:
        _, factory = load_eot(placement(models_dir, eot_threads=threads))
        assert factory.specs[0].intra_op_num_threads == threads
        assert factory.specs[0].inter_op_num_threads == 1


# --- Shared: devices, paths, no downloads, lazy imports --------------------------------------

ADAPTERS = [
    pytest.param("vad", "vad_device", "MIST_VOICE_VAD_DEVICE", id="silero"),
    pytest.param("eot", "eot_device", "MIST_VOICE_EOT_DEVICE", id="smart-turn"),
]


class TestDeviceSelection:
    @pytest.mark.parametrize(("kind", "field_name", "env_name"), ADAPTERS)
    def test_cpu_selects_only_the_cpu_provider(
        self, models_dir: Path, kind: str, field_name: str, env_name: str
    ) -> None:
        factory = load_either(
            kind, placement(models_dir, **{field_name: "cpu"}), providers=with_cuda
        )
        assert factory.specs[0].providers == (CPU,)

    @pytest.mark.parametrize(("kind", "field_name", "env_name"), ADAPTERS)
    @pytest.mark.parametrize(("device", "device_id"), [("cuda", 0), ("cuda:1", 1)])
    def test_cuda_selects_the_cuda_provider_when_listed(
        self,
        models_dir: Path,
        kind: str,
        field_name: str,
        env_name: str,
        device: str,
        device_id: int,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        session_inputs = SILERO_INPUTS if kind == "vad" else SMART_TURN_INPUTS
        session = FakeSession(session_inputs, silero_respond, providers=(CUDA, CPU))
        with caplog.at_level(logging.WARNING):
            factory = load_either(
                kind,
                placement(models_dir, **{field_name: device}),
                session=session,
                providers=with_cuda,
            )
        assert factory.specs[0].providers == (CUDA, CPU)
        assert factory.specs[0].device_id == device_id
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    @pytest.mark.parametrize(("kind", "field_name", "env_name"), ADAPTERS)
    def test_cuda_falls_back_to_cpu_with_a_warning_naming_the_setting(
        self,
        models_dir: Path,
        kind: str,
        field_name: str,
        env_name: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.WARNING):
            factory = load_either(
                kind, placement(models_dir, **{field_name: "cuda"}), providers=only_cpu
            )
        assert factory.specs[0].providers == (CPU,)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert env_name in warnings[0].getMessage()
        assert CUDA in warnings[0].getMessage()

    @pytest.mark.parametrize(("kind", "field_name", "env_name"), ADAPTERS)
    def test_a_session_that_dropped_cuda_is_reported(
        self,
        models_dir: Path,
        kind: str,
        field_name: str,
        env_name: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        session_inputs = SILERO_INPUTS if kind == "vad" else SMART_TURN_INPUTS
        session = FakeSession(session_inputs, silero_respond, providers=(CPU,))
        with caplog.at_level(logging.WARNING):
            load_either(
                kind,
                placement(models_dir, **{field_name: "cuda"}),
                session=session,
                providers=with_cuda,
            )
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert env_name in warnings[0].getMessage()


class TestModelPath:
    @pytest.mark.parametrize(
        ("kind", "relpath"), [("vad", VAD_DEFAULT_RELPATH), ("eot", EOT_DEFAULT_RELPATH)]
    )
    def test_default_setting_opens_the_pinned_file_under_models_dir(
        self, models_dir: Path, kind: str, relpath: Path
    ) -> None:
        factory = load_either(kind, placement(models_dir))
        assert Path(factory.specs[0].path) == models_dir / relpath

    @pytest.mark.parametrize(
        ("kind", "field_name", "inputs"),
        [("vad", "vad_model", SILERO_INPUTS), ("eot", "eot_model", SMART_TURN_INPUTS)],
    )
    def test_a_non_default_setting_moves_the_path(
        self, models_dir: Path, kind: str, field_name: str, inputs: tuple[str, ...]
    ) -> None:
        moved = stage(models_dir, Path("speech/example-org/other-model/weights/v9.onnx"))
        setting = "example-org/other-model@0123456789abcdef:weights/v9.onnx"
        session = FakeSession(inputs, silero_respond)
        factory = load_either(kind, placement(models_dir, **{field_name: setting}), session=session)
        assert Path(factory.specs[0].path) == moved

    @pytest.mark.parametrize("kind", ["vad", "eot"])
    def test_a_missing_file_raises_naming_its_path(self, tmp_path: Path, kind: str) -> None:
        relpath = VAD_DEFAULT_RELPATH if kind == "vad" else EOT_DEFAULT_RELPATH
        with pytest.raises(SpeechEngineError) as caught:
            load_either(kind, placement(tmp_path))
        assert str(tmp_path / relpath) in str(caught.value)

    @pytest.mark.parametrize(
        ("field_name", "setting"),
        [
            ("vad_model", "../escape@abc:model.onnx"),
            ("eot_model", "org/repo@abc:../../model.onnx"),
            ("eot_model", "org/repo@abc:/etc/model.onnx"),
            ("vad_model", "org/repo-without-revision:model.onnx"),
        ],
    )
    def test_a_setting_that_leaves_models_dir_or_is_malformed_raises(
        self, models_dir: Path, field_name: str, setting: str
    ) -> None:
        kind = "vad" if field_name == "vad_model" else "eot"
        with pytest.raises(SpeechEngineError, match=f"MIST_VOICE_{field_name.upper()}"):
            load_either(kind, placement(models_dir, **{field_name: setting}))


class TestModelId:
    """The engine walk reports each loaded engine's `model_id`, so both adapters carry one."""

    def test_the_engine_walk_reports_the_placement_models(self, models_dir: Path) -> None:
        place = placement(
            models_dir, stt_chain=(EngineOption("parakeet-cpu", "cpu"),), tts_chain=()
        )

        def no_spotter(placement: PlacementConfig) -> NameSpotter:
            raise AssertionError("no kws_model is set, so no spotter loads")

        loaders = EngineLoaders(
            chains={"parakeet-cpu": lambda settings, device: ScriptedTranscriber([])},
            vad=lambda placement: load_vad(placement)[0],
            eot=lambda placement, endpointing: load_eot(
                placement, endpointing=endpointing, feature_extractor=RecordingExtractor()
            )[0],
            kws=no_spotter,
        )

        engines = load_speech_engines(VoiceSettings(placement=place), loaders)

        assert (engines.statuses.vad.location, engines.statuses.vad.model_id) == (
            "cpu",
            place.vad_model,
        )
        assert (engines.statuses.eot.location, engines.statuses.eot.model_id) == (
            "cpu",
            place.eot_model,
        )

    @pytest.mark.parametrize(
        ("kind", "field_name", "inputs"),
        [("vad", "vad_model", SILERO_INPUTS), ("eot", "eot_model", SMART_TURN_INPUTS)],
    )
    def test_the_model_id_is_the_setting_the_engine_loaded_from(
        self, models_dir: Path, kind: str, field_name: str, inputs: tuple[str, ...]
    ) -> None:
        stage(models_dir, Path("speech/example-org/other-model/weights/v9.onnx"))
        setting = "example-org/other-model@0123456789abcdef:weights/v9.onnx"
        place = placement(models_dir, **{field_name: setting})
        session = FakeSession(inputs, silero_respond)
        if kind == "vad":
            engine: VoiceActivityDetector | TurnEndDetector = load_vad(place, session=session)[0]
        else:
            engine = load_eot(place, session=session, feature_extractor=RecordingExtractor())[0]
        assert engine.model_id == setting


class TestNoDownload:
    def test_a_hub_download_patched_to_fail_is_never_called(
        self, models_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        huggingface_hub = pytest.importorskip("huggingface_hub")
        calls: list[str] = []

        def refuse(*args: object, **kwargs: object) -> None:
            calls.append("download")
            raise AssertionError("the adapters must not download weights")

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", refuse)
        monkeypatch.setattr(huggingface_hub, "snapshot_download", refuse)

        vad, vad_factory = load_vad(placement(models_dir))
        eot, eot_factory = load_eot(placement(models_dir), feature_extractor=RecordingExtractor())
        vad.open_stream().score(window(0.5))
        eot.probability_complete(np.zeros(16000, dtype=np.float32))

        assert calls == []
        assert Path(vad_factory.specs[0].path).is_relative_to(models_dir)
        assert Path(eot_factory.specs[0].path).is_relative_to(models_dir)


class TestLazyImports:
    def test_importing_the_adapters_loads_no_engine_package(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for blocked in ("onnxruntime", "transformers", "huggingface_hub"):
            monkeypatch.setitem(sys.modules, blocked, None)
        for name in (silero_vad.__name__, smart_turn.__name__):
            monkeypatch.delitem(sys.modules, name)

        fresh_vad = importlib.import_module(silero_vad.__name__)
        fresh_turn = importlib.import_module(smart_turn.__name__)

        assert fresh_vad.SileroVad.__name__ == "SileroVad"
        assert fresh_turn.SmartTurnDetector.__name__ == "SmartTurnDetector"

    def test_the_real_session_factory_imports_onnxruntime_only_when_called(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "onnxruntime", None)
        spec = SessionSpec(
            path=str(tmp_path / "model.onnx"),
            providers=(CPU,),
            device_id=0,
            intra_op_num_threads=1,
            inter_op_num_threads=1,
        )
        with pytest.raises(SpeechEngineError, match="onnxruntime"):
            silero_vad.open_onnx_session(spec)
        with pytest.raises(SpeechEngineError, match="onnxruntime"):
            silero_vad.onnxruntime_providers()
