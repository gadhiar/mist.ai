"""T12: the speech engine walk, `build_mist_runtime`'s composition, and the fallback responder.

Every engine loads through a fake loader, and the GPU meter reads a fake torch: no test imports
a speech library or loads a model.
"""

from __future__ import annotations

import dataclasses
import logging
import sys
from types import SimpleNamespace

import pytest

from backend.errors import Neo4jConnectionError
from backend.factories import _real_engine_loaders, build_mist_runtime, build_speech_engines
from backend.knowledge.config import LLMConfig
from backend.realtime.budget import (
    BUDGET_PRESETS,
    CharacterConfig,
    EngineOption,
    PlacementConfig,
    VoiceSettings,
    load_voice_settings,
)
from backend.realtime.degraded import DirectResponder
from backend.realtime.errors import SpeechEngineError
from backend.realtime.lanes import InferenceLane
from backend.realtime.runtime import EngineLoaders, TorchGpuMeter
from backend.realtime.session import InferenceLanes, tts_voice_of
from backend.realtime.turn import TurnPlacement
from backend.realtime.voice_assets import VoiceAssetError
from tests.fakes.speech import (
    FakeSynthesizer,
    ScriptedNameSpotter,
    ScriptedTranscriber,
    ScriptedWakeVerifier,
)

from .turn_fakes import InlineLane

RUNTIME_LOGGER = "backend.realtime.runtime"


class FakeEngine(SimpleNamespace):
    pass


class Loaders:
    """Fake loaders that record each load in order; `fail` names options that raise.

    The verifier is built for the spotter `verifier_spotter` names, `mist-kws` (the settings'
    `kws_model`) unless a test gives another; `verify` False builds no verifier loader.
    """

    def __init__(
        self,
        *,
        fail: dict[str, Exception] | None = None,
        verifier_spotter: str = "mist-kws",
        verify: bool = True,
    ) -> None:
        self.fail = fail or {}
        self.calls: list[str] = []
        self.verifier_spotter = verifier_spotter
        self.verify = verify
        self.verifiers: list[ScriptedWakeVerifier] = []

    def _verifier(self) -> ScriptedWakeVerifier:
        verifier = ScriptedWakeVerifier(spotter_model_id=self.verifier_spotter)
        self.verifiers.append(verifier)
        return verifier

    def _load(self, name: str, engine: object) -> object:
        self.calls.append(name)
        if name in self.fail:
            raise self.fail[name]
        return engine

    def build(self) -> EngineLoaders:
        def chain_loader(engine_id: str, factory):
            return lambda settings, device: self._load(f"{engine_id}@{device}", factory())

        return EngineLoaders(
            chains={
                "nemotron-160": chain_loader(
                    "nemotron-160", lambda: ScriptedTranscriber([], model_id="nemotron-model")
                ),
                "parakeet-cpu": chain_loader(
                    "parakeet-cpu", lambda: ScriptedTranscriber([], model_id="parakeet-model")
                ),
                "pocket-clone": chain_loader(
                    "pocket-clone", lambda: FakeSynthesizer(model_id="pocket-model")
                ),
            },
            vad=lambda placement: self._load("vad", FakeEngine(model_id="vad-model")),
            eot=lambda placement, endpointing: self._load("eot", FakeEngine(model_id="eot-model")),
            kws=lambda placement: self._load(
                "kws", ScriptedNameSpotter(model_id="kws-model", hit_lag_ms=120)
            ),
            verify=(
                (lambda placement: self._load("verify", self._verifier())) if self.verify else None
            ),
        )


@pytest.fixture(autouse=True)
def _no_torch(monkeypatch):
    """The default GPU meter imports torch lazily; here it finds a torch with no device."""
    no_device = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", no_device)


def settings(**placement: object) -> VoiceSettings:
    defaults: dict[str, object] = {
        "kws_model": "mist-kws",
        "verify_model": "mist-verify",
        "tts_voice": "mist-v1",
    }
    defaults.update(placement)
    return VoiceSettings(placement=PlacementConfig(**defaults))  # type: ignore[arg-type]


def spotter_skips(caplog) -> list[str]:
    """The runtime's WARNINGs that say why the server spotter was not loaded."""
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == RUNTIME_LOGGER
        and r.levelno == logging.WARNING
        and r.getMessage().startswith("kws:")
    ]


def errors(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == RUNTIME_LOGGER and r.levelno == logging.ERROR
    ]


class FakeCuda:
    def __init__(self, reserved_mib: list[int]) -> None:
        self.reserved = list(reserved_mib)
        self.allocated = 0
        self.calls: list[str] = []

    def is_available(self) -> bool:
        return True

    def memory_allocated(self) -> int:
        self.calls.append("memory_allocated")
        return self.allocated

    def reset_peak_memory_stats(self) -> None:
        self.calls.append("reset_peak_memory_stats")

    def max_memory_reserved(self) -> int:
        self.calls.append("max_memory_reserved")
        return self.reserved.pop(0) * 1024 * 1024


class TestEngineWalk:
    def test_a_failing_gpu_option_falls_to_the_cpu_option(self, caplog):
        loaders = Loaders(fail={"nemotron-160@cuda": SpeechEngineError("no CUDA")})

        with caplog.at_level(logging.WARNING, logger=RUNTIME_LOGGER):
            engines = build_speech_engines(settings(), loaders=loaders.build())

        assert engines.transcriber.model_id == "parakeet-model"
        assert (engines.statuses.stt.location, engines.statuses.stt.model_id) == (
            "cpu",
            "parakeet-model",
        )
        assert any("nemotron-160@cuda" in r.getMessage() for r in caplog.records)

    def test_engines_load_one_at_a_time_stt_first(self):
        loaders = Loaders()

        build_speech_engines(settings(), loaders=loaders.build())

        assert loaders.calls == [
            "nemotron-160@cuda",
            "vad",
            "eot",
            "verify",
            "kws",
            "pocket-clone@cpu",
        ]

    @pytest.mark.parametrize("chain", [(), (EngineOption("parakeet-cpu", "cpu"),)])
    def test_an_empty_or_failed_chain_leaves_the_slot_off(self, chain):
        loaders = Loaders(fail={"parakeet-cpu@cpu": SpeechEngineError("broken")})

        engines = build_speech_engines(settings(stt_chain=chain), loaders=loaders.build())

        assert engines.transcriber is None
        assert engines.statuses.stt.location == "off"

    def test_stt_off_loads_no_vad_eot_or_spotter(self):
        loaders = Loaders()

        engines = build_speech_engines(settings(stt_chain=()), loaders=loaders.build())

        assert loaders.calls == ["pocket-clone@cpu"]
        assert (engines.vad, engines.turn_end, engines.spotter) == (None, None, None)
        assert engines.statuses.vad.location == "off"

    def test_the_vad_and_eot_statuses_carry_their_engines_model_ids(self):
        engines = build_speech_engines(settings(), loaders=Loaders().build())

        assert (engines.statuses.vad.location, engines.statuses.vad.model_id) == (
            "cpu",
            "vad-model",
        )
        assert (engines.statuses.eot.location, engines.statuses.eot.model_id) == (
            "cpu",
            "eot-model",
        )

    def test_an_unknown_id_or_unrecorded_device_is_skipped_with_one_error(self, caplog):
        chain = (
            EngineOption("whisper-big", "cuda"),
            EngineOption("parakeet-cpu", "cuda"),
            EngineOption("parakeet-cpu", "cpu"),
        )
        loaders = Loaders()

        with caplog.at_level(logging.ERROR, logger=RUNTIME_LOGGER):
            engines = build_speech_engines(settings(stt_chain=chain), loaders=loaders.build())

        assert engines.statuses.stt.location == "cpu"
        assert loaders.calls[0] == "parakeet-cpu@cpu"
        found = errors(caplog)
        assert len(found) == 2
        assert "whisper-big" in found[0] and "parakeet-cpu" in found[1]

    def test_a_tts_engine_in_the_stt_chain_is_skipped(self, caplog):
        chain = (EngineOption("pocket-clone", "cpu"), EngineOption("parakeet-cpu", "cpu"))

        with caplog.at_level(logging.ERROR, logger=RUNTIME_LOGGER):
            engines = build_speech_engines(settings(stt_chain=chain), loaders=Loaders().build())

        assert engines.statuses.stt.model_id == "parakeet-model"
        assert len(errors(caplog)) == 1

    def test_a_voice_asset_error_leaves_tts_off_with_one_error_and_no_transcript(self, caplog):
        error = VoiceAssetError("pocket-clone: voice 'mist-v1' clips/a.wav is missing")
        loaders = Loaders(fail={"pocket-clone@cpu": error})

        with caplog.at_level(logging.DEBUG):
            engines = build_speech_engines(settings(), loaders=loaders.build())

        assert engines.synthesizer is None
        assert engines.statuses.tts.location == "off"
        found = errors(caplog)
        assert len(found) == 1
        assert "pocket-clone" in found[0] and "clips/a.wav" in found[0]

    def test_no_kws_model_or_a_kws_load_error_leaves_the_spotter_off(self):
        no_model = build_speech_engines(settings(kws_model=None), loaders=Loaders().build())
        failing = build_speech_engines(
            settings(),
            loaders=Loaders(fail={"kws": SpeechEngineError("manifest refused")}).build(),
        )

        for engines in (no_model, failing):
            assert engines.spotter is None
            assert engines.statuses.kws.location == "off"

    def test_the_verifier_loads_beside_the_spotter(self, caplog):
        loaders = Loaders()

        with caplog.at_level(logging.WARNING, logger=RUNTIME_LOGGER):
            engines = build_speech_engines(settings(), loaders=loaders.build())

        assert engines.verifier is loaders.verifiers[0]
        assert engines.spotter is not None
        assert engines.statuses.kws.location == "cpu"
        assert spotter_skips(caplog) == []

    def test_a_verifier_loads_without_a_spotter(self):
        loaders = Loaders()

        engines = build_speech_engines(settings(kws_model=None), loaders=loaders.build())

        assert engines.verifier is loaders.verifiers[0]
        assert "kws" not in loaders.calls

    @pytest.mark.parametrize(
        ("placement", "loader_args", "reason"),
        [
            pytest.param({"verify_model": None}, {}, "MIST_VOICE_VERIFY_MODEL", id="no-model"),
            pytest.param({}, {"verify": False}, "did not load", id="no-loader"),
            pytest.param(
                {},
                {"fail": {"verify": SpeechEngineError("manifest refused")}},
                "did not load",
                id="load-error",
            ),
            pytest.param(
                {}, {"verifier_spotter": "mist-kws-old"}, "mist-kws-old", id="other-spotter"
            ),
        ],
    )
    def test_without_a_matching_verifier_the_spotter_is_never_loaded(
        self, caplog, placement, loader_args, reason
    ):
        loaders = Loaders(**loader_args)

        with caplog.at_level(logging.WARNING, logger=RUNTIME_LOGGER):
            engines = build_speech_engines(settings(**placement), loaders=loaders.build())

        assert "kws" not in loaders.calls
        assert engines.spotter is None
        assert engines.statuses.kws.location == "off"
        skips = spotter_skips(caplog)
        assert len(skips) == 1
        assert reason in skips[0] and "mist-kws" in skips[0]

    def test_a_verifier_for_another_spotter_still_serves_the_endpoint(self):
        loaders = Loaders(verifier_spotter="mist-kws-old")

        engines = build_speech_engines(settings(), loaders=loaders.build())

        assert engines.verifier is loaders.verifiers[0]

    def test_power_off_skips_every_cuda_option_and_walks_on(self):
        loaders = Loaders()

        engines = build_speech_engines(settings(), loaders=loaders.build(), power_level="off")

        assert "nemotron-160@cuda" not in loaders.calls
        assert engines.statuses.stt.location == "cpu"

    def test_power_off_with_only_cuda_options_leaves_the_slot_off(self):
        chain = (EngineOption("nemotron-160", "cuda"),)

        engines = build_speech_engines(
            settings(stt_chain=chain), loaders=Loaders().build(), power_level="off"
        )

        assert engines.transcriber is None


class TestVram:
    def test_each_gpu_engine_is_measured_and_a_sum_over_budget_warns(self, caplog):
        cuda = FakeCuda(reserved_mib=[5000])
        meter = TorchGpuMeter(lambda: SimpleNamespace(cuda=cuda))

        with caplog.at_level(logging.WARNING, logger=RUNTIME_LOGGER):
            engines = build_speech_engines(settings(), loaders=Loaders().build(), gpu_meter=meter)

        assert engines.statuses.stt.vram_mib == 5000
        assert engines.vram_over_budget is True
        assert any("over the 4500 MiB budget" in r.getMessage() for r in caplog.records)
        assert cuda.calls.index("reset_peak_memory_stats") < cuda.calls.index("max_memory_reserved")

    def test_a_cpu_only_placement_measures_nothing(self):
        cuda = FakeCuda(reserved_mib=[])
        meter = TorchGpuMeter(lambda: SimpleNamespace(cuda=cuda))
        chain = (EngineOption("parakeet-cpu", "cpu"),)

        engines = build_speech_engines(
            settings(stt_chain=chain), loaders=Loaders().build(), gpu_meter=meter
        )

        assert cuda.calls == []
        assert engines.vram_over_budget is False
        assert engines.statuses.stt.vram_mib is None


class TestVoiceReport:
    def test_tts_voice_reports_the_synthesizers_identity(self):
        engines = build_speech_engines(settings(), loaders=Loaders().build())

        voice = tts_voice_of(engines, settings())

        synthesizer = engines.synthesizer
        assert (voice.engine, voice.voice, voice.profile) == (
            synthesizer.voice.engine,
            synthesizer.voice.voice,
            synthesizer.voice.profile,
        )

    def test_with_tts_off_the_voice_names_the_configured_identity_on_engine_off(self):
        engines = build_speech_engines(settings(tts_chain=()), loaders=Loaders().build())

        voice = tts_voice_of(engines, settings())

        assert (voice.engine, voice.voice, voice.profile) == ("off", "mist-v1", "mist")


# -- build_mist_runtime ----------------------------------------------------------------------------


class FakeProvider:
    def __init__(self) -> None:
        self.warmups: list[object] = []

    def invoke_sync(self, request):
        self.warmups.append(request)


class FakeEmbedder:
    def __init__(self) -> None:
        self.warmed = 0

    def warmup(self) -> None:
        self.warmed += 1


def knowledge_config(*, enabled: bool = True, first_chars: int = 24) -> SimpleNamespace:
    return SimpleNamespace(
        enable_knowledge_integration=enabled,
        llm=LLMConfig(voice_first_segment_min_chars=first_chars),
    )


def inline_lanes() -> InferenceLanes:
    return InferenceLanes(
        vad=InlineLane(),
        eot=InlineLane(),
        stt=InlineLane(),
        tts=InlineLane(),
        kws=InlineLane(),
        verify=InlineLane(),
    )


class HandlerFactory:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[dict[str, object]] = []
        self.embedder = FakeEmbedder()

    def __call__(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(graph_store=SimpleNamespace(embedding_generator=self.embedder))


def runtime(factory: HandlerFactory, *, enabled: bool = True, **kwargs: object):
    provider = FakeProvider()
    built = build_mist_runtime(
        knowledge_config(enabled=enabled),  # type: ignore[arg-type]
        voice_settings=kwargs.pop("voice_settings", settings()),  # type: ignore[arg-type]
        llm_provider=provider,  # type: ignore[arg-type]
        handler_factory=factory,
        engine_loaders=Loaders().build(),
        lanes=inline_lanes(),
        **kwargs,  # type: ignore[arg-type]
    )
    return built, provider


class TestBuildMistRuntime:
    def test_the_handler_answers_turns_and_the_llm_and_embedder_are_warmed(self):
        factory = HandlerFactory()

        built, provider = runtime(factory)

        assert built.responder is built.conversation_handler
        assert built.agent_available is True
        assert factory.calls[0]["llm_provider"] is provider
        assert [(r.messages, r.max_tokens) for r in provider.warmups] == [
            ([{"role": "user", "content": "hi"}], 1)
        ]
        assert factory.embedder.warmed == 1
        assert built.services.responder is built.responder

    def test_knowledge_off_answers_with_the_direct_responder(self, caplog):
        factory = HandlerFactory()

        with caplog.at_level(logging.WARNING, logger="backend.factories"):
            built, _ = runtime(factory, enabled=False)

        assert isinstance(built.responder, DirectResponder)
        assert built.agent_available is False
        assert factory.calls == []
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_an_unreachable_graph_degrades_to_the_direct_responder(self, caplog):
        factory = HandlerFactory(Neo4jConnectionError("bolt://graph:7687 refused"))

        with caplog.at_level(logging.WARNING, logger="backend.factories"):
            built, _ = runtime(factory)

        assert isinstance(built.responder, DirectResponder)
        assert built.conversation_handler is None
        assert built.agent_available is False
        assert any("refused" in r.getMessage() for r in caplog.records)

    def test_any_other_error_fails_startup(self):
        with pytest.raises(RuntimeError, match="wiring bug"):
            runtime(HandlerFactory(RuntimeError("wiring bug")))

    def test_power_off_skips_the_llm_warmup(self):
        built, provider = runtime(HandlerFactory(), power_level=lambda: "off")

        assert provider.warmups == []
        assert built.services.power_level() == "off"

    def test_the_first_segment_cut_comes_from_the_llm_config(self):
        built, _ = runtime(HandlerFactory())

        assert built.settings.segmenter.first_segment_min_chars == 24
        assert built.services.settings is built.settings

    def test_power_level_defaults_to_full_and_placement_is_primary(self):
        built, _ = runtime(HandlerFactory())

        assert built.services.power_level() == "full"
        assert built.services.placement() == TurnPlacement(
            mode="primary", stt="cuda", tts="cpu", budget=built.settings.budget
        )
        assert built.settings.budget == BUDGET_PRESETS[built.settings.budget.preset]

    @pytest.mark.parametrize(
        ("environ", "character"),
        [({}, CharacterConfig()), ({"MIST_VOICE_CHARACTER": "false"}, None)],
    )
    def test_the_controllers_playback_config_carries_the_character_stage(self, environ, character):
        voice = load_voice_settings(environ)
        voice = dataclasses.replace(voice, placement=settings().placement)

        built, _ = runtime(HandlerFactory(), voice_settings=voice)

        assert built.services.settings.playback.character == character

    def test_capabilities_offer_wake_and_the_loaded_spotters(self):
        built, _ = runtime(HandlerFactory())
        no_kws, _ = runtime(HandlerFactory(), voice_settings=settings(kws_model=None))
        neither, _ = runtime(HandlerFactory(), voice_settings=settings(verify_model=None))

        voice = built.voice_capabilities()
        assert (voice.modes, voice.spotters, voice.reference) == (
            ["wake"],
            ["endpoint", "server"],
            True,
        )
        verifier_only = no_kws.voice_capabilities()
        assert (verifier_only.modes, verifier_only.spotters, verifier_only.reference) == (
            ["wake"],
            ["endpoint"],
            False,
        )
        # No verifier: no wake can be accepted, so no wake mode or spotter is offered.
        bare = neither.voice_capabilities()
        assert (bare.modes, bare.spotters, bare.reference) == ([], [], False)
        assert (bare.stt, bare.tts) == (True, True)

    def test_close_stops_the_lanes_and_closes_the_verifier(self):
        built, _ = runtime(HandlerFactory())
        verifier = built.engines.verifier

        built.close()

        assert verifier.closes == 1

    def test_the_default_lanes_give_the_verifier_its_own_lane(self):
        built = build_mist_runtime(
            knowledge_config(),  # type: ignore[arg-type]
            voice_settings=settings(),
            llm_provider=FakeProvider(),  # type: ignore[arg-type]
            handler_factory=HandlerFactory(),
            engine_loaders=Loaders().build(),
        )
        try:
            lanes = built.lanes
            assert isinstance(lanes.verify, InferenceLane)
            assert lanes.verify not in (lanes.vad, lanes.eot, lanes.stt, lanes.tts, lanes.kws)
        finally:
            built.close()

    def test_the_real_verify_loader_is_the_onnx_verifier_on_onnxruntime(self, monkeypatch):
        from backend.realtime.engines import silero_vad
        from backend.realtime.engines.verifier import OnnxWakeVerifier

        seen: list[tuple[object, dict[str, object]]] = []

        def load(placement: object, **kwargs: object) -> str:
            seen.append((placement, kwargs))
            return "loaded"

        monkeypatch.setattr(OnnxWakeVerifier, "load", load)
        placement = settings().placement

        assert _real_engine_loaders().verify(placement) == "loaded"
        assert seen == [
            (
                placement,
                {
                    "session_factory": silero_vad.open_onnx_session,
                    "available_providers": silero_vad.onnxruntime_providers,
                },
            )
        ]

    def test_close_with_no_verifier_stops_the_lanes_alone(self):
        built, _ = runtime(HandlerFactory(), voice_settings=settings(verify_model=None))

        built.close()

        assert built.engines.verifier is None
