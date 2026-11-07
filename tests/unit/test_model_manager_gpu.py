"""Tests for releasing and reloading the Whisper model around gaming mode.

`ModelManager` is built for real, with TTS off and knowledge integration
switched off at import, then given a recording embedding generator. Whisper,
`gc` and `torch` are monkeypatched on the module, so no model is loaded and no
GPU is touched. Every collaborator appends to one shared call log, so the tests
can assert order as well as presence.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from backend.config import VoiceConfig
from backend.errors import GpuYieldedError
from backend.voice_models import model_manager as mm_module
from backend.voice_models.model_manager import ModelManager


class FakeWhisper:
    """Stands in for `WhisperSTT`; records construction and transcription."""

    calls: list[str]
    fail_with: BaseException | None = None
    transcribe_started: threading.Event | None = None
    transcribe_release: threading.Event | None = None

    def __init__(self, model_size: str = "base") -> None:
        FakeWhisper.calls.append(f"whisper:{model_size}")
        if FakeWhisper.fail_with is not None:
            raise FakeWhisper.fail_with
        self.model_size = model_size

    def transcribe_audio(self, audio_data, sample_rate: int = 16000) -> str:
        if FakeWhisper.transcribe_started is not None:
            FakeWhisper.transcribe_started.set()
        if FakeWhisper.transcribe_release is not None:
            assert FakeWhisper.transcribe_release.wait(timeout=5.0)
        FakeWhisper.calls.append("whisper:transcribe")
        return "transcribed"


class RecordingProvider:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def invoke_sync(self, request) -> None:
        self._calls.append("llm:invoke_sync")


class RecordingEmbeddingGenerator:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def warmup(self) -> None:
        self._calls.append("embedding:warmup")


class FakeKnowledge:
    """Only the chain `load_all_models` walks to reach the embedding generator."""

    def __init__(self, embedding_generator) -> None:
        self.conversation_handler = SimpleNamespace(
            graph_store=SimpleNamespace(embedding_generator=embedding_generator)
        )

    def is_enabled(self) -> bool:
        return True


@pytest.fixture
def calls() -> list[str]:
    return []


@pytest.fixture
def cuda(monkeypatch, calls):
    """Replace `gc` and `torch` on the module with recorders; CUDA off by default."""
    state = SimpleNamespace(available=False)

    def is_available() -> bool:
        return state.available

    def empty_cache() -> None:
        calls.append("torch:empty_cache")

    monkeypatch.setattr(
        mm_module,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=is_available, empty_cache=empty_cache)),
    )
    monkeypatch.setattr(mm_module, "gc", SimpleNamespace(collect=lambda: calls.append("gc")))
    return state


@pytest.fixture
def manager(monkeypatch, calls, cuda) -> ModelManager:
    monkeypatch.setattr(mm_module, "KNOWLEDGE_AVAILABLE", False)
    FakeWhisper.calls = calls
    FakeWhisper.fail_with = None
    FakeWhisper.transcribe_started = None
    FakeWhisper.transcribe_release = None
    monkeypatch.setattr(mm_module, "WhisperSTT", FakeWhisper)
    config = VoiceConfig(whisper_model="tiny", tts_enabled=False)
    manager = ModelManager(config, llm_provider=RecordingProvider(calls))
    manager.knowledge = FakeKnowledge(RecordingEmbeddingGenerator(calls))
    return manager


def _loaded(manager: ModelManager, calls: list[str]) -> ModelManager:
    manager.load_all_models()
    calls.clear()
    return manager


class TestLoadAllModels:
    def test_default_load_keeps_todays_calls_and_order(self, manager, calls):
        manager.load_all_models()

        assert calls == ["whisper:tiny", "llm:invoke_sync", "embedding:warmup"]
        assert isinstance(manager.stt, FakeWhisper)

    def test_yielded_load_skips_whisper_and_llm_but_warms_embeddings(self, manager, calls):
        manager.load_all_models(yielded=True)

        assert calls == ["embedding:warmup"]
        assert manager.stt is None


class TestReleaseGpuModels:
    def test_release_drops_whisper_and_collects_without_cuda(self, manager, calls):
        _loaded(manager, calls)

        manager.release_gpu_models()

        assert manager.stt is None
        assert calls == ["gc"]

    def test_release_empties_the_cuda_cache_when_cuda_is_available(self, manager, calls, cuda):
        _loaded(manager, calls)
        cuda.available = True

        manager.release_gpu_models()

        assert calls == ["gc", "torch:empty_cache"]

    def test_second_release_is_a_no_op(self, manager, calls, cuda):
        _loaded(manager, calls)
        cuda.available = True
        manager.release_gpu_models()
        calls.clear()

        manager.release_gpu_models()

        assert calls == []
        assert manager.stt is None

    def test_release_touches_no_tts_state(self, manager, calls):
        _loaded(manager, calls)
        tts_sentinel = object()
        manager.tts = tts_sentinel
        before = (
            manager.tts_worker_thread,
            manager.tts_worker_running.is_set(),
            manager.tts_request_queue.qsize(),
            manager.tts_generation_id,
        )

        manager.release_gpu_models()

        assert manager.tts is tts_sentinel
        assert before == (
            manager.tts_worker_thread,
            manager.tts_worker_running.is_set(),
            manager.tts_request_queue.qsize(),
            manager.tts_generation_id,
        )

    def test_release_waits_for_an_in_flight_transcription(self, manager, calls):
        _loaded(manager, calls)
        FakeWhisper.transcribe_started = threading.Event()
        FakeWhisper.transcribe_release = threading.Event()
        transcription = threading.Thread(target=manager.transcribe_audio, args=([0.0],))
        transcription.start()
        assert FakeWhisper.transcribe_started.wait(timeout=5.0)

        released = threading.Event()

        def release() -> None:
            manager.release_gpu_models()
            released.set()

        releaser = threading.Thread(target=release)
        releaser.start()
        returned_early = released.wait(timeout=0.2)
        FakeWhisper.transcribe_release.set()
        transcription.join(timeout=5.0)
        releaser.join(timeout=5.0)

        assert returned_early is False
        assert released.is_set()
        assert calls == ["whisper:transcribe", "gc"]
        assert manager.stt is None


class TestReacquireGpuModels:
    def test_reacquire_loads_whisper_once_with_the_configured_model(self, manager, calls):
        _loaded(manager, calls)
        manager.release_gpu_models()
        calls.clear()

        manager.reacquire_gpu_models()

        assert calls == ["whisper:tiny"]
        assert isinstance(manager.stt, FakeWhisper)

    def test_failed_reacquire_leaves_stt_none_empties_cache_and_raises(self, manager, calls, cuda):
        _loaded(manager, calls)
        manager.release_gpu_models()
        calls.clear()
        cuda.available = True
        FakeWhisper.fail_with = RuntimeError("CUDA out of memory")

        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            manager.reacquire_gpu_models()

        assert manager.stt is None
        assert calls == ["whisper:tiny", "gc", "torch:empty_cache"]


class TestVoiceProcessorInitialize:
    """`VoiceProcessor.initialize` forwards `yielded` to `load_all_models`.

    The model manager, VAD loader and LLM provider factory are replaced, so the
    test exercises only the forwarding.
    """

    @pytest.fixture
    def processor(self, monkeypatch):
        from backend import factories
        from backend import voice_processor as vp_module

        loads: list[dict] = []

        class RecordingModelManager:
            def __init__(self, config, **kwargs) -> None:
                self.config = config

            def load_all_models(self, **kwargs) -> None:
                loads.append(kwargs)

        monkeypatch.setattr(vp_module, "ModelManager", RecordingModelManager)
        monkeypatch.setattr(factories, "build_llm_provider", lambda *a, **k: None)
        monkeypatch.setattr(
            vp_module,
            "torch",
            SimpleNamespace(hub=SimpleNamespace(load=lambda *a, **k: (None, None))),
        )
        monkeypatch.setattr(vp_module, "AudioStreamProcessor", lambda **kwargs: object())
        processor = vp_module.VoiceProcessor(VoiceConfig(tts_enabled=False), message_queue=None)
        return processor, loads

    @pytest.mark.asyncio
    async def test_initialize_passes_yielded_through(self, processor):
        voice_processor, loads = processor

        await voice_processor.initialize(yielded=True)

        assert loads == [{"yielded": True}]

    @pytest.mark.asyncio
    async def test_initialize_defaults_to_not_yielded(self, processor):
        voice_processor, loads = processor

        await voice_processor.initialize()

        assert loads == [{"yielded": False}]


class TestTranscribeWhileReleased:
    def test_transcribe_with_no_whisper_raises_gpu_yielded(self, manager):
        manager.load_all_models(yielded=True)

        with pytest.raises(GpuYieldedError):
            manager.transcribe_audio([0.0])

    def test_transcribe_after_release_raises_gpu_yielded(self, manager, calls):
        _loaded(manager, calls)
        manager.release_gpu_models()

        with pytest.raises(GpuYieldedError):
            manager.transcribe_audio([0.0])
