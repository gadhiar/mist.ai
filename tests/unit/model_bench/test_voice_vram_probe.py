"""The voice VRAM probe against fake speech engines and a fake `torch`.

`voice_vram.main` runs inside the mist-backend container, where it loads the realtime engines
and reads torch's peak CUDA memory. Here `build_speech_engines` is replaced by fake engines and
`torch` by a stand-in whose CUDA memory calls return fixed byte counts, so no GPU, model or
download is touched. What stays real is `load_voice_settings`, so the tests show the probe builds
its engines from the settings the live path reads (e553d886), and the one JSON line it prints.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from backend import factories  # noqa: E402
from backend.realtime import budget  # noqa: E402
from backend.realtime.budget import EngineOption, VoiceSettings  # noqa: E402
from backend.realtime.speech import CancelToken, TextSegment  # noqa: E402
from scripts.model_bench.probes import voice_vram  # noqa: E402

MIB = 1024 * 1024
RESERVED_BYTES = 3 * MIB + MIB // 2
ALLOCATED_BYTES = 2 * MIB
RESULT_KEYS = {
    "max_memory_reserved_mib",
    "max_memory_allocated_mib",
    "stt_ok",
    "tts_ok",
    "error",
}


class FakeCuda:
    """The four `torch.cuda` calls the probe makes, with fixed peaks."""

    def __init__(self, events: list[str], *, available: bool = True) -> None:
        self._events = events
        self._available = available

    def is_available(self) -> bool:
        return self._available

    def reset_peak_memory_stats(self) -> None:
        self._events.append("reset_peak")

    def max_memory_reserved(self) -> int:
        self._events.append("read_reserved")
        return RESERVED_BYTES

    def max_memory_allocated(self) -> int:
        self._events.append("read_allocated")
        return ALLOCATED_BYTES


class FakeStream:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def push(self, audio: object) -> None:
        self._events.append(f"stt_push:{len(audio)}")  # type: ignore[arg-type]

    def decode_final(self, token: CancelToken) -> str:
        assert isinstance(token, CancelToken)
        self._events.append("stt_decode_final")
        return ""

    def close(self) -> None:
        self._events.append("stt_close")


class FakeTranscriber:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def open_stream(self) -> FakeStream:
        self._events.append("stt_open")
        return FakeStream(self._events)


class FakeSynthesizer:
    def __init__(self, events: list[str]) -> None:
        self._events = events
        self.segments: list[TextSegment] = []

    def synthesize(self, segment: TextSegment, token: CancelToken):
        assert isinstance(token, CancelToken)
        self.segments.append(segment)
        self._events.append("tts_synthesize")
        yield b"\x00\x00"


class Built:
    """What the fake `build_speech_engines` was called with."""

    def __init__(self) -> None:
        self.settings: VoiceSettings | None = None
        self.gpu_meter: object = None
        self.environ: object = None


@pytest.fixture
def probe_env(monkeypatch):
    """A clean voice environment, the repo as the project root, and the real sys.path restored."""
    for key in list(os.environ):
        if key.startswith("MIST_VOICE_") or key == "VOICE_PROFILE":
            monkeypatch.delenv(key)
    monkeypatch.setenv("MIST_PROJECT_ROOT", str(_REPO_ROOT))
    monkeypatch.setenv("MIST_VOICE_STT_CHAIN", "parakeet-cpu@cpu")
    monkeypatch.setenv("VOICE_PROFILE", "vram-probe")
    monkeypatch.setattr(sys, "path", list(sys.path))


def install(
    monkeypatch,
    events: list[str],
    *,
    transcriber: object | None,
    synthesizer: object | None,
    cuda_available: bool = True,
) -> Built:
    """Fake `torch` and `build_speech_engines`; spy on the real `load_voice_settings`."""
    built = Built()
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=FakeCuda(events, available=cuda_available))
    )
    real_load = budget.load_voice_settings

    def spy_load(environ):
        built.environ = environ
        events.append("load_voice_settings")
        return real_load(environ)

    def fake_build(settings, *, gpu_meter=None, **kwargs):
        assert kwargs == {}, kwargs
        built.settings = settings
        built.gpu_meter = gpu_meter
        events.append("build_speech_engines")
        statuses = SimpleNamespace(stt="stt: off", tts="tts: off")
        return SimpleNamespace(transcriber=transcriber, synthesizer=synthesizer, statuses=statuses)

    monkeypatch.setattr(budget, "load_voice_settings", spy_load)
    monkeypatch.setattr(factories, "build_speech_engines", fake_build)
    return built


def run_probe(capsys) -> dict[str, object]:
    voice_vram.main()
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1, lines
    return json.loads(lines[0])


def test_the_probe_runs_both_engines_and_prints_the_whole_run_peak(probe_env, monkeypatch, capsys):
    events: list[str] = []
    synthesizer = FakeSynthesizer(events)
    install(monkeypatch, events, transcriber=FakeTranscriber(events), synthesizer=synthesizer)

    result = run_probe(capsys)

    assert set(result) == RESULT_KEYS
    assert result == {
        "max_memory_reserved_mib": 3.5,
        "max_memory_allocated_mib": 2.0,
        "stt_ok": True,
        "tts_ok": True,
        "error": None,
    }
    assert events == [
        "reset_peak",
        "load_voice_settings",
        "build_speech_engines",
        "stt_open",
        f"stt_push:{16000 * 2}",
        "stt_decode_final",
        "stt_close",
        "tts_synthesize",
        "read_reserved",
        "read_allocated",
    ]
    (segment,) = synthesizer.segments
    assert segment.text == voice_vram.PROBE_TEXT


def test_the_engines_are_built_from_load_voice_settings_of_the_environment(
    probe_env, monkeypatch, capsys
):
    events: list[str] = []
    built = install(
        monkeypatch,
        events,
        transcriber=FakeTranscriber(events),
        synthesizer=FakeSynthesizer(events),
    )

    run_probe(capsys)

    assert built.environ is os.environ
    assert isinstance(built.settings, VoiceSettings)
    assert built.settings.placement.stt_chain == (EngineOption("parakeet-cpu", "cpu"),)
    assert built.settings.profile == "vram-probe"
    # The meter never resets torch's peak statistics, so one reading covers every load.
    assert isinstance(built.gpu_meter, voice_vram._WholeRunMeter)
    assert built.gpu_meter.measure(lambda: "engine") == ("engine", 0)
    assert events.count("reset_peak") == 1


def test_a_missing_engine_is_reported_in_the_json_line(probe_env, monkeypatch, capsys):
    events: list[str] = []
    install(monkeypatch, events, transcriber=None, synthesizer=None, cuda_available=False)

    result = run_probe(capsys)

    assert set(result) == RESULT_KEYS
    assert (result["stt_ok"], result["tts_ok"]) == (False, False)
    assert (result["max_memory_reserved_mib"], result["max_memory_allocated_mib"]) == (
        None,
        None,
    )
    error = str(result["error"])
    assert "stt: RuntimeError('no STT engine loaded: stt: off')" in error
    assert "tts: RuntimeError('no TTS engine loaded: tts: off')" in error
    assert "reset_peak" not in events
