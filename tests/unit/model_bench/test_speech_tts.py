"""Run 1, speech synthesis: `tts_suite`, `probes/tts_probe.py` and `probes/tts_engines.py`.

Every engine is a fake at the `SpeechEngine` protocol, every clock is scripted, and every engine
package is a fake module handed to the adapters' importer, so no test needs a GPU, the network or
an engine package. The orchestration tests run the real probe in-process behind a fake container
runner, so the host's argv, the probe's outputs and the host's merge are checked together.
"""

from __future__ import annotations

import importlib
import json
import sys
import types
import wave
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from scripts.model_bench import speech_bench  # noqa: E402
from scripts.model_bench.speech import core, tts_suite  # noqa: E402
from scripts.model_bench.speech.probes import load_gen, tts_engines, tts_probe  # noqa: E402

ARMS = core.load_json(core.SPEECH_ARMS_PATH)
TTS = ARMS["tts"]
LINES_DOC = core.load_json(core.LINES_PATH)
LINES = {line["id"]: line for line in LINES_DOC["lines"]}
ACKS = {ack["id"]: ack for ack in LINES_DOC["acknowledgements"]}
EMOTIVE = [line["id"] for line in LINES_DOC["lines"] if line["emotive"]]
L_IDS = list(LINES)
GPU_ENGINES = ["qwen3", "turbo", "original"]


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class StepClock:
    def __init__(self, start: float = 0.0, step: float = 0.01):
        self.now = start
        self.step = step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


class ScriptedClock:
    def __init__(self, ticks):
        self.ticks = list(ticks)

    def __call__(self) -> float:
        return self.ticks.pop(0)


class FakeDevice:
    def __init__(self, peak: float | None = None):
        self.events: list[str] = []
        self.peak = peak

    def sync(self) -> None:
        self.events.append("sync")

    def reset_peak(self) -> None:
        self.events.append("reset")

    def peak_reserved_mib(self) -> float | None:
        return self.peak


class FakeEngine:
    def __init__(
        self,
        name: str = "fake",
        *,
        streaming: bool = True,
        voices=("v1", "v2"),
        sample_rate: int = 10,
        chunk_lengths=(5, 15),
        device: str = "cuda",
    ):
        self.name = name
        self.device = device
        self.streaming = streaming
        self.sample_rate = sample_rate
        self._voices = list(voices)
        self.chunk_lengths = tuple(chunk_lengths)
        self.voice: str | None = None
        self.setting = "default"
        self.loaded: list[str] = []
        self.calls: list[tuple[str, str | None, str]] = []

    def voices(self) -> list[str]:
        return list(self._voices)

    def load(self, voice: str) -> None:
        if voice not in self._voices:
            raise tts_engines.VoiceError(f"{self.name} has no stock voice {voice!r}")
        self.loaded.append(voice)
        self.voice = voice

    def select_setting(self, name: str) -> None:
        self.setting = name

    def stream(self, text: str):
        self.calls.append((text, self.voice, self.setting))
        for n in self.chunk_lengths:
            yield np.full(n, 0.25, dtype=np.float32)


def fake_factory(engines: dict):
    """An engine factory that builds a FakeEngine per engine id from speech_arms.json."""

    def build(engine_id: str, models_dir: Path):
        cfg = TTS[engine_id]
        engine = FakeEngine(
            engine_id,
            streaming=cfg["streaming"],
            voices=cfg["voices"],
            device=cfg["device"],
        )
        engines.setdefault(engine_id, []).append(engine)
        return engine

    return build


def run_probe(argv, *, engines=None, clock=None, emit=None):
    engines = {} if engines is None else engines
    lines: list[str] = []
    code = tts_probe.main(
        argv,
        engine_factory=fake_factory(engines),
        clock=clock or StepClock(),
        wall=StepClock(start=1_793_980_800.0, step=0.5),
        device_factory=lambda engine: FakeDevice(peak=512.0),
        emit=emit or lines.append,
    )
    return code, engines, lines


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# ---------------------------------------------------------------------------
# The first segment: first clause boundary ending at or after character 40
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param(
            "Your timer is set for ten minutes, and I'll let you know when it's done.",
            "Your timer is set for ten minutes, and I'll let you know when it's done.",
            id="comma-before-40-and-no-later-boundary-gives-the-whole-line",
        ),
        pytest.param(
            "a" * 39 + ", then the rest of the line.",
            "a" * 39 + ",",
            id="boundary-ending-exactly-at-40",
        ),
        pytest.param(
            "a" * 38 + ", then the rest of the line.",
            "a" * 38 + ", then the rest of the line.",
            id="boundary-ending-at-39-is-too-early",
        ),
        pytest.param(
            "a" * 38 + ", then more; and the rest",
            "a" * 38 + ", then more;",
            id="first-boundary-at-or-after-40-wins",
        ),
        pytest.param(
            "a" * 45 + ".",
            "a" * 45 + ".",
            id="final-stop-without-whitespace-is-no-boundary",
        ),
        pytest.param("a" * 41 + "! b", "a" * 41 + "!", id="exclamation"),
        pytest.param("a" * 41 + "? b", "a" * 41 + "?", id="question"),
        pytest.param("a" * 41 + ": b", "a" * 41 + ":", id="colon"),
        pytest.param("a" * 41 + "-- b", "a" * 41 + "-- b", id="dash-is-no-boundary"),
        pytest.param(
            LINES["L16"]["text"],
            "I've moved your dentist appointment to Thursday at half past three,",
            id="L16",
        ),
        pytest.param(
            LINES["L08"]["text"],
            "Careful there. Nobody actually has that number;",
            id="L08-semicolon",
        ),
    ],
)
def test_first_segment_cuts_at_the_first_clause_boundary_at_or_after_40(text, expected):
    assert tts_probe.first_segment(text) == expected


# ---------------------------------------------------------------------------
# First audio, first segment, total and compute per second from a scripted clock
# ---------------------------------------------------------------------------


def test_a_streaming_engine_times_first_audio_from_submit_to_the_first_chunk():
    engine = FakeEngine(streaming=True, sample_rate=10, chunk_lengths=(5, 15))
    device = FakeDevice(peak=700.0)

    timing = tts_probe.time_item(
        engine,
        "Short line.",
        clock=ScriptedClock([10.0, 10.25, 10.75, 11.0]),
        wall=ScriptedClock([1000.0, 1001.0]),
        device=device,
    )

    assert timing.first_audio_ms == pytest.approx(250.0)
    assert timing.first_segment_ms is None
    assert timing.total_ms == pytest.approx(1000.0)
    assert timing.audio_s == pytest.approx(2.0)
    assert timing.compute_per_speech_s == pytest.approx(0.5)
    assert timing.torch_peak_reserved_mib == 700.0
    assert (timing.t_start_unix, timing.t_end_unix) == (1000.0, 1001.0)


def test_a_streaming_engine_synchronizes_before_every_timestamp():
    engine = FakeEngine(streaming=True, chunk_lengths=(5, 15))
    device = FakeDevice()

    tts_probe.time_item(
        engine,
        "Short line.",
        clock=ScriptedClock([1.0, 2.0, 3.0, 4.0]),
        wall=ScriptedClock([0.0, 0.0]),
        device=device,
    )

    assert device.events == ["reset", "sync", "sync", "sync", "sync"]


def test_a_streaming_engine_skips_an_empty_first_chunk_for_first_audio():
    engine = FakeEngine(streaming=True, sample_rate=10, chunk_lengths=(0, 10))

    timing = tts_probe.time_item(
        engine,
        "Short line.",
        clock=ScriptedClock([0.0, 0.1, 0.4, 0.5]),
        wall=ScriptedClock([0.0, 0.0]),
        device=FakeDevice(),
    )

    assert timing.first_audio_ms == pytest.approx(400.0)


def test_a_whole_segment_engine_times_the_first_segment_then_the_whole_line():
    engine = FakeEngine(streaming=False, sample_rate=10, chunk_lengths=(30,))
    text = LINES["L16"]["text"]

    timing = tts_probe.time_item(
        engine,
        text,
        clock=ScriptedClock([5.0, 5.3, 6.5]),
        wall=ScriptedClock([2000.0, 2001.5]),
        device=FakeDevice(),
    )

    assert [call[0] for call in engine.calls] == [tts_probe.first_segment(text), text]
    assert timing.first_segment_ms == pytest.approx(300.0)
    assert timing.first_audio_ms == pytest.approx(300.0)
    assert timing.first_segment_chars == len(tts_probe.first_segment(text))
    assert timing.total_ms == pytest.approx(1200.0)
    assert timing.audio_s == pytest.approx(3.0)
    assert timing.compute_per_speech_s == pytest.approx(0.4)


def test_a_whole_segment_engine_with_no_cut_synthesizes_once():
    engine = FakeEngine(streaming=False, sample_rate=10, chunk_lengths=(20,))
    text = LINES["L01"]["text"]

    timing = tts_probe.time_item(
        engine,
        text,
        clock=ScriptedClock([1.0, 1.8]),
        wall=ScriptedClock([0.0, 0.0]),
        device=FakeDevice(),
    )

    assert [call[0] for call in engine.calls] == [text]
    assert timing.first_segment_ms == pytest.approx(800.0)
    assert timing.total_ms == pytest.approx(800.0)
    assert timing.compute_per_speech_s == pytest.approx(0.4)


def test_an_engine_that_yields_no_audio_is_an_error():
    engine = FakeEngine(streaming=True, chunk_lengths=(0,))

    with pytest.raises(core.SpeechBenchError, match="no audio"):
        tts_probe.time_item(
            engine,
            "Short line.",
            clock=StepClock(),
            wall=StepClock(),
            device=FakeDevice(),
        )


# ---------------------------------------------------------------------------
# Order and text
# ---------------------------------------------------------------------------


def test_the_timed_order_is_every_item_times_every_repeat_shuffled_by_the_seed():
    items = [*L_IDS, *ACKS]

    order = tts_probe.timed_order(items, repeats=3, seed=7)

    assert sorted(order) == sorted((i, rep) for rep in (1, 2, 3) for i in items)
    assert order != [(i, rep) for rep in (1, 2, 3) for i in items]


def test_the_timed_order_is_the_same_for_the_same_seed_and_differs_for_another():
    items = [*L_IDS, *ACKS]

    first = tts_probe.timed_order(items, repeats=3, seed=20261106)

    assert first == tts_probe.timed_order(items, repeats=3, seed=20261106)
    assert first != tts_probe.timed_order(items, repeats=3, seed=20261107)


@pytest.mark.parametrize(
    ("engine", "line_id", "field"),
    [
        pytest.param("turbo", "L03", "turbo_text", id="turbo-gets-its-tags"),
        pytest.param("turbo", "L07", "turbo_text", id="turbo-gets-chuckle"),
        pytest.param("turbo", "L01", "text", id="turbo-untagged-line-falls-back-to-text"),
        pytest.param("original", "L03", "text", id="original-gets-text"),
        pytest.param("qwen3", "L03", "text", id="qwen3-gets-text"),
        pytest.param("pocket", "L07", "text", id="pocket-gets-text"),
    ],
)
def test_turbo_receives_turbo_text_and_the_others_text(engine, line_id, field):
    assert tts_probe.text_for(LINES[line_id], TTS[engine]) == LINES[line_id][field]


# ---------------------------------------------------------------------------
# The probe's timed mode
# ---------------------------------------------------------------------------


def timed_args(engine: str, out: Path, *extra: str) -> list[str]:
    return [
        "--engine",
        engine,
        "--mode",
        "timed",
        "--condition",
        "idle",
        "--seed",
        "11",
        "--repeats",
        "3",
        "--models-dir",
        "/models/speech",
        "--out",
        str(out),
        *extra,
    ]


def test_the_timed_mode_loads_warms_up_three_times_then_times_every_item(tmp_path):
    events: list[str] = []
    engines: dict = {}

    def emit(line: str) -> None:
        events.append(f"{line}@{len(engines['qwen3'][0].calls)}")

    code, engines, _ = run_probe(timed_args("qwen3", tmp_path), engines=engines, emit=emit)

    engine = engines["qwen3"][0]
    rows = read_jsonl(tmp_path / "probe-idle.jsonl")
    samples = [r for r in rows if r["row_type"] == "sample"]
    assert code == 0
    assert engine.loaded == ["aiden"]
    assert [e for e in events if e.startswith("TIMED")] == ["TIMED_START@3", "TIMED_END@81"]
    assert len(samples) == (20 + 6) * 3
    assert [(r["line_id"], r["rep"]) for r in samples] == tts_probe.timed_order(
        [*L_IDS, *ACKS], repeats=3, seed=11
    )


def test_the_timed_mode_flags_acknowledgements_and_records_the_row_fields(tmp_path):
    code, _, _ = run_probe(timed_args("qwen3", tmp_path))

    rows = read_jsonl(tmp_path / "probe-idle.jsonl")
    load = [r for r in rows if r["row_type"] == "load"]
    samples = [r for r in rows if r["row_type"] == "sample"]
    kinds = {r["line_id"]: r["kind"] for r in samples}
    first = samples[0]
    assert code == 0
    assert {k for k, v in kinds.items() if v == "ack"} == set(ACKS)
    assert {k for k, v in kinds.items() if v == "line"} == set(L_IDS)
    assert len(load) == 1 and load[0]["load_s"] > 0 and load[0]["streaming"] is True
    assert {
        "engine",
        "voice",
        "setting",
        "condition",
        "line_id",
        "rep",
        "words",
        "chars",
        "first_audio_ms",
        "first_segment_ms",
        "total_ms",
        "audio_s",
        "compute_per_speech_s",
        "torch_peak_reserved_mib",
        "t_start_unix",
        "t_end_unix",
        "wav",
    } <= set(first)
    assert first["engine"] == "qwen3" and first["condition"] == "idle"
    assert first["voice"] == "aiden" and first["setting"] == "default"
    assert first["torch_peak_reserved_mib"] == 512.0
    item = LINES.get(first["line_id"]) or ACKS[first["line_id"]]
    assert first["words"] == len(item["text"].split())
    assert first["chars"] == len(item["text"])


def test_the_timed_mode_saves_every_clip_as_a_wav_at_its_results_path(tmp_path):
    code, _, _ = run_probe(timed_args("qwen3", tmp_path))

    samples = [r for r in read_jsonl(tmp_path / "probe-idle.jsonl") if r["row_type"] == "sample"]
    first = samples[0]
    name = f"idle_{first['line_id']}_r{first['rep']}.wav"
    assert code == 0
    assert first["wav"] == f"tts/qwen3/audio/{name}"
    assert len(list((tmp_path / "audio").glob("*.wav"))) == len(samples)
    with wave.open(str(tmp_path / "audio" / name), "rb") as clip:
        assert (clip.getnchannels(), clip.getsampwidth(), clip.getframerate()) == (1, 2, 10)
        assert clip.getnframes() == 20


def test_the_timed_mode_sends_turbo_its_tagged_text(tmp_path):
    code, engines, _ = run_probe(timed_args("turbo", tmp_path, "--lines", "L03", "--repeats", "1"))

    texts = [call[0] for call in engines["turbo"][0].calls]
    assert code == 0
    assert LINES["L03"]["turbo_text"] in texts
    assert LINES["L03"]["text"] not in texts


def test_the_timed_mode_restricts_to_the_given_lines(tmp_path):
    code, _, _ = run_probe(
        timed_args("pocket", tmp_path, "--lines", "L02", "--lines", "A01", "--repeats", "2")
    )

    samples = [r for r in read_jsonl(tmp_path / "probe-idle.jsonl") if r["row_type"] == "sample"]
    assert code == 0
    assert sorted((r["line_id"], r["rep"]) for r in samples) == [
        ("A01", 1),
        ("A01", 2),
        ("L02", 1),
        ("L02", 2),
    ]


def test_the_timed_mode_refuses_an_unknown_line_before_any_write(tmp_path):
    code, engines, _ = run_probe(timed_args("qwen3", tmp_path, "--lines", "L99"))

    assert code == 2
    assert engines == {}
    assert list(tmp_path.iterdir()) == []


def test_the_timed_mode_records_streaming_false_for_a_whole_segment_engine(tmp_path):
    code, _, _ = run_probe(timed_args("original", tmp_path, "--lines", "L16", "--repeats", "1"))

    rows = read_jsonl(tmp_path / "probe-idle.jsonl")
    load = next(r for r in rows if r["row_type"] == "load")
    sample = next(r for r in rows if r["row_type"] == "sample")
    assert code == 0
    assert load["streaming"] is False
    assert sample["first_segment_ms"] is not None


# ---------------------------------------------------------------------------
# The probe's render mode: listening voices, extra voices and `expressive`
# ---------------------------------------------------------------------------


def render_args(engine: str, out: Path, *extra: str) -> list[str]:
    return ["--engine", engine, "--mode", "render", "--out", str(out), *extra]


def test_the_render_mode_writes_repeat_1_of_every_line_per_listening_voice(tmp_path):
    code, engines, _ = run_probe(
        render_args("qwen3", tmp_path, "--voice", "ryan", "--voice", "vivian")
    )

    rows = read_jsonl(tmp_path / "listening.jsonl")
    assert code == 0
    for voice in ("ryan", "vivian"):
        names = sorted(p.name for p in (tmp_path / "listening" / voice).glob("*.wav"))
        assert names == [f"{i}.wav" for i in L_IDS]
    assert {(r["voice"], r["setting"], r["rep"]) for r in rows} == {
        ("ryan", "default", 1),
        ("vivian", "default", 1),
    }
    assert rows[0]["wav"] == "tts/qwen3/listening/ryan/L01.wav"
    assert engines["qwen3"][0].loaded == ["ryan", "vivian"]


def test_the_render_mode_writes_original_s_emotive_lines_in_its_expressive_setting(tmp_path):
    code, engines, _ = run_probe(render_args("original", tmp_path, "--expressive"))

    engine = engines["original"][0]
    names = sorted(p.name for p in (tmp_path / "listening" / "expressive").glob("*.wav"))
    rows = read_jsonl(tmp_path / "listening.jsonl")
    assert code == 0
    assert names == [f"{i}.wav" for i in EMOTIVE]
    assert {call[2] for call in engine.calls} == {"expressive"}
    assert {call[1] for call in engine.calls} == {"builtin"}
    assert {(r["voice"], r["setting"]) for r in rows} == {("builtin", "expressive")}
    assert rows[0]["wav"] == f"tts/original/listening/expressive/{EMOTIVE[0]}.wav"


def test_the_render_mode_refuses_expressive_for_an_engine_without_it(tmp_path):
    code, engines, _ = run_probe(render_args("turbo", tmp_path, "--expressive"))

    assert code == 2
    assert engines == {}
    assert list(tmp_path.iterdir()) == []


def test_the_render_mode_refuses_a_voice_the_engine_does_not_list_before_any_write(tmp_path):
    code, engines, _ = run_probe(render_args("qwen3", tmp_path, "--voice", "ryan", "--voice", "x"))

    assert code == 2
    assert engines == {}
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# The probe's loop mode (run 5's memory fit)
# ---------------------------------------------------------------------------


def test_the_loop_mode_prints_ready_then_synthesizes_until_the_seconds_pass_then_done():
    events: list[str] = []
    engines: dict = {}

    def emit(line: str) -> None:
        events.append(f"{line}@{len(engines['pocket'][0].calls)}")

    code, engines, _ = run_probe(
        ["--engine", "pocket", "--mode", "loop", "--line-id", "L05", "--seconds", "1"],
        engines=engines,
        clock=StepClock(step=0.1),
        emit=emit,
    )

    engine = engines["pocket"][0]
    marks = [e for e in events if e.split("@")[0] in ("READY", "DONE")]
    assert code == 0
    assert engine.loaded == ["alba"]
    assert marks[0] == "READY@3"
    assert marks[-1].startswith("DONE@")
    assert len(engine.calls) > 3
    assert {call[0] for call in engine.calls} == {LINES["L05"]["text"]}


# ---------------------------------------------------------------------------
# Adapters: stock voices only, the documented calls, lazy imports
# ---------------------------------------------------------------------------


class Recorder:
    def __init__(self):
        self.calls: list[tuple[str, tuple, dict]] = []

    def record(self, name, *args, **kwargs):
        self.calls.append((name, args, kwargs))


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return self

    def numpy(self):
        return self.values


def fake_importer(modules: dict, imported: list):
    def importer(name: str):
        imported.append(name)
        if name in modules:
            return modules[name]
        if name == "numpy":
            return importlib.import_module("numpy")
        raise ModuleNotFoundError(f"fake importer has no {name}", name=name)

    return importer


def fake_torch(rec: Recorder):
    return types.SimpleNamespace(
        bfloat16="bf16",
        set_num_threads=lambda n: rec.record("set_num_threads", n),
    )


def qwen3_modules(rec: Recorder):
    class Model:
        def warmup(self):
            rec.record("warmup")

        def generate_custom_voice_streaming(self, text, **kwargs):
            rec.record("generate", text, **kwargs)
            yield np.zeros(4, dtype=np.float32), 24000, {}
            yield np.zeros(6, dtype=np.float32), 24000, {}

    class FasterQwen3TTS:
        @staticmethod
        def from_pretrained(path, **kwargs):
            rec.record("from_pretrained", path, **kwargs)
            return Model()

    return {
        "faster_qwen3_tts": types.SimpleNamespace(FasterQwen3TTS=FasterQwen3TTS),
        "torch": fake_torch(rec),
    }


def chatterbox_modules(rec: Recorder):
    def model_class(label):
        class Model:
            sr = 24000

            def generate(self, *args, **kwargs):
                rec.record(f"{label}.generate", *args, **kwargs)
                return FakeTensor([[0.1, 0.2, 0.3]])

        class Loader:
            @staticmethod
            def from_local(ckpt_dir, device):
                rec.record(f"{label}.from_local", ckpt_dir, device)
                return Model()

        return Loader

    return {
        "chatterbox.tts": types.SimpleNamespace(ChatterboxTTS=model_class("original")),
        "chatterbox.tts_turbo": types.SimpleNamespace(ChatterboxTurboTTS=model_class("turbo")),
    }


def pocket_modules(rec: Recorder, *, streaming: bool):
    class Model:
        sample_rate = 24000

        def get_state_for_audio_prompt(self, name):
            rec.record("state", name)
            return f"state:{name}"

        def generate_audio(self, state, text):
            rec.record("generate_audio", state, text)
            return FakeTensor([0.1, 0.2])

    if streaming:

        def generate_audio_stream(self, state, text):
            rec.record("generate_audio_stream", state, text)
            yield FakeTensor([0.1])
            yield FakeTensor([0.2, 0.3])

        Model.generate_audio_stream = generate_audio_stream

    class TTSModel:
        @staticmethod
        def load_model():
            rec.record("load_model")
            return Model()

    return {"pocket_tts": types.SimpleNamespace(TTSModel=TTSModel), "torch": fake_torch(rec)}


@pytest.mark.parametrize("engine", list(TTS))
def test_each_adapter_lists_exactly_its_stock_voices(engine):
    adapter = tts_engines.make_engine(engine, importer=fake_importer({}, []))

    assert adapter.voices() == TTS[engine]["voices"]
    assert adapter.name == engine
    assert adapter.device == TTS[engine]["device"]


@pytest.mark.parametrize("engine", list(TTS))
@pytest.mark.parametrize("voice", ["nobody", "/models/speech/a.wav", "Builtin"])
def test_each_adapter_refuses_a_voice_it_does_not_list_before_importing_anything(engine, voice):
    imported: list = []
    adapter = tts_engines.make_engine(engine, importer=fake_importer({}, imported))

    with pytest.raises(tts_engines.VoiceError):
        adapter.load(voice)

    assert imported == []


def test_make_engine_refuses_an_unknown_engine():
    with pytest.raises(core.SpeechBenchError, match="unknown"):
        tts_engines.make_engine("nosuch")


def test_the_qwen3_adapter_loads_offline_and_streams_a_preset_speaker_at_chunk_size_8():
    rec = Recorder()
    adapter = tts_engines.make_engine(
        "qwen3", models_dir=Path("/models/speech"), importer=fake_importer(qwen3_modules(rec), [])
    )

    adapter.load("aiden")
    chunks = list(adapter.stream("Hello there."))

    name, args, kwargs = rec.calls[0]
    assert name == "from_pretrained"
    assert Path(args[0]) == Path("/models/speech/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    assert kwargs == {"device": "cuda", "dtype": "bf16"}
    assert rec.calls[1][0] == "warmup"
    gen = rec.calls[2]
    assert gen[1] == ("Hello there.",)
    assert gen[2] == {"speaker": "aiden", "language": "English", "chunk_size": 8}
    assert [len(c) for c in chunks] == [4, 6]
    assert adapter.sample_rate == 24000 and adapter.streaming is True


def test_the_qwen3_adapter_switches_speaker_without_reloading():
    rec = Recorder()
    adapter = tts_engines.make_engine("qwen3", importer=fake_importer(qwen3_modules(rec), []))

    adapter.load("aiden")
    adapter.load("ryan")
    list(adapter.stream("Hi."))

    assert [c[0] for c in rec.calls].count("from_pretrained") == 1
    assert rec.calls[-1][2]["speaker"] == "ryan"


@pytest.mark.parametrize(
    ("setting", "expected"),
    [
        pytest.param(
            "default",
            {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8},
            id="default",
        ),
        pytest.param(
            "expressive",
            {"exaggeration": 0.7, "cfg_weight": 0.3, "temperature": 0.8},
            id="expressive",
        ),
    ],
)
def test_the_original_adapter_uses_its_builtin_voice_at_the_named_setting(setting, expected):
    rec = Recorder()
    adapter = tts_engines.make_engine(
        "original",
        models_dir=Path("/models/speech"),
        importer=fake_importer(chatterbox_modules(rec), []),
    )

    adapter.load("builtin")
    adapter.select_setting(setting)
    chunks = list(adapter.stream("Hello there."))

    assert rec.calls[0][0] == "original.from_local"
    assert Path(rec.calls[0][1][0]) == Path("/models/speech/ResembleAI/chatterbox")
    assert rec.calls[0][1][1] == "cuda"
    assert rec.calls[1] == ("original.generate", ("Hello there.",), expected)
    assert len(chunks) == 1 and chunks[0].tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert adapter.sample_rate == 24000 and adapter.streaming is False


def test_the_turbo_adapter_passes_only_the_text():
    rec = Recorder()
    adapter = tts_engines.make_engine(
        "turbo",
        models_dir=Path("/models/speech"),
        importer=fake_importer(chatterbox_modules(rec), []),
    )

    adapter.load("builtin")
    list(adapter.stream("[sigh] Long day."))

    assert rec.calls[0][0] == "turbo.from_local"
    assert Path(rec.calls[0][1][0]) == Path("/models/speech/ResembleAI/chatterbox-turbo")
    assert rec.calls[1] == ("turbo.generate", ("[sigh] Long day.",), {})


def test_the_original_adapter_refuses_a_setting_it_does_not_have():
    adapter = tts_engines.make_engine("original", importer=fake_importer({}, []))

    with pytest.raises(tts_engines.SettingError):
        adapter.select_setting("whisper")


def test_the_pocket_adapter_streams_when_the_package_has_a_streaming_call():
    rec = Recorder()
    adapter = tts_engines.make_engine(
        "pocket", importer=fake_importer(pocket_modules(rec, streaming=True), [])
    )

    adapter.load("alba")
    chunks = list(adapter.stream("Hello."))

    assert ("set_num_threads", (4,), {}) in rec.calls
    assert ("state", ("alba",), {}) in rec.calls
    assert ("generate_audio_stream", ("state:alba", "Hello."), {}) in rec.calls
    assert [len(c) for c in chunks] == [1, 2]
    assert adapter.streaming is True and adapter.sample_rate == 24000


def test_the_pocket_adapter_yields_once_without_a_streaming_call():
    rec = Recorder()
    adapter = tts_engines.make_engine(
        "pocket", importer=fake_importer(pocket_modules(rec, streaming=False), [])
    )

    adapter.load("fantine")
    chunks = list(adapter.stream("Hello."))

    assert ("generate_audio", ("state:fantine", "Hello."), {}) in rec.calls
    assert len(chunks) == 1
    assert adapter.streaming is False


def test_no_adapter_passes_an_audio_argument_to_its_engine():
    rec = Recorder()
    modules = chatterbox_modules(rec) | qwen3_modules(rec) | pocket_modules(rec, streaming=False)
    for engine in TTS:
        adapter = tts_engines.make_engine(engine, importer=fake_importer(modules, []))
        adapter.load(TTS[engine]["timing_voice"])
        list(adapter.stream("Hello."))

    passed = [key for _, _, kwargs in rec.calls for key in kwargs]
    assert not [k for k in passed if "audio" in k or "prompt" in k or "wav" in k]


# ---------------------------------------------------------------------------
# Host argv
# ---------------------------------------------------------------------------

OUT = "/results/mh1/speech/tts/x"


def timed(engine: str, condition: str = "idle") -> list[str]:
    return tts_suite.timed_argv(
        engine, condition, out_dir=OUT, repo_dir="/repo", speech_models_dir="/models"
    )


def mounts(argv: list[str]) -> list[str]:
    return [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]


def envs(argv: list[str]) -> list[str]:
    return [argv[i + 1] for i, a in enumerate(argv) if a == "-e"]


def value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


@pytest.mark.parametrize("engine", GPU_ENGINES)
def test_a_gpu_engine_gets_all_gpus_and_the_bench_server_s_network(engine):
    argv = timed(engine)

    assert argv[:3] == ["docker", "run", "--rm"]
    assert value(argv, "--gpus") == "all"
    assert value(argv, "--network") == "container:mist-bench-llm"
    assert "--cpus" not in argv


def test_pocket_gets_no_gpu_no_network_and_at_most_4_threads():
    argv = timed("pocket")

    assert "--gpus" not in argv
    assert value(argv, "--network") == "none"
    assert float(value(argv, "--cpus")) <= 4
    assert {"OMP_NUM_THREADS=4", "MKL_NUM_THREADS=4"} <= set(envs(argv))


@pytest.mark.parametrize("engine", list(TTS))
def test_every_engine_container_is_offline_read_only_and_mounts_the_repository(engine):
    argv = timed(engine)

    assert "HF_HUB_OFFLINE=1" in envs(argv)
    assert "/repo:/work:ro" in mounts(argv)
    assert "/models:/models/speech:ro" in mounts(argv)
    assert [m for m in mounts(argv) if not m.endswith(":ro")] == [f"{OUT}:/out"]
    assert value(argv, "-w") == "/work"
    assert "-p" not in argv and "--publish" not in argv
    assert argv[argv.index(TTS[engine]["image"]) + 1 :][:3] == [
        "python",
        "-m",
        "scripts.model_bench.speech.probes.tts_probe",
    ]


def test_the_timed_argv_passes_the_condition_seed_repeats_and_lines():
    argv = tts_suite.timed_argv(
        "qwen3",
        "decoding",
        out_dir=OUT,
        repo_dir="/repo",
        speech_models_dir="/models",
        seed=5,
        repeats=1,
        lines=("L01", "A02"),
    )

    tail = argv[argv.index("scripts.model_bench.speech.probes.tts_probe") + 1 :]
    assert tail == [
        "--engine",
        "qwen3",
        "--models-dir",
        "/models/speech",
        "--mode",
        "timed",
        "--condition",
        "decoding",
        "--seed",
        "5",
        "--repeats",
        "1",
        "--out",
        "/out",
        "--lines",
        "L01",
        "--lines",
        "A02",
    ]


def test_the_render_argv_names_each_voice_and_the_expressive_setting():
    argv = tts_suite.render_argv(
        "original",
        ["builtin"],
        expressive=True,
        out_dir=OUT,
        repo_dir="/repo",
        speech_models_dir="/models",
    )

    tail = argv[argv.index("scripts.model_bench.speech.probes.tts_probe") + 1 :]
    assert value(tail, "--mode") == "render"
    assert value(tail, "--voice") == "builtin"
    assert "--expressive" in tail


@pytest.mark.parametrize("engine", list(TTS))
def test_the_loop_argv_runs_the_probe_s_loop_mode_with_no_output_mount(engine):
    argv = tts_suite.loop_container_argv(
        engine, line_id="L05", seconds=90, repo_dir="/repo", speech_models_dir="/models"
    )

    tail = argv[argv.index("scripts.model_bench.speech.probes.tts_probe") + 1 :]
    assert value(tail, "--mode") == "loop"
    assert value(tail, "--line-id") == "L05"
    assert float(value(tail, "--seconds")) == 90
    assert all(m.endswith(":ro") for m in mounts(argv))
    assert "HF_HUB_OFFLINE=1" in envs(argv)
    assert ("--gpus" in argv) == (TTS[engine]["device"] == "cuda")
    assert value(argv, "--name") == f"mist-bench-tts-loop-{engine}"


def test_the_loop_argv_refuses_an_unknown_line_or_engine():
    with pytest.raises(core.SpeechBenchError):
        tts_suite.loop_container_argv(
            "qwen3", line_id="L99", seconds=10, repo_dir="/r", speech_models_dir="/m"
        )
    with pytest.raises(core.SpeechBenchError):
        tts_suite.loop_container_argv(
            "nosuch", line_id="L01", seconds=10, repo_dir="/r", speech_models_dir="/m"
        )


# ---------------------------------------------------------------------------
# Listening voices and `--extra-voice` on the host
# ---------------------------------------------------------------------------


def test_parse_extra_voices_groups_stock_voices_by_engine():
    got = tts_suite.parse_extra_voices(["qwen3:dylan", "pocket:marius", "qwen3:eric"], TTS)

    assert got == {"qwen3": ("dylan", "eric"), "pocket": ("marius",)}


@pytest.mark.parametrize(
    "spec",
    [
        pytest.param("qwen3:nobody", id="voice-not-listed"),
        pytest.param("nosuch:alba", id="unknown-engine"),
        pytest.param("qwen3", id="no-colon"),
        pytest.param("pocket:/tmp/a.wav", id="a-path"),
    ],
)
def test_parse_extra_voices_refuses_a_name_the_engine_does_not_list(spec):
    with pytest.raises(tts_engines.VoiceError):
        tts_suite.parse_extra_voices([spec], TTS)


def test_render_plan_is_the_listening_voices_then_extra_voices_and_expressive_for_original():
    assert tts_suite.render_plan("qwen3", TTS["qwen3"], ("dylan", "ryan")) == (
        ("ryan", "vivian", "serena", "ono_anna", "sohee", "dylan"),
        False,
    )
    assert tts_suite.render_plan("original", TTS["original"], ()) == ((), True)
    assert tts_suite.render_plan("turbo", TTS["turbo"], ()) == ((), False)


def test_the_tts_subcommand_refuses_an_unlisted_extra_voice_before_any_write(tmp_path, capsys):
    models = tmp_path / "models"
    models.mkdir()
    results = tmp_path / "results"

    code = speech_bench.main(
        [
            "tts",
            "--results-root",
            str(results),
            "--run",
            "mh1",
            "--speech-models-dir",
            str(models),
            "--extra-voice",
            "qwen3:nobody",
        ]
    )

    assert code != 0
    assert "nobody" in capsys.readouterr().out
    assert not results.exists()


# ---------------------------------------------------------------------------
# Orchestration: run order, conditions, the load generator and the merge
# ---------------------------------------------------------------------------


class Now:
    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class FakeRunner:
    """Runs the real probe in-process for each argv, mapping `/out` to the host directory."""

    def __init__(self, log: list, now: Now, *, fail_on: str | None = None):
        self.log = log
        self.now = now
        self.fail_on = fail_on
        self.argvs: list[list[str]] = []
        self.engines: dict = {}

    def run(self, argv, *, log_path, on_line):
        self.argvs.append(list(argv))
        tail = argv[argv.index("scripts.model_bench.speech.probes.tts_probe") + 1 :]
        out = next(m[: -len(":/out")] for m in mounts(argv) if m.endswith(":/out"))
        tail = [out if (i > 0 and tail[i - 1] == "--out") else a for i, a in enumerate(tail)]
        label = f"{value(tail, '--engine')}:{value(tail, '--mode')}"
        if "--condition" in tail:
            label += f":{value(tail, '--condition')}"
        self.log.append(f"container {label}")
        if self.fail_on == label:
            return 1

        def emit(line: str) -> None:
            if line == tts_probe.TIMED_START:
                self.now.t = 100.0
            elif line == tts_probe.TIMED_END:
                self.now.t = 200.0
            on_line(line)

        Path(log_path).write_text("log\n", encoding="utf-8")
        code, _, _ = run_probe(tail, engines=self.engines, emit=emit)
        return code


class FakeLoadGen:
    def __init__(self, log: list):
        self.log = log

    def run_for(self, seconds):
        self.log.append(f"baseline {seconds:g}")
        return [completion(0, 5, n=300, ms=4000.0), completion(5, 10, n=300, ms=4000.0)]

    def start(self):
        self.log.append("loadgen start")

    def stop(self):
        self.log.append("loadgen stop")
        return [
            completion(90, 110, n=100, ms=1000.0),
            completion(110, 150, n=100, ms=1000.0),
            completion(150, 190, n=100, ms=2000.0),
            completion(190, 210, n=100, ms=1000.0),
        ]


def completion(t_request, t_end, *, n, ms):
    result = core.ChatStreamResult(
        content="x",
        reasoning_content="",
        tool_calls=(),
        t_request=float(t_request),
        t_first_content=float(t_request),
        t_first_tool_call=None,
        t_end=float(t_end),
        finish_reason="length",
        timings={"predicted_n": n, "predicted_ms": ms, "predicted_per_second": n / ms * 1000},
    )
    return load_gen.completion_from(result, seq=0, t_start_unix=0.0, t_end_unix=0.0)


class FakeSampler:
    def __init__(self, log: list, path: Path):
        self.log = log
        self.path = path

    def start(self):
        self.log.append(f"sampler start {self.path.parent.name}")

    def stop(self):
        self.log.append(f"sampler stop {self.path.parent.name}")


def request(tmp_path: Path, engines=("qwen3", "original"), **kwargs) -> tts_suite.TtsRequest:
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    defaults = {
        "results_root": tmp_path / "results",
        "run": "mh1",
        "engines": tuple(engines),
        "extra_voices": {},
        "seed": 3,
        "repeats": 1,
        "lines": ("L03", "L16", "A01"),
        "speech_models_dir": models,
        "baseline_s": 60.0,
    }
    defaults.update(kwargs)
    return tts_suite.TtsRequest(**defaults)


def run_suite(tmp_path, req=None, *, state="running", fail_on=None):
    log: list[str] = []
    now = Now()
    runner = FakeRunner(log, now, fail_on=fail_on)
    tts_suite.run_tts(
        req or request(tmp_path),
        runner=runner,
        container_state=lambda name: log.append(f"state {name}") or state,
        load_gen_factory=lambda: FakeLoadGen(log),
        sampler_factory=lambda path: FakeSampler(log, path),
        clock=now,
        git_sha="abc123",
        repo_root=_REPO_ROOT,
    )
    return log, runner


def test_run_tts_runs_each_engine_idle_then_baseline_then_decoding_then_renders(tmp_path):
    log, _ = run_suite(tmp_path)

    assert log == [
        "state mist-bench-llm",
        "sampler start qwen3",
        "container qwen3:timed:idle",
        "baseline 60",
        "loadgen start",
        "container qwen3:timed:decoding",
        "loadgen stop",
        "container qwen3:render",
        "sampler stop qwen3",
        "sampler start original",
        "container original:timed:idle",
        "baseline 60",
        "loadgen start",
        "container original:timed:decoding",
        "loadgen stop",
        "container original:render",
        "sampler stop original",
    ]


def test_run_tts_skips_the_render_container_when_an_engine_has_nothing_to_render(tmp_path):
    log, _ = run_suite(tmp_path, request(tmp_path, engines=("turbo",)))

    assert "container turbo:render" not in log
    assert log[-1] == "sampler stop turbo"


def test_run_tts_merges_samples_into_the_shared_file_with_the_run_stamp(tmp_path):
    run_suite(tmp_path)

    rows = read_jsonl(tmp_path / "results" / "mh1" / "speech" / "tts" / "samples.jsonl")
    assert len(rows) == 2 * 2 * 3
    assert {(r["run"], r["suite"], r["git_sha"]) for r in rows} == {("mh1", "tts", "abc123")}
    assert {(r["engine"], r["condition"]) for r in rows} == {
        ("qwen3", "idle"),
        ("qwen3", "decoding"),
        ("original", "idle"),
        ("original", "decoding"),
    }
    assert all("row_type" not in r for r in rows)
    for row in rows:
        assert (tmp_path / "results" / "mh1" / "speech" / row["wav"]).is_file()


def test_run_tts_records_load_and_decode_rates_with_and_without_speech(tmp_path):
    run_suite(tmp_path)

    rows = read_jsonl(tmp_path / "results" / "mh1" / "speech" / "tts" / "load.jsonl")
    qwen3 = next(r for r in rows if r["engine"] == "qwen3")
    assert [r["engine"] for r in rows] == ["qwen3", "original"]
    assert qwen3["load_s"] > 0 and qwen3["load_s_decoding"] > 0
    assert qwen3["streaming"] is True
    assert next(r for r in rows if r["engine"] == "original")["streaming"] is False
    assert qwen3["decode_tps_alone"] == pytest.approx(600 / 8.0)
    # Only the two completions wholly inside TIMED_START (100) to TIMED_END (200) count.
    assert qwen3["decode_tps_with_tts"] == pytest.approx(200 / 3.0)
    assert qwen3["decode_completions_with_tts"] == 2


def test_run_tts_renders_listening_voices_extra_voices_and_expressive(tmp_path):
    req = request(tmp_path, extra_voices={"qwen3": ("dylan",)})

    _, runner = run_suite(tmp_path, req)

    tts_dir = tmp_path / "results" / "mh1" / "speech" / "tts"
    qwen3_voices = sorted(p.name for p in (tts_dir / "qwen3" / "listening").iterdir())
    assert qwen3_voices == sorted([*TTS["qwen3"]["listening_voices"], "dylan"])
    assert sorted(p.name for p in (tts_dir / "qwen3" / "listening" / "dylan").iterdir()) == [
        "L03.wav",
        "L16.wav",
    ]
    assert [p.name for p in (tts_dir / "original" / "listening").iterdir()] == ["expressive"]
    assert [p.name for p in (tts_dir / "original" / "listening" / "expressive").iterdir()] == [
        "L03.wav"
    ]


def test_run_tts_refuses_unless_the_bench_server_is_running(tmp_path):
    with pytest.raises(core.SpeechBenchError, match="mist-bench-llm"):
        run_suite(tmp_path, state="exited")

    assert not (tmp_path / "results").exists()


def test_run_tts_refuses_a_results_root_inside_the_repository(tmp_path):
    req = request(tmp_path, results_root=_REPO_ROOT / "results-here")

    with pytest.raises(core.ResultsRootError):
        run_suite(tmp_path, req)

    assert not (_REPO_ROOT / "results-here").exists()


def test_run_tts_refuses_an_engine_whose_directory_exists_before_any_write(tmp_path):
    existing = tmp_path / "results" / "mh1" / "speech" / "tts" / "original"
    existing.mkdir(parents=True)

    with pytest.raises(core.OutputExistsError):
        run_suite(tmp_path)

    assert not (existing.parent / "qwen3").exists()


def test_run_tts_refuses_an_unlisted_extra_voice_before_checking_docker(tmp_path):
    req = request(tmp_path, extra_voices={"qwen3": ("nobody",)})
    log: list[str] = []

    with pytest.raises(tts_engines.VoiceError):
        tts_suite.run_tts(
            req,
            runner=FakeRunner(log, Now()),
            container_state=lambda name: log.append("state") or "running",
            load_gen_factory=lambda: FakeLoadGen(log),
            sampler_factory=lambda path: FakeSampler(log, path),
            git_sha="x",
        )

    assert log == []


def test_run_tts_raises_when_a_container_fails_and_still_stops_the_load_and_sampler(tmp_path):
    with pytest.raises(core.SpeechBenchError, match="exited 1"):
        run_suite(tmp_path, fail_on="qwen3:timed:decoding")


def test_the_mounted_results_path_is_the_claimed_directory_outside_the_repository(tmp_path):
    _, runner = run_suite(tmp_path)

    outs = {m for argv in runner.argvs for m in mounts(argv) if m.endswith(":/out")}
    for spec in outs:
        host = Path(spec[: -len(":/out")])
        assert host.parent == (tmp_path / "results" / "mh1" / "speech" / "tts").resolve()
        assert _REPO_ROOT.resolve() not in host.parents


# ---------------------------------------------------------------------------
# Registry: plan, selftest and inputs
# ---------------------------------------------------------------------------


def test_plan_lists_tts_suite_as_built_and_prints_its_argv(capsys):
    code = speech_bench.main(["plan"])

    out = capsys.readouterr().out
    assert code == 0
    assert "[skip] tts_suite" not in out
    for engine in TTS:
        assert f"[argv] tts {engine} idle:" in out
        assert f"[argv] tts {engine} decoding:" in out
        assert f"[argv] tts {engine} loop:" in out
    assert "[argv] tts qwen3 render:" in out
    assert "[argv] tts original render:" in out


def test_selftest_runs_the_tts_suite_selftest(capsys):
    code = speech_bench.main(["selftest"])

    assert code == 0
    assert "[ok] tts_suite" in capsys.readouterr().out


def test_tts_validate_inputs_finds_no_problem_in_the_committed_material():
    assert tts_suite.validate_inputs(_REPO_ROOT) == []
