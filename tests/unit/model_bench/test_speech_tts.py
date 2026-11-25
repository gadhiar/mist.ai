"""Run 1, speech synthesis: `tts_suite`, `probes/tts_probe.py` and `probes/tts_engines.py`.

Every engine is a fake at the `SpeechEngine` protocol, every clock is scripted, and every engine
package is a fake module handed to the adapters' importer, so no test needs a GPU, the network or
an engine package. The orchestration tests run the real probe in-process behind a fake container
runner, so the host's argv, the probe's outputs and the host's merge are checked together.
"""

from __future__ import annotations

import hashlib
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
import yaml  # noqa: E402

from scripts.model_bench import speech_bench  # noqa: E402
from scripts.model_bench.speech import core, tts_suite, voice_design  # noqa: E402
from scripts.model_bench.speech.probes import load_gen, tts_engines, tts_probe  # noqa: E402

ARMS = core.load_json(core.SPEECH_ARMS_PATH)
TTS = ARMS["tts"]
LINES_DOC = core.load_json(core.LINES_PATH)
LINES = {line["id"]: line for line in LINES_DOC["lines"]}
ACKS = {ack["id"]: ack for ack in LINES_DOC["acknowledgements"]}
EMOTIVE = [line["id"] for line in LINES_DOC["lines"] if line["emotive"]]
L_IDS = list(LINES)
GPU_ENGINES = ["qwen3", "turbo", "original", "qwen3-design"]
CLONE = "pocket-clone"
DESIGN = "qwen3-design"
# A distinctive string standing for the identity clip's transcript; it must never reach a result.
TRANSCRIPT = "zebra quartz lantern seventeen test line one"


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
        rss=lambda: 1536.0,
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
# The probe's resident memory peak: `VmHWM`, read after the load and after every sample
# ---------------------------------------------------------------------------

PROC_STATUS = """\
Name:\tpython
VmPeak:\t 4194304 kB
VmHWM:\t 1572864 kB
VmRSS:\t 1048576 kB
"""


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param(PROC_STATUS, 1536.0, id="vmhwm-in-kib"),
        pytest.param("Name:\tpython\nVmRSS:\t 1024 kB\n", None, id="no-vmhwm-line"),
        pytest.param("VmHWM:\t lots kB\n", None, id="not-a-number"),
        pytest.param("VmHWM:\t 1024 MB\n", None, id="another-unit"),
    ],
)
def test_rss_peak_mib_is_vmhwm_from_proc_status_or_null(tmp_path, text, expected):
    status = tmp_path / "status"
    status.write_text(text, encoding="utf-8")

    assert tts_probe.rss_peak_mib(status) == expected


def test_rss_peak_mib_is_null_where_proc_status_is_unreadable(tmp_path):
    assert tts_probe.rss_peak_mib(tmp_path / "absent") is None


def test_the_timed_mode_records_the_rss_peak_after_the_load_and_after_every_sample(tmp_path):
    engines: dict = {}
    readings = iter([700.0, *range(1000, 1100)])
    seen: list[str] = []

    def rss() -> float:
        engine = engines[CLONE][0]
        seen.append(f"{len(engine.loaded)}:{len(engine.calls)}")
        return float(next(readings))

    code = tts_probe.main(
        timed_args(CLONE, tmp_path, "--lines", "L02", "--lines", "L03", "--repeats", "1"),
        engine_factory=fake_factory(engines),
        clock=StepClock(),
        wall=StepClock(),
        device_factory=lambda engine: FakeDevice(),
        emit=lambda line: None,
        rss=rss,
    )

    rows = read_jsonl(tmp_path / "probe-idle.jsonl")
    assert code == 0
    assert [r["rss_peak_mib"] for r in rows] == [700.0, 1000.0, 1001.0]
    # After the load (no synthesis yet), then after the warm-ups and each timed item.
    assert seen == ["1:0", "1:4", "1:5"]


def test_the_timed_mode_records_a_null_rss_peak_where_it_is_unreadable(tmp_path):
    code = tts_probe.main(
        timed_args("pocket", tmp_path, "--lines", "L02", "--repeats", "1"),
        engine_factory=fake_factory({}),
        clock=StepClock(),
        wall=StepClock(),
        device_factory=lambda engine: FakeDevice(),
        emit=lambda line: None,
        rss=lambda: None,
    )

    rows = read_jsonl(tmp_path / "probe-idle.jsonl")
    assert code == 0
    assert [r["rss_peak_mib"] for r in rows] == [None, None]


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
        manual_seed=lambda seed: rec.record("manual_seed", seed),
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


POCKET_REPO = "kyutai/pocket-tts-without-voice-cloning"
POCKET_GATED_URI = (
    "hf://kyutai/pocket-tts/languages/english/model.safetensors"
    "@db04f0e653473e5c7333c69686ac92b7460b0c1d"
)
POCKET_MODEL_URI = (
    f"hf://{POCKET_REPO}/languages/english/model.safetensors"
    "@a7056f505a4a8ca72d8b5fb1440071d6f1a35739"
)
POCKET_TOKENIZER_URI = (
    f"hf://{POCKET_REPO}/languages/english/tokenizer.json"
    "@00eac05ed3d16bdc3f6b5d598874019c34a89214"
)

# The layout of Pocket TTS 4.0.1's installed `config/english.yaml`, shortened: its three `hf://`
# URIs (the gated `weights_path` included), a non-ASCII `replace_characters` key, and the two
# `weights_path` keys it leaves commented out.
INSTALLED_POCKET_YAML = f"""\
weights_path: {POCKET_GATED_URI}
weights_path_without_voice_cloning: {POCKET_MODEL_URI}
default_temperature: 0.3
replace_characters:
  "’": "'"
  "‘": "'"
flow_lm:
  insert_bos_before_voice: true
  dtype: float32
  flow:
    depth: 6
    dim: 512
  lookup_table:
    dim: 1024
    n_bins: 4000
    tokenizer: tokenizers
    tokenizer_path: {POCKET_TOKENIZER_URI}
  #weights_path: final.safetensors
mimi:
  dtype: float32
  sample_rate: 24000
  seanet:
    ratios:
    - 6
    - 5
    - 4
  #weights_path: codec.safetensors
"""


def pocket_setup(
    tmp_path: Path, *, installed: str = INSTALLED_POCKET_YAML, missing: tuple[str, ...] = ()
) -> tuple[Path, Path]:
    """A fake installed configs directory and a models directory with Pocket's local files."""
    configs_dir = tmp_path / "pocket_tts_config"
    configs_dir.mkdir()
    (configs_dir / "english.yaml").write_text(installed, encoding="utf-8")
    models_dir = tmp_path / "models"
    english = models_dir.joinpath(*POCKET_REPO.split("/"), "languages", "english")
    (english / "embeddings").mkdir(parents=True)
    files = ["model.safetensors", "tokenizer.json"]
    files += [f"embeddings/{voice}.safetensors" for voice in TTS["pocket"]["voices"]]
    for name in files:
        if name not in missing:
            (english / name).write_bytes(b"fake")
    return configs_dir, models_dir


def pocket_voice_state(models_dir: Path, voice: str) -> Path:
    return models_dir.joinpath(
        *POCKET_REPO.split("/"), "languages", "english", "embeddings", f"{voice}.safetensors"
    )


def pocket_modules(rec: Recorder, *, streaming: bool, configs_dir: Path):
    class Model:
        sample_rate = 24000
        has_voice_cloning = True

        def get_state_for_audio_prompt(self, source):
            rec.record("state", source)
            return f"state:{Path(source).stem}"

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
        # Each config `load_model` was given, as read from its file during the call.
        configs: list = []

        @staticmethod
        def load_model(**kwargs):
            rec.record("load_model", **kwargs)
            if kwargs.get("config") is None:
                raise AssertionError("load_model was called without a local config")
            path = Path(kwargs["config"])
            TTSModel.configs.append(yaml.safe_load(path.read_text(encoding="utf-8")))
            return Model()

    return {
        "pocket_tts": types.SimpleNamespace(TTSModel=TTSModel),
        "pocket_tts.utils.config": types.SimpleNamespace(CONFIGS_DIR=configs_dir),
        "torch": fake_torch(rec),
    }


def pocket_adapter(tmp_path: Path, rec: Recorder, *, streaming: bool = False, **setup):
    configs_dir, models_dir = pocket_setup(tmp_path, **setup)
    modules = pocket_modules(rec, streaming=streaming, configs_dir=configs_dir)
    adapter = tts_engines.make_engine(
        "pocket", models_dir=models_dir, importer=fake_importer(modules, [])
    )
    return adapter, modules, models_dir


def remote_strings(value) -> list[str]:
    if isinstance(value, dict):
        return [s for k, v in value.items() for s in remote_strings(k) + remote_strings(v)]
    if isinstance(value, list):
        return [s for item in value for s in remote_strings(item)]
    if isinstance(value, str) and value.startswith(("hf://", "http://", "https://")):
        return [value]
    return []


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


def test_the_pocket_adapter_streams_when_the_package_has_a_streaming_call(tmp_path):
    rec = Recorder()
    adapter, _, models_dir = pocket_adapter(tmp_path, rec, streaming=True)

    adapter.load("alba")
    chunks = list(adapter.stream("Hello."))

    assert ("set_num_threads", (4,), {}) in rec.calls
    assert ("state", (pocket_voice_state(models_dir, "alba"),), {}) in rec.calls
    assert ("generate_audio_stream", ("state:alba", "Hello."), {}) in rec.calls
    assert [len(c) for c in chunks] == [1, 2]
    assert adapter.streaming is True and adapter.sample_rate == 24000


def test_the_pocket_adapter_yields_once_without_a_streaming_call(tmp_path):
    rec = Recorder()
    adapter, _, _ = pocket_adapter(tmp_path, rec, streaming=False)

    adapter.load("fantine")
    chunks = list(adapter.stream("Hello."))

    assert ("generate_audio", ("state:fantine", "Hello."), {}) in rec.calls
    assert len(chunks) == 1
    assert adapter.streaming is False


def test_the_pocket_adapter_loads_a_copy_of_the_installed_config_naming_only_local_files(
    tmp_path,
):
    rec = Recorder()
    adapter, modules, models_dir = pocket_adapter(tmp_path, rec)
    installed = yaml.safe_load(INSTALLED_POCKET_YAML)
    english = models_dir.joinpath(*POCKET_REPO.split("/"), "languages", "english")

    adapter.load("alba")

    loads = [call for call in rec.calls if call[0] == "load_model"]
    assert len(loads) == 1 and list(loads[0][2]) == ["config"]
    config_path = Path(loads[0][2]["config"])
    assert config_path.name == "english.yaml"
    assert not config_path.exists()
    [written] = modules["pocket_tts"].TTSModel.configs
    # The three file keys are replaced and every other key is the installed one.
    expected = installed
    expected["weights_path"] = str(english / "model.safetensors")
    expected["weights_path_without_voice_cloning"] = None
    expected["flow_lm"]["lookup_table"]["tokenizer_path"] = str(english / "tokenizer.json")
    assert written == expected
    assert written["replace_characters"] == {"’": "'", "‘": "'"}
    assert "weights_path" not in written["flow_lm"] and "weights_path" not in written["mimi"]
    assert remote_strings(written) == []
    assert "kyutai/pocket-tts/" not in json.dumps(written, ensure_ascii=False)


def test_the_pocket_model_is_marked_as_having_no_voice_cloning_after_load(tmp_path):
    rec = Recorder()
    adapter, _, _ = pocket_adapter(tmp_path, rec)

    adapter.load("alba")

    assert adapter._model.has_voice_cloning is False


def _replace(old: str, new: str):
    return lambda text: text.replace(old, new)


@pytest.mark.parametrize(
    ("edit", "missing"),
    [
        pytest.param(
            _replace(
                f"weights_path_without_voice_cloning: {POCKET_MODEL_URI}",
                "weights_path_without_voice_cloning: "
                + POCKET_MODEL_URI.replace(POCKET_REPO, "kyutai/pocket-tts"),
            ),
            (),
            id="fallback-uri-in-another-repository",
        ),
        pytest.param(
            _replace(POCKET_MODEL_URI, POCKET_MODEL_URI[:-8] + "00000000"),
            (),
            id="fallback-uri-at-another-revision",
        ),
        pytest.param(
            _replace(POCKET_MODEL_URI, POCKET_MODEL_URI.replace("english", "french")),
            (),
            id="fallback-uri-at-another-path",
        ),
        pytest.param(
            _replace(POCKET_TOKENIZER_URI, POCKET_TOKENIZER_URI[:-8] + "00000000"),
            (),
            id="tokenizer-uri-at-another-revision",
        ),
        pytest.param(
            _replace("  #weights_path: codec.safetensors", "  weights_path: codec.safetensors"),
            (),
            id="mimi-weights-path-set",
        ),
        pytest.param(
            _replace("  #weights_path: final.safetensors", "  weights_path: final.safetensors"),
            (),
            id="flow-lm-weights-path-set",
        ),
        pytest.param(
            lambda text: text + "voice: https://example.org/voices/alba.safetensors\n",
            (),
            id="another-remote-file",
        ),
        pytest.param(lambda text: text, ("model.safetensors",), id="model-file-missing"),
        pytest.param(lambda text: text, ("tokenizer.json",), id="tokenizer-file-missing"),
    ],
)
def test_the_pocket_adapter_refuses_anything_but_the_local_4_0_1_files_before_loading(
    tmp_path, edit, missing
):
    rec = Recorder()
    adapter, _, _ = pocket_adapter(
        tmp_path, rec, installed=edit(INSTALLED_POCKET_YAML), missing=missing
    )

    with pytest.raises(core.SpeechBenchError):
        adapter.load("alba")

    assert "load_model" not in [call[0] for call in rec.calls]


@pytest.mark.parametrize("voice", TTS["pocket"]["voices"])
def test_the_pocket_adapter_passes_each_stock_voice_as_the_path_of_its_state_file(tmp_path, voice):
    rec = Recorder()
    adapter, _, models_dir = pocket_adapter(tmp_path, rec)

    adapter.load(voice)

    [state] = [args for name, args, _ in rec.calls if name == "state"]
    assert state == (pocket_voice_state(models_dir, voice),)
    assert isinstance(state[0], Path)


def test_the_pocket_adapter_refuses_a_voice_whose_state_file_is_missing(tmp_path):
    rec = Recorder()
    adapter, _, _ = pocket_adapter(tmp_path, rec, missing=("embeddings/marius.safetensors",))

    with pytest.raises(tts_engines.VoiceError, match="marius"):
        adapter.load("marius")

    assert "state" not in [call[0] for call in rec.calls]
    assert adapter.voice is None


def test_no_adapter_but_pocket_clone_passes_an_audio_argument_to_its_engine(tmp_path):
    rec = Recorder()
    configs_dir, models_dir = pocket_setup(tmp_path)
    modules = (
        chatterbox_modules(rec)
        | qwen3_modules(rec)
        | qwen3_design_modules(rec)
        | pocket_modules(rec, streaming=False, configs_dir=configs_dir)
    )
    for engine in [e for e in TTS if e != CLONE]:
        adapter = tts_engines.make_engine(
            engine, models_dir=models_dir, importer=fake_importer(modules, [])
        )
        adapter.load(TTS[engine]["timing_voice"])
        list(adapter.stream("Hello."))

    passed = [key for _, _, kwargs in rec.calls for key in kwargs]
    assert not [k for k in passed if "audio" in k or "prompt" in k or "wav" in k]
    assert [list(kw) for name, _, kw in rec.calls if name == "load_model"] == [["config"]]
    states = [args[0] for name, args, _ in rec.calls if name == "state"]
    assert all(Path(s).suffix == ".safetensors" for s in states)


# ---------------------------------------------------------------------------
# The two arms T41 adds: `pocket-clone` and `qwen3-design`
# ---------------------------------------------------------------------------

# The `pocket` and `qwen3` arms as they were before T41, which must not change.
BASE_POCKET_ARM = {
    "verified": False,
    "image": "mist-speech-bench-pocket",
    "device": "cpu",
    "threads": 4,
    "streaming": True,
    "packages": {"pocket-tts": "4.0.1", "torch": "2.6.0+cu124"},
    "model": {
        "repository": "kyutai/pocket-tts-without-voice-cloning",
        "gated_fallback": "kyutai/pocket-tts",
    },
    "licence": "code MIT, weights CC-BY-4.0",
    "settings": {"default": {}},
    "voices": ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"],
    "female_voices": ["alba", "fantine", "cosette", "eponine", "azelma"],
    "timing_voice": "alba",
    "listening_voices": ["fantine", "cosette", "eponine", "azelma"],
}
BASE_QWEN3_ARM = {
    "verified": False,
    "image": "mist-speech-bench-qwen3",
    "device": "cuda",
    "streaming": True,
    "packages": {
        "faster-qwen3-tts": "0.5.4",
        "transformers": "5.16.1",
        "qwen-tts-hf": "0.1.1.post1",
        "torch": "2.6.0+cu124",
    },
    "model": {
        "repository": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "revision": "85e237c1",
        "bytes": 2493919668,
    },
    "licence": "Apache-2.0",
    "settings": {
        "default": {"chunk_size": 8, "dtype": "bfloat16", "language": "English", "instruct": None}
    },
    "voices": [
        "vivian",
        "serena",
        "uncle_fu",
        "dylan",
        "eric",
        "ryan",
        "aiden",
        "ono_anna",
        "sohee",
    ],
    "female_voices": ["vivian", "serena", "ono_anna", "sohee"],
    "timing_voice": "aiden",
    "listening_voices": ["ryan", "vivian", "serena", "ono_anna", "sohee"],
}
CLONE_REVISION = "db04f0e653473e5c7333c69686ac92b7460b0c1d"


def test_the_arms_file_gains_pocket_clone_and_qwen3_design_after_the_four_base_arms():
    assert list(TTS) == ["qwen3", "turbo", "original", "pocket", CLONE, DESIGN]
    assert TTS["pocket"] == BASE_POCKET_ARM
    assert TTS["qwen3"] == BASE_QWEN3_ARM


def test_pocket_clone_is_the_cloning_pocket_model_on_4_cpu_threads_in_the_pocket_image():
    arm = TTS[CLONE]

    assert arm["model"] == {"repository": "kyutai/pocket-tts", "revision": CLONE_REVISION}
    assert (arm["device"], arm["threads"], arm["image"]) == ("cpu", 4, "mist-speech-bench-pocket")
    assert arm["verified"] is False
    assert arm["packages"] == TTS["pocket"]["packages"]
    assert arm["voices"] == [core.IDENTITY_VOICE] and arm["timing_voice"] == core.IDENTITY_VOICE
    assert arm["female_voices"] == [] and arm["listening_voices"] == []
    assert arm["settings"] == {"default": {}}
    # The arm's revision is the one the installed config's cloning-model URI names.
    assert tts_engines.POCKET_CLONE_MODEL_URI.endswith(f"@{CLONE_REVISION}")


def test_qwen3_design_is_the_voice_design_model_t40_pins_on_the_gpu_in_the_qwen3_image():
    arm = TTS[DESIGN]
    model = arm["model"]

    assert f"{model['repository']}@{model['revision']}" == voice_design.MODEL_PIN
    assert (arm["device"], arm["image"]) == ("cuda", "mist-speech-bench-qwen3")
    assert arm["verified"] is False
    assert arm["streaming"] is False
    assert arm["packages"] == {
        "qwen-tts-hf": "0.1.1.post1",
        "transformers": "5.16.1",
        "torch": "2.6.0+cu124",
    }
    assert arm["voices"] == ["designed"] and arm["timing_voice"] == "designed"
    default = arm["settings"]["default"]
    assert set(default) == {"instruct", "seed"}
    assert isinstance(default["instruct"], str) and default["instruct"].strip()
    assert isinstance(default["seed"], int)


def test_the_arms_comment_says_which_arm_clones_and_which_designs():
    comment = ARMS["_comment"]

    assert "nothing is cloned" not in comment
    assert "`pocket-clone` clones a private identity clip passed at run time" in comment
    assert "`qwen3-design` designs a voice from text" in comment
    assert "every other engine speaks a stock voice" in comment


def test_every_tts_arm_has_an_adapter_and_every_adapter_an_arm():
    assert list(tts_engines.ADAPTERS) == list(TTS)


CLONE_REPO = "kyutai/pocket-tts"


def clone_setup(tmp_path: Path, *, installed: str = INSTALLED_POCKET_YAML, missing=()):
    """Pocket's setup plus the cloning model's snapshot and a synthetic identity clip."""
    configs_dir, models_dir = pocket_setup(tmp_path, installed=installed, missing=missing)
    english = models_dir.joinpath(*CLONE_REPO.split("/"), "languages", "english")
    english.mkdir(parents=True)
    if "clone/model.safetensors" not in missing:
        (english / "model.safetensors").write_bytes(b"fake cloning model")
    clip = write_identity_clip(tmp_path / "identity")
    return configs_dir, models_dir, clip


def write_identity_clip(root: Path) -> Path:
    """A one-second sine tone and, beside it, a transcript file that nothing may read."""
    root.mkdir(parents=True, exist_ok=True)
    clip = root / "clip.wav"
    rate = 24000
    frames = (np.sin(2 * np.pi * 220.0 * np.arange(rate) / rate) * 8000).astype("<i2")
    with wave.open(str(clip), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames.tobytes())
    (root / "clip.txt").write_text(TRANSCRIPT, encoding="utf-8")
    return clip


def clone_adapter(tmp_path: Path, rec: Recorder, *, has_voice_cloning=True, **setup):
    configs_dir, models_dir, clip = clone_setup(tmp_path, **setup)
    modules = pocket_modules(rec, streaming=False, configs_dir=configs_dir)
    modules["pocket_tts"].TTSModel.configs = []
    loader = modules["pocket_tts"].TTSModel.load_model

    def load_model(**kwargs):
        model = loader(**kwargs)
        model.has_voice_cloning = has_voice_cloning
        return model

    modules["pocket_tts"].TTSModel.load_model = staticmethod(load_model)
    adapter = tts_engines.PocketCloneEngine(
        CLONE,
        TTS[CLONE],
        models_dir=models_dir,
        importer=fake_importer(modules, []),
        voice_clip=clip,
    )
    return adapter, modules, models_dir, clip


def test_make_engine_builds_the_clone_adapter_reading_the_fixed_container_clip():
    adapter = tts_engines.make_engine(CLONE, importer=fake_importer({}, []))

    assert isinstance(adapter, tts_engines.PocketCloneEngine)
    assert adapter.voice_clip == Path(tts_engines.CONTAINER_VOICE_CLIP)
    assert adapter.voices() == [core.IDENTITY_VOICE]


def test_the_clone_adapter_maps_the_three_installed_uris_to_their_local_snapshots(tmp_path):
    rec = Recorder()
    adapter, modules, models_dir, _ = clone_adapter(tmp_path, rec)
    installed = yaml.safe_load(INSTALLED_POCKET_YAML)
    ungated = models_dir.joinpath(*POCKET_REPO.split("/"), "languages", "english")
    cloning = models_dir.joinpath(*CLONE_REPO.split("/"), "languages", "english")

    adapter.load(core.IDENTITY_VOICE)

    [written] = modules["pocket_tts"].TTSModel.configs
    expected = installed
    expected["weights_path"] = str(cloning / "model.safetensors")
    expected["weights_path_without_voice_cloning"] = str(ungated / "model.safetensors")
    expected["flow_lm"]["lookup_table"]["tokenizer_path"] = str(ungated / "tokenizer.json")
    assert written == expected
    assert remote_strings(written) == []
    assert ("set_num_threads", (4,), {}) in rec.calls


def test_the_clone_adapter_leaves_voice_cloning_on(tmp_path):
    rec = Recorder()
    adapter, _, _, _ = clone_adapter(tmp_path, rec)

    adapter.load(core.IDENTITY_VOICE)

    assert adapter._model.has_voice_cloning is True


def test_the_clone_adapter_refuses_a_model_that_loaded_without_voice_cloning(tmp_path):
    rec = Recorder()
    adapter, _, _, _ = clone_adapter(tmp_path, rec, has_voice_cloning=False)

    with pytest.raises(core.SpeechBenchError, match="cloning"):
        adapter.load(core.IDENTITY_VOICE)

    assert "state" not in [call[0] for call in rec.calls]


@pytest.mark.parametrize(
    ("edit", "missing"),
    [
        pytest.param(
            _replace(POCKET_GATED_URI, POCKET_GATED_URI[:-8] + "00000000"),
            (),
            id="cloning-uri-at-another-revision",
        ),
        pytest.param(
            _replace(POCKET_GATED_URI, POCKET_GATED_URI.replace("english", "french")),
            (),
            id="cloning-uri-at-another-path",
        ),
        pytest.param(
            _replace(POCKET_MODEL_URI, POCKET_MODEL_URI[:-8] + "00000000"),
            (),
            id="ungated-uri-at-another-revision",
        ),
        pytest.param(
            _replace(POCKET_TOKENIZER_URI, POCKET_TOKENIZER_URI.replace(POCKET_REPO, CLONE_REPO)),
            (),
            id="tokenizer-uri-in-another-repository",
        ),
        pytest.param(
            _replace("  #weights_path: codec.safetensors", "  weights_path: codec.safetensors"),
            (),
            id="mimi-weights-path-set",
        ),
        pytest.param(
            lambda text: text + "voice: hf://kyutai/pocket-tts/embeddings/alba.safetensors@x\n",
            (),
            id="another-hf-uri",
        ),
        pytest.param(
            lambda text: text + "voice: https://example.org/voices/alba.safetensors\n",
            (),
            id="another-remote-file",
        ),
        pytest.param(lambda text: text, ("clone/model.safetensors",), id="cloning-model-missing"),
        pytest.param(lambda text: text, ("model.safetensors",), id="ungated-model-missing"),
        pytest.param(lambda text: text, ("tokenizer.json",), id="tokenizer-missing"),
    ],
)
def test_the_clone_adapter_refuses_any_other_uri_or_a_missing_snapshot_before_loading(
    tmp_path, edit, missing
):
    rec = Recorder()
    adapter, _, _, _ = clone_adapter(
        tmp_path, rec, installed=edit(INSTALLED_POCKET_YAML), missing=missing
    )

    with pytest.raises(core.SpeechBenchError):
        adapter.load(core.IDENTITY_VOICE)

    assert "load_model" not in [call[0] for call in rec.calls]


def test_the_clone_adapter_takes_its_voice_from_the_mounted_clip(tmp_path):
    rec = Recorder()
    adapter, _, _, clip = clone_adapter(tmp_path, rec)

    adapter.load(core.IDENTITY_VOICE)
    chunks = list(adapter.stream("Hello."))

    [state] = [args for name, args, _ in rec.calls if name == "state"]
    assert state == (clip,)
    assert ("generate_audio", ("state:clip", "Hello."), {}) in rec.calls
    assert len(chunks) == 1


def test_the_clone_adapter_refuses_when_no_clip_is_mounted(tmp_path):
    rec = Recorder()
    adapter, _, _, clip = clone_adapter(tmp_path, rec)
    clip.unlink()

    with pytest.raises(tts_engines.VoiceError):
        adapter.load(core.IDENTITY_VOICE)

    assert "state" not in [call[0] for call in rec.calls]
    assert adapter.voice is None


# ---------------------------------------------------------------------------
# `pocket-clone` with an identity directory (`--voice-dir`, plan v5 M17)
# ---------------------------------------------------------------------------

VOICE_ID = "mist-trained"
TRAINED_WEIGHTS = b"fake trained weights"
TRAINED_TOKENIZER = b'{"fake": "trained tokenizer"}'
# The runtime adapter's modules, which the clone adapter imports through its importer.
RUNTIME_MODULES = ("backend.realtime.voice_assets", "backend.realtime.engines.pocket_tts")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_identity(
    voice_root: Path,
    voice_id: str = VOICE_ID,
    *,
    trained: bool = True,
    tokenizer: bool = True,
    engine: str = CLONE,
    tokenizer_sha256: str | None = None,
) -> Path:
    """An identity directory as `student export` writes it: a neutral clip, maybe weights."""
    identity = voice_root / voice_id
    identity.mkdir(parents=True)
    rate = 24000
    frames = (np.sin(2 * np.pi * 220.0 * np.arange(4 * rate) / rate) * 8000).astype("<i2")
    with wave.open(str(identity / "neutral.wav"), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames.tobytes())
    (identity / "neutral.txt").write_text(TRANSCRIPT, encoding="utf-8")
    manifest: dict = {
        "schema": 1,
        "voice_id": voice_id,
        "status": "frozen",
        "clips": [
            {
                "state": "neutral",
                "audio": "neutral.wav",
                "transcript": "neutral.txt",
                "audio_sha256": _sha((identity / "neutral.wav").read_bytes()),
                "transcript_sha256": _sha(TRANSCRIPT.encode("utf-8")),
            }
        ],
    }
    if trained:
        (identity / "model.safetensors").write_bytes(TRAINED_WEIGHTS)
        model = {
            "engine": engine,
            "base": f"{CLONE_REPO}@{CLONE_REVISION}",
            "weights": "model.safetensors",
            "sha256": _sha(TRAINED_WEIGHTS),
            "tokenizer_sha256": tokenizer_sha256 or _sha(b"fake"),
        }
        if tokenizer:
            (identity / "tokenizer.json").write_bytes(TRAINED_TOKENIZER)
            model["tokenizer"] = "tokenizer.json"
            model["tokenizer_sha256"] = _sha(TRAINED_TOKENIZER)
        manifest.update(schema=2, model=model)
    (identity / "voice.json").write_text(json.dumps(manifest), encoding="utf-8")
    return identity


def identity_adapter(tmp_path: Path, rec: Recorder, **identity_kwargs):
    """The clone adapter on an identity, its models directory at `<root>/speech` as mounted."""
    configs_dir, models_dir, _ = clone_setup(tmp_path)
    speech_models = tmp_path / "models-root" / "speech"
    speech_models.parent.mkdir()
    models_dir.rename(speech_models)
    voice_root = tmp_path / "voice"
    identity = write_identity(voice_root, **identity_kwargs)
    modules = pocket_modules(rec, streaming=False, configs_dir=configs_dir)
    modules["pocket_tts"].TTSModel.configs = []
    modules.update({name: importlib.import_module(name) for name in RUNTIME_MODULES})
    adapter = tts_engines.PocketCloneEngine(
        CLONE,
        TTS[CLONE],
        models_dir=speech_models,
        importer=fake_importer(modules, []),
        voice_root=voice_root,
        voice_id=VOICE_ID,
    )
    return adapter, modules, speech_models, identity


def test_make_engine_builds_the_clone_adapter_on_the_identity_mounted_at_voice_slash_its_id():
    adapter = tts_engines.make_engine(CLONE, importer=fake_importer({}, []), voice_id=VOICE_ID)

    assert isinstance(adapter, tts_engines.PocketCloneEngine)
    assert adapter.voice_id == VOICE_ID
    assert adapter.voice_root == Path(tts_engines.CONTAINER_VOICE_ROOT)
    assert tts_engines.container_voice_dir(VOICE_ID) == f"/voice/{VOICE_ID}"
    assert tts_engines.CONTAINER_VOICE_CLIP == "/voice/identity.wav"


@pytest.mark.parametrize("engine", [e for e in TTS if e != CLONE])
def test_make_engine_refuses_an_identity_for_any_other_engine(engine):
    with pytest.raises(core.SpeechBenchError, match=CLONE):
        tts_engines.make_engine(engine, importer=fake_importer({}, []), voice_id=VOICE_ID)


def test_the_identity_adapter_loads_the_trained_weights_and_tokenizer(tmp_path):
    rec = Recorder()
    adapter, modules, speech_models, identity = identity_adapter(tmp_path, rec)
    ungated = speech_models.joinpath(*POCKET_REPO.split("/"), "languages", "english")

    adapter.load(core.IDENTITY_VOICE)

    [written] = modules["pocket_tts"].TTSModel.configs
    assert written["weights_path"] == str((identity / "model.safetensors").resolve())
    assert written["flow_lm"]["lookup_table"]["tokenizer_path"] == str(
        (identity / "tokenizer.json").resolve()
    )
    assert written["weights_path_without_voice_cloning"] == str(ungated / "model.safetensors")
    assert remote_strings(written) == []
    assert ("set_num_threads", (4,), {}) in rec.calls


def test_the_identity_adapter_without_a_tokenizer_keeps_the_stock_one(tmp_path):
    rec = Recorder()
    adapter, modules, speech_models, identity = identity_adapter(tmp_path, rec, tokenizer=False)
    ungated = speech_models.joinpath(*POCKET_REPO.split("/"), "languages", "english")

    adapter.load(core.IDENTITY_VOICE)

    [written] = modules["pocket_tts"].TTSModel.configs
    assert written["weights_path"] == str((identity / "model.safetensors").resolve())
    assert written["flow_lm"]["lookup_table"]["tokenizer_path"] == str(ungated / "tokenizer.json")


def test_a_stock_identity_loads_the_stock_cloning_weights(tmp_path):
    rec = Recorder()
    adapter, modules, speech_models, _ = identity_adapter(tmp_path, rec, trained=False)
    cloning = speech_models.joinpath(*CLONE_REPO.split("/"), "languages", "english")

    adapter.load(core.IDENTITY_VOICE)

    [written] = modules["pocket_tts"].TTSModel.configs
    assert written["weights_path"] == str(cloning / "model.safetensors")


def test_the_identity_adapter_takes_its_voice_from_the_identity_s_neutral_clip(tmp_path):
    rec = Recorder()
    adapter, _, _, identity = identity_adapter(tmp_path, rec)

    adapter.load(core.IDENTITY_VOICE)
    chunks = list(adapter.stream("Hello."))

    [state] = [args for name, args, _ in rec.calls if name == "state"]
    assert state == ((identity / "neutral.wav").resolve(),)
    assert ("generate_audio", ("state:neutral", "Hello."), {}) in rec.calls
    assert len(chunks) == 1


@pytest.mark.parametrize(
    "identity_kwargs",
    [
        pytest.param({"engine": "pocket"}, id="weights-for-another-engine"),
        pytest.param(
            {"tokenizer": False, "tokenizer_sha256": "0" * 64}, id="not-the-stock-tokenizer"
        ),
    ],
)
def test_the_identity_adapter_refuses_what_the_runtime_refuses_before_loading(
    tmp_path, identity_kwargs
):
    rec = Recorder()
    adapter, _, _, _ = identity_adapter(tmp_path, rec, **identity_kwargs)

    with pytest.raises(core.SpeechBenchError):
        adapter.load(core.IDENTITY_VOICE)

    assert "load_model" not in [call[0] for call in rec.calls]


def test_the_identity_adapter_refuses_a_broken_identity_naming_no_private_path(tmp_path):
    rec = Recorder()
    adapter, _, _, identity = identity_adapter(tmp_path, rec)
    (identity / "neutral.wav").write_bytes(b"not the clip the manifest hashed")

    with pytest.raises(tts_engines.VoiceError) as info:
        adapter.load(core.IDENTITY_VOICE)

    assert str(tmp_path) not in str(info.value) and TRANSCRIPT not in str(info.value)
    assert "load_model" not in [call[0] for call in rec.calls]


def test_the_identity_adapter_needs_the_models_directory_mounted_as_speech(tmp_path):
    rec = Recorder()
    adapter, modules, speech_models, _ = identity_adapter(tmp_path, rec)
    adapter._models_dir = speech_models.parent

    with pytest.raises(core.SpeechBenchError, match="speech"):
        adapter.load(core.IDENTITY_VOICE)


def test_the_probe_builds_the_clone_adapter_on_the_identity_it_names(tmp_path, monkeypatch):
    built = []

    def make_engine(engine_id, **kwargs):
        built.append((engine_id, kwargs))
        return fake_factory({})(engine_id, kwargs["models_dir"])

    monkeypatch.setattr(tts_engines, "make_engine", make_engine)

    code = tts_probe.main(
        timed_args(CLONE, tmp_path, "--lines", "L02", "--repeats", "1", "--voice-id", VOICE_ID),
        clock=StepClock(),
        wall=StepClock(),
        device_factory=lambda engine: FakeDevice(),
        emit=lambda line: None,
        rss=lambda: None,
    )

    assert code == 0
    assert built == [(CLONE, {"models_dir": Path("/models/speech"), "voice_id": VOICE_ID})]


@pytest.mark.parametrize("engine", [e for e in TTS if e != CLONE])
def test_the_probe_refuses_an_identity_for_any_other_engine_before_any_write(tmp_path, engine):
    code, engines, _ = run_probe(timed_args(engine, tmp_path, "--voice-id", VOICE_ID))

    assert code == 2
    assert engines == {}
    assert list(tmp_path.iterdir()) == []


def qwen3_design_modules(rec: Recorder):
    class Model:
        def generate_voice_design(self, text, instruct, language=None, **kwargs):
            rec.record("generate_voice_design", text=text, instruct=instruct, language=language)
            return [np.full(7, 0.1, dtype=np.float32)], 12000

    class Qwen3TTSModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            rec.record("design.from_pretrained", path, **kwargs)
            return Model()

    return {
        "qwen_tts": types.SimpleNamespace(Qwen3TTSModel=Qwen3TTSModel),
        "torch": fake_torch(rec),
    }


def test_the_design_adapter_loads_the_voice_design_model_as_the_design_probe_does():
    rec = Recorder()
    adapter = tts_engines.make_engine(
        DESIGN,
        models_dir=Path("/models/speech"),
        importer=fake_importer(qwen3_design_modules(rec), []),
    )

    adapter.load("designed")

    name, args, kwargs = rec.calls[0]
    assert name == "design.from_pretrained"
    assert Path(args[0]) == Path("/models/speech/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    assert kwargs == {"device_map": "cuda:0", "dtype": "bf16"}


def test_the_design_adapter_voices_each_text_with_the_arm_s_fixed_description_and_seed():
    rec = Recorder()
    default = TTS[DESIGN]["settings"]["default"]
    adapter = tts_engines.make_engine(DESIGN, importer=fake_importer(qwen3_design_modules(rec), []))

    adapter.load("designed")
    first = list(adapter.stream("Hello there."))
    list(adapter.stream("Goodbye."))

    calls = [c for c in rec.calls if c[0] in ("manual_seed", "generate_voice_design")]
    assert calls == [
        ("manual_seed", (default["seed"],), {}),
        (
            "generate_voice_design",
            (),
            {"text": "Hello there.", "instruct": default["instruct"], "language": "English"},
        ),
        ("manual_seed", (default["seed"],), {}),
        (
            "generate_voice_design",
            (),
            {"text": "Goodbye.", "instruct": default["instruct"], "language": "English"},
        ),
    ]
    assert len(first) == 1 and len(first[0]) == 7
    assert adapter.sample_rate == 12000 and adapter.streaming is False


# ---------------------------------------------------------------------------
# Host argv
# ---------------------------------------------------------------------------

OUT = "/results/mh1/speech/tts/x"


CLIP_HOST = "/private/identity/clip.wav"


def clip_for(engine: str) -> str | None:
    return CLIP_HOST if engine == CLONE else None


def timed(engine: str, condition: str = "idle") -> list[str]:
    return tts_suite.timed_argv(
        engine,
        condition,
        out_dir=OUT,
        repo_dir="/repo",
        speech_models_dir="/models",
        voice_clip=clip_for(engine),
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


@pytest.mark.parametrize("engine", ["pocket", CLONE])
def test_pocket_gets_no_gpu_no_network_and_at_most_4_threads(engine):
    argv = timed(engine)

    assert "--gpus" not in argv
    assert value(argv, "--network") == "none"
    assert float(value(argv, "--cpus")) <= 4
    assert {"OMP_NUM_THREADS=4", "MKL_NUM_THREADS=4"} <= set(envs(argv))


def test_pocket_clone_mounts_the_clip_read_only_at_the_fixed_container_path():
    argv = timed(CLONE)

    assert f"{CLIP_HOST}:{tts_engines.CONTAINER_VOICE_CLIP}:ro" in mounts(argv)
    tail = argv[argv.index("scripts.model_bench.speech.probes.tts_probe") + 1 :]
    assert CLIP_HOST not in " ".join(tail)


def test_pocket_clone_s_argv_refuses_to_be_built_without_a_clip():
    with pytest.raises(core.VoiceClipError):
        tts_suite.timed_argv(
            CLONE, "idle", out_dir=OUT, repo_dir="/repo", speech_models_dir="/models"
        )


@pytest.mark.parametrize("engine", [e for e in TTS if e != CLONE])
def test_no_other_engine_s_argv_takes_a_clip(engine):
    with pytest.raises(core.VoiceClipError):
        tts_suite.timed_argv(
            engine,
            "idle",
            out_dir=OUT,
            repo_dir="/repo",
            speech_models_dir="/models",
            voice_clip=CLIP_HOST,
        )


VOICE_DIR_HOST = "/private/voices/trained-run-2"


def timed_on_identity(engine: str = CLONE, **kwargs) -> list[str]:
    paths = {"voice_dir": VOICE_DIR_HOST, "voice_id": VOICE_ID, **kwargs}
    return tts_suite.timed_argv(
        engine, "idle", out_dir=OUT, repo_dir="/repo", speech_models_dir="/models", **paths
    )


def test_pocket_clone_mounts_an_identity_read_only_at_voice_slash_its_id():
    argv = timed_on_identity()

    assert f"{VOICE_DIR_HOST}:/voice/{VOICE_ID}:ro" in mounts(argv)
    assert not [m for m in mounts(argv) if tts_engines.CONTAINER_VOICE_CLIP in m]
    tail = argv[argv.index("scripts.model_bench.speech.probes.tts_probe") + 1 :]
    assert value(tail, "--voice-id") == VOICE_ID
    assert VOICE_DIR_HOST not in " ".join(tail)


@pytest.mark.parametrize("build", [tts_suite.timed_argv, tts_suite.render_argv])
def test_an_identity_and_a_clip_are_exclusive(build):
    extra = {"expressive": False, "voices": ()} if build is tts_suite.render_argv else {}
    first = () if build is tts_suite.render_argv else ("idle",)

    with pytest.raises(core.VoiceClipError, match="voice-dir"):
        build(
            CLONE,
            *first,
            out_dir=OUT,
            repo_dir="/repo",
            speech_models_dir="/models",
            voice_clip=CLIP_HOST,
            voice_dir=VOICE_DIR_HOST,
            voice_id=VOICE_ID,
            **extra,
        )


@pytest.mark.parametrize("engine", [e for e in TTS if e != CLONE])
def test_no_other_engine_s_argv_takes_an_identity(engine):
    with pytest.raises(core.VoiceClipError):
        timed_on_identity(engine)


def test_an_identity_mount_needs_its_voice_id():
    with pytest.raises(core.SpeechBenchError):
        timed_on_identity(voice_id=None)


def test_the_loop_argv_mounts_the_identity_too():
    argv = tts_suite.loop_container_argv(
        CLONE,
        line_id="L01",
        seconds=5,
        repo_dir="/repo",
        speech_models_dir="/models",
        voice_dir=VOICE_DIR_HOST,
        voice_id=VOICE_ID,
    )

    assert f"{VOICE_DIR_HOST}:/voice/{VOICE_ID}:ro" in mounts(argv)
    assert value(argv, "--voice-id") == VOICE_ID


@pytest.mark.parametrize("engine", list(TTS))
def test_every_engine_container_is_offline_read_only_and_mounts_the_repository(engine):
    argv = timed(engine)

    assert "HF_HUB_OFFLINE=1" in envs(argv)
    assert "/repo:/work:ro" in mounts(argv)
    assert "/models:/models/speech:ro" in mounts(argv)
    assert [m for m in mounts(argv) if not m.endswith(":ro")] == [f"{OUT}:/out"]
    clip_mounts = [m for m in mounts(argv) if tts_engines.CONTAINER_VOICE_CLIP in m]
    assert len(clip_mounts) == (1 if engine == CLONE else 0)
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
        engine,
        line_id="L05",
        seconds=90,
        repo_dir="/repo",
        speech_models_dir="/models",
        voice_clip=clip_for(engine),
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
# The cloned voice: which runs take a clip, and what they record of it
# ---------------------------------------------------------------------------


def test_a_default_run_leaves_out_pocket_clone_unless_a_clip_is_given(tmp_path):
    without = tts_suite.default_engines(TTS, None)
    with_clip = tts_suite.default_engines(TTS, tmp_path / "clip.wav")

    assert without == tuple(e for e in TTS if e != CLONE)
    assert with_clip == tuple(TTS)


def _refused_before_anything(tmp_path, req) -> str:
    log: list[str] = []
    with pytest.raises(core.VoiceClipError) as info:
        tts_suite.run_tts(
            req,
            runner=FakeRunner(log, Now()),
            container_state=lambda name: log.append("state") or "running",
            load_gen_factory=lambda: FakeLoadGen(log),
            sampler_factory=lambda path: FakeSampler(log, path),
            git_sha="x",
            repo_root=_REPO_ROOT,
        )
    assert log == []
    assert not (tmp_path / "results").exists()
    return str(info.value)


def test_run_tts_refuses_pocket_clone_without_a_clip_before_anything(tmp_path):
    _refused_before_anything(tmp_path, request(tmp_path, engines=(CLONE,)))


def test_run_tts_refuses_a_clip_for_a_run_without_pocket_clone(tmp_path):
    clip = write_identity_clip(tmp_path / "identity")

    message = _refused_before_anything(
        tmp_path, request(tmp_path, engines=("pocket",), voice_clip=clip)
    )

    assert str(clip) not in message and clip.as_posix() not in message


def test_run_tts_refuses_a_clip_that_does_not_exist(tmp_path):
    clip = tmp_path / "identity" / "absent.wav"

    message = _refused_before_anything(
        tmp_path, request(tmp_path, engines=(CLONE,), voice_clip=clip)
    )

    assert "absent" not in message


def _all_text_under(root: Path) -> str:
    return "\n".join(
        p.read_bytes().decode("utf-8", errors="replace") for p in root.rglob("*") if p.is_file()
    )


def test_a_cloned_run_records_the_identity_and_the_clip_s_sha256_prefix_only(tmp_path, caplog):
    caplog.set_level("DEBUG")
    identity_root = tmp_path / "identity"
    clip = write_identity_clip(identity_root)
    before = sorted((p.name, p.stat().st_mtime_ns) for p in identity_root.iterdir())
    sha = core.sha256_file(clip)[:12]

    _, runner = run_suite(tmp_path, request(tmp_path, engines=(CLONE,), voice_clip=clip))

    tts_dir = tmp_path / "results" / "mh1" / "speech" / "tts"
    samples = read_jsonl(tts_dir / "samples.jsonl")
    [load] = read_jsonl(tts_dir / "load.jsonl")
    assert samples and all(r["voice"] == core.IDENTITY_VOICE for r in samples)
    assert {r["voice_clip_sha256"] for r in samples} == {sha}
    assert (load["voice"], load["voice_clip_sha256"]) == (core.IDENTITY_VOICE, sha)
    clip_mount = f"{clip.resolve().as_posix()}:{tts_engines.CONTAINER_VOICE_CLIP}:ro"
    assert all(clip_mount in mounts(argv) for argv in runner.argvs)
    written = _all_text_under(tmp_path / "results")
    logged = "\n".join(r.getMessage() for r in caplog.records)
    for secret in (str(clip), clip.as_posix(), clip.resolve().as_posix(), TRANSCRIPT, "clip.wav"):
        assert secret not in written
        assert secret not in logged
    assert sorted((p.name, p.stat().st_mtime_ns) for p in identity_root.iterdir()) == before


def test_a_run_without_a_clip_records_no_clip_field(tmp_path):
    run_suite(tmp_path, request(tmp_path, engines=("pocket",)))

    rows = read_jsonl(tmp_path / "results" / "mh1" / "speech" / "tts" / "samples.jsonl")
    assert all("voice_clip_sha256" not in r for r in rows)


def test_the_docker_runner_logs_the_clip_mount_as_a_placeholder(tmp_path, monkeypatch):
    clip = write_identity_clip(tmp_path / "identity")

    class FakePopen:
        def __init__(self, argv, **kwargs):
            self.stdout = iter(["ready\n"])

        def wait(self):
            return 0

    monkeypatch.setattr(tts_suite.subprocess, "Popen", FakePopen)
    argv = tts_suite.timed_argv(
        CLONE,
        "idle",
        out_dir=OUT,
        repo_dir="/repo",
        speech_models_dir="/models",
        voice_clip=clip.as_posix(),
    )
    log_path = tmp_path / "idle.log"

    code = tts_suite.DockerRunner().run(argv, log_path=log_path, on_line=lambda line: None)

    text = log_path.read_text(encoding="utf-8")
    assert code == 0
    assert clip.as_posix() not in text and str(clip) not in text
    assert f"{core.VOICE_CLIP_PLACEHOLDER}:{tts_engines.CONTAINER_VOICE_CLIP}:ro" in text


# ---------------------------------------------------------------------------
# A trained identity: `--voice-dir`, what it records, and its refusals
# ---------------------------------------------------------------------------


def test_a_default_run_includes_pocket_clone_with_an_identity(tmp_path):
    assert tts_suite.default_engines(TTS, tmp_path / "voices" / VOICE_ID) == tuple(TTS)


def test_a_run_on_an_identity_records_its_id_weights_prefix_and_rss_but_no_path(tmp_path, caplog):
    caplog.set_level("DEBUG")
    voices = tmp_path / "private" / "voices"
    identity = write_identity(voices)
    before = sorted((p.name, p.stat().st_mtime_ns) for p in identity.iterdir())

    _, runner = run_suite(tmp_path, request(tmp_path, engines=(CLONE,), voice_dir=identity))

    tts_dir = tmp_path / "results" / "mh1" / "speech" / "tts"
    samples = read_jsonl(tts_dir / "samples.jsonl")
    [load] = read_jsonl(tts_dir / "load.jsonl")
    expected = {"voice_id": VOICE_ID, "voice_weights_sha256": _sha(TRAINED_WEIGHTS)[:12]}
    assert samples and all(r["voice"] == core.IDENTITY_VOICE for r in samples)
    assert all({k: r[k] for k in expected} == expected for r in samples)
    assert {k: load[k] for k in expected} == expected
    assert all(r["rss_peak_mib"] == 1536.0 for r in samples)
    assert all("voice_clip_sha256" not in r for r in [*samples, load])
    mount = f"{identity.resolve().as_posix()}:/voice/{VOICE_ID}:ro"
    assert all(mount in mounts(argv) for argv in runner.argvs)
    written = _all_text_under(tmp_path / "results")
    logged = "\n".join(r.getMessage() for r in caplog.records)
    for secret in (str(identity), identity.as_posix(), identity.resolve().as_posix(), TRANSCRIPT):
        assert secret not in written
        assert secret not in logged
    assert sorted((p.name, p.stat().st_mtime_ns) for p in identity.iterdir()) == before


def test_a_run_on_a_stock_identity_records_no_weights_prefix(tmp_path):
    identity = write_identity(tmp_path / "voices", trained=False)

    run_suite(tmp_path, request(tmp_path, engines=(CLONE,), voice_dir=identity))

    [load] = read_jsonl(tmp_path / "results" / "mh1" / "speech" / "tts" / "load.jsonl")
    assert (load["voice_id"], load["voice_weights_sha256"]) == (VOICE_ID, None)


def _refused_identity(tmp_path, req) -> str:
    message = _refused_before_anything(tmp_path, req)
    for secret in (str(tmp_path), tmp_path.as_posix(), "private-host-name"):
        assert secret not in message
    return message


def test_run_tts_refuses_an_identity_and_a_clip_together(tmp_path):
    identity = write_identity(tmp_path / "voices")
    clip = write_identity_clip(tmp_path / "identity")

    message = _refused_identity(
        tmp_path, request(tmp_path, engines=(CLONE,), voice_dir=identity, voice_clip=clip)
    )

    assert "--voice-clip" in message and "--voice-dir" in message


def test_run_tts_refuses_an_identity_for_a_run_without_pocket_clone(tmp_path):
    identity = write_identity(tmp_path / "voices")

    _refused_identity(tmp_path, request(tmp_path, engines=("pocket",), voice_dir=identity))


@pytest.mark.parametrize("case", ["missing", "no-manifest", "bad-voice-id"])
def test_run_tts_refuses_an_unusable_identity_directory(tmp_path, case):
    folder = tmp_path / "voices" / "private-host-name"
    if case != "missing":
        folder.mkdir(parents=True)
    if case == "bad-voice-id":
        (folder / "voice.json").write_text(json.dumps({"voice_id": "../up"}), encoding="utf-8")

    _refused_identity(tmp_path, request(tmp_path, engines=(CLONE,), voice_dir=folder))


def test_the_docker_runner_logs_the_identity_mount_as_a_placeholder(tmp_path, monkeypatch):
    identity = write_identity(tmp_path / "voices")

    class FakePopen:
        def __init__(self, argv, **kwargs):
            self.stdout = iter(["ready\n"])

        def wait(self):
            return 0

    monkeypatch.setattr(tts_suite.subprocess, "Popen", FakePopen)
    argv = timed_on_identity(voice_dir=identity.as_posix())
    log_path = tmp_path / "idle.log"

    code = tts_suite.DockerRunner().run(argv, log_path=log_path, on_line=lambda line: None)

    text = log_path.read_text(encoding="utf-8")
    assert code == 0
    assert identity.as_posix() not in text and str(identity) not in text
    assert f"{core.VOICE_DIR_PLACEHOLDER}:/voice/{VOICE_ID}:ro" in text


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
    clone_lines = [line for line in out.splitlines() if line.startswith("[argv] tts pocket-clone")]
    assert len(clone_lines) == 3
    placeholder = f"{core.VOICE_CLIP_PLACEHOLDER}:{tts_engines.CONTAINER_VOICE_CLIP}:ro"
    assert all(placeholder in line for line in clone_lines)
    others = [line for line in out.splitlines() if "pocket-clone" not in line]
    assert not [line for line in others if tts_engines.CONTAINER_VOICE_CLIP in line]


def test_selftest_runs_the_tts_suite_selftest(capsys):
    code = speech_bench.main(["selftest"])

    assert code == 0
    assert "[ok] tts_suite" in capsys.readouterr().out


def test_tts_validate_inputs_finds_no_problem_in_the_committed_material():
    assert tts_suite.validate_inputs(_REPO_ROOT) == []
