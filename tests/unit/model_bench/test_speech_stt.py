"""Run 3, speech to text: word error rate, the engine adapters, the timing and the `stt` suite.

No test needs a GPU, the network or an engine package: adapters are driven through fakes at the
`Transcriber` protocol, the streaming feeder through a scripted clock and sleep, and the suite
through a fake container runner that writes what the probe would. Audio is synthetic and made in
the test; no audio is committed.
"""

from __future__ import annotations

import ast
import json
import subprocess
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
from scripts.model_bench.speech import core, stt_suite, wer  # noqa: E402
from scripts.model_bench.speech.probes import stt_engines, stt_probe  # noqa: E402

SR = 16000
CPU_ENGINES = ("parakeet-cpu",)
GPU_ENGINES = ("whisper-base", "parakeet-gpu", "nemotron-160")


# ---------------------------------------------------------------------------
# Word error rate on hand-computed pairs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reference", "hypothesis", "counts", "rate"),
    [
        pytest.param("turn the lights on", "turn the light on", (1, 0, 0), 0.25, id="substitution"),
        pytest.param("play music", "play the music", (0, 0, 1), 0.5, id="insertion"),
        pytest.param("what time is it", "what is it", (0, 1, 0), 0.25, id="deletion"),
        pytest.param("a b c d", "a x c d e", (1, 0, 1), 0.5, id="substitution-and-insertion"),
        pytest.param(
            "Okay, that's enough for tonight. Go to sleep.",
            "okay that’s enough for tonight go to sleep",
            (0, 0, 0),
            0.0,
            id="punctuation-and-curly-apostrophe",
        ),
        pytest.param(
            "Play the playlist from ... last Sunday's run.",
            "play the playlist from last sundays run",
            (0, 0, 0),
            0.0,
            id="dropped-apostrophe",
        ),
        pytest.param(
            "Can you remind me to ... pick up the parcel tomorrow?",
            "Can you remind me to pick up the parcel tomorrow",
            (0, 0, 0),
            0.0,
            id="pause-mark",
        ),
        pytest.param("Set a timer for ten minutes.", "Set a timer for 10 minutes.", (0, 0, 0), 0.0),
        pytest.param(
            "Cancel the alarm for six thirty and set one for seven fifteen instead.",
            "Cancel the alarm for 6:30 and set one for 7:15 instead.",
            (0, 0, 0),
            0.0,
            id="times",
        ),
        pytest.param(
            "Turn the living room lights down to thirty percent.",
            "Turn the living room lights down to 30%.",
            (0, 0, 0),
            0.0,
            id="percent",
        ),
        pytest.param(
            "Show me the report on the second monitor.",
            "Show me the report on the 2nd monitor.",
            (0, 0, 0),
            0.0,
            id="ordinal",
        ),
        pytest.param(
            "Remind me to call the dentist on Monday at nine.",
            "Remind me to call the dentist on Monday at 9:00.",
            (0, 0, 0),
            0.0,
            id="on-the-hour",
        ),
        pytest.param(
            "Set a timer for ten minutes.", "Set a timer for 12 minutes.", (1, 0, 0), 1 / 6
        ),
    ],
)
def test_word_error_rate_matches_hand_computed_pairs(reference, hypothesis, counts, rate):
    result = wer.word_error_rate(reference, hypothesis)

    assert (result.substitutions, result.deletions, result.insertions) == counts
    assert result.wer == pytest.approx(rate)


@pytest.mark.parametrize(
    ("number", "words"),
    [
        (0, "zero"),
        (15, "fifteen"),
        (30, "thirty"),
        (42, "forty two"),
        (105, "one hundred five"),
        (2026, "two thousand twenty six"),
        (1_500_000, "one million five hundred thousand"),
    ],
)
def test_int_to_words(number, words):
    assert wer.int_to_words(number) == words


def test_numbers_with_decimals_and_grouping_are_written_out():
    assert (
        wer.normalise("2.5 km and 1,500 steps")
        == "two point five km and one thousand five hundred steps".split()
    )


def test_an_empty_hypothesis_deletes_every_reference_word():
    result = wer.word_error_rate("set a timer", "")

    assert (result.deletions, result.wer) == (3, 1.0)


# ---------------------------------------------------------------------------
# Whisper settings: MIST's own prompt and fp16=False
# ---------------------------------------------------------------------------


def _mist_initial_prompt() -> str:
    tree = ast.parse((_REPO_ROOT / "src" / "multimodal" / "stt.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == "initial_prompt":
                    return ast.literal_eval(node.value)
    raise AssertionError("src/multimodal/stt.py sets no self.initial_prompt")


def test_whisper_prompt_is_the_text_mist_passes_in_src_multimodal_stt():
    assert _mist_initial_prompt() == stt_engines.WHISPER_INITIAL_PROMPT


class FakeWhisperModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return {"text": "  Set a timer for ten minutes.  "}


def test_whisper_adapter_passes_fp16_false_and_mists_prompt_and_strips_the_text():
    model = FakeWhisperModel()
    audio = np.zeros(SR, dtype=np.int16).astype(np.float64)

    text = stt_engines.WhisperTranscriber(model).transcribe(audio, SR)

    assert text == "Set a timer for ten minutes."
    sent, kwargs = model.calls[0]
    assert kwargs == {"fp16": False, "initial_prompt": _mist_initial_prompt()}
    assert sent.dtype == np.float32


def test_whisper_adapter_refuses_audio_that_is_not_16_khz():
    with pytest.raises(ValueError, match="16000"):
        stt_engines.WhisperTranscriber(FakeWhisperModel()).transcribe(np.zeros(10), 8000)


def test_whisper_factory_loads_base_from_the_mounted_weights_on_cuda(monkeypatch, tmp_path):
    calls = {}
    fake = types.ModuleType("whisper")

    def load_model(name, device=None, download_root=None):
        calls.update(name=name, device=device, download_root=download_root)
        return FakeWhisperModel()

    fake.load_model = load_model
    monkeypatch.setitem(sys.modules, "whisper", fake)

    engine = stt_engines.make_transcriber("whisper-base", weights_dir=tmp_path)

    assert isinstance(engine, stt_engines.WhisperTranscriber)
    assert calls == {
        "name": str(tmp_path / "openai-whisper" / "base.pt"),
        "device": "cuda",
        "download_root": None,
    }


def test_factory_refuses_an_unknown_engine(tmp_path):
    with pytest.raises(KeyError, match="no-such"):
        stt_engines.make_transcriber("no-such", weights_dir=tmp_path)


def test_every_stt_engine_in_speech_arms_has_a_weights_path_and_a_factory():
    arms = core.load_json(core.SPEECH_ARMS_PATH)["stt"]

    assert set(stt_engines.WEIGHT_PATHS) == set(arms)
    assert tuple(arms) == stt_suite.STT_ENGINES


class FakeOnnxAsrModel:
    def __init__(self):
        self.calls = []

    def recognize(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return " Play something calm. "


def test_onnx_asr_adapter_passes_the_sample_rate_and_strips_the_text():
    model = FakeOnnxAsrModel()

    text = stt_engines.OnnxAsrTranscriber(model).transcribe(np.zeros(SR, np.float32), SR)

    assert text == "Play something calm."
    assert model.calls[0][1] == {"sample_rate": SR}


def test_parakeet_cpu_factory_loads_from_weights_on_cpu_with_the_thread_count(
    monkeypatch, tmp_path
):
    calls = {}

    class SessionOptions:
        pass

    ort = types.ModuleType("onnxruntime")
    ort.SessionOptions = SessionOptions
    asr = types.ModuleType("onnx_asr")

    def load_model(model, path=None, **kwargs):
        calls.update(model=model, path=path, **kwargs)
        return FakeOnnxAsrModel()

    asr.load_model = load_model
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setitem(sys.modules, "onnx_asr", asr)

    engine = stt_engines.make_transcriber("parakeet-cpu", weights_dir=tmp_path, threads=4)

    assert isinstance(engine, stt_engines.OnnxAsrTranscriber)
    assert calls["model"] == "nemo-parakeet-tdt-0.6b-v2"
    assert calls["path"] == str(tmp_path / stt_engines.WEIGHT_PATHS["parakeet-cpu"])
    assert calls["providers"] == ["CPUExecutionProvider"]
    assert calls["sess_options"].intra_op_num_threads == 4
    assert calls["sess_options"].inter_op_num_threads == 1


def test_parakeet_cpu_factory_refuses_more_than_four_threads(tmp_path):
    with pytest.raises(ValueError, match="4"):
        stt_engines.make_transcriber("parakeet-cpu", weights_dir=tmp_path, threads=8)


class FakeHypothesis:
    def __init__(self, text):
        self.text = text


class FakeNemoModel:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return self.reply


@pytest.mark.parametrize(
    "reply",
    [
        [FakeHypothesis(" turn on the lights ")],
        ([FakeHypothesis(" turn on the lights ")], None),
        [" turn on the lights "],
    ],
    ids=["hypotheses", "best-and-all", "strings"],
)
def test_nemo_batch_adapter_reads_the_text_from_each_documented_reply_shape(reply):
    model = FakeNemoModel(reply)

    text = stt_engines.NemoTranscriber(model).transcribe(np.zeros(SR, np.float32), SR)

    assert text == "turn on the lights"
    audio, kwargs = model.calls[0]
    assert len(audio) == 1 and audio[0].dtype == np.float32
    assert kwargs["batch_size"] == 1


# ---------------------------------------------------------------------------
# Timing: batch and the paced streaming feeder, on a scripted clock
# ---------------------------------------------------------------------------


class ScriptedClock:
    """A wall clock that only moves when the code under test sleeps or an engine works."""

    def __init__(self, start: float = 100.0):
        self.now = start
        self.sleeps = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += max(0.0, seconds)

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeStreamingEngine:
    def __init__(self, clock, *, feed_s=0.0, finish_s=0.05):
        self.clock = clock
        self.feed_s = feed_s
        self.finish_s = finish_s
        self.fed = []

    def transcribe(self, audio, sample_rate):
        return "whole"

    def feed(self, chunk):
        self.fed.append((self.clock(), len(chunk)))
        self.clock.advance(self.feed_s)

    def finish(self):
        self.clock.advance(self.finish_s)
        return "final text"


def test_streaming_feeder_paces_160_ms_chunks_to_the_wall_clock_and_times_from_the_last():
    clock = ScriptedClock()
    engine = FakeStreamingEngine(clock)
    syncs = []
    audio = np.zeros(8000, dtype=np.float32)  # 0.5 s: three full 160 ms chunks and 20 ms

    timing = stt_probe.stream_timed(
        engine,
        audio,
        sample_rate=SR,
        chunk_ms=160,
        clock=clock,
        sleep=clock.sleep,
        sync=lambda: syncs.append(clock()),
    )

    assert [n for _, n in engine.fed] == [2560, 2560, 2560, 320]
    fed_at = [round(t - 100.0, 6) for t, _ in engine.fed]
    assert fed_at == [0.16, 0.32, 0.48, 0.5]
    assert timing.text == "final text"
    assert timing.final_ms == pytest.approx(50.0)
    assert timing.feed_lag_ms == pytest.approx(0.0)
    assert syncs[-2:] == [pytest.approx(100.5), pytest.approx(100.55)]


def test_streaming_feeder_records_the_lag_when_the_engine_falls_behind_real_time():
    clock = ScriptedClock()
    engine = FakeStreamingEngine(clock, feed_s=0.2, finish_s=0.01)
    audio = np.zeros(2560 * 3, dtype=np.float32)

    timing = stt_probe.stream_timed(
        engine, audio, sample_rate=SR, chunk_ms=160, clock=clock, sleep=clock.sleep, sync=None
    )

    fed_at = [round(t - 100.0, 6) for t, _ in engine.fed]
    assert fed_at == [0.16, 0.36, 0.56]
    assert timing.feed_lag_ms == pytest.approx(80.0)
    assert timing.final_ms == pytest.approx(210.0)  # the last feed's own work, then finish


def test_batch_timing_runs_from_the_call_to_the_final_text_with_syncs_around_it():
    clock = ScriptedClock()
    events = []

    class Engine:
        def transcribe(self, audio, sample_rate):
            events.append(("call", clock()))
            clock.advance(0.042)
            return "text"

    timing = stt_probe.batch_timed(
        Engine(),
        np.zeros(SR, np.float32),
        sample_rate=SR,
        clock=clock,
        sync=lambda: events.append(("sync", clock())),
    )

    assert timing.final_ms == pytest.approx(42.0)
    assert [kind for kind, _ in events] == ["sync", "call", "sync"]


def test_loop_mode_warms_up_prints_ready_transcribes_for_the_seconds_then_prints_done():
    clock = ScriptedClock()
    calls = []

    class Engine:
        def transcribe(self, audio, sample_rate):
            calls.append(clock())
            clock.advance(0.25)
            return "text"

    lines = []

    class Out:
        def write(self, text):
            lines.append(text)

        def flush(self):
            lines.append("<flush>")

    count = stt_probe.run_loop(
        Engine(),
        np.zeros(SR, np.float32),
        sample_rate=SR,
        seconds=1.0,
        warmups=3,
        clock=clock,
        out=Out(),
    )

    text = "".join(lines)
    assert text.index("READY\n") < text.index("DONE\n")
    assert text.index("READY\n<flush>") >= 0
    assert text.endswith("DONE\n<flush>")
    assert count == 4  # 0.0, 0.25, 0.5, 0.75 after READY; stops at 1.0
    assert len(calls) == 3 + 4


def test_loop_parser_takes_engine_utterance_and_seconds():
    args = stt_probe.build_parser().parse_args(
        ["loop", "--engine", "parakeet-cpu", "--utt", "U01", "--seconds", "30"]
    )

    assert (args.mode, args.engine, args.utt, args.seconds) == ("loop", "parakeet-cpu", "U01", 30.0)


# ---------------------------------------------------------------------------
# Container argv per engine, and the loop argv run 5 uses
# ---------------------------------------------------------------------------


def _argv(engine, **kw):
    return stt_suite.stt_container_argv(
        engine,
        repo_dir="D:/repo",
        material_dir="D:/material",
        weights_dir="D:/models/speech",
        out_dir="D:/results/mh1/speech/stt/" + engine,
        **kw,
    )


def _pairs(argv, flag):
    return [argv[i + 1] for i, a in enumerate(argv) if a == flag]


def _image_index(argv):
    for i, a in enumerate(argv):
        if a.startswith("mist-speech-bench-"):
            return i
    raise AssertionError("no bench image in argv")


@pytest.mark.parametrize("engine", GPU_ENGINES + CPU_ENGINES)
def test_every_stt_container_is_offline_with_read_only_inputs(engine):
    argv = _argv(engine)
    docker_part = argv[: _image_index(argv)]

    assert argv[:3] == ["docker", "run", "--rm"]
    assert _pairs(docker_part, "--network") == ["none"]
    mounts = _pairs(docker_part, "-v")
    assert "D:/repo:/work:ro" in mounts
    assert "D:/material:/material:ro" in mounts
    assert "D:/models/speech:/weights:ro" in mounts
    assert f"D:/results/mh1/speech/stt/{engine}:/out" in mounts
    assert all(m.endswith(":ro") or m.endswith(":/out") for m in mounts)
    assert "HF_HUB_OFFLINE=1" in _pairs(docker_part, "-e")
    assert _pairs(docker_part, "-w") == ["/work"]


@pytest.mark.parametrize(
    ("engine", "image"),
    [
        ("whisper-base", "mist-speech-bench-base"),
        ("parakeet-gpu", "mist-speech-bench-asr"),
        ("nemotron-160", "mist-speech-bench-asr"),
        ("parakeet-cpu", "mist-speech-bench-asr"),
    ],
)
def test_each_engine_runs_on_its_image_and_calls_the_probe(engine, image):
    argv = _argv(engine, reps=3, warmups=3)
    tail = argv[_image_index(argv) :]

    assert tail[0] == image
    assert tail[1:5] == ["python", "-m", "scripts.model_bench.speech.probes.stt_probe", "run"]
    assert _pairs(tail, "--engine") == [engine]
    assert _pairs(tail, "--reps") == ["3"]
    assert _pairs(tail, "--warmups") == ["3"]


@pytest.mark.parametrize("engine", GPU_ENGINES)
def test_gpu_engines_get_all_gpus_and_no_cpu_limit(engine):
    argv = _argv(engine)

    assert _pairs(argv, "--gpus") == ["all"]
    assert "--cpus" not in argv


def test_parakeet_cpu_gets_no_gpus_and_four_threads():
    argv = _argv("parakeet-cpu")
    docker_part = argv[: _image_index(argv)]
    env = _pairs(docker_part, "-e")

    assert "--gpus" not in argv
    assert _pairs(docker_part, "--cpus") == ["4"]
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert f"{name}=4" in env
    assert _pairs(argv, "--threads") == ["4"]


def test_utterance_filter_is_passed_to_the_probe():
    argv = _argv("whisper-base", utt_ids=("U01", "P02"))

    assert _pairs(argv, "--utt") == ["U01", "P02"]


def test_an_unknown_engine_is_refused():
    with pytest.raises(core.SpeechBenchError, match="unknown"):
        _argv("no-such")


def test_loop_container_argv_runs_the_probe_loop_mode_for_the_given_seconds():
    argv = stt_suite.loop_container_argv(
        "parakeet-cpu",
        utt_id="U03",
        seconds=45,
        repo_dir="D:/repo",
        material_dir="D:/material",
        weights_dir="D:/models/speech",
    )
    image = _image_index(argv)
    docker_part, tail = argv[:image], argv[image:]

    assert argv[:3] == ["docker", "run", "--rm"]
    assert _pairs(docker_part, "--name") == ["mist-bench-stt-loop-parakeet-cpu"]
    assert _pairs(docker_part, "--network") == ["none"]
    assert "--gpus" not in argv
    assert _pairs(docker_part, "--cpus") == ["4"]
    assert ":/out" not in " ".join(_pairs(docker_part, "-v"))
    assert tail[:5] == [
        "mist-speech-bench-asr",
        "python",
        "-m",
        "scripts.model_bench.speech.probes.stt_probe",
        "loop",
    ]
    assert _pairs(tail, "--utt") == ["U03"]
    assert _pairs(tail, "--seconds") == ["45"]
    assert _pairs(tail, "--engine") == ["parakeet-cpu"]


def test_loop_container_argv_for_a_gpu_engine_gets_all_gpus():
    argv = stt_suite.loop_container_argv(
        "parakeet-gpu",
        utt_id="U01",
        seconds=10,
        repo_dir="D:/repo",
        material_dir="D:/material",
        weights_dir="D:/models/speech",
        name="fit-parakeet",
    )

    assert _pairs(argv, "--gpus") == ["all"]
    assert _pairs(argv, "--name") == ["fit-parakeet"]


def test_loop_container_argv_refuses_a_non_positive_duration():
    with pytest.raises(core.SpeechBenchError, match="seconds"):
        stt_suite.loop_container_argv(
            "parakeet-cpu",
            utt_id="U01",
            seconds=0,
            repo_dir="D:/repo",
            material_dir="D:/material",
            weights_dir="D:/models/speech",
        )


# ---------------------------------------------------------------------------
# The `stt` subcommand, end to end against a fake container runner
# ---------------------------------------------------------------------------


def _write_wav(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())


def _material(tmp_path: Path) -> Path:
    material = tmp_path / "material"
    material.mkdir()
    items = []
    for uid, kind, text in (
        ("U01", "complete", "Set a timer for ten minutes."),
        ("P01", "pause", "Can you remind me to ... pick up the parcel tomorrow?"),
    ):
        t = np.arange(SR * 2) / SR
        _write_wav(material / f"{uid}.wav", 0.3 * np.sin(2 * np.pi * 220 * t))
        items.append(
            {
                "id": uid,
                "file": f"{uid}.wav",
                "kind": kind,
                "text": text,
                "duration_s": 2.0,
                "sha256": core.sha256_file(material / f"{uid}.wav"),
            }
        )
    (material / "manifest.json").write_text(json.dumps({"items": items}), encoding="utf-8")
    return material


def _out_mount(argv):
    for mount in _pairs(argv, "-v"):
        if mount.endswith(":/out"):
            return Path(mount[: -len(":/out")])
    raise AssertionError("no /out mount")


class FakeRunner:
    """Stands in for `docker run`: writes the probe's raw rows into the `/out` mount."""

    def __init__(self, texts, *, not_loaded=()):
        self.texts = texts
        self.not_loaded = set(not_loaded)
        self.argvs = []

    def __call__(self, argv):
        self.argvs.append(argv)
        engine = _pairs(argv, "--engine")[0]
        out = _out_mount(argv)
        if engine in self.not_loaded:
            (out / "load.json").write_text(
                json.dumps({"engine": engine, "loaded": False, "error": "RuntimeError: boom"})
            )
            return subprocess.CompletedProcess(argv, stt_probe.NOT_LOADED_EXIT, "", "boom")
        (out / "load.json").write_text(json.dumps({"engine": engine, "loaded": True, "load_s": 1}))
        rows = []
        for uid, text in self.texts.items():
            for rep in (1, 2, 3):
                rows.append(
                    {
                        "utt_id": uid,
                        "rep": rep,
                        "audio_s": 2.0,
                        "final_ms": 40.0 + rep,
                        "text": text,
                        "torch_peak_reserved_mib": None,
                    }
                )
        (out / "raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        return subprocess.CompletedProcess(argv, 0, "probe ok\n", "")


def _stt_args(tmp_path, material, *extra):
    weights = tmp_path / "weights"
    weights.mkdir(exist_ok=True)
    return [
        "stt",
        "--results-root",
        str(tmp_path / "results"),
        "--run",
        "mh1",
        "--material-dir",
        str(material),
        "--weights-dir",
        str(weights),
        "--no-pins",
        *extra,
    ]


def test_host_and_probe_agree_on_the_not_loaded_exit_code():
    assert stt_suite.NOT_LOADED_EXIT == stt_probe.NOT_LOADED_EXIT


def test_stt_writes_one_row_per_rep_with_engine_device_and_word_error_rate(
    tmp_path, monkeypatch, capsys
):
    material = _material(tmp_path)
    runner = FakeRunner({"U01": "Set a timer for 10 minutes.", "P01": "Can you remind me to"})
    monkeypatch.setattr(stt_suite, "run_container", runner)

    code = speech_bench.main(_stt_args(tmp_path, material, "--engine", "parakeet-cpu"))

    assert code == 0, capsys.readouterr().out
    stt_dir = tmp_path / "results" / "mh1" / "speech" / "stt"
    rows = [json.loads(line) for line in (stt_dir / "samples.jsonl").read_text().splitlines()]
    assert len(rows) == 6
    first = rows[0]
    assert first["engine"] == "parakeet-cpu"
    assert first["device"] == "cpu"
    assert (first["utt_id"], first["rep"], first["final_ms"]) == ("U01", 1, 41.0)
    assert first["wer"] == 0.0
    assert {"run", "suite", "git_sha", "audio_s", "torch_peak_reserved_mib"} <= set(first)
    pause = [r for r in rows if r["utt_id"] == "P01"][0]
    assert pause["wer"] == pytest.approx(5 / 10)
    assert (stt_dir / "parakeet-cpu" / "container.log").is_file()
    assert len(runner.argvs) == 1


def test_stt_reports_an_engine_that_does_not_load_and_runs_the_others(
    tmp_path, monkeypatch, capsys
):
    material = _material(tmp_path)
    runner = FakeRunner({"U01": "set a timer for ten minutes"}, not_loaded={"nemotron-160"})
    monkeypatch.setattr(stt_suite, "run_container", runner)

    code = speech_bench.main(
        _stt_args(tmp_path, material, "--engine", "nemotron-160", "--engine", "whisper-base")
    )

    out = capsys.readouterr().out
    assert code == 0
    assert "[not loaded] nemotron-160: RuntimeError: boom" in out
    rows = (tmp_path / "results" / "mh1" / "speech" / "stt" / "samples.jsonl").read_text()
    engines = {json.loads(line)["engine"] for line in rows.splitlines()}
    assert engines == {"whisper-base"}


def test_stt_refuses_an_existing_engine_directory_before_running_anything(
    tmp_path, monkeypatch, capsys
):
    material = _material(tmp_path)
    runner = FakeRunner({"U01": "x"})
    monkeypatch.setattr(stt_suite, "run_container", runner)
    (tmp_path / "results" / "mh1" / "speech" / "stt" / "parakeet-gpu").mkdir(parents=True)

    code = speech_bench.main(
        _stt_args(tmp_path, material, "--engine", "whisper-base", "--engine", "parakeet-gpu")
    )

    assert code != 0
    assert "already exists" in capsys.readouterr().out
    assert runner.argvs == []
    assert not (tmp_path / "results" / "mh1" / "speech" / "stt" / "whisper-base").exists()


def test_stt_refuses_material_whose_sha256_differs_from_the_manifest(tmp_path, monkeypatch, capsys):
    material = _material(tmp_path)
    (material / "U01.wav").write_bytes(b"changed")
    runner = FakeRunner({"U01": "x"})
    monkeypatch.setattr(stt_suite, "run_container", runner)

    code = speech_bench.main(_stt_args(tmp_path, material))

    assert code != 0
    assert "sha256" in capsys.readouterr().out
    assert runner.argvs == []
    assert not (tmp_path / "results" / "mh1").exists()


def test_stt_refuses_an_utterance_the_manifest_does_not_list(tmp_path, monkeypatch, capsys):
    material = _material(tmp_path)
    runner = FakeRunner({"U01": "x"})
    monkeypatch.setattr(stt_suite, "run_container", runner)

    code = speech_bench.main(_stt_args(tmp_path, material, "--utt", "U99"))

    assert code != 0
    assert "U99" in capsys.readouterr().out
    assert runner.argvs == []


def test_stt_refuses_a_results_root_inside_the_repository(tmp_path, monkeypatch, capsys):
    material = _material(tmp_path)
    monkeypatch.setattr(stt_suite, "run_container", FakeRunner({}))
    args = _stt_args(tmp_path, material)
    args[args.index("--results-root") + 1] = str(_REPO_ROOT / "results")

    code = speech_bench.main(args)

    assert code != 0
    assert not (_REPO_ROOT / "results").exists()


def test_stt_checks_each_engines_weights_against_the_pin_file(tmp_path, monkeypatch, capsys):
    material = _material(tmp_path)
    runner = FakeRunner({"U01": "x"})
    monkeypatch.setattr(stt_suite, "run_container", runner)
    weights = tmp_path / "weights"
    (weights / "openai-whisper").mkdir(parents=True)
    (weights / "openai-whisper" / "base.pt").write_bytes(b"not the pinned weights")
    pins = tmp_path / "pins.json"
    pin = {
        "repository": "openai/whisper",
        "revision": "r",
        "file": "openai-whisper/base.pt",
        "sha256": "0" * 64,
        "size": len(b"not the pinned weights"),
    }
    pins.write_text(json.dumps({"models": {"whisper-base": pin}}))
    args = _stt_args(tmp_path, material, "--engine", "whisper-base", "--pins", str(pins))
    args.remove("--no-pins")

    code = speech_bench.main(args)

    assert code != 0
    assert "differs from the pin file" in capsys.readouterr().out
    assert runner.argvs == []


# ---------------------------------------------------------------------------
# Registry: plan and selftest see the suite as built
# ---------------------------------------------------------------------------


def test_plan_prints_the_stt_and_prepare_argv_and_ends_plan_ok(capsys):
    code = speech_bench.main(["plan"])

    out = capsys.readouterr().out
    assert code == 0
    assert "[skip] stt_suite" not in out
    for engine in GPU_ENGINES + CPU_ENGINES:
        assert f"[argv] stt {engine}:" in out
        assert f"[argv] stt loop {engine}:" in out
    assert "[argv] prepare:" in out
    assert out.strip().splitlines()[-1] == "plan ok"


def test_selftest_runs_the_stt_suite_selftest(capsys):
    code = speech_bench.main(["selftest"])

    out = capsys.readouterr().out
    assert code == 0, out
    assert "[ok] stt_suite" in out


def test_stt_suite_validate_inputs_finds_no_problem():
    assert stt_suite.validate_inputs(_REPO_ROOT) == []
