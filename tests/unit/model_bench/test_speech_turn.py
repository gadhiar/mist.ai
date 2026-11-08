"""Run 4, end of turn: decision points, truth labels, the Smart Turn input and the `turn` suite.

The waveforms are synthetic, made in the test with known silences: tones for speech and quiet
seeded noise for the room. Smart Turn is driven through fakes at the `TurnDetector` protocol (a
fake feature extractor and a fake onnxruntime session), so no engine package is imported.
"""

from __future__ import annotations

import json
import subprocess
import sys
import wave
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from scripts.model_bench import speech_bench  # noqa: E402
from scripts.model_bench.speech import core, turn_suite  # noqa: E402
from scripts.model_bench.speech.probes import prepare_material, turn_probe  # noqa: E402

SR = 16000


def _segments(*parts: tuple[str, float], seed: int = 0) -> np.ndarray:
    """Concatenate ("tone", seconds) and ("room", seconds) parts; the room is -60 dBFS noise."""
    rng = np.random.default_rng(seed)
    out = []
    for kind, seconds in parts:
        n = int(round(seconds * SR))
        if kind == "tone":
            t = np.arange(n) / SR
            out.append(0.3 * np.sin(2 * np.pi * 200 * t))
        else:
            out.append(rng.normal(0.0, 0.001, n))
    return np.concatenate(out).astype(np.float32)


# Speech at 0.3-1.3 s, 1.8-2.6 s and 2.7-3.3 s; a 500 ms pause, a 100 ms gap and 600 ms of
# trailing room after the last word.
KNOWN = (
    ("room", 0.3),
    ("tone", 1.0),
    ("room", 0.5),
    ("tone", 0.8),
    ("room", 0.1),
    ("tone", 0.6),
    ("room", 0.6),
)


# ---------------------------------------------------------------------------
# The frame detector and the decision points
# ---------------------------------------------------------------------------


def test_silences_are_found_in_20_ms_frames_after_the_first_word():
    audio = _segments(*KNOWN)

    found = prepare_material.silences(audio, SR)

    assert [(s.start_ms, s.end_ms) for s in found] == [(1300, 1800), (3300, 3900)]


def test_a_gap_shorter_than_200_ms_is_not_a_silence():
    audio = _segments(("room", 0.3), ("tone", 0.5), ("room", 0.18), ("tone", 0.5), ("room", 0.5))

    found = prepare_material.silences(audio, SR)

    assert [(s.start_ms, s.end_ms) for s in found] == [(1480, 1980)]


def test_the_threshold_follows_the_utterances_own_noise_floor():
    quiet = _segments(*KNOWN)
    loud_room = quiet.copy()
    # A noisier room (about -40 dBFS) still separates from speech at its own floor plus 10 dB.
    rng = np.random.default_rng(1)
    room = np.abs(quiet) < 0.01
    loud_room[room] = rng.normal(0.0, 0.01, int(room.sum()))

    assert [(s.start_ms, s.end_ms) for s in prepare_material.silences(loud_room, SR)] == [
        (1300, 1800),
        (3300, 3900),
    ]


def test_decision_points_cut_200_ms_into_each_silence_with_truth_labels():
    points = turn_probe.decision_points(_segments(*KNOWN), SR)

    assert [(p.cut_ms, p.truth) for p in points] == [
        (1500, "not_finished"),
        (3500, "finished"),
    ]


def test_a_pause_utterance_has_not_finished_points_before_its_last_word_and_one_finished():
    audio = _segments(
        ("room", 0.2),
        ("tone", 0.6),
        ("room", 1.0),  # the thinking pause
        ("tone", 0.4),
        ("room", 0.3),  # a comma
        ("tone", 0.5),
        ("room", 0.8),
    )

    points = turn_probe.decision_points(audio, SR)

    assert [(p.cut_ms, p.truth) for p in points] == [
        (1000, "not_finished"),
        (2400, "not_finished"),
        (3200, "finished"),
    ]


def test_leading_room_before_the_first_word_is_not_a_decision_point():
    audio = _segments(("room", 1.0), ("tone", 0.5), ("room", 0.5))

    points = turn_probe.decision_points(audio, SR)

    assert [(p.cut_ms, p.truth) for p in points] == [(1700, "finished")]


def test_trailing_silence_is_measured_from_the_last_speech_frame():
    assert prepare_material.trailing_silence_ms(_segments(*KNOWN), SR) == 600
    assert prepare_material.trailing_silence_ms(_segments(("room", 1.0)), SR) is None


# ---------------------------------------------------------------------------
# The Smart Turn input: the last 8 s at 16 kHz, zero-padded at the front
# ---------------------------------------------------------------------------


def test_a_short_input_is_zero_padded_at_the_front_to_8_s():
    audio = np.full(SR * 2, 0.5, dtype=np.float32)

    window = turn_probe.smart_turn_window(audio, SR)

    assert window.shape == (8 * SR,)
    assert window.dtype == np.float32
    assert not window[: 6 * SR].any()
    assert np.all(window[6 * SR :] == 0.5)


def test_a_long_input_keeps_only_its_last_8_s():
    audio = np.arange(SR * 10, dtype=np.float32)

    window = turn_probe.smart_turn_window(audio, SR)

    assert window.shape == (8 * SR,)
    assert window[0] == 2 * SR
    assert window[-1] == SR * 10 - 1


def test_the_window_refuses_audio_that_is_not_16_khz():
    with pytest.raises(ValueError, match="16000"):
        turn_probe.smart_turn_window(np.zeros(100, np.float32), 8000)


class FakeFeatures:
    def __init__(self, array):
        self.input_features = array


class FakeFeatureExtractor:
    def __init__(self):
        self.calls = []

    def __call__(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return FakeFeatures(np.ones((1, 80, 800), dtype=np.float64))


class FakeSession:
    def __init__(self, p):
        self.p = p
        self.feeds = []

    def run(self, outputs, feeds):
        self.feeds.append(feeds)
        return [np.array([[self.p]], dtype=np.float32)]


def test_smart_turn_adapter_feeds_the_8_s_window_as_float32_log_mel_features():
    fe = FakeFeatureExtractor()
    session = FakeSession(0.83)
    audio = np.full(SR * 3, 0.25, dtype=np.float32)

    p = turn_probe.SmartTurnDetector(session, fe).predict(audio)

    assert p == pytest.approx(0.83)
    window, kwargs = fe.calls[0]
    assert window.shape == (8 * SR,)
    assert not window[: 5 * SR].any()
    assert kwargs == {
        "sampling_rate": SR,
        "return_tensors": "np",
        "padding": "max_length",
        "max_length": 8 * SR,
        "truncation": True,
        "do_normalize": True,
    }
    features = session.feeds[0]["input_features"]
    assert features.dtype == np.float32
    assert features.shape == (1, 80, 800)


def test_called_finished_is_a_probability_of_one_half_or_more():
    assert turn_probe.called_finished(0.5) is True
    assert turn_probe.called_finished(0.4999) is False


def test_each_point_is_inferred_five_times_and_its_median_is_the_points_time():
    now = [0.0]
    durations = iter([0.010, 0.012, 0.030, 0.011, 0.009])

    class Detector:
        def predict(self, audio):
            now[0] += next(durations)
            return 0.7

    p, times = turn_probe.measure_point(
        Detector(), np.zeros(SR, np.float32), reps=5, clock=lambda: now[0]
    )

    assert p == pytest.approx(0.7)
    assert times == pytest.approx([10.0, 12.0, 30.0, 11.0, 9.0])
    assert turn_probe.median_ms(times) == pytest.approx(11.0)


def test_smart_turn_loader_uses_the_cpu_provider_with_four_threads(monkeypatch, tmp_path):
    import types

    calls = {}

    class SessionOptions:
        pass

    def inference_session(path, sess_options=None, providers=None):
        calls.update(path=path, options=sess_options, providers=providers)
        return FakeSession(0.1)

    ort = types.ModuleType("onnxruntime")
    ort.SessionOptions = SessionOptions
    ort.InferenceSession = inference_session
    transformers = types.ModuleType("transformers")

    class WhisperFeatureExtractor:
        def __init__(self, chunk_length=None):
            calls["chunk_length"] = chunk_length

    transformers.WhisperFeatureExtractor = WhisperFeatureExtractor
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    turn_probe.load_smart_turn(tmp_path / "m.onnx", threads=4)

    assert calls["path"] == str(tmp_path / "m.onnx")
    assert calls["providers"] == ["CPUExecutionProvider"]
    assert calls["options"].intra_op_num_threads == 4
    assert calls["options"].inter_op_num_threads == 1
    assert calls["chunk_length"] == 8


def test_smart_turn_weights_path_follows_speech_arms():
    cfg = core.load_json(core.SPEECH_ARMS_PATH)["turn"]["smart-turn-v3.2"]["model"]

    assert f"{cfg['repository']}/{cfg['file']}" == turn_probe.SMART_TURN_WEIGHTS


# ---------------------------------------------------------------------------
# The turn container's argv
# ---------------------------------------------------------------------------


def _argv(**kw):
    return turn_suite.turn_container_argv(
        repo_dir="D:/repo",
        material_dir="D:/material",
        weights_dir="D:/models/speech",
        out_dir="D:/results/mh1/speech/turn/smart-turn-v3.2",
        **kw,
    )


def _pairs(argv, flag):
    return [argv[i + 1] for i, a in enumerate(argv) if a == flag]


def test_smart_turn_container_has_no_gpus_four_threads_and_no_network():
    argv = _argv()
    image = argv.index("mist-speech-bench-base")
    docker_part, tail = argv[:image], argv[image:]

    assert argv[:3] == ["docker", "run", "--rm"]
    assert "--gpus" not in argv
    assert _pairs(docker_part, "--network") == ["none"]
    assert _pairs(docker_part, "--cpus") == ["4"]
    env = _pairs(docker_part, "-e")
    assert "HF_HUB_OFFLINE=1" in env
    assert "OMP_NUM_THREADS=4" in env
    mounts = _pairs(docker_part, "-v")
    assert "D:/repo:/work:ro" in mounts
    assert "D:/material:/material:ro" in mounts
    assert "D:/models/speech:/weights:ro" in mounts
    assert "D:/results/mh1/speech/turn/smart-turn-v3.2:/out" in mounts
    assert tail[1:4] == ["python", "-m", "scripts.model_bench.speech.probes.turn_probe"]
    assert _pairs(tail, "--threads") == ["4"]
    assert _pairs(tail, "--reps") == ["5"]
    assert _pairs(tail, "--warmups") == ["3"]


# ---------------------------------------------------------------------------
# The `turn` subcommand against a fake container runner
# ---------------------------------------------------------------------------


def _material(tmp_path: Path) -> Path:
    material = tmp_path / "material"
    material.mkdir()
    audio = _segments(*KNOWN)
    path = material / "U01.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes((audio * 32767).astype("<i2").tobytes())
    item = {
        "id": "U01",
        "file": "U01.wav",
        "kind": "complete",
        "text": "Set a timer for ten minutes.",
        "duration_s": 3.9,
        "sha256": core.sha256_file(path),
    }
    (material / "manifest.json").write_text(json.dumps({"items": [item]}))
    return material


def test_turn_writes_the_probes_rows_to_turn_samples_with_the_stamp(tmp_path, monkeypatch, capsys):
    material = _material(tmp_path)
    argvs = []

    def runner(argv):
        argvs.append(argv)
        out = Path(next(m for m in _pairs(argv, "-v") if m.endswith(":/out"))[: -len(":/out")])
        rows = [
            {
                "utt_id": "U01",
                "kind": "complete",
                "cut_ms": 1500,
                "truth": "not_finished",
                "p_finished": 0.2,
                "called_finished": False,
                "inference_ms": 11.0,
            },
            {
                "utt_id": "U01",
                "kind": "complete",
                "cut_ms": 3500,
                "truth": "finished",
                "p_finished": 0.9,
                "called_finished": True,
                "inference_ms": 12.0,
            },
        ]
        (out / "raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(turn_suite, "run_container", runner)
    weights = tmp_path / "weights"
    weights.mkdir()

    code = speech_bench.main(
        [
            "turn",
            "--results-root",
            str(tmp_path / "results"),
            "--run",
            "mh1",
            "--material-dir",
            str(material),
            "--weights-dir",
            str(weights),
            "--no-pins",
        ]
    )

    out = capsys.readouterr().out
    assert code == 0, out
    turn_dir = tmp_path / "results" / "mh1" / "speech" / "turn"
    rows = [json.loads(line) for line in (turn_dir / "samples.jsonl").read_text().splitlines()]
    assert [r["cut_ms"] for r in rows] == [1500, 3500]
    assert rows[0]["suite"] == "turn" and rows[0]["run"] == "mh1"
    assert (turn_dir / "smart-turn-v3.2" / "container.log").is_file()
    assert "2 of 2 points called right" in out
    assert len(argvs) == 1


def test_turn_refuses_a_second_run_into_the_same_directory(tmp_path, monkeypatch, capsys):
    material = _material(tmp_path)
    monkeypatch.setattr(turn_suite, "run_container", lambda argv: pytest.fail("ran docker"))
    (tmp_path / "results" / "mh1" / "speech" / "turn" / "smart-turn-v3.2").mkdir(parents=True)
    weights = tmp_path / "weights"
    weights.mkdir()

    code = speech_bench.main(
        [
            "turn",
            "--results-root",
            str(tmp_path / "results"),
            "--run",
            "mh1",
            "--material-dir",
            str(material),
            "--weights-dir",
            str(weights),
            "--no-pins",
        ]
    )

    assert code != 0
    assert "already exists" in capsys.readouterr().out


def test_plan_prints_the_turn_argv_and_selftest_runs_the_turn_suite(capsys):
    assert speech_bench.main(["plan"]) == 0
    out = capsys.readouterr().out
    assert "[skip] turn_suite" not in out
    assert "[argv] turn smart-turn-v3.2:" in out

    assert speech_bench.main(["selftest"]) == 0
    assert "[ok] turn_suite" in capsys.readouterr().out


def test_turn_suite_validate_inputs_finds_no_problem():
    assert turn_suite.validate_inputs(_REPO_ROOT) == []
