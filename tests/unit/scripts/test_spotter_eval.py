"""Tests for the server spotter's ambient evaluation, `scripts/wake/spotter_eval.py`.

Every WAV here is written by `wave` into pytest's temporary directory, each holding one constant
sample value, and the spotter is scripted by that value: `score_window` returns the value's
positive score and `score_series` its ambient series. So no recording and no model is read, and
a test can tell from the values which audio reached the spotter. The model manifest is the
microWakeWord format over a placeholder model file, which `read_manifest` hashes like a real one.
"""

import hashlib
import json
import math
import wave
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from backend.realtime.budget import PlacementConfig
from scripts.wake import manifest as wake_manifest
from scripts.wake import spotter_eval

RATE = 16000
MODEL_ID = "mist-wake-test"
COOLDOWN_STEPS = 2
RTF = 0.25
"""The scripted spotter's cost: seconds of the test clock per second of audio it scores."""

# Tune positives, by sample value, and their scripted `score_window`.
POSITIVE_SCORES = {2: 0.97, 3: 0.99, 4: 0.96}
POSITIVE_FILES = ("p/tune-a.wav", "p/tune-b.wav", "p/tune-c.wav")
# Two ambient files of 18 s each, 0.01 h in all, by sample value, and their scripted series.
AMBIENT_SERIES = {
    101: [0.1, 0.6, 0.6, 0.6, 0.2, 0.9],
    102: [0.3, 0.05],
}
AMBIENT_SECONDS = 18
CUTOFFS = "0.95,0.2,0.5,0.96"
# With a cooldown of 2 steps, hits per cutoff: 0.2 gives steps 1, 3, 5 of the first file and
# step 0 of the second; 0.5 gives steps 1, 3, 5 of the first; 0.95 and 0.96 none. Over 0.01 h:
EXPECTED_CUTOFFS = [0.2, 0.5, 0.95, 0.96]
EXPECTED_RATES = [400.0, 300.0, 0.0, 0.0]
EVAL_KEYS = ("cutoffs", "false_accepts_per_hour", "positive_scores")


# -- Audio and manifests ----------------------------------------------------------------------


def write_wav(
    path: Path,
    value: int,
    seconds: float,
    *,
    rate: int = RATE,
    channels: int = 1,
    width: int = 2,
) -> None:
    """A WAV whose every sample is `value`, which is how the scripted spotter tells it apart."""
    count = int(rate * seconds)
    if width == 1:
        data = np.full(count * channels, 128, dtype=np.uint8).tobytes()
    else:
        data = np.full(count * channels, value, dtype="<i2").tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(data)


def entry(
    file: str, value: int, label: str, split: str, *, phrase: str = "hey mist"
) -> tuple[dict[str, object], int]:
    item: dict[str, object] = {
        "file": file,
        "label": label,
        "split": split,
        "phrase": phrase,
        "condition": "headset",
        "duration_ms": 1000,
        "session": f"{split}-{label}",
    }
    return item, value


DEFAULT_ENTRIES = (
    entry("p/train.wav", 1, "positive", "train"),
    entry("p/tune-a.wav", 2, "positive", "tune"),
    entry("p/tune-b.wav", 3, "positive", "tune"),
    entry("p/tune-c.wav", 4, "positive", "tune"),
    entry("h/tune.wav", 5, "homophone", "tune", phrase="hey missed"),
    entry("p/test.wav", 6, "positive", "test"),
    entry("m/test.wav", 7, "media", "test", phrase=""),
)


def write_recordings(
    tmp_path: Path, entries: Sequence[tuple[Mapping[str, object], int]] = DEFAULT_ENTRIES
) -> Path:
    recordings = tmp_path / "recordings"
    for item, value in entries:
        write_wav(recordings / str(item["file"]), value, 1.0)
    path = tmp_path / "manifest.json"
    payload = {"root": str(recordings), "entries": [dict(item) for item, _ in entries]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_model(models_dir: Path, *, training: str) -> None:
    """A microWakeWord manifest over a placeholder model file, which `read_manifest` hashes."""
    directory = models_dir / "speech" / "wake" / MODEL_ID
    (directory / "device").mkdir(parents=True, exist_ok=True)
    payload = b"placeholder streaming model"
    (directory / "device" / "mist.tflite").write_bytes(payload)
    manifest = {
        "model_id": MODEL_ID,
        "names": ["mist"],
        "family": "microwakeword",
        "sample_rate": 16000,
        "feature_step_ms": 10,
        "window_ms": 1280,
        "sliding_window": 5,
        "warmup_ms": 1000,
        "model": {"file": "device/mist.tflite", "sha256": hashlib.sha256(payload).hexdigest()},
        "hit_lag_ms": 0,
        "calibration": {},
        "training": training,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def write_ambient(directory: Path, series: Mapping[int, Sequence[float]] = AMBIENT_SERIES) -> None:
    for value in series:
        write_wav(directory / f"meeting-{value}.wav", value, AMBIENT_SECONDS)
    # The lead's checksum file sits beside the WAVs and is not audio.
    (directory / "SHA256SUMS").write_text("not audio\n", encoding="utf-8")


# -- The scripted spotter ---------------------------------------------------------------------


class FakeClock:
    """A monotonic clock that moves only when the scripted spotter scores audio."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def sample_value(pcm: bytes) -> int:
    return int.from_bytes(pcm[:2], "little", signed=True)


@dataclass
class ScriptedSpotter:
    """Scores audio by its constant sample value, and records which audio it was given."""

    clock: FakeClock
    positives: Mapping[int, float] = field(default_factory=lambda: dict(POSITIVE_SCORES))
    series: Mapping[int, Sequence[float]] = field(default_factory=lambda: dict(AMBIENT_SERIES))
    cooldown_steps: int = COOLDOWN_STEPS
    windows: list[int] = field(default_factory=list)
    streams: list[int] = field(default_factory=list)
    placements: list[PlacementConfig] = field(default_factory=list)

    def score_window(self, pcm: bytes) -> float:
        value = sample_value(pcm)
        self.windows.append(value)
        return self.positives[value]

    def score_series(self, pcm: bytes) -> list[float]:
        value = sample_value(pcm)
        self.streams.append(value)
        self.clock.now += RTF * len(pcm) / 2 / RATE
        return list(self.series[value])


@dataclass
class Setup:
    manifest: Path
    models_dir: Path
    ambient: Path
    out: Path
    clock: FakeClock
    spotter: ScriptedSpotter

    def args(self, *extra: str, cutoffs: str | None = CUTOFFS) -> list[str]:
        argv = [
            "--manifest",
            str(self.manifest),
            "--model",
            MODEL_ID,
            "--models-dir",
            str(self.models_dir),
            "--ambient",
            str(self.ambient),
            "--out",
            str(self.out),
            "--min-hours",
            "0.005",
        ]
        if cutoffs is not None:
            argv += ["--cutoffs", cutoffs]
        return argv + list(extra)

    def run(self, *extra: str, cutoffs: str | None = CUTOFFS) -> int:
        def load(placement: PlacementConfig) -> ScriptedSpotter:
            self.spotter.placements.append(placement)
            return self.spotter

        return spotter_eval.main(
            self.args(*extra, cutoffs=cutoffs), load_spotter=load, environ={}, clock=self.clock
        )


def prepare(
    tmp_path: Path,
    *,
    entries: Sequence[tuple[Mapping[str, object], int]] = DEFAULT_ENTRIES,
    training: str | None = None,
    positives: Mapping[int, float] | None = None,
) -> Setup:
    manifest = write_recordings(tmp_path, entries)
    if training is None:
        training = wake_manifest.subset_sha256(wake_manifest.load_manifest(manifest), "train")
    models_dir = tmp_path / "models"
    write_model(models_dir, training=training)
    ambient = tmp_path / "ambient"
    write_ambient(ambient)
    clock = FakeClock()
    spotter = ScriptedSpotter(clock=clock)
    if positives is not None:
        spotter.positives = dict(positives)
    return Setup(manifest, models_dir, ambient, tmp_path / "out" / "eval.json", clock, spotter)


def converter_accepts(evaluation: object) -> None:
    """The converter's checks on an evaluation (`device_calibration.py` on the Voice PE branch).

    Exactly the three keys; finite numbers; cutoffs and positive scores from 0 to 1; no cutoff
    twice; one rate per cutoff, none negative; and no rate rising with the cutoff.
    """
    assert isinstance(evaluation, dict)
    assert sorted(evaluation) == sorted(EVAL_KEYS)
    for key in EVAL_KEYS:
        values = evaluation[key]
        assert isinstance(values, list) and values
        assert all(
            isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)
            for v in values
        )
    cutoffs = evaluation["cutoffs"]
    rates = evaluation["false_accepts_per_hour"]
    assert all(0.0 <= value <= 1.0 for value in cutoffs)
    assert all(0.0 <= value <= 1.0 for value in evaluation["positive_scores"])
    assert len(set(cutoffs)) == len(cutoffs)
    assert len(rates) == len(cutoffs)
    assert all(rate >= 0 for rate in rates)
    points = sorted(zip(cutoffs, rates, strict=True))
    assert all(high[1] <= low[1] for low, high in zip(points, points[1:], strict=False))


def refused(
    run: Setup, capsys: pytest.CaptureFixture[str], *extra: str, cutoffs: str | None = CUTOFFS
) -> str:
    """Run, assert the refusal (exit 2, nothing written, nothing printed), return stderr."""
    code = run.run(*extra, cutoffs=cutoffs)
    captured = capsys.readouterr()
    assert code == 2
    assert not run.out.exists()
    assert captured.out == ""
    assert captured.err.startswith("spotter_eval: ")
    return captured.err


# -- The evaluation ---------------------------------------------------------------------------


def test_the_evaluation_holds_exactly_the_three_keys_and_their_values(tmp_path: Path) -> None:
    run = prepare(tmp_path)

    assert run.run() == 0

    evaluation = json.loads(run.out.read_text(encoding="utf-8"))
    assert list(evaluation) == list(EVAL_KEYS)
    assert evaluation["cutoffs"] == EXPECTED_CUTOFFS
    assert evaluation["false_accepts_per_hour"] == pytest.approx(EXPECTED_RATES)
    assert evaluation["positive_scores"] == [0.97, 0.99, 0.96]


def test_the_evaluation_passes_the_converter_s_checks(tmp_path: Path) -> None:
    run = prepare(tmp_path)

    assert run.run() == 0

    converter_accepts(json.loads(run.out.read_text(encoding="utf-8")))


def test_each_ambient_file_is_one_series_and_only_tune_positives_are_windowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = prepare(tmp_path)
    read: list[Path] = []
    original = spotter_eval.read_pcm16

    def spy(path: Path) -> bytes:
        read.append(Path(path))
        return original(path)

    monkeypatch.setattr(spotter_eval, "read_pcm16", spy)

    assert run.run() == 0

    recordings = tmp_path / "recordings"
    assert sorted(run.spotter.windows) == [2, 3, 4]
    assert sorted(run.spotter.streams) == [101, 102]
    assert sorted(path.resolve() for path in read) == sorted(
        path.resolve()
        for path in [recordings / name for name in POSITIVE_FILES]
        + [run.ambient / "meeting-101.wav", run.ambient / "meeting-102.wav"]
    )
    for item, _ in DEFAULT_ENTRIES:
        if item["split"] in ("train", "test"):
            assert (recordings / str(item["file"])).resolve() not in {p.resolve() for p in read}


def test_the_default_cutoffs_run_from_0_01_to_0_99_by_0_01(tmp_path: Path) -> None:
    run = prepare(tmp_path)

    run.run(cutoffs=None)

    evaluation = json.loads(run.out.read_text(encoding="utf-8"))
    assert evaluation["cutoffs"] == [round(step / 100, 2) for step in range(1, 100)]
    assert len(evaluation["false_accepts_per_hour"]) == 99
    converter_accepts(evaluation)


def test_the_default_minimum_is_eight_hours(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = prepare(tmp_path)

    code = spotter_eval.main(
        [arg for arg in run.args() if arg not in ("--min-hours", "0.005")],
        load_spotter=lambda placement: run.spotter,
        environ={},
        clock=run.clock,
    )

    assert code == 2
    assert "8" in capsys.readouterr().err
    assert not run.out.exists()


def test_the_printed_lines_are_counts_rates_hours_and_timing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = prepare(tmp_path)

    assert run.run() == 0

    captured = capsys.readouterr()
    assert captured.out.splitlines() == [
        "ambient: 2 files, 0.010 hours",
        "scoring: 9.0 s wall, real-time factor 0.2500",
        "tune positives: 3",
        "cutoff 0.2: wakes 3/3, false wakes 4 (400.000 per hour)",
        "cutoff 0.5: wakes 3/3, false wakes 3 (300.000 per hour)",
        "cutoff 0.95: wakes 3/3, false wakes 0 (0.000 per hour)",
        "cutoff 0.96: wakes 3/3, false wakes 0 (0.000 per hour)",
        "bar: pass at 0.95",
    ]
    names = ["meeting-101", "meeting-102", "tune-a", "tune-b", "tune-c", "eval.json", "manifest"]
    for name in names:
        assert name not in captured.out
        assert name not in captured.err


def test_the_bar_passes_at_the_lowest_cutoff_meeting_both_bounds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = prepare(tmp_path)

    assert run.run() == 0

    assert capsys.readouterr().out.splitlines()[-1] == "bar: pass at 0.95"


def test_no_cutoff_passes_writes_the_evaluation_and_exits_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # At 0.95 one of three positives wakes; at 0.2 and 0.5 the ambient rate is too high.
    run = prepare(tmp_path, positives={2: 0.97, 3: 0.4, 4: 0.3})

    assert run.run() == 1

    lines = capsys.readouterr().out.splitlines()
    assert lines[-1] == "bar: no cutoff passes"
    assert "cutoff 0.95: wakes 1/3, false wakes 0 (0.000 per hour)" in lines
    evaluation = json.loads(run.out.read_text(encoding="utf-8"))
    assert evaluation["positive_scores"] == [0.97, 0.4, 0.3]
    converter_accepts(evaluation)


@pytest.mark.parametrize(
    ("positives", "required"), [(18, 14), (9, 7), (4, 4), (3, 3), (1, 1), (36, 28)]
)
def test_the_bar_needs_ceil_14n_over_18_wakes(positives: int, required: int) -> None:
    assert spotter_eval.required_wakes(positives) == required


def test_the_bar_takes_the_lowest_cutoff_with_enough_wakes_and_at_most_half_a_wake_an_hour() -> (
    None
):
    scores = [0.9] * 14 + [0.1] * 4
    # 0.1 has the wakes but not the rate; 0.3 both, at exactly 0.5 per hour; 0.95 too few wakes.
    assert spotter_eval.passing_cutoff([0.1, 0.3, 0.5, 0.95], [3.0, 0.5, 0.0, 0.0], scores) == 0.3
    assert spotter_eval.passing_cutoff([0.1, 0.3], [3.0, 0.51], scores) is None
    assert spotter_eval.passing_cutoff([0.95], [0.0], scores) is None
    assert spotter_eval.passing_cutoff([0.9], [0.0], [0.9] * 13 + [0.1] * 5) is None


# -- Refusals ---------------------------------------------------------------------------------


def test_a_model_trained_on_another_training_split_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = prepare(tmp_path, training="0" * 64)

    assert "training" in refused(run, capsys)
    assert run.spotter.placements == []


def test_a_manifest_with_no_tune_positive_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    entries = [item for item in DEFAULT_ENTRIES if item[0]["file"] not in POSITIVE_FILES]
    run = prepare(tmp_path, entries=entries)

    assert "no tune positive" in refused(run, capsys)
    assert run.spotter.windows == []


def test_an_ambient_directory_with_no_wav_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = prepare(tmp_path)
    for wav in run.ambient.glob("*.wav"):
        wav.unlink()

    assert "no .wav" in refused(run, capsys)


def test_a_missing_ambient_directory_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = prepare(tmp_path)
    run.ambient = tmp_path / "absent"

    assert "not a directory" in refused(run, capsys)


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"rate": 8000}, "8000 Hz"),
        ({"channels": 2}, "2 channels"),
        ({"width": 1}, "8-bit"),
    ],
)
def test_an_ambient_wav_not_16_khz_mono_pcm16_is_refused_naming_it(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    kwargs: Mapping[str, int],
    reason: str,
) -> None:
    run = prepare(tmp_path)
    write_wav(run.ambient / "odd-one.wav", 103, 1.0, **kwargs)

    err = refused(run, capsys)

    assert "odd-one.wav" in err
    assert reason in err
    assert run.spotter.streams == []


def test_an_ambient_file_that_is_not_a_wav_is_refused_naming_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = prepare(tmp_path)
    (run.ambient / "broken.wav").write_bytes(b"not a riff file")

    assert "broken.wav" in refused(run, capsys)


def test_ambient_audio_under_the_minimum_hours_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = prepare(tmp_path)

    err = refused(run, capsys, "--min-hours", "0.02")

    assert "0.010 hours" in err
    assert run.spotter.streams == []


@pytest.mark.parametrize(
    ("cutoffs", "reason"),
    [
        ("0.5,1.5", "from 0 to 1"),
        ("0.5,-0.1", "from 0 to 1"),
        ("0.5,0.2,0.5", "twice"),
        ("0.5,0.50", "twice"),
        ("0.5,nan", "from 0 to 1"),
        ("0.5,high", "numbers"),
    ],
)
def test_cutoffs_outside_0_to_1_or_listed_twice_are_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], cutoffs: str, reason: str
) -> None:
    run = prepare(tmp_path)

    assert reason in refused(run, capsys, cutoffs=cutoffs)


def test_cutoffs_at_0_and_1_are_accepted(tmp_path: Path) -> None:
    run = prepare(tmp_path)

    assert run.run(cutoffs="0,1") in (0, 1)

    assert json.loads(run.out.read_text(encoding="utf-8"))["cutoffs"] == [0.0, 1.0]


def test_an_existing_out_is_refused_and_kept(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = prepare(tmp_path)
    run.out.parent.mkdir(parents=True)
    run.out.write_text("an earlier evaluation\n", encoding="utf-8")

    code = run.run()

    captured = capsys.readouterr()
    assert code == 2
    assert "--out already exists" in captured.err
    assert str(run.out) not in captured.err
    assert run.out.read_text(encoding="utf-8") == "an earlier evaluation\n"
    assert run.spotter.placements == []


def test_rates_rising_with_the_cutoff_are_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    run = prepare(tmp_path)

    def rising(series: Sequence[float], threshold: float, cooldown_steps: int) -> list[int]:
        return list(range(int(threshold * 10)))

    monkeypatch.setattr(spotter_eval, "hit_steps", rising)

    assert "rise with the cutoff" in refused(run, capsys)


def test_the_hits_are_counted_with_the_model_s_cooldown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = prepare(tmp_path)
    seen: list[tuple[float, int]] = []
    original = spotter_eval.hit_steps

    def spy(series: Sequence[float], threshold: float, cooldown_steps: int) -> list[int]:
        seen.append((threshold, cooldown_steps))
        return original(series, threshold, cooldown_steps)

    monkeypatch.setattr(spotter_eval, "hit_steps", spy)

    assert run.run() == 0

    assert {steps for _, steps in seen} == {COOLDOWN_STEPS}
    assert sorted({threshold for threshold, _ in seen}) == EXPECTED_CUTOFFS


def test_a_spotter_that_cannot_load_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = prepare(tmp_path)

    def failing(placement: PlacementConfig) -> ScriptedSpotter:
        raise spotter_eval.SpeechEngineError("ai-edge-litert is not installed")

    code = spotter_eval.main(run.args(), load_spotter=failing, environ={}, clock=run.clock)

    assert code == 2
    assert "ai-edge-litert" in capsys.readouterr().err
    assert not run.out.exists()


# -- The real loader --------------------------------------------------------------------------


def test_the_real_loader_is_the_microwakeword_spotter_on_the_model_under_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = prepare(tmp_path)
    seen: list[tuple[PlacementConfig, object]] = []

    class Spotter:
        @staticmethod
        def load(placement: PlacementConfig, *, library_loader: object) -> ScriptedSpotter:
            seen.append((placement, library_loader))
            return run.spotter

    monkeypatch.setattr(spotter_eval, "MicroNameSpotter", Spotter)

    assert spotter_eval.main(run.args(), environ={}, clock=run.clock) == 0

    [(placement, loader)] = seen
    assert loader is spotter_eval.import_micro_runtime
    assert placement.kws_model == MODEL_ID
    assert placement.models_dir == str(run.models_dir)
