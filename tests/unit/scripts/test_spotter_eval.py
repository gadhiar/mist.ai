"""Tests for the server spotter's ambient evaluation, `scripts/wake/spotter_eval.py`.

Every WAV here is written by `wave` into pytest's temporary directory, each part of it holding
one constant sample value. A value is a file's key, plus `LOUD` on its speech parts, so
`EnergyVad` hears speech exactly there and a scripted engine can tell from any buffer which file
it came from. The spotter is scripted by key: `score_window` returns a positive's score,
`score_series` an ambient file's series, and a stream reports a hit at a frame of a file. The
gate's STT is scripted by key too, so no recording and no model is read. The model manifest is
the microWakeWord format over a placeholder model file, which `read_manifest` hashes like a real
one.
"""

import hashlib
import json
import math
import sys
import wave
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from backend.realtime.budget import PlacementConfig
from backend.realtime.speech import CancelToken, NameHit, TranscriptUpdate
from scripts.wake import eval_wake, spotter_eval
from scripts.wake import manifest as wake_manifest
from tests.fakes.speech import EnergyVad

RATE = 16000
FRAME_SAMPLES = 320
MODEL_ID = "mist-wake-test"
COOLDOWN_STEPS = 2
RTF = 0.25
"""The scripted spotter's cost: seconds of the test clock per second of audio it scores."""
DECODE_S = 0.5
"""The scripted STT's cost: seconds of the test clock per gate decode."""
LOUD = 5000
"""Added to a key on speech parts: `EnergyVad` scores 5000 and up as speech, 300 and under not."""
ADDRESSED = "mist, what's the time"
UNADDRESSED = "the mist rolled in"

# Tune positives, by key, and their scripted `score_window`.
POSITIVE_SCORES = {2: 0.97, 3: 0.99, 4: 0.96}
POSITIVE_FILES = ("p/tune-a.wav", "p/tune-b.wav", "p/tune-c.wav")
# Two ambient files of 18 s each, 0.01 h in all, by key: their parts as (value, seconds), and
# their scripted series.
AMBIENT_PARTS = {
    101: ((101, 4.0), (LOUD + 101, 2.0), (101, 12.0)),
    102: ((102, 8.0), (LOUD + 102, 2.0), (102, 8.0)),
}
AMBIENT_SERIES = {
    101: [0.1, 0.6, 0.6, 0.6, 0.2, 0.9],
    102: [0.3, 0.05],
}
AMBIENT_SECONDS = 18
# Stream hits by key, as (frame within the file, score). Each positive's hit is in its speech.
# Ambient file 101 has one in its speech (frames 200 to 299) at 0.955 and one at 0.957 at frame
# 600, in silence, which is reported and gated at 0.95 and under, and is below 0.96; file 102
# has one in its speech at 0.97.
STREAM_HITS: dict[int, tuple[tuple[int, float], ...]] = {
    2: ((30, 0.97),),
    3: ((30, 0.99),),
    4: ((30, 0.96),),
    101: ((250, 0.955), (600, 0.957)),
    102: ((450, 0.97),),
}
# The gate decode's text by key: every positive and file 101 addressed, file 102 not.
TEXTS = {2: ADDRESSED, 3: ADDRESSED, 4: ADDRESSED, 101: ADDRESSED, 102: UNADDRESSED}
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
    """A WAV whose every sample is `value`."""
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


def write_parts(path: Path, parts: Sequence[tuple[int, float]]) -> None:
    """A 16 kHz mono WAV of constant parts, each `(value, seconds)`."""
    data = np.concatenate(
        [np.full(int(RATE * seconds), value, dtype="<i2") for value, seconds in parts]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(data.tobytes())


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
    """Each recording is one second of speech holding its key."""
    recordings = tmp_path / "recordings"
    for item, value in entries:
        write_wav(recordings / str(item["file"]), LOUD + value, 1.0)
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


def write_ambient(directory: Path) -> None:
    for key, parts in AMBIENT_PARTS.items():
        write_parts(directory / f"meeting-{key}.wav", parts)
    # The lead's checksum file sits beside the WAVs and is not audio.
    (directory / "SHA256SUMS").write_text("not audio\n", encoding="utf-8")


# -- The scripted engines ---------------------------------------------------------------------


class FakeClock:
    """A monotonic clock that moves only when a scripted engine does costly work."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def key_of(pcm: bytes) -> int:
    """The file a buffer came from: its first sample, less `LOUD` on a speech part."""
    return int.from_bytes(pcm[:2], "little", signed=True) % LOUD


@dataclass
class KeyedStream:
    """A spotter stream that reports `STREAM_HITS` of the file it is fed, by frame in the file.

    Every frame of a recording is pushed once and in order from its first, so the first
    position a stream sees is the file's frame 0, and its first frame's key names the file.
    """

    hits: Mapping[int, Sequence[tuple[int, float]]]
    threshold: float
    events: list[tuple[object, ...]]
    key: int | None = None
    base: int = 0
    pushed: list[int] = field(default_factory=list)

    def _frame(self, pcm: bytes, pos: int) -> int:
        if self.key is None:
            self.key = key_of(pcm)
            self.base = pos
        return pos - self.base

    def push(self, pcm: bytes, pos: int) -> list[NameHit]:
        frame = self._frame(pcm, pos)
        self.pushed.append(frame)
        found = [
            NameHit(end_pos=pos, keyword="mist", score=score)
            for at, score in self.hits.get(self.key or 0, ())
            if at == frame and score >= self.threshold
        ]
        for item in found:
            self.events.append(("hit", self.key, self.threshold, item.score))
        return found

    def reset(self, *, warm_up: bool = True) -> None:
        raise AssertionError("the harness never resets a stream")


@dataclass
class ScriptedSpotter:
    """Scores and streams audio by its key, and records which audio it was given."""

    clock: FakeClock
    events: list[tuple[object, ...]]
    positives: Mapping[int, float] = field(default_factory=lambda: dict(POSITIVE_SCORES))
    series: Mapping[int, Sequence[float]] = field(default_factory=lambda: dict(AMBIENT_SERIES))
    hits: Mapping[int, Sequence[tuple[int, float]]] = field(
        default_factory=lambda: dict(STREAM_HITS)
    )
    cooldown_steps: int = COOLDOWN_STEPS
    window_ms: int = 1280
    windows: list[int] = field(default_factory=list)
    streams: list[int] = field(default_factory=list)
    opened: list[KeyedStream] = field(default_factory=list)
    placements: list[PlacementConfig] = field(default_factory=list)

    def score_window(self, pcm: bytes) -> float:
        key = key_of(pcm)
        self.windows.append(key)
        return self.positives[key]

    def score_series(self, pcm: bytes) -> list[float]:
        key = key_of(pcm)
        self.streams.append(key)
        self.clock.now += RTF * len(pcm) / 2 / RATE
        return list(self.series[key])

    def open_stream(self, threshold: float) -> KeyedStream:
        stream = KeyedStream(self.hits, threshold, self.events)
        self.opened.append(stream)
        return stream

    def gated(self, keys: Sequence[int]) -> list[float]:
        """The thresholds of the streams opened over the files with these keys, in order."""
        return [stream.threshold for stream in self.opened if stream.key in keys]


@dataclass
class KeyedDecode:
    """One gate decode: the text scripted for the key its audio starts with."""

    owner: "KeyedTranscriber"
    audio: list[np.ndarray] = field(default_factory=list)

    def push(self, pcm: np.ndarray) -> None:
        self.audio.append(pcm)

    def decode_partial(self, token: CancelToken) -> TranscriptUpdate:
        samples = np.concatenate(self.audio)
        key = int(round(float(samples[0]) * 32768)) % LOUD
        self.owner.decoded.append(key)
        self.owner.events.append(("decode", key))
        self.owner.clock.now += DECODE_S
        if key in self.owner.failing:
            raise spotter_eval.SpeechEngineError("scripted decode failure")
        text = self.owner.texts[key]
        return TranscriptUpdate(text, len(text), False, len(samples) * 1000 // RATE)

    def close(self) -> None:
        pass


@dataclass
class KeyedTranscriber:
    """The gate's STT, scripted by key, recording each decode's key."""

    clock: FakeClock
    events: list[tuple[object, ...]]
    texts: Mapping[int, str] = field(default_factory=lambda: dict(TEXTS))
    failing: frozenset[int] = frozenset()
    model_id: str = "keyed-stt"
    decoded: list[int] = field(default_factory=list)

    def open_stream(self) -> KeyedDecode:
        return KeyedDecode(self)


@dataclass
class Setup:
    manifest: Path
    models_dir: Path
    ambient: Path
    out: Path
    clock: FakeClock
    spotter: ScriptedSpotter
    transcriber: KeyedTranscriber
    events: list[tuple[object, ...]]
    vad: EnergyVad = field(default_factory=EnergyVad)

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

    def engines(self, placement: PlacementConfig) -> eval_wake.Engines:
        self.spotter.placements.append(placement)
        return eval_wake.Engines(
            spotter=self.spotter,  # type: ignore[arg-type]
            vad=self.vad,
            transcriber=self.transcriber,  # type: ignore[arg-type]
        )

    def run(self, *extra: str, cutoffs: str | None = CUTOFFS) -> int:
        return spotter_eval.main(
            self.args(*extra, cutoffs=cutoffs),
            load_engines=self.engines,
            environ={},
            clock=self.clock,
        )


def prepare(
    tmp_path: Path,
    *,
    entries: Sequence[tuple[Mapping[str, object], int]] = DEFAULT_ENTRIES,
    training: str | None = None,
    positives: Mapping[int, float] | None = None,
    texts: Mapping[int, str] | None = None,
    failing: frozenset[int] = frozenset(),
) -> Setup:
    manifest = write_recordings(tmp_path, entries)
    if training is None:
        training = wake_manifest.subset_sha256(wake_manifest.load_manifest(manifest), "train")
    models_dir = tmp_path / "models"
    write_model(models_dir, training=training)
    ambient = tmp_path / "ambient"
    write_ambient(ambient)
    clock = FakeClock()
    events: list[tuple[object, ...]] = []
    spotter = ScriptedSpotter(clock=clock, events=events)
    if positives is not None:
        # A positive's stream hit scores what its window does.
        spotter.positives = dict(positives)
        spotter.hits = {**STREAM_HITS, **{key: ((30, score),) for key, score in positives.items()}}
    transcriber = KeyedTranscriber(clock=clock, events=events, failing=failing)
    if texts is not None:
        transcriber.texts = {**TEXTS, **texts}
    return Setup(
        manifest,
        models_dir,
        ambient,
        tmp_path / "out" / "eval.json",
        clock,
        spotter,
        transcriber,
        events,
    )


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


def cutoff_lines(out: str) -> dict[float, str]:
    """The printed per-cutoff lines, by cutoff."""
    return {
        float(line.split(":")[0].removeprefix("cutoff ")): line
        for line in out.splitlines()
        if line.startswith("cutoff ")
    }


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


def test_the_evaluation_stays_spotter_only_whatever_the_gate_accepts(tmp_path: Path) -> None:
    # The gate accepts file 102's hit here, and rejects every positive's: neither moves `--out`.
    run = prepare(tmp_path, texts={2: UNADDRESSED, 3: UNADDRESSED, 102: ADDRESSED})

    assert run.run(cutoffs="0.95") == 1

    evaluation = json.loads(run.out.read_text(encoding="utf-8"))
    assert evaluation == {
        "cutoffs": [0.95],
        "false_accepts_per_hour": [0.0],
        "positive_scores": [0.97, 0.99, 0.96],
    }
    converter_accepts(evaluation)


def test_each_ambient_file_is_one_series_and_only_tune_positives_are_windowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = prepare(tmp_path)
    read: list[Path] = []

    def spy(module: object) -> None:
        original = module.read_pcm16  # type: ignore[attr-defined]

        def reading(path: Path) -> bytes:
            read.append(Path(path))
            return original(path)

        monkeypatch.setattr(module, "read_pcm16", reading)

    # Audio reaches the spotter here, and the VAD and the gate through eval_wake.
    spy(spotter_eval)
    spy(eval_wake)

    assert run.run() == 0

    recordings = tmp_path / "recordings"
    assert sorted(run.spotter.windows) == [2, 3, 4]
    assert sorted(run.spotter.streams) == [101, 102]
    allowed = {
        path.resolve()
        for path in [recordings / name for name in POSITIVE_FILES]
        + [run.ambient / "meeting-101.wav", run.ambient / "meeting-102.wav"]
    }
    assert {path.resolve() for path in read} == allowed
    assert {key for stream in run.spotter.opened for key in [stream.key]} == {2, 3, 4, 101, 102}
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
        load_engines=run.engines,
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
        "gate, tune positives: 4 cutoffs, 3 decodes, 0 failed, 1.5 s wall",
        "ambient VAD: 2 files, 0.0 s wall",
        "gate at 0.96: 0 false wakes, 1 decodes, 0 failed, 0.5 s wall",
        "tune positives: 3",
        "cutoff 0.2: wakes 3/3, false wakes 4 (400.000 per hour), "
        "gated wakes 3/3, gated false wakes not run",
        "cutoff 0.5: wakes 3/3, false wakes 3 (300.000 per hour), "
        "gated wakes 3/3, gated false wakes not run",
        "cutoff 0.95: wakes 3/3, false wakes 0 (0.000 per hour), "
        "gated wakes 3/3, gated false wakes not run",
        "cutoff 0.96: wakes 3/3, false wakes 0 (0.000 per hour), "
        "gated wakes 3/3, gated false wakes 0 (0.000 per hour)",
        "bar: pass at 0.96",
    ]
    names = ["meeting-101", "meeting-102", "tune-a", "tune-b", "tune-c", "eval.json", "manifest"]
    for name in names + ["what's", "rolled"]:
        assert name not in captured.out
        assert name not in captured.err


def test_the_progress_lines_are_flushed_as_each_gated_pass_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = prepare(tmp_path)

    class Recorder:
        def write(self, text: str) -> int:
            run.events.append(("write", text))
            return len(text)

        def flush(self) -> None:
            run.events.append(("flush",))

    monkeypatch.setattr(sys, "stdout", Recorder())

    assert run.run() == 0

    def flushed_before_next_work(prefix: str) -> None:
        (index,) = (
            number
            for number, event in enumerate(run.events)
            if event[0] == "write" and str(event[1]).startswith(prefix)
        )
        later = run.events[index + 1 :]
        flush = later.index(("flush",))
        assert all(event[0] in ("write", "flush") for event in later[:flush])

    flushed_before_next_work("gate, tune positives:")
    flushed_before_next_work("gate at 0.96:")
    # The tune gate line is out before the ambient gate decodes anything.
    tune_line = next(
        number
        for number, event in enumerate(run.events)
        if event[0] == "write" and str(event[1]).startswith("gate, tune")
    )
    ambient_decodes = [
        number
        for number, event in enumerate(run.events)
        if event[0] == "decode" and event[1] in AMBIENT_PARTS
    ]
    assert ambient_decodes and min(ambient_decodes) > tune_line


def test_gated_wakes_count_a_positive_only_when_one_of_its_hits_is_accepted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Positive 4's hit is decoded as unaddressed: it wakes the spotter, not her.
    run = prepare(tmp_path, texts={4: UNADDRESSED})

    assert run.run() == 1

    out = capsys.readouterr().out
    lines = cutoff_lines(out)
    assert "wakes 3/3," in lines[0.95] and "gated wakes 2/3," in lines[0.95]
    assert "gated wakes 2/3," in lines[0.2]
    assert "wakes 3/3," in lines[0.96] and "gated wakes 2/3," in lines[0.96]
    # No cutoff has the ceil(14n/18) = 3 gated wakes, so the ambient set is never gated.
    assert all(line.endswith("gated false wakes not run") for line in lines.values())
    assert run.spotter.gated([101, 102]) == []
    assert out.splitlines()[-1] == "bar: no cutoff passes"


def test_a_gated_false_wake_is_a_reported_hit_whose_decode_the_gate_accepts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = prepare(tmp_path)

    assert run.run(cutoffs="0.95") == 1

    out = capsys.readouterr().out
    # At 0.95 file 101's speech hit is addressed and counts, and so does its 0.957 hit at frame
    # 600, in silence, from a span of its own; file 102's is decoded "the mist rolled in" and
    # does not.
    assert cutoff_lines(out)[0.95].endswith("gated false wakes 2 (200.000 per hour)")
    assert "gate at 0.95: 2 false wakes, 3 decodes, 0 failed, 1.5 s wall" in out.splitlines()
    (stream_101,) = (stream for stream in run.spotter.opened if stream.key == 101)
    assert {250, 600} <= set(stream_101.pushed)
    assert ("hit", 101, 0.95, 0.957) in run.events
    assert sorted(key for key in run.transcriber.decoded if key in AMBIENT_PARTS) == [
        101,
        101,
        102,
    ]
    assert out.splitlines()[-1] == "bar: no cutoff passes"


def test_every_ambient_frame_is_pushed_once_and_in_order_at_the_gated_cutoff(
    tmp_path: Path,
) -> None:
    run = prepare(tmp_path)

    assert run.run() == 0

    for key in AMBIENT_PARTS:
        (stream,) = (stream for stream in run.spotter.opened if stream.key == key)
        assert stream.pushed == list(range(AMBIENT_SECONDS * RATE // FRAME_SAMPLES))


def test_only_the_highest_candidate_is_gated(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # 0.98 wakes only positive 3, so it is not a candidate; 0.96 is the highest that is.
    run = prepare(tmp_path)

    assert run.run(cutoffs="0.95,0.2,0.5,0.96,0.98") == 0

    lines = cutoff_lines(capsys.readouterr().out)
    assert run.spotter.gated([101, 102]) == [0.96, 0.96]
    assert "gated wakes 1/3," in lines[0.98]
    for cutoff in (0.2, 0.5, 0.95, 0.98):
        assert lines[cutoff].endswith("gated false wakes not run")
    assert lines[0.96].endswith("gated false wakes 0 (0.000 per hour)")


def test_one_listed_cutoff_that_is_a_candidate_is_gated_alone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # File 101's speech hit is unaddressed here, so 0.5's gated rate is 0.
    run = prepare(tmp_path, texts={101: UNADDRESSED})

    assert run.run(cutoffs="0.5") == 0

    assert run.spotter.gated([101, 102]) == [0.5, 0.5]
    assert capsys.readouterr().out.splitlines()[-1] == "bar: pass at 0.5"


def test_the_bar_fails_when_the_highest_candidate_s_gated_rate_is_over_half_an_hour(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # File 102's hit at 0.97 is addressed now, one gated false wake in 0.01 h at 0.96.
    run = prepare(tmp_path, texts={102: ADDRESSED})

    assert run.run() == 1

    out = capsys.readouterr().out
    assert cutoff_lines(out)[0.96].endswith("gated false wakes 1 (100.000 per hour)")
    assert out.splitlines()[-1] == "bar: no cutoff passes"
    assert run.spotter.gated([101, 102]) == [0.96, 0.96]
    converter_accepts(json.loads(run.out.read_text(encoding="utf-8")))


def test_a_failed_gate_decode_rejects_the_hit_and_is_counted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = prepare(tmp_path, texts={102: ADDRESSED}, failing=frozenset({102}))

    assert run.run() == 0

    out = capsys.readouterr().out.splitlines()
    assert "gate at 0.96: 0 false wakes, 1 decodes, 1 failed, 0.5 s wall" in out
    assert out[-1] == "bar: pass at 0.96"


def test_no_cutoff_passes_writes_the_evaluation_and_exits_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # At 0.95 one of three positives wakes; at 0.2 and 0.5 the ambient rate is too high.
    run = prepare(tmp_path, positives={2: 0.97, 3: 0.4, 4: 0.3})

    assert run.run() == 1

    lines = capsys.readouterr().out.splitlines()
    assert lines[-1] == "bar: no cutoff passes"
    assert any(line.startswith("cutoff 0.95: wakes 1/3, false wakes 0 (0.000") for line in lines)
    evaluation = json.loads(run.out.read_text(encoding="utf-8"))
    assert evaluation["positive_scores"] == [0.97, 0.4, 0.3]
    converter_accepts(evaluation)


@pytest.mark.parametrize(
    ("positives", "required"), [(18, 14), (9, 7), (4, 4), (3, 3), (1, 1), (36, 28)]
)
def test_the_bar_needs_ceil_14n_over_18_wakes(positives: int, required: int) -> None:
    assert spotter_eval.required_wakes(positives) == required


@pytest.mark.parametrize(
    ("wakes", "rates", "passing"),
    [
        # 0.5 is the highest with 14 of 18 gated wakes; 0.95 has 13. Only 0.5 was gated.
        ((18, 14, 14, 13), (None, None, 0.5, None), 0.5),
        ((18, 14, 14, 13), (None, None, 0.51, None), None),
        # A candidate below the highest that would pass does not make the bar pass.
        ((18, 14, 14, 13), (0.0, 0.0, 1.0, None), None),
        # Gated wakes need not fall as the cutoff rises: 0.95 is a candidate above others.
        ((18, 13, 12, 14), (None, None, None, 0.0), 0.95),
        ((13, 13, 12, 0), (0.0, 0.0, 0.0, 0.0), None),
        # The highest candidate not gated is no pass.
        ((18, 14, 14, 14), (0.0, 0.0, 0.0, None), None),
    ],
)
def test_the_bar_names_the_highest_candidate_when_its_gated_rate_is_at_most_half_an_hour(
    wakes: tuple[int, ...], rates: tuple[float | None, ...], passing: float | None
) -> None:
    cutoffs = [0.1, 0.3, 0.5, 0.95]
    assert spotter_eval.passing_cutoff(cutoffs, wakes, rates, 18) == passing


def test_the_gated_cutoff_is_the_highest_with_enough_gated_wakes() -> None:
    cutoffs = [0.1, 0.3, 0.5, 0.95]
    assert spotter_eval.gated_cutoff(cutoffs, [18, 14, 14, 13], 18) == 0.5
    assert spotter_eval.gated_cutoff(cutoffs, [18, 13, 12, 14], 18) == 0.95
    assert spotter_eval.gated_cutoff(cutoffs, [13, 13, 12, 0], 18) is None


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


def test_engines_that_cannot_load_are_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = prepare(tmp_path)

    def failing(placement: PlacementConfig) -> eval_wake.Engines:
        raise spotter_eval.SpeechEngineError("ai-edge-litert is not installed")

    code = spotter_eval.main(run.args(), load_engines=failing, environ={}, clock=run.clock)

    assert code == 2
    assert "ai-edge-litert" in capsys.readouterr().err
    assert not run.out.exists()


# -- The real engines -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("factory", "reason"),
    [
        (None, "--stt-factory is required"),
        ("scripts.wake.eval_wake", "<module>:<callable>"),
        ("scripts.wake.no_such_module:factory", "could not be loaded"),
        ("scripts.wake.eval_wake:no_such_factory", "could not be loaded"),
    ],
)
def test_a_missing_or_bad_stt_factory_is_refused_and_nothing_is_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], factory: str | None, reason: str
) -> None:
    run = prepare(tmp_path)
    extra = [] if factory is None else ["--stt-factory", factory]

    code = spotter_eval.main(run.args(*extra), environ={}, clock=run.clock)

    captured = capsys.readouterr()
    assert code == 2
    assert reason in captured.err
    assert captured.err.startswith("spotter_eval: ")
    assert captured.out == ""
    assert not run.out.exists()
    assert spotter_eval.RefusedError is eval_wake.RefusedError


def test_the_real_engines_are_eval_wake_s_on_the_model_under_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = prepare(tmp_path)
    seen: list[tuple[str, PlacementConfig]] = []

    class Spotter:
        @staticmethod
        def load(placement: PlacementConfig, *, library_loader: object) -> ScriptedSpotter:
            assert library_loader is eval_wake.import_micro_runtime
            seen.append(("spotter", placement))
            return run.spotter

    class Vad:
        @staticmethod
        def load(placement: PlacementConfig, **kwargs: object) -> EnergyVad:
            seen.append(("vad", placement))
            return run.vad

    def stt(placement: PlacementConfig) -> KeyedTranscriber:
        seen.append(("stt", placement))
        return run.transcriber

    monkeypatch.setattr(eval_wake, "MicroNameSpotter", Spotter)
    monkeypatch.setattr(eval_wake, "SileroVad", Vad)
    monkeypatch.setattr(eval_wake, "parakeet_cpu", stt)

    code = spotter_eval.main(
        run.args("--stt-factory", "scripts.wake.eval_wake:parakeet_cpu"),
        environ={},
        clock=run.clock,
    )

    assert code == 0
    assert sorted(name for name, _ in seen) == ["spotter", "stt", "vad"]
    for _, placement in seen:
        assert placement.kws_model == MODEL_ID
        assert placement.models_dir == str(run.models_dir)
