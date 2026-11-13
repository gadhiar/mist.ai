"""Tests for the wake model evaluation harness: the recording manifest and `eval_wake`.

Every WAV here is written by `wave` into pytest's temporary directory, and the engines are the
scripted fakes (`ScriptedNameSpotter`, `ScriptedTranscriber`, `EnergyVad`), so no recording and
no model is read. The model manifest is T34's format over three placeholder graph files, which
`read_manifest` hashes like real ones.

Scripted hits are addressed by uplink position. The harness numbers the frames of a split's
entries in manifest order, one position per 20 ms frame, so `offsets` below turns an entry and a
frame within it into the position a hit is scripted at. Transcripts are scripted in the order
the gate decodes run: entries in manifest order, one decode per distinct span.
"""

import hashlib
import json
import logging
import wave
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from backend.realtime.engines.kws import read_manifest
from backend.realtime.speech import NameHit
from scripts.wake import eval_wake
from scripts.wake import manifest as wake_manifest
from tests.fakes.speech import EnergyVad, ScriptedNameSpotter, ScriptedTranscriber

RATE = 16000
FRAME_MS = 20
MODEL_ID = "mist-wake-test"
SECRET_WORDS = ("quokka", "zanzibar")
ADDRESSED = "hey mist quokka"
UNADDRESSED = "zanzibar quokka"
REFERENCE_SAMPLE = 0x1234
REFERENCE_MARKER = REFERENCE_SAMPLE.to_bytes(2, "little") * 2
SWEEP = "0.3,0.4,0.5,0.6,0.7"


# -- Audio and manifests ----------------------------------------------------------------------


def write_wav(
    path: Path,
    *,
    tone_ms: int,
    silence_ms: int = 0,
    frequency: float = 440.0,
    rate: int = RATE,
    channels: int = 1,
    width: int = 2,
    constant: int | None = None,
) -> None:
    """A tone (or a constant sample) followed by silence, loud enough for `EnergyVad`."""
    tone = rate * tone_ms // 1000
    if constant is not None:
        samples = np.full(tone, constant, dtype="<i2")
    else:
        t = np.arange(tone) / rate
        samples = (0.3 * 32767 * np.sin(2 * np.pi * frequency * t)).astype("<i2")
    samples = np.concatenate([samples, np.zeros(rate * silence_ms // 1000, dtype="<i2")])
    if width == 1:
        data = ((samples.astype(np.int32) >> 8) + 128).astype(np.uint8).tobytes()
    elif channels == 2:
        data = np.stack([samples, samples], axis=1).astype("<i2").tobytes()
    else:
        data = samples.tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(data)


def entry(
    file: str,
    label: str = "positive",
    split: str = "tune",
    *,
    phrase: str = "hey mist",
    condition: str = "headset",
    session: str | None = None,
    duration_ms: int = 1000,
    reference_file: str | None = None,
) -> dict[str, object]:
    item: dict[str, object] = {
        "file": file,
        "label": label,
        "split": split,
        "phrase": phrase,
        "condition": condition,
        "duration_ms": duration_ms,
        "session": session if session is not None else f"{split}-{label}",
    }
    if reference_file is not None:
        item["reference_file"] = reference_file
    return item


def write_manifest(
    tmp_path: Path, entries: Sequence[Mapping[str, object]], *, root: str | None = None
) -> Path:
    """Write the manifest, and a distinct tone WAV for every entry whose file is not there yet."""
    recordings = tmp_path / "recordings"
    for index, item in enumerate(entries):
        path = recordings / str(item["file"])
        if not path.exists() and ".." not in str(item["file"]):
            write_wav(path, tone_ms=int(item["duration_ms"]), frequency=200.0 + 37.0 * index)
    manifest = tmp_path / "manifest.json"
    payload = {"root": root if root is not None else str(recordings), "entries": list(entries)}
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest


def refused(tmp_path: Path, entries: Sequence[Mapping[str, object]], **kwargs: object) -> str:
    path = write_manifest(tmp_path, entries, **kwargs)
    with pytest.raises(wake_manifest.ManifestError) as caught:
        wake_manifest.load_manifest(path)
    return str(caught.value)


def write_model(
    models_dir: Path,
    *,
    training: str,
    hit_lag_ms: int = 0,
    calibration: Mapping[str, object] | None = None,
) -> Path:
    """T34's manifest over three placeholder graphs, which `read_manifest` hashes."""
    directory = models_dir / "speech" / "wake" / MODEL_ID
    directory.mkdir(parents=True, exist_ok=True)
    specs: list[tuple[str, dict[str, object]]] = [
        (
            "melspectrogram",
            {
                "input": {"new_samples": 1280, "left_context_samples": 480},
                "frames_per_hop": 8,
                "bins": 32,
                "output_transform": {"scale": 0.1, "offset": 2.0},
            },
        ),
        ("embedding", {"input": {"window_frames": 76, "stride_frames": 8}, "dims": 96}),
        ("classifier", {"input": {"window_embeddings": "from_model"}}),
    ]
    graphs = []
    for role, spec in specs:
        payload = f"placeholder {role}".encode()
        (directory / f"{role}.onnx").write_bytes(payload)
        graphs.append(
            {
                "role": role,
                "file": f"{role}.onnx",
                "sha256": hashlib.sha256(payload).hexdigest(),
                **spec,
            }
        )
    manifest = {
        "model_id": MODEL_ID,
        "names": ["mist"],
        "family": "openwakeword",
        "sample_rate": 16000,
        "hop_ms": 80,
        "window_ms": 1280,
        "threshold": 0.5,
        "warmup_predictions": 5,
        "graphs": graphs,
        "hit_lag_ms": hit_lag_ms,
        "calibration": dict(calibration or {}),
        "training": training,
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


TUNED_CALIBRATION = {
    "server": {
        "false_accepts": [[0.5, 1.0], [0.6, 0.5]],
        "positive_quantile": [[0.5, 0.1], [0.6, 0.4]],
    }
}


# -- Engines ----------------------------------------------------------------------------------


class FakeClock:
    """A monotonic clock that moves only when a test moves it."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, ms: float) -> None:
        self.now += ms / 1000


@dataclass
class Rig:
    """The fakes one harness run is given, kept so a test can read what reached them."""

    spotter: ScriptedNameSpotter
    transcriber: ScriptedTranscriber
    clock: FakeClock = field(default_factory=FakeClock)
    vad: EnergyVad = field(default_factory=EnergyVad)

    def run(self, argv: Sequence[str]) -> int:
        return eval_wake.main(
            list(argv),
            load_engines=lambda placement: eval_wake.Engines(
                spotter=self.spotter, vad=self.vad, transcriber=self.transcriber
            ),
            environ={},
            clock=self.clock,
        )


def rig(
    hits: Sequence[NameHit],
    script: Sequence[str],
    *,
    decode_ms: Sequence[float] = (),
) -> Rig:
    """Scripted engines; each gate decode takes the next of `decode_ms` (else 50 ms)."""
    clock = FakeClock()
    pending = list(decode_ms)

    def decoding() -> None:
        clock.advance(pending.pop(0) if pending else 50.0)

    spotter = ScriptedNameSpotter(hits, markers={REFERENCE_MARKER: 0.9})
    transcriber = ScriptedTranscriber(script, ms_per_word=1, on_decode=decoding)
    return Rig(spotter=spotter, transcriber=transcriber, clock=clock)


def offsets(entries: Sequence[Mapping[str, object]], split: str) -> dict[str, int]:
    """Each entry's first uplink position within its split, by file id."""
    found: dict[str, int] = {}
    position = 0
    for item in entries:
        if item["split"] == split:
            found[str(item["file"])] = position
            position += int(item["duration_ms"]) // FRAME_MS
    return found


def hit(at: Mapping[str, int], file: str, frame: int, score: float) -> NameHit:
    return NameHit(end_pos=at[file] + frame, keyword="mist", score=score)


# -- The manifest: shape and audio ------------------------------------------------------------


def test_a_valid_manifest_loads_every_entry(tmp_path: Path) -> None:
    path = write_manifest(
        tmp_path,
        [
            entry("p/a.wav", split="train", session="s1"),
            entry("h/b.wav", "homophone", "tune", phrase="I missed it", session="s2"),
            entry("m/c.wav", "media", "test", phrase="", session="s3"),
        ],
    )

    loaded = wake_manifest.load_manifest(path)

    assert [item.file for item in loaded.entries] == ["p/a.wav", "h/b.wav", "m/c.wav"]
    assert [item.file for item in loaded.split("tune")] == ["h/b.wav"]
    assert (
        loaded.entries[0].sha256
        == hashlib.sha256((tmp_path / "recordings" / "p" / "a.wav").read_bytes()).hexdigest()
    )


def test_a_relative_root_is_refused(tmp_path: Path) -> None:
    message = refused(tmp_path, [entry("p/a.wav")], root="recordings")
    assert "root" in message and "absolute" in message


def test_a_file_outside_the_root_is_refused_naming_the_entry(tmp_path: Path) -> None:
    write_wav(tmp_path / "outside.wav", tone_ms=1000)
    message = refused(tmp_path, [entry("p/a.wav"), entry("../outside.wav")])
    assert "entry 1" in message and "../outside.wav" in message and "outside" in message


def test_a_missing_file_is_refused_naming_the_entry(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, [entry("p/a.wav"), entry("p/gone.wav")])
    (tmp_path / "recordings" / "p" / "gone.wav").unlink()
    with pytest.raises(wake_manifest.ManifestError, match=r"entry 1 \(p/gone\.wav\).*not found"):
        wake_manifest.load_manifest(path)


@pytest.mark.parametrize(
    ("key", "value"),
    [("label", "speech"), ("split", "eval"), ("label", "Positive"), ("split", "validation")],
)
def test_a_label_or_split_outside_its_set_is_refused(tmp_path: Path, key: str, value: str) -> None:
    item = entry("p/b.wav")
    item[key] = value
    message = refused(tmp_path, [entry("p/a.wav"), item])
    assert "entry 1 (p/b.wav)" in message and key in message and repr(value) in message


@pytest.mark.parametrize("session", [None, ""])
def test_every_entry_carries_a_session(tmp_path: Path, session: str | None) -> None:
    item = entry("p/b.wav")
    if session is None:
        del item["session"]
    else:
        item["session"] = session
    message = refused(tmp_path, [entry("p/a.wav"), item])
    assert "entry 1 (p/b.wav)" in message and "session" in message


@pytest.mark.parametrize(
    ("audio", "reason"),
    [
        ({"rate": 8000}, "8000"),
        ({"rate": 44100}, "44100"),
        ({"channels": 2}, "channel"),
        ({"width": 1}, "16-bit"),
    ],
)
def test_audio_other_than_16_khz_mono_pcm16_is_refused(
    tmp_path: Path, audio: dict[str, int], reason: str
) -> None:
    write_wav(tmp_path / "recordings" / "p" / "bad.wav", tone_ms=1000, **audio)
    message = refused(tmp_path, [entry("p/a.wav"), entry("p/bad.wav")])
    assert "entry 1 (p/bad.wav)" in message and reason in message


def test_a_reference_file_is_checked_like_the_file(tmp_path: Path) -> None:
    write_wav(tmp_path / "recordings" / "m" / "ref.wav", tone_ms=1000, rate=48000)
    message = refused(tmp_path, [entry("m/a.wav", "media", reference_file="m/ref.wav")])
    assert "entry 0 (m/a.wav)" in message and "m/ref.wav" in message and "48000" in message


def test_a_duration_that_does_not_match_the_audio_is_refused(tmp_path: Path) -> None:
    write_wav(tmp_path / "recordings" / "p" / "a.wav", tone_ms=500)
    message = refused(tmp_path, [entry("p/a.wav", duration_ms=1000)])
    assert "entry 0 (p/a.wav)" in message and "duration_ms" in message


# -- The manifest: split hygiene --------------------------------------------------------------


def test_a_positive_session_in_tune_and_test_is_refused_naming_both(tmp_path: Path) -> None:
    message = refused(
        tmp_path,
        [
            entry("p/train.wav", split="train", session="day-1"),
            entry("p/tune.wav", split="tune", session="day-2"),
            entry("p/test.wav", split="test", session="day-2"),
        ],
    )
    assert "entry 1 (p/tune.wav" in message and "entry 2 (p/test.wav" in message
    assert "day-2" in message


def test_a_media_session_in_train_and_test_is_refused_naming_both(tmp_path: Path) -> None:
    message = refused(
        tmp_path,
        [
            entry("m/a.wav", "media", "train", phrase="", session="film-night"),
            entry("m/b.wav", "media", "test", phrase="", session="film-night"),
        ],
    )
    assert "entry 0 (m/a.wav" in message and "entry 1 (m/b.wav" in message
    assert "film-night" in message


def test_a_session_shared_across_labels_and_splits_is_refused(tmp_path: Path) -> None:
    message = refused(
        tmp_path,
        [
            entry("p/a.wav", "positive", "train", session="evening"),
            entry("h/b.wav", "homophone", "tune", phrase="I missed it", session="evening"),
        ],
    )
    assert "entry 0 (p/a.wav" in message and "entry 1 (h/b.wav" in message


def test_one_file_in_two_splits_is_refused_naming_both(tmp_path: Path) -> None:
    message = refused(
        tmp_path,
        [
            entry("p/a.wav", split="train", session="s1"),
            entry("p/a.wav", split="tune", session="s2"),
        ],
    )
    assert "entry 0 (p/a.wav" in message and "entry 1 (p/a.wav" in message


def test_the_same_audio_under_two_names_in_two_splits_is_refused(tmp_path: Path) -> None:
    write_wav(tmp_path / "recordings" / "p" / "a.wav", tone_ms=1000)
    (tmp_path / "recordings" / "p" / "copy.wav").write_bytes(
        (tmp_path / "recordings" / "p" / "a.wav").read_bytes()
    )
    message = refused(
        tmp_path,
        [
            entry("p/a.wav", split="train", session="s1"),
            entry("p/copy.wav", split="test", session="s2"),
        ],
    )
    assert "entry 0 (p/a.wav" in message and "entry 1 (p/copy.wav" in message
    assert "sha256" in message


def test_a_homophone_sentence_in_two_splits_is_refused_naming_both(tmp_path: Path) -> None:
    message = refused(
        tmp_path,
        [
            entry("h/a.wav", "homophone", "train", phrase="I missed the bus", session="s1"),
            entry("h/b.wav", "homophone", "tune", phrase="I  missed the bus.", session="s2"),
        ],
    )
    assert "entry 0 (h/a.wav" in message and "entry 1 (h/b.wav" in message


def test_the_split_command_writes_the_training_subset_and_prints_its_sha256(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_manifest(
        tmp_path,
        [
            entry("p/a.wav", split="train", session="s1"),
            entry("p/b.wav", split="tune", session="s2"),
            entry("m/c.wav", "media", "train", phrase="", session="s3"),
            entry("p/d.wav", split="test", session="s4"),
        ],
    )
    out = tmp_path / "out" / "train_manifest.json"

    code = wake_manifest.main(
        ["split", "--manifest", str(path), "--split", "train", "--out", str(out)]
    )

    assert code == 0
    printed = capsys.readouterr().out.strip()
    assert printed == hashlib.sha256(out.read_bytes()).hexdigest()
    subset = json.loads(out.read_text(encoding="utf-8"))
    assert subset["root"] == str(tmp_path / "recordings")
    assert [item["file"] for item in subset["entries"]] == ["p/a.wav", "m/c.wav"]
    assert printed == wake_manifest.subset_sha256(wake_manifest.load_manifest(path), "train")
    # The subset is itself a manifest the check accepts.
    assert len(wake_manifest.load_manifest(out).entries) == 2


# -- The harness: refusals --------------------------------------------------------------------


def small_setup(tmp_path: Path, **model: object) -> tuple[Path, Path, list[dict[str, object]]]:
    entries = [
        entry("p/train.wav", split="train", session="s-train"),
        entry("p/tune.wav", split="tune", session="s-tune"),
        entry("p/test.wav", split="test", session="s-test"),
        entry("m/test.wav", "media", "test", phrase="", session="s-test-media"),
    ]
    path = write_manifest(tmp_path, entries)
    training = wake_manifest.subset_sha256(wake_manifest.load_manifest(path), "train")
    models_dir = tmp_path / "models"
    write_model(models_dir, training=str(model.pop("training", training)), **model)
    return path, models_dir, entries


def harness_args(path: Path, models_dir: Path, out: Path, split: str, thresholds: str) -> list[str]:
    return [
        "--manifest",
        str(path),
        "--model",
        MODEL_ID,
        "--models-dir",
        str(models_dir),
        "--split",
        split,
        "--thresholds",
        thresholds,
        "--out",
        str(out),
    ]


def test_the_train_split_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path, models_dir, _ = small_setup(tmp_path)
    out = tmp_path / "eval.json"
    engines = rig([], [])

    code = engines.run(harness_args(path, models_dir, out, "train", "0.5"))

    assert code == 2
    assert "train" in capsys.readouterr().err
    assert not out.exists()
    assert engines.vad.streams == []


def test_a_test_run_with_more_than_one_threshold_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path, models_dir, _ = small_setup(tmp_path, calibration=TUNED_CALIBRATION)
    out = tmp_path / "eval.json"
    engines = rig([], [])

    code = engines.run(harness_args(path, models_dir, out, "test", "0.5,0.6"))

    assert code == 2
    assert "exactly one threshold" in capsys.readouterr().err
    assert not out.exists()
    assert engines.vad.streams == []


def test_a_model_trained_on_another_training_split_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path, models_dir, _ = small_setup(tmp_path, training="0" * 64)
    out = tmp_path / "eval.json"
    engines = rig([], [])

    code = engines.run(harness_args(path, models_dir, out, "tune", "0.5"))

    assert code == 2
    assert "training" in capsys.readouterr().err
    assert not out.exists()
    assert engines.vad.streams == []


def test_a_test_run_before_the_tuning_pass_is_written_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path, models_dir, _ = small_setup(tmp_path)
    out = tmp_path / "eval.json"

    code = rig([], []).run(harness_args(path, models_dir, out, "test", "0.5"))

    assert code == 2
    assert "calibration" in capsys.readouterr().err
    assert not out.exists()


def test_the_real_engines_need_an_stt_factory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path, models_dir, _ = small_setup(tmp_path)
    out = tmp_path / "eval.json"

    code = eval_wake.main(harness_args(path, models_dir, out, "tune", "0.5"), environ={})

    assert code == 2
    assert "--stt-factory" in capsys.readouterr().err
    assert not out.exists()


# -- The harness: the tuning pass -------------------------------------------------------------


def tune_setup(tmp_path: Path) -> tuple[list[str], Rig, Path, list[dict[str, object]]]:
    """Four positives, two homophones and two media clips in `tune`, around other splits.

    Positives: p1 "hey mist" (headset, 0.65), p2 "hey mist" (desk, 0.45), p3 "mist" alone
    (headset, 0.55, 640 ms of speech then 640 ms of silence, the hit 100 ms after the speech
    end) and p4 "are you there mist" (desk, 0.35, a decode the gate rejects). Homophones h1 and
    h2 both score 0.6; only h1's decode is addressed. Media m1 scores 0.5 and is addressed; m2
    scores 0.7 but its reference holds the name at 0.9, so the gate rejects it as self media.
    """
    recordings = tmp_path / "recordings"
    write_wav(recordings / "p" / "p3.wav", tone_ms=640, silence_ms=640)
    write_wav(recordings / "m" / "m2_ref.wav", tone_ms=1000, constant=REFERENCE_SAMPLE)
    entries = [
        entry("p/train1.wav", split="train", session="train-1"),
        entry("m/train2.wav", "media", "train", phrase="", session="train-2"),
        entry("p/p1.wav", condition="headset", session="tune-1"),
        entry("p/p2.wav", condition="desk", session="tune-2"),
        entry("p/p3.wav", phrase="mist", condition="headset", session="tune-1", duration_ms=1280),
        entry("p/p4.wav", phrase="are you there mist", condition="desk", session="tune-2"),
        entry("h/h1.wav", "homophone", phrase="I missed the bus", session="tune-3"),
        entry("h/h2.wav", "homophone", phrase="the mist rolled in", session="tune-3"),
        entry("m/m1.wav", "media", phrase="", session="tune-4"),
        entry("m/m2.wav", "media", phrase="", session="tune-4", reference_file="m/m2_ref.wav"),
        entry("p/test1.wav", split="test", session="test-1"),
    ]
    path = write_manifest(tmp_path, entries)
    training = wake_manifest.subset_sha256(wake_manifest.load_manifest(path), "train")
    models_dir = tmp_path / "models"
    write_model(models_dir, training=training)
    at = offsets(entries, "tune")
    hits = [
        hit(at, "p/p1.wav", 20, 0.65),
        hit(at, "p/p2.wav", 20, 0.45),
        hit(at, "p/p3.wav", 36, 0.55),
        hit(at, "p/p4.wav", 20, 0.35),
        hit(at, "h/h1.wav", 20, 0.6),
        hit(at, "h/h2.wav", 20, 0.6),
        hit(at, "m/m1.wav", 20, 0.5),
        hit(at, "m/m2.wav", 25, 0.7),
    ]
    script = [ADDRESSED, ADDRESSED, "mist", UNADDRESSED, ADDRESSED, UNADDRESSED, ADDRESSED]
    engines = rig(hits, script, decode_ms=[100.0, 200.0, 300.0])
    out = tmp_path / "eval" / "wake_eval.json"
    return harness_args(path, models_dir, out, "tune", SWEEP), engines, out, entries


def by_threshold(report: Mapping[str, object]) -> dict[float, Mapping[str, object]]:
    return {row["threshold"]: row for row in report["thresholds"]}  # type: ignore[index]


def test_the_tuning_pass_reads_only_the_tune_split(tmp_path: Path) -> None:
    argv, engines, out, entries = tune_setup(tmp_path)

    assert engines.run(argv) == 0

    tune = [item for item in entries if item["split"] == "tune"]
    assert len(engines.vad.streams) == len(tune)
    tune_frames = sum(int(item["duration_ms"]) // FRAME_MS for item in tune)
    pushed = [pos for stream in engines.spotter.streams for pos in stream.pushed]
    assert pushed and max(pushed) < tune_frames
    text = out.read_text(encoding="utf-8")
    for item in entries:
        if item["split"] != "tune":
            assert str(item["file"]) not in text
    report = json.loads(text)
    assert report["split"] == "tune"
    assert report["entries"] == {"positive": 4, "homophone": 2, "media": 2}


def test_the_tuning_pass_writes_the_rates_per_threshold(tmp_path: Path) -> None:
    argv, engines, out, _ = tune_setup(tmp_path)

    assert engines.run(argv) == 0

    rows = by_threshold(json.loads(out.read_text(encoding="utf-8")))
    assert sorted(rows) == [0.3, 0.4, 0.5, 0.6, 0.7]

    low = rows[0.3]["false_reject"]
    assert low["overall"] == {"rate": 0.25, "rejected": 1, "total": 4}
    assert low["by_condition"]["headset"] == {"rate": 0.0, "rejected": 0, "total": 2}
    assert low["by_condition"]["desk"] == {"rate": 0.5, "rejected": 1, "total": 2}
    assert low["by_phrase"]["hey mist"] == {"rate": 0.0, "rejected": 0, "total": 2}
    assert low["by_phrase"]["mist"] == {"rate": 0.0, "rejected": 0, "total": 1}
    assert low["by_phrase"]["are you there mist"] == {"rate": 1.0, "rejected": 1, "total": 1}
    assert low["files"] == ["p/p4.wav"]
    assert low["reasons"] == {"not_addressed": 1}
    assert rows[0.4]["false_reject"]["reasons"] == {"no_hit": 1}

    middle = rows[0.5]["false_reject"]
    assert middle["overall"] == {"rate": 0.5, "rejected": 2, "total": 4}
    assert middle["by_condition"]["headset"]["rate"] == 0.0
    assert middle["by_condition"]["desk"]["rate"] == 1.0
    assert rows[0.7]["false_reject"]["overall"]["rate"] == 1.0

    media_hours = 2000 / 3_600_000
    for threshold in (0.3, 0.4, 0.5):
        media = rows[threshold]["media_false_accepts"]
        assert media["count"] == 1
        assert media["per_hour"] == pytest.approx(1 / media_hours)
        assert media["files"] == ["m/m1.wav"]
        assert media["reasons"] == {"self_media": 1}
    assert rows[0.6]["media_false_accepts"]["per_hour"] == 0.0

    for threshold in (0.3, 0.4, 0.5, 0.6):
        homophone = rows[threshold]["homophone_false_accepts"]
        assert homophone == {"accepted": 1, "total": 2, "share": 0.5, "files": ["h/h1.wav"]}
    assert rows[0.7]["homophone_false_accepts"]["share"] == 0.0


def test_the_tuning_pass_writes_latency_and_the_measured_hit_lag(tmp_path: Path) -> None:
    argv, engines, out, _ = tune_setup(tmp_path)

    assert engines.run(argv) == 0

    rows = by_threshold(json.loads(out.read_text(encoding="utf-8")))
    # Accepted positives at 0.3: p1, p2 and p3, decoded in 100, 200 and 300 ms after the
    # 300 ms gate tail.
    assert rows[0.3]["wake_accept_ms"] == {"p50": 500.0, "p90": 580.0, "count": 3}
    assert rows[0.5]["wake_accept_ms"] == {"p50": 500.0, "p90": 580.0, "count": 2}
    assert rows[0.7]["wake_accept_ms"] == {"p50": None, "p90": None, "count": 0}
    # p3's hit ends at 740 ms and its speech at 640 ms.
    for threshold in (0.3, 0.4, 0.5):
        assert rows[threshold]["hit_lag_ms"] == {"median": 100, "p90": 100.0, "count": 1}
        screen = rows[threshold]["wake_screen_ms"]
        assert screen["p50"] == 100.0 and screen["p90"] == 100.0
        assert screen["hit_lag_ms"] == 100
    assert rows[0.6]["hit_lag_ms"] == {"median": None, "p90": None, "count": 0}
    # One decode per distinct span: m2 never reaches the STT, and the sweep reuses the rest.
    assert len(engines.transcriber.streams) == 7


def test_the_tuning_pass_writes_both_calibration_tables(tmp_path: Path) -> None:
    argv, engines, out, _ = tune_setup(tmp_path)

    assert engines.run(argv) == 0

    report = json.loads(out.read_text(encoding="utf-8"))
    server = report["calibration"]["server"]
    rate = pytest.approx(1 / (2000 / 3_600_000))
    assert [score for score, _ in server["false_accepts"]] == [0.3, 0.4, 0.5, 0.6, 0.7]
    assert [value for _, value in server["false_accepts"]] == [rate, rate, rate, 0.0, 0.0]
    # Positive hits at the lowest threshold score 0.65, 0.45, 0.55 and 0.35.
    assert server["positive_quantile"] == [
        [0.3, 0.0],
        [0.4, 0.25],
        [0.5, 0.5],
        [0.6, 0.75],
        [0.7, 1.0],
    ]
    # The tables are in T34's format: a model manifest holding them loads.
    model = write_model(tmp_path / "check", training="a" * 64, calibration=report["calibration"])
    loaded = read_manifest(model, MODEL_ID)
    assert loaded.calibration["server"].positive_quantile[2] == (0.5, 0.5)


def test_the_false_accept_table_takes_the_running_minimum_from_low_scores_up() -> None:
    table = eval_wake.false_accept_table([(0.3, 2.0), (0.4, 3.0), (0.5, 1.0), (0.6, 1.5)])
    assert table == [(0.3, 2.0), (0.4, 2.0), (0.5, 1.0), (0.6, 1.0)]


def test_positive_quantiles_from_known_scores() -> None:
    scores = [0.2, 0.45, 0.5, 0.5, 0.9]
    table = eval_wake.positive_quantile_table(scores, [0.1, 0.5, 0.8, 1.0])
    assert table == [(0.1, 0.0), (0.5, 0.8), (0.8, 0.8), (1.0, 1.0)]


def test_no_transcript_text_reaches_the_output_or_the_logs(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv, engines, out, _ = tune_setup(tmp_path)
    caplog.set_level(logging.DEBUG)

    assert engines.run(argv) == 0

    captured = capsys.readouterr()
    written = out.read_text(encoding="utf-8")
    for word in SECRET_WORDS:
        assert word not in written
        assert word not in caplog.text
        assert word not in captured.out
        assert word not in captured.err
    assert caplog.records, "the run should log its progress"


# -- The harness: the test pass ---------------------------------------------------------------


def test_the_test_pass_reads_the_lag_and_tables_from_the_model_and_writes_none(
    tmp_path: Path,
) -> None:
    path, models_dir, entries = small_setup(tmp_path, hit_lag_ms=120, calibration=TUNED_CALIBRATION)
    at = offsets(entries, "test")
    engines = rig(
        [hit(at, "p/test.wav", 20, 0.65), hit(at, "m/test.wav", 20, 0.4)],
        [ADDRESSED],
        decode_ms=[150.0],
    )
    out = tmp_path / "eval.json"

    assert engines.run(harness_args(path, models_dir, out, "test", "0.5")) == 0

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["split"] == "test"
    assert "calibration" not in report
    assert report["hit_lag_ms"] == 120
    (row,) = report["thresholds"]
    assert row["threshold"] == 0.5
    assert "hit_lag_ms" not in row
    assert row["false_reject"]["overall"] == {"rate": 0.0, "rejected": 0, "total": 1}
    assert row["wake_accept_ms"] == {"p50": 450.0, "p90": 450.0, "count": 1}
    assert row["wake_screen_ms"] == {"p50": 120.0, "p90": 120.0, "count": 1, "hit_lag_ms": 120}
    assert row["media_false_accepts"]["count"] == 0
    assert len(engines.vad.streams) == 2
