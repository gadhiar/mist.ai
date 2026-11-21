"""Tests for the wake model evaluation harness: the recording manifest and `eval_wake`.

Every WAV here is written by `wave` into pytest's temporary directory, and the engines are the
scripted fakes (`ScriptedNameSpotter`, `ScriptedWakeVerifier`, `EnergyVad`), so no recording and
no model is read. The model manifest is the microWakeWord format over a placeholder model file,
which `read_manifest` hashes like a real one, and the verifier's directory holds its manifest, a
placeholder model and `provenance.json`, which `read_verifier_manifest` hashes the same way.

Scripted hits are addressed by uplink position. The harness numbers the frames of a split's
entries in manifest order, one position per 20 ms frame, so `offsets` below turns an entry and a
frame within it into the position a hit is scripted at. Verifier scores are scripted in the order
the windows are scored: entries in manifest order, one score per distinct hit position.
"""

import asyncio
import hashlib
import json
import logging
import wave
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from backend.realtime.budget import EndpointingConfig, PlacementConfig, load_voice_settings
from backend.realtime.engines.kws import read_manifest
from backend.realtime.engines.verifier import GRID, verify_window
from backend.realtime.errors import SpeechEngineError
from backend.realtime.ingress import (
    AudioIngress,
    IngressEngines,
    IngressLanes,
    SegmentStep,
    SpeechSegmentTracker,
    onset_lead_frames,
)
from backend.realtime.protocol import Frame, MsgType
from backend.realtime.speech import CancelToken, NameHit
from scripts.wake import eval_wake
from scripts.wake import manifest as wake_manifest
from tests.fakes.speech import (
    EnergyVad,
    ScriptedNameSpotter,
    ScriptedTranscriber,
    ScriptedWakeVerifier,
)

RATE = 16000
FRAME_MS = 20
MODEL_ID = "mist-wake-test"
VERIFIER_ID = "mist-verify-test"
SPOTTER_PAYLOAD = b"placeholder streaming model"
ADDRESSED = 0.9
"""A verifier score above the scripted threshold, 0.5: the hit is accepted."""
UNADDRESSED = 0.1
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
    """A microWakeWord manifest over a placeholder model file, which `read_manifest` hashes."""
    directory = models_dir / "speech" / "wake" / MODEL_ID
    (directory / "device").mkdir(parents=True, exist_ok=True)
    payload = SPOTTER_PAYLOAD
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


def write_verifier(
    models_dir: Path,
    *,
    training: str,
    listed: Sequence[str] = (),
    provenance_training: str | None = None,
    spotter_sha256: str | None = None,
) -> Path:
    """The verifier's directory: its manifest, a placeholder model and `provenance.json`.

    `listed` are the sha256 values the provenance says windows were read from, beside one of a
    training file of its own; the provenance records `provenance_training`, else `training`.
    """
    directory = models_dir / "speech" / "wake" / VERIFIER_ID
    directory.mkdir(parents=True, exist_ok=True)
    model = b"placeholder verifier model"
    (directory / "model.onnx").write_bytes(model)
    provenance = {
        "version": 1,
        "training": provenance_training or training,
        "file_sha256": sorted([hashlib.sha256(b"a training clip").hexdigest(), *listed]),
        "sources": {
            "train-split": {"licence": "personal", "version": "1", "files": 1, "hours": 0.5},
            "synthetic": {"licence": "MIT", "version": "1", "files": 3, "hours": 1.25},
        },
        "excluded": {"ami_dev_sha256sums": "d" * 64, "ami_eval_sha256sums": "e" * 64},
    }
    data = json.dumps(provenance, indent=2).encode("utf-8")
    (directory / "provenance.json").write_bytes(data)
    manifest = {
        "version": 1,
        "kind": "wake_verifier",
        "model_id": VERIFIER_ID,
        "file": "model.onnx",
        "sha256": hashlib.sha256(model).hexdigest(),
        "sample_rate": 16000,
        "frame_ms": 20,
        "before_ms": 2000,
        "after_ms": 300,
        "min_window_ms": 800,
        "spotter": {
            "model_id": MODEL_ID,
            "model_sha256": spotter_sha256 or hashlib.sha256(SPOTTER_PAYLOAD).hexdigest(),
        },
        "thresholds": {
            "cutoffs": list(GRID),
            "scores": [0.5] * len(GRID),
            "per_hour": 0.2,
            "hours": 30.0,
        },
        "training": training,
        "provenance_sha256": hashlib.sha256(data).hexdigest(),
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


# -- Engines ----------------------------------------------------------------------------------


class FakeClock:
    """A monotonic clock that moves only when a test moves it."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, ms: float) -> None:
        self.now += ms / 1000


class TimedVerifier(ScriptedWakeVerifier):
    """`ScriptedWakeVerifier` whose every score costs time on the test clock, and logs it."""

    def __init__(self, on_score: Callable[[], None], *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.on_score = on_score

    def score(self, pcm: bytes) -> float:
        self.on_score()
        return super().score(pcm)


@dataclass
class Rig:
    """The fakes one harness run is given, kept so a test can read what reached them."""

    spotter: ScriptedNameSpotter
    verifier: ScriptedWakeVerifier
    clock: FakeClock = field(default_factory=FakeClock)
    vad: EnergyVad = field(default_factory=EnergyVad)
    placements: list[PlacementConfig] = field(default_factory=list)

    @property
    def engines(self) -> eval_wake.Engines:
        return eval_wake.Engines(spotter=self.spotter, vad=self.vad, verifier=self.verifier)

    def load(self, placement: PlacementConfig) -> eval_wake.Engines:
        self.placements.append(placement)
        return self.engines

    def run(self, argv: Sequence[str]) -> int:
        return eval_wake.main(list(argv), load_engines=self.load, environ={}, clock=self.clock)


def rig(
    hits: Sequence[NameHit],
    scores: Sequence[float | SpeechEngineError] = (),
    *,
    verify_ms: Sequence[float] = (),
    min_window_ms: int = 400,
    thresholds: float | Callable[[float], float | None] = 0.5,
) -> Rig:
    """Scripted engines; each window scored takes the next of `verify_ms` (else 50 ms).

    The verifier's windows are 2000 ms before a hit and 300 ms after it, as the manifest's, but
    it scores a window of `min_window_ms` and up, 400 ms by default, so a hit 400 ms into a
    one-second clip is scored; past `scores` every window scores 1.0, against 0.5.
    """
    clock = FakeClock()
    pending = list(verify_ms)

    def verifying() -> None:
        clock.advance(pending.pop(0) if pending else 50.0)

    spotter = ScriptedNameSpotter(hits, markers={REFERENCE_MARKER: 0.9})
    verifier = TimedVerifier(
        verifying,
        scores,
        thresholds=thresholds,
        spotter_model_id=MODEL_ID,
        model_id=VERIFIER_ID,
        min_window_ms=min_window_ms,
    )
    return Rig(spotter=spotter, verifier=verifier, clock=clock)


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


def test_a_positive_session_in_tune_and_test_loads(tmp_path: Path) -> None:
    path = write_manifest(
        tmp_path,
        [
            entry("p/train.wav", split="train", session="day-1"),
            entry("p/tune.wav", split="tune", session="day-2"),
            entry("p/test.wav", split="test", session="day-2"),
        ],
    )

    loaded = wake_manifest.load_manifest(path)

    assert [(item.file, item.split) for item in loaded.entries if item.session == "day-2"] == [
        ("p/tune.wav", "tune"),
        ("p/test.wav", "test"),
    ]


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


@pytest.mark.parametrize("media_first", [True, False])
def test_a_session_with_a_media_entry_in_train_and_a_positive_in_tune_is_refused(
    tmp_path: Path, media_first: bool
) -> None:
    media = entry("m/a.wav", "media", "train", phrase="", session="living-room")
    positive = entry("p/b.wav", "positive", "tune", session="living-room")
    rows = [media, positive] if media_first else [positive, media]
    media_index = 0 if media_first else 1

    message = refused(tmp_path, rows)

    assert f"entry {media_index} (m/a.wav" in message
    assert f"entry {1 - media_index} (p/b.wav" in message
    assert "living-room" in message and "media" in message


def test_a_session_shared_across_labels_and_splits_loads(tmp_path: Path) -> None:
    path = write_manifest(
        tmp_path,
        [
            entry("p/a.wav", "positive", "train", session="evening"),
            entry("h/b.wav", "homophone", "tune", phrase="I missed it", session="evening"),
        ],
    )

    loaded = wake_manifest.load_manifest(path)

    assert [(item.file, item.split) for item in loaded.entries] == [
        ("p/a.wav", "train"),
        ("h/b.wav", "tune"),
    ]


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


# -- The manifest: assigning splits -----------------------------------------------------------

ASSIGN_DURATIONS = (800, 600, 1000, 400, 700, 900)
ASSIGN_CONDITIONS = {
    "positive": ("headset", "desk"),
    "homophone": ("headset", "desk"),
    "media": ("speakers", "tv"),
}
SPLIT_TARGETS = {"train": 0.6, "tune": 0.2, "test": 0.2}


def assign_entries() -> list[dict[str, object]]:
    """Six sessions per (label, condition), most without a split, some of them linked.

    Homophone headset sessions 0 and 1 hold one sentence, written two ways, and session 1 also
    holds a short positive, so those three recordings must share a split. Positive desk session
    2 holds two recordings. Entry 0 carries a split, which `assign` replaces.
    """
    rows: list[dict[str, object]] = []
    for label, conditions in ASSIGN_CONDITIONS.items():
        for c_index, condition in enumerate(conditions):
            for number, duration in enumerate(ASSIGN_DURATIONS):
                phrase = {"positive": "hey mist", "homophone": f"sentence {c_index} {number}"}
                rows.append(
                    entry(
                        f"clip{len(rows):02d}.wav",
                        label,
                        phrase=phrase.get(label, ""),
                        condition=condition,
                        session=f"{label}-{condition}-{number}",
                        duration_ms=duration,
                    )
                )
    for item in rows:
        if item["session"] == "homophone-headset-1":
            item["phrase"] = "Sentence  0 0."
    rows.append(
        entry(
            f"clip{len(rows):02d}.wav",
            condition="headset",
            session="homophone-headset-1",
            duration_ms=200,
        )
    )
    rows.append(
        entry(
            f"clip{len(rows):02d}.wav", condition="desk", session="positive-desk-2", duration_ms=300
        )
    )
    for index, item in enumerate(rows):
        if index:
            del item["split"]
        else:
            item["split"] = "test"
    return rows


def run_assign(path: Path, out: Path, seed: int) -> int:
    return wake_manifest.main(
        ["assign", "--manifest", str(path), "--seed", str(seed), "--out", str(out)]
    )


def split_groups(entries: Sequence[Mapping[str, object]]) -> list[list[Mapping[str, object]]]:
    """Recordings linked by a session or a homophone sentence, transitively."""
    groups: list[list[Mapping[str, object]]] = []
    for item in entries:
        joined = [
            group
            for group in groups
            if any(
                other["session"] == item["session"]
                or (
                    other["label"] == item["label"] == "homophone"
                    and wake_manifest.phrase_key(str(other["phrase"]))
                    == wake_manifest.phrase_key(str(item["phrase"]))
                )
                for other in group
            )
        ]
        merged = [member for group in joined for member in group] + [item]
        groups = [group for group in groups if group not in joined] + [merged]
    return groups


def test_assign_writes_splits_the_check_accepts(tmp_path: Path) -> None:
    rows = assign_entries()
    path = write_manifest(tmp_path, rows)
    out = tmp_path / "out" / "assigned.json"

    assert run_assign(path, out, 7) == 0

    loaded = wake_manifest.load_manifest(out)
    assert [item.file for item in loaded.entries] == [item["file"] for item in rows]
    assert {item.split for item in loaded.entries} == set(wake_manifest.SPLITS)
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["root"] == str(tmp_path / "recordings")
    for before, after in zip(rows, written["entries"], strict=True):
        assert {key: value for key, value in after.items() if key != "split"} == {
            key: value for key, value in before.items() if key != "split"
        }
    linked = {
        item.split
        for item in loaded.entries
        if item.session in ("homophone-headset-0", "homophone-headset-1")
    }
    assert len(linked) == 1
    assert len({item.split for item in loaded.entries if item.session == "positive-desk-2"}) == 1


def test_assign_is_reproducible_by_seed(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, assign_entries())

    assert run_assign(path, tmp_path / "a.json", 7) == 0
    assert run_assign(path, tmp_path / "b.json", 7) == 0
    assert run_assign(path, tmp_path / "c.json", 8) == 0

    first = (tmp_path / "a.json").read_bytes()
    assert first == (tmp_path / "b.json").read_bytes()
    assert first != (tmp_path / "c.json").read_bytes()


@pytest.mark.parametrize("seed", [1, 2, 3, 7, 1234])
def test_assign_keeps_each_split_within_one_group_of_its_share(tmp_path: Path, seed: int) -> None:
    path = write_manifest(tmp_path, assign_entries())
    out = tmp_path / "assigned.json"

    assert run_assign(path, out, seed) == 0

    entries = json.loads(out.read_text(encoding="utf-8"))["entries"]
    by_key: dict[tuple[str, str], list[list[Mapping[str, object]]]] = {}
    for group in split_groups(entries):
        assert len({item["split"] for item in group}) == 1
        totals: Counter[tuple[str, str]] = Counter()
        for item in group:
            totals[(str(item["label"]), str(item["condition"]))] += int(item["duration_ms"])
        by_key.setdefault(totals.most_common(1)[0][0], []).append(group)
    assert len(by_key) == 6
    for key, groups in by_key.items():
        assert len(groups) >= 5, key
        durations = [sum(int(item["duration_ms"]) for item in group) for group in groups]
        total = sum(durations)
        largest = max(durations) / total
        for split, target in SPLIT_TARGETS.items():
            share = (
                sum(
                    duration
                    for group, duration in zip(groups, durations, strict=True)
                    if group[0]["split"] == split
                )
                / total
            )
            assert abs(share - target) <= largest + 1e-9, (key, split, share)


def test_assign_refuses_one_audio_under_two_names_as_check_does(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_wav(tmp_path / "recordings" / "p" / "a.wav", tone_ms=1000)
    (tmp_path / "recordings" / "p" / "copy.wav").write_bytes(
        (tmp_path / "recordings" / "p" / "a.wav").read_bytes()
    )
    path = write_manifest(
        tmp_path,
        [entry("p/a.wav", split="train", session="s1"), entry("p/copy.wav", session="s2")],
    )
    out = tmp_path / "assigned.json"

    assert wake_manifest.main(["check", "--manifest", str(path)]) == 2
    checked = capsys.readouterr().err
    assert run_assign(path, out, 7) == 2

    message = capsys.readouterr().err
    assert not out.exists()
    for refusal in (checked, message):
        assert "entry 0 (p/a.wav" in refusal and "entry 1 (p/copy.wav" in refusal
        assert "hold audio with one sha256" in refusal


def test_assign_checks_every_other_field_as_check_does(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_wav(tmp_path / "recordings" / "p" / "a.wav", tone_ms=500)
    item = entry("p/a.wav", duration_ms=1000)
    del item["split"]
    path = write_manifest(tmp_path, [item])
    out = tmp_path / "assigned.json"

    assert run_assign(path, out, 7) == 2

    assert "duration_ms" in capsys.readouterr().err
    assert not out.exists()


def test_assign_refuses_an_out_that_exists(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_manifest(tmp_path, assign_entries())
    out = tmp_path / "assigned.json"
    out.write_text("kept\n", encoding="utf-8")

    assert run_assign(path, out, 7) == 2

    assert "--out" in capsys.readouterr().err
    assert out.read_text(encoding="utf-8") == "kept\n"


def test_assign_prints_the_seed_and_shares_and_no_file_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = assign_entries()
    path = write_manifest(tmp_path, rows)
    out = tmp_path / "assigned.json"

    assert run_assign(path, out, 7) == 0

    captured = capsys.readouterr()
    printed = captured.out + captured.err
    assert captured.out.splitlines()[0] == "seed 7"
    for label, conditions in ASSIGN_CONDITIONS.items():
        for condition in conditions:
            (line,) = (
                text
                for text in captured.out.splitlines()
                if text.startswith(f"{label} {condition}:")
            )
            assert all(f"{split} " in line and "%" in line for split in wake_manifest.SPLITS)
    for item in rows:
        assert str(item["file"]) not in printed
    for name in ("clip", ".wav", ".json", "assigned", "manifest", str(tmp_path)):
        assert name not in printed


# -- The harness: segmentation ----------------------------------------------------------------


def test_the_harness_segments_through_ingress_s_tracker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    steps: list[SegmentStep] = []

    class Recording(SpeechSegmentTracker):
        def window(self, speech: bool, first: int, last: int) -> SegmentStep:
            step = super().window(speech, first, last)
            steps.append(step)
            return step

    monkeypatch.setattr(eval_wake, "SpeechSegmentTracker", Recording)
    write_wav(tmp_path / "a.wav", tone_ms=640, silence_ms=640)
    cfg = EndpointingConfig()

    heard = eval_wake._listen(wake_manifest.read_pcm16(tmp_path / "a.wav"), EnergyVad(), cfg)

    assert len(steps) == 1280 * RATE // 1000 // 512
    closed = [step.closed for step in steps if step.closed is not None]
    assert heard.segments == tuple(closed) and len(closed) == 1
    (onset,) = (step.started[0] for step in steps if step.started is not None)
    assert heard.segments[0][0] == onset
    assert len(heard.frames) == 1280 // FRAME_MS


class ImmediateLane:
    """Runs each job at once on the event loop, as the harness makes every engine call."""

    async def run(self, fn: object, *args: object, token: CancelToken) -> object:
        return fn(*args)  # type: ignore[operator]


def tone_and_silence(parts_ms: Sequence[tuple[bool, int]]) -> np.ndarray:
    """Alternating tone and silence, each part a whole number of 20 ms frames."""
    pieces = []
    for loud, ms in parts_ms:
        count = RATE * ms // 1000
        t = np.arange(count) / RATE
        level = 0.3 * 32767 if loud else 0.0
        pieces.append((level * np.sin(2 * np.pi * 440.0 * t)).astype("<i2"))
    return np.concatenate(pieces)


@pytest.mark.asyncio
async def test_the_harness_feeds_the_spotter_what_ingress_feeds_it(tmp_path: Path) -> None:
    samples = tone_and_silence(
        [(False, 640), (True, 640), (False, 800), (True, 400), (False, 1200)]
    )
    recording = tmp_path / "recordings" / "p" / "a.wav"
    recording.parent.mkdir(parents=True)
    with wave.open(str(recording), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(samples.tobytes())
    duration_ms = len(samples) * 1000 // RATE
    path = write_manifest(tmp_path, [entry("p/a.wav", duration_ms=duration_ms)])
    (item,) = wake_manifest.load_manifest(path).entries
    settings = load_voice_settings({})

    live = ScriptedNameSpotter()
    lane = ImmediateLane()
    ingress = AudioIngress(
        engines=IngressEngines(
            vad=EnergyVad(), transcriber=ScriptedTranscriber([]), turn_end=None, spotter=live
        ),
        lanes=IngressLanes(vad=lane, stt=lane, eot=lane, kws=lane),
        endpointing=settings.endpointing,
        wake=settings.wake,
        emit=lambda event: None,
        retention_ms=7000,
        epoch=1,
        new_utterance_id=lambda: "u1",
        clock=lambda: 0,
    )
    pcm = samples.tobytes()
    frame_bytes = RATE * FRAME_MS // 1000 * 2
    for pos in range(len(pcm) // frame_bytes):
        payload = pcm[pos * frame_bytes : (pos + 1) * frame_bytes]
        ingress.push_frame(Frame(MsgType.MIC_AUDIO, 0, pos, RATE, payload), pos)
        for _ in range(20):
            await asyncio.sleep(0)
    ingress.close()

    offline = ScriptedNameSpotter()
    driver = eval_wake._Driver(
        eval_wake.Engines(spotter=offline, vad=EnergyVad(), verifier=ScriptedWakeVerifier()),
        settings,
        [settings.wake.kws_threshold],
        FakeClock(),
    )
    driver.run(item, 0)

    (heard,) = live.streams
    (scored,) = offline.streams
    assert heard.pushed == list(range(len(pcm) // frame_bytes))
    assert (scored.resets, scored.pushed) == (heard.resets, heard.pushed)


def test_the_harness_reports_a_hit_in_silence_and_spans_it(tmp_path: Path) -> None:
    # Silence (frames 0 to 31), tone (32 to 63), then 2400 ms of silence (64 to 183).
    samples = tone_and_silence([(False, 640), (True, 640), (False, 2400)])
    recording = tmp_path / "recordings" / "p" / "a.wav"
    recording.parent.mkdir(parents=True)
    with wave.open(str(recording), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(samples.tobytes())
    duration_ms = len(samples) * 1000 // RATE
    path = write_manifest(tmp_path, [entry("p/a.wav", duration_ms=duration_ms)])
    (item,) = wake_manifest.load_manifest(path).entries
    settings = load_voice_settings({})
    # Frame 150 is in silence more than 1000 ms after the segment's last speech.
    engines = rig([NameHit(end_pos=150, keyword="mist", score=0.9)], [ADDRESSED])
    driver = eval_wake._Driver(engines.engines, settings, [0.5], engines.clock)

    outcome = driver.run(item, 0)[0.5]

    assert [found.position for found in outcome.hits] == [150]
    assert outcome.accepted
    assert outcome.hits[0].verdict.check == "verified"
    # 2000 ms before the hit and 300 ms after it, inside the clip's 184 frames.
    frame_bytes = RATE * FRAME_MS // 1000 * 2
    assert engines.verifier.windows == [(15 + 100 + 1) * frame_bytes]
    assert engines.verifier.cutoffs == [0.5]


def write_recording(tmp_path: Path, samples: np.ndarray) -> Path:
    """One tune positive of these samples, and its manifest."""
    recording = tmp_path / "recordings" / "p" / "a.wav"
    recording.parent.mkdir(parents=True)
    with wave.open(str(recording), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(samples.tobytes())
    duration_ms = len(samples) * 1000 // RATE
    return write_manifest(tmp_path, [entry("p/a.wav", duration_ms=duration_ms)])


@pytest.mark.parametrize(
    ("hit_at", "window"),
    [
        # Inside the clip on both sides: the whole window.
        (120, (20, 135)),
        # Near the start: the clip's first frame bounds it, unpadded.
        (30, (0, 45)),
        # Near the end: the clip's last frame (149) bounds it.
        (145, (45, 149)),
    ],
)
def test_the_window_is_verify_window_over_the_clip_s_frames(
    tmp_path: Path, hit_at: int, window: tuple[int, int]
) -> None:
    # 3000 ms of tone: 150 frames.
    samples = tone_and_silence([(True, 3000)])
    (item,) = wake_manifest.load_manifest(write_recording(tmp_path, samples)).entries
    engines = rig([NameHit(end_pos=hit_at, keyword="mist", score=0.9)], min_window_ms=800)
    pushed: list[bytes] = []
    original = engines.verifier.score

    def scoring(pcm: bytes) -> float:
        pushed.append(pcm)
        return original(pcm)

    engines.verifier.score = scoring  # type: ignore[method-assign]
    driver = eval_wake._Driver(engines.engines, load_voice_settings({}), [0.5], engines.clock)

    driver.run(item, 0)

    first, last = window
    frames = eval_wake._frames(wake_manifest.read_pcm16(item.path))
    assert pushed == [b"".join(frames[first : last + 1])]
    # The window is the one training and ingress take, `verify_window`'s.
    shared = verify_window(
        lambda pos: 0 <= pos < len(frames),
        hit_at,
        before_frames=100,
        after_frames=15,
        min_frames=40,
    )
    assert shared == range(first, last + 1)


def test_a_window_under_the_minimum_is_rejected_unscored_and_counted(tmp_path: Path) -> None:
    # A hit 400 ms into a one-second clip: 20 frames before it and 15 after, 720 ms in all.
    samples = tone_and_silence([(True, 1000)])
    (item,) = wake_manifest.load_manifest(write_recording(tmp_path, samples)).entries
    engines = rig([NameHit(end_pos=20, keyword="mist", score=0.9)], min_window_ms=800)
    driver = eval_wake._Driver(engines.engines, load_voice_settings({}), [0.3, 0.5], engines.clock)

    outcomes = driver.run(item, 0)

    for outcome in outcomes.values():
        (found,) = outcome.hits
        assert not found.verdict.accepted
        assert found.verdict.check == "not_addressed"
        assert found.verify_score is None
    assert engines.verifier.windows == []
    assert (driver.counts.verifies, driver.counts.failures, driver.counts.short_windows) == (
        0,
        0,
        1,
    )


def test_a_failed_score_rejects_the_hit_and_is_counted(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    samples = tone_and_silence([(True, 1000)])
    (item,) = wake_manifest.load_manifest(write_recording(tmp_path, samples)).entries
    engines = rig(
        [NameHit(end_pos=30, keyword="mist", score=0.9)],
        [SpeechEngineError("scripted verify failure")],
        verify_ms=[40.0],
    )
    driver = eval_wake._Driver(engines.engines, load_voice_settings({}), [0.3, 0.5], engines.clock)
    caplog.set_level(logging.DEBUG)

    outcomes = driver.run(item, 0)

    assert [outcome.reason for outcome in outcomes.values()] == ["not_addressed"] * 2
    # Scored once, at the first threshold, and the failure reused at the second.
    assert len(engines.verifier.windows) == 1
    assert (driver.counts.verifies, driver.counts.failures, driver.counts.times_ms) == (
        1,
        1,
        [40.0],
    )
    assert "p/a.wav" not in caplog.text and "a.wav" not in caplog.text


def test_the_verifier_is_asked_for_the_threshold_of_each_cutoff(tmp_path: Path) -> None:
    # The table is stricter from 0.5 up: the one window's score, 0.7, passes at 0.3 alone.
    samples = tone_and_silence([(True, 1000)])
    (item,) = wake_manifest.load_manifest(write_recording(tmp_path, samples)).entries
    engines = rig(
        [NameHit(end_pos=30, keyword="mist", score=0.9)],
        [0.7],
        thresholds=lambda cutoff: 0.8 if cutoff >= 0.5 else 0.6,
    )
    driver = eval_wake._Driver(
        engines.engines, load_voice_settings({}), [0.3, 0.5, 0.6], engines.clock
    )

    outcomes = driver.run(item, 0)

    assert {cutoff: bool(outcome.accepted) for cutoff, outcome in outcomes.items()} == {
        0.3: True,
        0.5: False,
        0.6: False,
    }
    assert engines.verifier.cutoffs == [0.3, 0.5, 0.6]
    assert len(engines.verifier.windows) == 1


def test_a_cutoff_below_the_table_rejects_every_hit(tmp_path: Path) -> None:
    samples = tone_and_silence([(True, 1000)])
    (item,) = wake_manifest.load_manifest(write_recording(tmp_path, samples)).entries
    engines = rig(
        [NameHit(end_pos=30, keyword="mist", score=0.9)],
        thresholds=lambda cutoff: None if cutoff < 0.01 else 0.5,
    )
    driver = eval_wake._Driver(
        engines.engines, load_voice_settings({}), [0.005, 0.5], engines.clock
    )

    outcomes = driver.run(item, 0)

    assert outcomes[0.005].reason == "not_addressed"
    assert outcomes[0.5].accepted


# -- The harness: refusals --------------------------------------------------------------------


def small_setup(
    tmp_path: Path, *, verifier: Mapping[str, object] | None = None, **model: object
) -> tuple[Path, Path, list[dict[str, object]]]:
    """A manifest of one entry in each split and a test media clip, the model and its verifier.

    `verifier` holds `write_verifier`'s keyword arguments, its `training` defaulting to the
    manifest's train sha256.
    """
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
    options: dict[str, object] = dict(verifier or {})
    trained = str(options.pop("training", training))
    write_verifier(models_dir, training=trained, **options)  # type: ignore[arg-type]
    return path, models_dir, entries


def harness_args(
    path: Path,
    models_dir: Path,
    out: Path,
    split: str,
    thresholds: str,
    *,
    verifier: str | None = VERIFIER_ID,
) -> list[str]:
    argv = [
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
    return argv + (["--verifier", verifier] if verifier is not None else [])


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


@pytest.mark.parametrize(
    ("verifier", "reason"), [(None, "--verifier is required"), ("Bad_Id", "Bad_Id")]
)
def test_a_missing_or_malformed_verifier_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], verifier: str | None, reason: str
) -> None:
    path, models_dir, _ = small_setup(tmp_path)
    out = tmp_path / "eval.json"
    engines = rig([])

    code = engines.run(harness_args(path, models_dir, out, "tune", "0.5", verifier=verifier))

    assert code == 2
    assert reason in capsys.readouterr().err
    assert not out.exists()
    assert engines.placements == []


def provenance_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], split: str = "tune", **verifier: object
) -> str:
    """Run with this verifier; assert the refusal (exit 2, nothing written or scored)."""
    path, models_dir, entries = small_setup(
        tmp_path, verifier=verifier, calibration=TUNED_CALIBRATION
    )
    out = tmp_path / "eval.json"
    engines = rig([NameHit(end_pos=20, keyword="mist", score=0.9)])

    code = engines.run(harness_args(path, models_dir, out, split, "0.5"))

    err = capsys.readouterr().err
    assert code == 2
    assert err.startswith("eval_wake: ")
    assert not out.exists()
    assert engines.placements == []
    assert engines.vad.streams == engines.spotter.streams == []
    assert engines.verifier.windows == []
    # A refusal names no recording.
    for item in entries:
        assert Path(str(item["file"])).name not in err
    return err


def entry_sha256(tmp_path: Path, file: str) -> str:
    return hashlib.sha256((tmp_path / "recordings" / file).read_bytes()).hexdigest()


def test_a_verifier_trained_on_another_training_split_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    err = provenance_refused(tmp_path, capsys, training="0" * 64)

    assert VERIFIER_ID in err and "train" in err


def test_a_provenance_recording_another_training_split_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    err = provenance_refused(tmp_path, capsys, provenance_training="1" * 64)

    assert "provenance" in err and "training" in err


@pytest.mark.parametrize(
    ("file", "split"), [("p/tune.wav", "tune"), ("p/test.wav", "tune"), ("m/test.wav", "test")]
)
def test_a_tune_or_test_entry_in_the_provenance_is_refused_whatever_the_split_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], file: str, split: str
) -> None:
    # The recordings are written before the verifier, so their digests are known here.
    write_manifest(
        tmp_path,
        [
            entry("p/train.wav", split="train", session="s-train"),
            entry("p/tune.wav", split="tune", session="s-tune"),
            entry("p/test.wav", split="test", session="s-test"),
            entry("m/test.wav", "media", "test", phrase="", session="s-test-media"),
        ],
    )
    listed = [entry_sha256(tmp_path, file)]

    err = provenance_refused(tmp_path, capsys, split, listed=listed)

    assert "1 of the 3 tune and test entries" in err
    assert listed[0] not in err


def test_a_train_entry_in_the_provenance_is_what_it_should_hold(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_manifest(tmp_path, [entry("p/train.wav", split="train", session="s-train")])
    listed = [entry_sha256(tmp_path, "p/train.wav")]
    path, models_dir, _ = small_setup(tmp_path, verifier={"listed": listed})
    out = tmp_path / "eval.json"

    assert rig([]).run(harness_args(path, models_dir, out, "tune", "0.5")) == 0

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["provenance"] == {
        "files": 2,
        "sources": {
            "synthetic": {"files": 3, "hours": 1.25},
            "train-split": {"files": 1, "hours": 0.5},
        },
        "checked": {"tune": 1, "test": 2},
    }


def test_a_provenance_file_other_than_the_one_the_manifest_pins_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path, models_dir, _ = small_setup(tmp_path)
    provenance = models_dir / "speech" / "wake" / VERIFIER_ID / "provenance.json"
    provenance.write_text(provenance.read_text(encoding="utf-8") + " ", encoding="utf-8")
    out = tmp_path / "eval.json"
    engines = rig([])

    code = engines.run(harness_args(path, models_dir, out, "tune", "0.5"))

    err = capsys.readouterr().err
    assert code == 2
    assert "provenance" in err
    assert not out.exists()
    assert engines.placements == []


def test_a_verifier_for_another_spotter_model_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    err = provenance_refused(tmp_path, capsys, spotter_sha256="f" * 64)

    assert MODEL_ID in err


def test_the_real_engines_load_the_spotter_the_vad_and_the_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class Spotter:
        @staticmethod
        def load(placement: PlacementConfig, *, library_loader: object) -> str:
            seen["spotter"] = (placement, library_loader)
            return "spotter"

    class Vad:
        @staticmethod
        def load(
            placement: PlacementConfig, *, session_factory: object, available_providers: object
        ) -> str:
            seen["vad"] = (placement, session_factory, available_providers)
            return "vad"

    class Verifier:
        @staticmethod
        def load(
            placement: PlacementConfig, *, session_factory: object, available_providers: object
        ) -> str:
            seen["verifier"] = (placement, session_factory, available_providers)
            return "verifier"

    monkeypatch.setattr(eval_wake, "MicroNameSpotter", Spotter)
    monkeypatch.setattr(eval_wake, "SileroVad", Vad)
    monkeypatch.setattr(eval_wake, "OnnxWakeVerifier", Verifier)
    placement = PlacementConfig(models_dir="/models", kws_model=MODEL_ID)

    engines = eval_wake.real_engines(VERIFIER_ID)(placement)

    assert engines == eval_wake.Engines(spotter="spotter", vad="vad", verifier="verifier")  # type: ignore[arg-type]
    onnx = (eval_wake.open_onnx_session, eval_wake.onnxruntime_providers)
    assert seen == {
        "spotter": (placement, eval_wake.import_micro_runtime),
        "vad": (placement, *onnx),
        "verifier": (
            PlacementConfig(models_dir="/models", kws_model=MODEL_ID, verify_model=VERIFIER_ID),
            *onnx,
        ),
    }


def test_the_engines_are_loaded_with_the_model_and_the_verifier_under_test(
    tmp_path: Path,
) -> None:
    path, models_dir, _ = small_setup(tmp_path)
    engines = rig([])
    argv = harness_args(path, models_dir, tmp_path / "eval.json", "tune", "0.5")

    assert engines.run(argv) == 0

    (placement,) = engines.placements
    assert (placement.kws_model, placement.verify_model) == (MODEL_ID, VERIFIER_ID)
    assert placement.models_dir == str(models_dir)


# -- The harness: the tuning pass -------------------------------------------------------------


def tune_setup(tmp_path: Path) -> tuple[list[str], Rig, Path, list[dict[str, object]]]:
    """Four positives, two homophones and two media clips in `tune`, around other splits.

    Positives: p1 "hey mist" (headset, 0.65), p2 "hey mist" (desk, 0.45), p3 "mist" alone
    (headset, 0.55, 640 ms of speech then 640 ms of silence, the hit 100 ms after the speech
    end) and p4 "are you there mist" (desk, 0.35, a window the verifier rejects). Homophones h1
    and h2 both score 0.6; only h1's window passes the verifier. Media m1 scores 0.5 and passes;
    m2 scores 0.7 but its reference holds the name at 0.9, so the gate rejects it as self media
    before its window is scored. The first three windows take 100, 200 and 300 ms.
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
    write_verifier(models_dir, training=training)
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
    scores = [ADDRESSED, ADDRESSED, ADDRESSED, UNADDRESSED, ADDRESSED, UNADDRESSED, ADDRESSED]
    engines = rig(hits, scores, verify_ms=[100.0, 200.0, 300.0])
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
    # Nothing of what a transcript wrote: the rates, the files and the gate's reasons.
    assert set(low) == {"overall", "by_condition", "by_phrase", "files", "reasons"}
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


def test_each_hit_position_is_scored_once_whatever_the_threshold(tmp_path: Path) -> None:
    argv, engines, out, _ = tune_setup(tmp_path)

    assert engines.run(argv) == 0

    # Seven hits reach the verifier, each at up to five thresholds, and each window is scored
    # once: m2 is rejected as self media first, and the sweep reuses the rest.
    assert len(engines.verifier.windows) == 7
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["verifier_id"] == VERIFIER_ID
    assert report["verify"] == {
        "count": 7,
        "failures": 0,
        "short_windows": 0,
        "wall_s": 0.8,
        "p50_ms": 50.0,
        "p90_ms": 240.0,
    }
    # The verifier's after_ms, 300, plus the measured verify times.
    assert report["accept_ms"] == {"p50": 350.0, "p90": 540.0, "after_ms": 300}


def test_a_hit_rejected_before_verification_is_never_scored(tmp_path: Path) -> None:
    # The reference holds the name at 0.9, so the gate rejects p1's hit as self media.
    recordings = tmp_path / "recordings"
    write_wav(recordings / "p" / "ref.wav", tone_ms=1000, constant=REFERENCE_SAMPLE)
    entries = [
        entry("p/train.wav", split="train", session="s-train"),
        entry("p/p1.wav", session="s-tune", reference_file="p/ref.wav"),
    ]
    path = write_manifest(tmp_path, entries)
    training = wake_manifest.subset_sha256(wake_manifest.load_manifest(path), "train")
    write_model(tmp_path / "models", training=training)
    write_verifier(tmp_path / "models", training=training)
    engines = rig([hit(offsets(entries, "tune"), "p/p1.wav", 25, 0.7)], [ADDRESSED])
    out = tmp_path / "eval.json"

    assert engines.run(harness_args(path, tmp_path / "models", out, "tune", "0.5")) == 0

    report = json.loads(out.read_text(encoding="utf-8"))
    (row,) = report["thresholds"]
    assert row["false_reject"]["reasons"] == {"self_media": 1}
    assert engines.verifier.windows == []
    assert report["verify"]["count"] == 0
    assert report["accept_ms"] == {"p50": None, "p90": None, "after_ms": 300}


def test_the_tuning_pass_writes_latency_and_the_measured_hit_lag(tmp_path: Path) -> None:
    argv, engines, out, _ = tune_setup(tmp_path)

    assert engines.run(argv) == 0

    rows = by_threshold(json.loads(out.read_text(encoding="utf-8")))
    # Accepted positives at 0.3: p1, p2 and p3, verified in 100, 200 and 300 ms after the
    # verifier's 300 ms after the hit; at 0.5, p1 and p3.
    assert rows[0.3]["wake_accept_ms"] == {"p50": 500.0, "p90": 580.0, "count": 3}
    assert rows[0.5]["wake_accept_ms"] == {"p50": 500.0, "p90": 580.0, "count": 2}
    assert rows[0.7]["wake_accept_ms"] == {"p50": None, "p90": None, "count": 0}
    # p3's hit ends at 740 ms and its speech at 640 ms.
    for threshold in (0.3, 0.4, 0.5):
        assert rows[threshold]["hit_lag_ms"] == {"median": 100, "p90": 100.0, "count": 1}
        screen = rows[threshold]["wake_screen_ms"]
        assert screen["hit_lag_ms"] == 100
    # The screen moves at acceptance: each woken positive's hit lag plus its accept time.
    # At 0.3 and 0.4, 100 plus 400, 500 and 600 ms; at 0.5, plus 400 and 600 ms.
    for threshold in (0.3, 0.4):
        screen = rows[threshold]["wake_screen_ms"]
        assert (screen["p50"], screen["p90"], screen["count"]) == (600.0, 680.0, 3)
    screen = rows[0.5]["wake_screen_ms"]
    assert (screen["p50"], screen["p90"], screen["count"]) == (600.0, 680.0, 2)
    assert rows[0.6]["hit_lag_ms"] == {"median": None, "p90": None, "count": 0}


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


def test_no_recording_is_named_in_the_logs_or_the_printed_lines(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv, engines, out, entries = tune_setup(tmp_path)
    # p4's window fails, which is logged.
    engines.verifier.scores[3] = SpeechEngineError("scripted verify failure")
    caplog.set_level(logging.DEBUG)

    assert engines.run(argv) == 0

    captured = capsys.readouterr()
    for item in entries:
        name = Path(str(item["file"])).stem
        assert name not in caplog.text
        assert name not in captured.out
        assert name not in captured.err
    assert "Verify failed" in caplog.text
    assert caplog.records, "the run should log its progress"
    assert json.loads(out.read_text(encoding="utf-8"))["verify"]["failures"] == 1


# -- The driver's entry points for spotter_eval ------------------------------------------------


def test_accepted_wakes_equal_the_tuning_pass_count(tmp_path: Path) -> None:
    argv, engines, out, _ = tune_setup(tmp_path / "pass")
    assert engines.run(argv) == 0
    rows = by_threshold(json.loads(out.read_text(encoding="utf-8")))
    # A fresh rig over the same audio, scripts and positions: the positives lead the tune split,
    # so they sit at the same uplink positions with or without the entries after them.
    _, fresh, _, _ = tune_setup(tmp_path / "direct")
    tune = wake_manifest.load_manifest(tmp_path / "direct" / "manifest.json").split("tune")
    positives = [item for item in tune if item.label == "positive"]
    assert positives == list(tune[: len(positives)])

    counts = eval_wake.VerifyCounts()
    woken = eval_wake.accepted_wakes(
        positives,
        fresh.engines,
        load_voice_settings({}),
        [0.3, 0.4, 0.5, 0.6, 0.7],
        fresh.clock,
        counts=counts,
    )

    reported = {
        threshold: row["false_reject"]["overall"]["total"]
        - row["false_reject"]["overall"]["rejected"]
        for threshold, row in rows.items()
    }
    assert woken == reported == {0.3: 3, 0.4: 3, 0.5: 2, 0.6: 1, 0.7: 0}
    # The four positives' windows, each scored once, in 100, 200, 300 and 50 ms.
    assert (counts.verifies, counts.failures, counts.short_windows) == (4, 0, 0)
    assert counts.times_ms == pytest.approx([100.0, 200.0, 300.0, 50.0])


def tune_positives(tmp_path: Path) -> tuple[list[wake_manifest.Entry], Rig]:
    """`tune_setup`'s four positives, which lead its tune split, and a fresh rig."""
    _, engines, _, _ = tune_setup(tmp_path)
    tune = wake_manifest.load_manifest(tmp_path / "manifest.json").split("tune")
    return [item for item in tune if item.label == "positive"], engines


def test_gate_checks_count_each_positive_s_first_hit(tmp_path: Path) -> None:
    positives, engines = tune_positives(tmp_path)

    checks = eval_wake.gate_checks(
        positives, engines.engines, load_voice_settings({}), 0.3, engines.clock
    )

    # The verifier passes p1 to p3 and rejects p4.
    assert checks.first_hit == {"verified": 3, "not_addressed": 1}
    # p1 to p3 say the name alone ("hey mist", "mist"); p4 says it in a sentence.
    assert checks.first_hit_by_group == {
        "name_alone": {"verified": 3},
        "sentence": {"not_addressed": 1},
    }
    # p3's hit lands 100 ms after its speech; the others' hits are in their tone.
    assert checks.places == (
        ("inside", None),
        ("inside", None),
        ("after", 100),
        ("inside", None),
    )


def test_gate_checks_count_a_positive_with_no_hit(tmp_path: Path) -> None:
    positives, engines = tune_positives(tmp_path)

    checks = eval_wake.gate_checks(
        positives, engines.engines, load_voice_settings({}), 0.5, engines.clock
    )

    # At 0.5 p2 (0.45) and p4 (0.35) have no hit, and so no place. The scores are read in the
    # order the windows are scored, so p3's is the second, which passes.
    assert checks.first_hit == {"no_hit": 2, "verified": 2}
    assert checks.first_hit_by_group == {
        "name_alone": {"no_hit": 1, "verified": 2},
        "sentence": {"no_hit": 1},
    }
    assert len(checks.places) == 2


def test_gate_checks_count_a_first_hit_with_no_segment_before_it(tmp_path: Path) -> None:
    # 2400 ms of silence and then 640 ms of tone: the hit at frame 50 has no segment before it.
    samples = tone_and_silence([(False, 2400), (True, 640)])
    (item,) = wake_manifest.load_manifest(write_recording(tmp_path, samples)).entries
    engines = rig([NameHit(end_pos=50, keyword="mist", score=0.9)], [UNADDRESSED])

    checks = eval_wake.gate_checks(
        [item], engines.engines, load_voice_settings({}), 0.5, engines.clock
    )

    assert checks.places == (("no_segment", None),)
    assert checks.first_hit == {"not_addressed": 1}


def write_ambient(path: Path) -> None:
    """640 ms of silence, 640 ms of tone (frames 32 to 63), then 1200 ms of silence."""
    samples = tone_and_silence([(False, 640), (True, 640), (False, 1200)])
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(samples.tobytes())


# Every ambient file's stream reports a hit at frame 50, in its speech, and one at frame 90, in
# silence after the segment has closed, which is reported and gated too. Their windows differ.
AMBIENT_HITS = (
    NameHit(end_pos=50, keyword="mist", score=0.6),
    NameHit(end_pos=90, keyword="mist", score=0.7),
)


def test_the_ambient_gate_hears_each_file_once_and_scores_each_hit_position_once(
    tmp_path: Path,
) -> None:
    first, second = tmp_path / "ambient" / "a.wav", tmp_path / "ambient" / "b.wav"
    write_ambient(first)
    write_ambient(second)
    # The first file's speech hit is rejected and its silence hit passes; neither of the
    # second file's passes.
    engines = rig(AMBIENT_HITS, [UNADDRESSED, ADDRESSED, UNADDRESSED, UNADDRESSED])
    gate = eval_wake.AmbientGate(engines.engines, load_voice_settings({}), engines.clock)

    assert gate.accepted_hits(first, 0.5) == 1
    assert gate.accepted_hits(first, 0.65) == 1
    assert gate.accepted_hits(first, 0.3) == 1
    assert (len(engines.vad.streams), len(engines.verifier.windows)) == (1, 2)
    assert gate.counts.verifies == 2

    assert gate.accepted_hits(second, 0.5) == 0
    assert (len(engines.vad.streams), len(engines.verifier.windows)) == (2, 4)
    assert (gate.counts.verifies, gate.counts.failures) == (4, 0)

    assert [stream.threshold for stream in engines.spotter.streams] == [0.5, 0.65, 0.3, 0.5]
    for stream in engines.spotter.streams:
        assert stream.pushed == list(range(len(stream.pushed)))
        assert {50, 90} <= set(stream.pushed)


def test_a_failed_ambient_score_is_rejected_counted_and_logged_by_number(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "ambient" / "meeting-secret.wav"
    write_ambient(path)
    failure = SpeechEngineError("scripted verify failure")
    engines = rig(AMBIENT_HITS, [failure, failure])
    gate = eval_wake.AmbientGate(engines.engines, load_voice_settings({}), engines.clock)
    caplog.set_level(logging.DEBUG)

    assert gate.accepted_hits(path, 0.5) == 0
    assert gate.accepted_hits(path, 0.3) == 0

    # Both hits' windows fail once each, and neither is scored again at the second threshold.
    assert (gate.counts.verifies, gate.counts.failures, len(engines.verifier.windows)) == (2, 2, 2)
    assert "ambient file 1" in caplog.text
    assert "meeting-secret" not in caplog.text
    assert str(tmp_path) not in caplog.text


def test_the_ambient_split_counts_hits_near_a_segment_and_the_rest(tmp_path: Path) -> None:
    path = tmp_path / "ambient" / "a.wav"
    write_ambient(path)
    settings = load_voice_settings({})
    heard = eval_wake._listen(wake_manifest.read_pcm16(path), EnergyVad(), settings.endpointing)
    ((onset, last),) = heard.segments
    lead = onset_lead_frames(settings.endpointing)
    assert eval_wake.SPLIT_POST_ROLL_MS == 500
    positions = {
        "no segment": 2,
        "pre-roll": onset - lead,
        "speech": (onset + last) // 2,
        "400 ms past": last + 400 // FRAME_MS,
        "600 ms past": last + 600 // FRAME_MS,
    }
    assert positions["no segment"] < onset - lead
    assert positions["600 ms past"] < len(heard.frames)
    hits = [NameHit(end_pos=pos, keyword="mist", score=0.7) for pos in positions.values()]
    # Every window is rejected: the split counts hits reported, not accepted.
    engines = rig(hits, [UNADDRESSED] * len(hits))
    gate = eval_wake.AmbientGate(engines.engines, settings, engines.clock)

    assert gate.hit_places(path, 0.5) == (3, 2)

    # The split reads the hits `accepted_hits` recorded: no second pass over the file.
    assert gate.accepted_hits(path, 0.5) == 0
    verifies, streams = gate.counts.verifies, len(engines.spotter.streams)
    assert gate.hit_places(path, 0.5) == (3, 2)
    assert (gate.counts.verifies, len(engines.spotter.streams)) == (verifies, streams)
    # Above every hit's score nothing is reported.
    assert gate.hit_places(path, 0.8) == (0, 0)


# -- The harness: the test pass ---------------------------------------------------------------


def test_the_test_pass_reads_the_lag_and_tables_from_the_model_and_writes_none(
    tmp_path: Path,
) -> None:
    path, models_dir, entries = small_setup(tmp_path, hit_lag_ms=120, calibration=TUNED_CALIBRATION)
    at = offsets(entries, "test")
    engines = rig(
        [hit(at, "p/test.wav", 20, 0.65), hit(at, "m/test.wav", 20, 0.4)],
        [ADDRESSED],
        verify_ms=[150.0],
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
    # 300 ms after the hit and a 150 ms verify; the screen moves then, 120 ms of lag later.
    assert row["wake_accept_ms"] == {"p50": 450.0, "p90": 450.0, "count": 1}
    assert row["wake_screen_ms"] == {"p50": 570.0, "p90": 570.0, "count": 1, "hit_lag_ms": 120}
    assert row["media_false_accepts"]["count"] == 0
    assert len(engines.vad.streams) == 2
