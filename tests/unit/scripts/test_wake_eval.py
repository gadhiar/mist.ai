"""Tests for the wake model evaluation harness: the recording manifest and `eval_wake`.

Every WAV here is written by `wave` into pytest's temporary directory, and the engines are the
scripted fakes (`ScriptedNameSpotter`, `ScriptedTranscriber`, `EnergyVad`), so no recording and
no model is read. The model manifest is the microWakeWord format over a placeholder model file,
which `read_manifest` hashes like a real one.

Scripted hits are addressed by uplink position. The harness numbers the frames of a split's
entries in manifest order, one position per 20 ms frame, so `offsets` below turns an entry and a
frame within it into the position a hit is scripted at. Transcripts are scripted in the order
the gate decodes run: entries in manifest order, one decode per distinct span.
"""

import asyncio
import hashlib
import importlib
import json
import logging
import wave
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from backend.realtime.addressee import NameAlignment
from backend.realtime.budget import EndpointingConfig, PlacementConfig, load_voice_settings
from backend.realtime.engines.kws import read_manifest
from backend.realtime.engines.parakeet_stt import OnnxAsrLibrary, ParakeetTranscriber
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
from backend.realtime.speech import CancelToken, HeardWord, NameHit
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
        eval_wake.Engines(spotter=offline, vad=EnergyVad(), transcriber=ScriptedTranscriber([])),
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
    driver = eval_wake._Driver(
        eval_wake.Engines(
            spotter=engines.spotter, vad=engines.vad, transcriber=engines.transcriber
        ),
        settings,
        [0.5],
        engines.clock,
    )

    outcome = driver.run(item, 0)[0.5]

    assert [found.position for found in outcome.hits] == [150]
    assert outcome.accepted
    window = 1280 // FRAME_MS
    tail = settings.wake.gate_tail_ms // FRAME_MS
    # The window and the pre-roll before the hit, and nothing of the unchained segment.
    assert driver._span(outcome.heard, 150) == (150 - window - 15, 150 + tail)
    (decode,) = engines.transcriber.streams
    assert decode.pushed == [(window + 15 + tail + 1) * RATE * FRAME_MS // 1000]
    # Every hit's span starts where ingress's would: the shared chain start, bounded.
    for position in (40, 70, 100, 150, len(outcome.heard.frames) - 1):
        start = eval_wake.name_chain_start(
            outcome.heard.segments,
            position,
            join_gap_ms=settings.wake.join_gap_ms,
            pre_roll_frames=onset_lead_frames(settings.endpointing),
            window_frames=window,
        )
        bounded = max(start, position - settings.wake.lookback_ms // FRAME_MS, 0)
        end = min(position + tail, len(outcome.heard.frames) - 1)
        assert driver._span(outcome.heard, position) == (bounded, end)


def test_the_harness_keeps_the_last_speech_frame_heard_by_each_frame() -> None:
    # Silence (frames 0 to 31), tone (32 to 63), then silence again (64 to 123).
    samples = tone_and_silence([(False, 640), (True, 640), (False, 1200)])
    cfg = EndpointingConfig()

    heard = eval_wake._listen(samples.tobytes(), EnergyVad(), cfg)

    # Ingress's rule, from the windows themselves: a window is scored once the frame holding
    # its last sample arrives, and a speech window's last frame is the latest speech.
    scorer = EnergyVad().open_stream()
    floats = samples.astype(np.float32) / 32768.0
    expected = []
    latest = -1
    window = 0
    for pos in range(len(heard.frames)):
        while (window + 1) * 512 <= (pos + 1) * 320:
            start = window * 512
            if scorer.score(floats[start : start + 512]) >= cfg.vad_threshold:
                latest = (start + 511) // 320
            window += 1
        expected.append(latest)
    assert heard.last_speech == tuple(expected)
    assert heard.last_speech[31] == -1
    assert heard.last_speech[-1] == heard.segments[-1][1]
    assert all(value <= pos for pos, value in enumerate(heard.last_speech))


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


def spy_alignments(monkeypatch: pytest.MonkeyPatch) -> list[NameAlignment | None]:
    """Every alignment the gate's `decide` is given, in order."""
    seen: list[NameAlignment | None] = []
    original = eval_wake.AddresseeGate.decide

    def deciding(self: object, **kwargs: object) -> object:
        seen.append(kwargs.get("alignment"))  # type: ignore[arg-type]
        return original(self, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(eval_wake.AddresseeGate, "decide", deciding)
    return seen


@pytest.mark.parametrize(
    ("parts", "hit_at", "speech"),
    [
        # A hit in the silence after the tone: the speech ended before it.
        ([(False, 640), (True, 640), (False, 2400)], 70, "ended"),
        # A hit in the tone: speech goes on past it, through the gate tail.
        ([(False, 640), (True, 2000), (False, 1200)], 60, "going on"),
        # No speech at all: no speech end is known.
        ([(False, 3000)], 80, "none"),
    ],
)
def test_the_gate_reads_the_decode_s_words_where_the_spotter_heard_the_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parts: list[tuple[bool, int]],
    hit_at: int,
    speech: str,
) -> None:
    seen = spy_alignments(monkeypatch)
    (item,) = wake_manifest.load_manifest(
        write_recording(tmp_path, tone_and_silence(parts))
    ).entries
    settings = load_voice_settings({})
    spotter = ScriptedNameSpotter([NameHit(end_pos=hit_at, keyword="mist", score=0.9)])
    # Every word is the name, 200 ms each from the decode's first sample.
    transcriber = ScriptedTranscriber([" ".join(["mist"] * 60)], ms_per_word=200)
    driver = eval_wake._Driver(
        eval_wake.Engines(spotter=spotter, vad=EnergyVad(), transcriber=transcriber),
        settings,
        [0.5],
        FakeClock(),
    )

    outcome = driver.run(item, 0)[0.5]

    (found,) = outcome.hits
    start, end = driver._span(outcome.heard, hit_at)
    count = (end - start + 1) * FRAME_MS // 200
    words = tuple(HeardWord("mist", start + 10 * i, start + 10 * i + 9) for i in range(count))
    last = outcome.heard.last_speech[end]
    speech_end = last if last >= 0 else None
    assert seen == [NameAlignment(hit_at, FRAME_MS, speech_end, words)]
    if speech == "ended":
        assert speech_end is not None and speech_end < hit_at
    elif speech == "going on":
        assert speech_end is not None and speech_end > hit_at + 5
    else:
        assert speech_end is None
    # The words through the gate tail cover the hit, and every one is the name.
    assert found.at_hit == found.verdict.at_hit == "name"


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


def test_the_real_engines_load_the_microwakeword_spotter_and_the_named_stt(
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
        def load(placement: PlacementConfig, **kwargs: object) -> str:
            seen["vad"] = placement
            return "vad"

    def stt(placement: PlacementConfig) -> str:
        seen["stt"] = placement
        return "stt"

    monkeypatch.setattr(eval_wake, "MicroNameSpotter", Spotter)
    monkeypatch.setattr(eval_wake, "SileroVad", Vad)
    monkeypatch.setattr(eval_wake, "parakeet_cpu", stt)
    placement = PlacementConfig(models_dir="/models", kws_model=MODEL_ID)

    engines = eval_wake.real_engines("scripts.wake.eval_wake:parakeet_cpu")(placement)

    assert engines == eval_wake.Engines(spotter="spotter", vad="vad", transcriber="stt")  # type: ignore[arg-type]
    assert seen == {
        "spotter": (placement, eval_wake.import_micro_runtime),
        "vad": placement,
        "stt": placement,
    }


def test_parakeet_cpu_loads_parakeet_from_the_placement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights = tmp_path / "models" / "speech" / "istupakov" / "parakeet-tdt-0.6b-v2-onnx"
    weights.mkdir(parents=True)
    (weights / "vocab.txt").write_text("placeholder", encoding="utf-8")
    loads: list[tuple[tuple, dict]] = []
    timestamped: list[object] = []

    class Options:
        intra_op_num_threads = 0
        inter_op_num_threads = 0

    class Model:
        """onnx-asr's text adapter: the transcriber asks it for the timestamped one."""

        def with_timestamps(self) -> object:
            model = object()
            timestamped.append(model)
            return model

    def load_model(*args: object, **kwargs: object) -> Model:
        loads.append((args, kwargs))
        return Model()

    def fake_import() -> OnnxAsrLibrary:
        return OnnxAsrLibrary(load_model=load_model, session_options=Options, errors=())

    monkeypatch.setattr(eval_wake, "import_onnx_asr", fake_import)
    factory = importlib.import_module("scripts.wake.eval_wake").parakeet_cpu

    transcriber = factory(PlacementConfig(models_dir=str(tmp_path / "models")))

    assert isinstance(transcriber, ParakeetTranscriber)
    ((_, kwargs),) = loads
    assert kwargs["path"] == str(weights)
    assert len(timestamped) == 1


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


def test_the_report_counts_the_word_at_each_missed_positive_s_first_hit(tmp_path: Path) -> None:
    argv, engines, out, _ = tune_setup(tmp_path)

    assert engines.run(argv) == 0

    rows = by_threshold(json.loads(out.read_text(encoding="utf-8")))
    # At 0.3 p4 is missed on its decode, whose words the scripted STT times at the span's
    # start, inside the alias window before its hit at frame 20, and none is a name.
    assert rows[0.3]["false_reject"]["at_hit"] == {"other": 1}
    # From 0.4 up the missed positives have no hit, so no word at one.
    for threshold in (0.4, 0.5, 0.6, 0.7):
        assert rows[threshold]["false_reject"]["at_hit"] == {}


def test_a_hit_rejected_before_its_decode_has_no_timings(tmp_path: Path) -> None:
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
    engines = rig([hit(offsets(entries, "tune"), "p/p1.wav", 25, 0.7)], [ADDRESSED])
    out = tmp_path / "eval.json"

    assert engines.run(harness_args(path, tmp_path / "models", out, "tune", "0.5")) == 0

    (row,) = json.loads(out.read_text(encoding="utf-8"))["thresholds"]
    assert row["false_reject"]["reasons"] == {"self_media": 1}
    assert row["false_reject"]["at_hit"] == {"no_timings": 1}


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

    woken = eval_wake.accepted_wakes(
        positives,
        eval_wake.Engines(spotter=fresh.spotter, vad=fresh.vad, transcriber=fresh.transcriber),
        load_voice_settings({}),
        [0.3, 0.4, 0.5, 0.6, 0.7],
        fresh.clock,
    )

    reported = {
        threshold: row["false_reject"]["overall"]["total"]
        - row["false_reject"]["overall"]["rejected"]
        for threshold, row in rows.items()
    }
    assert woken == reported == {0.3: 3, 0.4: 3, 0.5: 2, 0.6: 1, 0.7: 0}


def tune_positives(tmp_path: Path) -> tuple[list[wake_manifest.Entry], Rig]:
    """`tune_setup`'s four positives, which lead its tune split, and a fresh rig."""
    _, engines, _, _ = tune_setup(tmp_path)
    tune = wake_manifest.load_manifest(tmp_path / "manifest.json").split("tune")
    return [item for item in tune if item.label == "positive"], engines


def test_gate_checks_count_each_positive_s_first_hit(tmp_path: Path) -> None:
    positives, engines = tune_positives(tmp_path)

    checks = eval_wake.gate_checks(
        positives,
        eval_wake.Engines(
            spotter=engines.spotter, vad=engines.vad, transcriber=engines.transcriber
        ),
        load_voice_settings({}),
        0.3,
        engines.clock,
    )

    # p1 and p2 open with "hey mist", p3 says "mist" alone and p4's decode is unaddressed.
    assert checks.first_hit == {"phrase": 2, "vocative": 1, "not_addressed": 1}
    # p1 to p3 say the name alone ("hey mist", "mist"); p4 says it in a sentence.
    assert checks.first_hit_by_group == {
        "name_alone": {"phrase": 2, "vocative": 1},
        "sentence": {"not_addressed": 1},
    }
    # The scripted STT times every word at its span's start: frame 0, in the alias window of a
    # hit at frame 20 (p1, p2, p4) and before the window of p3's hit at frame 36.
    assert checks.at_hit == {"name": 2, "other": 1, "none": 1}
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
        positives,
        eval_wake.Engines(
            spotter=engines.spotter, vad=engines.vad, transcriber=engines.transcriber
        ),
        load_voice_settings({}),
        0.5,
        engines.clock,
    )

    # At 0.5 p2 (0.45) and p4 (0.35) have no hit, and so no word or place at one. The script
    # is read in decode order, so p3's decode is the second, "hey mist quokka".
    assert checks.first_hit == {"no_hit": 2, "phrase": 2}
    assert checks.first_hit_by_group == {
        "name_alone": {"no_hit": 1, "phrase": 2},
        "sentence": {"no_hit": 1},
    }
    assert sum(checks.at_hit.values()) == len(checks.places) == 2


def test_gate_checks_count_a_first_hit_with_no_segment_before_it(tmp_path: Path) -> None:
    # 2400 ms of silence and then 640 ms of tone: the hit at frame 50 has no segment before it.
    samples = tone_and_silence([(False, 2400), (True, 640)])
    (item,) = wake_manifest.load_manifest(write_recording(tmp_path, samples)).entries
    engines = rig([NameHit(end_pos=50, keyword="mist", score=0.9)], [])

    checks = eval_wake.gate_checks(
        [item],
        eval_wake.Engines(
            spotter=engines.spotter, vad=engines.vad, transcriber=engines.transcriber
        ),
        load_voice_settings({}),
        0.5,
        engines.clock,
    )

    assert checks.places == (("no_segment", None),)
    # The scripted decode past its script is empty: no word at the hit.
    assert checks.at_hit == {"none": 1}
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
# silence after the segment has closed, which is reported and gated too. Their spans differ.
AMBIENT_HITS = (
    NameHit(end_pos=50, keyword="mist", score=0.6),
    NameHit(end_pos=90, keyword="mist", score=0.7),
)


def test_the_ambient_gate_hears_each_file_once_and_decodes_each_span_once(
    tmp_path: Path,
) -> None:
    first, second = tmp_path / "ambient" / "a.wav", tmp_path / "ambient" / "b.wav"
    write_ambient(first)
    write_ambient(second)
    # The first file's speech hit decodes unaddressed and its silence hit addressed; the second
    # file's decodes are past the script, so empty.
    engines = rig(AMBIENT_HITS, [UNADDRESSED, ADDRESSED])
    gate = eval_wake.AmbientGate(
        eval_wake.Engines(
            spotter=engines.spotter, vad=engines.vad, transcriber=engines.transcriber
        ),
        load_voice_settings({}),
        engines.clock,
    )

    assert gate.accepted_hits(first, 0.5) == 1
    assert gate.accepted_hits(first, 0.65) == 1
    assert gate.accepted_hits(first, 0.3) == 1
    assert (len(engines.vad.streams), len(engines.transcriber.streams), gate.decodes) == (1, 2, 2)

    assert gate.accepted_hits(second, 0.5) == 0
    assert (len(engines.vad.streams), len(engines.transcriber.streams), gate.decodes) == (2, 4, 4)
    assert gate.decode_errors == 0

    assert [stream.threshold for stream in engines.spotter.streams] == [0.5, 0.65, 0.3, 0.5]
    for stream in engines.spotter.streams:
        assert stream.pushed == list(range(len(stream.pushed)))
        assert {50, 90} <= set(stream.pushed)


def test_a_failed_ambient_decode_is_rejected_counted_and_logged_by_number(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "ambient" / "meeting-secret.wav"
    write_ambient(path)
    clock = FakeClock()

    def failing() -> None:
        raise SpeechEngineError("scripted decode failure")

    spotter = ScriptedNameSpotter(AMBIENT_HITS)
    transcriber = ScriptedTranscriber([ADDRESSED], ms_per_word=1, on_decode=failing)
    gate = eval_wake.AmbientGate(
        eval_wake.Engines(spotter=spotter, vad=EnergyVad(), transcriber=transcriber),
        load_voice_settings({}),
        clock,
    )
    caplog.set_level(logging.DEBUG)

    assert gate.accepted_hits(path, 0.5) == 0
    assert gate.accepted_hits(path, 0.3) == 0

    # Both hits' spans fail once each, and neither is decoded again at the second threshold.
    assert (gate.decodes, gate.decode_errors, len(transcriber.streams)) == (2, 2, 2)
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
    # Every decode is past the script, so empty: the split counts hits reported, not accepted.
    engines = rig(hits, [])
    gate = eval_wake.AmbientGate(
        eval_wake.Engines(
            spotter=engines.spotter, vad=engines.vad, transcriber=engines.transcriber
        ),
        settings,
        engines.clock,
    )

    assert gate.hit_places(path, 0.5) == (3, 2)

    # The split reads the hits `accepted_hits` recorded: no second pass over the file.
    assert gate.accepted_hits(path, 0.5) == 0
    decodes, streams = gate.decodes, len(engines.spotter.streams)
    assert gate.hit_places(path, 0.5) == (3, 2)
    assert (gate.decodes, len(engines.spotter.streams)) == (decodes, streams)
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
