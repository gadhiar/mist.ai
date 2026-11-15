"""Tests for voice identities and the keyed artefact cache in `backend/realtime/voice_assets.py`.

An identity is the private set of clips MIST's voice is cloned from, so a bad one has to fail
when the voice is configured, naming the field or the relative path, and never at a turn. The
cache sits on a volume several projects can share, so its lock, sweep and load rules are pinned
here as behaviour. Every identity is synthetic: sine tones written with `wave` into `tmp_path`.
"""

import builtins
import dataclasses
import fcntl
import hashlib
import io
import json
import logging
import os
import struct
import threading
import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from backend.realtime import voice_assets
from backend.realtime.errors import SpeechEngineError
from backend.realtime.speech import SPEAKING_STATES
from backend.realtime.voice_assets import (
    MANIFEST_SCHEMA,
    VOICE_MANIFEST,
    VoiceAssetError,
    VoiceClip,
    VoiceIdentity,
    artefact_key,
    ensure_artefacts,
    load_identity,
)

VOICE_ID = "test-voice"
TRANSCRIPT_MARK = "zebra quartz"
"""A distinctive transcript string no log record or error text may hold."""
LINES = {
    "neutral": f"test line one {TRANSCRIPT_MARK}",
    "calm": f"test line two {TRANSCRIPT_MARK}",
    "sleepy": f"test line three {TRANSCRIPT_MARK}",
}
TONES = {"neutral": 220.0, "calm": 330.0, "sleepy": 440.0}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_wav(
    path: Path,
    *,
    seconds: float = 4.0,
    rate: int = 24000,
    channels: int = 1,
    frequency: float = 220.0,
) -> None:
    t = np.arange(int(seconds * rate)) / rate
    tone = (0.3 * np.sin(2 * np.pi * frequency * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(np.repeat(tone, channels).tobytes())


def _write_float_wav(path: Path, *, seconds: float = 4.0, rate: int = 24000) -> None:
    t = np.arange(int(seconds * rate)) / rate
    data = (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype("<f4").tobytes()
    fmt = struct.pack("<HHIIHH", 3, 1, rate, rate * 4, 4, 32)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(data)) + data
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


def _read_manifest(identity: Path) -> dict:
    return json.loads((identity / VOICE_MANIFEST).read_text(encoding="utf-8"))


def _write_manifest(identity: Path, manifest: dict) -> None:
    (identity / VOICE_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")


def _refresh_hashes(identity: Path, manifest: dict, index: int) -> None:
    clip = manifest["clips"][index]
    clip["audio_sha256"] = _sha(identity / clip["audio"])
    clip["transcript_sha256"] = _sha(identity / clip["transcript"])


def make_identity(
    voice_root: Path,
    states: tuple[str, ...] = ("neutral",),
    *,
    voice_id: str = VOICE_ID,
    design_log: str | None = None,
) -> Path:
    identity = voice_root / voice_id
    identity.mkdir(parents=True)
    clips = []
    for state in states:
        _write_wav(identity / f"{state}.wav", frequency=TONES.get(state, 550.0))
        (identity / f"{state}.txt").write_text(LINES.get(state, "test line"), encoding="utf-8")
        clips.append(
            {
                "state": state,
                "audio": f"{state}.wav",
                "transcript": f"{state}.txt",
                "audio_sha256": _sha(identity / f"{state}.wav"),
                "transcript_sha256": _sha(identity / f"{state}.txt"),
            }
        )
    manifest: dict = {"schema": 1, "voice_id": voice_id, "status": "placeholder", "clips": clips}
    if design_log is not None:
        (identity / design_log).write_text("design notes", encoding="utf-8")
        manifest["design_log"] = design_log
    _write_manifest(identity, manifest)
    return identity


def _listing(root: Path) -> list[tuple[str, int, int]]:
    return sorted(
        (str(path.relative_to(root)), path.lstat().st_size, path.lstat().st_mtime_ns)
        for path in root.rglob("*")
    )


def _lock_is_held(cache_root: Path) -> bool:
    """A non-blocking `flock` from a second file descriptor fails while the lock is held."""
    fd = os.open(cache_root / ".lock", os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


class RecordingBuilder:
    """A `VoiceArtefactBuilder` that writes one small file per clip and records every call."""

    def __init__(
        self,
        *,
        engine_id: str = "fake-engine",
        model_pin: str = "test/model@abc123",
        artefact_version: int = 1,
        on_build: Callable[[VoiceClip, Path], object] | None = None,
        on_load: Callable[[Path], object] | None = None,
    ) -> None:
        self.engine_id = engine_id
        self.model_pin = model_pin
        self.artefact_version = artefact_version
        self.on_build = on_build
        self.on_load = on_load
        self.builds: list[str] = []
        self.loads: list[Path] = []

    def build(self, clip: VoiceClip, out_dir: Path) -> None:
        self.builds.append(clip.state)
        (out_dir / "artefact.bin").write_bytes(clip.audio_sha256.encode("ascii"))
        if self.on_build is not None:
            self.on_build(clip, out_dir)

    def load(self, artefact_dir: Path) -> tuple[str, tuple[str, ...]]:
        self.loads.append(artefact_dir)
        if self.on_load is not None:
            self.on_load(artefact_dir)
        return artefact_dir.parent.name, tuple(sorted(os.listdir(artefact_dir)))


@pytest.fixture
def voice_root(tmp_path: Path) -> Path:
    root = tmp_path / "voices"
    root.mkdir()
    return root


@pytest.fixture
def cache_root(tmp_path: Path) -> Path:
    return tmp_path / "voice-cache"


class TestLoadIdentity:
    def test_a_one_clip_identity_loads_with_its_neutral_clip(self, voice_root):
        identity_dir = make_identity(voice_root)

        identity = load_identity(voice_root, VOICE_ID)

        assert identity.voice_id == VOICE_ID
        assert identity.status == "placeholder"
        assert identity.states == frozenset({"neutral"})
        assert identity.root == identity_dir.resolve()
        (clip,) = identity.clips
        assert clip.audio == (identity_dir / "neutral.wav").resolve()
        assert clip.transcript == (identity_dir / "neutral.txt").resolve()
        assert clip.audio_sha256 == _sha(identity_dir / "neutral.wav")
        assert clip.transcript_sha256 == _sha(identity_dir / "neutral.txt")
        assert (clip.sample_rate, clip.duration_ms) == (24000, 4000)

    def test_a_three_state_identity_loads_with_its_tags_as_states(self, voice_root):
        make_identity(voice_root, ("neutral", "calm", "sleepy"))

        identity = load_identity(voice_root, VOICE_ID)

        assert identity.states == frozenset({"neutral", "calm", "sleepy"})
        assert identity.clip("calm").state == "calm"
        assert identity.clip("sleepy").audio.name == "sleepy.wav"

    def test_an_unknown_state_gives_the_neutral_clip(self, voice_root):
        make_identity(voice_root, ("neutral", "calm", "sleepy"))
        identity = load_identity(voice_root, VOICE_ID)

        assert identity.clip("excited") == identity.clip("neutral")
        assert identity.clip("furious").state == "neutral"

    def test_a_frozen_status_and_a_present_design_log_load(self, voice_root):
        identity_dir = make_identity(voice_root, design_log="design-log.md")
        manifest = _read_manifest(identity_dir)
        manifest["status"] = "frozen"
        _write_manifest(identity_dir, manifest)

        assert load_identity(voice_root, VOICE_ID).status == "frozen"

    @pytest.mark.parametrize(
        ("seconds", "rate"),
        [
            pytest.param(3.0, 16000, id="shortest-at-the-lowest-rate"),
            pytest.param(30.0, 48000, id="longest-at-the-highest-rate"),
        ],
    )
    def test_the_audio_bounds_are_inclusive(self, voice_root, seconds, rate):
        identity_dir = make_identity(voice_root)
        _write_wav(identity_dir / "neutral.wav", seconds=seconds, rate=rate)
        manifest = _read_manifest(identity_dir)
        _refresh_hashes(identity_dir, manifest, 0)
        _write_manifest(identity_dir, manifest)

        clip = load_identity(voice_root, VOICE_ID).clip("neutral")

        assert (clip.sample_rate, clip.duration_ms) == (rate, int(seconds * 1000))

    def test_the_value_types_are_frozen_and_slotted(self):
        for cls in (VoiceClip, VoiceIdentity):
            assert cls.__dataclass_params__.frozen
            assert "__slots__" in cls.__dict__

    def test_the_error_is_a_speech_engine_error(self):
        assert issubclass(VoiceAssetError, SpeechEngineError)
        assert (VOICE_MANIFEST, MANIFEST_SCHEMA) == ("voice.json", 1)


# -- every way an identity can be bad --------------------------------------------------------


def _edit(fn: Callable[[Path, dict], object]) -> Callable[[Path], None]:
    def apply(identity: Path) -> None:
        manifest = _read_manifest(identity)
        fn(identity, manifest)
        _write_manifest(identity, manifest)

    return apply


def _no_manifest(identity: Path) -> None:
    (identity / VOICE_MANIFEST).unlink()


def _clip0(**changes: object) -> Callable[[Path], None]:
    return _edit(lambda identity, manifest: manifest["clips"][0].update(changes))


def _symlink_out(identity: Path) -> None:
    outside = identity.parent.parent / "outside.wav"
    _write_wav(outside)
    os.symlink(outside, identity / "link.wav")
    manifest = _read_manifest(identity)
    manifest["clips"][0].update(audio="link.wav", audio_sha256=_sha(outside))
    _write_manifest(identity, manifest)


def _absolute(identity: Path) -> None:
    manifest = _read_manifest(identity)
    manifest["clips"][0]["audio"] = str((identity / "neutral.wav").resolve())
    _write_manifest(identity, manifest)


def _rewrite_audio(writer: Callable[[Path], None]) -> Callable[[Path], None]:
    def apply(identity: Path) -> None:
        writer(identity / "neutral.wav")
        manifest = _read_manifest(identity)
        _refresh_hashes(identity, manifest, 0)
        _write_manifest(identity, manifest)

    return apply


def _rewrite_transcript(data: bytes) -> Callable[[Path], None]:
    def apply(identity: Path) -> None:
        (identity / "neutral.txt").write_bytes(data)
        manifest = _read_manifest(identity)
        _refresh_hashes(identity, manifest, 0)
        _write_manifest(identity, manifest)

    return apply


BAD_IDENTITIES = [
    pytest.param(_no_manifest, "voice.json", id="no-manifest"),
    pytest.param(
        lambda identity: (identity / VOICE_MANIFEST).write_text("{not json", encoding="utf-8"),
        "voice.json",
        id="manifest-not-json",
    ),
    pytest.param(_edit(lambda i, m: m.update(schema=2)), "schema", id="schema-not-1"),
    pytest.param(_edit(lambda i, m: m.update(schema=True)), "schema", id="schema-a-bool"),
    pytest.param(
        _edit(lambda i, m: m.update(voice_id="other-voice")), "voice_id", id="voice-id-not-dir"
    ),
    pytest.param(_edit(lambda i, m: m.update(status="draft")), "status", id="status-unknown"),
    pytest.param(
        _edit(lambda i, m: m["clips"][1].update(state="furious")),
        "clips[1].state",
        id="state-outside-vocabulary",
    ),
    pytest.param(
        _edit(lambda i, m: m["clips"][1].update(state="neutral")),
        "clips[1].state",
        id="duplicate-state",
    ),
    pytest.param(
        _edit(lambda i, m: m["clips"][0].update(state="sleepy")), "neutral", id="no-neutral-clip"
    ),
    pytest.param(_edit(lambda i, m: m.update(clips=[])), "clips", id="no-clips"),
    pytest.param(_clip0(audio="../x.wav"), "../x.wav", id="parent-path"),
    pytest.param(_absolute, "clips[0].audio", id="absolute-path"),
    pytest.param(_symlink_out, "link.wav", id="symlink-out"),
    pytest.param(_clip0(audio="missing.wav"), "missing.wav", id="missing-audio"),
    pytest.param(_clip0(transcript="missing.txt"), "missing.txt", id="missing-transcript"),
    pytest.param(_clip0(audio_sha256="0" * 64), "audio_sha256", id="audio-sha-mismatch"),
    pytest.param(
        _clip0(transcript_sha256="0" * 64), "transcript_sha256", id="transcript-sha-mismatch"
    ),
    pytest.param(_clip0(audio_sha256="xyz"), "audio_sha256", id="audio-sha-not-hex"),
    pytest.param(
        _rewrite_audio(lambda path: _write_wav(path, channels=2)), "neutral.wav", id="stereo"
    ),
    pytest.param(_rewrite_audio(_write_float_wav), "neutral.wav", id="float-wav"),
    pytest.param(
        _rewrite_audio(lambda path: path.write_bytes(b"RIFF....not a wav")),
        "neutral.wav",
        id="not-a-wav",
    ),
    pytest.param(
        _rewrite_audio(lambda path: _write_wav(path, rate=8000)), "neutral.wav", id="rate-low"
    ),
    pytest.param(
        _rewrite_audio(lambda path: _write_wav(path, rate=96000)), "neutral.wav", id="rate-high"
    ),
    pytest.param(
        _rewrite_audio(lambda path: _write_wav(path, seconds=2.5)), "neutral.wav", id="too-short"
    ),
    pytest.param(
        _rewrite_audio(lambda path: _write_wav(path, seconds=30.5, rate=16000)),
        "neutral.wav",
        id="too-long",
    ),
    pytest.param(_rewrite_transcript(b""), "neutral.txt", id="empty-transcript"),
    pytest.param(_rewrite_transcript(b" \n\t"), "neutral.txt", id="blank-transcript"),
    pytest.param(_rewrite_transcript(b"\xff\xfe\x00"), "neutral.txt", id="transcript-not-utf8"),
    pytest.param(
        _edit(lambda i, m: m.update(design_log="design-log.md")),
        "design-log.md",
        id="design-log-missing",
    ),
]


class TestBadIdentities:
    @pytest.mark.parametrize(("spoil", "named"), BAD_IDENTITIES)
    def test_a_bad_identity_raises_naming_the_field_or_the_relative_path(
        self, voice_root, spoil, named
    ):
        identity_dir = make_identity(voice_root, ("neutral", "calm"))
        spoil(identity_dir)

        with pytest.raises(VoiceAssetError) as caught:
            load_identity(voice_root, VOICE_ID)

        assert named in str(caught.value)
        assert VOICE_ID in str(caught.value)

    @pytest.mark.parametrize("voice_id", ["../voices", "Test", "-lead", "", "a/b"])
    def test_a_malformed_voice_id_is_refused_before_any_path_is_built(self, voice_root, voice_id):
        with pytest.raises(VoiceAssetError, match="voice_id"):
            load_identity(voice_root, voice_id)

    def test_a_missing_identity_directory_names_the_manifest(self, voice_root):
        with pytest.raises(VoiceAssetError, match="voice.json"):
            load_identity(voice_root, "absent-voice")


class TestPrivacy:
    @pytest.mark.parametrize(("spoil", "named"), BAD_IDENTITIES)
    def test_no_error_or_log_record_holds_the_transcript(self, voice_root, caplog, spoil, named):
        identity_dir = make_identity(voice_root, ("neutral", "calm"))
        spoil(identity_dir)

        with caplog.at_level(logging.DEBUG), pytest.raises(VoiceAssetError) as caught:
            load_identity(voice_root, VOICE_ID)

        assert TRANSCRIPT_MARK not in str(caught.value)
        assert TRANSCRIPT_MARK not in repr(caught.value.__cause__)
        assert all(TRANSCRIPT_MARK not in record.getMessage() for record in caplog.records)

    def test_a_load_and_a_build_log_nothing_of_the_transcript(self, voice_root, cache_root, caplog):
        make_identity(voice_root, ("neutral", "calm", "sleepy"))

        with caplog.at_level(logging.DEBUG):
            identity = load_identity(voice_root, VOICE_ID)
            ensure_artefacts(identity, RecordingBuilder(), cache_root)

        assert all(TRANSCRIPT_MARK not in record.getMessage() for record in caplog.records)

    def test_the_design_log_is_never_opened(self, voice_root, cache_root, monkeypatch):
        identity_dir = make_identity(voice_root, design_log="design-log.md")
        opened: list[str] = []
        real_open, real_os_open = builtins.open, os.open

        def recording_open(file, *args, **kwargs):
            opened.append(os.fspath(file) if not isinstance(file, int) else "")
            return real_open(file, *args, **kwargs)

        def recording_os_open(path, *args, **kwargs):
            opened.append(os.fspath(path))
            return real_os_open(path, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", recording_open)
        monkeypatch.setattr(io, "open", recording_open)
        monkeypatch.setattr(os, "open", recording_os_open)

        identity = load_identity(voice_root, VOICE_ID)
        ensure_artefacts(identity, RecordingBuilder(), cache_root)

        design_log = (identity_dir / "design-log.md").resolve()
        assert any(Path(path).name == "neutral.wav" for path in opened)
        assert all(Path(path).resolve() != design_log for path in opened if path)


class TestArtefactKey:
    @pytest.fixture
    def clip(self, voice_root) -> VoiceClip:
        make_identity(voice_root)
        return load_identity(voice_root, VOICE_ID).clip("neutral")

    def test_the_key_hashes_the_five_inputs_joined_by_newlines(self, clip):
        builder = RecordingBuilder()
        joined = "\n".join(
            [
                builder.engine_id,
                builder.model_pin,
                str(builder.artefact_version),
                clip.audio_sha256,
                clip.transcript_sha256,
            ]
        )

        assert artefact_key(builder, clip) == hashlib.sha256(joined.encode("utf-8")).hexdigest()

    def test_the_key_changes_with_the_engine_id(self, clip):
        assert artefact_key(RecordingBuilder(engine_id="other-engine"), clip) != artefact_key(
            RecordingBuilder(), clip
        )

    def test_the_key_changes_with_the_model_pin(self, clip):
        assert artefact_key(RecordingBuilder(model_pin="test/model@def456"), clip) != (
            artefact_key(RecordingBuilder(), clip)
        )

    def test_the_key_changes_with_the_artefact_version(self, clip):
        assert artefact_key(RecordingBuilder(artefact_version=2), clip) != artefact_key(
            RecordingBuilder(), clip
        )

    def test_the_key_changes_with_the_audio_hash(self, clip):
        changed = dataclasses.replace(clip, audio_sha256="1" * 64)

        assert artefact_key(RecordingBuilder(), changed) != artefact_key(RecordingBuilder(), clip)

    def test_the_key_changes_with_the_transcript_hash(self, clip):
        changed = dataclasses.replace(clip, transcript_sha256="2" * 64)

        assert artefact_key(RecordingBuilder(), changed) != artefact_key(RecordingBuilder(), clip)

    def test_a_renamed_clip_file_keeps_the_key(self, clip):
        renamed = dataclasses.replace(
            clip,
            audio=clip.audio.with_name("renamed.wav"),
            transcript=clip.transcript.with_name("renamed.txt"),
        )

        assert artefact_key(RecordingBuilder(), renamed) == artefact_key(RecordingBuilder(), clip)


def _state_dir(cache_root: Path, state: str, engine: str = "fake-engine") -> Path:
    return cache_root / engine / VOICE_ID / state


def _tmp_dirs(cache_root: Path) -> list[Path]:
    return list(cache_root.rglob(".tmp-*"))


class _BuilderBugError(Exception):
    """An error type no adapter is expected to raise, so `ensure_artefacts` lets it through."""


class TestEnsureArtefacts:
    def test_each_clip_is_built_once_into_its_keyed_directory(self, voice_root, cache_root):
        make_identity(voice_root, ("neutral", "calm", "sleepy"))
        identity = load_identity(voice_root, VOICE_ID)
        builder = RecordingBuilder()

        ensure_artefacts(identity, builder, cache_root)

        assert sorted(builder.builds) == ["calm", "neutral", "sleepy"]
        for clip in identity.clips:
            key_dir = _state_dir(cache_root, clip.state) / artefact_key(builder, clip)
            assert (key_dir / "artefact.bin").read_bytes() == clip.audio_sha256.encode("ascii")
        assert _tmp_dirs(cache_root) == []

    def test_the_mapping_holds_what_load_read_by_state(self, voice_root, cache_root):
        make_identity(voice_root, ("neutral", "calm", "sleepy"))
        identity = load_identity(voice_root, VOICE_ID)

        loaded = ensure_artefacts(identity, RecordingBuilder(), cache_root)

        assert dict(loaded) == {
            state: (state, ("artefact.bin",)) for state in ("neutral", "calm", "sleepy")
        }

    def test_a_second_call_builds_nothing_and_loads_again(self, voice_root, cache_root):
        make_identity(voice_root, ("neutral", "calm"))
        identity = load_identity(voice_root, VOICE_ID)
        first = ensure_artefacts(identity, RecordingBuilder(), cache_root)
        builder = RecordingBuilder()

        second = ensure_artefacts(identity, builder, cache_root)

        assert builder.builds == []
        assert len(builder.loads) == 2
        assert dict(second) == dict(first)

    def test_a_changed_clip_rebuilds_only_its_state_and_removes_its_old_key(
        self, voice_root, cache_root
    ):
        identity_dir = make_identity(voice_root, ("neutral", "calm"))
        builder = RecordingBuilder()
        old = load_identity(voice_root, VOICE_ID)
        ensure_artefacts(old, builder, cache_root)
        _write_wav(identity_dir / "calm.wav", frequency=660.0)
        manifest = _read_manifest(identity_dir)
        _refresh_hashes(identity_dir, manifest, 1)
        _write_manifest(identity_dir, manifest)
        new = load_identity(voice_root, VOICE_ID)
        rebuilder = RecordingBuilder()

        ensure_artefacts(new, rebuilder, cache_root)

        assert rebuilder.builds == ["calm"]
        assert sorted(os.listdir(_state_dir(cache_root, "calm"))) == [
            artefact_key(builder, new.clip("calm"))
        ]
        assert sorted(os.listdir(_state_dir(cache_root, "neutral"))) == [
            artefact_key(builder, old.clip("neutral"))
        ]

    def test_a_changed_model_pin_rebuilds_every_state(self, voice_root, cache_root):
        make_identity(voice_root, ("neutral", "calm"))
        identity = load_identity(voice_root, VOICE_ID)
        ensure_artefacts(identity, RecordingBuilder(), cache_root)
        repinned = RecordingBuilder(model_pin="test/model@def456")

        ensure_artefacts(identity, repinned, cache_root)

        assert sorted(repinned.builds) == ["calm", "neutral"]
        assert len(os.listdir(_state_dir(cache_root, "calm"))) == 1

    def test_a_builder_that_raises_leaves_no_key_and_no_temporary_directory(
        self, voice_root, cache_root
    ):
        make_identity(voice_root)
        identity = load_identity(voice_root, VOICE_ID)

        def fail(clip: VoiceClip, out_dir: Path) -> None:
            raise SpeechEngineError("encoder failed")

        with pytest.raises(VoiceAssetError) as caught:
            ensure_artefacts(identity, RecordingBuilder(on_build=fail), cache_root)

        assert "fake-engine" in str(caught.value) and "neutral" in str(caught.value)
        state_dir = _state_dir(cache_root, "neutral")
        assert not state_dir.exists() or os.listdir(state_dir) == []

    def test_a_builder_raising_any_other_error_still_leaves_no_temporary_directory(
        self, voice_root, cache_root
    ):
        make_identity(voice_root)
        identity = load_identity(voice_root, VOICE_ID)

        def fail(clip: VoiceClip, out_dir: Path) -> None:
            raise _BuilderBugError("unexpected")

        with pytest.raises(_BuilderBugError):
            ensure_artefacts(identity, RecordingBuilder(on_build=fail), cache_root)

        state_dir = _state_dir(cache_root, "neutral")
        assert not state_dir.exists() or os.listdir(state_dir) == []

    def test_a_target_that_appears_during_a_build_wins(self, voice_root, cache_root):
        make_identity(voice_root)
        identity = load_identity(voice_root, VOICE_ID)
        builder = RecordingBuilder()

        def race(clip: VoiceClip, out_dir: Path) -> None:
            target = out_dir.parent / artefact_key(builder, clip)
            target.mkdir()
            (target / "winner.bin").write_bytes(b"first")

        builder.on_build = race

        loaded = ensure_artefacts(identity, builder, cache_root)

        key_dir = _state_dir(cache_root, "neutral") / artefact_key(
            builder, identity.clip("neutral")
        )
        assert os.listdir(key_dir) == ["winner.bin"]
        assert loaded["neutral"] == ("neutral", ("winner.bin",))
        assert _tmp_dirs(cache_root) == []

    def test_nothing_is_written_under_the_identity_root(self, voice_root, cache_root):
        identity_dir = make_identity(voice_root, ("neutral", "calm"), design_log="design-log.md")
        before = _listing(identity_dir)

        identity = load_identity(voice_root, VOICE_ID)
        ensure_artefacts(identity, RecordingBuilder(), cache_root)

        assert _listing(identity_dir) == before

    def test_a_cache_root_inside_the_identity_is_refused(self, voice_root):
        identity_dir = make_identity(voice_root)
        identity = load_identity(voice_root, VOICE_ID)
        before = _listing(identity_dir)

        with pytest.raises(VoiceAssetError, match="cache"):
            ensure_artefacts(identity, RecordingBuilder(), identity_dir / "cache")

        assert _listing(identity_dir) == before

    @pytest.mark.parametrize("engine_id", ["../escape", "Upper", ""])
    def test_an_engine_id_that_is_not_a_path_segment_is_refused(
        self, voice_root, cache_root, engine_id
    ):
        make_identity(voice_root)
        identity = load_identity(voice_root, VOICE_ID)

        with pytest.raises(VoiceAssetError, match="engine_id"):
            ensure_artefacts(identity, RecordingBuilder(engine_id=engine_id), cache_root)


class TestLockAndSweep:
    """Design review finding 7: two builders on one volume never interleave."""

    def test_a_second_call_waits_for_the_lock_and_then_builds_nothing(
        self, voice_root, cache_root, monkeypatch
    ):
        make_identity(voice_root)
        identity = load_identity(voice_root, VOICE_ID)
        building, release, second_at_lock = threading.Event(), threading.Event(), threading.Event()

        def hold(clip: VoiceClip, out_dir: Path) -> None:
            building.set()
            assert release.wait(timeout=10)

        first_builder = RecordingBuilder(on_build=hold)
        second_builder = RecordingBuilder()
        results: dict[str, object] = {}
        real_flock = fcntl.flock

        def spying_flock(fd, operation):
            if threading.current_thread().name == "second" and operation == fcntl.LOCK_EX:
                second_at_lock.set()
            return real_flock(fd, operation)

        monkeypatch.setattr(fcntl, "flock", spying_flock)

        def run(name: str, builder: RecordingBuilder) -> None:
            results[name] = dict(ensure_artefacts(identity, builder, cache_root))

        first = threading.Thread(target=run, args=("first", first_builder), name="first")
        second = threading.Thread(target=run, args=("second", second_builder), name="second")
        first.start()
        assert building.wait(timeout=10)
        second.start()
        assert second_at_lock.wait(timeout=10)
        second.join(timeout=0.2)
        waited = second.is_alive()
        release.set()
        first.join(timeout=10)
        second.join(timeout=10)

        assert waited
        assert first_builder.builds == ["neutral"]
        assert second_builder.builds == []
        assert results["second"] == results["first"]

    def test_a_stale_temporary_directory_is_swept_before_a_build(self, voice_root, cache_root):
        make_identity(voice_root)
        identity = load_identity(voice_root, VOICE_ID)
        stale = _state_dir(cache_root, "neutral") / ".tmp-killed"
        stale.mkdir(parents=True)
        (stale / "partial.bin").write_bytes(b"half")

        ensure_artefacts(identity, RecordingBuilder(), cache_root)

        assert not stale.exists()
        assert _tmp_dirs(cache_root) == []

    def test_a_stale_temporary_directory_is_swept_when_nothing_is_built(
        self, voice_root, cache_root
    ):
        make_identity(voice_root)
        identity = load_identity(voice_root, VOICE_ID)
        ensure_artefacts(identity, RecordingBuilder(), cache_root)
        stale = _state_dir(cache_root, "neutral") / ".tmp-killed"
        stale.mkdir()
        builder = RecordingBuilder()

        ensure_artefacts(identity, builder, cache_root)

        assert builder.builds == []
        assert not stale.exists()

    def test_only_sibling_names_of_64_hex_characters_are_swept_after_a_build(
        self, voice_root, cache_root
    ):
        make_identity(voice_root)
        identity = load_identity(voice_root, VOICE_ID)
        state_dir = _state_dir(cache_root, "neutral")
        old_key, short, other = state_dir / ("a" * 64), state_dir / ("b" * 63), state_dir / "notes"
        for path in (old_key, short, other):
            path.mkdir(parents=True)
        other_state = _state_dir(cache_root, "calm") / ("c" * 64)
        other_state.mkdir(parents=True)
        builder = RecordingBuilder()

        ensure_artefacts(identity, builder, cache_root)

        key = artefact_key(builder, identity.clip("neutral"))
        assert sorted(os.listdir(state_dir)) == sorted([key, "b" * 63, "notes"])
        assert other_state.exists()


class TestLoadUnderTheLock:
    """Design review N1: an adapter never reads the cache after the lock is released."""

    def test_every_load_runs_while_the_lock_is_held(self, voice_root, cache_root):
        make_identity(voice_root, ("neutral", "calm", "sleepy"))
        identity = load_identity(voice_root, VOICE_ID)
        held: dict[str, bool] = {}
        builder = RecordingBuilder(
            on_load=lambda artefact_dir: held.update(
                {artefact_dir.parent.name: _lock_is_held(cache_root)}
            )
        )

        ensure_artefacts(identity, builder, cache_root)

        assert held == {"neutral": True, "calm": True, "sleepy": True}
        assert _lock_is_held(cache_root) is False

    def test_a_load_that_raises_names_the_engine_and_state_and_keeps_the_key(
        self, voice_root, cache_root
    ):
        make_identity(voice_root, ("neutral", "calm"))
        identity = load_identity(voice_root, VOICE_ID)

        def fail_calm(artefact_dir: Path) -> None:
            if artefact_dir.parent.name == "calm":
                raise SpeechEngineError("voice state unreadable")

        builder = RecordingBuilder(on_load=fail_calm)

        with pytest.raises(VoiceAssetError) as caught:
            ensure_artefacts(identity, builder, cache_root)

        message = str(caught.value)
        assert "fake-engine" in message and "calm" in message
        assert (
            _state_dir(cache_root, "calm") / artefact_key(builder, identity.clip("calm"))
        ).is_dir()
        assert _lock_is_held(cache_root) is False


def test_the_vocabulary_the_manifest_checks_is_the_speech_contracts():
    assert voice_assets.SPEAKING_STATES is SPEAKING_STATES
