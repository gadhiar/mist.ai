"""Voice identities, the private clips MIST's voice is cloned from, and the artefacts built on them.

An identity is a directory `<voice_root>/<voice_id>/` under a private root: a manifest
(`voice.json`), one clip per speaking state, each with a transcript, and optionally the design
log. A schema-2 identity also carries trained weights (its `model` object), which the engine it
names loads in place of its stock clone weights. Loading only reads, so the root can be mounted
read-only: it reads the manifest, checks each file's sha256 and WAV header, and checks that the
design log exists without reading it. A bad identity fails at load with a `VoiceAssetError`
that names the field or the path relative to the identity, so a misspelt tag or a stale hash
fails when the voice is configured, not at every turn, and the message never holds a transcript
or audio.

Each engine adapter turns a clip into its own artefacts (Pocket: a voice state) through a
`VoiceArtefactBuilder`. They are cached under `<cache_root>/<engine_id>/<voice_id>/<state>/<key>/`,
where the key hashes everything the artefact depends on, so a changed clip, model pin or builder
rebuilds only what changed. The cache is a named volume that several builders may share, so
`ensure_artefacts` builds, sweeps and loads under one exclusive lock on the cache root.
"""

import fcntl
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import wave
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal, Protocol, TypeVar

from ..errors import MistError
from .errors import SpeechEngineError
from .speech import NEUTRAL_STATE, SPEAKING_STATES

logger = logging.getLogger(__name__)

VOICE_MANIFEST = "voice.json"
MANIFEST_SCHEMA = 1
MODEL_MANIFEST_SCHEMA = 2
"""The schema of an identity that carries trained weights; schema 1 carries clips only."""
SHA_PREFIX = 12
"""How many hex characters of a weights sha256 a log line shows, in place of its path."""

_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]*")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
# A key directory's name: what the sibling sweep may remove, and nothing else.
_KEY_DIR_PATTERN = re.compile(r"[0-9a-fA-F]{64}")
_STATUSES = ("placeholder", "frozen")
_MIN_RATE, _MAX_RATE = 16000, 48000
_MIN_SECONDS, _MAX_SECONDS = 3, 30
_TMP_PREFIX = ".tmp-"
_LOCK_NAME = ".lock"
_HASH_BLOCK = 1 << 20

# What an adapter's `build` or `load` raises for a bad clip or artefact: its own
# `SpeechEngineError` (adapters convert library errors, as `lanes.py` requires) or file errors.
_ADAPTER_ERRORS = (MistError, OSError, ValueError, RuntimeError)


class VoiceAssetError(SpeechEngineError):
    """A voice identity or its artefacts cannot be used.

    The message names the identity and the manifest field or the path relative to the identity,
    never a transcript or audio, so it is safe in the one ERROR line that leaves TTS off.
    """


@dataclass(frozen=True, slots=True)
class VoiceClip:
    """One reference clip of an identity, checked against its manifest entry.

    `audio` and `transcript` are absolute paths inside the identity directory. `sample_rate`
    and `duration_ms` are read from the WAV header.
    """

    state: str
    audio: Path
    transcript: Path
    audio_sha256: str
    transcript_sha256: str
    sample_rate: int
    duration_ms: int


@dataclass(frozen=True, slots=True)
class IdentityModel:
    """Trained weights a schema-2 identity carries, checked against its manifest entry.

    `engine` and `base` are as the manifest names them: the engine adapter decides whether it
    can load them. `weights_path` and `tokenizer_path` are absolute paths inside the identity
    directory, each matching its sha256. `tokenizer_path` is None when the weights were trained
    with the engine's stock tokenizer; `tokenizer_sha256` then names that stock file, which the
    adapter checks.
    """

    engine: str
    base: str
    weights_path: Path
    sha256: str
    tokenizer_path: Path | None
    tokenizer_sha256: str


@dataclass(frozen=True, slots=True)
class VoiceIdentity:
    """MIST's voice: one clip per speaking state, a `neutral` clip always.

    `voice_id` is the id `PlacementConfig.tts_voice` names and `TtsVoice.voice` reports.
    `status` is `placeholder` until a voice is frozen. `root` is the identity directory,
    resolved; nothing is ever written under it. `model` is the trained weights of a schema-2
    identity, None for schema 1.
    """

    voice_id: str
    status: Literal["placeholder", "frozen"]
    clips: tuple[VoiceClip, ...]
    root: Path
    model: IdentityModel | None = None

    def clip(self, state: str) -> VoiceClip:
        """The clip for `state`; any state the identity has no clip for gets the neutral clip."""
        by_state = {clip.state: clip for clip in self.clips}
        return by_state.get(state, by_state[NEUTRAL_STATE])

    @property
    def states(self) -> frozenset[str]:
        """The speaking states the identity has a clip for."""
        return frozenset(clip.state for clip in self.clips)


def _error(voice_id: str, where: str, reason: str) -> VoiceAssetError:
    return VoiceAssetError(f"voice identity {voice_id!r}: {where}: {reason}")


def _resolve_inside(root: Path, raw: object, field: str, voice_id: str) -> tuple[Path, str]:
    """Resolve a manifest path; returns the absolute path and its relative spelling."""
    if not isinstance(raw, str) or not raw:
        raise _error(voice_id, field, "must be a non-empty relative path")
    if PurePosixPath(raw).is_absolute() or PureWindowsPath(raw).anchor:
        raise _error(voice_id, field, "must be a path relative to the identity directory")
    where = f"{field} {raw!r}"
    if ".." in PurePosixPath(raw).parts:
        raise _error(voice_id, where, "leaves the identity directory")
    candidate = root / raw
    if not candidate.exists():
        raise _error(voice_id, where, "is missing")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise _error(voice_id, where, "resolves outside the identity directory")
    if not resolved.is_file():
        raise _error(voice_id, where, "is not a file")
    return resolved, raw


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while block := source.read(_HASH_BLOCK):
            digest.update(block)
    return digest.hexdigest()


def _check_sha256(value: object, field: str, voice_id: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise _error(voice_id, field, "is not 64 lowercase hexadecimal characters")
    return value


def _wav_header(path: Path, where: str, voice_id: str) -> tuple[int, int]:
    """Check a clip is PCM16 mono at an allowed rate and length; returns rate and duration."""
    try:
        with wave.open(str(path), "rb") as reader:
            channels = reader.getnchannels()
            width = reader.getsampwidth()
            rate = reader.getframerate()
            frames = reader.getnframes()
    except (wave.Error, EOFError):
        raise _error(voice_id, where, "is not a PCM WAV file") from None
    if channels != 1 or width != 2:
        raise _error(voice_id, where, f"is {channels} channel(s) of {8 * width}-bit audio")
    if not _MIN_RATE <= rate <= _MAX_RATE:
        raise _error(voice_id, where, f"sample rate {rate} Hz is outside 16000 to 48000 Hz")
    if not _MIN_SECONDS * rate <= frames <= _MAX_SECONDS * rate:
        raise _error(voice_id, where, f"length {frames / rate:.2f} s is outside 3 to 30 s")
    return rate, frames * 1000 // rate


def _check_transcript(path: Path, where: str, voice_id: str) -> None:
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        # `from None`: the decode error's repr holds the transcript's bytes.
        raise _error(voice_id, where, "is not UTF-8") from None
    if not text.strip():
        raise _error(voice_id, where, "is empty")


def _load_clip(root: Path, entry: object, index: int, voice_id: str) -> VoiceClip:
    field = f"clips[{index}]"
    if not isinstance(entry, dict):
        raise _error(voice_id, field, "is not an object")
    state = entry.get("state")
    audio, audio_rel = _resolve_inside(root, entry.get("audio"), f"{field}.audio", voice_id)
    transcript, transcript_rel = _resolve_inside(
        root, entry.get("transcript"), f"{field}.transcript", voice_id
    )
    audio_sha = _check_sha256(entry.get("audio_sha256"), f"{field}.audio_sha256", voice_id)
    transcript_sha = _check_sha256(
        entry.get("transcript_sha256"), f"{field}.transcript_sha256", voice_id
    )
    if _file_sha256(audio) != audio_sha:
        raise _error(voice_id, f"{field}.audio_sha256", f"does not match {audio_rel!r}")
    rate, duration_ms = _wav_header(audio, f"{field}.audio {audio_rel!r}", voice_id)
    if _file_sha256(transcript) != transcript_sha:
        raise _error(voice_id, f"{field}.transcript_sha256", f"does not match {transcript_rel!r}")
    _check_transcript(transcript, f"{field}.transcript {transcript_rel!r}", voice_id)
    return VoiceClip(
        state=str(state),
        audio=audio,
        transcript=transcript,
        audio_sha256=audio_sha,
        transcript_sha256=transcript_sha,
        sample_rate=rate,
        duration_ms=duration_ms,
    )


def _check_states(clips: object, voice_id: str) -> list[object]:
    if not isinstance(clips, list) or not clips:
        raise _error(voice_id, "clips", "must list at least one clip")
    seen: set[str] = set()
    for index, entry in enumerate(clips):
        state = entry.get("state") if isinstance(entry, dict) else None
        field = f"clips[{index}].state"
        if not isinstance(state, str) or state not in SPEAKING_STATES:
            raise _error(voice_id, field, f"{state!r} is not in SPEAKING_STATES")
        if state in seen:
            raise _error(voice_id, field, f"{state!r} is tagged on more than one clip")
        seen.add(state)
    if NEUTRAL_STATE not in seen:
        raise _error(voice_id, "clips", f"no clip is tagged {NEUTRAL_STATE!r}, the fallback")
    return clips


def _non_empty(value: object, field: str, voice_id: str) -> str:
    if not isinstance(value, str) or not value:
        raise _error(voice_id, field, "must be a non-empty string")
    return value


def _load_identity_model(root: Path, entry: object, voice_id: str) -> IdentityModel:
    """Check a schema-2 `model` object: its fields, and its files against their sha256."""
    if not isinstance(entry, dict):
        raise _error(voice_id, "model", "is not an object")
    engine = _non_empty(entry.get("engine"), "model.engine", voice_id)
    base = _non_empty(entry.get("base"), "model.base", voice_id)
    weights, weights_rel = _resolve_inside(root, entry.get("weights"), "model.weights", voice_id)
    sha = _check_sha256(entry.get("sha256"), "model.sha256", voice_id)
    tokenizer_sha = _check_sha256(entry.get("tokenizer_sha256"), "model.tokenizer_sha256", voice_id)
    if _file_sha256(weights) != sha:
        raise _error(voice_id, "model.sha256", f"does not match {weights_rel!r}")
    tokenizer = None
    if entry.get("tokenizer") is not None:
        tokenizer, tokenizer_rel = _resolve_inside(
            root, entry["tokenizer"], "model.tokenizer", voice_id
        )
        if _file_sha256(tokenizer) != tokenizer_sha:
            raise _error(voice_id, "model.tokenizer_sha256", f"does not match {tokenizer_rel!r}")
    return IdentityModel(
        engine=engine,
        base=base,
        weights_path=weights,
        sha256=sha,
        tokenizer_path=tokenizer,
        tokenizer_sha256=tokenizer_sha,
    )


def load_identity(voice_root: Path, voice_id: str) -> VoiceIdentity:
    """Load and check the identity `<voice_root>/<voice_id>/`.

    Every check runs here, so an identity that loads can be voiced: the manifest's schema, id,
    status and clip tags; each path relative and inside the identity directory, symlinks
    included; each file's sha256; each clip's WAV header; each transcript non-empty UTF-8; and
    the design log's existence, which is never opened. Schema 2 requires a `model` object and
    schema 1 refuses one, so trained weights are never ignored or guessed at; the weights and
    an identity tokenizer are checked like clips. Reads only, so the read-only mount is enough.

    Raises:
        VoiceAssetError: The identity is missing or breaks a rule; the message names the field
            or the relative path.
    """
    if not isinstance(voice_id, str) or not _ID_PATTERN.fullmatch(voice_id):
        raise _error(str(voice_id), "voice_id", "must match [a-z0-9][a-z0-9-]*")
    directory = Path(voice_root) / voice_id
    if not directory.is_dir():
        raise _error(voice_id, VOICE_MANIFEST, "is missing (no identity directory)")
    root = directory.resolve(strict=True)
    manifest_path, _ = _resolve_inside(root, VOICE_MANIFEST, VOICE_MANIFEST, voice_id)
    try:
        manifest = json.loads(manifest_path.read_bytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise _error(voice_id, VOICE_MANIFEST, "is not UTF-8 JSON") from None
    if not isinstance(manifest, dict):
        raise _error(voice_id, VOICE_MANIFEST, "is not a JSON object")

    schema = manifest.get("schema")
    # `True == 1` in Python, so a bool is refused before the comparison.
    if (
        isinstance(schema, bool)
        or not isinstance(schema, int)
        or schema not in (MANIFEST_SCHEMA, MODEL_MANIFEST_SCHEMA)
    ):
        raise _error(voice_id, "schema", f"must be {MANIFEST_SCHEMA} or {MODEL_MANIFEST_SCHEMA}")
    if schema == MANIFEST_SCHEMA and "model" in manifest:
        raise _error(
            voice_id,
            "model",
            f"is not allowed in schema {MANIFEST_SCHEMA}; trained weights need schema "
            f"{MODEL_MANIFEST_SCHEMA}",
        )
    if schema == MODEL_MANIFEST_SCHEMA and "model" not in manifest:
        raise _error(voice_id, "model", f"is required by schema {MODEL_MANIFEST_SCHEMA}")
    if manifest.get("voice_id") != voice_id:
        raise _error(voice_id, "voice_id", "does not equal the identity directory's name")
    status = manifest.get("status")
    if status not in _STATUSES:
        raise _error(voice_id, "status", "must be 'placeholder' or 'frozen'")
    entries = _check_states(manifest.get("clips"), voice_id)
    clips = tuple(_load_clip(root, entry, index, voice_id) for index, entry in enumerate(entries))
    if manifest.get("design_log") is not None:
        _resolve_inside(root, manifest["design_log"], "design_log", voice_id)
    model = (
        _load_identity_model(root, manifest["model"], voice_id)
        if schema == MODEL_MANIFEST_SCHEMA
        else None
    )

    identity = VoiceIdentity(voice_id=voice_id, status=status, clips=clips, root=root, model=model)
    # The weights are named by a sha prefix, never by a path under the private root.
    logger.info(
        "Loaded voice identity %r (%s): states %s%s",
        voice_id,
        status,
        ", ".join(sorted(identity.states)),
        "" if model is None else f"; trained weights sha256 {model.sha256[:SHA_PREFIX]}",
    )
    return identity


T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


class VoiceArtefactBuilder(Protocol[T_co]):
    """An engine adapter's recipe for turning one clip into its own files, and reading them back.

    `model_pin` is `<repo>@<revision>`, or `<repo>@<revision>+<sha256>` for an identity's
    trained weights, and `artefact_version` the builder's format version; both go into the
    cache key, so a new model, changed weights or a changed builder rebuilds. `build` writes into
    `out_dir`, a fresh temporary directory, and nowhere else; `load` reads a finished key
    directory (Pocket: the voice state). Both raise `SpeechEngineError` for a bad clip or
    artefact, converting their library's errors as every adapter does. Neither logs or raises
    with transcript text or audio.
    """

    engine_id: str
    model_pin: str
    artefact_version: int

    def build(self, clip: VoiceClip, out_dir: Path) -> None:
        """Write the clip's artefacts into `out_dir`."""
        ...

    def load(self, artefact_dir: Path) -> T_co:
        """Read the artefacts a `build` wrote, now in their key directory."""
        ...


def artefact_key(builder: VoiceArtefactBuilder[T], clip: VoiceClip) -> str:
    """The sha256 of what an artefact depends on: engine, model pin, builder version, clip.

    The clip counts by its audio's and transcript's hashes, not its file names, so renaming a
    clip file does not rebuild it.
    """
    material = "\n".join(
        (
            builder.engine_id,
            builder.model_pin,
            str(builder.artefact_version),
            clip.audio_sha256,
            clip.transcript_sha256,
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@contextmanager
def _cache_lock(cache_root: Path) -> Iterator[None]:
    fd = os.open(cache_root / _LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _sweep_temporaries(voice_dir: Path) -> None:
    """Remove what killed builds left; only safe under the lock, where no build is running."""
    if not voice_dir.is_dir():
        return
    for state_dir in voice_dir.iterdir():
        if state_dir.is_symlink() or not state_dir.is_dir():
            continue
        for entry in state_dir.iterdir():
            if entry.name.startswith(_TMP_PREFIX):
                _remove(entry)


def _sweep_siblings(state_dir: Path, keep: str) -> None:
    """Remove the state's other key directories, and only names a key could have."""
    for entry in state_dir.iterdir():
        if entry.name == keep or not _KEY_DIR_PATTERN.fullmatch(entry.name):
            continue
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)


def _fsync(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish(temporary: Path, target: Path) -> None:
    """Rename a finished build into place; a target that appeared meanwhile wins.

    The build's files are flushed before the rename and the rename after it, so a key
    directory that survives a crash of the whole VM holds complete files.
    """
    if target.exists():
        return
    for path in sorted(temporary.rglob("*"), reverse=True):
        if not path.is_symlink():
            _fsync(path)
    _fsync(temporary)
    try:
        os.replace(temporary, target)
    except OSError:
        if not target.is_dir():
            raise
        return
    _fsync(target.parent)


def _build(
    identity: VoiceIdentity, builder: VoiceArtefactBuilder[T], clip: VoiceClip, state_dir: Path
) -> Path:
    key = artefact_key(builder, clip)
    target = state_dir / key
    if target.is_dir():
        return target
    state_dir.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=_TMP_PREFIX, dir=state_dir))
    try:
        try:
            builder.build(clip, temporary)
        except _ADAPTER_ERRORS as exc:
            raise VoiceAssetError(
                f"engine {builder.engine_id!r} could not build voice {identity.voice_id!r} "
                f"state {clip.state!r}"
            ) from exc
        _publish(temporary, target)
    finally:
        if os.path.lexists(temporary):
            shutil.rmtree(temporary, ignore_errors=True)
    _sweep_siblings(state_dir, keep=key)
    logger.info(
        "Built %s artefacts for voice %r state %r",
        builder.engine_id,
        identity.voice_id,
        clip.state,
    )
    return target


def ensure_artefacts(
    identity: VoiceIdentity, builder: VoiceArtefactBuilder[T], cache_root: Path
) -> Mapping[str, T]:
    """Build what is missing for each of the identity's clips, then load every state's artefacts.

    The whole run holds an exclusive `flock` on `<cache_root>/.lock`, so two builders on one
    volume never interleave. Under it, every `.tmp-*` left in the identity's cache directories
    by a killed build is removed first. A build writes `.tmp-<random>` beside its key directory
    and renames it into place, then removes the state's older key directories. Every `load`
    runs under the lock too, so another project's sweep cannot remove a key between its build
    and its read.

    Returns:
        Each state's `builder.load` result.

    Raises:
        VoiceAssetError: The builder's id is not a path segment, the cache root is inside the
            identity, or a `build` or `load` failed; the message names the engine and the
            state. A failed build leaves no key and no temporary directory; a failed load
            leaves its key directory in place.
    """
    engine_id = builder.engine_id
    if not isinstance(engine_id, str) or not _ID_PATTERN.fullmatch(engine_id):
        raise VoiceAssetError(f"engine_id {engine_id!r} must match [a-z0-9][a-z0-9-]*")
    cache = Path(cache_root).resolve()
    if cache.is_relative_to(identity.root):
        raise VoiceAssetError(
            f"voice identity {identity.voice_id!r}: the cache root is inside the identity"
        )
    cache.mkdir(parents=True, exist_ok=True)
    voice_dir = cache / engine_id / identity.voice_id
    with _cache_lock(cache):
        _sweep_temporaries(voice_dir)
        key_dirs = {
            clip.state: _build(identity, builder, clip, voice_dir / clip.state)
            for clip in identity.clips
        }
        loaded: dict[str, T] = {}
        for state, key_dir in key_dirs.items():
            try:
                loaded[state] = builder.load(key_dir)
            except _ADAPTER_ERRORS as exc:
                raise VoiceAssetError(
                    f"engine {engine_id!r} could not load voice {identity.voice_id!r} "
                    f"state {state!r}"
                ) from exc
        return loaded
