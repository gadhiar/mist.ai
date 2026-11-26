"""Voice identities in goal mist-realtime-voice's format (v6 plan, lines 700-714).

An identity is `voices/<voice_id>/voice.json` and, per speaking state, one clip: a WAV and its
verbatim transcript, each pinned by its sha256. The realtime goal loads it; this goal reads one
as the source of a target (`target build --identity`) and writes a `placeholder` one for a seed
target (`identity export`), so every package's `voice_id` names an identity v6 can load. Freezing
an identity stays v6's step.

Schema 2 (mist-voice-fidelity plan v4, Interfaces) adds a `model` object: trained weights inside
the identity, pinned by their sha256, the engine and base they were trained for, and the sha256
of the tokenizer they were trained with, plus that tokenizer's file when it is not the engine's
stock one. Schema 1 refuses a `model`, and schema 2 requires one, as the realtime loader does
(`backend/realtime/voice_assets.py::load_identity`). `student export` writes schema 2.

`check` applies every one of v6's rules with the standard library's `wave` module. A transcript
is read as bytes to check its sha256 and that it is non-empty UTF-8, and is never printed: every
error names the clip's state and the field, never its content. The design log's existence is
checked; it is never opened.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .errors import InputError
from .root import Root
from .states import NEUTRAL_STATE, SPEAKING_STATES

SCHEMA = 1
# The schema of an identity that carries trained weights; schema 1 carries clips only.
MODEL_SCHEMA = 2
VOICES_DIR = "voices"
MANIFEST = "voice.json"
STATUSES = ("placeholder", "frozen")
PLACEHOLDER = "placeholder"
VOICE_ID = re.compile(r"[a-z0-9][a-z0-9-]*")
SHA256 = re.compile(r"[0-9a-f]{64}")
RATE_MIN_HZ = 16000
RATE_MAX_HZ = 48000
MIN_S = 3.0
MAX_S = 30.0
EXPORT_RATE_HZ = 24000
CLIP_KEYS = ("state", "audio", "transcript", "audio_sha256", "transcript_sha256")
MODEL_KEYS = ("engine", "base", "weights", "sha256", "tokenizer_sha256")
# The names `export` gives trained weights and a tokenizer inside the identity.
WEIGHTS_FILE = "model.safetensors"
TOKENIZER_FILE = "tokenizer.json"
_HASH_BLOCK = 1 << 20


class IdentityError(InputError):
    """An identity breaks one of v6's rules; the message names the field, never a value."""


@dataclass(frozen=True, slots=True)
class WavFacts:
    """What `wave` reports of a clip: PCM sample width in bytes, channels, rate and length."""

    sample_width: int
    channels: int
    rate: int
    frames: int

    @property
    def duration_s(self) -> float:
        """The clip's length in seconds."""
        return self.frames / self.rate if self.rate else 0.0


@dataclass(frozen=True, slots=True)
class IdentityClip:
    """One checked clip; paths are relative to the identity directory, in posix form."""

    state: str
    audio: str
    transcript: str
    audio_sha256: str
    transcript_sha256: str
    rate: int
    duration_s: float


@dataclass(frozen=True, slots=True)
class IdentityModel:
    """A schema-2 identity's checked trained weights; paths are relative to the identity.

    `tokenizer` is `None` when the weights were trained with the engine's stock tokenizer;
    `tokenizer_sha256` then names that stock file, which the engine's loader checks.
    """

    engine: str
    base: str
    weights: str
    sha256: str
    tokenizer: str | None
    tokenizer_sha256: str


@dataclass(frozen=True, slots=True)
class Identity:
    """A checked identity: its directory, id, status, clips in manifest order and its weights.

    `model` is `None` for schema 1.
    """

    path: Path
    voice_id: str
    status: str
    clips: tuple[IdentityClip, ...]
    model: IdentityModel | None = None

    def clip(self, state: str) -> IdentityClip | None:
        """Return the state's clip, or `None` when the identity has none for it."""
        return next((c for c in self.clips if c.state == state), None)


def voice_rel(voice_id: str) -> str:
    """Return an identity's directory, relative to the root."""
    return f"{VOICES_DIR}/{voice_id}"


def check_voice_id(voice_id: str) -> None:
    """Refuse a voice id that is not `[a-z0-9][a-z0-9-]*`.

    Raises:
        IdentityError: The id does not match.
    """
    if not VOICE_ID.fullmatch(voice_id):
        raise IdentityError("refused: a voice id is [a-z0-9][a-z0-9-]*")


def sha256_bytes(data: bytes) -> str:
    """Return the sha256 hex digest of `data`."""
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    # In blocks: trained weights run to hundreds of megabytes.
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_HASH_BLOCK):
            digest.update(block)
    return digest.hexdigest()


def wav_facts(path: Path, where: str) -> WavFacts:
    """Read a WAV's header with `wave`, which accepts uncompressed PCM only.

    Raises:
        IdentityError: The file is missing or not a PCM WAV; the message names `where`.
    """
    try:
        with wave.open(str(path), "rb") as handle:
            if handle.getcomptype() != "NONE":
                raise IdentityError(f"refused: {where} is not PCM")
            return WavFacts(
                sample_width=handle.getsampwidth(),
                channels=handle.getnchannels(),
                rate=handle.getframerate(),
                frames=handle.getnframes(),
            )
    except (OSError, EOFError, wave.Error) as exc:
        raise IdentityError(f"refused: {where} is not a readable PCM WAV") from exc


def check_wav(path: Path, where: str, *, rate: int | None = None) -> WavFacts:
    """Apply v6's audio rule: PCM16 mono, 16000-48000 Hz (or exactly `rate`), 3-30 s.

    Raises:
        IdentityError: A rule fails; the message names `where` and the rule.
    """
    facts = wav_facts(path, where)
    if facts.sample_width != 2 or facts.channels != 1:
        raise IdentityError(f"refused: {where} is not PCM16 mono")
    if rate is not None and facts.rate != rate:
        raise IdentityError(f"refused: {where} is not at {rate} Hz")
    if not RATE_MIN_HZ <= facts.rate <= RATE_MAX_HZ:
        raise IdentityError(f"refused: {where} is not at {RATE_MIN_HZ}-{RATE_MAX_HZ} Hz")
    if not MIN_S <= facts.duration_s <= MAX_S:
        raise IdentityError(f"refused: {where} is not {MIN_S:g}-{MAX_S:g} s long")
    return facts


def check_transcript(data: bytes, where: str) -> None:
    """Apply v6's transcript rule: non-empty UTF-8. The text itself is never shown.

    Raises:
        IdentityError: The bytes are not UTF-8, or hold only whitespace.
    """
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise IdentityError(f"refused: {where} is not UTF-8") from exc
    if not text.strip():
        raise IdentityError(f"refused: {where} is empty")


def _inside(directory: Path, rel: object, where: str) -> Path:
    # v6: relative, no `..`, no absolute path, no symlink out of the identity directory.
    if not isinstance(rel, str) or not rel or "\\" in rel:
        raise IdentityError(f"refused: {where} is not a relative path")
    pure = PurePosixPath(rel)
    if pure.is_absolute() or ".." in pure.parts or re.match(r"[A-Za-z]:", rel):
        raise IdentityError(f"refused: {where} is not a relative path inside the identity")
    path = (directory / rel).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise IdentityError(f"refused: {where} resolves outside the identity")
    if not path.is_file():
        raise IdentityError(f"refused: {where} is missing")
    return path


def _read_manifest(directory: Path) -> dict[str, object]:
    path = directory / MANIFEST
    if not path.is_file():
        raise IdentityError(f"refused: {MANIFEST} is missing")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IdentityError(f"refused: {MANIFEST} is not readable JSON") from exc
    if not isinstance(data, dict):
        raise IdentityError(f"refused: {MANIFEST} is not a JSON object")
    return data


def _check_clip(directory: Path, index: int, entry: object, seen: set[str]) -> IdentityClip:
    where = f"clips[{index}]"
    if not isinstance(entry, dict):
        raise IdentityError(f"refused: {where} is not an object")
    missing = [key for key in CLIP_KEYS if key not in entry]
    if missing:
        raise IdentityError(f"refused: {where} has no {', '.join(missing)}")
    state = entry["state"]
    if state not in SPEAKING_STATES:
        raise IdentityError(f"refused: {where}.state is not a speaking state")
    if state in seen:
        raise IdentityError(f"refused: {where}.state repeats a state")
    seen.add(state)
    for key in ("audio_sha256", "transcript_sha256"):
        if not isinstance(entry[key], str) or not SHA256.fullmatch(entry[key]):
            raise IdentityError(f"refused: {where}.{key} is not a sha256")
    audio = _inside(directory, entry["audio"], f"{where}.audio")
    transcript = _inside(directory, entry["transcript"], f"{where}.transcript")
    if sha256_bytes(audio.read_bytes()) != entry["audio_sha256"]:
        raise IdentityError(f"refused: {where}.audio does not match audio_sha256")
    data = transcript.read_bytes()
    if sha256_bytes(data) != entry["transcript_sha256"]:
        raise IdentityError(f"refused: {where}.transcript does not match transcript_sha256")
    check_transcript(data, f"{where}.transcript")
    facts = check_wav(audio, f"{where}.audio")
    return IdentityClip(
        state=state,
        audio=str(entry["audio"]),
        transcript=str(entry["transcript"]),
        audio_sha256=str(entry["audio_sha256"]),
        transcript_sha256=str(entry["transcript_sha256"]),
        rate=facts.rate,
        duration_s=facts.duration_s,
    )


def _check_model(directory: Path, entry: object) -> IdentityModel:
    # The realtime loader's checks of a schema-2 `model` (`voice_assets._load_identity_model`).
    if not isinstance(entry, dict):
        raise IdentityError("refused: model is not an object")
    missing = [key for key in MODEL_KEYS if key not in entry]
    if missing:
        raise IdentityError(f"refused: model has no {', '.join(missing)}")
    for key in ("engine", "base"):
        if not isinstance(entry[key], str) or not entry[key]:
            raise IdentityError(f"refused: model.{key} is not a non-empty string")
    for key in ("sha256", "tokenizer_sha256"):
        if not isinstance(entry[key], str) or not SHA256.fullmatch(entry[key]):
            raise IdentityError(f"refused: model.{key} is not a sha256")
    weights = _inside(directory, entry["weights"], "model.weights")
    if _sha256_path(weights) != entry["sha256"]:
        raise IdentityError("refused: model.weights does not match model.sha256")
    tokenizer = entry.get("tokenizer")
    if tokenizer is not None:
        path = _inside(directory, tokenizer, "model.tokenizer")
        if _sha256_path(path) != entry["tokenizer_sha256"]:
            raise IdentityError("refused: model.tokenizer does not match model.tokenizer_sha256")
    return IdentityModel(
        engine=entry["engine"],
        base=entry["base"],
        weights=str(entry["weights"]),
        sha256=entry["sha256"],
        tokenizer=None if tokenizer is None else str(tokenizer),
        tokenizer_sha256=entry["tokenizer_sha256"],
    )


def check(directory: Path) -> Identity:
    """Check the identity in `directory` against every one of v6's rules, and schema 2's.

    Raises:
        IdentityError: A rule fails; the message names the field that breaks it.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise IdentityError("refused: the identity directory is missing")
    data = _read_manifest(directory)
    schema = data.get("schema")
    if isinstance(schema, bool) or schema not in (SCHEMA, MODEL_SCHEMA):
        raise IdentityError(f"refused: {MANIFEST} schema is not {SCHEMA} or {MODEL_SCHEMA}")
    if schema == SCHEMA and "model" in data:
        raise IdentityError(f"refused: model needs schema {MODEL_SCHEMA}, not schema {SCHEMA}")
    if schema == MODEL_SCHEMA and "model" not in data:
        raise IdentityError(f"refused: schema {MODEL_SCHEMA} needs a model")
    voice_id = data.get("voice_id")
    if not isinstance(voice_id, str) or not VOICE_ID.fullmatch(voice_id):
        raise IdentityError("refused: voice_id is not [a-z0-9][a-z0-9-]*")
    if voice_id != directory.name:
        raise IdentityError("refused: voice_id is not the directory's name")
    status = data.get("status")
    if status not in STATUSES:
        raise IdentityError(f"refused: status is not one of {', '.join(STATUSES)}")
    clips = data.get("clips")
    if not isinstance(clips, list) or not clips:
        raise IdentityError("refused: clips is not a non-empty list")
    seen: set[str] = set()
    checked = tuple(_check_clip(directory, i, entry, seen) for i, entry in enumerate(clips))
    if NEUTRAL_STATE not in seen:
        raise IdentityError(f"refused: clips has no {NEUTRAL_STATE} clip")
    design_log = data.get("design_log")
    if design_log is not None:
        _inside(directory, design_log, "design_log")
    model = _check_model(directory, data["model"]) if schema == MODEL_SCHEMA else None
    return Identity(
        path=directory, voice_id=voice_id, status=str(status), clips=checked, model=model
    )


def check_in_root(root: Root, voice_id: str) -> Identity:
    """Check `voices/<voice_id>/` in the root.

    Raises:
        IdentityError: The id is malformed, or the identity breaks a rule.
    """
    check_voice_id(voice_id)
    return check(root.path / voice_rel(voice_id))


@dataclass(frozen=True, slots=True)
class ExportClip:
    """A clip to export: a state's reference WAV and its transcript, as root paths."""

    state: str
    audio: Path
    transcript: Path
    transcript_sha256: str


@dataclass(frozen=True, slots=True)
class ExportModel:
    """Trained weights to export into a schema-2 identity, with the files as paths to copy.

    `tokenizer` is the tokenizer the weights were trained with, given only when it is not the
    engine's stock tokenizer; `tokenizer_sha256` is that training tokenizer's sha256 either way.
    """

    engine: str
    base: str
    weights: Path
    tokenizer: Path | None
    tokenizer_sha256: str


def _model_entry(root: Root, staging: str, model: ExportModel) -> dict[str, str]:
    """Copy the weights (and a tokenizer) into the staging folder; return the `model` entry.

    The weights are copied in blocks rather than read whole, and hashed as copied.
    """
    if not SHA256.fullmatch(model.tokenizer_sha256):
        raise IdentityError("refused: the model's tokenizer_sha256 is not a sha256")
    if not model.weights.is_file():
        raise IdentityError("refused: the model's weights are missing")
    weights = root.writable(f"{staging}/{WEIGHTS_FILE}")
    weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(model.weights, weights)
    entry = {
        "engine": model.engine,
        "base": model.base,
        "weights": WEIGHTS_FILE,
        "sha256": _sha256_path(weights),
    }
    if model.tokenizer is not None:
        if not model.tokenizer.is_file():
            raise IdentityError("refused: the model's tokenizer is missing")
        data = model.tokenizer.read_bytes()
        if sha256_bytes(data) != model.tokenizer_sha256:
            raise IdentityError("refused: the model's tokenizer does not match tokenizer_sha256")
        root.write_bytes(f"{staging}/{TOKENIZER_FILE}", data)
        entry["tokenizer"] = TOKENIZER_FILE
    entry["tokenizer_sha256"] = model.tokenizer_sha256
    return entry


def export(
    root: Root,
    voice_id: str,
    clips: Sequence[ExportClip],
    *,
    status: str = PLACEHOLDER,
    model: ExportModel | None = None,
    rate: int | None = EXPORT_RATE_HZ,
) -> Identity:
    """Write `voices/<voice_id>/`, then check it as v6 would load it.

    Each clip is copied as `<state>.wav` and `<state>.private.txt`. With `model`, the identity
    is schema 2: the weights are copied as `model.safetensors` and a non-stock tokenizer as
    `tokenizer.json`. The files are written into a hidden folder beside the destination and
    renamed into place in one step, so a refused or interrupted export leaves no half-written
    identity behind. `rate` is the sample rate every clip must have; `None` takes any rate v6
    allows, for clips copied from an identity that already passed v6's rules.

    Raises:
        IdentityError: The id or status is malformed, the directory exists, a clip is not PCM16
            mono of 3-30 s at `rate`, a transcript or tokenizer does not match its sha256,
            `neutral` is missing, or the model's files are missing.
    """
    check_voice_id(voice_id)
    if status not in STATUSES:
        raise IdentityError(f"refused: status is not one of {', '.join(STATUSES)}")
    rel = voice_rel(voice_id)
    dest = root.writable(rel)
    if os.path.lexists(dest):
        raise IdentityError(f"refused: {rel} already exists; an export never replaces one")
    states = [clip.state for clip in clips]
    if NEUTRAL_STATE not in states or len(set(states)) != len(states):
        raise IdentityError(f"refused: an identity needs one clip per state, {NEUTRAL_STATE} too")
    entries: list[dict[str, str]] = []
    payload: dict[str, bytes] = {}
    for clip in clips:
        if clip.state not in SPEAKING_STATES:
            raise IdentityError("refused: a clip's state is not a speaking state")
        check_wav(clip.audio, f"the {clip.state} reference", rate=rate)
        if not clip.transcript.is_file():
            raise IdentityError(f"refused: the {clip.state} reference has no transcript")
        text = clip.transcript.read_bytes()
        if sha256_bytes(text) != clip.transcript_sha256:
            raise IdentityError(f"refused: the {clip.state} transcript does not match its sha256")
        check_transcript(text, f"the {clip.state} transcript")
        audio = clip.audio.read_bytes()
        audio_name, text_name = f"{clip.state}.wav", f"{clip.state}.private.txt"
        payload[audio_name], payload[text_name] = audio, text
        entries.append(
            {
                "state": clip.state,
                "audio": audio_name,
                "transcript": text_name,
                "audio_sha256": sha256_bytes(audio),
                "transcript_sha256": clip.transcript_sha256,
            }
        )
    staging = f"{VOICES_DIR}/.{voice_id}.export-{secrets.token_hex(4)}"
    try:
        for name, data in payload.items():
            root.write_bytes(f"{staging}/{name}", data)
        manifest: dict[str, object] = {
            "schema": SCHEMA if model is None else MODEL_SCHEMA,
            "voice_id": voice_id,
            "status": status,
            "clips": entries,
        }
        if model is not None:
            manifest["model"] = _model_entry(root, staging, model)
        text = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
        root.write_bytes(f"{staging}/{MANIFEST}", text)
        os.replace(root.writable(staging), dest)
    finally:
        leftover = root.path / staging
        if leftover.exists():
            shutil.rmtree(leftover)
    return check(dest)
