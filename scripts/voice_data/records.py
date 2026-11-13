"""Record models of the private root, and their atomic JSON and JSONL files.

Each record is a frozen slotted dataclass whose `from_json` rejects unknown and missing keys, so a
record written by an older or newer stage fails loudly at the reader instead of being half
understood. Nested maps whose keys vary by run (per-embedder SIM values, per-state target data)
stay plain JSON objects; fixed shapes (prosody, job inputs) are dataclasses of their own.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TypeVar

from .errors import InputError
from .root import Root

_R = TypeVar("_R", bound="_Record")


class _Record:
    """Shared JSON conversion for the record dataclasses.

    `_nested` names the fields holding another record (or a list of them), so `from_json` builds
    those as records too. Lists become tuples, keeping records immutable and comparable.
    """

    __slots__ = ()
    _nested: tuple[tuple[str, type[_Record]], ...] = ()

    def to_json(self) -> dict[str, object]:
        """Return the record as a JSON-ready dict, in field order."""
        return {f.name: _plain(getattr(self, f.name)) for f in fields(self)}  # type: ignore[arg-type]

    @classmethod
    def from_json(cls: type[_R], data: Mapping[str, object]) -> _R:
        """Build the record from a parsed JSON object.

        Raises:
            InputError: `data` is not an object, or has unknown or missing keys.
        """
        if not isinstance(data, Mapping):
            raise InputError(f"{cls.__name__}: expected a JSON object")
        names = [f.name for f in fields(cls)]  # type: ignore[arg-type]
        unknown = sorted(set(data) - set(names))
        if unknown:
            raise InputError(f"{cls.__name__}: unknown keys {unknown}")
        missing = [name for name in names if name not in data]
        if missing:
            raise InputError(f"{cls.__name__}: missing keys {missing}")
        nested = dict(cls._nested)
        values: dict[str, object] = {}
        for name in names:
            value = data[name]
            inner = nested.get(name)
            if inner is not None and isinstance(value, list):
                value = tuple(inner.from_json(item) for item in value)
            elif inner is not None:
                value = inner.from_json(value)  # type: ignore[arg-type]
            elif isinstance(value, list):
                value = tuple(value)
            values[name] = value
        return cls(**values)


def _plain(value: object) -> object:
    if isinstance(value, _Record):
        return value.to_json()
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class Prosody(_Record):
    """A clip's prosody summary, compared per state between seed, takes and the bench."""

    f0_mean_hz: float
    f0_std_hz: float
    rate_sps: float
    energy_db: float


@dataclass(frozen=True, slots=True)
class SourceRecord(_Record):
    """One line of `seed/inventory.jsonl`: a source file and the audio track taken from it.

    `path` is the only place a source file's name appears; every other record names the source
    by `source_id`.
    """

    source_id: str
    path: str
    container: str
    size_bytes: int
    mtime_ns: int
    streams: tuple[dict[str, object], ...]
    stream_index: int
    audio_index: int
    language: str | None
    codec: str
    profile: str | None
    bitrate_bps: int
    bitrate_from: str
    sample_rate: int
    channels: int
    layout: str
    duration_s: float
    lossless: bool
    effective_bandwidth_hz: int
    sha256: str
    track: str
    primary_channel: str


@dataclass(frozen=True, slots=True)
class SegmentRecord(_Record):
    """One line of a run's `segments.jsonl`: a candidate segment and every score it was gated on."""

    segment_id: str
    source_id: str
    start_s: float
    end_s: float
    channel: str
    channel_reason: str | None
    separator: str
    audio: str
    diar_cluster: str | None
    cluster: str
    flag_group: str | None
    sim: dict[str, float]
    overlap_ms: int
    snr_db: float
    c50_db: float
    transcripts: dict[str, str]
    asr_agreement_cer: float
    prosody: Prosody
    arousal: float
    state_proposed: str
    state: str | None
    set: str
    kept: bool
    reasons: tuple[str, ...]

    _nested = (("prosody", Prosody),)


@dataclass(frozen=True, slots=True)
class TargetRecord(_Record):
    """`targets/<target_id>/target.json`: the target voice, its distributions and state floors."""

    schema: int
    target_id: str
    voice_id: str
    mode: str
    seed_run: str | None
    identity: str | None
    embedders: dict[str, str]
    centroids: dict[str, str]
    distribution: dict[str, object]
    windows: dict[str, object]
    mode_check: tuple[dict[str, object], ...]
    states: dict[str, object]


@dataclass(frozen=True, slots=True)
class TakeRecord(_Record):
    """One line of `takes.jsonl`: a generated take and its gate scores."""

    take_id: str
    line_id: str
    state: str
    teacher: str
    seed: int
    round: int
    audio: str
    sim: dict[str, float]
    cer: float
    prosody: Prosody
    arousal: float
    kept: bool
    selected: bool
    reasons: tuple[str, ...]

    _nested = (("prosody", Prosody),)


@dataclass(frozen=True, slots=True)
class JobInput(_Record):
    """A job input: a root-relative path and the sha256 the job checks before it starts."""

    rel: str
    sha256: str


@dataclass(frozen=True, slots=True)
class JobRecord(_Record):
    """`runs/<job_id>/job.json`: what a container job runs, on which inputs, with which env names.

    Env vars appear by name only; a value is read from the environment at run time.
    """

    schema: int
    job_id: str
    family: str
    image: str
    module: str
    args: tuple[str, ...]
    inputs: tuple[JobInput, ...]
    outputs: str
    gpu: bool
    network: bool
    env_names: tuple[str, ...]
    optional_env_names: tuple[str, ...]
    est_hours: float

    _nested = (("inputs", JobInput),)


@dataclass(frozen=True, slots=True)
class LedgerRecord(_Record):
    """One line of `runs/ledger.jsonl`: a rental or a hosted-arm cost entry."""

    entry_id: str
    kind: str
    provider: str
    alias: str | None
    rate_usd_per_h: float | None
    chars: int | None
    usd_per_1k_chars: float | None
    start: str
    end: str | None
    usd: float
    closed: bool
    job_ids: tuple[str, ...]


def _payload(obj: _Record | Mapping[str, object]) -> object:
    return obj.to_json() if isinstance(obj, _Record) else dict(obj)


def write_json(root: Root, target: str | os.PathLike[str], obj: _Record | Mapping) -> Path:
    """Write one record or object as a JSON file, atomically, inside the root.

    Raises:
        RootError: The root refused the path.
    """
    text = json.dumps(_payload(obj), indent=2) + "\n"
    return root.write_bytes(target, text.encode("utf-8"))


def write_jsonl(
    root: Root, target: str | os.PathLike[str], rows: Iterable[_Record | Mapping]
) -> Path:
    """Write records as a JSONL file, one per line, atomically, inside the root.

    The whole file is replaced in one step, so a reader never sees a partial last line.

    Raises:
        RootError: The root refused the path.
    """
    text = "".join(json.dumps(_payload(row)) + "\n" for row in rows)
    return root.write_bytes(target, text.encode("utf-8"))


def read_json(path: Path) -> object:
    """Read and parse a JSON file.

    Raises:
        InputError: The file is missing, unreadable or not JSON.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputError(f"{path.name}: missing or not a JSON file") from exc


def read_record(path: Path, cls: type[_R]) -> _R:
    """Read a JSON file as one record of `cls`.

    Raises:
        InputError: The file is unreadable, or the record has unknown or missing keys.
    """
    return cls.from_json(read_json(path))  # type: ignore[arg-type]


def read_jsonl(path: Path, cls: type[_R]) -> list[_R]:
    """Read a JSONL file as records of `cls`, skipping blank lines.

    Raises:
        InputError: The file is unreadable, a line is not JSON, or a record is malformed.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise InputError(f"{path.name}: missing or unreadable") from exc
    rows: list[_R] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InputError(f"{path.name} line {number}: not JSON") from exc
        rows.append(cls.from_json(data))
    return rows


def sha256_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    """Return a file's sha256 hex digest, read in chunks so episode-sized files fit in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()
