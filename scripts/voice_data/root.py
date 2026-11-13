"""The private root: where it is, how it is marked, and the rules every writer follows.

Voice data is Raj's private material, and this repository is public. The root therefore has no
default location, must sit outside every git work tree (so no `git add` can reach it), and is
written only through `Root.write_bytes`, which refuses any path that resolves outside the root or
under `seed/raw/`, where Raj's original episode files stay untouched.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .errors import RootError

ENV_ROOT = "MIST_VOICE_PRIVATE_ROOT"
MARKER = ".mist-voice-private"
MARKER_SCHEMA = 1
KINDS = ("private", "rehearsal")
SEED_RAW = "seed/raw"

# Directories `init` creates. `voices/`, `design/` and `bench/` belong to goal mist-realtime-voice;
# they are created when missing so both goals see one layout.
LAYOUT = (
    "voices",
    "design",
    "bench",
    "guard",
    "config",
    "seed/raw",
    "seed/reference",
    "seed/tracks",
    "seed/runs",
    "targets",
    "text",
    "teachers/bakeoff",
    "corpus",
    "students",
    "packs",
    "runs",
    "cache",
)

# What a private root may already hold before `init`: the realtime goal's directories, Raj's
# episodes and reference clip, and his private terms list, all of which can predate `init`.
_V6_DIRS = frozenset({"voices", "design", "bench"})
_SEED_ENTRIES = frozenset({"raw", "reference"})
_GUARD_ENTRIES = frozenset({"terms.txt"})
# A rehearsal root may already hold only H5's public downloads.
_REHEARSAL_ENTRIES = frozenset({"public"})


@dataclass(frozen=True, slots=True)
class Root:
    """An opened, marked root.

    Attributes:
        path: The resolved root directory.
        kind: `private` or `rehearsal`, from the marker.
        created: The marker's ISO timestamp.
    """

    path: Path
    kind: str
    created: str

    def writable(self, target: str | os.PathLike[str]) -> Path:
        """Resolve a write target and refuse it unless it is a legal place to write.

        Resolving first means a symlink cannot carry a write out of the root or into
        `seed/raw/`. The git check runs on every write, not only at open, because a `.git` can
        appear after the root was opened.

        Args:
            target: A root-relative or absolute path.

        Returns:
            The resolved absolute path.

        Raises:
            RootError: The root sits in a git work tree, or the path resolves outside the root
                or under `seed/raw/`.
        """
        check_no_git(self.path)
        candidate = Path(target)
        if not candidate.is_absolute():
            candidate = self.path / candidate
        resolved = candidate.resolve()
        if resolved == self.path or not resolved.is_relative_to(self.path):
            raise RootError("refused: a write path resolves outside the root")
        raw = self.path / SEED_RAW
        for raw_dir in (raw, raw.resolve()):
            if resolved == raw_dir or resolved.is_relative_to(raw_dir):
                raise RootError("refused: seed/raw is read-only to the package")
        return resolved

    def write_bytes(self, target: str | os.PathLike[str], data: bytes) -> Path:
        """Write a file atomically: a temporary file beside it, then `os.replace`.

        A reader therefore sees the old file or the new one, never a partial write, and a crash
        leaves no half-written record behind.

        Args:
            target: A root-relative or absolute path inside the root.
            data: The complete file content.

        Returns:
            The resolved path written.

        Raises:
            RootError: `writable` refused the path.
        """
        dest = self.writable(target)
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, dest)
        finally:
            if os.path.lexists(temp):
                os.unlink(temp)
        return dest

    def rel(self, path: str | os.PathLike[str]) -> str:
        """Return a path relative to the root, in posix form, for output.

        Output names root-relative paths only, so the root's own location (and the user name in
        it) never reaches a log.
        """
        resolved = Path(path).resolve()
        if not resolved.is_relative_to(self.path):
            raise RootError("refused: a path resolves outside the root")
        return resolved.relative_to(self.path).as_posix()


def check_no_git(path: Path) -> None:
    """Refuse a root that is, or sits inside, a git work tree.

    Raises:
        RootError: The path or a parent holds a `.git` file or directory.
    """
    for directory in (path, *path.parents):
        if os.path.lexists(directory / ".git"):
            raise RootError(
                "refused: the root or a parent holds .git; the root must sit outside every "
                "git work tree"
            )


def resolve_root_path(arg: str | None, environ: Mapping[str, str] | None = None) -> Path:
    """Return the root path from `--root` or the environment, with no default.

    Raises:
        RootError: Neither is set.
    """
    env = os.environ if environ is None else environ
    value = arg or env.get(ENV_ROOT)
    if not value:
        raise RootError(f"refused: the root is unset; pass --root or set {ENV_ROOT}")
    return Path(value).resolve()


def open_root(arg: str | None, environ: Mapping[str, str] | None = None) -> Root:
    """Open a marked root.

    Raises:
        RootError: The root is unset, in a git work tree, or has no readable marker.
    """
    path = resolve_root_path(arg, environ)
    check_no_git(path)
    return _read_marker(path)


def init_root(path: str | os.PathLike[str], kind: str) -> Root:
    """Create the marker and the missing layout, touching nothing already present.

    On a root already marked with the same kind it only adds missing layout directories, so a
    rerun after an interrupted `init` completes it.

    Raises:
        RootError: An unknown kind, a git work tree, a root marked with the other kind, or a
            non-empty unmarked directory holding anything `init` does not accept.
    """
    if kind not in KINDS:
        raise RootError(f"refused: unknown kind; use one of {', '.join(KINDS)}")
    target = Path(path).resolve()
    check_no_git(target)
    if target.exists() and not target.is_dir():
        raise RootError("refused: the root is not a directory")
    if (target / MARKER).exists():
        root = _read_marker(target)
        if root.kind != kind:
            raise RootError(f"refused: the root is already marked kind {root.kind}")
    else:
        if target.exists() and not _accepts_existing(target, kind):
            raise RootError(
                f"refused: init found entries a {kind} root does not accept in this unmarked "
                "directory"
            )
        target.mkdir(parents=True, exist_ok=True)
        created = datetime.now(UTC).isoformat(timespec="seconds")
        root = Root(path=target, kind=kind, created=created)
        marker = {"schema": MARKER_SCHEMA, "kind": kind, "created": created}
        # The marker goes first, so an interrupted init is completed by a rerun, not refused.
        root.write_bytes(MARKER, (json.dumps(marker) + "\n").encode("utf-8"))
    for rel in LAYOUT:
        (target / rel).mkdir(parents=True, exist_ok=True)
    return root


def _accepts_existing(path: Path, kind: str) -> bool:
    with os.scandir(path) as listing:
        entries = {entry.name: entry for entry in listing}
    if kind == "rehearsal":
        return set(entries) <= _REHEARSAL_ENTRIES and all(e.is_dir() for e in entries.values())
    for name, entry in entries.items():
        if name in _V6_DIRS:
            accepted = entry.is_dir()
        elif name == "seed":
            accepted = entry.is_dir() and _holds_only(Path(entry.path), _SEED_ENTRIES, dirs=True)
        elif name == "guard":
            accepted = entry.is_dir() and _holds_only(Path(entry.path), _GUARD_ENTRIES, dirs=False)
        else:
            accepted = False
        if not accepted:
            return False
    return True


def _holds_only(directory: Path, names: frozenset[str], *, dirs: bool) -> bool:
    with os.scandir(directory) as entries:
        return all(
            entry.name in names and (entry.is_dir() if dirs else entry.is_file())
            for entry in entries
        )


def _read_marker(path: Path) -> Root:
    marker = path / MARKER
    if not marker.is_file():
        raise RootError("refused: no marker; run init --kind private or --kind rehearsal")
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RootError("refused: the marker is unreadable") from exc
    if (
        not isinstance(data, dict)
        or set(data) != {"schema", "kind", "created"}
        or data["schema"] != MARKER_SCHEMA
        or data["kind"] not in KINDS
        or not isinstance(data["created"], str)
    ):
        raise RootError("refused: the marker is not a schema 1 voice-data marker")
    return Root(path=path, kind=data["kind"], created=data["created"])


def source_id(number: int) -> str:
    """Return a source id; sources are numbered in sorted-path order by ingest."""
    return f"src-{number:04d}"


def segment_id(number: int) -> str:
    """Return a segment id."""
    return f"seg-{number:06d}"


def take_id(line_id: str, state: str, teacher: str, seed: int) -> str:
    """Return a take id: the first 12 hex digits of the sha256 of the take key.

    The key is hashed as a JSON list, so no choice of field values can make two different keys
    collide by concatenation, and a rerun recomputes the same id to skip takes already present.
    """
    key = json.dumps([line_id, state, teacher, seed]).encode("utf-8")
    return "take-" + hashlib.sha256(key).hexdigest()[:12]


def job_id(when: datetime, number: int) -> str:
    """Return a job id from a timezone-aware time, to the minute in UTC, and a counter."""
    return f"job-{when.astimezone(UTC).strftime('%Y%m%dT%H%MZ')}-{number}"
