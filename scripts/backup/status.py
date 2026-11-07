"""The backup status file: when the scheduler last tried, what happened, and the last success.

Plan v1, Interfaces and contracts item 6. Two processes depend on this file and
neither may assume the other is careful:

- `scripts.backup.scheduler` writes it after every attempt and reads it to
  decide whether the next one is due.
- `BackupProbe` in the backend reads it for `/health`, at any moment, from
  another container.

WHY THE WRITE IS ATOMIC. A reader that catches a half-written file sees
unparsable JSON, which `/health` reports as `query_failed` and the scheduler
treats as "no success", so a torn write would trigger a dump and a false alarm.
The file is therefore written to a temporary file in the SAME directory and
moved over the old one with `os.replace`, which is a single rename on one
filesystem: a reader sees the old file or the new one, never a mix. The same
directory matters because a rename across filesystems is not atomic and
`os.replace` refuses it.

WHY A CORRUPT FILE READS AS "NO STATUS" AND NOT AS AN ERROR. The scheduler's only
safe reaction to a file it cannot trust is to take a backup: if the last success
cannot be established, assuming one is recent could skip backups indefinitely.
The next attempt then overwrites the file. The corruption is logged so the
cause is not lost.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from .errors import BackupError

logger = logging.getLogger(__name__)

STATUS_VERSION = 1

# Item 6's keys, in the order they are written. `BackupProbe` treats a file
# missing any of them as `query_failed`, so the writer emits every key, with
# null before the first event, rather than omitting the ones it lacks.
STATUS_KEYS: tuple[str, ...] = (
    "v",
    "last_attempt_at",
    "last_result",
    "last_exit_code",
    "last_success_at",
    "last_label",
    "last_duration_s",
    "retained",
    "last_success_counts",
)

ResultName = Literal["ok", "failed", "refused", "suspect"]
RESULT_OK: ResultName = "ok"
RESULT_FAILED: ResultName = "failed"
RESULT_REFUSED: ResultName = "refused"
RESULT_SUSPECT: ResultName = "suspect"
RESULTS: frozenset[str] = frozenset({RESULT_OK, RESULT_FAILED, RESULT_REFUSED, RESULT_SUSPECT})

_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

# The permissions an ordinary write gets under the default umask. `mkstemp`
# creates its file 0600, and the status file is read by the backend container,
# so it is widened to what a plain `open(path, "w")` would have produced.
_STATUS_FILE_MODE = 0o644


class StatusFileError(BackupError):
    """Raised when a status file's content does not have item 6's shape."""


@dataclass(frozen=True, slots=True)
class SuccessCounts:
    """What the last success held, kept so the next attempt can be compared with it.

    Attributes:
        graph_nodes: `graph.nodes` from the success's manifest.
        vault_corpus_files: `vault.corpus_file_count` from that manifest.
        stores: The stores whose manifest entry had a non-null `file`. Key
            membership is not presence: the manifest's `stores` map names every
            live store whether or not it was captured
            (`scripts/backup/stores.py:46-50,86`).
    """

    graph_nodes: int
    vault_corpus_files: int
    stores: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Render as the `last_success_counts` object."""
        return {
            "graph_nodes": self.graph_nodes,
            "vault_corpus_files": self.vault_corpus_files,
            "stores": list(self.stores),
        }

    @classmethod
    def from_dict(cls, raw: object) -> SuccessCounts:
        """Rebuild from the `last_success_counts` object.

        Raises:
            StatusFileError: When a key is missing or has the wrong type.
        """
        if not isinstance(raw, dict):
            raise StatusFileError(f"last_success_counts is {type(raw).__name__}, not an object")
        stores = raw.get("stores")
        if not isinstance(stores, list) or not all(isinstance(name, str) for name in stores):
            raise StatusFileError("last_success_counts.stores is not a list of filenames")
        return cls(
            graph_nodes=_require_int(raw.get("graph_nodes"), "last_success_counts.graph_nodes"),
            vault_corpus_files=_require_int(
                raw.get("vault_corpus_files"), "last_success_counts.vault_corpus_files"
            ),
            stores=tuple(stores),
        )


@dataclass(frozen=True, slots=True)
class BackupStatus:
    """The content of the status file. Every field is None before its first event."""

    last_attempt_at: datetime | None = None
    last_result: ResultName | None = None
    last_exit_code: int | None = None
    last_success_at: datetime | None = None
    last_label: str | None = None
    last_duration_s: float | None = None
    retained: int | None = None
    last_success_counts: SuccessCounts | None = None

    def to_dict(self) -> dict[str, object]:
        """Render with exactly `STATUS_KEYS`, in that order."""
        return {
            "v": STATUS_VERSION,
            "last_attempt_at": _format_optional(self.last_attempt_at),
            "last_result": self.last_result,
            "last_exit_code": self.last_exit_code,
            "last_success_at": _format_optional(self.last_success_at),
            "last_label": self.last_label,
            "last_duration_s": self.last_duration_s,
            "retained": self.retained,
            "last_success_counts": (
                None if self.last_success_counts is None else self.last_success_counts.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, raw: object) -> BackupStatus:
        """Rebuild from parsed JSON, refusing anything that is not item 6's shape.

        Raises:
            StatusFileError: When the object is not a JSON object, lacks a key,
                declares another version, or holds a value of the wrong type.
        """
        if not isinstance(raw, dict):
            raise StatusFileError(f"the file holds {type(raw).__name__}, not a JSON object")
        missing = [key for key in STATUS_KEYS if key not in raw]
        if missing:
            raise StatusFileError(f"keys missing: {missing}")
        if raw["v"] != STATUS_VERSION or isinstance(raw["v"], bool):
            raise StatusFileError(f"version {raw['v']!r}; this build reads {STATUS_VERSION}")
        result = raw["last_result"]
        if result is not None and result not in RESULTS:
            raise StatusFileError(f"last_result {result!r} is not one of {sorted(RESULTS)}")
        label = raw["last_label"]
        if label is not None and not isinstance(label, str):
            raise StatusFileError("last_label is not a string")
        duration = raw["last_duration_s"]
        if duration is not None and (
            isinstance(duration, bool) or not isinstance(duration, int | float)
        ):
            raise StatusFileError("last_duration_s is not a number")
        counts = raw["last_success_counts"]
        return cls(
            last_attempt_at=_parse_optional(raw["last_attempt_at"], "last_attempt_at"),
            last_result=result,
            last_exit_code=_optional_int(raw["last_exit_code"], "last_exit_code"),
            last_success_at=_parse_optional(raw["last_success_at"], "last_success_at"),
            last_label=label,
            last_duration_s=None if duration is None else float(duration),
            retained=_optional_int(raw["retained"], "retained"),
            last_success_counts=None if counts is None else SuccessCounts.from_dict(counts),
        )


def format_utc(moment: datetime) -> str:
    """ISO-8601 UTC with a `Z` suffix at seconds resolution, as the dump's manifests use."""
    return moment.astimezone(UTC).strftime(_TIMESTAMP_FORMAT)


def parse_utc(text: str) -> datetime:
    """Parse an ISO-8601 timestamp into an aware UTC datetime; a naive one is taken as UTC.

    Raises:
        ValueError: When `text` is not ISO-8601.
    """
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def write_status(path: Path, status: BackupStatus) -> None:
    """Replace the status file atomically: temporary file in the same directory, then rename.

    The temporary file is removed when anything before the rename fails, so a
    failed write leaves the previous file exactly as it was and no debris
    beside it.

    Raises:
        OSError: When the directory is missing or not writable, or the rename
            fails. The previous file is untouched in every case.
    """
    payload = json.dumps(status.to_dict(), indent=2) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    replaced = False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, _STATUS_FILE_MODE)
        os.replace(temporary, path)
        replaced = True
    finally:
        # `finally` rather than an except arm, so an interrupt between the write
        # and the rename does not leave a temporary file behind either.
        if not replaced:
            Path(temporary).unlink(missing_ok=True)


def read_status(path: Path) -> BackupStatus | None:
    """Read the status file; None when it is absent or cannot be trusted.

    Never raises for a bad file: a corrupt one is logged with its path and the
    reason and reads as None, which the scheduler treats as "no success" (see
    the module docstring).
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        logger.warning(
            "[backup-status] %s could not be read (%s); treating it as no success. "
            "The next attempt overwrites it.",
            path,
            exc,
        )
        return None
    try:
        return BackupStatus.from_dict(json.loads(text))
    except (ValueError, StatusFileError) as exc:
        logger.warning(
            "[backup-status] %s is corrupt (%s); treating it as no success. "
            "The next attempt overwrites it.",
            path,
            exc,
        )
        return None


def _format_optional(moment: datetime | None) -> str | None:
    return None if moment is None else format_utc(moment)


def _parse_optional(value: object, key: str) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise StatusFileError(f"{key} is not a timestamp string")
    try:
        return parse_utc(value)
    except ValueError as exc:
        raise StatusFileError(f"{key} {value!r} is not ISO-8601") from exc


def _require_int(value: object, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StatusFileError(f"{key} is not an integer")
    return value


def _optional_int(value: object, key: str) -> int | None:
    return None if value is None else _require_int(value, key)
