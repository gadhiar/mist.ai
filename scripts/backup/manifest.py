"""What a backup directory says about itself, and the one version that is enforced.

TWO KINDS OF VERSION LIVE IN THIS MANIFEST AND THEY ARE TREATED OPPOSITELY.

`layout_version` describes the SHAPE of the artifact directory -- which files
exist and what the manifest keys mean. It IS checked on read, because code that
cannot parse a layout can only guess at it, and a guess during a restore loses a
field silently.

`stamps` (`ontology_version`, `extraction_version`, `model_hash`) describe the
BUILD that took the backup. They are recorded for audit and NEVER enforced.
This is deliberately the opposite of `SnapshotManifest.assert_fresh`
(`grep -n "def assert_fresh" scripts/hydration/manifest.py` -> :242), which
refuses a hydration fixture whose stamps drifted. That is correct for a fixture
that must match the code under test, and catastrophic here: a disaster-recovery
artifact that self-invalidates on an `EXTRACTION_VERSION` bump fails at the one
moment it is reached for. The graph leg already applies this rule to its own
envelope (`grep -n "deliberately not enforced" backend/knowledge/graph_artifact.py`
-> :412); the manifest applies it to the whole directory.

FOR THE RETENTION LEG (MIS-140 T2). Two entry points exist for pruning:
`is_backup_artifact_dir` is a cheap, exception-free predicate answering "is this
a MIST backup directory?", and `BackupManifest.created_at_datetime` returns the
timezone-aware UTC creation time to compare an age against. A directory that
fails the predicate has no manifest this code wrote, and must not be deleted by
age, because its age is unknown. `PARTIAL_SUFFIX` marks a dump that died before
its manifest was written; such a directory is never a valid artifact and never
carries a `created_at`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .errors import BackupManifestError

MANIFEST_FILENAME = "manifest.json"

# Identifies a directory as this tool's output. A string rather than mere
# presence of `manifest.json`, because `scripts/hydration/snapshot.py` writes a
# file of that name too (`grep -n "MANIFEST_FILENAME =" scripts/hydration/manifest.py`
# -> :63) and a retention pass must not confuse the two.
BACKUP_LAYOUT = "mist.backup"

# Bump when the set of files in an artifact directory changes, or when a
# manifest key changes meaning. Readers refuse an unknown value rather than
# guessing at it.
BACKUP_LAYOUT_VERSION = 1

# Appended to an artifact directory while it is being written. The finished
# directory is produced by a rename, so a directory carrying this suffix is a
# dump that failed part way and holds no manifest.
PARTIAL_SUFFIX = ".partial"

_HASH_BLOCK_BYTES = 65536


def utc_now_iso() -> str:
    """Current UTC time as ISO-8601 with a `Z` suffix, seconds resolution."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    """sha256 of one file, read in blocks so a multi-megabyte store does not load whole."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_HASH_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_artifact_files(artifact_dir: Path) -> dict[str, dict[str, Any]]:
    """Digest every file in the artifact, keyed by POSIX-relative path.

    `manifest.json` is excluded because it is the file being written; a digest
    of it inside itself cannot be computed. Paths are POSIX-relative so an
    artifact written on Windows and verified in the Linux container produces the
    same keys.

    Returns:
        `{"<relative/path>": {"sha256": ..., "bytes": ...}}`, sorted by key.
    """
    digests: dict[str, dict[str, Any]] = {}
    for path in sorted(p for p in artifact_dir.rglob("*") if p.is_file()):
        relative = path.relative_to(artifact_dir).as_posix()
        if relative == MANIFEST_FILENAME:
            continue
        digests[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return digests


@dataclass(frozen=True, slots=True)
class BackupManifest:
    """The self-description written into every completed backup directory."""

    layout: str
    layout_version: int
    created_at: str
    label: str
    git_head: str | None
    stamps: dict[str, str]
    source: dict[str, Any]
    files: dict[str, dict[str, Any]]
    stores: dict[str, dict[str, Any]]
    graph: dict[str, Any]
    vault: dict[str, Any]
    excluded: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return the manifest as a plain dict in the on-disk key order."""
        return {
            "layout": self.layout,
            "layout_version": self.layout_version,
            "created_at": self.created_at,
            "label": self.label,
            "git_head": self.git_head,
            "stamps": dict(self.stamps),
            "source": dict(self.source),
            "files": dict(self.files),
            "stores": dict(self.stores),
            "graph": dict(self.graph),
            "vault": dict(self.vault),
            "excluded": list(self.excluded),
        }

    def to_json(self) -> str:
        """Serialize deterministically: sorted keys, two-space indent, trailing newline."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def write(self, artifact_dir: Path) -> Path:
        """Write `manifest.json` into `artifact_dir` and return the path written."""
        path = artifact_dir / MANIFEST_FILENAME
        path.write_text(self.to_json(), encoding="utf-8", newline="\n")
        return path

    def created_at_datetime(self) -> datetime:
        """Parse `created_at` into a timezone-aware UTC datetime.

        The retention leg compares an artifact's age against a cutoff, and a
        naive datetime silently compares as local time. A value this cannot
        parse raises rather than defaulting to now, because defaulting to now
        would make an unparseable artifact look new and immortal -- or, with the
        opposite default, delete it.

        Raises:
            BackupManifestError: When `created_at` is not ISO-8601.
        """
        raw = self.created_at.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise BackupManifestError(
                f"manifest created_at {self.created_at!r} is not ISO-8601, so this "
                "artifact's age cannot be established. Leave it in place and check "
                "it by hand."
            ) from exc
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, where: str) -> BackupManifest:
        """Rebuild a manifest, refusing a layout this code cannot parse.

        Args:
            raw: Parsed manifest JSON.
            where: Path named in any refusal, so the operator knows which
                artifact was rejected.

        Raises:
            BackupManifestError: When the layout name or layout version is
                absent, or is not the one this build reads. Producer stamps are
                NOT checked here and must not be -- see the module docstring.
        """
        layout = raw.get("layout")
        if layout != BACKUP_LAYOUT:
            raise BackupManifestError(
                f"{where} declares layout {layout!r}; this build reads only "
                f"{BACKUP_LAYOUT!r}. It is not a MIST.AI backup directory."
            )
        version = raw.get("layout_version")
        # `isinstance(True, int)` is True, so a bool would pass an int check and
        # then compare equal to 1.
        if isinstance(version, bool) or not isinstance(version, int):
            raise BackupManifestError(
                f"{where} declares layout_version {version!r}, which is not an "
                f"integer. This build reads layout_version {BACKUP_LAYOUT_VERSION}."
            )
        if version != BACKUP_LAYOUT_VERSION:
            raise BackupManifestError(
                f"{where} is layout_version {version}; this build reads "
                f"{BACKUP_LAYOUT_VERSION}. The artifact layout changed, and it is "
                "not upgraded in place: guessing at another layout's file set is "
                "how a restore silently drops a leg. Use a build that reads this "
                "version."
            )
        return cls(
            layout=str(layout),
            layout_version=version,
            created_at=str(raw.get("created_at", "")),
            label=str(raw.get("label", "")),
            git_head=None if raw.get("git_head") is None else str(raw["git_head"]),
            stamps={str(k): str(v) for k, v in (raw.get("stamps") or {}).items()},
            source=dict(raw.get("source") or {}),
            files=dict(raw.get("files") or {}),
            stores=dict(raw.get("stores") or {}),
            graph=dict(raw.get("graph") or {}),
            vault=dict(raw.get("vault") or {}),
            excluded=list(raw.get("excluded") or []),
        )


def read_manifest(artifact_dir: Path) -> BackupManifest:
    """Load and layout-check the manifest in `artifact_dir`.

    Raises:
        BackupManifestError: When the manifest is absent, is not valid JSON, is
            not a JSON object, or declares a layout this build cannot read.
    """
    path = Path(artifact_dir) / MANIFEST_FILENAME
    if not path.is_file():
        raise BackupManifestError(
            f"{path} not found. A directory with no manifest cannot state what it "
            "holds, when it was taken, or whether it is complete, so it is not "
            "treated as a backup."
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BackupManifestError(f"{path} could not be read as JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise BackupManifestError(f"{path} is not a JSON object; it holds {type(raw).__name__}.")
    return BackupManifest.from_dict(raw, where=str(path))


def is_backup_artifact_dir(path: Path | str) -> bool:
    """Answer "is this a MIST.AI backup directory?" cheaply and without raising.

    The retention leg calls this before considering any directory for deletion,
    so it must be total: an unreadable, truncated or foreign directory returns
    False rather than raising, and False means "do not touch it". It reads only
    `manifest.json` and checks three fields, so scanning a backup root costs one
    small read per directory.

    A layout version this build cannot READ still returns True: the directory is
    unambiguously this tool's output, and answering False would tell a retention
    pass it was foreign rather than merely newer.
    """
    manifest_path = Path(path) / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return False
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(raw, dict) or raw.get("layout") != BACKUP_LAYOUT:
        return False
    version = raw.get("layout_version")
    if isinstance(version, bool) or not isinstance(version, int):
        return False
    created_at = raw.get("created_at")
    return isinstance(created_at, str) and bool(created_at)
