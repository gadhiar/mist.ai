"""Decide WHERE a backup is written, and refuse every destination that dies with its source.

This is the most consequential module in the package. A dump that writes the
wrong bytes is a bad backup; a dump that writes good bytes to a doomed directory
is not a backup at all, and it looks identical until the day it is reached for.

THERE IS NO DEFAULT DESTINATION. `MIST_BACKUP_ROOT` is required, and an unset
variable is a loud refusal rather than a fallback path. The fallback is what
this package exists to remove: `scripts/mist_admin.py graph-backup` defaulted to
`data/graph_snapshots/`
(`git show ebe1b0d:scripts/mist_admin.py | grep -n 'data/graph_snapshots'` ->
:427), which is INSIDE the live state root, so a single `rm -rf data` took the
event store, the extraction cache, the vault sidecar and every graph backup
together.

TWO GUARDS.

1. `assert_isolated_root`, imported UNMODIFIED from
   `backend/knowledge/eval_isolation.py:539`. It already refuses `data/`,
   `mist-memory/`, `/app/data`, `/app/mist-memory`, `~/.mist`, a filesystem or
   drive root, and the home directory itself. Its refusal text is preserved
   verbatim when it is re-raised as a `BackupDestinationError`, so a caller has
   one exception type to handle and the operator still reads the original
   reason.

2. A repository arm, implemented here, for the case the shared guard does not
   cover. `assert_isolated_root` refuses the repo root itself through its
   contains-arm, because the repo contains `data/`
   (`grep -n 'REPO_ROOT / "data"' backend/knowledge/eval_isolation.py` -> :531).
   It does NOT refuse `<repo>/backups`: that path is not equal to a live root,
   no live root is among its parents, and it is not among any live root's
   parents, so all three arms of the loop at
   `backend/knowledge/eval_isolation.py:570-587` pass it -- which
   `tests/unit/backup/test_destination.py` asserts directly rather than assumes.
   A backup inside the working tree is still lost with the working tree, so it
   is refused here. This arm is additive and lives in this file; the shared
   guard is not widened and must not be.

Every refusal names the path, the reason, and what to set instead, because the
operator reading it is not reading this file.
"""

from __future__ import annotations

import os
from pathlib import Path

from backend.knowledge.eval_isolation import REPO_ROOT, IsolatedRootError, assert_isolated_root

from .errors import BackupDestinationError

BACKUP_ROOT_ENV = "MIST_BACKUP_ROOT"

_WHAT_TO_DO = (
    f"Set {BACKUP_ROOT_ENV} to a directory on storage that does not fail with this "
    "machine's working copy -- an external disk, a NAS mount, or a remote "
    f"filesystem -- for example `export {BACKUP_ROOT_ENV}=/mnt/backup/mist`, then "
    "re-run. Nothing was written."
)


def assert_backup_destination(path: Path | str, *, purpose: str = "backup") -> Path:
    """Refuse a destination inside the repository, or on live state; return it resolved.

    Args:
        path: Candidate destination directory. Resolved before any check, so a
            symlink or a `..` segment cannot smuggle a path past the guards.
        purpose: Named in the refusal so an operator knows which tool refused.

    Returns:
        The resolved destination. Nothing is created.

    Raises:
        BackupDestinationError: When the destination is the repository, sits
            under it, or fails `assert_isolated_root`.
    """
    resolved = Path(path).expanduser().resolve()

    # Arm order is chosen for the MESSAGE, not for coverage: every path refused
    # by one of these is refused by the others' absence too, so what an arm's
    # position decides is which sentence the operator reads at 3am.
    #
    # The repo root gets this arm rather than the shared guard's contains-arm,
    # whose text is about a restore clearing its target -- true for hydration,
    # confusing here.
    if resolved == REPO_ROOT:
        raise BackupDestinationError(
            f"refusing {purpose} destination {resolved}: it IS the MIST.AI working "
            f"tree, which holds the live state this backup exists to outlive. "
            f"{_WHAT_TO_DO}"
        )

    # Live state next, so `<repo>/data/graph_snapshots` is refused as LIVE STATE
    # rather than merely as a path inside the repository. Both are true; the
    # first is the one worth telling the operator.
    try:
        assert_isolated_root(resolved, purpose=purpose)
    except IsolatedRootError as exc:
        # Re-raised, not re-implemented. The guard's own text names the path and
        # which live directory it collided with; this only adds the remedy.
        raise BackupDestinationError(f"{exc} {_WHAT_TO_DO}") from exc

    # The new arm, and the only one that catches `<repo>/backups`.
    if REPO_ROOT in resolved.parents:
        raise BackupDestinationError(
            f"refusing {purpose} destination {resolved}: it sits inside the MIST.AI "
            f"working tree at {REPO_ROOT}. A backup kept in the repository is "
            "destroyed by the same deletion, disk failure or bad checkout that "
            f"destroys the repository, so it cannot restore it. {_WHAT_TO_DO}"
        )

    return resolved


def resolve_backup_root(explicit: Path | str | None = None, *, purpose: str = "backup") -> Path:
    """Resolve the backup root from an explicit path or `MIST_BACKUP_ROOT`.

    Args:
        explicit: An operator-supplied destination, usually from `--output`.
            When given it replaces the environment variable entirely, and is
            checked by the same guards.
        purpose: Named in the refusal.

    Returns:
        The resolved, guarded root. Nothing is created.

    Raises:
        BackupDestinationError: When neither source supplies a destination, or
            the destination is refused.
    """
    if explicit is not None:
        return assert_backup_destination(explicit, purpose=purpose)

    raw = os.environ.get(BACKUP_ROOT_ENV, "").strip()
    if not raw:
        # An empty or whitespace-only value counts as unset: `export
        # MIST_BACKUP_ROOT=` in a shell profile, or a Compose `environment:`
        # entry with no value, both produce "" rather than absence, and
        # `Path("")` resolves to the current directory -- which for a backup run
        # started from the repo root is the repository itself.
        raise BackupDestinationError(
            f"{BACKUP_ROOT_ENV} is not set and this tool has NO default destination. "
            "Refusing to guess: a backup written beside the state it protects is "
            "lost with it, and that failure is invisible until a restore is "
            f"attempted. {_WHAT_TO_DO}"
        )
    return assert_backup_destination(raw, purpose=purpose)


def resolve_backup_file(
    filename: str,
    explicit: Path | str | None = None,
    *,
    purpose: str = "backup",
) -> Path:
    """Resolve a single output FILE, guarding the directory it would land in.

    The guards take directories, so an explicit file path is checked through its
    parent. That is what makes `--output data/graph_snapshots/x.json` a refusal:
    the parent sits under the live `data/` root.

    Args:
        filename: Name to use under the backup root when `explicit` is None.
        explicit: An operator-supplied file path.
        purpose: Named in the refusal.

    Returns:
        The resolved file path. Neither it nor its parent is created.

    Raises:
        BackupDestinationError: When the destination directory is refused, or no
            destination is available.
    """
    if explicit is None:
        return resolve_backup_root(purpose=purpose) / filename
    candidate = Path(explicit).expanduser().resolve()
    assert_backup_destination(candidate.parent, purpose=purpose)
    return candidate
