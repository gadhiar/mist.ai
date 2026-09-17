"""Delete old backup artifacts, and refuse to delete anything else.

MIS-140 T2. Retention is the only part of this package that deletes, so its
design is entirely about what it REFUSES.

THREE ARMS, AND THE THIRD IS THE ONE THAT MATTERS.

1. AGE COMES FROM THE MANIFEST, NEVER FROM THE FILESYSTEM. Artifacts are ordered
   by `manifest.json`'s `created_at`
   (`grep -n "def created_at_datetime" scripts/backup/manifest.py` -> :141), not
   by directory mtime and not by directory name. An mtime changes when anything
   touches the directory -- a virus scanner, a copy onto a new disk, a `chmod`,
   an rsync that rewrites metadata -- so mtime records when the artifact was
   last DISTURBED, which is unrelated to what it holds. A name is whatever
   `--label` said. The manifest timestamp is the only value that is a statement
   the artifact makes about itself.

2. NEVER BELOW ONE. `--retain 0` is clamped to 1 and says so. An operator who
   asks for zero at 3am is not asking to be left with nothing; a tool that
   obliges is a tool that turns a typo into the incident.

3. NEVER DELETE A DIRECTORY WITHOUT A MIST BACKUP MANIFEST. Every candidate goes
   through `is_backup_artifact_dir`, which is total and never raises
   (`grep -n "def is_backup_artifact_dir" scripts/backup/manifest.py` -> :239).
   This is the arm that stops a mis-pointed `MIST_BACKUP_ROOT` from eating
   unrelated files: pointed at a Documents folder, this prune deletes NOTHING
   and reports every directory it skipped. `test_prune.py` asserts that
   directly. A retention pass is the one job in this package that runs without
   an operator watching, so "refuse what you do not recognise" is not a
   nicety -- it is the only supervision there is.

`.partial` DIRECTORIES ARE PROTECTED BY ARM 3, FOR FREE. A dump builds its
artifact in `<label>.partial` and renames it only after the manifest is written
(`grep -n "working_dir.rename" scripts/backup/dump.py` -> :312), so a directory
carrying that suffix has no manifest and arm 3 skips it.

WHAT TO DO ABOUT ACCUMULATING PARTIALS, since this tool will never reap them:
each one is a dump that DIED, and it is evidence rather than litter. Read the
dump's exit code and log first -- a run of them means the dump leg is failing
and the backup you think you have is not being taken. Once you have a complete
artifact newer than the partial, delete it by hand (`rm -rf <root>/<label>.partial`).
Deleting one also unblocks reusing its `--label`, which the dump refuses while
it exists (`grep -n "already exists" scripts/backup/dump.py` -> :257,270).

Exit codes:
    0  the plan was printed, or the deletions were made
    2  the backup root was refused or is unset -- nothing was read
    1  a deletion failed part way

Usage:
    MIST_BACKUP_ROOT=/mnt/backup/mist python -m scripts.backup.prune
    MIST_BACKUP_ROOT=/mnt/backup/mist python -m scripts.backup.prune --retain 7 --confirm
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .destination import resolve_backup_root
from .errors import BackupDestinationError, BackupError, BackupManifestError
from .manifest import is_backup_artifact_dir, read_manifest

logger = logging.getLogger(__name__)

# How many artifacts survive a prune when the operator names no number.
DEFAULT_RETAIN = 7

# The floor, and it is not configurable. See arm 2 in the module docstring.
MINIMUM_RETAIN = 1

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_DESTINATION_REFUSED = 2


@dataclass(frozen=True, slots=True)
class PruneCandidate:
    """One directory under the backup root that IS a MIST.AI backup artifact."""

    path: Path
    created_at: datetime


@dataclass(frozen=True, slots=True)
class PrunePlan:
    """What a prune would do, computed before anything is deleted.

    Produced by `plan_prune` and printed in full by the CLI. A plan that is
    computed and shown before the destructive step is what lets an operator see
    a mis-pointed root -- `skipped` full of unfamiliar names is the signal.
    """

    root: Path
    retain: int
    requested_retain: int
    keep: tuple[PruneCandidate, ...]
    delete: tuple[PruneCandidate, ...]
    skipped: tuple[Path, ...]


def scan_artifacts(root: Path) -> tuple[list[PruneCandidate], list[Path]]:
    """Split the directories under `root` into backup artifacts and everything else.

    Only immediate children are considered. Recursing would let a backup root
    that happens to contain someone's project tree present that tree's
    subdirectories as candidates.

    Returns:
        `(candidates, skipped)`. A directory lands in `skipped` when
        `is_backup_artifact_dir` says no, or when its manifest parses but its
        `created_at` does not -- an artifact whose age cannot be established is
        never a deletion candidate, because every ordering of it is a guess.
    """
    candidates: list[PruneCandidate] = []
    skipped: list[Path] = []
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if not is_backup_artifact_dir(child):
            skipped.append(child)
            continue
        try:
            created_at = read_manifest(child).created_at_datetime()
        except BackupManifestError:
            skipped.append(child)
            continue
        candidates.append(PruneCandidate(path=child, created_at=created_at))
    return candidates, skipped


def plan_prune(root: Path, retain: int = DEFAULT_RETAIN) -> PrunePlan:
    """Decide which artifacts survive, newest first by manifest `created_at`.

    Args:
        root: The backup root, already guarded by `resolve_backup_root`.
        retain: How many artifacts to keep. Clamped up to `MINIMUM_RETAIN`.

    Returns:
        A `PrunePlan`. Computing it deletes nothing.

    Raises:
        BackupError: When `root` is not a directory. An absent root is a
            configuration error, not an empty backup set, and reporting "0
            artifacts, nothing to do" would hide it.
    """
    if not root.is_dir():
        raise BackupError(
            f"backup root {root} is not a directory. Refusing to report an empty "
            "retention plan for a path that does not exist -- that reads like a "
            "clean run and means the backups are somewhere else, or nowhere."
        )
    effective = max(int(retain), MINIMUM_RETAIN)
    candidates, skipped = scan_artifacts(root)
    # Newest first. Ties break on path so two artifacts written in the same
    # second order deterministically rather than by directory iteration order.
    ordered = sorted(candidates, key=lambda c: (c.created_at, str(c.path)), reverse=True)
    return PrunePlan(
        root=root,
        retain=effective,
        requested_retain=int(retain),
        keep=tuple(ordered[:effective]),
        delete=tuple(ordered[effective:]),
        skipped=tuple(skipped),
    )


def apply_prune(plan: PrunePlan) -> list[Path]:
    """Delete the artifacts the plan names, and only those.

    Re-checks `is_backup_artifact_dir` on each directory immediately before
    removing it. The plan was computed earlier and the check is cheap; what it
    buys is that a directory swapped, replaced or emptied between planning and
    deletion is not removed on the strength of a stale observation.

    Returns:
        The directories actually removed.

    Raises:
        BackupError: When a removal fails, naming what had already been removed.
    """
    removed: list[Path] = []
    for candidate in plan.delete:
        if not is_backup_artifact_dir(candidate.path):
            logger.warning(
                "[prune] %s stopped looking like a backup artifact between the plan "
                "and the deletion; leaving it alone.",
                candidate.path,
            )
            continue
        try:
            shutil.rmtree(candidate.path)
        except OSError as exc:
            raise BackupError(
                f"could not delete {candidate.path}: {exc}. Already removed: "
                f"{[str(p) for p in removed]}. The retained artifacts are untouched."
            ) from exc
        removed.append(candidate.path)
    return removed


def build_parser() -> argparse.ArgumentParser:
    """Build the prune CLI parser. `--confirm` is required before anything is deleted."""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.backup.prune",
        description=(
            "Delete backup artifacts older than the newest --retain of them, ordered "
            "by each artifact's own manifest timestamp."
        ),
        epilog=(
            "Without --confirm this prints the plan and deletes nothing. A directory "
            "with no MIST.AI backup manifest is never deleted, whatever its age or "
            "name -- that includes .partial directories, which carry no manifest by "
            "construction. --retain is clamped up to 1. Exit codes: 0 planned or "
            "pruned; 2 backup root unset or refused; 1 a deletion failed."
        ),
    )
    parser.add_argument(
        "--root",
        default=None,
        help=(
            "Backup root to prune. Overrides MIST_BACKUP_ROOT, which is otherwise "
            "required -- this tool has no default."
        ),
    )
    parser.add_argument(
        "--retain",
        type=int,
        default=DEFAULT_RETAIN,
        help=f"How many artifacts to keep (default: {DEFAULT_RETAIN}; never below 1).",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually delete. Without it the plan is printed and nothing is removed.",
    )
    parser.add_argument("--verbose", action="store_true", help="Log at DEBUG level.")
    return parser


def _print_plan(plan: PrunePlan) -> None:
    """Print the plan in full, including every directory that was skipped."""
    if plan.requested_retain < plan.retain:
        print(
            f"[prune] --retain {plan.requested_retain} raised to {plan.retain}: this "
            "tool never prunes below the most recent artifact."
        )
    print(f"[prune] {plan.root}: keeping {len(plan.keep)}, would delete {len(plan.delete)}.")
    for candidate in plan.keep:
        print(f"[prune]   KEEP   {candidate.path.name}  created_at {candidate.created_at}")
    for candidate in plan.delete:
        print(f"[prune]   DELETE {candidate.path.name}  created_at {candidate.created_at}")
    for path in plan.skipped:
        print(
            f"[prune]   SKIP   {path.name}  no readable MIST.AI backup manifest; "
            "never deleted by this tool"
        )
    if plan.skipped and not plan.keep:
        print(
            "[prune] WARNING: this root holds no MIST.AI backup artifacts at all, only "
            f"{len(plan.skipped)} directory(ies) this tool does not recognise. Check "
            "MIST_BACKUP_ROOT points where you think it does."
        )


def main(argv: list[str] | None = None) -> int:
    """Plan a prune, and delete only when `--confirm` is passed."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        root = resolve_backup_root(args.root, purpose="prune")
    except BackupDestinationError as exc:
        print(f"[prune] REFUSED: {exc}", file=sys.stderr)
        return EXIT_DESTINATION_REFUSED

    try:
        plan = plan_prune(root, args.retain)
    except BackupError as exc:
        print(f"[prune] FAILED: {exc}", file=sys.stderr)
        return EXIT_FAILED

    _print_plan(plan)

    if not args.confirm:
        print("[prune] Nothing deleted. Re-run with --confirm to apply this plan.")
        return EXIT_OK

    try:
        removed = apply_prune(plan)
    except BackupError as exc:
        print(f"[prune] FAILED: {exc}", file=sys.stderr)
        return EXIT_FAILED

    print(f"[prune] Deleted {len(removed)} artifact(s); {len(plan.keep)} remain.")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
