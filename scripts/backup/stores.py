"""Capture the SQLite stores under the live state root, by NAME and never by glob.

THE STORE SET IS ENUMERATED ON PURPOSE. `scripts/hydration/snapshot.py` globs
instead (`grep -n 'dev_root.glob' scripts/hydration/snapshot.py` -> :519), and
that is correct only because `dev-state/` is a directory the hydration tool
creates and owns. The LIVE `./data` is not clean. Listed on the live deployment
it holds five `.db` files: the three named below, plus
`event_store.pre-r1.4-backup-2026-07-31.db` (212992 bytes) and
`event_store.pre-reset-backup-2026-06-09.db` (14331904 bytes). A glob sweeps
both into every backup: the artifact doubles in size, and it gains a second,
older copy of the event store that a hurried restore could load instead of the
real one. `test_stores.py` fixes this with a negative test rather than a
comment.

A named store that is missing is RECORDED as missing rather than skipped
quietly. `extraction_cache.db` is legitimately absent on a fresh deployment, so
absence is not an error -- but a backup that omits a store without saying so is
the failure this package exists to prevent, so the manifest carries
`"present": false` and the dump prints a warning.

WHY `Connection.backup()` AND NOT `shutil.copy`. The stores run in WAL mode, so
the bytes of `foo.db` alone are missing everything still sitting in
`foo.db-wal` -- which is the most recent turns, the ones an operator most wants
back. `Connection.backup()` takes a read lock and produces one consistent file
while the backend keeps serving; the source is opened `file:...?mode=ro` so the
capture cannot write to live state even by accident.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from .errors import BackupError

# The three stores that hold state which cannot be rebuilt from anything else.
# Enumerated, never discovered -- see the module docstring.
LIVE_STORE_FILENAMES: tuple[str, ...] = (
    "event_store.db",
    "extraction_cache.db",
    "vault_sidecar.db",
)

# Subdirectory of the artifact the stores are written into.
STORES_DIRNAME = "stores"


@dataclass(frozen=True, slots=True)
class StoreCapture:
    """The outcome of capturing one named store."""

    filename: str
    present: bool
    destination: Path | None = None
    row_counts: dict[str, int] = field(default_factory=dict)

    def to_manifest_entry(self) -> dict[str, object]:
        """Render this capture as its `stores` entry in the manifest."""
        return {
            "present": self.present,
            "file": None if self.destination is None else f"{STORES_DIRNAME}/{self.filename}",
            "row_counts": dict(self.row_counts),
        }


def _strip_sqlite_sidecars(db_path: Path) -> None:
    """Remove the `-wal`/`-shm` files a WAL database leaves beside itself.

    Called AFTER the row count, not as a tail on the copy. The copy inherits WAL
    mode, so merely REOPENING it recreates both files -- stripping them at copy
    time and then counting rows produces an artifact that still carries an empty
    `-wal` and a 32KB `-shm`. `scripts/hydration/snapshot.py:539-552` records
    having hit exactly that. A clean close checkpoints everything into the
    `.db`, so the sidecars hold nothing, and leaving them in an artifact implies
    to a reader that they carry data.
    """
    for suffix in ("-wal", "-shm"):
        Path(f"{db_path}{suffix}").unlink(missing_ok=True)


def copy_store(source: Path, destination: Path) -> None:
    """Copy one live store to `destination` through the SQLite online backup API.

    ONE SIDE EFFECT ON LIVE STATE, AND IT IS NOT REMOVED. Opening a WAL database
    makes SQLite create its `-shm` shared-memory file even under `mode=ro`, so a
    capture against a store no process is currently holding open leaves an empty
    sidecar beside it (`tests/unit/backup/test_dump.py` asserts that this is the
    only difference, and that no `.db` byte changes). Those sidecars belong to
    the LIVE store and this tool never deletes them: a `-wal` beside a running
    backend holds committed data not yet checkpointed, and removing it would
    discard exactly the recent turns the backup exists to keep. The stripping in
    `_strip_sqlite_sidecars` applies only to the COPY.

    Args:
        source: The live `.db` file. Opened read-only via a `file:` URI.
        destination: Where to write the copy. Replaced if it exists.

    Raises:
        BackupError: When SQLite refuses the read or the write. The live file is
            never opened for writing, so a failure here leaves the source alone.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
        try:
            dst = sqlite3.connect(str(destination))
            try:
                src.backup(dst)
            finally:
                dst.close()
        finally:
            src.close()
    except sqlite3.Error as exc:
        raise BackupError(
            f"could not capture {source}: {exc}. The store was opened read-only, so "
            "live state is unchanged; the artifact is incomplete and must not be "
            "relied on."
        ) from exc


def count_rows(db_path: Path) -> dict[str, int]:
    """Row count for every user table in a store, read from its own schema.

    The table list is read from `sqlite_master` rather than enumerated in this
    file. An enumeration would need editing whenever a store gains a table, and
    a backup whose row counts silently stop covering a new table reports a
    completeness it does not have. `sqlite_%` names are SQLite's own internal
    tables.

    Raises:
        BackupError: When the copied store cannot be read back. That means the
            capture produced a file SQLite itself rejects.
    """
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            names = [
                str(row[0])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )
            ]
            counts: dict[str, int] = {}
            for name in names:
                quoted = '"' + name.replace('"', '""') + '"'
                counts[name] = int(conn.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0])
            return counts
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise BackupError(
            f"captured store {db_path} could not be read back: {exc}. A copy SQLite "
            "cannot open is not a backup of anything."
        ) from exc


def capture_stores(state_root: Path, artifact_dir: Path) -> list[StoreCapture]:
    """Capture each named store from `state_root` into `artifact_dir/stores/`.

    Args:
        state_root: The live state directory, normally `<repo>/data`.
        artifact_dir: The artifact being built.

    Returns:
        One `StoreCapture` per name in `LIVE_STORE_FILENAMES`, in that order,
        including the absent ones.

    Raises:
        BackupError: When a store exists but cannot be copied or read back.
    """
    destination_dir = artifact_dir / STORES_DIRNAME
    captures: list[StoreCapture] = []
    for filename in LIVE_STORE_FILENAMES:
        source = state_root / filename
        if not source.is_file():
            captures.append(StoreCapture(filename=filename, present=False))
            continue
        destination = destination_dir / filename
        copy_store(source, destination)
        row_counts = count_rows(destination)
        # Strip after the count: `count_rows` reopened the copy and recreated
        # the sidecars.
        _strip_sqlite_sidecars(destination)
        captures.append(
            StoreCapture(
                filename=filename,
                present=True,
                destination=destination,
                row_counts=row_counts,
            )
        )
    return captures
