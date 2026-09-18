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

import re
import sqlite3
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

from .errors import BackupError

# Pulled out of SQLite's own wording so a warning can NAME the module a reader
# is missing ("no such module: vec0") instead of quoting the whole error.
_NO_SUCH_MODULE = re.compile(r"no such module: (\S+)")

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
    uncounted_tables: tuple[str, ...] = ()
    missing_modules: tuple[str, ...] = ()

    def to_manifest_entry(self) -> dict[str, object]:
        """Render this capture as its `stores` entry in the manifest.

        `uncounted_tables` is what makes `row_counts` honest at layout version
        2: the counts may now be PARTIAL, and a reader that cannot tell a
        zero-row table from an uncounted one would read a gap as an emptiness.
        """
        return {
            "present": self.present,
            "file": None if self.destination is None else f"{STORES_DIRNAME}/{self.filename}",
            "row_counts": dict(self.row_counts),
            "uncounted_tables": list(self.uncounted_tables),
        }


@dataclass(frozen=True, slots=True)
class StoreReadback:
    """What reading one captured store back established, and what it did not."""

    row_counts: dict[str, int]
    # Table name -> the message SQLite gave when the count was attempted. The
    # reason is kept rather than discarded because "vec0 is not loaded" and
    # "this table is broken" are different operator actions.
    uncounted: dict[str, str]

    @property
    def missing_modules(self) -> tuple[str, ...]:
        """Module names named by the read failures, first-seen order, deduplicated."""
        modules: list[str] = []
        for reason in self.uncounted.values():
            match = _NO_SUCH_MODULE.search(reason)
            if match is not None and match.group(1) not in modules:
                modules.append(match.group(1))
        return tuple(modules)


@dataclass(frozen=True, slots=True)
class UncountedTables:
    """Tables in one captured store this reader could not count, and why."""

    filename: str
    tables: tuple[str, ...]
    missing_modules: tuple[str, ...]

    def warning(self) -> str:
        """One operator-facing line: what was not counted, why, and what to install."""
        modules = ", ".join(self.missing_modules) if self.missing_modules else "an unnamed module"
        return (
            f"{self.filename}: {len(self.tables)} table(s) could not be counted "
            f"-- {', '.join(self.tables)}. This process cannot load {modules}. "
            "The copy passed PRAGMA integrity_check, so the tables are uncounted "
            "because of what this reader lacks, not because of what the file "
            "holds; the artifact itself is complete. Install the module "
            "(pip install 'sqlite-vec>=0.1.3' supplies vec0) and re-run the dump "
            "to record their counts."
        )


def uncounted_from_captures(captures: list[StoreCapture]) -> tuple[UncountedTables, ...]:
    """Collect the captures that left tables uncounted, for the dump's report."""
    return tuple(
        UncountedTables(
            filename=capture.filename,
            tables=capture.uncounted_tables,
            missing_modules=capture.missing_modules,
        )
        for capture in captures
        if capture.uncounted_tables
    )


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


def load_sqlite_vec(conn: sqlite3.Connection) -> bool:
    """Load the sqlite-vec extension onto `conn`; report whether it is available.

    LOADED FOR EVERY STORE, NEVER KEYED ON A FILENAME. `vault_sidecar.db` is
    the store holding `vec0` tables today; gating the load on that name would
    let the next store that gains a virtual table reintroduce MIS-153 with no
    code change to point at.

    Returns:
        True when `vec0` is usable on this connection, False when the extension
        is absent or this interpreter cannot load extensions at all. Never
        raises: a reader's own limits are reported, not imposed on the artifact.
    """
    try:
        import sqlite_vec
    except ImportError:
        return False
    try:
        # A Python built with SQLITE_OMIT_LOAD_EXTENSION has no
        # `enable_load_extension` attribute at all, so this raises
        # AttributeError rather than any sqlite3.Error.
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
    except (AttributeError, sqlite3.Error):
        return False
    finally:
        # Re-closing the door is best effort: on the failure paths above it was
        # never opened, and a store's row counts are not worth an exception
        # raised out of a `finally`.
        with suppress(AttributeError, sqlite3.Error):
            conn.enable_load_extension(False)
    return True


def read_back_store(db_path: Path) -> StoreReadback:
    """Verify one captured store, and count the tables this reader can count.

    `PRAGMA integrity_check` IS THE GATE. The row counts are reporting.

    Those are two different questions and the old row-count loop conflated
    them. A `SELECT COUNT(*)` that fails on `vault_chunks_vec` establishes that
    THIS READER lacks the `vec0` module; it establishes nothing about the FILE.
    Gating on it tied artifact validity to the verifying environment, which is
    how MIS-153 came to refuse the dump on the machine it was built for while
    the copy already written passed `integrity_check` and held every row.

    `integrity_check` gates the corruption classes that matter and needs no
    loadable extension: it raises `DatabaseError: file is not a database` on
    garbage and reports `database disk image is malformed` on a truncated
    store, both verified against this code path rather than assumed.

    The counter-argument, recorded because it is true: neither check
    establishes vec0 SEMANTIC validity. `COUNT(*)` on a `vec0` table counts
    rows in its rowid shadow table, not vectors, so the row count never bought
    the property that gating on it implied.

    The table list is read from `sqlite_master` rather than enumerated in this
    file. An enumeration would need editing whenever a store gains a table, and
    a backup whose row counts silently stop covering a new table reports a
    completeness it does not have. `sqlite_%` names are SQLite's own internal
    tables. sqlite-vec's shadow tables are NOT filtered out: filtering them
    would mean hardcoding one extension's internal naming, the same
    store-specific knowledge `load_sqlite_vec` exists to avoid.

    Raises:
        BackupError: When the copy cannot be opened, or when it fails
            `integrity_check`. Both mean the file on disk is unsound.
    """
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise _unsound_copy(db_path, str(exc)) from exc
    try:
        try:
            row = conn.execute("PRAGMA integrity_check").fetchone()
        except sqlite3.Error as exc:
            raise _unsound_copy(db_path, str(exc)) from exc
        verdict = "no result" if row is None else str(row[0])
        if verdict != "ok":
            raise _unsound_copy(db_path, f"PRAGMA integrity_check reported {verdict!r}")

        load_sqlite_vec(conn)
        try:
            names = [
                str(row[0])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )
            ]
        except sqlite3.Error as exc:
            raise _unsound_copy(db_path, str(exc)) from exc

        counts: dict[str, int] = {}
        uncounted: dict[str, str] = {}
        for name in names:
            quoted = '"' + name.replace('"', '""') + '"'
            try:
                counts[name] = int(conn.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0])
            except sqlite3.Error as exc:
                # Degrade, never fail. The gate above has already established
                # the file is sound, so refusing here would throw away a good
                # backup over a missing module in the verifying process.
                uncounted[name] = str(exc)
        return StoreReadback(row_counts=counts, uncounted=uncounted)
    finally:
        conn.close()


def _unsound_copy(db_path: Path, detail: str) -> BackupError:
    """The one readback failure that is about the FILE rather than the reader."""
    return BackupError(
        f"captured store {db_path} failed readback: {detail}. This is the "
        "integrity gate, not a row count: SQLite could not open the copy or "
        "would not certify it, so the artifact is incomplete and must not be "
        "relied on. A table this process merely lacks the module to read is "
        "reported as uncounted instead and never reaches here."
    )


def capture_stores(state_root: Path, artifact_dir: Path) -> list[StoreCapture]:
    """Capture each named store from `state_root` into `artifact_dir/stores/`.

    Args:
        state_root: The live state directory, normally `<repo>/data`.
        artifact_dir: The artifact being built.

    Returns:
        One `StoreCapture` per name in `LIVE_STORE_FILENAMES`, in that order,
        including the absent ones.

    Raises:
        BackupError: When a store exists but cannot be copied, or when its copy
            fails the `integrity_check` gate. A table this process cannot read
            for want of a loadable module is recorded as uncounted instead.
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
        readback = read_back_store(destination)
        # Strip after the readback: `read_back_store` reopened the copy and
        # recreated the sidecars.
        _strip_sqlite_sidecars(destination)
        captures.append(
            StoreCapture(
                filename=filename,
                present=True,
                destination=destination,
                row_counts=readback.row_counts,
                uncounted_tables=tuple(sorted(readback.uncounted)),
                missing_modules=readback.missing_modules,
            )
        )
    return captures
