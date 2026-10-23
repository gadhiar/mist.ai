"""The SQLite database that capability stores share, each under its own component name.

One connection serves every store, opened on first use rather than in the constructor, as
`EventStore._get_connection` does. Constructing the object never touches the disk, so a missing
or unwritable path surfaces at the first `run` or `migrate`, not where the object is built.

Every call holds one lock, so two stores never interleave statements on the shared connection.
`run` executes in a worker thread, so its caller's event loop keeps running during the I/O.
`migrate` keeps a synchronous signature and runs on the calling thread: called from a coroutine,
it blocks that event loop for the migration and for any wait on the lock held by a `run` in
flight.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeVar

from backend.errors import CapabilityStoreError

logger = logging.getLogger(__name__)

T = TypeVar("T")

_MEMORY_PATH = ":memory:"
_VERSIONS_TABLE = (
    "CREATE TABLE IF NOT EXISTS schema_versions ("
    "component TEXT PRIMARY KEY, version INTEGER NOT NULL)"
)


class CapabilitiesDb:
    """One sqlite3 connection (`check_same_thread=False`), a `threading.Lock`, WAL mode."""

    def __init__(self, path: str) -> None:
        """Record the path; no file or directory is touched until the first call."""
        self._path = path
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None

    async def run(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        """Run `fn` with the connection in a worker thread, under the lock.

        Exceptions raised by `fn` propagate unchanged, so a store decides how to report its own
        statement failures.

        Raises:
            CapabilityStoreError: The database cannot be opened.
        """
        return await asyncio.to_thread(self._call, fn)

    def migrate(self, component: str, steps: Sequence[str]) -> int:
        """Apply the steps of `component` that the database has not seen, in order.

        Each component keeps its own version in `schema_versions`, equal to the number of steps
        applied, so stores evolve independently and a second call is a no-op. Each step runs in
        its own transaction with its version update, so a failed step leaves the earlier version
        intact. A step may hold several statements but must not manage transactions itself.

        It runs on the calling thread and takes the same lock as `run`, so it never interleaves
        with a call in flight; a coroutine that calls it blocks its event loop until it returns.

        Returns:
            The component's version after migrating.

        Raises:
            CapabilityStoreError: The database cannot be opened, a step fails, or the database
                records more steps than `steps` holds (it was written by newer code).
        """
        with self._lock:
            conn = self._connection()
            try:
                conn.execute(_VERSIONS_TABLE)
                row = conn.execute(
                    "SELECT version FROM schema_versions WHERE component = ?", (component,)
                ).fetchone()
            except sqlite3.Error as exc:
                raise CapabilityStoreError(
                    f"cannot read schema version of {component} in {self._path}: {exc}"
                ) from exc
            version = row[0] if row is not None else 0
            if version > len(steps):
                raise CapabilityStoreError(
                    f"{self._path} holds {component} schema version {version}, newer than the "
                    f"{len(steps)} steps this code knows"
                )
            for index in range(version, len(steps)):
                self._apply_step(conn, component, index + 1, steps[index])
            return len(steps)

    def close(self) -> None:
        """Close the connection; a later call opens a new one."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def _call(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        with self._lock:
            return fn(self._connection())

    def _connection(self) -> sqlite3.Connection:
        """Return the open connection, opening it first. The caller holds the lock."""
        if self._conn is not None:
            return self._conn
        try:
            if self._path != _MEMORY_PATH:
                Path(self._path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self._path, check_same_thread=False, isolation_level=None)
        except (OSError, sqlite3.Error) as exc:
            raise CapabilityStoreError(f"cannot open {self._path}: {exc}") from exc
        try:
            mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            conn.execute("PRAGMA foreign_keys=ON")
        except sqlite3.Error as exc:
            conn.close()
            raise CapabilityStoreError(f"cannot configure {self._path}: {exc}") from exc
        if mode != "wal" and self._path != _MEMORY_PATH:
            logger.warning(
                "Capabilities database %s runs in %s journal mode, not WAL", self._path, mode
            )
        self._conn = conn
        return conn

    def _apply_step(self, conn: sqlite3.Connection, component: str, number: int, step: str) -> None:
        # `executescript` commits any open transaction before it runs, so the script itself
        # opens the transaction and leaves it open for the version update.
        try:
            conn.executescript(f"BEGIN;\n{step};\n")
            conn.execute(
                "INSERT INTO schema_versions (component, version) VALUES (?, ?) "
                "ON CONFLICT (component) DO UPDATE SET version = excluded.version",
                (component, number),
            )
            conn.execute("COMMIT")
        except sqlite3.Error as exc:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise CapabilityStoreError(
                f"{component} migration step {number} failed in {self._path}: {exc}"
            ) from exc
