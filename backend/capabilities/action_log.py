"""The local action log: one row per capability invocation, for transparency.

Every call the host runs, proposes or refuses is written here, so what MIST did to the world can
be read back later. The log is a record, not a dependency: a write that fails is
logged at WARNING and dropped, so an unwritable database never fails an invocation, a turn or
the factory that builds the host.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime

from backend.errors import MistError

from .contracts import CallSource, OutcomeStatus
from .db import CapabilitiesDb
from .power import PowerLevel

logger = logging.getLogger(__name__)

MAX_ROWS = 10000
"""Rows kept; each write prunes everything older than the newest `MAX_ROWS`."""

_COMPONENT = "action_log"
_STEPS = (
    "CREATE TABLE action_log ("
    " id INTEGER PRIMARY KEY AUTOINCREMENT,"
    " at TEXT NOT NULL,"
    " session_id TEXT NOT NULL,"
    " turn_id TEXT NOT NULL,"
    " tool_call_id TEXT NOT NULL,"
    " tool TEXT NOT NULL,"
    " capability TEXT NOT NULL,"
    " source TEXT NOT NULL,"
    " status TEXT NOT NULL,"
    " args_summary TEXT NOT NULL,"
    " summary TEXT NOT NULL,"
    " reason TEXT,"
    " proposal_id TEXT,"
    " duration_ms INTEGER NOT NULL)",
    # Step 2 (T24): the power level read at the call, and the time from the handler's receipt
    # of the user message to the call's outcome. Rows written before it read null in both.
    "ALTER TABLE action_log ADD COLUMN power_level TEXT;"
    " ALTER TABLE action_log ADD COLUMN turn_ms INTEGER",
)
_INSERT = (
    "INSERT INTO action_log (at, session_id, turn_id, tool_call_id, tool, capability, source,"
    " status, args_summary, summary, reason, proposal_id, duration_ms, power_level, turn_ms)"
    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)
# Deletes every row older than the newest `limit` rows; a no-op while fewer exist.
_PRUNE = (
    "DELETE FROM action_log WHERE id <= ("
    "SELECT id FROM action_log ORDER BY id DESC LIMIT 1 OFFSET ?)"
)


@dataclass(frozen=True, slots=True)
class ActionLogEntry:
    """One invocation as the log stores it; the host redacts its text fields before building it.

    Attributes:
        at: Aware UTC time of the call, stored as an ISO 8601 string.
        session_id: The session the call came from.
        turn_id: The turn that produced it.
        tool_call_id: The id the call's tool-call events carry.
        tool: The tool name.
        capability: The capability that owns the tool.
        source: How the call was chosen: fast path, model or approved proposal.
        status: The outcome status.
        args_summary: The tool's summary of its arguments.
        summary: The outcome summary.
        reason: The outcome reason, if any.
        proposal_id: Set for a call that waits for confirmation.
        duration_ms: Time spent in the capability; 0 for a call that never reached it.
        power_level: The level read at the call; None from a writer that does not know it.
        turn_ms: Milliseconds from the handler's receipt of the user message to the call's
            outcome; None from a writer that does not know it.
    """

    at: datetime
    session_id: str
    turn_id: str
    tool_call_id: str
    tool: str
    capability: str
    source: CallSource
    status: OutcomeStatus
    args_summary: str
    summary: str
    reason: str | None
    proposal_id: str | None
    duration_ms: int
    power_level: PowerLevel | None = None
    turn_ms: int | None = None


class ActionLog:
    """Writes `ActionLogEntry` rows to the `action_log` table of the capabilities database.

    The constructor touches no file. The table is created by the first write, through
    `CapabilitiesDb.migrate` under component `action_log`, inside the same logged-failure path
    as the write itself; a failed creation is retried by the next write.
    """

    def __init__(self, db: CapabilitiesDb, *, max_rows: int = MAX_ROWS) -> None:
        """Record the database and the row limit; nothing is opened here.

        Raises:
            ValueError: `max_rows` is less than 1.
        """
        if max_rows < 1:
            raise ValueError(f"max_rows={max_rows} must be at least 1")
        self._db = db
        self._max_rows = max_rows
        self._migrated = False

    async def record(self, entry: ActionLogEntry) -> None:
        """Write one row and prune to the newest `max_rows`; never raises for a store failure.

        A `MistError` (the database cannot be opened or migrated) or a `sqlite3.Error` (a
        statement fails) is logged at WARNING and the entry is dropped.
        """
        try:
            if not self._migrated:
                # `migrate` is synchronous and holds the database lock, so it runs in a worker
                # thread to keep the event loop free, as `run` does.
                await asyncio.to_thread(self._db.migrate, _COMPONENT, _STEPS)
                self._migrated = True
            await self._db.run(lambda conn: self._write(conn, entry))
        except (MistError, sqlite3.Error) as exc:
            logger.warning(
                "Capability action log write failed for %s (%s); the entry is dropped: %s",
                entry.tool,
                entry.tool_call_id,
                exc,
            )

    def _write(self, conn: sqlite3.Connection, entry: ActionLogEntry) -> None:
        conn.execute("BEGIN")
        try:
            conn.execute(
                _INSERT,
                (
                    entry.at.isoformat(),
                    entry.session_id,
                    entry.turn_id,
                    entry.tool_call_id,
                    entry.tool,
                    entry.capability,
                    entry.source,
                    entry.status,
                    entry.args_summary,
                    entry.summary,
                    entry.reason,
                    entry.proposal_id,
                    entry.duration_ms,
                    entry.power_level,
                    entry.turn_ms,
                ),
            )
            conn.execute(_PRUNE, (self._max_rows,))
            conn.execute("COMMIT")
        except sqlite3.Error:
            conn.execute("ROLLBACK")
            raise
