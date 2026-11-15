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
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta, tzinfo

from backend.errors import CapabilityStoreError, MistError

from .contracts import CallSource, OutcomeStatus
from .db import CapabilitiesDb
from .power import PowerLevel

logger = logging.getLogger(__name__)

MAX_ROWS = 10000
"""Rows kept; each write prunes everything older than the newest `MAX_ROWS`."""

USAGE_MIN_DAYS = 1
USAGE_MAX_DAYS = 90
USAGE_SOURCES: tuple[CallSource, ...] = ("fastpath", "model", "proposal")
"""The `by_source` keys, each present every day; a stored value outside them is not counted."""
USAGE_LEVELS: tuple[PowerLevel, ...] = ("full", "low", "off")
"""The `by_level` keys, each present every day; a row with no level is not counted there."""

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
# The usage read selects no text a user wrote or a tool summarised: no arguments, summaries or
# reasons, so none of them can reach the usage view.
_SELECT_USAGE = (
    "SELECT at, capability, source, status, power_level, turn_ms FROM action_log WHERE at >= ?"
)
_SELECT_OLDEST = "SELECT at FROM action_log ORDER BY id LIMIT 1"
# Rows are pre-filtered on the text of `at`, the time in the offset it was written with (UTC from
# the host). Offsets reach +-14 h, so a stored date is at most one day from the local date; two
# days of margin never excludes a row that buckets into the window, as in
# `backend/event_store/usage.py`. The exact bucketing is done on the parsed instant.
_PREFILTER_MARGIN_DAYS = 2


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


@dataclass(frozen=True, slots=True)
class DayUsage:
    """The invocations logged on one local date, counts and latency only.

    Attributes:
        date: The local date.
        invocations: Every row of the day, whatever its source, level or status.
        by_capability: Rows per capability name, for the names that occur.
        by_source: Rows per `USAGE_SOURCES` key, every key present.
        by_level: Rows per `USAGE_LEVELS` key, every key present; rows written before the power
            port hold no level and are counted in no key.
        by_status: Rows per outcome status, for the statuses that occur.
        turn_ms_p50: The nearest-rank median of the day's `turn_ms`; None when no row has one.
        turn_ms_p90: The nearest-rank 90th percentile, likewise.
    """

    date: date
    invocations: int
    by_capability: dict[str, int]
    by_source: dict[str, int]
    by_level: dict[str, int]
    by_status: dict[str, int]
    turn_ms_p50: int | None
    turn_ms_p90: int | None


@dataclass(frozen=True, slots=True)
class ActionLogUsage:
    """What `ActionLog.usage` read: the days asked for, newest first, and the oldest kept row.

    Attributes:
        since: The time of the oldest row the log still holds, inside the window or not, so a
            reader knows how far back the capped log reaches; None when the log is empty.
        days: One entry per local date, newest first, empty days included.
    """

    since: datetime | None
    days: tuple[DayUsage, ...]


class ActionLog:
    """Writes `ActionLogEntry` rows to the `action_log` table, and counts them per day (`usage`).

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
            await self._migrate()
            await self._db.run(lambda conn: self._write(conn, entry))
        except (MistError, sqlite3.Error) as exc:
            logger.warning(
                "Capability action log write failed for %s (%s); the entry is dropped: %s",
                entry.tool,
                entry.tool_call_id,
                exc,
            )

    async def usage(self, days: int, tz: tzinfo, *, now: datetime | None = None) -> ActionLogUsage:
        """Count the logged invocations per local date in `tz`, over `days` days ending today.

        Unlike `record`, a failure raises: a usage reader asked for numbers, and an empty
        answer would read as no use at all. The table is created first when no write has yet,
        so a fresh database answers empty days rather than failing.

        Args:
            days: Number of local days to report, from `USAGE_MIN_DAYS` to `USAGE_MAX_DAYS`.
            tz: The zone whose dates the rows are counted on.
            now: The current instant (aware); None means the wall clock.

        Raises:
            ValueError: `days` is out of range.
            CapabilityStoreError: The database cannot be opened, migrated or read.
        """
        if not USAGE_MIN_DAYS <= days <= USAGE_MAX_DAYS:
            raise ValueError(
                f"days must be between {USAGE_MIN_DAYS} and {USAGE_MAX_DAYS}, got {days}"
            )
        current = now if now is not None else datetime.now(UTC)
        today = current.astimezone(tz).date()
        dates = tuple(today - timedelta(days=offset) for offset in range(days))
        floor = (dates[-1] - timedelta(days=_PREFILTER_MARGIN_DAYS)).isoformat()
        await self._migrate()
        try:
            oldest, rows = await self._db.run(lambda conn: self._read_usage(conn, floor))
        except sqlite3.Error as exc:
            raise CapabilityStoreError(f"capability action log usage read failed: {exc}") from exc

        per_day: dict[date, list[tuple[object, ...]]] = {day: [] for day in dates}
        for at, *fields in rows:
            day = _local_date(at, tz)
            if day in per_day:
                per_day[day].append(tuple(fields))
        return ActionLogUsage(
            since=_parse_time(oldest) if oldest is not None else None,
            days=tuple(_day_usage(day, per_day[day]) for day in dates),
        )

    async def _migrate(self) -> None:
        """Create or upgrade the table once per instance.

        Raises:
            CapabilityStoreError: The database cannot be opened or migrated.
        """
        if not self._migrated:
            # `migrate` is synchronous and holds the database lock, so it runs in a worker
            # thread to keep the event loop free, as `run` does.
            await asyncio.to_thread(self._db.migrate, _COMPONENT, _STEPS)
            self._migrated = True

    @staticmethod
    def _read_usage(
        conn: sqlite3.Connection, floor: str
    ) -> tuple[object | None, list[tuple[object, ...]]]:
        oldest = conn.execute(_SELECT_OLDEST).fetchone()
        rows = conn.execute(_SELECT_USAGE, (floor,)).fetchall()
        return (oldest[0] if oldest is not None else None), rows

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


def _parse_time(raw: object) -> datetime | None:
    """Read a stored `at`; None for a value that is not an ISO time.

    A value with no offset is read as UTC, the zone every writer uses, rather than as the host's
    local time, which `astimezone` would assume.
    """
    if not isinstance(raw, str):
        return None
    try:
        instant = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return instant if instant.tzinfo is not None else instant.replace(tzinfo=UTC)


def _local_date(raw: object, tz: tzinfo) -> date | None:
    instant = _parse_time(raw)
    return instant.astimezone(tz).date() if instant is not None else None


def _nearest_rank(ordered: list[int], percent: int) -> int | None:
    """The nearest-rank percentile of sorted values: always a value that was measured."""
    if not ordered:
        return None
    rank = -(-percent * len(ordered) // 100)
    return ordered[max(rank, 1) - 1]


def _day_usage(day: date, rows: list[tuple[object, ...]]) -> DayUsage:
    """Count one day's rows of (capability, source, status, power_level, turn_ms)."""
    by_capability: Counter[str] = Counter()
    by_status: Counter[str] = Counter()
    by_source = dict.fromkeys(USAGE_SOURCES, 0)
    by_level = dict.fromkeys(USAGE_LEVELS, 0)
    turn_ms: list[int] = []
    for capability, source, status, level, elapsed in rows:
        by_capability[str(capability)] += 1
        by_status[str(status)] += 1
        if source in by_source:
            by_source[source] += 1  # type: ignore[index]
        if level in by_level:
            by_level[level] += 1  # type: ignore[index]
        if isinstance(elapsed, int):
            turn_ms.append(elapsed)
    turn_ms.sort()
    return DayUsage(
        date=day,
        invocations=len(rows),
        by_capability=dict(by_capability),
        by_source=dict(by_source),
        by_level=dict(by_level),
        by_status=dict(by_status),
        turn_ms_p50=_nearest_rank(turn_ms, 50),
        turn_ms_p90=_nearest_rank(turn_ms, 90),
    )
