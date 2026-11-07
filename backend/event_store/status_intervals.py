"""The uptime ledger: MIST's own `/health` status over time, per local day.

The server samples its derived `/health` status every `MIST_UPTIME_RECORD_S`
seconds and records it here as intervals in `instance_status_intervals`. The
lead then reads, per day, how long MIST was healthy, degraded, yielded or
unhealthy, and how much of the day nothing was recorded (MIST not running, or
too wedged to tick), without reading a conversation: the table holds a status
word, a flag and two timestamps, and the reader returns only a zone name, a
status word, dates and integers.

Writer
------
`record_status_tick` opens its own connection, decides in one
`BEGIN IMMEDIATE` transaction, and closes the connection. It never uses
`EventStore`'s shared connection: that one is driven from the event loop's
callers without a lock of its own, and a tick runs in a worker thread through
`asyncio.to_thread`. `BEGIN IMMEDIATE` takes the write lock before the newest
row is read, so a competing writer cannot change that row between the read
and the write; the 5 s connect timeout is how long a tick waits for the lock.

Reader
------
`build_uptime_per_day` opens the store read-only through the same
percent-encoded `file:...?mode=ro` URI as `usage.py`, so a missing store is
reported, never created. Rows are read in `id` order and each row's start is
clipped to the latest end already counted, so two rows that overlap after the
clock stepped back count their overlap once. An interval ends at its
`last_seen_at`: the time since the newest tick is unrecorded, never up. Days
are local days in `MIST_TIMEZONE`, resolved through `pytz` as
`usage.py` does, so a DST change gives a day of 82800 s or 90000 s.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta, tzinfo

from backend.errors import MistError

from .usage import (
    BUSY_TIMEOUT_SECONDS,
    DEFAULT_DAYS,
    MAX_DAYS,
    MIN_DAYS,
    TIMEZONE_ENV,
    _readonly_connect,
    resolve_timezone,
)

logger = logging.getLogger(__name__)

STATUSES: tuple[str, ...] = ("healthy", "degraded", "yielded", "unhealthy")

# How long a tick waits on another writer's lock before giving up with
# `sqlite3.OperationalError`, which the tick loop logs and skips.
WRITE_TIMEOUT_SECONDS = 5.0

# Rows are pre-filtered in SQL on the text of `last_seen_at`, which the writer
# stores in UTC; a local day can start up to 14 h before its UTC date, so two
# days of margin keeps the filter from excluding a row that reaches the
# window. The exact clipping is done in Python on the parsed instants.
_PREFILTER_MARGIN_DAYS = 2

_NEWEST_ROW_SQL = (
    "SELECT id, status, degraded_beneath, last_seen_at FROM instance_status_intervals "
    "ORDER BY id DESC LIMIT 1"
)
_INSERT_SQL = (
    "INSERT INTO instance_status_intervals "
    "(status, degraded_beneath, started_at, last_seen_at) VALUES (?, ?, ?, ?)"
)
_EXTEND_SQL = "UPDATE instance_status_intervals SET last_seen_at = ? WHERE id = ?"
# Every ISO-8601 text sorts at or after "", so an empty bound reads every row.
_READ_SQL = (
    "SELECT status, degraded_beneath, started_at, last_seen_at "
    "FROM instance_status_intervals WHERE last_seen_at >= ? ORDER BY id"
)

_SECOND = timedelta(seconds=1)


class UptimeStoreUnavailableError(MistError):
    """The event store is missing, locked or has no ledger, so no uptime exists."""


@dataclass(frozen=True, slots=True)
class StatusInterval:
    """One stored row: a status held from `started_at` until `last_seen_at`."""

    status: str
    degraded_beneath: bool
    started_at: datetime
    last_seen_at: datetime


@dataclass(frozen=True, slots=True)
class UptimeDay:
    """Seconds per status on one local date.

    The first five values sum to the day's real length, or for today to the
    seconds elapsed. `yielded_degraded_s` is the part of `yielded_s` whose
    rows had `degraded_beneath`, and is not one of the five.
    """

    date: date
    healthy_s: int
    degraded_s: int
    yielded_s: int
    unhealthy_s: int
    unrecorded_s: int
    yielded_degraded_s: int


@dataclass(frozen=True, slots=True)
class UptimePerDay:
    """Dense per-day uptime, oldest first and ending today, and the live status."""

    timezone: str
    current: str
    days: tuple[UptimeDay, ...]

    def to_payload(self) -> dict:
        """Render the `GET /usage/uptime` body: a zone, a status, dates and integers."""
        return {
            "timezone": self.timezone,
            "current": self.current,
            "days": [
                {
                    "date": d.date.isoformat(),
                    "healthy_s": d.healthy_s,
                    "degraded_s": d.degraded_s,
                    "yielded_s": d.yielded_s,
                    "unhealthy_s": d.unhealthy_s,
                    "unrecorded_s": d.unrecorded_s,
                    "yielded_degraded_s": d.yielded_degraded_s,
                }
                for d in self.days
            ],
        }


def _parse_instant(raw: str) -> datetime:
    """Parse a stored timestamp; a value with no offset is read as UTC."""
    instant = datetime.fromisoformat(raw)
    return instant if instant.tzinfo is not None else instant.replace(tzinfo=UTC)


def record_status_tick(
    db_path: str,
    status: str,
    degraded_beneath: bool,
    now: datetime,
    *,
    max_gap_s: float,
    first_tick: bool,
) -> None:
    """Record one sample of the derived `/health` status.

    Rules, first match wins:

    1. a `status` outside `STATUSES`, or a naive `now`, raises `ValueError`
       before the store is opened;
    2. no row yet, or `first_tick`: insert a row starting and last seen at
       `now`, so a restart within `max_gap_s` still leaves its gap;
    3. the newest row's `last_seen_at` is after `now` (the clock stepped
       back): insert a row at `now` and leave the newest row as it is;
    4. more than `max_gap_s` since the newest row's `last_seen_at`: insert a
       row at `now`, leaving the gap unrecorded;
    5. the same `status` and `degraded_beneath`: move `last_seen_at` to `now`;
    6. otherwise, a change within the gap: insert a row starting at the newest
       row's `last_seen_at`, so the two rows meet, and last seen at `now`.

    Args:
        db_path: Path to the event store's SQLite file.
        status: The derived `/health` status.
        degraded_beneath: `degraded_beneath(snapshot)` for the same snapshot.
        now: The sample's instant, timezone-aware.
        max_gap_s: The longest silence that still continues a row.
        first_tick: The first sample since this process started.

    Raises:
        ValueError: `status` is not one of `STATUSES`, or `now` is naive.
        sqlite3.Error: the store could not be written, including a lock held
            past `WRITE_TIMEOUT_SECONDS`.
    """
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}, got {status!r}")
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError(f"now must be timezone-aware, got {now!r}")
    instant = now.astimezone(UTC)
    stamp = instant.isoformat()
    beneath = 1 if degraded_beneath else 0

    conn = sqlite3.connect(db_path, timeout=WRITE_TIMEOUT_SECONDS, isolation_level=None)
    try:
        # `with conn` commits on success and rolls back on any error; the
        # explicit BEGIN IMMEDIATE makes the transaction take the write lock
        # before the newest row is read.
        with conn:
            conn.execute("BEGIN IMMEDIATE")
            newest = conn.execute(_NEWEST_ROW_SQL).fetchone()
            if newest is None or first_tick:
                conn.execute(_INSERT_SQL, (status, beneath, stamp, stamp))
                return
            row_id, row_status, row_beneath, raw_last_seen = newest
            last_seen = _parse_instant(raw_last_seen)
            stepped_back = last_seen > instant  # rule 3
            past_gap = (instant - last_seen).total_seconds() > max_gap_s  # rule 4
            if stepped_back or past_gap:
                conn.execute(_INSERT_SQL, (status, beneath, stamp, stamp))
            elif row_status == status and row_beneath == beneath:
                conn.execute(_EXTEND_SQL, (stamp, row_id))
            else:
                conn.execute(_INSERT_SQL, (status, beneath, raw_last_seen, stamp))
    finally:
        conn.close()


def degraded_beneath(snapshot: Mapping) -> bool:
    """Whether a yielded snapshot also has a check other than a yielded one not up.

    A yield outranks `degraded` in `health.derive_status`, so a fault that
    starts while the GPU is yielded would otherwise be invisible in the
    ledger. "Up" means what it means to `derive_status`: `status` is `up` and
    the result is not stale, so a null (not measured) or stale check counts
    as not up, exactly as it would make the system `degraded` without the
    yield.

    Returns:
        False for every snapshot whose status is not `yielded`.
    """
    if snapshot.get("status") != "yielded":
        return False
    for check in (snapshot.get("checks") or {}).values():
        if check.get("status") == "yielded":
            continue
        if check.get("status") != "up" or check.get("stale"):
            return True
    return False


def read_status_intervals(
    db_path: str,
    *,
    since: date | None = None,
    busy_timeout_seconds: float = BUSY_TIMEOUT_SECONDS,
) -> list[StatusInterval]:
    """Read the ledger's rows in `id` order, through a read-only connection.

    Args:
        db_path: Path to the event store's SQLite file.
        since: When given, rows whose `last_seen_at` text sorts before this
            date are skipped in SQL. The caller supplies a margin; see
            `_PREFILTER_MARGIN_DAYS`.
        busy_timeout_seconds: How long to wait on a writer's lock.

    Raises:
        UptimeStoreUnavailableError: the file is missing, locked, or has no
            ledger table.
    """
    bound = since.isoformat() if since is not None else ""
    try:
        conn = _readonly_connect(db_path, busy_timeout_seconds)
    except sqlite3.Error as exc:
        raise UptimeStoreUnavailableError(f"event store could not be opened: {exc}") from exc
    try:
        rows = conn.execute(_READ_SQL, (bound,)).fetchall()
    except sqlite3.Error as exc:
        raise UptimeStoreUnavailableError(f"event store could not be read: {exc}") from exc
    finally:
        conn.close()

    intervals: list[StatusInterval] = []
    unparsable = 0
    for status, beneath, raw_start, raw_end in rows:
        try:
            start, end = _parse_instant(raw_start), _parse_instant(raw_end)
        except (TypeError, ValueError):
            unparsable += 1
            continue
        intervals.append(StatusInterval(status, bool(beneath), start, end))
    if unparsable:
        logger.warning("Skipped %d uptime row(s) with an unparsable timestamp", unparsable)
    return intervals


def _local_midnight(day: date, zone: tzinfo) -> datetime:
    """The UTC instant at which `day` starts in `zone`.

    `pytz` zones need `localize`, not `replace(tzinfo=...)`, which would
    attach the zone's first historical offset (LMT) instead of the one in
    force that day.
    """
    naive = datetime.combine(day, time())
    localize = getattr(zone, "localize", None)
    if localize is None:
        return naive.replace(tzinfo=zone).astimezone(UTC)
    local = localize(naive)
    normalize = getattr(zone, "normalize", None)
    if normalize is not None:
        local = normalize(local)
    return local.astimezone(UTC)


def _counted_spans(
    intervals: Iterable[StatusInterval],
) -> list[tuple[StatusInterval, datetime, datetime]]:
    """Clip each row's start to the latest end already counted, in `id` order."""
    spans: list[tuple[StatusInterval, datetime, datetime]] = []
    counted_end: datetime | None = None
    for interval in intervals:
        start = interval.started_at
        if counted_end is not None and start < counted_end:
            start = counted_end
        if interval.last_seen_at > start:
            spans.append((interval, start, interval.last_seen_at))
        if counted_end is None or interval.last_seen_at > counted_end:
            counted_end = interval.last_seen_at
    return spans


def uptime_per_day(
    intervals: Sequence[StatusInterval], zone: tzinfo, *, now: datetime, days: int
) -> tuple[UptimeDay, ...]:
    """Split the intervals over the `days` local dates ending today, densely.

    Each status total is floored to whole seconds and `unrecorded_s` takes
    the rest of the day, so the five values sum to the day exactly and a
    fraction of a second is never reported as up.
    """
    current = now.astimezone(UTC)
    today = current.astimezone(zone).date()
    first = today - timedelta(days=days - 1)
    spans = _counted_spans(intervals)

    result: list[UptimeDay] = []
    for offset in range(days):
        day = first + timedelta(days=offset)
        day_start = _local_midnight(day, zone)
        day_end = current if day == today else _local_midnight(day + timedelta(days=1), zone)
        totals = {name: timedelta(0) for name in STATUSES}
        yielded_degraded = timedelta(0)
        for interval, start, end in spans:
            overlap = min(end, day_end) - max(start, day_start)
            if overlap <= timedelta(0):
                continue
            totals[interval.status] += overlap
            if interval.status == "yielded" and interval.degraded_beneath:
                yielded_degraded += overlap
        seconds = {name: totals[name] // _SECOND for name in STATUSES}
        length = max(timedelta(0), day_end - day_start) // _SECOND
        result.append(
            UptimeDay(
                date=day,
                healthy_s=seconds["healthy"],
                degraded_s=seconds["degraded"],
                yielded_s=seconds["yielded"],
                unhealthy_s=seconds["unhealthy"],
                unrecorded_s=max(0, length - sum(seconds.values())),
                yielded_degraded_s=yielded_degraded // _SECOND,
            )
        )
    return tuple(result)


def build_uptime_per_day(
    db_path: str,
    *,
    current: str,
    days: int = DEFAULT_DAYS,
    timezone_name: str | None = None,
    now: datetime | None = None,
) -> UptimePerDay:
    """Seconds per status per local day over `days` days ending today.

    Args:
        db_path: Path to the event store's SQLite file.
        current: The live `/health` status at the time of the request; it is
            reported as given, not read from the ledger.
        days: Number of local days to report, from 1 to 90.
        timezone_name: IANA zone name; None reads `MIST_TIMEZONE`.
        now: The current instant (aware); None means the wall clock.

    Raises:
        ValueError: `days` is outside 1-90.
        UptimeStoreUnavailableError: the store cannot be read.
    """
    if not MIN_DAYS <= days <= MAX_DAYS:
        raise ValueError(f"days must be between {MIN_DAYS} and {MAX_DAYS}, got {days}")
    if timezone_name is None:
        timezone_name = os.getenv(TIMEZONE_ENV)
    zone, zone_name = resolve_timezone(timezone_name)
    instant = now if now is not None else datetime.now(UTC)
    today = instant.astimezone(zone).date()

    since = today - timedelta(days=days - 1 + _PREFILTER_MARGIN_DAYS)
    intervals = read_status_intervals(db_path, since=since)
    return UptimePerDay(
        timezone=zone_name,
        current=current,
        days=uptime_per_day(intervals, zone, now=instant, days=days),
    )
