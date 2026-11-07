"""Real-origin conversation turns per local day, counts only.

ADR-042 measures real use from real-origin turns and reports it; nothing waits
on the number. This module counts those turns per local day from the event
store and returns nothing but a zone name, dates and integers.
The query selects one column, `conversation_turn_events.timestamp`; no session
id, utterance or response is ever read into this process.

Origin rule
-----------
A turn is real when `COALESCE(s.origin, 'real') = 'real'` over
`conversation_turn_events` LEFT JOIN `conversation_sessions`, the rule
`EventStore.get_all_turns_for_reextraction` applies
(`backend/event_store/store.py:409-419`). The LEFT JOIN is deliberate: a turn
whose session row is missing is counted as real rather than silently dropped,
matching the store's default direction for `origin`.

Read-only access
----------------
The store is opened through a percent-encoded `file:...?mode=ro` URI, as
`backend/event_store/uptime.py:390-406` does, never through `EventStore`.
`EventStore._get_connection` is a read-write `sqlite3.connect` that sets
`PRAGMA journal_mode=WAL` and creates the file when absent; a counts endpoint
must not create, migrate or write the store it reports on.

Local dates
-----------
Turns are bucketed by their date in the zone named by `MIST_USAGE_TIMEZONE`
(default UTC), resolved through `pytz`. `pytz` rather than `zoneinfo`: the
backend image ships no IANA database for `zoneinfo` (no `tzdata` package and
no system zone files), while `pytz` carries its own and is already installed
as a dependency of `neo4j`. An unknown zone name falls back to UTC, logs a
warning and is reported as `UTC`, so the caller always knows which zone the
dates are in. A stored timestamp with no offset is read as UTC, the zone every
writer uses (`ConversationTurnEvent.to_dict` normalises aware values to UTC).
"""

from __future__ import annotations

import logging
import os
import sqlite3
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta, tzinfo
from pathlib import Path
from urllib.parse import quote

import pytz

from backend.errors import MistError

from .store import _DEFAULT_DB_PATH

logger = logging.getLogger(__name__)

TIMEZONE_ENV = "MIST_USAGE_TIMEZONE"
DEFAULT_TIMEZONE = "UTC"
ORIGIN = "real"

DEFAULT_DAYS = 7
MIN_DAYS = 1
MAX_DAYS = 90

# How long a read waits on a writer's lock before reporting the store
# unavailable. sqlite3's own default is 5 s; a counts read has no reason to
# hold an HTTP request that long.
BUSY_TIMEOUT_SECONDS = 1.0

# Rows are pre-filtered in SQL on the date prefix of the stored text, which is
# the date in whatever offset the row was written with. Offsets reach +-14 h,
# so a stored date can differ from the local date by at most one day; two days
# of margin keeps the filter from ever excluding a row that buckets into the
# window. The exact bucketing is done in Python on the parsed instant.
_PREFILTER_MARGIN_DAYS = 2

_REAL_TURN_TIMESTAMPS_SQL = (
    "SELECT e.timestamp FROM conversation_turn_events e "
    "LEFT JOIN conversation_sessions s ON e.session_id = s.session_id "
    "WHERE COALESCE(s.origin, 'real') = 'real'"
)


class UsageStoreUnavailableError(MistError):
    """The event store is missing, locked or not an event store, so no count exists."""


@dataclass(frozen=True, slots=True)
class DayCount:
    """Real-origin turns on one local date."""

    date: date
    real_turns: int


@dataclass(frozen=True, slots=True)
class TurnsPerDay:
    """Real use as reported: dense per-day counts, oldest first and ending today.

    Only the counts and the zone they are dated in; whoever reads them decides
    what they mean.
    """

    timezone: str
    days: tuple[DayCount, ...]

    def to_payload(self) -> dict:
        """Render the `GET /usage/turns-per-day` body: dates and integers only."""
        return {
            "timezone": self.timezone,
            "origin": ORIGIN,
            "days": [{"date": d.date.isoformat(), "real_turns": d.real_turns} for d in self.days],
        }


def resolve_store_path(configured: str | None) -> str:
    """Return the store path `EventStore` would use for the same configuration.

    `EventStoreConfig.db_path` (from `EVENT_STORE_DB_PATH`) is None when
    unset, and `EventStore.__init__` then falls back to `_DEFAULT_DB_PATH`;
    counting a different file from the one the live handler writes would
    report real use from the wrong data.
    """
    return configured if configured else str(_DEFAULT_DB_PATH)


def resolve_timezone(name: str | None) -> tuple[tzinfo, str]:
    """Resolve a zone name to a `pytz` zone and the name actually used.

    Unset or blank means the default, UTC, without a warning. An unknown name
    also yields UTC, but with a warning naming the value, because a typo in
    `MIST_USAGE_TIMEZONE` shifts every late-evening turn onto the wrong date.

    Returns:
        The zone, and its canonical name (`UTC` on fallback).
    """
    if name is None or not name.strip():
        return pytz.utc, DEFAULT_TIMEZONE
    try:
        zone = pytz.timezone(name.strip())
    except pytz.exceptions.UnknownTimeZoneError:
        logger.warning("%s=%r is not a known time zone; counting in UTC", TIMEZONE_ENV, name)
        return pytz.utc, DEFAULT_TIMEZONE
    return zone, zone.zone


def local_date(raw: str, zone: tzinfo) -> date | None:
    """Return the date of a stored timestamp in `zone`, or None if it does not parse.

    A value with no offset is read as UTC rather than as the host's local time,
    which is what `astimezone` would assume for a naive value.
    """
    try:
        instant = datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        return None
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=UTC)
    return instant.astimezone(zone).date()


def _readonly_connect(db_path: str, busy_timeout_seconds: float) -> sqlite3.Connection:
    """Open `db_path` read-only through a percent-encoded `file:...?mode=ro` URI.

    Percent-encoded, not interpolated: an un-encoded `#` or `?` in the path is
    URI syntax and would truncate the filename. `mode=ro` also means a missing
    file is an error instead of a new empty database.
    """
    uri = "file:" + quote(str(Path(db_path).resolve())) + "?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=busy_timeout_seconds)


def read_real_turn_timestamps(
    db_path: str,
    *,
    since: date | None = None,
    busy_timeout_seconds: float = BUSY_TIMEOUT_SECONDS,
) -> list[str]:
    """Read the stored timestamp of every real-origin turn, and nothing else.

    Args:
        db_path: Path to the event store's SQLite file.
        since: When given, rows whose stored text sorts before this date are
            skipped in SQL. The caller supplies a margin; see
            `_PREFILTER_MARGIN_DAYS`.
        busy_timeout_seconds: How long to wait on a writer's lock.

    Raises:
        UsageStoreUnavailableError: the file is missing, locked, or has no
            turn table.
    """
    sql = _REAL_TURN_TIMESTAMPS_SQL
    params: tuple[str, ...] = ()
    if since is not None:
        sql += " AND e.timestamp >= ?"
        params = (since.isoformat(),)
    try:
        conn = _readonly_connect(db_path, busy_timeout_seconds)
    except sqlite3.Error as exc:
        raise UsageStoreUnavailableError(f"event store could not be opened: {exc}") from exc
    try:
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.Error as exc:
        raise UsageStoreUnavailableError(f"event store could not be read: {exc}") from exc
    finally:
        conn.close()
    return [row[0] for row in rows]


def count_turns_per_day(
    timestamps: Iterable[str], zone: tzinfo, *, today: date, days: int
) -> tuple[DayCount, ...]:
    """Bucket timestamps into the `days` local dates ending `today`, densely.

    Dates step by calendar day, so a DST change shortens or lengthens a day's
    span without adding or losing a bucket. Timestamps outside the window are
    ignored; unparsable ones are skipped and their number logged, since the
    contract has no field to carry them.
    """
    first = today - timedelta(days=days - 1)
    counts: Counter[date] = Counter()
    unparsable = 0
    for raw in timestamps:
        day = local_date(raw, zone)
        if day is None:
            unparsable += 1
        elif first <= day <= today:
            counts[day] += 1
    if unparsable:
        logger.warning("Skipped %d turn(s) with an unparsable timestamp", unparsable)
    return tuple(
        DayCount(first + timedelta(days=offset), counts[first + timedelta(days=offset)])
        for offset in range(days)
    )


def build_turns_per_day(
    db_path: str,
    *,
    days: int = DEFAULT_DAYS,
    timezone_name: str | None = None,
    now: datetime | None = None,
) -> TurnsPerDay:
    """Count real-origin turns per local day over `days` days ending today.

    Only the days asked for are read and bucketed: the SQL pre-filter starts
    `_PREFILTER_MARGIN_DAYS` before the first of them, and nothing older is
    counted.

    Args:
        db_path: Path to the event store's SQLite file.
        days: Number of local days to report, from 1 to 90.
        timezone_name: IANA zone name; None reads `MIST_USAGE_TIMEZONE`.
        now: The current instant (aware); None means the wall clock.

    Raises:
        ValueError: `days` is outside 1-90.
        UsageStoreUnavailableError: the store cannot be read.
    """
    if not MIN_DAYS <= days <= MAX_DAYS:
        raise ValueError(f"days must be between {MIN_DAYS} and {MAX_DAYS}, got {days}")
    if timezone_name is None:
        timezone_name = os.getenv(TIMEZONE_ENV)
    zone, zone_name = resolve_timezone(timezone_name)
    current = now if now is not None else datetime.now(UTC)
    today = current.astimezone(zone).date()

    since = today - timedelta(days=days - 1 + _PREFILTER_MARGIN_DAYS)
    timestamps = read_real_turn_timestamps(db_path, since=since)
    counts = count_turns_per_day(timestamps, zone, today=today, days=days)

    return TurnsPerDay(timezone=zone_name, days=counts)
