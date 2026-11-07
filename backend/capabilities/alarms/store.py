"""Durable alarms and timers in the shared capabilities database (component `alarms`).

Alarm state lives in SQLite rather than in memory because a broadcast with no client connected
is lost: ringing and missed alarms must still be there for the snapshot a client receives on
connect, and after a restart.

Times are stored as UTC ISO strings and a repeat rule as JSON (`RepeatRule.to_json`). Rows are
fetched with text as bytes and decoded one at a time, so a row that does not decode costs only
itself: one whose times, repeat rule, kind, state, label or late-by value is malformed, of the
wrong type, or text that is not valid UTF-8. `due`, `next_due` and `list` skip such a row and log
it at ERROR with its id once per process, and never raise for it. `cancel` still deletes it,
because a delete reads nothing.

Every write that changes a row also increments a store-wide `rev` in the same transaction. The
counter is stored, so it keeps increasing across restarts and a client can order snapshots by it.

The constructor touches no file; `migrate` creates the tables, so an unwritable database fails
whoever calls `migrate`, not whoever builds the store.
"""

from __future__ import annotations

import builtins
import logging
import sqlite3
import threading
import uuid
from collections.abc import Callable, Collection
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, TypeVar

from backend.capabilities.db import CapabilitiesDb
from backend.errors import CapabilityStoreError

from .timeparse import RepeatRule

logger = logging.getLogger(__name__)

T = TypeVar("T")

AlarmKind = Literal["alarm", "timer"]
AlarmState = Literal["scheduled", "ringing", "snoozed", "done", "missed"]
"""States a stored alarm can hold. A cancelled alarm holds none: `cancel` deletes its row."""

_KINDS: frozenset[str] = frozenset({"alarm", "timer"})
_STATES: frozenset[str] = frozenset({"scheduled", "ringing", "snoozed", "done", "missed"})

_COMPONENT = "alarms"
_STEPS = (
    "CREATE TABLE alarms ("
    "id TEXT PRIMARY KEY, "
    "kind TEXT NOT NULL, "
    "label TEXT, "
    "state TEXT NOT NULL, "
    "scheduled_for TEXT NOT NULL, "
    "next_fire_at TEXT NOT NULL, "
    "repeat TEXT, "
    "late_by_s REAL, "
    "fired_at TEXT);\n"
    "CREATE INDEX alarms_state ON alarms (state);\n"
    "CREATE TABLE alarm_rev (id INTEGER PRIMARY KEY CHECK (id = 1), rev INTEGER NOT NULL);\n"
    "INSERT INTO alarm_rev (id, rev) VALUES (1, 0)",
)

# Every statement is a literal, and each transition names its source states in its WHERE
# clause, so a transition from any other state changes no row.
_INSERT = (
    "INSERT INTO alarms (id, kind, label, state, scheduled_for, next_fire_at, repeat, "
    "late_by_s, fired_at) VALUES (?, ?, ?, 'scheduled', ?, ?, ?, NULL, NULL)"
)
_SELECT_ALL = (
    "SELECT id, kind, label, state, scheduled_for, next_fire_at, repeat, late_by_s, fired_at "
    "FROM alarms"
)
_SELECT_PENDING = (
    "SELECT id, kind, label, state, scheduled_for, next_fire_at, repeat, late_by_s, fired_at "
    "FROM alarms WHERE state IN ('scheduled', 'snoozed')"
)
_TO_RINGING = (
    "UPDATE alarms SET state = 'ringing', late_by_s = ?, fired_at = ? "
    "WHERE id = ? AND state IN ('scheduled', 'snoozed')"
)
_TO_SNOOZED = (
    "UPDATE alarms SET state = 'snoozed', next_fire_at = ? WHERE id = ? AND state = 'ringing'"
)
_TO_DONE = "UPDATE alarms SET state = 'done' WHERE id = ? AND state IN ('ringing', 'snoozed')"
_TO_MISSED = (
    "UPDATE alarms SET state = 'missed' "
    "WHERE id = ? AND state IN ('scheduled', 'snoozed', 'ringing')"
)
_DELETE = "DELETE FROM alarms WHERE id = ?"
_BUMP_REV = "UPDATE alarm_rev SET rev = rev + 1 WHERE id = 1"
_SELECT_REV = "SELECT rev FROM alarm_rev WHERE id = 1"

_reported_lock = threading.Lock()
_reported_ids: set[str] = set()
"""Ids of undecodable rows already logged in this process."""


@dataclass(frozen=True, slots=True)
class Alarm:
    """One alarm or timer as stored.

    `scheduled_for` is the occurrence the user set; `next_fire_at` is when it next fires, which a
    snooze moves later. `late_by_s` and `fired_at` are set when it starts ringing.
    """

    alarm_id: str
    kind: AlarmKind
    label: str | None
    state: AlarmState
    scheduled_for: datetime
    next_fire_at: datetime
    repeat: RepeatRule | None
    late_by_s: float | None
    fired_at: datetime | None


class AlarmStore:
    """Alarms and timers on a `CapabilitiesDb`, with guarded state transitions.

    Transitions apply only from the states listed on each method, and return False when the
    alarm is missing or in another state, so a stale caller cannot, for example, re-ring an
    alarm that was stopped. Database failures raise `CapabilityStoreError` naming the alarm
    store.
    """

    def __init__(self, db: CapabilitiesDb) -> None:
        """Keep the database; no file is touched until `migrate` or the first call."""
        self._db = db

    def migrate(self) -> None:
        """Create or upgrade the alarm tables, on the calling thread.

        Raises:
            CapabilityStoreError: The database cannot be opened or migrated; the message names
                the alarm store and the database.
        """
        try:
            self._db.migrate(_COMPONENT, _STEPS)
        except CapabilityStoreError as exc:
            raise CapabilityStoreError(f"alarm store cannot migrate: {exc}") from exc

    async def create(
        self,
        kind: AlarmKind,
        fire_at: datetime,
        *,
        label: str | None = None,
        repeat: RepeatRule | None = None,
    ) -> Alarm:
        """Store a new scheduled alarm that fires at `fire_at`.

        Raises:
            ValueError: `kind` is unknown or `fire_at` is naive.
            CapabilityStoreError: The write failed.
        """
        if kind not in _KINDS:
            raise ValueError(f"unknown alarm kind {kind!r}")
        alarm = Alarm(
            alarm_id=uuid.uuid4().hex,
            kind=kind,
            label=label,
            state="scheduled",
            scheduled_for=_to_utc(fire_at, "fire_at"),
            next_fire_at=_to_utc(fire_at, "fire_at"),
            repeat=repeat,
            late_by_s=None,
            fired_at=None,
        )

        params = (
            alarm.alarm_id,
            alarm.kind,
            alarm.label,
            _encode_time(alarm.scheduled_for),
            _encode_time(alarm.next_fire_at),
            None if repeat is None else repeat.to_json(),
        )
        await self._write("create an alarm", _INSERT, params)
        return alarm

    async def list(self, states: Collection[AlarmState] | None = None) -> builtins.list[Alarm]:
        """Return the decodable alarms, in `states` when given, by `next_fire_at` then id.

        Raises:
            CapabilityStoreError: The read failed.
        """
        alarms = self._decode_all(await self._read_rows("list alarms", _SELECT_ALL))
        if states is not None:
            alarms = [a for a in alarms if a.state in states]
        return alarms

    async def due(self, now: datetime) -> builtins.list[Alarm]:
        """Return the scheduled or snoozed alarms whose `next_fire_at` is at or before `now`.

        Times are compared as instants after decoding, not as stored strings, so a row whose
        time does not decode is skipped and reported rather than silently ordered.

        Raises:
            ValueError: `now` is naive.
            CapabilityStoreError: The read failed.
        """
        cutoff = _to_utc(now, "now")
        pending = self._decode_all(await self._read_rows("read due alarms", _SELECT_PENDING))
        return [a for a in pending if a.next_fire_at <= cutoff]

    async def next_due(self) -> Alarm | None:
        """Return the scheduled or snoozed alarm that fires first, or None.

        Raises:
            CapabilityStoreError: The read failed.
        """
        pending = self._decode_all(await self._read_rows("read the next alarm", _SELECT_PENDING))
        return pending[0] if pending else None

    async def rev(self) -> int:
        """Return the store-wide revision, which every changing write increments.

        Raises:
            CapabilityStoreError: The read failed.
        """

        def read(conn: sqlite3.Connection) -> int:
            return int(conn.execute(_SELECT_REV).fetchone()[0])

        return await self._run("read the alarm revision", read)

    async def mark_ringing(self, alarm_id: str, fired_at: datetime, late_by_s: float) -> bool:
        """Move a scheduled or snoozed alarm to ringing, recording when and how late it rang.

        Raises:
            ValueError: `fired_at` is naive or `late_by_s` is negative.
            CapabilityStoreError: The write failed.
        """
        if late_by_s < 0:
            raise ValueError(f"late_by_s must not be negative: {late_by_s}")
        params = (float(late_by_s), _encode_time(_to_utc(fired_at, "fired_at")), alarm_id)
        return await self._write(f"mark alarm {alarm_id} ringing", _TO_RINGING, params)

    async def snooze(self, alarm_id: str, until: datetime) -> bool:
        """Move a ringing alarm to snoozed, firing again at `until`.

        Raises:
            ValueError: `until` is naive.
            CapabilityStoreError: The write failed.
        """
        params = (_encode_time(_to_utc(until, "until")), alarm_id)
        return await self._write(f"snooze alarm {alarm_id}", _TO_SNOOZED, params)

    async def mark_done(self, alarm_id: str) -> bool:
        """Move a ringing or snoozed alarm to done.

        Raises:
            CapabilityStoreError: The write failed.
        """
        return await self._write(f"mark alarm {alarm_id} done", _TO_DONE, (alarm_id,))

    async def mark_missed(self, alarm_id: str) -> bool:
        """Move a scheduled, snoozed or ringing alarm to missed.

        Raises:
            CapabilityStoreError: The write failed.
        """
        return await self._write(f"mark alarm {alarm_id} missed", _TO_MISSED, (alarm_id,))

    async def cancel(self, alarm_id: str) -> bool:
        """Delete an alarm in any state, including one whose row does not decode.

        Returns:
            True when a row was deleted, False when no alarm has that id.

        Raises:
            CapabilityStoreError: The write failed.
        """
        return await self._write(f"cancel alarm {alarm_id}", _DELETE, (alarm_id,))

    async def _write(self, action: str, sql: str, params: tuple[object, ...]) -> bool:
        """Run `sql` and, when it changed a row, the `rev` increment, in one transaction.

        Returns:
            True when the statement changed a row.
        """

        def transaction(conn: sqlite3.Connection) -> bool:
            conn.execute("BEGIN IMMEDIATE")
            try:
                changed = conn.execute(sql, params).rowcount > 0
                if changed:
                    conn.execute(_BUMP_REV)
                conn.execute("COMMIT")
            finally:
                # Still open only when something above raised; the shared connection must not
                # carry the half-done transaction into the next caller's statements.
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
            return changed

        return await self._run(action, transaction)

    async def _run(self, action: str, fn: Callable[[sqlite3.Connection], T]) -> T:
        try:
            return await self._db.run(fn)
        except (sqlite3.Error, CapabilityStoreError) as exc:
            # `CapabilityStoreError` comes from `CapabilitiesDb` when the file cannot be opened.
            raise CapabilityStoreError(f"alarm store cannot {action}: {exc}") from exc

    async def _read_rows(self, action: str, sql: str) -> builtins.list[tuple[object, ...]]:
        def select(conn: sqlite3.Connection) -> builtins.list[tuple[object, ...]]:
            # Text comes back as bytes and is decoded per row in `_decode_row`: with the default
            # factory, one cell of invalid UTF-8 makes the whole fetch raise, which would cost
            # every alarm instead of the one damaged row. The factory is restored before the
            # shared connection's lock is released.
            previous = conn.text_factory
            conn.text_factory = bytes
            try:
                return [tuple(row) for row in conn.execute(sql).fetchall()]
            finally:
                conn.text_factory = previous

        return await self._run(action, select)

    @staticmethod
    def _decode_all(rows: builtins.list[tuple[object, ...]]) -> builtins.list[Alarm]:
        alarms: builtins.list[Alarm] = []
        for row in rows:
            try:
                alarms.append(_decode_row(row))
            except ValueError as exc:
                _report_undecodable(_printable_id(row[0]), exc)
        alarms.sort(key=lambda a: (a.next_fire_at, a.alarm_id))
        return alarms


def _decode_row(row: tuple[object, ...]) -> Alarm:
    """Build an `Alarm` from a stored row whose text cells are still bytes.

    Raises:
        ValueError: A field does not decode, including text that is not valid UTF-8
            (`UnicodeDecodeError` is a `ValueError`); the message names the field.
    """
    alarm_id, kind, label, state, scheduled_for, next_fire_at, repeat, late_by_s, fired_at = row
    kind_text = _decode_text(kind, "kind")
    if kind_text not in _KINDS:
        raise ValueError(f"kind {kind_text!r} is not a known kind")
    state_text = _decode_text(state, "state")
    if state_text not in _STATES:
        raise ValueError(f"state {state_text!r} is not a known state")
    if late_by_s is not None and (
        isinstance(late_by_s, bool) or not isinstance(late_by_s, int | float)
    ):
        raise ValueError(f"late_by_s {late_by_s!r} is not a number")
    repeat_text = None if repeat is None else _decode_text(repeat, "repeat")
    return Alarm(
        alarm_id=_decode_text(alarm_id, "id"),
        kind=kind_text,  # type: ignore[arg-type]  # checked against _KINDS above
        label=None if label is None else _decode_text(label, "label"),
        state=state_text,  # type: ignore[arg-type]  # checked against _STATES above
        scheduled_for=_decode_time(scheduled_for, "scheduled_for"),
        next_fire_at=_decode_time(next_fire_at, "next_fire_at"),
        repeat=None if repeat_text is None else RepeatRule.from_json(repeat_text),
        late_by_s=None if late_by_s is None else float(late_by_s),
        fired_at=None if fired_at is None else _decode_time(fired_at, "fired_at"),
    )


def _decode_text(value: object, field: str) -> str:
    if not isinstance(value, bytes):
        raise ValueError(f"{field} {value!r} is not text")
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{field} is not valid UTF-8: {value!r}") from exc


def _printable_id(value: object) -> str:
    """The row id for a log line, readable even when the id itself does not decode."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _decode_time(value: object, field: str) -> datetime:
    text = _decode_text(value, field)
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} {text!r} has no UTC offset")
    try:
        return parsed.astimezone(UTC)
    except OverflowError as exc:
        # A time at the edge of the calendar ("0001-01-01T00:00:00+05:00") has no UTC form.
        raise ValueError(f"{field} {text!r} is outside the UTC range") from exc


def _encode_time(value: datetime) -> str:
    # Fixed microsecond precision keeps every stored time the same width.
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _to_utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be an aware datetime")
    return value.astimezone(UTC)


def _report_undecodable(alarm_id: str, exc: ValueError) -> None:
    with _reported_lock:
        if alarm_id in _reported_ids:
            return
        _reported_ids.add(alarm_id)
    logger.error("Skipping alarm %s: its stored row does not decode (%s)", alarm_id, exc)
