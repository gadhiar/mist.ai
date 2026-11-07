"""Alarm and timer records, their live occurrences and the event trail (component `alarm_records`).

An alarm or timer is a record, one row per distinct alarm, never deleted, so "list my saved
alarms" can show each with how often it was used. Arming a record creates an occurrence; an
occurrence that ends is deleted in the same transaction that updates its record's counters and
`last_outcome` and appends its event, so the record keeps the history and the occurrence table
holds only what is live. A record is active while it has a live occurrence; there is no enabled
flag to drift from that.

Identity. `identity_for` maps a request to its category and identity: a label makes it tagged, days
without a label make it recurring, and anything else is a one-off. Arming follows the rules of the
category (`AlarmRecords.arm`). A record is repeating when it has a weekday set, whatever its
category, so firing and cancelling key on the set.

Storage. Times are UTC ISO strings at microsecond precision, so one instant has one spelling;
`wall_time` is `HH:MM`; `days`, `sent_to` and `delivered_to` are sorted comma lists; `ring_on` is
`all` or a sorted comma list. Every write that changes a live occurrence bumps the stored `rev`
once, in the same transaction, so a client can order snapshots by it across restarts. Every write
appends its events and trims the trail to `max_events`, oldest first, in that same transaction.

Quarantine (the rule of the alarm store before it, ported). Rows are fetched with text as bytes and
decoded one at a time, so a bad row costs only itself. A row fails when it does not decode or
decodes to values outside the alarm domain (an unknown category or state, a time with no offset or
past year 9000, an identity that disagrees with its fields). Every read skips a failing row, logs
it once at ERROR with its id, and lists it in `quarantined()`; an occurrence whose record fails is
skipped with it. Entries are keyed by the SQL id and a fingerprint of the row as SQL returned it, so
a committed write that changes or deletes the row drops the entry, a read that finds the row changed
checks it again, and a full read whose result lacks the id drops it. `cancel` still deletes a
failing occurrence, because it reads only the columns it needs.

The constructor touches no file; `migrate` creates the tables, so an unwritable database fails
whoever calls `migrate`, not whoever builds the store.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import sqlite3
import uuid
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta, tzinfo
from typing import Literal, TypeVar

from backend.capabilities.db import CapabilitiesDb
from backend.errors import CapabilityStoreError

from .timeparse import RepeatRule, next_occurrence

logger = logging.getLogger(__name__)

T = TypeVar("T")

AlarmKind = Literal["alarm", "timer"]
AlarmCategory = Literal["one_off", "recurring", "tagged"]
OccurrenceState = Literal["scheduled", "ringing", "snoozed", "missed"]
MissedReason = Literal["late", "unheard"]
RecordOutcome = Literal[
    "stopped", "timed_out", "missed", "dismissed", "expired", "cancelled", "moved"
]
EventKind = Literal[
    "set",
    "fired",
    "snoozed",
    "stopped",
    "timed_out",
    "missed",
    "dismissed",
    "expired",
    "cancelled",
    "moved",
    "fallback",
    "reported",
]
EventSource = Literal["fastpath", "model", "proposal", "alarm_action", "alarm_report", "scheduler"]
RingTarget = Literal["all"] | tuple[str, ...]
"""`all`, or endpoint names: sorted, unique, at least one."""

MAX_EVENTS = 20_000
MAX_DURATION_S = 86_400

_KINDS: frozenset[str] = frozenset({"alarm", "timer"})
_CATEGORIES: frozenset[str] = frozenset({"one_off", "recurring", "tagged"})
_STATES: frozenset[str] = frozenset({"scheduled", "ringing", "snoozed", "missed"})
_MISSED_REASONS: frozenset[str] = frozenset({"late", "unheard"})
_OUTCOMES: frozenset[str] = frozenset(
    {"stopped", "timed_out", "missed", "dismissed", "expired", "cancelled", "moved"}
)
_SOURCES: frozenset[str] = frozenset(
    {"fastpath", "model", "proposal", "alarm_action", "alarm_report", "scheduler"}
)
_ALL = "all"
_WALL_TIME = re.compile(r"([01]\d|2[0-3]):([0-5]\d)")
# The latest stored time a row may hold. The furthest a caller adds to a stored time is the ring
# timeout or the missed expiry; this leaves 999 years below `datetime.max` for that sum, so no real
# alarm overflows. A later time is a damaged row, not an alarm.
_LATEST_TIME = datetime(9000, 1, 1, tzinfo=UTC)

_COMPONENT = "alarm_records"
_STEPS = (
    "CREATE TABLE alarm_records ("
    "id TEXT PRIMARY KEY, kind TEXT NOT NULL, category TEXT NOT NULL, "
    "identity TEXT NOT NULL UNIQUE, label TEXT, wall_time TEXT, days TEXT, duration_s INTEGER, "
    "ring_on TEXT NOT NULL, use_count INTEGER NOT NULL, fire_count INTEGER NOT NULL, "
    "created_at TEXT NOT NULL, last_set_at TEXT NOT NULL, last_fired_at TEXT, last_outcome TEXT);\n"
    "CREATE TABLE alarm_occurrences ("
    "id TEXT PRIMARY KEY, record_id TEXT NOT NULL REFERENCES alarm_records(id), "
    "state TEXT NOT NULL, ring_on TEXT NOT NULL, scheduled_for TEXT NOT NULL, "
    "next_fire_at TEXT NOT NULL, fired_at TEXT, late_by_s REAL, missed_at TEXT, "
    "missed_reason TEXT, sent_to TEXT NOT NULL, delivered_to TEXT NOT NULL);\n"
    "CREATE INDEX alarm_occurrences_state ON alarm_occurrences (state);\n"
    "CREATE INDEX alarm_occurrences_record ON alarm_occurrences (record_id);\n"
    "CREATE TABLE alarm_events ("
    "seq INTEGER PRIMARY KEY AUTOINCREMENT, record_id TEXT NOT NULL, occurrence_id TEXT, "
    "at TEXT NOT NULL, kind TEXT NOT NULL, source TEXT, endpoint TEXT, late_by_s REAL, "
    "detail TEXT);\n"
    "CREATE INDEX alarm_events_occurrence ON alarm_events (occurrence_id);\n"
    "CREATE TABLE alarm_records_rev (id INTEGER PRIMARY KEY CHECK (id = 1), rev INTEGER NOT NULL);\n"
    "INSERT INTO alarm_records_rev (id, rev) VALUES (1, 0)",
)

# Every statement is a literal. Each transition names its source states in its WHERE clause, so a
# transition from any other state changes no row.
_RECORD_COLUMNS = (
    "r.id, r.kind, r.category, r.identity, r.label, r.wall_time, r.days, r.duration_s, "
    "r.ring_on, r.use_count, r.fire_count, r.created_at, r.last_set_at, r.last_fired_at, "
    "r.last_outcome"
)
_OCCURRENCE_COLUMNS = (
    "o.id, o.record_id, o.state, o.ring_on, o.scheduled_for, o.next_fire_at, o.fired_at, "
    "o.late_by_s, o.missed_at, o.missed_reason, o.sent_to, o.delivered_to"
)
_RECORD_WIDTH = 15
_OCCURRENCE_WIDTH = 12
_ACTIVE = "EXISTS (SELECT 1 FROM alarm_occurrences x WHERE x.record_id = r.id)"
_SELECT_RECORDS = f"SELECT {_RECORD_COLUMNS}, {_ACTIVE} FROM alarm_records r"  # nosec B608
_SELECT_RECORD_BY_IDENTITY = f"{_SELECT_RECORDS} WHERE r.identity = ?"
_SELECT_LIVE = (
    f"SELECT {_OCCURRENCE_COLUMNS}, {_RECORD_COLUMNS} FROM alarm_occurrences o "  # nosec B608
    "JOIN alarm_records r ON r.id = o.record_id"
)
_SELECT_PENDING = f"{_SELECT_LIVE} WHERE o.state IN ('scheduled', 'snoozed')"
_SELECT_ONE_LIVE = f"{_SELECT_LIVE} WHERE o.id = ?"
_SELECT_TAG_IDENTITIES = "SELECT kind, identity FROM alarm_records WHERE category = 'tagged'"

_INSERT_RECORD = (
    "INSERT INTO alarm_records (id, kind, category, identity, label, wall_time, days, "
    "duration_s, ring_on, use_count, fire_count, created_at, last_set_at, last_fired_at, "
    "last_outcome) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 0, ?, ?, NULL, NULL)"
)
_RECORD_ARMED = (
    "UPDATE alarm_records SET use_count = use_count + 1, last_set_at = ?, ring_on = ? WHERE id = ?"
)
_RECORD_TAKES_VALUES = (
    "UPDATE alarm_records SET wall_time = ?, days = ?, duration_s = ? WHERE id = ?"
)
_RECORD_RETARGETED = "UPDATE alarm_records SET ring_on = ? WHERE id = ?"
_RECORD_FIRED = (
    "UPDATE alarm_records SET fire_count = fire_count + 1, last_fired_at = ? WHERE id = ?"
)
_RECORD_OUTCOME = "UPDATE alarm_records SET last_outcome = ? WHERE id = ?"

_INSERT_OCCURRENCE = (
    "INSERT INTO alarm_occurrences (id, record_id, state, ring_on, scheduled_for, next_fire_at, "
    "fired_at, late_by_s, missed_at, missed_reason, sent_to, delivered_to) "
    "VALUES (?, ?, 'scheduled', ?, ?, ?, NULL, NULL, NULL, NULL, '', '')"
)
_SELECT_SCHEDULED_OF = (
    "SELECT id, scheduled_for, ring_on FROM alarm_occurrences "
    "WHERE record_id = ? AND state = 'scheduled' ORDER BY next_fire_at, id"
)
_SELECT_FIRE_TARGET = (
    "SELECT o.state, o.record_id, o.ring_on, r.days IS NOT NULL FROM alarm_occurrences o "
    "JOIN alarm_records r ON r.id = o.record_id WHERE o.id = ?"
)
_SELECT_STATE = "SELECT state, record_id FROM alarm_occurrences WHERE id = ?"
_SELECT_RINGING = (
    "SELECT record_id, sent_to, delivered_to FROM alarm_occurrences "
    "WHERE id = ? AND state = 'ringing'"
)
_SELECT_MISSED = "SELECT id, record_id, missed_at FROM alarm_occurrences WHERE state = 'missed'"
_SELECT_CANCEL_TARGET = (
    "SELECT o.record_id, r.days IS NOT NULL FROM alarm_occurrences o "
    "JOIN alarm_records r ON r.id = o.record_id WHERE o.id = ?"
)
_SELECT_SERIES = "SELECT id FROM alarm_occurrences WHERE record_id = ? ORDER BY next_fire_at, id"
_TO_RINGING = (
    "UPDATE alarm_occurrences SET state = 'ringing', fired_at = ?, late_by_s = ?, "
    "sent_to = '', delivered_to = '' WHERE id = ? AND state = ?"
)
_TO_MISSED_LATE = (
    "UPDATE alarm_occurrences SET state = 'missed', missed_at = ?, missed_reason = 'late', "
    "late_by_s = ? WHERE id = ? AND state = ?"
)
_TO_MISSED_UNHEARD = (
    "UPDATE alarm_occurrences SET state = 'missed', missed_at = ?, missed_reason = 'unheard' "
    "WHERE id = ? AND state = 'ringing'"
)
_TO_SNOOZED = (
    "UPDATE alarm_occurrences SET state = 'snoozed', next_fire_at = ? "
    "WHERE id = ? AND state = 'ringing'"
)
_SET_SENT_TO = "UPDATE alarm_occurrences SET sent_to = ? WHERE id = ? AND state = 'ringing'"
_SET_DELIVERED_TO = (
    "UPDATE alarm_occurrences SET delivered_to = ? WHERE id = ? AND state = 'ringing'"
)
_SET_OCCURRENCE_RING_ON = "UPDATE alarm_occurrences SET ring_on = ? WHERE id = ?"
_DELETE_OCCURRENCE = "DELETE FROM alarm_occurrences WHERE id = ?"

_INSERT_EVENT = (
    "INSERT INTO alarm_events (record_id, occurrence_id, at, kind, source, endpoint, late_by_s, "
    "detail) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
)
# Sequence numbers only grow and only the oldest rows are deleted, so the rows above
# `MAX(seq) - max_events` are the newest `max_events`, and never more than that.
_TRIM_EVENTS = "DELETE FROM alarm_events WHERE seq <= (SELECT MAX(seq) FROM alarm_events) - ?"
_SELECT_EVENTS_SINCE = "SELECT at, kind, detail FROM alarm_events WHERE at >= ? ORDER BY seq"
_SELECT_OLDEST_EVENT = "SELECT at FROM alarm_events ORDER BY seq LIMIT 1"
_BUMP_REV = "UPDATE alarm_records_rev SET rev = rev + 1 WHERE id = 1"
_SELECT_REV = "SELECT rev FROM alarm_records_rev WHERE id = 1"

_Row = tuple[object, ...]
_Table = Literal["record", "occurrence"]


@dataclass(frozen=True, slots=True)
class AlarmRecord:
    """One distinct alarm or timer, kept for good once set.

    `ring_on` is the target it was last armed with, the default when it is armed again.
    `use_count` counts arms that created an occurrence; `fire_count` counts first fires only, so a
    re-ring after a snooze adds nothing. `active` is derived: the record has a live occurrence.
    """

    record_id: str
    kind: AlarmKind
    category: AlarmCategory
    identity: str
    label: str | None
    wall_time: time | None
    days: frozenset[int] | None
    duration_s: int | None
    ring_on: RingTarget
    use_count: int
    fire_count: int
    created_at: datetime
    last_set_at: datetime
    last_fired_at: datetime | None
    last_outcome: RecordOutcome | None
    active: bool


@dataclass(frozen=True, slots=True)
class Occurrence:
    """One live arming of a record; its id is the `alarm_id` of every WebSocket message.

    `scheduled_for` is the instant it was armed for; `next_fire_at` is when it next fires, which
    a snooze moves later. `sent_to` holds the sinks that emitted the current ring and
    `delivered_to` the endpoints that acknowledged playing it.
    """

    occurrence_id: str
    record: AlarmRecord
    state: OccurrenceState
    ring_on: RingTarget
    scheduled_for: datetime
    next_fire_at: datetime
    fired_at: datetime | None
    late_by_s: float | None
    missed_at: datetime | None
    missed_reason: MissedReason | None
    sent_to: frozenset[str]
    delivered_to: frozenset[str]


@dataclass(frozen=True, slots=True)
class ArmRequest:
    """What to arm. `fire_at` None re-arms a tagged record as saved, and then carries no values.

    `ring_on` None takes the record's last target, or `all` for a new record.
    """

    kind: AlarmKind
    fire_at: datetime | None
    wall_time: time | None
    duration_s: int | None
    label: str | None
    days: frozenset[int] | None
    ring_on: RingTarget | None = None


@dataclass(frozen=True, slots=True)
class ArmResult:
    """The occurrence an arm created or found; `moved` names the occurrences a tagged move ended."""

    occurrence: Occurrence
    created_record: bool
    already_set: bool
    moved: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ArmWrite:
    """What `arm`'s transaction did, for the work done after it commits."""

    row: _Row
    created_record: bool
    already_set: bool
    moved: tuple[str, ...]


def normalise_label(label: str) -> str:
    """Return the form labels are matched in: casefolded, with runs of whitespace as one space."""
    return " ".join(label.split()).casefold()


def identity_for(request: ArmRequest) -> tuple[AlarmCategory, str]:
    """Return the category and the unique identity of the record `request` arms.

    A label makes it tagged, `<kind>:tagged:<normalised label>`. Days without a label make it
    recurring, `alarm:recurring:<HH:MM>:<days>`. Anything else is a one-off,
    `alarm:one_off:<HH:MM>` or `timer:one_off:<duration_s>`.

    Raises:
        ValueError: The request is incoherent: an unknown kind, a timer with days or a wall time,
            an alarm with a duration or without a wall time, a wall time with seconds, a duration
            outside 1 to 86400 s, an empty label or weekday set, a day outside 0 to 6, a naive
            `fire_at`, or a re-arm (`fire_at` None) without a label or with values of its own.
    """
    _check_request(request)
    if request.label is not None:
        return "tagged", f"{request.kind}:tagged:{normalise_label(request.label)}"
    return _value_identity(request.kind, request.wall_time, request.days, request.duration_s)


class AlarmRecords:
    """Alarm records, occurrences and events on a `CapabilitiesDb`, each change one transaction.

    Transitions apply only from the states each method names and return False (or None) when the
    occurrence is missing or in another state, so a stale caller cannot, for example, re-ring an
    occurrence that was stopped. Database failures raise `CapabilityStoreError` naming the alarm
    records.
    """

    def __init__(self, db: CapabilitiesDb, *, max_events: int = MAX_EVENTS) -> None:
        """Keep the database; no file is touched until `migrate` or the first call.

        Raises:
            ValueError: `max_events` is not a positive integer.
        """
        if isinstance(max_events, bool) or not isinstance(max_events, int) or max_events < 1:
            raise ValueError(f"max_events must be a positive integer: {max_events!r}")
        self._db = db
        self._max_events = max_events
        self._tags: dict[str, set[str]] = {"alarm": set(), "timer": set()}
        # Quarantined rows per table: printable id -> fingerprint of the row as SQL returned it,
        # or None for an id set aside before any read saw its row (see `quarantine`).
        self._held: dict[_Table, dict[str, str | None]] = {"record": {}, "occurrence": {}}
        # Fingerprint of each row last admitted, so `quarantine` can key a row by what it holds.
        self._seen: dict[_Table, dict[str, str]] = {"record": {}, "occurrence": {}}

    def migrate(self) -> None:
        """Create or upgrade the tables and load the tag labels, on the calling thread.

        Raises:
            CapabilityStoreError: The database cannot be opened, migrated or read; the message
                names the alarm records and the database.
        """
        try:
            self._db.migrate(_COMPONENT, _STEPS)
            # `CapabilitiesDb` has no public synchronous read, and `tag_labels` must be right from
            # the first sync call, so the load uses its locked call directly, as `migrate` does.
            rows = self._db._call(lambda conn: _fetch_rows(conn, _SELECT_TAG_IDENTITIES, ()))
        except (sqlite3.Error, CapabilityStoreError) as exc:
            raise CapabilityStoreError(f"alarm records cannot migrate: {exc}") from exc
        tags: dict[str, set[str]] = {"alarm": set(), "timer": set()}
        for kind, identity in rows:
            try:
                kind_text = _decode_text(kind, "kind")
                identity_text = _decode_text(identity, "identity")
            except ValueError:
                continue  # A damaged row is quarantined by the first read that meets it.
            prefix = f"{kind_text}:tagged:"
            if kind_text in tags and identity_text.startswith(prefix):
                tags[kind_text].add(identity_text[len(prefix) :])
        self._tags = tags

    async def arm(
        self, request: ArmRequest, *, at: datetime, tz: tzinfo, source: EventSource
    ) -> ArmResult:
        """Arm `request` by the rules of its category, in one write transaction.

        Every arm that creates an occurrence adds 1 to `use_count` and appends a `set` event; one
        that finds the request already set adds nothing. Only a `scheduled` occurrence counts as
        set: one ringing, snoozed or missed ends its own way, whatever is armed beside it.

        - A new identity inserts its record with the first occurrence.
        - One-off: a scheduled occurrence for the same instant makes it `already_set`; otherwise
          the record gains another occurrence.
        - Recurring: any scheduled occurrence makes it `already_set`; otherwise one is armed.
        - Tagged, label alone (`fire_at` None): `already_set` while it has a scheduled occurrence;
          otherwise it arms the next instant of its saved time and days in `tz`, or its saved
          duration from `at`. With values: the same values and instant as its scheduled
          occurrence make it `already_set`; otherwise the record takes the new values, any
          scheduled occurrence ends `moved`, and the new one is armed.

        `already_set` with a `ring_on` retargets that occurrence, and the record's default, and
        nothing else; without one it writes nothing.

        Raises:
            ValueError: The request is incoherent (`identity_for`), `at` is naive, `source` is
                unknown, or `ring_on` is not `all` or a non-empty collection of endpoint names.
            LookupError: A re-arm names a tag no record of that kind holds; nothing is written.
            CapabilityStoreError: The write failed, or the record's stored row does not decode.
        """
        category, identity = identity_for(request)
        now = _to_utc(at, "at")
        _check_source(source)
        ring_on = None if request.ring_on is None else _normalise_ring_on(request.ring_on)
        fire_at = None if request.fire_at is None else _to_utc(request.fire_at, "fire_at")
        values = (
            None if request.wall_time is None else _encode_wall_time(request.wall_time),
            None if request.days is None else _encode_days(request.days),
            request.duration_s,
        )
        new_record_id = uuid.uuid4().hex
        new_id = uuid.uuid4().hex
        stamp = _encode_time(now)

        def work(conn: sqlite3.Connection) -> _ArmWrite:
            found = _fetch_rows(conn, _SELECT_RECORD_BY_IDENTITY, (identity,))
            if not found:
                if fire_at is None:
                    raise LookupError(f"no {request.kind} is tagged {request.label!r}")
                target = _encode_ring_on(_ALL if ring_on is None else ring_on)
                conn.execute(
                    _INSERT_RECORD,
                    (
                        new_record_id,
                        request.kind,
                        category,
                        identity,
                        None if request.label is None else request.label.strip(),
                        *values,
                        target,
                        stamp,
                        stamp,
                    ),
                )
                self._insert_armed(conn, new_id, new_record_id, target, fire_at, stamp, source)
                return _ArmWrite(_fetch_one_live(conn, new_id), True, False, ())
            try:
                record = _decode_record(found[0][:_RECORD_WIDTH], active=bool(found[0][-1]))
            except ValueError as exc:
                raise CapabilityStoreError(
                    f"record {_printable_id(found[0][0])} does not decode: {exc}"
                ) from exc
            scheduled = conn.execute(_SELECT_SCHEDULED_OF, (record.record_id,)).fetchall()
            target = _encode_ring_on(record.ring_on if ring_on is None else ring_on)
            arm_at = fire_at
            moved: list[str] = []
            if category == "one_off":
                assert fire_at is not None  # identity_for: only a tag re-arm has no fire_at
                encoded = _encode_time(fire_at)
                match = [row for row in scheduled if row[1] == encoded]
                if match:
                    return self._already_set(conn, match[0], record.record_id, ring_on)
            elif category == "recurring":
                if scheduled:
                    return self._already_set(conn, scheduled[0], record.record_id, ring_on)
            elif fire_at is None:
                if scheduled:
                    return self._already_set(conn, scheduled[0], record.record_id, ring_on)
                arm_at = _saved_instant(record, now, tz)
            else:
                same = values == (
                    None if record.wall_time is None else _encode_wall_time(record.wall_time),
                    None if record.days is None else _encode_days(record.days),
                    record.duration_s,
                )
                encoded = _encode_time(fire_at)
                match = [row for row in scheduled if same and row[1] == encoded]
                if match:
                    return self._already_set(conn, match[0], record.record_id, ring_on)
                for row in scheduled:
                    occurrence_id = str(row[0])
                    conn.execute(_DELETE_OCCURRENCE, (occurrence_id,))
                    self._event(
                        conn,
                        record.record_id,
                        occurrence_id,
                        stamp,
                        "moved",
                        source,
                        detail={"to": new_id},
                    )
                    moved.append(occurrence_id)
                if moved:
                    conn.execute(_RECORD_OUTCOME, ("moved", record.record_id))
                conn.execute(_RECORD_TAKES_VALUES, (*values, record.record_id))
            assert arm_at is not None
            conn.execute(_RECORD_ARMED, (stamp, target, record.record_id))
            self._insert_armed(conn, new_id, record.record_id, target, arm_at, stamp, source)
            return _ArmWrite(_fetch_one_live(conn, new_id), False, False, tuple(moved))

        written = await self._transact(f"arm {identity}", work)
        occurrence = _decode_live(written.row)
        self._forget("record", [occurrence.record.record_id])
        self._forget("occurrence", [occurrence.occurrence_id, *written.moved])
        if category == "tagged":
            self._tags[request.kind].add(identity.split(":", 2)[2])
        return ArmResult(occurrence, written.created_record, written.already_set, written.moved)

    async def find_tag(self, kind: AlarmKind, label: str) -> AlarmRecord | None:
        """Return the tagged record of `kind` whose label matches `label`, or None.

        Labels match as `normalise_label` gives them, so case and repeated spaces do not count.
        A record whose row fails is quarantined and reads as None.

        Raises:
            CapabilityStoreError: The read failed.
        """
        identity = f"{kind}:tagged:{normalise_label(label)}"
        found = await self._run(
            f"find {identity}",
            lambda conn: _fetch_rows(conn, _SELECT_RECORD_BY_IDENTITY, (identity,)),
        )
        for row in found:
            return self._admit_record(row)
        return None

    def tag_labels(self, kind: AlarmKind) -> frozenset[str]:
        """Return the normalised labels of the tagged records of `kind`, synchronously.

        For the synchronous fast-path grammar. They are held in memory: `migrate` loads them and
        each committed arm of a tagged record adds its own.
        """
        return frozenset(self._tags.get(kind, ()))

    async def snapshot(self) -> tuple[int, list[Occurrence]]:
        """Return `rev` and every live occurrence, read together in one read transaction.

        A published state pairs the list with the `rev` it was read at; two separate reads would
        let a write land between them and pair a list with a later `rev`.

        Raises:
            CapabilityStoreError: The read failed.
        """

        def read(conn: sqlite3.Connection) -> tuple[int, list[_Row]]:
            conn.execute("BEGIN")
            try:
                rev = int(conn.execute(_SELECT_REV).fetchone()[0])
                rows = _fetch_rows(conn, _SELECT_LIVE, ())
                conn.execute("COMMIT")
            finally:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
            return rev, rows

        rev, rows = await self._run("read the alarm snapshot", read)
        return rev, self._admit_live(rows, complete=True)

    async def saved(self, limit: int | None = None) -> list[AlarmRecord]:
        """Return the records, active or not, by `use_count` then `last_set_at`, most first.

        `limit` None returns every record. A quarantined record is skipped before the limit is
        applied, so it never takes the place of a good one.

        Raises:
            ValueError: `limit` is negative or not an integer.
            CapabilityStoreError: The read failed.
        """
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 0
        ):
            raise ValueError(f"limit must be None or a non-negative integer: {limit!r}")
        rows = await self._run(
            "list the saved alarms", lambda conn: _fetch_rows(conn, _SELECT_RECORDS, ())
        )
        self._prune("record", rows)
        records = [r for r in (self._admit_record(row) for row in rows) if r is not None]
        records.sort(key=lambda r: (-r.use_count, -r.last_set_at.timestamp(), r.record_id))
        return records if limit is None else records[:limit]

    async def due(self, now: datetime) -> list[Occurrence]:
        """Return the scheduled or snoozed occurrences whose `next_fire_at` is at or before `now`.

        Raises:
            ValueError: `now` is naive.
            CapabilityStoreError: The read failed.
        """
        cutoff = _to_utc(now, "now")
        pending = await self._pending("read the due alarms")
        return [o for o in pending if o.next_fire_at <= cutoff]

    async def next_due(self) -> datetime | None:
        """Return when the first scheduled or snoozed occurrence fires, or None.

        Raises:
            CapabilityStoreError: The read failed.
        """
        pending = await self._pending("read the next alarm")
        return pending[0].next_fire_at if pending else None

    async def fire(
        self,
        occurrence_id: str,
        *,
        outcome: Literal["ringing", "missed"],
        at: datetime,
        late_by_s: float,
        next_fire_at: datetime | None,
    ) -> bool:
        """Fire a due occurrence as `outcome`, in one write transaction.

        From `scheduled`: it rings (or is missed, reason `late`, past the grace window),
        `fire_count` goes up by 1, the `fired` (or `missed`) event is appended and, when
        `next_fire_at` is given, the record's next scheduled occurrence is inserted with the same
        target. All of it lands or none does, so a retry of an occurrence still due adds exactly
        one next occurrence. From `snoozed`: it rings again with `sent_to` and `delivered_to`
        empty, with no count and no next occurrence, whatever `next_fire_at` is.

        Returns:
            True when it fired; False, with nothing written, when it is missing or neither
            scheduled nor snoozed.

        Raises:
            ValueError: `outcome` is unknown, `at` or `next_fire_at` is naive, `late_by_s` is
                negative or not finite, or `next_fire_at` is given for a record with no weekday
                set; nothing is written.
            CapabilityStoreError: The write failed.
        """
        if outcome not in ("ringing", "missed"):
            raise ValueError(f"unknown fire outcome {outcome!r}")
        if isinstance(late_by_s, bool) or not math.isfinite(late_by_s) or late_by_s < 0:
            raise ValueError(f"late_by_s must be finite and not negative: {late_by_s!r}")
        stamp = _encode_time(_to_utc(at, "at"))
        next_at = (
            None if next_fire_at is None else _encode_time(_to_utc(next_fire_at, "next_fire_at"))
        )
        next_id = uuid.uuid4().hex

        def work(conn: sqlite3.Connection) -> bool:
            target = conn.execute(_SELECT_FIRE_TARGET, (occurrence_id,)).fetchone()
            if target is None or target[0] not in ("scheduled", "snoozed"):
                return False
            from_state, record_id, ring_on, repeating = target
            if outcome == "ringing":
                params = (stamp, float(late_by_s), occurrence_id, from_state)
                changed = conn.execute(_TO_RINGING, params).rowcount > 0
            else:
                params = (stamp, float(late_by_s), occurrence_id, from_state)
                changed = conn.execute(_TO_MISSED_LATE, params).rowcount > 0
            if not changed:
                return False
            detail: dict[str, object] = {}
            if from_state == "scheduled":
                conn.execute(_RECORD_FIRED, (stamp, record_id))
                if next_at is not None:
                    if not repeating:
                        raise ValueError(
                            f"occurrence {occurrence_id} has no weekday set, so no next occurrence"
                        )
                    conn.execute(
                        _INSERT_OCCURRENCE, (next_id, record_id, ring_on, next_at, next_at)
                    )
                    detail["next_occurrence_id"] = next_id
            else:
                detail["after_snooze"] = True
            if outcome == "missed":
                conn.execute(_RECORD_OUTCOME, ("missed", record_id))
                detail["reason"] = "late"
            kind: EventKind = "fired" if outcome == "ringing" else "missed"
            self._event(
                conn,
                record_id,
                occurrence_id,
                stamp,
                kind,
                "scheduler",
                late_by_s=float(late_by_s),
                detail=detail or None,
            )
            self._close(conn, bump=True)
            return True

        fired = await self._transact(f"fire occurrence {occurrence_id} as {outcome}", work)
        if fired:
            self._forget("occurrence", [occurrence_id])
        return fired

    async def sent(
        self, occurrence_id: str, endpoints: Collection[str], *, fallback: bool, at: datetime
    ) -> bool:
        """Add the sinks that emitted a ringing occurrence's ring to its `sent_to`.

        With `fallback`, a `fallback` event records that none of its targets was present and the
        ring went to every present sink.

        Returns:
            True when the occurrence is ringing; False, with nothing written, otherwise.

        Raises:
            ValueError: An endpoint name is empty, holds a comma or is `all`, or `at` is naive.
            CapabilityStoreError: The write failed.
        """
        names = _check_names(endpoints)
        stamp = _encode_time(_to_utc(at, "at"))

        def work(conn: sqlite3.Connection) -> bool:
            row = conn.execute(_SELECT_RINGING, (occurrence_id,)).fetchone()
            if row is None:
                return False
            record_id, sent_to, _ = row
            merged = _decode_names(sent_to, "sent_to") | names
            changed = _encode_names(merged) != sent_to
            if changed:
                conn.execute(_SET_SENT_TO, (_encode_names(merged), occurrence_id))
            if fallback:
                self._event(
                    conn,
                    record_id,
                    occurrence_id,
                    stamp,
                    "fallback",
                    "scheduler",
                    detail={"sent_to": sorted(names)},
                )
            self._close(conn, bump=changed)
            return True

        return await self._changed(
            f"record where occurrence {occurrence_id} rang", occurrence_id, work
        )

    async def acknowledge_ring(self, occurrence_id: str, endpoint: str, *, at: datetime) -> bool:
        """Add `endpoint` to a ringing occurrence's `delivered_to`: it acknowledged playing it.

        Returns:
            True when the occurrence is ringing and now lists the endpoint; False otherwise.

        Raises:
            ValueError: The endpoint name is empty, holds a comma or is `all`, or `at` is naive.
            CapabilityStoreError: The write failed.
        """
        (name,) = _check_names([endpoint])
        _to_utc(at, "at")

        def work(conn: sqlite3.Connection) -> bool:
            row = conn.execute(_SELECT_RINGING, (occurrence_id,)).fetchone()
            if row is None:
                return False
            delivered = _decode_names(row[2], "delivered_to")
            if name not in delivered:
                conn.execute(_SET_DELIVERED_TO, (_encode_names(delivered | {name}), occurrence_id))
                self._close(conn, bump=True)
            return True

        return await self._changed(f"acknowledge the ring of {occurrence_id}", occurrence_id, work)

    async def snooze(
        self,
        occurrence_id: str,
        *,
        until: datetime,
        at: datetime,
        source: EventSource,
        endpoint: str | None,
    ) -> bool:
        """Move a ringing occurrence to snoozed, firing again at `until`.

        Returns:
            True when it was ringing; False, with nothing written, otherwise.

        Raises:
            ValueError: `until` or `at` is naive, or `source` is unknown.
            CapabilityStoreError: The write failed.
        """
        next_at = _encode_time(_to_utc(until, "until"))
        stamp = _encode_time(_to_utc(at, "at"))
        _check_source(source)

        def work(conn: sqlite3.Connection) -> bool:
            row = conn.execute(_SELECT_STATE, (occurrence_id,)).fetchone()
            if row is None or conn.execute(_TO_SNOOZED, (next_at, occurrence_id)).rowcount == 0:
                return False
            self._event(
                conn,
                row[1],
                occurrence_id,
                stamp,
                "snoozed",
                source,
                endpoint=endpoint,
                detail={"until": next_at},
            )
            self._close(conn, bump=True)
            return True

        return await self._changed(f"snooze occurrence {occurrence_id}", occurrence_id, work)

    async def finish(
        self,
        occurrence_id: str,
        *,
        outcome: Literal["stopped", "timed_out"],
        at: datetime,
        source: EventSource,
        endpoint: str | None,
    ) -> bool:
        """End an occurrence as `outcome`: deleted, its record updated, its event appended.

        `stopped` ends a ringing or snoozed occurrence; `timed_out` ends a ringing one.

        Returns:
            True when it ended; False, with nothing written, when it is missing or in another
            state.

        Raises:
            ValueError: `outcome` or `source` is unknown, or `at` is naive.
            CapabilityStoreError: The write failed.
        """
        if outcome not in ("stopped", "timed_out"):
            raise ValueError(f"unknown finish outcome {outcome!r}")
        stamp = _encode_time(_to_utc(at, "at"))
        _check_source(source)
        states = ("ringing", "snoozed") if outcome == "stopped" else ("ringing",)

        def work(conn: sqlite3.Connection) -> bool:
            row = conn.execute(_SELECT_STATE, (occurrence_id,)).fetchone()
            if row is None or row[0] not in states:
                return False
            self._end(conn, occurrence_id, row[1], stamp, outcome, source, endpoint)
            self._close(conn, bump=True)
            return True

        return await self._changed(f"finish occurrence {occurrence_id}", occurrence_id, work)

    async def time_out(
        self, occurrence_id: str, *, at: datetime
    ) -> Literal["timed_out", "missed"] | None:
        """End a ringing occurrence at its ring timeout.

        One that an endpoint acknowledged playing (`delivered_to` not empty) finishes
        `timed_out`. One that nobody acknowledged becomes missed, reason `unheard`, with its
        `sent_to` in the event's detail, so the missed list shows a ring that was sent but never
        heard.

        Returns:
            `timed_out` or `missed`; None, with nothing written, when it is no longer ringing.

        Raises:
            ValueError: `at` is naive.
            CapabilityStoreError: The write failed.
        """
        stamp = _encode_time(_to_utc(at, "at"))

        def work(conn: sqlite3.Connection) -> Literal["timed_out", "missed"] | None:
            row = conn.execute(_SELECT_RINGING, (occurrence_id,)).fetchone()
            if row is None:
                return None
            record_id, sent_to, delivered_to = row
            delivered = _decode_names(delivered_to, "delivered_to")
            if delivered:
                self._end(
                    conn,
                    occurrence_id,
                    record_id,
                    stamp,
                    "timed_out",
                    "scheduler",
                    None,
                    detail={"delivered_to": sorted(delivered)},
                )
                self._close(conn, bump=True)
                return "timed_out"
            conn.execute(_TO_MISSED_UNHEARD, (stamp, occurrence_id))
            conn.execute(_RECORD_OUTCOME, ("missed", record_id))
            self._event(
                conn,
                record_id,
                occurrence_id,
                stamp,
                "missed",
                "scheduler",
                detail={"reason": "unheard", "sent_to": sorted(_decode_names(sent_to, "sent_to"))},
            )
            self._close(conn, bump=True)
            return "missed"

        result = await self._transact(f"time out occurrence {occurrence_id}", work)
        if result is not None:
            self._forget("occurrence", [occurrence_id])
        return result

    async def acknowledge(
        self, occurrence_id: str, *, at: datetime, by: Literal["dismiss", "expiry"]
    ) -> bool:
        """End a missed occurrence, recorded as `dismissed` or `expired`.

        Returns:
            True when it was missed; False, with nothing written, otherwise.

        Raises:
            ValueError: `by` is unknown or `at` is naive.
            CapabilityStoreError: The write failed.
        """
        outcome = _acknowledged_outcome(by)
        stamp = _encode_time(_to_utc(at, "at"))
        source: EventSource | None = "scheduler" if by == "expiry" else None

        def work(conn: sqlite3.Connection) -> bool:
            row = conn.execute(_SELECT_STATE, (occurrence_id,)).fetchone()
            if row is None or row[0] != "missed":
                return False
            self._end(conn, occurrence_id, row[1], stamp, outcome, source, None)
            self._close(conn, bump=True)
            return True

        return await self._changed(f"acknowledge occurrence {occurrence_id}", occurrence_id, work)

    async def acknowledge_missed_before(self, cutoff: datetime, *, at: datetime) -> int:
        """Expire every missed occurrence missed before `cutoff`, in one transaction.

        A missed occurrence whose `missed_at` does not decode is left for a read to quarantine.

        Returns:
            The number expired.

        Raises:
            ValueError: `cutoff` or `at` is naive.
            CapabilityStoreError: The write failed.
        """
        limit = _to_utc(cutoff, "cutoff")
        stamp = _encode_time(_to_utc(at, "at"))

        def work(conn: sqlite3.Connection) -> list[str]:
            expired: list[str] = []
            for occurrence_id, record_id, missed_at in _fetch_rows(conn, _SELECT_MISSED, ()):
                try:
                    if _decode_time(missed_at, "missed_at") >= limit:
                        continue
                    occurrence_text = _decode_text(occurrence_id, "id")
                    record_text = _decode_text(record_id, "record_id")
                except ValueError:
                    continue
                self._end(conn, occurrence_text, record_text, stamp, "expired", "scheduler", None)
                expired.append(occurrence_text)
            self._close(conn, bump=bool(expired))
            return expired

        expired = await self._transact("expire missed alarms", work)
        self._forget("occurrence", expired)
        return len(expired)

    async def cancel(
        self, occurrence_ids: Collection[str], *, at: datetime, source: EventSource
    ) -> int:
        """Delete the occurrences named, each with a `cancelled` event, in one transaction.

        A one-off's occurrences go one by one. Naming any occurrence of a repeating record (one
        with a weekday set) deletes its whole series, every live occurrence it has. The record
        stays saved, and setting it again re-arms it. A quarantined occurrence is deleted too.

        Returns:
            The number of occurrences deleted.

        Raises:
            ValueError: `at` is naive or `source` is unknown.
            CapabilityStoreError: The write failed.
        """
        stamp = _encode_time(_to_utc(at, "at"))
        _check_source(source)
        named = list(dict.fromkeys(occurrence_ids))

        def work(conn: sqlite3.Connection) -> list[str]:
            targets: dict[str, str] = {}
            for occurrence_id in named:
                row = conn.execute(_SELECT_CANCEL_TARGET, (occurrence_id,)).fetchone()
                if row is None:
                    continue
                record_id, repeating = row
                if repeating:
                    for (series_id,) in conn.execute(_SELECT_SERIES, (record_id,)).fetchall():
                        targets.setdefault(series_id, record_id)
                else:
                    targets.setdefault(occurrence_id, record_id)
            for occurrence_id, record_id in targets.items():
                self._end(conn, occurrence_id, record_id, stamp, "cancelled", source, None)
            self._close(conn, bump=bool(targets))
            return list(targets)

        cancelled = await self._transact("cancel alarms", work)
        self._forget("occurrence", cancelled)
        return len(cancelled)

    async def event_counts(self, since: datetime, tz: tzinfo) -> dict[str, dict[str, int]]:
        """Count the kept events at or after `since`, per local date in `tz` and event kind.

        Dates are ISO dates. Missed events are split by reason, as `missed:late` and
        `missed:unheard`. An event whose time does not decode is not counted.

        Raises:
            ValueError: `since` is naive.
            CapabilityStoreError: The read failed.
        """
        start = _encode_time(_to_utc(since, "since"))
        rows = await self._run(
            "count alarm events", lambda conn: _fetch_rows(conn, _SELECT_EVENTS_SINCE, (start,))
        )
        counts: dict[str, dict[str, int]] = {}
        for at, kind, detail in rows:
            try:
                day = _decode_time(at, "at").astimezone(tz).date().isoformat()
                key = _decode_text(kind, "kind")
            except ValueError:
                continue
            if key == "missed":
                key = f"missed:{_detail_reason(detail)}"
            per_day = counts.setdefault(day, {})
            per_day[key] = per_day.get(key, 0) + 1
        return counts

    async def oldest_event_at(self) -> datetime | None:
        """Return the time of the oldest kept event, or None when the trail is empty.

        Raises:
            CapabilityStoreError: The read failed, or the oldest event's time does not decode.
        """
        rows = await self._run(
            "read the oldest alarm event", lambda conn: _fetch_rows(conn, _SELECT_OLDEST_EVENT, ())
        )
        if not rows:
            return None
        try:
            return _decode_time(rows[0][0], "at")
        except ValueError as exc:
            raise CapabilityStoreError(
                f"alarm records: the oldest event is damaged: {exc}"
            ) from exc

    async def rev(self) -> int:
        """Return the store-wide revision, which every write that changes a live row increments.

        Raises:
            CapabilityStoreError: The read failed.
        """
        return await self._run(
            "read the alarm revision", lambda conn: int(conn.execute(_SELECT_REV).fetchone()[0])
        )

    def quarantine(self, row_id: str, reason: str) -> None:
        """Set a row aside until it changes: every read skips it from now on.

        For a row whose handling raised a data error, which a retry cannot clear. It is keyed by
        the fingerprint of the row the store last returned under that id, so a store write that
        changes the row, or an edit a later read finds, checks it again. An id no read has seen
        is taken as an occurrence's, the scheduler's case, and holds the row the next read finds.
        The caller logs the failure; this records it at DEBUG with `reason`.
        """
        if row_id in self._seen["occurrence"]:
            self._held["occurrence"][row_id] = self._seen["occurrence"][row_id]
        elif row_id in self._seen["record"]:
            self._held["record"][row_id] = self._seen["record"][row_id]
        else:
            self._held["occurrence"][row_id] = None
        logger.debug("Alarm row %s quarantined until it changes: %s", row_id, reason)

    def quarantined(self) -> tuple[str, ...]:
        """Return the sorted ids of the rows set aside, for the capability's status.

        Each is still stored and still failing as far as the store has seen: a write that
        changes or deletes a row, a read that finds it changed, and a full read whose result no
        longer holds it, each drop its entry.
        """
        return tuple(sorted(self._held["record"].keys() | self._held["occurrence"].keys()))

    def _insert_armed(
        self,
        conn: sqlite3.Connection,
        occurrence_id: str,
        record_id: str,
        target: str,
        fire_at: datetime,
        stamp: str,
        source: EventSource,
    ) -> None:
        """Insert a scheduled occurrence with its `set` event, and close the transaction's work."""
        when = _encode_time(fire_at)
        conn.execute(_INSERT_OCCURRENCE, (occurrence_id, record_id, target, when, when))
        self._event(conn, record_id, occurrence_id, stamp, "set", source, detail={"fire_at": when})
        self._close(conn, bump=True)

    def _already_set(
        self,
        conn: sqlite3.Connection,
        scheduled: tuple[object, ...],
        record_id: str,
        ring_on: RingTarget | None,
    ) -> _ArmWrite:
        """Retarget the scheduled occurrence an arm found, when the arm names a target."""
        occurrence_id = str(scheduled[0])
        if ring_on is not None and _encode_ring_on(ring_on) != scheduled[2]:
            target = _encode_ring_on(ring_on)
            conn.execute(_SET_OCCURRENCE_RING_ON, (target, occurrence_id))
            conn.execute(_RECORD_RETARGETED, (target, record_id))
            self._close(conn, bump=True)
        return _ArmWrite(_fetch_one_live(conn, occurrence_id), False, True, ())

    def _end(
        self,
        conn: sqlite3.Connection,
        occurrence_id: str,
        record_id: str,
        stamp: str,
        outcome: RecordOutcome,
        source: EventSource | None,
        endpoint: str | None,
        *,
        detail: dict[str, object] | None = None,
    ) -> None:
        """Delete an occurrence, set its record's `last_outcome` and append its event."""
        conn.execute(_DELETE_OCCURRENCE, (occurrence_id,))
        conn.execute(_RECORD_OUTCOME, (outcome, record_id))
        self._event(
            conn, record_id, occurrence_id, stamp, outcome, source, endpoint=endpoint, detail=detail
        )

    @staticmethod
    def _event(
        conn: sqlite3.Connection,
        record_id: object,
        occurrence_id: str | None,
        stamp: str,
        kind: EventKind,
        source: EventSource | None,
        *,
        endpoint: str | None = None,
        late_by_s: float | None = None,
        detail: dict[str, object] | None = None,
    ) -> None:
        encoded = None if detail is None else json.dumps(detail, sort_keys=True)
        conn.execute(
            _INSERT_EVENT,
            (record_id, occurrence_id, stamp, kind, source, endpoint, late_by_s, encoded),
        )

    def _close(self, conn: sqlite3.Connection, *, bump: bool) -> None:
        """Trim the trail and, when a live row changed, bump `rev`: the last writes of a change.

        Called once per path through a transaction, so `rev` goes up once per write.
        """
        conn.execute(_TRIM_EVENTS, (self._max_events,))
        if bump:
            conn.execute(_BUMP_REV)

    async def _changed(
        self, action: str, occurrence_id: str, work: Callable[[sqlite3.Connection], bool]
    ) -> bool:
        changed = await self._transact(action, work)
        if changed:
            self._forget("occurrence", [occurrence_id])
        return changed

    async def _transact(self, action: str, work: Callable[[sqlite3.Connection], T]) -> T:
        """Run `work` in one write transaction, committed when it returns, rolled back if it raises.

        `BEGIN IMMEDIATE` takes the write lock before the first read, so the rows a transition
        reads cannot change before it writes.
        """

        def transaction(conn: sqlite3.Connection) -> T:
            conn.execute("BEGIN IMMEDIATE")
            try:
                result = work(conn)
                conn.execute("COMMIT")
            finally:
                # Still open only when something above raised; the shared connection must not
                # carry the half-done transaction into the next caller's statements.
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
            return result

        return await self._run(action, transaction)

    async def _run(self, action: str, fn: Callable[[sqlite3.Connection], T]) -> T:
        try:
            return await self._db.run(fn)
        except (sqlite3.Error, CapabilityStoreError) as exc:
            # `CapabilityStoreError` comes from `CapabilitiesDb` when the file cannot be opened.
            raise CapabilityStoreError(f"alarm records cannot {action}: {exc}") from exc

    async def _pending(self, action: str) -> list[Occurrence]:
        rows = await self._run(action, lambda conn: _fetch_rows(conn, _SELECT_PENDING, ()))
        return self._admit_live(rows, complete=False)

    def _admit_live(self, rows: list[_Row], *, complete: bool) -> list[Occurrence]:
        """Decode joined rows, skipping any whose occurrence or record fails, by next fire then id.

        `complete` says the SQL selected every live occurrence, so an occurrence entry whose id
        the result does not hold is for a row since deleted, and is dropped. That is judged on
        the ids SQL returned, before any row is decoded or skipped.
        """
        if complete:
            self._prune("occurrence", [row[:_OCCURRENCE_WIDTH] for row in rows])
        occurrences: list[Occurrence] = []
        for row in rows:
            occurrence_part = row[:_OCCURRENCE_WIDTH]
            record = self._admit(
                "record", row[_OCCURRENCE_WIDTH:], lambda part: _decode_record(part, active=True)
            )
            if record is None:
                continue
            occurrence = self._admit(
                "occurrence",
                occurrence_part,
                lambda part, record=record: _decode_occurrence(part, record),
            )
            if occurrence is not None:
                occurrences.append(occurrence)
        occurrences.sort(key=lambda o: (o.next_fire_at, o.occurrence_id))
        return occurrences

    def _admit_record(self, row: _Row) -> AlarmRecord | None:
        active = bool(row[_RECORD_WIDTH])
        return self._admit(
            "record", row[:_RECORD_WIDTH], lambda part: _decode_record(part, active=active)
        )

    def _admit(self, table: _Table, row: _Row, decode: Callable[[_Row], T]) -> T | None:
        """Decode one row unless it is quarantined; quarantine and log it once if it fails."""
        row_id = _printable_id(row[0])
        fingerprint = _fingerprint(row)
        held = self._held[table]
        if row_id in held:
            if held[row_id] is None:
                # Set aside before any read saw its row: this is the row it was set aside at.
                held[row_id] = fingerprint
                return None
            if held[row_id] == fingerprint:
                return None
            # The row changed since it was set aside, so it is checked again below.
            del held[row_id]
        try:
            value = decode(row)
        except ValueError as exc:
            held[row_id] = fingerprint
            self._seen[table].pop(row_id, None)
            logger.error("Alarm %s %s quarantined until its row changes: %s", table, row_id, exc)
            return None
        self._seen[table][row_id] = fingerprint
        return value

    def _prune(self, table: _Table, rows: Iterable[_Row]) -> None:
        """Drop what the store holds for ids a complete read of `table` no longer returned."""
        returned = {_printable_id(row[0]) for row in rows}
        held = self._held[table]
        for row_id in held.keys() - returned:
            del held[row_id]
        self._seen[table] = {k: v for k, v in self._seen[table].items() if k in returned}

    def _forget(self, table: _Table, row_ids: Iterable[str]) -> None:
        """Drop what the store holds for rows a committed write changed or deleted."""
        for row_id in row_ids:
            self._held[table].pop(row_id, None)
            self._seen[table].pop(row_id, None)


def _fetch_rows(conn: sqlite3.Connection, sql: str, params: tuple[object, ...]) -> list[_Row]:
    """Run `sql` with text returned as bytes, restoring the connection's text factory after.

    Each row is decoded on its own: with the default factory, one cell of invalid UTF-8 makes the
    whole fetch raise, which would cost every alarm instead of the one damaged row. The factory
    is restored before the shared connection's lock is released.
    """
    previous = conn.text_factory
    conn.text_factory = bytes
    try:
        return [tuple(row) for row in conn.execute(sql, params).fetchall()]
    finally:
        conn.text_factory = previous


def _fetch_one_live(conn: sqlite3.Connection, occurrence_id: str) -> _Row:
    [row] = _fetch_rows(conn, _SELECT_ONE_LIVE, (occurrence_id,))
    return row


def _decode_live(row: _Row) -> Occurrence:
    """Decode a joined row an arm just wrote, which the store's own encoding makes decodable."""
    try:
        record = _decode_record(row[_OCCURRENCE_WIDTH:], active=True)
        return _decode_occurrence(row[:_OCCURRENCE_WIDTH], record)
    except ValueError as exc:
        raise CapabilityStoreError(f"alarm records: an armed row does not decode: {exc}") from exc


def _decode_record(row: _Row, *, active: bool) -> AlarmRecord:
    """Build an `AlarmRecord` from a stored row whose text cells are still bytes.

    Raises:
        ValueError: A field does not decode or is outside the domain, or the fields disagree
            with each other or with the identity; the message names the field.
    """
    (
        record_id,
        kind,
        category,
        identity,
        label,
        wall_time,
        days,
        duration_s,
        ring_on,
        use_count,
        fire_count,
        created_at,
        last_set_at,
        last_fired_at,
        last_outcome,
    ) = row
    kind_text = _decode_choice(kind, "kind", _KINDS)
    category_text = _decode_choice(category, "category", _CATEGORIES)
    label_text = None if label is None else _decode_text(label, "label")
    wall = None if wall_time is None else _decode_wall_time(wall_time)
    day_set = None if days is None else _decode_days(_decode_text(days, "days"))
    duration = None if duration_s is None else _decode_count(duration_s, "duration_s", 1)
    if duration is not None and duration > MAX_DURATION_S:
        raise ValueError(f"duration_s {duration} is over {MAX_DURATION_S}")
    record = AlarmRecord(
        record_id=_decode_text(record_id, "id"),
        kind=kind_text,  # type: ignore[arg-type]  # checked against _KINDS
        category=category_text,  # type: ignore[arg-type]  # checked against _CATEGORIES
        identity=_decode_text(identity, "identity"),
        label=label_text,
        wall_time=wall,
        days=day_set,
        duration_s=duration,
        ring_on=_decode_ring_on(ring_on),
        use_count=_decode_count(use_count, "use_count", 0),
        fire_count=_decode_count(fire_count, "fire_count", 0),
        created_at=_decode_time(created_at, "created_at"),
        last_set_at=_decode_time(last_set_at, "last_set_at"),
        last_fired_at=(
            None if last_fired_at is None else _decode_time(last_fired_at, "last_fired_at")
        ),
        last_outcome=(
            None
            if last_outcome is None
            else _decode_choice(last_outcome, "last_outcome", _OUTCOMES)
        ),  # type: ignore[arg-type]  # checked against _OUTCOMES
        active=active,
    )
    _check_record_shape(record)
    return record


def _check_record_shape(record: AlarmRecord) -> None:
    if record.kind == "alarm" and (record.wall_time is None or record.duration_s is not None):
        raise ValueError("an alarm record needs a wall_time and no duration_s")
    if record.kind == "timer" and (
        record.duration_s is None or record.wall_time is not None or record.days is not None
    ):
        raise ValueError("a timer record needs a duration_s and no wall_time or days")
    if record.category == "tagged":
        if record.label is None or not normalise_label(record.label):
            raise ValueError("a tagged record needs a label")
        expected = f"{record.kind}:tagged:{normalise_label(record.label)}"
    else:
        if record.label is not None:
            raise ValueError(f"a {record.category} record takes no label")
        category, expected = _value_identity(
            record.kind, record.wall_time, record.days, record.duration_s
        )
        if category != record.category:
            raise ValueError(f"category {record.category!r} disagrees with its fields")
    if record.identity != expected:
        raise ValueError(f"identity {record.identity!r} disagrees with its fields")


def _decode_occurrence(row: _Row, record: AlarmRecord) -> Occurrence:
    """Build an `Occurrence` of `record` from a stored row whose text cells are still bytes.

    Raises:
        ValueError: A field does not decode or is outside the domain, or the state disagrees
            with the fields it needs; the message names the field.
    """
    (
        occurrence_id,
        record_id,
        state,
        ring_on,
        scheduled_for,
        next_fire_at,
        fired_at,
        late_by_s,
        missed_at,
        missed_reason,
        sent_to,
        delivered_to,
    ) = row
    if _decode_text(record_id, "record_id") != record.record_id:
        raise ValueError("record_id disagrees with the joined record")
    state_text = _decode_choice(state, "state", _STATES)
    reason = (
        None
        if missed_reason is None
        else _decode_choice(missed_reason, "missed_reason", _MISSED_REASONS)
    )
    occurrence = Occurrence(
        occurrence_id=_decode_text(occurrence_id, "id"),
        record=record,
        state=state_text,  # type: ignore[arg-type]  # checked against _STATES
        ring_on=_decode_ring_on(ring_on),
        scheduled_for=_decode_time(scheduled_for, "scheduled_for"),
        next_fire_at=_decode_time(next_fire_at, "next_fire_at"),
        fired_at=None if fired_at is None else _decode_time(fired_at, "fired_at"),
        late_by_s=None if late_by_s is None else _decode_late_by(late_by_s),
        missed_at=None if missed_at is None else _decode_time(missed_at, "missed_at"),
        missed_reason=reason,  # type: ignore[arg-type]  # checked against _MISSED_REASONS
        sent_to=_decode_names(sent_to, "sent_to"),
        delivered_to=_decode_names(delivered_to, "delivered_to"),
    )
    missed = occurrence.state == "missed"
    if missed != (occurrence.missed_at is not None) or missed != (reason is not None):
        raise ValueError("missed_at and missed_reason are set exactly when the state is missed")
    if occurrence.state in ("ringing", "snoozed") and occurrence.fired_at is None:
        raise ValueError(f"a {occurrence.state} occurrence needs fired_at")
    return occurrence


def _check_request(request: ArmRequest) -> None:
    if not isinstance(request.kind, str) or request.kind not in _KINDS:
        raise ValueError(f"unknown alarm kind {request.kind!r}")
    if request.label is not None and (
        not isinstance(request.label, str) or not normalise_label(request.label)
    ):
        raise ValueError(f"a label must be non-empty text: {request.label!r}")
    if request.fire_at is None:
        if request.label is None:
            raise ValueError("a request without fire_at re-arms a tag, so it needs a label")
        if (request.wall_time, request.days, request.duration_s) != (None, None, None):
            raise ValueError("a tag re-arm carries no time, days or duration of its own")
        return
    _to_utc(request.fire_at, "fire_at")
    if request.days is not None:
        _check_days(request.days)
    if request.kind == "timer":
        if request.days is not None or request.wall_time is not None:
            raise ValueError("a timer takes a duration, never a wall time or days")
        duration = request.duration_s
        if isinstance(duration, bool) or not isinstance(duration, int):
            raise ValueError(f"a timer needs an integer duration_s: {duration!r}")
        if not 1 <= duration <= MAX_DURATION_S:
            raise ValueError(f"duration_s must be 1 to {MAX_DURATION_S}: {duration}")
        return
    if request.duration_s is not None:
        raise ValueError("an alarm takes a wall time, never a duration")
    wall = request.wall_time
    if not isinstance(wall, time):
        raise ValueError(f"an alarm needs a wall_time: {wall!r}")
    if wall.second or wall.microsecond:
        raise ValueError(f"an alarm wall_time is hours and minutes only: {wall!r}")


def _check_days(days: object) -> None:
    if not isinstance(days, set | frozenset) or not days:
        raise ValueError(f"days must be a non-empty set of weekdays: {days!r}")
    for day in days:
        if isinstance(day, bool) or not isinstance(day, int) or not 0 <= day <= 6:
            raise ValueError(f"a weekday must be an integer from 0 to 6: {day!r}")


def _value_identity(
    kind: str, wall_time: time | None, days: frozenset[int] | None, duration_s: int | None
) -> tuple[AlarmCategory, str]:
    """The identity of an untagged record, from the values that make it distinct."""
    if days is not None:
        assert wall_time is not None
        return "recurring", f"alarm:recurring:{_encode_wall_time(wall_time)}:{_encode_days(days)}"
    if kind == "alarm":
        assert wall_time is not None
        return "one_off", f"alarm:one_off:{_encode_wall_time(wall_time)}"
    return "one_off", f"timer:one_off:{duration_s}"


def _saved_instant(record: AlarmRecord, at: datetime, tz: tzinfo) -> datetime:
    """The next instant a tagged record rings when re-armed as saved."""
    if record.duration_s is not None:
        return at + timedelta(seconds=record.duration_s)
    assert record.wall_time is not None  # `_check_record_shape`: an alarm has a wall time
    hour, minute = record.wall_time.hour, record.wall_time.minute
    if record.days is None:
        # The first instant of the wall time after `at`, today or tomorrow.
        return next_occurrence(RepeatRule("daily", hour, minute), at, tz)
    return next_occurrence(RepeatRule("custom", hour, minute, record.days), at, tz)


def _normalise_ring_on(value: object) -> RingTarget:
    if value == _ALL:
        return _ALL
    if isinstance(value, str) or not isinstance(value, tuple | list | set | frozenset):
        raise ValueError(f"ring_on must be 'all' or a collection of endpoint names: {value!r}")
    names = _check_names(value)
    if not names:
        raise ValueError("ring_on must name at least one endpoint")
    return tuple(sorted(names))


def _check_names(names: Iterable[object]) -> frozenset[str]:
    if isinstance(names, str):
        raise ValueError(f"endpoint names must be a collection, not one string: {names!r}")
    checked: set[str] = set()
    for name in names:
        if not isinstance(name, str) or not name or name != name.strip() or "," in name:
            raise ValueError(f"an endpoint name must be non-empty text without commas: {name!r}")
        if name == _ALL:
            raise ValueError("'all' is a ring target, not an endpoint name")
        checked.add(name)
    return frozenset(checked)


def _encode_ring_on(value: RingTarget) -> str:
    return _ALL if value == _ALL else ",".join(value)


def _decode_ring_on(value: object) -> RingTarget:
    text = _decode_text(value, "ring_on")
    if text == _ALL:
        return _ALL
    names = _decode_names(value, "ring_on")
    if not names:
        raise ValueError("ring_on names no endpoint")
    return tuple(sorted(names))


def _encode_names(names: Iterable[str]) -> str:
    return ",".join(sorted(names))


def _decode_names(value: object, field: str) -> frozenset[str]:
    text = value if isinstance(value, str) else _decode_text(value, field)
    if not text:
        return frozenset()
    parts = text.split(",")
    if parts != sorted(set(parts)) or any(not p or p != p.strip() or p == _ALL for p in parts):
        raise ValueError(f"{field} {text!r} is not a sorted list of endpoint names")
    return frozenset(parts)


def _encode_days(days: Iterable[int]) -> str:
    return ",".join(str(day) for day in sorted(days))


def _decode_days(text: str) -> frozenset[int]:
    parts = text.split(",")
    if not all(part in "0123456" and len(part) == 1 for part in parts):
        raise ValueError(f"days {text!r} is not a list of weekdays 0 to 6")
    days = [int(part) for part in parts]
    if days != sorted(set(days)):
        raise ValueError(f"days {text!r} is not sorted and unique")
    return frozenset(days)


def _encode_wall_time(value: time) -> str:
    return f"{value.hour:02d}:{value.minute:02d}"


def _decode_wall_time(value: object) -> time:
    text = _decode_text(value, "wall_time")
    match = _WALL_TIME.fullmatch(text)
    if match is None:
        raise ValueError(f"wall_time {text!r} is not HH:MM")
    return time(int(match.group(1)), int(match.group(2)))


def _decode_choice(value: object, field: str, choices: frozenset[str]) -> str:
    text = _decode_text(value, field)
    if text not in choices:
        raise ValueError(f"{field} {text!r} is not one of {sorted(choices)}")
    return text


def _decode_count(value: object, field: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} {value!r} is not an integer of at least {minimum}")
    return value


def _decode_late_by(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"late_by_s {value!r} is not a number")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"late_by_s {value!r} is not finite and non-negative")
    return float(value)


def _decode_text(value: object, field: str) -> str:
    if not isinstance(value, bytes):
        raise ValueError(f"{field} {value!r} is not text")
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{field} is not valid UTF-8: {value!r}") from exc


def _decode_time(value: object, field: str) -> datetime:
    text = _decode_text(value, field)
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} {text!r} has no UTC offset")
    try:
        instant = parsed.astimezone(UTC)
    except OverflowError as exc:
        # A time at the edge of the calendar ("0001-01-01T00:00:00+05:00") has no UTC form.
        raise ValueError(f"{field} {text!r} is outside the UTC range") from exc
    if instant > _LATEST_TIME:
        raise ValueError(f"{field} {text!r} is past {_LATEST_TIME.year}")
    return instant


def _encode_time(value: datetime) -> str:
    # Fixed microsecond precision keeps one spelling per instant, so stored times compare as text.
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _to_utc(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be an aware datetime")
    return value.astimezone(UTC)


def _check_source(source: object) -> None:
    if source not in _SOURCES:
        raise ValueError(f"unknown event source {source!r}")


def _acknowledged_outcome(by: object) -> RecordOutcome:
    if by == "dismiss":
        return "dismissed"
    if by == "expiry":
        return "expired"
    raise ValueError(f"unknown acknowledgement {by!r}")


def _detail_reason(detail: object) -> str:
    """The missed reason in an event's detail, or `unknown` when the detail does not hold one."""
    try:
        data = json.loads(_decode_text(detail, "detail"))
    except ValueError:
        return "unknown"
    reason = data.get("reason") if isinstance(data, dict) else None
    return reason if reason in _MISSED_REASONS else "unknown"


def _printable_id(value: object) -> str:
    """The row id for a log line, readable even when the id itself does not decode."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _fingerprint(row: _Row) -> str:
    """A digest of a row as SQL returned it, so an entry can tell when its row has changed.

    It digests the row's `repr`, which is stable for the bytes, numbers and None a row holds,
    including a NaN, which never compares equal to itself.
    """
    return hashlib.sha256(repr(row).encode("utf-8")).hexdigest()
