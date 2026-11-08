"""How the alarm scheduler tells the outside world that an occurrence rang, stopped or changed.

The scheduler depends on the `AlarmNotifier` protocol, not on a transport, so a room unit or a
TTS sink can be added beside the WebSocket sink without touching the scheduler. Notifiers are
told, never asked: `AlarmRecords` stays the source of truth, which is why `alarm_state` is built
from the live occurrences of one snapshot and carries that snapshot's `rev`.

`alarm_state_payload` is the one builder of `alarm_state`, shared by `WsAlarmNotifier` and the
on-connect snapshot, so the snapshot and the broadcast cannot drift apart. It sends the settings
an endpoint rings by (the ring timeout, the snooze, the effective grace window, the fallback
delay and the time zone), so an endpoint that loses the backend can still ring and time out on
its own rules.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime, tzinfo
from typing import Literal, Protocol

from backend.errors import MistError

from ..config import AlarmSettings
from ..events import (
    AlarmEntry,
    AlarmFiredEvent,
    AlarmStateEvent,
    EventSink,
    MissedEntry,
    RingingEntry,
)
from .records import Occurrence, RingTarget

logger = logging.getLogger(__name__)

# Failures one notifier in a `CompositeNotifier` contains rather than passing on: the MIST error
# tree, storage and OS-level I/O (a closed socket is an `OSError`). Anything else is a bug and
# propagates. No broader catch is added.
_CONTAINED_ERRORS = (MistError, sqlite3.Error, OSError)

GRACE_FLOOR_S = 60.0
"""The shortest grace window: an occurrence found at most this late always rings."""

_PRESETS: dict[frozenset[int], Literal["daily", "weekdays", "weekends"]] = {
    frozenset(range(7)): "daily",
    frozenset(range(5)): "weekdays",
    frozenset({5, 6}): "weekends",
}


def effective_grace_s(missed_grace_s: float) -> float:
    """Return the grace window in force: `missed_grace_s`, but never under `GRACE_FLOOR_S`.

    The scheduler's grace rule and the window `alarm_state` sends are this one value, so the
    backend and an endpoint ringing alone agree on when an occurrence is too late to ring.
    """
    return max(GRACE_FLOOR_S, missed_grace_s)


class AlarmNotifier(Protocol):
    """Where the scheduler reports occurrence events; every method is called on the main loop.

    The scheduler awaits these methods while it holds its lock, so an implementation must not
    call back into the scheduler (`schedule`, `cancel`, `snooze`, `stop_ringing`, `dismiss`):
    those take the same lock, which is not reentrant, and the call would wait forever.
    """

    async def ring(self, occurrence: Occurrence, ring_timeout_s: float) -> None:
        """Announce that `occurrence`, now stored as ringing, has started ringing.

        `occurrence.fired_at` and `occurrence.late_by_s` are set. The scheduler ends it after
        `ring_timeout_s` unless it is stopped or snoozed first.
        """
        ...

    async def stopped(self, occurrence_id: str) -> None:
        """Announce that the occurrence `occurrence_id` has stopped ringing (stopped, snoozed,
        timed out or cancelled).
        """
        ...

    async def state_changed(
        self, occurrences: Sequence[Occurrence], rev: int, at: datetime
    ) -> None:
        """Announce the live occurrences as of revision `rev`, read at `at`."""
        ...


def alarm_state_payload(
    occurrences: Sequence[Occurrence],
    rev: int,
    now: datetime,
    *,
    settings: AlarmSettings,
    tz: tzinfo,
    fallback_after_s: float,
) -> dict[str, object]:
    """Build the `alarm_state` WS payload from live occurrences, through `AlarmStateEvent`.

    `ringing` lists each ringing occurrence with its `late_by_s`, so a client that connects after
    the `alarm_fired` broadcast still sees how late it rang; `missed` lists each missed one with
    its reason. Occurrences keep the order they are given in and every time is converted to UTC.
    `missed_grace_s` is the effective window (`effective_grace_s`), and `tz` is named by its
    IANA key where it has one.

    `fallback_after_s` is a parameter, not a setting, until the ringing router reads it from
    configuration.

    Raises:
        pydantic.ValidationError: A value falls outside the protocol, such as a negative `rev`.
    """
    event = AlarmStateEvent(
        type="alarm_state",
        rev=rev,
        alarms=[_entry(o) for o in occurrences],
        ringing=[
            RingingEntry(alarm_id=o.occurrence_id, late_by_s=o.late_by_s or 0.0)
            for o in occurrences
            if o.state == "ringing"
        ],
        missed=[
            MissedEntry(
                alarm_id=o.occurrence_id,
                record_id=o.record.record_id,
                label=o.record.label,
                scheduled_for=_utc(o.scheduled_for),
                reason=o.missed_reason,  # type: ignore[arg-type]  # set on every missed row
            )
            for o in occurrences
            if o.state == "missed"
        ],
        fallback_after_s=fallback_after_s,
        missed_grace_s=effective_grace_s(settings.missed_grace_s),
        ring_timeout_s=settings.ring_timeout_s,
        snooze_minutes=settings.snooze_minutes,
        tz=_tz_name(tz, now),
        at=_utc(now),
    )
    return event.to_payload()


class WsAlarmNotifier:
    """Sends alarm events to WebSocket clients through the host's event sink."""

    def __init__(
        self,
        event_sink: EventSink,
        settings: AlarmSettings,
        tz: tzinfo,
        *,
        fallback_after_s: float,
    ) -> None:
        """Keep the sink that broadcasts to connected clients and what `alarm_state` sends."""
        self._sink = event_sink
        self._settings = settings
        self._tz = tz
        self._fallback_after_s = fallback_after_s

    async def ring(self, occurrence: Occurrence, ring_timeout_s: float) -> None:
        """Emit `alarm_fired` for a ringing occurrence, built through `AlarmFiredEvent`.

        `fallback` is always False here: this sink rings every connected client, so no named
        endpoint can be absent from it.
        """
        record = occurrence.record
        fired_at = (
            occurrence.fired_at if occurrence.fired_at is not None else occurrence.next_fire_at
        )
        event = AlarmFiredEvent(
            type="alarm_fired",
            alarm_id=occurrence.occurrence_id,
            record_id=record.record_id,
            kind=record.kind,
            category=record.category,
            label=record.label,
            scheduled_for=_utc(occurrence.scheduled_for),
            fired_at=_utc(fired_at),
            late_by_s=occurrence.late_by_s or 0.0,
            ring_timeout_s=ring_timeout_s,
            ring_on=_wire_target(occurrence.ring_on),
            fallback=False,
        )
        self._sink.emit(event.to_payload())

    async def stopped(self, occurrence_id: str) -> None:
        """Emit nothing: protocol 1.4.0 has no stop event, and the `alarm_state` that follows
        every stop carries the occurrence's new state.
        """

    async def state_changed(
        self, occurrences: Sequence[Occurrence], rev: int, at: datetime
    ) -> None:
        """Emit `alarm_state`, exactly as `alarm_state_payload` builds it."""
        self._sink.emit(
            alarm_state_payload(
                occurrences,
                rev,
                at,
                settings=self._settings,
                tz=self._tz,
                fallback_after_s=self._fallback_after_s,
            )
        )


class CompositeNotifier:
    """Passes every event to each of several notifiers, in order.

    One notifier failing with a contained error is logged at ERROR, naming its class, and the
    rest are still called, so a broken room unit cannot silence the WebSocket clients.
    """

    def __init__(self, notifiers: Sequence[AlarmNotifier]) -> None:
        """Keep the notifiers, in the order they are called."""
        self._notifiers = tuple(notifiers)

    async def ring(self, occurrence: Occurrence, ring_timeout_s: float) -> None:
        """Call `ring` on every notifier."""
        for notifier in self._notifiers:
            try:
                await notifier.ring(occurrence, ring_timeout_s)
            except _CONTAINED_ERRORS as exc:
                _log_failure(notifier, "ring", exc)

    async def stopped(self, occurrence_id: str) -> None:
        """Call `stopped` on every notifier."""
        for notifier in self._notifiers:
            try:
                await notifier.stopped(occurrence_id)
            except _CONTAINED_ERRORS as exc:
                _log_failure(notifier, "stopped", exc)

    async def state_changed(
        self, occurrences: Sequence[Occurrence], rev: int, at: datetime
    ) -> None:
        """Call `state_changed` on every notifier."""
        for notifier in self._notifiers:
            try:
                await notifier.state_changed(occurrences, rev, at)
            except _CONTAINED_ERRORS as exc:
                _log_failure(notifier, "state_changed", exc)


def _entry(occurrence: Occurrence) -> AlarmEntry:
    record = occurrence.record
    days = record.days
    return AlarmEntry(
        alarm_id=occurrence.occurrence_id,
        record_id=record.record_id,
        kind=record.kind,
        category=record.category,
        label=record.label,
        state=occurrence.state,
        next_fire_at=_utc(occurrence.next_fire_at),
        wall_time=None if record.wall_time is None else record.wall_time.strftime("%H:%M"),
        repeat="none" if days is None else _PRESETS.get(days, "custom"),
        days=None if days is None else sorted(days),
        duration_s=record.duration_s,
        ring_on=_wire_target(occurrence.ring_on),
    )


def _wire_target(target: RingTarget) -> Literal["all"] | list[str]:
    return "all" if target == "all" else list(target)


def _tz_name(tz: tzinfo, now: datetime) -> str:
    """Name `tz` by its IANA key (`zoneinfo.ZoneInfo.key`), else by its name at `now`."""
    key = getattr(tz, "key", None)
    if isinstance(key, str) and key:
        return key
    return tz.tzname(now.astimezone(tz)) or str(tz)


def _log_failure(notifier: AlarmNotifier, method: str, exc: BaseException) -> None:
    logger.error(
        "Alarm notifier %s failed in %s: %s", type(notifier).__name__, method, exc, exc_info=exc
    )


def _utc(value: datetime) -> datetime:
    return value.astimezone(UTC)
