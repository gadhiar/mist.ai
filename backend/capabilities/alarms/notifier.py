"""How the alarm scheduler tells the outside world that an alarm rang, stopped or changed.

The scheduler depends on the `AlarmNotifier` protocol, not on a transport, so a room unit or a
TTS sink can be added beside the WebSocket sink without touching the scheduler. Notifiers are
told, never asked: the alarm store stays the source of truth, which is why `alarm_state` is
built from stored alarms and carries the store's `rev`.

`alarm_state_payload` is the one builder of `alarm_state`, shared by `WsAlarmNotifier` and the
on-connect snapshot, so the snapshot and the broadcast cannot drift apart.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Protocol

from backend.errors import MistError

from ..events import (
    AlarmEntry,
    AlarmFiredEvent,
    AlarmStateEvent,
    EventSink,
    MissedEntry,
    RingingEntry,
)
from .store import Alarm

logger = logging.getLogger(__name__)

# Failures one notifier in a `CompositeNotifier` contains rather than passing on: the MIST error
# tree, storage and OS-level I/O (a closed socket is an `OSError`). Anything else is a bug and
# propagates. No broader catch is added.
_CONTAINED_ERRORS = (MistError, sqlite3.Error, OSError)


class AlarmNotifier(Protocol):
    """Where the scheduler reports alarm events; every method is called on the main loop.

    The scheduler awaits these methods while it holds its lock, so an implementation must not
    call back into the scheduler (`schedule`, `cancel`, `snooze`, `stop_ringing`): those take the
    same lock, which is not reentrant, and the call would wait forever.
    """

    async def ring(self, alarm: Alarm, ring_timeout_s: float) -> None:
        """Announce that `alarm`, now stored as ringing, has started ringing.

        `alarm.fired_at` and `alarm.late_by_s` are set. It stops by itself after
        `ring_timeout_s` unless it is stopped or snoozed first.
        """
        ...

    async def stopped(self, alarm_id: str) -> None:
        """Announce that the alarm `alarm_id` has stopped ringing (stopped, snoozed, timed out or
        cancelled).
        """
        ...

    async def state_changed(self, alarms: Sequence[Alarm], rev: int, at: datetime) -> None:
        """Announce the stored alarms as of revision `rev`, read at `at`."""
        ...


def alarm_state_payload(alarms: Sequence[Alarm], rev: int, now: datetime) -> dict[str, object]:
    """Build the `alarm_state` WS payload from stored alarms, through `AlarmStateEvent`.

    Alarms in state `done` are left out, because they will not fire again and the store keeps
    them. `ringing` lists each ringing alarm with its `late_by_s`, so a client that connects after
    the `alarm_fired` broadcast still sees how late it rang; `missed` lists each missed alarm.
    Alarms keep the order they are given in, every time is converted to UTC, and a repeat is
    named by its kind (`"none"` for none).

    Raises:
        pydantic.ValidationError: A value falls outside the protocol, such as a negative `rev`.
    """
    live = [a for a in alarms if a.state != "done"]
    event = AlarmStateEvent(
        type="alarm_state",
        rev=rev,
        alarms=[
            AlarmEntry(
                alarm_id=a.alarm_id,
                kind=a.kind,
                label=a.label,
                state=a.state,
                next_fire_at=_utc(a.next_fire_at),
                repeat="none" if a.repeat is None else a.repeat.kind,
            )
            for a in live
        ],
        ringing=[
            RingingEntry(alarm_id=a.alarm_id, late_by_s=a.late_by_s or 0.0)
            for a in live
            if a.state == "ringing"
        ],
        missed=[
            MissedEntry(alarm_id=a.alarm_id, label=a.label, scheduled_for=_utc(a.scheduled_for))
            for a in live
            if a.state == "missed"
        ],
        at=_utc(now),
    )
    return event.to_payload()


class WsAlarmNotifier:
    """Sends alarm events to WebSocket clients through the host's event sink."""

    def __init__(self, event_sink: EventSink) -> None:
        """Keep the sink that broadcasts to connected clients."""
        self._sink = event_sink

    async def ring(self, alarm: Alarm, ring_timeout_s: float) -> None:
        """Emit `alarm_fired` for a ringing alarm, built through `AlarmFiredEvent`."""
        fired_at = alarm.fired_at if alarm.fired_at is not None else alarm.next_fire_at
        event = AlarmFiredEvent(
            type="alarm_fired",
            alarm_id=alarm.alarm_id,
            kind=alarm.kind,
            label=alarm.label,
            scheduled_for=_utc(alarm.scheduled_for),
            fired_at=_utc(fired_at),
            late_by_s=alarm.late_by_s or 0.0,
            ring_timeout_s=ring_timeout_s,
        )
        self._sink.emit(event.to_payload())

    async def stopped(self, alarm_id: str) -> None:
        """Emit nothing: protocol 1.4.0 has no stop event, and the `alarm_state` that follows
        every stop carries the alarm's new state.
        """

    async def state_changed(self, alarms: Sequence[Alarm], rev: int, at: datetime) -> None:
        """Emit `alarm_state`, exactly as `alarm_state_payload` builds it."""
        self._sink.emit(alarm_state_payload(alarms, rev, at))


class CompositeNotifier:
    """Passes every event to each of several notifiers, in order.

    One notifier failing with a contained error is logged at ERROR, naming its class, and the
    rest are still called, so a broken room unit cannot silence the WebSocket clients.
    """

    def __init__(self, notifiers: Sequence[AlarmNotifier]) -> None:
        """Keep the notifiers, in the order they are called."""
        self._notifiers = tuple(notifiers)

    async def ring(self, alarm: Alarm, ring_timeout_s: float) -> None:
        """Call `ring` on every notifier."""
        for notifier in self._notifiers:
            try:
                await notifier.ring(alarm, ring_timeout_s)
            except _CONTAINED_ERRORS as exc:
                _log_failure(notifier, "ring", exc)

    async def stopped(self, alarm_id: str) -> None:
        """Call `stopped` on every notifier."""
        for notifier in self._notifiers:
            try:
                await notifier.stopped(alarm_id)
            except _CONTAINED_ERRORS as exc:
                _log_failure(notifier, "stopped", exc)

    async def state_changed(self, alarms: Sequence[Alarm], rev: int, at: datetime) -> None:
        """Call `state_changed` on every notifier."""
        for notifier in self._notifiers:
            try:
                await notifier.state_changed(alarms, rev, at)
            except _CONTAINED_ERRORS as exc:
                _log_failure(notifier, "state_changed", exc)


def _log_failure(notifier: AlarmNotifier, method: str, exc: BaseException) -> None:
    logger.error(
        "Alarm notifier %s failed in %s: %s", type(notifier).__name__, method, exc, exc_info=exc
    )


def _utc(value: datetime) -> datetime:
    return value.astimezone(UTC)
