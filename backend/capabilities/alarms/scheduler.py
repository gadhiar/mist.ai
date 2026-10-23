"""`AlarmScheduler`: the one writer of alarm state transitions, run as a supervised task.

Capabilities and `alarm_action` change alarms only through the scheduler's methods (`schedule`,
`cancel`, `snooze`, `stop_ringing`). Each method writes through the store, publishes the new
state when the write changed an alarm, and calls `wake()`, so the task recomputes its next
deadline at once instead of finishing a sleep that the change made wrong. Method writes and the task's ticks hold one lock, so their
store writes never interleave and each published state is read after the write it follows.

The task sleeps in chunks of at most 60 s on the injected `Clock` and recomputes from the wall
clock on every tick, so a clock jump or a host sleep is noticed within 60 s of waking. An alarm
fired more than 60 s after its time follows the grace rule: within the grace window it rings with
its `late_by_s`, beyond it it becomes `missed`. A ringing alarm stops by itself after the ring
timeout, counted from its stored `fired_at`, so the timeout survives a restart.

Supervision. Each tick contains `_CONTAINED_ERRORS`, per alarm where it handles one, so an alarm
whose state cannot be written stays due and is retried without holding back another one due in
the same tick. After a tick with a contained error the task backs off (1 s, doubling, capped at
60 s) and reports itself `failing`; a clean tick restores `running`. Any other exception is a bug:
it ends the task, and the done-callback logs it and records the task dead. The done-callback reads
the flag `stop()` sets before it cancels, not the cancellation itself, so a cancel that `stop()`
did not ask for is a death too.

Quarantine. Every alarm the scheduler reads is checked against the alarm domain before it acts on
it or publishes it, and an alarm whose handling raises `ValueError` or `ArithmeticError` is set
aside the same way. Such an alarm is a data error, which a retry cannot clear, so it is logged
once and skipped rather than marking the tick failing; every other alarm carries on. It is keyed
by its whole decoded content, so a row that is edited or replaced is checked again.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import math
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, tzinfo
from typing import Literal

from backend.errors import CapabilityStoreError, MistError

from ..clock import Clock
from ..config import AlarmSettings
from .notifier import AlarmNotifier
from .store import Alarm, AlarmKind, AlarmStore
from .timeparse import RepeatRule, next_occurrence

logger = logging.getLogger(__name__)

# Failures a tick contains rather than dying on, as the extraction dispatcher's set does: the MIST
# error tree (the alarm store raises `CapabilityStoreError`), storage, and OS-level I/O. Anything
# else is a bug and ends the task (see `_on_task_done`). No broader catch is added.
_CONTAINED_ERRORS = (MistError, sqlite3.Error, OSError)

# Handling one alarm contains more than a tick does. The added two are data errors in that one
# alarm, which a retry cannot clear: `ValueError` covers a payload that fails its protocol model
# (pydantic's `ValidationError`) and a date out of range, and `ArithmeticError` covers
# `OverflowError` from date arithmetic. They quarantine the alarm rather than failing the tick.
_PER_ALARM_ERRORS = (*_CONTAINED_ERRORS, ValueError, ArithmeticError)

# The latest stored time the scheduler acts on. The furthest it adds to a stored time is the ring
# timeout, added to `fired_at`; this leaves 999 years below `datetime.max` for that sum, so no
# real alarm and no plausible timeout overflows. A later time is a damaged row, not an alarm.
_LATEST_TIME = datetime(9000, 1, 1, tzinfo=UTC)
_KINDS = frozenset({"alarm", "timer"})
_STATES = frozenset({"scheduled", "ringing", "snoozed", "done", "missed"})
_REPEAT_KINDS = frozenset({"daily", "weekdays", "weekends"})

_TASK_NAME = "alarm-scheduler"
_CHUNK_S = 60.0
"""The longest sleep, and so the longest a clock jump or host sleep can go unnoticed."""
_LATE_S = 60.0
"""An alarm fired more than this late follows the grace rule."""
_BACKOFF_INITIAL_S = 1.0
_BACKOFF_CAP_S = 60.0
# Bounds the doubling exponent so a long outage cannot overflow the float.
_MAX_EXPONENT = 30

SchedulerState = Literal["stopped", "running", "failing", "dead"]


@dataclass(frozen=True, slots=True)
class SchedulerHealth:
    """The scheduler task's health, read from memory.

    Attributes:
        state: `stopped` before `start()` and after `stop()`; `running`; `failing` while the
            last tick hit a contained error; `dead` once the task ended other than by `stop()`.
        reason: Why the state is `failing` or `dead`, naming the error's class, never its text;
            None otherwise.
    """

    state: SchedulerState
    reason: str | None


class AlarmScheduler:
    """Fires due alarms, applies the grace rule and ring timeout, and writes every transition.

    The store, clock, notifier, alarm settings and the time zone repeats are computed in are all
    injected. The time zone is the user's (`MIST_TIMEZONE`), in which a repeat keeps its wall time.
    """

    def __init__(
        self,
        store: AlarmStore,
        clock: Clock,
        notifier: AlarmNotifier,
        settings: AlarmSettings,
        tz: tzinfo,
    ) -> None:
        """Keep the collaborators; no task starts until `start()`."""
        self._store = store
        self._clock = clock
        self._notifier = notifier
        self._settings = settings
        self._tz = tz
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._stopping = False
        self._dead_reason: str | None = None
        self._failing_reason: str | None = None
        # Set when a state change has not reached the notifier, so the next tick publishes it.
        self._state_dirty = False
        # Alarms set aside for the life of the process, keyed by their whole decoded content.
        self._quarantined: set[Alarm] = set()
        self._wake = asyncio.Event()
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Create the task and attach its done-callback; returns at once.

        A running scheduler is left as it is, and a dead one is not restarted: a backend restart
        does that.
        """
        if self._dead_reason is not None or (self._task is not None and not self._task.done()):
            return
        self._stopping = False
        self._failing_reason = None
        self._running = True
        self._task = asyncio.create_task(self._run(), name=_TASK_NAME)
        self._task.add_done_callback(self._on_task_done)

    async def stop(self) -> None:
        """Cancel the task and wait for it to end.

        The stopping flag is set before the cancel, so the done-callback reads the end as orderly
        and marks nothing. A task that has already died stays recorded as dead.
        """
        self._stopping = True
        self._running = False
        task = self._task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.wait({task})

    def health(self) -> SchedulerHealth:
        """Return the task's health; `dead` outranks every other state."""
        if self._dead_reason is not None:
            return SchedulerHealth(state="dead", reason=self._dead_reason)
        if not self._running:
            return SchedulerHealth(state="stopped", reason=None)
        if self._failing_reason is not None:
            return SchedulerHealth(state="failing", reason=self._failing_reason)
        return SchedulerHealth(state="running", reason=None)

    def task_dead(self) -> bool:
        """Return True once the task has ended other than by `stop()`."""
        return self._dead_reason is not None

    def wake(self) -> None:
        """Make the task recompute its next deadline now rather than after its current sleep.

        A backoff sleep after a failed tick is not cut short, so repeated changes cannot turn a
        failing store into a tight retry loop.
        """
        self._wake.set()

    async def schedule(
        self,
        kind: AlarmKind,
        fire_at: datetime,
        *,
        label: str | None = None,
        repeat: RepeatRule | None = None,
    ) -> Alarm:
        """Store a new alarm or timer that fires at `fire_at`, then publish and wake.

        Raises:
            ValueError: `kind` is unknown or `fire_at` is naive.
            CapabilityStoreError: The write failed.
        """
        try:
            async with self._lock:
                alarm = await self._store.create(kind, fire_at, label=label, repeat=repeat)
                await self._publish_contained()
        finally:
            self.wake()
        return alarm

    async def cancel(self, alarm_id: str) -> bool:
        """Delete an alarm in any state, stopping it first when it is ringing.

        Returns:
            True when an alarm was deleted, False when none has that id.

        Raises:
            CapabilityStoreError: The read or the delete failed.
        """
        try:
            async with self._lock:
                ringing = {a.alarm_id for a in await self._store.list(("ringing",))}
                deleted = await self._store.cancel(alarm_id)
                if deleted:
                    if alarm_id in ringing:
                        await self._notify_stopped(alarm_id)
                    await self._publish_contained()
        finally:
            self.wake()
        return deleted

    async def snooze(self, alarm_id: str, minutes: int | None = None) -> bool:
        """Stop a ringing alarm and ring it again after `minutes`, or the configured snooze.

        Returns:
            True when the alarm was ringing and is now snoozed; False otherwise.

        Raises:
            ValueError: `minutes` is less than 1.
            CapabilityStoreError: The write failed.
        """
        snooze_minutes = self._settings.snooze_minutes if minutes is None else minutes
        if snooze_minutes < 1:
            raise ValueError(f"snooze minutes must be at least 1: {snooze_minutes}")
        try:
            async with self._lock:
                until = self._clock.now() + timedelta(minutes=snooze_minutes)
                snoozed = await self._store.snooze(alarm_id, until)
                if snoozed:
                    await self._notify_stopped(alarm_id)
                    await self._publish_contained()
        finally:
            self.wake()
        return snoozed

    async def stop_ringing(self, alarm_id: str) -> bool:
        """Stop a ringing or snoozed alarm for good, moving it to `done`.

        The notifier hears `stopped` only for an alarm that was ringing; a snoozed one was
        already reported stopped when it was snoozed.

        Returns:
            True when the alarm was ringing or snoozed; False otherwise.

        Raises:
            CapabilityStoreError: The read or the write failed.
        """
        try:
            async with self._lock:
                ringing = {a.alarm_id for a in await self._store.list(("ringing",))}
                stopped = await self._store.mark_done(alarm_id)
                if stopped:
                    if alarm_id in ringing:
                        await self._notify_stopped(alarm_id)
                    await self._publish_contained()
        finally:
            self.wake()
        return stopped

    async def _run(self) -> None:
        failures = 0
        while True:
            delay, failure = await self._tick()
            if failure is None:
                failures = 0
                self._failing_reason = None
                await self._sleep(delay, interruptible=True)
                continue
            failures += 1
            self._failing_reason = _failure_reason(failure)
            backoff_s = _backoff_s(failures)
            logger.warning(
                "Alarm scheduler tick failed (%d in a row): %s; next tick in %.0f s",
                failures,
                self._failing_reason,
                backoff_s,
            )
            # Outside the contained `try` in `_tick`: `TimeoutError` is an `OSError`, so a failure
            # of the sleep itself must never read as a failed tick.
            await self._sleep(backoff_s, interruptible=False)

    async def _tick(self) -> tuple[float, BaseException | None]:
        """Handle everything due now; return the next sleep and the tick's contained error.

        A failed read skips the rest of the tick. A failed write for one alarm is logged with its
        id, and the tick carries on with the next alarm. Only alarms that pass `_usable` are
        acted on or published.
        """
        self._wake.clear()
        failure: BaseException | None = None
        async with self._lock:
            now = self._clock.now()
            try:
                changed = False
                for alarm in self._usable(await self._store.due(now)):
                    try:
                        changed = await self._fire(alarm, now) or changed
                    except _PER_ALARM_ERRORS as exc:
                        failure = self._alarm_failed(alarm, exc, "fired") or failure
                deadlines: list[datetime] = []
                for alarm in self._usable(await self._store.list(("ringing",))):
                    try:
                        ends_at = self._ring_ends_at(alarm)
                        if ends_at > now:
                            deadlines.append(ends_at)
                            continue
                        changed = await self._time_out(alarm) or changed
                    except _PER_ALARM_ERRORS as exc:
                        stop_failure = self._alarm_failed(alarm, exc, "stopped at its ring timeout")
                        failure = stop_failure or failure
                if changed:
                    self._state_dirty = True
                if self._state_dirty:
                    await self._publish_state()
                pending = self._usable(await self._store.list(("scheduled", "snoozed")))
            except _CONTAINED_ERRORS as exc:
                logger.error("Alarm scheduler tick skipped: %s", exc, exc_info=True)
                return 0.0, exc
        deadlines.extend(a.next_fire_at for a in pending)
        delay = _CHUNK_S
        if deadlines:
            until_next = (min(deadlines) - self._clock.now()).total_seconds()
            delay = min(delay, max(0.0, until_next))
        return delay, failure

    async def _fire(self, alarm: Alarm, now: datetime) -> bool:
        """Ring a due alarm, or mark it missed under the grace rule; return whether it changed.

        A repeating alarm's next occurrence is created on its first firing only: a snoozed alarm
        ringing again is the same occurrence.
        """
        late_by_s = max(0.0, (now - alarm.next_fire_at).total_seconds())
        changed = False
        if alarm.state == "scheduled" and alarm.repeat is not None:
            changed = await self._schedule_next(alarm, alarm.repeat, now)
        if late_by_s > _LATE_S and late_by_s > self._settings.missed_grace_s:
            if await self._store.mark_missed(alarm.alarm_id):
                logger.warning(
                    "Alarm %s missed: found %.0f s late, past the %.0f s grace window",
                    alarm.alarm_id,
                    late_by_s,
                    self._settings.missed_grace_s,
                )
                return True
            return changed
        if not await self._store.mark_ringing(alarm.alarm_id, now, late_by_s):
            return changed
        rung = dataclasses.replace(alarm, state="ringing", fired_at=now, late_by_s=late_by_s)
        try:
            await self._notifier.ring(rung, self._settings.ring_timeout_s)
        except _CONTAINED_ERRORS as exc:
            # The ringing state is stored, so the next published state still shows it.
            logger.error("Alarm notifier failed to ring alarm %s: %s", alarm.alarm_id, exc)
        return True

    async def _schedule_next(self, alarm: Alarm, rule: RepeatRule, now: datetime) -> bool:
        """Create the occurrence after `now` of a repeating alarm, unless it already exists.

        It is created before the firing alarm's own transition, so a series is never lost to a
        transition that succeeds after a create that fails. A transition that fails leaves the
        alarm due, and the retry finds the occurrence already stored rather than adding a second.
        """
        next_at = next_occurrence(rule, now, self._tz)
        for other in self._usable(await self._store.list(("scheduled",))):
            if (
                other.alarm_id != alarm.alarm_id
                and other.scheduled_for == next_at
                and (other.kind, other.label, other.repeat) == (alarm.kind, alarm.label, rule)
            ):
                return False
        await self._store.create(alarm.kind, next_at, label=alarm.label, repeat=rule)
        return True

    async def _time_out(self, alarm: Alarm) -> bool:
        if not await self._store.mark_done(alarm.alarm_id):
            return False
        logger.info(
            "Alarm %s stopped after its %.0f s ring timeout",
            alarm.alarm_id,
            self._settings.ring_timeout_s,
        )
        await self._notify_stopped(alarm.alarm_id)
        return True

    def _ring_ends_at(self, alarm: Alarm) -> datetime:
        fired_at = alarm.fired_at if alarm.fired_at is not None else alarm.next_fire_at
        return fired_at + timedelta(seconds=self._settings.ring_timeout_s)

    async def _notify_stopped(self, alarm_id: str) -> None:
        try:
            await self._notifier.stopped(alarm_id)
        except _CONTAINED_ERRORS as exc:
            # The stored state is already written; the published state that follows carries it.
            logger.error("Alarm notifier failed to report alarm %s stopped: %s", alarm_id, exc)

    async def _publish_state(self) -> None:
        """Send the usable stored alarms and `rev` to the notifier; raises a contained error."""
        alarms = self._usable(await self._store.list())
        rev = await self._store.rev()
        await self._notifier.state_changed(alarms, rev, self._clock.now())
        self._state_dirty = False

    async def _publish_contained(self) -> None:
        """Publish after a method's write, leaving a failure to the next tick to retry.

        The write has already succeeded, so its caller gets its result rather than an error.
        """
        self._state_dirty = True
        try:
            await self._publish_state()
        except _CONTAINED_ERRORS as exc:
            logger.error("Alarm state not published; the next tick retries: %s", exc)

    def _usable(self, alarms: Iterable[Alarm]) -> list[Alarm]:
        """Return the alarms the scheduler may act on, quarantining any outside the domain."""
        usable: list[Alarm] = []
        for alarm in alarms:
            if alarm in self._quarantined:
                continue
            field = _out_of_domain_field(alarm)
            if field is None:
                usable.append(alarm)
                continue
            self._quarantined.add(alarm)
            logger.error(
                "Alarm %s quarantined until its row changes: field %s is outside the alarm "
                "domain",
                alarm.alarm_id,
                field,
            )
        return usable

    def _alarm_failed(self, alarm: Alarm, exc: BaseException, action: str) -> BaseException | None:
        """Log one alarm's failed handling; return the error when it should fail the tick.

        A contained error is returned, so the tick reports failing and the alarm is retried. A
        data error quarantines the alarm instead and returns None.
        """
        if isinstance(exc, _CONTAINED_ERRORS):
            logger.error(
                "Alarm %s could not be %s, so it is retried next tick: %s",
                alarm.alarm_id,
                action,
                exc,
                exc_info=exc,
            )
            return exc
        self._quarantined.add(alarm)
        logger.error(
            "Alarm %s quarantined until its row changes: it could not be %s (%s)",
            alarm.alarm_id,
            action,
            type(exc).__name__,
            exc_info=exc,
        )
        return None

    async def _sleep(self, seconds: float, *, interruptible: bool) -> None:
        """Sleep on the clock for `seconds`, in a task of its own, cut short by `wake()` when
        `interruptible`.

        Every sleep takes the same path, so the task that calls `Clock.sleep` is always the
        inner one and ends when the sleep does.
        """
        if seconds <= 0 or (interruptible and self._wake.is_set()):
            return
        sleeper = asyncio.create_task(self._clock.sleep(seconds), name=f"{_TASK_NAME}-sleep")
        waiters = {sleeper}
        if interruptible:
            waiters.add(asyncio.create_task(self._wake.wait(), name=f"{_TASK_NAME}-wake"))
        try:
            await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for waiter in waiters:
                waiter.cancel()
            await asyncio.wait(waiters)
        if not sleeper.cancelled():
            # Re-raises a failure of the clock itself, which is a bug and ends the task.
            sleeper.result()

    def _on_task_done(self, task: asyncio.Task[None]) -> None:
        """Record how the task ended.

        An end while stopping is orderly: an exception raised while closing is logged at WARNING
        and marks nothing. Any other end is a death, logged at ERROR: an exception (with its
        traceback), a return, or a cancellation `stop()` did not ask for.
        """
        if self._stopping:
            if not task.cancelled():
                exc = task.exception()
                if exc is not None:
                    logger.warning("Alarm scheduler raised while stopping: %s", exc, exc_info=exc)
            return
        if task.cancelled():
            reason = "alarm scheduler task was cancelled without stop() (CancelledError)"
            logger.error("%s; alarms do not fire until the backend restarts", reason)
        else:
            exc = task.exception()
            if exc is not None:
                reason = f"alarm scheduler task died: {type(exc).__name__}"
                logger.error("Alarm scheduler task died: %s", exc, exc_info=exc)
            else:
                reason = "alarm scheduler task returned, which it never does by design"
                logger.error("%s", reason)
        self._dead_reason = reason


def _out_of_domain_field(alarm: Alarm) -> str | None:
    """Name the first field of `alarm` outside the alarm domain, or None when every field is in.

    The store decodes types; this checks values the store's own writes never produce, such as a
    negative `late_by_s` or a time so late that the ring timeout cannot be added to it.
    """
    if alarm.kind not in _KINDS:
        return "kind"
    if alarm.state not in _STATES:
        return "state"
    if alarm.repeat is not None and (
        not isinstance(alarm.repeat, RepeatRule) or alarm.repeat.kind not in _REPEAT_KINDS
    ):
        return "repeat"
    late_by_s = alarm.late_by_s
    if late_by_s is not None and (
        isinstance(late_by_s, bool)
        or not isinstance(late_by_s, int | float)
        or not math.isfinite(late_by_s)
        or late_by_s < 0
    ):
        return "late_by_s"
    times = (
        ("scheduled_for", alarm.scheduled_for),
        ("next_fire_at", alarm.next_fire_at),
        ("fired_at", alarm.fired_at),
    )
    for name, value in times:
        if value is None and name == "fired_at":
            continue
        if not isinstance(value, datetime) or value.utcoffset() is None or value > _LATEST_TIME:
            return name
    return None


def _failure_reason(exc: BaseException) -> str:
    """Name where a tick failed and the error's class, leaving its text to the log."""
    if isinstance(exc, sqlite3.Error | CapabilityStoreError):
        return f"alarm store failed ({type(exc).__name__})"
    return f"alarm scheduler tick failed ({type(exc).__name__})"


def _backoff_s(failures: int) -> float:
    """The sleep after `failures` consecutive failed ticks (at least 1): 1, 2, 4, ... 60 s."""
    exponent = min(max(failures - 1, 0), _MAX_EXPONENT)
    return min(_BACKOFF_CAP_S, _BACKOFF_INITIAL_S * 2.0**exponent)
