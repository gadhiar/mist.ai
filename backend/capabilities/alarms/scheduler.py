"""`AlarmScheduler`: the one writer of occurrence transitions, run as a supervised task.

Capabilities and `alarm_action` change alarms only through the scheduler's methods (`schedule`,
`cancel`, `snooze`, `stop_ringing`, `dismiss`), and every write goes through `AlarmRecords`: an
alarm is a record, armed into occurrences, so a daily alarm keeps one record however often it
fires. Each method writes, publishes the new state when the write changed an occurrence, and
calls `wake()`, so the task recomputes its next deadline at once instead of finishing a sleep
that the change made wrong. Method writes and the task's ticks hold one lock, so their writes
never interleave and each published state is read after the write it follows.

The task sleeps in chunks of at most 60 s on the injected `Clock` and recomputes from the wall
clock on every tick, so a clock jump or a host sleep is noticed within 60 s of waking. Each tick:

- fires every due occurrence through `AlarmRecords.fire`. One found more than the grace window
  late (`effective_grace_s`, never under 60 s) is fired as missed, reason `late`, otherwise it
  rings with its `late_by_s`. The first fire of a repeating record's occurrence passes the
  series' next instant, computed by `next_occurrence` over the record's weekday set in the user's
  time zone, so the record gains its next occurrence in the same transaction. A snoozed
  occurrence ringing again is fired from `snoozed`, which counts nothing and arms nothing;
- ends each ringing occurrence past its ring timeout, counted from its stored `fired_at` so the
  timeout survives a restart, through `AlarmRecords.time_out`. That finishes it `timed_out` when
  an endpoint acknowledged playing it, and makes it missed, reason `unheard`, when none did;
- expires every missed occurrence at least `MISSED_ACK_AFTER` old.

Supervision. Each tick contains `_CONTAINED_ERRORS`, per occurrence where it handles one, so an
occurrence whose state cannot be written stays due and is retried without holding back another
one due in the same tick. After a tick with a contained error the task backs off (1 s, doubling,
capped at 60 s) and reports itself `failing`; a clean tick restores `running`. Any other
exception is a bug: it ends the task, and the done-callback logs it and records the task dead.
The done-callback reads the flag `stop()` sets before it cancels, not the cancellation itself,
so a cancel that `stop()` did not ask for is a death too. `stop()` waits at most
`stop_timeout_s` for the task to end, so a write or a sink stuck past a cancel cannot hold up the
backend's shutdown.

Quarantine. `AlarmRecords` skips rows outside the alarm domain on every read. An occurrence whose
handling raises `ValueError` or `ArithmeticError` here is a data error a retry cannot clear, so
it is handed to `AlarmRecords.quarantine`, logged once and skipped rather than marking the tick
failing; every other occurrence carries on.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import math
import sqlite3
from collections.abc import Collection
from dataclasses import dataclass
from datetime import datetime, timedelta, tzinfo
from typing import Literal

from backend.errors import CapabilityStoreError, MistError

from ..clock import Clock
from ..config import AlarmSettings
from .notifier import AlarmNotifier, effective_grace_s
from .records import AlarmRecords, ArmRequest, ArmResult, EventSource, Occurrence
from .timeparse import RepeatRule, next_occurrence

logger = logging.getLogger(__name__)

MISSED_ACK_AFTER = timedelta(hours=24)
"""How long a missed occurrence stays listed before a tick expires it unacknowledged."""

# Failures a tick contains rather than dying on, as the extraction dispatcher's set does: the MIST
# error tree (`AlarmRecords` raises `CapabilityStoreError`), storage, and OS-level I/O. Anything
# else is a bug and ends the task (see `_on_task_done`). No broader catch is added.
_CONTAINED_ERRORS = (MistError, sqlite3.Error, OSError)

# Handling one occurrence contains more than a tick does. The added two are data errors in that
# one occurrence, which a retry cannot clear: `ValueError` covers a payload that fails its
# protocol model (pydantic's `ValidationError`), a date out of range and a repeating record with
# no wall time, and `ArithmeticError` covers `OverflowError` from date arithmetic.
_PER_OCCURRENCE_ERRORS = (*_CONTAINED_ERRORS, ValueError, ArithmeticError)

_TASK_NAME = "alarm-scheduler"
_CHUNK_S = 60.0
"""The longest sleep, and so the longest a clock jump or host sleep can go unnoticed."""
_BACKOFF_INITIAL_S = 1.0
_BACKOFF_CAP_S = 60.0
# Bounds the doubling exponent so a long outage cannot overflow the float.
_MAX_EXPONENT = 30
# `acknowledge_missed_before` expires occurrences missed strictly before its cutoff. Adding the
# stored precision makes the rule "at least `MISSED_ACK_AFTER` old", so the tick that wakes at an
# expiry deadline expires it rather than computing a zero sleep for it again.
_STORED_PRECISION = timedelta(microseconds=1)

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
    """Fires due occurrences, applies the grace rule, ring timeout and missed expiry.

    The records, clock, notifier, alarm settings and the time zone repeats are computed in are all
    injected. The time zone is the user's (`MIST_TIMEZONE`), in which a repeat keeps its wall time.
    """

    def __init__(
        self,
        records: AlarmRecords,
        clock: Clock,
        notifier: AlarmNotifier,
        settings: AlarmSettings,
        tz: tzinfo,
        *,
        stop_timeout_s: float = 5.0,
    ) -> None:
        """Keep the collaborators; no task starts until `start()`.

        Raises:
            ValueError: `stop_timeout_s` is not a finite number above 0.
        """
        if (
            isinstance(stop_timeout_s, bool)
            or not isinstance(stop_timeout_s, int | float)
            or not math.isfinite(stop_timeout_s)
            or stop_timeout_s <= 0
        ):
            raise ValueError(f"stop_timeout_s must be finite and above 0: {stop_timeout_s!r}")
        self._records = records
        self._clock = clock
        self._notifier = notifier
        self._settings = settings
        self._tz = tz
        self._stop_timeout_s = float(stop_timeout_s)
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._stopping = False
        self._dead_reason: str | None = None
        self._failing_reason: str | None = None
        # Set when a state change has not reached the notifier, so the next tick publishes it.
        self._state_dirty = False
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
        """Cancel the task and wait at most `stop_timeout_s` for it to end.

        The stopping flag is set before the cancel, so the done-callback reads the end as orderly
        and marks nothing, however late it comes. A task still running at the bound is logged at
        WARNING and left to end on its own: shutdown goes on rather than wait on a stuck write or
        sink. A task that has already died stays recorded as dead.
        """
        self._stopping = True
        self._running = False
        task = self._task
        if task is None or task.done():
            return
        task.cancel()
        done, _ = await asyncio.wait({task}, timeout=self._stop_timeout_s)
        if not done:
            logger.warning(
                "Alarm scheduler task did not end within %g s of its cancel; shutdown goes on "
                "without it",
                self._stop_timeout_s,
            )

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

        A backoff sleep after a failed tick is not cut short, so repeated changes cannot turn
        failing records into a tight retry loop.
        """
        self._wake.set()

    async def schedule(self, request: ArmRequest, *, source: EventSource) -> ArmResult:
        """Arm `request` through `AlarmRecords.arm` at the clock's now, then publish and wake.

        The records decide by category whether it adds an occurrence, is already set or moves a
        tagged one, so no request creates a second record for an identity.

        Raises:
            ValueError: The request is incoherent or `source` is unknown (`AlarmRecords.arm`).
            LookupError: A re-arm names a tag no record holds.
            CapabilityStoreError: The write failed.
        """
        try:
            async with self._lock:
                result = await self._records.arm(
                    request, at=self._clock.now(), tz=self._tz, source=source
                )
                await self._publish_contained()
        finally:
            self.wake()
        return result

    async def cancel(self, occurrence_ids: Collection[str], *, source: EventSource) -> int:
        """Cancel the occurrences named, in any state, stopping any that is ringing.

        Naming any occurrence of a repeating record cancels its whole series (`AlarmRecords.
        cancel`), so a ringing occurrence of that series is stopped too. The record stays saved.

        Returns:
            The number of occurrences cancelled.

        Raises:
            ValueError: `source` is unknown.
            CapabilityStoreError: The read or the write failed.
        """
        named = set(occurrence_ids)
        try:
            async with self._lock:
                _, live = await self._records.snapshot()
                series = {
                    o.record.record_id
                    for o in live
                    if o.occurrence_id in named and o.record.days is not None
                }
                ringing = [
                    o.occurrence_id
                    for o in live
                    if o.state == "ringing"
                    and (o.occurrence_id in named or o.record.record_id in series)
                ]
                cancelled = await self._records.cancel(
                    occurrence_ids, at=self._clock.now(), source=source
                )
                if cancelled:
                    for occurrence_id in ringing:
                        await self._notify_stopped(occurrence_id)
                    await self._publish_contained()
        finally:
            self.wake()
        return cancelled

    async def snooze(
        self,
        occurrence_id: str,
        minutes: int | None = None,
        *,
        source: EventSource,
        endpoint: str | None,
    ) -> bool:
        """Stop a ringing occurrence and ring it again after `minutes`, or the configured snooze.

        `endpoint` is recorded as the one that snoozed it.

        Returns:
            True when the occurrence was ringing and is now snoozed; False otherwise.

        Raises:
            ValueError: `minutes` is less than 1, or `source` is unknown.
            CapabilityStoreError: The write failed.
        """
        snooze_minutes = self._settings.snooze_minutes if minutes is None else minutes
        if snooze_minutes < 1:
            raise ValueError(f"snooze minutes must be at least 1: {snooze_minutes}")
        try:
            async with self._lock:
                now = self._clock.now()
                snoozed = await self._records.snooze(
                    occurrence_id,
                    until=now + timedelta(minutes=snooze_minutes),
                    at=now,
                    source=source,
                    endpoint=endpoint,
                )
                if snoozed:
                    await self._notify_stopped(occurrence_id)
                    await self._publish_contained()
        finally:
            self.wake()
        return snoozed

    async def stop_ringing(
        self, occurrence_id: str, *, source: EventSource, endpoint: str | None
    ) -> bool:
        """End a ringing or snoozed occurrence for good, recorded as `stopped`.

        The notifier hears `stopped` only for an occurrence that was ringing; a snoozed one was
        already reported stopped when it was snoozed. `endpoint` is recorded as the one that
        stopped it.

        Returns:
            True when the occurrence was ringing or snoozed; False otherwise.

        Raises:
            ValueError: `source` is unknown.
            CapabilityStoreError: The read or the write failed.
        """
        try:
            async with self._lock:
                _, live = await self._records.snapshot()
                was_ringing = any(
                    o.occurrence_id == occurrence_id and o.state == "ringing" for o in live
                )
                stopped = await self._records.finish(
                    occurrence_id,
                    outcome="stopped",
                    at=self._clock.now(),
                    source=source,
                    endpoint=endpoint,
                )
                if stopped:
                    if was_ringing:
                        await self._notify_stopped(occurrence_id)
                    await self._publish_contained()
        finally:
            self.wake()
        return stopped

    async def dismiss(self, occurrence_id: str, *, endpoint: str | None) -> bool:
        """Acknowledge a missed occurrence, recorded as `dismissed`, so it leaves the missed list.

        Returns:
            True when the occurrence was missed; False otherwise, with nothing written.

        Raises:
            CapabilityStoreError: The write failed.
        """
        try:
            async with self._lock:
                dismissed = await self._records.acknowledge(
                    occurrence_id, at=self._clock.now(), by="dismiss"
                )
                if dismissed:
                    logger.info(
                        "Missed alarm %s dismissed from %s", occurrence_id, endpoint or "a turn"
                    )
                    await self._publish_contained()
        finally:
            self.wake()
        return dismissed

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

        A failed read skips the rest of the tick. A failed write for one occurrence is logged
        with its id, and the tick carries on with the next.
        """
        self._wake.clear()
        failure: BaseException | None = None
        async with self._lock:
            now = self._clock.now()
            try:
                changed = False
                for occurrence in await self._records.due(now):
                    try:
                        changed = await self._fire(occurrence, now) or changed
                    except _PER_OCCURRENCE_ERRORS as exc:
                        failure = self._occurrence_failed(occurrence, exc, "fired") or failure
                try:
                    expired = await self._records.acknowledge_missed_before(
                        now - MISSED_ACK_AFTER + _STORED_PRECISION, at=now
                    )
                except _CONTAINED_ERRORS as exc:
                    logger.error("Missed alarms not expired; the next tick retries: %s", exc)
                    failure = exc
                    expired = 0
                if expired:
                    logger.info("%d missed alarm(s) expired unacknowledged", expired)
                    changed = True
                deadlines: list[datetime] = []
                _, live = await self._records.snapshot()
                for occurrence in live:
                    if occurrence.state == "missed":
                        if occurrence.missed_at is not None:
                            deadlines.append(occurrence.missed_at + MISSED_ACK_AFTER)
                        continue
                    if occurrence.state != "ringing":
                        deadlines.append(occurrence.next_fire_at)
                        continue
                    try:
                        ends_at = self._ring_ends_at(occurrence)
                        if ends_at > now:
                            deadlines.append(ends_at)
                            continue
                        changed = await self._time_out(occurrence, now) or changed
                    except _PER_OCCURRENCE_ERRORS as exc:
                        action = "ended at its ring timeout"
                        failure = self._occurrence_failed(occurrence, exc, action) or failure
                if changed:
                    self._state_dirty = True
                if self._state_dirty:
                    await self._publish_state()
            except _CONTAINED_ERRORS as exc:
                logger.error("Alarm scheduler tick skipped: %s", exc, exc_info=True)
                return 0.0, exc
        delay = _CHUNK_S
        if deadlines:
            until_next = (min(deadlines) - self._clock.now()).total_seconds()
            delay = min(delay, max(0.0, until_next))
        return delay, failure

    async def _fire(self, occurrence: Occurrence, now: datetime) -> bool:
        """Ring a due occurrence, or fire it missed under the grace rule; return whether it fired.

        Only a scheduled occurrence of a repeating record passes the series' next instant: a
        snoozed one ringing again is the occurrence already counted, whose next is already armed.
        """
        late_by_s = max(0.0, (now - occurrence.next_fire_at).total_seconds())
        grace_s = effective_grace_s(self._settings.missed_grace_s)
        outcome: Literal["ringing", "missed"] = "missed" if late_by_s > grace_s else "ringing"
        next_fire_at = None
        if occurrence.state == "scheduled" and occurrence.record.days is not None:
            next_fire_at = self._next_after(occurrence, now)
        fired = await self._records.fire(
            occurrence.occurrence_id,
            outcome=outcome,
            at=now,
            late_by_s=late_by_s,
            next_fire_at=next_fire_at,
        )
        if not fired:
            return False
        if outcome == "missed":
            logger.warning(
                "Alarm %s missed: found %.0f s late, past the %.0f s grace window",
                occurrence.occurrence_id,
                late_by_s,
                grace_s,
            )
            return True
        rung = dataclasses.replace(
            occurrence,
            state="ringing",
            fired_at=now,
            late_by_s=late_by_s,
            sent_to=frozenset(),
            delivered_to=frozenset(),
        )
        try:
            await self._notifier.ring(rung, self._settings.ring_timeout_s)
        except _PER_OCCURRENCE_ERRORS as exc:
            # The ringing state is stored and sound, so the published state still shows it and
            # its ring timeout still ends it; only this announcement is lost.
            logger.error(
                "Alarm notifier failed to ring alarm %s: %s",
                occurrence.occurrence_id,
                exc,
                exc_info=exc,
            )
        return True

    def _next_after(self, occurrence: Occurrence, now: datetime) -> datetime:
        """Return the series' first instant after `now`, at the record's wall time and days.

        From `now`, not from the occurrence: one found days late arms the next instant still to
        come rather than another one already past.

        Raises:
            ValueError: The repeating record has no wall time.
        """
        record = occurrence.record
        if record.wall_time is None or record.days is None:
            raise ValueError(f"repeating record {record.record_id} has no wall time or days")
        rule = RepeatRule(
            kind="custom",
            hour=record.wall_time.hour,
            minute=record.wall_time.minute,
            days=record.days,
        )
        return next_occurrence(rule, now, self._tz)

    async def _time_out(self, occurrence: Occurrence, now: datetime) -> bool:
        ended = await self._records.time_out(occurrence.occurrence_id, at=now)
        if ended is None:
            return False
        logger.info(
            "Alarm %s ended %s after its %.0f s ring timeout",
            occurrence.occurrence_id,
            "timed out" if ended == "timed_out" else "missed, unheard,",
            self._settings.ring_timeout_s,
        )
        await self._notify_stopped(occurrence.occurrence_id)
        return True

    def _ring_ends_at(self, occurrence: Occurrence) -> datetime:
        fired_at = (
            occurrence.fired_at if occurrence.fired_at is not None else occurrence.next_fire_at
        )
        return fired_at + timedelta(seconds=self._settings.ring_timeout_s)

    async def _notify_stopped(self, occurrence_id: str) -> None:
        try:
            await self._notifier.stopped(occurrence_id)
        except _CONTAINED_ERRORS as exc:
            # The stored state is already written; the published state that follows carries it.
            logger.error("Alarm notifier failed to report alarm %s stopped: %s", occurrence_id, exc)

    async def _publish_state(self) -> None:
        """Send one snapshot's live occurrences and `rev`; raises a contained error."""
        rev, live = await self._records.snapshot()
        await self._notifier.state_changed(live, rev, self._clock.now())
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

    def _occurrence_failed(
        self, occurrence: Occurrence, exc: BaseException, action: str
    ) -> BaseException | None:
        """Log one occurrence's failed handling; return the error when it should fail the tick.

        A contained error is returned, so the tick reports failing and the occurrence is retried.
        A data error quarantines the occurrence instead and returns None.
        """
        if isinstance(exc, _CONTAINED_ERRORS):
            logger.error(
                "Alarm %s could not be %s, so it is retried next tick: %s",
                occurrence.occurrence_id,
                action,
                exc,
                exc_info=exc,
            )
            return exc
        self._records.quarantine(occurrence.occurrence_id, f"could not be {action}")
        logger.error(
            "Alarm %s quarantined until its row changes: it could not be %s (%s)",
            occurrence.occurrence_id,
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


def _failure_reason(exc: BaseException) -> str:
    """Name where a tick failed and the error's class, leaving its text to the log."""
    if isinstance(exc, sqlite3.Error | CapabilityStoreError):
        return f"alarm records failed ({type(exc).__name__})"
    return f"alarm scheduler tick failed ({type(exc).__name__})"


def _backoff_s(failures: int) -> float:
    """The sleep after `failures` consecutive failed ticks (at least 1): 1, 2, 4, ... 60 s."""
    exponent = min(max(failures - 1, 0), _MAX_EXPONENT)
    return min(_BACKOFF_CAP_S, _BACKOFF_INITIAL_S * 2.0**exponent)
