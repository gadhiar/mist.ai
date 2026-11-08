"""Tests for `AlarmScheduler` on `AlarmRecords`: firing, the grace rule, endings and supervision.

The scheduler runs on a `SteppedClock` over a real `AlarmRecords`. It sleeps in a task of its own,
so `FakeClock.advance` waits only for that inner task, never for the scheduler; a test waits for
the scheduler's next sleep with `SteppedClock.slept`, which wakes on each `sleep` call rather than
polling, so database work in the worker thread completes before the test reads it.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import math
import sqlite3
from collections.abc import AsyncIterator, Callable, Collection
from datetime import UTC, datetime, time, timedelta, timezone, tzinfo

import pytest

from backend.capabilities.alarms import scheduler as scheduler_module
from backend.capabilities.alarms.records import (
    AlarmRecords,
    ArmRequest,
    ArmResult,
    Occurrence,
)
from backend.capabilities.alarms.ringing import DesktopSink, RingDelivery, RingRouter
from backend.capabilities.alarms.scheduler import (
    MISSED_ACK_AFTER,
    AlarmScheduler,
    SchedulerHealth,
)
from backend.capabilities.config import AlarmSettings
from backend.capabilities.db import CapabilitiesDb
from backend.errors import CapabilityStoreError, MistError
from tests.unit.capabilities.fakes import DEFAULT_START, FakeClock, RecordingEventSink

T0 = DEFAULT_START
"""Monday 5 January 2026, 07:00 UTC."""
DAY = timedelta(days=1)
SCHEDULER_LOGGER = "backend.capabilities.alarms.scheduler"
TASK_NAME = "alarm-scheduler"
SETTINGS = AlarmSettings(
    ring_timeout_s=600.0, snooze_minutes=9, missed_grace_s=900.0, fallback_after_s=30.0
)


class SteppedClock(FakeClock):
    """A `FakeClock` whose wall time can jump, and which signals every `sleep` call.

    `jump` moves `now()` without waking any sleeper, as a host sleep or a clock change does.
    """

    def __init__(self) -> None:
        super().__init__()
        self.offset = timedelta(0)
        self._sleep_called = asyncio.Event()
        self.cancelled: list[float] = []
        """Every sleep cancelled before its deadline, by a wake or by `stop`, in order."""

    def now(self) -> datetime:
        return super().now() + self.offset

    def jump(self, seconds: float) -> None:
        self.offset += timedelta(seconds=seconds)

    async def sleep(self, seconds: float) -> None:
        # Set before the sleep registers, which it does before its first await, so a waiter that
        # wakes on this event already sees the sleep recorded and pending.
        self._sleep_called.set()
        try:
            await super().sleep(seconds)
        except asyncio.CancelledError:
            self.cancelled.append(seconds)
            raise

    async def slept(self, count: int) -> None:
        """Wait until `sleep` has been called `count` times.

        The bound is the clock's own `settle_timeout_s`, reached only when the scheduler never
        sleeps again, which is a failure.
        """
        try:
            async with asyncio.timeout(self.settle_timeout_s):
                while len(self.sleeps) < count:
                    self._sleep_called.clear()
                    await self._sleep_called.wait()
        except TimeoutError:
            raise AssertionError(
                f"expected {count} scheduler sleeps, saw {len(self.sleeps)}: {self.sleeps}"
            ) from None


class FlakyRecords(AlarmRecords):
    """A real `AlarmRecords` whose reads and writes can be made to raise.

    `read_error` is raised by `due`. `write_error` is raised by `fire`, `time_out`, `finish`,
    `snooze` and `acknowledge`, for every occurrence, or only for the ids in `fail_ids` when that
    set is not empty. `fire_calls` records each `fire` id and `due_calls` each instant `due` was
    asked about. `on_due` callbacks run one per `due` call, in order, before it reads.
    """

    def __init__(self, db: CapabilitiesDb) -> None:
        super().__init__(db)
        self.on_due: list[Callable[[], None]] = []
        self.read_error: BaseException | None = None
        self.write_error: BaseException | None = None
        self.fail_ids: set[str] = set()
        self.fire_calls: list[str] = []
        self.due_calls: list[datetime] = []
        self.quarantine_calls: list[str] = []

    def _check_write(self, occurrence_id: str) -> None:
        if self.write_error is not None and (not self.fail_ids or occurrence_id in self.fail_ids):
            raise self.write_error

    async def due(self, now: datetime) -> list[Occurrence]:
        self.due_calls.append(now)
        if self.on_due:
            self.on_due.pop(0)()
        if self.read_error is not None:
            raise self.read_error
        return await super().due(now)

    async def fire(self, occurrence_id: str, **kwargs: object) -> bool:  # type: ignore[override]
        self.fire_calls.append(occurrence_id)
        self._check_write(occurrence_id)
        return await super().fire(occurrence_id, **kwargs)  # type: ignore[arg-type]

    async def time_out(self, occurrence_id: str, **kwargs: object):  # type: ignore[override]
        self._check_write(occurrence_id)
        return await super().time_out(occurrence_id, **kwargs)  # type: ignore[arg-type]

    async def finish(self, occurrence_id: str, **kwargs: object) -> bool:  # type: ignore[override]
        self._check_write(occurrence_id)
        return await super().finish(occurrence_id, **kwargs)  # type: ignore[arg-type]

    async def snooze(self, occurrence_id: str, **kwargs: object) -> bool:  # type: ignore[override]
        self._check_write(occurrence_id)
        return await super().snooze(occurrence_id, **kwargs)  # type: ignore[arg-type]

    async def acknowledge(self, occurrence_id: str, **kwargs: object) -> bool:  # type: ignore[override]
        self._check_write(occurrence_id)
        return await super().acknowledge(occurrence_id, **kwargs)  # type: ignore[arg-type]

    def quarantine(self, row_id: str, reason: str) -> None:
        self.quarantine_calls.append(row_id)
        super().quarantine(row_id, reason)


class RecordingNotifier:
    """An `AlarmNotifier` that records every call.

    `ring_errors` maps an occurrence id to the error its `ring` raises after recording it.
    `state_errors` are raised by the next `state_changed` calls, one each, after recording.
    `ring_hold`, when set, makes `ring` wait for it and swallow cancellation until it is set,
    as a sink stuck in I/O that ignores a cancel would.
    """

    def __init__(self) -> None:
        self.rings: list[tuple[Occurrence, float]] = []
        self.stops: list[str] = []
        self.states: list[tuple[list[Occurrence], int, datetime]] = []
        self.ring_errors: dict[str, BaseException] = {}
        self.state_errors: list[BaseException] = []
        self.ring_hold: asyncio.Event | None = None

    async def ring(self, occurrence: Occurrence, ring_timeout_s: float) -> RingDelivery:
        self.rings.append((occurrence, ring_timeout_s))
        if self.ring_hold is not None:
            try:
                await self.ring_hold.wait()
            except asyncio.CancelledError:
                await self.ring_hold.wait()
                raise
        if occurrence.occurrence_id in self.ring_errors:
            raise self.ring_errors[occurrence.occurrence_id]
        return RingDelivery(sent_to=(), fallback=False)

    async def stopped(self, occurrence_id: str) -> None:
        self.stops.append(occurrence_id)

    async def state_changed(self, occurrences: list[Occurrence], rev: int, at: datetime) -> None:
        self.states.append((list(occurrences), rev, at))
        if self.state_errors:
            raise self.state_errors.pop(0)


@pytest.fixture()
def db(tmp_path):
    database = CapabilitiesDb(str(tmp_path / "capabilities.db"))
    yield database
    database.close()


@pytest.fixture()
def records(db):
    store = FlakyRecords(db)
    store.migrate()
    return store


@pytest.fixture()
def clock():
    stepped = SteppedClock()
    yield stepped
    # Every woken sleeper settled: `advance` never waited out its bound on the scheduler.
    assert stepped.stalled == []


@pytest.fixture()
def notifier():
    return RecordingNotifier()


def _scheduler(
    records: AlarmRecords,
    clock: SteppedClock,
    notifier: RecordingNotifier | RingRouter,
    settings: AlarmSettings = SETTINGS,
    tz: tzinfo = UTC,
    **kwargs: float,
) -> AlarmScheduler:
    return AlarmScheduler(records, clock, notifier, settings, tz, **kwargs)  # type: ignore[arg-type]


@contextlib.asynccontextmanager
async def _running(scheduler: AlarmScheduler) -> AsyncIterator[AlarmScheduler]:
    await scheduler.start()
    try:
        yield scheduler
    finally:
        await scheduler.stop()


def alarm(
    fire_at: datetime | None,
    hh: int = 7,
    mm: int = 0,
    *,
    label: str | None = None,
    days: Collection[int] | None = None,
) -> ArmRequest:
    return ArmRequest(
        kind="alarm",
        fire_at=fire_at,
        wall_time=time(hh, mm),
        duration_s=None,
        label=label,
        days=None if days is None else frozenset(days),
    )


def timer(fire_at: datetime, duration_s: int = 600, *, label: str | None = None) -> ArmRequest:
    return ArmRequest(
        kind="timer",
        fire_at=fire_at,
        wall_time=None,
        duration_s=duration_s,
        label=label,
        days=None,
    )


async def _arm(records: AlarmRecords, request: ArmRequest, at: datetime = T0) -> str:
    """Arm directly through the records, as an earlier run of the backend would have."""
    result = await records.arm(request, at=at, tz=UTC, source="fastpath")
    return result.occurrence.occurrence_id


async def _live(records: AlarmRecords) -> dict[str, Occurrence]:
    _, occurrences = await records.snapshot()
    return {o.occurrence_id: o for o in occurrences}


async def _states(records: AlarmRecords) -> dict[str, str]:
    return {key: o.state for key, o in (await _live(records)).items()}


async def _of_state(records: AlarmRecords, state: str) -> list[Occurrence]:
    return [o for o in (await _live(records)).values() if o.state == state]


async def _events(db: CapabilitiesDb) -> list[tuple[str, str | None, str | None]]:
    def read(conn: sqlite3.Connection) -> list[tuple[str, str | None, str | None]]:
        sql = "SELECT kind, occurrence_id, detail FROM alarm_events ORDER BY seq"
        return [tuple(row) for row in conn.execute(sql).fetchall()]

    return await db.run(read)


def _wake_waiters() -> list[asyncio.Task[object]]:
    """The scheduler's pending wake-waiter tasks: one beside a chunk sleep, none beside a backoff."""
    return [t for t in asyncio.all_tasks() if t.get_name() == f"{TASK_NAME}-wake" and not t.done()]


def _scheduler_task() -> asyncio.Task[object]:
    return next(t for t in asyncio.all_tasks() if t.get_name() == TASK_NAME)


async def _ended(task: asyncio.Task[object]) -> None:
    """Wait for the task to end; its done-callback, attached first, has run by the return."""
    await asyncio.wait({task})


def _errors_naming(caplog: pytest.LogCaptureFixture, alarm_id: str) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno == logging.ERROR and alarm_id in r.getMessage()]


class TestFiring:
    @pytest.mark.asyncio
    async def test_a_scheduled_occurrence_fires_at_its_time(self, records, clock, notifier):
        occurrence_id = await _arm(records, timer(T0 + timedelta(seconds=90), 90))

        async with _running(_scheduler(records, clock, notifier)):
            await clock.slept(1)
            assert clock.sleeps == [60.0]
            await clock.advance(60)
            await clock.slept(2)
            assert clock.sleeps == [60.0, 30.0]
            assert notifier.rings == []

            await clock.advance(30)
            await clock.slept(3)

            [(rung, ring_timeout_s)] = notifier.rings
            assert rung.occurrence_id == occurrence_id
            assert rung.state == "ringing"
            assert (rung.fired_at, rung.late_by_s) == (T0 + timedelta(seconds=90), 0.0)
            assert ring_timeout_s == 600.0
            stored = (await _live(records))[occurrence_id]
            assert (stored.state, stored.late_by_s) == ("ringing", 0.0)
            assert stored.record.fire_count == 1

            occurrences, rev, at = notifier.states[-1]
            assert rev == await records.rev()
            assert [(o.occurrence_id, o.state) for o in occurrences] == [(occurrence_id, "ringing")]
            assert at == T0 + timedelta(seconds=90)

    @pytest.mark.asyncio
    async def test_schedule_arms_through_the_records_and_fires(self, records, clock, notifier):
        scheduler = _scheduler(records, clock, notifier)
        async with _running(scheduler):
            await clock.slept(1)
            result = await scheduler.schedule(
                timer(T0 + timedelta(seconds=20), 20, label="tea"), source="fastpath"
            )

            assert isinstance(result, ArmResult)
            occurrence = result.occurrence
            assert (occurrence.record.kind, occurrence.record.label) == ("timer", "tea")
            assert occurrence.state == "scheduled"
            assert notifier.states[-1][1] == await records.rev()
            await clock.slept(2)
            assert clock.sleeps == [60.0, 20.0]
            await clock.advance(20)
            await clock.slept(3)

            assert [o.occurrence_id for o, _ in notifier.rings] == [occurrence.occurrence_id]

    @pytest.mark.asyncio
    async def test_scheduling_the_same_recurring_alarm_twice_keeps_one_record(
        self, records, clock, notifier
    ):
        scheduler = _scheduler(records, clock, notifier)
        request = alarm(T0 + DAY, days=range(7))
        async with _running(scheduler):
            await clock.slept(1)
            first = await scheduler.schedule(request, source="fastpath")
            again = await scheduler.schedule(request, source="model")

        assert (first.already_set, again.already_set) == (False, True)
        assert again.occurrence.occurrence_id == first.occurrence.occurrence_id
        [record] = await records.saved()
        assert record.use_count == 1

    @pytest.mark.asyncio
    async def test_sleep_chunks_are_at_most_60_seconds(self, records, clock, notifier):
        await _arm(records, alarm(T0 + timedelta(hours=3)))

        async with _running(_scheduler(records, clock, notifier)):
            for count in range(1, 5):
                await clock.slept(count)
                await clock.advance(60)
            await clock.slept(5)

        assert clock.sleeps == [60.0] * 5
        assert notifier.rings == []

    @pytest.mark.asyncio
    async def test_a_clock_jump_forward_fires_within_one_chunk(self, records, clock, notifier):
        occurrence_id = await _arm(records, alarm(T0 + timedelta(hours=3)))

        async with _running(_scheduler(records, clock, notifier)):
            await clock.slept(1)
            clock.jump(3 * 3600 - 30)
            await clock.advance(60)
            await clock.slept(2)

            [(rung, _)] = notifier.rings
            assert rung.occurrence_id == occurrence_id
            assert rung.late_by_s == 30.0

    @pytest.mark.asyncio
    async def test_the_ring_timeout_calls_time_out_and_an_unheard_ring_ends_missed(
        self, db, records, clock, notifier
    ):
        settings = AlarmSettings(
            ring_timeout_s=90.0, snooze_minutes=9, missed_grace_s=900.0, fallback_after_s=30.0
        )
        occurrence_id = await _arm(records, alarm(T0))

        async with _running(_scheduler(records, clock, notifier, settings)):
            await clock.slept(1)
            assert [o.occurrence_id for o, _ in notifier.rings] == [occurrence_id]
            assert clock.sleeps == [60.0]
            await clock.advance(60)
            await clock.slept(2)
            assert clock.sleeps == [60.0, 30.0]
            assert notifier.stops == []

            await clock.advance(30)
            await clock.slept(3)

            # No endpoint acknowledged playing it (`delivered_to` is empty), so the timeout
            # records it unheard rather than timed out.
            assert notifier.stops == [occurrence_id]
            stored = (await _live(records))[occurrence_id]
            assert (stored.state, stored.missed_reason) == ("missed", "unheard")
            assert stored.missed_at == T0 + timedelta(seconds=90)
            assert stored.record.last_outcome == "missed"
            assert notifier.states[-1][1] == await records.rev()
            assert [o.state for o in notifier.states[-1][0]] == ["missed"]

        kind, event_id, detail = (await _events(db))[-1]
        assert (kind, event_id) == ("missed", occurrence_id)
        assert '"reason": "unheard"' in (detail or "")

    @pytest.mark.asyncio
    async def test_snooze_reschedules_by_the_configured_minutes_and_keeps_the_count(
        self, db, records, clock, notifier
    ):
        settings = AlarmSettings(
            ring_timeout_s=600.0, snooze_minutes=5, missed_grace_s=900.0, fallback_after_s=30.0
        )
        scheduler = _scheduler(records, clock, notifier, settings)
        occurrence_id = await _arm(records, alarm(T0))

        async with _running(scheduler):
            await clock.slept(1)
            assert await scheduler.snooze(occurrence_id, source="alarm_action", endpoint="desktop")

            snoozed = (await _live(records))[occurrence_id]
            assert (snoozed.state, snoozed.next_fire_at) == ("snoozed", T0 + timedelta(minutes=5))
            assert notifier.stops == [occurrence_id]
            for count in range(2, 7):
                await clock.slept(count)
                await clock.advance(60)
            await clock.slept(7)

            assert [o.occurrence_id for o, _ in notifier.rings] == [occurrence_id] * 2
            assert notifier.rings[-1][0].fired_at == T0 + timedelta(minutes=5)
            assert notifier.rings[-1][0].late_by_s == 0.0
            rerung = (await _live(records))[occurrence_id]
            assert rerung.state == "ringing"
            # The re-ring went through `fire` from snoozed, which counts nothing.
            assert rerung.record.fire_count == 1
            assert records.fire_calls == [occurrence_id, occurrence_id]

        kinds = [kind for kind, _, _ in await _events(db)]
        assert kinds == ["set", "fired", "snoozed", "fired"]

    @pytest.mark.asyncio
    async def test_snooze_takes_explicit_minutes(self, records, clock, notifier):
        scheduler = _scheduler(records, clock, notifier)
        occurrence_id = await _arm(records, alarm(T0))

        async with _running(scheduler):
            await clock.slept(1)
            assert await scheduler.snooze(occurrence_id, 2, source="model", endpoint=None)

        assert (await _live(records))[occurrence_id].next_fire_at == T0 + timedelta(minutes=2)

    @pytest.mark.asyncio
    async def test_snooze_refuses_less_than_a_minute(self, records, clock, notifier):
        scheduler = _scheduler(records, clock, notifier)
        occurrence_id = await _arm(records, alarm(T0))

        async with _running(scheduler):
            await clock.slept(1)
            with pytest.raises(ValueError, match="at least 1"):
                await scheduler.snooze(occurrence_id, 0, source="model", endpoint=None)

        assert await _states(records) == {occurrence_id: "ringing"}

    @pytest.mark.asyncio
    async def test_a_daily_alarm_fired_on_3_days_keeps_one_record(self, records, clock, notifier):
        # 07:01 daily, so each day's tick before it sleeps the last minute and fires on time.
        scheduler = _scheduler(records, clock, notifier)
        first_id = await _arm(records, alarm(T0 + timedelta(minutes=1), 7, 1, days=range(7)))
        sleeps = 1

        async with _running(scheduler):
            for day in range(3):
                await clock.slept(sleeps)
                await clock.advance(60)
                await clock.slept(sleeps + 1)
                [ringing] = await _of_state(records, "ringing")
                assert ringing.fired_at == T0 + day * DAY + timedelta(minutes=1)
                assert ringing.late_by_s == 0.0
                assert await scheduler.stop_ringing(
                    ringing.occurrence_id, source="alarm_action", endpoint="desktop"
                )
                await clock.slept(sleeps + 2)
                # The tick after the stop sleeps a chunk; jump to a minute before the next ring.
                clock.jump((DAY - timedelta(minutes=1)).total_seconds() - 60)
                await clock.advance(60)
                sleeps += 3

            await clock.slept(sleeps)

        [record] = await records.saved()
        assert (record.fire_count, record.use_count) == (3, 1)
        assert record.last_outcome == "stopped"
        [upcoming] = (await _live(records)).values()
        assert upcoming.state == "scheduled"
        assert upcoming.record.record_id == record.record_id
        assert upcoming.next_fire_at == T0 + 3 * DAY + timedelta(minutes=1)
        assert first_id not in await _live(records)
        assert len(notifier.rings) == 3

    @pytest.mark.asyncio
    async def test_a_custom_weekday_set_arms_its_next_day_in_the_users_zone(
        self, records, clock, notifier
    ):
        # 02:00 at UTC-5 is T0, a Monday; the set is Monday and Thursday, so next is Thursday.
        tz = timezone(timedelta(hours=-5))
        scheduler = _scheduler(records, clock, notifier, tz=tz)
        occurrence_id = await _arm(records, alarm(T0, 2, 0, label="gym", days={0, 3}))

        async with _running(scheduler):
            await clock.slept(1)

        [upcoming] = await _of_state(records, "scheduled")
        assert upcoming.occurrence_id != occurrence_id
        assert upcoming.next_fire_at == T0 + 3 * DAY
        assert upcoming.record.record_id == (await _live(records))[occurrence_id].record.record_id

    @pytest.mark.asyncio
    async def test_a_snoozed_repeat_ringing_again_arms_no_second_next_one(
        self, records, clock, notifier
    ):
        scheduler = _scheduler(records, clock, notifier)
        occurrence_id = await _arm(records, alarm(T0, days=range(7)))

        async with _running(scheduler):
            await clock.slept(1)
            [upcoming] = await _of_state(records, "scheduled")
            assert upcoming.next_fire_at == T0 + DAY

            await scheduler.snooze(occurrence_id, 1, source="alarm_action", endpoint=None)
            await clock.slept(2)
            await clock.advance(60)
            await clock.slept(3)

            assert len(notifier.rings) == 2
            assert [o.occurrence_id for o in await _of_state(records, "scheduled")] == [
                upcoming.occurrence_id
            ]
            [record] = await records.saved()
            assert record.fire_count == 1

    @pytest.mark.asyncio
    async def test_a_missed_repeat_arms_the_occurrence_after_now(self, records, clock, notifier):
        # Two days and an hour late: the occurrence after the missed one is already past too,
        # so the next one is computed from now, not from the missed occurrence.
        await _arm(records, alarm(T0 - timedelta(days=2, hours=1), 6, 0, days=range(7)))

        async with _running(_scheduler(records, clock, notifier)):
            await clock.slept(1)

        [upcoming] = await _of_state(records, "scheduled")
        assert upcoming.next_fire_at == datetime(2026, 1, 6, 6, 0, tzinfo=UTC)
        [missed] = await _of_state(records, "missed")
        assert missed.missed_reason == "late"
        assert notifier.rings == []

    @pytest.mark.asyncio
    async def test_a_retried_repeat_does_not_duplicate_its_next_occurrence(
        self, records, clock, notifier
    ):
        occurrence_id = await _arm(records, alarm(T0, days=range(7)))
        records.write_error = sqlite3.OperationalError("database is locked")
        records.fail_ids = {occurrence_id}

        async with _running(_scheduler(records, clock, notifier)):
            await clock.slept(1)
            assert clock.sleeps == [1.0]
            records.write_error = None
            await clock.advance(1)
            await clock.slept(2)

        assert records.fire_calls == [occurrence_id, occurrence_id]
        [upcoming] = await _of_state(records, "scheduled")
        assert upcoming.next_fire_at == T0 + DAY
        assert len(await records.saved()) == 1


class TestGraceRule:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("late_s", "expected"),
        [(300, "ringing"), (900, "ringing"), (901, "missed"), (3600, "missed")],
    )
    async def test_an_occurrence_late_at_startup(self, records, clock, notifier, late_s, expected):
        occurrence_id = await _arm(records, alarm(T0 - timedelta(seconds=late_s)))

        async with _running(_scheduler(records, clock, notifier)):
            await clock.slept(1)

        stored = (await _live(records))[occurrence_id]
        assert stored.state == expected
        assert stored.record.fire_count == 1
        if expected == "ringing":
            assert [(o.occurrence_id, o.late_by_s) for o, _ in notifier.rings] == [
                (occurrence_id, float(late_s))
            ]
            assert notifier.rings[0][0].fired_at == T0
            assert (stored.late_by_s, stored.fired_at) == (float(late_s), T0)
        else:
            assert notifier.rings == []
            assert stored.missed_reason == "late"
            assert [o.state for o in notifier.states[-1][0]] == ["missed"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("jump_s", "expected"), [(5 * 60, "ringing"), (20 * 60, "missed")])
    async def test_an_occurrence_late_after_a_jump(
        self, records, clock, notifier, jump_s, expected
    ):
        occurrence_id = await _arm(records, timer(T0 + timedelta(seconds=30), 30))

        async with _running(_scheduler(records, clock, notifier)):
            await clock.slept(1)
            assert clock.sleeps == [30.0]
            clock.jump(jump_s)
            await clock.advance(30)
            await clock.slept(2)

        assert await _states(records) == {occurrence_id: expected}
        if expected == "ringing":
            assert [o.late_by_s for o, _ in notifier.rings] == [float(jump_s)]
        else:
            assert notifier.rings == []

    @pytest.mark.asyncio
    async def test_a_zero_grace_window_still_rings_an_occurrence_up_to_60_s_late(
        self, records, clock, notifier
    ):
        settings = AlarmSettings(
            ring_timeout_s=600.0, snooze_minutes=9, missed_grace_s=0.0, fallback_after_s=30.0
        )
        on_edge = await _arm(records, timer(T0 - timedelta(seconds=60), 60))
        under = await _arm(records, timer(T0 - timedelta(seconds=45), 45))
        over = await _arm(records, timer(T0 - timedelta(seconds=61), 61))

        async with _running(_scheduler(records, clock, notifier, settings)):
            await clock.slept(1)

        assert await _states(records) == {on_edge: "ringing", under: "ringing", over: "missed"}

    @pytest.mark.asyncio
    async def test_a_snoozed_occurrence_found_past_the_grace_window_is_missed(
        self, records, clock, notifier
    ):
        scheduler = _scheduler(records, clock, notifier)
        occurrence_id = await _arm(records, alarm(T0))

        async with _running(scheduler):
            await clock.slept(1)
            await scheduler.snooze(occurrence_id, 1, source="alarm_action", endpoint=None)
            await clock.slept(2)
            clock.jump(3600)
            await clock.advance(60)
            await clock.slept(3)

        stored = (await _live(records))[occurrence_id]
        assert (stored.state, stored.missed_reason) == ("missed", "late")
        assert stored.record.fire_count == 1
        assert len(notifier.rings) == 1


class TestMissed:
    def test_missed_occurrences_are_acknowledged_after_24_hours(self):
        assert timedelta(hours=24) == MISSED_ACK_AFTER

    @pytest.mark.asyncio
    async def test_dismiss_acknowledges_a_missed_occurrence(self, db, records, clock, notifier):
        occurrence_id = await _arm(records, alarm(T0 - timedelta(hours=1)))
        scheduler = _scheduler(records, clock, notifier)

        async with _running(scheduler):
            await clock.slept(1)
            assert await _states(records) == {occurrence_id: "missed"}

            assert await scheduler.dismiss(occurrence_id, endpoint="desktop") is True

            assert await _live(records) == {}
            assert notifier.states[-1][0] == []
            assert notifier.states[-1][1] == await records.rev()
            await clock.slept(2)

        [record] = await records.saved()
        assert (record.last_outcome, record.active) == ("dismissed", False)
        assert [kind for kind, _, _ in await _events(db)][-1] == "dismissed"

    @pytest.mark.asyncio
    async def test_dismiss_returns_false_for_an_occurrence_that_is_not_missed(
        self, records, clock, notifier
    ):
        ringing_id = await _arm(records, alarm(T0))
        scheduled_id = await _arm(records, timer(T0 + timedelta(hours=1)))
        scheduler = _scheduler(records, clock, notifier)

        async with _running(scheduler):
            await clock.slept(1)
            assert await scheduler.dismiss(ringing_id, endpoint="desktop") is False
            assert await scheduler.dismiss(scheduled_id, endpoint=None) is False
            assert await scheduler.dismiss("no-such-occurrence", endpoint=None) is False

        assert await _states(records) == {ringing_id: "ringing", scheduled_id: "scheduled"}

    @pytest.mark.asyncio
    async def test_each_tick_expires_missed_occurrences_older_than_24_hours(
        self, db, records, clock, notifier
    ):
        occurrence_id = await _arm(records, alarm(T0 - timedelta(hours=1)))

        async with _running(_scheduler(records, clock, notifier)):
            await clock.slept(1)
            [missed] = await _of_state(records, "missed")
            assert missed.missed_at == T0

            # One chunk short of 24 h: still listed.
            clock.jump((MISSED_ACK_AFTER - timedelta(seconds=90)).total_seconds())
            await clock.advance(60)
            await clock.slept(2)
            assert await _states(records) == {occurrence_id: "missed"}
            # The tick sleeps only until the expiry is due.
            assert clock.sleeps[-1] == 30.0

            await clock.advance(30)
            await clock.slept(3)

            assert await _live(records) == {}
            assert notifier.states[-1][0] == []

        [record] = await records.saved()
        assert record.last_outcome == "expired"
        assert [kind for kind, _, _ in await _events(db)][-1] == "expired"


class TestWake:
    @pytest.mark.asyncio
    async def test_schedule_reschedules_without_waiting_out_a_chunk(self, records, clock, notifier):
        scheduler = _scheduler(records, clock, notifier)
        async with _running(scheduler):
            await clock.slept(1)
            assert len(_wake_waiters()) == 1
            await scheduler.schedule(timer(T0 + timedelta(seconds=10), 10), source="fastpath")
            await clock.slept(2)

        assert clock.sleeps == [60.0, 10.0]

    @pytest.mark.asyncio
    async def test_cancel_reschedules_and_deletes(self, records, clock, notifier):
        scheduler = _scheduler(records, clock, notifier)
        occurrence_id = await _arm(records, timer(T0 + timedelta(seconds=30), 30))
        async with _running(scheduler):
            await clock.slept(1)
            assert await scheduler.cancel([occurrence_id], source="model") == 1
            await clock.slept(2)

            assert clock.sleeps == [30.0, 60.0]
            assert await _live(records) == {}
            assert notifier.states[-1][0] == []
            assert notifier.stops == []

        [record] = await records.saved()
        assert record.last_outcome == "cancelled"

    @pytest.mark.asyncio
    async def test_cancelling_a_ringing_occurrence_stops_it(self, records, clock, notifier):
        scheduler = _scheduler(records, clock, notifier)
        occurrence_id = await _arm(records, alarm(T0))
        async with _running(scheduler):
            await clock.slept(1)
            assert await scheduler.cancel([occurrence_id], source="alarm_action") == 1

        assert notifier.stops == [occurrence_id]

    @pytest.mark.asyncio
    async def test_cancelling_the_next_of_a_ringing_series_stops_the_whole_series(
        self, records, clock, notifier
    ):
        scheduler = _scheduler(records, clock, notifier)
        ringing_id = await _arm(records, alarm(T0, days=range(7)))
        async with _running(scheduler):
            await clock.slept(1)
            [upcoming] = await _of_state(records, "scheduled")

            assert await scheduler.cancel([upcoming.occurrence_id], source="model") == 2

            assert notifier.stops == [ringing_id]
            assert await _live(records) == {}
        assert len(await records.saved()) == 1

    @pytest.mark.asyncio
    async def test_stop_ringing_reschedules_and_ends_the_occurrence(self, records, clock, notifier):
        scheduler = _scheduler(records, clock, notifier)
        occurrence_id = await _arm(records, alarm(T0))
        async with _running(scheduler):
            await clock.slept(1)
            assert await scheduler.stop_ringing(
                occurrence_id, source="alarm_action", endpoint="desktop"
            )
            await clock.slept(2)

            assert clock.sleeps == [60.0, 60.0]
            assert await _live(records) == {}
            assert notifier.stops == [occurrence_id]
            assert notifier.states[-1][1] == await records.rev()

        [record] = await records.saved()
        assert record.last_outcome == "stopped"

    @pytest.mark.asyncio
    async def test_snooze_reschedules_without_waiting_out_a_chunk(self, records, clock, notifier):
        scheduler = _scheduler(records, clock, notifier)
        occurrence_id = await _arm(records, alarm(T0))
        async with _running(scheduler):
            await clock.slept(1)
            await scheduler.snooze(occurrence_id, 1, source="alarm_action", endpoint=None)
            await clock.slept(2)

        assert len(clock.sleeps) == 2

    @pytest.mark.asyncio
    async def test_stop_ringing_a_snoozed_occurrence_ends_it_without_a_second_stop(
        self, records, clock, notifier
    ):
        scheduler = _scheduler(records, clock, notifier)
        occurrence_id = await _arm(records, alarm(T0))
        async with _running(scheduler):
            await clock.slept(1)
            await scheduler.snooze(occurrence_id, source="alarm_action", endpoint=None)
            assert await scheduler.stop_ringing(occurrence_id, source="model", endpoint=None)

        assert await _live(records) == {}
        assert notifier.stops == [occurrence_id]

    @pytest.mark.asyncio
    async def test_a_wake_during_a_tick_runs_the_next_tick_without_sleeping(
        self, records, clock, notifier
    ):
        scheduler = _scheduler(records, clock, notifier)
        records.on_due.append(scheduler.wake)

        async with _running(scheduler):
            await clock.slept(1)
            # The wake arrived after the first tick cleared it, so the task went straight into a
            # second tick: no sleep was started for it and then cancelled.
            assert records.due_calls == [T0, T0]
            assert clock.sleeps == [60.0]
            assert clock.cancelled == []

    @pytest.mark.asyncio
    async def test_wake_does_not_cut_a_backoff_sleep_short(self, records, clock, notifier):
        await _arm(records, alarm(T0))
        records.write_error = sqlite3.OperationalError("database is locked")
        scheduler = _scheduler(records, clock, notifier)

        async with _running(scheduler):
            await clock.slept(1)
            # No wake waiter sits beside a backoff sleep, so no wake can end it early. Checked
            # directly because the tick times cannot show it: a wake's early tick would race the
            # `advance` below and read T0 + 1 s either way.
            assert _wake_waiters() == []
            await scheduler.schedule(timer(T0 + timedelta(hours=1)), source="fastpath")
            await clock.advance(1)
            await clock.slept(2)

        assert records.due_calls == [T0, T0 + timedelta(seconds=1)]
        assert clock.sleeps == [1.0, 2.0]

    @pytest.mark.asyncio
    async def test_a_method_that_changes_nothing_returns_false(self, records, clock, notifier):
        scheduler = _scheduler(records, clock, notifier)
        async with _running(scheduler):
            await clock.slept(1)
            assert await scheduler.stop_ringing("none", source="model", endpoint=None) is False
            assert await scheduler.snooze("none", source="model", endpoint=None) is False
            assert await scheduler.cancel(["none"], source="model") == 0
            assert await scheduler.dismiss("none", endpoint=None) is False


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_health_is_stopped_before_start(self, records, clock, notifier):
        scheduler = _scheduler(records, clock, notifier)

        assert scheduler.health() == SchedulerHealth(state="stopped", reason=None)
        assert scheduler.task_dead() is False

    @pytest.mark.asyncio
    async def test_stop_cancels_the_task_cleanly(self, records, clock, notifier, caplog):
        scheduler = _scheduler(records, clock, notifier)
        with caplog.at_level(logging.WARNING, logger=SCHEDULER_LOGGER):
            await scheduler.start()
            await clock.slept(1)
            task = _scheduler_task()
            assert scheduler.health() == SchedulerHealth(state="running", reason=None)

            await scheduler.stop()

        assert task.done()
        assert clock.pending == 0
        assert scheduler.health() == SchedulerHealth(state="stopped", reason=None)
        assert scheduler.task_dead() is False
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []

    @pytest.mark.asyncio
    async def test_stop_waits_at_most_its_bound_then_warns_and_returns(
        self, records, clock, notifier, caplog
    ):
        await _arm(records, alarm(T0))
        notifier.ring_hold = asyncio.Event()
        scheduler = _scheduler(records, clock, notifier, stop_timeout_s=0.05)
        loop = asyncio.get_running_loop()

        with caplog.at_level(logging.WARNING, logger=SCHEDULER_LOGGER):
            await scheduler.start()
            task = _scheduler_task()
            async with asyncio.timeout(2.0):
                while not notifier.rings:
                    await asyncio.sleep(0)
            started = loop.time()
            await scheduler.stop()
            waited = loop.time() - started

        assert 0.05 <= waited < 1.0
        assert not task.done()
        [warning] = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert "0.05" in warning.getMessage()
        assert scheduler.health() == SchedulerHealth(state="stopped", reason=None)

        # The stuck ring lets go; the task then ends as the cancel asked, recorded as orderly.
        notifier.ring_hold.set()
        await _ended(task)
        assert task.cancelled()
        assert scheduler.task_dead() is False

    @pytest.mark.parametrize("bound", [0.0, -1.0, math.nan, math.inf, -math.inf])
    def test_a_stop_bound_that_is_not_finite_and_positive_is_refused(
        self, records, clock, notifier, bound
    ):
        with pytest.raises(ValueError, match="stop_timeout_s"):
            _scheduler(records, clock, notifier, stop_timeout_s=bound)

    @pytest.mark.asyncio
    async def test_start_twice_runs_one_task(self, records, clock, notifier):
        scheduler = _scheduler(records, clock, notifier)
        async with _running(scheduler):
            await scheduler.start()
            await clock.slept(1)
            assert [t.get_name() for t in asyncio.all_tasks()].count(TASK_NAME) == 1


class TestSupervision:
    def test_the_contained_set_is_a_module_constant(self):
        assert (MistError, sqlite3.Error, OSError) == scheduler_module._CONTAINED_ERRORS
        assert "except Exception" not in inspect.getsource(scheduler_module)

    @pytest.mark.asyncio
    async def test_one_occurrence_failing_does_not_hold_back_another(
        self, records, clock, notifier, caplog
    ):
        # A second earlier, so the records return the broken occurrence first.
        broken = await _arm(records, timer(T0 - timedelta(seconds=1), 1))
        healthy = await _arm(records, timer(T0, 2))
        records.write_error = CapabilityStoreError("alarm records cannot write")
        records.fail_ids = {broken}

        with caplog.at_level(logging.ERROR, logger=SCHEDULER_LOGGER):
            async with _running(_scheduler(records, clock, notifier)):
                await clock.slept(1)

        assert [o.occurrence_id for o, _ in notifier.rings] == [healthy]
        assert await _states(records) == {broken: "scheduled", healthy: "ringing"}
        assert any(broken in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_one_ring_timeout_failing_does_not_hold_back_another(
        self, records, clock, notifier
    ):
        settings = AlarmSettings(
            ring_timeout_s=60.0, snooze_minutes=9, missed_grace_s=900.0, fallback_after_s=30.0
        )
        broken = await _arm(records, timer(T0 - timedelta(seconds=1), 1))
        healthy = await _arm(records, timer(T0, 2))

        async with _running(_scheduler(records, clock, notifier, settings)):
            await clock.slept(1)
            records.write_error = sqlite3.OperationalError("database is locked")
            records.fail_ids = {broken}
            await clock.advance(60)
            await clock.slept(2)

            assert await _states(records) == {broken: "ringing", healthy: "missed"}
            assert notifier.stops == [healthy]

    @pytest.mark.asyncio
    async def test_a_failing_notifier_ring_is_logged_and_does_not_fail_the_tick(
        self, records, clock, notifier, caplog
    ):
        occurrence_id = await _arm(records, alarm(T0))
        notifier.ring_errors[occurrence_id] = OSError("room unit unreachable")
        scheduler = _scheduler(records, clock, notifier)

        with caplog.at_level(logging.ERROR, logger=SCHEDULER_LOGGER):
            async with _running(scheduler):
                await clock.slept(1)
                assert scheduler.health() == SchedulerHealth(state="running", reason=None)
                assert clock.sleeps == [60.0]

        assert await _states(records) == {occurrence_id: "ringing"}
        assert len(_errors_naming(caplog, occurrence_id)) == 1

    @pytest.mark.asyncio
    async def test_schedule_returns_when_publishing_fails_and_a_tick_publishes_again(
        self, records, clock, notifier
    ):
        scheduler = _scheduler(records, clock, notifier)
        async with _running(scheduler):
            await clock.slept(1)
            notifier.state_errors.append(OSError("socket closed"))

            result = await scheduler.schedule(timer(T0 + timedelta(seconds=10)), source="model")
            await clock.slept(2)

            assert len(notifier.states) == 2
            assert [o.occurrence_id for o in notifier.states[-1][0]] == [
                result.occurrence.occurrence_id
            ]
            assert notifier.states[-1][1] == await records.rev()

    @pytest.mark.asyncio
    async def test_a_failed_read_skips_the_tick_and_the_task_carries_on(
        self, records, clock, notifier
    ):
        occurrence_id = await _arm(records, alarm(T0))
        records.read_error = sqlite3.OperationalError("disk I/O error at /srv/private/x.db")
        scheduler = _scheduler(records, clock, notifier)

        async with _running(scheduler):
            await clock.slept(1)
            assert notifier.rings == []
            assert clock.sleeps == [1.0]
            assert not _scheduler_task().done()
            health = scheduler.health()
            assert health.state == "failing"
            assert "alarm records" in (health.reason or "")
            assert "OperationalError" in (health.reason or "")
            assert "disk I/O" not in (health.reason or "")
            assert "/srv/private" not in (health.reason or "")

            records.read_error = None
            await clock.advance(1)
            await clock.slept(2)

            assert [o.occurrence_id for o, _ in notifier.rings] == [occurrence_id]
            assert scheduler.health() == SchedulerHealth(state="running", reason=None)

    @pytest.mark.asyncio
    async def test_unwritable_records_back_off_then_ring_late_on_recovery(
        self, records, clock, notifier
    ):
        occurrence_id = await _arm(records, alarm(T0))
        records.write_error = sqlite3.OperationalError("attempt to write a readonly database")
        scheduler = _scheduler(records, clock, notifier)
        backoff = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0, 60.0]

        async with _running(scheduler):
            for count in range(1, len(backoff)):
                await clock.slept(count)
                await clock.advance(clock.sleeps[-1])
            await clock.slept(len(backoff))

            assert clock.sleeps == backoff
            health = scheduler.health()
            assert health.state == "failing"
            assert "alarm records" in (health.reason or "")
            assert "readonly" not in (health.reason or "")
            assert not scheduler.task_dead()
            assert await _states(records) == {occurrence_id: "scheduled"}

            records.write_error = None
            await clock.advance(60)
            await clock.slept(len(backoff) + 1)

            late_by_s = sum(backoff)
            assert [(o.occurrence_id, o.late_by_s) for o, _ in notifier.rings] == [
                (occurrence_id, late_by_s)
            ]
            assert scheduler.health() == SchedulerHealth(state="running", reason=None)
            assert clock.sleeps[-1] == 60.0
            assert 0 not in clock.sleeps

    @pytest.mark.asyncio
    async def test_a_clean_tick_resets_the_backoff(self, records, clock, notifier):
        await _arm(records, alarm(T0))
        records.write_error = sqlite3.OperationalError("database is locked")

        async with _running(_scheduler(records, clock, notifier)):
            for count in range(1, 3):
                await clock.slept(count)
                await clock.advance(clock.sleeps[-1])
            await clock.slept(3)
            assert clock.sleeps == [1.0, 2.0, 4.0]

            records.write_error = None
            await clock.advance(4)
            await clock.slept(4)
            records.read_error = sqlite3.OperationalError("database is locked")
            await clock.advance(60)
            await clock.slept(5)

        assert clock.sleeps == [1.0, 2.0, 4.0, 60.0, 1.0]

    @pytest.mark.asyncio
    async def test_an_unexpected_error_ends_the_task_and_is_recorded(
        self, records, clock, notifier, caplog
    ):
        records.read_error = RuntimeError("internal detail 7731")
        scheduler = _scheduler(records, clock, notifier)

        with caplog.at_level(logging.ERROR, logger=SCHEDULER_LOGGER):
            await scheduler.start()
            await _ended(_scheduler_task())

        health = scheduler.health()
        assert health == SchedulerHealth(
            state="dead", reason="alarm scheduler task died: RuntimeError"
        )
        assert scheduler.task_dead() is True
        [record] = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert "Traceback" in (record.exc_text or "")
        assert "internal detail 7731" in (record.exc_text or "")

        await scheduler.stop()

        assert scheduler.health() == health
        assert scheduler.task_dead() is True

    @pytest.mark.asyncio
    async def test_a_dead_scheduler_is_not_restarted(self, records, clock, notifier):
        records.read_error = RuntimeError("bug")
        scheduler = _scheduler(records, clock, notifier)
        await scheduler.start()
        await _ended(_scheduler_task())

        await scheduler.start()

        assert TASK_NAME not in [t.get_name() for t in asyncio.all_tasks()]
        assert scheduler.health().state == "dead"

    @pytest.mark.asyncio
    async def test_a_cancel_without_stop_is_a_death(self, records, clock, notifier, caplog):
        scheduler = _scheduler(records, clock, notifier)

        with caplog.at_level(logging.ERROR, logger=SCHEDULER_LOGGER):
            await scheduler.start()
            await clock.slept(1)
            task = _scheduler_task()
            task.cancel()
            await _ended(task)

        health = scheduler.health()
        assert health.state == "dead"
        assert "CancelledError" in (health.reason or "")
        assert scheduler.task_dead() is True
        assert [r.levelno for r in caplog.records] == [logging.ERROR]

        await scheduler.stop()

        assert scheduler.health() == health

    @pytest.mark.asyncio
    async def test_a_restarted_scheduler_still_records_a_death(self, records, clock, notifier):
        scheduler = _scheduler(records, clock, notifier)
        await scheduler.start()
        await clock.slept(1)
        await scheduler.stop()

        # The second run is not stopping, so its death is not read as an orderly end.
        records.read_error = RuntimeError("bug")
        await scheduler.start()
        await _ended(_scheduler_task())

        assert scheduler.health() == SchedulerHealth(
            state="dead", reason="alarm scheduler task died: RuntimeError"
        )
        assert scheduler.task_dead() is True

    @pytest.mark.asyncio
    async def test_a_failed_tick_publish_reports_the_error_class_not_its_text(
        self, records, clock, notifier
    ):
        occurrence_id = await _arm(records, alarm(T0))
        notifier.state_errors.append(OSError("socket closed by peer"))
        scheduler = _scheduler(records, clock, notifier)

        async with _running(scheduler):
            await clock.slept(1)

            assert clock.sleeps == [1.0]
            assert scheduler.health() == SchedulerHealth(
                state="failing", reason="alarm scheduler tick failed (OSError)"
            )

            await clock.advance(1)
            await clock.slept(2)

            # The ringing state was stored, so the retried publish carries it.
            assert [(o.occurrence_id, o.state) for o in notifier.states[-1][0]] == [
                (occurrence_id, "ringing")
            ]
            assert scheduler.health() == SchedulerHealth(state="running", reason=None)

        assert [o.occurrence_id for o, _ in notifier.rings] == [occurrence_id]


class TestQuarantine:
    """An occurrence whose handling raises a data error costs only itself."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error", [OverflowError("date value out of range"), ValueError("bad value")]
    )
    async def test_an_occurrence_whose_handling_raises_a_data_error_is_quarantined(
        self, records, clock, notifier, caplog, error
    ):
        broken = await _arm(records, timer(T0 - timedelta(seconds=1), 1))
        healthy = await _arm(records, timer(T0, 2))
        records.write_error = error
        records.fail_ids = {broken}
        scheduler = _scheduler(records, clock, notifier)

        with caplog.at_level(logging.ERROR, logger=SCHEDULER_LOGGER):
            async with _running(scheduler):
                await clock.slept(1)
                await clock.advance(60)
                await clock.slept(2)

                assert scheduler.health() == SchedulerHealth(state="running", reason=None)
                assert clock.sleeps == [60.0, 60.0]

        assert [o.occurrence_id for o, _ in notifier.rings] == [healthy]
        assert records.fire_calls.count(broken) == 1
        assert records.quarantine_calls == [broken]
        assert broken in records.quarantined()
        assert len(_errors_naming(caplog, broken)) == 1

    @pytest.mark.asyncio
    async def test_a_notifier_ring_raising_value_error_costs_only_that_occurrence(
        self, records, clock, notifier
    ):
        broken = await _arm(records, timer(T0 - timedelta(seconds=1), 1))
        healthy = await _arm(records, timer(T0, 2))
        notifier.ring_errors[broken] = ValueError("payload out of range")
        scheduler = _scheduler(records, clock, notifier)

        async with _running(scheduler):
            await clock.slept(1)

            assert scheduler.health() == SchedulerHealth(state="running", reason=None)
            assert sorted(o.occurrence_id for o, _ in notifier.rings) == sorted([broken, healthy])
            assert (await _states(records))[healthy] == "ringing"
            # Its ringing state is stored and sound, so it is not set aside: its ring timeout
            # still ends it.
            assert (await _states(records))[broken] == "ringing"
            assert records.quarantine_calls == []


SHORT_RING = AlarmSettings(
    ring_timeout_s=60.0, snooze_minutes=5, missed_grace_s=900.0, fallback_after_s=30.0
)


def _desktop_router(
    records: AlarmRecords,
    clock: SteppedClock,
    *,
    present: bool,
    settings: AlarmSettings = SHORT_RING,
) -> tuple[RingRouter, RecordingEventSink]:
    events = RecordingEventSink()
    sink = DesktopSink(events, lambda: present)
    return RingRouter([sink], records, clock, settings, UTC), events


async def _last_event(db: CapabilitiesDb, occurrence_id: str) -> tuple[str, dict[str, object]]:
    kind, _, detail = [e for e in await _events(db) if e[1] == occurrence_id][-1]
    return kind, json.loads(detail or "{}")


class TestDelivery:
    """At the ring timeout, an acknowledged ring times out and an unacknowledged one is missed."""

    @pytest.mark.asyncio
    async def test_with_no_sink_present_the_ring_ends_missed_with_nothing_sent(
        self, db, records, clock
    ):
        router, events = _desktop_router(records, clock, present=False)
        occurrence_id = await _arm(records, alarm(T0))

        async with _running(_scheduler(records, clock, router, SHORT_RING)):
            await clock.slept(1)
            assert (await _live(records))[occurrence_id].sent_to == frozenset()
            await clock.advance(60)
            await clock.slept(2)

        assert events.of_type("alarm_fired") == []
        stored = (await _live(records))[occurrence_id]
        assert (stored.state, stored.missed_reason) == ("missed", "unheard")
        assert await _last_event(db, occurrence_id) == (
            "missed",
            {"reason": "unheard", "sent_to": []},
        )

    @pytest.mark.asyncio
    async def test_a_ring_sent_but_never_acknowledged_ends_missed_with_sent_to(
        self, db, records, clock
    ):
        router, events = _desktop_router(records, clock, present=True)
        occurrence_id = await _arm(records, alarm(T0))

        async with _running(_scheduler(records, clock, router, SHORT_RING)):
            await clock.slept(1)
            assert [p["alarm_id"] for p in events.of_type("alarm_fired")] == [occurrence_id]
            assert (await _live(records))[occurrence_id].sent_to == frozenset({"desktop"})
            await clock.advance(60)
            await clock.slept(2)

        stored = (await _live(records))[occurrence_id]
        assert (stored.state, stored.missed_reason) == ("missed", "unheard")
        assert await _last_event(db, occurrence_id) == (
            "missed",
            {"reason": "unheard", "sent_to": ["desktop"]},
        )

    @pytest.mark.asyncio
    async def test_an_acknowledged_ring_ends_timed_out(self, db, records, clock):
        router, _ = _desktop_router(records, clock, present=True)
        occurrence_id = await _arm(records, alarm(T0))
        scheduler = _scheduler(records, clock, router, SHORT_RING)

        async with _running(scheduler):
            await clock.slept(1)
            assert await scheduler.ring_acknowledged(occurrence_id, endpoint="desktop") is True
            assert (await _live(records))[occurrence_id].delivered_to == frozenset({"desktop"})
            # The acknowledgement woke the task, which cut its sleep short.
            await clock.slept(2)
            assert clock.cancelled == [60.0]
            await clock.advance(60)
            await clock.slept(3)

        assert await _live(records) == {}
        [record] = await records.saved()
        assert record.last_outcome == "timed_out"
        assert await _last_event(db, occurrence_id) == ("timed_out", {"delivered_to": ["desktop"]})

    @pytest.mark.asyncio
    async def test_acknowledging_an_occurrence_that_is_not_ringing_returns_false(
        self, records, clock, notifier
    ):
        occurrence_id = await _arm(records, alarm(T0 + timedelta(hours=1)))
        scheduler = _scheduler(records, clock, notifier)

        assert await scheduler.ring_acknowledged(occurrence_id, endpoint="desktop") is False
        assert notifier.states == []


class TestReport:
    @pytest.mark.asyncio
    async def test_a_rang_report_on_a_ringing_occurrence_is_the_acknowledgement(
        self, db, records, clock
    ):
        router, _ = _desktop_router(records, clock, present=True)
        occurrence_id = await _arm(records, alarm(T0))
        scheduler = _scheduler(records, clock, router, SHORT_RING)

        async with _running(scheduler):
            await clock.slept(1)
            result = await scheduler.report(
                occurrence_id,
                endpoint="desktop",
                outcome="rang",
                at=T0,
                rang_at=T0,
                snoozed_until=None,
            )
            assert result == "applied"
            assert (await _live(records))[occurrence_id].delivered_to == frozenset({"desktop"})
            await clock.slept(2)
            assert clock.cancelled == [60.0]
            await clock.advance(60)
            await clock.slept(3)

        assert await _live(records) == {}
        assert (await _last_event(db, occurrence_id))[0] == "timed_out"

    @pytest.mark.asyncio
    async def test_a_report_on_a_due_repeat_fires_it_arms_the_next_and_rings_the_others(
        self, records, clock, notifier
    ):
        occurrence_id = await _arm(records, alarm(T0, days=range(7)))
        scheduler = _scheduler(records, clock, notifier)

        result = await scheduler.report(
            occurrence_id,
            endpoint="desktop",
            outcome="rang",
            at=T0,
            rang_at=T0,
            snoozed_until=None,
        )

        assert result == "applied"
        live = await _live(records)
        fired = live.pop(occurrence_id)
        assert (fired.state, fired.fired_at, fired.delivered_to) == (
            "ringing",
            T0,
            frozenset({"desktop"}),
        )
        [upcoming] = live.values()
        assert (upcoming.state, upcoming.next_fire_at) == ("scheduled", T0 + DAY)
        assert [(o.occurrence_id, o.fired_at) for o, _ in notifier.rings] == [(occurrence_id, T0)]
        assert notifier.states[-1][1] == await records.rev()

    @pytest.mark.asyncio
    async def test_a_report_on_an_occurrence_not_yet_due_is_not_due(self, records, clock, notifier):
        occurrence_id = await _arm(records, alarm(T0 + timedelta(hours=1)))
        scheduler = _scheduler(records, clock, notifier)

        result = await scheduler.report(
            occurrence_id,
            endpoint="desktop",
            outcome="rang",
            at=T0,
            rang_at=T0,
            snoozed_until=None,
        )

        assert result == "not_due"
        assert await _states(records) == {occurrence_id: "scheduled"}
        assert notifier.rings == []

    @pytest.mark.asyncio
    async def test_a_stopped_report_on_a_ringing_occurrence_stops_it(
        self, records, clock, notifier
    ):
        occurrence_id = await _arm(records, alarm(T0))
        scheduler = _scheduler(records, clock, notifier)

        async with _running(scheduler):
            await clock.slept(1)
            result = await scheduler.report(
                occurrence_id,
                endpoint="desktop",
                outcome="stopped",
                at=T0,
                rang_at=T0,
                snoozed_until=None,
            )

            assert result == "applied"
            assert notifier.stops == [occurrence_id]
            assert notifier.states[-1][0] == []

        [record] = await records.saved()
        assert record.last_outcome == "stopped"

    @pytest.mark.parametrize(
        ("rang_after_s", "result", "states"),
        [(61, "late", {"missed"}), (60, "applied", set())],
        ids=["past-the-window", "inside-the-window"],
    )
    @pytest.mark.asyncio
    async def test_a_report_on_a_missed_occurrence_is_bound_by_the_effective_window(
        self, db, records, clock, notifier, rang_after_s, result, states
    ):
        # `missed_grace_s` 0 has the effective window of 60 s that `alarm_state` sends.
        settings = AlarmSettings(
            ring_timeout_s=600.0, snooze_minutes=9, missed_grace_s=0.0, fallback_after_s=30.0
        )
        fire_at = T0 - timedelta(hours=2)
        occurrence_id = await _arm(records, alarm(fire_at, 5, 0))
        scheduler = _scheduler(records, clock, notifier, settings)

        async with _running(scheduler):
            await clock.slept(1)
            assert (await _live(records))[occurrence_id].missed_reason == "late"
            rang_at = fire_at + timedelta(seconds=rang_after_s)

            answer = await scheduler.report(
                occurrence_id,
                endpoint="desktop",
                outcome="rang",
                at=rang_at,
                rang_at=rang_at,
                snoozed_until=None,
            )

        assert answer == result
        assert {o.state for o in (await _live(records)).values()} == states
        kinds = [kind for kind, event_id, _ in await _events(db) if event_id == occurrence_id]
        assert "reported" in kinds
        assert ("timed_out" in kinds) is (result == "applied")

    @pytest.mark.asyncio
    async def test_a_snoozed_ring_acknowledged_on_its_second_ring_ends_timed_out(
        self, db, records, clock
    ):
        router, events = _desktop_router(records, clock, present=True)
        occurrence_id = await _arm(records, alarm(T0))
        scheduler = _scheduler(records, clock, router, SHORT_RING)

        async with _running(scheduler):
            await clock.slept(1)
            assert await scheduler.snooze(occurrence_id, source="alarm_action", endpoint="desktop")
            for count in range(2, 7):
                await clock.slept(count)
                await clock.advance(60)
            await clock.slept(7)

            second = T0 + timedelta(minutes=5)
            rung = (await _live(records))[occurrence_id]
            assert (rung.state, rung.fired_at, rung.delivered_to) == (
                "ringing",
                second,
                frozenset(),
            )
            assert len(events.of_type("alarm_fired")) == 2
            result = await scheduler.report(
                occurrence_id,
                endpoint="desktop",
                outcome="rang",
                at=second,
                rang_at=second,
                snoozed_until=None,
            )
            assert result == "applied"
            await clock.slept(8)
            await clock.advance(60)
            await clock.slept(9)

        assert await _live(records) == {}
        kinds = [kind for kind, event_id, _ in await _events(db) if event_id == occurrence_id]
        assert kinds[-1] == "timed_out"
        assert "missed" not in kinds
