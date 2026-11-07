"""Tests for `AlarmScheduler`: firing, chunked sleeping, the grace rule, and supervision.

The scheduler runs on a `SteppedClock` over a real `AlarmStore`. It sleeps in a task of its own,
so `FakeClock.advance` waits only for that inner task, never for the scheduler; a test waits for
the scheduler's next sleep with `SteppedClock.slept`, which wakes on each `sleep` call rather than
polling, so store work in the database's worker thread completes before the test reads it.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import sqlite3
import uuid
from collections.abc import AsyncIterator, Callable, Collection
from datetime import UTC, datetime, timedelta, timezone, tzinfo

import pytest

from backend.capabilities.alarms import scheduler as scheduler_module
from backend.capabilities.alarms.scheduler import AlarmScheduler, SchedulerHealth
from backend.capabilities.alarms.store import Alarm, AlarmState, AlarmStore
from backend.capabilities.alarms.timeparse import RepeatRule
from backend.capabilities.config import AlarmSettings
from backend.capabilities.db import CapabilitiesDb
from backend.errors import CapabilityStoreError, MistError
from tests.unit.capabilities.fakes import DEFAULT_START, FakeClock

T0 = DEFAULT_START
SCHEDULER_LOGGER = "backend.capabilities.alarms.scheduler"
TASK_NAME = "alarm-scheduler"
SETTINGS = AlarmSettings(ring_timeout_s=600.0, snooze_minutes=9, missed_grace_s=900.0)
FAR_FUTURE = datetime(9999, 12, 31, tzinfo=UTC)


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


class FlakyStore(AlarmStore):
    """A real `AlarmStore` whose reads and writes can be made to raise.

    `read_error` is raised by `due`. `write_error` is raised by every state transition, for
    every alarm, or only for the ids in `fail_ids` when that set is not empty. `ringing_calls`
    records each `mark_ringing` id, and `due_calls` each instant `due` was asked about.
    `on_due` callbacks run one per `due` call, in order, before it reads.
    """

    def __init__(self, db: CapabilitiesDb) -> None:
        super().__init__(db)
        self.on_due: list[Callable[[], None]] = []
        self.read_error: BaseException | None = None
        self.write_error: BaseException | None = None
        self.fail_ids: set[str] = set()
        self.ringing_calls: list[str] = []
        self.due_calls: list[datetime] = []

    def _check_write(self, alarm_id: str) -> None:
        if self.write_error is not None and (not self.fail_ids or alarm_id in self.fail_ids):
            raise self.write_error

    async def due(self, now: datetime) -> list[Alarm]:
        self.due_calls.append(now)
        if self.on_due:
            self.on_due.pop(0)()
        if self.read_error is not None:
            raise self.read_error
        return await super().due(now)

    async def mark_ringing(self, alarm_id: str, fired_at: datetime, late_by_s: float) -> bool:
        self.ringing_calls.append(alarm_id)
        self._check_write(alarm_id)
        return await super().mark_ringing(alarm_id, fired_at, late_by_s)

    async def mark_missed(self, alarm_id: str) -> bool:
        self._check_write(alarm_id)
        return await super().mark_missed(alarm_id)

    async def mark_done(self, alarm_id: str) -> bool:
        self._check_write(alarm_id)
        return await super().mark_done(alarm_id)

    async def snooze(self, alarm_id: str, until: datetime) -> bool:
        self._check_write(alarm_id)
        return await super().snooze(alarm_id, until)


class RecordingNotifier:
    """An `AlarmNotifier` that records every call.

    `ring_errors` maps an alarm id to the error its `ring` raises after recording it.
    `state_errors` are raised by the next `state_changed` calls, one each, after recording.
    """

    def __init__(self) -> None:
        self.rings: list[tuple[Alarm, float]] = []
        self.stops: list[str] = []
        self.states: list[tuple[list[Alarm], int, datetime]] = []
        self.ring_errors: dict[str, BaseException] = {}
        self.state_errors: list[BaseException] = []

    async def ring(self, alarm: Alarm, ring_timeout_s: float) -> None:
        self.rings.append((alarm, ring_timeout_s))
        if alarm.alarm_id in self.ring_errors:
            raise self.ring_errors[alarm.alarm_id]

    async def stopped(self, alarm_id: str) -> None:
        self.stops.append(alarm_id)

    async def state_changed(self, alarms: list[Alarm], rev: int, at: datetime) -> None:
        self.states.append((list(alarms), rev, at))
        if self.state_errors:
            raise self.state_errors.pop(0)


@pytest.fixture()
def db(tmp_path):
    database = CapabilitiesDb(str(tmp_path / "capabilities.db"))
    yield database
    database.close()


@pytest.fixture()
def store(db):
    alarm_store = FlakyStore(db)
    alarm_store.migrate()
    return alarm_store


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
    store: AlarmStore,
    clock: SteppedClock,
    notifier: RecordingNotifier,
    settings: AlarmSettings = SETTINGS,
    tz: tzinfo = UTC,
) -> AlarmScheduler:
    return AlarmScheduler(store, clock, notifier, settings, tz)  # type: ignore[arg-type]


@contextlib.asynccontextmanager
async def _running(scheduler: AlarmScheduler) -> AsyncIterator[AlarmScheduler]:
    await scheduler.start()
    try:
        yield scheduler
    finally:
        await scheduler.stop()


async def _slept(clock: SteppedClock, count: int) -> None:
    await clock.slept(count)


def _wake_waiters() -> list[asyncio.Task[object]]:
    """The scheduler's pending wake-waiter tasks: one beside a chunk sleep, none beside a backoff."""
    return [t for t in asyncio.all_tasks() if t.get_name() == f"{TASK_NAME}-wake" and not t.done()]


async def _states(store: AlarmStore) -> dict[str, AlarmState]:
    return {a.alarm_id: a.state for a in await store.list()}


async def _of_state(store: AlarmStore, states: Collection[AlarmState]) -> list[Alarm]:
    return await store.list(states)


def _scheduler_task() -> asyncio.Task[object]:
    return next(t for t in asyncio.all_tasks() if t.get_name() == TASK_NAME)


async def _ended(task: asyncio.Task[object]) -> None:
    """Wait for the task to end; its done-callback, attached first, has run by the return."""
    await asyncio.wait({task})


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat(timespec="microseconds")


async def _insert_raw(
    db: CapabilitiesDb,
    *,
    state: str = "scheduled",
    next_fire_at: datetime = T0,
    late_by_s: float | None = None,
    fired_at: datetime | None = None,
) -> str:
    """Insert a row that decodes but holds what the store's own writes never produce.

    `scheduled_for` is T0, so only the columns a test passes can be out of the domain.
    """
    alarm_id = f"raw-{uuid.uuid4().hex}"

    def insert(conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO alarms (id, kind, label, state, scheduled_for, next_fire_at, repeat, "
            "late_by_s, fired_at) VALUES (?, 'alarm', NULL, ?, ?, ?, NULL, ?, ?)",
            (alarm_id, state, _iso(T0), _iso(next_fire_at), late_by_s, _iso(fired_at)),
        )

    await db.run(insert)
    return alarm_id


async def _set_next_fire_at(db: CapabilitiesDb, alarm_id: str, when: datetime) -> None:
    def update(conn: sqlite3.Connection) -> None:
        conn.execute(
            "UPDATE alarms SET next_fire_at = ?, scheduled_for = ? WHERE id = ?",
            (_iso(when), _iso(when), alarm_id),
        )

    await db.run(update)


def _errors_naming(caplog: pytest.LogCaptureFixture, alarm_id: str) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno == logging.ERROR and alarm_id in r.getMessage()]


class TestFiring:
    @pytest.mark.asyncio
    async def test_a_scheduled_alarm_fires_at_its_time(self, store, clock, notifier):
        alarm = await store.create("alarm", T0 + timedelta(seconds=90), label="wake")

        async with _running(_scheduler(store, clock, notifier)):
            await _slept(clock, 1)
            assert clock.sleeps == [60.0]
            await clock.advance(60)
            await _slept(clock, 2)
            assert clock.sleeps == [60.0, 30.0]
            assert notifier.rings == []

            await clock.advance(30)
            await _slept(clock, 3)

            [(rung, ring_timeout_s)] = notifier.rings
            assert rung.alarm_id == alarm.alarm_id
            assert rung.state == "ringing"
            assert rung.fired_at == T0 + timedelta(seconds=90)
            assert rung.late_by_s == 0.0
            assert ring_timeout_s == 600.0
            [stored] = await store.list()
            assert (stored.state, stored.late_by_s) == ("ringing", 0.0)

            alarms, rev, at = notifier.states[-1]
            assert rev == await store.rev()
            assert [(a.alarm_id, a.state) for a in alarms] == [(alarm.alarm_id, "ringing")]
            assert at == T0 + timedelta(seconds=90)

    @pytest.mark.asyncio
    async def test_schedule_stores_the_alarm_and_fires_it(self, store, clock, notifier):
        scheduler = _scheduler(store, clock, notifier)
        async with _running(scheduler):
            await _slept(clock, 1)
            alarm = await scheduler.schedule("timer", T0 + timedelta(seconds=20), label="tea")

            assert (alarm.kind, alarm.label, alarm.state) == ("timer", "tea", "scheduled")
            assert notifier.states[-1][1] == await store.rev()
            await _slept(clock, 2)
            await clock.advance(20)
            await _slept(clock, 3)

            assert [a.alarm_id for a, _ in notifier.rings] == [alarm.alarm_id]

    @pytest.mark.asyncio
    async def test_sleep_chunks_are_at_most_60_seconds(self, store, clock, notifier):
        await store.create("alarm", T0 + timedelta(hours=3))

        async with _running(_scheduler(store, clock, notifier)):
            for count in range(1, 5):
                await _slept(clock, count)
                await clock.advance(60)
            await _slept(clock, 5)

        assert clock.sleeps == [60.0] * 5
        assert notifier.rings == []

    @pytest.mark.asyncio
    async def test_a_clock_jump_forward_fires_within_one_chunk(self, store, clock, notifier):
        alarm = await store.create("alarm", T0 + timedelta(hours=3))

        async with _running(_scheduler(store, clock, notifier)):
            await _slept(clock, 1)
            clock.jump(3 * 3600 - 30)
            await clock.advance(60)
            await _slept(clock, 2)

            [(rung, _)] = notifier.rings
            assert rung.alarm_id == alarm.alarm_id
            assert rung.late_by_s == 30.0

    @pytest.mark.asyncio
    async def test_the_ring_timeout_auto_stops(self, store, clock, notifier):
        settings = AlarmSettings(ring_timeout_s=90.0, snooze_minutes=9, missed_grace_s=900.0)
        alarm = await store.create("alarm", T0)

        async with _running(_scheduler(store, clock, notifier, settings)):
            await _slept(clock, 1)
            assert [a.alarm_id for a, _ in notifier.rings] == [alarm.alarm_id]
            assert clock.sleeps == [60.0]
            await clock.advance(60)
            await _slept(clock, 2)
            assert clock.sleeps == [60.0, 30.0]
            assert notifier.stops == []

            await clock.advance(30)
            await _slept(clock, 3)

            assert notifier.stops == [alarm.alarm_id]
            assert await _states(store) == {alarm.alarm_id: "done"}
            assert notifier.states[-1][1] == await store.rev()

    @pytest.mark.asyncio
    async def test_snooze_reschedules_by_the_configured_minutes(self, store, clock, notifier):
        settings = AlarmSettings(ring_timeout_s=600.0, snooze_minutes=5, missed_grace_s=900.0)
        scheduler = _scheduler(store, clock, notifier, settings)
        alarm = await store.create("alarm", T0)

        async with _running(scheduler):
            await _slept(clock, 1)
            assert await scheduler.snooze(alarm.alarm_id) is True

            [snoozed] = await store.list()
            assert (snoozed.state, snoozed.next_fire_at) == ("snoozed", T0 + timedelta(minutes=5))
            assert notifier.stops == [alarm.alarm_id]
            for count in range(2, 7):
                await _slept(clock, count)
                await clock.advance(60)
            await _slept(clock, 7)

            assert [a.alarm_id for a, _ in notifier.rings] == [alarm.alarm_id] * 2
            assert notifier.rings[-1][0].fired_at == T0 + timedelta(minutes=5)
            assert notifier.rings[-1][0].late_by_s == 0.0

    @pytest.mark.asyncio
    async def test_snooze_takes_explicit_minutes(self, store, clock, notifier):
        scheduler = _scheduler(store, clock, notifier)
        alarm = await store.create("alarm", T0)

        async with _running(scheduler):
            await _slept(clock, 1)
            assert await scheduler.snooze(alarm.alarm_id, minutes=2) is True

        [snoozed] = await store.list()
        assert snoozed.next_fire_at == T0 + timedelta(minutes=2)

    @pytest.mark.asyncio
    async def test_snooze_refuses_less_than_a_minute(self, store, clock, notifier):
        scheduler = _scheduler(store, clock, notifier)
        alarm = await store.create("alarm", T0)

        async with _running(scheduler):
            await _slept(clock, 1)
            with pytest.raises(ValueError, match="at least 1"):
                await scheduler.snooze(alarm.alarm_id, minutes=0)

        assert await _states(store) == {alarm.alarm_id: "ringing"}

    @pytest.mark.asyncio
    async def test_a_repeat_creates_its_next_occurrence_once(self, store, clock, notifier):
        tz = timezone(timedelta(hours=-5))
        rule = RepeatRule(kind="daily", hour=2, minute=0)  # 02:00 at UTC-5 is T0
        scheduler = _scheduler(store, clock, notifier, tz=tz)
        alarm = await store.create("alarm", T0, label="wake", repeat=rule)

        async with _running(scheduler):
            await _slept(clock, 1)
            [upcoming] = await _of_state(store, ["scheduled"])
            assert upcoming.alarm_id != alarm.alarm_id
            assert upcoming.scheduled_for == T0 + timedelta(days=1)
            assert (upcoming.kind, upcoming.label, upcoming.repeat) == ("alarm", "wake", rule)

            # A snoozed alarm ringing again is the same occurrence: no second next one.
            await scheduler.snooze(alarm.alarm_id, minutes=1)
            await _slept(clock, 2)
            await clock.advance(60)
            await _slept(clock, 3)

            assert len(notifier.rings) == 2
            assert [a.alarm_id for a in await _of_state(store, ["scheduled"])] == [
                upcoming.alarm_id
            ]

    @pytest.mark.asyncio
    async def test_a_snoozed_repeat_ringing_again_does_not_restore_a_cancelled_next_one(
        self, store, clock, notifier
    ):
        rule = RepeatRule(kind="daily", hour=7, minute=0)
        scheduler = _scheduler(store, clock, notifier)
        alarm = await store.create("alarm", T0, repeat=rule)

        async with _running(scheduler):
            await _slept(clock, 1)
            [upcoming] = await _of_state(store, ["scheduled"])
            assert await scheduler.cancel(upcoming.alarm_id) is True
            await _slept(clock, 2)
            await scheduler.snooze(alarm.alarm_id, minutes=1)
            await _slept(clock, 3)
            await clock.advance(60)
            await _slept(clock, 4)

            # The snoozed ring is the occurrence the user already handled, so tomorrow's copy
            # they cancelled stays gone.
            assert len(notifier.rings) == 2
            assert await _states(store) == {alarm.alarm_id: "ringing"}

    @pytest.mark.asyncio
    async def test_a_missed_repeat_creates_the_occurrence_after_now(self, store, clock, notifier):
        # Two days and an hour late: the occurrence after the missed one is already past too,
        # so the next one is computed from now, not from the missed occurrence.
        rule = RepeatRule(kind="daily", hour=6, minute=0)
        await store.create("alarm", T0 - timedelta(days=2, hours=1), repeat=rule)

        async with _running(_scheduler(store, clock, notifier)):
            await _slept(clock, 1)

        [upcoming] = await _of_state(store, ["scheduled"])
        assert upcoming.scheduled_for == datetime(2026, 1, 6, 6, 0, tzinfo=UTC)
        assert len(await _of_state(store, ["missed"])) == 1

    @pytest.mark.asyncio
    async def test_a_retried_repeat_does_not_duplicate_its_next_occurrence(
        self, store, clock, notifier
    ):
        rule = RepeatRule(kind="daily", hour=7, minute=0)
        alarm = await store.create("alarm", T0, repeat=rule)
        store.write_error = sqlite3.OperationalError("database is locked")
        store.fail_ids = {alarm.alarm_id}

        async with _running(_scheduler(store, clock, notifier)):
            await _slept(clock, 1)
            assert clock.sleeps == [1.0]
            store.write_error = None
            await clock.advance(1)
            await _slept(clock, 2)

        assert store.ringing_calls == [alarm.alarm_id, alarm.alarm_id]
        [upcoming] = await _of_state(store, ["scheduled"])
        assert upcoming.scheduled_for == T0 + timedelta(days=1)


class TestGraceRule:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("late_s", "expected"),
        [(300, "ringing"), (900, "ringing"), (901, "missed"), (3600, "missed")],
    )
    async def test_an_alarm_late_at_startup(self, store, clock, notifier, late_s, expected):
        alarm = await store.create("alarm", T0 - timedelta(seconds=late_s))

        async with _running(_scheduler(store, clock, notifier)):
            await _slept(clock, 1)

        [stored] = await store.list()
        assert stored.state == expected
        if expected == "ringing":
            assert [(a.alarm_id, a.late_by_s) for a, _ in notifier.rings] == [
                (alarm.alarm_id, float(late_s))
            ]
            assert notifier.rings[0][0].fired_at == T0
            assert stored.late_by_s == float(late_s)
            assert stored.fired_at == T0
        else:
            assert notifier.rings == []
            assert [a.state for a in notifier.states[-1][0]] == ["missed"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("jump_s", "expected"), [(5 * 60, "ringing"), (20 * 60, "missed")])
    async def test_an_alarm_late_after_a_jump(self, store, clock, notifier, jump_s, expected):
        alarm = await store.create("alarm", T0 + timedelta(seconds=30))

        async with _running(_scheduler(store, clock, notifier)):
            await _slept(clock, 1)
            assert clock.sleeps == [30.0]
            clock.jump(jump_s)
            await clock.advance(30)
            await _slept(clock, 2)

        assert await _states(store) == {alarm.alarm_id: expected}
        if expected == "ringing":
            assert [a.late_by_s for a, _ in notifier.rings] == [float(jump_s)]
        else:
            assert notifier.rings == []

    @pytest.mark.asyncio
    async def test_a_zero_grace_window_still_rings_an_alarm_up_to_60_s_late(
        self, store, clock, notifier
    ):
        settings = AlarmSettings(ring_timeout_s=600.0, snooze_minutes=9, missed_grace_s=0.0)
        on_edge = await store.create("alarm", T0 - timedelta(seconds=60))
        under = await store.create("alarm", T0 - timedelta(seconds=45))
        over = await store.create("alarm", T0 - timedelta(seconds=61))

        async with _running(_scheduler(store, clock, notifier, settings)):
            await _slept(clock, 1)

        assert await _states(store) == {
            on_edge.alarm_id: "ringing",
            under.alarm_id: "ringing",
            over.alarm_id: "missed",
        }


class TestWake:
    @pytest.mark.asyncio
    async def test_schedule_reschedules_without_waiting_out_a_chunk(self, store, clock, notifier):
        scheduler = _scheduler(store, clock, notifier)
        async with _running(scheduler):
            await _slept(clock, 1)
            assert len(_wake_waiters()) == 1
            await scheduler.schedule("timer", T0 + timedelta(seconds=10))
            await _slept(clock, 2)

        assert clock.sleeps == [60.0, 10.0]

    @pytest.mark.asyncio
    async def test_cancel_reschedules_and_deletes(self, store, clock, notifier):
        scheduler = _scheduler(store, clock, notifier)
        alarm = await store.create("alarm", T0 + timedelta(seconds=30))
        async with _running(scheduler):
            await _slept(clock, 1)
            assert await scheduler.cancel(alarm.alarm_id) is True
            await _slept(clock, 2)

            assert clock.sleeps == [30.0, 60.0]
            assert await store.list() == []
            assert notifier.states[-1][0] == []
            assert notifier.stops == []

    @pytest.mark.asyncio
    async def test_cancelling_a_ringing_alarm_stops_it(self, store, clock, notifier):
        scheduler = _scheduler(store, clock, notifier)
        alarm = await store.create("alarm", T0)
        async with _running(scheduler):
            await _slept(clock, 1)
            assert await scheduler.cancel(alarm.alarm_id) is True

        assert notifier.stops == [alarm.alarm_id]

    @pytest.mark.asyncio
    async def test_stop_ringing_reschedules_and_marks_done(self, store, clock, notifier):
        scheduler = _scheduler(store, clock, notifier)
        alarm = await store.create("alarm", T0)
        async with _running(scheduler):
            await _slept(clock, 1)
            assert await scheduler.stop_ringing(alarm.alarm_id) is True
            await _slept(clock, 2)

            assert clock.sleeps == [60.0, 60.0]
            assert await _states(store) == {alarm.alarm_id: "done"}
            assert notifier.stops == [alarm.alarm_id]
            assert notifier.states[-1][1] == await store.rev()

    @pytest.mark.asyncio
    async def test_snooze_reschedules_without_waiting_out_a_chunk(self, store, clock, notifier):
        scheduler = _scheduler(store, clock, notifier)
        alarm = await store.create("alarm", T0)
        async with _running(scheduler):
            await _slept(clock, 1)
            await scheduler.snooze(alarm.alarm_id, minutes=1)
            await _slept(clock, 2)

        assert len(clock.sleeps) == 2

    @pytest.mark.asyncio
    async def test_stop_ringing_a_snoozed_alarm_ends_it_without_a_second_stop(
        self, store, clock, notifier
    ):
        scheduler = _scheduler(store, clock, notifier)
        alarm = await store.create("alarm", T0)
        async with _running(scheduler):
            await _slept(clock, 1)
            await scheduler.snooze(alarm.alarm_id)
            assert await scheduler.stop_ringing(alarm.alarm_id) is True

        assert await _states(store) == {alarm.alarm_id: "done"}
        assert notifier.stops == [alarm.alarm_id]

    @pytest.mark.asyncio
    async def test_a_wake_during_a_tick_runs_the_next_tick_without_sleeping(
        self, store, clock, notifier
    ):
        scheduler = _scheduler(store, clock, notifier)
        store.on_due.append(scheduler.wake)

        async with _running(scheduler):
            await _slept(clock, 1)
            # The wake arrived after the first tick cleared it, so the task went straight into a
            # second tick: no sleep was started for it and then cancelled.
            assert store.due_calls == [T0, T0]
            assert clock.sleeps == [60.0]
            assert clock.cancelled == []

    @pytest.mark.asyncio
    async def test_wake_does_not_cut_a_backoff_sleep_short(self, store, clock, notifier):
        await store.create("alarm", T0)
        store.write_error = sqlite3.OperationalError("database is locked")
        scheduler = _scheduler(store, clock, notifier)

        async with _running(scheduler):
            await _slept(clock, 1)
            # No wake waiter sits beside a backoff sleep, so no wake can end it early. Checked
            # directly because the tick times cannot show it: a wake's early tick would race the
            # `advance` below and read T0 + 1 s either way.
            assert _wake_waiters() == []
            await scheduler.schedule("timer", T0 + timedelta(hours=1))
            await clock.advance(1)
            await _slept(clock, 2)

        assert store.due_calls == [T0, T0 + timedelta(seconds=1)]
        assert clock.sleeps == [1.0, 2.0]

    @pytest.mark.asyncio
    async def test_a_method_that_changes_nothing_returns_false(self, store, clock, notifier):
        scheduler = _scheduler(store, clock, notifier)
        async with _running(scheduler):
            await _slept(clock, 1)
            assert await scheduler.stop_ringing("no-such-alarm") is False
            assert await scheduler.snooze("no-such-alarm") is False
            assert await scheduler.cancel("no-such-alarm") is False


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_health_is_stopped_before_start(self, store, clock, notifier):
        scheduler = _scheduler(store, clock, notifier)

        assert scheduler.health() == SchedulerHealth(state="stopped", reason=None)
        assert scheduler.task_dead() is False

    @pytest.mark.asyncio
    async def test_stop_cancels_the_task_cleanly(self, store, clock, notifier, caplog):
        scheduler = _scheduler(store, clock, notifier)
        with caplog.at_level(logging.WARNING, logger=SCHEDULER_LOGGER):
            await scheduler.start()
            await _slept(clock, 1)
            task = _scheduler_task()
            assert scheduler.health() == SchedulerHealth(state="running", reason=None)

            await scheduler.stop()

        assert task.done()
        assert clock.pending == 0
        assert scheduler.health() == SchedulerHealth(state="stopped", reason=None)
        assert scheduler.task_dead() is False
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []

    @pytest.mark.asyncio
    async def test_start_twice_runs_one_task(self, store, clock, notifier):
        scheduler = _scheduler(store, clock, notifier)
        async with _running(scheduler):
            await scheduler.start()
            await _slept(clock, 1)
            assert [t.get_name() for t in asyncio.all_tasks()].count(TASK_NAME) == 1


class TestSupervision:
    def test_the_contained_set_is_a_module_constant(self):
        assert (MistError, sqlite3.Error, OSError) == scheduler_module._CONTAINED_ERRORS
        assert "except Exception" not in inspect.getsource(scheduler_module)

    @pytest.mark.asyncio
    async def test_one_alarm_failing_does_not_hold_back_another(
        self, store, clock, notifier, caplog
    ):
        # A second earlier, so the store returns the broken alarm first.
        broken = await store.create("alarm", T0 - timedelta(seconds=1))
        healthy = await store.create("alarm", T0)
        store.write_error = CapabilityStoreError("alarm store cannot write")
        store.fail_ids = {broken.alarm_id}

        with caplog.at_level(logging.ERROR, logger=SCHEDULER_LOGGER):
            async with _running(_scheduler(store, clock, notifier)):
                await _slept(clock, 1)

        assert [a.alarm_id for a, _ in notifier.rings] == [healthy.alarm_id]
        assert await _states(store) == {broken.alarm_id: "scheduled", healthy.alarm_id: "ringing"}
        assert any(broken.alarm_id in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_one_ring_timeout_failing_does_not_hold_back_another(
        self, store, clock, notifier
    ):
        settings = AlarmSettings(ring_timeout_s=60.0, snooze_minutes=9, missed_grace_s=900.0)
        # A second earlier, so the store returns the broken alarm first; both fire at T0.
        broken = await store.create("alarm", T0 - timedelta(seconds=1))
        healthy = await store.create("alarm", T0)

        async with _running(_scheduler(store, clock, notifier, settings)):
            await _slept(clock, 1)
            store.write_error = sqlite3.OperationalError("database is locked")
            store.fail_ids = {broken.alarm_id}
            await clock.advance(60)
            await _slept(clock, 2)

            assert await _states(store) == {
                broken.alarm_id: "ringing",
                healthy.alarm_id: "done",
            }
            assert notifier.stops == [healthy.alarm_id]

    @pytest.mark.asyncio
    async def test_a_failing_notifier_ring_is_logged_and_does_not_fail_the_tick(
        self, store, clock, notifier, caplog
    ):
        alarm = await store.create("alarm", T0)
        notifier.ring_errors[alarm.alarm_id] = OSError("room unit unreachable")
        scheduler = _scheduler(store, clock, notifier)

        with caplog.at_level(logging.ERROR, logger=SCHEDULER_LOGGER):
            async with _running(scheduler):
                await _slept(clock, 1)
                assert scheduler.health() == SchedulerHealth(state="running", reason=None)
                assert clock.sleeps == [60.0]

        assert await _states(store) == {alarm.alarm_id: "ringing"}
        assert len(_errors_naming(caplog, alarm.alarm_id)) == 1

    @pytest.mark.asyncio
    async def test_schedule_returns_when_publishing_fails_and_a_tick_publishes_again(
        self, store, clock, notifier
    ):
        scheduler = _scheduler(store, clock, notifier)
        async with _running(scheduler):
            await _slept(clock, 1)
            notifier.state_errors.append(OSError("socket closed"))

            alarm = await scheduler.schedule("timer", T0 + timedelta(seconds=10))
            await _slept(clock, 2)

            assert len(notifier.states) == 2
            assert [a.alarm_id for a in notifier.states[-1][0]] == [alarm.alarm_id]
            assert notifier.states[-1][1] == await store.rev()

    @pytest.mark.asyncio
    async def test_a_failed_read_skips_the_tick_and_the_task_carries_on(
        self, store, clock, notifier
    ):
        alarm = await store.create("alarm", T0)
        store.read_error = sqlite3.OperationalError("disk I/O error at /srv/private/x.db")
        scheduler = _scheduler(store, clock, notifier)

        async with _running(scheduler):
            await _slept(clock, 1)
            assert notifier.rings == []
            assert clock.sleeps == [1.0]
            assert not _scheduler_task().done()
            health = scheduler.health()
            assert health.state == "failing"
            assert "alarm store" in (health.reason or "")
            assert "OperationalError" in (health.reason or "")
            assert "disk I/O" not in (health.reason or "")
            assert "/srv/private" not in (health.reason or "")

            store.read_error = None
            await clock.advance(1)
            await _slept(clock, 2)

            assert [a.alarm_id for a, _ in notifier.rings] == [alarm.alarm_id]
            assert scheduler.health() == SchedulerHealth(state="running", reason=None)

    @pytest.mark.asyncio
    async def test_unwritable_store_backs_off_then_rings_late_on_recovery(
        self, store, clock, notifier
    ):
        alarm = await store.create("alarm", T0)
        store.write_error = sqlite3.OperationalError("attempt to write a readonly database")
        scheduler = _scheduler(store, clock, notifier)
        backoff = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0, 60.0]

        async with _running(scheduler):
            for count in range(1, len(backoff)):
                await _slept(clock, count)
                await clock.advance(clock.sleeps[-1])
            await _slept(clock, len(backoff))

            assert clock.sleeps == backoff
            health = scheduler.health()
            assert health.state == "failing"
            assert "alarm store" in (health.reason or "")
            assert "readonly" not in (health.reason or "")
            assert not scheduler.task_dead()
            assert await _states(store) == {alarm.alarm_id: "scheduled"}

            store.write_error = None
            await clock.advance(60)
            await _slept(clock, len(backoff) + 1)

            late_by_s = sum(backoff)
            assert [(a.alarm_id, a.late_by_s) for a, _ in notifier.rings] == [
                (alarm.alarm_id, late_by_s)
            ]
            assert scheduler.health() == SchedulerHealth(state="running", reason=None)
            assert clock.sleeps[-1] == 60.0
            assert 0 not in clock.sleeps

    @pytest.mark.asyncio
    async def test_a_clean_tick_resets_the_backoff(self, store, clock, notifier):
        await store.create("alarm", T0)
        store.write_error = sqlite3.OperationalError("database is locked")

        async with _running(_scheduler(store, clock, notifier)):
            for count in range(1, 3):
                await _slept(clock, count)
                await clock.advance(clock.sleeps[-1])
            await _slept(clock, 3)
            assert clock.sleeps == [1.0, 2.0, 4.0]

            store.write_error = None
            await clock.advance(4)
            await _slept(clock, 4)
            store.read_error = sqlite3.OperationalError("database is locked")
            await clock.advance(60)
            await _slept(clock, 5)

        assert clock.sleeps == [1.0, 2.0, 4.0, 60.0, 1.0]

    @pytest.mark.asyncio
    async def test_an_unexpected_error_ends_the_task_and_is_recorded(
        self, store, clock, notifier, caplog
    ):
        store.read_error = RuntimeError("internal detail 7731")
        scheduler = _scheduler(store, clock, notifier)

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
    async def test_a_dead_scheduler_is_not_restarted(self, store, clock, notifier):
        store.read_error = RuntimeError("bug")
        scheduler = _scheduler(store, clock, notifier)
        await scheduler.start()
        await _ended(_scheduler_task())

        await scheduler.start()

        assert TASK_NAME not in [t.get_name() for t in asyncio.all_tasks()]
        assert scheduler.health().state == "dead"

    @pytest.mark.asyncio
    async def test_a_cancel_without_stop_is_a_death(self, store, clock, notifier, caplog):
        scheduler = _scheduler(store, clock, notifier)

        with caplog.at_level(logging.ERROR, logger=SCHEDULER_LOGGER):
            await scheduler.start()
            await _slept(clock, 1)
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
    async def test_a_restarted_scheduler_still_records_a_death(self, store, clock, notifier):
        scheduler = _scheduler(store, clock, notifier)
        await scheduler.start()
        await _slept(clock, 1)
        await scheduler.stop()

        # The second run is not stopping, so its death is not read as an orderly end.
        store.read_error = RuntimeError("bug")
        await scheduler.start()
        await _ended(_scheduler_task())

        assert scheduler.health() == SchedulerHealth(
            state="dead", reason="alarm scheduler task died: RuntimeError"
        )
        assert scheduler.task_dead() is True

    @pytest.mark.asyncio
    async def test_a_failed_tick_publish_reports_the_error_class_not_its_text(
        self, store, clock, notifier
    ):
        alarm = await store.create("alarm", T0)
        notifier.state_errors.append(OSError("socket closed by peer"))
        scheduler = _scheduler(store, clock, notifier)

        async with _running(scheduler):
            await _slept(clock, 1)

            assert clock.sleeps == [1.0]
            assert scheduler.health() == SchedulerHealth(
                state="failing", reason="alarm scheduler tick failed (OSError)"
            )

            await clock.advance(1)
            await _slept(clock, 2)

            # The ringing state was stored, so the retried publish carries it.
            assert [(a.alarm_id, a.state) for a in notifier.states[-1][0]] == [
                (alarm.alarm_id, "ringing")
            ]
            assert scheduler.health() == SchedulerHealth(state="running", reason=None)

        assert [a.alarm_id for a, _ in notifier.rings] == [alarm.alarm_id]


def _domain_alarm(**changes: object) -> Alarm:
    fields: dict[str, object] = {
        "alarm_id": "a1",
        "kind": "alarm",
        "label": None,
        "state": "ringing",
        "scheduled_for": T0,
        "next_fire_at": T0,
        "repeat": RepeatRule(kind="daily", hour=7, minute=0),
        "late_by_s": 0.0,
        "fired_at": T0,
    }
    fields.update(changes)
    return Alarm(**fields)  # type: ignore[arg-type]


class TestQuarantine:
    """Rows that decode but fall outside the alarm domain cost only themselves."""

    @pytest.mark.parametrize(
        ("changes", "field"),
        [
            ({}, None),
            ({"fired_at": None, "late_by_s": None, "repeat": None}, None),
            ({"late_by_s": 3}, None),
            ({"kind": "reminder"}, "kind"),
            ({"state": "cancelled"}, "state"),
            ({"repeat": "daily"}, "repeat"),
            ({"late_by_s": -0.5}, "late_by_s"),
            ({"late_by_s": float("nan")}, "late_by_s"),
            ({"late_by_s": float("inf")}, "late_by_s"),
            ({"late_by_s": True}, "late_by_s"),
            ({"late_by_s": "5"}, "late_by_s"),
            ({"scheduled_for": datetime(2026, 1, 5, 7, 0)}, "scheduled_for"),
            ({"next_fire_at": None}, "next_fire_at"),
            ({"next_fire_at": FAR_FUTURE}, "next_fire_at"),
            ({"fired_at": FAR_FUTURE}, "fired_at"),
            ({"fired_at": "2026-01-05T07:00:00+00:00"}, "fired_at"),
            ({"fired_at": datetime(9000, 1, 1, tzinfo=UTC)}, None),
        ],
    )
    def test_the_domain_check_names_the_first_field_outside_it(self, changes, field):
        assert scheduler_module._out_of_domain_field(_domain_alarm(**changes)) == field

    def test_the_latest_time_leaves_room_for_any_plausible_ring_timeout(self):
        # A century of ringing still fits after the bound.
        assert scheduler_module._LATEST_TIME + timedelta(days=36525) < datetime.max.replace(
            tzinfo=UTC
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("columns", "field"),
        [
            ({"state": "ringing", "late_by_s": -5.0, "fired_at": T0}, "late_by_s"),
            ({"state": "ringing", "late_by_s": 0.0, "fired_at": FAR_FUTURE}, "fired_at"),
            ({"state": "scheduled", "next_fire_at": FAR_FUTURE}, "next_fire_at"),
        ],
    )
    async def test_an_out_of_domain_row_is_skipped_and_logged_once(
        self, db, store, clock, notifier, caplog, columns, field
    ):
        bad_id = await _insert_raw(db, **columns)
        healthy = await store.create("alarm", T0 + timedelta(seconds=30))
        scheduler = _scheduler(store, clock, notifier)

        with caplog.at_level(logging.ERROR, logger=SCHEDULER_LOGGER):
            async with _running(scheduler):
                await _slept(clock, 1)
                await clock.advance(30)
                await _slept(clock, 2)
                await clock.advance(60)
                await _slept(clock, 3)

                assert scheduler.health() == SchedulerHealth(state="running", reason=None)
                assert [a.alarm_id for a, _ in notifier.rings] == [healthy.alarm_id]
                assert all(
                    bad_id not in [a.alarm_id for a in alarms] for alarms, _, _ in notifier.states
                )

        [record] = _errors_naming(caplog, bad_id)
        assert field in record.getMessage()

    @pytest.mark.asyncio
    async def test_an_out_of_domain_ringing_row_is_not_timed_out(self, db, store, clock, notifier):
        rang_at = T0 - timedelta(seconds=SETTINGS.ring_timeout_s + 100)
        bad_id = await _insert_raw(db, state="ringing", late_by_s=-5.0, fired_at=rang_at)
        healthy = await store.create("alarm", rang_at)
        assert await store.mark_ringing(healthy.alarm_id, rang_at, 0.0) is True
        scheduler = _scheduler(store, clock, notifier)

        async with _running(scheduler):
            await _slept(clock, 1)

            # Both are past their ring timeout; only the alarm inside the domain is stopped.
            assert await _states(store) == {bad_id: "ringing", healthy.alarm_id: "done"}
            assert notifier.stops == [healthy.alarm_id]
            assert scheduler.health() == SchedulerHealth(state="running", reason=None)

    @pytest.mark.asyncio
    async def test_an_alarm_whose_handling_overflows_is_quarantined(
        self, store, clock, notifier, caplog
    ):
        broken = await store.create("alarm", T0 - timedelta(seconds=1))
        healthy = await store.create("alarm", T0)
        store.write_error = OverflowError("date value out of range")
        store.fail_ids = {broken.alarm_id}
        scheduler = _scheduler(store, clock, notifier)

        with caplog.at_level(logging.ERROR, logger=SCHEDULER_LOGGER):
            async with _running(scheduler):
                await _slept(clock, 1)
                await clock.advance(60)
                await _slept(clock, 2)

                assert scheduler.health() == SchedulerHealth(state="running", reason=None)
                assert clock.sleeps == [60.0, 60.0]

        assert [a.alarm_id for a, _ in notifier.rings] == [healthy.alarm_id]
        assert store.ringing_calls.count(broken.alarm_id) == 1
        assert len(_errors_naming(caplog, broken.alarm_id)) == 1

    @pytest.mark.asyncio
    async def test_a_notifier_ring_raising_value_error_costs_only_that_alarm(
        self, store, clock, notifier
    ):
        broken = await store.create("alarm", T0 - timedelta(seconds=1))
        healthy = await store.create("alarm", T0)
        notifier.ring_errors[broken.alarm_id] = ValueError("payload out of range")
        scheduler = _scheduler(store, clock, notifier)

        async with _running(scheduler):
            await _slept(clock, 1)

            assert scheduler.health() == SchedulerHealth(state="running", reason=None)
            assert sorted(a.alarm_id for a, _ in notifier.rings) == sorted(
                [broken.alarm_id, healthy.alarm_id]
            )
            assert (await _states(store))[healthy.alarm_id] == "ringing"

    @pytest.mark.asyncio
    async def test_schedule_returns_while_a_quarantined_row_is_stored(
        self, db, store, clock, notifier
    ):
        bad_id = await _insert_raw(db, state="ringing", late_by_s=-5.0, fired_at=T0)
        scheduler = _scheduler(store, clock, notifier)

        async with _running(scheduler):
            await _slept(clock, 1)
            alarm = await scheduler.schedule("timer", T0 + timedelta(seconds=10))

            assert alarm.state == "scheduled"
            assert [a.alarm_id for a in notifier.states[-1][0]] == [alarm.alarm_id]
            assert bad_id in await _states(store)

    @pytest.mark.asyncio
    async def test_an_edited_row_is_checked_again_and_fires(self, db, store, clock, notifier):
        bad_id = await _insert_raw(db, state="scheduled", next_fire_at=FAR_FUTURE)
        scheduler = _scheduler(store, clock, notifier)

        async with _running(scheduler):
            await _slept(clock, 1)
            assert clock.sleeps == [60.0]

            await _set_next_fire_at(db, bad_id, T0 + timedelta(seconds=30))
            scheduler.wake()
            await _slept(clock, 2)
            assert clock.sleeps == [60.0, 30.0]
            await clock.advance(30)
            await _slept(clock, 3)

            assert [a.alarm_id for a, _ in notifier.rings] == [bad_id]
