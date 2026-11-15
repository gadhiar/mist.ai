"""Tests for the action log: one row per capability invocation, pruned, never failing a call."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta, timezone
from typing import TypeVar

import pytest

from backend.capabilities.action_log import MAX_ROWS, ActionLog, ActionLogEntry, DayUsage
from backend.capabilities.contracts import InvocationContext
from backend.capabilities.db import CapabilitiesDb
from backend.capabilities.decoder import ActionDecoder
from backend.capabilities.host import CapabilityHost, HostSettings
from backend.capabilities.policy import PolicyTable
from backend.capabilities.router import RouterConfig, ToolRouter
from backend.capabilities.secrets import SecretRegistry
from backend.chat.context_budget import ApproximateTokenCounter
from backend.errors import CapabilityStoreError
from backend.llm.models import ToolCall
from tests.unit.capabilities.fakes import (
    DEFAULT_START,
    FakeCapability,
    FakeClock,
    RecordingEventSink,
)

T = TypeVar("T")

# The action log's component step 1, copied here so the migration test starts from the schema a
# step-1 database holds, whatever later steps the code gains.
STEP_ONE = (
    "CREATE TABLE action_log ("
    " id INTEGER PRIMARY KEY AUTOINCREMENT,"
    " at TEXT NOT NULL,"
    " session_id TEXT NOT NULL,"
    " turn_id TEXT NOT NULL,"
    " tool_call_id TEXT NOT NULL,"
    " tool TEXT NOT NULL,"
    " capability TEXT NOT NULL,"
    " source TEXT NOT NULL,"
    " status TEXT NOT NULL,"
    " args_summary TEXT NOT NULL,"
    " summary TEXT NOT NULL,"
    " reason TEXT,"
    " proposal_id TEXT,"
    " duration_ms INTEGER NOT NULL)",
)


class CountingDb(CapabilitiesDb):
    """A real database that counts `run` calls, to show writes go through it."""

    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.runs = 0

    async def run(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        self.runs += 1
        return await super().run(fn)


def _entry(index: int = 0, **changes: object) -> ActionLogEntry:
    fields: dict[str, object] = {
        "at": DEFAULT_START + timedelta(seconds=index),
        "session_id": "s1",
        "turn_id": "t1",
        "tool_call_id": f"call-{index}",
        "tool": "fake.do",
        "capability": "fake",
        "source": "model",
        "status": "ok",
        "args_summary": "do it",
        "summary": "fake.do done.",
        "reason": None,
        "proposal_id": None,
        "duration_ms": 3,
    }
    fields.update(changes)
    return ActionLogEntry(**fields)  # type: ignore[arg-type]


async def _rows(db: CapabilitiesDb) -> list[tuple[object, ...]]:
    return await db.run(
        lambda conn: conn.execute(
            "SELECT tool_call_id, session_id, turn_id, tool, capability, source, status,"
            " args_summary, summary, reason, proposal_id, duration_ms, at"
            " FROM action_log ORDER BY id"
        ).fetchall()
    )


def _unwritable_path(tmp_path) -> str:
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("a regular file", encoding="utf-8")
    return str(blocker / "capabilities.db")


def _host(action_log: ActionLog, capability: FakeCapability) -> CapabilityHost:
    counter = ApproximateTokenCounter()
    router = ToolRouter(
        [capability], RouterConfig(token_budget=1200, reserved_tokens=100), counter, None
    )
    return CapabilityHost(
        [capability],
        router=router,
        counter=counter,
        policy=PolicyTable.from_specs(capability.tools(), None),
        decoder=ActionDecoder(),
        secrets=SecretRegistry(),
        action_log=action_log,
        event_sink=RecordingEventSink(),
        clock=FakeClock(),
        settings=HostSettings(),
    )


def _ctx(call_id: str = "call-1") -> InvocationContext:
    return InvocationContext(
        session_id="s1",
        turn_id="t1",
        utterance="do it",
        tool_call_id=call_id,
        source="fastpath",
        now=DEFAULT_START,
    )


class TestWrites:
    @pytest.mark.asyncio
    async def test_writes_one_row_per_entry_through_db_run(self, tmp_path):
        db = CountingDb(str(tmp_path / "capabilities.db"))
        log = ActionLog(db)

        await log.record(_entry(0))
        runs_after_first = db.runs
        await log.record(_entry(1, status="error", reason="boom", proposal_id="p1"))
        runs_after_second = db.runs

        rows = await _rows(db)
        assert [row[0] for row in rows] == ["call-0", "call-1"]
        assert rows[1][1:12] == (
            "s1",
            "t1",
            "fake.do",
            "fake",
            "model",
            "error",
            "do it",
            "fake.do done.",
            "boom",
            "p1",
            3,
        )
        assert rows[0][12] == (DEFAULT_START).isoformat()
        assert runs_after_first == 1
        assert runs_after_second == 2

    @pytest.mark.asyncio
    async def test_prunes_to_the_newest_rows(self, tmp_path):
        db = CapabilitiesDb(str(tmp_path / "capabilities.db"))
        log = ActionLog(db, max_rows=3)

        for index in range(5):
            await log.record(_entry(index))

        assert [row[0] for row in await _rows(db)] == ["call-2", "call-3", "call-4"]

    @pytest.mark.asyncio
    async def test_default_limit_is_ten_thousand_rows(self, tmp_path):
        db = CapabilitiesDb(str(tmp_path / "capabilities.db"))
        log = ActionLog(db)
        await log.record(_entry(0))
        filler = [(f"fill-{index}", DEFAULT_START.isoformat()) for index in range(MAX_ROWS - 1)]

        def fill(conn: sqlite3.Connection) -> None:
            # One transaction: the connection autocommits, and a commit per row is slow.
            conn.execute("BEGIN")
            conn.executemany(
                "INSERT INTO action_log (tool_call_id, at, session_id, turn_id, tool,"
                " capability, source, status, args_summary, summary, duration_ms)"
                " VALUES (?, ?, 's', 't', 'x.y', 'x', 'model', 'ok', '', '', 0)",
                filler,
            )
            conn.execute("COMMIT")

        await db.run(fill)

        await log.record(_entry(1))

        count, oldest = await db.run(
            lambda conn: conn.execute(
                "SELECT COUNT(*), (SELECT tool_call_id FROM action_log ORDER BY id LIMIT 1)"
                " FROM action_log"
            ).fetchone()
        )
        assert MAX_ROWS == 10000
        assert count == MAX_ROWS
        assert oldest == "fill-0"


class TestStepTwo:
    @pytest.mark.asyncio
    async def test_writes_the_level_and_turn_ms(self, tmp_path):
        db = CapabilitiesDb(str(tmp_path / "capabilities.db"))
        log = ActionLog(db)

        await log.record(_entry(0, power_level="low", turn_ms=412))
        await log.record(_entry(1))

        rows = await db.run(
            lambda conn: conn.execute(
                "SELECT tool_call_id, power_level, turn_ms FROM action_log ORDER BY id"
            ).fetchall()
        )
        assert rows == [("call-0", "low", 412), ("call-1", None, None)]

    @pytest.mark.asyncio
    async def test_step_one_database_migrates_and_its_rows_read_null(self, tmp_path):
        db = CapabilitiesDb(str(tmp_path / "capabilities.db"))
        db.migrate("action_log", STEP_ONE)

        def old_row(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT INTO action_log (tool_call_id, at, session_id, turn_id, tool,"
                " capability, source, status, args_summary, summary, duration_ms)"
                " VALUES ('old', ?, 's', 't', 'x.y', 'x', 'model', 'ok', '', '', 0)",
                (DEFAULT_START.isoformat(),),
            )

        await db.run(old_row)

        await ActionLog(db).record(_entry(1, power_level="off", turn_ms=5))

        rows = await db.run(
            lambda conn: conn.execute(
                "SELECT tool_call_id, power_level, turn_ms FROM action_log ORDER BY id"
            ).fetchall()
        )
        version = await db.run(
            lambda conn: conn.execute(
                "SELECT version FROM schema_versions WHERE component = 'action_log'"
            ).fetchone()
        )
        assert rows == [("old", None, None), ("call-1", "off", 5)]
        assert version == (2,)


class TestFailures:
    def test_constructor_touches_no_file(self, tmp_path):
        path = tmp_path / "nested" / "capabilities.db"

        ActionLog(CapabilitiesDb(str(path)))

        assert not path.parent.exists()

    @pytest.mark.asyncio
    async def test_write_failure_is_logged_and_does_not_raise(self, tmp_path, caplog):
        log = ActionLog(CapabilitiesDb(_unwritable_path(tmp_path)))

        with caplog.at_level(logging.WARNING):
            await log.record(_entry(0))

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "action log" in warnings[0].getMessage()

    @pytest.mark.asyncio
    async def test_statement_failure_is_logged_and_does_not_raise(self, tmp_path, caplog):
        db = CapabilitiesDb(str(tmp_path / "capabilities.db"))
        log = ActionLog(db)
        await log.record(_entry(0))
        await db.run(lambda conn: conn.execute("DROP TABLE action_log"))

        with caplog.at_level(logging.WARNING):
            await log.record(_entry(1))

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "action log" in warnings[0].getMessage()

    @pytest.mark.asyncio
    async def test_unwritable_database_never_fails_an_invocation(self, tmp_path, caplog):
        capability = FakeCapability("fake")
        host = _host(ActionLog(CapabilitiesDb(_unwritable_path(tmp_path))), capability)

        with caplog.at_level(logging.WARNING):
            record = await host.invoke(
                ToolCall(id="call-1", name="fake.do", arguments={}), _ctx(), None
            )

        assert record.outcome.status == "ok"
        assert len(capability.invocations) == 1
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "action log" in warnings[0].getMessage()

    @pytest.mark.asyncio
    async def test_host_writes_one_row_per_invocation(self, tmp_path):
        db = CountingDb(str(tmp_path / "capabilities.db"))
        host = _host(ActionLog(db), FakeCapability("fake"))

        await host.invoke(ToolCall(id="call-1", name="fake.do", arguments={}), _ctx(), None)
        await host.invoke(ToolCall(id="call-2", name="fake.do", arguments={}), _ctx("call-2"), None)

        rows = await _rows(db)
        assert [(row[0], row[5], row[6]) for row in rows] == [
            ("call-1", "fastpath", "ok"),
            ("call-2", "fastpath", "ok"),
        ]


# -- Usage -----------------------------------------------------------------------------------------

NOW = datetime(2026, 1, 5, 7, 0, tzinfo=UTC)


def _empty_day(day: date) -> DayUsage:
    return DayUsage(
        date=day,
        invocations=0,
        by_capability={},
        by_source={"fastpath": 0, "model": 0, "proposal": 0},
        by_level={"full": 0, "low": 0, "off": 0},
        by_status={},
        turn_ms_p50=None,
        turn_ms_p90=None,
    )


async def _seed(log: ActionLog, *rows: dict[str, object]) -> None:
    for index, row in enumerate(rows):
        await log.record(_entry(index, **row))


class TestUsage:
    @pytest.mark.asyncio
    async def test_counts_each_day_by_capability_source_level_and_status(self, tmp_path):
        log = ActionLog(CapabilitiesDb(str(tmp_path / "capabilities.db")))
        await _seed(
            log,
            {
                "at": datetime(2026, 1, 5, 1, 0, tzinfo=UTC),
                "capability": "lights",
                "source": "fastpath",
                "power_level": "full",
                "turn_ms": 100,
            },
            {
                "at": datetime(2026, 1, 5, 2, 0, tzinfo=UTC),
                "capability": "lights",
                "source": "model",
                "power_level": "low",
                "turn_ms": 200,
            },
            {
                "at": datetime(2026, 1, 5, 3, 0, tzinfo=UTC),
                "capability": "alarms",
                "source": "proposal",
                "status": "pending_confirmation",
                "power_level": "off",
                "turn_ms": 300,
            },
            # A row written before the power port: no level and no turn time.
            {
                "at": datetime(2026, 1, 5, 4, 0, tzinfo=UTC),
                "capability": "alarms",
                "source": "fastpath",
                "status": "denied",
            },
            {
                "at": datetime(2026, 1, 4, 9, 0, tzinfo=UTC),
                "capability": "clock",
                "status": "error",
                "power_level": "full",
                "turn_ms": 50,
            },
        )

        usage = await log.usage(3, UTC, now=NOW)

        assert usage.days == (
            DayUsage(
                date=date(2026, 1, 5),
                invocations=4,
                by_capability={"lights": 2, "alarms": 2},
                by_source={"fastpath": 2, "model": 1, "proposal": 1},
                by_level={"full": 1, "low": 1, "off": 1},
                by_status={"ok": 2, "pending_confirmation": 1, "denied": 1},
                turn_ms_p50=200,
                turn_ms_p90=300,
            ),
            DayUsage(
                date=date(2026, 1, 4),
                invocations=1,
                by_capability={"clock": 1},
                by_source={"fastpath": 0, "model": 1, "proposal": 0},
                by_level={"full": 1, "low": 0, "off": 0},
                by_status={"error": 1},
                turn_ms_p50=50,
                turn_ms_p90=50,
            ),
            _empty_day(date(2026, 1, 3)),
        )

    @pytest.mark.asyncio
    async def test_turn_ms_percentiles_are_nearest_rank(self, tmp_path):
        log = ActionLog(CapabilitiesDb(str(tmp_path / "capabilities.db")))
        # Written out of order, so the percentiles cannot come from the insertion order.
        await _seed(log, *({"turn_ms": ms} for ms in (70, 10, 100, 40, 20, 90, 30, 60, 80, 50)))

        (today,) = (await log.usage(1, UTC, now=NOW)).days

        assert (today.turn_ms_p50, today.turn_ms_p90) == (50, 90)

    @pytest.mark.asyncio
    async def test_every_day_is_present_newest_first_on_an_empty_log(self, tmp_path):
        log = ActionLog(CapabilitiesDb(str(tmp_path / "capabilities.db")))

        usage = await log.usage(7, UTC, now=NOW)

        assert usage.since is None
        assert usage.days == tuple(
            _empty_day(date(2026, 1, 5) - timedelta(days=n)) for n in range(7)
        )

    @pytest.mark.asyncio
    async def test_rows_land_on_local_dates_across_a_utc_midnight(self, tmp_path):
        log = ActionLog(CapabilitiesDb(str(tmp_path / "capabilities.db")))
        await _seed(
            log,
            # 22:00 on 4 Jan five hours behind UTC, though 5 Jan in UTC.
            {"at": datetime(2026, 1, 5, 3, 0, tzinfo=UTC), "capability": "behind"},
            # 01:00 on 5 Jan nine hours ahead of UTC, though 4 Jan in UTC.
            {"at": datetime(2026, 1, 4, 16, 0, tzinfo=UTC), "capability": "ahead"},
        )

        west = await log.usage(2, timezone(timedelta(hours=-5)), now=NOW)
        east = await log.usage(2, timezone(timedelta(hours=9)), now=NOW)

        assert [(d.date, d.by_capability) for d in west.days] == [
            (date(2026, 1, 5), {}),
            (date(2026, 1, 4), {"behind": 1, "ahead": 1}),
        ]
        assert [(d.date, d.by_capability) for d in east.days] == [
            (date(2026, 1, 5), {"behind": 1, "ahead": 1}),
            (date(2026, 1, 4), {}),
        ]

    @pytest.mark.asyncio
    async def test_since_is_the_oldest_row_even_outside_the_window(self, tmp_path):
        log = ActionLog(CapabilitiesDb(str(tmp_path / "capabilities.db")))
        await _seed(
            log,
            {"at": datetime(2025, 12, 1, 8, 30, tzinfo=UTC)},
            {"at": datetime(2026, 1, 5, 6, 0, tzinfo=UTC)},
        )

        usage = await log.usage(2, UTC, now=NOW)

        assert usage.since == datetime(2025, 12, 1, 8, 30, tzinfo=UTC)
        assert [d.invocations for d in usage.days] == [1, 0]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("days", [0, 91])
    async def test_days_outside_one_to_ninety_is_refused(self, tmp_path, days):
        log = ActionLog(CapabilitiesDb(str(tmp_path / "capabilities.db")))

        with pytest.raises(ValueError, match="days"):
            await log.usage(days, UTC, now=NOW)

    @pytest.mark.asyncio
    async def test_an_unopenable_database_raises_a_store_error(self, tmp_path):
        log = ActionLog(CapabilitiesDb(_unwritable_path(tmp_path)))

        with pytest.raises(CapabilityStoreError):
            await log.usage(7, UTC, now=NOW)

    @pytest.mark.asyncio
    async def test_a_failing_read_raises_a_store_error(self, tmp_path):
        db = CapabilitiesDb(str(tmp_path / "capabilities.db"))
        log = ActionLog(db)
        await log.record(_entry(0))
        await db.run(lambda conn: conn.execute("DROP TABLE action_log"))

        with pytest.raises(CapabilityStoreError, match="action log"):
            await log.usage(7, UTC, now=NOW)
