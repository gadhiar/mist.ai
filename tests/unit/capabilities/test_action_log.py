"""Tests for the action log: one row per capability invocation, pruned, never failing a call."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from datetime import timedelta
from typing import TypeVar

import pytest

from backend.capabilities.action_log import MAX_ROWS, ActionLog, ActionLogEntry
from backend.capabilities.contracts import InvocationContext
from backend.capabilities.db import CapabilitiesDb
from backend.capabilities.decoder import ActionDecoder
from backend.capabilities.host import CapabilityHost, HostSettings
from backend.capabilities.policy import PolicyTable
from backend.capabilities.router import RouterConfig, ToolRouter
from backend.capabilities.secrets import SecretRegistry
from backend.chat.context_budget import ApproximateTokenCounter
from backend.llm.models import ToolCall
from tests.unit.capabilities.fakes import (
    DEFAULT_START,
    FakeCapability,
    FakeClock,
    RecordingEventSink,
)

T = TypeVar("T")


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
