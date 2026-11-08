"""Tests for `GET /usage/uptime` and the server's uptime tick loop.

Every store is synthetic, built in `tmp_path` from the real `schema.sql`, and
`EVENT_STORE_DB_PATH` always points at it, so no test reaches `~/.mist`.
`TestClient` is constructed without the `with` block, as in
`tests/unit/test_usage_endpoint.py`: entering it would run the ASGI lifespan
and load the model stack. The live status comes from a stub registry bound to
`server._health_registry`.

The tick loop is driven with a stub snapshot, a stub tick and an injected
sleep, which ends a test by raising `_StopLoopError` once it has seen enough.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import inspect
import logging
import re
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend import server
from backend.errors import MistError
from backend.event_store import status_intervals

_SCHEMA_PATH = Path(status_intervals.__file__).parent / "schema.sql"
_FIVE = ("healthy_s", "degraded_s", "yielded_s", "unhealthy_s", "unrecorded_s")
_SESSION_ID = "session-7f3a-private"
_UTTERANCE = "remind me about the dentist on thursday"


def _make_store(path: Path) -> Path:
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()
    conn.close()
    return path


def _insert(path: Path, status: str, start: datetime, end: datetime) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO instance_status_intervals "
        "(status, degraded_beneath, started_at, last_seen_at) VALUES (?, 0, ?, ?)",
        (status, start.isoformat(), end.isoformat()),
    )
    conn.commit()
    conn.close()


class _StubRegistry:
    def __init__(self, status: str) -> None:
        self.status = status

    def snapshot(self) -> dict:
        return {"status": self.status, "probe_loop": "alive", "checks": {}}


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = _make_store(tmp_path / "event_store.db")
    monkeypatch.setenv("EVENT_STORE_DB_PATH", str(path))
    monkeypatch.setenv("MIST_TIMEZONE", "UTC")
    monkeypatch.setattr(server, "_health_registry", _StubRegistry("healthy"))
    return path


@pytest.fixture
def client() -> TestClient:
    return TestClient(server.app)


# --- the route ------------------------------------------------------------------


class TestUptimeRoute:
    def test_returns_the_contract_shape(self, store: Path, client: TestClient) -> None:
        response = client.get("/usage/uptime")

        assert response.status_code == 200
        body = response.json()
        assert set(body) == {"timezone", "current", "days"}
        assert body["timezone"] == "UTC"
        assert len(body["days"]) == 7
        assert all(set(day) == {"date", *_FIVE, "yielded_degraded_s"} for day in body["days"])
        assert all(isinstance(day[key], int) for day in body["days"] for key in _FIVE)

    def test_current_is_the_live_registry_status(
        self, store: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(server, "_health_registry", _StubRegistry("yielded"))

        assert client.get("/usage/uptime").json()["current"] == "yielded"

    def test_counts_stored_intervals_and_full_days_sum_to_the_day(
        self, store: Path, client: TestClient
    ) -> None:
        # Whole seconds: each day's share is floored, so an interval with a
        # fraction of a second that crosses midnight would lose up to 1 s.
        now = datetime.now(UTC).replace(microsecond=0)
        _insert(store, "degraded", now - timedelta(days=3, hours=2), now - timedelta(days=3))

        body = client.get("/usage/uptime", params={"days": 10}).json()

        assert sum(day["degraded_s"] for day in body["days"]) == 7200
        assert all(sum(day[key] for key in _FIVE) == 86400 for day in body["days"][:-1])

    def test_days_are_dense_oldest_first_and_end_today(
        self, store: Path, client: TestClient
    ) -> None:
        before = datetime.now(UTC).date()
        body = client.get("/usage/uptime", params={"days": 10}).json()
        after = datetime.now(UTC).date()

        dates = [datetime.strptime(d["date"], "%Y-%m-%d").date() for d in body["days"]]
        assert len(dates) == 10
        assert all(b - a == timedelta(days=1) for a, b in zip(dates, dates[1:], strict=False))
        assert dates[-1] in {before, after}

    @pytest.mark.parametrize("days", [1, 90])
    def test_days_at_the_bounds_are_accepted(
        self, store: Path, client: TestClient, days: int
    ) -> None:
        response = client.get("/usage/uptime", params={"days": days})

        assert response.status_code == 200
        assert len(response.json()["days"]) == days

    @pytest.mark.parametrize("days", ["0", "-1", "91", "seven"])
    def test_days_outside_one_to_ninety_is_422(
        self, store: Path, client: TestClient, days: str
    ) -> None:
        assert client.get("/usage/uptime", params={"days": days}).status_code == 422

    def test_missing_store_is_503_and_is_not_created(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        client: TestClient,
    ) -> None:
        missing = tmp_path / "nowhere" / "event_store.db"
        monkeypatch.setenv("EVENT_STORE_DB_PATH", str(missing))
        monkeypatch.setattr(server, "_health_registry", _StubRegistry("healthy"))

        response = client.get("/usage/uptime")

        assert response.status_code == 503
        assert response.json() == {"error": "store_unavailable"}
        assert not missing.exists()

    def test_locked_store_is_503(self, store: Path, client: TestClient) -> None:
        holder = sqlite3.connect(store, isolation_level=None)
        holder.execute("PRAGMA locking_mode=EXCLUSIVE")
        holder.execute("BEGIN EXCLUSIVE")
        try:
            response = client.get("/usage/uptime")
        finally:
            holder.execute("ROLLBACK")
            holder.close()

        assert response.status_code == 503
        assert response.json() == {"error": "store_unavailable"}

    def test_unknown_zone_falls_back_to_utc_and_is_logged(
        self,
        store: Path,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setenv("MIST_TIMEZONE", "Atlantis/Capital")

        with caplog.at_level(logging.WARNING):
            response = client.get("/usage/uptime")

        assert response.status_code == 200
        assert response.json()["timezone"] == "UTC"
        assert "Atlantis/Capital" in caplog.text

    def test_configured_zone_is_reported(
        self, store: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MIST_TIMEZONE", "Asia/Tokyo")

        assert client.get("/usage/uptime").json()["timezone"] == "Asia/Tokyo"

    def test_the_old_zone_variable_is_ignored(
        self, store: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MIST_TIMEZONE")
        monkeypatch.setenv("MIST_USAGE_TIMEZONE", "Asia/Tokyo")

        assert client.get("/usage/uptime").json()["timezone"] == "UTC"

    def test_response_holds_no_session_id_utterance_or_time_of_day(
        self, store: Path, client: TestClient
    ) -> None:
        conn = sqlite3.connect(store)
        conn.execute(
            "INSERT INTO conversation_sessions (session_id, started_at, origin) VALUES (?, ?, ?)",
            (_SESSION_ID, "2026-01-01T00:00:00+00:00", "real"),
        )
        conn.execute(
            "INSERT INTO conversation_turn_events "
            "(event_id, session_id, turn_index, timestamp, user_utterance, system_response) "
            "VALUES ('e1', ?, 0, ?, ?, 'a reply')",
            (_SESSION_ID, datetime.now(UTC).isoformat(), _UTTERANCE),
        )
        conn.commit()
        conn.close()
        now = datetime.now(UTC)
        _insert(store, "healthy", now - timedelta(hours=30), now - timedelta(hours=1))

        response = client.get("/usage/uptime", params={"days": 90})
        text = response.text

        assert response.status_code == 200
        for secret in (_SESSION_ID, _UTTERANCE, "session", "utterance", "timestamp"):
            assert secret not in text
        assert not re.search(r"\d{1,2}:\d{2}", text)
        assert not re.search(r"\d{4}-\d{2}-\d{2}T", text)
        assert all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", d["date"]) for d in response.json()["days"])

    def test_route_docstring_states_the_counts_only_contract(self) -> None:
        doc = " ".join((server.usage_uptime.__doc__ or "").split())

        assert "counts only" in doc.lower()
        assert "unrecorded, never up" in doc


# --- the tick loop --------------------------------------------------------------


class _StopLoopError(Exception):
    """Raised by the injected sleep to end a loop under test."""


def _measured(age: float | None) -> dict:
    return {"status": "up", "age_seconds": age, "stale": age is None}


class _Snapshots:
    """Answers unmeasured for the first `unmeasured` calls, then measured."""

    def __init__(self, unmeasured: int, status: str = "healthy") -> None:
        self.calls = 0
        self.unmeasured = unmeasured
        self.status = status

    def __call__(self) -> dict:
        self.calls += 1
        age = None if self.calls <= self.unmeasured else 1.0
        return {
            "status": self.status,
            "checks": {"llm": _measured(1.0), "neo4j": _measured(age)},
        }


class _Sleeper:
    """Records each sleep; raises `_StopLoopError` on the `stop_after`-th period sleep."""

    def __init__(self, period: float, stop_after: int) -> None:
        self.period = period
        self.stop_after = stop_after
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        if seconds == self.period and self.calls.count(self.period) >= self.stop_after:
            raise _StopLoopError


class _Ticks:
    """Records each tick's arguments; raises the queued errors in order."""

    def __init__(self, errors: list[BaseException | None] | None = None) -> None:
        self.calls: list[dict] = []
        self.threads: list[int] = []
        self.errors = list(errors or [])

    def __call__(self, db_path, status, beneath, now, *, max_gap_s, first_tick) -> None:
        self.threads.append(threading.get_ident())
        self.calls.append(
            {
                "db_path": db_path,
                "status": status,
                "beneath": beneath,
                "now": now,
                "max_gap_s": max_gap_s,
                "first_tick": first_tick,
            }
        )
        error = self.errors.pop(0) if self.errors else None
        if error is not None:
            raise error


_NOW = datetime(2026, 10, 20, 10, 0, tzinfo=UTC)


async def _run(snapshots, ticks: _Ticks, sleeper: _Sleeper, period: float = 60.0) -> None:
    with pytest.raises(_StopLoopError):
        await server.uptime_record_loop(
            "/data/event_store.db",
            snapshots,
            period_s=period,
            tick=ticks,
            sleep=sleeper,
            clock=lambda: _NOW,
        )


class TestUptimeRecordLoop:
    @pytest.mark.asyncio
    async def test_first_tick_waits_until_every_check_is_measured(self) -> None:
        ticks, sleeper = _Ticks(), _Sleeper(60.0, stop_after=2)

        await _run(_Snapshots(unmeasured=3), ticks, sleeper)

        assert sleeper.calls == [1.0, 1.0, 1.0, 60.0, 60.0]
        assert [c["first_tick"] for c in ticks.calls] == [True, False]

    @pytest.mark.asyncio
    async def test_first_tick_runs_anyway_after_sixty_seconds(self) -> None:
        ticks, sleeper = _Ticks(), _Sleeper(60.0, stop_after=1)

        await _run(_Snapshots(unmeasured=10_000), ticks, sleeper)

        assert sleeper.calls == [1.0] * 60 + [60.0]
        assert [c["first_tick"] for c in ticks.calls] == [True]

    @pytest.mark.asyncio
    async def test_ticks_carry_status_path_gap_and_run_in_a_worker_thread(self) -> None:
        ticks, sleeper = _Ticks(), _Sleeper(40.0, stop_after=2)

        await _run(_Snapshots(unmeasured=0, status="degraded"), ticks, sleeper, period=40.0)

        assert [c["status"] for c in ticks.calls] == ["degraded", "degraded"]
        assert {c["db_path"] for c in ticks.calls} == {"/data/event_store.db"}
        assert {c["max_gap_s"] for c in ticks.calls} == {100.0}
        assert {c["now"] for c in ticks.calls} == {_NOW}
        assert {c["beneath"] for c in ticks.calls} == {False}
        assert threading.get_ident() not in ticks.threads

    @pytest.mark.asyncio
    async def test_degraded_beneath_is_taken_from_the_snapshot(self) -> None:
        def snapshot() -> dict:
            return {
                "status": "yielded",
                "checks": {
                    "llm": {"status": "yielded", "age_seconds": 1.0, "stale": False},
                    "backup": {"status": "down", "age_seconds": 1.0, "stale": False},
                },
            }

        ticks, sleeper = _Ticks(), _Sleeper(60.0, stop_after=1)

        await _run(snapshot, ticks, sleeper)

        assert ticks.calls[0]["status"] == "yielded"
        assert ticks.calls[0]["beneath"] is True

    @pytest.mark.asyncio
    async def test_keeps_looping_after_a_sqlite_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        ticks = _Ticks(errors=[None, sqlite3.OperationalError("database is locked")])
        sleeper = _Sleeper(60.0, stop_after=3)

        with caplog.at_level(logging.WARNING, logger=server.logger.name):
            await _run(_Snapshots(unmeasured=0), ticks, sleeper)

        assert len(ticks.calls) == 3
        assert "database is locked" in caplog.text

    @pytest.mark.asyncio
    async def test_keeps_looping_after_a_mist_error(self) -> None:
        ticks = _Ticks(errors=[None, MistError("store went away")])
        sleeper = _Sleeper(60.0, stop_after=3)

        await _run(_Snapshots(unmeasured=0), ticks, sleeper)

        assert len(ticks.calls) == 3

    @pytest.mark.asyncio
    async def test_a_failed_first_tick_leaves_the_next_one_first(self) -> None:
        ticks = _Ticks(errors=[sqlite3.OperationalError("database is locked")])
        sleeper = _Sleeper(60.0, stop_after=3)

        await _run(_Snapshots(unmeasured=0), ticks, sleeper)

        assert [c["first_tick"] for c in ticks.calls] == [True, True, False]

    @pytest.mark.asyncio
    async def test_writes_a_real_store_through_the_real_tick(self, tmp_path: Path) -> None:
        path = _make_store(tmp_path / "event_store.db")
        clock_values = iter([_NOW, _NOW + timedelta(seconds=60)])
        sleeper = _Sleeper(60.0, stop_after=2)

        with pytest.raises(_StopLoopError):
            await server.uptime_record_loop(
                str(path),
                _Snapshots(unmeasured=0),
                period_s=60.0,
                sleep=sleeper,
                clock=lambda: next(clock_values),
            )

        conn = sqlite3.connect(path)
        rows = conn.execute(
            "SELECT status, started_at, last_seen_at FROM instance_status_intervals"
        ).fetchall()
        conn.close()
        assert rows == [
            ("healthy", _NOW.isoformat(), (_NOW + timedelta(seconds=60)).isoformat()),
        ]


class _StubEventStore:
    def __init__(self, db_path) -> None:
        self.db_path = db_path


class TestBuildAndStop:
    def test_no_event_store_builds_no_recorder(self) -> None:
        assert server._build_uptime_recorder(None, server._health_snapshot, 60.0) is None

    def test_an_in_memory_event_store_builds_no_recorder(self) -> None:
        store = _StubEventStore(":memory:")

        assert server._build_uptime_recorder(store, server._health_snapshot, 60.0) is None

    def test_the_recorder_uses_the_event_stores_database_path(self, tmp_path: Path) -> None:
        path = tmp_path / "event_store.db"

        recorder = server._build_uptime_recorder(
            _StubEventStore(path), server._health_snapshot, 45.0
        )

        assert isinstance(recorder, functools.partial)
        assert recorder.func is server.uptime_record_loop
        assert recorder.args == (str(path), server._health_snapshot)
        assert recorder.keywords == {"period_s": 45.0}

    @pytest.mark.asyncio
    async def test_stop_cancels_and_awaits_a_running_loop(self) -> None:
        async def forever() -> None:
            await asyncio.sleep(3600)

        task = asyncio.create_task(forever())
        await asyncio.sleep(0)

        await server._stop_uptime_recorder(task)

        assert task.cancelled()

    @pytest.mark.asyncio
    async def test_stop_is_a_no_op_without_a_task(self) -> None:
        await server._stop_uptime_recorder(None)

    def test_lifespan_starts_the_recorder_and_stops_it_at_shutdown(self) -> None:
        source = inspect.getsource(server.lifespan)
        startup, _, shutdown = source.partition("\n    yield\n")

        assert "_build_uptime_recorder(" in startup
        assert "uptime_record_s()" in startup
        assert "await _stop_uptime_recorder(uptime_task)" in shutdown

    def test_done_callback_ignores_a_cancel(self) -> None:
        loop = asyncio.new_event_loop()
        try:
            task = loop.create_task(asyncio.sleep(3600))
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                loop.run_until_complete(task)
            server._on_uptime_loop_done(task)
        finally:
            loop.close()


class TestRecordPeriod:
    def test_default_is_sixty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MIST_UPTIME_RECORD_S", raising=False)

        assert server.uptime_record_s() == 60.0

    def test_a_positive_value_is_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MIST_UPTIME_RECORD_S", "15")

        assert server.uptime_record_s() == 15.0

    @pytest.mark.parametrize("raw", ["0", "-5", "soon"])
    def test_a_bad_value_is_logged_and_replaced(
        self, raw: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("MIST_UPTIME_RECORD_S", raw)

        with caplog.at_level(logging.ERROR, logger=server.logger.name):
            assert server.uptime_record_s() == 60.0

        assert "MIST_UPTIME_RECORD_S" in caplog.text
