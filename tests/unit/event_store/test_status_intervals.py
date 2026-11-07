"""Tests for backend/event_store/status_intervals.py -- the uptime ledger.

Every store is synthetic: built in `tmp_path` from the real `schema.sql` (or
through `EventStore.initialize()`), so the table and its CHECK constraints are
the store's own and nothing reads `~/.mist`.

Every `now` is injected. Expected values are bare literals worked out by hand
from the rows each test writes, never produced by the code under test.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import pytz

from backend.errors import MistError
from backend.event_store import status_intervals
from backend.event_store.status_intervals import (
    StatusInterval,
    UptimeStoreUnavailableError,
    build_uptime_per_day,
    degraded_beneath,
    read_status_intervals,
    record_status_tick,
    uptime_per_day,
)
from backend.event_store.store import EventStore

_SCHEMA_PATH = Path(status_intervals.__file__).parent / "schema.sql"
_GAP = 150.0
_T0 = datetime(2026, 10, 20, 10, 0, 0, tzinfo=UTC)
_FIVE = ("healthy_s", "degraded_s", "yielded_s", "unhealthy_s", "unrecorded_s")


def _make_store(path: Path) -> Path:
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()
    conn.close()
    return path


def _rows(path: Path) -> list[tuple]:
    conn = sqlite3.connect(path)
    try:
        return conn.execute(
            "SELECT id, status, degraded_beneath, started_at, last_seen_at "
            "FROM instance_status_intervals ORDER BY id"
        ).fetchall()
    finally:
        conn.close()


def _insert(path: Path, status: str, start: datetime, end: datetime, beneath: int = 0) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO instance_status_intervals "
        "(status, degraded_beneath, started_at, last_seen_at) VALUES (?, ?, ?, ?)",
        (status, beneath, start.isoformat(), end.isoformat()),
    )
    conn.commit()
    conn.close()


def _tick(path: Path, status: str, now: datetime, *, beneath=False, first=False) -> None:
    record_status_tick(str(path), status, beneath, now, max_gap_s=_GAP, first_tick=first)


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat()


@pytest.fixture
def store(tmp_path: Path) -> Path:
    return _make_store(tmp_path / "event_store.db")


# --- schema ---------------------------------------------------------------------


class TestSchema:
    def test_initialize_twice_is_idempotent(self, tmp_path: Path) -> None:
        path = tmp_path / "event_store.db"
        es = EventStore(db_path=str(path))
        es.initialize()
        es.initialize()
        es.close()

        _tick(path, "healthy", _T0, first=True)

        assert len(_rows(path)) == 1

    def test_initialize_adds_the_table_and_keeps_existing_rows(self, tmp_path: Path) -> None:
        path = tmp_path / "event_store.db"
        es = EventStore(db_path=str(path))
        es.initialize()
        es.start_session("s-before", origin="real")
        es.close()
        conn = sqlite3.connect(path)
        conn.execute("DROP TABLE instance_status_intervals")
        conn.commit()
        conn.close()

        again = EventStore(db_path=str(path))
        again.initialize()
        again.close()

        conn = sqlite3.connect(path)
        try:
            sessions = conn.execute("SELECT session_id FROM conversation_sessions").fetchall()
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE name = 'instance_status_intervals'"
            ).fetchall()
        finally:
            conn.close()
        assert sessions == [("s-before",)]
        assert tables == [("instance_status_intervals",)]

    def test_check_constraints_refuse_an_unknown_status(self, store: Path) -> None:
        with pytest.raises(sqlite3.IntegrityError):
            _insert(store, "asleep", _T0, _T0)


# --- writer ---------------------------------------------------------------------


class TestRecordStatusTick:
    def test_first_tick_on_an_empty_store_inserts_a_row(self, store: Path) -> None:
        _tick(store, "healthy", _T0)

        assert _rows(store) == [(1, "healthy", 0, _iso(_T0), _iso(_T0))]

    def test_first_tick_within_the_gap_still_inserts_so_a_restart_leaves_a_gap(
        self, store: Path
    ) -> None:
        _tick(store, "healthy", _T0, first=True)
        restart = _T0 + timedelta(seconds=30)

        _tick(store, "healthy", restart, first=True)

        assert _rows(store) == [
            (1, "healthy", 0, _iso(_T0), _iso(_T0)),
            (2, "healthy", 0, _iso(restart), _iso(restart)),
        ]

    def test_same_status_within_the_gap_extends_the_row(self, store: Path) -> None:
        _tick(store, "degraded", _T0, first=True)
        later = _T0 + timedelta(seconds=60)

        _tick(store, "degraded", later)

        assert _rows(store) == [(1, "degraded", 0, _iso(_T0), _iso(later))]

    def test_same_status_after_a_longer_gap_inserts_a_row_and_leaves_the_gap(
        self, store: Path
    ) -> None:
        _tick(store, "healthy", _T0, first=True)
        later = _T0 + timedelta(seconds=151)

        _tick(store, "healthy", later)

        assert _rows(store) == [
            (1, "healthy", 0, _iso(_T0), _iso(_T0)),
            (2, "healthy", 0, _iso(later), _iso(later)),
        ]

    def test_a_status_change_within_the_gap_starts_where_the_last_row_ends(
        self, store: Path
    ) -> None:
        _tick(store, "healthy", _T0, first=True)
        seen = _T0 + timedelta(seconds=60)
        _tick(store, "healthy", seen)
        changed = _T0 + timedelta(seconds=120)

        _tick(store, "yielded", changed)

        assert _rows(store) == [
            (1, "healthy", 0, _iso(_T0), _iso(seen)),
            (2, "yielded", 0, _iso(seen), _iso(changed)),
        ]

    def test_a_degraded_beneath_change_within_the_gap_starts_where_the_last_row_ends(
        self, store: Path
    ) -> None:
        _tick(store, "yielded", _T0, first=True)
        changed = _T0 + timedelta(seconds=60)

        _tick(store, "yielded", changed, beneath=True)

        assert _rows(store) == [
            (1, "yielded", 0, _iso(_T0), _iso(_T0)),
            (2, "yielded", 1, _iso(_T0), _iso(changed)),
        ]

    def test_a_clock_step_back_inserts_a_row_and_leaves_the_newest_unchanged(
        self, store: Path
    ) -> None:
        _tick(store, "healthy", _T0, first=True)
        seen = _T0 + timedelta(seconds=60)
        _tick(store, "healthy", seen)
        stepped_back = _T0 + timedelta(seconds=30)

        _tick(store, "healthy", stepped_back)

        assert _rows(store) == [
            (1, "healthy", 0, _iso(_T0), _iso(seen)),
            (2, "healthy", 0, _iso(stepped_back), _iso(stepped_back)),
        ]

    def test_an_unknown_status_raises_before_any_write(self, store: Path) -> None:
        _tick(store, "healthy", _T0, first=True)

        with pytest.raises(ValueError, match="asleep"):
            _tick(store, "asleep", _T0 + timedelta(seconds=10))

        assert _rows(store) == [(1, "healthy", 0, _iso(_T0), _iso(_T0))]

    def test_an_unknown_status_does_not_open_the_store(self, tmp_path: Path) -> None:
        missing = tmp_path / "never.db"

        with pytest.raises(ValueError):
            _tick(missing, "asleep", _T0)

        assert not missing.exists()

    def test_a_naive_now_raises_before_any_write(self, store: Path) -> None:
        with pytest.raises(ValueError, match="aware"):
            _tick(store, "healthy", datetime(2026, 10, 20, 10, 0, 0))

        assert _rows(store) == []

    def test_a_tick_waits_out_a_competing_writer_and_then_succeeds(self, store: Path) -> None:
        holder = sqlite3.connect(store, isolation_level=None, check_same_thread=False)
        holder.execute("BEGIN IMMEDIATE")
        released = threading.Event()

        def release() -> None:
            holder.execute("COMMIT")
            holder.close()
            released.set()

        timer = threading.Timer(1.0, release)
        started = time.monotonic()
        timer.start()
        try:
            _tick(store, "healthy", _T0, first=True)
        finally:
            timer.join()
        waited = time.monotonic() - started

        assert released.is_set()
        assert waited >= 0.9
        assert _rows(store) == [(1, "healthy", 0, _iso(_T0), _iso(_T0))]

    def test_a_tick_opens_and_closes_its_own_connection_beside_an_event_store(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "event_store.db"
        es = EventStore(db_path=str(path))
        es.initialize()
        shared = es._get_connection()
        opened: list[sqlite3.Connection] = []
        real_connect = sqlite3.connect

        def spy(*args, **kwargs):
            conn = real_connect(*args, **kwargs)
            opened.append(conn)
            return conn

        monkeypatch.setattr(status_intervals.sqlite3, "connect", spy)
        _tick(path, "healthy", _T0, first=True)
        monkeypatch.undo()

        assert len(opened) == 1
        assert opened[0] is not shared
        with pytest.raises(sqlite3.ProgrammingError):
            opened[0].execute("SELECT 1")
        es.start_session("s-after", origin="real")
        assert es.get_session("s-after") is not None
        es.close()


# --- degraded_beneath -----------------------------------------------------------


def _check(status: str | None, severity: str = "degrades", stale: bool = False) -> dict:
    return {
        "status": status,
        "reason": "ok",
        "age_seconds": None if status is None else 1.0,
        "stale": stale,
        "latency_ms": 1.0,
        "consecutive_failures": 0,
        "severity": severity,
        "restart_repairs": False,
    }


def _snapshot(status: str, **checks: dict) -> dict:
    return {"status": status, "probe_loop": "alive", "checks": checks}


class TestDegradedBeneath:
    def test_true_for_a_yielded_snapshot_with_a_degrading_check_down(self) -> None:
        snap = _snapshot(
            "yielded",
            llm=_check("yielded", "fatal"),
            neo4j=_check("up", "fatal"),
            backup=_check("down"),
        )

        assert degraded_beneath(snap) is True

    def test_false_for_a_yielded_snapshot_with_every_other_check_up(self) -> None:
        snap = _snapshot(
            "yielded",
            llm=_check("yielded", "fatal"),
            neo4j=_check("up", "fatal"),
            backup=_check("up"),
        )

        assert degraded_beneath(snap) is False

    @pytest.mark.parametrize("status", ["healthy", "degraded", "unhealthy"])
    def test_false_for_every_non_yielded_snapshot(self, status: str) -> None:
        snap = _snapshot(status, llm=_check("down", "fatal"), backup=_check("down"))

        assert degraded_beneath(snap) is False


# --- reader ---------------------------------------------------------------------


def _day(result, iso_date: str) -> dict:
    for entry in result.to_payload()["days"]:
        if entry["date"] == iso_date:
            return entry
    raise AssertionError(f"{iso_date} not in the window")


def _five_sum(entry: dict) -> int:
    return sum(entry[key] for key in _FIVE)


class TestReader:
    def test_an_interval_across_local_midnight_is_split(self, store: Path) -> None:
        _insert(store, "healthy", datetime(2026, 10, 19, 23, 0, tzinfo=UTC), _T0)

        result = build_uptime_per_day(
            str(store), days=2, current="healthy", timezone_name="UTC", now=_T0
        )

        assert _day(result, "2026-10-19")["healthy_s"] == 3600
        assert _day(result, "2026-10-20")["healthy_s"] == 36000

    def test_split_follows_the_configured_zone(self, store: Path) -> None:
        # 03:30Z-04:30Z is 23:30-00:30 in New York (EDT, -4).
        _insert(
            store,
            "degraded",
            datetime(2026, 10, 20, 3, 30, tzinfo=UTC),
            datetime(2026, 10, 20, 4, 30, tzinfo=UTC),
        )

        result = build_uptime_per_day(
            str(store), days=2, current="healthy", timezone_name="America/New_York", now=_T0
        )

        assert result.timezone == "America/New_York"
        assert _day(result, "2026-10-19")["degraded_s"] == 1800
        assert _day(result, "2026-10-20")["degraded_s"] == 1800

    def test_spring_forward_day_sums_to_82800(self, store: Path) -> None:
        zone = pytz.timezone("America/New_York")
        _insert(
            store,
            "healthy",
            zone.localize(datetime(2026, 3, 8, 1, 0)),
            zone.localize(datetime(2026, 3, 8, 4, 0)),
        )
        now = zone.localize(datetime(2026, 3, 9, 12, 0))

        result = build_uptime_per_day(
            str(store), days=2, current="healthy", timezone_name="America/New_York", now=now
        )

        day = _day(result, "2026-03-08")
        # 01:00 EST to 04:00 EDT is two real hours.
        assert day["healthy_s"] == 7200
        assert _five_sum(day) == 82800

    def test_fall_back_day_sums_to_90000(self, store: Path) -> None:
        zone = pytz.timezone("America/New_York")
        _insert(
            store,
            "yielded",
            zone.localize(datetime(2026, 11, 1, 0, 30)),
            zone.localize(datetime(2026, 11, 1, 3, 0)),
        )
        now = zone.localize(datetime(2026, 11, 2, 12, 0))

        result = build_uptime_per_day(
            str(store), days=2, current="healthy", timezone_name="America/New_York", now=now
        )

        day = _day(result, "2026-11-01")
        # 00:30 EDT to 03:00 EST is three and a half real hours.
        assert day["yielded_s"] == 12600
        assert _five_sum(day) == 90000

    def test_a_day_with_no_rows_is_all_unrecorded_and_today_sums_to_elapsed(
        self, store: Path
    ) -> None:
        _insert(store, "healthy", datetime(2026, 10, 20, 9, 0, tzinfo=UTC), _T0)

        result = build_uptime_per_day(
            str(store), days=3, current="healthy", timezone_name="UTC", now=_T0
        )

        payload = result.to_payload()
        assert [d["date"] for d in payload["days"]] == ["2026-10-18", "2026-10-19", "2026-10-20"]
        empty = _day(result, "2026-10-18")
        assert empty["unrecorded_s"] == 86400
        assert _five_sum(empty) == 86400
        today = _day(result, "2026-10-20")
        assert _five_sum(today) == 36000
        assert today["healthy_s"] == 3600
        assert today["unrecorded_s"] == 32400

    def test_time_after_the_newest_last_seen_is_unrecorded(self, store: Path) -> None:
        _insert(
            store,
            "healthy",
            datetime(2026, 10, 20, 8, 0, tzinfo=UTC),
            datetime(2026, 10, 20, 9, 0, tzinfo=UTC),
        )

        today = _day(
            build_uptime_per_day(
                str(store), days=1, current="healthy", timezone_name="UTC", now=_T0
            ),
            "2026-10-20",
        )

        assert today["healthy_s"] == 3600
        assert today["unrecorded_s"] == 36000 - 3600

    def test_overlapping_rows_count_their_overlap_once(self, store: Path) -> None:
        # Row 2 was written after the clock stepped back 30 minutes.
        _insert(
            store,
            "healthy",
            datetime(2026, 10, 20, 6, 0, tzinfo=UTC),
            datetime(2026, 10, 20, 7, 0, tzinfo=UTC),
        )
        _insert(
            store,
            "degraded",
            datetime(2026, 10, 20, 6, 30, tzinfo=UTC),
            datetime(2026, 10, 20, 7, 30, tzinfo=UTC),
        )
        # Row 3 lies wholly inside time already counted: a negative duration.
        _insert(
            store,
            "unhealthy",
            datetime(2026, 10, 20, 6, 40, tzinfo=UTC),
            datetime(2026, 10, 20, 6, 50, tzinfo=UTC),
        )

        today = _day(
            build_uptime_per_day(
                str(store), days=1, current="healthy", timezone_name="UTC", now=_T0
            ),
            "2026-10-20",
        )

        assert today["healthy_s"] == 3600
        assert today["degraded_s"] == 1800
        assert today["unhealthy_s"] == 0
        assert _five_sum(today) == 36000

    def test_yielded_degraded_counts_only_yielded_rows_with_degraded_beneath(
        self, store: Path
    ) -> None:
        base = datetime(2026, 10, 20, 0, 0, tzinfo=UTC)
        _insert(store, "yielded", base, base + timedelta(hours=1), beneath=0)
        _insert(store, "yielded", base + timedelta(hours=1), base + timedelta(hours=3), beneath=1)
        _insert(store, "degraded", base + timedelta(hours=3), base + timedelta(hours=4))

        today = _day(
            build_uptime_per_day(
                str(store), days=1, current="yielded", timezone_name="UTC", now=_T0
            ),
            "2026-10-20",
        )

        assert today["yielded_s"] == 10800
        assert today["yielded_degraded_s"] == 7200
        assert today["degraded_s"] == 3600
        assert today["yielded_degraded_s"] <= today["yielded_s"]

    def test_payload_shape_and_current(self, store: Path) -> None:
        result = build_uptime_per_day(
            str(store), days=1, current="degraded", timezone_name="UTC", now=_T0
        )

        payload = result.to_payload()
        assert set(payload) == {"timezone", "current", "days"}
        assert payload["current"] == "degraded"
        assert set(payload["days"][0]) == {"date", *_FIVE, "yielded_degraded_s"}

    def test_unknown_zone_falls_back_to_utc(self, store: Path) -> None:
        result = build_uptime_per_day(
            str(store), days=1, current="healthy", timezone_name="Atlantis/Capital", now=_T0
        )

        assert result.timezone == "UTC"

    @pytest.mark.parametrize("days", [0, 91])
    def test_days_outside_one_to_ninety_raise(self, store: Path, days: int) -> None:
        with pytest.raises(ValueError):
            build_uptime_per_day(
                str(store), days=days, current="healthy", timezone_name="UTC", now=_T0
            )

    def test_a_missing_store_is_unavailable_and_not_created(self, tmp_path: Path) -> None:
        missing = tmp_path / "nowhere" / "event_store.db"

        with pytest.raises(UptimeStoreUnavailableError) as raised:
            read_status_intervals(str(missing))

        assert isinstance(raised.value, MistError)
        assert not missing.exists()

    def test_a_store_without_the_table_is_unavailable(self, tmp_path: Path) -> None:
        path = tmp_path / "other.db"
        sqlite3.connect(path).close()

        with pytest.raises(UptimeStoreUnavailableError):
            read_status_intervals(str(path))

    def test_a_path_with_uri_syntax_is_opened_as_a_file(self, tmp_path: Path) -> None:
        odd = tmp_path / "odd#dir?x"
        odd.mkdir()
        path = _make_store(odd / "event_store.db")
        _insert(path, "healthy", _T0 - timedelta(hours=1), _T0)

        intervals = read_status_intervals(str(path))

        assert intervals == [
            StatusInterval("healthy", False, _T0 - timedelta(hours=1), _T0),
        ]

    def test_uptime_per_day_is_dense_and_ends_today(self) -> None:
        days = uptime_per_day([], pytz.utc, now=_T0, days=5)

        assert [d.date.isoformat() for d in days] == [
            "2026-10-16",
            "2026-10-17",
            "2026-10-18",
            "2026-10-19",
            "2026-10-20",
        ]
