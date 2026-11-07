"""Tests for backend/event_store/usage.py -- real turns per local day, counts only.

Every store here is synthetic: built in `tmp_path` from the real `schema.sql`, so
the column names and the `origin` default are the store's own, and nothing reads
`~/.mist` or any other live data.

Expected values are bare literals derived by hand from the rows each test
inserts, never produced by calling the code under test.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from backend.errors import MistError
from backend.event_store import usage
from backend.event_store.usage import (
    DayCount,
    UsageStoreUnavailableError,
    build_turns_per_day,
    count_turns_per_day,
    local_date,
    read_real_turn_timestamps,
    resolve_store_path,
    resolve_timezone,
)

_SCHEMA_PATH = Path(usage.__file__).parent / "schema.sql"


def _make_store(path: Path) -> Path:
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()
    conn.close()
    return path


def _add_session(path: Path, session_id: str, origin: str) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO conversation_sessions (session_id, started_at, origin) VALUES (?, ?, ?)",
        (session_id, "2026-01-01T00:00:00+00:00", origin),
    )
    conn.commit()
    conn.close()


def _add_turn(path: Path, session_id: str, timestamp: str, utterance: str = "hello there") -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO conversation_turn_events "
        "(event_id, session_id, turn_index, timestamp, user_utterance, system_response) "
        "VALUES (?, ?, 0, ?, ?, ?)",
        (str(uuid.uuid4()), session_id, timestamp, utterance, "a reply"),
    )
    conn.commit()
    conn.close()


@pytest.fixture
def store(tmp_path: Path) -> Path:
    return _make_store(tmp_path / "event_store.db")


# --- origin rule ---------------------------------------------------------------


def test_only_real_origin_turns_are_read_and_orphans_count_as_real(store: Path) -> None:
    _add_session(store, "s-real", "real")
    _add_session(store, "s-test", "test")
    _add_session(store, "s-seed", "seed")
    _add_turn(store, "s-real", "2026-10-20T10:00:00+00:00")
    _add_turn(store, "s-test", "2026-10-20T11:00:00+00:00")
    _add_turn(store, "s-seed", "2026-10-20T12:00:00+00:00")
    # No session row at all: the store's rule is COALESCE(origin, 'real').
    _add_turn(store, "s-orphan", "2026-10-20T13:00:00+00:00")

    stamps = read_real_turn_timestamps(str(store))

    assert sorted(stamps) == ["2026-10-20T10:00:00+00:00", "2026-10-20T13:00:00+00:00"]


def test_build_counts_exclude_test_and_seed_turns(store: Path) -> None:
    _add_session(store, "s-real", "real")
    _add_session(store, "s-test", "test")
    _add_session(store, "s-seed", "seed")
    for hour in (9, 10, 11):
        _add_turn(store, "s-test", f"2026-10-23T{hour:02d}:00:00+00:00")
        _add_turn(store, "s-seed", f"2026-10-23T{hour:02d}:30:00+00:00")
    _add_turn(store, "s-real", "2026-10-23T08:00:00+00:00")
    _add_turn(store, "s-orphan", "2026-10-23T08:30:00+00:00")

    result = build_turns_per_day(
        str(store), days=1, timezone_name="UTC", now=datetime(2026, 10, 23, 20, tzinfo=UTC)
    )

    assert result.days == (DayCount(date(2026, 10, 23), 2),)


# --- read-only access ----------------------------------------------------------


def test_reading_never_creates_a_missing_store(tmp_path: Path) -> None:
    missing = tmp_path / "absent" / "event_store.db"

    with pytest.raises(UsageStoreUnavailableError):
        read_real_turn_timestamps(str(missing))

    assert not missing.exists()
    assert not missing.parent.exists()


def test_store_unavailable_is_a_mist_error() -> None:
    assert issubclass(UsageStoreUnavailableError, MistError)


def test_a_store_without_the_turn_table_is_unavailable(tmp_path: Path) -> None:
    empty = tmp_path / "empty.db"
    sqlite3.connect(empty).close()

    with pytest.raises(UsageStoreUnavailableError):
        read_real_turn_timestamps(str(empty))


def test_a_wal_store_held_in_exclusive_locking_mode_is_unavailable(store: Path) -> None:
    # `schema.sql` puts the store in WAL mode, where an ordinary write
    # transaction does not block readers. A connection in exclusive locking
    # mode does, and that is the lock a WAL reader can actually meet.
    holder = sqlite3.connect(store, isolation_level=None)
    holder.execute("PRAGMA locking_mode=EXCLUSIVE")
    holder.execute("BEGIN EXCLUSIVE")
    try:
        with pytest.raises(UsageStoreUnavailableError):
            read_real_turn_timestamps(str(store), busy_timeout_seconds=0.05)
    finally:
        holder.execute("ROLLBACK")
        holder.close()


def test_a_rollback_journal_store_under_a_write_lock_is_unavailable(tmp_path: Path) -> None:
    path = _make_store(tmp_path / "event_store.db")
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.close()
    holder = sqlite3.connect(path, isolation_level=None)
    holder.execute("BEGIN EXCLUSIVE")
    try:
        with pytest.raises(UsageStoreUnavailableError):
            read_real_turn_timestamps(str(path), busy_timeout_seconds=0.05)
    finally:
        holder.execute("ROLLBACK")
        holder.close()


def test_read_does_not_write_to_the_store(store: Path) -> None:
    _add_turn(store, "s-orphan", "2026-10-20T10:00:00+00:00")
    before = store.read_bytes()

    read_real_turn_timestamps(str(store))

    # A WAL reader may create the `-shm` index and an empty `-wal`, which is
    # SQLite's own bookkeeping; any page written would land in `-wal`.
    assert store.read_bytes() == before
    wal = store.with_name(store.name + "-wal")
    assert not wal.exists() or wal.stat().st_size == 0


def test_a_path_with_uri_syntax_characters_is_opened_as_a_filename(tmp_path: Path) -> None:
    # Un-encoded, `#` would end the path at the URI fragment boundary.
    odd_dir = tmp_path / "a#b?c"
    odd_dir.mkdir()
    odd_store = _make_store(odd_dir / "event_store.db")
    _add_turn(odd_store, "s-orphan", "2026-10-20T10:00:00+00:00")

    assert read_real_turn_timestamps(str(odd_store)) == ["2026-10-20T10:00:00+00:00"]


def test_store_path_matches_the_event_store_default() -> None:
    from backend.event_store.store import EventStore

    assert resolve_store_path("/data/es.db") == "/data/es.db"
    # Constructing `EventStore` opens nothing; only its resolved path is compared.
    assert resolve_store_path(None) == str(EventStore(None).db_path)
    assert resolve_store_path("") == str(EventStore(None).db_path)


def test_since_skips_rows_before_the_cutoff_date(store: Path) -> None:
    _add_turn(store, "s-orphan", "2026-10-01T10:00:00+00:00")
    _add_turn(store, "s-orphan", "2026-10-20T10:00:00+00:00")

    stamps = read_real_turn_timestamps(str(store), since=date(2026, 10, 15))

    assert stamps == ["2026-10-20T10:00:00+00:00"]


# --- time zone -----------------------------------------------------------------


def test_resolve_timezone_defaults_to_utc_when_unset_or_blank() -> None:
    for name in (None, "", "   "):
        _zone, used = resolve_timezone(name)
        assert used == "UTC"


def test_resolve_timezone_accepts_an_iana_name() -> None:
    zone, used = resolve_timezone("America/New_York")

    assert used == "America/New_York"
    assert local_date("2026-07-01T03:00:00+00:00", zone) == date(2026, 6, 30)


def test_unknown_timezone_falls_back_to_utc_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger=usage.__name__):
        _zone, used = resolve_timezone("Mars/Olympus_Mons")

    assert used == "UTC"
    assert "Mars/Olympus_Mons" in caplog.text


def test_build_reports_the_fallback_zone(store: Path) -> None:
    result = build_turns_per_day(
        str(store),
        days=7,
        timezone_name="Not/AZone",
        now=datetime(2026, 10, 23, 12, tzinfo=UTC),
    )

    assert result.timezone == "UTC"


def test_build_reads_the_zone_from_the_environment(
    store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MIST_USAGE_TIMEZONE", "Asia/Tokyo")

    result = build_turns_per_day(str(store), days=1, now=datetime(2026, 10, 23, 20, tzinfo=UTC))

    # 20:00Z on the 23rd is 05:00 on the 24th in Tokyo (UTC+9).
    assert result.timezone == "Asia/Tokyo"
    assert result.days[-1].date == date(2026, 10, 24)


def test_a_timestamp_without_an_offset_is_read_as_utc() -> None:
    zone, _ = resolve_timezone("America/New_York")

    # 03:30Z on 9 March is 23:30 EDT on 8 March.
    assert local_date("2026-03-09T03:30:00", zone) == date(2026, 3, 8)
    assert local_date("2026-03-09T03:30:00+00:00", zone) == date(2026, 3, 8)


def test_an_unparsable_timestamp_has_no_local_date() -> None:
    zone, _ = resolve_timezone("UTC")

    assert local_date("not a time", zone) is None


@pytest.mark.parametrize(
    ("stored", "expected_local_date"),
    [
        # Spring forward, Sunday 8 March 2026: 23:30 EDT (UTC-4) is 03:30Z on the 9th.
        ("2026-03-09T03:30:00+00:00", date(2026, 3, 8)),
        # Fall back, Sunday 1 November 2026: 23:30 EST (UTC-5) is 04:30Z on the 2nd.
        ("2026-11-02T04:30:00+00:00", date(2026, 11, 1)),
        # The first night after spring forward: 00:15 EDT on the 9th is 04:15Z.
        # A fixed EST offset (or pytz's LMT from `replace(tzinfo=...)`) puts it on
        # the 8th, and a fixed EDT offset fails the fall-back case above, so only
        # a zone-aware conversion passes all three.
        ("2026-03-09T04:15:00+00:00", date(2026, 3, 9)),
    ],
)
def test_a_late_turn_on_a_dst_change_day_lands_on_its_local_date(
    store: Path, stored: str, expected_local_date: date
) -> None:
    _add_turn(store, "s-orphan", stored)
    next_day = expected_local_date + timedelta(days=1)
    # 16:00Z on the next day is late morning of that same date in New York.
    now = datetime(next_day.year, next_day.month, next_day.day, 16, tzinfo=UTC)

    result = build_turns_per_day(str(store), days=2, timezone_name="America/New_York", now=now)

    assert result.days == (DayCount(expected_local_date, 1), DayCount(next_day, 0))


# --- dense days ------------------------------------------------------------------


def test_days_are_dense_oldest_first_and_end_today() -> None:
    zone, _ = resolve_timezone("UTC")
    stamps = ["2026-10-21T10:00:00+00:00", "2026-10-21T11:00:00+00:00", "2026-10-23T01:00:00+00:00"]

    counts = count_turns_per_day(stamps, zone, today=date(2026, 10, 23), days=4)

    assert counts == (
        DayCount(date(2026, 10, 20), 0),
        DayCount(date(2026, 10, 21), 2),
        DayCount(date(2026, 10, 22), 0),
        DayCount(date(2026, 10, 23), 1),
    )


def test_turns_outside_the_window_are_not_counted() -> None:
    zone, _ = resolve_timezone("UTC")
    stamps = ["2026-10-19T23:59:59+00:00", "2026-10-24T00:00:00+00:00", "garbage"]

    counts = count_turns_per_day(stamps, zone, today=date(2026, 10, 23), days=4)

    assert [c.real_turns for c in counts] == [0, 0, 0, 0]


def test_unparsable_timestamps_are_skipped_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    zone, _ = resolve_timezone("UTC")

    with caplog.at_level(logging.WARNING, logger=usage.__name__):
        counts = count_turns_per_day(
            ["garbage", "2026-10-23T01:00:00+00:00"], zone, today=date(2026, 10, 23), days=1
        )

    assert counts == (DayCount(date(2026, 10, 23), 1),)
    assert "Skipped 1 turn(s) with an unparsable timestamp" in caplog.text


@pytest.mark.parametrize("bad_days", [0, -1, 91])
def test_build_refuses_days_outside_one_to_ninety(store: Path, bad_days: int) -> None:
    with pytest.raises(ValueError):
        build_turns_per_day(str(store), days=bad_days, timezone_name="UTC")


def test_ninety_days_is_accepted(store: Path) -> None:
    result = build_turns_per_day(
        str(store), days=90, timezone_name="UTC", now=datetime(2026, 10, 23, 12, tzinfo=UTC)
    )

    assert len(result.days) == 90
    assert result.days[0].date == date(2026, 7, 26)
    assert result.days[-1].date == date(2026, 10, 23)


# --- read span -------------------------------------------------------------------


def test_one_day_reads_from_the_prefilter_margin_only(
    store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[date | None] = []

    def capture(db_path: str, *, since: date | None = None) -> list[str]:
        seen.append(since)
        return []

    monkeypatch.setattr(usage, "read_real_turn_timestamps", capture)

    build_turns_per_day(
        str(store), days=1, timezone_name="UTC", now=datetime(2026, 10, 23, 18, tzinfo=UTC)
    )

    # One day asked for: today, less the two-day margin, is 21 October. A read
    # widened to a week would start on 15 October.
    assert seen == [date(2026, 10, 21)]


def test_the_counts_cover_only_the_days_asked_for(store: Path) -> None:
    for day in (10, 18, 20, 22, 23):
        _add_turn(store, "s-orphan", f"2026-10-{day:02d}T12:00:00+00:00")
    now = datetime(2026, 10, 23, 18, tzinfo=UTC)

    wide = build_turns_per_day(str(store), days=30, timezone_name="UTC", now=now)
    narrow = build_turns_per_day(str(store), days=1, timezone_name="UTC", now=now)

    assert sum(1 for d in wide.days if d.real_turns) == 5
    assert narrow.days == (DayCount(date(2026, 10, 23), 1),)


# --- payload -------------------------------------------------------------------


def test_payload_has_exactly_the_contract_shape(store: Path) -> None:
    _add_session(store, "session-secret-id", "real")
    _add_turn(store, "session-secret-id", "2026-10-22T09:15:42+00:00", "my private words")

    payload = build_turns_per_day(
        str(store), days=2, timezone_name="UTC", now=datetime(2026, 10, 23, 12, tzinfo=UTC)
    ).to_payload()

    assert set(payload) == {"timezone", "origin", "days"}
    assert payload == {
        "timezone": "UTC",
        "origin": "real",
        "days": [
            {"date": "2026-10-22", "real_turns": 1},
            {"date": "2026-10-23", "real_turns": 0},
        ],
    }
