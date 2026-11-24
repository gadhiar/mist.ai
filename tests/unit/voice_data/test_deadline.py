"""`deadline.py`: `--until <HH:MM>` host local time to the job's `--deadline-utc`.

A job's container has no time zone and reads its clock as UTC, so the host converts. Each test
pins the host's zone with a POSIX `TZ` rule (glibc reads it without zoneinfo files) and passes a
fixed now, so no test depends on the machine's clock or zone.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest

from scripts.voice_data import deadline
from scripts.voice_data.errors import InputError

# New York's rule: EST, five hours behind UTC, and EDT from the second Sunday of March to the
# first Sunday of November.
NEW_YORK = "EST5EDT,M3.2.0,M11.1.0"
# Kolkata: five and a half hours ahead of UTC, no daylight time.
KOLKATA = "IST-5:30"

pytestmark = pytest.mark.skipif(not hasattr(time, "tzset"), reason="needs time.tzset (POSIX)")


@pytest.fixture
def zone(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Let a test set `TZ`; the host's own zone comes back afterwards."""
    yield
    monkeypatch.undo()
    time.tzset()


def use_zone(monkeypatch: pytest.MonkeyPatch, rule: str) -> None:
    monkeypatch.setenv("TZ", rule)
    time.tzset()


def utc(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


def test_until_later_today_in_new_york(zone: None, monkeypatch: pytest.MonkeyPatch) -> None:
    use_zone(monkeypatch, NEW_YORK)
    now = utc("2026-11-23T15:00:00")  # 10:00 EST
    assert deadline.deadline_utc("13:30", now=now) == utc("2026-11-23T18:30:00")


def test_until_past_local_midnight_is_tomorrow(zone: None, monkeypatch: pytest.MonkeyPatch) -> None:
    use_zone(monkeypatch, NEW_YORK)
    now = utc("2026-11-24T03:30:00")  # 22:30 EST on 23 Nov
    assert deadline.deadline_utc("01:15", now=now) == utc("2026-11-24T06:15:00")


def test_until_crosses_the_utc_day_but_not_the_local_one(
    zone: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_zone(monkeypatch, NEW_YORK)
    now = utc("2026-11-23T23:50:00")  # 18:50 EST
    assert deadline.deadline_utc("19:30", now=now) == utc("2026-11-24T00:30:00")


def test_a_time_already_past_today_is_tomorrow(zone: None, monkeypatch: pytest.MonkeyPatch) -> None:
    use_zone(monkeypatch, KOLKATA)
    now = utc("2026-11-23T12:00:00")  # 17:30 in Kolkata
    assert deadline.deadline_utc("17:00", now=now) == utc("2026-11-24T11:30:00")


def test_daylight_time_ends_overnight(zone: None, monkeypatch: pytest.MonkeyPatch) -> None:
    # 1 Nov 2026 is the first Sunday of November: 02:00 EDT becomes 01:00 EST.
    use_zone(monkeypatch, NEW_YORK)
    now = utc("2026-11-01T03:00:00")  # 23:00 EDT on 31 Oct
    assert deadline.deadline_utc("07:00", now=now) == utc("2026-11-01T12:00:00")


def test_refuses_a_time_under_ten_minutes_ahead(
    zone: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_zone(monkeypatch, NEW_YORK)
    now = utc("2026-11-23T15:00:00")  # 10:00 EST
    with pytest.raises(InputError, match="10 minutes to 24 h ahead"):
        deadline.deadline_utc("10:09", now=now)
    assert deadline.deadline_utc("10:10", now=now) == now + timedelta(minutes=10)


def test_the_current_minute_means_tomorrow(zone: None, monkeypatch: pytest.MonkeyPatch) -> None:
    # Now is 10:00:30, so today's 10:00 has passed and tomorrow's is 23 h 59.5 min away.
    use_zone(monkeypatch, NEW_YORK)
    now = utc("2026-11-23T15:00:30")
    assert deadline.deadline_utc("10:00", now=now) == utc("2026-11-24T15:00:00")


def test_refuses_more_than_24_hours_ahead(zone: None, monkeypatch: pytest.MonkeyPatch) -> None:
    # The night daylight time ends has 25 hours, so a wall-clock time just before now is more
    # than 24 h away; one an hour earlier is exactly 24 h away and accepted.
    use_zone(monkeypatch, NEW_YORK)
    now = utc("2026-11-01T03:00:00")  # 23:00 EDT on 31 Oct
    with pytest.raises(InputError, match="10 minutes to 24 h ahead"):
        deadline.deadline_utc("22:59", now=now)
    assert deadline.deadline_utc("22:00", now=now) == utc("2026-11-02T03:00:00")


@pytest.mark.parametrize("text", ["7:30", "24:00", "12:60", "12-30", "", "noon", "12:30:00"])
def test_refuses_a_malformed_time(text: str) -> None:
    with pytest.raises(InputError, match="HH:MM"):
        deadline.deadline_utc(text, now=utc("2026-11-23T15:00:00"))


def test_format_and_parse_round_trip() -> None:
    when = utc("2026-11-24T06:15:00")
    assert deadline.format_utc(when) == "2026-11-24T06:15:00Z"
    assert deadline.parse_utc("2026-11-24T06:15:00Z") == when


@pytest.mark.parametrize("text", ["2026-11-24T06:15:00", "2026-11-24T06:15:00+01:00", "soon"])
def test_parse_refuses_anything_but_utc_z(text: str) -> None:
    with pytest.raises(ValueError, match="ISO 8601 UTC"):
        deadline.parse_utc(text)
