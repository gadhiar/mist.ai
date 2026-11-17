"""Tests for backend.memory.temporal: deterministic temporal phrases to time windows.

No test names an IANA zone: the image has no tz database, so the zones are fixed-offset
`datetime.timezone` objects and `_ShiftZone`, a hand-written zone with two daylight-saving
changes.
"""

from datetime import UTC, datetime, timedelta, timezone, tzinfo

import pytest

from backend.memory.models import TimeWindow
from backend.memory.temporal import parse_window
from tests.fakes.memory_fakes import FakeClock

PLUS_0530 = timezone(timedelta(hours=5, minutes=30))
MINUS_0700 = timezone(timedelta(hours=-7))

# 20:00 UTC on Thursday 2026-10-22 is 01:30 on Friday 2026-10-23 at +05:30 and 13:00 on
# Thursday 2026-10-22 at -07:00, so the two zones disagree on the local date.
NOW = FakeClock(datetime(2026, 10, 22, 20, 0, tzinfo=UTC)).now()


def _local(zone: tzinfo, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(2026, month, day, hour, minute, tzinfo=zone)


# (text, (start month, day, hour), (end month, day, hour)) in local wall-clock time.
_PLUS_0530_CASES = [
    ("today", (10, 23, 0), (10, 24, 0)),
    ("yesterday", (10, 22, 0), (10, 23, 0)),
    ("this morning", (10, 23, 5), (10, 23, 12)),
    ("this afternoon", (10, 23, 12), (10, 23, 17)),
    ("this evening", (10, 23, 17), (10, 24, 0)),
    ("last night", (10, 22, 17), (10, 23, 5)),
    ("3 days ago", (10, 20, 0), (10, 21, 0)),
    ("three days ago", (10, 20, 0), (10, 21, 0)),
    ("1 day ago", (10, 22, 0), (10, 23, 0)),
    ("last week", (10, 12, 0), (10, 19, 0)),
    ("on tuesday", (10, 20, 0), (10, 21, 0)),
    ("last tuesday", (10, 20, 0), (10, 21, 0)),
    ("last friday", (10, 16, 0), (10, 17, 0)),
    ("on thursday", (10, 22, 0), (10, 23, 0)),
    ("the other day", (10, 19, 0), (10, 22, 0)),
]

_MINUS_0700_CASES = [
    ("today", (10, 22, 0), (10, 23, 0)),
    ("yesterday", (10, 21, 0), (10, 22, 0)),
    ("this morning", (10, 22, 5), (10, 22, 12)),
    ("this afternoon", (10, 22, 12), (10, 22, 17)),
    ("this evening", (10, 22, 17), (10, 23, 0)),
    ("last night", (10, 21, 17), (10, 22, 5)),
    ("3 days ago", (10, 19, 0), (10, 20, 0)),
    ("three days ago", (10, 19, 0), (10, 20, 0)),
    ("1 day ago", (10, 21, 0), (10, 22, 0)),
    ("last week", (10, 12, 0), (10, 19, 0)),
    ("on tuesday", (10, 20, 0), (10, 21, 0)),
    ("last tuesday", (10, 20, 0), (10, 21, 0)),
    ("last friday", (10, 16, 0), (10, 17, 0)),
    ("on thursday", (10, 15, 0), (10, 16, 0)),
    ("the other day", (10, 18, 0), (10, 21, 0)),
]


@pytest.mark.parametrize(
    ("zone", "text", "start", "end"),
    [
        pytest.param(PLUS_0530, text, start, end, id=f"+0530-{text}")
        for text, start, end in _PLUS_0530_CASES
    ]
    + [
        pytest.param(MINUS_0700, text, start, end, id=f"-0700-{text}")
        for text, start, end in _MINUS_0700_CASES
    ],
)
def test_each_phrase_maps_to_its_window_in_two_fixed_offset_zones(
    zone: tzinfo,
    text: str,
    start: tuple[int, int, int],
    end: tuple[int, int, int],
) -> None:
    window = parse_window(f"what did we say {text}?", now=NOW, tz=zone)

    assert window is not None
    assert window.start == _local(zone, *start)
    assert window.end == _local(zone, *end)
    assert window.phrase == text
    assert not window.previous_conversation


@pytest.mark.parametrize(
    ("zone", "start"),
    [
        pytest.param(PLUS_0530, _local(PLUS_0530, 10, 23), id="+0530"),
        pytest.param(MINUS_0700, _local(MINUS_0700, 10, 22), id="-0700"),
    ],
)
def test_earlier_today_runs_from_local_midnight_to_now(zone: tzinfo, start: datetime) -> None:
    window = parse_window("you mentioned it earlier today", now=NOW, tz=zone)

    assert window is not None
    assert window.start == start
    assert window.end == NOW
    assert window.phrase == "earlier today"


def test_earlier_today_at_local_midnight_is_empty_so_returns_none() -> None:
    midnight = _local(MINUS_0700, 10, 22)

    assert parse_window("earlier today", now=midnight, tz=MINUS_0700) is None


@pytest.mark.parametrize("zone", [PLUS_0530, MINUS_0700], ids=["+0530", "-0700"])
def test_last_time_names_the_previous_conversation_without_bounds(zone: tzinfo) -> None:
    window = parse_window("what did we talk about last time?", now=NOW, tz=zone)

    assert window == TimeWindow(
        start=None, end=None, phrase="last time", previous_conversation=True
    )


def test_bounds_are_utc_instants_so_arithmetic_is_exact() -> None:
    window = parse_window("yesterday", now=NOW, tz=PLUS_0530)

    assert window is not None
    assert window.start is not None and window.end is not None
    assert window.start.tzinfo is UTC
    assert window.end.tzinfo is UTC
    assert window.start == datetime(2026, 10, 21, 18, 30, tzinfo=UTC)


class TestMidnightBoundary:
    """The local date, not the UTC date, decides "yesterday".

    Both instants fall on 2026-10-23 in UTC, but one second apart they sit on either side of
    local midnight at -07:00, so "yesterday" moves by a day between them.
    """

    def test_one_second_before_local_midnight(self) -> None:
        now = datetime(2026, 10, 23, 6, 59, 59, tzinfo=UTC)  # 23:59:59 on Oct 22 at -07:00

        window = parse_window("yesterday", now=now, tz=MINUS_0700)

        assert window is not None
        assert window.start == _local(MINUS_0700, 10, 21)
        assert window.end == _local(MINUS_0700, 10, 22)

    def test_at_local_midnight(self) -> None:
        now = datetime(2026, 10, 23, 7, 0, tzinfo=UTC)  # 00:00 on Oct 23 at -07:00

        window = parse_window("yesterday", now=now, tz=MINUS_0700)

        assert window is not None
        assert window.start == _local(MINUS_0700, 10, 22)
        assert window.end == _local(MINUS_0700, 10, 23)


# A fictional zone: UTC-05:00 standard, UTC-04:00 daylight. In 2026 the clock jumps from 02:00
# to 03:00 on March 8 and falls back from 02:00 daylight to 01:00 standard on November 1.
_STD = timedelta(hours=-5)
_DST = timedelta(hours=-4)
_SPRING_LOCAL = datetime(2026, 3, 8, 2)  # first wall time skipped
_SPRING_LOCAL_END = datetime(2026, 3, 8, 3)  # first wall time after the gap
_FALL_LOCAL = datetime(2026, 11, 1, 1)  # first wall time repeated
_FALL_LOCAL_END = datetime(2026, 11, 1, 2)  # first wall time after the repeat
_SPRING_UTC = datetime(2026, 3, 8, 7)  # 02:00 standard
_FALL_UTC = datetime(2026, 11, 1, 6)  # 02:00 daylight


class _ShiftZone(tzinfo):
    """A PEP 495 zone with one spring-forward and one fall-back change in 2026."""

    def _is_dst(self, dt: datetime) -> bool:
        wall = dt.replace(tzinfo=None)
        if _SPRING_LOCAL <= wall < _SPRING_LOCAL_END:
            return dt.fold == 1  # in the gap, fold=0 keeps the offset from before the jump
        if _FALL_LOCAL <= wall < _FALL_LOCAL_END:
            return dt.fold == 0  # the first pass through the repeated hour is daylight time
        return _SPRING_LOCAL_END <= wall < _FALL_LOCAL

    def utcoffset(self, dt: datetime | None) -> timedelta:
        if dt is None:
            return _STD
        return _DST if self._is_dst(dt) else _STD

    def dst(self, dt: datetime | None) -> timedelta:
        if dt is None:
            return timedelta(0)
        return _DST - _STD if self._is_dst(dt) else timedelta(0)

    def tzname(self, dt: datetime | None) -> str:
        return "SHIFT-DST" if dt is not None and self._is_dst(dt) else "SHIFT-STD"

    def fromutc(self, dt: datetime) -> datetime:
        utc_wall = dt.replace(tzinfo=None)
        if _SPRING_UTC <= utc_wall < _FALL_UTC:
            return (utc_wall + _DST).replace(tzinfo=self)
        local = (utc_wall + _STD).replace(tzinfo=self)
        if _FALL_UTC <= utc_wall < _FALL_UTC + timedelta(hours=1):
            return local.replace(fold=1)
        return local


SHIFT = _ShiftZone()


def _utc(month: int, day: int, hour: int) -> datetime:
    return datetime(2026, month, day, hour, tzinfo=UTC)


def test_shift_zone_changes_offset_on_both_change_days() -> None:
    assert datetime(2026, 3, 8, 1, 59, tzinfo=SHIFT).utcoffset() == _STD
    assert datetime(2026, 3, 8, 3, 0, tzinfo=SHIFT).utcoffset() == _DST
    assert datetime(2026, 11, 1, 0, 59, tzinfo=SHIFT).utcoffset() == _DST
    assert datetime(2026, 11, 1, 2, 0, tzinfo=SHIFT).utcoffset() == _STD
    assert _utc(11, 1, 5).astimezone(SHIFT).fold == 0
    assert _utc(11, 1, 6).astimezone(SHIFT).fold == 1


class TestDaylightSavingChanges:
    """Pin "yesterday" and "last night" around both changes, in UTC so the lengths show.

    Each change is tested with `now` on the change day (the window crosses the change only for
    "last night") and on the day after (the change day is "yesterday" and has 23 or 25 hours).
    """

    @pytest.mark.parametrize(
        ("now", "start", "end", "hours"),
        [
            pytest.param(_utc(3, 8, 16), _utc(3, 7, 5), _utc(3, 8, 5), 24, id="spring-day"),
            pytest.param(_utc(3, 9, 16), _utc(3, 8, 5), _utc(3, 9, 4), 23, id="spring-next"),
            pytest.param(_utc(11, 1, 17), _utc(10, 31, 4), _utc(11, 1, 4), 24, id="fall-day"),
            pytest.param(_utc(11, 2, 17), _utc(11, 1, 4), _utc(11, 2, 5), 25, id="fall-next"),
        ],
    )
    def test_yesterday(self, now: datetime, start: datetime, end: datetime, hours: int) -> None:
        window = parse_window("yesterday", now=now, tz=SHIFT)

        assert window is not None
        assert (window.start, window.end) == (start, end)
        assert window.end - window.start == timedelta(hours=hours)

    @pytest.mark.parametrize(
        ("now", "start", "end", "hours"),
        [
            pytest.param(_utc(3, 8, 16), _utc(3, 7, 22), _utc(3, 8, 9), 11, id="spring-day"),
            pytest.param(_utc(3, 9, 16), _utc(3, 8, 21), _utc(3, 9, 9), 12, id="spring-next"),
            pytest.param(_utc(11, 1, 17), _utc(10, 31, 21), _utc(11, 1, 10), 13, id="fall-day"),
            pytest.param(_utc(11, 2, 17), _utc(11, 1, 22), _utc(11, 2, 10), 12, id="fall-next"),
        ],
    )
    def test_last_night(self, now: datetime, start: datetime, end: datetime, hours: int) -> None:
        window = parse_window("last night", now=now, tz=SHIFT)

        assert window is not None
        assert (window.start, window.end) == (start, end)
        assert window.end - window.start == timedelta(hours=hours)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("", id="empty"),
        pytest.param("tell me about the kestrel project", id="no-phrase"),
        pytest.param("what happened last weekend", id="last-weekend"),
        pytest.param("my yesterdays", id="yesterdays"),
        pytest.param("on mondays I swim", id="plural-weekday"),
        pytest.param("tuesday", id="bare-weekday"),
        pytest.param("0 days ago", id="zero-days"),
        pytest.param("12345 days ago", id="five-digit-days"),
        pytest.param("this week", id="this-week"),
    ],
)
def test_no_match_returns_none(text: str) -> None:
    assert parse_window(text, now=NOW, tz=MINUS_0700) is None


def test_matching_ignores_case_and_collapses_whitespace() -> None:
    window = parse_window("What did I say LAST\n  Night?", now=NOW, tz=MINUS_0700)

    assert window is not None
    assert window.phrase == "last night"
    assert window.start == _local(MINUS_0700, 10, 21, 17)


def test_the_earliest_phrase_in_the_text_wins() -> None:
    window = parse_window("yesterday, or maybe last week", now=NOW, tz=MINUS_0700)

    assert window is not None
    assert window.phrase == "yesterday"


def test_a_day_count_before_year_one_returns_none() -> None:
    assert parse_window("9999 days ago", now=datetime(1, 1, 5, tzinfo=UTC), tz=UTC) is None


def test_naive_now_is_rejected() -> None:
    with pytest.raises(ValueError, match="aware"):
        parse_window("yesterday", now=datetime(2026, 10, 22, 12), tz=MINUS_0700)
