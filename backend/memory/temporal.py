"""Deterministic reading of temporal phrases in a user turn into a time window.

A phrase such as "yesterday" or "the other day" tells the read path which span of time the user
means, so an item inside that span can outrank one that is merely recent. No model is involved:
the same words, `now` and zone always give the same window, so a replay under a pinned clock
makes the same decision the live turn made.

Bounds are computed on the local wall clock of the configured zone and returned as UTC
instants. Python compares and subtracts two datetimes that share one `tzinfo` by wall-clock
time, ignoring the offset, so a "yesterday" kept in the zone would claim 24 hours on a
daylight-saving change day that had 23 or 25. UTC bounds keep every comparison against an
item's timestamp, and every length, exact.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta, tzinfo

from .models import TimeWindow

# Parts of the day, as local wall-clock hours. Night runs from midnight to the start of the
# morning, so "last night" (the previous evening and the small hours after it) ends exactly
# where "this morning" begins and the two never overlap.
MORNING_START_HOUR = 5
AFTERNOON_START_HOUR = 12
EVENING_START_HOUR = 17

# "the other day" covers the local days 2, 3 and 4 days before today.
OTHER_DAY_NEAREST = 2
OTHER_DAY_FARTHEST = 4

_WEEKDAYS: tuple[str, ...] = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)

_NUMBER_WORDS: dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}

# One alternation, searched once: the leftmost phrase in the turn wins, and "earlier today"
# starts before the "today" inside it. The trailing word boundary keeps "last weekend",
# "yesterdays" and "mondays" from matching. Digits are capped at four so a long digit run
# never reaches `int`, which refuses strings past its digit limit.
_PATTERN = re.compile(
    r"\b(?:"
    r"(?P<earlier_today>earlier\s+today)"
    r"|(?P<today>today)"
    r"|(?P<yesterday>yesterday)"
    r"|this\s+(?P<part>morning|afternoon|evening)"
    r"|(?P<last_night>last\s+night)"
    r"|(?P<last_week>last\s+week)"
    r"|(?P<last_time>last\s+time)"
    rf"|(?:on|last)\s+(?P<weekday>{'|'.join(_WEEKDAYS)})"
    rf"|(?P<count>\d{{1,4}}|{'|'.join(_NUMBER_WORDS)})\s+days?\s+ago"
    r"|(?P<other_day>the\s+other\s+day)"
    r")\b",
    re.IGNORECASE,
)

_PART_HOURS: dict[str, tuple[int, int | None]] = {
    "morning": (MORNING_START_HOUR, AFTERNOON_START_HOUR),
    "afternoon": (AFTERNOON_START_HOUR, EVENING_START_HOUR),
    "evening": (EVENING_START_HOUR, None),  # None: up to the next local midnight
}


def parse_window(text: str, *, now: datetime, tz: tzinfo) -> TimeWindow | None:
    """Read the first temporal phrase in `text` as a window relative to `now` in `tz`.

    Recognised phrases: today, yesterday, this morning, this afternoon, this evening, last
    night, earlier today, N days ago (digits or one to twelve), last week (the previous
    Monday-to-Monday calendar week), on or last <weekday> (the most recent such day before
    today, so a week back when it names today), the other day (2 to 4 days ago), and last time.

    "last time" names the previous conversation, which only conversation state can date, so it
    comes back with no bounds and `previous_conversation=True`. The fetcher either replaces it
    with the bounds of that conversation (`dataclasses.replace`, keeping the flag) or, while no
    conversation state exists, lets ranking fall back to recency.

    Args:
        text: The user's words for this turn.
        now: The turn's instant from the handler's fact-time seam; must be aware.
        tz: The configured zone, in which day boundaries and parts of the day fall.

    Returns:
        The window, with UTC bounds, or None when no phrase matches or the phrase names an
        empty or unrepresentable span ("earlier today" at local midnight, a day count that
        reaches before year 1).

    Raises:
        ValueError: `now` is naive, so its local date is unknown.
    """
    if now.utcoffset() is None:
        raise ValueError(f"now must be timezone-aware, got {now!r}")
    match = _PATTERN.search(text)
    if match is None:
        return None
    phrase = " ".join(match.group(0).lower().split())
    if match.group("last_time"):
        return TimeWindow(start=None, end=None, phrase=phrase, previous_conversation=True)

    today = now.astimezone(tz).date()
    try:
        bounds = _bounds(match, today, now, tz)
    except OverflowError:
        return None
    if bounds is None:
        return None
    return TimeWindow(start=bounds[0], end=bounds[1], phrase=phrase)


def _bounds(
    match: re.Match[str], today: date, now: datetime, tz: tzinfo
) -> tuple[datetime, datetime] | None:
    """The UTC bounds a matched phrase names, or None for an empty span."""
    if match.group("earlier_today"):
        start = _at(today, 0, tz)
        end = now.astimezone(UTC)
        return (start, end) if end > start else None
    if match.group("today"):
        return _days(today, 1, tz)
    if match.group("yesterday"):
        return _days(today - timedelta(days=1), 1, tz)
    part = match.group("part")
    if part:
        first_hour, last_hour = _PART_HOURS[part.lower()]
        if last_hour is None:
            return _at(today, first_hour, tz), _at(today + timedelta(days=1), 0, tz)
        return _at(today, first_hour, tz), _at(today, last_hour, tz)
    if match.group("last_night"):
        return (
            _at(today - timedelta(days=1), EVENING_START_HOUR, tz),
            _at(today, MORNING_START_HOUR, tz),
        )
    if match.group("last_week"):
        this_monday = today - timedelta(days=today.weekday())
        return _days(this_monday - timedelta(days=7), 7, tz)
    weekday = match.group("weekday")
    if weekday:
        back = (today.weekday() - _WEEKDAYS.index(weekday.lower())) % 7 or 7
        return _days(today - timedelta(days=back), 1, tz)
    count = match.group("count")
    if count:
        days = int(count) if count.isdigit() else _NUMBER_WORDS[count.lower()]
        if days < 1:
            return None
        return _days(today - timedelta(days=days), 1, tz)
    # The only alternative left is "the other day".
    span = OTHER_DAY_FARTHEST - OTHER_DAY_NEAREST + 1
    return _days(today - timedelta(days=OTHER_DAY_FARTHEST), span, tz)


def _days(first: date, count: int, tz: tzinfo) -> tuple[datetime, datetime]:
    """Bounds of `count` whole local days starting at `first`."""
    return _at(first, 0, tz), _at(first + timedelta(days=count), 0, tz)


def _at(day: date, hour: int, tz: tzinfo) -> datetime:
    """The UTC instant at which the wall clock in `tz` reads `hour`:00 on `day`.

    `fold=0` picks the first of two instants when the wall time repeats, and for a wall time a
    spring-forward change skips it applies the offset from before the change, which for a
    change at exactly this hour is the instant of the change itself (PEP 495).
    """
    return datetime(day.year, day.month, day.day, hour, tzinfo=tz).astimezone(UTC)
