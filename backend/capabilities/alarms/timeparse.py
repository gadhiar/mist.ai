"""Spoken alarm times and repeat rules, resolved to UTC instants in the user's time zone.

The model passes the user's own words ("7:30 pm", "in 20 minutes") rather than a timestamp,
because its prompt holds no current time, so the server resolves them here against an injected
`now`. Every result is an aware UTC datetime, the form the alarm store keeps.

Wall times become instants through `zoneinfo` rules with `fold=0`: a time that falls in a
spring-forward gap shifts forward by the gap, and an ambiguous fall-back time takes its first
occurrence. A repeat rule keeps its local hour and minute rather than a UTC instant, so a daily
7:00 alarm stays at 7:00 across a daylight-saving change.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta, tzinfo
from typing import Literal

from backend.errors import TimeParseError

RepeatKind = Literal["daily", "weekdays", "weekends"]

_REPEAT_DAYS: dict[str, frozenset[int]] = {
    "daily": frozenset(range(7)),
    "weekdays": frozenset(range(5)),
    "weekends": frozenset({5, 6}),
}
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

_RELATIVE = re.compile(r"in (\d{1,4}|an?) (minutes?|hours?)")
# Minutes follow a colon or a space: T3's normaliser turns "seven thirty pm" into "7 30 pm".
_MERIDIEM = re.compile(r"(\d{1,2})(?:[: ](\d{2}))? ?(am|pm)")
_CLOCK = re.compile(r"(\d{1,2})[: ](\d{2})")
_BARE_HOUR = re.compile(r"\d{1,2}")
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_NAMED_TIMES = {"noon": (12, 0), "midnight": (0, 0)}


@dataclass(frozen=True, slots=True)
class RepeatRule:
    """A repeating alarm: which days it rings and at what local wall time.

    Raises:
        ValueError: `kind` is not a known rule, or `hour` or `minute` is not an integer in range,
            whatever the wrong value's type.
    """

    kind: RepeatKind
    hour: int
    minute: int

    def __post_init__(self) -> None:
        # The type check comes first: an unhashable kind such as a list would make the
        # membership test raise `TypeError`, which a reader of stored rules does not expect.
        if not isinstance(self.kind, str) or self.kind not in _REPEAT_DAYS:
            raise ValueError(f"unknown repeat kind {self.kind!r}")
        for name, value, limit in (("hour", self.hour, 23), ("minute", self.minute, 59)):
            # `bool` is an `int` subclass, and `true` must not read as hour 1.
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= limit:
                raise ValueError(f"repeat {name} must be an integer from 0 to {limit}: {value!r}")

    def to_json(self) -> str:
        """Encode the rule as the JSON object the alarm store keeps."""
        return json.dumps({"kind": self.kind, "hour": self.hour, "minute": self.minute})

    @classmethod
    def from_json(cls, text: str) -> RepeatRule:
        """Decode a rule written by `to_json`.

        Every malformed input raises `ValueError` and nothing else, so a reader of stored rules
        can treat that one error as a damaged row.

        Raises:
            ValueError: `text` is not a string or not JSON (`json.JSONDecodeError` is a
                `ValueError`), is not an object holding exactly `kind`, `hour` and `minute`, or
                holds a value of the wrong type or out of range.
        """
        if not isinstance(text, str):
            raise ValueError(f"repeat rule must be JSON text: {text!r}")
        try:
            data = json.loads(text)
        except RecursionError as exc:
            raise ValueError("repeat rule JSON is nested too deeply") from exc
        if not isinstance(data, dict) or set(data) != {"kind", "hour", "minute"}:
            raise ValueError(f"repeat rule must be an object of kind, hour and minute: {text!r}")
        return cls(kind=data["kind"], hour=data["hour"], minute=data["minute"])


def parse_alarm_time(time_text: str, day: str | None, now: datetime, tz: tzinfo) -> datetime:
    """Resolve a spoken time, and an optional day, to the instant it names.

    Times: "7am" and "7 am", "7:30pm" and "7:30 pm", 24-hour "19:30", "noon" and "midnight".
    Minutes may follow a space instead of a colon ("7 30 pm", "7 30"), as exactly two digits. An
    hour from 1 to 12 with no meridiem and no leading zero ("7", "6:30", "7 30") is the earlier
    of its morning and evening occurrences after `now`; 0, 13 to 23, and an hour written with a
    leading zero ("07:30") read as a 24-hour clock. "in N minutes" and "in N hours" (also "in a
    minute", "in an hour") are elapsed time from `now`, and take no day but "today".

    Days: "today", "tomorrow", a weekday name (its next date, today included while the time is
    still ahead) and an ISO date. With no day, a time that has already passed today rolls to
    tomorrow; with a day, a time already passed on it is an error. Case and surrounding space are
    ignored, and "a.m." and "p.m." read as "am" and "pm".

    Args:
        time_text: The time as the user said it.
        day: The day as the user said it, or None or "" for none.
        now: The current instant; aware.
        tz: The user's time zone, in which the wall time is read.

    Returns:
        The instant, as an aware UTC datetime strictly after `now`.

    Raises:
        TimeParseError: The time or day cannot be read, or names a moment already past; the
            message quotes the input.
        ValueError: `now` is naive.
    """
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("parse_alarm_time needs an aware now")
    text = _normalise(time_text)
    day_text = _normalise(day or "")

    relative = _RELATIVE.fullmatch(text)
    if relative is not None:
        return _relative(time_text, relative, day_text, now)

    candidates = _wall_times(text)
    if candidates is None:
        raise TimeParseError(f"cannot read {time_text!r} as an alarm time")
    today = now.astimezone(tz).date()
    for candidate_date in _candidate_dates(day, day_text, today):
        for hour, minute in candidates:
            instant = _instant(candidate_date, hour, minute, tz)
            if instant > now:
                return instant
    raise TimeParseError(f"{time_text!r} on {day!r} has already passed")


def next_occurrence(rule: RepeatRule, after: datetime, tz: tzinfo) -> datetime:
    """Return the first instant strictly after `after` at which `rule` rings in `tz`.

    Strictly after, so passing the occurrence that just rang gives the following one, and the
    second pass through an ambiguous fall-back hour never rings the rule twice.

    Raises:
        ValueError: `after` is naive.
    """
    if after.tzinfo is None or after.utcoffset() is None:
        raise ValueError("next_occurrence needs an aware after")
    days = _REPEAT_DAYS[rule.kind]
    start = after.astimezone(tz).date()
    # Every rule rings at least once in any 7 consecutive days; day 8 covers a start date
    # whose own occurrence is already past.
    for offset in range(8):
        candidate_date = start + timedelta(days=offset)
        if candidate_date.weekday() not in days:
            continue
        instant = _instant(candidate_date, rule.hour, rule.minute, tz)
        if instant > after:
            return instant
    raise AssertionError(f"no occurrence of {rule} within 8 days of {after.isoformat()}")


def _normalise(text: str) -> str:
    lowered = text.lower().replace("a.m.", "am").replace("p.m.", "pm")
    return " ".join(lowered.split())


def _relative(time_text: str, match: re.Match[str], day_text: str, now: datetime) -> datetime:
    if day_text not in ("", "today"):
        raise TimeParseError(f"{time_text!r} is relative to now and cannot be on {day_text!r}")
    amount_text, unit = match.groups()
    amount = 1 if amount_text in ("a", "an") else int(amount_text)
    if amount == 0:
        raise TimeParseError(f"cannot set an alarm {time_text!r}: that is now")
    delta = timedelta(hours=amount) if unit.startswith("hour") else timedelta(minutes=amount)
    # Added on the UTC instant: arithmetic on an aware datetime in a zone adds wall time,
    # which is an hour off across a daylight-saving change.
    return now.astimezone(UTC) + delta


def _wall_times(text: str) -> list[tuple[int, int]] | None:
    """Return the wall times `text` may mean, earliest first, or None when it is not a time."""
    if text in _NAMED_TIMES:
        return [_NAMED_TIMES[text]]
    meridiem = _MERIDIEM.fullmatch(text)
    if meridiem is not None:
        hour, minute = int(meridiem.group(1)), int(meridiem.group(2) or 0)
        if not 1 <= hour <= 12 or minute > 59:
            return None
        return [(hour % 12 + (12 if meridiem.group(3) == "pm" else 0), minute)]
    clock = _CLOCK.fullmatch(text)
    if clock is not None:
        hour_text, minute = clock.group(1), int(clock.group(2))
    elif _BARE_HOUR.fullmatch(text) is not None:
        hour_text, minute = text, 0
    else:
        return None
    hour = int(hour_text)
    if hour > 23 or minute > 59:
        return None
    # A written leading zero ("07:30") is a 24-hour clock, so only an hour written without
    # one is ambiguous between morning and evening.
    if 1 <= hour <= 12 and not hour_text.startswith("0"):
        return sorted([(hour % 12, minute), (hour % 12 + 12, minute)])
    return [(hour, minute)]


def _candidate_dates(day: str | None, day_text: str, today: date) -> list[date]:
    """Return the dates a time may fall on, in the order they are tried."""
    if day_text == "":
        return [today, today + timedelta(days=1)]
    if day_text == "today":
        return [today]
    if day_text == "tomorrow":
        return [today + timedelta(days=1)]
    if day_text in _WEEKDAYS:
        first = today + timedelta(days=(_WEEKDAYS.index(day_text) - today.weekday()) % 7)
        return [first, first + timedelta(days=7)]
    if _ISO_DATE.fullmatch(day_text) is not None:
        try:
            return [date.fromisoformat(day_text)]
        except ValueError:
            pass
    raise TimeParseError(f"cannot read {day!r} as a day")


def _instant(on: date, hour: int, minute: int, tz: tzinfo) -> datetime:
    # `fold=0` is the first occurrence of an ambiguous time, and for a time inside a gap it
    # carries the offset from before the gap, so the conversion lands after the gap.
    return datetime.combine(on, time(hour, minute), tzinfo=tz).astimezone(UTC)
