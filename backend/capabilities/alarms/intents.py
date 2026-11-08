"""The alarms, timers and clock rows of the fast-path grammar, and the ringing phrases.

An anchored grammar is the one deterministic test of an explicit command, so "set an alarm for
7", "10 minute timer" and "what's the time" run without the model, at every power level. Every
function reads `normalize_utterance` output, which has already turned number words into digits,
"a.m." into "am", "7.30" into "7:30" and "what's" into "whats". The rows (build plan, Fast-path
intents; `[]` optional, `|` alternatives):

- alarms: `[set] [an] alarm (for|at) <time> [[on] <day>]` and `wake me [up] at <time> [[on]
  <day>]`, giving `alarms.set`. `<day>` is "today", "tomorrow" or a weekday name. The time and
  day must parse with `parse_alarm_time` against the caller's now and zone, so a time it refuses
  ("whenever", "25:00", a time already past "today") is not a match and the model gets the
  utterance;
- timers: `[set|start] [a] timer for <duration>` and `[set|start] [a] <N> <unit> timer`, giving
  `timers.start`. `<duration>` is one to three "N unit" parts, each unit at most once,
  optionally joined by "and", totalling 1 s to 24 h;
- clock: "what time is it", "whats the time", "what is the time", "whats the date", "what is the
  date" and "what day is it", giving `clock.now`.

The ringing phrases are not fast-path rows. The host asks for them first, while an item rings
(pre-LLM step 1), so "turn off" stops a ringing alarm rather than reaching lights, and a bare
"stop" never collides with a proposal's "no".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, tzinfo
from typing import Literal

from backend.errors import TimeParseError

from ..contracts import FastIntent
from .records import MAX_DURATION_S, AlarmKind
from .timeparse import parse_alarm_time

ALARMS_SET = "alarms.set"
TIMERS_START = "timers.start"
CLOCK_NOW = "clock.now"

_WEEKDAYS = frozenset(
    {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
)
_DAY_WORDS = _WEEKDAYS | {"today", "tomorrow"}

_ALARM_HEADS = (
    re.compile(r"(?:set )?(?:an )?alarm (?:for|at) (?P<rest>.+)"),
    re.compile(r"wake me (?:up )?at (?P<rest>.+)"),
)

_UNIT_SECONDS = {
    "hour": 3600,
    "hours": 3600,
    "hr": 3600,
    "hrs": 3600,
    "minute": 60,
    "minutes": 60,
    "min": 60,
    "mins": 60,
    "second": 1,
    "seconds": 1,
    "sec": 1,
    "secs": 1,
}
_UNIT_SLOTS = {3600: "hours", 60: "minutes", 1: "seconds"}
_UNIT = "(?:" + "|".join(sorted(_UNIT_SECONDS, key=len, reverse=True)) + ")"
_PART = rf"\d{{1,5}} {_UNIT}"
_DURATION = rf"{_PART}(?:(?: and)? {_PART}){{0,2}}"
_TIMER_FOR = re.compile(rf"(?:(?:set|start) )?(?:an? )?timer for (?P<duration>{_DURATION})")
_N_UNIT_TIMER = re.compile(rf"(?:(?:set|start) )?(?:an? )?(?P<duration>{_PART}) timer")
_PART_VALUES = re.compile(rf"(\d+) ({_UNIT})")

_CLOCK_PHRASES = frozenset(
    {
        "what time is it",
        "whats the time",
        "what is the time",
        "whats the date",
        "what is the date",
        "what day is it",
    }
)

# Each stop phrase, with the kind of item it names, if any.
_STOP_PHRASES: dict[str, AlarmKind | None] = {
    "stop": None,
    "stop it": None,
    "turn it off": None,
    "turn off": None,
    "turn off the alarm": "alarm",
    "turn the alarm off": "alarm",
    "turn off the timer": "timer",
    "turn the timer off": "timer",
    "stop the alarm": "alarm",
    "stop the timer": "timer",
    "alarm off": "alarm",
    "timer off": "timer",
    "ok im up": None,
    "okay im up": None,
}
_SNOOZE = re.compile(r"snooze(?: it)?(?:(?: for)? (?P<minutes>\d{1,3}) minutes?)?")
SNOOZE_MINUTES_MAX = 60
"""The longest snooze a phrase or `alarms.ringing` may ask for, in minutes."""


@dataclass(frozen=True, slots=True)
class RingingPhrase:
    """A stop or snooze phrase said while an item rings.

    Attributes:
        action: `stop` ends the ringing item; `snooze` rings it again later.
        minutes: The snooze length said, 1 to `SNOOZE_MINUTES_MAX`; None for the default, and
            always None for `stop`.
        kind: The kind of item the phrase names ("stop the timer" names a timer); None when
            it names none, so it applies to whatever rings.
    """

    action: Literal["stop", "snooze"]
    minutes: int | None
    kind: AlarmKind | None


def match_alarm_intent(text: str, now: datetime, tz: tzinfo) -> FastIntent | None:
    """Match the alarm rows; the time and day must parse in `tz` at `now`.

    Returns:
        `alarms.set` with the user's `time` words and, when one was said, `day`. The capability
        parses them again at the call, so the arguments stay the user's words, as a model call
        passes them. None when no row matches or the time does not parse.
    """
    for head in _ALARM_HEADS:
        match = head.fullmatch(text)
        if match is None:
            continue
        time_text, day = _split_day(match["rest"])
        if not time_text:
            return None
        try:
            parse_alarm_time(time_text, day, now, tz)
        except TimeParseError:
            return None
        args: dict[str, object] = {"time": time_text}
        if day is not None:
            args["day"] = day
        return FastIntent(tool=ALARMS_SET, args=args)
    return None


def match_timer_intent(text: str) -> FastIntent | None:
    """Match the timer rows.

    Returns:
        `timers.start` with `hours`, `minutes` and `seconds` as said, each present only when
        its unit was. None when no row matches, a unit repeats, or the total is outside 1 s to
        24 h, so a request the tool would refuse goes to the model instead.
    """
    for pattern in (_TIMER_FOR, _N_UNIT_TIMER):
        match = pattern.fullmatch(text)
        if match is not None:
            return _timer_intent(match["duration"])
    return None


def match_clock_intent(text: str) -> FastIntent | None:
    """Match the clock rows, giving `clock.now` with no arguments."""
    return FastIntent(tool=CLOCK_NOW, args={}) if text in _CLOCK_PHRASES else None


def match_ringing_phrase(text: str) -> RingingPhrase | None:
    """Match a stop or snooze phrase.

    Stop: "stop", "stop it", "turn it off", "turn off", "turn off the alarm", "turn the alarm
    off", "turn off the timer", "turn the timer off", "stop the alarm", "stop the timer",
    "alarm off", "timer off", "ok I'm up". Snooze: "snooze", optionally "it", optionally "[for]
    N minutes" with N from 1 to `SNOOZE_MINUTES_MAX`.

    Returns:
        The phrase, or None for anything else, including a snooze length out of range.
    """
    if text in _STOP_PHRASES:
        return RingingPhrase(action="stop", minutes=None, kind=_STOP_PHRASES[text])
    match = _SNOOZE.fullmatch(text)
    if match is None:
        return None
    if match["minutes"] is None:
        return RingingPhrase(action="snooze", minutes=None, kind=None)
    minutes = int(match["minutes"])
    if not 1 <= minutes <= SNOOZE_MINUTES_MAX:
        return None
    return RingingPhrase(action="snooze", minutes=minutes, kind=None)


def _split_day(rest: str) -> tuple[str, str | None]:
    """Split "<time> [[on] <day>]" into the time words and the day word."""
    words = rest.split()
    if len(words) < 2 or words[-1] not in _DAY_WORDS:
        return rest, None
    day = words[-1]
    time_words = words[:-1]
    if day in _WEEKDAYS and time_words[-1] == "on":
        time_words = time_words[:-1]
    return " ".join(time_words), day


def _timer_intent(duration: str) -> FastIntent | None:
    args: dict[str, object] = {}
    total = 0
    for amount_text, unit in _PART_VALUES.findall(duration):
        seconds = _UNIT_SECONDS[unit]
        slot = _UNIT_SLOTS[seconds]
        if slot in args:
            return None
        amount = int(amount_text)
        args[slot] = amount
        total += amount * seconds
    if not 1 <= total <= MAX_DURATION_S:
        return None
    return FastIntent(tool=TIMERS_START, args=args)
