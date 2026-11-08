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

The read, cancel and re-arm rows (plan v2, Fast-path grammar additions), so the common requests
run with no model decode and work at the off power level:

- `[set] [an] alarm (for|at) <time> <repeat>` and `wake me [up] at <time> <repeat>`, giving
  `alarms.set` with `repeat` and, for `custom`, `days`. `<repeat>` is what `parse_repeat` reads;
  the time must be a clock time, as a repeat keeps a wall time and "in 20 minutes" names none;
- `[set] [my] <saved tag> alarm [for <time>]`, giving `alarms.set` with `label`;
- `[start] [my|the] <saved tag> timer`, giving `timers.start` with `label`;
- `(list|show) my alarms`, "what alarms do i have" and "what are my alarms", giving
  `alarms.list`; `(list|show) my saved alarms`, giving it with `scope` `saved`;
- `(list|show) my timers`, "what timers do i have" and `how long is left [on my timer]`, giving
  `timers.list`;
- `cancel my <time> alarm` and `cancel [my|the] <saved tag> alarm`, giving `alarms.cancel` with
  `time` or `label`;
- `cancel [my|the] timer`, giving `timers.cancel` with no arguments: the capability then
  cancels the one live timer, or says none runs, or asks which among several, so the grammar
  needs no live count; and `cancel [my|the] <saved tag> timer`, giving it with `label`.

A `<saved tag>` must name a label in the caller's saved tags (`AlarmRecords.tag_labels`, held in
memory, so the grammar stays synchronous), compared as `normalise_label` gives them, which
ignores case. Words that name no saved tag are not a match, and the model gets them.

The ringing phrases are not fast-path rows. The host asks for them first, while an item rings
(pre-LLM step 1), so "turn off" stops a ringing alarm rather than reaching lights, and a bare
"stop" never collides with a proposal's "no".
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, tzinfo
from typing import Literal

from backend.errors import TimeParseError

from ..contracts import FastIntent
from .records import MAX_DURATION_S, AlarmKind, normalise_label
from .timeparse import parse_alarm_time, parse_repeat

ALARMS_SET = "alarms.set"
ALARMS_LIST = "alarms.list"
ALARMS_CANCEL = "alarms.cancel"
TIMERS_START = "timers.start"
TIMERS_LIST = "timers.list"
TIMERS_CANCEL = "timers.cancel"
CLOCK_NOW = "clock.now"

_WEEKDAY_ORDER = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_WEEKDAYS = frozenset(_WEEKDAY_ORDER)
_DAY_WORDS = _WEEKDAYS | {"today", "tomorrow"}

_ALARM_READS: dict[str, dict[str, object]] = {
    "list my alarms": {},
    "show my alarms": {},
    "what alarms do i have": {},
    "what are my alarms": {},
    "list my saved alarms": {"scope": "saved"},
    "show my saved alarms": {"scope": "saved"},
}
_TIMER_READS = frozenset(
    {
        "list my timers",
        "show my timers",
        "what timers do i have",
        "how long is left",
        "how long is left on my timer",
    }
)
_TAG_ALARM = re.compile(r"(?P<head>.+?) alarm(?: for (?P<time>.+))?")
_TAG_TIMER = re.compile(r"(?P<head>.+) timer")
_CANCEL_ALARM = re.compile(r"cancel (?P<head>.+) alarm")
_CANCEL_TIMER = re.compile(r"cancel (?:(?:my|the) )?timer")
_CANCEL_TAG_TIMER = re.compile(r"cancel (?P<head>.+) timer")
# The optional leading words each tag row allows, one group per bracket of its row, in order.
_SET_TAG_WORDS = (("set",), ("my",))
_START_TAG_WORDS = (("start",), ("my", "the"))
_CANCEL_TAG_WORDS = (("my", "the"),)

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


def match_alarms(text: str, now: datetime, tz: tzinfo, tags: frozenset[str]) -> FastIntent | None:
    """Match every alarm row: set by time or repeat, re-arm a saved tag, list and cancel.

    Args:
        text: `normalize_utterance` output.
        now: The caller's now, for the time checks.
        tz: The user's zone.
        tags: The saved alarm tags, normalised (`AlarmRecords.tag_labels("alarm")`).

    Returns:
        The call a row gives, or None when no row matches.
    """
    if text in _ALARM_READS:
        return FastIntent(tool=ALARMS_LIST, args=dict(_ALARM_READS[text]))
    cancel = _CANCEL_ALARM.fullmatch(text)
    if cancel is not None:
        return _alarm_cancel(cancel["head"], now, tz, tags)
    return match_alarm_intent(text, now, tz) or _alarm_tag(text, now, tz, tags)


def match_timers(text: str, tags: frozenset[str]) -> FastIntent | None:
    """Match every timer row: start by duration, re-arm a saved tag, list and cancel.

    Args:
        text: `normalize_utterance` output.
        tags: The saved timer tags, normalised (`AlarmRecords.tag_labels("timer")`).

    Returns:
        The call a row gives, or None when no row matches.
    """
    if text in _TIMER_READS:
        return FastIntent(tool=TIMERS_LIST, args={})
    if _CANCEL_TIMER.fullmatch(text) is not None:
        return FastIntent(tool=TIMERS_CANCEL, args={})
    cancel = _CANCEL_TAG_TIMER.fullmatch(text)
    if cancel is not None:
        label = _saved_tag(cancel["head"], _CANCEL_TAG_WORDS, tags)
        return None if label is None else FastIntent(tool=TIMERS_CANCEL, args={"label": label})
    started = match_timer_intent(text)
    if started is not None:
        return started
    tagged = _TAG_TIMER.fullmatch(text)
    if tagged is None:
        return None
    label = _saved_tag(tagged["head"], _START_TAG_WORDS, tags)
    return None if label is None else FastIntent(tool=TIMERS_START, args={"label": label})


def match_alarm_intent(text: str, now: datetime, tz: tzinfo) -> FastIntent | None:
    """Match the alarm rows that set by time; the time and day must parse in `tz` at `now`.

    Returns:
        `alarms.set` with the user's `time` words and, when one was said, `day`; or, after a
        repeat phrase, `repeat` and, for `custom`, `days` (weekday names, Monday first). The
        capability parses the words again at the call, so the arguments stay the user's words,
        as a model call passes them. None when no row matches or the time does not parse.
    """
    for head in _ALARM_HEADS:
        match = head.fullmatch(text)
        if match is None:
            continue
        repeating = _repeat_intent(match["rest"], now, tz)
        if repeating is not None:
            return repeating
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


def _clock_time(time_text: str, now: datetime, tz: tzinfo) -> datetime | None:
    """Return the local instant a clock time names next, or None for elapsed or unread words.

    Elapsed time ("in 20 minutes") names one instant rather than a wall time, which neither a
    repeat nor a cancel by time can use.
    """
    if time_text.startswith("in "):
        return None
    try:
        return parse_alarm_time(time_text, None, now, tz).astimezone(tz)
    except TimeParseError:
        return None


def _repeat_intent(rest: str, now: datetime, tz: tzinfo) -> FastIntent | None:
    """Read "<time> <repeat>", trying each split of the words, shortest time first."""
    words = rest.split()
    for split in range(1, len(words)):
        time_text = " ".join(words[:split])
        first = _clock_time(time_text, now, tz)
        if first is None:
            continue
        rule = parse_repeat(" ".join(words[split:]), hour=first.hour, minute=first.minute)
        if rule is None:
            continue
        args: dict[str, object] = {"time": time_text, "repeat": rule.kind}
        if rule.days is not None:
            args["days"] = [_WEEKDAY_ORDER[day] for day in sorted(rule.days)]
        return FastIntent(tool=ALARMS_SET, args=args)
    return None


def _alarm_tag(text: str, now: datetime, tz: tzinfo, tags: frozenset[str]) -> FastIntent | None:
    """Match `[set] [my] <saved tag> alarm [for <time>]`."""
    match = _TAG_ALARM.fullmatch(text)
    if match is None:
        return None
    label = _saved_tag(match["head"], _SET_TAG_WORDS, tags)
    if label is None:
        return None
    args: dict[str, object] = {"label": label}
    time_text = match["time"]
    if time_text is not None:
        try:
            parse_alarm_time(time_text, None, now, tz)
        except TimeParseError:
            return None
        args["time"] = time_text
    return FastIntent(tool=ALARMS_SET, args=args)


def _alarm_cancel(head: str, now: datetime, tz: tzinfo, tags: frozenset[str]) -> FastIntent | None:
    """Match the words between "cancel" and "alarm": `[my|the] <saved tag>` or `my <time>`.

    A saved tag is tried first: a label names one record, where a time may match several.
    """
    label = _saved_tag(head, _CANCEL_TAG_WORDS, tags)
    if label is not None:
        return FastIntent(tool=ALARMS_CANCEL, args={"label": label})
    if not head.startswith("my "):
        return None
    time_text = head[len("my ") :]
    if _clock_time(time_text, now, tz) is None:
        return None
    return FastIntent(tool=ALARMS_CANCEL, args={"time": time_text})


def _saved_tag(
    head: str, optional: tuple[tuple[str, ...], ...], tags: frozenset[str]
) -> str | None:
    """Return the saved tag `head` names once its optional leading words are dropped.

    Each group of `optional` is one bracket of the row ("[set] [my]"), dropped or kept in
    order. When more than one reading names a saved tag, the one keeping the most words wins,
    so a tag called "my gym" is not read as "gym".

    Returns:
        The label as said, or None when no reading names a saved tag.
    """
    readings = sorted(
        set(_tag_readings(head.split(), optional)), key=lambda r: (len(r), r), reverse=True
    )
    for label in readings:
        if normalise_label(label) in tags:
            return label
    return None


def _tag_readings(words: list[str], optional: tuple[tuple[str, ...], ...]) -> Iterator[str]:
    if not optional:
        if words:
            yield " ".join(words)
        return
    group, rest = optional[0], optional[1:]
    yield from _tag_readings(words, rest)
    if words and words[0] in group:
        yield from _tag_readings(words[1:], rest)


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
