"""Tests for the alarms, timers and clock rows of the fast-path grammar, and the ringing phrases.

Every utterance goes through `normalize_utterance` first, as the host passes it, so the fixtures
are written the way a person types or a transcriber renders them.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from backend.capabilities.alarms.intents import (
    RingingPhrase,
    match_alarm_intent,
    match_clock_intent,
    match_ringing_phrase,
    match_timer_intent,
)
from backend.capabilities.contracts import FastIntent
from backend.capabilities.intents import normalize_utterance

WAKE_WORDS = ("mist", "hey mist", "ok mist", "okay mist")
NOW = datetime(2026, 1, 5, 7, 0, tzinfo=UTC)
"""Monday 5 January 2026, 07:00 UTC."""


def _text(raw: str) -> str:
    return normalize_utterance(raw, WAKE_WORDS)


ALARM_MATCHES = [
    ("set an alarm for 7", {"time": "7"}),
    ("wake me up at 6:30 tomorrow", {"time": "6:30", "day": "tomorrow"}),
    ("Set an alarm for 7.30 a.m.", {"time": "7:30 am"}),
    ("alarm for 6am", {"time": "6am"}),
    ("set alarm at 9 pm", {"time": "9 pm"}),
    ("an alarm for noon", {"time": "noon"}),
    ("wake me at 5:45 on Friday", {"time": "5:45", "day": "friday"}),
    ("Hey MIST, set an alarm for seven thirty pm please", {"time": "7 30 pm"}),
    ("set an alarm for 19:30", {"time": "19:30"}),
    ("wake me up at 6 tomorrow", {"time": "6", "day": "tomorrow"}),
    ("set an alarm for 8 am monday", {"time": "8 am", "day": "monday"}),
    ("Could you set an alarm for midnight", {"time": "midnight"}),
    ("alarm at 10:15 pm today", {"time": "10:15 pm", "day": "today"}),
    ("set an alarm for in 20 minutes", {"time": "in 20 minutes"}),
    ("wake me up at 7am on Saturday", {"time": "7am", "day": "saturday"}),
    ("set an alarm for 6:30 on sunday", {"time": "6:30", "day": "sunday"}),
]

ALARM_MISSES = [
    "I set an alarm yesterday",
    "set an alarm for whenever",
    "what time does the store open",
    "timer",
    "set an alarm",
    "set an alarm for 25:00",
    "set an alarm for 7 tomorrow morning",
    "wake me up",
    "cancel my alarm",
    "alarm",
    "set an alarm for 6:30 am today",
    "set an alarm for 7 and turn off the lights",
    "set an alarm for 13 pm",
    "how do I set an alarm for 7",
    "set an alarm for 7 in the morning",
    "wake me at 7 every day",
    "set 2 alarms for 7",
]

TIMER_MATCHES = [
    ("10 minute timer", {"minutes": 10}),
    ("set a timer for 1 hour 30 minutes", {"hours": 1, "minutes": 30}),
    ("set a timer for twenty five minutes", {"minutes": 25}),
    ("start a 5 minute timer", {"minutes": 5}),
    ("timer for 30 seconds", {"seconds": 30}),
    ("set a timer for half an hour", {"minutes": 30}),
    ("a minute timer", {"minutes": 1}),
    ("set a timer for an hour", {"hours": 1}),
    ("start a timer for 2 hours and 15 minutes", {"hours": 2, "minutes": 15}),
    ("set a 90 second timer", {"seconds": 90}),
    ("Hey mist, set a timer for 3 minutes please", {"minutes": 3}),
    ("timer for 1 hour 5 minutes 10 seconds", {"hours": 1, "minutes": 5, "seconds": 10}),
    ("set timer for 45 mins", {"minutes": 45}),
    ("start a 2 hour timer", {"hours": 2}),
    ("set a timer for 12 minutes now", {"minutes": 12}),
    ("an 8 minute timer", {"minutes": 8}),
]

TIMER_MISSES = [
    "timer",
    "set a timer",
    "how long is left on my timer",
    "cancel the timer",
    "set a timer for 0 minutes",
    "set a timer for 25 hours",
    "set a timer for 5 minutes 3 minutes",
    "set a timer for a while",
    "I set a timer for 10 minutes",
    "10 minute",
    "set a timer for 10 minutes and play jazz",
    "set a timer for 1 hour 2 minutes 3 seconds 4 seconds",
    "what timers do I have",
    "timer for 10",
    "set a 10 minute alarm",
    "set a timer for 10 days",
]

CLOCK_MATCHES = [
    "what time is it",
    "What time is it?",
    "what's the time",
    "What's the time?",
    "whats the time",
    "what is the time",
    "what's the date",
    "What is the date?",
    "what day is it",
    "Hey MIST, what time is it",
    "what time is it please",
    "MIST what's the date",
    "what's the time now",
    "Okay MIST, what day is it?",
    "what is the date please",
    "what's the date, thanks",
]

CLOCK_MISSES = [
    "what time does the store open",
    "what time is the meeting",
    "what is the time in tokyo",
    "what day is it today",
    "what date is christmas",
    "what's the weather",
    "time",
    "the time",
    "what time is it in london",
    "is it time to go",
    "what day is the meeting",
    "what is the date tomorrow",
    "whats the time difference",
    "what time zone am I in",
    "tell me a time",
    "set an alarm for 7",
]


class TestAlarmRows:
    def test_the_fixtures_meet_the_minimum_sizes(self):
        assert len(ALARM_MATCHES) >= 15
        assert len(ALARM_MISSES) >= 15

    @pytest.mark.parametrize(("raw", "args"), ALARM_MATCHES)
    def test_a_matching_utterance_gives_alarms_set(self, raw, args):
        assert match_alarm_intent(_text(raw), NOW, UTC) == FastIntent("alarms.set", args)

    @pytest.mark.parametrize("raw", ALARM_MISSES)
    def test_a_non_matching_utterance_gives_none(self, raw):
        assert match_alarm_intent(_text(raw), NOW, UTC) is None

    def test_a_time_the_parser_rejects_is_not_a_match(self):
        # 6:30 am has passed today at 07:00, so `parse_alarm_time` refuses "today"; the same
        # words a day earlier match.
        text = _text("set an alarm for 6:30 am today")

        assert match_alarm_intent(text, NOW, UTC) is None
        assert match_alarm_intent(text, NOW - timedelta(hours=1), UTC) == FastIntent(
            "alarms.set", {"time": "6:30 am", "day": "today"}
        )

    def test_the_time_is_checked_in_the_users_zone(self):
        # 07:00 UTC is 02:00 in UTC-5, so 6:30 am is still ahead there today.
        text = _text("set an alarm for 6:30 am today")
        zone = timezone(timedelta(hours=-5))

        assert match_alarm_intent(text, NOW, zone) == FastIntent(
            "alarms.set", {"time": "6:30 am", "day": "today"}
        )


class TestTimerRows:
    def test_the_fixtures_meet_the_minimum_sizes(self):
        assert len(TIMER_MATCHES) >= 15
        assert len(TIMER_MISSES) >= 15

    @pytest.mark.parametrize(("raw", "args"), TIMER_MATCHES)
    def test_a_matching_utterance_gives_timers_start(self, raw, args):
        assert match_timer_intent(_text(raw)) == FastIntent("timers.start", args)

    @pytest.mark.parametrize("raw", TIMER_MISSES)
    def test_a_non_matching_utterance_gives_none(self, raw):
        assert match_timer_intent(_text(raw)) is None

    def test_twenty_four_hours_is_the_longest_timer(self):
        assert match_timer_intent("set a timer for 24 hours") == FastIntent(
            "timers.start", {"hours": 24}
        )
        assert match_timer_intent("set a timer for 24 hours 1 second") is None


class TestClockRows:
    def test_the_fixtures_meet_the_minimum_sizes(self):
        assert len(CLOCK_MATCHES) >= 15
        assert len(CLOCK_MISSES) >= 15

    @pytest.mark.parametrize("raw", CLOCK_MATCHES)
    def test_a_matching_utterance_gives_clock_now(self, raw):
        assert match_clock_intent(_text(raw)) == FastIntent("clock.now", {})

    @pytest.mark.parametrize("raw", CLOCK_MISSES)
    def test_a_non_matching_utterance_gives_none(self, raw):
        assert match_clock_intent(_text(raw)) is None


STOP_PHRASES = [
    ("stop", None),
    ("stop it", None),
    ("turn it off", None),
    ("turn off", None),
    ("turn off the alarm", "alarm"),
    ("turn the alarm off", "alarm"),
    ("turn off the timer", "timer"),
    ("turn the timer off", "timer"),
    ("stop the alarm", "alarm"),
    ("stop the timer", "timer"),
    ("alarm off", "alarm"),
    ("timer off", "timer"),
    ("ok I'm up", None),
]


class TestRingingPhrases:
    @pytest.mark.parametrize(("raw", "kind"), STOP_PHRASES)
    def test_each_stop_phrase_stops(self, raw, kind):
        assert match_ringing_phrase(_text(raw)) == RingingPhrase("stop", None, kind)

    def test_snooze_takes_the_default(self):
        assert match_ringing_phrase(_text("Snooze")) == RingingPhrase("snooze", None, None)

    def test_snooze_takes_minutes(self):
        assert match_ringing_phrase(_text("snooze 5 minutes")) == RingingPhrase("snooze", 5, None)
        assert match_ringing_phrase(_text("snooze for ten minutes")) == RingingPhrase(
            "snooze", 10, None
        )

    @pytest.mark.parametrize(
        "raw",
        [
            "turn off the lights",
            "stop the music",
            "snooze 0 minutes",
            "snooze 61 minutes",
            "what time is it",
            "stop playing",
            "set an alarm for 7",
            "",
        ],
    )
    def test_anything_else_is_not_a_ringing_phrase(self, raw):
        assert match_ringing_phrase(_text(raw)) is None
