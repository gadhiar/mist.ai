"""Tests for `normalize_utterance`, the text the fast path and the decision matcher both read."""

from __future__ import annotations

import pytest

from backend.capabilities.intents import normalize_utterance

WAKE_WORDS = ("mist", "hey mist", "ok mist", "okay mist")


class TestPlanCases:
    """The cases T3's acceptance names, each with its expected output."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("What's the time?", "whats the time"),
            ("Set an alarm for 7.30 a.m.", "set an alarm for 7:30 am"),
            ("set a timer for twenty five minutes", "set a timer for 25 minutes"),
            ("a minute timer", "1 minute timer"),
            ("Hey MIST, could you turn off the lights please", "turn off the lights"),
            ("dim the lights to 30%", "dim the lights to 30 percent"),
        ],
    )
    def test_plan_case(self, raw, expected):
        assert normalize_utterance(raw, WAKE_WORDS) == expected


class TestPunctuation:
    def test_apostrophes_are_dropped_not_spaced(self):
        assert normalize_utterance("Don't stop", WAKE_WORDS) == "dont stop"

    def test_curly_apostrophe_is_dropped(self):
        curly = "what" + chr(0x2019) + "s playing"

        assert normalize_utterance(curly, WAKE_WORDS) == "whats playing"

    def test_pm_with_dots_and_space(self):
        assert normalize_utterance("wake me at 6 p.m. tomorrow", WAKE_WORDS) == (
            "wake me at 6 pm tomorrow"
        )

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("wake me at 7a.m.", "wake me at 7am"),
            ("6p.m.", "6pm"),
            ("alarm for 7.30a.m.", "alarm for 7:30am"),
            ("Set an alarm for 7.30 A.M.", "set an alarm for 7:30 am"),
            ("alarm at 7.30am", "alarm at 7:30am"),
            ("alarm at 6 P.M", "alarm at 6 pm"),
            ("alarm at 7.30.", "alarm at 7:30"),
        ],
    )
    def test_meridiem_and_dotted_time_attached_or_spaced(self, raw, expected):
        """The meridiem keeps the spacing it was written with: "7a.m." gives "7am"."""
        assert normalize_utterance(raw, WAKE_WORDS) == expected

    def test_meridiem_needs_its_own_word(self):
        assert normalize_utterance("a.mist rose", WAKE_WORDS) == "a mist rose"

    def test_colon_survives(self):
        assert normalize_utterance("alarm at 7:45!", WAKE_WORDS) == "alarm at 7:45"

    def test_per_cent_becomes_percent(self):
        assert normalize_utterance("volume to 40 per cent", WAKE_WORDS) == ("volume to 40 percent")

    def test_other_punctuation_becomes_space_and_whitespace_collapses(self):
        assert normalize_utterance("  turn   on,the-lamp...  ", WAKE_WORDS) == ("turn on the lamp")

    def test_empty_text_stays_empty(self):
        assert normalize_utterance("   ", WAKE_WORDS) == ""


class TestNumberWords:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("zero", "0"),
            ("seven", "7"),
            ("nineteen", "19"),
            ("twenty", "20"),
            ("ninety nine", "99"),
            ("forty-two", "42"),
            ("set the volume to sixty", "set the volume to 60"),
            ("wake me at six thirty", "wake me at 6 30"),
        ],
    )
    def test_number_word(self, raw, expected):
        assert normalize_utterance(raw, WAKE_WORDS) == expected

    def test_an_before_a_time_unit_is_one(self):
        assert normalize_utterance("set a timer for an hour", WAKE_WORDS) == (
            "set a timer for 1 hour"
        )

    def test_a_before_seconds_unit_is_one(self):
        assert normalize_utterance("a second timer", WAKE_WORDS) == "1 second timer"

    def test_a_before_other_words_is_kept(self):
        assert normalize_utterance("set an alarm for a day trip", WAKE_WORDS) == (
            "set an alarm for a day trip"
        )

    def test_half_an_hour_is_thirty_minutes(self):
        assert normalize_utterance("set a timer for half an hour", WAKE_WORDS) == (
            "set a timer for 30 minutes"
        )


class TestStripping:
    def test_leading_wake_words_and_courtesies_strip_repeatedly(self):
        assert normalize_utterance("Okay MIST, please, can you pause", WAKE_WORDS) == "pause"

    def test_wake_word_must_be_a_whole_word(self):
        assert normalize_utterance("misty mornings", WAKE_WORDS) == "misty mornings"

    def test_trailing_courtesies_strip_repeatedly(self):
        assert normalize_utterance("play some jazz for me now thank you", WAKE_WORDS) == (
            "play some jazz"
        )

    def test_wake_word_in_the_middle_is_kept(self):
        assert normalize_utterance("tell mist a joke", WAKE_WORDS) == "tell mist a joke"

    def test_configured_wake_words_are_normalised_too(self):
        assert normalize_utterance("Hi, Robot! pause", ("hi, robot",)) == "pause"

    def test_no_wake_words_strips_only_courtesies(self):
        assert normalize_utterance("mist, would you pause", ()) == "mist would you pause"

    def test_courtesy_alone_strips_to_empty(self):
        assert normalize_utterance("Thanks!", WAKE_WORDS) == ""
