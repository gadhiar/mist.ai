"""Tests for the lights rows of the fast-path grammar (`lights/intents.py`).

Every utterance goes through `normalize_utterance` first, as the host's fast path does, so the
fixtures are written as a user would say or type them.
"""

from __future__ import annotations

import pytest

from backend.capabilities.contracts import FastIntent
from backend.capabilities.intents import normalize_utterance
from backend.capabilities.lights.intents import SLOT_STOP_WORDS, match_lights_intent

WAKE_WORDS = ("mist", "hey mist", "ok mist", "okay mist")
CATALOG = ("Bedroom", "Kitchen", "Living Room")


def _match(utterance: str, known: tuple[str, ...] = CATALOG) -> FastIntent | None:
    return match_lights_intent(normalize_utterance(utterance, WAKE_WORDS), known)


def _power(on: bool, target: str | None = None) -> FastIntent:
    args: dict[str, object] = {"on": on}
    if target is not None:
        args["target"] = target
    return FastIntent(tool="lights.power", args=args)


def _adjust(percent: int, target: str | None = None) -> FastIntent:
    args: dict[str, object] = {"brightness_pct": percent}
    if target is not None:
        args["target"] = target
    return FastIntent(tool="lights.adjust", args=args)


MATCHING = [
    ("lights off", _power(False)),
    ("Lights on.", _power(True)),
    ("turn off the lights", _power(False)),
    ("Turn on the light, please", _power(True)),
    ("switch off the lamp", _power(False)),
    ("turn off the bedroom lights", _power(False, "bedroom")),
    ("Hey MIST, turn the kitchen lamp on", _power(True, "kitchen")),
    ("switch on the lights in the living room", _power(True, "living room")),
    ("dim the lights to 30 percent", _adjust(30)),
    ("dim the lights to thirty per cent", _adjust(30)),
    ("set the lights to 75%", _adjust(75)),
    ("turn on the kitchen light", _power(True, "kitchen")),
    ("switch the bedroom lights off", _power(False, "bedroom")),
    ("the lights off", _power(False)),
    ("bedroom lights on", _power(True, "bedroom")),
    ("turn lights on", _power(True)),
    ("turn off the bedroom", _power(False, "bedroom")),
    ("switch on kitchen", _power(True, "kitchen")),
    ("turn off the lights in bedroom", _power(False, "bedroom")),
    ("set the living room lights to 75 percent", _adjust(75, "living room")),
    ("dim the bedroom lamp to 10 percent", _adjust(10, "bedroom")),
    ("set the kitchen to 40 percent", _adjust(40, "kitchen")),
    ("dim bedroom to five percent", _adjust(5, "bedroom")),
    ("turn off the garage lights", _power(False, "garage")),
    ("turn on the front porch lamps", _power(True, "front porch")),
    ("Living room lamps off, thanks", _power(False, "living room")),
]

NOT_MATCHING = [
    "the lights were off when I got home",
    "turn off the lights and play jazz",
    "turn on the lamp then dim it",
    "should I turn the lights off",
    "lights",
    "turn off the music",
    "turn off the alarm",
    "turn off the timer",
    "turn off the tv",
    "set the volume to 40 percent",
    "turn off",
    "turn off the bedroom and kitchen lights",
    "dim the lights",
    "set the lights to blue",
    "are the lights on",
    "did you leave the kitchen lights on",
    "what lights are on",
    "turn off the garage",
    "turn off the very big old dusty garage lights",
    "turn off all the lights",
]


class TestMatching:
    def test_fixture_sizes(self):
        assert len(MATCHING) >= 20
        assert len(NOT_MATCHING) >= 15

    @pytest.mark.parametrize(("utterance", "expected"), MATCHING)
    def test_matching_utterances_give_the_expected_call(self, utterance, expected):
        assert _match(utterance) == expected

    @pytest.mark.parametrize("utterance", NOT_MATCHING)
    def test_non_matching_utterances_give_none(self, utterance):
        assert _match(utterance) is None

    def test_brightness_is_passed_unvalidated(self):
        # The tool's arguments model rejects 150, so the call gives `invalid_args`.
        assert _match("dim the lights to 150 percent") == _adjust(150)


class TestTargets:
    @pytest.mark.parametrize(
        "utterance",
        [
            "turn off the hall and stairs lights",
            "switch on the lights in the den then kitchen",
            "kitchen also lights off",
            "dim the desk with shade lamp to 20 percent",
            "turn off the salt and pepper",
        ],
    )
    def test_a_target_with_a_slot_stop_word_never_matches(self, utterance):
        known = (*CATALOG, "Salt and Pepper", "Hall and Stairs")
        assert _match(utterance, known) is None

    def test_slot_stop_words(self):
        assert frozenset({"and", "then", "also", "with"}) == SLOT_STOP_WORDS

    def test_a_known_target_without_a_light_word_needs_the_catalog(self):
        assert _match("turn off the bedroom") == _power(False, "bedroom")
        assert _match("turn off the bedroom", ()) is None
        assert _match("set the kitchen to 40 percent", ()) is None

    def test_a_known_target_matches_ignoring_case(self):
        assert _match("turn off the bedroom", ("BEDROOM",)) == _power(False, "bedroom")

    def test_a_light_named_music_takes_turn_off_the_music(self):
        assert _match("turn off the music", (*CATALOG, "Music")) == _power(False, "music")

    def test_a_target_absent_from_the_catalog_next_to_a_light_word_matches(self):
        assert _match("turn off the garage lights") == _power(False, "garage")
        assert _match("switch on the lights in the attic") == _power(True, "attic")

    def test_a_light_word_that_completes_a_known_name_stays_in_the_target(self):
        known = (*CATALOG, "Bedroom Lamp", "Lamp")
        assert _match("turn off the bedroom lamp", known) == _power(False, "bedroom lamp")
        assert _match("turn the lamp on", known) == _power(True, "lamp")
        assert _match("dim the bedroom lamp to 20 percent", known) == _adjust(20, "bedroom lamp")

    def test_without_the_catalog_name_the_light_word_is_dropped(self):
        assert _match("turn off the bedroom lamp") == _power(False, "bedroom")
