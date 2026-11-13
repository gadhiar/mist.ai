"""Tests for the addressee gate, which decides whether an utterance naming MIST was said to her.

The name spotter hears the name; it cannot tell "mist, dim the lights" from "the mist rolled in",
from her own voice saying her name, or from a film on the PC saying it. Wake by name opens a turn
only when the gate accepts, so a false accept starts a turn nobody asked for and a false reject
ignores her user. Each accepted sentence is pinned with the check that accepted it, because the
check is the reason a rejection reports. The table runs against the shipped `WakeConfig`
defaults, since those are what a fresh install hears with. The module is imported alone, because
it must run without a controller, ingress or engine loaded.
"""

import subprocess
import sys

import pytest

from backend.realtime.addressee import (
    AddresseeGate,
    GateVerdict,
    WakeConfig,
    normalize,
    vocative_match,
)

DEFAULTS = WakeConfig()


class TestNormalize:
    def test_lower_cases_and_keeps_commas_question_marks_and_full_stops(self):
        assert normalize("So, Mist, what's up? Fine.") == [
            "so",
            ",",
            "mist",
            ",",
            "what's",
            "up",
            "?",
            "fine",
            ".",
        ]

    def test_drops_other_punctuation(self):
        assert normalize("Mist! Lights - off") == ["mist", "lights", "off"]

    def test_a_typographic_apostrophe_reads_as_a_plain_one(self):
        assert normalize("What’s up") == ["what's", "up"]

    def test_empty_text_gives_no_tokens(self):
        assert normalize("") == []


class TestVocativeMatchDefaults:
    @pytest.mark.parametrize(
        ("text", "check"),
        [
            ("hey mist, what's the time", "phrase"),
            ("Mist, how's it going?", "phrase"),
            ("mist what's up", "phrase"),
            ("are you there mist", "phrase"),
            ("Mist, set a timer for ten minutes", "vocative"),
            ("mist can you dim the lights", "vocative"),
            ("so, mist, what do you think", "vocative"),
            ("what do you reckon, mist?", "vocative"),
            ("morning mist", "vocative"),
            ("good morning mist", "vocative"),
        ],
    )
    def test_accepts_an_addressed_sentence_with_its_check(self, text, check):
        assert vocative_match(text, DEFAULTS) == check

    @pytest.mark.parametrize(
        "text",
        [
            "I missed the bus",
            "the mist rolled in",
            "a thick mist this morning",
            "mist is a good name",
            # `stop_before` away from the start: "morning" is a noun's modifier here.
            "I love the morning mist",
            # "good" is not the first word, so the greeting rule does not apply.
            "it was a good morning mist",
            # The last-word rule does not count a name after "into".
            "we drove into mist",
            "mistakes were made",
            "",
            "what's the weather like tomorrow",
        ],
    )
    def test_rejects_a_sentence_not_said_to_her(self, text):
        assert vocative_match(text, DEFAULTS) is None


class TestVocativeMatchConfigured:
    def test_a_custom_template_matches_what_the_defaults_reject(self):
        # The lead's ruling of 12 Nov 2026: "ok" replaces v5's "yo", which the shipped greetings
        # already hold, so the defaults would accept "yo mist lights off" as a vocative.
        text = "ok mist lights off"
        custom = WakeConfig(phrases=(*DEFAULTS.phrases, "ok {name}"))

        assert vocative_match(text, DEFAULTS) is None
        assert vocative_match(text, custom) == "phrase"

    def test_the_name_slot_matches_every_configured_name(self):
        two_names = WakeConfig(names=("mist", "misty"))

        assert vocative_match("hey misty", DEFAULTS) is None
        assert vocative_match("hey misty", two_names) == "phrase"
        assert vocative_match("hey mist", two_names) == "phrase"


class TestPreDecode:
    def _gate(self) -> AddresseeGate:
        return AddresseeGate(DEFAULTS, echo_guard_ms=2000)

    def test_her_own_recent_speech_holding_a_name_is_self_speech(self):
        verdict = self._gate().pre_decode(
            score=0.9, reference_score=None, recent_tts_text="I'm Mist, by the way"
        )

        assert verdict == GateVerdict(
            accepted=False, check="self_speech", score=0.9, reference_score=None
        )

    def test_her_recent_speech_counts_when_the_reference_heard_no_name(self):
        # Another session's speaker can reach this microphone, and this session's reference
        # carries only its own playback, so a low reference does not clear her recent speech.
        verdict = self._gate().pre_decode(
            score=0.9, reference_score=0.1, recent_tts_text="I'm Mist, by the way"
        )

        assert verdict == GateVerdict(
            accepted=False, check="self_speech", score=0.9, reference_score=0.1
        )

    @pytest.mark.parametrize("reference_score", [0.5, 0.8])
    def test_a_reference_at_or_above_the_threshold_is_self_media(self, reference_score):
        verdict = self._gate().pre_decode(
            score=0.7, reference_score=reference_score, recent_tts_text=""
        )

        assert verdict == GateVerdict(
            accepted=False, check="self_media", score=0.7, reference_score=reference_score
        )

    def test_self_media_wins_over_self_speech(self):
        verdict = self._gate().pre_decode(
            score=0.7, reference_score=0.6, recent_tts_text="I'm Mist, by the way"
        )

        assert verdict is not None
        assert verdict.check == "self_media"

    @pytest.mark.parametrize(
        ("reference_score", "recent_tts_text"),
        [
            (None, ""),
            (None, "The timer is set for ten minutes."),
            (0.49, "It looks misty outside."),
        ],
    )
    def test_neither_holding_asks_for_a_decode(self, reference_score, recent_tts_text):
        verdict = self._gate().pre_decode(
            score=0.9, reference_score=reference_score, recent_tts_text=recent_tts_text
        )

        assert verdict is None

    def test_the_gate_keeps_its_echo_guard_for_the_controller(self):
        assert AddresseeGate(DEFAULTS, echo_guard_ms=1500).echo_guard_ms == 1500


class TestDecide:
    @pytest.mark.parametrize(
        ("text", "check"),
        [
            ("hey mist, what's the time", "phrase"),
            ("mist can you dim the lights", "vocative"),
        ],
    )
    def test_an_addressed_text_is_accepted_with_its_check_and_both_scores(self, text, check):
        verdict = AddresseeGate(DEFAULTS, echo_guard_ms=2000).decide(
            text=text, score=0.83, reference_score=0.12
        )

        assert verdict == GateVerdict(accepted=True, check=check, score=0.83, reference_score=0.12)

    def test_a_text_not_said_to_her_is_not_addressed_with_both_scores(self):
        verdict = AddresseeGate(DEFAULTS, echo_guard_ms=2000).decide(
            text="the mist rolled in", score=0.61, reference_score=None
        )

        assert verdict == GateVerdict(
            accepted=False, check="not_addressed", score=0.61, reference_score=None
        )


class TestPurity:
    def test_importing_the_gate_loads_no_controller_ingress_or_engine(self):
        probe = (
            "import sys\n"
            "import backend.realtime.addressee\n"
            "for name in sorted(sys.modules):\n"
            "    print(name)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True
        )
        loaded = result.stdout.split()

        assert "backend.realtime.addressee" in loaded
        assert "backend.realtime.turn" not in loaded
        assert "backend.realtime.ingress" not in loaded
        assert not [name for name in loaded if name.startswith("backend.realtime.engines")]
