"""Tests for the addressee gate, which decides whether a name hit was said to MIST.

The name spotter hears the name; it cannot tell "mist, dim the lights" from "the mist rolled in",
from her own voice saying her name, or from a film on the PC saying it. Wake by name opens a turn
only when the gate accepts, so a false accept starts a turn nobody asked for and a false reject
ignores her user. Her own output is ruled out by `pre_decode`, and the acoustic verifier's score
decides the rest through `verified`; no transcript text takes part, so the module's checks are
pinned to those four. The checks run against the shipped `WakeConfig` defaults, since those are
what a fresh install hears with. The module is imported alone, because it must run without a
controller, ingress or engine loaded.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from backend.realtime import addressee as addressee_module
from backend.realtime.addressee import AddresseeGate, GateVerdict, WakeConfig, normalize

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

    def test_an_alias_in_her_recent_speech_is_not_her_name(self):
        # "must" and "miss" are common in her replies; counting them would reject genuine wakes
        # for the whole echo guard. Her own playback is the reference score's to catch.
        verdict = self._gate().pre_decode(
            score=0.9, reference_score=None, recent_tts_text="You must have missed it, miss."
        )

        assert verdict is None

    def test_the_gate_keeps_its_echo_guard_for_the_controller(self):
        assert AddresseeGate(DEFAULTS, echo_guard_ms=1500).echo_guard_ms == 1500


class TestVerified:
    """The verifier's verdict: its score against the threshold for the spotter's cutoff."""

    @staticmethod
    def _gate(threshold: float | None = 0.6, asked: list[float] | None = None) -> AddresseeGate:
        def thresholds(cutoff: float) -> float | None:
            if asked is not None:
                asked.append(cutoff)
            return threshold

        return AddresseeGate(DEFAULTS, echo_guard_ms=2000, thresholds=thresholds)

    @pytest.mark.parametrize("verify_score", [0.6, 0.61, 0.99], ids=["at", "above", "well-above"])
    def test_a_score_at_or_above_the_threshold_is_verified(self, verify_score):
        verdict = self._gate().verified(
            score=0.83, reference_score=0.12, verify_score=verify_score, cutoff=0.5
        )

        assert verdict == GateVerdict(
            accepted=True, check="verified", score=0.83, reference_score=0.12
        )

    def test_a_score_below_the_threshold_is_not_addressed(self):
        verdict = self._gate().verified(
            score=0.83, reference_score=None, verify_score=0.59, cutoff=0.5
        )

        assert verdict == GateVerdict(
            accepted=False, check="not_addressed", score=0.83, reference_score=None
        )

    def test_no_verifier_score_is_not_addressed(self):
        # A failed verify, a short window or no verifier at all reaches the gate as None.
        verdict = self._gate().verified(
            score=0.9, reference_score=None, verify_score=None, cutoff=0.5
        )

        assert verdict == GateVerdict(
            accepted=False, check="not_addressed", score=0.9, reference_score=None
        )

    def test_a_cutoff_with_no_threshold_is_not_addressed(self):
        verdict = self._gate(threshold=None).verified(
            score=0.9, reference_score=None, verify_score=1.0, cutoff=0.005
        )

        assert verdict == GateVerdict(
            accepted=False, check="not_addressed", score=0.9, reference_score=None
        )

    def test_a_gate_without_thresholds_verifies_nothing(self):
        verdict = AddresseeGate(DEFAULTS, echo_guard_ms=2000).verified(
            score=0.9, reference_score=None, verify_score=1.0, cutoff=0.5
        )

        assert verdict == GateVerdict(
            accepted=False, check="not_addressed", score=0.9, reference_score=None
        )

    def test_the_threshold_is_read_for_the_given_cutoff(self):
        asked: list[float] = []

        self._gate(asked=asked).verified(
            score=0.9, reference_score=None, verify_score=0.7, cutoff=216 / 255
        )

        assert asked == [216 / 255]

    def test_verified_is_a_gate_check(self):
        assert "verified" in addressee_module.GateCheck.__args__


class TestNoTextRule:
    """No transcript text decides a wake (A66): the verifier and her own output do."""

    def test_the_gate_checks_are_the_verifier_s_and_the_self_checks(self):
        assert addressee_module.GateCheck.__args__ == (
            "verified",
            "self_speech",
            "self_media",
            "not_addressed",
        )

    def test_the_module_exports_the_gate_its_verdict_and_the_tokeniser_only(self):
        assert sorted(addressee_module.__all__) == [
            "AddresseeGate",
            "GateCheck",
            "GateVerdict",
            "WakeConfig",
            "normalize",
        ]

    def test_a_verdict_carries_its_check_and_the_spotter_s_scores_only(self):
        assert list(GateVerdict.__dataclass_fields__) == [
            "accepted",
            "check",
            "score",
            "reference_score",
        ]

    def test_the_gate_decides_by_the_verifier_or_her_own_output_only(self):
        public = sorted(name for name in vars(AddresseeGate) if not name.startswith("_"))

        assert public == ["Thresholds", "pre_decode", "verified"]


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

    def test_the_gate_imports_only_the_wake_settings(self):
        tree = ast.parse(Path(addressee_module.__file__).read_text(encoding="utf-8"))
        relative = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.level
        }
        absolute = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and not node.level
        }

        assert relative == {"budget"}
        assert absolute == {"re", "dataclasses", "typing"}
