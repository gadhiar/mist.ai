"""Tests for the addressee gate, which decides whether an utterance naming MIST was said to her.

The name spotter hears the name; it cannot tell "mist, dim the lights" from "the mist rolled in",
from her own voice saying her name, or from a film on the PC saying it. Wake by name opens a turn
only when the gate accepts, so a false accept starts a turn nobody asked for and a false reject
ignores her user. Each accepted sentence is pinned with the check that accepted it, because the
check is the reason a rejection reports. The table runs against the shipped `WakeConfig`
defaults, since those are what a fresh install hears with. The module is imported alone, because
it must run without a controller, ingress or engine loaded.
"""

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

from backend.realtime import addressee as addressee_module
from backend.realtime.addressee import (
    AddresseeGate,
    GateVerdict,
    NameAlignment,
    WakeConfig,
    normalize,
    vocative_match,
)
from backend.realtime.budget import DEFAULT_NAME_ALIASES
from backend.realtime.speech import HeardWord
from scripts.wake import record

DEFAULTS = WakeConfig()
ALIASES = dict(DEFAULT_NAME_ALIASES)["mist"]
FRAME_MS = 20
WORD_FRAMES = 15  # 300 ms a word
FIRST_POS = 100
SPEECH_ON = 30  # frames: speech went on 600 ms past the hit


def timed(text: str) -> tuple[HeardWord, ...]:
    """`text`'s words, each heard for 300 ms one after another from frame 100."""
    return tuple(
        HeardWord(word, FIRST_POS + index * WORD_FRAMES, FIRST_POS + (index + 1) * WORD_FRAMES - 1)
        for index, word in enumerate(text.split())
    )


def hit_on(
    text: str, index: int, *, speech_past: int = 0, offset: int = 0, words: bool = True
) -> NameAlignment:
    """A hit at the end of word `index`, moved `offset` frames, with speech `speech_past` after.

    The spotter hits once a name has been heard, so a hit at a word is at its last frame.
    """
    heard = timed(text)
    hit = heard[index].last_pos + offset
    return NameAlignment(hit, FRAME_MS, hit + speech_past, heard if words else None)


def word_index(text: str, wanted: frozenset[str]) -> int:
    """The first word of `text` whose tokens hold one of `wanted`."""
    return next(
        index
        for index, word in enumerate(text.split())
        if any(token in wanted for token in normalize(word))
    )


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


def _truncated_sentences() -> list[object]:
    """Each homophone and fog sentence cut after its name or alias word, and after the next.

    A gate decode runs only `gate_tail_ms` past the hit, so it can end there; the recogniser may
    close it with a full stop, which is what makes "A cold mist." look final.
    """
    wanted = frozenset({"mist", *ALIASES})
    cases: dict[str, int] = {}
    for sentence in (*record.MISSED, *record.FOG):
        words = sentence.split()
        at = word_index(sentence, wanted)
        for end in (at + 1, at + 2):
            if end <= len(words):
                for ending in ("", "."):
                    cases.setdefault(" ".join(words[:end]) + ending, at)
    return [pytest.param(text, at, id=text) for text, at in cases.items()]


def _aliased_prompts() -> list[object]:
    """The recorded wake prompts as written, commas kept, with "mist" spelled as each alias."""
    prompts = [*record.NAME_ALONE, *(prompt for pair in record.WAKE_PHRASES for prompt in pair)]
    return [
        pytest.param(re.sub(r"\bmist\b", alias, prompt), alias, id=f"{prompt} as {alias}")
        for prompt in prompts
        for alias in ALIASES
    ]


class TestTheNameAtTheHit:
    @pytest.mark.parametrize(("text", "at"), _truncated_sentences())
    def test_a_sentence_cut_off_at_the_name_or_an_alias_while_speech_goes_on_is_rejected(
        self, text, at
    ):
        alignment = hit_on(text, at, speech_past=SPEECH_ON)

        assert vocative_match(text, DEFAULTS, alignment) is None

    def test_the_guard_closes_the_hole_a_cut_off_name_leaves_without_it(self):
        text = "A cold mist."

        assert vocative_match(text, DEFAULTS) == "vocative"
        assert vocative_match(text, DEFAULTS, hit_on(text, 2, speech_past=SPEECH_ON)) is None
        assert vocative_match(text, DEFAULTS, hit_on(text, 2)) == "vocative"

    @pytest.mark.parametrize(("text", "alias"), _aliased_prompts())
    def test_a_recorded_prompt_with_an_alias_at_the_hit_and_speech_ended_is_accepted(
        self, text, alias
    ):
        alignment = hit_on(text, word_index(text, frozenset({alias})))

        assert vocative_match(text, DEFAULTS, alignment) is not None

    @pytest.mark.parametrize("offset", [30, -30], ids=["600-ms-after", "600-ms-before"])
    @pytest.mark.parametrize(("text", "alias"), _aliased_prompts())
    def test_the_same_prompt_with_the_hit_600_ms_from_the_alias_is_rejected(
        self, text, alias, offset
    ):
        alignment = hit_on(text, word_index(text, frozenset({alias})), offset=offset)

        assert vocative_match(text, DEFAULTS, alignment) is None

    @pytest.mark.parametrize(
        ("text", "alias", "check"),
        [
            # An alias before a bare address start, or last after an ordinary word, is the word.
            ("missed call from mum", "missed", None),
            ("I just missed.", "missed", None),
            ("Missed what's the time", "missed", None),
            # Its punctuation marks the vocative.
            ("Missed. What's the time?", "missed", "vocative"),
            ("what's the time, missed?", "missed", "vocative"),
            ("Missed, set a timer", "missed", "vocative"),
            ("Missed.", "missed", "vocative"),
            # The risks A62 names: these are said to a person, and wake her on a hit.
            ("Hi miss", "miss", "vocative"),
            ("thank you, miss", "miss", "vocative"),
        ],
    )
    def test_an_aligned_alias_needs_the_punctuation_of_a_vocative(self, text, alias, check):
        alignment = hit_on(text, word_index(text, frozenset({alias})))

        assert vocative_match(text, DEFAULTS, alignment) == check

    @pytest.mark.parametrize(
        ("text", "check"),
        [
            ("missed, what's up", None),
            ("hey missed", None),
            ("Missed.", None),
            ("Mist. What's the time?", None),
            ("A cold mist.", "vocative"),
            ("hey mist, what's the time", "phrase"),
        ],
    )
    def test_without_an_alignment_the_verdicts_are_today_s(self, text, check):
        assert vocative_match(text, DEFAULTS) == check
        assert AddresseeGate(DEFAULTS, echo_guard_ms=2000).decide(
            text=text, score=0.9, reference_score=None
        ) == GateVerdict(
            accepted=check is not None,
            check=check or "not_addressed",
            score=0.9,
            reference_score=None,
            at_hit=None,
        )

    @pytest.mark.parametrize(
        ("text", "at", "speech_past", "check"),
        [
            # No timings: no alias counts, wherever the hit was.
            ("hey missed", 1, 0, None),
            ("missed, what's up", 0, 0, None),
            # The names keep their rules, and the guard still reads the speech end. "hey mist"
            # is the whole decode and ends at the name, so its template is guarded; the
            # greeting rule, which is not, accepts it.
            ("hey mist", 1, SPEECH_ON, "vocative"),
            ("A cold mist.", 2, SPEECH_ON, None),
            ("A cold mist.", 2, 0, "vocative"),
        ],
    )
    def test_without_words_only_names_count_under_the_guard(self, text, at, speech_past, check):
        alignment = hit_on(text, at, speech_past=speech_past, words=False)

        assert vocative_match(text, DEFAULTS, alignment) == check

    @pytest.mark.parametrize(
        ("text", "speech_past", "check"),
        [
            # A template that covers the whole decode and ends at the name is anchored at the
            # end too: with speech going on past the hit, the decode was cut at the name.
            ("are you there missed", SPEECH_ON, None),
            ("are you there mist", SPEECH_ON, None),
            ("are you there missed", 0, "phrase"),
            ("are you there mist", 0, "phrase"),
        ],
    )
    def test_a_whole_decode_template_ending_at_the_name_is_guarded(self, text, speech_past, check):
        alignment = hit_on(text, 3, speech_past=speech_past)

        assert vocative_match(text, DEFAULTS, alignment) == check

    def test_words_of_another_text_count_no_alias(self):
        text = "hey missed"
        alignment = NameAlignment(114 + 15, FRAME_MS, 129, timed("hey there"))

        assert vocative_match(text, DEFAULTS, alignment) is None

    @pytest.mark.parametrize(
        ("text", "check"),
        [
            ("Mist. What's the time?", "vocative"),
            ("Mist? Turn the lights off.", "vocative"),
            ("Missed. Can you dim the lights?", "vocative"),
            ("Mist. The fog lifted.", None),
            ("Missed. The bus left.", None),
        ],
    )
    def test_the_full_stop_form_needs_an_address_start_after_it(self, text, check):
        alignment = hit_on(text, 0, speech_past=SPEECH_ON)

        assert vocative_match(text, DEFAULTS, alignment) == check

    @pytest.mark.parametrize("text", ["so are you there mist", "so are you there missed"])
    def test_a_template_matched_at_the_end_counts_only_when_speech_ended_at_the_hit(self, text):
        last = len(text.split()) - 1
        # 100 ms past the hit still counts: the VAD's last speech window can end after it.
        ended = hit_on(text, last, speech_past=5)
        went_on = hit_on(text, last, speech_past=6)

        assert vocative_match(text, DEFAULTS, ended) == "phrase"
        assert vocative_match(text, DEFAULTS, went_on) is None

    @pytest.mark.parametrize(
        ("text", "at", "check"),
        [
            ("mist, what's up", 0, "phrase"),
            ("hey mist, turn the lights off", 1, "phrase"),
            ("mist can you dim the lights", 0, "vocative"),
            ("so, mist, what do you think", 1, "vocative"),
            ("missed, what's up", 0, "phrase"),
            ("hey missed, turn the lights off", 1, "phrase"),
        ],
    )
    def test_a_name_first_or_greeting_prompt_with_speech_going_on_is_accepted(
        self, text, at, check
    ):
        # The guard reads only the rules anchored at the end; these are said from the start.
        alignment = hit_on(text, at, speech_past=100)

        assert vocative_match(text, DEFAULTS, alignment) == check

    @pytest.mark.parametrize(
        ("text", "at", "offset", "at_hit"),
        [
            ("hey mist", 1, 0, "name"),
            ("hey missed", 1, 0, "alias"),
            ("missed mist", 1, 0, "name"),
            ("hello there", 1, 0, "other"),
            ("hello there", 1, 60, "none"),
        ],
    )
    def test_the_verdict_records_what_was_written_at_the_hit(self, text, at, offset, at_hit):
        alignment = hit_on(text, at, offset=offset)

        verdict = AddresseeGate(DEFAULTS, echo_guard_ms=2000).decide(
            text=text, score=0.7, reference_score=None, alignment=alignment
        )

        assert verdict.at_hit == at_hit

    def test_an_alignment_without_words_records_nothing_at_the_hit(self):
        verdict = AddresseeGate(DEFAULTS, echo_guard_ms=2000).decide(
            text="hey mist",
            score=0.7,
            reference_score=0.1,
            alignment=hit_on("hey mist", 1, words=False),
        )

        assert verdict == GateVerdict(
            accepted=True, check="phrase", score=0.7, reference_score=0.1, at_hit=None
        )

    def test_an_aligned_alias_is_accepted_with_its_check_scores_and_kind(self):
        text = "what's the time, missed?"

        verdict = AddresseeGate(DEFAULTS, echo_guard_ms=2000).decide(
            text=text, score=0.62, reference_score=None, alignment=hit_on(text, 3)
        )

        assert verdict == GateVerdict(
            accepted=True, check="vocative", score=0.62, reference_score=None, at_hit="alias"
        )

    def test_aliases_turned_off_count_nothing_at_the_hit(self):
        text = "hey missed"
        config = WakeConfig(aliases=())

        assert vocative_match(text, config, hit_on(text, 1)) is None

    def test_an_alias_of_a_name_not_configured_counts_nothing(self):
        text = "hey missed"
        config = WakeConfig(names=("nova",))

        assert vocative_match(text, config, hit_on(text, 1)) is None


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

    def test_the_gate_imports_only_the_wake_settings_and_the_speech_contract(self):
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

        assert relative == {"budget", "speech"}
        assert absolute == {"re", "dataclasses", "typing"}
