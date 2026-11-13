"""The addressee gate: whether an utterance that holds MIST's name was said to her.

The name spotter answers only "was the name heard". Wake by name would otherwise open a turn for
her own voice saying her name, for a film on the PC saying it, and for every ordinary use of the
word ("the mist rolled in"). The gate turns that hit into a verdict in two steps: `pre_decode`
rejects her own speech and played media before any transcript is decoded, and `decide` reads the
gate decode's text with the vocative rules.

The module is pure: it reads a `WakeConfig` and strings, and imports no controller, ingress or
engine, so it runs, and is tested, without a model loaded.
`WakeConfig` is re-exported so that a caller holding only this module can build one.
"""

import re
from dataclasses import dataclass
from typing import Literal

from .budget import WakeConfig

__all__ = [
    "AddresseeGate",
    "GateCheck",
    "GateVerdict",
    "WakeConfig",
    "normalize",
    "vocative_match",
]

GateCheck = Literal[
    "phrase", "vocative", "followup", "endpoint", "self_speech", "self_media", "not_addressed"
]

_NAME_SLOT = "{name}"
_WORD = r"[a-z0-9]+(?:'[a-z0-9]+)*"
_TEXT_TOKEN = re.compile(rf"{_WORD}|[,?.]")
_TEMPLATE_TOKEN = re.compile(rf"\{{name\}}|{_WORD}|[,?.]")
_PUNCTUATION = frozenset({",", "?", "."})
# Tokens the last-word rule lets follow the name: "what do you reckon, mist?", "mist, please".
_LAST_WORD_TRAILERS = frozenset({"?", ".", "please"})
_TYPOGRAPHIC_APOSTROPHES = str.maketrans({"’": "'", "‘": "'"})


@dataclass(frozen=True, slots=True)
class GateVerdict:
    """The gate's answer for one name hit, with the scores the wake event reports.

    `check` names the rule that decided, accepted or not, so that one rejection reason can be
    told apart from another.
    """

    accepted: bool
    check: GateCheck
    score: float
    reference_score: float | None


def normalize(text: str) -> list[str]:
    """Split a transcript into lower-case words and the punctuation the rules read.

    Commas, "?" and "." are kept as tokens because they mark a vocative ("so, mist, what do you
    think"); other punctuation carries nothing the rules use and is dropped. A typographic
    apostrophe reads as a plain one, so "what’s" from one recogniser equals "what's" from
    another and from the configured word lists.
    """
    return _TEXT_TOKEN.findall(text.lower().translate(_TYPOGRAPHIC_APOSTROPHES))


def _template_words(template: str) -> list[str]:
    return [
        token
        for token in _TEMPLATE_TOKEN.findall(template.lower().translate(_TYPOGRAPHIC_APOSTROPHES))
        if token not in _PUNCTUATION
    ]


def _slots_match(words: list[str], slots: list[str], names: frozenset[str]) -> bool:
    if not slots or len(words) != len(slots):
        return False
    return all(
        word in names if slot == _NAME_SLOT else word == slot
        for word, slot in zip(words, slots, strict=True)
    )


def _phrase(words: list[str], config: WakeConfig, names: frozenset[str]) -> bool:
    for template in config.phrases:
        slots = _template_words(template)
        if len(words) < len(slots):
            continue
        if _slots_match(words[: len(slots)], slots, names):
            return True
        if _slots_match(words[len(words) - len(slots) :], slots, names):
            return True
    return False


def _greeting_at_start(tokens: list[str], config: WakeConfig, names: frozenset[str]) -> bool:
    greetings = frozenset(config.greetings)
    if len(tokens) >= 2 and tokens[0] in greetings and tokens[1] in names:
        return True
    return (
        len(tokens) >= 3 and tokens[0] == "good" and tokens[1] in greetings and tokens[2] in names
    )


def _vocative_position(tokens: list[str], config: WakeConfig, names: frozenset[str]) -> bool:
    stop_before = frozenset(config.stop_before)
    address_starts = frozenset(config.address_starts)

    def counts(index: int) -> bool:
        if tokens[index] not in names:
            return False
        return index == 0 or tokens[index - 1] not in stop_before

    if counts(0) and len(tokens) >= 2 and (tokens[1] == "," or tokens[1] in address_starts):
        return True

    last = len(tokens) - 1
    while last >= 0 and tokens[last] in _LAST_WORD_TRAILERS:
        last -= 1
    if last >= 0 and counts(last):
        return True

    return any(
        tokens[index - 1] == "," and tokens[index + 1] == "," and counts(index)
        for index in range(1, len(tokens) - 1)
    )


def vocative_match(text: str, config: WakeConfig) -> Literal["phrase", "vocative"] | None:
    """Return which rule finds `text` addressed to her, or None when none does.

    A name token is any of `config.names`. The rules run in this order and the first match wins:

    1. `phrase`: a template matches the utterance's first words or its last words, its `{name}`
       slot matching a name token. Punctuation is skipped on both sides when comparing, because
       recognisers place commas unreliably: "mist what's up" matches "{name}, what's up", and
       "are you there mist?" matches "are you there {name}".
    2. `vocative`, a greeting at the start: the first token is a greeting, or the first is "good"
       and the second a greeting, and the next token is a name ("morning mist", "good morning
       mist"). It is read before rule 3 because at the start of an utterance "morning" greets.
    3. A name token directly after a word in `stop_before` never counts for rule 4, so "I love the
       morning mist" and "we drove into mist" match nothing.
    4. `vocative` by position: the first token is a name followed by a comma or a word in
       `address_starts`; or the last word is a name, followed only by "?", "." or "please"; or a
       name stands between two commas.
    """
    tokens = normalize(text)
    if not tokens:
        return None
    names = frozenset(config.names)
    words = [token for token in tokens if token not in _PUNCTUATION]
    if _phrase(words, config, names):
        return "phrase"
    if _greeting_at_start(tokens, config, names):
        return "vocative"
    if _vocative_position(tokens, config, names):
        return "vocative"
    return None


class AddresseeGate:
    """Stage 2 of wake by name: turns a name hit into an accepted or rejected verdict.

    `echo_guard_ms` is how far back the controller asks the registry for her spoken text before
    calling `pre_decode`; the gate holds it so that the self-speech policy lives in one object.
    """

    def __init__(self, config: WakeConfig, *, echo_guard_ms: int) -> None:
        self._config = config
        self._names = frozenset(config.names)
        self.echo_guard_ms = echo_guard_ms

    def pre_decode(
        self, *, score: float, reference_score: float | None, recent_tts_text: str
    ) -> GateVerdict | None:
        """Reject a hit her own output explains, or return None to decode and call `decide`.

        `self_media` is checked first: a reference score at or above `kws_threshold` means the
        PC played the name, which is direct acoustic evidence, while her recent text only says
        she might have. `self_speech` does not wait for the reference to be absent: a low
        reference clears only this session's playback, and another session's speaker (the room
        unit's, say) can reach this microphone, which is why `recent_tts_text` spans every
        session. Any name token in her recent text counts, whatever its position, because the
        spotter hears the sound of the name, not its grammar.
        """
        if reference_score is not None and reference_score >= self._config.kws_threshold:
            return GateVerdict(
                accepted=False, check="self_media", score=score, reference_score=reference_score
            )
        if any(token in self._names for token in normalize(recent_tts_text)):
            return GateVerdict(
                accepted=False, check="self_speech", score=score, reference_score=reference_score
            )
        return None

    def decide(self, *, text: str, score: float, reference_score: float | None) -> GateVerdict:
        """Read the gate decode's text and accept it when a vocative rule finds her addressed."""
        check = vocative_match(text, self._config)
        if check is None:
            return GateVerdict(
                accepted=False,
                check="not_addressed",
                score=score,
                reference_score=reference_score,
            )
        return GateVerdict(accepted=True, check=check, score=score, reference_score=reference_score)
