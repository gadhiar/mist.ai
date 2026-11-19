"""The addressee gate: whether an utterance that holds MIST's name was said to her.

The name spotter answers only "was the name heard". Wake by name would otherwise open a turn for
her own voice saying her name, for a film on the PC saying it, and for every ordinary use of the
word ("the mist rolled in"). The gate turns that hit into a verdict in two steps: `pre_decode`
rejects her own speech and played media before any transcript is decoded, and `decide` reads the
gate decode's text with the vocative rules.

For a server hit, `decide` also reads where the spotter heard the name (`NameAlignment`): the
words the recogniser placed at the hit, and whether the VAD heard speech go on past it.

The module is pure: it reads a `WakeConfig`, strings and timed words, and imports no
controller, ingress or engine, so it runs, and is tested, without a model loaded.
`WakeConfig` is re-exported so that a caller holding only this module can build one.
"""

import re
from dataclasses import dataclass
from typing import Literal

from .budget import WakeConfig
from .speech import HeardWord

__all__ = [
    "AddresseeGate",
    "AtHit",
    "GateCheck",
    "GateVerdict",
    "NameAlignment",
    "WakeConfig",
    "normalize",
    "vocative_match",
]

GateCheck = Literal[
    "phrase", "vocative", "followup", "endpoint", "self_speech", "self_media", "not_addressed"
]
AtHit = Literal["name", "alias", "other", "none"]
"""What the recogniser wrote in the alias window before a hit, for counts only."""
_Mark = Literal["name", "alias"]

_NAME_SLOT = "{name}"
_WORD = r"[a-z0-9]+(?:'[a-z0-9]+)*"
_TEXT_TOKEN = re.compile(rf"{_WORD}|[,?.]")
_TEMPLATE_TOKEN = re.compile(rf"\{{name\}}|{_WORD}|[,?.]")
_PUNCTUATION = frozenset({",", "?", "."})
# Tokens the last-word rule lets follow the name: "what do you reckon, mist?", "mist, please".
_LAST_WORD_TRAILERS = frozenset({"?", ".", "please"})
# What may end a leading name in the full-stop form, before an address start.
_FULL_STOPS = frozenset({".", "?"})
_TYPOGRAPHIC_APOSTROPHES = str.maketrans({"’": "'", "‘": "'"})


@dataclass(frozen=True, slots=True)
class GateVerdict:
    """The gate's answer for one name hit, with the scores the wake event reports.

    `check` names the rule that decided, accepted or not, so that one rejection reason can be
    told apart from another. `at_hit`, when `decide` had timed words, records what overlapped
    the alias window: `name` if a configured name did, else `alias` if an alias did, else
    `other` if any word did, else `none`; it is None otherwise.
    """

    accepted: bool
    check: GateCheck
    score: float
    reference_score: float | None
    at_hit: AtHit | None = None


@dataclass(frozen=True, slots=True)
class NameAlignment:
    """Where the spotter heard the name, so the gate can read the decode there.

    `hit_pos` is the hit's uplink position and `frame_ms` the length of one position. `words`
    are the decode's words with the positions they were heard in, or None when the engine gave
    no timings. `speech_end_pos` is the last frame the VAD judged speech when the decode
    started, or None when no speech is known to follow: a finished utterance's final
    transcript holds all of it, so its end is its own.
    """

    hit_pos: int
    frame_ms: int
    speech_end_pos: int | None
    words: tuple[HeardWord, ...] | None


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


def _aliases(config: WakeConfig) -> frozenset[str]:
    """The configured names' near spellings, less any that is a name itself."""
    names = frozenset(config.names)
    return (
        frozenset(alias for name, words in config.aliases if name in names for alias in words)
        - names
    )


def _in_window(word: HeardWord, alignment: NameAlignment, config: WakeConfig) -> bool:
    """Whether `word` overlaps the alias window, `alias_window_ms` before the hit to the hit."""
    start = alignment.hit_pos - config.alias_window_ms // alignment.frame_ms
    return word.first_pos <= alignment.hit_pos and word.last_pos >= start


def _marks(
    tokens: list[str], config: WakeConfig, alignment: NameAlignment | None
) -> list[_Mark | None]:
    """Each token's reading as the name: a configured name anywhere, an alias only at the hit.

    Each timed word gives the tokens `normalize` gives its text, and in order they are the
    text's tokens, since no token spans whitespace. Words that do not give them belong to
    another text, so no alias is read from them.
    """
    names = frozenset(config.names)
    marks: list[_Mark | None] = ["name" if token in names else None for token in tokens]
    if alignment is None or alignment.words is None:
        return marks
    aliases = _aliases(config)
    per_word = [normalize(word.text) for word in alignment.words]
    if [token for word_tokens in per_word for token in word_tokens] != tokens:
        return marks
    index = 0
    for word, word_tokens in zip(alignment.words, per_word, strict=True):
        at_hit = _in_window(word, alignment, config)
        for token in word_tokens:
            if at_hit and marks[index] is None and token in aliases:
                marks[index] = "alias"
            index += 1
    return marks


def _ends_count(config: WakeConfig, alignment: NameAlignment | None) -> bool:
    """The trailing-speech guard: whether a rule anchored at the utterance's end may count."""
    if alignment is None or alignment.speech_end_pos is None:
        return True
    past_ms = (alignment.speech_end_pos - alignment.hit_pos) * alignment.frame_ms
    return past_ms <= config.trailing_speech_ms


def _slots_match(words: list[str], marks: list[_Mark | None], slots: list[str]) -> bool:
    if not slots or len(words) != len(slots):
        return False
    return all(
        mark is not None if slot == _NAME_SLOT else word == slot
        for word, mark, slot in zip(words, marks, slots, strict=True)
    )


def _phrase(words: list[str], marks: list[_Mark | None], config: WakeConfig, *, ends: bool) -> bool:
    for template in config.phrases:
        slots = _template_words(template)
        if len(words) < len(slots):
            continue
        if _slots_match(words[: len(slots)], marks[: len(slots)], slots):
            return True
        tail = len(words) - len(slots)
        if ends and _slots_match(words[tail:], marks[tail:], slots):
            return True
    return False


def _greeting_at_start(tokens: list[str], marks: list[_Mark | None], config: WakeConfig) -> bool:
    greetings = frozenset(config.greetings)
    if len(tokens) >= 2 and tokens[0] in greetings and marks[1] is not None:
        return True
    return (
        len(tokens) >= 3 and tokens[0] == "good" and tokens[1] in greetings and marks[2] is not None
    )


def _vocative_position(
    tokens: list[str],
    marks: list[_Mark | None],
    config: WakeConfig,
    *,
    aligned: bool,
    ends: bool,
) -> bool:
    stop_before = frozenset(config.stop_before)
    address_starts = frozenset(config.address_starts)

    def counts(index: int) -> bool:
        if marks[index] is None:
            return False
        return index == 0 or tokens[index - 1] not in stop_before

    if counts(0) and len(tokens) >= 2:
        after = tokens[1]
        if after == ",":
            return True
        if marks[0] == "name" and after in address_starts:
            return True
        if aligned and after in _FULL_STOPS and len(tokens) >= 3 and tokens[2] in address_starts:
            return True

    last = len(tokens) - 1
    while last >= 0 and tokens[last] in _LAST_WORD_TRAILERS:
        last -= 1
    if (
        ends
        and last >= 0
        and counts(last)
        and (marks[last] == "name" or last == 0 or tokens[last - 1] == ",")
    ):
        return True

    return any(
        tokens[index - 1] == "," and tokens[index + 1] == "," and counts(index)
        for index in range(1, len(tokens) - 1)
    )


def vocative_match(
    text: str, config: WakeConfig, alignment: NameAlignment | None = None
) -> Literal["phrase", "vocative"] | None:
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

    Without `alignment` that is all. With one, for a server hit, the rules read the name where
    the spotter heard it:
    - The alias window: an alias of a configured name (`config.aliases`) is a name token too,
      but only in a word overlapping the `alias_window_ms` before the hit through the hit, and
      only when `alignment.words` is given. "missed" sounds exactly like "mist", so anywhere
      else it is the ordinary word.
    - The alias punctuation rule: as the first word, an alias counts in rule 4 only before ","
      or the full-stop form below, never before a bare address start ("missed call from mum");
      as the last word, only when "," comes before it or it is the only word ("I just missed."
      is not said to her, "Missed." and "what's the time, missed?" are). Rules 1 and 2 and the
      two commas need words of their own, and treat an aligned alias as a name.
    - The trailing-speech guard: the last-word rule and a template matched at the end count
      only when `speech_end_pos` is None or at most `trailing_speech_ms` past the hit, because
      the gate decodes only `gate_tail_ms` past the hit and a cut-off decode ("A cold mist."
      of "a cold mist hung over the lake") can end at the name. A template matched at the
      start, the greeting and the two commas do not depend on where the decode ends.
    - The full-stop form: a name or alias first, then "." or "?" and a word in
      `address_starts`, counts in rule 4 ("Mist. What's the time?"); "Mist. The fog lifted."
      does not.
    """
    tokens = normalize(text)
    if not tokens:
        return None
    marks = _marks(tokens, config, alignment)
    ends = _ends_count(config, alignment)
    spoken = [index for index, token in enumerate(tokens) if token not in _PUNCTUATION]
    words = [tokens[index] for index in spoken]
    word_marks = [marks[index] for index in spoken]
    if _phrase(words, word_marks, config, ends=ends):
        return "phrase"
    if _greeting_at_start(tokens, marks, config):
        return "vocative"
    if _vocative_position(tokens, marks, config, aligned=alignment is not None, ends=ends):
        return "vocative"
    return None


def _at_hit(config: WakeConfig, alignment: NameAlignment | None) -> AtHit | None:
    if alignment is None or alignment.words is None:
        return None
    heard = [word for word in alignment.words if _in_window(word, alignment, config)]
    if not heard:
        return "none"
    tokens = {token for word in heard for token in normalize(word.text)}
    if tokens & frozenset(config.names):
        return "name"
    if tokens & _aliases(config):
        return "alias"
    return "other"


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

    def decide(
        self,
        *,
        text: str,
        score: float,
        reference_score: float | None,
        alignment: NameAlignment | None = None,
    ) -> GateVerdict:
        """Read the gate decode's text and accept it when a vocative rule finds her addressed.

        `alignment`, for a server hit, lets the rules read the name at the hit
        (`vocative_match`), and records in `at_hit` what the recogniser wrote there.
        """
        check = vocative_match(text, self._config, alignment)
        at_hit = _at_hit(self._config, alignment)
        if check is None:
            return GateVerdict(
                accepted=False,
                check="not_addressed",
                score=score,
                reference_score=reference_score,
                at_hit=at_hit,
            )
        return GateVerdict(
            accepted=True,
            check=check,
            score=score,
            reference_score=reference_score,
            at_hit=at_hit,
        )
