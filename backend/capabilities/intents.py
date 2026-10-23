"""Utterance normalisation shared by the fast-path grammars and the decision matcher.

Speech-to-text and typing render the same request in many surface forms ("7.30 a.m.",
"seven thirty am", "Hey MIST, could you ..."). The anchored grammars and the confirmation
matcher are only as reliable as the text they see, so both are written against one canonical
form produced here, and a grammar author writes patterns against that form alone.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

# The ASCII apostrophe, the left and right single quotation marks and the modifier apostrophe.
_APOSTROPHES = re.compile("[" + "".join(map(chr, (0x27, 0x2018, 0x2019, 0x02BC))) + "]")
# "a.m." or "p.m." anywhere a letter does not precede it, so "7a.m." converts as well as
# "7 a.m."; the spacing written before it is kept ("7am", "7 am").
_MERIDIEM = re.compile(r"(?<![^\W\d_])([ap])\.\s*m(?:\.|(?!\w))")
# "7.30" not inside a longer number, so "7.30am" converts as well as "7.30 am".
_DOTTED_TIME = re.compile(r"(?<![\d.])(\d{1,2})\.(\d{2})(?![\d])")
_PER_CENT = re.compile(r"\bper\s+cent\b")
# Everything that is not a word character, whitespace or ":" ("7:30" must survive).
_PUNCTUATION = re.compile(r"[^\w\s:]")

_UNITS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_TIME_UNITS = frozenset({"second", "minute", "hour"})
_ARTICLES = frozenset({"a", "an"})

_LEADING_COURTESIES = (
    ("please",),
    ("can", "you"),
    ("could", "you"),
    ("would", "you"),
    ("will", "you"),
)
_TRAILING_COURTESIES = (
    ("please",),
    ("thanks",),
    ("thank", "you"),
    ("for", "me"),
    ("now",),
)


def normalize_utterance(text: str, wake_words: Sequence[str]) -> str:
    """Return the canonical form of `text` that grammars and the decision matcher read.

    The steps, in order:

    - casefold; "a.m." and "p.m." become "am" and "pm", keeping the spacing before them
      ("7a.m." gives "7am", "7 a.m." gives "7 am"); "7.30" becomes "7:30", also when a
      meridiem follows with no space ("7.30a.m." gives "7:30am"); "%" and "per cent" become
      " percent"; apostrophes are dropped ("don't" becomes "dont"); any
      other punctuation except ":" becomes a space; whitespace collapses;
    - number words from zero to ninety-nine become digits ("twenty five" becomes "25"), "half
      an hour" becomes "30 minutes", and "a" or "an" before second, minute or hour becomes "1";
    - a leading wake word or courtesy ("please", "can you", "could you", "would you", "will
      you") is stripped repeatedly, and so is a trailing "please", "thanks", "thank you", "for
      me" or "now".

    Args:
        text: The user's utterance, as transcribed or typed.
        wake_words: Leading phrases to strip (`MIST_CAPABILITY_WAKE_WORDS`). Each goes through
            the same text steps first, so "Hey, MIST" in the setting matches "hey mist".

    Returns:
        The normalised text; empty when nothing but wake words and courtesies remain.
    """
    words = _number_words(_surface(text).split())
    wake = sorted(
        {tuple(_surface(word).split()) for word in wake_words} - {()},
        key=len,
        reverse=True,
    )
    leading = (*wake, *_LEADING_COURTESIES)
    changed = True
    while changed and words:
        changed = False
        for phrase in leading:
            if tuple(words[: len(phrase)]) == phrase:
                words = words[len(phrase) :]
                changed = True
                break
    changed = True
    while changed and words:
        changed = False
        for phrase in _TRAILING_COURTESIES:
            if len(words) >= len(phrase) and tuple(words[-len(phrase) :]) == phrase:
                words = words[: -len(phrase)]
                changed = True
                break
    return " ".join(words)


def _surface(text: str) -> str:
    """Apply the character-level steps and collapse whitespace."""
    text = text.casefold()
    text = _MERIDIEM.sub(r"\1m", text)
    text = _DOTTED_TIME.sub(r"\1:\2", text)
    text = text.replace("%", " percent")
    text = _PER_CENT.sub("percent", text)
    text = _APOSTROPHES.sub("", text)
    text = _PUNCTUATION.sub(" ", text)
    return " ".join(text.split())


def _number_words(words: list[str]) -> list[str]:
    """Turn number words into digits and spoken durations into "N unit" form."""
    out: list[str] = []
    index = 0
    while index < len(words):
        word = words[index]
        following = words[index + 1] if index + 1 < len(words) else ""
        digit = _UNITS.get(following, 10)
        if word == "half" and words[index + 1 : index + 3] == ["an", "hour"]:
            out.extend(("30", "minutes"))
            index += 3
        elif word in _TENS and 0 < digit < 10:
            out.append(str(_TENS[word] + digit))
            index += 2
        elif word in _TENS:
            out.append(str(_TENS[word]))
            index += 1
        elif word in _UNITS:
            out.append(str(_UNITS[word]))
            index += 1
        elif word in _ARTICLES and following in _TIME_UNITS:
            out.append("1")
            index += 1
        else:
            out.append(word)
            index += 1
    return out
