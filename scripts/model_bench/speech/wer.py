"""Word error rate for run 3 (mist-model-host plan v2, Run 3).

Both the script text and the engine's text are normalised the same way before they are aligned:
lower-cased, numbers written as words, punctuation removed. Numbers matter because the scripts
spell them out ("ten minutes", "six thirty") while an engine may print digits ("10 minutes",
"6:30"); counting that as an error would measure formatting, not recognition. Stdlib only: the
host computes the rate from the probe's raw text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen"
).split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_SCALES = ((1_000_000_000, "billion"), (1_000_000, "million"), (1_000, "thousand"))
_ORDINAL_IRREGULAR = {
    "one": "first",
    "two": "second",
    "three": "third",
    "five": "fifth",
    "eight": "eighth",
    "nine": "ninth",
    "twelve": "twelfth",
}

_TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_ORDINAL_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b")
_GROUPED_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+\b")
_DECIMAL_RE = re.compile(r"\b(\d+)\.(\d+)\b")
_INT_RE = re.compile(r"\d+")


def int_to_words(n: int) -> str:
    """`n` in words, without "and" ("one hundred five"), as the scripts write numbers."""
    if n < 0:
        return "minus " + int_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, rest = divmod(n, 10)
        return _TENS[tens] + ("" if rest == 0 else " " + _ONES[rest])
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return _ONES[hundreds] + " hundred" + ("" if rest == 0 else " " + int_to_words(rest))
    for scale, name in _SCALES:
        if n >= scale:
            head, rest = divmod(n, scale)
            return int_to_words(head) + f" {name}" + ("" if rest == 0 else " " + int_to_words(rest))
    raise AssertionError("unreachable")


def ordinal_words(n: int) -> str:
    """`n` as an ordinal in words ("second", "twenty first")."""
    words = int_to_words(n).split()
    last = words[-1]
    if last in _ORDINAL_IRREGULAR:
        words[-1] = _ORDINAL_IRREGULAR[last]
    elif last.endswith("y"):
        words[-1] = last[:-1] + "ieth"
    else:
        words[-1] = last + "th"
    return " ".join(words)


def _digits_words(digits: str) -> str:
    return " ".join(_ONES[int(d)] for d in digits)


def _time_words(match: re.Match[str]) -> str:
    # "6:30" is said "six thirty", "7:05" "seven oh five" and "9:00" just "nine", as the scripts
    # say times ("at nine", "six thirty").
    hour, minute = int(match.group(1)), int(match.group(2))
    if minute == 0:
        return f" {int_to_words(hour)} "
    if minute < 10:
        return f" {int_to_words(hour)} oh {int_to_words(minute)} "
    return f" {int_to_words(hour)} {int_to_words(minute)} "


def _number_words(text: str) -> str:
    text = _TIME_RE.sub(_time_words, text)
    text = _GROUPED_RE.sub(lambda m: m.group(0).replace(",", ""), text)
    text = _PERCENT_RE.sub(lambda m: f"{m.group(1)} percent", text)
    text = _ORDINAL_RE.sub(lambda m: f" {ordinal_words(int(m.group(1)))} ", text)
    text = _DECIMAL_RE.sub(
        lambda m: f" {int_to_words(int(m.group(1)))} point {_digits_words(m.group(2))} ", text
    )
    return _INT_RE.sub(lambda m: f" {int_to_words(int(m.group(0)))} ", text)


def normalise(text: str) -> list[str]:
    """The words `text` is scored on: lower-cased, numbers as words, punctuation removed.

    Apostrophes are dropped inside words and every other mark, the pause mark `...` included,
    becomes a space, so a contraction or possessive scores the same whether or not an engine
    prints its apostrophe.
    """
    text = _number_words(text.lower()).replace("&", " and ")
    text = re.sub(r"['’]", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.split()


@dataclass(frozen=True, slots=True)
class WerResult:
    """The edit counts of one alignment and the reference length they are divided by."""

    substitutions: int
    deletions: int
    insertions: int
    reference_words: int

    @property
    def errors(self) -> int:
        """Substitutions, deletions and insertions together."""
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        """Errors over reference words; an empty reference scores the hypothesis length."""
        if self.reference_words == 0:
            return float(self.insertions)
        return self.errors / self.reference_words


def align(reference: list[str], hypothesis: list[str]) -> WerResult:
    """Levenshtein alignment over words, keeping the substitution, deletion and insertion counts.

    Among alignments of equal cost the one with fewer substitutions wins, then fewer deletions,
    so the counts are deterministic.
    """
    # Each cell is (cost, substitutions, deletions, insertions).
    previous = [(j, 0, 0, j) for j in range(len(hypothesis) + 1)]
    for i, ref_word in enumerate(reference, start=1):
        current = [(i, 0, i, 0)]
        for j, hyp_word in enumerate(hypothesis, start=1):
            diag = previous[j - 1]
            if ref_word == hyp_word:
                options = [diag]
            else:
                options = [(diag[0] + 1, diag[1] + 1, diag[2], diag[3])]
            up = previous[j]
            options.append((up[0] + 1, up[1], up[2] + 1, up[3]))
            left = current[j - 1]
            options.append((left[0] + 1, left[1], left[2], left[3] + 1))
            current.append(min(options))
        previous = current
    _, subs, dels, ins = previous[-1]
    return WerResult(subs, dels, ins, len(reference))


def word_error_rate(reference: str, hypothesis: str) -> WerResult:
    """Score `hypothesis` against the script text `reference`, both normalised first."""
    return align(normalise(reference), normalise(hypothesis))
