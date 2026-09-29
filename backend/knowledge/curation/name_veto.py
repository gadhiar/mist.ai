"""Deterministic name veto for Tier-3 (embedding) entity deduplication.

Embedding similarity cannot tell apart names that differ only in a number or a
date. Measured on the live container, distinct extracted pairs such as
'P95' / 'P99 latency', 'March 3' / 'March 13, 2026' and 'Q1' / 'Q2 2026' score
0.96-0.99 on Neo4j's `vector.similarity.cosine` scale, above any usable
threshold. Merging them destroys a distinct fact, so Tier 3 asks this module
whether a candidate is vetoed before taking it.

The rule is purely lexical, so the same two names always give the same answer:

- Numeric tokens: every maximal run of decimal digits, compared by integer
  value, so '03' and '3' are the same token. 'P95' -> {95};
  'March 13, 2026' -> {13, 2026}; 'Q1' -> {1}.
- Month tokens: English month names and their standard abbreviations
  (jan, feb, mar, apr, may, jun, jul, aug, sep, sept, oct, nov, dec), matched
  case-insensitively against whole runs of letters and normalised to the month
  number, so 'March', 'MAR' and 'mar' are one token and 'Mars' is none.

Two names are vetoed when their numeric-token sets differ or their month-token
sets differ. Names with no numeric and no month tokens are never vetoed.
Word-order swaps and sibling or antonym names are not detected here.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

__all__ = ["NameTokens", "name_tokens", "names_conflict"]

# `\d` matches any Unicode decimal digit (category Nd); `_canonical_integer`
# maps each one to its value, so a digit run means the same number in any script.
_DIGIT_RUN = re.compile(r"\d+")
# A run of letters: word characters that are neither digits nor underscore.
_LETTER_RUN = re.compile(r"[^\W\d_]+")

_MONTHS: dict[str, int] = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}


@dataclass(frozen=True, slots=True)
class NameTokens:
    """The numeric and month tokens of one entity name.

    Attributes:
        numbers: Canonical decimal strings of each digit run's integer value
            (no leading zeros; '0' for an all-zero run). Strings rather than
            `int` so an arbitrarily long digit run cannot hit Python's
            int-from-string digit limit; equality is the same as comparing the
            integers.
        months: Month numbers 1-12 for each month name or abbreviation.
    """

    numbers: frozenset[str]
    months: frozenset[int]


def _canonical_integer(run: str) -> str:
    """Return the decimal string of a digit run's integer value."""
    ascii_digits = "".join(str(unicodedata.decimal(ch)) for ch in run)
    return ascii_digits.lstrip("0") or "0"


def name_tokens(name: str) -> NameTokens:
    """Extract the numeric and month tokens of `name`.

    Args:
        name: An entity display name (or id when the node has no display name).

    Returns:
        The name's `NameTokens`.
    """
    numbers = frozenset(_canonical_integer(run) for run in _DIGIT_RUN.findall(name))
    months = frozenset(
        _MONTHS[word]
        for word in (run.casefold() for run in _LETTER_RUN.findall(name))
        if word in _MONTHS
    )
    return NameTokens(numbers=numbers, months=months)


def names_conflict(a: str, b: str) -> bool:
    """Return True when Tier 3 must not merge names `a` and `b`.

    The pair is vetoed when the numeric-token sets differ or the month-token
    sets differ (see the module docstring). The result is symmetric and depends
    only on the two strings.

    Args:
        a: The incoming entity's display name.
        b: The candidate's display name, or its id when the name is null.

    Returns:
        True if the merge is vetoed, False if the names may merge.
    """
    return name_tokens(a) != name_tokens(b)
