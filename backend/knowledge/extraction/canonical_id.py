"""Id canonicalization shared by the normalizer and the eval scorer.

The graph and the scorer import these functions rather than keeping two copies, so a
stored id and a scored id cannot drift apart, and a metric's id never depends on the
LLM's word order, its spelling of a unit, or whether it wrote a decimal point.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from types import MappingProxyType


def canonical_id(name: str) -> str:
    """Canonical entity id: lowercase, hyphenated, alnum+hyphen only.

    Version collapsing is not part of the slug; it is a separate rail that only some
    entity types take.
    """
    s = name.lower().strip()
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"[^a-z0-9\-]", "", s)
    s = re.sub(r"-{2,}", "-", s)
    s = s.strip("-")
    return s or name.lower().replace(" ", "-")


def _slug(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[\s_/]+", "-", s)
    s = re.sub(r"[^a-z0-9\-]", "", s)
    s = re.sub(r"-{2,}", "-", s)
    return s.strip("-")


# One spelling per unit (ADR-054 Decision 5). The canonical forms are the ones the gold
# sets and the retrieval queries already write (`120-ms-p95-latency`,
# `83-tokens-per-second`, `71-percent-peak-cpu`), so those ids stay fixed points. Keys are
# slugged spellings; a unit missing from the table keeps its slug.
_SPELLINGS_BY_CANONICAL: dict[str, tuple[str, ...]] = {
    "percent": ("pct", "percentage", "per-cent"),
    "ms": ("msec", "millisecond", "milliseconds"),
    "seconds": ("s", "sec", "secs", "second"),
    "requests-per-second": (
        "rps",
        "req-s",
        "requests-s",
        "requests-sec",
        "requests-per-sec",
        "request-per-second",
    ),
    "tokens-per-second": (
        "tok-s",
        "tokens-s",
        "tokens-sec",
        "tokens-per-sec",
        "token-per-second",
    ),
    "gb": ("gigabyte", "gigabytes"),
}
UNIT_SPELLINGS: Mapping[str, str] = MappingProxyType(
    {
        spelling: canonical
        for canonical, spellings in _SPELLINGS_BY_CANONICAL.items()
        for spelling in spellings
    }
)

_DIGIT_SEPARATOR_RE = re.compile(r"(?<=\d)[,_ ](?=\d)")
_DECIMAL_RE = re.compile(r"(\d+)\.(\d+)")
# A number token inside a raw id: an integer or a dotted decimal, with optional thousands
# commas. A hyphen-spelled decimal (`0-2`) reads as two numbers.
_NUMBER_TOKEN_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")


def _with_percent_word(text: str) -> str:
    """Spell `%` as a word, since `_slug` would otherwise drop it."""
    return text.replace("%", " percent ")


def canonical_metric_value(value: str | float | int) -> str:
    """A metric value as an id token, with a decimal kept distinct from an integer.

    Digit separators (`,`, `_`, a space between digits) are dropped; a decimal point
    becomes a hyphen, so 3.1 gives `3-1` and never collides with 31; trailing fractional
    zeros are dropped, so `3.80` and 3.8 agree and 12000.0 gives `12000`.

    Args:
        value: The value as the model or the gold wrote it.

    Returns:
        The value's id token.
    """
    text = _DIGIT_SEPARATOR_RE.sub("", str(value).strip().lower())
    match = _DECIMAL_RE.fullmatch(text)
    if match:
        whole, fraction = match.group(1), match.group(2).rstrip("0")
        return f"{whole}-{fraction}" if fraction else whole
    return _slug(re.sub(r"(?<=\d)\.(?=\d)", "-", text))


def canonical_unit(unit: str) -> str:
    """A unit in its one spelling, through `UNIT_SPELLINGS`; any other unit keeps its slug.

    Args:
        unit: The unit as written (`%`, `msec`, `tokens/sec`).

    Returns:
        The canonical spelling, or the unit's slug when the table does not name it.
    """
    slug = _slug(_with_percent_word(str(unit)))
    return UNIT_SPELLINGS.get(slug, slug)


def canonical_metric_id(
    value: str | float | int, unit: str | None, quantity: str | None = None
) -> str:
    """Deterministic Metric id as `<value>-<unit>[-<quantity>]`, the form
    `data/gold/extraction/LABELLING.md` writes.

    Value always leads, so `12000-requests-per-second` and `requests-per-second-12000`
    collapse. The quantity is part of the id so that a p95 and a p99 latency of `120 ms`
    get `120-ms-p95-latency` and `120-ms-p99-latency` rather than one shared `120-ms`.

    The value is spelled by `canonical_metric_value` and the unit by `canonical_unit`, so
    2.4 s gives `2-4-seconds` and 71 % gives `71-percent`. The id joins the non-empty
    parts in that order, so a count with no unit still gets `180-max-queue-depth`. A
    quantity is dropped when the unit already ends with it on a token boundary
    (`percent-coverage` with `coverage`), which keeps an id from repeating its tail when
    the model folds the quantity into the unit.

    Args:
        value: The metric's value.
        unit: The unit, or None or empty for a unitless count.
        quantity: What is measured (`p95 latency`), or None or empty.

    Returns:
        The parts joined with `-`.
    """
    value_slug = canonical_metric_value(value)
    unit_slug = canonical_unit(str(unit)) if unit else ""
    quantity_slug = _slug(str(quantity)) if quantity else ""
    if quantity_slug and (unit_slug == quantity_slug or unit_slug.endswith(f"-{quantity_slug}")):
        quantity_slug = ""
    return "-".join(part for part in (value_slug, unit_slug, quantity_slug) if part)


def canonical_metric_id_from_id(id_str: str) -> str:
    """Metric id from the raw `id` field, for a Metric whose properties cannot rebuild it.

    It reads the raw field rather than its slug, because the slug drops a decimal point.
    When the id holds exactly one number, an integer or a dotted decimal
    (`tool-selection-0.94`), that number, spelled by `canonical_metric_value`, moves to the
    front and the other tokens are slugged in their order, so
    `requests-per-second-12000` and `12000-requests-per-second` collapse. Of the tokens
    after the number, the longest run starting at the first that `UNIT_SPELLINGS` names
    takes its spelling (`120-msec-p95-latency` gives `120-ms-p95-latency`); no later token
    is looked up, so a canonical id is a fixed point. An id with no number, or with two or
    more (a hyphen-spelled decimal counts as two), is returned as `canonical_id` gives it.

    Args:
        id_str: The raw id as the model or the gold wrote it.

    Returns:
        The value-first Metric id.
    """
    text = _with_percent_word(id_str.lower())
    tokens = [t.strip(".,") for t in re.split(r"[^a-z0-9.,]+", text)]
    tokens = [t for t in tokens if t]
    numbers = [i for i, t in enumerate(tokens) if _NUMBER_TOKEN_RE.fullmatch(t)]
    if len(numbers) != 1:
        return canonical_id(id_str)
    i = numbers[0]
    before = [_slug(t) for t in tokens[:i]]
    after = [_slug(t) for t in tokens[i + 1 :]]
    for end in range(len(after), 0, -1):
        spelling = UNIT_SPELLINGS.get("-".join(after[:end]))
        if spelling is not None:
            after = [spelling, *after[end:]]
            break
    parts = [canonical_metric_value(tokens[i]), *before, *after]
    return "-".join(part for part in parts if part)
