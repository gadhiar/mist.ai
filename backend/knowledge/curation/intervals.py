"""Closed-open valid-time intervals with precision-aware parsing (C1, design 4.5).

Every valid-time bound is an (instant, precision) normalized to a closed-open
range: "2024" covers [2024-01-01, 2025-01-01). A bound used as a FROM takes
the range start; used as a TO it takes the range end (the stated period is
included). The ALWAYS sentinel ("-inf") marks facts stated as "always/since
forever" -- distinct from "started when mentioned" (which defaults to the
utterance's recorded_at).

All datetimes are UTC-aware; naive inputs are assumed UTC. Comparisons are
strict (<) so touching closed-open intervals do not overlap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

ALWAYS = "-inf"

_YEAR_RE = re.compile(r"^(\d{4})$")
_MONTH_RE = re.compile(r"^(\d{4})-(\d{2})$")
_DAY_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")


@dataclass(frozen=True, slots=True)
class Interval:
    """Closed-open [start, end). None start = -infinity; None end = open."""

    start: datetime | None
    end: datetime | None


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _parse_range(value: str) -> tuple[datetime, datetime] | None:
    """Parse a date-prefix string into its covered [start, end) range."""
    m = _YEAR_RE.match(value)
    if m:
        y = int(m.group(1))
        return datetime(y, 1, 1, tzinfo=UTC), datetime(y + 1, 1, 1, tzinfo=UTC)
    m = _MONTH_RE.match(value)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        nxt = (y + 1, 1) if mo == 12 else (y, mo + 1)
        return datetime(y, mo, 1, tzinfo=UTC), datetime(nxt[0], nxt[1], 1, tzinfo=UTC)
    m = _DAY_RE.match(value)
    if m:
        d = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=UTC)
        return d, d + timedelta(days=1)
    try:
        d = _aware(datetime.fromisoformat(value))
    except ValueError:
        return None
    return d, d


def parse_from_bound(value: str | None) -> datetime | None:
    """Lower bound of the stated period. None for ALWAYS/missing/unparsable."""
    if not value or value == ALWAYS:
        return None
    rng = _parse_range(value.strip())
    return rng[0] if rng else None


def parse_to_bound(value: str | None) -> datetime | None:
    """Exclusive upper bound covering the stated period. None when open/unknown."""
    if not value or value == ALWAYS:
        return None
    rng = _parse_range(value.strip())
    return rng[1] if rng else None


def overlaps(a: Interval, b: Interval) -> bool:
    """Closed-open intersection is non-empty. None-aware (-inf / +inf).

    Empty intervals ([t, t) retraction markers) hold no instants and overlap
    nothing: without this guard a backdated re-assertion after a RETRACT
    would REINFORCE the empty marker instead of appending a new open version,
    leaving the fact permanently not-current.
    """
    if is_empty(a) or is_empty(b):
        return False
    a_start = a.start or datetime.min.replace(tzinfo=UTC)
    b_start = b.start or datetime.min.replace(tzinfo=UTC)
    a_end = a.end or datetime.max.replace(tzinfo=UTC)
    b_end = b.end or datetime.max.replace(tzinfo=UTC)
    return a_start < b_end and b_start < a_end


def is_empty(iv: Interval) -> bool:
    """[x, y) with x >= y holds no instants (retraction markers use [t, t))."""
    return iv.start is not None and iv.end is not None and iv.start >= iv.end
