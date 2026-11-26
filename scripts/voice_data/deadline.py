"""A job's deadline: `--until <HH:MM>` in host local time, as `--deadline-utc <ISO 8601 Z>`.

A container gets no time zone and reads its clock as UTC, so a wall-clock time given on the host
means nothing inside the job. The host turns it into an absolute UTC instant here, and the job
compares its own UTC clock against that. `HH:MM` names the next time the host's clock shows it:
later today, or tomorrow when that time has passed. Daylight-saving changes are the host
zone's, applied by the C library through `datetime.astimezone`.

A deadline under 10 minutes away leaves no time to load a model and finish one line, and one
more than 24 hours away cannot come from `HH:MM` except across a 25-hour day, where it is more
likely a typo than a plan; both are refused.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, time, timedelta

from .errors import InputError

MIN_AHEAD = timedelta(minutes=10)
MAX_AHEAD = timedelta(hours=24)
_HH_MM = re.compile(r"(?P<h>[01][0-9]|2[0-3]):(?P<m>[0-5][0-9])")
_UTC_Z = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")


def deadline_utc(until: str, *, now: datetime) -> datetime:
    """Return the next instant the host's local clock shows `until`, in UTC.

    Args:
        until: A 24-hour `HH:MM` in the host's local time.
        now: The current time, timezone-aware.

    Raises:
        InputError: `until` is not `HH:MM`, or the instant is not 10 minutes to 24 h ahead.
    """
    match = _HH_MM.fullmatch(until)
    if match is None:
        raise InputError("refused: --until takes a 24-hour local time as HH:MM")
    wall_now = now.astimezone().replace(tzinfo=None)
    wall = datetime.combine(wall_now.date(), time(int(match["h"]), int(match["m"])))
    if wall <= wall_now:
        wall += timedelta(days=1)
    # A naive datetime's `astimezone` reads it as the host's local time, daylight rules included.
    when = wall.astimezone().astimezone(UTC)
    ahead = when - now
    if not MIN_AHEAD <= ahead <= MAX_AHEAD:
        raise InputError("refused: --until must be 10 minutes to 24 h ahead")
    return when


def format_utc(when: datetime) -> str:
    """Return a timezone-aware instant as `YYYY-MM-DDTHH:MM:SSZ`, the job's argument form."""
    return when.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(text: str) -> datetime:
    """Parse `YYYY-MM-DDTHH:MM:SSZ` as a UTC instant.

    Raises:
        ValueError: `text` is not in that form.
    """
    if not _UTC_Z.fullmatch(text):
        raise ValueError("a deadline is ISO 8601 UTC, YYYY-MM-DDTHH:MM:SSZ")
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
