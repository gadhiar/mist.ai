"""The user's time zone: one variable, `MIST_TIMEZONE`, and one resolver for every reader.

Memory config reads dates such as "yesterday" in this zone and usage counting buckets turns into
its local days. Both resolve through `resolve_user_timezone`, so they cannot disagree about which
day a turn fell on, and the unset warning goes out once per process however many readers ask.

The resolver returns a `tzinfo`, not a `ZoneInfo`: an unset zone, and a zone that cannot be
loaded because no zone database is installed, both give `datetime.timezone.utc`, which needs no
database. A caller never depends on loading `UTC` from the database.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from datetime import UTC, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones

logger = logging.getLogger(__name__)

USER_TIMEZONE_ENV = "MIST_TIMEZONE"

ZoneLoader = Callable[[str], tzinfo]

# Set once the unset-zone warning has been logged. Startup builds the memory config several
# times, health probes can rebuild it, and usage resolves the zone on every request; one line
# says everything. Tests reset it with monkeypatch.
_unset_warned = False

# Zone names already reported as unloadable for want of a zone database, so each usage
# request (usage resolves on each one) does not repeat the same warning.
_missing_database_warned: set[str] = set()


def resolve_user_timezone(
    environ: Mapping[str, str], *, load_zone: ZoneLoader | None = None
) -> tzinfo:
    """Resolve `MIST_TIMEZONE` from `environ` to the user's zone.

    The zone is loaded first, and the zone database is listed only when that lookup fails, to
    tell a missing database from a bad name; a valid zone never pays for the listing.

    Args:
        environ: Where to read `MIST_TIMEZONE`, normally `os.environ`.
        load_zone: Replaces the zone lookup (`zoneinfo.ZoneInfo`); tests pass a fake.

    Returns:
        `datetime.timezone.utc` when the variable is unset or blank (one WARNING per process),
        or when the zone cannot be loaded because no zone database is installed (one WARNING
        naming the zone); otherwise the loaded zone, a `ZoneInfo` in production.

    Raises:
        ValueError: The name is not a zone in the installed database, or is not a valid zone
            key at all. `ZoneInfoNotFoundError` is a KeyError, so it is re-raised as ValueError
            for callers to catch one type.
    """
    global _unset_warned
    raw = environ.get(USER_TIMEZONE_ENV)
    name = raw.strip() if raw else ""
    if not name:
        if not _unset_warned:
            _unset_warned = True
            logger.warning("%s is unset; dates and local days are read in UTC", USER_TIMEZONE_ENV)
        return UTC
    loader = ZoneInfo if load_zone is None else load_zone
    try:
        return loader(name)
    except ZoneInfoNotFoundError as exc:
        if available_timezones():
            raise ValueError(f"{USER_TIMEZONE_ENV}={name!r} is not a known time zone") from exc
        if name not in _missing_database_warned:
            _missing_database_warned.add(name)
            logger.warning(
                "%s=%r cannot be loaded because no time zone database is installed; "
                "reading dates in UTC",
                USER_TIMEZONE_ENV,
                name,
            )
        return UTC
    except IsADirectoryError as exc:
        # A region such as `America` is a directory of zones in the database, not a zone.
        raise ValueError(
            f"{USER_TIMEZONE_ENV}={name!r} names a region of zones, not a zone"
        ) from exc
    except ValueError as exc:
        # An absolute path, a key outside the database, or a file that is not TZif.
        raise ValueError(f"{USER_TIMEZONE_ENV}={name!r} is not a valid zone name: {exc}") from exc
