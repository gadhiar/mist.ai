"""Night extraction stopped on purpose: the host-written paused marker.

The daily instance runs its extraction services only inside a night window,
and never while a game holds the GPU. `scripts/daily/instance.py extraction
apply` converges the two extraction containers on that schedule, and before
it stops them, or while it starts them, it writes
`<MIST_OPS_DIR>/extraction-paused`:

    {"v": 1, "since": <UTC>, "reason": <r>, "until": <UTC or null>}

`<r>` is one of `window`, `mode`, `gaming`, `starting` or `unconfigured`. The
CLI removes the file only once both services are healthy and still wanted,
and the backend reads it read-only, as it reads `gpu-yielded`. While it holds,
`/health` reports the stopped service as `paused` rather than as a `down`
that would call the schedule a fault.

Unlike `gpu-yielded`, whose bare existence is the signal, this marker is read
for its content, and only a well-formed, current one counts:
- `until` bounds a hold the CLI expects to end: for `window` it is the next
  window start plus 15 minutes, for `starting` the start's deadline. A marker
  past its `until` is a hold that should have ended, such as a start that hung
  or a CLI that died between steps, so it is read as absent and the probe
  reports what the service is actually doing.
- A malformed or unreadable marker is read as absent, with a warning. Reading
  it as a pause would hide a stopped service all day on a file nobody can show
  the CLI meant, while reading it as absent costs at worst a `down` that is
  true of the service.

Each warning is logged once until the marker's condition changes, because the
probe reads the file every cycle.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# The file name the host CLI writes under `MIST_OPS_DIR`. `scripts/daily/instance.py`
# holds the same constant; each side's tests pin it to plan v4's contract.
EXTRACTION_PAUSED_MARKER = "extraction-paused"

MARKER_VERSION = 1

HOLD_REASONS: frozenset[str] = frozenset({"window", "mode", "gaming", "starting", "unconfigured"})

_KEYS: frozenset[str] = frozenset({"v", "since", "reason", "until"})


@dataclass(frozen=True, slots=True)
class ExtractionHold:
    """A current, well-formed paused marker.

    Attributes:
        since: When the CLI paused extraction, UTC.
        reason: One of `HOLD_REASONS`.
        until: When the hold is expected to end, UTC, or None for a hold that
            lasts until the CLI removes it.
    """

    since: datetime
    reason: str
    until: datetime | None


class _MalformedMarkerError(ValueError):
    """The marker exists but does not meet the contract; the message says how."""


class ExtractionHoldReader:
    """Read the paused marker on demand, as the `extraction_service` probe asks.

    Holds no state but the warning it last logged: every `current` call reads
    the file and the clock again, so a marker the CLI writes or removes is seen
    on the next probe cycle.
    """

    def __init__(self, marker_path: Path, *, wall_clock: Callable[[], datetime]) -> None:
        """Bind the reader to one marker path.

        Args:
            marker_path: `<MIST_OPS_DIR>/extraction-paused`.
            wall_clock: Timezone-aware wall time, compared with `until`, which
                the CLI writes in UTC.
        """
        self._marker_path = marker_path
        self._wall_clock = wall_clock
        self._last_warning: str | None = None

    def current(self) -> ExtractionHold | None:
        """Return the hold the marker describes, or None when there is none in force.

        Blocking file I/O: the probe calls it through `asyncio.to_thread`.

        Returns:
            The hold, or None when the marker is absent, past its `until`,
            malformed or unreadable.
        """
        try:
            text = self._marker_path.read_text(encoding="utf-8")
        except (FileNotFoundError, NotADirectoryError):
            self._last_warning = None
            return None
        except (OSError, UnicodeDecodeError) as exc:
            self._warn(f"cannot be read ({type(exc).__name__})")
            return None

        try:
            hold = _parse(text)
        except _MalformedMarkerError as exc:
            self._warn(f"is malformed ({exc})")
            return None

        if hold.until is not None and hold.until <= self._wall_clock():
            self._warn(
                f"is past its until {hold.until.isoformat()} (reason {hold.reason}); "
                "the hold should have ended"
            )
            return None
        self._last_warning = None
        return hold

    def _warn(self, detail: str) -> None:
        """Log one warning per distinct condition, so a lasting one does not repeat."""
        if detail == self._last_warning:
            return
        self._last_warning = detail
        logger.warning(
            "Extraction paused marker %s %s; reading it as absent", self._marker_path, detail
        )


def _parse(text: str) -> ExtractionHold:
    """Validate a marker body against the contract.

    Raises:
        _MalformedMarkerError: The body is not the contract's JSON object.
    """
    try:
        body = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _MalformedMarkerError("not JSON") from exc
    if not isinstance(body, dict):
        raise _MalformedMarkerError("not a JSON object")
    missing = _KEYS - body.keys()
    if missing:
        raise _MalformedMarkerError(f"missing {sorted(missing)}")
    # `bool` is an `int`, and `True == 1`; the version must be the integer itself.
    version = body["v"]
    if isinstance(version, bool) or not isinstance(version, int) or version != MARKER_VERSION:
        raise _MalformedMarkerError(f"version {version!r}")
    reason = body["reason"]
    if not isinstance(reason, str) or reason not in HOLD_REASONS:
        raise _MalformedMarkerError(f"reason {reason!r}")
    since = _parse_utc(body["since"], "since")
    if since is None:
        raise _MalformedMarkerError("since is null")
    return ExtractionHold(since=since, reason=reason, until=_parse_utc(body["until"], "until"))


def _parse_utc(value: object, key: str) -> datetime | None:
    """Parse an ISO-8601 timestamp, reading one without an offset as UTC."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise _MalformedMarkerError(f"{key} is not a string")
    try:
        moment = datetime.fromisoformat(value)
        # An offset at the edge of `datetime`'s range has no UTC equivalent,
        # and `astimezone` raises `OverflowError` for it rather than `ValueError`.
        return moment.astimezone(UTC) if moment.tzinfo is not None else moment.replace(tzinfo=UTC)
    except ValueError as exc:
        raise _MalformedMarkerError(f"{key} is not ISO-8601") from exc
    except OverflowError as exc:
        raise _MalformedMarkerError(f"{key} is outside the UTC range") from exc


def build_extraction_hold_reader(
    ops_dir: str | None,
    *,
    wall_clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> ExtractionHoldReader | None:
    """Build the reader on `<ops_dir>/extraction-paused`, or None without an ops dir.

    None when `ops_dir` (`MIST_OPS_DIR`) is unset or blank: the dev stack has
    no host CLI pausing extraction, so its probe must behave exactly as it did
    before the marker existed, rather than look for a file relative to the
    working directory.

    Args:
        ops_dir: The `MIST_OPS_DIR` value.
        wall_clock: Timezone-aware wall time.

    Returns:
        The reader, or None.
    """
    if ops_dir is None or not ops_dir.strip():
        return None
    return ExtractionHoldReader(
        Path(ops_dir.strip()) / EXTRACTION_PAUSED_MARKER, wall_clock=wall_clock
    )
