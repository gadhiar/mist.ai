"""The clock capability code reads time through.

Code that needs the time takes a `Clock` by injection, so tests drive time with a fake and
production passes `SystemClock`. `build_now_fn` (`backend/factories.py`) is not a substitute:
`MIST_FIXED_CLOCK` can freeze it for replay, and a frozen clock would stop alarms from firing.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Protocol


class Clock(Protocol):
    """Wall-clock time and sleeping, injectable for tests."""

    def now(self) -> datetime:
        """Return the current time as an aware UTC datetime."""
        ...

    async def sleep(self, seconds: float) -> None:
        """Suspend the caller for `seconds`."""
        ...


class SystemClock:
    """The real clock: `datetime.now(UTC)` and `asyncio.sleep`."""

    def now(self) -> datetime:
        """Return the current time as an aware UTC datetime."""
        return datetime.now(UTC)

    async def sleep(self, seconds: float) -> None:
        """Suspend the caller for `seconds` on the running loop."""
        await asyncio.sleep(seconds)
