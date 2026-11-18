"""How fast the device listener lets a token be guessed.

Docker Desktop's port publishing replaces the client's address with the proxy's, so the peer
address cannot key a limit. The limiter keys on the device id the token claims: 5 failures in
60 s block that id for 300 s, and a registered device whose id an attacker sprays is blocked
only for as long as the spray lasts. A global cap stops a spray of made-up ids: more than 30
failures in a minute pause every authentication for 60 s, logged once per pause.

Only checked failures count. An attempt refused because its id is blocked or because
authentication is paused is never checked, so it adds nothing, and a blocked id cannot extend
its own block or pause the other devices.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Protocol

logger = logging.getLogger(__name__)

PER_ID = (5, 60.0, 300.0)
"""Failures, window in seconds and block in seconds for one claimed device id."""
GLOBAL = (30, 60.0, 60.0)
"""Failures tolerated in the window, window in seconds and pause in seconds for every id."""


class Clock(Protocol):
    """A monotonic clock in seconds; tests pass a fake."""

    def monotonic(self) -> float:
        """Seconds from an arbitrary origin, never going back."""
        ...


class AuthLimiter:
    """Per-id blocks and the global pause.

    Args:
        clock: The monotonic clock the windows are measured on.
        per_id: `(failures, window_s, block_s)`: `failures` within `window_s` block the id.
        global_: `(failures, window_s, pause_s)`: more than `failures` within `window_s`
            pause every authentication.
    """

    def __init__(
        self,
        clock: Clock,
        per_id: tuple[int, float, float] = PER_ID,
        global_: tuple[int, float, float] = GLOBAL,
    ) -> None:
        self._clock = clock
        self._id_failures, self._id_window, self._id_block = per_id
        self._global_failures, self._global_window, self._global_pause = global_
        self._by_id: dict[str, deque[float]] = {}
        self._blocked_until: dict[str, float] = {}
        self._recent: deque[float] = deque()
        self._paused_until: float | None = None

    def paused(self) -> bool:
        """Whether every authentication is paused now."""
        until = self._paused_until
        if until is None:
            return False
        if self._clock.monotonic() >= until:
            self._paused_until = None
            return False
        return True

    def allowed(self, device_id: str | None) -> bool:
        """Whether an attempt claiming `device_id` (None: no valid id) may be checked now."""
        if self.paused():
            return False
        if device_id is None:
            return True
        until = self._blocked_until.get(device_id)
        if until is None:
            return True
        if self._clock.monotonic() >= until:
            del self._blocked_until[device_id]
            return True
        return False

    def failed(self, device_id: str | None) -> None:
        """Count one checked failure, for `device_id` when the token claimed a valid one."""
        now = self._clock.monotonic()
        self._forget(now)
        self._recent.append(now)
        if device_id is not None:
            failures = self._by_id.setdefault(device_id, deque())
            failures.append(now)
            if len(failures) >= self._id_failures:
                self._blocked_until[device_id] = now + self._id_block
                failures.clear()
        if len(self._recent) > self._global_failures and not self.paused():
            self._paused_until = now + self._global_pause
            logger.warning("device_auth_paused failures=%d", len(self._recent))
            # The pause answers these failures; the next pause needs a fresh spray.
            self._recent.clear()

    def tracked_ids(self) -> int:
        """How many ids have failures or a block on record; bounded by the windows."""
        return len(self._by_id.keys() | self._blocked_until.keys())

    def _forget(self, now: float) -> None:
        """Drop failures older than their window and blocks that ended."""
        while self._recent and now - self._recent[0] > self._global_window:
            self._recent.popleft()
        for device_id, failures in list(self._by_id.items()):
            while failures and now - failures[0] > self._id_window:
                failures.popleft()
            if not failures:
                del self._by_id[device_id]
        for device_id, until in list(self._blocked_until.items()):
            if now >= until:
                del self._blocked_until[device_id]
