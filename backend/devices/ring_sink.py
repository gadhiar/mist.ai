"""A room device's ring endpoint: `alarm_fired` to that device's own sessions.

Each registered, unrevoked device is one `RingSink` named as the device is, so an alarm's
`ring_on` can name it, and `RingRouter` falls back to every present endpoint while it is not
connected. The sink holds no session: `send` and `present` reach the session registry when they
are called, because the capability host and its sinks are built before the registry exists.

Only `alarm_fired` is the sink's own. `alarm_state` already reaches every session, a device's
included, through the capability host's event sink, so `stopped` and `state_changed` send
nothing: a stop from the device (`alarm_action`) or a spoken one ends the occurrence in the
scheduler, and the `alarm_state` that follows tells every endpoint. The sinks are fixed when the
host is built, so a device registered later rings once the backend restarts.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

from backend.capabilities.alarms.notifier import alarm_fired_payload
from backend.capabilities.alarms.records import Occurrence

from .registry import DeviceRecord

Send = Callable[[str, dict[str, object]], None]
"""Queue a payload on the sessions bound to an endpoint."""

Present = Callable[[str], bool]
"""Whether a session bound to an endpoint is connected."""


class DeviceRingSink:
    """The ring endpoint of one room device, named as the device is."""

    __slots__ = ("name", "_send", "_present")

    def __init__(self, name: str, send: Send, present: Present) -> None:
        """Keep the device's name and the callables that reach its sessions."""
        self.name = name
        self._send = send
        self._present = present

    def present(self) -> bool:
        """Return whether the device has a live session now."""
        return self._present(self.name)

    async def ring(self, occurrence: Occurrence, ring_timeout_s: float, *, fallback: bool) -> bool:
        """Send `alarm_fired` to this device's sessions only; return whether one was connected."""
        if not self._present(self.name):
            return False
        self._send(self.name, alarm_fired_payload(occurrence, ring_timeout_s, fallback=fallback))
        return True

    async def stopped(self, occurrence_id: str) -> None:
        """Send nothing: the `alarm_state` that follows every stop reaches the device."""

    async def state_changed(self, payload: Mapping[str, object]) -> None:
        """Send nothing: the capability host's event sink sends `alarm_state` to every session."""


def device_ring_sinks(
    devices: Iterable[DeviceRecord], send: Send, present: Present
) -> tuple[DeviceRingSink, ...]:
    """Return one sink per device that is not revoked, in registry order."""
    return tuple(
        DeviceRingSink(device.name, send, present) for device in devices if not device.revoked
    )
