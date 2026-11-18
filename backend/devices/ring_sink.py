"""A room device's ring endpoint: `alarm_fired` to that device's own sessions.

Each registered, unrevoked device is one `RingSink` named as the device is, so an alarm's
`ring_on` can name it. When none of the endpoints a ring names is connected, `RingRouter` rings
every present endpoint instead; a ring that names a connected endpoint beside an absent device
goes to the connected one only. The sink holds no session: `send` and `present` reach the session
registry when they are called, because the capability host and its sinks are built before the
registry exists (`registry_ring_sinks`).

Only `alarm_fired` is the sink's own. `alarm_state` already reaches every session, a device's
included, through the capability host's event sink, so `stopped` and `state_changed` send
nothing: a stop from the device (`alarm_action`) or a spoken one ends the occurrence in the
scheduler, and the `alarm_state` that follows tells every endpoint. The sinks are fixed when the
host is built, so a device registered later rings once the backend restarts.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from typing import Protocol

from backend.capabilities.alarms.notifier import alarm_fired_payload
from backend.capabilities.alarms.records import Occurrence

from .registry import DeviceRecord

Send = Callable[[str, dict[str, object]], None]
"""Queue a payload on the sessions bound to an endpoint."""

Present = Callable[[str], bool]
"""Whether a session bound to an endpoint is connected."""


class EndpointRoutes(Protocol):
    """The session registry's side the sinks use: `SessionRegistry`."""

    def endpoint_present(self, endpoint: str) -> bool:
        """Whether a session bound to `endpoint` is connected."""
        ...

    def broadcast(self, message: object, *, endpoint: str | None = None) -> None:
        """Queue a message on every session's `bulk` lane, or one endpoint's sessions'."""
        ...


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


def registry_ring_sinks(
    devices: Iterable[DeviceRecord], registry: Callable[[], EndpointRoutes | None]
) -> tuple[DeviceRingSink, ...]:
    """Return one sink per device that is not revoked, over the session registry.

    `registry` is called at each `present` and `ring`, so sinks built before the session
    registry exists reach it once it does; until then no device is present and nothing is sent.
    A payload goes out as JSON text, as the capability host's event sink sends its own.
    """

    def present(endpoint: str) -> bool:
        routes = registry()
        return routes is not None and routes.endpoint_present(endpoint)

    def send(endpoint: str, payload: dict[str, object]) -> None:
        routes = registry()
        if routes is not None:
            routes.broadcast(json.dumps(payload), endpoint=endpoint)

    return device_ring_sinks(devices, send, present)
