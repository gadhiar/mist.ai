"""Capabilities' `EventSink` over the session registry: house-wide state to every session (D7).

The capability host emits its broadcast payloads here, from the loop or from a worker thread.
`alarm_fired` is the desktop's ring, so it goes to the sessions bound `desktop` only; every
other payload (`capabilities_status`, `alarm_state`, `action_proposed`, `action_resolved` and
the rest) is queued on every session. A `room` session writes only the types its device acts on
(`ROOM_SERVER_MESSAGE_TYPES`, checked in `RealtimeSession._write`), so of these a device is sent
`alarm_state` alone; its own `alarm_fired` comes from its ring sink (`backend/devices/`).

The sink also answers D8's `alarm_ringing`. Capabilities has no synchronous ringing query, so
the sink keeps the `ringing` list of the last `alarm_state` it relayed: an alarm rings while that
list is not empty.

The registry is attached after the sink is built, because the capability host is built and
started before the runtime and its registry exist, so an alarm due at boot rings without waiting
for the speech engines. A payload emitted with no registry attached reaches no session, as no
session exists yet; its `alarm_state` still sets `ringing`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import Mapping
from typing import Literal, Protocol

logger = logging.getLogger(__name__)

DESKTOP_ONLY_TYPES = frozenset({"alarm_fired"})


class CapabilityRoutes(Protocol):
    """The registry's side of the sink: `SessionRegistry.broadcast`."""

    def broadcast(
        self, message: object, *, endpoint: Literal["desktop", "room"] | None = None
    ) -> None:
        """Queue a message on every session's `bulk` lane, or one endpoint's sessions'."""
        ...


class CapabilitySink:
    """Routes the capability host's payloads to the sessions; `emit` is safe from any thread.

    Args:
        registry: Where payloads go; None until `attach`.
    """

    def __init__(self, registry: CapabilityRoutes | None = None) -> None:
        self._registry = registry
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: int | None = None
        self._ringing = False

    def attach(self, registry: CapabilityRoutes, loop: asyncio.AbstractEventLoop) -> None:
        """Route through `registry`, whose sessions live on `loop`."""
        self._registry = registry
        self._loop = loop
        self._loop_thread = threading.get_ident()

    def ringing(self) -> bool:
        """Whether the last `alarm_state` relayed had a ringing occurrence; False before any."""
        return self._ringing

    def emit(self, payload: dict[str, object]) -> None:
        """Serialise `payload` now, note its ringing list, and route it on the loop's thread."""
        if payload.get("type") == "alarm_state":
            ringing = payload.get("ringing")
            self._ringing = isinstance(ringing, list) and len(ringing) > 0
        message = json.dumps(payload)
        loop = self._loop
        if loop is None or threading.get_ident() == self._loop_thread:
            self._route(payload, message)
            return
        try:
            loop.call_soon_threadsafe(self._route, payload, message)
        except RuntimeError:
            logger.warning(
                "capability sink: the loop is closed; a %s was not sent", payload.get("type")
            )

    def _route(self, payload: Mapping[str, object], message: str) -> None:
        registry = self._registry
        if registry is None:
            return
        if payload.get("type") in DESKTOP_ONLY_TYPES:
            registry.broadcast(message, endpoint="desktop")
        else:
            registry.broadcast(message)
