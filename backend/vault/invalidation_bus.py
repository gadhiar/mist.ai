"""Lightweight in-process pub/sub for vault-change events.

Producers: VaultFilewatcher (publishes after a user edit is reindexed)
Consumers: ConversationHandler (subscribes to evict mist_context cache)

R1.3: the event carries a path and nothing else. It once described a graph
rebuild; under Inv-A1 a vault edit produces no graph write, so what remains is
purely a read-path cache-invalidation signal. The type lives here, not in the
knowledge-curation package, so the vault layer does not depend on the curation
layer for a type it owns.

Listener exceptions are logged and isolated; one bad listener does not
prevent others from being called. Best-effort delivery by design.
"""

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class VaultChangeEvent:
    """A user edit to a vault file, after sidecar reindexing completed.

    Attributes:
        path: Absolute path to the edited vault markdown file.
    """

    path: Path


Listener = Callable[[VaultChangeEvent], Awaitable[None]]


class InvalidationBus:
    """In-process pub/sub bus for vault-change events.

    Producers call `publish` after a vault edit is reindexed. Consumers
    register async listeners via `subscribe`. Listener exceptions are caught,
    logged, and isolated so one failing listener does not block the others.
    """

    def __init__(self) -> None:
        """Initialise with an empty listener list."""
        self._listeners: list[Listener] = []

    def subscribe(self, listener: Listener) -> None:
        """Register an async listener to be called on each published event."""
        self._listeners.append(listener)

    async def publish(self, event: VaultChangeEvent) -> None:
        """Notify all registered listeners with the given vault-change event.

        Each listener is awaited in registration order. If a listener raises,
        the exception is logged and the remaining listeners are still called.
        """
        for listener in self._listeners:
            try:
                await listener(event)
            except Exception:
                logger.exception(
                    "InvalidationBus listener failed for event path=%s",
                    event.path,
                )
