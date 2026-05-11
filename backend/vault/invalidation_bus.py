"""Lightweight in-process pub/sub for vault-rebuild completion events.

Producers: GraphRegenerator (publishes after rebuild_from_path completes)
Consumers: ConversationHandler (subscribes to evict mist_context cache)

Listener exceptions are logged and isolated; one bad listener does not
prevent others from being called. Best-effort delivery by design.
"""

import logging
from collections.abc import Awaitable, Callable

from backend.knowledge.curation.graph_regenerator import RebuildResult

logger = logging.getLogger(__name__)

Listener = Callable[[RebuildResult], Awaitable[None]]


class InvalidationBus:
    """In-process pub/sub bus for vault-rebuild completion events.

    Producers call `publish` after a rebuild completes. Consumers register
    async listeners via `subscribe`. Listener exceptions are caught, logged,
    and isolated so one failing listener does not block the others.
    """

    def __init__(self) -> None:
        """Initialise with an empty listener list."""
        self._listeners: list[Listener] = []

    def subscribe(self, listener: Listener) -> None:
        """Register an async listener to be called on each published event."""
        self._listeners.append(listener)

    async def publish(self, event: RebuildResult) -> None:
        """Notify all registered listeners with the given rebuild event.

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
