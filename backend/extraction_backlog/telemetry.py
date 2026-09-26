"""Process-wide count of turns the event log never recorded.

`ModelManager.generate_llm_response` has two branches. With knowledge
integration enabled it streams through `KnowledgeIntegration`, whose
`ConversationHandler` owns the event store and records every turn. With it off
or unavailable it calls the chat model directly, and that turn is answered and
then lost: it is written to neither the event log nor the extraction backlog.

Recording it there is not possible, and not merely unimplemented:

- The event store is constructed in exactly one place on the live path,
  `ConversationHandler.__init__` (`grep -n "EventStore(" backend/` -> the
  `conversation_handler.py` constructor and the admin CLI in this package).
- That handler is built only by `KnowledgeIntegration.__init__`
  (`grep -n "build_conversation_handler(" backend/chat/knowledge_integration.py`),
  and when that construction raises, the same `__init__` leaves
  `conversation_handler = None` and `enabled = False` (its `except Exception`
  branch). The fallback branch runs exactly when `enabled` is False or no
  `KnowledgeIntegration` exists, so there is no event store to write to.
- The fallback branch's only I/O is `self._llm_provider.generate_sync`
  (`grep -n "generate_sync" backend/voice_models/model_manager.py`).

So the loss is made visible instead: one structured WARNING per turn, and a
counter surfaced as `ExtractionStatus.unrecorded_turns` on the WebSocket,
`GET /extraction/status` and `/health`.

The counter is per process and starts at zero on every start; it is not
persisted, because the turns it counts were not persisted either.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_unrecorded_turns = 0


def record_unrecorded_turn(request_id: str | None) -> int:
    """Count one turn answered without being logged, and warn about it.

    Thread-safe: the voice path runs `generate_llm_response` on a worker
    thread (`backend.request_context.spawn_with_context`).

    Args:
        request_id: The turn's `current_request_id`, or None when unset.

    Returns:
        The process-wide count after this turn.
    """
    global _unrecorded_turns
    with _lock:
        _unrecorded_turns += 1
        total = _unrecorded_turns
    logger.warning(
        "unrecorded_turn request_id=%s unrecorded_total=%d: knowledge integration is off "
        "or unavailable, so this turn is answered but written to neither the event log "
        "nor the extraction backlog",
        request_id or "-",
        total,
    )
    return total


def unrecorded_turns() -> int:
    """Turns answered without being logged since this process started."""
    with _lock:
        return _unrecorded_turns


def reset_unrecorded_turns() -> None:
    """Zero the counter. For tests; production never resets it."""
    global _unrecorded_turns
    with _lock:
        _unrecorded_turns = 0
