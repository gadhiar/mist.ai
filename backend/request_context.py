"""Request context propagation via contextvars.

Provides a per-request ID and a per-connection session ID that flow through
async and threaded call chains, enabling log records to be grouped by
conversation turn and each turn to be attributed to the WebSocket connection
that originated it.
"""

import itertools
import threading
from collections.abc import Callable
from contextvars import ContextVar, copy_context
from typing import Any

current_request_id: ContextVar[str | None] = ContextVar("current_request_id", default=None)

# Per-connection session ID. A ContextVar rather than instance state, so that
# no object shared by every connection holds one connection's session.
#
# Default is None, not a placeholder string. A silently-defaulted session ID
# is what previously routed every connection into a single session named
# "default"; consumers must treat None as a wiring error rather than
# substitute a fallback.
current_session_id: ContextVar[str | None] = ContextVar("current_session_id", default=None)

_counter = itertools.count(1)
_counter_lock = threading.Lock()


def new_request_id() -> str:
    """Generate a new monotonic request ID and set it in the current context.

    Returns:
        The new request ID string (e.g. ``turn-1``, ``turn-42``).
    """
    with _counter_lock:
        seq = next(_counter)
    rid = f"turn-{seq}"
    current_request_id.set(rid)
    return rid


def spawn_with_context(
    target: Callable[..., Any], *args: Any, **thread_kwargs: Any
) -> threading.Thread:
    """Spawn a daemon thread that inherits the current `contextvars` context.

    This ensures ``current_request_id`` (and any other context vars) propagate
    into the new thread, which plain ``threading.Thread`` does not do.

    Args:
        target: Callable to run in the new thread.
        *args: Positional arguments forwarded to *target*.
        **thread_kwargs: Extra keyword arguments passed to ``threading.Thread``
            (e.g. ``name``).  ``daemon`` defaults to ``True`` if not provided.

    Returns:
        The started ``threading.Thread`` instance.
    """
    ctx = copy_context()
    thread_kwargs.setdefault("daemon", True)
    t = threading.Thread(target=ctx.run, args=(target, *args), **thread_kwargs)
    t.start()
    return t
