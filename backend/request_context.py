"""Request context propagation via contextvars.

Provides a per-request ID, a per-connection session ID, and per-turn result
side-channels that flow through async and threaded call chains, enabling log
records to be grouped by conversation turn and each turn to be attributed to
the WebSocket connection that originated it.
"""

import itertools
import threading
from collections.abc import Callable
from contextvars import ContextVar, copy_context
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.chat.stream_events import Complete

current_request_id: ContextVar[str | None] = ContextVar("current_request_id", default=None)

# Per-connection session ID, minted at the WebSocket handshake and read by the
# conversation pipeline. This is a ContextVar rather than instance state
# because `voice_processor` is a module-level singleton shared by every
# connection in `active_connections`: storing the session on the instance lets
# one connection overwrite another's. Each turn's call chain carries its own
# value here instead.
#
# Default is None, not a placeholder string. A silently-defaulted session ID
# is what previously routed every connection into a single session named
# "default"; consumers must treat None as a wiring error rather than
# substitute a fallback.
current_session_id: ContextVar[str | None] = ContextVar("current_session_id", default=None)

# Per-TURN result side-channel. `generate_response_streaming` yields only
# strings and dicts, but ADR-017's `stream_complete` needs `duration_ms` /
# `tool_calls_used`, and a bridge timeout needs a discriminated error rather
# than error text yielded as a fake token. Producer and consumer run on the
# same thread -- the consumer drives the producer's generator, and a generator
# body executes in its caller's context -- so a ContextVar carries the value
# back without the singleton instance attribute these replace.
#
# The producer resets both at the top of `generate_response_streaming`, before
# it yields anything, so a consumer never reads a prior turn's value. No
# consumer-side reset exists or is needed: `KnowledgeIntegration.enabled` is
# set once in __init__ and never flips, so the producer either runs on every
# turn of the process or on none of them -- there is no turn that skips it
# after another turn set a value. Executor threads ARE pooled and reused with
# no fresh context, so if `enabled` ever becomes runtime-mutable, that
# assumption breaks and the consumer must reset again. The pending-input
# respawn path (`voice_processor.py:745`, inside `spawn_with_context`) also
# inherits these vars via `copy_context` rather than starting fresh, so it is
# this producer-side reset-before-first-yield -- not anything on the respawn
# path -- that keeps a respawned turn from reading the finishing turn's values.
current_turn_complete: ContextVar["Complete | None"] = ContextVar(
    "current_turn_complete", default=None
)
current_turn_error: ContextVar[tuple[str, str] | None] = ContextVar(
    "current_turn_error", default=None
)

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
