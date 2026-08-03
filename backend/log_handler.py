"""WebSocket log handler for real-time log streaming.

Captures Python log records, formats them as structured dicts, and queues
them for broadcast over the existing WebSocket message queue.  Includes
per-logger gating, a token-bucket rate limiter, and a re-entrancy guard.
"""

import asyncio
import contextlib
import json
import logging
import threading
import time
from typing import Any

# Package-qualified: `request_context` and `backend.request_context` are two
# distinct module objects (backend/ is a namespace package that is also on
# sys.path), so importers must agree on one name or the ContextVars diverge.
from backend.request_context import current_request_id

_thread_local = threading.local()

# Default gating levels for noisy third-party loggers.
_SUPPRESSED_LOGGERS: dict[str, int] = {
    name: logging.WARNING
    for name in (
        "httpx",
        "urllib3",
        "uvicorn.access",
        "uvicorn.error",
        "httpcore",
        "hpack",
        "diffusers",
        "transformers",
        "sentence_transformers",
        "asyncio",
        "neo4j",
        "torch",
        "PIL",
        "backend.log_handler",
    )
}


class WebSocketLogHandler(logging.Handler):
    """Logging handler that queues structured log records for WebSocket broadcast.

    Args:
        event_loop: The running ``asyncio`` event loop used for thread-safe
            queue access via ``asyncio.run_coroutine_threadsafe``.
        message_queue: The ``asyncio.Queue`` shared with
            ``broadcast_messages()`` in ``server.py``.
    """

    def __init__(
        self,
        event_loop: asyncio.AbstractEventLoop,
        message_queue: asyncio.Queue,
    ) -> None:
        super().__init__()
        self._loop = event_loop
        self._queue = message_queue

        # -- Per-logger gating map -------------------------------------------
        # Key "" is the default level for loggers without an explicit entry.
        self._gating_map: dict[str, int] = {"": logging.INFO}
        self._gating_map.update(_SUPPRESSED_LOGGERS)

        # -- Token-bucket rate limiter ---------------------------------------
        self._bucket_capacity: int = 100
        self._bucket_tokens: float = float(self._bucket_capacity)
        self._refill_rate: float = 100.0  # tokens per second
        self._last_refill: float = time.monotonic()
        self._dropped_count: int = 0
        self._drop_start: float = 0.0
        self._bucket_lock = threading.Lock()

        # -- ADR-017 subscribe_logs streaming gate ---------------------------
        # WebSocket log streaming is opt-in (off by default to avoid flooding
        # clients with backend logs). FE sends `subscribe_logs {enabled,
        # levels}` to toggle. `_level_filter=None` means all levels allowed;
        # otherwise emit only records whose levelname is in the set.
        self._streaming_enabled: bool = False
        self._level_filter: frozenset[str] | None = None

    # -- Public API ----------------------------------------------------------

    def set_logger_level(self, logger_name: str, level: str) -> None:
        """Update the gating level for a specific logger.

        Args:
            logger_name: Dotted logger name, or ``"root"`` to change the
                default level for all loggers without an explicit override.
            level: One of ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``.
        """
        level_int = getattr(logging, level.upper(), None)
        if level_int is None:
            return
        key = "" if logger_name == "root" else logger_name
        self._gating_map[key] = level_int

    def set_streaming(self, enabled: bool, levels: list[str] | None) -> None:
        """Configure WebSocket log streaming per ADR-017 subscribe_logs.

        Streaming is off by default. When enabled with ``levels=None`` all
        log levels are emitted (subject to per-logger gating and the rate
        limiter). When ``levels`` is a list, only records whose levelname is
        in the list pass through (case-insensitive, normalized to uppercase).

        Args:
            enabled: True to start streaming, False to stop.
            levels: List of level names (DEBUG/INFO/WARNING/ERROR/CRITICAL),
                or None to allow all levels.
        """
        self._streaming_enabled = bool(enabled)
        if levels is None:
            self._level_filter = None
        else:
            valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
            normalized = frozenset(level.upper() for level in levels if level.upper() in valid)
            self._level_filter = normalized if normalized else None

    # -- Handler implementation ----------------------------------------------

    def emit(self, record: logging.LogRecord) -> None:
        """Format and enqueue a log record for WebSocket broadcast.

        Applies a re-entrancy guard, per-logger gating, and rate limiting
        before queuing.
        """
        # Re-entrancy guard: prevent infinite recursion if broadcast path logs.
        if getattr(_thread_local, "_in_emit", False):
            return
        _thread_local._in_emit = True
        try:
            self._guarded_emit(record)
        finally:
            _thread_local._in_emit = False

    # -- Internal helpers ----------------------------------------------------

    def _guarded_emit(self, record: logging.LogRecord) -> None:
        """Core emit logic, called inside the re-entrancy guard."""
        # ADR-017 subscribe_logs gate: WebSocket log streaming is opt-in.
        # When disabled, no log records reach the WebSocket. The persistent
        # JSONL file log is unaffected (separate FileHandler attached at
        # server.py startup).
        if not self._streaming_enabled:
            return
        # ADR-017 level filter (None = all levels allowed).
        if self._level_filter is not None and record.levelname not in self._level_filter:
            return

        # Per-logger gating check (e.g., httpx default WARNING to avoid spam).
        floor = self._gating_map.get(record.name, self._gating_map[""])
        if record.levelno < floor:
            return

        # Rate-limit check.
        if not self._try_consume_token():
            return

        self._enqueue(record)

    def _try_consume_token(self) -> bool:
        """Attempt to consume a token from the bucket.

        Returns:
            ``True`` if a token was consumed (message may proceed),
            ``False`` if the bucket is exhausted (message is dropped).
        """
        drop_summary: tuple[int, float] | None = None
        with self._bucket_lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._bucket_tokens = min(
                float(self._bucket_capacity),
                self._bucket_tokens + elapsed * self._refill_rate,
            )
            self._last_refill = now

            if self._bucket_tokens >= 1.0:
                # Bucket has tokens -- consume one.
                if self._dropped_count > 0:
                    drop_summary = (
                        self._dropped_count,
                        now - self._drop_start,
                    )
                    self._dropped_count = 0
                    self._drop_start = 0.0

                self._bucket_tokens -= 1.0
            else:
                # Exhausted -- track the drop.
                if self._dropped_count == 0:
                    self._drop_start = now
                self._dropped_count += 1
                return False

        if drop_summary is not None:
            # Emit a recovery summary (exempt from rate limit).
            count, duration = drop_summary
            self._enqueue_summary(count, duration)

        return True

    def _enqueue(self, record: logging.LogRecord) -> None:
        """Serialize a log record to JSON and push onto the async message queue.

        The broadcaster (server.py:73-93) only handles bytes/str on the wire;
        emitting a dict here previously caused silent drops (MIS-106 dead code
        on the wire). Timestamp is unix milliseconds per ADR-017.
        """
        request_id = current_request_id.get()
        record_dict: dict[str, Any] = {
            "type": "log",
            "timestamp": int(record.created * 1000),
            "level": record.levelname,
            "levelno": record.levelno,
            "logger": record.name,
            "request_id": request_id,
            "message": self.format(record),
        }
        with contextlib.suppress(RuntimeError):
            asyncio.run_coroutine_threadsafe(self._queue.put(json.dumps(record_dict)), self._loop)

    def _enqueue_summary(self, count: int, duration: float) -> None:
        """Enqueue a rate-limit recovery summary as JSON (exempt from limiter)."""
        record_dict: dict[str, Any] = {
            "type": "log",
            "timestamp": int(time.time() * 1000),
            "level": "WARNING",
            "levelno": logging.WARNING,
            "logger": "backend.log_handler",
            "request_id": None,
            "message": (
                f"Dropped {count} log messages in the last " f"{duration:.1f}s (rate limit)"
            ),
        }
        with contextlib.suppress(RuntimeError):
            asyncio.run_coroutine_threadsafe(self._queue.put(json.dumps(record_dict)), self._loop)
