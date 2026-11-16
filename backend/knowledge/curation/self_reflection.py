"""Batch internal derivation over recent conversation events.

Complements the real-time per-turn derivation from the extraction pipeline
by re-processing recent events on a scheduled cadence. This catches signals
that the real-time path may have missed or that benefit from accumulated
context.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from backend.extraction_backlog.eligibility import policy_skip_reason
from backend.interfaces import EventStoreProvider
from backend.knowledge.extraction.internal_derivation import InternalKnowledgeDeriver
from backend.knowledge.extraction.signal_detector import SignalDetector
from backend.knowledge.storage.graph_executor import GraphExecutor

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK_HOURS = 24


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class ReflectionResult:
    """Result of a batch self-reflection run."""

    events_processed: int
    operations_applied: int
    duration_ms: float


class SelfReflectionJob:
    """Run batch internal derivation over recent conversation events.

    The job remembers, in memory, the end of its last completed window, and
    the next scheduled run starts there. The scheduler defers this job while
    gaming mode holds the GPU, so a fixed 24-hour lookback would lose the
    deferred hours; starting at the previous window's end covers them. The
    first run in a process looks back `DEFAULT_LOOKBACK_HOURS`. Each window
    is half-open, `[since, now)`, so consecutive windows share no turn.
    """

    def __init__(
        self,
        executor: GraphExecutor,
        internal_deriver: InternalKnowledgeDeriver,
        signal_detector: SignalDetector,
        event_store: EventStoreProvider | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._executor = executor
        self._deriver = internal_deriver
        self._signal_detector = signal_detector
        self._event_store = event_store
        self._clock = clock
        self._last_window_end: datetime | None = None

    async def run(self, lookback_hours: int | None = None) -> ReflectionResult:
        """Run batch internal derivation over recent conversation events.

        Args:
            lookback_hours: How far back to look for events. When given, the
                window is exactly that and does not move the remembered window
                end, so a short manual run cannot make the next scheduled run
                skip hours. When None, the window starts at the end of the last
                completed window, or `DEFAULT_LOOKBACK_HOURS` back on the first
                run.

        Returns:
            ReflectionResult with counts.
        """
        if self._event_store is None:
            return ReflectionResult(events_processed=0, operations_applied=0, duration_ms=0.0)

        start = time.perf_counter()

        now = self._clock()
        if lookback_hours is not None:
            since = now - timedelta(hours=lookback_hours)
        elif self._last_window_end is not None:
            since = self._last_window_end
        else:
            since = now - timedelta(hours=DEFAULT_LOOKBACK_HOURS)
        # The window is `[since, now)`. `get_turns_since` has no upper bound, so
        # a turn stamped at or after `now` and committed before the query would
        # also fall in the next window, which starts at `now` inclusive, and be
        # derived twice. The bound is compared as the same ISO-8601 text the
        # store compares `since` with, so the two windows meet exactly.
        window_end = now.isoformat()
        turns = [
            turn
            for turn in self._event_store.get_turns_since(since)
            if turn["timestamp"] < window_end
        ]

        total_ops = 0
        events_processed = 0

        for turn in turns:
            utterance = turn.get("user_utterance", "")
            response = turn.get("system_response", "")
            session_id = turn.get("session_id", "")
            event_id = turn.get("event_id", "")

            if not utterance:
                continue
            # A turn a capability answered alone is a command and a template;
            # there is nothing in it about the user or MIST to reflect on.
            if policy_skip_reason(turn.get("answered_by")) is not None:
                continue

            signals = self._signal_detector.detect(
                utterance,
                assistant_response=response,
            )

            result = await self._deriver.derive(
                utterance=utterance,
                assistant_response=response,
                signals=signals,
                session_id=session_id,
                event_id=event_id,
            )

            total_ops += len(result.operations)
            events_processed += 1

        if lookback_hours is None:
            self._last_window_end = now

        elapsed = (time.perf_counter() - start) * 1000

        logger.info(
            "Self-reflection: processed %d events, %d operations in %.1fms",
            events_processed,
            total_ops,
            elapsed,
        )

        return ReflectionResult(
            events_processed=events_processed,
            operations_applied=total_ops,
            duration_ms=elapsed,
        )
