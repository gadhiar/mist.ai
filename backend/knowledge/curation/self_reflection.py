"""Batch internal derivation over recent conversation events.

Complements the real-time per-turn derivation from the extraction pipeline
by re-processing recent events on a scheduled cadence. This catches signals
that the real-time path may have missed or that benefit from accumulated
context.
"""

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from backend.interfaces import EventStoreProvider
from backend.knowledge.extraction.internal_derivation import InternalKnowledgeDeriver
from backend.knowledge.extraction.signal_detector import SignalDetector
from backend.knowledge.storage.graph_executor import GraphExecutor

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ReflectionResult:
    """Result of a batch self-reflection run."""

    events_processed: int
    operations_applied: int
    duration_ms: float


class SelfReflectionJob:
    """Run batch internal derivation over recent conversation events."""

    def __init__(
        self,
        executor: GraphExecutor,
        internal_deriver: InternalKnowledgeDeriver,
        signal_detector: SignalDetector,
        event_store: EventStoreProvider | None = None,
    ) -> None:
        self._executor = executor
        self._deriver = internal_deriver
        self._signal_detector = signal_detector
        self._event_store = event_store

    async def run(self, lookback_hours: int = 24) -> ReflectionResult:
        """Run batch internal derivation over recent conversation events.

        Args:
            lookback_hours: How far back to look for events.

        Returns:
            ReflectionResult with counts.
        """
        if self._event_store is None:
            return ReflectionResult(events_processed=0, operations_applied=0, duration_ms=0.0)

        start = time.perf_counter()

        since = datetime.now(UTC) - timedelta(hours=lookback_hours)
        turns = self._event_store.get_turns_since(since)

        total_ops = 0
        events_processed = 0

        for turn in turns:
            utterance = turn.get("user_utterance", "")
            response = turn.get("system_response", "")
            session_id = turn.get("session_id", "")
            event_id = turn.get("event_id", "")

            if not utterance:
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
