"""Per-turn structured JSONL debug logger.

Emits one or two JSONL records per turn (keyed by event_id) to a file path
configured via the `MIST_DEBUG_JSONL` environment variable. When the env var
is unset, `DebugJSONLLogger.begin_turn()` returns a no-op record with zero
overhead.

Record shape:

    {"phase": "turn",
     "ts_iso": "...",
     "event_id": "...",
     "session_id": "...",
     "user_id": "...",
     "utterance": "...",
     "retrieval": {...} | null,
     "llm_passes": [{...}, {...}],
     "total_turn_ms": 1234.5}

    {"phase": "extraction",
     "ts_iso": "...",
     "event_id": "...",
     "parse_ok": true,
     "extraction": {entity_count, relationship_count, avg_confidence, duration_ms},
     "graph_writes": {entities_created, entities_updated, relationships_created, ...} | null}

The two records are emitted separately because extraction runs fire-and-forget
(`asyncio.create_task`) after the turn completes and the LLM response has been
returned to the caller. Downstream consumers join on `event_id`.

Spec: ~/.claude/plans/nimble-forage-cinder.md Part 3 (structured JSONL debug log).
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.knowledge.models import RetrievalResult
    from backend.llm.models import LLMResponse


ENV_VAR = "MIST_DEBUG_JSONL"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


class DebugJSONLLogger:
    """Per-turn structured JSONL sink.

    Construct with a path to enable, or `None` to produce no-op records.
    `DebugJSONLLogger.from_env()` reads `MIST_DEBUG_JSONL` for convenience.
    """

    def __init__(self, output_path: Path | str | None):
        self._path: Path | None = Path(output_path) if output_path else None
        self._lock = threading.Lock()
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls, env_var: str = ENV_VAR) -> DebugJSONLLogger:
        """Construct from env var. Returns a disabled logger if var is unset."""
        value = os.environ.get(env_var)
        return cls(value if value else None)

    @property
    def enabled(self) -> bool:
        """True when a valid output path was configured."""
        return self._path is not None

    @property
    def path(self) -> Path | None:
        """Absolute path to the JSONL file, or None when disabled."""
        return self._path

    def begin_turn(
        self,
        *,
        event_id: str,
        session_id: str,
        user_id: str,
        utterance: str,
    ) -> TurnRecord:
        """Start a new turn record. Cheap no-op when disabled."""
        if not self.enabled:
            return _NoOpTurnRecord()
        return _LiveTurnRecord(
            logger=self,
            event_id=event_id,
            session_id=session_id,
            user_id=user_id,
            utterance=utterance,
        )

    def _emit(self, record: dict[str, Any]) -> None:
        """Serialize and append a single JSONL line. Thread-safe."""
        if not self.enabled or self._path is None:
            return
        line = json.dumps(record, default=str, ensure_ascii=False)
        with self._lock, self._path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# TurnRecord API (no-op + live implementations share the method surface)
# ---------------------------------------------------------------------------


class TurnRecord:
    """Interface for recording turn phases. See _LiveTurnRecord for behavior."""

    def record_retrieval(self, result: RetrievalResult) -> None:
        """Capture retrieval results for the turn. No-op when disabled."""
        raise NotImplementedError

    def record_llm_response(
        self,
        response: LLMResponse,
        pass_num: int,
        timing_ms: float | None = None,
    ) -> None:
        """Capture an LLM response pass (1 or 2) with optional latency."""
        raise NotImplementedError

    def flush_turn(self) -> None:
        """Emit the accumulated turn record to the JSONL sink."""
        raise NotImplementedError

    def record_extraction(
        self,
        result: Any,
        duration_ms: float,
        parse_ok: bool = True,
    ) -> None:
        """Capture the extraction phase result with duration and parse outcome."""
        raise NotImplementedError

    def flush_extraction(self) -> None:
        """Emit the accumulated extraction record to the JSONL sink."""
        raise NotImplementedError


class _NoOpTurnRecord(TurnRecord):
    """Zero-overhead fallback when logging is disabled."""

    def record_retrieval(self, result: RetrievalResult) -> None:
        return

    def record_llm_response(
        self,
        response: LLMResponse,
        pass_num: int,
        timing_ms: float | None = None,
    ) -> None:
        return

    def flush_turn(self) -> None:
        return

    def record_extraction(
        self,
        result: Any,
        duration_ms: float,
        parse_ok: bool = True,
    ) -> None:
        return

    def flush_extraction(self) -> None:
        return


class _LiveTurnRecord(TurnRecord):
    """Active turn record. Accumulates data and flushes JSONL lines on demand."""

    def __init__(
        self,
        *,
        logger: DebugJSONLLogger,
        event_id: str,
        session_id: str,
        user_id: str,
        utterance: str,
    ):
        self._logger = logger
        self._event_id = event_id
        self._session_id = session_id
        self._user_id = user_id
        self._utterance = utterance
        self._ts_start = time.time()
        self._retrieval: dict[str, Any] | None = None
        self._llm_passes: list[dict[str, Any]] = []
        self._extraction: dict[str, Any] | None = None
        self._graph_writes: dict[str, Any] | None = None
        self._turn_flushed = False
        self._extraction_flushed = False

    def record_retrieval(self, result: RetrievalResult) -> None:
        top = []
        for fact in result.facts[:5] if result.facts else []:
            top.append(
                {
                    "subject": fact.subject,
                    "predicate": fact.predicate,
                    "object": fact.object,
                    "similarity": fact.similarity_score,
                    "graph_distance": fact.graph_distance,
                }
            )
        self._retrieval = {
            "query": result.query,
            "entities_found": result.entities_found,
            "total_facts": result.total_facts,
            "top_facts": top,
            "retrieval_time_ms": result.retrieval_time_ms,
            "vector_search_time_ms": result.vector_search_time_ms,
            "graph_traversal_time_ms": result.graph_traversal_time_ms,
        }

    def record_llm_response(
        self,
        response: LLMResponse,
        pass_num: int,
        timing_ms: float | None = None,
    ) -> None:
        tool_calls = []
        for tc in response.tool_calls or []:
            tool_calls.append(
                {
                    "name": getattr(tc, "name", None),
                    "id": getattr(tc, "id", None),
                    "arg_keys": list(tc.arguments.keys()) if getattr(tc, "arguments", None) else [],
                }
            )
        usage = None
        if getattr(response, "usage", None):
            usage = {
                "input_tokens": getattr(response.usage, "input_tokens", None),
                "output_tokens": getattr(response.usage, "output_tokens", None),
            }
        self._llm_passes.append(
            {
                "pass": pass_num,
                "content_len": len(response.content) if response.content else 0,
                "tool_calls": tool_calls,
                "partial": bool(getattr(response, "partial", False)),
                "usage": usage,
                "timing_ms": timing_ms,
            }
        )

    def flush_turn(self) -> None:
        """Emit the `phase: "turn"` JSONL line. Idempotent."""
        if self._turn_flushed:
            return
        record = {
            "phase": "turn",
            "ts_iso": datetime.fromtimestamp(self._ts_start, tz=UTC).isoformat(),
            "event_id": self._event_id,
            "session_id": self._session_id,
            "user_id": self._user_id,
            "utterance": self._utterance,
            "retrieval": self._retrieval,
            "llm_passes": self._llm_passes,
            "total_turn_ms": (time.time() - self._ts_start) * 1000,
        }
        self._logger._emit(record)
        self._turn_flushed = True

    def record_extraction(
        self,
        result: Any,
        duration_ms: float,
        parse_ok: bool = True,
    ) -> None:
        """Record extraction result. Polymorphic over ValidationResult | CurationResult.

        Pass `result=None` with `parse_ok=False` to record a failed extraction.
        """
        if result is None:
            self._extraction = {
                "parse_ok": parse_ok,
                "entity_count": 0,
                "relationship_count": 0,
                "avg_confidence": None,
                "duration_ms": duration_ms,
            }
            return

        # CurationResult wraps a ValidationResult and adds a write_result +
        # curation_time_ms. Prefer the inner ValidationResult when present so
        # entity/relationship counts reflect extracted content, not curation side.
        inner = getattr(result, "validation_result", None)
        if inner is not None:
            entities = getattr(inner, "entities", None)
            relationships = getattr(inner, "relationships", None)
        else:
            entities = getattr(result, "entities", None)
            relationships = getattr(result, "relationships", None)

        write_result = getattr(result, "write_result", None)

        confidences: list[float] = []
        if entities:
            confidences.extend(float(e.confidence) for e in entities if hasattr(e, "confidence"))
        if relationships:
            confidences.extend(
                float(r.confidence) for r in relationships if hasattr(r, "confidence")
            )
        avg_conf = sum(confidences) / len(confidences) if confidences else None

        self._extraction = {
            "parse_ok": parse_ok,
            "entity_count": len(entities) if entities else 0,
            "relationship_count": len(relationships) if relationships else 0,
            "avg_confidence": avg_conf,
            "duration_ms": duration_ms,
        }

        if write_result is not None:
            self._graph_writes = {
                "entities_created": getattr(write_result, "entities_created", 0),
                "entities_updated": getattr(write_result, "entities_updated", 0),
                "relationships_created": getattr(write_result, "relationships_created", 0),
                "relationships_updated": getattr(write_result, "relationships_updated", 0),
                "relationships_superseded": getattr(write_result, "relationships_superseded", 0),
            }

    def flush_extraction(self) -> None:
        """Emit the `phase: "extraction"` JSONL line. Idempotent."""
        if self._extraction_flushed:
            return
        record = {
            "phase": "extraction",
            "ts_iso": _now_iso(),
            "event_id": self._event_id,
            "extraction": self._extraction,
            "graph_writes": self._graph_writes,
        }
        self._logger._emit(record)
        self._extraction_flushed = True
