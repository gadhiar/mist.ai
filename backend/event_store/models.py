"""Event Store data models.

Dataclasses for conversation sessions and turn events. These map directly
to the SQLite tables in schema.sql. All timestamps are ISO-8601 strings
when serialized. Complex fields (context_window, retrieval_context,
tool_calls, llm_parameters) are JSON-serialized for storage.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from backend.knowledge.version_stamps import ONTOLOGY_VERSION


@dataclass
class ConversationSession:
    """Top-level conversation container.

    Maps to the conversation_sessions table. A session groups related
    turns and tracks lifecycle (start/end).
    """

    session_id: str
    started_at: datetime
    ended_at: datetime | None = None
    turn_count: int = 0
    input_modality: str = "voice"  # "voice", "text", "api"
    # R1.4 Task 3 added the `origin` column (schema.sql) and start_session's
    # write side; the read side was never wired up, so the column was
    # write-only -- readable only via raw SQL, not through this model or
    # get_session(). R1.4 Task 10 closes that: "real" | "test" | "seed".
    origin: str = "real"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for SQLite insertion.

        Returns:
            Dict with string keys matching column names.
        """
        return {
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "turn_count": self.turn_count,
            "input_modality": self.input_modality,
            "origin": self.origin,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "ConversationSession":
        """Deserialize from a SQLite row dict.

        Args:
            row: Dict from sqlite3.Row or cursor description mapping.

        Returns:
            ConversationSession instance.
        """
        return cls(
            session_id=row["session_id"],
            started_at=datetime.fromisoformat(row["started_at"]),
            ended_at=(datetime.fromisoformat(row["ended_at"]) if row.get("ended_at") else None),
            turn_count=row.get("turn_count", 0),
            input_modality=row.get("input_modality", "voice"),
            origin=row.get("origin", "real"),
        )


@dataclass
class ConversationTurnEvent:
    """Single conversation turn. Immutable after creation.

    Maps to the conversation_turn_events table. Contains the raw
    user utterance and system response, plus all context that was
    available at generation time: the LLM context window, retrieval
    context, tool calls, audio references, and model metadata.

    The event_id is assigned at append time by EventStore, not here.
    """

    session_id: str
    turn_index: int
    timestamp: datetime

    # Raw conversation data -- NEVER changes after creation
    user_utterance: str
    system_response: str

    # Context window (what the LLM actually saw)
    context_window: list[dict[str, str]] | None = None

    # RAG context (what retrieval injected)
    retrieval_context: dict[str, Any] | None = None

    # Tool usage
    tool_calls: list[dict[str, Any]] | None = None

    # Audio reference
    audio_hash: str | None = None
    audio_format: str | None = None
    audio_duration_ms: int | None = None
    audio_sample_rate: int | None = None

    # Model metadata
    stt_model: str | None = None
    tts_model: str | None = None
    llm_model: str | None = None
    llm_parameters: dict[str, Any] | None = None

    # Versioning -- derived, never restated (see backend.knowledge.version_stamps).
    ontology_version: str = ONTOLOGY_VERSION

    # Assigned by EventStore.append_turn(), not by caller
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for SQLite insertion.

        JSON-encodes complex fields. Converts datetime to ISO-8601 string.

        `timestamp` is normalised to UTC first, and that is load-bearing rather
        than tidiness. SQLite compares TEXT lexicographically, and
        `get_all_turns_for_reextraction` orders the REPLAY by this column -- so
        lexicographic order equals chronological order only while every stored
        value carries the same offset. `2026-01-01T10:00:00+05:00` (05:00Z) sorts
        after `2026-01-01T06:00:00+00:00` (06:00Z) as a string while being earlier
        as an instant, so one mixed-offset writer silently misorders a rebuild --
        and replay order decides dedup outcomes
        (`curation/deduplication.py:130-172`).

        Every caller happens to pass UTC today (`datetime.now(UTC)` on the live
        path; `load_hydration_clock` normalises and refuses naive values). That is
        a property of the CALLERS. Normalising here makes the canonical stored
        format a property of the STORE, so the ordering's correctness does not
        depend on auditing writers. MIS-138.

        Naive datetimes are left alone deliberately: `astimezone` on a naive value
        assumes system-local time, which would invent an offset. The hydration
        clock already refuses naive values at the boundary that matters.

        Returns:
            Dict with string keys matching column names.
        """
        stamp = self.timestamp
        if stamp.tzinfo is not None:
            stamp = stamp.astimezone(UTC)
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "turn_index": self.turn_index,
            "timestamp": stamp.isoformat(),
            "user_utterance": self.user_utterance,
            "system_response": self.system_response,
            "context_window": (
                json.dumps(self.context_window) if self.context_window is not None else None
            ),
            "retrieval_context": (
                json.dumps(self.retrieval_context) if self.retrieval_context is not None else None
            ),
            "tool_calls": (json.dumps(self.tool_calls) if self.tool_calls is not None else None),
            "audio_hash": self.audio_hash,
            "audio_format": self.audio_format,
            "audio_duration_ms": self.audio_duration_ms,
            "audio_sample_rate": self.audio_sample_rate,
            "stt_model": self.stt_model,
            "tts_model": self.tts_model,
            "llm_model": self.llm_model,
            "llm_parameters": (
                json.dumps(self.llm_parameters) if self.llm_parameters is not None else None
            ),
            "ontology_version": self.ontology_version,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "ConversationTurnEvent":
        """Deserialize from a SQLite row dict.

        JSON-decodes complex fields. Parses ISO-8601 timestamps.

        Args:
            row: Dict from sqlite3.Row or cursor description mapping.

        Returns:
            ConversationTurnEvent instance.
        """

        def _parse_json(value: str | None) -> Any:
            """Parse JSON string, returning None if input is None or invalid."""
            if value is None:
                return None
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return None

        return cls(
            event_id=row["event_id"],
            session_id=row["session_id"],
            turn_index=row["turn_index"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            user_utterance=row["user_utterance"],
            system_response=row["system_response"],
            context_window=_parse_json(row.get("context_window")),
            retrieval_context=_parse_json(row.get("retrieval_context")),
            tool_calls=_parse_json(row.get("tool_calls")),
            audio_hash=row.get("audio_hash"),
            audio_format=row.get("audio_format"),
            audio_duration_ms=row.get("audio_duration_ms"),
            audio_sample_rate=row.get("audio_sample_rate"),
            stt_model=row.get("stt_model"),
            tts_model=row.get("tts_model"),
            llm_model=row.get("llm_model"),
            llm_parameters=_parse_json(row.get("llm_parameters")),
            ontology_version=row.get("ontology_version", ONTOLOGY_VERSION),
        )
