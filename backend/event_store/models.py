"""Event Store data models.

Dataclasses for conversation sessions and turn events. These map directly
to the SQLite tables in schema.sql. All timestamps are ISO-8601 strings
when serialized. Complex fields (context_window, retrieval_context,
tool_calls, llm_parameters) are JSON-serialized for storage.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


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

    # Versioning
    ontology_version: str = "1.0.0"

    # Assigned by EventStore.append_turn(), not by caller
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for SQLite insertion.

        JSON-encodes complex fields. Converts datetime to ISO-8601 string.

        Returns:
            Dict with string keys matching column names.
        """
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "turn_index": self.turn_index,
            "timestamp": self.timestamp.isoformat(),
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
            ontology_version=row.get("ontology_version", "1.0.0"),
        )
