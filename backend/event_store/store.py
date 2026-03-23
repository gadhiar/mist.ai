"""Append-only event store backed by SQLite.

Layer 1 of the MIST data architecture. Every conversation turn is
recorded immutably. The knowledge graph (Layer 3) can be fully
rebuilt from these events plus the ontology (Layer 2).

Thread safety: Each public method acquires its own connection from
a shared connection with check_same_thread=False. All writes are
serialized by SQLite's WAL-mode writer lock.
"""

import contextlib
import json
import logging
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.event_store.models import ConversationSession, ConversationTurnEvent

logger = logging.getLogger(__name__)

# Default paths under ~/.mist/
_DEFAULT_DB_DIR = Path.home() / ".mist"
_DEFAULT_DB_PATH = _DEFAULT_DB_DIR / "event_store.db"
_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class EventStore:
    """Append-only event store backed by SQLite.

    Layer 1 of the MIST data architecture. All writes are immutable --
    once a turn is appended, it is never modified or deleted.
    """

    def __init__(
        self,
        db_path: str | None = None,
    ) -> None:
        """Initialize with database path.

        Args:
            db_path: Path to SQLite database file. Defaults to ~/.mist/event_store.db.
        """
        self.db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self._conn: sqlite3.Connection | None = None

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create the database connection.

        Returns:
            sqlite3.Connection configured for WAL mode and dict rows.
        """
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                isolation_level=None,  # autocommit for PRAGMAs
            )
            self._conn.row_factory = sqlite3.Row
            # Enable WAL and foreign keys on every new connection
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")

        return self._conn

    def initialize(self) -> None:
        """Create database file, tables, and indexes. Idempotent.

        Creates the parent directory if it does not exist, then
        executes schema.sql against the database.
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        schema_sql = _SCHEMA_PATH.read_text(encoding="utf-8")

        conn = self._get_connection()
        conn.executescript(schema_sql)

        logger.info("Event store initialized at %s", self.db_path)

    def start_session(self, input_modality: str = "voice") -> str:
        """Start a new conversation session.

        Args:
            input_modality: How the user is interacting. One of "voice", "text", "api".

        Returns:
            session_id as a UUID string.
        """
        session_id = str(uuid.uuid4())
        started_at = datetime.now(UTC).isoformat()

        conn = self._get_connection()
        conn.execute(
            """
            INSERT INTO conversation_sessions (session_id, started_at, input_modality)
            VALUES (?, ?, ?)
            """,
            (session_id, started_at, input_modality),
        )

        logger.info("Started session %s (modality=%s)", session_id, input_modality)
        return session_id

    def end_session(self, session_id: str) -> None:
        """Mark a session as ended.

        Sets ended_at to the current timestamp. No-op if session
        does not exist or is already ended.

        Args:
            session_id: UUID of the session to end.
        """
        ended_at = datetime.now(UTC).isoformat()

        conn = self._get_connection()
        cursor = conn.execute(
            """
            UPDATE conversation_sessions
            SET ended_at = ?
            WHERE session_id = ? AND ended_at IS NULL
            """,
            (ended_at, session_id),
        )

        if cursor.rowcount == 0:
            logger.warning(
                "end_session called for session %s but no active session found",
                session_id,
            )
        else:
            logger.info("Ended session %s", session_id)

    def append_turn(self, event: ConversationTurnEvent) -> str:
        """Append a conversation turn event. Immutable after write.

        Assigns a UUID event_id if not already set, inserts the row,
        and increments the session turn_count atomically.

        Args:
            event: The turn event to append.

        Returns:
            The event_id (UUID string) of the appended turn.

        Raises:
            sqlite3.IntegrityError: If session_id does not exist in
                conversation_sessions (foreign key violation).
        """
        # Ensure event_id is set
        if not event.event_id:
            event.event_id = str(uuid.uuid4())

        row = event.to_dict()

        conn = self._get_connection()
        try:
            conn.execute("BEGIN")

            conn.execute(
                """
                INSERT INTO conversation_turn_events (
                    event_id, session_id, turn_index, timestamp,
                    user_utterance, system_response,
                    context_window, retrieval_context, tool_calls,
                    audio_hash, audio_format, audio_duration_ms, audio_sample_rate,
                    stt_model, tts_model, llm_model, llm_parameters,
                    ontology_version
                ) VALUES (
                    :event_id, :session_id, :turn_index, :timestamp,
                    :user_utterance, :system_response,
                    :context_window, :retrieval_context, :tool_calls,
                    :audio_hash, :audio_format, :audio_duration_ms, :audio_sample_rate,
                    :stt_model, :tts_model, :llm_model, :llm_parameters,
                    :ontology_version
                )
                """,
                row,
            )

            conn.execute(
                """
                UPDATE conversation_sessions
                SET turn_count = turn_count + 1
                WHERE session_id = ?
                """,
                (event.session_id,),
            )

            conn.execute("COMMIT")

        except Exception:
            conn.execute("ROLLBACK")
            logger.error(
                "Failed to append turn %s for session %s",
                event.event_id,
                event.session_id,
                exc_info=True,
            )
            raise

        logger.debug(
            "Appended turn %s (session=%s, index=%d)",
            event.event_id,
            event.session_id,
            event.turn_index,
        )

        return event.event_id

    def get_session(self, session_id: str) -> ConversationSession | None:
        """Retrieve session metadata.

        Args:
            session_id: UUID of the session.

        Returns:
            ConversationSession or None if not found.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM conversation_sessions WHERE session_id = ?",
            (session_id,),
        )
        row = cursor.fetchone()

        if row is None:
            return None

        return ConversationSession.from_row(dict(row))

    def get_turns(self, session_id: str) -> list[dict[str, Any]]:
        """Retrieve all turns for a session, ordered by turn_index.

        Args:
            session_id: UUID of the session.

        Returns:
            List of turn dicts with JSON fields decoded.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT * FROM conversation_turn_events
            WHERE session_id = ?
            ORDER BY turn_index ASC
            """,
            (session_id,),
        )
        return [self._decode_turn_row(dict(row)) for row in cursor.fetchall()]

    def get_turns_since(self, since: datetime) -> list[dict[str, Any]]:
        """Retrieve all turns since a timestamp.

        Used by the self-reflection curation job to find recent turns.

        Args:
            since: Datetime threshold (inclusive).

        Returns:
            List of turn dicts ordered by timestamp ascending.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT * FROM conversation_turn_events
            WHERE timestamp >= ?
            ORDER BY timestamp ASC
            """,
            (since.isoformat(),),
        )
        return [self._decode_turn_row(dict(row)) for row in cursor.fetchall()]

    def get_all_turns_for_reextraction(
        self,
        ontology_version: str | None = None,
        after_event_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve turns for re-extraction during ontology migration.

        Optionally filters by the ontology_version they were originally
        extracted under, and supports cursor-based resumption via
        after_event_id.

        Args:
            ontology_version: Only return turns tagged with this version.
            after_event_id: Resume after this event_id (for job checkpointing).

        Returns:
            List of turn dicts ordered by rowid (insertion order).
        """
        conditions: list[str] = []
        params: list[str] = []

        if ontology_version is not None:
            conditions.append("ontology_version = ?")
            params.append(ontology_version)

        if after_event_id is not None:
            # Use rowid for stable ordering since event_id is a UUID
            conditions.append(
                "rowid > (SELECT rowid FROM conversation_turn_events WHERE event_id = ?)"
            )
            params.append(after_event_id)

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        query = f"""
            SELECT * FROM conversation_turn_events
            {where_clause}
            ORDER BY rowid ASC
        """  # nosec B608 -- where_clause is built from hardcoded conditions with parameterized values

        conn = self._get_connection()
        cursor = conn.execute(query, params)
        return [self._decode_turn_row(dict(row)) for row in cursor.fetchall()]

    def get_turn_count(self) -> int:
        """Total number of stored turns across all sessions.

        Returns:
            Integer count.
        """
        conn = self._get_connection()
        cursor = conn.execute("SELECT COUNT(*) FROM conversation_turn_events")
        result = cursor.fetchone()
        return result[0] if result else 0

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.info("Event store connection closed")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_turn_row(row: dict[str, Any]) -> dict[str, Any]:
        """Decode JSON-serialized fields in a turn row.

        Modifies the dict in-place and returns it. JSON fields that
        fail to parse are left as their raw string value.

        Args:
            row: Raw dict from sqlite3.Row.

        Returns:
            Dict with context_window, retrieval_context, tool_calls,
            and llm_parameters decoded from JSON strings.
        """
        json_fields = ("context_window", "retrieval_context", "tool_calls", "llm_parameters")
        for field_name in json_fields:
            value = row.get(field_name)
            if value is not None and isinstance(value, str):
                with contextlib.suppress(json.JSONDecodeError, TypeError):
                    row[field_name] = json.loads(value)
        return row
