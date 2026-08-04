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

        # `CREATE TABLE IF NOT EXISTS` leaves a pre-existing table untouched,
        # so a database created before `origin` existed needs the column added
        # explicitly. Guarded on PRAGMA rather than caught-and-ignored so a
        # genuine failure still surfaces.
        columns = {row[1] for row in conn.execute("PRAGMA table_info(conversation_sessions)")}
        if "origin" not in columns:
            conn.execute(
                "ALTER TABLE conversation_sessions ADD COLUMN origin TEXT NOT NULL DEFAULT 'real'"
            )
            logger.info("Event store: added `origin` column to conversation_sessions")

        # R1.4 Task 7: `epoch_ledger` predates the `provisional` column --
        # same guard shape as `origin` above, and for the same reason: a
        # database created before this change needs the column added
        # explicitly, since `CREATE TABLE IF NOT EXISTS` leaves it alone.
        epoch_columns = {row[1] for row in conn.execute("PRAGMA table_info(epoch_ledger)")}
        if "provisional" not in epoch_columns:
            conn.execute(
                "ALTER TABLE epoch_ledger ADD COLUMN provisional INTEGER NOT NULL DEFAULT 0"
            )
            logger.info("Event store: added `provisional` column to epoch_ledger")

        logger.info("Event store initialized at %s", self.db_path)

    def start_session(
        self, session_id: str, input_modality: str = "voice", origin: str = "real"
    ) -> str:
        """Start a new conversation session.

        R1.3.1 fix round 1: `session_id` is now supplied by the caller
        rather than minted here. Previously this method generated its own
        `uuid4`, giving the event store a session-id namespace independent
        of the chat layer's -- `ConversationHandler` bridged the two via an
        `_es_session_ids` dict, and every downstream consumer that assumed
        a single namespace (the vault path allocator, the Neo4j
        `ConversationContext` anchor, startup catch-up) silently broke on
        that assumption. Collapsing the namespaces -- the event store's
        `session_id` now IS the chat layer's `session_id` -- removes the
        bridge and the class of bug it enabled, rather than patching each
        consumer to translate correctly.

        Args:
            session_id: The session identifier to record under. Callers
                pass the chat-layer session id directly; this is now the
                single identifier used across SQLite, the Neo4j
                `ConversationContext` anchor, and vault note paths.
            input_modality: How the user is interacting. One of "voice", "text", "api".
            origin: Provenance of this session. "real" is genuine usage and is
                what R1.6's cutover rebuilds from; "test" marks harness and
                probe traffic; "seed" is reserved. Defaults to "real" so a
                caller that forgets is counted as real rather than silently
                excluded from a rebuild.

        Returns:
            The `session_id` that was passed in, for call-site symmetry
            with the pre-collapse signature (callers that used the return
            value continue to work unchanged).
        """
        started_at = datetime.now(UTC).isoformat()

        conn = self._get_connection()
        conn.execute(
            """
            INSERT INTO conversation_sessions (session_id, started_at, input_modality, origin)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, started_at, input_modality, origin),
        )

        logger.info(
            "Started session %s (modality=%s, origin=%s)", session_id, input_modality, origin
        )
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
        origins: tuple[str, ...] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve turns for re-extraction during ontology migration.

        Optionally filters by the ontology_version they were originally
        extracted under, by the provenance of the session they belong to, and
        supports cursor-based resumption via after_event_id.

        All three filters default to None (no filtering) because this is a
        neutral store read, not the rebuild's policy. The rebuild decides what
        it is a projection of; see `LogRegenerator.rebuild`, which passes an
        epoch-derived `ontology_version` and a fail-closed `origins`.

        NULL / absent origin: a turn is joined to its session with a LEFT JOIN
        and a missing or NULL origin is COALESCEd to 'real'. Two reasons, both
        already-settled rulings in this codebase rather than a new one.
        (1) `initialize()` adds the column with `NOT NULL DEFAULT 'real'`, and
        SQLite back-fills every pre-existing row with that default -- so rows
        that predate the discriminator are already 'real' on disk, and
        excluding NULL would contradict the migration. (2) `start_session`
        documents the same ruling for its own default: "a caller that forgets
        is counted as real rather than silently excluded from a rebuild".
        The residual NULL case is a turn whose session row is absent entirely
        (a legacy database written with foreign keys off). Counting it as real
        replays it; excluding it would silently drop history from a graph whose
        entire contract is that it is a total function of the log. Losing
        history fails the contract; replaying an unmarked turn does not.

        Args:
            ontology_version: Only return turns tagged with this version.
            after_event_id: Resume after this event_id (for job checkpointing).
            origins: Only return turns whose session has one of these origins
                ('real', 'test', 'seed'). None disables the filter entirely.
                An empty tuple is rejected -- it would select nothing, which a
                caller cannot distinguish from an empty log.

        Returns:
            List of turn dicts ordered by rowid (insertion order).

        Raises:
            ValueError: If origins is an empty tuple.
        """
        if origins is not None and not origins:
            raise ValueError(
                "origins must be a non-empty tuple of provenance values, or None to "
                "disable origin filtering. An empty tuple selects no turns at all, which "
                "a rebuild cannot distinguish from an empty log."
            )

        conditions: list[str] = []
        params: list[str] = []

        if ontology_version is not None:
            conditions.append("e.ontology_version = ?")
            params.append(ontology_version)

        if after_event_id is not None:
            # Use rowid for stable ordering since event_id is a UUID
            conditions.append(
                "e.rowid > (SELECT rowid FROM conversation_turn_events WHERE event_id = ?)"
            )
            params.append(after_event_id)

        if origins is not None:
            placeholders = ", ".join(["?"] * len(origins))
            conditions.append(f"COALESCE(s.origin, 'real') IN ({placeholders})")
            params.extend(origins)

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        # LEFT JOIN, not INNER: an inner join would silently drop a turn whose
        # session row is missing, turning a data-integrity problem into missing
        # history. session_id is the sessions table's PRIMARY KEY, so the join
        # cannot multiply rows.
        query = f"""
            SELECT e.* FROM conversation_turn_events AS e
            LEFT JOIN conversation_sessions AS s ON s.session_id = e.session_id
            {where_clause}
            ORDER BY e.rowid ASC
        """  # nosec B608 -- where_clause is built from hardcoded conditions with parameterized values

        conn = self._get_connection()
        cursor = conn.execute(query, params)
        return [self._decode_turn_row(dict(row)) for row in cursor.fetchall()]

    def list_sessions_with_turns(self) -> list[str]:
        """Session ids having at least one recorded turn, oldest first.

        Used by the startup catch-up to find sessions that may need a vault
        note. R1.3.1 fix round 1 collapsed the event store's session-id
        namespace into the chat layer's: `start_session` now takes the
        session id as a parameter rather than minting its own `uuid4`, so
        there is exactly one namespace. A result from this method IS the
        chat-layer session id -- the same id `ConversationHandler` uses for
        `_vault_paths`, and the same id the Neo4j `ConversationContext`
        anchor is keyed on. No translation or lookup is needed to resolve
        one of these back to anything else.

        Ordering is by the session's earliest turn rowid so a backlog
        drains in conversation order.

        Returns:
            List of session_id strings, oldest first. Empty if no session
            has a recorded turn.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT session_id
            FROM conversation_turn_events
            GROUP BY session_id
            ORDER BY MIN(rowid)
            """
        )
        return [row[0] for row in cursor.fetchall()]

    def get_turn_count(self) -> int:
        """Total number of stored turns across all sessions.

        Returns:
            Integer count.
        """
        conn = self._get_connection()
        cursor = conn.execute("SELECT COUNT(*) FROM conversation_turn_events")
        result = cursor.fetchone()
        return result[0] if result else 0

    def append_epoch(
        self,
        ontology_version: str,
        extraction_version: str,
        model_hash: str,
        activated_at: str,
    ) -> int:
        """Append a new epoch unless the latest already has this stamp triple.

        Idempotent on an unchanged triple: returns the existing epoch_id without
        inserting. Returns the (new or existing) epoch_id.
        """
        current = self.get_current_epoch()
        if current is not None and (
            current["ontology_version"],
            current["extraction_version"],
            current["model_hash"],
        ) == (ontology_version, extraction_version, model_hash):
            return int(current["epoch_id"])

        prev_id = int(current["epoch_id"]) if current is not None else None
        conn = self._get_connection()
        cursor = conn.execute(
            """
            INSERT INTO epoch_ledger (
                ontology_version, extraction_version, model_hash, activated_at, prev_epoch_id
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (ontology_version, extraction_version, model_hash, activated_at, prev_id),
        )
        return int(cursor.lastrowid)

    def get_current_epoch(self) -> dict[str, Any] | None:
        """Return the latest epoch row as a dict, or None if the ledger is empty."""
        conn = self._get_connection()
        row = conn.execute("SELECT * FROM epoch_ledger ORDER BY epoch_id DESC LIMIT 1").fetchone()
        return dict(row) if row else None

    def list_epochs(self) -> list[dict[str, Any]]:
        """Return all epochs in insertion order (oldest first)."""
        conn = self._get_connection()
        rows = conn.execute("SELECT * FROM epoch_ledger ORDER BY epoch_id ASC").fetchall()
        return [dict(r) for r in rows]

    def ensure_initial_epoch(
        self,
        *,
        now_iso: str,
        ontology_version: str | None = None,
        extraction_version: str | None = None,
        model_hash: str | None = None,
    ) -> dict[str, Any]:
        """Ensure the epoch ledger has a reference epoch, seeding one if empty.

        R1.4 Task 7 (spec 4.3, O2): `epoch_ledger` starts with 0 rows, so
        Gate 1 (rebuild equality) has nothing to rebuild against yet. This
        writes a minimal epoch marked `provisional=1` -- a real column, not
        a comment -- so R1.6 stays free to redefine epoch semantics when it
        gives a consumer to the `ontology_version` / `extraction_version` /
        `model_hash` stamps this table carries (the `RebuildStamps`
        fields). This method's only job is to guarantee SOME reference
        epoch exists; it does not un-defer O4 -- nothing yet reads these
        columns back out.

        Idempotent on more than "no exception raised": if the ledger
        already holds any epoch -- the provisional one this method wrote on
        a prior call, or a genuine one written later via `append_epoch` --
        this returns that epoch unchanged rather than inserting a second
        row. A caller that wants to replace a provisional epoch with a real
        one should call `append_epoch` directly; this method never
        overwrites an existing epoch, provisional or not.

        `now_iso` is a caller-supplied parameter, never a clock read --
        R1.3.1 shipped a `datetime.now()` fallback that drifted across UTC
        midnight and mis-dated the only note MIST had ever written.

        Args:
            now_iso: ISO-8601 timestamp to stamp as `activated_at` on the
                inserted epoch. Unused if an epoch already exists.
            ontology_version: Stamp to write. Defaults to
                `ONTOLOGY_V1_0_0.version` (`backend.knowledge.ontologies`)
                when not given -- pass explicitly to keep a caller (tests
                included) from depending on the knowledge layer's config.
            extraction_version: Stamp to write. Defaults to
                `KnowledgeConfig.from_env().extraction_version` when not
                given.
            model_hash: Stamp to write. Defaults to
                `KnowledgeConfig.from_env().model_hash` when not given.

        Returns:
            The current epoch row as a dict -- either the one just
            inserted, or the pre-existing one.
        """
        current = self.get_current_epoch()
        if current is not None:
            return current

        if ontology_version is None or extraction_version is None or model_hash is None:
            # Local import, not module-level, and only reached when a
            # caller relies on a default: these stamp values are Layer-2
            # (ontology / extraction) concepts recorded in the Layer-1
            # ledger. Scoping the import keeps every other EventStore call
            # site -- and any caller that passes all three explicitly --
            # free of the knowledge-layer dependency.
            from backend.knowledge.config import KnowledgeConfig
            from backend.knowledge.ontologies import ONTOLOGY_V1_0_0

            # .from_env() rather than the process-global get_config(): this
            # writes a value permanently into the ledger, so it should
            # reflect a fresh read of the deployment's actual env vars
            # rather than whatever the mutable config singleton happens to
            # hold at this moment (get_config()/set_config() can be
            # repointed by unrelated code, e.g. test fixtures, for the life
            # of the process).
            config = KnowledgeConfig.from_env()
            ontology_version = ontology_version or ONTOLOGY_V1_0_0.version
            extraction_version = extraction_version or config.extraction_version
            model_hash = model_hash or config.model_hash

        conn = self._get_connection()
        cursor = conn.execute(
            """
            INSERT INTO epoch_ledger (
                ontology_version, extraction_version, model_hash, activated_at,
                prev_epoch_id, provisional
            ) VALUES (?, ?, ?, ?, NULL, 1)
            """,
            (ontology_version, extraction_version, model_hash, now_iso),
        )

        logger.info(
            "Event store: wrote provisional initial epoch %d (ontology=%s, extraction=%s)",
            cursor.lastrowid,
            ontology_version,
            extraction_version,
        )

        return {
            "epoch_id": int(cursor.lastrowid),
            "ontology_version": ontology_version,
            "extraction_version": extraction_version,
            "model_hash": model_hash,
            "activated_at": now_iso,
            "prev_epoch_id": None,
            "provisional": 1,
        }

    def create_reextraction_job(
        self,
        job_id: str,
        target_ontology_version: str,
        source_ontology_version: str | None,
        total_events: int,
        started_at: str,
    ) -> None:
        """Insert a new re-extraction job row in status 'running'."""
        conn = self._get_connection()
        conn.execute(
            "INSERT INTO re_extraction_jobs "
            "(job_id, target_ontology_version, source_ontology_version, status, "
            " total_events, processed, failed, last_event_id, started_at, updated_at) "
            "VALUES (?, ?, ?, 'running', ?, 0, 0, NULL, ?, ?)",
            (
                job_id,
                target_ontology_version,
                source_ontology_version,
                total_events,
                started_at,
                started_at,
            ),
        )

    def checkpoint_reextraction_job(
        self, job_id: str, last_event_id: str, processed: int, updated_at: str
    ) -> None:
        """Advance a job's checkpoint cursor + processed count."""
        conn = self._get_connection()
        conn.execute(
            "UPDATE re_extraction_jobs SET last_event_id = ?, processed = ?, updated_at = ? "
            "WHERE job_id = ?",
            (last_event_id, processed, updated_at, job_id),
        )

    def finalize_reextraction_job(
        self,
        job_id: str,
        status: str,
        failed: int,
        errors: str | None,
        updated_at: str,
    ) -> None:
        """Transition a job to a terminal status.

        Sets status, failed count, errors JSON, and updated_at timestamp.
        Status must be one of the terminal values ('completed', 'failed').

        Args:
            job_id: The job to finalize.
            status: Terminal status string ('completed' or 'failed').
            failed: Count of turns that had curation stage errors.
            errors: JSON-encoded list of error strings, or None if no errors.
            updated_at: ISO-8601 timestamp of the last processed turn (or
                epoch activated_at when no turns were processed).
        """
        conn = self._get_connection()
        conn.execute(
            "UPDATE re_extraction_jobs SET status=?, failed=?, errors=?, updated_at=? "
            "WHERE job_id=?",
            (status, failed, errors, updated_at, job_id),
        )

    def get_reextraction_job(self, job_id: str) -> dict[str, Any] | None:
        """Return the job row as a dict, or None if absent."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM re_extraction_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        return dict(row) if row is not None else None

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
