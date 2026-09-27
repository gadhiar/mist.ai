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

from backend.errors import MistError
from backend.event_store.models import ConversationSession, ConversationTurnEvent

logger = logging.getLogger(__name__)

# Epoch cutover lifecycle (T2b). See the `epoch_cutover` table in schema.sql.
CUTOVER_STATES = ("filling", "ready", "checked", "promoted", "abandoned")
OPEN_CUTOVER_STATES = ("filling", "ready", "checked")
_CUTOVER_CHECK_COLUMNS = frozenset({"rebuild_job_id", "rebuilt_through_event_id", "check_report"})


class EpochCutoverStateError(MistError):
    """A cutover operation was refused because of the cutover's or the ledger's state."""


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
            # "After" must mean after IN REPLAY ORDER, not after by rowid. A cursor
            # keyed on a different order than the ORDER BY below can skip or repeat
            # turns the moment the two disagree -- which is exactly what inserting
            # out of chronological order produces. Row-value comparison keeps the
            # cursor and the order the same expression (SQLite >= 3.15).
            conditions.append(
                "(e.timestamp, e.session_id, e.turn_index) > "
                "(SELECT timestamp, session_id, turn_index "
                " FROM conversation_turn_events WHERE event_id = ?)"
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
        # Ordered by CONTENT, not by the database file. MIS-138.
        #
        # This read was `ORDER BY e.rowid ASC`, whose comment said "use rowid for
        # stable ordering since event_id is a UUID" -- right about `event_id`
        # (assigned `str(uuid.uuid4())`, so ordering by it is an arbitrary
        # permutation) and wrong about the remedy. `schema.sql` declares
        # `event_id TEXT PRIMARY KEY` with no INTEGER PRIMARY KEY, so rowid is an
        # implicit physical row number that SQLite documents VACUUM may renumber.
        # Replay order was therefore a property of the FILE, while ADR-023 claims
        # the entity subgraph is a function of the LOG.
        #
        # Not cosmetic: `EntityDeduplicator._find_existing` resolves each incoming
        # entity against whatever is already in the target graph, so processing
        # order decides which entity wins display_name, description, entity_type
        # and the alias union.
        #
        # `timestamp` leads because it is the order the LIVE path applied turns in
        # (arrival order), which is what a rebuild must reproduce;
        # (session_id, turn_index) is the deterministic tiebreak for equal stamps.
        # Lexicographic TEXT ordering equals chronological ordering because
        # `ConversationTurnEvent.to_row` normalises the column to UTC.
        query = f"""
            SELECT e.* FROM conversation_turn_events AS e
            LEFT JOIN conversation_sessions AS s ON s.session_id = e.session_id
            {where_clause}
            ORDER BY e.timestamp ASC, e.session_id ASC, e.turn_index ASC
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

    # ------------------------------------------------------------------
    # Curation observability (D3)
    # ------------------------------------------------------------------

    def append_curation_job_run(
        self,
        *,
        run_id: str,
        job_name: str,
        trigger_source: str,
        started_at: str,
        duration_ms: float,
        outcome: str,
        result_type: str | None,
        examined: int | None,
        produced: int | None,
        metrics: str | None,
        error: str | None,
    ) -> str:
        """Record one curation job execution. Append-only.

        Every parameter is keyword-only: the row has four integer-ish columns
        in a row (`duration_ms`, `examined`, `produced`) whose meanings are not
        recoverable from position at a call site.

        `examined` and `produced` are stored as separate nullable columns
        rather than folded into `metrics` so a later query can filter on them
        without parsing JSON -- see `schema.sql` for what each NULL means.

        Args:
            run_id: Unique id for this execution.
            job_name: The `JobConfig.name` that ran.
            trigger_source: 'scheduled' (the loop) or 'manual' (`run_once`).
            started_at: ISO-8601 timestamp taken before the job was awaited.
            duration_ms: Scheduler-measured wall clock, not the job's own.
            outcome: 'completed' if `run()` returned, 'failed' if it raised.
            result_type: Class name of the returned result, None on failure.
            examined: Units of input the job reports having looked at.
            produced: Units of output the job reports having written.
            metrics: JSON object of every field of the result, or None.
            error: Exception text, None unless outcome is 'failed'.

        Returns:
            The `run_id` that was written.
        """
        conn = self._get_connection()
        conn.execute(
            """
            INSERT INTO curation_job_runs (
                run_id, job_name, trigger_source, started_at, duration_ms,
                outcome, result_type, examined, produced, metrics, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                job_name,
                trigger_source,
                started_at,
                duration_ms,
                outcome,
                result_type,
                examined,
                produced,
                metrics,
                error,
            ),
        )
        return run_id

    def get_curation_job_runs(
        self, job_name: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Read back curation job runs, most recent first.

        Ordered by rowid rather than `started_at`: two runs of different jobs
        in the same scheduler pass can share an ISO timestamp to the
        microsecond, and insertion order is the only total order available.

        Args:
            job_name: Restrict to one job. None returns every job's runs.
            limit: Maximum rows to return.

        Returns:
            List of row dicts with `metrics` decoded from JSON.
        """
        conn = self._get_connection()
        if job_name is None:
            cursor = conn.execute(
                "SELECT * FROM curation_job_runs ORDER BY rowid DESC LIMIT ?",
                (limit,),
            )
        else:
            cursor = conn.execute(
                "SELECT * FROM curation_job_runs WHERE job_name = ? ORDER BY rowid DESC LIMIT ?",
                (job_name, limit),
            )
        return [self._decode_metrics_row(dict(row)) for row in cursor.fetchall()]

    def append_graph_health_event(
        self,
        *,
        event_id: str,
        timestamp: str,
        health_score: float,
        metrics: str,
        entity_count: int | None,
        relationship_count: int | None,
        archived_count: int | None = None,
        community_count: int | None = None,
    ) -> str:
        """Record one graph health measurement. Append-only.

        This is the health TIME SERIES the `graph_health_events` table was
        declared for in Phase 4 and that nothing ever wrote to. It is not a
        substitute for `curation_job_runs`: `health_score` is NOT NULL, so a
        health run that RAISED has no representable row here. The run ledger
        records that; this records the measurement.

        `archived_count` and `community_count` stay None. They are outputs of
        `ConfidenceDecayJob` and `CommunityDetector`, which run on their own
        intervals -- filling them from a different job's most recent run would
        stamp this measurement with numbers that were not measured with it.

        Args:
            event_id: Unique id for this measurement.
            timestamp: ISO-8601 timestamp of the measurement.
            health_score: The composite 0-100 `overall` score.
            metrics: JSON object of the component sub-scores.
            entity_count: Active entity count at measurement time.
            relationship_count: Current-belief relationship count.
            archived_count: Not populated by the scheduler. See above.
            community_count: Not populated by the scheduler. See above.

        Returns:
            The `event_id` that was written.
        """
        conn = self._get_connection()
        conn.execute(
            """
            INSERT INTO graph_health_events (
                event_id, timestamp, health_score, metrics,
                entity_count, relationship_count, archived_count, community_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                timestamp,
                health_score,
                metrics,
                entity_count,
                relationship_count,
                archived_count,
                community_count,
            ),
        )
        return event_id

    def get_graph_health_events(self, limit: int = 100) -> list[dict[str, Any]]:
        """Read back graph health measurements, most recent first.

        Args:
            limit: Maximum rows to return.

        Returns:
            List of row dicts with `metrics` decoded from JSON.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM graph_health_events ORDER BY rowid DESC LIMIT ?",
            (limit,),
        )
        return [self._decode_metrics_row(dict(row)) for row in cursor.fetchall()]

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

    # ------------------------------------------------------------------
    # Extraction backlog (T2a)
    # ------------------------------------------------------------------

    def list_turn_keys_in_replay_order(self) -> list[dict[str, Any]]:
        """Every logged turn's identity, in replay order, without its payload.

        The backlog's ordering source. Same ORDER BY as
        `get_all_turns_for_reextraction` (`timestamp, session_id, turn_index` --
        see the MIS-138 comment there for why content order and not rowid), so
        the live dispatcher applies turns in the order a rebuild replays them.

        Deliberately unfiltered: EVERY turn the live path records is eligible,
        whatever its session's `origin` and whatever `ontology_version` it was
        logged under. That matches what the pre-T2a live path did -- it fired
        extraction for every recorded turn (`conversation_handler.py`, the
        `if event_id:` branch after `_record_turn_event`) with no origin or
        ontology check. A rebuild still scopes itself (`LogRegenerator.rebuild`
        passes `origins=CANONICAL_ORIGINS`); that is the rebuild's policy, not
        the backlog's.

        Returns:
            Dicts with `event_id`, `session_id`, `turn_index` and `timestamp`
            only -- the full row (context window, retrieval context) is read per
            turn by `get_turn` when the turn reaches the head.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT event_id, session_id, turn_index, timestamp
            FROM conversation_turn_events
            ORDER BY timestamp ASC, session_id ASC, turn_index ASC
            """
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_turn(self, event_id: str) -> dict[str, Any] | None:
        """One turn row with its JSON fields decoded, or None if absent."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM conversation_turn_events WHERE event_id = ?", (event_id,)
        ).fetchone()
        return self._decode_turn_row(dict(row)) if row is not None else None

    def get_extraction_activation(self, epoch_id: int) -> dict[str, Any] | None:
        """The first-activation record for an epoch, or None before activation."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM extraction_activation WHERE epoch_id = ?", (epoch_id,)
        ).fetchone()
        return dict(row) if row is not None else None

    def record_extraction_activation(
        self,
        *,
        epoch_id: int,
        activated_at: str,
        turns_at_activation: int,
        applied_event_ids: list[str],
        legacy_event_ids: list[str],
    ) -> bool:
        """Write the first-activation floor for an epoch, in ONE transaction.

        One transaction so a crash cannot leave a floor row without its markers
        (the next start would then treat the floor as done and dispatch the
        legacy turns it failed to list) or markers without the floor row (the
        next start would re-run activation over them -- harmless, but the count
        in the floor row would be wrong).

        Idempotent: returns False and writes nothing when the epoch already has
        a floor, so a restart never moves it.

        Returns:
            True when the floor was written by this call.
        """
        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT 1 FROM extraction_activation WHERE epoch_id = ?", (epoch_id,)
            ).fetchone()
            if existing is not None:
                conn.execute("COMMIT")
                return False
            conn.execute(
                "INSERT INTO extraction_activation (epoch_id, activated_at, turns_at_activation, "
                "marked_applied, legacy_unextracted) VALUES (?, ?, ?, ?, ?)",
                (
                    epoch_id,
                    activated_at,
                    turns_at_activation,
                    len(applied_event_ids),
                    len(legacy_event_ids),
                ),
            )
            conn.executemany(
                "INSERT OR IGNORE INTO extraction_applied "
                "(event_id, epoch_id, stage, source, updated_at) "
                "VALUES (?, ?, 'applied', 'activation', ?)",
                [(eid, epoch_id, activated_at) for eid in applied_event_ids],
            )
            conn.executemany(
                "INSERT OR IGNORE INTO extraction_legacy_turns (event_id, epoch_id) VALUES (?, ?)",
                [(eid, epoch_id) for eid in legacy_event_ids],
            )
            conn.execute("COMMIT")
        except sqlite3.Error:
            conn.execute("ROLLBACK")
            raise
        return True

    def get_extraction_legacy_turns(self, epoch_id: int) -> set[str]:
        """Event ids logged before first activation with no cache row (never dispatched)."""
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT event_id FROM extraction_legacy_turns WHERE epoch_id = ?", (epoch_id,)
        ).fetchall()
        return {row[0] for row in rows}

    def get_extraction_applied(self, epoch_id: int) -> dict[str, str]:
        """Map event_id -> apply stage ('curated' or 'applied') for an epoch."""
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT event_id, stage FROM extraction_applied WHERE epoch_id = ?", (epoch_id,)
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    def mark_extraction_stage(
        self, *, event_id: str, epoch_id: int, stage: str, updated_at: str
    ) -> None:
        """Upsert a turn's apply stage for an epoch ('curated' then 'applied').

        Raises:
            ValueError: `stage` is not 'curated' or 'applied'.
        """
        if stage not in ("curated", "applied"):
            raise ValueError(f"unknown extraction apply stage {stage!r}")
        conn = self._get_connection()
        conn.execute(
            "INSERT INTO extraction_applied (event_id, epoch_id, stage, source, updated_at) "
            "VALUES (?, ?, ?, 'dispatcher', ?) "
            "ON CONFLICT(event_id, epoch_id) DO UPDATE SET stage = excluded.stage, "
            "source = excluded.source, updated_at = excluded.updated_at",
            (event_id, epoch_id, stage, updated_at),
        )

    def delete_extraction_applied(self, event_id: str, epoch_id: int) -> bool:
        """Remove a turn's apply marker for an epoch. True when a row was deleted."""
        conn = self._get_connection()
        cursor = conn.execute(
            "DELETE FROM extraction_applied WHERE event_id = ? AND epoch_id = ?",
            (event_id, epoch_id),
        )
        return cursor.rowcount > 0

    def append_extraction_attempt(
        self,
        *,
        event_id: str,
        epoch_id: int,
        attempt: int,
        job_id: str,
        request_id: str,
        turn_id: str,
        started_at: str,
        finished_at: str,
        duration_ms: float,
        error_code: str | None,
        outcome: str,
        counted: bool,
    ) -> None:
        """Record one extraction-service call for a turn. Append-only."""
        conn = self._get_connection()
        conn.execute(
            "INSERT INTO extraction_attempts (event_id, epoch_id, attempt, job_id, request_id, "
            "turn_id, started_at, finished_at, duration_ms, error_code, outcome, counted) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event_id,
                epoch_id,
                attempt,
                job_id,
                request_id,
                turn_id,
                started_at,
                finished_at,
                duration_ms,
                error_code,
                outcome,
                1 if counted else 0,
            ),
        )

    def count_extraction_attempts(self, event_id: str, epoch_id: int) -> int:
        """All recorded service calls for a turn and epoch, retired ones included."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT COUNT(*) FROM extraction_attempts WHERE event_id = ? AND epoch_id = ?",
            (event_id, epoch_id),
        ).fetchone()
        return int(row[0])

    def count_counted_extraction_failures(self, event_id: str, epoch_id: int) -> int:
        """Job-attributable failures since the turn last (re-)entered the backlog."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT COUNT(*) FROM extraction_attempts "
            "WHERE event_id = ? AND epoch_id = ? AND counted = 1 AND retired = 0",
            (event_id, epoch_id),
        ).fetchone()
        return int(row[0])

    def retire_extraction_attempts(self, event_id: str, epoch_id: int) -> int:
        """Mark a turn's attempts retired so its failure count restarts at zero."""
        conn = self._get_connection()
        cursor = conn.execute(
            "UPDATE extraction_attempts SET retired = 1 "
            "WHERE event_id = ? AND epoch_id = ? AND retired = 0",
            (event_id, epoch_id),
        )
        return cursor.rowcount

    def list_extraction_attempts(
        self, *, event_id: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Recorded service calls, oldest first, optionally for one turn."""
        conn = self._get_connection()
        if event_id is None:
            cursor = conn.execute(
                "SELECT * FROM extraction_attempts ORDER BY attempt_row ASC LIMIT ?", (limit,)
            )
        else:
            cursor = conn.execute(
                "SELECT * FROM extraction_attempts WHERE event_id = ? "
                "ORDER BY attempt_row ASC LIMIT ?",
                (event_id, limit),
            )
        return [dict(row) for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # Epoch cutover (T2b)
    # ------------------------------------------------------------------

    def begin_epoch_cutover(
        self,
        *,
        ontology_version: str,
        extraction_version: str,
        model_hash: str,
        bare_model_hash: str,
        source_epoch_id: int,
        requested_at: str,
    ) -> int | None:
        """Open a cutover candidate in state 'filling', unless one is already open.

        The candidate goes into `epoch_cutover`, NOT `epoch_ledger`: the latest
        ledger row is the active epoch (`get_current_epoch`), so a ledger row
        here would make the candidate live before it was filled or checked.

        The at-most-one-open rule is checked and written inside one
        `BEGIN IMMEDIATE` transaction, so two concurrent callers cannot both
        open one.

        Args:
            model_hash: The COMPOSED stamp (`compose_model_hash`), the value the
                extraction cache is keyed under.
            bare_model_hash: The service's own model identity (`/v1/info`).
            source_epoch_id: The active epoch the cutover starts from;
                promotion refuses if the ledger has moved on since.

        Returns:
            The new `cutover_id`, or None when a cutover is already open.
        """
        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            placeholders = ", ".join("?" * len(OPEN_CUTOVER_STATES))
            existing = conn.execute(
                f"SELECT 1 FROM epoch_cutover WHERE state IN ({placeholders})",  # nosec B608
                OPEN_CUTOVER_STATES,
            ).fetchone()
            if existing is not None:
                conn.execute("COMMIT")
                return None
            cursor = conn.execute(
                "INSERT INTO epoch_cutover (ontology_version, extraction_version, model_hash, "
                "bare_model_hash, requested_at, state, source_epoch_id, updated_at) "
                "VALUES (?, ?, ?, ?, ?, 'filling', ?, ?)",
                (
                    ontology_version,
                    extraction_version,
                    model_hash,
                    bare_model_hash,
                    requested_at,
                    source_epoch_id,
                    requested_at,
                ),
            )
            conn.execute("COMMIT")
        except sqlite3.Error:
            conn.execute("ROLLBACK")
            raise
        return int(cursor.lastrowid)

    def get_open_epoch_cutover(self) -> dict[str, Any] | None:
        """The open cutover (filling, ready or checked), or None."""
        conn = self._get_connection()
        placeholders = ", ".join("?" * len(OPEN_CUTOVER_STATES))
        row = conn.execute(
            f"SELECT * FROM epoch_cutover WHERE state IN ({placeholders}) "  # nosec B608
            "ORDER BY cutover_id DESC LIMIT 1",
            OPEN_CUTOVER_STATES,
        ).fetchone()
        return dict(row) if row is not None else None

    def get_epoch_cutover(self, cutover_id: int) -> dict[str, Any] | None:
        """One cutover row by id, or None."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM epoch_cutover WHERE cutover_id = ?", (cutover_id,)
        ).fetchone()
        return dict(row) if row is not None else None

    def list_epoch_cutovers(self) -> list[dict[str, Any]]:
        """Every cutover, oldest first."""
        conn = self._get_connection()
        rows = conn.execute("SELECT * FROM epoch_cutover ORDER BY cutover_id ASC").fetchall()
        return [dict(r) for r in rows]

    def transition_epoch_cutover(
        self,
        cutover_id: int,
        *,
        from_states: tuple[str, ...],
        to_state: str,
        updated_at: str,
        fields: dict[str, Any] | None = None,
    ) -> bool:
        """Compare-and-set a cutover's state, optionally writing check columns.

        Refuses 'promoted': only `promote_epoch_cutover` may set it, because
        promotion must be atomic with the ledger append and the activation.

        Args:
            from_states: The update applies only while the row is in one of these.
            to_state: The new state.
            fields: Optional values for `rebuild_job_id`,
                `rebuilt_through_event_id` and `check_report` (a JSON string).
                A None value writes SQL NULL.

        Returns:
            True when the row was in one of `from_states` and was updated.

        Raises:
            ValueError: `to_state` is unknown or 'promoted', or `fields` names a
                column this method does not write.
        """
        if to_state not in CUTOVER_STATES or to_state == "promoted":
            raise ValueError(f"cannot transition a cutover to {to_state!r} here")
        values = dict(fields or {})
        unknown = set(values) - _CUTOVER_CHECK_COLUMNS
        if unknown:
            raise ValueError(f"not a writable cutover column: {sorted(unknown)}")
        assignments = ["state = ?", "updated_at = ?"] + [f"{name} = ?" for name in values]
        params: list[Any] = [to_state, updated_at, *values.values(), cutover_id, *from_states]
        placeholders = ", ".join("?" * len(from_states))
        conn = self._get_connection()
        cursor = conn.execute(
            f"UPDATE epoch_cutover SET {', '.join(assignments)} "  # nosec B608 -- names allowlisted
            f"WHERE cutover_id = ? AND state IN ({placeholders})",
            params,
        )
        return cursor.rowcount > 0

    def promote_epoch_cutover(self, *, cutover_id: int, activated_at: str) -> dict[str, Any]:
        """Make a checked cutover the active epoch, in ONE transaction.

        Inside a single `BEGIN IMMEDIATE`:

        1. append the candidate's stamp triple to `epoch_ledger` (prev = the
           cutover's source epoch), which makes it the active epoch;
        2. write the new epoch's `extraction_activation` row FIRST-HAND, with
           `legacy_unextracted = 0`;
        3. mark `applied` (source 'cutover') exactly the logged turns up to and
           including `rebuilt_through_event_id` in replay order -- the swapped-in
           graph already contains them -- and nothing after it, so the
           dispatcher applies the later turns to the new live graph in order;
        4. set the cutover to 'promoted'.

        Step 2 is what bypasses T2a's automatic first activation
        (`BacklogStore.ensure_activation`), which would otherwise run on the
        dispatcher's next step, find no activation row for the new epoch, and
        mark EVERY turn with a candidate cache row applied (including the ones
        after `rebuilt_through_event_id`, which the swapped graph lacks) and
        call every uncached turn legacy. With the row present it returns the
        stored floor unchanged.

        Atomic because steps 1 and 2 must not be separable: a ledger row with
        no activation is exactly the state in which the automatic rule would
        fire. A crash anywhere before COMMIT leaves none of the four written.

        Raises:
            EpochCutoverStateError: The cutover is not 'checked', has no
                `rebuilt_through_event_id`, that turn is not in the log, or the
                active epoch is no longer the cutover's source epoch.
        """
        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM epoch_cutover WHERE cutover_id = ?", (cutover_id,)
            ).fetchone()
            if row is None or row["state"] != "checked":
                state = None if row is None else row["state"]
                raise EpochCutoverStateError(
                    f"cutover {cutover_id} is {state!r}; only a 'checked' cutover can be promoted"
                )
            through = row["rebuilt_through_event_id"]
            if not through:
                raise EpochCutoverStateError(
                    f"cutover {cutover_id} has no rebuilt_through_event_id; re-run the check"
                )
            current = conn.execute(
                "SELECT * FROM epoch_ledger ORDER BY epoch_id DESC LIMIT 1"
            ).fetchone()
            if current is None or int(current["epoch_id"]) != row["source_epoch_id"]:
                raise EpochCutoverStateError(
                    f"the active epoch is "
                    f"{None if current is None else int(current['epoch_id'])}, but cutover "
                    f"{cutover_id} began from epoch {row['source_epoch_id']}; abandon it and "
                    "begin again"
                )
            keys = [
                r[0]
                for r in conn.execute(
                    "SELECT event_id FROM conversation_turn_events "
                    "ORDER BY timestamp ASC, session_id ASC, turn_index ASC"
                ).fetchall()
            ]
            if through not in keys:
                raise EpochCutoverStateError(
                    f"rebuilt_through_event_id {through!r} is not in the event log"
                )
            applied = keys[: keys.index(through) + 1]

            cursor = conn.execute(
                "INSERT INTO epoch_ledger (ontology_version, extraction_version, model_hash, "
                "activated_at, prev_epoch_id, provisional) VALUES (?, ?, ?, ?, ?, 0)",
                (
                    row["ontology_version"],
                    row["extraction_version"],
                    row["model_hash"],
                    activated_at,
                    int(current["epoch_id"]),
                ),
            )
            epoch_id = int(cursor.lastrowid)
            self._promotion_fault_point("ledger_appended")
            conn.execute(
                "INSERT INTO extraction_activation (epoch_id, activated_at, turns_at_activation, "
                "marked_applied, legacy_unextracted) VALUES (?, ?, ?, ?, 0)",
                (epoch_id, activated_at, len(keys), len(applied)),
            )
            conn.executemany(
                "INSERT INTO extraction_applied (event_id, epoch_id, stage, source, updated_at) "
                "VALUES (?, ?, 'applied', 'cutover', ?)",
                [(event_id, epoch_id, activated_at) for event_id in applied],
            )
            conn.execute(
                "UPDATE epoch_cutover SET state = 'promoted', promoted_epoch_id = ?, "
                "updated_at = ? WHERE cutover_id = ?",
                (epoch_id, activated_at, cutover_id),
            )
            conn.execute("COMMIT")
        except BaseException:
            # BaseException, not Exception: a KeyboardInterrupt (or any abort)
            # between the ledger append and the activation must not leave the
            # connection holding a half-written transaction a later statement
            # on this shared connection could COMMIT.
            conn.execute("ROLLBACK")
            raise
        return {
            "epoch_id": epoch_id,
            "turns_at_activation": len(keys),
            "marked_applied": len(applied),
        }

    def _promotion_fault_point(self, step: str) -> None:
        """No-op seam between promotion's ledger append and its activation write.

        Tests replace it to inject a crash at exactly that point and prove the
        transaction leaves neither written. Production never overrides it.
        """

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

    @staticmethod
    def _decode_metrics_row(row: dict[str, Any]) -> dict[str, Any]:
        """Decode the `metrics` JSON column on a curation or health row.

        A value that fails to parse is left as its raw string rather than
        dropped -- a reader diagnosing a job would rather see malformed JSON
        than a silent None.

        Args:
            row: Raw dict from sqlite3.Row.

        Returns:
            The same dict with `metrics` decoded when it parses.
        """
        value = row.get("metrics")
        if isinstance(value, str):
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                row["metrics"] = json.loads(value)
        return row
