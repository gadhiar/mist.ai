"""Content-addressed extraction cache (F3).

Caches the Stage-2 extraction DECISION (outcome, and on a hit its entities +
relationships) keyed by (event_id, extraction_version, model_hash) -- what fed
the LLM. `ontology_version` is stored on the row for audit but deliberately
excluded from the key (spec D3): see `cache_key()` for why. A rebuild reuses
the cached decision instead of re-running the (non-bit-stable) LLM; the LLM
runs only on a stamp miss. This is the determinism boundary for Inv-A4.

No live-pipeline wiring lives here -- the rebuild driver (R1) and the live
extraction path are the callers; F3 ships the store + key function only.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

_DDL = """
CREATE TABLE IF NOT EXISTS extraction_cache (
    cache_key TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    ontology_version TEXT NOT NULL,
    extraction_version TEXT NOT NULL,
    model_hash TEXT NOT NULL,
    outcome TEXT NOT NULL DEFAULT 'extracted',
    skip_reason TEXT,
    scope TEXT,
    scope_confidence REAL,
    payload TEXT NOT NULL,
    created_at TEXT,
    derivation TEXT,
    service_stamps TEXT
);
"""

# SQLite has no ADD COLUMN IF NOT EXISTS. `initialize()` reads PRAGMA
# table_info and adds only what is missing, mirroring the conditional
# ALTER TABLE the event store already uses.
#
# `derivation` and `service_stamps` (T2a, extraction backlog) are nullable JSON
# columns and are NOT part of `cache_key()`. `derivation` holds the Stage 9
# operations the extraction service returned, persisted BEFORE the backend
# applies them, so a crash between the service reply and the graph write
# re-applies from this row instead of re-running inference. `service_stamps` is
# audit only: which service build, prompt hash and adapter produced the row.
_MIGRATION_COLUMNS: tuple[tuple[str, str], ...] = (
    ("outcome", "TEXT NOT NULL DEFAULT 'extracted'"),
    ("skip_reason", "TEXT"),
    ("scope", "TEXT"),
    ("scope_confidence", "REAL"),
    ("derivation", "TEXT"),
    ("service_stamps", "TEXT"),
)

OUTCOME_EXTRACTED = "extracted"
OUTCOME_SKIPPED = "skipped"

SKIP_TOO_SHORT = "too_short"
SKIP_RATE_LIMITED = "rate_limited"
SKIP_BELOW_SIGNIFICANCE = "below_significance"
SKIP_DUPLICATE = "duplicate"
# T2a dead-letter: the extraction service failed this turn on every allowed
# attempt (retryable `upstream_llm`, `timeout`, or a reply that failed
# validation). Recorded as a skip so the turn never blocks the backlog head, and
# deletable by `python -m backend.extraction_backlog.admin retry-dead-letters`,
# which puts the turn back into the backlog.
SKIP_EXTRACTION_FAILED = "extraction_failed"

VALID_SKIP_REASONS = frozenset(
    {
        SKIP_TOO_SHORT,
        SKIP_RATE_LIMITED,
        SKIP_BELOW_SIGNIFICANCE,
        SKIP_DUPLICATE,
        SKIP_EXTRACTION_FAILED,
    }
)


def cache_key(event_id: str, extraction_version: str, model_hash: str) -> str:
    r"""Content-address an extraction by event plus the stamps that fed the LLM.

    `ontology_version` is deliberately ABSENT (spec D3). Stage 5 (normalize) and
    Stage 6 (validate) are the only ontology consumers in code that runs AFTER the
    cached LLM call -- both now run in REPLAYED code, so an ontology change there
    is re-derived on every rebuild rather than invalidating the cache. Verified via
    `grep -rn "ALLOWED_ENTITY_TYPES\|ALLOWED_RELATIONSHIP_TYPES" backend/ scripts/`:
    both constants are DEFINED in `ontology_extractor.py` and READ only inside
    `validator.py` -- repo-wide, not just in those two files.

    Prompt-VISIBLE ontology changes still invalidate, with no discipline
    required: `prompts.py` holds the entity list as literal text -- a THIRD
    ontology consumer, but one that runs BEFORE the cached LLM call rather than
    after -- so any edit fails the pinned sha256 in
    `tests/unit/knowledge/extraction/test_prompts.py::TestExtractionVersionDriftGuard`
    until `EXTRACTION_VERSION` is bumped -- and that IS in the key.

    `ontology_version` is still stored as a column, for audit. Do not put it back
    in the key: the key answers "what fed the LLM", the stamps on graph edges
    answer "what produced this graph". They are different questions.
    """
    raw = "|".join([event_id, extraction_version, model_hash])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ExtractionCache:
    """SQLite-backed cache of Stage-2 extraction output."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def initialize(self) -> None:
        """Create the cache table and add any missing columns. Idempotent."""
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = self._get_connection()
        conn.executescript(_DDL)
        existing = {row[1] for row in conn.execute("PRAGMA table_info(extraction_cache)")}
        for name, decl in _MIGRATION_COLUMNS:
            if name not in existing:
                conn.execute(f"ALTER TABLE extraction_cache ADD COLUMN {name} {decl}")

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path), check_same_thread=False, isolation_level=None
            )
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def get(self, event_id: str, extraction_version: str, model_hash: str) -> dict[str, Any] | None:
        r"""Return the cached extraction DECISION for the stamp pair, or None on miss.

        None means "this turn was never recorded". It is NOT the same as an
        entry whose outcome is 'skipped' with empty lists -- that one means "the
        live pipeline looked and decided not to extract". This method keeps that
        distinction on every read; the two call sites in
        `backend/knowledge/regeneration/log_regenerator.py` (`grep -n
        "self._cache.get(" backend/knowledge/regeneration/log_regenerator.py`)
        now act on it differently, and correctly so for different reasons:
        `_assert_cache_coverage` still tests `is None` only -- correctly, because
        a recorded decision (either outcome) IS coverage; a 'skipped' row is not
        a gap. The replay loop in `rebuild()` branches on `outcome` (`grep -n
        'cached\["outcome"\]' backend/knowledge/regeneration/log_regenerator.py`)
        and treats a `skipped` row as a no-op turn rather than re-deciding it --
        spec D1's target state, landed in this same branch (Task 6), not a
        pending follow-up.
        """
        key = cache_key(event_id, extraction_version, model_hash)
        row = (
            self._get_connection()
            .execute(
                "SELECT ontology_version, outcome, skip_reason, scope, "
                "scope_confidence, payload, derivation, service_stamps "
                "FROM extraction_cache WHERE cache_key = ?",
                (key,),
            )
            .fetchone()
        )
        if row is None:
            return None
        payload = json.loads(row["payload"])
        return {
            "ontology_version": row["ontology_version"],
            "outcome": row["outcome"],
            "skip_reason": row["skip_reason"],
            "scope": row["scope"],
            "scope_confidence": row["scope_confidence"],
            "entities": payload.get("entities", []),
            "relationships": payload.get("relationships", []),
            "derivation": _loads_or_none(row["derivation"]),
            "service_stamps": _loads_or_none(row["service_stamps"]),
        }

    def event_ids_for(self, extraction_version: str, model_hash: str) -> dict[str, str | None]:
        """Map every cached event_id under this stamp pair to its skip_reason.

        The backlog reads this once per scan instead of issuing one `get()` per
        logged turn. The value is the row's `skip_reason` (None for an
        'extracted' row), which is what lets a caller count dead-lettered turns
        (`SKIP_EXTRACTION_FAILED`) without a second query.

        Filtered on the two stamp COLUMNS rather than on `cache_key`: the key is
        a hash of `event_id|extraction_version|model_hash` (`cache_key()`), so
        both select exactly the rows `get()` would hit for the same stamps.

        Args:
            extraction_version: The epoch's extraction_version stamp.
            model_hash: The epoch's (composed) model_hash stamp.

        Returns:
            A dict of event_id -> skip_reason (None when outcome='extracted').
        """
        rows = (
            self._get_connection()
            .execute(
                "SELECT event_id, skip_reason FROM extraction_cache "
                "WHERE extraction_version = ? AND model_hash = ?",
                (extraction_version, model_hash),
            )
            .fetchall()
        )
        return {row["event_id"]: row["skip_reason"] for row in rows}

    def delete(self, event_id: str, extraction_version: str, model_hash: str) -> bool:
        """Delete the cached decision for one turn under one stamp pair.

        Exists for exactly one caller: `retry-dead-letters`
        (`backend/extraction_backlog/admin.py`), which removes an
        `extraction_failed` row so the turn re-enters the backlog as
        inference-pending. Deleting any other row makes that turn look never
        recorded, so a rebuild raises `ColdCacheError` on it until it is
        re-extracted -- which is the intended effect of a retry and nothing
        else.

        Returns:
            True when a row was deleted, False when there was none.
        """
        key = cache_key(event_id, extraction_version, model_hash)
        cursor = self._get_connection().execute(
            "DELETE FROM extraction_cache WHERE cache_key = ?", (key,)
        )
        return cursor.rowcount > 0

    def put(
        self,
        event_id: str,
        ontology_version: str,
        extraction_version: str,
        model_hash: str,
        *,
        outcome: str,
        created_at: str,
        entities: list[dict[str, Any]] | None = None,
        relationships: list[dict[str, Any]] | None = None,
        skip_reason: str | None = None,
        scope: str | None = None,
        scope_confidence: float | None = None,
        derivation: dict[str, Any] | None = None,
        service_stamps: dict[str, Any] | None = None,
    ) -> None:
        """Record what the live pipeline DECIDED for this turn.

        Fails closed on an inconsistent decision rather than storing it. A row
        that says 'skipped' with no reason, or 'extracted' with one, is a
        decision nobody can replay -- and a rebuild reading it would produce a
        graph whose provenance is unexplainable.

        `derivation` and `service_stamps` are optional JSON objects written by
        the extraction-backlog dispatcher (see `_MIGRATION_COLUMNS`); None
        stores SQL NULL, which is what every in-process caller writes.
        """
        if outcome not in (OUTCOME_EXTRACTED, OUTCOME_SKIPPED):
            raise ValueError(f"unknown outcome {outcome!r}")
        if outcome == OUTCOME_SKIPPED:
            if skip_reason is None:
                raise ValueError("outcome='skipped' requires a skip_reason")
            if skip_reason not in VALID_SKIP_REASONS:
                raise ValueError(f"unknown skip_reason {skip_reason!r}")
        elif skip_reason is not None:
            raise ValueError("outcome='extracted' must not carry a skip_reason")

        key = cache_key(event_id, extraction_version, model_hash)
        payload = json.dumps(
            {"entities": entities or [], "relationships": relationships or []},
            sort_keys=True,
        )
        self._get_connection().execute(
            "INSERT OR REPLACE INTO extraction_cache "
            "(cache_key, event_id, ontology_version, extraction_version, model_hash, "
            "outcome, skip_reason, scope, scope_confidence, payload, created_at, "
            "derivation, service_stamps) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                key,
                event_id,
                ontology_version,
                extraction_version,
                model_hash,
                outcome,
                skip_reason,
                scope,
                scope_confidence,
                payload,
                created_at,
                None if derivation is None else json.dumps(derivation, sort_keys=True),
                None if service_stamps is None else json.dumps(service_stamps, sort_keys=True),
            ),
        )

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None


def _loads_or_none(value: str | None) -> Any:
    """Decode a nullable JSON column; SQL NULL stays None."""
    return None if value is None else json.loads(value)
