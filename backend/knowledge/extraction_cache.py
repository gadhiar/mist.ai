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
    created_at TEXT
);
"""

# SQLite has no ADD COLUMN IF NOT EXISTS. `initialize()` reads PRAGMA
# table_info and adds only what is missing, mirroring the conditional
# ALTER TABLE the event store already uses.
_MIGRATION_COLUMNS: tuple[tuple[str, str], ...] = (
    ("outcome", "TEXT NOT NULL DEFAULT 'extracted'"),
    ("skip_reason", "TEXT"),
    ("scope", "TEXT"),
    ("scope_confidence", "REAL"),
)

OUTCOME_EXTRACTED = "extracted"
OUTCOME_SKIPPED = "skipped"

SKIP_TOO_SHORT = "too_short"
SKIP_RATE_LIMITED = "rate_limited"
SKIP_BELOW_SIGNIFICANCE = "below_significance"
SKIP_DUPLICATE = "duplicate"

VALID_SKIP_REASONS = frozenset(
    {SKIP_TOO_SHORT, SKIP_RATE_LIMITED, SKIP_BELOW_SIGNIFICANCE, SKIP_DUPLICATE}
)


def cache_key(event_id: str, extraction_version: str, model_hash: str) -> str:
    """Content-address an extraction by event plus the stamps that fed the LLM.

    `ontology_version` is deliberately ABSENT (spec D3). Stage 5 (normalize) and
    Stage 6 (validate) are the only ontology consumers, and both now run in
    REPLAYED code -- so an ontology change is re-derived on every rebuild rather
    than invalidating the cache.

    Prompt-VISIBLE ontology changes still invalidate, with no discipline
    required: `prompts.py` holds the entity list as literal text, so any edit
    fails the pinned sha256 in
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
        """Return the cached extraction DECISION for the stamp pair, or None on miss.

        None means "this turn was never recorded". It is NOT the same as an
        entry whose outcome is 'skipped' with empty lists -- that one means "the
        live pipeline looked and decided not to extract". Keeping those distinct
        end-to-end is the whole point of spec D1.
        """
        key = cache_key(event_id, extraction_version, model_hash)
        row = (
            self._get_connection()
            .execute(
                "SELECT ontology_version, outcome, skip_reason, scope, "
                "scope_confidence, payload FROM extraction_cache WHERE cache_key = ?",
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
        }

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
    ) -> None:
        """Record what the live pipeline DECIDED for this turn.

        Fails closed on an inconsistent decision rather than storing it. A row
        that says 'skipped' with no reason, or 'extracted' with one, is a
        decision nobody can replay -- and a rebuild reading it would produce a
        graph whose provenance is unexplainable.
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
            "outcome, skip_reason, scope, scope_confidence, payload, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
            ),
        )

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
