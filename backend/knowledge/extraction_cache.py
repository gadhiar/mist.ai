"""Content-addressed extraction cache (F3).

Caches the Stage-2 extraction output (entities + relationships) keyed by
(event_id, ontology_version, extraction_version, model_hash). A rebuild reuses
the cached output instead of re-running the (non-bit-stable) LLM; the LLM runs
only on a stamp miss. This is the determinism boundary for Inv-A4.

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
    payload TEXT NOT NULL,
    created_at TEXT
);
"""


def cache_key(
    event_id: str, ontology_version: str, extraction_version: str, model_hash: str
) -> str:
    """Stable content-address for an extraction by event + epoch stamp triple."""
    raw = "|".join([event_id, ontology_version, extraction_version, model_hash])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ExtractionCache:
    """SQLite-backed cache of Stage-2 extraction output."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def initialize(self) -> None:
        """Create the cache table. Idempotent."""
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._get_connection().executescript(_DDL)

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path), check_same_thread=False, isolation_level=None
            )
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def get(
        self, event_id: str, ontology_version: str, extraction_version: str, model_hash: str
    ) -> dict[str, Any] | None:
        """Return the cached extraction for the stamp triple, or None on miss."""
        key = cache_key(event_id, ontology_version, extraction_version, model_hash)
        row = (
            self._get_connection()
            .execute("SELECT payload FROM extraction_cache WHERE cache_key = ?", (key,))
            .fetchone()
        )
        return json.loads(row["payload"]) if row else None

    def put(
        self,
        event_id: str,
        ontology_version: str,
        extraction_version: str,
        model_hash: str,
        entities: list[dict[str, Any]],
        relationships: list[dict[str, Any]],
        created_at: str,
    ) -> None:
        """Cache (or overwrite) the extraction output for the stamp triple."""
        key = cache_key(event_id, ontology_version, extraction_version, model_hash)
        payload = json.dumps({"entities": entities, "relationships": relationships}, sort_keys=True)
        self._get_connection().execute(
            "INSERT OR REPLACE INTO extraction_cache "
            "(cache_key, event_id, ontology_version, extraction_version, model_hash, "
            "payload, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (key, event_id, ontology_version, extraction_version, model_hash, payload, created_at),
        )

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
