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
from dataclasses import dataclass
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
    service_stamps TEXT,
    identity_namespace TEXT,
    batch_record TEXT,
    turn_plan TEXT
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
#
# `identity_namespace` (mist-extraction-epoch-identity) IS part of the key when
# set (`cache_key()`). NULL is the 3-part namespace: every row written before
# the column existed, and every row an unbound or 3-part-bound epoch writes.
#
# `batch_record` and `turn_plan` are nullable JSON columns and are NOT part of
# `cache_key()`. They are what a rail may read beside the payload and the scope
# (`backend/knowledge/extraction/normalizer.py::rail_inputs`): one entry per
# Stage 2 call the turn took, and the record of its plan call. NULL in either is
# a turn extracted as one batch, or with no plan: every row written before the
# columns existed, every row the in-process pipeline writes, and every skip row.
_MIGRATION_COLUMNS: tuple[tuple[str, str], ...] = (
    ("outcome", "TEXT NOT NULL DEFAULT 'extracted'"),
    ("skip_reason", "TEXT"),
    ("scope", "TEXT"),
    ("scope_confidence", "REAL"),
    ("derivation", "TEXT"),
    ("service_stamps", "TEXT"),
    ("identity_namespace", "TEXT"),
    ("batch_record", "TEXT"),
    ("turn_plan", "TEXT"),
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


def keyed_namespace(serving_identity: str, dispatch_identity: str) -> str:
    """The cache namespace of an epoch bound outside the 3-part key."""
    return f"{serving_identity}+{dispatch_identity}"


@dataclass(frozen=True, slots=True)
class ServiceRecord:
    """One extracted cache row that carries the dispatcher's `service_stamps`."""

    event_id: str
    service_stamps: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AdoptionRecords:
    """What the 3-part rows of one (extraction_version, model_hash) pair record.

    The input to binding an unbound epoch (Migration, rule 5): whether there
    are rows at all, and what the newest service-written one says about the
    configuration that wrote it.

    Attributes:
        rows: Rows of any outcome (extracted, gate-skipped, dead-lettered).
        extracted: Rows with outcome 'extracted'.
        extracted_without_service_stamps: Extracted rows with NULL
            `service_stamps`, which the in-process pipeline writes. Skip rows
            never carry stamps, so they are not counted here.
        run_identities: The distinct `stamps.run_identity` values among
            extracted rows with stamps, sorted, None (a pre-1.2.0 service)
            first.
        location_labels: The distinct `location_label` values among the same
            rows, sorted the same way.
        newest: The extracted row with stamps written last, by rowid: `put` is
            `INSERT OR REPLACE`, which deletes and reinserts, so the latest
            write holds the largest rowid. `created_at` is the turn's own
            timestamp, not the time of extraction, so it cannot order writes.
    """

    rows: int
    extracted: int
    extracted_without_service_stamps: int
    run_identities: tuple[str | None, ...]
    location_labels: tuple[str | None, ...]
    newest: ServiceRecord | None


def cache_key(
    event_id: str,
    extraction_version: str,
    model_hash: str,
    identity_namespace: str | None = None,
) -> str:
    r"""Content-address an extraction by event plus the stamps that fed the LLM.

    With no `identity_namespace` the hashed string is
    `event_id|extraction_version|model_hash`, byte for byte the key every row
    written before namespaces existed carries. A keyed namespace
    (`keyed_namespace()`) is appended as a fourth part, so an epoch whose
    serving or dispatch settings differ from the 3-part namespace's owner
    gets rows of its own under the same triple.

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
    parts = [event_id, extraction_version, model_hash]
    if identity_namespace is not None:
        parts.append(identity_namespace)
    raw = "|".join(parts)
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

    def get(
        self,
        event_id: str,
        extraction_version: str,
        model_hash: str,
        *,
        identity_namespace: str | None = None,
    ) -> dict[str, Any] | None:
        r"""Return the cached extraction DECISION for the stamp pair, or None on miss.

        Reads only `identity_namespace` (None: the 3-part namespace); a row of
        the same triple in another namespace is a miss.

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

        `batch_record` and `turn_plan` are always keys of the returned dict.
        Each holds what `put` was given, or None when the column is NULL: a
        turn extracted as one batch, or with no plan.

        A file may lack those two columns altogether: the rebuild CLI reads
        the cache without calling `initialize()`, so a file no backend of
        this code has opened (a restored backup) still has the older table.
        Every row in it was written before the columns existed, so both keys
        read None, the same as a NULL. The row is therefore selected whole
        and the two columns are looked up among the ones it came back with,
        on every call: nothing is remembered, so a later `initialize()` on
        this object is seen by the next read. The read adds no column.
        """
        key = cache_key(event_id, extraction_version, model_hash, identity_namespace)
        row = (
            self._get_connection()
            .execute(
                "SELECT * FROM extraction_cache WHERE cache_key = ? AND identity_namespace IS ?",
                (key, identity_namespace),
            )
            .fetchone()
        )
        if row is None:
            return None
        columns = row.keys()
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
            "batch_record": (
                _loads_or_none(row["batch_record"]) if "batch_record" in columns else None
            ),
            "turn_plan": _loads_or_none(row["turn_plan"]) if "turn_plan" in columns else None,
        }

    def event_ids_for(
        self,
        extraction_version: str,
        model_hash: str,
        *,
        identity_namespace: str | None = None,
    ) -> dict[str, str | None]:
        """Map every cached event_id under this stamp pair to its skip_reason.

        The backlog reads this once per scan instead of issuing one `get()` per
        logged turn. The value is the row's `skip_reason` (None for an
        'extracted' row), which is what lets a caller count dead-lettered turns
        (`SKIP_EXTRACTION_FAILED`) without a second query.

        Filtered on the two stamp COLUMNS and the namespace column rather than
        on `cache_key`: the key is a hash of the same values (`cache_key()`),
        so both select exactly the rows `get()` would hit for the same stamps.
        A None namespace filters `identity_namespace IS NULL`, so a 3-part
        scan never sees a keyed row of the same pair, and a keyed scan never
        sees a 3-part one.

        Args:
            extraction_version: The epoch's extraction_version stamp.
            model_hash: The epoch's (composed) model_hash stamp.
            identity_namespace: The epoch's namespace; None is the 3-part one.

        Returns:
            A dict of event_id -> skip_reason (None when outcome='extracted').
        """
        rows = (
            self._get_connection()
            .execute(
                "SELECT event_id, skip_reason FROM extraction_cache "
                "WHERE extraction_version = ? AND model_hash = ? AND identity_namespace IS ?",
                (extraction_version, model_hash, identity_namespace),
            )
            .fetchall()
        )
        return {row["event_id"]: row["skip_reason"] for row in rows}

    def adoption_records(self, extraction_version: str, model_hash: str) -> AdoptionRecords:
        """Summarize the 3-part rows of a pair, for binding an unbound epoch.

        Only the 3-part namespace is read: it is the one an unbound epoch has
        been writing, and the one a binding can claim. See `AdoptionRecords`.
        """
        conn = self._get_connection()
        where = "extraction_version = ? AND model_hash = ? AND identity_namespace IS NULL"
        pair = (extraction_version, model_hash)
        counts = conn.execute(
            "SELECT COUNT(*), "
            "COALESCE(SUM(outcome = ?), 0), "
            "COALESCE(SUM(outcome = ? AND service_stamps IS NULL), 0) "
            f"FROM extraction_cache WHERE {where}",  # nosec B608 -- fixed text
            (OUTCOME_EXTRACTED, OUTCOME_EXTRACTED, *pair),
        ).fetchone()
        stamped = conn.execute(
            "SELECT event_id, service_stamps FROM extraction_cache "
            f"WHERE {where} AND outcome = ? AND service_stamps IS NOT NULL "  # nosec B608
            "ORDER BY rowid",
            (*pair, OUTCOME_EXTRACTED),
        )
        run_identities: set[str | None] = set()
        location_labels: set[str | None] = set()
        newest: ServiceRecord | None = None
        for row in stamped:
            stamps = json.loads(row["service_stamps"])
            run_identities.add((stamps.get("stamps") or {}).get("run_identity"))
            location_labels.add(stamps.get("location_label"))
            newest = ServiceRecord(event_id=row["event_id"], service_stamps=stamps)
        return AdoptionRecords(
            rows=int(counts[0]),
            extracted=int(counts[1]),
            extracted_without_service_stamps=int(counts[2]),
            run_identities=_sorted_with_none(run_identities),
            location_labels=_sorted_with_none(location_labels),
            newest=newest,
        )

    def delete(
        self,
        event_id: str,
        extraction_version: str,
        model_hash: str,
        *,
        only_skip_reason: str | None = None,
        identity_namespace: str | None = None,
    ) -> bool:
        """Delete the cached decision for one turn under one stamp pair.

        Exists for exactly one caller: `retry-dead-letters`
        (`backend/extraction_backlog/admin.py`), which removes an
        `extraction_failed` row so the turn re-enters the backlog as
        inference-pending. Deleting any other row makes that turn look never
        recorded, so a rebuild raises `ColdCacheError` on it until it is
        re-extracted -- which is the intended effect of a retry and nothing
        else.

        Args:
            event_id: The turn whose row to delete.
            extraction_version: Stamp half of the cache key.
            model_hash: Stamp half of the cache key.
            only_skip_reason: When given, the row is deleted only if its
                `skip_reason` still equals this value, checked in the same
                `DELETE` statement. The retry passes `extraction_failed`, so a
                real extraction that replaced the dead letter between the
                caller's read and this delete (a concurrent retry freed the
                turn and the dispatcher re-extracted it) is left alone.
            identity_namespace: The namespace whose row to delete; None is the
                3-part one. A row of the same triple in another namespace is
                never touched.

        Returns:
            True when a row was deleted, False when there was none (or it no
            longer matched `only_skip_reason`).
        """
        key = cache_key(event_id, extraction_version, model_hash, identity_namespace)
        if only_skip_reason is None:
            cursor = self._get_connection().execute(
                "DELETE FROM extraction_cache WHERE cache_key = ? AND identity_namespace IS ?",
                (key, identity_namespace),
            )
        else:
            cursor = self._get_connection().execute(
                "DELETE FROM extraction_cache WHERE cache_key = ? AND identity_namespace IS ? "
                "AND skip_reason = ?",
                (key, identity_namespace, only_skip_reason),
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
        identity_namespace: str | None = None,
        batch_record: list[dict[str, Any]] | None = None,
        turn_plan: dict[str, Any] | None = None,
    ) -> None:
        """Record what the live pipeline DECIDED for this turn.

        Fails closed on an inconsistent decision rather than storing it. A row
        that says 'skipped' with no reason, or 'extracted' with one, is a
        decision nobody can replay -- and a rebuild reading it would produce a
        graph whose provenance is unexplainable.

        `derivation` and `service_stamps` are optional JSON objects written by
        the extraction-backlog dispatcher (see `_MIGRATION_COLUMNS`); None
        stores SQL NULL, which is what every in-process caller writes.
        `identity_namespace` is the writing epoch's namespace, stored on the
        row and folded into its key; None (the 3-part namespace) is what every
        caller wrote before namespaces existed.

        `batch_record` and `turn_plan` are the rail inputs the extraction
        service returned with the payload (see `_MIGRATION_COLUMNS`), stored as
        JSON and read back by `get` under the same names. None stores SQL NULL.
        Neither is part of the key.
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

        key = cache_key(event_id, extraction_version, model_hash, identity_namespace)
        payload = json.dumps(
            {"entities": entities or [], "relationships": relationships or []},
            sort_keys=True,
        )
        self._get_connection().execute(
            "INSERT OR REPLACE INTO extraction_cache "
            "(cache_key, event_id, ontology_version, extraction_version, model_hash, "
            "outcome, skip_reason, scope, scope_confidence, payload, created_at, "
            "derivation, service_stamps, identity_namespace, batch_record, turn_plan) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                identity_namespace,
                None if batch_record is None else json.dumps(batch_record, sort_keys=True),
                None if turn_plan is None else json.dumps(turn_plan, sort_keys=True),
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


def _sorted_with_none(values: set[str | None]) -> tuple[str | None, ...]:
    """Sorted, with None (a value the row did not record) first."""
    return tuple(sorted(values, key=lambda v: (v is not None, v or "")))
