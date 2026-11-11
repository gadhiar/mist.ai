"""Reading a cache file that has no `batch_record` and no `turn_plan` column.

The rebuild CLI opens the cache without calling `initialize()`: it is read-only on
its sources, and its gate requires `identity_namespace` only. So a file that no
backend of this code has opened (a restored backup, a file written by an older
build) reaches `ExtractionCache.get` without the two rail-input columns. Such a file
holds only rows written before the columns existed, and a row without them reads as
a turn extracted as one batch with no plan.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from backend.knowledge.extraction_cache import (
    OUTCOME_EXTRACTED,
    ExtractionCache,
    cache_key,
)

# The table as `initialize()` created it before the two columns were added
# (`git show bde25f7:backend/knowledge/extraction_cache.py`, `_DDL`). Kept as text:
# the point is a file this code's `initialize()` never touched.
_DDL_BEFORE_RAIL_COLUMNS = """
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
    identity_namespace TEXT
);
"""

ONTOLOGY_VERSION = "1.4.0"
EXTRACTION_VERSION = "2026-06-14-r5"
MODEL_HASH = "model-abc"
OLD_EVENT = "evt-old"
NEW_EVENT = "evt-new"

ENTITIES = [{"id": "ada", "type": "Person"}]
RELATIONSHIPS = [{"source": "ada", "target": "platform", "type": "MEMBER_OF"}]
DERIVATION = {"operations": [{"op": "upsert", "id": "ada"}]}
SERVICE_STAMPS = {"location_label": "host-a", "stamps": {"run_identity": "run-1"}}

BATCH_RECORD = [
    {
        "units": [{"text": "Bo is on Platform", "context": [], "kind": "bullet"}],
        "context": [],
        "payload": {"entities": [{"id": "bo"}], "relationships": []},
        "prompt_tokens": 90,
        "completion_tokens": 20,
        "finish": "stop",
    }
]
TURN_PLAN = {
    "payload": {"templates": [{"anchor": "platform"}]},
    "prompt_tokens": 80,
    "completion_tokens": 12,
    "finish_reason": "stop",
}

# What `get` returned for the old row before the columns existed, plus the two keys.
OLD_ROW_AS_READ = {
    "ontology_version": ONTOLOGY_VERSION,
    "outcome": OUTCOME_EXTRACTED,
    "skip_reason": None,
    "scope": "work",
    "scope_confidence": 0.75,
    "entities": ENTITIES,
    "relationships": RELATIONSHIPS,
    "derivation": DERIVATION,
    "service_stamps": SERVICE_STAMPS,
    "batch_record": None,
    "turn_plan": None,
}


def _columns(db_path: str) -> list[str]:
    conn = sqlite3.connect(db_path)
    try:
        return [row[1] for row in conn.execute("PRAGMA table_info(extraction_cache)")]
    finally:
        conn.close()


def _schema_objects(db_path: str) -> list[tuple[str, str]]:
    conn = sqlite3.connect(db_path)
    try:
        return sorted(conn.execute("SELECT type, name FROM sqlite_master"))
    finally:
        conn.close()


@pytest.fixture
def old_file(tmp_path) -> str:
    """A cache file with one extracted row, built without this code's `initialize()`."""
    db_path = str(tmp_path / "extraction_cache.db")
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_DDL_BEFORE_RAIL_COLUMNS)
        conn.execute(
            "INSERT INTO extraction_cache "
            "(cache_key, event_id, ontology_version, extraction_version, model_hash, "
            "outcome, skip_reason, scope, scope_confidence, payload, created_at, "
            "derivation, service_stamps, identity_namespace) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                cache_key(OLD_EVENT, EXTRACTION_VERSION, MODEL_HASH),
                OLD_EVENT,
                ONTOLOGY_VERSION,
                EXTRACTION_VERSION,
                MODEL_HASH,
                OUTCOME_EXTRACTED,
                None,
                "work",
                0.75,
                json.dumps({"entities": ENTITIES, "relationships": RELATIONSHIPS}),
                "2026-06-14T10:00:00+00:00",
                json.dumps(DERIVATION),
                json.dumps(SERVICE_STAMPS),
                None,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


@pytest.fixture
def cache(old_file):
    """An `ExtractionCache` over the old file, opened the way the rebuild CLI opens it."""
    cache = ExtractionCache(old_file)
    yield cache
    cache.close()


class TestReadingAFileWithoutTheColumns:
    def test_the_fixture_file_lacks_both_columns(self, old_file):
        columns = _columns(old_file)

        assert "identity_namespace" in columns
        assert "batch_record" not in columns
        assert "turn_plan" not in columns

    def test_the_row_reads_as_one_batch_with_no_plan(self, cache):
        assert cache.get(OLD_EVENT, EXTRACTION_VERSION, MODEL_HASH) == OLD_ROW_AS_READ

    def test_a_turn_the_file_does_not_hold_is_still_a_miss(self, cache):
        assert cache.get("evt-absent", EXTRACTION_VERSION, MODEL_HASH) is None

    def test_the_read_does_not_change_the_file(self, cache, old_file):
        columns_before = _columns(old_file)
        objects_before = _schema_objects(old_file)

        cache.get(OLD_EVENT, EXTRACTION_VERSION, MODEL_HASH)
        cache.get("evt-absent", EXTRACTION_VERSION, MODEL_HASH)

        assert _columns(old_file) == columns_before
        assert _schema_objects(old_file) == objects_before


class TestTheSameObjectAfterInitialize:
    """`get` may not remember that the file lacked the columns."""

    def test_initialize_adds_the_columns(self, cache, old_file):
        cache.get(OLD_EVENT, EXTRACTION_VERSION, MODEL_HASH)

        cache.initialize()

        columns = _columns(old_file)
        assert "batch_record" in columns
        assert "turn_plan" in columns

    def test_the_old_row_still_reads_as_one_batch_with_no_plan(self, cache):
        cache.get(OLD_EVENT, EXTRACTION_VERSION, MODEL_HASH)

        cache.initialize()

        assert cache.get(OLD_EVENT, EXTRACTION_VERSION, MODEL_HASH) == OLD_ROW_AS_READ

    def test_a_row_written_after_it_reads_back_what_was_stored(self, cache):
        cache.get(OLD_EVENT, EXTRACTION_VERSION, MODEL_HASH)

        cache.initialize()
        cache.put(
            NEW_EVENT,
            ONTOLOGY_VERSION,
            EXTRACTION_VERSION,
            MODEL_HASH,
            outcome=OUTCOME_EXTRACTED,
            created_at="2026-06-15T10:00:00+00:00",
            entities=ENTITIES,
            relationships=RELATIONSHIPS,
            batch_record=BATCH_RECORD,
            turn_plan=TURN_PLAN,
        )

        new_row = cache.get(NEW_EVENT, EXTRACTION_VERSION, MODEL_HASH)
        assert new_row is not None
        assert new_row["batch_record"] == BATCH_RECORD
        assert new_row["turn_plan"] == TURN_PLAN
        assert cache.get(OLD_EVENT, EXTRACTION_VERSION, MODEL_HASH) == OLD_ROW_AS_READ


class TestWritingStillNeedsAnInitializedFile:
    def test_put_on_a_file_without_the_columns_is_refused_by_sqlite(self, cache, old_file):
        with pytest.raises(sqlite3.OperationalError):
            cache.put(
                NEW_EVENT,
                ONTOLOGY_VERSION,
                EXTRACTION_VERSION,
                MODEL_HASH,
                outcome=OUTCOME_EXTRACTED,
                created_at="2026-06-15T10:00:00+00:00",
                entities=ENTITIES,
                relationships=RELATIONSHIPS,
            )

        assert "batch_record" not in _columns(old_file)
        assert cache.get(NEW_EVENT, EXTRACTION_VERSION, MODEL_HASH) is None
