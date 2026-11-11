"""The extraction cache's rail inputs: a turn's batch record and its plan.

The rails read the cached row and the pack, and nothing else, so what a rail may read
beyond the payload is stored beside it: two nullable JSON columns, outside the cache
key. A row written before the columns existed holds NULL in both, and is read as a
turn extracted as one batch with no plan.
"""

from __future__ import annotations

import hashlib
import sqlite3

from backend.knowledge.extraction.normalizer import rail_inputs
from backend.knowledge.extraction_cache import (
    OUTCOME_EXTRACTED,
    ExtractionCache,
    cache_key,
)

EXTRACTION_VERSION = "2026-06-14-r5"
MODEL_HASH = "model-abc"

BATCH_RECORD = [
    {
        "units": [{"text": "Ada is on Platform", "context": ["# Teams"], "kind": "bullet"}],
        "context": [],
        "payload": {"entities": [{"id": "ada"}], "relationships": []},
        "prompt_tokens": 120,
        "completion_tokens": 30,
        "finish": "stop",
    },
    {
        "units": [{"text": "Bo is on Platform", "context": ["# Teams"], "kind": "bullet"}],
        "context": ["Ada is on Platform"],
        "payload": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "finish": None,
    },
]
TURN_PLAN = {
    "payload": {"templates": [{"anchor": "platform"}]},
    "prompt_tokens": 80,
    "completion_tokens": 12,
    "finish_reason": "stop",
}


def _cache() -> ExtractionCache:
    cache = ExtractionCache(":memory:")
    cache.initialize()
    return cache


def _put_extracted(cache: ExtractionCache, event_id: str, **rail_columns) -> None:
    cache.put(
        event_id,
        "1.4.0",
        EXTRACTION_VERSION,
        MODEL_HASH,
        outcome=OUTCOME_EXTRACTED,
        entities=[{"id": "ada"}],
        relationships=[],
        created_at="2026-08-18T00:00:00+00:00",
        **rail_columns,
    )


def _create_table_without_the_rail_columns(db_path) -> None:
    """The table as it stood before `batch_record` and `turn_plan`, holding one row."""
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE extraction_cache (
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
    )
    conn.execute(
        "INSERT INTO extraction_cache (cache_key, event_id, ontology_version, "
        "extraction_version, model_hash, outcome, payload, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            cache_key("evt-old", EXTRACTION_VERSION, MODEL_HASH),
            "evt-old",
            "1.4.0",
            EXTRACTION_VERSION,
            MODEL_HASH,
            OUTCOME_EXTRACTED,
            '{"entities": [{"id": "ada"}], "relationships": []}',
            "2026-08-18T00:00:00+00:00",
        ),
    )
    conn.commit()
    conn.close()


class TestRoundTrip:
    def test_the_batch_record_and_the_plan_read_back_as_stored(self):
        cache = _cache()
        _put_extracted(cache, "evt-b", batch_record=BATCH_RECORD, turn_plan=TURN_PLAN)

        row = cache.get("evt-b", EXTRACTION_VERSION, MODEL_HASH)

        assert (row["batch_record"], row["turn_plan"]) == (BATCH_RECORD, TURN_PLAN)

    def test_a_row_put_without_them_holds_none_under_both_keys(self):
        """Both keys are present and None, so a reader can index them on any row."""
        cache = _cache()
        _put_extracted(cache, "evt-plain")

        row = cache.get("evt-plain", EXTRACTION_VERSION, MODEL_HASH)

        assert ("batch_record" in row, "turn_plan" in row) == (True, True)
        assert (row["batch_record"], row["turn_plan"]) == (None, None)

    def test_one_may_be_stored_without_the_other(self):
        """A turn extracted in batches with the plan stage off has a record and no plan."""
        cache = _cache()
        _put_extracted(cache, "evt-r", batch_record=BATCH_RECORD)

        row = cache.get("evt-r", EXTRACTION_VERSION, MODEL_HASH)

        assert (row["batch_record"], row["turn_plan"]) == (BATCH_RECORD, None)


class TestWhatTheRailsAreHanded:
    def test_a_row_written_before_the_columns_reads_as_one_batch_with_no_plan(self, tmp_path):
        """T45 Acceptance 2: null columns give the rails nothing.

        `rail_inputs` is what both replay paths pass to `normalize`; an empty dict is
        the call they made before the columns existed.
        """
        db = tmp_path / "before-rail-columns.db"
        _create_table_without_the_rail_columns(db)
        cache = ExtractionCache(str(db))
        cache.initialize()

        row = cache.get("evt-old", EXTRACTION_VERSION, MODEL_HASH)

        assert (row["batch_record"], row["turn_plan"]) == (None, None)
        assert row["entities"] == [{"id": "ada"}]
        assert rail_inputs(row) == {}

    def test_a_row_that_holds_them_hands_both_to_the_rails(self):
        cache = _cache()
        _put_extracted(cache, "evt-b", batch_record=BATCH_RECORD, turn_plan=TURN_PLAN)

        row = cache.get("evt-b", EXTRACTION_VERSION, MODEL_HASH)

        assert rail_inputs(row) == {"batch_record": BATCH_RECORD, "turn_plan": TURN_PLAN}


class TestSchema:
    def test_migration_adds_both_columns_as_nullable_text(self, tmp_path):
        db = tmp_path / "before-rail-columns.db"
        _create_table_without_the_rail_columns(db)
        cache = ExtractionCache(str(db))
        cache.initialize()
        cache.initialize()  # idempotent

        columns = {
            row[1]: (row[2], row[3])
            for row in cache._get_connection().execute("PRAGMA table_info(extraction_cache)")
        }

        assert columns["batch_record"] == ("TEXT", 0)
        assert columns["turn_plan"] == ("TEXT", 0)

    def test_a_fresh_table_declares_them_as_a_migrated_one_does(self, tmp_path):
        fresh = _cache()
        db = tmp_path / "before-rail-columns.db"
        _create_table_without_the_rail_columns(db)
        migrated = ExtractionCache(str(db))
        migrated.initialize()

        def declared(cache: ExtractionCache) -> set[tuple]:
            return {
                (row[1], row[2], row[3], row[4])
                for row in cache._get_connection().execute("PRAGMA table_info(extraction_cache)")
                if row[1] in ("batch_record", "turn_plan")
            }

        assert declared(fresh) == declared(migrated) != set()

    def test_neither_column_is_part_of_the_key(self):
        """Two puts that differ only in the record and the plan are one row."""
        cache = _cache()
        _put_extracted(cache, "evt-k")
        _put_extracted(cache, "evt-k", batch_record=BATCH_RECORD, turn_plan=TURN_PLAN)

        count = cache._get_connection().execute("SELECT COUNT(*) FROM extraction_cache").fetchone()

        assert count[0] == 1
        assert (
            cache_key("evt-k", EXTRACTION_VERSION, MODEL_HASH)
            == hashlib.sha256(b"evt-k|2026-06-14-r5|model-abc").hexdigest()
        )
