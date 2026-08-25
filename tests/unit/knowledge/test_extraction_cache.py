"""Content-addressed extraction cache (F3)."""

import pytest

from backend.knowledge.extraction_cache import (
    OUTCOME_EXTRACTED,
    OUTCOME_SKIPPED,
    SKIP_RATE_LIMITED,
    ExtractionCache,
    cache_key,
)


def test_cache_key_is_deterministic():
    """Same inputs, same key -- a prerequisite for content-addressing, not D3 itself.

    Mutant this kills: seeding the hash with anything non-deterministic (e.g. a
    timestamp, `id()`, or `random`) would make two calls with identical arguments
    diverge.
    """
    a = cache_key("evt-1", "2026-06-14-r5", "model-abc")
    b = cache_key("evt-1", "2026-06-14-r5", "model-abc")
    assert a == b


def test_cache_key_changes_with_extraction_version_and_model():
    base = cache_key("evt-1", "2026-06-14-r5", "model-abc")
    assert cache_key("evt-1", "2026-08-18-r6", "model-abc") != base
    assert cache_key("evt-1", "2026-06-14-r5", "model-xyz") != base
    assert cache_key("evt-2", "2026-06-14-r5", "model-abc") != base


def test_ontology_bump_alone_collides_with_the_prior_entry_and_last_write_wins():
    """D3: the key excludes ontology_version, so bumping it alone is NOT a fresh
    key -- the second `put` overwrites the first row rather than adding one.

    Mutant this kills: folding `ontology_version` back into `cache_key()` (the
    regression D3 exists to prevent) would give the second `put` a DIFFERENT
    key, leaving two rows instead of one, and `hit["ontology_version"]` would
    still read the FIRST value instead of the second. Both are asserted below,
    so either mutant fails this test.
    """
    cache = ExtractionCache(":memory:")
    cache.initialize()
    cache.put(
        "evt-1",
        "1.4.0",
        "2026-06-14-r5",
        "model-abc",
        outcome=OUTCOME_EXTRACTED,
        entities=[{"id": "user", "name": "User", "type": "User"}],
        relationships=[],
        created_at="2026-08-18T00:00:00+00:00",
    )
    # Same event, same extraction/model stamps -- ONLY the ontology moved.
    cache.put(
        "evt-1",
        "1.5.0",
        "2026-06-14-r5",
        "model-abc",
        outcome=OUTCOME_EXTRACTED,
        entities=[{"id": "user", "name": "User", "type": "User", "role": "admin"}],
        relationships=[],
        created_at="2026-08-18T00:01:00+00:00",
    )

    row_count = (
        cache._get_connection()
        .execute("SELECT COUNT(*) AS c FROM extraction_cache")
        .fetchone()["c"]
    )
    assert row_count == 1  # one key, one row -- the two "epochs" collided.

    hit = cache.get("evt-1", "2026-06-14-r5", "model-abc")
    assert hit is not None
    assert hit["ontology_version"] == "1.5.0"  # last write wins, including on audit columns
    assert hit["entities"] == [{"id": "user", "name": "User", "type": "User", "role": "admin"}]


def test_skipped_turn_round_trips_with_its_reason():
    """D1: 'empty' and 'absent' must never render identically."""
    cache = ExtractionCache(":memory:")
    cache.initialize()
    cache.put(
        "evt-2",
        "1.4.0",
        "2026-06-14-r5",
        "model-abc",
        outcome=OUTCOME_SKIPPED,
        skip_reason=SKIP_RATE_LIMITED,
        created_at="2026-08-18T00:00:00+00:00",
    )
    hit = cache.get("evt-2", "2026-06-14-r5", "model-abc")
    assert hit is not None
    assert hit["outcome"] == OUTCOME_SKIPPED
    assert hit["skip_reason"] == SKIP_RATE_LIMITED
    assert hit["entities"] == []
    assert hit["relationships"] == []
    # A turn that was never cached is a different observation entirely.
    assert cache.get("evt-never-seen", "2026-06-14-r5", "model-abc") is None


def test_scope_and_scope_confidence_round_trip():
    """`get()` also returns `scope`/`scope_confidence`; nothing else in this task
    reads either field back, so a mix-up would otherwise go uncaught.

    Mutant this kills: transposing the two -- e.g. `get()` returning
    `row["scope_confidence"]` under the `scope` key -- would pass every other
    test in this file (none of them pass or assert these two fields) but fails
    the equality checks below, since a string and a float are never equal.
    """
    cache = ExtractionCache(":memory:")
    cache.initialize()
    cache.put(
        "evt-7",
        "1.4.0",
        "2026-06-14-r5",
        "model-abc",
        outcome=OUTCOME_EXTRACTED,
        entities=[],
        relationships=[],
        scope="user_only",
        scope_confidence=0.42,
        created_at="2026-08-18T00:00:00+00:00",
    )
    hit = cache.get("evt-7", "2026-06-14-r5", "model-abc")
    assert hit is not None
    assert hit["scope"] == "user_only"
    assert hit["scope_confidence"] == 0.42


def test_put_refuses_skipped_without_a_reason():
    cache = ExtractionCache(":memory:")
    cache.initialize()
    with pytest.raises(ValueError, match="requires a skip_reason"):
        cache.put(
            "evt-3",
            "1.4.0",
            "2026-06-14-r5",
            "model-abc",
            outcome=OUTCOME_SKIPPED,
            created_at="2026-08-18T00:00:00+00:00",
        )


def test_put_refuses_extracted_with_a_reason():
    cache = ExtractionCache(":memory:")
    cache.initialize()
    with pytest.raises(ValueError, match="must not carry"):
        cache.put(
            "evt-4",
            "1.4.0",
            "2026-06-14-r5",
            "model-abc",
            outcome=OUTCOME_EXTRACTED,
            entities=[],
            relationships=[],
            skip_reason=SKIP_RATE_LIMITED,
            created_at="2026-08-18T00:00:00+00:00",
        )


def test_put_refuses_an_unknown_skip_reason():
    cache = ExtractionCache(":memory:")
    cache.initialize()
    with pytest.raises(ValueError, match="unknown skip_reason"):
        cache.put(
            "evt-5",
            "1.4.0",
            "2026-06-14-r5",
            "model-abc",
            outcome=OUTCOME_SKIPPED,
            skip_reason="i_felt_like_it",
            created_at="2026-08-18T00:00:00+00:00",
        )


def test_put_refuses_an_unknown_outcome():
    """The `outcome not in (OUTCOME_EXTRACTED, OUTCOME_SKIPPED)` guard had no
    test at all -- deleting it left all 8 original tests green.
    """
    cache = ExtractionCache(":memory:")
    cache.initialize()
    with pytest.raises(ValueError, match="unknown outcome"):
        cache.put(
            "evt-6",
            "1.4.0",
            "2026-06-14-r5",
            "model-abc",
            outcome="not_a_real_outcome",
            created_at="2026-08-18T00:00:00+00:00",
        )


def _create_pre_phase_1_table(db_path) -> None:
    """Build a table in the OLD (pre-Task-1) shape: no outcome/skip_reason/scope columns."""
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE extraction_cache (
            cache_key TEXT PRIMARY KEY,
            event_id TEXT NOT NULL,
            ontology_version TEXT NOT NULL,
            extraction_version TEXT NOT NULL,
            model_hash TEXT NOT NULL,
            payload TEXT NOT NULL,
            created_at TEXT
        );
        """
    )
    conn.close()


def test_migration_is_idempotent_and_adds_the_new_columns(tmp_path):
    """An existing pre-phase-1 table must gain the new columns, not crash."""
    db = tmp_path / "old.db"
    _create_pre_phase_1_table(db)

    cache = ExtractionCache(str(db))
    cache.initialize()
    cache.initialize()  # idempotent

    cols = {
        row[1] for row in cache._get_connection().execute("PRAGMA table_info(extraction_cache)")
    }
    assert {"outcome", "skip_reason", "scope", "scope_confidence"} <= cols


def test_migration_preserves_existing_rows_and_backfills_the_outcome_default(tmp_path):
    """Gaining columns is not enough -- migrating must not lose data that predates it.

    Mutant this kills: replacing the ALTER-TABLE migration with `DROP TABLE` then
    `executescript(_DDL)` would ALSO leave the four new columns in place (passing
    the column-existence test above) while silently destroying every row cached
    before the migration ran. This test inserts a row under the pre-phase-1 shape
    BEFORE migrating, so only data-preserving behaviour can pass it.
    """
    db = tmp_path / "old.db"
    _create_pre_phase_1_table(db)

    import sqlite3

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    old_key = cache_key("evt-pre-migration", "2026-06-14-r5", "model-abc")
    conn.execute(
        "INSERT INTO extraction_cache "
        "(cache_key, event_id, ontology_version, extraction_version, model_hash, "
        "payload, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            old_key,
            "evt-pre-migration",
            "1.4.0",
            "2026-06-14-r5",
            "model-abc",
            '{"entities": [], "relationships": []}',
            "2026-08-01T00:00:00+00:00",
        ),
    )
    conn.commit()
    conn.close()

    cache = ExtractionCache(str(db))
    cache.initialize()

    hit = cache.get("evt-pre-migration", "2026-06-14-r5", "model-abc")
    assert hit is not None  # the pre-migration row survived the ALTER TABLE
    assert hit["outcome"] == OUTCOME_EXTRACTED  # NOT NULL DEFAULT 'extracted' backfilled it


def test_a_freshly_created_table_and_a_migrated_one_have_the_same_columns(tmp_path):
    """`_DDL` and `_MIGRATION_COLUMNS` declare the four new columns independently.

    Mutant this kills: adding (or renaming) a column in `_DDL` alone, without the
    matching entry in `_MIGRATION_COLUMNS`, gives freshly created databases a
    schema that migrated (pre-phase-1) databases never receive -- `put()`'s fixed
    INSERT column list would then raise `sqlite3.OperationalError` on an older
    database only, an age-dependent failure this test catches at the schema level
    before any INSERT is attempted.
    """
    fresh = ExtractionCache(":memory:")
    fresh.initialize()
    fresh_cols = {
        row[1] for row in fresh._get_connection().execute("PRAGMA table_info(extraction_cache)")
    }

    db = tmp_path / "old.db"
    _create_pre_phase_1_table(db)
    migrated = ExtractionCache(str(db))
    migrated.initialize()
    migrated_cols = {
        row[1] for row in migrated._get_connection().execute("PRAGMA table_info(extraction_cache)")
    }

    assert fresh_cols == migrated_cols
