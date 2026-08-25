"""Content-addressed extraction cache (F3)."""

import pytest

from backend.knowledge.extraction_cache import (
    OUTCOME_EXTRACTED,
    OUTCOME_SKIPPED,
    SKIP_RATE_LIMITED,
    ExtractionCache,
    cache_key,
)


def test_cache_key_ignores_ontology_version():
    """D3: the key answers 'what fed the LLM'. The ontology does not."""
    a = cache_key("evt-1", "2026-06-14-r5", "model-abc")
    b = cache_key("evt-1", "2026-06-14-r5", "model-abc")
    assert a == b


def test_cache_key_changes_with_extraction_version_and_model():
    base = cache_key("evt-1", "2026-06-14-r5", "model-abc")
    assert cache_key("evt-1", "2026-08-18-r6", "model-abc") != base
    assert cache_key("evt-1", "2026-06-14-r5", "model-xyz") != base
    assert cache_key("evt-2", "2026-06-14-r5", "model-abc") != base


def test_ontology_bump_alone_keeps_the_entry_reachable():
    """The whole point of D3: bump the ontology, the cache stays warm."""
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
    # Same event, same prompt, same model -- ONLY the ontology moved.
    hit = cache.get("evt-1", "2026-06-14-r5", "model-abc")
    assert hit is not None
    assert hit["outcome"] == OUTCOME_EXTRACTED
    assert hit["entities"] == [{"id": "user", "name": "User", "type": "User"}]


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


def test_put_refuses_skipped_without_a_reason():
    cache = ExtractionCache(":memory:")
    cache.initialize()
    with pytest.raises(ValueError, match="skip_reason"):
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
    with pytest.raises(ValueError, match="skip_reason"):
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


def test_initialize_is_idempotent_and_migrates_an_old_table(tmp_path):
    """An existing pre-phase-1 table must gain the new columns, not crash."""
    import sqlite3

    db = tmp_path / "old.db"
    conn = sqlite3.connect(str(db))
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

    cache = ExtractionCache(str(db))
    cache.initialize()
    cache.initialize()  # idempotent

    cols = {
        row[1] for row in cache._get_connection().execute("PRAGMA table_info(extraction_cache)")
    }
    assert {"outcome", "skip_reason", "scope", "scope_confidence"} <= cols
