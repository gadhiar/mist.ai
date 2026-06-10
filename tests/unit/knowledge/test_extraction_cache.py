"""Content-addressed extraction cache (F3)."""

from backend.knowledge.extraction_cache import ExtractionCache, cache_key


def _cache(tmp_path):
    c = ExtractionCache(str(tmp_path / "cache.db"))
    c.initialize()
    return c


class TestCacheKey:
    def test_stable(self):
        assert cache_key("e1", "1.1.0", "v1", "m1") == cache_key("e1", "1.1.0", "v1", "m1")

    def test_distinct_on_any_component(self):
        base = cache_key("e1", "1.1.0", "v1", "m1")
        assert base != cache_key("e2", "1.1.0", "v1", "m1")
        assert base != cache_key("e1", "1.2.0", "v1", "m1")
        assert base != cache_key("e1", "1.1.0", "v2", "m1")
        assert base != cache_key("e1", "1.1.0", "v1", "m2")


class TestExtractionCache:
    def test_put_get_roundtrip(self, tmp_path):
        c = _cache(tmp_path)
        c.put(
            "e1",
            "1.1.0",
            "v1",
            "m1",
            entities=[{"id": "rust", "type": "Technology"}],
            relationships=[{"source": "user", "target": "rust", "type": "USES"}],
            created_at="t",
        )
        got = c.get("e1", "1.1.0", "v1", "m1")
        assert got == {
            "entities": [{"id": "rust", "type": "Technology"}],
            "relationships": [{"source": "user", "target": "rust", "type": "USES"}],
        }
        c.close()

    def test_miss_returns_none(self, tmp_path):
        c = _cache(tmp_path)
        assert c.get("e1", "1.1.0", "v1", "m1") is None
        c.close()

    def test_stamp_change_is_a_miss(self, tmp_path):
        c = _cache(tmp_path)
        c.put("e1", "1.1.0", "v1", "m1", entities=[], relationships=[], created_at="t")
        assert c.get("e1", "1.2.0", "v1", "m1") is None  # ontology drift -> miss
        c.close()

    def test_put_is_idempotent_on_same_key(self, tmp_path):
        c = _cache(tmp_path)
        c.put(
            "e1",
            "1.1.0",
            "v1",
            "m1",
            entities=[{"id": "a", "type": "Topic"}],
            relationships=[],
            created_at="t1",
        )
        c.put(
            "e1",
            "1.1.0",
            "v1",
            "m1",
            entities=[{"id": "b", "type": "Topic"}],
            relationships=[],
            created_at="t2",
        )
        # INSERT OR REPLACE -> one row, last write wins.
        got = c.get("e1", "1.1.0", "v1", "m1")
        assert got["entities"] == [{"id": "b", "type": "Topic"}]
        c.close()
