"""One-shot bitemporal backfill for legacy fact edges (C1, design 4.4)."""

from backend.knowledge.admin import backfill_bitemporal
from tests.mocks.neo4j import FakeNeo4jConnection


class TestBackfillBitemporal:
    def test_guards_on_missing_source_utterance_id(self):
        conn = FakeNeo4jConnection(write_results=[{"updated": 3}])
        result = backfill_bitemporal(conn, ontology_version="1.2.0")
        query, params = conn.writes[0]
        assert "r.source_utterance_id IS NULL" in query  # idempotency guard
        assert "type(r) IN $extractable" in query
        assert params["ontology_version"] == "1.2.0"
        assert result["edges_backfilled"] == 3

    def test_seed_edges_map_to_canonical_seed_source(self):
        conn = FakeNeo4jConnection(write_results=[{"updated": 1}])
        backfill_bitemporal(conn, ontology_version="1.2.0")
        query, _ = conn.writes[0]
        # Legacy seed edges (event_id='seed') stamp 'seed' for both the
        # source and the version_key so the new seed MERGE matches them.
        assert "WHEN r.event_id = 'seed' THEN 'seed'" in query
        assert "r.version_key" in query

    def test_dry_run_counts_without_writing(self):
        conn = FakeNeo4jConnection(query_results=[{"n": 7}])
        result = backfill_bitemporal(conn, ontology_version="1.2.0", dry_run=True)
        assert result == {"candidates": 7}
        conn.assert_no_writes()
