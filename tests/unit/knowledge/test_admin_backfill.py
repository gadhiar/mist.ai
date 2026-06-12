"""One-shot bitemporal backfill for legacy fact edges (C1, design 4.4)."""

from backend.knowledge.admin import backfill_bitemporal
from tests.mocks.neo4j import FakeNeo4jConnection


class TestBackfillBitemporal:
    def test_guards_on_missing_source_utterance_id(self):
        conn = FakeNeo4jConnection(write_results=[{"updated": 3}])
        result = backfill_bitemporal(conn, ontology_version="1.2.0")
        query, params = conn.writes[0]
        assert "r.source_utterance_id IS NULL" in query  # idempotency guard
        assert "type(r) IN $backfill_types" in query
        assert params["ontology_version"] == "1.2.0"
        assert result["edges_backfilled"] == 3

    def test_seeded_internal_persona_edges_within_scope(self):
        # read-path-currency-4: HAS_TRAIT etc. are seeded under the same
        # version_key='seed' MERGE contract as anchor edges; excluding them
        # from backfill leaves pre-C1 persona edges unstamped, so the next
        # re-seed MERGE misses them and duplicates every identity edge.
        conn = FakeNeo4jConnection(write_results=[{"updated": 1}])
        backfill_bitemporal(conn, ontology_version="1.2.0")
        _, params = conn.writes[0]
        for predicate in ("HAS_TRAIT", "HAS_CAPABILITY", "HAS_PREFERENCE", "IS_UNCERTAIN_ABOUT"):
            assert predicate in params["backfill_types"]
        assert "USES" in params["backfill_types"]  # extractables still covered

    def test_seed_edges_map_to_canonical_seed_source(self):
        conn = FakeNeo4jConnection(write_results=[{"updated": 1}])
        backfill_bitemporal(conn, ontology_version="1.2.0")
        query, _ = conn.writes[0]
        # Legacy seed edges (event_id='seed') stamp 'seed' for both the
        # source and the version_key so the new seed MERGE matches them.
        assert "WHEN r.event_id = 'seed' THEN 'seed'" in query
        assert "r.version_key" in query

    def test_dry_run_counts_without_writing(self):
        from backend.knowledge.admin import _BACKFILL_GUARD

        conn = FakeNeo4jConnection(query_results=[{"n": 7}])
        result = backfill_bitemporal(conn, ontology_version="1.2.0", dry_run=True)
        # Dry/wet parity pin (tests-quality-9): the dry count must use the
        # SAME candidate-selection predicate the wet run writes with.
        query, params = conn.queries[0]
        assert _BACKFILL_GUARD in query
        assert "HAS_TRAIT" in params["backfill_types"]
        assert result["candidates"] == 7
        # The fake returns the same row for every per-type reverse count, so
        # the exact total is fake-shaped; what matters is the key exists and
        # the dry run surfaced the canonicalization workload.
        assert "undirected_reverse_candidates" in result
        conn.assert_no_writes()

    def test_wet_run_canonicalizes_reverse_undirected_edges(self):
        # recon-engine-5: the engine reads/writes only the lexical
        # (min)->(max) direction, so reverse-direction legacy WORKS_WITH /
        # RELATED_TO rows are invisible to same-fact fetches and accumulate
        # as permanent duplicates. Backfill must reverse them idempotently.
        conn = FakeNeo4jConnection(write_results=[{"updated": 3}])
        result = backfill_bitemporal(conn, ontology_version="1.2.0")
        canonicalize_writes = [q for q, _ in conn.writes if "WHERE a.id > b.id" in q]
        assert any("[r:WORKS_WITH]" in q for q in canonicalize_writes)
        assert any("[r:RELATED_TO]" in q for q in canonicalize_writes)
        for q in canonicalize_writes:
            assert "MERGE (b)-[c:" in q and "{version_key: r.version_key}" in q
            assert "DELETE r" in q
        assert "undirected_canonicalized" in result
