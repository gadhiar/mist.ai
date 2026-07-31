"""Tests for `wipe_seed_version` and `reseed` in backend.knowledge.seed.applier.

The wipe is the single most destructive operation in the R1.4 sub-project: it
is the only mechanism that makes the graph track the seed source instead of
only ever accumulating it (MERGE alone cannot remove a fact deleted from the
source). Every scoping test here asks not just "does the query mention
`seed_version`" but "would this test still pass if the WHERE clause were
dropped and only the parameter survived" -- a params-only assertion cannot
tell the difference between a query that filters on the stamp and one that
ignores it entirely (see Task 4's finding on `FakeNeo4jConnection` recording
whatever params it is handed regardless of whether the query text uses them).
"""

from pathlib import Path

import pytest

from backend.errors import SeedSourceError
from backend.knowledge.seed.applier import reseed, wipe_seed_version
from backend.knowledge.seed.models import SeedDocument, SeedFact
from backend.knowledge.storage.partitions import ENTITY_LABEL, SELF_MODEL_LABEL

_NOW = "2026-07-31T00:00:00+00:00"


def _doc(
    *,
    version: str = "profile-v1",
    facts: list[tuple[str, str, str]] | None = None,
    body: str = "test body",
    source_path: Path = Path("test.md"),
    partition: str = ENTITY_LABEL,
) -> SeedDocument:
    """Build a valid SeedDocument. `facts` is a list of (subject, predicate, object)."""
    fact_objs = [SeedFact(subject=s, predicate=p, object=o) for s, p, o in (facts or [])]
    return SeedDocument(
        seed_version=version,
        facts=fact_objs,
        body=body,
        source_path=source_path,
        partition=partition,
    )


class TestWipeScoping:
    def test_wipe_only_touches_stamped_nodes(self, fake_connection):
        wipe_seed_version(fake_connection, "profile-v1")

        for query, params in fake_connection.writes:
            assert "seed_version" in query, (
                "wipe must scope on seed_version -- an unscoped DELETE would "
                "destroy real conversation-derived facts"
            )
            assert params["seed_version"] == "profile-v1"

    def test_edge_wipe_filters_on_seed_version_in_a_where_clause(self, fake_connection):
        """The broader `"seed_version" in query` assertion above would still pass
        if the WHERE clause were dropped but the string survived elsewhere in the
        query (e.g. moved into a comment or an unrelated SET). Pin the exact
        filter clause so that regression cannot slip through.
        """
        wipe_seed_version(fake_connection, "profile-v1")

        edge_query, _params = fake_connection.writes[0]
        assert "WHERE r.seed_version = $seed_version" in edge_query

    def test_node_wipe_filters_on_seed_version_in_a_where_clause(self, fake_connection):
        wipe_seed_version(fake_connection, "profile-v1")

        node_query, _params = fake_connection.writes[1]
        assert "WHERE n.seed_version = $seed_version" in node_query

    def test_wipe_deletes_edges_before_nodes(self, fake_connection):
        """Order matters: the node delete's `NOT (n)--()` guard only finds a node
        orphaned once its own seeded edges are already gone. Reversed, every
        seeded node would still have a relationship and the node delete would
        silently remove nothing.
        """
        wipe_seed_version(fake_connection, "profile-v1")

        assert len(fake_connection.writes) == 2
        edge_query, _ = fake_connection.writes[0]
        node_query, _ = fake_connection.writes[1]
        assert "DELETE r" in edge_query
        assert "DELETE n" in node_query

    def test_node_wipe_keeps_nodes_that_still_have_a_relationship(self, fake_connection):
        """Deliberate asymmetry: a seeded node that has since acquired a
        conversation-derived edge must survive the wipe, because dropping it
        would delete a real fact the seed layer has no authority over. The fake
        connection never executes Cypher, so this is a structural check that the
        guard clause is present in the query Neo4j would enforce -- the
        keep/drop behaviour itself is exercised by Task 10's live apply.
        """
        wipe_seed_version(fake_connection, "profile-v1")

        _edge_query, _ = fake_connection.writes[0]
        node_query, _ = fake_connection.writes[1]
        assert "NOT (n)--()" in node_query

    def test_wipe_scopes_to_the_version_passed_not_a_hardcoded_value(self, fake_connection):
        """Proves the scope is parametrised rather than baked in: wiping a
        different version must carry that version, never `profile-v1`, in every
        write -- this is the guard firing on content it does not own.
        """
        wipe_seed_version(fake_connection, "some-other-version")

        for _query, params in fake_connection.writes:
            assert params["seed_version"] == "some-other-version"
            assert params["seed_version"] != "profile-v1"

    def test_wipe_returns_counts_keyed_edges_and_nodes(self, fake_connection):
        result = wipe_seed_version(fake_connection, "profile-v1")

        assert result == {"edges": 0, "nodes": 0}


class TestReseed:
    def test_reseed_is_idempotent(self, fake_connection):
        docs = [_doc(version="profile-v1", facts=[("user", "WORKS_AT", "slalom")])]

        first = reseed(fake_connection, docs, seed_version="profile-v1", now_iso=_NOW)
        fake_connection.writes.clear()
        second = reseed(fake_connection, docs, seed_version="profile-v1", now_iso=_NOW)

        assert first == second

    def test_reseed_drops_a_fact_removed_from_source(self, fake_connection):
        """The behaviour MERGE-only cannot provide."""
        two = [
            _doc(
                version="profile-v1",
                facts=[("user", "WORKS_AT", "slalom"), ("user", "USES", "python")],
            )
        ]
        one = [_doc(version="profile-v1", facts=[("user", "WORKS_AT", "slalom")])]

        reseed(fake_connection, two, seed_version="profile-v1", now_iso=_NOW)
        fake_connection.writes.clear()
        counts = reseed(fake_connection, one, seed_version="profile-v1", now_iso=_NOW)

        assert counts["facts"] == 1
        assert any("DELETE" in q.upper() for q, _ in fake_connection.writes)

    def test_reseed_wipes_before_reapplying(self, fake_connection):
        """The wipe's DELETE writes must precede the re-apply's MERGE writes --
        proves `reseed` does not apply first and wipe second, which would delete
        the content it just wrote.
        """
        docs = [_doc(facts=[("user", "WORKS_AT", "slalom")])]

        reseed(fake_connection, docs, seed_version="profile-v1", now_iso=_NOW)

        queries = [q for q, _ in fake_connection.writes]
        delete_indexes = [i for i, q in enumerate(queries) if "DELETE" in q.upper()]
        merge_indexes = [i for i, q in enumerate(queries) if "MERGE" in q.upper()]
        assert delete_indexes, "expected wipe DELETE writes"
        assert merge_indexes, "expected re-apply MERGE writes"
        assert max(delete_indexes) < min(merge_indexes)

    def test_reseed_forwards_now_iso_to_the_reapply_rather_than_reading_the_clock(
        self, fake_connection
    ):
        """`now_iso` is a required kwarg, not a clock read -- R1.3.1 shipped a
        `datetime.now()` fallback that drifted across UTC midnight. `reseed` must
        forward the exact value it was given, never default it.
        """
        docs = [_doc(facts=[("user", "WORKS_AT", "slalom")])]

        reseed(
            fake_connection,
            docs,
            seed_version="profile-v1",
            now_iso="2020-06-15T00:00:00+00:00",
        )

        reapply_writes = [(q, p) for q, p in fake_connection.writes if p.get("now") is not None]
        assert reapply_writes, "expected re-apply writes carrying now"
        for _query, params in reapply_writes:
            assert params["now"] == "2020-06-15T00:00:00+00:00"

    def test_reseed_stamps_reapplied_writes_with_the_given_seed_version(self, fake_connection):
        docs = [_doc(version="profile-v1", facts=[("user", "WORKS_AT", "slalom")])]

        reseed(fake_connection, docs, seed_version="profile-v2", now_iso=_NOW)

        for _query, params in fake_connection.writes:
            assert params["seed_version"] == "profile-v2"

    def test_reseed_validates_predicates_before_wiping(self, fake_connection):
        """A bad predicate in the new source must abort before the wipe runs.
        Without this, a typo in a source edit would empty a previously-good
        graph via the wipe and then fail the re-apply, leaving a real data-loss
        window open until the typo is fixed.
        """
        docs = [_doc(facts=[("user", "NOT_A_REAL_PREDICATE", "thing")])]

        with pytest.raises(SeedSourceError):
            reseed(fake_connection, docs, seed_version="profile-v1", now_iso=_NOW)

        fake_connection.assert_no_writes()

    def test_reseed_validates_partition_conflicts_before_wiping(self, fake_connection):
        """The same data-loss shape as the predicate case above, for the newer
        partition-conflict guard: a node id claimed by two different partitions
        must abort before the wipe runs, not after (R1.4 Task 4 rework).
        """
        docs = [
            _doc(
                facts=[("shared-id", "USES", "python")],
                partition=ENTITY_LABEL,
                source_path=Path("a.md"),
            ),
            _doc(
                facts=[("shared-id", "HAS_TRAIT", "trait-warm")],
                partition=SELF_MODEL_LABEL,
                source_path=Path("b.md"),
            ),
        ]

        with pytest.raises(SeedSourceError, match="shared-id"):
            reseed(fake_connection, docs, seed_version="profile-v1", now_iso=_NOW)

        fake_connection.assert_no_writes()

    def test_reseed_returns_the_reapply_counts(self, fake_connection):
        docs = [
            _doc(
                facts=[("user", "WORKS_AT", "slalom"), ("user", "USES", "python")],
            )
        ]

        counts = reseed(fake_connection, docs, seed_version="profile-v1", now_iso=_NOW)

        assert counts == {"nodes": 3, "facts": 2}
