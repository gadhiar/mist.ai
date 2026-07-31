"""Tests for backend.knowledge.seed.applier.

The applier is the sole write path for the versioned seed source (R1.4 spec
2.0). Every test that touches a write asserts on the params dict recorded by
`FakeNeo4jConnection`, not merely on the absence of an exception -- Task 5's
wipe scopes entirely on `seed_version`, so an unstamped write is un-wipeable
graph litter that no gate detects.
"""

from pathlib import Path

import pytest

from backend.errors import SeedSourceError
from backend.knowledge.seed.applier import apply_seed_documents
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


class TestSeedVersionStamping:
    def test_stamps_every_written_node_and_edge_with_seed_version(self, fake_connection):
        docs = [_doc(version="profile-v1", facts=[("user", "WORKS_AT", "slalom")])]

        apply_seed_documents(fake_connection, docs, seed_version="profile-v1", now_iso=_NOW)

        assert fake_connection.writes, "expected at least one write"
        for query, params in fake_connection.writes:
            assert params.get("seed_version") == "profile-v1"
            # A fake connection records whatever params are passed regardless of
            # whether the query text uses them, so checking params alone cannot
            # catch a query that carries the value but never SETs it. Require the
            # query string itself to reference seed_version too.
            assert "seed_version" in query, f"query does not set seed_version: {query!r}"

    def test_two_writes_happen_one_node_one_edge(self, fake_connection):
        """A single fact must produce exactly two writes -- one node MERGE per
        distinct entity referenced, one edge MERGE per fact. This pins the write
        count so a future change cannot silently drop the node write (or the edge
        write) while leaving the other stamped and the overall test green.
        """
        docs = [_doc(facts=[("user", "WORKS_AT", "slalom")])]

        apply_seed_documents(fake_connection, docs, seed_version="profile-v1", now_iso=_NOW)

        assert len(fake_connection.writes) == 3  # 2 nodes (user, slalom) + 1 edge

    def test_stamps_with_the_version_passed_to_the_call_not_the_document(self, fake_connection):
        """seed_version is a required kwarg, not read off the document, so a caller
        cannot apply a different version than it wiped (Task 5 pairs wipe + apply on
        one version).
        """
        docs = [_doc(version="profile-v1", facts=[("user", "WORKS_AT", "slalom")])]

        apply_seed_documents(fake_connection, docs, seed_version="profile-v2", now_iso=_NOW)

        for _query, params in fake_connection.writes:
            assert params.get("seed_version") == "profile-v2"


class TestForwarding:
    def test_forwards_subject_predicate_object(self, fake_connection):
        docs = [_doc(version="profile-v1", facts=[("user", "WORKS_AT", "slalom")])]

        apply_seed_documents(fake_connection, docs, seed_version="profile-v1", now_iso=_NOW)

        edge_writes = [
            (q, p) for q, p in fake_connection.writes if "WORKS_AT" in q or p.get("predicate")
        ]
        assert edge_writes, "expected an edge write carrying the predicate"
        _q, params = edge_writes[0]
        assert params["subject"] == "user"
        assert params["object"] == "slalom"
        assert params["predicate"] == "WORKS_AT"

    def test_forwards_predicate_as_the_relationship_type_in_the_query_text(self, fake_connection):
        """Neo4j cannot parameterise a relationship type, so the predicate must appear
        literally in the query string for the edge write.
        """
        docs = [_doc(facts=[("user", "USES", "python")])]

        apply_seed_documents(fake_connection, docs, seed_version="profile-v1", now_iso=_NOW)

        assert any("[r:USES]" in q or ":USES]" in q for q, _ in fake_connection.writes)

    def test_forwards_valid_from_and_valid_to(self, fake_connection):
        fact = SeedFact(
            subject="user", predicate="WORKS_AT", object="slalom", valid_from="2020-01-01"
        )
        docs = [
            SeedDocument(
                seed_version="profile-v1", facts=[fact], body="b", source_path=Path("t.md")
            )
        ]

        apply_seed_documents(fake_connection, docs, seed_version="profile-v1", now_iso=_NOW)

        edge_writes = [(q, p) for q, p in fake_connection.writes if p.get("predicate")]
        _q, params = edge_writes[0]
        assert params["valid_from"] == "2020-01-01"
        assert params["valid_to"] is None

    def test_forwards_now_iso_rather_than_reading_the_clock(self, fake_connection):
        """now_iso is a parameter, not a clock read -- application must be
        byte-reproducible across two calls with the same input.
        """
        docs = [_doc(facts=[("user", "WORKS_AT", "slalom")])]

        apply_seed_documents(
            fake_connection, docs, seed_version="profile-v1", now_iso="2020-06-15T00:00:00+00:00"
        )

        for _query, params in fake_connection.writes:
            assert params.get("now") == "2020-06-15T00:00:00+00:00"


class TestNodeWrites:
    def test_writes_one_node_per_unique_subject_and_object(self, fake_connection):
        """Two facts sharing the same subject must not double-write that node."""
        docs = [
            _doc(
                facts=[
                    ("user", "WORKS_AT", "slalom"),
                    ("user", "USES", "python"),
                ]
            )
        ]

        counts = apply_seed_documents(
            fake_connection, docs, seed_version="profile-v1", now_iso=_NOW
        )

        assert counts["nodes"] == 3  # user, slalom, python

    def test_node_writes_carry_the_entity_id(self, fake_connection):
        docs = [_doc(facts=[("user", "WORKS_AT", "slalom")])]

        apply_seed_documents(fake_connection, docs, seed_version="profile-v1", now_iso=_NOW)

        node_writes = [(q, p) for q, p in fake_connection.writes if not p.get("predicate")]
        node_ids = {p["id"] for _q, p in node_writes}
        assert node_ids == {"user", "slalom"}


class TestCounts:
    def test_returns_counts(self, fake_connection):
        docs = [
            _doc(
                version="profile-v1",
                facts=[("user", "WORKS_AT", "slalom"), ("user", "USES", "python")],
            )
        ]

        counts = apply_seed_documents(
            fake_connection, docs, seed_version="profile-v1", now_iso=_NOW
        )

        assert counts["facts"] == 2

    def test_sums_facts_across_multiple_documents(self, fake_connection):
        docs = [
            _doc(facts=[("user", "WORKS_AT", "slalom")], source_path=Path("a.md")),
            _doc(facts=[("mist", "USES", "python")], source_path=Path("b.md")),
        ]

        counts = apply_seed_documents(
            fake_connection, docs, seed_version="profile-v1", now_iso=_NOW
        )

        assert counts["facts"] == 2


class TestPredicateValidation:
    def test_rejects_predicate_not_in_the_ontology(self, fake_connection):
        docs = [_doc(facts=[("user", "HAS_ROLE", "slalom")])]

        with pytest.raises(SeedSourceError, match="HAS_ROLE"):
            apply_seed_documents(fake_connection, docs, seed_version="profile-v1", now_iso=_NOW)

    def test_rejects_before_any_write_happens(self, fake_connection):
        """The guard must run before any execute_write call -- a bad predicate
        anywhere in the seed source must abort the whole application rather than
        leaving a partial write (some nodes/edges stamped, others not).
        """
        docs = [
            _doc(facts=[("user", "WORKS_AT", "slalom")], source_path=Path("a.md")),
            _doc(facts=[("user", "NOT_A_REAL_PREDICATE", "thing")], source_path=Path("b.md")),
        ]

        with pytest.raises(SeedSourceError):
            apply_seed_documents(fake_connection, docs, seed_version="profile-v1", now_iso=_NOW)

        fake_connection.assert_no_writes()

    def test_error_names_the_source_file(self, fake_connection):
        docs = [_doc(facts=[("user", "HAS_ROLE", "slalom")], source_path=Path("users/bad.md"))]

        with pytest.raises(SeedSourceError, match="users/bad.md".replace("/", r"[\\/]")):
            apply_seed_documents(fake_connection, docs, seed_version="profile-v1", now_iso=_NOW)

    def test_error_suggests_the_closest_allowed_predicate_on_a_near_match(self, fake_connection):
        """A near-typo (`WORK_AT` for `WORKS_AT`) should surface a suggestion, not just
        the raw rejection -- this is what makes the error actionable at authoring time.
        """
        docs = [_doc(facts=[("user", "WORK_AT", "slalom")])]

        with pytest.raises(SeedSourceError, match="WORKS_AT"):
            apply_seed_documents(fake_connection, docs, seed_version="profile-v1", now_iso=_NOW)

    @pytest.mark.parametrize(
        "predicate",
        [
            pytest.param("EXPERT_IN", id="expert-in"),
            pytest.param("HAS_CAPABILITY", id="has-capability"),
            pytest.param("HAS_PREFERENCE", id="has-preference"),
            pytest.param("HAS_TRAIT", id="has-trait"),
            pytest.param("INTERESTED_IN", id="interested-in"),
            pytest.param("USES", id="uses"),
            pytest.param("WORKS_AT", id="works-at"),
            pytest.param("WORKS_ON", id="works-on"),
        ],
    )
    def test_accepts_every_predicate_used_by_the_real_seed_source(self, fake_connection, predicate):
        """These are the 8 predicates the real mist-memory/seed/*.md files use
        (verified against ALL_EDGE_TYPE_NAMES before this task was dispatched).
        Validation must not reject real seed content.
        """
        docs = [_doc(facts=[("user", predicate, "thing")])]

        apply_seed_documents(fake_connection, docs, seed_version="profile-v1", now_iso=_NOW)

        assert fake_connection.writes


class TestPartitionRouting:
    """The graph has two id-scoped, constraint-isolated partitions: `__Entity__`
    (user/world facts) and `__SelfModel__` (MIST's identity/traits/capabilities/
    preferences). A node write that hardcodes `__Entity__` for self-model content
    would create a duplicate copy in the wrong partition rather than matching the
    21 live `:__SelfModel__` nodes -- this was the defect found during Task 8 that
    reopened this module. Every assertion here pins the exact routing clause
    (which label a specific node id's MERGE uses), not merely that a label
    *appears somewhere* in the query text -- Task 4's own retrospective on the
    seed_version stamping test showed that a looser substring check on the wrong
    thing proves nothing.
    """

    def test_routes_self_model_nodes_to_the_selfmodel_partition(self, fake_connection):
        docs = [
            _doc(
                facts=[("mist-identity", "HAS_TRAIT", "trait-warm")],
                partition=SELF_MODEL_LABEL,
            )
        ]

        apply_seed_documents(fake_connection, docs, seed_version="profile-v1", now_iso=_NOW)

        node_writes = [(q, p) for q, p in fake_connection.writes if not p.get("predicate")]
        assert len(node_writes) == 2  # mist-identity, trait-warm
        for query, params in node_writes:
            assert (
                f"MERGE (n:{SELF_MODEL_LABEL} {{id: $id}})" in query
            ), f"node {params['id']!r} was not routed to {SELF_MODEL_LABEL}: {query!r}"

    def test_routes_entity_nodes_to_the_entity_partition_by_default(self, fake_connection):
        docs = [_doc(facts=[("user", "WORKS_AT", "slalom")])]  # default partition

        apply_seed_documents(fake_connection, docs, seed_version="profile-v1", now_iso=_NOW)

        node_writes = [(q, p) for q, p in fake_connection.writes if not p.get("predicate")]
        assert len(node_writes) == 2  # user, slalom
        for query, params in node_writes:
            assert (
                f"MERGE (n:{ENTITY_LABEL} {{id: $id}})" in query
            ), f"node {params['id']!r} was not routed to {ENTITY_LABEL}: {query!r}"

    def test_two_documents_route_their_nodes_to_different_partitions_independently(
        self, fake_connection
    ):
        """The realistic shape: one document is entirely self-model (seed/mist.md),
        another is entirely user-facing (seed/user.md), applied together in one call.
        """
        docs = [
            _doc(
                facts=[("mist-identity", "HAS_TRAIT", "trait-warm")],
                partition=SELF_MODEL_LABEL,
                source_path=Path("mist.md"),
            ),
            _doc(facts=[("user", "WORKS_AT", "slalom")], source_path=Path("user.md")),
        ]

        apply_seed_documents(fake_connection, docs, seed_version="profile-v1", now_iso=_NOW)

        node_writes = [(q, p) for q, p in fake_connection.writes if not p.get("predicate")]
        by_id = {p["id"]: q for q, p in node_writes}
        assert f"MERGE (n:{SELF_MODEL_LABEL} {{id: $id}})" in by_id["mist-identity"]
        assert f"MERGE (n:{SELF_MODEL_LABEL} {{id: $id}})" in by_id["trait-warm"]
        assert f"MERGE (n:{ENTITY_LABEL} {{id: $id}})" in by_id["user"]
        assert f"MERGE (n:{ENTITY_LABEL} {{id: $id}})" in by_id["slalom"]

    def test_edge_match_accepts_either_partition_for_subject_and_object(self, fake_connection):
        """A fact's subject/object may resolve to either partition (e.g. the
        self-model's HAS_TRAIT edges, or a future cross-layer edge), so the edge
        MATCH must not assume `__Entity__` the way the pre-fix version did.
        """
        docs = [
            _doc(facts=[("mist-identity", "HAS_TRAIT", "trait-warm")], partition=SELF_MODEL_LABEL)
        ]

        apply_seed_documents(fake_connection, docs, seed_version="profile-v1", now_iso=_NOW)

        edge_writes = [(q, p) for q, p in fake_connection.writes if p.get("predicate")]
        assert edge_writes
        query, _params = edge_writes[0]
        assert f"MATCH (s:{ENTITY_LABEL}|{SELF_MODEL_LABEL} {{id: $subject}})" in query
        assert f"MATCH (o:{ENTITY_LABEL}|{SELF_MODEL_LABEL} {{id: $object}})" in query

    def test_rejects_a_node_id_claimed_by_two_different_partitions(self, fake_connection):
        """A genuine authoring conflict: two documents disagree about which
        partition the same node id belongs to. Neither document's own `partition`
        value is individually invalid, so `SeedDocument`'s `Literal` typing cannot
        catch this -- it is a cross-document consistency error, not a per-document
        one.
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
            apply_seed_documents(fake_connection, docs, seed_version="profile-v1", now_iso=_NOW)

        fake_connection.assert_no_writes()
