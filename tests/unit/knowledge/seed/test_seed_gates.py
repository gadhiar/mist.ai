"""Tests for backend.knowledge.seed.gates.

Each gate is tested on input where it actually fires, not only on clean
input -- a gate exercised solely by input where the guarded thing never
happens proves nothing (this exact hole appeared three times in R1.3.1
and twice more during R1.4's own applier/wipe work). Where a test can
only be satisfied by `FakeNeo4jConnection` echoing back whatever it is
handed (it does not interpret Cypher), the query string itself is
asserted against directly -- pinning the specific clause, not merely
that a token like `seed_version` appears somewhere in the query.
"""

from pathlib import Path

from backend.knowledge.seed.gates import (
    check_containment,
    check_facts_present,
    check_negation_proximity,
)
from backend.knowledge.seed.models import SeedDocument, SeedFact
from backend.knowledge.storage.partitions import ENTITY_LABEL, SELF_MODEL_LABEL
from tests.mocks.neo4j import FakeNeo4jConnection


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


class TestFactsPresent:
    def test_facts_present_fails_when_a_fact_is_missing(self, fake_connection):
        fake_connection.query_results = [[]]  # graph returns nothing
        docs = [_doc(facts=[("user", "WORKS_AT", "slalom")])]

        result = check_facts_present(fake_connection, docs, seed_version="profile-v1")

        assert not result.passed
        assert "WORKS_AT" in result.failures[0]

    def test_passes_when_the_fact_is_present(self):
        connection = FakeNeo4jConnection(query_results=[{"n": 1}])
        docs = [_doc(facts=[("user", "WORKS_AT", "slalom")])]

        result = check_facts_present(connection, docs, seed_version="profile-v1")

        assert result.passed
        assert result.failures == []

    def test_reports_one_failure_per_missing_fact_across_documents(self):
        connection = FakeNeo4jConnection(query_results=[])
        docs = [
            _doc(facts=[("user", "WORKS_AT", "slalom")], source_path=Path("a.md")),
            _doc(facts=[("mist", "USES", "python")], source_path=Path("b.md")),
        ]

        result = check_facts_present(connection, docs, seed_version="profile-v1")

        assert len(result.failures) == 2

    def test_never_writes(self):
        """Gate 2 is a read-only check -- it must never issue execute_write."""
        connection = FakeNeo4jConnection(query_results=[{"n": 1}])
        docs = [_doc(facts=[("user", "WORKS_AT", "slalom")])]

        check_facts_present(connection, docs, seed_version="profile-v1")

        connection.assert_no_writes()

    def test_forwards_subject_object_and_seed_version_as_params(self):
        connection = FakeNeo4jConnection(query_results=[{"n": 1}])
        docs = [_doc(facts=[("user", "WORKS_AT", "slalom")])]

        check_facts_present(connection, docs, seed_version="profile-v1")

        assert len(connection.queries) == 1
        _query, params = connection.queries[0]
        assert params["subject"] == "user"
        assert params["object"] == "slalom"
        assert params["seed_version"] == "profile-v1"

    def test_uses_the_predicate_as_the_relationship_type_in_the_query_text(self):
        """Neo4j cannot parameterize a relationship type, so the predicate must
        appear literally in the query string, the same way the applier's edge
        write does.
        """
        connection = FakeNeo4jConnection(query_results=[{"n": 1}])
        docs = [_doc(facts=[("user", "USES", "python")])]

        check_facts_present(connection, docs, seed_version="profile-v1")

        query, _params = connection.queries[0]
        assert "[r:USES]" in query

    def test_filters_on_seed_version_in_the_where_clause(self):
        """Pins the exact filtering clause, not merely that the token
        `seed_version` appears in the query text -- a query that carried the
        parameter but never referenced it in a WHERE (or dropped the WHERE
        entirely, leaving the token alive only in a comment) would still pass
        a bare substring-of-the-whole-query check. This is the exact hole
        Task 5 found in the wipe query.
        """
        connection = FakeNeo4jConnection(query_results=[{"n": 1}])
        docs = [_doc(facts=[("user", "WORKS_AT", "slalom")])]

        check_facts_present(connection, docs, seed_version="profile-v1")

        query, _params = connection.queries[0]
        assert "WHERE r.seed_version = $seed_version" in query

    def test_matches_subject_and_object_across_both_graph_partitions(self):
        """The graph has two id-scoped, label-constrained partitions
        (__Entity__ and __SelfModel__) that cannot see each other in a MATCH
        restricted to one label. A self-model fact's subject (e.g.
        `mist-identity`) must still be found, so the query must MATCH the
        label union on both the subject and the object -- not just one side.
        """
        connection = FakeNeo4jConnection(query_results=[{"n": 1}])
        docs = [
            _doc(
                facts=[("mist-identity", "HAS_TRAIT", "trait-warm")],
                partition=SELF_MODEL_LABEL,
            )
        ]

        check_facts_present(connection, docs, seed_version="profile-v1")

        query, _params = connection.queries[0]
        assert f"MATCH (s:{ENTITY_LABEL}|{SELF_MODEL_LABEL} {{id: $subject}})" in query
        assert f"MATCH (o:{ENTITY_LABEL}|{SELF_MODEL_LABEL} {{id: $object}})" in query

    def test_selfmodel_only_fact_is_not_reported_missing_due_to_partition_restriction(self):
        """Guards against the regression this gate exists partly to catch: a
        query hardcoded to `:__Entity__` alone would return zero rows for
        every self-model fact and every one would be misreported as missing,
        even when it is actually present in the graph.
        """
        connection = FakeNeo4jConnection(query_results=[{"n": 1}])
        docs = [
            _doc(
                facts=[("mist-identity", "HAS_TRAIT", "trait-warm")],
                partition=SELF_MODEL_LABEL,
            )
        ]

        result = check_facts_present(connection, docs, seed_version="profile-v1")

        assert result.passed


class TestContainment:
    def test_containment_fails_when_object_absent_from_prose(self):
        docs = [_doc(facts=[("user", "WORKS_AT", "Slalom")], body="Raj works at Google.")]

        result = check_containment(docs)

        assert not result.passed
        assert "Slalom" in result.failures[0]

    def test_containment_passes_when_object_present(self):
        docs = [_doc(facts=[("user", "WORKS_AT", "Slalom")], body="Raj works at Slalom.")]

        assert check_containment(docs).passed

    def test_passes_vacuously_for_a_prose_only_document_with_no_facts(self):
        """A document with no `facts:` (e.g. an identity narrative) has nothing
        to contain -- this is a legitimate case documented in the loader, not
        a bug the gate should flag.
        """
        docs = [_doc(facts=[], body="Just prose, no assertions.")]

        assert check_containment(docs).passed

    def test_reports_one_failure_per_missing_object_not_per_document(self):
        docs = [
            _doc(
                facts=[("user", "WORKS_AT", "Slalom"), ("user", "USES", "Rust")],
                body="Raj works at Slalom and uses Python.",
            )
        ]

        result = check_containment(docs)

        assert len(result.failures) == 1
        assert "Rust" in result.failures[0]

    def test_checks_only_the_object_not_the_subject_or_predicate(self):
        """The brief scopes containment to the fact's object only -- subject
        wording ('Raj' vs 'the user') and predicate wording are not checked.
        """
        docs = [_doc(facts=[("user", "WORKS_AT", "Slalom")], body="He is employed by Slalom.")]

        assert check_containment(docs).passed


class TestNegationProximity:
    def test_negation_proximity_flags_inversion(self):
        docs = [
            _doc(
                facts=[("user", "WORKS_AT", "Slalom")],
                body="Raj no longer works at Slalom.",
            )
        ]

        result = check_negation_proximity(docs)

        assert not result.passed

    def test_negation_proximity_allows_distant_negation(self):
        """A negation elsewhere in the body must not trip the gate."""
        docs = [
            _doc(
                facts=[("user", "WORKS_AT", "Slalom")],
                body="Raj works at Slalom. He does not use Java anywhere.",
            )
        ]

        assert check_negation_proximity(docs).passed

    def test_ignores_a_real_marker_that_is_outside_the_proximity_window(self):
        """Distinguishes "no marker anywhere in the document" (the case above,
        which passes trivially with no marker present at all) from "a marker
        genuinely exists in the document, but far enough from this fact's
        object that the window correctly excludes it." Without this test, a
        gate that ignored the window entirely (flagged on ANY marker anywhere
        in the body) would still pass every test in this file.
        """
        padding = "x" * 100  # comfortably beyond the 60-char window either side
        body = f"Raj works at Slalom. {padding} He left the company years later."
        docs = [_doc(facts=[("user", "WORKS_AT", "Slalom")], body=body)]

        assert check_negation_proximity(docs).passed

    def test_flags_a_marker_within_the_window_even_when_not_adjacent(self):
        """The window is not mere adjacency-to-the-object -- a marker several
        words away, but still inside the window, must be caught.
        """
        padding = "x" * 30  # well inside the 60-char window, not touching the object
        body = f"Raj left {padding} Slalom behind for a new role."
        docs = [_doc(facts=[("user", "WORKS_AT", "Slalom")], body=body)]

        result = check_negation_proximity(docs)

        assert not result.passed
        assert "Slalom" in result.failures[0]

    def test_is_case_insensitive(self):
        docs = [
            _doc(
                facts=[("user", "WORKS_AT", "Slalom")],
                body="RAJ NO LONGER WORKS AT SLALOM.",
            )
        ]

        assert not check_negation_proximity(docs).passed

    def test_reports_one_failure_per_fact_not_per_marker_occurrence(self):
        """A fact whose object appears twice, both times near markers, must
        still produce exactly one failure line -- the gate flags the fact,
        not each occurrence.
        """
        docs = [
            _doc(
                facts=[("user", "WORKS_AT", "Slalom")],
                body="Formerly at Slalom, now elsewhere. Left Slalom in 2020.",
            )
        ]

        result = check_negation_proximity(docs)

        assert len(result.failures) == 1

    def test_passes_for_a_prose_only_document_with_no_facts(self):
        docs = [_doc(facts=[], body="No longer relevant prose with no assertions.")]

        assert check_negation_proximity(docs).passed
