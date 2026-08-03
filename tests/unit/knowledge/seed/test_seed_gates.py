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

from backend.knowledge.embeddings.embedding_text import embedding_text_for
from backend.knowledge.seed.gates import (
    _node_by_id,
    _search_term_for,
    check_containment,
    check_embeddings,
    check_facts_present,
    check_negation_proximity,
    check_node_definitions,
)
from backend.knowledge.seed.loader import load_seed_documents
from backend.knowledge.seed.models import SeedDocument, SeedFact, SeedNode
from backend.knowledge.storage.partitions import ENTITY_LABEL, SELF_MODEL_LABEL
from tests.mocks.embeddings import EMBEDDING_DIMENSION, FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection

# Resolve path relative to repo root regardless of pytest invocation directory,
# matching tests/unit/knowledge/test_seed_data.py's convention.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_REAL_SEED_DIR = _REPO_ROOT / "mist-memory" / "seed"


def _doc(
    *,
    version: str = "profile-v1",
    facts: list[tuple[str, str, str]] | None = None,
    nodes: list[SeedNode] | None = None,
    body: str = "test body",
    source_path: Path = Path("test.md"),
    partition: str = ENTITY_LABEL,
) -> SeedDocument:
    """Build a valid SeedDocument. `facts` is a list of (subject, predicate, object).

    `nodes` defaults to empty -- most existing gate tests here (Gate 2,
    negation proximity, and containment's raw-id-fallback cases) do not
    need node definitions at all; tests exercising `check_node_definitions`
    or containment's display_name matching pass `nodes` explicitly.
    """
    fact_objs = [SeedFact(subject=s, predicate=p, object=o) for s, p, o in (facts or [])]
    return SeedDocument(
        seed_version=version,
        nodes=nodes or [],
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


class TestNodeDefinitions:
    """R1.4 Task 14: the gate Task 10's live defect needed and did not have.
    Every test here uses a node's `type` in the fixture that differs from
    its `id` (`MistTrait` vs `trait-warm`), so a query that accidentally
    matched on `id` instead of the interpolated type label could not pass
    by coincidence.
    """

    def test_fires_on_a_node_stripped_of_its_label_and_display_name(self, fake_connection):
        """THE test main asked for explicitly: a graph state where the gate
        must actually fire, not only a healthy one. `FakeNeo4jConnection`'s
        default `query_results=[]` simulates exactly Task 10's live defect --
        a node that exists but no longer matches the labeled, display_name-
        bearing MATCH pattern (its own label/properties were stripped by the
        wipe-and-recreate cycle).
        """
        docs = [
            _doc(
                nodes=[SeedNode(id="trait-warm", type="MistTrait", display_name="Warm")],
                partition=SELF_MODEL_LABEL,
            )
        ]

        result = check_node_definitions(fake_connection, docs, seed_version="profile-v1")

        assert not result.passed
        assert "trait-warm" in result.failures[0]
        assert "MistTrait" in result.failures[0]

    def test_passes_when_the_node_is_correctly_labeled(self):
        connection = FakeNeo4jConnection(query_results=[{"n": 1}])
        docs = [
            _doc(
                nodes=[SeedNode(id="trait-warm", type="MistTrait", display_name="Warm")],
                partition=SELF_MODEL_LABEL,
            )
        ]

        result = check_node_definitions(connection, docs, seed_version="profile-v1")

        assert result.passed
        assert result.failures == []

    def test_reports_one_failure_per_missing_node_across_documents(self):
        connection = FakeNeo4jConnection(query_results=[])
        docs = [
            _doc(
                nodes=[SeedNode(id="user", type="User", display_name="Raj")],
                source_path=Path("a.md"),
            ),
            _doc(
                nodes=[SeedNode(id="mist-identity", type="MistIdentity", display_name="MIST")],
                partition=SELF_MODEL_LABEL,
                source_path=Path("b.md"),
            ),
        ]

        result = check_node_definitions(connection, docs, seed_version="profile-v1")

        assert len(result.failures) == 2

    def test_never_writes(self):
        connection = FakeNeo4jConnection(query_results=[{"n": 1}])
        docs = [_doc(nodes=[SeedNode(id="user", type="User", display_name="Raj")])]

        check_node_definitions(connection, docs, seed_version="profile-v1")

        connection.assert_no_writes()

    def test_forwards_id_and_seed_version_as_params(self):
        connection = FakeNeo4jConnection(query_results=[{"n": 1}])
        docs = [_doc(nodes=[SeedNode(id="user", type="User", display_name="Raj")])]

        check_node_definitions(connection, docs, seed_version="profile-v1")

        assert len(connection.queries) == 1
        _query, params = connection.queries[0]
        assert params["id"] == "user"
        assert params["seed_version"] == "profile-v1"

    def test_uses_the_partition_and_type_as_labels_in_the_query_text(self):
        """Neo4j cannot parameterize a label, so both the document's
        partition label and the node's ontology type label must appear
        literally in the query, in one MATCH clause -- a node missing
        either fails to bind. Pins the exact clause shape, not merely
        that the label strings appear somewhere in the query.
        """
        connection = FakeNeo4jConnection(query_results=[{"n": 1}])
        docs = [
            _doc(
                nodes=[SeedNode(id="trait-warm", type="MistTrait", display_name="Warm")],
                partition=SELF_MODEL_LABEL,
            )
        ]

        check_node_definitions(connection, docs, seed_version="profile-v1")

        query, _params = connection.queries[0]
        assert f"MATCH (n:{SELF_MODEL_LABEL}:MistTrait {{id: $id}})" in query

    def test_checks_display_name_is_not_null_in_the_where_clause(self):
        """Pins the exact filtering clause, matching this sub-project's
        established discipline (Task 5's wipe-query finding): a token
        appearing anywhere in the query text proves nothing about what it
        does. `display_name IS NOT NULL` must be in a WHERE, not a comment.
        """
        connection = FakeNeo4jConnection(query_results=[{"n": 1}])
        docs = [_doc(nodes=[SeedNode(id="user", type="User", display_name="Raj")])]

        check_node_definitions(connection, docs, seed_version="profile-v1")

        query, _params = connection.queries[0]
        assert "WHERE n.seed_version = $seed_version AND n.display_name IS NOT NULL" in query

    def test_entity_partition_node_uses_entity_label(self):
        connection = FakeNeo4jConnection(query_results=[{"n": 1}])
        docs = [
            _doc(
                nodes=[SeedNode(id="slalom", type="Organization", display_name="Slalom")],
                partition=ENTITY_LABEL,
            )
        ]

        check_node_definitions(connection, docs, seed_version="profile-v1")

        query, _params = connection.queries[0]
        assert f"MATCH (n:{ENTITY_LABEL}:Organization {{id: $id}})" in query

    def test_passes_vacuously_for_a_document_defining_no_nodes(self):
        connection = FakeNeo4jConnection(query_results=[])
        docs = [_doc(nodes=[])]

        result = check_node_definitions(connection, docs, seed_version="profile-v1")

        assert result.passed
        connection.assert_no_writes()


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

    def test_matches_on_display_name_when_the_kebab_id_never_appears_in_prose(self):
        """R1.4 Task 14, the exact real-world shape: `trait-warm`'s display
        name is `Warm`. The raw id never appears in prose that says
        `**Warm**` -- only the pre-fix implementation's literal id
        substring check would fail this; matching on display_name passes it.
        """
        docs = [
            _doc(
                facts=[("mist-identity", "HAS_TRAIT", "trait-warm")],
                nodes=[
                    SeedNode(id="mist-identity", type="MistIdentity", display_name="MIST"),
                    SeedNode(id="trait-warm", type="MistTrait", display_name="Warm"),
                ],
                partition=SELF_MODEL_LABEL,
                body="MIST's default register is **Warm** and engaged.",
            )
        ]

        assert check_containment(docs).passed

    def test_fails_when_neither_display_name_nor_raw_id_appears(self):
        docs = [
            _doc(
                facts=[("mist-identity", "HAS_TRAIT", "trait-warm")],
                nodes=[
                    SeedNode(id="mist-identity", type="MistIdentity", display_name="MIST"),
                    SeedNode(id="trait-warm", type="MistTrait", display_name="Warm"),
                ],
                partition=SELF_MODEL_LABEL,
                body="MIST is described elsewhere. No trait is mentioned here.",
            )
        ]

        result = check_containment(docs)

        assert not result.passed
        assert "Warm" in result.failures[0]

    def test_is_case_insensitive_on_display_name(self):
        docs = [
            _doc(
                facts=[("user", "WORKS_AT", "slalom")],
                nodes=[SeedNode(id="slalom", type="Organization", display_name="Slalom")],
                body="Raj works at SLALOM as a consultant.",
            )
        ]

        assert check_containment(docs).passed

    def test_falls_back_to_raw_id_when_the_object_has_no_matching_node(self):
        """Referential integrity is load_seed_documents' job (Task 11), not
        this gate's -- containment must not silently pass (or crash) on a
        fact whose object has no SeedNode; it falls back to checking the
        raw id, preserving the pre-Task-14 behavior for that case.
        """
        docs = [_doc(facts=[("user", "WORKS_AT", "Slalom")], body="Raj works at Slalom.", nodes=[])]

        assert check_containment(docs).passed

    def test_falls_back_to_raw_id_when_the_node_defines_no_display_name(self):
        docs = [
            _doc(
                facts=[("user", "WORKS_AT", "Slalom")],
                nodes=[SeedNode(id="Slalom", type="Organization")],  # no display_name
                body="Raj works at Slalom.",
            )
        ]

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

    def test_flags_a_marker_near_a_display_name_when_the_raw_kebab_id_never_appears_in_prose(
        self,
    ):
        """C1 (R1.4 whole-branch review), the exact real-content shape Gate 3's
        equivalent test (`test_matches_on_display_name_when_the_kebab_id_never_
        appears_in_prose`) already covers for containment: `trait-transparent`'s
        display name is `Transparent`. The raw id never appears in prose that
        says `no longer **Transparent**` -- every fixture in this class before
        this one used a fact object that was ALSO the literal word the body
        used, which is exactly why this gate's raw-id defect (searching
        `fact.object` instead of the resolved display name) survived: those
        fixtures made the guarded thing reachable when real seed content does
        not. Pre-fix, `_find_all` would return `[]` for `trait-transparent`
        against this body, the scan loop would never run, and the gate would
        report `passed=True` having examined nothing.
        """
        docs = [
            _doc(
                facts=[("mist-identity", "HAS_TRAIT", "trait-transparent")],
                nodes=[
                    SeedNode(id="mist-identity", type="MistIdentity", display_name="MIST"),
                    SeedNode(id="trait-transparent", type="MistTrait", display_name="Transparent"),
                ],
                partition=SELF_MODEL_LABEL,
                body="MIST is no longer **Transparent** about her reasoning.",
            )
        ]

        result = check_negation_proximity(docs)

        assert not result.passed
        assert "trait-transparent" in result.failures[0]

    def test_falls_back_to_raw_id_when_the_object_has_no_matching_node(self):
        """Referential integrity is load_seed_documents' job (Task 11), not
        this gate's -- mirrors check_containment's identical fallback test.
        """
        docs = [
            _doc(
                facts=[("user", "WORKS_AT", "Slalom")],
                body="Raj no longer works at Slalom.",
                nodes=[],
            )
        ]

        assert not check_negation_proximity(docs).passed

    def test_falls_back_to_raw_id_when_the_node_defines_no_display_name(self):
        docs = [
            _doc(
                facts=[("user", "WORKS_AT", "Slalom")],
                nodes=[SeedNode(id="Slalom", type="Organization")],  # no display_name
                body="Raj no longer works at Slalom.",
            )
        ]

        assert not check_negation_proximity(docs).passed


class TestNegationProximityRealSource:
    """C1 (R1.4 whole-branch review): a gate exercised only by fixtures where
    the guarded thing is reachable proves nothing about whether it is
    reachable against real content -- exactly how this gate's raw-id defect
    survived Task 14, which fixed the identical defect in check_containment
    and left this gate on the raw id. Live measurement against the real
    source before the fix: 4 of 30 facts scannable overall, 0 of 20 in
    `seed/mist.md` (the entire persona layer). These tests run against the
    actual `mist-memory/seed/*.md` source, not a synthetic fixture, so a
    regression back to raw-id searching is caught here even if every
    fixture-based test in `TestNegationProximity` kept passing.
    """

    def test_a_non_trivial_number_of_real_facts_are_reachable(self):
        """Quantitative regression guard. `check_negation_proximity(...).passed`
        was already, uselessly, `True` before the C1 fix -- a gate that
        examines nothing still reports success. This asserts on the thing
        that actually changed: whether each fact's resolved search term is
        found in its document's body at all, which is what makes the scan
        loop run in the first place. A drop back toward the pre-fix count
        (4/30, 0/20 in mist.md) means the raw-id regression is back.
        """
        documents = load_seed_documents(_REAL_SEED_DIR)
        node_by_id = _node_by_id(documents)

        total = 0
        reachable = 0
        per_doc: dict[str, int] = {}
        for doc in documents:
            body_lower = doc.body.lower()
            doc_reachable = 0
            for fact in doc.facts:
                total += 1
                term = _search_term_for(fact.object, node_by_id).lower()
                if term and term in body_lower:
                    reachable += 1
                    doc_reachable += 1
            per_doc[str(doc.source_path)] = doc_reachable

        assert total >= 25, "seed source shrank enough to invalidate this test's assumptions"
        assert reachable >= 20, (
            f"only {reachable}/{total} facts reachable -- expected display-name "
            "resolution to make nearly all of them scannable; a drop toward the "
            "pre-fix count (4/30) means the raw-id regression is back"
        )
        mist_doc = next(p for p in per_doc if p.endswith("mist.md"))
        assert per_doc[mist_doc] >= 15, (
            f"seed/mist.md (the persona layer) was 0/20 reachable before the C1 fix; "
            f"now {per_doc[mist_doc]}, expected most of it reachable"
        )

    def test_passes_on_the_unmodified_real_source(self):
        """Sanity: the fix must not introduce false positives against real,
        unmodified content.
        """
        documents = load_seed_documents(_REAL_SEED_DIR)

        result = check_negation_proximity(documents)

        assert result.passed, result.failures

    def test_flags_a_negation_planted_next_to_a_real_display_name(self):
        """The mutation proof: take a real document, plant a real negation
        marker directly next to a real fact's display name exactly as it
        appears in the real prose, and confirm the gate fires. The marker
        sits next to `**Transparent**`, which is what the fixed search term
        resolves to; the pre-fix search term was `trait-transparent`, absent
        from the body entirely, so this exact plant would have passed
        silently under the old implementation.
        """
        documents = load_seed_documents(_REAL_SEED_DIR)
        doc = next(d for d in documents if d.partition == SELF_MODEL_LABEL)
        assert "- **Transparent**" in doc.body, "seed content changed; update this plant"

        poisoned_body = doc.body.replace("- **Transparent**", "- no longer **Transparent**", 1)
        poisoned_doc = doc.model_copy(update={"body": poisoned_body})

        result = check_negation_proximity([poisoned_doc])

        assert not result.passed
        assert any("trait-transparent" in f for f in result.failures)


def _embedding_of(text: str) -> list[float]:
    """The vector `FakeEmbeddingGenerator` produces for `text`.

    Built with a throwaway generator so arranging a graph row never
    pollutes the `calls` list of the generator injected into the gate --
    several tests below assert on exactly what text the gate asked to
    embed, and a shared instance would make those assertions read back
    the arrange step's own call.
    """
    return FakeEmbeddingGenerator().generate_embedding(text)


def _router_by_id(rows_by_id: dict[str, list[dict]]):
    """Route the gate's per-node read to a per-node response.

    The gate issues one byte-identical parameterized query per node, so
    the connection fake's pattern-keyed `query_responses` cannot
    distinguish them -- only the bound `id` differs. Map an id to `[]` to
    simulate a node absent from the graph.
    """

    def route(_query: str, params: dict | None) -> list[dict] | None:
        return rows_by_id.get((params or {}).get("id"))

    return route


class TestEmbeddings:
    """I7: the fifth gate. Two live losses of seed embeddings have already
    happened and nothing in the codebase could see either, because
    `canonical_serialize` excludes `embedding` from the canonical form --
    every determinism and equality check is byte-identical whether
    embeddings are present, absent, or all-zero.

    Every failing case below is a state the live graph can actually reach:
    a node wiped and recreated without its vector, a `--no-embeddings`
    seed, a backfill that raised after the graph writes committed, a
    dimension drift from an `EMBEDDING_DIMENSION` change, the
    `generate_embedding` empty-text branch that returns `[0.0] * 384`, and
    -- the one no presence-based check can reach -- a node whose authored
    text changed while its vector did not.
    """

    def test_fails_when_the_node_is_absent_from_the_graph(self, fake_connection):
        docs = [_doc(nodes=[SeedNode(id="slalom", type="Organization", display_name="Slalom")])]

        result = check_embeddings(
            fake_connection,
            docs,
            seed_version="profile-v1",
            embedding_generator=FakeEmbeddingGenerator(),
            expected_dimension=EMBEDDING_DIMENSION,
        )

        assert not result.passed
        assert "slalom" in result.failures[0]

    def test_fails_when_the_stored_embedding_is_null(self):
        """The `--no-embeddings` seed, and the post-wipe loss mode: the node
        is present and correctly labeled, so Gate 2 and the node-definition
        gate both pass, and it has no vector at all.
        """
        connection = FakeNeo4jConnection(query_results=[{"embedding": None}])
        docs = [_doc(nodes=[SeedNode(id="slalom", type="Organization", display_name="Slalom")])]

        result = check_embeddings(
            connection,
            docs,
            seed_version="profile-v1",
            embedding_generator=FakeEmbeddingGenerator(),
            expected_dimension=EMBEDDING_DIMENSION,
        )

        assert not result.passed
        assert "slalom" in result.failures[0]
        assert "null" in result.failures[0]

    def test_fails_when_the_stored_embedding_has_the_wrong_dimension(self):
        """`EMBEDDING_DIMENSION` drift: the model was swapped without a
        reindex, so old vectors survive at the old width.

        Pins the dimension check's own message shape, not merely the two
        numbers in it. The generator here is the default 384-d one, so the
        LATER `len(recomputed) != len(stored)` condition would also fire on
        this input -- and its message names `384` and `128` too, so bare
        substring assertions on those numbers are satisfied whichever
        condition produced the line. That is how `!=` -> `>` stayed green:
        this test passed while the condition it is named for never ran.
        """
        connection = FakeNeo4jConnection(query_results=[{"embedding": [0.1] * 128}])
        docs = [_doc(nodes=[SeedNode(id="slalom", type="Organization", display_name="Slalom")])]

        result = check_embeddings(
            connection,
            docs,
            seed_version="profile-v1",
            embedding_generator=FakeEmbeddingGenerator(),
            expected_dimension=EMBEDDING_DIMENSION,
        )

        assert not result.passed
        assert "dimensions, expected" in result.failures[0], (
            "the failure line did not come from the dimension check -- a later "
            f"condition produced it: {result.failures[0]!r}"
        )
        assert "128" in result.failures[0]
        assert str(EMBEDDING_DIMENSION) in result.failures[0]

    def test_fails_when_the_stored_embedding_is_wider_than_configured(self):
        """The direction the narrow case cannot cover, and the one that shipped
        a hole.

        `test_fails_when_the_stored_embedding_has_the_wrong_dimension` feeds a
        vector NARROWER than `expected_dimension`. In that direction the later
        `len(recomputed) != len(stored)` guard fires too, so the dimension
        check is never the only thing standing -- and mutating `!=` to `<` in
        that check left the ENTIRE suite byte-identical.

        Wide-and-self-consistent is the live scenario: `EMBEDDING_MODEL` is
        upgraded to a 768-d model while `EMBEDDING_DIMENSION` stays 384 (two
        independent env vars). Stored vectors are 768 wide, the generator
        recomputes 768 wide, so the length-agreement check passes and cosine is
        1.0. Only the dimension check can see it, and only if it compares for
        INEQUALITY rather than for "too small".

        Consequence if it passes: the gate reports PASS on a graph whose every
        vector is the wrong width for the configured dimension and for the
        vector index built at 384.
        """

        class _WideGenerator:
            """Returns a 768-d vector identical to the stored one, so every
            downstream check agrees and only the dimension check can fire.
            """

            def generate_embedding(self, text: str) -> list[float]:
                return [0.1] * 768

        connection = FakeNeo4jConnection(query_results=[{"embedding": [0.1] * 768}])
        docs = [_doc(nodes=[SeedNode(id="slalom", type="Organization", display_name="Slalom")])]

        result = check_embeddings(
            connection,
            docs,
            seed_version="profile-v1",
            embedding_generator=_WideGenerator(),
            expected_dimension=EMBEDDING_DIMENSION,
        )

        assert not result.passed, (
            "a 768-d vector passed a gate configured for 384 -- the dimension "
            "check is comparing magnitude rather than inequality, so an "
            "embedding-model upgrade without a reindex reports healthy"
        )
        assert "768" in result.failures[0]
        assert str(EMBEDDING_DIMENSION) in result.failures[0]

    def test_fails_when_the_stored_embedding_is_narrower_than_configured_and_self_consistent(self):
        """The mirror of the wide case, and the mutant the wide case leaves alive.

        Adding the wide test killed `!=` -> `<`, but `!=` -> `>` survived the
        whole suite afterwards. Both surviving operators are the ones that are
        FALSE on equality (`<`, `>`, `!=`); every operator that is TRUE on
        equality (`<=`, `>=`, `==`) fires on healthy 384-d vectors and breaks
        the passing tests loudly, so those need no guarding.

        `test_fails_when_the_stored_embedding_has_the_wrong_dimension` cannot
        kill `>`, because it injects a 384-d generator against a 128-d stored
        vector: with `>`, the dimension check falls through and the later
        `len(recomputed) != len(stored)` condition fires instead. The gate
        still fails, so the test still passes -- by way of a condition it was
        not written to exercise.

        Self-consistency is what removes that backstop. The live scenario is
        `EMBEDDING_MODEL` swapped DOWN (a 128-d or 256-d model) while
        `EMBEDDING_DIMENSION` stays 384: stored and recomputed are both narrow,
        so they agree, cosine is 1.0, and only an inequality comparison can see
        it. Same defect as the wide case, opposite direction -- a graph whose
        every vector is the wrong width for the vector index reports healthy.
        """
        narrow_dimension = 128
        stored = FakeEmbeddingGenerator(dimension=narrow_dimension).generate_embedding("Slalom")
        connection = FakeNeo4jConnection(query_results=[{"embedding": stored}])
        docs = [_doc(nodes=[SeedNode(id="slalom", type="Organization", display_name="Slalom")])]

        result = check_embeddings(
            connection,
            docs,
            seed_version="profile-v1",
            embedding_generator=FakeEmbeddingGenerator(dimension=narrow_dimension),
            expected_dimension=EMBEDDING_DIMENSION,
        )

        assert not result.passed, (
            "a 128-d vector passed a gate configured for 384 -- the stored vector "
            "and the recomputation agree with each other, so cosine is 1.0 and the "
            "dimension check is the only condition that can see the drift"
        )
        assert "dimensions, expected" in result.failures[0], (
            "the failure line did not come from the dimension check -- a later "
            f"condition produced it: {result.failures[0]!r}"
        )
        assert str(narrow_dimension) in result.failures[0]
        assert str(EMBEDDING_DIMENSION) in result.failures[0]

    def test_fails_when_the_stored_embedding_is_not_a_list(self):
        """The `isinstance` branch had zero test reach.

        What removing the branch actually does, measured rather than assumed:
        a `str` does NOT raise `TypeError` inside `len()` -- it is Sized, so
        `len("not-a-vector")` is 12 and the dimension check reports
        `"has an embedding of 12 dimensions, expected 384"`. A confidently
        wrong line about the wrong condition is worse than a traceback,
        because it sends the reader after a dimension drift that never
        happened. Only a non-Sized value (a float, an int) tracebacks.

        Low likelihood either way -- the driver returns lists for array
        properties -- but the branch existed unexercised, so deleting it was
        invisible to the suite. The assertion below is on the type name, which
        is the one thing no other condition's message can produce.
        """
        connection = FakeNeo4jConnection(query_results=[{"embedding": "not-a-vector"}])
        docs = [_doc(nodes=[SeedNode(id="slalom", type="Organization", display_name="Slalom")])]

        result = check_embeddings(
            connection,
            docs,
            seed_version="profile-v1",
            embedding_generator=FakeEmbeddingGenerator(),
            expected_dimension=EMBEDDING_DIMENSION,
        )

        assert not result.passed
        assert "str" in result.failures[0]

    def test_fails_when_the_stored_embedding_is_a_zero_vector(self):
        """`EmbeddingGenerator.generate_embedding` returns `[0.0] * 384` for
        empty text. That vector is the right width and is not null, so only
        a norm check can see it -- and it matches nothing at query time.
        """
        connection = FakeNeo4jConnection(query_results=[{"embedding": [0.0] * EMBEDDING_DIMENSION}])
        docs = [_doc(nodes=[SeedNode(id="slalom", type="Organization", display_name="Slalom")])]

        result = check_embeddings(
            connection,
            docs,
            seed_version="profile-v1",
            embedding_generator=FakeEmbeddingGenerator(),
            expected_dimension=EMBEDDING_DIMENSION,
        )

        assert not result.passed
        assert "zero vector" in result.failures[0]

    def test_fails_when_the_stored_embedding_was_computed_from_different_text(self):
        """THE condition this gate exists for, and the only one that catches
        the live-reachable mode nothing else can see: `_WIPE_NODES` spares a
        seeded node that has acquired a conversation-derived edge, the
        applier's `ON MATCH SET n += $properties` refreshes its
        `display_name`/`description` but not its `embedding`, and the
        backfill then skips it because `WHERE n.embedding IS NULL` is false.
        The node keeps a vector computed from text that is no longer
        authored anywhere -- present, 384-d, unit-norm, and wrong.
        """
        connection = FakeNeo4jConnection(
            query_results=[{"embedding": _embedding_of("Slalom — Consulting")}]
        )
        docs = [
            _doc(
                nodes=[
                    SeedNode(
                        id="slalom",
                        type="Organization",
                        display_name="Slalom",
                        description="Consulting firm",  # the source text moved on
                    )
                ]
            )
        ]

        result = check_embeddings(
            connection,
            docs,
            seed_version="profile-v1",
            embedding_generator=FakeEmbeddingGenerator(),
            expected_dimension=EMBEDDING_DIMENSION,
        )

        assert not result.passed
        assert "slalom" in result.failures[0]
        assert "cosine" in result.failures[0]

    def test_passes_when_the_stored_embedding_matches_the_authored_source_text(self):
        connection = FakeNeo4jConnection(
            query_results=[
                {"embedding": _embedding_of(embedding_text_for("Slalom", "Consulting firm", "s"))}
            ]
        )
        docs = [
            _doc(
                nodes=[
                    SeedNode(
                        id="slalom",
                        type="Organization",
                        display_name="Slalom",
                        description="Consulting firm",
                    )
                ]
            )
        ]

        result = check_embeddings(
            connection,
            docs,
            seed_version="profile-v1",
            embedding_generator=FakeEmbeddingGenerator(),
            expected_dimension=EMBEDDING_DIMENSION,
        )

        assert result.passed, result.failures
        assert result.examined == 1

    def test_recomputes_from_the_authored_source_not_from_the_graph(self):
        """Gate 2's rationale applied to vectors: the authored file is a
        reference that cannot be vacuously satisfied. A gate that recomputed
        from the graph's own `display_name` would agree with itself by
        construction whenever the applier and the backfill were both wrong
        in the same way -- and stale text is exactly the case where the
        graph's properties are RIGHT and its vector is stale, so reading
        them back would make the comparison a tautology.

        Pinned two ways: the read query returns nothing but the vector (so
        graph-sourced text is not merely unused, it is unavailable), and the
        text handed to the generator is the one built from the authored
        `SeedNode`.
        """
        connection = FakeNeo4jConnection(
            query_results=[
                {"embedding": _embedding_of(embedding_text_for("Slalom", "Consulting firm", "s"))}
            ]
        )
        generator = FakeEmbeddingGenerator()
        docs = [
            _doc(
                nodes=[
                    SeedNode(
                        id="slalom",
                        type="Organization",
                        display_name="Slalom",
                        description="Consulting firm",
                    )
                ]
            )
        ]

        check_embeddings(
            connection,
            docs,
            seed_version="profile-v1",
            embedding_generator=generator,
            expected_dimension=EMBEDDING_DIMENSION,
        )

        query, _params = connection.queries[0]
        assert "RETURN n.embedding AS embedding" in query
        assert "display_name" not in query
        assert "description" not in query
        assert generator.calls == [embedding_text_for("Slalom", "Consulting firm", "slalom")]

    def test_fails_closed_when_the_documents_define_no_nodes(self):
        """R1.4's C1 in one assertion: `check_negation_proximity` reported
        `passed=True` having examined 0 of 20 facts. A gate that examined
        nothing must never report success, however clean the input looked.
        """
        connection = FakeNeo4jConnection(query_results=[])
        docs = [_doc(facts=[("user", "WORKS_AT", "slalom")])]  # facts, but no node definitions

        result = check_embeddings(
            connection,
            docs,
            seed_version="profile-v1",
            embedding_generator=FakeEmbeddingGenerator(),
            expected_dimension=EMBEDDING_DIMENSION,
        )

        assert not result.passed
        assert result.examined == 0
        assert "0 nodes" in result.failures[0]

    def test_fails_closed_on_an_empty_document_list(self):
        connection = FakeNeo4jConnection(query_results=[])

        result = check_embeddings(
            connection,
            [],
            seed_version="profile-v1",
            embedding_generator=FakeEmbeddingGenerator(),
            expected_dimension=EMBEDDING_DIMENSION,
        )

        assert not result.passed
        assert result.examined == 0

    def test_examined_counts_every_node_the_gate_looked_at(self):
        """`examined` is what makes "did it actually look at anything" a
        query rather than a hope. It counts nodes inspected, passing and
        failing alike -- a gate reporting `examined=0` for a source that
        defines nodes is broken even if `failures` is empty.
        """
        connection = FakeNeo4jConnection(
            query_router=_router_by_id(
                {
                    "slalom": [{"embedding": _embedding_of("Slalom")}],
                    "python": [{"embedding": _embedding_of("Python")}],
                }
            )
        )
        docs = [
            _doc(
                nodes=[
                    SeedNode(id="slalom", type="Organization", display_name="Slalom"),
                    SeedNode(id="python", type="Technology", display_name="Python"),
                ]
            )
        ]

        result = check_embeddings(
            connection,
            docs,
            seed_version="profile-v1",
            embedding_generator=FakeEmbeddingGenerator(),
            expected_dimension=EMBEDDING_DIMENSION,
        )

        assert result.passed, result.failures
        assert result.examined == 2

    def test_reports_one_failure_per_bad_node_and_leaves_good_ones_alone(self):
        connection = FakeNeo4jConnection(
            query_router=_router_by_id(
                {
                    "slalom": [{"embedding": _embedding_of("Slalom")}],
                    "python": [{"embedding": _embedding_of("a different node entirely")}],
                }
            )
        )
        docs = [
            _doc(
                nodes=[
                    SeedNode(id="slalom", type="Organization", display_name="Slalom"),
                    SeedNode(id="python", type="Technology", display_name="Python"),
                ]
            )
        ]

        result = check_embeddings(
            connection,
            docs,
            seed_version="profile-v1",
            embedding_generator=FakeEmbeddingGenerator(),
            expected_dimension=EMBEDDING_DIMENSION,
        )

        assert result.examined == 2
        assert len(result.failures) == 1
        assert "python" in result.failures[0]

    def test_never_writes(self):
        """A read-only check -- it must never issue execute_write."""
        connection = FakeNeo4jConnection(query_results=[{"embedding": _embedding_of("Slalom")}])
        docs = [_doc(nodes=[SeedNode(id="slalom", type="Organization", display_name="Slalom")])]

        check_embeddings(
            connection,
            docs,
            seed_version="profile-v1",
            embedding_generator=FakeEmbeddingGenerator(),
            expected_dimension=EMBEDDING_DIMENSION,
        )

        connection.assert_no_writes()

    def test_forwards_id_and_seed_version_as_params(self):
        connection = FakeNeo4jConnection(query_results=[{"embedding": _embedding_of("Slalom")}])
        docs = [_doc(nodes=[SeedNode(id="slalom", type="Organization", display_name="Slalom")])]

        check_embeddings(
            connection,
            docs,
            seed_version="profile-v1",
            embedding_generator=FakeEmbeddingGenerator(),
            expected_dimension=EMBEDDING_DIMENSION,
        )

        assert len(connection.queries) == 1
        _query, params = connection.queries[0]
        assert params["id"] == "slalom"
        assert params["seed_version"] == "profile-v1"

    def test_uses_the_documents_partition_as_the_label_in_the_query_text(self):
        """Neo4j cannot parameterize a label, and the two partitions cannot
        see each other: a read restricted to `:__Entity__` returns no bind
        for a `:__SelfModel__` node, which would report every persona node
        -- the entire 21-node self-model layer, all of it embedded -- as
        having no embedding.
        """
        connection = FakeNeo4jConnection(query_results=[{"embedding": _embedding_of("Warm")}])
        docs = [
            _doc(
                nodes=[SeedNode(id="trait-warm", type="MistTrait", display_name="Warm")],
                partition=SELF_MODEL_LABEL,
            )
        ]

        check_embeddings(
            connection,
            docs,
            seed_version="profile-v1",
            embedding_generator=FakeEmbeddingGenerator(),
            expected_dimension=EMBEDDING_DIMENSION,
        )

        query, _params = connection.queries[0]
        assert f"MATCH (n:{SELF_MODEL_LABEL} {{id: $id}})" in query

    def test_filters_on_seed_version_in_the_where_clause(self):
        """Pins the exact filtering clause, not merely that the token
        `seed_version` appears somewhere in the query -- the hole Task 5
        found in the wipe query.
        """
        connection = FakeNeo4jConnection(query_results=[{"embedding": _embedding_of("Slalom")}])
        docs = [_doc(nodes=[SeedNode(id="slalom", type="Organization", display_name="Slalom")])]

        check_embeddings(
            connection,
            docs,
            seed_version="profile-v1",
            embedding_generator=FakeEmbeddingGenerator(),
            expected_dimension=EMBEDDING_DIMENSION,
        )

        query, _params = connection.queries[0]
        assert "WHERE n.seed_version = $seed_version" in query

    def test_falls_back_to_the_bare_id_when_the_node_defines_no_display_name(self):
        """The fallback path is legitimate (a node reseed() recreated fresh)
        but must be reached deliberately: the gate embeds `node.id`, not an
        empty string, which `generate_embedding` would turn into the zero
        vector this same gate rejects.
        """
        connection = FakeNeo4jConnection(
            query_results=[{"embedding": _embedding_of("mist-identity")}]
        )
        generator = FakeEmbeddingGenerator()
        docs = [
            _doc(
                nodes=[SeedNode(id="mist-identity", type="MistIdentity")],  # no display_name
                partition=SELF_MODEL_LABEL,
            )
        ]

        result = check_embeddings(
            connection,
            docs,
            seed_version="profile-v1",
            embedding_generator=generator,
            expected_dimension=EMBEDDING_DIMENSION,
        )

        assert result.passed, result.failures
        assert generator.calls == ["mist-identity"]


def _consistent_graph_rows(documents: list[SeedDocument]) -> dict[str, list[dict]]:
    """A graph whose stored vectors agree with what `documents` authors.

    Models exactly what `_backfill_embeddings_for_seed` writes: the
    vector of `embedding_text_for(...)` for each authored node. Building
    the graph side through the same builder the gate recomputes with is
    deliberate but does have one blind spot worth naming -- a bug INSIDE
    `embedding_text_for` moves both sides together and is invisible here.
    That is precisely what `tests/unit/knowledge/test_embedding_text.py`
    pins independently, by codepoint. What this fixture DOES establish is
    the property that matters for the gate: authored text and stored
    vector are two separate artifacts, and the mutation proof below moves
    one without the other.
    """
    rows: dict[str, list[dict]] = {}
    for doc in documents:
        for node in doc.nodes:
            text = embedding_text_for(
                getattr(node, "display_name", None), getattr(node, "description", None), node.id
            )
            rows[node.id] = [{"embedding": _embedding_of(text)}]
    return rows


class TestEmbeddingGateRealSource:
    """The same discipline `TestNegationProximityRealSource` established, for
    the embedding gate: fixtures prove a gate CAN fire, only real source
    proves it DOES reach real content. C1 passed every fixture-based test
    it had while examining 0 of 20 real facts.

    `FakeEmbeddingGenerator` is the right double here despite its docstring
    warning that it is unsuitable for similarity-threshold testing. That
    warning is about distinguishing SEMANTICALLY near texts, which this
    gate never does: condition 4 compares a vector against a recomputation
    from identical input text, and identical input yields cosine exactly
    1.0 under any deterministic generator, SHA-256-derived or otherwise.
    Loading the real MiniLM model would add several seconds and an
    external dependency to the unit tier (budget: whole suite under 30s,
    no external deps) and would prove nothing this does not. Do not
    "fix" this to use the real generator.
    """

    def test_examines_every_node_the_real_source_defines(self):
        """The quantitative floor. `mist-memory/seed/` currently defines 32
        nodes (11 in `user.md`, 21 in `mist.md`) -- the exact count the live
        graph holds, all 32 embedded. A floor rather than an equality so
        adding a node does not break this, but a gate that stops reaching
        real nodes -- the C1 failure mode, which was silent -- fails loudly
        here.
        """
        documents = load_seed_documents(_REAL_SEED_DIR)
        connection = FakeNeo4jConnection(
            query_router=_router_by_id(_consistent_graph_rows(documents))
        )

        result = check_embeddings(
            connection,
            documents,
            seed_version="profile-v1",
            embedding_generator=FakeEmbeddingGenerator(),
            expected_dimension=EMBEDDING_DIMENSION,
        )

        assert result.examined >= 32, (
            f"gate examined only {result.examined} real nodes; the real seed source "
            "defines 32 (11 in user.md, 21 in mist.md). A drop means the gate has "
            "stopped reaching real content -- the C1 failure mode, which reported "
            "passed=True the whole time it was examining nothing"
        )

    def test_passes_against_unmodified_real_source(self):
        """Sanity: no false positives against real content whose stored
        vectors agree with it. Without this, a gate that failed everything
        would satisfy the mutation proof below just as well.
        """
        documents = load_seed_documents(_REAL_SEED_DIR)
        connection = FakeNeo4jConnection(
            query_router=_router_by_id(_consistent_graph_rows(documents))
        )

        result = check_embeddings(
            connection,
            documents,
            seed_version="profile-v1",
            embedding_generator=FakeEmbeddingGenerator(),
            expected_dimension=EMBEDDING_DIMENSION,
        )

        assert result.passed, result.failures

    def test_every_real_node_authors_its_own_display_name(self):
        """`embedding_text_for`'s fallback to the bare id is legitimate but
        must be reached deliberately, never by accident: all 32 real nodes
        author a `display_name` today, so any node whose embedded text is
        just its kebab id is a source-authoring defect, not a design
        choice. Also pins that the 32 texts are distinct -- matching the
        live graph's verified 32 distinct vectors. Two nodes embedding the
        same text would be indistinguishable to vector retrieval.
        """
        documents = load_seed_documents(_REAL_SEED_DIR)

        texts = [
            embedding_text_for(
                getattr(node, "display_name", None), getattr(node, "description", None), node.id
            )
            for doc in documents
            for node in doc.nodes
        ]
        bare_ids = [
            node.id
            for doc in documents
            for node in doc.nodes
            if embedding_text_for(
                getattr(node, "display_name", None), getattr(node, "description", None), node.id
            )
            == node.id
        ]

        assert bare_ids == [], f"real nodes fell back to the bare id: {bare_ids}"
        assert len(set(texts)) == len(texts), "two real nodes embed identical text"

    def test_a_display_name_edit_that_left_the_vector_behind_fires_on_that_node_alone(self):
        """The mutation proof, and the whole reason this gate exists: change
        one real node's authored text, leave the graph's vector where it
        was, and the gate must name that node and only that node.

        This is the live-reachable loss mode from `_WIPE_NODES` sparing a
        node with a conversation-derived edge -- `ON MATCH SET` refreshes
        the properties, the backfill's `WHERE n.embedding IS NULL` guard
        skips it, and the vector goes stale in place. Every other gate
        passes on this graph: the node is present, correctly labeled, has a
        non-null `display_name`, its facts are intact, and its embedding is
        384-d and unit-norm.
        """
        documents = load_seed_documents(_REAL_SEED_DIR)
        rows = _consistent_graph_rows(documents)  # the graph as it was BEFORE the edit

        target_doc = next(d for d in documents if any(n.id == "slalom" for n in d.nodes))
        assert any(
            getattr(n, "display_name", None) == "Slalom" for n in target_doc.nodes
        ), "seed content changed; update this plant"
        edited_nodes = [
            (
                n.model_copy(update={"display_name": "Slalom Consulting Group"})
                if n.id == "slalom"
                else n
            )
            for n in target_doc.nodes
        ]
        edited_doc = target_doc.model_copy(update={"nodes": edited_nodes})
        edited_documents = [edited_doc if d is target_doc else d for d in documents]

        result = check_embeddings(
            FakeNeo4jConnection(query_router=_router_by_id(rows)),
            edited_documents,
            seed_version="profile-v1",
            embedding_generator=FakeEmbeddingGenerator(),
            expected_dimension=EMBEDDING_DIMENSION,
        )

        assert not result.passed
        assert len(result.failures) == 1, result.failures
        assert "slalom" in result.failures[0]
        assert "cosine" in result.failures[0]
        assert result.examined >= 32

    def test_real_source_exercises_both_branches_of_the_text_builder(self):
        """`embedding_text_for` has two branches and real source uses both, so
        the real-source proof must reach both.

        `seed/user.md` authors no `description` on any of its 11 nodes;
        `seed/mist.md` authors one on 14 of its 21. Every entity-partition
        node therefore embeds on `display_name` alone while most of the
        self-model embeds `display_name` + separator + description. That
        asymmetry is legitimate authored content, not a defect -- which is
        exactly why this asserts coverage of both branches and NOT that any
        node must carry a description.

        Floors of one, not of the current 18/14 split: a seed edit that adds
        or removes a description is normal authoring and must not fail a
        test. A seed edit that leaves the gate exercising only one branch
        against real content is the C1 shape -- coverage quietly evaporating
        while every test stays green -- and fails here.
        """
        documents = load_seed_documents(_REAL_SEED_DIR)

        # Branch is derived from the builder's own contract -- a text longer
        # than its leading component took the description branch -- rather
        # than by searching for the separator, so this does not become a
        # second place the separator is written down.
        joined: list[str] = []
        name_only: list[str] = []
        for doc in documents:
            for node in doc.nodes:
                display_name = getattr(node, "display_name", None)
                description = getattr(node, "description", None)
                text = embedding_text_for(display_name, description, node.id)
                if text == (display_name or node.id):
                    name_only.append(text)
                else:
                    joined.append(text)

        total = len(joined) + len(name_only)
        assert len(joined) >= 1, (
            "no real node exercises the display_name + description branch "
            f"(observed at authoring time: 14 of mist.md's 21 nodes); {total} nodes total"
        )
        assert len(name_only) >= 1, (
            "no real node exercises the display_name-only branch (observed at "
            f"authoring time: all 11 of user.md's nodes); {total} nodes total"
        )

    def test_a_description_edit_that_left_the_vector_behind_fires_on_that_node_alone(self):
        """The second half of the mutation proof, on the description branch.

        The `display_name` plant above perturbs a `seed/user.md` node, which
        has no description -- so on its own it proves only that the gate
        notices drift in the FIRST component of the embedded text. A gate
        that ignored `description` entirely would survive it. This one
        perturbs the description of a real `seed/mist.md` node and leaves
        its `display_name` untouched, so only the second component moves.
        """
        documents = load_seed_documents(_REAL_SEED_DIR)
        rows = _consistent_graph_rows(documents)  # the graph as it was BEFORE the edit

        target_doc = next(d for d in documents if any(n.id == "trait-transparent" for n in d.nodes))
        target = next(n for n in target_doc.nodes if n.id == "trait-transparent")
        assert getattr(target, "description", None), "seed content changed; update this plant"

        edited_nodes = [
            (
                n.model_copy(update={"description": "Shows its work, including tool calls."})
                if n.id == "trait-transparent"
                else n
            )
            for n in target_doc.nodes
        ]
        edited_doc = target_doc.model_copy(update={"nodes": edited_nodes})
        edited_documents = [edited_doc if d is target_doc else d for d in documents]

        result = check_embeddings(
            FakeNeo4jConnection(query_router=_router_by_id(rows)),
            edited_documents,
            seed_version="profile-v1",
            embedding_generator=FakeEmbeddingGenerator(),
            expected_dimension=EMBEDDING_DIMENSION,
        )

        assert not result.passed
        assert len(result.failures) == 1, result.failures
        assert "trait-transparent" in result.failures[0]
        assert "cosine" in result.failures[0]
        assert result.examined >= 32
