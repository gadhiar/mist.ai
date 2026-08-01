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
from backend.knowledge.seed.models import SeedDocument, SeedFact, SeedNode
from backend.knowledge.storage.partitions import ENTITY_LABEL, SELF_MODEL_LABEL

_NOW = "2026-07-31T00:00:00+00:00"

# R1.4 Task 12 (addendum): apply_seed_documents now requires every fact's
# subject/object to have a matching SeedNode -- the applier's own defense
# for a caller that constructs SeedDocuments directly, mirroring Task 11's
# loader-level referential integrity. This placeholder type is used for
# _doc()'s auto-generated nodes; tests that care about node type/property
# content pass `nodes=` explicitly (see TestNodeDefinitionWrites).
_PLACEHOLDER_TYPE = "Concept"


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

    When `nodes` is omitted, a `SeedNode` (placeholder type) is auto-generated
    for every unique subject/object referenced by `facts`, so tests exercising
    fact/partition/predicate behavior do not each need to author their own
    `nodes:` block. Pass `nodes` explicitly to control type/properties.
    """
    fact_objs = [SeedFact(subject=s, predicate=p, object=o) for s, p, o in (facts or [])]
    if nodes is not None:
        node_objs = nodes
    else:
        node_ids = sorted({n for f in fact_objs for n in (f.subject, f.object)})
        node_objs = [SeedNode(id=i, type=_PLACEHOLDER_TYPE) for i in node_ids]
    return SeedDocument(
        seed_version=version,
        nodes=node_objs,
        facts=fact_objs,
        body=body,
        source_path=source_path,
        partition=partition,
    )


def _seed_version_param(params: dict) -> object:
    """Extract seed_version from a write's params, wherever it lives.

    Edge writes (`_MERGE_EDGE`) carry it as a top-level param. Node writes
    (`_MERGE_NODE`, R1.4 Task 12) carry it inside the `properties` map,
    matching `admin.py`'s established `merge_params` shape. Returns
    `None` if present in neither location.
    """
    if "seed_version" in params:
        return params["seed_version"]
    return params.get("properties", {}).get("seed_version")


class TestSeedVersionStamping:
    def test_stamps_every_written_node_and_edge_with_seed_version(self, fake_connection):
        docs = [_doc(version="profile-v1", facts=[("user", "WORKS_AT", "slalom")])]

        apply_seed_documents(fake_connection, docs, seed_version="profile-v1", now_iso=_NOW)

        assert fake_connection.writes, "expected at least one write"
        for query, params in fake_connection.writes:
            assert _seed_version_param(params) == "profile-v1"
            # A fake connection records whatever params are passed regardless of
            # whether the query text uses them, so checking params alone cannot
            # catch a query that carries the value but never applies it. Edge
            # writes SET seed_version with a literal clause; node writes (R1.4
            # Task 12) merge it via `n += $properties` (admin.py's established
            # map-merge shape, which has no per-field literal to pin) -- check
            # whichever mechanism this specific write actually uses.
            if "seed_version" in params:  # edge write: literal SET
                assert "seed_version" in query, f"query does not set seed_version: {query!r}"
            else:  # node write: map-merge
                assert "n += $properties" in query, f"query does not merge properties: {query!r}"

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
            assert _seed_version_param(params) == "profile-v2"


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
                seed_version="profile-v1",
                nodes=[
                    SeedNode(id="user", type=_PLACEHOLDER_TYPE),
                    SeedNode(id="slalom", type=_PLACEHOLDER_TYPE),
                ],
                facts=[fact],
                body="b",
                source_path=Path("t.md"),
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


# ---------------------------------------------------------------------------
# R1.4 Task 12 (addendum): node definitions written to the graph
#
# `_MERGE_NODE` used to set only `seed_version`/`created_at`/`updated_at`.
# Task 10's live run proved that a wipe-and-recreate cycle then strips every
# ontology label and descriptive property, because MERGE preserves untouched
# properties on a MATCH but a fresh CREATE gets nothing beyond what the query
# explicitly sets. THE ROUND-TRIP TEST BELOW IS THE HEADLINE: "a write
# occurred" is precisely the assertion that let the original defect ship --
# every test here asserts the WRITTEN properties equal the SOURCE properties,
# not merely that execute_write was called.
# ---------------------------------------------------------------------------


class TestNodeDefinitionWrites:
    def test_round_trip_every_written_property_matches_the_source(self, fake_connection):
        """THE headline assertion of this task. Every property on the source
        `SeedNode` -- not a sample, all of them -- must appear in the written
        `properties` map with the exact source value. This is the test that
        would have failed loudly against the pre-Task-12 applier: the old
        `_MERGE_NODE` wrote none of display_name/description/pronouns, so
        this assertion would have found an empty properties dict where a
        full one belongs.
        """
        docs = [
            _doc(
                facts=[("mist-identity", "HAS_TRAIT", "trait-warm")],
                partition=SELF_MODEL_LABEL,
                nodes=[
                    SeedNode(
                        id="mist-identity",
                        type="MistIdentity",
                        display_name="MIST",
                        pronouns="she/her",
                        self_concept="a cognitive architecture",
                    ),
                    SeedNode(
                        id="trait-warm",
                        type="MistTrait",
                        display_name="Warm",
                        axis="Persona",
                        description="Default register is warm and engaged.",
                    ),
                ],
            )
        ]

        apply_seed_documents(fake_connection, docs, seed_version="profile-v1", now_iso=_NOW)

        node_writes = [(q, p) for q, p in fake_connection.writes if not p.get("predicate")]
        by_id = {p["id"]: p["properties"] for _q, p in node_writes}

        assert by_id["mist-identity"] == {
            "entity_type": "MistIdentity",
            "seed_version": "profile-v1",
            "updated_at": _NOW,
            "display_name": "MIST",
            "pronouns": "she/her",
            "self_concept": "a cognitive architecture",
        }
        assert by_id["trait-warm"] == {
            "entity_type": "MistTrait",
            "seed_version": "profile-v1",
            "updated_at": _NOW,
            "display_name": "Warm",
            "axis": "Persona",
            "description": "Default register is warm and engaged.",
        }

    def test_sets_the_ontology_type_as_a_graph_label(self, fake_connection):
        docs = [
            _doc(
                facts=[("slalom", "WORKS_ON", "mist-ai")],
                nodes=[
                    SeedNode(id="slalom", type="Organization"),
                    SeedNode(id="mist-ai", type="Project"),
                ],
            )
        ]

        apply_seed_documents(fake_connection, docs, seed_version="profile-v1", now_iso=_NOW)

        node_writes = [(q, p) for q, p in fake_connection.writes if not p.get("predicate")]
        by_id = {p["id"]: q for q, p in node_writes}
        assert "SET n:Organization" in by_id["slalom"]
        assert "SET n:Project" in by_id["mist-ai"]

    def test_properties_are_merged_not_set_field_by_field(self, fake_connection):
        """`n += $properties` (admin.py's established shape) is what makes
        re-seeding enforce the source as ground truth for every property it
        defines while leaving properties the applier does not own (e.g.
        `embedding`) untouched. Pin the merge clause itself, not just that
        SOME node write happened.
        """
        docs = [_doc(facts=[("user", "WORKS_AT", "slalom")])]

        apply_seed_documents(fake_connection, docs, seed_version="profile-v1", now_iso=_NOW)

        node_writes = [(q, p) for q, p in fake_connection.writes if not p.get("predicate")]
        for query, _params in node_writes:
            assert "ON CREATE SET n.created_at = $now, n += $properties" in query
            assert "ON MATCH SET n += $properties" in query

    def test_created_at_is_create_only(self, fake_connection):
        """`created_at` must appear only in the ON CREATE branch -- re-seeding
        an existing node must not overwrite its original creation timestamp,
        mirroring admin.py's `_seed_internal_nodes` create-only field.
        """
        docs = [_doc(facts=[("user", "WORKS_AT", "slalom")])]

        apply_seed_documents(fake_connection, docs, seed_version="profile-v1", now_iso=_NOW)

        node_writes = [(q, p) for q, p in fake_connection.writes if not p.get("predicate")]
        for query, _params in node_writes:
            on_create, _, rest = query.partition("ON MATCH")
            assert "n.created_at = $now" in on_create
            assert "n.created_at" not in rest

    def test_entity_type_property_matches_the_interpolated_label(self, fake_connection):
        """`entity_type` is stored as a PROPERTY in addition to the graph
        LABEL (`SET n:{type}`) -- mirrors admin.py's `_seed_internal_nodes`
        (`merge_params = {"entity_type": label, ...}`), since some readers
        (e.g. `count_nodes_by_type`) query the property, not `labels(n)`.
        """
        docs = [
            _doc(
                facts=[("neo4j", "USES", "python")],
                nodes=[
                    SeedNode(id="neo4j", type="Technology"),
                    SeedNode(id="python", type="Technology"),
                ],
            )
        ]

        apply_seed_documents(fake_connection, docs, seed_version="profile-v1", now_iso=_NOW)

        node_writes = [(q, p) for q, p in fake_connection.writes if not p.get("predicate")]
        for _query, params in node_writes:
            assert params["properties"]["entity_type"] == "Technology"


class TestAuthoredStampsNeverWinOverTheAppliersOwn:
    """R1.4 whole-branch review, I4: `SeedNode.extra="allow"` let an authored
    property share a name with one of the applier's own bookkeeping stamps
    (`entity_type`/`seed_version`/`updated_at`/`created_at`). The original
    `properties` dict literal spread the authored extras LAST, so an
    authored `seed_version` silently won over the applier's own -- for
    `seed_version` specifically, that makes the node un-wipeable by any
    future `wipe_seed_version` call scoped on the real version, permanent
    graph litter.

    `SeedNode._no_applier_owned_extras` (models.py) now rejects these four
    names as extras at construction time, for every normal caller (the
    loader's `SeedNode(**n)` included). The tests below prove the applier's
    OWN ordering is an independent second layer, not a decoration on top of
    that validator: they build the poisoned node via `SeedNode.
    model_construct`, which bypasses Pydantic validation entirely (the
    model's own documented escape hatch for exactly this "what if an
    upstream invariant were violated" scenario) -- constructing this input
    through ordinary `SeedNode(...)` is no longer possible at all, which is
    itself a demonstration that the model-level guard works.

    `SeedDocument` is ALSO built via `model_construct` in these two tests,
    not ordinary construction: `SeedDocument(nodes=[...])` re-validates
    each list item even when it is already a `SeedNode` instance (pydantic-
    core's handling of a `list[Model]` field re-runs item validation
    regardless of `revalidate_instances`), which would re-trigger
    `_no_applier_owned_extras` on `poisoned_node` at the OUTER construction
    site and defeat the point of building it unvalidated in the first
    place.
    """

    def test_authored_stamps_never_win_over_the_appliers_own_values(self, fake_connection):
        poisoned_node = SeedNode.model_construct(
            id="user",
            type="User",
            display_name="Raj Gadhia",
            entity_type="Bogus",
            seed_version="evil-version",
            updated_at="1999-01-01T00:00:00+00:00",
            created_at="1999-01-01T00:00:00+00:00",
        )
        docs = [
            SeedDocument.model_construct(
                seed_version="profile-v1",
                nodes=[poisoned_node, SeedNode(id="python", type="Technology")],
                facts=[SeedFact(subject="user", predicate="USES", object="python")],
                body="b",
                source_path=Path("t.md"),
                partition=ENTITY_LABEL,
            )
        ]

        apply_seed_documents(fake_connection, docs, seed_version="profile-v1", now_iso=_NOW)

        node_writes = [(q, p) for q, p in fake_connection.writes if not p.get("predicate")]
        by_id = {p["id"]: p["properties"] for _q, p in node_writes}
        user_props = by_id["user"]

        assert user_props["entity_type"] == "User", user_props
        assert user_props["seed_version"] == "profile-v1", user_props
        assert user_props["updated_at"] == _NOW, user_props
        # A legitimate extra property (not one of the four reserved names)
        # must still flow through untouched -- this is not a lockdown of
        # extra="allow", only of the four applier-owned names.
        assert user_props["display_name"] == "Raj Gadhia"

    def test_authored_created_at_never_enters_the_properties_map_at_all(self, fake_connection):
        """`created_at` is a stricter case than the other three: it must not
        merely lose a values comparison, it must never be a KEY in
        `properties` at all, because `properties` is merged via `n +=
        $properties` on both the ON CREATE and ON MATCH branches -- an
        authored `created_at` reaching that map would overwrite the real
        creation timestamp on every future re-seed, not just the first
        write, corrupting `_MERGE_NODE`'s create-only guarantee for that
        field.
        """
        poisoned_node = SeedNode.model_construct(
            id="user", type="User", created_at="1999-01-01T00:00:00+00:00"
        )
        docs = [
            SeedDocument.model_construct(
                seed_version="profile-v1",
                nodes=[poisoned_node, SeedNode(id="python", type="Technology")],
                facts=[SeedFact(subject="user", predicate="USES", object="python")],
                body="b",
                source_path=Path("t.md"),
                partition=ENTITY_LABEL,
            )
        ]

        apply_seed_documents(fake_connection, docs, seed_version="profile-v1", now_iso=_NOW)

        node_writes = [(q, p) for q, p in fake_connection.writes if not p.get("predicate")]
        by_id = {p["id"]: p["properties"] for _q, p in node_writes}

        assert "created_at" not in by_id["user"], by_id["user"]


class TestNodeTypeValidation:
    """Mirrors TestPredicateValidation: the type label is interpolated at the
    same Cypher boundary as the predicate, and Task 11's loader-level check
    does not protect a caller that constructs SeedDocuments directly.
    """

    def test_rejects_type_not_in_the_ontology(self, fake_connection):
        docs = [
            _doc(
                facts=[("thing", "USES", "other")],
                nodes=[
                    SeedNode(id="thing", type="NotARealType"),
                    SeedNode(id="other", type="Concept"),
                ],
            )
        ]

        with pytest.raises(SeedSourceError, match="NotARealType"):
            apply_seed_documents(fake_connection, docs, seed_version="profile-v1", now_iso=_NOW)

    def test_rejects_before_any_write_happens(self, fake_connection):
        docs = [
            _doc(
                facts=[("thing", "USES", "other")],
                nodes=[
                    SeedNode(id="thing", type="NotARealType"),
                    SeedNode(id="other", type="Concept"),
                ],
            )
        ]

        with pytest.raises(SeedSourceError):
            apply_seed_documents(fake_connection, docs, seed_version="profile-v1", now_iso=_NOW)

        fake_connection.assert_no_writes()

    def test_error_names_the_source_file(self, fake_connection):
        docs = [
            _doc(
                facts=[("thing", "USES", "other")],
                nodes=[
                    SeedNode(id="thing", type="NotARealType"),
                    SeedNode(id="other", type="Concept"),
                ],
                source_path=Path("users/bad.md"),
            )
        ]

        with pytest.raises(SeedSourceError, match="users/bad.md".replace("/", r"[\\/]")):
            apply_seed_documents(fake_connection, docs, seed_version="profile-v1", now_iso=_NOW)

    def test_error_suggests_the_closest_allowed_type_on_a_near_match(self, fake_connection):
        docs = [
            _doc(
                facts=[("thing", "USES", "other")],
                nodes=[
                    SeedNode(id="thing", type="Organizaton"),
                    SeedNode(id="other", type="Concept"),
                ],
            )
        ]

        with pytest.raises(SeedSourceError, match="Organization"):
            apply_seed_documents(fake_connection, docs, seed_version="profile-v1", now_iso=_NOW)

    @pytest.mark.parametrize(
        "node_type",
        [
            pytest.param("MistIdentity", id="mist-identity"),
            pytest.param("MistTrait", id="mist-trait"),
            pytest.param("MistCapability", id="mist-capability"),
            pytest.param("MistPreference", id="mist-preference"),
            pytest.param("User", id="user"),
            pytest.param("Organization", id="organization"),
            pytest.param("Technology", id="technology"),
        ],
    )
    def test_accepts_every_type_used_by_the_real_seed_source(self, fake_connection, node_type):
        """These are the node types the real mist-memory/seed/*.md files will
        use once Task 13 lands (verified against ALL_NODE_TYPE_NAMES).
        """
        docs = [
            _doc(
                facts=[("thing", "USES", "other")],
                nodes=[SeedNode(id="thing", type=node_type), SeedNode(id="other", type="Concept")],
            )
        ]

        apply_seed_documents(fake_connection, docs, seed_version="profile-v1", now_iso=_NOW)

        assert fake_connection.writes


class TestApplierOwnDefenses:
    """Task 11's loader (`load_seed_documents`) already rejects a fact
    referencing an undefined node and a duplicate node id -- but a caller
    that constructs `SeedDocument`s directly bypasses the loader entirely.
    These tests prove the applier does not silently trust that upstream
    check, mirroring the same defensive posture `_validate_node_types`
    takes for `type`.
    """

    def test_rejects_a_fact_referencing_an_undefined_node(self, fake_connection):
        """Bypasses `_doc()`'s auto-node-generation to construct a document
        where a fact references an id with no `nodes:` entry at all -- the
        exact shape of the R1.4 Task 10 live defect.
        """
        docs = [
            SeedDocument(
                seed_version="profile-v1",
                nodes=[SeedNode(id="user", type=_PLACEHOLDER_TYPE)],
                facts=[SeedFact(subject="user", predicate="USES", object="ghost")],
                body="b",
                source_path=Path("t.md"),
            )
        ]

        with pytest.raises(SeedSourceError, match="ghost"):
            apply_seed_documents(fake_connection, docs, seed_version="profile-v1", now_iso=_NOW)

        fake_connection.assert_no_writes()

    def test_rejects_a_duplicate_node_id_across_documents(self, fake_connection):
        docs = [
            SeedDocument(
                seed_version="profile-v1",
                nodes=[SeedNode(id="user", type=_PLACEHOLDER_TYPE)],
                facts=[],
                body="b",
                source_path=Path("a.md"),
            ),
            SeedDocument(
                seed_version="profile-v1",
                nodes=[SeedNode(id="user", type=_PLACEHOLDER_TYPE)],
                facts=[],
                body="b",
                source_path=Path("b.md"),
            ),
        ]

        with pytest.raises(SeedSourceError, match="user"):
            apply_seed_documents(fake_connection, docs, seed_version="profile-v1", now_iso=_NOW)

        fake_connection.assert_no_writes()
