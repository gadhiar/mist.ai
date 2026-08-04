"""Canonical graph serializer (F3) -- deterministic, wall-clock-free form."""

from backend.knowledge.canonical_serialize import (
    _canon_props,
    _node,
    _rel,
    _rel_key,
    canonical_graph_form,
)
from tests.mocks.neo4j import FakeNeo4jConnection


def _conn(nodes, rels):
    # dump_graph_json -> _dump_subgraph runs the node query then the rel query;
    # FakeNeo4jConnection matches by query substring.
    return FakeNeo4jConnection(
        query_responses={
            "RETURN n.id AS id": nodes,
            "RETURN s.id AS source": rels,
        }
    )


# Real event ids are uuid4 (`backend/event_store/models.py`), so they are fixed
# width. Two that sort in a known order, so an ordering assertion can name which
# comes first without depending on how a uuid happens to render.
EID_EARLY = "0a1b2c3d-1111-4000-8000-000000000001"
EID_LATE = "f9e8d7c6-2222-4000-8000-000000000002"


def _version_key(event_id: str, valid_from: str | None = None, valid_to: str | None = None) -> str:
    """Mirror of `reconciliation._version_key` -- the C2 MERGE identity."""
    return f"{event_id}|{valid_from or 'open'}|{valid_to or 'open'}"


def _fact_edge(
    source: str,
    rel_type: str,
    target: str,
    *,
    event_id: str,
    valid_from: str | None = None,
    valid_to: str | None = None,
) -> dict:
    """Build an edge in the present-day (post-C1) reconciliation shape.

    Matches what `reconciliation._apply_append_version` writes: a uuid4
    `source_utterance_id`, a `version_key` of `event_id|valid_from|valid_to`,
    and no `source_event_id` at all.
    """
    return {
        "source": source,
        "type": rel_type,
        "target": target,
        "properties": {
            "source_utterance_id": event_id,
            "version_key": _version_key(event_id, valid_from, valid_to),
            "valid_from": valid_from,
            "valid_to": valid_to,
            "confidence": 0.9,
            "created_at": "2026-08-03T00:00:00Z",
        },
    }


class TestCanonicalGraphForm:
    def test_order_and_wallclock_independent(self):
        nodes_a = [
            {
                "id": "rust",
                "labels": ["Technology"],
                "properties": {
                    "display_name": "Rust",
                    "created_at": "2026-01-01",
                    "aliases": ["rs", "rustlang"],
                    "source_utterance_id": "e1",
                },
            },
            {
                "id": "python",
                "labels": ["Technology"],
                "properties": {"created_at": "2026-01-01", "display_name": "Python"},
            },
        ]
        rels_a = [
            {
                "source": "user",
                "type": "USES",
                "target": "rust",
                "properties": {"created_at": "2026-01-01", "source_utterance_id": "e1"},
            },
        ]
        # Same content; shuffled node order, different wall-clock, shuffled aliases.
        nodes_b = [
            {
                "id": "python",
                "labels": ["Technology"],
                "properties": {"display_name": "Python", "created_at": "2099-12-31"},
            },
            {
                "id": "rust",
                "labels": ["Technology"],
                "properties": {
                    "aliases": ["rustlang", "rs"],
                    "display_name": "Rust",
                    "created_at": "2099-12-31",
                    "source_utterance_id": "e1",
                },
            },
        ]
        rels_b = [
            {
                "source": "user",
                "type": "USES",
                "target": "rust",
                "properties": {"source_utterance_id": "e1", "created_at": "2099-12-31"},
            },
        ]

        form_a = canonical_graph_form(_conn(nodes_a, rels_a))
        form_b = canonical_graph_form(_conn(nodes_b, rels_b))
        assert form_a == form_b

    def test_excludes_audit_and_embedding_retains_provenance(self):
        nodes = [
            {
                "id": "rust",
                "labels": ["Technology"],
                "properties": {
                    "display_name": "Rust",
                    "created_at": "x",
                    "updated_at": "y",
                    "embedding": [0.1, 0.2],
                    "source_utterance_id": "e1",
                },
            }
        ]
        form = canonical_graph_form(_conn(nodes, []))
        assert "created_at" not in form
        assert "updated_at" not in form
        assert "embedding" not in form
        assert "Rust" in form
        assert "source_utterance_id" in form  # deterministic provenance retained

    def test_un_backfilled_pre_c1_rows_sort_by_their_legacy_event_id(self):
        # NOT general multi-edge coverage -- this is the pre-C1 row shape only, and
        # specifically the UN-BACKFILLED one. Two USES edges user->rust from
        # different events must sort stably by the legacy `source_event_id`: the
        # pre-C1 property name that no write path sets today, but that `_rel_key`
        # still reads first so pre-C1 rows keep their tiebreak. Present-day rows
        # carry `source_utterance_id` instead, and are covered in
        # TestRelKeyOrderingComponents. Once `admin.backfill_bitemporal` has run,
        # such a row carries BOTH properties with equal values (it stamps
        # source_utterance_id = coalesce(source_event_id, ...) without removing
        # the original), so this fixture models the state before that pass.
        rels = [
            {
                "source": "user",
                "type": "USES",
                "target": "rust",
                "properties": {"source_event_id": "e2", "confidence": 0.9},
            },
            {
                "source": "user",
                "type": "USES",
                "target": "rust",
                "properties": {"source_event_id": "e1", "confidence": 0.8},
            },
        ]
        form1 = canonical_graph_form(_conn([], rels))
        form2 = canonical_graph_form(_conn([], list(reversed(rels))))
        assert form1 == form2  # input order does not change canonical output
        assert form1.index('"e1"') < form1.index('"e2"')  # ordered by source_event_id


class TestCanonPropsEpochAndConfidence:
    def test_canon_props_drops_epoch_stamps_for_both_nodes_and_edges(self):
        props = {
            "display_name": "Python",
            "ontology_version": "1.4.0",
            "extraction_version": "2026-06-14-r5",
            "model_hash": "gemma-x",
        }
        node_out = _canon_props(props, is_node=True)
        edge_out = _canon_props(props, is_node=False)
        for out in (node_out, edge_out):
            assert "ontology_version" not in out
            assert "extraction_version" not in out
            assert "model_hash" not in out
            assert out["display_name"] == "Python"

    def test_canon_props_drops_confidence_for_nodes_keeps_for_edges(self):
        props = {"display_name": "Python", "confidence": 0.73}
        assert "confidence" not in _canon_props(props, is_node=True)
        assert _canon_props(props, is_node=False)["confidence"] == 0.73

    def test_node_and_rel_apply_the_right_kind(self):
        n = {
            "id": "python",
            "labels": ["__Entity__"],
            "properties": {"confidence": 0.9, "model_hash": "x"},
        }
        r = {
            "source": "user",
            "type": "USES",
            "target": "python",
            "properties": {"confidence": 0.9, "model_hash": "x"},
        }
        assert "confidence" not in _node(n)["properties"]  # node: dropped
        assert "model_hash" not in _node(n)["properties"]
        assert _rel(r)["properties"]["confidence"] == 0.9  # edge: kept
        assert "model_hash" not in _rel(r)["properties"]


class TestRelKey:
    def test_rel_key_distinguishes_same_turn_valid_time_versions(self):
        # Two versions of the same (s,type,t,event_id), differing only in valid-time
        # -- they MUST get distinct keys so the canonical sort is order-independent.
        base = {"source": "user", "type": "WORKS_AT", "target": "acme"}
        v1 = {
            **base,
            "properties": {
                "source_utterance_id": "e1",
                "version_key": "e1|2020|2022",
                "valid_from": "2020",
                "valid_to": "2022",
            },
        }
        v2 = {
            **base,
            "properties": {
                "source_utterance_id": "e1",
                "version_key": "e1|2022|open",
                "valid_from": "2022",
                "valid_to": None,
            },
        }
        assert _rel_key(v1) != _rel_key(v2)

    def test_rel_key_stable_for_durable_null_version_key_edges(self):
        # DURABLE edges carry no version_key; they are unique per (s,type,t), so the
        # leading triple disambiguates and the key is still total.
        a = {"source": "python", "type": "IS_A", "target": "language", "properties": {}}
        b = {"source": "python", "type": "DEPENDS_ON", "target": "cpython", "properties": {}}
        assert _rel_key(a) != _rel_key(b)
        assert _rel_key(a) == _rel_key(dict(a))  # deterministic for identical input


class TestRelKeyOrderingComponents:
    """One ordering component per test, so a dead component fails exactly one.

    `_rel_key` is the total order under `assert_rebuild_twice_identical`. A
    component that no test can distinguish is a component that can be deleted
    without the determinism gate noticing -- the gate keeps passing and stops
    meaning anything. Each test below holds every component constant except the
    one it names, so neutralizing that component collapses the two edges to an
    equal key; `sorted` is stable, so a tie makes the canonical output follow
    Neo4j's row order, which the dump's `ORDER BY s.id, type(r), t.id` does not
    pin for parallel edges. Reversing the input therefore detects the tie.

    Two components resist isolation for a structural reason, documented at their
    tests: `version_key` is a concatenation of `source_utterance_id`,
    `valid_from` and `valid_to`, so it is never the sole separator of a
    reconciliation row, and the `source_utterance_id` arm of the utterance
    tiebreak is mutually redundant with it for the same reason.
    """

    def _assert_order_independent(self, rels):
        """Canonical output must not depend on the order Neo4j returned rows in."""
        forward = canonical_graph_form(_conn([], rels))
        reverse = canonical_graph_form(_conn([], list(reversed(rels))))
        assert forward == reverse
        return forward

    def test_two_facts_from_one_utterance_are_ordered_by_source(self):
        # "I work at Acme, and so does Bob" -- one utterance, one valid interval,
        # so both edges carry the SAME source_utterance_id and the SAME
        # version_key (`_version_key` keys on the event, not on the triple).
        # Only the leading `source` separates them.
        rels = [
            _fact_edge("bob", "WORKS_AT", "acme", event_id=EID_EARLY),
            _fact_edge("alice", "WORKS_AT", "acme", event_id=EID_EARLY),
        ]

        form = self._assert_order_independent(rels)

        assert form.index('"alice"') < form.index('"bob"')

    def test_two_predicates_from_one_utterance_are_ordered_by_type(self):
        # "I founded Acme and I work there" -- same subject, same object, same
        # utterance, so source/target/source_utterance_id/version_key all match.
        rels = [
            _fact_edge("raj", "WORKS_AT", "acme", event_id=EID_EARLY),
            _fact_edge("raj", "FOUNDED", "acme", event_id=EID_EARLY),
        ]

        form = self._assert_order_independent(rels)

        assert form.index('"FOUNDED"') < form.index('"WORKS_AT"')

    def test_two_objects_from_one_utterance_are_ordered_by_target(self):
        # "I work at Acme and Globex" -- only the target separates the two edges.
        rels = [
            _fact_edge("raj", "WORKS_AT", "globex", event_id=EID_EARLY),
            _fact_edge("raj", "WORKS_AT", "acme", event_id=EID_EARLY),
        ]

        form = self._assert_order_independent(rels)

        assert form.index('"acme"') < form.index('"globex"')

    def test_post_c1_versions_of_one_fact_are_ordered_by_their_utterance_id(self):
        # The present-day shape the whole graph is written in: two versions of one
        # fact, appended by two different turns. No `source_event_id` anywhere.
        #
        # This test does NOT isolate the `source_utterance_id` arm. Because
        # `version_key` embeds the event id as its fixed-width leading field, the
        # arm and `version_key` always agree on the order, and neutralizing either
        # one alone leaves the other to separate the edges. That mutual redundancy
        # is the finding, not a gap in the fixture -- no production row shape can
        # separate them. What this pins is that the post-C1 shape sorts totally
        # and stably at all, which fails the moment both are lost.
        rels = [
            _fact_edge("raj", "WORKS_AT", "acme", event_id=EID_LATE, valid_from="2024"),
            _fact_edge("raj", "WORKS_AT", "acme", event_id=EID_EARLY, valid_from="2020"),
        ]

        form = self._assert_order_independent(rels)

        assert form.index(EID_EARLY) < form.index(EID_LATE)

    def test_pre_c1_and_post_c1_rows_of_one_triple_do_not_collide(self):
        # Both row shapes coexist for the whole pre-C1/post-C1 window, which is
        # exactly the window `live == rebuilt` runs in: an un-backfilled legacy row
        # carries only `source_event_id`, a rebuilt row only `source_utterance_id`.
        # The `or` chain reads each row's own id, so the keys stay distinct and the
        # order is total. Neutralizing either arm alone keeps them distinct (one
        # side collapses to ""), so this test deliberately survives both single-arm
        # mutations -- it is a collision detector for the mixed shape, not an
        # arm-isolating test.
        rels = [
            {
                "source": "raj",
                "type": "WORKS_AT",
                "target": "acme",
                "properties": {"source_event_id": EID_LATE, "confidence": 0.8},
            },
            _fact_edge("raj", "WORKS_AT", "acme", event_id=EID_EARLY),
        ]

        self._assert_order_independent(rels)

    def test_backfilled_legacy_rows_are_ordered_by_valid_to(self):
        # `admin.backfill_bitemporal` sets version_key = source_event_id, so unlike
        # a reconciliation row its version_key does NOT embed valid-time. Two
        # pre-C1 parallel rows stamped from one event therefore share every
        # component except valid_to, which the backfill closes only for rows whose
        # legacy temporal_status was 'past'. valid_to is the sole separator here.
        common = {"source_event_id": EID_EARLY, "source_utterance_id": EID_EARLY}
        rels = [
            {
                "source": "raj",
                "type": "WORKS_AT",
                "target": "acme",
                "properties": {**common, "version_key": EID_EARLY, "valid_to": "2024-01-01"},
            },
            {
                "source": "raj",
                "type": "WORKS_AT",
                "target": "acme",
                "properties": {**common, "version_key": EID_EARLY, "valid_to": None},
            },
        ]

        form = self._assert_order_independent(rels)

        assert form.index('"valid_to": null') < form.index('"valid_to": "2024-01-01"')

    def test_backfilled_legacy_rows_are_ordered_by_valid_from(self):
        # Same backfilled shape, but separated by a valid_from the legacy schema
        # already carried (the backfill stamps version_key without reading it).
        common = {"source_event_id": EID_EARLY, "source_utterance_id": EID_EARLY}
        rels = [
            {
                "source": "raj",
                "type": "WORKS_AT",
                "target": "acme",
                "properties": {**common, "version_key": EID_EARLY, "valid_from": "2024-01-01"},
            },
            {
                "source": "raj",
                "type": "WORKS_AT",
                "target": "acme",
                "properties": {**common, "version_key": EID_EARLY, "valid_from": "2020-01-01"},
            },
        ]

        form = self._assert_order_independent(rels)

        assert form.index('"2020-01-01"') < form.index('"2024-01-01"')

    def test_live_arm_supplies_the_tiebreak_when_no_other_component_can(self):
        # Contract pin for the `source_utterance_id` arm, which today's MERGE
        # contracts cannot produce a fixture for: two parallel edges on one triple
        # that carry an utterance id and no version_key are barred by
        # `_apply_structural`'s bare-triple MERGE (one such edge per triple) and by
        # `_apply_append_version` (every row it writes has a version_key). Every
        # OTHER shape leaves the arm redundant with version_key.
        #
        # Named for what it is: the arm has no reachable ordering fixture, so this
        # asserts the arm's contract directly rather than a state the graph has
        # been observed in. It is here so that deleting the arm -- which no other
        # test in this suite notices -- fails something.
        rels = [
            {
                "source": "raj",
                "type": "WORKS_AT",
                "target": "acme",
                "properties": {"source_utterance_id": EID_LATE},
            },
            {
                "source": "raj",
                "type": "WORKS_AT",
                "target": "acme",
                "properties": {"source_utterance_id": EID_EARLY},
            },
        ]

        form = self._assert_order_independent(rels)

        assert form.index(EID_EARLY) < form.index(EID_LATE)
