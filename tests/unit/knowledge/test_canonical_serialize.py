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

    def test_multi_edge_between_same_pair_sorts_deterministically(self):
        # Two USES edges user->rust from different events must sort stably by the
        # legacy `source_event_id` -- the pre-C1 property name that no write path
        # sets today, but that `_rel_key` still reads first so pre-C1 rows keep
        # their tiebreak. Present-day rows carry `source_utterance_id` instead.
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
