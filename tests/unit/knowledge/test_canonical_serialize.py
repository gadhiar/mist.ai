"""Canonical graph serializer (F3) -- deterministic, wall-clock-free form."""

from backend.knowledge.canonical_serialize import canonical_graph_form
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
        # Two USES edges user->rust from different events must sort stably by
        # source_event_id (the property the live graph actually persists).
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
