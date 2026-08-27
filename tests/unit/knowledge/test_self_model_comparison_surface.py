"""MIS-131: make the self-model observable to the `live == rebuilt` gate.

Today the gate compares `:__Entity__` only. `canonical_graph_form` ->
`dump_graph_json` -> `_dump_subgraph(conn, "__Entity__")`, and the relationship
query requires BOTH endpoints to carry the label. On the live graph that is 11
nodes and 10 relationships out of 32 and 30 -- twenty-one nodes and twenty
relationships structurally invisible.

Why it matters is sequencing, not completeness. The closure design's corrected
order puts this BEFORE the seed-apply change, because otherwise: delete
copy-forward, wire a seed-apply that writes zero nodes, and both gates stay
green -- determinism passes because two empty self-models are byte-identical,
and `live == rebuilt` passes because it never looks at that partition. The
retirement would be proven by nothing.

So the requirement is not merely "see the partition". It is that **"applied
nothing" and "applied correctly" must be different observables**, which is what
`assert_self_model_applied` enforces: non-zero AND equal, never just equal.
"""

from __future__ import annotations

import json

import pytest

from backend.knowledge import admin
from backend.knowledge.canonical_serialize import canonical_graph_form
from backend.knowledge.regeneration.rebuild_gate import (
    RebuildVacuityError,
    assert_self_model_applied,
)
from tests.mocks.neo4j import FakeNeo4jConnection


def _connection(self_model_nodes=(), entity_nodes=(), cross_edges=()):
    return FakeNeo4jConnection(
        query_responses={
            "MATCH (n:__Entity__)": list(entity_nodes),
            "MATCH (s:__Entity__)-[r]->(t:__Entity__)": [],
            "MATCH (n:__SelfModel__)": list(self_model_nodes),
            "MATCH (s:__SelfModel__)-[r]->(t:__SelfModel__)": [],
            "WHERE (s:__Entity__ AND t:__SelfModel__)": list(cross_edges),
        }
    )


def _node(node_id, label="MistTrait"):
    return {
        "id": node_id,
        "labels": ["__SelfModel__", label],
        "properties": {"display_name": node_id},
    }


class TestDumpGraphJson:
    def test_default_still_excludes_the_self_model(self):
        """Additive. Every existing caller keeps the surface it had."""
        conn = _connection()
        result = admin.dump_graph_json(conn)
        assert "self_model" not in result
        assert not any("__SelfModel__" in q for q, _ in conn.queries)

    def test_include_self_model_emits_the_partition(self):
        conn = _connection(self_model_nodes=[_node("mist-identity", "MistIdentity")])
        result = admin.dump_graph_json(conn, include_self_model=True)
        assert [n["id"] for n in result["self_model"]["nodes"]] == ["mist-identity"]

    def test_it_is_independent_of_include_provenance(self):
        """Two orthogonal switches, not a ladder.

        A caller wanting the self-model must not be forced to also take
        provenance, which is log-derived but explicitly NOT gated.
        """
        conn = _connection(self_model_nodes=[_node("t1")])
        result = admin.dump_graph_json(conn, include_self_model=True)
        assert "self_model" in result
        assert "provenance" not in result

    def test_cross_partition_edges_are_emitted(self):
        """Mirrors the provenance plumbing: the subgraph AND its boundary.

        An edge from a self-model node into an entity node belongs to neither
        `_dump_subgraph` call -- both require BOTH endpoints to carry the
        label -- so without this it is invisible to the gate exactly like the
        partition was.
        """
        conn = _connection(
            self_model_nodes=[_node("t1")],
            cross_edges=[
                {"source": "user", "type": "ADAPTED_FOR", "target": "t1", "properties": {}}
            ],
        )
        result = admin.dump_graph_json(conn, include_self_model=True)
        assert result["self_model_cross_layer_edges"][0]["type"] == "ADAPTED_FOR"


class TestCanonicalGraphForm:
    def test_default_form_is_unchanged(self):
        """The gate's current contract must not shift underneath it."""
        form = json.loads(canonical_graph_form(_connection()))
        assert set(form) == {"nodes", "relationships"}

    def test_self_model_appears_under_its_own_key(self):
        conn = _connection(self_model_nodes=[_node("t1"), _node("t2")])
        form = json.loads(canonical_graph_form(conn, include_self_model=True))
        assert [n["id"] for n in form["self_model"]["nodes"]] == ["t1", "t2"]

    def test_self_model_nodes_are_canonicalised_like_entity_nodes(self):
        """Same exclusion rules, or the two partitions diverge on wall-clock noise."""
        node = _node("t1")
        node["properties"]["created_at"] = "2026-08-26T00:00:00+00:00"
        node["properties"]["embedding"] = [0.1, 0.2]
        form = json.loads(canonical_graph_form(_connection([node]), include_self_model=True))
        props = form["self_model"]["nodes"][0]["properties"]
        assert "created_at" not in props
        assert "embedding" not in props

    def test_ordering_is_deterministic_regardless_of_query_order(self):
        a = json.loads(
            canonical_graph_form(_connection([_node("b"), _node("a")]), include_self_model=True)
        )
        b = json.loads(
            canonical_graph_form(_connection([_node("a"), _node("b")]), include_self_model=True)
        )
        assert a == b


class TestAssertSelfModelApplied:
    """The sub-gate: non-zero AND equal, never merely equal."""

    @staticmethod
    def _form(count):
        return json.dumps({"self_model": {"nodes": [{"id": f"n{i}"} for i in range(count)]}})

    def test_matching_non_zero_counts_pass(self):
        assert_self_model_applied(self._form(21), self._form(21))

    def test_two_empty_self_models_are_refused(self):
        """THE case this gate exists for.

        Delete copy-forward, wire a seed-apply that writes zero nodes, and an
        equality-only check passes -- two empty partitions are identical. The
        retirement would be certified by nothing at all.
        """
        with pytest.raises(RebuildVacuityError, match="0"):
            assert_self_model_applied(self._form(0), self._form(0))

    def test_a_mismatch_is_refused(self):
        with pytest.raises(RebuildVacuityError, match="21.*20|20.*21"):
            assert_self_model_applied(self._form(21), self._form(20))

    def test_an_empty_rebuild_against_a_populated_live_is_refused(self):
        """'Applied nothing' -- distinct from a mismatch, and worth its own message."""
        with pytest.raises(RebuildVacuityError, match="applied nothing"):
            assert_self_model_applied(self._form(21), self._form(0))

    def test_a_form_without_the_partition_is_refused(self):
        """Calling this on a form built WITHOUT include_self_model is a caller bug.

        Returning "equal" for two forms that both lack the key would be the
        most dangerous possible pass: the gate would report the self-model
        verified while never having looked at it.
        """
        with pytest.raises(RebuildVacuityError, match="include_self_model"):
            assert_self_model_applied(json.dumps({"nodes": []}), json.dumps({"nodes": []}))
