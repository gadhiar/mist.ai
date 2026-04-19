"""Unit tests for admin.reset_graph provenance behaviour (ADR-009 Task 10).

Validates two properties:

1. ``reset_graph(connection, include_derived=True)`` wipes both
   ``:__Entity__`` nodes and ``:__Provenance__`` nodes.

2. ``reset_graph(connection, include_derived=False)`` wipes only
   ``:__Entity__`` nodes — provenance survives so that iterative gauntlet
   runs can preserve the seed entity baseline while wiping conversation
   artifacts.
"""

from __future__ import annotations

import pytest

from backend.knowledge import admin
from tests.mocks.neo4j import FakeNeo4jConnection


def _make_connection(non_seed_count: int = 0) -> FakeNeo4jConnection:
    """Return a FakeNeo4jConnection pre-wired with query responses.

    The fake must return sensible values for:
    - count_non_seed_entities (provenance != 'seed')
    - before-wipe counts for __Entity__ nodes and relationships
    - before-wipe count for __Provenance__ nodes (when include_derived=True)
    """
    return FakeNeo4jConnection(
        query_responses={
            # count_non_seed_entities query
            "coalesce(n.provenance": [{"count": non_seed_count}],
            # before-wipe entity node count
            "MATCH (n:__Entity__) RETURN count(n)": [{"count": 3}],
            # before-wipe relationship count
            "MATCH (:__Entity__)-[r]->(:__Entity__)": [{"count": 2}],
            # before-wipe provenance node count
            "MATCH (n:__Provenance__) RETURN count(n)": [{"count": 5}],
        }
    )


# ---------------------------------------------------------------------------
# Task 10a — include_derived=True wipes both label families
# ---------------------------------------------------------------------------


def test_graph_reset_with_include_derived_wipes_both_labels():
    """include_derived=True must issue DETACH DELETE for both __Entity__ and
    __Provenance__ nodes.
    """
    connection = _make_connection(non_seed_count=2)

    admin.reset_graph(connection, include_derived=True)

    issued_writes = [q for q, _ in connection.writes]
    assert any(
        "MATCH (n:__Entity__) DETACH DELETE n" in q for q in issued_writes
    ), f"Expected __Entity__ DETACH DELETE, got: {issued_writes}"
    assert any(
        "MATCH (n:__Provenance__) DETACH DELETE n" in q for q in issued_writes
    ), f"Expected __Provenance__ DETACH DELETE, got: {issued_writes}"


def test_graph_reset_with_include_derived_returns_provenance_count():
    """include_derived=True result dict must carry provenance_nodes_removed."""
    connection = _make_connection(non_seed_count=1)

    result = admin.reset_graph(connection, include_derived=True)

    assert (
        "provenance_nodes_removed" in result
    ), f"Expected 'provenance_nodes_removed' key in result, got: {list(result.keys())}"
    assert result["provenance_nodes_removed"] == 5


# ---------------------------------------------------------------------------
# Task 10b — include_derived=False wipes only __Entity__
# ---------------------------------------------------------------------------


def test_graph_reset_without_include_derived_wipes_only_entity():
    """include_derived=False (the default) must NOT issue a DETACH DELETE for
    __Provenance__ nodes — provenance survives the reset.
    """
    connection = _make_connection(non_seed_count=0)

    admin.reset_graph(connection, include_derived=False)

    issued_writes = [q for q, _ in connection.writes]
    assert any(
        "MATCH (n:__Entity__) DETACH DELETE n" in q for q in issued_writes
    ), f"Expected __Entity__ DETACH DELETE, got: {issued_writes}"
    assert not any(
        "__Provenance__" in q and "DETACH DELETE" in q for q in issued_writes
    ), f"__Provenance__ DETACH DELETE must NOT be issued without include_derived: {issued_writes}"


def test_graph_reset_without_include_derived_result_has_no_provenance_count():
    """include_derived=False result dict must NOT carry provenance_nodes_removed
    (or must carry zero) so callers can distinguish the two modes.
    """
    connection = _make_connection(non_seed_count=0)

    result = admin.reset_graph(connection, include_derived=False)

    # Either key is absent or value is 0 — both are acceptable contracts
    prov_removed = result.get("provenance_nodes_removed", 0)
    assert (
        prov_removed == 0
    ), f"Expected no provenance nodes removed without include_derived, got: {prov_removed}"


# ---------------------------------------------------------------------------
# Safety guard still enforced
# ---------------------------------------------------------------------------


def test_graph_reset_refuses_when_non_seed_entities_present_without_flag():
    """Safety guard: reset_graph must raise when non-seed entities exist and
    include_derived is False.
    """
    from backend.errors import Neo4jQueryError

    connection = _make_connection(non_seed_count=3)

    with pytest.raises(Neo4jQueryError):
        admin.reset_graph(connection, include_derived=False)


# ---------------------------------------------------------------------------
# Task 11 — graph-stats provenance helpers
# ---------------------------------------------------------------------------


def test_provenance_counts_by_type_issues_provenance_match():
    """provenance_counts_by_type must query MATCH (n:__Provenance__)."""
    connection = FakeNeo4jConnection(
        query_responses={
            "MATCH (n:__Provenance__)": [{"entity_type": "Person", "count": 3}],
        }
    )

    result = admin.provenance_counts_by_type(connection)

    issued = [q for q, _ in connection.queries]
    assert any(
        "MATCH (n:__Provenance__)" in q for q in issued
    ), f"Expected MATCH (n:__Provenance__) query, got: {issued}"
    assert result == [{"entity_type": "Person", "count": 3}]


def test_provenance_relationship_counts_issues_provenance_to_provenance_match():
    """provenance_relationship_counts_by_type must query (:__Provenance__)-[r]->(:__Provenance__)."""
    connection = FakeNeo4jConnection(
        query_responses={
            "(:__Provenance__)-[r]->(:__Provenance__)": [{"rel_type": "OBSERVED_IN", "count": 2}],
        }
    )

    result = admin.provenance_relationship_counts_by_type(connection)

    issued = [q for q, _ in connection.queries]
    assert any(
        "(:__Provenance__)-[r]->(:__Provenance__)" in q for q in issued
    ), f"Expected (:__Provenance__)-[r]->(:__Provenance__) query, got: {issued}"
    assert result == [{"rel_type": "OBSERVED_IN", "count": 2}]


def test_cross_layer_relationship_counts_issues_mixed_endpoint_match():
    """cross_layer_relationship_counts must query both directions:
    (:__Entity__)-[r]->(:__Provenance__) OR (:__Provenance__)-[r]->(:__Entity__).
    """
    connection = FakeNeo4jConnection(
        query_responses={
            "s:__Entity__ AND t:__Provenance__": [{"rel_type": "HAS_PROVENANCE", "count": 7}],
        }
    )

    result = admin.cross_layer_relationship_counts(connection)

    issued = [q for q, _ in connection.queries]
    assert any(
        "s:__Entity__ AND t:__Provenance__" in q for q in issued
    ), f"Expected cross-layer entity/provenance query, got: {issued}"
    assert any(
        "s:__Provenance__ AND t:__Entity__" in q for q in issued
    ), f"Expected reverse direction (Provenance->Entity) in same query, got: {issued}"
    assert result == [{"rel_type": "HAS_PROVENANCE", "count": 7}]


# ---------------------------------------------------------------------------
# Task 12 — graph-dump --include-provenance
# ---------------------------------------------------------------------------


def test_graph_dump_default_excludes_provenance():
    """dump_graph_json with default args queries only MATCH (n:__Entity__) and
    must NOT issue a MATCH (n:__Provenance__) query.
    """
    connection = FakeNeo4jConnection(
        query_responses={
            "MATCH (n:__Entity__)": [
                {"id": "e1", "labels": ["__Entity__", "Person"], "properties": {"name": "Alice"}},
            ],
            "MATCH (s:__Entity__)-[r]->(t:__Entity__)": [],
        }
    )

    result = admin.dump_graph_json(connection)

    issued = [q for q, _ in connection.queries]
    assert any(
        "MATCH (n:__Entity__)" in q for q in issued
    ), f"Expected MATCH (n:__Entity__) query, got: {issued}"
    assert not any(
        "MATCH (n:__Provenance__)" in q for q in issued
    ), f"Provenance query must NOT be issued by default, got: {issued}"
    # Return shape must contain entity nodes and relationships
    assert "nodes" in result
    assert "relationships" in result
    # Provenance keys must be absent in the default output
    assert "provenance" not in result
    assert "cross_layer_edges" not in result


def test_graph_dump_include_provenance_emits_both_subgraphs():
    """dump_graph_json(include_provenance=True) must issue queries for both
    MATCH (n:__Entity__) and MATCH (n:__Provenance__), plus a cross-layer
    edge query, and return the results under separate keys.
    """
    connection = FakeNeo4jConnection(
        query_responses={
            "MATCH (n:__Entity__)": [
                {"id": "e1", "labels": ["__Entity__", "Person"], "properties": {"name": "Alice"}},
            ],
            "MATCH (s:__Entity__)-[r]->(t:__Entity__)": [],
            "MATCH (n:__Provenance__)": [
                {
                    "id": "p1",
                    "labels": ["__Provenance__"],
                    "properties": {"source": "conv-001"},
                },
            ],
            "MATCH (s:__Provenance__)-[r]->(t:__Provenance__)": [],
            # Cross-layer query — match on a substring common to the implementation
            "s:__Entity__ AND t:__Provenance__": [
                {"source": "e1", "type": "HAS_PROVENANCE", "target": "p1", "properties": {}},
            ],
        }
    )

    result = admin.dump_graph_json(connection, include_provenance=True)

    issued = [q for q, _ in connection.queries]
    assert any(
        "MATCH (n:__Entity__)" in q for q in issued
    ), f"Expected MATCH (n:__Entity__) query, got: {issued}"
    assert any(
        "MATCH (n:__Provenance__)" in q for q in issued
    ), f"Expected MATCH (n:__Provenance__) query, got: {issued}"
    assert any(
        "s:__Entity__ AND t:__Provenance__" in q for q in issued
    ), f"Expected cross-layer query, got: {issued}"

    # Return shape must include provenance and cross_layer_edges keys
    assert (
        "provenance" in result
    ), f"Expected 'provenance' key in result, got: {list(result.keys())}"
    assert (
        "cross_layer_edges" in result
    ), f"Expected 'cross_layer_edges' key in result, got: {list(result.keys())}"
    # Core entity keys must still be present
    assert "nodes" in result
    assert "relationships" in result
    # Provenance section must itself contain nodes and relationships
    assert "nodes" in result["provenance"]
    assert "relationships" in result["provenance"]
