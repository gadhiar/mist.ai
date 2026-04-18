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
