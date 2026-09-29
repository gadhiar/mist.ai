"""The graph-reset guard against real Neo4j (`admin.RESET_GUARD_CYPHER`).

`tests/unit/test_admin_reset_guard.py` evaluates the guard's Cypher in a
subset interpreter; this runs the same cases through Neo4j itself, so the
NULL handling, the label tests and the one-row aggregate are Neo4j's own. The
pure-seed case is written by the real seed applier; the others add elements
with plain CREATE statements shaped like the writers that produce them.

Runs ONLY against the disposable eval instance (docker-compose.eval-neo4j.yml)
and skips when it is not reachable. It empties that instance before and after
every case. Start the target first:

  docker compose -f docker-compose.yml -f docker-compose.eval-neo4j.yml \
    --profile eval up -d mist-neo4j-eval
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from backend.errors import Neo4jQueryError
from backend.knowledge import admin
from backend.knowledge.config import Neo4jConfig
from backend.knowledge.eval_isolation import LIVE_NEO4J_ENDPOINTS, assert_neo4j_uri_not_live
from backend.knowledge.seed.applier import apply_seed_documents
from backend.knowledge.seed.models import SeedDocument, SeedFact, SeedNode
from backend.knowledge.storage.neo4j_connection import Neo4jConnection

# In-container service name first; host-published port as fallback. Every
# candidate is an eval endpoint (DEFAULT_EVAL_NEO4J_ENDPOINTS); none is live.
_CANDIDATES = [("mist-neo4j-eval", 7687), ("localhost", 7688), ("127.0.0.1", 7688)]
assert not set(_CANDIDATES) & LIVE_NEO4J_ENDPOINTS


def _eval_endpoint() -> tuple[str, int] | None:
    for host, port in _CANDIDATES:
        try:
            sock = socket.create_connection((host, port), timeout=2)
            sock.close()
            return host, port
        except OSError:
            continue
    return None


_ENDPOINT = _eval_endpoint()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        _ENDPOINT is None,
        reason=(
            "disposable eval Neo4j not running (docker compose -f docker-compose.yml "
            "-f docker-compose.eval-neo4j.yml --profile eval up -d mist-neo4j-eval)"
        ),
    ),
]

_SEED_VERSION = "reset-guard-eval-v1"
_NOW = "2026-09-01T00:00:00+00:00"
_STAMPS = "ontology_version: '1.2.0', extraction_version: 'v-x', model_hash: 'm-x'"


def _documents() -> list[SeedDocument]:
    return [
        SeedDocument(
            seed_version=_SEED_VERSION,
            nodes=[
                SeedNode(id="rg-user", type="User", display_name="Eval User"),
                SeedNode(id="rg-rust", type="Technology", display_name="Rust"),
            ],
            facts=[SeedFact(subject="rg-user", predicate="USES", object="rg-rust")],
            body="reset guard eval seed",
            source_path=Path("reset-guard-eval.md"),
        )
    ]


@pytest.fixture
def eval_connection():
    host, port = _ENDPOINT
    uri = f"bolt://{host}:{port}"
    # Belt and braces: the candidate list is eval-only, and this test empties
    # the graph it connects to.
    assert_neo4j_uri_not_live(uri, action="the reset-guard eval test (full graph wipe)")
    conn = Neo4jConnection(Neo4jConfig(uri=uri, username="neo4j", password="password"))
    conn.connect()
    conn.execute_write("MATCH (n) DETACH DELETE n", {})
    apply_seed_documents(conn, _documents(), seed_version=_SEED_VERSION, now_iso=_NOW)
    yield conn
    conn.execute_write("MATCH (n) DETACH DELETE n", {})
    conn.disconnect()


_PAIR = "MATCH (u:__Entity__ {id: 'rg-user'}), (t:__Entity__ {id: 'rg-rust'}) "

# (extra statement, nodes counted, relationships counted)
GUARDED = [
    pytest.param(
        _PAIR + f"CREATE (u)-[:WORKS_AT {{provenance: 'extraction', {_STAMPS}}}]->(t)",
        0,
        1,
        id="seed-pair-extraction-edge",
    ),
    pytest.param(
        _PAIR + "CREATE (u)-[:USES {provenance: 'seed', source_type: 'stated', "
        f"confidence: 1.0, seed_origin_version: '{_SEED_VERSION}', {_STAMPS}}}]->(t)",
        0,
        1,
        id="clamped-seed-copy",
    ),
    pytest.param(
        f"CREATE (:__Entity__:User {{id: 'rg-old', seed_version: '{_SEED_VERSION}'}})",
        1,
        0,
        id="pre-kg125-seed-node",
    ),
    pytest.param(
        f"MATCH ()-[r:USES]->() SET r += {{{_STAMPS}}}",
        0,
        1,
        id="adopted-relationship",
    ),
    pytest.param(
        _PAIR + "CREATE (u)-[:EXTRACTED_FROM]->(:__Provenance__:ConversationContext "
        "{id: 'rg-ctx'})",
        0,
        1,
        id="entity-to-non-entity-edge",
    ),
    pytest.param(
        f"CREATE (:__Entity__:Technology {{id: 'rg-py', provenance: 'extraction', {_STAMPS}}})",
        1,
        0,
        id="extraction-derived-node",
    ),
]


def test_a_pure_seed_graph_counts_nothing_and_resets(eval_connection):
    conn = eval_connection
    assert admin.count_reset_guard_elements(conn) == (0, 0)
    result = admin.reset_graph(conn, include_derived=False)
    assert result["nodes_removed"] == 2
    assert conn.execute_query("MATCH (n:__Entity__) RETURN count(n) AS c", {})[0]["c"] == 0


def test_an_empty_graph_counts_nothing(eval_connection):
    conn = eval_connection
    conn.execute_write("MATCH (n) DETACH DELETE n", {})
    assert admin.count_reset_guard_elements(conn) == (0, 0)


@pytest.mark.parametrize("statement, nodes, relationships", GUARDED)
def test_guarded_graphs_are_refused_and_left_intact(
    eval_connection, statement, nodes, relationships
):
    conn = eval_connection
    conn.execute_write(statement, {})
    assert admin.count_reset_guard_elements(conn) == (nodes, relationships)
    assert admin.count_non_seed_entities(conn) == nodes + relationships
    before = conn.execute_query("MATCH (n) RETURN count(n) AS c", {})[0]["c"]
    with pytest.raises(Neo4jQueryError):
        admin.reset_graph(conn, include_derived=False)
    assert conn.execute_query("MATCH (n) RETURN count(n) AS c", {})[0]["c"] == before


@pytest.mark.parametrize("statement, nodes, relationships", GUARDED)
def test_include_derived_resets_guarded_graphs(eval_connection, statement, nodes, relationships):
    conn = eval_connection
    conn.execute_write(statement, {})
    admin.reset_graph(conn, include_derived=True)
    assert conn.execute_query("MATCH (n:__Entity__) RETURN count(n) AS c", {})[0]["c"] == 0
    assert conn.execute_query("MATCH (n:__Provenance__) RETURN count(n) AS c", {})[0]["c"] == 0
