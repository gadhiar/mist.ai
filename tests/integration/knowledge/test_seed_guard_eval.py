"""MIS-177 D1 against real Neo4j: what the seed guard counts, and that a refusal writes nothing.

`tests/unit/knowledge/seed/test_seed_guard.py` answers the guard's two
statements with chosen rows. Only Neo4j shows what those statements count: an
empty graph and a seed-only graph must answer zero in every column (the
one-row aggregates, the OPTIONAL MATCH over no relationship), each blocking
element must land in exactly its own column, and a MistIdentity node shaped
like `GraphStore.ensure_mist_identity`'s (`ontology_version` its only stamp)
must not block.

Each blocking case sits in the partition that isolates its arm: the stamp and
provenance cases are `:__SelfModel__`-only, which `admin.RESET_GUARD_CYPHER`
never sees.

Runs ONLY against the disposable eval instance (docker-compose.eval-neo4j.yml)
and skips when it is not reachable. It empties that instance before and after
every case. Start the target first:

  docker compose -f docker-compose.yml -f docker-compose.eval-neo4j.yml \
    --profile eval up -d mist-neo4j-eval
"""

from __future__ import annotations

import socket
from dataclasses import replace
from pathlib import Path

import pytest

from backend.errors import SeedTargetNotSeedOnlyError
from backend.knowledge.config import Neo4jConfig
from backend.knowledge.eval_isolation import LIVE_NEO4J_ENDPOINTS, assert_neo4j_uri_not_live
from backend.knowledge.seed.applier import (
    SeedGuardCounts,
    apply_seed_documents,
    count_seed_guard_elements,
    reseed,
)
from backend.knowledge.seed.models import SeedDocument, SeedFact, SeedNode
from backend.knowledge.storage.neo4j_connection import Neo4jConnection
from backend.knowledge.storage.partitions import ENTITY_LABEL, SELF_MODEL_LABEL

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

_SEED_VERSION = "mis177-guard-eval-v1"
_NOW = "2026-09-29T00:00:00+00:00"
_ZERO = SeedGuardCounts(0, 0, 0, 0, 0, 0)


def _documents() -> list[SeedDocument]:
    return [
        SeedDocument(
            seed_version=_SEED_VERSION,
            nodes=[
                SeedNode(id="sg-user", type="User", display_name="Eval User"),
                SeedNode(id="sg-rust", type="Technology", display_name="Rust"),
            ],
            facts=[SeedFact(subject="sg-user", predicate="USES", object="sg-rust")],
            body="MIS-177 guard eval seed",
            source_path=Path("mis177-guard-user.md"),
            partition=ENTITY_LABEL,
        ),
        SeedDocument(
            seed_version=_SEED_VERSION,
            nodes=[
                SeedNode(id="sg-mist", type="MistIdentity", display_name="MIST"),
                SeedNode(id="sg-warm", type="MistTrait", display_name="Warm"),
            ],
            facts=[SeedFact(subject="sg-mist", predicate="HAS_TRAIT", object="sg-warm")],
            body="MIS-177 guard eval self-model seed",
            source_path=Path("mis177-guard-mist.md"),
            partition=SELF_MODEL_LABEL,
        ),
    ]


def _apply(conn) -> dict[str, int]:
    return apply_seed_documents(conn, _documents(), seed_version=_SEED_VERSION, now_iso=_NOW)


def _reseed(conn) -> dict[str, int]:
    return reseed(conn, _documents(), seed_version=_SEED_VERSION, now_iso=_NOW)


_ENTRY_POINTS = {"apply_seed_documents": _apply, "reseed": _reseed}


@pytest.fixture(params=sorted(_ENTRY_POINTS))
def seed(request):
    return _ENTRY_POINTS[request.param]


@pytest.fixture
def eval_connection():
    host, port = _ENDPOINT
    uri = f"bolt://{host}:{port}"
    # Belt and braces: the candidate list is eval-only, and this test empties
    # the graph it connects to.
    assert_neo4j_uri_not_live(uri, action="the MIS-177 seed guard eval test (full graph wipe)")
    conn = Neo4jConnection(Neo4jConfig(uri=uri, username="neo4j", password="password"))
    conn.connect()
    conn.execute_write("MATCH (n) DETACH DELETE n", {})
    yield conn
    conn.execute_write("MATCH (n) DETACH DELETE n", {})
    conn.disconnect()


def _snapshot(conn) -> tuple[list, list]:
    nodes = conn.execute_query(
        "MATCH (n) RETURN n.id AS id, labels(n) AS labels, properties(n) AS props ORDER BY id",
        {},
    )
    relationships = conn.execute_query(
        "MATCH (s)-[r]->(o) RETURN s.id AS s, type(r) AS t, o.id AS o, "
        "properties(r) AS props ORDER BY s, t, o",
        {},
    )
    return (
        [(r["id"], sorted(r["labels"]), r["props"]) for r in nodes],
        [(r["s"], r["t"], r["o"], r["props"]) for r in relationships],
    )


def _count(conn, query: str) -> int:
    return conn.execute_query(query, {})[0]["c"]


class TestAGraphTheSeedMayWrite:
    def test_an_empty_graph_counts_nothing_and_is_seeded(self, eval_connection, seed):
        conn = eval_connection
        assert count_seed_guard_elements(conn) == _ZERO

        assert seed(conn) == {"nodes": 4, "facts": 2}
        assert _count(conn, "MATCH (n) RETURN count(n) AS c") == 4

    def test_a_seed_only_graph_counts_nothing_and_is_seeded_again(self, eval_connection, seed):
        conn = eval_connection
        _apply(conn)
        conn.execute_write(
            "MATCH (n) WHERE n.seed_version = $v SET n.embedding = [0.1, 0.2]",
            {"v": _SEED_VERSION},
        )
        assert count_seed_guard_elements(conn) == _ZERO

        assert seed(conn) == {"nodes": 4, "facts": 2}
        assert _count(conn, "MATCH (n) RETURN count(n) AS c") == 4
        assert _count(conn, "MATCH ()-[r]->() RETURN count(r) AS c") == 2

    def test_a_startup_mist_identity_does_not_block(self, eval_connection, seed):
        """`ensure_mist_identity`'s node: `ontology_version` is its only stamp."""
        conn = eval_connection
        conn.execute_write(
            "CREATE (:__SelfModel__:MistIdentity {id: 'mist-identity', "
            "entity_type: 'MistIdentity', display_name: 'MIST', confidence: 1.0, "
            "status: 'active', created_at: datetime(), ontology_version: '1.2.0'})",
            {},
        )
        assert count_seed_guard_elements(conn) == _ZERO

        seed(conn)
        assert _count(conn, "MATCH (n) RETURN count(n) AS c") == 5


_SM_PAIR = "MATCH (m:__SelfModel__ {id: 'sg-mist'}), (t:__SelfModel__ {id: 'sg-warm'}) "
_ENTITY_PAIR = "MATCH (u:__Entity__ {id: 'sg-user'}), (t:__Entity__ {id: 'sg-rust'}) "

# (statement added to a seeded graph, the one SeedGuardCounts field it must raise)
_BLOCKING = [
    pytest.param(
        "CREATE (:__Entity__ {id: 'sg-py', entity_type: 'Technology', display_name: 'Python'})",
        "reset_guard_nodes",
        id="non-seed-entity-node",
    ),
    pytest.param(
        _ENTITY_PAIR + "CREATE (u)-[:WORKS_AT {confidence: 0.8}]->(t)",
        "reset_guard_relationships",
        id="entity-relationship-without-seed-version",
    ),
    pytest.param(
        "CREATE (:__SelfModel__:MistTrait {id: 'sg-curious', entity_type: 'MistTrait', "
        "extraction_version: 'v-x', model_hash: 'm-x'})",
        "stamped_nodes",
        id="extraction-stamped-self-model-node",
    ),
    pytest.param(
        _SM_PAIR + "CREATE (m)-[:HAS_TRAIT {model_hash: 'm-x'}]->(t)",
        "stamped_relationships",
        id="extraction-stamped-relationship",
    ),
    pytest.param(
        "CREATE (:__SelfModel__:MistTrait {id: 'sg-bold', entity_type: 'MistTrait', "
        "provenance: 'extraction'})",
        "extraction_nodes",
        id="provenance-extraction-node",
    ),
    pytest.param(
        _SM_PAIR + "CREATE (m)-[:HAS_TRAIT {provenance: 'extraction'}]->(t)",
        "extraction_relationships",
        id="provenance-extraction-relationship",
    ),
]


@pytest.mark.parametrize("statement, field", _BLOCKING)
def test_a_blocking_element_is_counted_in_its_own_column(eval_connection, statement, field):
    conn = eval_connection
    _apply(conn)
    conn.execute_write(statement, {})

    counts = count_seed_guard_elements(conn)

    assert counts == replace(_ZERO, **{field: 1}), counts


@pytest.mark.parametrize("statement, field", _BLOCKING)
def test_a_blocking_element_is_refused_and_the_graph_left_intact(
    eval_connection, seed, statement, field
):
    conn = eval_connection
    _apply(conn)
    conn.execute_write(statement, {})
    before = _snapshot(conn)

    with pytest.raises(SeedTargetNotSeedOnlyError):
        seed(conn)

    assert _snapshot(conn) == before, "a refused seed changed the graph"
