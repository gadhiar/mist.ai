"""Live-Cypher bitemporal currency semantics (deep review tests-quality-3).

Unit fakes can only assert the filter STRINGS; the '-inf' sentinel, the
valid_to == $now boundary, ISO string ordering, and mixed-type comparisons
are evaluated by Neo4j itself. These tests pin the closed-open read boundary
(valid_to <= read-time is EXCLUDED) against the disposable eval instance
(docker-compose.eval-neo4j.yml) -- the canonical graph is never touched.

Start the target first:
  docker compose -f docker-compose.yml -f docker-compose.eval-neo4j.yml \
    --profile eval up -d mist-neo4j-eval
"""

from __future__ import annotations

import socket
from datetime import UTC, datetime, timedelta

import pytest

from backend.knowledge.config import Neo4jConfig
from backend.knowledge.storage.graph_store import GraphStore
from backend.knowledge.storage.neo4j_connection import Neo4jConnection
from tests.mocks.embeddings import FakeEmbeddingGenerator

# In-container service name first; host-published port as fallback.
_CANDIDATES = [("mist-neo4j-eval", 7687), ("localhost", 7688), ("127.0.0.1", 7688)]


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

pytestmark = pytest.mark.skipif(
    _ENDPOINT is None,
    reason=(
        "disposable eval Neo4j not running (docker compose -f docker-compose.yml "
        "-f docker-compose.eval-neo4j.yml --profile eval up -d mist-neo4j-eval)"
    ),
)

_UID = "currencytest-user"


@pytest.fixture
def store():
    host, port = _ENDPOINT
    conn = Neo4jConnection(
        Neo4jConfig(uri=f"bolt://{host}:{port}", username="neo4j", password="password")
    )
    conn.connect()
    gs = GraphStore(connection=conn, embedding_generator=FakeEmbeddingGenerator())
    yield gs
    conn.execute_write(
        "MATCH (n:__Entity__) WHERE n.id STARTS WITH 'currencytest' DETACH DELETE n", {}
    )
    conn.disconnect()


def _seed(store: GraphStore) -> None:
    now = datetime.now(UTC)
    past = (now - timedelta(days=30)).isoformat()
    past_end = (now - timedelta(days=1)).isoformat()
    future = (now + timedelta(days=30)).isoformat()
    store.connection.execute_write(
        """
        CREATE (u:__Entity__ {id: $uid, entity_type: 'User', status: 'active'})
        CREATE (a:__Entity__ {id: 'currencytest-always', entity_type: 'Technology', status: 'active'})
        CREATE (o:__Entity__ {id: 'currencytest-open', entity_type: 'Technology', status: 'active'})
        CREATE (c:__Entity__ {id: 'currencytest-copy', entity_type: 'Technology', status: 'active'})
        CREATE (f:__Entity__ {id: 'currencytest-future', entity_type: 'Technology', status: 'active'})
        CREATE (n:__Entity__ {id: 'currencytest-native', entity_type: 'Technology', status: 'active'})
        CREATE (u)-[:USES {is_latest_belief: true, valid_from: '-inf', valid_to: NULL}]->(a)
        CREATE (u)-[:USES {is_latest_belief: true, valid_from: $past, valid_to: NULL}]->(o)
        CREATE (u)-[:USES {is_latest_belief: true, valid_from: $past, valid_to: $past_end}]->(c)
        CREATE (u)-[:USES {is_latest_belief: true, valid_from: $future, valid_to: NULL}]->(f)
        CREATE (u)-[:USES {is_latest_belief: true, valid_from: $past, valid_to: datetime($past_end)}]->(n)
        """,
        {"uid": _UID, "past": past, "past_end": past_end, "future": future},
    )


class TestClosedOpenReadBoundary:
    def test_all_user_relationships_returns_only_currently_valid(self, store):
        _seed(store)

        rows = store.get_all_user_relationships(_UID)
        targets = {r["entity_id"] for r in rows}

        # '-inf' sentinel and open current edges ARE current.
        assert "currencytest-always" in targets
        assert "currencytest-open" in targets
        # A clamped history copy keeps is_latest_belief=true but its past
        # valid_to fails the currency arm -- the exclusion unit fakes
        # cannot evaluate.
        assert "currencytest-copy" not in targets
        # Future-dated facts are not yet true.
        assert "currencytest-future" not in targets
        # Mixed-type guard: a NATIVE datetime() valid_to compared against the
        # string $now yields null in Cypher, so the row is excluded. Pinned
        # so a backfill from legacy datetime() writers cannot silently
        # resurrect ended facts.
        assert "currencytest-native" not in targets

    def test_targeted_user_relationships_apply_the_same_boundary(self, store):
        _seed(store)

        rows = store.get_user_relationships_to_entities(
            _UID,
            [
                "currencytest-always",
                "currencytest-open",
                "currencytest-copy",
                "currencytest-future",
                "currencytest-native",
            ],
        )
        targets = {r["entity_id"] for r in rows}

        assert targets == {"currencytest-always", "currencytest-open"}
