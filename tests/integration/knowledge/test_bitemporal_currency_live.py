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
from backend.knowledge.curation.graph_writer import RebuildStamps
from backend.knowledge.curation.reconciliation import ReconciliationEngine
from backend.knowledge.storage.graph_executor import GraphExecutor
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


_STAMPS = RebuildStamps(
    ontology_version="1.2.1", extraction_version="v-int-test", model_hash="m-int-test"
)
_RECORDED_AT = "2026-06-10T12:00:00+00:00"


class TestSameTurnSingleArbitrationLive:
    """Same-turn SINGLE arbitration end-state on live Cypher (deep-review #2).

    'I left Zeta for Acme' in ONE turn: WORKS_AT cease(zeta, stated stop) +
    assert(acme). The unit layer can only witness the processing ORDER (the
    static fake cannot model write-read feedback); only a real Neo4j run can
    prove the end-state -- zeta's belief closed at the STATED stop (not
    recorded_at), close reason 'cease', no cease_without_prior flag, acme open,
    exactly one closed copy of zeta.
    """

    def _seed_open_zeta(self, store: GraphStore, zeta_from: str) -> None:
        store.connection.execute_write(
            """
            MERGE (u:__Entity__ {id: $uid})
              ON CREATE SET u.entity_type = 'User', u.status = 'active'
            MERGE (z:__Entity__ {id: 'currencytest-zeta'})
              ON CREATE SET z.entity_type = 'Organization', z.status = 'active'
            MERGE (a:__Entity__ {id: 'currencytest-acme'})
              ON CREATE SET a.entity_type = 'Organization', a.status = 'active'
            CREATE (u)-[:WORKS_AT {
                is_latest_belief: true, valid_from: $zeta_from, valid_to: NULL,
                recorded_at: $zeta_from, source_utterance_id: 'seed-zeta',
                version_key: 'seed-zeta|open|open', confidence: 0.9,
                source_type: 'extracted', temporal_status: 'current'
            }]->(z)
            """,
            {"uid": _UID, "zeta_from": zeta_from},
        )

    @pytest.mark.asyncio
    async def test_same_turn_cease_then_assert_closes_zeta_at_stated_stop(self, store):
        # Seed an open WORKS_AT(user, zeta) starting well before the stated stop.
        self._seed_open_zeta(store, zeta_from="2024-01-01T00:00:00+00:00")
        # Stated stop 'left in March 2026'. valid_to is a closed-open EXCLUSIVE
        # upper bound (intervals design 4.5): a month-precision TO bound covers
        # the whole stated period, so '2026-03' clamps to 2026-04-01 (valid
        # THROUGH March). The point under test is that the close uses this
        # STATED stop, not recorded_at (2026-06-10).
        stated_stop = "2026-04-01T00:00:00+00:00"

        engine = ReconciliationEngine(
            executor=GraphExecutor(store.connection), rebuild_stamps=_STAMPS
        )
        # Same turn, listed assert-first so only the kind-ordering sort can put
        # the cease before the assert ('acme' < 'zeta' alphabetically).
        rels = [
            {
                "source": _UID,
                "target": "currencytest-acme",
                "type": "WORKS_AT",
                "properties": {"assertion_kind": "assert", "confidence": 0.9},
            },
            {
                "source": _UID,
                "target": "currencytest-zeta",
                "type": "WORKS_AT",
                "properties": {
                    "assertion_kind": "cease",
                    "temporal_status": "past",
                    "end_date": "2026-03",
                },
            },
        ]

        result = await engine.reconcile_turn(
            rels, recorded_at=_RECORDED_AT, event_id="int-e1", session_id="int-s1"
        )

        # No spurious cease_without_prior (the cease found its open prior).
        assert not any("cease_without_prior" in f for f in result.flags), result.flags

        # acme is the sole currently-valid WORKS_AT for the user.
        open_rows = store.connection.execute_query(
            "MATCH (u:__Entity__ {id: $uid})-[r:WORKS_AT]->(t:__Entity__) "
            "WHERE coalesce(r.is_latest_belief, true) AND r.valid_to IS NULL "
            "RETURN t.id AS target, r.supersession_reason AS reason",
            {"uid": _UID},
        )
        open_targets = {row["target"] for row in open_rows}
        assert open_targets == {"currencytest-acme"}

        # zeta: exactly one surviving (latest) version, closed at the STATED
        # stop -- not recorded_at -- with close reason 'cease'.
        zeta_latest = store.connection.execute_query(
            "MATCH (u:__Entity__ {id: $uid})-[r:WORKS_AT]->(t:__Entity__ {id: 'currencytest-zeta'})"
            " WHERE coalesce(r.is_latest_belief, true) "
            "RETURN r.valid_to AS valid_to, r.supersession_reason AS reason, "
            "r.temporal_status AS temporal_status",
            {"uid": _UID},
        )
        assert len(zeta_latest) == 1
        assert zeta_latest[0]["valid_to"] == stated_stop  # stated, not recorded_at
        assert zeta_latest[0]["valid_to"] != _RECORDED_AT
        assert zeta_latest[0]["reason"] == "cease"
        assert zeta_latest[0]["temporal_status"] == "past"

        # The original open prior was transaction-closed (no longer latest).
        closed_prior = store.connection.execute_query(
            "MATCH (u:__Entity__ {id: $uid})-[r:WORKS_AT]->(t:__Entity__ {id: 'currencytest-zeta'})"
            " WHERE r.is_latest_belief = false AND r.source_utterance_id = 'seed-zeta' "
            "RETURN count(r) AS n",
            {"uid": _UID},
        )
        assert closed_prior[0]["n"] == 1
