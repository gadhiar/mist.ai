"""Live-Cypher paging of `current_facts_query` against the disposable eval Neo4j.

Unit tests can only assert the Cypher text. Whether every latest-belief version comes back as
its own row (two intervals of one fact, a clamped copy, a tombstone, a future-dated version),
whether the excluded edges stay out, and whether keyset paging on `after_key` neither repeats
nor drops a row are evaluated by Neo4j itself. These tests run against the eval instance
(docker-compose.eval-neo4j.yml) only -- the canonical graph is never touched -- and skip when
it is not running.

Start the target first:
  docker compose -f docker-compose.yml -f docker-compose.eval-neo4j.yml \
    --profile eval up -d mist-neo4j-eval
"""

from __future__ import annotations

import socket
from datetime import UTC, datetime, timedelta

import pytest

from backend.knowledge.config import Neo4jConfig
from backend.knowledge.storage.graph_executor import GraphExecutor
from backend.knowledge.storage.graph_store import current_facts_query
from backend.knowledge.storage.neo4j_connection import Neo4jConnection
from backend.memory.fact_index import fact_row_key
from backend.memory.fact_source import GraphFactSource
from backend.memory.models import fact_key

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

_PREFIX = "currentfactstest"
_UID = f"{_PREFIX}-user"


def _cleanup(conn: Neo4jConnection) -> None:
    """Delete every node this file seeds, whatever its label.

    The `__Provenance__` node must go too: `provenance_id_unique` would refuse the next seed's
    copy of it.
    """
    conn.execute_write(
        "MATCH (n) WHERE n.id STARTS WITH $prefix DETACH DELETE n", {"prefix": _PREFIX}
    )


@pytest.fixture
def connection():
    host, port = _ENDPOINT
    conn = Neo4jConnection(
        Neo4jConfig(uri=f"bolt://{host}:{port}", username="neo4j", password="password")
    )
    conn.connect()
    # A run that died before its teardown leaves nodes behind; clear them before seeding.
    _cleanup(conn)
    yield conn
    _cleanup(conn)
    conn.disconnect()


def _seed(conn: Neo4jConnection) -> dict[str, str]:
    """Seed one fact with every kind of latest-belief version, plus edges that must stay out.

    Returns the version keys that must come back, by what they stand for.
    """
    now = datetime.now(UTC)
    early = (now - timedelta(days=400)).isoformat()
    early_end = (now - timedelta(days=300)).isoformat()
    later = (now - timedelta(days=30)).isoformat()
    clamp_end = (now - timedelta(days=10)).isoformat()
    tomb = (now - timedelta(days=5)).isoformat()
    future = (now + timedelta(days=30)).isoformat()
    keys = {
        "first_interval": f"e1|{early}|{early_end}",
        "second_interval": f"e2|{later}|open",
        "clamped_copy": f"e3|{later}|{clamp_end}",
        "tombstone": f"e4|{tomb}|{tomb}",
        "future": f"e5|{future}|open",
    }
    conn.execute_write(
        """
        CREATE (u:__Entity__ {id: $uid, display_name: 'User', entity_type: 'User',
                              status: 'active'})
        CREATE (v:__Entity__ {id: $prefix + '-vim', display_name: 'Vim',
                              entity_type: 'Technology', status: 'active'})
        CREATE (r:__Entity__ {id: $prefix + '-rust', display_name: 'Rust',
                              entity_type: 'Technology', status: 'active'})
        CREATE (z:__Entity__ {id: $prefix + '-zig', entity_type: 'Technology',
                              status: 'active'})
        CREATE (p:__Provenance__ {id: $prefix + '-utterance'})
        CREATE (u)-[:USES {is_latest_belief: true, version_key: $first_interval,
                           valid_from: $early, valid_to: $early_end, recorded_at: $early,
                           confidence: 0.9, source_utterance_id: 'e1'}]->(v)
        CREATE (u)-[:USES {is_latest_belief: true, version_key: $second_interval,
                           valid_from: $later, valid_to: NULL, recorded_at: $later,
                           confidence: 0.8, source_utterance_id: 'e2'}]->(v)
        CREATE (u)-[:USES {is_latest_belief: true, version_key: $clamped_copy,
                           valid_from: $later, valid_to: $clamp_end, recorded_at: $clamp_end,
                           confidence: 0.8, source_utterance_id: 'e3'}]->(v)
        CREATE (u)-[:USES {is_latest_belief: true, version_key: $tombstone,
                           valid_from: $tomb, valid_to: $tomb, recorded_at: $tomb,
                           confidence: 0.8, source_utterance_id: 'e4'}]->(v)
        CREATE (u)-[:USES {is_latest_belief: true, version_key: $future,
                           valid_from: $future_at, valid_to: NULL, recorded_at: $later,
                           confidence: 0.7, source_utterance_id: 'e5'}]->(v)
        CREATE (u)-[:USES {is_latest_belief: false, version_key: 'e0|superseded|open',
                           valid_from: $early, valid_to: NULL, recorded_at: $early}]->(v)
        CREATE (u)-[:USES {is_latest_belief: true, status: 'orphaned',
                           version_key: 'e6|orphan|open', recorded_at: $later}]->(r)
        CREATE (r)-[:IS_A]->(z)
        CREATE (u)-[:SUPERSEDES {is_latest_belief: true}]->(z)
        CREATE (u)-[:EXTRACTED_FROM]->(p)
        """,
        {
            "uid": _UID,
            "prefix": _PREFIX,
            "early": early,
            "early_end": early_end,
            "later": later,
            "clamp_end": clamp_end,
            "tomb": tomb,
            "future_at": future,
            **keys,
        },
    )
    return keys


def _ours(rows: list[dict]) -> list[dict]:
    return [row for row in rows if str(row["subject_id"]).startswith(_PREFIX)]


def _page_all(conn: Neo4jConnection, limit: int) -> list[dict]:
    rows: list[dict] = []
    after: str | None = None
    while True:
        page = conn.execute_query(*current_facts_query(after, limit))
        rows.extend(page)
        if len(page) < limit:
            return rows
        after = page[-1]["page_key"]


class TestCurrentFactsLive:
    def test_every_latest_belief_version_is_its_own_row(self, connection):
        keys = _seed(connection)

        rows = _ours(_page_all(connection, 1000))

        vim = [row for row in rows if row["object_id"] == f"{_PREFIX}-vim"]
        assert sorted(row["version_key"] for row in vim) == sorted(keys.values())
        # The structural IS_A edge has no version key and still comes back, once.
        structural = [row for row in rows if row["rel_type"] == "IS_A"]
        assert len(structural) == 1
        assert structural[0]["version_key"] is None
        assert structural[0]["object_name"] == f"{_PREFIX}-zig"  # name falls back to the id
        # Superseded, orphaned, non-allowlisted and provenance edges stay out.
        assert len(rows) == len(keys) + 1
        assert {row["rel_type"] for row in rows} == {"USES", "IS_A"}
        # Both endpoint types come from `entity_type`.
        assert {(row["subject_type"], row["object_type"]) for row in vim} == {
            ("User", "Technology")
        }
        assert (structural[0]["subject_type"], structural[0]["object_type"]) == (
            "Technology",
            "Technology",
        )

    def test_a_node_without_an_entity_type_gives_none(self, connection):
        connection.execute_write(
            """
            CREATE (u:__Entity__ {id: $uid, display_name: 'User', entity_type: 'User'})
            CREATE (x:__Entity__ {id: $prefix + '-untyped', display_name: 'Untyped'})
            CREATE (u)-[:USES {is_latest_belief: true, version_key: 'e1|open|open'}]->(x)
            """,
            {"uid": _UID, "prefix": _PREFIX},
        )

        (row,) = _ours(_page_all(connection, 1000))

        assert (row["subject_type"], row["object_type"]) == ("User", None)

    @pytest.mark.parametrize("rel_type", ["RECOMMENDS", "HAS_HABIT"])
    def test_the_v1_3_0_user_centric_edges_are_served(self, connection, rel_type):
        connection.execute_write(
            f"""
            CREATE (u:__Entity__ {{id: $uid, display_name: 'User', entity_type: 'User'}})
            CREATE (x:__Entity__ {{id: $prefix + '-target', display_name: 'Target',
                                  entity_type: 'Concept'}})
            CREATE (u)-[:{rel_type} {{is_latest_belief: true, version_key: 'e1|open|open'}}]->(x)
            """,
            {"uid": _UID, "prefix": _PREFIX},
        )

        (row,) = _ours(_page_all(connection, 1000))

        assert row["rel_type"] == rel_type

    def test_paging_neither_repeats_nor_drops_a_row(self, connection):
        _seed(connection)

        whole = _page_all(connection, 100_000)
        paged = _page_all(connection, 2)

        paged_keys = [row["page_key"] for row in paged]
        assert len(paged_keys) == len(set(paged_keys))
        assert paged_keys == sorted(paged_keys)
        assert set(paged_keys) == {row["page_key"] for row in whole}
        assert len(_ours(paged)) == 6

    @pytest.mark.asyncio
    async def test_graph_fact_source_keys_each_version(self, connection):
        keys = _seed(connection)

        rows = await GraphFactSource(GraphExecutor(connection), page_size=2).snapshot()

        ours = [row for row in rows if row.subject_id.startswith(_PREFIX)]
        vim_key = fact_key(_UID, "USES", f"{_PREFIX}-vim")
        assert {row.row_key for row in ours if row.fact_key == vim_key} == {
            fact_row_key(vim_key, version_key) for version_key in keys.values()
        }
        assert len({row.row_key for row in rows}) == len(rows)
        assert {(row.subject_type, row.object_type) for row in ours} == {
            ("User", "Technology"),
            ("Technology", "Technology"),
        }
