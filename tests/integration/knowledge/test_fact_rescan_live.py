"""The cost of an unchanged 10k-fact rescan against the disposable eval Neo4j.

The refresher rescans the whole graph every `MIST_MEMORY_RESCAN_INTERVAL_S` (default 300 s)
besides its dirty marks, so an unchanged rescan must stay cheap. The unit tier times it with
fakes at the graph; this file times the real paging Cypher on Neo4j, and the gate is 3 s, 1% of
the default interval. It runs against the eval instance (docker-compose.eval-neo4j.yml) only --
the canonical graph is never touched -- and skips when it is not running.

Start the target first:
  docker compose -f docker-compose.yml -f docker-compose.eval-neo4j.yml \
    --profile eval up -d mist-neo4j-eval
"""

from __future__ import annotations

import socket
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC

import pytest

from backend.knowledge.config import Neo4jConfig
from backend.knowledge.storage.graph_executor import GraphExecutor
from backend.knowledge.storage.neo4j_connection import Neo4jConnection
from backend.memory.fact_index import FactIndex
from backend.memory.fact_source import FactIndexRefresher, GraphFactSource
from backend.memory.index_db import IndexRebuild, open_index, open_reader
from tests.fakes.memory_fakes import FakeClock, FakeEmbedder

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

_PREFIX = "factrescantest"
_UID = f"{_PREFIX}-user"
_MODEL = "all-MiniLM-L6-v2"
_FACTS = 10_000
_UNCHANGED_RESCAN_LIMIT_S = 3.0


def _cleanup(conn: Neo4jConnection) -> None:
    """Delete every node this file seeds, with its edges."""
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


def _seed(conn: Neo4jConnection) -> None:
    """Seed `_FACTS` current USES facts from one user, each to its own technology."""
    conn.execute_write(
        """
        CREATE (u:__Entity__ {id: $uid, display_name: 'User', entity_type: 'User',
                              status: 'active'})
        WITH u
        UNWIND range(0, $count - 1) AS i
        CREATE (t:__Entity__ {id: $prefix + '-tech-' + toString(i),
                              display_name: 'Tech ' + toString(i),
                              entity_type: 'Technology', status: 'active'})
        CREATE (u)-[:USES {is_latest_belief: true,
                           version_key: 'e' + toString(i) + '|open|open',
                           valid_from: '2026-01-01T00:00:00+00:00', valid_to: NULL,
                           recorded_at: '2026-01-01T00:00:00+00:00', confidence: 0.9,
                           source_utterance_id: 'utt-' + toString(i)}]->(t)
        """,
        {"uid": _UID, "prefix": _PREFIX, "count": _FACTS},
    )


class TestFactRescanLive:
    @pytest.mark.asyncio
    async def test_an_unchanged_10k_fact_rescan_takes_at_most_three_seconds(
        self, connection, tmp_path, capsys
    ):
        _seed(connection)
        path = tmp_path / "memory_index.db"
        writer = open_index(path, embedding_model_id=_MODEL)
        reader = open_reader(path, embedding_model_id=_MODEL)
        index = FactIndex(writer, FakeEmbedder(), tz=UTC, reader=reader)
        try:
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="memory-index") as pool:
                refresher = FactIndexRefresher(
                    GraphFactSource(GraphExecutor(connection)),
                    index,
                    pool,
                    clock=FakeClock(),
                    debounce_s=30.0,
                    rescan_interval_s=300.0,
                    periodic=False,
                    rebuild=IndexRebuild(path, embedding_model_id=_MODEL),
                )
                cold = await refresher.refresh()

                started = time.perf_counter()
                unchanged = await refresher.refresh()
                unchanged_s = time.perf_counter() - started
        finally:
            reader.close()
            writer.close()

        with capsys.disabled():
            print(
                f"\nunchanged rescan of {unchanged.unchanged} facts on the eval Neo4j: "
                f"{unchanged_s:.2f} s"
            )
        assert cold.inserted >= _FACTS
        assert unchanged.unchanged >= _FACTS
        assert (unchanged.inserted, unchanged.updated, unchanged.deleted) == (0, 0, 0)
        assert unchanged_s <= _UNCHANGED_RESCAN_LIMIT_S
