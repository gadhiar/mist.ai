"""KG-125 (MIS-175) against real Neo4j: seed provenance survives a clamp.

Unit fakes pin the parameters and the query text; only Neo4j shows what the
statements actually leave on the elements -- in particular that a NULL
`$seed_origin_version` writes no property, that `MERGE ... SET` on the seed
edge lands every Option A value, and that the seed-only cutover probe
(`extraction_backlog/cutover.py`, `probe_graph`) passes a freshly seeded graph
and refuses one holding a clamped copy of a seed edge.

Runs ONLY against the disposable eval instance (docker-compose.eval-neo4j.yml)
and skips when it is not reachable. It empties that instance first: the probe
counts every element in the graph, so a pass is only meaningful over a graph
holding nothing but this test's seed. Start the target first:

  docker compose -f docker-compose.yml -f docker-compose.eval-neo4j.yml \
    --profile eval up -d mist-neo4j-eval
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from backend.extraction_backlog.cutover import probe_graph
from backend.knowledge.config import Neo4jConfig
from backend.knowledge.curation.graph_writer import RebuildStamps
from backend.knowledge.curation.reconciliation import ReconciliationEngine
from backend.knowledge.eval_isolation import LIVE_NEO4J_ENDPOINTS, assert_neo4j_uri_not_live
from backend.knowledge.seed.applier import apply_seed_documents
from backend.knowledge.seed.models import SeedDocument, SeedFact, SeedNode
from backend.knowledge.storage.graph_executor import GraphExecutor
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

_SEED_VERSION = "kg125-eval-v1"
_NOW = "2026-09-01T00:00:00+00:00"
_RECORDED_AT = "2026-09-10T12:00:00+00:00"
_STAMPS = RebuildStamps(
    ontology_version="1.2.0", extraction_version="v-kg125", model_hash="m-kg125"
)


def _documents() -> list[SeedDocument]:
    return [
        SeedDocument(
            seed_version=_SEED_VERSION,
            nodes=[
                SeedNode(id="kg125-user", type="User", display_name="Eval User"),
                SeedNode(id="kg125-rust", type="Technology", display_name="Rust"),
                SeedNode(id="kg125-acme", type="Organization", display_name="Acme"),
            ],
            facts=[
                SeedFact(subject="kg125-user", predicate="USES", object="kg125-rust"),
                SeedFact(subject="kg125-user", predicate="WORKS_AT", object="kg125-acme"),
            ],
            body="KG-125 eval seed",
            source_path=Path("kg125-eval.md"),
        )
    ]


@pytest.fixture
def eval_connection():
    host, port = _ENDPOINT
    uri = f"bolt://{host}:{port}"
    # Belt and braces: the candidate list is eval-only, and this test empties
    # the graph it connects to.
    assert_neo4j_uri_not_live(uri, action="the KG-125 eval test (full graph wipe)")
    conn = Neo4jConnection(Neo4jConfig(uri=uri, username="neo4j", password="password"))
    conn.connect()
    conn.execute_write("MATCH (n) DETACH DELETE n", {})
    yield conn
    conn.execute_write("MATCH (n) DETACH DELETE n", {})
    conn.disconnect()


@pytest.mark.asyncio
async def test_clamped_seed_copy_keeps_seed_origin_and_fails_the_seed_only_probe(
    eval_connection,
):
    conn = eval_connection
    apply_seed_documents(conn, _documents(), seed_version=_SEED_VERSION, now_iso=_NOW)

    # --- the seeded graph: every element is seed, and the probe passes ---
    edges = conn.execute_query(
        "MATCH ()-[r]->() RETURN type(r) AS t, r.provenance AS provenance, "
        "r.source_type AS source_type, r.confidence AS confidence, "
        "r.seed_version AS seed_version, r.ontology_version AS ov, "
        "r.extraction_version AS ev, r.model_hash AS mh",
        {},
    )
    assert len(edges) == 2
    for e in edges:
        assert (e["provenance"], e["source_type"], e["confidence"]) == ("seed", "stated", 1.0)
        assert e["seed_version"] == _SEED_VERSION
        assert (e["ov"], e["ev"], e["mh"]) == (None, None, None)
    nodes = conn.execute_query("MATCH (n) RETURN n.provenance AS provenance", {})
    assert len(nodes) == 3
    assert {n["provenance"] for n in nodes} == {"seed"}

    before = probe_graph(conn)
    assert before.violations() == [], before.as_dict()

    # --- clamp one seed edge: the user stopped using Rust ---
    engine = ReconciliationEngine(executor=GraphExecutor(conn), rebuild_stamps=_STAMPS)
    result = await engine.reconcile_turn(
        [
            {
                "source": "kg125-user",
                "target": "kg125-rust",
                "type": "USES",
                "assertion_kind": "cease",
                "properties": {},
            }
        ],
        recorded_at=_RECORDED_AT,
        event_id="kg125-e1",
        session_id="kg125-s1",
    )
    assert result.closed == 1 and result.appended == 1, result

    rows = conn.execute_query(
        "MATCH (:__Entity__ {id: 'kg125-user'})-[r:USES]->(:__Entity__ {id: 'kg125-rust'}) "
        "RETURN r.provenance AS provenance, r.source_type AS source_type, "
        "r.confidence AS confidence, r.seed_version AS seed_version, "
        "r.seed_origin_version AS seed_origin_version, r.is_latest_belief AS latest, "
        "r.valid_to AS valid_to, r.model_hash AS mh, "
        "keys(r) AS keys",
        {},
    )
    assert len(rows) == 2
    original = [r for r in rows if r["seed_version"] is not None]
    copy = [r for r in rows if r["seed_version"] is None]
    assert len(original) == 1 and len(copy) == 1

    # The applier's row is transaction-closed but still the applier's.
    assert original[0]["latest"] is False
    assert original[0]["seed_version"] == _SEED_VERSION
    assert "seed_origin_version" not in original[0]["keys"]

    # The clamped copy is still a seed fact, with its lineage, and still
    # carries the extraction stamps.
    c = copy[0]
    assert (c["provenance"], c["source_type"], c["confidence"]) == ("seed", "stated", 1.0)
    assert c["seed_origin_version"] == _SEED_VERSION
    assert "seed_version" not in c["keys"]
    assert c["latest"] is True
    assert c["valid_to"] == _RECORDED_AT
    assert c["mh"] == "m-kg125"

    # --- and the seed-only probe now refuses the graph ---
    after = probe_graph(conn)
    assert after.relationships_without_seed_version == 1, after.as_dict()
    assert after.relationships_with_extraction_stamp == 1, after.as_dict()
    assert after.violations(), after.as_dict()
