"""Typed labels against real Neo4j: each seed node carries its own type label.

The seed writes the type as `entity_type` and, since typed labels (6 Nov,
reversing MIS-177 D2, under which an `:__Entity__` node kept only `:User`), as
the node's one ontology type label. Unit fakes pin the query text
(`test_seed_applier.py`, `test_seed_gates.py`); only Neo4j shows what
`SET n:<Type> REMOVE n:A:B:...` leaves on a node, including one the MERGE
matched rather than created, and that `check_node_definitions` binds the
result.

Runs ONLY against the disposable eval instance (docker-compose.eval-neo4j.yml)
and skips when it is not reachable. It empties that instance first and last.
Start the target first:

  docker compose -f docker-compose.yml -f docker-compose.eval-neo4j.yml \
    --profile eval up -d mist-neo4j-eval
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from backend.knowledge.config import Neo4jConfig
from backend.knowledge.eval_isolation import LIVE_NEO4J_ENDPOINTS, assert_neo4j_uri_not_live
from backend.knowledge.seed.applier import apply_seed_documents
from backend.knowledge.seed.gates import check_node_definitions
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

_SEED_VERSION = "mis177-eval-v1"
_NOW = "2026-09-29T00:00:00+00:00"


def _documents() -> list[SeedDocument]:
    return [
        SeedDocument(
            seed_version=_SEED_VERSION,
            nodes=[
                SeedNode(id="lbl-user", type="User", display_name="Eval User"),
                SeedNode(id="lbl-rust", type="Technology", display_name="Rust"),
                SeedNode(id="lbl-acme", type="Organization", display_name="Acme"),
            ],
            facts=[
                SeedFact(subject="lbl-user", predicate="USES", object="lbl-rust"),
                SeedFact(subject="lbl-user", predicate="WORKS_AT", object="lbl-acme"),
            ],
            body="MIS-177 eval seed",
            source_path=Path("mis177-user.md"),
            partition=ENTITY_LABEL,
        ),
        SeedDocument(
            seed_version=_SEED_VERSION,
            nodes=[
                SeedNode(id="lbl-mist", type="MistIdentity", display_name="MIST"),
                SeedNode(id="lbl-trait", type="MistTrait", display_name="Warm"),
            ],
            facts=[SeedFact(subject="lbl-mist", predicate="HAS_TRAIT", object="lbl-trait")],
            body="MIS-177 eval self-model seed",
            source_path=Path("mis177-mist.md"),
            partition=SELF_MODEL_LABEL,
        ),
    ]


@pytest.fixture
def eval_connection():
    host, port = _ENDPOINT
    uri = f"bolt://{host}:{port}"
    # Belt and braces: the candidate list is eval-only, and this test empties
    # the graph it connects to.
    assert_neo4j_uri_not_live(uri, action="the MIS-177 label eval test (full graph wipe)")
    conn = Neo4jConnection(Neo4jConfig(uri=uri, username="neo4j", password="password"))
    conn.connect()
    conn.execute_write("MATCH (n) DETACH DELETE n", {})
    yield conn
    conn.execute_write("MATCH (n) DETACH DELETE n", {})
    conn.disconnect()


def _labels_by_id(conn) -> dict[str, set[str]]:
    rows = conn.execute_query("MATCH (n) RETURN n.id AS id, labels(n) AS labels", {})
    return {row["id"]: set(row["labels"]) for row in rows}


def test_seed_writes_each_node_its_own_type_label(eval_connection):
    conn = eval_connection
    # Two nodes earlier seeds wrote, both seed-stamped. `lbl-acme` carries a
    # stray type label (an older seed authored another type); `lbl-rust` was
    # seeded under MIS-177 D2 and carries none. The MERGE matches both: the SET
    # must add the node's own label and the REMOVE must strip the stray one.
    conn.execute_write(
        "CREATE (:__Entity__:Project {id: 'lbl-acme', entity_type: 'Project', "
        "display_name: 'Acme', provenance: 'seed', seed_version: $v})",
        {"v": _SEED_VERSION},
    )
    conn.execute_write(
        "CREATE (:__Entity__ {id: 'lbl-rust', entity_type: 'Technology', "
        "display_name: 'Rust', provenance: 'seed', seed_version: $v})",
        {"v": _SEED_VERSION},
    )

    apply_seed_documents(conn, _documents(), seed_version=_SEED_VERSION, now_iso=_NOW)

    labels = _labels_by_id(conn)
    assert labels["lbl-user"] == {ENTITY_LABEL, "User"}
    assert labels["lbl-rust"] == {ENTITY_LABEL, "Technology"}
    assert labels["lbl-acme"] == {ENTITY_LABEL, "Organization"}
    assert labels["lbl-mist"] == {SELF_MODEL_LABEL, "MistIdentity"}
    assert labels["lbl-trait"] == {SELF_MODEL_LABEL, "MistTrait"}

    rows = conn.execute_query("MATCH (n) RETURN n.id AS id, n.entity_type AS entity_type", {})
    assert {r["id"]: r["entity_type"] for r in rows} == {
        "lbl-user": "User",
        "lbl-rust": "Technology",
        "lbl-acme": "Organization",
        "lbl-mist": "MistIdentity",
        "lbl-trait": "MistTrait",
    }

    gate = check_node_definitions(conn, _documents(), seed_version=_SEED_VERSION)
    assert gate.passed, gate.failures


def test_node_definitions_gate_fails_on_a_stray_or_missing_label_or_a_wrong_type(
    eval_connection,
):
    conn = eval_connection
    apply_seed_documents(conn, _documents(), seed_version=_SEED_VERSION, now_iso=_NOW)

    conn.execute_write("MATCH (n:__Entity__ {id: 'lbl-rust'}) SET n:Project", {})
    conn.execute_write("MATCH (n:__Entity__ {id: 'lbl-acme'}) SET n.entity_type = 'Project'", {})
    conn.execute_write("MATCH (n:__Entity__ {id: 'lbl-user'}) REMOVE n:User", {})

    gate = check_node_definitions(conn, _documents(), seed_version=_SEED_VERSION)
    assert not gate.passed
    failed = " ".join(gate.failures)
    for node_id in ("'lbl-rust'", "'lbl-acme'", "'lbl-user'"):
        assert node_id in failed, gate.failures
    assert len(gate.failures) == 3, gate.failures


def test_node_definitions_gate_fails_on_a_node_seeded_without_its_type_label(eval_connection):
    """A graph seeded under MIS-177 D2 fails `seed-verify` until it is reseeded."""
    conn = eval_connection
    apply_seed_documents(conn, _documents(), seed_version=_SEED_VERSION, now_iso=_NOW)

    conn.execute_write("MATCH (n:__Entity__ {id: 'lbl-acme'}) REMOVE n:Organization", {})

    gate = check_node_definitions(conn, _documents(), seed_version=_SEED_VERSION)
    assert not gate.passed
    assert len(gate.failures) == 1 and "'lbl-acme'" in gate.failures[0], gate.failures
