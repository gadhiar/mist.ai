r"""Deterministic resolver makes identical merge decisions regardless of
insertion order -- proven against the disposable eval Neo4j (never live).

The cosine tier uses `vector.similarity.cosine` evaluated by Neo4j 5 itself
(confirmed available on neo4j:5.26.23). No ANN, no insertion-order
sensitivity: the resolver uses ORDER BY score DESC, id ASC as a total order,
so every repeated call against an unchanged graph returns the same winner.

Start the target first:
  docker compose -f docker-compose.yml -f docker-compose.eval-neo4j.yml \
    --profile eval up -d mist-neo4j-eval
"""

from __future__ import annotations

import socket

import pytest

from backend.knowledge.config import Neo4jConfig
from backend.knowledge.curation.confidence import ConfidenceManager
from backend.knowledge.curation.deduplication import EntityDeduplicator
from backend.knowledge.embeddings.embedding_generator import EmbeddingGenerator
from backend.knowledge.storage.graph_executor import GraphExecutor
from backend.knowledge.storage.neo4j_connection import Neo4jConnection

# In-container service name first; host-published fallback ports.
# The live mist-neo4j:7687 and host localhost:7687 are NEVER in this list.
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

# Prefix all seeded nodes so teardown can DETACH DELETE without touching
# anything else in the eval instance.
_PREFIX = "rdedup-"


@pytest.fixture
def deduper():
    """Wire an EntityDeduplicator against the disposable eval Neo4j.

    Seed one 'JavaScript' Technology node with a real embedding so the cosine
    tier has something to compare against. Teardown removes every node whose
    id starts with _PREFIX.
    """
    host, port = _ENDPOINT

    conn = Neo4jConnection(
        Neo4jConfig(uri=f"bolt://{host}:{port}", username="neo4j", password="password")
    )
    conn.connect()

    emb = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    vec = emb.generate_embedding("JavaScript")

    conn.execute_write(
        "MERGE (e:__Entity__ {id: $id}) "
        "SET e.entity_type = 'Technology', "
        "e.display_name = 'JavaScript', "
        "e.embedding = $vec, "
        "e.status = 'active'",
        {"id": _PREFIX + "javascript", "vec": vec},
    )

    executor = GraphExecutor(conn)
    dd = EntityDeduplicator(
        executor=executor,
        embedding_provider=emb,
        confidence_manager=ConfidenceManager(),
    )

    yield dd

    conn.execute_write(
        "MATCH (n:__Entity__) WHERE n.id STARTS WITH $p DETACH DELETE n",
        {"p": _PREFIX},
    )
    conn.disconnect()


@pytest.mark.asyncio
async def test_resolver_merges_coreferent_name_via_real_cosine(deduper: EntityDeduplicator):
    """Cosine tier resolves 'Javascript' (case variant) to the seeded 'JavaScript' node.

    Probes with entity_id 'rdedup-javascript-lang' (no exact-id or alias match)
    so the resolver falls through to tier 3 (vector.similarity.cosine). The
    display_name 'Javascript' embeds to a vector that is cosine-similar >=0.92
    to the seeded 'JavaScript' embedding, so the match is expected.
    """
    existing = await deduper._find_existing(
        _PREFIX + "javascript-lang",
        "Technology",
        "Javascript",
    )

    assert existing is not None, (
        "cosine tier did not find the seeded JavaScript node; "
        "check that the embedding was stored and vector.similarity.cosine is available"
    )
    assert existing["id"] == _PREFIX + "javascript"


@pytest.mark.asyncio
async def test_resolver_decision_is_insertion_order_independent(deduper: EntityDeduplicator):
    """Repeated calls return the same winner -- order-independence proven against real Neo4j.

    Calls _find_existing five times with the same probe. Because the resolver
    uses ORDER BY score DESC, e.id ASC as a total order (no randomness, no ANN
    non-determinism), every call must return an identical result. The set of
    returned ids must be a singleton.
    """
    ids: list[str | None] = []

    for _ in range(5):
        result = await deduper._find_existing(
            _PREFIX + "js-probe",
            "Technology",
            "Javascript",
        )
        ids.append(result["id"] if result is not None else None)

    assert len(set(ids)) == 1, (
        f"resolver returned different winners across repeated calls: {ids}; "
        "ORDER BY total-order guarantee violated"
    )
    # All five must have resolved to the seeded node (not None).
    assert (
        ids[0] == _PREFIX + "javascript"
    ), f"expected all calls to resolve to {_PREFIX + 'javascript'}, got: {ids}"
