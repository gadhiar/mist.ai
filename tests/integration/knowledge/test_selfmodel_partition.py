"""Integration coverage for the :__SelfModel__ partition (R1.0).

Validates four behaviours against the disposable eval Neo4j
(``docker-compose.eval-neo4j.yml``).  The live ``mist-neo4j`` container is
NEVER targeted -- the candidate list below excludes it by design.

Start the eval instance first::

    docker compose -f docker-compose.yml -f docker-compose.eval-neo4j.yml
      --profile eval up -d mist-neo4j-eval
"""

from __future__ import annotations

import socket

import pytest

from backend.knowledge import admin
from backend.knowledge.config import Neo4jConfig
from backend.knowledge.curation.graph_writer import RebuildStamps
from backend.knowledge.curation.reconciliation import ReconciliationEngine
from backend.knowledge.storage.graph_executor import GraphExecutor
from backend.knowledge.storage.graph_store import GraphStore
from backend.knowledge.storage.neo4j_connection import Neo4jConnection
from backend.knowledge.storage.partitions import SELF_MODEL_TYPES
from scripts.migrations.selfmodel_partition import migrate
from tests.mocks.embeddings import FakeEmbeddingGenerator

# ---------------------------------------------------------------------------
# Eval endpoint resolution -- NEVER targets the live mist-neo4j:7687
# ---------------------------------------------------------------------------
# Priority order: in-container service name (mist-neo4j-eval:7687), then the
# host-published port fallback (localhost/127.0.0.1:7688).
# The live container is mist-neo4j:7687 and is intentionally absent.
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

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        _ENDPOINT is None,
        reason=(
            "disposable eval Neo4j not running "
            "(docker compose -f docker-compose.yml -f docker-compose.eval-neo4j.yml "
            "--profile eval up -d mist-neo4j-eval)"
        ),
    ),
]

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

_STAMPS = RebuildStamps(
    ontology_version="1.3.0", extraction_version="v-selfmodel-int", model_hash="m-int-test"
)
_RECORDED_AT = "2026-06-15T10:00:00+00:00"

# Prefix for any :__Entity__ nodes this test suite creates so teardown is
# targeted and reruns remain deterministic.
_TEST_ENTITY_PREFIX = "selfmodeltest-"
_TRAIT_TARGET_ID = f"{_TEST_ENTITY_PREFIX}curiosity-int"


@pytest.fixture
def store():
    """GraphStore connected to the eval Neo4j; cleans up self-model and test entities on exit."""
    host, port = _ENDPOINT  # type: ignore[misc]
    conn = Neo4jConnection(
        Neo4jConfig(uri=f"bolt://{host}:{port}", username="neo4j", password="password")
    )
    conn.connect()
    gs = GraphStore(connection=conn, embedding_generator=FakeEmbeddingGenerator())

    # Initialise schema (idempotent on eval) so constraints + indexes exist
    # before any test in this class calls ensure_mist_identity().
    gs.initialize_schema()

    yield gs

    # Teardown: remove the mist-identity self-model node and any test
    # :__Entity__ nodes created by the round-trip test.  The eval DB is
    # disposable, but clean teardown keeps reruns deterministic.
    conn.execute_write(
        "MATCH (n:__SelfModel__) DETACH DELETE n",
        {},
    )
    conn.execute_write(
        "MATCH (n:__Entity__) WHERE n.id STARTS WITH $prefix DETACH DELETE n",
        {"prefix": _TEST_ENTITY_PREFIX},
    )
    # The migration round-trip test creates an :__Entity__:MistIdentity with the
    # canonical (un-prefixed) id 'mist-identity'. If that test fails before the
    # relabel, the node stays in :__Entity__ and escapes both sweeps above, so
    # sweep it by id to keep reruns deterministic.
    conn.execute_write("MATCH (n {id: 'mist-identity'}) DETACH DELETE n", {})
    conn.disconnect()


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestSelfModelPartition:
    """End-to-end partition isolation for the :__SelfModel__ label (R1.0)."""

    def test_mist_identity_carries_selfmodel_label_not_entity(self, store: GraphStore):
        """After ensure_mist_identity(), mist-identity is in :__SelfModel__ and NOT :__Entity__."""
        # Arrange -- schema already initialised by fixture
        store.ensure_mist_identity()

        # Act
        rows = store.connection.execute_query(
            "MATCH (m {id: 'mist-identity'}) "
            "RETURN m:__SelfModel__ AS in_selfmodel, m:__Entity__ AS in_entity",
            {},
        )

        # Assert
        assert rows, "mist-identity node not found after ensure_mist_identity()"
        row = rows[0]
        assert (
            row["in_selfmodel"] is True
        ), "mist-identity missing :__SelfModel__ label -- T3 migration may not have applied"
        assert (
            row["in_entity"] is False
        ), "mist-identity carries :__Entity__ label -- partition isolation violated"

    def test_persona_read_returns_graph_identity_not_fallback(self, store: GraphStore):
        """get_mist_identity_context() reads the graph (seeded HAS_TRAIT), not the fallback.

        The fallback dict in get_mist_identity_context (graph_store.py ~1477)
        returns the SAME id/display_name as a real node, so asserting only on
        identity fields is tautological -- it passes even if the graph was never
        matched. A seeded :__SelfModel__:MistTrait flowing through the persona
        traits list can ONLY happen when the seeded HAS_TRAIT traversal actually
        runs against the graph (the fallback returns EMPTY traits). This also
        exercises T7's seeded self-model traversal end-to-end on real Neo4j.
        """
        # Arrange: identity node + one current HAS_TRAIT edge to a self-model
        # trait. The edge properties satisfy every arm of the `currency` filter
        # in get_mist_identity_context: status != 'orphaned',
        # is_latest_belief=true, valid_to IS NULL, valid_from='-inf' sentinel.
        store.ensure_mist_identity()
        store.connection.execute_write(
            """
            MERGE (t:__SelfModel__:MistTrait {id: 'trait-test-warm'})
            ON CREATE SET t.display_name = 'Warm', t.entity_type = 'MistTrait',
                          t.axis = 'Temperament', t.status = 'active'
            WITH t
            MATCH (m:MistIdentity {id: 'mist-identity'})
            MERGE (m)-[r:HAS_TRAIT]->(t)
            ON CREATE SET r.status = 'active', r.is_latest_belief = true,
                          r.valid_from = '-inf'
            """,
            {},
        )

        # Act
        context = store.get_mist_identity_context()

        # Assert: identity resolved AND the seeded trait flowed through.
        identity = context["identity"]
        assert (
            identity["id"] == "mist-identity"
        ), f"Expected graph node id='mist-identity', got: {identity!r}"
        assert any(t.get("display_name") == "Warm" for t in context["traits"]), (
            "Seeded HAS_TRAIT trait 'Warm' not in context['traits'] "
            f"(empty traits => persona read fell back instead of reading the graph): "
            f"{context['traits']!r}"
        )

    def test_reset_preserves_selfmodel_partition(self, store: GraphStore):
        """reset_graph(include_derived=True) wipes :__Entity__ but leaves :__SelfModel__ intact."""
        # Arrange: seed an :__Entity__ node alongside the self-model node.
        store.ensure_mist_identity()
        store.connection.execute_write(
            "CREATE (:__Entity__ {id: $id, entity_type: 'Concept', status: 'active'})",
            {"id": f"{_TEST_ENTITY_PREFIX}reset-canary"},
        )

        # Confirm both exist before reset.
        pre_sm = store.connection.execute_query(
            "MATCH (m:__SelfModel__ {id: 'mist-identity'}) RETURN count(m) AS n", {}
        )
        assert pre_sm[0]["n"] == 1, "Self-model node missing before reset"
        pre_canary = store.connection.execute_query(
            "MATCH (e:__Entity__ {id: $id}) RETURN count(e) AS n",
            {"id": f"{_TEST_ENTITY_PREFIX}reset-canary"},
        )
        assert pre_canary[0]["n"] == 1, "Canary :__Entity__ node missing before reset"

        # Act: full reset including derived/provenance nodes.
        admin.reset_graph(store.connection, include_derived=True)

        # Assert: :__SelfModel__ partition survives the reset.
        post_sm = store.connection.execute_query(
            "MATCH (m:__SelfModel__ {id: 'mist-identity'}) RETURN count(m) AS n", {}
        )
        assert post_sm[0]["n"] == 1, (
            "mist-identity :__SelfModel__ node was deleted by reset_graph -- "
            "partition isolation violated"
        )

        # Assert: :__Entity__ nodes were wiped (reset did its job).
        post_canary = store.connection.execute_query(
            "MATCH (e:__Entity__ {id: $id}) RETURN count(e) AS n",
            {"id": f"{_TEST_ENTITY_PREFIX}reset-canary"},
        )
        assert (
            post_canary[0]["n"] == 0
        ), "Canary :__Entity__ node survived reset -- reset_graph not cleaning up correctly"

    @pytest.mark.asyncio
    async def test_mist_has_trait_round_trip_from_selfmodel_source(self, store: GraphStore):
        """MIST_HAS_TRAIT from mist-identity lands in the graph when source is :__SelfModel__."""
        # Arrange: ensure the self-model identity node exists.
        store.ensure_mist_identity()

        # The reconciliation Cypher is:
        #   MATCH (s:__Entity__|__SelfModel__ {id: $source})
        #   MATCH (t:__Entity__ {id: $target})
        #   MERGE (s)-[r:MIST_HAS_TRAIT {version_key: $vk}]->(t)
        # The target must exist as an :__Entity__ node for the MATCH to succeed.
        store.connection.execute_write(
            "CREATE (:__Entity__ {id: $id, entity_type: 'Concept', status: 'active'})",
            {"id": _TRAIT_TARGET_ID},
        )

        engine = ReconciliationEngine(
            executor=GraphExecutor(store.connection), rebuild_stamps=_STAMPS
        )

        rel = {
            "source": "mist-identity",
            "type": "MIST_HAS_TRAIT",
            "target": _TRAIT_TARGET_ID,
            "confidence": 0.9,
            "properties": {},
        }

        # Act
        result = await engine.reconcile_turn(
            [rel],
            recorded_at=_RECORDED_AT,
            event_id="int-selfmodel-e1",
            session_id="int-selfmodel-s1",
        )

        # Assert: reconciliation appended at least one edge version.
        assert result.appended >= 1, (
            f"Expected at least 1 appended edge, got appended={result.appended}; "
            f"flags={result.flags!r}"
        )

        # Assert: the edge actually exists in the graph linking :__SelfModel__
        # source to :__Entity__ target.  Without the T8 source-disjunction fix
        # (MATCH (s:__Entity__|__SelfModel__ ...)) this edge would be silently
        # dropped because mist-identity is not an :__Entity__.
        edge_count = store.connection.execute_query(
            "MATCH (m:__SelfModel__ {id: 'mist-identity'})"
            "-[r:MIST_HAS_TRAIT]->"
            "(t:__Entity__ {id: $tid}) "
            "RETURN count(r) AS n",
            {"tid": _TRAIT_TARGET_ID},
        )
        assert edge_count[0]["n"] >= 1, (
            "MIST_HAS_TRAIT edge from :__SelfModel__ source not found in graph -- "
            "T8 source-disjunction fix may not have applied to the reconciliation writer"
        )

    @pytest.mark.asyncio
    async def test_migrate_reconciles_gap_window_orphan_to_single_root(self, store: GraphStore):
        """Gap-window orphan + real :__Entity__ root -> one clean :__SelfModel__ root.

        Reproduces the live R1.0 pre-migration state: a backend boot in the gap
        window between the R1.0 writers shipping and this migration running calls
        ensure_mist_identity(), which MERGEs a :__SelfModel__:MistIdentity stub
        because the real, edge-bearing root still carries :__Entity__. Relabeling
        naively would move the real root into :__SelfModel__ and collide with the
        stub on selfmodel_id_unique (constraint present via initialize_schema in
        the fixture). The migration must delete the stub (the :__Entity__
        original wins) before relabeling.
        """
        # Arrange: the real, edge-bearing root in :__Entity__ with a HAS_TRAIT
        # edge to a self-model trait...
        store.connection.execute_write(
            """
            CREATE (m:__Entity__:MistIdentity {
                id: 'mist-identity', entity_type: 'MistIdentity',
                display_name: 'MIST', self_concept: 'real-root', pronouns: 'she/her'
            })
            CREATE (t:__SelfModel__:MistTrait {
                id: $trait_id, entity_type: 'MistTrait',
                display_name: 'Curious', status: 'active'
            })
            MERGE (m)-[r:HAS_TRAIT]->(t)
            ON CREATE SET r.status = 'active', r.is_latest_belief = true,
                          r.valid_from = '-inf'
            """,
            {"trait_id": _TRAIT_TARGET_ID},
        )
        # ...and the gap-window orphan stub in :__SelfModel__ (no edges), exactly
        # what ensure_mist_identity() MERGEs on a boot in the gap window.
        store.connection.execute_write(
            """
            CREATE (s:__SelfModel__:MistIdentity {
                id: 'mist-identity', entity_type: 'MistIdentity', display_name: 'MIST'
            })
            """,
            {},
        )

        # Act
        await migrate(GraphExecutor(store.connection))

        # Assert: exactly one mist-identity node survives.
        roots = store.connection.execute_query(
            "MATCH (m {id: 'mist-identity'}) RETURN count(m) AS roots", {}
        )
        assert roots[0]["roots"] == 1, "expected a single mist-identity root after migration"

        # Assert: the survivor is the real edge-bearing root, now relabeled into
        # :__SelfModel__ and out of :__Entity__ -- NOT the empty stub.
        detail = store.connection.execute_query(
            "MATCH (m {id: 'mist-identity'}) "
            "OPTIONAL MATCH (m)-[:HAS_TRAIT]->(t) "
            "RETURN m:__SelfModel__ AS in_selfmodel, m:__Entity__ AS in_entity, "
            "m.self_concept AS self_concept, count(t) AS trait_edges",
            {},
        )
        row = detail[0]
        assert row["in_selfmodel"] is True, "survivor must carry :__SelfModel__"
        assert row["in_entity"] is False, "survivor must no longer carry :__Entity__"
        assert (
            row["self_concept"] == "real-root"
        ), "the edge-bearing :__Entity__ root must survive, not the empty stub"
        assert row["trait_edges"] == 1, "the real root's HAS_TRAIT edge must be preserved"

        # Assert: no self-model node remains in :__Entity__.
        leftover = store.connection.execute_query(
            "MATCH (e:__Entity__) WHERE e.entity_type IN $types RETURN count(e) AS n",
            {"types": sorted(SELF_MODEL_TYPES)},
        )
        assert leftover[0]["n"] == 0, "self-model nodes must not remain in :__Entity__"

    @pytest.mark.asyncio
    async def test_dedup_removes_mist_prefixed_shadows_keeps_kebab(self, store: GraphStore):
        """Dual-seeded self-model -> only the canonical kebab set survives.

        Reproduces the live dual-seed: the canonical kebab trait (`trait-warm`,
        rich props, currency-stamped HAS_TRAIT from the ADR-008 bootstrap) plus
        the sparse mist-prefixed shadow (`mist-trait-Warm` from the vault->graph
        regenerator). The dedup migration must delete the shadow and leave the
        kebab node + its HAS_TRAIT edge from the identity root intact, so the
        persona reads the trait exactly once.
        """
        from scripts.migrations.selfmodel_dedup import migrate as dedup_migrate

        # Arrange: identity root + canonical kebab trait (rich, HAS_TRAIT) ...
        store.connection.execute_write(
            """
            CREATE (m:__SelfModel__:MistIdentity {
                id: 'mist-identity', entity_type: 'MistIdentity', display_name: 'MIST'
            })
            CREATE (k:__SelfModel__:MistTrait {
                id: 'trait-warm', entity_type: 'MistTrait', display_name: 'Warm',
                axis: 'tone', description: 'warm and kind', status: 'active'
            })
            MERGE (m)-[:HAS_TRAIT {is_latest_belief: true, valid_from: '-inf'}]->(k)
            """,
            {},
        )
        # ... and the sparse mist-prefixed shadow trait (no content, HAS_TRAIT).
        store.connection.execute_write(
            """
            MATCH (m:__SelfModel__:MistIdentity {id: 'mist-identity'})
            CREATE (s:__SelfModel__:MistTrait {
                id: 'mist-trait-Warm', entity_type: 'MistTrait',
                display_name: 'Warm', status: 'active'
            })
            MERGE (m)-[:HAS_TRAIT]->(s)
            """,
            {},
        )

        # Act
        await dedup_migrate(GraphExecutor(store.connection))

        # Assert: the shadow is gone; the kebab survives with its content.
        traits = store.connection.execute_query(
            "MATCH (t:__SelfModel__:MistTrait) RETURN t.id AS id, t.axis AS axis ORDER BY id",
            {},
        )
        assert [t["id"] for t in traits] == [
            "trait-warm"
        ], "only the canonical kebab trait must survive the dedup"
        assert traits[0]["axis"] == "tone", "the surviving trait must keep its kebab content"

        # Assert: the persona reads the trait exactly once.
        persona = store.connection.execute_query(
            "MATCH (m:MistIdentity {id: 'mist-identity'})-[:HAS_TRAIT]->(t) "
            "RETURN count(t) AS trait_edges",
            {},
        )
        assert persona[0]["trait_edges"] == 1, "persona must list the trait once after dedup"

        # Assert: idempotent -- a second run deletes nothing more.
        await dedup_migrate(GraphExecutor(store.connection))
        recount = store.connection.execute_query(
            "MATCH (t:__SelfModel__:MistTrait) RETURN count(t) AS n", {}
        )
        assert recount[0]["n"] == 1, "second dedup run must be a no-op"
