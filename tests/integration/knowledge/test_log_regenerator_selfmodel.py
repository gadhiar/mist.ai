"""Integration: self-model copy-forward + cross-layer re-derivation (R1.2 Task 4).

Requires BOTH mist-neo4j-eval (source) and mist-neo4j-staging (target) up.
"""

from __future__ import annotations

import socket

import pytest

from backend.knowledge.config import Neo4jConfig
from backend.knowledge.regeneration.log_regenerator import LogRegenerator
from backend.knowledge.storage.neo4j_connection import Neo4jConnection


def _reachable(host: str, port: int) -> bool:
    try:
        socket.create_connection((host, port), timeout=2).close()
        return True
    except OSError:
        return False


_SRC = (
    ("mist-neo4j-eval", 7687)
    if _reachable("mist-neo4j-eval", 7687)
    else (("localhost", 7688) if _reachable("localhost", 7688) else None)
)
_DST = (
    ("mist-neo4j-staging", 7687)
    if _reachable("mist-neo4j-staging", 7687)
    else (("localhost", 7689) if _reachable("localhost", 7689) else None)
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        _SRC is None or _DST is None,
        reason="needs both eval (source) and staging (target) Neo4j up",
    ),
]


def _conn(hp):
    c = Neo4jConnection(
        Neo4jConfig(uri=f"bolt://{hp[0]}:{hp[1]}", username="neo4j", password="password")
    )
    c.connect()
    return c


@pytest.fixture
def conns():
    src, dst = _conn(_SRC), _conn(_DST)
    src.execute_write("MATCH (n) DETACH DELETE n", {})
    dst.execute_write("MATCH (n) DETACH DELETE n", {})
    yield src, dst
    src.execute_write("MATCH (n) DETACH DELETE n", {})
    dst.execute_write("MATCH (n) DETACH DELETE n", {})
    src.disconnect()
    dst.disconnect()


class TestSelfModelCopyForward:
    def test_copy_then_rederive(self, conns):
        src, dst = conns
        # Source: self-model root + one trait (intra-edge) + one cross-layer edge to an entity.
        src.execute_write(
            "CREATE (m:__SelfModel__:MistIdentity {id:'mist-identity', entity_type:'MistIdentity', display_name:'MIST'}) "
            "CREATE (t:__SelfModel__:MistTrait {id:'trait-warm', entity_type:'MistTrait', display_name:'Warm', axis:'tone'}) "
            "MERGE (m)-[:HAS_TRAIT]->(t) "
            "CREATE (e:__Entity__ {id:'python', entity_type:'Technology', display_name:'Python'}) "
            "MERGE (m)-[:MIST_HAS_CAPABILITY]->(e)",
            {},
        )
        # Staging: the replay already rebuilt the entity (fresh node).
        dst.execute_write(
            "CREATE (e:__Entity__ {id:'python', entity_type:'Technology', display_name:'Python'})",
            {},
        )

        regen = LogRegenerator(
            event_store=None, extraction_cache=None, staging_curation_pipeline=None
        )
        copied = regen.copy_self_model_partition(src, dst)
        rederived = regen.rederive_self_model_cross_layer_edges(src, dst)

        # Self-model nodes + intra-edge copied verbatim.
        assert copied == 2  # MistIdentity + MistTrait
        assert (
            dst.execute_query(
                "MATCH (:__SelfModel__:MistIdentity {id:'mist-identity'})-[:HAS_TRAIT]->"
                "(:__SelfModel__:MistTrait {id:'trait-warm'}) RETURN count(*) AS n",
                {},
            )[0]["n"]
            == 1
        )
        # Cross-layer edge re-derived onto the staging entity (by id).
        assert rederived["edges"] == 1 and rederived["skipped"] == 0
        assert (
            dst.execute_query(
                "MATCH (:__SelfModel__:MistIdentity {id:'mist-identity'})-[:MIST_HAS_CAPABILITY]->"
                "(:__Entity__ {id:'python'}) RETURN count(*) AS n",
                {},
            )[0]["n"]
            == 1
        )

    def test_rederive_skips_target_absent_from_staging(self, conns):
        src, dst = conns
        src.execute_write(
            "CREATE (m:__SelfModel__:MistIdentity {id:'mist-identity', entity_type:'MistIdentity'}) "
            "CREATE (e:__Entity__ {id:'ghost', entity_type:'Technology'}) "
            "MERGE (m)-[:MIST_HAS_CAPABILITY]->(e)",
            {},
        )
        dst.execute_write(
            "CREATE (m:__SelfModel__:MistIdentity {id:'mist-identity', entity_type:'MistIdentity'})",
            {},
        )
        regen = LogRegenerator(
            event_store=None, extraction_cache=None, staging_curation_pipeline=None
        )
        regen.copy_self_model_partition(src, dst)
        rederived = regen.rederive_self_model_cross_layer_edges(src, dst)
        assert rederived["edges"] == 0 and rederived["skipped"] == 1  # 'ghost' not in staging
