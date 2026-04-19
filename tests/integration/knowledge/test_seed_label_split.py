"""ADR-009 acceptance: seed produces clean :__Entity__ / :__Provenance__ split."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from backend.knowledge import admin
from backend.knowledge.config import Neo4jConfig
from backend.knowledge.storage.neo4j_connection import Neo4jConnection

pytestmark = pytest.mark.integration


@pytest.fixture
def real_neo4j_connection():
    """Provide a real Neo4jConnection targeting the running mist-neo4j container."""
    config = Neo4jConfig(
        uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        username=os.environ.get("NEO4J_USER", "neo4j"),
        password=os.environ.get("NEO4J_PASSWORD", "password"),
    )
    conn = Neo4jConnection(config=config)
    conn.connect()
    yield conn
    conn.disconnect()


def test_seed_yields_only_entity_nodes(real_neo4j_connection):
    """After graph-reset --include-derived + seed, graph has 31 :__Entity__ / 0 :__Provenance__."""
    conn = real_neo4j_connection

    # Full wipe — removes both :__Entity__ and :__Provenance__ nodes.
    admin.reset_graph(conn, include_derived=True)
    admin.ensure_schema(conn)

    seed_path = Path(__file__).parents[3] / "scripts" / "seed_data.yaml"
    seed_data = admin.load_seed_yaml(seed_path)
    admin.apply_seed(conn, seed_data)

    # --- :__Entity__ count ---
    entity_rows = admin.count_nodes_by_type(conn)
    total_entities = sum(row["count"] for row in entity_rows)
    assert (
        total_entities == 31
    ), f"Expected 31 seeded :__Entity__ nodes, got {total_entities}: {entity_rows}"

    # --- :__Provenance__ count (must be 0 post-seed) ---
    provenance_rows = admin.provenance_counts_by_type(conn)
    total_prov = sum(row["count"] for row in provenance_rows)
    assert (
        total_prov == 0
    ), f"Seed should create 0 :__Provenance__ nodes, got {total_prov}: {provenance_rows}"
