"""Tests for ADR-009: __Provenance__ schema is installed alongside __Entity__."""

from __future__ import annotations

from backend.knowledge.storage.graph_store import GraphStore
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection


class TestInitializeSchemaProvenance:
    def test_ensure_schema_installs_provenance_constraint(self) -> None:
        conn = FakeNeo4jConnection()
        store = GraphStore(connection=conn, embedding_generator=FakeEmbeddingGenerator())

        store.initialize_schema()

        issued = [q for q, _ in conn.writes]
        assert any(
            "CONSTRAINT provenance_id_unique" in q and "__Provenance__" in q and "p.id" in q
            for q in issued
        ), f"Expected __Provenance__ uniqueness constraint, got writes: {issued}"

    def test_ensure_schema_installs_provenance_type_index(self) -> None:
        conn = FakeNeo4jConnection()
        store = GraphStore(connection=conn, embedding_generator=FakeEmbeddingGenerator())

        store.initialize_schema()

        issued = [q for q, _ in conn.writes]
        assert any(
            "INDEX provenance_type_idx" in q and "__Provenance__" in q and "p.entity_type" in q
            for q in issued
        ), f"Expected provenance_type_idx, got writes: {issued}"
