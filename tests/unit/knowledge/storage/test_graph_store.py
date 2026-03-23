"""Unit tests for GraphStore.

Uses FakeNeo4jConnection and FakeEmbeddingGenerator via DI --
no patching needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.knowledge.storage.graph_store import GraphStore
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection

# -------------------------------------------------------------------
# Minimal duck-typed stand-ins for LLMGraphTransformer output
# -------------------------------------------------------------------


@dataclass
class _FakeNode:
    id: str
    type: str = "Person"
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class _FakeRelationship:
    source: _FakeNode
    target: _FakeNode
    type: str = "KNOWS"
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class _FakeGraphDocument:
    nodes: list[_FakeNode] = field(default_factory=list)
    relationships: list[_FakeRelationship] = field(default_factory=list)


# -------------------------------------------------------------------
# initialize_schema()
# -------------------------------------------------------------------


class TestInitializeSchema:
    def test_initialize_schema_creates_constraints(self):
        fake_conn = FakeNeo4jConnection()
        store = GraphStore(connection=fake_conn, embedding_generator=FakeEmbeddingGenerator())

        store.initialize_schema()

        constraint_writes = [q for q, _ in fake_conn.writes if "CREATE CONSTRAINT" in q]
        assert len(constraint_writes) >= 1, "Expected at least one CREATE CONSTRAINT write"

    def test_initialize_schema_creates_indexes(self):
        fake_conn = FakeNeo4jConnection()
        store = GraphStore(connection=fake_conn, embedding_generator=FakeEmbeddingGenerator())

        store.initialize_schema()

        index_writes = [q for q, _ in fake_conn.writes if "CREATE INDEX" in q]
        assert len(index_writes) >= 1, "Expected at least one CREATE INDEX write"


# -------------------------------------------------------------------
# store_extracted_entities()
# -------------------------------------------------------------------


class TestStoreExtractedEntities:
    def test_store_entity_forwards_params_to_connection(self):
        fake_conn = FakeNeo4jConnection()
        store = GraphStore(connection=fake_conn, embedding_generator=FakeEmbeddingGenerator())
        doc = _FakeGraphDocument(
            nodes=[_FakeNode(id="Alice", type="Person")],
            relationships=[],
        )

        store.store_extracted_entities(doc, utterance_id="utt-001")

        # Verify the write was executed and params forwarded correctly
        assert len(fake_conn.writes) == 1
        query, params = fake_conn.writes[0]
        assert params["node_id"] == "Alice"
        assert params["entity_type"] == "Person"
        assert params["source_id"] == "utt-001"

    def test_store_entity_generates_embedding(self):
        fake_conn = FakeNeo4jConnection()
        fake_embeddings = FakeEmbeddingGenerator()
        store = GraphStore(connection=fake_conn, embedding_generator=fake_embeddings)
        doc = _FakeGraphDocument(
            nodes=[_FakeNode(id="Python", type="Technology")],
            relationships=[],
        )

        store.store_extracted_entities(doc, utterance_id="utt-002")

        assert len(fake_embeddings.calls) > 0, "Expected embedding generation to be called"


# -------------------------------------------------------------------
# Side-effect boundaries
# -------------------------------------------------------------------


class TestSideEffectBoundaries:
    def test_no_writes_when_no_entities(self):
        fake_conn = FakeNeo4jConnection()
        store = GraphStore(connection=fake_conn, embedding_generator=FakeEmbeddingGenerator())
        doc = _FakeGraphDocument(nodes=[], relationships=[])

        store.store_extracted_entities(doc, utterance_id="utt-003")

        fake_conn.assert_no_writes()
