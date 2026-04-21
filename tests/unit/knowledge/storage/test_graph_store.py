"""Unit tests for GraphStore.

Uses FakeNeo4jConnection and FakeEmbeddingGenerator via DI --
no patching needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.errors import Neo4jQueryError
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


# -------------------------------------------------------------------
# vector_indexes_available flag and initialize_schema() tracking
# -------------------------------------------------------------------


class TestVectorIndexAvailability:
    def test_vector_indexes_unavailable_before_initialize_schema(self):
        # Arrange
        store = GraphStore(
            connection=FakeNeo4jConnection(),
            embedding_generator=FakeEmbeddingGenerator(),
        )

        # Assert -- default state is False until initialize_schema() runs
        assert store.vector_indexes_available is False

    def test_vector_indexes_available_after_successful_initialize_schema(self):
        # Arrange
        fake_conn = FakeNeo4jConnection()
        store = GraphStore(connection=fake_conn, embedding_generator=FakeEmbeddingGenerator())

        # Act
        store.initialize_schema()

        # Assert -- all vector writes succeeded, so flag is True
        assert store.vector_indexes_available is True

    def test_vector_indexes_unavailable_when_all_vector_index_writes_fail(self):
        # Arrange
        fake_conn = FakeNeo4jConnection(
            write_errors={
                "CREATE VECTOR INDEX": Neo4jQueryError("vector index not supported"),
            }
        )
        store = GraphStore(connection=fake_conn, embedding_generator=FakeEmbeddingGenerator())

        # Act
        store.initialize_schema()

        # Assert -- all vector index writes failed, flag remains False
        assert store.vector_indexes_available is False

    def test_vector_indexes_unavailable_when_partial_vector_index_writes_fail(self):
        # Arrange: fail only one of the two vector index writes
        call_count = {"n": 0}
        _ = FakeNeo4jConnection.execute_write

        class _PartialFailConnection(FakeNeo4jConnection):
            def execute_write(self, query, params=None):
                if "CREATE VECTOR INDEX" in query:
                    call_count["n"] += 1
                    if call_count["n"] == 1:
                        # Record the call before raising so write log is accurate
                        self.writes.append((query, params))
                        raise Neo4jQueryError("first vector index failed")
                return super().execute_write(query, params)

        fake_conn = _PartialFailConnection()
        store = GraphStore(connection=fake_conn, embedding_generator=FakeEmbeddingGenerator())

        # Act
        store.initialize_schema()

        # Assert -- only 1 of 2 succeeded, so flag is False
        assert store.vector_indexes_available is False

    def test_property_exposes_internal_flag(self):
        # Arrange
        store = GraphStore(
            connection=FakeNeo4jConnection(),
            embedding_generator=FakeEmbeddingGenerator(),
        )

        # Assert -- property mirrors private attribute
        store._vector_indexes_available = True
        assert store.vector_indexes_available is True

        store._vector_indexes_available = False
        assert store.vector_indexes_available is False


# -------------------------------------------------------------------
# search_similar_entities() -- graceful fallback
# -------------------------------------------------------------------


class TestSearchSimilarEntitiesGracefulFallback:
    def test_returns_empty_list_when_vector_indexes_unavailable(self):
        # Arrange
        store = GraphStore(
            connection=FakeNeo4jConnection(),
            embedding_generator=FakeEmbeddingGenerator(),
        )
        # Fake returns [] for SHOW INDEXES probe -> lazy flag resolves False

        # Act
        results = store.search_similar_entities("Python programming")

        # Assert
        assert results == []

    def test_does_not_issue_search_query_when_vector_indexes_unavailable(self):
        # Arrange
        fake_conn = FakeNeo4jConnection()
        store = GraphStore(connection=fake_conn, embedding_generator=FakeEmbeddingGenerator())

        # Act
        store.search_similar_entities("Python programming")

        # Assert -- the SHOW INDEXES probe ran, but the vector search did not
        assert not any("db.index.vector.queryNodes" in q for q, _ in fake_conn.queries)

    def test_returns_results_when_vector_indexes_available(self):
        # Arrange
        record = {
            "entity_id": "Alice",
            "entity_type": "Person",
            "similarity": 0.9,
            "properties": {},
        }

        class _RecordLike:
            def __iter__(self):
                return iter(record.items())

            def keys(self):
                return record.keys()

            def __getitem__(self, key):
                return record[key]

        fake_conn = FakeNeo4jConnection(
            query_responses={"db.index.vector.queryNodes": [_RecordLike()]}
        )
        store = GraphStore(connection=fake_conn, embedding_generator=FakeEmbeddingGenerator())
        store._vector_indexes_available = True

        # Act
        results = store.search_similar_entities("Alice", limit=5)

        # Assert
        assert len(results) == 1

    def test_disables_flag_and_returns_empty_on_runtime_query_failure(self):
        # Arrange
        fake_conn = FakeNeo4jConnection(
            query_errors={
                "db.index.vector.queryNodes": Neo4jQueryError("index not found"),
            }
        )
        store = GraphStore(connection=fake_conn, embedding_generator=FakeEmbeddingGenerator())
        store._vector_indexes_available = True

        # Act
        results = store.search_similar_entities("test query")

        # Assert -- returns empty and disables the flag
        assert results == []
        assert store.vector_indexes_available is False


# -------------------------------------------------------------------
# search_document_chunks() -- graceful fallback
# -------------------------------------------------------------------


class TestSearchDocumentChunksGracefulFallback:
    def test_returns_empty_list_when_vector_indexes_unavailable(self):
        # Arrange
        store = GraphStore(
            connection=FakeNeo4jConnection(),
            embedding_generator=FakeEmbeddingGenerator(),
        )
        # Fake returns [] for SHOW INDEXES probe -> lazy flag resolves False

        # Act
        results = store.search_document_chunks("knowledge graph")

        # Assert
        assert results == []

    def test_does_not_issue_search_query_when_vector_indexes_unavailable(self):
        # Arrange
        fake_conn = FakeNeo4jConnection()
        store = GraphStore(connection=fake_conn, embedding_generator=FakeEmbeddingGenerator())

        # Act
        store.search_document_chunks("knowledge graph")

        # Assert -- the SHOW INDEXES probe ran, but the chunk search did not
        assert not any("db.index.vector.queryNodes" in q for q, _ in fake_conn.queries)

    def test_disables_flag_and_returns_empty_on_runtime_query_failure(self):
        # Arrange
        fake_conn = FakeNeo4jConnection(
            query_errors={
                "db.index.vector.queryNodes": Neo4jQueryError("chunk index not found"),
            }
        )
        store = GraphStore(connection=fake_conn, embedding_generator=FakeEmbeddingGenerator())
        store._vector_indexes_available = True

        # Act
        results = store.search_document_chunks("test query")

        # Assert -- returns empty and disables the flag
        assert results == []
        assert store.vector_indexes_available is False


# -------------------------------------------------------------------
# get_mist_identity_context()
# -------------------------------------------------------------------


class TestGetMistIdentityContext:
    """Cluster 3: GraphStore.get_mist_identity_context fetches identity + edges."""

    def test_returns_identity_traits_capabilities_preferences(self):
        """Happy path: identity node plus all edge types present."""
        identity_row = {
            "id": "mist-identity",
            "display_name": "MIST",
            "pronouns": "she/her",
            "self_concept": "A cognitive architecture.",
        }
        trait_rows = [
            {
                "id": "trait-warm",
                "display_name": "Warm",
                "axis": "Persona",
                "description": "Friendly.",
            },
            {
                "id": "trait-technical",
                "display_name": "Technical",
                "axis": "Persona",
                "description": "Precise.",
            },
        ]
        cap_rows = [
            {"id": "cap-tool-use", "display_name": "Tool use", "description": "MCP tools."},
        ]
        pref_rows = [
            {
                "id": "pref-no-emoji",
                "display_name": "No emoji",
                "enforcement": "absolute",
                "context": "Hard rule.",
            },
            {
                "id": "pref-no-ai-slop",
                "display_name": "No slop",
                "enforcement": "absolute",
                "context": "Hard rule.",
            },
        ]

        # query_responses uses substring matching in insertion order: more specific
        # edge-type patterns must come before the generic identity pattern because
        # HAS_TRAIT/HAS_CAPABILITY/HAS_PREFERENCE queries also contain 'mist-identity'.
        fake_conn = FakeNeo4jConnection(
            query_responses={
                "HAS_TRAIT": trait_rows,
                "HAS_CAPABILITY": cap_rows,
                "HAS_PREFERENCE": pref_rows,
                "mist-identity'})": [identity_row],
            }
        )
        store = GraphStore(connection=fake_conn, embedding_generator=FakeEmbeddingGenerator())
        result = store.get_mist_identity_context()

        assert result["identity"]["display_name"] == "MIST"
        assert result["identity"]["pronouns"] == "she/her"
        assert len(result["traits"]) == 2
        assert len(result["capabilities"]) == 1
        assert len(result["preferences"]) == 2
        pref_ids = {p["id"] for p in result["preferences"]}
        assert pref_ids == {"pref-no-emoji", "pref-no-ai-slop"}

    def test_returns_default_identity_when_missing(self):
        """If mist-identity node is absent, return a minimal safe default."""
        # All queries return empty lists — node not in graph yet (pre-seed state).
        fake_conn = FakeNeo4jConnection(query_results=[])
        store = GraphStore(connection=fake_conn, embedding_generator=FakeEmbeddingGenerator())
        result = store.get_mist_identity_context()

        assert result["identity"]["display_name"] == "MIST"
        assert result["identity"]["pronouns"] == "she/her"
        assert result["traits"] == []
        assert result["capabilities"] == []
        assert result["preferences"] == []
