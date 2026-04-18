"""Tests for ADR-009: __Provenance__ schema is installed alongside __Entity__."""

from __future__ import annotations

from backend.knowledge.storage.graph_store import _USER_FACING_REL_TYPES, GraphStore
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


class TestGraphHopEntityFilter:
    """ADR-009 v1.1: graph-hop expansion must filter all nodes to :__Entity__
    and restrict relationship types to the user-facing allowlist.
    """

    # ---- module-level constant checks ------------------------------------------------

    def test_user_facing_rel_types_excludes_provenance_edges(self) -> None:
        """Provenance audit-trail edge types must NOT appear in the allowlist."""
        provenance_edges = {
            "DERIVED_FROM",
            "EXTRACTED_FROM",
            "LEARNED_FROM",
            "ABOUT",
            "SOURCED_FROM",
            "REFERENCES",
        }
        overlap = provenance_edges & set(_USER_FACING_REL_TYPES)
        assert not overlap, f"Provenance edge types found in _USER_FACING_REL_TYPES: {overlap}"

    def test_user_facing_rel_types_contains_required_types(self) -> None:
        """Core user-facing relationship types must be present."""
        required = {
            "USES",
            "LEARNING",
            "WORKS_ON",
            "WORKS_AT",
            "KNOWS",
            "INTERESTED_IN",
            "IS_A",
            "PART_OF",
            "DEPENDS_ON",
            "RELATED_TO",
            "HAS_TRAIT",
            "HAS_CAPABILITY",
            "HAS_PREFERENCE",
        }
        missing = required - set(_USER_FACING_REL_TYPES)
        assert not missing, f"Missing required types in _USER_FACING_REL_TYPES: {missing}"

    # ---- get_entity_neighborhood (multi-hop expansion) --------------------------------

    def test_multi_hop_expansion_filters_all_path_nodes_to_entity(self) -> None:
        """The expansion Cypher must contain the ALL(nodes) filter so that
        intermediate nodes on variable-length paths cannot be :__Provenance__.
        """
        conn = FakeNeo4jConnection()
        store = GraphStore(connection=conn, embedding_generator=FakeEmbeddingGenerator())

        store.get_entity_neighborhood("User", max_hops=2)

        assert conn.queries, "No queries were executed"
        query, _ = conn.queries[-1]
        assert (
            "ALL(node IN nodes(path) WHERE node:__Entity__)" in query
        ), f"Expected ALL(node IN nodes(path) WHERE node:__Entity__) in query:\n{query}"

    def test_multi_hop_expansion_restricts_rel_types_to_user_facing(self) -> None:
        """The expansion Cypher must pass $allowed_rel_types and the params dict
        must reference the user-facing allowlist when no explicit types are given.
        """
        conn = FakeNeo4jConnection()
        store = GraphStore(connection=conn, embedding_generator=FakeEmbeddingGenerator())

        store.get_entity_neighborhood("User", max_hops=2)

        assert conn.queries, "No queries were executed"
        query, params = conn.queries[-1]
        assert (
            "type(rel) IN $allowed_rel_types" in query
        ), f"Expected type(rel) IN $allowed_rel_types in query:\n{query}"
        assert (
            params is not None and "allowed_rel_types" in params
        ), f"Expected allowed_rel_types in params, got: {params}"
        assert set(params["allowed_rel_types"]) == set(_USER_FACING_REL_TYPES), (
            f"allowed_rel_types in params does not match _USER_FACING_REL_TYPES.\n"
            f"Got: {params['allowed_rel_types']}"
        )

    def test_multi_hop_expansion_caller_subset_overrides_allowlist(self) -> None:
        """When the caller passes an explicit relationship_types subset,
        that subset is used instead of the full module-level allowlist.
        """
        conn = FakeNeo4jConnection()
        store = GraphStore(connection=conn, embedding_generator=FakeEmbeddingGenerator())

        store.get_entity_neighborhood("User", max_hops=1, relationship_types=["KNOWS"])

        _, params = conn.queries[-1]
        assert params is not None
        assert params["allowed_rel_types"] == [
            "KNOWS"
        ], f"Expected caller-provided subset ['KNOWS'] in params, got: {params}"

    # ---- single-hop queries (get_user_relationships_to_entities, get_all_user_relationships) -

    def test_single_hop_get_user_relationships_to_entities_restricts_rel_types(self) -> None:
        """get_user_relationships_to_entities must pass $allowed_rel_types in the
        WHERE clause and in params, defaulting to _USER_FACING_REL_TYPES.
        """
        conn = FakeNeo4jConnection()
        store = GraphStore(connection=conn, embedding_generator=FakeEmbeddingGenerator())

        store.get_user_relationships_to_entities("User", entity_ids=["entity-1"])

        assert conn.queries, "No queries were executed"
        query, params = conn.queries[-1]
        assert (
            "type(r) IN $allowed_rel_types" in query
        ), f"Expected type(r) IN $allowed_rel_types in query:\n{query}"
        assert (
            params is not None and "allowed_rel_types" in params
        ), f"Expected allowed_rel_types in params, got: {params}"
        assert set(params["allowed_rel_types"]) == set(_USER_FACING_REL_TYPES)

    def test_single_hop_get_all_user_relationships_restricts_rel_types(self) -> None:
        """get_all_user_relationships must pass $allowed_rel_types in the WHERE
        clause and in params, defaulting to _USER_FACING_REL_TYPES.
        """
        conn = FakeNeo4jConnection()
        store = GraphStore(connection=conn, embedding_generator=FakeEmbeddingGenerator())

        store.get_all_user_relationships("User")

        assert conn.queries, "No queries were executed"
        query, params = conn.queries[-1]
        assert (
            "type(r) IN $allowed_rel_types" in query
        ), f"Expected type(r) IN $allowed_rel_types in query:\n{query}"
        assert (
            params is not None and "allowed_rel_types" in params
        ), f"Expected allowed_rel_types in params, got: {params}"
        assert set(params["allowed_rel_types"]) == set(_USER_FACING_REL_TYPES)

    def test_single_hop_get_all_user_relationships_caller_subset_overrides(self) -> None:
        """Caller-provided relationship_types for get_all_user_relationships
        must override the module-level default.
        """
        conn = FakeNeo4jConnection()
        store = GraphStore(connection=conn, embedding_generator=FakeEmbeddingGenerator())

        store.get_all_user_relationships("User", relationship_types=["WORKS_ON"])

        _, params = conn.queries[-1]
        assert params is not None
        assert params["allowed_rel_types"] == [
            "WORKS_ON"
        ], f"Expected caller-provided subset ['WORKS_ON'], got: {params}"
