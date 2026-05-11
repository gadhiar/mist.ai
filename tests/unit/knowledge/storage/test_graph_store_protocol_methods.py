"""Phase 5.5 Dispatch 2: Unit tests for new GraphStoreProtocol methods.

Tests for:
- mark_orphaned_by_provenance_path
- current_ontology_version
- get_orphaned_provenance_paths
- upsert_identity
- upsert_user

All tests use FakeNeo4jConnection — no real Neo4j. Async methods are tested
with pytest.mark.asyncio.
"""

from __future__ import annotations

import pytest

from backend.knowledge.curation.bucket1_reader import (
    IdentityPreference,
    ParsedIdentity,
    ParsedUser,
)
from backend.knowledge.storage.graph_store import GraphStore
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(
    *,
    query_results=None,
    query_responses=None,
    write_results=None,
    ontology_version: str | None = None,
) -> GraphStore:
    """Build a GraphStore with a FakeNeo4jConnection."""
    conn = FakeNeo4jConnection(
        query_results=query_results or [],
        query_responses=query_responses or {},
        write_results=write_results or [],
    )
    store = GraphStore(connection=conn, embedding_generator=FakeEmbeddingGenerator())
    if ontology_version is not None:
        store._ontology_version = ontology_version
    return store


def _store_with_conn(conn: FakeNeo4jConnection) -> GraphStore:
    return GraphStore(connection=conn, embedding_generator=FakeEmbeddingGenerator())


# ---------------------------------------------------------------------------
# TestCurrentOntologyVersion
# ---------------------------------------------------------------------------


class TestCurrentOntologyVersion:
    """current_ontology_version() is synchronous and returns the version string."""

    def test_returns_string(self):
        store = _make_store()

        result = store.current_ontology_version()

        assert isinstance(result, str)

    def test_returns_semver_shaped_string(self):
        store = _make_store()

        result = store.current_ontology_version()

        parts = result.split(".")
        assert len(parts) == 3, f"Expected X.Y.Z format, got: {result}"

    def test_custom_version_is_returned(self):
        store = _make_store(ontology_version="2.3.4")

        result = store.current_ontology_version()

        assert result == "2.3.4"

    def test_default_version_matches_config(self):
        from backend.knowledge.storage.graph_store import _DEFAULT_ONTOLOGY_VERSION

        store = _make_store()

        result = store.current_ontology_version()

        assert result == _DEFAULT_ONTOLOGY_VERSION


# ---------------------------------------------------------------------------
# TestMarkOrphanedByProvenancePath
# ---------------------------------------------------------------------------


class TestMarkOrphanedByProvenancePath:
    """mark_orphaned_by_provenance_path issues a DERIVED_FROM Cypher SET."""

    @pytest.mark.asyncio
    async def test_issues_write_query_with_path_param(self):
        conn = FakeNeo4jConnection(write_results=[{"marked_count": 3}])
        store = _store_with_conn(conn)

        await store.mark_orphaned_by_provenance_path("/vault/sessions/2026-05-01.md")

        assert conn.writes, "Expected at least one write query"
        write_query, params = conn.writes[-1]
        assert "DERIVED_FROM" in write_query or "orphaned" in write_query.lower()
        assert params is not None
        assert "/vault/sessions/2026-05-01.md" in str(params)

    @pytest.mark.asyncio
    async def test_returns_integer_count(self):
        conn = FakeNeo4jConnection(write_results=[{"marked_count": 5}])
        store = _store_with_conn(conn)

        result = await store.mark_orphaned_by_provenance_path("/vault/users/raj.md")

        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_triples_match(self):
        conn = FakeNeo4jConnection(write_results=[])
        store = _store_with_conn(conn)

        result = await store.mark_orphaned_by_provenance_path("/vault/nonexistent.md")

        assert result == 0

    @pytest.mark.asyncio
    async def test_sets_orphaned_status_in_cypher(self):
        conn = FakeNeo4jConnection(write_results=[])
        store = _store_with_conn(conn)

        await store.mark_orphaned_by_provenance_path("/vault/identity/mist.md")

        conn.assert_write_executed("orphaned")

    @pytest.mark.asyncio
    async def test_passes_path_as_query_param_not_interpolated(self):
        """Path must arrive as a query param, not interpolated into the Cypher string."""
        conn = FakeNeo4jConnection(write_results=[])
        store = _store_with_conn(conn)
        target_path = "/vault/sessions/2026-05-01-session.md"

        await store.mark_orphaned_by_provenance_path(target_path)

        _, params = conn.writes[-1]
        assert params is not None, "Params dict must not be None"
        assert target_path in str(params.values()), "Path must appear in params values"


# ---------------------------------------------------------------------------
# TestGetOrphanedProvenancePaths
# ---------------------------------------------------------------------------


class TestGetOrphanedProvenancePaths:
    """get_orphaned_provenance_paths returns distinct orphaned path strings."""

    @pytest.mark.asyncio
    async def test_returns_list(self):
        conn = FakeNeo4jConnection(query_results=[])
        store = _store_with_conn(conn)

        result = await store.get_orphaned_provenance_paths()

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_orphans(self):
        conn = FakeNeo4jConnection(query_results=[])
        store = _store_with_conn(conn)

        result = await store.get_orphaned_provenance_paths()

        assert result == []

    @pytest.mark.asyncio
    async def test_returns_paths_from_query_result(self):
        rows = [{"path": "/vault/sessions/2026-05-01.md"}, {"path": "/vault/users/raj.md"}]
        conn = FakeNeo4jConnection(
            query_responses={"orphaned": rows},
        )
        store = _store_with_conn(conn)

        result = await store.get_orphaned_provenance_paths()

        assert "/vault/sessions/2026-05-01.md" in result
        assert "/vault/users/raj.md" in result

    @pytest.mark.asyncio
    async def test_queries_for_orphaned_status(self):
        conn = FakeNeo4jConnection(query_results=[])
        store = _store_with_conn(conn)

        await store.get_orphaned_provenance_paths()

        assert conn.queries, "Expected at least one query"
        any_orphaned_query = any("orphaned" in q.lower() for q, _ in conn.queries)
        assert any_orphaned_query, "Expected a query filtering on 'orphaned' status"


# ---------------------------------------------------------------------------
# TestUpsertIdentity
# ---------------------------------------------------------------------------


class TestUpsertIdentity:
    """upsert_identity writes HAS_TRAIT/HAS_CAPABILITY/HAS_PREFERENCE edges idempotently."""

    def _make_identity(
        self,
        traits: list[str] | None = None,
        capabilities: list[str] | None = None,
        preferences: list[IdentityPreference] | None = None,
    ) -> ParsedIdentity:
        return ParsedIdentity(
            traits=traits or [],
            capabilities=capabilities or [],
            preferences=preferences or [],
        )

    @pytest.mark.asyncio
    async def test_returns_integer(self):
        conn = FakeNeo4jConnection()
        store = _store_with_conn(conn)
        parsed = self._make_identity(traits=["warm"])

        result = await store.upsert_identity(parsed, derived_from_path="/vault/identity/mist.md")

        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_issues_write_for_each_trait(self):
        conn = FakeNeo4jConnection()
        store = _store_with_conn(conn)
        parsed = self._make_identity(traits=["warm", "curious"])

        await store.upsert_identity(parsed, derived_from_path="/vault/identity/mist.md")

        has_trait_writes = [
            (q, p)
            for q, p in conn.writes
            if "HAS_TRAIT" in q or ("MistTrait" in q and "warm" in str(p))
        ]
        assert has_trait_writes, "Expected writes for HAS_TRAIT edges"

    @pytest.mark.asyncio
    async def test_issues_write_for_each_capability(self):
        conn = FakeNeo4jConnection()
        store = _store_with_conn(conn)
        parsed = self._make_identity(capabilities=["tool-use", "code-generation"])

        await store.upsert_identity(parsed, derived_from_path="/vault/identity/mist.md")

        cap_writes = [q for q, _ in conn.writes if "HAS_CAPABILITY" in q]
        assert cap_writes, "Expected writes for HAS_CAPABILITY edges"

    @pytest.mark.asyncio
    async def test_issues_write_for_each_preference(self):
        conn = FakeNeo4jConnection()
        store = _store_with_conn(conn)
        prefs = [
            IdentityPreference(slug="no-emoji", enforcement="absolute"),
            IdentityPreference(slug="no-slop", enforcement="absolute"),
        ]
        parsed = self._make_identity(preferences=prefs)

        await store.upsert_identity(parsed, derived_from_path="/vault/identity/mist.md")

        pref_writes = [q for q, _ in conn.writes if "HAS_PREFERENCE" in q]
        assert pref_writes, "Expected writes for HAS_PREFERENCE edges"

    @pytest.mark.asyncio
    async def test_uses_merge_not_create_for_idempotency(self):
        """Cypher must use MERGE, not CREATE, to avoid duplicates."""
        conn = FakeNeo4jConnection()
        store = _store_with_conn(conn)
        parsed = self._make_identity(traits=["warm"])

        await store.upsert_identity(parsed, derived_from_path="/vault/identity/mist.md")

        merge_writes = [q for q, _ in conn.writes if "MERGE" in q.upper()]
        assert merge_writes, "Expected MERGE writes for idempotency"

    @pytest.mark.asyncio
    async def test_passes_derived_from_path_in_params(self):
        conn = FakeNeo4jConnection()
        store = _store_with_conn(conn)
        path = "/vault/identity/mist.md"
        parsed = self._make_identity(traits=["warm"])

        await store.upsert_identity(parsed, derived_from_path=path)

        path_in_params = any(
            params is not None and path in str(params.values()) for _, params in conn.writes
        )
        assert path_in_params, "Expected derived_from_path to appear in write params"

    @pytest.mark.asyncio
    async def test_empty_parsed_identity_returns_zero(self):
        conn = FakeNeo4jConnection()
        store = _store_with_conn(conn)
        parsed = self._make_identity()

        result = await store.upsert_identity(parsed, derived_from_path="/vault/identity/mist.md")

        assert result == 0


# ---------------------------------------------------------------------------
# TestUpsertUser
# ---------------------------------------------------------------------------


class TestUpsertUser:
    """upsert_user writes per-section typed edges from User node idempotently."""

    def _make_user(
        self,
        user_id: str = "user-raj",
        tools: list[str] | None = None,
        expertise: list[str] | None = None,
        learning: list[str] | None = None,
        projects: list[str] | None = None,
        affiliations: list[str] | None = None,
        interests: list[str] | None = None,
        goals: list[str] | None = None,
        preferences: list[str] | None = None,
        people: list[str] | None = None,
    ) -> ParsedUser:
        return ParsedUser(
            user_id=user_id,
            tools_and_technologies=tools or [],
            expertise=expertise or [],
            currently_learning=learning or [],
            projects=projects or [],
            affiliations=affiliations or [],
            interests=interests or [],
            goals=goals or [],
            preferences=preferences or [],
            people=people or [],
        )

    @pytest.mark.asyncio
    async def test_returns_integer(self):
        conn = FakeNeo4jConnection()
        store = _store_with_conn(conn)
        parsed = self._make_user(tools=["Python"])

        result = await store.upsert_user(parsed, derived_from_path="/vault/users/raj.md")

        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_writes_uses_edges_for_tools(self):
        conn = FakeNeo4jConnection()
        store = _store_with_conn(conn)
        parsed = self._make_user(tools=["Python", "Neo4j"])

        await store.upsert_user(parsed, derived_from_path="/vault/users/raj.md")

        uses_writes = [q for q, _ in conn.writes if "USES" in q]
        assert uses_writes, "Expected USES edge writes for tools_and_technologies"

    @pytest.mark.asyncio
    async def test_writes_learning_edges(self):
        conn = FakeNeo4jConnection()
        store = _store_with_conn(conn)
        parsed = self._make_user(learning=["Cypher", "RAG"])

        await store.upsert_user(parsed, derived_from_path="/vault/users/raj.md")

        learning_writes = [q for q, _ in conn.writes if "LEARNING" in q]
        assert learning_writes, "Expected LEARNING edge writes"

    @pytest.mark.asyncio
    async def test_writes_expert_in_edges(self):
        conn = FakeNeo4jConnection()
        store = _store_with_conn(conn)
        parsed = self._make_user(expertise=["FastAPI", "Python"])

        await store.upsert_user(parsed, derived_from_path="/vault/users/raj.md")

        exp_writes = [q for q, _ in conn.writes if "EXPERT_IN" in q]
        assert exp_writes, "Expected EXPERT_IN edge writes"

    @pytest.mark.asyncio
    async def test_uses_merge_for_idempotency(self):
        conn = FakeNeo4jConnection()
        store = _store_with_conn(conn)
        parsed = self._make_user(tools=["Python"])

        await store.upsert_user(parsed, derived_from_path="/vault/users/raj.md")

        merge_writes = [q for q, _ in conn.writes if "MERGE" in q.upper()]
        assert merge_writes, "Expected MERGE writes for idempotency"

    @pytest.mark.asyncio
    async def test_passes_derived_from_path_in_params(self):
        conn = FakeNeo4jConnection()
        store = _store_with_conn(conn)
        path = "/vault/users/raj.md"
        parsed = self._make_user(tools=["Python"])

        await store.upsert_user(parsed, derived_from_path=path)

        path_in_params = any(
            params is not None and path in str(params.values()) for _, params in conn.writes
        )
        assert path_in_params, "Expected derived_from_path to appear in write params"

    @pytest.mark.asyncio
    async def test_empty_user_returns_zero(self):
        conn = FakeNeo4jConnection()
        store = _store_with_conn(conn)
        parsed = self._make_user()

        result = await store.upsert_user(parsed, derived_from_path="/vault/users/raj.md")

        assert result == 0

    @pytest.mark.asyncio
    async def test_works_on_edges_written_for_projects(self):
        conn = FakeNeo4jConnection()
        store = _store_with_conn(conn)
        parsed = self._make_user(projects=["MIST.AI", "Hana"])

        await store.upsert_user(parsed, derived_from_path="/vault/users/raj.md")

        works_on_writes = [q for q, _ in conn.writes if "WORKS_ON" in q]
        assert works_on_writes, "Expected WORKS_ON edge writes"

    @pytest.mark.asyncio
    async def test_knows_person_edges_written_for_people(self):
        conn = FakeNeo4jConnection()
        store = _store_with_conn(conn)
        parsed = self._make_user(people=["Alice", "Bob"])

        await store.upsert_user(parsed, derived_from_path="/vault/users/raj.md")

        knows_writes = [q for q, _ in conn.writes if "KNOWS_PERSON" in q]
        assert knows_writes, "Expected KNOWS_PERSON edge writes"
