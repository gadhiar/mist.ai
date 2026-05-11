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


# ---------------------------------------------------------------------------
# TestDerivedFromProvenanceEdges
# Phase 5.5 Bucket 1 fix: upsert_identity and upsert_user must write
# DERIVED_FROM edges from each typed entity to the VaultNote so that
# mark_orphaned_by_provenance_path finds them.
# ---------------------------------------------------------------------------


class TestUpsertIdentityWritesDerivedFromProvenance:
    """upsert_identity must MERGE a DERIVED_FROM edge for each typed entity."""

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
    async def test_upsert_identity_writes_derived_from_edge_for_each_trait(self):
        """Each HAS_TRAIT write must be accompanied by a DERIVED_FROM edge write."""
        conn = FakeNeo4jConnection()
        store = _store_with_conn(conn)
        parsed = self._make_identity(traits=["warm", "curious"])

        await store.upsert_identity(parsed, derived_from_path="/vault/identity/mist.md")

        derived_from_writes = [q for q, _ in conn.writes if "DERIVED_FROM" in q]
        assert len(derived_from_writes) >= 2, (
            f"Expected at least 2 DERIVED_FROM writes (one per trait); "
            f"got {len(derived_from_writes)}: {derived_from_writes}"
        )

    @pytest.mark.asyncio
    async def test_upsert_identity_writes_derived_from_edge_for_each_capability(self):
        """Each HAS_CAPABILITY write must be accompanied by a DERIVED_FROM edge write."""
        conn = FakeNeo4jConnection()
        store = _store_with_conn(conn)
        parsed = self._make_identity(capabilities=["tool-use", "code-generation"])

        await store.upsert_identity(parsed, derived_from_path="/vault/identity/mist.md")

        derived_from_writes = [q for q, _ in conn.writes if "DERIVED_FROM" in q]
        assert len(derived_from_writes) >= 2, (
            f"Expected at least 2 DERIVED_FROM writes (one per capability); "
            f"got {len(derived_from_writes)}"
        )

    @pytest.mark.asyncio
    async def test_upsert_identity_writes_derived_from_edge_for_each_preference(self):
        """Each HAS_PREFERENCE write must be accompanied by a DERIVED_FROM edge write."""
        conn = FakeNeo4jConnection()
        store = _store_with_conn(conn)
        prefs = [
            IdentityPreference(slug="no-emoji", enforcement="absolute"),
            IdentityPreference(slug="no-slop", enforcement="absolute"),
        ]
        parsed = self._make_identity(preferences=prefs)

        await store.upsert_identity(parsed, derived_from_path="/vault/identity/mist.md")

        derived_from_writes = [q for q, _ in conn.writes if "DERIVED_FROM" in q]
        assert len(derived_from_writes) >= 2, (
            f"Expected at least 2 DERIVED_FROM writes (one per preference); "
            f"got {len(derived_from_writes)}"
        )

    @pytest.mark.asyncio
    async def test_upsert_identity_derived_from_targets_vault_note(self):
        """DERIVED_FROM edges must target the VaultNote node, not an entity."""
        conn = FakeNeo4jConnection()
        store = _store_with_conn(conn)
        parsed = self._make_identity(traits=["warm"])

        await store.upsert_identity(parsed, derived_from_path="/vault/identity/mist.md")

        derived_writes = [q for q, _ in conn.writes if "DERIVED_FROM" in q]
        assert derived_writes, "Expected at least one DERIVED_FROM write"
        # The write must reference VaultNote (the provenance node type)
        vault_note_in_query = any("VaultNote" in q for q in derived_writes)
        assert vault_note_in_query, (
            "DERIVED_FROM write must reference VaultNote node; " f"got queries: {derived_writes}"
        )


class TestUpsertUserWritesDerivedFromProvenance:
    """upsert_user must MERGE a DERIVED_FROM edge for each typed entity."""

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
    async def test_upsert_user_writes_derived_from_edge_for_each_typed_edge(self):
        """Each USES/EXPERT_IN/etc write must be accompanied by a DERIVED_FROM edge."""
        conn = FakeNeo4jConnection()
        store = _store_with_conn(conn)
        parsed = self._make_user(tools=["Python", "Neo4j"], expertise=["FastAPI"])

        await store.upsert_user(parsed, derived_from_path="/vault/users/raj.md")

        derived_from_writes = [q for q, _ in conn.writes if "DERIVED_FROM" in q]
        # 3 typed edges (USES x2, EXPERT_IN x1) -> 3 DERIVED_FROM writes
        assert len(derived_from_writes) >= 3, (
            f"Expected at least 3 DERIVED_FROM writes (one per typed edge); "
            f"got {len(derived_from_writes)}"
        )

    @pytest.mark.asyncio
    async def test_upsert_user_derived_from_targets_vault_note(self):
        """DERIVED_FROM edges must target the VaultNote node."""
        conn = FakeNeo4jConnection()
        store = _store_with_conn(conn)
        parsed = self._make_user(tools=["Python"])

        await store.upsert_user(parsed, derived_from_path="/vault/users/raj.md")

        derived_writes = [q for q, _ in conn.writes if "DERIVED_FROM" in q]
        assert derived_writes, "Expected at least one DERIVED_FROM write"
        vault_note_in_query = any("VaultNote" in q for q in derived_writes)
        assert vault_note_in_query, (
            "DERIVED_FROM write must reference VaultNote node; " f"got queries: {derived_writes}"
        )

    @pytest.mark.asyncio
    async def test_upsert_user_derived_from_uses_merge_for_idempotency(self):
        """DERIVED_FROM edge writes must use MERGE, not CREATE."""
        conn = FakeNeo4jConnection()
        store = _store_with_conn(conn)
        parsed = self._make_user(tools=["Python"])

        await store.upsert_user(parsed, derived_from_path="/vault/users/raj.md")

        derived_writes = [q for q, _ in conn.writes if "DERIVED_FROM" in q]
        assert derived_writes, "Expected DERIVED_FROM write"
        all_use_merge = all("MERGE" in q for q in derived_writes)
        assert all_use_merge, (
            "DERIVED_FROM edge writes must use MERGE for idempotency; "
            f"non-MERGE queries found: {[q for q in derived_writes if 'MERGE' not in q]}"
        )


class TestMarkOrphanedAfterUpsert:
    """mark_orphaned_by_provenance_path finds Bucket 1 triples after the upsert fix."""

    def _make_identity(self, traits: list[str]) -> ParsedIdentity:
        return ParsedIdentity(traits=traits, capabilities=[], preferences=[])

    def _make_user(self, tools: list[str]) -> ParsedUser:
        return ParsedUser(
            user_id="user-raj",
            tools_and_technologies=tools,
            expertise=[],
            currently_learning=[],
            projects=[],
            affiliations=[],
            interests=[],
            goals=[],
            preferences=[],
            people=[],
        )

    @pytest.mark.asyncio
    async def test_mark_orphaned_after_upsert_identity_finds_the_triples(self):
        """After upsert_identity, mark_orphaned_by_provenance_path marks >0 triples.

        Uses FakeGraphStore (not real GraphStore) so the test runs on the host
        without Neo4j. Exercises the FakeGraphStore's own DERIVED_FROM tracking.
        """
        from tests.fakes.graph_store import FakeGraphStore

        store = FakeGraphStore()
        path = "/vault/identity/mist.md"
        parsed = self._make_identity(traits=["warm", "curious"])

        await store.upsert_identity(parsed, derived_from_path=path)
        marked = await store.mark_orphaned_by_provenance_path(path)

        assert marked > 0, (
            f"mark_orphaned_by_provenance_path must find >0 triples after "
            f"upsert_identity; got {marked}. "
            f"FakeGraphStore likely does not track DERIVED_FROM provenance."
        )

    @pytest.mark.asyncio
    async def test_mark_orphaned_after_upsert_user_finds_the_triples(self):
        """After upsert_user, mark_orphaned_by_provenance_path marks >0 triples.

        Uses FakeGraphStore (not real GraphStore) so the test runs on the host
        without Neo4j. Exercises the FakeGraphStore's own DERIVED_FROM tracking.
        """
        from tests.fakes.graph_store import FakeGraphStore

        store = FakeGraphStore()
        path = "/vault/users/raj.md"
        parsed = self._make_user(tools=["Python", "Neo4j"])

        await store.upsert_user(parsed, derived_from_path=path)
        marked = await store.mark_orphaned_by_provenance_path(path)

        assert marked > 0, (
            f"mark_orphaned_by_provenance_path must find >0 triples after "
            f"upsert_user; got {marked}. "
            f"FakeGraphStore likely does not track DERIVED_FROM provenance."
        )

    @pytest.mark.asyncio
    async def test_mark_orphaned_after_upsert_identity_marks_correct_count(self):
        """mark_orphaned returns count equal to number of typed entities upserted."""
        from tests.fakes.graph_store import FakeGraphStore

        store = FakeGraphStore()
        path = "/vault/identity/mist.md"
        parsed = self._make_identity(traits=["warm", "curious", "direct"])

        await store.upsert_identity(parsed, derived_from_path=path)
        marked = await store.mark_orphaned_by_provenance_path(path)

        # 3 traits -> 3 DERIVED_FROM edges -> 3 marked
        assert marked == 3, f"Expected 3 triples marked (one per trait); got {marked}"

    @pytest.mark.asyncio
    async def test_mark_orphaned_does_not_affect_different_path(self):
        """mark_orphaned on path A must not affect triples derived from path B."""
        from tests.fakes.graph_store import FakeGraphStore

        store = FakeGraphStore()
        path_a = "/vault/users/raj.md"
        path_b = "/vault/users/alice.md"
        parsed_a = self._make_user(tools=["Python"])
        parsed_b = self._make_user(tools=["Rust"])

        await store.upsert_user(parsed_a, derived_from_path=path_a)
        await store.upsert_user(parsed_b, derived_from_path=path_b)

        marked = await store.mark_orphaned_by_provenance_path(path_a)

        assert marked == 1, f"Expected 1 triple marked (path_a only); got {marked}"
        # path_b triple must still be active
        triple_b = store.get_triple("user-raj", "USES", "Rust")
        assert (
            triple_b is not None and triple_b.status == "active"
        ), "Triple from path_b must remain active after marking path_a orphaned"


# ---------------------------------------------------------------------------
# TestMarkOrphanedTypedEdges
# Phase 5.5 tie-up (P3 #1): mark_orphaned_by_provenance_path must ALSO mark
# the typed edges (r.derived_from_path == path) so retrieval can filter on
# the edge's own status field without a JOIN through DERIVED_FROM.
# ---------------------------------------------------------------------------


class TestMarkOrphanedTypedEdges:
    """mark_orphaned_by_provenance_path must mark typed edges by derived_from_path.

    The real GraphStore issues a second Cypher statement that sets
    r.status='orphaned' on all relationships where r.derived_from_path==path.
    The FakeGraphStore propagates the same update to FakeTriple.status.

    Tests in this class exercise both the real GraphStore (via FakeNeo4jConnection
    write inspection) and the FakeGraphStore (via triple-level state assertions).
    """

    @pytest.mark.asyncio
    async def test_mark_orphaned_issues_typed_edge_write(self):
        """mark_orphaned_by_provenance_path must issue a write that updates typed edges.

        The second write (beyond the existing DERIVED_FROM write) must reference
        derived_from_path so it can select typed edges whose r.derived_from_path
        matches the path being orphaned.
        """
        conn = FakeNeo4jConnection(write_results=[{"marked_count": 1}])
        store = _store_with_conn(conn)

        await store.mark_orphaned_by_provenance_path("/app/mist-memory/users/raj.md")

        # Must have at least two writes: one for DERIVED_FROM edges, one for typed edges
        assert len(conn.writes) >= 2, (
            f"Expected at least 2 writes (DERIVED_FROM + typed edges); got {len(conn.writes)}: "
            f"{[q[:60] for q, _ in conn.writes]}"
        )

    @pytest.mark.asyncio
    async def test_mark_orphaned_typed_edge_write_references_derived_from_path(self):
        """The typed-edge write must match on r.derived_from_path = $path."""
        conn = FakeNeo4jConnection(write_results=[{"marked_count": 1}])
        store = _store_with_conn(conn)
        target_path = "/app/mist-memory/users/raj.md"

        await store.mark_orphaned_by_provenance_path(target_path)

        # At least one write must reference derived_from_path as a selector
        writes_with_derived_from_path = [
            (q, p)
            for q, p in conn.writes
            if "derived_from_path" in q and p is not None and target_path in str(p.values())
        ]
        assert writes_with_derived_from_path, (
            "Expected a write that selects edges by derived_from_path; "
            f"writes were: {[(q[:80], p) for q, p in conn.writes]}"
        )

    @pytest.mark.asyncio
    async def test_mark_orphaned_typed_edge_write_sets_status_orphaned(self):
        """The typed-edge write must SET r.status='orphaned'."""
        conn = FakeNeo4jConnection(write_results=[{"marked_count": 1}])
        store = _store_with_conn(conn)

        await store.mark_orphaned_by_provenance_path("/app/mist-memory/users/raj.md")

        # At least one write beyond the DERIVED_FROM write must set 'orphaned'
        orphaned_writes = [q for q, _ in conn.writes if "orphaned" in q.lower()]
        assert len(orphaned_writes) >= 2, (
            f"Expected at least 2 writes setting 'orphaned' (one for DERIVED_FROM, "
            f"one for typed edges); got {len(orphaned_writes)}: {orphaned_writes}"
        )

    @pytest.mark.asyncio
    async def test_fake_graph_store_mark_orphaned_marks_typed_triples(self):
        """FakeGraphStore.mark_orphaned must mark FakeTriple.status='orphaned'.

        This is the core assertion for the retrieval filter: after marking,
        any FakeTriple with derived_from_path==path must have status='orphaned'
        so that retrieval queries filtering on r.status can exclude them.
        """
        from tests.fakes.graph_store import FakeGraphStore

        store = FakeGraphStore()
        path = "/app/mist-memory/users/raj.md"
        store.add_triple(subject="User", predicate="USES", object="Python", derived_from_path=path)

        await store.mark_orphaned_by_provenance_path(path)

        triple = store.get_triple("User", "USES", "Python")
        assert triple is not None
        assert (
            triple.status == "orphaned"
        ), f"FakeTriple.status must be 'orphaned' after mark_orphaned; got {triple.status!r}"

    @pytest.mark.asyncio
    async def test_fake_graph_store_mark_orphaned_leaves_other_path_triples_active(self):
        """mark_orphaned on path A must leave triples from path B active."""
        from tests.fakes.graph_store import FakeGraphStore

        store = FakeGraphStore()
        path_a = "/app/mist-memory/users/raj.md"
        path_b = "/app/mist-memory/users/alice.md"
        store.add_triple(
            subject="User", predicate="USES", object="Python", derived_from_path=path_a
        )
        store.add_triple(subject="User", predicate="USES", object="Rust", derived_from_path=path_b)

        await store.mark_orphaned_by_provenance_path(path_a)

        triple_b = store.get_triple("User", "USES", "Rust")
        assert (
            triple_b is not None and triple_b.status == "active"
        ), f"Triple from path_b must remain active; got {triple_b.status!r}"

    @pytest.mark.asyncio
    async def test_mark_orphaned_idempotent_on_typed_edges(self):
        """Calling mark_orphaned twice on the same path must not double-count."""
        from tests.fakes.graph_store import FakeGraphStore

        store = FakeGraphStore()
        path = "/app/mist-memory/users/raj.md"
        store.add_triple(subject="User", predicate="USES", object="Python", derived_from_path=path)
        store.add_triple(
            subject="User", predicate="EXPERT_IN", object="FastAPI", derived_from_path=path
        )

        first_count = await store.mark_orphaned_by_provenance_path(path)
        second_count = await store.mark_orphaned_by_provenance_path(path)

        assert first_count == 2, f"Expected 2 on first call; got {first_count}"
        assert second_count == 0, f"Expected 0 on second call (idempotent); got {second_count}"


# ---------------------------------------------------------------------------
# TestRetrievalOrphanFilter
# Phase 5.5 tie-up (P3 #1): the three retrieval Cypher queries in GraphStore
# must filter on the edge's own status field.
# ---------------------------------------------------------------------------


class TestRetrievalQueryOrphanFilter:
    """GraphStore retrieval queries must include a status filter on the edge.

    Specifically, get_user_relationships_to_entities, get_entity_neighborhood,
    and get_all_user_relationships must include:
        WHERE (r.status IS NULL OR r.status <> 'orphaned')
    or an equivalent predicate so that typed edges marked by
    mark_orphaned_by_provenance_path are excluded from retrieval.
    """

    def test_get_user_relationships_to_entities_filters_orphaned_edges(self):
        """get_user_relationships_to_entities Cypher must exclude orphaned edges."""
        conn = FakeNeo4jConnection(query_results=[])
        store = _store_with_conn(conn)

        store.get_user_relationships_to_entities(
            user_id="User", entity_ids=["python"], relationship_types=None
        )

        assert conn.queries, "Expected at least one query"
        query_text, _ = conn.queries[-1]
        # The query must filter on edge status
        assert "orphaned" in query_text.lower(), (
            "get_user_relationships_to_entities must filter on orphaned edge status; "
            f"query was:\n{query_text}"
        )

    def test_get_entity_neighborhood_filters_orphaned_edges(self):
        """get_entity_neighborhood Cypher must exclude orphaned edges."""
        conn = FakeNeo4jConnection(query_results=[])
        store = _store_with_conn(conn)

        store.get_entity_neighborhood(entity_id="python", max_hops=1)

        assert conn.queries, "Expected at least one query"
        query_text, _ = conn.queries[-1]
        assert "orphaned" in query_text.lower(), (
            "get_entity_neighborhood must filter on orphaned edge status; "
            f"query was:\n{query_text}"
        )

    def test_get_all_user_relationships_filters_orphaned_edges(self):
        """get_all_user_relationships Cypher must exclude orphaned edges."""
        conn = FakeNeo4jConnection(query_results=[])
        store = _store_with_conn(conn)

        store.get_all_user_relationships(user_id="User")

        assert conn.queries, "Expected at least one query"
        query_text, _ = conn.queries[-1]
        assert "orphaned" in query_text.lower(), (
            "get_all_user_relationships must filter on orphaned edge status; "
            f"query was:\n{query_text}"
        )
