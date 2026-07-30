"""Unit tests for GraphStore's ontology-version accessor and retrieval orphan filter.

R1.3 Task 5 deleted `GraphStore.mark_orphaned_by_provenance_path` and
`GraphStore.get_orphaned_provenance_paths` -- the write side of orphan
marking, whose only consumer (`GraphRegenerator.rebuild_from_path`) stopped
re-deriving triples in an earlier R1.3 task. The classes that tested those
two methods (directly or via FakeGraphStore's mirror of them) are deleted
with them; see git history (pre-Task-5) for the removed coverage.

What remains and why:
- TestCurrentOntologyVersion: `current_ontology_version` was NOT deleted.
  R1.3 Task 6 deleted its last caller (GraphRegenerator, via the structural
  `_OntologyVersionSource` protocol it defined), leaving the accessor with
  zero callers in backend/ or scripts/ today. Task 6 chose to retain it as a
  public GraphStore accessor anyway: it is a trivial `return
  self._ontology_version`, and later R1.3 work (the seed-utterance migration,
  the rebuild-closure task) will plausibly want it. This class stays as its
  coverage.
- TestRetrievalQueryOrphanFilter: the read-side orphan filter in
  get_user_relationships_to_entities / get_entity_neighborhood /
  get_all_user_relationships is untouched by Task 5. It stays load-bearing
  for pre-R1.3 graph data already marked orphaned, even though nothing
  marks new edges orphaned going forward.
- test_graph_store_has_no_upsert_user: relocated here in Task 3 specifically
  to escape Task 6's deletion of the curation test file it used to live in.
- test_orphan_marking_surface_is_retired / test_regenerator_protocols_are_retired:
  relocated here in Task 5's fix round for the same reason -- both were
  originally appended to tests/unit/knowledge/curation/test_graph_regenerator.py,
  which Task 6 deletes wholesale along with GraphRegenerator. They are R1.3's
  only guards that GraphStore has no mark_orphaned_by_provenance_path /
  get_orphaned_provenance_paths and that backend.interfaces exports no
  GraphStoreProtocol / ExtractionPipelineProtocol; losing that home would
  have silently dropped both contracts in Task 6.
- test_ensure_mist_identity_uses_selfmodel_partition: guards the 2026-06-29
  `:__SelfModel__` partition migration, unrelated to R1.3.

All tests use FakeNeo4jConnection -- no real Neo4j.
"""

from __future__ import annotations

from backend.knowledge.storage.graph_store import GraphStore
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(*, ontology_version: str | None = None) -> GraphStore:
    """Build a GraphStore with a FakeNeo4jConnection."""
    conn = FakeNeo4jConnection()
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


def test_graph_store_has_no_upsert_user() -> None:
    """The Bucket-1 fact sink is deleted, not merely unreferenced (R1.3).

    Lives in this storage-layer test file, not the curation test file that
    exercises GraphRegenerator dispatch: this is a storage-layer contract
    (GraphStore's own surface), and the curation test file (Task 6, R1.3)
    is deleted wholesale when GraphRegenerator itself retires.
    """
    assert not hasattr(
        GraphStore, "upsert_user"
    ), "R1.3: GraphStore.upsert_user is the Bucket-1 fact sink and retires with it"


def test_orphan_marking_surface_is_retired() -> None:
    """R1.3: with no re-derivation, orphan-marking has no consumer.

    The methods existed to preserve triples for a rebuild that no longer
    happens. Leaving them would leave a status field nothing ever clears.

    Relocated here from test_graph_regenerator.py in Task 5's fix round --
    that file is deleted wholesale in Task 6 along with GraphRegenerator,
    which would have taken this guard down with it.
    """
    assert not hasattr(GraphStore, "mark_orphaned_by_provenance_path")
    assert not hasattr(GraphStore, "get_orphaned_provenance_paths")


def test_regenerator_protocols_are_retired() -> None:
    """The protocols typed only the regenerator's dependencies.

    Relocated here from test_graph_regenerator.py in Task 5's fix round,
    same reason as test_orphan_marking_surface_is_retired above.
    """
    import backend.interfaces as interfaces

    assert not hasattr(interfaces, "GraphStoreProtocol")
    assert not hasattr(interfaces, "ExtractionPipelineProtocol")


# ---------------------------------------------------------------------------
# SelfModel partition tests (T6)
# ---------------------------------------------------------------------------


def test_ensure_mist_identity_uses_selfmodel_partition():
    conn = FakeNeo4jConnection()
    store = _store_with_conn(conn)

    store.ensure_mist_identity()

    issued = [q for q, _ in conn.writes]
    assert any("__SelfModel__:MistIdentity" in q for q in issued), issued
    assert not any("__Entity__:MistIdentity" in q for q in issued), issued


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
    or an equivalent predicate so that typed edges marked by the (now
    retired) mark_orphaned_by_provenance_path -- or any legacy orphaned edge
    already in the graph -- are excluded from retrieval.
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
