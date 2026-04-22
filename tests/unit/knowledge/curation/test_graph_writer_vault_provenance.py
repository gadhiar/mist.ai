"""ADR-010 Cluster 8 Phase 6: vault-note provenance writes.

Verifies that `CurationGraphWriter.write` emits the load-bearing
`DERIVED_FROM` edges from extracted entities to a `:__Provenance__:VaultNote`
provenance node when `vault_note_path` is supplied. This is the architectural
bridge that makes the graph rebuildable from the vault.

The legacy pre-Phase-6 path (vault_note_path=None) is verified to remain
emission-free so document-ingest paths and tests with the vault layer
disabled continue to behave unchanged.
"""

from __future__ import annotations

import pytest

from backend.knowledge.curation.confidence import ConfidenceManager
from backend.knowledge.curation.graph_writer import CurationGraphWriter
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeGraphExecutor, FakeNeo4jConnection
from tests.unit.knowledge.curation.conftest import make_entity_dict, make_relationship_dict


def _make_writer() -> tuple[CurationGraphWriter, FakeNeo4jConnection]:
    conn = FakeNeo4jConnection()
    executor = FakeGraphExecutor(connection=conn)
    writer = CurationGraphWriter(executor, FakeEmbeddingGenerator(), ConfidenceManager())
    return writer, conn


def _writes_matching(conn: FakeNeo4jConnection, *needles: str) -> list[tuple[str, dict]]:
    return [(q, p) for q, p in conn.writes if all(n in q for n in needles)]


def _vault_note_node_merges(conn: FakeNeo4jConnection) -> list[tuple[str, dict]]:
    """Queries that MERGE the VaultNote NODE (not edges that match it)."""
    return [(q, p) for q, p in conn.writes if "MERGE (vn:__Provenance__:VaultNote" in q]


# ---------------------------------------------------------------------------
# TestEnsureVaultNote -- node creation
# ---------------------------------------------------------------------------


class TestEnsureVaultNote:
    @pytest.mark.asyncio
    async def test_creates_vault_note_with_provenance_label(self) -> None:
        # Arrange
        writer, conn = _make_writer()

        # Act
        await writer._ensure_vault_note(
            vault_note_path="/vault/sessions/2026-04-22-foo.md",
            event_id="evt-001",
            now="2026-04-22T00:00:00Z",
        )

        # Assert
        merges = _vault_note_node_merges(conn)
        assert merges, f"Expected a VaultNote MERGE, got writes: {conn.writes}"
        merge_query = merges[0][0]
        assert (
            "__Provenance__:VaultNote" in merge_query
        ), f"VaultNote must carry :__Provenance__, got: {merge_query}"
        assert (
            "__Entity__:VaultNote" not in merge_query
        ), f"VaultNote must not carry :__Entity__, got: {merge_query}"

    @pytest.mark.asyncio
    async def test_vault_note_keyed_by_path(self) -> None:
        # Arrange
        writer, conn = _make_writer()

        # Act
        await writer._ensure_vault_note(
            vault_note_path="/vault/sessions/2026-04-22-foo.md",
            event_id="evt-001",
            now="2026-04-22T00:00:00Z",
        )

        # Assert -- the MERGE key is `path`, not event_id (event_id is a property only)
        merges = _vault_note_node_merges(conn)
        merge_query = merges[0][0]
        assert "{path:" in merge_query, (
            "VaultNote MERGE must key on path so subsequent turns of the same "
            f"session reuse the node, got: {merge_query}"
        )
        assert "{path: $path}" in merge_query
        params = merges[0][1]
        assert params["path"] == "/vault/sessions/2026-04-22-foo.md"
        assert params["event_id"] == "evt-001"

    @pytest.mark.asyncio
    async def test_repeated_calls_with_same_path_are_idempotent(self) -> None:
        # Arrange
        writer, conn = _make_writer()
        path = "/vault/sessions/2026-04-22-foo.md"

        # Act -- two calls with the same path (simulating two turns of one session)
        await writer._ensure_vault_note(path, "evt-001", "2026-04-22T00:00:00Z")
        await writer._ensure_vault_note(path, "evt-002", "2026-04-22T00:01:00Z")

        # Assert -- both calls execute MERGE; idempotency comes from MERGE semantics,
        # not from caller-side dedup. Verify the second call carries the new event_id.
        merges = _vault_note_node_merges(conn)
        assert len(merges) == 2, "Expected two MERGE calls (idempotency via Cypher MERGE)"
        assert merges[0][1]["event_id"] == "evt-001"
        assert merges[1][1]["event_id"] == "evt-002"

    @pytest.mark.asyncio
    async def test_set_clauses_separate_first_and_last_event(self) -> None:
        # Arrange
        writer, conn = _make_writer()

        # Act
        await writer._ensure_vault_note(
            vault_note_path="/vault/sessions/2026-04-22-x.md",
            event_id="evt-init",
            now="2026-04-22T00:00:00Z",
        )

        # Assert -- ON CREATE stamps both first_event_id + last_event_id;
        # ON MATCH refreshes only last_event_id. This preserves a stable
        # creation audit while letting last_event_id track the most recent
        # turn that wrote into the note.
        merge_query = _vault_note_node_merges(conn)[0][0]
        assert "ON CREATE SET vn.first_event_id = $event_id" in merge_query
        assert "vn.last_event_id = $event_id" in merge_query
        assert "ON MATCH SET vn.last_event_id = $event_id" in merge_query


# ---------------------------------------------------------------------------
# TestCreateVaultNoteProvenance -- DERIVED_FROM edge
# ---------------------------------------------------------------------------


class TestCreateVaultNoteProvenance:
    @pytest.mark.asyncio
    async def test_emits_derived_from_edge_to_vault_note(self) -> None:
        # Arrange
        writer, conn = _make_writer()

        # Act
        await writer._create_vault_note_provenance(
            entity_id="python",
            vault_note_path="/vault/sessions/2026-04-22-foo.md",
            event_id="evt-001",
            now="2026-04-22T00:00:00Z",
        )

        # Assert
        edges = _writes_matching(conn, "DERIVED_FROM", "VaultNote")
        assert edges, f"Expected a DERIVED_FROM->VaultNote MERGE, got writes: {conn.writes}"
        edge_query = edges[0][0]
        assert "MERGE (e)-[r:DERIVED_FROM]->(vn)" in edge_query
        assert ":__Entity__ {id: $entity_id}" in edge_query
        assert "VaultNote {path: $path}" in edge_query

    @pytest.mark.asyncio
    async def test_edge_carries_event_id_and_timestamps(self) -> None:
        # Arrange
        writer, conn = _make_writer()

        # Act
        await writer._create_vault_note_provenance(
            entity_id="neo4j",
            vault_note_path="/vault/sessions/2026-04-22-bar.md",
            event_id="evt-042",
            now="2026-04-22T01:23:45Z",
        )

        # Assert -- edge carries event_id on both create and match;
        # created_at on first creation, updated_at on subsequent matches.
        edge_query = _writes_matching(conn, "DERIVED_FROM", "VaultNote")[0][0]
        assert "ON CREATE SET r.event_id = $event_id" in edge_query
        assert "r.created_at = $now" in edge_query
        assert "ON MATCH SET r.event_id = $event_id" in edge_query
        assert "r.updated_at = $now" in edge_query

        params = _writes_matching(conn, "DERIVED_FROM", "VaultNote")[0][1]
        assert params["entity_id"] == "neo4j"
        assert params["path"] == "/vault/sessions/2026-04-22-bar.md"
        assert params["event_id"] == "evt-042"


# ---------------------------------------------------------------------------
# TestWriteWithVaultNotePath -- end-to-end emission via write()
# ---------------------------------------------------------------------------


class TestWriteWithVaultNotePath:
    @pytest.mark.asyncio
    async def test_write_emits_vault_note_node_once_per_call(self) -> None:
        # Arrange
        writer, conn = _make_writer()
        entities = [
            make_entity_dict(entity_id="python", display_name="Python"),
            make_entity_dict(entity_id="neo4j", entity_type="Technology", display_name="Neo4j"),
        ]

        # Act
        await writer.write(
            entities=entities,
            relationships=[],
            merge_actions=[],
            supersession_actions=[],
            event_id="evt-001",
            session_id="sess-001",
            vault_note_path="/vault/sessions/2026-04-22-foo.md",
        )

        # Assert -- exactly one VaultNote node MERGE for the write call,
        # not one per entity (per-entity work is the DERIVED_FROM edge).
        vn_merges = _vault_note_node_merges(conn)
        assert len(vn_merges) == 1, (
            f"Expected one VaultNote node MERGE for two entities, got {len(vn_merges)} "
            f"queries: {[q for q, _ in vn_merges]}"
        )

    @pytest.mark.asyncio
    async def test_write_emits_one_derived_from_edge_per_entity(self) -> None:
        # Arrange
        writer, conn = _make_writer()
        entities = [
            make_entity_dict(entity_id="python", display_name="Python"),
            make_entity_dict(entity_id="neo4j", entity_type="Technology", display_name="Neo4j"),
            make_entity_dict(entity_id="lancedb", entity_type="Technology", display_name="LanceDB"),
        ]

        # Act
        result = await writer.write(
            entities=entities,
            relationships=[],
            merge_actions=[],
            supersession_actions=[],
            event_id="evt-001",
            session_id="sess-001",
            vault_note_path="/vault/sessions/2026-04-22-foo.md",
        )

        # Assert -- one DERIVED_FROM edge per upserted entity
        edges = _writes_matching(conn, "DERIVED_FROM", "VaultNote")
        assert (
            len(edges) == 3
        ), f"Expected 3 DERIVED_FROM->VaultNote edges (one per entity), got {len(edges)}"
        assert result.vault_note_provenance_edges == 3

    @pytest.mark.asyncio
    async def test_write_does_not_emit_vault_node_when_only_relationships(self) -> None:
        # Arrange -- relationships without entities don't anchor to a vault note
        # because the entities they reference were created in a prior write call
        # (which already anchored them).
        writer, conn = _make_writer()
        relationships = [make_relationship_dict()]

        # Act
        await writer.write(
            entities=[],
            relationships=relationships,
            merge_actions=[],
            supersession_actions=[],
            event_id="evt-001",
            session_id="sess-001",
            vault_note_path="/vault/sessions/2026-04-22-foo.md",
        )

        # Assert -- no VaultNote node MERGE because there were no entities to anchor
        vn_merges = _vault_note_node_merges(conn)
        assert vn_merges == [], (
            f"Expected zero VaultNote node MERGEs when entities=[], got: "
            f"{[q for q, _ in vn_merges]}"
        )

    @pytest.mark.asyncio
    async def test_write_emits_no_vault_writes_when_path_is_none(self) -> None:
        # Arrange -- legacy pre-Phase-6 path
        writer, conn = _make_writer()
        entities = [make_entity_dict(entity_id="python", display_name="Python")]

        # Act -- no vault_note_path
        result = await writer.write(
            entities=entities,
            relationships=[],
            merge_actions=[],
            supersession_actions=[],
            event_id="evt-001",
            session_id="sess-001",
        )

        # Assert -- zero VaultNote writes, zero DERIVED_FROM->VaultNote edges
        assert _vault_note_node_merges(conn) == []
        assert _writes_matching(conn, "DERIVED_FROM", "VaultNote") == []
        assert result.vault_note_provenance_edges == 0
        # The legacy ConversationContext provenance must still fire
        assert result.provenance_edges_created == 1

    @pytest.mark.asyncio
    async def test_write_emits_vault_provenance_alongside_conversation_context(self) -> None:
        # Arrange -- both legacy ConversationContext + new VaultNote provenance
        # coexist on a conversation-turn write.
        writer, conn = _make_writer()
        entities = [make_entity_dict(entity_id="python", display_name="Python")]

        # Act
        result = await writer.write(
            entities=entities,
            relationships=[],
            merge_actions=[],
            supersession_actions=[],
            event_id="evt-001",
            session_id="sess-001",
            vault_note_path="/vault/sessions/2026-04-22-foo.md",
        )

        # Assert -- both provenance counters incremented
        assert result.provenance_edges_created == 1, "EXTRACTED_FROM still fires"
        assert result.vault_note_provenance_edges == 1, "DERIVED_FROM->VaultNote also fires"
        assert _writes_matching(conn, "EXTRACTED_FROM", "ConversationContext") != []
        assert _writes_matching(conn, "DERIVED_FROM", "VaultNote") != []

    @pytest.mark.asyncio
    async def test_two_turns_same_path_produce_idempotent_vault_note(self) -> None:
        # Arrange -- two consecutive turns of the same session both target
        # the same vault note. The node MERGE is idempotent on path; the
        # DERIVED_FROM edges are idempotent on (entity, vault_note).
        writer, conn = _make_writer()
        path = "/vault/sessions/2026-04-22-multi-turn.md"
        entities_t1 = [make_entity_dict(entity_id="python", display_name="Python")]
        entities_t2 = [make_entity_dict(entity_id="python", display_name="Python")]

        # Act
        await writer.write(
            entities=entities_t1,
            relationships=[],
            merge_actions=[],
            supersession_actions=[],
            event_id="evt-turn-1",
            session_id="sess-001",
            vault_note_path=path,
        )
        await writer.write(
            entities=entities_t2,
            relationships=[],
            merge_actions=[],
            supersession_actions=[],
            event_id="evt-turn-2",
            session_id="sess-001",
            vault_note_path=path,
        )

        # Assert -- two node MERGE calls in Cypher (one per turn); idempotency is
        # enforced by Neo4j's MERGE semantics, not by caller-side dedup.
        # Both turns target the same path, so the second call produces an
        # ON MATCH branch in production.
        vn_merges = _vault_note_node_merges(conn)
        assert len(vn_merges) == 2
        assert vn_merges[0][1]["path"] == path
        assert vn_merges[1][1]["path"] == path
        # Edge MERGEs: also two (one per turn), each refreshing event_id.
        edges = _writes_matching(conn, "DERIVED_FROM", "VaultNote")
        assert len(edges) == 2
        assert edges[0][1]["event_id"] == "evt-turn-1"
        assert edges[1][1]["event_id"] == "evt-turn-2"

    @pytest.mark.asyncio
    async def test_derived_from_to_vault_note_excluded_from_user_facing_traversal(self) -> None:
        # Arrange + Assert -- DERIVED_FROM is a provenance edge type and must
        # NOT appear in the user-facing relationship-type allowlist used by
        # graph traversal. Crossing into VaultNote at hop 2+ would defeat
        # ADR-009's :__Entity__ vs :__Provenance__ separation.
        from backend.knowledge.storage.graph_store import _USER_FACING_REL_TYPES

        assert "DERIVED_FROM" not in _USER_FACING_REL_TYPES, (
            "DERIVED_FROM is a provenance edge (entity -> VaultNote / VectorChunk / "
            "ExternalSource) and must be excluded from user-facing traversal per "
            "ADR-009 + ADR-010."
        )


# ---------------------------------------------------------------------------
# TestPhase8RebuildStamps -- ontology_version + extraction_version + model_hash
# ---------------------------------------------------------------------------


class TestPhase8RebuildStamps:
    """ADR-010 Phase 8: DERIVED_FROM edges carry rebuild-determinism stamps
    when `RebuildStamps` is injected at construction. `vault-rebuild` compares
    these against current values to detect ontology / prompt / model drift.
    """

    def _writer_with_stamps(
        self,
        ontology_version: str = "1.0.0",
        extraction_version: str = "2026-04-17-r1",
        model_hash: str = "gemma-4-e4b-q5-k-m-test",
    ) -> tuple[CurationGraphWriter, FakeNeo4jConnection]:
        from backend.knowledge.curation.graph_writer import RebuildStamps

        conn = FakeNeo4jConnection()
        executor = FakeGraphExecutor(connection=conn)
        stamps = RebuildStamps(
            ontology_version=ontology_version,
            extraction_version=extraction_version,
            model_hash=model_hash,
        )
        writer = CurationGraphWriter(
            executor,
            FakeEmbeddingGenerator(),
            ConfidenceManager(),
            rebuild_stamps=stamps,
        )
        return writer, conn

    def test_rebuild_stamps_dataclass_is_frozen(self) -> None:
        # Frozen so the stamps cannot drift mid-process; rebuild determinism
        # depends on a stable per-deployment value set.
        from dataclasses import FrozenInstanceError

        from backend.knowledge.curation.graph_writer import RebuildStamps

        stamps = RebuildStamps(
            ontology_version="1.0.0",
            extraction_version="2026-04-17-r1",
            model_hash="gemma-4-e4b",
        )

        with pytest.raises(FrozenInstanceError):
            stamps.ontology_version = "2.0.0"  # type: ignore[misc]

    @pytest.mark.asyncio
    async def test_edge_omits_stamps_when_rebuild_stamps_is_none(self) -> None:
        # Arrange -- the Phase 6 default; preserves backward compatibility
        # with deployments that have not yet adopted Phase 8 config wiring.
        writer, conn = _make_writer()

        # Act
        await writer._create_vault_note_provenance(
            entity_id="python",
            vault_note_path="/vault/sessions/2026-04-22-foo.md",
            event_id="evt-001",
            now="2026-04-22T00:00:00Z",
        )

        # Assert -- only Phase 6 properties on the edge
        edge_query = _writes_matching(conn, "DERIVED_FROM", "VaultNote")[0][0]
        assert "ontology_version" not in edge_query
        assert "extraction_version" not in edge_query
        assert "model_hash" not in edge_query
        assert "derived_at" not in edge_query
        assert "r.event_id = $event_id" in edge_query

    @pytest.mark.asyncio
    async def test_edge_carries_all_three_stamps_when_set(self) -> None:
        # Arrange
        writer, conn = self._writer_with_stamps(
            ontology_version="1.0.0",
            extraction_version="2026-04-17-r1",
            model_hash="gemma-4-e4b-q5-k-m-carteakey-full-v1",
        )

        # Act
        await writer._create_vault_note_provenance(
            entity_id="python",
            vault_note_path="/vault/sessions/2026-04-22-foo.md",
            event_id="evt-001",
            now="2026-04-22T00:00:00Z",
        )

        # Assert -- all three stamps + derived_at appear on the edge,
        # and parameters carry the configured values.
        query, params = _writes_matching(conn, "DERIVED_FROM", "VaultNote")[0]
        assert "r.ontology_version = $ontology_version" in query
        assert "r.extraction_version = $extraction_version" in query
        assert "r.model_hash = $model_hash" in query
        assert "r.derived_at = $now" in query

        assert params["ontology_version"] == "1.0.0"
        assert params["extraction_version"] == "2026-04-17-r1"
        assert params["model_hash"] == "gemma-4-e4b-q5-k-m-carteakey-full-v1"

    @pytest.mark.asyncio
    async def test_stamps_appear_on_both_create_and_match_branches(self) -> None:
        # Arrange -- ADR-010 says re-extraction should land the CURRENT
        # stamp set, not retain the original. So ON MATCH must update
        # the stamps too, not just touch updated_at.
        writer, conn = self._writer_with_stamps()

        # Act
        await writer._create_vault_note_provenance(
            entity_id="python",
            vault_note_path="/vault/sessions/2026-04-22-foo.md",
            event_id="evt-001",
            now="2026-04-22T00:00:00Z",
        )

        # Assert -- both ON CREATE and ON MATCH set the stamps
        query = _writes_matching(conn, "DERIVED_FROM", "VaultNote")[0][0]
        # Split on ON MATCH to verify both branches contain the stamps
        on_create_idx = query.find("ON CREATE SET")
        on_match_idx = query.find("ON MATCH SET")
        assert on_create_idx >= 0 and on_match_idx > on_create_idx
        on_create_clause = query[on_create_idx:on_match_idx]
        on_match_clause = query[on_match_idx:]
        for stamp in ("ontology_version", "extraction_version", "model_hash"):
            assert stamp in on_create_clause, f"{stamp} missing from ON CREATE"
            assert stamp in on_match_clause, f"{stamp} missing from ON MATCH"

    @pytest.mark.asyncio
    async def test_stamps_propagate_through_write_method(self) -> None:
        # Arrange -- end-to-end via write(), not just the helper. Confirms
        # the constructor injection survives the full write path.
        writer, conn = self._writer_with_stamps(
            ontology_version="1.0.0",
            extraction_version="custom-version-test",
            model_hash="custom-model-hash-test",
        )
        entities = [make_entity_dict(entity_id="python", display_name="Python")]

        # Act
        await writer.write(
            entities=entities,
            relationships=[],
            merge_actions=[],
            supersession_actions=[],
            event_id="evt-001",
            session_id="sess-001",
            vault_note_path="/vault/sessions/2026-04-22-foo.md",
        )

        # Assert
        edges = _writes_matching(conn, "DERIVED_FROM", "VaultNote")
        assert len(edges) == 1
        params = edges[0][1]
        assert params["extraction_version"] == "custom-version-test"
        assert params["model_hash"] == "custom-model-hash-test"


class TestPhase8FactoryWiring:
    """Verifies build_curation_pipeline constructs RebuildStamps from config."""

    def test_factory_constructs_stamps_from_config(self) -> None:
        from backend.factories import build_curation_pipeline
        from backend.knowledge.curation.graph_writer import RebuildStamps
        from tests.mocks.config import build_test_config
        from tests.mocks.neo4j import FakeGraphExecutor, FakeNeo4jConnection

        # Arrange
        config = build_test_config()
        config.extraction_version = "factory-test-version"
        config.model_hash = "factory-test-model"
        config.ontology_version = "1.0.0"

        executor = FakeGraphExecutor(connection=FakeNeo4jConnection())

        # Act
        pipeline = build_curation_pipeline(config, executor)

        # Assert -- the pipeline's graph writer carries stamps matching config
        stamps = pipeline._graph_writer._rebuild_stamps  # type: ignore[attr-defined]
        assert isinstance(stamps, RebuildStamps)
        assert stamps.ontology_version == "1.0.0"
        assert stamps.extraction_version == "factory-test-version"
        assert stamps.model_hash == "factory-test-model"
