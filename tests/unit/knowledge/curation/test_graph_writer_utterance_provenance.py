"""R1.3: entity provenance anchors to the utterance, never to a vault note.

Replaces test_graph_writer_vault_provenance.py. Under R1 Inv-A1 the vault is
not a fact source, so the curation graph writer must emit NO VaultNote node
and NO DERIVED_FROM->VaultNote edge on the conversational path. The single
entity-level provenance anchor is EXTRACTED_FROM->ConversationContext, whose
`source_utterance_id` names the most recent utterance in this session that
produced the entity -- last-writer-wins on re-extraction, not pinned to the
originating utterance the way C2 stamps the identically-named property on
reconciled relationship edges (reconciliation.py).
"""

import pytest

from backend.knowledge.curation.graph_writer import (
    RebuildStamps,
    WriteResult,
)

# Reuse the fake connection helpers from the file this replaces.
from tests.unit.knowledge.curation._graph_writer_fakes import (
    make_writer,
    writes_matching,
)


class TestNoVaultNoteWrites:
    """R1.3 core contract: no vault anchor is ever written."""

    @pytest.mark.asyncio
    async def test_write_emits_no_vault_note_node(self) -> None:
        writer, conn = make_writer()
        await writer.write(
            entities=[{"id": "python", "type": "Technology", "name": "Python"}],
            merge_actions=[],
            event_id="evt-1",
            session_id="sess-1",
        )
        assert (
            writes_matching(conn, "VaultNote") == []
        ), "R1.3: no VaultNote node may be written on the conversational path"

    @pytest.mark.asyncio
    async def test_write_emits_no_derived_from_edge(self) -> None:
        writer, conn = make_writer()
        await writer.write(
            entities=[{"id": "python", "type": "Technology", "name": "Python"}],
            merge_actions=[],
            event_id="evt-1",
            session_id="sess-1",
        )
        assert (
            writes_matching(conn, "DERIVED_FROM") == []
        ), "R1.3: entity provenance no longer uses DERIVED_FROM on this path"

    def test_write_result_has_no_vault_note_counter(self) -> None:
        assert not hasattr(
            WriteResult(), "vault_note_provenance_edges"
        ), "R1.3: the VaultNote edge counter is retired with the edge"


class TestUtteranceAnchor:
    """EXTRACTED_FROM carries source_utterance_id, matching C2's vocabulary."""

    @pytest.mark.asyncio
    async def test_extracted_from_edge_sets_source_utterance_id(self) -> None:
        writer, conn = make_writer()
        await writer.write(
            entities=[{"id": "python", "type": "Technology", "name": "Python"}],
            merge_actions=[],
            event_id="evt-42",
            session_id="sess-1",
        )
        edges = writes_matching(conn, "EXTRACTED_FROM")
        assert len(edges) == 1, "exactly one entity-provenance edge per entity"
        query, params = edges[0]
        assert "r.source_utterance_id = $event_id" in query
        assert "r.event_id" not in query, "the old property name is retired"
        assert params["event_id"] == "evt-42"

    @pytest.mark.asyncio
    async def test_extracted_from_edge_omits_stamps_when_unset(self) -> None:
        writer, conn = make_writer(rebuild_stamps=None)
        await writer.write(
            entities=[{"id": "python", "type": "Technology", "name": "Python"}],
            merge_actions=[],
            event_id="evt-1",
            session_id="sess-1",
        )
        query, params = writes_matching(conn, "EXTRACTED_FROM")[0]
        assert "ontology_version" not in query
        assert "ontology_version" not in params
        # Guards the SET-clause concatenation: a lost separator or comma
        # between the status literal and the following ON MATCH keyword
        # would still leave every substring assertion above passing, so
        # this pins the exact boundary a malformed splice would break.
        assert "r.status = 'active' ON MATCH SET" in query

    @pytest.mark.asyncio
    async def test_extracted_from_edge_carries_stamps_on_both_branches(self) -> None:
        stamps = RebuildStamps(
            ontology_version="1.4.0",
            extraction_version="2026-06-14-r5",
            model_hash="abc123",
        )
        writer, conn = make_writer(rebuild_stamps=stamps)
        await writer.write(
            entities=[{"id": "python", "type": "Technology", "name": "Python"}],
            merge_actions=[],
            event_id="evt-1",
            session_id="sess-1",
        )
        query, params = writes_matching(conn, "EXTRACTED_FROM")[0]
        create_clause, match_clause = query.split("ON MATCH SET")
        for clause in (create_clause, match_clause):
            assert "r.ontology_version = $ontology_version" in clause
            assert "r.extraction_version = $extraction_version" in clause
            assert "r.model_hash = $model_hash" in clause
            # Guards a lost leading comma on stamp_clause: without it this
            # branch emits `r.status = 'active' r.ontology_version = ...`,
            # which is runtime-fatal Cypher, but every assertion above still
            # passes. Checked per-branch (not just `in query`) so a comma
            # loss on only one of CREATE/MATCH is still caught.
            assert "r.status = 'active', r.ontology_version = $ontology_version" in clause
        assert params["ontology_version"] == "1.4.0"
        assert params["extraction_version"] == "2026-06-14-r5"
        assert params["model_hash"] == "abc123"
        # Guards the SET-clause concatenation: same rationale as the
        # unstamped case above, pinned to the stamped clause's own tail.
        assert "r.derived_at = $now ON MATCH SET" in query

    @pytest.mark.asyncio
    async def test_provenance_edge_counter_still_increments(self) -> None:
        writer, conn = make_writer()
        result = await writer.write(
            entities=[
                {"id": "python", "type": "Technology", "name": "Python"},
                {"id": "neo4j", "type": "Technology", "name": "Neo4j"},
            ],
            merge_actions=[],
            event_id="evt-1",
            session_id="sess-1",
        )
        assert result.provenance_edges_created == 2


class TestVaultNotePathParameterRetired:
    """R1.3: the parameter is gone from every curation-path signature.

    A signature check, not a behavior check: the risk this guards is a future
    change re-threading a vault path into the fact writer and silently
    restoring a vault->graph anchor.
    """

    def test_graph_writer_write_has_no_vault_note_path_param(self) -> None:
        import inspect

        from backend.knowledge.curation.graph_writer import CurationGraphWriter

        params = inspect.signature(CurationGraphWriter.write).parameters
        assert "vault_note_path" not in params

    def test_curation_pipeline_has_no_vault_note_path_param(self) -> None:
        import inspect

        from backend.knowledge.curation.pipeline import CurationPipeline

        params = inspect.signature(CurationPipeline.curate_and_store).parameters
        assert "vault_note_path" not in params

    def test_extraction_pipeline_has_no_vault_note_path_param(self) -> None:
        import inspect

        from backend.knowledge.extraction.pipeline import ExtractionPipeline

        params = inspect.signature(ExtractionPipeline.extract_from_utterance).parameters
        assert "vault_note_path" not in params
