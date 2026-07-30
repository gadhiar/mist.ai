"""R1.3: entity provenance anchors to the utterance, never to a vault note.

Replaces test_graph_writer_vault_provenance.py. Under R1 Inv-A1 the vault is
not a fact source, so the curation graph writer must emit NO VaultNote node
and NO DERIVED_FROM->VaultNote edge on the conversational path. The single
entity-level provenance anchor is EXTRACTED_FROM->ConversationContext, whose
`source_utterance_id` names the utterance the fact came from -- the same
property C2 stamps on reconciled relationship edges (reconciliation.py).
"""

import pytest

from backend.knowledge.curation.graph_writer import (
    RebuildStamps,
    WriteResult,
)

# Reuse the fake executor/connection helpers from the file this replaces.
from tests.unit.knowledge.curation._graph_writer_fakes import (  # see Step 1b
    make_writer,
    writes_matching,
)


class TestNoVaultNoteWrites:
    """R1.3 core contract: no vault anchor is ever written."""

    @pytest.mark.asyncio
    async def test_write_emits_no_vault_note_node(self) -> None:
        writer, executor = make_writer()
        await writer.write(
            entities=[{"id": "python", "type": "Technology", "name": "Python"}],
            merge_actions=[],
            event_id="evt-1",
            session_id="sess-1",
            vault_note_path="/vault/sessions/2026-07-30.md",
        )
        assert (
            writes_matching(executor, "VaultNote") == []
        ), "R1.3: no VaultNote node may be written on the conversational path"

    @pytest.mark.asyncio
    async def test_write_emits_no_derived_from_edge(self) -> None:
        writer, executor = make_writer()
        await writer.write(
            entities=[{"id": "python", "type": "Technology", "name": "Python"}],
            merge_actions=[],
            event_id="evt-1",
            session_id="sess-1",
            vault_note_path="/vault/sessions/2026-07-30.md",
        )
        assert (
            writes_matching(executor, "DERIVED_FROM") == []
        ), "R1.3: entity provenance no longer uses DERIVED_FROM on this path"

    def test_write_result_has_no_vault_note_counter(self) -> None:
        assert not hasattr(
            WriteResult(), "vault_note_provenance_edges"
        ), "R1.3: the VaultNote edge counter is retired with the edge"


class TestUtteranceAnchor:
    """EXTRACTED_FROM carries source_utterance_id, matching C2's vocabulary."""

    @pytest.mark.asyncio
    async def test_extracted_from_edge_sets_source_utterance_id(self) -> None:
        writer, executor = make_writer()
        await writer.write(
            entities=[{"id": "python", "type": "Technology", "name": "Python"}],
            merge_actions=[],
            event_id="evt-42",
            session_id="sess-1",
        )
        edges = writes_matching(executor, "EXTRACTED_FROM")
        assert len(edges) == 1, "exactly one entity-provenance edge per entity"
        query, params = edges[0]
        assert "r.source_utterance_id = $event_id" in query
        assert "r.event_id" not in query, "the old property name is retired"
        assert params["event_id"] == "evt-42"

    @pytest.mark.asyncio
    async def test_extracted_from_edge_omits_stamps_when_unset(self) -> None:
        writer, executor = make_writer(rebuild_stamps=None)
        await writer.write(
            entities=[{"id": "python", "type": "Technology", "name": "Python"}],
            merge_actions=[],
            event_id="evt-1",
            session_id="sess-1",
        )
        query, params = writes_matching(executor, "EXTRACTED_FROM")[0]
        assert "ontology_version" not in query
        assert "ontology_version" not in params

    @pytest.mark.asyncio
    async def test_extracted_from_edge_carries_stamps_on_both_branches(self) -> None:
        stamps = RebuildStamps(
            ontology_version="1.4.0",
            extraction_version="2026-06-14-r5",
            model_hash="abc123",
        )
        writer, executor = make_writer(rebuild_stamps=stamps)
        await writer.write(
            entities=[{"id": "python", "type": "Technology", "name": "Python"}],
            merge_actions=[],
            event_id="evt-1",
            session_id="sess-1",
        )
        query, params = writes_matching(executor, "EXTRACTED_FROM")[0]
        create_clause, match_clause = query.split("ON MATCH SET")
        for clause in (create_clause, match_clause):
            assert "r.ontology_version = $ontology_version" in clause
            assert "r.extraction_version = $extraction_version" in clause
            assert "r.model_hash = $model_hash" in clause
        assert params["ontology_version"] == "1.4.0"
        assert params["extraction_version"] == "2026-06-14-r5"
        assert params["model_hash"] == "abc123"

    @pytest.mark.asyncio
    async def test_provenance_edge_counter_still_increments(self) -> None:
        writer, executor = make_writer()
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
