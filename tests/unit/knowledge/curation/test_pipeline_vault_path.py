"""ADR-010 Cluster 8 Phase 6: vault_note_path forwarding through CurationPipeline.

`CurationPipeline.curate_and_store` accepts an optional `vault_note_path`
and forwards it to `CurationGraphWriter.write` so the graph writer can
emit the load-bearing DERIVED_FROM edges. Verifies forwarding semantics
and the legacy None-default path.
"""

from __future__ import annotations

from typing import Any

import pytest

from backend.knowledge.curation.confidence import ConfidenceManager
from backend.knowledge.curation.conflict_resolver import ConflictResolver, SupersessionAction
from backend.knowledge.curation.deduplication import EntityDeduplicator
from backend.knowledge.curation.graph_writer import (
    CurationGraphWriter,
    SourceMetadata,
    WriteResult,
)
from backend.knowledge.curation.pipeline import CurationPipeline
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeGraphExecutor, FakeNeo4jConnection
from tests.unit.knowledge.curation.conftest import make_entity_dict, make_validation_result


class _RecordingGraphWriter(CurationGraphWriter):
    """Wraps CurationGraphWriter to record arguments to write()."""

    def __init__(self, executor, embeddings, confidence) -> None:
        super().__init__(executor, embeddings, confidence)
        self.write_calls: list[dict[str, Any]] = []

    async def write(  # type: ignore[override]
        self,
        entities: list[dict],
        relationships: list[dict],
        merge_actions: list,
        supersession_actions: list[SupersessionAction],
        event_id: str,
        session_id: str,
        source_metadata: SourceMetadata | None = None,
        vault_note_path: str | None = None,
    ) -> WriteResult:
        self.write_calls.append(
            {
                "entities": entities,
                "relationships": relationships,
                "merge_actions": merge_actions,
                "supersession_actions": supersession_actions,
                "event_id": event_id,
                "session_id": session_id,
                "source_metadata": source_metadata,
                "vault_note_path": vault_note_path,
            }
        )
        return await super().write(
            entities=entities,
            relationships=relationships,
            merge_actions=merge_actions,
            supersession_actions=supersession_actions,
            event_id=event_id,
            session_id=session_id,
            source_metadata=source_metadata,
            vault_note_path=vault_note_path,
        )


def _build_pipeline_with_recorder() -> tuple[CurationPipeline, _RecordingGraphWriter]:
    conn = FakeNeo4jConnection()
    executor = FakeGraphExecutor(connection=conn)
    embeddings = FakeEmbeddingGenerator()
    confidence = ConfidenceManager()

    recorder = _RecordingGraphWriter(executor, embeddings, confidence)
    pipeline = CurationPipeline(
        deduplicator=EntityDeduplicator(executor, embeddings, confidence),
        conflict_resolver=ConflictResolver(executor),
        graph_writer=recorder,
    )
    return pipeline, recorder


class TestVaultNotePathForwarding:
    @pytest.mark.asyncio
    async def test_forwards_vault_note_path_to_graph_writer(self) -> None:
        # Arrange
        pipeline, recorder = _build_pipeline_with_recorder()
        validation = make_validation_result(entities=[make_entity_dict()])

        # Act
        await pipeline.curate_and_store(
            validation,
            event_id="evt-001",
            session_id="sess-001",
            vault_note_path="/vault/sessions/2026-04-22-foo.md",
        )

        # Assert
        assert len(recorder.write_calls) == 1
        assert recorder.write_calls[0]["vault_note_path"] == "/vault/sessions/2026-04-22-foo.md"

    @pytest.mark.asyncio
    async def test_defaults_vault_note_path_to_none(self) -> None:
        # Arrange
        pipeline, recorder = _build_pipeline_with_recorder()
        validation = make_validation_result(entities=[make_entity_dict()])

        # Act -- omit vault_note_path
        await pipeline.curate_and_store(
            validation,
            event_id="evt-001",
            session_id="sess-001",
        )

        # Assert -- default propagates as None to graph writer
        assert len(recorder.write_calls) == 1
        assert recorder.write_calls[0]["vault_note_path"] is None

    @pytest.mark.asyncio
    async def test_short_circuit_does_not_invoke_writer_with_path(self) -> None:
        # Arrange -- empty entities short-circuits before write
        pipeline, recorder = _build_pipeline_with_recorder()
        validation = make_validation_result(entities=[], relationships=[])

        # Act
        await pipeline.curate_and_store(
            validation,
            event_id="evt-001",
            session_id="sess-001",
            vault_note_path="/vault/sessions/x.md",
        )

        # Assert -- writer never called when there's nothing to write
        assert recorder.write_calls == []

    @pytest.mark.asyncio
    async def test_vault_path_coexists_with_source_metadata(self) -> None:
        # Arrange -- vault_note_path is independent of document source_metadata.
        # In practice, a turn produces ONE of (conversation -> ConversationContext +
        # VaultNote) or (ingest -> ExternalSource), but the API supports both
        # being supplied for forward compatibility.
        pipeline, recorder = _build_pipeline_with_recorder()
        validation = make_validation_result(entities=[make_entity_dict()])
        source = SourceMetadata(source_uri="https://example.com/x.pdf", source_type="document")

        # Act
        await pipeline.curate_and_store(
            validation,
            event_id="evt-001",
            session_id="sess-001",
            source_metadata=source,
            vault_note_path="/vault/sessions/2026-04-22-foo.md",
        )

        # Assert -- both arrive at the writer
        call = recorder.write_calls[0]
        assert call["source_metadata"] is source
        assert call["vault_note_path"] == "/vault/sessions/2026-04-22-foo.md"
