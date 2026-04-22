"""ADR-010 Cluster 8 Phase 6: vault_note_path forwarding through ExtractionPipeline.

`ExtractionPipeline.extract_from_utterance` accepts an optional
`vault_note_path` and forwards it to `CurationPipeline.curate_and_store`
when curation is enabled. Verifies forwarding semantics and the
None-default backward-compatibility path.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from backend.knowledge.config import ExtractionConfig
from backend.knowledge.extraction.confidence import ConfidenceScorer
from backend.knowledge.extraction.normalizer import EntityNormalizer
from backend.knowledge.extraction.ontology_extractor import (
    ExtractionResult,
    OntologyConstrainedExtractor,
)
from backend.knowledge.extraction.pipeline import ExtractionPipeline
from backend.knowledge.extraction.preprocessor import PreProcessor
from backend.knowledge.extraction.temporal import TemporalResolver
from backend.knowledge.extraction.validator import ExtractionValidator
from backend.knowledge.storage.graph_store import GraphStore
from tests.mocks.config import TEST_EVENT_ID, TEST_SESSION_ID
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection


class _RecordingCurationPipeline:
    """Test double recording every curate_and_store call's kwargs."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def curate_and_store(
        self,
        validation_result,
        event_id: str,
        session_id: str,
        source_metadata=None,
        vault_note_path: str | None = None,
    ):
        self.calls.append(
            {
                "validation_result": validation_result,
                "event_id": event_id,
                "session_id": session_id,
                "source_metadata": source_metadata,
                "vault_note_path": vault_note_path,
            }
        )

        # Return a CurationResult-shaped object with empty defaults so the
        # pipeline's logging path doesn't crash.
        from backend.knowledge.curation.conflict_resolver import ConflictResolutionResult
        from backend.knowledge.curation.deduplication import DeduplicationResult
        from backend.knowledge.curation.graph_writer import WriteResult
        from backend.knowledge.curation.pipeline import CurationResult

        return CurationResult(
            write_result=WriteResult(),
            dedup_result=DeduplicationResult(entities=[], merge_actions=[], entities_merged=0),
            conflict_result=ConflictResolutionResult(relationships=[]),
            curation_time_ms=0.0,
        )


def _entity_extractor() -> AsyncMock:
    """Return an extractor that produces one entity, so curation is invoked."""
    mock = AsyncMock(spec=OntologyConstrainedExtractor)
    mock.extract.return_value = ExtractionResult(
        entities=[
            {
                "id": "python",
                "type": "Technology",
                "name": "Python",
                "confidence": 0.9,
                "source_type": "extracted",
                "aliases": [],
                "description": "",
            }
        ],
        relationships=[],
        raw_llm_output='{"entities": [{"id": "python"}]}',
        extraction_time_ms=1.0,
        source_utterance="",
    )
    return mock


def _build_pipeline(curation_pipeline: _RecordingCurationPipeline) -> ExtractionPipeline:
    conn = FakeNeo4jConnection()
    embeddings = FakeEmbeddingGenerator()
    graph_store = GraphStore(connection=conn, embedding_generator=embeddings)

    extraction_config = ExtractionConfig(
        significance_threshold=0.0,
        rate_limit_max_per_minute=1000,
        dedup_similarity_threshold=0.99,
    )

    return ExtractionPipeline(
        preprocessor=PreProcessor(),
        extractor=_entity_extractor(),
        confidence_scorer=ConfidenceScorer(),
        temporal_resolver=TemporalResolver(),
        normalizer=EntityNormalizer(embedding_generator=embeddings, executor=None),
        validator=ExtractionValidator(min_confidence=0.0),
        graph_store=graph_store,
        event_store=None,
        curation_pipeline=curation_pipeline,  # type: ignore[arg-type]
        internal_deriver=None,
        embedding_provider=FakeEmbeddingGenerator(),
        extraction_config=extraction_config,
    )


class TestVaultNotePathForwarding:
    @pytest.mark.asyncio
    async def test_forwards_vault_note_path_to_curation(self) -> None:
        # Arrange
        recorder = _RecordingCurationPipeline()
        pipeline = _build_pipeline(recorder)

        # Act
        await pipeline.extract_from_utterance(
            utterance="I love working with Python and Neo4j on a daily basis.",
            conversation_history=[],
            event_id=TEST_EVENT_ID,
            session_id=TEST_SESSION_ID,
            vault_note_path="/vault/sessions/2026-04-22-foo.md",
        )

        # Assert
        assert len(recorder.calls) == 1
        assert recorder.calls[0]["vault_note_path"] == "/vault/sessions/2026-04-22-foo.md"
        assert recorder.calls[0]["event_id"] == TEST_EVENT_ID
        assert recorder.calls[0]["session_id"] == TEST_SESSION_ID

    @pytest.mark.asyncio
    async def test_defaults_vault_note_path_to_none(self) -> None:
        # Arrange
        recorder = _RecordingCurationPipeline()
        pipeline = _build_pipeline(recorder)

        # Act -- omit vault_note_path
        await pipeline.extract_from_utterance(
            utterance="Working with TypeScript and React lately for the frontend.",
            conversation_history=[],
            event_id=TEST_EVENT_ID,
            session_id=TEST_SESSION_ID,
        )

        # Assert
        assert len(recorder.calls) == 1
        assert recorder.calls[0]["vault_note_path"] is None

    @pytest.mark.asyncio
    async def test_extract_from_event_does_not_carry_vault_note_path(self) -> None:
        # Arrange -- the event-store re-extraction path (used by graph_regenerator
        # and Phase 11 vault-rebuild). For now extract_from_event has no vault
        # awareness; it always passes None through to the curation pipeline.
        # Phase 11 may extend this.
        from datetime import datetime

        from backend.event_store.models import ConversationTurnEvent

        recorder = _RecordingCurationPipeline()
        pipeline = _build_pipeline(recorder)

        event = ConversationTurnEvent(
            session_id="es-session-001",
            turn_index=0,
            timestamp=datetime.now(),
            user_utterance="Replaying about Python and FastAPI architectures.",
            system_response="OK",
        )

        # Act
        await pipeline.extract_from_event(event=event, conversation_context=[])

        # Assert -- curation called, vault_note_path defaulted to None
        assert len(recorder.calls) == 1
        assert recorder.calls[0]["vault_note_path"] is None
