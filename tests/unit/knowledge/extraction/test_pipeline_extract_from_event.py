"""ExtractionPipeline.extract_from_event: fact-time anchoring for event replay.

extract_from_event is the event-store re-extraction path used by
GraphRegenerator and the R1 rebuild machinery. This file used to also cover
vault_note_path forwarding (ADR-010 Cluster 8 Phase 6); R1.3 retired that
parameter from the whole curation/extraction path, so only the recorded_at
anchoring behavior remains -- it is real production behavior with no other
test coverage.
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
        recorded_at: str | None = None,
    ):
        self.calls.append(
            {
                "validation_result": validation_result,
                "event_id": event_id,
                "session_id": session_id,
                "source_metadata": source_metadata,
                "recorded_at": recorded_at,
            }
        )

        # Return a CurationResult-shaped object with empty defaults so the
        # pipeline's logging path doesn't crash.
        from backend.knowledge.curation.deduplication import DeduplicationResult
        from backend.knowledge.curation.graph_writer import WriteResult
        from backend.knowledge.curation.pipeline import CurationResult
        from backend.knowledge.curation.reconciliation import ReconcileTurnResult

        return CurationResult(
            write_result=WriteResult(),
            dedup_result=DeduplicationResult(entities=[], merge_actions=[], entities_merged=0),
            reconcile_result=ReconcileTurnResult(),
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


class TestExtractFromEventFactTime:
    @pytest.mark.asyncio
    async def test_anchors_recorded_at_to_event_timestamp(self) -> None:
        # Arrange -- the event-store re-extraction path (used by
        # graph_regenerator and the R1 rebuild machinery).
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

        # Assert -- R1 rebuild fact-time contract: replaying a stored event
        # anchors recorded_at to the EVENT's timestamp so the rebuild resolves
        # relative dates identically to the original live turn.
        assert len(recorder.calls) == 1
        assert recorder.calls[0]["recorded_at"] == event.timestamp.isoformat()
