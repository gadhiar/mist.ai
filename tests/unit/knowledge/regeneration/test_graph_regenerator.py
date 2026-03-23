"""Tests for GraphRegenerator ExtractionPipeline migration."""

import pytest

from backend.knowledge.extraction.validator import ValidationResult
from backend.knowledge.models import Utterance
from backend.knowledge.regeneration.graph_regenerator import GraphRegenerator
from tests.mocks.config import build_test_config
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection


class FakeExtractionPipeline:
    """Records calls for verification."""

    def __init__(self):
        self.calls: list[dict] = []

    async def extract_from_utterance(self, **kwargs):
        self.calls.append(kwargs)
        return ValidationResult(
            valid=True,
            entities=[
                {
                    "id": "python",
                    "type": "Technology",
                    "name": "Python",
                    "confidence": 0.85,
                    "description": "",
                }
            ],
            relationships=[],
        )


class TestExtractAndStore:
    @pytest.mark.asyncio
    async def test_uses_pipeline_with_synthetic_ids(self):
        from datetime import datetime

        from backend.knowledge.storage.graph_store import GraphStore

        conn = FakeNeo4jConnection()
        gs = GraphStore(conn, FakeEmbeddingGenerator())
        pipeline = FakeExtractionPipeline()

        regenerator = GraphRegenerator(
            config=build_test_config(),
            extraction_pipeline=pipeline,
            graph_store=gs,
        )

        utterance = Utterance(
            utterance_id="utt-001",
            conversation_id="conv-001",
            text="I use Python",
            timestamp=datetime.now(),
        )

        entities, rels = await regenerator._extract_and_store(utterance)

        assert len(pipeline.calls) == 1
        assert pipeline.calls[0]["event_id"] == "regen_utt-001"
        assert pipeline.calls[0]["session_id"] == "conv-001"
        assert entities == 1
        assert rels == 0


class TestConstructorDI:
    def test_accepts_pipeline_and_graph_store(self):
        from backend.knowledge.storage.graph_store import GraphStore

        conn = FakeNeo4jConnection()
        gs = GraphStore(conn, FakeEmbeddingGenerator())
        pipeline = FakeExtractionPipeline()

        regenerator = GraphRegenerator(
            config=build_test_config(),
            extraction_pipeline=pipeline,
            graph_store=gs,
        )
        assert regenerator._pipeline is pipeline
        assert regenerator.graph_store is gs
