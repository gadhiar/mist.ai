"""Integration tests for ExtractionPipeline against real Neo4j + Ollama.

Requires running services. Skip automatically if services unavailable.
Start with: ./scripts/start-dev.sh
"""

import pytest

from tests.integration.conftest import skipif_no_services


@skipif_no_services
class TestExtractionPipelineIntegration:
    """End-to-end extraction pipeline tests with real LLM + graph."""

    @pytest.mark.asyncio
    async def test_extract_entities_from_simple_statement(self, integration_config):
        """Extract entities from a clear factual statement."""
        from backend.factories import build_extraction_pipeline

        pipeline = build_extraction_pipeline(
            integration_config, include_curation=False, include_internal_derivation=False
        )

        result = await pipeline.extract_from_utterance(
            utterance="I use Python and JavaScript for web development",
            conversation_history=[],
            event_id="integration-test-001",
            session_id="integration-session",
        )

        assert result.valid
        # Should extract at least Python and JavaScript as Technology entities
        entity_ids = [e.get("id", "") for e in result.entities]
        assert (
            len(result.entities) >= 2
        ), f"Expected at least 2 entities, got {len(result.entities)}: {entity_ids}"

        # Check that recognized technologies are present
        entity_types = [e.get("type", "") for e in result.entities]
        assert "Technology" in entity_types, f"Expected Technology type, got: {entity_types}"

    @pytest.mark.asyncio
    async def test_extract_nothing_from_greeting(self, integration_config):
        """Greetings should produce no entities."""
        from backend.factories import build_extraction_pipeline

        pipeline = build_extraction_pipeline(
            integration_config, include_curation=False, include_internal_derivation=False
        )

        result = await pipeline.extract_from_utterance(
            utterance="Hello, how are you today?",
            conversation_history=[],
            event_id="integration-test-002",
            session_id="integration-session",
        )

        assert result.valid
        # Greetings typically yield no entities
        assert (
            len(result.entities) <= 1
        ), f"Expected 0-1 entities from greeting, got {len(result.entities)}"

    @pytest.mark.asyncio
    async def test_extract_relationships(self, integration_config):
        """Extract relationships from a statement about skills."""
        from backend.factories import build_extraction_pipeline

        pipeline = build_extraction_pipeline(
            integration_config, include_curation=False, include_internal_derivation=False
        )

        result = await pipeline.extract_from_utterance(
            utterance="I work at Google and I'm an expert in machine learning",
            conversation_history=[],
            event_id="integration-test-003",
            session_id="integration-session",
        )

        assert result.valid
        assert len(result.entities) >= 2
        # Should have relationships like WORKS_AT, EXPERT_IN
        assert (
            len(result.relationships) >= 1
        ), f"Expected relationships, got {len(result.relationships)}"


@skipif_no_services
class TestSignalDetectorIntegration:
    """Test signal detection with real conversation patterns."""

    def test_detects_feedback_in_real_message(self):
        from backend.knowledge.extraction.signal_detector import SignalDetector

        detector = SignalDetector()

        # Positive feedback
        result = detector.detect("I love when you give me step by step explanations")
        assert result.has_signals
        assert "feedback" in result.signal_types

        # No signal
        result = detector.detect("What is the capital of France?")
        assert not result.has_signals


@skipif_no_services
class TestGraphStoreIntegration:
    """Test graph operations against real Neo4j."""

    @pytest.mark.asyncio
    async def test_ensure_mist_identity_creates_singleton(self, integration_config):
        from backend.factories import build_graph_store

        gs = build_graph_store(integration_config)
        gs.ensure_mist_identity()

        # Verify it exists
        results = gs.connection.execute_query(
            "MATCH (m:MistIdentity {id: 'mist-identity'}) RETURN m.entity_type AS type"
        )
        assert len(results) == 1
        assert results[0]["type"] == "MistIdentity"

        # Second call is idempotent
        gs.ensure_mist_identity()
        results = gs.connection.execute_query("MATCH (m:MistIdentity) RETURN count(m) AS count")
        assert results[0]["count"] == 1

        # Cleanup
        gs.connection.execute_write("MATCH (m:MistIdentity {id: 'mist-identity'}) DETACH DELETE m")
