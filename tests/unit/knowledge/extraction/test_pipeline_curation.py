"""Tests for ExtractionPipeline curation integration."""

from backend.knowledge.extraction.pipeline import ExtractionPipeline
from tests.mocks.config import build_test_config


class TestCurationIntegration:
    def test_accepts_curation_pipeline_param(self):
        """ExtractionPipeline constructor accepts curation_pipeline=None."""
        from backend.knowledge.extraction.confidence import ConfidenceScorer
        from backend.knowledge.extraction.normalizer import EntityNormalizer
        from backend.knowledge.extraction.ontology_extractor import OntologyConstrainedExtractor
        from backend.knowledge.extraction.preprocessor import PreProcessor
        from backend.knowledge.extraction.temporal import TemporalResolver
        from backend.knowledge.extraction.validator import ExtractionValidator
        from backend.knowledge.storage.graph_store import GraphStore
        from tests.mocks.embeddings import FakeEmbeddingGenerator
        from tests.mocks.neo4j import FakeNeo4jConnection
        from tests.mocks.ollama import FakeLLM

        conn = FakeNeo4jConnection()
        embeddings = FakeEmbeddingGenerator()
        gs = GraphStore(conn, embeddings)
        config = build_test_config()

        pipeline = ExtractionPipeline(
            preprocessor=PreProcessor(),
            extractor=OntologyConstrainedExtractor(config, llm=FakeLLM()),
            confidence_scorer=ConfidenceScorer(),
            temporal_resolver=TemporalResolver(),
            normalizer=EntityNormalizer(embedding_generator=embeddings, executor=None),
            validator=ExtractionValidator(min_confidence=0.5),
            graph_store=gs,
            curation_pipeline=None,
        )
        assert pipeline._curation_pipeline is None
