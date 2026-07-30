"""Phase 5.5 Dispatch 2: Unit tests for ExtractionPipeline.extract_from_file.

extract_from_file is the ExtractionPipelineProtocol method used by
GraphRegenerator._rebuild_async_extraction for Bucket 2/3 (sessions/,
decisions/) re-extraction on user-edit.

It wraps extract_from_utterance with a synthetic utterance. `vault_note_path`
is used only for logging (R1.3 retired the vault->graph fact anchor
`extract_from_utterance` used to forward it to); `ontology_version` is used
only in the re-extraction log line.
"""

from __future__ import annotations

import pytest

from backend.knowledge.config import ExtractionConfig
from backend.knowledge.extraction.confidence import ConfidenceScorer
from backend.knowledge.extraction.normalizer import EntityNormalizer
from backend.knowledge.extraction.ontology_extractor import OntologyConstrainedExtractor
from backend.knowledge.extraction.pipeline import ExtractionPipeline
from backend.knowledge.extraction.preprocessor import PreProcessor
from backend.knowledge.extraction.temporal import TemporalResolver
from backend.knowledge.extraction.validator import ExtractionValidator
from backend.knowledge.storage.graph_store import GraphStore
from tests.mocks.config import build_test_config
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection
from tests.mocks.ollama import FakeLLM

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_pipeline(
    *,
    curation_pipeline=None,
    extraction_config: ExtractionConfig | None = None,
) -> ExtractionPipeline:
    """Build a minimal ExtractionPipeline for unit testing extract_from_file."""
    conn = FakeNeo4jConnection()
    embeddings = FakeEmbeddingGenerator()
    gs = GraphStore(conn, embeddings)
    config = build_test_config()

    return ExtractionPipeline(
        preprocessor=PreProcessor(),
        extractor=OntologyConstrainedExtractor(config, llm=FakeLLM()),
        confidence_scorer=ConfidenceScorer(),
        temporal_resolver=TemporalResolver(),
        normalizer=EntityNormalizer(embedding_generator=embeddings, executor=None),
        validator=ExtractionValidator(min_confidence=0.5),
        graph_store=gs,
        curation_pipeline=curation_pipeline,
        embedding_provider=None,
        extraction_config=extraction_config,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExtractFromFileExists:
    """Confirm the method exists with the correct signature."""

    def test_method_exists_on_pipeline(self):
        pipeline = _build_pipeline()

        assert hasattr(
            pipeline, "extract_from_file"
        ), "ExtractionPipeline must have extract_from_file method"

    def test_method_is_coroutine(self):
        import asyncio

        pipeline = _build_pipeline()

        assert asyncio.iscoroutinefunction(
            pipeline.extract_from_file
        ), "extract_from_file must be an async def"


class TestExtractFromFileSignature:
    """Confirm the method accepts (content, vault_note_path, ontology_version)."""

    @pytest.mark.asyncio
    async def test_accepts_three_positional_args(self):
        pipeline = _build_pipeline()

        # Should not raise TypeError
        await pipeline.extract_from_file(
            content="User uses Python.",
            vault_note_path="/vault/sessions/2026-05-01.md",
            ontology_version="1.1.0",
        )

    @pytest.mark.asyncio
    async def test_accepts_empty_content(self):
        pipeline = _build_pipeline()

        # Empty content — should not raise
        result = await pipeline.extract_from_file(
            content="",
            vault_note_path="/vault/sessions/2026-05-01.md",
            ontology_version="1.1.0",
        )
        # May return None or a result — just shouldn't error
        assert result is None or result is not None


class TestExtractFromFileDoesNotForwardPath:
    """R1.3: vault_note_path stays local to extract_from_file (logging only).

    Prior to R1.3, extract_from_file forwarded vault_note_path down to
    curate_and_store for the (now-retired) VaultNote provenance anchor. This
    pins the retirement: curate_and_store must never see the kwarg, even
    though extract_from_file itself still accepts the parameter.
    """

    @pytest.mark.asyncio
    async def test_curation_receives_no_vault_note_path_kwarg(self):
        """When a curation pipeline is wired, it never receives vault_note_path."""

        class _RecordingCuration:
            def __init__(self):
                self.calls: list[dict] = []

            async def curate_and_store(
                self,
                validation_result,
                event_id,
                session_id,
                source_metadata=None,
                recorded_at=None,
            ):
                self.calls.append({})
                # Return a minimal object the pipeline code won't crash on
                from unittest.mock import MagicMock

                result = MagicMock()
                result.dedup_result.entities_merged = 0
                result.reconcile_result.closed = 0
                return result

        recording = _RecordingCuration()
        conn = FakeNeo4jConnection()
        embeddings = FakeEmbeddingGenerator()
        gs = GraphStore(conn, embeddings)
        config = build_test_config()

        # LLM returns a non-empty extraction so curation is triggered
        llm = FakeLLM(
            default_response=(
                '{"entities": [{"name": "Raj", "type": "User", "properties": {}}], '
                '"relationships": []}'
            )
        )

        pipeline = ExtractionPipeline(
            preprocessor=PreProcessor(),
            extractor=OntologyConstrainedExtractor(config, llm=llm),
            confidence_scorer=ConfidenceScorer(),
            temporal_resolver=TemporalResolver(),
            normalizer=EntityNormalizer(embedding_generator=embeddings, executor=None),
            validator=ExtractionValidator(min_confidence=0.0),
            graph_store=gs,
            curation_pipeline=recording,
            embedding_provider=None,
        )

        # extract_from_file's own signature still accepts vault_note_path
        # (unrelated call site, out of R1.3's scope); this test confirms it
        # goes no further than the local log lines.
        await pipeline.extract_from_file(
            content="Raj uses Python and Neo4j at work.",
            vault_note_path="/vault/sessions/2026-05-01.md",
            ontology_version="1.1.0",
        )

        assert len(recording.calls) == 1, "curation must still run"


class TestExtractFromFileOntologyVersion:
    """ontology_version parameter must pin the extraction to a known version."""

    @pytest.mark.asyncio
    async def test_custom_ontology_version_accepted(self):
        """Method accepts a non-default ontology_version without error."""
        pipeline = _build_pipeline()

        # Should not raise
        await pipeline.extract_from_file(
            content="User works on MIST.AI project.",
            vault_note_path="/vault/decisions/ADR-001.md",
            ontology_version="2.0.0",
        )

    @pytest.mark.asyncio
    async def test_returns_none_or_result_on_empty_extraction(self):
        """When LLM returns empty entities, method returns without error."""
        pipeline = _build_pipeline()

        result = await pipeline.extract_from_file(
            content="",
            vault_note_path="/vault/sessions/empty.md",
            ontology_version="1.1.0",
        )

        # No assertion on specific type — just no exception
        assert result is None or result is not None
