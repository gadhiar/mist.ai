"""RebuildStamps construction contract: config -> CurationGraphWriter wiring.

R1.3 retired the DERIVED_FROM->VaultNote edge (see
tests/unit/knowledge/curation/test_graph_writer_utterance_provenance.py) and,
with it, deleted test_graph_writer_vault_provenance.py. That deletion was
justified by claiming `TestPhase8FactoryWiring` duplicated coverage already in
test_factories_phase3.py -- that claim was false (test_factories_phase3.py
never references RebuildStamps or `_rebuild_stamps`), so three contracts were
lost with no replacement. These tests port them forward unchanged in
substance.

R1.3 also increases RebuildStamps' blast radius: it now stamps every
EXTRACTED_FROM edge instead of only DERIVED_FROM->VaultNote, so its
construction contract -- `build_curation_pipeline` wiring config values into a
frozen `RebuildStamps`, and `model_hash` folding in the embedding-model
identity (R1.1e) -- must stay guarded.

Dependency note: `build_curation_pipeline` imports `backend.factories`, which
eagerly imports `EmbeddingGenerator` and therefore `sentence_transformers`
(Linux/container only). Tests that call it are marked
`@requires_sentence_transformers` and import `backend.factories` lazily
inside the test body, matching the convention in test_factories_phase3.py.
"""

import pytest

from backend.knowledge.curation.graph_writer import RebuildStamps

_SENTENCE_TRANSFORMERS_AVAILABLE = False
try:
    import sentence_transformers as _st  # noqa: F401

    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

requires_sentence_transformers = pytest.mark.skipif(
    not _SENTENCE_TRANSFORMERS_AVAILABLE,
    reason="sentence_transformers not available on this platform",
)


class TestRebuildStampsDataclass:
    def test_rebuild_stamps_dataclass_is_frozen(self) -> None:
        # Frozen so the stamps cannot drift mid-process; rebuild determinism
        # depends on a stable per-deployment value set.
        from dataclasses import FrozenInstanceError

        stamps = RebuildStamps(
            ontology_version="1.0.0",
            extraction_version="2026-04-17-r1",
            model_hash="gemma-4-e4b",
        )

        with pytest.raises(FrozenInstanceError):
            stamps.ontology_version = "2.0.0"  # type: ignore[misc]


class TestBuildCurationPipelineWiring:
    """Verifies build_curation_pipeline constructs RebuildStamps from config."""

    @requires_sentence_transformers
    def test_factory_constructs_stamps_from_config(self) -> None:
        from backend.factories import build_curation_pipeline
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
        assert stamps.model_hash == "factory-test-model|emb:test-model"

    @requires_sentence_transformers
    def test_rebuild_stamps_model_hash_includes_embedding_model_identity(self) -> None:
        """model_hash must embed the embedding-model identity so an embedding-model
        swap invalidates the extraction cache and re-baselines the epoch.

        Task R1.1e: cosine comparisons in the deterministic resolver use the stored
        embedding vectors; a different embedding model produces different vectors so
        a near-0.92 merge can flip -> a different graph. Folding the embedding model
        into model_hash makes the swap a new epoch, preventing a silent cross-epoch
        determinism break.
        """
        from backend.factories import build_curation_pipeline
        from tests.mocks.config import build_test_config
        from tests.mocks.neo4j import FakeGraphExecutor, FakeNeo4jConnection

        # Arrange -- use distinct LLM hash and embedding model so the assertion
        # cannot accidentally pass if only one of the two is included.
        cfg = build_test_config(embedding_model="all-MiniLM-L6-v2-custom")
        cfg.model_hash = "gemma-test-hash"

        executor = FakeGraphExecutor(connection=FakeNeo4jConnection())

        # Act
        pipeline = build_curation_pipeline(cfg, executor)

        # Assert -- model_hash must contain BOTH the LLM hash and the embedding model name.
        stamps = pipeline._graph_writer._rebuild_stamps  # type: ignore[attr-defined]
        assert isinstance(stamps, RebuildStamps)
        assert cfg.model_hash in stamps.model_hash
        assert cfg.embedding.model_name in stamps.model_hash
