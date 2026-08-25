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


class TestCrossFactoryStampAgreement:
    """`RebuildStamps` is constructed at two sites in `backend/factories.py`
    (`grep -n "rebuild_stamps = RebuildStamps(" backend/factories.py` -> two
    hits): once inside `build_curation_pipeline` (above), once inside
    `build_extraction_pipeline`. With `include_curation=True` (the default),
    a single call to `build_extraction_pipeline` constructs BOTH -- so this
    is not two independent call graphs, it is one production call
    constructing the stamps object twice.

    Task 5 review Important-2: `test_factory_constructs_stamps_from_config`
    above and Task 5's own wiring test
    (`test_live_pipeline_rebuild_stamps_match_config`,
    tests/unit/test_factories_wire_extraction_cache.py) each pin three NAMED
    fields against config, independently, per factory. Both would keep
    passing if a fourth field were added to `RebuildStamps` and wired into
    only ONE of the two construction sites -- the exact "two sites
    disagreeing on N of M fields" shape review finding L4 (2026-08-02) was,
    reachable today despite both per-field pins being green. `RebuildStamps`
    is a frozen dataclass with generated field-wise `__eq__`
    (`test_rebuild_stamps_dataclass_is_frozen` above pins `frozen=True`), so
    comparing the two objects for equality -- rather than field by field --
    covers a field added to the dataclass later automatically, without this
    test needing to be edited when it grows.
    """

    def test_extraction_and_curation_stamps_are_identical(self) -> None:
        """Mutant this kills: editing ONE of the two `RebuildStamps(...)`
        construction sites in `backend/factories.py` (e.g. swapping which
        config field feeds `model_hash`, or hardcoding a value) while leaving
        the other unchanged. Verified: changed `build_extraction_pipeline`'s
        `rebuild_stamps = RebuildStamps(...)` call to pass
        `ontology_version="mutated"` unconditionally instead of
        `config.ontology_version`, re-ran this test -- FAILED with
        `extraction_stamps.ontology_version == 'mutated'` against
        `curation_stamps.ontology_version` carrying the real config value, an
        inequality neither existing per-field-pin test can see (each only
        checks its own factory's output against config, never the other
        factory's output). Reverted via Edit, re-ran -- PASSED.
        """
        from backend.factories import build_extraction_pipeline
        from tests.mocks.config import build_test_config
        from tests.mocks.embeddings import FakeEmbeddingGenerator
        from tests.mocks.neo4j import FakeNeo4jConnection
        from tests.mocks.ollama import FakeLLM

        class _FakeGraphStore:
            """Both attributes `build_extraction_pipeline` reads off `gs`.

            `embedding_generator` is a real `FakeEmbeddingGenerator` (not
            `None`) so `build_curation_pipeline`'s
            `if embedding_provider is None: embedding_provider =
            EmbeddingGenerator(...)` fallback never fires -- this test needs
            no real SentenceTransformer and no `@requires_sentence_transformers`
            marker.
            """

            def __init__(self) -> None:
                self.connection = FakeNeo4jConnection()
                self.embedding_generator = FakeEmbeddingGenerator()

        config = build_test_config()
        gs = _FakeGraphStore()

        # include_curation=True (the default) is the point of this test --
        # it is what makes build_extraction_pipeline construct BOTH stamps
        # objects in one call. include_internal_derivation=False keeps this
        # graph-safe: build_extraction_pipeline only ever opens a real
        # GraphStore/Neo4jConnection when `graph_store` is falsy (it is not,
        # here -- `gs` is truthy so `gs = graph_store or build_graph_store(config)`
        # short-circuits) or when include_internal_derivation's default
        # branch calls `gs.ensure_mist_identity()` (turned off here).
        pipeline = build_extraction_pipeline(
            config,
            graph_store=gs,
            llm_provider=FakeLLM(),
            include_curation=True,
            include_internal_derivation=False,
        )

        extraction_stamps = pipeline._rebuild_stamps  # type: ignore[attr-defined]
        curation_stamps = (
            pipeline._curation_pipeline._graph_writer._rebuild_stamps  # type: ignore[attr-defined]
        )

        assert isinstance(extraction_stamps, RebuildStamps)
        assert isinstance(curation_stamps, RebuildStamps)
        assert extraction_stamps == curation_stamps
        # Positive proof this exercised both construction sites rather than
        # comparing a stamps object against itself or two Nones.
        assert extraction_stamps is not curation_stamps
        # Graph-safety: no write occurred through either fake connection.
        gs.connection.assert_no_writes()
