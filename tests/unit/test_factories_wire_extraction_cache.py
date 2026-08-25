"""Task 5 wiring, proved without touching Neo4j, the LLM, or an embedding model.

This is the commit that makes the extraction cache live: before it, Tasks 1-4
built a cache, a purity diagnostic, a gate, and five write sites that write
nowhere in production because no factory has ever passed an ExtractionCache
into ExtractionPipeline. `build_extraction_pipeline` is the sole composition
root for the pipeline (`backend/server.py`'s `build_conversation_handler` ->
`build_extraction_pipeline`), so this test targets that function directly.

Behavioural, not AST: HIGH-1 of the 2026-08-05 review defeated three static
AST checks (grepping for a call, an import, a symbol name) with two ordinary
edits -- moving a call one frame out, and an aliased import. AST checks a
refactor slides past; this test builds the real pipeline object and reads its
attributes, so it only passes if the wiring actually happened.
"""

from backend.knowledge.extraction_cache import ExtractionCache


class _RefusingConnection:
    """Stands in for Neo4jConnection. Fails loudly if the builder queries anything."""

    def execute_query(self, *args, **kwargs):
        raise AssertionError("the wiring test must not read a graph")

    def execute_write(self, *args, **kwargs):
        raise AssertionError("the wiring test must not write a graph")


class _FakeGraphStore:
    r"""Exactly the two attributes `build_extraction_pipeline` reads off `gs`.

    Verified via `grep -n "gs\." backend/factories.py`: inside
    `build_extraction_pipeline` the hits are `gs.connection`,
    `gs.embedding_generator` (three times), and `gs.ensure_mist_identity()`.
    This test passes `include_internal_derivation=False` so the last of those
    is never called -- see the docstring on the test below for why that
    matters.
    """

    def __init__(self):
        self.connection = _RefusingConnection()
        self.embedding_generator = None


class _FakeLLM:
    """Never called. Present only so no real provider is constructed."""


def test_live_pipeline_is_built_with_a_real_extraction_cache(tmp_path):
    """Behavioural, not AST: build the real object and look at what it holds.

    Mutant this kills: deleting (or no-oping) the `extraction_cache=` /
    `rebuild_stamps=` arguments on the `ExtractionPipeline(...)` construction
    in `build_extraction_pipeline`. With either argument dropped, `_extraction_cache`
    on the built pipeline stays `None` (Task 3's default) and the first
    assertion fails. A mutant that passes a *fresh, uninitialised*
    `ExtractionCache()` instead of one that ran `.initialize()` is also
    caught -- the PRAGMA assertion below reads the actual sqlite schema, not
    a mock.

    Graph-safety mechanism (why this cannot reach `build_graph_store`):
    `build_extraction_pipeline` only ever opens a real `GraphStore` /
    `Neo4jConnection` / `SentenceTransformer` when its `graph_store` argument
    is falsy (`gs = graph_store or build_graph_store(config)`) or when
    `include_internal_derivation` is left at its `True` default (the branch
    that calls `gs.ensure_mist_identity()`, which issues a live `MERGE`).
    This test supplies a truthy `graph_store` (`_FakeGraphStore`, so the `or`
    short-circuits before `build_graph_store` is even referenced) AND passes
    `include_internal_derivation=False` (so the `ensure_mist_identity` branch
    is never entered) -- either one alone would already prevent the graph
    call; both together mean there is no code path left in this function that
    reaches Neo4j. `_FakeGraphStore.connection` is `_RefusingConnection`,
    which raises `AssertionError` on any query or write, as a second,
    independent tripwire in case a future edit routes a call through
    `gs.connection` directly instead of through `build_graph_store`.
    """
    from backend.factories import build_extraction_pipeline
    from tests.mocks.config import build_test_config

    # tests/CLAUDE.md: "Config factory: build_test_config() -- never
    # KnowledgeConfig.from_env()". event_store_db_path is a first-class
    # keyword on build_test_config, so the cache path (derived from it, the
    # same convention `_build_log_regenerator` in scripts/mist_admin.py uses:
    # `grep -nE 'event_store_path = |cache_path = ' scripts/mist_admin.py`)
    # needs no monkeypatch.
    config = build_test_config(event_store_db_path=str(tmp_path / "event_store.db"))

    pipeline = build_extraction_pipeline(
        config,
        graph_store=_FakeGraphStore(),
        llm_provider=_FakeLLM(),
        include_curation=False,
        include_internal_derivation=False,
    )

    assert isinstance(pipeline._extraction_cache, ExtractionCache)
    # The cache sits beside the event store -- one convention, not two.
    assert pipeline._extraction_cache.db_path == str(tmp_path / "extraction_cache.db")
    # initialize() must have run -- the table exists.
    cols = {
        row[1]
        for row in pipeline._extraction_cache._get_connection().execute(
            "PRAGMA table_info(extraction_cache)"
        )
    }
    assert "outcome" in cols


def test_live_pipeline_rebuild_stamps_match_config(tmp_path):
    """The `RebuildStamps` injected into the pipeline must come from the SAME
    construction site's config values, not a second, independently-derived set.

    Mutant this kills: hand-writing a second `RebuildStamps(...)` (or
    hardcoding a stamp field) instead of deriving all three fields from
    `config.ontology_version` / `config.extraction_version` /
    `compose_model_hash(config)`, mirroring `build_curation_pipeline`'s
    construction exactly (`grep -n "rebuild_stamps = RebuildStamps("
    backend/factories.py` -- confirmed one hit before this task, a second
    after). Review finding L4 (2026-08-02) was two divergent construction
    sites disagreeing on 2 of 3 fields, turning every rebuild into a
    permanent ColdCacheError; this test pins the values so a future edit that
    reintroduces that drift fails here rather than at rebuild time.
    """
    from backend.factories import build_extraction_pipeline
    from backend.knowledge.version_stamps import compose_model_hash
    from tests.mocks.config import build_test_config

    config = build_test_config(event_store_db_path=str(tmp_path / "event_store.db"))
    config.extraction_version = "wiring-test-extraction-version"
    config.ontology_version = "wiring-test-ontology-version"

    pipeline = build_extraction_pipeline(
        config,
        graph_store=_FakeGraphStore(),
        llm_provider=_FakeLLM(),
        include_curation=False,
        include_internal_derivation=False,
    )

    stamps = pipeline._rebuild_stamps
    assert stamps.ontology_version == "wiring-test-ontology-version"
    assert stamps.extraction_version == "wiring-test-extraction-version"
    assert stamps.model_hash == compose_model_hash(config)
