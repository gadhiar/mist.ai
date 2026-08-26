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

import pytest

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
    in `build_extraction_pipeline`. Three distinct drops, each verified
    directly against running source:
      - Drop only `extraction_cache=`: Task 3's pairing guard raises
        `ValueError: extraction_cache is required when rebuild_stamps is
        provided` from inside `ExtractionPipeline.__init__` -- the call in
        `build_extraction_pipeline` itself raises, so the test fails on an
        uncaught `ValueError`, never reaching the `isinstance` line.
      - Drop only `rebuild_stamps=`: the guard's other branch raises
        `ValueError: rebuild_stamps is required when extraction_cache is
        provided` -- same failure mode, opposite message.
      - Drop both: both-None is the pairing guard's legal default (no
        `ValueError`), so `_extraction_cache` on the built pipeline stays
        `None` (Task 3's default) and the `isinstance` assertion below fails
        with a plain `AssertionError`.
    All three are caught by this test, by three different exceptions -- only
    the "drop both" case reaches the assertion the docstring below names. A
    mutant that passes a *fresh, uninitialised* `ExtractionCache()` instead of
    one that ran `.initialize()` is also caught -- the PRAGMA assertion below
    reads the actual sqlite schema, not a mock.

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


def test_live_pipeline_cache_stays_in_memory_when_event_store_is_in_memory():
    """The ":memory:" sentinel must propagate from the event store path into
    the derived cache path, not get silently dropped.

    `build_test_config()`'s default `event_store_db_path` is ":memory:"
    (`tests/mocks/config.py:41`), which is truthy, so `event_store_path or
    ...` never falls back to the `~/.mist/...` default -- it is
    ":memory:" verbatim. `Path(":memory:").parent` is a relative "." with no
    meaningful parent, so deriving the cache path with the same
    `.parent / "extraction_cache.db"` join used for a real path would
    silently produce the bare relative path "extraction_cache.db" -- an
    on-disk file in the process CWD -- for a caller whose event store is
    explicitly in-memory (no rebuildability intended).

    Mutant this kills: reverting the cache-path derivation to the
    unconditional `str(_Path(event_store_path).parent /
    "extraction_cache.db")` (this task's original form, without the
    ":memory:" special case). Verified: with that reverted, this test failed
    with `assert 'extraction_cache.db' == ':memory:'` -- the previous
    behaviour, confirmed directly rather than assumed.
    """
    from backend.factories import build_extraction_pipeline
    from tests.mocks.config import build_test_config

    config = build_test_config()
    assert config.event_store.db_path == ":memory:"

    pipeline = build_extraction_pipeline(
        config,
        graph_store=_FakeGraphStore(),
        llm_provider=_FakeLLM(),
        include_curation=False,
        include_internal_derivation=False,
    )

    assert pipeline._extraction_cache.db_path == ":memory:"


def test_live_pipeline_degrades_gracefully_when_cache_initialization_fails(tmp_path):
    """The cache's `initialize()` -- `mkdir` + `sqlite3.connect` -- is I/O that can
    fail on an unwritable or absent data root (I1, whole-branch review). Before
    this fix, that failure propagated straight out of `build_extraction_pipeline`,
    through `build_conversation_handler`'s unguarded call site, into
    `KnowledgeIntegration.__init__`'s outer `except Exception` -- which logs
    "Knowledge integration disabled" and silently degrades MIST to a plain LLM:
    no graph, no retrieval, no extraction, no vault. This test proves the
    pipeline itself now absorbs the failure and comes up in the SAME degraded
    mode `ExtractionPipeline` already supports on purpose: `extraction_cache`
    and `rebuild_stamps` both `None`, which `_record_skip` / `_record_extraction`
    already no-op on (`grep -n "if self._extraction_cache is None" \
    backend/knowledge/extraction/pipeline.py`).

    Mutant this kills: deleting the `try/except (sqlite3.Error, OSError)` around
    the cache construction in `build_extraction_pipeline` (or narrowing it to a
    non-matching exception type). With the guard removed, this test fails with
    the raw `OSError` subclass propagating out of `build_extraction_pipeline`
    instead of a pipeline being returned; with the guard restored, it fails
    instead (correctly) if `extraction_cache` or `rebuild_stamps` comes back
    non-None, since the pairing guard in `ExtractionPipeline.__init__` would
    then have rejected a mismatched pair before construction even completed.

    Forces the failure deterministically rather than relying on OS permission
    bits (unreliable across platforms and CI users): the event store's parent
    directory is a FILE, not a directory, so `Path(cache_path).parent.mkdir(
    parents=True, exist_ok=True)` inside `ExtractionCache.initialize()` raises
    an `OSError` subclass (`NotADirectoryError` on POSIX, `FileExistsError` on
    Windows) when it tries to create a directory where a file already sits.
    Verified directly against `ExtractionCache` below, before trusting the
    factory to hit the same branch through `production_cache_path`.
    """
    from backend.factories import build_extraction_pipeline
    from tests.mocks.config import build_test_config

    blocker = tmp_path / "blocked-by-a-file"
    blocker.write_text("not a directory")
    event_store_path = str(blocker / "subdir" / "event_store.db")
    cache_path = str(blocker / "subdir" / "extraction_cache.db")

    with pytest.raises(OSError):
        ExtractionCache(cache_path).initialize()

    config = build_test_config(event_store_db_path=event_store_path)

    pipeline = build_extraction_pipeline(
        config,
        graph_store=_FakeGraphStore(),
        llm_provider=_FakeLLM(),
        include_curation=False,
        include_internal_derivation=False,
    )

    assert pipeline._extraction_cache is None
    assert pipeline._rebuild_stamps is None
