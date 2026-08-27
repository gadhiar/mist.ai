"""Composition root for MIST.AI backend.

All dependency wiring lives here. Classes accept required constructor
params -- this module provides the factory functions that know how to
assemble them with real implementations.

Usage:
    from backend.factories import build_graph_store
    graph_store = build_graph_store(config)

For tests, bypass factories and pass fakes directly to constructors.
"""

import logging
import os
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from backend.interfaces import (
    EmbeddingProvider,
    EventStoreProvider,
    GraphConnection,
    SidecarIndexProtocol,
)

if TYPE_CHECKING:
    from backend.vault import VaultFilewatcher, VaultWriter
    from backend.vault.invalidation_bus import InvalidationBus
    from backend.vault.sidecar_index import VaultSidecarIndex
from backend.knowledge.config import KnowledgeConfig
from backend.knowledge.curation.confidence import ConfidenceManager
from backend.knowledge.curation.deduplication import EntityDeduplicator
from backend.knowledge.curation.graph_writer import CurationGraphWriter
from backend.knowledge.curation.pipeline import CurationPipeline
from backend.knowledge.curation.reconciliation import ReconciliationEngine
from backend.knowledge.extraction.confidence import ConfidenceScorer
from backend.knowledge.extraction.normalizer import EntityNormalizer
from backend.knowledge.extraction.ontology_extractor import OntologyConstrainedExtractor
from backend.knowledge.extraction.pipeline import ExtractionPipeline
from backend.knowledge.extraction.preprocessor import PreProcessor
from backend.knowledge.extraction.scope_classifier import SubjectScopeClassifier
from backend.knowledge.extraction.temporal import TemporalResolver
from backend.knowledge.extraction.validator import ExtractionValidator
from backend.knowledge.storage.graph_executor import GraphExecutor
from backend.knowledge.storage.graph_store import GraphStore
from backend.knowledge.storage.neo4j_connection import Neo4jConnection
from backend.llm import StreamingLLMProvider

logger = logging.getLogger(__name__)

# Replay-determinism clock seam (config-from-env, mirroring LLM_TEMPERATURE et
# al. in KnowledgeConfig.from_env). PRODUCTION leaves MIST_FIXED_CLOCK unset ->
# every clock falls back to wall-clock (behavior unchanged). The F2 replay path
# sets it to a single fixed ISO-8601 instant shared by det1 and det2 so the
# user-snapshot `rendered_at` stamped into the chat system prompt is reproducible
# (a wall-clock value perturbs the turn-1 greedy reply and diverges the
# conversation history). See scripts/eval_harness/extraction_probe_set_design.md.
_FIXED_CLOCK_ENV = "MIST_FIXED_CLOCK"


def resolve_fixed_rendered_at() -> str | None:
    """Return the replay-pinned ISO timestamp, or None for wall-clock.

    Reads `MIST_FIXED_CLOCK`. Unset (the production default) yields None, which
    every caller maps to live `datetime.now(UTC)`. When set, the value is
    validated as ISO 8601 and returned verbatim so it round-trips byte-for-byte
    into the seeded `users/<id>.md` Provenance block.

    Raises:
        ValueError: when the env var is set but not parseable as ISO 8601. A
            malformed pin must fail loudly rather than silently degrade an eval
            run to nondeterministic wall-clock.
    """
    raw = os.getenv(_FIXED_CLOCK_ENV)
    if raw is None or raw.strip() == "":
        return None
    value = raw.strip()
    # Validate (and normalize-check) -- raises ValueError on a bad pin.
    datetime.fromisoformat(value)
    return value


_HYDRATION_CLOCK_ENV = "MIST_HYDRATION_CLOCK"


def build_hydration_clock():
    """Build the B2 authored-timestamp clock, or None for every normal process.

    Reads MIST_HYDRATION_CLOCK, a path to a JSONL corpus carrying `session_id`,
    `turn_index` and `timestamp` per line -- which the golden log already does,
    so no new fixture format is involved.

    REFUSES unless MIST_HYDRATION_ISOLATION is also set. A keyed clock silently
    rewriting fact-time is exactly the thing that must never be reachable on
    live: every bitemporal edge a turn produces inherits `recorded_at`, so a
    clock active by accident would author a false history that both sides of
    the gate would then agree on. Requiring the isolation flag means the live
    backend cannot build one even if the path variable leaks into its
    environment.

    Returns:
        A `HydrationClock`, or None when MIST_HYDRATION_CLOCK is unset.

    Raises:
        HydrationClockError: when the path is set outside a hydration-isolated
            process, or the corpus cannot be loaded. Both fail the process
            rather than degrading to wall-clock -- a hydration run that
            silently used the wall clock would produce a green gate over a
            timeline that never existed.
    """
    from backend.chat.hydration_clock import HydrationClockError, load_hydration_clock
    from backend.knowledge.eval_isolation import is_hydration_isolation_active

    raw = os.getenv(_HYDRATION_CLOCK_ENV)
    if raw is None or raw.strip() == "":
        return None
    if not is_hydration_isolation_active():
        raise HydrationClockError(
            f"{_HYDRATION_CLOCK_ENV} is set but MIST_HYDRATION_ISOLATION is not. A "
            "keyed clock rewrites `recorded_at`, the fact-time authority for every "
            "bitemporal edge a turn produces, so it must be unreachable outside a "
            "hydration-isolated process. Refusing to build one."
        )
    return load_hydration_clock(raw.strip())


def build_now_fn() -> Callable[[], datetime]:
    """Build the injectable clock for ConversationHandler.

    Production (no `MIST_FIXED_CLOCK`): returns a wall-clock callable
    (`lambda: datetime.now(UTC)`). Replay (env set): returns a callable that
    yields the SAME fixed tz-aware instant on every call, so det1 and det2 share
    one constant and the user-snapshot timestamp is reproducible.
    """
    fixed = resolve_fixed_rendered_at()
    if fixed is None:
        return lambda: datetime.now(UTC)
    fixed_dt = datetime.fromisoformat(fixed)
    return lambda: fixed_dt


def build_vector_store(config: KnowledgeConfig) -> "LanceDBVectorStore":  # noqa: F821
    """Create and connect a LanceDBVectorStore.

    Args:
        config: Knowledge subsystem configuration.

    Returns:
        Connected LanceDBVectorStore ready for use.
    """
    from backend.knowledge.storage.vector_store import LanceDBVectorStore

    store = LanceDBVectorStore(config.vector_store)
    store.connect()
    return store


def build_neo4j_connection(config: KnowledgeConfig) -> Neo4jConnection:
    """Create and connect a Neo4jConnection."""
    conn = Neo4jConnection(config.neo4j)
    conn.connect()
    return conn


def build_graph_executor(
    config: KnowledgeConfig, connection: GraphConnection | None = None
) -> GraphExecutor:
    """Create a GraphExecutor with async boundary."""
    conn = connection or build_neo4j_connection(config)
    return GraphExecutor(conn)


def build_graph_store(
    config: KnowledgeConfig,
    connection: GraphConnection | None = None,
    embedding_generator: EmbeddingProvider | None = None,
) -> GraphStore:
    """Create a GraphStore with injected dependencies."""
    conn = connection or build_neo4j_connection(config)
    if embedding_generator is None:
        from backend.knowledge.embeddings import EmbeddingGenerator

        embedding_generator = EmbeddingGenerator(config.embedding.model_name)
    return GraphStore(conn, embedding_generator)


def build_llm_provider(
    config: KnowledgeConfig,
    debug_logger: "DebugJSONLLogger | None" = None,  # noqa: F821
) -> StreamingLLMProvider:
    """Create the LLM provider based on config.

    Args:
        config: Knowledge subsystem configuration.
        debug_logger: Optional DebugJSONLLogger. When provided and the logger's
            `llm_call_enabled` gate is True, the returned provider is wrapped
            in `InstrumentedStreamingLLMProvider` so every non-partial response
            emits a `phase: "llm_call"` JSONL record. When the gate is False
            (or the logger is None) the concrete provider is returned directly
            with no wrapper overhead.

    Returns:
        StreamingLLMProvider instance (LlamaServerProvider or OllamaProvider),
        optionally wrapped by InstrumentedStreamingLLMProvider.
    """
    llm_config = config.llm
    if llm_config.backend == "llamacpp":
        from backend.llm.llama_server_provider import LlamaServerProvider

        inner: StreamingLLMProvider = LlamaServerProvider(
            base_url=llm_config.base_url,
            model=llm_config.model,
        )
    elif llm_config.backend == "ollama":
        from backend.llm.ollama_provider import OllamaProvider

        inner = OllamaProvider(
            base_url=llm_config.base_url,
            model=llm_config.model,
        )
    else:
        raise ValueError(f"Unknown LLM backend: {llm_config.backend}")

    if debug_logger is not None and debug_logger.llm_call_enabled:
        from backend.llm.instrumented_provider import InstrumentedStreamingLLMProvider

        logger.info("LLM provider wrapped with observability instrumentation")
        return InstrumentedStreamingLLMProvider(inner, debug_logger)

    return inner


def build_curation_pipeline(
    config: KnowledgeConfig,
    executor: GraphExecutor,
    debug_logger: "DebugJSONLLogger | None" = None,  # noqa: F821
    embedding_provider: "EmbeddingProvider | None" = None,  # noqa: F821
) -> CurationPipeline:
    """Create a fully wired CurationPipeline (bitemporal engine, C2).

    Pass `embedding_provider` to share an already-warmed model: a private
    EmbeddingGenerator here is never warmed by ModelManager, so the first
    turn reaching tier-3 dedup or a new-entity write lazy-loads a
    SentenceTransformer ON the event loop UNDER the curation lock.
    """
    from backend.knowledge.curation.graph_writer import RebuildStamps
    from backend.knowledge.embeddings import EmbeddingGenerator
    from backend.knowledge.version_stamps import compose_model_hash

    if embedding_provider is None:
        embedding_provider = EmbeddingGenerator(config.embedding.model_name)
    confidence_mgr = ConfidenceManager()
    # ADR-010 Phase 8 rebuild-determinism stamps. Written to every fact edge
    # (C1 4.7, reconciliation.py) and every EXTRACTED_FROM->ConversationContext
    # edge (R1.3 moved this anchor off DERIVED_FROM->VaultNote) so a future
    # consumer can detect when the ontology, extraction prompt, or model
    # binary has drifted from the values active at extraction time -- no
    # command reads them for that purpose today.
    rebuild_stamps = RebuildStamps(
        ontology_version=config.ontology_version,
        extraction_version=config.extraction_version,
        model_hash=compose_model_hash(config),
    )
    return CurationPipeline(
        deduplicator=EntityDeduplicator(executor, embedding_provider, confidence_mgr),
        reconciliation_engine=ReconciliationEngine(
            executor=executor,
            rebuild_stamps=rebuild_stamps,
            debug_logger=debug_logger,
        ),
        graph_writer=CurationGraphWriter(
            executor, embedding_provider, confidence_mgr, rebuild_stamps=rebuild_stamps
        ),
        confidence_manager=confidence_mgr,
    )


def production_cache_path(config: KnowledgeConfig) -> str:
    """The one path the LIVE extraction cache lives at.

    Named rather than inlined so the golden-log generator can refuse to equal
    it (spec D10, `scripts/golden_log/generate.py:assert_not_production_root`):
    two literals in two files is how they drift.

    `_build_log_regenerator` in scripts/mist_admin.py CALLS this function
    rather than deriving the path itself (I2, whole-branch review). Before
    that fix it re-derived the same expression inline instead -- a second
    copy of the expression this paragraph already named when it existed (see
    below), not something the docstring concealed or got wrong: the opening
    claim above is about what THIS function is FOR, and remains true on its
    own terms. What was true was UNDER-ACHIEVED while the second copy stood:
    a caller could still drift the two derivations apart, exactly the
    failure mode the golden-log generator's refuse-to-equal check exists to
    catch, just not caught for path DERIVATION itself. Propagates the
    ":memory:" sentinel: `ExtractionCache.initialize()` already special-cases
    it (`grep -n 'if self.db_path != ":memory:"' backend/knowledge/extraction_cache.py`),
    but `Path(event_store_path).parent` on the bare sentinel resolves to a
    relative "." -- silently landing an on-disk "extraction_cache.db" in the
    process CWD for any caller with an in-memory event store, rather than the
    in-memory cache that setup implies. `grep -n 'production_cache_path'
    scripts/mist_admin.py` now returns three hits -- the import, a comment
    naming this function, and the call -- zero of which are a second
    derivation of the path expression itself.
    """
    from pathlib import Path

    event_store_path = config.event_store.db_path or str(Path.home() / ".mist" / "event_store.db")
    if event_store_path == ":memory:":
        return ":memory:"
    return str(Path(event_store_path).parent / "extraction_cache.db")


def resolve_internal_derivation(explicit: bool | None) -> bool:
    """Whether Stage 9 (internal knowledge derivation) may run.

    Stage 9 runs on the LIVE path (`pipeline.py:813`) and NEVER on rebuild --
    `log_regenerator.py` has zero references to it. It MERGEs into
    `SELF_MODEL_LABEL`, so today it sits outside the `:__Entity__`-only
    comparison surface and the asymmetry is invisible. The moment MIS-131 adds
    `include_self_model=True` it becomes a live-only writer INSIDE the compared
    surface, and the gate goes RED for a reason unrelated to seed-apply.

    Hydration isolation forces it OFF and an explicit `True` cannot override
    that -- structural no beats explicit yes, the same rule
    `CurationScheduler.start()` follows (B1). A caller passing True is
    asserting intent about ingestion, not about whether the comparison surface
    stays derivable.

    Args:
        explicit: A caller's stated preference, or None for "decide from
            context". None yields True in production, so nothing about live
            changes by this function existing.
    """
    from backend.knowledge.eval_isolation import (
        EvalIsolationError,
        is_hydration_isolation_active,
    )

    try:
        isolated = is_hydration_isolation_active()
    except EvalIsolationError as exc:
        # Same rationale as `curation_scheduler_enabled`, which this call site
        # originally missed. `is_hydration_isolation_active` RAISES on an
        # unrecognized value, which is right for a CLI that can print a refusal
        # -- but this runs during `build_extraction_pipeline`, inside
        # `KnowledgeIntegration.__init__`'s broad `except Exception`. So
        # a mistyped MIST_HYDRATION_ISOLATION did not disable Stage 9,
        # it silently disabled the ENTIRE knowledge subsystem (no graph, no
        # retrieval, no extraction, no vault) with one warning line.
        #
        # Degrade toward isolated=True: Stage 9 OFF is the safe direction, and
        # an unparsable isolation flag is not evidence that we are NOT
        # hydrating.
        logger.warning(
            "MIST_HYDRATION_ISOLATION unparsable, disabling Stage 9 internal "
            "derivation as the safe default: %s",
            exc,
        )
        return False

    if isolated:
        return False
    return True if explicit is None else explicit


def build_extraction_pipeline(
    config: KnowledgeConfig,
    graph_store: GraphStore | None = None,
    llm_provider: StreamingLLMProvider | None = None,
    include_curation: bool = True,
    include_internal_derivation: bool | None = None,
    debug_logger: "DebugJSONLLogger | None" = None,  # noqa: F821
) -> ExtractionPipeline:
    """Create a fully wired ExtractionPipeline."""
    from backend.knowledge.curation.graph_writer import RebuildStamps
    from backend.knowledge.extraction_cache import ExtractionCache
    from backend.knowledge.version_stamps import compose_model_hash

    gs = graph_store or build_graph_store(config)
    executor = build_graph_executor(config, gs.connection)

    curation = (
        build_curation_pipeline(
            config,
            executor,
            debug_logger=debug_logger,
            # Share the graph store's embedding model: ModelManager warms
            # exactly that instance at startup (off-loop), so curation never
            # cold-loads a second SentenceTransformer under the write lock.
            embedding_provider=gs.embedding_generator,
        )
        if include_curation
        else None
    )

    provider = llm_provider or build_llm_provider(config)

    internal_deriver = None
    if resolve_internal_derivation(include_internal_derivation):
        from backend.knowledge.extraction.internal_derivation import InternalKnowledgeDeriver

        internal_deriver = InternalKnowledgeDeriver(
            llm=provider, executor=executor, temperature=config.llm.temperature
        )
        # Ensure MistIdentity singleton exists (sync call, OK in factory context)
        gs.ensure_mist_identity()

    # Stage 1.5: subject-scope classifier (Cluster 1). Only built when the
    # feature flag is enabled in config. When disabled, pipeline skips
    # Stage 1.5 entirely and Stage 2 treats scope as "unknown".
    scope_classifier: SubjectScopeClassifier | None = None
    if config.scope_classifier.enabled:
        scope_classifier = SubjectScopeClassifier(
            llm=provider,
            config=config.scope_classifier,
        )

    # F3 (extraction-cache-phase-1) Task 5: cache sits beside the event store
    # -- one path convention, not two. Task 7 pulled the derivation out to
    # `production_cache_path` above so the golden-log generator can refuse to
    # equal it (spec D10) without duplicating this expression a third time.
    #
    # I1 (whole-branch review): construction is I/O -- `initialize()` creates
    # the parent directory and opens a sqlite3 connection, and both raise on
    # an unwritable or absent data root. This call sits at the composition
    # root every production path runs through (`build_conversation_handler`
    # -> `build_extraction_pipeline`), and an uncaught exception here
    # propagates up into `KnowledgeIntegration.__init__`'s outer
    # `except Exception`, which silently degrades MIST to a plain LLM -- no
    # graph, no retrieval, no extraction, no vault -- reported as one
    # WARNING line. `sqlite3.Error` and `OSError` are the same pair
    # `_record_skip` / `_record_extraction` already narrow to (`grep -n
    # "except (sqlite3.Error, OSError)" backend/knowledge/extraction/pipeline.py`),
    # and this call is failure-isolated the same way the vector store below
    # already is (`grep -n "vector_store = None" backend/factories.py`) --
    # the new code was the exception to a pattern its own neighbours follow.
    try:
        extraction_cache = ExtractionCache(production_cache_path(config))
        extraction_cache.initialize()
    except (sqlite3.Error, OSError) as exc:
        # This runs once, at composition-root construction time -- there is
        # no single "turn" to blame it on. The degradation is PROCESS-WIDE:
        # every turn this process extracts for the rest of its lifetime will
        # be unrebuildable, not just whichever turn happens to trigger the
        # next factory build. Extraction and the conversation itself proceed
        # unaffected; only rebuildability is lost.
        logger.warning(
            "Extraction cache unavailable at %s -- this process will run "
            "without one: every turn it extracts will be unrebuildable for "
            "the rest of this process's lifetime, though extraction and the "
            "conversation itself are unaffected: %s",
            production_cache_path(config),
            exc,
        )
        extraction_cache = None

    # Constructed here, from the same KnowledgeConfig that
    # build_curation_pipeline constructs its own RebuildStamps from -- both
    # real construction sites, and only those two, are found by `grep -nE
    # "^\s+(rebuild_stamps = )?RebuildStamps\(" backend/factories.py`
    # (build_curation_pipeline's assignment, and this function's own
    # RebuildStamps(...) call below; the plain `rebuild_stamps =
    # RebuildStamps(` substring this replaced also matched THIS comment once
    # the second site became a conditional expression in I1's fix, which is
    # why the pattern is anchored to line-start rather than substring-matched
    # -- verify with the command above before trusting the count again). NOT
    # one construction site per process: with
    # include_curation=True (the default), this function also calls
    # build_curation_pipeline above, so a single production call constructs
    # BOTH stamps objects in one process. Both sites derive all three fields
    # from this same config (never the bare config.model_hash), and
    # tests/unit/test_factories_rebuild_stamps.py::TestCrossFactoryStampAgreement
    # pins the two outputs equal so a future edit cannot silently diverge
    # them. Review finding L4 (2026-08-02) was two sites disagreeing on 2 of
    # 3 fields, which made every rebuild a permanent ColdCacheError rather
    # than a mislabel.
    #
    # None when the cache above failed to initialize: `ExtractionPipeline`'s
    # constructor pairing guard requires `extraction_cache` and
    # `rebuild_stamps` to be both None or both set, and both-None is the
    # degraded mode that guard exists to allow, not a violation of it --
    # `_record_skip` / `_record_extraction` already no-op on it.
    rebuild_stamps = (
        RebuildStamps(
            ontology_version=config.ontology_version,
            extraction_version=config.extraction_version,
            model_hash=compose_model_hash(config),
        )
        if extraction_cache is not None
        else None
    )

    return ExtractionPipeline(
        preprocessor=PreProcessor(),
        extractor=OntologyConstrainedExtractor(config, llm=provider),
        confidence_scorer=ConfidenceScorer(),
        temporal_resolver=TemporalResolver(),
        normalizer=EntityNormalizer(
            embedding_generator=gs.embedding_generator,
            executor=executor,
        ),
        validator=ExtractionValidator(
            min_confidence=config.extraction.min_extraction_confidence,
        ),
        graph_store=gs,
        curation_pipeline=curation,
        internal_deriver=internal_deriver,
        embedding_provider=gs.embedding_generator,
        extraction_config=config.extraction,
        scope_classifier=scope_classifier,
        extraction_cache=extraction_cache,
        rebuild_stamps=rebuild_stamps,
    )


def build_conversation_handler(
    config: KnowledgeConfig,
    llm_provider: StreamingLLMProvider | None = None,
    vault_writer: "VaultWriter | None" = None,
    vault_sidecar: SidecarIndexProtocol | None = None,
    invalidation_bus: "InvalidationBus | None" = None,
):
    """Create a fully wired ConversationHandler.

    Builds a hybrid retriever with optional vector store support.
    If vector store creation fails (e.g. LanceDB not available),
    the retriever falls back to graph-only behaviour.

    Observability (Cluster 5):
    - `MIST_DEBUG_JSONL=<path>` activates the base debug sink (turn + extraction).
    - `MIST_DEBUG_LLM_JSONL=1` additionally wraps the provider with
      InstrumentedStreamingLLMProvider to emit `phase: "llm_call"` records.
    - `MIST_DEBUG_RETRIEVAL_JSONL=1` activates retrieval candidate records in
      the KnowledgeRetriever.
    - `MIST_DEBUG_LLM_REQUESTS=1` activates pre-validation LLMRequest dumps in
      the ConversationHandler.

    Cluster 8 Phase 5 (vault layer):
    - When `vault_writer` is None, attempts to build one from `config.vault`.
      Returns None when `config.vault.enabled` is False.
    - Lifecycle: caller (server lifespan or test) owns start/stop.

    Cluster 8 Phase 9 (vault sidecar in retrieval):
    - `vault_sidecar` is forwarded to `build_knowledge_retriever` so the
      retriever's `historical` intent and three-way RRF merge route
      to the sidecar. Caller is responsible for sidecar.initialize()
      before this call.

    Phase 3 Task 21 (invalidation bus):
    - `invalidation_bus` must be the SAME instance wired into the filewatcher
      (from `build_phase3_components`). The handler subscribes its
      `_on_vault_rebuild` listener so vault-change events evict stale
      `_mist_context_cache` entries. When None, no subscription is registered.

    Args:
        config: Knowledge subsystem configuration.
        llm_provider: Optional pre-built LLM provider.
        vault_writer: Optional pre-built VaultWriter. When None, one is
            constructed from config. Pass an explicit None-equivalent by
            disabling config.vault.enabled.
        vault_sidecar: Optional pre-built VaultSidecarIndex. When set the
            retriever supports historical / hybrid vault retrieval; when
            None those branches degrade to graph + vector only.
        invalidation_bus: Optional InvalidationBus shared with the filewatcher.
            When provided the handler registers `_on_vault_rebuild` to receive
            vault-change events and evict stale mist_context caches. The
            bus must be the same instance returned by `build_phase3_components`.
    """
    from pathlib import Path

    from backend.chat.conversation_handler import ConversationHandler
    from backend.debug_jsonl_logger import DebugJSONLLogger
    from backend.errors import VectorStoreError
    from backend.knowledge.extraction.tool_usage_tracker import ToolUsageTracker
    from backend.vault.conventions import ConventionsLoader

    debug_logger = DebugJSONLLogger.from_env()
    if debug_logger.enabled:
        gates = []
        if debug_logger.llm_call_enabled:
            gates.append("llm_call")
        if debug_logger.retrieval_candidates_enabled:
            gates.append("retrieval_candidates")
        if debug_logger.llm_request_dump_enabled:
            gates.append("llm_request_raw")
        if debug_logger.reconciliation_enabled:
            gates.append("reconciliation")
        gate_summary = ", ".join(gates) if gates else "turn + extraction only"
        logger.info(
            "Debug JSONL logging enabled at %s (phases: %s)",
            debug_logger.path,
            gate_summary,
        )

    gs = build_graph_store(config)
    provider = llm_provider or build_llm_provider(config, debug_logger=debug_logger)
    pipeline = build_extraction_pipeline(
        config,
        graph_store=gs,
        llm_provider=provider,
        include_curation=True,
        debug_logger=debug_logger,
    )

    # Build vector store with graceful fallback
    vector_store = None
    try:
        vector_store = build_vector_store(config)
    except (VectorStoreError, Exception) as exc:
        logger.warning("Vector store unavailable, falling back to graph-only retrieval: %s", exc)

    retriever = build_knowledge_retriever(
        config=config,
        graph_store=gs,
        vector_store=vector_store,
        embedding_provider=gs.embedding_generator,
        debug_logger=debug_logger,
        vault_sidecar=vault_sidecar,
    )

    tracker = ToolUsageTracker(config.skill_derivation)

    # ADR-014: vault-root MIST.md auto-load into every turn's prompt.
    conventions_loader = ConventionsLoader(vault_root=Path(config.vault.root))

    # Cluster 8 Phase 5: vault_writer is caller-provided (or None). Auto-build
    # removed to avoid two writers racing on the same vault root -- the
    # server lifespan owns the single VaultWriter and plumbs it through
    # VoiceProcessor -> ModelManager -> KnowledgeIntegration -> here. Unit
    # tests that want wiring coverage pass an explicit writer or None.
    return ConversationHandler(
        config=config,
        graph_store=gs,
        extraction_pipeline=pipeline,
        retriever=retriever,
        llm_provider=provider,
        conventions_loader=conventions_loader,
        tool_usage_tracker=tracker,
        debug_logger=debug_logger,
        vault_writer=vault_writer,
        invalidation_bus=invalidation_bus,
        # Replay-determinism clock seam: wall-clock in production (env unset),
        # a fixed instant under MIST_FIXED_CLOCK for reproducible replays.
        now_fn=build_now_fn(),
        hydration_clock=build_hydration_clock(),
        # R1.4 Task 10: MIST_SESSION_ORIGIN (default "real") -- the eval
        # harness / CLI probes set it to "test" so their sessions are
        # excludable from an R1.6 rebuild.
        session_origin=config.event_store.session_origin,
    )


def build_curation_scheduler(
    config: KnowledgeConfig,
    event_store: EventStoreProvider | None = None,
    tracker: "ToolUsageTracker | None" = None,  # noqa: F821
    llm_provider: StreamingLLMProvider | None = None,
):
    """Create a fully wired CurationScheduler with all maintenance jobs.

    Every parameter below defaults to None so tests can build a scheduler
    without a live stack. In PRODUCTION all three must be supplied from the
    live ConversationHandler -- `backend/server.py` does this. Until
    2026-08-03 it did not, and the defaults documented here as a test
    affordance were the only behaviour the running system ever had.

    Args:
        config: Knowledge subsystem configuration.
        event_store: Event store backing SelfReflectionJob, and (D3) the
            scheduler's run recorder. When None, `SelfReflectionJob.run`
            returns zero counts on its FIRST line and no internal knowledge is
            derived from conversation history at all -- the job is inert, not
            merely idle -- AND every job's result is logged and discarded
            rather than written to `curation_job_runs`. A production None is
            legitimate only when `config.event_store.enabled` is False.
        tracker: ToolUsageTracker backing SkillDerivationJob. Must be the SAME
            instance ConversationHandler records tool calls into (its
            `_tool_usage_tracker`, wired by `build_conversation_handler`).
            When None, a fresh tracker is built here that nothing ever records
            into, so `detect_patterns()` returns nothing on every run and no
            Skill / MistCapability is ever derived. This is a wrong-instance
            hazard, not merely a missing-value one: passing SOME tracker is
            not enough, it must be that one.
        llm_provider: Provider for InternalKnowledgeDeriver. When None, a
            second provider is built here via `build_llm_provider(config)`
            with no `debug_logger`, so the deriver's LLM calls emit no
            `phase: "llm_call"` JSONL records even when the gate is on, and
            the extra `LlamaServerProvider` opens a duplicate pair of OpenAI
            clients that nothing closes. Pass the live handler's provider.
    """
    from backend.knowledge.curation.centrality import CentralityAnalyzer
    from backend.knowledge.curation.community import CommunityDetector
    from backend.knowledge.curation.confidence_decay import ConfidenceDecayJob
    from backend.knowledge.curation.embedding_maintenance import EmbeddingMaintenance
    from backend.knowledge.curation.health import GraphHealthScorer
    from backend.knowledge.curation.orphan_detector import OrphanDetector
    from backend.knowledge.curation.scheduler import CurationScheduler, JobConfig
    from backend.knowledge.curation.self_reflection import SelfReflectionJob
    from backend.knowledge.curation.skill_derivation import SkillDerivationJob
    from backend.knowledge.curation.staleness import StalenessDetector
    from backend.knowledge.embeddings import EmbeddingGenerator
    from backend.knowledge.extraction.internal_derivation import InternalKnowledgeDeriver
    from backend.knowledge.extraction.signal_detector import SignalDetector
    from backend.knowledge.extraction.tool_usage_tracker import ToolUsageTracker

    gs = build_graph_store(config)
    executor = build_graph_executor(config, gs.connection)
    embedding_provider = EmbeddingGenerator(config.embedding.model_name)

    provider = llm_provider or build_llm_provider(config)
    internal_deriver = InternalKnowledgeDeriver(llm=provider, executor=executor)

    skill_config = config.skill_derivation
    usage_tracker = tracker or ToolUsageTracker(skill_config)
    skill_job = SkillDerivationJob(
        tracker=usage_tracker,
        executor=executor,
        config=skill_config,
    )

    return CurationScheduler(
        jobs=[
            (
                JobConfig(name="confidence_decay", interval_seconds=86400),
                ConfidenceDecayJob(executor),
            ),
            (
                JobConfig(name="staleness_detection", interval_seconds=604800),
                StalenessDetector(executor),
            ),
            (JobConfig(name="orphan_detection", interval_seconds=604800), OrphanDetector(executor)),
            (JobConfig(name="health_scoring", interval_seconds=86400), GraphHealthScorer(executor)),
            (
                JobConfig(name="self_reflection", interval_seconds=86400),
                SelfReflectionJob(
                    executor=executor,
                    internal_deriver=internal_deriver,
                    signal_detector=SignalDetector(),
                    event_store=event_store,
                ),
            ),
            (
                JobConfig(name="community_detection", interval_seconds=604800, enabled=False),
                CommunityDetector(executor),
            ),
            (
                JobConfig(name="centrality_analysis", interval_seconds=604800, enabled=False),
                CentralityAnalyzer(executor),
            ),
            (
                JobConfig(name="embedding_maintenance", interval_seconds=2592000, enabled=False),
                EmbeddingMaintenance(executor, embedding_provider),
            ),
            (
                JobConfig(
                    name="skill_derivation",
                    interval_seconds=86400,
                    enabled=skill_config.enabled,
                ),
                skill_job,
            ),
        ],
        # D3: the same store, in its second role. Job results were discarded by
        # `_loop` entirely until 2026-08-03; they now land in
        # `curation_job_runs` (and health scores additionally in
        # `graph_health_events`) in the event-store SQLite DB. None here means
        # the scheduler logs results and drops them, which is why
        # `CurationScheduler.__init__` warns rather than defaulting quietly.
        run_recorder=event_store,
    )


def build_knowledge_retriever(
    config: KnowledgeConfig,
    graph_store: GraphStore | None = None,
    vector_store: "VectorStoreProvider | None" = None,  # noqa: F821
    embedding_provider: EmbeddingProvider | None = None,
    debug_logger: "DebugJSONLLogger | None" = None,  # noqa: F821
    vault_sidecar: "SidecarIndexProtocol | None" = None,  # noqa: F821
) -> "KnowledgeRetriever":  # noqa: F821
    """Create a fully wired KnowledgeRetriever with hybrid retrieval.

    Builds missing dependencies from config. When no explicit
    embedding_provider is given, reuses graph_store.embedding_generator
    so a single model instance serves both backends.

    Args:
        config: Knowledge subsystem configuration.
        graph_store: Optional pre-built graph store.
        vector_store: Optional pre-built vector store.
        embedding_provider: Optional pre-built embedding provider.
        debug_logger: Optional DebugJSONLLogger forwarded to the retriever for
            Cluster 5 `retrieval_candidates` observability.
        vault_sidecar: Optional pre-built vault sidecar index (ADR-010 Phase 9).
            When provided, the `historical` intent and the third leg of the
            `hybrid` RRF merge route to it. Caller (typically server
            lifespan) is responsible for sidecar.initialize() before this
            call. None preserves pre-Phase-9 two-way merge behavior.

    Returns:
        Ready-to-use KnowledgeRetriever instance.
    """
    from backend.knowledge.retrieval.knowledge_retriever import KnowledgeRetriever
    from backend.knowledge.retrieval.query_classifier import QueryClassifier

    gs = graph_store or build_graph_store(config)
    vs = vector_store  # None is acceptable -- retriever degrades gracefully
    ep = embedding_provider or gs.embedding_generator
    classifier = QueryClassifier(config.query_intent)

    return KnowledgeRetriever(
        config=config,
        graph_store=gs,
        vector_store=vs,
        query_classifier=classifier,
        embedding_provider=ep,
        debug_logger=debug_logger,
        vault_sidecar=vault_sidecar,
    )


def build_ingestion_pipeline(
    config: KnowledgeConfig,
    vector_store: "VectorStoreProvider | None" = None,  # noqa: F821
    embedding_provider: EmbeddingProvider | None = None,
    graph_store: "GraphStore | None" = None,
) -> "IngestionPipeline":  # noqa: F821
    """Create a fully wired IngestionPipeline.

    Args:
        config: Knowledge subsystem configuration.
        vector_store: Optional pre-built vector store. Built from config
            if not provided.
        embedding_provider: Optional pre-built embedding provider. Built
            from config if not provided.
        graph_store: Optional pre-built graph store for provenance tracking.
            When provided, ExternalSource and VectorChunk nodes are created
            in Neo4j after each successful ingestion.

    Returns:
        Ready-to-use IngestionPipeline instance.
    """
    from backend.knowledge.ingestion.pipeline import IngestionPipeline

    vs = vector_store or build_vector_store(config)
    if embedding_provider is None:
        from backend.knowledge.embeddings import EmbeddingGenerator

        embedding_provider = EmbeddingGenerator(config.embedding.model_name)
    ep = embedding_provider
    return IngestionPipeline(
        vector_store=vs,
        embedding_provider=ep,
        config=config.ingestion,
        graph_store=graph_store,
    )


def build_vault_writer(
    config: KnowledgeConfig,
    debug_logger: "DebugJSONLLogger | None" = None,  # noqa: F821
) -> "VaultWriter | None":
    """Create a VaultWriter from config.vault.

    Returns None when config.vault.enabled is False -- the caller (typically
    build_conversation_handler) treats None as "no vault layer" and skips
    write calls. Does NOT call .start() -- the lifecycle owner (server
    lifespan or test) is responsible for start/stop.

    Args:
        config: Knowledge subsystem configuration.
        debug_logger: Optional DebugJSONLLogger (Cluster 8 Phase 12). When
            set + `MIST_DEBUG_VAULT_JSONL=1`, every consumer-side write op
            emits a `phase: "vault"` JSONL record.

    Returns:
        Unstarted VaultWriter, or None if vault is disabled.
    """
    if not config.vault.enabled:
        logger.info("Vault layer disabled (config.vault.enabled=False); skipping VaultWriter")
        return None
    from backend.vault import VaultWriter

    # Phase 8 stamp: same model_hash that flows into RebuildStamps for the
    # EXTRACTED_FROM->ConversationContext and reconciled fact edges (see
    # build_curation_pipeline's RebuildStamps construction above). Populates
    # the `model_hash` frontmatter field on every newly created session note;
    # no command reconciles it against current config today.
    return VaultWriter(config.vault, debug_logger=debug_logger, model_hash=config.model_hash)


def build_sidecar_index(
    config: KnowledgeConfig,
    embedding_provider: EmbeddingProvider | None = None,
) -> "VaultSidecarIndex | None":
    """Create and initialize a VaultSidecarIndex.

    Returns None when config.sidecar_index.enabled is False. Calls .initialize()
    before returning so the SQLite schema is in place. The caller is
    responsible for calling .close() at shutdown.

    Args:
        config: Knowledge subsystem configuration.
        embedding_provider: Optional embedding provider. When None, builds
            EmbeddingGenerator from config.embedding.model_name.

    Returns:
        Initialized VaultSidecarIndex, or None if disabled.
    """
    if not config.sidecar_index.enabled:
        logger.info("Sidecar index disabled; skipping VaultSidecarIndex")
        return None
    from backend.vault.sidecar_index import VaultSidecarIndex

    if embedding_provider is None:
        from backend.knowledge.embeddings import EmbeddingGenerator

        embedding_provider = EmbeddingGenerator(config.embedding.model_name)
    index = VaultSidecarIndex(config.sidecar_index, embedding_provider)
    index.initialize()
    return index


def build_filewatcher(
    config: KnowledgeConfig,
    sidecar_index: "VaultSidecarIndex | None" = None,
    writer: "VaultWriter | None" = None,
) -> "VaultFilewatcher | None":
    """Create a VaultFilewatcher.

    Returns None when config.filewatcher.enabled is False, when config.vault
    is disabled (no vault to watch), or when sidecar_index is None (nothing
    to reindex into). The lifecycle owner (server lifespan) is responsible
    for calling .start(loop) and .stop().

    This is a thin wrapper around build_phase3_components for callers that
    only need the filewatcher. For callers that also need the InvalidationBus
    (e.g. ConversationHandler wiring -- Task 21), use build_phase3_components
    directly to get the shared bus instance.

    Args:
        config: Knowledge subsystem configuration.
        sidecar_index: The sidecar to reindex into on file events.
        writer: VaultWriter for session note writes. May be None.

    Returns:
        Unstarted VaultFilewatcher, or None if filewatcher/vault/sidecar
        is disabled.
    """
    from backend.vault.invalidation_bus import InvalidationBus

    if not config.filewatcher.enabled:
        logger.info("Filewatcher disabled; skipping VaultFilewatcher")
        return None
    if not config.vault.enabled:
        logger.info("Vault disabled; skipping VaultFilewatcher")
        return None
    if sidecar_index is None:
        logger.warning(
            "build_filewatcher called with sidecar_index=None; filewatcher cannot reindex"
        )
        return None

    from backend.vault import VaultFilewatcher

    # Callers that don't need the bus exposed should use build_phase3_components
    # and carry the bus themselves. This wrapper builds a throwaway bus so the
    # constructor signature is satisfied; it will receive events but nobody can
    # subscribe to it. Task 21 migrates server.py to use build_phase3_components.
    bus = InvalidationBus()
    return VaultFilewatcher(
        config.filewatcher,
        config.vault.root,
        sidecar_index,
        invalidation_bus=bus,
        writer=writer,
    )


# ---------------------------------------------------------------------------
# Phase3Components -- bundles filewatcher + bus for shared-instance wiring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Phase3Components:
    """Holds a VaultFilewatcher and the InvalidationBus it publishes to.

    Both share the SAME bus instance. ConversationHandler (Task 21) calls
    `components.invalidation_bus.subscribe(listener)` to receive vault-change
    events emitted by the filewatcher after vault-edit processing.

    Produced by `build_phase3_components`. Consumed by the server lifespan
    (Session A's scope) to wire the handler and start the filewatcher.
    """

    filewatcher: "VaultFilewatcher"
    invalidation_bus: "InvalidationBus"


def build_phase3_components(
    config: KnowledgeConfig,
    sidecar_index: "VaultSidecarIndex | None",
    writer: "VaultWriter | None" = None,
) -> "Phase3Components | None":
    """Create a Phase3Components: VaultFilewatcher + InvalidationBus (shared).

    Returns None when config.filewatcher.enabled is False, config.vault is
    disabled, or sidecar_index is None. These are the same guards as
    build_filewatcher.

    The InvalidationBus on the returned dataclass is the SAME instance wired
    into the filewatcher, so any listener subscribed to `components.invalidation_bus`
    will receive every vault-change event published by the filewatcher.

    Args:
        config: Knowledge subsystem configuration.
        sidecar_index: Initialized VaultSidecarIndex. None returns None.
        writer: Pre-built VaultWriter. REQUIRED whenever the vault is
            enabled: without it the filewatcher cannot flip authored_by or
            publish cache-invalidation events, so user edits would index but
            never propagate to the read path.

    Returns:
        Phase3Components(filewatcher, invalidation_bus), or None when any
        prerequisite is disabled.

    Raises:
        ValueError: When the vault is enabled but `writer` is None -- the
            composition would silently break the vault-edit read-path (the
            exact production wiring bug the deep review surfaced).
    """
    from backend.vault import VaultFilewatcher
    from backend.vault.invalidation_bus import InvalidationBus

    if not config.filewatcher.enabled:
        logger.info("Filewatcher disabled; skipping Phase3Components")
        return None
    if not config.vault.enabled:
        logger.info("Vault disabled; skipping Phase3Components")
        return None
    if sidecar_index is None:
        logger.warning(
            "build_phase3_components called with sidecar_index=None; skipping Phase3Components"
        )
        return None
    if writer is None:
        raise ValueError(
            "build_phase3_components requires writer= when the vault is "
            "enabled: a writer-less filewatcher cannot flip authored_by or "
            "publish cache-invalidation events, so user edits would index but "
            "never propagate to the read path."
        )

    bus = InvalidationBus()
    filewatcher = VaultFilewatcher(
        config.filewatcher,
        config.vault.root,
        sidecar_index,
        invalidation_bus=bus,
        writer=writer,
    )
    # MIST-write self-marking: consumer handlers mark each path right before
    # mutating it so the filewatcher classifies the resulting event as
    # MIST-origin. Without this every per-turn session append runs the
    # user-edit invariant-5 sequence (spurious authored_by corruption; the
    # graph-side steps that sequence used to also trigger -- provenance
    # orphaning and a full-note re-extraction -- retired under R1.3).
    writer.set_mist_write_marker(filewatcher.mark_mist_write)
    return Phase3Components(filewatcher=filewatcher, invalidation_bus=bus)
