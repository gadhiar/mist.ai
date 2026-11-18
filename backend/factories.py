"""Composition root for MIST.AI backend.

All dependency wiring lives here. Classes accept required constructor
params -- this module provides the factory functions that know how to
assemble them with real implementations.

Usage:
    from backend.factories import build_graph_store
    graph_store = build_graph_store(config)

For tests, bypass factories and pass fakes directly to constructors.
"""

import asyncio
import functools
import logging
import os
import random
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import httpx

from backend.capabilities.action_log import ActionLog
from backend.capabilities.alarms.capability import (
    TZ_UNSET_REASON,
    AlarmsCapability,
    ClockCapability,
    RingingTracker,
    SchedulerAlarmControl,
    TimersCapability,
    UserZone,
)
from backend.capabilities.alarms.records import AlarmRecords
from backend.capabilities.alarms.ringing import DesktopSink, RingRouter
from backend.capabilities.alarms.scheduler import AlarmScheduler
from backend.capabilities.clock import Clock
from backend.capabilities.config import CapabilitiesConfig
from backend.capabilities.contracts import Capability, CapabilityStatus
from backend.capabilities.db import CapabilitiesDb
from backend.capabilities.decoder import DECODE_INSTRUCTIONS, ActionDecoder
from backend.capabilities.events import EventSink
from backend.capabilities.host import CapabilityHost, HostSettings
from backend.capabilities.lights.capability import (
    LightAdjustment,
    LightsCapability,
    StatusReading,
)
from backend.capabilities.lights.catalog import EMPTY_CATALOG, LightCatalog, LightTarget
from backend.capabilities.lights.home_assistant import HomeAssistantLightsProvider
from backend.capabilities.mcp.connection import BackoffPolicy, McpConnection
from backend.capabilities.mcp.transport import McpTransport
from backend.capabilities.music.capability import MusicCapability
from backend.capabilities.music.spotify_api import SpotifyAccounts, SpotifyApi, SpotifyTokenManager
from backend.capabilities.music.token_store import SpotifyTokenStore
from backend.capabilities.policy import PolicyTable
from backend.capabilities.power import PowerState, TierProviders
from backend.capabilities.replies import ReplyRegister
from backend.capabilities.router import Embedder, RouterConfig, ToolRouter
from backend.capabilities.secrets import Secret, SecretRegistry, SecretSource
from backend.errors import CapabilityConfigError, CapabilityUnavailableError, SecretMissingError
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
from backend.knowledge.config import ContextBudgetConfig, KnowledgeConfig, LLMConfig, LLMRole
from backend.knowledge.curation.confidence import ConfidenceManager
from backend.knowledge.curation.deduplication import EntityDeduplicator
from backend.knowledge.curation.graph_writer import CurationGraphWriter, RebuildStamps
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
from backend.origins import ALLOW_ALL, ALLOWED_ORIGINS_ENV, allowed_origins
from backend.user_timezone import USER_TIMEZONE_ENV, resolve_user_timezone

if TYPE_CHECKING:
    from backend.chat.context_budget import TokenCounter
    from backend.realtime.budget import VoiceSettings
    from backend.realtime.power import PowerGate
    from backend.realtime.protocol import PowerLevel
    from backend.realtime.runtime import EngineLoaders, GpuMeter, MistRuntime
    from backend.realtime.session import CapabilityMessage, InferenceLanes, SpeechEngines

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

        embedding_generator = EmbeddingGenerator(
            config.embedding.model_name, device=config.embedding.resolved_device()
        )
    return GraphStore(conn, embedding_generator)


def build_llm_provider(
    config: KnowledgeConfig,
    debug_logger: "DebugJSONLLogger | None" = None,  # noqa: F821
    *,
    role: LLMRole | None = None,
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
        role: The caller's LLM role. The provider targets
            `config.llm.endpoint(role)`: the role's own endpoint when its URL
            variable is set, else `base_url` and `model`. None, as extraction
            and curation pass, is always the base. The wrappers below apply to
            a role's provider exactly as to the base.

    Returns:
        StreamingLLMProvider instance (LlamaServerProvider or OllamaProvider),
        optionally wrapped by InstrumentedStreamingLLMProvider, then by
        AdaptiveThinkingProvider (MIS-171 T4 decision 6) when
        `config.llm.tool_thinking_budget_tokens` is not None. The adaptive
        wrapper sits OUTERMOST so both its budgeted attempt and any
        unbudgeted retry each pass through the instrumented wrapper and emit
        their own `llm_call` record. This function is also the extraction
        pipeline's provider factory (`build_extraction_pipeline` calls it
        too) -- harmless there because the wrapper only ever acts on
        requests that carry `tools`, which extraction/scope-classifier calls
        do not (`grep -n 'tools=' backend/knowledge/extraction/*.py` has no
        hits). The out-of-process extraction microservice
        (`backend/extraction_service/`) has its own settings and never calls
        this function at all (`grep -rn build_llm_provider
        backend/extraction_service` has no hits).
    """
    llm_config = config.llm
    endpoint = llm_config.endpoint(role)
    if llm_config.backend == "llamacpp":
        from backend.llm.llama_server_provider import LlamaServerProvider

        provider: StreamingLLMProvider = LlamaServerProvider(
            base_url=endpoint.base_url,
            model=endpoint.model,
        )
    elif llm_config.backend == "ollama":
        from backend.llm.ollama_provider import OllamaProvider

        provider = OllamaProvider(
            base_url=endpoint.base_url,
            model=endpoint.model,
        )
    else:
        raise ValueError(f"Unknown LLM backend: {llm_config.backend}")

    if debug_logger is not None and debug_logger.llm_call_enabled:
        from backend.llm.instrumented_provider import InstrumentedStreamingLLMProvider

        logger.info("LLM provider wrapped with observability instrumentation")
        provider = InstrumentedStreamingLLMProvider(provider, debug_logger)

    if llm_config.tool_thinking_budget_tokens is not None:
        from backend.llm.adaptive_thinking import AdaptiveThinkingProvider

        provider = AdaptiveThinkingProvider(
            provider, budget_tokens=llm_config.tool_thinking_budget_tokens
        )

    return provider


def _probe_llama_server_n_ctx(base_url: str, timeout: float = 2.0) -> int | None:
    """Synchronous GET `base_url/props`, returning `default_generation_settings.n_ctx`.

    A plain synchronous `httpx.Client`, not
    `LlamaServerProvider.server_context_size()` (which is async) --
    `build_conversation_handler` is a sync function that may itself run
    inside the server's already-running event loop (the FastAPI lifespan),
    where `asyncio.run(...)` would raise "cannot run event loop while
    another loop is running". A bare sync call has no such hazard, at the
    cost of duplicating the parse logic in `LlamaServerProvider.server_context_size`
    (both anchor to the same field path, verified there).

    Returns None (never raises) on a connection failure, a non-200 status, a
    non-JSON body, or a JSON body missing an int at that path.
    """
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(f"{base_url}/props")
    except httpx.HTTPError:
        return None
    if r.status_code != 200:
        return None
    try:
        data = r.json()
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    generation_settings = data.get("default_generation_settings")
    if not isinstance(generation_settings, dict):
        return None
    n_ctx = generation_settings.get("n_ctx")
    return n_ctx if isinstance(n_ctx, int) else None


def resolve_context_budget_window(
    context_budget: ContextBudgetConfig,
    llm_config: LLMConfig,
    timeout: float = 2.0,
) -> int:
    """Resolve `ContextBudgetConfig.context_window` to a concrete int.

    MIS-171 T4 decision (A): an explicit int always wins (tests, the eval
    harness, an operator override) -- returned as-is with no I/O. The
    unresolved `"auto"` sentinel (the new `from_env()` default when
    `MIST_CTX_BUDGET_WINDOW` is unset or literal `"auto"`) triggers a
    startup-time GET to llama-server's `/props`, reading
    `default_generation_settings.n_ctx` -- the same field path
    `scripts/model_bench/bench_host.py:2257` already parses from that
    endpoint (`props_now.get("default_generation_settings", {}).get("n_ctx",
    32768)`), the one existing repo call site for this response shape.

    Never blocks startup and never raises: any httpx error, non-200 status,
    or malformed `/props` body falls back to `LLM_CTX_SIZE` (matching
    `docker-compose.yml`'s `LLAMA_ARG_CTX_SIZE=${LLM_CTX_SIZE:-32768}`), else
    the literal 32768 when that env var is also unset -- logging a WARNING
    naming the fallback either way.
    """
    if isinstance(context_budget.context_window, int):
        return context_budget.context_window

    n_ctx = _probe_llama_server_n_ctx(llm_config.base_url, timeout=timeout)
    if n_ctx is not None:
        return n_ctx

    fallback = int(os.getenv("LLM_CTX_SIZE", "32768"))
    logger.warning(
        "ContextBudgetConfig.context_window=auto but llama-server /props at "
        "%s/props was unreachable or returned no usable n_ctx; falling back "
        "to LLM_CTX_SIZE=%d",
        llm_config.base_url,
        fallback,
    )
    return fallback


def writer_stamps_from_config(config: KnowledgeConfig) -> RebuildStamps:
    """The curation stamp triple for `config`.

    `build_curation_pipeline` passes the result to its graph writer and
    reconciliation engine, and `build_extraction_dispatcher` passes it to the
    dispatcher's writer-stamp guard. Both derive the triple through this one
    function, so for one config the two values cannot differ in how they are
    derived. They are equal only when the two callers get equal configs: in
    the server each reads its own `KnowledgeConfig.from_env()`
    (`backend/server.py`, `_start_extraction_dispatcher`'s caller;
    `backend/voice_models/model_manager.py` for the curation pipeline), and
    the two agree because both read the same environment. The model hash is
    `compose_model_hash(config)`, never the bare `config.model_hash`.
    """
    from backend.knowledge.version_stamps import compose_model_hash

    # Assigned rather than returned inline so the construction-site grep cited
    # in `build_extraction_pipeline` and tests/unit/test_factories_rebuild_stamps.py
    # still counts this site.
    rebuild_stamps = RebuildStamps(
        ontology_version=config.ontology_version,
        extraction_version=config.extraction_version,
        model_hash=compose_model_hash(config),
    )
    return rebuild_stamps


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
    from backend.knowledge.embeddings import EmbeddingGenerator

    if embedding_provider is None:
        embedding_provider = EmbeddingGenerator(
            config.embedding.model_name, device=config.embedding.resolved_device()
        )
    confidence_mgr = ConfidenceManager()
    # ADR-010 Phase 8 rebuild-determinism stamps. Written to every fact edge
    # (C1 4.7, reconciliation.py) and every EXTRACTED_FROM->ConversationContext
    # edge (R1.3 moved this anchor off DERIVED_FROM->VaultNote) to record the
    # ontology, extraction prompt, and model binary active when each edge was
    # written. Nothing reads them back off the edges to compare: the two
    # queries that name them test only non-null (SEED_ONLY_PROBE_CYPHER in
    # extraction_backlog/cutover.py, RESET_GUARD_CYPHER in knowledge/admin.py:
    # grep -rnE 'model_hash IS NOT NULL|in _EXTRACTION_STAMP_PROPERTIES'
    # backend). The extraction dispatcher's writer-stamp
    # guard compares a triple derived by the same function (from its own
    # config) with the active epoch, not the stored edges, when no cutover is open.
    rebuild_stamps = writer_stamps_from_config(config)
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
    # build_curation_pipeline builds its own RebuildStamps from (through
    # writer_stamps_from_config) -- both
    # real construction sites, and only those two, are found by `grep -nE
    # "^\s+(rebuild_stamps = )?RebuildStamps\(" backend/factories.py`
    # (writer_stamps_from_config's assignment, and this function's own
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


def build_extraction_dispatcher(
    config: KnowledgeConfig,
    *,
    pipeline: ExtractionPipeline,
    event_store: "EventStore",  # noqa: F821
    settings: "DispatcherSettings | None" = None,  # noqa: F821
    extraction_cache: "ExtractionCache | None" = None,  # noqa: F821
    http_client: "httpx.AsyncClient | None" = None,  # noqa: F821
):
    """Build (but do NOT start) the T2a extraction-backlog dispatcher.

    Not starting it is deliberate: admin scripts and benches that build a
    conversation handler must keep the in-process extraction path, and a
    dispatcher that is built, attached and never started would silently stop
    their extraction. The server starts it (`backend/server.py`).

    Args:
        config: Knowledge configuration; supplies the embedding model identity
            the dispatcher folds into the service's model hash, and the writer
            stamps the dispatcher compares with the active epoch before it
            applies (`writer_stamps_from_config`, the function
            `build_curation_pipeline` derives its stamps with).
        pipeline: The handler's extraction pipeline (gates, Stage 9 context,
            apply step). Its own LLM stages are never called by the dispatcher.
        event_store: The handler's event store -- the log the backlog reads.
        settings: Dispatcher settings; `DispatcherSettings.from_env()` when None.
        extraction_cache: The cache to read and write. Defaults to the
            pipeline's own cache instance, then to a fresh one at
            `production_cache_path(config)`.
        http_client: Client for the extraction service. Built (and closed on
            `stop()`) here when None, with `settings.request_timeout_s` as the
            read timeout.

    Returns:
        An `ExtractionDispatcher`, not started.
    """
    import httpx

    from backend.extraction_backlog.dispatcher import ExtractionDispatcher
    from backend.extraction_backlog.inference import RemoteExtractionInference
    from backend.extraction_backlog.settings import DispatcherSettings
    from backend.extraction_backlog.store import BacklogStore
    from backend.knowledge.extraction_cache import ExtractionCache

    resolved_settings = settings or DispatcherSettings.from_env()
    cache = extraction_cache or pipeline.extraction_cache
    if cache is None:
        cache = ExtractionCache(production_cache_path(config))
        cache.initialize()

    inference = None
    on_stop = None
    if resolved_settings.mode == "service":
        client = http_client
        if client is None:
            client = httpx.AsyncClient(
                timeout=httpx.Timeout(resolved_settings.request_timeout_s, connect=5.0)
            )
            on_stop = client.aclose
        inference = RemoteExtractionInference(client, resolved_settings.service_url)

    return ExtractionDispatcher(
        store=BacklogStore(event_store, cache),
        pipeline=pipeline,
        inference=inference,
        settings=resolved_settings,
        embedding_model_name=config.embedding.model_name,
        writer_stamps=writer_stamps_from_config(config),
        on_stop=on_stop,
    )


def build_conversation_handler(
    config: KnowledgeConfig,
    llm_provider: StreamingLLMProvider | None = None,
    vault_writer: "VaultWriter | None" = None,
    vault_sidecar: SidecarIndexProtocol | None = None,
    invalidation_bus: "InvalidationBus | None" = None,
    graph_store: GraphStore | None = None,
    vector_store: "VectorStoreProvider | None" = None,  # noqa: F821
    with_extraction_dispatcher: bool = False,
    *,
    conversation_llm_provider: StreamingLLMProvider | None = None,
    register_prompt: str | None = None,
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

    T39 (LLM roles): the base provider and the conversation's are split.
    The extraction pipeline always gets the base (`llm_provider`, or one
    built from `base_url` and `model`). The handler gets the conversation
    role's, so a smaller conversation model on its own server never moves
    extraction. With the role unset the handler gets the base object itself,
    so both hold one provider, as before roles existed. Curation takes the
    base from `MistRuntime.llm_provider`, not from the handler
    (`backend/server.py::_resolve_curation_dependencies`).

    Args:
        config: Knowledge subsystem configuration.
        llm_provider: Optional pre-built base LLM provider, for the
            extraction pipeline and, with no conversation role set, the
            handler. The live server injects the one `lifespan` builds, through
            `build_mist_runtime`.
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
        graph_store: Optional pre-built GraphStore. When None, one is
            constructed from config via `build_graph_store` (opens a real
            Neo4j connection). Tests inject a `GraphStore` wrapping a fake
            connection to avoid touching Neo4j.
        vector_store: Optional pre-built vector store. When None, one is
            constructed via `build_vector_store(config)` with the existing
            graceful fallback to graph-only retrieval on failure. Tests
            inject a fake vector store to avoid touching LanceDB.
        with_extraction_dispatcher: T2a. When True and the handler has an
            event store, build an extraction-backlog dispatcher over the
            handler's own pipeline and event store (`build_extraction_dispatcher`)
            and attach it, so recorded turns wake the backlog instead of
            spawning in-process extraction. The dispatcher is NOT started; the
            caller owns `start()`/`stop()`. Default False keeps every existing
            caller (admin scripts, benches) on the in-process path. The server
            does not use this flag: its handler is built inside
            `build_mist_runtime`, so `backend/server.py` attaches one after the
            fact via `ConversationHandler.attach_extraction_dispatcher`.
        conversation_llm_provider: Optional pre-built provider for the
            handler alone. When None, the handler gets
            `build_llm_provider(config, role="conversation")` if the
            conversation role is set in `config.llm.roles`, else the base
            provider. The extraction pipeline never receives it.
        register_prompt: The persona's register sentence for the prompt's stable prefix
            (`PersonaRegister.prompt_line()`); None leaves the prompt without one.
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

    gs = graph_store
    if gs is None:
        gs = build_graph_store(config)
    provider = llm_provider or build_llm_provider(config, debug_logger=debug_logger)
    if conversation_llm_provider is not None:
        handler_provider = conversation_llm_provider
    elif "conversation" in config.llm.roles:
        handler_provider = build_llm_provider(
            config, debug_logger=debug_logger, role="conversation"
        )
    else:
        handler_provider = provider
    pipeline = build_extraction_pipeline(
        config,
        graph_store=gs,
        llm_provider=provider,
        include_curation=True,
        debug_logger=debug_logger,
    )

    # Build vector store with graceful fallback
    if vector_store is None:
        try:
            vector_store = build_vector_store(config)
        except (VectorStoreError, Exception) as exc:
            logger.warning(
                "Vector store unavailable, falling back to graph-only retrieval: %s", exc
            )

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

    # MIS-171 T4 decision (A): resolve ContextBudgetConfig.context_window
    # ("auto" or an explicit int) to a concrete int and build the planner
    # here, rather than letting ConversationHandler's own fallback
    # (`config.context_budget.enabled` -> `ContextBudgetPlanner(config.context_budget)`)
    # construct one from a possibly-unresolved config. Passing an explicit
    # `budget_planner` short-circuits that fallback entirely (see
    # `ConversationHandler.__init__`). Disabled budgeting (`enabled=False`)
    # keeps `budget_planner=None`, matching pre-existing legacy behavior.
    budget_planner = None
    if config.context_budget.enabled:
        from backend.chat.context_budget import ContextBudgetPlanner

        resolved_window = resolve_context_budget_window(config.context_budget, config.llm)
        resolved_context_budget = replace(config.context_budget, context_window=resolved_window)
        budget_planner = ContextBudgetPlanner(resolved_context_budget)

    # Cluster 8 Phase 5: vault_writer is caller-provided (or None). Auto-build
    # removed to avoid two writers racing on the same vault root -- the
    # server lifespan owns the single VaultWriter and plumbs it through
    # `build_mist_runtime` -> here. Unit
    # tests that want wiring coverage pass an explicit writer or None.
    handler = ConversationHandler(
        config=config,
        graph_store=gs,
        extraction_pipeline=pipeline,
        retriever=retriever,
        llm_provider=handler_provider,
        conventions_loader=conventions_loader,
        tool_usage_tracker=tracker,
        debug_logger=debug_logger,
        budget_planner=budget_planner,
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
        register_prompt=register_prompt,
    )
    if with_extraction_dispatcher and handler.event_store is not None:
        handler.attach_extraction_dispatcher(
            build_extraction_dispatcher(config, pipeline=pipeline, event_store=handler.event_store)
        )
    return handler


def build_curation_scheduler(
    config: KnowledgeConfig,
    event_store: EventStoreProvider | None = None,
    tracker: "ToolUsageTracker | None" = None,  # noqa: F821
    llm_provider: StreamingLLMProvider | None = None,
    defer_llm_jobs: Callable[[], bool] | None = None,
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
        defer_llm_jobs: True while gaming mode holds the GPU, which stops
            mist-llm. The scheduler then skips `self_reflection`, the only job
            built with the LLM (through `InternalKnowledgeDeriver`), without
            advancing its `last_run`, so it runs once the deferral ends. None,
            as with no gaming mode configured, never defers.
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
    embedding_provider = EmbeddingGenerator(
        config.embedding.model_name, device=config.embedding.resolved_device()
    )

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
                JobConfig(name="self_reflection", interval_seconds=86400, requires_llm=True),
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
        defer_llm_jobs=defer_llm_jobs,
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

        embedding_provider = EmbeddingGenerator(
            config.embedding.model_name, device=config.embedding.resolved_device()
        )
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

        embedding_provider = EmbeddingGenerator(
            config.embedding.model_name, device=config.embedding.resolved_device()
        )
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


# ---------------------------------------------------------------------------
# Capabilities -- the capability host and what it owns
# ---------------------------------------------------------------------------

# Bounds each Spotify request in every phase (connect, read, write, pool), as the authorize CLI
# does (`backend.capabilities.music.authorize.HTTP_TIMEOUT_S`).
_SPOTIFY_HTTP_TIMEOUT_S = 15.0


@dataclass(frozen=True, slots=True)
class CapabilityBuild:
    """What `build_capability_host` built: a host, or the reason there is none.

    Disabled means no host at all, never an inert one: the server keeps `disabled_reason` only
    so `GET /capabilities` can answer `enabled: false` with it.

    The build owns what the host does not close: the capabilities database and the Spotify
    HTTP client. `close` releases both, after `host.stop()`.

    Attributes:
        host: None whenever `disabled_reason` is set.
        disabled_reason: Why no host was built, for example "capabilities disabled".
        capabilities: The capabilities the host holds, in registration order, so
            `GET /capabilities` can list each one's tools and provider tools; empty with no host.
    """

    host: CapabilityHost | None
    disabled_reason: str | None
    capabilities: tuple[Capability, ...] = field(default=(), compare=False)
    _db: CapabilitiesDb | None = field(default=None, compare=False, repr=False)
    _http: httpx.AsyncClient | None = field(default=None, compare=False, repr=False)

    async def close(self) -> None:
        """Close the Spotify HTTP client and the database; a disabled build closes nothing.

        Call it after `host.stop()`, so no capability is still using either. A second call is
        harmless: both closes are idempotent.
        """
        if self._http is not None:
            await self._http.aclose()
        if self._db is not None:
            # `close` takes the database lock, which a worker-thread call may still hold.
            await asyncio.to_thread(self._db.close)


class _UnavailableLights:
    """A `LightsProvider` that cannot serve, for lights configured without a usable connection.

    Lights then stay listed as `unavailable` with the reason, rather than missing from the
    status surface and `GET /capabilities`.
    """

    def __init__(self, reason: str, clock: Clock) -> None:
        self._reason = reason
        self._clock = clock

    async def start(self) -> None:
        """Start nothing: there is no connection."""

    async def stop(self) -> None:
        """Stop nothing: there is no connection."""

    def status(self) -> CapabilityStatus:
        """Return `unavailable` with the reason the lights cannot connect."""
        return CapabilityStatus(
            name="lights", state="unavailable", reason=self._reason, checked_at=self._clock.now()
        )

    def catalog(self) -> LightCatalog:
        """Return the empty catalog."""
        return EMPTY_CATALOG

    def resolve(self, target: str) -> LightTarget | None:
        """Know no target."""
        return None

    def provider_tools(self) -> tuple[str, ...]:
        """Report no provider tools."""
        return ()

    async def power(self, target: LightTarget, on: bool) -> None:
        """Refuse; the host's status gate stops a call before it gets here."""
        raise CapabilityUnavailableError(self._reason)

    async def adjust(self, target: LightTarget, adjustment: LightAdjustment) -> None:
        """Refuse; the host's status gate stops a call before it gets here."""
        raise CapabilityUnavailableError(self._reason)

    async def read_status(self, target: LightTarget | None) -> StatusReading:
        """Refuse; the host's status gate stops a call before it gets here."""
        raise CapabilityUnavailableError(self._reason)


def _capability_disabled_reason(config: CapabilitiesConfig) -> str | None:
    """Return why no capability host may be built now, or None when one may.

    An isolation flag that does not parse fails closed: an eval or hydration run must never
    reach real lights, music or alarms because of a typo.
    """
    from backend.knowledge.eval_isolation import (
        EvalIsolationError,
        is_eval_isolation_active,
        is_hydration_isolation_active,
    )

    if not config.enabled:
        return "capabilities disabled"
    if ALLOW_ALL in allowed_origins():
        return f"{ALLOWED_ORIGINS_ENV} allows every origin"
    try:
        if is_eval_isolation_active():
            return "eval isolation active (MIST_EVAL_ISOLATION)"
        if is_hydration_isolation_active():
            return "hydration isolation active (MIST_HYDRATION_ISOLATION)"
    except EvalIsolationError as exc:
        return f"isolation flag invalid: {exc}"
    return None


def _user_zone(name: str | None) -> UserZone:
    """Resolve `MIST_TIMEZONE` for alarms, timers and the clock.

    An unset zone is checked first, without the resolver: the resolver's UTC default serves
    dates, but an alarm in a guessed zone rings at the wrong hour. A set name that does not
    resolve to a `ZoneInfo` (an unknown name, or the resolver's UTC fallback when tzdata is
    missing) gives no zone either, so an alarm never rings in a fallback zone.
    """
    if name is None:
        return UserZone(name=None, tz=None, reason=TZ_UNSET_REASON)
    try:
        tz = resolve_user_timezone({USER_TIMEZONE_ENV: name})
    except ValueError:
        tz = None
    if not isinstance(tz, ZoneInfo):
        return UserZone(name=name, tz=None, reason=f"timezone {name} not found (tzdata missing?)")
    return UserZone(name=name, tz=tz, reason=None)


def _lights_transport(
    secret_source: SecretSource,
    secret_registry: SecretRegistry,
    transport_factory: "Callable[[str, Secret], McpTransport] | None",
) -> "tuple[Secret, Callable[[str, Secret], McpTransport]] | str":
    """Return the Home Assistant token and transport factory, or why lights cannot connect."""
    try:
        token = secret_source.get("MIST_HA_TOKEN")
    except SecretMissingError as exc:
        return str(exc)
    if token is None:
        return "MIST_HA_TOKEN unset"
    secret_registry.register(token)
    if transport_factory is None:
        try:
            from backend.capabilities.mcp.sdk_transport import (  # type: ignore[import-not-found]
                McpSdkTransport,
            )
        except ImportError:
            return "mcp package not installed"
        transport_factory = McpSdkTransport
    return token, transport_factory


def build_capability_host(
    config: CapabilitiesConfig,
    *,
    event_sink: EventSink,
    secret_source: SecretSource,
    secret_registry: SecretRegistry,
    clock: Clock,
    embedder: Embedder | None,
    token_counter: "TokenCounter",
    transport_factory: "Callable[[str, Secret], McpTransport] | None",
    power: PowerState,
    tiers: TierProviders,
    register: ReplyRegister,
    desktop_present: Callable[[], bool],
) -> CapabilityBuild:
    """Assemble the capabilities named in `MIST_CAPABILITIES` behind one `CapabilityHost`.

    No host is built while capabilities are disabled, while `MIST_ALLOWED_ORIGINS` allows every
    origin (any web page could then drive devices), or while eval or hydration isolation is
    active or unreadable. Lights are built only with `MIST_HA_MCP_URL`, music only with
    `MIST_SPOTIFY_CLIENT_ID`. Nothing here opens a file or a socket: the database opens on
    first use, so an unwritable path makes alarms and timers `unavailable` at start instead of
    failing the build.

    Alarms and timers share one store, one scheduler and one `RingingTracker`, which wraps the
    `RingRouter` that rings the desktop endpoint, and the host's `SchedulerAlarmControl` runs
    over the same three. Without a usable `MIST_TIMEZONE` none of them is built, and alarms,
    timers and the clock report the zone's reason.

    Args:
        config: The parsed capability environment.
        event_sink: Broadcasts host and alarm payloads to every connection.
        secret_source: Where the Home Assistant token is read from.
        secret_registry: Receives every secret value, for redaction.
        clock: The host's and every capability's clock.
        embedder: For the router's embedding stage; None turns that stage off.
        token_counter: Counts catalog tokens for the router and the lights enum check.
        transport_factory: Builds a Home Assistant MCP transport from the URL and token; None
            uses the MCP SDK transport, imported here so a missing `mcp` package costs only
            lights.
        power: The level each record and broadcast carries.
        tiers: Whose `interaction` provider gates the action decode.
        register: The address form of each session's first templated reply.
        desktop_present: Whether a client is connected, for the desktop ring endpoint.

    Returns:
        A build with a host and no reason, or no host and the reason.
    """
    reason = _capability_disabled_reason(config)
    if reason is not None:
        logger.info("No capability host built: %s", reason)
        return CapabilityBuild(host=None, disabled_reason=reason)

    threshold = config.router.embedding_threshold
    if threshold is not None and embedder is None:
        logger.warning(
            "MIST_CAPABILITY_ROUTER_EMBEDDING_THRESHOLD is set but no embedder is available; "
            "the router's embedding stage stays off"
        )
        threshold = None
    try:
        router_config = RouterConfig(
            token_budget=config.tool_token_budget,
            reserved_tokens=token_counter.count(DECODE_INSTRUCTIONS),
            sticky_turns=config.router.sticky_turns,
            embedding_threshold=threshold,
        )
    except CapabilityConfigError as exc:
        logger.error("No capability host built: %s", exc)
        return CapabilityBuild(host=None, disabled_reason=f"capability config invalid: {exc}")

    names = set(config.capabilities)
    db = CapabilitiesDb(config.db_path)
    records = AlarmRecords(db)
    zone = _user_zone(config.timezone) if names & {"alarms", "timers", "clock"} else None
    scheduler: AlarmScheduler | None = None
    tracker: RingingTracker | None = None
    alarm_control: SchedulerAlarmControl | None = None
    endpoints: Callable[[], tuple[str, ...]] = tuple
    if zone is not None and zone.tz is not None and names & {"alarms", "timers"}:
        sinks = [DesktopSink(event_sink, desktop_present)]
        ring_router = RingRouter(sinks, records, clock, config.alarms, zone.tz)
        tracker = RingingTracker(ring_router)
        scheduler = AlarmScheduler(records, clock, tracker, config.alarms, zone.tz)
        alarm_control = SchedulerAlarmControl(records, scheduler, clock, config.alarms, zone.tz)
        endpoints = ring_router.endpoints

    capabilities: list[Capability] = []
    connection: McpConnection | None = None
    http: httpx.AsyncClient | None = None
    for name in config.capabilities:
        if name in ("alarms", "timers") and zone is not None:
            scheduled = AlarmsCapability if name == "alarms" else TimersCapability
            capabilities.append(
                scheduled(
                    records,
                    scheduler,
                    clock=clock,
                    zone=zone,
                    ringing=tracker,
                    endpoints=endpoints,
                    snooze_minutes=config.alarms.snooze_minutes,
                )
            )
        elif name == "clock" and zone is not None:
            capabilities.append(ClockCapability(clock=clock, zone=zone))
        elif name == "lights" and config.lights.ha_mcp_url is not None:
            url = config.lights.ha_mcp_url
            transport = _lights_transport(secret_source, secret_registry, transport_factory)
            provider: HomeAssistantLightsProvider | _UnavailableLights
            if isinstance(transport, str):
                logger.warning("Lights unavailable: %s", transport)
                provider = _UnavailableLights(transport, clock)
            else:
                token, make = transport
                rng = random.SystemRandom()
                connection = McpConnection(
                    functools.partial(make, url, token),
                    clock,
                    # Equal jitter, so several clients do not reconnect in step.
                    BackoffPolicy(jitter=lambda delay, rng=rng: rng.uniform(delay / 2, delay)),
                )
                provider = HomeAssistantLightsProvider(
                    connection, clock, call_timeout_s=config.tool_timeout_s
                )
            capabilities.append(
                LightsCapability(
                    provider, default_target=config.lights.default_target, counter=token_counter
                )
            )
        elif name == "music" and config.music.spotify_client_id is not None:
            http = httpx.AsyncClient(timeout=_SPOTIFY_HTTP_TIMEOUT_S)
            token_store = SpotifyTokenStore(Path(config.music.spotify_token_path))
            accounts = SpotifyAccounts(
                http,
                client_id=config.music.spotify_client_id,
                registry=secret_registry,
                clock=clock,
            )
            tokens = SpotifyTokenManager(
                store=token_store, accounts=accounts, registry=secret_registry, clock=clock
            )
            capabilities.append(
                MusicCapability(
                    SpotifyApi(http, tokens),
                    device_name=config.music.spotify_device,
                    clock=clock,
                    token_store=token_store,
                )
            )

    specs = [spec for capability in capabilities for spec in capability.tools()]
    try:
        policy = PolicyTable.from_specs(specs, config.policy_file)
    except CapabilityConfigError as exc:
        logger.warning("Capability policy file ignored, defaults in force: %s", exc)
        policy = PolicyTable.from_specs(specs, None)
    host = CapabilityHost(
        capabilities,
        router=ToolRouter(capabilities, router_config, token_counter, embedder),
        counter=token_counter,
        policy=policy,
        decoder=ActionDecoder(),
        secrets=secret_registry,
        action_log=ActionLog(db),
        event_sink=event_sink,
        clock=clock,
        settings=HostSettings.from_config(config),
        alarm_control=alarm_control,
        power=power,
        tiers=tiers,
        register=register,
    )
    if connection is not None:
        connection.set_state_listener(host.status_changed)
    logger.info(
        "Capability host built: %s", ", ".join(c.name for c in capabilities) or "no capabilities"
    )
    return CapabilityBuild(
        host=host,
        disabled_reason=None,
        capabilities=tuple(capabilities),
        _db=db,
        _http=http,
    )


def build_speech_engines(
    settings: "VoiceSettings",
    *,
    loaders: "EngineLoaders | None" = None,
    power_level: "PowerLevel" = "full",
    gpu_meter: "GpuMeter | None" = None,
) -> "SpeechEngines":
    """Load each speech slot from its placement chain, one engine at a time.

    `loaders` defaults to the real adapters, each importing its library only when its option is
    tried; `gpu_meter` defaults to torch's allocator figures, imported on the first CUDA load.

    Args:
        settings: The voice settings; the placement chains and the voice identity are read here.
        loaders: How each engine loads; tests pass fakes.
        power_level: The level at boot; at `off` the GPU is yielded and CUDA options are skipped.
        gpu_meter: Measures each CUDA load's VRAM.
    """
    import importlib

    from backend.realtime.runtime import TorchGpuMeter, load_speech_engines

    return load_speech_engines(
        settings,
        loaders if loaders is not None else _real_engine_loaders(),
        power_level=power_level,
        gpu_meter=(
            gpu_meter
            if gpu_meter is not None
            else TorchGpuMeter(lambda: importlib.import_module("torch"))
        ),
    )


def _real_engine_loaders() -> "EngineLoaders":
    """The T6, T7 and T8 adapters, each loaded through its classmethod `load`."""
    from backend.realtime.engines.kws import OnnxNameSpotter
    from backend.realtime.engines.nemotron_stt import NemotronStreamingTranscriber, import_nemo
    from backend.realtime.engines.parakeet_stt import ParakeetTranscriber, import_onnx_asr
    from backend.realtime.engines.pocket_tts import PocketCloneSynthesizer, import_pocket
    from backend.realtime.engines.silero_vad import (
        SileroVad,
        onnxruntime_providers,
        open_onnx_session,
    )
    from backend.realtime.engines.smart_turn import SmartTurnDetector, whisper_features
    from backend.realtime.runtime import EngineLoaders

    return EngineLoaders(
        chains={
            "pocket-clone": lambda s, device: PocketCloneSynthesizer.load(
                s.placement, s.profile, library_loader=import_pocket, device=device
            ),
            "nemotron-160": lambda s, device: NemotronStreamingTranscriber.load(
                s.placement, library_loader=import_nemo, device=device
            ),
            "parakeet-cpu": lambda s, device: ParakeetTranscriber.load(
                s.placement, library_loader=import_onnx_asr, device=device
            ),
        },
        vad=lambda placement: SileroVad.load(
            placement, session_factory=open_onnx_session, available_providers=onnxruntime_providers
        ),
        eot=lambda placement, endpointing: SmartTurnDetector.load(
            placement,
            endpointing,
            session_factory=open_onnx_session,
            available_providers=onnxruntime_providers,
            feature_extractor_factory=whisper_features,
        ),
        kws=lambda placement: OnnxNameSpotter.load(placement, session_factory=open_onnx_session),
    )


def build_mist_runtime(
    config: KnowledgeConfig,
    *,
    voice_settings: "VoiceSettings",
    llm_provider: StreamingLLMProvider | None = None,
    handler_factory: Callable[..., object] | None = None,
    engine_loaders: "EngineLoaders | None" = None,
    gpu_meter: "GpuMeter | None" = None,
    lanes: "InferenceLanes | None" = None,
    power_level: "Callable[[], PowerLevel] | None" = None,
    tts_enabled: bool = True,
    vault_writer: "VaultWriter | None" = None,
    vault_sidecar: SidecarIndexProtocol | None = None,
    invalidation_bus: "InvalidationBus | None" = None,
    power_gate: "PowerGate | None" = None,
    alarm_ringing: Callable[[], bool] | None = None,
    on_capability_message: "Callable[[str, CapabilityMessage], None] | None" = None,
    register_prompt: str | None = None,
    torch_loader: Callable[[], object] | None = None,
) -> "MistRuntime":
    """Compose what every realtime session shares: responder, speech engines, lanes, settings.

    The handler is built by `handler_factory` (`build_conversation_handler`). Only a `MistError`
    from that build (Neo4j unreachable raises `Neo4jConnectionError`) degrades to
    `DirectResponder`; any other exception is a bug and fails startup with its traceback. The LLM
    is warmed unless the GPU is yielded at boot, and the embedding model always.

    `llm_provider` is the base provider: the handler build gives it to the extraction pipeline
    and, by T39's rule, the conversation role's provider to the handler. `MistRuntime.llm_provider`
    is that base, which curation and `/health` read. The runtime's `gpu` is the GPU control
    gaming mode's `GpuYield` reaches through `release_gpu_engines` and `reload_gpu_engines`; the
    caller attaches it to the loop and the registry.

    Args:
        config: The knowledge configuration; `llm.voice_first_segment_min_chars` sets the
            segmenter's first cut, so the handler's first emission and the first TTS segment
            cut at the same point.
        voice_settings: From `load_voice_settings`.
        llm_provider: The interactive provider; built from `config` when None.
        handler_factory: Builds the handler; `build_conversation_handler` when None.
        engine_loaders: How each speech engine loads; the real adapters when None.
        gpu_meter: Measures each CUDA engine's VRAM; torch's figures when None.
        lanes: The inference lanes; one `InferenceLane` per engine kind when None.
        power_level: The process's power level; `full` until T22 wires its state.
        tts_enabled: Whether voice turns may be spoken.
        vault_writer: Passed to the handler build.
        vault_sidecar: Passed to the handler build.
        invalidation_bus: Passed to the handler build.
        power_gate: The gate every session applies to turns and voice (T22); its level is the
            process's when `power_level` is None.
        alarm_ringing: Whether an alarm rings now (D8), for every controller.
        on_capability_message: Where each session hands `action_decision`, `alarm_action` and
            `alarm_report`.
        register_prompt: The persona's register sentence, passed to the handler build.
        torch_loader: Imports torch for the GPU control's cache release; torch itself when None.
    """
    import importlib

    from backend.errors import MistError
    from backend.llm.models import LLMRequest
    from backend.realtime.degraded import DirectResponder
    from backend.realtime.lanes import InferenceLane
    from backend.realtime.runtime import GpuEngines, MistRuntime, TorchGpuMeter, primary_placement
    from backend.realtime.session import InferenceLanes, SessionServices

    provider = llm_provider if llm_provider is not None else build_llm_provider(config)
    level: Callable[[], PowerLevel] = power_level or (
        power_gate.level if power_gate is not None else (lambda: "full")
    )
    settings = replace(
        voice_settings,
        segmenter=replace(
            voice_settings.segmenter,
            first_segment_min_chars=config.llm.voice_first_segment_min_chars,
        ),
    )

    handler: object | None = None
    if not config.enable_knowledge_integration:
        logger.warning(
            "Knowledge integration is disabled: turns are answered without memory and are "
            "not recorded"
        )
    else:
        build_handler = handler_factory or build_conversation_handler
        try:
            handler = build_handler(
                config=config,
                llm_provider=provider,
                vault_writer=vault_writer,
                vault_sidecar=vault_sidecar,
                invalidation_bus=invalidation_bus,
                register_prompt=register_prompt,
            )
        except MistError as error:
            logger.warning("Knowledge integration unavailable, answering without memory: %s", error)
    responder = handler if handler is not None else DirectResponder(provider)

    load_torch = torch_loader or (lambda: importlib.import_module("torch"))
    meter = gpu_meter if gpu_meter is not None else TorchGpuMeter(load_torch)
    loaders = engine_loaders if engine_loaders is not None else _real_engine_loaders()
    engines = build_speech_engines(settings, loaders=loaders, power_level=level(), gpu_meter=meter)
    gpu = GpuEngines(engines, settings, loaders, torch_loader=load_torch, gpu_meter=meter)
    if lanes is None:
        lanes = InferenceLanes(
            vad=InferenceLane("vad"),
            eot=InferenceLane("eot"),
            stt=InferenceLane("stt"),
            tts=InferenceLane("tts"),
            kws=InferenceLane("kws"),
        )

    if level() == "off":
        logger.info("GPU yielded at boot: skipping the LLM warmup")
    else:
        provider.invoke_sync(LLMRequest(messages=[{"role": "user", "content": "hi"}], max_tokens=1))
    graph_store = getattr(handler, "graph_store", None)
    embedder = getattr(graph_store, "embedding_generator", None)
    warmup = getattr(embedder, "warmup", None)
    if warmup is not None:
        logger.info("Warming up embedding model...")
        warmup()

    extra: dict[str, object] = {}
    if alarm_ringing is not None:
        extra["alarm_ringing"] = alarm_ringing
    if on_capability_message is not None:
        extra["on_capability_message"] = on_capability_message
    services = SessionServices(
        responder=responder,  # type: ignore[arg-type]
        engines=engines,
        lanes=lanes,
        settings=settings,
        placement=lambda: primary_placement(engines, settings),
        power_level=level,
        tts_enabled=tts_enabled,
        power_gate=power_gate,
        **extra,  # type: ignore[arg-type]
    )
    return MistRuntime(
        gpu=gpu,
        llm_provider=provider,
        conversation_handler=handler,
        responder=responder,  # type: ignore[arg-type]
        engines=engines,
        lanes=lanes,
        settings=settings,
        services=services,
        agent_available=handler is not None,
    )
