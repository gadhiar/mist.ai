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

    if embedding_provider is None:
        embedding_provider = EmbeddingGenerator(config.embedding.model_name)
    confidence_mgr = ConfidenceManager()
    # ADR-010 Phase 8 rebuild-determinism stamps. Written to every fact edge
    # (C1 4.7) and every DERIVED_FROM->VaultNote edge so rebuilds can detect
    # when the ontology, extraction prompt, or model binary has drifted from
    # the values active at extraction time.
    rebuild_stamps = RebuildStamps(
        ontology_version=config.ontology_version,
        extraction_version=config.extraction_version,
        model_hash=f"{config.model_hash}|emb:{config.embedding.model_name}",
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


def build_extraction_pipeline(
    config: KnowledgeConfig,
    graph_store: GraphStore | None = None,
    llm_provider: StreamingLLMProvider | None = None,
    include_curation: bool = True,
    include_internal_derivation: bool = True,
    debug_logger: "DebugJSONLLogger | None" = None,  # noqa: F821
) -> ExtractionPipeline:
    """Create a fully wired ExtractionPipeline."""
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
    if include_internal_derivation:
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
      `_on_vault_rebuild` listener so vault-edit rebuilds evict stale
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
            rebuild-completion events and evict stale mist_context caches. The
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
    )


def build_graph_regenerator(config: KnowledgeConfig):
    """Quarantined: the legacy utterance->graph regenerator is superseded.

    Raises immediately instead of constructing graph/LLM wiring for a
    component ADR-010 retired (the utterance->graph rebuild is being
    redesigned under sub-project A R1). Kept as a tombstone so stale
    callers fail with direction instead of resurrecting the legacy path
    (deep review vault-layer-adr010-7).
    """
    raise RuntimeError(
        "build_graph_regenerator is quarantined (ADR-010): the legacy "
        "regeneration path is superseded. Use 'mist_admin vault-rebuild' for "
        "vault-derived rebuilds; the utterance->graph regenerator ships with "
        "sub-project A R1."
    )


def build_curation_scheduler(
    config: KnowledgeConfig,
    event_store: EventStoreProvider | None = None,
    tracker: "ToolUsageTracker | None" = None,  # noqa: F821
    llm_provider: StreamingLLMProvider | None = None,
):
    """Create a fully wired CurationScheduler with all maintenance jobs.

    Args:
        config: Knowledge subsystem configuration.
        event_store: Optional event store for SelfReflectionJob. When None,
            the reflection job returns immediately with zero counts.
        tracker: Optional ToolUsageTracker for SkillDerivationJob. When None,
            a default tracker is created from config.
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
        ]
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

    # Phase 8 stamp: same model_hash that flows into RebuildStamps for graph
    # DERIVED_FROM->VaultNote edges (line 151 above). Populates the
    # `model_hash` frontmatter field on every newly created session note so
    # vault rebuild can reconcile session-note vintage against current config.
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
    regenerator: "GraphRegenerator | None" = None,  # noqa: F821
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
        regenerator: Unused (R1.3 Task 6 deleted GraphRegenerator; the
            filewatcher no longer accepts a graph-store dependency). Kept as
            an accepted-but-ignored parameter so existing callers do not
            break at the call site; Task 7 removes it.
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
    `components.invalidation_bus.subscribe(listener)` to receive rebuild
    completion events emitted by the filewatcher after vault-edit processing.

    Produced by `build_phase3_components`. Consumed by the server lifespan
    (Session A's scope) to wire the handler and start the filewatcher.
    """

    filewatcher: "VaultFilewatcher"
    invalidation_bus: "InvalidationBus"


def build_phase3_components(
    config: KnowledgeConfig,
    sidecar_index: "VaultSidecarIndex | None",
    regenerator: "GraphRegenerator | None" = None,  # noqa: F821
    writer: "VaultWriter | None" = None,
) -> "Phase3Components | None":
    """Create a Phase3Components: VaultFilewatcher + InvalidationBus (shared).

    Returns None when config.filewatcher.enabled is False, config.vault is
    disabled, or sidecar_index is None. These are the same guards as
    build_filewatcher.

    The InvalidationBus on the returned dataclass is the SAME instance wired
    into the filewatcher, so any listener subscribed to `components.invalidation_bus`
    will receive every rebuild-completion event published by the filewatcher.

    Args:
        config: Knowledge subsystem configuration.
        sidecar_index: Initialized VaultSidecarIndex. None returns None.
        regenerator: Unused (R1.3 Task 6 deleted GraphRegenerator; the
            filewatcher no longer accepts a graph-store dependency). Kept as
            an accepted-but-ignored parameter so existing callers do not
            break at the call site; Task 7 removes it.
        writer: Pre-built VaultWriter. REQUIRED whenever the vault is
            enabled: without it the filewatcher cannot run the ADR-010
            invariant-5 chain (authored_by writeback -> graph rebuild ->
            cache invalidation) and user edits silently stop propagating.

    Returns:
        Phase3Components(filewatcher, invalidation_bus), or None when any
        prerequisite is disabled.

    Raises:
        ValueError: When the vault is enabled but `writer` is None -- the
            composition would silently disable invariant 5 (the exact
            production wiring bug the deep review surfaced).
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
            "enabled: a writer-less filewatcher cannot run the ADR-010 "
            "invariant-5 chain, so user edits would index but never flip "
            "authored_by, rebuild the graph, or evict caches."
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
