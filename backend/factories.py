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

from backend.interfaces import EmbeddingProvider, EventStoreProvider, GraphConnection
from backend.knowledge.config import KnowledgeConfig
from backend.knowledge.curation.confidence import ConfidenceManager
from backend.knowledge.curation.conflict_resolver import ConflictResolver
from backend.knowledge.curation.deduplication import EntityDeduplicator
from backend.knowledge.curation.graph_writer import CurationGraphWriter
from backend.knowledge.curation.pipeline import CurationPipeline
from backend.knowledge.embeddings import EmbeddingGenerator
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
    embeddings = embedding_generator or EmbeddingGenerator(config.embedding.model_name)
    return GraphStore(conn, embeddings)


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


def build_curation_pipeline(config: KnowledgeConfig, executor: GraphExecutor) -> CurationPipeline:
    """Create a fully wired CurationPipeline."""
    embedding_provider = EmbeddingGenerator(config.embedding.model_name)
    confidence_mgr = ConfidenceManager()
    return CurationPipeline(
        deduplicator=EntityDeduplicator(executor, embedding_provider, confidence_mgr),
        conflict_resolver=ConflictResolver(executor),
        graph_writer=CurationGraphWriter(executor, embedding_provider, confidence_mgr),
    )


def build_extraction_pipeline(
    config: KnowledgeConfig,
    graph_store: GraphStore | None = None,
    llm_provider: StreamingLLMProvider | None = None,
    include_curation: bool = True,
    include_internal_derivation: bool = True,
) -> ExtractionPipeline:
    """Create a fully wired ExtractionPipeline."""
    gs = graph_store or build_graph_store(config)
    executor = build_graph_executor(config, gs.connection)

    curation = build_curation_pipeline(config, executor) if include_curation else None

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
    """
    from backend.chat.conversation_handler import ConversationHandler
    from backend.debug_jsonl_logger import DebugJSONLLogger
    from backend.errors import VectorStoreError
    from backend.knowledge.extraction.tool_usage_tracker import ToolUsageTracker

    debug_logger = DebugJSONLLogger.from_env()
    if debug_logger.enabled:
        gates = []
        if debug_logger.llm_call_enabled:
            gates.append("llm_call")
        if debug_logger.retrieval_candidates_enabled:
            gates.append("retrieval_candidates")
        if debug_logger.llm_request_dump_enabled:
            gates.append("llm_request_raw")
        gate_summary = ", ".join(gates) if gates else "turn + extraction only"
        logger.info(
            "Debug JSONL logging enabled at %s (phases: %s)",
            debug_logger.path,
            gate_summary,
        )

    gs = build_graph_store(config)
    provider = llm_provider or build_llm_provider(config, debug_logger=debug_logger)
    pipeline = build_extraction_pipeline(
        config, graph_store=gs, llm_provider=provider, include_curation=True
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
    )

    tracker = ToolUsageTracker(config.skill_derivation)

    return ConversationHandler(
        config=config,
        graph_store=gs,
        extraction_pipeline=pipeline,
        retriever=retriever,
        llm_provider=provider,
        tool_usage_tracker=tracker,
        debug_logger=debug_logger,
    )


def build_graph_regenerator(config: KnowledgeConfig):
    """Create a fully wired GraphRegenerator (no curation)."""
    from backend.knowledge.regeneration.graph_regenerator import GraphRegenerator

    gs = build_graph_store(config)
    pipeline = build_extraction_pipeline(
        config, graph_store=gs, include_curation=False, include_internal_derivation=False
    )
    return GraphRegenerator(
        config=config,
        extraction_pipeline=pipeline,
        graph_store=gs,
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
    ep = embedding_provider or EmbeddingGenerator(config.embedding.model_name)
    return IngestionPipeline(
        vector_store=vs,
        embedding_provider=ep,
        config=config.ingestion,
        graph_store=graph_store,
    )
