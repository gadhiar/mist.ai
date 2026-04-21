"""Knowledge Retriever.

Hybrid retrieval system combining vector search with graph traversal.
Routes queries by classified intent (factual, relational, hybrid, live)
and merges results using Reciprocal Rank Fusion when both backends
are consulted.
"""

import logging
import time
from typing import Any

from backend.errors import VectorStoreError
from backend.interfaces import EmbeddingProvider, VectorStoreProvider
from backend.knowledge.config import KnowledgeConfig, QueryIntentConfig
from backend.knowledge.models import (
    RetrievalFilters,
    RetrievalResult,
    RetrievedFact,
    VectorSearchResult,
)
from backend.knowledge.retrieval.query_classifier import QueryClassifier
from backend.knowledge.storage import GraphStore

logger = logging.getLogger(__name__)

# Sentinel value for graph_distance on vector-only results.
# Must be large enough to sort after real graph matches in _rank_facts
# but NOT -1 (which would break the ascending sort).
_VECTOR_DISTANCE_SENTINEL = 999

# Static keyword -> MCP tool mapping for live intent routing.
_LIVE_TOOL_MAP: dict[str, str] = {
    "linear": "mcp__linear__list_issues",
    "ticket": "mcp__linear__get_issue",
    "issue": "mcp__linear__list_issues",
    "sprint": "mcp__linear__list_cycles",
    "github": "mcp__github__search_code",
    "pr": "mcp__github__list_pull_requests",
    "pull request": "mcp__github__list_pull_requests",
    "commit": "mcp__github__list_commits",
    "branch": "mcp__github__create_branch",
    "repo": "mcp__github__get_file_contents",
}


class KnowledgeRetriever:
    """Retrieves relevant knowledge from graph and vector stores.

    Combines:
    - Query intent classification (route to correct backend)
    - Vector similarity search (find relevant document chunks)
    - Graph traversal (expand entity context)
    - RRF merge (combine ranked lists from both backends)
    - User-centric relationships (personalization)

    Example:
        retriever = KnowledgeRetriever(config, graph_store)
        result = await retriever.retrieve(
            query="What programming languages do I know?",
            user_id="User",
            limit=20
        )
        print(result.formatted_context)
    """

    def __init__(
        self,
        config: KnowledgeConfig,
        graph_store: GraphStore,
        vector_store: VectorStoreProvider | None = None,
        query_classifier: QueryClassifier | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ):
        """Initialize retriever.

        Args:
            config: Knowledge system configuration.
            graph_store: Graph storage instance.
            vector_store: Optional vector store for document chunk retrieval.
            query_classifier: Optional query classifier. When None, all
                queries default to relational intent (preserving legacy behaviour).
            embedding_provider: Optional embedding provider for query vectorisation.
                When None and vector_store is provided, falls back to
                graph_store.embedding_generator.
        """
        self.config = config
        self.graph_store = graph_store
        self._vector_store = vector_store
        self._query_classifier = query_classifier
        self._embedding_provider = embedding_provider or getattr(
            graph_store, "embedding_generator", None
        )

        # Default retrieval parameters (can be overridden)
        self.default_limit = 20
        self.default_similarity_threshold = 0.6
        self.default_max_hops = 2

        logger.info("KnowledgeRetriever initialized")

    async def retrieve(
        self,
        query: str,
        user_id: str = "User",
        limit: int | None = None,
        similarity_threshold: float | None = None,
        max_hops: int | None = None,
        filters: RetrievalFilters | None = None,
    ) -> RetrievalResult:
        """Retrieve relevant knowledge for a query.

        Process:
        1. Classify intent (if classifier available)
        2. Route by intent to appropriate backend(s)
        3. Merge results with RRF for hybrid queries
        4. Rank and format context for LLM

        Args:
            query: User's question or search query.
            user_id: User entity ID (default "User").
            limit: Max facts to return (default from config).
            similarity_threshold: Min similarity for vector search (default 0.6).
            max_hops: Graph traversal depth (default 2).
            filters: Optional filters for retrieval.

        Returns:
            RetrievalResult with facts, formatted context, and intent metadata.
        """
        start_time = time.time()

        # Use defaults if not specified
        limit = limit or self.default_limit
        similarity_threshold = similarity_threshold or self.default_similarity_threshold
        max_hops = max_hops or self.default_max_hops
        intent_config = self.config.query_intent or QueryIntentConfig()

        logger.info(f"Retrieving knowledge for query: '{query}'")
        logger.info(
            f"Parameters: limit={limit}, threshold={similarity_threshold}, max_hops={max_hops}"
        )

        # Step 1: Classify intent
        if self._query_classifier is not None:
            intent_result = self._query_classifier.classify(query)
            intent = intent_result.intent
        else:
            intent = "relational"

        logger.info(f"Query intent: {intent}")

        # Step 2: Route by intent
        # Priority 0: Identity intent — return persona block, skip graph/vector paths.
        if intent == "identity":
            ctx = await self.retrieve_mist_context()
            total_time = (time.time() - start_time) * 1000
            return RetrievalResult(
                query=query,
                user_id=user_id,
                facts=[],
                entities_found=0,
                total_facts=0,
                formatted_context=ctx.as_system_prompt_block(),
                retrieval_time_ms=total_time,
                vector_search_time_ms=0.0,
                graph_traversal_time_ms=0.0,
                config_used={
                    "limit": limit,
                    "similarity_threshold": similarity_threshold,
                    "max_hops": max_hops,
                },
                intent=intent,
                requires_mcp=False,
                suggested_tools=(),
            )

        if intent == "live":
            # Early return for live queries
            suggested = self._map_live_tools(query)
            total_time = (time.time() - start_time) * 1000
            return RetrievalResult(
                query=query,
                user_id=user_id,
                facts=[],
                entities_found=0,
                total_facts=0,
                formatted_context="This query requires live data from external tools.",
                retrieval_time_ms=total_time,
                vector_search_time_ms=0,
                graph_traversal_time_ms=0,
                config_used={
                    "limit": limit,
                    "similarity_threshold": similarity_threshold,
                    "max_hops": max_hops,
                },
                intent=intent,
                requires_mcp=True,
                suggested_tools=suggested,
            )

        # Generate embedding once for reuse across backends
        query_embedding: list[float] | None = None
        if intent in ("factual", "hybrid") and self._embedding_provider is not None:
            query_embedding = self._embedding_provider.generate_embedding(query)

        graph_facts: list[RetrievedFact] = []
        vector_facts: list[RetrievedFact] = []
        vector_time = 0.0
        graph_time = 0.0
        entities_found = 0
        document_chunks_used = 0

        # Relational path (graph)
        if intent in ("relational", "hybrid"):
            graph_start = time.time()
            graph_limit = intent_config.max_graph_results if intent == "hybrid" else limit
            graph_facts = await self._relational_retrieve(
                query=query,
                user_id=user_id,
                limit=graph_limit,
                similarity_threshold=similarity_threshold,
                max_hops=max_hops,
                filters=filters,
            )
            graph_time = (time.time() - graph_start) * 1000
            # Count unique entities from graph facts
            entity_ids = set()
            for f in graph_facts:
                entity_ids.add(f.subject)
                entity_ids.add(f.object)
            entities_found = len(entity_ids)

        # Factual path (vector store)
        if intent in ("factual", "hybrid") and self._vector_store is not None:
            vector_start = time.time()
            vector_limit = intent_config.max_vector_results if intent == "hybrid" else limit
            vector_facts = await self._factual_retrieve(
                query=query,
                embedding=query_embedding,
                limit=vector_limit,
            )
            vector_time = (time.time() - vector_start) * 1000
            document_chunks_used = len(vector_facts)

        # Step 3: Merge results
        if intent == "hybrid" and graph_facts and vector_facts:
            merged = self._merge_rrf(graph_facts, vector_facts, intent_config)
            ranked_facts = merged[:limit]
        elif intent == "factual":
            ranked_facts = self._rank_facts(vector_facts, limit)
        else:
            # relational or hybrid where one list is empty
            combined = graph_facts + vector_facts
            ranked_facts = self._rank_facts(combined, limit)

        # Step 4: Format context
        formatted_context = self._format_context(ranked_facts, query)

        total_time = (time.time() - start_time) * 1000

        result = RetrievalResult(
            query=query,
            user_id=user_id,
            facts=ranked_facts,
            entities_found=entities_found,
            total_facts=len(ranked_facts),
            formatted_context=formatted_context,
            retrieval_time_ms=total_time,
            vector_search_time_ms=vector_time,
            graph_traversal_time_ms=graph_time,
            config_used={
                "limit": limit,
                "similarity_threshold": similarity_threshold,
                "max_hops": max_hops,
                "filters": filters,
            },
            intent=intent,
            document_chunks_used=document_chunks_used,
        )

        logger.info(f"Retrieval complete: {result.summary()}")

        return result

    async def retrieve_mist_context(self):  # noqa: ANN201
        """Fetch MIST's seeded identity + outgoing trait/capability/preference edges.

        Returns a MistContext (backend.chat.mist_context) carrying traits,
        capabilities, preferences, and an as_system_prompt_block() renderer.

        Populated from the graph's mist-identity node via the structurally-scoped
        GraphStore.get_mist_identity_context() Cypher. Consumed by ConversationHandler
        for persona system-prompt injection.

        Traversal is anchored at the mist-identity node so HAS_TRAIT /
        HAS_CAPABILITY / HAS_PREFERENCE edges from user entities (if any
        existed) are never picked up here. ADR-009 :__Entity__-only rule
        honored at the Cypher layer.

        Import of MistContext types is deferred to avoid the circular import:
        backend.chat.__init__ re-exports ConversationHandler which imports
        KnowledgeRetriever; a top-level import here would form a cycle.
        """
        # Deferred import to break backend.chat <-> backend.knowledge.retrieval cycle.
        from backend.chat.mist_context import (  # noqa: PLC0415
            MistCapability,
            MistContext,
            MistPreference,
            MistTrait,
        )

        raw = self.graph_store.get_mist_identity_context()
        identity = raw["identity"]
        traits = [
            MistTrait(
                id=t["id"],
                display_name=t["display_name"],
                axis=t.get("axis", "Persona"),
                description=t.get("description", ""),
            )
            for t in raw.get("traits", [])
        ]
        capabilities = [
            MistCapability(
                id=c["id"],
                display_name=c["display_name"],
                description=c.get("description", ""),
            )
            for c in raw.get("capabilities", [])
        ]
        preferences = [
            MistPreference(
                id=p["id"],
                display_name=p["display_name"],
                enforcement=p.get("enforcement", "informational"),
                context=p.get("context", ""),
            )
            for p in raw.get("preferences", [])
        ]
        return MistContext(
            display_name=identity.get("display_name", "MIST"),
            pronouns=identity.get("pronouns", "she/her"),
            self_concept=identity.get("self_concept", "") or "",
            traits=traits,
            capabilities=capabilities,
            preferences=preferences,
        )

    # -- Retrieval backends ---------------------------------------------------

    async def _relational_retrieve(
        self,
        query: str,
        user_id: str,
        limit: int,
        similarity_threshold: float,
        max_hops: int,
        filters: RetrievalFilters | None,
    ) -> list[RetrievedFact]:
        """Retrieve facts via graph traversal (entity similarity + neighbourhood).

        This is the original retrieve() logic extracted into its own method.

        Args:
            query: User's natural language query.
            user_id: User entity ID.
            limit: Max results for vector entity search.
            similarity_threshold: Min similarity for entity matching.
            max_hops: Graph traversal depth.
            filters: Optional retrieval filters.

        Returns:
            Deduplicated, filtered list of RetrievedFact from the graph.
        """
        similar_entities = await self._vector_search(query, similarity_threshold, limit * 2)

        if not similar_entities:
            return []

        facts = await self._gather_facts(
            user_id=user_id,
            similar_entities=similar_entities,
            max_hops=max_hops,
            filters=filters,
        )

        return facts

    async def _factual_retrieve(
        self,
        query: str,
        embedding: list[float] | None,
        limit: int,
    ) -> list[RetrievedFact]:
        """Retrieve facts from vector store (document chunks).

        Converts VectorSearchResult objects into RetrievedFact with
        graph_distance=999 so they sort after direct graph matches.

        Args:
            query: User's natural language query.
            embedding: Pre-computed query embedding (may be None).
            limit: Max results to return.

        Returns:
            List of RetrievedFact mapped from vector search results.
        """
        if self._vector_store is None:
            return []

        if embedding is None and self._embedding_provider is not None:
            embedding = self._embedding_provider.generate_embedding(query)

        if embedding is None:
            logger.warning("No embedding available for factual retrieval")
            return []

        results = self._vector_store_search(embedding, limit)

        facts: list[RetrievedFact] = []
        for vsr in results:
            # Derive a display title from metadata or source_id
            title = vsr.metadata.get("title", vsr.source_id) if vsr.metadata else vsr.source_id
            fact = RetrievedFact(
                subject="Document",
                subject_type="DocumentChunk",
                predicate="CONTAINS",
                object=title,
                object_type=vsr.source_type,
                properties={
                    "chunk_id": vsr.chunk_id,
                    "text": vsr.text,
                    "source_id": vsr.source_id,
                },
                similarity_score=vsr.similarity,
                graph_distance=_VECTOR_DISTANCE_SENTINEL,
            )
            facts.append(fact)

        return facts

    def _vector_store_search(self, embedding: list[float], limit: int) -> list[VectorSearchResult]:
        """Execute raw vector store search with error handling.

        Args:
            embedding: Query vector.
            limit: Max results.

        Returns:
            List of VectorSearchResult, empty on failure.
        """
        if self._vector_store is None:
            return []

        try:
            return self._vector_store.search(query_embedding=embedding, limit=limit)
        except VectorStoreError as exc:
            logger.warning("Vector store search failed, degrading to graph-only: %s", exc)
            return []

    # -- RRF merge ------------------------------------------------------------

    @staticmethod
    def _merge_rrf(
        graph_facts: list[RetrievedFact],
        vector_facts: list[RetrievedFact],
        config: QueryIntentConfig,
    ) -> list[RetrievedFact]:
        """Merge two ranked lists using Reciprocal Rank Fusion.

        For each fact at rank r (1-based) in a list, its RRF score
        contribution is weight * (1 / (k + r)).  Facts appearing in
        both lists accumulate scores from both.  Deduplication uses
        the (subject, predicate, object) triple as key.

        Args:
            graph_facts: Ranked facts from graph retrieval.
            vector_facts: Ranked facts from vector retrieval.
            config: RRF parameters (k, weights).

        Returns:
            Merged list sorted by descending RRF score.
        """
        k = config.rrf_k
        scores: dict[tuple[str, str, str], float] = {}
        fact_map: dict[tuple[str, str, str], RetrievedFact] = {}

        for rank, fact in enumerate(graph_facts, start=1):
            key = (fact.subject, fact.predicate, fact.object)
            scores[key] = scores.get(key, 0.0) + config.rrf_graph_weight * (1.0 / (k + rank))
            if key not in fact_map:
                fact_map[key] = fact

        for rank, fact in enumerate(vector_facts, start=1):
            key = (fact.subject, fact.predicate, fact.object)
            scores[key] = scores.get(key, 0.0) + config.rrf_vector_weight * (1.0 / (k + rank))
            if key not in fact_map:
                fact_map[key] = fact

        sorted_keys = sorted(scores, key=lambda k_: scores[k_], reverse=True)
        return [fact_map[k_] for k_ in sorted_keys]

    # -- Live intent ----------------------------------------------------------

    @staticmethod
    def _map_live_tools(query: str) -> tuple[str, ...]:
        """Map query keywords to suggested MCP tool names.

        Args:
            query: User's natural language query.

        Returns:
            Tuple of MCP tool name strings matching keywords in the query.
        """
        query_lower = query.lower()
        tools: list[str] = []
        seen: set[str] = set()
        for keyword, tool in _LIVE_TOOL_MAP.items():
            if keyword in query_lower and tool not in seen:
                tools.append(tool)
                seen.add(tool)
        return tuple(tools)

    # -- Existing graph helpers (unchanged) -----------------------------------

    async def _vector_search(
        self, query: str, similarity_threshold: float, limit: int
    ) -> list[dict]:
        """Perform vector similarity search on graph entities.

        Returns list of entities sorted by similarity.
        """
        results = self.graph_store.search_similar_entities(
            query_text=query, limit=limit, similarity_threshold=similarity_threshold
        )

        return results

    @staticmethod
    def _is_provenance_id(node_id: str) -> bool:
        """Return True if *node_id* identifies a :__Provenance__ node.

        ADR-009 v1.1: provenance nodes must never appear in retrieval output.
        GraphStore.get_entity_neighborhood and get_user_relationships_to_entities
        enforce this at the Cypher level via :__Entity__-only path anchors.
        This guard provides a second line of defence at the retriever boundary.
        """
        return "__prov__" in node_id or "__Provenance__" in node_id

    async def _gather_facts(
        self,
        user_id: str,
        similar_entities: list[dict],
        max_hops: int,
        filters: RetrievalFilters | None,
    ) -> list[RetrievedFact]:
        """Gather facts from graph.

        Process:
        1. Get User's direct relationships to found entities
        2. Get neighborhood around each found entity
        3. Convert to RetrievedFact objects
        4. Apply filters
        """
        facts = []
        entity_ids = [e["entity_id"] for e in similar_entities]
        entity_similarity_map = {e["entity_id"]: e["similarity"] for e in similar_entities}

        # Get User's relationships to found entities
        user_rels = self.graph_store.get_user_relationships_to_entities(
            user_id=user_id,
            entity_ids=entity_ids,
            relationship_types=filters.relationship_types if filters else None,
        )

        for rel in user_rels:
            # ADR-009 v1.1: second-line defence — skip provenance IDs that
            # should never reach this point (GraphStore enforces at Cypher level).
            if self._is_provenance_id(rel["entity_id"]):
                logger.warning(
                    "Skipping provenance-shaped entity_id in user_rels: %s",
                    rel["entity_id"],
                )
                continue
            fact = RetrievedFact(
                subject=user_id,
                subject_type="Person",  # Assume User is Person
                predicate=rel["relationship_type"],
                object=rel["entity_id"],
                object_type=rel["entity_type"],
                properties=rel["properties"] or {},
                similarity_score=entity_similarity_map.get(rel["entity_id"], 0.0),
                graph_distance=0,  # Direct connection to User
                source_utterance_id=(
                    rel["properties"].get("created_from_utterance") if rel["properties"] else None
                ),
                created_at=rel["properties"].get("created_at") if rel["properties"] else None,
            )
            facts.append(fact)

        # Get neighborhood around each found entity (if max_hops > 1)
        if max_hops > 1:
            for entity in similar_entities[:10]:  # Limit to top 10 to avoid explosion
                neighborhood = self.graph_store.get_entity_neighborhood(
                    entity_id=entity["entity_id"],
                    max_hops=max_hops - 1,  # Already used 1 hop for User relationships
                    relationship_types=filters.relationship_types if filters else None,
                )

                for rel in neighborhood:
                    # ADR-009 v1.1: second-line defence on both path endpoints.
                    if self._is_provenance_id(rel["source"]) or self._is_provenance_id(
                        rel["target"]
                    ):
                        logger.warning(
                            "Skipping provenance-shaped node in neighborhood: source=%s target=%s",
                            rel["source"],
                            rel["target"],
                        )
                        continue
                    fact = RetrievedFact(
                        subject=rel["source"],
                        subject_type=rel["source_type"],
                        predicate=rel["relationship"],
                        object=rel["target"],
                        object_type=rel["target_type"],
                        properties=rel["properties"] or {},
                        similarity_score=entity_similarity_map.get(rel["source"], 0.0),
                        graph_distance=rel["path_length"],
                        source_utterance_id=(
                            rel["properties"].get("created_from_utterance")
                            if rel["properties"]
                            else None
                        ),
                        created_at=(
                            rel["properties"].get("created_at") if rel["properties"] else None
                        ),
                    )
                    facts.append(fact)

        # Apply filters
        if filters:
            facts = self._apply_filters(facts, filters)

        # Deduplicate (same subject-predicate-object)
        seen = set()
        unique_facts = []
        for fact in facts:
            key = (fact.subject, fact.predicate, fact.object)
            if key not in seen:
                seen.add(key)
                unique_facts.append(fact)

        return unique_facts

    def _apply_filters(
        self, facts: list[RetrievedFact], filters: RetrievalFilters
    ) -> list[RetrievedFact]:
        """Apply optional filters to facts."""
        filtered = facts

        if filters.entity_types:
            filtered = [
                f
                for f in filtered
                if f.subject_type in filters.entity_types or f.object_type in filters.entity_types
            ]

        if filters.exclude_entity_types:
            filtered = [
                f
                for f in filtered
                if f.subject_type not in filters.exclude_entity_types
                and f.object_type not in filters.exclude_entity_types
            ]

        if filters.relationship_types:
            filtered = [f for f in filtered if f.predicate in filters.relationship_types]

        if filters.exclude_relationship_types:
            filtered = [
                f for f in filtered if f.predicate not in filters.exclude_relationship_types
            ]

        if filters.min_similarity:
            filtered = [f for f in filtered if f.similarity_score >= filters.min_similarity]

        return filtered

    def _rank_facts(self, facts: list[RetrievedFact], limit: int) -> list[RetrievedFact]:
        """Rank facts by relevance.

        Sorting:
        1. Primary: Similarity score (higher = better)
        2. Secondary: Graph distance (lower = better, closer to User)
        """
        sorted_facts = sorted(facts, key=lambda f: (-f.similarity_score, f.graph_distance))

        return sorted_facts[:limit]

    def _format_context(self, facts: list[RetrievedFact], query: str) -> str:
        """Format facts as natural language context for LLM.

        Partitions facts into graph facts (graph_distance < 999) and
        document facts (graph_distance == 999) and formats each group
        with appropriate presentation.

        Returns formatted string ready for LLM consumption.
        """
        if not facts:
            return "No relevant knowledge found in the graph."

        # Partition into graph vs document facts
        graph_facts = [f for f in facts if f.graph_distance < _VECTOR_DISTANCE_SENTINEL]
        doc_facts = [f for f in facts if f.graph_distance >= _VECTOR_DISTANCE_SENTINEL]

        lines = [f"Relevant knowledge from your graph (query: '{query}'):", ""]

        # Format graph facts grouped by subject
        if graph_facts:
            facts_by_subject: dict[str, list[RetrievedFact]] = {}
            for fact in graph_facts:
                if fact.subject not in facts_by_subject:
                    facts_by_subject[fact.subject] = []
                facts_by_subject[fact.subject].append(fact)

            for subject, subject_facts in facts_by_subject.items():
                lines.append(f"### {subject}")

                for fact in subject_facts:
                    # Format relationship
                    rel_str = fact.predicate.replace("_", " ").lower()

                    # Format properties
                    prop_str = ""
                    if fact.properties:
                        relevant_props = {
                            k: v
                            for k, v in fact.properties.items()
                            if k
                            not in [
                                "created_at",
                                "ontology_version",
                                "created_from_utterance",
                                "embedding",
                            ]
                        }
                        if relevant_props:
                            prop_parts = [f"{k}={v}" for k, v in relevant_props.items()]
                            prop_str = f" [{', '.join(prop_parts)}]"

                    # Format line
                    lines.append(f"  - {rel_str} {fact.object} ({fact.object_type}){prop_str}")

                lines.append("")  # Blank line between subjects

        # Format document facts
        if doc_facts:
            lines.append("### Relevant Documents")
            for idx, fact in enumerate(doc_facts, start=1):
                title = fact.object
                score = fact.similarity_score
                chunk_text = fact.properties.get("text", "")
                truncated = chunk_text[:500] if len(chunk_text) > 500 else chunk_text
                lines.append(f"[doc-{idx}] Source: {title} (similarity: {score:.2f})")
                lines.append(f"    {truncated}")
            lines.append("")

        # Add metadata
        lines.append("---")
        lines.append(f"Total facts: {len(facts)}")

        return "\n".join(lines)

    async def search_documents(
        self, query: str, limit: int = 5, similarity_threshold: float = 0.7
    ) -> list[dict[str, Any]]:
        """Search DocumentChunks using vector similarity.

        This is RAG retrieval - finds relevant document chunks based on
        semantic similarity to the query.

        Args:
            query: Search query
            limit: Maximum chunks to return
            similarity_threshold: Minimum similarity score (0.0-1.0)

        Returns:
            List of document chunks with metadata:
            [
                {
                    "chunk_id": "...",
                    "text": "...",
                    "similarity": 0.85,
                    "source_title": "...",
                    "source_file": "...",
                    "position": 0
                }
            ]
        """
        logger.info(
            f"Searching documents for: '{query}' (limit={limit}, threshold={similarity_threshold})"
        )

        # Use graph store's vector search on DocumentChunks
        results = self.graph_store.search_document_chunks(
            query_text=query, limit=limit, similarity_threshold=similarity_threshold
        )

        logger.info(f"Found {len(results)} document chunks")

        return results
