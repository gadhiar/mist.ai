"""
Knowledge Retriever

Hybrid retrieval system combining vector search with graph traversal.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import time

from backend.knowledge.config import KnowledgeConfig
from backend.knowledge.storage import GraphStore
from backend.knowledge.models import (
    RetrievalFilters,
    RetrievedFact,
    RetrievalResult
)

logger = logging.getLogger(__name__)


class KnowledgeRetriever:
    """
    Retrieves relevant knowledge from graph for answering questions

    Combines:
    - Vector similarity search (find relevant entities)
    - Graph traversal (expand context)
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

    def __init__(self, config: KnowledgeConfig, graph_store: GraphStore):
        """
        Initialize retriever

        Args:
            config: Knowledge system configuration
            graph_store: Graph storage instance
        """
        self.config = config
        self.graph_store = graph_store

        # Default retrieval parameters (can be overridden)
        self.default_limit = 20
        self.default_similarity_threshold = 0.6
        self.default_max_hops = 2

        logger.info("KnowledgeRetriever initialized")

    async def retrieve(
        self,
        query: str,
        user_id: str = "User",
        limit: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        max_hops: Optional[int] = None,
        filters: Optional[RetrievalFilters] = None
    ) -> RetrievalResult:
        """
        Retrieve relevant knowledge for a query

        Process:
        1. Vector search for relevant entities
        2. Get User's relationships to found entities
        3. Expand graph context (N hops from found entities)
        4. Apply filters (if specified)
        5. Rank and score results
        6. Format context for LLM

        Args:
            query: User's question or search query
            user_id: User entity ID (default "User")
            limit: Max facts to return (default from config)
            similarity_threshold: Min similarity for vector search (default 0.6)
            max_hops: Graph traversal depth (default 2)
            filters: Optional filters for retrieval

        Returns:
            RetrievalResult with facts and formatted context
        """
        start_time = time.time()

        # Use defaults if not specified
        limit = limit or self.default_limit
        similarity_threshold = similarity_threshold or self.default_similarity_threshold
        max_hops = max_hops or self.default_max_hops

        logger.info(f"Retrieving knowledge for query: '{query}'")
        logger.info(f"Parameters: limit={limit}, threshold={similarity_threshold}, max_hops={max_hops}")

        # Step 1: Vector search
        vector_start = time.time()
        similar_entities = await self._vector_search(query, similarity_threshold, limit * 2)
        vector_time = (time.time() - vector_start) * 1000

        logger.info(f"Vector search found {len(similar_entities)} entities")

        if not similar_entities:
            # No entities found
            return RetrievalResult(
                query=query,
                user_id=user_id,
                facts=[],
                entities_found=0,
                total_facts=0,
                formatted_context="No relevant knowledge found in the graph.",
                retrieval_time_ms=(time.time() - start_time) * 1000,
                vector_search_time_ms=vector_time,
                graph_traversal_time_ms=0,
                config_used={
                    "limit": limit,
                    "similarity_threshold": similarity_threshold,
                    "max_hops": max_hops
                }
            )

        # Step 2 & 3: Graph traversal
        graph_start = time.time()
        facts = await self._gather_facts(
            user_id=user_id,
            similar_entities=similar_entities,
            max_hops=max_hops,
            filters=filters
        )
        graph_time = (time.time() - graph_start) * 1000

        logger.info(f"Graph traversal gathered {len(facts)} facts")

        # Step 4: Rank and limit
        ranked_facts = self._rank_facts(facts, limit)

        # Step 5: Format context
        formatted_context = self._format_context(ranked_facts, query)

        total_time = (time.time() - start_time) * 1000

        result = RetrievalResult(
            query=query,
            user_id=user_id,
            facts=ranked_facts,
            entities_found=len(similar_entities),
            total_facts=len(ranked_facts),
            formatted_context=formatted_context,
            retrieval_time_ms=total_time,
            vector_search_time_ms=vector_time,
            graph_traversal_time_ms=graph_time,
            config_used={
                "limit": limit,
                "similarity_threshold": similarity_threshold,
                "max_hops": max_hops,
                "filters": filters
            }
        )

        logger.info(f"Retrieval complete: {result.summary()}")

        return result

    async def _vector_search(
        self,
        query: str,
        similarity_threshold: float,
        limit: int
    ) -> List[Dict]:
        """
        Perform vector similarity search

        Returns list of entities sorted by similarity
        """
        results = self.graph_store.search_similar_entities(
            query_text=query,
            limit=limit,
            similarity_threshold=similarity_threshold
        )

        return results

    async def _gather_facts(
        self,
        user_id: str,
        similar_entities: List[Dict],
        max_hops: int,
        filters: Optional[RetrievalFilters]
    ) -> List[RetrievedFact]:
        """
        Gather facts from graph

        Process:
        1. Get User's direct relationships to found entities
        2. Get neighborhood around each found entity
        3. Convert to RetrievedFact objects
        4. Apply filters
        """
        facts = []
        entity_ids = [e['entity_id'] for e in similar_entities]
        entity_similarity_map = {e['entity_id']: e['similarity'] for e in similar_entities}

        # Get User's relationships to found entities
        user_rels = self.graph_store.get_user_relationships_to_entities(
            user_id=user_id,
            entity_ids=entity_ids,
            relationship_types=filters.relationship_types if filters else None
        )

        for rel in user_rels:
            fact = RetrievedFact(
                subject=user_id,
                subject_type="Person",  # Assume User is Person
                predicate=rel['relationship_type'],
                object=rel['entity_id'],
                object_type=rel['entity_type'],
                properties=rel['properties'] or {},
                similarity_score=entity_similarity_map.get(rel['entity_id'], 0.0),
                graph_distance=0,  # Direct connection to User
                source_utterance_id=rel['properties'].get('created_from_utterance') if rel['properties'] else None,
                created_at=rel['properties'].get('created_at') if rel['properties'] else None
            )
            facts.append(fact)

        # Get neighborhood around each found entity (if max_hops > 1)
        if max_hops > 1:
            for entity in similar_entities[:10]:  # Limit to top 10 to avoid explosion
                neighborhood = self.graph_store.get_entity_neighborhood(
                    entity_id=entity['entity_id'],
                    max_hops=max_hops - 1,  # Already used 1 hop for User relationships
                    relationship_types=filters.relationship_types if filters else None
                )

                for rel in neighborhood:
                    fact = RetrievedFact(
                        subject=rel['source'],
                        subject_type=rel['source_type'],
                        predicate=rel['relationship'],
                        object=rel['target'],
                        object_type=rel['target_type'],
                        properties=rel['properties'] or {},
                        similarity_score=entity_similarity_map.get(rel['source'], 0.0),
                        graph_distance=rel['path_length'],
                        source_utterance_id=rel['properties'].get('created_from_utterance') if rel['properties'] else None,
                        created_at=rel['properties'].get('created_at') if rel['properties'] else None
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
        self,
        facts: List[RetrievedFact],
        filters: RetrievalFilters
    ) -> List[RetrievedFact]:
        """Apply optional filters to facts"""
        filtered = facts

        if filters.entity_types:
            filtered = [
                f for f in filtered
                if f.subject_type in filters.entity_types or f.object_type in filters.entity_types
            ]

        if filters.exclude_entity_types:
            filtered = [
                f for f in filtered
                if f.subject_type not in filters.exclude_entity_types
                and f.object_type not in filters.exclude_entity_types
            ]

        if filters.relationship_types:
            filtered = [f for f in filtered if f.predicate in filters.relationship_types]

        if filters.exclude_relationship_types:
            filtered = [f for f in filtered if f.predicate not in filters.exclude_relationship_types]

        if filters.min_similarity:
            filtered = [f for f in filtered if f.similarity_score >= filters.min_similarity]

        return filtered

    def _rank_facts(self, facts: List[RetrievedFact], limit: int) -> List[RetrievedFact]:
        """
        Rank facts by relevance

        Sorting:
        1. Primary: Similarity score (higher = better)
        2. Secondary: Graph distance (lower = better, closer to User)
        """
        sorted_facts = sorted(
            facts,
            key=lambda f: (-f.similarity_score, f.graph_distance)
        )

        return sorted_facts[:limit]

    def _format_context(self, facts: List[RetrievedFact], query: str) -> str:
        """
        Format facts as natural language context for LLM

        Returns formatted string ready for LLM consumption
        """
        if not facts:
            return "No relevant knowledge found in the graph."

        lines = [
            f"Relevant knowledge from your graph (query: '{query}'):",
            ""
        ]

        # Group facts by subject for better readability
        facts_by_subject = {}
        for fact in facts:
            if fact.subject not in facts_by_subject:
                facts_by_subject[fact.subject] = []
            facts_by_subject[fact.subject].append(fact)

        for subject, subject_facts in facts_by_subject.items():
            lines.append(f"### {subject}")

            for fact in subject_facts:
                # Format relationship
                rel_str = fact.predicate.replace('_', ' ').lower()

                # Format properties
                prop_str = ""
                if fact.properties:
                    relevant_props = {
                        k: v for k, v in fact.properties.items()
                        if k not in ['created_at', 'ontology_version', 'created_from_utterance', 'embedding']
                    }
                    if relevant_props:
                        prop_parts = [f"{k}={v}" for k, v in relevant_props.items()]
                        prop_str = f" [{', '.join(prop_parts)}]"

                # Format line
                lines.append(f"  - {rel_str} {fact.object} ({fact.object_type}){prop_str}")

            lines.append("")  # Blank line between subjects

        # Add metadata
        lines.append("---")
        lines.append(f"Total facts: {len(facts)}")

        return "\n".join(lines)
