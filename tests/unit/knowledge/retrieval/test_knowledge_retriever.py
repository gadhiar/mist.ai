"""Unit tests for KnowledgeRetriever.

Covers intent-based routing, RRF merge logic, graceful fallback when the
vector store is absent or raises, and context formatting.
"""

import pytest

from backend.errors import VectorStoreError
from backend.knowledge.config import KnowledgeConfig, QueryIntentConfig
from backend.knowledge.models import QueryIntent, RetrievedFact, VectorSearchResult
from backend.knowledge.retrieval.knowledge_retriever import (
    _VECTOR_DISTANCE_SENTINEL,
    KnowledgeRetriever,
)
from backend.knowledge.storage.graph_store import GraphStore
from tests.mocks.config import build_test_config
from tests.mocks.neo4j import FakeNeo4jConnection
from tests.unit.knowledge.conftest import FakeEmbeddingProvider, FakeVectorStore

MODULE = "backend.knowledge.retrieval.knowledge_retriever"


# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------


def build_retrieved_fact(
    *,
    subject: str = "User",
    subject_type: str = "Person",
    predicate: str = "USES",
    object: str = "Python",
    object_type: str = "Technology",
    properties: dict | None = None,
    similarity_score: float = 0.9,
    graph_distance: int = 0,
) -> RetrievedFact:
    """Build a valid RetrievedFact with overridable fields."""
    return RetrievedFact(
        subject=subject,
        subject_type=subject_type,
        predicate=predicate,
        object=object,
        object_type=object_type,
        properties=properties or {},
        similarity_score=similarity_score,
        graph_distance=graph_distance,
    )


def build_vector_search_result(
    *,
    chunk_id: str = "chunk-001",
    text: str = "Python is great for data science.",
    similarity: float = 0.85,
    source_id: str = "src-001",
    source_type: str = "markdown",
    metadata: dict | None = None,
) -> VectorSearchResult:
    """Build a valid VectorSearchResult with overridable fields."""
    return VectorSearchResult(
        chunk_id=chunk_id,
        text=text,
        similarity=similarity,
        source_id=source_id,
        source_type=source_type,
        metadata=metadata or {},
    )


class StubQueryClassifier:
    """Minimal QueryClassifier replacement that returns a fixed intent."""

    def __init__(self, intent: str) -> None:
        self._intent = intent
        self.calls: list[str] = []

    def classify(self, query: str) -> QueryIntent:
        self.calls.append(query)
        store_map: dict[str, tuple[str, ...]] = {
            "factual": ("vector",),
            "relational": ("graph",),
            "hybrid": ("vector", "graph"),
            "live": ("mcp",),
            "identity": ("mist",),
        }
        return QueryIntent(
            intent=self._intent,
            confidence=0.9,
            suggested_stores=store_map[self._intent],
        )


def _make_retriever(
    *,
    intent: str | None = None,
    vector_store: FakeVectorStore | None = None,
    fake_connection: FakeNeo4jConnection | None = None,
    embedding_provider: FakeEmbeddingProvider | None = None,
    config: KnowledgeConfig | None = None,
    search_similar_results: list[dict] | None = None,
    user_rels_results: list[dict] | None = None,
    neighborhood_results: list[dict] | None = None,
) -> tuple[KnowledgeRetriever, FakeNeo4jConnection]:
    """Build a KnowledgeRetriever wired with test doubles.

    Patches graph_store methods on the instance level so FakeNeo4jConnection
    is used for the connection but individual GraphStore calls are controlled
    via simple replacement callables.
    """
    conn = fake_connection or FakeNeo4jConnection()
    emb = embedding_provider or FakeEmbeddingProvider()
    cfg = config or build_test_config()

    graph_store = GraphStore(connection=conn, embedding_generator=emb)

    # Replace graph_store methods with deterministic stubs.
    _similar = search_similar_results if search_similar_results is not None else []
    _user_rels = user_rels_results if user_rels_results is not None else []
    _neighborhood = neighborhood_results if neighborhood_results is not None else []

    graph_store.search_similar_entities = lambda **kwargs: _similar  # type: ignore[method-assign]
    graph_store.get_user_relationships_to_entities = lambda **kwargs: _user_rels  # type: ignore[method-assign]
    graph_store.get_entity_neighborhood = lambda **kwargs: _neighborhood  # type: ignore[method-assign]

    classifier = StubQueryClassifier(intent) if intent is not None else None

    retriever = KnowledgeRetriever(
        config=cfg,
        graph_store=graph_store,
        vector_store=vector_store,
        query_classifier=classifier,
        embedding_provider=emb,
    )
    return retriever, conn


# ---------------------------------------------------------------------------
# TestRetrieveRouting
# ---------------------------------------------------------------------------


class TestRetrieveRouting:
    """retrieve() routes to the correct backend(s) based on classified intent."""

    @pytest.mark.asyncio
    async def test_factual_intent_calls_vector_store_not_graph(self):
        # Arrange
        vector_store = FakeVectorStore()
        retriever, _ = _make_retriever(
            intent="factual",
            vector_store=vector_store,
            search_similar_results=[],  # graph search should not be called
        )

        # Act
        result = await retriever.retrieve(query="What is event sourcing?", user_id="User")

        # Assert -- vector store was consulted
        vector_store.assert_called("search")
        # graph traversal time must be zero (graph path not taken)
        assert result.graph_traversal_time_ms == 0.0
        assert result.intent == "factual"

    @pytest.mark.asyncio
    async def test_relational_intent_calls_graph_not_vector_store(self):
        # Arrange
        vector_store = FakeVectorStore()
        retriever, _ = _make_retriever(
            intent="relational",
            vector_store=vector_store,
            search_similar_results=[],
        )

        # Act
        result = await retriever.retrieve(query="What do I use at work?", user_id="User")

        # Assert -- vector store search was NOT called
        vector_store.assert_not_called("search")
        assert result.intent == "relational"

    @pytest.mark.asyncio
    async def test_hybrid_intent_calls_both_backends(self):
        # Arrange
        vector_store = FakeVectorStore()
        similar_entity = {
            "entity_id": "python",
            "entity_type": "Technology",
            "similarity": 0.9,
            "name": "Python",
        }
        retriever, _ = _make_retriever(
            intent="hybrid",
            vector_store=vector_store,
            search_similar_results=[similar_entity],
            user_rels_results=[
                {
                    "relationship_type": "USES",
                    "entity_id": "python",
                    "entity_type": "Technology",
                    "properties": {},
                }
            ],
        )

        # Act
        result = await retriever.retrieve(query="What Python concepts do I know?", user_id="User")

        # Assert -- both backends consulted: vector store was searched and
        # graph facts are present in the result. Timing may be 0.0ms because
        # stubs return instantly, so check intent and fact content instead.
        vector_store.assert_called("search")
        assert result.intent == "hybrid"
        assert result.total_facts >= 1

    @pytest.mark.asyncio
    async def test_live_intent_returns_mcp_suggestions_without_retrieval(self):
        # Arrange
        vector_store = FakeVectorStore()
        retriever, _ = _make_retriever(
            intent="live",
            vector_store=vector_store,
        )

        # Act
        result = await retriever.retrieve(
            query="What Linear tickets are open right now?", user_id="User"
        )

        # Assert -- no retrieval backends touched
        vector_store.assert_not_called("search")
        assert result.requires_mcp is True
        assert result.intent == "live"
        assert result.facts == []
        assert result.total_facts == 0

    @pytest.mark.asyncio
    async def test_live_intent_maps_linear_keyword_to_tool(self):
        # Arrange
        retriever, _ = _make_retriever(intent="live")

        # Act
        result = await retriever.retrieve(
            query="Show open linear issues for this sprint", user_id="User"
        )

        # Assert
        assert result.requires_mcp is True
        assert any("linear" in tool for tool in result.suggested_tools)

    @pytest.mark.asyncio
    async def test_no_classifier_defaults_to_relational_intent(self):
        # Arrange -- pass no classifier (None)
        vector_store = FakeVectorStore()
        retriever, _ = _make_retriever(
            intent=None,  # no classifier injected
            vector_store=vector_store,
            search_similar_results=[],
        )

        # Act
        result = await retriever.retrieve(query="Anything at all", user_id="User")

        # Assert -- defaults to relational path, vector store not called
        vector_store.assert_not_called("search")
        assert result.intent == "relational"


# ---------------------------------------------------------------------------
# TestMergeRRF
# ---------------------------------------------------------------------------


class TestMergeRRF:
    """_merge_rrf() combines two ranked lists with correct deduplication and weighting."""

    def _default_config(self) -> QueryIntentConfig:
        return QueryIntentConfig(
            rrf_k=60,
            rrf_graph_weight=0.5,
            rrf_vector_weight=0.5,
        )

    def test_deduplicates_by_subject_predicate_object_key(self):
        # Arrange -- same triple appears in both lists
        shared = build_retrieved_fact(subject="User", predicate="USES", object="Python")
        graph_facts = [shared]
        vector_facts = [shared]

        # Act
        merged = KnowledgeRetriever._merge_rrf(graph_facts, vector_facts, self._default_config())

        # Assert -- exactly one entry in output
        assert len(merged) == 1
        assert merged[0].subject == "User"
        assert merged[0].predicate == "USES"
        assert merged[0].object == "Python"

    def test_shared_fact_accumulates_score_from_both_lists(self):
        # Arrange
        shared = build_retrieved_fact(subject="User", predicate="USES", object="Python")
        unique_graph = build_retrieved_fact(subject="User", predicate="KNOWS", object="Alice")
        unique_vector = build_retrieved_fact(
            subject="Document", predicate="CONTAINS", object="FastAPI"
        )

        graph_facts = [shared, unique_graph]
        vector_facts = [shared, unique_vector]
        config = self._default_config()

        # Act
        merged = KnowledgeRetriever._merge_rrf(graph_facts, vector_facts, config)

        # Assert -- shared fact is first (highest combined score)
        assert merged[0].subject == "User"
        assert merged[0].predicate == "USES"
        assert merged[0].object == "Python"

    def test_graph_weight_higher_promotes_graph_only_fact(self):
        # Arrange -- graph weight >> vector weight
        config = QueryIntentConfig(rrf_k=60, rrf_graph_weight=1.0, rrf_vector_weight=0.01)
        graph_only = build_retrieved_fact(subject="User", predicate="EXPERT_IN", object="Neo4j")
        vector_only = build_retrieved_fact(
            subject="Document", predicate="CONTAINS", object="LanceDB"
        )

        graph_facts = [graph_only]
        vector_facts = [vector_only]

        # Act
        merged = KnowledgeRetriever._merge_rrf(graph_facts, vector_facts, config)

        # Assert -- graph fact ranks higher
        assert merged[0].object == "Neo4j"

    def test_empty_graph_facts_returns_vector_only(self):
        # Arrange
        vector_fact = build_retrieved_fact(subject="Document", predicate="CONTAINS", object="RRF")
        config = self._default_config()

        # Act
        merged = KnowledgeRetriever._merge_rrf([], [vector_fact], config)

        # Assert
        assert len(merged) == 1
        assert merged[0].object == "RRF"

    def test_empty_vector_facts_returns_graph_only(self):
        # Arrange
        graph_fact = build_retrieved_fact(subject="User", predicate="USES", object="Python")
        config = self._default_config()

        # Act
        merged = KnowledgeRetriever._merge_rrf([graph_fact], [], config)

        # Assert
        assert len(merged) == 1
        assert merged[0].object == "Python"

    def test_both_empty_returns_empty_list(self):
        # Act
        merged = KnowledgeRetriever._merge_rrf([], [], self._default_config())

        # Assert
        assert merged == []

    def test_rank_order_preserved_for_distinct_facts(self):
        # Arrange -- three distinct graph facts; first should outscore third
        f1 = build_retrieved_fact(subject="User", predicate="USES", object="Python")
        f2 = build_retrieved_fact(subject="User", predicate="USES", object="FastAPI")
        f3 = build_retrieved_fact(subject="User", predicate="USES", object="Docker")
        config = self._default_config()

        # Act -- graph only (vector empty means pure graph ordering preserved)
        merged = KnowledgeRetriever._merge_rrf([f1, f2, f3], [], config)

        # Assert -- order matches input rank (rank 1 > rank 2 > rank 3)
        objects = [f.object for f in merged]
        assert objects == ["Python", "FastAPI", "Docker"]


# ---------------------------------------------------------------------------
# TestGracefulFallback
# ---------------------------------------------------------------------------


class TestGracefulFallback:
    """Retriever degrades cleanly when the vector store is missing or fails."""

    @pytest.mark.asyncio
    async def test_none_vector_store_uses_graph_only_for_factual_intent(self):
        # Arrange -- no vector store injected
        retriever, _ = _make_retriever(
            intent="factual",
            vector_store=None,
            search_similar_results=[],
        )

        # Act
        result = await retriever.retrieve(query="What is dependency injection?", user_id="User")

        # Assert -- still returns a valid result, not an exception
        assert result.facts == []
        assert result.intent == "factual"
        assert result.vector_search_time_ms == 0.0

    @pytest.mark.asyncio
    async def test_none_vector_store_relational_query_returns_facts_from_graph(self):
        # Arrange -- vector_store=None; graph returns a relationship
        similar_entity = {
            "entity_id": "python",
            "entity_type": "Technology",
            "similarity": 0.88,
            "name": "Python",
        }
        retriever, _ = _make_retriever(
            intent="relational",
            vector_store=None,
            search_similar_results=[similar_entity],
            user_rels_results=[
                {
                    "relationship_type": "USES",
                    "entity_id": "python",
                    "entity_type": "Technology",
                    "properties": {},
                }
            ],
        )

        # Act
        result = await retriever.retrieve(query="What do I use for scripting?", user_id="User")

        # Assert
        assert len(result.facts) == 1
        assert result.facts[0].object == "python"

    @pytest.mark.asyncio
    async def test_vector_store_error_falls_back_gracefully(self):
        # Arrange -- vector_store.search() raises VectorStoreError
        class ErroringVectorStore(FakeVectorStore):
            def search(self, query_embedding, limit, filters=None):
                raise VectorStoreError("LanceDB unavailable")

        retriever, _ = _make_retriever(
            intent="factual",
            vector_store=ErroringVectorStore(),
            search_similar_results=[],
        )

        # Act -- must not raise
        result = await retriever.retrieve(
            query="What is the knowledge graph architecture?", user_id="User"
        )

        # Assert -- graceful empty result, not an unhandled exception
        assert result.facts == []
        assert result.intent == "factual"

    @pytest.mark.asyncio
    async def test_hybrid_with_vector_store_error_still_returns_graph_facts(self):
        # Arrange -- vector store errors; graph returns a fact
        class ErroringVectorStore(FakeVectorStore):
            def search(self, query_embedding, limit, filters=None):
                raise VectorStoreError("backend down")

        similar_entity = {
            "entity_id": "neo4j",
            "entity_type": "Technology",
            "similarity": 0.92,
            "name": "Neo4j",
        }
        retriever, _ = _make_retriever(
            intent="hybrid",
            vector_store=ErroringVectorStore(),
            search_similar_results=[similar_entity],
            user_rels_results=[
                {
                    "relationship_type": "USES",
                    "entity_id": "neo4j",
                    "entity_type": "Technology",
                    "properties": {},
                }
            ],
        )

        # Act
        result = await retriever.retrieve(
            query="What graph database concepts do I know?", user_id="User"
        )

        # Assert -- graph facts present despite vector store failure
        assert len(result.facts) >= 1
        assert result.facts[0].object == "neo4j"


# ---------------------------------------------------------------------------
# TestFormatContext
# ---------------------------------------------------------------------------


class TestFormatContext:
    """_format_context() produces LLM-ready strings from retrieved facts."""

    def _retriever(self) -> KnowledgeRetriever:
        """Build a retriever instance for direct method testing."""
        conn = FakeNeo4jConnection()
        emb = FakeEmbeddingProvider()
        graph_store = GraphStore(connection=conn, embedding_generator=emb)
        graph_store.search_similar_entities = lambda **kwargs: []  # type: ignore[method-assign]
        graph_store.get_user_relationships_to_entities = lambda **kwargs: []  # type: ignore[method-assign]
        graph_store.get_entity_neighborhood = lambda **kwargs: []  # type: ignore[method-assign]
        return KnowledgeRetriever(
            config=build_test_config(),
            graph_store=graph_store,
        )

    def test_empty_facts_returns_no_knowledge_found_message(self):
        # Arrange
        retriever = self._retriever()

        # Act
        context = retriever._format_context([], "What do I know about Python?")

        # Assert
        assert "No relevant knowledge found" in context

    def test_graph_facts_appear_under_subject_heading(self):
        # Arrange
        retriever = self._retriever()
        facts = [
            build_retrieved_fact(
                subject="User",
                predicate="USES",
                object="Python",
                object_type="Technology",
                graph_distance=0,
            )
        ]

        # Act
        context = retriever._format_context(facts, "Python usage")

        # Assert
        assert "### User" in context
        assert "Python" in context
        assert "uses" in context  # predicate formatted lowercase

    def test_document_facts_appear_under_relevant_documents_heading(self):
        # Arrange
        retriever = self._retriever()
        doc_fact = build_retrieved_fact(
            subject="Document",
            subject_type="DocumentChunk",
            predicate="CONTAINS",
            object="Architecture Guide",
            object_type="markdown",
            properties={"text": "The system uses Neo4j.", "chunk_id": "c1", "source_id": "s1"},
            similarity_score=0.88,
            graph_distance=_VECTOR_DISTANCE_SENTINEL,
        )

        # Act
        context = retriever._format_context([doc_fact], "architecture")

        # Assert
        assert "### Relevant Documents" in context
        assert "Architecture Guide" in context

    def test_properties_excluded_from_formatting(self):
        # Arrange -- created_at, embedding, and ontology_version should not appear
        retriever = self._retriever()
        facts = [
            build_retrieved_fact(
                subject="User",
                predicate="USES",
                object="Python",
                properties={
                    "created_at": "2026-01-01",
                    "embedding": [0.1, 0.2],
                    "ontology_version": "1.0.0",
                    "proficiency": "expert",
                },
                graph_distance=0,
            )
        ]

        # Act
        context = retriever._format_context(facts, "skills")

        # Assert -- excluded keys not present, allowed key present
        assert "created_at" not in context
        assert "embedding" not in context
        assert "ontology_version" not in context
        assert "proficiency=expert" in context

    def test_total_facts_count_present_in_output(self):
        # Arrange
        retriever = self._retriever()
        facts = [
            build_retrieved_fact(subject="User", predicate="USES", object="Python"),
            build_retrieved_fact(subject="User", predicate="KNOWS", object="Alice"),
        ]

        # Act
        context = retriever._format_context(facts, "query")

        # Assert
        assert "Total facts: 2" in context

    def test_query_text_appears_in_header(self):
        # Arrange
        retriever = self._retriever()
        facts = [build_retrieved_fact()]

        # Act
        context = retriever._format_context(facts, "my Python skills")

        # Assert
        assert "my Python skills" in context


# ---------------------------------------------------------------------------
# TestRetrieveMistContext
# ---------------------------------------------------------------------------

_SEEDED_IDENTITY_RAW = {
    "identity": {
        "id": "mist-identity",
        "display_name": "MIST",
        "pronouns": "she/her",
        "self_concept": "A cognitive architecture for personal knowledge.",
    },
    "traits": [
        {"id": "trait-warm", "display_name": "Warm", "axis": "Persona", "description": "Friendly."},
    ],
    "capabilities": [
        {"id": "cap-tool-use", "display_name": "Tool use", "description": "MCP tools."},
    ],
    "preferences": [
        {
            "id": "pref-no-emoji",
            "display_name": "No emoji",
            "enforcement": "absolute",
            "context": "Hard rule.",
        },
        {
            "id": "pref-no-ai-slop",
            "display_name": "No slop",
            "enforcement": "absolute",
            "context": "Hard rule.",
        },
    ],
}


def _make_identity_retriever() -> KnowledgeRetriever:
    """Build a KnowledgeRetriever wired for identity-intent retrieval tests."""
    from unittest.mock import MagicMock

    conn = FakeNeo4jConnection()
    emb = FakeEmbeddingProvider()
    graph_store = GraphStore(connection=conn, embedding_generator=emb)
    # Replace with a sync stub — get_mist_identity_context is sync on GraphStore.
    graph_store.get_mist_identity_context = MagicMock(return_value=_SEEDED_IDENTITY_RAW)  # type: ignore[method-assign]

    classifier = StubQueryClassifier("identity")

    return KnowledgeRetriever(
        config=build_test_config(),
        graph_store=graph_store,
        vector_store=None,
        query_classifier=classifier,
        embedding_provider=emb,
    )


class TestRetrieveMistContext:
    """Cluster 3: retrieve_mist_context pulls MistIdentity + edges; identity intent routes here."""

    @pytest.mark.asyncio
    async def test_returns_mist_context_object(self):
        from backend.chat.mist_context import MistContext

        retriever = _make_identity_retriever()
        ctx = await retriever.retrieve_mist_context()
        assert isinstance(ctx, MistContext)
        assert ctx.display_name == "MIST"
        assert ctx.pronouns == "she/her"

    @pytest.mark.asyncio
    async def test_includes_all_traits(self):
        retriever = _make_identity_retriever()
        ctx = await retriever.retrieve_mist_context()
        trait_names = {t.display_name for t in ctx.traits}
        assert trait_names == {"Warm"}

    @pytest.mark.asyncio
    async def test_includes_absolute_preferences(self):
        retriever = _make_identity_retriever()
        ctx = await retriever.retrieve_mist_context()
        pref_ids = {p.id for p in ctx.preferences}
        assert "pref-no-emoji" in pref_ids
        assert "pref-no-ai-slop" in pref_ids

    @pytest.mark.asyncio
    async def test_identity_intent_routes_to_mist_context_retrieval(self):
        """retrieve() with identity-classified query returns formatted persona block."""
        retriever = _make_identity_retriever()
        result = await retriever.retrieve(
            query="what are your preferences?",
            user_id="User",
        )
        assert result.intent == "identity"
        # Formatted context should contain persona content.
        assert "MIST" in result.formatted_context
        assert "No emoji" in result.formatted_context
        assert "No slop" in result.formatted_context
        # No facts list for identity intent — persona surfaces via formatted_context.
        assert result.facts == []

    @pytest.mark.asyncio
    async def test_identity_intent_does_not_require_mcp(self):
        retriever = _make_identity_retriever()
        result = await retriever.retrieve(query="who are you?", user_id="User")
        assert result.requires_mcp is False
