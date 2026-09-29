"""Regression tests for the user-node id used by the retriever's graph leg.

Root cause pinned here: the stored user node is `(:__Entity__ {id: "user"})`
(writers: `backend/knowledge/extraction/prompts.py` `{"id": "user", ...}` and
`backend/knowledge/curation/skill_derivation.py` `MERGE (u:__Entity__ {id: ...})`),
but the retriever defaulted to `"User"`. Neo4j property matching is
case-sensitive, so `GraphStore.get_user_relationships_to_entities`
(`MATCH (user:__Entity__ {id: $user_id})`) matched nothing.
"""

import inspect

import pytest

from backend.knowledge.retrieval.knowledge_retriever import KnowledgeRetriever
from backend.knowledge.storage.graph_store import GraphStore
from backend.knowledge.storage.partitions import USER_ENTITY_ID
from tests.mocks.config import build_test_config
from tests.mocks.neo4j import FakeNeo4jConnection
from tests.unit.knowledge.conftest import FakeEmbeddingProvider


def _retriever_with_recording_graph_store() -> tuple[KnowledgeRetriever, list[dict]]:
    """Real retriever over a GraphStore whose reads are recording stubs.

    No classifier is supplied, so retrieve() takes the relational (graph) leg.
    """
    user_rel_calls: list[dict] = []
    emb = FakeEmbeddingProvider()
    graph_store = GraphStore(connection=FakeNeo4jConnection(), embedding_generator=emb)

    graph_store.search_similar_entities = lambda **kwargs: [  # type: ignore[method-assign]
        {"entity_id": "python", "entity_type": "Technology", "similarity": 0.9}
    ]

    def _user_rels(**kwargs):
        user_rel_calls.append(kwargs)
        return []

    graph_store.get_user_relationships_to_entities = _user_rels  # type: ignore[method-assign]
    graph_store.get_entity_neighborhood = lambda **kwargs: []  # type: ignore[method-assign]

    retriever = KnowledgeRetriever(
        config=build_test_config(), graph_store=graph_store, embedding_provider=emb
    )
    return retriever, user_rel_calls


class TestUserEntityId:
    def test_canonical_constant_is_the_stored_lowercase_id(self):
        assert USER_ENTITY_ID == "user"

    def test_retrieve_default_user_id_is_the_canonical_constant(self):
        default = inspect.signature(KnowledgeRetriever.retrieve).parameters["user_id"].default

        assert default == USER_ENTITY_ID

    @pytest.mark.asyncio
    async def test_graph_leg_queries_user_relationships_with_the_stored_id_by_default(self):
        retriever, user_rel_calls = _retriever_with_recording_graph_store()

        await retriever.retrieve(query="what do I use")

        assert len(user_rel_calls) == 1
        assert user_rel_calls[0]["user_id"] == "user"
