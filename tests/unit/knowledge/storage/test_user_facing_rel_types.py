"""Every relation the extractor may produce is user-facing, so every read path serves it.

`_USER_FACING_REL_TYPES` is the default allowlist of the memory index's fact scan and of the
three GraphStore reads below. An extractable type missing from it is written to the graph and
then never read back, which is how RECOMMENDS and HAS_HABIT went unserved after v1.3.0 added
them. These tests pin the allowlist as a superset of `EXTRACTABLE_RELATIONSHIP_TYPES` and check
that each default use passes the whole of it.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from backend.knowledge.ontologies import EXTRACTABLE_RELATIONSHIP_TYPES
from backend.knowledge.storage.graph_store import (
    _USER_FACING_REL_TYPES,
    GraphStore,
    current_facts_query,
)
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection


def test_allowlist_is_a_superset_of_the_extractable_relationship_types():
    missing = set(EXTRACTABLE_RELATIONSHIP_TYPES) - set(_USER_FACING_REL_TYPES)

    assert not missing, f"extractable relation types the read paths never serve: {missing}"


@pytest.mark.parametrize("rel_type", ["RECOMMENDS", "HAS_HABIT"])
def test_allowlist_serves_the_v1_3_0_user_centric_edges(rel_type):
    assert rel_type in _USER_FACING_REL_TYPES


def test_allowlist_has_no_duplicates():
    assert len(_USER_FACING_REL_TYPES) == len(set(_USER_FACING_REL_TYPES))


def _store_params(call: Callable[[GraphStore], object]) -> dict[str, object]:
    conn = FakeNeo4jConnection()
    store = GraphStore(connection=conn, embedding_generator=FakeEmbeddingGenerator())
    call(store)
    assert conn.queries, "no query was executed"
    _, params = conn.queries[-1]
    assert params is not None
    return params


_DEFAULT_USES: dict[str, Callable[[], dict[str, object]]] = {
    "current_facts_query": lambda: current_facts_query(None, 10)[1],
    "get_entity_neighborhood": lambda: _store_params(
        lambda store: store.get_entity_neighborhood("user", max_hops=2)
    ),
    "get_user_relationships_to_entities": lambda: _store_params(
        lambda store: store.get_user_relationships_to_entities("user", entity_ids=["e-1"])
    ),
    "get_all_user_relationships": lambda: _store_params(
        lambda store: store.get_all_user_relationships("user")
    ),
}


@pytest.mark.parametrize("use", sorted(_DEFAULT_USES))
def test_every_default_use_passes_every_extractable_relationship_type(use):
    params = _DEFAULT_USES[use]()

    allowed = params["allowed_rel_types"]
    assert set(allowed) == set(_USER_FACING_REL_TYPES)
    assert set(EXTRACTABLE_RELATIONSHIP_TYPES) <= set(allowed)
