"""Unit tests for the self-model partition relabel migration."""

import pytest

from backend.knowledge.storage.partitions import SELF_MODEL_TYPES
from scripts.migrations.selfmodel_partition import CYPHER, migrate
from tests.mocks.neo4j import FakeGraphExecutor, FakeNeo4jConnection


def test_cypher_relabels_entity_to_selfmodel_for_self_model_types():
    joined = "\n".join(CYPHER)
    assert "REMOVE e:__Entity__" in joined
    assert "SET e:__SelfModel__" in joined
    assert "e.entity_type IN" in joined


def test_cypher_backfills_each_typed_label():
    joined = "\n".join(CYPHER)
    for typed in sorted(SELF_MODEL_TYPES):
        assert f"SET e:{typed}" in joined, f"missing typed-label backfill for {typed}"


@pytest.mark.asyncio
async def test_migrate_issues_every_cypher_statement():
    conn = FakeNeo4jConnection()
    executor = FakeGraphExecutor(connection=conn)

    await migrate(executor)

    assert len(conn.writes) == len(CYPHER)
