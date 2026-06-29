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


def test_cypher_reconciles_colliding_selfmodel_stub_before_relabel():
    """A gap-window backend boot MERGEs a :__SelfModel__ self-model stub that
    collides by id with the still-:__Entity__ original. The migration must
    delete that stub (the :__Entity__ original wins) BEFORE the relabel, or the
    relabel duplicates the singleton root (and violates selfmodel_id_unique).
    """
    joined = "\n".join(CYPHER)

    assert "DETACH DELETE" in joined, "migration must reconcile colliding self-model stubs"

    reconcile_idx = joined.index("DETACH DELETE")
    relabel_idx = joined.index("REMOVE e:__Entity__")
    assert reconcile_idx < relabel_idx, "reconcile must run before the relabel"


@pytest.mark.asyncio
async def test_migrate_issues_every_cypher_statement():
    conn = FakeNeo4jConnection()
    executor = FakeGraphExecutor(connection=conn)

    await migrate(executor)

    assert len(conn.writes) == len(CYPHER)
