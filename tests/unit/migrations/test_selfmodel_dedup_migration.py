"""Unit tests for the self-model dedup migration."""

import pytest

from scripts.migrations.selfmodel_dedup import CYPHER, migrate
from tests.mocks.neo4j import FakeGraphExecutor, FakeNeo4jConnection


def test_cypher_deletes_mist_prefixed_shadow_nodes():
    joined = "\n".join(CYPHER)
    assert "DETACH DELETE" in joined
    assert "mist-trait-" in joined
    assert "mist-cap-" in joined
    assert "mist-pref-" in joined


def test_cypher_is_double_guarded_by_type_and_prefix():
    """The delete must require BOTH a self-model trait/capability/preference
    entity_type AND a mist- id prefix, so a canonical kebab node (id
    'trait-warm') can never be caught.
    """
    joined = "\n".join(CYPHER)
    assert "s.entity_type IN ['MistTrait', 'MistCapability', 'MistPreference']" in joined
    assert "STARTS WITH 'mist-trait-'" in joined
    assert "STARTS WITH 'mist-cap-'" in joined
    assert "STARTS WITH 'mist-pref-'" in joined


def test_cypher_excludes_identity_root():
    """MistIdentity must not appear in the delete: the singleton mist-identity
    root is the keeper (both seed paths MERGE onto it).
    """
    joined = "\n".join(CYPHER)
    assert "MistIdentity" not in joined


@pytest.mark.asyncio
async def test_migrate_issues_every_cypher_statement():
    conn = FakeNeo4jConnection()
    executor = FakeGraphExecutor(connection=conn)

    await migrate(executor)

    assert len(conn.writes) == len(CYPHER)
