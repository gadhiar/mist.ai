"""Unit test fixtures.

Shared fixtures for all unit tests. Import specific fixtures from
tests/mocks/fixtures/ using Pattern B (explicit import + noqa).
"""

import pytest

from tests.mocks.config import build_test_config
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeGraphExecutor, FakeNeo4jConnection


@pytest.fixture
def fake_connection():
    """A FakeNeo4jConnection with no pre-configured results."""
    return FakeNeo4jConnection()


@pytest.fixture
def fake_executor(fake_connection):
    """A FakeGraphExecutor wrapping the fake connection."""
    return FakeGraphExecutor(connection=fake_connection)


@pytest.fixture
def fake_embeddings():
    """A FakeEmbeddingGenerator with default 384 dimensions."""
    return FakeEmbeddingGenerator()


@pytest.fixture
def test_config():
    """A KnowledgeConfig with test defaults."""
    return build_test_config()
