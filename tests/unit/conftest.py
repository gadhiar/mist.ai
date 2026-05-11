"""Unit test fixtures.

Shared fixtures for all unit tests. Import specific fixtures from
tests/mocks/fixtures/ using Pattern B (explicit import + noqa).
"""

import tempfile
from pathlib import Path

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


def make_test_conventions_loader():
    """Return a ConventionsLoader pointing at a fresh empty temp directory.

    Used to satisfy the required `conventions_loader` DI parameter in tests
    that do not need vault-root content. The empty dir means load_vault_root()
    returns None and no conventions user message is injected.
    """
    from backend.vault.conventions import ConventionsLoader

    tmp_dir = Path(tempfile.mkdtemp())
    return ConventionsLoader(vault_root=tmp_dir)


@pytest.fixture
def null_conventions_loader():
    """A ConventionsLoader whose vault root is empty (no MIST.md / CLAUDE.md).

    Satisfies the required `conventions_loader` DI parameter for tests that do
    not exercise the conventions injection path.
    """
    from backend.vault.conventions import ConventionsLoader

    tmp_dir = Path(tempfile.mkdtemp())
    return ConventionsLoader(vault_root=tmp_dir)
