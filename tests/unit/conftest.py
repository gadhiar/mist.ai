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


@pytest.fixture(autouse=True)
def _guard_unit_tier_against_live_neo4j(monkeypatch):
    """Force every unit test to run as an eval-isolated run.

    Sets MIST_EVAL_ISOLATION=1 and unsets MIST_EVAL_NEO4J_HOSTS for the
    duration of the test, so `Neo4jConnection.connect()` -- which calls
    `assert_neo4j_isolated(self.config)` before creating a driver -- refuses
    any (host, port) outside the eval allowlist
    (DEFAULT_EVAL_NEO4J_ENDPOINTS: mist-neo4j-eval:7687, localhost:7688,
    127.0.0.1:7688). In the live dev container NEO4J_URI defaults to
    bolt://mist-neo4j:7687, so a unit test that reaches `connect()` would
    otherwise write to the canonical graph.

    Function-scoped (the default) and NOT session- or module-scoped: a
    broader scope would leak MIST_EVAL_ISOLATION into integration tests
    collected in the same pytest run.

    What this does NOT catch:
    - A test that names an eval endpoint itself (e.g. bolt://mist-neo4j-eval:7687
      or bolt://localhost:7688) passes the guard; the guard only refuses
      non-eval endpoints, it does not forbid connecting at all.
    - Module-scoped fixtures or import-time code that reach Neo4j before this
      function-scoped fixture runs -- autouse fixtures still run in fixture
      dependency/scope order, so a broader-scoped fixture executes first.
    - Running with `--noconftest`, which skips this file entirely.
    - A test that itself clears or overwrites MIST_EVAL_ISOLATION or
      MIST_EVAL_NEO4J_HOSTS after this fixture runs -- the test body's own
      monkeypatch calls win because they share the same function-scoped
      monkeypatch and run later.

    This is a live-write guard, not a hermeticity guarantee: it stops
    `Neo4jConnection.connect()` from reaching the live graph, but does not by
    itself make a test deterministic, network-free, or free of other I/O.
    """
    monkeypatch.setenv("MIST_EVAL_ISOLATION", "1")
    monkeypatch.delenv("MIST_EVAL_NEO4J_HOSTS", raising=False)


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
