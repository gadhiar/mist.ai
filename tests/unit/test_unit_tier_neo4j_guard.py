"""Guard tests for the unit-tier autouse Neo4j isolation fixture.

`tests/unit/conftest.py::_guard_unit_tier_against_live_neo4j` sets
MIST_EVAL_ISOLATION=1 and unsets MIST_EVAL_NEO4J_HOSTS for every unit test.
These tests exercise that fixture's effect directly -- they do NOT set
MIST_EVAL_ISOLATION themselves, because doing so would test the isolation
guard in `backend/knowledge/eval_isolation.py` (already covered by
`tests/unit/test_eval_isolation.py`) rather than the autouse fixture that
activates it for this tier.

What these tests do NOT prove:
- That MIST_EVAL_ISOLATION protects any code path other than
  `Neo4jConnection.connect()` (the only call site of `assert_neo4j_isolated`
  as of this writing).
- That the autouse fixture runs before EVERY other fixture in the test
  tree -- a module-scoped fixture or import-time code that reaches Neo4j
  executes before this function-scoped fixture regardless of what this file
  asserts.
- Real Neo4j connectivity or query behavior; `GraphDatabase` is patched, per
  tests/CLAUDE.md's "Patching (Last Resort)" convention, because
  `Neo4jConnection` has no DI seam for the driver.
- Anything about `pytest --noconftest`, or a test that clears the isolation
  env vars itself after the autouse fixture runs -- both are named as blind
  spots in the fixture's own docstring, not proven or disproven here.

Six of the eight tests below depend on MIST_EVAL_ISOLATION being active
WITHOUT setting it locally, and are what fail if the autouse fixture in
conftest.py is removed (or its `autouse=True` is disabled): verified by
disabling it and rerunning this file, which produced exactly this 6-failed/
2-passed split. The two that keep passing are unrelated to the fixture --
`test_mist_eval_neo4j_hosts_is_unset` (the var is absent whether or not the
fixture ran) and `test_build_test_config_defaults_to_localhost_7687_when_unset`
(asserts a config default, not an isolation outcome) -- and their continuing
to pass is expected, not a gap in this file's coverage of the fixture. The
six that fail: `test_mist_eval_isolation_env_var_is_set` and
`test_is_eval_isolation_active_reports_true` because the flag and
`is_eval_isolation_active()` revert to unset/False; the three
`test_connect_raises_before_creating_a_driver` cases and
`test_connect_with_the_fallback_config_raises_before_creating_a_driver`
because `assert_neo4j_isolated()` then no-ops, `GraphDatabase.driver` IS
called, and no `EvalIsolationError` is raised.
"""

import os
from unittest.mock import patch

import pytest

from backend.knowledge.config import Neo4jConfig
from backend.knowledge.eval_isolation import EvalIsolationError, is_eval_isolation_active
from backend.knowledge.storage.neo4j_connection import Neo4jConnection
from tests.mocks.config import build_test_config

MODULE = "backend.knowledge.storage.neo4j_connection"


class TestFlagIsActiveByDefault:
    """The autouse fixture activates eval isolation for every unit test."""

    def test_mist_eval_isolation_env_var_is_set(self):
        assert os.environ["MIST_EVAL_ISOLATION"] == "1"

    def test_is_eval_isolation_active_reports_true(self):
        assert is_eval_isolation_active() is True

    def test_mist_eval_neo4j_hosts_is_unset(self):
        assert "MIST_EVAL_NEO4J_HOSTS" not in os.environ


class TestConnectRefusesLiveEndpoints:
    """connect() refuses every live spelling of NEO4J_URI before the driver is created."""

    @pytest.mark.parametrize(
        "live_uri",
        [
            pytest.param("bolt://mist-neo4j:7687", id="live-service-name"),
            pytest.param("bolt://localhost:7687", id="live-host-published-port"),
            pytest.param("bolt://127.0.0.1:7687", id="live-loopback-ip"),
        ],
    )
    @patch(f"{MODULE}.GraphDatabase")
    def test_connect_raises_before_creating_a_driver(self, mock_gdb, live_uri):
        # Arrange
        config = Neo4jConfig(uri=live_uri)
        conn = Neo4jConnection(config)

        # Act / Assert
        with pytest.raises(EvalIsolationError):
            conn.connect()

        # Side-effect boundary: the guard must refuse before any driver exists.
        mock_gdb.driver.assert_not_called()


class TestUnsetNeo4jUriFallback:
    """build_test_config()'s NEO4J_URI fallback also lands outside the eval allowlist."""

    def test_build_test_config_defaults_to_localhost_7687_when_unset(self, monkeypatch):
        # Arrange
        monkeypatch.delenv("NEO4J_URI", raising=False)

        # Act
        config = build_test_config()

        # Assert
        assert config.neo4j.uri == "bolt://localhost:7687"

    @patch(f"{MODULE}.GraphDatabase")
    def test_connect_with_the_fallback_config_raises_before_creating_a_driver(
        self, mock_gdb, monkeypatch
    ):
        # Arrange
        monkeypatch.delenv("NEO4J_URI", raising=False)
        config = build_test_config()
        conn = Neo4jConnection(config.neo4j)

        # Act / Assert
        with pytest.raises(EvalIsolationError):
            conn.connect()

        # Side-effect boundary: the guard must refuse before any driver exists.
        mock_gdb.driver.assert_not_called()
