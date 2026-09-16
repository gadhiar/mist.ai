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

This file proves the autouse fixture's two effects independently: that it
sets MIST_EVAL_ISOLATION (the flag classes), and that it clears
MIST_EVAL_NEO4J_HOSTS so a widened allowlist cannot silently admit the live
endpoint (TestEnvOverrideIsCleared). It does not prove these are the ONLY
effects of removing the fixture would have, only that removing it (or
disabling its `autouse=True`) is caught: the tests that assert the flag is
set, and the tests that assert `connect()` raises `EvalIsolationError`, flip
to failing or erroring when the fixture is gone -- each such claim was
verified by disabling the fixture and rerunning this file, not assumed. Any
test whose assertion holds independently of the fixture (for example, that
an env var neither the fixture nor the test sets is absent) proves nothing
about the fixture and is not evidence for its removal; see
TestEnvOverrideIsCleared's docstring for why the old version of that check
was exactly this trap.
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


class TestEnvOverrideIsCleared:
    """The fixture's delenv, not just its setenv, is load-bearing.

    Asserting only that MIST_EVAL_NEO4J_HOSTS is absent (with nothing ever
    setting it) is a guard that can be silently SATISFIED: the assertion
    passes whether or not the fixture's delenv ever ran, because there was
    never a value to clear (tests/CLAUDE.md, "Say what a guard does AND does
    not catch"). This class instead ARRANGES the hole the delenv exists to
    close -- a class-scoped override widening MIST_EVAL_NEO4J_HOSTS to admit
    the live endpoint -- then proves the per-test fixture still wins.

    Pytest sets up higher-scoped fixtures before function-scoped autouse
    fixtures, so the class-scoped `_widen_allowlist_to_admit_live` below
    runs first on every test in this class, and the function-scoped autouse
    `_guard_unit_tier_against_live_neo4j` in conftest.py runs after it,
    deleting the override each time.

    Verified by mutation: removing ONLY the `monkeypatch.delenv(...)` line
    from the autouse fixture (leaving its `setenv` line intact) makes
    `test_connect_still_raises_despite_the_class_scoped_override` fail with
    `DID NOT RAISE`, because the override then survives into the test body
    and admits the live endpoint. Restoring the delenv line makes it pass
    again; both states were run, not assumed.
    """

    @pytest.fixture(scope="class", autouse=True)
    def _widen_allowlist_to_admit_live(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("MIST_EVAL_NEO4J_HOSTS", "mist-neo4j:7687")
            yield

    def test_override_is_absent_by_the_time_the_test_runs(self):
        assert "MIST_EVAL_NEO4J_HOSTS" not in os.environ

    @patch(f"{MODULE}.GraphDatabase")
    def test_connect_still_raises_despite_the_class_scoped_override(self, mock_gdb):
        # Arrange
        config = Neo4jConfig(uri="bolt://mist-neo4j:7687")
        conn = Neo4jConnection(config)

        # Act / Assert
        with pytest.raises(EvalIsolationError):
            conn.connect()

        # Side-effect boundary: the guard must refuse before any driver exists.
        mock_gdb.driver.assert_not_called()


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
