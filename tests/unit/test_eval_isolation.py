"""Fail-closed Neo4j eval-isolation guard (F1, hardened by the deep review).

foundation-f123-1: the original guard was a single-hostname DENYLIST -- every
alternate spelling of the live endpoint (localhost:7687 via the host-published
bolt port, 127.0.0.1, container IP) passed. The guard is now an ALLOWLIST of
disposable eval endpoints compared on (host, port).

foundation-f123-2: unrecognized MIST_EVAL_ISOLATION values raise instead of
silently deactivating the guard.
"""

import pytest

from backend.knowledge.config import Neo4jConfig
from backend.knowledge.eval_isolation import (
    EvalIsolationError,
    assert_neo4j_isolated,
    is_eval_isolation_active,
)


class TestIsEvalIsolationActive:
    def test_inactive_when_unset(self, monkeypatch):
        monkeypatch.delenv("MIST_EVAL_ISOLATION", raising=False)
        assert is_eval_isolation_active() is False

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
    def test_active_for_truthy_values(self, monkeypatch, value):
        monkeypatch.setenv("MIST_EVAL_ISOLATION", value)
        assert is_eval_isolation_active() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
    def test_inactive_for_falsy_values(self, monkeypatch, value):
        monkeypatch.setenv("MIST_EVAL_ISOLATION", value)
        assert is_eval_isolation_active() is False

    @pytest.mark.parametrize("value", ["on1", "enabled", "Y", "TRUE=1"])
    def test_unrecognized_values_refuse_instead_of_deactivating(self, monkeypatch, value):
        # A fail-closed guard must not treat an operator typo as "off".
        monkeypatch.setenv("MIST_EVAL_ISOLATION", value)
        with pytest.raises(EvalIsolationError, match="Unrecognized"):
            is_eval_isolation_active()


class TestAssertNeo4jIsolated:
    def test_noop_when_isolation_inactive_even_if_live(self, monkeypatch):
        monkeypatch.delenv("MIST_EVAL_ISOLATION", raising=False)
        cfg = Neo4jConfig(uri="bolt://mist-neo4j:7687")
        # Normal live use is unaffected -- must not raise.
        assert_neo4j_isolated(cfg)

    @pytest.mark.parametrize(
        "uri",
        [
            pytest.param("bolt://mist-neo4j:7687", id="live-service-name"),
            pytest.param("bolt://localhost:7687", id="live-host-published-port"),
            pytest.param("bolt://127.0.0.1:7687", id="live-loopback-ip"),
            pytest.param("bolt://some-dns-alias:7687", id="unknown-alias"),
            pytest.param("bolt://mist-neo4j-eval", id="eval-host-without-port"),
            pytest.param("localhost:7687", id="no-scheme"),
        ],
    )
    def test_refuses_everything_outside_the_allowlist(self, monkeypatch, uri):
        monkeypatch.setenv("MIST_EVAL_ISOLATION", "1")
        with pytest.raises(EvalIsolationError):
            assert_neo4j_isolated(Neo4jConfig(uri=uri))

    @pytest.mark.parametrize(
        "uri",
        [
            pytest.param("bolt://mist-neo4j-eval:7687", id="eval-in-network"),
            pytest.param("bolt://localhost:7688", id="eval-host-published"),
            pytest.param("bolt://127.0.0.1:7688", id="eval-loopback"),
        ],
    )
    def test_allows_disposable_eval_endpoints(self, monkeypatch, uri):
        monkeypatch.setenv("MIST_EVAL_ISOLATION", "1")
        assert_neo4j_isolated(Neo4jConfig(uri=uri))

    def test_allowlist_is_env_overridable(self, monkeypatch):
        monkeypatch.setenv("MIST_EVAL_ISOLATION", "1")
        monkeypatch.setenv("MIST_EVAL_NEO4J_HOSTS", "ci-neo4j:9999")
        assert_neo4j_isolated(Neo4jConfig(uri="bolt://ci-neo4j:9999"))
        with pytest.raises(EvalIsolationError):
            assert_neo4j_isolated(Neo4jConfig(uri="bolt://mist-neo4j-eval:7687"))

    def test_malformed_allowlist_entry_refuses(self, monkeypatch):
        monkeypatch.setenv("MIST_EVAL_ISOLATION", "1")
        monkeypatch.setenv("MIST_EVAL_NEO4J_HOSTS", "no-port-here")
        with pytest.raises(EvalIsolationError, match="Malformed"):
            assert_neo4j_isolated(Neo4jConfig(uri="bolt://mist-neo4j-eval:7687"))


class TestConnectGuard:
    def test_connect_refuses_live_in_eval_mode_before_driver(self, monkeypatch):
        # Eval mode + live URI -> refuse at the guard, before any real driver.
        monkeypatch.setenv("MIST_EVAL_ISOLATION", "1")
        from backend.knowledge.storage.neo4j_connection import Neo4jConnection

        conn = Neo4jConnection(Neo4jConfig(uri="bolt://mist-neo4j:7687"))
        with pytest.raises(EvalIsolationError):
            conn.connect()

    def test_connect_refuses_unrecognized_isolation_value(self, monkeypatch):
        monkeypatch.setenv("MIST_EVAL_ISOLATION", "on1")
        from backend.knowledge.storage.neo4j_connection import Neo4jConnection

        conn = Neo4jConnection(Neo4jConfig(uri="bolt://mist-neo4j:7687"))
        with pytest.raises(EvalIsolationError):
            conn.connect()
