"""Fail-closed Neo4j eval-isolation guard (F1)."""

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

    def test_inactive_for_falsy_value(self, monkeypatch):
        monkeypatch.setenv("MIST_EVAL_ISOLATION", "0")
        assert is_eval_isolation_active() is False


class TestAssertNeo4jIsolated:
    def test_noop_when_isolation_inactive_even_if_live(self, monkeypatch):
        monkeypatch.delenv("MIST_EVAL_ISOLATION", raising=False)
        cfg = Neo4jConfig(uri="bolt://mist-neo4j:7687")
        # Normal live use is unaffected -- must not raise.
        assert_neo4j_isolated(cfg)

    def test_raises_when_active_and_uri_is_live(self, monkeypatch):
        monkeypatch.setenv("MIST_EVAL_ISOLATION", "1")
        cfg = Neo4jConfig(uri="bolt://mist-neo4j:7687")
        with pytest.raises(EvalIsolationError):
            assert_neo4j_isolated(cfg)

    def test_passes_when_active_and_uri_is_eval(self, monkeypatch):
        monkeypatch.setenv("MIST_EVAL_ISOLATION", "1")
        cfg = Neo4jConfig(uri="bolt://mist-neo4j-eval:7687")
        # Disposable eval host -- allowed.
        assert_neo4j_isolated(cfg)

    def test_respects_live_host_override(self, monkeypatch):
        monkeypatch.setenv("MIST_EVAL_ISOLATION", "1")
        monkeypatch.setenv("MIST_LIVE_NEO4J_HOST", "prod-neo4j")
        cfg = Neo4jConfig(uri="bolt://prod-neo4j:7687")
        with pytest.raises(EvalIsolationError):
            assert_neo4j_isolated(cfg)


class TestConnectGuard:
    def test_connect_refuses_live_in_eval_mode_before_driver(self, monkeypatch):
        # Eval mode + live URI -> refuse at the guard, before any real driver.
        monkeypatch.setenv("MIST_EVAL_ISOLATION", "1")
        from backend.knowledge.storage.neo4j_connection import Neo4jConnection

        conn = Neo4jConnection(Neo4jConfig(uri="bolt://mist-neo4j:7687"))
        with pytest.raises(EvalIsolationError):
            conn.connect()
