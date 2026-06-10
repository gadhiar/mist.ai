"""Fail-closed Neo4j eval-isolation guard (F1)."""

import pytest

from backend.knowledge.eval_isolation import is_eval_isolation_active


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
