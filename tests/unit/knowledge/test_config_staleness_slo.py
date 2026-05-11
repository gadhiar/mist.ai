"""Tests for FilewatcherConfig staleness_slo_seconds -> audit_interval_seconds relationship.

Fix B (P3 #3): staleness_slo_seconds was dead config. This suite verifies that
audit_interval_seconds defaults to staleness_slo_seconds // 2, creating the
ADR-010 SLO-driven audit budget.

- Default staleness_slo_seconds = 5 -> default audit_interval_seconds = 2 (5 // 2)
  NOTE: default audit_interval_seconds changes from 60 to 2 after this fix.
- When staleness_slo_seconds is explicitly overridden via env, audit_interval_seconds
  recomputes from the new SLO unless MIST_FILEWATCHER_AUDIT_INTERVAL_SECONDS is
  also explicitly set.
- When MIST_FILEWATCHER_AUDIT_INTERVAL_SECONDS is explicitly set, that value wins
  regardless of staleness_slo_seconds.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

from backend.knowledge.config import FilewatcherConfig


@contextmanager
def _env(**values):
    original = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, v in original.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


_ALL_FILEWATCHER_KEYS = (
    "MIST_FILEWATCHER_ENABLED",
    "MIST_FILEWATCHER_OBSERVER_TYPE",
    "MIST_FILEWATCHER_DEBOUNCE_MS",
    "MIST_FILEWATCHER_STALENESS_SLO_SECONDS",
    "MIST_FILEWATCHER_AUDIT_INTERVAL_SECONDS",
)


class TestStalenessSloDrivesAuditInterval:
    """audit_interval_seconds defaults to staleness_slo_seconds // 2."""

    def test_default_audit_interval_is_slo_half(self):
        """Default audit_interval_seconds == staleness_slo_seconds // 2 == 2."""
        with _env(**{k: None for k in _ALL_FILEWATCHER_KEYS}):
            config = FilewatcherConfig.from_env()
        assert config.staleness_slo_seconds == 5
        assert config.audit_interval_seconds == config.staleness_slo_seconds // 2

    def test_custom_staleness_slo_drives_audit_interval(self):
        """When only SLO is overridden, audit_interval recomputes from SLO."""
        cleared = {k: None for k in _ALL_FILEWATCHER_KEYS}
        cleared["MIST_FILEWATCHER_STALENESS_SLO_SECONDS"] = "20"
        with _env(**cleared):
            config = FilewatcherConfig.from_env()
        assert config.staleness_slo_seconds == 20
        assert config.audit_interval_seconds == 10  # 20 // 2

    def test_explicit_audit_interval_wins_over_slo(self):
        """When MIST_FILEWATCHER_AUDIT_INTERVAL_SECONDS is set, it overrides SLO-derived value."""
        cleared = {k: None for k in _ALL_FILEWATCHER_KEYS}
        cleared["MIST_FILEWATCHER_STALENESS_SLO_SECONDS"] = "20"
        cleared["MIST_FILEWATCHER_AUDIT_INTERVAL_SECONDS"] = "45"
        with _env(**cleared):
            config = FilewatcherConfig.from_env()
        assert config.staleness_slo_seconds == 20
        assert config.audit_interval_seconds == 45  # Explicit value wins.

    def test_audit_interval_explicit_default_slo_not_set(self):
        """Explicit audit_interval wins even when SLO is at default."""
        cleared = {k: None for k in _ALL_FILEWATCHER_KEYS}
        cleared["MIST_FILEWATCHER_AUDIT_INTERVAL_SECONDS"] = "120"
        with _env(**cleared):
            config = FilewatcherConfig.from_env()
        assert config.audit_interval_seconds == 120

    def test_dataclass_docstring_describes_slo_relationship(self):
        """FilewatcherConfig docstring documents the SLO->audit relationship."""
        doc = FilewatcherConfig.__doc__
        assert doc is not None
        # The docstring must mention the relationship so operators understand the coupling.
        assert "staleness_slo_seconds" in doc
        assert "audit_interval_seconds" in doc
