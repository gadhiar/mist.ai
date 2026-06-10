"""Fail-closed eval isolation guard for the Neo4j graph leg.

The throwaway-trio (event store / vault / sidecar) is isolated via env vars
(see tests/unit/test_eval_paths_overridable.py). Neo4j is Community edition
(a single user database), so its isolation is by SEPARATE INSTANCE -- a
disposable `mist-neo4j-eval` container (see docker-compose.eval-neo4j.yml)
selected via NEO4J_URI, not a separate database name.

When an eval run is active (MIST_EVAL_ISOLATION truthy), the configured
NEO4J_URI must NOT point at the live graph host, or the run is refused before
any connection or write. This prevents an eval/gauntlet from polluting the
canonical graph (the Phase-2 source of truth).
"""

from __future__ import annotations

import os

# Live graph host = the docker-compose service name the backend talks to.
DEFAULT_LIVE_NEO4J_HOST = "mist-neo4j"

_TRUTHY = {"1", "true", "yes", "on"}


class EvalIsolationError(RuntimeError):
    """Raised when an eval-isolated run would target the live graph."""


def is_eval_isolation_active() -> bool:
    """True when the caller declared an isolated eval run.

    Activation is explicit via MIST_EVAL_ISOLATION (the eval runbook sets it on
    every run). Explicit rather than inferred: docker-compose bakes the trio env
    vars to their live defaults inside the container, so "is the var set" cannot
    distinguish an eval run from the live runtime.
    """
    return os.getenv("MIST_EVAL_ISOLATION", "").strip().lower() in _TRUTHY
