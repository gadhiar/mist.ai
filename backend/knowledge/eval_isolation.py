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
from urllib.parse import urlparse

from backend.knowledge.config import Neo4jConfig

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


def _uri_host(uri: str) -> str | None:
    return urlparse(uri).hostname


def assert_neo4j_isolated(neo4j_config: Neo4jConfig) -> None:
    """Refuse (fail-closed) if an eval-isolated run targets the live graph.

    No-op when isolation is not active, so normal runtime/admin use against the
    live graph is unaffected.

    Raises:
        EvalIsolationError: when isolation is active AND NEO4J_URI's host equals
            the live host.
    """
    if not is_eval_isolation_active():
        return
    live_host = os.getenv("MIST_LIVE_NEO4J_HOST", DEFAULT_LIVE_NEO4J_HOST)
    host = _uri_host(neo4j_config.uri)
    if host == live_host:
        raise EvalIsolationError(
            f"Eval isolation is active (MIST_EVAL_ISOLATION) but NEO4J_URI host "
            f"'{host}' is the live graph host. Point NEO4J_URI at a disposable "
            f"eval instance (e.g. bolt://mist-neo4j-eval:7687) before running. "
            f"Refusing to run to avoid polluting the canonical graph."
        )
