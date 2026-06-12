"""Fail-closed eval isolation guard for the Neo4j graph leg.

The throwaway-trio (event store / vault / sidecar) is isolated via env vars
(see tests/unit/test_eval_paths_overridable.py). Neo4j is Community edition
(a single user database), so its isolation is by SEPARATE INSTANCE -- a
disposable `mist-neo4j-eval` container (see docker-compose.eval-neo4j.yml)
selected via NEO4J_URI, not a separate database name.

When an eval run is active (MIST_EVAL_ISOLATION truthy), the configured
NEO4J_URI must match a known DISPOSABLE EVAL ENDPOINT (host AND port), or
the run is refused before any connection or write. An allowlist, not a
denylist: the live instance has many spellings (mist-neo4j, localhost:7687
via the host-published bolt port, 127.0.0.1, container IP, DNS aliases) and
refusing only one of them fails open for the rest. The port matters --
host-side, live is localhost:7687 and eval is localhost:7688, which a
hostname-only check cannot distinguish.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

from backend.knowledge.config import Neo4jConfig

# Disposable eval endpoints: in-network service name + the host-published
# ports from docker-compose.eval-neo4j.yml. Override via MIST_EVAL_NEO4J_HOSTS
# (comma-separated host:port list).
DEFAULT_EVAL_NEO4J_ENDPOINTS = "mist-neo4j-eval:7687,localhost:7688,127.0.0.1:7688"

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"", "0", "false", "no", "off"}


class EvalIsolationError(RuntimeError):
    """Raised when an eval-isolated run would target the live graph."""


def is_eval_isolation_active() -> bool:
    """True when the caller declared an isolated eval run.

    Activation is explicit via MIST_EVAL_ISOLATION (the eval runbook sets it on
    every run). Explicit rather than inferred: docker-compose bakes the trio env
    vars to their live defaults inside the container, so "is the var set" cannot
    distinguish an eval run from the live runtime.

    Raises:
        EvalIsolationError: for unrecognized values. A fail-closed guard must
            not treat an operator typo ('ture', 'enabled') as "isolation off"
            and silently run unguarded against the live graph.
    """
    raw = os.getenv("MIST_EVAL_ISOLATION", "")
    value = raw.strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    raise EvalIsolationError(
        f"Unrecognized MIST_EVAL_ISOLATION value {raw!r}; use '1' to activate "
        "or unset to deactivate. Refusing to guess for a fail-closed guard."
    )


def _allowed_endpoints() -> set[tuple[str, int]]:
    """Parse the eval-endpoint allowlist from env (or the default)."""
    raw = os.getenv("MIST_EVAL_NEO4J_HOSTS", DEFAULT_EVAL_NEO4J_ENDPOINTS)
    allowed: set[tuple[str, int]] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        host, _, port = item.rpartition(":")
        if not host or not port.isdigit():
            raise EvalIsolationError(
                f"Malformed MIST_EVAL_NEO4J_HOSTS entry {item!r}; expected host:port."
            )
        allowed.add((host.lower(), int(port)))
    if not allowed:
        raise EvalIsolationError(
            "MIST_EVAL_NEO4J_HOSTS resolved to an empty allowlist; an eval run "
            "needs at least one disposable endpoint."
        )
    return allowed


def assert_neo4j_isolated(neo4j_config: Neo4jConfig) -> None:
    """Refuse (fail-closed) unless an eval run targets a known eval endpoint.

    No-op when isolation is not active, so normal runtime/admin use against the
    live graph is unaffected.

    Raises:
        EvalIsolationError: when isolation is active AND NEO4J_URI's
            (host, port) is not in the eval-endpoint allowlist -- including
            URIs whose host or port cannot be parsed at all.
    """
    if not is_eval_isolation_active():
        return
    parsed = urlparse(neo4j_config.uri)
    host = parsed.hostname
    try:
        port = parsed.port
    except ValueError:
        port = None
    if host is None or port is None:
        raise EvalIsolationError(
            f"Eval isolation is active but NEO4J_URI {neo4j_config.uri!r} has no "
            "parseable host:port. Point NEO4J_URI at a disposable eval instance "
            "(e.g. bolt://mist-neo4j-eval:7687). Refusing to run."
        )
    if (host.lower(), port) not in _allowed_endpoints():
        raise EvalIsolationError(
            f"Eval isolation is active but NEO4J_URI '{neo4j_config.uri}' is not "
            f"a recognized disposable eval endpoint. Allowed: "
            f"{sorted(_allowed_endpoints())} (override via MIST_EVAL_NEO4J_HOSTS). "
            "The live bolt port is host-published, so only explicit eval "
            "endpoints pass. Refusing to avoid polluting the canonical graph."
        )
