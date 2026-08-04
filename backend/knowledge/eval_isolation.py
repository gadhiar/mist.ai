"""Fail-closed isolation guards for runs that must not touch live state.

Four guards live here, one per way a non-live run can reach live state. Three
of the four are ALLOWLISTS over `(host, port)`, and that uniformity is the
point: the fourth was a denylist, and the denylist was the one with the hole.

**Neo4j, eval (`assert_neo4j_isolated`).** The throwaway-trio (event store /
vault / sidecar) is isolated via env vars (see
tests/unit/test_eval_paths_overridable.py). Neo4j is Community edition (a single
user database), so its isolation is by SEPARATE INSTANCE -- a disposable
`mist-neo4j-eval` container (see docker-compose.eval-neo4j.yml) selected via
NEO4J_URI, not a separate database name. When an eval run is active
(MIST_EVAL_ISOLATION truthy), the configured NEO4J_URI must match a known
DISPOSABLE EVAL ENDPOINT (host AND port), or the run is refused before any
connection or write.

**Neo4j, dev/hydration (`assert_neo4j_dev_isolated`).** The same allowlist
shape against the `dev` profile's endpoints, with one deliberate difference:
it is NOT gated on an activation env var. `assert_neo4j_isolated` must no-op
for normal runtime because live traffic flows through the same `connect()`.
Nothing legitimately runs hydration tooling against the live graph, so its
guard has no "off" -- which makes it strictly stronger, because it cannot be
defeated by forgetting to set a variable.

**Neo4j, rebuild (`assert_rebuild_target_not_live`).** The rebuild's WRITE
target, which it `DETACH DELETE`s before loading. Also an allowlist, narrower
still: staging only, since eval and dev are other people's disposable graphs.

**Filesystem (`assert_isolated_root`).** Refuses a dev/hydration state root
that is, sits under, or CONTAINS a live state directory.

Allowlist for endpoints, denylist for paths -- and the asymmetry is not an
inconsistency. The live Neo4j instance has many spellings (mist-neo4j,
localhost:7687 via the host-published bolt port, 127.0.0.1, container IP, DNS
aliases) with no canonical form, so refusing one of them fails open for the
rest; only an allowlist closes it. A filesystem path DOES have a canonical
form: `Path.resolve()` follows symlinks, collapses `..`, absolutizes, and
case-folds on Windows, so every spelling of `mist-memory/` converges on one
value and a denylist over resolved paths cannot be evaded by respelling. The
port matters for endpoints -- host-side, live is localhost:7687, eval is
localhost:7688 and dev is localhost:7690, which a hostname-only check cannot
distinguish.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

from backend.errors import MistError
from backend.knowledge.config import Neo4jConfig

# Disposable eval endpoints: in-network service name + the host-published
# ports from docker-compose.eval-neo4j.yml. Override via MIST_EVAL_NEO4J_HOSTS
# (comma-separated host:port list).
DEFAULT_EVAL_NEO4J_ENDPOINTS = "mist-neo4j-eval:7687,localhost:7688,127.0.0.1:7688"

# Dev-profile endpoints from docker-compose.dev-hydration.yml (R1.4.6 T1).
# Override via MIST_DEV_NEO4J_HOSTS.
DEFAULT_DEV_NEO4J_ENDPOINTS = "mist-neo4j-dev:7687,localhost:7690,127.0.0.1:7690"

# Rebuild WRITE targets from docker-compose.staging-neo4j.yml. Staging only --
# deliberately excluding eval and dev, which are other people's disposable
# graphs, not scratch space for a rebuild. Override via MIST_REBUILD_NEO4J_HOSTS.
DEFAULT_REBUILD_NEO4J_ENDPOINTS = "mist-neo4j-staging:7687,localhost:7689,127.0.0.1:7689"

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"", "0", "false", "no", "off"}

# backend/knowledge/eval_isolation.py -> backend/knowledge -> backend -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]


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


def _parse_endpoint_allowlist(env_var: str, default: str) -> set[tuple[str, int]]:
    """Parse a `host:port` allowlist from `env_var` (or `default`).

    Shared by the eval and dev guards so the two cannot drift on parsing or on
    the empty-allowlist refusal. An empty result is an error, never "allow
    everything": a fail-closed guard whose allowlist evaporates must refuse.
    """
    raw = os.getenv(env_var, default)
    allowed: set[tuple[str, int]] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        host, _, port = item.rpartition(":")
        if not host or not port.isdigit():
            raise EvalIsolationError(f"Malformed {env_var} entry {item!r}; expected host:port.")
        allowed.add((host.lower(), int(port)))
    if not allowed:
        raise EvalIsolationError(
            f"{env_var} resolved to an empty allowlist; an isolated run needs at "
            "least one disposable endpoint."
        )
    return allowed


def _allowed_endpoints() -> set[tuple[str, int]]:
    """Parse the eval-endpoint allowlist from env (or the default)."""
    return _parse_endpoint_allowlist("MIST_EVAL_NEO4J_HOSTS", DEFAULT_EVAL_NEO4J_ENDPOINTS)


def _allowed_dev_endpoints() -> set[tuple[str, int]]:
    """Parse the dev/hydration endpoint allowlist from env (or the default)."""
    return _parse_endpoint_allowlist("MIST_DEV_NEO4J_HOSTS", DEFAULT_DEV_NEO4J_ENDPOINTS)


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


class RebuildTargetError(MistError):
    """Raised when a graph rebuild's WRITE target is not a disposable instance."""


def _allowed_rebuild_endpoints() -> set[tuple[str, int]]:
    """Parse the rebuild-target allowlist, re-raising as `RebuildTargetError`.

    The conversion matters at the CLI: `mist_admin` catches `RebuildTargetError`
    to print a clean refusal, so leaking `EvalIsolationError` from a malformed
    override would turn a guarded refusal into a traceback. Still fail-closed
    either way -- it just reads as a crash instead of a decision.
    """
    try:
        return _parse_endpoint_allowlist(
            "MIST_REBUILD_NEO4J_HOSTS", DEFAULT_REBUILD_NEO4J_ENDPOINTS
        )
    except EvalIsolationError as exc:
        raise RebuildTargetError(str(exc)) from exc


def assert_rebuild_target_not_live(target_uri: str, live_uri: str) -> None:
    """Fail-closed: refuse a rebuild whose write target is not disposable staging.

    The rebuild opens a read-only connection to the live `source` and a
    read-write connection to `staging`, then `MATCH (n) DETACH DELETE n` on the
    target. The process-global eval guard cannot express "read live, write
    staging", so this dedicated check protects the write target.

    **This is an ALLOWLIST, and it used to be a denylist of one spelling.** The
    old check asked only whether `target_uri` was textually different from
    `live_uri` -- a string comparison, not an address comparison. `bolt://
    mist-neo4j:7687` was refused while `bolt://localhost:7687` PASSED, and the
    live bolt port is host-published, so those are the same database. That is
    the natural thing to type when running the rebuild from the host, and the
    unconditional wipe behind the guard made it a live-graph loss.

    The allowlist is deliberately narrow -- staging only, not eval and not dev.
    `docker-compose.staging-neo4j.yml` keeps staging distinct from
    `mist-neo4j-eval` precisely so a rebuild dry-run cannot clobber the test DB,
    and the same argument now covers `mist-neo4j-dev`: R1.6 compares live
    against rebuilt on the dev stack, where the DEV graph is the "live" side, so
    admitting it as a WRITE target would let a rebuild delete an 87-turn
    hydrated fixture.

    `live_uri` is still compared, and is not redundant: an operator who widens
    `MIST_REBUILD_NEO4J_HOSTS` to include a live endpoint is caught by the
    second arm rather than being handed the wipe by their own override.

    Args:
        target_uri: The rebuild's WRITE target. Must be an allowlisted endpoint.
        live_uri: The canonical graph, refused as a target under any spelling
            the allowlist might have been widened to admit.

    Raises:
        RebuildTargetError: When `target_uri` is unparsable, is not a known
            disposable rebuild endpoint, or resolves to `live_uri`.
    """
    target = urlparse(target_uri)
    live = urlparse(live_uri)
    if target.hostname is None or live.hostname is None:
        raise RebuildTargetError(
            f"Cannot parse host from target {target_uri!r} or live {live_uri!r}; "
            "refusing the rebuild for a fail-closed guard."
        )
    try:
        target_port = target.port
        same_port = target_port == live.port
    except ValueError as exc:
        raise RebuildTargetError(f"Unparsable port in {target_uri!r}/{live_uri!r}") from exc
    if target_port is None:
        raise RebuildTargetError(
            f"Cannot parse port from rebuild target {target_uri!r}; refusing. The live "
            "and staging instances differ only by port on the host, so a portless "
            "target cannot be told apart from the live graph."
        )

    if target.hostname.lower() == live.hostname.lower() and same_port:
        raise RebuildTargetError(
            f"Rebuild write target {target_uri!r} resolves to the live graph "
            f"({live_uri!r}). The rebuild must write only to a disposable staging "
            "instance. Refusing to avoid corrupting the canonical graph."
        )
    if (target.hostname.lower(), target_port) not in _allowed_rebuild_endpoints():
        raise RebuildTargetError(
            f"Rebuild write target '{target_uri}' is not a recognized disposable "
            f"staging endpoint. Allowed: {sorted(_allowed_rebuild_endpoints())} "
            "(override via MIST_REBUILD_NEO4J_HOSTS). The rebuild DETACH DELETEs its "
            "target, and the live bolt port is host-published -- so a target is "
            "accepted only by being on this list, never by merely differing from the "
            "live URI. Refusing to avoid destroying a graph."
        )


class IsolatedRootError(MistError):
    """Raised when a dev/hydration state root resolves onto live state."""


def live_state_roots() -> list[Path]:
    """Directories holding LIVE state that a dev/hydration root must not touch.

    Both the host spelling and the in-container spelling of each store are
    listed, because the same guard runs in both places and each sees only its
    own. On the host `/app/data` resolves to something inert; an over-broad
    denylist entry costs nothing, an absent one costs the store.

    Deliberately NOT filtered by `exists()`. `scripts/golden_log/generate.py`
    carries a narrower local twin of this guard that drops non-existent
    candidates, which fails OPEN in the case that matters most: a live root
    absent from the current filesystem view (host-side `/app/data`, a fresh
    clone with no `mist-memory/`) silently leaves the denylist, and the guard
    then blesses a path that a later container run resolves straight onto live
    state. Presence on disk is not what makes a directory live.
    """
    return [
        (REPO_ROOT / "data").resolve(),
        (REPO_ROOT / "mist-memory").resolve(),
        Path("/app/data").resolve(),
        Path("/app/mist-memory").resolve(),
        (Path.home() / ".mist").resolve(),
    ]


def assert_isolated_root(root: Path | str, *, purpose: str = "hydration") -> None:
    """Refuse a state root that is, sits under, or CONTAINS a live directory.

    The containment arm is the one R1.4.5's precedent lacks, and it is the arm
    that matters here. `generate.py` checks only "root is live" and "root under
    live", so it would accept the repo root itself -- harmless for a generator
    that creates two files, and catastrophic for `restore`, which CLEARS its
    target before writing. A restore pointed at the repo root would delete
    `data/` and `mist-memory/` on the way in.

    Args:
        root: The candidate dev/hydration state root.
        purpose: Named in the refusal so the operator knows which tool refused.

    Raises:
        IsolatedRootError: When `root` resolves onto live state, onto a
            filesystem/drive root, or onto the home directory itself.
    """
    resolved = Path(root).resolve()

    if resolved.parent == resolved:
        raise IsolatedRootError(
            f"refusing {purpose} root {resolved}: it is a filesystem/drive root. "
            "Point it at a dedicated directory such as <repo>/dev-state."
        )
    if resolved == Path.home().resolve():
        raise IsolatedRootError(
            f"refusing {purpose} root {resolved}: it is the home directory itself, "
            "which contains ~/.mist and every other user store."
        )

    for live in live_state_roots():
        if resolved == live:
            raise IsolatedRootError(
                f"refusing {purpose} root {resolved}: it IS the live state directory "
                f"{live}. Point it at a dedicated directory such as <repo>/dev-state."
            )
        if live in resolved.parents:
            raise IsolatedRootError(
                f"refusing {purpose} root {resolved}: it sits under the live state "
                f"directory {live}. A hydration run must be structurally incapable "
                "of reaching live stores, not merely configured away from them."
            )
        if resolved in live.parents:
            raise IsolatedRootError(
                f"refusing {purpose} root {resolved}: it CONTAINS the live state "
                f"directory {live}. Restore clears its target before writing, so "
                "this root would delete live state on the way in."
            )


def assert_neo4j_dev_isolated(uri: str) -> None:
    """Refuse a dev/hydration Neo4j target outside the dev-endpoint allowlist.

    Ungated on purpose -- see the module docstring. `assert_neo4j_isolated` has
    an activation flag because live runtime shares its call site; this guard's
    callers are hydration-only, so an "off" switch would only ever be a way to
    lose the guard.

    Raises:
        EvalIsolationError: When `uri` has no parseable host:port, or its
            (host, port) is not a known dev endpoint.
    """
    parsed = urlparse(uri)
    host = parsed.hostname
    try:
        port = parsed.port
    except ValueError:
        port = None
    if host is None or port is None:
        raise EvalIsolationError(
            f"Hydration target {uri!r} has no parseable host:port. Point it at the "
            "dev instance (e.g. bolt://mist-neo4j-dev:7687). Refusing to run."
        )
    if (host.lower(), port) not in _allowed_dev_endpoints():
        raise EvalIsolationError(
            f"Hydration target '{uri}' is not a recognized dev endpoint. Allowed: "
            f"{sorted(_allowed_dev_endpoints())} (override via MIST_DEV_NEO4J_HOSTS). "
            "The live bolt port is host-published, so only explicit dev endpoints "
            "pass. Refusing to avoid writing the canonical graph."
        )
