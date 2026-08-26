"""Fail-closed isolation guards for runs that must not touch live state.

Five guards live here, one per way a non-live run can reach live state. Four
of the five are ALLOWLISTS over `(host, port)`, and that uniformity is the
point: the remaining one was a denylist, and the denylist was the one with the
hole.

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

**Backend WebSocket, hydration (`assert_ws_target_not_live`).** The hydrator's
`--ws-url`. The only guard over HTTP rather than bolt, and the only one whose
subject is the backend rather than a database. Live publishes 8001, dev
publishes 8002; one digit, and the dev stack sets MIST_SESSION_ORIGIN=real, so
a misdirected run is unrecoverable rather than merely wrong. Pure -- the
network handshake that pairs with it is `scripts/hydration/target.py`.

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

# The hydrator's WebSocket target (F3). `mist-backend-dev:8001` is the
# container-internal spelling; `localhost:8002` is the host spelling the
# dev-hydration runbook documents. Override via MIST_DEV_WS_HOSTS.
DEFAULT_DEV_WS_ENDPOINTS = "mist-backend-dev:8001,localhost:8002,127.0.0.1:8002"

# The canonical graph, in every spelling that reaches it.
#
# DELIBERATELY NOT env-overridable, and deliberately not derived from config.
# This is a DENYLIST, checked BEFORE every allowlist in this module, and a
# denylist an operator can empty is an allowlist wearing a different name.
#
# Why it is needed on top of the allowlists: every `_parse_endpoint_allowlist`
# override REPLACES its allowlist rather than extending it (see
# `test_allowlist_is_env_overridable`). So a single environment variable could
# admit a live endpoint to a guard that gates a `DETACH DELETE`.
# `assert_rebuild_target_not_live` already had a second arm against that, but it
# compares against a `live_uri` its CALLER supplies -- and
# `cmd_graph_rebuild_from_log` infers that from ambient config, which inside
# `mist-backend-dev` resolves to the DEV instance, making the arm vacuous in the
# only process where the rebuild can run. `assert_neo4j_dev_isolated` had no
# second arm at all, while gating the more destructive tool (snapshot restore).
#
# `localhost:7687` and `127.0.0.1:7687` are here because the live bolt port is
# host-published in `docker-compose.yml`, so they are the SAME DATABASE as
# `mist-neo4j:7687`. That equivalence is what made the original live-graph loss
# possible: the pre-allowlist guard compared URI strings, so the host spelling
# passed a check the service-name spelling failed.
LIVE_NEO4J_ENDPOINTS: frozenset[tuple[str, int]] = frozenset(
    {
        ("mist-neo4j", 7687),
        ("localhost", 7687),
        ("127.0.0.1", 7687),
    }
)

# The live BACKEND, same treatment (F3). `docker-compose.yml:11` publishes
# `8001:8001`; the dev override publishes `8002:8001`. Host-side the two differ
# by one digit, and the dev stack sets MIST_SESSION_ORIGIN=real, so turns
# misdirected here are indistinguishable from genuine usage and un-excludable
# from every future rebuild.
#
# Note `localhost:8001` is refused even though, from INSIDE the dev container,
# it names the dev backend. That is a false refusal, not a false pass, and the
# documented runbook is host-side. Fail-closed in the safe direction.
LIVE_WS_ENDPOINTS: frozenset[tuple[str, int]] = frozenset(
    {
        ("mist-backend", 8001),
        ("localhost", 8001),
        ("127.0.0.1", 8001),
    }
)

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"", "0", "false", "no", "off"}


def _assert_not_live_endpoint(
    host: str,
    port: int,
    uri: str,
    error_cls: type[Exception],
    *,
    action: str,
) -> None:
    """Refuse any (host, port) naming the canonical graph, whatever the allowlists say.

    Args:
        host: Parsed hostname, compared case-insensitively.
        port: Parsed port. A portless URI must be refused by the caller before
            reaching here -- live and the disposable instances differ only by
            port on the host, so a missing port cannot be told apart from live.
        uri: The original URI, for the message.
        error_cls: Raised type. Callers pass the error their own CLI catches, so
            a refusal reads as a refusal rather than a crash.
        action: What the caller was about to do, named in the message so an
            operator knows which tool refused and why.
    """
    if (host.lower(), port) in LIVE_NEO4J_ENDPOINTS:
        raise error_cls(
            f"{uri!r} is the live graph, and {action} would destroy it. Refused by "
            f"the hardcoded live denylist {sorted(LIVE_NEO4J_ENDPOINTS)}, which no "
            "environment variable can widen or empty -- including the "
            "MIST_*_NEO4J_HOSTS overrides, which REPLACE their allowlists and so "
            "could otherwise admit this endpoint. The live bolt port is "
            "host-published, so 'localhost:7687' and 'mist-neo4j:7687' are the same "
            "database."
        )


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
    return parse_isolation_flag("MIST_EVAL_ISOLATION")


def parse_isolation_flag(env_var: str) -> bool:
    """Parse one isolation activation flag, failing closed on anything unrecognized.

    Shared by `is_eval_isolation_active` and `is_hydration_isolation_active` so
    there is ONE definition of what "isolation on" means. Two independent
    parses of the same concept can disagree, and a disagreement here reads as
    "isolated" to one caller and "not isolated" to another.

    Args:
        env_var: Name of the variable to read. Named in the error message so an
            operator knows which one they mistyped.

    Raises:
        EvalIsolationError: for unrecognized values. A fail-closed guard must
            not treat an operator typo ('ture', 'enabled') as "isolation off"
            and silently run unguarded against the live graph.
    """
    raw = os.getenv(env_var, "")
    value = raw.strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    raise EvalIsolationError(
        f"Unrecognized {env_var} value {raw!r}; use '1' to activate "
        "or unset to deactivate. Refusing to guess for a fail-closed guard."
    )


def is_hydration_isolation_active() -> bool:
    """True when this process is running as the hydration target (F4).

    Set to `1` on `mist-backend-dev` in `docker-compose.dev-hydration.yml`. The
    LIVE backend never sets it, and that asymmetry is the whole mechanism: the
    hydrator reads this back off its target's `/health` and refuses to send a
    single turn unless it is True.

    Before F4 the variable was decorative -- set in the dev compose under a
    comment claiming it "marks this container as the hydration target", with no
    production reader anywhere. It converts a mistyped `--ws-url` (live is
    `:8001`, dev is `:8002`) from "87 fictional turns written to the live event
    store" into "refused in under a second".

    The turns would have been unrecoverable, not merely wrong: the dev stack
    sets `MIST_SESSION_ORIGIN=real`, so they would be indistinguishable from
    genuine usage and un-excludable from every future rebuild.
    """
    return parse_isolation_flag("MIST_HYDRATION_ISOLATION")


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


def assert_neo4j_uri_not_live(uri: str, *, action: str) -> None:
    """Refuse `uri` if it names the canonical graph, whatever else is true of it.

    The denylist arm on its own, exposed for callers that need "not live"
    without also asserting "and is one of MY disposable endpoints". Seed-apply
    is the motivating case (F1): it legitimately targets staging, dev, AND live,
    so an allowlist would be wrong -- only the live case needs to be spelled out
    by the caller.

    Args:
        uri: A bolt URI. A portless or unparsable URI is refused: live and the
            disposable instances differ only by port on the host.
        action: What the caller was about to do, named in the refusal so an
            operator knows which tool refused and why.

    Raises:
        EvalIsolationError: when the URI names live, or cannot be parsed well
            enough to prove it does not.
    """
    parsed = urlparse(uri)
    host = parsed.hostname
    try:
        port = parsed.port
    except ValueError:
        port = None
    if host is None or port is None:
        raise EvalIsolationError(
            f"{uri!r} has no parseable host:port, so {action} cannot be shown to "
            "miss the live graph. Live and the disposable instances differ only "
            "by port on the host. Refusing rather than guessing."
        )
    _assert_not_live_endpoint(host, port, uri, EvalIsolationError, action=action)


def assert_ws_target_not_live(ws_url: str) -> None:
    """Refuse a hydrator WebSocket target that is, or might be, the live backend (F3).

    The fifth guard, and the only one over HTTP rather than bolt. Same shape as
    its siblings: a hardcoded denylist that no environment variable can widen,
    checked BEFORE an env-overridable allowlist. That ordering is F5's lesson --
    `MIST_DEV_WS_HOSTS` REPLACES the allowlist wholesale, so an allowlist-only
    guard could be pointed back at live by the very override meant to widen it
    for CI.

    Pure by design: no network call. The handshake that DOES touch the network
    lives in `scripts/hydration/target.py` and calls this first, so a hydrator
    aimed at live is refused without ever contacting live.

    Args:
        ws_url: The `--ws-url` the operator passed. Required, never defaulted --
            a default is a value nobody checked.

    Raises:
        EvalIsolationError: when the target names the live backend, is not a
            recognized dev endpoint, or has no parseable host:port. Live and dev
            differ ONLY by port on the host, so a portless URL is unresolvable
            and is refused rather than defaulted.
    """
    parsed = urlparse(ws_url)
    host = parsed.hostname
    try:
        port = parsed.port
    except ValueError:
        port = None
    if host is None or port is None:
        raise EvalIsolationError(
            f"--ws-url {ws_url!r} has no parseable host:port. Live and dev differ "
            "only by port on the host (live 8001, dev 8002), so a missing port "
            "cannot be told apart from live. Pass the full URL, e.g. "
            "ws://localhost:8002/ws. Refusing to guess."
        )
    if (host.lower(), port) in LIVE_WS_ENDPOINTS:
        raise EvalIsolationError(
            f"--ws-url {ws_url!r} is the live backend, and hydration would write "
            f"authored turns into the live event store. Refused by the hardcoded "
            f"denylist {sorted(LIVE_WS_ENDPOINTS)}, which no environment variable "
            "can widen or empty -- including MIST_DEV_WS_HOSTS, which REPLACES "
            "its allowlist and so could otherwise admit this endpoint. The dev "
            "stack sets MIST_SESSION_ORIGIN=real, so such turns would be "
            "indistinguishable from genuine usage and un-excludable from every "
            "future rebuild. Use the dev port (ws://localhost:8002/ws)."
        )
    allowed = _parse_endpoint_allowlist("MIST_DEV_WS_HOSTS", DEFAULT_DEV_WS_ENDPOINTS)
    if (host.lower(), port) not in allowed:
        raise EvalIsolationError(
            f"--ws-url {ws_url!r} is not a recognized hydration endpoint. "
            f"Allowed: {sorted(allowed)} (override via MIST_DEV_WS_HOSTS). "
            "Allowlist, not denylist: an unrecognized target fails closed rather "
            "than falling through."
        )


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
    admitting it as a WRITE target would let a rebuild delete the hydrated
    fixture. (This sentence said "an 87-turn hydrated fixture" until 2026-08-26.
    No such fixture exists or ever did: the dev graph is 4 nodes and 1
    relationship with 0 conversation turns, per
    `data/hydration-snapshots/r1.4.6-smoke/manifest.json`. The 87 belongs to the
    golden log, an authored corpus. The exclusion is correct on its own merits;
    only the stated size was invented.)

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

    # Hardcoded denylist FIRST: the two arms below both depend on inputs a
    # caller or an operator controls (`live_uri`, and the env-overridable
    # allowlist). This one depends on neither.
    _assert_not_live_endpoint(
        target.hostname,
        target_port,
        target_uri,
        RebuildTargetError,
        action="a rebuild's unconditional MATCH (n) DETACH DELETE n",
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
    # Hardcoded denylist FIRST -- see LIVE_NEO4J_ENDPOINTS. The allowlist below
    # is env-overridable and the override REPLACES it, so without this arm one
    # variable could point this guard's callers at the canonical graph. Those
    # callers include `snapshot restore`, which runs
    # `MATCH (n) WITH n LIMIT 10000 DETACH DELETE n`.
    _assert_not_live_endpoint(
        host,
        port,
        uri,
        EvalIsolationError,
        action="a hydration or snapshot-restore write",
    )

    if (host.lower(), port) not in _allowed_dev_endpoints():
        raise EvalIsolationError(
            f"Hydration target '{uri}' is not a recognized dev endpoint. Allowed: "
            f"{sorted(_allowed_dev_endpoints())} (override via MIST_DEV_NEO4J_HOSTS). "
            "The live bolt port is host-published, so only explicit dev endpoints "
            "pass. Refusing to avoid writing the canonical graph."
        )
