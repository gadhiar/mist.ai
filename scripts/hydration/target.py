"""F3/F4: the hydrator's positive target handshake.

Written BEFORE the hydrator's send loop, deliberately. The closure design's
corrected sequencing (`docs/superpowers/specs/2026-08-26-r1.6-closure-design.md`
section 5, step 2) puts this first because "the handshake must exist before
there is anything to misdirect" -- a guard added after the loop it protects has
already had a window in which it did not exist.

Two layers, and they fail in different directions on purpose:

- `assert_ws_target_not_live` (in `backend/knowledge/eval_isolation.py`, pure)
  refuses the live spellings we thought of. It runs FIRST, so a hydrator aimed
  at live never opens a socket to live -- not even to ask permission.
- `assert_hydration_target` (here) asks the target to identify itself and
  refuses unless it says yes. It catches the spellings we did not think of,
  because it does not reason about the URL at all.

A denylist can only refuse what it enumerates. The handshake inverts that: the
live backend never sets MIST_HYDRATION_ISOLATION, so it cannot answer yes no
matter how it is addressed.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from urllib.parse import urlparse, urlunparse

from backend.knowledge.eval_isolation import assert_ws_target_not_live

logger = logging.getLogger(__name__)

DEFAULT_HANDSHAKE_TIMEOUT_SECONDS = 5.0

_WS_TO_HTTP = {"ws": "http", "wss": "https"}


class HydrationTargetError(RuntimeError):
    """Raised when a hydration target cannot be confirmed as the dev container."""


def health_url_for(ws_url: str) -> str:
    """Map a `--ws-url` onto the `/health` URL on the same host:port.

    `wss` maps to `https`, never `http`: silently downgrading the scheme of a
    security-relevant check is how a handshake ends up trivially spoofable.
    """
    parsed = urlparse(ws_url)
    scheme = _WS_TO_HTTP.get(parsed.scheme.lower())
    if scheme is None:
        raise HydrationTargetError(
            f"--ws-url {ws_url!r} has scheme {parsed.scheme!r}; expected ws or wss."
        )
    return urlunparse((scheme, parsed.netloc, "/health", "", "", ""))


def _fetch_health(url: str, timeout: float) -> dict:
    """GET `url` and parse it as JSON. Seam for tests; the only I/O in this module."""
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def assert_hydration_target(
    ws_url: str,
    *,
    timeout: float = DEFAULT_HANDSHAKE_TIMEOUT_SECONDS,
) -> None:
    """Refuse unless `ws_url` is a dev endpoint AND identifies as the hydration target.

    Call this once before the first turn is sent. It is cheap (one local HTTP
    GET) and the failure it prevents is unrecoverable: 87 authored turns in the
    live event store, stamped MIST_SESSION_ORIGIN=real, un-excludable from every
    future rebuild.

    Args:
        ws_url: The operator's `--ws-url`. Required upstream, never defaulted.
        timeout: Seconds to wait for `/health`. Short by design -- this runs
            against a container on the same host, and a long timeout turns a
            misconfiguration into a hang.

    Raises:
        EvalIsolationError: the URL names live, is unrecognized, or is
            unparsable. Raised BEFORE any network call.
        HydrationTargetError: the target could not be reached, answered
            unparseably, or did not report `hydration_isolation: true`.
    """
    assert_ws_target_not_live(ws_url)

    url = health_url_for(ws_url)
    try:
        payload = _fetch_health(url, timeout=timeout)
    except (urllib.error.URLError, OSError) as exc:
        raise HydrationTargetError(
            f"Hydration target {url!r} could not be reached ({exc}). No answer is "
            "not a yes: refusing rather than sending turns to a backend that has "
            "not identified itself. Is the dev stack up? See "
            "docker-compose.dev-hydration.yml."
        ) from exc
    except (json.JSONDecodeError, ValueError) as exc:
        raise HydrationTargetError(
            f"Hydration target {url!r} returned a response that is not JSON "
            f"({exc}). Refusing an unverifiable target."
        ) from exc

    if "hydration_isolation" not in payload:
        raise HydrationTargetError(
            f"Hydration target {url!r} did not report `hydration_isolation`. That "
            "field is how a backend identifies itself as the hydration container; "
            "a backend too old to report it cannot be confirmed, and an "
            "unconfirmed target is refused. Rebuild the dev image."
        )

    if payload["hydration_isolation"] is not True:
        raise HydrationTargetError(
            f"Hydration target {url!r} reports hydration_isolation="
            f"{payload['hydration_isolation']!r}. Only the dev container sets "
            "MIST_HYDRATION_ISOLATION=1; the live backend never does. Either this "
            "is the live backend, or the dev container was started without the "
            "flag. Refusing to send turns to an unconfirmed target."
        )

    logger.info("Hydration target confirmed: %s (hydration_isolation=true)", url)
