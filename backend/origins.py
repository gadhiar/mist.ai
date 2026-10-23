"""The browser origins allowed to reach the backend over HTTP and the WebSocket.

Browsers apply CORS to HTTP but not to WebSockets: a page on any site can open
`ws://localhost:8001/ws`, and the browser sends its `Origin` header without
enforcing anything. Only the server can refuse it. Without that check, any page
open on the host could drive MIST, read what she answers from memory, and add
turns that count toward the real-use gate (ADR-042).

A client that sends no `Origin` at all is not a browser page (a browser always
sends one on a WebSocket upgrade), so it is a local program such as
`scripts/mist_admin.py`, and it is allowed.
"""

from __future__ import annotations

import os

ALLOWED_ORIGINS_ENV = "MIST_ALLOWED_ORIGINS"

# The Tauri webview's origins (`http://tauri.localhost` on Windows,
# `tauri://localhost` on macOS and Linux, and the `https` form Tauri can be
# configured to use), plus the Vite dev server the workstation frontend runs on.
DEFAULT_ALLOWED_ORIGINS: tuple[str, ...] = (
    "http://tauri.localhost",
    "https://tauri.localhost",
    "tauri://localhost",
    "http://localhost:1420",
    "http://127.0.0.1:1420",
)

# The entry that allows every origin, as `CORSMiddleware` reads it too.
ALLOW_ALL = "*"


def allowed_origins() -> tuple[str, ...]:
    """Read the allowlist from `MIST_ALLOWED_ORIGINS`, or return the default.

    The value is comma-separated and replaces the default rather than adding to
    it, so a deployment can narrow the list (the daily compose passes the Tauri
    origins only). Entries are stripped and lowercased, and empty entries are
    dropped. A blank value counts as unset, matching compose's `${VAR:-default}`.
    An entry of `*` allows every origin, the behaviour before this allowlist.

    Read at call time, so the `/ws` check and `configure_cors` see the
    environment of the moment they run.

    Returns:
        The allowed origins, lowercased, in the order given.
    """
    raw = os.environ.get(ALLOWED_ORIGINS_ENV, "")
    entries = tuple(entry.strip().lower() for entry in raw.split(",") if entry.strip())
    return entries if entries else DEFAULT_ALLOWED_ORIGINS


def origin_allowed(origin: str | None, allowed: tuple[str, ...]) -> bool:
    """Whether a request carrying `origin` may be served.

    Matching is exact after lowercasing, never by prefix or suffix, so
    `http://tauri.localhost.evil.example` does not match `http://tauri.localhost`.
    The opaque origin `null`, which sandboxed frames and `file://` pages send,
    matches nothing unless it is listed.

    Args:
        origin: The request's `Origin` header, or None when it sent none.
        allowed: The allowlist, as `allowed_origins()` returns it.

    Returns:
        True for None, for any origin when `allowed` holds `*`, or for an origin
        that equals an entry after lowercasing.
    """
    if origin is None:
        return True
    if ALLOW_ALL in allowed:
        return True
    return origin.lower() in {entry.lower() for entry in allowed}
