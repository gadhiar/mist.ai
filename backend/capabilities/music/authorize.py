"""The one-time Spotify authorization: `python -m backend.capabilities.music.authorize [--check]`.

MIST runs in a container with no browser, so the CLI runs the PKCE authorization-code flow in
manual-paste mode:

1. It prints the authorize URL, with an S256 challenge and the loopback redirect
   (`MIST_SPOTIFY_REDIRECT_URI`).
2. The user approves in any browser, which then fails to load the loopback address; the user
   pastes that address back, and the CLI checks its `state` against the one it sent.
3. It exchanges the code with the verifier and writes the token file
   (`MIST_SPOTIFY_TOKEN_PATH`).

`--check` refreshes the stored token and prints how many devices Spotify reports, which shows
that the token file can be read, renewed and written back.

Exit codes:

- 0: success;
- 1: the token file cannot be read (`--check`) or written;
- 2: a malformed paste or a `state` mismatch;
- 3: an HTTP error from the token exchange or refresh, or from the device list;
- 4: `MIST_SPOTIFY_CLIENT_ID` is unset, or the capability settings do not parse.

No token value is ever printed.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import secrets
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import parse_qs, urlsplit

import httpx

from backend.capabilities.clock import Clock, SystemClock
from backend.capabilities.config import load_capabilities_config
from backend.capabilities.secrets import SecretRegistry
from backend.errors import CapabilityConfigError, SpotifyError

from .pkce import SCOPES, PkcePair, authorize_url, generate_pkce, generate_state
from .spotify_api import SpotifyAccounts, SpotifyApi, SpotifyTokenManager
from .token_store import SpotifyTokenStore

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_BAD_PASTE = 2
EXIT_EXCHANGE_FAILED = 3
EXIT_NO_CLIENT_ID = 4

HTTP_TIMEOUT_S = 15.0
"""Bound on each request to Spotify; the CLI is interactive, so it fails rather than hangs."""


class _Reader(Protocol):
    """Where the paste is read: `sys.stdin`, or a `StringIO` in tests."""

    def readline(self) -> str:
        """Return one line, with its newline; empty at end of input."""
        ...


class _Writer(Protocol):
    """Where messages go: `sys.stdout`, or a `StringIO` in tests."""

    def write(self, text: str) -> int:
        """Write `text`."""
        ...

    def flush(self) -> None:
        """Push written text to the terminal."""
        ...


@dataclass(frozen=True, slots=True)
class _Settings:
    client_id: str
    redirect_uri: str
    token_path: str


class _BadPasteError(ValueError):
    """The pasted address carries no usable code, or a `state` other than the one sent."""


def main(
    argv: Sequence[str] | None = None,
    *,
    stdin: _Reader | None = None,
    stdout: _Writer | None = None,
    env: Mapping[str, str] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    clock: Clock | None = None,
    new_pkce: Callable[[], PkcePair] = generate_pkce,
    new_state: Callable[[], str] = generate_state,
) -> int:
    """Run the CLI and return its exit code.

    Args:
        argv: The arguments after the program name; `sys.argv[1:]` when None.
        stdin: Where the pasted redirect address is read; `sys.stdin` when None.
        stdout: Where every message goes; `sys.stdout` when None.
        env: The environment the settings come from; `os.environ` when None.
        transport: The HTTP transport; httpx's default network transport when None. Tests
            pass an `httpx.MockTransport`.
        clock: Turns `expires_in` into the stored expiry; `SystemClock` when None.
        new_pkce: Makes the verifier and challenge.
        new_state: Makes the `state` value.
    """
    parser = argparse.ArgumentParser(
        prog="python -m backend.capabilities.music.authorize",
        description="Authorize MIST's Spotify access, or check the stored authorization.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="refresh the stored token and print how many devices Spotify reports",
    )
    options = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    out = stdout or sys.stdout
    settings = _settings(os.environ if env is None else env, out)
    if settings is None:
        return EXIT_NO_CLIENT_ID
    run = _Run(settings, out, clock or SystemClock(), transport)
    if options.check:
        return asyncio.run(run.check())
    return asyncio.run(run.authorize(stdin or sys.stdin, new_pkce(), new_state()))


def _settings(env: Mapping[str, str], out: _Writer) -> _Settings | None:
    try:
        music = load_capabilities_config(env).music
    except CapabilityConfigError as exc:
        _say(out, f"The capability settings do not parse: {exc}")
        return None
    if music.spotify_client_id is None:
        _say(
            out,
            "MIST_SPOTIFY_CLIENT_ID is unset. Create an app at developer.spotify.com, register "
            f"the redirect URI {music.spotify_redirect_uri}, and set its client id.",
        )
        return None
    return _Settings(
        client_id=music.spotify_client_id,
        redirect_uri=music.spotify_redirect_uri,
        token_path=music.spotify_token_path,
    )


class _Run:
    def __init__(
        self,
        settings: _Settings,
        out: _Writer,
        clock: Clock,
        transport: httpx.AsyncBaseTransport | None,
    ) -> None:
        self._settings = settings
        self._out = out
        self._clock = clock
        self._transport = transport
        self._store = SpotifyTokenStore(Path(settings.token_path))
        # The registry is required by the accounts client; nothing here logs a token, and the
        # CLI prints none.
        self._registry = SecretRegistry()

    def _http(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=self._transport, timeout=HTTP_TIMEOUT_S)

    def _accounts(self, http: httpx.AsyncClient) -> SpotifyAccounts:
        return SpotifyAccounts(
            http, client_id=self._settings.client_id, registry=self._registry, clock=self._clock
        )

    async def authorize(self, stdin: _Reader, pkce: PkcePair, state: str) -> int:
        settings = self._settings
        url = authorize_url(
            client_id=settings.client_id,
            redirect_uri=settings.redirect_uri,
            challenge=pkce.challenge,
            state=state,
        )
        _say(self._out, "Open this address in a browser, sign in to Spotify and approve MIST:")
        _say(self._out, "")
        _say(self._out, url)
        _say(self._out, "")
        _say(
            self._out,
            f"The browser then tries to open {settings.redirect_uri}?code=... and fails to load"
            " it, which is expected. Paste that whole address here and press Enter:",
        )
        try:
            code = _code_from_paste(stdin.readline(), state)
        except _BadPasteError as exc:
            _say(self._out, f"Not authorized: {exc}")
            return EXIT_BAD_PASTE
        async with self._http() as http:
            try:
                token = await self._accounts(http).exchange_code(
                    code=code, verifier=pkce.verifier, redirect_uri=settings.redirect_uri
                )
            except SpotifyError as exc:
                _say(self._out, f"The code exchange failed: {exc}")
                return EXIT_EXCHANGE_FAILED
        try:
            await asyncio.to_thread(self._store.save, token)
        except SpotifyError as exc:
            _say(self._out, f"Authorized, but the token was not saved: {exc}")
            return EXIT_FAILED
        missing = [scope for scope in SCOPES if scope not in token.scope.split()]
        if missing:
            _say(self._out, f"Spotify did not grant {', '.join(missing)}; some tools will fail.")
        _say(self._out, f"Spotify authorized. The token is saved at {self._store.path}.")
        return EXIT_OK

    async def check(self) -> int:
        try:
            token = await asyncio.to_thread(self._store.load)
        except SpotifyError as exc:
            _say(self._out, f"No usable token: {exc}")
            return EXIT_FAILED
        self._registry.register(token.access_token)
        self._registry.register(token.refresh_token)
        async with self._http() as http:
            accounts = self._accounts(http)
            try:
                renewed = await accounts.refresh(token)
            except SpotifyError as exc:
                _say(self._out, f"The token refresh failed: {exc}")
                return EXIT_EXCHANGE_FAILED
            try:
                await asyncio.to_thread(self._store.save, renewed)
            except SpotifyError as exc:
                _say(self._out, f"The token refreshed but was not saved: {exc}")
                return EXIT_FAILED
            tokens = SpotifyTokenManager(
                store=self._store, accounts=accounts, registry=self._registry, clock=self._clock
            )
            try:
                devices = await SpotifyApi(http, tokens).devices()
            except SpotifyError as exc:
                _say(self._out, f"The token refreshed, but the device list failed: {exc}")
                return EXIT_EXCHANGE_FAILED
        noun = "device" if len(devices) == 1 else "devices"
        _say(self._out, f"Spotify token refreshed and saved; {len(devices)} {noun} visible.")
        return EXIT_OK


def _code_from_paste(paste: str, state: str) -> str:
    """Return the authorization code from the pasted redirect address.

    The address may be pasted whole or as its query string alone.

    Raises:
        _BadPasteError: The paste is blank, carries Spotify's `error`, lacks a code, or carries
            a `state` other than `state`, which would mean the code answers another request.
    """
    text = paste.strip()
    if not text:
        raise _BadPasteError("nothing was pasted")
    query = urlsplit(text).query if "?" in text else text
    params = parse_qs(query)
    got_state = params.get("state", [""])[0]
    if not secrets.compare_digest(got_state.encode(), state.encode()):
        raise _BadPasteError("the pasted address does not carry the state this run sent")
    if "error" in params:
        raise _BadPasteError(f"Spotify answered {params['error'][0]!r} instead of a code")
    code = params.get("code", [""])[0]
    if not code:
        raise _BadPasteError("the pasted address carries no code")
    return code


def _say(out: _Writer, line: str) -> None:
    out.write(f"{line}\n")
    out.flush()


if __name__ == "__main__":
    raise SystemExit(main())
