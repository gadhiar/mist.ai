"""The Spotify token file: the one place a refresh token outlives the process.

The file is replaced atomically, so a crash mid-write leaves the previous token rather than a
truncated one; Spotify can rotate the refresh token on any refresh, and losing the new one would
need a fresh authorization. The file is mode 0600 because it grants control of Raj's account.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from backend.capabilities.secrets import Secret
from backend.errors import SpotifyAuthError, SpotifyError

AUTHORIZE_COMMAND = "python -m backend.capabilities.music.authorize"
NOT_AUTHORIZED = "not_authorized"

_FILE_MODE = 0o600
_FIELDS = ("access_token", "refresh_token", "expires_at", "scope", "token_type")


@dataclass(frozen=True, slots=True)
class SpotifyToken:
    """An access token, the refresh token that renews it, and when the access token expires."""

    access_token: Secret
    refresh_token: Secret
    expires_at: datetime  # aware UTC
    scope: str
    token_type: str = "Bearer"


def not_authorized(detail: str) -> SpotifyAuthError:
    """Build the error every unusable-token path raises; it names the CLI that fixes it.

    The message starts with the reason word `not_authorized`, which the music capability
    reports as the outcome reason.
    """
    return SpotifyAuthError(f"{NOT_AUTHORIZED}: {detail}; run `{AUTHORIZE_COMMAND}`")


class SpotifyTokenStore:
    """Reads and atomically writes one token file.

    Methods are synchronous file I/O on a file of a few hundred bytes; async callers run them
    through `asyncio.to_thread`.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        """The token file's path."""
        return self._path

    def load(self) -> SpotifyToken:
        """Return the stored token.

        Raises:
            SpotifyAuthError: the file is missing, unreadable or not a valid token; the message
                names the authorize CLI. The message never holds file content, which may be a
                token.
        """
        try:
            text = self._path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise not_authorized(f"no Spotify token file at {self._path}") from exc
        except OSError as exc:
            raise not_authorized(
                f"the Spotify token file at {self._path} cannot be read ({type(exc).__name__})"
            ) from exc
        try:
            return _decode(text)
        except ValueError as exc:
            # `from None`: a JSON error's message quotes the document, which may hold a token.
            raise not_authorized(
                f"the Spotify token file at {self._path} is not a valid token ({exc})"
            ) from None

    def save(self, token: SpotifyToken) -> None:
        """Write `token` through a temp file in the same directory, then `os.replace`.

        Raises:
            SpotifyError: the directory or file cannot be written; the previous file, if any, is
                left as it was.
        """
        content = json.dumps(_encode(token), indent=2) + "\n"
        directory = self._path.parent
        temp_name: str | None = None
        try:
            directory.mkdir(parents=True, exist_ok=True)
            # mkstemp creates the file 0600, so the token is never readable by others, even
            # before the chmod below.
            fd, temp_name = tempfile.mkstemp(
                dir=directory, prefix=f".{self._path.name}.", suffix=".tmp"
            )
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temp_name, _FILE_MODE)
            os.replace(temp_name, self._path)
            temp_name = None
            os.chmod(self._path, _FILE_MODE)
        except OSError as exc:
            raise SpotifyError(
                f"the Spotify token file at {self._path} cannot be written ({type(exc).__name__})"
            ) from exc
        finally:
            if temp_name is not None:
                Path(temp_name).unlink(missing_ok=True)


def _encode(token: SpotifyToken) -> dict[str, str]:
    return {
        "access_token": token.access_token.reveal(),
        "refresh_token": token.refresh_token.reveal(),
        "expires_at": token.expires_at.isoformat(),
        "scope": token.scope,
        "token_type": token.token_type,
    }


def _decode(text: str) -> SpotifyToken:
    """Parse the file; every failure is a `ValueError` whose message holds no field value."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        raise ValueError("not JSON") from None
    if not isinstance(data, dict):
        raise ValueError("not a JSON object")
    missing = [name for name in _FIELDS if name not in data]
    if missing:
        raise ValueError(f"missing {', '.join(missing)}")
    for name in _FIELDS:
        if not isinstance(data[name], str):
            raise ValueError(f"{name} is not a string")
    for name in ("access_token", "refresh_token", "expires_at"):
        if not data[name]:
            raise ValueError(f"{name} is empty")
    try:
        expires_at = datetime.fromisoformat(data["expires_at"])
    except ValueError:
        raise ValueError("expires_at is not an ISO 8601 time") from None
    if expires_at.tzinfo is None:
        raise ValueError("expires_at has no time zone")
    return SpotifyToken(
        access_token=Secret(data["access_token"]),
        refresh_token=Secret(data["refresh_token"]),
        expires_at=expires_at.astimezone(UTC),
        scope=data["scope"],
        token_type=data["token_type"],
    )
