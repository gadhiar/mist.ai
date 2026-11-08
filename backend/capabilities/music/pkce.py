"""PKCE (RFC 7636) helpers for Spotify's authorization-code flow.

MIST is a public client: it holds a client id and no client secret, so the code exchange is
bound to a one-time verifier instead. Only the S256 method is offered, because `plain` would
send the verifier itself through the browser.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import string
from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import urlencode

AUTHORIZE_URL = "https://accounts.spotify.com/authorize"

# What the music capability needs: read the player and devices, control playback, and read
# the current item.
SCOPES: tuple[str, ...] = (
    "user-read-playback-state",
    "user-modify-playback-state",
    "user-read-currently-playing",
)

# RFC 7636 section 4.1: the verifier uses the unreserved URI characters, 43 to 128 of them.
_VERIFIER_ALPHABET = string.ascii_letters + string.digits + "-._~"
MIN_VERIFIER_LENGTH = 43
MAX_VERIFIER_LENGTH = 128
DEFAULT_VERIFIER_LENGTH = 64

# 32 random bytes, the same entropy RFC 7636 recommends for the verifier.
_STATE_BYTES = 32


@dataclass(frozen=True, slots=True)
class PkcePair:
    """A verifier, kept by the client, and the challenge sent in the authorize URL."""

    verifier: str
    challenge: str


def generate_verifier(length: int = DEFAULT_VERIFIER_LENGTH) -> str:
    """Return a random verifier of `length` unreserved characters.

    Raises:
        ValueError: `length` is outside 43 to 128, which RFC 7636 forbids.
    """
    if not MIN_VERIFIER_LENGTH <= length <= MAX_VERIFIER_LENGTH:
        raise ValueError(
            f"a PKCE verifier has {MIN_VERIFIER_LENGTH} to {MAX_VERIFIER_LENGTH} characters,"
            f" got {length}"
        )
    return "".join(secrets.choice(_VERIFIER_ALPHABET) for _ in range(length))


def s256_challenge(verifier: str) -> str:
    """Return BASE64URL(SHA256(verifier)) without padding (RFC 7636 section 4.2)."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def generate_pkce(length: int = DEFAULT_VERIFIER_LENGTH) -> PkcePair:
    """Return a fresh verifier and its S256 challenge."""
    verifier = generate_verifier(length)
    return PkcePair(verifier=verifier, challenge=s256_challenge(verifier))


def generate_state() -> str:
    """Return a fresh, URL-safe `state` value; the caller checks it on the redirect."""
    return secrets.token_urlsafe(_STATE_BYTES)


def authorize_url(
    *,
    client_id: str,
    redirect_uri: str,
    challenge: str,
    state: str,
    scopes: Sequence[str] = SCOPES,
) -> str:
    """Return the Spotify authorize URL for an S256 PKCE authorization-code request."""
    query = urlencode(
        {
            "client_id": client_id,
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "code_challenge_method": "S256",
            "code_challenge": challenge,
            "state": state,
            "scope": " ".join(scopes),
        }
    )
    return f"{AUTHORIZE_URL}?{query}"
