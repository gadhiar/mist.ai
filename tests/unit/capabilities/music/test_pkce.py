"""Tests for the PKCE verifier, S256 challenge, `state` and authorize URL."""

from __future__ import annotations

import base64
import re
from urllib.parse import parse_qs, urlsplit

import pytest

from backend.capabilities.music.pkce import (
    AUTHORIZE_URL,
    SCOPES,
    authorize_url,
    generate_pkce,
    generate_state,
    generate_verifier,
    s256_challenge,
)

# RFC 7636 appendix B: the 32 octets, the verifier they encode, and its S256 challenge.
RFC_OCTETS = bytes(
    [
        116, 24, 223, 180, 151, 153, 224, 37, 79, 250, 96, 125, 216, 173, 187, 186,
        22, 212, 37, 77, 105, 214, 191, 240, 91, 88, 5, 88, 83, 132, 141, 121,
    ]
)  # fmt: skip
RFC_VERIFIER = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
RFC_CHALLENGE = "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"

UNRESERVED = re.compile(r"^[A-Za-z0-9\-._~]+$")
CLIENT_ID = "fake-client-id-0001"
REDIRECT = "http://127.0.0.1:8888/callback"


def test_rfc_vector_verifier_encodes_the_appendix_octets() -> None:
    assert base64.urlsafe_b64encode(RFC_OCTETS).rstrip(b"=").decode("ascii") == RFC_VERIFIER


def test_s256_challenge_matches_rfc_7636_appendix_b() -> None:
    assert s256_challenge(RFC_VERIFIER) == RFC_CHALLENGE


def test_default_verifier_length_is_within_43_to_128_and_unreserved() -> None:
    verifier = generate_verifier()
    assert 43 <= len(verifier) <= 128
    assert UNRESERVED.match(verifier)


@pytest.mark.parametrize("length", [43, 64, 128])
def test_verifier_honours_lengths_inside_the_range(length: int) -> None:
    verifier = generate_verifier(length)
    assert len(verifier) == length
    assert UNRESERVED.match(verifier)


@pytest.mark.parametrize("length", [0, 42, 129])
def test_verifier_refuses_lengths_outside_the_range(length: int) -> None:
    with pytest.raises(ValueError, match="43"):
        generate_verifier(length)


def test_verifiers_differ_per_call() -> None:
    assert len({generate_verifier() for _ in range(20)}) == 20


def test_generate_pkce_pairs_a_verifier_with_its_challenge() -> None:
    pair = generate_pkce()
    assert 43 <= len(pair.verifier) <= 128
    assert pair.challenge == s256_challenge(pair.verifier)
    assert "=" not in pair.challenge


def test_state_is_random_per_call_and_url_safe() -> None:
    states = [generate_state() for _ in range(20)]
    assert len(set(states)) == 20
    for state in states:
        assert len(state) >= 32
        assert re.match(r"^[A-Za-z0-9_\-]+$", state)


def test_authorize_url_carries_every_pkce_parameter() -> None:
    url = authorize_url(
        client_id=CLIENT_ID,
        redirect_uri=REDIRECT,
        challenge=RFC_CHALLENGE,
        state="fake-state-0001",
    )
    parts = urlsplit(url)
    assert f"{parts.scheme}://{parts.netloc}{parts.path}" == AUTHORIZE_URL
    query = parse_qs(parts.query, strict_parsing=True)
    assert query == {
        "client_id": [CLIENT_ID],
        "response_type": ["code"],
        "redirect_uri": [REDIRECT],
        "code_challenge_method": ["S256"],
        "code_challenge": [RFC_CHALLENGE],
        "state": ["fake-state-0001"],
        "scope": [" ".join(SCOPES)],
    }
    assert set(SCOPES) == {
        "user-read-playback-state",
        "user-modify-playback-state",
        "user-read-currently-playing",
    }
