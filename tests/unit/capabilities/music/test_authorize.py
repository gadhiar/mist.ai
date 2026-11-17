"""Tests for the Spotify authorize CLI (`python -m backend.capabilities.music.authorize`).

stdin, stdout, the environment and the HTTP transport are injected, so no browser, terminal or
network is involved. Client ids, codes and tokens are synthetic.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from backend.capabilities.music.authorize import (
    EXIT_BAD_PASTE,
    EXIT_EXCHANGE_FAILED,
    EXIT_FAILED,
    EXIT_NO_CLIENT_ID,
    EXIT_OK,
    main,
)
from backend.capabilities.music.pkce import AUTHORIZE_URL, SCOPES, PkcePair, s256_challenge
from backend.capabilities.music.spotify_api import API_BASE, TOKEN_URL
from backend.capabilities.music.token_store import SpotifyToken, SpotifyTokenStore
from backend.capabilities.secrets import Secret
from tests.unit.capabilities.fakes import FakeClock

CLIENT_ID = "fake-client-id-0001"
REDIRECT = "http://127.0.0.1:8888/callback"
STATE = "fake-state-0001"
VERIFIER = "v" * 64
CODE = "fake-code-0001"
ACCESS = "fake-access-token-0001"
REFRESH = "fake-refresh-token-0001"
NEW_ACCESS = "fake-access-token-0002"


def _token_response(access: str = ACCESS, refresh: str | None = REFRESH) -> httpx.Response:
    body: dict[str, object] = {
        "access_token": access,
        "token_type": "Bearer",
        "scope": " ".join(SCOPES),
        "expires_in": 3600,
    }
    if refresh is not None:
        body["refresh_token"] = refresh
    return httpx.Response(200, json=body)


@dataclass
class Spotify:
    """Answers the token endpoint and the device list; records every request."""

    token: Callable[[], httpx.Response | BaseException] = _token_response
    requests: list[httpx.Request] = field(default_factory=list)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if str(request.url) == TOKEN_URL:
            step = self.token()
            if isinstance(step, BaseException):
                raise step
            return step
        if str(request.url) == f"{API_BASE}/me/player/devices":
            devices = [
                {"id": "d1", "is_active": False, "name": "Desk", "type": "Computer"},
                {"id": "d2", "is_active": True, "name": "Phone", "type": "Smartphone"},
            ]
            return httpx.Response(200, json={"devices": devices})
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    def forms(self) -> list[dict[str, list[str]]]:
        return [
            parse_qs(r.content.decode("ascii")) for r in self.requests if str(r.url) == TOKEN_URL
        ]


@dataclass
class Result:
    code: int
    output: str
    spotify: Spotify
    token_path: Path


def _env(tmp_path: Path, **extra: str) -> dict[str, str]:
    env = {
        "MIST_SPOTIFY_CLIENT_ID": CLIENT_ID,
        "MIST_SPOTIFY_TOKEN_PATH": str(tmp_path / "secrets" / "spotify_token.json"),
    }
    env.update(extra)
    return env


def _run(
    tmp_path: Path,
    paste: str = "",
    *,
    argv: list[str] | None = None,
    env: dict[str, str] | None = None,
    spotify: Spotify | None = None,
    pkce: PkcePair | None = None,
) -> Result:
    spotify = spotify or Spotify()
    env = _env(tmp_path) if env is None else env
    stdout = io.StringIO()
    pair = pkce or PkcePair(verifier=VERIFIER, challenge=s256_challenge(VERIFIER))
    code = main(
        argv or [],
        stdin=io.StringIO(paste),
        stdout=stdout,
        env=env,
        transport=httpx.MockTransport(spotify),
        clock=FakeClock(),
        new_pkce=lambda: pair,
        new_state=lambda: STATE,
    )
    path = Path(env.get("MIST_SPOTIFY_TOKEN_PATH", tmp_path / "unset.json"))
    return Result(code, stdout.getvalue(), spotify, path)


def _paste(state: str = STATE, code: str = CODE) -> str:
    return f"{REDIRECT}?code={code}&state={state}\n"


def _authorize_query(output: str) -> dict[str, list[str]]:
    (line,) = (line for line in output.splitlines() if line.startswith(AUTHORIZE_URL))
    return parse_qs(urlsplit(line).query, strict_parsing=True)


def _save_token(path: Path, access: str = ACCESS) -> None:
    clock = FakeClock()
    SpotifyTokenStore(path).save(
        SpotifyToken(
            access_token=Secret(access),
            refresh_token=Secret(REFRESH),
            expires_at=clock.now() + timedelta(hours=1),
            scope=" ".join(SCOPES),
        )
    )


# -- authorize -------------------------------------------------------------------------------


def test_prints_an_authorize_url_with_every_field_and_the_three_scopes(tmp_path: Path) -> None:
    result = _run(tmp_path, _paste(), pkce=None)
    query = _authorize_query(result.output)
    assert query["client_id"] == [CLIENT_ID]
    assert query["response_type"] == ["code"]
    assert query["redirect_uri"] == [REDIRECT]
    assert query["code_challenge_method"] == ["S256"]
    assert query["code_challenge"] == [s256_challenge(VERIFIER)]
    assert query["state"] == [STATE]
    assert query["scope"][0].split() == [
        "user-read-playback-state",
        "user-modify-playback-state",
        "user-read-currently-playing",
    ]


def test_the_challenge_matches_the_verifier_sent_in_the_exchange(tmp_path: Path) -> None:
    # The real generator: the challenge printed must be the S256 of the verifier exchanged.
    from backend.capabilities.music.pkce import generate_pkce

    result = _run(tmp_path, _paste(), pkce=generate_pkce())
    (form,) = result.spotify.forms()
    challenge = _authorize_query(result.output)["code_challenge"][0]
    assert s256_challenge(form["code_verifier"][0]) == challenge


def test_the_redirect_uri_follows_the_setting(tmp_path: Path) -> None:
    redirect = "http://127.0.0.1:9999/cb"
    env = _env(tmp_path, MIST_SPOTIFY_REDIRECT_URI=redirect)
    result = _run(tmp_path, f"{redirect}?code={CODE}&state={STATE}\n", env=env)
    assert result.code == EXIT_OK
    assert _authorize_query(result.output)["redirect_uri"] == [redirect]
    assert result.spotify.forms()[0]["redirect_uri"] == [redirect]


def test_exchanges_the_code_with_the_verifier_writes_the_token_file_and_exits_0(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path, _paste())
    assert result.code == EXIT_OK
    (form,) = result.spotify.forms()
    assert form == {
        "grant_type": ["authorization_code"],
        "code": [CODE],
        "redirect_uri": [REDIRECT],
        "client_id": [CLIENT_ID],
        "code_verifier": [VERIFIER],
    }
    saved = json.loads(result.token_path.read_text(encoding="utf-8"))
    assert (saved["access_token"], saved["refresh_token"]) == (ACCESS, REFRESH)
    assert str(result.token_path) in result.output
    # Token values never reach the terminal.
    assert ACCESS not in result.output and REFRESH not in result.output


def test_a_bare_query_string_paste_is_accepted(tmp_path: Path) -> None:
    result = _run(tmp_path, f"code={CODE}&state={STATE}\n")
    assert result.code == EXIT_OK


@pytest.mark.parametrize(
    "paste",
    [_paste(state="another-state"), f"{REDIRECT}?code={CODE}\n"],
    ids=["mismatched-state", "missing-state"],
)
def test_a_state_mismatch_exits_2_without_an_exchange(tmp_path: Path, paste: str) -> None:
    result = _run(tmp_path, paste)
    assert result.code == EXIT_BAD_PASTE == 2
    assert "state" in result.output
    assert result.spotify.requests == []
    assert not result.token_path.exists()


@pytest.mark.parametrize(
    "paste",
    [
        "",
        "\n",
        "not a url at all\n",
        f"{REDIRECT}?state={STATE}\n",
        f"{REDIRECT}?error=access_denied&state={STATE}\n",
    ],
    ids=["eof", "blank", "garbage", "no-code", "denied"],
)
def test_a_malformed_paste_exits_2(tmp_path: Path, paste: str) -> None:
    result = _run(tmp_path, paste)
    assert result.code == EXIT_BAD_PASTE
    assert result.spotify.requests == []
    assert not result.token_path.exists()


@pytest.mark.parametrize(
    "step",
    [
        lambda: httpx.Response(400, json={"error": "invalid_grant"}),
        lambda: httpx.Response(500),
        lambda: httpx.Response(429, headers={"Retry-After": "5"}),
        lambda: httpx.ConnectError("synthetic"),
    ],
    ids=["refused", "server-error", "rate-limited", "unreachable"],
)
def test_a_token_exchange_http_error_exits_3(
    tmp_path: Path, step: Callable[[], httpx.Response | BaseException]
) -> None:
    result = _run(tmp_path, _paste(), spotify=Spotify(token=step))
    assert result.code == EXIT_EXCHANGE_FAILED == 3
    assert not result.token_path.exists()


def test_an_unset_client_id_exits_4_before_printing_a_url(tmp_path: Path) -> None:
    env = _env(tmp_path)
    del env["MIST_SPOTIFY_CLIENT_ID"]
    result = _run(tmp_path, _paste(), env=env)
    assert result.code == EXIT_NO_CLIENT_ID == 4
    assert "MIST_SPOTIFY_CLIENT_ID" in result.output
    assert AUTHORIZE_URL not in result.output
    assert result.spotify.requests == []


def test_an_unwritable_token_file_exits_1(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where the directory should be", encoding="utf-8")
    env = _env(tmp_path, MIST_SPOTIFY_TOKEN_PATH=str(blocker / "spotify_token.json"))
    result = _run(tmp_path, _paste(), env=env)
    assert result.code == EXIT_FAILED == 1
    assert ACCESS not in result.output


# -- --check ---------------------------------------------------------------------------------


def test_check_refreshes_the_token_and_prints_the_device_count(tmp_path: Path) -> None:
    env = _env(tmp_path)
    _save_token(Path(env["MIST_SPOTIFY_TOKEN_PATH"]))
    spotify = Spotify(token=lambda: _token_response(access=NEW_ACCESS, refresh=None))
    result = _run(tmp_path, argv=["--check"], env=env, spotify=spotify)
    assert result.code == EXIT_OK
    assert "2 devices" in result.output
    (form,) = spotify.forms()
    assert form["grant_type"] == ["refresh_token"]
    assert form["refresh_token"] == [REFRESH]
    saved = json.loads(result.token_path.read_text(encoding="utf-8"))
    assert (saved["access_token"], saved["refresh_token"]) == (NEW_ACCESS, REFRESH)
    devices = [r for r in spotify.requests if str(r.url).startswith(API_BASE)]
    assert [r.headers["Authorization"] for r in devices] == [f"Bearer {NEW_ACCESS}"]
    assert NEW_ACCESS not in result.output and REFRESH not in result.output


def test_check_prints_no_prompt_and_reads_no_stdin(tmp_path: Path) -> None:
    env = _env(tmp_path)
    _save_token(Path(env["MIST_SPOTIFY_TOKEN_PATH"]))
    result = _run(tmp_path, "unread\n", argv=["--check"], env=env)
    assert result.code == EXIT_OK
    assert AUTHORIZE_URL not in result.output


def test_check_with_a_refused_refresh_exits_3(tmp_path: Path) -> None:
    env = _env(tmp_path)
    _save_token(Path(env["MIST_SPOTIFY_TOKEN_PATH"]))
    spotify = Spotify(token=lambda: httpx.Response(400, json={"error": "invalid_grant"}))
    result = _run(tmp_path, argv=["--check"], env=env, spotify=spotify)
    assert result.code == EXIT_EXCHANGE_FAILED
    assert "invalid_grant" in result.output


def test_check_without_a_token_file_exits_1_naming_the_cli(tmp_path: Path) -> None:
    result = _run(tmp_path, argv=["--check"])
    assert result.code == EXIT_FAILED
    assert "backend.capabilities.music.authorize" in result.output
    assert result.spotify.requests == []


def test_check_with_an_unset_client_id_exits_4(tmp_path: Path) -> None:
    env = _env(tmp_path)
    del env["MIST_SPOTIFY_CLIENT_ID"]
    result = _run(tmp_path, argv=["--check"], env=env)
    assert result.code == EXIT_NO_CLIENT_ID
