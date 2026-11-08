"""Tests for the Spotify token file: atomic writes, its mode, and refusing a bad file."""

from __future__ import annotations

import json
import os
import stat
from datetime import UTC, datetime
from pathlib import Path

import pytest

from backend.capabilities.music import token_store
from backend.capabilities.music.token_store import (
    AUTHORIZE_COMMAND,
    SpotifyToken,
    SpotifyTokenStore,
)
from backend.capabilities.secrets import Secret
from backend.errors import SpotifyAuthError, SpotifyError

EXPIRES = datetime(2026, 11, 7, 12, 0, tzinfo=UTC)
ACCESS = "fake-access-token-0001"
REFRESH = "fake-refresh-token-0001"


def _token(access: str = ACCESS, refresh: str = REFRESH) -> SpotifyToken:
    return SpotifyToken(
        access_token=Secret(access),
        refresh_token=Secret(refresh),
        expires_at=EXPIRES,
        scope="user-read-playback-state",
    )


def _nested_path(tmp_path: Path) -> Path:
    return tmp_path / "data" / "secrets" / "spotify_token.json"


def test_save_creates_the_missing_parent_and_round_trips(tmp_path: Path) -> None:
    path = _nested_path(tmp_path)
    assert not path.parent.exists()
    store = SpotifyTokenStore(path)
    store.save(_token())
    loaded = store.load()
    assert loaded.access_token.reveal() == ACCESS
    assert loaded.refresh_token.reveal() == REFRESH
    assert loaded.expires_at == EXPIRES
    assert loaded.scope == "user-read-playback-state"
    assert loaded.token_type == "Bearer"


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes are not enforced on Windows")
def test_saved_file_has_mode_0600(tmp_path: Path) -> None:
    path = _nested_path(tmp_path)
    SpotifyTokenStore(path).save(_token())
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes are not enforced on Windows")
def test_overwrite_resets_a_wider_mode_to_0600(tmp_path: Path) -> None:
    path = _nested_path(tmp_path)
    store = SpotifyTokenStore(path)
    store.save(_token())
    path.chmod(0o644)
    store.save(_token(access="fake-access-token-0002"))
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_save_writes_a_temp_file_in_the_same_directory_then_replaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _nested_path(tmp_path)
    calls: list[tuple[Path, Path, str]] = []
    real_replace = os.replace

    def recording_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        source = Path(src)
        calls.append((source, Path(dst), source.read_text(encoding="utf-8")))
        real_replace(src, dst)

    monkeypatch.setattr(token_store.os, "replace", recording_replace)
    SpotifyTokenStore(path).save(_token())

    assert len(calls) == 1
    source, destination, content = calls[0]
    assert destination == path
    assert source.parent == path.parent
    assert source != path
    assert json.loads(content)["access_token"] == ACCESS
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]


def test_failed_replace_keeps_the_old_file_and_removes_the_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _nested_path(tmp_path)
    store = SpotifyTokenStore(path)
    store.save(_token())
    before = path.read_text(encoding="utf-8")

    def failing_replace(src: object, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(token_store.os, "replace", failing_replace)
    with pytest.raises(SpotifyError, match="cannot be written") as excinfo:
        store.save(_token(access="fake-access-token-0002"))
    assert not isinstance(excinfo.value, SpotifyAuthError)
    assert path.read_text(encoding="utf-8") == before
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]


def test_saved_file_never_holds_the_masked_form(tmp_path: Path) -> None:
    path = _nested_path(tmp_path)
    SpotifyTokenStore(path).save(_token())
    content = path.read_text(encoding="utf-8")
    assert "***" not in content
    assert REFRESH in content


def test_missing_file_raises_auth_error_naming_the_authorize_cli(tmp_path: Path) -> None:
    store = SpotifyTokenStore(_nested_path(tmp_path))
    with pytest.raises(SpotifyAuthError) as excinfo:
        store.load()
    message = str(excinfo.value)
    assert message.startswith("not_authorized")
    assert AUTHORIZE_COMMAND in message
    assert AUTHORIZE_COMMAND == "python -m backend.capabilities.music.authorize"


@pytest.mark.parametrize(
    "content",
    [
        "not json at all",
        "[]",
        "{}",
        json.dumps({"access_token": ACCESS, "refresh_token": REFRESH}),
        json.dumps(
            {
                "access_token": ACCESS,
                "refresh_token": REFRESH,
                "expires_at": "yesterday",
                "scope": "",
                "token_type": "Bearer",
            }
        ),
        json.dumps(
            {
                "access_token": ACCESS,
                "refresh_token": REFRESH,
                "expires_at": "2026-11-07T12:00:00",
                "scope": "",
                "token_type": "Bearer",
            }
        ),
        json.dumps(
            {
                "access_token": "",
                "refresh_token": REFRESH,
                "expires_at": EXPIRES.isoformat(),
                "scope": "",
                "token_type": "Bearer",
            }
        ),
        json.dumps(
            {
                "access_token": ACCESS,
                "refresh_token": 7,
                "expires_at": EXPIRES.isoformat(),
                "scope": "",
                "token_type": "Bearer",
            }
        ),
    ],
    ids=[
        "not-json",
        "list",
        "empty-object",
        "no-expiry",
        "bad-expiry",
        "naive-expiry",
        "empty-access",
        "non-string-refresh",
    ],
)
def test_corrupt_file_raises_auth_error_naming_the_authorize_cli(
    tmp_path: Path, content: str
) -> None:
    path = _nested_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(content, encoding="utf-8")
    with pytest.raises(SpotifyAuthError) as excinfo:
        SpotifyTokenStore(path).load()
    message = str(excinfo.value)
    assert message.startswith("not_authorized")
    assert AUTHORIZE_COMMAND in message
    assert ACCESS not in message
    assert REFRESH not in message


def test_token_repr_masks_both_tokens() -> None:
    text = repr(_token())
    assert ACCESS not in text
    assert REFRESH not in text
