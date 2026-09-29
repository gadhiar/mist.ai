"""Guard: the backend image must be able to serve WebSocket connections (MIS-176).

The backend image built on 2026-09-27 had no WebSocket library. uvicorn logged
`No supported WebSocket library detected` and answered `GET /ws` with 404, which
locked out the frontend, voice and the extraction E2E test. Root
`requirements.txt` had pinned plain `uvicorn==0.46.0`; in uvicorn 0.46.0
`websockets` is pulled in only by the `standard` extra (`wsproto`, the other
library uvicorn can use, is in no extra).

Two independent checks, because neither covers the other:

- Runtime: uvicorn resolves a WebSocket protocol class in the interpreter running
  the tests. This fails in an environment that lacks both `websockets` and
  `wsproto`, but passes anywhere one of them happens to be installed, so it cannot
  by itself catch a regression in `requirements.txt`.
- Static: root `requirements.txt` declares `uvicorn[standard]==...` and an exactly
  pinned `websockets==...`. This catches the regression regardless of what is
  installed in the environment running the tests.

Hermetic: no network, no sockets, no Neo4j, no LLM. `uvicorn.Config.load()`
resolves the HTTP, WebSocket and lifespan classes and wraps the app; it does not
bind a port.
"""

import re
from pathlib import Path

import uvicorn

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"

# `name[extra1,extra2]==version`, optionally followed by an inline comment.
PINNED_LINE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)"
    r"(?:\[(?P<extras>[^\]]*)\])?"
    r"==(?P<version>[^\s#;]+)"
)


async def _asgi_app(scope, receive, send) -> None:
    """Minimal ASGI3 callable handed to `uvicorn.Config`; its body never runs."""


def _pinned_requirements() -> dict[str, tuple[set[str], str]]:
    """Map lowercased distribution name -> (lowercased extras, exact version)."""
    pins: dict[str, tuple[set[str], str]] = {}
    for line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = PINNED_LINE.match(stripped)
        if match is None:
            continue
        extras = {
            extra.strip().lower()
            for extra in (match.group("extras") or "").split(",")
            if extra.strip()
        }
        pins[match.group("name").lower()] = (extras, match.group("version"))
    return pins


class TestUvicornResolvesWebSocketProtocol:
    def test_auto_websocket_protocol_is_available(self) -> None:
        from uvicorn.protocols.websockets.auto import AutoWebSocketsProtocol

        assert AutoWebSocketsProtocol is not None, (
            "uvicorn found neither `websockets` nor `wsproto`; GET /ws would 404. "
            "Install uvicorn[standard] and a pinned websockets."
        )

    def test_config_load_resolves_ws_protocol_class(self) -> None:
        config = uvicorn.Config(_asgi_app, ws="auto", log_config=None)
        config.load()

        assert config.ws_protocol_class is not None, (
            "uvicorn.Config(ws='auto').load() left ws_protocol_class as None; "
            "WebSocket upgrades would be refused"
        )


class TestRootRequirementsPinWebSocketSupport:
    def test_uvicorn_is_pinned_with_standard_extra(self) -> None:
        pins = _pinned_requirements()

        assert "uvicorn" in pins, "requirements.txt must pin uvicorn exactly (uvicorn==...)"
        extras, _version = pins["uvicorn"]
        assert "standard" in extras, (
            "requirements.txt must declare uvicorn[standard]==...; plain uvicorn installs "
            "no WebSocket library and GET /ws returns 404"
        )

    def test_websockets_is_pinned_exactly(self) -> None:
        pins = _pinned_requirements()

        assert "websockets" in pins, (
            "requirements.txt must carry an explicit `websockets==...` line next to "
            "uvicorn[standard]"
        )
        _extras, version = pins["websockets"]
        assert re.fullmatch(r"\d+(\.\d+)*", version), f"websockets pin {version!r} is not exact"


# Members of uvicorn's `standard` extra that have open version ranges upstream.
# Pinned to the versions read from the live backend image (MIS-177, i112) so a
# rebuild cannot silently change the server stack. Static on purpose: these
# checks parse the files and never compare against installed package versions.
STANDARD_EXTRA_PINS = {
    "httptools": "0.8.0",
    "watchfiles": "1.3.0",
    "uvloop": "0.22.1",
}
CHATTERBOX_PIN = "0.1.7"
DOCKERFILE_PATH = REPO_ROOT / "docker" / "backend" / "Dockerfile"


def _requirement_lines() -> list[str]:
    """Non-comment, non-blank requirement lines with inline comments removed."""
    lines = []
    for line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.split(" #", 1)[0].strip()
        if stripped and not stripped.startswith("#"):
            lines.append(stripped)
    return lines


class TestStandardExtraMembersArePinned:
    def test_standard_extra_members_are_pinned_to_live_versions(self) -> None:
        pins = _pinned_requirements()

        for name, expected in STANDARD_EXTRA_PINS.items():
            assert name in pins, f"requirements.txt must pin `{name}=={expected}` exactly"
            _extras, version = pins[name]
            assert (
                version == expected
            ), f"{name} is pinned to {version!r}; the live backend image runs {expected}"

    def test_uvloop_pin_carries_non_windows_marker(self) -> None:
        uvloop_lines = [line for line in _requirement_lines() if line.lower().startswith("uvloop")]

        assert len(uvloop_lines) == 1, "requirements.txt must contain exactly one uvloop line"
        marker = re.sub(r"\s+", "", uvloop_lines[0].partition(";")[2])
        assert marker == 'sys_platform!="win32"', (
            "uvloop has no Windows wheels; its pin must end with "
            f'`; sys_platform != "win32"`, got {uvloop_lines[0]!r}'
        )


class TestBackendDockerfilePinsChatterbox:
    def test_chatterbox_tts_is_pinned_in_dockerfile(self) -> None:
        text = DOCKERFILE_PATH.read_text(encoding="utf-8")

        install_lines = [
            line for line in text.splitlines() if re.search(r"pip install\b.*chatterbox-tts", line)
        ]
        assert install_lines, "docker/backend/Dockerfile must install chatterbox-tts"
        for line in install_lines:
            assert (
                f"chatterbox-tts=={CHATTERBOX_PIN}" in line
            ), f"chatterbox-tts must be pinned to =={CHATTERBOX_PIN} in the Dockerfile: {line!r}"

    def test_stale_backend_requirements_file_is_gone(self) -> None:
        assert not (REPO_ROOT / "backend" / "requirements.txt").exists(), (
            "backend/requirements.txt was deleted (MIS-177, i111): the image installs the "
            "root requirements.txt"
        )
