"""Guard: root `requirements.txt` pins the capability layer's dependencies exactly.

- `mcp` is the Home Assistant MCP client that `backend/capabilities/mcp/sdk_transport.py`
  builds on.
- `tzdata` supplies the IANA zone data that the stdlib `zoneinfo` reads for alarms. The
  backend image has no system zone database, so without it `zoneinfo.ZoneInfo` raises
  `ZoneInfoNotFoundError` for every zone name.

Static on purpose: the test parses `requirements.txt` and never imports `mcp` or `tzdata`,
so it passes in an image that does not have them installed yet.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"

# `name[extra1,extra2]==version`, optionally followed by a marker or an inline comment.
PINNED_LINE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)"
    r"(?:\[(?P<extras>[^\]]*)\])?"
    r"==(?P<version>[^\s#;]+)"
)

CAPABILITY_PINS = {
    "mcp": "2.3.0",
    "tzdata": "2026.5",
}


def _pinned_requirements() -> dict[str, str]:
    """Map lowercased distribution name -> exact pinned version."""
    pins: dict[str, str] = {}
    for line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = PINNED_LINE.match(stripped)
        if match is None:
            continue
        pins[match.group("name").lower()] = match.group("version")
    return pins


class TestCapabilityDependenciesArePinned:
    def test_mcp_is_pinned_exactly(self) -> None:
        pins = _pinned_requirements()

        assert "mcp" in pins, "requirements.txt must pin `mcp==2.3.0` exactly"
        assert (
            pins["mcp"] == CAPABILITY_PINS["mcp"]
        ), f"mcp is pinned to {pins['mcp']!r}; the capability layer expects 2.3.0"

    def test_tzdata_is_pinned_exactly(self) -> None:
        pins = _pinned_requirements()

        assert "tzdata" in pins, "requirements.txt must pin `tzdata==2026.5` exactly"
        assert (
            pins["tzdata"] == CAPABILITY_PINS["tzdata"]
        ), f"tzdata is pinned to {pins['tzdata']!r}; the capability layer expects 2026.5"
