"""Guard: the capability layer's dependency pins in `requirements.txt` and `pyproject.toml`.

- `mcp` is pinned for the Home Assistant MCP SDK transport that a later task of the
  capabilities goal (T16) adds at `backend/capabilities/mcp/sdk_transport.py`.
- `tzdata` supplies the IANA zone data that the stdlib `zoneinfo` reads for alarms. The
  backend image has no system zone database, so without it `zoneinfo.ZoneInfo` raises
  `ZoneInfoNotFoundError` for every zone name.

It also pins `pyarrow`, which `lancedb` imports. pyarrow 26 needs NumPy 2 at import, but
`chatterbox-tts` holds numpy below 2.0, so a fresh install ends at numpy 1.26.x and an
unbounded `pyarrow` range resolves to a release whose import fails. The pin stays at the
last release that imports under numpy 1.26 until `chatterbox-tts` allows numpy 2.

`requirements.txt` carries the exact pins the image installs; `pyproject.toml` carries the
matching ranges (`mcp>=2.3`, `tzdata>=2026.5`, `pyarrow>=17.0,<26`), and both are checked.

Static on purpose: the test parses both files and never imports `mcp`, `tzdata` or
`pyarrow`, so it passes in an image that does not have them installed yet. Specifiers are
compared as strings rather than through `packaging`, which is not a declared dependency.
"""

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"

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


PYARROW_PIN = "25.0.1"


class TestPyarrowIsPinnedBelowNumpy2:
    def test_pyarrow_is_pinned_exactly(self) -> None:
        pins = _pinned_requirements()

        assert "pyarrow" in pins, f"requirements.txt must pin `pyarrow=={PYARROW_PIN}` exactly"
        assert pins["pyarrow"] == PYARROW_PIN, (
            f"pyarrow is pinned to {pins['pyarrow']!r}; {PYARROW_PIN} is the last release "
            "that imports under the numpy<2 that chatterbox-tts requires"
        )


# `[project] dependencies` entry -> its comma-separated version clauses, whitespace removed.
PYPROJECT_RANGES = {
    "mcp": {">=2.3"},
    "tzdata": {">=2026.5"},
    "pyarrow": {">=17.0", "<26"},
}

# `name[extras] specifiers ; marker`; extras and marker are not needed for these entries.
PYPROJECT_ENTRY = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*(?P<specs>[^;]*)"
)


def _pyproject_dependencies() -> dict[str, set[str]]:
    """Map lowercased distribution name -> set of version clauses in `[project] dependencies`."""
    with PYPROJECT_PATH.open("rb") as handle:
        entries = tomllib.load(handle)["project"]["dependencies"]
    ranges: dict[str, set[str]] = {}
    for entry in entries:
        match = PYPROJECT_ENTRY.match(entry.strip())
        assert match is not None, f"cannot parse pyproject dependency {entry!r}"
        clauses = {
            re.sub(r"\s+", "", clause)
            for clause in match.group("specs").split(",")
            if clause.strip()
        }
        ranges[match.group("name").lower()] = clauses
    return ranges


class TestPyprojectRangesMatchPins:
    def test_pyproject_declares_capability_and_pyarrow_ranges(self) -> None:
        ranges = _pyproject_dependencies()

        for name, expected in PYPROJECT_RANGES.items():
            assert name in ranges, f"pyproject.toml [project] dependencies must list {name}"
            assert ranges[name] == expected, (
                f"pyproject.toml declares {name} with {sorted(ranges[name])}; "
                f"expected {sorted(expected)}"
            )
