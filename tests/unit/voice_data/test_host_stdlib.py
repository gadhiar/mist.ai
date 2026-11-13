"""Host-side purity: host modules of `scripts/voice_data` import only stdlib and `scripts.`.

The host runs these modules with whatever Python Raj has, outside any container, so a third-party
import there would fail on the host while every container test passed. The scan covers every
module present now or added later; only the container-side folders below are exempt (Test
strategy, Host-side purity).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[3] / "scripts" / "voice_data"
EXEMPT_DIRS = ("stages", "bindings", "teachers", "students", "convert", "bench/probes")
EXEMPT_FILES = ("rehearsal/mix.py",)


def is_host_module(rel: str) -> bool:
    """Return whether a package-relative posix path is a host-side module."""
    if rel in EXEMPT_FILES:
        return False
    return not any(rel == d or rel.startswith(d + "/") for d in EXEMPT_DIRS)


def forbidden_imports(source: str) -> list[str]:
    """Return the absolute top-level imports that are neither stdlib nor `scripts`."""
    allowed = set(sys.stdlib_module_names) | {"scripts"}
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        else:
            continue
        found.extend(name for name in names if name.split(".")[0] not in allowed)
    return found


def host_modules() -> list[Path]:
    """Return every host-side module under the package."""
    return sorted(
        path
        for path in PACKAGE.rglob("*.py")
        if is_host_module(path.relative_to(PACKAGE).as_posix())
    )


def test_scan_sees_the_foundation_modules() -> None:
    rels = {path.relative_to(PACKAGE).as_posix() for path in host_modules()}
    assert {"cli.py", "root.py", "records.py", "gates.py", "commands/doctor.py"} <= rels


@pytest.mark.parametrize("path", host_modules(), ids=lambda p: p.relative_to(PACKAGE).as_posix())
def test_host_module_imports_only_stdlib_and_scripts(path: Path) -> None:
    assert forbidden_imports(path.read_text(encoding="utf-8")) == []


def test_checker_flags_third_party_imports() -> None:
    source = "import numpy as np\nfrom soundfile import read\nimport os\nfrom . import root\n"
    assert forbidden_imports(source) == ["numpy", "soundfile"]


def test_checker_allows_scripts_imports() -> None:
    assert forbidden_imports("from scripts.voice_data import root\nimport json\n") == []


@pytest.mark.parametrize(
    ("rel", "host"),
    [
        ("cli.py", True),
        ("rehearsal/build.py", True),
        ("bench/report.py", True),
        ("stages/extract.py", False),
        ("bindings/x.py", False),
        ("teachers/a.py", False),
        ("students/b.py", False),
        ("convert/c.py", False),
        ("rehearsal/mix.py", False),
        ("bench/probes/p.py", False),
    ],
)
def test_exemptions_match_the_test_strategy(rel: str, host: bool) -> None:
    assert is_host_module(rel) is host
