"""Guard: the backend build context carries `docker/backend/models.lock` and nothing else new.

The backend Dockerfile copies `docker/backend/models.lock` into the model layer (plan T12), but
`.dockerignore` excludes the whole `docker/` tree. The lock comes back by the same chained
re-include `docker/extraction/requirements.txt` uses: un-exclude the directory, re-exclude
everything directly inside it, re-include the one file. These tests parse `.dockerignore` with
Docker's matching rules and check that the lock is in the context while every other path under
`docker/backend/` (the Dockerfile, a host `.env`, a nested directory) stays out.

Hermetic: parses the file only; no docker build.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERIGNORE_PATH = REPO_ROOT / ".dockerignore"
LOCK = "docker/backend/models.lock"
EXTRACTION_REQUIREMENTS = "docker/extraction/requirements.txt"
CHAIN = ["!docker/backend/", "docker/backend/*", "!docker/backend/models.lock"]


def _translate(pattern: str) -> re.Pattern[str]:
    """Docker's pattern as a regex: `*` and `?` stop at `/`, `**` crosses it."""
    out, index = "", 0
    while index < len(pattern):
        char = pattern[index]
        if pattern.startswith("**/", index):
            out += "(?:.*/)?"
            index += 3
            continue
        if pattern.startswith("**", index):
            out += ".*"
            index += 2
            continue
        if char == "*":
            out += "[^/]*"
        elif char == "?":
            out += "[^/]"
        elif char == "[":
            end = pattern.index("]", index)
            out += pattern[index : end + 1]
            index = end
        else:
            out += re.escape(char)
        index += 1
    return re.compile(rf"^{out}$")


def _patterns() -> list[tuple[bool, str, re.Pattern[str]]]:
    """(exception, cleaned pattern, regex) for each `.dockerignore` line, in order.

    Docker trims each line, skips blanks and `#` comments, treats a leading `!` as an
    exception, and cleans the pattern the way Go's `filepath.Clean` does, so a trailing `/`
    is dropped and a leading `/` is ignored.
    """
    patterns = []
    for raw in DOCKERIGNORE_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        exception = line.startswith("!")
        pattern = line[1:].strip() if exception else line
        pattern = os.path.normpath(pattern).replace("\\", "/").lstrip("/")
        patterns.append((exception, pattern, _translate(pattern)))
    return patterns


def _excluded(path: str) -> bool:
    """moby/patternmatcher's `MatchesOrParentMatches`: the last pattern matching the path or
    one of its parent directories decides, and an exception pattern un-excludes.
    """
    parts = path.split("/")
    candidates = ["/".join(parts[: i + 1]) for i in range(len(parts))]
    excluded = False
    for exception, _pattern, regex in _patterns():
        if any(regex.match(candidate) for candidate in candidates):
            excluded = not exception
    return excluded


def _files_under(directory: str) -> list[str]:
    root = REPO_ROOT / directory
    return sorted(
        path.relative_to(REPO_ROOT).as_posix() for path in root.rglob("*") if path.is_file()
    )


class TestMatcher:
    """The matcher reproduces the existing rules before it is trusted with the new ones."""

    @pytest.mark.parametrize(
        "path",
        ["docker/backend/Dockerfile", "docker/extraction/Dockerfile", ".git/HEAD", "a.wav"],
    )
    def test_excluded(self, path: str) -> None:
        assert _excluded(path)

    @pytest.mark.parametrize(
        "path", [EXTRACTION_REQUIREMENTS, "requirements.txt", "backend/server.py", "x/a.wav"]
    )
    def test_included(self, path: str) -> None:
        assert not _excluded(path)


class TestModelsLockContext:
    def test_lock_is_in_the_build_context(self) -> None:
        assert not _excluded(LOCK)

    def test_checker_script_is_in_the_build_context(self) -> None:
        assert not _excluded("scripts/check_models_lock.py")

    @pytest.mark.parametrize(
        "path",
        [
            "docker/backend/Dockerfile",
            "docker/backend/.env",
            "docker/backend/models.lock.bak",
            "docker/backend/README.md",
            "docker/backend/sub/models.lock",
            "docker/backend/sub/file.txt",
        ],
    )
    def test_every_other_path_under_docker_backend_stays_out(self, path: str) -> None:
        assert _excluded(path)

    def test_the_repository_docker_tree_contributes_only_the_two_files(self) -> None:
        """Of every file under `docker/` in the checkout, only the two re-included ones ship."""
        files = _files_under("docker")
        assert LOCK in files and "docker/backend/Dockerfile" in files

        included = {path for path in files if not _excluded(path)}

        assert included == {EXTRACTION_REQUIREMENTS, LOCK}

    def test_the_chain_is_the_only_rule_naming_docker_backend(self) -> None:
        """The three chained lines, in order and after `docker/`, and no other rule, so the
        re-include cannot widen beyond the lock and nothing outside `docker/backend/` moves.
        """
        lines = [
            line.strip()
            for line in DOCKERIGNORE_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        naming = [line for line in lines if "docker/backend" in line]

        assert naming == CHAIN
        assert lines.index("docker/") < lines.index(CHAIN[0])
