"""decision_rules.json plan v2 (2026-09-25): the pre-registered v1 `rules` and
`constants` sections must stay byte-for-byte unchanged in content after the
`exploratory_rules` addition -- only new top-level keys (`supersedes`,
`exploratory_rules`) may be added; `rules`/`thresholds`/`constants`/`metrics`/
`statistics` are never touched by this change.

Loads the v1 file via `git show <v1 commit>:scripts/model_bench/decision_rules.json`
(subprocess git against this repo's own history -- read-only, no network needed, per
the worker container's git-metadata access) rather than a hand-copied fixture, so this
test compares against the actual committed v1 file, not a transcription of it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from scripts.model_bench import analyse  # noqa: E402

# The commit that landed the v1 (pre-registered, 2026-09-24) decision_rules.json --
# agent/mist-model-bench/integration's HEAD when this task's worktree was created.
V1_COMMIT = "dced3d69831f5dcacb438c4024537c997056d885"
V1_SHA256 = "f6a42ba8d36084fd5c893ac430294493cd4d1f7cb8da2d40a8f29add49aa3fae"


def _load_v1_decision_rules() -> dict:
    proc = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "show", f"{V1_COMMIT}:scripts/model_bench/decision_rules.json"],
        capture_output=True,
        text=True,
        shell=False,
    )
    if proc.returncode != 0:
        pytest.skip(
            f"cannot read v1 decision_rules.json via `git show {V1_COMMIT}:...` "
            f"(exit {proc.returncode}): {proc.stderr.strip()}"
        )
    return json.loads(proc.stdout)


def test_v1_file_sha256_matches_the_supersedes_entry():
    import hashlib

    proc = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "cat-file", "-p", f"{V1_COMMIT}:scripts/model_bench/decision_rules.json"],
        capture_output=True,
        text=True,
        shell=False,
    )
    if proc.returncode != 0:
        pytest.skip(f"cannot read v1 decision_rules.json via git cat-file: {proc.stderr.strip()}")
    digest = hashlib.sha256(proc.stdout.encode("utf-8")).hexdigest()
    assert digest == V1_SHA256


def test_v1_rules_and_constants_unchanged_in_content():
    v1 = _load_v1_decision_rules()
    current = analyse.load_decision_rules()
    assert current["rules"] == v1["rules"]
    assert current["constants"] == v1["constants"]
    assert current["thresholds"] == v1["thresholds"]
    assert current["metrics"] == v1["metrics"]
    assert current["statistics"] == v1["statistics"]


def test_v1_sha_is_listed_in_current_supersedes():
    current = analyse.load_decision_rules()
    listed = {e["sha256"] for e in current.get("supersedes", [])}
    assert V1_SHA256 in listed


def test_exploratory_rules_are_all_marked_not_pre_registered():
    current = analyse.load_decision_rules()
    exploratory = current.get("exploratory_rules")
    assert exploratory is not None
    rules = exploratory["rules"]
    assert len(rules) >= 3
    for rule in rules:
        assert rule["pre_registered"] is False
        assert rule["basis"]
