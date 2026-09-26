"""check_decision_rules_clean(): tracked+clean, untracked, and dirty, against
a real temporary git repo (subprocess git, no network needed for
init/add/commit).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from scripts.model_bench.bench_host import DecisionRulesError, check_decision_rules_clean  # noqa: E402


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    # See bench_host.py's _run_git: the worker container exports GIT_DIR /
    # GIT_WORK_TREE pinned to this worktree, which would otherwise hijack
    # `git init` (and everything after it) in a freshly created temp repo.
    env = {k: v for k, v in os.environ.items() if k not in ("GIT_DIR", "GIT_WORK_TREE")}
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, shell=False, env=env
    )


@pytest.fixture
def temp_git_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(["init", "-q"], cwd=repo)
    _git(["config", "user.email", "test@example.invalid"], cwd=repo)
    _git(["config", "user.name", "Test"], cwd=repo)
    return repo


def test_missing_file_raises(temp_git_repo):
    with pytest.raises(DecisionRulesError):
        check_decision_rules_clean(temp_git_repo / "decision_rules.json", temp_git_repo)


def test_untracked_file_raises(temp_git_repo):
    path = temp_git_repo / "decision_rules.json"
    path.write_text('{"rules": []}\n', encoding="utf-8")
    with pytest.raises(DecisionRulesError):
        check_decision_rules_clean(path, temp_git_repo)


def test_tracked_and_clean_returns_sha256(temp_git_repo):
    path = temp_git_repo / "decision_rules.json"
    path.write_text('{"rules": []}\n', encoding="utf-8")
    _git(["add", "decision_rules.json"], cwd=temp_git_repo)
    _git(["commit", "-q", "-m", "add decision rules"], cwd=temp_git_repo)

    import hashlib

    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    digest = check_decision_rules_clean(path, temp_git_repo)
    assert digest == expected


def test_dirty_tracked_file_raises(temp_git_repo):
    path = temp_git_repo / "decision_rules.json"
    path.write_text('{"rules": []}\n', encoding="utf-8")
    _git(["add", "decision_rules.json"], cwd=temp_git_repo)
    _git(["commit", "-q", "-m", "add decision rules"], cwd=temp_git_repo)

    path.write_text('{"rules": ["changed"]}\n', encoding="utf-8")
    with pytest.raises(DecisionRulesError):
        check_decision_rules_clean(path, temp_git_repo)


def test_staged_but_uncommitted_change_raises(temp_git_repo):
    path = temp_git_repo / "decision_rules.json"
    path.write_text('{"rules": []}\n', encoding="utf-8")
    _git(["add", "decision_rules.json"], cwd=temp_git_repo)
    _git(["commit", "-q", "-m", "add decision rules"], cwd=temp_git_repo)

    path.write_text('{"rules": ["changed"]}\n', encoding="utf-8")
    _git(["add", "decision_rules.json"], cwd=temp_git_repo)
    with pytest.raises(DecisionRulesError):
        check_decision_rules_clean(path, temp_git_repo)
