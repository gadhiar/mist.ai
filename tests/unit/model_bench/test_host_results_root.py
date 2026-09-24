"""--results-root must resolve outside the git work tree containing bench_host.py."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from scripts.model_bench.bench_host import ResultsRootError, check_results_root_outside_repo  # noqa: E402


def test_results_root_inside_repo_refused():
    with pytest.raises(ResultsRootError):
        check_results_root_outside_repo(_REPO_ROOT / "results", repo_root=_REPO_ROOT)


def test_results_root_equal_to_repo_root_refused():
    with pytest.raises(ResultsRootError):
        check_results_root_outside_repo(_REPO_ROOT, repo_root=_REPO_ROOT)


def test_results_root_deeply_nested_inside_repo_refused():
    with pytest.raises(ResultsRootError):
        check_results_root_outside_repo(_REPO_ROOT / "a" / "b" / "c", repo_root=_REPO_ROOT)


def test_results_root_outside_repo_accepted(tmp_path):
    outside = tmp_path / "mist-model-bench-results"
    check_results_root_outside_repo(outside, repo_root=_REPO_ROOT)


def test_results_root_sibling_directory_with_repo_name_prefix_not_confused(tmp_path):
    # A sibling directory whose name merely starts with the repo dir's name
    # (e.g. repo at .../mist.ai, candidate at .../mist.ai-results) must not
    # be treated as "inside" by a naive string-prefix check.
    repo_root = tmp_path / "mist.ai"
    repo_root.mkdir()
    sibling = tmp_path / "mist.ai-results"
    check_results_root_outside_repo(sibling, repo_root=repo_root)
