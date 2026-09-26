"""`run` suite resolution, and the harness test-order mirror.

A suite that `run` silently skips would leave a result directory that looks complete, so
`validate_run_suites` must refuse up front. `HARNESS_DEFAULT_TEST_ORDER` is a stdlib copy of the
harness's `DEFAULT_TEST_ORDER`, so a drift test pins the two together.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from scripts.model_bench.bench_host import (  # noqa: E402
    HARNESS_DEFAULT_TEST_ORDER,
    ArmConfigError,
    load_arms_doc,
    resolve_arm,
    validate_run_suites,
)

ARMS_DOC = load_arms_doc()
LAYOUT_DIR = Path("/nonexistent/layout-perception")


def test_default_suites_come_back_in_run_order():
    arm = resolve_arm(ARMS_DOC, "c0")
    assert validate_run_suites(arm, None, LAYOUT_DIR) == ["ttft", "correctness", "harness", "layout"]


def test_requested_subset_is_reordered_to_run_order():
    arm = resolve_arm(ARMS_DOC, "c0")
    assert validate_run_suites(arm, ["layout", "ttft"], LAYOUT_DIR) == ["ttft", "layout"]


def test_unknown_suite_is_refused():
    arm = resolve_arm(ARMS_DOC, "c0")
    with pytest.raises(ArmConfigError, match="unknown suite"):
        validate_run_suites(arm, ["speed"], LAYOUT_DIR)


def test_suite_not_declared_for_arm_is_refused():
    arm = resolve_arm(ARMS_DOC, "c1-512")
    with pytest.raises(ArmConfigError, match="not declared"):
        validate_run_suites(arm, ["harness"], LAYOUT_DIR)


def test_layout_without_layout_dir_is_refused():
    arm = resolve_arm(ARMS_DOC, "c1-256")
    with pytest.raises(ArmConfigError, match="layout-dir"):
        validate_run_suites(arm, None, None)


def test_non_layout_arm_needs_no_layout_dir():
    arm = resolve_arm(ARMS_DOC, "a1")
    assert validate_run_suites(arm, None, None) == ["ttft", "correctness", "harness"]


def test_harness_default_test_order_mirror_matches_the_harness():
    from scripts.eval_harness.run import DEFAULT_TEST_ORDER

    assert tuple(HARNESS_DEFAULT_TEST_ORDER) == tuple(DEFAULT_TEST_ORDER)
