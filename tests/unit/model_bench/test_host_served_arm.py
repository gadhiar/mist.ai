"""check_served_arm(): refuse to run suites against the wrong served arm."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from scripts.model_bench.bench_host import (  # noqa: E402
    ServedArmMismatchError,
    build_model_args,
    check_served_arm,
    extract_model_path_from_props,
    load_arms_doc,
    resolve_arm,
)

ARMS_DOC = load_arms_doc()


def test_matching_args_and_model_path_passes():
    arm = resolve_arm(ARMS_DOC, "c0")
    expected = build_model_args(arm, ARMS_DOC, {})
    props = {"model_path": f"/models/{arm['gguf']}"}
    check_served_arm(expected, expected, props, arm)  # must not raise


def test_mismatched_args_raises():
    arm = resolve_arm(ARMS_DOC, "c0")
    expected = build_model_args(arm, ARMS_DOC, {})
    running = [*expected[:-1], "different-tail-token"]
    props = {"model_path": f"/models/{arm['gguf']}"}
    with pytest.raises(ServedArmMismatchError):
        check_served_arm(running, expected, props, arm)


def test_wrong_model_path_raises_even_with_matching_args():
    arm = resolve_arm(ARMS_DOC, "c2")
    expected = build_model_args(arm, ARMS_DOC, {})
    props = {"model_path": "/models/unsloth/gemma-4-E4B-it-Q5_K_M.gguf"}  # wrong arm's gguf
    with pytest.raises(ServedArmMismatchError):
        check_served_arm(expected, expected, props, arm)


def test_extract_model_path_from_props_top_level_key():
    assert extract_model_path_from_props({"model_path": "/models/x.gguf"}) == "/models/x.gguf"


def test_extract_model_path_from_props_nested_fallback():
    props = {"default_generation_settings": {"model": "/models/y.gguf"}}
    assert extract_model_path_from_props(props) == "/models/y.gguf"


def test_extract_model_path_from_props_missing_returns_none():
    assert extract_model_path_from_props({}) is None
