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

COMPOSE_IMAGE_REF = "ghcr.io/ggml-org/llama.cpp:server-cuda-b11151@sha256:" + "ab" * 32
SNAPSHOT_IMAGE_ID = "sha256:" + "cd" * 32


def _check(
    running_args,
    expected_args,
    props,
    arm,
    *,
    image_token="compose:mist-llm",
    image_ref=COMPOSE_IMAGE_REF,
    running_config_image=COMPOSE_IMAGE_REF,
    running_image_id="sha256:" + "ef" * 32,
):
    check_served_arm(
        running_args, expected_args, props, arm,
        image_token=image_token, image_ref=image_ref,
        running_config_image=running_config_image, running_image_id=running_image_id,
    )


def test_matching_args_and_model_path_passes():
    arm = resolve_arm(ARMS_DOC, "c0")
    expected = build_model_args(arm, ARMS_DOC, {})
    props = {"model_path": f"/models/{arm['gguf']}"}
    _check(expected, expected, props, arm)  # must not raise


def test_mismatched_args_raises():
    arm = resolve_arm(ARMS_DOC, "c0")
    expected = build_model_args(arm, ARMS_DOC, {})
    running = [*expected[:-1], "different-tail-token"]
    props = {"model_path": f"/models/{arm['gguf']}"}
    with pytest.raises(ServedArmMismatchError):
        _check(running, expected, props, arm)


def test_wrong_model_path_raises_even_with_matching_args():
    arm = resolve_arm(ARMS_DOC, "c2")
    expected = build_model_args(arm, ARMS_DOC, {})
    props = {"model_path": "/models/unsloth/gemma-4-E4B-it-Q5_K_M.gguf"}  # wrong arm's gguf
    with pytest.raises(ServedArmMismatchError):
        _check(expected, expected, props, arm)


# --- image comparison (finding 6) ------------------------------------------


def test_compose_image_matches_passes():
    arm = resolve_arm(ARMS_DOC, "c0")
    expected = build_model_args(arm, ARMS_DOC, {})
    props = {"model_path": f"/models/{arm['gguf']}"}
    _check(
        expected, expected, props, arm,
        image_token="compose:mist-llm", image_ref=COMPOSE_IMAGE_REF,
        running_config_image=COMPOSE_IMAGE_REF,
    )  # must not raise


def test_compose_image_mismatch_raises():
    arm = resolve_arm(ARMS_DOC, "c0")
    expected = build_model_args(arm, ARMS_DOC, {})
    props = {"model_path": f"/models/{arm['gguf']}"}
    with pytest.raises(ServedArmMismatchError):
        _check(
            expected, expected, props, arm,
            image_token="compose:mist-llm", image_ref=COMPOSE_IMAGE_REF,
            running_config_image="ghcr.io/ggml-org/llama.cpp:server-cuda-b99999@sha256:" + "00" * 32,
        )


def test_snapshot_image_id_matches_passes():
    arm = resolve_arm(ARMS_DOC, "c0-old")
    expected = build_model_args(arm, ARMS_DOC, {})
    props = {"model_path": f"/models/{arm['gguf']}"}
    _check(
        expected, expected, props, arm,
        image_token="snapshot:mist-llm", image_ref=SNAPSHOT_IMAGE_ID,
        running_image_id=SNAPSHOT_IMAGE_ID,
    )  # must not raise


def test_snapshot_image_id_mismatch_raises():
    arm = resolve_arm(ARMS_DOC, "c0-old")
    expected = build_model_args(arm, ARMS_DOC, {})
    props = {"model_path": f"/models/{arm['gguf']}"}
    with pytest.raises(ServedArmMismatchError):
        _check(
            expected, expected, props, arm,
            image_token="snapshot:mist-llm", image_ref=SNAPSHOT_IMAGE_ID,
            running_image_id="sha256:" + "99" * 32,
        )


def test_extract_model_path_from_props_top_level_key():
    assert extract_model_path_from_props({"model_path": "/models/x.gguf"}) == "/models/x.gguf"


def test_extract_model_path_from_props_nested_fallback():
    props = {"default_generation_settings": {"model": "/models/y.gguf"}}
    assert extract_model_path_from_props(props) == "/models/y.gguf"


def test_extract_model_path_from_props_missing_returns_none():
    assert extract_model_path_from_props({}) is None
