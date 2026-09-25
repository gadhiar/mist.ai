"""arms.json resolution and server-argument building for every arm.

Covers: base inheritance (thinking/suites/harness overrides, family
carried through), the unknown-arm and unknown-key refusals, the
base-inheritance cycle refusal, and the REQUIRED-param refusal at
build_server_args() time.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from scripts.model_bench.bench_host import (  # noqa: E402
    ArmConfigError,
    MissingParamError,
    build_docker_run_args,
    build_model_args,
    build_server_args,
    layout_max_tokens,
    load_arms_doc,
    resolve_all_arms,
    resolve_arm,
    resolve_harness_tests,
)

ARMS_DOC = load_arms_doc()

ALL_ARM_IDS = list(ARMS_DOC["arms"])

REQUIRED_PARAM_ARMS = {"c3", "c3-think512", "c4", "c4-think512"}


def test_all_arms_resolve():
    resolved = resolve_all_arms(ARMS_DOC)
    assert set(resolved) == set(ALL_ARM_IDS)
    for arm_id, arm in resolved.items():
        assert arm["id"] == arm_id
        assert "gguf" in arm and arm["gguf"]
        assert "image" in arm and arm["image"]
        assert isinstance(arm["suites"], list)


@pytest.mark.parametrize("arm_id", ALL_ARM_IDS)
def test_build_server_args_argv_is_all_strings(arm_id):
    arm = resolve_arm(ARMS_DOC, arm_id)
    params = {p: "12" for p in arm["params_required"]}
    args = build_server_args(ARMS_DOC, arm, params)
    assert all(isinstance(tok, str) for tok in args)
    # Common args must always be present verbatim.
    assert "--ctx-size" in args
    assert "32768" in args
    assert "--no-kv-unified" in args


@pytest.mark.parametrize("arm_id", ALL_ARM_IDS)
def test_build_docker_run_args_argv_is_all_strings(arm_id):
    arm = resolve_arm(ARMS_DOC, arm_id)
    params = {p: "12" for p in arm["params_required"]}
    argv = build_docker_run_args(
        arm, ARMS_DOC, image_ref="dummy:image@sha256:" + "a" * 64, models_dir="/models", params=params
    )
    assert all(isinstance(tok, str) for tok in argv)
    assert argv[0] == "docker"
    assert argv[1] == "run"
    assert "mist-bench-llm" in argv
    assert "--gpus" in argv
    assert "127.0.0.1:8080:8080" in argv
    assert argv[-1] != argv[-2]  # sanity: not accidentally duplicated tail


@pytest.mark.parametrize("arm_id", sorted(REQUIRED_PARAM_ARMS))
def test_required_param_refusal(arm_id):
    arm = resolve_arm(ARMS_DOC, arm_id)
    assert arm["params_required"] == ["ncmoe"]
    with pytest.raises(MissingParamError):
        build_server_args(ARMS_DOC, arm, {})
    # Supplying it succeeds and the flag+value both appear, adjacent.
    args = build_server_args(ARMS_DOC, arm, {"ncmoe": "12"})
    idx = args.index("-ncmoe")
    assert args[idx + 1] == "12"


def test_c3_and_c4_load_mode_and_batch_overrides():
    # c3/c4 and every arm that inherits from them (think512, a3, a4) must carry
    # `-lm none` (b11151's replacement for `--no-mmap`) and the CPU-MoE prompt
    # processing batch sizes `-b 2048 -ub 2048`, applied in place over
    # common_args' `-b 1024 -ub 512` rather than appended as a duplicate.
    for arm_id in ("c3", "c4", "c3-think512", "c4-think512", "a3", "a4"):
        arm = resolve_arm(ARMS_DOC, arm_id)
        args = build_server_args(ARMS_DOC, arm, {"ncmoe": "12"})
        assert "--no-mmap" not in args, arm_id
        assert "-lm" in args, arm_id
        assert args[args.index("-lm") + 1] == "none", arm_id
        assert args.count("-b") == 1, arm_id
        assert args[args.index("-b") + 1] == "2048", arm_id
        assert args.count("-ub") == 1, arm_id
        assert args[args.index("-ub") + 1] == "2048", arm_id
    for arm_id in ("c3", "c4"):
        arm = resolve_arm(ARMS_DOC, arm_id)
        assert arm["stop_neo4j"] is True


def test_c0_and_c2_keep_default_batch_sizes():
    for arm_id in ("c0", "c0-old", "c0-prod", "c2", "c1-256", "c2-think512", "a1", "a2"):
        arm = resolve_arm(ARMS_DOC, arm_id)
        args = build_server_args(ARMS_DOC, arm, {})
        assert "-lm" not in args, arm_id
        assert args.count("-b") == 1, arm_id
        assert args[args.index("-b") + 1] == "1024", arm_id
        assert args.count("-ub") == 1, arm_id
        assert args[args.index("-ub") + 1] == "512", arm_id


@pytest.mark.parametrize("arm_id", ALL_ARM_IDS)
def test_no_arm_argv_has_a_duplicated_flag(arm_id):
    arm = resolve_arm(ARMS_DOC, arm_id)
    params = {p: "12" for p in arm["params_required"]}
    args = build_server_args(ARMS_DOC, arm, params)
    flags = [tok for tok in args if tok.startswith("-")]
    assert len(flags) == len(set(flags)), f"{arm_id}: duplicated flag(s) in {flags}"


def test_c0_old_has_no_thinking_flags():
    arm = resolve_arm(ARMS_DOC, "c0-old")
    assert arm["thinking"] is None
    args = build_server_args(ARMS_DOC, arm, {})
    assert "-rea" not in args


def test_thinking_on_budget_substitution():
    arm = resolve_arm(ARMS_DOC, "c1-256")
    args = build_server_args(ARMS_DOC, arm, {})
    idx = args.index("-rea")
    assert args[idx + 1] == "on"
    budget_idx = args.index("--reasoning-budget")
    assert args[budget_idx + 1] == "256"


def test_base_inheritance_overrides_thinking_suites_harness():
    base = resolve_arm(ARMS_DOC, "c0")
    child = resolve_arm(ARMS_DOC, "c1-256")
    # Inherited (not overridden by c1-256):
    assert child["gguf"] == base["gguf"]
    assert child["image"] == base["image"]
    assert child["family"] == base["family"]
    # Overridden by c1-256:
    assert child["thinking"] == {"mode": "on", "budget": 256}
    assert child["suites"] == ["layout"]
    assert child["harness"] is None


def test_tuning_arms_inherit_base_server_config_exactly():
    base = resolve_arm(ARMS_DOC, "c0")
    a1 = resolve_arm(ARMS_DOC, "a1")
    assert a1["gguf"] == base["gguf"]
    assert a1["image"] == base["image"]
    assert a1["thinking"] == base["thinking"]
    assert a1["family"] == base["family"]
    assert a1["tuning"] is True
    a3 = resolve_arm(ARMS_DOC, "a3")
    base3 = resolve_arm(ARMS_DOC, "c3")
    assert a3["gguf"] == base3["gguf"]
    assert a3["params_required"] == base3["params_required"]


def test_unknown_arm_id_raises():
    with pytest.raises(ArmConfigError):
        resolve_arm(ARMS_DOC, "does-not-exist")


def test_unknown_key_raises():
    doc = {"arms": {"bad": {"gguf": "x.gguf", "image": "compose:mist-llm", "not_a_real_key": 1}}}
    with pytest.raises(ArmConfigError):
        resolve_arm(doc, "bad")


def test_base_cycle_raises():
    doc = {
        "arms": {
            "x": {"base": "y", "gguf": "x.gguf", "image": "compose:mist-llm"},
            "y": {"base": "x", "gguf": "y.gguf", "image": "compose:mist-llm"},
        }
    }
    with pytest.raises(ArmConfigError):
        resolve_arm(doc, "x")


def test_missing_required_key_after_resolution_raises():
    doc = {"arms": {"bad": {"image": "compose:mist-llm"}}}
    with pytest.raises(ArmConfigError):
        resolve_arm(doc, "bad")


def test_required_param_without_arg_map_raises():
    doc = {
        "arms": {
            "bad": {
                "gguf": "x.gguf",
                "image": "compose:mist-llm",
                "params_required": ["ncmoe"],
            }
        }
    }
    with pytest.raises(ArmConfigError):
        resolve_arm(doc, "bad")


def test_harness_tests_sentinels():
    default_tests = resolve_harness_tests({"tests": "default", "candidate": "x", "iterations": 10})
    assert "schema_conformance_json_object" in default_tests
    assert "speed_minimal" in default_tests
    tuning_tests = resolve_harness_tests({"tests": "tuning", "candidate": "x", "iterations": 10})
    assert tuning_tests == ["schema_conformance"]


def test_harness_arms_use_default_and_tuning_arms_use_tuning():
    for arm_id in ("c0-old", "c0", "c0-prod", "c2", "c3"):
        arm = resolve_arm(ARMS_DOC, arm_id)
        assert arm["harness"]["tests"] == "default"
    for arm_id in ("a1", "a2", "a3"):
        arm = resolve_arm(ARMS_DOC, arm_id)
        assert arm["harness"]["tests"] == "tuning"
        assert arm["harness"]["iterations"] == 10


def test_layout_max_tokens():
    assert layout_max_tokens(None) == 256
    assert layout_max_tokens({"mode": "off"}) == 256
    assert layout_max_tokens({"mode": "on", "budget": 512}) == 4096


def test_build_model_args_includes_model_flag_before_server_args():
    arm = resolve_arm(ARMS_DOC, "c0")
    argv = build_model_args(arm, ARMS_DOC, {})
    assert argv[0] == "-m"
    assert argv[1] == f"/models/{arm['gguf']}"
