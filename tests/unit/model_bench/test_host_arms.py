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

REQUIRED_PARAM_ARMS = {"c3", "c3-think512", "c4", "c4-think512", "c3-q3", "c3-iq4"}


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
    # Common args must always be present verbatim, except where arg_overrides
    # deliberately replaces a value in place (plan v2's context arms override
    # --ctx-size away from the common 32768 -- that override is itself checked by
    # test_context_arms_ctx_size_and_kv_quant).
    assert "--ctx-size" in args
    expected_ctx_size = arm["arg_overrides"].get("--ctx-size", "32768")
    assert args[args.index("--ctx-size") + 1] == expected_ctx_size
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


# --- plan v2 (2026-09-25): c1-1024 and the E4B context arms --------------


def test_c1_1024_thinking_budget_and_suites():
    arm = resolve_arm(ARMS_DOC, "c1-1024")
    assert arm["thinking"] == {"mode": "on", "budget": 1024}
    assert arm["suites"] == ["layout", "ttft"]
    assert arm["harness"] is None
    assert arm["tokens_vs_c0"] == "differs-by-design"
    args = build_server_args(ARMS_DOC, arm, {})
    idx = args.index("--reasoning-budget")
    assert args[idx + 1] == "1024"


@pytest.mark.parametrize(
    "arm_id,expected_ctx,expect_q4_kv",
    [
        ("c0-ctx64k", "65536", False),
        ("c0-ctx128k", "131072", False),
        ("c0-ctx128k-q4kv", "131072", True),
    ],
)
def test_context_arms_ctx_size_and_kv_quant(arm_id, expected_ctx, expect_q4_kv):
    arm = resolve_arm(ARMS_DOC, arm_id)
    assert arm["suites"] == ["ttft", "correctness", "harness"]
    assert arm["harness"] == {"candidate": "bench-c0", "tests": "context", "iterations": 10}
    args = build_server_args(ARMS_DOC, arm, {})
    assert args.count("--ctx-size") == 1
    assert args[args.index("--ctx-size") + 1] == expected_ctx
    ctk = args[args.index("-ctk") + 1]
    ctv = args[args.index("-ctv") + 1]
    if expect_q4_kv:
        assert (ctk, ctv) == ("q4_0", "q4_0")
    else:
        assert (ctk, ctv) == ("q8_0", "q8_0")


def test_context_arms_tokens_vs_c0_labels():
    assert resolve_arm(ARMS_DOC, "c0-ctx64k")["tokens_vs_c0"] == "expected-identical-unverified"
    assert resolve_arm(ARMS_DOC, "c0-ctx128k")["tokens_vs_c0"] == "expected-identical-unverified"
    assert resolve_arm(ARMS_DOC, "c0-ctx128k-q4kv")["tokens_vs_c0"] == "may-differ"


def test_context_harness_tests_sentinel():
    tests = resolve_harness_tests({"tests": "context", "candidate": "bench-c0", "iterations": 10})
    assert tests == ["schema_conformance", "schema_conformance_json_object", "tool_selection"]


def test_c0_ub1024_batch_override_and_label():
    arm = resolve_arm(ARMS_DOC, "c0-ub1024")
    assert arm["tokens_vs_c0"] == "may-differ"
    args = build_server_args(ARMS_DOC, arm, {})
    assert args.count("-b") == 1
    assert args[args.index("-b") + 1] == "2048"
    assert args.count("-ub") == 1
    assert args[args.index("-ub") + 1] == "1024"


def test_default_arms_have_no_tokens_vs_c0_label():
    # Pre-plan-v2 arms make no tokens_vs_c0 claim; the key is None, not one of the
    # allowed labels.
    for arm_id in ("c0", "c2", "c3", "c1-256", "c1-512"):
        assert resolve_arm(ARMS_DOC, arm_id)["tokens_vs_c0"] is None


def test_unknown_tokens_vs_c0_raises():
    doc = {
        "arms": {
            "bad": {
                "gguf": "x.gguf",
                "image": "compose:mist-llm",
                "tokens_vs_c0": "not-a-real-label",
            }
        }
    }
    with pytest.raises(ArmConfigError):
        resolve_arm(doc, "bad")


# --- T5 (plan v3, 2026-09-25): c1-2048/c1-unbudgeted, c5/c5-think1024, c6, -----
# --- c3-q3/c3-iq4 ---------------------------------------------------------


@pytest.mark.parametrize(
    "arm_id,expected_budget",
    [("c1-2048", 2048), ("c1-unbudgeted", -1)],
)
def test_c1_2048_thinking_budget_and_suites(arm_id, expected_budget):
    arm = resolve_arm(ARMS_DOC, arm_id)
    assert arm["thinking"] == {"mode": "on", "budget": expected_budget}
    assert arm["suites"] == ["layout", "ttft"]
    assert arm["harness"] is None
    assert arm["tokens_vs_c0"] == "differs-by-design"
    args = build_server_args(ARMS_DOC, arm, {})
    idx = args.index("--reasoning-budget")
    assert args[idx + 1] == str(expected_budget)
    assert args.count("--reasoning-budget") == 1
    assert args.count("-rea") == 1


def test_c6_is_c0_with_only_gguf_and_harness_overridden():
    base = resolve_arm(ARMS_DOC, "c0")
    c6 = resolve_arm(ARMS_DOC, "c6")
    assert c6["gguf"] == "unsloth/gemma-4-E4B-it-Q8_0.gguf"
    assert c6["gguf"] != base["gguf"]
    assert c6["image"] == base["image"]
    assert c6["family"] == base["family"]
    assert c6["thinking"] == base["thinking"]
    assert c6["suites"] == base["suites"]
    assert c6["harness"] == {"candidate": "bench-c6", "tests": "default", "iterations": 10}
    assert c6["tokens_vs_c0"] == "may-differ"
    args = build_server_args(ARMS_DOC, c6, {})
    assert "-rea" in args and args[args.index("-rea") + 1] == "off"


def test_c5_full_card_thinking_off_family_qwen():
    arm = resolve_arm(ARMS_DOC, "c5")
    assert arm["gguf"] == "unsloth/Qwen3.5-9B-Q8_0.gguf"
    assert arm["family"] == "qwen"
    assert arm["thinking"] == {"mode": "off"}
    assert arm["suites"] == ["ttft", "correctness", "harness", "layout"]
    assert arm["harness"] == {"candidate": "bench-c5", "tests": "default", "iterations": 10}
    assert arm["params_required"] == []
    assert arm["tokens_vs_c0"] == "differs-by-design"
    args = build_server_args(ARMS_DOC, arm, {})
    assert "-ncmoe" not in args
    assert args.count("--ctx-size") == 1
    assert args[args.index("--ctx-size") + 1] == "32768"
    assert "-rea" in args and args[args.index("-rea") + 1] == "off"
    # Qwen sampling, not gemma's.
    assert "--presence-penalty" in args
    assert args[args.index("--top-k") + 1] == "20"


def test_c5_think1024_inherits_c5_and_sets_budget():
    base = resolve_arm(ARMS_DOC, "c5")
    arm = resolve_arm(ARMS_DOC, "c5-think1024")
    assert arm["gguf"] == base["gguf"]
    assert arm["family"] == base["family"]
    assert arm["thinking"] == {"mode": "on", "budget": 1024}
    assert arm["suites"] == ["layout"]
    assert arm["harness"] is None
    args = build_server_args(ARMS_DOC, arm, {})
    idx = args.index("--reasoning-budget")
    assert args[idx + 1] == "1024"


@pytest.mark.parametrize(
    "arm_id,expected_gguf,expected_candidate",
    [
        ("c3-q3", "unsloth/gemma-4-26B-A4B-it-UD-Q3_K_XL.gguf", "bench-c3-q3"),
        ("c3-iq4", "unsloth/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf", "bench-c3-iq4"),
    ],
)
def test_c3_lower_quants_override_only_gguf_and_harness(arm_id, expected_gguf, expected_candidate):
    base = resolve_arm(ARMS_DOC, "c3")
    arm = resolve_arm(ARMS_DOC, arm_id)
    assert arm["gguf"] == expected_gguf
    assert arm["gguf"] != base["gguf"]
    assert arm["image"] == base["image"]
    assert arm["family"] == base["family"]
    assert arm["suites"] == base["suites"]
    assert arm["params_required"] == ["ncmoe"]
    assert arm["param_arg_map"] == {"ncmoe": "-ncmoe"}
    assert arm["stop_neo4j"] is True
    assert arm["harness"] == {"candidate": expected_candidate, "tests": "default", "iterations": 10}
    assert arm["tokens_vs_c0"] == "differs-by-design"

    with pytest.raises(MissingParamError):
        build_server_args(ARMS_DOC, arm, {})

    args = build_server_args(ARMS_DOC, arm, {"ncmoe": "12"})
    idx = args.index("-ncmoe")
    assert args[idx + 1] == "12"
    assert "--no-mmap" not in args
    assert args[args.index("-lm") + 1] == "none"
    assert args.count("-b") == 1 and args[args.index("-b") + 1] == "2048"
    assert args.count("-ub") == 1 and args[args.index("-ub") + 1] == "2048"
