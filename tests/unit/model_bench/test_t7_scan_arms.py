"""T7 (plan v3 scan) additive arms: c7 (Granite 4.2 8B), c8 (Spark-X2.5-4B),
c9 (gpt-oss-20b). Same additive-only discipline as T5: no existing arm,
common_args, thinking_args, or decision_rules.json changes; only new arm ids
and, where needed, new `sampling` family keys.

Covers: server-argv contents for each new/sibling arm, no duplicate flags
(the ALL_ARM_IDS-parametrized checks in test_host_arms.py already cover
this generically once these arms exist in arms.json), the ncmoe REQUIRED
param on c9/c9-medium, the tokens_vs_c0 labels (including the verified
inheritance behaviour -- a `base`-arm that does not itself set
tokens_vs_c0 inherits the base's resolved value; see
test_thinking_budget_siblings_inherit_tokens_vs_c0_from_their_root below),
and each new bench-cN harness candidate wiring.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from scripts.model_bench.bench_host import (  # noqa: E402
    MissingParamError,
    arm_targets_pinned_build,
    build_model_args,
    build_server_args,
    check_arm_flags,
    load_arms_doc,
    resolve_all_arms,
    resolve_arm,
)
from scripts.model_bench.probes.help_flags import parse_help_flags  # noqa: E402

ARMS_DOC = load_arms_doc()
FIXTURES_DIR = _REPO_ROOT / "tests" / "unit" / "model_bench" / "fixtures" / "host"
HELP_TEXT = (FIXTURES_DIR / "llama_server_help_b11151.txt").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# c7: Granite 4.2 8B -- dense, full card
# ---------------------------------------------------------------------------


def test_c7_full_card_thinking_off():
    arm = resolve_arm(ARMS_DOC, "c7")
    assert arm["gguf"] == "ibm-granite/granite-4.2-8b-Q6_K.gguf"
    assert arm["image"] == "compose:mist-llm"
    assert arm["thinking"] == {"mode": "off"}
    assert arm["suites"] == ["ttft", "correctness", "harness", "layout"]
    assert arm["harness"] == {"candidate": "bench-c7", "tests": "default", "iterations": 10}
    assert arm["params_required"] == []
    assert arm["tokens_vs_c0"] == "differs-by-design"
    args = build_server_args(ARMS_DOC, arm, {})
    assert "-rea" in args and args[args.index("-rea") + 1] == "off"
    assert "-ncmoe" not in args
    assert "--chat-template-kwargs" not in args


def test_c7_think_inherits_c7_and_sets_budget_1024():
    base = resolve_arm(ARMS_DOC, "c7")
    arm = resolve_arm(ARMS_DOC, "c7-think")
    assert arm["gguf"] == base["gguf"]
    assert arm["family"] == base["family"]
    assert arm["thinking"] == {"mode": "on", "budget": 1024}
    assert arm["suites"] == ["layout"]
    assert arm["harness"] is None
    args = build_server_args(ARMS_DOC, arm, {})
    assert "-rea" in args and args[args.index("-rea") + 1] == "on"
    idx = args.index("--reasoning-budget")
    assert args[idx + 1] == "1024"


# ---------------------------------------------------------------------------
# c8: Spark-X2.5-4B -- dense, full card
# ---------------------------------------------------------------------------


def test_c8_full_card_thinking_off():
    arm = resolve_arm(ARMS_DOC, "c8")
    assert arm["gguf"] == "XHToken/Spark-X2.5-4B-Q8_0.gguf"
    assert arm["image"] == "compose:mist-llm"
    assert arm["thinking"] == {"mode": "off"}
    assert arm["suites"] == ["ttft", "correctness", "harness", "layout"]
    assert arm["harness"] == {"candidate": "bench-c8", "tests": "default", "iterations": 10}
    assert arm["params_required"] == []
    assert arm["tokens_vs_c0"] == "differs-by-design"
    args = build_server_args(ARMS_DOC, arm, {})
    assert "-rea" in args and args[args.index("-rea") + 1] == "off"


def test_c8_think_inherits_c8_and_sets_budget_1024():
    base = resolve_arm(ARMS_DOC, "c8")
    arm = resolve_arm(ARMS_DOC, "c8-think")
    assert arm["gguf"] == base["gguf"]
    assert arm["family"] == base["family"]
    assert arm["thinking"] == {"mode": "on", "budget": 1024}
    assert arm["suites"] == ["layout"]
    assert arm["harness"] is None
    args = build_server_args(ARMS_DOC, arm, {})
    idx = args.index("--reasoning-budget")
    assert args[idx + 1] == "1024"


# ---------------------------------------------------------------------------
# c9: gpt-oss-20b -- MoE, ncmoe REQUIRED, Harmony template, reasoning effort
# ---------------------------------------------------------------------------


def test_c9_required_ncmoe_and_cpu_moe_batch_overrides():
    arm = resolve_arm(ARMS_DOC, "c9")
    assert arm["gguf"] == "ggml-org/gpt-oss-20b-MXFP4.gguf"
    assert arm["params_required"] == ["ncmoe"]
    assert arm["param_arg_map"] == {"ncmoe": "-ncmoe"}
    assert arm["stop_neo4j"] is True
    with pytest.raises(MissingParamError):
        build_server_args(ARMS_DOC, arm, {})
    args = build_server_args(ARMS_DOC, arm, {"ncmoe": "12"})
    idx = args.index("-ncmoe")
    assert args[idx + 1] == "12"
    assert "--no-mmap" not in args
    assert args[args.index("-lm") + 1] == "none"
    assert args.count("-b") == 1 and args[args.index("-b") + 1] == "2048"
    assert args.count("-ub") == 1 and args[args.index("-ub") + 1] == "2048"


def test_c9_reasoning_always_on_unbudgeted_plus_effort_low():
    # thinking: {"mode": "on", "budget": -1} -- gpt-oss always reasons, so
    # -rea on --reasoning-budget -1 (unrestricted) matches that, and also
    # gives the layout suite the 4096-token budget every other thinking-on
    # arm gets (layout_max_tokens keys off thinking's mode). Effort is a
    # separate axis, set via --reasoning-effort in extra_args. See
    # README.md's c9 note.
    arm = resolve_arm(ARMS_DOC, "c9")
    assert arm["thinking"] == {"mode": "on", "budget": -1}
    args = build_server_args(ARMS_DOC, arm, {"ncmoe": "12"})
    assert args.count("-rea") == 1
    assert args[args.index("-rea") + 1] == "on"
    assert args.count("--reasoning-budget") == 1
    assert args[args.index("--reasoning-budget") + 1] == "-1"
    idx = args.index("--reasoning-effort")
    assert args[idx + 1] == "low"


def test_c9_suites_and_harness_and_label():
    arm = resolve_arm(ARMS_DOC, "c9")
    assert arm["suites"] == ["ttft", "correctness", "harness", "layout"]
    assert arm["harness"] == {"candidate": "bench-c9", "tests": "default", "iterations": 10}
    assert arm["tokens_vs_c0"] == "differs-by-design"


def test_c9_medium_inherits_ncmoe_thinking_and_overrides_effort():
    base = resolve_arm(ARMS_DOC, "c9")
    arm = resolve_arm(ARMS_DOC, "c9-medium")
    assert arm["gguf"] == base["gguf"]
    assert arm["params_required"] == ["ncmoe"]
    assert arm["stop_neo4j"] is True
    assert arm["suites"] == ["layout"]
    assert arm["harness"] is None
    # thinking is inherited unchanged from c9 (not overridden by c9-medium).
    assert arm["thinking"] == {"mode": "on", "budget": -1}
    args = build_server_args(ARMS_DOC, arm, {"ncmoe": "12"})
    assert args.count("-rea") == 1
    assert args[args.index("-rea") + 1] == "on"
    assert args.count("--reasoning-budget") == 1
    assert args[args.index("--reasoning-budget") + 1] == "-1"
    idx = args.index("--reasoning-effort")
    assert args[idx + 1] == "medium"
    assert "low" not in args
    assert args.count("--reasoning-effort") == 1
    assert args[args.index("-lm") + 1] == "none"
    assert args.count("-b") == 1 and args[args.index("-b") + 1] == "2048"
    assert args.count("-ub") == 1 and args[args.index("-ub") + 1] == "2048"


# ---------------------------------------------------------------------------
# tokens_vs_c0 inheritance -- verified behaviour, not assumed
# ---------------------------------------------------------------------------


def test_thinking_budget_siblings_inherit_tokens_vs_c0_from_their_root():
    # resolve_arm's merge starts from a copy of the resolved base arm, so a
    # child that does not itself set tokens_vs_c0 inherits the base's
    # resolved value rather than defaulting to None. Verified here (not
    # assumed) against the precedent this repo already has, c5-think1024,
    # whose base c5 carries "differs-by-design".
    assert resolve_arm(ARMS_DOC, "c5-think1024")["tokens_vs_c0"] == "differs-by-design"
    assert resolve_arm(ARMS_DOC, "c7-think")["tokens_vs_c0"] == "differs-by-design"
    assert resolve_arm(ARMS_DOC, "c8-think")["tokens_vs_c0"] == "differs-by-design"
    assert resolve_arm(ARMS_DOC, "c9-medium")["tokens_vs_c0"] == "differs-by-design"


# ---------------------------------------------------------------------------
# Flag check against the real b11151 capture, for the three new root arms
# and their siblings (test_host_help_flags.py's
# test_every_pinned_build_arm_flag_is_found_in_the_real_capture already
# covers every arm generically; this pins the specific new-arm ids so a
# regression here fails with a narrower, arm-named test too).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arm_id", ["c7", "c7-think", "c8", "c8-think", "c9", "c9-medium"]
)
def test_new_arm_flags_are_all_known_to_the_real_capture(arm_id):
    help_flags = parse_help_flags(HELP_TEXT)
    arm = resolve_all_arms(ARMS_DOC)[arm_id]
    assert arm_targets_pinned_build(arm) is True
    params = {p: "12" for p in arm["params_required"]}
    unknown = check_arm_flags(help_flags, ARMS_DOC, arm, params)
    assert unknown == [], f"{arm_id}: unknown flags {unknown}"


def test_build_model_args_new_arms_include_model_flag():
    for arm_id, params in (("c7", {}), ("c8", {}), ("c9", {"ncmoe": "12"})):
        arm = resolve_all_arms(ARMS_DOC)[arm_id]
        argv = build_model_args(arm, ARMS_DOC, params)
        assert argv[0] == "-m"
        assert argv[1] == f"/models/{arm['gguf']}"
