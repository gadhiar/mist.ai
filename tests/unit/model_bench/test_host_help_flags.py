"""scripts/model_bench/probes/help_flags.py: the `llama-server --help` flag parser
and the per-arm flag checker it feeds `plan --host-checks`.

Fixture: tests/unit/model_bench/fixtures/host/llama_server_help_excerpt.txt, a
hand-written excerpt in the style of `llama-server --help` (NOT a capture -- see
that file's header comment). It includes `-lm, --load-mode` and omits `--no-mmap`,
matching what the lead verified against the pinned b11151 digest on 2026-09-24.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.model_bench.bench_host import (  # noqa: E402
    FIXTURES_DIR,
    arm_targets_pinned_build,
    build_model_args,
    check_arm_flags,
    load_arms_doc,
    resolve_all_arms,
)
from scripts.model_bench.probes.help_flags import parse_help_flags, unknown_flags  # noqa: E402

HELP_TEXT = (FIXTURES_DIR / "llama_server_help_excerpt.txt").read_text(encoding="utf-8")


def test_parser_extracts_both_short_and_long_forms():
    flags = parse_help_flags(HELP_TEXT)
    assert "-lm" in flags
    assert "--load-mode" in flags
    assert "-ncmoe" in flags
    assert "--n-cpu-moe" in flags
    assert "-rea" in flags
    assert "--reasoning" in flags


def test_parser_ignores_section_headers_usage_and_continuation_lines():
    flags = parse_help_flags(HELP_TEXT)
    assert "-----" not in flags
    assert not any(f.strip("-") == "" for f in flags)
    # The wrapped description line under -lm/-rea must not itself be read as
    # option flags (it has no leading dash at all).
    assert "model" not in flags
    assert "controls" not in flags


def test_help_fixture_omits_no_mmap():
    flags = parse_help_flags(HELP_TEXT)
    assert "--no-mmap" not in flags


def test_parser_accepts_column_0_option_lines():
    # llama.cpp's own common/arg.cpp (common_arg::to_string()) prints each
    # option starting at column 0, no leading indent -- the fixture's own
    # option lines are mostly this shape; this test isolates it.
    flags = parse_help_flags("-lm,   --load-mode {auto|none|mmap}\n")
    assert flags == {"-lm", "--load-mode"}


def test_parser_accepts_three_comma_separated_forms():
    flags = parse_help_flags("-h,    --help, --usage        print usage and exit\n")
    assert flags == {"-h", "--help", "--usage"}


def test_parser_ignores_deeply_indented_continuation_lines_with_dashes():
    # Description prose can contain dash-prefixed words; a continuation line indented
    # past 8 columns must never contribute flags.
    text = (
        "-rea,  --reasoning [on|off|auto]\n"
        "                                        -1 disables; --foo is prose here\n"
    )
    assert parse_help_flags(text) == {"-rea", "--reasoning"}


def test_checker_flags_no_mmap_on_the_old_c3_argv():
    old_c3_argv = [
        "-m", "/models/x.gguf",
        "--ctx-size", "32768",
        "--no-mmap",
        "-ub", "2048",
        "-ncmoe", "12",
    ]
    help_flags = parse_help_flags(HELP_TEXT)
    unknown = unknown_flags(old_c3_argv, help_flags)
    assert "--no-mmap" in unknown


def test_every_pinned_build_arm_passes_the_fixture():
    arms_doc = load_arms_doc()
    resolved = resolve_all_arms(arms_doc)
    help_flags = parse_help_flags(HELP_TEXT)
    checked = 0
    for arm_id, arm in resolved.items():
        if not arm_targets_pinned_build(arm):
            continue
        checked += 1
        params = {p: "12" for p in arm["params_required"]}
        unknown = check_arm_flags(help_flags, arms_doc, arm, params)
        assert unknown == [], f"{arm_id}: unknown flags {unknown}"
    assert checked > 0


def test_c0_old_does_not_target_the_pinned_build():
    arms_doc = load_arms_doc()
    resolved = resolve_all_arms(arms_doc)
    assert arm_targets_pinned_build(resolved["c0-old"]) is False


def test_unknown_flags_is_order_preserving_and_deduplicated():
    help_flags = {"-a"}
    argv = ["-a", "-b", "-c", "-b", "value"]
    assert unknown_flags(argv, help_flags) == ["-b", "-c"]


def test_build_model_args_used_by_checker_includes_model_flag():
    arms_doc = load_arms_doc()
    arm = resolve_all_arms(arms_doc)["c3"]
    argv = build_model_args(arm, arms_doc, {"ncmoe": "12"})
    assert argv[0] == "-m"
