"""scripts/model_bench/probes/help_flags.py: the `llama-server --help` flag parser
and the per-arm flag checker it feeds `plan --host-checks`.

Fixture: tests/unit/model_bench/fixtures/host/llama_server_help_b11151.txt, a REAL capture (not
hand-built, the one exception in fixtures/host/ -- see README.md's "Fixtures are hand-built, not
recorded"). The lead ran `docker run --rm <pinned compose:mist-llm image> --help` against the
pinned digest `ghcr.io/ggml-org/llama.cpp:server-cuda-b11151@sha256:014f7212...dc765c` on the host
on 2026-09-24 and saved the raw output verbatim (730 lines, scanned for personal data, none found).
It confirms the parser's column-0 assumption: 254 option lines start with `-` or `--` plus a letter
at column 0 (`grep -c -E '^--?[A-Za-z]'`; 258 lines start with `-` once the 4 `-----` section
headers are counted), and it
contains no `--no-mmap`.
"""

from __future__ import annotations

import hashlib
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

CAPTURE_PATH = FIXTURES_DIR / "llama_server_help_b11151.txt"
HELP_TEXT = CAPTURE_PATH.read_text(encoding="utf-8")

# Pins the real capture against accidental edits: a later change to this fixture (or a corrupted
# re-save) changes this hash, and the sha256/line-count test below catches it.
CAPTURE_EXPECTED_LINE_COUNT = 730
CAPTURE_EXPECTED_SHA256 = "1df5fa3916b239b2ace319411e3e6043025d8d937aff1d239b75cf2d9c551ad3"


def test_capture_line_count_and_sha256_are_pinned():
    assert len(HELP_TEXT.splitlines()) == CAPTURE_EXPECTED_LINE_COUNT
    digest = hashlib.sha256(CAPTURE_PATH.read_bytes()).hexdigest()
    assert digest == CAPTURE_EXPECTED_SHA256


def test_parser_extracts_both_short_and_long_forms():
    flags = parse_help_flags(HELP_TEXT)
    assert "-lm" in flags
    assert "--load-mode" in flags
    assert "-ncmoe" in flags
    assert "--n-cpu-moe" in flags
    assert "-rea" in flags
    assert "--reasoning" in flags


def test_parser_ignores_section_headers_and_usage_line():
    flags = parse_help_flags(HELP_TEXT)
    assert "-----" not in flags
    assert not any(f.strip("-") == "" for f in flags)
    # Every parsed flag is dash-prefixed by construction (the regex only ever
    # captures _FLAG_TOKEN matches), so this also rules out the parser ever
    # picking up a bare description word.
    assert all(f.startswith("-") for f in flags)


def test_real_capture_parses_at_least_200_flags():
    # Guards against a parser that silently matches nothing (or almost nothing) against real
    # output -- the failure mode the column-0 fix (a98f2f9) exists to prevent.
    flags = parse_help_flags(HELP_TEXT)
    assert len(flags) >= 200, f"only {len(flags)} flags parsed from the real capture"


def test_deeply_indented_continuation_lines_never_contribute_flags():
    # Description prose wraps onto lines indented well past the flag column (~40 spaces in the
    # real capture) and can itself start with `-` (e.g. "- auto: mmap, unless a device does not
    # support it", under -lm/--load-mode). Isolate exactly those lines and confirm they parse to
    # no flags at all, not just that the full-text parse happens to exclude them.
    continuation_lines = []
    for line in HELP_TEXT.splitlines():
        stripped = line.lstrip(" \t")
        if not stripped.startswith("-"):
            continue
        indent = len(line) - len(stripped)
        if indent > 8:
            continuation_lines.append(line)
    assert len(continuation_lines) >= 30, f"only {len(continuation_lines)} qualifying lines found"
    assert parse_help_flags("\n".join(continuation_lines)) == set()


def test_parser_accepts_column_0_option_lines():
    # llama.cpp's own common/arg.cpp (common_arg::to_string()) prints each
    # option starting at column 0, no leading indent -- confirmed by the real
    # capture (254 option lines start this way); this test isolates it.
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


def test_capture_omits_no_mmap():
    flags = parse_help_flags(HELP_TEXT)
    assert "--no-mmap" not in flags


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


def test_every_pinned_build_arm_flag_is_found_in_the_real_capture():
    # Every flag emitted by every arm in arms.json (c0-old excluded: it targets the b8808
    # snapshot, a different build -- see arm_targets_pinned_build) must appear in the real
    # capture's parsed flag set, with every REQUIRED param filled by a dummy value.
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


def test_check_all_arm_flags_passes_on_the_real_capture():
    from scripts.model_bench.bench_host import check_all_arm_flags

    assert check_all_arm_flags(HELP_TEXT, "pinned") is True


def test_check_all_arm_flags_fails_when_help_parses_to_nothing():
    from scripts.model_bench.bench_host import check_all_arm_flags

    assert check_all_arm_flags("no flags here\n", "pinned") is False


def test_check_all_arm_flags_fails_when_an_arm_flag_is_missing_from_help():
    # Drop the load-mode option line: c3/c4 then carry a flag the "build" does not list.
    from scripts.model_bench.bench_host import check_all_arm_flags

    stripped = "\n".join(
        line for line in HELP_TEXT.splitlines() if not line.startswith("-lm,")
    )
    assert check_all_arm_flags(stripped, "pinned") is False


def test_unknown_flags_is_order_preserving_and_deduplicated():
    help_flags = {"-a"}
    argv = ["-a", "-b", "-c", "-b", "value"]
    assert unknown_flags(argv, help_flags) == ["-b", "-c"]


def test_build_model_args_used_by_checker_includes_model_flag():
    arms_doc = load_arms_doc()
    arm = resolve_all_arms(arms_doc)["c3"]
    argv = build_model_args(arm, arms_doc, {"ncmoe": "12"})
    assert argv[0] == "-m"
