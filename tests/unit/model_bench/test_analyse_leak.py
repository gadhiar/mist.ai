"""No model text (decision 8): the fixture's sentinel string must appear in no output file."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.model_bench import analyse  # noqa: E402

FIXTURE_RUN = Path(__file__).resolve().parent / "fixtures" / "analyse" / "run"
SENTINEL = "SENTINEL_DO_NOT_LEAK_9f3a21"


def test_sentinel_present_in_raw_fixture():
    # Sanity: prove the sentinel really is in the raw input, so the absence
    # check below is meaningful rather than vacuous.
    raw = (FIXTURE_RUN / "c0" / "harness" / "harness" / "bench-c0.jsonl").read_text(encoding="utf-8")
    assert SENTINEL in raw


def test_sentinel_absent_from_every_generated_output():
    outputs = analyse.generate_outputs(FIXTURE_RUN)
    for rel_path, content in outputs.files.items():
        assert SENTINEL not in content, f"leaked into {rel_path}"


def test_grades_harness_rows_are_allowlisted_keys_only():
    outputs = analyse.generate_outputs(FIXTURE_RUN)
    allowed = {"test_name", "case_id", "iteration", "score", "passed", "finish_reason", "errored"}
    for rel_path, content in outputs.files.items():
        if not rel_path.startswith("grades/") or "/harness.jsonl" not in rel_path:
            continue
        for line in content.splitlines():
            import json

            row = json.loads(line)
            assert set(row) <= allowed, f"{rel_path} row has extra keys: {set(row) - allowed}"


def test_grades_layout_rows_are_allowlisted_keys_only():
    outputs = analyse.generate_outputs(FIXTURE_RUN)
    allowed = set(analyse.GRADES_LAYOUT_ALLOWLIST) | {"usage"}
    for rel_path, content in outputs.files.items():
        if not rel_path.startswith("grades/") or "/layout_" not in rel_path:
            continue
        for line in content.splitlines():
            import json

            row = json.loads(line)
            assert set(row) <= allowed, f"{rel_path} row has extra keys: {set(row) - allowed}"
