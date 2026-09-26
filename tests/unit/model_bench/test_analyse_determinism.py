"""Two independent runs of generate_outputs() on the same fixture must be byte-identical."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.model_bench import analyse  # noqa: E402

FIXTURE_RUN = Path(__file__).resolve().parent / "fixtures" / "analyse" / "run"


def test_generate_outputs_is_byte_deterministic():
    out1 = analyse.generate_outputs(FIXTURE_RUN)
    out2 = analyse.generate_outputs(FIXTURE_RUN)
    assert out1.files.keys() == out2.files.keys()
    for rel_path in out1.files:
        assert out1.files[rel_path] == out2.files[rel_path], f"non-deterministic output: {rel_path}"


def test_outputs_use_lf_line_endings_only():
    outputs = analyse.generate_outputs(FIXTURE_RUN)
    for rel_path, content in outputs.files.items():
        assert "\r" not in content, f"{rel_path} contains a CR byte"


def test_write_then_check_round_trips(tmp_path):
    outputs = analyse.generate_outputs(FIXTURE_RUN)
    analyse.write_outputs(outputs, tmp_path)
    problems = analyse.diff_outputs(outputs, tmp_path)
    assert problems == []
