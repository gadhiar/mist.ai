"""CLI behavior: --check (rules-only and against --out), and threshold-driven verdict flips."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.model_bench import analyse  # noqa: E402

FIXTURE_RUN = Path(__file__).resolve().parent / "fixtures" / "analyse" / "run"


def test_check_with_no_results_validates_rules_only():
    assert analyse.main(["--check"]) == 0


def test_check_against_real_out_directory_is_clean(tmp_path):
    outputs = analyse.generate_outputs(FIXTURE_RUN)
    analyse.write_outputs(outputs, tmp_path)
    rc = analyse.main(["--check", "--results", str(FIXTURE_RUN), "--out", str(tmp_path)])
    assert rc == 0


def test_check_detects_a_modified_output_file(tmp_path):
    outputs = analyse.generate_outputs(FIXTURE_RUN)
    analyse.write_outputs(outputs, tmp_path)
    (tmp_path / "REPORT.md").write_text("tampered\n", encoding="utf-8")
    rc = analyse.main(["--check", "--results", str(FIXTURE_RUN), "--out", str(tmp_path)])
    assert rc == 1


def test_check_detects_a_missing_output_file(tmp_path):
    outputs = analyse.generate_outputs(FIXTURE_RUN)
    analyse.write_outputs(outputs, tmp_path)
    (tmp_path / "summary.json").unlink()
    rc = analyse.main(["--check", "--results", str(FIXTURE_RUN), "--out", str(tmp_path)])
    assert rc == 1


def test_check_detects_an_extra_file_not_generated(tmp_path):
    outputs = analyse.generate_outputs(FIXTURE_RUN)
    analyse.write_outputs(outputs, tmp_path)
    (tmp_path / "extra_file.txt").write_text("surprise\n", encoding="utf-8")
    rc = analyse.main(["--check", "--results", str(FIXTURE_RUN), "--out", str(tmp_path)])
    assert rc == 1


def test_run_writes_expected_files(tmp_path):
    rc = analyse.main(["--results", str(FIXTURE_RUN), "--out", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "REPORT.md").exists()
    assert (tmp_path / "summary.json").exists()
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema"] == 1
    assert "rules" in summary


def test_threshold_change_flips_r1_verdict(tmp_path):
    rules = json.loads((_REPO_ROOT / "scripts" / "model_bench" / "decision_rules.json").read_text(encoding="utf-8"))
    outputs_before = analyse.generate_outputs(FIXTURE_RUN)
    summary_before = json.loads(outputs_before.files["summary.json"])
    r1_before = next(r for r in summary_before["rules"] if r["id"] == "R1")
    assert r1_before["verdict"] == "pass"

    # Raise the threshold above c1-512's actual accuracy so the same data now fails R1.
    rules["thresholds"]["r1_keep_e4b_layout_acc_min"] = 0.999
    tampered_path = tmp_path / "decision_rules.json"
    tampered_path.write_text(json.dumps(rules, sort_keys=True, indent=2), encoding="utf-8")

    outputs_after = analyse.generate_outputs(FIXTURE_RUN, rules_path=tampered_path)
    summary_after = json.loads(outputs_after.files["summary.json"])
    r1_after = next(r for r in summary_after["rules"] if r["id"] == "R1")
    assert r1_after["verdict"] == "fail"


def test_validate_decision_rules_flags_unknown_arm(tmp_path):
    rules = json.loads((_REPO_ROOT / "scripts" / "model_bench" / "decision_rules.json").read_text(encoding="utf-8"))
    arms = analyse.load_arms()
    rules["constants"]["r7_build_pair"]["new"] = "does-not-exist"
    problems = analyse.validate_decision_rules(rules, arms)
    assert any("does-not-exist" in p for p in problems)
