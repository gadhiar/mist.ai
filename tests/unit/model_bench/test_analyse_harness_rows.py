"""Harness rows pair each statistic with its own interval (finding i67, decision D6).

`harness_score[test]` is a MEAN of graded case scores: it carries the cluster bootstrap CI
and no Wilson interval. `harness_pass_rate[test]` is the pass PROPORTION: it carries the
Wilson interval on (pass_count, n). The report must never print a mean beside a Wilson
interval.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval_harness import scorers as harness_scorers  # noqa: E402
from scripts.model_bench import analyse  # noqa: E402

FIXTURE_RUN = Path(__file__).resolve().parent / "fixtures" / "analyse" / "run"

STATS_CFG = {
    "wilson_z": analyse.WILSON_Z_DEFAULT,
    "B": 200,
    "seed": analyse.BOOTSTRAP_SEED_DEFAULT,
    "confidence": analyse.BOOTSTRAP_CONFIDENCE_DEFAULT,
}


def _case(case_id: str, iteration: int, passed: bool, score: float) -> harness_scorers.CaseScore:
    return harness_scorers.CaseScore(
        candidate_id="cand",
        test_name="t",
        case_id=case_id,
        iteration=iteration,
        passed=passed,
        score=score,
        breakdown={},
        examined=None,
        error=None,
    )


def _test_scores() -> harness_scorers.TestScores:
    """Four graded cases whose mean score (0.625) differs from the pass rate (0.5)."""
    ts = harness_scorers.TestScores(test_name="t")
    for case in (
        _case("a", 1, True, 1.0),
        _case("a", 2, False, 0.5),
        _case("b", 1, True, 1.0),
        _case("b", 2, False, 0.0),
    ):
        ts.case_scores.append(case)
        if case.passed:
            ts.pass_count += 1
        else:
            ts.fail_count += 1
    return ts


def _score_row(ts: harness_scorers.TestScores, **overrides) -> analyse.MetricResult:
    fields = {
        "value": ts.mean_score,
        "n": len(ts.case_scores),
        "n_expected": len(ts.case_scores),
        "k": ts.pass_count,
        "bootstrap": (0.1, 0.9),
    }
    fields.update(overrides)
    return analyse.MetricResult(**fields)


# ---------------------------------------------------------------------------
# Acceptance 1: the mean-score row has bootstrap, no Wilson, value == mean_score
# ---------------------------------------------------------------------------


def test_harness_score_rows_carry_bootstrap_and_no_wilson():
    inputs = analyse.load_arm_inputs(FIXTURE_RUN, "c0")
    scores, raw = analyse.compute_harness_scores_for_arm(inputs, STATS_CFG)
    assert scores, "fixture arm c0 must have harness rows"
    for name, row in scores.items():
        assert row.wilson is None, name
        assert row.bootstrap is not None, name
        assert row.value == raw[name].mean_score, name


# ---------------------------------------------------------------------------
# Acceptance 2: the pass-rate row
# ---------------------------------------------------------------------------


def test_pass_rate_row_value_k_and_wilson_match_pass_count_over_n():
    ts = _test_scores()
    assert ts.mean_score != ts.pass_count / len(ts.case_scores)  # the two statistics differ
    rows = analyse.compute_harness_pass_rates({"t": _score_row(ts)}, {"t": ts}, STATS_CFG)
    row = rows["t"]
    assert row.value == 2 / 4
    assert row.k == 2
    assert row.n == 4
    assert row.wilson == analyse.wilson_interval(2, 4, STATS_CFG["wilson_z"])
    assert row.wilson is not None
    assert row.bootstrap is None
    assert row.usable()


def test_pass_rate_rows_from_fixture_arm_match_wilson_of_pass_count():
    inputs = analyse.load_arm_inputs(FIXTURE_RUN, "c0")
    scores, raw = analyse.compute_harness_scores_for_arm(inputs, STATS_CFG)
    rates = analyse.compute_harness_pass_rates(scores, raw, STATS_CFG)
    assert rates.keys() == scores.keys()
    for name, row in rates.items():
        n = len(raw[name].case_scores)
        assert row.value == raw[name].pass_count / n
        assert row.k == raw[name].pass_count
        assert row.n == n
        assert row.wilson == analyse.wilson_interval(raw[name].pass_count, n, STATS_CFG["wilson_z"])


def test_pass_rate_row_inherits_incomplete_coverage_from_score_row():
    ts = _test_scores()
    score_row = _score_row(
        ts, n_expected=10, complete=False, note="incomplete coverage: 4/10"
    )
    row = analyse.compute_harness_pass_rates({"t": score_row}, {"t": ts}, STATS_CFG)["t"]
    assert row.complete is False
    assert row.note == "incomplete coverage: 4/10"
    assert row.n == 4
    assert row.n_expected == 10
    assert not row.usable()


def test_pass_rate_row_for_test_without_records_has_the_missing_metric_shape():
    missing_score = analyse.missing_metric("no harness records for test 't'")
    row = analyse.compute_harness_pass_rates({"t": missing_score}, {}, STATS_CFG)["t"]
    assert row == analyse.missing_metric("no harness records for test 't'")
    assert row.to_dict() == missing_score.to_dict()
    assert row.missing is True and row.value is None and row.wilson is None


def test_pass_rate_row_with_zero_cases_has_no_value_and_no_wilson():
    ts = harness_scorers.TestScores(test_name="t")
    score_row = analyse.MetricResult(
        value=0.0, n=0, n_expected=4, k=0, complete=False, note="incomplete coverage: 0/4"
    )
    row = analyse.compute_harness_pass_rates({"t": score_row}, {"t": ts}, STATS_CFG)["t"]
    assert row.value is None
    assert row.wilson is None
    assert row.complete is False
    assert not row.usable()


# ---------------------------------------------------------------------------
# Acceptance 3: rendered outputs show both rows, never a mean beside a Wilson interval
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def outputs():
    return analyse.generate_outputs(FIXTURE_RUN)


def test_report_shows_both_rows_and_no_mean_beside_wilson(outputs):
    report = outputs.files["REPORT.md"]
    assert "harness_pass_rate[schema_conformance]" in report
    score_lines = [ln for ln in report.splitlines() if ln.startswith("| harness_score[")]
    rate_lines = [ln for ln in report.splitlines() if ln.startswith("| harness_pass_rate[")]
    assert score_lines and rate_lines
    for ln in score_lines:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        assert cells[5] == "n/a", ln  # wilson column of a mean-score row
        assert cells[6] != "n/a", ln  # bootstrap column present
    for ln in rate_lines:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        assert cells[5] != "n/a", ln  # wilson column present
        assert cells[6] == "n/a", ln  # no bootstrap on a proportion


def test_summary_json_carries_pass_rate_rows_beside_score_rows(outputs):
    summary = json.loads(outputs.files["summary.json"])
    arm = summary["arms"]["c0"]
    assert arm["harness_pass_rate"].keys() == arm["harness_score"].keys()
    for name, score in arm["harness_score"].items():
        assert score["wilson"] is None
        assert score["bootstrap"] is not None
        rate = arm["harness_pass_rate"][name]
        assert rate["wilson"] is not None
        assert rate["k"] == score["k"]
        assert rate["value"] == rate["k"] / rate["n"]


def test_coverage_section_is_unchanged_by_pass_rate_rows(outputs):
    summary = json.loads(outputs.files["summary.json"])
    for arm_id, cov in summary["coverage"].items():
        if "harness" in cov:
            assert not any("pass_rate" in key for key in cov["harness"]), arm_id
