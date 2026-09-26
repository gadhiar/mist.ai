"""Rule evaluation: R1-R7, both against hand-built synthetic metric bundles (for branches the
fixture does not exercise on its own -- the anchor demotion, an explicit F fail, R5's
needs-review-only branch, and a wholly-absent arm) and end to end against the fixture directory
under fixtures/analyse/run/ (see its README.md for what each arm was built to demonstrate).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from scripts.model_bench import analyse  # noqa: E402

FIXTURE_RUN = Path(__file__).resolve().parent / "fixtures" / "analyse" / "run"
RULES = analyse.load_decision_rules()


def test_correctness_expected_prompts_matches_the_probe():
    from scripts.model_bench.probes import correctness as correctness_probe

    assert RULES["constants"]["correctness_expected_prompts"] == len(correctness_probe.PROMPTS)


def _mr(value, *, bootstrap=None, complete=True, missing=False, n=10, n_expected=10):
    return analyse.MetricResult(
        value=value, n=n, n_expected=n_expected, bootstrap=bootstrap, complete=complete, missing=missing
    )


def _empty_arm_metrics(arm_id: str) -> analyse.ArmMetrics:
    return analyse.ArmMetrics(arm_id=arm_id)


def _run_metrics(arms: dict[str, analyse.ArmMetrics], *, lower=None, upper=None, total_mib=None) -> analyse.RunMetrics:
    return analyse.RunMetrics(
        arms=arms,
        raw_test_scores={a: {} for a in arms},
        raw_layout_rows={a: {} for a in arms},
        raw_correctness={a: {} for a in arms},
        voice_vram_lower_mib=lower or analyse.missing_metric("n/a"),
        voice_vram_upper_mib=upper or analyse.missing_metric("n/a"),
        total_mib=total_mib,
        manual={},
        arm_order=list(arms),
    )


# ---------------------------------------------------------------------------
# Synthetic: R2 anchor-below-threshold demotes S1/S2 to needs-review
# ---------------------------------------------------------------------------


def test_r2_anchor_below_threshold_forces_needs_review():
    cand = _empty_arm_metrics("test-cand")
    cand.harness_score = {"schema_conformance_json_object": _mr(0.9), "tool_selection": _mr(0.95)}
    cand.decode_tps = _mr(40.0)
    cand.layout_acc = {"screen": _mr(0.80, bootstrap=(0.75, 0.85))}
    cand.layout_p95_wall_ms = {"screen": _mr(5000.0)}
    cand.arm_peak_mib = _mr(1000.0)

    prod = _empty_arm_metrics("c0-prod")
    prod.harness_score = {"schema_conformance_json_object": _mr(0.5), "tool_selection": _mr(0.5)}

    metrics = _run_metrics(
        {"test-cand": cand, "c0-prod": prod},
        lower=_mr(1000.0), upper=_mr(2000.0), total_mib=20000.0,
    )
    result = analyse.evaluate_r2("switch_to_test", "test-cand", "test-cand-think512", metrics, RULES)
    s1 = next(c for c in result.clauses if c.id == "S1")
    s2 = next(c for c in result.clauses if c.id == "S2")
    assert s1.verdict == "needs-review"
    assert s1.note == "anchor_below_threshold"
    assert s1.extra["anchor_value"] == 0.5
    assert s2.verdict == "needs-review"
    assert s2.note == "anchor_below_threshold"
    assert result.verdict == "needs-review"


def test_r2_missing_when_anchor_arm_absent():
    cand = _empty_arm_metrics("test-cand")
    cand.harness_score = {"schema_conformance_json_object": _mr(0.9)}
    metrics = _run_metrics({"test-cand": cand}, lower=_mr(1000.0), upper=_mr(2000.0), total_mib=20000.0)
    result = analyse.evaluate_r2("switch_to_test", "test-cand", "test-cand-think512", metrics, RULES)
    s1 = next(c for c in result.clauses if c.id == "S1")
    assert s1.verdict == "missing"


def test_r2_fully_missing_when_candidate_arm_absent():
    metrics = _run_metrics({}, lower=_mr(3000.0), upper=_mr(4000.0), total_mib=20000.0)
    result = analyse.evaluate_r2("switch_to_ghost", "ghost-arm", "ghost-arm-think512", metrics, RULES)
    assert result.verdict == "missing"
    for clause in result.clauses:
        assert clause.verdict == "missing"


def test_f_clause_fail_when_even_the_lower_estimate_does_not_fit():
    cand = _empty_arm_metrics("test-cand")
    cand.arm_peak_mib = _mr(15000.0)
    metrics = _run_metrics({"test-cand": cand}, lower=_mr(1000.0), upper=_mr(2000.0), total_mib=16000.0)
    clause = analyse._f_clause("test-cand", metrics, RULES["constants"]["margin_mib"])
    assert clause.verdict == "fail"


def test_f_clause_pass_when_upper_estimate_fits():
    cand = _empty_arm_metrics("test-cand")
    cand.arm_peak_mib = _mr(1000.0)
    metrics = _run_metrics({"test-cand": cand}, lower=_mr(500.0), upper=_mr(1000.0), total_mib=10000.0)
    clause = analyse._f_clause("test-cand", metrics, RULES["constants"]["margin_mib"])
    assert clause.verdict == "pass"


def test_f_clause_needs_review_between_lower_and_upper():
    cand = _empty_arm_metrics("test-cand")
    cand.arm_peak_mib = _mr(8500.0)
    metrics = _run_metrics({"test-cand": cand}, lower=_mr(2500.0), upper=_mr(3700.0), total_mib=12288.0)
    clause = analyse._f_clause("test-cand", metrics, RULES["constants"]["margin_mib"])
    assert clause.verdict == "needs-review"


def test_f_clause_missing_when_voice_metrics_absent():
    cand = _empty_arm_metrics("test-cand")
    cand.arm_peak_mib = _mr(1000.0)
    metrics = _run_metrics({"test-cand": cand}, total_mib=10000.0)
    clause = analyse._f_clause("test-cand", metrics, RULES["constants"]["margin_mib"])
    assert clause.verdict == "missing"


# ---------------------------------------------------------------------------
# Synthetic: exploratory X1/X3 (NOT pre-registered, plan v2 2026-09-25) -- hand-built,
# a single fixture run cannot exercise X1's pass/fail/missing branches simultaneously
# since X1 evaluates exactly one arm (c1-1024). See fixtures/analyse/run/README.md.
# ---------------------------------------------------------------------------


def _exploratory_rule_doc(rule_id: str, label: str) -> dict:
    matches = [
        r for r in RULES["exploratory_rules"]["rules"] if r["id"] == rule_id and r["label"] == label
    ]
    assert len(matches) == 1, f"expected exactly one exploratory rule {rule_id}/{label}"
    return matches[0]


def test_x1_pass_when_layout_p_and_d_all_clear():
    arm = _empty_arm_metrics("c1-1024")
    arm.layout_acc = {"screen": _mr(0.90, bootstrap=(0.85, 0.95))}
    arm.layout_p95_wall_ms = {"screen": _mr(5000.0)}
    arm.decode_tps = _mr(40.0)
    metrics = _run_metrics({"c1-1024": arm})
    result = analyse.evaluate_x1(metrics, RULES, _exploratory_rule_doc("X1", "c1_1024_budget"))
    assert result.pre_registered is False
    assert result.basis
    assert result.verdict == "pass"
    for c in result.clauses:
        assert c.verdict == "pass", f"{c.id}: {c}"


def test_x1_fail_when_decode_tps_too_low():
    arm = _empty_arm_metrics("c1-1024")
    arm.layout_acc = {"screen": _mr(0.90, bootstrap=(0.85, 0.95))}
    arm.layout_p95_wall_ms = {"screen": _mr(5000.0)}
    arm.decode_tps = _mr(10.0)  # below R2's D threshold (35)
    metrics = _run_metrics({"c1-1024": arm})
    result = analyse.evaluate_x1(metrics, RULES, _exploratory_rule_doc("X1", "c1_1024_budget"))
    d_clause = next(c for c in result.clauses if c.id == "D")
    assert d_clause.verdict == "fail"
    assert result.verdict == "fail"


def test_x1_missing_when_arm_absent():
    metrics = _run_metrics({})
    result = analyse.evaluate_x1(metrics, RULES, _exploratory_rule_doc("X1", "c1_1024_budget"))
    assert result.verdict == "missing"
    for c in result.clauses:
        assert c.verdict == "missing"


def test_x3_f_sep_passes_where_v1_f_fails_outright():
    # A peak large enough that v1's F fails even the LOWER voice estimate (not merely
    # needs-review), so F_sep passing here is a genuine "F failed, F_sep did not" case.
    cand = _empty_arm_metrics("c2")
    cand.harness_score = {"schema_conformance_json_object": _mr(0.9), "tool_selection": _mr(0.95)}
    cand.decode_tps = _mr(40.0)
    cand.layout_acc = {"screen": _mr(0.80, bootstrap=(0.75, 0.85))}
    cand.layout_p95_wall_ms = {"screen": _mr(5000.0)}
    cand.arm_peak_mib = _mr(15000.0)

    anchor = _empty_arm_metrics("c0-prod")
    anchor.harness_score = {"schema_conformance_json_object": _mr(0.9), "tool_selection": _mr(0.95)}

    metrics = _run_metrics(
        {"c2": cand, "c0-prod": anchor}, lower=_mr(1000.0), upper=_mr(2000.0), total_mib=16000.0,
    )

    f_v1 = analyse._f_clause("c2", metrics, RULES["constants"]["margin_mib"])
    assert f_v1.verdict == "fail"  # 15000 + 1000 (lower) + 512 > 16000

    rule_doc = _exploratory_rule_doc("X3", "switch_to_c2_sepvoice")
    result = analyse.evaluate_x3(rule_doc, metrics, RULES)
    assert result.pre_registered is False
    f_sep = next(c for c in result.clauses if c.id == "F_sep")
    assert f_sep.verdict == "pass"  # 15000 + 512 <= 16000, voice excluded
    for c in result.clauses:
        assert c.verdict == "pass", f"{c.id}: {c}"
    assert result.verdict == "pass"
    assert result.info["compares_against"] == "v1 R2/switch_to_c2"


def test_x3_missing_when_candidate_arm_absent():
    metrics = _run_metrics({}, lower=_mr(1000.0), upper=_mr(2000.0), total_mib=16000.0)
    rule_doc = _exploratory_rule_doc("X3", "switch_to_c4_sepvoice")
    result = analyse.evaluate_x3(rule_doc, metrics, RULES)
    assert result.verdict == "missing"


# ---------------------------------------------------------------------------
# Synthetic: R5 needs-review-only branch (upper crosses, lower does not)
# ---------------------------------------------------------------------------


def test_r5_needs_review_when_only_upper_crosses():
    metrics = _run_metrics({}, lower=_mr(1000.0), upper=_mr(3000.0), total_mib=12288.0)
    result = analyse.evaluate_r5(metrics, RULES, r2_results=[])
    assert result.verdict == "needs-review"


def test_r5_not_triggered_when_neither_crosses():
    metrics = _run_metrics({}, lower=_mr(500.0), upper=_mr(1000.0), total_mib=12288.0)
    result = analyse.evaluate_r5(metrics, RULES, r2_results=[])
    assert result.verdict == "not-triggered"


def test_r5_missing_when_only_the_lower_bound_is_usable():
    # A lower bound below the trigger cannot rule out an upper bound at needs-review level.
    metrics = _run_metrics({}, lower=_mr(1000.0), total_mib=12288.0)
    result = analyse.evaluate_r5(metrics, RULES, r2_results=[])
    assert result.verdict == "missing"


def test_r5_missing_when_only_the_upper_bound_is_usable():
    metrics = _run_metrics({}, upper=_mr(4096.0), total_mib=12288.0)
    result = analyse.evaluate_r5(metrics, RULES, r2_results=[])
    assert result.verdict == "missing"


def test_r5_missing_when_both_voice_metrics_absent():
    metrics = _run_metrics({}, total_mib=12288.0)
    result = analyse.evaluate_r5(metrics, RULES, r2_results=[])
    assert result.verdict == "missing"


def _clause(clause_id: str, verdict: str) -> analyse.ClauseResult:
    return analyse.ClauseResult(
        id=clause_id, metric="m", arm=None, value=None, threshold=None, op=None, verdict=verdict, margin="n/a",
    )


def test_r5_missing_not_not_triggered_when_lower_triggers_and_r2_incomplete():
    # Lower bound clears the trigger threshold, and no R2 rule can be confirmed as
    # matching ("all-pass-except-F, F bad") because its S1 clause reads "missing" --
    # that candidate might have matched had the data been complete, so this must not
    # silently read as a clean not-triggered.
    r2_incomplete = analyse.RuleResult(
        id="R2", kind="gate", label="switch_to_incomplete",
        question="q", verdict="missing",
        clauses=[
            _clause("L", "pass"), _clause("S1", "missing"), _clause("S2", "pass"),
            _clause("D", "pass"), _clause("P", "pass"), _clause("F", "fail"),
        ],
        info={"candidate": "incomplete-cand"},
    )
    metrics = _run_metrics({}, lower=_mr(4096.0), upper=_mr(4096.0), total_mib=12288.0)
    result = analyse.evaluate_r5(metrics, RULES, r2_results=[r2_incomplete])
    assert result.verdict == "missing"
    assert result.verdict != "not-triggered"


def test_r5_not_triggered_stays_clean_when_r2_fully_evaluated_and_none_match():
    # Same lower-triggers-and-nothing-matches shape, but every R2 clause is
    # conclusively pass/fail (no "missing") -- this IS a clean not-triggered.
    r2_clean = analyse.RuleResult(
        id="R2", kind="gate", label="switch_to_clean",
        question="q", verdict="fail",
        clauses=[
            _clause("L", "fail"), _clause("S1", "pass"), _clause("S2", "pass"),
            _clause("D", "pass"), _clause("P", "pass"), _clause("F", "pass"),
        ],
        info={"candidate": "clean-cand"},
    )
    metrics = _run_metrics({}, lower=_mr(4096.0), upper=_mr(4096.0), total_mib=12288.0)
    result = analyse.evaluate_r5(metrics, RULES, r2_results=[r2_clean])
    assert result.verdict == "not-triggered"


# ---------------------------------------------------------------------------
# Synthetic: R6 determinism clauses in isolation
# ---------------------------------------------------------------------------


EXPECTED_PROMPTS = RULES["constants"]["correctness_expected_prompts"]


def _correctness_rows(n: int, *, mismatch_pid: str | None = None, error_pid: str | None = None) -> list[dict]:
    """Build n synthetic correctness.jsonl rows, p01..p0n, each with distinct tokens.

    `mismatch_pid` gives that one row different tokens from the "canonical" set (used to
    build a second, comparison row list); `error_pid` marks that one row as errored.
    """
    rows = []
    for i in range(1, n + 1):
        pid = f"p{i:02d}"
        if pid == error_pid:
            rows.append({"prompt_id": pid, "tokens": None, "error": "server returned 500"})
            continue
        tokens = [i, i + 1] if pid != mismatch_pid else [999, 998]
        rows.append({"prompt_id": pid, "tokens": tokens, "error": None})
    return rows


def test_r6_determinism_clause_missing_when_files_absent():
    clause = analyse._determinism_clause("x", None, None, "note", EXPECTED_PROMPTS)
    assert clause.verdict == "missing"


def test_r6_determinism_clause_fail_on_mismatch():
    a = _correctness_rows(EXPECTED_PROMPTS)
    b = _correctness_rows(EXPECTED_PROMPTS, mismatch_pid="p01")
    clause = analyse._determinism_clause("x", a, b, "note", EXPECTED_PROMPTS)
    assert clause.verdict == "fail"


def test_r6_determinism_clause_pass_on_match():
    a = _correctness_rows(EXPECTED_PROMPTS)
    b = _correctness_rows(EXPECTED_PROMPTS)
    clause = analyse._determinism_clause("x", a, b, "note", EXPECTED_PROMPTS)
    assert clause.verdict == "pass"


def test_r6_determinism_clause_missing_when_prompt_count_wrong():
    a = _correctness_rows(EXPECTED_PROMPTS - 1)
    b = _correctness_rows(EXPECTED_PROMPTS)
    clause = analyse._determinism_clause("x", a, b, "note", EXPECTED_PROMPTS)
    assert clause.verdict == "missing"


def test_r6_determinism_clause_missing_when_a_prompt_id_is_duplicated():
    # Right row count, incomplete prompt set: p19 twice and no p20, identically in both reps.
    a = _correctness_rows(EXPECTED_PROMPTS)
    b = _correctness_rows(EXPECTED_PROMPTS)
    for rows in (a, b):
        rows[-1] = dict(rows[-2])
    clause = analyse._determinism_clause("x", a, b, "note", EXPECTED_PROMPTS)
    assert clause.verdict == "missing"


def test_r6_determinism_clause_missing_when_prompt_id_sets_differ():
    a = _correctness_rows(EXPECTED_PROMPTS)
    b = _correctness_rows(EXPECTED_PROMPTS)
    b[0]["prompt_id"] = "pXX"
    clause = analyse._determinism_clause("x", a, b, "note", EXPECTED_PROMPTS)
    assert clause.verdict == "missing"


def test_r6_determinism_clause_missing_not_pass_when_p20_errored_in_both_reps():
    # Regression for the bug where an errored prompt, present in BOTH files, was
    # simply dropped from the comparison and the clause still read as "pass" because
    # the remaining prompts happened to agree.
    a = _correctness_rows(EXPECTED_PROMPTS, error_pid="p20")
    b = _correctness_rows(EXPECTED_PROMPTS, error_pid="p20")
    clause = analyse._determinism_clause("x", a, b, "note", EXPECTED_PROMPTS)
    assert clause.verdict == "missing"
    assert clause.verdict != "pass"


# ---------------------------------------------------------------------------
# End to end against the hand-built fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def summary() -> dict:
    outputs = analyse.generate_outputs(FIXTURE_RUN)
    return json.loads(outputs.files["summary.json"])


def _rule(summary_doc: dict, label: str) -> dict:
    matches = [r for r in summary_doc["rules"] if r["label"] == label]
    assert len(matches) == 1, f"expected exactly one rule labeled {label!r}, found {len(matches)}"
    return matches[0]


def test_r1_keep_e4b_budget_passes_cleanly(summary):
    r1 = _rule(summary, "keep_e4b_budget")
    assert r1["verdict"] == "pass"
    clause = r1["clauses"][0]
    assert clause["value"] == pytest.approx(65 / 72, abs=1e-5)
    assert clause["verdict"] == "pass"


def test_r2_switch_to_c2_passes_via_thinking_branch(summary):
    r2 = _rule(summary, "switch_to_c2")
    l_clause = next(c for c in r2["clauses"] if c["id"] == "L")
    assert l_clause["verdict"] == "pass"
    assert l_clause["extra"]["branch"] == "on"
    assert l_clause["arm"] == "c2-think512"
    assert r2["verdict"] == "pass"
    for clause in r2["clauses"]:
        assert clause["verdict"] == "pass", f"{clause['id']}: {clause}"


def test_r2_switch_to_c3_needs_review_on_f_only(summary):
    r3 = _rule(summary, "switch_to_c3")
    f_clause = next(c for c in r3["clauses"] if c["id"] == "F")
    assert f_clause["verdict"] == "needs-review"
    other_clauses = [c for c in r3["clauses"] if c["id"] != "F"]
    for clause in other_clauses:
        assert clause["verdict"] == "pass", f"{clause['id']}: {clause}"
    assert r3["verdict"] == "needs-review"


def test_r2_switch_to_c4_entirely_missing(summary):
    r4 = _rule(summary, "switch_to_c4")
    assert r4["verdict"] == "missing"
    for clause in r4["clauses"]:
        assert clause["verdict"] == "missing"


def test_r3_drop_moe_triggered(summary):
    r3 = _rule(summary, "drop_moe")
    assert r3["verdict"] == "triggered"
    clause = r3["clauses"][0]
    assert clause["value"] == pytest.approx(2 / 12, abs=1e-5)


def test_r4_thinking_is_lever_triggered(summary):
    r4 = _rule(summary, "thinking_is_lever")
    assert r4["verdict"] == "triggered"
    clause = r4["clauses"][0]
    assert clause["value"] == pytest.approx(30 / 72, abs=1e-5)


def test_r5_gtx1070_moves_up_triggered(summary):
    r5 = _rule(summary, "gtx1070_moves_up")
    assert r5["verdict"] == "triggered"


def test_r6_tuning_gate_a1_passes(summary):
    a1 = _rule(summary, "tuning_gate_a1")
    assert a1["verdict"] == "pass"
    for clause in a1["clauses"]:
        assert clause["verdict"] == "pass"


def test_r6_tuning_gate_a2_missing_manual(summary):
    a2 = _rule(summary, "tuning_gate_a2")
    assert a2["verdict"] == "missing"
    manual_clause = next(c for c in a2["clauses"] if c["id"] == "manual_clean")
    assert manual_clause["verdict"] == "missing"
    other = [c for c in a2["clauses"] if c["id"] != "manual_clean"]
    for clause in other:
        assert clause["verdict"] == "pass"


def test_r6_tuning_gate_a3_fails_determinism(summary):
    a3 = _rule(summary, "tuning_gate_a3")
    assert a3["verdict"] == "fail"
    tuned_clause = next(c for c in a3["clauses"] if c["id"] == "tuned_matches_base")
    assert tuned_clause["verdict"] == "fail"


def test_r7_build_effect_is_informational_with_deltas(summary):
    r7 = _rule(summary, "build_effect")
    assert r7["verdict"] == "n/a"
    deltas = r7["info"]["deltas"]
    assert "layout_acc" in deltas
    assert "harness_score_schema_conformance" in deltas
    sc_delta = deltas["harness_score_schema_conformance"]
    if not sc_delta.get("missing"):
        assert sc_delta["delta"] == pytest.approx(0.0, abs=1e-5)


def test_finalist_pass_supersedes_screen_for_c3(summary):
    c3 = summary["arms"]["c3"]
    screen = c3["layout_acc"]["screen"]
    finalist = c3["layout_acc"]["finalist"]
    assert screen["complete"] is True
    assert finalist["complete"] is True
    assert finalist["value"] == pytest.approx(180 / 216, abs=1e-5)
    # The L clause on switch_to_c3 must report the finalist pass, not the screen pass.
    r_c3 = _rule(summary, "switch_to_c3")
    l_clause = next(c for c in r_c3["clauses"] if c["id"] == "L")
    assert l_clause["extra"]["pass_used"] == "finalist"


def test_incomplete_layout_coverage_reported_but_not_used(summary):
    c1_256 = summary["arms"]["c1-256"]
    screen = c1_256["layout_acc"]["screen"]
    assert screen["complete"] is False
    assert screen["n"] == 40
    assert screen["n_expected"] == 72
    # c1-256 feeds no clause: R1 only names c1-512.
    r1 = _rule(summary, "keep_e4b_budget")
    assert all(c["arm"] != "c1-256" for c in r1["clauses"])
    assert "c1_256_layout_acc" in r1["info"]


def test_decision_rules_sha_mismatch_warning_for_c1_256(summary):
    assert any("c1-256" in w for w in summary["decision_rules_sha256_warnings"])


def test_missing_inputs_lists_switch_to_c4(summary):
    assert any("switch_to_c4" in m for m in summary["missing_inputs"])


# ---------------------------------------------------------------------------
# Exploratory rules (X1/X2/X3), end to end against the fixture directory --
# plan v2 (2026-09-25). See fixtures/analyse/run/README.md for c1-1024/c0-ctx64k.
# ---------------------------------------------------------------------------


def _exploratory(summary_doc: dict, rule_id: str, label: str) -> dict:
    matches = [
        r for r in summary_doc["exploratory_rules"] if r["id"] == rule_id and r["label"] == label
    ]
    assert len(matches) == 1, f"expected exactly one exploratory rule {rule_id}/{label}"
    return matches[0]


def test_v1_rules_key_excludes_exploratory_rules(summary):
    v1_ids = {r["id"] for r in summary["rules"]}
    assert v1_ids == {"R1", "R2", "R3", "R4", "R5", "R6", "R7"}
    for r in summary["rules"]:
        assert "pre_registered" not in r
        assert "basis" not in r


def test_x1_c1_1024_budget_passes_from_fixture_data(summary):
    x1 = _exploratory(summary, "X1", "c1_1024_budget")
    assert x1["pre_registered"] is False
    assert x1["verdict"] == "pass"
    for clause in x1["clauses"]:
        assert clause["verdict"] == "pass", f"{clause['id']}: {clause}"


def test_sha_info_not_warn_for_c1_1024_superseded_sha(summary):
    # c1-1024's fixture meta.json deliberately stores the v1 sha, which the current
    # decision_rules.json lists in its own `supersedes` -- an [INFO], never a [WARN].
    assert any("c1-1024" in i for i in summary["decision_rules_sha256_info"])
    assert not any("c1-1024" in w for w in summary["decision_rules_sha256_warnings"])


def test_x2_context_arms_report_rows(summary):
    x2 = _exploratory(summary, "X2", "context_arms_report")
    assert x2["verdict"] == "n/a"
    assert x2["clauses"] == []
    arms = x2["info"]["arms"]
    assert set(arms) == {"c0-ctx64k", "c0-ctx128k", "c0-ctx128k-q4kv"}
    # c0-ctx64k has real fixture data.
    ctx64k = arms["c0-ctx64k"]
    assert ctx64k["present"] is True
    assert ctx64k["tokens_vs_c0"] == "expected-identical-unverified"
    assert ctx64k["correctness_tokens_vs_c0"] == "identical"
    assert ctx64k["decode_tps"]["value"] == pytest.approx(39.0, abs=1e-5)
    assert "65000" in ctx64k["ttft_ms"]
    # The other two context arms have no fixture directory at all.
    assert arms["c0-ctx128k"]["present"] is False
    assert arms["c0-ctx128k-q4kv"]["present"] is False


def test_x3_switch_to_c3_sepvoice_passes_where_v1_f_was_needs_review(summary):
    r_c3 = _rule(summary, "switch_to_c3")
    f_v1 = next(c for c in r_c3["clauses"] if c["id"] == "F")
    assert f_v1["verdict"] == "needs-review"

    x3 = _exploratory(summary, "X3", "switch_to_c3_sepvoice")
    f_sep = next(c for c in x3["clauses"] if c["id"] == "F_sep")
    assert f_sep["verdict"] == "pass"
    assert x3["info"]["compares_against"] == "v1 R2/switch_to_c3"


def test_x3_switch_to_c4_sepvoice_entirely_missing(summary):
    x3 = _exploratory(summary, "X3", "switch_to_c4_sepvoice")
    assert x3["verdict"] == "missing"
    for clause in x3["clauses"]:
        assert clause["verdict"] == "missing"
