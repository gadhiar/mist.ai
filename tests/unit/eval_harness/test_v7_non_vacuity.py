"""V7 must not report a pass over negative controls it never examined.

Verified by execution on 2026-08-05, before any change: 20 positive probes that
all join and fire correctly, plus 5 negative-control probes that never join
(no matching debug-JSONL record -- verdict "missing" for all 5):

    precision 1.0, recall 1.0, false_positives 0, true_negatives 0, missing 5
    acceptance_pass() -> True

`false_positives <= NEGATIVE_FALSE_POSITIVE_LIMIT` (0 <= 0) is trivially true
whether every negative was adjudicated clean or every negative never joined at
all -- `false_positives` alone cannot distinguish "checked, found nothing
wrong" from "never checked." `render_markdown` reproduces the exact ambiguity
the audit named: "False positives on negatives: 0 / 5" and
"[PASS] False positives on negatives <= 0" render identically in both cases,
and **Verdict:** PASS follows from `acceptance_pass()` alone -- the report's
"Missing (no debug record for utterance): 5" line and the per-outcome
[MISSING] entries under "Failures and Misses" are present elsewhere in the
same document, but neither feeds `acceptance_pass()`, so `--strict` exits 0.

This is the scorer whose 0.650 -> 0.950 jump was independently confirmed real
(commit 3784a36, bounded at +1 TP by `acceptable_tools` touching exactly one
probe) -- the defect this test targets is narrower: the negative-control rule
specifically, not the scorer's overall permissiveness.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval_harness.score_v7_probe_run import (  # noqa: E402
    ProbeOutcome,
    TurnObservation,
    V7Probe,
    V7Report,
    render_json,
    render_markdown,
    score_run,
)


def _positive_outcome(n: int) -> ProbeOutcome:
    """One clean true-positive outcome, distinct utterance per call."""
    probe = V7Probe(
        tag=f"v7-{n:02d}-direct-tech-recall",
        utterance=f"positive utterance {n}",
        expected_tool="query_knowledge_graph",
        rationale=None,
        acceptable_tools=("query_knowledge_graph",),
    )
    return ProbeOutcome(probe=probe, observation=None, verdict="tp")


def _missing_negative_outcome(n: int) -> ProbeOutcome:
    """A negative-control probe with no joined observation -- verdict "missing"."""
    probe = V7Probe(
        tag=f"v7-{n:02d}-neg-general-knowledge",
        utterance=f"negative utterance {n}",
        expected_tool=None,
        rationale=None,
        acceptable_tools=(),
    )
    return ProbeOutcome(probe=probe, observation=None, verdict="missing")


def _clean_negative_outcome(n: int) -> ProbeOutcome:
    """A negative-control probe that joined and correctly stayed silent (tn)."""
    probe = V7Probe(
        tag=f"v7-{n:02d}-neg-general-knowledge",
        utterance=f"negative utterance {n}",
        expected_tool=None,
        rationale=None,
        acceptable_tools=(),
    )
    return ProbeOutcome(probe=probe, observation=None, verdict="tn")


def _build_report(*, positives: int, missing_negatives: int, clean_negatives: int) -> V7Report:
    outcomes = [_positive_outcome(i) for i in range(positives)]
    outcomes += [_missing_negative_outcome(i) for i in range(missing_negatives)]
    outcomes += [_clean_negative_outcome(i) for i in range(clean_negatives)]
    return V7Report(outcomes=outcomes)


def test_zero_examined_negatives_cannot_pass():
    """All 5 negatives missing, 20 positives clean -- must FAIL, not PASS."""
    report = _build_report(positives=20, missing_negatives=5, clean_negatives=0)

    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.false_positives == 0
    assert report.acceptance_pass() is False


def test_negatives_vacuous_is_the_reason_the_zero_examined_case_fails():
    """A vacuous negative-control block must say WHY it failed.

    Failing closed is necessary but not sufficient: an operator whose debug
    JSONL never captured the negative-control turns needs to see that
    nothing was examined, not a `false_positives <= 0` row that looks
    identical to a genuinely clean run.
    """
    report = _build_report(positives=20, missing_negatives=5, clean_negatives=0)

    assert report.negatives_vacuous is True
    assert report.examined_negatives == 0


def test_at_least_one_examined_negative_is_not_vacuous():
    """One real tn/fp adjudication is enough to clear the vacuity gate --
    the requirement is that the block ran at all, not that every negative
    joined.
    """
    report = _build_report(positives=20, missing_negatives=4, clean_negatives=1)

    assert report.negatives_vacuous is False
    assert report.examined_negatives == 1


def test_a_real_clean_run_with_all_negatives_examined_still_passes():
    """Regression guard: a normal run where every negative joined and stayed
    silent must behave exactly as before the fix -- PASS, not a new FAIL.
    """
    report = _build_report(positives=20, missing_negatives=0, clean_negatives=5)

    assert report.negatives_vacuous is False
    assert report.examined_negatives == 5
    assert report.false_positives == 0
    assert report.acceptance_pass() is True


def test_a_real_clean_run_with_a_genuine_false_positive_still_fails():
    """Regression guard: a real false positive on an examined negative must
    still fail acceptance -- the fix must not paper over a genuine violation.
    """
    outcomes = [_positive_outcome(i) for i in range(20)]
    outcomes += [_clean_negative_outcome(i) for i in range(4)]
    fp_probe = V7Probe(
        tag="v7-99-neg-x",
        utterance="fp utterance",
        expected_tool=None,
        rationale=None,
        acceptable_tools=(),
    )
    outcomes.append(ProbeOutcome(probe=fp_probe, observation=None, verdict="fp"))
    report = V7Report(outcomes=outcomes)

    assert report.negatives_vacuous is False
    assert report.examined_negatives == 5
    assert report.false_positives == 1
    assert report.acceptance_pass() is False


def test_the_examined_count_is_visible_beside_the_fp_figure_in_markdown():
    """The count must appear in the output beside the FP figure, so
    'FP 0/5' and '0 negatives examined' are distinguishable to a reader.
    """
    report = _build_report(positives=20, missing_negatives=5, clean_negatives=0)

    md = render_markdown(report)
    fp_line = next(ln for ln in md.splitlines() if ln.startswith("- False positives on negatives"))

    assert "0 / 5" in fp_line
    assert "0 examined" in fp_line


def test_markdown_shows_fail_verdict_and_names_the_vacuous_gate():
    report = _build_report(positives=20, missing_negatives=5, clean_negatives=0)

    md = render_markdown(report)

    assert "Verdict:** FAIL" in md
    assert "[FAIL]" in md
    assert "examined" in md.lower()


def test_markdown_still_shows_pass_when_negatives_are_genuinely_examined():
    """Wiring proof: the acceptance-criteria row for the vacuity gate must
    itself read PASS on a real clean run, not just not-block it.
    """
    report = _build_report(positives=20, missing_negatives=0, clean_negatives=5)

    md = render_markdown(report)

    assert "Verdict:** PASS" in md
    assert "5 examined" in md


def test_render_json_includes_examined_negatives_and_the_vacuous_flag():
    """The JSON dump is the durable, machine-readable artifact -- a
    markdown-only fix does not help a reader working from the JSON file.
    """
    report = _build_report(positives=20, missing_negatives=5, clean_negatives=0)

    payload = json.loads(render_json(report))

    assert payload["totals"]["examined_negatives"] == 0
    assert payload["acceptance"]["negatives_vacuous"] is True
    assert payload["acceptance"]["passed"] is False


def test_render_json_reflects_a_non_vacuous_pass():
    report = _build_report(positives=20, missing_negatives=0, clean_negatives=5)

    payload = json.loads(render_json(report))

    assert payload["totals"]["examined_negatives"] == 5
    assert payload["acceptance"]["negatives_vacuous"] is False
    assert payload["acceptance"]["passed"] is True


def test_score_run_end_to_end_reproduces_and_then_fixes_the_audited_scenario():
    """End-to-end join proof, not a hand-built V7Report: 20 positives that
    join and fire correctly, 5 negatives that are simply never in the debug
    JSONL (the realistic trigger -- a truncated run, a session-id mismatch,
    or negatives that were never actually asked).
    """
    probes = []
    index = {}
    for i in range(20):
        probe = V7Probe(
            tag=f"v7-{i:02d}-direct-tech-recall",
            utterance=f"positive utterance {i}",
            expected_tool="query_knowledge_graph",
            rationale=None,
            acceptable_tools=("query_knowledge_graph",),
        )
        probes.append(probe)
        index[probe.utterance] = TurnObservation(
            utterance=probe.utterance,
            session_id="s",
            event_id="e",
            tool_calls=("query_knowledge_graph",),
        )
    for i in range(20, 25):
        probes.append(
            V7Probe(
                tag=f"v7-{i:02d}-neg-general-knowledge",
                utterance=f"negative utterance {i}",
                expected_tool=None,
                rationale=None,
                acceptable_tools=(),
            )
        )
        # deliberately absent from `index` -- never joined

    report = score_run(probes, index)

    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.false_positives == 0
    assert report.missing == 5
    assert report.negatives_vacuous is True
    assert report.acceptance_pass() is False
