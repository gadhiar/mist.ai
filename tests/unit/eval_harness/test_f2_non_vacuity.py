"""F2 must not report a pass over a corpus it never read.

Verified by execution on 2026-08-05, before any change:

    r = score_run([], {})
    -> total_probes=0, entity_precision=1.0, rel_precision=1.0,
       typing_accuracy=1.0, negative_violations=0
    _gates_pass(r) -> True

Every gate passes on an empty gold corpus, including the join-integrity check
`matched_probes == total_probes` that exists to stop exactly this -- on an empty
corpus it is `0 == 0`. `--strict` therefore exits 0.

A mistyped path is NOT the trigger: `main():843-845` checks `args.gold.exists()`
and exits 2. The trigger is a gold file that exists and yields zero probes --
empty, blank-line-only, all-comment, truncated, or over-filtered. Verified by
execution: all three of those shapes return 0 probes and pass every gate.

This is the scorer whose numbers gated ontology v1.4.0 and closed MIS-124.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval_harness.score_extraction_run import (  # noqa: E402
    Report,
    _gates_pass,
    _render_count,
    _row,
    render_json,
    render_markdown,
    score_run,
)


def test_an_empty_gold_corpus_cannot_pass():
    report = score_run([], {})

    assert report.total_probes == 0
    assert _gates_pass(report) is False


def test_the_empty_case_is_distinguishable_from_a_real_failure():
    """A vacuous run must say WHY it failed, not just fail.

    Failing closed is necessary but not sufficient: an operator who points
    `--gold` at the wrong path needs to see that no probes were read, not a
    wall of 0.000 gate rows that looks like a catastrophic quality regression.
    """
    report = score_run([], {})

    assert report.vacuous is True


def test_a_gate_row_shows_the_count_behind_it():
    """A ratio without its denominator is not falsifiable after the fact."""
    row = _row("Entity precision", 0.833, 0.90, ">=", numerator=25, denominator=30)

    assert "25" in row and "30" in row


def test_a_gate_row_with_no_supplied_count_renders_n_a_not_zero():
    """No count passed in must not read as a real, counted zero."""
    row = _row("Valid-time accuracy", 1.0, None, "")

    assert "n/a" in row
    assert "0/0" not in row


def test_render_count_pins_the_unknown_marker():
    """Pin the exact marker so a future edit can't silently drift it."""
    assert _render_count(None, None) == "n/a"


def test_render_count_pins_a_real_zero_distinctly_from_unknown():
    """0/0 is a real fact (counted, found nothing) -- distinct from n/a."""
    assert _render_count(0, 0) == "0/0"


def test_precision_recall_denominator_properties_sum_tp_fp_fn():
    """The denominators behind entity/rel precision and recall are derived
    from the existing tp/fp/fn tallies, not separately tracked counters.
    """
    report = Report(entity_tp=11, entity_fp=2, entity_fn=3, rel_tp=17, rel_fp=4, rel_fn=5)

    assert report.entity_precision_denominator == 13
    assert report.entity_recall_denominator == 14
    assert report.rel_precision_denominator == 21
    assert report.rel_recall_denominator == 22


def test_every_gate_row_carries_its_own_denominator_in_render_markdown():
    """Wiring proof: each gate's rendered count must come from ITS OWN field.

    Every count below is distinct and non-overlapping across rows, so a count
    wired to the wrong Report field fails this test even though the wrong
    field is also an integer -- the bug class this task exists to catch.
    """
    report = Report(
        total_probes=1,
        matched_probes=1,
        entity_tp=11,
        entity_fp=2,
        entity_fn=3,
        rel_tp=17,
        rel_fp=4,
        rel_fn=5,
        typing_total=29,
        typing_ok=23,
        related_to_count=6,
        produced_rel_total=31,
        valid_time_total=19,
        valid_time_ok=7,
        specificity_numerator=3,
        specificity_denominator=8,
        negative_probes=9,
        negative_violations=1,
    )

    lines = render_markdown(report).splitlines()

    def _line_for(label: str) -> str:
        matches = [ln for ln in lines if label in ln]
        assert len(matches) == 1, f"expected exactly one row for {label!r}, found {matches}"
        return matches[0]

    assert "11/13" in _line_for("Entity precision")
    assert "11/14" in _line_for("Entity recall")
    assert "17/21" in _line_for("Relationship precision")
    assert "17/22" in _line_for("Relationship recall")
    assert "23/29" in _line_for("Typing accuracy")
    assert "6/31" in _line_for("RELATED_TO rate")
    assert "7/19" in _line_for("Valid-time accuracy")
    assert "3/8" in _line_for("Specificity")
    assert "1/9" in _line_for("Negative-control violations")


def test_vacuous_run_still_shows_counts_alongside_the_banner():
    """Task 1's vacuous banner and Task 2's per-gate counts must coexist: a
    reader who misses the banner should still see 0/0 counts as a second,
    independent signal that nothing was examined.
    """
    report = score_run([], {})

    md = render_markdown(report)

    assert "VACUOUS RUN" in md
    assert "0/0" in md


def test_render_json_includes_the_raw_counts_behind_every_ratio():
    """The JSON dump is the durable, machine-readable artifact -- a
    markdown-only fix does not help a reader working from the JSON file.
    """
    report = Report(
        total_probes=1,
        matched_probes=1,
        entity_tp=25,
        entity_fp=5,
        entity_fn=1,
        rel_tp=10,
        rel_fp=2,
        rel_fn=1,
        typing_total=12,
        typing_ok=11,
        related_to_count=1,
        produced_rel_total=12,
        valid_time_total=4,
        valid_time_ok=4,
        specificity_numerator=2,
        specificity_denominator=2,
        negative_probes=3,
        negative_violations=0,
    )

    payload = json.loads(render_json(report))

    assert payload["entity_tp"] == 25
    assert payload["entity_fp"] == 5
    assert payload["entity_fn"] == 1
    assert payload["rel_tp"] == 10
    assert payload["rel_fp"] == 2
    assert payload["rel_fn"] == 1
    assert payload["typing_ok"] == 11
    assert payload["typing_total"] == 12
    assert payload["related_to_count"] == 1
    assert payload["produced_rel_total"] == 12
    assert payload["valid_time_ok"] == 4
    assert payload["valid_time_total"] == 4
    assert payload["specificity_numerator"] == 2
    assert payload["specificity_denominator"] == 2
    assert payload["negative_probes"] == 3
    assert payload["negative_violations"] == 0
