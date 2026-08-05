"""A report must show the count behind every score it prints.

The audit's finding: a robust pass and a thin one were typographically
identical in the report, which is why a documented 0.83 relationship-precision
near-miss cannot be re-examined today -- the report never recorded whether it
rested on 25 comparisons or 5. This does not fix past runs; it stops the next
one from being unfalsifiable.

`report.py` has no separable `render_test_row` function -- the per-test matrix
builds its cells inline inside `_write_per_test_matrix`. This suite extracts
and tests `_render_test_cell` (the matrix cell for one candidate/test pair)
and `_render_examined` (the shared None-vs-0 rendering rule), then verifies
the same rule reaches the two other places a `TestScores`/`CaseScore` score is
printed: the winners section and the failure drill-down.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from eval_harness.report import (  # noqa: E402  -- after sys.path insertion
    _render_examined,
    _render_test_cell,
    _write_failure_drill_down,
    _write_winners,
)
from eval_harness.scorers import (  # noqa: E402  -- after sys.path insertion
    CandidateScores,
    CaseScore,
    RunScores,
)
from eval_harness.scorers import TestScores as _TestScores  # noqa: E402

# Aliased: pytest's "Test*" collection pattern would otherwise try to collect
# this dataclass as a test class and warn, since it has an __init__.


def test_a_rendered_test_cell_shows_its_examined_count():
    """A score without its denominator is not falsifiable after the fact."""
    scores = _TestScores(test_name="schema_conformance")
    scores.case_scores.append(
        CaseScore(
            candidate_id="c1",
            test_name="schema_conformance",
            case_id="k1",
            iteration=1,
            passed=True,
            score=1.0,
            breakdown={},
            examined=12,
            error=None,
        )
    )
    scores.pass_count = 1

    rendered = _render_test_cell(scores)

    assert "12" in rendered


def test_an_unknown_examined_total_renders_distinctly_from_zero():
    """None must not print as 0 -- they are different facts."""
    scores = _TestScores(test_name="coherence")
    scores.case_scores.append(
        CaseScore(
            candidate_id="c1",
            test_name="coherence",
            case_id="k1",
            iteration=1,
            passed=True,
            score=1.0,
            breakdown={},
            examined=None,
            error=None,
        )
    )
    scores.pass_count = 1

    rendered = _render_test_cell(scores)

    assert "examined=0" not in rendered
    assert "examined=n/a" in rendered


def test_render_examined_pins_the_unknown_marker():
    """Pin the exact marker so a future edit can't silently drift it."""
    assert _render_examined(None) == "n/a"


def test_render_examined_pins_zero_distinctly_from_none():
    """0 is a real fact (looked, found nothing) and must stay a real 0."""
    assert _render_examined(0) == "0"


def test_winners_section_shows_examined_count():
    """The winners section re-prints TestScores.mean_score -- it needs the
    same count beside it, or a reader trusts an uncountable claim there.
    """
    ts = _TestScores(test_name="relationship_precision")
    ts.case_scores.append(
        CaseScore(
            candidate_id="c1",
            test_name="relationship_precision",
            case_id="k1",
            iteration=1,
            passed=True,
            score=0.83,
            breakdown={},
            examined=25,
            error=None,
        )
    )
    ts.pass_count = 1
    candidate_scores = CandidateScores(candidate_id="c1", per_test={"relationship_precision": ts})
    run_scores = RunScores(per_candidate={"c1": candidate_scores})

    lines: list[str] = []
    _write_winners(lines, run_scores, candidate_by_id={}, test_order=["relationship_precision"])

    rendered = "\n".join(lines)
    assert "examined=25" in rendered


def test_failure_drill_down_shows_examined_count():
    """Each failing case prints its own score -- and now its own count."""
    ts = _TestScores(test_name="relationship_precision")
    ts.case_scores.append(
        CaseScore(
            candidate_id="c1",
            test_name="relationship_precision",
            case_id="k1",
            iteration=1,
            passed=False,
            score=0.4,
            breakdown={},
            examined=5,
            error=None,
        )
    )
    ts.fail_count = 1
    candidate_scores = CandidateScores(candidate_id="c1", per_test={"relationship_precision": ts})
    run_scores = RunScores(per_candidate={"c1": candidate_scores})

    lines: list[str] = []
    _write_failure_drill_down(lines, run_scores, candidate_by_id={})

    rendered = "\n".join(lines)
    assert "examined=5" in rendered
