"""A scorer must not report a pass over nothing.

Verified on 2026-08-04, by execution rather than inspection:

    score_schema_conformance({'response_content': '{"entities": [], "relationships": []}'}, {})
      -> (True, 1.0)
    score_personality({'response_content': ''}, {})
      -> (True, 1.0)

Five structural checks all pass on an empty extraction because each is a
`len(bad_things) == 0` test and the loop that populates `bad_things` never
runs. A candidate that returned an empty extraction for every probe would have
scored a flawless 1.0 on schema conformance.

The same scorer returns (False, 0.8) for one bad entity type, so it works on
real input. The hole is specifically the empty case, which is why the fix is a
floor rather than a rewrite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from eval_harness.scorers import (  # noqa: E402  -- after sys.path insertion
    SCORER_REGISTRY,
    ScoreOutcome,
    enforce_non_vacuity,
    score_personality,
    score_schema_conformance,
)


def test_zero_examined_cannot_pass():
    vacuous = ScoreOutcome(passed=True, score=1.0, breakdown={}, examined=0)

    guarded = enforce_non_vacuity(vacuous)

    assert guarded.passed is False
    assert guarded.score == 0.0
    assert guarded.breakdown["vacuous"] is True


def test_a_real_pass_is_untouched():
    real = ScoreOutcome(passed=True, score=1.0, breakdown={"json_valid": True}, examined=7)

    guarded = enforce_non_vacuity(real)

    assert guarded.passed is True
    assert guarded.score == 1.0
    assert "vacuous" not in guarded.breakdown
    assert guarded.examined == 7


def test_a_real_failure_is_untouched():
    failure = ScoreOutcome(passed=False, score=0.8, breakdown={}, examined=3)

    assert enforce_non_vacuity(failure) == failure


def test_unknown_examined_is_not_treated_as_zero():
    """`None` means the scorer cannot distinguish empty from broken.

    That is a declared gap, not a hard failure -- forcing it to fail would make
    every scorer that cannot count its own input permanently red, which trains
    people to ignore the signal. It stays visible in the breakdown instead.
    """
    unknown = ScoreOutcome(passed=True, score=1.0, breakdown={}, examined=None)

    guarded = enforce_non_vacuity(unknown)

    assert guarded.passed is True
    assert guarded.breakdown["examined_unknown"] is True


def test_an_empty_extraction_no_longer_passes():
    """The exact input that scored (True, 1.0) before this change."""
    outcome = score_schema_conformance(
        {"response_content": '{"entities": [], "relationships": []}'}, {}
    )

    assert outcome.examined == 0
    assert enforce_non_vacuity(outcome).passed is False


def test_a_real_extraction_still_scores_normally():
    """Guard against fixing the empty case by breaking the real one."""
    real = '{"entities": [{"id": "x", "type": "Person"}], "relationships": []}'

    outcome = score_schema_conformance({"response_content": real}, {})

    assert outcome.examined == 1
    assert enforce_non_vacuity(outcome).passed is True


def test_a_bad_entity_type_still_fails():
    """Verified pre-change: this input returned (False, 0.8)."""
    bad = '{"entities": [{"id": "x", "type": "NotAType"}], "relationships": []}'

    outcome = score_schema_conformance({"response_content": bad}, {})

    assert outcome.passed is False
    assert outcome.examined == 1


def test_an_empty_response_no_longer_passes_personality():
    outcome = score_personality({"response_content": ""}, {})

    assert enforce_non_vacuity(outcome).passed is False


@pytest.mark.parametrize("name", sorted(SCORER_REGISTRY))
def test_every_registered_scorer_returns_a_score_outcome(name):
    """Applies to every entry, so a scorer added later is covered automatically."""
    scorer = SCORER_REGISTRY[name]

    outcome = scorer({"response_content": ""}, {})

    assert isinstance(outcome, ScoreOutcome), f"{name} did not return a ScoreOutcome"
