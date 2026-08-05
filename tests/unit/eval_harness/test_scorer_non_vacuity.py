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

import json
import subprocess
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


def test_every_scorer_declares_what_examined_means():
    """A count nobody can interpret is not an improvement on no count.

    This test is the anti-rot mechanism: a scorer added to SCORER_REGISTRY
    without a SCORER_EXAMINES entry fails the suite rather than shipping an
    uninterpretable number. Mirrors run_record.py's registry guard.
    """
    from eval_harness.scorers import SCORER_EXAMINES

    missing = sorted(set(SCORER_REGISTRY) - set(SCORER_EXAMINES))
    assert not missing, f"scorers missing an examined declaration: {missing}"

    stale = sorted(set(SCORER_EXAMINES) - set(SCORER_REGISTRY))
    assert not stale, f"declarations for scorers that no longer exist: {stale}"

    for name, description in SCORER_EXAMINES.items():
        assert len(description) >= 20, f"{name}'s declaration is too short to be useful"


def _extract_heredoc(source: str, start_marker: str) -> str:
    """Pull one `python <<PYEOF ... PYEOF` block's body out of orchestrator shell source.

    The D2 kill-switch is embedded Python inside phase3_orchestrator.sh, not an
    importable function -- there is nothing to `import` and call directly. This
    reads the block out of the live file at test time, so the test exercises
    exactly what the shell script would run, and fails if a future edit strips
    `enforce_non_vacuity` from that block, not just from a copy pasted into this
    test file.
    """
    start = source.index(start_marker) + len(start_marker)
    end = source.index("\nPYEOF", start)
    return source[start:end]


def test_d2_kill_switch_does_not_score_a_vacuous_extraction_as_a_pass(tmp_path):
    """The D2 kill-switch decides whether phases D3-D5 run at all.

    Before this branch's fix, this block read `.score` off a raw ScoreOutcome:
    an empty extraction scored 1.0 (the audit's own canonical vacuous example,
    reproduced verbatim in this file's module docstring) and the kill-switch
    would never trip on it. It must now score 0.0.
    """
    orchestrator_path = _REPO_ROOT / "scripts" / "eval_harness" / "phase3_orchestrator.sh"
    source = orchestrator_path.read_text(encoding="utf-8")
    heredoc = _extract_heredoc(source, "local best_schema; best_schema=$(python <<PYEOF\n")
    assert "enforce_non_vacuity" in heredoc, "guard is missing from the extracted block"

    candidate_id = "vacuous-candidate"
    jsonl_path = tmp_path / f"{candidate_id}.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "test_name": "schema_conformance",
                "response_content": '{"entities": [], "relationships": []}',
                "expected": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    script = heredoc.replace("${d2_result_dir}", str(tmp_path)).replace(
        "${best_gemma_d2}", candidate_id
    )

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "0.000"
