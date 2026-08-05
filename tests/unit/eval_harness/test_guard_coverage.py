"""The `enforce_non_vacuity` dispatch-site list must be a mechanism, not a memo.

`enforce_non_vacuity`'s own docstring (scripts/eval_harness/scorers.py) names five
call sites and says plainly: "That list is maintained by hand, not enforced --
nothing currently fails a build if a new dispatch site reads `.score` off a
`ScoreOutcome` without routing through this function first." Four of the five
sites were missed exactly that way on 2026-08-04, including the D2 kill-switch
that gates phases D3-D5, and were only caught by a whole-branch review.

This file is the mechanism `SCORER_EXAMINES` already has for a different gap
(a scorer without a declared `examined` meaning) applied to this one: a
static, file-scoped check that a new unguarded `.score` read fails the suite
instead of waiting for the next review pass to notice.

Scope: `scripts/eval_harness/exoneration_verdict.py` and the Python heredocs
embedded in `scripts/eval_harness/phase3_orchestrator.sh` -- the two files
named in the task. `scorers.py` itself is deliberately out of scope: its one
dispatch site (`_ingest_record`) sits a few lines below `enforce_non_vacuity`'s
own definition in the same file, and `scorers.py` also contains `.score` reads
that are NOT raw scorer-result reads and must not be flagged --
`score_schema_conformance_lenient` reads `.score` off its inner delegate
scorer's outcome before that outcome has been guarded, by design: it
constructs its own `ScoreOutcome` with `examined=outcome.examined` copied
through unguarded, so vacuity information survives to whichever dispatch site
later calls `enforce_non_vacuity` on ITS result. `report.py` and
`TestScores.mean_score` read `.score` off `CaseScore`, which is populated only
after `_ingest_record` has already applied the guard. A rule that flagged
either of those would be checking the wrong thing.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_EXONERATION_VERDICT = _REPO_ROOT / "scripts" / "eval_harness" / "exoneration_verdict.py"
_ORCHESTRATOR = _REPO_ROOT / "scripts" / "eval_harness" / "phase3_orchestrator.sh"


def _is_enforce_non_vacuity_call(node: ast.expr) -> bool:
    """True if `node` is a call to `enforce_non_vacuity`, bare or attribute-qualified.

    Matches both `enforce_non_vacuity(...)` (direct import, as in
    exoneration_verdict.py) and `scorers.enforce_non_vacuity(...)` (qualified,
    as in the orchestrator heredocs) -- the two forms the codebase actually
    uses -- without requiring the exact module alias.
    """
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "enforce_non_vacuity"
    if isinstance(func, ast.Attribute):
        return func.attr == "enforce_non_vacuity"
    return False


def _unguarded_score_reads(tree: ast.AST, label: str) -> tuple[list[str], int]:
    """Return (violation messages, number of `.score` reads examined) in `tree`.

    A `.score` read is guarded when its value expression is itself a call to
    `enforce_non_vacuity` -- the exact shape every known dispatch site uses:
    `enforce_non_vacuity(scorer(...)).score`. Any other `.score` attribute
    access is reported as a violation naming the file and line, so a sixth
    site added later without the guard fails here instead of at the next
    whole-branch review.
    """
    violations: list[str] = []
    examined = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "score":
            examined += 1
            if not _is_enforce_non_vacuity_call(node.value):
                line = getattr(node, "lineno", "?")
                violations.append(
                    f"{label}:{line}: `.score` read not immediately wrapped in "
                    "enforce_non_vacuity(...)"
                )
    return violations, examined


def _extract_heredocs(source: str, delimiter: str = "PYEOF") -> list[tuple[int, str]]:
    """Pull every `<<DELIM ... DELIM` heredoc body out of shell source.

    Generalizes test_scorer_non_vacuity.py's `_extract_heredoc`, which pulls
    ONE heredoc identified by a hardcoded start-marker string. That approach
    does not scale to "every heredoc in the file" without one hardcoded
    marker per block, so this walks the source for every occurrence of a
    `<<PYEOF` (or `<<'PYEOF'`) opener and reads to the next line that is
    exactly `PYEOF`, the same index-based technique, applied repeatedly. A
    heredoc added later needs no new marker here to be picked up.

    Reads the live file at test time, like the original, so this exercises
    exactly what the shell script would run rather than a copy pasted into
    this test file.
    """
    heredocs: list[tuple[int, str]] = []
    start_re = re.compile(rf"<<-?\s*['\"]?{re.escape(delimiter)}['\"]?\n")
    closing_re = re.compile(rf"^{re.escape(delimiter)}\s*$", re.MULTILINE)
    pos = 0
    while True:
        start_match = start_re.search(source, pos)
        if start_match is None:
            break
        body_start = start_match.end()
        closing_match = closing_re.search(source, body_start)
        if closing_match is None:
            break
        body = source[body_start : closing_match.start()]
        start_line = source.count("\n", 0, start_match.start()) + 1
        heredocs.append((start_line, body))
        pos = closing_match.end()
    return heredocs


def test_exoneration_verdict_guards_every_score_read():
    source = _EXONERATION_VERDICT.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_EXONERATION_VERDICT))

    violations, examined = _unguarded_score_reads(tree, _EXONERATION_VERDICT.name)

    assert examined > 0, (
        "found zero `.score` reads in exoneration_verdict.py -- this check is not "
        "firing, not that the file has none"
    )
    assert not violations, "\n".join(violations)


def test_orchestrator_heredocs_guard_every_score_read():
    source = _ORCHESTRATOR.read_text(encoding="utf-8")
    heredocs = _extract_heredocs(source)

    assert (
        heredocs
    ), "found zero PYEOF heredocs in phase3_orchestrator.sh -- extraction is not firing"

    all_violations: list[str] = []
    total_examined = 0
    for start_line, body in heredocs:
        label = f"{_ORCHESTRATOR.name}:heredoc@line{start_line}"
        tree = ast.parse(body, filename=label)
        violations, examined = _unguarded_score_reads(tree, label)
        all_violations.extend(violations)
        total_examined += examined

    assert total_examined > 0, (
        "found zero `.score` reads across all heredocs -- this check is not firing, "
        "not that the script has none"
    )
    assert not all_violations, "\n".join(all_violations)
