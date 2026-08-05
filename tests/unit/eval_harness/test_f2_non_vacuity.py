"""F2 must not report a pass over a corpus it never read.

Verified by execution on 2026-08-05, before any change:

    r = score_run([], {})
    -> total_probes=0, entity_precision=1.0, rel_precision=1.0,
       typing_accuracy=1.0, negative_violations=0
    _gates_pass(r) -> True

Every gate passes on an empty gold corpus, including the join-integrity check
`matched_probes == total_probes` that exists to stop exactly this -- on an empty
corpus it is `0 == 0`. `--strict` therefore exits 0.

A mistyped path is NOT the trigger: `main():689-690` checks `args.gold.exists()`
and exits 2. The trigger is a gold file that exists and yields zero probes --
empty, blank-line-only, all-comment, truncated, or over-filtered. Verified by
execution: all three of those shapes return 0 probes and pass every gate.

This is the scorer whose numbers gated ontology v1.4.0 and closed MIS-124.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[3]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from scripts.eval_harness.score_extraction_run import (  # noqa: E402
    _gates_pass,
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
