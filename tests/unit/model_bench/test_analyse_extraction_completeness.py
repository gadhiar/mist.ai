"""i66: extraction completeness fallback for analyse.py's "Extraction quality" section.

Before this fix, `render_report` (around the c0 baseline check and the per-arm render
loop) treated an `extraction_summary.json` with no `complete` key as complete, no matter
what its `matched_probes`/`total_probes` said. The hand-built fixture summaries
(`fixtures/analyse/run/{c0,c1-512}/extraction_summary.json`) have no `complete` key and
are 5/5 -- they must keep rendering as complete -- but a real (non-fixture) summary
missing the key with a genuine partial count (e.g. 59/60, a probe that errored or timed
out) was rendered as complete too, hiding the shortfall.

`_extraction_is_complete` is the single helper that both call sites (the c0 baseline
check and the per-arm render loop) now use:
- `complete` is used when present, regardless of the probe counts;
- when absent, falls back to `matched_probes == total_probes`, with both required to be
  present ints;
- a missing or non-int count means incomplete.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.model_bench import analyse  # noqa: E402

# ---------------------------------------------------------------------------
# Unit tests: the helper itself
# ---------------------------------------------------------------------------


def test_no_key_partial_count_is_incomplete():
    # no `complete` key, 59/60 matched -- a real partial run, not a fixture's 5/5.
    summ = {"matched_probes": 59, "total_probes": 60}
    assert analyse._extraction_is_complete(summ) is False


def test_no_key_full_count_is_complete():
    # no `complete` key, 60/60 matched -- the fallback must treat this as complete.
    summ = {"matched_probes": 60, "total_probes": 60}
    assert analyse._extraction_is_complete(summ) is True


def test_explicit_complete_false_is_incomplete_even_with_full_count():
    # an explicit `complete: false` wins over the probe counts, even if they look full.
    summ = {"complete": False, "matched_probes": 60, "total_probes": 60}
    assert analyse._extraction_is_complete(summ) is False


def test_explicit_complete_true_is_complete():
    summ = {"complete": True, "matched_probes": 1, "total_probes": 60}
    assert analyse._extraction_is_complete(summ) is True


def test_no_key_missing_counts_is_incomplete():
    # no `complete` key and no probe counts at all -- incomplete, not a default pass.
    assert analyse._extraction_is_complete({}) is False


def test_no_key_one_count_missing_is_incomplete():
    assert analyse._extraction_is_complete({"matched_probes": 5}) is False
    assert analyse._extraction_is_complete({"total_probes": 5}) is False


def test_no_key_non_int_counts_is_incomplete():
    # a non-int count (e.g. a string, or None from a null in JSON) is not treated as a
    # comparable int -- incomplete, not a crash and not a silent pass.
    assert analyse._extraction_is_complete({"matched_probes": "5", "total_probes": 5}) is False
    assert analyse._extraction_is_complete({"matched_probes": None, "total_probes": 5}) is False


def test_none_summary_is_incomplete():
    assert analyse._extraction_is_complete(None) is False


# ---------------------------------------------------------------------------
# Integration: the c0 baseline with no key and a partial count gives no delta
# ---------------------------------------------------------------------------


def _empty_arm_metrics(arm_id: str) -> analyse.ArmMetrics:
    return analyse.ArmMetrics(arm_id=arm_id)


def _run_metrics(arm_ids: list[str]) -> analyse.RunMetrics:
    arms = {a: _empty_arm_metrics(a) for a in arm_ids}
    return analyse.RunMetrics(
        arms=arms,
        raw_test_scores={a: {} for a in arms},
        raw_layout_rows={a: {} for a in arms},
        raw_correctness={a: {} for a in arms},
        voice_vram_lower_mib=analyse.missing_metric("n/a"),
        voice_vram_upper_mib=analyse.missing_metric("n/a"),
        total_mib=None,
        manual={},
        arm_order=list(arm_ids),
    )


def _render(extraction_summaries: dict) -> str:
    metrics = _run_metrics(list(extraction_summaries))
    return analyse.render_report(
        metrics=metrics,
        rule_results=[],
        exploratory_results=[],
        finalist_candidates=[],
        coverage={},
        missing_inputs=[],
        sha_warnings=[],
        sha_infos=[],
        decision_rules_sha="test-sha",
        extraction_summaries=extraction_summaries,
    )


def _extraction_summary(**overrides) -> dict:
    base = {
        "entity_precision": 0.9,
        "entity_recall": 0.9,
        "rel_precision": 0.9,
        "rel_recall": 0.9,
        "rel_f1": 0.9,
        "typing_accuracy": 0.9,
        "matched_probes": 60,
        "total_probes": 60,
    }
    base.update(overrides)
    return base


def _extraction_quality_section(report: str) -> str:
    # `render_report` has TWO "### <arm_id>" header sets: one under "## Per-arm metrics"
    # and one under "## Extraction quality" -- isolate the latter before splitting on
    # a specific arm's own "### <arm_id>" header.
    return report.split("## Extraction quality")[1].split("## Finalist candidates")[0]


def test_c0_baseline_no_key_partial_count_gives_no_delta():
    # c0 has no `complete` key and a genuine partial count (59/60) -- it must be
    # excluded as a delta baseline (same as an explicit `complete: false` baseline),
    # so c1's row reports "n/a" for every delta-vs-c0 cell rather than a computed
    # number against an incomplete baseline.
    c0_summary = _extraction_summary(matched_probes=59, total_probes=60)
    c1_summary = _extraction_summary(entity_precision=0.5)
    report = _render({"c0": c0_summary, "c1": c1_summary})
    extraction_section = _extraction_quality_section(report)

    # c0 itself renders as incomplete (no metrics table, no delta column at all).
    c0_section = extraction_section.split("### c0")[1].split("### c1")[0]
    assert "incomplete: 59/60 probes matched" in c0_section

    # c1 is complete (60/60, no key) and must render its table, but every delta cell
    # is "n/a" because the baseline (c0) is incomplete.
    c1_section = extraction_section.split("### c1")[1]
    assert "incomplete" not in c1_section
    for line in c1_section.splitlines():
        if line.startswith("| entity_precision") or line.startswith("| rel_f1"):
            assert line.rstrip().endswith("| n/a |"), line


def test_arm_no_key_full_count_renders_complete_with_metrics_table():
    summ = _extraction_summary()
    report = _render({"c0": summ})
    section = _extraction_quality_section(report).split("### c0")[1]
    assert "incomplete" not in section
    assert "| entity_precision |" in section
