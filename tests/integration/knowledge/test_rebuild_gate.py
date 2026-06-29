"""Unit tests for the rebuild determinism gate + divergence report (R1.2)."""

import pytest

from backend.knowledge.regeneration.rebuild_gate import (
    RebuildDeterminismError,
    assert_rebuild_twice_identical,
    live_vs_rebuilt_report,
)


def test_identical_builds_pass():
    form = '{"nodes": [], "relationships": []}\n'
    assert_rebuild_twice_identical(form, form)  # no raise


def test_divergent_builds_raise_with_diff():
    a = '{"nodes": [{"id": "a"}]}\n'
    b = '{"nodes": [{"id": "b"}]}\n'
    with pytest.raises(RebuildDeterminismError, match="rebuild-twice"):
        assert_rebuild_twice_identical(a, b)


def test_live_vs_rebuilt_report_equal_is_explicit():
    form = '{"nodes": []}\n'
    report = live_vs_rebuilt_report(form, form)
    assert "no divergence" in report.lower()


def test_live_vs_rebuilt_report_shows_diff_when_different():
    report = live_vs_rebuilt_report('{"x": 1}\n', '{"x": 2}\n')
    assert "-" in report and "+" in report  # unified diff markers
