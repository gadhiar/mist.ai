"""Statistics primitives: Wilson interval, nearest-rank percentile, cluster bootstrap.

Wilson is checked against an independent reference implementation of the textbook
formula (not analyse.py's own code), at a known k/n. The bootstrap is checked for
exact reproducibility given the same seed and data, per decision_rules.json's
statistics.bootstrap (B=10000, seed=20260924, percentile method).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from scripts.model_bench import analyse  # noqa: E402


def _reference_wilson(k: int, n: int, z: float) -> tuple[float, float]:
    """Independent reimplementation of the Wilson score interval (textbook formula)."""
    phat = k / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half_width = (z / denom) * math.sqrt((phat * (1 - phat) / n) + (z**2 / (4 * n**2)))
    return (centre - half_width, centre + half_width)


@pytest.mark.parametrize("k,n", [(8, 10), (1, 3), (50, 72), (0, 5), (5, 5)])
def test_wilson_matches_reference_formula(k, n):
    z = 1.9599639845400545
    expected_lo, expected_hi = _reference_wilson(k, n, z)
    lo, hi = analyse.wilson_interval(k, n, z)
    assert lo == pytest.approx(max(0.0, expected_lo), abs=1e-9)
    assert hi == pytest.approx(min(1.0, expected_hi), abs=1e-9)


def test_wilson_none_when_n_is_zero():
    assert analyse.wilson_interval(0, 0, 1.96) is None


def test_nearest_rank_percentile_known_values():
    values = [float(i) for i in range(1, 11)]  # 1..10
    assert analyse.nearest_rank_percentile(values, 95.0) == 10.0
    assert analyse.nearest_rank_percentile(values, 50.0) == 5.0
    assert analyse.nearest_rank_percentile(values, 1.0) == 1.0
    assert analyse.nearest_rank_percentile([], 95.0) is None


def test_nearest_rank_percentile_unsorted_input():
    values = [5.0, 1.0, 9.0, 3.0, 7.0]
    assert analyse.nearest_rank_percentile(values, 95.0) == 9.0


def test_bootstrap_reproducible_with_seed():
    values_by_cluster = {
        "a": [1.0, 1.0, 0.0],
        "b": [0.0, 0.0],
        "c": [1.0],
        "d": [1.0, 1.0, 1.0, 0.0],
    }
    ci1 = analyse.cluster_bootstrap_ci(values_by_cluster, B=2000, seed=20260924, confidence=0.95)
    ci2 = analyse.cluster_bootstrap_ci(values_by_cluster, B=2000, seed=20260924, confidence=0.95)
    assert ci1 == ci2  # exact equality: same seed, same data, same algorithm


def test_bootstrap_different_seed_can_differ():
    values_by_cluster = {"a": [1.0, 0.0], "b": [1.0], "c": [0.0, 0.0, 1.0]}
    ci_a = analyse.cluster_bootstrap_ci(values_by_cluster, B=500, seed=1, confidence=0.95)
    ci_b = analyse.cluster_bootstrap_ci(values_by_cluster, B=500, seed=2, confidence=0.95)
    assert ci_a is not None and ci_b is not None
    # Not asserting inequality (they could coincide), only that both are valid CIs.
    for lo, hi in (ci_a, ci_b):
        assert 0.0 <= lo <= hi <= 1.0


def test_bootstrap_empty_clusters_returns_none():
    assert analyse.cluster_bootstrap_ci({}, B=100, seed=1, confidence=0.95) is None


def test_bootstrap_ci_contains_sample_mean_for_homogeneous_data():
    # All clusters identical value -> CI collapses to a point at that value.
    values_by_cluster = {"a": [1.0, 1.0], "b": [1.0], "c": [1.0, 1.0, 1.0]}
    lo, hi = analyse.cluster_bootstrap_ci(values_by_cluster, B=1000, seed=42, confidence=0.95)
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)


@pytest.mark.parametrize(
    "value,ci,threshold,op,expected",
    [
        (0.9, (0.86, 0.94), 0.85, ">=", "clear"),
        (0.86, (0.80, 0.90), 0.85, ">=", "within-noise"),
        (0.5, (0.4, 0.6), 0.85, ">=", "clear"),
        (10.0, None, 5.0, ">=", "n/a"),
    ],
)
def test_margin_for(value, ci, threshold, op, expected):
    assert analyse.margin_for(value, ci, threshold, op) == expected
