"""Metric math for the dedup calibration tool, checked against hand-computed values."""

from __future__ import annotations

import math

import pytest

from scripts.dedup_calibration.metrics import (
    best_f1,
    confusion_at,
    cosine_from_neo4j_score,
    cosine_similarity,
    distribution,
    neo4j_score_from_cosine,
    percentile,
    point_at,
    point_to_dict,
    zero_false_merge,
)

# Hand-worked fixture. Merge means score >= threshold.
#   duplicates: 0.95 0.90 0.80      distinct: 0.85 0.70 0.60
DUP = [0.95, 0.90, 0.80]
DISTINCT = [0.85, 0.70, 0.60]


class TestScoreConversion:
    def test_current_threshold_is_raw_cosine_point_84(self):
        assert cosine_from_neo4j_score(0.92) == pytest.approx(0.84)
        assert neo4j_score_from_cosine(0.84) == pytest.approx(0.92)

    @pytest.mark.parametrize(
        ("cosine", "score"), [(1.0, 1.0), (0.0, 0.5), (-1.0, 0.0), (0.5, 0.75), (-0.5, 0.25)]
    )
    def test_neo4j_score_is_one_plus_cos_over_two(self, cosine, score):
        assert neo4j_score_from_cosine(cosine) == pytest.approx(score)

    @pytest.mark.parametrize("cosine", [-1.0, -0.3, 0.0, 0.42, 1.0])
    def test_round_trip(self, cosine):
        assert cosine_from_neo4j_score(neo4j_score_from_cosine(cosine)) == pytest.approx(cosine)


class TestCosineSimilarity:
    def test_identical_vectors_are_one(self):
        assert cosine_similarity([0.3, 0.4, 0.5], [0.3, 0.4, 0.5]) == pytest.approx(1.0)

    def test_result_never_leaves_the_unit_interval(self):
        for v in ([0.1] * 7, [1e-9, 3.0, 2.0], [0.3, 0.4, 0.5]):
            assert -1.0 <= cosine_similarity(v, v) <= 1.0
            assert -1.0 <= cosine_similarity(v, [-x for x in v]) <= 1.0

    def test_orthogonal_is_zero(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_opposite_is_minus_one(self):
        assert cosine_similarity([1.0, 2.0], [-1.0, -2.0]) == pytest.approx(-1.0)

    def test_known_angle(self):
        assert cosine_similarity([1.0, 0.0], [0.6, 0.8]) == pytest.approx(0.6)

    def test_scale_invariant(self):
        assert cosine_similarity([1.0, 2.0], [10.0, 20.0]) == pytest.approx(1.0)

    def test_zero_vector_rejected(self):
        with pytest.raises(ValueError, match="zero vector"):
            cosine_similarity([0.0, 0.0], [1.0, 0.0])

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="length mismatch"):
            cosine_similarity([1.0], [1.0, 2.0])


class TestDistribution:
    def test_percentile_interpolates_linearly(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert percentile(values, 0) == 1.0
        assert percentile(values, 25) == 2.0
        assert percentile(values, 50) == 3.0
        assert percentile(values, 10) == pytest.approx(1.4)
        assert percentile(values, 100) == 5.0

    def test_percentile_single_value(self):
        assert percentile([7.0], 90) == 7.0

    def test_percentile_rejects_empty_and_bad_q(self):
        with pytest.raises(ValueError):
            percentile([], 50)
        with pytest.raises(ValueError):
            percentile([1.0], 101)

    def test_distribution_summary(self):
        d = distribution([5.0, 1.0, 3.0, 2.0, 4.0])
        assert d["n"] == 5
        assert d["min"] == 1.0
        assert d["max"] == 5.0
        assert d["median"] == 3.0
        assert d["mean"] == 3.0
        assert d["p25"] == 2.0
        assert d["p75"] == 4.0

    def test_empty_distribution_is_all_none_with_zero_n(self):
        d = distribution([])
        assert d["n"] == 0
        assert all(v is None for k, v in d.items() if k != "n")


class TestConfusionAndF1:
    def test_confusion_counts_at_0_90(self):
        c = confusion_at(DUP, DISTINCT, 0.90)
        assert (c.tp, c.fp, c.fn, c.tn) == (2, 0, 1, 3)
        assert c.precision == 1.0
        assert c.recall == pytest.approx(2 / 3)
        assert c.f1 == pytest.approx(0.8)

    def test_confusion_counts_at_0_80(self):
        c = confusion_at(DUP, DISTINCT, 0.80)
        assert (c.tp, c.fp, c.fn, c.tn) == (3, 1, 0, 2)
        assert c.precision == 0.75
        assert c.recall == 1.0
        assert c.f1 == pytest.approx(6 / 7)

    def test_boundary_is_inclusive_like_the_cypher_where(self):
        # WHERE score >= $threshold: a score exactly at the threshold merges.
        c = confusion_at([0.9], [0.9], 0.9)
        assert (c.tp, c.fp) == (1, 1)

    def test_no_predicted_positive_gives_none_precision_and_zero_f1(self):
        c = confusion_at(DUP, DISTINCT, 0.99)
        assert c.tp == 0 and c.fp == 0
        assert c.precision is None
        assert c.recall == 0.0
        assert c.f1 == 0.0

    def test_no_duplicates_gives_none_recall(self):
        c = confusion_at([], [0.5], 0.4)
        assert c.recall is None
        assert c.f1 == 0.0

    def test_point_to_dict_adds_raw_cosine(self):
        d = point_to_dict(point_at(DUP, DISTINCT, 0.92))
        assert d["threshold"] == 0.92
        assert d["raw_cosine"] == pytest.approx(0.84)
        assert point_to_dict(None) is None


class TestZeroFalseMerge:
    def test_lowest_duplicate_score_above_every_distinct_pair(self):
        result = zero_false_merge(DUP, DISTINCT)
        assert result.max_distinct_score == 0.85
        assert result.point is not None
        assert result.point.threshold == 0.90
        assert (result.point.tp, result.point.fp, result.point.fn) == (2, 0, 1)

    def test_no_false_merge_at_returned_threshold_and_one_at_the_next_lower_score(self):
        result = zero_false_merge(DUP, DISTINCT)
        assert result.point is not None
        assert confusion_at(DUP, DISTINCT, result.point.threshold).fp == 0
        assert confusion_at(DUP, DISTINCT, 0.85).fp == 1

    def test_none_when_no_duplicate_outscores_every_distinct_pair(self):
        result = zero_false_merge([0.5, 0.6], [0.9])
        assert result.max_distinct_score == 0.9
        assert result.point is None

    def test_a_tie_with_the_highest_distinct_score_does_not_count(self):
        # A duplicate scoring exactly max distinct still merges that distinct pair.
        result = zero_false_merge([0.9, 0.95], [0.9])
        assert result.point is not None
        assert result.point.threshold == 0.95

    def test_no_distinct_pairs_means_every_duplicate_is_reachable(self):
        result = zero_false_merge([0.7, 0.8], [])
        assert result.max_distinct_score is None
        assert result.point is not None
        assert result.point.threshold == 0.7
        assert result.point.recall == 1.0


class TestBestF1:
    def test_picks_threshold_with_highest_f1(self):
        best = best_f1(DUP, DISTINCT)
        assert best is not None
        assert best.threshold == 0.80
        assert best.f1 == pytest.approx(6 / 7)
        assert (best.tp, best.fp, best.fn, best.tn) == (3, 1, 0, 2)

    def test_perfectly_separable_data_reaches_f1_one(self):
        best = best_f1([0.9, 0.8], [0.7])
        assert best is not None
        assert best.f1 == 1.0
        assert best.threshold == 0.8

    def test_ties_go_to_the_higher_threshold(self):
        # F1 is 2/3 at both 0.5 and 0.9; the conservative (higher) one wins.
        best = best_f1([0.9, 0.5], [0.7, 0.6])
        assert best is not None
        assert math.isclose(best.f1, 2 / 3)
        assert best.threshold == 0.9

    def test_none_without_scores(self):
        assert best_f1([], []) is None
