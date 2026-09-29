"""Pure metric math for the dedup threshold calibration tool.

Two score scales appear throughout:

- raw cosine, in [-1, 1]
- the Neo4j score, `(1 + cos) / 2`, in [0, 1]. This is what
  `vector.similarity.cosine` returns and what `SIMILARITY_THRESHOLD` in
  `backend/knowledge/curation/deduplication.py` is compared against
  (`WHERE score >= $threshold`), so all thresholds here are Neo4j scores and
  merge means `score >= threshold`.

Nothing in this module touches an embedding model or the filesystem.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass


def neo4j_score_from_cosine(cosine: float) -> float:
    """Convert raw cosine to the Neo4j `vector.similarity.cosine` score."""
    return (1.0 + cosine) / 2.0


def cosine_from_neo4j_score(score: float) -> float:
    """Inverse of `neo4j_score_from_cosine`."""
    return 2.0 * score - 1.0


def cosine_similarity(u: Sequence[float], v: Sequence[float]) -> float:
    """Cosine similarity of two equal-length vectors.

    Raises:
        ValueError: On a length mismatch or a zero vector (cosine is undefined;
            `EmbeddingGenerator.generate_embedding` returns zeros for empty text).
    """
    if len(u) != len(v):
        raise ValueError(f"vector length mismatch: {len(u)} vs {len(v)}")
    dot = math.fsum(x * y for x, y in zip(u, v))
    norm_u = math.sqrt(math.fsum(x * x for x in u))
    norm_v = math.sqrt(math.fsum(y * y for y in v))
    if norm_u == 0.0 or norm_v == 0.0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    # Clamp float drift so the cosine never leaves [-1, 1] and the Neo4j
    # score stays inside [0, 1].
    return max(-1.0, min(1.0, dot / (norm_u * norm_v)))


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile of an ascending list, q in [0, 100]."""
    if not sorted_values:
        raise ValueError("percentile of an empty sequence")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q must be within [0, 100], got {q}")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    """Summary statistics; every statistic except `n` is None for empty input."""
    if not values:
        return {
            "n": 0,
            "min": None,
            "p10": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "max": None,
            "mean": None,
        }
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "p10": percentile(ordered, 10),
        "p25": percentile(ordered, 25),
        "median": percentile(ordered, 50),
        "p75": percentile(ordered, 75),
        "p90": percentile(ordered, 90),
        "max": ordered[-1],
        "mean": math.fsum(ordered) / len(ordered),
    }


@dataclass(frozen=True, slots=True)
class Confusion:
    """Merge decisions at one threshold. Positive = the pair is a duplicate."""

    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float | None:
        """TP / (TP + FP); None when nothing is predicted a duplicate."""
        predicted = self.tp + self.fp
        return self.tp / predicted if predicted else None

    @property
    def recall(self) -> float | None:
        """TP / (TP + FN); None when the data holds no duplicate pairs."""
        actual = self.tp + self.fn
        return self.tp / actual if actual else None

    @property
    def f1(self) -> float:
        """2TP / (2TP + FP + FN); 0.0 when that denominator is zero."""
        denominator = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / denominator if denominator else 0.0


def confusion_at(
    duplicate_scores: Sequence[float], distinct_scores: Sequence[float], threshold: float
) -> Confusion:
    """Confusion counts when a pair merges iff `score >= threshold`."""
    tp = sum(1 for s in duplicate_scores if s >= threshold)
    fp = sum(1 for s in distinct_scores if s >= threshold)
    return Confusion(
        tp=tp,
        fp=fp,
        fn=len(duplicate_scores) - tp,
        tn=len(distinct_scores) - fp,
    )


@dataclass(frozen=True, slots=True)
class ThresholdPoint:
    """An operating point on the Neo4j-score scale, with its confusion counts."""

    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float | None
    recall: float | None
    f1: float


def point_at(
    duplicate_scores: Sequence[float], distinct_scores: Sequence[float], threshold: float
) -> ThresholdPoint:
    """Build the `ThresholdPoint` for a fixed threshold."""
    c = confusion_at(duplicate_scores, distinct_scores, threshold)
    return ThresholdPoint(
        threshold=threshold,
        tp=c.tp,
        fp=c.fp,
        fn=c.fn,
        tn=c.tn,
        precision=c.precision,
        recall=c.recall,
        f1=c.f1,
    )


@dataclass(frozen=True, slots=True)
class ZeroFalseMerge:
    """The lowest zero-false-merge operating point on the labelled distinct pairs.

    A merge fires on `score >= threshold`, so every threshold strictly above
    `max_distinct_score` gives zero false merges. Predictions only change at
    observed duplicate scores, so `threshold` is the lowest observed duplicate
    score above `max_distinct_score`: the smallest cutoff that both has zero
    false merges and is reached by at least one duplicate. It is None when no
    duplicate scores above every distinct pair, in which case no threshold has
    zero false merges and non-zero recall.
    """

    max_distinct_score: float | None
    point: ThresholdPoint | None


def zero_false_merge(
    duplicate_scores: Sequence[float], distinct_scores: Sequence[float]
) -> ZeroFalseMerge:
    """Find the lowest zero-false-merge operating point (see `ZeroFalseMerge`)."""
    max_distinct = max(distinct_scores) if distinct_scores else None
    above = [s for s in duplicate_scores if max_distinct is None or s > max_distinct]
    if not above:
        return ZeroFalseMerge(max_distinct_score=max_distinct, point=None)
    threshold = min(above)
    return ZeroFalseMerge(
        max_distinct_score=max_distinct,
        point=point_at(duplicate_scores, distinct_scores, threshold),
    )


def best_f1(
    duplicate_scores: Sequence[float], distinct_scores: Sequence[float]
) -> ThresholdPoint | None:
    """The threshold with the highest F1, searched over every observed score.

    Ties go to the higher threshold, the more conservative merge policy.
    Returns None when there are no scores at all.
    """
    candidates = sorted({*duplicate_scores, *distinct_scores})
    if not candidates:
        return None
    best: ThresholdPoint | None = None
    for candidate in candidates:
        point = point_at(duplicate_scores, distinct_scores, candidate)
        if best is None or point.f1 >= best.f1:
            best = point
    return best


def point_to_dict(point: ThresholdPoint | None) -> dict | None:
    """JSON-ready dict for a threshold point, adding the raw-cosine equivalent."""
    if point is None:
        return None
    out = asdict(point)
    out["raw_cosine"] = cosine_from_neo4j_score(point.threshold)
    return out
