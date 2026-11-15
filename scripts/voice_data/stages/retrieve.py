"""Target-speaker retrieval on embeddings: the anchor check, both passes, the separator score.

Design 4 in order. The anchors must agree. In each episode the diarized cluster nearest the
anchor centroid is primary when it reaches the anchor floor on both embedders; every other
cluster that reaches the floor is flagged, because a split by register and a similar-sounding
cast member reach it alike. The target model is then built from primary turns only, so a flagged
cluster cannot pull the centroid, and a turn is kept when it beats every non-admitted cluster of
its episode by the margin and reaches the floor of the target's leave-one-out distribution.

An `Embedding` maps each embedder's name to one clip's vector, and every rule holds on all of
them. SIMs equal up to rounding pass a floor: a voice whose clips embed identically must not be
dropped by the last bit of a cosine.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .score import Distribution, centroid, cosine, distribution, leave_one_out_scores

Embedding = dict[str, np.ndarray]
EPS = 1e-9
PRIMARY = "primary"
FLAGGED = "flagged"
OTHER = "other"


@dataclass(frozen=True, slots=True)
class ClusterRole:
    """A diarized cluster's role in its episode and the medians that decided it."""

    label: str
    role: str
    turns: int
    median_sim: dict[str, float]


@dataclass(frozen=True, slots=True)
class TargetModel:
    """The target's centroid, leave-one-out distribution and floor, per embedder."""

    centroid: dict[str, np.ndarray]
    distribution: dict[str, Distribution]
    floor: dict[str, float]


@dataclass(frozen=True, slots=True)
class Verdict:
    """The second pass on one clip: whether it is kept, why not, its SIMs and its margin.

    `margin` is the smallest, over embedders, of the clip's SIM to the target less its highest
    SIM to a rival cluster; `None` when its episode has no rival.
    """

    passed: bool
    reasons: tuple[str, ...]
    sim: dict[str, float]
    margin: float | None


def _names(embs: Sequence[Mapping[str, np.ndarray]]) -> list[str]:
    return list(embs[0]) if embs else []


def centroids(embs: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Return the centroid of the clips on each embedder.

    Raises:
        ValueError: `embs` is empty.
    """
    if not embs:
        raise ValueError("a centroid needs at least one clip")
    return {name: centroid([e[name] for e in embs]) for name in _names(embs)}


def sims(emb: Mapping[str, np.ndarray], centre: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Return the clip's SIM to a centroid on each embedder."""
    return {name: cosine(emb[name], centre[name]) for name in centre}


def anchor_check(
    anchors: Sequence[Mapping[str, np.ndarray]], sim_min: float
) -> tuple[bool, list[dict[str, float]]]:
    """Check that the anchors agree: each one's SIM to the centroid of the others.

    Returns:
        Whether every anchor reaches `sim_min` on every embedder, and each anchor's SIMs. A
        single anchor has no others, so it passes with no SIM.
    """
    if len(anchors) < 2:
        return True, [{} for _ in anchors]
    found = []
    for i, anchor in enumerate(anchors):
        others = [a for j, a in enumerate(anchors) if j != i]
        found.append(sims(anchor, centroids(others)))
    ok = all(value >= sim_min - EPS for row in found for value in row.values())
    return ok, found


def _by_label(turns: Sequence[tuple[str, Embedding]]) -> dict[str, list[Embedding]]:
    grouped: dict[str, list[Embedding]] = {}
    for label, emb in turns:
        grouped.setdefault(label, []).append(emb)
    return dict(sorted(grouped.items()))


def cluster_roles(
    turns: Sequence[tuple[str, Embedding]], anchor_centroid: Mapping[str, np.ndarray], floor: float
) -> dict[str, ClusterRole]:
    """The first pass: each cluster of one episode is `primary`, `flagged` or `other`.

    A cluster reaches the floor when its median SIM to the anchor centroid does on every
    embedder. Of those, the one with the highest mean of its medians is primary (ties to the
    cluster with more turns, then the first label); the rest are flagged.

    Args:
        turns: Each turn's diarizer label and embedding.
        anchor_centroid: The anchors' centroid on each embedder.
        floor: The anchor floor.
    """
    medians: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}
    for label, embs in _by_label(turns).items():
        counts[label] = len(embs)
        medians[label] = {
            name: float(np.median([cosine(e[name], centre) for e in embs]))
            for name, centre in anchor_centroid.items()
        }
    reaching = [
        label for label, row in medians.items() if all(v >= floor - EPS for v in row.values())
    ]
    primary = None
    if reaching:
        means = {lb: float(np.mean(list(medians[lb].values()))) for lb in reaching}
        best = max(means.values())
        tied = [lb for lb in reaching if means[lb] >= best - EPS]
        primary = min(tied, key=lambda lb: (-counts[lb], lb))
    roles = {}
    for label, row in medians.items():
        role = PRIMARY if label == primary else FLAGGED if label in reaching else OTHER
        roles[label] = ClusterRole(label=label, role=role, turns=counts[label], median_sim=row)
    return roles


def centroids_by_label(
    turns: Sequence[tuple[str, Embedding]], labels: set[str] | frozenset[str]
) -> dict[str, dict[str, np.ndarray]]:
    """Return the centroid of each named cluster's turns."""
    return {label: centroids(embs) for label, embs in _by_label(turns).items() if label in labels}


def rival_centroids(
    turns: Sequence[tuple[str, Embedding]], roles: Mapping[str, ClusterRole]
) -> dict[str, dict[str, np.ndarray]]:
    """Return the centroids of the clusters not admitted; primary and flagged are not rivals."""
    others = {label for label, role in roles.items() if role.role == OTHER}
    return centroids_by_label(turns, others)


def target_model(
    embs: Sequence[Embedding], sources: Sequence[str], percentile: float
) -> TargetModel | None:
    """Build the target model from primary turns: centroid, leave-one-out distribution, floor.

    The floor is the `percentile` of each clip's SIM to the centroid of the others (other
    sources' clips where any exist), the same quantity a gate compares.

    Returns:
        The model, or `None` with fewer than two clips, where no leave-one-out exists.
    """
    if len(embs) < 2:
        return None
    centre: dict[str, np.ndarray] = {}
    dist: dict[str, Distribution] = {}
    floor: dict[str, float] = {}
    for name in _names(embs):
        vectors = [e[name] for e in embs]
        scores = leave_one_out_scores(vectors, sources)
        centre[name] = centroid(vectors)
        dist[name] = distribution(scores.tolist())
        floor[name] = float(np.percentile(scores, percentile))
    return TargetModel(centroid=centre, distribution=dist, floor=floor)


def judge(
    emb: Mapping[str, np.ndarray],
    model: TargetModel,
    rivals: Mapping[str, Mapping[str, np.ndarray]],
    margin_min: float,
) -> Verdict:
    """The second pass on one clip: the floor and the margin over every rival, per embedder.

    Reasons are `floor:<embedder>` for each embedder below the floor, then `margin:<embedder>`
    for each where some rival comes within `margin_min`.
    """
    sim = sims(emb, model.centroid)
    floors = [f"floor:{name}" for name in sim if not sim[name] >= model.floor[name] - EPS]
    margins: list[str] = []
    margin = None
    if rivals:
        gaps = {}
        for name in sim:
            nearest = max(cosine(emb[name], rival[name]) for rival in rivals.values())
            gaps[name] = sim[name] - nearest
            if not gaps[name] >= margin_min - EPS:
                margins.append(f"margin:{name}")
        margin = min(gaps.values())
    reasons = tuple(floors + margins)
    return Verdict(passed=not reasons, reasons=reasons, sim=sim, margin=margin)


def candidate_score(
    windows: Sequence[Mapping[str, np.ndarray]],
    reference: Mapping[str, np.ndarray],
    best_n: int = 50,
) -> float | None:
    """Score a separator candidate: the median SIM of its `best_n` best windows to a reference.

    The median is taken per embedder and the candidate's score is their mean.

    Returns:
        The score, or `None` when the candidate gave no speech window.
    """
    if not windows:
        return None
    medians = []
    for name, centre in reference.items():
        values = sorted((cosine(w[name], centre) for w in windows), reverse=True)
        medians.append(float(np.median(values[:best_n])))
    return float(np.mean(medians))


def choose_separator(scores: Mapping[str, float | None]) -> str:
    """Return the candidate with the highest score; ties go to the earlier one (`none` first)."""
    names = list(scores)
    best = names[0]
    for name in names[1:]:
        value, current = scores[name], scores[best]
        if value is not None and (current is None or value > current):
            best = name
    return best
