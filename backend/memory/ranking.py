"""The order of the memory read path: RRF across the pools, then an order by rank.

The fetcher composes these pure functions around the reranker call it times against the
deadline. `shortlist` fuses the candidate pools and keeps the first `rerank_top_n`; the
reranker scores as many of those, in pool order, as fit the remaining budget; `order_candidates`
sorts every candidate it is given by rank (plan v7 A37): inside the user's dated window first,
then `order_score`, which fuses the pool rank with the cross-encoder's rank, then recency,
confidence, id and kind. No candidate is dropped here: the fetcher selects the first
`select_k`. Keeping the arithmetic here, free of I/O and clocks, lets each key be tested on its
own and makes a replay order the same way.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from backend.vault.models import AuthoredBy

from .models import Candidate, EpisodeSource, FactSource, NoteSource, TimeWindow

RRF_K = 60

# Recency half-lives. Episodes and exchanges are continuity, which goes stale in weeks; facts
# are durable beliefs, measured from `recorded_at`. Notes are durable documents like facts, so
# they decay on the fact half-life (the design names no half-life for notes).
EPISODE_HALF_LIFE_DAYS = 14.0
FACT_HALF_LIFE_DAYS = 90.0
_HALF_LIFE_DAYS: dict[str, float] = {
    "fact": FACT_HALF_LIFE_DAYS,
    "note": FACT_HALF_LIFE_DAYS,
    "episode": EPISODE_HALF_LIFE_DAYS,
    "exchange": EPISODE_HALF_LIFE_DAYS,
}

# Confidence by provenance. A note the user wrote or edited is their own statement; one MIST
# wrote is its reading of the user, and one MIST flagged for review is a reading it was unsure
# of, so it is trusted no more than raw conversation. A summarized episode has been condensed by
# a model pass that drops chatter; a raw episode or a single exchange is unfiltered
# conversation. The author sets hold the enum values, so a rename in AuthoredBy breaks this
# import, and `_author_value` reduces either a member or a plain frontmatter string to one.
USER_AUTHORS = frozenset({AuthoredBy.USER.value, AuthoredBy.USER_EDIT.value})
PENDING_REVIEW_AUTHORS = frozenset({AuthoredBy.MIST_PENDING_REVIEW.value})
USER_NOTE_CONFIDENCE = 0.9
MIST_NOTE_CONFIDENCE = 0.7
SUMMARIZED_EPISODE_CONFIDENCE = 0.7
RAW_CONFIDENCE = 0.5
PENDING_REVIEW_NOTE_CONFIDENCE = RAW_CONFIDENCE
# A fact candidate always carries its edge confidence; one that does not is scored like raw
# conversation rather than rejected, so a source bug costs ranking quality, not the turn.
MISSING_FACT_CONFIDENCE = RAW_CONFIDENCE

_SECONDS_PER_DAY = 86_400.0


@dataclass(frozen=True, slots=True)
class FusedCandidate:
    """A candidate after RRF fusion.

    Attributes:
        candidate: The candidate, with the ranks of every pool that offered it merged.
        rrf: Its reciprocal rank fusion score, `sum(1 / (k + rank))` over those pools.
        pool_rank: Its 1-based position in the fused order. It stays the candidate's pool rank
            when the fetcher drops others from the shortlist, so the order term it feeds does
            not change with what else was dropped.
    """

    candidate: Candidate
    rrf: float
    pool_rank: int


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    """A candidate with its place in the order and the keys that set it, for the trace and tests.

    Attributes:
        candidate: The candidate ordered.
        score: Its order score, `order_score(pool_rank, ce_rank)`.
        position: Its 1-based place in the sorted order, the key the block packs by. The
            fetcher gives a fact that expansion places the position of the candidate that
            brought it, when that is earlier than its own.
        pool_rank: Its 1-based position in the fused order.
        ce_rank: Its 1-based place among the reranked candidates by cross-encoder
            probability, ties by pool rank; None when the cross-encoder did not score it.
        p_ce: The cross-encoder's probability; None when it did not score it.
        time_term: The window-match term when `window_applied`, otherwise recency.
        window_applied: Whether the turn named a dated window, whose items sort first.
        confidence: The confidence term.
    """

    candidate: Candidate
    score: float
    position: int
    pool_rank: int
    ce_rank: int | None
    p_ce: float | None
    time_term: float
    window_applied: bool
    confidence: float


def rrf_score(ranks: Mapping[str, int], *, k: int = RRF_K) -> float:
    """Reciprocal rank fusion over the pools that returned an item.

    RRF needs no score calibration between pools (BM25, cosine and entity anchoring are on
    unrelated scales), and an item several pools agree on rises above one that only a single
    pool ranks first.

    Args:
        ranks: 1-based rank of the item in each pool, by pool name.
        k: Damping constant; 60 by the design.

    Raises:
        ValueError: A rank is below 1.
    """
    for pool, rank in ranks.items():
        if rank < 1:
            raise ValueError(f"rank in pool {pool!r} must be 1 or more, got {rank}")
    return sum(1.0 / (k + rank) for rank in ranks.values())


def fuse(candidates: Iterable[Candidate], *, k: int = RRF_K) -> list[FusedCandidate]:
    """Merge candidates offered by several sources and order them by RRF, best first.

    Two candidates with the same kind and item id are one item: their ranks are merged,
    keeping the better rank where both name a pool, and the first candidate's other fields
    stand. Equal RRF scores are ordered by item id, then kind, so the order never depends on
    which source answered first.
    """
    merged: dict[tuple[str, str], Candidate] = {}
    for candidate in candidates:
        key = (candidate.kind, candidate.item_id)
        seen = merged.get(key)
        if seen is None:
            merged[key] = candidate
            continue
        ranks = dict(seen.ranks)
        for pool, rank in candidate.ranks.items():
            ranks[pool] = min(rank, ranks.get(pool, rank))
        merged[key] = replace(seen, ranks=ranks)
    scored = [(candidate, rrf_score(candidate.ranks, k=k)) for candidate in merged.values()]
    scored.sort(key=lambda pair: (-pair[1], pair[0].item_id, pair[0].kind))
    return [
        FusedCandidate(candidate=candidate, rrf=rrf, pool_rank=position)
        for position, (candidate, rrf) in enumerate(scored, start=1)
    ]


def shortlist(
    candidates: Iterable[Candidate], *, top_n: int, k: int = RRF_K
) -> list[FusedCandidate]:
    """The first `top_n` candidates by RRF, the only ones that go on to the rerank and order."""
    return fuse(candidates, k=k)[:top_n]


def recency_term(candidate: Candidate, now: datetime) -> float:
    """Exponential recency, `0.5 ** (age / half_life)`, by the candidate's kind.

    A timestamp after `now` (a clock skew between writer and reader) counts as brand new
    rather than scoring above 1. A candidate without a timestamp scores 0: nothing says it is
    recent.

    Raises:
        ValueError: `now` or the candidate's `when` is naive.
    """
    if candidate.when is None:
        return 0.0
    _require_aware("now", now)
    _require_aware("candidate when", candidate.when)
    age_days = (now.astimezone(UTC) - candidate.when.astimezone(UTC)).total_seconds()
    age_days /= _SECONDS_PER_DAY
    if age_days <= 0:
        return 1.0
    return float(0.5 ** (age_days / _HALF_LIFE_DAYS[candidate.kind]))


def confidence_term(candidate: Candidate) -> float:
    """How far the item can be trusted as a statement about the user, in [0, 1].

    Facts carry the edge confidence the extraction assigned (clamped to [0, 1]); notes follow
    `authored_by`; episodes follow whether they are summarized; exchanges are raw.
    """
    source = candidate.source
    if isinstance(source, FactSource):
        if candidate.confidence is None:
            return MISSING_FACT_CONFIDENCE
        return min(1.0, max(0.0, candidate.confidence))
    if isinstance(source, NoteSource):
        author = _author_value(source.authored_by)
        if author in USER_AUTHORS:
            return USER_NOTE_CONFIDENCE
        if author in PENDING_REVIEW_AUTHORS:
            return PENDING_REVIEW_NOTE_CONFIDENCE
        return MIST_NOTE_CONFIDENCE
    if isinstance(source, EpisodeSource):
        if source.summary_state == "summarized":
            return SUMMARIZED_EPISODE_CONFIDENCE
        return RAW_CONFIDENCE
    return RAW_CONFIDENCE


def window_term(candidate: Candidate, window: TimeWindow) -> float:
    """1 when the item falls in the window, 0 otherwise.

    An episode with a span matches when the span overlaps the window, so a conversation that
    ran across midnight counts for both days; any other item matches when its `when` falls in
    [start, end).

    Raises:
        ValueError: The window has no bounds ("last time" not yet dated by the fetcher), or a
            window bound, the candidate's `when` or its span is naive, which would make the
            comparison raise TypeError or, against another naive value, compare wall clocks.
    """
    if window.start is None or window.end is None:
        raise ValueError(f"window {window.phrase!r} has no bounds to match against")
    _require_aware("window start", window.start)
    _require_aware("window end", window.end)
    if candidate.when is not None:
        _require_aware("candidate when", candidate.when)
    if candidate.span_start is not None:
        _require_aware("candidate span_start", candidate.span_start)
    if candidate.span_end is not None:
        _require_aware("candidate span_end", candidate.span_end)
    if candidate.span_start is not None and candidate.span_end is not None:
        overlaps = candidate.span_start < window.end and candidate.span_end >= window.start
        return 1.0 if overlaps else 0.0
    if candidate.when is None:
        return 0.0
    return 1.0 if window.contains(candidate.when) else 0.0


def order_score(pool_rank: int, ce_rank: int | None, *, k: int = RRF_K) -> float:
    """`1/(k + pool_rank) + 1/(k + ce_rank)`, the second term 0 when not reranked.

    RRF over the two rankings the fetch has: the fused pools and the cross-encoder over the
    shortlist's head. The diagnosis found that fusing a cosine ranking with a cross-encoder
    ranking beats either alone; this applies that to the evidence the fetch has.
    """
    score = 1.0 / (k + pool_rank)
    if ce_rank is not None:
        score += 1.0 / (k + ce_rank)
    return score


def order_candidates(
    shortlisted: Sequence[FusedCandidate],
    rerank_scores: Sequence[float] | None,
    *,
    now: datetime,
    window: TimeWindow | None,
) -> list[ScoredCandidate]:
    """Every candidate given, sorted by rank and numbered by its place (plan v7 A37).

    `rerank_scores[i]` belongs to `shortlisted[i]`: the reranker scores the head of the
    shortlist, in pool order, until the deadline. A reranked candidate's CE rank is its place
    among them by probability, ties by pool rank. The sort key is `(outside, -order_score,
    -recency, -confidence, item_id, kind)`: `outside` is 0 for a candidate inside a dated
    window the user named and 1 otherwise, and 1 for all when the turn names no dated window
    or names "last time", which has no bounds. A candidate the reranker did not reach, or
    every candidate when it reached none, sorts by its pool rank alone, so a fetch the
    reranker missed keeps the pool order without a special case. A facet has no time, so its
    recency is 0.

    Args:
        shortlisted: The shortlist after the fetcher's drops, in pool order; each keeps its
            pool rank.
        rerank_scores: Cross-encoder probabilities in [0, 1] for a prefix of `shortlisted`,
            or None for none.
        now: The turn's instant, for recency.
        window: The turn's temporal window, if the user named one.

    Returns:
        Every candidate of `shortlisted`, none dropped, in sorted order with `position` 1..n.

    Raises:
        ValueError: There are more rerank scores than candidates, or one is outside [0, 1].
    """
    scores = tuple(rerank_scores) if rerank_scores is not None else ()
    if len(scores) > len(shortlisted):
        raise ValueError(f"{len(scores)} rerank scores for {len(shortlisted)} shortlisted items")
    for value in scores:
        if not 0.0 <= value <= 1.0 or math.isnan(value):
            raise ValueError(f"rerank scores must be in [0, 1], got {value}")
    dated = window if window is not None and window.start is not None else None
    by_probability = sorted(
        range(len(scores)), key=lambda index: (-scores[index], shortlisted[index].pool_rank)
    )
    ce_ranks = {index: rank for rank, index in enumerate(by_probability, start=1)}

    keyed: list[tuple[tuple[int, float, float, float, str, str], ScoredCandidate]] = []
    for index, fused in enumerate(shortlisted):
        candidate = fused.candidate
        ce_rank = ce_ranks.get(index)
        score = order_score(fused.pool_rank, ce_rank)
        recency = recency_term(candidate, now)
        time_term = recency if dated is None else window_term(candidate, dated)
        confidence = confidence_term(candidate)
        outside = 0 if dated is not None and time_term >= 1.0 else 1
        key = (outside, -score, -recency, -confidence, candidate.item_id, candidate.kind)
        keyed.append(
            (
                key,
                ScoredCandidate(
                    candidate=candidate,
                    score=score,
                    position=0,
                    pool_rank=fused.pool_rank,
                    ce_rank=ce_rank,
                    p_ce=scores[index] if ce_rank is not None else None,
                    time_term=time_term,
                    window_applied=dated is not None,
                    confidence=confidence,
                ),
            )
        )
    keyed.sort(key=lambda pair: pair[0])
    return [
        replace(scored, position=position) for position, (_, scored) in enumerate(keyed, start=1)
    ]


def _author_value(authored_by: str | None) -> str | None:
    """The plain string of an author, whether a NoteSource carries an AuthoredBy or a str.

    The sets above hold plain values; reducing a member to its value keeps the lookup
    independent of how Enum hashes its members.
    """
    if isinstance(authored_by, AuthoredBy):
        return authored_by.value
    return authored_by


def _require_aware(name: str, value: datetime) -> None:
    if value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware, got {value!r}")
