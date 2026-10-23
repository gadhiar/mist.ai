"""Score fusion for the memory read path: RRF across the pools, then the weighted final score.

The fetcher composes these pure functions around the reranker call it times against the
deadline. `shortlist` fuses the candidate pools and keeps the `rerank_top_n` best; the reranker
scores as many of those, best first, as fit the remaining budget; `score_shortlist` mixes the
scores with recency (or the window-match term) and confidence, drops what falls below
`min_score` and orders the rest. Keeping the arithmetic here, free of I/O and clocks, lets each
term be tested on its own and makes a replay rank the same way.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from backend.vault.models import AuthoredBy

from .config import MemoryConfig
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
        rrf_norm: `rrf` divided by the best `rrf` in the batch, in [0, 1]; the batch's best
            item scores 1. It is the relevance of every item when the reranker scored none,
            and when it scored only a prefix it sets the order of the unscored items, which
            `score_shortlist` scales under the prefix's lowest rerank score.
    """

    candidate: Candidate
    rrf: float
    rrf_norm: float


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    """A candidate with its final score and the terms that made it, for the trace and tests.

    Attributes:
        candidate: The candidate scored.
        score: `rerank_weight * relevance + recency_weight * time_term + confidence_weight *
            confidence`.
        relevance: The rerank score; for an item the reranker did not reach, `rrf_norm`
            (nothing reranked) or `rrf_norm` scaled under the reranked prefix (see
            `score_shortlist`).
        reranked: Whether `relevance` came from the reranker.
        time_term: The window-match term when `window_applied`, otherwise recency.
        window_applied: Whether a dated window replaced recency for this turn.
        confidence: The confidence term.
    """

    candidate: Candidate
    score: float
    relevance: float
    reranked: bool
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
    best = scored[0][1] if scored else 0.0
    return [
        FusedCandidate(candidate=candidate, rrf=rrf, rrf_norm=rrf / best if best > 0 else 0.0)
        for candidate, rrf in scored
    ]


def shortlist(
    candidates: Iterable[Candidate], *, top_n: int, k: int = RRF_K
) -> list[FusedCandidate]:
    """The `top_n` best candidates by RRF, the only ones that go on to the rerank and score.

    `rrf_norm` is computed over the whole batch before the cut, so the best item scores 1.
    """
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


def score_shortlist(
    shortlisted: Sequence[FusedCandidate],
    rerank_scores: Sequence[float] | None,
    *,
    now: datetime,
    window: TimeWindow | None,
    config: MemoryConfig,
) -> list[ScoredCandidate]:
    """Final scores for the shortlist, thresholded at `min_score` and ordered best first.

    `rerank_scores[i]` belongs to `shortlisted[i]`: the reranker scores the head of the
    shortlist, best RRF first, until the deadline (Design point 6, step 2). Relevance then is:

    - no scores (an empty sequence, which is what the fetcher passes when the NullReranker's
      `pairs_within` cuts the batch to nothing or the deadline falls before the first pair, or
      None, still accepted): every item keeps its `rrf_norm`, the fallback the design gives for
      a reranker that cannot run;
    - scores for the first m items, 1 <= m < len: an unscored item i gets
      `min(rerank_scores) * rrf_norm_i / rrf_norm_m`, where item m is the last one scored.

    The design says unscored pairs keep their RRF-normalised score (Design point 6, step 2).
    Taken literally, an RRF score of 0.6 would outrank a reranked item the cross-encoder
    scored 0.4, so a deadline cut would promote exactly the items nobody read.
    The scaling (a lead ruling, 2026-10-23) keeps the RRF order among the unscored items and
    puts every one of them at or under the lowest reranked score; when `rrf_norm_m` is 0 every
    later item's is too, and they get 0. A prefix minimum of 0 sinks the whole tail, normally
    under `min_score`, which is intended: the reranker found nothing relevant that far down.

    The window-match term replaces recency only for a dated window. A "last time" window the
    fetcher could not date carries no span to match, so recency stands in: the previous
    conversation is, by definition, recent.

    Args:
        shortlisted: Output of `shortlist`, in its order.
        rerank_scores: Reranker scores in [0, 1] for a prefix of `shortlisted`, or None.
        now: The turn's instant, for recency.
        window: The turn's temporal window, if the user named one.
        config: Supplies the three weights and `min_score`.

    Returns:
        The items scoring at least `min_score`, by score descending, then item id, then kind.

    Raises:
        ValueError: There are more rerank scores than items, or one is outside [0, 1].
    """
    scores = tuple(rerank_scores) if rerank_scores is not None else ()
    if len(scores) > len(shortlisted):
        raise ValueError(f"{len(scores)} rerank scores for {len(shortlisted)} shortlisted items")
    for value in scores:
        if not 0.0 <= value <= 1.0 or math.isnan(value):
            raise ValueError(f"rerank scores must be in [0, 1], got {value}")
    dated = window if window is not None and window.start is not None else None
    tail_scale = _unscored_scale(shortlisted, scores)

    results: list[ScoredCandidate] = []
    for index, fused in enumerate(shortlisted):
        candidate = fused.candidate
        reranked = index < len(scores)
        relevance = scores[index] if reranked else fused.rrf_norm * tail_scale
        time_term = recency_term(candidate, now) if dated is None else window_term(candidate, dated)
        confidence = confidence_term(candidate)
        score = (
            config.rerank_weight * relevance
            + config.recency_weight * time_term
            + config.confidence_weight * confidence
        )
        if score < config.min_score:
            continue
        results.append(
            ScoredCandidate(
                candidate=candidate,
                score=score,
                relevance=relevance,
                reranked=reranked,
                time_term=time_term,
                window_applied=dated is not None,
                confidence=confidence,
            )
        )
    results.sort(key=lambda item: (-item.score, item.candidate.item_id, item.candidate.kind))
    return results


def _unscored_scale(shortlisted: Sequence[FusedCandidate], scores: Sequence[float]) -> float:
    """Factor applied to `rrf_norm` for items past the reranked prefix.

    1 when nothing was reranked (pure RRF fallback); otherwise `min(scores) / rrf_norm_m` for
    the last scored item m, or 0 when that `rrf_norm` is 0 (every later one is 0 as well, since
    the shortlist is in RRF order).
    """
    if not scores:
        return 1.0
    last_scored_norm = shortlisted[len(scores) - 1].rrf_norm
    if last_scored_norm <= 0:
        return 0.0
    return min(scores) / last_scored_norm


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
