"""Tests for backend.memory.ranking: RRF fusion, the score terms, the threshold and the order.

Each term is isolated by a config whose weights put all of the score on that one term, so a
change to another term cannot hide a regression in the one under test.
"""

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta, timezone

import pytest

from backend.memory.config import MemoryConfig
from backend.memory.models import (
    Candidate,
    EpisodeSource,
    ExchangeSource,
    FactSource,
    NoteSource,
    SummaryState,
    TimeWindow,
)
from backend.memory.ranking import (
    EPISODE_HALF_LIFE_DAYS,
    FACT_HALF_LIFE_DAYS,
    RRF_K,
    FusedCandidate,
    confidence_term,
    fuse,
    recency_term,
    rrf_score,
    score_shortlist,
    shortlist,
    window_term,
)
from tests.fakes.memory_fakes import FakeClock

NOW = FakeClock().now()  # 2026-10-22 18:00 UTC

RELEVANCE_ONLY = MemoryConfig(
    rerank_weight=1.0, recency_weight=0.0, confidence_weight=0.0, min_score=0.0
)
RECENCY_ONLY = MemoryConfig(
    rerank_weight=0.0, recency_weight=1.0, confidence_weight=0.0, min_score=0.0
)
CONFIDENCE_ONLY = MemoryConfig(
    rerank_weight=0.0, recency_weight=0.0, confidence_weight=1.0, min_score=0.0
)

YESTERDAY = TimeWindow(
    start=datetime(2026, 10, 21, tzinfo=UTC),
    end=datetime(2026, 10, 22, tzinfo=UTC),
    phrase="yesterday",
)


def _fact(
    item_id: str,
    ranks: Mapping[str, int] | None = None,
    *,
    when: datetime | None = NOW,
    confidence: float | None = 0.8,
) -> Candidate:
    return Candidate(
        kind="fact",
        item_id=item_id,
        text=f"fact {item_id}",
        when=when,
        source=FactSource(
            fact_key=item_id, row_key=f"row-{item_id}", recorded_at=None, source_utterance_id=None
        ),
        ranks=ranks if ranks is not None else {"fts": 1},
        confidence=confidence,
    )


def _episode(
    item_id: str,
    *,
    when: datetime | None = NOW,
    summary_state: SummaryState = "summarized",
    span: tuple[datetime, datetime] | None = None,
) -> Candidate:
    return Candidate(
        kind="episode",
        item_id=item_id,
        text=f"episode {item_id}",
        when=when,
        source=EpisodeSource(
            episode_id=item_id,
            conversation_id="conv-1",
            first_event_id="ev-1",
            last_event_id="ev-2",
            summary_state=summary_state,
        ),
        ranks={"episodes": 1},
        span_start=span[0] if span else None,
        span_end=span[1] if span else None,
    )


def _exchange(item_id: str, *, when: datetime | None = NOW) -> Candidate:
    return Candidate(
        kind="exchange",
        item_id=item_id,
        text=f"exchange {item_id}",
        when=when,
        source=ExchangeSource(event_id=item_id, conversation_id=None, episode_id=None),
        ranks={"exchanges": 1},
    )


def _note(item_id: str, authored_by: str | None, *, when: datetime | None = NOW) -> Candidate:
    return Candidate(
        kind="note",
        item_id=item_id,
        text=f"note {item_id}",
        when=when,
        source=NoteSource(path=f"notes/{item_id}.md", heading=None, authored_by=authored_by),
        ranks={"sidecar": 1},
    )


def _days_ago(days: float) -> datetime:
    return NOW - timedelta(days=days)


def _fused(candidate: Candidate, rrf_norm: float = 1.0) -> FusedCandidate:
    return FusedCandidate(candidate=candidate, rrf=rrf_norm, rrf_norm=rrf_norm)


class TestRrf:
    def test_k_is_60(self) -> None:
        assert RRF_K == 60

    def test_score_sums_reciprocal_ranks_over_pools(self) -> None:
        assert rrf_score({"fts": 1, "vec": 3}) == pytest.approx(1 / 61 + 1 / 63)

    def test_no_pools_scores_zero(self) -> None:
        assert rrf_score({}) == 0.0

    def test_rank_below_one_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="rank"):
            rrf_score({"fts": 0})

    def test_fuse_orders_by_rrf_and_normalises_to_the_best(self) -> None:
        both = _fact("b", {"fts": 2, "vec": 2})
        top_single = _fact("a", {"fts": 1})
        low_single = _fact("c", {"vec": 5})

        fused = fuse([low_single, top_single, both])

        assert [f.candidate.item_id for f in fused] == ["b", "a", "c"]
        assert fused[0].rrf == pytest.approx(2 / 62)
        assert fused[0].rrf_norm == 1.0
        assert fused[1].rrf_norm == pytest.approx((1 / 61) / (2 / 62))
        assert fused[2].rrf_norm == pytest.approx((1 / 65) / (2 / 62))

    def test_fuse_merges_one_item_offered_twice_keeping_the_best_rank_per_pool(self) -> None:
        first = _fact("a", {"fts": 4, "vec": 2})
        second = _fact("a", {"fts": 1, "anchor": 3})

        fused = fuse([first, second])

        assert len(fused) == 1
        assert dict(fused[0].candidate.ranks) == {"fts": 1, "vec": 2, "anchor": 3}
        assert fused[0].rrf == pytest.approx(1 / 61 + 1 / 62 + 1 / 63)

    def test_fuse_keeps_different_kinds_with_one_id_apart(self) -> None:
        fused = fuse([_fact("x"), _exchange("x")])

        assert sorted(f.candidate.kind for f in fused) == ["exchange", "fact"]

    def test_fuse_breaks_rrf_ties_on_item_id(self) -> None:
        fused = fuse([_fact("m"), _fact("c"), _fact("x")])

        assert [f.candidate.item_id for f in fused] == ["c", "m", "x"]

    def test_fuse_of_nothing_is_empty(self) -> None:
        assert fuse([]) == []

    def test_shortlist_keeps_the_top_n_best_first(self) -> None:
        candidates = [_fact(f"f{rank:02d}", {"fts": rank}) for rank in range(25, 0, -1)]

        kept = shortlist(candidates, top_n=RELEVANCE_ONLY.rerank_top_n)

        assert RELEVANCE_ONLY.rerank_top_n == 20
        assert [f.candidate.item_id for f in kept] == [f"f{rank:02d}" for rank in range(1, 21)]


class TestRecencyTerm:
    @pytest.mark.parametrize(
        ("candidate", "expected"),
        [
            pytest.param(_fact("f", when=_days_ago(90)), 0.5, id="fact-one-half-life"),
            pytest.param(_fact("f", when=_days_ago(180)), 0.25, id="fact-two-half-lives"),
            pytest.param(_episode("e", when=_days_ago(14)), 0.5, id="episode-one-half-life"),
            pytest.param(_exchange("x", when=_days_ago(28)), 0.25, id="exchange-two-half-lives"),
            pytest.param(_note("n", "user", when=_days_ago(90)), 0.5, id="note-fact-half-life"),
            pytest.param(_fact("f", when=NOW), 1.0, id="now"),
            pytest.param(_fact("f", when=NOW + timedelta(hours=1)), 1.0, id="future-clamped"),
            pytest.param(_fact("f", when=None), 0.0, id="no-timestamp"),
        ],
    )
    def test_exponential_decay_by_kind(self, candidate: Candidate, expected: float) -> None:
        assert recency_term(candidate, NOW) == pytest.approx(expected)

    def test_half_lives(self) -> None:
        assert FACT_HALF_LIFE_DAYS == 90.0
        assert EPISODE_HALF_LIFE_DAYS == 14.0

    def test_age_is_measured_across_zones(self) -> None:
        # 20:00 at +02:00 is 18:00 UTC, so this is exactly one episode half-life before NOW.
        when = datetime(2026, 10, 8, 20, 0, tzinfo=timezone(timedelta(hours=2)))

        assert recency_term(_episode("e", when=when), NOW) == pytest.approx(0.5)

    def test_naive_timestamp_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="aware"):
            recency_term(_fact("f", when=datetime(2026, 10, 1)), NOW)


class TestConfidenceTerm:
    @pytest.mark.parametrize(
        ("candidate", "expected"),
        [
            pytest.param(_note("n", "user"), 0.9, id="note-user"),
            pytest.param(_note("n", "user-edit"), 0.9, id="note-user-edit"),
            pytest.param(_note("n", "mist"), 0.7, id="note-mist"),
            pytest.param(_note("n", None), 0.7, id="note-unknown-author"),
            pytest.param(_fact("f", confidence=0.65), 0.65, id="fact-edge-confidence"),
            pytest.param(_fact("f", confidence=None), 0.5, id="fact-without-confidence"),
            pytest.param(_fact("f", confidence=1.4), 1.0, id="fact-clamped-high"),
            pytest.param(_episode("e", summary_state="summarized"), 0.7, id="episode-summarized"),
            pytest.param(_episode("e", summary_state="raw"), 0.5, id="episode-raw"),
            pytest.param(_exchange("x"), 0.5, id="exchange"),
        ],
    )
    def test_confidence_by_kind_and_source(self, candidate: Candidate, expected: float) -> None:
        assert confidence_term(candidate) == pytest.approx(expected)


class TestWindowTerm:
    @pytest.mark.parametrize(
        ("candidate", "expected"),
        [
            pytest.param(_fact("f", when=datetime(2026, 10, 21, 9, tzinfo=UTC)), 1.0, id="inside"),
            pytest.param(_fact("f", when=datetime(2026, 10, 22, tzinfo=UTC)), 0.0, id="at-end"),
            pytest.param(_fact("f", when=datetime(2026, 10, 21, tzinfo=UTC)), 1.0, id="at-start"),
            pytest.param(_fact("f", when=_days_ago(5)), 0.0, id="before"),
            pytest.param(_fact("f", when=None), 0.0, id="no-timestamp"),
            pytest.param(
                _episode(
                    "e",
                    when=datetime(2026, 10, 22, 1, tzinfo=UTC),
                    span=(
                        datetime(2026, 10, 21, 23, tzinfo=UTC),
                        datetime(2026, 10, 22, 1, tzinfo=UTC),
                    ),
                ),
                1.0,
                id="episode-span-overlaps",
            ),
            pytest.param(
                _episode(
                    "e",
                    when=datetime(2026, 10, 20, 12, tzinfo=UTC),
                    span=(
                        datetime(2026, 10, 20, 10, tzinfo=UTC),
                        datetime(2026, 10, 20, 12, tzinfo=UTC),
                    ),
                ),
                0.0,
                id="episode-span-before",
            ),
        ],
    )
    def test_match_is_one_inside_the_window_and_zero_outside(
        self, candidate: Candidate, expected: float
    ) -> None:
        assert window_term(candidate, YESTERDAY) == expected

    def test_unbound_previous_conversation_window_is_rejected(self) -> None:
        last_time = TimeWindow(start=None, end=None, phrase="last time", previous_conversation=True)

        with pytest.raises(ValueError, match="bounds"):
            window_term(_fact("f"), last_time)


class TestFinalScore:
    def test_relevance_only_uses_rerank_scores_then_rrf_for_the_unscored_tail(self) -> None:
        listed = [_fused(_fact("a"), 1.0), _fused(_fact("b"), 0.6), _fused(_fact("c"), 0.4)]

        scored = score_shortlist(listed, [0.3, 0.9], now=NOW, window=None, config=RELEVANCE_ONLY)

        assert [(s.candidate.item_id, s.score, s.reranked) for s in scored] == [
            ("b", pytest.approx(0.9), True),
            ("c", pytest.approx(0.4), False),
            ("a", pytest.approx(0.3), True),
        ]

    def test_no_rerank_scores_falls_back_to_rrf_for_every_item(self) -> None:
        listed = [_fused(_fact("a"), 1.0), _fused(_fact("b"), 0.5)]

        scored = score_shortlist(listed, None, now=NOW, window=None, config=RELEVANCE_ONLY)

        assert [(s.candidate.item_id, s.score) for s in scored] == [("a", 1.0), ("b", 0.5)]
        assert not any(s.reranked for s in scored)

    def test_recency_only(self) -> None:
        listed = [_fused(_fact("old", when=_days_ago(90))), _fused(_episode("new", when=NOW))]

        scored = score_shortlist(listed, None, now=NOW, window=None, config=RECENCY_ONLY)

        assert [(s.candidate.item_id, s.score) for s in scored] == [
            ("new", pytest.approx(1.0)),
            ("old", pytest.approx(0.5)),
        ]
        assert not any(s.window_applied for s in scored)

    def test_confidence_only(self) -> None:
        listed = [_fused(_note("mine", "user")), _fused(_note("mist", "mist"))]

        scored = score_shortlist(listed, None, now=NOW, window=None, config=CONFIDENCE_ONLY)

        assert [(s.candidate.item_id, s.score) for s in scored] == [
            ("mine", pytest.approx(0.9)),
            ("mist", pytest.approx(0.7)),
        ]

    def test_window_match_replaces_recency_when_a_window_is_present(self) -> None:
        inside_but_older = _fact("inside", when=datetime(2026, 10, 21, 9, tzinfo=UTC))
        outside_but_newer = _fact("outside", when=NOW)
        listed = [_fused(outside_but_newer), _fused(inside_but_older)]

        scored = score_shortlist(listed, None, now=NOW, window=YESTERDAY, config=RECENCY_ONLY)

        assert [(s.candidate.item_id, s.score) for s in scored] == [
            ("inside", 1.0),
            ("outside", 0.0),
        ]
        assert all(s.window_applied for s in scored)

    def test_a_bound_last_time_window_is_matched_like_any_window(self) -> None:
        conversation = TimeWindow(
            start=datetime(2026, 10, 20, 9, tzinfo=UTC),
            end=datetime(2026, 10, 20, 10, tzinfo=UTC),
            phrase="last time",
            previous_conversation=True,
        )
        listed = [_fused(_exchange("then", when=datetime(2026, 10, 20, 9, 30, tzinfo=UTC)))]

        scored = score_shortlist(listed, None, now=NOW, window=conversation, config=RECENCY_ONLY)

        assert scored[0].score == 1.0
        assert scored[0].window_applied

    def test_an_unbound_last_time_window_falls_back_to_recency(self) -> None:
        last_time = TimeWindow(start=None, end=None, phrase="last time", previous_conversation=True)
        listed = [_fused(_episode("e", when=_days_ago(14)))]

        scored = score_shortlist(listed, None, now=NOW, window=last_time, config=RECENCY_ONLY)

        assert scored[0].score == pytest.approx(0.5)
        assert not scored[0].window_applied

    def test_default_weights_combine_the_three_terms(self) -> None:
        config = MemoryConfig(min_score=0.0)
        listed = [_fused(_episode("e", when=_days_ago(14), summary_state="raw"), 0.2)]

        scored = score_shortlist(listed, [0.75], now=NOW, window=None, config=config)

        assert (config.rerank_weight, config.recency_weight, config.confidence_weight) == (
            0.8,
            0.1,
            0.1,
        )
        assert scored[0].score == pytest.approx(0.8 * 0.75 + 0.1 * 0.5 + 0.1 * 0.5)
        assert (scored[0].relevance, scored[0].time_term, scored[0].confidence) == (
            0.75,
            pytest.approx(0.5),
            0.5,
        )

    def test_items_below_min_score_are_dropped_and_equal_ones_kept(self) -> None:
        config = MemoryConfig(
            rerank_weight=1.0, recency_weight=0.0, confidence_weight=0.0, min_score=0.5
        )
        listed = [_fused(_fact("below"), 0.49), _fused(_fact("equal"), 0.5), _fused(_fact("a"))]

        scored = score_shortlist(listed, None, now=NOW, window=None, config=config)

        assert [s.candidate.item_id for s in scored] == ["a", "equal"]

    def test_equal_scores_are_ordered_by_item_id(self) -> None:
        listed = [_fused(_fact(item_id)) for item_id in ("q", "b", "k")]

        scored = score_shortlist(
            listed, [0.5, 0.5, 0.5], now=NOW, window=None, config=MemoryConfig()
        )

        assert [s.candidate.item_id for s in scored] == ["b", "k", "q"]

    def test_more_rerank_scores_than_items_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="rerank scores"):
            score_shortlist(
                [_fused(_fact("a"))], [0.5, 0.5], now=NOW, window=None, config=RELEVANCE_ONLY
            )

    @pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan")], ids=["negative", "above", "nan"])
    def test_rerank_score_outside_unit_interval_is_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            score_shortlist(
                [_fused(_fact("a"))], [bad], now=NOW, window=None, config=RELEVANCE_ONLY
            )
