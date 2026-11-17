"""Tests for backend.memory.ranking: RRF fusion, the order terms and the order by rank.

The order is a sort key, not a weighted score, so each test sets up candidates that tie on
every key but the one under test.
"""

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta, timezone

import pytest

from backend.memory.config import MemoryConfig
from backend.memory.models import (
    Candidate,
    EpisodeSource,
    ExchangeSource,
    FacetSource,
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
    order_candidates,
    order_score,
    recency_term,
    rrf_score,
    shortlist,
    window_term,
)
from backend.vault.models import AuthoredBy
from tests.fakes.memory_fakes import FakeClock

NOW = FakeClock().now()  # 2026-10-22 18:00 UTC

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


def _facet(item_id: str) -> Candidate:
    return Candidate(
        kind="facet",
        item_id=item_id,
        text=f"You use a tool {item_id}.",
        view_text=f"You use a tool {item_id}.",
        when=None,
        source=FacetSource(
            facet_key=item_id, subject_id="user", rel_type="USES", object_type="Tool"
        ),
        ranks={"fts": 1},
    )


def _fused(candidate: Candidate, pool_rank: int) -> FusedCandidate:
    return FusedCandidate(candidate=candidate, rrf=1.0 / (RRF_K + pool_rank), pool_rank=pool_rank)


def _pooled(*candidates: Candidate) -> list[FusedCandidate]:
    """The candidates as a shortlist in their given order, pool ranks 1, 2, ..."""
    return [_fused(candidate, rank) for rank, candidate in enumerate(candidates, start=1)]


def _ids(ordered: list) -> list[str]:
    return [scored.candidate.item_id for scored in ordered]


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

    def test_fuse_orders_by_rrf_and_numbers_the_pool_ranks(self) -> None:
        both = _fact("b", {"fts": 2, "vec": 2})
        top_single = _fact("a", {"fts": 1})
        low_single = _fact("c", {"vec": 5})

        fused = fuse([low_single, top_single, both])

        assert [f.candidate.item_id for f in fused] == ["b", "a", "c"]
        assert fused[0].rrf == pytest.approx(2 / 62)
        assert [f.pool_rank for f in fused] == [1, 2, 3]

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

        kept = shortlist(candidates, top_n=MemoryConfig().rerank_top_n)

        assert MemoryConfig().rerank_top_n == 20
        assert [f.candidate.item_id for f in kept] == [f"f{rank:02d}" for rank in range(1, 21)]
        assert [f.pool_rank for f in kept] == list(range(1, 21))


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
            pytest.param(_note("n", "mist-pending-review"), 0.5, id="note-pending-review"),
            pytest.param(_note("n", AuthoredBy.USER), 0.9, id="note-enum-user"),
            pytest.param(_note("n", AuthoredBy.USER_EDIT), 0.9, id="note-enum-user-edit"),
            pytest.param(_note("n", AuthoredBy.MIST), 0.7, id="note-enum-mist"),
            pytest.param(
                _note("n", AuthoredBy.MIST_PENDING_REVIEW), 0.5, id="note-enum-pending-review"
            ),
            pytest.param(_fact("f", confidence=0.65), 0.65, id="fact-edge-confidence"),
            pytest.param(_fact("f", confidence=None), 0.5, id="fact-without-confidence"),
            pytest.param(_fact("f", confidence=1.4), 1.0, id="fact-clamped-high"),
            pytest.param(_episode("e", summary_state="summarized"), 0.7, id="episode-summarized"),
            pytest.param(_episode("e", summary_state="raw"), 0.5, id="episode-raw"),
            pytest.param(_exchange("x"), 0.5, id="exchange"),
            pytest.param(_facet("g"), 0.5, id="facet"),
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

    @pytest.mark.parametrize(
        "candidate",
        [
            pytest.param(_fact("f", when=datetime(2026, 10, 21, 9)), id="naive-when"),
            pytest.param(
                _episode(
                    "e",
                    when=datetime(2026, 10, 21, 9, tzinfo=UTC),
                    span=(datetime(2026, 10, 21, 8), datetime(2026, 10, 21, 9, tzinfo=UTC)),
                ),
                id="naive-span-start",
            ),
            pytest.param(
                _episode(
                    "e",
                    when=datetime(2026, 10, 21, 9, tzinfo=UTC),
                    span=(datetime(2026, 10, 21, 8, tzinfo=UTC), datetime(2026, 10, 21, 9)),
                ),
                id="naive-span-end",
            ),
        ],
    )
    def test_naive_item_timestamp_is_rejected(self, candidate: Candidate) -> None:
        with pytest.raises(ValueError, match="aware"):
            window_term(candidate, YESTERDAY)

    def test_naive_window_bound_is_rejected(self) -> None:
        # TimeWindow refuses naive bounds when built; this one is forced past that check to
        # show window_term does not rely on it.
        window = TimeWindow(
            start=datetime(2026, 10, 21, tzinfo=UTC),
            end=datetime(2026, 10, 22, tzinfo=UTC),
            phrase="yesterday",
        )
        object.__setattr__(window, "start", datetime(2026, 10, 21))

        with pytest.raises(ValueError, match="aware"):
            window_term(_fact("f"), window)

    def test_unbound_previous_conversation_window_is_rejected(self) -> None:
        last_time = TimeWindow(start=None, end=None, phrase="last time", previous_conversation=True)

        with pytest.raises(ValueError, match="bounds"):
            window_term(_fact("f"), last_time)


class TestOrderCandidates:
    def test_order_score_sums_the_pool_and_ce_reciprocal_ranks(self) -> None:
        assert order_score(3, 1) == pytest.approx(1 / 63 + 1 / 61)
        assert order_score(3, None) == pytest.approx(1 / 63)

    def test_order_score_by_hand_with_ce_ties_broken_by_pool_rank(self) -> None:
        # CE probabilities for the first three; b and c tie at 0.9, so b (pool 2) takes CE
        # rank 1 and c (pool 3) CE rank 2; a is CE rank 3 and d was not reranked.
        listed = _pooled(_fact("a"), _fact("b"), _fact("c"), _fact("d"))

        ordered = order_candidates(listed, [0.2, 0.9, 0.9], now=NOW, window=None)

        by_id = {scored.candidate.item_id: scored for scored in ordered}
        assert [(by_id[i].pool_rank, by_id[i].ce_rank) for i in "abcd"] == [
            (1, 3),
            (2, 1),
            (3, 2),
            (4, None),
        ]
        assert [by_id[i].p_ce for i in "abcd"] == [0.2, 0.9, 0.9, None]
        assert by_id["a"].score == pytest.approx(1 / 61 + 1 / 63)
        assert by_id["b"].score == pytest.approx(1 / 62 + 1 / 61)
        assert by_id["c"].score == pytest.approx(1 / 63 + 1 / 62)
        assert by_id["d"].score == pytest.approx(1 / 64)
        assert _ids(ordered) == ["b", "a", "c", "d"]
        assert [scored.position for scored in ordered] == [1, 2, 3, 4]

    def test_no_candidate_is_dropped(self) -> None:
        listed = _pooled(*(_fact(f"f{n}") for n in range(6)))

        ordered = order_candidates(listed, [0.0] * 6, now=NOW, window=None)

        assert sorted(_ids(ordered)) == [f"f{n}" for n in range(6)]

    def test_without_ce_scores_the_order_is_the_pool_order(self) -> None:
        # Recency and confidence differ, but the pool rank alone sets the order score.
        listed = _pooled(
            _fact("z", when=_days_ago(400), confidence=0.1),
            _fact("y", when=NOW, confidence=1.0),
            _note("x", "user"),
        )

        for scores in ((), None):
            ordered = order_candidates(listed, scores, now=NOW, window=None)

            assert _ids(ordered) == ["z", "y", "x"]
            assert all(scored.ce_rank is None and scored.p_ce is None for scored in ordered)

    def test_an_unreranked_candidate_follows_a_reranked_one_of_equal_pool_rank(self) -> None:
        # Pool ranks are positions, so two equal ones only arise by construction; the CE term
        # alone separates them, whatever the ids say.
        listed = [_fused(_fact("b"), 3), _fused(_fact("a"), 3)]

        ordered = order_candidates(listed, [0.0], now=NOW, window=None)

        assert _ids(ordered) == ["b", "a"]

    def test_inside_a_dated_window_first_whatever_the_order_score(self) -> None:
        inside = _fact("inside", when=datetime(2026, 10, 21, 9, tzinfo=UTC))
        outside = _fact("outside", when=NOW)
        listed = _pooled(outside, inside)

        ordered = order_candidates(listed, [0.99, 0.01], now=NOW, window=YESTERDAY)

        assert _ids(ordered) == ["inside", "outside"]
        assert [scored.time_term for scored in ordered] == [1.0, 0.0]
        assert all(scored.window_applied for scored in ordered)

    @pytest.mark.parametrize(
        "window",
        [
            pytest.param(None, id="no-window"),
            pytest.param(
                TimeWindow(start=None, end=None, phrase="last time", previous_conversation=True),
                id="last-time",
            ),
        ],
    )
    def test_no_window_precedence_without_a_dated_window(self, window: TimeWindow | None) -> None:
        inside = _fact("inside", when=datetime(2026, 10, 21, 9, tzinfo=UTC))
        outside = _fact("outside", when=NOW)
        listed = _pooled(outside, inside)

        ordered = order_candidates(listed, [0.99, 0.01], now=NOW, window=window)

        assert _ids(ordered) == ["outside", "inside"]
        assert not any(scored.window_applied for scored in ordered)

    def test_equal_order_scores_fall_to_recency_then_confidence_then_id_then_kind(self) -> None:
        # Every candidate has pool rank 5 and no CE term, so the order score ties.
        listed = [
            _fused(_fact("old", when=_days_ago(90), confidence=0.9), 5),
            _fused(_fact("new-low", when=NOW, confidence=0.2), 5),
            _fused(_fact("new-high", when=NOW, confidence=0.9), 5),
            _fused(_fact("tie-b", when=_days_ago(30), confidence=0.5), 5),
            _fused(_fact("tie-a", when=_days_ago(30), confidence=0.5), 5),
            # No time, confidence 0.5: these three tie on everything but id and kind.
            _fused(_fact("same", when=None, confidence=0.5), 5),
            _fused(_exchange("same", when=None), 5),
            _fused(_facet("group"), 5),
        ]

        ordered = order_candidates(listed, (), now=NOW, window=None)

        assert [(s.candidate.item_id, s.candidate.kind) for s in ordered] == [
            ("new-high", "fact"),
            ("new-low", "fact"),
            ("tie-a", "fact"),
            ("tie-b", "fact"),
            ("old", "fact"),
            ("group", "facet"),
            ("same", "exchange"),
            ("same", "fact"),
        ]

    def test_a_facet_has_no_recency(self) -> None:
        [scored] = order_candidates(_pooled(_facet("group")), (), now=NOW, window=YESTERDAY)

        assert scored.time_term == 0.0 and scored.confidence == 0.5

    def test_more_rerank_scores_than_items_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="rerank scores"):
            order_candidates(_pooled(_fact("a")), [0.5, 0.5], now=NOW, window=None)

    @pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan")])
    def test_rerank_score_outside_unit_interval_is_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            order_candidates(_pooled(_fact("a")), [bad], now=NOW, window=None)
