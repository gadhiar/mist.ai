"""Tests for backend.memory.fetcher: the per-turn fetch, its deadline, prefetch and lists.

The fetcher runs against a real FactIndex on a tmp database, real small executors and the
shared fakes. Slow and failing pools come from `HookedIndex`, which runs a hook before the
real search. A hook that blocks on a `threading.Event` holds an executor thread past the
deadline in real time; one that advances the FakeClock returns a result after the deadline by
the fetcher's clock, which the fetcher must discard.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest

from backend.errors import EmbeddingError, MemoryIndexError, MemoryRerankError
from backend.knowledge.models import graph_ref, vault_ref
from backend.memory.block import HEADER, MemoryBlockBuilder
from backend.memory.config import MemoryConfig
from backend.memory.fact_index import FactGroup, FactIndex, IndexedFact
from backend.memory.fetcher import (
    STAGE_CANDIDATE_CAP,
    FetcherExecutors,
    MemoryFetcher,
    SystemClock,
    jaccard,
    normalise,
)
from backend.memory.index_db import open_index, open_reader
from backend.memory.models import (
    Candidate,
    FactRow,
    MemoryRequest,
    NoteSource,
    ServingProfile,
    fact_key,
)
from backend.memory.rerank import NullReranker
from tests.fakes.memory_fakes import FakeClock, FakeEmbedder, FakeReranker, FakeTokenCounter

_MODEL = "all-MiniLM-L6-v2"
_NOW = datetime(2026, 10, 22, 18, 0, tzinfo=UTC)  # FakeClock's default start
_WAIT_S = 5.0
_ROOMY_MS = 5000  # a deadline no healthy test run reaches, for tests not about the deadline


def _fact(
    object_name: str,
    *,
    rel_type: str = "WORKS_AT",
    row_key: str | None = None,
    utterance: str | None = None,
    valid_from: str | None = None,
    valid_to: str | None = None,
    object_type: str | None = "Organization",
) -> FactRow:
    object_id = object_name.lower()
    return FactRow(
        row_key=row_key or f"rk-{rel_type}-{object_id}",
        fact_key=fact_key("user", rel_type, object_id),
        subject_id="user",
        subject_name="User",
        subject_type="User",
        rel_type=rel_type,
        object_id=object_id,
        object_name=object_name,
        object_type=object_type,
        valid_from=valid_from,
        valid_to=valid_to,
        recorded_at="2026-01-01T00:00:00+00:00",
        confidence=0.9,
        source_utterance_id=utterance or f"utt-{object_id}",
    )


def _key(object_name: str, rel_type: str = "WORKS_AT") -> str:
    return fact_key("user", rel_type, object_name.lower())


def _request(
    text: str,
    *,
    session: str = "s1",
    turns: tuple[str, ...] = (),
    budget: int | None = None,
    serving: ServingProfile = ServingProfile("full", "top"),
) -> MemoryRequest:
    return MemoryRequest(
        session_id=session,
        conversation_id="c1",
        query_text=text,
        now=_NOW,
        recent_user_turns=turns,
        turn_index=1,
        budget_tokens=budget,
        serving=serving,
    )


class HookedIndex:
    """A FactIndex whose searches run `hooks[name]` first and are recorded by name."""

    def __init__(self, inner: FactIndex) -> None:
        self.inner = inner
        self.hooks: dict[str, Callable[[], None]] = {}
        self.calls: list[str] = []

    def _enter(self, name: str) -> None:
        self.calls.append(name)
        hook = self.hooks.get(name)
        if hook is not None:
            hook()

    def search_fts(self, query_text: str, *, limit: int = 20) -> list[FactGroup]:
        self._enter("search_fts")
        return self.inner.search_fts(query_text, limit=limit)

    def search_vector(self, embedding: Sequence[float], *, limit: int = 20) -> list[FactGroup]:
        self._enter("search_vector")
        return self.inner.search_vector(embedding, limit=limit)

    def search_names(self, texts: Sequence[str], *, limit: int = 20) -> list[FactGroup]:
        self._enter("search_names")
        return self.inner.search_names(texts, limit=limit)

    def siblings(
        self, subject_id: str, rel_type: str, source_utterance_id: str, *, now: datetime
    ) -> tuple[IndexedFact, ...]:
        self._enter("siblings")
        return self.inner.siblings(subject_id, rel_type, source_utterance_id, now=now)


class ScriptedReranker:
    """Scores a text by the first scripted word it contains; 0 when it contains none."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def score(
        self, query: str, texts: Sequence[str], budget_ms: float | None = None
    ) -> list[float]:
        self.calls.append((query, tuple(texts)))
        return [self._one(text) for text in texts]

    def _one(self, text: str) -> float:
        lowered = text.lower()
        for word, value in self.scores.items():
            if word in lowered:
                return value
        return 0.0

    def pairs_within(self, budget_ms: float) -> int:
        return 1000


class FailingReranker:
    def score(
        self, query: str, texts: Sequence[str], budget_ms: float | None = None
    ) -> list[float]:
        raise MemoryRerankError("reranker down")

    def pairs_within(self, budget_ms: float) -> int:
        return 100


class FailingEmbedder:
    def generate_embedding(self, text: str) -> list[float]:
        raise EmbeddingError("embedder down")

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        raise EmbeddingError("embedder down")


class StaticSuppression:
    def __init__(self, keys: frozenset[str]) -> None:
        self.keys = keys

    def suppressed_fact_keys(self) -> frozenset[str]:
        return self.keys


class StaticNotes:
    def __init__(self, candidates: list[Candidate]) -> None:
        self.candidates = candidates

    async def gather(
        self,
        query_text: str,
        query_embedding: Sequence[float],
        *,
        profile_path: str | None = None,
    ) -> list[Candidate]:
        return list(self.candidates)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "memory_index.db"


@pytest.fixture
def fact_index(db_path: Path) -> Iterator[FactIndex]:
    writer = open_index(db_path, embedding_model_id=_MODEL)
    reader = open_reader(db_path, embedding_model_id=_MODEL)
    yield FactIndex(writer, FakeEmbedder(), tz=UTC, reader=reader)
    reader.close()
    writer.close()


@pytest.fixture
def index(fact_index: FactIndex) -> HookedIndex:
    return HookedIndex(fact_index)


@pytest.fixture
def executors() -> Iterator[FetcherExecutors]:
    index_pool = ThreadPoolExecutor(2, thread_name_prefix="memory-index")
    model_pool = ThreadPoolExecutor(1, thread_name_prefix="memory-model")
    yield FetcherExecutors(index=index_pool, model=model_pool)
    index_pool.shutdown(wait=True, cancel_futures=True)
    model_pool.shutdown(wait=True, cancel_futures=True)


@pytest.fixture
def release() -> Iterator[threading.Event]:
    """An event blocking hooks wait on; set at teardown so no executor thread is left blocked."""
    event = threading.Event()
    yield event
    event.set()


def _config(**overrides: object) -> MemoryConfig:
    values: dict[str, object] = {
        "min_score": 0.2,
        "rerank_top_n": 20,
        "block_tokens": 300,
        "deadline_ms": _ROOMY_MS,
        "list_max_items": 10,
    }
    values.update(overrides)
    return MemoryConfig(**values)  # type: ignore[arg-type]


def _fetcher(
    index: HookedIndex,
    executors: FetcherExecutors,
    *,
    config: MemoryConfig | None = None,
    clock: object | None = None,
    reranker: object | None = None,
    embedder: object | None = None,
    index_state: Callable[[], str | None] = lambda: None,
    notes: StaticNotes | None = None,
    suppression: StaticSuppression | None = None,
    pool_size: int = 20,
) -> MemoryFetcher:
    config = config or _config()
    return MemoryFetcher(
        index=index,
        embedder=embedder or FakeEmbedder(),  # type: ignore[arg-type]
        reranker=reranker or FakeReranker(),  # type: ignore[arg-type]
        block_builder=MemoryBlockBuilder(
            counter=FakeTokenCounter(), config=config, placement="system_tail"
        ),
        config=config,
        executors=executors,
        clock=clock or FakeClock(),  # type: ignore[arg-type]
        index_state=index_state,  # type: ignore[arg-type]
        notes=notes,
        suppression=suppression,
        pool_size=pool_size,
    )


def _block(event: threading.Event) -> Callable[[], None]:
    def hook() -> None:
        event.wait(_WAIT_S)

    return hook


# -- the fetch --------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_injects_the_matching_fact_with_its_refs(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    fact_index.upsert([_fact("Tidewater Survey"), _fact("Harbor Bakery", rel_type="PREFERS")])
    reranker = FakeReranker()

    result = await _fetcher(index, executors, reranker=reranker).fetch(
        _request("does the user still work at tidewater survey")
    )

    assert not result.degraded
    assert result.items[0].item_id == _key("Tidewater Survey")
    assert result.block_text is not None and "Tidewater Survey" in result.block_text
    assert result.reranked == len(reranker.calls[0][1]) > 0
    assert result.timings.prefetch_hit is False
    stage = result.stage
    assert stage.ran and stage.prefetch_reuse == "none"
    assert (stage.level, stage.tier) == ("full", "top")
    first = stage.candidates[0]
    assert first.disposition == "injected" and first.item == 1 and first.source == "graph"
    assert first.ref == graph_ref("user", "WORKS_AT", "tidewater survey")
    assert first.alt_ref == graph_ref("tidewater survey", "WORKS_AT", "user")


@pytest.mark.asyncio
async def test_rerank_is_cut_to_the_pairs_that_fit(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    fact_index.upsert([_fact(f"Survey Office {n}") for n in range(5)])
    reranker = FakeReranker(pairs_per_ms=2 / _ROOMY_MS)  # two pairs fit the whole deadline

    result = await _fetcher(index, executors, reranker=reranker).fetch(
        _request("which survey office")
    )

    assert len(reranker.calls) == 1 and len(reranker.calls[0][1]) == 2
    assert result.reranked == result.stage.reranked == 2


@pytest.mark.asyncio
async def test_the_outcome_carries_the_fused_list_and_the_scored_shortlist(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    rows = [_fact(f"Survey Office {n}") for n in range(6)]
    fact_index.upsert(rows)
    reranker = FakeReranker()
    fetcher = _fetcher(index, executors, reranker=reranker, config=_config(rerank_top_n=4))

    outcome = await fetcher.fetch_outcome(_request("which survey office"))

    # Every fused candidate, in RRF order, before the top_n cut; its head is what was scored.
    assert len(outcome.fused) == len(rows)
    rrf = [fused.rrf for fused in outcome.fused]
    assert rrf == sorted(rrf, reverse=True)
    [(_, scored_texts, _)] = reranker.calls
    assert [fused.candidate.text for fused in outcome.fused[:4]] == list(scored_texts)
    assert outcome.result.reranked == 4
    # The scored shortlist, unthresholded, in the order and with the scores of the stage.
    refs = {row.fact_key: graph_ref("user", "WORKS_AT", row.object_id) for row in rows}
    by_rank = sorted(outcome.result.stage.candidates, key=lambda candidate: candidate.rank)
    assert [(refs[s.candidate.item_id], s.score) for s in outcome.ranked] == [
        (candidate.ref, candidate.score) for candidate in by_rank
    ]
    assert {s.candidate.item_id for s in outcome.ranked} == {
        fused.candidate.item_id for fused in outcome.fused[:4]
    }


@pytest.mark.asyncio
async def test_an_outcome_without_a_stage2_carries_no_candidates(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    fact_index.upsert([_fact("Tidewater Survey")])
    fetcher = _fetcher(index, executors, index_state=lambda: "index_rebuilding")

    outcome = await fetcher.fetch_outcome(_request("tidewater survey"))

    assert outcome.result.degraded_reason == "index_rebuilding"
    assert (outcome.fused, outcome.ranked) == ((), ())


@pytest.mark.asyncio
async def test_each_fact_is_offered_once_by_its_version_current_at_now(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    fact_index.upsert(
        [
            _fact(
                "Tidewater Survey", row_key="rk-old", valid_from="2023-01-01", valid_to="2025-01-01"
            ),
            _fact("Tidewater Survey", row_key="rk-new", valid_from="2025-01-01"),
            _fact("Future Labs", valid_from="2027-06-01"),
        ]
    )

    result = await _fetcher(index, executors).fetch(_request("tidewater survey future labs"))

    tidewater = [item for item in result.items if item.item_id == _key("Tidewater Survey")]
    assert len(tidewater) == 1
    assert tidewater[0].source.row_key == "rk-new"  # type: ignore[union-attr]
    graph_refs = {candidate.ref for candidate in result.stage.candidates}
    assert graph_ref("user", "WORKS_AT", "future labs") not in graph_refs
    assert len([c for c in result.stage.candidates if c.source == "graph"]) == 1


@pytest.mark.asyncio
async def test_serving_off_skips_without_work(
    index: HookedIndex, executors: FetcherExecutors
) -> None:
    embedder = FakeEmbedder()
    result = await _fetcher(index, executors, embedder=embedder).fetch(
        _request("anything", serving=ServingProfile("off", "top"))
    )

    assert result.stage.ran is False and result.stage.skip_reason == "off"
    assert not result.degraded and result.items == ()
    assert index.calls == [] and embedder.calls == []


@pytest.mark.parametrize(
    "request_kwargs",
    [{"budget": 0}, {"serving": ServingProfile("low", "interaction")}],
    ids=["request-budget-0", "low-level-budget-0"],
)
@pytest.mark.asyncio
async def test_zero_budget_does_no_work(
    index: HookedIndex, executors: FetcherExecutors, request_kwargs: dict[str, object]
) -> None:
    embedder = FakeEmbedder()
    state_calls: list[int] = []

    def state() -> None:
        state_calls.append(1)

    request = _request("anything", **request_kwargs)  # type: ignore[arg-type]
    fetcher = _fetcher(
        index, executors, embedder=embedder, index_state=state, config=_config(block_tokens_low=0)
    )

    result = await fetcher.fetch(request)

    assert result.stage.skip_reason == "zero_budget" and result.stage.ran is False
    assert result.budget_tokens == 0 and result.block_text is None
    assert index.calls == [] and embedder.calls == [] and state_calls == []


@pytest.mark.asyncio
async def test_a_rebuild_in_progress_returns_an_empty_index_rebuilding_result(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    fact_index.upsert([_fact("Tidewater Survey")])

    result = await _fetcher(index, executors, index_state=lambda: "index_rebuilding").fetch(
        _request("tidewater survey")
    )

    assert result.degraded_reason == "index_rebuilding" and result.items == ()
    assert result.stage.ran and index.calls == []


@pytest.mark.asyncio
async def test_the_index_state_is_checked_on_every_fetch(
    index: HookedIndex, executors: FetcherExecutors
) -> None:
    calls: list[int] = []

    def state() -> None:
        calls.append(1)

    fetcher = _fetcher(index, executors, index_state=state)
    await fetcher.fetch(_request("one"))
    await fetcher.fetch(_request("one"))

    assert len(calls) == 2


@pytest.mark.asyncio
async def test_an_unreadable_index_state_returns_index_error(
    index: HookedIndex, executors: FetcherExecutors
) -> None:
    def state() -> None:
        raise MemoryIndexError("database is locked")

    result = await _fetcher(index, executors, index_state=state).fetch(_request("anything"))

    assert result.degraded_reason == "index_error" and result.items == ()


@pytest.mark.asyncio
async def test_a_failing_pool_returns_index_error(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    fact_index.upsert([_fact("Tidewater Survey")])

    def fail() -> None:
        raise MemoryIndexError("disk I/O error")

    index.hooks["search_fts"] = fail
    result = await _fetcher(index, executors).fetch(_request("tidewater survey"))

    assert result.degraded_reason == "index_error" and result.items == ()


@pytest.mark.asyncio
async def test_a_failing_reranker_keeps_the_rrf_scores(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    fact_index.upsert([_fact("Tidewater Survey")])

    result = await _fetcher(index, executors, reranker=FailingReranker()).fetch(
        _request("tidewater survey")
    )

    assert result.degraded_reason == "reranker_unavailable"
    assert result.reranked == 0
    assert [item.item_id for item in result.items] == [_key("Tidewater Survey")]


@pytest.mark.parametrize(("backend", "reason"), [("local", "reranker_unavailable"), ("off", None)])
@pytest.mark.asyncio
async def test_a_null_reranker_is_degraded_only_when_a_model_was_wanted(
    fact_index: FactIndex,
    index: HookedIndex,
    executors: FetcherExecutors,
    backend: str,
    reason: str | None,
) -> None:
    fact_index.upsert([_fact("Tidewater Survey")])
    fetcher = _fetcher(
        index, executors, reranker=NullReranker(), config=_config(rerank_backend=backend)
    )

    result = await fetcher.fetch(_request("tidewater survey"))

    assert result.degraded_reason == reason
    assert result.items and result.reranked == 0


@pytest.mark.asyncio
async def test_a_failing_embedder_costs_only_the_vector_and_note_pools(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    fact_index.upsert([_fact("Tidewater Survey")])

    result = await _fetcher(index, executors, embedder=FailingEmbedder()).fetch(
        _request("tidewater survey")
    )

    assert not result.degraded
    assert [item.item_id for item in result.items] == [_key("Tidewater Survey")]
    assert "search_vector" not in index.calls


@pytest.mark.parametrize("error", [TypeError("bad fake"), sqlite3.OperationalError("boom")])
@pytest.mark.asyncio
async def test_an_unexpected_failure_is_contained_as_an_error_stage(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors, error: Exception
) -> None:
    fact_index.upsert([_fact("Tidewater Survey")])

    def fail() -> None:
        raise error

    index.hooks["search_fts"] = fail
    result = await _fetcher(index, executors).fetch(_request("tidewater survey"))

    assert result.stage.ran is False and result.stage.skip_reason == "error"
    assert not result.degraded and result.items == ()


@pytest.mark.asyncio
async def test_a_note_candidate_is_recorded_under_its_vault_ref(
    index: HookedIndex, executors: FetcherExecutors
) -> None:
    note = Candidate(
        kind="note",
        item_id="projects/garden.md#Plan",
        text="The garden plan has tomatoes along the south fence.",
        when=None,
        source=NoteSource(path="projects/garden.md", heading="Plan", authored_by="user"),
        ranks={"note_vector": 1, "note_fts": 1},
    )

    result = await _fetcher(index, executors, notes=StaticNotes([note])).fetch(
        _request("what is in the garden plan")
    )

    [candidate] = result.stage.candidates
    assert candidate.source == "vault" and candidate.alt_ref is None
    assert candidate.ref == vault_ref("projects/garden.md", "Plan", note.text)
    assert candidate.disposition == "injected"


# -- the deadline -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_stage1_outliving_the_deadline_returns_an_empty_deadline_result(
    index: HookedIndex, executors: FetcherExecutors, release: threading.Event
) -> None:
    index.hooks["search_names"] = _block(release)

    result = await _fetcher(index, executors, config=_config(deadline_ms=50)).fetch(
        _request("where does she work", turns=("I met Wren yesterday",))
    )

    assert result.degraded_reason == "deadline" and result.items == ()
    assert result.stage.ran and result.stage.prefetch_reuse == "none"


@pytest.mark.asyncio
async def test_a_slow_pool_leaves_the_finished_pools_ranked(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors, release: threading.Event
) -> None:
    fact_index.upsert([_fact("Tidewater Survey")])
    index.hooks["search_vector"] = _block(release)

    result = await _fetcher(index, executors, config=_config(deadline_ms=50)).fetch(
        _request("tidewater survey")
    )

    assert result.degraded_reason == "deadline"
    assert [item.item_id for item in result.items] == [_key("Tidewater Survey")]


@pytest.mark.asyncio
async def test_a_result_arriving_after_the_deadline_is_discarded(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    fact_index.upsert([_fact("Tidewater Survey")])
    clock = FakeClock()
    index.hooks["search_fts"] = lambda: clock.advance(ms=60)

    result = await _fetcher(
        index, executors, clock=clock, embedder=FailingEmbedder(), config=_config(deadline_ms=50)
    ).fetch(_request("tidewater survey"))

    assert result.degraded_reason == "deadline" and result.items == ()
    assert result.timings.total_ms == pytest.approx(60.0)


@pytest.mark.asyncio
async def test_queue_wait_is_recorded(fact_index: FactIndex, index: HookedIndex) -> None:
    fact_index.upsert([_fact("Tidewater Survey")])
    index_pool = ThreadPoolExecutor(1, thread_name_prefix="memory-index")
    model_pool = ThreadPoolExecutor(1, thread_name_prefix="memory-model")
    try:
        executors = FetcherExecutors(index=index_pool, model=model_pool)
        index_pool.submit(time.sleep, 0.1)  # earlier work holding the only index thread

        result = await _fetcher(index, executors, clock=SystemClock()).fetch(
            _request("tidewater survey")
        )
    finally:
        index_pool.shutdown(wait=True)
        model_pool.shutdown(wait=True)

    assert result.timings.queue_wait_ms >= 50.0
    assert result.timings.total_ms >= result.timings.queue_wait_ms / 2


# -- candidates in the stage ------------------------------------------------------------------


def _words(prefix: str, count: int) -> str:
    return " ".join(f"{prefix}{n}" for n in range(count))


@pytest.mark.asyncio
async def test_the_stage_records_every_disposition(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    header = FakeTokenCounter().count(HEADER)
    rows = {
        "alpha": _fact("Alpha " + _words("a", 28), rel_type="USES"),
        "bravo": _fact("Bravo " + _words("b", 28), rel_type="USES"),
        "charlie": _fact("Charlie", rel_type="USES"),
        "delta": _fact("Delta " + _words("d", 200), rel_type="USES"),
        "echo": _fact("Echo", rel_type="USES"),
        "foxtrot": _fact("Foxtrot", rel_type="USES"),
    }
    fact_index.upsert(list(rows.values()))
    reranker = ScriptedReranker(
        {"alpha": 0.99, "bravo": 0.98, "charlie": 0.97, "delta": 0.96, "foxtrot": 0.6}
    )
    fetcher = _fetcher(
        index,
        executors,
        reranker=reranker,
        suppression=StaticSuppression(frozenset({rows["charlie"].fact_key})),
        config=_config(min_score=0.3, block_tokens=header + 50),
    )

    result = await fetcher.fetch(_request("which tools does the user use"))

    by_key = {candidate.ref: candidate.disposition for candidate in result.stage.candidates}
    expected = {
        "alpha": "injected",
        "bravo": "dropped_budget",
        "charlie": "dropped_suppressed",
        "delta": "dropped_no_fit",
        "echo": "dropped_min_score",
        "foxtrot": "injected",
    }
    for name, disposition in expected.items():
        row = rows[name]
        assert by_key[graph_ref("user", "USES", row.object_id)] == disposition, name
    assert result.suppressed == result.stage.suppressed == 1
    assert all(rows["charlie"].object_name not in item.text for item in result.items)


@pytest.mark.asyncio
async def test_the_stage_keeps_fifty_candidates_injected_first(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    header = FakeTokenCounter().count(HEADER)
    rows = [_fact("Lead " + _words("l", 20), rel_type="USES")]
    rows += [_fact(f"Long{n:02d} " + _words("x", 80), rel_type="USES") for n in range(57)]
    rows += [_fact("Small", rel_type="USES"), _fact("Zero", rel_type="USES")]
    fact_index.upsert(rows)
    scores = {"lead": 0.99, "small": 0.4}
    scores.update({f"long{n:02d}": 0.98 - n * 0.001 for n in range(57)})
    fetcher = _fetcher(
        index,
        executors,
        reranker=ScriptedReranker(scores),
        pool_size=60,
        config=_config(min_score=0.3, rerank_top_n=60, block_tokens=header + 40),
    )

    result = await fetcher.fetch(_request("which tools does the user use"))

    candidates = result.stage.candidates
    assert len(candidates) == STAGE_CANDIDATE_CAP and result.stage.truncated == 10
    assert [(c.rank, c.disposition) for c in candidates[:2]] == [
        (1, "injected"),
        (59, "injected"),
    ]
    assert [c.rank for c in candidates[2:]] == list(range(2, 50))
    assert all(c.disposition != "injected" for c in candidates[2:])
    small = graph_ref("user", "USES", "small")
    assert candidates[1].ref == small


# -- sibling lists ----------------------------------------------------------------------------


def _tools(utterance: str = "utt-9") -> list[FactRow]:
    rows = [
        _fact(name, rel_type="USES", utterance=utterance, object_type="Skill")
        for name in ("Vim", "Rust", "Zig", "Emacs")
    ]
    return [*rows, _fact("Go", rel_type="USES", utterance="utt-2", object_type="Skill")]


@pytest.mark.asyncio
async def test_two_siblings_over_min_score_bring_the_whole_list(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    fact_index.upsert(_tools())
    fetcher = _fetcher(
        index,
        executors,
        reranker=ScriptedReranker({"vim": 0.9, "rust": 0.9}),
        config=_config(min_score=0.5),
    )

    outcome = await fetcher.fetch_outcome(_request("do I use vim and rust"))

    [placed] = outcome.lists
    # FactIndex.siblings order: recorded_at, then the folded object name.
    assert [fact.object_name for fact in placed.facts.facts] == ["Emacs", "Rust", "Vim", "Zig"]
    [item] = outcome.result.items
    assert "Zig" in item.text and "Go" not in item.text
    injected = [c for c in outcome.result.stage.candidates if c.disposition == "injected"]
    assert {c.ref for c in injected} >= {
        graph_ref("user", "USES", "vim"),
        graph_ref("user", "USES", "rust"),
    }
    assert {c.item for c in injected} == {1}


@pytest.mark.asyncio
async def test_a_list_is_capped_at_list_max_items(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    fact_index.upsert(_tools())
    fetcher = _fetcher(
        index,
        executors,
        reranker=ScriptedReranker({"vim": 0.9, "zig": 0.9}),
        config=_config(min_score=0.5, list_max_items=3),
    )

    outcome = await fetcher.fetch_outcome(_request("do I use vim and zig"))

    [placed] = outcome.lists
    names = [fact.object_name for fact in placed.facts.facts]
    assert len(names) == 3 and {"Vim", "Zig"} <= set(names)


@pytest.mark.asyncio
async def test_one_sibling_over_min_score_stays_a_single_item(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    fact_index.upsert(_tools())
    fetcher = _fetcher(
        index, executors, reranker=ScriptedReranker({"vim": 0.9}), config=_config(min_score=0.5)
    )

    outcome = await fetcher.fetch_outcome(_request("do I use vim"))

    assert outcome.lists == ()
    assert [item.item_id for item in outcome.result.items] == [_key("Vim", "USES")]
    assert "siblings" not in index.calls


@pytest.mark.asyncio
async def test_a_suppressed_sibling_stays_out_of_the_list(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    fact_index.upsert(_tools())
    fetcher = _fetcher(
        index,
        executors,
        reranker=ScriptedReranker({"vim": 0.9, "rust": 0.9}),
        suppression=StaticSuppression(frozenset({_key("Zig", "USES")})),
        config=_config(min_score=0.5),
    )

    outcome = await fetcher.fetch_outcome(_request("do I use vim and rust"))

    [placed] = outcome.lists
    assert [fact.object_name for fact in placed.facts.facts] == ["Emacs", "Rust", "Vim"]


# -- prefetch ---------------------------------------------------------------------------------


_QUERY = "where does the user work these days"


def _prefetch_tasks(session: str = "s1") -> list[asyncio.Task[object]]:
    return [task for task in asyncio.all_tasks() if task.get_name() == f"memory-prefetch-{session}"]


async def _settle(session: str = "s1") -> None:
    """Wait for the session's prefetch work, by the task names `prefetch` documents."""
    names = {f"memory-prefetch-{session}", f"memory-stage1-{session}"}
    pending = [task for task in asyncio.all_tasks() if task.get_name() in names]
    await asyncio.gather(*pending, return_exceptions=True)


async def _warm(fetcher: MemoryFetcher) -> tuple[str, ...]:
    """One fetch, so the session has the context a prefetch borrows; returns the next turns."""
    await fetcher.fetch(_request("hello there"))
    return ("hello there",)


@pytest.mark.asyncio
async def test_prefetch_block_is_reused_on_an_equal_normalised_text(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    fact_index.upsert([_fact("Tidewater Survey")])
    embedder = FakeEmbedder()
    fetcher = _fetcher(index, executors, embedder=embedder)
    turns = await _warm(fetcher)
    await fetcher.prefetch("s1", "Where does  the user WORK these days")
    await _settle()
    searches, embeds = len(index.calls), len(embedder.calls)

    result = await fetcher.fetch(_request(_QUERY, turns=turns))

    assert result.stage.prefetch_reuse == "block" and result.timings.prefetch_hit
    assert [item.item_id for item in result.items] == [_key("Tidewater Survey")]
    assert (len(index.calls), len(embedder.calls)) == (searches, embeds)


@pytest.mark.asyncio
async def test_fetch_awaits_a_running_prefetch_of_the_same_text(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors, release: threading.Event
) -> None:
    fact_index.upsert([_fact("Tidewater Survey")])
    fetcher = _fetcher(index, executors)
    turns = await _warm(fetcher)
    index.hooks["search_fts"] = _block(release)
    await fetcher.prefetch("s1", _QUERY)
    asyncio.get_running_loop().call_later(0.05, release.set)

    result = await fetcher.fetch(_request(_QUERY, turns=turns))

    assert result.stage.prefetch_reuse == "block"
    assert [item.item_id for item in result.items] == [_key("Tidewater Survey")]
    assert index.calls.count("search_fts") == 2  # the warm-up fetch and the prefetch


@pytest.mark.asyncio
async def test_a_running_prefetch_past_the_deadline_gives_a_deadline_result(
    index: HookedIndex, executors: FetcherExecutors, release: threading.Event
) -> None:
    fetcher = _fetcher(index, executors, config=_config(deadline_ms=50))
    turns = await _warm(fetcher)
    index.hooks["search_fts"] = _block(release)
    await fetcher.prefetch("s1", _QUERY)

    result = await fetcher.fetch(_request(_QUERY, turns=turns))

    assert result.degraded_reason == "deadline" and result.items == ()


@pytest.mark.asyncio
async def test_a_close_text_reuses_the_prefetched_stage1(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    fact_index.upsert([_fact("Tidewater Survey")])
    fetcher = _fetcher(index, executors)
    turns = await _warm(fetcher)
    await fetcher.prefetch("s1", _QUERY)
    await _settle()
    query = _QUERY + " now"
    assert jaccard(_QUERY, query) >= 0.8
    names_before = index.calls.count("search_names")

    result = await fetcher.fetch(_request(query, turns=turns))

    assert result.stage.prefetch_reuse == "stage1" and result.timings.prefetch_hit
    assert index.calls.count("search_names") == names_before + 1  # the query's names only
    assert [item.item_id for item in result.items] == [_key("Tidewater Survey")]


@pytest.mark.asyncio
async def test_a_different_text_runs_fresh(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    fact_index.upsert([_fact("Tidewater Survey")])
    fetcher = _fetcher(index, executors)
    turns = await _warm(fetcher)
    await fetcher.prefetch("s1", _QUERY)
    await _settle()

    result = await fetcher.fetch(_request("tell me about tidewater survey", turns=turns))

    assert result.stage.prefetch_reuse == "none" and not result.timings.prefetch_hit


@pytest.mark.asyncio
async def test_speech_start_prefetch_caches_stage1_for_the_endpoint(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    fact_index.upsert([_fact("Tidewater Survey")])
    fetcher = _fetcher(index, executors)
    turns = await _warm(fetcher)
    await fetcher.prefetch("s1")
    await _settle()

    result = await fetcher.fetch(_request("tidewater survey", turns=turns))

    assert result.stage.prefetch_reuse == "stage1"
    assert result.items


@pytest.mark.asyncio
async def test_the_latest_prefetch_text_wins(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors, release: threading.Event
) -> None:
    fact_index.upsert([_fact("Tidewater Survey"), _fact("Harbor Bakery")])
    fetcher = _fetcher(index, executors)
    turns = await _warm(fetcher)
    index.hooks["search_fts"] = _block(release)
    await fetcher.prefetch("s1", "where is harbor bakery")
    [first] = _prefetch_tasks()
    await fetcher.prefetch("s1", _QUERY)
    [second] = [task for task in _prefetch_tasks() if task is not first]
    await asyncio.sleep(0)
    assert first.cancelled()
    release.set()
    await _settle()
    assert second.done() and not second.cancelled()

    latest = await fetcher.fetch(_request(_QUERY, turns=turns))
    assert latest.stage.prefetch_reuse == "block"


@pytest.mark.asyncio
async def test_invalidate_drops_the_prefetched_block(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    fact_index.upsert([_fact("Tidewater Survey")])
    fetcher = _fetcher(index, executors)
    turns = await _warm(fetcher)
    await fetcher.prefetch("s1", _QUERY)
    await _settle()

    fetcher.invalidate()
    result = await fetcher.fetch(_request(_QUERY, turns=turns))

    assert result.stage.prefetch_reuse == "none"


@pytest.mark.asyncio
async def test_invalidate_of_one_session_keeps_the_others(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    fact_index.upsert([_fact("Tidewater Survey")])
    fetcher = _fetcher(index, executors)
    await fetcher.fetch(_request("hello there", session="s1"))
    await fetcher.fetch(_request("hello there", session="s2"))
    for session in ("s1", "s2"):
        await fetcher.prefetch(session, _QUERY)
        await _settle(session)

    fetcher.invalidate("s1")
    one = await fetcher.fetch(_request(_QUERY, session="s1", turns=("hello there",)))
    two = await fetcher.fetch(_request(_QUERY, session="s2", turns=("hello there",)))

    assert (one.stage.prefetch_reuse, two.stage.prefetch_reuse) == ("none", "block")


@pytest.mark.asyncio
async def test_a_prefetch_finished_after_invalidate_is_not_kept(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors, release: threading.Event
) -> None:
    fact_index.upsert([_fact("Tidewater Survey")])
    fetcher = _fetcher(index, executors)
    turns = await _warm(fetcher)
    index.hooks["search_fts"] = _block(release)
    await fetcher.prefetch("s1", _QUERY)
    await asyncio.sleep(0.01)
    fetcher.invalidate()
    release.set()
    await _settle()
    index.hooks.clear()

    result = await fetcher.fetch(_request(_QUERY, turns=turns))

    assert result.stage.prefetch_reuse != "block"


# -- helpers ----------------------------------------------------------------------------------


def test_normalise_collapses_whitespace_and_folds_case() -> None:
    assert normalise("  Where does\tthe USER  work ") == "where does the user work"


def test_jaccard_compares_token_sets() -> None:
    assert jaccard("a b c d", "a b c d") == 1.0
    assert jaccard("a b c d", "a b c e") == pytest.approx(3 / 5)
    assert jaccard("", "") == 1.0
