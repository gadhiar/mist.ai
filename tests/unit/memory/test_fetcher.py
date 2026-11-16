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
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.errors import EmbeddingError, MemoryIndexError, MemoryRerankError
from backend.knowledge.models import graph_ref, vault_ref
from backend.memory.block import HEADER, MemoryBlockBuilder
from backend.memory.config import MemoryConfig
from backend.memory.facets import facet_key
from backend.memory.fact_index import FactGroup, FactIndex, IndexedFacet, IndexedFact
from backend.memory.fetcher import (
    _DEGRADED_PRIORITY,
    STAGE_CANDIDATE_CAP,
    FetcherExecutors,
    FetchOutcome,
    MemoryFetcher,
    SystemClock,
    jaccard,
    normalise,
)
from backend.memory.index_db import open_index, open_reader
from backend.memory.models import (
    Candidate,
    FactRow,
    MemoryCandidate,
    MemoryRequest,
    MemoryResult,
    NeedTrace,
    NoteSource,
    ServingProfile,
    fact_key,
)
from backend.memory.need_gate import NeedDecision, context_text
from backend.memory.rerank import NullReranker
from backend.memory.sources import ENTITY_POOL, FTS_POOL
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
    """A FactIndex whose searches run `hooks[name]` first and are recorded by name.

    `script` pins the pool order a test needs: the FTS pool then holds exactly the scripted
    facts in that order and the vector and entity pools hold none, so each fact's pool rank is
    its place in the script. `script_facets` has the facet FTS and entity pools both hold the
    scripted facets and the facet vector pool none, so a scripted facet fuses ahead of every
    scripted fact.
    """

    def __init__(self, inner: FactIndex) -> None:
        self.inner = inner
        self.hooks: dict[str, Callable[[], None]] = {}
        self.calls: list[str] = []
        self.member_lookups: list[tuple[str, ...]] = []
        self.scripted: tuple[FactRow, ...] | None = None
        self.scripted_facets: tuple[str, ...] | None = None

    def script(self, *rows: FactRow) -> None:
        self.scripted = rows

    def script_facets(self, *facet_keys: str) -> None:
        self.scripted_facets = facet_keys

    def _enter(self, name: str) -> None:
        self.calls.append(name)
        hook = self.hooks.get(name)
        if hook is not None:
            hook()

    def _scripted_groups(self) -> list[FactGroup]:
        groups: list[FactGroup] = []
        for row in self.scripted or ():
            found = self.inner.search_names([row.object_name], limit=20)
            groups.extend(group for group in found if group.fact_key == row.fact_key)
        return groups

    def search_fts(self, query_text: str, *, limit: int = 20) -> list[FactGroup]:
        self._enter("search_fts")
        if self.scripted is not None:
            return self._scripted_groups()[:limit]
        return self.inner.search_fts(query_text, limit=limit)

    def search_vector(self, embedding: Sequence[float], *, limit: int = 20) -> list[FactGroup]:
        self._enter("search_vector")
        if self.scripted is not None:
            return []
        return self.inner.search_vector(embedding, limit=limit)

    def search_names(self, texts: Sequence[str], *, limit: int = 20) -> list[FactGroup]:
        self._enter("search_names")
        if self.scripted is not None:
            return []
        return self.inner.search_names(texts, limit=limit)

    def siblings(
        self, subject_id: str, rel_type: str, source_utterance_id: str, *, now: datetime
    ) -> tuple[IndexedFact, ...]:
        self._enter("siblings")
        return self.inner.siblings(subject_id, rel_type, source_utterance_id, now=now)

    def _scripted_facet_list(self) -> list[IndexedFacet]:
        # Every test fact's subject is the user, so this finds every facet of the index.
        found = {facet.facet_key: facet for facet in self.inner.search_facet_names(["User"])}
        return [found[key] for key in self.scripted_facets or () if key in found]

    def search_facets_fts(self, query_text: str, *, limit: int = 20) -> list[IndexedFacet]:
        self._enter("search_facets_fts")
        if self.scripted_facets is not None:
            return self._scripted_facet_list()[:limit]
        return self.inner.search_facets_fts(query_text, limit=limit)

    def search_facets_vector(
        self, embedding: Sequence[float], *, limit: int = 20
    ) -> list[IndexedFacet]:
        self._enter("search_facets_vector")
        if self.scripted_facets is not None:
            return []
        return self.inner.search_facets_vector(embedding, limit=limit)

    def search_facet_names(self, texts: Sequence[str], *, limit: int = 20) -> list[IndexedFacet]:
        self._enter("search_facet_names")
        if self.scripted_facets is not None:
            return self._scripted_facet_list()[:limit]
        return self.inner.search_facet_names(texts, limit=limit)

    def facet_members_many(
        self, facet_keys: Iterable[str], *, now: datetime
    ) -> dict[str, tuple[IndexedFact, ...]]:
        keys = tuple(facet_keys)
        self.member_lookups.append(keys)
        self._enter("facet_members_many")
        return self.inner.facet_members_many(keys, now=now)


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
    gate_pool = ThreadPoolExecutor(1, thread_name_prefix="memory-gate")
    yield FetcherExecutors(index=index_pool, model=model_pool, gate=gate_pool)
    for pool in (index_pool, model_pool, gate_pool):
        pool.shutdown(wait=True, cancel_futures=True)


@pytest.fixture
def release() -> Iterator[threading.Event]:
    """An event blocking hooks wait on; set at teardown so no executor thread is left blocked."""
    event = threading.Event()
    yield event
    event.set()


def _config(**overrides: object) -> MemoryConfig:
    # Facets off unless a test is about them: the test facts share a subject, relation and
    # object type, so most of them would form facets the test does not look for.
    values: dict[str, object] = {
        "rerank_top_n": 20,
        "select_k": 8,
        "block_tokens": 300,
        "deadline_ms": _ROOMY_MS,
        "list_max_items": 10,
        "facets": False,
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
    need_gate: object | None = None,
    need_mode: str = "on",
    need_context: bool = True,
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
        need_gate=need_gate,  # type: ignore[arg-type]
        need_mode=need_mode,  # type: ignore[arg-type]
        need_context=need_context,
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
    fetcher = _fetcher(
        index, executors, reranker=reranker, config=_config(rerank_top_n=4, select_k=4)
    )

    outcome = await fetcher.fetch_outcome(_request("which survey office"))

    # Every fused candidate, in pool order, before the top_n cut; its head is what was scored,
    # each by its fact view.
    assert len(outcome.fused) == len(rows)
    rrf = [fused.rrf for fused in outcome.fused]
    assert rrf == sorted(rrf, reverse=True)
    assert [fused.pool_rank for fused in outcome.fused] == list(range(1, len(rows) + 1))
    [(_, scored_texts, _)] = reranker.calls
    views = [fused.candidate.view_text for fused in outcome.fused[:4]]
    assert all(views) and views == list(scored_texts)
    assert outcome.result.reranked == 4
    # The ordered shortlist, in the order and with the scores and ranks of the stage.
    refs = {row.fact_key: graph_ref("user", "WORKS_AT", row.object_id) for row in rows}
    by_rank = sorted(outcome.result.stage.candidates, key=lambda candidate: candidate.rank)
    assert [
        (refs[s.candidate.item_id], s.score, s.position, s.pool_rank, s.ce_rank, s.p_ce)
        for s in outcome.ranked
    ] == [(c.ref, c.score, c.rank, c.pool_rank, c.ce_rank, c.p_ce) for c in by_rank]
    assert {s.candidate.item_id for s in outcome.ranked} == {
        fused.candidate.item_id for fused in outcome.fused[:4]
    }
    # Selection: the first select_k of the order, in packing order.
    assert outcome.selected == outcome.ranked


@pytest.mark.asyncio
async def test_an_outcome_without_a_stage2_carries_no_candidates(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    fact_index.upsert([_fact("Tidewater Survey")])
    fetcher = _fetcher(index, executors, index_state=lambda: "index_rebuilding")

    outcome = await fetcher.fetch_outcome(_request("tidewater survey"))

    assert outcome.result.degraded_reason == "index_rebuilding"
    assert (outcome.fused, outcome.ranked, outcome.selected) == ((), (), ())


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
async def test_a_failing_reranker_keeps_the_pool_order(
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
    gate_pool = ThreadPoolExecutor(1, thread_name_prefix="memory-gate")
    try:
        executors = FetcherExecutors(index=index_pool, model=model_pool, gate=gate_pool)
        index_pool.submit(time.sleep, 0.1)  # earlier work holding the only index thread

        result = await _fetcher(index, executors, clock=SystemClock()).fetch(
            _request("tidewater survey")
        )
    finally:
        index_pool.shutdown(wait=True)
        model_pool.shutdown(wait=True)
        gate_pool.shutdown(wait=True)

    assert result.timings.queue_wait_ms >= 50.0
    assert result.timings.total_ms >= result.timings.queue_wait_ms / 2


# -- candidates in the stage ------------------------------------------------------------------


def _words(prefix: str, count: int) -> str:
    return " ".join(f"{prefix}{n}" for n in range(count))


def _ref(row: FactRow) -> str:
    return graph_ref("user", row.rel_type, row.object_id)


def _by_ref(result: MemoryResult) -> dict[str, MemoryCandidate]:
    return {candidate.ref: candidate for candidate in result.stage.candidates}


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
        "foxtrot": _fact("Foxtrot", rel_type="USES"),
        "echo": _fact("Echo", rel_type="USES"),
    }
    fact_index.upsert(list(rows.values()))
    index.script(*rows.values())
    reranker = ScriptedReranker(
        {"alpha": 0.99, "bravo": 0.98, "charlie": 0.97, "delta": 0.96, "foxtrot": 0.6}
    )
    fetcher = _fetcher(
        index,
        executors,
        reranker=reranker,
        suppression=StaticSuppression(frozenset({rows["charlie"].fact_key})),
        config=_config(select_k=4, block_tokens=header + 50),
    )

    result = await fetcher.fetch(_request("which tools does the user use"))

    by_ref = _by_ref(result)
    expected = {
        "alpha": ("injected", 1, 1),
        "bravo": ("dropped_budget", 2, 2),
        "delta": ("dropped_no_fit", 3, 4),
        "foxtrot": ("injected", 4, 5),
        "echo": ("dropped_rank", 5, 6),
        # Suppressed before the cross-encoder: no pair, no slot, traced after the order.
        "charlie": ("dropped_suppressed", 6, 3),
    }
    for name, (disposition, rank, pool_rank) in expected.items():
        candidate = by_ref[_ref(rows[name])]
        assert (candidate.disposition, candidate.rank, candidate.pool_rank) == (
            disposition,
            rank,
            pool_rank,
        ), name
    assert by_ref[_ref(rows["alpha"])].ce_rank == 1
    assert by_ref[_ref(rows["alpha"])].p_ce == pytest.approx(0.99)
    charlie = by_ref[_ref(rows["charlie"])]
    assert (charlie.ce_rank, charlie.p_ce) == (None, None)
    assert charlie.score == pytest.approx(1 / (60 + 3))
    [(_, texts)] = reranker.calls
    assert len(texts) == 5 and not any("charlie" in text.lower() for text in texts)
    assert result.suppressed == result.stage.suppressed == 1
    assert all(rows["charlie"].object_name not in item.text for item in result.items)


@pytest.mark.asyncio
async def test_exactly_select_k_are_selected_and_the_rest_dropped_by_rank(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    rows = [_fact(name, rel_type="USES") for name in ("Ash", "Birch", "Cedar", "Damson", "Elm")]
    fact_index.upsert(rows)
    index.script(*rows)
    reranker = ScriptedReranker({"ash": 0.9, "birch": 0.8, "cedar": 0.7, "damson": 0.6})
    fetcher = _fetcher(index, executors, reranker=reranker, config=_config(select_k=3))

    outcome = await fetcher.fetch_outcome(_request("which trees"))

    assert [scored.position for scored in outcome.ranked] == [1, 2, 3, 4, 5]
    assert outcome.selected == outcome.ranked[:3]
    assert [item.item_id for item in outcome.result.items] == [row.fact_key for row in rows[:3]]
    by_ref = _by_ref(outcome.result)
    assert [by_ref[_ref(row)].disposition for row in rows] == [
        "injected",
        "injected",
        "injected",
        "dropped_rank",
        "dropped_rank",
    ]


@pytest.mark.asyncio
async def test_without_the_cross_encoder_the_order_is_the_pool_order(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    rows = [_fact(name, rel_type="USES") for name in ("Elm", "Damson", "Cedar")]
    fact_index.upsert(rows)
    index.script(*rows)
    fetcher = _fetcher(index, executors, reranker=FailingReranker())

    outcome = await fetcher.fetch_outcome(_request("which trees"))

    assert outcome.result.degraded_reason == "reranker_unavailable"
    assert [scored.candidate.item_id for scored in outcome.ranked] == [r.fact_key for r in rows]
    assert all(scored.ce_rank is None for scored in outcome.ranked)


@pytest.mark.asyncio
async def test_the_stage_keeps_fifty_candidates_injected_first(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    header = FakeTokenCounter().count(HEADER)
    lead = _fact("Lead " + _words("l", 20), rel_type="USES")
    longs = [_fact(f"Long{n:02d} " + _words("x", 80), rel_type="USES") for n in range(57)]
    small, zero = _fact("Small", rel_type="USES"), _fact("Zero", rel_type="USES")
    fact_index.upsert([lead, *longs, small, zero])
    # Pool order and cross-encoder order agree, so the order is the script's: Small is 25th.
    index.script(lead, *longs[:23], small, *longs[23:], zero)
    scores = {"lead": 0.99, "small": 0.9575}
    scores.update({f"long{n:02d}": 0.98 - n * 0.001 for n in range(57)})
    fetcher = _fetcher(
        index,
        executors,
        reranker=ScriptedReranker(scores),
        pool_size=60,
        config=_config(select_k=25, rerank_top_n=60, block_tokens=header + 40),
    )

    result = await fetcher.fetch(_request("which tools does the user use"))

    candidates = result.stage.candidates
    assert len(candidates) == STAGE_CANDIDATE_CAP and result.stage.truncated == 10
    assert [(c.rank, c.disposition) for c in candidates[:2]] == [
        (1, "injected"),
        (25, "injected"),
    ]
    assert [c.rank for c in candidates[2:]] == [*range(2, 25), *range(26, 51)]
    assert all(c.disposition != "injected" for c in candidates[2:])
    assert candidates[1].ref == _ref(small)


# -- sibling lists ----------------------------------------------------------------------------


def _tools(utterance: str = "utt-9") -> list[FactRow]:
    rows = [
        _fact(name, rel_type="USES", utterance=utterance, object_type="Skill")
        for name in ("Vim", "Rust", "Zig", "Emacs")
    ]
    return [*rows, _fact("Go", rel_type="USES", utterance="utt-2", object_type="Skill")]


def _scripted_tools(index: HookedIndex, first: Sequence[str]) -> None:
    """Script the tools' pool order: the named ones first, in order, then the rest."""
    rows = {row.object_name: row for row in _tools()}
    index.script(*(rows[name] for name in first), *(r for n, r in rows.items() if n not in first))


@pytest.mark.asyncio
async def test_two_selected_siblings_bring_the_whole_list(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    fact_index.upsert(_tools())
    _scripted_tools(index, ["Vim", "Rust"])
    fetcher = _fetcher(
        index,
        executors,
        reranker=ScriptedReranker({"vim": 0.9, "rust": 0.9}),
        config=_config(select_k=2),
    )

    outcome = await fetcher.fetch_outcome(_request("do I use vim and rust"))

    [placed] = outcome.lists
    # FactIndex.siblings order: recorded_at, then the folded object name.
    assert [fact.object_name for fact in placed.facts.facts] == ["Emacs", "Rust", "Vim", "Zig"]
    [item] = outcome.result.items
    assert "Zig" in item.text and "Go" not in item.text
    by_ref = _by_ref(outcome.result)
    # Zig and Emacs ranked after the selection; the list places them, so they are injected.
    for name in ("Vim", "Rust", "Zig", "Emacs"):
        assert (by_ref[graph_ref("user", "USES", name.lower())].disposition, name) == (
            "injected",
            name,
        )
    assert {c.item for c in by_ref.values() if c.disposition == "injected"} == {1}
    assert by_ref[graph_ref("user", "USES", "go")].disposition == "dropped_rank"
    # A placed sibling takes the position of the earliest selected member.
    positions = {s.candidate.item_id: s.position for s in outcome.selected}
    assert positions[_key("Zig", "USES")] == positions[_key("Vim", "USES")] == 1


@pytest.mark.asyncio
async def test_a_list_is_capped_at_list_max_items(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    fact_index.upsert(_tools())
    _scripted_tools(index, ["Vim", "Zig"])
    fetcher = _fetcher(
        index,
        executors,
        reranker=ScriptedReranker({"vim": 0.9, "zig": 0.9}),
        config=_config(select_k=2, list_max_items=3),
    )

    outcome = await fetcher.fetch_outcome(_request("do I use vim and zig"))

    [placed] = outcome.lists
    names = [fact.object_name for fact in placed.facts.facts]
    assert len(names) == 3 and {"Vim", "Zig"} <= set(names)


@pytest.mark.asyncio
async def test_one_selected_sibling_stays_a_single_item(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    fact_index.upsert(_tools())
    _scripted_tools(index, ["Vim"])
    fetcher = _fetcher(
        index, executors, reranker=ScriptedReranker({"vim": 0.9}), config=_config(select_k=1)
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
    _scripted_tools(index, ["Vim", "Rust"])
    fetcher = _fetcher(
        index,
        executors,
        reranker=ScriptedReranker({"vim": 0.9, "rust": 0.9}),
        suppression=StaticSuppression(frozenset({_key("Zig", "USES")})),
        config=_config(select_k=2),
    )

    outcome = await fetcher.fetch_outcome(_request("do I use vim and rust"))

    [placed] = outcome.lists
    assert [fact.object_name for fact in placed.facts.facts] == ["Emacs", "Rust", "Vim"]


# -- facets -----------------------------------------------------------------------------------


_CLUBS = facet_key("user", "MEMBER_OF", "Organization")


def _club(name: str, recorded: str, *, utterance: str | None = None, **kwargs: object) -> FactRow:
    row = _fact(name, rel_type="MEMBER_OF", utterance=utterance, **kwargs)  # type: ignore[arg-type]
    return replace(row, recorded_at=f"2026-{recorded}T00:00:00+00:00")


def _clubs() -> dict[str, FactRow]:
    """Four clubs of separate utterances, newest recorded first: Kestrel, Wren, Gull, Heron."""
    return {
        "heron": _club("Heron Society", "01-01"),
        "gull": _club("Gull Club", "02-01"),
        "wren": _club("Wren Circle", "03-01"),
        "kestrel": _club("Kestrel Guild", "04-01"),
    }


def _member_scores() -> ScriptedReranker:
    # The facet's view names no member, so it scores by its type word alone.
    return ScriptedReranker({"heron": 0.9, "filler": 0.85, "kestrel": 0.1, "organization": 0.8})


async def _member_fetch(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors, *, select_k: int
) -> tuple[FetchOutcome, dict[str, FactRow]]:
    """Order: Heron (1), the facet (2), Filler (3), Kestrel (4); Gull and Wren unranked.

    The facet is first in the pools (two facet pools find it), Heron first by the
    cross-encoder; the order scores put the facet between Heron and Filler.
    """
    rows = _clubs()
    filler = _fact("Filler Works")
    fact_index.upsert([*rows.values(), filler])
    index.script(rows["heron"], rows["kestrel"], filler)
    index.script_facets(_CLUBS)
    fetcher = _fetcher(
        index,
        executors,
        reranker=_member_scores(),
        config=_config(facets=True, select_k=select_k, list_max_items=3),
    )
    outcome = await fetcher.fetch_outcome(_request("which clubs"))
    return outcome, {**rows, "filler": filler}


@pytest.mark.asyncio
async def test_a_selected_facet_places_its_members_selected_first_then_newest(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    outcome, rows = await _member_fetch(fact_index, index, executors, select_k=2)

    order = [(s.candidate.kind, s.candidate.item_id) for s in outcome.ranked]
    assert order == [
        ("fact", rows["heron"].fact_key),
        ("facet", _CLUBS),
        ("fact", rows["filler"].fact_key),
        ("fact", rows["kestrel"].fact_key),
    ]
    # Heron is selected alone, so it comes first; then the newest: Kestrel, then Wren. The
    # cap of three leaves Gull out.
    placed = {s.candidate.item_id: s.position for s in outcome.selected}
    assert placed == {
        rows["heron"].fact_key: 1,
        _CLUBS: 2,
        rows["kestrel"].fact_key: 2,
        rows["wren"].fact_key: 2,
    }
    assert [s.position for s in outcome.selected] == sorted(s.position for s in outcome.selected)
    texts = [item.text for item in outcome.result.items]
    assert len(texts) == 3 and not any("Gull" in text for text in texts)
    by_ref = _by_ref(outcome.result)
    # Kestrel ranked past the selection; the facet placed it, so it is injected.
    kestrel = by_ref[_ref(rows["kestrel"])]
    assert (kestrel.rank, kestrel.disposition) == (4, "injected")
    assert by_ref[_ref(rows["filler"])].disposition == "dropped_rank"
    facet = by_ref[f"facet:{_CLUBS}"]
    heron = by_ref[_ref(rows["heron"])]
    assert (facet.source, facet.alt_ref, facet.rank) == ("facet", None, 2)
    assert facet.disposition == "injected" and facet.item == heron.item == 1
    assert index.member_lookups == [(_CLUBS,)]


@pytest.mark.asyncio
async def test_a_member_selected_alone_keeps_an_earlier_position_and_takes_the_facets_otherwise(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    outcome, rows = await _member_fetch(fact_index, index, executors, select_k=4)

    ranked = {s.candidate.item_id: s for s in outcome.ranked}
    selected = {s.candidate.item_id: s for s in outcome.selected}
    facet = ranked[_CLUBS]
    # Heron (1) is earlier than the facet (2) and keeps its place; Kestrel (4) moves up to it.
    assert selected[rows["heron"].fact_key] == ranked[rows["heron"].fact_key]
    kestrel = selected[rows["kestrel"].fact_key]
    assert (kestrel.position, kestrel.score) == (facet.position, facet.score)
    assert ranked[rows["kestrel"].fact_key].position == 4
    assert selected[rows["filler"].fact_key].position == 3


@pytest.mark.asyncio
async def test_a_facet_and_a_sibling_list_of_one_utterance_make_one_list(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    rows = [
        _club(name, "01-01", utterance="utt-clubs")
        for name in ("Gull Club", "Heron Society", "Kestrel Guild")
    ]
    fact_index.upsert(rows)
    gull, heron, kestrel = rows
    index.script(gull, heron)
    index.script_facets(_CLUBS)
    fetcher = _fetcher(
        index,
        executors,
        reranker=ScriptedReranker({"gull": 0.9, "heron": 0.85, "organization": 0.8}),
        config=_config(facets=True, select_k=3),
    )

    outcome = await fetcher.fetch_outcome(_request("which clubs"))

    # Fused in pool order, ordered by position, selected in packing order.
    assert [fused.pool_rank for fused in outcome.fused] == [1, 2, 3]
    assert [(f.candidate.kind, f.candidate.item_id) for f in outcome.fused] == [
        ("facet", _CLUBS),
        ("fact", gull.fact_key),
        ("fact", heron.fact_key),
    ]
    assert [(s.candidate.kind, s.candidate.item_id, s.position) for s in outcome.ranked] == [
        ("fact", gull.fact_key, 1),
        ("facet", _CLUBS, 2),
        ("fact", heron.fact_key, 3),
    ]
    selected = [(s.candidate.kind, s.candidate.item_id, s.position) for s in outcome.selected]
    assert selected == sorted(
        [
            ("fact", gull.fact_key, 1),
            ("facet", _CLUBS, 2),
            ("fact", heron.fact_key, 2),
            ("fact", kestrel.fact_key, 2),
        ],
        key=lambda entry: (entry[2], entry[1], entry[0]),
    )
    # One list, each fact once: the block's refusal of a repeated fact never fires.
    [placed] = outcome.lists
    assert sorted(fact.fact_key for fact in placed.facts.facts) == sorted(r.fact_key for r in rows)
    [item] = outcome.result.items
    assert outcome.result.stage.skip_reason is None and outcome.result.block_text is not None
    by_ref = _by_ref(outcome.result)
    assert {by_ref[_ref(row)].item for row in (gull, heron)} == {1}
    assert by_ref[f"facet:{_CLUBS}"].item == 1


@pytest.mark.parametrize("reason", ["suppressed", "not_current"])
@pytest.mark.asyncio
async def test_a_facet_without_two_current_members_leaves_before_the_cross_encoder(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors, reason: str
) -> None:
    if reason == "not_current":
        gull = _club("Gull Club", "01-01", valid_from="2024-01-01", valid_to="2025-01-01")
    else:
        gull = _club("Gull Club", "01-01")
    heron = _club("Heron Society", "02-01")
    fact_index.upsert([gull, heron])
    index.script(heron)
    index.script_facets(_CLUBS)
    reranker = FakeReranker()
    suppressed = frozenset({gull.fact_key}) if reason == "suppressed" else frozenset()
    fetcher = _fetcher(
        index,
        executors,
        reranker=reranker,
        suppression=StaticSuppression(suppressed),
        config=_config(facets=True, select_k=1),
    )

    outcome = await fetcher.fetch_outcome(_request("which clubs"))

    facet = _by_ref(outcome.result)[f"facet:{_CLUBS}"]
    assert (facet.disposition, facet.rank, facet.ce_rank) == ("dropped_no_members", 2, None)
    # It took no pair and no slot: the one slot went to Heron.
    [(_, texts, _)] = reranker.calls
    assert len(texts) == 1
    assert [item.item_id for item in outcome.result.items] == [heron.fact_key]
    assert index.member_lookups == [(_CLUBS,)]


@pytest.mark.asyncio
async def test_a_late_member_lookup_drops_every_facet_and_serves_the_facts(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    rows = _clubs()
    fact_index.upsert(list(rows.values()))
    index.script(rows["heron"])
    index.script_facets(_CLUBS)
    clock = FakeClock()
    index.hooks["facet_members_many"] = lambda: clock.advance(ms=60)
    fetcher = _fetcher(index, executors, clock=clock, config=_config(facets=True, deadline_ms=50))

    result = await fetcher.fetch(_request("which clubs"))

    assert result.degraded_reason == "deadline"
    assert _by_ref(result)[f"facet:{_CLUBS}"].disposition == "dropped_late"
    assert [item.item_id for item in result.items] == [rows["heron"].fact_key]


@pytest.mark.asyncio
async def test_a_failing_member_lookup_gives_no_block_and_index_error(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    fact_index.upsert(list(_clubs().values()))
    index.script_facets(_CLUBS)

    def fail() -> None:
        raise MemoryIndexError("disk I/O error")

    index.hooks["facet_members_many"] = fail
    result = await _fetcher(index, executors, config=_config(facets=True)).fetch(
        _request("which clubs")
    )

    assert result.degraded_reason == "index_error" and result.items == ()
    assert result.block_text is None


@pytest.mark.asyncio
async def test_a_selected_facet_none_of_whose_members_fit_is_dropped_budget(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    header = FakeTokenCounter().count(HEADER)
    heron = _club("Heron " + _words("h", 200), "01-01")
    gull = _club("Gull " + _words("g", 200), "02-01")
    fact_index.upsert([heron, gull])
    index.script(heron, gull)
    index.script_facets(_CLUBS)
    fetcher = _fetcher(
        index,
        executors,
        reranker=ScriptedReranker({"organization": 0.9}),
        config=_config(facets=True, select_k=1, block_tokens=header + 50),
    )

    outcome = await fetcher.fetch_outcome(_request("which clubs"))

    assert [s.candidate.item_id for s in outcome.ranked][0] == _CLUBS
    by_ref = _by_ref(outcome.result)
    facet = by_ref[f"facet:{_CLUBS}"]
    # Its members are too large for the whole budget, so the facet placed nothing.
    assert (facet.disposition, facet.item) == ("dropped_budget", None)
    assert {by_ref[_ref(row)].disposition for row in (heron, gull)} == {"dropped_no_fit"}
    assert outcome.result.items == ()


@pytest.mark.asyncio
async def test_two_selected_facets_share_one_member_lookup(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    tools = facet_key("user", "USES", "Organization")
    fact_index.upsert(
        [*_clubs().values(), _fact("Vim", rel_type="USES"), _fact("Rust", rel_type="USES")]
    )
    index.script()
    index.script_facets(_CLUBS, tools)
    fetcher = _fetcher(index, executors, config=_config(facets=True, select_k=2))

    outcome = await fetcher.fetch_outcome(_request("which clubs and tools"))

    selected = {s.candidate.item_id for s in outcome.selected if s.candidate.kind == "facet"}
    assert selected == {_CLUBS, tools}
    [keys] = index.member_lookups
    assert sorted(keys) == sorted([_CLUBS, tools])


@pytest.mark.asyncio
async def test_facets_come_from_their_own_pools_and_stage1_anchors(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    fact_index.upsert(list(_clubs().values()))
    fetcher = _fetcher(index, executors, config=_config(facets=True))

    outcome = await fetcher.fetch_outcome(
        _request("which organisations am I a member of", turns=("the user joined a club",))
    )

    assert {"search_facets_fts", "search_facets_vector"} <= set(index.calls)
    # Once for the recent turns in stage 1, once for the query's own entity pool.
    assert index.calls.count("search_facet_names") == 2
    [facet] = [f for f in outcome.fused if f.candidate.kind == "facet"]
    assert facet.candidate.item_id == _CLUBS
    # Only the recent turn names the user, so the entity rank came from stage 1's anchors.
    assert ENTITY_POOL in facet.candidate.ranks and FTS_POOL in facet.candidate.ranks
    assert facet.candidate.text == facet.candidate.view_text
    assert facet.candidate.when is None
    stage_facet = _by_ref(outcome.result)[f"facet:{_CLUBS}"]
    assert (stage_facet.source, stage_facet.alt_ref) == ("facet", None)


@pytest.mark.asyncio
async def test_facets_off_runs_no_facet_pool_and_offers_no_facet(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    fact_index.upsert(list(_clubs().values()))
    fetcher = _fetcher(index, executors, config=_config(facets=False))

    outcome = await fetcher.fetch_outcome(
        _request("which organisations am I a member of", turns=("the user joined a club",))
    )

    facet_calls = {
        "search_facets_fts",
        "search_facets_vector",
        "search_facet_names",
        "facet_members_many",
    }
    assert not facet_calls & set(index.calls)
    assert all(fused.candidate.kind != "facet" for fused in outcome.fused)


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
    # Let the prefetch finish on this loop, so none of its tasks outlives the test.
    release.set()
    await _settle()

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


# -- the need gate ----------------------------------------------------------------------------

_THETA_LOW = 0.3
_THETA_HIGH = 0.6
_TIMER = "set a timer for ten minutes"
_FIRST = "where does the user work these days"
_FOLLOW = "and her sister"
_CONTEXT = context_text(_FIRST, _FOLLOW)
_NEED_IDS = ("a-need-001", "b-need-002")
_NO_NEED_IDS = ("a-no-need-001", "b-no-need-002")


class FakeGate:
    """`NeedGate`'s attributes and calls with scripted probabilities, no bank and no model file.

    A vector's probability is the scripted one of the text whose FakeEmbedder vector it is,
    else `default`; FakeEmbedder is deterministic, so an unrecorded twin computes the keys.
    `embedded` records each `embed_query` text with the thread it ran on, `decided` each
    probability `decide` returned.
    """

    def __init__(
        self,
        scores: dict[str, float] | None = None,
        *,
        default: float = 0.9,
        shares_query_vector: bool = False,
        embedder: object | None = None,
        events: list[str] | None = None,
        refuse: bool = False,
        before_decide: Callable[[], None] | None = None,
    ) -> None:
        self.embedder = embedder if embedder is not None else FakeEmbedder()
        self.shares_query_vector = shares_query_vector
        self.query_prefix = ""
        self.model = SimpleNamespace(theta_low=_THETA_LOW, theta_high=_THETA_HIGH, band_k=1)
        twin = FakeEmbedder()
        self._scores = {
            tuple(twin.generate_embedding(text)): p for text, p in (scores or {}).items()
        }
        self.default = default
        self.events = events
        self.refuse = refuse
        self.before_decide = before_decide
        self.embedded: list[tuple[str, str]] = []
        self.decided: list[float] = []

    def embed_query(self, text: str) -> list[float]:
        self.embedded.append((text, threading.current_thread().name))
        return self.embedder.generate_embedding(self.query_prefix + text)  # type: ignore[attr-defined]

    def decide(self, vector: Sequence[float]) -> NeedDecision:
        if self.refuse:
            raise ValueError("a query vector of length 3; the bank's vectors have length 384")
        if self.before_decide is not None:
            self.before_decide()
        p = self._scores.get(tuple(vector), self.default)
        self.decided.append(p)
        if self.events is not None:
            self.events.append("decide")
        band = "closed" if p < _THETA_LOW else "band" if p < _THETA_HIGH else "open"
        return NeedDecision(
            band=band,  # type: ignore[arg-type]
            p=p,
            theta_low=_THETA_LOW,
            theta_high=_THETA_HIGH,
            band_k=1,
            need_ids=_NEED_IDS,
            no_need_ids=_NO_NEED_IDS,
        )

    def texts(self) -> list[str]:
        return [text for text, _ in self.embedded]


class WrappingEmbedder:
    """Another object around the same model, as the harness's `--trace` wraps the embedder."""

    def __init__(self, inner: FakeEmbedder) -> None:
        self.inner = inner
        self.calls = 0

    def generate_embedding(self, text: str) -> list[float]:
        self.calls += 1
        return self.inner.generate_embedding(text)

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self.inner.generate_embeddings(texts)


class HookedEmbedder(FakeEmbedder):
    """A FakeEmbedder that runs `hook(text)` before embedding a text `applies(text)` selects."""

    def __init__(
        self, hook: Callable[[str], None], applies: Callable[[str], bool] = lambda _: True
    ) -> None:
        super().__init__()
        self.hook = hook
        self.applies = applies

    def generate_embedding(self, text: str) -> list[float]:
        if self.applies(text):
            self.hook(text)
        return super().generate_embedding(text)


class FailingOn(FakeEmbedder):
    """Raises the production embedder's error for a text `applies(text)` selects."""

    def __init__(self, applies: Callable[[str], bool] = lambda _: True) -> None:
        super().__init__()
        self.applies = applies

    def generate_embedding(self, text: str) -> list[float]:
        if self.applies(text):
            raise EmbeddingError("embedder down")
        return super().generate_embedding(text)


class RecordingExecutor(Executor):
    """Records each submission by the executor's name in a shared event list."""

    def __init__(self, inner: Executor, name: str, events: list[str]) -> None:
        self.inner = inner
        self.name = name
        self.events = events

    def submit(self, fn, /, *args, **kwargs) -> Future:  # type: ignore[no-untyped-def]
        self.events.append(self.name)
        return self.inner.submit(fn, *args, **kwargs)

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        """The fixture owns the inner executor."""


def _is_context(text: str) -> bool:
    return "\n" in text


def _without_need(outcome: FetchOutcome) -> FetchOutcome:
    result = outcome.result
    return replace(outcome, result=replace(result, stage=replace(result.stage, need=None)))


def _trace(band: str, p: float | None, **overrides: object) -> NeedTrace:
    values: dict[str, object] = {
        "mode": "on",
        "band": band,
        "p": p,
        "theta_low": _THETA_LOW,
        "theta_high": _THETA_HIGH,
        "band_k": 1,
        "need_ids": _NEED_IDS,
        "no_need_ids": _NO_NEED_IDS,
        "context": False,
        "context_p": None,
        "degraded": None,
    }
    values.update(overrides)
    return NeedTrace(**values)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_a_closed_gate_serves_no_block_and_makes_no_rerank_call(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    fact_index.upsert([_fact("Tidewater Survey")])
    reranker = FakeReranker()
    gate = FakeGate({_TIMER: 0.1})

    result = await _fetcher(index, executors, reranker=reranker, need_gate=gate).fetch(
        _request(_TIMER)
    )

    assert result.items == () and result.block_text is None and not result.degraded
    assert result.stage.ran is False and result.stage.skip_reason == "no_need"
    assert result.stage.need == _trace("closed", 0.1)
    assert reranker.calls == [] and result.reranked == 0


@pytest.mark.asyncio
async def test_a_band_turn_proceeds_as_open_and_traces_its_band(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    fact_index.upsert([_fact("Tidewater Survey")])
    reranker = FakeReranker()
    gate = FakeGate({"tidewater survey": 0.45})

    result = await _fetcher(index, executors, reranker=reranker, need_gate=gate).fetch(
        _request("tidewater survey")
    )

    assert [item.item_id for item in result.items] == [_key("Tidewater Survey")]
    assert result.stage.ran and result.stage.need == _trace("band", 0.45)
    assert reranker.calls


@pytest.mark.asyncio
async def test_an_open_gate_serves_the_block_and_traces_its_decision(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    fact_index.upsert([_fact("Tidewater Survey")])
    gate = FakeGate({"tidewater survey": 0.8})

    outcome = await _fetcher(index, executors, need_gate=gate).fetch_outcome(
        _request("tidewater survey")
    )
    ungated = await _fetcher(index, executors).fetch_outcome(_request("tidewater survey"))

    assert outcome.result.stage.need == _trace("open", 0.8)
    assert _without_need(outcome) == ungated


@pytest.mark.parametrize("mode", ["on", "shadow"])
@pytest.mark.asyncio
async def test_without_a_gate_every_fetch_is_open_and_traces_no_need(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors, mode: str
) -> None:
    """`MIST_MEMORY_NEED_GATE=off` gives the fetcher no gate; `need_mode` is then not read."""
    fact_index.upsert([_fact("Tidewater Survey")])
    fetcher = _fetcher(index, executors, need_gate=None, need_mode=mode)

    result = await fetcher.fetch(_request("tidewater survey"))

    assert result.stage.need is None
    assert [item.item_id for item in result.items] == [_key("Tidewater Survey")]


def test_the_degraded_priority_ranks_need_unknown_after_the_index_reasons() -> None:
    assert _DEGRADED_PRIORITY == (
        "index_error",
        "index_rebuilding",
        "need_unknown",
        "deadline",
        "reranker_unavailable",
    )


@pytest.mark.parametrize("wrapped", [False, True], ids=["same-object", "wrapped"])
@pytest.mark.asyncio
async def test_a_gate_sharing_the_query_vector_embeds_once_per_fetch(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors, wrapped: bool
) -> None:
    fact_index.upsert([_fact("Tidewater Survey")])
    model = FakeEmbedder()
    gate = FakeGate(shares_query_vector=True, embedder=model)
    fetch_embedder = WrappingEmbedder(model) if wrapped else model
    fetcher = _fetcher(index, executors, embedder=fetch_embedder, need_gate=gate)

    for n, text in enumerate(("tidewater survey", "where is tidewater survey"), start=1):
        result = await fetcher.fetch(_request(text, session=f"s{n}"))
        assert len(model.calls) == n
        assert result.stage.need is not None

    assert gate.embedded == [] and len(gate.decided) == 2


@pytest.mark.asyncio
async def test_another_gate_embedder_runs_on_the_gate_executor_while_the_model_is_busy(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors, release: threading.Event
) -> None:
    fact_index.upsert([_fact("Tidewater Survey")])
    held = executors.model.submit(release.wait, _WAIT_S)
    embedder = FakeEmbedder()
    gate = FakeGate({_TIMER: 0.1})
    try:
        result = await asyncio.wait_for(
            _fetcher(index, executors, embedder=embedder, need_gate=gate).fetch(_request(_TIMER)),
            timeout=_WAIT_S,
        )

        [(text, thread)] = gate.embedded
        assert text == _TIMER and thread.startswith("memory-gate")
        assert result.stage.skip_reason == "no_need"
        assert embedder.calls == []  # the fetch's own embedding still waits for the model thread
    finally:
        release.set()
        held.result(timeout=_WAIT_S)


@pytest.mark.asyncio
async def test_a_late_gate_vector_closes_with_deadline(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    fact_index.upsert([_fact("Tidewater Survey")])
    clock = FakeClock()
    reranker = FakeReranker()
    gate = FakeGate(embedder=HookedEmbedder(lambda _: clock.advance(ms=_ROOMY_MS + 10)))

    result = await _fetcher(index, executors, clock=clock, reranker=reranker, need_gate=gate).fetch(
        _request("tidewater survey")
    )

    assert result.degraded_reason == "deadline" and result.items == ()
    assert result.stage.ran
    assert result.stage.need == _trace(
        "closed", None, need_ids=(), no_need_ids=(), degraded="deadline"
    )
    assert reranker.calls == []


@pytest.mark.parametrize(
    "case", ["gate-embedder-raises", "decide-refuses", "shared-embedder-raises"]
)
@pytest.mark.asyncio
async def test_a_failed_gate_vector_closes_with_need_unknown(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors, case: str
) -> None:
    fact_index.upsert([_fact("Tidewater Survey")])
    reranker = FakeReranker()
    embedder: object = FakeEmbedder()
    if case == "gate-embedder-raises":
        gate = FakeGate(embedder=FailingEmbedder())
    elif case == "decide-refuses":
        gate = FakeGate(refuse=True)
    else:
        embedder = FailingEmbedder()
        gate = FakeGate(shares_query_vector=True, embedder=embedder)

    result = await _fetcher(
        index, executors, embedder=embedder, reranker=reranker, need_gate=gate
    ).fetch(_request("tidewater survey"))

    assert result.degraded_reason == "need_unknown" and result.items == ()
    assert result.stage.ran
    assert result.stage.need == _trace(
        "closed", None, need_ids=(), no_need_ids=(), degraded="need_unknown"
    )
    assert reranker.calls == []


@pytest.mark.asyncio
async def test_a_prefetch_is_gated_and_its_reused_outcome_keeps_its_need(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    fact_index.upsert([_fact("Tidewater Survey")])
    reranker = FakeReranker()
    gate = FakeGate({_QUERY: 0.1})
    fetcher = _fetcher(index, executors, reranker=reranker, need_gate=gate, need_context=False)
    turns = await _warm(fetcher)
    reranks = len(reranker.calls)
    await fetcher.prefetch("s1", "Where does  the user WORK these days")
    await _settle()
    assert gate.decided[-1] == 0.1
    decided = len(gate.decided)

    result = await fetcher.fetch(_request(_QUERY, turns=turns))

    assert result.stage.prefetch_reuse == "block" and result.stage.skip_reason == "no_need"
    assert result.stage.need == _trace("closed", 0.1)
    assert len(gate.decided) == decided and len(reranker.calls) == reranks


@pytest.mark.asyncio
async def test_a_blank_query_keeps_the_ungated_empty_result(
    index: HookedIndex, executors: FetcherExecutors
) -> None:
    gate = FakeGate(default=0.1)

    gated = await _fetcher(index, executors, need_gate=gate).fetch_outcome(_request("   "))
    ungated = await _fetcher(index, executors).fetch_outcome(_request("   "))

    assert gated == ungated and gated.result.stage.need is None
    assert gate.embedded == [] and gate.decided == []


# -- shadow mode ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_shadow_gate_that_would_close_serves_the_open_block(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    fact_index.upsert([_fact("Tidewater Survey")])
    reranker = FakeReranker()
    gate = FakeGate({"tidewater survey": 0.1})

    shadow = await _fetcher(
        index, executors, reranker=reranker, need_gate=gate, need_mode="shadow"
    ).fetch_outcome(_request("tidewater survey"))
    ungated = await _fetcher(index, executors).fetch_outcome(_request("tidewater survey"))

    stage = shadow.result.stage
    assert stage.ran and stage.skip_reason is None
    assert stage.need == _trace("closed", 0.1, mode="shadow")
    assert [item.item_id for item in shadow.result.items] == [_key("Tidewater Survey")]
    assert reranker.calls
    assert _without_need(shadow) == ungated


@pytest.mark.parametrize(("mode", "p"), [("on", 0.9), ("on", 0.45), ("shadow", 0.1)])
@pytest.mark.asyncio
async def test_the_cross_encoder_starts_after_the_gate_decides(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors, mode: str, p: float
) -> None:
    fact_index.upsert([_fact("Tidewater Survey")])
    events: list[str] = []
    recorded = FetcherExecutors(
        index=executors.index,
        model=RecordingExecutor(executors.model, "model", events),
        gate=RecordingExecutor(executors.gate, "gate", events),
    )
    slow = HookedEmbedder(lambda _: time.sleep(0.2))  # the pools finish long before it
    gate = FakeGate({"tidewater survey": p}, embedder=slow, events=events)
    reranker = FakeReranker()

    result = await _fetcher(
        index, recorded, reranker=reranker, need_gate=gate, need_mode=mode
    ).fetch(_request("tidewater survey"))

    assert reranker.calls and result.items
    assert events.count("model") == 2  # the fetch's own query embedding, then the rerank
    rerank_submitted = len(events) - 1 - events[::-1].index("model")
    assert events.index("gate") < events.index("decide") < rerank_submitted


@pytest.mark.asyncio
async def test_a_late_shadow_vector_leaves_the_outcome_of_an_ungated_fetch_past_its_deadline(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    """The shared gate's vector is the fetch's own; it arrives after the deadline in both runs.

    Both pools and the embedding start before the clock moves, and the pools return only after
    it, so the two runs see the same expired deadline at the same points.
    """
    fact_index.upsert([_fact("Tidewater Survey")])

    async def run(gated: bool) -> FetchOutcome:
        clock = FakeClock()
        started = threading.Barrier(3)
        moved = threading.Event()

        def pool_hook() -> None:
            started.wait(_WAIT_S)
            moved.wait(_WAIT_S)

        def expire(_: str) -> None:
            started.wait(_WAIT_S)
            clock.advance(ms=_ROOMY_MS + 10)
            moved.set()

        index.hooks["search_fts"] = pool_hook
        index.hooks["search_names"] = pool_hook
        embedder = HookedEmbedder(expire)
        gate = FakeGate(shares_query_vector=True, embedder=embedder) if gated else None
        try:
            return await _fetcher(
                index,
                executors,
                clock=clock,
                embedder=embedder,
                need_gate=gate,
                need_mode="shadow",
            ).fetch_outcome(_request("tidewater survey"))
        finally:
            moved.set()
            index.hooks.clear()

    shadow = await run(gated=True)
    ungated = await run(gated=False)

    assert shadow.result.stage.need == _trace(
        "closed", None, mode="shadow", need_ids=(), no_need_ids=(), degraded="deadline"
    )
    assert shadow.result.degraded_reason == "deadline"
    assert _without_need(shadow) == ungated


@pytest.mark.parametrize("case", ["gate-embedder-raises", "decide-refuses"])
@pytest.mark.asyncio
async def test_a_failed_shadow_vector_leaves_the_outcome_of_an_ungated_fetch(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors, case: str
) -> None:
    fact_index.upsert([_fact("Tidewater Survey")])
    gate = (
        FakeGate(embedder=FailingEmbedder())
        if case == "gate-embedder-raises"
        else FakeGate(refuse=True)
    )

    shadow = await _fetcher(index, executors, need_gate=gate, need_mode="shadow").fetch_outcome(
        _request("tidewater survey")
    )
    ungated = await _fetcher(index, executors).fetch_outcome(_request("tidewater survey"))

    assert shadow.result.stage.need == _trace(
        "closed", None, mode="shadow", need_ids=(), no_need_ids=(), degraded="need_unknown"
    )
    assert shadow.result.degraded_reason is None and shadow.result.items
    assert _without_need(shadow) == ungated


# -- follow-ups -------------------------------------------------------------------------------


async def _after_a_served_fetch(fetcher: MemoryFetcher) -> tuple[str, ...]:
    """One fetch of `_FIRST` in session s1; returns the turns the follow-up carries."""
    await fetcher.fetch(_request(_FIRST))
    return (_FIRST,)


@pytest.mark.parametrize(
    ("query_p", "context_p", "band", "p", "moved"),
    [
        (0.1, 0.9, "open", 0.9, True),
        (0.45, 0.9, "open", 0.9, True),
        (0.1, 0.45, "band", 0.45, True),
        (0.1, 0.2, "closed", 0.2, False),
        (0.45, 0.35, "band", 0.45, False),
    ],
    ids=["closed-to-open", "band-to-open", "closed-to-band", "higher-same-band", "lower"],
)
@pytest.mark.asyncio
async def test_a_follow_up_takes_the_higher_probability_and_its_band(
    fact_index: FactIndex,
    index: HookedIndex,
    executors: FetcherExecutors,
    query_p: float,
    context_p: float,
    band: str,
    p: float,
    moved: bool,
) -> None:
    fact_index.upsert([_fact("Tidewater Survey")])
    gate = FakeGate({_FIRST: 0.9, _FOLLOW: query_p, _CONTEXT: context_p})
    fetcher = _fetcher(index, executors, need_gate=gate)
    turns = await _after_a_served_fetch(fetcher)

    result = await fetcher.fetch(_request(_FOLLOW, turns=turns))

    assert result.stage.need == _trace(band, p, context=moved, context_p=context_p)
    assert (result.stage.skip_reason == "no_need") == (band == "closed")


@pytest.mark.asyncio
async def test_an_open_query_does_not_take_the_context_probability(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    fact_index.upsert([_fact("Tidewater Survey")])
    gate = FakeGate({_FIRST: 0.9, _FOLLOW: 0.7, _CONTEXT: 0.95})
    fetcher = _fetcher(index, executors, need_gate=gate)
    turns = await _after_a_served_fetch(fetcher)

    result = await fetcher.fetch(_request(_FOLLOW, turns=turns))

    assert result.stage.need == _trace("open", 0.7)


@pytest.mark.parametrize(
    "case", ["no-recent-turns", "after-a-closed-fetch", "context-off", "after-a-shadow-close"]
)
@pytest.mark.asyncio
async def test_no_context_embedding_without_turns_a_served_fetch_or_the_setting(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors, case: str
) -> None:
    fact_index.upsert([_fact("Tidewater Survey")])
    first_p = 0.1 if case in ("after-a-closed-fetch", "after-a-shadow-close") else 0.9
    gate = FakeGate({_FIRST: first_p, _FOLLOW: 0.1, _CONTEXT: 0.9})
    fetcher = _fetcher(
        index,
        executors,
        need_gate=gate,
        need_context=case != "context-off",
        need_mode="shadow" if case == "after-a-shadow-close" else "on",
    )
    turns = await _after_a_served_fetch(fetcher)

    result = await fetcher.fetch(
        _request(_FOLLOW, turns=() if case == "no-recent-turns" else turns)
    )

    assert gate.texts() == [_FIRST, _FOLLOW]
    assert result.stage.need is not None and result.stage.need.context_p is None


@pytest.mark.asyncio
async def test_a_shadow_fetch_that_decided_band_starts_the_context_path(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    fact_index.upsert([_fact("Tidewater Survey")])
    gate = FakeGate({_FIRST: 0.45, _FOLLOW: 0.1, _CONTEXT: 0.9})
    fetcher = _fetcher(index, executors, need_gate=gate, need_mode="shadow")
    turns = await _after_a_served_fetch(fetcher)

    result = await fetcher.fetch(_request(_FOLLOW, turns=turns))

    assert gate.texts() == [_FIRST, _FOLLOW, _CONTEXT]
    assert result.stage.need == _trace("open", 0.9, mode="shadow", context=True, context_p=0.9)


@pytest.mark.asyncio
async def test_the_context_embedding_runs_on_the_gate_executor_after_the_query(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    fact_index.upsert([_fact("Tidewater Survey")])
    gate = FakeGate({_FIRST: 0.9, _FOLLOW: 0.1, _CONTEXT: 0.9})
    fetcher = _fetcher(index, executors, need_gate=gate)
    turns = await _after_a_served_fetch(fetcher)
    gate.embedded.clear()

    await fetcher.fetch(_request(_FOLLOW, turns=turns))

    assert gate.texts() == [_FOLLOW, _CONTEXT]
    assert all(thread.startswith("memory-gate") for _, thread in gate.embedded)


@pytest.mark.asyncio
async def test_an_open_query_never_waits_for_its_context_vector(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors, release: threading.Event
) -> None:
    fact_index.upsert([_fact("Tidewater Survey")])
    started = threading.Event()

    def hold(_: str) -> None:
        started.set()
        release.wait(_WAIT_S)

    blocked = HookedEmbedder(hold, applies=_is_context)
    gate = FakeGate({_FIRST: 0.9, _FOLLOW: 0.8, _CONTEXT: 0.1}, embedder=blocked)
    fetcher = _fetcher(index, executors, need_gate=gate)
    turns = await _after_a_served_fetch(fetcher)
    # The query decides only once the context embedding holds the gate thread.
    gate.before_decide = lambda: started.wait(_WAIT_S)

    result = await asyncio.wait_for(
        fetcher.fetch(_request(_FOLLOW, turns=turns)), timeout=_WAIT_S / 2
    )

    assert started.is_set() and not release.is_set()
    assert result.stage.need == _trace("open", 0.8)
    assert result.stage.ran and result.stage.skip_reason is None


@pytest.mark.asyncio
async def test_a_closed_query_waits_for_its_context_vector(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    fact_index.upsert([_fact("Tidewater Survey")])
    slow = HookedEmbedder(lambda _: time.sleep(0.2), applies=_is_context)
    gate = FakeGate({_FIRST: 0.9, _FOLLOW: 0.1, _CONTEXT: 0.9}, embedder=slow)
    fetcher = _fetcher(index, executors, need_gate=gate)
    turns = await _after_a_served_fetch(fetcher)

    result = await fetcher.fetch(_request(_FOLLOW, turns=turns))

    assert result.stage.need == _trace("open", 0.9, context=True, context_p=0.9)
    assert result.stage.ran and result.stage.skip_reason is None


@pytest.mark.parametrize(("case", "reason"), [("late", "deadline"), ("raises", "need_unknown")])
@pytest.mark.asyncio
async def test_a_late_or_failed_context_vector_leaves_the_gate_closed(
    fact_index: FactIndex,
    index: HookedIndex,
    executors: FetcherExecutors,
    case: str,
    reason: str,
) -> None:
    fact_index.upsert([_fact("Tidewater Survey")])
    clock = FakeClock()
    decided = threading.Event()

    def expire(_: str) -> None:
        decided.wait(_WAIT_S)  # the query's vector is in time and decided first
        clock.advance(ms=_ROOMY_MS + 10)

    embedder = (
        HookedEmbedder(expire, applies=_is_context) if case == "late" else FailingOn(_is_context)
    )
    gate = FakeGate({_FIRST: 0.9, _FOLLOW: 0.1, _CONTEXT: 0.9}, embedder=embedder)
    fetcher = _fetcher(index, executors, clock=clock, need_gate=gate)
    turns = await _after_a_served_fetch(fetcher)
    gate.before_decide = decided.set

    result = await fetcher.fetch(_request(_FOLLOW, turns=turns))

    assert result.degraded_reason == reason and result.items == ()
    assert result.stage.need == _trace("closed", 0.1, degraded=reason)


@pytest.mark.asyncio
async def test_a_shared_embedder_is_never_called_from_two_threads_at_once(
    fact_index: FactIndex, index: HookedIndex, executors: FetcherExecutors
) -> None:
    """Under `shared` the context embedding (gate thread) and the fetch's own query embedding
    (model thread) use one embedder object; the fetcher serializes the two calls.
    """
    fact_index.upsert([_fact("Tidewater Survey")])
    lock = threading.Lock()
    active = [0]
    peak = [0]
    threads: list[str] = []

    def overlap(_: str) -> None:
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
            threads.append(threading.current_thread().name)
        time.sleep(0.1)
        with lock:
            active[0] -= 1

    shared = HookedEmbedder(overlap)
    gate = FakeGate(
        {_FIRST: 0.9, _FOLLOW: 0.1, _CONTEXT: 0.9}, shares_query_vector=True, embedder=shared
    )
    fetcher = _fetcher(index, executors, embedder=shared, need_gate=gate)
    turns = await _after_a_served_fetch(fetcher)
    threads.clear()

    result = await fetcher.fetch(_request(_FOLLOW, turns=turns))

    assert sorted(name.split("_")[0] for name in threads) == ["memory-gate", "memory-model"]
    assert peak[0] == 1
    assert result.stage.need == _trace("open", 0.9, context=True, context_p=0.9)


# -- helpers ----------------------------------------------------------------------------------


def test_normalise_collapses_whitespace_and_folds_case() -> None:
    assert normalise("  Where does\tthe USER  work ") == "where does the user work"


def test_jaccard_compares_token_sets() -> None:
    assert jaccard("a b c d", "a b c d") == 1.0
    assert jaccard("a b c d", "a b c e") == pytest.approx(3 / 5)
    assert jaccard("", "") == 1.0
