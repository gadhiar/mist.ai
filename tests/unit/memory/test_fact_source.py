"""Tests for backend.memory.fact_source: the graph fact source and the index refresher.

The graph is a FakeGraphExecutor whose connection answers `current_facts_query` the way Neo4j
would (filter on the page key, order, limit), and the index is a real in-memory FactIndex
with FakeEmbedder. The refresher's clock is FakeClock and its sleep is a ManualSleeper that
returns only when the test releases it, so the debounce is driven step by step and no test
waits on the wall clock.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from datetime import UTC

import pytest
from neo4j.exceptions import ServiceUnavailable

from backend.errors import (
    MemoryConfigError,
    MemoryIndexError,
    Neo4jConnectionError,
    Neo4jQueryError,
)
from backend.knowledge.storage.graph_store import current_facts_query
from backend.memory.fact_index import DiffReport, FactIndex, fact_row_key, structural_row_key
from backend.memory.fact_source import (
    DEFAULT_PAGE_SIZE,
    FactIndexRefresher,
    FactSnapshotSource,
    GraphFactSource,
)
from backend.memory.index_db import open_index, open_reader
from backend.memory.models import FactRow, fact_key
from backend.memory.ranking import MISSING_FACT_CONFIDENCE
from tests.fakes.memory_fakes import FakeClock, FakeEmbedder
from tests.mocks.neo4j import FakeGraphExecutor, FakeNeo4jConnection

_MODEL = "all-MiniLM-L6-v2"
_DEBOUNCE_S = 30.0
_WAIT_S = 5.0


# -- the fake graph -----------------------------------------------------------------------------


def _record(
    page_key: str,
    subject_id: str,
    rel_type: str,
    object_id: str,
    version_key: str | None = "e1|open|open",
    **overrides: object,
) -> dict[str, object]:
    """One row as `current_facts_query` returns it."""
    record: dict[str, object] = {
        "page_key": page_key,
        "subject_id": subject_id,
        "subject_name": subject_id.title(),
        "subject_type": "User" if subject_id == "user" else "Concept",
        "rel_type": rel_type,
        "object_id": object_id,
        "object_name": object_id.title(),
        "object_type": "Technology",
        "version_key": version_key,
        "valid_from": "2026-01-01T00:00:00+00:00",
        "valid_to": None,
        "recorded_at": "2026-01-01T00:00:00+00:00",
        "confidence": 0.9,
        "source_utterance_id": "utt-1",
    }
    record.update(overrides)
    return record


class FakeGraph:
    """Answers `current_facts_query` pages from a list of records, as Neo4j would.

    `fail_after` makes the page after that many successful pages raise `error`, to stand in
    for a graph that goes away mid-scan.
    """

    def __init__(self, records: Sequence[dict[str, object]] = ()) -> None:
        self.records = list(records)
        self.fail_after: int | None = None
        self.error: BaseException = Neo4jQueryError("graph went away")
        self.pages = 0
        self.connection = FakeNeo4jConnection(query_router=self._route)
        self.executor = FakeGraphExecutor(self.connection)

    def _route(self, query: str, params: dict[str, object] | None) -> list[dict[str, object]]:
        assert params is not None
        if self.fail_after is not None and self.pages >= self.fail_after:
            raise self.error
        self.pages += 1
        after = params["after_key"]
        rows = sorted(
            (r for r in self.records if after is None or r["page_key"] > after),
            key=lambda r: r["page_key"],
        )
        return rows[: params["limit"]]

    @property
    def requests(self) -> list[tuple[str, dict[str, object] | None]]:
        return self.connection.queries


def _facts(count: int, *, prefix: str = "tech") -> list[dict[str, object]]:
    return [
        _record(f"k{index:06d}", "user", "USES", f"{prefix}-{index:06d}") for index in range(count)
    ]


# -- refresher harness --------------------------------------------------------------------------


class InlineExecutor(Executor):
    """Runs each submitted call at once on the calling thread, so tests stay on the loop."""

    def __init__(self) -> None:
        self.calls = 0

    def submit(self, fn: Callable[..., object], /, *args: object, **kwargs: object) -> Future:
        self.calls += 1
        future: Future = Future()
        future.set_result(fn(*args, **kwargs))
        return future


class ManualSleeper:
    """An injected sleep that returns only when the test releases it, advancing the clock."""

    def __init__(self, clock: FakeClock) -> None:
        self._clock = clock
        self.requests: list[float] = []
        self._pending: list[tuple[float, asyncio.Future[None]]] = []

    async def __call__(self, seconds: float) -> None:
        self.requests.append(seconds)
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._pending.append((seconds, future))
        await future

    @property
    def waiting(self) -> int:
        return sum(1 for _, future in self._pending if not future.done())

    def release(self) -> None:
        seconds, future = self._pending.pop(0)
        self._clock.advance(seconds=seconds)
        future.set_result(None)


class GatedSource:
    """A snapshot source that blocks each scan until the test opens the gate."""

    def __init__(self, inner: GraphFactSource) -> None:
        self._inner = inner
        self.started = 0
        self.gate = asyncio.Event()

    async def snapshot(self) -> list[FactRow]:
        self.started += 1
        await self.gate.wait()
        return await self._inner.snapshot()


class SpyIndex:
    """Delegates to a FactIndex and counts `diff_apply` calls; optionally blocks inside one."""

    def __init__(self, index: FactIndex, *, block: threading.Event | None = None) -> None:
        self._index = index
        self._block = block
        self.entered = threading.Event()
        self.applies = 0
        self.finished = 0

    def diff_apply(self, snapshot: Sequence[FactRow]) -> DiffReport:
        self.applies += 1
        self.entered.set()
        if self._block is not None:
            assert self._block.wait(_WAIT_S)
        report = self._index.diff_apply(snapshot)
        self.finished += 1
        return report


async def _settle() -> None:
    """Let every ready task run until each is waiting on something the test controls."""
    for _ in range(50):
        await asyncio.sleep(0)


@pytest.fixture
def index(fake_embedder: FakeEmbedder) -> Iterator[FactIndex]:
    writer = open_index(":memory:", embedding_model_id=_MODEL)
    yield FactIndex(writer, fake_embedder, tz=UTC, reader=writer)
    writer.close()


def _refresher(
    graph: FakeGraph,
    index: FactIndex | SpyIndex,
    clock: FakeClock,
    sleeper: ManualSleeper,
    *,
    periodic: bool = False,
    executor: Executor | None = None,
    source: FactSnapshotSource | None = None,
) -> FactIndexRefresher:
    return FactIndexRefresher(
        source if source is not None else GraphFactSource(graph.executor, page_size=2),
        index,
        executor if executor is not None else InlineExecutor(),
        clock=clock,
        debounce_s=_DEBOUNCE_S,
        periodic=periodic,
        sleep=sleeper,
    )


# -- GraphFactSource ----------------------------------------------------------------------------


class TestGraphFactSourcePaging:
    @pytest.mark.asyncio
    async def test_requests_pages_after_the_last_key_until_a_short_page(self):
        graph = FakeGraph(_facts(5))
        source = GraphFactSource(graph.executor, page_size=2)

        rows = await source.snapshot()

        assert len(rows) == 5
        assert graph.requests == [
            current_facts_query(None, 2),
            current_facts_query("k000001", 2),
            current_facts_query("k000003", 2),
        ]

    @pytest.mark.asyncio
    async def test_a_full_last_page_costs_one_empty_request(self):
        graph = FakeGraph(_facts(4))
        source = GraphFactSource(graph.executor, page_size=2)

        rows = await source.snapshot()

        assert len(rows) == 4
        assert [params["after_key"] for _, params in graph.requests] == [
            None,
            "k000001",
            "k000003",
        ]

    @pytest.mark.asyncio
    async def test_an_empty_graph_is_an_empty_snapshot(self):
        graph = FakeGraph()

        assert await GraphFactSource(graph.executor, page_size=10).snapshot() == []
        assert len(graph.requests) == 1

    @pytest.mark.asyncio
    async def test_a_page_key_that_does_not_advance_raises(self):
        stuck = [_record("k1", "user", "USES", "python"), _record("k1", "user", "USES", "rust")]
        connection = FakeNeo4jConnection(query_results=stuck)
        source = GraphFactSource(FakeGraphExecutor(connection), page_size=2)

        with pytest.raises(MemoryIndexError, match="page key"):
            await source.snapshot()
        assert len(connection.queries) == 2

    def test_rejects_a_non_positive_page_size(self):
        with pytest.raises(ValueError, match="page_size"):
            GraphFactSource(FakeGraphExecutor(), page_size=0)


class TestGraphFactSourceRows:
    @pytest.mark.asyncio
    async def test_maps_each_record_to_a_fact_row_keyed_through_fact_row_key(self):
        graph = FakeGraph(
            [
                _record(
                    "k1",
                    "user",
                    "WORKS_AT",
                    "acme",
                    "e7|2025-01-01|open",
                    subject_name="User",
                    object_name="Acme Labs",
                    object_type="Organization",
                    valid_from="2025-01-01T00:00:00+00:00",
                    valid_to="2026-02-01T00:00:00+00:00",
                    recorded_at="2025-01-03T10:00:00+00:00",
                    confidence=0.75,
                    source_utterance_id="utt-7",
                )
            ]
        )

        (row,) = await GraphFactSource(graph.executor, page_size=10).snapshot()

        key = fact_key("user", "WORKS_AT", "acme")
        assert row == FactRow(
            row_key=fact_row_key(key, "e7|2025-01-01|open"),
            fact_key=key,
            subject_id="user",
            subject_name="User",
            subject_type="User",
            rel_type="WORKS_AT",
            object_id="acme",
            object_name="Acme Labs",
            object_type="Organization",
            valid_from="2025-01-01T00:00:00+00:00",
            valid_to="2026-02-01T00:00:00+00:00",
            recorded_at="2025-01-03T10:00:00+00:00",
            confidence=0.75,
            source_utterance_id="utt-7",
        )

    @pytest.mark.asyncio
    async def test_a_structural_edge_uses_the_structural_key(self):
        graph = FakeGraph([_record("k1", "python", "IS_A", "language", None)])

        (row,) = await GraphFactSource(graph.executor, page_size=10).snapshot()

        key = fact_key("python", "IS_A", "language")
        assert row.row_key == structural_row_key(key) == fact_row_key(key, None)

    @pytest.mark.asyncio
    async def test_the_same_version_key_on_two_facts_gives_two_row_keys(self):
        # Seed edges all carry 'seed'; version keys are unique only within one triple.
        graph = FakeGraph(
            [
                _record("k1", "user", "USES", "python", "seed"),
                _record("k2", "user", "USES", "rust", "seed"),
            ]
        )

        rows = await GraphFactSource(graph.executor, page_size=10).snapshot()

        assert len({row.row_key for row in rows}) == 2
        assert all(row.row_key != "seed" for row in rows)

    @pytest.mark.asyncio
    async def test_versions_of_one_fact_each_keep_a_row(self):
        graph = FakeGraph(
            [
                _record("k1", "user", "USES", "vim", "e1|2024|2025", valid_to="2025"),
                _record("k2", "user", "USES", "vim", "e9|2026|open", valid_from="2026"),
                _record("k3", "user", "USES", "vim", "e5|t|t", valid_from="t", valid_to="t"),
            ]
        )

        rows = await GraphFactSource(graph.executor, page_size=2).snapshot()

        assert len({row.row_key for row in rows}) == 3
        assert len({row.fact_key for row in rows}) == 1

    @pytest.mark.asyncio
    async def test_an_undirected_fact_keys_in_canonical_order(self):
        graph = FakeGraph([_record("k1", "zeta", "RELATED_TO", "alpha", None)])

        (row,) = await GraphFactSource(graph.executor, page_size=10).snapshot()

        assert row.fact_key == fact_key("alpha", "RELATED_TO", "zeta")
        assert (row.subject_id, row.object_id) == ("zeta", "alpha")

    @pytest.mark.asyncio
    async def test_both_orientations_of_one_undirected_version_index_once(self, caplog):
        graph = FakeGraph(
            [
                _record("k2", "zeta", "RELATED_TO", "alpha", "e1|open|open"),
                _record("k1", "alpha", "RELATED_TO", "zeta", "e1|open|open"),
            ]
        )

        with caplog.at_level(logging.WARNING, logger="backend.memory.fact_source"):
            rows = await GraphFactSource(graph.executor, page_size=10).snapshot()

        assert len(rows) == 1
        # Kept by content, not by page order, so a rebuild keeps the same one.
        assert (rows[0].subject_id, rows[0].object_id) == ("alpha", "zeta")
        assert fact_key("alpha", "RELATED_TO", "zeta") in caplog.text

    @pytest.mark.parametrize("open_first", [True, False])
    @pytest.mark.asyncio
    async def test_the_same_row_wins_whatever_the_page_order(self, open_first):
        # Two parallel edges of one triple with one version key differ only in `valid_to`;
        # the content order puts the open one (no `valid_to`) first.
        open_edge = _record("k1", "user", "USES", "vim", "e1|x|open", valid_to=None)
        closed_edge = _record("k2", "user", "USES", "vim", "e1|x|open", valid_to="2026-05")
        if not open_first:
            open_edge["page_key"], closed_edge["page_key"] = "k2", "k1"
        graph = FakeGraph([open_edge, closed_edge])

        (row,) = await GraphFactSource(graph.executor, page_size=10).snapshot()

        assert row.valid_to is None

    @pytest.mark.asyncio
    async def test_missing_optional_properties(self):
        graph = FakeGraph(
            [
                _record(
                    "k1",
                    "user",
                    "USES",
                    "python",
                    confidence=None,
                    recorded_at=None,
                    valid_from=None,
                    source_utterance_id=None,
                )
            ]
        )

        (row,) = await GraphFactSource(graph.executor, page_size=10).snapshot()

        assert row.confidence == MISSING_FACT_CONFIDENCE
        assert row.recorded_at is None
        assert row.valid_from is None
        assert row.source_utterance_id is None

    @pytest.mark.parametrize("missing", [None, ""])
    @pytest.mark.asyncio
    async def test_a_node_without_an_entity_type_gives_no_type(self, missing):
        graph = FakeGraph(
            [_record("k1", "user", "USES", "python", subject_type=missing, object_type=missing)]
        )

        (row,) = await GraphFactSource(graph.executor, page_size=10).snapshot()

        assert (row.subject_type, row.object_type) == (None, None)

    @pytest.mark.asyncio
    async def test_a_record_without_type_keys_gives_no_type(self):
        record = _record("k1", "user", "USES", "python")
        del record["subject_type"], record["object_type"]

        (row,) = await GraphFactSource(FakeGraph([record]).executor, page_size=10).snapshot()

        assert (row.subject_type, row.object_type) == (None, None)


class TestGraphFactSourceErrors:
    @pytest.mark.asyncio
    async def test_a_query_error_propagates(self):
        graph = FakeGraph(_facts(5))
        graph.fail_after = 1

        with pytest.raises(Neo4jQueryError):
            await GraphFactSource(graph.executor, page_size=2).snapshot()

    @pytest.mark.asyncio
    async def test_a_driver_error_becomes_a_connection_error(self):
        graph = FakeGraph(_facts(5))
        graph.fail_after = 0
        graph.error = ServiceUnavailable("down")

        with pytest.raises(Neo4jConnectionError):
            await GraphFactSource(graph.executor, page_size=2).snapshot()


# -- FactIndexRefresher: one-shot refresh -------------------------------------------------------


class TestOneShotRefresh:
    @pytest.mark.asyncio
    async def test_returns_the_diff_report(self, index, fake_clock):
        graph = FakeGraph(_facts(5))
        refresher = _refresher(graph, index, fake_clock, ManualSleeper(fake_clock))

        first = await refresher.refresh()
        second = await refresher.refresh()

        assert first == DiffReport(inserted=5, embedded=5)
        assert second == DiffReport(unchanged=5)
        assert index.count() == 5

    @pytest.mark.asyncio
    async def test_reembeds_only_rows_that_changed(self, index, fake_clock, fake_embedder):
        graph = FakeGraph(_facts(5))
        refresher = _refresher(graph, index, fake_clock, ManualSleeper(fake_clock))
        await refresher.refresh()
        fake_embedder.calls.clear()

        graph.records[0]["object_name"] = "Renamed"  # text changes: re-embed
        graph.records[1]["confidence"] = 0.1  # stored field only: rewrite, no embedding
        del graph.records[4]
        graph.records.append(_record("k999999", "user", "USES", "new-tool"))
        report = await refresher.refresh()

        assert report == DiffReport(inserted=1, updated=2, deleted=1, unchanged=2, embedded=2)
        assert len(fake_embedder.calls) == 2
        assert any("Renamed" in text for text in fake_embedder.calls)

    @pytest.mark.asyncio
    async def test_runs_the_diff_on_the_injected_executor(self, index, fake_clock):
        executor = InlineExecutor()
        refresher = _refresher(
            FakeGraph(_facts(3)), index, fake_clock, ManualSleeper(fake_clock), executor=executor
        )

        await refresher.refresh()

        assert executor.calls == 1

    @pytest.mark.asyncio
    async def test_a_graph_error_mid_scan_leaves_the_index_untouched(
        self, index, fake_clock, fake_embedder
    ):
        graph = FakeGraph(_facts(5))
        spy = SpyIndex(index)
        refresher = _refresher(graph, spy, fake_clock, ManualSleeper(fake_clock))
        await refresher.refresh()
        keys = [row.row_key for row in await GraphFactSource(graph.executor).snapshot()]
        before = [index.get(key) for key in keys]
        embedded = len(fake_embedder.calls)

        del graph.records[1:]  # a partial snapshot would delete four rows
        graph.records.extend(_facts(3, prefix="other"))
        graph.pages = 0
        graph.fail_after = 1
        with pytest.raises(Neo4jQueryError):
            await refresher.refresh()

        assert spy.applies == 1
        assert index.count() == 5
        assert [index.get(key) for key in keys] == before
        assert len(fake_embedder.calls) == embedded


# -- FactIndexRefresher: mark_dirty -------------------------------------------------------------


class TestMarkDirty:
    def test_is_a_plain_synchronous_method(self, index, fake_clock):
        refresher = _refresher(FakeGraph(), index, fake_clock, ManualSleeper(fake_clock))

        assert not inspect.iscoroutinefunction(refresher.mark_dirty)
        assert inspect.signature(refresher.mark_dirty).parameters == {}
        assert refresher.mark_dirty() is None

    @pytest.mark.asyncio
    async def test_returns_without_awaiting_a_running_rescan(self, index, fake_clock):
        graph = FakeGraph(_facts(3))
        source = GatedSource(GraphFactSource(graph.executor, page_size=2))
        refresher = _refresher(graph, index, fake_clock, ManualSleeper(fake_clock), source=source)
        refresher.start()
        await _settle()
        assert source.started == 1  # the startup scan is blocked inside the source

        result = refresher.mark_dirty()

        assert result is None
        assert source.started == 1
        assert index.count() == 0
        source.gate.set()
        await refresher.stop()

    @pytest.mark.asyncio
    async def test_does_not_scan_by_itself(self, index, fake_clock):
        graph = FakeGraph(_facts(3))
        refresher = _refresher(graph, index, fake_clock, ManualSleeper(fake_clock))

        refresher.mark_dirty()
        await _settle()

        assert graph.requests == []


# -- FactIndexRefresher: the loop ---------------------------------------------------------------


class TestRefreshLoop:
    @pytest.mark.asyncio
    async def test_refreshes_once_at_startup_then_waits_for_a_mark(self, index, fake_clock):
        graph = FakeGraph(_facts(3))
        sleeper = ManualSleeper(fake_clock)
        refresher = _refresher(graph, index, fake_clock, sleeper)

        refresher.start()
        await _settle()

        assert index.count() == 3
        assert graph.pages == 2
        assert sleeper.requests == []
        await refresher.stop()

    @pytest.mark.asyncio
    async def test_marks_are_debounced_to_one_refresh_per_interval(self, index, fake_clock):
        graph = FakeGraph(_facts(3))
        sleeper = ManualSleeper(fake_clock)
        refresher = _refresher(graph, index, fake_clock, sleeper)
        refresher.start()
        await _settle()
        scans_after_startup = graph.pages

        fake_clock.advance(seconds=10)
        refresher.mark_dirty()
        await _settle()
        refresher.mark_dirty()
        refresher.mark_dirty()
        await _settle()

        # Ten seconds after the startup refresh, the next may start twenty seconds later.
        assert sleeper.requests == [pytest.approx(_DEBOUNCE_S - 10)]
        assert graph.pages == scans_after_startup
        graph.records.append(_record("k999999", "user", "USES", "late"))

        sleeper.release()
        await _settle()

        assert index.count() == 4
        assert graph.pages == scans_after_startup + 3  # four rows in pages of two, then empty
        assert sleeper.waiting == 0
        await refresher.stop()

    @pytest.mark.asyncio
    async def test_a_mark_after_a_quiet_interval_refreshes_at_once(self, index, fake_clock):
        graph = FakeGraph(_facts(1))
        sleeper = ManualSleeper(fake_clock)
        refresher = _refresher(graph, index, fake_clock, sleeper)
        refresher.start()
        await _settle()

        fake_clock.advance(seconds=_DEBOUNCE_S + 1)
        graph.records.append(_record("k999999", "user", "USES", "late"))
        refresher.mark_dirty()
        await _settle()

        assert sleeper.requests == []
        assert index.count() == 2
        await refresher.stop()

    @pytest.mark.asyncio
    async def test_a_mark_during_a_scan_brings_another_refresh(self, index, fake_clock):
        graph = FakeGraph(_facts(2))
        source = GatedSource(GraphFactSource(graph.executor, page_size=2))
        sleeper = ManualSleeper(fake_clock)
        refresher = _refresher(graph, index, fake_clock, sleeper, source=source)
        refresher.start()
        await _settle()

        refresher.mark_dirty()  # lands while the startup scan is in flight
        source.gate.set()
        await _settle()

        assert source.started == 1
        assert sleeper.requests == [pytest.approx(_DEBOUNCE_S)]
        sleeper.release()
        await _settle()
        assert source.started == 2
        await refresher.stop()

    @pytest.mark.asyncio
    async def test_periodic_mode_refreshes_on_the_interval_without_marks(self, index, fake_clock):
        graph = FakeGraph(_facts(1))
        sleeper = ManualSleeper(fake_clock)
        refresher = _refresher(graph, index, fake_clock, sleeper, periodic=True)
        refresher.start()
        await _settle()

        assert sleeper.requests == [pytest.approx(_DEBOUNCE_S)]
        graph.records.append(_record("k999999", "user", "USES", "late"))
        sleeper.release()
        await _settle()

        assert index.count() == 2
        assert sleeper.requests == [pytest.approx(_DEBOUNCE_S)] * 2
        await refresher.stop()

    def test_periodic_mode_needs_a_positive_interval(self, index, fake_clock):
        with pytest.raises(MemoryConfigError, match="MIST_MEMORY_REFRESH_DEBOUNCE_S"):
            FactIndexRefresher(
                GraphFactSource(FakeGraphExecutor()),
                index,
                InlineExecutor(),
                clock=fake_clock,
                debounce_s=0,
                periodic=True,
            )

    @pytest.mark.asyncio
    async def test_graph_unreachable_is_logged_the_index_kept_and_retried(
        self, index, fake_clock, caplog
    ):
        graph = FakeGraph(_facts(3))
        sleeper = ManualSleeper(fake_clock)
        refresher = _refresher(graph, index, fake_clock, sleeper)
        await refresher.refresh()
        graph.records.append(_record("k999999", "user", "USES", "late"))
        graph.pages = 0
        graph.fail_after = 0
        graph.error = ServiceUnavailable("down")

        with caplog.at_level(logging.WARNING, logger="backend.memory.fact_source"):
            refresher.start()
            await _settle()

        assert "keeping the last index" in caplog.text
        assert index.count() == 3
        # The failed refresh leaves the index dirty, so it is retried after the interval.
        assert sleeper.requests == [pytest.approx(_DEBOUNCE_S)]
        graph.fail_after = None
        sleeper.release()
        await _settle()
        assert index.count() == 4
        await refresher.stop()

    @pytest.mark.asyncio
    async def test_a_failed_refresh_waits_a_retry_floor_with_no_debounce(self, index, fake_clock):
        graph = FakeGraph(_facts(1))
        graph.fail_after = 0
        sleeper = ManualSleeper(fake_clock)
        refresher = FactIndexRefresher(
            GraphFactSource(graph.executor),
            index,
            InlineExecutor(),
            clock=fake_clock,
            debounce_s=0,
            periodic=False,
            sleep=sleeper,
        )

        refresher.start()
        await _settle()

        # Without a floor, a debounce of zero would retry a failing graph in a busy loop.
        assert sleeper.requests == [pytest.approx(1.0)]
        assert refresher.running
        await refresher.stop()

    @pytest.mark.asyncio
    async def test_an_embedder_failure_is_logged_and_the_loop_recovers(self, fake_clock, caplog):
        embedder = FlakyEmbedder()
        writer = open_index(":memory:", embedding_model_id=_MODEL)
        index = FactIndex(writer, embedder, tz=UTC, reader=writer)
        graph = FakeGraph(_facts(3))
        sleeper = ManualSleeper(fake_clock)
        refresher = _refresher(graph, index, fake_clock, sleeper)
        embedder.failing = True

        with caplog.at_level(logging.WARNING, logger="backend.memory.fact_source"):
            refresher.start()
            await _settle()

        assert "keeping the last index" in caplog.text
        assert "MemoryIndexError" in caplog.text
        assert refresher.running
        assert index.count() == 0

        embedder.failing = False
        refresher.mark_dirty()
        sleeper.release()
        await _settle()

        assert index.count() == 3
        assert refresher.running
        await refresher.stop()
        writer.close()


class FlakyEmbedder(FakeEmbedder):
    """A FakeEmbedder that raises RuntimeError, as a model failing to load does, while set."""

    def __init__(self) -> None:
        super().__init__()
        self.failing = False

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        if self.failing:
            raise RuntimeError("CUDA out of memory")
        return super().generate_embeddings(texts)


class ExplodingSource:
    """A source raising an error the loop does not contain, to end the loop."""

    async def snapshot(self) -> list[FactRow]:
        raise KeyError("not a MistError")


class TestStartStop:
    @pytest.mark.asyncio
    async def test_a_loop_that_dies_is_logged_and_stop_reports_it(self, index, fake_clock, caplog):
        refresher = _refresher(
            FakeGraph(), index, fake_clock, ManualSleeper(fake_clock), source=ExplodingSource()
        )

        with caplog.at_level(logging.ERROR, logger="backend.memory.fact_source"):
            refresher.start()
            await _settle()

        assert not refresher.running
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert errors and "KeyError" in errors[0].getMessage()
        with pytest.raises(KeyError):
            await refresher.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_a_waiting_loop_and_is_idempotent(self, index, fake_clock):
        sleeper = ManualSleeper(fake_clock)
        refresher = _refresher(FakeGraph(_facts(1)), index, fake_clock, sleeper, periodic=True)
        refresher.start()
        await _settle()
        assert sleeper.waiting == 1

        await refresher.stop()
        await refresher.stop()

        assert not refresher.running

    @pytest.mark.asyncio
    async def test_start_twice_raises(self, index, fake_clock):
        refresher = _refresher(FakeGraph(), index, fake_clock, ManualSleeper(fake_clock))
        refresher.start()
        try:
            with pytest.raises(RuntimeError, match="already"):
                refresher.start()
        finally:
            await refresher.stop()

    @pytest.mark.asyncio
    async def test_stop_mid_scan_applies_nothing(self, index, fake_clock):
        graph = FakeGraph(_facts(3))
        source = GatedSource(GraphFactSource(graph.executor, page_size=2))
        spy = SpyIndex(index)
        refresher = _refresher(graph, spy, fake_clock, ManualSleeper(fake_clock), source=source)
        refresher.start()
        await _settle()

        await refresher.stop()

        assert spy.applies == 0
        assert index.count() == 0

    @pytest.mark.asyncio
    async def test_stop_during_a_diff_waits_for_the_whole_diff(self, index, fake_clock):
        graph = FakeGraph(_facts(3))
        release = threading.Event()
        spy = SpyIndex(index, block=release)
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="memory-index") as pool:
            refresher = _refresher(graph, spy, fake_clock, ManualSleeper(fake_clock), executor=pool)
            refresher.start()
            assert await asyncio.to_thread(spy.entered.wait, _WAIT_S)

            stopping = asyncio.create_task(refresher.stop())
            await _settle()
            assert not stopping.done()  # the diff is still running in its thread

            release.set()
            await asyncio.wait_for(stopping, _WAIT_S)

        assert spy.finished == 1
        assert index.count() == 3


# -- scale ---------------------------------------------------------------------------------------


@pytest.fixture(params=["memory", "file"])
def scale_index(
    request: pytest.FixtureRequest, tmp_path, fake_embedder: FakeEmbedder
) -> Iterator[tuple[str, FactIndex]]:
    """A FactIndex in memory, or on a WAL file with its own reader as production opens it."""
    if request.param == "memory":
        writer = open_index(":memory:", embedding_model_id=_MODEL)
        yield "in-memory", FactIndex(writer, fake_embedder, tz=UTC, reader=writer)
        writer.close()
        return
    path = tmp_path / "memory_index.db"
    writer = open_index(path, embedding_model_id=_MODEL)
    reader = open_reader(path, embedding_model_id=_MODEL)
    assert writer.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    yield "file-backed WAL", FactIndex(writer, fake_embedder, tz=UTC, reader=reader)
    reader.close()
    writer.close()


class TestScale:
    def test_ten_thousand_facts_take_at_most_two_full_pages(self):
        assert DEFAULT_PAGE_SIZE >= 5000

    @pytest.mark.asyncio
    async def test_a_10k_fact_rescan_stays_under_five_seconds(
        self, scale_index, fake_clock, capsys
    ):
        backing, index = scale_index
        graph = FakeGraph(_facts(10_000))
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="memory-index") as pool:
            refresher = FactIndexRefresher(
                GraphFactSource(graph.executor),
                index,
                pool,
                clock=fake_clock,
                debounce_s=_DEBOUNCE_S,
                periodic=False,
            )

            started = time.perf_counter()
            first = await refresher.refresh()
            cold_s = time.perf_counter() - started
            cold_pages = graph.pages

            started = time.perf_counter()
            second = await refresher.refresh()
            warm_s = time.perf_counter() - started

        with capsys.disabled():
            print(
                f"\n10k-fact rescan with FakeEmbedder, {backing}: cold {cold_s:.2f} s "
                f"({first.embedded} embedded, {cold_pages} pages), unchanged {warm_s:.2f} s"
            )
        assert cold_pages <= 3  # two full pages and the empty one that ends the scan
        assert first == DiffReport(inserted=10_000, embedded=10_000)
        assert second == DiffReport(unchanged=10_000)
        assert cold_s < 5.0
        assert warm_s < 5.0
