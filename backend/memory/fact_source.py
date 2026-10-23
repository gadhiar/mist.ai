"""Feed the memory index from the graph: a paged fact source and a debounced refresher.

`GraphFactSource` reads every latest-belief version of the user-facing facts with
`current_facts_query`, one page per `GraphExecutor.execute_query` await, so nothing synchronous
runs on the event loop. `FactIndexRefresher` turns each full rescan into one
`FactIndex.diff_apply` on the injected `memory-index` executor, which re-embeds only rows
whose content changed.

A full rescan, not a watermark delta, because it needs no new graph property and cannot
drift: a row the graph dropped is dropped from the index by the same diff. The price is that
a rescan must be complete before it is applied. A scan that fails part-way applies nothing,
because a partial snapshot handed to `diff_apply` would delete every row it missed.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import Executor
from typing import Protocol

from neo4j.exceptions import DriverError, Neo4jError

from backend.errors import (
    MemoryConfigError,
    MemoryIndexError,
    MistError,
    Neo4jConnectionError,
    Neo4jQueryError,
)
from backend.knowledge.eval_isolation import EvalIsolationError
from backend.knowledge.storage.graph_store import current_facts_query

from .fact_index import DiffReport, fact_row_key
from .models import FactRow, fact_key
from .ranking import MISSING_FACT_CONFIDENCE

logger = logging.getLogger(__name__)

# Rows per page. Paging by `elementId(r)` is linear only within one page: no index orders
# relationships by element id, so every page query expands and filters every matching edge
# before it skips to `after_key`, and a full rescan costs (edges x pages). A large page keeps
# the page count, and so the rescan, near one pass: 10k facts take two full pages and the
# empty one that ends the scan.
DEFAULT_PAGE_SIZE = 5000

# Least wait before retrying a failed refresh, so a debounce of zero cannot retry an
# unreachable graph in a busy loop.
_RETRY_FLOOR_S = 1.0


class QueryExecutor(Protocol):
    """The read half of `GraphExecutor`: run one query string off the event loop."""

    async def execute_query(
        self, query: str, params: dict[str, object] | None = None
    ) -> list[dict]:
        """Run `query` with `params` and return its records as dicts."""
        ...


class FactSnapshotSource(Protocol):
    """Produces a complete snapshot of the facts the index should hold, or raises."""

    async def snapshot(self) -> list[FactRow]:
        """Every fact version the index should hold, each `row_key` once; raises MistError."""
        ...


class DiffTarget(Protocol):
    """What the refresher writes to: `FactIndex.diff_apply`."""

    def diff_apply(self, snapshot: Sequence[FactRow]) -> DiffReport:
        """Make the index equal to `snapshot` in one write; blocking."""
        ...


class MonotonicClock(Protocol):
    """A monotonic clock in seconds. The `time` module satisfies it; tests pass FakeClock."""

    def monotonic(self) -> float:
        """Seconds from an arbitrary origin that never goes backwards."""
        ...


class GraphFactSource:
    """Every latest-belief version of the user-facing facts, read page by page.

    Args:
        executor: A `GraphExecutor` (or anything with its async `execute_query`).
        page_size: Rows per page.

    Raises:
        ValueError: `page_size` is not positive.
    """

    def __init__(self, executor: QueryExecutor, *, page_size: int = DEFAULT_PAGE_SIZE):
        if page_size <= 0:
            raise ValueError(f"page_size must be positive, got {page_size}")
        self._executor = executor
        self._page_size = page_size

    async def snapshot(self) -> list[FactRow]:
        """Read every page and return one FactRow per distinct `row_key`.

        Each row's `row_key` is `fact_row_key(fact_key, version_key)`: a raw `version_key` is
        unique only within one triple. Two edges can still share a row key when an undirected
        fact is stored in both orientations with the same version; one of them is kept, chosen
        by content so a rebuild keeps the same one, and each such fact is logged by key.

        Returns:
            The complete snapshot, ordered by page key.

        Raises:
            Neo4jConnectionError: The graph cannot be reached (a driver error, a refused
                connect, or a socket error).
            Neo4jQueryError: A query fails.
            MemoryIndexError: The page key does not advance, so paging would never end.
        """
        rows: dict[str, FactRow] = {}
        duplicates = 0
        after: str | None = None
        while True:
            page = await self._page(after)
            for record in page:
                row = _fact_row(record)
                kept = rows.get(row.row_key)
                if kept is not None:
                    duplicates += 1
                    row = min(kept, row, key=_content_order)
                    logger.warning(
                        "fact scan found two edges with row key %s of fact %s (%s %s %s); "
                        "kept the one ordered first by content",
                        row.row_key,
                        row.fact_key,
                        row.subject_id,
                        row.rel_type,
                        row.object_id,
                    )
                rows[row.row_key] = row
            if len(page) < self._page_size:
                break
            last = page[-1].get("page_key")
            if not isinstance(last, str) or (after is not None and last <= after):
                raise MemoryIndexError(
                    f"fact scan page key did not advance past {after!r} (got {last!r})"
                )
            after = last
        if duplicates:
            logger.warning("fact scan dropped %d edges that shared a row key", duplicates)
        return list(rows.values())

    async def _page(self, after: str | None) -> list[dict]:
        try:
            return await self._executor.execute_query(*current_facts_query(after, self._page_size))
        except Neo4jError as exc:
            raise Neo4jQueryError(f"fact scan query failed: {exc}") from exc
        except (DriverError, EvalIsolationError, OSError) as exc:
            raise Neo4jConnectionError(f"fact scan could not reach the graph: {exc}") from exc


def _fact_row(record: dict[str, object]) -> FactRow:
    subject_id = str(record["subject_id"])
    rel_type = str(record["rel_type"])
    object_id = str(record["object_id"])
    key = fact_key(subject_id, rel_type, object_id)
    version_key = record.get("version_key")
    return FactRow(
        row_key=fact_row_key(key, None if version_key is None else str(version_key)),
        fact_key=key,
        subject_id=subject_id,
        subject_name=str(record.get("subject_name") or subject_id),
        rel_type=rel_type,
        object_id=object_id,
        object_name=str(record.get("object_name") or object_id),
        valid_from=_text(record.get("valid_from")),
        valid_to=_text(record.get("valid_to")),
        recorded_at=_text(record.get("recorded_at")),
        confidence=_confidence(record.get("confidence")),
        source_utterance_id=_text(record.get("source_utterance_id")),
    )


def _text(value: object) -> str | None:
    """A stored property as the string the index keeps.

    Bounds are ISO strings on every current writer; a legacy native `datetime()` value is
    stringified rather than dropped, so the version is still indexed.
    """
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)


def _confidence(value: object) -> float:
    """The edge confidence, or the ranking's missing-confidence value when it is absent."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return MISSING_FACT_CONFIDENCE


def _content_order(row: FactRow) -> tuple[str | float, ...]:
    """A total order on a row's content, to pick one of two rows sharing a row key.

    The lower id first puts an undirected fact's canonical orientation first, the one
    reconciliation stores.
    """
    return (
        row.subject_id,
        row.object_id,
        row.valid_from or "",
        row.valid_to or "",
        row.recorded_at or "",
        row.source_utterance_id or "",
        row.confidence,
        row.subject_name,
        row.object_name,
    )


class FactIndexRefresher:
    """Keeps the fact index equal to the graph's latest-belief versions.

    It refreshes once when started, then whenever marked dirty, at most once per
    `debounce_s` measured start to start; in `periodic` mode, for a process with no extraction
    dispatcher to mark it, it refreshes every `debounce_s` instead. A refresh is a full rescan
    through `source` followed by one `diff_apply` on `executor`.

    Callers:
        - MemoryRuntime (T9) calls `start()` from its `start()` and awaits `stop()` from
          `aclose()`, before it shuts the executors down. `run()` is the same loop for a caller
          that owns the task itself; cancel the task to stop it.
        - `mist_admin memory-index-rebuild --facts` (T13) awaits `refresh()`, which returns the
          DiffReport and raises on failure.
        - The dispatcher's apply listener (T33) and the applier's outcome listener (T34) are
          async adapters that call `mark_dirty()`.

    A failed refresh inside the loop (graph unreachable, a query, index or embedder error) is
    logged, the last index is kept, and the refresher stays dirty, so it retries after
    `debounce_s`, or after a one-second floor when `debounce_s` is zero. Only MistError is
    contained: every source error arrives as one (GraphFactSource wraps the driver's), and
    FactIndex raises MemoryIndexError, including for the embedder's RuntimeError, OSError and
    ValueError. Anything else ends the loop: the task's done-callback logs it at error level
    with its type, `running` turns False, and `stop()` re-raises it.

    Stopping never leaves a half-applied diff: `diff_apply` is one write transaction, and a
    stop that lands while it runs waits for it to finish before the loop ends. A stop during
    the scan applies nothing.

    Args:
        source: Produces the complete snapshot, normally a GraphFactSource.
        index: The FactIndex to diff-apply into.
        executor: The `memory-index` executor, owned by the caller; `diff_apply` blocks on
            SQLite and the embedder, so it never runs on the event loop.
        clock: Monotonic seconds for the debounce.
        debounce_s: Least time between the starts of two refreshes, and the timer interval in
            `periodic` mode (`MemoryConfig.refresh_debounce_s`).
        periodic: Refresh on the timer whether or not marked; for a process with no
            dispatcher attached.
        sleep: Waits that many seconds; tests inject one driven by their fake clock.

    Raises:
        MemoryConfigError: `debounce_s` is negative, or zero in `periodic` mode, where it
            would make the timer a busy loop; startup fails naming the variable.
    """

    def __init__(
        self,
        source: FactSnapshotSource,
        index: DiffTarget,
        executor: Executor,
        *,
        clock: MonotonicClock,
        debounce_s: float,
        periodic: bool,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        if debounce_s < 0 or (periodic and debounce_s == 0):
            needed = "positive when no dispatcher is attached" if periodic else "not negative"
            raise MemoryConfigError(
                f"MIST_MEMORY_REFRESH_DEBOUNCE_S must be {needed}, got {debounce_s}"
            )
        self._source = source
        self._index = index
        self._executor = executor
        self._clock = clock
        self._debounce_s = debounce_s
        self._periodic = periodic
        self._sleep = sleep
        self._dirty = asyncio.Event()
        self._refresh_lock = asyncio.Lock()
        self._next_allowed = float("-inf")
        self._task: asyncio.Task[None] | None = None

    @property
    def running(self) -> bool:
        """Whether the loop task started by `start()` is still running."""
        return self._task is not None and not self._task.done()

    def mark_dirty(self) -> None:
        """Note that the graph changed, so the loop refreshes within `debounce_s`.

        Synchronous, takes no argument and returns at once: it only sets a flag, so the
        dispatcher's inline await of its apply listener is never held by a rescan. Call it on
        the event loop's thread, as the async listener adapters do.
        """
        self._dirty.set()

    async def refresh(self) -> DiffReport:
        """Rescan the graph and diff-apply the snapshot now, ignoring the debounce.

        Refreshes are serialised, so this and the loop never interleave. The dirty mark is
        cleared when the scan starts; a mark that lands during the scan stays set and brings
        another refresh.

        Returns:
            What the diff changed.

        Raises:
            MistError: The scan or the write failed; the index is as it was before the call.
            Any other exception the source or the embedder raises propagates unchanged.
        """
        async with self._refresh_lock:
            self._dirty.clear()
            self._next_allowed = self._clock.monotonic() + self._debounce_s
            snapshot = await self._source.snapshot()
            return await self._apply(snapshot)

    async def run(self) -> None:
        """Refresh at once, then on each mark or timer tick, until cancelled."""
        await self._refresh_logged()
        while True:
            if not self._periodic:
                await self._dirty.wait()
            delay = self._next_allowed - self._clock.monotonic()
            if delay > 0:
                await self._sleep(delay)
            await self._refresh_logged()

    def start(self) -> None:
        """Start `run()` as a task on the running loop.

        Raises:
            RuntimeError: The loop is already running, or no event loop is running.
        """
        if self.running:
            raise RuntimeError("the fact index refresher is already running")
        task = asyncio.get_running_loop().create_task(self.run(), name="memory-fact-refresher")
        task.add_done_callback(_log_unexpected_end)
        self._task = task

    async def stop(self) -> None:
        """Cancel the loop and wait for it to end; a no-op when it is not running.

        Raises:
            Exception: Whatever ended the loop early, other than cancellation.
        """
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        await asyncio.wait([task])
        if not task.cancelled() and task.exception() is not None:
            raise task.exception()

    async def _refresh_logged(self) -> None:
        try:
            report = await self.refresh()
        except MistError as exc:
            self._dirty.set()
            retry_s = max(self._debounce_s, _RETRY_FLOOR_S)
            self._next_allowed = max(self._next_allowed, self._clock.monotonic() + retry_s)
            logger.warning(
                "fact index refresh failed, keeping the last index; retrying in %.0f s: %s: %s",
                retry_s,
                type(exc).__name__,
                exc,
            )
            return
        logger.info(
            "fact index refreshed: %d inserted, %d updated, %d deleted, %d unchanged, "
            "%d embedded",
            report.inserted,
            report.updated,
            report.deleted,
            report.unchanged,
            report.embedded,
        )

    async def _apply(self, snapshot: list[FactRow]) -> DiffReport:
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self._executor, self._index.diff_apply, snapshot)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            # The write keeps running in its thread whatever happens here; wait for it so a
            # stop returns only once the index is quiescent and the executor can shut down.
            await asyncio.wait([future])
            if not future.cancelled() and future.exception() is not None:
                logger.warning(
                    "fact index write failed while the refresher stopped: %s", future.exception()
                )
            raise


def _log_unexpected_end(task: asyncio.Task[None]) -> None:
    """Make a refresher loop that ends on its own loud: it should run until cancelled.

    Retrieving the exception here also keeps asyncio from reporting it as never retrieved;
    the task stays referenced, so `stop()` still re-raises it.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is None:
        logger.error("fact index refresher loop returned; the index is no longer refreshed")
        return
    logger.error(
        "fact index refresher loop died with %s; the index is no longer refreshed: %s",
        type(exc).__name__,
        exc,
        exc_info=exc,
    )
