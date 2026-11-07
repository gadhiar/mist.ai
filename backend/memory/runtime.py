"""The memory read path's lifecycle: the refresher, the named executors and the index location.

`MemoryRuntime` owns what outlives one fetch: the FactIndexRefresher's task and the two thread
pools, `memory-index` (SQLite reads and the refresher's writes, two threads) and
`memory-model` (the query embedding and the rerank, one thread, so a CPU cross-encoder never
runs twice at once). The handler starts it at startup and closes it at shutdown; the listeners
of T33 and T34 call `mark_dirty` through their own async adapters.

`memory_index_path` resolves the index file the way `mist_admin memory-index-rebuild` does, so
the serving process and the admin command always open the same file.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

from backend.errors import MemoryIndexError

from .config import MemoryConfig
from .fetcher import INDEX_EXECUTOR, MODEL_EXECUTOR, FetcherExecutors
from .index_db import MEMORY_PATH, degraded_reason, open_reader
from .models import DegradedReason

logger = logging.getLogger(__name__)

INDEX_THREADS = 2
MODEL_THREADS = 1

# The file name beside the event store; mist_admin's `_MEMORY_INDEX_FILENAME` is the same.
MEMORY_INDEX_FILENAME = "memory_index.db"


class EventStoreLocation(Protocol):
    """The part of `EventStoreConfig` the index path depends on."""

    db_path: str | None


class Refresher(Protocol):
    """The lifecycle of `FactIndexRefresher` the runtime drives."""

    def start(self) -> None:
        """Start the refresher's task; RuntimeError if it is already running."""
        ...

    async def stop(self) -> None:
        """Stop the task and wait for it."""
        ...

    def mark_dirty(self) -> None:
        """Schedule a debounced refresh."""
        ...


def build_memory_executors() -> dict[str, ThreadPoolExecutor]:
    """The two named executors the fetcher and the refresher share; the runtime closes them."""
    return {
        INDEX_EXECUTOR: ThreadPoolExecutor(INDEX_THREADS, thread_name_prefix=INDEX_EXECUTOR),
        MODEL_EXECUTOR: ThreadPoolExecutor(MODEL_THREADS, thread_name_prefix=MODEL_EXECUTOR),
    }


def memory_index_path(memory: MemoryConfig, event_store: EventStoreLocation) -> str:
    """The index file: `MIST_MEMORY_INDEX_PATH`, else `memory_index.db` beside the event store.

    The event store's default is the one `factories.py` uses (`~/.mist/event_store.db`). An
    in-memory event store gives an in-memory index. The order is mist_admin's
    `_memory_index_path` without its `--index-path` override; a test pins the two together.
    """
    if memory.index_path:
        return memory.index_path
    store = event_store.db_path or str(Path.home() / ".mist" / "event_store.db")
    if store == MEMORY_PATH:
        return MEMORY_PATH
    return str(Path(store).parent / MEMORY_INDEX_FILENAME)


class IndexStateProbe:
    """`index_db.degraded_reason` on a read connection of its own, for the fetcher's check.

    The fetcher checks the index state on every fetch, from the `memory-index` executor, whose
    two threads may run checks at once; the lock keeps them off the one connection together.
    The FactIndex reader is not shared for this, because its lock is FactIndex's own. An
    in-memory index has no second connection to open and cannot be rebuilt under a running
    process, so its probe holds none and always reports None.

    Args:
        conn: A connection to the index file, or None for an in-memory index.
    """

    def __init__(self, conn: sqlite3.Connection | None) -> None:
        self._conn = conn
        self._lock = threading.Lock()

    @classmethod
    def open(cls, path: str, *, embedding_model_id: str) -> IndexStateProbe:
        """A probe on a new reader of `path`; one with no connection for `:memory:`.

        Raises:
            MemoryIndexError: The file cannot be opened as the index.
        """
        if path == MEMORY_PATH:
            return cls(None)
        return cls(open_reader(path, embedding_model_id=embedding_model_id))

    def __call__(self) -> DegradedReason | None:
        """The index's degraded reason now.

        Raises:
            MemoryIndexError: The database fails, or the probe is closed.
        """
        if self._conn is None:
            return None
        with self._lock:
            return degraded_reason(self._conn)

    def close(self) -> None:
        """Close the connection; a later call raises MemoryIndexError."""
        if self._conn is None:
            return
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error as exc:
                raise MemoryIndexError(f"cannot close the index state probe: {exc}") from exc


class MemoryRuntime:
    """Starts and stops the memory read path's background work.

    Args:
        refresher: The FactIndexRefresher, built on the `memory-index` executor.
        executors: The named executors from `build_memory_executors`; the runtime shuts them
            down on `aclose`.

    Raises:
        ValueError: An executor is missing.
    """

    def __init__(self, refresher: Refresher, executors: Mapping[str, ThreadPoolExecutor]) -> None:
        missing = [name for name in (INDEX_EXECUTOR, MODEL_EXECUTOR) if name not in executors]
        if missing:
            raise ValueError(f"MemoryRuntime needs the executors {missing}")
        self._refresher = refresher
        self._executors = dict(executors)
        self._started = False
        self._closed = False

    @property
    def executors(self) -> FetcherExecutors:
        """The executors as the fetcher takes them."""
        return FetcherExecutors(
            index=self._executors[INDEX_EXECUTOR], model=self._executors[MODEL_EXECUTOR]
        )

    @property
    def executors_by_name(self) -> Mapping[str, ThreadPoolExecutor]:
        """The executors by name, for wiring the refresher and for tests."""
        return dict(self._executors)

    async def start(self) -> None:
        """Start the refresher's task, once; a second call does nothing.

        Raises:
            RuntimeError: The runtime was closed.
        """
        if self._closed:
            raise RuntimeError("MemoryRuntime is closed")
        if self._started:
            return
        self._refresher.start()
        self._started = True

    async def aclose(self) -> None:
        """Stop the refresher, then shut the executors down; safe twice and before `start`.

        The executors are shut down off the event loop, cancelling queued work and waiting for
        the calls already running, so no thread outlives the runtime.
        """
        if self._closed:
            return
        self._closed = True
        try:
            if self._started:
                await self._refresher.stop()
        finally:
            for executor in self._executors.values():
                await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=True)

    def mark_dirty(self) -> None:
        """Tell the refresher the graph changed; it refreshes after its debounce."""
        self._refresher.mark_dirty()


__all__ = [
    "INDEX_THREADS",
    "MEMORY_INDEX_FILENAME",
    "MODEL_THREADS",
    "IndexStateProbe",
    "MemoryRuntime",
    "Refresher",
    "build_memory_executors",
    "memory_index_path",
]
