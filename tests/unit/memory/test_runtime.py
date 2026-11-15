"""Tests for backend.memory.runtime: the lifecycle handle, the executors and the index path."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.errors import MemoryIndexError
from backend.knowledge.config import EventStoreConfig
from backend.memory.config import MemoryConfig
from backend.memory.fetcher import GATE_EXECUTOR, INDEX_EXECUTOR, MODEL_EXECUTOR
from backend.memory.index_db import STATE_REBUILDING, open_index, set_meta, write_transaction
from backend.memory.runtime import (
    GATE_THREADS,
    INDEX_THREADS,
    MODEL_THREADS,
    IndexStateProbe,
    MemoryRuntime,
    build_memory_executors,
    memory_index_path,
)
from scripts.mist_admin import _memory_index_path

_MODEL = "all-MiniLM-L6-v2"


class FakeRefresher:
    """Records the lifecycle calls MemoryRuntime makes on the FactIndexRefresher."""

    def __init__(self) -> None:
        self.starts = 0
        self.stops = 0
        self.dirty = 0

    @property
    def running(self) -> bool:
        return self.starts > self.stops

    def start(self) -> None:
        if self.running:
            raise RuntimeError("already running")
        self.starts += 1

    async def stop(self) -> None:
        self.stops += 1

    def mark_dirty(self) -> None:
        self.dirty += 1


@pytest.fixture
def refresher() -> FakeRefresher:
    return FakeRefresher()


@pytest.fixture
def runtime(refresher: FakeRefresher) -> Iterator[MemoryRuntime]:
    executors = build_memory_executors()
    yield MemoryRuntime(refresher, executors)  # type: ignore[arg-type]
    for executor in executors.values():
        executor.shutdown(wait=True, cancel_futures=True)


def _ran_on(runtime: MemoryRuntime, name: str) -> str:
    executor = runtime.executors_by_name[name]
    return executor.submit(lambda: threading.current_thread().name).result(timeout=5)


@pytest.mark.asyncio
async def test_start_starts_the_refresher_once(
    runtime: MemoryRuntime, refresher: FakeRefresher
) -> None:
    await runtime.start()
    await runtime.start()

    assert refresher.starts == 1


@pytest.mark.asyncio
async def test_aclose_stops_the_refresher_and_shuts_the_executors_down(
    runtime: MemoryRuntime, refresher: FakeRefresher
) -> None:
    await runtime.start()
    assert _ran_on(runtime, INDEX_EXECUTOR).startswith("memory-index")

    await runtime.aclose()

    assert refresher.stops == 1
    for executor in runtime.executors_by_name.values():
        with pytest.raises(RuntimeError):
            executor.submit(lambda: None)


@pytest.mark.asyncio
async def test_aclose_is_safe_twice(runtime: MemoryRuntime, refresher: FakeRefresher) -> None:
    await runtime.start()

    await runtime.aclose()
    await runtime.aclose()

    assert refresher.stops == 1


@pytest.mark.asyncio
async def test_aclose_before_start_shuts_down_without_stopping(
    runtime: MemoryRuntime, refresher: FakeRefresher
) -> None:
    await runtime.aclose()

    assert refresher.stops == 0
    with pytest.raises(RuntimeError):
        runtime.executors_by_name[MODEL_EXECUTOR].submit(lambda: None)


@pytest.mark.asyncio
async def test_start_after_aclose_is_refused(runtime: MemoryRuntime) -> None:
    await runtime.aclose()

    with pytest.raises(RuntimeError):
        await runtime.start()


def test_mark_dirty_is_synchronous_and_only_marks_the_refresher(
    runtime: MemoryRuntime, refresher: FakeRefresher
) -> None:
    assert runtime.mark_dirty() is None

    assert refresher.dirty == 1 and refresher.starts == 0


def test_the_executors_are_named_and_sized(runtime: MemoryRuntime) -> None:
    executors = runtime.executors_by_name

    assert set(executors) == {INDEX_EXECUTOR, MODEL_EXECUTOR, GATE_EXECUTOR}
    assert executors[INDEX_EXECUTOR]._max_workers == INDEX_THREADS == 2
    assert executors[MODEL_EXECUTOR]._max_workers == MODEL_THREADS == 1
    assert executors[GATE_EXECUTOR]._max_workers == GATE_THREADS == 1
    assert _ran_on(runtime, MODEL_EXECUTOR).startswith("memory-model")
    assert _ran_on(runtime, GATE_EXECUTOR).startswith("memory-gate")
    assert runtime.executors.index is executors[INDEX_EXECUTOR]
    assert runtime.executors.model is executors[MODEL_EXECUTOR]
    assert runtime.executors.gate is executors[GATE_EXECUTOR]


def test_the_gate_executor_is_its_own_thread(runtime: MemoryRuntime) -> None:
    """The gate's embeddings never queue behind the cross-encoder on the model thread."""
    hold = threading.Event()
    executors = runtime.executors
    held = executors.model.submit(hold.wait, 5)
    try:
        name = executors.gate.submit(lambda: threading.current_thread().name).result(timeout=5)
    finally:
        hold.set()
        held.result(timeout=5)

    assert name.startswith("memory-gate")


@pytest.mark.parametrize("missing", [MODEL_EXECUTOR, GATE_EXECUTOR])
def test_a_runtime_needs_every_executor(refresher: FakeRefresher, missing: str) -> None:
    executors = build_memory_executors()
    try:
        with pytest.raises(ValueError, match=missing):
            MemoryRuntime(
                refresher,  # type: ignore[arg-type]
                {name: pool for name, pool in executors.items() if name != missing},
            )
    finally:
        for executor in executors.values():
            executor.shutdown(wait=True)


# -- the index path ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("index_path", "db_path"),
    [
        ("/data/custom/memory.db", "/data/mist/event_store.db"),  # MIST_MEMORY_INDEX_PATH set
        (None, "/data/mist/event_store.db"),  # beside the event store
        (None, ":memory:"),  # an in-memory event store keeps the index in memory
        (None, None),  # the event store's home default
    ],
    ids=["explicit", "beside-event-store", "in-memory", "home-default"],
)
def test_the_index_path_resolves_as_the_admin_command_does(
    index_path: str | None, db_path: str | None
) -> None:
    memory = MemoryConfig(index_path=index_path)
    event_store = EventStoreConfig(db_path=db_path)

    resolved = memory_index_path(memory, event_store)

    assert resolved == _memory_index_path(
        SimpleNamespace(memory=memory, event_store=event_store), None
    )
    if index_path is not None:
        assert resolved == index_path
    elif db_path == ":memory:":
        assert resolved == ":memory:"
    else:
        assert Path(resolved).name == "memory_index.db"


# -- the index state probe ---------------------------------------------------------------------


def test_the_probe_reports_a_rebuild_in_progress(tmp_path: Path) -> None:
    path = tmp_path / "memory_index.db"
    writer = open_index(path, embedding_model_id=_MODEL)
    probe = IndexStateProbe.open(str(path), embedding_model_id=_MODEL)
    try:
        assert probe() is None
        with write_transaction(writer):
            set_meta(writer, "index_state", STATE_REBUILDING)
        assert probe() == "index_rebuilding"
    finally:
        probe.close()
        writer.close()


def test_the_probe_of_an_in_memory_index_never_degrades() -> None:
    probe = IndexStateProbe.open(":memory:", embedding_model_id=_MODEL)

    assert probe() is None
    probe.close()


def test_a_closed_probe_raises_the_index_error() -> None:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    probe = IndexStateProbe(conn)
    probe.close()

    with pytest.raises(MemoryIndexError):
        probe()
