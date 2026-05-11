"""Unit tests for InvalidationBus."""

import asyncio
from pathlib import Path

from backend.knowledge.curation.graph_regenerator import RebuildResult
from backend.vault.invalidation_bus import InvalidationBus


def make_event(path_str: str) -> RebuildResult:
    return RebuildResult(
        path=Path(path_str),
        bucket="1",
        orphaned_triple_count=0,
        new_triple_count=0,
        ontology_version="1.1.0",
        deferred=False,
    )


def test_publish_calls_all_listeners():
    bus = InvalidationBus()
    seen: list[RebuildResult] = []

    async def listener(event):
        seen.append(event)

    bus.subscribe(listener)
    asyncio.run(bus.publish(make_event("/x.md")))
    assert len(seen) == 1


def test_publish_isolates_listener_exceptions():
    bus = InvalidationBus()
    seen: list[str] = []

    async def bad_listener(event):
        raise RuntimeError("boom")

    async def good_listener(event):
        seen.append("ok")

    bus.subscribe(bad_listener)
    bus.subscribe(good_listener)
    # Bad listener does not prevent good listener from being called
    asyncio.run(bus.publish(make_event("/x.md")))
    assert seen == ["ok"]


def test_publish_with_no_listeners_is_noop():
    bus = InvalidationBus()
    asyncio.run(bus.publish(make_event("/x.md")))


def test_subscribe_multiple_listeners():
    bus = InvalidationBus()
    counts = [0, 0, 0]

    async def make_listener(idx):
        async def fn(event):
            counts[idx] += 1

        return fn

    bus.subscribe(asyncio.run(make_listener(0)))
    bus.subscribe(asyncio.run(make_listener(1)))
    bus.subscribe(asyncio.run(make_listener(2)))
    asyncio.run(bus.publish(make_event("/x.md")))
    assert counts == [1, 1, 1]
