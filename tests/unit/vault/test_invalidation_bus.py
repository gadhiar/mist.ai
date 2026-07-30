"""InvalidationBus pub/sub transport tests.

R1.3: the bus event is owned by the vault layer (VaultChangeEvent) rather than
imported from the retired curation GraphRegenerator. The bus carries read-path
cache invalidation only -- it never signalled a graph write, and after R1.3 no
graph write exists for it to signal.
"""

import asyncio
import dataclasses
from pathlib import Path

import pytest

from backend.vault.invalidation_bus import InvalidationBus, VaultChangeEvent


def make_event(path_str: str) -> VaultChangeEvent:
    return VaultChangeEvent(path=Path(path_str))


def test_vault_change_event_is_frozen() -> None:
    event = make_event("/vault/users/raj.md")
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.path = Path("/vault/other.md")  # type: ignore[misc]


def test_vault_change_event_carries_path() -> None:
    event = make_event("/vault/users/raj.md")
    assert event.path == Path("/vault/users/raj.md")


def test_bus_does_not_import_from_curation() -> None:
    """Guards the layering fix: vault must not depend on knowledge.curation."""
    import inspect

    import backend.vault.invalidation_bus as bus_module

    source = inspect.getsource(bus_module)
    assert "knowledge.curation" not in source
    assert "RebuildResult" not in source


def test_publish_calls_all_listeners() -> None:
    bus = InvalidationBus()
    seen: list[Path] = []

    async def listener(event: VaultChangeEvent) -> None:
        seen.append(event.path)

    bus.subscribe(listener)
    bus.subscribe(listener)
    asyncio.run(bus.publish(make_event("/vault/users/raj.md")))
    assert seen == [Path("/vault/users/raj.md")] * 2


def test_publish_isolates_listener_exceptions() -> None:
    bus = InvalidationBus()
    seen: list[str] = []

    async def bad(event: VaultChangeEvent) -> None:
        raise RuntimeError("listener exploded")

    async def good(event: VaultChangeEvent) -> None:
        seen.append("good")

    bus.subscribe(bad)
    bus.subscribe(good)
    asyncio.run(bus.publish(make_event("/vault/users/raj.md")))
    assert seen == ["good"], "a failing listener must not block the next one"


def test_publish_with_no_listeners_is_noop() -> None:
    bus = InvalidationBus()
    asyncio.run(bus.publish(make_event("/vault/users/raj.md")))


def test_subscribe_multiple_listeners() -> None:
    bus = InvalidationBus()
    calls: list[str] = []

    async def first(event: VaultChangeEvent) -> None:
        calls.append("first")

    async def second(event: VaultChangeEvent) -> None:
        calls.append("second")

    bus.subscribe(first)
    bus.subscribe(second)
    asyncio.run(bus.publish(make_event("/vault/sessions/x.md")))
    assert calls == ["first", "second"], "listeners fire in registration order"
