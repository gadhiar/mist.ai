"""The dispatcher stores the service's stamps whole, so the run identity reaches the cache row.

`ExtractionDispatcher._dispatch` writes `response.stamps.model_dump()` into the
row's `service_stamps` through `put_extracted`. Contract 1.2.0 adds
`ResultStamps.run_identity`; these tests pin that every production turn
records it with no dispatcher change, and that a 1.1.0 service's result
(no run identity) still stores, as null.
"""

from __future__ import annotations

import pytest

from tests.unit.extraction_backlog.conftest import wait_until

_RUN_IDENTITY = "ri1-" + "b" * 64


async def _extract_one(world, ts) -> dict:
    event_id = world.log_turn(
        session_id="s1", turn_index=0, timestamp=ts(0), utterance="I really use rust"
    )
    dispatcher = world.build_dispatcher()
    await dispatcher.start()
    drained = await dispatcher.drain(timeout=5.0)
    if not drained:
        await wait_until(lambda: dispatcher.snapshot().backlog_depth == 0)
    epoch = world.store.active_epoch()
    row = world.cache.get(event_id, epoch.extraction_version, epoch.model_hash)
    assert row is not None, "the extracted turn left no cache row"
    return row


@pytest.mark.asyncio
async def test_the_run_identity_is_stored_unchanged(backlog_world, ts):
    backlog_world.service.run_identity = _RUN_IDENTITY

    row = await _extract_one(backlog_world, ts)

    assert row["service_stamps"]["stamps"]["run_identity"] == _RUN_IDENTITY


@pytest.mark.asyncio
async def test_the_other_stamps_are_stored_beside_it(backlog_world, ts):
    backlog_world.service.run_identity = _RUN_IDENTITY

    stamps = (await _extract_one(backlog_world, ts))["service_stamps"]["stamps"]

    assert stamps == {
        "extraction_version": backlog_world.service.extraction_version,
        "model_hash": backlog_world.service.model_hash,
        "prompt_sha256": "0" * 64,
        "llama_cpp_build": "b0",
        "adapter": "fake",
        "run_identity": _RUN_IDENTITY,
    }


@pytest.mark.asyncio
async def test_a_result_without_a_run_identity_stores_null(backlog_world, ts):
    assert backlog_world.service.run_identity is None  # a 1.1.0 service

    row = await _extract_one(backlog_world, ts)

    assert row["service_stamps"]["stamps"]["run_identity"] is None
    assert backlog_world.curation.event_ids, "the turn was not applied"
