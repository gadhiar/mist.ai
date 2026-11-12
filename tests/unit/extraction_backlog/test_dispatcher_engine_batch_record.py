"""The batch record the engine writes reaches the cache row through the dispatcher.

`test_dispatcher_rail_inputs.py` stores a record written by hand. Here the record is
the one the extraction engine writes for a list turn, one entry per Stage 2 call, and
the fake service answers with it, so the row a replay reads holds what the engine
recorded about each call.
"""

from __future__ import annotations

import pytest

from tests.unit.extraction_backlog.conftest import wait_until
from tests.unit.extraction_service.test_list_pass import ScriptedLLM, make_engine, make_request

_LIST = (
    "Tools I picked up this year:\n"
    "- Velk, for the backend\n"
    "- Ornate\n"
    "- Quillet\n"
    "- Barrow\n"
    "- Tamsin, the same way as the first one"
)


@pytest.mark.asyncio
async def test_the_engines_batch_record_is_the_rows(backlog_world, ts):
    extracted = await make_engine(ScriptedLLM()).run(make_request(_LIST))
    backlog_world.service.batch_record = extracted.batch_record
    event_id = backlog_world.log_turn(
        session_id="s1", turn_index=0, timestamp=ts(0), utterance=_LIST
    )
    dispatcher = backlog_world.build_dispatcher()

    await dispatcher.start()
    if not await dispatcher.drain(timeout=5.0):
        await wait_until(lambda: dispatcher.snapshot().backlog_depth == 0)

    epoch = backlog_world.store.active_epoch()
    row = backlog_world.cache.get(event_id, epoch.extraction_version, epoch.model_hash)
    assert row is not None, "the extracted turn left no cache row"
    assert row["batch_record"] == [
        entry.model_dump(mode="json") for entry in extracted.batch_record
    ]
    assert [len(entry["units"]) for entry in row["batch_record"]] == [4, 1]
    assert row["batch_record"][1]["context"][0] == (
        "Tools I picked up this year: Velk, for the backend"
    )
