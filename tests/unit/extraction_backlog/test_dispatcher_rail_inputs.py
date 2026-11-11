"""The dispatcher stores a reply's batch record and turn plan in the cache row.

A rail reads the cached row and the pack, and nothing else, so what the service
recorded about a turn's Stage 2 calls and its plan call has to reach the row the
replay reads. `ExtractionDispatcher._dispatch` writes both through
`put_extracted`, in the shape the contract model dumps, beside the payload and
before the apply.
"""

from __future__ import annotations

import pytest

from backend.extraction_contract.models import (
    BatchRecordEntry,
    ExtractionPayload,
    TurnPlanOut,
)
from backend.knowledge.extraction.normalizer import rail_inputs
from tests.unit.extraction_backlog.conftest import wait_until

PROMPT = "a" * 64
RAILS = "b" * 64

BATCH_RECORD = [
    BatchRecordEntry(
        units=[{"text": "I really use rust", "context": [], "kind": "paragraph"}],
        context=[],
        payload=ExtractionPayload(
            entities=[{"id": "rust", "type": "Technology", "name": "Rust"}], relationships=[]
        ),
        prompt_tokens=120,
        completion_tokens=30,
        finish="stop",
    ),
    BatchRecordEntry(
        units=[{"text": "and zig", "context": ["I really use rust"], "kind": "paragraph"}],
        context=["I really use rust"],
        payload=None,
    ),
]
TURN_PLAN = TurnPlanOut(
    payload={"templates": []}, prompt_tokens=80, completion_tokens=12, finish_reason="stop"
)


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
async def test_the_batch_record_is_stored_as_the_contract_dumps_it(backlog_world, ts):
    backlog_world.service.batch_record = BATCH_RECORD

    row = await _extract_one(backlog_world, ts)

    assert row["batch_record"] == [entry.model_dump(mode="json") for entry in BATCH_RECORD]
    assert row["batch_record"][0]["units"] == [
        {"text": "I really use rust", "context": [], "kind": "paragraph"}
    ]
    assert row["batch_record"][1]["payload"] is None


@pytest.mark.asyncio
async def test_the_turn_plan_is_stored_as_the_contract_dumps_it(backlog_world, ts):
    backlog_world.service.turn_plan = TURN_PLAN

    row = await _extract_one(backlog_world, ts)

    assert row["turn_plan"] == {
        "payload": {"templates": []},
        "prompt_tokens": 80,
        "completion_tokens": 12,
        "finish_reason": "stop",
    }


@pytest.mark.asyncio
async def test_both_reach_the_rails_on_replay_and_the_turn_is_applied(backlog_world, ts):
    """The stored row hands both to `normalize`, and the apply that reads it completes."""
    backlog_world.service.batch_record = BATCH_RECORD
    backlog_world.service.turn_plan = TURN_PLAN

    row = await _extract_one(backlog_world, ts)

    assert rail_inputs(row) == {
        "batch_record": [entry.model_dump(mode="json") for entry in BATCH_RECORD],
        "turn_plan": TURN_PLAN.model_dump(mode="json"),
    }
    assert backlog_world.curation.event_ids, "the turn was not applied"


@pytest.mark.asyncio
async def test_a_reply_without_them_stores_null_in_both_columns(backlog_world, ts):
    """A service that records neither: the row reads as one batch with no plan."""
    row = await _extract_one(backlog_world, ts)

    assert (row["batch_record"], row["turn_plan"]) == (None, None)
    assert rail_inputs(row) == {}
    assert backlog_world.curation.event_ids, "the turn was not applied"


@pytest.mark.asyncio
async def test_the_replys_pack_hashes_are_stored_in_the_rows_service_stamps(backlog_world, ts):
    """The stamps are stored whole, so what fed the model is on the row it produced."""
    backlog_world.service.pack_prompt_hash = PROMPT
    backlog_world.service.pack_rails_hash = RAILS

    stamps = (await _extract_one(backlog_world, ts))["service_stamps"]["stamps"]

    assert (stamps["pack_prompt_hash"], stamps["pack_rails_hash"]) == (PROMPT, RAILS)
