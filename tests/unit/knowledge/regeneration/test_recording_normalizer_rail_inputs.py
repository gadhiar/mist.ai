"""The regeneration suite's recording normalizer takes what a replay passes the real one.

`LogRegenerator` calls `normalize(extraction, **rail_inputs(cached))`. Once a
cache row holds a batch record or a turn plan, that call carries keywords, and a
recorder that takes the extraction only would raise `TypeError` in every test
that observes stage order over such a row.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.knowledge.extraction.normalizer import rail_inputs
from tests.unit.knowledge.regeneration.conftest import _RecordingNormalizer

_ROW = {
    "entities": [],
    "relationships": [],
    "batch_record": [{"units": [], "context": [], "payload": None}],
    "turn_plan": {"payload": {"templates": []}},
}


@pytest.mark.asyncio
async def test_the_recorder_takes_the_rail_inputs_of_a_row_and_records_the_call():
    sink: list[str] = []
    extraction = SimpleNamespace(entities=[{"id": "ada"}], relationships=[])

    result = await _RecordingNormalizer(sink).normalize(extraction, **rail_inputs(_ROW))

    assert result is extraction
    assert sink == ["normalize"]


@pytest.mark.asyncio
async def test_the_recorder_still_takes_the_extraction_alone():
    sink: list[str] = []
    extraction = SimpleNamespace(entities=[], relationships=[])

    assert await _RecordingNormalizer(sink).normalize(extraction) is extraction
    assert sink == ["normalize"]
