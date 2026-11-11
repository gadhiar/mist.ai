"""The cutover suite's normalizer double takes what a replay passes the real one.

Both replay paths call `normalize(extraction, **rail_inputs(cached))`. Once a
cache row holds a batch record or a turn plan, that call carries keywords, and a
double that takes the extraction only would raise `TypeError` inside the rebuild
it stands in for.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.knowledge.extraction.normalizer import rail_inputs
from tests.unit.extraction_backlog.test_identity_cutover import _PassThroughNormalizer

_ROW = {
    "entities": [],
    "relationships": [],
    "batch_record": [{"units": [], "context": [], "payload": None}],
    "turn_plan": {"payload": {"templates": []}},
}


@pytest.mark.asyncio
async def test_the_pass_through_normalizer_takes_the_rail_inputs_of_a_row():
    extraction = SimpleNamespace(entities=[{"id": "ada"}], relationships=[])

    result = await _PassThroughNormalizer().normalize(extraction, **rail_inputs(_ROW))

    assert result is extraction


@pytest.mark.asyncio
async def test_the_pass_through_normalizer_still_takes_the_extraction_alone():
    extraction = SimpleNamespace(entities=[], relationships=[])

    assert await _PassThroughNormalizer().normalize(extraction) is extraction
