"""A turn whose every unit the pack's item filters drop is answered without a model call.

The item filters run on the split units before anything is sent, so a turn whose
every unit they drop has nothing left to extract. The service answers it at once:
outcome `extracted`, an empty payload, an empty batch record, and 0 for both
`attempts` and `stage2_calls`, which the contract admits only together. With the
turn plan flag on there is no plan call either, and no scope call, since the scope
label is read by Stage 2 alone.

The cases run on the base fixture pack, whose filter drops an unchecked item unless
a line above it holds `drill` or `watch list`.
"""

from __future__ import annotations

import dataclasses

import httpx
import pytest
import pytest_asyncio

from backend.extraction_contract.models import ExtractResponse
from backend.extraction_service.app import create_app
from backend.extraction_service.engine import ExtractionEngine
from backend.knowledge.ontologies.pack import load_pack
from tests.unit.extraction_service.conftest import make_extract_request

BASE = load_pack("tests.unit.knowledge.ontologies.fixtures.base_pack")

_ALL_DROPPED = "Deck jobs for Saturday:\n- [ ] coil the stern lines\n- [ ] scrub the foredeck\n"
_ONE_KEPT = "Deck jobs for Saturday:\n- [ ] coil the stern lines\n- [x] check the bilge pump\n"
_EXTRACT_REPLY = '{"entities": [], "relationships": []}'


@pytest_asyncio.fixture
async def plan_client(wired_llm, adapter, service_settings, health_probe, ctx_size_source):
    """The service on the base fixture pack, with the turn plan flag on."""
    settings = dataclasses.replace(service_settings, turn_plan=True)
    engine = ExtractionEngine(llm=wired_llm, adapter=adapter, settings=settings, pack=BASE)
    app = create_app(settings, engine, health_probe, ctx_size_source)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://service") as c:
        yield c


@pytest.mark.asyncio
async def test_a_turn_with_no_unit_left_is_answered_without_a_model_call(
    plan_client, fake_llama_state
):
    response = await plan_client.post(
        "/v1/extract", json=make_extract_request(utterance=_ALL_DROPPED)
    )

    assert response.status_code == 200, response.text
    assert fake_llama_state.chat_requests == []
    body = response.json()
    assert body["outcome"] == "extracted"
    assert body["payload"] == {"entities": [], "relationships": []}
    assert body["batch_record"] == []
    assert (body["attempts"], body["stage2_calls"]) == (0, 0)
    assert "turn_plan" not in body
    assert ExtractResponse.model_validate(body).stage2_calls == 0


@pytest.mark.asyncio
async def test_a_turn_that_keeps_one_unit_makes_at_least_one_call(plan_client, fake_llama_state):
    fake_llama_state.chat_responses = [
        '{"scope": "user-scope", "confidence": 0.9, "reasoning": "first person"}',
        '{"entities": [], "templates": [], "back_references": []}',
        _EXTRACT_REPLY,
    ]

    response = await plan_client.post("/v1/extract", json=make_extract_request(utterance=_ONE_KEPT))

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["attempts"] >= 1
    assert body["stage2_calls"] >= 1
    assert len(body["batch_record"]) == body["stage2_calls"]
