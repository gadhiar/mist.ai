"""Each Stage 2 call leaves one entry in the response's batch record, in call order.

An entry holds the batch's own units as the splitter gave them, the earlier units the
call was shown for reference, the payload the call returned, its token counts and why
it stopped. A repair retry and a call cut short at the cap are calls too: each leaves
an entry with no payload. So the record has `stage2_calls` entries, and a rail can
read which batch an entity came from rather than infer it.
"""

from __future__ import annotations

import json

import pytest

from backend.extraction_contract.models import ExtractionPayload, ExtractResponse
from backend.llm.models import LLMRequest, LLMResponse, UsageMetadata
from tests.unit.extraction_service.conftest import make_extract_request
from tests.unit.extraction_service.test_list_pass import (
    ScriptedLLM,
    batch_text,
    echo,
    make_engine,
    make_request,
    payload_for,
)

_LIST = (
    "Tools I picked up this year:\n"
    "- Velk, for the backend\n"
    "- Ornate\n"
    "  - its plugin manager\n"
    "- Quillet\n"
    "- Barrow, the same way as the first one"
)


def with_usage(prompt: int, completion: int):
    """Answer as `echo` does, with the token counts llama-server reports."""

    def answer(request: LLMRequest, call: int) -> LLMResponse:
        response = echo(request, call)
        usage = UsageMetadata(prompt_tokens=prompt + call, completion_tokens=completion + call)
        return response.model_copy(update={"usage": usage})

    return answer


@pytest.mark.asyncio
async def test_each_stage2_call_leaves_one_entry_in_call_order():
    llm = ScriptedLLM(with_usage(1000, 50))

    response = await make_engine(llm).run(make_request(_LIST))

    first, second = response.batch_record
    assert response.stage2_calls == 2
    assert first.units == [
        {
            "text": "Velk, for the backend",
            "context": ["Tools I picked up this year"],
            "kind": "list_item",
        },
        {"text": "Ornate", "context": ["Tools I picked up this year"], "kind": "list_item"},
        {
            "text": "its plugin manager",
            "context": ["Tools I picked up this year", "Ornate"],
            "kind": "list_item",
        },
        {"text": "Quillet", "context": ["Tools I picked up this year"], "kind": "list_item"},
    ]
    assert first.context == []
    assert second.units == [
        {
            "text": "Barrow, the same way as the first one",
            "context": ["Tools I picked up this year"],
            "kind": "list_item",
        }
    ]
    assert second.context == [
        "Tools I picked up this year: Velk, for the backend",
        "Tools I picked up this year: Ornate",
        "Tools I picked up this year > Ornate: its plugin manager",
        "Tools I picked up this year: Quillet",
    ]
    assert (first.prompt_tokens, first.completion_tokens, first.finish) == (1001, 51, "stop")
    assert (second.prompt_tokens, second.completion_tokens, second.finish) == (1002, 52, "stop")


@pytest.mark.asyncio
async def test_an_entrys_payload_is_what_its_call_returned():
    llm = ScriptedLLM()

    response = await make_engine(llm).run(make_request(_LIST))

    returned = [payload_for(batch_text(r)) for r in llm.stage2_requests]
    assert [entry.payload for entry in response.batch_record] == [
        ExtractionPayload.model_validate(payload) for payload in returned
    ]


@pytest.mark.asyncio
async def test_a_prose_turn_has_one_entry_holding_its_utterance():
    llm = ScriptedLLM()

    response = await make_engine(llm).run(make_request("I use Velk for my backend work"))

    (entry,) = response.batch_record
    assert entry.units == [
        {"text": "I use Velk for my backend work", "context": [], "kind": "prose"}
    ]
    assert entry.context == []
    assert entry.payload == response.payload
    assert (entry.prompt_tokens, entry.completion_tokens, entry.finish) == (None, None, "stop")


@pytest.mark.asyncio
async def test_a_repair_retry_leaves_an_entry_with_no_payload_before_its_own():
    def answer(request: LLMRequest, call: int) -> LLMResponse:
        if call == 1:
            return LLMResponse(content="not json", finish_reason="stop")
        return echo(request, call)

    response = await make_engine(ScriptedLLM(answer)).run(make_request("I use Velk at work"))

    failed, repaired = response.batch_record
    assert response.stage2_calls == 2
    assert (failed.payload, failed.finish) == (None, "stop")
    assert repaired.payload == response.payload
    assert failed.units == repaired.units


@pytest.mark.asyncio
async def test_a_call_cut_short_leaves_an_entry_and_each_half_its_own():
    def answer(request: LLMRequest, call: int) -> LLMResponse:
        if "\n" in batch_text(request):
            return LLMResponse(content='{"entities": [', finish_reason="length")
        return echo(request, call)

    text = "Errands:\n- post the parcel\n- ring Brenna"
    response = await make_engine(ScriptedLLM(answer)).run(make_request(text))

    whole, first, second = response.batch_record
    assert response.stage2_calls == 3
    assert (whole.payload, whole.finish) == (None, "length")
    assert len(whole.units) == 2
    assert [e.units[0]["text"] for e in (first, second)] == ["post the parcel", "ring Brenna"]
    assert [e.finish for e in (first, second)] == ["stop", "stop"]


@pytest.mark.asyncio
async def test_the_service_answers_with_the_batch_record(client, fake_llama_state):
    fake_llama_state.chat_responses = [
        '{"scope": "user-scope", "confidence": 0.9, "reasoning": "first person"}',
        '{"entities": [], "relationships": []}',
    ]

    response = await client.post("/v1/extract", json=make_extract_request(utterance=_LIST))

    assert response.status_code == 200
    body = response.json()
    assert len(body["batch_record"]) == body["stage2_calls"] == 2
    assert body["batch_record"][1]["context"][0] == (
        "Tools I picked up this year: Velk, for the backend"
    )
    restored = ExtractResponse.model_validate(body)
    assert json.loads(restored.model_dump_json())["batch_record"] == body["batch_record"]
