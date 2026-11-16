"""With `EXTRACTION_TURN_PLAN` off, the turn plan stage changes nothing the service sends.

Every literal here but `SERVING_IDENTITY` was captured at 058e52c, the commit
before the turn plan stage, except where T48 or T60 re-pinned one, as noted at
each. The file imports nothing the stage added. It pins what a flag-off service
sends and reports:

- every request body the fake llama-server receives for one structured turn
  and one prose turn, as the sha256 of their canonical JSON. T49 put a
  reasoning budget on the Stage 2 requests and made them streaming requests,
  and T62 put `cache_prompt: false` and the turn's date
  (`chat_template_kwargs.current_date`) on every request; the bodies are
  hashed with those fields as 058e52c sent them (`split_t49`), and what T49
  and T62 set is pinned beside them (`split_t49`'s second value, and
  `split_t62`), so the hashes are still the 058e52c ones;
- the three stage output schemas;
- `prompt_sha256` for the gpt-oss adapter;
- the serving identity of the `service_settings` fixture, which T49 moved
  for that budget and T62 for the cache flag, and the 058e52c identity it is
  with both taken out;
- the keys of the `/v1/extract` response, with no `turn_plan` among them.

The request bodies hold the prompts, so a prompt edit moves the two body
hashes and `prompt_sha256` together. Re-pin all three in that edit's commit,
and only there. The bodies also hold each stage's `max_tokens`, which
`prompt_sha256` leaves out, so a cap edit moves the two body hashes alone.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from backend.extraction_service import engine as engine_module
from backend.extraction_service import schemas
from backend.extraction_service.adapters import get_adapter
from backend.extraction_service.app import serving_identity_components
from backend.knowledge.extraction.serving_identity import (
    components_dict,
    compute_serving_identity,
)
from tests.unit.extraction_service.conftest import make_extract_request
from tests.unit.extraction_service.fake_llama import FAKE_N_CTX

_SCOPE_REPLY = '{"scope": "user-scope", "confidence": 0.9, "reasoning": "first person"}'
_EXTRACT_REPLY = '{"entities": [], "relationships": []}'
_DERIVE_REPLY = '{"operations": []}'
_DERIVATION = {
    "signal_types": ["feedback"],
    "matched_patterns": ["feedback:stop"],
    "existing_internal_entities": "No existing internal entities.",
    "assistant_response": "Got it.",
}

# Five items under one lead-in: two Stage 2 batches.
STRUCTURED_UTTERANCE = "Things I want to try this year:\n" + "\n".join(
    f"{n}. thing {n}" for n in range(1, 6)
)
PROSE_UTTERANCE = "I have been learning Rust for a few months"

# T48: the second batch, `5. thing 5`, refers back to no item, so it is no longer sent
# the four units before it as context lines; at 058e52c this was
# acf2adde1d0473a0af20963052b08e27bc1ad9df02748dc4ba16013089aef834.
# T52: the bodies hold the system prompt, whose Rule 28 and Example 30 now show earlier
# units as the engine sends them, and a Stage 2 body the schema generated from the
# pack's declarations; before T52 the two were
# f41d3afdced9a0f9e158a52f9abf5b38db9ca20a8eb79060e78704078f8f7dbc and
# 1c2489eb2a125f17e56bac300608e1798be6d447dab861447fe8e0e01cb35eb0.
# T60: the scope body is sent with `max_tokens` 512 in place of 96; before T60 the two were
# 49903b6ece2a885e20b813bd77acf90cd8a49110a4f44f8213da66dc8d815b5d and
# 11b54c027d71919b5bcbf7b2c5dd3aa80b4f6d200d444928215c6d9d9b8cf281.
# T64: the bodies hold the system prompt and Stage 2 schema before T52 again, with T60's
# scope `max_tokens`; before T64, under T52's prompt and schema, the two were
# 91ee7c0912e38bb495d1528d83f37929a4c74cc138b619f523ca3c53b1452f2c and
# 2370541fe87ff38807fd6baaad3ad8d90233592d196797d3a41873a2dc8a2fd2.
STRUCTURED_BODIES_SHA256 = "a40685dba9adfc5a172d650b7c96ccd3629ac835042bf7003a2fdb1d01ebfb76"
PROSE_BODIES_SHA256 = "6665623b4756371e02e7a4991932eec3538e2b61f2f53c83474f05308cb8ddde"
# T48: `EARLIER_ITEMS_HEADING` joined the hash; at 058e52c this was
# 6c7cfb5fb49e1c8382fb298d0b50607cd6d6c8c5f43c9277f1f59ef9a76c590e.
# T52: Rule 28, Example 30 and the generated schema moved it; before T52 it was
# dfae2785c08ededffbbf4369ed468b9e803abde321566481d8f7bc4aae80298b.
# T64: undid T52's move, restoring the prompt and schema before T52; under T52 it was
# 3db8b2a7e7346efa14835792c38ff82123043bec67b82a017d5f67d18f719212.
PROMPT_SHA256 = "dfae2785c08ededffbbf4369ed468b9e803abde321566481d8f7bc4aae80298b"
SERVING_IDENTITY_AT_058E52C = "si1-0a4fdfffca232a0e3d27716f7ab0e41ed75715ed7eb932b05939b692bc358cb1"
# T49: `request.extract.thinking_budget_tokens` went from None to 8192.
# T62: `request.<stage>.cache_prompt` joined it, false at every stage; at b56f28a it was
# si1-f967ffbbd60a599a2627654e2ac373bbb787390b374cf1e9a3e2c87dc3b83050.
SERVING_IDENTITY = "si1-b72fd980a8b8b480a5aab0e6f7c319fd656283d0a3cc0c03bb4c625bd3797c81"
# What T49 set on a Stage 2 request: the budget, and a streamed reply with its
# usage chunk. A scope or derive request has neither, and is not streamed.
T49_EXTRACT_FIELDS = {
    "reasoning_budget_tokens": 8192,
    "stream": True,
    "stream_options": {"include_usage": True},
}
T49_OTHER_FIELDS = {"stream": False}
# What T62 set on every request: no prompt cache, and the turn's reference date,
# the date of `make_extract_request`'s `recorded_at`, for the template's date line.
T62_FIELDS = {"cache_prompt": False, "current_date": "2026-04-21"}
RESPONSE_KEYS = [
    "contract_version",
    "job_id",
    "outcome",
    "scope",
    "payload",
    "derivation",
    "stamps",
    "timings_ms",
    "attempts",
    "stage2_calls",
    "warnings",
    # T48: one entry per Stage 2 call.
    "batch_record",
]


def canonical_sha256(value: object) -> str:
    """sha256 of `value` as canonical JSON: sorted keys, no whitespace."""
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _without_t62(body: dict) -> dict:
    """`body` less what T62 set: `cache_prompt`, and `current_date` from its template kwargs.

    The kwargs key itself goes too when the date was all it held.
    """
    out = {k: v for k, v in body.items() if k != "cache_prompt"}
    kwargs = dict(body.get("chat_template_kwargs") or {})
    kwargs.pop("current_date", None)
    if kwargs:
        out["chat_template_kwargs"] = kwargs
    else:
        out.pop("chat_template_kwargs", None)
    return out


def split_t49(bodies: list[dict]) -> tuple[list[dict], list[dict]]:
    """Each body as 058e52c sent it, and beside it what T49 set on it.

    At 058e52c no request was streamed and none carried a budget or stream
    options. Nor did any carry what T62 set, which the first value leaves out
    too and `split_t62` returns.
    """
    before = [
        {
            **{k: v for k, v in _without_t62(body).items() if k not in T49_EXTRACT_FIELDS},
            "stream": False,
        }
        for body in bodies
    ]
    set_by_t49 = [{k: body[k] for k in T49_EXTRACT_FIELDS if k in body} for body in bodies]
    return before, set_by_t49


def split_t62(bodies: list[dict]) -> list[dict]:
    """What T62 set on each body: `cache_prompt` and the template's `current_date`, when sent."""
    set_by_t62: list[dict] = []
    for body in bodies:
        fields = {}
        if "cache_prompt" in body:
            fields["cache_prompt"] = body["cache_prompt"]
        kwargs = body.get("chat_template_kwargs") or {}
        if "current_date" in kwargs:
            fields["current_date"] = kwargs["current_date"]
        set_by_t62.append(fields)
    return set_by_t62


def test_the_pinned_date_is_the_requests():
    assert make_extract_request()["recorded_at"].startswith(T62_FIELDS["current_date"] + "T")


@pytest.mark.asyncio
async def test_a_structured_turn_sends_the_pinned_request_bodies(client, fake_llama_state):
    fake_llama_state.chat_responses = [_SCOPE_REPLY, _EXTRACT_REPLY, _EXTRACT_REPLY, _DERIVE_REPLY]

    response = await client.post(
        "/v1/extract",
        json=make_extract_request(utterance=STRUCTURED_UTTERANCE, derivation=_DERIVATION),
    )

    assert response.status_code == 200, response.text
    assert len(fake_llama_state.chat_requests) == 4
    before, set_by_t49 = split_t49(fake_llama_state.chat_requests)
    assert canonical_sha256(before) == STRUCTURED_BODIES_SHA256
    assert set_by_t49 == [
        T49_OTHER_FIELDS,
        T49_EXTRACT_FIELDS,
        T49_EXTRACT_FIELDS,
        T49_OTHER_FIELDS,
    ]
    assert split_t62(fake_llama_state.chat_requests) == [T62_FIELDS] * 4
    assert list(response.json()) == RESPONSE_KEYS


@pytest.mark.asyncio
async def test_a_prose_turn_sends_the_pinned_request_bodies(client, fake_llama_state):
    fake_llama_state.chat_responses = [_SCOPE_REPLY, _EXTRACT_REPLY, _DERIVE_REPLY]

    response = await client.post(
        "/v1/extract",
        json=make_extract_request(utterance=PROSE_UTTERANCE, derivation=_DERIVATION),
    )

    assert response.status_code == 200, response.text
    assert len(fake_llama_state.chat_requests) == 3
    before, set_by_t49 = split_t49(fake_llama_state.chat_requests)
    assert canonical_sha256(before) == PROSE_BODIES_SHA256
    assert set_by_t49 == [T49_OTHER_FIELDS, T49_EXTRACT_FIELDS, T49_OTHER_FIELDS]
    assert split_t62(fake_llama_state.chat_requests) == [T62_FIELDS] * 3
    assert list(response.json()) == RESPONSE_KEYS


PINNED_SCOPE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "scope": {"type": "string"},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["scope"],
    "additionalProperties": True,
}
PINNED_EXTRACTION_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {"type": "object", "additionalProperties": True},
        },
        "relationships": {
            "type": "array",
            "items": {"type": "object", "additionalProperties": True},
        },
    },
    "required": ["entities", "relationships"],
    "additionalProperties": True,
}
PINNED_DERIVATION_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "operations": {
            "type": "array",
            "items": {"type": "object", "additionalProperties": True},
        },
    },
    "required": ["operations"],
    "additionalProperties": True,
}


def test_the_three_output_schemas_are_the_pinned_ones():
    assert schemas.SCOPE_OUTPUT_SCHEMA == PINNED_SCOPE_OUTPUT_SCHEMA
    assert schemas.EXTRACTION_OUTPUT_SCHEMA == PINNED_EXTRACTION_OUTPUT_SCHEMA
    assert schemas.DERIVATION_OUTPUT_SCHEMA == PINNED_DERIVATION_OUTPUT_SCHEMA


def test_the_prompt_hash_is_the_pinned_one():
    assert engine_module._compute_prompt_sha256(get_adapter("gptoss")) == PROMPT_SHA256


def test_the_serving_identity_is_the_pinned_one(service_settings, engine):
    components = serving_identity_components(service_settings, engine, FAKE_N_CTX)

    assert compute_serving_identity(components) == SERVING_IDENTITY
    assert "si1-" + canonical_sha256(components_dict(components)) == SERVING_IDENTITY
    assert "turn_plan" not in components_dict(components)


def test_the_serving_identity_is_the_058e52c_one_but_for_the_extract_budget_and_cache_flag(
    service_settings, engine
):
    flat = components_dict(serving_identity_components(service_settings, engine, FAKE_N_CTX))

    assert flat["request.extract.thinking_budget_tokens"] == 8192
    flat["request.extract.thinking_budget_tokens"] = None
    for stage in ("scope", "extract", "derive"):
        assert flat.pop(f"request.{stage}.cache_prompt") is False

    assert "si1-" + canonical_sha256(flat) == SERVING_IDENTITY_AT_058E52C
