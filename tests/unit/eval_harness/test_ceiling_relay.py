"""The ceiling relay: llama-server's three routes, answered from a store.

The service under run A talks to the relay through the unchanged `LlamaServerProvider`
and reads the reply through the unchanged `GptOssAdapter.final_text`, so these tests drive
both against a relay served by uvicorn on a loopback port, and do not copy their parser.
The provider keeps its own `AsyncOpenAI` client, retries included, so the pending test
also shows that a pending key costs exactly one request.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import httpx  # noqa: E402
import openai  # noqa: E402
import pytest  # noqa: E402

from backend.extraction_service.adapters import GptOssAdapter  # noqa: E402
from backend.extraction_service.schemas import (  # noqa: E402
    DERIVATION_OUTPUT_SCHEMA,
    EXTRACTION_OUTPUT_SCHEMA,
    SCOPE_OUTPUT_SCHEMA,
)
from backend.llm.llama_server_provider import LlamaServerProvider  # noqa: E402
from backend.llm.models import LLMRequest, LLMResponse  # noqa: E402
from scripts.eval_harness.ceiling import relay as rl  # noqa: E402
from scripts.eval_harness.ceiling.store import served_text  # noqa: E402

MODEL = "answerer-relay"


def _body(
    schema: dict = EXTRACTION_OUTPUT_SCHEMA,
    *,
    messages: list[dict] | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    effort: str = "low",
) -> dict:
    return {
        "model": MODEL,
        "messages": messages
        or [
            {"role": "system", "content": "synthetic system prompt"},
            {"role": "user", "content": "synthetic utterance"},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 1.0,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": schema, "strict": True},
        },
        "chat_template_kwargs": {"reasoning_effort": effort},
    }


def _extract_request() -> LLMRequest:
    """An extract request prepared the way the engine prepares one for gpt-oss."""
    request = LLMRequest(
        messages=[
            {"role": "system", "content": "synthetic system prompt"},
            {"role": "user", "content": "synthetic utterance about café plans"},
        ],
        temperature=0.0,
        max_tokens=2048,
        top_p=1.0,
        response_schema=EXTRACTION_OUTPUT_SCHEMA,
    )
    return GptOssAdapter(reasoning_effort="low").prepare(request, stage="extract")


async def _call(provider: LlamaServerProvider, request: LLMRequest) -> LLMResponse:
    responses = [r async for r in provider.generate(request)]
    assert len(responses) == 1
    return responses[0]


# ---------------------------------------------------------------------------
# The request key
# ---------------------------------------------------------------------------


def test_no_body_key_is_excluded() -> None:
    assert rl.EXCLUDED_BODY_KEYS == ()


def test_request_key_is_the_sha256_of_the_sorted_compact_dump() -> None:
    body = _body()
    expected = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()

    assert rl.request_key(body) == expected


def test_request_key_is_stable_under_dict_order() -> None:
    body = _body()
    reordered = {k: body[k] for k in reversed(list(body))}
    reordered["response_format"] = {
        "json_schema": {"strict": True, "schema": EXTRACTION_OUTPUT_SCHEMA, "name": "response"},
        "type": "json_schema",
    }

    assert rl.request_key(reordered) == rl.request_key(body)


@pytest.mark.parametrize(
    "changed",
    [
        pytest.param(
            _body(messages=[{"role": "system", "content": "synthetic system prompt"}]),
            id="messages",
        ),
        pytest.param(
            _body(
                messages=[
                    {"role": "system", "content": "synthetic system prompt"},
                    {"role": "user", "content": "synthetic utterance."},
                ]
            ),
            id="message-text",
        ),
        pytest.param(_body(SCOPE_OUTPUT_SCHEMA), id="schema"),
        pytest.param(_body(temperature=0.7), id="temperature"),
        pytest.param(_body(max_tokens=96), id="max_tokens"),
        pytest.param(_body(effort="high"), id="reasoning_effort"),
        pytest.param({**_body(), "model": "other"}, id="model"),
    ],
)
def test_request_key_changes_with_any_message_schema_or_parameter(changed: dict) -> None:
    assert rl.request_key(changed) != rl.request_key(_body())


# ---------------------------------------------------------------------------
# Call kinds
# ---------------------------------------------------------------------------


def test_request_kind_follows_the_imported_stage_schemas() -> None:
    repair_messages = [
        {"role": "system", "content": "synthetic system prompt"},
        {"role": "user", "content": "synthetic utterance"},
        {"role": "assistant", "content": "not json"},
        {"role": "user", "content": "repair"},
    ]

    assert rl.request_kind(_body(SCOPE_OUTPUT_SCHEMA)) == "scope"
    assert rl.request_kind(_body(EXTRACTION_OUTPUT_SCHEMA)) == "extract"
    assert rl.request_kind(_body(messages=repair_messages)) == "repair"
    assert rl.request_kind(_body(DERIVATION_OUTPUT_SCHEMA)) == "derive"
    assert rl.request_kind(_body({"type": "object"})) is None
    assert rl.request_kind({"model": MODEL, "messages": []}) is None


def test_offered_request_carries_the_key_kind_and_schema_of_a_logged_request() -> None:
    log = rl.RelayLog()
    body = _body(SCOPE_OUTPUT_SCHEMA)
    entry = log.record(tag="p1", body=body, kind="scope", answered=False)

    offered = rl.offered_request(entry)

    assert offered.key == rl.request_key(body)
    assert offered.kind == "scope"
    assert offered.schema == SCOPE_OUTPUT_SCHEMA


# ---------------------------------------------------------------------------
# /health, /props, attribution and the log
# ---------------------------------------------------------------------------


def test_health_and_props_reply_as_llama_server_does() -> None:
    with rl.running_relay(rl.build_relay({}, rl.RelayLog())) as base_url:
        health = httpx.get(f"{base_url}/health")
        props = httpx.get(f"{base_url}/props")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert props.status_code == 200
    assert props.json() == {"default_generation_settings": {"n_ctx": 16384}}


@pytest.mark.asyncio
async def test_the_unchanged_provider_reads_health_and_context_size() -> None:
    with rl.running_relay(rl.build_relay({}, rl.RelayLog())) as base_url:
        provider = LlamaServerProvider(base_url=base_url, model=MODEL)
        assert await provider.health_check() is True
        assert await provider.server_context_size() == 16384


def test_requests_are_attributed_to_the_latest_probe_tag_in_arrival_order() -> None:
    log = rl.RelayLog()
    scope, extract = _body(SCOPE_OUTPUT_SCHEMA), _body()
    with rl.running_relay(rl.build_relay({rl.request_key(extract): {"x": 1}}, log)) as base_url:
        untagged = httpx.post(f"{base_url}/v1/chat/completions", json=scope)
        tagged = httpx.post(f"{base_url}/_relay/probe", json={"tag": "dvl-01-synthetic"})
        httpx.post(f"{base_url}/v1/chat/completions", json=scope)
        httpx.post(f"{base_url}/_relay/probe", json={"tag": "dvl-02-synthetic"})
        httpx.post(f"{base_url}/v1/chat/completions", json=extract)
        served_log = httpx.get(f"{base_url}/_relay/log")

    assert untagged.status_code == 400
    assert tagged.status_code == 204
    assert tagged.content == b""
    assert served_log.status_code == 200
    assert served_log.json() == [
        {
            "seq": 1,
            "tag": None,
            "key": rl.request_key(scope),
            "kind": "scope",
            "answered": False,
        },
        {
            "seq": 2,
            "tag": "dvl-01-synthetic",
            "key": rl.request_key(scope),
            "kind": "scope",
            "answered": False,
        },
        {
            "seq": 3,
            "tag": "dvl-02-synthetic",
            "key": rl.request_key(extract),
            "kind": "extract",
            "answered": True,
        },
    ]
    assert [e.body for e in log.entries()] == [scope, scope, extract]
    assert [e.public() for e in log.entries()] == served_log.json()


def test_probe_route_refuses_a_body_without_a_string_tag() -> None:
    with rl.running_relay(rl.build_relay({}, rl.RelayLog())) as base_url:
        missing = httpx.post(f"{base_url}/_relay/probe", json={"name": "x"})
        not_json = httpx.post(f"{base_url}/_relay/probe", content=b"{")

    assert missing.status_code == 400
    assert not_json.status_code == 400


def test_headers_never_reach_the_key() -> None:
    log = rl.RelayLog()
    body = _body()
    with rl.running_relay(rl.build_relay({}, log)) as base_url:
        httpx.post(f"{base_url}/v1/chat/completions", json=body)
        httpx.post(
            f"{base_url}/v1/chat/completions",
            json=body,
            headers={"x-stainless-retry-count": "1", "x-request-id": "abc"},
        )

    first, second = log.entries()
    assert first.key == second.key == rl.request_key(body)


# ---------------------------------------------------------------------------
# Stored and pending keys through the unchanged provider
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_stored_answer_reaches_final_text_unchanged_with_finish_reason_stop() -> None:
    request = _extract_request()
    answer = {
        "entities": [{"id": "e1", "type": "Place", "name": "café"}],
        "relationships": [{"source": "user", "predicate": "VISITS", "target": "e1"}],
    }

    # Learn the key from the relay itself, so the test never re-derives the SDK's body.
    probe_log = rl.RelayLog()
    with (
        rl.running_relay(rl.build_relay({}, probe_log)) as base_url,
        pytest.raises(openai.APIError),
    ):
        await _call(LlamaServerProvider(base_url=base_url, model=MODEL), request)
    (pending,) = probe_log.entries()
    assert pending.kind == "extract"
    assert set(pending.body) == {
        "model",
        "messages",
        "temperature",
        "top_p",
        "max_tokens",
        "stream",
        "response_format",
        "chat_template_kwargs",
    }
    assert pending.body["chat_template_kwargs"] == {"reasoning_effort": "low"}

    log = rl.RelayLog()
    with rl.running_relay(rl.build_relay({pending.key: answer}, log)) as base_url:
        response = await _call(LlamaServerProvider(base_url=base_url, model=MODEL), request)
        again = await _call(LlamaServerProvider(base_url=base_url, model=MODEL), request)

    assert response.finish_reason == "stop"
    assert response.content == served_text(answer)
    assert GptOssAdapter().final_text(response) == served_text(answer)
    assert json.loads(GptOssAdapter().final_text(response)) == answer
    assert again.content == response.content
    # Identical calls through the real SDK send identical bodies: no field varies.
    assert [e.key for e in log.entries()] == [pending.key, pending.key]
    assert [e.answered for e in log.entries()] == [True, True]


def test_a_stored_answer_has_the_llama_server_reply_shape() -> None:
    body = _body()
    key = rl.request_key(body)
    answer = {"entities": [], "relationships": []}
    with rl.running_relay(rl.build_relay({key: answer}, rl.RelayLog())) as base_url:
        reply = httpx.post(f"{base_url}/v1/chat/completions", json=body)

    assert reply.status_code == 200
    assert reply.json() == {
        "id": f"chatcmpl-relay-{key[:12]}",
        "object": "chat.completion",
        "created": 0,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": served_text(answer),
                    "reasoning_content": "",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@pytest.mark.asyncio
async def test_a_pending_key_gives_one_400_and_no_sdk_retry() -> None:
    log = rl.RelayLog()
    with rl.running_relay(rl.build_relay({}, log)) as base_url:
        provider = LlamaServerProvider(base_url=base_url, model=MODEL)
        # The provider's own client retries 5xx and connection errors; a 400 must not be one.
        assert provider._async_client.max_retries == 2
        with pytest.raises(openai.APIError) as caught:
            await _call(provider, _extract_request())

    assert isinstance(caught.value, openai.APIStatusError)
    assert caught.value.status_code == 400
    (entry,) = log.entries()
    assert entry.answered is False
    assert caught.value.body == {
        "code": 400,
        "type": "relay_pending",
        "message": f"pending {entry.key}",
    }


def test_a_pending_reply_has_the_contract_error_body() -> None:
    body = _body()
    with rl.running_relay(rl.build_relay({}, rl.RelayLog())) as base_url:
        reply = httpx.post(f"{base_url}/v1/chat/completions", json=body)

    assert reply.status_code == 400
    assert reply.json() == {
        "error": {
            "code": 400,
            "type": "relay_pending",
            "message": f"pending {rl.request_key(body)}",
        }
    }


def test_an_unclassified_body_is_refused_and_logged_without_a_kind() -> None:
    log = rl.RelayLog()
    body = _body({"type": "object"})
    with rl.running_relay(rl.build_relay({rl.request_key(body): {}}, log)) as base_url:
        reply = httpx.post(f"{base_url}/v1/chat/completions", json=body)
        not_json = httpx.post(f"{base_url}/v1/chat/completions", content=b"[1, 2")

    assert reply.status_code == 400
    assert reply.json()["error"]["type"] == "relay_unknown_kind"
    assert not_json.status_code == 400
    assert not_json.json()["error"]["type"] == "relay_bad_request"
    (entry,) = log.entries()
    assert entry.kind is None
    assert entry.answered is False


# ---------------------------------------------------------------------------
# Streamed replies, as llama-server's OpenAI-compatible route sends them
# ---------------------------------------------------------------------------


async def _streamed_call(provider: LlamaServerProvider, request: LLMRequest) -> LLMResponse:
    """One streamed call read as the engine reads it: the partials joined, the final's rest."""
    parts: list[str] = []
    finals: list[LLMResponse] = []
    async for chunk in provider.generate(request, stream=True):
        if chunk.partial:
            parts.append(chunk.content or "")
        else:
            finals.append(chunk)
    (final,) = finals
    return final.model_copy(update={"content": "".join(parts)})


def _events(reply: httpx.Response) -> list[str]:
    """The `data:` payloads of a server-sent event stream, in order."""
    return [
        line.removeprefix("data: ") for line in reply.text.split("\n") if line.startswith("data: ")
    ]


@pytest.mark.asyncio
async def test_a_streamed_answer_gives_the_single_reply_text_with_finish_reason_stop() -> None:
    request = _extract_request()
    answer = {
        "entities": [{"id": "e1", "type": "Place", "name": "café"}],
        "relationships": [{"source": "user", "predicate": "VISITS", "target": "e1"}],
    }

    # Learn the streamed request's key from the relay itself, as the stored-answer test does.
    probe_log = rl.RelayLog()
    with (
        rl.running_relay(rl.build_relay({}, probe_log)) as base_url,
        pytest.raises(openai.APIError),
    ):
        await _streamed_call(LlamaServerProvider(base_url=base_url, model=MODEL), request)
    (pending,) = probe_log.entries()
    assert pending.kind == "extract"
    assert pending.body["stream"] is True
    assert pending.body["stream_options"] == {"include_usage": True}

    log = rl.RelayLog()
    with rl.running_relay(rl.build_relay({pending.key: answer}, log)) as base_url:
        streamed = await _streamed_call(
            LlamaServerProvider(base_url=base_url, model=MODEL), request
        )

    assert streamed.finish_reason == "stop"
    assert streamed.content == served_text(answer)
    assert GptOssAdapter().final_text(streamed) == served_text(answer)
    assert json.loads(GptOssAdapter().final_text(streamed)) == answer
    assert streamed.usage is not None
    assert [e.answered for e in log.entries()] == [True]


@pytest.mark.asyncio
async def test_a_pending_key_asked_for_as_a_stream_gives_one_400_and_no_sdk_retry() -> None:
    log = rl.RelayLog()
    with rl.running_relay(rl.build_relay({}, log)) as base_url:
        provider = LlamaServerProvider(base_url=base_url, model=MODEL)
        with pytest.raises(openai.APIError) as caught:
            await _streamed_call(provider, _extract_request())

    assert isinstance(caught.value, openai.APIStatusError)
    assert caught.value.status_code == 400
    (entry,) = log.entries()
    assert entry.answered is False
    assert entry.body["stream"] is True
    assert caught.value.body == {
        "code": 400,
        "type": "relay_pending",
        "message": f"pending {entry.key}",
    }


@pytest.mark.parametrize("include_usage", [True, False, None], ids=["usage", "no-usage", "unset"])
def test_a_streamed_reply_has_the_llama_server_event_shape(include_usage: bool | None) -> None:
    body = {**_body(), "stream": True}
    if include_usage is not None:
        body["stream_options"] = {"include_usage": include_usage}
    key = rl.request_key(body)
    answer = {"entities": [], "relationships": []}
    with rl.running_relay(rl.build_relay({key: answer}, rl.RelayLog())) as base_url:
        reply = httpx.post(f"{base_url}/v1/chat/completions", json=body)

    assert reply.status_code == 200
    assert reply.headers["content-type"].startswith("text/event-stream")
    *chunks, done = _events(reply)
    assert done == "[DONE]"
    parsed = [json.loads(c) for c in chunks]
    head = {
        "id": f"chatcmpl-relay-{key[:12]}",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": MODEL,
    }
    expected = [
        {
            **head,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": served_text(answer)},
                    "finish_reason": None,
                }
            ],
        },
        {**head, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    if include_usage:
        expected.append(
            {
                **head,
                "choices": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        )
    assert parsed == expected


def test_a_body_with_stream_false_gets_the_single_completion() -> None:
    body = _body()
    assert body["stream"] is False
    key = rl.request_key(body)
    with rl.running_relay(rl.build_relay({key: {"x": 1}}, rl.RelayLog())) as base_url:
        reply = httpx.post(f"{base_url}/v1/chat/completions", json=body)

    assert reply.headers["content-type"].startswith("application/json")
    assert reply.json()["object"] == "chat.completion"


def test_the_relay_listens_on_loopback_only() -> None:
    with rl.running_relay(rl.build_relay({}, rl.RelayLog())) as base_url:
        assert base_url.startswith("http://127.0.0.1:")
