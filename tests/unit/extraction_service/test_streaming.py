"""The Stage 2 calls are streaming requests, and a call the service leaves closes its stream.

An extract call, a repair retry and the turn plan call are sent with
`stream: true` and read to their end; scope and derive are single replies.
A call that times out, or whose run is cancelled, closes its stream: the
fake llama-server counts such a call in `chat_aborted`. The turn plan call's
request is pinned in `test_turn_plan.py`.
"""

from __future__ import annotations

import asyncio
import dataclasses

import httpx
import pytest

from backend.extraction_contract.models import ExtractRequest
from backend.extraction_service.adapters import get_adapter
from backend.extraction_service.app import create_app
from backend.extraction_service.engine import ExtractionEngine, UpstreamLLMError
from backend.llm.models import LLMRequest, LLMResponse
from tests.unit.extraction_service.conftest import make_extract_request, until
from tests.unit.extraction_service.test_list_pass import ScriptedLLM, make_engine, make_request

_SCOPE_REPLY = '{"scope": "user-scope", "confidence": 0.9, "reasoning": "first person"}'
_EXTRACT_REPLY = (
    '{"entities": [{"id": "user", "name": "User", "type": "User"}, '
    '{"id": "rust", "name": "Rust", "type": "Technology"}], '
    '"relationships": [{"source": "user", "target": "rust", "type": "USES", '
    '"properties": {}}]}'
)
_DERIVE_REPLY = '{"operations": []}'
_DERIVATION = {
    "signal_types": ["feedback"],
    "matched_patterns": ["feedback:stop"],
    "existing_internal_entities": "No existing internal entities.",
    "assistant_response": "Got it.",
}


def _client_for(settings, wired_llm, health_probe, ctx_size_source) -> httpx.AsyncClient:
    """A client of the service under `settings`, wired as the `service_app` fixture is."""
    adapter = get_adapter(
        settings.adapter_name,
        reasoning_effort=settings.reasoning_effort,
        reasoning_budget_tokens=settings.reasoning_budget_tokens,
    )
    engine = ExtractionEngine(llm=wired_llm, adapter=adapter, settings=settings)
    app = create_app(settings, engine, health_probe, ctx_size_source)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://service")


@pytest.mark.asyncio
class TestWhatIsStreamed:
    async def test_the_extract_request_streams_and_scope_and_derive_do_not(
        self, client, fake_llama_state
    ):
        fake_llama_state.chat_responses = [_SCOPE_REPLY, _EXTRACT_REPLY, _DERIVE_REPLY]

        response = await client.post(
            "/v1/extract", json=make_extract_request(derivation=_DERIVATION)
        )

        assert response.status_code == 200, response.text
        sent = fake_llama_state.chat_requests
        assert [body["stream"] for body in sent] == [False, True, False]
        # The usage chunk is asked for, so a streamed call still reports its counts.
        assert [body.get("stream_options") for body in sent] == [
            None,
            {"include_usage": True},
            None,
        ]

    async def test_a_streamed_reply_is_read_to_its_end(self, client, fake_llama_state):
        """The fake sends the content in two deltas; the payload is the whole of it."""
        fake_llama_state.chat_responses = [_SCOPE_REPLY, _EXTRACT_REPLY]

        response = await client.post("/v1/extract", json=make_extract_request())

        assert response.status_code == 200, response.text
        payload = response.json()["payload"]
        assert [entity["id"] for entity in payload["entities"]] == ["user", "rust"]
        assert [(r["source"], r["type"], r["target"]) for r in payload["relationships"]] == [
            ("user", "USES", "rust")
        ]
        assert fake_llama_state.chat_aborted == 0
        assert fake_llama_state.streams_open == 0

    async def test_the_repair_retry_streams_too(self, client, fake_llama_state):
        fake_llama_state.chat_responses = [_SCOPE_REPLY, "not json at all", _EXTRACT_REPLY]

        response = await client.post("/v1/extract", json=make_extract_request())

        assert response.status_code == 200, response.text
        assert response.json()["attempts"] == 2
        assert [body["stream"] for body in fake_llama_state.chat_requests] == [False, True, True]


@pytest.mark.asyncio
class TestACallTheServiceLeaves:
    async def test_a_timed_out_extract_call_closes_its_stream(
        self,
        fake_llama_state,
        held_stream,
        service_settings,
        wired_llm,
        health_probe,
        ctx_size_source,
    ):
        # The reply is held after its first delta on an event nobody sets, so
        # the call outlasts any timeout: the short budget only sets how long
        # the test takes.
        settings = dataclasses.replace(
            service_settings, llm_timeout_seconds=0.05, scope_enabled=False
        )

        async with _client_for(settings, wired_llm, health_probe, ctx_size_source) as client:
            response = await client.post("/v1/extract", json=make_extract_request())

        assert response.status_code == 504
        assert response.json()["error"]["code"] == "timeout"
        assert [body["stream"] for body in fake_llama_state.chat_requests] == [True]
        assert fake_llama_state.chat_aborted == 1
        assert fake_llama_state.streams_open == 0
        assert not held_stream.is_set()

    async def test_a_cancelled_run_closes_its_stream(self, fake_llama_state, held_stream, engine):
        run = asyncio.create_task(engine.run(ExtractRequest(**make_extract_request())))
        await until(lambda: fake_llama_state.streams_open == 1)

        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run

        assert fake_llama_state.chat_aborted == 1
        assert fake_llama_state.streams_open == 0
        assert not held_stream.is_set()

    async def test_a_held_stream_that_is_released_ends_as_any_other(
        self, client, fake_llama_state, held_stream
    ):
        fake_llama_state.chat_responses = [_SCOPE_REPLY, _EXTRACT_REPLY]
        post = asyncio.create_task(client.post("/v1/extract", json=make_extract_request()))
        await until(lambda: fake_llama_state.streams_open == 1)

        held_stream.set()
        response = await post

        assert response.status_code == 200, response.text
        assert [entity["id"] for entity in response.json()["payload"]["entities"]] == [
            "user",
            "rust",
        ]
        assert fake_llama_state.chat_aborted == 0

    async def test_a_stream_cut_short_is_an_upstream_failure(self, client, fake_llama_state):
        """A reply that breaks off part-way is never read as an extraction."""
        fake_llama_state.chat_responses = [_SCOPE_REPLY, _EXTRACT_REPLY]
        fake_llama_state.stream_drop = True

        response = await client.post("/v1/extract", json=make_extract_request())

        assert response.status_code == 502
        assert response.json()["error"]["code"] == "upstream_llm"
        assert fake_llama_state.streams_open == 0


class _NoFinalChunk(ScriptedLLM):
    """A provider whose streamed reply ends after its content, with no final chunk."""

    async def generate(self, request: LLMRequest, *, stream: bool = False):
        if not stream:
            async for response in super().generate(request, stream=False):
                yield response
            return
        self.requests.append(request)
        yield LLMResponse(content='{"entities": [], "relationships": []}', partial=True)


@pytest.mark.asyncio
class TestTheEngineReadsAStream:
    async def test_only_the_stage_2_calls_are_streamed(self):
        llm = ScriptedLLM()

        await make_engine(llm).run(make_request("I like tea", derivation=_DERIVATION))

        assert llm.streamed == [False, True, False]

    async def test_every_batch_of_a_structured_turn_is_streamed(self):
        llm = ScriptedLLM()

        await make_engine(llm).run(
            make_request("Teas I like:\n" + "\n".join(f"{n}. tea {n}" for n in range(1, 6)))
        )

        assert len(llm.stage2_requests) == 2
        assert llm.streamed == [False, True, True]

    async def test_a_stream_with_no_final_chunk_is_an_upstream_failure(self):
        """Content with no final chunk behind it is not parsed as the whole reply."""
        llm = _NoFinalChunk()

        with pytest.raises(UpstreamLLMError, match="final chunk"):
            await make_engine(llm).run(make_request("I like tea"))
