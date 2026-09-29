"""End-to-end ASGI tests for the extraction service.

Exercises `POST /v1/extract` (and the health gate it depends on) against a
FastAPI app wired to a fake llama-server, over `httpx.ASGITransport` --
no network, no GPU. Covers the T0 brief's acceptance criteria 2-4.
"""

from __future__ import annotations

import asyncio
import dataclasses

import httpx
import pytest

from backend.extraction_service.app import LlamaPropsContextSize, create_app
from backend.extraction_service.engine import ExtractionEngine
from backend.extraction_service.schemas import (
    DERIVATION_OUTPUT_SCHEMA,
    EXTRACTION_OUTPUT_SCHEMA,
    SCOPE_OUTPUT_SCHEMA,
)
from backend.knowledge.version_stamps import EXTRACTION_VERSION
from tests.unit.extraction_service.conftest import FAKE_BASE_URL, make_extract_request
from tests.unit.extraction_service.fake_llama import FAKE_N_CTX


@pytest.mark.asyncio
class TestHappyPath:
    async def test_extract_returns_stamped_response_with_derivation(
        self, client, fake_llama_state, service_settings
    ):
        fake_llama_state.chat_responses = [
            '{"scope": "user-scope", "confidence": 0.9, "reasoning": "first person"}',
            (
                '{"entities": [{"id": "user", "name": "User", "type": "User"}, '
                '{"id": "rust", "name": "Rust", "type": "Technology"}], '
                '"relationships": [{"source": "user", "target": "rust", "type": "USES", '
                '"properties": {}}]}'
            ),
            '{"operations": []}',
        ]
        body = make_extract_request(
            derivation={
                "signal_types": ["feedback"],
                "matched_patterns": ["feedback:stop"],
                "existing_internal_entities": "No existing internal entities.",
                "assistant_response": "Got it.",
            }
        )

        response = await client.post("/v1/extract", json=body)

        assert response.status_code == 200
        data = response.json()
        assert data["outcome"] == "extracted"
        assert data["scope"]["label"] == "user-scope"
        assert [e["id"] for e in data["payload"]["entities"]] == ["user", "rust"]
        assert data["derivation"]["operations"] == []
        assert data["stamps"]["extraction_version"] == EXTRACTION_VERSION
        assert data["stamps"]["model_hash"] == service_settings.model_hash
        assert data["stamps"]["adapter"] == "gptoss"
        assert data["stamps"]["llama_cpp_build"] == service_settings.llama_cpp_build
        assert data["attempts"] == 1
        assert data["timings_ms"]["scope"] is not None
        assert data["timings_ms"]["derive"] is not None
        assert len(fake_llama_state.chat_requests) == 3

        # gpt-oss adapter forwards reasoning_effort via chat_template_kwargs
        # and omits reasoning_budget_tokens entirely when unset.
        first_request = fake_llama_state.chat_requests[0]
        assert first_request["chat_template_kwargs"]["reasoning_effort"] == "low"
        assert "reasoning_budget_tokens" not in first_request

    async def test_extract_without_derivation_input_skips_stage_9(self, client, fake_llama_state):
        fake_llama_state.chat_responses = [
            '{"scope": "unknown", "confidence": 0.0}',
            '{"entities": [], "relationships": []}',
        ]
        body = make_extract_request(derivation=None)

        response = await client.post("/v1/extract", json=body)

        assert response.status_code == 200
        data = response.json()
        assert data["derivation"] is None
        assert data["timings_ms"]["derive"] is None
        assert len(fake_llama_state.chat_requests) == 2


@pytest.mark.asyncio
class TestGptOssDefaultConstrainedMode:
    """Pins the outgoing request body each stage builds under the gpt-oss default.

    With `EXTRACTION_CONSTRAINED_MODE` unset, the engine falls through to
    `GptOssAdapter.default_constrained_mode` ("schema"). Each stage must then
    send llama-server a `json_schema` response_format carrying that stage's
    own output schema -- never an empty schema, and never plain `json_object`.
    """

    async def test_scope_extract_derive_each_send_their_own_json_schema(
        self, client, fake_llama_state, service_settings
    ):
        assert service_settings.constrained_mode is None
        fake_llama_state.chat_responses = [
            '{"scope": "user-scope", "confidence": 0.9}',
            '{"entities": [], "relationships": []}',
            '{"operations": []}',
        ]
        body = make_extract_request(
            derivation={
                "signal_types": ["feedback"],
                "matched_patterns": ["feedback:stop"],
                "existing_internal_entities": "No existing internal entities.",
                "assistant_response": "Got it.",
            }
        )

        response = await client.post("/v1/extract", json=body)

        assert response.status_code == 200
        requests = fake_llama_state.chat_requests
        assert len(requests) == 3
        expected_schemas = [
            SCOPE_OUTPUT_SCHEMA,
            EXTRACTION_OUTPUT_SCHEMA,
            DERIVATION_OUTPUT_SCHEMA,
        ]
        for request, schema in zip(requests, expected_schemas, strict=True):
            assert request["response_format"] == {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema, "strict": True},
            }


@pytest.mark.asyncio
class TestIdempotency:
    async def test_same_job_id_twice_returns_identical_body_one_llm_call_set(
        self, client, fake_llama_state
    ):
        fake_llama_state.chat_responses = [
            '{"scope": "unknown", "confidence": 0.0}',
            '{"entities": [], "relationships": []}',
        ]
        body = make_extract_request(job_id="job-idem")

        first = await client.post("/v1/extract", json=body)
        second = await client.post("/v1/extract", json=body)

        assert first.status_code == 200
        assert second.status_code == 200
        assert first.json() == second.json()
        assert len(fake_llama_state.chat_requests) == 2

    async def test_concurrent_duplicate_requests_are_single_flighted(
        self, client, fake_llama_state, blocked_chat
    ):
        fake_llama_state.chat_responses = [
            '{"scope": "unknown", "confidence": 0.0}',
            '{"entities": [], "relationships": []}',
        ]
        body = make_extract_request(job_id="job-concurrent")

        # Both requests are sent while the first LLM call is held, so the
        # second can only join the in-flight run or find the stored result.
        requests = [asyncio.create_task(client.post("/v1/extract", json=body)) for _ in range(2)]
        while not fake_llama_state.chat_requests:
            await asyncio.sleep(0)
        for _ in range(20):
            await asyncio.sleep(0)
        blocked_chat.set()
        results = await asyncio.gather(*requests)

        assert all(r.status_code == 200 for r in results)
        assert results[0].json() == results[1].json()
        assert len(fake_llama_state.chat_requests) == 2


@pytest.mark.asyncio
class TestErrors:
    async def test_epoch_mismatch_wrong_extraction_version_makes_zero_llm_calls(
        self, client, fake_llama_state
    ):
        body = make_extract_request(extraction_version="a-stale-version")

        response = await client.post("/v1/extract", json=body)

        assert response.status_code == 409
        assert response.json()["error"]["code"] == "epoch_mismatch"
        assert fake_llama_state.chat_requests == []

    async def test_epoch_mismatch_wrong_model_hash_makes_zero_llm_calls(
        self, client, fake_llama_state
    ):
        body = make_extract_request(model_hash="wrong-hash")

        response = await client.post("/v1/extract", json=body)

        assert response.status_code == 409
        assert response.json()["error"]["code"] == "epoch_mismatch"
        assert fake_llama_state.chat_requests == []

    async def test_contract_mismatch_on_incompatible_major_version(self, client, fake_llama_state):
        body = make_extract_request(contract_version="2.0.0")

        response = await client.post("/v1/extract", json=body)

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "contract_mismatch"
        assert fake_llama_state.chat_requests == []

    async def test_malformed_body_returns_error_envelope(self, client):
        response = await client.post("/v1/extract", json={"job_id": "only-this-field"})

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "contract_mismatch"

    async def test_model_loading_when_health_reports_loading(self, client, fake_llama_state):
        fake_llama_state.health_status = 503
        body = make_extract_request()

        response = await client.post("/v1/extract", json=body)

        assert response.status_code == 503
        assert response.json()["error"]["code"] == "model_loading"
        assert fake_llama_state.chat_requests == []

    async def test_upstream_llm_error_when_chat_completions_returns_500(
        self, client, fake_llama_state
    ):
        fake_llama_state.chat_status_code = 500
        body = make_extract_request()

        response = await client.post("/v1/extract", json=body)

        assert response.status_code == 502
        assert response.json()["error"]["code"] == "upstream_llm"

    async def test_timeout_maps_to_504(
        self,
        fake_llama_state,
        blocked_chat,
        service_settings,
        wired_llm,
        adapter,
        health_probe,
        ctx_size_source,
    ):
        # Every chat call blocks on an event nobody sets, so each call
        # outlasts any timeout: the short budget below only sets how long
        # the test takes, and no margin can make it flake.
        settings = dataclasses.replace(service_settings, llm_timeout_seconds=0.05)
        engine = ExtractionEngine(llm=wired_llm, adapter=adapter, settings=settings)
        app = create_app(settings, engine, health_probe, ctx_size_source)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://service") as client:
            response = await client.post("/v1/extract", json=make_extract_request())

        assert response.status_code == 504
        assert response.json()["error"]["code"] == "timeout"
        # Scope timed out (degraded), then the extraction call timed out.
        assert len(fake_llama_state.chat_requests) == 2
        assert not blocked_chat.is_set()

    async def test_unparseable_then_valid_output_succeeds_with_two_attempts(
        self, client, fake_llama_state
    ):
        fake_llama_state.chat_responses = [
            '{"scope": "unknown", "confidence": 0.0}',
            "not json at all",
            '{"entities": [], "relationships": []}',
        ]
        body = make_extract_request()

        response = await client.post("/v1/extract", json=body)

        assert response.status_code == 200
        assert response.json()["attempts"] == 2

    async def test_always_unparseable_output_gives_502_never_empty_extraction(
        self, client, fake_llama_state
    ):
        fake_llama_state.chat_responses = [
            '{"scope": "unknown", "confidence": 0.0}',
            "not json at all",
        ]
        body = make_extract_request()

        response = await client.post("/v1/extract", json=body)

        assert response.status_code == 502
        assert response.json()["error"]["code"] == "upstream_llm"


@pytest.mark.asyncio
class TestHealthAndInfo:
    async def test_health_ok_when_fake_llama_health_is_200(self, client):
        response = await client.get("/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["llm_reachable"] is True

    async def test_health_loading_when_fake_llama_health_is_503(self, client, fake_llama_state):
        fake_llama_state.health_status = 503

        response = await client.get("/v1/health")

        assert response.status_code == 200
        assert response.json()["status"] == "loading"

    async def test_info_reports_configured_stamps(self, client, service_settings):
        response = await client.get("/v1/info")

        assert response.status_code == 200
        data = response.json()
        assert data["extraction_version"] == EXTRACTION_VERSION
        assert data["model_hash"] == service_settings.model_hash
        assert data["adapter"] == "gptoss"
        assert data["location_label"] == service_settings.location_label


async def _get_info(app) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://service") as client:
        return await client.get("/v1/info")


def _props_client(handler) -> httpx.AsyncClient:
    """An httpx client whose every request is answered (or raised) by `handler`."""
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
class TestInfoServingConfig:
    """`/v1/info` reports the serving config a result was produced under (contract 1.1.0)."""

    async def test_info_reports_the_settings_and_the_props_ctx_size(self, client, service_settings):
        response = await client.get("/v1/info")

        assert response.status_code == 200
        data = response.json()
        assert data["contract_version"] == "1.1.0"
        # No settings override: the gpt-oss adapter's default mode is in effect.
        assert service_settings.constrained_mode is None
        assert data["constrained_mode"] == "schema"
        assert data["reasoning_effort"] == service_settings.reasoning_effort == "low"
        assert data["temperature"] == service_settings.temperature == 0.0
        assert data["ctx_size"] == FAKE_N_CTX

    async def test_a_constrained_mode_override_is_what_info_reports(
        self, service_settings, wired_llm, adapter, health_probe, ctx_size_source
    ):
        settings = dataclasses.replace(
            service_settings,
            constrained_mode="json_object",
            reasoning_effort="high",
            temperature=0.7,
        )
        engine = ExtractionEngine(llm=wired_llm, adapter=adapter, settings=settings)

        response = await _get_info(create_app(settings, engine, health_probe, ctx_size_source))

        data = response.json()
        assert data["constrained_mode"] == "json_object"
        assert data["reasoning_effort"] == "high"
        assert data["temperature"] == 0.7

    async def test_an_empty_constrained_mode_reports_the_adapter_default(
        self, service_settings, wired_llm, adapter, health_probe, ctx_size_source
    ):
        # compose.host.yml forwards EXTRACTION_CONSTRAINED_MODE=${...:-}, so an
        # unset host variable arrives as "" -- and the engine uses the adapter
        # default for it, which is what info must report.
        settings = dataclasses.replace(service_settings, constrained_mode="")
        engine = ExtractionEngine(llm=wired_llm, adapter=adapter, settings=settings)

        response = await _get_info(create_app(settings, engine, health_probe, ctx_size_source))

        assert response.json()["constrained_mode"] == "schema"
        assert engine._resolve_constrained_mode() == "schema"

    async def test_an_unknown_adapter_with_no_override_reports_a_null_constrained_mode(
        self, service_settings, health_probe, ctx_size_source
    ):
        class _UnknownAdapterEngine:
            adapter_name = "not-a-registered-adapter"

        app = create_app(
            service_settings, _UnknownAdapterEngine(), health_probe, ctx_size_source  # type: ignore[arg-type]
        )

        response = await _get_info(app)

        assert response.status_code == 200
        assert response.json()["constrained_mode"] is None

    async def test_props_is_fetched_once_across_repeated_info_calls(self, client, fake_llama_state):
        responses = [await client.get("/v1/info") for _ in range(3)]

        assert [r.json()["ctx_size"] for r in responses] == [FAKE_N_CTX] * 3
        assert fake_llama_state.props_requests == 1

    async def test_concurrent_first_info_calls_share_one_props_fetch(
        self, client, fake_llama_state
    ):
        responses = await asyncio.gather(*(client.get("/v1/info") for _ in range(3)))

        assert [r.json()["ctx_size"] for r in responses] == [FAKE_N_CTX] * 3
        assert fake_llama_state.props_requests == 1

    async def test_concurrent_calls_under_a_failing_props_retry_one_at_a_time(
        self, service_settings, engine, health_probe
    ):
        # A failure is not cached, so every caller queued on the lock makes its
        # own request -- sequentially, never overlapping, each with the timeout.
        in_flight = 0
        max_in_flight = 0
        timeouts: list[dict] = []

        async def failing(request: httpx.Request) -> httpx.Response:
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            timeouts.append(request.extensions["timeout"])
            for _ in range(5):  # yield, so an unlocked caller could overlap
                await asyncio.sleep(0)
            in_flight -= 1
            return httpx.Response(503, json={"error": "loading"})

        source = LlamaPropsContextSize(
            http_client=_props_client(failing), base_url=FAKE_BASE_URL, timeout=1.5
        )
        app = create_app(service_settings, engine, health_probe, source)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://service") as client:
            responses = await asyncio.gather(*(client.get("/v1/info") for _ in range(8)))

        assert [r.json()["ctx_size"] for r in responses] == [None] * 8
        assert len(timeouts) == 8  # one request per caller: nothing was cached
        assert max_in_flight == 1  # one at a time, under the lock
        assert all(t["read"] == 1.5 and t["connect"] == 1.5 for t in timeouts)

    async def test_props_non_2xx_gives_a_null_ctx_size(self, client, fake_llama_state):
        fake_llama_state.props_status = 503

        response = await client.get("/v1/info")

        assert response.status_code == 200
        assert response.json()["ctx_size"] is None
        assert fake_llama_state.props_requests == 1

    async def test_a_failed_props_fetch_is_not_cached(self, client, fake_llama_state):
        # llama-server still loading when info is first asked, then ready.
        fake_llama_state.props_status = 503
        first = await client.get("/v1/info")
        fake_llama_state.props_status = 200
        second = await client.get("/v1/info")
        third = await client.get("/v1/info")

        assert first.json()["ctx_size"] is None
        assert second.json()["ctx_size"] == FAKE_N_CTX
        assert third.json()["ctx_size"] == FAKE_N_CTX
        assert fake_llama_state.props_requests == 2

    @pytest.mark.parametrize(
        "body",
        [
            pytest.param({}, id="no-default-generation-settings"),
            pytest.param({"default_generation_settings": {}}, id="no-n-ctx"),
            pytest.param({"default_generation_settings": {"n_ctx": "8192"}}, id="n-ctx-string"),
            pytest.param({"default_generation_settings": {"n_ctx": True}}, id="n-ctx-bool"),
            pytest.param({"default_generation_settings": [8192]}, id="settings-not-an-object"),
            pytest.param([8192], id="body-not-an-object"),
        ],
    )
    async def test_props_without_an_int_n_ctx_gives_a_null_ctx_size(
        self, client, fake_llama_state, body
    ):
        fake_llama_state.props_body = body

        response = await client.get("/v1/info")

        assert response.status_code == 200
        assert response.json()["ctx_size"] is None

    async def test_props_down_gives_a_null_ctx_size(self, service_settings, engine, health_probe):
        calls = 0

        def refuse(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            raise httpx.ConnectError("connection refused", request=request)

        source = LlamaPropsContextSize(http_client=_props_client(refuse), base_url=FAKE_BASE_URL)

        response = await _get_info(create_app(service_settings, engine, health_probe, source))

        assert response.status_code == 200
        assert response.json()["ctx_size"] is None
        assert calls == 1

    async def test_props_non_json_body_gives_a_null_ctx_size(
        self, service_settings, engine, health_probe
    ):
        def html(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>not json</html>")

        source = LlamaPropsContextSize(http_client=_props_client(html), base_url=FAKE_BASE_URL)

        response = await _get_info(create_app(service_settings, engine, health_probe, source))

        assert response.json()["ctx_size"] is None

    async def test_props_is_read_from_the_configured_llama_base_url(
        self, service_settings, engine, health_probe
    ):
        seen: list[str] = []

        def record(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            return httpx.Response(200, json={"default_generation_settings": {"n_ctx": 4096}})

        source = LlamaPropsContextSize(
            http_client=_props_client(record), base_url=service_settings.llm_base_url
        )

        response = await _get_info(create_app(service_settings, engine, health_probe, source))

        assert response.json()["ctx_size"] == 4096
        assert seen == [f"{service_settings.llm_base_url}/props"]
