"""End-to-end ASGI tests for the extraction service.

Exercises `POST /v1/extract` (and the health gate it depends on) against a
FastAPI app wired to a fake llama-server, over `httpx.ASGITransport` --
no network, no GPU. Covers the T0 brief's acceptance criteria 2-4.
"""

from __future__ import annotations

import asyncio

import pytest

from backend.knowledge.version_stamps import EXTRACTION_VERSION
from tests.unit.extraction_service.conftest import make_extract_request


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
        self, client, fake_llama_state
    ):
        fake_llama_state.delay_seconds = 0.05
        fake_llama_state.chat_responses = [
            '{"scope": "unknown", "confidence": 0.0}',
            '{"entities": [], "relationships": []}',
        ]
        body = make_extract_request(job_id="job-concurrent")

        results = await asyncio.gather(
            client.post("/v1/extract", json=body),
            client.post("/v1/extract", json=body),
        )

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

    async def test_timeout_maps_to_504(self, client, fake_llama_state, service_settings):
        fake_llama_state.delay_seconds = service_settings.llm_timeout_seconds + 1.0
        body = make_extract_request()

        response = await client.post("/v1/extract", json=body)

        assert response.status_code == 504
        assert response.json()["error"]["code"] == "timeout"

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
