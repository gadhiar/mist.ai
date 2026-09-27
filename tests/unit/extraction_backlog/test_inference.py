"""RemoteExtractionInference: every HTTP status and transport failure is classified.

Each case answers through `httpx.MockTransport` (in-process, no network) so one
status or exception can be pinned per test.
"""

from __future__ import annotations

import httpx
import pytest

from backend.extraction_backlog.errors import (
    InferenceResponseInvalidError,
    InferenceServiceError,
    InferenceUnreachableError,
)
from backend.extraction_backlog.inference import RemoteExtractionInference
from backend.extraction_contract.models import (
    CONTRACT_VERSION,
    ErrorCode,
    ExpectStamps,
    ExtractRequest,
    error_envelope,
)

BASE = "http://svc"

_INFO = {
    "contract_version": CONTRACT_VERSION,
    "extraction_version": "ev",
    "model_hash": "m",
    "model_file": "f.gguf",
    "llama_cpp_build": "b1",
    "adapter": "a",
    "location_label": "box",
}


def _request() -> ExtractRequest:
    return ExtractRequest(
        contract_version=CONTRACT_VERSION,
        job_id="job-1",
        event_id="evt-1",
        turn_id="s:0",
        request_id="req-1",
        session_id="s",
        recorded_at="2026-09-01T10:00:00+00:00",
        utterance="I use rust daily",
        conversation_history=[],
        expect=ExpectStamps(extraction_version="ev", model_hash="m"),
    )


def _client(handler) -> RemoteExtractionInference:
    return RemoteExtractionInference(
        httpx.AsyncClient(transport=httpx.MockTransport(handler)), BASE
    )


def _envelope(code: ErrorCode, retryable: bool | None = None) -> dict:
    return error_envelope(code, "boom", retryable).model_dump(mode="json")


class TestInfo:
    @pytest.mark.asyncio
    async def test_parses_a_valid_info_body(self):
        client = _client(lambda req: httpx.Response(200, json=_INFO))

        info = await client.info()

        assert info.model_hash == "m"
        assert info.location_label == "box"

    @pytest.mark.asyncio
    async def test_requests_the_info_path_under_the_base_url(self):
        seen = []

        def handler(req: httpx.Request) -> httpx.Response:
            seen.append(str(req.url))
            return httpx.Response(200, json=_INFO)

        await _client(handler).info()

        assert seen == [f"{BASE}/v1/info"]

    @pytest.mark.asyncio
    async def test_invalid_info_body_is_an_invalid_response(self):
        client = _client(lambda req: httpx.Response(200, json={"contract_version": "1.0.0"}))

        with pytest.raises(InferenceResponseInvalidError):
            await client.info()

    @pytest.mark.asyncio
    async def test_connect_error_is_unreachable(self):
        def handler(req):
            raise httpx.ConnectError("refused", request=req)

        with pytest.raises(InferenceUnreachableError) as excinfo:
            await _client(handler).info()

        assert excinfo.value.code is None
        assert excinfo.value.retryable is True

    @pytest.mark.asyncio
    async def test_read_timeout_on_info_is_unreachable(self):
        def handler(req):
            raise httpx.ReadTimeout("slow", request=req)

        with pytest.raises(InferenceUnreachableError):
            await _client(handler).info()


class TestHealth:
    @pytest.mark.asyncio
    async def test_parses_a_valid_health_body(self):
        body = {"status": "loading", "llm_reachable": True, "uptime_s": 3.0}
        health = await _client(lambda req: httpx.Response(200, json=body)).health()

        assert health.status == "loading"


class TestExtract:
    @pytest.mark.asyncio
    async def test_posts_the_request_as_json(self):
        seen = []

        def handler(req: httpx.Request) -> httpx.Response:
            seen.append((req.method, str(req.url), req.content))
            return httpx.Response(502, json=_envelope(ErrorCode.UPSTREAM_LLM))

        with pytest.raises(InferenceServiceError):
            await _client(handler).extract(_request())

        method, url, content = seen[0]
        assert (method, url) == ("POST", f"{BASE}/v1/extract")
        assert ExtractRequest.model_validate_json(content) == _request()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "code, retryable",
        [
            pytest.param(ErrorCode.EPOCH_MISMATCH, False, id="epoch-mismatch"),
            pytest.param(ErrorCode.CONTRACT_MISMATCH, False, id="contract-mismatch"),
            pytest.param(ErrorCode.MODEL_LOADING, True, id="model-loading"),
            pytest.param(ErrorCode.UPSTREAM_LLM, True, id="upstream-llm"),
            pytest.param(ErrorCode.TIMEOUT, True, id="timeout"),
        ],
    )
    async def test_an_error_envelope_becomes_a_service_error_with_its_code(self, code, retryable):
        from backend.extraction_contract.models import ERROR_HTTP_STATUS

        status = ERROR_HTTP_STATUS[code]
        client = _client(lambda req: httpx.Response(status, json=_envelope(code)))

        with pytest.raises(InferenceServiceError) as excinfo:
            await client.extract(_request())

        assert excinfo.value.code is code
        assert excinfo.value.retryable is retryable
        assert excinfo.value.http_status == status

    @pytest.mark.asyncio
    async def test_envelope_retryable_override_is_preserved(self):
        body = _envelope(ErrorCode.UPSTREAM_LLM, retryable=False)
        client = _client(lambda req: httpx.Response(502, json=body))

        with pytest.raises(InferenceServiceError) as excinfo:
            await client.extract(_request())

        assert excinfo.value.retryable is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [502, 503, 504])
    async def test_a_gateway_status_without_an_envelope_is_unreachable(self, status):
        client = _client(lambda req: httpx.Response(status, text="<html>bad gateway</html>"))

        with pytest.raises(InferenceUnreachableError) as excinfo:
            await client.extract(_request())

        assert excinfo.value.http_status == status

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [400, 404, 500])
    async def test_any_other_status_without_an_envelope_is_an_invalid_response(self, status):
        client = _client(lambda req: httpx.Response(status, text="Internal Server Error"))

        with pytest.raises(InferenceResponseInvalidError) as excinfo:
            await client.extract(_request())

        assert excinfo.value.http_status == status

    @pytest.mark.asyncio
    async def test_a_2xx_body_that_is_not_an_extract_response_is_invalid(self):
        client = _client(lambda req: httpx.Response(200, json={"job_id": "job-1"}))

        with pytest.raises(InferenceResponseInvalidError):
            await client.extract(_request())

    @pytest.mark.asyncio
    async def test_a_read_timeout_on_extract_is_the_jobs_timeout(self):
        def handler(req):
            raise httpx.ReadTimeout("slow", request=req)

        with pytest.raises(InferenceServiceError) as excinfo:
            await _client(handler).extract(_request())

        assert excinfo.value.code is ErrorCode.TIMEOUT
        assert excinfo.value.retryable is True

    @pytest.mark.asyncio
    async def test_a_connect_timeout_on_extract_is_unreachable(self):
        def handler(req):
            raise httpx.ConnectTimeout("no route", request=req)

        with pytest.raises(InferenceUnreachableError):
            await _client(handler).extract(_request())
