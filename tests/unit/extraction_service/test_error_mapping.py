"""Every `/v1/extract` failure leaves the service as an error envelope, never a bare 500.

Covers wrong item shapes in the model's JSON (Stage 2 goes through the repair
retry and then 502 `upstream_llm`; Stage 9 degrades to `derivation=None`) and
a pydantic `ValidationError` raised while the response is built (502, as
defence in depth).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.extraction_contract.models import ExtractionPayload, ExtractRequest
from backend.extraction_service.app import create_app
from tests.unit.extraction_service.conftest import make_extract_request

_SCOPE = '{"scope": "unknown", "confidence": 0.0}'
_VALID = '{"entities": [{"id": "rust", "name": "Rust", "type": "Technology"}], "relationships": []}'
_DERIVATION = {
    "signal_types": ["feedback"],
    "matched_patterns": ["feedback:stop"],
    "existing_internal_entities": "No existing internal entities.",
    "assistant_response": "Got it.",
}

_WRONG_SHAPES = [
    pytest.param('{"entities": ["Alice"], "relationships": []}', id="entity-is-a-string"),
    pytest.param('{"entities": [], "relationships": [1]}', id="relationship-is-a-number"),
    pytest.param('{"entities": [[]], "relationships": []}', id="entity-is-a-list"),
    pytest.param('{"entities": [null], "relationships": []}', id="entity-is-null"),
]


def _assert_envelope(response, status: int, code: str) -> None:
    assert response.status_code == status, response.text
    body = response.json()
    assert body["error"]["code"] == code
    assert isinstance(body["error"]["message"], str) and body["error"]["message"]


@pytest.mark.asyncio
class TestStage2ItemShapes:
    @pytest.mark.parametrize("wrong", _WRONG_SHAPES)
    async def test_a_wrong_item_shape_goes_into_the_repair_retry(
        self, client, fake_llama_state, wrong
    ):
        fake_llama_state.chat_responses = [_SCOPE, wrong, _VALID]

        response = await client.post("/v1/extract", json=make_extract_request())

        assert response.status_code == 200, response.text
        data = response.json()
        assert data["attempts"] == 2
        assert [e["id"] for e in data["payload"]["entities"]] == ["rust"]
        # The repair turn carried the rejected output back to the model.
        repair_messages = fake_llama_state.chat_requests[2]["messages"]
        assert repair_messages[-2] == {"role": "assistant", "content": wrong}

    @pytest.mark.parametrize("wrong", _WRONG_SHAPES)
    async def test_a_wrong_item_shape_on_every_attempt_is_a_502_envelope(
        self, client, fake_llama_state, wrong
    ):
        fake_llama_state.chat_responses = [_SCOPE, wrong]

        response = await client.post("/v1/extract", json=make_extract_request())

        _assert_envelope(response, 502, "upstream_llm")
        assert len(fake_llama_state.chat_requests) == 3  # scope + max_attempts=2


@pytest.mark.asyncio
class TestStage9ItemShapes:
    @pytest.mark.parametrize(
        "wrong",
        [
            pytest.param('{"operations": ["x"]}', id="operation-is-a-string"),
            pytest.param('{"operations": [3, {"op": "noop"}]}', id="one-operation-is-a-number"),
        ],
    )
    async def test_a_wrong_operation_shape_degrades_to_no_derivation(
        self, client, fake_llama_state, wrong
    ):
        fake_llama_state.chat_responses = [_SCOPE, _VALID, wrong]

        response = await client.post(
            "/v1/extract", json=make_extract_request(derivation=_DERIVATION)
        )

        assert response.status_code == 200, response.text
        data = response.json()
        assert data["derivation"] is None
        assert "derivation_failed" in data["warnings"]
        assert [e["id"] for e in data["payload"]["entities"]] == ["rust"]


class _EngineRaising:
    """An engine whose run raises what building a response can raise."""

    adapter_name = "fake"

    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.runs = 0

    async def run(self, req: ExtractRequest):
        self.runs += 1
        raise self._exc


def _a_validation_error() -> ValidationError:
    try:
        ExtractionPayload(entities=["Alice"], relationships=[])  # type: ignore[list-item]
    except ValidationError as exc:
        return exc
    raise AssertionError("ExtractionPayload accepted a string entity")


@pytest.mark.asyncio
class TestResponseBuildFailures:
    async def test_a_validation_error_while_building_the_response_is_a_502_envelope(
        self, service_settings, health_probe, ctx_size_source
    ):
        import httpx

        engine = _EngineRaising(_a_validation_error())
        app = create_app(
            service_settings, engine, health_probe, ctx_size_source  # type: ignore[arg-type]
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://service") as client:
            response = await client.post("/v1/extract", json=make_extract_request())

        _assert_envelope(response, 502, "upstream_llm")
        assert engine.runs == 1
