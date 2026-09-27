"""A malformed `recorded_at` is the caller's contract error, refused before any LLM call.

`ExtractRequest.recorded_at` is a plain `str` in the contract, so the model
accepts any string; the service parses it (`datetime.fromisoformat`) as the
reference date for temporal resolution. A value that does not parse is a
422 `contract_mismatch` envelope, not an unhandled 500.
"""

from __future__ import annotations

import pytest

from tests.unit.extraction_service.conftest import make_extract_request


def _assert_contract_mismatch(response) -> None:
    assert response.status_code == 422, response.text
    error = response.json()["error"]
    assert error["code"] == "contract_mismatch"
    assert error["retryable"] is False
    assert "recorded_at" in error["message"]


@pytest.mark.asyncio
class TestRecordedAt:
    @pytest.mark.parametrize(
        "recorded_at", ["yesterday", "", "2026-13-45T99:00:00", "2026-09-01T10:00:00+25:00"]
    )
    async def test_a_malformed_recorded_at_is_a_contract_mismatch_before_any_llm_call(
        self, client, fake_llama_state, recorded_at
    ):
        response = await client.post(
            "/v1/extract", json=make_extract_request(recorded_at=recorded_at)
        )

        _assert_contract_mismatch(response)
        assert fake_llama_state.chat_requests == []

    async def test_a_malformed_recorded_at_is_refused_even_while_the_model_loads(
        self, client, fake_llama_state
    ):
        fake_llama_state.health_status = 503

        response = await client.post(
            "/v1/extract", json=make_extract_request(recorded_at="not-a-time")
        )

        _assert_contract_mismatch(response)

    async def test_the_engine_refuses_it_too_when_called_directly(self, engine, fake_llama_state):
        from backend.extraction_contract.models import ExtractRequest
        from backend.extraction_service.engine import InvalidRequestError

        req = ExtractRequest(**make_extract_request(recorded_at="yesterday"))

        with pytest.raises(InvalidRequestError, match="recorded_at"):
            await engine.run(req)
        assert fake_llama_state.chat_requests == []

    async def test_a_naive_iso_timestamp_is_accepted(self, client, fake_llama_state):
        fake_llama_state.chat_responses = [
            '{"scope": "unknown", "confidence": 0.0}',
            '{"entities": [], "relationships": []}',
        ]

        response = await client.post(
            "/v1/extract", json=make_extract_request(recorded_at="2026-09-01T10:00:00")
        )

        assert response.status_code == 200, response.text
