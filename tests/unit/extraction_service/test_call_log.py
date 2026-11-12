"""Every LLM call leaves one INFO line: its cap, token counts and finish reason.

A candidate run's prompt size cannot be counted offline (the tokenizer needs
the network inside the container), so the service reports what llama-server
counted, per call, in its own log. The line carries counts and the finish
reason only, never prompt or response text, because production turns are
real conversations. The cap names the stage, as on the length-stop line.
"""

from __future__ import annotations

import dataclasses
import logging

import pytest

from backend.extraction_contract.models import ExtractRequest
from backend.extraction_service.adapters import get_adapter
from backend.extraction_service.engine import ExtractionEngine, UpstreamLLMError
from backend.llm.models import LLMRequest, LLMResponse, UsageMetadata
from tests.unit.extraction_service.conftest import make_extract_request
from tests.unit.extraction_service.scripted_llm import ScriptedProvider

_ENGINE_LOGGER = "backend.extraction_service.engine"
_PREFIX = "extraction call:"

_DERIVATION = {
    "signal_types": ["feedback"],
    "matched_patterns": ["feedback:stop"],
    "existing_internal_entities": "No existing internal entities.",
    "assistant_response": "Got it.",
}

_SCOPE_REPLY = '{"scope": "user-scope", "confidence": 0.9, "reasoning": "first person"}'
_EXTRACT_REPLY = '{"entities": [], "relationships": []}'
_DERIVE_REPLY = '{"operations": []}'

# Distinctive text, so a test can show that none of it reaches the log.
_REASONING = "Considering the turn about Orrin Vale and his marmot sanctuary. " * 40
_CONTENT = '{"entities": [{"id": "orrin-vale", "type": "Person"}'


class _ScriptedLLM(ScriptedProvider):
    """A provider that answers each call with the next scripted response."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[LLMRequest] = []

    async def reply(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return self._responses[min(len(self.requests), len(self._responses)) - 1]


def _engine(service_settings, responses, **changes) -> tuple[ExtractionEngine, _ScriptedLLM]:
    settings = dataclasses.replace(service_settings, **changes)
    adapter = get_adapter(settings.adapter_name, reasoning_effort=settings.reasoning_effort)
    llm = _ScriptedLLM(responses)
    return ExtractionEngine(llm=llm, adapter=adapter, settings=settings), llm


def _call_lines(caplog) -> list[logging.LogRecord]:
    return [
        r for r in caplog.records if r.name == _ENGINE_LOGGER and r.getMessage().startswith(_PREFIX)
    ]


def _usage(prompt: int, completion: int) -> UsageMetadata:
    return UsageMetadata(
        prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
    )


@pytest.mark.asyncio
class TestTheCallLine:
    @pytest.mark.parametrize(
        "effort, caps",
        [
            pytest.param("low", (96, 2048, 400), id="low"),
            pytest.param("medium", (96, 32768, 400), id="medium"),
        ],
    )
    async def test_every_stage_logs_one_line_at_info(self, service_settings, caplog, effort, caps):
        responses = [
            LLMResponse(content=_SCOPE_REPLY, finish_reason="stop", usage=_usage(1200, 40)),
            LLMResponse(content=_EXTRACT_REPLY, finish_reason="stop", usage=_usage(9300, 700)),
            LLMResponse(content=_DERIVE_REPLY, finish_reason="stop", usage=_usage(800, 30)),
        ]
        engine, llm = _engine(
            service_settings, responses, scope_enabled=True, reasoning_effort=effort
        )
        caplog.set_level(logging.INFO, logger=_ENGINE_LOGGER)

        await engine.run(ExtractRequest(**make_extract_request(derivation=_DERIVATION)))

        scope_cap, extract_cap, derive_cap = caps
        assert len(llm.requests) == 3
        assert [(r.levelno, r.getMessage()) for r in _call_lines(caplog)] == [
            (
                logging.INFO,
                f"{_PREFIX} cap {scope_cap} prompt_tokens 1200 completion_tokens 40 finish stop",
            ),
            (
                logging.INFO,
                f"{_PREFIX} cap {extract_cap} prompt_tokens 9300 completion_tokens 700 "
                "finish stop",
            ),
            (
                logging.INFO,
                f"{_PREFIX} cap {derive_cap} prompt_tokens 800 completion_tokens 30 finish stop",
            ),
        ]

    async def test_absent_usage_prints_none(self, service_settings, caplog):
        responses = [LLMResponse(content=_EXTRACT_REPLY, finish_reason="stop")]
        engine, _llm = _engine(service_settings, responses, scope_enabled=False)
        caplog.set_level(logging.INFO, logger=_ENGINE_LOGGER)

        await engine.run(ExtractRequest(**make_extract_request()))

        assert [r.getMessage() for r in _call_lines(caplog)] == [
            f"{_PREFIX} cap 2048 prompt_tokens None completion_tokens None finish stop"
        ]

    async def test_a_capped_call_logs_its_line_beside_the_length_stop(
        self, service_settings, caplog
    ):
        response = LLMResponse(
            content="",
            reasoning_content=_REASONING,
            finish_reason="length",
            usage=_usage(9327, 2048),
        )
        engine, llm = _engine(service_settings, [response], scope_enabled=False)
        caplog.set_level(logging.INFO, logger=_ENGINE_LOGGER)

        with pytest.raises(UpstreamLLMError):
            await engine.run(ExtractRequest(**make_extract_request()))

        lines = [r.getMessage() for r in _call_lines(caplog)]
        assert lines == [
            f"{_PREFIX} cap 2048 prompt_tokens 9327 completion_tokens 2048 finish length"
        ] * len(llm.requests)

    async def test_each_retry_is_a_call_of_its_own(self, service_settings, caplog):
        """An unparsable reply is retried, and the retry logs its own line."""
        responses = [
            LLMResponse(content="not json", finish_reason="stop", usage=_usage(9300, 12)),
            LLMResponse(content=_EXTRACT_REPLY, finish_reason="stop", usage=_usage(9400, 20)),
        ]
        engine, llm = _engine(service_settings, responses, scope_enabled=False)
        caplog.set_level(logging.INFO, logger=_ENGINE_LOGGER)

        await engine.run(ExtractRequest(**make_extract_request()))

        assert len(llm.requests) == 2
        assert [r.getMessage() for r in _call_lines(caplog)] == [
            f"{_PREFIX} cap 2048 prompt_tokens 9300 completion_tokens 12 finish stop",
            f"{_PREFIX} cap 2048 prompt_tokens 9400 completion_tokens 20 finish stop",
        ]

    async def test_the_line_holds_none_of_the_prompt_or_response_text(
        self, service_settings, caplog
    ):
        responses = [
            LLMResponse(
                content=_CONTENT,
                reasoning_content=_REASONING,
                finish_reason="length",
                usage=_usage(9327, 2048),
            )
        ]
        engine, llm = _engine(service_settings, responses, scope_enabled=False)
        caplog.set_level(logging.DEBUG, logger=_ENGINE_LOGGER)
        request = make_extract_request(utterance="Orrin Vale keeps a marmot sanctuary in Tusk.")

        with pytest.raises(UpstreamLLMError):
            await engine.run(ExtractRequest(**request))

        records = _call_lines(caplog)
        assert records
        for record in records:
            assert all(isinstance(arg, int | str) or arg is None for arg in record.args)
            logged = record.getMessage() + repr(record.args)
            for fragment in ("Orrin", "Vale", "marmot", "Tusk", "orrin-vale", "Person"):
                assert fragment not in logged
        prompt_text = repr(llm.requests[0].messages)
        assert "Orrin Vale" in prompt_text
