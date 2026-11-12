"""A call that stops at its output cap leaves one log line: counts and a repetition flag.

Nothing in the service shows why a capped call ran long: the extract stage
raises before it reads the text, and llama-server logs no generated text. The
line says whether the call was looping (`repeat true`) or still progressing,
without the text itself, because production turns are real conversations.
"""

from __future__ import annotations

import dataclasses
import logging

import pytest

from backend.extraction_contract.models import ExtractRequest
from backend.extraction_service.adapters import get_adapter
from backend.extraction_service.engine import ExtractionEngine, UpstreamLLMError, repetition_flag
from backend.llm.models import LLMRequest, LLMResponse, UsageMetadata
from tests.unit.extraction_service.conftest import make_extract_request

_ENGINE_LOGGER = "backend.extraction_service.engine"
_PREFIX = "extraction length stop:"

_SCOPE_REPLY = '{"scope": "user-scope", "confidence": 0.9, "reasoning": "first person"}'
_EXTRACT_REPLY = '{"entities": [], "relationships": []}'


def _loop(period: int, length: int = 3000) -> str:
    """`length` characters of a block of `period` distinct-looking characters, repeated."""
    block = "".join(chr(ord("a") + (i * 7) % 26) for i in range(period - 1)) + "|"
    return (block * (length // period + 1))[:length]


def _non_repeating(length: int = 2000) -> str:
    """Counting text: no 200-character window occurs twice."""
    return "".join(f"{i:05d}," for i in range(length // 6 + 1))[-length:]


class TestRepetitionFlag:
    @pytest.mark.parametrize("period", [50, 600])
    def test_a_loop_is_flagged(self, period):
        assert repetition_flag(_loop(period)) is True

    def test_text_that_never_repeats_is_not(self):
        text = _non_repeating(2000)
        assert len(text) == 2000

        assert repetition_flag(text) is False

    def test_text_shorter_than_the_probe_is_not(self):
        assert repetition_flag("x" * 199) is False

    def test_only_the_tail_is_read(self):
        """A loop that ended more than `tail_chars` ago is not flagged."""
        assert repetition_flag(_loop(50, 3000) + _non_repeating(2000)) is False

    def test_the_count_is_a_threshold(self):
        # The probe (the last 200 characters) occurs twice, not three times.
        probe = _non_repeating(200)
        text = probe + _non_repeating(1000).replace(probe, "") + probe

        assert repetition_flag(text) is False
        assert repetition_flag(text, min_count=2) is True


class _ScriptedLLM:
    """A provider that answers each call with the next scripted response."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[LLMRequest] = []

    async def invoke(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return self._responses[min(len(self.requests), len(self._responses)) - 1]


def _engine(service_settings, responses, **changes) -> tuple[ExtractionEngine, _ScriptedLLM]:
    settings = dataclasses.replace(service_settings, **changes)
    adapter = get_adapter(settings.adapter_name, reasoning_effort=settings.reasoning_effort)
    llm = _ScriptedLLM(responses)
    return ExtractionEngine(llm=llm, adapter=adapter, settings=settings), llm


def _length_stops(caplog) -> list[logging.LogRecord]:
    return [
        r for r in caplog.records if r.name == _ENGINE_LOGGER and r.getMessage().startswith(_PREFIX)
    ]


# Distinctive text, so a test can show that none of it reaches the log.
_REASONING = "Considering the turn about Orrin Vale and his marmot sanctuary. " * 40
_CONTENT = '{"entities": [{"id": "orrin-vale", "type": "Person"}'


@pytest.mark.asyncio
class TestTheLengthStopLine:
    async def test_a_capped_extract_call_logs_one_line_with_its_counts(
        self, service_settings, caplog
    ):
        # A model looping in its reasoning reached the cap before any content.
        response = LLMResponse(
            content="",
            reasoning_content=_REASONING,
            finish_reason="length",
            usage=UsageMetadata(prompt_tokens=900, completion_tokens=2048, total_tokens=2948),
        )
        engine, _llm = _engine(service_settings, [response], scope_enabled=False)
        caplog.set_level(logging.WARNING, logger=_ENGINE_LOGGER)

        with pytest.raises(UpstreamLLMError):
            await engine.run(ExtractRequest(**make_extract_request()))

        records = _length_stops(caplog)
        assert [(r.levelno, r.getMessage()) for r in records] == [
            (
                logging.WARNING,
                f"{_PREFIX} cap 2048 completion_tokens 2048 "
                f"reasoning_chars {len(_REASONING)} content_chars 0 repeat true",
            )
        ]

    async def test_the_line_holds_none_of_the_response_text(self, service_settings, caplog):
        response = LLMResponse(
            content=_CONTENT,
            reasoning_content=_REASONING,
            finish_reason="length",
            usage=UsageMetadata(completion_tokens=2048),
        )
        engine, _llm = _engine(service_settings, [response], scope_enabled=False)
        caplog.set_level(logging.DEBUG, logger=_ENGINE_LOGGER)

        with pytest.raises(UpstreamLLMError):
            await engine.run(ExtractRequest(**make_extract_request()))

        (record,) = _length_stops(caplog)
        assert all(isinstance(arg, int | str) for arg in record.args)
        logged = record.getMessage() + repr(record.args)
        for fragment in ("Orrin", "Vale", "marmot", "orrin-vale", "Person", "entities"):
            assert fragment not in logged

    async def test_the_cap_names_the_stage(self, service_settings, caplog):
        """A capped scope call logs scope's cap; the job carries on."""
        responses = [
            LLMResponse(content="", reasoning_content=_non_repeating(500), finish_reason="length"),
            LLMResponse(content=_EXTRACT_REPLY, finish_reason="stop"),
        ]
        engine, _llm = _engine(service_settings, responses, reasoning_effort="medium")
        caplog.set_level(logging.WARNING, logger=_ENGINE_LOGGER)

        await engine.run(ExtractRequest(**make_extract_request()))

        assert [r.getMessage() for r in _length_stops(caplog)] == [
            f"{_PREFIX} cap 96 completion_tokens None reasoning_chars 500 content_chars 0 "
            "repeat false"
        ]

    async def test_a_call_that_stopped_on_its_own_logs_none(self, service_settings, caplog):
        responses = [
            LLMResponse(content=_SCOPE_REPLY, finish_reason="stop"),
            LLMResponse(content=_EXTRACT_REPLY, reasoning_content=_loop(50), finish_reason="stop"),
        ]
        engine, llm = _engine(service_settings, responses)
        caplog.set_level(logging.DEBUG, logger=_ENGINE_LOGGER)

        await engine.run(ExtractRequest(**make_extract_request()))

        assert len(llm.requests) == 2
        assert _length_stops(caplog) == []
