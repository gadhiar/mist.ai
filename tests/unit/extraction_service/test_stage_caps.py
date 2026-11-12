"""Each stage's output token cap follows the reasoning effort that stage is sent at.

Low keeps the named limits (96, 2048, 400), so a low-effort service sends
exactly what it sent before caps depended on effort. Medium and high give
extract room to reason. Scope and derive run at low effort, and so at their
96 and 400 caps, under every setting: scope is a classification, and no
effort run measures derive above low. Qwen and Gemma send no effort, so their
caps stay low's whatever the setting says.
"""

from __future__ import annotations

import dataclasses
from types import MappingProxyType

import httpx
import pytest

from backend.extraction_contract.models import ExtractRequest
from backend.extraction_service.adapters import (
    GemmaAdapter,
    GptOssAdapter,
    QwenAdapter,
    get_adapter,
)
from backend.extraction_service.app import create_app, run_identity_components
from backend.extraction_service.engine import (
    DERIVE_MAX_TOKENS,
    EXTRACT_MAX_TOKENS,
    PROMPT_RESERVE_TOKENS,
    SCOPE_MAX_TOKENS,
    STAGE_CAPS,
    ExtractionEngine,
    StageCaps,
    UpstreamLLMError,
    stage_caps,
)
from backend.extraction_service.settings import ServiceSettings
from backend.knowledge.extraction.serving_identity import REQUEST_STAGES
from backend.llm.models import LLMRequest, LLMResponse
from tests.unit.extraction_service.conftest import make_extract_request

_DERIVATION = {
    "signal_types": ["feedback"],
    "matched_patterns": ["feedback:stop"],
    "existing_internal_entities": "No existing internal entities.",
    "assistant_response": "Got it.",
}

_REPLIES = [
    '{"scope": "user-scope", "confidence": 0.9, "reasoning": "first person"}',
    '{"entities": [], "relationships": []}',
    '{"operations": []}',
]


def _adapter_for(settings: ServiceSettings):
    return get_adapter(
        settings.adapter_name,
        reasoning_effort=settings.reasoning_effort,
        reasoning_budget_tokens=settings.reasoning_budget_tokens,
    )


async def _sent_requests(settings, wired_llm, health_probe, ctx_size_source, fake_llama_state):
    """The bodies llama-server receives for one job with every stage running."""
    settings = dataclasses.replace(settings, scope_enabled=True)
    engine = ExtractionEngine(llm=wired_llm, adapter=_adapter_for(settings), settings=settings)
    app = create_app(settings, engine, health_probe, ctx_size_source)
    fake_llama_state.chat_responses = list(_REPLIES)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://service") as client:
        response = await client.post(
            "/v1/extract", json=make_extract_request(derivation=_DERIVATION)
        )
    assert response.status_code == 200, response.text
    return fake_llama_state.chat_requests


def _sent_effort(body: dict) -> str | None:
    return (body.get("chat_template_kwargs") or {}).get("reasoning_effort")


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


class TestTheCapsTable:
    def test_low_is_the_named_limits(self):
        assert stage_caps("low") == StageCaps(scope=96, extract=2048, derive=400)
        assert stage_caps("low") == StageCaps(
            scope=SCOPE_MAX_TOKENS, extract=EXTRACT_MAX_TOKENS, derive=DERIVE_MAX_TOKENS
        )

    def test_no_effort_gives_low(self):
        assert stage_caps(None) == stage_caps("low")

    def test_medium(self):
        assert stage_caps("medium") == StageCaps(scope=1024, extract=32768, derive=400)

    def test_high_takes_medium(self):
        assert stage_caps("high") == stage_caps("medium")

    def test_the_table_is_read_only_and_names_every_effort(self):
        assert isinstance(STAGE_CAPS, MappingProxyType)
        assert set(STAGE_CAPS) == {"low", "medium", "high"}
        with pytest.raises(TypeError):
            STAGE_CAPS["low"] = StageCaps(scope=1, extract=1, derive=1)  # type: ignore[index]

    def test_stage_caps_is_frozen_with_slots(self):
        caps = stage_caps("low")

        with pytest.raises(dataclasses.FrozenInstanceError):
            caps.scope = 1  # type: ignore[misc]
        assert not hasattr(caps, "__dict__")

    def test_the_fields_are_the_request_stages(self):
        assert tuple(f.name for f in dataclasses.fields(StageCaps)) == REQUEST_STAGES

    def test_the_prompt_reserve(self):
        assert PROMPT_RESERVE_TOKENS == 14336

    def test_low_fills_a_16384_context_exactly(self):
        # The largest reserve with which low still fits the 16,384 context a
        # run beside the main stack, host-1070 and the fake llama-server use.
        assert stage_caps("low").extract + PROMPT_RESERVE_TOKENS == 16384

    def test_medium_needs_47104(self):
        assert stage_caps("medium").extract + PROMPT_RESERVE_TOKENS == 47104


# ---------------------------------------------------------------------------
# The effort each adapter applies to each stage
# ---------------------------------------------------------------------------


class TestStageEffort:
    @pytest.mark.parametrize("effort", ["low", "medium", "high"])
    def test_gpt_oss_sends_its_effort_to_extract_and_low_to_scope_and_derive(self, effort):
        adapter = GptOssAdapter(reasoning_effort=effort)

        assert [adapter.stage_effort(stage) for stage in REQUEST_STAGES] == [
            "low",
            effort,
            "low",
        ]

    @pytest.mark.parametrize("effort", ["low", "medium", "high"])
    def test_gpt_oss_prepares_each_stage_at_its_stage_effort(self, effort):
        """The budget goes on extract alone; scope and derive are sent none."""
        adapter = GptOssAdapter(reasoning_effort=effort, reasoning_budget_tokens=64)
        request = LLMRequest(messages=[])

        prepared = [adapter.prepare(request, stage=stage).thinking for stage in REQUEST_STAGES]

        assert [(t.effort, t.budget_tokens) for t in prepared] == [
            ("low", None),
            (effort, 64),
            ("low", None),
        ]

    @pytest.mark.parametrize("adapter", [QwenAdapter(), GemmaAdapter()], ids=["qwen", "gemma"])
    def test_qwen_and_gemma_apply_no_effort(self, adapter):
        assert [adapter.stage_effort(stage) for stage in REQUEST_STAGES] == [None, None, None]


# ---------------------------------------------------------------------------
# What each stage sends: the literal values, read off the wire
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestWhatEachStageSends:
    async def test_gpt_oss_at_low_sends_low_and_the_named_limits_on_every_stage(
        self, service_settings, wired_llm, health_probe, ctx_size_source, fake_llama_state
    ):
        assert service_settings.reasoning_effort == "low"

        sent = await _sent_requests(
            service_settings, wired_llm, health_probe, ctx_size_source, fake_llama_state
        )

        assert [(_sent_effort(body), body["max_tokens"]) for body in sent] == [
            ("low", 96),
            ("low", 2048),
            ("low", 400),
        ]

    @pytest.mark.parametrize("effort", ["medium", "high"])
    async def test_gpt_oss_above_low_sends_extract_at_its_effort_and_the_rest_at_low(
        self, service_settings, wired_llm, health_probe, ctx_size_source, fake_llama_state, effort
    ):
        settings = dataclasses.replace(service_settings, reasoning_effort=effort)

        sent = await _sent_requests(
            settings, wired_llm, health_probe, ctx_size_source, fake_llama_state
        )

        assert [(_sent_effort(body), body["max_tokens"]) for body in sent] == [
            ("low", 96),
            (effort, 32768),
            ("low", 400),
        ]

    @pytest.mark.parametrize("adapter_name", ["qwen", "gemma"])
    @pytest.mark.parametrize("effort", ["low", "medium", "high"])
    async def test_qwen_and_gemma_send_no_effort_and_low_caps_on_every_stage(
        self,
        service_settings,
        wired_llm,
        health_probe,
        ctx_size_source,
        fake_llama_state,
        adapter_name,
        effort,
    ):
        settings = dataclasses.replace(
            service_settings, adapter_name=adapter_name, reasoning_effort=effort
        )

        sent = await _sent_requests(
            settings, wired_llm, health_probe, ctx_size_source, fake_llama_state
        )

        assert [(_sent_effort(body), body["max_tokens"]) for body in sent] == [
            (None, 96),
            (None, 2048),
            (None, 400),
        ]


# ---------------------------------------------------------------------------
# The request profile and the run identity record the effective caps
# ---------------------------------------------------------------------------


class TestTheRecordedCaps:
    def test_the_request_profile_at_medium_has_scope_and_derive_at_low(
        self, service_settings, wired_llm
    ):
        settings = dataclasses.replace(service_settings, reasoning_effort="medium")
        engine = ExtractionEngine(llm=wired_llm, adapter=_adapter_for(settings), settings=settings)

        profiles = engine.request_profile()

        assert [(p.stage, p.thinking_effort) for p in profiles] == [
            ("scope", "low"),
            ("extract", "medium"),
            ("derive", "low"),
        ]

    @pytest.mark.parametrize(
        ("adapter_name", "effort", "caps"),
        [
            ("gptoss", "low", (96, 2048, 400)),
            ("gptoss", "medium", (96, 32768, 400)),
            ("gptoss", "high", (96, 32768, 400)),
            ("qwen", "medium", (96, 2048, 400)),
            ("gemma", "high", (96, 2048, 400)),
        ],
    )
    def test_the_run_identity_records_the_caps_each_stage_is_sent_with(
        self, service_settings, adapter_name, effort, caps
    ):
        settings = dataclasses.replace(
            service_settings, adapter_name=adapter_name, reasoning_effort=effort
        )

        components = run_identity_components(settings, _adapter_for(settings))

        assert (
            components.scope_max_tokens,
            components.extract_max_tokens,
            components.derive_max_tokens,
        ) == caps

    @pytest.mark.parametrize(("effort", "cap"), [("low", 2048), ("medium", 32768), ("high", 32768)])
    def test_the_engine_reports_each_stage_cap(self, service_settings, wired_llm, effort, cap):
        settings = dataclasses.replace(service_settings, reasoning_effort=effort)
        engine = ExtractionEngine(llm=wired_llm, adapter=_adapter_for(settings), settings=settings)

        assert engine.stage_cap("scope") == 96
        assert engine.stage_cap("extract") == cap
        assert engine.stage_cap("derive") == 400


class _LengthStopLLM:
    """A provider whose every reply stopped at its output cap."""

    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    async def invoke(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(content='{"entities": [', finish_reason="length")


@pytest.mark.asyncio
@pytest.mark.parametrize(("effort", "cap"), [("low", 2048), ("medium", 32768)])
async def test_the_truncation_error_names_the_extract_cap_sent(service_settings, effort, cap):
    settings = dataclasses.replace(service_settings, reasoning_effort=effort, scope_enabled=False)
    llm = _LengthStopLLM()
    engine = ExtractionEngine(llm=llm, adapter=_adapter_for(settings), settings=settings)

    with pytest.raises(UpstreamLLMError, match=rf"max_tokens \({cap}\) on a single unit"):
        await engine.run(ExtractRequest(**make_extract_request()))

    assert [r.max_tokens for r in llm.requests] == [cap]
