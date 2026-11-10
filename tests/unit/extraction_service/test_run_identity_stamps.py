"""The service stamps one run identity, computed at start, on `/v1/info` and every result.

`build_app` is the wiring `build_app_from_env` runs, with the LLM provider and
the HTTP client passed in; here they reach the fake llama-server, so the
identity is computed exactly as the deployed service computes it. The
identity is recorded, not keyed: requests still match on `expect`'s two
stamps alone.
"""

from __future__ import annotations

import dataclasses
import uuid
from pathlib import Path
from types import MappingProxyType, ModuleType

import httpx
import pytest

from backend.extraction_contract.models import ExpectStamps
from backend.extraction_service import app as app_module
from backend.extraction_service import engine as engine_module
from backend.extraction_service.adapters import get_adapter
from backend.extraction_service.app import build_app, create_app, run_identity_components
from backend.extraction_service.engine import (
    DERIVE_MAX_TOKENS,
    EXTRACT_MAX_TOKENS,
    RUN_IDENTITY_CODE,
    SCOPE_MAX_TOKENS,
    ExtractionEngine,
)
from backend.extraction_service.settings import ServiceSettings
from backend.knowledge.extraction.run_identity import code_sha256, compute_run_identity
from backend.knowledge.extraction.serving_identity import REQUEST_STAGES
from backend.knowledge.version_stamps import EXTRACTION_VERSION, ONTOLOGY_VERSION
from tests.unit.extraction_service.conftest import make_extract_request

_DERIVATION = {
    "signal_types": ["feedback"],
    "matched_patterns": ["feedback:stop"],
    "existing_internal_entities": "No existing internal entities.",
    "assistant_response": "Got it.",
}


def _adapter_for(settings: ServiceSettings):
    return get_adapter(
        settings.adapter_name,
        reasoning_effort=settings.reasoning_effort,
        reasoning_budget_tokens=settings.reasoning_budget_tokens,
    )


def _identity(settings: ServiceSettings, adapter=None) -> str:
    return compute_run_identity(
        run_identity_components(settings, adapter or _adapter_for(settings))
    )


async def _client_for(application) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://service"
    )


@pytest.fixture
def built_app(service_settings, wired_llm, fake_llama_app):
    """The service as `build_app_from_env` wires it, against the fake llama-server."""
    http_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=fake_llama_app))
    return build_app(service_settings, wired_llm, http_client)


# ---------------------------------------------------------------------------
# The engine constants
# ---------------------------------------------------------------------------


class TestStageTokenLimits:
    def test_the_named_limits(self):
        assert (SCOPE_MAX_TOKENS, EXTRACT_MAX_TOKENS, DERIVE_MAX_TOKENS) == (96, 2048, 400)

    @pytest.mark.asyncio
    async def test_each_stage_sends_its_named_limit(self, built_app, fake_llama_state):
        fake_llama_state.chat_responses = [
            '{"scope": "user-scope", "confidence": 0.9, "reasoning": "first person"}',
            '{"entities": [], "relationships": []}',
            '{"operations": []}',
        ]

        async with await _client_for(built_app) as client:
            response = await client.post(
                "/v1/extract", json=make_extract_request(derivation=_DERIVATION)
            )

        assert response.status_code == 200
        assert [r["max_tokens"] for r in fake_llama_state.chat_requests] == [
            SCOPE_MAX_TOKENS,
            EXTRACT_MAX_TOKENS,
            DERIVE_MAX_TOKENS,
        ]


# ---------------------------------------------------------------------------
# What the service reports
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestServiceReportsOneIdentity:
    async def test_info_reports_the_identity_and_its_components(self, built_app, service_settings):
        components = run_identity_components(service_settings, _adapter_for(service_settings))

        async with await _client_for(built_app) as client:
            data = (await client.get("/v1/info")).json()

        assert data["contract_version"] == "1.3.0"
        assert data["run_identity"] == compute_run_identity(components)
        assert data["run_identity_components"] == dataclasses.asdict(components)

    async def test_the_components_read_the_running_setup(self, built_app, service_settings):
        async with await _client_for(built_app) as client:
            components = (await client.get("/v1/info")).json()["run_identity_components"]

        assert components == {
            "prompt_sha256": engine_module._compute_prompt_sha256(_adapter_for(service_settings)),
            "code_sha256": code_sha256(RUN_IDENTITY_CODE),
            "ontology_version": ONTOLOGY_VERSION,
            "extraction_version": EXTRACTION_VERSION,
            "adapter": "gptoss",
            "model_hash": "test-model-hash",
            "llama_cpp_build": service_settings.llama_cpp_build,
            "constrained_mode": "schema",
            "reasoning_effort": "low",
            "reasoning_budget_tokens": None,
            "temperature": 0.0,
            "max_attempts": 2,
            "scope_enabled": True,
            "scope_max_tokens": 96,
            "extract_max_tokens": 2048,
            "derive_max_tokens": 400,
            "list_batch_max_units": engine_module.LIST_BATCH_MAX_UNITS,
            "list_context_max_units": engine_module.LIST_CONTEXT_MAX_UNITS,
        }

    async def test_every_result_carries_the_identity_info_reports(
        self, built_app, fake_llama_state
    ):
        fake_llama_state.chat_responses = [
            '{"scope": "user-scope", "confidence": 0.9, "reasoning": "first person"}',
            '{"entities": [], "relationships": []}',
        ]

        async with await _client_for(built_app) as client:
            identity = (await client.get("/v1/info")).json()["run_identity"]
            stamps = []
            for index in range(3):
                response = await client.post(
                    "/v1/extract",
                    json=make_extract_request(job_id=f"job-{index}", event_id=f"evt-{index}"),
                )
                assert response.status_code == 200
                stamps.append(response.json()["stamps"]["run_identity"])

        assert identity is not None
        assert stamps == [identity] * 3

    async def test_a_replayed_job_carries_the_same_identity(self, built_app):
        async with await _client_for(built_app) as client:
            first = (await client.post("/v1/extract", json=make_extract_request())).json()
            again = (await client.post("/v1/extract", json=make_extract_request())).json()
            identity = (await client.get("/v1/info")).json()["run_identity"]

        assert first["stamps"]["run_identity"] == again["stamps"]["run_identity"] == identity

    async def test_the_identity_is_not_part_of_the_epoch_check(self, built_app):
        """`expect` never asks for a run identity: the serving identity is what is keyed."""
        async with await _client_for(built_app) as client:
            response = await client.post("/v1/extract", json=make_extract_request())

        assert response.status_code == 200
        assert set(ExpectStamps.model_fields) == {
            "extraction_version",
            "model_hash",
            "serving_identity",
        }

    async def test_an_app_built_without_an_identity_reports_none(self, service_app, client):
        """`create_app` alone, as a test wires it, stamps nothing it was not given."""
        info = (await client.get("/v1/info")).json()
        result = (await client.post("/v1/extract", json=make_extract_request())).json()

        assert info["run_identity"] is None
        assert info["run_identity_components"] is None
        assert result["stamps"]["run_identity"] is None


class TestCreateAppConsistency:
    def test_create_app_refuses_an_engine_that_stamps_a_different_identity(
        self, service_settings, wired_llm, adapter, health_probe, ctx_size_source
    ):
        components = run_identity_components(service_settings, adapter)
        engine = ExtractionEngine(
            llm=wired_llm, adapter=adapter, settings=service_settings, run_identity="ri1-other"
        )

        with pytest.raises(ValueError, match="run identity"):
            create_app(
                service_settings, engine, health_probe, ctx_size_source, run_identity=components
            )


class TestBuildAppFromEnv:
    def test_it_wires_the_env_settings_through_build_app(self, monkeypatch):
        monkeypatch.setenv("EXTRACTION_MODEL_HASH", "env-model-hash")
        monkeypatch.setenv("LLAMA_CPP_BUILD", "b99999")
        seen = {}

        def spy(settings, llm, http_client):
            seen.update(settings=settings, llm=llm, http_client=http_client)
            return "app"

        monkeypatch.setattr(app_module, "build_app", spy)

        assert app_module.build_app_from_env() == "app"
        assert seen["settings"].model_hash == "env-model-hash"
        assert seen["settings"].llama_cpp_build == "b99999"
        assert isinstance(seen["llm"], app_module.LlamaServerProvider)
        assert isinstance(seen["http_client"], httpx.AsyncClient)


# ---------------------------------------------------------------------------
# Each component moves the service's identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"model_hash": "another-model"}, id="model-hash"),
        pytest.param({"llama_cpp_build": "b12000"}, id="llama-cpp-build"),
        pytest.param({"constrained_mode": "json_object"}, id="constrained-mode"),
        pytest.param({"reasoning_effort": "high"}, id="reasoning-effort"),
        pytest.param({"reasoning_budget_tokens": 256}, id="reasoning-budget"),
        pytest.param({"temperature": 0.7}, id="temperature"),
        pytest.param({"max_attempts": 3}, id="max-attempts"),
        pytest.param({"scope_enabled": False}, id="scope-enabled"),
    ],
)
def test_each_request_setting_and_deployment_value_moves_the_identity(service_settings, change):
    moved = dataclasses.replace(service_settings, **change)

    assert _identity(moved) != _identity(service_settings)


def test_the_adapter_moves_the_identity(service_settings):
    qwen = dataclasses.replace(service_settings, adapter_name="qwen")

    assert _identity(qwen) != _identity(service_settings)


def test_the_effective_constrained_mode_is_recorded(service_settings):
    """An empty override means the adapter default, as the engine reads it."""
    empty = dataclasses.replace(service_settings, constrained_mode="")
    explicit = dataclasses.replace(service_settings, constrained_mode="schema")

    assert run_identity_components(empty, _adapter_for(empty)).constrained_mode == "schema"
    assert _identity(empty) == _identity(explicit)


@pytest.mark.parametrize(
    "name",
    [
        "EXTRACTION_SYSTEM_PROMPT",
        "SCOPE_CLASSIFIER_SYSTEM_PROMPT",
        "INTERNAL_DERIVATION_SYSTEM_PROMPT",
        "_REPAIR_INSTRUCTION",
    ],
)
def test_a_prompt_edit_moves_the_identity(service_settings, monkeypatch, name):
    before = _identity(service_settings)
    monkeypatch.setattr(engine_module, name, getattr(engine_module, name) + " (edited)")

    assert _identity(service_settings) != before


@pytest.mark.parametrize(
    "name",
    [
        "scope",
        "extract",
        "derive",
        "ONTOLOGY_VERSION",
        "EXTRACTION_VERSION",
    ],
)
def test_a_limit_or_version_change_moves_the_identity(service_settings, monkeypatch, name):
    before = _identity(service_settings)
    if name in REQUEST_STAGES:
        # The caps are read from STAGE_CAPS at the effort each stage is sent at.
        low = engine_module.STAGE_CAPS["low"]
        moved = dataclasses.replace(low, **{name: getattr(low, name) + 1})
        monkeypatch.setattr(
            engine_module,
            "STAGE_CAPS",
            MappingProxyType({**engine_module.STAGE_CAPS, "low": moved}),
        )
    else:
        value = getattr(app_module, name)
        monkeypatch.setattr(app_module, name, value + 1 if isinstance(value, int) else value + "x")

    assert _identity(service_settings) != before


def _edited_copy(tmp_path: Path, module: ModuleType, source: str) -> ModuleType:
    """A module object with `module`'s name whose source file holds `source`."""
    path = tmp_path / f"m_{uuid.uuid4().hex}.py"
    path.write_text(source, encoding="utf-8")
    copy = ModuleType(module.__name__)
    copy.__file__ = str(path)
    return copy


def _with_replaced(index: int, replacement: ModuleType) -> tuple[ModuleType, ...]:
    return tuple(replacement if i == index else m for i, m in enumerate(RUN_IDENTITY_CODE))


@pytest.mark.parametrize(
    "index", range(len(RUN_IDENTITY_CODE)), ids=[m.__name__ for m in RUN_IDENTITY_CODE]
)
def test_a_code_edit_to_each_covered_module_moves_the_identity(
    service_settings, monkeypatch, tmp_path, index
):
    module = RUN_IDENTITY_CODE[index]
    source = Path(module.__file__).read_text(encoding="utf-8")
    before = _identity(service_settings)

    edited = _edited_copy(tmp_path, module, source + "\n_RUN_IDENTITY_PROBE = 1\n")
    monkeypatch.setattr(app_module, "RUN_IDENTITY_CODE", _with_replaced(index, edited))

    assert _identity(service_settings) != before


@pytest.mark.parametrize(
    "index", range(len(RUN_IDENTITY_CODE)), ids=[m.__name__ for m in RUN_IDENTITY_CODE]
)
def test_a_comment_or_docstring_edit_to_a_covered_module_does_not(
    service_settings, monkeypatch, tmp_path, index
):
    module = RUN_IDENTITY_CODE[index]
    source = Path(module.__file__).read_text(encoding="utf-8")
    before = _identity(service_settings)
    # Every covered module opens with a docstring; reword it and add a comment.
    assert source.startswith('"""')
    edited_source = '"""Reworded.\n\nAnd longer.' + source[3:] + "\n# A trailing comment.\n"

    edited = _edited_copy(tmp_path, module, edited_source)
    monkeypatch.setattr(app_module, "RUN_IDENTITY_CODE", _with_replaced(index, edited))

    assert _identity(service_settings) == before
