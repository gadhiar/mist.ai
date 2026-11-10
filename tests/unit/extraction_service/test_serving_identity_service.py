"""The service computes, reports and checks its serving identity (contract 1.3.0).

The serving identity keys an extraction epoch on every serving setting that
changes extraction output (MIS-196). These tests hold the service to that:

- every `ServiceSettings` field is either a component source or excluded with
  a reason, and changing it moves the identity exactly when it is keyed;
- the request profile is what the stages really send, shown by capturing every
  request the fake llama-server receives during a job;
- `/v1/info` reports the identity, or None with the reason;
- `/v1/extract` refuses a job whose expected identity differs, and stamps the
  identity on every result.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import httpx
import pytest

from backend.extraction_service import engine as engine_module
from backend.extraction_service.adapters import get_adapter
from backend.extraction_service.app import (
    create_app,
    run_identity_components,
    serving_identity_components,
)
from backend.extraction_service.engine import (
    DERIVE_MAX_TOKENS,
    EXTRACT_MAX_TOKENS,
    SCOPE_MAX_TOKENS,
    STAGE_CAPS,
    ExtractionEngine,
)
from backend.extraction_service.settings import ServiceSettings
from backend.knowledge.extraction.run_identity import RunIdentityComponents
from backend.knowledge.extraction.serving_identity import (
    REQUEST_STAGES,
    SERVICE_SETTINGS_EXCLUDED,
    ServingIdentityComponents,
    StageRequestProfile,
    components_dict,
    compute_serving_identity,
)
from tests.unit.extraction_service.conftest import make_extract_request
from tests.unit.extraction_service.fake_llama import FAKE_N_CTX

_DERIVATION = {
    "signal_types": ["feedback"],
    "matched_patterns": ["feedback:stop"],
    "existing_internal_entities": "No existing internal entities.",
    "assistant_response": "Got it.",
}

_SCOPE_REPLY = '{"scope": "user-scope", "confidence": 0.9, "reasoning": "first person"}'
_EXTRACT_REPLY = '{"entities": [], "relationships": []}'
_DERIVE_REPLY = '{"operations": []}'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _engine_for(settings: ServiceSettings, llm) -> ExtractionEngine:
    adapter = get_adapter(
        settings.adapter_name,
        reasoning_effort=settings.reasoning_effort,
        reasoning_budget_tokens=settings.reasoning_budget_tokens,
    )
    return ExtractionEngine(llm=llm, adapter=adapter, settings=settings)


def _components(settings: ServiceSettings, llm, n_ctx: int = FAKE_N_CTX):
    return serving_identity_components(settings, _engine_for(settings, llm), n_ctx)


def _identity(settings: ServiceSettings, llm, n_ctx: int = FAKE_N_CTX) -> str:
    return compute_serving_identity(_components(settings, llm, n_ctx))


def _app_for(settings: ServiceSettings, llm, health_probe, ctx_size_source):
    return create_app(settings, _engine_for(settings, llm), health_probe, ctx_size_source)


def _client_for(application) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://service"
    )


def _without_server_key(settings: ServiceSettings, key: str) -> ServiceSettings:
    return dataclasses.replace(
        settings, server_config=tuple(kv for kv in settings.server_config if kv[0] != key)
    )


def _with_server_value(settings: ServiceSettings, key: str, value: str) -> ServiceSettings:
    kept = [kv for kv in settings.server_config if kv[0] != key]
    return dataclasses.replace(settings, server_config=tuple(sorted([*kept, (key, value)])))


# ---------------------------------------------------------------------------
# Completeness and sensitivity: every ServiceSettings field is classified
# ---------------------------------------------------------------------------

# One changed value per ServiceSettings field, applied to the `service_settings`
# fixture (the gpt-oss adapter). A field added to ServiceSettings fails
# `test_every_settings_field_has_a_mutation` until it is listed here, and the
# sensitivity test then requires it to move the identity unless it is in
# SERVICE_SETTINGS_EXCLUDED.
_MUTATIONS: dict[str, object] = {
    "llm_base_url": "http://elsewhere:8080",
    "model_hash": "another-model",
    "model_file": "another-model.gguf",
    "adapter_name": "qwen",
    "reasoning_effort": "high",
    "reasoning_budget_tokens": 256,
    "constrained_mode": "json_object",
    "location_label": "gtx1070-host",
    "llm_timeout_seconds": 99.0,
    "max_attempts": 3,
    "idempotency_cache_size": 3,
    "scope_enabled": False,
    "temperature": 0.7,
    "llama_cpp_build": "b12000",
    "port": 9999,
    "server_config": "add n_cpu_moe",
}

_SETTINGS_FIELDS = {f.name for f in dataclasses.fields(ServiceSettings)}


def _mutated(settings: ServiceSettings, name: str) -> ServiceSettings:
    if name == "server_config":
        return _with_server_value(settings, "n_cpu_moe", "13")
    return dataclasses.replace(settings, **{name: _MUTATIONS[name]})


class TestSettingsClassification:
    def test_every_settings_field_has_a_mutation(self):
        assert set(_MUTATIONS) == _SETTINGS_FIELDS, (
            "A ServiceSettings field was added or removed. Classify it: if it can change "
            "the bytes the model returns, make it a serving identity component; if it "
            "provably cannot, add it to SERVICE_SETTINGS_EXCLUDED with the reason. Then "
            "list a changed value for it in _MUTATIONS here."
        )

    def test_the_exclusions_name_settings_fields(self):
        assert set(SERVICE_SETTINGS_EXCLUDED) <= _SETTINGS_FIELDS

    def test_the_excluded_settings(self):
        """The classification table's "no" rows, as ServiceSettings fields."""
        assert set(SERVICE_SETTINGS_EXCLUDED) == {
            "model_hash",
            "llm_base_url",
            "port",
            "location_label",
            "idempotency_cache_size",
            "llm_timeout_seconds",
        }

    @pytest.mark.parametrize("name", sorted(_MUTATIONS))
    def test_a_keyed_setting_moves_the_identity_and_an_excluded_one_does_not(
        self, service_settings, wired_llm, name
    ):
        before = _identity(service_settings, wired_llm)

        after = _identity(_mutated(service_settings, name), wired_llm)

        if name in SERVICE_SETTINGS_EXCLUDED:
            assert after == before, f"{name} is excluded but moved the identity"
        else:
            assert after != before, f"{name} is keyed but did not move the identity"

    def test_the_observed_n_ctx_moves_the_identity(self, service_settings, wired_llm):
        assert _identity(service_settings, wired_llm, n_ctx=32768) != _identity(
            service_settings, wired_llm
        )

    @pytest.mark.parametrize("adapter_name", ["qwen", "gemma"])
    def test_an_effort_the_adapter_never_sends_does_not_move_the_identity(
        self, service_settings, wired_llm, adapter_name
    ):
        """The profile records what is sent: only gpt-oss sends a reasoning effort."""
        low = dataclasses.replace(service_settings, adapter_name=adapter_name)
        high = dataclasses.replace(low, reasoning_effort="high", reasoning_budget_tokens=64)

        assert _identity(high, wired_llm) == _identity(low, wired_llm)

    def test_an_empty_constrained_mode_is_the_adapter_default(self, service_settings, wired_llm):
        empty = dataclasses.replace(service_settings, constrained_mode="")
        explicit = dataclasses.replace(service_settings, constrained_mode="schema")

        assert _identity(empty, wired_llm) == _identity(explicit, wired_llm)

    def test_the_server_config_is_reported_as_declared(self, service_settings, wired_llm):
        flat = components_dict(_components(service_settings, wired_llm))

        assert {k: v for k, v in flat.items() if k.startswith("server.")} == {
            f"server.{key}": value for key, value in service_settings.server_config
        }


# ---------------------------------------------------------------------------
# Agreement with the run identity
# ---------------------------------------------------------------------------

_SHARED_FIELDS = sorted(
    {f.name for f in dataclasses.fields(RunIdentityComponents)}
    & {f.name for f in dataclasses.fields(ServingIdentityComponents)}
)

_VARIANTS = [
    pytest.param({}, id="gptoss-defaults"),
    pytest.param(
        {
            "constrained_mode": "json_object",
            "reasoning_effort": "medium",
            "reasoning_budget_tokens": 64,
            "temperature": 0.3,
            "max_attempts": 3,
            "scope_enabled": False,
            "llama_cpp_build": "b12000",
        },
        id="gptoss-overrides",
    ),
    pytest.param({"adapter_name": "qwen", "constrained_mode": "none"}, id="qwen-none"),
    pytest.param({"adapter_name": "gemma"}, id="gemma-defaults"),
]


class TestAgreementWithTheRunIdentity:
    def test_the_shared_fields(self):
        assert ["adapter", "llama_cpp_build", "max_attempts", "scope_enabled"] == _SHARED_FIELDS

    @pytest.mark.parametrize("change", _VARIANTS)
    def test_every_shared_field_has_the_same_value(self, service_settings, wired_llm, change):
        settings = dataclasses.replace(service_settings, **change)
        engine = _engine_for(settings, wired_llm)
        adapter = get_adapter(
            settings.adapter_name,
            reasoning_effort=settings.reasoning_effort,
            reasoning_budget_tokens=settings.reasoning_budget_tokens,
        )

        run = run_identity_components(settings, adapter)
        serving = serving_identity_components(settings, engine, FAKE_N_CTX)

        assert {n: getattr(serving, n) for n in _SHARED_FIELDS} == {
            n: getattr(run, n) for n in _SHARED_FIELDS
        }

    @pytest.mark.parametrize("change", _VARIANTS)
    def test_every_stage_sends_the_temperature_and_mode_the_run_identity_records(
        self, service_settings, wired_llm, change
    ):
        settings = dataclasses.replace(service_settings, **change)
        adapter = get_adapter(
            settings.adapter_name,
            reasoning_effort=settings.reasoning_effort,
            reasoning_budget_tokens=settings.reasoning_budget_tokens,
        )
        run = run_identity_components(settings, adapter)

        profiles = _engine_for(settings, wired_llm).request_profile()

        assert {(p.temperature, p.constrained) for p in profiles} == {
            (run.temperature, run.constrained_mode)
        }

    def test_gpt_oss_sends_the_effort_and_budget_the_run_identity_records(
        self, service_settings, wired_llm
    ):
        settings = dataclasses.replace(
            service_settings, reasoning_effort="medium", reasoning_budget_tokens=64
        )
        run = run_identity_components(settings, get_adapter("gptoss", **_reasoning(settings)))

        profiles = _engine_for(settings, wired_llm).request_profile()

        # Derive is sent at low effort under every setting (`GptOssAdapter.stage_effort`).
        assert {(p.stage, p.thinking_effort, p.thinking_budget_tokens) for p in profiles} == {
            ("scope", run.reasoning_effort, run.reasoning_budget_tokens),
            ("extract", run.reasoning_effort, run.reasoning_budget_tokens),
            ("derive", "low", run.reasoning_budget_tokens),
        }


def _reasoning(settings: ServiceSettings) -> dict:
    return {
        "reasoning_effort": settings.reasoning_effort,
        "reasoning_budget_tokens": settings.reasoning_budget_tokens,
    }


# ---------------------------------------------------------------------------
# The request profile is what the stages send
# ---------------------------------------------------------------------------


def _prepare_stages(source: str) -> tuple[list[str], list[str]]:
    """The `stage=` literals of every `.prepare(...)` call in `source`, and the calls without one."""
    literals: list[str] = []
    unpinned: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "prepare"
        ):
            continue
        stage = next((k.value for k in node.keywords if k.arg == "stage"), None)
        if isinstance(stage, ast.Constant) and isinstance(stage.value, str):
            literals.append(stage.value)
        else:
            unpinned.append(ast.unparse(node))
    return literals, unpinned


_ENGINE_SOURCE = Path(engine_module.__file__).read_text(encoding="utf-8")


class TestRequestStagesInTheEngine:
    def test_every_prepare_call_names_its_stage_as_a_literal(self):
        _literals, unpinned = _prepare_stages(_ENGINE_SOURCE)

        assert unpinned == [], "adapter.prepare called without a stage= string literal"

    def test_every_stage_literal_is_a_request_stage(self):
        literals, _unpinned = _prepare_stages(_ENGINE_SOURCE)

        assert literals, "the scan found no adapter.prepare call in engine.py"
        assert set(literals) == set(REQUEST_STAGES), (
            "engine.py prepares a request for a stage REQUEST_STAGES does not list, or "
            "REQUEST_STAGES lists one the engine never sends. Add the stage to "
            "REQUEST_STAGES (backend/knowledge/extraction/serving_identity.py) and to "
            "ExtractionEngine.request_profile."
        )

    def test_the_scan_sees_a_new_stage(self):
        source = _ENGINE_SOURCE + '\nadapter.prepare(request, stage="rerank")\n'

        literals, _unpinned = _prepare_stages(source)

        assert set(literals) - set(REQUEST_STAGES) == {"rerank"}

    def test_the_scan_sees_a_stage_that_is_not_a_literal(self):
        literals, unpinned = _prepare_stages("adapter.prepare(request, stage=name)\n")

        assert (literals, unpinned) == ([], ["adapter.prepare(request, stage=name)"])

    def test_the_profile_has_one_entry_per_stage_in_order(self, engine):
        profiles = engine.request_profile()

        assert all(isinstance(p, StageRequestProfile) for p in profiles)
        assert tuple(p.stage for p in profiles) == REQUEST_STAGES


_STAGE_BY_MAX_TOKENS = {
    SCOPE_MAX_TOKENS: "scope",
    EXTRACT_MAX_TOKENS: "extract",
    DERIVE_MAX_TOKENS: "derive",
    STAGE_CAPS["medium"].scope: "scope",
    STAGE_CAPS["medium"].extract: "extract",
}
_CONSTRAINED_BY_FORMAT = {"json_schema": "schema", "json_object": "json_object", None: "none"}


def _sent_profile(body: dict) -> dict:
    """The profile fields of one captured request body, as llama-server received it."""
    return {
        "temperature": body["temperature"],
        "top_p": body["top_p"],
        "constrained": _CONSTRAINED_BY_FORMAT[(body.get("response_format") or {}).get("type")],
        "chat_template_kwargs": body.get("chat_template_kwargs"),
        "reasoning_budget_tokens": body.get("reasoning_budget_tokens"),
    }


def _expected_profile(profile: StageRequestProfile) -> dict:
    kwargs = {}
    if profile.thinking_effort is not None:
        kwargs["reasoning_effort"] = profile.thinking_effort
    if profile.thinking_enabled is not None:
        kwargs["enable_thinking"] = profile.thinking_enabled
    return {
        "temperature": profile.temperature,
        "top_p": profile.top_p,
        "constrained": profile.constrained,
        "chat_template_kwargs": kwargs or None,
        "reasoning_budget_tokens": profile.thinking_budget_tokens,
    }


@pytest.mark.asyncio
class TestCapturedRequestsMatchTheProfile:
    @pytest.mark.parametrize("change", _VARIANTS)
    async def test_every_request_of_a_job_matches_its_stage_profile(
        self, service_settings, wired_llm, health_probe, ctx_size_source, fake_llama_state, change
    ):
        settings = dataclasses.replace(service_settings, **{**change, "scope_enabled": True})
        engine = _engine_for(settings, wired_llm)
        app = create_app(settings, engine, health_probe, ctx_size_source)
        # A repair retry, so Stage 2 sends twice: both must match the profile.
        fake_llama_state.chat_responses = [
            _SCOPE_REPLY,
            "not json",
            _EXTRACT_REPLY,
            _DERIVE_REPLY,
        ]

        async with _client_for(app) as client:
            response = await client.post(
                "/v1/extract", json=make_extract_request(derivation=_DERIVATION)
            )

        assert response.status_code == 200, response.text
        profiles = {p.stage: p for p in engine.request_profile()}
        sent = [
            (_STAGE_BY_MAX_TOKENS[body["max_tokens"]], body)
            for body in fake_llama_state.chat_requests
        ]
        assert [stage for stage, _body in sent] == ["scope", "extract", "extract", "derive"]
        for stage, body in sent:
            assert _sent_profile(body) == _expected_profile(profiles[stage]), stage


# ---------------------------------------------------------------------------
# /v1/info
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestInfoReportsTheServingIdentity:
    async def test_the_identity_its_components_and_no_error(self, client, service_settings, engine):
        components = serving_identity_components(service_settings, engine, FAKE_N_CTX)

        data = (await client.get("/v1/info")).json()

        assert data["contract_version"] == "1.3.0"
        assert data["serving_identity"] == compute_serving_identity(components)
        assert data["serving_identity_components"] == components_dict(components)
        assert data["serving_identity_error"] is None

    @pytest.mark.parametrize("key", ["image", "device", "ctx_size"])
    @pytest.mark.parametrize("how", ["absent", "empty"])
    async def test_a_missing_required_declaration_gives_no_identity(
        self, service_settings, wired_llm, health_probe, ctx_size_source, key, how
    ):
        settings = (
            _without_server_key(service_settings, key)
            if how == "absent"
            else _with_server_value(service_settings, key, "")
        )

        async with _client_for(
            _app_for(settings, wired_llm, health_probe, ctx_size_source)
        ) as client:
            data = (await client.get("/v1/info")).json()

        assert data["serving_identity"] is None
        assert f"EXTRACTION_SERVER_{key.upper()}" in data["serving_identity_error"]

    async def test_unreadable_props_gives_no_identity(self, client, fake_llama_state):
        fake_llama_state.props_status = 503

        data = (await client.get("/v1/info")).json()

        assert data["serving_identity"] is None
        assert data["serving_identity_components"] is None
        assert "/props" in data["serving_identity_error"]

    async def test_a_context_size_that_differs_from_the_declared_one_gives_no_identity(
        self, client, fake_llama_state
    ):
        fake_llama_state.props_body = {"default_generation_settings": {"n_ctx": 4096}}

        data = (await client.get("/v1/info")).json()

        assert data["serving_identity"] is None
        assert data["serving_identity_error"] == (
            f"declared ctx_size {FAKE_N_CTX} != observed n_ctx 4096"
        )
        assert data["serving_identity_components"]["n_ctx"] == 4096

    async def test_a_declared_context_size_that_is_not_an_integer_gives_no_identity(
        self, service_settings, wired_llm, health_probe, ctx_size_source
    ):
        settings = _with_server_value(service_settings, "ctx_size", "16k")

        async with _client_for(
            _app_for(settings, wired_llm, health_probe, ctx_size_source)
        ) as client:
            data = (await client.get("/v1/info")).json()

        assert data["serving_identity"] is None
        assert "16k" in data["serving_identity_error"]

    async def test_a_llama_server_restart_with_another_context_is_seen_at_once(
        self, client, fake_llama_state
    ):
        first = (await client.get("/v1/info")).json()
        fake_llama_state.props_body = {"default_generation_settings": {"n_ctx": 32768}}
        second = (await client.get("/v1/info")).json()

        assert first["serving_identity"] is not None
        assert second["serving_identity"] is None
        assert second["serving_identity_components"]["n_ctx"] == 32768
        assert fake_llama_state.props_requests == 2


# ---------------------------------------------------------------------------
# /v1/extract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestExtractChecksTheServingIdentity:
    async def test_a_matching_expected_identity_is_served_and_stamped(self, client):
        identity = (await client.get("/v1/info")).json()["serving_identity"]

        response = await client.post(
            "/v1/extract", json=make_extract_request(serving_identity=identity)
        )

        assert response.status_code == 200, response.text
        assert response.json()["stamps"]["serving_identity"] == identity

    async def test_a_request_without_an_expected_identity_is_served_as_before(self, client):
        """A pre-1.3.0 backend sends no `expect.serving_identity`: no check, still stamped."""
        identity = (await client.get("/v1/info")).json()["serving_identity"]
        body = make_extract_request()
        assert "serving_identity" not in body["expect"]

        response = await client.post("/v1/extract", json=body)

        assert response.status_code == 200, response.text
        assert response.json()["stamps"]["serving_identity"] == identity

    async def test_a_mismatched_identity_is_a_409_naming_the_settings_that_differ(
        self, client, fake_llama_state
    ):
        expected = (await client.get("/v1/info")).json()["serving_identity"]
        fake_llama_state.props_body = {"default_generation_settings": {"n_ctx": 32768}}

        response = await client.post(
            "/v1/extract", json=make_extract_request(serving_identity=expected)
        )

        assert response.status_code == 409
        error = response.json()["error"]
        assert error["code"] == "epoch_mismatch"
        assert error["retryable"] is False
        assert f"n_ctx: {FAKE_N_CTX} != 32768" in error["message"]
        assert expected in error["message"]
        assert fake_llama_state.chat_requests == []

    async def test_an_identity_this_process_never_computed_lists_every_component(
        self, client, fake_llama_state
    ):
        unknown = "si1-" + "0" * 64

        response = await client.post(
            "/v1/extract", json=make_extract_request(serving_identity=unknown)
        )

        assert response.status_code == 409
        message = response.json()["error"]["message"]
        assert 'adapter: <missing> != "gptoss"' in message
        assert f"n_ctx: <missing> != {FAKE_N_CTX}" in message
        assert fake_llama_state.chat_requests == []

    async def test_an_expected_identity_against_a_service_without_one_is_a_409(
        self, service_settings, wired_llm, health_probe, ctx_size_source, fake_llama_state
    ):
        settings = _without_server_key(service_settings, "device")
        app = _app_for(settings, wired_llm, health_probe, ctx_size_source)

        async with _client_for(app) as client:
            response = await client.post(
                "/v1/extract", json=make_extract_request(serving_identity="si1-" + "0" * 64)
            )

        assert response.status_code == 409
        assert "EXTRACTION_SERVER_DEVICE" in response.json()["error"]["message"]
        assert fake_llama_state.chat_requests == []

    @pytest.mark.parametrize("expected", [None, "si1-" + "0" * 64])
    async def test_unreadable_props_is_a_retryable_503_with_no_llm_call(
        self, client, fake_llama_state, expected
    ):
        fake_llama_state.props_status = 503

        response = await client.post(
            "/v1/extract", json=make_extract_request(serving_identity=expected)
        )

        assert response.status_code == 503
        error = response.json()["error"]
        assert error["code"] == "model_loading"
        assert error["retryable"] is True
        assert "/props" in error["message"]
        assert fake_llama_state.chat_requests == []

    async def test_props_is_read_on_every_extract_call(self, client, fake_llama_state):
        for index in range(3):
            response = await client.post(
                "/v1/extract",
                json=make_extract_request(job_id=f"job-{index}", event_id=f"evt-{index}"),
            )
            assert response.status_code == 200

        assert fake_llama_state.props_requests == 3

    async def test_a_service_without_an_identity_still_serves_a_pre_1_3_0_request(
        self, service_settings, wired_llm, health_probe, ctx_size_source
    ):
        settings = _without_server_key(service_settings, "device")
        app = _app_for(settings, wired_llm, health_probe, ctx_size_source)

        async with _client_for(app) as client:
            response = await client.post("/v1/extract", json=make_extract_request())

        assert response.status_code == 200, response.text
        assert response.json()["stamps"]["serving_identity"] is None
