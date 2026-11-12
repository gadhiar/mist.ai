"""Tests for backend.extraction_contract.models.

Covers round-trip serialization for every wire model, `is_compatible`
version parsing, `error_envelope` defaulting, and the package leaf
constraint (no `backend.*` or third-party imports beyond `pydantic`).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.extraction_contract import (
    CONTRACT_VERSION,
    ERROR_DEFAULT_RETRYABLE,
    ERROR_HTTP_STATUS,
    CutoverStatus,
    DerivationInput,
    DerivationOut,
    ErrorCode,
    ErrorDetail,
    ErrorEnvelope,
    ExpectStamps,
    ExtractionPayload,
    ExtractionStatus,
    ExtractRequest,
    ExtractResponse,
    HealthResponse,
    HistoryMessage,
    InfoResponse,
    LastJob,
    ResultStamps,
    ScopeOut,
    ServiceStatus,
    TimingsMs,
    error_envelope,
    is_compatible,
)
from backend.extraction_contract.models import BatchRecordEntry, TurnPlanOut

# ---------------------------------------------------------------------------
# Round-trip fixtures -- one minimal-but-valid instance per model
# ---------------------------------------------------------------------------


def _history_message() -> HistoryMessage:
    return HistoryMessage(role="user", content="hello")


def _expect_stamps() -> ExpectStamps:
    return ExpectStamps(extraction_version="2026-06-14-r5", model_hash="abc123")


def _derivation_input() -> DerivationInput:
    return DerivationInput(
        signal_types=["preference"],
        matched_patterns=["likes X"],
        existing_internal_entities="MIST prefers dark mode",
        assistant_response="Noted.",
    )


def _extract_request() -> ExtractRequest:
    return ExtractRequest(
        contract_version="1.0.0",
        job_id="job-1",
        event_id="evt-1",
        turn_id="turn-1",
        request_id="req-1",
        session_id="sess-1",
        recorded_at="2026-09-26T00:00:00Z",
        turn_index=0,
        utterance="I like Python",
        conversation_history=[_history_message()],
        expect=_expect_stamps(),
        derivation=_derivation_input(),
    )


def _scope_out() -> ScopeOut:
    return ScopeOut(label="user-scope", confidence=0.9)


def _extraction_payload() -> ExtractionPayload:
    return ExtractionPayload(
        entities=[{"id": "e1", "type": "Skill", "name": "Python"}],
        relationships=[{"source": "e1", "target": "e2", "type": "USES"}],
    )


def _derivation_out() -> DerivationOut:
    return DerivationOut(operations=[{"op": "create_entity", "entity_type": "Trait"}])


def _result_stamps() -> ResultStamps:
    return ResultStamps(
        extraction_version="2026-06-14-r5",
        model_hash="abc123",
        prompt_sha256="deadbeef",
        llama_cpp_build="b11151",
        adapter="gpt-oss-20b",
    )


def _timings_ms() -> TimingsMs:
    return TimingsMs(scope=12.5, extract=340.1, derive=50.0, total=402.6)


def _extract_response() -> ExtractResponse:
    return ExtractResponse(
        contract_version="1.0.0",
        job_id="job-1",
        outcome="extracted",
        scope=_scope_out(),
        payload=_extraction_payload(),
        derivation=_derivation_out(),
        stamps=_result_stamps(),
        timings_ms=_timings_ms(),
        attempts=1,
        warnings=[],
    )


def _turn_plan_out() -> TurnPlanOut:
    return TurnPlanOut(
        payload={"entities": [], "templates": [], "back_references": []},
        prompt_tokens=812,
        completion_tokens=96,
        finish_reason="stop",
    )


def _failed_turn_plan_out() -> TurnPlanOut:
    return TurnPlanOut(payload=None, prompt_tokens=812, completion_tokens=1500)


def _extract_response_with_a_turn_plan() -> ExtractResponse:
    return _extract_response().model_copy(update={"turn_plan": _turn_plan_out()})


def _extract_response_with_a_failed_turn_plan() -> ExtractResponse:
    return _extract_response().model_copy(update={"turn_plan": _failed_turn_plan_out()})


def _error_detail() -> ErrorDetail:
    return ErrorDetail(code=ErrorCode.TIMEOUT, message="upstream timed out", retryable=True)


def _error_envelope() -> ErrorEnvelope:
    return ErrorEnvelope(error=_error_detail())


def _health_response() -> HealthResponse:
    return HealthResponse(status="ok", llm_reachable=True, uptime_s=12.3)


def _info_response() -> InfoResponse:
    return InfoResponse(
        contract_version="1.1.0",
        extraction_version="2026-06-14-r5",
        model_hash="abc123",
        model_file="gpt-oss-20b.gguf",
        llama_cpp_build="b11151",
        adapter="gpt-oss-20b",
        location_label="local",
        constrained_mode="schema",
        reasoning_effort="low",
        temperature=0.0,
        ctx_size=8192,
    )


_RUN_IDENTITY_COMPONENTS: dict[str, str | int | float | bool | None] = {
    "prompt_sha256": "p" * 64,
    "adapter": "gptoss",
    "temperature": 0.0,
    "max_attempts": 2,
    "scope_enabled": True,
    "reasoning_budget_tokens": None,
}


def _result_stamps_1_2_0() -> ResultStamps:
    return _result_stamps().model_copy(update={"run_identity": "ri1-" + "a" * 64})


def _extract_response_1_2_0() -> ExtractResponse:
    return _extract_response().model_copy(
        update={"contract_version": "1.2.0", "stamps": _result_stamps_1_2_0()}
    )


def _info_response_1_2_0() -> InfoResponse:
    return _info_response().model_copy(
        update={
            "contract_version": "1.2.0",
            "run_identity": "ri1-" + "a" * 64,
            "run_identity_components": dict(_RUN_IDENTITY_COMPONENTS),
        }
    )


_SERVING_IDENTITY = "si1-" + "b" * 64

_SERVING_IDENTITY_COMPONENTS: dict[str, str | int | float | bool | None] = {
    "adapter": "gptoss",
    "n_ctx": 16384,
    "scope_enabled": True,
    "request.scope.temperature": 0.0,
    "request.scope.thinking_budget_tokens": None,
    "server.device": "rtx4070s",
}


def _expect_stamps_1_3_0() -> ExpectStamps:
    return _expect_stamps().model_copy(update={"serving_identity": _SERVING_IDENTITY})


def _result_stamps_1_3_0() -> ResultStamps:
    return _result_stamps_1_2_0().model_copy(update={"serving_identity": _SERVING_IDENTITY})


def _info_response_1_3_0() -> InfoResponse:
    return _info_response_1_2_0().model_copy(
        update={
            "contract_version": "1.3.0",
            "serving_identity": _SERVING_IDENTITY,
            "serving_identity_components": dict(_SERVING_IDENTITY_COMPONENTS),
        }
    )


def _info_response_1_3_0_without_an_identity() -> InfoResponse:
    return _info_response_1_2_0().model_copy(
        update={
            "contract_version": "1.3.0",
            "serving_identity_error": "declared llama-server settings missing: "
            "EXTRACTION_SERVER_DEVICE",
        }
    )


# The /v1/info body a contract 1.0.0 server sends: the seven original fields
# and nothing else.
_INFO_1_0_0_PAYLOAD = {
    "contract_version": "1.0.0",
    "extraction_version": "2026-06-14-r5",
    "model_hash": "abc123",
    "model_file": "gpt-oss-20b.gguf",
    "llama_cpp_build": "b11151",
    "adapter": "gptoss",
    "location_label": "gtx1070-host",
}


def _service_status() -> ServiceStatus:
    return ServiceStatus(
        reachable=True,
        location_label="local",
        model_id="gpt-oss-20b",
        extraction_version="2026-06-14-r5",
        contract_version="1.0.0",
        last_health_ms=5,
    )


def _last_job() -> LastJob:
    return LastJob(
        event_id="evt-1",
        turn_id="turn-1",
        request_id="req-1",
        duration_ms=123.4,
        outcome="applied",
        finished_ms=1000,
    )


def _cutover_status() -> CutoverStatus:
    return CutoverStatus(
        state="filling",
        target_extraction_version="2026-07-01-r6",
        target_model_hash="def456",
        covered=3,
        total=10,
    )


def _extraction_status() -> ExtractionStatus:
    return ExtractionStatus(
        state="working",
        backlog_depth=2,
        apply_pending=1,
        dead_lettered=0,
        oldest_pending_age_ms=500,
        unrecorded_turns=0,
        legacy_unextracted=3,
        service=_service_status(),
        last_job=_last_job(),
        cutover=_cutover_status(),
    )


def _checked_cutover_status() -> CutoverStatus:
    return CutoverStatus(
        state="checked",
        target_extraction_version="2026-07-01-r6",
        target_model_hash="def456",
        covered=10,
        total=10,
    )


def _service_status_1_3_0() -> ServiceStatus:
    return _service_status().model_copy(update={"serving_identity": _SERVING_IDENTITY})


def _cutover_status_1_3_0() -> CutoverStatus:
    return _cutover_status().model_copy(update={"target_serving_identity": _SERVING_IDENTITY})


ROUND_TRIP_FIXTURES = [
    _history_message,
    _expect_stamps,
    _expect_stamps_1_3_0,
    _derivation_input,
    _extract_request,
    _scope_out,
    _extraction_payload,
    _derivation_out,
    _result_stamps,
    _result_stamps_1_2_0,
    _result_stamps_1_3_0,
    _timings_ms,
    _extract_response,
    _extract_response_1_2_0,
    _turn_plan_out,
    _failed_turn_plan_out,
    _extract_response_with_a_turn_plan,
    _extract_response_with_a_failed_turn_plan,
    _error_detail,
    _error_envelope,
    _health_response,
    _info_response,
    _info_response_1_2_0,
    _info_response_1_3_0,
    _info_response_1_3_0_without_an_identity,
    _service_status,
    _service_status_1_3_0,
    _last_job,
    _cutover_status,
    _cutover_status_1_3_0,
    _checked_cutover_status,
    _extraction_status,
]


@pytest.mark.parametrize("build", ROUND_TRIP_FIXTURES, ids=lambda f: f.__name__)
def test_round_trips_through_json_dump(build):
    instance = build()
    model_cls = type(instance)

    dumped = instance.model_dump(mode="json")
    restored = model_cls.model_validate(dumped)

    assert restored == instance


def test_extract_request_derivation_optional():
    request = _extract_request()
    request = request.model_copy(update={"derivation": None})

    restored = ExtractRequest.model_validate(request.model_dump(mode="json"))

    assert restored.derivation is None


def test_extract_response_derivation_optional():
    response = _extract_response()
    response = response.model_copy(update={"derivation": None})

    restored = ExtractResponse.model_validate(response.model_dump(mode="json"))

    assert restored.derivation is None


class TestExtractionStatus:
    def test_dumps_with_type_key(self):
        status = _extraction_status()

        dumped = status.model_dump(mode="json")

        assert dumped["type"] == "extraction_status"

    def test_default_type_field(self):
        status = ExtractionStatus(
            state="idle",
            backlog_depth=0,
            apply_pending=0,
            dead_lettered=0,
            oldest_pending_age_ms=None,
            service=_service_status(),
            last_job=None,
        )

        assert status.type == "extraction_status"
        assert status.cutover is None
        assert status.unrecorded_turns == 0
        assert status.legacy_unextracted == 0

    def test_legacy_unextracted_defaults_to_zero_when_omitted_from_the_payload(self):
        """A pre-existing payload with no `legacy_unextracted` key still validates."""
        payload = _extraction_status().model_dump(mode="json")
        del payload["legacy_unextracted"]

        restored = ExtractionStatus.model_validate(payload)

        assert restored.legacy_unextracted == 0

    def test_last_job_and_cutover_optional_none(self):
        status = ExtractionStatus(
            state="disabled",
            backlog_depth=0,
            apply_pending=0,
            dead_lettered=0,
            oldest_pending_age_ms=None,
            service=_service_status(),
            last_job=None,
            cutover=None,
        )
        dumped = status.model_dump(mode="json")
        restored = ExtractionStatus.model_validate(dumped)

        assert restored.last_job is None
        assert restored.cutover is None


class TestInfoResponseServingConfig:
    def test_a_1_0_0_payload_with_only_the_seven_original_fields_still_parses(self):
        info = InfoResponse.model_validate(_INFO_1_0_0_PAYLOAD)

        assert info.contract_version == "1.0.0"
        assert info.location_label == "gtx1070-host"
        assert info.constrained_mode is None
        assert info.reasoning_effort is None
        assert info.temperature is None
        assert info.ctx_size is None

    def test_a_1_0_0_payload_is_compatible_with_this_contract(self):
        info = InfoResponse.model_validate(_INFO_1_0_0_PAYLOAD)

        assert is_compatible(info.contract_version) is True

    def test_the_serving_config_fields_round_trip(self):
        dumped = _info_response().model_dump(mode="json")

        assert dumped["constrained_mode"] == "schema"
        assert dumped["reasoning_effort"] == "low"
        assert dumped["temperature"] == 0.0
        assert dumped["ctx_size"] == 8192

    def test_ctx_size_may_be_null_on_a_1_1_0_payload(self):
        payload = _info_response().model_dump(mode="json")
        payload["ctx_size"] = None

        assert InfoResponse.model_validate(payload).ctx_size is None


class TestContractVersion:
    def test_the_contract_is_1_4_0(self):
        assert CONTRACT_VERSION == "1.4.0"

    def test_this_contract_accepts_its_own_version(self):
        assert is_compatible(CONTRACT_VERSION) is True

    @pytest.mark.parametrize("version", ["1.2.0", "1.3.0", "1.4.0"])
    def test_an_earlier_minor_and_this_one_are_compatible(self, version):
        assert is_compatible(version) is True


class TestServingIdentityFields:
    """Contract 1.3.0 adds the serving identity, optional everywhere on the wire."""

    def test_a_1_2_0_info_validates_to_no_serving_identity(self):
        payload = _info_response_1_2_0().model_dump(mode="json")
        for name in ("serving_identity", "serving_identity_components", "serving_identity_error"):
            del payload[name]

        info = InfoResponse.model_validate(payload)

        assert info.contract_version == "1.2.0"
        assert info.serving_identity is None
        assert info.serving_identity_components is None
        assert info.serving_identity_error is None
        assert info.run_identity == "ri1-" + "a" * 64

    def test_info_gains_exactly_the_three_serving_identity_fields(self):
        fields_1_2_0 = {
            *_INFO_1_0_0_PAYLOAD,
            "constrained_mode",
            "reasoning_effort",
            "temperature",
            "ctx_size",
            "run_identity",
            "run_identity_components",
            "pack_prompt_hash",
            "pack_rails_hash",
        }

        assert set(InfoResponse.model_fields) - fields_1_2_0 == {
            "serving_identity",
            "serving_identity_components",
            "serving_identity_error",
        }

    def test_a_1_2_0_request_expects_no_serving_identity(self):
        payload = _extract_request().model_dump(mode="json")
        del payload["expect"]["serving_identity"]

        request = ExtractRequest.model_validate(payload)

        assert request.expect.serving_identity is None

    def test_a_1_2_0_result_carries_no_serving_identity(self):
        payload = _extract_response_1_2_0().model_dump(mode="json")
        del payload["stamps"]["serving_identity"]

        response = ExtractResponse.model_validate(payload)

        assert response.stamps.serving_identity is None
        assert response.stamps.run_identity == "ri1-" + "a" * 64

    def test_the_components_keep_their_json_types(self):
        dumped = _info_response_1_3_0().model_dump(mode="json")

        restored = InfoResponse.model_validate(dumped).serving_identity_components

        assert restored == _SERVING_IDENTITY_COMPONENTS
        assert {k: type(v) for k, v in restored.items()} == {
            k: type(v) for k, v in _SERVING_IDENTITY_COMPONENTS.items()
        }

    def test_a_component_that_is_not_a_scalar_is_rejected(self):
        payload = _info_response_1_3_0().model_dump(mode="json")
        payload["serving_identity_components"] = {"server": {"device": "x"}}

        with pytest.raises(ValidationError):
            InfoResponse.model_validate(payload)

    def test_the_status_fields_default_to_none(self):
        assert _service_status().serving_identity is None
        assert _cutover_status().target_serving_identity is None

    @pytest.mark.parametrize(
        "model",
        [
            ExpectStamps,
            ExtractRequest,
            ResultStamps,
            ExtractResponse,
            InfoResponse,
            ServiceStatus,
            CutoverStatus,
            ExtractionStatus,
        ],
    )
    def test_no_model_forbids_unknown_fields(self, model):
        """An older peer must ignore the fields a newer one adds."""
        assert model.model_config.get("extra") != "forbid"


class TestRunIdentityFields:
    """Contract 1.2.0 adds the run identity, optional on both sides of the wire."""

    def test_a_1_1_0_response_without_a_run_identity_parses(self):
        payload = _extract_response().model_dump(mode="json")
        payload["contract_version"] = "1.1.0"
        del payload["stamps"]["run_identity"]

        response = ExtractResponse.model_validate(payload)

        assert response.stamps.run_identity is None
        assert is_compatible(response.contract_version) is True

    def test_a_1_1_0_info_without_a_run_identity_parses(self):
        payload = _info_response().model_dump(mode="json")
        del payload["run_identity"]
        del payload["run_identity_components"]

        info = InfoResponse.model_validate(payload)

        assert info.contract_version == "1.1.0"
        assert info.run_identity is None
        assert info.run_identity_components is None

    def test_a_1_2_0_response_carries_its_run_identity(self):
        restored = ExtractResponse.model_validate(_extract_response_1_2_0().model_dump(mode="json"))

        assert restored.stamps.run_identity == "ri1-" + "a" * 64

    def test_the_components_keep_their_json_types(self):
        dumped = _info_response_1_2_0().model_dump(mode="json")

        restored = InfoResponse.model_validate(dumped).run_identity_components

        assert restored == _RUN_IDENTITY_COMPONENTS
        assert {k: type(v) for k, v in restored.items()} == {
            k: type(v) for k, v in _RUN_IDENTITY_COMPONENTS.items()
        }

    def test_a_component_that_is_not_a_scalar_is_rejected(self):
        payload = _info_response_1_2_0().model_dump(mode="json")
        payload["run_identity_components"] = {"nested": {"a": 1}}

        with pytest.raises(ValidationError):
            InfoResponse.model_validate(payload)


class TestStage2Calls:
    """Contract 1.2.0 adds the Stage 2 call count, defaulting to the one call of 1.1.0."""

    def test_a_1_1_0_response_without_stage2_calls_reports_one(self):
        payload = _extract_response().model_dump(mode="json")
        payload["contract_version"] = "1.1.0"
        del payload["stage2_calls"]

        assert ExtractResponse.model_validate(payload).stage2_calls == 1

    def test_a_1_2_0_response_carries_its_count(self):
        response = _extract_response_1_2_0().model_copy(update={"stage2_calls": 7})

        restored = ExtractResponse.model_validate(response.model_dump(mode="json"))

        assert restored.stage2_calls == 7

    def test_zero_calls_is_rejected(self):
        payload = _extract_response().model_dump(mode="json")
        payload["stage2_calls"] = 0

        with pytest.raises(ValidationError):
            ExtractResponse.model_validate(payload)


class TestTurnPlan:
    """The turn plan rides on the response only when the service made the plan call.

    A response without one serializes without the key, so a service that has
    the plan off answers byte for byte as it did before the field existed.
    """

    def test_a_response_has_no_turn_plan_by_default(self):
        assert _extract_response().turn_plan is None

    def test_an_absent_turn_plan_is_left_out_of_every_serialization(self):
        response = _extract_response()

        assert "turn_plan" not in response.model_dump()
        assert "turn_plan" not in response.model_dump(mode="json")
        assert "turn_plan" not in response.model_dump_json()

    def test_a_response_without_one_serializes_to_the_fields_it_had_before(self):
        assert list(_extract_response().model_dump(mode="json")) == [
            "contract_version",
            "job_id",
            "outcome",
            "scope",
            "payload",
            "derivation",
            "stamps",
            "timings_ms",
            "attempts",
            "stage2_calls",
            "warnings",
        ]

    def test_a_payload_without_the_key_validates_to_no_turn_plan(self):
        payload = _extract_response().model_dump(mode="json")

        assert ExtractResponse.model_validate(payload).turn_plan is None

    def test_a_turn_plan_is_serialized_with_its_four_fields(self):
        dumped = _extract_response_with_a_turn_plan().model_dump(mode="json")

        assert dumped["turn_plan"] == {
            "payload": {"entities": [], "templates": [], "back_references": []},
            "prompt_tokens": 812,
            "completion_tokens": 96,
            "finish_reason": "stop",
        }
        assert list(dumped)[-1] == "turn_plan"

    def test_a_failed_plan_call_is_serialized_with_a_null_payload(self):
        dumped = _extract_response_with_a_failed_turn_plan().model_dump(mode="json")

        assert dumped["turn_plan"] == {
            "payload": None,
            "prompt_tokens": 812,
            "completion_tokens": 1500,
            "finish_reason": None,
        }

    def test_a_turn_plan_needs_its_payload_field(self):
        """`payload` has no default: null says the call failed, and that is said, not implied."""
        with pytest.raises(ValidationError):
            TurnPlanOut.model_validate({"prompt_tokens": 1})

    def test_the_contract_version_did_not_move_for_it(self):
        # 1.4.0 is `ErrorCode.CANCELLED`'s minor, not this field's.
        assert CONTRACT_VERSION == "1.4.0"


_PROMPT_HASH = "a" * 64
_RAILS_HASH = "b" * 64


def _batch_record_entry() -> BatchRecordEntry:
    return BatchRecordEntry(
        units=[{"text": "Biscuit, a terrier", "context": ["Our pets"], "kind": "list_item"}],
        context=["Mango, a parrot"],
        payload=ExtractionPayload(entities=[{"id": "biscuit", "type": "Pet"}], relationships=[]),
        prompt_tokens=640,
        completion_tokens=52,
        finish="stop",
    )


class TestPackHashes:
    """Both pack hashes ride on the stamps and on the info response once a service reports them.

    Until then each is None and left out of the serialized model, so a result's stored
    stamps and a `/v1/info` body are byte for byte what they were before the fields.
    """

    @pytest.mark.parametrize("build", [_result_stamps, _info_response])
    def test_the_hashes_default_to_none_and_are_left_out_of_every_serialization(self, build):
        instance = build()

        assert (instance.pack_prompt_hash, instance.pack_rails_hash) == (None, None)
        for dumped in (instance.model_dump(), instance.model_dump(mode="json")):
            assert "pack_prompt_hash" not in dumped
            assert "pack_rails_hash" not in dumped
        assert "pack_" not in instance.model_dump_json()

    def test_stamps_without_the_hashes_serialize_to_the_fields_they_had_before(self):
        assert list(_result_stamps().model_dump(mode="json")) == [
            "extraction_version",
            "model_hash",
            "prompt_sha256",
            "llama_cpp_build",
            "adapter",
            "run_identity",
            "serving_identity",
        ]
        assert "pack_" not in _extract_response().model_dump_json()

    def test_an_info_response_without_the_hashes_serializes_as_before(self):
        dumped = _info_response_1_3_0().model_dump(mode="json")

        assert list(dumped)[-1] == "serving_identity_error"
        assert len(dumped) == len(InfoResponse.model_fields) - 2

    @pytest.mark.parametrize("build", [_result_stamps, _info_response])
    def test_reported_hashes_are_serialized_last_and_round_trip(self, build):
        reported = build().model_copy(
            update={"pack_prompt_hash": _PROMPT_HASH, "pack_rails_hash": _RAILS_HASH}
        )

        dumped = reported.model_dump(mode="json")
        restored = type(reported).model_validate(dumped)

        assert list(dumped)[-2:] == ["pack_prompt_hash", "pack_rails_hash"]
        assert (restored.pack_prompt_hash, restored.pack_rails_hash) == (_PROMPT_HASH, _RAILS_HASH)

    def test_one_reported_hash_is_serialized_without_the_other(self):
        dumped = _result_stamps().model_copy(update={"pack_rails_hash": _RAILS_HASH}).model_dump()

        assert dumped["pack_rails_hash"] == _RAILS_HASH
        assert "pack_prompt_hash" not in dumped

    def test_a_response_carries_the_hashes_inside_its_stamps(self):
        stamps = _result_stamps().model_copy(
            update={"pack_prompt_hash": _PROMPT_HASH, "pack_rails_hash": _RAILS_HASH}
        )
        response = _extract_response().model_copy(update={"stamps": stamps})

        restored = ExtractResponse.model_validate(response.model_dump(mode="json"))

        assert restored.stamps.pack_rails_hash == _RAILS_HASH

    def test_the_contract_version_did_not_move_for_them(self):
        # 1.4.0 is `ErrorCode.CANCELLED`'s minor, not these fields'.
        assert CONTRACT_VERSION == "1.4.0"


class TestBatchRecord:
    """The batch record rides on the response only when the service records its calls."""

    def test_a_response_has_no_batch_record_by_default_and_serializes_without_the_key(self):
        response = _extract_response()

        assert response.batch_record is None
        assert "batch_record" not in response.model_dump()
        assert "batch_record" not in response.model_dump(mode="json")
        assert "batch_record" not in response.model_dump_json()

    def test_an_entry_is_serialized_with_its_six_fields(self):
        response = _extract_response().model_copy(update={"batch_record": [_batch_record_entry()]})

        dumped = response.model_dump(mode="json")

        assert list(dumped)[-1] == "batch_record"
        assert dumped["batch_record"] == [
            {
                "units": [
                    {"text": "Biscuit, a terrier", "context": ["Our pets"], "kind": "list_item"}
                ],
                "context": ["Mango, a parrot"],
                "payload": {"entities": [{"id": "biscuit", "type": "Pet"}], "relationships": []},
                "prompt_tokens": 640,
                "completion_tokens": 52,
                "finish": "stop",
            }
        ]
        assert ExtractResponse.model_validate(dumped).batch_record == [_batch_record_entry()]

    def test_the_record_follows_the_turn_plan_when_a_response_has_both(self):
        response = _extract_response_with_a_turn_plan().model_copy(
            update={"batch_record": [_batch_record_entry()]}
        )

        assert list(response.model_dump(mode="json"))[-2:] == ["turn_plan", "batch_record"]

    def test_a_call_that_gave_no_payload_is_recorded_with_a_null_payload(self):
        entry = BatchRecordEntry(units=[], context=[], payload=None)

        assert entry.model_dump(mode="json") == {
            "units": [],
            "context": [],
            "payload": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "finish": None,
        }

    def test_an_entry_needs_its_payload_field(self):
        with pytest.raises(ValidationError):
            BatchRecordEntry.model_validate({"units": [], "context": []})

    def test_the_call_count_stays_a_number_beside_the_record(self):
        response = _extract_response().model_copy(
            update={"stage2_calls": 2, "batch_record": [_batch_record_entry()] * 2}
        )

        dumped = response.model_dump(mode="json")

        assert dumped["stage2_calls"] == 2
        assert len(dumped["batch_record"]) == 2


class TestCutoverStatusChecked:
    def test_checked_state_round_trips(self):
        status = _checked_cutover_status()

        restored = CutoverStatus.model_validate(status.model_dump(mode="json"))

        assert restored.state == "checked"

    def test_checked_state_survives_inside_extraction_status(self):
        status = _extraction_status().model_copy(update={"cutover": _checked_cutover_status()})

        restored = ExtractionStatus.model_validate(status.model_dump(mode="json"))

        assert restored.cutover is not None
        assert restored.cutover.state == "checked"


class TestScopeOutConfidenceBounds:
    def test_confidence_zero_valid(self):
        assert ScopeOut(label="unknown", confidence=0.0).confidence == 0.0

    def test_confidence_one_valid(self):
        assert ScopeOut(label="third-party", confidence=1.0).confidence == 1.0

    def test_confidence_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            ScopeOut(label="unknown", confidence=-0.01)

    def test_confidence_above_one_rejected(self):
        with pytest.raises(ValidationError):
            ScopeOut(label="unknown", confidence=1.01)

    def test_all_four_scope_labels_accepted(self):
        for label in ("user-scope", "system-scope", "third-party", "unknown"):
            assert ScopeOut(label=label, confidence=0.5).label == label


# ---------------------------------------------------------------------------
# is_compatible
# ---------------------------------------------------------------------------


class TestIsCompatible:
    def test_same_major_minor_patch(self):
        assert is_compatible("1.0.0") is True

    def test_same_major_different_minor_patch(self):
        assert is_compatible("1.4.2") is True

    def test_different_major_incompatible(self):
        assert is_compatible("2.0.0") is False

    def test_two_part_version_malformed(self):
        assert is_compatible("1.0") is False

    def test_non_numeric_malformed(self):
        assert is_compatible("x") is False

    def test_empty_string_malformed(self):
        assert is_compatible("") is False

    def test_four_part_version_malformed(self):
        assert is_compatible("1.0.0.0") is False


# ---------------------------------------------------------------------------
# ErrorCode / error_envelope
# ---------------------------------------------------------------------------


class TestErrorEnvelope:
    @pytest.mark.parametrize(
        ("code", "expected_status", "expected_retryable"),
        [
            (ErrorCode.EPOCH_MISMATCH, 409, False),
            (ErrorCode.CONTRACT_MISMATCH, 422, False),
            (ErrorCode.MODEL_LOADING, 503, True),
            (ErrorCode.UPSTREAM_LLM, 502, True),
            (ErrorCode.TIMEOUT, 504, True),
            (ErrorCode.CANCELLED, 499, True),
        ],
    )
    def test_default_status_and_retryability(self, code, expected_status, expected_retryable):
        assert ERROR_HTTP_STATUS[code] == expected_status
        assert ERROR_DEFAULT_RETRYABLE[code] == expected_retryable

        envelope = error_envelope(code, "boom")

        assert envelope.error.code == code
        assert envelope.error.retryable is expected_retryable
        assert envelope.error.message == "boom"

    def test_explicit_retryable_overrides_default(self):
        envelope = error_envelope(ErrorCode.TIMEOUT, "boom", retryable=False)

        assert envelope.error.retryable is False

    def test_cancelled_is_its_own_code_on_the_wire(self):
        """A cancelled job is told apart from a failed one by the code alone."""
        assert ErrorCode.CANCELLED.value == "cancelled"
        assert error_envelope(ErrorCode.CANCELLED, "job cancelled").model_dump(mode="json") == {
            "error": {"code": "cancelled", "message": "job cancelled", "retryable": True}
        }

    def test_no_two_codes_share_a_status(self):
        assert len(set(ERROR_HTTP_STATUS.values())) == len(ErrorCode)

    def test_every_error_code_has_a_status_and_default(self):
        for code in ErrorCode:
            assert code in ERROR_HTTP_STATUS
            assert code in ERROR_DEFAULT_RETRYABLE


# ---------------------------------------------------------------------------
# Leaf package constraint
# ---------------------------------------------------------------------------


class TestPackageIsLeaf:
    def test_no_backend_or_non_pydantic_third_party_imports(self):
        """Parse every module in backend/extraction_contract with `ast` and
        assert it imports nothing from `backend.*` and no third-party
        package other than `pydantic`.
        """
        import ast
        from pathlib import Path

        package_dir = Path(__file__).resolve().parents[3] / "backend" / "extraction_contract"
        assert package_dir.is_dir(), f"expected package dir at {package_dir}"

        allowed_third_party = {"pydantic"}

        violations: list[str] = []
        for py_file in sorted(package_dir.glob("*.py")):
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        if top == "backend":
                            violations.append(f"{py_file.name}: import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    top = module.split(".")[0]
                    if node.level and node.level > 0:
                        # Relative import within the package itself -- fine.
                        continue
                    is_disallowed_backend = top == "backend"
                    is_disallowed_third_party = (
                        top and top not in allowed_third_party and _is_third_party(top)
                    )
                    if is_disallowed_backend or is_disallowed_third_party:
                        violations.append(f"{py_file.name}: from {module} import ...")

        assert violations == [], f"leaf-package violations: {violations}"


def _is_third_party(top_module: str) -> bool:
    """Best-effort check that `top_module` is not a stdlib module.

    Uses `sys.stdlib_module_names` (Python 3.10+) as the source of truth so
    this test does not need its own hand-maintained allowlist.
    """
    import sys

    return top_module not in sys.stdlib_module_names
