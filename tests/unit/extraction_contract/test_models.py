"""Tests for backend.extraction_contract.models.

Covers round-trip serialization for every wire model, `is_compatible`
version parsing, `error_envelope` defaulting, and the package leaf
constraint (no `backend.*` or third-party imports beyond `pydantic`).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.extraction_contract import (
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


def _error_detail() -> ErrorDetail:
    return ErrorDetail(code=ErrorCode.TIMEOUT, message="upstream timed out", retryable=True)


def _error_envelope() -> ErrorEnvelope:
    return ErrorEnvelope(error=_error_detail())


def _health_response() -> HealthResponse:
    return HealthResponse(status="ok", llm_reachable=True, uptime_s=12.3)


def _info_response() -> InfoResponse:
    return InfoResponse(
        contract_version="1.0.0",
        extraction_version="2026-06-14-r5",
        model_hash="abc123",
        model_file="gpt-oss-20b.gguf",
        llama_cpp_build="b11151",
        adapter="gpt-oss-20b",
        location_label="local",
    )


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


ROUND_TRIP_FIXTURES = [
    _history_message,
    _expect_stamps,
    _derivation_input,
    _extract_request,
    _scope_out,
    _extraction_payload,
    _derivation_out,
    _result_stamps,
    _timings_ms,
    _extract_response,
    _error_detail,
    _error_envelope,
    _health_response,
    _info_response,
    _service_status,
    _last_job,
    _cutover_status,
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
