"""Wire models for the backend <-> extraction-service contract.

These Pydantic models are the ONLY thing the backend dispatcher (T2) and the
extraction service (T1) agree on. Names and shapes here are fixed by the T0
brief: T1, T2, T3, and T4 all import them directly. Do not rename a field or
a model without updating every caller.

`CONTRACT_VERSION` follows MAJOR.MINOR.PATCH. A MAJOR bump means a breaking
wire change; MINOR/PATCH are additive or non-breaking. `is_compatible()`
lets either side reject a request/response pair produced by an
incompatible MAJOR version before touching its fields.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

# 1.1.0: additive -- InfoResponse gained four optional serving-config fields.
CONTRACT_VERSION = "1.1.0"
CONTRACT_MAJOR = 1

_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def is_compatible(version: str) -> bool:
    """Return True when `version` is well-formed and shares CONTRACT_MAJOR.

    `version` must parse as MAJOR.MINOR.PATCH (all-numeric parts). A
    malformed string (wrong number of parts, non-numeric parts, or empty)
    is never compatible.

    Args:
        version: A version string such as "1.0.0".

    Returns:
        True iff `version` parses as MAJOR.MINOR.PATCH and MAJOR equals
        `CONTRACT_MAJOR`.
    """
    match = _VERSION_RE.match(version)
    if not match:
        return False
    major = int(match.group(1))
    return major == CONTRACT_MAJOR


# ---------------------------------------------------------------------------
# Request side
# ---------------------------------------------------------------------------


class HistoryMessage(BaseModel):
    """One turn of conversation history sent to the service for context."""

    role: Literal["user", "assistant", "system"]
    content: str


class ExpectStamps(BaseModel):
    """Version stamps the backend expects the service's result to carry.

    `model_hash` here is the service's BARE LLM model identity -- the value
    the service's own `MIST_MODEL_HASH` holds. It is NOT the backend's
    embedding-composed stamp produced by
    `backend.knowledge.version_stamps.compose_model_hash` (which folds the
    embedding model identity in alongside the LLM identity for the
    deterministic-identity-resolver reasons that function documents). The
    backend is responsible for composing its own graph-write stamp; this
    field only carries the bare LLM identity the service is expected to
    report back in `ResultStamps.model_hash`.
    """

    extraction_version: str
    model_hash: str


class DerivationInput(BaseModel):
    """Stage 9 (internal derivation) context the service cannot read itself.

    The extraction service is stateless and has no graph access, so the
    backend gathers the signal/pattern/entity context Stage 9 needs and
    sends it along with the request.
    """

    signal_types: list[str]
    matched_patterns: list[str]
    existing_internal_entities: str
    assistant_response: str = ""


class ExtractRequest(BaseModel):
    """Request body for the extraction service's extract endpoint."""

    contract_version: str
    job_id: str
    event_id: str
    turn_id: str
    request_id: str
    session_id: str
    recorded_at: str  # ISO-8601 timestamp
    turn_index: int = 0
    utterance: str
    conversation_history: list[HistoryMessage]
    expect: ExpectStamps
    derivation: DerivationInput | None = None


# ---------------------------------------------------------------------------
# Response side
# ---------------------------------------------------------------------------

# Mirrors backend.knowledge.extraction.scope_classifier.ScopeLabel exactly.
# If the two ever diverge, the classifier's set is authoritative.
ScopeLabel = Literal["user-scope", "system-scope", "third-party", "unknown"]


class ScopeOut(BaseModel):
    """Stage 1.5 scope classification result."""

    label: ScopeLabel
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractionPayload(BaseModel):
    """Stage 2 raw extraction payload -- entities and relationships.

    Shapes are intentionally untyped dicts here: the backend's
    `ExtractionCache` stores this RAW record verbatim, and the ontology
    schema for entities/relationships lives in the backend
    (`backend.knowledge.ontologies`), not in this leaf contract package.
    """

    entities: list[dict[str, Any]]
    relationships: list[dict[str, Any]]


class DerivationOut(BaseModel):
    """Stage 9 derivation result -- internal-knowledge graph operations."""

    operations: list[dict[str, Any]]


class ResultStamps(BaseModel):
    """Version/identity stamps the service attaches to a completed result."""

    extraction_version: str
    model_hash: str
    prompt_sha256: str
    llama_cpp_build: str
    adapter: str


class TimingsMs(BaseModel):
    """Per-stage wall-clock timings for a completed extraction job."""

    scope: float | None
    extract: float
    derive: float | None
    total: float


class ExtractResponse(BaseModel):
    """Response body for a successfully completed extraction job."""

    contract_version: str
    job_id: str
    outcome: Literal["extracted"]
    scope: ScopeOut
    payload: ExtractionPayload
    derivation: DerivationOut | None
    stamps: ResultStamps
    timings_ms: TimingsMs
    attempts: int = Field(ge=1)
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ErrorCode(StrEnum):
    """Extraction service error taxonomy, mapped to HTTP status + retryability."""

    EPOCH_MISMATCH = "epoch_mismatch"
    CONTRACT_MISMATCH = "contract_mismatch"
    MODEL_LOADING = "model_loading"
    UPSTREAM_LLM = "upstream_llm"
    TIMEOUT = "timeout"


ERROR_HTTP_STATUS: dict[ErrorCode, int] = {
    ErrorCode.EPOCH_MISMATCH: 409,
    ErrorCode.CONTRACT_MISMATCH: 422,
    ErrorCode.MODEL_LOADING: 503,
    ErrorCode.UPSTREAM_LLM: 502,
    ErrorCode.TIMEOUT: 504,
}

# Default retryability per code. epoch_mismatch and contract_mismatch are
# permanent for the request as given (the caller must fix the request, not
# retry it unchanged); model_loading, upstream_llm, and timeout are
# transient conditions worth retrying.
ERROR_DEFAULT_RETRYABLE: dict[ErrorCode, bool] = {
    ErrorCode.EPOCH_MISMATCH: False,
    ErrorCode.CONTRACT_MISMATCH: False,
    ErrorCode.MODEL_LOADING: True,
    ErrorCode.UPSTREAM_LLM: True,
    ErrorCode.TIMEOUT: True,
}


class ErrorDetail(BaseModel):
    """Structured error detail carried inside an ErrorEnvelope."""

    code: ErrorCode
    message: str
    retryable: bool


class ErrorEnvelope(BaseModel):
    """Top-level error response body."""

    error: ErrorDetail


def error_envelope(code: ErrorCode, message: str, retryable: bool | None = None) -> ErrorEnvelope:
    """Build an ErrorEnvelope, applying the code's default retryability.

    Args:
        code: The ErrorCode identifying the failure.
        message: A human-readable description of the failure.
        retryable: Overrides the default retryability for `code` when given.
            When None, `ERROR_DEFAULT_RETRYABLE[code]` is used.

    Returns:
        A populated ErrorEnvelope.
    """
    if retryable is None:
        retryable = ERROR_DEFAULT_RETRYABLE[code]
    return ErrorEnvelope(error=ErrorDetail(code=code, message=message, retryable=retryable))


# ---------------------------------------------------------------------------
# Health / info
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response body for the service's health endpoint."""

    status: Literal["ok", "loading", "degraded"]
    llm_reachable: bool
    uptime_s: float


class InfoResponse(BaseModel):
    """Response body for the service's info endpoint.

    The last four fields were added in contract 1.1.0 and describe the
    serving configuration a result was produced under, so a gauntlet run can
    record what it measured. Each defaults to None: a 1.0.0 server's payload
    (the first seven fields only) still validates, and None always means
    "not reported" -- an older server, or a value the service could not read
    (`ctx_size` when llama-server's `/props` was unavailable).

    Attributes:
        constrained_mode: The constrained-decoding mode the service's LLM
            calls use ("schema", "json_object", or "none") -- the effective
            mode, after a settings override is applied to the adapter's
            default.
        reasoning_effort: The configured reasoning-effort hint ("low",
            "medium", "high"). Only the gpt-oss adapter sends it to the model.
        temperature: The sampling temperature every LLM-calling stage uses.
        ctx_size: llama-server's context window (`n_ctx`), as its `/props`
            endpoint reports it.
    """

    contract_version: str
    extraction_version: str
    model_hash: str
    model_file: str
    llama_cpp_build: str
    adapter: str
    location_label: str
    constrained_mode: str | None = None
    reasoning_effort: str | None = None
    temperature: float | None = None
    ctx_size: int | None = None


# ---------------------------------------------------------------------------
# Extraction status (WebSocket + GET /extraction/status; ADR-017 additive)
# ---------------------------------------------------------------------------


class ServiceStatus(BaseModel):
    """Extraction service reachability/identity, as observed by the backend."""

    reachable: bool
    location_label: str | None
    model_id: str | None
    extraction_version: str | None
    contract_version: str | None
    last_health_ms: int | None


class LastJob(BaseModel):
    """Summary of the most recently finished extraction job."""

    event_id: str
    turn_id: str
    request_id: str
    duration_ms: float
    outcome: Literal["applied", "skipped", "dead_lettered", "failed"]
    finished_ms: int


class CutoverStatus(BaseModel):
    """Progress of a model/version cutover in flight, when one is running."""

    state: Literal["filling", "ready", "checked", "promoted", "abandoned"]
    target_extraction_version: str
    target_model_hash: str
    covered: int
    total: int


class ExtractionStatus(BaseModel):
    """ADR-017-additive status message: WebSocket push and GET /extraction/status.

    Flat JSON with a `type` discriminator at the top level, matching the
    existing `system_status` / `health_status` message shapes in
    `backend/server.py`. `model_dump(mode="json")` on this model always
    contains `"type": "extraction_status"`.
    """

    type: Literal["extraction_status"] = "extraction_status"
    state: Literal["idle", "working", "stalled", "unreachable", "epoch_mismatch", "disabled"]
    backlog_depth: int
    apply_pending: int
    dead_lettered: int
    oldest_pending_age_ms: int | None
    unrecorded_turns: int = 0
    legacy_unextracted: int = 0
    service: ServiceStatus
    last_job: LastJob | None
    cutover: CutoverStatus | None = None
