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

from pydantic import BaseModel, Field, SerializerFunctionWrapHandler, model_serializer

# 1.1.0: additive -- InfoResponse gained four optional serving-config fields.
# 1.2.0: additive -- ResultStamps.run_identity, and InfoResponse.run_identity
# with its components, all optional; ExtractResponse.stage2_calls, default 1.
# 1.3.0: additive -- the serving identity: ExpectStamps.serving_identity,
# ResultStamps.serving_identity, InfoResponse.serving_identity with its
# components and error, ServiceStatus.serving_identity and
# CutoverStatus.target_serving_identity, all optional.
# 1.4.0: additive -- the one minor above 1.3.0 for T42's, T43's and T49's
# fields together. T49's: ErrorCode.CANCELLED, the code a request waiting on a
# cancelled job is answered with, and CancelResponse with its CancelOutcome,
# the reply of `POST /v1/jobs/{job_id}/cancel`. T42's turn plan fields and
# T43's pack hashes, below, are covered by it, though none of them alone moved
# the version.
# Not a version: ExtractResponse.turn_plan is optional and left out of the
# serialized response when None, which it is unless the service runs with its
# off-by-default turn plan flag. A response without the flag is unchanged, so
# the version stays; it moves when the plan ships.
# Not a version either, for the same reason: ExtractResponse.batch_record, and
# pack_prompt_hash and pack_rails_hash on ResultStamps and InfoResponse. Each is
# optional and left out of the serialized model when None, which it is until
# the service reports it.
CONTRACT_VERSION = "1.4.0"
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

    Attributes:
        serving_identity: The serving identity the epoch is bound to (contract
            1.3.0). The service refuses the job with `epoch_mismatch` when its
            own identity differs. None asks for no check, which is what a
            pre-1.3.0 backend sends.
    """

    extraction_version: str
    model_hash: str
    serving_identity: str | None = None


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


_PACK_HASH_FIELDS: tuple[str, ...] = ("pack_prompt_hash", "pack_rails_hash")


def _without_absent(data: dict[str, Any], model: BaseModel, names: tuple[str, ...]) -> dict:
    """Drop from a serialized model each of `names` whose field is None.

    A field that a service does not report yet is left out of the wire form, so the
    model serializes byte for byte as it did before the field existed.
    """
    for name in names:
        if getattr(model, name) is None:
            data.pop(name, None)
    return data


class ResultStamps(BaseModel):
    """Version/identity stamps the service attaches to a completed result.

    Attributes:
        run_identity: The run identity of the setup that produced this result
            (contract 1.2.0): one hash over the prompt, the code that shapes
            extraction output, the request settings, the versions, the
            adapter, the model and the llama.cpp build. It is recorded, not
            matched: no epoch or cache key reads it. None means not reported,
            by a pre-1.2.0 service or one wired without it.
        serving_identity: The serving identity the service computed for this
            job (contract 1.3.0): one hash over every serving setting that
            changes extraction output, which an epoch is keyed on. None means
            not reported, by a pre-1.3.0 service, or one that could not compute
            it (see `InfoResponse.serving_identity_error`).
        pack_prompt_hash: The ontology pack's prompt hash: sha256 over what the
            pack feeds the model. None means not reported, and the serialized
            stamps then have no such key.
        pack_rails_hash: The pack's rails hash: sha256 over the rules that run
            on a payload after the model. None and its serialization likewise.
    """

    extraction_version: str
    model_hash: str
    prompt_sha256: str
    llama_cpp_build: str
    adapter: str
    run_identity: str | None = None
    serving_identity: str | None = None
    pack_prompt_hash: str | None = None
    pack_rails_hash: str | None = None

    @model_serializer(mode="wrap")
    def _omit_absent_pack_hashes(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Serialize as usual, then drop each pack hash that is not reported."""
        return _without_absent(handler(self), self, _PACK_HASH_FIELDS)


class TimingsMs(BaseModel):
    """Per-stage wall-clock timings for a completed extraction job."""

    scope: float | None
    extract: float
    derive: float | None
    total: float


class TurnPlanOut(BaseModel):
    """The record of one turn's plan call: what it returned and what it cost.

    A service running with its turn plan flag makes one plan call per
    structured turn, before the Stage 2 batches, and this is that call as the
    raw record keeps it.

    Attributes:
        payload: The plan the call returned, as the service parsed it. None
            when the call failed, stopped at its output cap or returned a
            plan the service rejected; the batches then ran without a plan
            and `ExtractResponse.warnings` holds `turn_plan_failed`.
        prompt_tokens: The prompt tokens llama-server counted for the call.
            None when the call returned no usage, or did not return.
        completion_tokens: The completion tokens it counted; None likewise.
        finish_reason: Why the call stopped ("stop", "length"). None when the
            call did not return: a transport failure or a timeout.
    """

    payload: dict[str, Any] | None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None


class BatchRecordEntry(BaseModel):
    """The record of one Stage 2 call: what the model was given and what it returned.

    A turn's batch record holds one entry per call, in call order, so which
    batch an entity came from is logged, not inferred.

    Attributes:
        units: The batch's own units, each as the structure splitter gave it:
            its `text`, its `context` (the headings, lead-in and parent items
            above it) and its `kind`.
        context: The earlier units of the same message the call was given for
            reference, as rendered; empty when it was given none.
        payload: The entities and relationships the call returned, as the
            service parsed them. None when the call gave no usable payload.
        prompt_tokens: The prompt tokens llama-server counted for the call.
            None when the call returned no usage, or did not return.
        completion_tokens: The completion tokens it counted; None likewise.
        finish: Why the call stopped ("stop", "length"). None when the call
            did not return.
    """

    units: list[dict[str, Any]]
    context: list[str]
    payload: ExtractionPayload | None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish: str | None = None


class ExtractResponse(BaseModel):
    """Response body for a successfully completed extraction job.

    Attributes:
        attempts: The most attempts any one Stage 2 call sequence took: 1, or
            more when a repair retry ran. The plan call is not an attempt.
        stage2_calls: Every Stage 2 model call the turn took (contract 1.2.0):
            one per batch of units, plus splits and repair retries. Defaults
            to 1, the one call a pre-1.2.0 service makes. The plan call is not
            counted, so the count means the same with the plan on or off.
        turn_plan: The turn's plan call, when the service runs with its turn
            plan flag and the turn is structured. None otherwise, and then
            the serialized response has no `turn_plan` key at all, so a
            response from a service without the flag is byte-for-byte the
            response it gave before the field existed. The plan call's time
            is part of `timings_ms.total` and of no stage's own figure:
            `timings_ms.extract` stays the time of the batch calls.
        batch_record: One entry per Stage 2 call, in call order, when the
            service records them. None otherwise, and then the serialized
            response has no `batch_record` key, as with `turn_plan`.
            `stage2_calls` stays the count; this is the calls themselves.
    """

    contract_version: str
    job_id: str
    outcome: Literal["extracted"]
    scope: ScopeOut
    payload: ExtractionPayload
    derivation: DerivationOut | None
    stamps: ResultStamps
    timings_ms: TimingsMs
    attempts: int = Field(ge=1)
    stage2_calls: int = Field(default=1, ge=1)
    warnings: list[str] = Field(default_factory=list)
    turn_plan: TurnPlanOut | None = None
    batch_record: list[BatchRecordEntry] | None = None

    @model_serializer(mode="wrap")
    def _omit_the_absent_records(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Serialize as usual, then drop `turn_plan` and `batch_record` when there is none."""
        return _without_absent(handler(self), self, ("turn_plan", "batch_record"))


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
    CANCELLED = "cancelled"


# cancelled takes 499, a status no standard assigns, so that no two codes
# share one: 409, the nearest standard status, is epoch_mismatch's.
ERROR_HTTP_STATUS: dict[ErrorCode, int] = {
    ErrorCode.EPOCH_MISMATCH: 409,
    ErrorCode.CONTRACT_MISMATCH: 422,
    ErrorCode.MODEL_LOADING: 503,
    ErrorCode.UPSTREAM_LLM: 502,
    ErrorCode.TIMEOUT: 504,
    ErrorCode.CANCELLED: 499,
}

# Default retryability per code. epoch_mismatch and contract_mismatch are
# permanent for the request as given (the caller must fix the request, not
# retry it unchanged); model_loading, upstream_llm, and timeout are
# transient conditions worth retrying. cancelled is retryable because the
# service stores nothing for a cancelled job: the same job_id, sent again
# unchanged, runs.
ERROR_DEFAULT_RETRYABLE: dict[ErrorCode, bool] = {
    ErrorCode.EPOCH_MISMATCH: False,
    ErrorCode.CONTRACT_MISMATCH: False,
    ErrorCode.MODEL_LOADING: True,
    ErrorCode.UPSTREAM_LLM: True,
    ErrorCode.TIMEOUT: True,
    ErrorCode.CANCELLED: True,
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
# Cancel
# ---------------------------------------------------------------------------

CancelOutcome = Literal["cancelled", "finished", "unknown"]


class CancelResponse(BaseModel):
    """Response body of `POST /v1/jobs/{job_id}/cancel`.

    The route always answers 200; the status says what the service found.
    A request waiting on a cancelled job is answered `ErrorCode.CANCELLED`.

    Attributes:
        job_id: The job the request named.
        status: `cancelled` when a run was in flight and is now cancelled;
            `finished` when the job already has a stored result, which is
            kept; `unknown` when the service holds neither.
    """

    job_id: str
    status: CancelOutcome


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

    The four fields after `location_label` were added in contract 1.1.0 and
    describe the serving configuration a result was produced under, so a
    gauntlet run can record what it measured; the two `run_identity` fields
    were added in 1.2.0, and the three `serving_identity` fields in 1.3.0.
    Each defaults to None: a 1.0.0 server's payload (the
    first seven fields only) still validates, and None always means "not
    reported" -- an older server, or a value the service could not read
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
        run_identity: The run identity every result from this process is
            stamped with (contract 1.2.0), computed once at service start.
        run_identity_components: The values `run_identity` hashes, by name,
            so a report can say which of them two runs differ in.
        serving_identity: The serving identity of this call (contract 1.3.0),
            computed from llama-server's `/props` read for it. None when it
            cannot be computed; `serving_identity_error` says why.
        serving_identity_components: The flat values `serving_identity`
            hashes, so a backend can name the settings two identities differ
            in. Reported whenever `/props` gave a context size, with or
            without an identity.
        serving_identity_error: Why `serving_identity` is None: a required
            `EXTRACTION_SERVER_*` declaration is missing, `/props` is
            unreadable, or the declared context size differs from the observed
            one. None when the identity is reported, or by a pre-1.3.0 service.
        pack_prompt_hash: The prompt hash of the ontology pack the service
            runs with. None means not reported, and the serialized response
            then has no such key.
        pack_rails_hash: That pack's rails hash; None and its serialization
            likewise.
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
    run_identity: str | None = None
    run_identity_components: dict[str, str | int | float | bool | None] | None = None
    serving_identity: str | None = None
    serving_identity_components: dict[str, str | int | float | bool | None] | None = None
    serving_identity_error: str | None = None
    pack_prompt_hash: str | None = None
    pack_rails_hash: str | None = None

    @model_serializer(mode="wrap")
    def _omit_absent_pack_hashes(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Serialize as usual, then drop each pack hash that is not reported."""
        return _without_absent(handler(self), self, _PACK_HASH_FIELDS)


# ---------------------------------------------------------------------------
# Extraction status (WebSocket + GET /extraction/status; ADR-017 additive)
# ---------------------------------------------------------------------------


class ServiceStatus(BaseModel):
    """Extraction service reachability/identity, as observed by the backend.

    Attributes:
        serving_identity: The serving identity the service last reported
            (contract 1.3.0). None when it reported none, or is unknown.
    """

    reachable: bool
    location_label: str | None
    model_id: str | None
    extraction_version: str | None
    contract_version: str | None
    last_health_ms: int | None
    serving_identity: str | None = None


class LastJob(BaseModel):
    """Summary of the most recently finished extraction job."""

    event_id: str
    turn_id: str
    request_id: str
    duration_ms: float
    outcome: Literal["applied", "skipped", "dead_lettered", "failed"]
    finished_ms: int


class CutoverStatus(BaseModel):
    """Progress of a model/version cutover in flight, when one is running.

    Attributes:
        target_serving_identity: The serving identity the candidate is bound
            to (contract 1.3.0). None while it is unbound.
    """

    state: Literal["filling", "ready", "checked", "promoted", "abandoned"]
    target_extraction_version: str
    target_model_hash: str
    covered: int
    total: int
    target_serving_identity: str | None = None


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
