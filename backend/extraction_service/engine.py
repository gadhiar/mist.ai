"""ExtractionEngine: orchestrates Stages 1, 1.5, 2, and 9 for one job.

This is the service's whole reason to exist: reuse the render/parse
functions from `backend.knowledge.extraction` (scope_classifier,
ontology_extractor, internal_derivation) against a locally-configured
`StreamingLLMProvider` and model-family adapter, with strict parsing and
parse-and-repair so a failed job is reported as an error -- never cached
as an empty extraction.

Stage 2 runs the list pass: `structure_splitter` cuts a structured utterance
into units that keep their parent context, the pack's item filters drop the
units it names, and each batch of units under one lead-in is one Stage 2 call.
A batch's utterance holds its own units only; a later batch one of whose items
refers back is shown the earlier units under `EARLIER_ITEMS_HEADING`, for
reference only. An utterance without structure is one batch, sent exactly as
before the list pass existed. Every Stage 2 call leaves one entry in the
response's `batch_record`. A turn with no unit left makes no plan call and no
Stage 2 call; its scope call is made, and so is its derivation call when the
request asks for one.

With `ServiceSettings.turn_plan` on, a structured utterance first gets one
plan call over all of its units (`backend.extraction_service.turn_plan`),
and every batch then carries the plan in front of its user message. The flag
is off by default, and with it off no request differs from one sent before
the plan existed.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import importlib
import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType, ModuleType
from typing import Any

from openai import APIError
from pydantic import ValidationError

from backend.errors import (
    ExtractionValidationError,
    LLMConnectionError,
    LLMResponseError,
    MistError,
)
from backend.extraction_contract.models import (
    CONTRACT_VERSION,
    BatchRecordEntry,
    DerivationOut,
    ExtractionPayload,
    ExtractRequest,
    ExtractResponse,
    ResultStamps,
    ScopeOut,
    TimingsMs,
    TurnPlanOut,
)
from backend.extraction_service.adapters import ModelFamilyAdapter
from backend.extraction_service.schemas import (
    DERIVATION_OUTPUT_SCHEMA,
    SCOPE_OUTPUT_SCHEMA,
)
from backend.extraction_service.settings import ConstrainedMode, ReasoningEffort, ServiceSettings
from backend.extraction_service.turn_plan import (
    PLAN_BLOCK_CONSTANTS,
    PLAN_OUTPUT_SCHEMA,
    PLAN_REQUEST_TEMPLATE,
    is_structured_turn,
    parse_turn_plan,
    render_plan_block,
    render_plan_request,
)
from backend.knowledge.extraction.internal_derivation import (
    parse_derivation_output,
    render_derivation_messages,
)
from backend.knowledge.extraction.internal_prompts import (
    INTERNAL_DERIVATION_SYSTEM_PROMPT,
    INTERNAL_DERIVATION_USER_TEMPLATE,
)
from backend.knowledge.extraction.ontology_extractor import (
    NO_PRIOR_CONTEXT,
    parse_extraction_output,
    render_extraction_messages,
)
from backend.knowledge.extraction.pre_llm import filter_units
from backend.knowledge.extraction.preprocessor import PreProcessedInput, PreProcessor
from backend.knowledge.extraction.prompts import EARLIER_ITEMS_HEADING, EXTRACTION_USER_TEMPLATE
from backend.knowledge.extraction.scope_classifier import (
    SCOPE_CLASSIFIER_SYSTEM_PROMPT,
    SCOPE_USER_TEMPLATE,
    parse_scope_output,
    render_scope_messages,
)
from backend.knowledge.extraction.serving_identity import REQUEST_STAGES, StageRequestProfile
from backend.knowledge.extraction.structure_splitter import (
    InputUnit,
    batch_units,
    is_numbered_list,
    merge_payloads,
    needs_earlier_units,
    render_batch,
    render_unit,
    split_structured,
)
from backend.knowledge.ontologies.pack import Pack
from backend.knowledge.ontologies.registry import active_pack
from backend.knowledge.version_stamps import EXTRACTION_VERSION
from backend.llm.models import LLMRequest, LLMResponse
from backend.llm.provider import StreamingLLMProvider

logger = logging.getLogger(__name__)

# Exceptions a failed LLM transport call surfaces as, across the openai
# client (APIError and its APIConnectionError/APIStatusError/RateLimitError
# subclasses) and the in-repo LLMProvider contract (LLMConnectionError,
# LLMResponseError -- e.g. malformed tool-call JSON). Deliberately not a
# bare `except Exception`: anything outside this set is a real bug, not a
# transport failure, and should propagate.
_LLM_TRANSPORT_ERRORS = (APIError, LLMConnectionError, LLMResponseError)

_REPAIR_INSTRUCTION = (
    "Your previous output was not valid JSON, was missing the required "
    '"entities"/"relationships" list fields, or had list items that were not '
    "JSON objects. Return ONLY a JSON object "
    'matching {"entities": [...], "relationships": [...]}. No prose, no '
    "markdown code fences, no explanation."
)

# Output token limits at low reasoning effort, one per LLM-calling stage. Named
# so the run identity can record them: a limit that cuts an output short
# changes what the stage returns as surely as a prompt edit does.
SCOPE_MAX_TOKENS = 96
EXTRACT_MAX_TOKENS = 2048
DERIVE_MAX_TOKENS = 400


@dataclass(frozen=True, slots=True)
class StageCaps:
    """The output token cap of each LLM-calling stage, at one reasoning effort.

    The fields are named as the stages are in `REQUEST_STAGES`, so a stage's
    cap is `getattr(caps, stage)`.
    """

    scope: int
    extract: int
    derive: int


# A model reasoning at medium or high effort spends its output tokens on
# reasoning before the answer, so low's caps would cut most calls short. The
# medium extract cap sits above the longest extract call that stopped on its
# own at high effort (28,916 tokens) and cuts a runaway call a third sooner than
# the 49,152 that run used. High takes medium's caps; no run has measured them
# at high. Scope and derive keep low's caps (96 and 400) at every effort:
# `effective_cap` sizes a stage by the effort the adapter sends it, and the
# adapters send both at low under every setting (`GptOssAdapter.stage_effort`),
# so the medium and high scope entries below are never sent. Scope is a
# classification, sent at low as one; derive stays at low because no effort run
# calls it, so derive above low would ship unmeasured.
STAGE_CAPS: Mapping[str, StageCaps] = MappingProxyType(
    {
        "low": StageCaps(
            scope=SCOPE_MAX_TOKENS, extract=EXTRACT_MAX_TOKENS, derive=DERIVE_MAX_TOKENS
        ),
        "medium": StageCaps(scope=1024, extract=32768, derive=DERIVE_MAX_TOKENS),
        "high": StageCaps(scope=1024, extract=32768, derive=DERIVE_MAX_TOKENS),
    }
)

# The context a prompt may take beside the extract cap. The largest prompt
# measured on the r6 prompt was 9,327 tokens, and the prompt's stated
# conventions add about 2,600 more by estimate, so the reserve is the largest
# value with which low's extract cap still fits a 16,384 context (2,048 +
# 14,336), the context a run beside the main stack and host-1070 use. Medium
# then needs 47,104. The `extraction call:` line reports each call's real
# prompt_tokens. The service withholds its serving identity when the extract
# cap plus this reserve exceeds the context llama-server reports
# (`backend.extraction_service.app._serving_identity_error`).
PROMPT_RESERVE_TOKENS: int = 14336


def stage_caps(effort: ReasoningEffort | None) -> StageCaps:
    """The caps for `effort`; None, an adapter that sends no effort, gets low's."""
    return STAGE_CAPS["low" if effort is None else effort]


def effective_cap(adapter: ModelFamilyAdapter, stage: str) -> int:
    """The output token cap `stage` is sent with under `adapter`.

    It follows the effort the adapter applies to that stage, not the
    configured effort, so an adapter that sends no effort keeps low's caps.
    """
    return getattr(stage_caps(adapter.stage_effort(stage)), stage)


# The Stage 2 list pass's limits: the most units one call extracts, and the
# most earlier units of the same message a later batch sees as context. Small
# batches keep a long list inside EXTRACT_MAX_TOKENS; the earlier units let an
# item refer back to one in an earlier batch. They live here, beside the token
# limits, so the run identity reads them from the module that applies them.
LIST_BATCH_MAX_UNITS: int = 4
LIST_CONTEXT_MAX_UNITS: int = 8

# The code the run identity covers (`code_sha256` in
# `backend.knowledge.extraction.run_identity`): this module, the adapters,
# the schemas and the turn plan, every `backend.knowledge.extraction` module
# imported above, whose renderers and strict parsers Stages 1.5, 2 and 9 call,
# the pack loader, the registry and the pre-LLM interpreters, which decide the
# prompt, the schema and the units the model is given, and the provider module
# the service builds. Whole modules rather than functions, so a helper a covered
# function calls is covered too; an edit elsewhere in a covered module costs one
# `CODE_REVISIONS` line, never a silent miss.
RUN_IDENTITY_CODE: tuple[ModuleType, ...] = tuple(
    importlib.import_module(name)
    for name in (
        __name__,
        "backend.extraction_service.adapters",
        "backend.extraction_service.schemas",
        "backend.extraction_service.turn_plan",
        "backend.knowledge.extraction.internal_derivation",
        "backend.knowledge.extraction.internal_prompts",
        "backend.knowledge.extraction.ontology_extractor",
        "backend.knowledge.extraction.pre_llm",
        "backend.knowledge.extraction.preprocessor",
        "backend.knowledge.extraction.prompts",
        "backend.knowledge.extraction.scope_classifier",
        "backend.knowledge.extraction.structure_splitter",
        "backend.knowledge.ontologies.pack",
        "backend.knowledge.ontologies.registry",
        "backend.llm.llama_server_provider",
    )
)

# The backend modules this module, the adapters and the turn plan import from
# that the run identity does not hash, each with the reason. The coverage test in
# tests/unit/knowledge/extraction/test_run_identity.py fails on an import in
# neither tuple.
RUN_IDENTITY_EXCLUDED: tuple[tuple[str, str], ...] = (
    (
        "backend.errors",
        "exception types only: a failure is raised, never returned as a result",
    ),
    (
        "backend.extraction_contract.models",
        "the wire models; every response carries their CONTRACT_VERSION",
    ),
    (
        "backend.llm.models",
        "request and response containers; the covered provider module turns them "
        "into the llama-server call",
    ),
    (
        "backend.llm.provider",
        "the provider Protocol, with no implementation",
    ),
    (
        "backend.extraction_service.settings",
        "its values are components already: temperature, attempts, constrained mode, "
        "reasoning, the scope switch, the model hash and the llama.cpp build",
    ),
    (
        "backend.knowledge.version_stamps",
        "its versions are components already, and it holds CODE_REVISIONS: covering it "
        "would move the code identity each new entry has to name, so the drift guard "
        "could never pass",
    ),
    (
        "backend.knowledge.extraction.serving_identity",
        "the serving identity's types and hash: request_profile reads the requests the "
        "stages build and changes none of them",
    ),
    # No entry for `backend.knowledge.ontologies`: no service module imports an
    # ontology's own modules. The covered registry reaches a pack's types and rules,
    # the turn plan's vocabulary among them, and pack_prompt_hash and
    # pack_rails_hash, stamped on every result, cover their content.
)


class ExtractionServiceError(MistError):
    """Base class for extraction-service-specific failures.

    Subclasses map to the service's HTTP error taxonomy
    (`backend.extraction_contract.models.ErrorCode`) at the `app.py` layer.
    """


class UpstreamLLMError(ExtractionServiceError):
    """The llama-server call failed, repair retries ran out, or one unit's output was cut short.

    Maps to HTTP 502 / `ErrorCode.UPSTREAM_LLM`.
    """


class ExtractionTimeoutError(ExtractionServiceError):
    """An LLM call exceeded `ServiceSettings.llm_timeout_seconds`.

    Maps to HTTP 504 / `ErrorCode.TIMEOUT`.
    """


class _OutputTruncatedError(ExtractionServiceError):
    """A Stage 2 output stopped at the extract cap (`finish_reason == "length"`).

    Internal to Stage 2: the batch is split and retried, and only a single
    unit cut short leaves the engine, as `UpstreamLLMError`.
    """


class InvalidRequestError(ExtractionServiceError):
    """The request passed the contract model but carries an unusable value.

    Maps to HTTP 422 / `ErrorCode.CONTRACT_MISMATCH`: the caller must fix the
    request, not retry it unchanged.
    """


def parse_recorded_at(recorded_at: str) -> datetime:
    """Parse `ExtractRequest.recorded_at`, the turn's reference date.

    The contract types it as a plain `str` (an ISO-8601 timestamp by
    convention), so the model accepts any string; this is where a value that
    is not ISO-8601 is refused.

    Raises:
        InvalidRequestError: `recorded_at` is not an ISO-8601 timestamp.
    """
    try:
        return datetime.fromisoformat(recorded_at)
    except ValueError as exc:
        raise InvalidRequestError(
            f"recorded_at {recorded_at!r} is not an ISO-8601 timestamp: {exc}"
        ) from exc


def _require_object_items(field: str, items: list[Any]) -> list[dict[str, Any]]:
    """Return `items` when every element is a JSON object, else raise.

    The strict parsers (`parse_extraction_output`, `parse_derivation_output`)
    check that `entities`/`relationships`/`operations` are lists, not what the
    lists hold; the contract models type them `list[dict[str, Any]]`. A model
    that answers `{"entities": ["Alice"], ...}` must count as unparsable
    output here -- repaired, or reported -- not fail later inside a pydantic
    constructor as an unhandled error.

    Raises:
        ExtractionValidationError: An element is not a JSON object.
    """
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ExtractionValidationError(
                f"{field}[{index}] is a JSON {_json_type(item)}, not an object"
            )
    return items


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return type(value).__name__


def _strict_extraction_payload(raw: str) -> ExtractionPayload:
    """Stage 2 strict parse: well-formed JSON, list fields, object items.

    Raises:
        ExtractionValidationError: Any of those fails, including the payload
            model rejecting the parsed value.
    """
    parsed = parse_extraction_output(raw, strict=True)
    entities = _require_object_items("entities", parsed["entities"])
    relationships = _require_object_items("relationships", parsed["relationships"])
    try:
        return ExtractionPayload(entities=entities, relationships=relationships)
    except ValidationError as exc:
        raise ExtractionValidationError(f"Extraction payload rejected: {exc}") from exc


def _strict_derivation(raw: str) -> DerivationOut:
    """Stage 9 strict parse: well-formed JSON, a list of object operations.

    Raises:
        ExtractionValidationError: Any of those fails, including the
            derivation model rejecting the parsed value.
    """
    operations = _require_object_items("operations", parse_derivation_output(raw, strict=True))
    try:
        return DerivationOut(operations=operations)
    except ValidationError as exc:
        raise ExtractionValidationError(f"Derivation output rejected: {exc}") from exc


@dataclass(slots=True)
class _Stage2Run:
    """What one turn's Stage 2 calls add up to, across its batches.

    Attributes:
        calls: Every batch call, splits and repair retries included. The
            plan call is not one of them.
        attempts: The most attempts any one batch took.
        elapsed_ms: The time the batch calls took. The plan call's time is
            not in it; it shows in the turn's total alone.
        payloads: One payload per batch, in order.
        plan_block: The text every batch of the turn carries in front of its
            user message, or the empty string when the turn has no plan: the
            flag is off, the turn is prose, or the plan call failed.
        turn_plan: The record of the plan call, or None when none was made.
        batch_record: One entry per batch call, in call order, repair
            retries and calls cut short included, so it has `calls` entries.
        numbered: Whether the turn's message is a numbered list, in which
            an item that cites another by number refers back.
    """

    calls: int = 0
    attempts: int = 0
    elapsed_ms: float = 0.0
    payloads: list[ExtractionPayload] = field(default_factory=list)
    plan_block: str = ""
    turn_plan: TurnPlanOut | None = None
    batch_record: list[BatchRecordEntry] = field(default_factory=list)
    numbered: bool = False


@dataclass(frozen=True, slots=True)
class _Batch:
    """One batch call's input, and what its batch record entry repeats of it.

    Attributes:
        pre_processed: The turn's input, narrowed to the batch.
        units: The batch's own units, as the structure splitter gave them.
        earlier: The earlier units of the message the call is shown for
            reference, rendered; empty when it is shown none.
    """

    pre_processed: PreProcessedInput
    units: tuple[InputUnit, ...]
    earlier: tuple[str, ...]


def _batch_input(
    pre_processed: PreProcessedInput,
    units: Sequence[InputUnit],
    start: int,
    size: int,
    *,
    numbered: bool,
) -> _Batch:
    """The turn's pre-processed input, narrowed to the batch `units[start:start + size]`.

    The batch's rendered units replace the utterance, so a call sees its own
    units only. When one of them refers back (`needs_earlier_units`), the
    nearest earlier units of the same message, at most
    `LIST_CONTEXT_MAX_UNITS`, follow the turn's history under
    `EARLIER_ITEMS_HEADING`, so the item can be read; Rule 8 extracts what the
    utterance states, so they are not extracted again. A batch whose items do
    not refer back is shown none, and neither is the first batch, so a turn
    without structure renders exactly as it did before.
    """
    batch = tuple(units[start : start + size])
    text = render_batch(batch)
    earlier: tuple[str, ...] = ()
    if start > 0 and needs_earlier_units(batch, numbered=numbered):
        earlier = tuple(
            render_unit(unit) for unit in units[max(0, start - LIST_CONTEXT_MAX_UNITS) : start]
        )
    context = [*pre_processed.conversation_context]
    if earlier:
        context += [EARLIER_ITEMS_HEADING, *earlier]
    narrowed = dataclasses.replace(
        pre_processed, original_text=text, resolved_text=text, conversation_context=context
    )
    return _Batch(pre_processed=narrowed, units=batch, earlier=earlier)


def _record_entry(
    batch: _Batch, response: LLMResponse, payload: ExtractionPayload | None
) -> BatchRecordEntry:
    """The batch record's entry for one call of `batch` that returned `response`."""
    usage = response.usage
    return BatchRecordEntry(
        units=[
            {"text": unit.text, "context": list(unit.context), "kind": unit.kind}
            for unit in batch.units
        ],
        context=list(batch.earlier),
        payload=payload,
        prompt_tokens=None if usage is None else usage.prompt_tokens,
        completion_tokens=None if usage is None else usage.completion_tokens,
        finish=response.finish_reason,
    )


def _with_plan_block(messages: list[dict], plan_block: str) -> list[dict]:
    """`messages` with `plan_block` in front of the first user message's text.

    The block goes before everything a batch renders for itself, so the
    batches of one turn share a prefix that runs through the system prompt
    and the whole block, and llama-server's prompt cache covers both.
    """
    system, user, *rest = messages
    return [system, {**user, "content": plan_block + user["content"]}, *rest]


def _compute_prompt_sha256(
    adapter: ModelFamilyAdapter, *, turn_plan: bool = False, pack: Pack | None = None
) -> str:
    """Hash the prompt text and output schemas sent to the model, plus the adapter identity.

    `prompt_sha256` is a per-result provenance stamp: the record of what the
    model was asked. It is not the epoch's identity, which is
    `extraction_version` plus the composed `model_hash`
    (`ExtractionDispatcher._epoch_matches`).

    Covered: the raw, unformatted templates of all three stages (the scope
    system prompt and user template, the extraction system prompt and user
    template, the derivation system prompt and user template), the heading a
    later Stage 2 batch's earlier units stand under, the Stage 2
    empty-context placeholder, the Stage 2 repair instruction sent on a
    retry, the three constrained-decoding output schemas (serialised with
    sorted keys; hashed whether or not the active mode sends them, which
    only makes the stamp stricter), and the adapter's own name/version.

    With `turn_plan` on, three more follow, in this order: the plan request
    text, every constant string of the plan block, and the plan's output
    schema. With it off nothing is added, so the hash is the one a service
    without the turn plan computes.

    Not covered: the PreProcessor's context formatting (code, not a
    template) and request parameters (temperature, max_tokens, reasoning
    effort and budget, constrained mode), which are configuration and change
    no prompt text. The run identity covers both, beside this hash
    (`backend.extraction_service.app.run_identity_components`).

    Components are joined with a unit separator (0x1F) rather than
    concatenated, so moving text from one component to its neighbour cannot
    produce the same hash.

    Computed once at engine construction (not per request): none of its
    inputs vary per job.

    Args:
        adapter: The model-family adapter the service runs.
        turn_plan: Whether the service runs with the turn plan
            (`ServiceSettings.turn_plan`).
        pack: The pack the service extracts with: its assembled system prompt
            and its generated schema are the Stage 2 components. None takes
            the registry's active pack, as the engine's constructor does:
            no module holds a prompt or an extraction schema of its own.
            The pack's two hashes are stamps of their own and are no
            component here.
    """
    if pack is None:
        pack = active_pack()
    components = [
        SCOPE_CLASSIFIER_SYSTEM_PROMPT,
        SCOPE_USER_TEMPLATE,
        pack.system_prompt,
        EXTRACTION_USER_TEMPLATE,
        EARLIER_ITEMS_HEADING,
        NO_PRIOR_CONTEXT,
        INTERNAL_DERIVATION_SYSTEM_PROMPT,
        INTERNAL_DERIVATION_USER_TEMPLATE,
        _REPAIR_INSTRUCTION,
        json.dumps(SCOPE_OUTPUT_SCHEMA, sort_keys=True),
        json.dumps(pack.extraction_schema, sort_keys=True),
        json.dumps(DERIVATION_OUTPUT_SCHEMA, sort_keys=True),
        adapter.name,
        adapter.version,
    ]
    if turn_plan:
        components += [
            PLAN_REQUEST_TEMPLATE,
            *PLAN_BLOCK_CONSTANTS,
            json.dumps(PLAN_OUTPUT_SCHEMA, sort_keys=True),
        ]
    return hashlib.sha256("\x1f".join(components).encode("utf-8")).hexdigest()


def repetition_flag(
    text: str, *, tail_chars: int = 2000, probe_chars: int = 200, min_count: int = 3
) -> bool:
    """True when the last `probe_chars` of `text`'s tail occur `min_count` times in that tail.

    The tail is the last `tail_chars` characters. A model stuck in a loop
    repeats its last stretch of text, so any loop whose period is at most
    `tail_chars / min_count` characters (about 666 at the defaults) is
    flagged, while text still making progress is not. Occurrences may
    overlap, so a loop shorter than the probe counts every repetition.
    """
    tail = text[-tail_chars:]
    if len(tail) < probe_chars:
        return False
    probe = tail[-probe_chars:]
    count = 0
    start = tail.find(probe)
    while start != -1:
        count += 1
        if count >= min_count:
            return True
        start = tail.find(probe, start + 1)
    return False


def _log_length_stop(request: LLMRequest, response: LLMResponse) -> None:
    """Log a call that stopped at its cap: counts and the repetition flag, never text.

    The cap names the stage, since the three caps differ at every effort. The
    response text stays out of the log because production turns are real
    conversations; the flag is what tells a repetition loop from reasoning
    that was still progressing when the cap cut it.
    """
    reasoning = response.reasoning_content or ""
    content = response.content or ""
    completion_tokens = None if response.usage is None else response.usage.completion_tokens
    logger.warning(
        "extraction length stop: cap %d completion_tokens %s reasoning_chars %d "
        "content_chars %d repeat %s",
        request.max_tokens,
        completion_tokens,
        len(reasoning),
        len(content),
        "true" if repetition_flag(reasoning[-2000:] + content) else "false",
    )


def _log_call(request: LLMRequest, response: LLMResponse) -> None:
    """Log one LLM call's cap, token counts and finish reason, never text.

    A run's prompt size cannot be counted offline (the tokenizer needs the
    network inside the container), so this line is how a run records what
    llama-server counted for every call, context lines included. The cap
    names the stage, as on the length-stop line. A count the server did not
    report prints as None.
    """
    usage = response.usage
    logger.info(
        "extraction call: cap %d prompt_tokens %s completion_tokens %s finish %s",
        request.max_tokens,
        None if usage is None else usage.prompt_tokens,
        None if usage is None else usage.completion_tokens,
        response.finish_reason,
    )


def _transport_failure(exc: Exception) -> str:
    """A failed transport call as its exception type and HTTP status, never its message.

    The openai client puts an error response's body in the exception's
    message. Whether llama-server's error bodies quote the request or the
    model's reply is not verified, and production turns are real
    conversations, so the message stays out of the log.
    """
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__
    return name if status is None else f"{name} (HTTP {status})"


def _profile_of(stage: str, request: LLMRequest) -> StageRequestProfile:
    """The output-changing fields of `request`, read as the provider sends them.

    The constraint follows `LlamaServerProvider._build_kwargs`: a schema wins
    over JSON mode, and neither means an unconstrained reply.
    """
    if request.response_schema is not None:
        constrained = "schema"
    elif request.json_mode:
        constrained = "json_object"
    else:
        constrained = "none"
    thinking = request.thinking
    return StageRequestProfile(
        stage=stage,
        temperature=request.temperature,
        top_p=request.top_p,
        constrained=constrained,
        thinking_effort=None if thinking is None else thinking.effort,
        thinking_enabled=None if thinking is None else thinking.enabled,
        thinking_budget_tokens=None if thinking is None else thinking.budget_tokens,
    )


class ExtractionEngine:
    """Runs Stages 1 / 1.5 / 2 / 9 for one `ExtractRequest`.

    Stateless across requests except for the once-computed prompt hash --
    every method reads only its arguments and the injected collaborators.
    """

    def __init__(
        self,
        llm: StreamingLLMProvider,
        adapter: ModelFamilyAdapter,
        settings: ServiceSettings,
        run_identity: str | None = None,
        pack: Pack | None = None,
    ) -> None:
        """Initialize the engine.

        Args:
            llm: LLM provider for the service's own llama-server.
            adapter: Model-family adapter matching `llm`'s model.
            settings: Service configuration.
            run_identity: The run identity to stamp on every result, computed
                once at service start by the caller (`build_app`). The engine
                takes it rather than computing it, so it imports no backend
                module beyond those `RUN_IDENTITY_CODE` and
                `RUN_IDENTITY_EXCLUDED` name. None stamps nothing.
            pack: The ontology pack to extract with: its item filters choose
                the units the model is given, its assembled prompt and its
                generated schema are what Stage 2 sends, and its two hashes
                are stamped on every result. None takes the registry's
                active pack.
        """
        self._llm = llm
        self._adapter = adapter
        self._settings = settings
        self._run_identity = run_identity
        self._pack = pack if pack is not None else active_pack()
        self._preprocessor = PreProcessor()
        self._prompt_sha256 = _compute_prompt_sha256(
            adapter, turn_plan=settings.turn_plan, pack=self._pack
        )

    @property
    def adapter_name(self) -> str:
        """The active model-family adapter's name (e.g. "gptoss")."""
        return self._adapter.name

    @property
    def pack(self) -> Pack:
        """The ontology pack this engine extracts with."""
        return self._pack

    @property
    def run_identity(self) -> str | None:
        """The run identity this engine stamps on every result."""
        return self._run_identity

    def stage_cap(self, stage: str) -> int:
        """The output token cap `stage`'s request is sent with (`effective_cap`)."""
        return effective_cap(self._adapter, stage)

    def request_profile(self) -> tuple[StageRequestProfile, ...]:
        """The output-changing fields of each stage's request, in `REQUEST_STAGES` order.

        Each profile comes from the builder the stage itself calls, so it is
        the request as sent, after the adapter has prepared it: a setting the
        adapter ignores is absent. The messages are left empty, because the
        profile records how a request is sampled and constrained, not what it
        says. Every stage is listed, Stage 1.5 even when `scope_enabled` is
        off, since that switch is a component of its own. The turn plan call
        has no entry: it is sent as stage `extract`, with that stage's cap and
        thinking controls, and differs from a batch only in its messages and
        in which schema constrains it.
        """
        builders = {
            "scope": self._scope_request,
            "extract": self._extract_request,
            "derive": self._derive_request,
        }
        return tuple(_profile_of(stage, builders[stage](messages=[])) for stage in REQUEST_STAGES)

    async def run(
        self, req: ExtractRequest, *, serving_identity: str | None = None
    ) -> ExtractResponse:
        """Run the full stage sequence for one job and build its response.

        Args:
            req: The job.
            serving_identity: The serving identity the caller computed for this
                job, stamped on the result. None stamps none.

        Raises:
            InvalidRequestError: `req.recorded_at` is not ISO-8601; raised
                before any LLM call.
            UpstreamLLMError: The extraction LLM call failed, or the model
                never produced parseable output within `max_attempts`.
            ExtractionTimeoutError: An LLM call exceeded
                `settings.llm_timeout_seconds`.
        """
        total_start = time.perf_counter()
        warnings: list[str] = []

        reference_date = parse_recorded_at(req.recorded_at)
        conversation_history = [
            {"role": m.role, "content": m.content} for m in req.conversation_history
        ]
        pre_processed = self._preprocessor.pre_process(
            utterance=req.utterance,
            conversation_history=conversation_history,
            reference_date=reference_date,
            turn_index=req.turn_index,
        )

        # Every turn makes its scope call: the label is stored on the cache row
        # and read after Stage 2 too. The pack's item filters run on the split
        # units before Stage 2, and a turn they leave no unit of makes no plan
        # or Stage 2 call and is answered with an empty payload.
        scope_out, scope_ms = await self._run_scope(pre_processed, warnings)
        units = filter_units(split_structured(pre_processed.original_text), self._pack)
        stage2 = await self._run_extraction(pre_processed, units, warnings)
        payload, merge_warnings = merge_payloads(stage2.payloads)
        warnings.extend(merge_warnings)
        derivation_out, derive_ms = await self._run_derivation(req, warnings)

        total_ms = (time.perf_counter() - total_start) * 1000

        return ExtractResponse(
            contract_version=CONTRACT_VERSION,
            job_id=req.job_id,
            outcome="extracted",
            scope=scope_out,
            payload=payload,
            derivation=derivation_out,
            stamps=ResultStamps(
                extraction_version=EXTRACTION_VERSION,
                model_hash=self._settings.model_hash,
                prompt_sha256=self._prompt_sha256,
                llama_cpp_build=self._settings.llama_cpp_build,
                adapter=self._adapter.name,
                run_identity=self._run_identity,
                serving_identity=serving_identity,
                pack_prompt_hash=self._pack.pack_prompt_hash,
                pack_rails_hash=self._pack.pack_rails_hash,
            ),
            timings_ms=TimingsMs(
                scope=scope_ms,
                extract=stage2.elapsed_ms,
                derive=derive_ms,
                total=total_ms,
            ),
            attempts=stage2.attempts,
            stage2_calls=stage2.calls,
            warnings=warnings,
            turn_plan=stage2.turn_plan,
            batch_record=stage2.batch_record,
        )

    def _resolve_constrained_mode(self) -> ConstrainedMode:
        """Settings override the active adapter's default, when set."""
        return self._settings.constrained_mode or self._adapter.default_constrained_mode

    def _build_request(self, *, messages: list[dict], max_tokens: int, schema: dict) -> LLMRequest:
        """Build a model-agnostic LLMRequest for one stage call.

        The adapter's `prepare()` still needs to run on the result to add
        thinking controls -- this only sets the fields common to every
        stage plus the constrained-decoding fields.
        """
        mode = self._resolve_constrained_mode()
        return LLMRequest(
            messages=messages,
            temperature=self._settings.temperature,
            max_tokens=max_tokens,
            json_mode=(mode == "json_object"),
            response_schema=schema if mode == "schema" else None,
        )

    def _scope_request(self, *, messages: list[dict]) -> LLMRequest:
        """Stage 1.5's request for `messages`, prepared by the adapter."""
        request = self._build_request(
            messages=messages, max_tokens=self.stage_cap("scope"), schema=SCOPE_OUTPUT_SCHEMA
        )
        return self._adapter.prepare(request, stage="scope")

    def _extract_request(self, *, messages: list[dict]) -> LLMRequest:
        """Stage 2's request for `messages`, prepared by the adapter."""
        request = self._build_request(
            messages=messages,
            max_tokens=self.stage_cap("extract"),
            schema=dict(self._pack.extraction_schema),
        )
        return self._adapter.prepare(request, stage="extract")

    def _extraction_messages(self, batch_input: PreProcessedInput) -> list[dict]:
        """Stage 2's messages for `batch_input`, under the pack's system prompt.

        The shared renderer writes the user message, and a system message from
        the prompt module's own template. The system message sent is the pack's
        assembled prompt with the same reference date, which for a pack that
        holds no prompt fragment is the same text.
        """
        messages = render_extraction_messages(batch_input)
        system_prompt = self._pack.system_prompt.format(
            reference_date=batch_input.reference_date.strftime("%Y-%m-%d")
        )
        return [{"role": "system", "content": system_prompt}, *messages[1:]]

    def _plan_request(self, *, messages: list[dict]) -> LLMRequest:
        """The turn plan's request for `messages`, prepared by the adapter as stage `extract`.

        The plan is part of Stage 2, made with the extract stage's cap and
        thinking controls, so the request stages stay the three the serving
        identity profiles. Only the schema differs from a batch's request.
        """
        request = self._build_request(
            messages=messages,
            max_tokens=self.stage_cap("extract"),
            schema=PLAN_OUTPUT_SCHEMA,
        )
        return self._adapter.prepare(request, stage="extract")

    def _derive_request(self, *, messages: list[dict]) -> LLMRequest:
        """Stage 9's request for `messages`, prepared by the adapter."""
        request = self._build_request(
            messages=messages,
            max_tokens=self.stage_cap("derive"),
            schema=DERIVATION_OUTPUT_SCHEMA,
        )
        return self._adapter.prepare(request, stage="derive")

    async def _invoke(self, request: LLMRequest, *, stream: bool = False):
        """Invoke the LLM under the configured timeout.

        Every response is logged (`_log_call`, at INFO), and one that stopped
        at its output cap is logged again (`_log_length_stop`, at WARNING),
        both before it is returned, so before the stage that called raises,
        retries or degrades on it. Every stage's call passes here.

        Args:
            request: The stage's request, as the adapter prepared it.
            stream: Send a streaming request and read it to its end
                (`_read_stream`). The Stage 2 calls pass True, so that a call
                which times out here, or whose run is cancelled, closes its
                stream. Whether llama-server stops generating when the stream
                closes is not measured here.

        The call runs in the caller's task under `asyncio.timeout`, not in a
        task of its own under `asyncio.wait_for`: on Python 3.11 `wait_for`
        returns the result of a call that finished by the time a cancel of
        the run woke it, and drops the cancel, so the run went on to its next
        call. Here the cancel stays pending and ends the run at its next
        await.

        Raises:
            TimeoutError: The call exceeded `settings.llm_timeout_seconds`.
            openai.APIError / LLMConnectionError / LLMResponseError: The
                transport or provider failed.
        """
        async with asyncio.timeout(self._settings.llm_timeout_seconds):
            if stream:
                response = await self._read_stream(request)
            else:
                response = await self._llm.invoke(request)
        _log_call(request, response)
        if response.finish_reason == "length":
            _log_length_stop(request, response)
        return response

    async def _read_stream(self, request: LLMRequest) -> LLMResponse:
        """One streamed call, as the response a call that is not streamed returns.

        The content is the partial chunks joined; the finish reason, usage
        and reasoning are the final chunk's. The provider's generator is
        closed on every way out, a cancellation or a timeout included, and
        closing it closes the stream.

        Raises:
            LLMResponseError: The stream ended with no final chunk, so
                nothing says the content is the whole reply.
        """
        parts: list[str] = []
        final: LLMResponse | None = None
        async with contextlib.aclosing(self._llm.generate(request, stream=True)) as chunks:
            async for chunk in chunks:
                if chunk.partial:
                    parts.append(chunk.content or "")
                else:
                    final = chunk
        if final is None:
            raise LLMResponseError("the streamed LLM call ended with no final chunk")
        return final.model_copy(update={"content": "".join(parts)})

    async def _run_scope(
        self, pre_processed: PreProcessedInput, warnings: list[str]
    ) -> tuple[ScopeOut, float | None]:
        """Stage 1.5. Never raises -- any failure degrades to "unknown"."""
        if not self._settings.scope_enabled:
            pre_processed.metadata["subject_scope"] = "unknown"
            pre_processed.metadata["subject_scope_confidence"] = 0.0
            return ScopeOut(label="unknown", confidence=0.0), None

        start = time.perf_counter()
        request = self._scope_request(messages=render_scope_messages(pre_processed))

        try:
            response = await self._invoke(request)
        except (TimeoutError, *_LLM_TRANSPORT_ERRORS) as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning("Scope classification failed (%s): %s", type(exc).__name__, exc)
            warnings.append("scope_classification_failed")
            pre_processed.metadata["subject_scope"] = "unknown"
            pre_processed.metadata["subject_scope_confidence"] = 0.0
            return ScopeOut(label="unknown", confidence=0.0), elapsed

        raw = self._adapter.final_text(response)
        label, confidence, _reasoning = parse_scope_output(raw)
        elapsed = (time.perf_counter() - start) * 1000
        pre_processed.metadata["subject_scope"] = label
        pre_processed.metadata["subject_scope_confidence"] = confidence
        if label == "unknown":
            warnings.append("scope_classification_failed")
        return ScopeOut(label=label, confidence=confidence), elapsed

    async def _run_extraction(
        self,
        pre_processed: PreProcessedInput,
        units: Sequence[InputUnit],
        warnings: list[str],
    ) -> _Stage2Run:
        """Stage 2: one call sequence per batch of at most `LIST_BATCH_MAX_UNITS` units.

        A batch ends at its maximum or where the lead-in changes. Batches run
        in order, one at a time, and any batch failing fails the turn: there
        are no partial results. With the turn plan on, a structured turn's
        plan call goes out first, and its failure adds a warning to
        `warnings` and fails nothing.

        Args:
            pre_processed: The turn's input.
            units: The turn's units, after the pack's item filters ran, so a
                unit they drop reaches neither a batch, nor a later batch's
                context, nor the plan call. A turn with none makes no call.

        Raises:
            ExtractionTimeoutError: An attempt exceeded the LLM timeout.
            UpstreamLLMError: The LLM call failed, or every attempt's
                output stayed unparsable.
        """
        run = _Stage2Run(numbered=is_numbered_list(pre_processed.original_text))
        if not units:
            return run
        if self._settings.turn_plan and is_structured_turn(units):
            run.turn_plan = await self._run_turn_plan(pre_processed, units, warnings)
            if run.turn_plan.payload is not None:
                run.plan_block = render_plan_block(run.turn_plan.payload)
        start = 0
        for batch in batch_units(units, LIST_BATCH_MAX_UNITS):
            await self._extract_units(pre_processed, units, start, len(batch), run)
            start += len(batch)
        return run

    async def _run_turn_plan(
        self,
        pre_processed: PreProcessedInput,
        units: Sequence[InputUnit],
        warnings: list[str],
    ) -> TurnPlanOut:
        """The turn's one plan call, over all of `units`. Never raises on a failed plan.

        The call goes out under the system message the batches use, rendered
        for the same reference date, so llama-server's cached prefix holds
        from the plan call through every batch. It is made once: an
        unparsable plan gets no repair retry and a plan cut short at the cap
        is not split, because the batches extract the same turn without it.

        A call that fails at the transport, times out, stops at its output
        cap or returns a plan the strict parser rejects logs one
        `turn plan failed:` line, adds `turn_plan_failed` to `warnings`, and
        is returned with no payload; the batches then run as they do with
        the plan off. The line carries the reason and never the turn's or
        the reply's text.
        """
        system = self._extraction_messages(pre_processed)[0]
        context = (
            "\n".join(pre_processed.conversation_context)
            if pre_processed.conversation_context
            else NO_PRIOR_CONTEXT
        )
        user = render_plan_request(
            units,
            context=context,
            subject_scope=pre_processed.metadata.get("subject_scope", "unknown"),
        )
        request = self._plan_request(messages=[system, {"role": "user", "content": user}])

        payload: dict[str, Any] | None = None
        response: LLMResponse | None = None
        failure: str | None = None
        try:
            response = await self._invoke(request, stream=True)
        except TimeoutError:
            failure = f"timed out after {self._settings.llm_timeout_seconds}s"
        except _LLM_TRANSPORT_ERRORS as exc:
            failure = _transport_failure(exc)
        else:
            if response.finish_reason == "length":
                failure = f"output stopped at max_tokens ({request.max_tokens})"
            else:
                try:
                    payload = parse_turn_plan(self._adapter.final_text(response))
                except ExtractionValidationError as exc:
                    failure = str(exc)

        if failure is not None:
            logger.warning("turn plan failed: %s", failure)
            warnings.append("turn_plan_failed")
        usage = None if response is None else response.usage
        return TurnPlanOut(
            payload=payload,
            prompt_tokens=None if usage is None else usage.prompt_tokens,
            completion_tokens=None if usage is None else usage.completion_tokens,
            finish_reason=None if response is None else response.finish_reason,
        )

    async def _extract_units(
        self,
        pre_processed: PreProcessedInput,
        units: Sequence[InputUnit],
        start: int,
        size: int,
        run: _Stage2Run,
    ) -> None:
        """Extract `units[start:start + size]`, halving the batch while its output is cut short.

        An output that stops at the extract cap is never repaired: the
        repair retry would ask for the same output under the same limit. The
        batch is split in two and each half extracted in turn, each half that
        refers back seeing the units before it, down to single units.

        Raises:
            UpstreamLLMError: A single unit's output was still cut short, or
                a call failed or stayed unparsable.
            ExtractionTimeoutError: An attempt exceeded the LLM timeout.
        """
        batch = _batch_input(pre_processed, units, start, size, numbered=run.numbered)
        try:
            payload = await self._extract_batch(batch, run)
        except _OutputTruncatedError as exc:
            if size == 1:
                raise UpstreamLLMError(
                    f"extraction output truncated at max_tokens ({self.stage_cap('extract')}) "
                    "on a single unit"
                ) from exc
            half = (size + 1) // 2
            logger.warning(
                "Extraction output cut short at max_tokens on %d units; retrying as %d and %d",
                size,
                half,
                size - half,
            )
            await self._extract_units(pre_processed, units, start, half, run)
            await self._extract_units(pre_processed, units, start + half, size - half, run)
            return
        run.payloads.append(payload)

    async def _extract_batch(self, batch: _Batch, run: _Stage2Run) -> ExtractionPayload:
        """One batch's Stage 2 calls: strict parse with repair-retry up to `max_attempts`.

        Adds its calls, attempts, time and batch record entries to `run`.

        Raises:
            ExtractionTimeoutError: An attempt exceeded the LLM timeout.
            UpstreamLLMError: The LLM call failed, or every attempt's
                output stayed unparsable.
        """
        start = time.perf_counter()
        try:
            return await self._extract_attempts(batch, run)
        finally:
            run.elapsed_ms += (time.perf_counter() - start) * 1000

    async def _extract_attempts(self, batch: _Batch, run: _Stage2Run) -> ExtractionPayload:
        """The calls of `_extract_batch`; each call that returns leaves a record entry.

        A call that raises at the transport or times out fails the turn, so
        it leaves no entry: there is no response to record.
        """
        messages = self._extraction_messages(batch.pre_processed)
        if run.plan_block:
            messages = _with_plan_block(messages, run.plan_block)
        last_error: Exception | None = None

        for attempt in range(1, self._settings.max_attempts + 1):
            run.calls += 1
            run.attempts = max(run.attempts, attempt)
            request = self._extract_request(messages=messages)

            try:
                response = await self._invoke(request, stream=True)
            except TimeoutError as exc:
                raise ExtractionTimeoutError(
                    f"Extraction LLM call timed out after {self._settings.llm_timeout_seconds}s"
                    f" (attempt {attempt}/{self._settings.max_attempts})"
                ) from exc
            except _LLM_TRANSPORT_ERRORS as exc:
                raise UpstreamLLMError(f"Extraction LLM call failed: {exc}") from exc

            if response.finish_reason == "length":
                run.batch_record.append(_record_entry(batch, response, None))
                raise _OutputTruncatedError(
                    f"Extraction output reached max_tokens={request.max_tokens}"
                    f" (attempt {attempt}/{self._settings.max_attempts})"
                )
            raw = self._adapter.final_text(response)
            try:
                payload = _strict_extraction_payload(raw)
            except ExtractionValidationError as exc:
                run.batch_record.append(_record_entry(batch, response, None))
                last_error = exc
                logger.warning(
                    "Extraction attempt %d/%d unparsable: %s",
                    attempt,
                    self._settings.max_attempts,
                    exc,
                )
                messages = [
                    *messages,
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": _REPAIR_INSTRUCTION},
                ]
                continue

            run.batch_record.append(_record_entry(batch, response, payload))
            return payload

        raise UpstreamLLMError(
            f"Extraction output unparsable after {self._settings.max_attempts} attempts"
        ) from last_error

    async def _run_derivation(
        self, req: ExtractRequest, warnings: list[str]
    ) -> tuple[DerivationOut | None, float | None]:
        """Stage 9. Only runs when `req.derivation` is set. Never raises."""
        if req.derivation is None:
            return None, None

        start = time.perf_counter()
        request = self._derive_request(
            messages=render_derivation_messages(
                utterance=req.utterance,
                assistant_response=req.derivation.assistant_response,
                signal_types=req.derivation.signal_types,
                matched_patterns=req.derivation.matched_patterns,
                existing_internal_entities=req.derivation.existing_internal_entities,
            )
        )

        try:
            response = await self._invoke(request)
            raw = self._adapter.final_text(response)
            derivation = _strict_derivation(raw)
        except (TimeoutError, ExtractionValidationError, *_LLM_TRANSPORT_ERRORS) as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning("Derivation failed (%s): %s", type(exc).__name__, exc)
            warnings.append("derivation_failed")
            return None, elapsed

        elapsed = (time.perf_counter() - start) * 1000
        return derivation, elapsed
