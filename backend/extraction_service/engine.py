"""ExtractionEngine: orchestrates Stages 1, 1.5, 2, and 9 for one job.

This is the service's whole reason to exist: reuse the render/parse
functions from `backend.knowledge.extraction` (scope_classifier,
ontology_extractor, internal_derivation) against a locally-configured
`StreamingLLMProvider` and model-family adapter, with strict parsing and
parse-and-repair so a failed job is reported as an error -- never cached
as an empty extraction.

Stage 2 cuts the utterance into units with `structure_splitter`. A turn with
list items or table rows takes the list frame path: one frame call decides
what each group of items states, item calls read the items a few at a time,
and `list_assembly` builds the payload in code. A turn without them takes the
r6 list pass, where each batch of units is one Stage 2 call; an utterance
without structure is one batch, sent exactly as before the list pass existed.
The r6 pass is also the list path's fallback, for the whole turn when its
frame fails and for one item when that item's call does.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import importlib
import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from types import ModuleType
from typing import Any, TypeVar

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
    DerivationOut,
    ExtractionPayload,
    ExtractRequest,
    ExtractResponse,
    ResultStamps,
    ScopeOut,
    TimingsMs,
)
from backend.extraction_service.adapters import ModelFamilyAdapter
from backend.extraction_service.list_schemas import (
    LIST_FRAME_OUTPUT_SCHEMA,
    LIST_ITEM_OUTPUT_SCHEMA,
    list_frame_output_schema,
    list_item_output_schema,
)
from backend.extraction_service.schemas import (
    DERIVATION_OUTPUT_SCHEMA,
    EXTRACTION_OUTPUT_SCHEMA,
    SCOPE_OUTPUT_SCHEMA,
)
from backend.extraction_service.settings import ConstrainedMode, ServiceSettings
from backend.knowledge.extraction.internal_derivation import (
    parse_derivation_output,
    render_derivation_messages,
)
from backend.knowledge.extraction.internal_prompts import (
    INTERNAL_DERIVATION_SYSTEM_PROMPT,
    INTERNAL_DERIVATION_USER_TEMPLATE,
)
from backend.knowledge.extraction.list_assembly import assemble
from backend.knowledge.extraction.list_frames import (
    FrameOutput,
    ItemOutput,
    ListGroup,
    group_units,
    item_slots,
    parse_frame_output,
    parse_item_output,
    render_frame_input,
)
from backend.knowledge.extraction.list_prompts import (
    LIST_FRAME_REPAIR_INSTRUCTION,
    LIST_FRAME_USER_TEMPLATE,
    LIST_ITEM_REPAIR_INSTRUCTION,
    LIST_ITEM_USER_TEMPLATE,
    LIST_REFUSAL_TEMPLATE,
    LIST_SYSTEM_PROMPT,
    list_repair_text,
    render_frames_for_items,
)
from backend.knowledge.extraction.ontology_extractor import (
    NO_PRIOR_CONTEXT,
    parse_extraction_output,
    render_extraction_messages,
)
from backend.knowledge.extraction.preprocessor import PreProcessedInput, PreProcessor
from backend.knowledge.extraction.prompts import EXTRACTION_SYSTEM_PROMPT, EXTRACTION_USER_TEMPLATE
from backend.knowledge.extraction.scope_classifier import (
    SCOPE_CLASSIFIER_SYSTEM_PROMPT,
    SCOPE_USER_TEMPLATE,
    parse_scope_output,
    render_scope_messages,
)
from backend.knowledge.extraction.structure_splitter import (
    InputUnit,
    batch_units,
    merge_payloads,
    render_batch,
    render_unit,
    split_structured,
)
from backend.knowledge.version_stamps import EXTRACTION_VERSION
from backend.llm.models import LLMRequest
from backend.llm.provider import StreamingLLMProvider

logger = logging.getLogger(__name__)

# What a Stage 2 call's strict parser returns: a payload for an r6 batch, a
# frame output for a frame call, the item outputs for an item call.
_Parsed = TypeVar("_Parsed")

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

# Output token limits, one per LLM-calling stage. Named so the run identity
# can record them: a limit that cuts an output short changes what the stage
# returns as surely as a prompt edit does.
SCOPE_MAX_TOKENS = 96
EXTRACT_MAX_TOKENS = 2048
DERIVE_MAX_TOKENS = 400

# The Stage 2 list pass's limits: the most units one call extracts, and the
# most earlier units of the same message a later batch sees as context. Small
# batches keep a long list inside EXTRACT_MAX_TOKENS; the earlier units let an
# item refer back to one in an earlier batch. They live here, beside the token
# limits, so the run identity reads them from the module that applies them.
LIST_BATCH_MAX_UNITS: int = 4
LIST_CONTEXT_MAX_UNITS: int = 8

# The list frame path's limits: the frame and item calls' output limits, the
# most items one item call extracts, and the most characters a frame or item
# user message may hold. The r6 limits above stay, because its fallback uses
# them.
LIST_FRAME_MAX_TOKENS: int = 2048
LIST_ITEM_MAX_TOKENS: int = 2048
LIST_ITEM_BATCH_MAX_UNITS: int = 4
# 14,336 input tokens (ctx 16,384 less the 2,048-token output), less the list
# system prompt at its 42,000-character cap (about 11,667 tokens at the main
# prompt's measured 3.6 characters a token) and a 256-token framing reserve,
# leave about 2,413 tokens; at 3 characters a token, a conservative figure for
# history text, that is about 7,200 characters. Estimates, not a tokenizer count.
LIST_USER_MAX_CHARS: int = 7200

_LIST_INPUT_TOKENS = 14336
_LIST_FRAMING_RESERVE_TOKENS = 256

# The warnings the list path adds; the assembly adds its own.
_LIST_FRAME_FALLBACK = "list_frame_fallback"
_LIST_ITEM_FALLBACK = "list_item_fallback"
_LIST_FRAME_OVER_BUDGET = "list_frame_over_budget"
_LIST_HISTORY_TRIMMED = "list_history_trimmed"
_LIST_REPAIR_SKIPPED = "list_repair_skipped"


def list_call_fits(messages: Sequence[Mapping[str, str]]) -> bool:
    """Whether a list call sending `messages` stays inside the model's input budget.

    The system prompt counts at 3.6 characters a token, the main prompt's
    measured ratio, and every other message at 3, the conservative figure for
    history and model output, plus the framing reserve. A repair resends each
    failed output with its instruction, so the estimate runs over the whole
    list the call would send.
    """
    system = sum(len(m["content"]) for m in messages if m["role"] == "system")
    other = sum(len(m["content"]) for m in messages if m["role"] != "system")
    tokens = -(-system * 10 // 36) + -(-other // 3) + _LIST_FRAMING_RESERVE_TOKENS
    return tokens <= _LIST_INPUT_TOKENS


# The code the run identity covers (`code_sha256` in
# `backend.knowledge.extraction.run_identity`): this module, the adapters and
# both schema modules, every `backend.knowledge.extraction` module imported
# above, whose renderers, strict parsers and list assembly Stages 1.5, 2 and 9
# call, `canonical_id`, which the list assembly builds Metric ids with, the
# validator, whose relationship constraints the list parsers check templates
# and item entities against, and the provider module the service builds. Whole
# modules rather than functions, so a helper a covered function calls is
# covered too; an edit elsewhere in a covered module costs one `CODE_REVISIONS`
# line, never a silent miss.
RUN_IDENTITY_CODE: tuple[ModuleType, ...] = tuple(
    importlib.import_module(name)
    for name in (
        __name__,
        "backend.extraction_service.adapters",
        "backend.extraction_service.schemas",
        "backend.extraction_service.list_schemas",
        "backend.knowledge.extraction.canonical_id",
        "backend.knowledge.extraction.internal_derivation",
        "backend.knowledge.extraction.internal_prompts",
        "backend.knowledge.extraction.list_assembly",
        "backend.knowledge.extraction.list_frames",
        "backend.knowledge.extraction.list_prompts",
        "backend.knowledge.extraction.ontology_extractor",
        "backend.knowledge.extraction.preprocessor",
        "backend.knowledge.extraction.prompts",
        "backend.knowledge.extraction.scope_classifier",
        "backend.knowledge.extraction.structure_splitter",
        "backend.knowledge.extraction.validator",
        "backend.llm.llama_server_provider",
    )
)

# The backend modules this module, the adapters and the list modules import
# from that the run identity does not hash, each with the reason. The coverage
# test in tests/unit/knowledge/extraction/test_run_identity.py fails on an
# import in neither tuple.
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
        "backend.knowledge.ontologies",
        "its type and predicate lists reach the model only through the list schemas, "
        "which prompt_sha256 covers; its relationship constraints reach the list parsers "
        "through the validator, which code_sha256 covers; and ontology_version, a "
        "component already, moves with any change to either",
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
    """A Stage 2 output stopped at its call's `max_tokens` (`finish_reason == "length"`).

    Internal to Stage 2: the batch is split and retried, and only a single
    unit cut short leaves the engine, as `UpstreamLLMError`.
    """


class _OutputUnparsableError(UpstreamLLMError):
    """A Stage 2 output stayed unparsable after the attempts its call was allowed.

    A subclass of `UpstreamLLMError`, so a caller that does not fall back sees
    what it saw before the attempt loop was shared between stages.

    Attributes:
        repair_skipped: True when the attempts ended early because the next
            repair would not fit the model's input (`fits` returned False).
    """

    def __init__(self, message: str, *, repair_skipped: bool = False) -> None:
        super().__init__(message)
        self.repair_skipped = repair_skipped


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
    """What one turn's Stage 2 calls add up to, across its batches."""

    calls: int = 0
    attempts: int = 0
    elapsed_ms: float = 0.0
    payloads: list[ExtractionPayload] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _ListTurn:
    """What every list call of one structured turn shares.

    Attributes:
        pre_processed: The turn's pre-processed input, its full history included.
        units: The turn's units, from `split_structured`.
        groups: The turn's groups, from `group_units`.
        prose: The turn's prose units, from `group_units`.
        items: `(item id, position in units)` of every item, in turn order.
        system: The list system prompt message, its reference date filled in.
        render: The turn as `render_frame_input` writes it.
    """

    pre_processed: PreProcessedInput
    units: tuple[InputUnit, ...]
    groups: tuple[ListGroup, ...]
    prose: tuple[InputUnit, ...]
    items: tuple[tuple[str, int], ...]
    system: dict[str, str]
    render: str

    @property
    def subject_scope(self) -> str:
        return self.pre_processed.metadata.get("subject_scope", "unknown")

    @property
    def item_ids(self) -> tuple[str, ...]:
        """Every item id of the turn, group by group, as the render lists them."""
        return tuple(item for group in self.groups for item in group.item_ids)


def _list_turn(
    pre_processed: PreProcessedInput,
    units: Sequence[InputUnit],
    groups: Sequence[ListGroup],
    prose: Sequence[InputUnit],
) -> _ListTurn:
    positions = {id(unit): index for index, unit in enumerate(units)}
    items = sorted(
        (
            (item_id, positions[id(unit)])
            for group in groups
            for item_id, unit in zip(group.item_ids, group.items, strict=True)
        ),
        key=lambda item: item[1],
    )
    system = LIST_SYSTEM_PROMPT.format(
        reference_date=pre_processed.reference_date.strftime("%Y-%m-%d")
    )
    return _ListTurn(
        pre_processed=pre_processed,
        units=tuple(units),
        groups=tuple(groups),
        prose=tuple(prose),
        items=tuple(items),
        system={"role": "system", "content": system},
        render=render_frame_input(groups, prose),
    )


def _context_text(context: Sequence[str]) -> str:
    return "\n".join(context) if context else NO_PRIOR_CONTEXT


def _fit_user_message(
    build: Callable[[Sequence[str]], str], context: Sequence[str]
) -> tuple[str | None, tuple[str, ...], bool]:
    """Build a list user message, dropping the oldest history while it is over the limit.

    Returns:
        The message, or None when it is still over `LIST_USER_MAX_CHARS` with
        no history left; the history it kept; and whether any was dropped.
    """
    kept = tuple(context)
    message = build(kept)
    trimmed = False
    while len(message) > LIST_USER_MAX_CHARS and kept:
        kept = kept[1:]
        trimmed = True
        message = build(kept)
    return (message if len(message) <= LIST_USER_MAX_CHARS else None), kept, trimmed


def _warn_once(run: _Stage2Run, warning: str) -> None:
    if warning not in run.warnings:
        run.warnings.append(warning)


def _batch_input(
    pre_processed: PreProcessedInput, units: Sequence[InputUnit], start: int, size: int
) -> PreProcessedInput:
    """The turn's pre-processed input, narrowed to the batch `units[start:start + size]`.

    The batch's rendered units replace the utterance. The nearest earlier units
    of the same message, at most `LIST_CONTEXT_MAX_UNITS`, follow the turn's
    history as `[user]:` lines, the form the PreProcessor gives a user line,
    so an item can refer back to one in an earlier batch. The first batch adds
    none, so a turn without structure renders exactly as it did before.
    """
    text = render_batch(units[start : start + size])
    earlier = units[max(0, start - LIST_CONTEXT_MAX_UNITS) : start]
    return dataclasses.replace(
        pre_processed,
        original_text=text,
        resolved_text=text,
        conversation_context=[
            *pre_processed.conversation_context,
            *(f"[user]: {render_unit(unit)}" for unit in earlier),
        ],
    )


def _compute_prompt_sha256(adapter: ModelFamilyAdapter) -> str:
    """Hash the prompt text and output schemas sent to the model, plus the adapter identity.

    `prompt_sha256` is a per-result provenance stamp: the record of what the
    model was asked. It is not the epoch's identity, which is
    `extraction_version` plus the composed `model_hash`
    (`ExtractionDispatcher._epoch_matches`).

    Covered: the raw, unformatted templates of all three stages (the scope
    system prompt and user template, the extraction system prompt and user
    template, the derivation system prompt and user template), the Stage 2
    empty-context placeholder, the Stage 2 repair instruction sent on a
    retry, the three constrained-decoding output schemas (serialised with
    sorted keys; hashed whether or not the active mode sends them, which
    only makes the stamp stricter), the list path's system prompt, its two
    user templates, two repair instructions and the refusal line each list
    repair ends with, its two output schemas, and the adapter's own
    name/version. The list schemas are serialised in declared key order,
    because the grammar emits keys in that order, so a reorder changes
    decoding. They are the static supersets every call's built schema
    narrows; the builders are code, which `code_sha256` covers.

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
    """
    material = "\x1f".join(
        [
            SCOPE_CLASSIFIER_SYSTEM_PROMPT,
            SCOPE_USER_TEMPLATE,
            EXTRACTION_SYSTEM_PROMPT,
            EXTRACTION_USER_TEMPLATE,
            NO_PRIOR_CONTEXT,
            INTERNAL_DERIVATION_SYSTEM_PROMPT,
            INTERNAL_DERIVATION_USER_TEMPLATE,
            _REPAIR_INSTRUCTION,
            json.dumps(SCOPE_OUTPUT_SCHEMA, sort_keys=True),
            json.dumps(EXTRACTION_OUTPUT_SCHEMA, sort_keys=True),
            json.dumps(DERIVATION_OUTPUT_SCHEMA, sort_keys=True),
            LIST_SYSTEM_PROMPT,
            LIST_FRAME_USER_TEMPLATE,
            LIST_ITEM_USER_TEMPLATE,
            LIST_FRAME_REPAIR_INSTRUCTION,
            LIST_ITEM_REPAIR_INSTRUCTION,
            LIST_REFUSAL_TEMPLATE,
            json.dumps(LIST_FRAME_OUTPUT_SCHEMA, sort_keys=False),
            json.dumps(LIST_ITEM_OUTPUT_SCHEMA, sort_keys=False),
            adapter.name,
            adapter.version,
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


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
        """
        self._llm = llm
        self._adapter = adapter
        self._settings = settings
        self._run_identity = run_identity
        self._preprocessor = PreProcessor()
        self._prompt_sha256 = _compute_prompt_sha256(adapter)

    @property
    def adapter_name(self) -> str:
        """The active model-family adapter's name (e.g. "gptoss")."""
        return self._adapter.name

    @property
    def run_identity(self) -> str | None:
        """The run identity this engine stamps on every result."""
        return self._run_identity

    async def run(self, req: ExtractRequest) -> ExtractResponse:
        """Run the full stage sequence for one job and build its response.

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

        scope_out, scope_ms = await self._run_scope(pre_processed, warnings)
        stage2 = await self._run_extraction(pre_processed)
        payload, merge_warnings = merge_payloads(stage2.payloads)
        warnings.extend(stage2.warnings)
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

    async def _invoke(self, request: LLMRequest):
        """Invoke the LLM under the configured timeout.

        Raises:
            TimeoutError: The call exceeded `settings.llm_timeout_seconds`.
            openai.APIError / LLMConnectionError / LLMResponseError: The
                transport or provider failed.
        """
        return await asyncio.wait_for(
            self._llm.invoke(request), timeout=self._settings.llm_timeout_seconds
        )

    async def _run_scope(
        self, pre_processed: PreProcessedInput, warnings: list[str]
    ) -> tuple[ScopeOut, float | None]:
        """Stage 1.5. Never raises -- any failure degrades to "unknown"."""
        if not self._settings.scope_enabled:
            pre_processed.metadata["subject_scope"] = "unknown"
            pre_processed.metadata["subject_scope_confidence"] = 0.0
            return ScopeOut(label="unknown", confidence=0.0), None

        start = time.perf_counter()
        request = self._build_request(
            messages=render_scope_messages(pre_processed),
            max_tokens=SCOPE_MAX_TOKENS,
            schema=SCOPE_OUTPUT_SCHEMA,
        )
        request = self._adapter.prepare(request, stage="scope")

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

    async def _run_extraction(self, pre_processed: PreProcessedInput) -> _Stage2Run:
        """Stage 2: the list frame path for a structured turn, else the r6 list pass.

        A turn is structured when `group_units` finds a group of list items
        or table rows in it.

        Raises:
            ExtractionTimeoutError: An attempt exceeded the LLM timeout.
            UpstreamLLMError: The LLM call failed, or an r6 batch's output
                stayed unparsable or was cut short on a single unit.
        """
        units = split_structured(pre_processed.original_text)
        run = _Stage2Run()
        groups, prose = group_units(units)
        if groups:
            await self._run_list_frames(_list_turn(pre_processed, units, groups, prose), run)
        else:
            await self._run_list_pass(pre_processed, units, run)
        return run

    async def _run_list_pass(
        self, pre_processed: PreProcessedInput, units: Sequence[InputUnit], run: _Stage2Run
    ) -> None:
        """The r6 list pass: one call sequence per batch of at most `LIST_BATCH_MAX_UNITS` units.

        Batches run in order, one at a time, and any batch failing fails the
        turn: there are no partial results.

        Raises:
            ExtractionTimeoutError: An attempt exceeded the LLM timeout.
            UpstreamLLMError: The LLM call failed, or every attempt's
                output stayed unparsable.
        """
        start = 0
        for batch in batch_units(units, LIST_BATCH_MAX_UNITS):
            await self._extract_units(pre_processed, units, start, len(batch), run, run.payloads)
            start += len(batch)

    async def _run_list_frames(self, turn: _ListTurn, run: _Stage2Run) -> None:
        """The list frame path: one frame call, then item calls, then assembly.

        The frame message is fitted to `LIST_USER_MAX_CHARS` by dropping the
        oldest history; with no history left and still over, the turn takes
        the r6 list pass before any list call. A frame cut short or still
        unparsable sends the turn through the r6 list pass too. The item calls
        start from the history the frame message kept.

        Each call sends the schema built for it, so most shapes the parsers
        refuse cannot be decoded: the frame call's from the turn's groups and
        items, each item call's from the item slots the parsed frame gives its
        batch. A refused output is repaired with the stage's repair text and a
        line naming the refusal, because at temperature 0 a repair that names
        no reason tends to repeat the output.

        Adds the assembled payload and the path's warnings to `run`.

        Raises:
            ExtractionTimeoutError: A list or fallback call exceeded the LLM timeout.
            UpstreamLLMError: A list or fallback call failed, or a fallback's
                output stayed unparsable.
        """
        pre_processed = turn.pre_processed
        utterance = pre_processed.original_text

        def frame_message(context: Sequence[str]) -> str:
            return LIST_FRAME_USER_TEMPLATE.format(
                context=_context_text(context),
                subject_scope=turn.subject_scope,
                utterance=utterance,
                render=turn.render,
            )

        user, kept, trimmed = _fit_user_message(frame_message, pre_processed.conversation_context)
        if trimmed:
            _warn_once(run, _LIST_HISTORY_TRIMMED)
        if user is None:
            logger.warning("List frame message over %d characters; r6 pass", LIST_USER_MAX_CHARS)
            run.warnings.append(_LIST_FRAME_OVER_BUDGET)
            await self._run_list_pass(pre_processed, turn.units, run)
            return

        schema = list_frame_output_schema(
            [group.gid for group in turn.groups],
            {group.gid for group in turn.groups if group.parent is not None},
            turn.item_ids,
        )
        try:
            frame = await self._list_attempts(
                user,
                turn,
                run,
                max_tokens=LIST_FRAME_MAX_TOKENS,
                schema=schema,
                parse=lambda raw: parse_frame_output(raw, turn.groups),
                repair=lambda error: list_repair_text(LIST_FRAME_REPAIR_INSTRUCTION, error),
            )
        except (_OutputTruncatedError, _OutputUnparsableError) as exc:
            if isinstance(exc, _OutputUnparsableError) and exc.repair_skipped:
                run.warnings.append(_LIST_REPAIR_SKIPPED)
            logger.warning("List frame call failed (%s); r6 pass for the turn", exc)
            run.warnings.append(_LIST_FRAME_FALLBACK)
            await self._run_list_pass(pre_processed, turn.units, run)
            return

        frames = render_frames_for_items(frame, turn.groups)
        outputs: list[ItemOutput] = []
        fallbacks: dict[str, ExtractionPayload] = {}
        for start in range(0, len(turn.items), LIST_ITEM_BATCH_MAX_UNITS):
            batch = turn.items[start : start + LIST_ITEM_BATCH_MAX_UNITS]
            await self._list_items(turn, kept, frame, frames, batch, run, outputs, fallbacks)

        payload, warnings = assemble(frame, outputs, turn.groups, turn.prose, utterance, fallbacks)
        run.payloads.append(payload)
        run.warnings.extend(warnings)

    async def _list_items(
        self,
        turn: _ListTurn,
        context: Sequence[str],
        frame: FrameOutput,
        frames: str,
        batch: Sequence[tuple[str, int]],
        run: _Stage2Run,
        outputs: list[ItemOutput],
        fallbacks: dict[str, ExtractionPayload],
    ) -> None:
        """Extract one batch of items, halving it while a call is cut short or unparsable.

        A single item still cut short or unparsable, or a batch whose message
        is over `LIST_USER_MAX_CHARS` with no history left, takes the item
        fallback: the r6 pass over that item's unit alone. The parser checks
        each item's ends against `frame` and the outputs parsed so far.

        Raises:
            ExtractionTimeoutError: A call exceeded the LLM timeout.
            UpstreamLLMError: A call failed, or a fallback's output stayed
                unparsable or was cut short.
        """
        ids = [item_id for item_id, _position in batch]

        def item_message(kept: Sequence[str]) -> str:
            return LIST_ITEM_USER_TEMPLATE.format(
                context=_context_text(kept),
                subject_scope=turn.subject_scope,
                utterance=turn.pre_processed.original_text,
                render=turn.render,
                frames=frames,
                items=", ".join(ids),
            )

        user, _kept, trimmed = _fit_user_message(item_message, context)
        if trimmed:
            _warn_once(run, _LIST_HISTORY_TRIMMED)
        if user is None:
            logger.warning(
                "List item message over %d characters; item fallback", LIST_USER_MAX_CHARS
            )
            for item in batch:
                await self._item_fallback(turn, item, run, fallbacks)
            return

        earlier = {output.item: output for output in outputs}
        schema = list_item_output_schema(
            item_slots(frame, turn.groups, ids),
            [entity.key for entity in frame.frame_entities],
            turn.item_ids,
        )
        try:
            outputs.extend(
                await self._list_attempts(
                    user,
                    turn,
                    run,
                    max_tokens=LIST_ITEM_MAX_TOKENS,
                    schema=schema,
                    parse=lambda raw: parse_item_output(raw, ids, frame, turn.groups, earlier),
                    repair=lambda error: list_repair_text(LIST_ITEM_REPAIR_INSTRUCTION, error),
                )
            )
            return
        except (_OutputTruncatedError, _OutputUnparsableError) as exc:
            if isinstance(exc, _OutputUnparsableError) and exc.repair_skipped:
                run.warnings.append(_LIST_REPAIR_SKIPPED)
            logger.warning("List item call on %d items failed: %s", len(batch), exc)

        if len(batch) == 1:
            await self._item_fallback(turn, batch[0], run, fallbacks)
            return
        half = (len(batch) + 1) // 2
        for part in (batch[:half], batch[half:]):
            await self._list_items(turn, context, frame, frames, part, run, outputs, fallbacks)

    async def _item_fallback(
        self,
        turn: _ListTurn,
        item: tuple[str, int],
        run: _Stage2Run,
        fallbacks: dict[str, ExtractionPayload],
    ) -> None:
        """Send one item's unit through the r6 pass, with r6's context, for its payload.

        Raises:
            As `_extract_units`.
        """
        item_id, position = item
        run.warnings.append(_LIST_ITEM_FALLBACK)
        payloads: list[ExtractionPayload] = []
        await self._extract_units(turn.pre_processed, turn.units, position, 1, run, payloads)
        (fallbacks[item_id],) = payloads

    async def _extract_units(
        self,
        pre_processed: PreProcessedInput,
        units: Sequence[InputUnit],
        start: int,
        size: int,
        run: _Stage2Run,
        out: list[ExtractionPayload],
    ) -> None:
        """Extract `units[start:start + size]`, halving the batch while its output is cut short.

        An output that stops at `EXTRACT_MAX_TOKENS` is never repaired: the
        repair retry would ask for the same output under the same limit. The
        batch is split in two and each half extracted in turn, each seeing
        the units before it as context, down to single units. Each payload is
        appended to `out`, in unit order.

        Raises:
            UpstreamLLMError: A single unit's output was still cut short, or
                a call failed or stayed unparsable.
            ExtractionTimeoutError: An attempt exceeded the LLM timeout.
        """
        batch_input = _batch_input(pre_processed, units, start, size)
        try:
            payload = await self._extract_batch(batch_input, run)
        except _OutputTruncatedError as exc:
            if size == 1:
                raise UpstreamLLMError(
                    f"extraction output truncated at max_tokens ({EXTRACT_MAX_TOKENS}) "
                    "on a single unit"
                ) from exc
            half = (size + 1) // 2
            logger.warning(
                "Extraction output cut short at max_tokens on %d units; retrying as %d and %d",
                size,
                half,
                size - half,
            )
            await self._extract_units(pre_processed, units, start, half, run, out)
            await self._extract_units(pre_processed, units, start + half, size - half, run, out)
            return
        out.append(payload)

    async def _extract_batch(
        self, batch_input: PreProcessedInput, run: _Stage2Run
    ) -> ExtractionPayload:
        """One batch's Stage 2 calls: strict parse with repair-retry up to `max_attempts`.

        Adds its calls, attempts and time to `run`.

        Raises:
            ExtractionTimeoutError: An attempt exceeded the LLM timeout.
            UpstreamLLMError: The LLM call failed, or every attempt's
                output stayed unparsable.
        """
        start = time.perf_counter()
        try:
            return await self._extract_attempts(
                render_extraction_messages(batch_input),
                run,
                max_tokens=EXTRACT_MAX_TOKENS,
                schema=EXTRACTION_OUTPUT_SCHEMA,
                parse=_strict_extraction_payload,
                repair=_REPAIR_INSTRUCTION,
            )
        finally:
            run.elapsed_ms += (time.perf_counter() - start) * 1000

    async def _list_attempts(
        self,
        user: str,
        turn: _ListTurn,
        run: _Stage2Run,
        *,
        max_tokens: int,
        schema: dict,
        parse: Callable[[str], _Parsed],
        repair: str | Callable[[ExtractionValidationError], str],
    ) -> _Parsed:
        """One frame or item call sequence, timed into `run`, its repairs budgeted.

        Raises:
            As `_extract_attempts`; a skipped repair raises
            `_OutputUnparsableError` with `repair_skipped` set.
        """
        start = time.perf_counter()
        try:
            return await self._extract_attempts(
                [turn.system, {"role": "user", "content": user}],
                run,
                max_tokens=max_tokens,
                schema=schema,
                parse=parse,
                repair=repair,
                fits=list_call_fits,
            )
        finally:
            run.elapsed_ms += (time.perf_counter() - start) * 1000

    async def _extract_attempts(
        self,
        messages: list[dict],
        run: _Stage2Run,
        *,
        max_tokens: int,
        schema: dict,
        parse: Callable[[str], _Parsed],
        repair: str | Callable[[ExtractionValidationError], str],
        fits: Callable[[Sequence[Mapping[str, str]]], bool] | None = None,
    ) -> _Parsed:
        """One Stage 2 call sequence: strict parse with repair-retry up to `max_attempts`.

        Each stage passes its own limit, schema, strict parser and repair text,
        so the r6 batches and the list calls share the attempt count, the
        timeout and the error mapping. Every attempt passes `stage="extract"`,
        so the adapter treats them alike.

        Args:
            messages: The first attempt's messages; each repair appends the
                failed output and `repair`.
            run: The turn's Stage 2 totals, which every attempt adds to.
            max_tokens: The call's output limit.
            schema: The output schema sent in `schema` constrained mode.
            parse: The strict parser; an `ExtractionValidationError` from it
                gets the repair retry.
            repair: The instruction sent after a failed output, or a function
                of the parser's error that returns it. The list calls pass a
                function, so the repair names the refusal; the r6 batches pass
                the fixed `_REPAIR_INSTRUCTION`, so their requests stay as
                they were.
            fits: When given, called with the whole message list before each
                repair attempt; a False return ends the attempts as an
                unparsable output, with `repair_skipped` set. The r6 batches
                pass none, so their repair is unbudgeted, as before.

        Raises:
            ExtractionTimeoutError: An attempt exceeded the LLM timeout.
            UpstreamLLMError: The LLM call failed.
            _OutputUnparsableError: Every attempt's output stayed unparsable,
                or the next repair would not fit.
            _OutputTruncatedError: An output stopped at `max_tokens`.
        """
        last_error: Exception | None = None

        for attempt in range(1, self._settings.max_attempts + 1):
            if attempt > 1 and fits is not None and not fits(messages):
                logger.warning(
                    "Extraction repair %d/%d skipped: it would not fit the input budget",
                    attempt,
                    self._settings.max_attempts,
                )
                raise _OutputUnparsableError(
                    f"Extraction output unparsable; repair attempt {attempt} would not fit",
                    repair_skipped=True,
                ) from last_error
            run.calls += 1
            run.attempts = max(run.attempts, attempt)
            request = self._build_request(messages=messages, max_tokens=max_tokens, schema=schema)
            request = self._adapter.prepare(request, stage="extract")

            try:
                response = await self._invoke(request)
            except TimeoutError as exc:
                raise ExtractionTimeoutError(
                    f"Extraction LLM call timed out after {self._settings.llm_timeout_seconds}s"
                    f" (attempt {attempt}/{self._settings.max_attempts})"
                ) from exc
            except _LLM_TRANSPORT_ERRORS as exc:
                raise UpstreamLLMError(f"Extraction LLM call failed: {exc}") from exc

            if response.finish_reason == "length":
                raise _OutputTruncatedError(
                    f"Extraction output reached max_tokens={max_tokens}"
                    f" (attempt {attempt}/{self._settings.max_attempts})"
                )
            raw = self._adapter.final_text(response)
            try:
                parsed = parse(raw)
            except ExtractionValidationError as exc:
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
                    {"role": "user", "content": repair if isinstance(repair, str) else repair(exc)},
                ]
                continue

            return parsed

        raise _OutputUnparsableError(
            f"Extraction output unparsable after {self._settings.max_attempts} attempts"
        ) from last_error

    async def _run_derivation(
        self, req: ExtractRequest, warnings: list[str]
    ) -> tuple[DerivationOut | None, float | None]:
        """Stage 9. Only runs when `req.derivation` is set. Never raises."""
        if req.derivation is None:
            return None, None

        start = time.perf_counter()
        request = self._build_request(
            messages=render_derivation_messages(
                utterance=req.utterance,
                assistant_response=req.derivation.assistant_response,
                signal_types=req.derivation.signal_types,
                matched_patterns=req.derivation.matched_patterns,
                existing_internal_entities=req.derivation.existing_internal_entities,
            ),
            max_tokens=DERIVE_MAX_TOKENS,
            schema=DERIVATION_OUTPUT_SCHEMA,
        )
        request = self._adapter.prepare(request, stage="derive")

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
