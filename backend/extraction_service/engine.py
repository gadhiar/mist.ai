"""ExtractionEngine: orchestrates Stages 1, 1.5, 2, and 9 for one job.

This is the service's whole reason to exist: reuse the render/parse
functions from `backend.knowledge.extraction` (scope_classifier,
ontology_extractor, internal_derivation) against a locally-configured
`StreamingLLMProvider` and model-family adapter, with strict parsing and
parse-and-repair so a failed job is reported as an error -- never cached
as an empty extraction.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from datetime import datetime

from openai import APIError

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
from backend.knowledge.extraction.ontology_extractor import (
    parse_extraction_output,
    render_extraction_messages,
)
from backend.knowledge.extraction.preprocessor import PreProcessedInput, PreProcessor
from backend.knowledge.extraction.prompts import EXTRACTION_SYSTEM_PROMPT, EXTRACTION_USER_TEMPLATE
from backend.knowledge.extraction.scope_classifier import (
    SCOPE_CLASSIFIER_SYSTEM_PROMPT,
    parse_scope_output,
    render_scope_messages,
)
from backend.knowledge.version_stamps import EXTRACTION_VERSION
from backend.llm.models import LLMRequest
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
    "Your previous output was not valid JSON, or was missing the required "
    '"entities"/"relationships" list fields. Return ONLY a JSON object '
    'matching {"entities": [...], "relationships": [...]}. No prose, no '
    "markdown code fences, no explanation."
)


class ExtractionServiceError(MistError):
    """Base class for extraction-service-specific failures.

    Subclasses map to the service's HTTP error taxonomy
    (`backend.extraction_contract.models.ErrorCode`) at the `app.py` layer.
    """


class UpstreamLLMError(ExtractionServiceError):
    """The configured llama-server call failed, or repair retries were exhausted.

    Maps to HTTP 502 / `ErrorCode.UPSTREAM_LLM`.
    """


class ExtractionTimeoutError(ExtractionServiceError):
    """An LLM call exceeded `ServiceSettings.llm_timeout_seconds`.

    Maps to HTTP 504 / `ErrorCode.TIMEOUT`.
    """


def _compute_prompt_sha256(adapter: ModelFamilyAdapter) -> str:
    """Hash the scope/extraction/derivation prompt templates plus the adapter identity.

    Computed once at engine construction (not per request) since none of
    its inputs vary per job -- the raw, unformatted prompt templates and
    the adapter's own name/version.
    """
    material = "".join(
        [
            SCOPE_CLASSIFIER_SYSTEM_PROMPT,
            EXTRACTION_SYSTEM_PROMPT,
            EXTRACTION_USER_TEMPLATE,
            INTERNAL_DERIVATION_SYSTEM_PROMPT,
            INTERNAL_DERIVATION_USER_TEMPLATE,
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
    ) -> None:
        """Initialize the engine.

        Args:
            llm: LLM provider for the service's own llama-server.
            adapter: Model-family adapter matching `llm`'s model.
            settings: Service configuration.
        """
        self._llm = llm
        self._adapter = adapter
        self._settings = settings
        self._preprocessor = PreProcessor()
        self._prompt_sha256 = _compute_prompt_sha256(adapter)

    @property
    def adapter_name(self) -> str:
        """The active model-family adapter's name (e.g. "gptoss")."""
        return self._adapter.name

    async def run(self, req: ExtractRequest) -> ExtractResponse:
        """Run the full stage sequence for one job and build its response.

        Raises:
            UpstreamLLMError: The extraction LLM call failed, or the model
                never produced parseable output within `max_attempts`.
            ExtractionTimeoutError: An LLM call exceeded
                `settings.llm_timeout_seconds`.
        """
        total_start = time.perf_counter()
        warnings: list[str] = []

        reference_date = datetime.fromisoformat(req.recorded_at)
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
        payload, extract_ms, attempts = await self._run_extraction(pre_processed)
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
            ),
            timings_ms=TimingsMs(
                scope=scope_ms,
                extract=extract_ms,
                derive=derive_ms,
                total=total_ms,
            ),
            attempts=attempts,
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
            max_tokens=96,
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

    async def _run_extraction(
        self, pre_processed: PreProcessedInput
    ) -> tuple[ExtractionPayload, float, int]:
        """Stage 2. Strict parse with repair-retry up to `max_attempts`.

        Raises:
            ExtractionTimeoutError: An attempt exceeded the LLM timeout.
            UpstreamLLMError: The LLM call failed, or every attempt's
                output stayed unparsable.
        """
        start = time.perf_counter()
        messages = render_extraction_messages(pre_processed)
        last_error: Exception | None = None

        for attempt in range(1, self._settings.max_attempts + 1):
            request = self._build_request(
                messages=messages, max_tokens=2048, schema=EXTRACTION_OUTPUT_SCHEMA
            )
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

            raw = self._adapter.final_text(response)
            try:
                parsed = parse_extraction_output(raw, strict=True)
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
                    {"role": "user", "content": _REPAIR_INSTRUCTION},
                ]
                continue

            elapsed = (time.perf_counter() - start) * 1000
            return (
                ExtractionPayload(
                    entities=parsed["entities"], relationships=parsed["relationships"]
                ),
                elapsed,
                attempt,
            )

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
        request = self._build_request(
            messages=render_derivation_messages(
                utterance=req.utterance,
                assistant_response=req.derivation.assistant_response,
                signal_types=req.derivation.signal_types,
                matched_patterns=req.derivation.matched_patterns,
                existing_internal_entities=req.derivation.existing_internal_entities,
            ),
            max_tokens=400,
            schema=DERIVATION_OUTPUT_SCHEMA,
        )
        request = self._adapter.prepare(request, stage="derive")

        try:
            response = await self._invoke(request)
            raw = self._adapter.final_text(response)
            operations = parse_derivation_output(raw, strict=True)
        except (TimeoutError, ExtractionValidationError, *_LLM_TRANSPORT_ERRORS) as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning("Derivation failed (%s): %s", type(exc).__name__, exc)
            warnings.append("derivation_failed")
            return None, elapsed

        elapsed = (time.perf_counter() - start) * 1000
        return DerivationOut(operations=operations), elapsed
