"""FastAPI application for the extraction service.

`create_app()` is pure dependency injection -- settings, engine, and a
health probe are all constructed by the caller and passed in, so unit
tests can wire a `FakeGraphExecutor`-style fake llama-server (an ASGI app
behind `httpx.ASGITransport`) without touching a real network.
`build_app_from_env()` is the one function that wires the real
`LlamaServerProvider` and a real health probe from `EXTRACTION_*` env vars.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal, Protocol

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from backend.extraction_contract.models import (
    CONTRACT_MAJOR,
    CONTRACT_VERSION,
    ErrorCode,
    ExtractRequest,
    ExtractResponse,
    HealthResponse,
    InfoResponse,
    error_envelope,
    is_compatible,
)
from backend.extraction_service.adapters import get_adapter
from backend.extraction_service.engine import (
    ExtractionEngine,
    ExtractionTimeoutError,
    UpstreamLLMError,
)
from backend.extraction_service.settings import ServiceSettings
from backend.knowledge.version_stamps import EXTRACTION_VERSION
from backend.llm.llama_server_provider import LlamaServerProvider

logger = logging.getLogger(__name__)

_ERROR_HTTP_STATUS: dict[ErrorCode, int] = {
    ErrorCode.EPOCH_MISMATCH: 409,
    ErrorCode.CONTRACT_MISMATCH: 422,
    ErrorCode.MODEL_LOADING: 503,
    ErrorCode.UPSTREAM_LLM: 502,
    ErrorCode.TIMEOUT: 504,
}


@dataclass(frozen=True, slots=True)
class HealthProbeResult:
    """One health check's outcome."""

    status: Literal["ok", "loading", "degraded"]
    llm_reachable: bool


class HealthProbe(Protocol):
    """Contract for checking the configured llama-server's readiness."""

    async def check(self) -> HealthProbeResult:
        """Return this probe's current read of the configured llama-server."""
        ...


class LlamaHealthProbe:
    """Default HealthProbe: GETs `{base_url}/health` via an injected httpx client.

    200 -> ok. 503 -> loading (llama-server is up but still loading the
    model). Anything else, or a transport failure, -> degraded with
    `llm_reachable=False`.
    """

    def __init__(self, http_client: httpx.AsyncClient, base_url: str) -> None:
        self._client = http_client
        self._base_url = base_url

    async def check(self) -> HealthProbeResult:
        """GET `{base_url}/health` and classify the result."""
        try:
            response = await self._client.get(f"{self._base_url}/health")
        except httpx.HTTPError:
            return HealthProbeResult(status="degraded", llm_reachable=False)
        if response.status_code == 200:
            return HealthProbeResult(status="ok", llm_reachable=True)
        if response.status_code == 503:
            return HealthProbeResult(status="loading", llm_reachable=True)
        return HealthProbeResult(status="degraded", llm_reachable=False)


class _IdempotencyCache:
    """LRU of completed `ExtractResponse`s, keyed on `job_id`, with single-flight.

    A `job_id` already in the LRU returns the stored response without
    re-running `factory`. A `job_id` currently being computed by another
    caller is awaited rather than re-run -- concurrent duplicate requests
    for the same job produce exactly one underlying run. A run that
    raises is never cached, so a resubmitted job_id after a failure tries
    again rather than replaying the failure forever.
    """

    def __init__(self, maxsize: int) -> None:
        self._maxsize = maxsize
        self._results: OrderedDict[str, ExtractResponse] = OrderedDict()
        self._in_flight: dict[str, asyncio.Future[ExtractResponse]] = {}
        self._lock = asyncio.Lock()

    async def get_or_run(
        self, job_id: str, factory: Callable[[], Awaitable[ExtractResponse]]
    ) -> ExtractResponse:
        async with self._lock:
            cached = self._results.get(job_id)
            if cached is not None:
                self._results.move_to_end(job_id)
                return cached

            future = self._in_flight.get(job_id)
            owns = future is None
            if owns:
                future = asyncio.ensure_future(factory())
                self._in_flight[job_id] = future

        try:
            result = await future
        finally:
            if owns:
                async with self._lock:
                    self._in_flight.pop(job_id, None)

        async with self._lock:
            self._results[job_id] = result
            self._results.move_to_end(job_id)
            while len(self._results) > self._maxsize:
                self._results.popitem(last=False)
        return result


def _error_response(code: ErrorCode, message: str) -> JSONResponse:
    envelope = error_envelope(code, message)
    return JSONResponse(
        status_code=_ERROR_HTTP_STATUS[code],
        content=envelope.model_dump(mode="json"),
    )


def create_app(
    settings: ServiceSettings,
    engine: ExtractionEngine,
    health_probe: HealthProbe,
) -> FastAPI:
    """Build the extraction service's FastAPI app from injected collaborators."""
    app = FastAPI(title="MIST extraction service")
    cache = _IdempotencyCache(maxsize=settings.idempotency_cache_size)
    started_at = time.monotonic()

    @app.exception_handler(RequestValidationError)
    async def _on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _error_response(
            ErrorCode.CONTRACT_MISMATCH,
            f"Request body failed validation: {exc.errors()}",
        )

    @app.post("/v1/extract")
    async def extract(req: ExtractRequest):
        if not is_compatible(req.contract_version):
            return _error_response(
                ErrorCode.CONTRACT_MISMATCH,
                f"Unsupported contract_version {req.contract_version!r}; "
                f"this service speaks contract major {CONTRACT_MAJOR}",
            )

        if (
            req.expect.extraction_version != EXTRACTION_VERSION
            or req.expect.model_hash != settings.model_hash
        ):
            return _error_response(
                ErrorCode.EPOCH_MISMATCH,
                "expect stamps "
                f"(extraction_version={req.expect.extraction_version!r}, "
                f"model_hash={req.expect.model_hash!r}) do not match this "
                f"service's epoch (extraction_version={EXTRACTION_VERSION!r}, "
                f"model_hash={settings.model_hash!r})",
            )

        probe_result = await health_probe.check()
        if probe_result.status != "ok":
            return _error_response(
                ErrorCode.MODEL_LOADING,
                f"llama-server not ready (status={probe_result.status})",
            )

        request_start = time.perf_counter()
        try:
            response = await cache.get_or_run(req.job_id, lambda: engine.run(req))
        except ExtractionTimeoutError as exc:
            logger.info(
                "job failed request_id=%s job_id=%s event_id=%s turn_id=%s outcome=timeout: %s",
                req.request_id,
                req.job_id,
                req.event_id,
                req.turn_id,
                exc,
            )
            return _error_response(ErrorCode.TIMEOUT, str(exc))
        except UpstreamLLMError as exc:
            logger.info(
                "job failed request_id=%s job_id=%s event_id=%s turn_id=%s "
                "outcome=upstream_llm: %s",
                req.request_id,
                req.job_id,
                req.event_id,
                req.turn_id,
                exc,
            )
            return _error_response(ErrorCode.UPSTREAM_LLM, str(exc))
        except ValidationError as exc:
            # Defence in depth: the engine's strict parse rejects bad model
            # output before any contract model sees it, so this means a
            # response model rejected a value the parse let through. Report
            # it as the model's failure, in the envelope, never a bare 500.
            logger.error(
                "job failed request_id=%s job_id=%s event_id=%s turn_id=%s "
                "outcome=upstream_llm (response model rejected the result): %s",
                req.request_id,
                req.job_id,
                req.event_id,
                req.turn_id,
                exc,
            )
            return _error_response(
                ErrorCode.UPSTREAM_LLM,
                f"The model's output could not be built into a response: {exc}",
            )

        duration_ms = (time.perf_counter() - request_start) * 1000
        logger.info(
            "job complete request_id=%s job_id=%s event_id=%s turn_id=%s "
            "scope_ms=%s extract_ms=%.1f derive_ms=%s attempts=%d "
            "outcome=extracted request_duration_ms=%.1f",
            req.request_id,
            req.job_id,
            req.event_id,
            req.turn_id,
            response.timings_ms.scope,
            response.timings_ms.extract,
            response.timings_ms.derive,
            response.attempts,
            duration_ms,
        )
        return response

    @app.get("/v1/health")
    async def health() -> HealthResponse:
        """Report the configured llama-server's readiness and this process's uptime."""
        result = await health_probe.check()
        return HealthResponse(
            status=result.status,
            llm_reachable=result.llm_reachable,
            uptime_s=time.monotonic() - started_at,
        )

    @app.get("/v1/info")
    async def info() -> InfoResponse:
        """Report this service's configured identity and version stamps."""
        return InfoResponse(
            contract_version=CONTRACT_VERSION,
            extraction_version=EXTRACTION_VERSION,
            model_hash=settings.model_hash,
            model_file=settings.model_file,
            llama_cpp_build=settings.llama_cpp_build,
            adapter=engine.adapter_name,
            location_label=settings.location_label,
        )

    return app


def build_app_from_env() -> FastAPI:
    """Wire the real `LlamaServerProvider` and health probe from env vars."""
    settings = ServiceSettings.from_env()
    llm = LlamaServerProvider(base_url=settings.llm_base_url, model=settings.model_file)
    adapter = get_adapter(
        settings.adapter_name,
        reasoning_effort=settings.reasoning_effort,
        reasoning_budget_tokens=settings.reasoning_budget_tokens,
    )
    engine = ExtractionEngine(llm=llm, adapter=adapter, settings=settings)
    health_probe = LlamaHealthProbe(http_client=httpx.AsyncClient(), base_url=settings.llm_base_url)
    return create_app(settings, engine, health_probe)
