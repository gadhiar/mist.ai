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

from backend.errors import LLMConnectionError, LLMResponseError
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
    InvalidRequestError,
    UpstreamLLMError,
    parse_recorded_at,
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


class ContextSizeSource(Protocol):
    """Contract for reading the configured llama-server's context window."""

    async def ctx_size(self) -> int | None:
        """Return llama-server's `n_ctx`, or None when it cannot be read."""
        ...


class LlamaPropsContextSize:
    """Default ContextSizeSource: reads `n_ctx` from llama-server's `GET /props`.

    The field path is `default_generation_settings.n_ctx`, the same one
    `LlamaServerProvider.server_context_size` and
    `backend.factories._probe_llama_server_n_ctx` parse.

    The first successful read is cached for the life of the process, so
    repeated `/v1/info` calls make one `/props` request between them. A
    failed read is NOT cached: it returns None and the next call tries
    again, so a service that starts before its llama-server has loaded
    reports the real value once llama-server is up, rather than None
    forever. A lock makes concurrent first calls share one request.
    """

    def __init__(self, http_client: httpx.AsyncClient, base_url: str, timeout: float = 2.0) -> None:
        self._client = http_client
        self._base_url = base_url
        self._timeout = timeout
        self._cached: int | None = None
        self._lock = asyncio.Lock()

    async def ctx_size(self) -> int | None:
        """Return the cached `n_ctx`, fetching it first if nothing is cached."""
        if self._cached is not None:
            return self._cached
        async with self._lock:
            if self._cached is not None:
                return self._cached
            try:
                self._cached = await self._fetch()
            except (LLMConnectionError, LLMResponseError) as exc:
                logger.warning("llama-server context size unavailable; ctx_size=null: %s", exc)
                return None
            return self._cached

    async def _fetch(self) -> int:
        """GET `{base_url}/props` and return `default_generation_settings.n_ctx`.

        Raises:
            LLMConnectionError: The request failed at the transport level.
            LLMResponseError: A non-2xx status, a non-JSON body, or a body
                without an int at `default_generation_settings.n_ctx`.
        """
        url = f"{self._base_url}/props"
        try:
            response = await self._client.get(url, timeout=self._timeout)
        except httpx.HTTPError as exc:
            raise LLMConnectionError(f"GET {url} failed: {exc!r}") from exc
        if not 200 <= response.status_code < 300:
            raise LLMResponseError(f"GET {url} returned HTTP {response.status_code}")
        try:
            data = response.json()
        except ValueError as exc:
            raise LLMResponseError(f"GET {url} returned a non-JSON body") from exc
        settings_block = data.get("default_generation_settings") if isinstance(data, dict) else None
        n_ctx = settings_block.get("n_ctx") if isinstance(settings_block, dict) else None
        # bool is an int subclass; a JSON true/false is not a context size.
        if not isinstance(n_ctx, int) or isinstance(n_ctx, bool):
            raise LLMResponseError(f"GET {url} carried no int at default_generation_settings.n_ctx")
        return n_ctx


class _IdempotencyCache:
    """LRU of completed `ExtractResponse`s, keyed on `job_id`, with single-flight.

    A `job_id` already in the LRU returns the stored response without
    re-running `factory`. A `job_id` currently being computed is joined
    rather than re-run -- concurrent duplicate requests for the same job
    produce exactly one underlying run. A run that raises is never cached,
    so a resubmitted job_id after a failure tries again rather than
    replaying the failure forever. `maxsize=0` retains nothing: duplicates
    still share an in-flight run, but a finished job runs again.

    Concurrency relies on the event loop, not a lock: `get_or_run` does its
    lookups and registers a new run with no `await` in between, and the run
    task's done callback stores the result and THEN un-registers the run,
    also synchronously. So at every point a duplicate can observe, the job
    is either in flight or stored -- never neither, which would start a
    second run.

    Each caller awaits the shared run through `asyncio.shield`: a caller
    cancelled mid-run (its client disconnected) stops waiting, but the run
    carries on for the other callers and is still stored when it finishes.
    """

    def __init__(self, maxsize: int) -> None:
        """Create an empty cache.

        Raises:
            ValueError: `maxsize` is negative.
        """
        if maxsize < 0:
            raise ValueError(f"maxsize must be >= 0, got {maxsize}")
        self._maxsize = maxsize
        self._results: OrderedDict[str, ExtractResponse] = OrderedDict()
        self._in_flight: dict[str, asyncio.Task[ExtractResponse]] = {}

    def knows(self, job_id: str) -> bool:
        """True when `job_id` has a stored result or a run in flight."""
        return job_id in self._results or job_id in self._in_flight

    async def get_or_run(
        self, job_id: str, factory: Callable[[], Awaitable[ExtractResponse]]
    ) -> ExtractResponse:
        """Return the job's response, running `factory` only if nothing has it.

        Raises:
            Whatever the shared run raised (every joined caller sees it), or
            `asyncio.CancelledError` when THIS caller is cancelled.
        """
        cached = self._results.get(job_id)
        if cached is not None:
            self._results.move_to_end(job_id)
            return cached

        run = self._in_flight.get(job_id)
        if run is None:
            run = asyncio.ensure_future(factory())
            self._in_flight[job_id] = run
            run.add_done_callback(lambda done: self._on_run_done(job_id, done))
        return await asyncio.shield(run)

    def _on_run_done(self, job_id: str, run: asyncio.Task[ExtractResponse]) -> None:
        # Synchronous: store, then un-register, with no await in between.
        if not run.cancelled() and run.exception() is None:
            self._store(job_id, run.result())
        if self._in_flight.get(job_id) is run:
            del self._in_flight[job_id]

    def _store(self, job_id: str, result: ExtractResponse) -> None:
        if self._maxsize == 0:
            return
        self._results[job_id] = result
        self._results.move_to_end(job_id)
        while len(self._results) > self._maxsize:
            self._results.popitem(last=False)


def _error_response(code: ErrorCode, message: str) -> JSONResponse:
    envelope = error_envelope(code, message)
    return JSONResponse(
        status_code=_ERROR_HTTP_STATUS[code],
        content=envelope.model_dump(mode="json"),
    )


def _effective_constrained_mode(settings: ServiceSettings, adapter_name: str) -> str | None:
    """The constrained-decoding mode the engine's LLM calls use.

    Mirrors `ExtractionEngine._resolve_constrained_mode`, which reads
    `settings.constrained_mode or adapter.default_constrained_mode`: a truthy
    settings override wins, else the adapter's default. The test is truthiness,
    not `is None`, because `compose.host.yml` forwards
    `EXTRACTION_CONSTRAINED_MODE=${EXTRACTION_CONSTRAINED_MODE:-}`, so an unset
    host variable arrives as an empty string and the engine falls through to
    the adapter default. Adapter defaults are dataclass field defaults, so a
    fresh `get_adapter(name)` carries the same value as the engine's instance.
    None only when neither is known -- no override and an adapter name
    `get_adapter` does not recognise.
    """
    if settings.constrained_mode:
        return settings.constrained_mode
    try:
        return get_adapter(adapter_name).default_constrained_mode
    except ValueError:
        return None


def create_app(
    settings: ServiceSettings,
    engine: ExtractionEngine,
    health_probe: HealthProbe,
    ctx_size_source: ContextSizeSource,
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

        try:
            parse_recorded_at(req.recorded_at)
        except InvalidRequestError as exc:
            return _error_response(ErrorCode.CONTRACT_MISMATCH, str(exc))

        # A job already stored or in flight needs no model: answer it (or
        # join it) even while llama-server reloads, rather than 503 a result
        # this process already holds or is about to have.
        if not cache.knows(req.job_id):
            probe_result = await health_probe.check()
            if probe_result.status != "ok":
                return _error_response(
                    ErrorCode.MODEL_LOADING,
                    f"llama-server not ready (status={probe_result.status})",
                )

        request_start = time.perf_counter()
        try:
            response = await cache.get_or_run(req.job_id, lambda: engine.run(req))
        except InvalidRequestError as exc:
            # Checked above; kept so the engine's own refusal is never a 500.
            return _error_response(ErrorCode.CONTRACT_MISMATCH, str(exc))
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
        """Report this service's identity, version stamps and serving config."""
        return InfoResponse(
            contract_version=CONTRACT_VERSION,
            extraction_version=EXTRACTION_VERSION,
            model_hash=settings.model_hash,
            model_file=settings.model_file,
            llama_cpp_build=settings.llama_cpp_build,
            adapter=engine.adapter_name,
            location_label=settings.location_label,
            constrained_mode=_effective_constrained_mode(settings, engine.adapter_name),
            reasoning_effort=settings.reasoning_effort,
            temperature=settings.temperature,
            ctx_size=await ctx_size_source.ctx_size(),
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
    http_client = httpx.AsyncClient()
    health_probe = LlamaHealthProbe(http_client=http_client, base_url=settings.llm_base_url)
    ctx_size_source = LlamaPropsContextSize(http_client=http_client, base_url=settings.llm_base_url)
    return create_app(settings, engine, health_probe, ctx_size_source)
