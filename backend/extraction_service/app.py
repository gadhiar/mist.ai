"""FastAPI application for the extraction service.

`create_app()` is pure dependency injection -- settings, engine, and a
health probe are all constructed by the caller and passed in, so unit
tests can wire a `FakeGraphExecutor`-style fake llama-server (an ASGI app
behind `httpx.ASGITransport`) without touching a real network.
`build_app_from_env()` is the one function that wires the real
`LlamaServerProvider` and a real health probe from `EXTRACTION_*` env vars;
`build_app()` is the wiring it runs, given the provider and HTTP client, and
the place the run identity is computed, once per process.

The serving identity, by contrast, is computed on every `/v1/info` and
`/v1/extract` call, because one of its components, llama-server's context
size, is observed through `/props` for each call rather than configured.
"""

from __future__ import annotations

import asyncio
import dataclasses
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
from pydantic import BaseModel, ValidationError

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
from backend.extraction_service.adapters import ModelFamilyAdapter, get_adapter
from backend.extraction_service.engine import (
    LIST_BATCH_MAX_UNITS,
    LIST_CONTEXT_MAX_UNITS,
    PROMPT_RESERVE_TOKENS,
    RUN_IDENTITY_CODE,
    ExtractionEngine,
    ExtractionServiceError,
    ExtractionTimeoutError,
    InvalidRequestError,
    UpstreamLLMError,
    _compute_prompt_sha256,
    effective_cap,
    parse_recorded_at,
)
from backend.extraction_service.settings import SERVER_ENV_PREFIX, ServiceSettings
from backend.knowledge.extraction.run_identity import (
    RunIdentityComponents,
    code_sha256,
    compute_run_identity,
)
from backend.knowledge.extraction.serving_identity import (
    REQUIRED_SERVER_KEYS,
    ServingIdentityComponents,
    components_dict,
    compute_serving_identity,
    diff_components,
)
from backend.knowledge.ontologies.pack import Pack
from backend.knowledge.ontologies.registry import active_pack
from backend.knowledge.version_stamps import EXTRACTION_VERSION, ONTOLOGY_VERSION
from backend.llm.llama_server_provider import LlamaServerProvider
from backend.llm.provider import StreamingLLMProvider

logger = logging.getLogger(__name__)

_ERROR_HTTP_STATUS: dict[ErrorCode, int] = {
    ErrorCode.EPOCH_MISMATCH: 409,
    ErrorCode.CONTRACT_MISMATCH: 422,
    ErrorCode.MODEL_LOADING: 503,
    ErrorCode.UPSTREAM_LLM: 502,
    ErrorCode.TIMEOUT: 504,
    ErrorCode.CANCELLED: 499,
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


@dataclass(frozen=True, slots=True)
class PropsRead:
    """One read of llama-server's context window.

    Attributes:
        n_ctx: The context size `/props` reported, or None when it could not
            be read.
        error: Why `n_ctx` is None; None when it was read.
    """

    n_ctx: int | None
    error: str | None = None


class ContextSizeSource(Protocol):
    """Contract for reading the configured llama-server's context window."""

    async def read(self) -> PropsRead:
        """Read llama-server's `n_ctx` now, or say why it cannot be read."""
        ...


class LlamaPropsContextSize:
    """Default ContextSizeSource: reads `n_ctx` from llama-server's `GET /props`.

    The field path is `default_generation_settings.n_ctx`, the same one
    `LlamaServerProvider.server_context_size` and
    `backend.factories._probe_llama_server_n_ctx` parse.

    Every read fetches: nothing is kept once a fetch has finished, so a
    llama-server restarted with another context size is seen on the next
    call. The context size is part of the serving identity, which keys an
    extraction epoch, so a stale value would stamp results with a setup that
    no longer runs. The cost is one GET per `/v1/info` and `/v1/extract`
    call, milliseconds against a job of seconds.

    Concurrent reads share the fetch in flight, failed or not, so N callers
    at once make one request bounded by `timeout`. Each caller waits through
    `asyncio.shield`: a caller cancelled mid-fetch stops waiting, and the
    fetch carries on for the others.
    """

    def __init__(self, http_client: httpx.AsyncClient, base_url: str, timeout: float = 2.0) -> None:
        self._client = http_client
        self._base_url = base_url
        self._timeout = timeout
        self._in_flight: asyncio.Task[PropsRead] | None = None

    async def read(self) -> PropsRead:
        """Join the fetch in flight, or start one when none is."""
        fetch = self._in_flight
        if fetch is None or fetch.done():
            fetch = asyncio.ensure_future(self._read_once())
            self._in_flight = fetch
        return await asyncio.shield(fetch)

    async def _read_once(self) -> PropsRead:
        try:
            return PropsRead(n_ctx=await self._fetch())
        except (LLMConnectionError, LLMResponseError) as exc:
            logger.warning("llama-server context size unavailable; ctx_size=null: %s", exc)
            return PropsRead(n_ctx=None, error=str(exc))

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


class JobCancelledError(ExtractionServiceError):
    """The job's run was cancelled before it finished; nothing is stored for it.

    Maps to HTTP 499 / `ErrorCode.CANCELLED`.
    """


class ClientGoneError(ExtractionServiceError):
    """This caller's client disconnected before the job's run finished.

    The caller's own condition, not the job's: the run carries on while
    another caller waits on it.
    """


CancelOutcome = Literal["cancelled", "finished", "unknown"]


class CancelResponse(BaseModel):
    """Response body of `POST /v1/jobs/{job_id}/cancel`.

    Local to the service: the contract package has no model for it yet, and
    no client in this repository calls the route.

    Attributes:
        job_id: The job the request named.
        status: `cancelled` when a run was in flight and is now cancelled;
            `finished` when the job already has a stored result, which is
            kept; `unknown` when this process holds neither.
    """

    job_id: str
    status: CancelOutcome


async def _client_gone(request: Request) -> None:
    """Return once the client of `request` has disconnected.

    Read after the request body was consumed, so the next ASGI message is
    the disconnect, whenever it comes. This is how the handler learns that
    its client left under a server that keeps the handler running.
    """
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            return


@dataclass(slots=True)
class _SharedRun:
    """One job's run in flight, the callers waiting on it, and whether it was cancelled."""

    task: asyncio.Task[ExtractResponse]
    waiters: int = 0
    cancelled: bool = False


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
    also synchronously. So at every point a duplicate can observe, a job
    whose run was not cancelled is either in flight or stored -- never
    neither, which would start a second run.

    A run ends early in two ways, and both store nothing and un-register the
    run at once, so the same `job_id` sent again starts a run of its own
    even while the cancelled one is still unwinding:

    - `cancel` cancels the job: every caller waiting on its run gets
      `JobCancelledError`.
    - The last caller waiting on a run leaves, because it was cancelled or
      its client disconnected. A caller that leaves while another still
      waits takes nothing with it: the run carries on and is stored when it
      finishes.
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
        self._in_flight: dict[str, _SharedRun] = {}

    def knows(self, job_id: str) -> bool:
        """True when `job_id` has a stored result or a run in flight."""
        return job_id in self._results or job_id in self._in_flight

    async def get_or_run(
        self,
        job_id: str,
        factory: Callable[[], Awaitable[ExtractResponse]],
        *,
        gone: Callable[[], Awaitable[object]] | None = None,
    ) -> ExtractResponse:
        """Return the job's response, running `factory` only if nothing has it.

        Args:
            job_id: The job to answer.
            factory: Starts the job's run when none is stored or in flight.
            gone: Awaited beside the run when given; it returns when this
                caller's client has disconnected.

        Raises:
            JobCancelledError: The job was cancelled while this caller waited.
            ClientGoneError: `gone` returned before the run finished.
            asyncio.CancelledError: THIS caller was cancelled.
            Exception: Whatever the shared run raised; every caller waiting
                on it sees the same error.
        """
        cached = self._results.get(job_id)
        if cached is not None:
            self._results.move_to_end(job_id)
            return cached

        run = self._in_flight.get(job_id)
        if run is None:
            run = _SharedRun(task=asyncio.ensure_future(factory()))
            self._in_flight[job_id] = run
            run.task.add_done_callback(lambda _done, run=run: self._on_run_done(job_id, run))

        watcher = None if gone is None else asyncio.ensure_future(gone())
        run.waiters += 1
        try:
            # `asyncio.wait` cancels neither future when this caller is
            # cancelled, so the run outlives a caller that leaves.
            await asyncio.wait(
                {run.task} if watcher is None else {run.task, watcher},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if run.cancelled or run.task.cancelled():
                raise JobCancelledError(f"job {job_id!r} was cancelled")
            if run.task.done():
                return run.task.result()
            raise ClientGoneError(f"the client waiting on job {job_id!r} disconnected")
        finally:
            if watcher is not None:
                watcher.cancel()
            run.waiters -= 1
            if run.waiters == 0 and not run.task.done():
                self._drop(job_id, run)

    def cancel(self, job_id: str) -> CancelOutcome:
        """Cancel `job_id`'s run if one is in flight, and say what was found.

        Returns:
            `cancelled` when a run was in flight: it is cancelled, nothing
            is stored for it, and every caller waiting on it gets
            `JobCancelledError`. `finished` when the job has a stored
            result, which is kept. `unknown` otherwise: a job never seen, a
            job already cancelled, or a finished job this cache no longer
            holds.
        """
        run = self._in_flight.get(job_id)
        if run is not None:
            self._drop(job_id, run)
            return "cancelled"
        return "finished" if job_id in self._results else "unknown"

    def _drop(self, job_id: str, run: _SharedRun) -> None:
        # Synchronous: mark, un-register, then cancel. The mark keeps a run
        # that finishes anyway out of the store and tells its waiters.
        run.cancelled = True
        if self._in_flight.get(job_id) is run:
            del self._in_flight[job_id]
        run.task.cancel()

    def _on_run_done(self, job_id: str, run: _SharedRun) -> None:
        # Synchronous: store, then un-register, with no await in between.
        task = run.task
        # Read the exception even for a dropped run, so the loop never
        # reports it as one nobody retrieved.
        succeeded = not task.cancelled() and task.exception() is None
        if succeeded and not run.cancelled:
            self._store(job_id, task.result())
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


def run_identity_components(
    settings: ServiceSettings, adapter: ModelFamilyAdapter, pack: Pack | None = None
) -> RunIdentityComponents:
    """The run identity's components for a service running `settings` with `adapter`.

    Every value is read from where the engine reads it: the prompt hash from
    the same function the engine stamps with, each stage's token cap as the
    engine sizes it for `adapter` (`effective_cap`), the list-pass limits from
    the engine's constants, and the constrained mode as the engine resolves it
    (`settings.constrained_mode or adapter.default_constrained_mode`).

    `pack` is the pack the engine extracts with, whose assembled prompt and
    generated schema the prompt hash covers. None takes the registry's active
    pack, as the engine does when it is given none; `build_app` passes the
    pack it gives the engine. The pack's own two hashes are no component:
    they are stamped beside the run identity, not into it.
    """
    return RunIdentityComponents(
        prompt_sha256=_compute_prompt_sha256(adapter, turn_plan=settings.turn_plan, pack=pack),
        code_sha256=code_sha256(RUN_IDENTITY_CODE),
        ontology_version=ONTOLOGY_VERSION,
        extraction_version=EXTRACTION_VERSION,
        adapter=adapter.name,
        model_hash=settings.model_hash,
        llama_cpp_build=settings.llama_cpp_build,
        constrained_mode=settings.constrained_mode or adapter.default_constrained_mode,
        reasoning_effort=settings.reasoning_effort,
        reasoning_budget_tokens=settings.reasoning_budget_tokens,
        temperature=settings.temperature,
        max_attempts=settings.max_attempts,
        scope_enabled=settings.scope_enabled,
        scope_max_tokens=effective_cap(adapter, "scope"),
        extract_max_tokens=effective_cap(adapter, "extract"),
        derive_max_tokens=effective_cap(adapter, "derive"),
        list_batch_max_units=LIST_BATCH_MAX_UNITS,
        list_context_max_units=LIST_CONTEXT_MAX_UNITS,
    )


def serving_identity_components(
    settings: ServiceSettings, engine: ExtractionEngine, n_ctx: int
) -> ServingIdentityComponents:
    """The serving identity's components for `engine` running `settings`, at `n_ctx`.

    The request profile comes from the engine's own stage builders, so it is
    what each stage sends; the adapter is the one the engine runs. Every other
    value is read from `settings`, and `n_ctx` is what `/props` reported for
    this call.
    """
    return ServingIdentityComponents(
        adapter=engine.adapter_name,
        model_file=settings.model_file,
        llama_cpp_build=settings.llama_cpp_build,
        max_attempts=settings.max_attempts,
        scope_enabled=settings.scope_enabled,
        requests=engine.request_profile(),
        server=tuple(sorted(settings.server_config)),
        n_ctx=n_ctx,
        turn_plan=settings.turn_plan,
    )


def _serving_identity_error(
    settings: ServiceSettings, props: PropsRead, extract_cap: int
) -> str | None:
    """Why the service has no serving identity for this call, or None when it has one.

    Beside the declarations, the extract cap plus `PROMPT_RESERVE_TOKENS` must
    fit the observed context. A deployment whose longest extract call could
    exceed its context (medium's caps at low's 16,384 context) gets no
    identity, and `/v1/info` names the shortfall, the same way a declared
    context size that differs from the observed one is reported.
    """
    declared = dict(settings.server_config)
    problems: list[str] = []
    missing = [key for key in REQUIRED_SERVER_KEYS if not declared.get(key)]
    if missing:
        names = ", ".join(SERVER_ENV_PREFIX + key.upper() for key in missing)
        problems.append(f"declared llama-server settings missing: {names}")
    if props.n_ctx is None:
        problems.append(f"llama-server /props unreadable: {props.error}")
    elif declared.get("ctx_size"):
        ctx_size = declared["ctx_size"]
        try:
            matches = int(ctx_size) == props.n_ctx
        except ValueError:
            problems.append(f"declared ctx_size {ctx_size!r} is not an integer")
        else:
            if not matches:
                problems.append(f"declared ctx_size {ctx_size} != observed n_ctx {props.n_ctx}")
    if props.n_ctx is not None and extract_cap + PROMPT_RESERVE_TOKENS > props.n_ctx:
        problems.append(
            f"extract cap {extract_cap} + prompt reserve {PROMPT_RESERVE_TOKENS} "
            f"> observed n_ctx {props.n_ctx}"
        )
    return "; ".join(problems) or None


@dataclass(frozen=True, slots=True)
class _ServingView:
    """The serving identity of one call, as `/v1/info` reports it."""

    props: PropsRead
    identity: str | None
    components: dict[str, str | int | float | bool | None] | None
    error: str | None


class _SeenIdentities:
    """The components of each serving identity this process has reported, newest last.

    A request names its expected identity by hash alone, so the service can
    say which settings differ only for an identity it computed itself: in one
    process the declarations are fixed, and the observed `n_ctx` is what
    moves, when llama-server restarts with another context size. Bounded,
    because each new context size adds an entry.
    """

    def __init__(self, maxsize: int = 32) -> None:
        self._maxsize = maxsize
        self._components: OrderedDict[str, dict[str, str | int | float | bool | None]] = (
            OrderedDict()
        )

    def remember(
        self, identity: str, components: dict[str, str | int | float | bool | None]
    ) -> None:
        """Record `identity`'s components, evicting the oldest past `maxsize`."""
        self._components[identity] = components
        self._components.move_to_end(identity)
        while len(self._components) > self._maxsize:
            self._components.popitem(last=False)

    def components_of(self, identity: str) -> dict[str, str | int | float | bool | None]:
        """The components of `identity`, or an empty mapping when it was never seen here."""
        return self._components.get(identity, {})


def create_app(
    settings: ServiceSettings,
    engine: ExtractionEngine,
    health_probe: HealthProbe,
    ctx_size_source: ContextSizeSource,
    run_identity: RunIdentityComponents | None = None,
) -> FastAPI:
    """Build the extraction service's FastAPI app from injected collaborators.

    Args:
        settings: Service configuration.
        engine: The engine that runs each job.
        health_probe: Reads llama-server's readiness.
        ctx_size_source: Reads llama-server's context window, once per
            `/v1/info` and `/v1/extract` call, for `ctx_size` and the serving
            identity.
        run_identity: The components of the run identity `engine` stamps,
            reported on `/v1/info`. None reports no run identity.

    Raises:
        ValueError: `run_identity` is given and its hash differs from the one
            `engine` stamps, so `/v1/info` would describe other results than
            the service returns.
    """
    identity = None if run_identity is None else compute_run_identity(run_identity)
    if run_identity is not None and engine.run_identity != identity:
        raise ValueError(
            f"engine stamps run identity {engine.run_identity!r}, but the components "
            f"given hash to {identity!r}"
        )
    identity_components = None if run_identity is None else dataclasses.asdict(run_identity)
    app = FastAPI(title="MIST extraction service")
    cache = _IdempotencyCache(maxsize=settings.idempotency_cache_size)
    seen = _SeenIdentities()
    started_at = time.monotonic()

    async def serving_view() -> _ServingView:
        # Read for every call: the context size is observed, never cached.
        props = await ctx_size_source.read()
        error = _serving_identity_error(settings, props, engine.stage_cap("extract"))
        if props.n_ctx is None:
            return _ServingView(props=props, identity=None, components=None, error=error)
        components = serving_identity_components(settings, engine, props.n_ctx)
        flat = components_dict(components)
        identity = None
        if error is None:
            identity = compute_serving_identity(components)
            seen.remember(identity, flat)
        return _ServingView(props=props, identity=identity, components=flat, error=error)

    @app.exception_handler(RequestValidationError)
    async def _on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _error_response(
            ErrorCode.CONTRACT_MISMATCH,
            f"Request body failed validation: {exc.errors()}",
        )

    @app.post("/v1/extract")
    async def extract(req: ExtractRequest, request: Request):
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

        # The serving identity is computed for this job, from a fresh /props
        # read. Without a context size there is no identity to check or stamp,
        # and the job waits for llama-server like a loading model does. This
        # runs before the stored-job replay below, so a job this process
        # already holds is also answered 503 while /props is unreadable.
        # Whether /props answers during a model load is not verified.
        view = await serving_view()
        if view.props.n_ctx is None:
            return _error_response(
                ErrorCode.MODEL_LOADING,
                f"llama-server context size unavailable: {view.error}",
            )
        expected = req.expect.serving_identity
        if expected is not None and expected != view.identity:
            return _error_response(
                ErrorCode.EPOCH_MISMATCH,
                _serving_mismatch_message(expected, view, seen.components_of(expected)),
            )

        # A job already stored or in flight needs no model: answer it (or
        # join it) even while the health probe reports llama-server not
        # ready, rather than 503 a result this process already holds or is
        # about to have. The /props read above still applies to it.
        if not cache.knows(req.job_id):
            probe_result = await health_probe.check()
            if probe_result.status != "ok":
                return _error_response(
                    ErrorCode.MODEL_LOADING,
                    f"llama-server not ready (status={probe_result.status})",
                )

        request_start = time.perf_counter()
        try:
            response = await cache.get_or_run(
                req.job_id,
                lambda: engine.run(req, serving_identity=view.identity),
                gone=lambda: _client_gone(request),
            )
        except (JobCancelledError, ClientGoneError) as exc:
            # A client that left reads no answer; the envelope is what the
            # handler returns either way.
            outcome = "cancelled" if isinstance(exc, JobCancelledError) else "client_gone"
            logger.info(
                "job ended early request_id=%s job_id=%s event_id=%s turn_id=%s outcome=%s",
                req.request_id,
                req.job_id,
                req.event_id,
                req.turn_id,
                outcome,
            )
            return _error_response(ErrorCode.CANCELLED, str(exc))
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

    @app.post("/v1/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> CancelResponse:
        """Cancel the job's run if one is in flight; always 200, the body says what was found.

        Every request waiting on a cancelled run is answered
        `ErrorCode.CANCELLED`, nothing is stored for the job, and the same
        `job_id` runs when it is sent again.
        """
        outcome = cache.cancel(job_id)
        logger.info("cancel requested job_id=%s outcome=%s", job_id, outcome)
        return CancelResponse(job_id=job_id, status=outcome)

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
        view = await serving_view()
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
            ctx_size=view.props.n_ctx,
            run_identity=identity,
            run_identity_components=identity_components,
            serving_identity=view.identity,
            serving_identity_components=view.components,
            serving_identity_error=view.error,
            pack_prompt_hash=engine.pack.pack_prompt_hash,
            pack_rails_hash=engine.pack.pack_rails_hash,
        )

    return app


def _serving_mismatch_message(
    expected: str,
    view: _ServingView,
    expected_components: dict[str, str | int | float | bool | None],
) -> str:
    """The 409 message: both identities, why this one is None if it is, and the diff.

    The diff is against the expected identity's components when this process
    computed that identity; otherwise every component shows `<missing>` on
    the expected side, and the backend diffs against the components it bound.
    """
    parts = [
        f"expect.serving_identity {expected!r} does not match this service's "
        f"serving identity {view.identity!r}"
    ]
    if view.error is not None:
        parts.append(f"this service has no serving identity: {view.error}")
    if not expected_components:
        parts.append(
            f"{expected!r} was not computed by this process, so its components are unknown"
        )
    diff = diff_components(expected_components, view.components or {})
    if diff:
        parts.append("differs in: " + "; ".join(diff))
    return ". ".join(parts)


def build_app(
    settings: ServiceSettings, llm: StreamingLLMProvider, http_client: httpx.AsyncClient
) -> FastAPI:
    """Wire the service around `llm`, computing its run identity once.

    The identity is computed here, before the engine exists, and handed both
    to the engine, which stamps it on every result, and to `create_app`,
    which reports it with its components on `/v1/info`.

    Args:
        settings: Service configuration.
        llm: The provider for the service's own llama-server.
        http_client: The client the health probe and context-size read use.
    """
    adapter = get_adapter(
        settings.adapter_name,
        reasoning_effort=settings.reasoning_effort,
        reasoning_budget_tokens=settings.reasoning_budget_tokens,
    )
    pack = active_pack()
    components = run_identity_components(settings, adapter, pack)
    engine = ExtractionEngine(
        llm=llm,
        adapter=adapter,
        settings=settings,
        run_identity=compute_run_identity(components),
        pack=pack,
    )
    health_probe = LlamaHealthProbe(http_client=http_client, base_url=settings.llm_base_url)
    ctx_size_source = LlamaPropsContextSize(http_client=http_client, base_url=settings.llm_base_url)
    return create_app(settings, engine, health_probe, ctx_size_source, run_identity=components)


def build_app_from_env() -> FastAPI:
    """Wire the real `LlamaServerProvider` and health probe from env vars.

    The provider's clients take `llm_timeout_seconds`, the bound the engine's
    `asyncio.wait_for` puts on each call, so the SDK's 600 s read default no
    longer cuts a call the engine still allows.
    """
    settings = ServiceSettings.from_env()
    llm = LlamaServerProvider(
        base_url=settings.llm_base_url,
        model=settings.model_file,
        timeout=settings.llm_timeout_seconds,
    )
    return build_app(settings, llm, httpx.AsyncClient())
