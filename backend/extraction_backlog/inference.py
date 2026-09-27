"""The `ExtractionInference` seam and its remote (HTTP) implementation.

The dispatcher never talks HTTP itself. It holds an `ExtractionInference`, which
T2a implements over the extraction service's HTTP API
(`backend/extraction_service/app.py`: `POST /v1/extract`, `GET /v1/info`,
`GET /v1/health`). An in-process implementation is T2b's; none exists here, and
nothing in this package falls back to the backend's own chat model.

Every call either returns a validated contract object or raises one of the typed
errors in `errors.py`. Every HTTP status is checked explicitly: a 2xx body must
parse as the contract model, a non-2xx body must parse as an `ErrorEnvelope` or
it is classified by status (502/503/504 without an envelope are gateway errors,
i.e. unreachable; anything else is an invalid response).
"""

from __future__ import annotations

import json
from typing import Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from backend.extraction_contract.models import (
    ErrorCode,
    ErrorEnvelope,
    ExtractRequest,
    ExtractResponse,
    HealthResponse,
    InfoResponse,
)

from .errors import (
    InferenceResponseInvalidError,
    InferenceServiceError,
    InferenceUnreachableError,
)

_ModelT = TypeVar("_ModelT", bound=BaseModel)

# Statuses a reverse proxy or load balancer answers with when the service
# behind it is down. Without a contract envelope they say nothing about the
# job, so they classify as unreachable rather than as a job failure.
_GATEWAY_STATUSES = frozenset({502, 503, 504})


class ExtractionInference(Protocol):
    """Where Stage 1.5 / 2 / 9 inference happens, as seen by the dispatcher.

    Implementations raise `InferenceUnreachableError`, `InferenceServiceError`
    or `InferenceResponseInvalidError` (all `ExtractionInferenceError`) and
    nothing else for an inference failure.
    """

    async def info(self) -> InfoResponse:
        """The service's identity and version stamps."""
        ...

    async def health(self) -> HealthResponse:
        """The service's readiness."""
        ...

    async def extract(self, req: ExtractRequest) -> ExtractResponse:
        """Run one extraction job."""
        ...


class RemoteExtractionInference:
    """`ExtractionInference` over the extraction service's HTTP API.

    The `httpx.AsyncClient` is injected (and owned by the caller), so tests
    reach a fake ASGI service through `httpx.ASGITransport` and production
    configures timeouts in `backend.factories.build_extraction_dispatcher`.
    """

    def __init__(self, client: httpx.AsyncClient, base_url: str) -> None:
        """Initialize the client.

        Args:
            client: The HTTP client every call goes through.
            base_url: Service root, e.g. `http://mist-extraction:8090`.
        """
        self._client = client
        self._base_url = base_url.rstrip("/")

    @property
    def base_url(self) -> str:
        """The service root this client talks to."""
        return self._base_url

    async def info(self) -> InfoResponse:
        """GET /v1/info."""
        response = await self._send("GET", "/v1/info", request_timeout_is_job_failure=False)
        return self._parse(response, InfoResponse)

    async def health(self) -> HealthResponse:
        """GET /v1/health."""
        response = await self._send("GET", "/v1/health", request_timeout_is_job_failure=False)
        return self._parse(response, HealthResponse)

    async def extract(self, req: ExtractRequest) -> ExtractResponse:
        """POST /v1/extract."""
        response = await self._send(
            "POST",
            "/v1/extract",
            json_body=req.model_dump(mode="json"),
            request_timeout_is_job_failure=True,
        )
        return self._parse(response, ExtractResponse)

    async def _send(
        self,
        method: str,
        path: str,
        *,
        request_timeout_is_job_failure: bool,
        json_body: dict | None = None,
    ) -> httpx.Response:
        """Send one request, mapping transport failures to typed errors.

        A connect failure is always unreachable. A timeout AFTER the request was
        sent (read/write/pool) on `/v1/extract` means the service took the job
        and did not finish it in time -- that is the job's `timeout`, and it
        counts. The same timeout on `/v1/info` or `/v1/health` is unreachable.
        """
        url = f"{self._base_url}{path}"
        try:
            return await self._client.request(method, url, json=json_body)
        except httpx.ConnectTimeout as exc:
            raise InferenceUnreachableError(f"{method} {url}: connect timeout: {exc}") from exc
        except httpx.TimeoutException as exc:
            if request_timeout_is_job_failure:
                raise InferenceServiceError(
                    f"{method} {url}: no reply before the client timeout: {exc}",
                    code=ErrorCode.TIMEOUT,
                    retryable=True,
                    http_status=None,
                ) from exc
            raise InferenceUnreachableError(f"{method} {url}: timeout: {exc}") from exc
        except httpx.TransportError as exc:
            raise InferenceUnreachableError(f"{method} {url}: {exc!r}") from exc

    @staticmethod
    def _parse(response: httpx.Response, model: type[_ModelT]) -> _ModelT:
        """Validate a response against its contract model, checking the status first."""
        status = response.status_code
        if 200 <= status < 300:
            try:
                return model.model_validate_json(response.content)
            except ValidationError as exc:
                raise InferenceResponseInvalidError(
                    f"{model.__name__} failed validation (HTTP {status}): {exc}",
                    http_status=status,
                ) from exc

        try:
            envelope = ErrorEnvelope.model_validate_json(response.content)
        except (ValidationError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            snippet = response.text[:200]
            if status in _GATEWAY_STATUSES:
                raise InferenceUnreachableError(
                    f"HTTP {status} with no error envelope: {snippet!r}", http_status=status
                ) from exc
            raise InferenceResponseInvalidError(
                f"HTTP {status} with no error envelope: {snippet!r}", http_status=status
            ) from exc

        detail = envelope.error
        raise InferenceServiceError(
            f"{detail.code.value} (HTTP {status}): {detail.message}",
            code=detail.code,
            retryable=detail.retryable,
            http_status=status,
        )
