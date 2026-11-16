"""`McpSdkTransport`: an `McpTransport` over the `mcp` SDK's Streamable HTTP client.

This is the only module that imports the `mcp` SDK (2.3.0), so a backend without the package
still starts: the factory imports this module lazily and reports lights unavailable instead.

The SDK's client holds an anyio task group for the life of its context, entered by whichever task
entered the context. `open` enters it in the calling task and `close` exits it, so the connection
supervisor, which alone calls both, owns that task group (see `McpConnection`).

That ownership means an exception escaping one of the SDK's request tasks would cancel the
supervisor itself. The SDK turns an HTTP error status into a JSON-RPC error for the waiting call,
but lets an `httpx2` transport failure (connection refused, a body cut off) escape its request
task. `_GuardedHttpTransport` therefore sits under the SDK's HTTP client: it records a transport
failure and answers with a synthesised HTTP 503 instead, so the call fails on its own and the
supervisor lives to reconnect. It also records HTTP 401 and 403, which the SDK reports only as
a generic "error response", so the caller can tell a refused credential from an outage.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AsyncExitStack

import anyio
import httpx2
from mcp import Client, MCPError
from mcp.client.streamable_http import streamable_http_client
from mcp.types import REQUEST_TIMEOUT, CallToolResult, TextContent
from pydantic import ValidationError

from backend.errors import CapabilityUnavailableError, McpTransportError

from ..secrets import MASK, Secret
from .transport import McpToolResult, McpToolSpec

logger = logging.getLogger(__name__)

MAX_CALL_TIMEOUT_S = 3600.0
"""The most any one tool call may wait, in seconds.

An hour is far above any Home Assistant call and the 8 s default. The configured call timeout
has no upper bound, so this clamp keeps an unbounded value out of the SDK's deadline arithmetic.
"""

AUTH_REJECTED = "auth rejected"

# The SDK's default HTTP timeouts: a server may hold a response stream open, so reads wait longer.
_HTTP_TIMEOUT = httpx2.Timeout(30.0, read=300.0)

# Bounds the handshake and each tool listing; a tool call takes its own timeout instead.
_SESSION_TIMEOUT_S = 30.0

# A server that keeps returning a next cursor is broken; stop rather than page forever.
_MAX_TOOL_PAGES = 100

_AUTH_STATUSES = frozenset({401, 403})

# The failures the SDK reports for a broken session or call: MCP errors (a JSON-RPC error, a
# request timeout), HTTP and stream errors, closed anyio streams, a malformed result, and OS-level
# I/O. Anything else is a bug and propagates unchanged.
_SDK_ERRORS: tuple[type[Exception], ...] = (
    MCPError,
    httpx2.HTTPError,
    httpx2.StreamError,
    anyio.BrokenResourceError,
    anyio.ClosedResourceError,
    anyio.EndOfStream,
    ValidationError,
    OSError,
)
_CAUGHT: tuple[type[Exception], ...] = (*_SDK_ERRORS, ExceptionGroup)


def _sdk_timeout(timeout_s: float, *, warn: bool = True) -> float:
    """Clamp a call timeout to `MAX_CALL_TIMEOUT_S` and convert it to the SDK's form.

    The SDK's `Client.call_tool` takes `read_timeout_seconds: float | None`, so the SDK form is
    float seconds. A value that is not at most the bound (`inf` and `nan` included) is clamped,
    with a WARNING naming the requested value and the bound unless `warn` is False.
    """
    if timeout_s <= MAX_CALL_TIMEOUT_S:
        return float(timeout_s)
    if warn:
        logger.warning(
            "MCP call timeout %r s is past the bound; clamping it to %g s",
            timeout_s,
            MAX_CALL_TIMEOUT_S,
        )
    return MAX_CALL_TIMEOUT_S


class _GuardedBody(httpx2.AsyncByteStream):
    """A response body whose transport failures are recorded and raised as `StreamError`.

    The SDK's response readers contain a `StreamError` and fail the one call; a raw transport
    error would escape into the SDK's task group.
    """

    def __init__(self, inner: httpx2.AsyncByteStream, guard: _GuardedHttpTransport) -> None:
        self._inner = inner
        self._guard = guard

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self._inner:
                yield chunk
        except httpx2.TransportError as exc:
            self._guard.record_failure(exc)
            raise httpx2.StreamError(self._guard.failure or "the response body failed") from exc

    async def aclose(self) -> None:
        await self._inner.aclose()


class _GuardedHttpTransport(httpx2.AsyncBaseTransport):
    """Wraps the real HTTP transport: records 401, 403 and transport failures for the mapping."""

    def __init__(self, inner: httpx2.AsyncBaseTransport) -> None:
        self._inner = inner
        self.auth_rejected = False
        self.failure: str | None = None

    def record_failure(self, exc: httpx2.TransportError) -> None:
        self.failure = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        try:
            response = await self._inner.handle_async_request(request)
        except httpx2.TransportError as exc:
            self.record_failure(exc)
            return httpx2.Response(503, request=request)
        if response.status_code in _AUTH_STATUSES:
            self.auth_rejected = True
        if isinstance(response.stream, httpx2.AsyncByteStream):
            response.stream = _GuardedBody(response.stream, self)
        return response

    async def aclose(self) -> None:
        await self._inner.aclose()


class McpSdkTransport:
    """One Streamable HTTP session with Home Assistant's MCP server, as an `McpTransport`.

    A transport serves one connect attempt: `McpConnection` builds a new one for each.
    """

    def __init__(
        self,
        url: str,
        token: Secret,
        *,
        http_transport: httpx2.AsyncBaseTransport | None = None,
        client_factory: Callable[[], Client] | None = None,
    ) -> None:
        """Build a closed transport; nothing connects until `open`.

        Args:
            url: The server's MCP endpoint, for example `http://homeassistant.local:8123/api/mcp`.
            token: The long-lived access token, sent as a bearer Authorization header.
            http_transport: The HTTP transport under the SDK's HTTP client; None for real
                sockets. Tests pass an `httpx2.MockTransport`.
            client_factory: Builds the SDK client in place of the Streamable HTTP one, so tests
                can reach an in-process server; `url`, `token` and `http_transport` then go
                unused.
        """
        self._url = url
        self._token = token
        self._http_transport = http_transport
        self._client_factory = client_factory
        self._client: Client | None = None
        self._stack: AsyncExitStack | None = None
        self._guard: _GuardedHttpTransport | None = None
        self._clamp_warned = False

    def __repr__(self) -> str:
        return f"McpSdkTransport({self._url!r})"

    async def open(self) -> None:
        """Enter the SDK client's context in this task and complete the handshake.

        Raises:
            CapabilityUnavailableError: The server answered HTTP 401 or 403 ("auth rejected").
            McpTransportError: The server could not be reached or the handshake failed.

        A failed or cancelled open releases what it entered before the exception propagates.
        """
        if self._client is not None:
            raise McpTransportError("the MCP session is already open")
        try:
            async with AsyncExitStack() as stack:
                client = await self._enter_client(stack)
                self._stack = stack.pop_all()
                self._client = client
        except _CAUGHT as exc:
            mapped = self._mapped(exc, "could not open the MCP session")
            if mapped is None:
                raise
            raise mapped from self._cause(exc)

    async def list_tools(self) -> list[McpToolSpec]:
        """Return every tool the server lists now, following its pages."""
        client = self._require_client()
        specs: list[McpToolSpec] = []
        cursor: str | None = None
        try:
            for _ in range(_MAX_TOOL_PAGES):
                page = await client.list_tools(cursor=cursor, cache_mode="refresh")
                specs.extend(
                    McpToolSpec(
                        name=tool.name,
                        description=tool.description or "",
                        input_schema=dict(tool.input_schema),
                    )
                    for tool in page.tools
                )
                cursor = page.next_cursor
                if cursor is None:
                    return specs
        except _CAUGHT as exc:
            mapped = self._mapped(exc, "listing the MCP tools failed")
            if mapped is None:
                raise
            raise mapped from self._cause(exc)
        raise McpTransportError(f"the MCP tool listing did not end after {_MAX_TOOL_PAGES} pages")

    async def call_tool(
        self, name: str, arguments: Mapping[str, object], timeout_s: float
    ) -> McpToolResult:
        """Call one tool; a result the server marks as an error is returned, not raised.

        Raises:
            CapabilityUnavailableError: The server answered HTTP 401 or 403 ("auth rejected").
            McpTransportError: The call failed, or outlived `timeout_s` (clamped to
                `MAX_CALL_TIMEOUT_S`).
        """
        client = self._require_client()
        sdk_timeout = _sdk_timeout(timeout_s, warn=not self._clamp_warned)
        if sdk_timeout != timeout_s:
            self._clamp_warned = True
        try:
            result = await client.call_tool(name, dict(arguments), read_timeout_seconds=sdk_timeout)
        except _CAUGHT as exc:
            if isinstance(exc, MCPError) and exc.code == REQUEST_TIMEOUT:
                raise McpTransportError(
                    f"MCP tool {name} timed out after {sdk_timeout:g} s"
                ) from exc
            mapped = self._mapped(exc, f"MCP tool {name} failed")
            if mapped is None:
                raise
            raise mapped from self._cause(exc)
        return _tool_result(result)

    async def close(self) -> None:
        """Exit the SDK client's context; call it from the task that called `open`.

        Raises:
            McpTransportError: The session failed while it was being shut down; it is released
                all the same.
        """
        stack = self._stack
        self._stack = None
        self._client = None
        if stack is None:
            return
        try:
            await stack.aclose()
        except _CAUGHT as exc:
            mapped = self._mapped(exc, "closing the MCP session failed")
            if mapped is None:
                raise
            raise mapped from self._cause(exc)

    async def _enter_client(self, stack: AsyncExitStack) -> Client:
        if self._client_factory is not None:
            return await stack.enter_async_context(self._client_factory())
        inner = self._http_transport or httpx2.AsyncHTTPTransport()
        self._guard = _GuardedHttpTransport(inner)
        http = await stack.enter_async_context(
            httpx2.AsyncClient(
                headers={"Authorization": f"Bearer {self._token.reveal()}"},
                timeout=_HTTP_TIMEOUT,
                transport=self._guard,
            )
        )
        client = Client(
            streamable_http_client(self._url, http_client=http),
            read_timeout_seconds=_SESSION_TIMEOUT_S,
        )
        return await stack.enter_async_context(client)

    def _require_client(self) -> Client:
        if self._client is None:
            raise McpTransportError("the MCP session is not open")
        return self._client

    def _mapped(self, exc: Exception, doing: str) -> Exception | None:
        """Map an SDK failure to MIST's error; None for anything else, which is re-raised."""
        failure = _sdk_failure(exc)
        if failure is None:
            return None
        guard = self._guard
        if guard is not None and guard.auth_rejected:
            return CapabilityUnavailableError(AUTH_REJECTED)
        if guard is not None and guard.failure is not None:
            detail = guard.failure
        elif isinstance(failure, MCPError):
            detail = failure.message
        else:
            detail = (
                f"{type(failure).__name__}: {failure}" if str(failure) else type(failure).__name__
            )
        return McpTransportError(self._redacted(f"{doing}: {detail}"))

    def _cause(self, exc: Exception) -> Exception | None:
        """Chain the SDK's exception unless its text carries the token."""
        token = self._token.reveal()
        if token and (token in str(exc) or token in repr(exc)):
            return None
        return exc

    def _redacted(self, text: str) -> str:
        token = self._token.reveal()
        return text.replace(token, MASK) if token else text


def _sdk_failure(exc: Exception) -> Exception | None:
    """Return the SDK failure `exc` stands for, or None when it is, or holds, anything else."""
    if isinstance(exc, ExceptionGroup):
        matched, rest = exc.split(_SDK_ERRORS)
        if matched is None or rest is not None:
            return None
        leaf: Exception = matched
        while isinstance(leaf, ExceptionGroup):
            leaf = leaf.exceptions[0]
        return leaf
    return exc if isinstance(exc, _SDK_ERRORS) else None


def _tool_result(result: CallToolResult) -> McpToolResult:
    text = "\n".join(block.text for block in result.content if isinstance(block, TextContent))
    structured = result.structured_content
    return McpToolResult(
        is_error=bool(result.is_error),
        text=text,
        structured=dict(structured) if structured is not None else None,
    )
