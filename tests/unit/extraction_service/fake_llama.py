"""A tiny fake llama-server as an ASGI app, for wiring into LlamaServerProvider.

Implements just enough of llama-server's surface the extraction service
needs: `GET /health`, `GET /props` and `POST /v1/chat/completions`, the last
as one JSON completion or, for a request with `stream: true`, as server-sent
events. Tests mutate a shared `FakeLlamaState` to script responses (canned
content per call, an HTTP error status, a health status, a `/props` status
and body, a gate that holds every chat call until a test releases it, a gate
that holds a streamed reply part-way, or a streamed reply cut short) and to
inspect what was actually sent (`chat_requests`, `props_requests`) and which
calls the client left before their reply was complete (`chat_aborted`).

`StreamingASGITransport` is the httpx transport to reach the app through
when a test streams: unlike `httpx.ASGITransport`, it lets a client be
part-way through a reply, and so leave one.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

# Low effort's context, the `EXTRACTION_LLM_CTX_SIZE` a run beside the main
# stack sets (the compose default is medium's 65,536). Low's extract cap and
# the prompt reserve fill it exactly (2,048 + 14,336), so a fake-backed service
# at low effort has a serving identity.
FAKE_N_CTX = 16384

_USAGE = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}


class StreamDroppedError(Exception):
    """The fake cut a streamed reply short (`FakeLlamaState.stream_drop`)."""


@dataclass
class FakeLlamaState:
    """Mutable script + recorder for one fake llama-server instance."""

    health_status: int = 200
    # Canned assistant `content` strings returned in call order; once the
    # list is exhausted, the last entry repeats.
    chat_responses: list[str] = field(
        default_factory=lambda: ['{"entities": [], "relationships": []}']
    )
    # When set, /v1/chat/completions returns this HTTP status instead of a
    # completion (simulates an upstream 5xx).
    chat_status_code: int | None = None
    # When set, every /v1/chat/completions call records its request and then
    # blocks on this event before answering. A test that never sets it gets
    # a call that never answers (the engine's timeout fires); a test that
    # sets it controls exactly when in-flight calls complete. No wall clock.
    # Use the `blocked_chat` fixture, which releases the gate at teardown.
    chat_gate: asyncio.Event | None = None
    # When set, a streamed reply sends its first content delta and then
    # blocks on this event: the call is held with its stream open. Use the
    # `held_stream` fixture, which releases the gate at teardown.
    stream_gate: asyncio.Event | None = None
    # When True, a streamed reply ends after its first content delta with no
    # finish reason, as a connection that dropped.
    stream_drop: bool = False
    chat_requests: list[dict] = field(default_factory=list)
    # Streamed replies that have sent their first content delta and not ended.
    streams_open: int = 0
    # Chat calls the client left before the reply was complete: its request
    # cancelled while the call was held, or its stream closed part-way.
    chat_aborted: int = 0
    # GET /props: the status and JSON body to answer with.
    props_status: int = 200
    props_body: object = field(
        default_factory=lambda: {"default_generation_settings": {"n_ctx": FAKE_N_CTX}}
    )
    props_requests: int = 0

    def next_response(self) -> str:
        index = min(len(self.chat_requests) - 1, len(self.chat_responses) - 1)
        return self.chat_responses[max(index, 0)]


def _event(model: str, *, delta: dict | None = None, finish_reason: str | None = None) -> bytes:
    """One `chat.completion.chunk` event; with no `delta`, the usage-only chunk."""
    payload: dict = {
        "id": "chatcmpl-fake",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": (
            [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
        ),
    }
    if delta is None:
        payload["usage"] = _USAGE
    return f"data: {json.dumps(payload)}\n\n".encode()


async def _stream_events(state: FakeLlamaState, body: dict, content: str) -> AsyncIterator[bytes]:
    """The reply as server-sent events: the content in two deltas, then the end.

    The end is a finish chunk, the usage-only chunk when the request asked
    for one, and `[DONE]`.
    """
    model = body.get("model", "fake-model")
    half = len(content) // 2
    # True once the fake itself ended the reply, completely or by dropping it.
    ended_here = False
    opened = False
    try:
        yield _event(model, delta={"role": "assistant", "content": content[:half]})
        state.streams_open += 1
        opened = True
        if state.stream_gate is not None:
            await state.stream_gate.wait()
        if state.stream_drop:
            ended_here = True
            raise StreamDroppedError
        yield _event(model, delta={"content": content[half:]})
        yield _event(model, delta={}, finish_reason="stop")
        if (body.get("stream_options") or {}).get("include_usage"):
            yield _event(model)
        yield b"data: [DONE]\n\n"
        ended_here = True
    finally:
        if opened:
            state.streams_open -= 1
        if not ended_here:
            state.chat_aborted += 1


def build_fake_llama_app(state: FakeLlamaState) -> Starlette:
    """Build a Starlette ASGI app backed by `state`."""

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"}, status_code=state.health_status)

    async def props(request: Request) -> JSONResponse:
        state.props_requests += 1
        return JSONResponse(state.props_body, status_code=state.props_status)

    async def chat_completions(request: Request) -> Response:
        body = await request.json()
        state.chat_requests.append(body)

        if state.chat_gate is not None:
            try:
                await state.chat_gate.wait()
            except asyncio.CancelledError:
                state.chat_aborted += 1
                raise

        if state.chat_status_code is not None:
            return JSONResponse(
                {"error": "fake upstream failure"}, status_code=state.chat_status_code
            )

        content = state.next_response()
        if body.get("stream"):
            return StreamingResponse(
                _stream_events(state, body, content), media_type="text/event-stream"
            )
        return JSONResponse(
            {
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "created": 0,
                "model": body.get("model", "fake-model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": _USAGE,
            }
        )

    return Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/props", props, methods=["GET"]),
            Route("/v1/chat/completions", chat_completions, methods=["POST"]),
        ]
    )


# ---------------------------------------------------------------------------
# A transport a client can be part-way through a reply on
# ---------------------------------------------------------------------------

_END = object()


@dataclass
class _Reply:
    """What the app has sent of one response."""

    chunks: asyncio.Queue = field(default_factory=asyncio.Queue)
    complete: bool = False


class _ReplyStream(httpx.AsyncByteStream):
    """The body of one response, read as the app sends it."""

    def __init__(self, app_task: asyncio.Future, reply: _Reply) -> None:
        self._app_task = app_task
        self._reply = reply

    async def __aiter__(self) -> AsyncIterator[bytes]:
        while True:
            chunk = await self._reply.chunks.get()
            if chunk is _END:
                break
            yield chunk
        if not self._reply.complete:
            raise httpx.RemoteProtocolError("the app ended its response before the body was sent")

    async def aclose(self) -> None:
        await _stop(self._app_task)


async def _stop(app_task: asyncio.Future) -> None:
    """Cancel the app's handler if it still runs, and wait until it has ended."""
    if not app_task.done():
        app_task.cancel()
    await asyncio.wait({app_task})
    if not app_task.cancelled():
        # Read, so a handler that failed is not reported as never retrieved;
        # the reader has already raised for it.
        app_task.exception()


class StreamingASGITransport(httpx.AsyncBaseTransport):
    """An httpx transport that hands over an ASGI app's response as the app sends it.

    `httpx.ASGITransport` returns a response only once the app has finished
    it, so a client is never part-way through a stream. Here the app runs in
    its own task and each body chunk reaches the client as it is sent.
    Closing the response before its end, or cancelling the request before the
    response starts, cancels the app's handler: what the app sees of a client
    that went away. An app that fails after its response started reaches the
    client as `httpx.RemoteProtocolError`.
    """

    def __init__(self, app) -> None:
        self._app = app

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = await request.aread()
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": request.method,
            "headers": [(name.lower(), value) for name, value in request.headers.raw],
            "scheme": request.url.scheme,
            "path": request.url.path,
            "raw_path": request.url.raw_path.split(b"?")[0],
            "query_string": request.url.query,
            "server": (request.url.host, request.url.port),
            "client": ("127.0.0.1", 123),
            "root_path": "",
        }
        started: asyncio.Future = asyncio.get_running_loop().create_future()
        reply = _Reply()
        body_read = False

        async def receive() -> dict:
            nonlocal body_read
            if not body_read:
                body_read = True
                return {"type": "http.request", "body": body, "more_body": False}
            # The client stays connected until the handler is cancelled.
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        async def send(message: dict) -> None:
            if message["type"] == "http.response.start":
                started.set_result((message["status"], message.get("headers", [])))
            elif message["type"] == "http.response.body":
                data = message.get("body", b"")
                if data:
                    reply.chunks.put_nowait(data)
                if not message.get("more_body", False):
                    reply.complete = True
                    reply.chunks.put_nowait(_END)

        app_task = asyncio.ensure_future(self._app(scope, receive, send))
        app_task.add_done_callback(lambda _done: reply.chunks.put_nowait(_END))
        try:
            await asyncio.wait({app_task, started}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            await _stop(app_task)
            raise
        if not started.done():
            app_task.result()
            raise httpx.RemoteProtocolError("the app returned without a response")
        status, headers = started.result()
        return httpx.Response(status, headers=headers, stream=_ReplyStream(app_task, reply))
