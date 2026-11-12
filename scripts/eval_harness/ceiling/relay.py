"""A relay that answers the unchanged extraction service as llama-server would, from a store.

Under run A the service's `EXTRACTION_LLM_BASE_URL` points here. The relay serves the
three routes the service calls (`GET /health`, `GET /props`, `POST /v1/chat/completions`),
keys each chat request by the SHA-256 of its whole body, and answers it from the store's
key-to-answer mapping. A body with `"stream": true` is answered with server-sent events, as
llama-server answers it; any other body with one JSON completion. A key with no answer gets
HTTP 400 before any stream starts, which the openai SDK does not retry, so each pending
request reaches the relay exactly once and is logged for the round tool to offer to the
answerer. Two routes of its own, `/_relay/probe` and `/_relay/log`, attribute requests to
probe tags and report them in arrival order.

It listens on 127.0.0.1 only: the answers are eval material, not a service.
"""

from __future__ import annotations

import hashlib
import json
import socket
import threading
import time
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from backend.errors import MistError
from backend.extraction_service.schemas import DERIVATION_OUTPUT_SCHEMA, SCOPE_OUTPUT_SCHEMA

from .store import OfferedRequest, served_text

EXCLUDED_BODY_KEYS: tuple[str, ...] = ()
CONTEXT_SIZE = 16384
LOOPBACK = "127.0.0.1"


class RelayError(MistError):
    """Base of the relay's own failures, as distinct from the replies it serves."""


class RelayStartError(RelayError):
    """uvicorn did not start serving the relay."""


class UnclassifiedRequestError(RelayError):
    """A logged request matched none of the stage schemas, so it cannot be offered."""


@dataclass(frozen=True, slots=True)
class LogEntry:
    """One chat request as it reached the relay.

    `body` is kept in process for the round tool, which needs the messages and schema to
    write the pending file; `/_relay/log` serves only `public()`.
    """

    seq: int
    tag: str | None
    key: str
    kind: str | None
    answered: bool
    body: dict[str, Any]

    def public(self) -> dict[str, Any]:
        """Return the entry as `/_relay/log` serves it, without the body."""
        return {
            "seq": self.seq,
            "tag": self.tag,
            "key": self.key,
            "kind": self.kind,
            "answered": self.answered,
        }


class RelayLog:
    """Every chat request the relay saw, in arrival order, and the current probe tag.

    uvicorn serves from its own thread while the round tool reads from another, so every
    access holds a lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: list[LogEntry] = []
        self._tag: str | None = None

    @property
    def tag(self) -> str | None:
        """The tag later requests are attributed to; None before the first probe."""
        with self._lock:
            return self._tag

    def set_tag(self, tag: str) -> None:
        """Attribute every later request to `tag`, as `POST /_relay/probe` does."""
        with self._lock:
            self._tag = tag

    def record(
        self, *, tag: str | None, body: dict[str, Any], kind: str | None, answered: bool
    ) -> LogEntry:
        """Append one request, so its key and kind are kept whether or not it was answered."""
        with self._lock:
            entry = LogEntry(
                seq=len(self._entries) + 1,
                tag=tag,
                key=request_key(body),
                kind=kind,
                answered=answered,
                body=body,
            )
            self._entries.append(entry)
            return entry

    def entries(self) -> tuple[LogEntry, ...]:
        """Return a snapshot, so a reader never sees the list change under it."""
        with self._lock:
            return tuple(self._entries)

    def pending(self) -> tuple[LogEntry, ...]:
        """Return the requests the relay could not answer, in arrival order."""
        return tuple(e for e in self.entries() if not e.answered)


def request_key(body: Mapping[str, Any]) -> str:
    """Key a chat request by its whole POST body, so any change to it needs a new answer.

    Headers never take part: the SDK's `x-stainless-retry-count` varies between attempts.
    No body field varies between runs, so `EXCLUDED_BODY_KEYS` is empty; a field found
    to vary is a stop to report, not a key to add there.
    """
    kept = {k: v for k, v in body.items() if k not in EXCLUDED_BODY_KEYS}
    text = json.dumps(kept, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _body_schema(body: Mapping[str, Any]) -> Any:
    response_format = body.get("response_format")
    if not isinstance(response_format, dict):
        return None
    json_schema = response_format.get("json_schema")
    if not isinstance(json_schema, dict):
        return None
    return json_schema.get("schema")


def request_kind(body: Mapping[str, Any]) -> str | None:
    """Name the engine stage that made a request, from the schema it constrains to.

    The schemas are the ones the service sends: its scope and derivation constants, and
    for Stage 2 the schema the registry's active pack generated, the pack the service
    reads. So a schema change in the service shows up here as None rather than as a
    wrong kind. An extract request that already carries an assistant turn is the
    engine's repair attempt.

    Returns:
        `scope`, `extract`, `repair` or `derive`, or None for a body matching no stage.
    """
    from backend.knowledge.ontologies import registry

    schema = _body_schema(body)
    if schema == SCOPE_OUTPUT_SCHEMA:
        return "scope"
    if schema == DERIVATION_OUTPUT_SCHEMA:
        return "derive"
    if schema == registry.active_pack().extraction_schema:
        messages = body.get("messages")
        if isinstance(messages, list) and any(
            isinstance(m, dict) and m.get("role") == "assistant" for m in messages
        ):
            return "repair"
        return "extract"
    return None


def offered_request(entry: LogEntry) -> OfferedRequest:
    """Turn a pending log entry into what the store's `offer` takes.

    The schema comes from the request body itself, so `add` holds the answer to exactly
    the constraint the service sent.

    Raises:
        UnclassifiedRequestError: The entry matched no stage schema.
    """
    if entry.kind is None:
        raise UnclassifiedRequestError(f"request {entry.key[:12]} matches no stage schema")
    return OfferedRequest(key=entry.key, kind=entry.kind, schema=_body_schema(entry.body))


def _error(kind: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=400, content={"error": {"code": 400, "type": kind, "message": message}}
    )


async def _json_object(request: Request) -> dict[str, Any] | None:
    try:
        body = json.loads(await request.body())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return body if isinstance(body, dict) else None


_ZERO_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _chunk(body: Mapping[str, Any], key: str, choices: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-relay-{key[:12]}",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": body.get("model"),
        "choices": choices,
    }


def _event(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


async def _stream_events(body: Mapping[str, Any], key: str, text: str) -> AsyncIterator[bytes]:
    """A stored answer as llama-server streams a chat reply, so the service reads its own path.

    The whole text is one content delta, carried by the first chunk with the role. Then a
    finish chunk with an empty delta, the usage-only chunk with no choices when the body's
    `stream_options` asks for it, and `[DONE]`. The shape is the fake llama-server's in the
    service's tests; it is not checked here against a real llama-server's stream.
    """
    yield _event(
        _chunk(
            body,
            key,
            [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": text},
                    "finish_reason": None,
                }
            ],
        )
    )
    yield _event(_chunk(body, key, [{"index": 0, "delta": {}, "finish_reason": "stop"}]))
    stream_options = body.get("stream_options")
    if isinstance(stream_options, dict) and stream_options.get("include_usage") is True:
        yield _event({**_chunk(body, key, []), "usage": _ZERO_USAGE})
    yield b"data: [DONE]\n\n"


def build_relay(answers: Mapping[str, Any], log: RelayLog) -> FastAPI:
    """Build the relay app over a key-to-answer mapping, logging every chat request to `log`.

    Every answer's served text is computed here, so a bad answer fails at build time and
    never becomes a 500 that the SDK would retry.

    Raises:
        ValueError: An answer holds NaN or an infinity.
        TypeError: An answer holds a value JSON cannot encode.
    """
    texts = {key: served_text(answer) for key, answer in answers.items()}
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.get("/props")
    async def props() -> JSONResponse:
        return JSONResponse({"default_generation_settings": {"n_ctx": CONTEXT_SIZE}})

    @app.post("/_relay/probe")
    async def probe(request: Request) -> Response:
        body = await _json_object(request)
        if body is None or not isinstance(body.get("tag"), str):
            return _error("relay_bad_request", 'expected {"tag": <string>}')
        log.set_tag(body["tag"])
        return Response(status_code=204)

    @app.get("/_relay/log")
    async def relay_log() -> JSONResponse:
        return JSONResponse([e.public() for e in log.entries()])

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        body = await _json_object(request)
        if body is None:
            return _error("relay_bad_request", "the body is not a JSON object")
        kind = request_kind(body)
        key = request_key(body)
        text = texts.get(key) if kind is not None else None
        log.record(tag=log.tag, body=body, kind=kind, answered=text is not None)
        if kind is None:
            return _error("relay_unknown_kind", f"no stage schema matches {key}")
        if text is None:
            return _error("relay_pending", f"pending {key}")
        if body.get("stream") is True:
            return StreamingResponse(
                _stream_events(body, key, text), media_type="text/event-stream"
            )
        return JSONResponse(
            {
                "id": f"chatcmpl-relay-{key[:12]}",
                "object": "chat.completion",
                "created": 0,
                "model": body.get("model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": text,
                            "reasoning_content": "",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        )

    return app


@contextmanager
def running_relay(app: FastAPI, port: int = 0, *, start_timeout: float = 10.0) -> Iterator[str]:
    """Serve `app` with uvicorn on 127.0.0.1 in a background thread, yielding its base URL.

    The socket is bound before uvicorn starts, so port 0 yields the port the system chose,
    and a caller never races another process for a fixed port it has already named. The
    host is fixed at loopback, so no caller can expose the answers.

    Raises:
        RelayStartError: uvicorn did not start within `start_timeout` seconds.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((LOOPBACK, port))
    config = uvicorn.Config(
        app,
        log_level="warning",
        access_log=False,
        lifespan="off",
        ws="none",
        timeout_graceful_shutdown=2,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + start_timeout
        while not server.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                raise RelayStartError(f"the relay did not start on {LOOPBACK}:{port}")
            time.sleep(0.01)
        yield f"http://{LOOPBACK}:{sock.getsockname()[1]}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        if thread.is_alive():
            server.force_exit = True
            thread.join(timeout=5)
        sock.close()
