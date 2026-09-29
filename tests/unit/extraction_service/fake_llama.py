"""A tiny fake llama-server as an ASGI app, for wiring into LlamaServerProvider.

Implements just enough of llama-server's surface the extraction service
needs: `GET /health`, `GET /props` and `POST /v1/chat/completions`. Tests
mutate a shared `FakeLlamaState` to script responses (canned content per
call, an HTTP error status, a health status, a `/props` status and body, or
a gate that holds every chat call until a test releases it) and to inspect
what was actually sent (`chat_requests`, `props_requests`).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

FAKE_N_CTX = 8192


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
    chat_requests: list[dict] = field(default_factory=list)
    # GET /props: the status and JSON body to answer with.
    props_status: int = 200
    props_body: object = field(
        default_factory=lambda: {"default_generation_settings": {"n_ctx": FAKE_N_CTX}}
    )
    props_requests: int = 0

    def next_response(self) -> str:
        index = min(len(self.chat_requests) - 1, len(self.chat_responses) - 1)
        return self.chat_responses[max(index, 0)]


def build_fake_llama_app(state: FakeLlamaState) -> Starlette:
    """Build a Starlette ASGI app backed by `state`."""

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"}, status_code=state.health_status)

    async def props(request: Request) -> JSONResponse:
        state.props_requests += 1
        return JSONResponse(state.props_body, status_code=state.props_status)

    async def chat_completions(request: Request) -> JSONResponse:
        body = await request.json()
        state.chat_requests.append(body)

        if state.chat_gate is not None:
            await state.chat_gate.wait()

        if state.chat_status_code is not None:
            return JSONResponse(
                {"error": "fake upstream failure"}, status_code=state.chat_status_code
            )

        content = state.next_response()
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
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )

    return Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/props", props, methods=["GET"]),
            Route("/v1/chat/completions", chat_completions, methods=["POST"]),
        ]
    )
