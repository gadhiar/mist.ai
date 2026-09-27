"""A tiny fake llama-server as an ASGI app, for wiring into LlamaServerProvider.

Implements just enough of the OpenAI-compatible surface the extraction
service needs: `GET /health` and `POST /v1/chat/completions`. Tests mutate
a shared `FakeLlamaState` to script responses (canned content per call,
an HTTP error status, a health status, or an artificial delay for timeout
tests) and to inspect what was actually sent (`chat_requests`).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route


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
    # Artificial delay before responding, to trigger the engine's timeout.
    delay_seconds: float = 0.0
    chat_requests: list[dict] = field(default_factory=list)

    def next_response(self) -> str:
        index = min(len(self.chat_requests) - 1, len(self.chat_responses) - 1)
        return self.chat_responses[max(index, 0)]


def build_fake_llama_app(state: FakeLlamaState) -> Starlette:
    """Build a Starlette ASGI app backed by `state`."""

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"}, status_code=state.health_status)

    async def chat_completions(request: Request) -> JSONResponse:
        body = await request.json()
        state.chat_requests.append(body)

        if state.delay_seconds:
            await asyncio.sleep(state.delay_seconds)

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
            Route("/v1/chat/completions", chat_completions, methods=["POST"]),
        ]
    )
