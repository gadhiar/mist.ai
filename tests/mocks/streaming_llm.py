"""Fake StreamingLLMProvider for streaming-pipeline tests (T3 / MIS-171).

`FakeLLM` (tests/mocks/ollama.py) is a single-response test double built
for the pre-streaming pipeline; its `streaming_chunks` option is a flat
list of strings with no support for tool_calls on the terminal chunk or
for controllable async delays between chunks. The T3 streaming pipeline
(`backend.chat.conversation_handler.ConversationHandler.handle_message_streaming`)
needs both, so this module provides a purpose-built double instead of
overloading FakeLLM further.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass, field

from backend.llm.models import LLMRequest, LLMResponse, ToolCall, UsageMetadata
from backend.llm.provider import StreamingLLMProvider


@dataclass
class ScriptedPass:
    """One LLM pass's script.

    `chunks` is played back in order. A plain string is emitted as one
    `partial=True` content chunk. An `asyncio.Event` is awaited (not
    emitted) before continuing to the next item -- this is how a test pins
    "chunk N arrives before some later event fires" without a real clock
    or sleep. After every chunk is played, the contract-mandated terminal
    `partial=False` chunk is emitted, carrying `tool_calls` /
    `finish_reason` / `usage` / `reasoning_content` and `content=None`.

    `error`, when set, is raised after every chunk is played and before the
    terminal chunk -- a provider that fails mid-stream (or, with no chunks,
    before its first token).
    """

    chunks: list[str | asyncio.Event] = field(default_factory=list)
    tool_calls: list[ToolCall] | None = None
    finish_reason: str | None = None
    usage: UsageMetadata | None = None
    reasoning_content: str | None = None
    error: Exception | None = None


class FakeStreamingLLMProvider(StreamingLLMProvider):
    """Scripted `StreamingLLMProvider` double, one `ScriptedPass` per call.

    Each call to `generate(..., stream=True)` pops the next `ScriptedPass`
    off the queue given at construction and replays it faithfully to the
    real contract (backend/llm/provider.py): content deltas as
    `partial=True` chunks, then exactly one terminal `partial=False` chunk
    whose `content` is always `None`.

    `stream=False` and `generate_sync` are not implemented: the T3
    streaming pipeline never calls either for its conversation passes, and
    a fake that silently degraded to something else would hide a
    regression rather than fail it.
    """

    model: str = "fake-streaming-model"

    def __init__(self, passes: list[ScriptedPass]) -> None:
        self._passes = list(passes)
        self.calls: list[LLMRequest] = []

    async def generate(
        self, request: LLMRequest, *, stream: bool = False
    ) -> AsyncGenerator[LLMResponse, None]:
        self.calls.append(request)
        if not stream:
            raise NotImplementedError("FakeStreamingLLMProvider only supports stream=True")
        if not self._passes:
            raise AssertionError(
                f"FakeStreamingLLMProvider ran out of scripted passes on call {len(self.calls)}"
            )
        step = self._passes.pop(0)
        for chunk in step.chunks:
            if isinstance(chunk, asyncio.Event):
                await chunk.wait()
                continue
            yield LLMResponse(content=chunk, partial=True)
        if step.error is not None:
            raise step.error
        yield LLMResponse(
            content=None,
            tool_calls=step.tool_calls,
            finish_reason=step.finish_reason,
            usage=step.usage,
            reasoning_content=step.reasoning_content,
            partial=False,
        )

    def generate_sync(
        self, request: LLMRequest, *, stream: bool = False
    ) -> Generator[LLMResponse, None, None]:
        raise NotImplementedError("FakeStreamingLLMProvider is async-only")
