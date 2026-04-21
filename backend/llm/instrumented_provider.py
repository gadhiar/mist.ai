"""Observability wrapper for StreamingLLMProvider.

`InstrumentedStreamingLLMProvider` wraps any concrete provider and emits a
`phase: "llm_call"` JSONL record via `DebugJSONLLogger` each time the inner
generator yields a non-partial response. The wrapper is transparent — it
delegates every call, including streaming chunks and health checks, without
modifying requests or responses.

Caller metadata (session_id, event_id, call_site, pass_num) is threaded
through a `ContextVar` via the `llm_call_context` context manager so that
instrumentation does not pollute the existing provider interface.

Usage:

    from backend.llm.instrumented_provider import (
        InstrumentedStreamingLLMProvider,
        llm_call_context,
    )

    provider = InstrumentedStreamingLLMProvider(LlamaServerProvider(...), logger)
    with llm_call_context(session_id=sid, call_site="chat.initial"):
        response = await provider.invoke(request)

The wrapper is a no-op on the recording path when
`DebugJSONLLogger.llm_call_enabled` is False. It still delegates every call
transparently, so it is safe to leave installed in production.

Spec: cluster-execution-roadmap.md Cluster 5 deliverable 1.
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator, Generator, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from backend.debug_jsonl_logger import DebugJSONLLogger
from backend.llm.models import LLMRequest, LLMResponse
from backend.llm.provider import StreamingLLMProvider

_CALL_CONTEXT: ContextVar[dict[str, Any]] = ContextVar("mist_llm_call_context", default={})


@contextmanager
def llm_call_context(**fields: Any) -> Iterator[None]:
    """Associate caller metadata with LLM calls nested inside the block.

    The InstrumentedStreamingLLMProvider reads this context at invoke time
    and embeds any of {session_id, event_id, call_site, pass_num} in the
    emitted llm_call record. Nested contexts merge with inner fields taking
    precedence; fields not set remain at their outer value.

    Using a ContextVar makes this safe under asyncio (each task inherits its
    own copy) and across threads (ContextVar is thread-aware in CPython).
    """
    current = _CALL_CONTEXT.get()
    merged = {**current, **fields}
    token = _CALL_CONTEXT.set(merged)
    try:
        yield
    finally:
        _CALL_CONTEXT.reset(token)


def get_llm_call_context() -> dict[str, Any]:
    """Return a copy of the current LLM call context (empty dict when unset)."""
    return dict(_CALL_CONTEXT.get())


_ALLOWED_CONTEXT_KEYS = frozenset({"event_id", "session_id", "call_site", "pass_num"})


def _filter_context(context: dict[str, Any]) -> dict[str, Any]:
    """Keep only the fields the logger understands so callers can stash extras."""
    return {k: v for k, v in context.items() if k in _ALLOWED_CONTEXT_KEYS}


class InstrumentedStreamingLLMProvider(StreamingLLMProvider):
    """Transparent StreamingLLMProvider that records llm_call JSONL records.

    Wraps any `StreamingLLMProvider`. On every non-partial response yielded by
    `generate()` or `generate_sync()`, it measures wall-clock latency since
    the generator started and emits one JSONL record via the supplied
    `DebugJSONLLogger`.

    Recording is gated on `DebugJSONLLogger.llm_call_enabled`, so leaving the
    wrapper installed in production is free when the gate is off.

    Streaming chunks (partial=True) pass through unrecorded. Only the final
    aggregated response triggers emission. This matches the single-row-per-call
    mental model of the existing turn / extraction records.

    The wrapper does not mutate the request or response objects. All
    exceptions raised by the inner provider propagate unchanged.
    """

    def __init__(self, inner: StreamingLLMProvider, logger: DebugJSONLLogger) -> None:
        """Initialize the instrumented wrapper.

        Args:
            inner: The concrete provider being wrapped (LlamaServerProvider,
                OllamaProvider, FakeLLM in tests). Must satisfy the
                StreamingLLMProvider ABC.
            logger: The DebugJSONLLogger that will receive records. If its
                `llm_call_enabled` gate is False, the wrapper records nothing
                but still delegates transparently.
        """
        self._inner = inner
        self._logger = logger
        self.model = inner.model

    @property
    def inner(self) -> StreamingLLMProvider:
        """Expose the wrapped provider (introspection and tests)."""
        return self._inner

    async def generate(
        self, request: LLMRequest, *, stream: bool = False
    ) -> AsyncGenerator[LLMResponse, None]:
        """Delegate to inner.generate and emit llm_call record on final yield."""
        start = time.perf_counter()
        async for response in self._inner.generate(request, stream=stream):
            if not response.partial and self._logger.llm_call_enabled:
                latency_ms = (time.perf_counter() - start) * 1000.0
                self._logger.record_llm_call(
                    request=request,
                    response=response,
                    latency_ms=latency_ms,
                    model=self.model,
                    **_filter_context(get_llm_call_context()),
                )
            yield response

    def generate_sync(
        self, request: LLMRequest, *, stream: bool = False
    ) -> Generator[LLMResponse, None, None]:
        """Delegate to inner.generate_sync and emit llm_call record on final yield."""
        start = time.perf_counter()
        for response in self._inner.generate_sync(request, stream=stream):
            if not response.partial and self._logger.llm_call_enabled:
                latency_ms = (time.perf_counter() - start) * 1000.0
                self._logger.record_llm_call(
                    request=request,
                    response=response,
                    latency_ms=latency_ms,
                    model=self.model,
                    **_filter_context(get_llm_call_context()),
                )
            yield response

    async def health_check(self) -> bool:
        """Pass-through. Health checks are not instrumented."""
        return await self._inner.health_check()
