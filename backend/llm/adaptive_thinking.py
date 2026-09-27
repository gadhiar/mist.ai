"""AdaptiveThinkingProvider -- tool-dispatch retry-once thinking budget wrapper.

MIS-171 T4 decision 6: tool-dispatch turns (`LLMRequest.tools` set, caller
left `LLMRequest.thinking` unset) get a bounded thinking budget on the first
attempt -- `ThinkingConfig(budget_tokens=<configured>, default 1024)` -- so a
model that reasons before emitting a tool call cannot silently burn the
completion budget on reasoning tokens with nothing left for the call itself.
If that attempt fails validation, exactly one retry follows UNBUDGETED
(`thinking=None`, letting llama-server's own `--reasoning-budget` default
apply, per `ThinkingConfig`'s own docstring on why `None` -- not `-1` --
means unbudgeted).

Validation failure is exactly one of:
- `LLMResponseError` (`backend.errors`) -- malformed tool-call arguments
  JSON, raised by `LlamaServerProvider._finalize_tool_calls`.
- `finish_reason == "length"` with no content and no tool_calls -- the
  budget (or max_tokens) was exhausted before any usable output appeared.
- a tool call whose name is not present in the request's own tool schemas.

Whether Gemma 4 E4B's chat template honours `reasoning_budget_tokens` with
reasoning actually enabled on the production server is UNVERIFIED here --
llama-server may run with reasoning off entirely in production, in which
case llama.cpp ignores `reasoning_budget_tokens` and this wrapper is inert:
it sends the field, nothing consumes it, and none of the three failure
conditions above are caused by it. See the T4 worker report for a host
check the lead can run against the live server to settle this.

Streaming: a retry is issued only while no content partial has yet reached
the caller. `LlamaServerProvider.generate()` never yields a content partial
for reasoning-only output -- reasoning deltas are aggregated silently into
the final chunk's `reasoning_content`, never as `partial=True` chunks (see
`llama_server_provider.py`) -- so the expected failure mode (budget consumed
entirely by reasoning, zero content, `finish_reason=length`) always still
has `content_yielded=False` when the final chunk arrives. Once real content
has reached the caller:
- an `LLMResponseError` raised after content was yielded is RE-RAISED
  (it is already propagating as an exception through the async generator;
  it is not caught).
- a length/unknown-tool failure detected in the final chunk after content
  was yielded is PASSED THROUGH unchanged -- the caller (e.g. a TTS
  pipeline) already consumed and committed to this stream, so silently
  replacing it with a second, retried stream would mean either duplicating
  or contradicting content already delivered downstream.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import AsyncGenerator, Generator

from backend.errors import LLMResponseError
from backend.llm.models import LLMRequest, LLMResponse, ThinkingConfig
from backend.llm.provider import StreamingLLMProvider

logger = logging.getLogger(__name__)

_retry_count_lock = threading.Lock()
_retry_count = 0


def get_adaptive_thinking_retry_count() -> int:
    """Process-wide count of unbudgeted retries issued so far.

    Thread-safe: `generate_sync` runs on a worker thread (voice pipeline)
    while `generate` runs on the event loop, and both increment this counter.
    """
    with _retry_count_lock:
        return _retry_count


def reset_adaptive_thinking_retry_count() -> None:
    """Reset the process-wide retry counter. Test-only."""
    global _retry_count
    with _retry_count_lock:
        _retry_count = 0


def _record_retry(*, reason: str, budget: int, attempt: int) -> None:
    """Increment the process-wide counter and log one structured WARNING line."""
    global _retry_count
    with _retry_count_lock:
        _retry_count += 1
    logger.warning(
        "AdaptiveThinkingProvider retry: reason=%s budget=%d attempt=%d",
        reason,
        budget,
        attempt,
    )


def _allowed_tool_names(request: LLMRequest) -> set[str]:
    """Extract tool names from the request's own OpenAI-format tool schemas."""
    names: set[str] = set()
    for tool in request.tools or []:
        name = (tool.get("function") or {}).get("name")
        if name:
            names.add(name)
    return names


def _failure_reason(response: LLMResponse, allowed_tool_names: set[str]) -> str | None:
    """Return a failure-reason string for the two non-exception failure shapes.

    Does not check for `LLMResponseError` -- that arrives as a raised
    exception while iterating the inner generator, not as a field on a
    successfully-returned `LLMResponse`.
    """
    if response.finish_reason == "length" and not response.content and not response.tool_calls:
        return "length_exhausted_no_output"
    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call.name not in allowed_tool_names:
                return f"unknown_tool_name:{tool_call.name}"
    return None


class AdaptiveThinkingProvider(StreamingLLMProvider):
    """Wraps a StreamingLLMProvider with a bounded thinking budget for tool turns.

    Wired outermost around `InstrumentedStreamingLLMProvider` by
    `backend.factories.build_llm_provider` so both the budgeted attempt and
    any unbudgeted retry each emit their own `llm_call` JSONL record.
    """

    def __init__(self, inner: StreamingLLMProvider, budget_tokens: int = 1024) -> None:
        """Initialize the wrapper.

        Args:
            inner: The wrapped provider (in production, the
                InstrumentedStreamingLLMProvider).
            budget_tokens: `reasoning_budget_tokens` sent on the first
                attempt of any tools-bearing, thinking-unset request.
                Sourced from `LLMConfig.tool_thinking_budget_tokens` by
                `build_llm_provider`, which never constructs this wrapper at
                all when that config value is `None` ("off") -- so this
                constructor never itself receives a "disabled" sentinel.
        """
        self._inner = inner
        self._budget_tokens = budget_tokens
        self.model = inner.model

    @property
    def inner(self) -> StreamingLLMProvider:
        """Expose the wrapped provider (introspection and tests)."""
        return self._inner

    def _applies(self, request: LLMRequest) -> bool:
        """Only tool-dispatch turns with no caller-set thinking are budgeted."""
        return bool(request.tools) and request.thinking is None

    def _budgeted(self, request: LLMRequest) -> LLMRequest:
        return request.model_copy(
            update={"thinking": ThinkingConfig(budget_tokens=self._budget_tokens)}
        )

    def _unbudgeted(self, request: LLMRequest) -> LLMRequest:
        return request.model_copy(update={"thinking": None})

    async def generate(
        self, request: LLMRequest, *, stream: bool = False
    ) -> AsyncGenerator[LLMResponse, None]:
        """Delegate to inner.generate, retrying once unbudgeted on failure.

        Every inner generator is closed explicitly (`contextlib.aclosing`)
        when this method leaves it -- in particular the first attempt's,
        BEFORE the retry request is sent. Breaking out of an `async for`
        does not close an async generator; it is only finalized later, when
        the event loop's asyncgen finalizer hook schedules its `aclose()`,
        so without this the first attempt's HTTP stream could still be open
        while the retry runs.
        """
        if not self._applies(request):
            async with contextlib.aclosing(self._inner.generate(request, stream=stream)) as inner:
                async for response in inner:
                    yield response
            return

        allowed_tool_names = _allowed_tool_names(request)
        budgeted_request = self._budgeted(request)
        content_yielded = False
        retry_reason: str | None = None

        try:
            async with contextlib.aclosing(
                self._inner.generate(budgeted_request, stream=stream)
            ) as first:
                async for response in first:
                    if response.partial:
                        if response.content:
                            content_yielded = True
                        yield response
                        continue
                    reason = _failure_reason(response, allowed_tool_names)
                    if reason is not None and not content_yielded:
                        retry_reason = reason
                        break
                    yield response
                    return
        except LLMResponseError:
            if content_yielded:
                raise
            retry_reason = "malformed_tool_call_json"

        if retry_reason is None:
            return

        # Outside the try/except above on purpose: a second failure (raised
        # or a second failing final chunk) must propagate/pass through
        # unchanged, not trigger a second retry.
        _record_retry(reason=retry_reason, budget=self._budget_tokens, attempt=1)
        unbudgeted_request = self._unbudgeted(request)
        async with contextlib.aclosing(
            self._inner.generate(unbudgeted_request, stream=stream)
        ) as retry:
            async for retry_response in retry:
                yield retry_response

    def generate_sync(
        self, request: LLMRequest, *, stream: bool = False
    ) -> Generator[LLMResponse, None, None]:
        """Delegate to inner.generate_sync, retrying once unbudgeted on failure.

        The first attempt's generator is closed explicitly
        (`contextlib.closing`) before the retry request is sent, rather than
        left to garbage collection. CPython's reference counting happens to
        close it at the `break` already; the explicit close makes that
        ordering part of the code instead of an interpreter detail.
        """
        if not self._applies(request):
            yield from self._inner.generate_sync(request, stream=stream)
            return

        allowed_tool_names = _allowed_tool_names(request)
        budgeted_request = self._budgeted(request)
        content_yielded = False
        retry_reason: str | None = None

        try:
            with contextlib.closing(
                self._inner.generate_sync(budgeted_request, stream=stream)
            ) as first:
                for response in first:
                    if response.partial:
                        if response.content:
                            content_yielded = True
                        yield response
                        continue
                    reason = _failure_reason(response, allowed_tool_names)
                    if reason is not None and not content_yielded:
                        retry_reason = reason
                        break
                    yield response
                    return
        except LLMResponseError:
            if content_yielded:
                raise
            retry_reason = "malformed_tool_call_json"

        if retry_reason is None:
            return

        _record_retry(reason=retry_reason, budget=self._budget_tokens, attempt=1)
        unbudgeted_request = self._unbudgeted(request)
        yield from self._inner.generate_sync(unbudgeted_request, stream=stream)

    async def health_check(self) -> bool:
        """Pass-through."""
        return await self._inner.health_check()

    async def server_context_size(self) -> int | None:
        """Pass-through."""
        return await self._inner.server_context_size()
