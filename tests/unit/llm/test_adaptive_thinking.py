"""Tests for backend.llm.adaptive_thinking.AdaptiveThinkingProvider.

Covers (MIS-171 T4 decision 6):
- Applies only to tools-bearing requests with no caller-set thinking.
- Budgets the first attempt at the configured budget_tokens.
- Retries exactly once, unbudgeted, on each of the three failure kinds:
  LLMResponseError, finish_reason="length" with no output, and an unknown
  tool name.
- A second failure propagates (exception re-raised, or the still-failing
  response passed through) rather than retrying again.
- Streaming: no retry once a content partial has reached the caller.
- The process-wide retry counter increments exactly once per retry.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator

import pytest

from backend.errors import LLMResponseError
from backend.llm.adaptive_thinking import (
    AdaptiveThinkingProvider,
    get_adaptive_thinking_retry_count,
    reset_adaptive_thinking_retry_count,
)
from backend.llm.models import LLMRequest, LLMResponse, ThinkingConfig, ToolCall
from backend.llm.provider import StreamingLLMProvider

TOOLS = [
    {"type": "function", "function": {"name": "query_knowledge_graph"}},
    {"type": "function", "function": {"name": "extract_knowledge"}},
]


def _request(*, tools=None, thinking=None, messages=None) -> LLMRequest:
    return LLMRequest(
        messages=messages or [{"role": "user", "content": "hi"}],
        tools=tools,
        thinking=thinking,
    )


class ScriptedProvider(StreamingLLMProvider):
    """Returns one scripted sequence of chunks (or a raise) per call.

    `scripts` is a list of "steps" lists, one per call to generate/generate_sync
    (in order). Each step is either an LLMResponse (yielded) or an Exception
    instance (raised after preceding steps in that call are yielded).
    """

    def __init__(self, scripts: list[list]) -> None:
        self.model = "fake-model"
        self._scripts = scripts
        self.received_requests: list[LLMRequest] = []
        self.received_streams: list[bool] = []
        self._async_call_index = 0
        self._sync_call_index = 0

    async def generate(
        self, request: LLMRequest, *, stream: bool = False
    ) -> AsyncGenerator[LLMResponse, None]:
        self.received_requests.append(request)
        self.received_streams.append(stream)
        steps = self._scripts[self._async_call_index]
        self._async_call_index += 1
        for step in steps:
            if isinstance(step, Exception):
                raise step
            yield step

    def generate_sync(
        self, request: LLMRequest, *, stream: bool = False
    ) -> Generator[LLMResponse, None, None]:
        self.received_requests.append(request)
        self.received_streams.append(stream)
        steps = self._scripts[self._sync_call_index]
        self._sync_call_index += 1
        for step in steps:
            if isinstance(step, Exception):
                raise step
            yield step

    async def health_check(self) -> bool:
        return True

    async def server_context_size(self) -> int | None:
        return None

    @property
    def call_count(self) -> int:
        return max(self._async_call_index, self._sync_call_index)


@pytest.fixture(autouse=True)
def _reset_counter():
    reset_adaptive_thinking_retry_count()
    yield
    reset_adaptive_thinking_retry_count()


def _success_response(content: str = "done") -> LLMResponse:
    return LLMResponse(content=content, partial=False, finish_reason="stop")


async def _collect_async(gen) -> list[LLMResponse]:
    return [r async for r in gen]


# ---------------------------------------------------------------------------
# _applies gating
# ---------------------------------------------------------------------------


class TestAppliesGating:
    @pytest.mark.asyncio
    async def test_tools_request_gets_budget_on_attempt_1_generate_nonstream(self):
        inner = ScriptedProvider([[_success_response()]])
        provider = AdaptiveThinkingProvider(inner, budget_tokens=1024)
        req = _request(tools=TOOLS)

        await _collect_async(provider.generate(req, stream=False))

        assert len(inner.received_requests) == 1
        sent = inner.received_requests[0]
        assert sent.thinking == ThinkingConfig(budget_tokens=1024)
        # Original request object is untouched (pydantic model_copy, not mutation).
        assert req.thinking is None

    @pytest.mark.asyncio
    async def test_tools_request_gets_budget_on_attempt_1_generate_stream(self):
        inner = ScriptedProvider([[_success_response()]])
        provider = AdaptiveThinkingProvider(inner, budget_tokens=1024)
        req = _request(tools=TOOLS)

        await _collect_async(provider.generate(req, stream=True))

        assert inner.received_requests[0].thinking == ThinkingConfig(budget_tokens=1024)

    def test_tools_request_gets_budget_on_attempt_1_generate_sync(self):
        inner = ScriptedProvider([[_success_response()]])
        provider = AdaptiveThinkingProvider(inner, budget_tokens=1024)
        req = _request(tools=TOOLS)

        list(provider.generate_sync(req, stream=False))

        assert inner.received_requests[0].thinking == ThinkingConfig(budget_tokens=1024)

    @pytest.mark.asyncio
    async def test_tools_request_gets_budget_via_invoke(self):
        inner = ScriptedProvider([[_success_response()]])
        provider = AdaptiveThinkingProvider(inner, budget_tokens=1024)
        req = _request(tools=TOOLS)

        result = await provider.invoke(req)

        assert result.content == "done"
        assert inner.received_requests[0].thinking == ThinkingConfig(budget_tokens=1024)

    def test_tools_request_gets_budget_via_invoke_sync(self):
        inner = ScriptedProvider([[_success_response()]])
        provider = AdaptiveThinkingProvider(inner, budget_tokens=1024)
        req = _request(tools=TOOLS)

        result = provider.invoke_sync(req)

        assert result.content == "done"
        assert inner.received_requests[0].thinking == ThinkingConfig(budget_tokens=1024)

    @pytest.mark.asyncio
    async def test_no_tools_request_is_untouched(self):
        inner = ScriptedProvider([[_success_response()]])
        provider = AdaptiveThinkingProvider(inner, budget_tokens=1024)
        req = _request(tools=None)

        await _collect_async(provider.generate(req, stream=False))

        sent = inner.received_requests[0]
        assert sent.thinking is None
        assert sent == req

    @pytest.mark.asyncio
    async def test_caller_set_thinking_is_untouched(self):
        inner = ScriptedProvider([[_success_response()]])
        provider = AdaptiveThinkingProvider(inner, budget_tokens=1024)
        caller_thinking = ThinkingConfig(effort="high")
        req = _request(tools=TOOLS, thinking=caller_thinking)

        await _collect_async(provider.generate(req, stream=False))

        sent = inner.received_requests[0]
        assert sent.thinking == caller_thinking
        assert get_adaptive_thinking_retry_count() == 0


# ---------------------------------------------------------------------------
# Failure kinds -- each triggers exactly one unbudgeted retry
# ---------------------------------------------------------------------------


class TestRetryOnFailure:
    @pytest.mark.asyncio
    async def test_llm_response_error_triggers_one_retry_then_succeeds(self):
        inner = ScriptedProvider(
            [
                [LLMResponseError("malformed tool-call JSON")],
                [_success_response("recovered")],
            ]
        )
        provider = AdaptiveThinkingProvider(inner, budget_tokens=1024)
        req = _request(tools=TOOLS)

        results = await _collect_async(provider.generate(req, stream=False))

        assert len(inner.received_requests) == 2
        assert inner.received_requests[0].thinking == ThinkingConfig(budget_tokens=1024)
        assert inner.received_requests[1].thinking is None
        assert results[-1].content == "recovered"
        assert get_adaptive_thinking_retry_count() == 1

    @pytest.mark.asyncio
    async def test_llm_response_error_second_failure_propagates(self):
        inner = ScriptedProvider(
            [
                [LLMResponseError("first failure")],
                [LLMResponseError("second failure")],
            ]
        )
        provider = AdaptiveThinkingProvider(inner, budget_tokens=1024)
        req = _request(tools=TOOLS)

        with pytest.raises(LLMResponseError, match="second failure"):
            await _collect_async(provider.generate(req, stream=False))

        assert len(inner.received_requests) == 2
        assert get_adaptive_thinking_retry_count() == 1

    @pytest.mark.asyncio
    async def test_length_no_output_triggers_one_retry_then_succeeds(self):
        length_failure = LLMResponse(content=None, tool_calls=None, finish_reason="length")
        inner = ScriptedProvider([[length_failure], [_success_response("recovered")]])
        provider = AdaptiveThinkingProvider(inner, budget_tokens=1024)
        req = _request(tools=TOOLS)

        results = await _collect_async(provider.generate(req, stream=False))

        assert len(inner.received_requests) == 2
        assert inner.received_requests[1].thinking is None
        assert results[-1].content == "recovered"
        assert get_adaptive_thinking_retry_count() == 1

    @pytest.mark.asyncio
    async def test_length_no_output_second_failure_passes_through(self):
        length_failure = LLMResponse(content=None, tool_calls=None, finish_reason="length")
        inner = ScriptedProvider([[length_failure], [length_failure]])
        provider = AdaptiveThinkingProvider(inner, budget_tokens=1024)
        req = _request(tools=TOOLS)

        results = await _collect_async(provider.generate(req, stream=False))

        assert len(inner.received_requests) == 2
        assert results[-1].finish_reason == "length"
        assert get_adaptive_thinking_retry_count() == 1

    @pytest.mark.asyncio
    async def test_unknown_tool_name_triggers_one_retry_then_succeeds(self):
        bad_call = LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="1", name="not_a_real_tool", arguments={})],
            finish_reason="tool_calls",
        )
        inner = ScriptedProvider([[bad_call], [_success_response("recovered")]])
        provider = AdaptiveThinkingProvider(inner, budget_tokens=1024)
        req = _request(tools=TOOLS)

        results = await _collect_async(provider.generate(req, stream=False))

        assert len(inner.received_requests) == 2
        assert results[-1].content == "recovered"
        assert get_adaptive_thinking_retry_count() == 1

    @pytest.mark.asyncio
    async def test_known_tool_name_does_not_retry(self):
        good_call = LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="1", name="query_knowledge_graph", arguments={})],
            finish_reason="tool_calls",
        )
        inner = ScriptedProvider([[good_call]])
        provider = AdaptiveThinkingProvider(inner, budget_tokens=1024)
        req = _request(tools=TOOLS)

        results = await _collect_async(provider.generate(req, stream=False))

        assert len(inner.received_requests) == 1
        assert results[-1].tool_calls[0].name == "query_knowledge_graph"
        assert get_adaptive_thinking_retry_count() == 0

    def test_sync_llm_response_error_triggers_one_retry(self):
        inner = ScriptedProvider(
            [[LLMResponseError("malformed")], [_success_response("recovered")]]
        )
        provider = AdaptiveThinkingProvider(inner, budget_tokens=1024)
        req = _request(tools=TOOLS)

        results = list(provider.generate_sync(req, stream=False))

        assert len(inner.received_requests) == 2
        assert results[-1].content == "recovered"
        assert get_adaptive_thinking_retry_count() == 1


# ---------------------------------------------------------------------------
# Streaming: no retry once content has been yielded to the caller
# ---------------------------------------------------------------------------


class TestStreamingContentAlreadyYielded:
    @pytest.mark.asyncio
    async def test_length_failure_after_content_yielded_passes_through_no_retry(self):
        partial = LLMResponse(content="partial text", partial=True)
        length_failure = LLMResponse(content=None, tool_calls=None, finish_reason="length")
        inner = ScriptedProvider([[partial, length_failure]])
        provider = AdaptiveThinkingProvider(inner, budget_tokens=1024)
        req = _request(tools=TOOLS)

        results = await _collect_async(provider.generate(req, stream=True))

        assert len(inner.received_requests) == 1  # no retry call made
        assert results[0].content == "partial text"
        assert results[-1].finish_reason == "length"
        assert get_adaptive_thinking_retry_count() == 0

    @pytest.mark.asyncio
    async def test_llm_response_error_after_content_yielded_reraises_no_retry(self):
        partial = LLMResponse(content="partial text", partial=True)
        inner = ScriptedProvider([[partial, LLMResponseError("malformed after content")]])
        provider = AdaptiveThinkingProvider(inner, budget_tokens=1024)
        req = _request(tools=TOOLS)

        with pytest.raises(LLMResponseError, match="malformed after content"):
            await _collect_async(provider.generate(req, stream=True))

        assert len(inner.received_requests) == 1  # no retry call made
        assert get_adaptive_thinking_retry_count() == 0

    @pytest.mark.asyncio
    async def test_length_failure_before_any_content_retries(self):
        """The common real failure mode: reasoning exhausts budget, no content ever yielded."""
        length_failure = LLMResponse(content=None, tool_calls=None, finish_reason="length")
        inner = ScriptedProvider([[length_failure], [_success_response("recovered")]])
        provider = AdaptiveThinkingProvider(inner, budget_tokens=1024)
        req = _request(tools=TOOLS)

        results = await _collect_async(provider.generate(req, stream=True))

        assert len(inner.received_requests) == 2
        assert results[-1].content == "recovered"


# ---------------------------------------------------------------------------
# UsageMetadata passthrough sanity, retry counter
# ---------------------------------------------------------------------------


class TestRetryCounter:
    @pytest.mark.asyncio
    async def test_counter_increments_once_per_retry_across_calls(self):
        assert get_adaptive_thinking_retry_count() == 0
        inner = ScriptedProvider(
            [
                [LLMResponseError("fail 1")],
                [_success_response()],
                [LLMResponseError("fail 2")],
                [_success_response()],
            ]
        )
        provider = AdaptiveThinkingProvider(inner, budget_tokens=1024)

        await _collect_async(provider.generate(_request(tools=TOOLS), stream=False))
        await _collect_async(provider.generate(_request(tools=TOOLS), stream=False))

        assert get_adaptive_thinking_retry_count() == 2

    def test_reset_zeroes_counter(self):
        reset_adaptive_thinking_retry_count()
        assert get_adaptive_thinking_retry_count() == 0


# ---------------------------------------------------------------------------
# A server that ignores `reasoning_budget_tokens` (e.g. live mist-llm at b8808,
# reasoning_format none). It answers the budgeted request exactly as if no
# budget had been sent: no reasoning_content, a normal tool call or reply.
# The wrapper must pass that through with ONE call, no retry, no error.
# ---------------------------------------------------------------------------


def _ignored_budget_tool_call() -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(id="c1", name="query_knowledge_graph", arguments={"q": "x"})],
        partial=False,
        finish_reason="tool_calls",
        reasoning_content=None,
    )


class TestServerIgnoresBudgetField:
    @pytest.mark.asyncio
    async def test_nonstream_tool_call_passes_through_without_retry(self):
        inner = ScriptedProvider([[_ignored_budget_tool_call()]])
        provider = AdaptiveThinkingProvider(inner, budget_tokens=1024)

        out = await _collect_async(provider.generate(_request(tools=TOOLS), stream=False))

        assert inner.call_count == 1
        assert get_adaptive_thinking_retry_count() == 0
        assert [r.tool_calls[0].name for r in out if r.tool_calls] == ["query_knowledge_graph"]

    @pytest.mark.asyncio
    async def test_stream_plain_reply_passes_through_without_retry(self):
        inner = ScriptedProvider(
            [
                [
                    LLMResponse(content="Hello ", partial=True),
                    LLMResponse(content="there.", partial=True),
                    LLMResponse(content=None, partial=False, finish_reason="stop"),
                ]
            ]
        )
        provider = AdaptiveThinkingProvider(inner, budget_tokens=1024)

        out = await _collect_async(provider.generate(_request(tools=TOOLS), stream=True))

        assert inner.call_count == 1
        assert get_adaptive_thinking_retry_count() == 0
        assert "".join(r.content for r in out if r.partial and r.content) == "Hello there."

    def test_sync_invoke_passes_through_without_retry(self):
        inner = ScriptedProvider([[_ignored_budget_tool_call()]])
        provider = AdaptiveThinkingProvider(inner, budget_tokens=1024)

        response = provider.invoke_sync(_request(tools=TOOLS))

        assert inner.call_count == 1
        assert get_adaptive_thinking_retry_count() == 0
        assert response.tool_calls[0].name == "query_knowledge_graph"


# ---------------------------------------------------------------------------
# Generator lifecycle: the first attempt is closed before the retry starts
# ---------------------------------------------------------------------------


class LifecycleProvider(StreamingLLMProvider):
    """Records when each call's generator starts and when it is closed.

    The first call yields a failing final chunk (length exhausted, no output)
    and would then yield more -- so the wrapper's `break` leaves it suspended
    at a `yield`, not finished. Its `finally` records the close. The second
    call succeeds.
    """

    def __init__(self) -> None:
        self.model = "fake-model"
        self.log: list[str] = []
        self._calls = 0

    def _script(self) -> tuple[int, list[LLMResponse]]:
        self._calls += 1
        n = self._calls
        if n == 1:
            failing = LLMResponse(content=None, partial=False, finish_reason="length")
            return n, [failing, _success_response("never reached")]
        return n, [_success_response("retried")]

    async def generate(
        self, request: LLMRequest, *, stream: bool = False
    ) -> AsyncGenerator[LLMResponse, None]:
        n, steps = self._script()
        self.log.append(f"start-{n}")
        try:
            for step in steps:
                yield step
        finally:
            self.log.append(f"closed-{n}")

    def generate_sync(
        self, request: LLMRequest, *, stream: bool = False
    ) -> Generator[LLMResponse, None, None]:
        n, steps = self._script()
        self.log.append(f"start-{n}")
        try:
            yield from steps
        finally:
            self.log.append(f"closed-{n}")

    async def health_check(self) -> bool:
        return True

    async def server_context_size(self) -> int | None:
        return None


class TestFirstAttemptClosedBeforeRetry:
    """Review finding 6: breaking out of an `async for` does not close the
    async generator; before the fix the first attempt was only finalized
    later by the event loop's asyncgen hook, after the retry had started.
    """

    @pytest.mark.asyncio
    async def test_async_first_attempt_closed_before_retry_starts(self):
        inner = LifecycleProvider()
        provider = AdaptiveThinkingProvider(inner, budget_tokens=1024)

        responses = await _collect_async(provider.generate(_request(tools=TOOLS), stream=True))

        assert [r.content for r in responses] == ["retried"]
        assert inner.log[:3] == ["start-1", "closed-1", "start-2"], inner.log

    def test_sync_first_attempt_closed_before_retry_starts(self):
        inner = LifecycleProvider()
        provider = AdaptiveThinkingProvider(inner, budget_tokens=1024)

        responses = list(provider.generate_sync(_request(tools=TOOLS), stream=True))

        assert [r.content for r in responses] == ["retried"]
        assert inner.log[:3] == ["start-1", "closed-1", "start-2"], inner.log


# ---------------------------------------------------------------------------
# model attribute and passthrough methods
# ---------------------------------------------------------------------------


class TestPassthrough:
    def test_model_attribute_forwarded(self):
        inner = ScriptedProvider([])
        provider = AdaptiveThinkingProvider(inner, budget_tokens=1024)
        assert provider.model == "fake-model"

    @pytest.mark.asyncio
    async def test_health_check_forwarded(self):
        inner = ScriptedProvider([])
        provider = AdaptiveThinkingProvider(inner, budget_tokens=1024)
        assert await provider.health_check() is True

    @pytest.mark.asyncio
    async def test_server_context_size_forwarded(self):
        inner = ScriptedProvider([])
        provider = AdaptiveThinkingProvider(inner, budget_tokens=1024)
        assert await provider.server_context_size() is None

    def test_inner_property_exposes_wrapped_provider(self):
        inner = ScriptedProvider([])
        provider = AdaptiveThinkingProvider(inner, budget_tokens=1024)
        assert provider.inner is inner
