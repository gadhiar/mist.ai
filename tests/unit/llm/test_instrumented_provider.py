"""Unit tests for backend.llm.instrumented_provider.

Validates:
- Transparent delegation: invoke/invoke_sync/generate/generate_sync/health_check
  return the same values the inner provider produces, with no mutation.
- Recording: a single llm_call record is emitted per non-partial response.
- Gating: no records when logger.llm_call_enabled is False.
- Streaming: partial chunks pass through without emission.
- Context threading: llm_call_context values (session_id, event_id, call_site,
  pass_num) surface in the emitted record.
- Unknown fields in the context do not crash record_llm_call.

Spec: cluster-execution-roadmap.md Cluster 5 deliverable 1.
"""

from __future__ import annotations

import json

import pytest

from backend.debug_jsonl_logger import DebugJSONLLogger
from backend.llm.instrumented_provider import (
    InstrumentedStreamingLLMProvider,
    get_llm_call_context,
    llm_call_context,
)
from backend.llm.models import LLMRequest
from tests.mocks.ollama import FakeLLM


def _read_jsonl(path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(ln) for ln in lines if ln.strip()]


def _basic_request(content: str = "hello") -> LLMRequest:
    return LLMRequest(messages=[{"role": "user", "content": content}])


# ---------------------------------------------------------------------------
# Pass-through behavior (instrumentation must not alter delegate semantics)
# ---------------------------------------------------------------------------


class TestTransparentDelegation:
    def test_exposes_inner_model_attribute(self):
        fake = FakeLLM()
        logger = DebugJSONLLogger(None)
        wrapped = InstrumentedStreamingLLMProvider(fake, logger)

        assert wrapped.model == fake.model

    def test_inner_accessor_returns_wrapped_provider(self):
        fake = FakeLLM()
        logger = DebugJSONLLogger(None)
        wrapped = InstrumentedStreamingLLMProvider(fake, logger)

        assert wrapped.inner is fake

    @pytest.mark.asyncio
    async def test_invoke_returns_inner_response_content(self, tmp_path, monkeypatch):
        # Gate off — we're only checking pass-through here.
        monkeypatch.delenv("MIST_DEBUG_LLM_JSONL", raising=False)
        fake = FakeLLM(default_response="inner content")
        logger = DebugJSONLLogger(tmp_path / "d.jsonl")
        wrapped = InstrumentedStreamingLLMProvider(fake, logger)

        response = await wrapped.invoke(_basic_request())

        assert response.content == "inner content"
        assert response.partial is False

    def test_invoke_sync_delegates_to_inner_generate_sync(self, monkeypatch):
        monkeypatch.delenv("MIST_DEBUG_LLM_JSONL", raising=False)
        fake = FakeLLM(default_response="sync content")
        logger = DebugJSONLLogger(None)
        wrapped = InstrumentedStreamingLLMProvider(fake, logger)

        response = wrapped.invoke_sync(_basic_request())

        assert response.content == "sync content"

    @pytest.mark.asyncio
    async def test_streaming_partial_chunks_pass_through(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIST_DEBUG_LLM_JSONL", "1")
        fake = FakeLLM(streaming_chunks=["a", "b", "c"])
        path = tmp_path / "d.jsonl"
        logger = DebugJSONLLogger(path)
        wrapped = InstrumentedStreamingLLMProvider(fake, logger)

        chunks = []
        async for r in wrapped.generate(_basic_request(), stream=True):
            chunks.append(r.content)

        assert chunks == ["a", "b", "c"]
        # No non-partial response was yielded, so no llm_call record should emit.
        assert not path.exists() or _read_jsonl(path) == []

    @pytest.mark.asyncio
    async def test_health_check_passes_through(self):
        fake = FakeLLM()
        logger = DebugJSONLLogger(None)
        wrapped = InstrumentedStreamingLLMProvider(fake, logger)

        assert await wrapped.health_check() is True


# ---------------------------------------------------------------------------
# Recording behavior (emits records when gate is open)
# ---------------------------------------------------------------------------


class TestRecording:
    @pytest.mark.asyncio
    async def test_invoke_emits_one_llm_call_record_when_gate_open(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIST_DEBUG_LLM_JSONL", "1")
        path = tmp_path / "d.jsonl"
        fake = FakeLLM(default_response="ok")
        logger = DebugJSONLLogger(path)
        wrapped = InstrumentedStreamingLLMProvider(fake, logger)

        await wrapped.invoke(_basic_request("hello world"))

        lines = _read_jsonl(path)
        assert len(lines) == 1
        assert lines[0]["phase"] == "llm_call"
        assert lines[0]["response"]["content"] == "ok"
        assert lines[0]["request"]["messages"][0]["content"] == "hello world"

    @pytest.mark.asyncio
    async def test_no_records_when_gate_closed(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MIST_DEBUG_LLM_JSONL", raising=False)
        path = tmp_path / "d.jsonl"
        fake = FakeLLM()
        logger = DebugJSONLLogger(path)
        wrapped = InstrumentedStreamingLLMProvider(fake, logger)

        await wrapped.invoke(_basic_request())

        assert not path.exists() or _read_jsonl(path) == []

    def test_invoke_sync_emits_record_when_gate_open(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIST_DEBUG_LLM_JSONL", "1")
        path = tmp_path / "d.jsonl"
        fake = FakeLLM(default_response="sync ok")
        logger = DebugJSONLLogger(path)
        wrapped = InstrumentedStreamingLLMProvider(fake, logger)

        wrapped.invoke_sync(_basic_request())

        lines = _read_jsonl(path)
        assert len(lines) == 1
        assert lines[0]["phase"] == "llm_call"
        assert lines[0]["response"]["content"] == "sync ok"

    @pytest.mark.asyncio
    async def test_latency_ms_is_non_negative_float(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIST_DEBUG_LLM_JSONL", "1")
        path = tmp_path / "d.jsonl"
        fake = FakeLLM()
        logger = DebugJSONLLogger(path)
        wrapped = InstrumentedStreamingLLMProvider(fake, logger)

        await wrapped.invoke(_basic_request())

        line = _read_jsonl(path)[0]
        assert isinstance(line["latency_ms"], float)
        assert line["latency_ms"] >= 0.0


# ---------------------------------------------------------------------------
# ContextVar threading via llm_call_context
# ---------------------------------------------------------------------------


class TestCallContext:
    def test_get_llm_call_context_defaults_empty(self):
        # Defensive copy should be empty when no context has been set.
        assert get_llm_call_context() == {}

    def test_llm_call_context_sets_and_resets(self):
        with llm_call_context(session_id="s1", call_site="t"):
            ctx = get_llm_call_context()
            assert ctx["session_id"] == "s1"
            assert ctx["call_site"] == "t"

        # On exit the context must be empty again (no leaks across tests).
        assert get_llm_call_context() == {}

    def test_nested_contexts_merge_with_inner_override(self):
        with llm_call_context(session_id="outer", call_site="outer-site"):
            with llm_call_context(call_site="inner-site", pass_num=2):
                ctx = get_llm_call_context()
                assert ctx["session_id"] == "outer"
                assert ctx["call_site"] == "inner-site"
                assert ctx["pass_num"] == 2
            # Inner reset: outer values restored.
            ctx_outer = get_llm_call_context()
            assert ctx_outer["call_site"] == "outer-site"
            assert "pass_num" not in ctx_outer

    @pytest.mark.asyncio
    async def test_context_fields_appear_in_emitted_record(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIST_DEBUG_LLM_JSONL", "1")
        path = tmp_path / "d.jsonl"
        fake = FakeLLM()
        logger = DebugJSONLLogger(path)
        wrapped = InstrumentedStreamingLLMProvider(fake, logger)

        with llm_call_context(
            session_id="sess-777",
            event_id="evt-777",
            call_site="chat.final",
            pass_num=2,
        ):
            await wrapped.invoke(_basic_request())

        line = _read_jsonl(path)[0]
        assert line["session_id"] == "sess-777"
        assert line["event_id"] == "evt-777"
        assert line["call_site"] == "chat.final"
        assert line["pass_num"] == 2

    @pytest.mark.asyncio
    async def test_unknown_context_fields_do_not_leak_into_record(self, tmp_path, monkeypatch):
        """Callers might stash unrelated context; the wrapper filters to the
        allowed keys so a future field addition on the logger is the only
        way to expand the record surface.
        """
        monkeypatch.setenv("MIST_DEBUG_LLM_JSONL", "1")
        path = tmp_path / "d.jsonl"
        fake = FakeLLM()
        logger = DebugJSONLLogger(path)
        wrapped = InstrumentedStreamingLLMProvider(fake, logger)

        with llm_call_context(
            session_id="s",
            some_unrelated_field="garbage",
        ):
            await wrapped.invoke(_basic_request())

        line = _read_jsonl(path)[0]
        # The record must not have adopted the unrelated field.
        assert "some_unrelated_field" not in line
        assert line["session_id"] == "s"
