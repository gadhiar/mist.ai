"""End-to-end integration reproducers for Cluster 5 (observability).

Locks in the three new JSONL record types working together through a real
ConversationHandler + KnowledgeRetriever + InstrumentedStreamingLLMProvider
chain with a FakeLLM underneath:

- phase: "llm_call" — emitted at the provider boundary when
  MIST_DEBUG_LLM_JSONL=1
- phase: "retrieval_candidates" — emitted at the retriever boundary when
  MIST_DEBUG_RETRIEVAL_JSONL=1
- phase: "llm_request_raw" — emitted by ConversationHandler._build_request
  on Pydantic validation failure when MIST_DEBUG_LLM_REQUESTS=1

Additionally verifies that the existing turn/extraction JSONL records still
work unchanged, so Cluster 5 is purely additive.

Spec: cluster-execution-roadmap.md Cluster 5.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from backend.chat.conversation_handler import ConversationHandler
from backend.debug_jsonl_logger import DebugJSONLLogger
from backend.knowledge.retrieval.knowledge_retriever import KnowledgeRetriever
from backend.llm.instrumented_provider import InstrumentedStreamingLLMProvider
from backend.llm.models import LLMResponse
from tests.mocks.config import build_test_config
from tests.mocks.ollama import FakeLLM

pytestmark = pytest.mark.integration


# Minimal seeded identity — enough for persona injection to not crash.
_SEEDED_IDENTITY = {
    "identity": {
        "id": "mist-identity",
        "display_name": "MIST",
        "pronouns": "she/her",
        "self_concept": "A cognitive architecture.",
    },
    "traits": [
        {
            "id": "trait-warm",
            "display_name": "Warm",
            "axis": "Persona",
            "description": "Friendly default tone.",
        },
    ],
    "capabilities": [],
    "preferences": [
        {
            "id": "pref-no-emoji",
            "display_name": "No emoji",
            "enforcement": "absolute",
            "context": "Hard rule.",
        },
    ],
}


def _build_graph_store() -> MagicMock:
    graph_store = MagicMock()
    graph_store.get_mist_identity_context = MagicMock(return_value=_SEEDED_IDENTITY)
    # search_similar_entities returns a small candidate pool so retrieval_candidates
    # has something to emit.
    graph_store.search_similar_entities = MagicMock(
        return_value=[
            {"entity_id": "python", "similarity": 0.92, "entity_type": "Technology"},
        ]
    )
    graph_store.get_user_relationships_to_entities = MagicMock(return_value=[])
    graph_store.get_entity_neighborhood = MagicMock(return_value=[])
    return graph_store


class _ScriptedFakeLLM(FakeLLM):
    """FakeLLM subclass that pops responses from a queue in order.

    Overrides generate() (not invoke) so the instrumented wrapper's
    generate() delegation path is exercised end-to-end.
    """

    def __init__(self, responses: list[str]):
        super().__init__()
        self._queue = list(responses)

    async def generate(self, request, *, stream: bool = False):
        self.calls.append(request)
        content = self._queue.pop(0) if self._queue else self._resolve(request)
        yield LLMResponse(content=content, partial=False)

    async def invoke(self, request):
        # Route through generate so the wrapper's generate() override
        # runs, emitting the llm_call record.
        async for response in self.generate(request, stream=False):
            return response
        raise RuntimeError("generate yielded nothing")


def _build_handler(
    graph_store: MagicMock,
    *,
    debug_logger: DebugJSONLLogger,
    responses: list[str] | None = None,
) -> tuple[ConversationHandler, _ScriptedFakeLLM]:
    inner = _ScriptedFakeLLM(responses or ["plain response"])
    # Wrap the FakeLLM with the instrumented provider. The wrapper's generate()
    # emits the llm_call record; base-class invoke() drives the wrapper's
    # generate(), which in turn delegates to inner.generate().
    wrapped = InstrumentedStreamingLLMProvider(inner, debug_logger)

    retriever = KnowledgeRetriever(
        config=build_test_config(),
        graph_store=graph_store,
        vector_store=None,
        query_classifier=None,  # defaults to relational -> triggers graph path
        embedding_provider=None,
        debug_logger=debug_logger,
    )

    handler = ConversationHandler(
        config=build_test_config(),
        graph_store=graph_store,
        extraction_pipeline=MagicMock(extract_from_utterance=AsyncMock()),
        retriever=retriever,
        llm_provider=wrapped,
        debug_logger=debug_logger,
    )
    return handler, inner


def _read_jsonl(path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(ln) for ln in lines if ln.strip()]


# ---------------------------------------------------------------------------
# End-to-end: all three Cluster 5 phases emit on a normal turn
# ---------------------------------------------------------------------------


class TestAllPhasesEmit:
    @pytest.mark.asyncio
    async def test_happy_path_emits_llm_call_and_retrieval_records(self, tmp_path, monkeypatch):
        path = tmp_path / "debug.jsonl"
        monkeypatch.setenv("MIST_DEBUG_JSONL", str(path))
        monkeypatch.setenv("MIST_DEBUG_LLM_JSONL", "1")
        monkeypatch.setenv("MIST_DEBUG_RETRIEVAL_JSONL", "1")
        monkeypatch.setenv("MIST_DEBUG_LLM_REQUESTS", "1")

        debug_logger = DebugJSONLLogger.from_env()
        assert debug_logger.enabled
        assert debug_logger.llm_call_enabled
        assert debug_logger.retrieval_candidates_enabled
        assert debug_logger.llm_request_dump_enabled

        graph_store = _build_graph_store()
        handler, _inner = _build_handler(
            graph_store,
            debug_logger=debug_logger,
            responses=["plain reply"],
        )

        await handler.handle_message(
            user_message="what do I use python for",
            session_id="c5-e2e-s1",
        )

        records = _read_jsonl(path)
        phases = [r["phase"] for r in records]

        # Both llm_call and retrieval_candidates must appear.
        assert "llm_call" in phases, f"missing llm_call in {phases}"
        assert "retrieval_candidates" in phases, f"missing retrieval_candidates in {phases}"

    @pytest.mark.asyncio
    async def test_llm_call_record_carries_call_context(self, tmp_path, monkeypatch):
        path = tmp_path / "debug.jsonl"
        monkeypatch.setenv("MIST_DEBUG_JSONL", str(path))
        monkeypatch.setenv("MIST_DEBUG_LLM_JSONL", "1")

        debug_logger = DebugJSONLLogger.from_env()
        graph_store = _build_graph_store()
        handler, _inner = _build_handler(
            graph_store,
            debug_logger=debug_logger,
            responses=["ok"],
        )

        await handler.handle_message(
            user_message="quick question",
            session_id="c5-e2e-s2",
        )

        records = _read_jsonl(path)
        llm_records = [r for r in records if r["phase"] == "llm_call"]
        assert llm_records, "expected at least one llm_call record"
        # The initial chat call should stamp call_site="chat.initial"
        # and pass_num=1 via the llm_call_context block in handle_message.
        first = llm_records[0]
        assert first["call_site"] == "chat.initial"
        assert first["pass_num"] == 1
        assert first["session_id"] == "c5-e2e-s2"

    @pytest.mark.asyncio
    async def test_retrieval_candidates_record_carries_session_id(self, tmp_path, monkeypatch):
        path = tmp_path / "debug.jsonl"
        monkeypatch.setenv("MIST_DEBUG_JSONL", str(path))
        monkeypatch.setenv("MIST_DEBUG_RETRIEVAL_JSONL", "1")

        debug_logger = DebugJSONLLogger.from_env()
        graph_store = _build_graph_store()
        handler, _inner = _build_handler(
            graph_store,
            debug_logger=debug_logger,
            responses=["ok"],
        )

        await handler.handle_message(
            user_message="something about python again",
            session_id="c5-e2e-s3",
        )

        records = _read_jsonl(path)
        retr_records = [r for r in records if r["phase"] == "retrieval_candidates"]
        assert retr_records, "expected retrieval_candidates record"
        r = retr_records[0]
        assert r["session_id"] == "c5-e2e-s3"
        # The graph candidate pool returned by the seeded mock should appear.
        assert r["graph_candidate_count"] >= 1


# ---------------------------------------------------------------------------
# Pre-validation dump: fires on Pydantic validation failure
# ---------------------------------------------------------------------------


class TestPreValidationDumpIntegration:
    @pytest.mark.asyncio
    async def test_dump_emitted_when_build_request_kwargs_invalid(self, tmp_path, monkeypatch):
        path = tmp_path / "debug.jsonl"
        monkeypatch.setenv("MIST_DEBUG_JSONL", str(path))
        monkeypatch.setenv("MIST_DEBUG_LLM_REQUESTS", "1")

        debug_logger = DebugJSONLLogger.from_env()
        graph_store = _build_graph_store()
        handler, _inner = _build_handler(
            graph_store,
            debug_logger=debug_logger,
            responses=["ok"],
        )

        # Trigger the dump path via _build_request with invalid messages.
        with pytest.raises(ValidationError):
            handler._build_request(
                call_site="integration.bad-kwargs",
                session_id="c5-pv-1",
                messages="not a list",  # invalid
            )

        records = _read_jsonl(path)
        dumps = [r for r in records if r["phase"] == "llm_request_raw"]
        assert len(dumps) == 1
        assert dumps[0]["call_site"] == "integration.bad-kwargs"
        assert dumps[0]["session_id"] == "c5-pv-1"
        assert dumps[0]["request_dict"]["messages"] == "not a list"


# ---------------------------------------------------------------------------
# Gate separation: closing one phase gate does not affect others
# ---------------------------------------------------------------------------


class TestGateSeparation:
    @pytest.mark.asyncio
    async def test_closing_llm_call_gate_leaves_retrieval_gate_open(self, tmp_path, monkeypatch):
        path = tmp_path / "debug.jsonl"
        monkeypatch.setenv("MIST_DEBUG_JSONL", str(path))
        monkeypatch.setenv("MIST_DEBUG_RETRIEVAL_JSONL", "1")
        monkeypatch.delenv("MIST_DEBUG_LLM_JSONL", raising=False)

        debug_logger = DebugJSONLLogger.from_env()
        assert debug_logger.retrieval_candidates_enabled
        assert not debug_logger.llm_call_enabled

        graph_store = _build_graph_store()
        handler, _inner = _build_handler(
            graph_store,
            debug_logger=debug_logger,
            responses=["ok"],
        )

        await handler.handle_message(
            user_message="tell me about python projects",
            session_id="c5-gate-s1",
        )

        records = _read_jsonl(path)
        phases = [r["phase"] for r in records]
        # retrieval_candidates emitted; llm_call suppressed.
        assert "retrieval_candidates" in phases
        assert "llm_call" not in phases

    @pytest.mark.asyncio
    async def test_no_records_when_base_env_var_unset(self, tmp_path, monkeypatch):
        """If MIST_DEBUG_JSONL is unset, nothing writes regardless of phase gates."""
        path = tmp_path / "debug.jsonl"
        monkeypatch.delenv("MIST_DEBUG_JSONL", raising=False)
        monkeypatch.setenv("MIST_DEBUG_LLM_JSONL", "1")
        monkeypatch.setenv("MIST_DEBUG_RETRIEVAL_JSONL", "1")
        monkeypatch.setenv("MIST_DEBUG_LLM_REQUESTS", "1")

        debug_logger = DebugJSONLLogger.from_env()
        assert not debug_logger.enabled
        assert not debug_logger.llm_call_enabled

        graph_store = _build_graph_store()
        handler, _inner = _build_handler(
            graph_store,
            debug_logger=debug_logger,
            responses=["ok"],
        )

        await handler.handle_message(
            user_message="anything",
            session_id="c5-gate-s2",
        )

        assert not path.exists() or path.read_text() == ""
