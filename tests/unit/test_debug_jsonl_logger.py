"""Unit tests for backend.debug_jsonl_logger.

Validates:
- Disabled (env unset) returns no-op record with zero file I/O
- Enabled (path set) emits valid JSONL lines keyed by event_id
- Turn + extraction records emit separately and both carry event_id
- Polymorphic extraction result handling (ValidationResult | CurationResult)
- Thread safety when multiple turns interleave flushes
- Idempotent flush (calling twice does not duplicate the line)

Spec: ~/.claude/plans/nimble-forage-cinder.md Part 3 / Part 6 Task 5.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass

import pytest

from backend.debug_jsonl_logger import DebugJSONLLogger, _NoOpTurnRecord

# ---------------------------------------------------------------------------
# Lightweight fakes — avoid importing RetrievalResult / LLMResponse which pull
# in heavy backend modules.
# ---------------------------------------------------------------------------


@dataclass
class FakeFact:
    subject: str = "user"
    predicate: str = "USES"
    object: str = "python"
    similarity_score: float = 0.85
    graph_distance: int = 0


@dataclass
class FakeRetrievalResult:
    query: str = "test"
    facts: list[FakeFact] | None = None
    entities_found: int = 1
    total_facts: int = 1
    retrieval_time_ms: float = 12.3
    vector_search_time_ms: float = 4.1
    graph_traversal_time_ms: float = 8.2

    def __post_init__(self):
        if self.facts is None:
            self.facts = [FakeFact()]


@dataclass
class FakeToolCall:
    name: str = "query_knowledge_graph"
    id: str = "tc-1"
    arguments: dict = None

    def __post_init__(self):
        if self.arguments is None:
            self.arguments = {"query": "test"}


@dataclass
class FakeUsage:
    input_tokens: int = 100
    output_tokens: int = 50


@dataclass
class FakeLLMResponse:
    content: str | None = "hello"
    tool_calls: list[FakeToolCall] | None = None
    partial: bool = False
    usage: FakeUsage | None = None


@dataclass
class FakeEntity:
    name: str = "python"
    confidence: float = 0.9


@dataclass
class FakeRelationship:
    subject: str = "user"
    predicate: str = "USES"
    object: str = "python"
    confidence: float = 0.85


@dataclass
class FakeValidationResult:
    entities: list[FakeEntity] = None
    relationships: list[FakeRelationship] = None
    valid: bool = True

    def __post_init__(self):
        if self.entities is None:
            self.entities = [FakeEntity(), FakeEntity(name="java", confidence=0.7)]
        if self.relationships is None:
            self.relationships = [FakeRelationship()]


@dataclass
class FakeWriteResult:
    entities_created: int = 2
    entities_updated: int = 1
    relationships_created: int = 1
    relationships_updated: int = 0
    relationships_superseded: int = 0


@dataclass
class FakeCurationResult:
    validation_result: FakeValidationResult = None
    write_result: FakeWriteResult = None
    curation_time_ms: float = 45.6

    def __post_init__(self):
        if self.validation_result is None:
            self.validation_result = FakeValidationResult()
        if self.write_result is None:
            self.write_result = FakeWriteResult()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_jsonl(path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# ---------------------------------------------------------------------------
# Disabled behavior
# ---------------------------------------------------------------------------


class TestDisabledLogger:
    def test_from_env_unset_is_disabled(self, monkeypatch):
        monkeypatch.delenv("MIST_DEBUG_JSONL", raising=False)
        logger = DebugJSONLLogger.from_env()
        assert logger.enabled is False
        assert logger.path is None

    def test_begin_turn_returns_noop_when_disabled(self):
        logger = DebugJSONLLogger(None)
        record = logger.begin_turn(event_id="e", session_id="s", user_id="u", utterance="hi")
        assert isinstance(record, _NoOpTurnRecord)

    def test_noop_methods_do_not_raise(self):
        record = _NoOpTurnRecord()
        record.record_retrieval(FakeRetrievalResult())
        record.record_llm_response(FakeLLMResponse(), pass_num=1)
        record.record_extraction(FakeValidationResult(), duration_ms=1.0)
        record.flush_turn()
        record.flush_extraction()

    def test_disabled_logger_writes_nothing(self, tmp_path):
        path = tmp_path / "should_not_exist.jsonl"
        logger = DebugJSONLLogger(None)
        record = logger.begin_turn(event_id="e1", session_id="s1", user_id="u1", utterance="hi")
        record.record_retrieval(FakeRetrievalResult())
        record.record_llm_response(FakeLLMResponse(), pass_num=1)
        record.flush_turn()
        assert not path.exists()


# ---------------------------------------------------------------------------
# Enabled behavior — turn record
# ---------------------------------------------------------------------------


class TestEnabledTurnRecord:
    def test_from_env_uses_path(self, monkeypatch, tmp_path):
        target = tmp_path / "debug.jsonl"
        monkeypatch.setenv("MIST_DEBUG_JSONL", str(target))
        logger = DebugJSONLLogger.from_env()
        assert logger.enabled is True
        assert logger.path == target

    def test_turn_flush_emits_one_jsonl_line(self, tmp_path):
        path = tmp_path / "debug.jsonl"
        logger = DebugJSONLLogger(path)
        record = logger.begin_turn(
            event_id="evt-1", session_id="sess-1", user_id="user", utterance="hi"
        )
        record.record_retrieval(FakeRetrievalResult())
        record.record_llm_response(FakeLLMResponse(), pass_num=1, timing_ms=12.3)
        record.flush_turn()

        lines = _read_jsonl(path)
        assert len(lines) == 1
        assert lines[0]["phase"] == "turn"
        assert lines[0]["event_id"] == "evt-1"
        assert lines[0]["session_id"] == "sess-1"
        assert lines[0]["user_id"] == "user"
        assert lines[0]["utterance"] == "hi"

    def test_turn_record_captures_retrieval_metadata(self, tmp_path):
        path = tmp_path / "debug.jsonl"
        logger = DebugJSONLLogger(path)
        record = logger.begin_turn(event_id="e", session_id="s", user_id="u", utterance="test")
        record.record_retrieval(FakeRetrievalResult(total_facts=3, retrieval_time_ms=99.0))
        record.flush_turn()

        retrieval = _read_jsonl(path)[0]["retrieval"]
        assert retrieval["total_facts"] == 3
        assert retrieval["retrieval_time_ms"] == 99.0
        assert retrieval["top_facts"][0]["subject"] == "user"

    def test_turn_record_captures_multiple_llm_passes(self, tmp_path):
        path = tmp_path / "debug.jsonl"
        logger = DebugJSONLLogger(path)
        record = logger.begin_turn(event_id="e", session_id="s", user_id="u", utterance="test")
        record.record_llm_response(
            FakeLLMResponse(content="first", tool_calls=[FakeToolCall()]),
            pass_num=1,
        )
        record.record_llm_response(
            FakeLLMResponse(content="second"),
            pass_num=2,
        )
        record.flush_turn()

        passes = _read_jsonl(path)[0]["llm_passes"]
        assert len(passes) == 2
        assert passes[0]["pass"] == 1
        assert passes[0]["tool_calls"][0]["name"] == "query_knowledge_graph"
        assert passes[0]["tool_calls"][0]["arg_keys"] == ["query"]
        assert passes[1]["pass"] == 2
        assert passes[1]["content_len"] == len("second")

    def test_flush_turn_is_idempotent(self, tmp_path):
        path = tmp_path / "debug.jsonl"
        logger = DebugJSONLLogger(path)
        record = logger.begin_turn(event_id="e", session_id="s", user_id="u", utterance="x")
        record.flush_turn()
        record.flush_turn()
        record.flush_turn()
        assert len(_read_jsonl(path)) == 1


# ---------------------------------------------------------------------------
# Enabled behavior — extraction record
# ---------------------------------------------------------------------------


class TestEnabledExtractionRecord:
    def test_extraction_with_validation_result_only(self, tmp_path):
        path = tmp_path / "debug.jsonl"
        logger = DebugJSONLLogger(path)
        record = logger.begin_turn(event_id="e", session_id="s", user_id="u", utterance="x")
        record.record_extraction(FakeValidationResult(), duration_ms=55.0)
        record.flush_extraction()

        line = _read_jsonl(path)[0]
        assert line["phase"] == "extraction"
        assert line["extraction"]["entity_count"] == 2
        assert line["extraction"]["relationship_count"] == 1
        assert line["extraction"]["avg_confidence"] == pytest.approx((0.9 + 0.7 + 0.85) / 3)
        assert line["extraction"]["duration_ms"] == 55.0
        assert line["graph_writes"] is None

    def test_extraction_with_curation_result_unwraps_write_counts(self, tmp_path):
        path = tmp_path / "debug.jsonl"
        logger = DebugJSONLLogger(path)
        record = logger.begin_turn(event_id="e", session_id="s", user_id="u", utterance="x")
        record.record_extraction(FakeCurationResult(), duration_ms=120.0)
        record.flush_extraction()

        line = _read_jsonl(path)[0]
        assert line["extraction"]["entity_count"] == 2
        assert line["graph_writes"] is not None
        assert line["graph_writes"]["entities_created"] == 2
        assert line["graph_writes"]["relationships_created"] == 1

    def test_extraction_and_turn_records_share_event_id(self, tmp_path):
        path = tmp_path / "debug.jsonl"
        logger = DebugJSONLLogger(path)
        record = logger.begin_turn(event_id="evt-42", session_id="s", user_id="u", utterance="x")
        record.record_llm_response(FakeLLMResponse(), pass_num=1)
        record.flush_turn()
        record.record_extraction(FakeCurationResult(), duration_ms=10.0)
        record.flush_extraction()

        lines = _read_jsonl(path)
        assert len(lines) == 2
        assert {ln["phase"] for ln in lines} == {"turn", "extraction"}
        assert all(ln["event_id"] == "evt-42" for ln in lines)

    def test_flush_extraction_is_idempotent(self, tmp_path):
        path = tmp_path / "debug.jsonl"
        logger = DebugJSONLLogger(path)
        record = logger.begin_turn(event_id="e", session_id="s", user_id="u", utterance="x")
        record.record_extraction(FakeValidationResult(), duration_ms=1.0)
        record.flush_extraction()
        record.flush_extraction()
        assert len(_read_jsonl(path)) == 1

    def test_parse_ok_false_propagates(self, tmp_path):
        path = tmp_path / "debug.jsonl"
        logger = DebugJSONLLogger(path)
        record = logger.begin_turn(event_id="e", session_id="s", user_id="u", utterance="x")
        record.record_extraction(FakeValidationResult(), duration_ms=1.0, parse_ok=False)
        record.flush_extraction()

        assert _read_jsonl(path)[0]["extraction"]["parse_ok"] is False


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_concurrent_turns_produce_well_formed_lines(self, tmp_path):
        path = tmp_path / "debug.jsonl"
        logger = DebugJSONLLogger(path)
        n_threads = 8
        per_thread = 10

        def worker(tid: int):
            for i in range(per_thread):
                r = logger.begin_turn(
                    event_id=f"t{tid}-e{i}",
                    session_id=f"s{tid}",
                    user_id="u",
                    utterance=f"msg {tid}.{i}",
                )
                r.record_llm_response(FakeLLMResponse(), pass_num=1)
                r.flush_turn()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        lines = _read_jsonl(path)
        assert len(lines) == n_threads * per_thread
        # Every line is valid JSON with the expected event_id pattern.
        for line in lines:
            assert line["phase"] == "turn"
            assert line["event_id"].startswith("t")


# ---------------------------------------------------------------------------
# Cluster 5 — phase gates
# ---------------------------------------------------------------------------


class TestPhaseGates:
    """The three Cluster 5 record types each require MIST_DEBUG_JSONL to be set
    AND their own boolean env var to be truthy. Neither alone is sufficient.
    """

    def test_llm_call_gate_requires_base_env_and_phase_env(self, tmp_path, monkeypatch):
        path = tmp_path / "debug.jsonl"
        logger = DebugJSONLLogger(path)
        # Base enabled (path set), but phase flag not set -> gate closed.
        monkeypatch.delenv("MIST_DEBUG_LLM_JSONL", raising=False)
        assert logger.llm_call_enabled is False

        monkeypatch.setenv("MIST_DEBUG_LLM_JSONL", "1")
        assert logger.llm_call_enabled is True

    def test_llm_call_gate_closed_when_base_disabled(self, monkeypatch):
        # No output path -> base disabled -> phase gate is also closed.
        monkeypatch.setenv("MIST_DEBUG_LLM_JSONL", "1")
        logger = DebugJSONLLogger(None)
        assert logger.llm_call_enabled is False

    def test_retrieval_candidates_gate_independent_of_llm_call_gate(self, tmp_path, monkeypatch):
        path = tmp_path / "debug.jsonl"
        logger = DebugJSONLLogger(path)
        monkeypatch.setenv("MIST_DEBUG_RETRIEVAL_JSONL", "1")
        monkeypatch.delenv("MIST_DEBUG_LLM_JSONL", raising=False)

        assert logger.retrieval_candidates_enabled is True
        assert logger.llm_call_enabled is False

    def test_llm_request_dump_gate_accepts_truthy_spellings(self, tmp_path, monkeypatch):
        path = tmp_path / "debug.jsonl"
        logger = DebugJSONLLogger(path)
        for value in ("1", "true", "yes", "on", "TRUE", "On"):
            monkeypatch.setenv("MIST_DEBUG_LLM_REQUESTS", value)
            assert logger.llm_request_dump_enabled is True, f"value={value!r}"

        for value in ("0", "false", "no", "off", ""):
            monkeypatch.setenv("MIST_DEBUG_LLM_REQUESTS", value)
            assert logger.llm_request_dump_enabled is False, f"value={value!r}"


# ---------------------------------------------------------------------------
# Cluster 5 — phase: "llm_call" record
# ---------------------------------------------------------------------------


@dataclass
class FakeLLMRequest:
    messages: list = None
    tools: list = None
    temperature: float = 0.7
    max_tokens: int = 400
    top_p: float = 0.9
    json_mode: bool = False

    def __post_init__(self):
        if self.messages is None:
            self.messages = [{"role": "user", "content": "hi"}]


class TestLLMCallRecord:
    def test_record_llm_call_noop_when_gate_closed(self, tmp_path, monkeypatch):
        path = tmp_path / "debug.jsonl"
        monkeypatch.delenv("MIST_DEBUG_LLM_JSONL", raising=False)
        logger = DebugJSONLLogger(path)

        logger.record_llm_call(
            request=FakeLLMRequest(),
            response=FakeLLMResponse(),
            latency_ms=12.3,
        )
        # File may exist (path.parent.mkdir) but must contain no JSONL lines.
        if path.exists():
            assert _read_jsonl(path) == []

    def test_record_llm_call_emits_one_line_when_gate_open(self, tmp_path, monkeypatch):
        path = tmp_path / "debug.jsonl"
        monkeypatch.setenv("MIST_DEBUG_LLM_JSONL", "1")
        logger = DebugJSONLLogger(path)

        logger.record_llm_call(
            request=FakeLLMRequest(
                messages=[{"role": "user", "content": "hello"}],
                temperature=0.5,
                max_tokens=256,
            ),
            response=FakeLLMResponse(content="hi there"),
            latency_ms=42.1,
            event_id="evt-1",
            session_id="sess-1",
            call_site="chat.initial",
            pass_num=1,
        )

        lines = _read_jsonl(path)
        assert len(lines) == 1
        line = lines[0]
        assert line["phase"] == "llm_call"
        assert line["event_id"] == "evt-1"
        assert line["session_id"] == "sess-1"
        assert line["call_site"] == "chat.initial"
        assert line["pass_num"] == 1
        assert line["latency_ms"] == 42.1
        assert line["request"]["temperature"] == 0.5
        assert line["request"]["max_tokens"] == 256
        assert line["request"]["messages"][0]["content"] == "hello"
        assert line["response"]["content"] == "hi there"
        assert line["response"]["partial"] is False

    def test_record_llm_call_captures_full_content_not_length(self, tmp_path, monkeypatch):
        path = tmp_path / "debug.jsonl"
        monkeypatch.setenv("MIST_DEBUG_LLM_JSONL", "1")
        logger = DebugJSONLLogger(path)
        long_content = "A" * 5000

        logger.record_llm_call(
            request=FakeLLMRequest(),
            response=FakeLLMResponse(content=long_content),
            latency_ms=1.0,
        )

        line = _read_jsonl(path)[0]
        assert line["response"]["content"] == long_content

    def test_record_llm_call_serializes_tool_calls_and_usage(self, tmp_path, monkeypatch):
        path = tmp_path / "debug.jsonl"
        monkeypatch.setenv("MIST_DEBUG_LLM_JSONL", "1")
        logger = DebugJSONLLogger(path)

        response = FakeLLMResponse(
            content="calling tool",
            tool_calls=[FakeToolCall(name="search_kg", id="tc-1")],
            usage=FakeUsage(input_tokens=250, output_tokens=40),
        )
        logger.record_llm_call(
            request=FakeLLMRequest(),
            response=response,
            latency_ms=99.0,
        )

        line = _read_jsonl(path)[0]
        assert len(line["response"]["tool_calls"]) == 1
        tc = line["response"]["tool_calls"][0]
        assert tc["name"] == "search_kg"
        assert tc["id"] == "tc-1"
        assert tc["arguments"] == {"query": "test"}

    def test_record_llm_call_optional_fields_default_to_none(self, tmp_path, monkeypatch):
        path = tmp_path / "debug.jsonl"
        monkeypatch.setenv("MIST_DEBUG_LLM_JSONL", "1")
        logger = DebugJSONLLogger(path)

        logger.record_llm_call(
            request=FakeLLMRequest(),
            response=FakeLLMResponse(),
            latency_ms=1.0,
        )
        line = _read_jsonl(path)[0]
        assert line["event_id"] is None
        assert line["session_id"] is None
        assert line["call_site"] is None
        assert line["pass_num"] is None


# ---------------------------------------------------------------------------
# Cluster 5 — phase: "retrieval_candidates" record
# ---------------------------------------------------------------------------


class TestRetrievalCandidatesRecord:
    def test_noop_when_gate_closed(self, tmp_path, monkeypatch):
        path = tmp_path / "debug.jsonl"
        monkeypatch.delenv("MIST_DEBUG_RETRIEVAL_JSONL", raising=False)
        logger = DebugJSONLLogger(path)

        logger.record_retrieval_candidates(
            query="anything",
            intent="hybrid",
            graph_candidates=[{"entity_id": "x", "similarity": 0.9}],
        )
        if path.exists():
            assert _read_jsonl(path) == []

    def test_emits_full_pools_with_counts(self, tmp_path, monkeypatch):
        path = tmp_path / "debug.jsonl"
        monkeypatch.setenv("MIST_DEBUG_RETRIEVAL_JSONL", "1")
        logger = DebugJSONLLogger(path)

        graph = [
            {"entity_id": "python", "similarity": 0.92},
            {"entity_id": "rust", "similarity": 0.71},
        ]
        vector = [
            {"chunk_id": "c1", "source_id": "doc-a", "similarity": 0.88},
        ]
        logger.record_retrieval_candidates(
            query="what do I know about python",
            intent="hybrid",
            graph_candidates=graph,
            vector_candidates=vector,
            merged_count=3,
            final_count=3,
            event_id="evt-2",
            session_id="sess-2",
        )

        line = _read_jsonl(path)[0]
        assert line["phase"] == "retrieval_candidates"
        assert line["query"] == "what do I know about python"
        assert line["intent"] == "hybrid"
        assert line["graph_candidate_count"] == 2
        assert line["vector_candidate_count"] == 1
        assert line["merged_count"] == 3
        assert line["final_count"] == 3
        assert line["graph_candidates"][0]["entity_id"] == "python"
        assert line["vector_candidates"][0]["chunk_id"] == "c1"
        assert line["event_id"] == "evt-2"
        assert line["session_id"] == "sess-2"

    def test_empty_pools_still_emit_line(self, tmp_path, monkeypatch):
        path = tmp_path / "debug.jsonl"
        monkeypatch.setenv("MIST_DEBUG_RETRIEVAL_JSONL", "1")
        logger = DebugJSONLLogger(path)

        logger.record_retrieval_candidates(
            query="rare query",
            intent="relational",
            graph_candidates=None,
            vector_candidates=None,
            final_count=0,
        )
        line = _read_jsonl(path)[0]
        assert line["graph_candidates"] == []
        assert line["vector_candidates"] == []
        assert line["graph_candidate_count"] == 0
        assert line["vector_candidate_count"] == 0
        assert line["final_count"] == 0


# ---------------------------------------------------------------------------
# Cluster 5 — phase: "llm_request_raw" record
# ---------------------------------------------------------------------------


class TestLLMRequestDumpRecord:
    def test_noop_when_gate_closed(self, tmp_path, monkeypatch):
        path = tmp_path / "debug.jsonl"
        monkeypatch.delenv("MIST_DEBUG_LLM_REQUESTS", raising=False)
        logger = DebugJSONLLogger(path)

        logger.record_llm_request_dump(
            request_dict={"messages": []},
            error_message="nope",
        )
        if path.exists():
            assert _read_jsonl(path) == []

    def test_emits_full_kwargs_and_error(self, tmp_path, monkeypatch):
        path = tmp_path / "debug.jsonl"
        monkeypatch.setenv("MIST_DEBUG_LLM_REQUESTS", "1")
        logger = DebugJSONLLogger(path)

        request_dict = {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": None,
            "temperature": 0.7,
            "max_tokens": 400,
        }
        logger.record_llm_request_dump(
            request_dict=request_dict,
            error_message="ValidationError: tool_calls wrong shape",
            call_site="chat.initial",
            session_id="sess-3",
        )

        line = _read_jsonl(path)[0]
        assert line["phase"] == "llm_request_raw"
        assert line["call_site"] == "chat.initial"
        assert line["session_id"] == "sess-3"
        assert line["error"].startswith("ValidationError")
        assert line["request_dict"]["temperature"] == 0.7
        assert line["request_dict"]["messages"][0]["content"] == "hi"

    def test_non_serializable_values_are_stringified(self, tmp_path, monkeypatch):
        """The underlying emit uses json.dumps(default=str); non-trivial objects
        should stringify rather than raising, so a Bug C-class dump is never lost.
        """
        path = tmp_path / "debug.jsonl"
        monkeypatch.setenv("MIST_DEBUG_LLM_REQUESTS", "1")
        logger = DebugJSONLLogger(path)

        class Opaque:
            def __repr__(self) -> str:
                return "<Opaque>"

        logger.record_llm_request_dump(
            request_dict={"weird": Opaque(), "messages": []},
            error_message="boom",
        )
        line = _read_jsonl(path)[0]
        assert "<Opaque>" in line["request_dict"]["weird"]
