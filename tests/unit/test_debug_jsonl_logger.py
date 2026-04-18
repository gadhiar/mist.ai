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
