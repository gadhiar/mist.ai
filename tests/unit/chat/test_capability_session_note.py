"""The live session note leaves capability-only turns out (T31).

`end_session` synthesizes the note from the session's logged turns. A turn a
capability answered alone is a command and a template, so only the turns
`policy_skip_reason` leaves eligible reach `synthesize`. The note path is still
the one allocated at the session's first turn, capability turn or not, because
session catch-up derives the same path from the first logged turn.

These tests drive whole turns through `handle_message_streaming` with a real
in-memory `EventStore`, a fake capability port, the real `SessionSynthesizer`
over a recording LLM, and a recording vault writer.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from backend.capabilities.contracts import ActionRecord, PreTurnResolution, ToolOutcome
from backend.chat.conversation_handler import ConversationHandler
from backend.chat.session_synthesizer import SessionSynthesis, SessionSynthesizer
from backend.knowledge.retrieval.knowledge_retriever import KnowledgeRetriever
from backend.knowledge.storage.graph_store import GraphStore
from tests.mocks.config import build_test_config
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection
from tests.mocks.streaming_llm import FakeStreamingLLMProvider, ScriptedPass
from tests.unit.conftest import make_test_conventions_loader

_COMMANDS = {
    "set a timer for ten minutes": "Timer set.",
    "stop the alarm": "Alarm stopped.",
}


def _resolution(reply: str) -> PreTurnResolution:
    record = ActionRecord(
        tool_call_id="call-1",
        tool="timers.start",
        capability="timers",
        risk="reversible",
        args_summary="minutes=10",
        outcome=ToolOutcome(status="ok", summary=reply, data=None, reason=None),
        duration_ms=12,
        power_level="full",
        ack="On it.",
        view="timers",
        source="fastpath",
    )
    return PreTurnResolution(
        handled=True, reply=reply, remainder=None, records=(record,), events=()
    )


class _CommandPort:
    """A `CapabilityTurnPort` that answers `_COMMANDS` and leaves the rest to the model."""

    async def pre_llm(
        self, session_id: str, turn_id: str, utterance: str, *, received_at: float | None = None
    ) -> PreTurnResolution | None:
        reply = _COMMANDS.get(utterance)
        return None if reply is None else _resolution(reply)

    def plan_turn(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("no tiers are attached, so the turn is not routed")

    async def run_actions(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("no tiers are attached, so no action decode runs")

    async def decide(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("no proposal is decided here")

    def static_sentence(self) -> str:
        return "Actions run only through the action system."

    def redact(self, text: str) -> str:
        return text

    def record_turn(self, session_id: str, used: Any) -> None:
        return None

    def drain_events(self, session_id: str) -> list[dict[str, object]]:
        return []


class _RecordingSynthesisLLM:
    """The synthesizer's LLM: records each synthesis request and answers a note."""

    def __init__(self) -> None:
        self.requests: list[Any] = []

    async def invoke(self, request: Any) -> SimpleNamespace:
        self.requests.append(request)
        return SimpleNamespace(content="TITLE: Vault talk\n\n### What Was Accomplished\n- Talked\n")

    async def health_check(self) -> bool:
        return True


class _RecordingSynthesizer:
    """Records the turns `end_session` passes, then runs the real synthesizer."""

    def __init__(self, inner: SessionSynthesizer) -> None:
        self._inner = inner
        self.calls: list[list[dict]] = []

    async def synthesize(self, turns: list[dict]) -> SessionSynthesis | None:
        self.calls.append(list(turns))
        return await self._inner.synthesize(turns)


class _RecordingVaultWriter:
    def __init__(self) -> None:
        self.writes: list[dict[str, Any]] = []

    def session_path(self, date: str, slug: str) -> str:
        return f"/tmp/vault/sessions/{date}-{slug}.md"

    async def write_session_note(
        self,
        vault_note_path: str,
        synthesis: SessionSynthesis | None,
        related_entities: list[str] | None = None,
        status: str = "completed",
    ) -> str:
        self.writes.append({"path": vault_note_path, "synthesis": synthesis, "status": status})
        return vault_note_path


class _FakeExtractionPipeline:
    async def extract_from_utterance(self, **_kwargs: Any) -> Any:
        from backend.knowledge.extraction.validator import ValidationResult

        return ValidationResult(valid=True, entities=[], relationships=[])


def _make_handler(
    passes: list[ScriptedPass],
) -> tuple[
    ConversationHandler, _RecordingSynthesizer, _RecordingSynthesisLLM, _RecordingVaultWriter
]:
    config = build_test_config(event_store_enabled=True, event_store_db_path=":memory:")
    graph_store = GraphStore(FakeNeo4jConnection(), FakeEmbeddingGenerator())
    writer = _RecordingVaultWriter()
    handler = ConversationHandler(
        config=config,
        graph_store=graph_store,
        extraction_pipeline=_FakeExtractionPipeline(),
        retriever=KnowledgeRetriever(config=config, graph_store=graph_store),
        llm_provider=FakeStreamingLLMProvider(passes),
        conventions_loader=make_test_conventions_loader(),
        vault_writer=writer,
    )
    assert handler.event_store is not None
    llm = _RecordingSynthesisLLM()
    synthesizer = _RecordingSynthesizer(
        SessionSynthesizer(llm, temperature=0.3, max_tokens=256)  # type: ignore[arg-type]
    )
    handler.session_synthesizer = synthesizer  # type: ignore[assignment]
    handler.attach_capability_host(_CommandPort())  # type: ignore[arg-type]
    return handler, synthesizer, llm, writer


async def _turn(handler: ConversationHandler, message: str, session_id: str = "s1") -> None:
    async for _event in handler.handle_message_streaming(
        user_message=message, session_id=session_id
    ):
        pass


class TestLiveSessionNote:
    @pytest.mark.asyncio
    async def test_a_session_of_capability_turns_only_makes_no_synthesis_call(self):
        handler, synthesizer, llm, writer = _make_handler([])
        await _turn(handler, "set a timer for ten minutes")
        await _turn(handler, "stop the alarm")
        assert [t["answered_by"] for t in handler.event_store.get_turns("s1")] == [
            "capability",
            "capability",
        ]

        await handler.end_session("s1")

        assert synthesizer.calls == [[]]
        assert llm.requests == []
        assert writer.writes == []

    @pytest.mark.asyncio
    async def test_a_mixed_session_synthesizes_only_eligible_turns_at_the_first_turns_path(self):
        handler, synthesizer, llm, writer = _make_handler(
            [ScriptedPass(chunks=["It has three layers."])]
        )
        await _turn(handler, "set a timer for ten minutes")
        path = handler._vault_paths["s1"]
        await _turn(handler, "tell me about the vault architecture")
        await _turn(handler, "stop the alarm")

        await handler.end_session("s1")

        assert [[t["user_utterance"] for t in call] for call in synthesizer.calls] == [
            ["tell me about the vault architecture"]
        ]
        assert len(llm.requests) == 1
        prompt = llm.requests[0].messages[0]["content"]
        assert "tell me about the vault architecture" in prompt
        assert "set a timer for ten minutes" not in prompt
        assert [w["path"] for w in writer.writes] == [path]

    @pytest.mark.asyncio
    async def test_old_turns_without_the_field_are_still_synthesized(self):
        handler, synthesizer, llm, writer = _make_handler([])
        handler._get_or_allocate_vault_path("s-old", first_utterance="what did we decide")
        handler.event_store.start_session("s-old", input_modality="text")
        conn = handler.event_store._get_connection()
        conn.execute(
            "INSERT INTO conversation_turn_events (event_id, session_id, turn_index, timestamp, "
            "user_utterance, system_response) VALUES (?, ?, ?, ?, ?, ?)",
            ("old-1", "s-old", 0, "2026-07-01T10:00:00+00:00", "what did we decide", "Rust."),
        )

        await handler.end_session("s-old")

        assert [[t["event_id"] for t in call] for call in synthesizer.calls] == [["old-1"]]
        assert len(llm.requests) == 1
        assert len(writer.writes) == 1
