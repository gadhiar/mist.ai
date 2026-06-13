"""Unit tests for scripts/mist_admin.py extraction-only Tier 3 replay mode.

The extraction-only path (`run_extraction_only`) drives MIST's production
extraction pipeline for a single utterance WITHOUT generating a conversational
reply. It exists to make the F2 extraction baseline reproducible: the chat
reply is conversational noise the gold corpus does not encode and is the sole
source of F2 nondeterminism (flash-attn FP noise on long greedy generations).
The extraction calls themselves are deterministic at temperature 0.

Covered:
- `run_extraction_only` invokes the handler's extraction path directly
  (`_extract_knowledge_async`) and NEVER calls `handle_message` (no chat reply)
- It pre-allocates the vault path (Step 0) and records the turn event so the
  same scaffolding production extraction needs (event_id, vault_note_path,
  recorded_at) is present
- It forwards an empty conversation_history and empty assistant_message so no
  same-turn assistant reply enters the extraction context
- It drains the in-flight background extraction task before returning (without
  the drain, `asyncio.run` would cancel the fire-and-forget task)
- It returns a structured record with timing + ok flag
- It captures handler exceptions as `ok=False` without propagating
- `run_extraction_only_replay` iterates inputs preserving per-entry metadata

These behaviors guarantee the emitted debug-jsonl records (the `extraction.*`
`llm_call` records) are produced by the SAME production extraction code the
F2 scorer already consumes, so `score_extraction_run.py` works unchanged.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Make `scripts` importable without installing the repo as a package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from mist_admin import run_extraction_only, run_extraction_only_replay  # noqa: E402

# ---------------------------------------------------------------------------
# Test double
# ---------------------------------------------------------------------------


class FakeExtractionHandler:
    """Duck-typed ConversationHandler exposing only the extraction-only seams.

    Mirrors the real handler's extraction scaffolding methods:
    `_get_or_allocate_vault_path`, `_record_turn_event`, `_extract_knowledge_async`,
    and `_drain_extraction_tasks`. Records every call for assertion and, like
    the production handler, spawns the extraction work as a real background task
    so the runner's drain logic is exercised end to end.

    Crucially does NOT implement `handle_message`: any attempt by the runner to
    generate a chat reply raises AttributeError and fails the test loudly.
    """

    def __init__(self, *, extract_error: Exception | None = None) -> None:
        self.extract_error = extract_error
        self.vault_path_calls: list[dict] = []
        self.record_event_calls: list[dict] = []
        self.extract_calls: list[dict] = []
        self.drain_calls: list[dict] = []
        # Mirrors ConversationHandler._extraction_tasks (task -> session_id).
        self._extraction_tasks: dict[asyncio.Task, str] = {}
        self._extraction_completed: list[str] = []

    def _get_or_allocate_vault_path(
        self, session_id: str, first_utterance: str | None = None
    ) -> str | None:
        self.vault_path_calls.append({"session_id": session_id, "first_utterance": first_utterance})
        return f"/vault/{session_id}.md"

    def _record_turn_event(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        context_window=None,
        retrieval_result=None,
        tool_calls=None,
    ) -> tuple[str | None, str | None]:
        self.record_event_calls.append(
            {
                "session_id": session_id,
                "user_message": user_message,
                "assistant_message": assistant_message,
            }
        )
        return f"evt-{session_id}", "2026-06-13T00:00:00+00:00"

    async def _extract_knowledge_async(
        self,
        utterance: str,
        conversation_history,
        event_id: str,
        session_id: str,
        assistant_message: str = "",
        turn_record=None,
        vault_note_path: str | None = None,
        recorded_at: str | None = None,
    ) -> None:
        self.extract_calls.append(
            {
                "utterance": utterance,
                "conversation_history": conversation_history,
                "event_id": event_id,
                "session_id": session_id,
                "assistant_message": assistant_message,
                "vault_note_path": vault_note_path,
                "recorded_at": recorded_at,
            }
        )
        if self.extract_error is not None:
            raise self.extract_error
        # Simulate background completion so the drain has something to await.
        await asyncio.sleep(0)
        self._extraction_completed.append(session_id)

    async def _drain_extraction_tasks(
        self, session_id: str | None = None, timeout: float = 60.0
    ) -> None:
        self.drain_calls.append({"session_id": session_id, "timeout": timeout})
        tasks = [
            t
            for t, sid in list(self._extraction_tasks.items())
            if not t.done() and (session_id is None or sid == session_id)
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


# ---------------------------------------------------------------------------
# run_extraction_only
# ---------------------------------------------------------------------------


class TestRunExtractionOnly:
    def test_invokes_extraction_path_not_chat(self):
        # Arrange
        handler = FakeExtractionHandler()

        # Act
        result = asyncio.run(run_extraction_only(handler, "I use Rust", "s1", "User"))

        # Assert: extraction ran exactly once; no chat reply was generated
        assert result["ok"] is True
        assert len(handler.extract_calls) == 1
        assert handler.extract_calls[0]["utterance"] == "I use Rust"
        assert not hasattr(handler, "handle_message")

    def test_omits_same_turn_assistant_reply_from_extraction_context(self):
        # Arrange
        handler = FakeExtractionHandler()

        # Act
        asyncio.run(run_extraction_only(handler, "I use Rust", "s1"))

        # Assert: empty history + empty assistant_message -> no reply contamination
        call = handler.extract_calls[0]
        assert call["conversation_history"] == []
        assert call["assistant_message"] == ""

    def test_pre_allocates_vault_path_with_first_utterance(self):
        # Arrange
        handler = FakeExtractionHandler()

        # Act
        asyncio.run(run_extraction_only(handler, "I use Rust", "s1"))

        # Assert: Step 0 vault path pre-allocation mirrors handle_message
        assert len(handler.vault_path_calls) == 1
        assert handler.vault_path_calls[0]["session_id"] == "s1"
        assert handler.vault_path_calls[0]["first_utterance"] == "I use Rust"
        # The allocated path is threaded into extraction for DERIVED_FROM.
        assert handler.extract_calls[0]["vault_note_path"] == "/vault/s1.md"

    def test_records_turn_event_with_empty_assistant_message(self):
        # Arrange
        handler = FakeExtractionHandler()

        # Act
        asyncio.run(run_extraction_only(handler, "I use Rust", "s1"))

        # Assert: the event store write happens (event_id needed for provenance),
        # but with an EMPTY assistant_message -- no chat reply was produced.
        assert len(handler.record_event_calls) == 1
        assert handler.record_event_calls[0]["assistant_message"] == ""
        assert handler.extract_calls[0]["event_id"] == "evt-s1"
        assert handler.extract_calls[0]["recorded_at"] == "2026-06-13T00:00:00+00:00"

    def test_drains_background_extraction_task(self):
        # Arrange
        handler = FakeExtractionHandler()

        # Act
        result = asyncio.run(run_extraction_only(handler, "I use Rust", "s1"))

        # Assert: the fire-and-forget task was awaited to completion, not abandoned
        assert handler.drain_calls, "expected the runner to drain extraction tasks"
        assert handler._extraction_completed == ["s1"]
        assert result["extraction_duration_ms"] >= 0.0

    def test_forwards_session_and_user(self):
        # Arrange
        handler = FakeExtractionHandler()

        # Act
        result = asyncio.run(run_extraction_only(handler, "I use Rust", "sess-abc", user_id="raj"))

        # Assert
        assert result["session_id"] == "sess-abc"
        assert result["user_id"] == "raj"
        assert handler.extract_calls[0]["session_id"] == "sess-abc"

    def test_captures_exception_without_propagating(self):
        # Arrange
        handler = FakeExtractionHandler(extract_error=RuntimeError("extractor down"))

        # Act
        result = asyncio.run(run_extraction_only(handler, "I use Rust", "s1"))

        # Assert
        assert result["ok"] is False
        assert result["error"] == "RuntimeError: extractor down"


# ---------------------------------------------------------------------------
# run_extraction_only_replay
# ---------------------------------------------------------------------------


class TestRunExtractionOnlyReplay:
    def test_processes_each_input_in_order(self):
        # Arrange
        handler = FakeExtractionHandler()
        inputs = [
            {"utterance": "first"},
            {"utterance": "second"},
            {"utterance": "third"},
        ]

        # Act
        results = asyncio.run(run_extraction_only_replay(handler, inputs, "s"))

        # Assert
        assert len(results) == 3
        assert [c["utterance"] for c in handler.extract_calls] == [
            "first",
            "second",
            "third",
        ]

    def test_propagates_tag_to_results(self):
        # Arrange
        handler = FakeExtractionHandler()
        inputs = [
            {"utterance": "u1", "tag": "ext-01"},
            {"utterance": "u2"},
        ]

        # Act
        results = asyncio.run(run_extraction_only_replay(handler, inputs, "s"))

        # Assert
        assert results[0]["tag"] == "ext-01"
        assert "tag" not in results[1]

    def test_failure_on_single_input_does_not_abort_batch(self):
        # Arrange: middle input's extraction raises
        class PickyHandler(FakeExtractionHandler):
            async def _extract_knowledge_async(self, utterance, *args, **kwargs):
                if utterance == "boom":
                    raise RuntimeError("boom")
                return await super()._extract_knowledge_async(utterance, *args, **kwargs)

        handler = PickyHandler()
        inputs = [{"utterance": "ok1"}, {"utterance": "boom"}, {"utterance": "ok2"}]

        # Act
        results = asyncio.run(run_extraction_only_replay(handler, inputs, "s"))

        # Assert: batch completes; only the middle record is a failure
        assert [r["ok"] for r in results] == [True, False, True]
        assert results[1]["error"] == "RuntimeError: boom"
