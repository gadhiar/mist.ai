"""Unit tests for scripts/mist_admin.py Tier 3 chat + replay subcommands.

Validates the testable core of the Tier 3 CLI surface without building a real
ConversationHandler (which would require Neo4j + llama-server + embedding model).

Covered:
- `run_chat` returns a structured record with timing + response on success
- `run_chat` captures handler exceptions as `ok=False` without propagating
- `run_chat` forwards session_id + user_id to the handler verbatim
- `run_replay` iterates inputs preserving per-entry metadata (tag, expected_behavior)
- `run_replay` applies defaults when inputs omit session_id/user_id
- `run_replay` honours per-line overrides when present
- `_read_replay_inputs` parses JSONL objects, JSONL string shorthand,
   plain-text lines, and skips comments + blank lines
- `_read_replay_inputs` surfaces invalid JSON with a line number

Spec: ~/.claude/plans/nimble-forage-cinder.md Part 3 / Part 6 Task 10.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

# Make `scripts` importable without installing the repo as a package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from mist_admin import _read_replay_inputs, run_chat, run_replay  # noqa: E402

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeConversationHandler:
    """Duck-typed ConversationHandler for Tier 3 CLI tests.

    Records every call for assertion. Returns either a scripted response, an
    echo of the utterance, or raises a configured exception.
    """

    def __init__(
        self,
        *,
        responses: list[str] | None = None,
        error: Exception | None = None,
        delay_s: float = 0.0,
    ):
        self.responses = list(responses or [])
        self.error = error
        self.delay_s = delay_s
        self.calls: list[dict] = []

    async def handle_message(
        self,
        user_message: str,
        session_id: str,
        user_id: str = "User",
        max_history: int = 10,
    ) -> str:
        self.calls.append(
            {
                "user_message": user_message,
                "session_id": session_id,
                "user_id": user_id,
                "max_history": max_history,
            }
        )
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.error is not None:
            raise self.error
        if self.responses:
            return self.responses.pop(0)
        return f"echo: {user_message}"


# ---------------------------------------------------------------------------
# run_chat
# ---------------------------------------------------------------------------


class TestRunChat:
    def test_returns_ok_result_with_response_and_timing(self):
        # Arrange
        handler = FakeConversationHandler(responses=["hello back"])

        # Act
        result = asyncio.run(run_chat(handler, "hi", "s1", "User"))

        # Assert
        assert result["ok"] is True
        assert result["utterance"] == "hi"
        assert result["session_id"] == "s1"
        assert result["user_id"] == "User"
        assert result["response"] == "hello back"
        assert result["error"] is None
        assert isinstance(result["duration_ms"], float)
        assert result["duration_ms"] >= 0.0

    def test_captures_exception_without_propagating(self):
        # Arrange
        handler = FakeConversationHandler(error=RuntimeError("llm down"))

        # Act
        result = asyncio.run(run_chat(handler, "hi", "s1"))

        # Assert
        assert result["ok"] is False
        assert result["response"] is None
        assert result["error"] == "RuntimeError: llm down"
        assert result["duration_ms"] >= 0.0

    def test_forwards_session_and_user_to_handler(self):
        # Arrange
        handler = FakeConversationHandler()

        # Act
        asyncio.run(run_chat(handler, "x", "sess-abc", user_id="raj"))

        # Assert
        assert len(handler.calls) == 1
        call = handler.calls[0]
        assert call["user_message"] == "x"
        assert call["session_id"] == "sess-abc"
        assert call["user_id"] == "raj"

    def test_duration_reflects_handler_latency(self):
        # Arrange
        handler = FakeConversationHandler(delay_s=0.05)

        # Act
        result = asyncio.run(run_chat(handler, "x", "s"))

        # Assert: at least ~50ms; allow upper bound generous for CI noise
        assert result["duration_ms"] >= 40.0


# ---------------------------------------------------------------------------
# run_replay
# ---------------------------------------------------------------------------


class TestRunReplay:
    def test_processes_each_input_in_order(self):
        # Arrange
        handler = FakeConversationHandler(responses=["a", "b", "c"])
        inputs = [
            {"utterance": "first"},
            {"utterance": "second"},
            {"utterance": "third"},
        ]

        # Act
        results = asyncio.run(run_replay(handler, inputs, "s"))

        # Assert
        assert len(results) == 3
        assert [r["response"] for r in results] == ["a", "b", "c"]
        assert [c["user_message"] for c in handler.calls] == ["first", "second", "third"]

    def test_applies_default_session_and_user(self):
        # Arrange
        handler = FakeConversationHandler()
        inputs = [{"utterance": "u1"}, {"utterance": "u2"}]

        # Act
        results = asyncio.run(run_replay(handler, inputs, "default-sess", "default-user"))

        # Assert
        for r in results:
            assert r["session_id"] == "default-sess"
            assert r["user_id"] == "default-user"

    def test_per_line_session_and_user_override_defaults(self):
        # Arrange
        handler = FakeConversationHandler()
        inputs = [
            {"utterance": "u1"},
            {"utterance": "u2", "session_id": "override-sess"},
            {"utterance": "u3", "user_id": "override-user"},
        ]

        # Act
        results = asyncio.run(run_replay(handler, inputs, "default-sess", "default-user"))

        # Assert
        assert results[0]["session_id"] == "default-sess"
        assert results[0]["user_id"] == "default-user"
        assert results[1]["session_id"] == "override-sess"
        assert results[1]["user_id"] == "default-user"
        assert results[2]["session_id"] == "default-sess"
        assert results[2]["user_id"] == "override-user"

    def test_propagates_tag_and_expected_behavior_to_results(self):
        # Arrange
        handler = FakeConversationHandler()
        inputs = [
            {"utterance": "", "tag": "empty", "expected_behavior": "skip-no-write"},
            {"utterance": "   ", "tag": "whitespace"},
            {"utterance": "plain"},
        ]

        # Act
        results = asyncio.run(run_replay(handler, inputs, "s"))

        # Assert
        assert results[0]["tag"] == "empty"
        assert results[0]["expected_behavior"] == "skip-no-write"
        assert results[1]["tag"] == "whitespace"
        assert "expected_behavior" not in results[1]
        assert "tag" not in results[2]

    def test_handler_failure_on_single_input_does_not_abort_batch(self):
        # Arrange: three inputs; middle one raises
        class PickyHandler(FakeConversationHandler):
            async def handle_message(
                self, user_message, session_id, user_id="User", max_history=10
            ):
                if user_message == "boom":
                    raise RuntimeError("boom")
                return await super().handle_message(user_message, session_id, user_id, max_history)

        handler = PickyHandler()
        inputs = [{"utterance": "ok1"}, {"utterance": "boom"}, {"utterance": "ok2"}]

        # Act
        results = asyncio.run(run_replay(handler, inputs, "s"))

        # Assert: batch completes, failure is captured in the middle record only
        assert [r["ok"] for r in results] == [True, False, True]
        assert results[1]["error"] == "RuntimeError: boom"


# ---------------------------------------------------------------------------
# _read_replay_inputs
# ---------------------------------------------------------------------------


class TestReadReplayInputs:
    def test_jsonl_full_object_form(self, tmp_path):
        # Arrange
        path = tmp_path / "inputs.jsonl"
        path.write_text(
            '{"utterance": "hi", "tag": "greet"}\n' '{"utterance": "bye", "session_id": "s2"}\n',
            encoding="utf-8",
        )

        # Act
        items = _read_replay_inputs(path)

        # Assert
        assert items == [
            {"utterance": "hi", "tag": "greet"},
            {"utterance": "bye", "session_id": "s2"},
        ]

    def test_jsonl_string_shorthand_is_wrapped_to_utterance(self, tmp_path):
        # Arrange
        path = tmp_path / "inputs.jsonl"
        path.write_text('"hello"\n"world"\n', encoding="utf-8")

        # Act
        items = _read_replay_inputs(path)

        # Assert
        assert items == [{"utterance": "hello"}, {"utterance": "world"}]

    def test_jsonl_skips_blank_and_commented_lines(self, tmp_path):
        # Arrange
        path = tmp_path / "inputs.jsonl"
        path.write_text(
            "# leading comment\n" "\n" '{"utterance": "real"}\n' "   \n" "# trailing comment\n",
            encoding="utf-8",
        )

        # Act
        items = _read_replay_inputs(path)

        # Assert
        assert items == [{"utterance": "real"}]

    def test_jsonl_invalid_json_raises_with_line_number(self, tmp_path):
        # Arrange
        path = tmp_path / "inputs.jsonl"
        path.write_text(
            '{"utterance": "ok"}\n' "{not json\n",
            encoding="utf-8",
        )

        # Act / Assert
        with pytest.raises(ValueError, match="line 2"):
            _read_replay_inputs(path)

    def test_jsonl_non_dict_non_string_raises(self, tmp_path):
        # Arrange
        path = tmp_path / "inputs.jsonl"
        path.write_text("[1, 2, 3]\n", encoding="utf-8")

        # Act / Assert
        with pytest.raises(ValueError, match="expected object or string"):
            _read_replay_inputs(path)

    def test_plain_text_one_per_line(self, tmp_path):
        # Arrange
        path = tmp_path / "inputs.txt"
        path.write_text(
            "first line\n"
            "  second line with surrounding whitespace   \n"
            "# a comment\n"
            "\n"
            "third line\n",
            encoding="utf-8",
        )

        # Act
        items = _read_replay_inputs(path)

        # Assert
        assert items == [
            {"utterance": "first line"},
            {"utterance": "second line with surrounding whitespace"},
            {"utterance": "third line"},
        ]

    def test_empty_file_returns_empty_list(self, tmp_path):
        # Arrange
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")

        # Act
        items = _read_replay_inputs(path)

        # Assert
        assert items == []


# ---------------------------------------------------------------------------
# Integration between read_replay_inputs and run_replay
# ---------------------------------------------------------------------------


class TestReadAndReplayIntegration:
    def test_read_then_replay_forwards_all_fields(self, tmp_path):
        # Arrange
        path = tmp_path / "inputs.jsonl"
        payload = [
            {"utterance": "hi", "tag": "greet"},
            {"utterance": "debug this", "session_id": "dev"},
        ]
        path.write_text("\n".join(json.dumps(p) for p in payload), encoding="utf-8")
        handler = FakeConversationHandler(responses=["r1", "r2"])

        # Act
        inputs = _read_replay_inputs(path)
        results = asyncio.run(run_replay(handler, inputs, default_session_id="base"))

        # Assert
        assert results[0]["tag"] == "greet"
        assert results[0]["session_id"] == "base"
        assert results[1]["session_id"] == "dev"
        assert [c["session_id"] for c in handler.calls] == ["base", "dev"]
