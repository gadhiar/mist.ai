"""Tests for StreamEvent types used by handle_message_streaming.

The stream API yields a sequence of StreamEvent subclasses. Token and
Complete are the load-bearing events emitted by v1; Thinking and Filler
are extensibility hooks reserved for future iterations (model reasoning
display + bridge text during retrieval/tool latency).
"""

import dataclasses

import pytest

from backend.chat.stream_events import (
    Complete,
    Filler,
    StreamEvent,
    Thinking,
    Token,
)


class TestTokenEvent:
    def test_holds_text(self):
        event = Token(text="hello")
        assert event.text == "hello"

    def test_default_pass_num_is_1(self):
        event = Token(text="hi")
        assert event.pass_num == 1

    def test_pass_num_can_be_2_for_post_tool_response(self):
        event = Token(text="hi", pass_num=2)
        assert event.pass_num == 2

    def test_is_streamevent(self):
        assert isinstance(Token(text="x"), StreamEvent)

    def test_is_frozen(self):
        event = Token(text="x")
        with pytest.raises(dataclasses.FrozenInstanceError):
            event.text = "y"  # type: ignore[misc]

    def test_uses_slots(self):
        event = Token(text="x")
        # Adding a non-slot attribute must fail. Python 3.11 raises TypeError
        # via the auto-generated dataclass __setattr__'s super() chain when
        # frozen+slots+inheritance combine; later versions raise AttributeError.
        # Either signals the slots discipline is enforced.
        with pytest.raises((AttributeError, TypeError)):
            event.unexpected = "value"  # type: ignore[attr-defined]
        assert not hasattr(event, "__dict__"), "slots should suppress __dict__"


class TestThinkingEvent:
    def test_holds_text(self):
        event = Thinking(text="reasoning step")
        assert event.text == "reasoning step"

    def test_is_streamevent(self):
        assert isinstance(Thinking(text="x"), StreamEvent)

    def test_is_frozen(self):
        event = Thinking(text="x")
        with pytest.raises(dataclasses.FrozenInstanceError):
            event.text = "y"  # type: ignore[misc]


class TestFillerEvent:
    def test_holds_text(self):
        event = Filler(text="One moment...")
        assert event.text == "One moment..."

    def test_is_streamevent(self):
        assert isinstance(Filler(text="x"), StreamEvent)

    def test_is_frozen(self):
        event = Filler(text="x")
        with pytest.raises(dataclasses.FrozenInstanceError):
            event.text = "y"  # type: ignore[misc]


class TestCompleteEvent:
    def test_holds_final_response(self):
        event = Complete(final_response="full text")
        assert event.final_response == "full text"

    def test_default_tool_calls_used_is_0(self):
        event = Complete(final_response="x")
        assert event.tool_calls_used == 0

    def test_default_duration_ms_is_0(self):
        event = Complete(final_response="x")
        assert event.duration_ms == 0.0

    def test_carries_metrics(self):
        event = Complete(final_response="x", tool_calls_used=2, duration_ms=12345.6)
        assert event.tool_calls_used == 2
        assert event.duration_ms == pytest.approx(12345.6)

    def test_is_streamevent(self):
        assert isinstance(Complete(final_response="x"), StreamEvent)

    def test_is_frozen(self):
        event = Complete(final_response="x")
        with pytest.raises(dataclasses.FrozenInstanceError):
            event.final_response = "y"  # type: ignore[misc]


class TestPatternMatching:
    """Validate Python 3.10+ structural pattern matching for type narrowing."""

    def test_match_dispatches_correctly(self):
        events: list[StreamEvent] = [
            Token(text="a"),
            Token(text="b"),
            Thinking(text="reason"),
            Filler(text="bridge"),
            Complete(final_response="ab"),
        ]
        token_count = 0
        thinking_count = 0
        filler_count = 0
        complete_response = None
        for evt in events:
            match evt:
                case Token():
                    token_count += 1
                case Thinking():
                    thinking_count += 1
                case Filler():
                    filler_count += 1
                case Complete(final_response=resp):
                    complete_response = resp
        assert token_count == 2
        assert thinking_count == 1
        assert filler_count == 1
        assert complete_response == "ab"
