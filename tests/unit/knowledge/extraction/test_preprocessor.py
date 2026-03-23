"""Tests for PreProcessor extraction pipeline stage."""

from datetime import datetime

from backend.knowledge.extraction.preprocessor import PreProcessedInput, PreProcessor


class TestContextAssembly:
    """PreProcessor assembles conversation context for extraction."""

    def test_assembles_context_from_conversation_history(self):
        # Arrange
        processor = PreProcessor()
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "Tell me about Python"},
            {"role": "assistant", "content": "Python is a language"},
        ]
        ref = datetime(2025, 6, 15)

        # Act
        result = processor.pre_process("What about Django?", history, ref)

        # Assert
        assert isinstance(result, PreProcessedInput)
        assert result.original_text == "What about Django?"
        assert result.resolved_text == "What about Django?"
        assert len(result.conversation_context) == 4
        assert result.conversation_context[0] == "[user]: Hello"
        assert result.conversation_context[1] == "[assistant]: Hi there"
        assert result.conversation_context[2] == "[user]: Tell me about Python"
        assert result.conversation_context[3] == "[assistant]: Python is a language"

    def test_truncates_context_to_max_messages(self):
        # Arrange -- 8 messages exceeds MAX_CONTEXT_MESSAGES (6)
        processor = PreProcessor()
        history = [{"role": "user", "content": f"message {i}"} for i in range(8)]
        ref = datetime(2025, 6, 15)

        # Act
        result = processor.pre_process("latest", history, ref)

        # Assert -- only last 6 messages are kept
        assert len(result.conversation_context) == 6
        assert result.conversation_context[0] == "[user]: message 2"
        assert result.conversation_context[5] == "[user]: message 7"

    def test_handles_empty_conversation_history(self):
        # Arrange
        processor = PreProcessor()
        ref = datetime(2025, 6, 15)

        # Act
        result = processor.pre_process("Hello", [], ref)

        # Assert
        assert isinstance(result, PreProcessedInput)
        assert result.original_text == "Hello"
        assert result.conversation_context == []
        assert result.turn_index == 0


class TestPreProcessedInputFields:
    """PreProcessedInput has correct field values after pre-processing."""

    def test_reference_date_is_forwarded(self):
        # Arrange
        processor = PreProcessor()
        ref = datetime(2025, 3, 22, 14, 30, 0)

        # Act
        result = processor.pre_process("test", [], ref)

        # Assert
        assert result.reference_date == ref

    def test_turn_index_is_forwarded(self):
        # Arrange
        processor = PreProcessor()
        ref = datetime(2025, 6, 15)

        # Act
        result = processor.pre_process("test", [], ref, turn_index=5)

        # Assert
        assert result.turn_index == 5

    def test_metadata_defaults_to_empty_dict(self):
        # Arrange
        processor = PreProcessor()
        ref = datetime(2025, 6, 15)

        # Act
        result = processor.pre_process("test", [], ref)

        # Assert
        assert result.metadata == {}

    def test_missing_role_defaults_to_user(self):
        # Arrange -- dict without "role" key
        processor = PreProcessor()
        history = [{"content": "no role specified"}]
        ref = datetime(2025, 6, 15)

        # Act
        result = processor.pre_process("test", history, ref)

        # Assert
        assert result.conversation_context[0] == "[user]: no role specified"
