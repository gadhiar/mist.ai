"""Tests for SignalDetector -- internal knowledge signal pre-check."""


class TestFeedbackSignals:
    def test_detects_positive_feedback(self):
        from backend.knowledge.extraction.signal_detector import SignalDetector

        detector = SignalDetector()
        result = detector.detect("I like when you explain things step by step")
        assert result.has_signals
        assert "feedback" in result.signal_types

    def test_detects_negative_feedback(self):
        from backend.knowledge.extraction.signal_detector import SignalDetector

        detector = SignalDetector()
        result = detector.detect("Stop summarizing everything, just give me the answer")
        assert result.has_signals
        assert "feedback" in result.signal_types

    def test_detects_correction(self):
        from backend.knowledge.extraction.signal_detector import SignalDetector

        detector = SignalDetector()
        result = detector.detect("No that's wrong, it should be Python not Java")
        assert result.has_signals
        assert "correction" in result.signal_types


class TestPreferenceSignals:
    def test_detects_preference_expression(self):
        from backend.knowledge.extraction.signal_detector import SignalDetector

        detector = SignalDetector()
        result = detector.detect("I prefer concise technical answers")
        assert result.has_signals
        assert "preference" in result.signal_types


class TestCapabilitySignals:
    def test_detects_capability_evidence_from_tool_calls(self):
        from backend.knowledge.extraction.signal_detector import SignalDetector

        detector = SignalDetector()
        result = detector.detect(
            "good job finding that",
            tool_calls=[{"name": "query_knowledge_graph", "success": True}],
        )
        assert result.has_signals
        assert "capability" in result.signal_types


class TestNoSignals:
    def test_no_signals_for_factual_statement(self):
        from backend.knowledge.extraction.signal_detector import SignalDetector

        detector = SignalDetector()
        result = detector.detect("I use Python and React for web development")
        assert not result.has_signals

    def test_no_signals_for_question(self):
        from backend.knowledge.extraction.signal_detector import SignalDetector

        detector = SignalDetector()
        result = detector.detect("What programming languages do I know?")
        assert not result.has_signals

    def test_no_signals_for_short_message(self):
        from backend.knowledge.extraction.signal_detector import SignalDetector

        detector = SignalDetector()
        result = detector.detect("ok")
        assert not result.has_signals
