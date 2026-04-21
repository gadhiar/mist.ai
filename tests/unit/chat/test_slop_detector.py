"""Unit tests for backend.chat.slop_detector."""

from backend.chat.slop_detector import (
    PATTERNS,
    SlopDetector,
    SlopPattern,
)


class TestSlopPatternsCatalogue:
    """The PATTERNS list is the source-of-truth catalogue."""

    def test_patterns_is_populated(self):
        assert len(PATTERNS) >= 7

    def test_each_pattern_has_required_fields(self):
        for p in PATTERNS:
            assert isinstance(p, SlopPattern)
            assert p.name
            assert p.severity in ("critical", "warning", "info")
            assert isinstance(p.fixable, bool)

    def test_emoji_patterns_are_critical_and_fixable(self):
        emoji_patterns = [p for p in PATTERNS if "emoji" in p.name or p.name == "arrow_symbols"]
        assert len(emoji_patterns) >= 3
        for p in emoji_patterns:
            assert p.severity == "critical", f"{p.name} should be critical"
            assert p.fixable is True, f"{p.name} should be fixable"


class TestSlopDetectorDetect:
    """SlopDetector.detect returns findings by severity floor."""

    def test_detects_emoji_in_text(self):
        detector = SlopDetector()
        findings = detector.detect("Great work! 🎉 Ship it.", severity_floor="critical")
        assert any(f.pattern_name in ("emoji", "emoji_symbols") for f in findings)

    def test_detects_arrow_symbols(self):
        detector = SlopDetector()
        findings = detector.detect("User → graph → response.", severity_floor="critical")
        assert any(f.pattern_name == "arrow_symbols" for f in findings)

    def test_detects_superlatives_at_warning_floor(self):
        detector = SlopDetector()
        findings = detector.detect("That's an amazing approach.", severity_floor="warning")
        assert any(f.pattern_name == "superlatives" for f in findings)

    def test_superlatives_excluded_at_critical_floor(self):
        detector = SlopDetector()
        findings = detector.detect("That's an amazing approach.", severity_floor="critical")
        superlative_findings = [f for f in findings if f.pattern_name == "superlatives"]
        assert superlative_findings == []

    def test_clean_text_produces_no_findings(self):
        detector = SlopDetector()
        findings = detector.detect("Understood. Running the pipeline now.", severity_floor="info")
        assert findings == []

    def test_finding_contains_matched_text(self):
        detector = SlopDetector()
        findings = detector.detect("First and foremost, ship it.", severity_floor="info")
        filler_finding = next(f for f in findings if f.pattern_name == "filler_phrases")
        assert "first and foremost" in filler_finding.matched_text.lower()


class TestSlopDetectorStripFixable:
    """strip_fixable mechanically removes fixable-severity patterns."""

    def test_strip_removes_emojis(self):
        detector = SlopDetector()
        out = detector.strip_fixable("Great work 🎉 ship it.")
        assert "🎉" not in out
        assert "ship it" in out

    def test_strip_replaces_arrows_with_ascii(self):
        detector = SlopDetector()
        out = detector.strip_fixable("User → graph")
        assert "→" not in out
        assert "->" in out

    def test_strip_preserves_non_slop_text(self):
        detector = SlopDetector()
        out = detector.strip_fixable("Plain response with no issues.")
        assert out == "Plain response with no issues."

    def test_strip_preserves_superlatives(self):
        detector = SlopDetector()
        out = detector.strip_fixable("That is an amazing result.")
        assert "amazing" in out
