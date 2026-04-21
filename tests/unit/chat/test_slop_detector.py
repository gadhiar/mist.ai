"""Unit tests for backend.chat.slop_detector."""

from backend.chat.slop_detector import (
    PATTERNS,
    SlopDetector,
    SlopFinding,
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


class TestSlopFindingReplacementContract:
    """Each SlopFinding exposes a replacement reflecting fixability."""

    def test_arrow_finding_replacement_is_ascii_arrow(self):
        detector = SlopDetector()
        findings = detector.detect("User -> agent → response", severity_floor="critical")
        arrow = next(f for f in findings if f.pattern_name == "arrow_symbols")
        assert isinstance(arrow, SlopFinding)
        assert arrow.replacement == "->"

    def test_superlative_finding_replacement_is_none(self):
        detector = SlopDetector()
        findings = detector.detect("That is amazing.", severity_floor="warning")
        superlative = next(f for f in findings if f.pattern_name == "superlatives")
        assert isinstance(superlative, SlopFinding)
        assert superlative.replacement is None

    def test_emoji_finding_replacement_is_empty_string(self):
        detector = SlopDetector()
        findings = detector.detect("Ship it 🚀", severity_floor="critical")
        emoji = next(f for f in findings if f.pattern_name == "emoji")
        assert isinstance(emoji, SlopFinding)
        assert emoji.replacement == ""


class TestAdditionalPatternCoverage:
    """Coverage for patterns not exercised by the core suite."""

    def test_detects_exclamation_spam_and_strips_to_single_bang(self):
        detector = SlopDetector()
        findings = detector.detect("That's huge!!!", severity_floor="info")
        spam = next(f for f in findings if f.pattern_name == "exclamation_spam")
        assert spam.severity == "info"
        assert spam.replacement == "!"
        # Two bangs are allowed (pattern is !{3,}), three or more are not
        out = detector.strip_fixable("That's huge!!!")
        assert "!!!" not in out
        assert out.endswith("!")
        # Borderline: exactly two should be preserved
        assert detector.strip_fixable("Wait!!") == "Wait!!"

    def test_detects_hype_words_at_warning_floor(self):
        detector = SlopDetector()
        text = "A seamless, cutting-edge, enterprise-grade platform."
        findings = detector.detect(text, severity_floor="warning")
        hype = [f for f in findings if f.pattern_name == "hype_words"]
        assert len(hype) >= 3
        matched = {f.matched_text.lower() for f in hype}
        assert "seamless" in matched
        assert "cutting-edge" in matched
        assert "enterprise-grade" in matched
        # Hype words are not fixable, so replacement is None
        for f in hype:
            assert f.severity == "warning"
            assert f.replacement is None
        # strip_fixable must leave them intact
        assert detector.strip_fixable(text) == text


class TestEdgeCases:
    """Boundary inputs: empty strings and mixed-severity text."""

    def test_empty_string_detect_returns_no_findings(self):
        detector = SlopDetector()
        assert detector.detect("", severity_floor="info") == []

    def test_empty_string_strip_returns_empty_string(self):
        detector = SlopDetector()
        assert detector.detect("", severity_floor="info") == []
        assert detector.strip_fixable("") == ""

    def test_mixed_severity_text_yields_multiple_pattern_names(self):
        detector = SlopDetector()
        # emoji (critical) + amazing (warning) + first and foremost (info)
        text = "First and foremost, 🎉 that is an amazing outcome."
        findings = detector.detect(text, severity_floor="info")
        pattern_names = {f.pattern_name for f in findings}
        assert "superlatives" in pattern_names
        assert "filler_phrases" in pattern_names
        assert pattern_names & {"emoji", "emoji_symbols"}
        # Must cover at least 3 distinct categories across severity bands
        assert len(pattern_names) >= 3
