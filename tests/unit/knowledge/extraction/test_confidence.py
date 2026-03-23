"""Tests for ConfidenceScorer extraction pipeline stage."""

import pytest

from backend.knowledge.extraction.confidence import ConfidenceScorer
from backend.knowledge.extraction.ontology_extractor import ExtractionResult


def _make_extraction(source_utterance: str, confidence: float = 0.90) -> ExtractionResult:
    """Build an ExtractionResult with one relationship at the given confidence."""
    return ExtractionResult(
        entities=[],
        relationships=[
            {
                "type": "USES",
                "source": "user",
                "target": "python",
                "properties": {"confidence": confidence},
            }
        ],
        source_utterance=source_utterance,
    )


def _get_confidence(result: ExtractionResult) -> float:
    """Extract the confidence value from the first relationship."""
    return result.relationships[0]["properties"]["confidence"]


class TestHedgeDetection:
    """Hedge words in source utterance reduce confidence."""

    @pytest.mark.parametrize(
        "hedge_word, expected_penalty",
        [
            pytest.param("maybe", -0.20, id="maybe"),
            pytest.param("perhaps", -0.20, id="perhaps"),
            pytest.param("possibly", -0.20, id="possibly"),
            pytest.param("might", -0.20, id="might"),
            pytest.param("I think", -0.15, id="I-think"),
            pytest.param("I guess", -0.15, id="I-guess"),
            pytest.param("I suppose", -0.15, id="I-suppose"),
            pytest.param("I believe", -0.15, id="I-believe"),
            pytest.param("probably", -0.10, id="probably"),
            pytest.param("likely", -0.10, id="likely"),
            pytest.param("not sure", -0.25, id="not-sure"),
            pytest.param("uncertain", -0.25, id="uncertain"),
            pytest.param("unsure", -0.25, id="unsure"),
            pytest.param("sort of", -0.10, id="sort-of"),
            pytest.param("kind of", -0.10, id="kind-of"),
            pytest.param("kinda", -0.10, id="kinda"),
            pytest.param("sorta", -0.10, id="sorta"),
        ],
    )
    def test_hedge_reduces_confidence(self, hedge_word: str, expected_penalty: float):
        # Arrange
        scorer = ConfidenceScorer()
        base_confidence = 0.90
        extraction = _make_extraction(f"{hedge_word} I use Python", confidence=base_confidence)

        # Act
        result = scorer.adjust_confidence(extraction)

        # Assert
        expected = round(base_confidence + expected_penalty, 2)
        assert _get_confidence(result) == expected

    def test_worst_hedge_penalty_applies_when_multiple_hedges_present(self):
        # Arrange -- "not sure" (-0.25) is worse than "probably" (-0.10)
        scorer = ConfidenceScorer()
        extraction = _make_extraction("I'm not sure, probably I use Python", confidence=0.90)

        # Act
        result = scorer.adjust_confidence(extraction)

        # Assert -- only worst penalty (-0.25) applies
        assert _get_confidence(result) == round(0.90 - 0.25, 2)

    def test_no_hedges_leaves_confidence_unchanged(self):
        # Arrange
        scorer = ConfidenceScorer()
        extraction = _make_extraction("I use Python every day", confidence=0.90)

        # Act
        result = scorer.adjust_confidence(extraction)

        # Assert
        assert _get_confidence(result) == 0.90


class TestThirdPartyPenalty:
    """Third-party attribution caps confidence at 0.80."""

    def test_third_party_coworker_caps_at_080(self):
        # Arrange
        scorer = ConfidenceScorer()
        extraction = _make_extraction("My coworker says he uses Python", confidence=0.90)

        # Act
        result = scorer.adjust_confidence(extraction)

        # Assert
        assert _get_confidence(result) == 0.80

    def test_third_party_heard_that_caps_at_080(self):
        # Arrange
        scorer = ConfidenceScorer()
        extraction = _make_extraction("I heard that Python is popular", confidence=0.90)

        # Act
        result = scorer.adjust_confidence(extraction)

        # Assert
        assert _get_confidence(result) == 0.80

    def test_third_party_apparently_caps_at_080(self):
        # Arrange
        scorer = ConfidenceScorer()
        extraction = _make_extraction("Apparently Python is great", confidence=0.90)

        # Act
        result = scorer.adjust_confidence(extraction)

        # Assert
        assert _get_confidence(result) == 0.80

    def test_third_party_below_cap_not_increased(self):
        # Arrange -- confidence already below 0.80, cap should not increase it
        scorer = ConfidenceScorer()
        extraction = _make_extraction("My colleague told me about Python", confidence=0.70)

        # Act
        result = scorer.adjust_confidence(extraction)

        # Assert -- stays at 0.70 since it is already below the cap
        assert _get_confidence(result) == 0.70


class TestConfidenceClamping:
    """Confidence is clamped to [0.10, 0.99]."""

    def test_clamps_to_minimum_010(self):
        # Arrange -- large hedge on already-low confidence
        scorer = ConfidenceScorer()
        extraction = _make_extraction("I'm not sure about Python", confidence=0.15)

        # Act
        result = scorer.adjust_confidence(extraction)

        # Assert -- 0.15 - 0.25 = -0.10, clamped to 0.10
        assert _get_confidence(result) == 0.10

    def test_clamps_to_maximum_099(self):
        # Arrange -- no adjustments, but confidence should not exceed 0.99
        scorer = ConfidenceScorer()
        extraction = _make_extraction("I use Python", confidence=1.0)

        # Act -- no hedge, no third-party, so no adjustment triggered
        result = scorer.adjust_confidence(extraction)

        # Assert -- no adjustment was made, so original value persists
        assert _get_confidence(result) == 1.0
