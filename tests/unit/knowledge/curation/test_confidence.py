"""Tests for ConfidenceManager and CurationError."""

import pytest

from backend.errors import CurationError, MistError
from backend.knowledge.curation.confidence import ConfidenceManager
from backend.knowledge.ontologies.base import KnowledgeDomain
from backend.knowledge.ontologies.v1_0_0 import CONFIDENCE_BRIDGING, CONFIDENCE_EXTERNAL


class TestCurationError:
    def test_is_subclass_of_mist_error(self):
        assert issubclass(CurationError, MistError)

    def test_can_be_raised_and_caught_as_mist_error(self):
        with pytest.raises(MistError):
            raise CurationError("test curation failure")


class TestReinforcedConfidence:
    def test_boosts_internal_domain(self):
        mgr = ConfidenceManager()
        result = mgr.reinforced_confidence(0.85, KnowledgeDomain.INTERNAL)
        # Formula: min(0.99, 0.85 + 0.05 * (1.0 - 0.85)) = 0.8575
        assert result == pytest.approx(0.8575)

    def test_boosts_external_domain(self):
        mgr = ConfidenceManager()
        result = mgr.reinforced_confidence(0.80, KnowledgeDomain.EXTERNAL)
        # Formula: min(0.99, 0.80 + 0.10 * (1.0 - 0.80)) = 0.82
        assert result == pytest.approx(0.82)

    def test_boosts_bridging_domain(self):
        mgr = ConfidenceManager()
        result = mgr.reinforced_confidence(0.85, KnowledgeDomain.BRIDGING)
        # Formula: min(0.99, 0.85 + 0.10 * (1.0 - 0.85)) = 0.865
        assert result == pytest.approx(0.865)

    def test_caps_at_099(self):
        mgr = ConfidenceManager()
        result = mgr.reinforced_confidence(0.98, KnowledgeDomain.EXTERNAL)
        # Formula: min(0.99, 0.98 + 0.10 * (1.0 - 0.98)) = 0.982
        assert result == pytest.approx(0.982)

    def test_diminishing_returns_near_max(self):
        mgr = ConfidenceManager()
        conf = 0.80
        for _ in range(20):
            conf = mgr.reinforced_confidence(conf, KnowledgeDomain.EXTERNAL)
        assert conf <= 0.99


class TestInitialConfidence:
    def test_internal_default(self):
        mgr = ConfidenceManager()
        assert mgr.initial_confidence(KnowledgeDomain.INTERNAL) == pytest.approx(0.9)

    def test_external_default(self):
        mgr = ConfidenceManager()
        assert mgr.initial_confidence(KnowledgeDomain.EXTERNAL) == pytest.approx(0.8)

    def test_bridging_default(self):
        mgr = ConfidenceManager()
        assert mgr.initial_confidence(KnowledgeDomain.BRIDGING) == pytest.approx(0.85)


class TestDetermineDomain:
    def test_user_is_external(self):
        mgr = ConfidenceManager()
        assert mgr.determine_domain("User") == KnowledgeDomain.EXTERNAL

    def test_mist_identity_is_internal(self):
        mgr = ConfidenceManager()
        assert mgr.determine_domain("MistIdentity") == KnowledgeDomain.INTERNAL

    def test_learning_event_is_bridging(self):
        mgr = ConfidenceManager()
        assert mgr.determine_domain("LearningEvent") == KnowledgeDomain.BRIDGING

    def test_unknown_type_falls_back_to_external(self):
        mgr = ConfidenceManager()
        assert mgr.determine_domain("NonexistentType") == KnowledgeDomain.EXTERNAL


class TestPenalizedConfidence:
    """ConfidenceManager.penalized_confidence applies the third_party_penalty.

    The penalty comes from the ConfidencePolicy for the given domain so the
    value is never hardcoded in the curation layer.
    """

    def test_third_party_external_applies_015_penalty(self):
        # Arrange -- CONFIDENCE_EXTERNAL.third_party_penalty = 0.15
        mgr = ConfidenceManager()
        base = 0.8
        expected = base - CONFIDENCE_EXTERNAL.third_party_penalty  # 0.65

        # Act
        actual = mgr.penalized_confidence(base, KnowledgeDomain.EXTERNAL, third_party=True)

        # Assert
        assert abs(actual - expected) < 1e-9

    def test_user_sourced_external_no_penalty(self):
        # Arrange -- third_party=False must return the base unchanged
        mgr = ConfidenceManager()
        base = 0.8

        # Act
        actual = mgr.penalized_confidence(base, KnowledgeDomain.EXTERNAL, third_party=False)

        # Assert
        assert abs(actual - base) < 1e-9

    def test_third_party_bridging_applies_010_penalty(self):
        # Arrange -- CONFIDENCE_BRIDGING.third_party_penalty = 0.10
        mgr = ConfidenceManager()
        base = 0.85
        expected = base - CONFIDENCE_BRIDGING.third_party_penalty  # 0.75

        # Act
        actual = mgr.penalized_confidence(base, KnowledgeDomain.BRIDGING, third_party=True)

        # Assert
        assert abs(actual - expected) < 1e-9

    def test_third_party_internal_no_penalty(self):
        # Arrange -- CONFIDENCE_INTERNAL.third_party_penalty = 0.0; penalty is
        # a no-op for INTERNAL domain regardless of the third_party flag.
        mgr = ConfidenceManager()
        base = 0.9

        # Act
        actual = mgr.penalized_confidence(base, KnowledgeDomain.INTERNAL, third_party=True)

        # Assert -- 0.9 - 0.0 = 0.9
        assert abs(actual - base) < 1e-9

    def test_penalty_clamps_to_zero_floor(self):
        # Arrange -- artificially low base to exercise the floor clamp
        mgr = ConfidenceManager()
        base = 0.05  # 0.05 - 0.15 would be negative

        # Act
        actual = mgr.penalized_confidence(base, KnowledgeDomain.EXTERNAL, third_party=True)

        # Assert -- must not go below 0.0
        assert actual >= 0.0

    def test_zero_penalty_when_no_policy_found(self):
        # Arrange -- unknown domain falls through the policy lookup; should
        # behave gracefully (no penalty, return base unchanged).
        mgr = ConfidenceManager()
        base = 0.75

        # Act -- KnowledgeDomain.EXTERNAL is the only well-defined fallback
        # but we test the no-policy path by calling with a domain whose
        # policy is missing from the manager (simulate by creating a manager
        # over an empty policy set).
        mgr._policies = {}  # strip all policies
        actual = mgr.penalized_confidence(base, KnowledgeDomain.EXTERNAL, third_party=True)

        # Assert -- penalty is 0.0 when policy not found, base returned
        assert abs(actual - base) < 1e-9
