"""Tests for ConfidenceManager and CurationError."""

import pytest

from backend.errors import CurationError, MistError
from backend.knowledge.curation.confidence import ConfidenceManager
from backend.knowledge.ontologies.base import KnowledgeDomain


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
