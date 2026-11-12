"""The dispatch identity: completeness, sensitivity, and agreement with the gates.

The dispatch identity keys the backend settings that decide what the cache
holds for a turn. Completeness keeps every `DispatcherSettings` field
classified; sensitivity shows each component moves the identity and each
excluded setting does not; agreement shows `dispatch_gate_settings()` reports
the values the gates actually decide with, by mutating each one and watching
`evaluate_dispatch_gates` change its decision.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from backend.extraction_backlog.dispatch_identity import (
    DISPATCH_IDENTITY_PREFIX,
    DISPATCHER_SETTINGS_EXCLUDED,
    GATE_FIELDS,
    DispatchIdentityComponents,
    components_dict,
    compute_dispatch_identity,
    dispatch_identity_for,
)
from backend.extraction_backlog.settings import DispatcherSettings
from backend.knowledge.config import ExtractionConfig
from backend.knowledge.extraction.confidence import ConfidenceScorer
from backend.knowledge.extraction.normalizer import EntityNormalizer
from backend.knowledge.extraction.ontology_extractor import OntologyConstrainedExtractor
from backend.knowledge.extraction.pipeline import ExtractionPipeline
from backend.knowledge.extraction.preprocessor import PreProcessor
from backend.knowledge.extraction.temporal import TemporalResolver
from backend.knowledge.extraction.validator import ExtractionValidator
from backend.knowledge.extraction_cache import SKIP_BELOW_SIGNIFICANCE, SKIP_DUPLICATE
from backend.knowledge.storage.graph_store import GraphStore
from tests.mocks.config import build_test_config
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection
from tests.mocks.ollama import FakeLLM

FIRST = "I really enjoy writing rust code"
SECOND = "I really enjoy writing zig code"


class _Vectors:
    """Embeddings chosen so FIRST and SECOND have cosine similarity 0.9."""

    _BY_TEXT = {FIRST: [1.0, 0.0], SECOND: [0.9, math.sqrt(1 - 0.81)]}

    def generate_embedding(self, text: str) -> list[float]:
        return list(self._BY_TEXT[text])

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [self.generate_embedding(t) for t in texts]


def _pipeline(config: ExtractionConfig, *, embeddings=None) -> ExtractionPipeline:
    return ExtractionPipeline(
        preprocessor=PreProcessor(),
        extractor=OntologyConstrainedExtractor(build_test_config(), llm=FakeLLM()),
        confidence_scorer=ConfidenceScorer(),
        temporal_resolver=TemporalResolver(),
        normalizer=EntityNormalizer(embedding_generator=None, executor=None),
        validator=ExtractionValidator(min_confidence=0.0),
        graph_store=GraphStore(FakeNeo4jConnection(), FakeEmbeddingGenerator()),
        embedding_provider=embeddings,
        extraction_config=config,
    )


# A configuration under which SECOND is a Gate 3 duplicate of FIRST (0.9 >= 0.85).
_BASE = ExtractionConfig(
    significance_threshold=0.0,
    dedup_similarity_threshold=0.85,
    dedup_cache_size=200,
    dedup_cache_ttl_seconds=300,
)


def _second_decision(pipeline: ExtractionPipeline) -> str | None:
    """Gate FIRST, teach Gate 3 about it as the dispatcher does, then gate SECOND."""
    first = pipeline.evaluate_dispatch_gates(FIRST)
    pipeline.note_extraction_completed(FIRST, first.embedding)
    return pipeline.evaluate_dispatch_gates(SECOND).skip_reason


class TestCompleteness:
    def test_every_dispatcher_setting_is_a_component_or_excluded(self):
        settings_fields = {f.name for f in dataclasses.fields(DispatcherSettings)}
        components = {f.name for f in dataclasses.fields(DispatchIdentityComponents)}

        unclassified = settings_fields - components - set(DISPATCHER_SETTINGS_EXCLUDED)

        assert unclassified == set()
        assert settings_fields & components == {"history_messages"}

    def test_the_exclusions_name_real_settings_with_a_reason(self):
        settings_fields = {f.name for f in dataclasses.fields(DispatcherSettings)}

        assert set(DISPATCHER_SETTINGS_EXCLUDED) <= settings_fields
        assert all(reason.strip() for reason in DISPATCHER_SETTINGS_EXCLUDED.values())

    def test_every_other_component_is_a_gate_setting_the_pipeline_reports(self):
        components = {f.name for f in dataclasses.fields(DispatchIdentityComponents)}

        reported = set(_pipeline(_BASE).dispatch_gate_settings())

        assert components - {"history_messages"} == set(GATE_FIELDS) == reported

    def test_gate_settings_with_an_unknown_name_are_refused(self):
        gates = {**_pipeline(_BASE).dispatch_gate_settings(), "rate_limit": 30}

        with pytest.raises(ValueError, match="not the keyed gate fields"):
            dispatch_identity_for(gates, DispatcherSettings())


_COMPONENTS = DispatchIdentityComponents(
    history_messages=10,
    significance_threshold=0.3,
    dedup_similarity_threshold=0.95,
    dedup_cache_size=200,
    dedup_cache_ttl_seconds=300,
    embedding_gates=True,
)
_CHANGED = {
    "history_messages": 4,
    "significance_threshold": 0.4,
    "dedup_similarity_threshold": 0.9,
    "dedup_cache_size": 100,
    "dedup_cache_ttl_seconds": 60,
    "embedding_gates": False,
}
_EXCLUDED_CHANGES = {
    "mode": "off",
    "service_url": "http://elsewhere:8090",
    "max_attempts": 9,
    "backoff_base_s": 7.0,
    "backoff_cap_s": 70.0,
    "request_timeout_s": 12.0,
    "idle_poll_s": 1.0,
    "stall_recheck_s": 2.0,
}


class TestSensitivity:
    def test_the_identity_is_prefixed_and_stable(self):
        identity = compute_dispatch_identity(_COMPONENTS)

        assert identity.startswith(DISPATCH_IDENTITY_PREFIX)
        assert identity == compute_dispatch_identity(dataclasses.replace(_COMPONENTS))

    def test_every_component_has_a_change(self):
        assert set(_CHANGED) == {f.name for f in dataclasses.fields(DispatchIdentityComponents)}

    @pytest.mark.parametrize("name", sorted(_CHANGED))
    def test_changing_a_component_moves_the_identity(self, name):
        changed = dataclasses.replace(_COMPONENTS, **{name: _CHANGED[name]})

        assert compute_dispatch_identity(changed) != compute_dispatch_identity(_COMPONENTS)

    def test_every_excluded_setting_has_a_change(self):
        assert set(_EXCLUDED_CHANGES) == set(DISPATCHER_SETTINGS_EXCLUDED)

    @pytest.mark.parametrize("name", sorted(_EXCLUDED_CHANGES))
    def test_changing_an_excluded_setting_does_not_move_the_identity(self, name):
        gates = _pipeline(_BASE).dispatch_gate_settings()
        base = DispatcherSettings()
        changed = dataclasses.replace(base, **{name: _EXCLUDED_CHANGES[name]})

        assert dispatch_identity_for(gates, changed) == dispatch_identity_for(gates, base)

    def test_an_int_threshold_hashes_as_the_same_float(self):
        as_int = dataclasses.replace(_COMPONENTS, significance_threshold=1)
        as_float = dataclasses.replace(_COMPONENTS, significance_threshold=1.0)

        assert compute_dispatch_identity(as_int) == compute_dispatch_identity(as_float)
        assert components_dict(as_int)["significance_threshold"] == 1.0


# (gate field, mutated value, embedding provider, SECOND's decision once mutated)
_AGREEMENT_CASES = [
    pytest.param("significance_threshold", 0.99, _Vectors, SKIP_BELOW_SIGNIFICANCE, id="gate-2"),
    pytest.param("dedup_similarity_threshold", 0.95, _Vectors, None, id="gate-3-threshold"),
    pytest.param("dedup_cache_size", 0, _Vectors, None, id="gate-3-size"),
    # -1: every entry is older than the TTL at once, with no clock in the test.
    pytest.param("dedup_cache_ttl_seconds", -1, _Vectors, None, id="gate-3-ttl"),
    pytest.param("embedding_gates", False, None, None, id="no-embedding-provider"),
]


class TestAgreementWithTheGates:
    """Each reported value is one the gates read: changing it changes a gate decision."""

    @pytest.mark.parametrize("name,value,embeddings,expected", _AGREEMENT_CASES)
    def test_mutating_a_gate_value_changes_the_decision_and_the_report(
        self, name, value, embeddings, expected
    ):
        # Arrange
        base = _pipeline(_BASE, embeddings=_Vectors())
        if name == "embedding_gates":
            mutated = _pipeline(_BASE, embeddings=None)
        else:
            mutated = _pipeline(
                dataclasses.replace(_BASE, **{name: value}),
                embeddings=embeddings() if embeddings else None,
            )

        # Act
        base_decision = _second_decision(base)
        mutated_decision = _second_decision(mutated)
        base_report = base.dispatch_gate_settings()
        mutated_report = mutated.dispatch_gate_settings()

        # Assert
        assert base_decision == SKIP_DUPLICATE
        assert mutated_decision == expected
        assert mutated_report[name] == value
        assert {k for k in base_report if base_report[k] != mutated_report[k]} == {name}

    def test_every_gate_field_has_an_agreement_case(self):
        assert {case.values[0] for case in _AGREEMENT_CASES} == set(GATE_FIELDS)

    def test_the_threshold_is_the_one_resolved_for_conversation(self):
        pipeline = _pipeline(dataclasses.replace(_BASE, significance_threshold=0.42))

        assert pipeline.dispatch_gate_settings()["significance_threshold"] == 0.42


class TestTheDispatcherComputesIt:
    def test_from_its_pipeline_and_settings_with_no_constructor_parameter(self, backlog_world):
        dispatcher = backlog_world.build_dispatcher(history_messages=6)

        expected = dispatch_identity_for(
            backlog_world.build_pipeline().dispatch_gate_settings(),
            dataclasses.replace(DispatcherSettings(), history_messages=6),
        )

        assert dispatcher.dispatch_identity == expected
        assert dispatcher.dispatch_identity.components["history_messages"] == 6
