"""Tests for the curation result -> run-ledger mapping (D3).

The registry in `run_record._RESULT_COUNTERS` is the declaration of which
field of each job result carries "what it examined" and which carries "what it
produced". A missing entry does not raise -- it writes NULLs -- so the drift it
would cause is exactly the silence the ledger exists to end.
`TestRegistryCompleteness` is what makes that loud instead.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass

import pytest

from backend.knowledge.curation.confidence_decay import DecayResult
from backend.knowledge.curation.health import HealthScore
from backend.knowledge.curation.run_record import (
    _RESULT_COUNTERS,
    describe_result,
    health_event_fields,
)
from backend.knowledge.curation.self_reflection import ReflectionResult
from backend.knowledge.curation.staleness import StalenessResult


def _health_score(**overrides) -> HealthScore:
    """A HealthScore with every field set, overridable per test."""
    values = {
        "overall": 72.5,
        "freshness": 80.0,
        "confidence": 70.0,
        "connectivity": 60.0,
        "consistency": 100.0,
        "coverage": 50.0,
        "self_model": 100.0,
        "entity_count": 214,
        "relationship_count": 530,
    }
    values.update(overrides)
    return HealthScore(**values)


class TestDescribeResult:
    def test_splits_examined_from_produced(self):
        facts = describe_result(
            DecayResult(
                entities_scanned=120, entities_decayed=4, entities_archived=2, duration_ms=9.0
            )
        )

        assert facts.result_type == "DecayResult"
        assert facts.examined == 120
        assert facts.produced == 6

    def test_sums_the_tiers_that_make_up_the_examined_universe(self):
        """`StalenessResult.active_count` is only the healthiest tier, so
        recording it alone would under-report what the scan looked at.
        """
        facts = describe_result(
            StalenessResult(
                active_count=10,
                stale_count=3,
                very_stale_count=2,
                confirmation_list=(),
                duration_ms=5.0,
            )
        )

        assert facts.examined == 15
        assert facts.produced == 5

    def test_zero_examined_is_recorded_as_zero_not_null(self):
        """The distinction the whole table rests on: 0 asserts the job looked
        and found nothing, NULL asserts it reports no such counter.
        """
        facts = describe_result(
            DecayResult(
                entities_scanned=0, entities_decayed=0, entities_archived=0, duration_ms=0.3
            )
        )

        assert facts.examined == 0
        assert facts.produced == 0

    def test_read_only_result_produces_null_not_zero(self):
        facts = describe_result(_health_score())

        assert facts.examined == 214
        assert facts.produced is None

    def test_metrics_preserve_every_field_including_self_reported_duration(self):
        """The summary columns are lossy on purpose; the JSON is not. The
        reflection job's self-reported `duration_ms` of exactly 0.0 is the only
        signal separating its inert path from a genuinely empty log, so it
        must survive.
        """
        facts = describe_result(
            ReflectionResult(events_processed=0, operations_applied=0, duration_ms=0.0)
        )

        assert json.loads(facts.metrics) == {
            "events_processed": 0,
            "operations_applied": 0,
            "duration_ms": 0.0,
        }

    def test_health_subscores_survive_in_metrics(self):
        """Seven sub-scores previously reached `logger.info` and nothing else."""
        facts = describe_result(_health_score())

        metrics = json.loads(facts.metrics)
        assert metrics["coverage"] == 50.0
        assert metrics["self_model"] == 100.0
        assert metrics["overall"] == 72.5

    def test_none_result_yields_all_nulls(self):
        facts = describe_result(None)

        assert (facts.result_type, facts.examined, facts.produced, facts.metrics) == (
            None,
            None,
            None,
            None,
        )

    def test_unregistered_dataclass_is_still_recorded_with_null_counters(self):
        """A result type nobody mapped must still leave a row. Refusing to
        record an unfamiliar result would reintroduce the silence.
        """

        @dataclass(frozen=True, slots=True)
        class UnmappedResult:
            widgets: int

        facts = describe_result(UnmappedResult(widgets=3))

        assert facts.result_type == "UnmappedResult"
        assert facts.examined is None
        assert facts.produced is None
        assert json.loads(facts.metrics) == {"widgets": 3}

    def test_non_dataclass_result_is_recorded_by_type_alone(self):
        facts = describe_result({"status": "ok"})

        assert facts.result_type == "dict"
        assert facts.metrics is None

    def test_non_serializable_field_degrades_instead_of_losing_the_row(self):
        """`json.dumps` raising would cost the entire run record, which is a
        worse outcome than a stringified field.
        """

        @dataclass(frozen=True, slots=True)
        class OddResult:
            payload: object

        facts = describe_result(OddResult(payload=object()))

        assert "object" in json.loads(facts.metrics)["payload"]


class TestHealthEventFields:
    def test_projects_a_health_score_onto_the_series_columns(self):
        fields_out = health_event_fields(_health_score())

        assert fields_out["health_score"] == 72.5
        assert fields_out["entity_count"] == 214
        assert fields_out["relationship_count"] == 530
        assert json.loads(fields_out["metrics"]) == {
            "freshness": 80.0,
            "confidence": 70.0,
            "connectivity": 60.0,
            "consistency": 100.0,
            "coverage": 50.0,
            "self_model": 100.0,
        }

    def test_overall_is_not_duplicated_into_the_metrics_blob(self):
        """It has its own NOT NULL column; two copies can disagree."""
        assert "overall" not in json.loads(health_event_fields(_health_score())["metrics"])

    @pytest.mark.parametrize(
        "result",
        [
            pytest.param(None, id="none"),
            pytest.param({"overall": 50.0}, id="dict-shaped-like-a-score"),
            pytest.param(
                DecayResult(
                    entities_scanned=1, entities_decayed=0, entities_archived=0, duration_ms=1.0
                ),
                id="another-job-result",
            ),
        ],
    )
    def test_returns_none_for_anything_that_is_not_a_health_score(self, result):
        assert health_event_fields(result) is None


class TestRegistryCompleteness:
    """Every job the composition root registers must have a mapping.

    Without this, adding a tenth curation job produces rows with NULL counters
    that look like "this job reports no counters" rather than "nobody mapped
    it" -- a wrong diagnosis is worse than an absent one.
    """

    def test_every_registered_job_result_type_is_mapped(self):
        from backend.knowledge.curation import (
            centrality,
            community,
            confidence_decay,
            embedding_maintenance,
            health,
            orphan_detector,
            self_reflection,
            skill_derivation,
            staleness,
        )

        job_classes = [
            confidence_decay.ConfidenceDecayJob,
            staleness.StalenessDetector,
            orphan_detector.OrphanDetector,
            health.GraphHealthScorer,
            self_reflection.SelfReflectionJob,
            community.CommunityDetector,
            centrality.CentralityAnalyzer,
            embedding_maintenance.EmbeddingMaintenance,
            skill_derivation.SkillDerivationJob,
        ]

        unmapped = []
        for job_class in job_classes:
            annotation = inspect.signature(job_class.run).return_annotation
            name = annotation if isinstance(annotation, str) else annotation.__name__
            if name not in _RESULT_COUNTERS:
                unmapped.append(f"{job_class.__name__} -> {name}")

        assert not unmapped, (
            "Curation job result types with no entry in run_record._RESULT_COUNTERS. "
            "Their runs would be recorded with NULL examined/produced, which reads as "
            f"'this job reports no counters' rather than 'nobody mapped it': {unmapped}"
        )

    def test_the_registry_names_only_fields_that_exist(self):
        """A typo'd field name silently yields NULL rather than raising."""
        from dataclasses import fields as dataclass_fields

        from backend.knowledge.curation.centrality import CentralityResult
        from backend.knowledge.curation.community import CommunityResult
        from backend.knowledge.curation.embedding_maintenance import EmbeddingMaintenanceResult
        from backend.knowledge.curation.orphan_detector import OrphanResult
        from backend.knowledge.curation.skill_derivation import SkillDerivationResult

        result_classes = {
            cls.__name__: cls
            for cls in (
                DecayResult,
                StalenessResult,
                OrphanResult,
                HealthScore,
                ReflectionResult,
                CommunityResult,
                CentralityResult,
                EmbeddingMaintenanceResult,
                SkillDerivationResult,
            )
        }

        assert set(_RESULT_COUNTERS) == set(result_classes), "registry and job results disagree"

        bad = []
        for name, (examined, produced) in _RESULT_COUNTERS.items():
            declared = {f.name for f in dataclass_fields(result_classes[name])}
            for field_name in (*examined, *produced):
                if field_name not in declared:
                    bad.append(f"{name}.{field_name}")

        assert not bad, f"registry names fields that do not exist: {bad}"
