"""Stages 3-6 must be pure: same input twice -> identical output, no clock.

Spec D2 moves the cache boundary to just after Stage 2 and lets a rebuild
re-run Stages 3-6. That is only sound if those stages are deterministic
functions of their arguments. This module is the proof, and it is deliberately
written BEFORE the code that assumes it.
"""

import copy
from datetime import datetime

import pytest

from backend.knowledge.extraction.confidence import ConfidenceScorer
from backend.knowledge.extraction.ontology_extractor import ExtractionResult
from backend.knowledge.extraction.temporal import TemporalResolver
from backend.knowledge.extraction.validator import ExtractionValidator

REFERENCE_DATE = datetime.fromisoformat("2026-08-18T12:00:00+00:00")


def _sample_extraction() -> ExtractionResult:
    return ExtractionResult(
        entities=[
            {"id": "user", "name": "User", "type": "User"},
            {"id": "python", "name": "Python", "type": "Technology"},
        ],
        relationships=[
            {
                "source": "user",
                "target": "python",
                "type": "USES",
                "properties": {
                    "confidence": 0.9,
                    "temporal_status": "current",
                    "start_date": None,
                    "end_date": None,
                    "temporal_expression": "for about 5 years",
                    "context": None,
                    "negated": False,
                    "assertion_kind": "assert",
                },
            }
        ],
        source_utterance="I have used Python for about 5 years",
    )


def test_stage_3_confidence_is_deterministic():
    scorer = ConfidenceScorer()
    first = scorer.adjust_confidence(copy.deepcopy(_sample_extraction()))
    second = scorer.adjust_confidence(copy.deepcopy(_sample_extraction()))
    assert first.entities == second.entities
    assert first.relationships == second.relationships


def test_stage_4_temporal_is_deterministic_for_a_fixed_reference_date():
    resolver = TemporalResolver()
    first = resolver.resolve(copy.deepcopy(_sample_extraction()), REFERENCE_DATE)
    second = resolver.resolve(copy.deepcopy(_sample_extraction()), REFERENCE_DATE)
    assert first.relationships == second.relationships


def test_stage_4_never_reads_the_wall_clock(monkeypatch):
    """C1 anchors reference_date to recorded_at. A clock read would defeat it."""
    import backend.knowledge.extraction.temporal as temporal_module

    class ExplodingDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            raise AssertionError("Stage 4 read the wall clock")

        @classmethod
        def today(cls):
            raise AssertionError("Stage 4 read the wall clock")

    monkeypatch.setattr(temporal_module, "datetime", ExplodingDatetime)
    TemporalResolver().resolve(_sample_extraction(), REFERENCE_DATE)


def test_stage_6_validation_is_deterministic_and_order_stable():
    validator = ExtractionValidator()
    first = validator.validate(copy.deepcopy(_sample_extraction()))
    second = validator.validate(copy.deepcopy(_sample_extraction()))
    assert first.entities == second.entities
    assert first.relationships == second.relationships
    assert first.warnings == second.warnings
    assert first.errors == second.errors


@pytest.mark.asyncio
async def test_stage_5_normalize_is_pure_and_issues_no_graph_queries():
    r"""Stage 5 is a pure function of its argument.

    Both constructor dependencies are vestigial: `embedding_generator` and
    `executor` are stored in `EntityNormalizer.__init__` and never read
    anywhere else in the module (`grep -n "self\._embedding_generator\|self\._executor"
    backend/knowledge/extraction/normalizer.py` -- both hits are the assignments
    in `__init__`, no other match in the file), and `normalize()`'s own
    docstring states it "issues no graph queries" -- graph-identity resolution
    moved to the curation deduper (Stage 7a). Passing None for both is
    therefore valid, and a rebuild can run Stage 5 with no graph and no model.

    If passing None raises, that assumption is dead and spec D2 needs revisiting
    -- stop and report rather than supplying a real collaborator to make it pass.
    """
    from backend.knowledge.extraction.normalizer import EntityNormalizer

    normalizer = EntityNormalizer(embedding_generator=None, executor=None)
    first = await normalizer.normalize(copy.deepcopy(_sample_extraction()))
    second = await normalizer.normalize(copy.deepcopy(_sample_extraction()))
    assert first.entities == second.entities
    assert first.relationships == second.relationships
